"""Reproduces the paper's three headline experiments in miniature, on a laptop CPU:

  1. RL from scratch on one unseen "chip" vs random / greedy / simulated annealing
  2. Pre-train on a corpus of netlists, then zero-shot + fine-tune on the unseen chip
     (the paper's Fig. 3/4: pre-training -> better start, faster convergence)
  3. Zero-shot generalisation of the pre-trained policy across held-out chips

Usage:  python run_experiments.py            (~30 min on a 4-core laptop)
        python run_experiments.py --quick    (smoke test, ~2 min)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from minichip.baselines import greedy_placement, random_placement, simulated_annealing
from minichip.env import placement_cost
from minichip.model import PlacementPolicy
from minichip.netlist import random_netlist
from minichip.ppo import rollout, train
from minichip.viz import curves_figure, placement_gif, placements_figure

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ITERS = 12 if args.quick else 250           # RL iterations on the target chip
PRETRAIN_ITERS = 12 if args.quick else 300  # iterations over the training corpus
BATCH = 64                                  # placements per iteration
N_TRAIN = 64                                # netlists in the pre-training corpus
HELD_OUT = list(range(1000, 1004 if args.quick else 1008))
TARGET = 1000                               # "chip-1000" is never seen in pre-training
BUDGET = ITERS * BATCH                      # same #placement evaluations for every search method

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
out, figs, ckpt = Path("results"), Path("figures"), Path("checkpoints")
for p in (out, figs, ckpt):
    p.mkdir(exist_ok=True)

target = random_netlist(TARGET)
corpus = [random_netlist(s) for s in range(N_TRAIN)]
results = {"budget_placements": BUDGET, "target": target.name}
t_all = time.time()


def cost(nl, a):
    return placement_cost(nl, a)["cost"]


# ---------------------------------------------------------------- 1. baselines on the target
print(f"== baselines on {target.name} (budget {BUDGET} placements)")
rand = [random_placement(target, rng) for _ in range(BUDGET)]
rand_costs = np.array([cost(target, a) for a in rand])
greedy = greedy_placement(target)
sa_runs = [simulated_annealing(target, BUDGET, np.random.default_rng(100 + s)) for s in range(3)]
sa_best = min(sa_runs, key=lambda r: r[1][-1])
results["baselines"] = {
    "random_mean": float(rand_costs.mean()),
    "random_best_of_budget": float(rand_costs.min()),
    "greedy": float(cost(target, greedy)),
    "sa_best_of_3_seeds": float(sa_best[1][-1]),
    "sa_mean_of_3_seeds": float(np.mean([r[1][-1] for r in sa_runs])),
}
print(json.dumps(results["baselines"], indent=2))

# ---------------------------------------------------------------- 2a. RL from scratch
print("== RL from scratch on the target chip")
scratch = PlacementPolicy()
h_scratch, best_scratch, a_scratch = train(scratch, lambda b: [target] * b, ITERS, BATCH, label="scratch")

# ---------------------------------------------------------------- 2b. pre-train -> fine-tune
pre_path = ckpt / "pretrained.pt"
print(f"== pre-training on {N_TRAIN} netlists")
pre = PlacementPolicy()
h_pre, _, _ = train(pre, lambda b: [corpus[i] for i in rng.integers(N_TRAIN, size=b)],
                    PRETRAIN_ITERS, BATCH, label="pretrain")
torch.save(pre.state_dict(), pre_path)

pre.eval()
_, zs_greedy, a_zs, _ = rollout(pre, [target], greedy=True)
_, zs_samples, _, _ = rollout(pre, [target] * BATCH)
results["zero_shot"] = {"greedy_decode": float(zs_greedy[0]), "best_of_64_samples": float(zs_samples.min())}
print("zero-shot:", results["zero_shot"])

print("== fine-tuning the pre-trained policy on the target chip")
ft = copy.deepcopy(pre)
ft.train()
h_ft, best_ft, a_ft = train(ft, lambda b: [target] * b, ITERS, BATCH, label="finetune")
results["rl"] = {
    "scratch_best": best_scratch,
    "finetune_best": best_ft,
    "scratch_curve_best": h_scratch["best_cost"],
    "finetune_curve_best": h_ft["best_cost"],
    "scratch_curve_mean": h_scratch["mean_cost"],
    "finetune_curve_mean": h_ft["mean_cost"],
}

# ---------------------------------------------------------------- 3. zero-shot generalisation
print("== zero-shot on held-out chips")
gen = []
for s in HELD_OUT:
    nl = random_netlist(s)
    _, zg, _, _ = rollout(pre, [nl], greedy=True)
    _, zb, _, _ = rollout(pre, [nl] * BATCH)
    r = np.array([cost(nl, random_placement(nl, rng)) for _ in range(BATCH)])
    gen.append({"chip": nl.name, "random_mean": float(r.mean()), "random_best_of_64": float(r.min()),
                "greedy": float(cost(nl, greedy_placement(nl))),
                "rl_zero_shot_greedy": float(zg[0]), "rl_zero_shot_best_of_64": float(zb.min())})
    print(gen[-1])
results["zero_shot_generalisation"] = gen

# ---------------------------------------------------------------- figures
ev = np.array(h_scratch["evals"])
curves_figure(
    {"RL from scratch": (ev, h_scratch["best_cost"]),
     "RL pre-trained + fine-tuned": (ev, h_ft["best_cost"]),
     "simulated annealing (best of 3)": (np.arange(1, BUDGET + 1), sa_best[1])},
    {"greedy": results["baselines"]["greedy"],
     "random (best of budget)": results["baselines"]["random_best_of_budget"]},
    figs / "convergence.png",
    f"{target.name} (unseen during pre-training): best cost vs. search budget")

curves_figure(
    {"RL from scratch": (ev, h_scratch["mean_cost"]),
     "RL pre-trained + fine-tuned": (ev, h_ft["mean_cost"])},
    {"random policy (mean)": results["baselines"]["random_mean"]},
    figs / "policy_mean_cost.png",
    "average cost of the policy's sampled placements (what the agent has learned)")

placements_figure(target, {
    "Random (typical)": rand[0],
    "Greedy (wirelength)": greedy,
    "Simulated annealing": sa_best[0],
    "RL zero-shot (no training on this chip)": a_zs[0],
    "RL pre-trained + fine-tuned": a_ft,
}, figs / "placements.png")

placement_gif(target, a_ft, target.placement_order(), figs / "agent.gif")

results["wall_clock_minutes"] = (time.time() - t_all) / 60
(out / "results.json").write_text(json.dumps(results, indent=1))
print(f"done in {results['wall_clock_minutes']:.1f} min")
