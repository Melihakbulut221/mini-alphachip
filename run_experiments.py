"""The paper's headline experiments in miniature, on laptop CPU cores.

  A. One unseen chip, equal search budget for everyone:
     random search, greedy, simulated annealing (8 chains), RL from scratch (3 seeds),
     RL pre-trained on other chips then fine-tuned (3 seeds).
  B. Zero-shot transfer: the pre-trained policy on 20 chips it has never seen, with no
     training on them, as a function of how many chips it was pre-trained on (4 / 16 / 64).
  C. Ablation: RL from scratch without the wire-mask input feature.

    python run_experiments.py            # ~15 min on a 10-core laptop
    python run_experiments.py --quick    # smoke test, ~1 min

Hyper-parameters (learning rate, entropy bonus, annealing temperature) were chosen on a separate
development chip (chip-900), never on the chips reported here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from minichip.baselines import greedy_placement, random_placements, simulated_annealing
from minichip.jobs import BATCH, run_parallel
from minichip.netlist import random_netlist

TARGET = 1000                       # the unseen chip of experiment A
HELD_OUT = list(range(1000, 1020))  # experiment B; netlist seeds 0..63 are the training corpus
CORPUS_SIZES = (4, 16, 64)
SEEDS = (0, 1, 2)
SA_CHAINS = 8
SA_T_MAX, SA_T_MIN = 0.05, 3e-4
FINETUNE = dict(entropy_coef=0.01, lr=3e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    iters, pre_iters = (6, 6) if args.quick else (200, 300)
    held_out = HELD_OUT[:3] if args.quick else HELD_OUT
    budget = iters * BATCH
    out = Path("results/quick" if args.quick else "results")
    ckpt = Path("checkpoints/quick" if args.quick else "checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    grid = list(range(1, BATCH + 1)) + list(range(2 * BATCH, budget + 1, BATCH))  # x axis of all curves
    res = {"target": TARGET, "budget": budget, "evals": grid, "held_out": held_out}

    # ------------------------------------------------------------------ baselines
    print(f"== baselines (budget: {budget} placements per method)", flush=True)
    target = random_netlist(TARGET)
    gen = torch.Generator().manual_seed(0)
    r_anchors, r_cost = random_placements(target, budget, gen)
    r_curve = np.minimum.accumulate(r_cost["cost"].numpy())
    g_anchors, g_cost = greedy_placement(target)
    sa_anchors, sa_cost, sa_curve = simulated_annealing(target, budget - 1, SA_CHAINS, gen, t_max=SA_T_MAX, t_min=SA_T_MIN)
    idx = np.array(grid) - 1
    res["random"] = {"mean": r_cost["cost"].mean().item(), "curve": r_curve[idx].tolist(),
                     "example_anchors": r_anchors[0].tolist()}
    res["greedy"] = {"cost": g_cost["cost"], "anchors": g_anchors.tolist()}
    res["sa"] = {"final": sa_cost.tolist(), "curves": sa_curve[:, idx].tolist(),
                 "best_anchors": sa_anchors[sa_cost.argmin()].tolist()}
    res["held_out_baselines"] = {}
    for s in held_out:
        nl = random_netlist(s)
        res["held_out_baselines"][str(s)] = {
            "random_mean": random_placements(nl, 256, gen)[1]["cost"].mean().item(),
            "greedy": greedy_placement(nl)[1]["cost"]}
    print(f"   random mean {res['random']['mean']:.3f}, best of budget {r_curve[-1]:.3f} | "
          f"greedy {g_cost['cost']:.3f} | SA mean {sa_cost.mean():.3f}, best chain {sa_cost.min():.3f}", flush=True)

    # ------------------------------------------------------------------ stage 1: independent runs
    stage1 = [dict(name=f"pretrain_{n}", seed=0, chips=list(range(n)), iterations=pre_iters,
                   save=str(ckpt / f"pretrained_{n}.pt"), eval_chips=held_out) for n in CORPUS_SIZES[::-1]]
    stage1 += [dict(name=f"scratch_s{s}", seed=s, chips=[TARGET], iterations=iters) for s in SEEDS]
    stage1 += [dict(name=f"nowire_s{s}", seed=s, chips=[TARGET], iterations=iters, use_wire_mask=False)
               for s in SEEDS]
    print(f"== stage 1: {len(stage1)} training runs on {args.workers} worker processes", flush=True)
    res["runs"] = run_parallel(stage1, args.workers)

    # ------------------------------------------------------------------ stage 2: fine-tuning
    best_pre = str(ckpt / f"pretrained_{CORPUS_SIZES[-1]}.pt")
    stage2 = [dict(name=f"finetune_s{s}", seed=s, chips=[TARGET], iterations=iters, init=best_pre,
                   save=str(ckpt / f"finetuned_s{s}.pt"), **FINETUNE) for s in SEEDS]
    print(f"== stage 2: fine-tuning {best_pre} on chip-{TARGET}", flush=True)
    res["runs"].update(run_parallel(stage2, args.workers))

    res["minutes"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(res))
    print(f"== done in {res['minutes']:.1f} min -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
