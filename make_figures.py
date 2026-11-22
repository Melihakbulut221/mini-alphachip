"""Turns results/results.json (+ checkpoints) into the figures and the summary table.

    python make_figures.py            # after run_experiments.py
    python make_figures.py --quick    # after run_experiments.py --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minichip import viz
from minichip.env import evaluate
from minichip.model import PlacementPolicy
from minichip.netlist import Batch, random_netlist

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
sub = "quick" if args.quick else ""
res = json.loads((Path("results") / sub / "results.json").read_text())
ckpt, figs = Path("checkpoints") / sub, Path("figures") / sub
figs.mkdir(parents=True, exist_ok=True)

runs, x = res["runs"], np.array(res["evals"])
target = random_netlist(res["target"])
seeds = sorted(int(k.split("_s")[1]) for k in runs if k.startswith("scratch_s"))
corpus = sorted(int(k.split("_")[1]) for k in runs if k.startswith("pretrain_"))


def curve(prefix):  # best-so-far cost of every seed on the common x grid
    return np.array([runs[f"{prefix}_s{s}"]["first_batch"] + runs[f"{prefix}_s{s}"]["best_cost"][1:]
                     for s in seeds])


def score(anchors):
    out = evaluate(Batch.repeat(target, 1), torch.as_tensor(anchors)[None])
    return {k: v.item() for k, v in out.items()}


curves = {"sa": np.array(res["sa"]["curves"]), "scratch": curve("scratch"),
          "finetune": curve("finetune"), "nowire": curve("nowire"),
          "random": np.array([res["random"]["curve"]])}
greedy = res["greedy"]["cost"]

# ---------------------------------------------------------------- figure 1: search efficiency
viz.convergence_figure(
    {"finetune": ("RL, pre-trained on 64 other chips then fine-tuned", x, curves["finetune"]),
     "scratch": ("RL from scratch", x, curves["scratch"]),
     "sa": ("simulated annealing", x, curves["sa"])},
    greedy, figs / "convergence.png",
    f"Placing an unseen chip: best cost vs. search budget\n"
    f"lines = mean, bands = min-max over {len(seeds)} RL seeds / {len(curves['sa'])} annealing chains")

# ---------------------------------------------------------------- figure 2: the placements
best_ft = min((runs[f"finetune_s{s}"] for s in seeds), key=lambda r: r["best_cost"][-1])
zs = runs[f"pretrain_{corpus[-1]}"]["zero_shot"][str(res["target"])]
panels = {
    "Random": res["random"]["example_anchors"],
    "Greedy heuristic": res["greedy"]["anchors"],
    "Simulated annealing (best chain)": res["sa"]["best_anchors"],
    "RL zero-shot (never saw this chip)": zs["argmax_anchors"],
    "RL pre-trained + fine-tuned (best)": best_ft["best_anchors"],
}
viz.placements_figure(target, {k: (a, score(a)) for k, a in panels.items()}, figs / "placements.png")

# ---------------------------------------------------------------- figure 3: zero-shot transfer
held = [str(s) for s in res["held_out"]]
base = res["held_out_baselines"]


def held_mean(n, key):
    return float(np.mean([runs[f"pretrain_{n}"]["zero_shot"][s][key] for s in held]))


rows = [("Random placement", float(np.mean([base[s]["random_mean"] for s in held])), False),
        ("Greedy heuristic", float(np.mean([base[s]["greedy"] for s in held])), False)]
rows += [(f"RL zero-shot, pre-trained on {n} chips", held_mean(n, "argmax"), True) for n in corpus]
rows += [(f"RL zero-shot, {corpus[-1]} chips, best of 64 samples", held_mean(corpus[-1], "best_of_samples"), True)]
viz.bars_figure(rows, figs / "zero_shot.png",
                f"Zero-shot transfer: mean cost on {len(held)} chips never seen in training",
                "placement cost (lower is better)",
                "Zero-shot = one forward pass per macro, no training on the chip. Grey = no learning involved.")

# ---------------------------------------------------------------- animation
policy = PlacementPolicy()
policy.load_state_dict(torch.load(ckpt / f"finetuned_s{seeds[0]}.pt"))
viz.agent_gif(policy, target, figs / "agent.gif",
              "mini-AlphaChip: blue = where the policy wants the next macro")

# ---------------------------------------------------------------- summary table (markdown)
budgets = [b for b in (64, 1024, 4096, res["budget"]) if b <= res["budget"]]
col = [int(np.searchsorted(x, b)) for b in budgets]


def row(name, c):
    cells = [f"{c[:, j].mean():.3f} ± {c[:, j].std():.3f}" if len(c) > 1 else f"{c[0, j]:.3f}" for j in col]
    return f"| {name} | " + " | ".join(cells) + " |"


wins = sum(runs[f"pretrain_{corpus[-1]}"]["zero_shot"][s]["argmax"] < base[s]["greedy"] for s in held)
wins64 = sum(runs[f"pretrain_{corpus[-1]}"]["zero_shot"][s]["best_of_samples"] < base[s]["greedy"] for s in held)
lines = [
    f"Best cost found on chip-{res['target']} after N placement evaluations (mean ± std over runs):", "",
    "| Method | " + " | ".join(f"N = {b:,}" for b in budgets) + " |",
    "|---|" + "---:|" * len(budgets),
    row("Random search", curves["random"]),
    row(f"Simulated annealing ({len(curves['sa'])} chains)", curves["sa"]),
    row(f"RL from scratch, no wire mask ({len(seeds)} seeds)", curves["nowire"]),
    row(f"RL from scratch ({len(seeds)} seeds)", curves["scratch"]),
    row(f"RL pre-trained + fine-tuned ({len(seeds)} seeds)", curves["finetune"]),
    "",
    f"Greedy heuristic (1 placement): {greedy:.3f}.  "
    f"RL zero-shot argmax (1 placement): {zs['argmax']:.3f}.  "
    f"Best single run: SA {curves['sa'][:, -1].min():.3f}, RL scratch {curves['scratch'][:, -1].min():.3f}, "
    f"RL fine-tuned {curves['finetune'][:, -1].min():.3f}.", "",
    f"Zero-shot on {len(held)} held-out chips (mean cost):", "",
    "| Method | Mean cost |", "|---|---:|",
] + [f"| {name} | {value:.3f} |" for name, value, _ in rows] + [
    "",
    f"Pre-trained on {corpus[-1]} chips, zero-shot beats the greedy heuristic on {wins}/{len(held)} chips "
    f"with a single argmax pass and on {wins64}/{len(held)} with the best of 64 samples.",
    f"Total wall-clock for all runs: {res['minutes']:.1f} min.",
]
(Path("results") / sub / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
