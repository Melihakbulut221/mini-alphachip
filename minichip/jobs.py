"""One training run = one job. Jobs are plain dicts so they can be farmed out to worker
processes (one CPU thread each) and their results dumped straight to JSON."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import torch

from .model import PlacementPolicy
from .netlist import Batch, random_netlist
from .rl import train, zero_shot

BATCH = 64            # placements per PPO iteration
CHIPS_PER_BATCH = 8   # when training on several chips: 8 chips x 8 rollouts each


def train_job(spec: dict) -> dict:
    """spec keys: name, seed, chips (list of netlist seeds), iterations, and optionally
    init / save (checkpoint paths), entropy_coef, lr, use_wire_mask, eval_chips."""
    torch.set_num_threads(1)
    torch.manual_seed(spec["seed"])
    use_wm = spec.get("use_wire_mask", True)
    policy = PlacementPolicy()
    if spec.get("init"):
        policy.load_state_dict(torch.load(spec["init"]))

    chips = [random_netlist(s) for s in spec["chips"]]
    if len(chips) == 1:
        def sample():
            return Batch.repeat(chips[0], BATCH)
    else:
        def sample():
            picks = torch.randint(len(chips), (CHIPS_PER_BATCH,)).tolist()
            return Batch([chips[i] for i in picks for _ in range(BATCH // CHIPS_PER_BATCH)])

    t0 = time.time()
    out = train(policy, sample, spec["iterations"], lr=spec.get("lr", 3e-4),
                entropy_coef=spec.get("entropy_coef", 0.01), use_wire_mask=use_wm,
                label=spec["name"])
    out.update(name=spec["name"], spec=spec, minutes=(time.time() - t0) / 60)
    if spec.get("save"):
        torch.save(policy.state_dict(), spec["save"])
    if spec.get("eval_chips"):
        out["zero_shot"] = {str(s): zero_shot(policy, random_netlist(s), use_wire_mask=use_wm)
                            for s in spec["eval_chips"]}
    return out


def run_parallel(specs: list, workers: int) -> dict:
    """Run jobs in separate processes; returns {name: result}."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return {r["name"]: r for r in pool.map(train_job, specs)}
