"""Rollouts and PPO (Schulman et al., 2017) for a sparse, end-of-episode placement reward."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.distributions import Categorical

from .env import PlacementEnv
from .model import PlacementPolicy
from .netlist import Batch

OBS_KEYS = ("nodes", "adj", "cur", "meta", "grid", "mask")


@torch.no_grad()
def rollout(policy: PlacementPolicy, batch: Batch, greedy: bool = False, use_wire_mask: bool = True):
    """Place every netlist of the batch once. Returns the transitions and the final costs."""
    env = PlacementEnv(batch, use_wire_mask)
    obs = env.reset()
    steps = []
    while not env.done:
        logits, value = policy(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        steps.append((obs, action, dist.log_prob(action), value))
        obs = env.step(action)
    return steps, env.cost(), env.anchors


def centred_returns(cost: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Reward = -cost, minus the mean reward of the other rollouts of the same chip in this
    batch. Chips differ in how cheaply they can be placed at all; this shared baseline removes
    that between-chip variance, which matters when pre-training on many chips at once."""
    n = int(group.max()) + 1
    mean = torch.zeros(n).scatter_add_(0, group, cost) / torch.bincount(group, minlength=n)
    return mean[group] - cost


def ppo_update(policy, optimizer, steps, returns, *, epochs=3, minibatch=256, clip=0.2,
               value_coef=0.5, entropy_coef=0.01):
    """The only reward arrives at the end of the episode, so with discount 1 the return of
    every step of an episode is that same number."""
    data = {k: torch.cat([s[0][k] for s in steps]) for k in OBS_KEYS}
    action = torch.cat([s[1] for s in steps])
    old_logp = torch.cat([s[2] for s in steps])
    old_value = torch.cat([s[3] for s in steps])
    ret = returns.repeat(len(steps))
    adv = ret - old_value
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(action)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            logits, value = policy({k: v[idx] for k, v in data.items()})
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(action[idx]) - old_logp[idx])
            surrogate = torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            loss = (-surrogate.mean()
                    + value_coef * (value - ret[idx]).pow(2).mean()
                    - entropy_coef * dist.entropy().mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def train(policy: PlacementPolicy, sample_batch, iterations: int, *, lr: float = 3e-4,
          entropy_coef: float = 0.01, use_wire_mask: bool = True, label: str = "",
          log_every: int = 25, **ppo_kwargs) -> dict:
    """sample_batch() -> Batch of netlists to place this iteration.

    Returns the learning curves: mean cost of the sampled placements per iteration, the best
    placement found so far, and the number of placements evaluated."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = {"evals": [], "mean_cost": [], "best_cost": []}
    best, best_anchors, evals, t0 = float("inf"), None, 0, time.time()
    for it in range(1, iterations + 1):
        batch = sample_batch()
        steps, cost, anchors = rollout(policy, batch, use_wire_mask=use_wire_mask)
        c = cost["cost"]
        ppo_update(policy, optimizer, steps, centred_returns(c, batch.group),
                   entropy_coef=entropy_coef, **ppo_kwargs)
        if c.min().item() < best:
            best, best_anchors = c.min().item(), anchors[c.argmin()].clone()
        if it == 1:  # sample-by-sample view of the untrained (or pre-trained, zero-shot) policy
            hist["first_batch"] = np.minimum.accumulate(c.numpy()).tolist()
        evals += batch.B
        hist["evals"].append(evals)
        hist["mean_cost"].append(c.mean().item())
        hist["best_cost"].append(best)
        if label and (it % log_every == 0 or it == 1):
            print(f"[{label}] iter {it:4d}  mean {c.mean():.4f}  best {best:.4f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
    hist["best_anchors"] = best_anchors.tolist()
    return hist


@torch.no_grad()
def zero_shot(policy: PlacementPolicy, netlist, samples: int = 64, use_wire_mask: bool = True) -> dict:
    """Apply a policy to a chip without any training on it: one argmax decode + `samples` samples."""
    policy.eval()
    _, greedy_cost, greedy_anchors = rollout(policy, Batch.repeat(netlist, 1), greedy=True,
                                             use_wire_mask=use_wire_mask)
    _, sampled, _ = rollout(policy, Batch.repeat(netlist, samples), use_wire_mask=use_wire_mask)
    policy.train()
    curve = np.minimum.accumulate(sampled["cost"].numpy())
    return {"argmax": greedy_cost["cost"].item(), "argmax_anchors": greedy_anchors[0].tolist(),
            "best_of_samples": float(curve[-1]), "sample_curve": curve.tolist()}
