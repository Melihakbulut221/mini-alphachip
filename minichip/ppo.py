"""PPO (Schulman et al. 2017) for sparse, end-of-episode placement rewards."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.distributions import Categorical

from .env import PlacementEnv
from .model import PlacementPolicy


def _batch(obs_list, adjs):
    return dict(
        nodes=torch.from_numpy(np.stack([o["nodes"] for o in obs_list])),
        A=torch.from_numpy(np.stack(adjs)),
        cur=torch.tensor([o["cur"] for o in obs_list]),
        meta=torch.from_numpy(np.stack([o["meta"] for o in obs_list])),
        grid_obs=torch.from_numpy(np.stack([o["grid"] for o in obs_list])),
        mask=torch.from_numpy(np.stack([o["mask"] for o in obs_list])),
    )


@torch.no_grad()
def rollout(policy: PlacementPolicy, netlists, greedy: bool = False):
    """Place every netlist in `netlists` once (batched). Returns transitions and final costs."""
    envs = [PlacementEnv(nl) for nl in netlists]
    adjs = [e.A for e in envs]
    obs = [e.reset() for e in envs]
    steps = envs[0].nl.n_macros
    assert all(e.nl.n_macros == steps for e in envs), "batch netlists must share a macro count"
    traj = []
    for _ in range(steps):
        b = _batch(obs, adjs)
        logits, value = policy(**b)
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        traj.append((b, a, dist.log_prob(a), value))
        nxt, infos = [], []
        for e, ai in zip(envs, a.tolist()):
            o, r, done, info = e.step(ai)
            nxt.append(o)
            infos.append(info)
        obs = nxt
    costs = np.array([i["cost"] for i in infos])
    anchors = [e.anchors.copy() for e in envs]
    return traj, costs, anchors, infos


class PPOTrainer:
    def __init__(self, policy: PlacementPolicy, lr: float = 3e-4, clip: float = 0.2, epochs: int = 4,
                 minibatch: int = 256, vf_coef: float = 0.5, ent_coef: float = 0.01):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.vf_coef, self.ent_coef = vf_coef, ent_coef

    def update(self, traj, costs):
        R = torch.tensor(-costs, dtype=torch.float32)  # only terminal reward, gamma = 1
        keys = traj[0][0].keys()
        data = {k: torch.cat([t[0][k] for t in traj]) for k in keys}
        act = torch.cat([t[1] for t in traj])
        old_logp = torch.cat([t[2] for t in traj])
        old_v = torch.cat([t[3] for t in traj])
        ret = R.repeat(len(traj))
        adv = ret - old_v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(act)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                logits, v = self.policy(**{k: x[idx] for k, x in data.items()})
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act[idx]) - old_logp[idx])
                pg = -torch.min(ratio * adv[idx],
                                ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                vloss = (v - ret[idx]).pow(2).mean()
                loss = pg + self.vf_coef * vloss - self.ent_coef * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()


def train(policy, sample_netlists, iterations: int, batch: int = 64, log_every: int = 10,
          label: str = "", **ppo_kw):
    """sample_netlists(batch) -> list of netlists for this iteration.
    Returns per-iteration history (mean / best-so-far cost, #placements evaluated)."""
    trainer = PPOTrainer(policy, **ppo_kw)
    hist = {"iter": [], "mean_cost": [], "best_cost": [], "evals": []}
    best, best_anchors, t0 = np.inf, None, time.time()
    for it in range(1, iterations + 1):
        nls = sample_netlists(batch)
        traj, costs, anchors, _ = rollout(policy, nls)
        trainer.update(traj, costs)
        if costs.min() < best:
            best, best_anchors = float(costs.min()), anchors[int(costs.argmin())]
        hist["iter"].append(it)
        hist["mean_cost"].append(float(costs.mean()))
        hist["best_cost"].append(best)
        hist["evals"].append(it * batch)
        if it % log_every == 0 or it == 1:
            print(f"[{label}] iter {it:4d}  mean cost {costs.mean():.4f}  best {best:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return hist, best, best_anchors
