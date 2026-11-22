"""Non-learning baselines, all scored with the same cost model as the RL agent.

  random     uniformly random legal placements
  greedy     same macro order as the agent; each macro goes where it adds the least wirelength
  annealing  simulated annealing with swap / shift / relocate moves and a geometric cooling
             schedule, as in the paper's "Comparing with simulated annealing" section
"""
from __future__ import annotations

import torch

from .env import PlacementEnv, evaluate, is_legal
from .netlist import Batch, Netlist


@torch.no_grad()
def random_placements(netlist: Netlist, n: int, generator: torch.Generator):
    env = PlacementEnv(Batch.repeat(netlist, n))
    obs = env.reset()
    while not env.done:
        obs = env.step(torch.multinomial(obs["mask"].float(), 1, generator=generator).squeeze(1))
    return env.anchors, env.cost()


@torch.no_grad()
def greedy_placement(netlist: Netlist):
    """Minimise the wirelength added by each macro (ties -> closest to the canvas centre)."""
    env = PlacementEnv(Batch.repeat(netlist, 1))
    obs = env.reset()
    G = netlist.grid
    cell = torch.arange(G * G)
    to_centre = torch.hypot((cell % G) - G / 2, (cell // G) - G / 2) / G
    while not env.done:
        score = obs["grid"][:, 2].flatten(1) + 1e-3 * to_centre
        obs = env.step(score.masked_fill(~obs["mask"], float("inf")).argmin(1))
    return env.anchors[0], {k: v.item() for k, v in env.cost().items()}


@torch.no_grad()
def simulated_annealing(netlist: Netlist, steps: int, chains: int, generator: torch.Generator,
                        t_max: float = 0.05, t_min: float = 3e-4, tries: int = 8):
    """`chains` independent annealing runs in parallel. Every step costs each chain exactly one
    placement evaluation: `tries` moves are proposed and the first legal one is scored, then
    accepted with the Metropolis rule.

    Returns best anchors (chains, M, 2), best cost (chains,) and the best-so-far curves
    (chains, steps + 1), where column i is the best cost after i + 1 evaluations."""
    b = Batch.repeat(netlist, chains)
    M, G, C = b.M, b.G, chains
    rows = torch.arange(C)
    wide = Batch.repeat(netlist, C * tries)

    def rand(*shape, high):
        return torch.randint(0, high, shape, generator=generator)

    cur, cost = random_placements(netlist, C, generator)
    cur, cur_cost = cur.clone(), cost["cost"]
    best, best_cost = cur.clone(), cur_cost.clone()
    curve = torch.empty(C, steps + 1)
    curve[:, 0] = best_cost

    for step in range(1, steps + 1):
        temperature = t_max * (t_min / t_max) ** (step / steps)

        # Propose `tries` moves per chain.
        cand = cur[:, None].expand(C, tries, M, 2).clone()
        r, k = rows[:, None].expand(C, tries), torch.arange(tries)[None].expand(C, tries)
        i = rand(C, tries, high=M)
        j = (i + 1 + rand(C, tries, high=M - 1)) % M                  # j != i
        kind = rand(C, tries, high=3)
        radius = 1 + rand(C, tries, 1, high=3)
        shift = (torch.rand(C, tries, 2, generator=generator) * (2 * radius + 1)).long() - radius
        anywhere = rand(C, tries, 2, high=G)
        pi, pj = cand[r, k, i], cand[r, k, j]
        new_i = torch.where((kind == 0)[..., None], pj,                # swap
                            torch.where((kind == 1)[..., None], pi + shift, anywhere))
        new_j = torch.where((kind == 0)[..., None], pi, pj)
        cand[r, k, i], cand[r, k, j] = new_i, new_j

        # First legal proposal of each chain (if none is legal the chain stays put).
        legal = is_legal(wide, cand.view(C * tries, M, 2)).view(C, tries)
        pick = legal.float().argmax(1)
        new = torch.where(legal.any(1)[:, None, None], cand[rows, pick], cur)
        new_cost = evaluate(b, new)["cost"]

        accept = torch.rand(C, generator=generator) < torch.exp((cur_cost - new_cost) / temperature)
        cur = torch.where(accept[:, None, None], new, cur)
        cur_cost = torch.where(accept, new_cost, cur_cost)
        improved = cur_cost < best_cost
        best = torch.where(improved[:, None, None], cur, best)
        best_cost = torch.where(improved, cur_cost, best_cost)
        curve[:, step] = best_cost
    return best, best_cost, curve
