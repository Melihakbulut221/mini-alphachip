"""Non-learning baselines: random, greedy, and simulated annealing (as described in the paper's
'Comparing with simulated annealing' section: swap / shift moves, exponential cooling)."""
from __future__ import annotations

import numpy as np

from .env import PlacementEnv, feasible_mask, hpwl, node_positions, occupancy, placement_cost
from .netlist import Netlist


def random_placement(nl: Netlist, rng: np.random.Generator) -> np.ndarray:
    env = PlacementEnv(nl)
    while not env.done:
        env.step(rng.choice(np.flatnonzero(env.mask().ravel())))
    return env.anchors.copy()


def greedy_placement(nl: Netlist) -> np.ndarray:
    """Same macro order as the agent; each macro goes to the feasible cell that minimises the
    wirelength of the nets connecting it to what is already placed (ties -> canvas centre)."""
    env = PlacementEnv(nl)
    G = nl.grid
    while not env.done:
        m = env.current
        cand = np.flatnonzero(env.mask().ravel())
        best, best_a = np.inf, None
        for a in cand:
            y, x = divmod(int(a), G)
            anchors = env.anchors.copy()
            anchors[m] = (x, y)
            placed = env.placed.copy()
            placed[m] = True
            pos = node_positions(nl, np.maximum(anchors, 0))
            c = hpwl(nl, pos, placed)
            c += 1e-4 * np.hypot(x - G / 2, y - G / 2)  # tie-break
            if c < best:
                best, best_a = c, a
        env.step(best_a)
    return env.anchors.copy()


def _fits(nl, anchors, m, xy):
    others = np.ones(nl.n_nodes, dtype=bool)
    others[m] = False
    others[nl.n_macros:] = False
    occ = occupancy(nl, anchors, others)
    w, h = nl.sizes[m]
    x, y = xy
    return 0 <= x <= nl.grid - w and 0 <= y <= nl.grid - h and feasible_mask(occ, w, h)[y, x]


def simulated_annealing(nl: Netlist, evals: int, rng: np.random.Generator,
                        t_max: float = 0.05, t_min: float = 1e-4, init: np.ndarray | None = None):
    """Each step proposes a swap of two macros or a shift of one macro (radius 1-3 cells) and
    accepts it with the Metropolis rule. Returns best anchors and the best-so-far curve."""
    cur = random_placement(nl, rng) if init is None else init.copy()
    cur_cost = placement_cost(nl, cur)["cost"]
    best, best_cost = cur.copy(), cur_cost
    curve = np.empty(evals)
    M = nl.n_macros
    for step in range(evals):
        t = t_max * (t_min / t_max) ** (step / evals)
        new = cur.copy()
        if rng.random() < 0.5:  # swap
            i, j = rng.choice(M, 2, replace=False)
            new[i], new[j] = cur[j], cur[i]
            ok = _fits(nl, new, i, new[i]) and _fits(nl, new, j, new[j])
        else:                   # shift
            i = rng.integers(M)
            r = rng.integers(1, 4)
            new[i] = cur[i] + rng.integers(-r, r + 1, size=2)
            ok = _fits(nl, new, i, new[i])
        if ok:
            c = placement_cost(nl, new)["cost"]
            if c < cur_cost or rng.random() < np.exp((cur_cost - c) / t):
                cur, cur_cost = new, c
                if c < best_cost:
                    best, best_cost = new.copy(), c
        curve[step] = best_cost
    return best, curve
