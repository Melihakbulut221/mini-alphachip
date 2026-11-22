"""Macro placement as an MDP, following Mirhoseini et al. (Nature 2021):

* state  - partial placement + netlist graph + which macro comes next
* action - a grid cell for the current macro (infeasible cells are masked out)
* reward - 0 at every step, and at the end  -(wirelength + lambda * congestion)
           (density is a hard constraint: overlapping placements are masked, never sampled)
"""
from __future__ import annotations

import numpy as np

from .netlist import Netlist

CONGESTION_WEIGHT = 0.5      # lambda in the paper's eq. (2) (scaled to our normalisation)
TRACKS_PER_CELL = 4.0        # routing capacity (tracks) of a free cell
MACRO_TRACK_FRACTION = 0.5   # macros block half of the routing resources above them
N_GRID_CHANNELS = 8


# ----------------------------------------------------------------------------- cost model
def node_positions(nl: Netlist, anchors: np.ndarray) -> np.ndarray:
    """Centres of all nodes. anchors: (M, 2) lower-left cell of each macro (-1 = unplaced)."""
    pos = np.empty((nl.n_nodes, 2))
    pos[: nl.n_macros] = anchors + nl.sizes / 2.0
    pos[nl.n_macros:] = nl.port_xy
    return pos


def hpwl(nl: Netlist, pos: np.ndarray, placed: np.ndarray | None = None) -> float:
    """Half-perimeter wirelength, normalised by (#nets * grid) like the paper's eq. (5).
    With `placed` given, only already-placed pins count (used by the greedy baseline)."""
    total = 0.0
    for net in nl.nets:
        if placed is not None:
            net = net[placed[net]]
            if len(net) < 2:
                continue
        p = pos[net]
        total += np.ptp(p[:, 0]) + np.ptp(p[:, 1])
    return total / (len(nl.nets) * nl.grid)


def occupancy(nl: Netlist, anchors: np.ndarray, placed: np.ndarray) -> np.ndarray:
    occ = np.zeros((nl.grid, nl.grid), dtype=np.float32)  # occ[y, x]
    for m in np.flatnonzero(placed[: nl.n_macros]):
        x, y = anchors[m]
        w, h = nl.sizes[m]
        occ[y:y + h, x:x + w] = 1.0
    return occ


def congestion_maps(nl: Netlist, pos: np.ndarray, occ: np.ndarray):
    """RUDY-style estimate: every net spreads its horizontal / vertical wire evenly over its
    bounding box. Separate H and V demand, divided by the per-cell routing supply."""
    G = nl.grid
    H = np.zeros((G, G))
    V = np.zeros((G, G))
    for net in nl.nets:
        p = pos[net]
        x0, y0 = np.clip(np.floor(p.min(0)).astype(int), 0, G - 1)
        x1, y1 = np.clip(np.floor(p.max(0) - 1e-9).astype(int), 0, G - 1)
        x1, y1 = max(x0, x1), max(y0, y1)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        H[y0:y1 + 1, x0:x1 + 1] += 1.0 / h   # total horizontal wire = w cells over w*h area
        V[y0:y1 + 1, x0:x1 + 1] += 1.0 / w
    cap = TRACKS_PER_CELL * (1.0 - MACRO_TRACK_FRACTION * occ)
    return H / cap, V / cap


def congestion(nl: Netlist, pos: np.ndarray, occ: np.ndarray) -> float:
    """Average of the top-10% most congested (cell, direction) values - the paper's ABA10-like proxy."""
    H, V = congestion_maps(nl, pos, occ)
    vals = np.sort(np.concatenate([H.ravel(), V.ravel()]))
    k = max(1, len(vals) // 10)
    return float(vals[-k:].mean())


def placement_cost(nl: Netlist, anchors: np.ndarray) -> dict:
    placed = np.ones(nl.n_nodes, dtype=bool)
    pos = node_positions(nl, anchors)
    occ = occupancy(nl, anchors, placed)
    wl = hpwl(nl, pos)
    cg = congestion(nl, pos, occ)
    return {"cost": wl + CONGESTION_WEIGHT * cg, "wirelength": wl, "congestion": cg}


def feasible_mask(occ: np.ndarray, w: int, h: int) -> np.ndarray:
    """mask[y, x] = True if a w x h macro with lower-left corner at (x, y) fits without overlap."""
    G = occ.shape[0]
    S = np.zeros((G + 1, G + 1))
    S[1:, 1:] = occ.cumsum(0).cumsum(1)
    mask = np.zeros((G, G), dtype=bool)
    ny, nx = G - h + 1, G - w + 1
    if ny <= 0 or nx <= 0:
        return mask
    win = S[h:h + ny, w:w + nx] - S[0:ny, w:w + nx] - S[h:h + ny, 0:nx] + S[0:ny, 0:nx]
    mask[:ny, :nx] = win < 0.5
    return mask


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    G = a.shape[0]
    p = np.pad(a, r)
    S = np.zeros((G + 2 * r + 1, G + 2 * r + 1))
    S[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (S[k:k + G, k:k + G] - S[0:G, k:k + G] - S[k:k + G, 0:G] + S[0:G, 0:G]) / (k * k)


# ----------------------------------------------------------------------------- environment
class PlacementEnv:
    def __init__(self, nl: Netlist):
        self.nl = nl
        self.order = nl.placement_order()
        self.A = nl.adjacency()
        G = nl.grid
        ys, xs = np.mgrid[0:G, 0:G]
        self._coords = np.stack([xs / (G - 1), ys / (G - 1)]).astype(np.float32)
        self.reset()

    def reset(self):
        nl = self.nl
        self.t = 0
        self.anchors = np.full((nl.n_macros, 2), -1, dtype=int)
        self.placed = np.zeros(nl.n_nodes, dtype=bool)
        self.placed[nl.n_macros:] = True  # ports are fixed
        self.occ = np.zeros((nl.grid, nl.grid), dtype=np.float32)
        return self.observe()

    @property
    def current(self) -> int:
        return int(self.order[self.t])

    @property
    def done(self) -> bool:
        return self.t >= self.nl.n_macros

    def mask(self) -> np.ndarray:
        w, h = self.nl.sizes[self.current]
        return feasible_mask(self.occ, w, h)

    def observe(self) -> dict:
        nl, G = self.nl, self.nl.grid
        cur = self.current if not self.done else 0
        pos = node_positions(nl, np.maximum(self.anchors, 0))

        # Node features for the Edge-GNN: w, h, is_port, is_placed, x, y, is_current.
        f = np.zeros((nl.n_nodes, 7), dtype=np.float32)
        f[: nl.n_macros, 0:2] = nl.sizes / G
        f[nl.n_macros:, 2] = 1.0
        f[:, 3] = self.placed
        f[:, 4:6] = np.where(self.placed[:, None], pos / G, 0.0)
        f[cur, 6] = 1.0

        # Image features for the CNN branch.
        pull = np.zeros((G, G), dtype=np.float32)  # where the current macro's neighbours sit
        for j in np.flatnonzero((self.A[cur] > 0) & self.placed):
            x, y = np.clip(pos[j].astype(int), 0, G - 1)
            pull[y, x] += self.A[cur, j]
        mask = self.mask() if not self.done else np.zeros((G, G), dtype=bool)
        w, h = nl.sizes[cur]
        grid_obs = np.stack([
            self.occ, mask.astype(np.float32), pull, _box_blur(pull, 2) * 4.0,
            self._coords[0], self._coords[1],
            np.full((G, G), w / G, np.float32), np.full((G, G), h / G, np.float32),
        ]).astype(np.float32)
        meta = np.array([self.t / nl.n_macros, w / G, h / G, self.occ.mean()], dtype=np.float32)
        return {"nodes": f, "grid": grid_obs, "mask": mask.ravel(), "cur": cur, "meta": meta}

    def step(self, action: int):
        G = self.nl.grid
        y, x = divmod(int(action), G)
        m = self.current
        w, h = self.nl.sizes[m]
        assert self.mask()[y, x], "infeasible action"
        self.anchors[m] = (x, y)
        self.placed[m] = True
        self.occ[y:y + h, x:x + w] = 1.0
        self.t += 1
        if self.done:
            c = placement_cost(self.nl, self.anchors)
            return None, -c["cost"], True, c
        return self.observe(), 0.0, False, {}
