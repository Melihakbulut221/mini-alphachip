"""Macro placement as a Markov decision process (Mirhoseini et al., Nature 2021), batched.

  state   partial placement + netlist graph + the macro to place next
  action  a grid cell for that macro's lower-left corner; cells that would overlap an
          already placed macro or leave the canvas are masked out (density = hard constraint)
  reward  0 at every step; after the last macro  -(wirelength + lambda * congestion)

Everything here operates on a `Batch`: B placements are simulated and scored in parallel with
plain tensor ops, and RL, simulated annealing and the simple baselines share this one cost model.
"""
from __future__ import annotations

import torch

from .netlist import Batch

LAMBDA_CONGESTION = 0.5   # weight of congestion relative to wirelength (eq. 2 in the paper)
TRACKS_PER_CELL = 4.0     # routing supply of an empty grid cell
MACRO_BLOCKAGE = 0.5      # a macro blocks this fraction of the routing tracks above it
TOP_FRACTION = 0.10       # congestion = mean of the 10% most congested values (paper: "top 10%")
GRID_CHANNELS = 8
NODE_FEATURES = 7
_FAR = 1e6


# ------------------------------------------------------------------------------ geometry
def centres(b: Batch, anchors: torch.Tensor) -> torch.Tensor:
    """(B, N, 2) positions of all pins: macro centres followed by the fixed ports."""
    return torch.cat([anchors.float() + b.sizes.float() / 2, b.port_xy], dim=1)


def rasterize(b: Batch, anchors: torch.Tensor, placed: torch.Tensor) -> torch.Tensor:
    """(B, G, G) occupancy map [y, x] of the macros flagged in `placed` (B, M)."""
    xs = torch.arange(b.G).view(1, 1, 1, -1)
    ys = torch.arange(b.G).view(1, 1, -1, 1)
    x, y = anchors[..., 0, None, None], anchors[..., 1, None, None]
    w, h = b.sizes[..., 0, None, None], b.sizes[..., 1, None, None]
    inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h) & placed[..., None, None]
    return inside.any(1).float()


def net_boxes(b: Batch, pos: torch.Tensor, placed: torch.Tensor):
    """Bounding box of the placed pins of every net: lo, hi (B, E, 2) and pin count (B, E)."""
    rows = b.rows[:, None, None]
    ok = b.pin_ok & placed[rows, b.pins]
    p = pos[rows, b.pins]                                    # B, E, K, 2
    lo = torch.where(ok[..., None], p, torch.full_like(p, _FAR)).amin(2)
    hi = torch.where(ok[..., None], p, torch.full_like(p, -_FAR)).amax(2)
    return lo, hi, ok.sum(2)


# ------------------------------------------------------------------------------ cost model
def wirelength(b: Batch, lo, hi, count) -> torch.Tensor:
    """Half-perimeter wirelength, normalised by (#nets * grid) as in the paper's eq. 5."""
    span = (hi - lo).sum(-1) * (count >= 2)
    return span.sum(1) / (b.n_nets * b.G)


def routing_demand(b: Batch, lo, hi, count):
    """RUDY-style congestion estimate. Each net spreads its horizontal and its vertical wire
    uniformly over its bounding box; returns demand maps H, V of shape (B, G, G).
    Boxes are painted in O(1) each with a 2-D difference array + cumulative sums."""
    G, S = b.G, b.G + 1
    live = (count >= 2).float()
    c0 = lo.floor().clamp(0, G - 1).long()
    c1 = torch.maximum(hi.floor().clamp(0, G - 1).long(), c0)
    x0, y0, x1, y1 = c0[..., 0], c0[..., 1], c1[..., 0] + 1, c1[..., 1] + 1
    w, h = (x1 - x0).float(), (y1 - y0).float()

    def paint(value):
        value = value * live
        d = torch.zeros(b.B, S * S)
        d.scatter_add_(1, y0 * S + x0, value)
        d.scatter_add_(1, y0 * S + x1, -value)
        d.scatter_add_(1, y1 * S + x0, -value)
        d.scatter_add_(1, y1 * S + x1, value)
        return d.view(b.B, S, S).cumsum(1).cumsum(2)[:, :G, :G]

    return paint(1.0 / h), paint(1.0 / w)   # w cells of horizontal wire spread over w*h cells


def utilisation(H, V, occ):
    """Routing demand / supply per cell; macros take away part of the supply."""
    supply = TRACKS_PER_CELL * (1.0 - MACRO_BLOCKAGE * occ)
    return H / supply, V / supply


def congestion(H, V, occ) -> torch.Tensor:
    uh, uv = utilisation(H, V, occ)
    values = torch.cat([uh.flatten(1), uv.flatten(1)], dim=1)
    k = max(1, int(TOP_FRACTION * values.shape[1]))
    return values.topk(k, dim=1).values.mean(1)


def evaluate(b: Batch, anchors: torch.Tensor) -> dict:
    """Cost of complete placements. anchors: (B, M, 2) lower-left cell (x, y) of every macro."""
    all_macros = torch.ones(b.B, b.M, dtype=torch.bool)
    all_nodes = torch.ones(b.B, b.N, dtype=torch.bool)
    occ = rasterize(b, anchors, all_macros)
    lo, hi, count = net_boxes(b, centres(b, anchors), all_nodes)
    wl = wirelength(b, lo, hi, count)
    cg = congestion(*routing_demand(b, lo, hi, count), occ)
    return {"cost": wl + LAMBDA_CONGESTION * cg, "wirelength": wl, "congestion": cg}


def is_legal(b: Batch, anchors: torch.Tensor) -> torch.Tensor:
    """(B,) True if every macro is inside the canvas and no two macros overlap."""
    lo, hi = anchors, anchors + b.sizes
    inside = ((lo >= 0) & (hi <= b.G)).all(-1).all(-1)
    sep = (hi[:, :, None, :] <= lo[:, None, :, :]) | (hi[:, None, :, :] <= lo[:, :, None, :])
    overlap = ~sep.any(-1)                                   # B, M, M
    overlap = overlap & ~torch.eye(b.M, dtype=torch.bool)
    return inside & ~overlap.flatten(1).any(1)


# ------------------------------------------------------------------------------ per-step maps
def feasible_mask(occ: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """mask[b, y, x]: a w[b] x h[b] macro with lower-left corner (x, y) fits on free cells."""
    B, G, _ = occ.shape
    S = G + 1
    integral = torch.zeros(B, S, S)
    integral[:, 1:, 1:] = occ.cumsum(1).cumsum(2)
    xs = torch.arange(G).view(1, 1, G).expand(B, G, G)
    ys = torch.arange(G).view(1, G, 1).expand(B, G, G)
    x2, y2 = xs + w.view(B, 1, 1), ys + h.view(B, 1, 1)
    fits = (x2 <= G) & (y2 <= G)
    x2, y2 = x2.clamp(max=G), y2.clamp(max=G)
    flat = integral.view(B, -1)

    def at(y, x):
        return flat.gather(1, (y * S + x).view(B, -1)).view(B, G, G)

    covered = at(y2, x2) - at(ys, x2) - at(y2, xs) + at(ys, xs)
    return fits & (covered < 0.5)


def wire_mask(b: Batch, lo, hi, count, cur, w, h) -> torch.Tensor:
    """(B, G, G) map: how much wirelength (in cells) the current macro would add if its corner
    went to each cell, given the pins placed so far. This is the "wire mask" input feature of
    MaskPlace (Lai et al., NeurIPS 2022); it is exact, cheap and separable in x and y."""
    touches = ((b.pins == cur[:, None, None]) & b.pin_ok).any(2) & (count >= 1)   # B, E
    axis = torch.arange(b.G).float()
    out = []
    for d, size in ((0, w), (1, h)):
        c = (axis[None, :] + size[:, None].float() / 2)[:, None, :]              # B, 1, G
        extra = (lo[..., d, None] - c).clamp(min=0) + (c - hi[..., d, None]).clamp(min=0)
        out.append((extra * touches[..., None]).sum(1))                           # B, G
    return out[1][:, :, None] + out[0][:, None, :]                                # [y, x]


# ------------------------------------------------------------------------------ environment
class PlacementEnv:
    """B placement episodes running in lock-step (one macro per step)."""

    def __init__(self, batch: Batch, use_wire_mask: bool = True):
        self.b = batch
        self.use_wire_mask = use_wire_mask
        G = batch.G
        self._xs = torch.arange(G).view(1, 1, G)
        self._ys = torch.arange(G).view(1, G, 1)
        self._coords = torch.stack([self._xs.expand(1, G, G), self._ys.expand(1, G, G)], 1).float() / (G - 1)
        self.reset()

    def reset(self) -> dict:
        b = self.b
        self.t = 0
        self.anchors = torch.zeros(b.B, b.M, 2, dtype=torch.long)
        self.placed = torch.zeros(b.B, b.N, dtype=torch.bool)
        self.placed[:, b.M:] = True                      # ports are fixed
        self.occ = torch.zeros(b.B, b.G, b.G)
        return self.observe()

    @property
    def done(self) -> bool:
        return self.t >= self.b.M

    def observe(self) -> dict:
        b, G = self.b, self.b.G
        cur = b.order[:, self.t]
        wh = b.sizes[b.rows, cur]
        w, h = wh[:, 0], wh[:, 1]
        mask = feasible_mask(self.occ, w, h)
        if not mask.flatten(1).any(1).all():
            raise RuntimeError("dead end: a macro has no legal position left")

        pos = centres(b, self.anchors)
        lo, hi, count = net_boxes(b, pos, self.placed)
        wm = wire_mask(b, lo, hi, count, cur, w, h) / G if self.use_wire_mask else torch.zeros_like(self.occ)
        uh, uv = utilisation(*routing_demand(b, lo, hi, count), self.occ)

        ones = torch.ones(b.B, G, G)
        grid = torch.stack([
            self.occ, mask.float(), wm, torch.maximum(uh, uv),
            self._coords[:, 0].expand(b.B, G, G), self._coords[:, 1].expand(b.B, G, G),
            ones * (w.view(-1, 1, 1) / G), ones * (h.view(-1, 1, 1) / G),
        ], dim=1)

        nodes = torch.zeros(b.B, b.N, NODE_FEATURES)
        nodes[:, : b.M, 0:2] = b.sizes.float() / G
        nodes[:, b.M:, 2] = 1.0                                  # is_port
        nodes[:, :, 3] = self.placed.float()
        nodes[:, :, 4:6] = pos / G * self.placed[..., None]      # position, 0 if unplaced
        nodes[b.rows, cur, 6] = 1.0                              # is the macro being placed
        meta = torch.stack([torch.full((b.B,), self.t / b.M), w / G, h / G, self.occ.mean((1, 2))], 1)
        return {"nodes": nodes, "adj": b.adj, "cur": cur, "meta": meta,
                "grid": grid, "mask": mask.flatten(1)}

    def step(self, action: torch.Tensor):
        b, G = self.b, self.b.G
        cur = b.order[:, self.t]
        x, y = action % G, action // G
        wh = b.sizes[b.rows, cur]
        self.anchors[b.rows, cur] = torch.stack([x, y], 1)
        self.placed[b.rows, cur] = True
        x, y, w, h = (v.view(-1, 1, 1) for v in (x, y, wh[:, 0], wh[:, 1]))
        rect = (self._xs >= x) & (self._xs < x + w) & (self._ys >= y) & (self._ys < y + h)
        self.occ = torch.maximum(self.occ, rect.float())
        self.t += 1
        return None if self.done else self.observe()

    def cost(self) -> dict:
        assert self.done
        return evaluate(self.b, self.anchors)
