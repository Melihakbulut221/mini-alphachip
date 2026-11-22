"""Synthetic netlists.

A real AlphaChip netlist is a TPU block: hundreds of macros (SRAMs) plus millions of standard
cells. We keep only what the RL agent actually decides on - the macros - and generate small
random netlists: `n_macros` rectangles, `n_ports` fixed I/O pins on the canvas edge, and nets
(hyperedges) between them. Macros belong to hidden "functional units" that are wired together
more densely, so there is structure for a policy to learn and to transfer between chips.

`Batch` stacks netlists into padded tensors so that B placements - of the same chip or of
different chips - can be simulated and scored in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

MAX_PINS = 5  # largest net: 4 macros + 1 port


@dataclass
class Netlist:
    grid: int              # canvas is grid x grid cells
    sizes: np.ndarray      # (M, 2) int: macro width, height in cells
    port_xy: np.ndarray    # (P, 2) float: fixed port positions on the canvas boundary
    nets: list             # list of int arrays; node ids 0..M-1 are macros, M..M+P-1 ports
    name: str = "chip"
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_macros(self) -> int:
        return len(self.sizes)

    @property
    def n_nodes(self) -> int:
        return len(self.sizes) + len(self.port_xy)

    def adjacency(self) -> np.ndarray:
        """Clique expansion of the hypergraph: a k-pin net adds 1/(k-1) between each pin pair."""
        A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for net in self.nets:
            for i in net:
                for j in net:
                    if i != j:
                        A[i, j] += 1.0 / (len(net) - 1)
        return A

    def placement_order(self) -> np.ndarray:
        """The paper's heuristic: largest macros first (so late macros still fit),
        ties broken by connectivity."""
        area = self.sizes[:, 0] * self.sizes[:, 1]
        degree = self.adjacency()[: self.n_macros].sum(1)
        return np.lexsort((-degree, -area))

    def tensors(self) -> dict:
        if not self._cache:
            E = len(self.nets)
            pins = torch.zeros(E, MAX_PINS, dtype=torch.long)
            pin_ok = torch.zeros(E, MAX_PINS, dtype=torch.bool)
            for e, net in enumerate(self.nets):
                pins[e, : len(net)] = torch.as_tensor(net)
                pin_ok[e, : len(net)] = True
            self._cache = dict(
                sizes=torch.as_tensor(self.sizes, dtype=torch.long),
                port_xy=torch.as_tensor(self.port_xy, dtype=torch.float32),
                pins=pins, pin_ok=pin_ok,
                adj=torch.as_tensor(self.adjacency()),
                order=torch.as_tensor(self.placement_order().copy(), dtype=torch.long),
            )
        return self._cache


class Batch:
    """B netlists as padded tensors. All must share grid size, macro count and port count."""

    def __init__(self, netlists):
        t = [nl.tensors() for nl in netlists]
        first = netlists[0]
        assert all(nl.grid == first.grid and nl.sizes.shape == first.sizes.shape
                   and nl.port_xy.shape == first.port_xy.shape for nl in netlists)
        self.G, self.M, self.N, self.B = first.grid, first.n_macros, first.n_nodes, len(netlists)
        E = max(x["pins"].shape[0] for x in t)

        def pad(x):  # pad the net dimension; padded nets have no valid pins
            return torch.nn.functional.pad(x, (0, 0, 0, E - x.shape[0]))

        self.sizes = torch.stack([x["sizes"] for x in t])
        self.port_xy = torch.stack([x["port_xy"] for x in t])
        self.pins = torch.stack([pad(x["pins"]) for x in t])
        self.pin_ok = torch.stack([pad(x["pin_ok"]) for x in t])
        self.adj = torch.stack([x["adj"] for x in t])
        self.order = torch.stack([x["order"] for x in t])
        self.n_nets = torch.tensor([float(len(nl.nets)) for nl in netlists])
        self.rows = torch.arange(self.B)
        ids = {}
        self.group = torch.tensor([ids.setdefault(id(nl), len(ids)) for nl in netlists])  # same chip -> same id

    @classmethod
    def repeat(cls, netlist: Netlist, n: int) -> "Batch":
        """n parallel copies of one netlist (shares memory, nothing is copied)."""
        b = cls([netlist])
        for k in ("sizes", "port_xy", "pins", "pin_ok", "adj", "order", "n_nets"):
            v = getattr(b, k)
            setattr(b, k, v.expand(n, *v.shape[1:]))
        b.B, b.rows, b.group = n, torch.arange(n), torch.zeros(n, dtype=torch.long)
        return b


def random_netlist(seed: int, grid: int = 16, n_macros: int = 12, n_ports: int = 6,
                   n_nets: int = 26, n_units: int = 3, max_util: float = 0.40) -> Netlist:
    rng = np.random.default_rng(seed)

    # Macro sizes of 1..4 cells per side, resampled until the total area fits the budget.
    while True:
        sizes = rng.integers(1, 5, size=(n_macros, 2))
        if (sizes[:, 0] * sizes[:, 1]).sum() <= max_util * grid * grid:
            break

    # I/O ports on the four edges of the canvas.
    port_xy = np.zeros((n_ports, 2))
    for p in range(n_ports):
        t = rng.uniform(0.1, 0.9) * grid
        port_xy[p] = [(t, 0.0), (t, grid), (0.0, t), (grid, t)][rng.integers(4)]

    # Hidden functional units: nets mostly stay inside one unit.
    macro_unit = rng.integers(n_units, size=n_macros)
    port_unit = rng.integers(n_units, size=n_ports)
    nets = []
    for _ in range(n_nets):
        u = rng.integers(n_units)
        k = rng.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
        local = np.flatnonzero(macro_unit == u)
        pool = local if len(local) >= 2 and rng.random() < 0.8 else np.arange(n_macros)
        pins = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        if rng.random() < 0.35:  # some nets leave the block through a port of the same unit
            ports = np.flatnonzero(port_unit == u)
            if len(ports):
                pins.append(n_macros + int(rng.choice(ports)))
        if len(pins) >= 2:
            nets.append(np.array(sorted(set(int(p) for p in pins))))

    for m in range(n_macros):  # no floating macros
        if not any(m in net for net in nets):
            other = int(rng.choice([x for x in range(n_macros) if x != m]))
            nets.append(np.array(sorted([m, other])))

    return Netlist(grid=grid, sizes=sizes, port_xy=port_xy, nets=nets, name=f"chip-{seed}")
