"""Synthetic netlists: macros (to be placed) + fixed I/O ports, connected by hyperedges (nets).

Real AlphaChip netlists come from TPU blocks: hundreds of macros (SRAMs) plus millions of
standard cells clustered into a few thousand groups. Here we keep only the part the RL agent
actually decides on - macros - and generate small random netlists with hidden structure
(macros belong to "functional units" that are wired together more densely), so there is
something for a policy to learn and transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Netlist:
    grid: int                 # canvas is grid x grid cells
    sizes: np.ndarray         # (M, 2) int  macro width/height in cells
    port_xy: np.ndarray       # (P, 2) float fixed port centres (on the canvas boundary)
    nets: list                # list of int arrays; node ids 0..M-1 macros, M..M+P-1 ports
    name: str = "netlist"
    _adj: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_macros(self) -> int:
        return len(self.sizes)

    @property
    def n_ports(self) -> int:
        return len(self.port_xy)

    @property
    def n_nodes(self) -> int:
        return self.n_macros + self.n_ports

    def adjacency(self) -> np.ndarray:
        """Clique model of the hypergraph: a k-pin net adds 1/(k-1) to every pin pair."""
        if self._adj is None:
            A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            for net in self.nets:
                w = 1.0 / (len(net) - 1)
                for i in net:
                    for j in net:
                        if i != j:
                            A[i, j] += w
            self._adj = A
        return self._adj

    def placement_order(self) -> np.ndarray:
        """Same heuristic as the paper: biggest macros first (fewer dead ends later),
        ties broken by connectivity."""
        area = self.sizes[:, 0] * self.sizes[:, 1]
        degree = self.adjacency()[: self.n_macros].sum(1)
        return np.lexsort((-degree, -area))  # primary key: -area


def random_netlist(seed: int, grid: int = 16, n_macros: int = 12, n_ports: int = 6,
                   n_nets: int = 26, n_units: int = 3, max_util: float = 0.40) -> Netlist:
    rng = np.random.default_rng(seed)

    # Macro sizes 1..4 cells per side, rejected until total area fits the utilisation budget.
    while True:
        sizes = rng.integers(1, 5, size=(n_macros, 2))
        if (sizes[:, 0] * sizes[:, 1]).sum() <= max_util * grid * grid:
            break

    # Ports on the four edges of the canvas.
    port_xy = np.zeros((n_ports, 2))
    for p in range(n_ports):
        t = rng.uniform(0.1, 0.9) * grid
        side = rng.integers(4)
        port_xy[p] = [(t, 0.0), (t, grid), (0.0, t), (grid, t)][side]

    # Hidden functional units: each macro / port belongs to one.
    macro_unit = rng.integers(n_units, size=n_macros)
    port_unit = rng.integers(n_units, size=n_ports)

    nets = []
    for _ in range(n_nets):
        u = rng.integers(n_units)
        k = rng.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
        local = np.flatnonzero(macro_unit == u)
        pool = local if len(local) >= 2 and rng.random() < 0.8 else np.arange(n_macros)
        pins = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        if rng.random() < 0.35:  # some nets go off-block through a port of the same unit
            cand = np.flatnonzero(port_unit == u)
            if len(cand):
                pins.append(n_macros + int(rng.choice(cand)))
        if len(pins) >= 2:
            nets.append(np.array(sorted(set(pins))))

    # Every macro must be connected to something.
    for m in range(n_macros):
        if not any(m in n for n in nets):
            other = rng.choice([x for x in range(n_macros) if x != m])
            nets.append(np.array(sorted([m, other])))

    return Netlist(grid=grid, sizes=sizes, port_xy=port_xy, nets=nets, name=f"chip-{seed}")
