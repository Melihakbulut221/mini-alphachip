"""Policy / value network in the spirit of the paper's Fig. 2.

  netlist graph --Edge-GNN--> node embeddings --+-- mean ------------> graph embedding
                                                +-- index[current] --> current-macro embedding
  [graph, current, metadata] --fc--> state vector --deconv x2--> coarse G x G map  --+
  canvas image (occupancy, mask, neighbour pull, coords) --CNN--> G x G features ---+--> logits
  state vector + pooled CNN features --fc--> value

The paper decodes the policy purely with deconvolutions from the state vector. We add a small
CNN over the canvas (a common practical tweak) so a CPU-sized model learns in minutes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGNN(nn.Module):
    """Eq. (3) of the paper, on a dense weighted adjacency matrix:
        e_ij = fc_e(concat(v_i, v_j))            (edge update)
        v_i  = mean_{j in N(i)} w_ij * e_ij       (node update, weighted by connectivity)
    """

    def __init__(self, f_in: int, d: int = 32, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(f_in, d)
        self.edge_fc = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(layers))
        self.node_fc = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.inp(x))                                   # B N d
        B, N, d = v.shape
        w = A.unsqueeze(-1)                                       # B N N 1
        norm = w.sum(2).clamp_min(1e-6)
        for fe, fv in zip(self.edge_fc, self.node_fc):
            pair = torch.cat([v.unsqueeze(2).expand(B, N, N, d),
                              v.unsqueeze(1).expand(B, N, N, d)], -1)
            e = F.relu(fe(pair))                                  # B N N d
            agg = (e * w).sum(2) / norm                           # B N d
            v = v + F.relu(fv(agg))                               # residual node update
        return v


class PlacementPolicy(nn.Module):
    def __init__(self, grid: int = 16, f_node: int = 7, c_grid: int = 8, f_meta: int = 4, d: int = 32):
        super().__init__()
        assert grid % 4 == 0
        self.grid = grid
        self.gnn = EdgeGNN(f_node, d)
        self.state_fc = nn.Sequential(nn.Linear(2 * d + f_meta, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.to_map = nn.Linear(128, 32 * (grid // 4) ** 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(c_grid, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(40, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.value = nn.Sequential(nn.Linear(128 + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, nodes, A, cur, meta, grid_obs, mask):
        v = self.gnn(nodes, A)
        B = v.shape[0]
        g = v.mean(1)
        c = v[torch.arange(B), cur]
        s = self.state_fc(torch.cat([g, c, meta], -1))
        q = self.grid // 4
        coarse = self.deconv(self.to_map(s).view(B, 32, q, q))
        h = self.cnn(grid_obs)
        logits = self.head(torch.cat([coarse, h], 1)).flatten(1)
        logits = logits.masked_fill(~mask, -1e9)
        value = self.value(torch.cat([s, h.mean((2, 3))], -1)).squeeze(-1)
        return logits, value
