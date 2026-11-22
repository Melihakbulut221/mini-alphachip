"""Policy and value network, following Fig. 2 of the paper.

  netlist graph --Edge-GNN--> node embeddings --+-- mean ------------> graph embedding
                                                +-- index[current] --> current-macro embedding
  [graph, current macro, metadata] --fc--> state vector --deconv--> G x G map --+
  canvas maps (occupancy, mask, wire mask, ...) --CNN------------> G x G map ---+--> logits
  state vector + pooled canvas features --fc--> value

The paper decodes the policy from the state vector alone, with five deconvolutions up to a
128 x 128 grid. On a 16 x 16 grid two deconvolutions are enough, and a small CNN over the
canvas lets a CPU-sized model see local geometry directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import GRID_CHANNELS, NODE_FEATURES


class EdgeGNN(nn.Module):
    """Eq. 3 of the paper on a dense, weighted adjacency matrix:

        e_ij = fc_e(concat(v_i, v_j))                edge update
        v_i  = mean over neighbours j of w_ij e_ij   node update
    """

    def __init__(self, dim: int = 32, rounds: int = 2):
        super().__init__()
        self.embed = nn.Linear(NODE_FEATURES, dim)
        self.edge_fc = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(rounds))
        self.node_fc = nn.ModuleList(nn.Linear(dim, dim) for _ in range(rounds))

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.embed(nodes))                             # B, N, d
        B, N, d = v.shape
        weight = adj.unsqueeze(-1)
        degree = weight.sum(2).clamp_min(1e-6)
        for edge_fc, node_fc in zip(self.edge_fc, self.node_fc):
            pairs = torch.cat([v[:, :, None].expand(B, N, N, d), v[:, None].expand(B, N, N, d)], -1)
            e = F.relu(edge_fc(pairs))                            # B, N, N, d
            v = v + F.relu(node_fc((e * weight).sum(2) / degree))
        return v


class PlacementPolicy(nn.Module):
    def __init__(self, grid: int = 16, dim: int = 32, channels: int = 24):
        super().__init__()
        assert grid % 4 == 0
        self.grid = grid
        self.gnn = EdgeGNN(dim)
        self.state_fc = nn.Sequential(nn.Linear(2 * dim + 4, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.seed_map = nn.Linear(96, 16 * (grid // 4) ** 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1), nn.ReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(channels + 8, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, 1, 1))
        self.value = nn.Sequential(nn.Linear(96 + channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs: dict):
        v = self.gnn(obs["nodes"], obs["adj"])
        B = v.shape[0]
        current = v[torch.arange(B), obs["cur"]]
        state = self.state_fc(torch.cat([v.mean(1), current, obs["meta"]], -1))
        q = self.grid // 4
        from_state = self.deconv(self.seed_map(state).view(B, 16, q, q))
        from_canvas = self.cnn(obs["grid"])
        logits = self.head(torch.cat([from_canvas, from_state], 1)).flatten(1)
        logits = logits.masked_fill(~obs["mask"], -1e9)           # infeasible cells: probability 0
        value = self.value(torch.cat([state, from_canvas.mean((2, 3))], -1)).squeeze(-1)
        return logits, value
