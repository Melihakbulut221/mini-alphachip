"""Checks the vectorised cost model against slow, obviously-correct reference code.

Run with `python -m pytest tests` or simply `python tests/test_env.py`.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minichip import env as E  # noqa: E402
from minichip.baselines import greedy_placement, random_placements, simulated_annealing  # noqa: E402
from minichip.netlist import Batch, random_netlist  # noqa: E402


def ref_positions(nl, anchors):
    return np.concatenate([anchors + nl.sizes / 2.0, nl.port_xy])


def ref_hpwl(nl, pos, placed=None):
    total = 0.0
    for net in nl.nets:
        pins = net if placed is None else [p for p in net if placed[p]]
        if len(pins) >= 2:
            p = pos[pins]
            total += np.ptp(p[:, 0]) + np.ptp(p[:, 1])
    return total / (len(nl.nets) * nl.grid)


def ref_occupancy(nl, anchors):
    occ = np.zeros((nl.grid, nl.grid))
    for (x, y), (w, h) in zip(anchors, nl.sizes):
        assert occ[y:y + h, x:x + w].sum() == 0, "overlap"
        occ[y:y + h, x:x + w] = 1
    return occ


def ref_congestion(nl, pos, occ):
    G = nl.grid
    H, V = np.zeros((G, G)), np.zeros((G, G))
    for net in nl.nets:
        p = pos[net]
        x0, y0 = np.clip(np.floor(p.min(0)), 0, G - 1).astype(int)
        x1, y1 = np.clip(np.floor(p.max(0)), 0, G - 1).astype(int)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        H[y0:y1 + 1, x0:x1 + 1] += 1.0 / h
        V[y0:y1 + 1, x0:x1 + 1] += 1.0 / w
    supply = E.TRACKS_PER_CELL * (1 - E.MACRO_BLOCKAGE * occ)
    values = np.sort(np.concatenate([(H / supply).ravel(), (V / supply).ravel()]))
    return values[-int(E.TOP_FRACTION * values.size):].mean()


def test_cost_matches_reference():
    gen = torch.Generator().manual_seed(0)
    for seed in (3, 17, 1000):
        nl = random_netlist(seed)
        anchors, cost = random_placements(nl, 16, gen)
        assert E.is_legal(Batch.repeat(nl, 16), anchors).all()
        for a, c, wl, cg in zip(anchors.numpy(), cost["cost"], cost["wirelength"], cost["congestion"]):
            pos = ref_positions(nl, a)
            occ = ref_occupancy(nl, a)                       # also asserts no overlap
            assert abs(ref_hpwl(nl, pos) - wl.item()) < 1e-5
            assert abs(ref_congestion(nl, pos, occ) - cg.item()) < 1e-5
            assert abs(wl + E.LAMBDA_CONGESTION * cg - c) < 1e-6


def test_mask_and_wire_mask_match_reference():
    """Step through an episode; at each step compare the feasibility mask and the wire mask
    with brute force over all 256 cells."""
    gen = torch.Generator().manual_seed(1)
    nl = random_netlist(5)
    G, M = nl.grid, nl.n_macros
    env = E.PlacementEnv(Batch.repeat(nl, 1))
    obs = env.reset()
    while not env.done:
        m = int(obs["cur"][0])
        w, h = nl.sizes[m]
        occ = env.occ[0].numpy()
        placed = env.placed[0].numpy().copy()
        anchors = env.anchors[0].numpy().copy()
        before = ref_hpwl(nl, ref_positions(nl, anchors), placed)
        placed[m] = True
        for y in range(G):
            for x in range(G):
                fits = x + w <= G and y + h <= G and occ[y:y + h, x:x + w].sum() == 0
                assert bool(obs["mask"][0, y * G + x]) == fits
                anchors[m] = (x, y)
                added = ref_hpwl(nl, ref_positions(nl, anchors), placed) - before
                got = obs["grid"][0, 2, y, x].item() * G / (len(nl.nets) * G)
                assert abs(added - got) < 1e-5, (m, x, y, added, got)
        obs = env.step(torch.multinomial(obs["mask"].float(), 1, generator=gen).squeeze(1))


def test_is_legal_detects_overlap_and_bounds():
    nl = random_netlist(5)
    b = Batch.repeat(nl, 1)
    anchors, _ = random_placements(nl, 1, torch.Generator().manual_seed(2))
    assert E.is_legal(b, anchors).item()
    clash = anchors.clone()
    clash[0, 1] = clash[0, 0]
    assert not E.is_legal(b, clash).item()
    outside = anchors.clone()
    outside[0, 0, 0] = nl.grid
    assert not E.is_legal(b, outside).item()


def test_baselines_are_legal_and_ordered():
    nl = random_netlist(1000)
    gen = torch.Generator().manual_seed(0)
    _, rnd = random_placements(nl, 256, gen)
    anchors, greedy = greedy_placement(nl)
    best, best_cost, curve = simulated_annealing(nl, steps=1500, chains=4, generator=gen)
    b = Batch.repeat(nl, 4)
    assert E.is_legal(Batch.repeat(nl, 1), anchors[None]).item()
    assert E.is_legal(b, best).all()
    assert torch.allclose(E.evaluate(b, best)["cost"], best_cost, atol=1e-6)
    assert (curve[:, 1:] <= curve[:, :-1] + 1e-9).all()      # best-so-far never gets worse
    assert greedy["cost"] < rnd["cost"].mean().item()
    assert best_cost.mean().item() < rnd["cost"].mean().item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
