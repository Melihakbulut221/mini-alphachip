from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .env import congestion_maps, node_positions, occupancy, placement_cost  # noqa: E402

UNIT_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def draw_placement(ax, nl, anchors, title="", show_congestion=True, placed=None):
    G = nl.grid
    if placed is None:
        placed = np.ones(nl.n_nodes, dtype=bool)
    else:
        placed = placed.copy()
    placed[nl.n_macros:] = True
    pos = node_positions(nl, np.maximum(anchors, 0))
    if show_congestion and placed[: nl.n_macros].all():
        occ = occupancy(nl, anchors, placed)
        H, V = congestion_maps(nl, pos, occ)
        ax.imshow(np.maximum(H, V), origin="lower", extent=(0, G, 0, G), cmap="Reds",
                  vmin=0, vmax=1.6, alpha=0.55, interpolation="nearest")
    for net in nl.nets:
        net = net[placed[net]]
        if len(net) < 2:
            continue
        p = pos[net]
        c = p.mean(0)
        for q in p:
            ax.plot([c[0], q[0]], [c[1], q[1]], color="#4a5568", lw=0.6, alpha=0.5, zorder=2)
    for m in range(nl.n_macros):
        if not placed[m]:
            continue
        x, y = anchors[m]
        w, h = nl.sizes[m]
        ax.add_patch(Rectangle((x, y), w, h, fc=UNIT_COLORS[m % 20], ec="black", lw=1.0, zorder=3))
        ax.text(x + w / 2, y + h / 2, str(m), ha="center", va="center", fontsize=7, zorder=4)
    ax.scatter(nl.port_xy[:, 0], nl.port_xy[:, 1], marker="s", s=28, c="#111", zorder=5, clip_on=False)
    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_xticks(range(0, G + 1, 4))
    ax.set_yticks(range(0, G + 1, 4))
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_aspect("equal")
    if placed[: nl.n_macros].all() and title:
        c = placement_cost(nl, anchors)
        title += f"\ncost {c['cost']:.3f}  (WL {c['wirelength']:.3f}, cong {c['congestion']:.2f})"
    ax.set_title(title, fontsize=9)


def placements_figure(nl, named_anchors: dict, path):
    n = len(named_anchors)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.9))
    for ax, (name, a) in zip(np.atleast_1d(axes), named_anchors.items()):
        draw_placement(ax, nl, a, name)
    fig.suptitle(f"{nl.name}: {nl.n_macros} macros, {nl.n_ports} I/O ports, {len(nl.nets)} nets  "
                 "(red = routing congestion)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves_figure(curves: dict, refs: dict, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (x, y) in curves.items():
        ax.plot(x, y, label=name, lw=2)
    styles = ["--", ":", "-."]
    for i, (name, v) in enumerate(refs.items()):
        ax.axhline(v, ls=styles[i % 3], color="gray", lw=1.2)
        ax.text(0, v, f" {name}", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("placements evaluated")
    ax.set_ylabel("best placement cost so far (lower is better)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def placement_gif(nl, anchors, order, path, title="RL agent placing macros"):
    from PIL import Image
    frames = []
    for t in range(nl.n_macros + 1):
        placed = np.zeros(nl.n_nodes, dtype=bool)
        placed[order[:t]] = True
        fig, ax = plt.subplots(figsize=(4, 4.3))
        draw_placement(ax, nl, anchors, f"{title}  (step {t}/{nl.n_macros})",
                       show_congestion=(t == nl.n_macros), placed=placed)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames += [frames[-1]] * 4
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0)
