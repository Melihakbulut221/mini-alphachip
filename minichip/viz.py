"""Figures. One visual language throughout: neutral macros, a single blue ramp for magnitudes
(congestion, policy probability), three fixed series colours for the three search methods."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import env as E  # noqa: E402
from .netlist import Batch  # noqa: E402

SURFACE, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MACRO, MACRO_EDGE, WIRE = "#dddcd6", "#52514e", "#8a897f"
SERIES = {"sa": "#eb6834", "scratch": "#2a78d6", "finetune": "#1baf7a"}   # orange, blue, aqua
BLUES = LinearSegmentedColormap.from_list("blues", [SURFACE, "#b7d3f6", "#5598e7", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": RULE, "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False,
})


def congestion_map(nl, anchors) -> np.ndarray:
    b = Batch.repeat(nl, 1)
    a = torch.as_tensor(np.asarray(anchors))[None]
    everything = torch.ones(1, b.N, dtype=torch.bool)
    lo, hi, count = E.net_boxes(b, E.centres(b, a), everything)
    occ = E.rasterize(b, a, everything[:, : b.M])
    uh, uv = E.utilisation(*E.routing_demand(b, lo, hi, count), occ)
    return torch.maximum(uh, uv)[0].numpy()


def draw_canvas(ax, nl, anchors, placed=None, heat=None, heat_max=1.5, highlight=None, title=""):
    """Macros as neutral rectangles, nets as thin star wires, ports as squares on the edge."""
    G, M = nl.grid, nl.n_macros
    anchors = np.asarray(anchors)
    placed = np.ones(M, bool) if placed is None else np.asarray(placed, bool)
    on = np.concatenate([placed, np.ones(len(nl.port_xy), bool)])
    pos = np.concatenate([anchors + nl.sizes / 2.0, nl.port_xy])
    if heat is not None:
        ax.imshow(heat, origin="lower", extent=(0, G, 0, G), cmap=BLUES, vmin=0, vmax=heat_max,
                  interpolation="nearest", zorder=0)
    for net in nl.nets:
        pins = [p for p in net if on[p]]
        if len(pins) >= 2:
            hub = pos[pins].mean(0)
            for p in pins:
                ax.plot([hub[0], pos[p][0]], [hub[1], pos[p][1]], color=WIRE, lw=0.6, alpha=0.7, zorder=2)
    for m in np.flatnonzero(placed):
        (x, y), (w, h) = anchors[m], nl.sizes[m]
        new = highlight is not None and m == highlight
        ax.add_patch(Rectangle((x + 0.06, y + 0.06), w - 0.12, h - 0.12, zorder=3,
                               fc="#f4c9b5" if new else MACRO, ec=INK if new else MACRO_EDGE,
                               lw=1.6 if new else 0.8))
        ax.text(x + w / 2, y + h / 2, str(m), ha="center", va="center", fontsize=7, color=INK, zorder=4)
    ax.scatter(nl.port_xy[:, 0], nl.port_xy[:, 1], marker="s", s=22, color=INK, zorder=5, clip_on=False)
    ax.set(xlim=(0, G), ylim=(0, G), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set(visible=True, color=INK2, linewidth=0.8)
    ax.set_title(title, fontsize=9, fontweight="normal", loc="left")


def placements_figure(nl, panels: dict, path):
    """panels: {title: (anchors, cost dict)}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.75))
    for ax, (name, (anchors, c)) in zip(axes, panels.items()):
        draw_canvas(ax, nl, anchors, heat=congestion_map(nl, anchors),
                    title=f"{name}\ncost {c['cost']:.3f}  (wire {c['wirelength']:.2f}, cong. {c['congestion']:.2f})")
    fig.suptitle(f"{nl.name}: {nl.n_macros} macros, {len(nl.port_xy)} I/O ports, {len(nl.nets)} nets."
                 "  Blue shading = routing congestion.", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def convergence_figure(curves: dict, greedy_cost: float, path, title: str):
    """curves: {key: (label, evals array, runs array [n_runs, len(evals)])}.
    Line = mean over runs, band = min..max over runs."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ends = []
    for key, (label, x, runs) in curves.items():
        mean = runs.mean(0)
        ax.fill_between(x, runs.min(0), runs.max(0), color=SERIES[key], alpha=0.12, lw=0)
        ax.plot(x, mean, color=SERIES[key], label=label)
        ax.scatter([x[-1]], [mean[-1]], s=42, color=SERIES[key], ec=SURFACE, lw=1.5, zorder=5)
        ends.append((mean[-1], x[-1]))
    ax.axhline(greedy_cost, color=INK2, lw=1)
    ax.text(1.05, greedy_cost, "greedy heuristic (1 placement)", va="bottom", fontsize=8, color=INK2)
    # end-of-line value labels, nudged apart so they never collide
    ys = np.array([e[0] for e in ends])
    order = np.argsort(ys)
    label_y = ys.copy()
    gap = 0.022 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 2
    for a, b in zip(order[:-1], order[1:]):
        if label_y[b] - label_y[a] < gap:
            label_y[b] = label_y[a] + gap
    for (y, x), ly in zip(ends, label_y):
        ax.text(x * 1.12, ly, f"{y:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, ends[0][1] * 2.2)
    ax.set_xlabel("placements evaluated (log scale)")
    ax.set_ylabel("best cost found so far  (lower is better)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars_figure(rows: list, path, title: str, xlabel: str, note: str = ""):
    """rows: [(label, value, emphasised)] -> horizontal bars with the value at the tip."""
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(rows) + 1.3))
    labels, values, strong = zip(*rows)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, values, height=0.42, color=["#2a78d6" if s else "#b9b8b0" for s in strong])
    for yi, v in zip(y, values):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel(xlabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def agent_gif(policy, nl, path, caption: str):
    """Animate one argmax rollout: the blue heat is the policy's probability for where the next
    macro should go; the last frame shows the finished placement with its routing congestion."""
    from PIL import Image

    policy.eval()
    env = E.PlacementEnv(Batch.repeat(nl, 1))
    obs = env.reset()
    frames = []

    def snap(heat, heat_max, highlight, title, caption=caption):
        fig, ax = plt.subplots(figsize=(4.2, 4.75), dpi=130)
        draw_canvas(ax, nl, env.anchors[0].numpy(), placed=env.placed[0, : nl.n_macros].numpy(),
                    heat=heat, heat_max=heat_max, highlight=highlight, title=title)
        fig.text(0.03, 0.02, caption, fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

    last = None
    while not env.done:
        logits, _ = policy(obs)
        prob = torch.softmax(logits, -1).view(nl.grid, nl.grid).numpy()
        cur = int(obs["cur"][0])
        w, h = nl.sizes[cur]
        # spread each corner probability over the macro footprint so the heat shows the body
        body = np.zeros_like(prob)
        for y, x in zip(*np.nonzero(prob > 1e-4)):
            body[y:y + h, x:x + w] += prob[y, x]
        snap(body, max(body.max(), 1e-6), last,
             f"step {env.t + 1}/{nl.n_macros}: where should macro {cur} ({w}x{h}) go?")
        obs = env.step(logits.argmax(-1))
        last = cur
    c = {k: v.item() for k, v in env.cost().items()}
    snap(congestion_map(nl, env.anchors[0].numpy()), 1.5, None,
         f"done: cost {c['cost']:.3f}  (wire {c['wirelength']:.2f}, cong. {c['congestion']:.2f})",
         caption="mini-AlphaChip: finished placement, blue = routing congestion")
    frames += [frames[-1]] * 5
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=550, loop=0)
    policy.train()
    return env.anchors[0].tolist(), c
