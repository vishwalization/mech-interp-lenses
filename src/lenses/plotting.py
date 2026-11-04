"""
Figures. All functions return the ``Figure`` and save to ``out`` when given.

Colour convention used across the repo: **warm** (YlOrRd) = easy / common
items with their own token, **cool** (GnBu) = hard / rare items, grey = other
items drawn for context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_STYLE = {
    "figure.dpi": 140,
    "font.size": 11,
    "axes.edgecolor": "#c9ced6",
    "axes.linewidth": 1.0,
    "text.color": "#2b2f36",
    "axes.labelcolor": "#2b2f36",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
}


def _plt():
    import matplotlib

    matplotlib.use(matplotlib.get_backend() if matplotlib.get_backend().lower() != "agg" else "agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)
    return plt


def item_colors(known: Sequence[bool]):
    """Warm ramp for own-token items, cool ramp for fallback items, in list order."""
    plt = _plt()
    n_warm, n_cool = sum(known), len(known) - sum(known)
    warm = plt.cm.YlOrRd(np.linspace(0.45, 0.9, max(n_warm, 1)))
    cool = plt.cm.GnBu(np.linspace(0.45, 0.92, max(n_cool, 1)))
    colors, wi, ci = {}, 0, 0
    for i, k in enumerate(known):
        if k:
            colors[i] = warm[wi]; wi += 1
        else:
            colors[i] = cool[ci]; ci += 1
    return colors


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# emergence curves
# ---------------------------------------------------------------------------

def plot_emergence_curves(
    curves: Dict[int, Sequence[float]],
    items: Sequence[str],
    known: Sequence[bool],
    *,
    title: str = "Emergence of the concept across layers",
    subtitle: str = "warm = own token · cool = fallback category token",
    ylabel: str = "P(target token)",
    label_min_peak: float = 0.08,
    out: Optional[Path] = None,
):
    """One line per item with collision-aware peak labels (the main figure)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    colors = item_colors(known)
    n_layers = len(next(iter(curves.values())))
    x = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(axis="y", color="#eef0f3", linewidth=1.0, zorder=0)
    _despine(ax)

    targets = []
    for i, item in enumerate(items):
        y = np.asarray(curves[i], dtype=float)
        c = colors[i]
        ax.plot(x, y, color=c, lw=2.2, alpha=0.95, zorder=3, solid_capstyle="round")
        pk = int(np.argmax(y)); yp = float(y[pk])
        ax.scatter([pk], [yp], s=42, color=c, edgecolor="white", linewidth=1.4, zorder=4)
        if yp >= label_min_peak:
            targets.append((i, pk, yp, item, c))

    targets.sort(key=lambda t: (t[1], -t[2]))
    ranks = {t[0]: sum(1 for j in range(k) if abs(targets[j][1] - t[1]) <= 2) for k, t in enumerate(targets)}
    for i, pk, yp, label, c in targets:
        rank = ranks[i]
        if pk <= 4:
            dx, dy, va = 8, (4 if rank == 0 else -2 - rank * 12), ("bottom" if rank == 0 else "top")
        elif pk >= n_layers - 8:
            dx, dy, va = 8, 6 + rank * 12, "bottom"
        else:
            dx, dy, va = 6, 4 + rank * 11, "bottom"
        ax.annotate(
            label, (pk, yp), xytext=(dx, dy), textcoords="offset points", color=c, fontsize=8.5,
            fontweight="bold", ha="left", va=va, zorder=5,
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.55, shrinkA=0, shrinkB=2)
            if (rank or pk <= 4 or pk >= n_layers - 8) else None,
        )

    ax.set_xlim(-0.5, n_layers - 0.5 + 3)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("Transformer layer  (logit-lens depth)", fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14, loc="left")
    ax.text(0, 1.06, subtitle, transform=ax.get_yaxis_transform(), fontsize=10, color="#6b7280", ha="left", va="bottom")

    handles = [Line2D([0], [0], color=colors[i], lw=3, label=(items[i] if known[i] else f"{items[i]} †")) for i in range(len(items))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9,
              handlelength=1.4, labelspacing=0.45, borderaxespad=0.0, title="† = fallback token", title_fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


def plot_small_multiples(
    curves: Dict[int, Sequence[float]],
    items: Sequence[str],
    known: Sequence[bool],
    *,
    ncols: int = 5,
    title: str = "Emergence of each concept across layers",
    ylabel: str = "P(target token)",
    out: Optional[Path] = None,
):
    """One mini-panel per item, all other curves faint behind it."""
    plt = _plt()
    colors = item_colors(known)
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    n_layers = len(next(iter(curves.values())))
    x = np.arange(n_layers)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#f1f3f6", linewidth=0.9, zorder=0)
        _despine(ax)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d3d7de")
        if i >= n:
            ax.axis("off"); continue
        for j in range(n):
            if j != i:
                ax.plot(x, curves[j], color="#dfe3e9", lw=1.0, zorder=1)
        y, c = np.asarray(curves[i], dtype=float), colors[i]
        ax.plot(x, y, color=c, lw=2.4, zorder=3, solid_capstyle="round")
        pk = int(np.argmax(y))
        ax.scatter([pk], [y[pk]], s=40, color=c, edgecolor="white", linewidth=1.3, zorder=4)
        ax.annotate(f"L{pk} · {y[pk]:.2f}", (pk, y[pk]), xytext=(4, 3), textcoords="offset points", color=c, fontsize=8.5, fontweight="bold", zorder=5)
        ax.set_title(items[i] if known[i] else f"{items[i]} †", fontsize=11, fontweight="bold", color=c, loc="left", pad=6)
        ax.set_ylim(-0.03, 1.05)
        ax.tick_params(labelsize=8, colors="#6b7280")

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.9])
    fig.supxlabel("Transformer layer  (logit-lens depth)", fontsize=12, color="#2b2f36")
    fig.supylabel(ylabel, fontsize=12, color="#2b2f36")
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.01, y=0.99, ha="left")
    fig.text(0.01, 0.925, "warm = own token · cool = fallback (†) · gray = other items", fontsize=10, color="#6b7280", ha="left")
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# image grids
# ---------------------------------------------------------------------------

def plot_image_grid(
    paths: Dict[Tuple[int, object], Path],
    row_labels: Sequence[str],
    col_keys: Sequence[object],
    *,
    title: str = "Diffusion lens — pure (L,L,L) embeds",
    col_fmt=lambda k: f"L{k}",
    scale: float = 1.6,
    out: Optional[Path] = None,
):
    """rows = prompts, cols = depths/combos; each cell an image path."""
    plt = _plt()
    from PIL import Image

    n_r, n_c = len(row_labels), len(col_keys)
    fig, axes = plt.subplots(n_r, n_c, figsize=(scale * n_c, scale * n_r))
    axes = np.asarray(axes).reshape(n_r, n_c)
    for i in range(n_r):
        for j, key in enumerate(col_keys):
            ax = axes[i, j]
            p = paths.get((i, key))
            if p is not None and Path(p).exists():
                ax.imshow(Image.open(p))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_fmt(key), fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


def plot_image_strip(paths: Sequence[Path], labels: Sequence[str], *, title: str = "", out: Optional[Path] = None):
    """A single row of images (one prompt × a sweep of layer combos)."""
    plt = _plt()
    from PIL import Image

    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    for ax, p, lab in zip(np.atleast_1d(axes), paths, labels):
        if Path(p).exists():
            ax.imshow(Image.open(p))
        ax.set_title(lab, fontsize=11)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.subplots_adjust(wspace=0.08)
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# J-lens comparisons & token cards
# ---------------------------------------------------------------------------

def plot_j_vs_logit(curves: Dict[str, Dict[int, list]], items: Sequence[str], easy_count: int, *, out: Optional[Path] = None):
    plt = _plt()
    n_layers = len(curves["j_prob"][0])
    x = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for b, item in enumerate(items):
        c = plt.cm.Oranges(0.35 + 0.5 * b / max(easy_count, 1)) if b < easy_count else plt.cm.Blues(0.35 + 0.5 * (b - easy_count) / max(len(items) - easy_count, 1))
        axes[0].plot(x, curves["j_prob"][b], color=c, lw=1.6, label=item)
        axes[1].plot(x, curves["logit_prob"][b], color=c, lw=1.6, label=item)
    axes[0].set_title("DIY J-lens  P(target mass)")
    axes[1].set_title("Raw logit lens  P(target mass)")
    for ax in axes:
        ax.set_xlabel("layer"); ax.set_ylabel("prob mass of target variants"); ax.set_xlim(0, n_layers - 1); _despine(ax)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Emergence: J-transport vs raw logit lens  (warm = easy, cool = hard)", y=1.02)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


def plot_rank_curves(ranks: Dict[int, list], items: Sequence[str], easy_count: int, *, out: Optional[Path] = None):
    plt = _plt()
    n_layers = len(ranks[0])
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(10, 5))
    for b, item in enumerate(items):
        ys = [np.log10(r) if r is not None else np.nan for r in ranks[b]]
        c = plt.cm.Oranges(0.35 + 0.5 * b / max(easy_count, 1)) if b < easy_count else plt.cm.Blues(0.35 + 0.5 * (b - easy_count) / max(len(items) - easy_count, 1))
        ax.plot(x, ys, color=c, lw=1.6, label=item)
    ax.set_xlabel("layer"); ax.set_ylabel("log10(rank of best target variant)")
    ax.set_title("DIY J-lens rank curves (lower = more verbalizable)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    _despine(ax); fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig


def plot_emergence_scatter(df, *, out: Optional[Path] = None):
    """Verbal (mean of logit/J) vs visual L* per item, coloured by pattern."""
    from .token_cards import PATTERN_COLORS

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    import pandas as pd

    vals = []
    for _, row in df.iterrows():
        verbals = [float(x) for x in (row["Lstar_logit"], row["Lstar_J"]) if not pd.isna(x)]
        if not verbals or pd.isna(row["Lstar_diff"]):
            continue
        xv, yv = float(np.mean(verbals)), float(row["Lstar_diff"])
        vals += [xv, yv]
        ax.scatter(xv, yv, s=90, color=PATTERN_COLORS.get(row["pattern"], "#9ca3af"), edgecolor="white", zorder=3)
        ax.text(xv + 0.3, yv + 0.3, row["item"], fontsize=8, color="#111")
    lim = (max(vals) if vals else 36) + 2
    ax.plot([0, lim], [0, lim], "--", color="#d1d5db", lw=1)
    ax.set_xlabel(r"Verbal emergence  mean($L^*_{logit}, L^*_J$)")
    ax.set_ylabel(r"Visual emergence  $L^*_{diff}$")
    ax.set_title("Token cards in emergence space")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for name, col in PATTERN_COLORS.items():
        ax.scatter([], [], color=col, label=name)
    ax.legend(title="pattern", fontsize=8, loc="upper left", frameon=False)
    _despine(ax); fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig
