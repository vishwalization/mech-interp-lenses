"""
Logit lens (nostalgebraist, 2020) on the Qwen3 text encoder.

Question answered: *if I decode this layer's residual as if it were the final
one, which word comes out?*

All functions take a :class:`~lenses.activations.HiddenStates` (CPU snapshots)
and an :class:`~lenses.unembed.Unembed`; nothing here touches the GPU or the
live model, so re-running is deterministic and cannot corrupt the cache.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .activations import HiddenStates
from .unembed import Unembed

TopK = List[Tuple[str, float]]


@torch.inference_mode()
def logit_lens(
    hs: HiddenStates,
    unembed: Unembed,
    b: int,
    pos: int,
    topk: int = 5,
    layers: Optional[Sequence[int]] = None,
) -> Dict[int, TopK]:
    """Top-``k`` decoded tokens at every layer for one ``(prompt, position)``.

    The final RMSNorm is applied to every layer except the last, which HF
    already returns post-norm (see :class:`HiddenStates`).
    """
    layers = range(hs.n_layers) if layers is None else layers
    return {
        L: unembed.topk(hs.vector(L, b, pos), k=topk, apply_norm=(L != hs.last))
        for L in layers
    }


@torch.inference_mode()
def emergence_curve(
    hs: HiddenStates,
    unembed: Unembed,
    b: int,
    pos: int,
    vocab_ids: Sequence[int],
) -> List[float]:
    """``P(target)`` per layer, summed over all vocab variants of the target word."""
    ids = torch.as_tensor(list(vocab_ids), dtype=torch.long)
    out = []
    for L in range(hs.n_layers):
        probs = unembed.probs(hs.vector(L, b, pos), apply_norm=(L != hs.last))
        out.append(float(probs[ids].sum()) if len(ids) else 0.0)
    return out


@torch.inference_mode()
def rank_curve(
    hs: HiddenStates,
    unembed: Unembed,
    b: int,
    pos: int,
    vocab_ids: Sequence[int],
    transport=None,
) -> Tuple[List[float], List[Optional[int]]]:
    """``(prob_mass, rank_of_best_variant)`` per layer.

    ``transport`` is an optional callable ``(v, L) -> v'`` applied *before* the
    norm — pass a :class:`lenses.j_lens.DiagonalTransport` to get J-lens curves
    from the same function.
    """
    masses, ranks = [], []
    for L in range(hs.n_layers):
        v = hs.vector(L, b, pos)
        if transport is not None:
            v = transport(v, L)
        m, r = unembed.target_mass_and_rank(v, list(vocab_ids), apply_norm=(L != hs.last))
        masses.append(m)
        ranks.append(r)
    return masses, ranks


def first_emergence_layer(
    prob_curve: Sequence[float],
    rank_curve_: Optional[Sequence[Optional[int]]] = None,
    *,
    prob_thresh: float = 0.05,
    rank_thresh: Optional[int] = 50,
    skip_embedding_layer: bool = True,
) -> Optional[int]:
    """``L*`` — the first layer where the target is "verbal".

    A layer counts if ``P(target) ≥ prob_thresh`` **or** the best variant ranks
    within ``rank_thresh``. Layer 0 is the raw token embedding, which trivially
    decodes to itself, so it is skipped by default.
    """
    start = 1 if skip_embedding_layer else 0
    for L in range(start, len(prob_curve)):
        ok = prob_curve[L] >= prob_thresh
        if not ok and rank_curve_ is not None and rank_thresh is not None:
            r = rank_curve_[L]
            ok = r is not None and r <= rank_thresh
        if ok:
            return L
    return None


def top1_table(
    hs: HiddenStates,
    unembed: Unembed,
    positions: Dict[int, int],
    items: Sequence[str],
):
    """Layer × item DataFrame of top-1 ``(token, prob)`` — the "word prob table"."""
    import pandas as pd

    cols = {}
    for b, item in enumerate(items):
        rows = logit_lens(hs, unembed, b, positions[b], topk=1)
        cols[(item, "token")] = [rows[L][0][0] for L in range(hs.n_layers)]
        cols[(item, "prob")] = [rows[L][0][1] for L in range(hs.n_layers)]
    df = pd.DataFrame(cols, index=pd.Index(range(hs.n_layers), name="layer"))
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["item", ""])
    return df


def format_side_by_side(
    left: Dict[int, TopK],
    right: Dict[int, TopK],
    left_name: str = "with chat template",
    right_name: str = "without chat template",
    width: int = 70,
) -> str:
    """Pretty two-column dump used by the CLI."""
    lines = [f"{'L':>3}  {left_name:<{width}}  {right_name}", "-" * (width * 2 + 8)]
    for L in sorted(left):
        lines.append(f"{L:3d}  {str(left[L]):<{width}}  {right.get(L, '')}")
    return "\n".join(lines)
