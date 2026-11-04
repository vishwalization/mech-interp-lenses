"""
Token cards — fuse the three emergence depths per item and classify a pattern.

For every item we have (or will have) three numbers:

* ``L*_logit`` — first layer the target is verbal under the raw logit lens
* ``L*_J``     — first layer the target is verbalizable under the J-lens
* ``L*_diff``  — first layer the image clearly shows the object

and a *bucket* (``easy`` = common animal/fruit with its own token, ``hard`` =
rare, multi-token). :func:`classify` turns a card into one of seven patterns:

====  ==========================  ========================================
code  name                        rule of thumb
====  ==========================  ========================================
A     early verbal, late visual   logit & J early; diff ≥ LATE_GAP later
B     visual before verbal        diff early; logit/J late
C     easy triad                  all three early, easy bucket (control)
D     hard triad                  all late / missing, hard bucket
E     J-only intermediate         manual — needs multi-hop prompts
F     chat-template shift         manual — compare with/without template
G     logit ≠ J                   |L*_logit − L*_J| ≥ DISAGREE
?     incomplete                  fewer than two L* values
====  ==========================  ========================================

When one pattern repeats ≥ 3 times across items, that is the research path.
No GPU is needed here — it only reads the JSON partials the other lenses
write.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

PATTERN_NAMES = {
    "A": "early verbal, late visual",
    "B": "visual before verbal",
    "C": "easy triad",
    "D": "hard triad",
    "E": "J-only intermediate",
    "F": "chat-template shift",
    "G": "logit ≠ J",
    "?": "incomplete (fewer than two lenses run)",
}

PATTERN_COLORS = {
    "A": "#d97706", "B": "#2563eb", "C": "#16a34a", "D": "#7c3aed",
    "E": "#db2777", "F": "#0891b2", "G": "#dc2626", "?": "#9ca3af",
}


@dataclass
class Thresholds:
    """Layer thresholds for the classifier (tune to ``n_layers``; defaults assume 37)."""

    early: int = 12        # layers 0..early are "early"
    late_gap: int = 8      # diff lag vs verbal to call pattern A
    disagree: int = 8      # |logit − J| to call pattern G
    n_layers: int = 37

    def is_early(self, L: Optional[int]) -> bool:
        return L is not None and L <= self.early

    def is_late(self, L: Optional[int]) -> bool:
        return L is None or L >= max(self.early + self.late_gap, self.n_layers - 10)


LENS_KEYS = {"logit": "Lstar_logit", "J": "Lstar_J", "diff": "Lstar_diff"}


def classify(row: dict, th: Thresholds = Thresholds()) -> str:
    """Pattern code for one card.

    ``row`` is a dict/Series with ``Lstar_logit``, ``Lstar_J``, ``Lstar_diff`` and
    ``bucket``. A value of ``None`` means *the lens ran and the target never
    emerged* (which counts as "late"). A lens that was **not run** is marked by
    the optional ``measured`` field (e.g. ``"logit,J"``) or by the key being
    absent; fewer than two measured lenses → ``"?"``.
    """
    Ll, Lj, Ld = (_none_if_nan(row.get(k)) for k in ("Lstar_logit", "Lstar_J", "Lstar_diff"))
    bucket = row.get("bucket", "easy")

    measured = row.get("measured")
    if measured is None or (isinstance(measured, float) and pd.isna(measured)):
        measured = ",".join(name for name, key in LENS_KEYS.items() if key in row)
    ran = {m for m in str(measured).split(",") if m}
    if len(ran) < 2:
        return "?"
    has_logit, has_j, has_diff = "logit" in ran, "J" in ran, "diff" in ran

    # values of lenses that did not run are unknown, not "never"
    vals = {"logit": Ll if has_logit else None, "J": Lj if has_j else None, "diff": Ld if has_diff else None}
    Ll, Lj, Ld = vals["logit"], vals["J"], vals["diff"]
    verbal_lenses = [n for n in ("logit", "J") if n in ran]
    verbals = [vals[n] for n in verbal_lenses if vals[n] is not None]
    verbal = min(verbals) if verbals else None          # earliest verbal depth; None = never verbal

    # G: the two verbal lenses disagree
    if has_logit and has_j and Ll is not None and Lj is not None and abs(Ll - Lj) >= th.disagree:
        return "G"
    # A: verbal early, visual ≥ late_gap later (or never)
    if has_diff and verbal is not None and th.is_early(verbal) and (Ld is None or Ld >= verbal + th.late_gap):
        return "A"
    # B: visual early, every verbal lens that ran is ≥ late_gap later (or never)
    if has_diff and verbal_lenses and Ld is not None and th.is_early(Ld) \
            and all(vals[n] is None or vals[n] >= Ld + th.late_gap for n in verbal_lenses):
        return "B"
    # C / D: every lens that ran agrees
    if bucket == "easy" and all(th.is_early(vals[n]) for n in ran):
        return "C"
    if bucket == "hard" and all(th.is_late(vals[n]) for n in ran):
        return "D"

    # soft defaults
    if bucket == "easy" and verbal is not None and th.is_early(verbal):
        return "C"
    if bucket == "hard":
        return "D"
    return "?"


def load_partials(results_root: Path) -> Dict[str, dict]:
    """Read whichever ``*_card_partial.json`` files exist under ``results_root``."""
    results_root = Path(results_root)
    paths = {
        "logit": results_root / "logit_lens" / "logit_card_partial.json",
        "diffusion": results_root / "diffusion_lens" / "diffusion_card_partial.json",
        "j_lens": results_root / "j_lens" / "j_lens_card_partial.json",
    }
    loaded = {}
    for k, p in paths.items():
        if p.exists():
            loaded[k] = json.loads(p.read_text())
    return loaded


def build_cards(
    loaded: Dict[str, dict],
    items: Optional[Sequence[str]] = None,
    easy_count: Optional[int] = None,
    manual: Optional[Dict[str, dict]] = None,
    th: Thresholds = Thresholds(),
) -> pd.DataFrame:
    """Merge partials (+ manual overrides) into one DataFrame with a ``pattern`` column."""
    if items is None:
        for k in ("j_lens", "diffusion", "logit"):
            if k in loaded and loaded[k].get("items"):
                items = loaded[k]["items"]
                break
    if items is None:
        raise ValueError("No items given and no partial JSON found to infer them from")
    items = list(items)
    easy_count = easy_count if easy_count is not None else len(items) // 2

    j_rows = {r["item"]: r for r in loaded.get("j_lens", {}).get("rows", [])}
    l_rows = {r["item"]: r for r in loaded.get("logit", {}).get("rows", [])}
    d_human = loaded.get("diffusion", {}).get("Lstar_human", {})

    cards = []
    for i, item in enumerate(items):
        jr, lr = j_rows.get(item, {}), l_rows.get(item, {})
        measured = []
        if "Lstar_logit" in lr or "Lstar_logit" in jr:
            measured.append("logit")
        if "Lstar_J_diy" in jr:
            measured.append("J")
        if item in d_human:
            measured.append("diff")
        cards.append(
            {
                "item": item,
                "bucket": "easy" if i < easy_count else "hard",
                "Lstar_logit": lr.get("Lstar_logit", jr.get("Lstar_logit")),
                "Lstar_J": jr.get("Lstar_J_diy"),
                "Lstar_diff": d_human.get(item),
                "measured": ",".join(measured),
                "notes": "",
            }
        )
    df = pd.DataFrame(cards).astype({"Lstar_logit": "object", "Lstar_J": "object", "Lstar_diff": "object"})

    for item, overrides in (manual or {}).items():
        mask = df["item"] == item
        for k, v in overrides.items():
            df.loc[mask, k] = v
            # a manual L* counts as a measurement of that lens
            for name, key in LENS_KEYS.items():
                if k == key:
                    df.loc[mask, "measured"] = df.loc[mask, "measured"].map(
                        lambda m: ",".join(sorted(set([x for x in m.split(",") if x] + [name]), key=list(LENS_KEYS).index))
                    )

    df["pattern"] = df.apply(lambda r: classify(r.to_dict(), th), axis=1)
    df["pattern_name"] = df["pattern"].map(PATTERN_NAMES)
    for c in ("Lstar_logit", "Lstar_J", "Lstar_diff"):   # nullable ints: prints 7 / <NA>, not 7.0 / NaN
        df[c] = pd.array([_none_if_nan(x) for x in df[c]], dtype="Int64")
    return df


def leading_pattern(df: pd.DataFrame, min_repeats: int = 3) -> Optional[dict]:
    """Most frequent non-``?`` pattern and whether it clears ``min_repeats``."""
    counts = Counter(df["pattern"])
    candidates = sorted(((k, v) for k, v in counts.items() if k != "?"), key=lambda kv: -kv[1])
    if not candidates:
        return None
    code, n = candidates[0]
    return {
        "pattern": code,
        "name": PATTERN_NAMES[code],
        "count": n,
        "strong": n >= min_repeats,
        "items": df.loc[df["pattern"] == code, "item"].tolist(),
        "histogram": dict(counts),
    }


RESEARCH_PATHS = {
    "A": ("Verbal concepts in Qwen3 lock before FLUX can use them visually",
          "Patch the layer-L*_logit 'dog' residual into a hard prompt; measure image CLIP"),
    "D": ("Rare tokens only form category features, not identity features",
          "Compare nearest neighbours in residual space for hard vs easy at the final layer"),
    "G": ("Raw logit lens misleads on this encoder; J-transport is required",
          "Quantify the disagreement rate; ablate the diagonal transport"),
    "C": ("Controls work; need harder multi-hop prompts to surface pattern E",
          "Build an 'animal that spins webs' style prompt set on the encoder"),
}


def _none_if_nan(x):
    try:
        return None if x is None or pd.isna(x) else int(x)
    except (TypeError, ValueError):
        return None
