"""
Prompt → item → token-position → target-vocab plumbing shared by all lenses.

Every lens reads the residual stream *at the position of the item token*
("dog" in ``"a dog over a white table"``). Two practical wrinkles:

* Rare items are split into several BPE pieces (``Ġax ol ot l``); we read at
  the **last** piece by default, where the full-word identity is assembled.
* "Is the target word present?" must be computed against *every* vocab entry
  that decodes to that word (``apple``, ``Ġapple``, ``Apple`` …), otherwise
  probability mass is split across case/space variants.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch

_STOP = (
    "over|on|above|under|beside|near|in|at|with|by|inside|behind|between|"
    "parked|standing|sitting|lying|swimming|flying|floating|walking|running|resting|placed|sits|stands"
)
_ARTICLE_RE = re.compile(rf"^\s*(?:a|an|the)\s+(?P<item>.+?)\s+(?:{_STOP})\b", re.IGNORECASE)


def parse_item(prompt: str) -> str:
    """Extract the subject noun phrase from a templated prompt.

    ``"a quokka over a white table"`` → ``"quokka"``; ``"A spaceship parked near …"``
    → ``"spaceship"``. The item runs from the article to the first preposition or
    common verb, so multi-word items (``"a golden retriever over …"``) survive.
    Falls back to the first non-article word if nothing matches. (A naive
    ``replace("a ", "")`` turns *quokka over* into *quokkover* — hence the regex.)
    For free-form prompts, list ``items:`` explicitly in the config instead.
    """
    m = _ARTICLE_RE.match(prompt)
    if m:
        return m.group("item").strip()
    words = [w for w in re.findall(r"[A-Za-z\-]+", prompt) if w.lower() not in {"a", "an", "the"}]
    return words[0] if words else prompt.strip()


def item_token_span(tokenizer, text: str, item: str) -> Optional[Tuple[int, int]]:
    """``[start, end)`` token indices covering ``item`` inside ``text``.

    Uses the fast tokenizer's offset mapping so it works for any BPE split.
    Returns ``None`` if the item string is not found in ``text``.
    """
    lo = text.lower().find(item.lower())
    if lo < 0:
        return None
    hi = lo + len(item)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    idx = [i for i, (s, e) in enumerate(offsets) if e > lo and s < hi and e > s]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def find_item_position(
    tokenizer,
    input_ids_row: "torch.Tensor",
    item: str,
    *,
    which: str = "last",
    max_scan: int = 64,
) -> int:
    """Index of the item token inside one (possibly chat-templated) id row.

    Strategy: decode the first ``max_scan`` ids back to text, locate ``item``
    with offsets, and map back. ``which`` selects ``"first"`` or ``"last"``
    sub-token for multi-piece items.

    Raises ``ValueError`` if the item cannot be located — better than silently
    reading the wrong position.
    """
    ids = input_ids_row[:max_scan].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    item_l = item.lower()

    # 1) whole-word match on a single token (Ġdog / dog)
    for i, t in enumerate(toks):
        if t.replace("Ġ", "").replace("▁", "").lower() == item_l:
            return i

    # 2) multi-piece: greedily concatenate pieces until they spell the item
    for i in range(len(toks)):
        if not toks[i].startswith(("Ġ", "▁")):
            continue
        acc = ""
        for j in range(i, min(i + 12, len(toks))):
            acc += toks[j].replace("Ġ", "").replace("▁", "")
            if acc.lower() == item_l:
                return j if which == "last" else i
            if not item_l.startswith(acc.lower()):
                break

    # 3) offsets-based fallback on the decoded text
    text = tokenizer.decode(ids)
    span = item_token_span(tokenizer, text, item)
    if span is not None:
        return span[1] - 1 if which == "last" else span[0]

    raise ValueError(f"Could not locate item {item!r} in tokens {toks[:20]}…")


def find_item_positions(
    tokenizer, input_ids: "torch.Tensor", items: Sequence[str], which: str = "last"
) -> Dict[int, int]:
    """Vectorised :func:`find_item_position` → ``{batch_index: position}``."""
    return {b: find_item_position(tokenizer, input_ids[b], item, which=which) for b, item in enumerate(items)}


# ---------------------------------------------------------------------------
# target vocab ids
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _decoded_vocab(tok_key: int, vocab_size: int, tokenizer=None) -> Tuple[str, ...]:
    # tok_key is id(tokenizer) so the cache is per tokenizer instance.
    return tuple(tokenizer.decode([i]).strip().lower() for i in range(vocab_size))


def target_vocab_ids(tokenizer, word: str) -> List[int]:
    """Every vocab id whose decoded string equals ``word`` (case/space-insensitive).

    First call decodes the whole vocabulary (~150k ids, a few seconds) and is
    cached per tokenizer; later calls are instant.
    """
    decoded = _decoded_vocab(id(tokenizer), len(tokenizer), tokenizer)
    w = word.lower()
    return [i for i, s in enumerate(decoded) if s == w]


def resolve_targets(
    tokenizer,
    items: Sequence[str],
    fallback: Optional[str] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, bool]]:
    """Map each item to its target vocab ids.

    Items that have no single-token form (``axolotl``, ``jabuticaba``) get the
    ids of ``fallback`` (e.g. ``"animal"`` or ``"fruit"``) so their curve
    measures *category* emergence instead. Returns ``(vocab_ids, own_token)``
    where ``own_token[i]`` says whether the item had its own token.
    """
    vocab_ids: Dict[int, List[int]] = {}
    own: Dict[int, bool] = {}
    fb_ids = target_vocab_ids(tokenizer, fallback) if fallback else []
    for i, item in enumerate(items):
        ids = target_vocab_ids(tokenizer, item)
        own[i] = bool(ids)
        vocab_ids[i] = ids if ids else list(fb_ids)
    return vocab_ids, own
