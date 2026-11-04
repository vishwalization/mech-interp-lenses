"""
lenses — mechanistic-interpretability lenses for the Qwen3 text encoder inside
FLUX.2 Klein 4B.

Three complementary read-outs of the same residual stream:

* :mod:`lenses.logit_lens`      – unembed a layer's residual → words
* :mod:`lenses.diffusion_lens`  – feed a layer's residual to the DiT → image
* :mod:`lenses.j_lens`          – transport a layer into the final basis, then unembed

plus :mod:`lenses.token_cards`, which fuses the three emergence depths
(L*_logit, L*_diff, L*_J) per token and classifies the pattern (A–G).

Everything that touches the model goes through :mod:`lenses.model` and
:mod:`lenses.activations`, which handle the one engineering hazard of this
setup: ``enable_sequential_cpu_offload()`` parks weights on the ``meta``
device and re-uses GPU memory aggressively, so activations are always
snapshotted to CPU float32 and weights are materialised through accelerate's
public API before use.

Sub-modules are imported lazily so that the CPU-only pieces
(:mod:`lenses.token_cards`, :mod:`lenses.config`) work on a machine without
torch installed.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_LAZY = {
    "HiddenStates": "lenses.activations",
    "capture_hidden_states": "lenses.activations",
    "fingerprint": "lenses.activations",
    "Unembed": "lenses.unembed",
    "rms_norm": "lenses.unembed",
    "parse_item": "lenses.tokens",
    "find_item_position": "lenses.tokens",
    "target_vocab_ids": "lenses.tokens",
    "resolve_targets": "lenses.tokens",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'lenses' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
