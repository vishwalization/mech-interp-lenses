"""
Residual-stream capture that survives ``enable_sequential_cpu_offload()``.

Why this module exists
----------------------
Under sequential offload, GPU activations returned by
``text_encoder(..., output_hidden_states=True)`` are *views into memory that
later kernels are free to reuse*. Keeping them on the GPU and reading them
across cells/calls produced the classic symptom: early layers decode fine on
the first pass, later layers come out as near-uniform softmax garbage, and a
second pass is garbage everywhere. The fix that actually held up:

1. ``detach().float().cpu().contiguous().clone()`` every layer immediately;
2. never pass a view of the cache into an ``nn.Module`` (RMSNorm kernels can
   write through aliased storage) — do the maths functionally on a copy.

:func:`capture_hidden_states` does (1); :mod:`lenses.unembed` does (2).
:func:`fingerprint` is a cheap checksum to prove the cache is not mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import torch


@dataclass
class HiddenStates:
    """CPU float32 snapshots of every layer of the text encoder.

    Attributes:
        layers: tuple of ``(B, S, D)`` float32 CPU tensors, length ``n_layers``.
            Index 0 is the token embedding; indices ``1..n-2`` are pre-norm
            residuals *entering* each block; the **last** index is already
            post-final-RMSNorm (HF ``Qwen*Model.forward`` appends
            ``self.norm(hidden_states)`` after the block loop). Lenses must
            therefore apply the final norm to every layer *except* the last.
        input_ids: ``(B, S)`` CPU long tensor.
        attention_mask: ``(B, S)`` CPU long tensor.
        chat_template: whether the FLUX chat template was applied.
        prompts: the raw prompt strings, in batch order.
    """

    layers: Tuple[torch.Tensor, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chat_template: bool = True
    prompts: Sequence[str] = field(default_factory=tuple)

    # -- convenience ---------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def last(self) -> int:
        """Index of the post-final-norm entry."""
        return len(self.layers) - 1

    @property
    def batch_size(self) -> int:
        return int(self.layers[0].shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.layers[0].shape[1])

    @property
    def d_model(self) -> int:
        return int(self.layers[0].shape[2])

    def vector(self, layer: int, b: int, pos: int) -> torch.Tensor:
        """A *copy* of the residual at ``(layer, batch b, position pos)``."""
        return self.layers[layer][b, pos].float().clone()

    def tokens(self, b: int, tokenizer, n: int | None = None) -> list[str]:
        ids = self.input_ids[b] if n is None else self.input_ids[b][:n]
        return tokenizer.convert_ids_to_tokens(ids)


def _snapshot(hidden_states) -> Tuple[torch.Tensor, ...]:
    return tuple(h.detach().float().cpu().contiguous().clone() for h in hidden_states)


def apply_flux_chat_template(tokenizer, prompt: str) -> str:
    """The exact template FLUX.2 Klein uses before its Qwen3 encoder."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def capture_hidden_states(
    text_encoder,
    tokenizer,
    prompts: Sequence[str] | str,
    *,
    device: str = "cuda",
    chat_template: bool = True,
    max_sequence_length: int = 512,
) -> HiddenStates:
    """Run the text encoder once for a batch of prompts and snapshot every layer.

    Args:
        text_encoder: ``pipe.text_encoder`` (Qwen3ForCausalLM).
        tokenizer: ``pipe.tokenizer``.
        prompts: one prompt or a list.
        device: execution device for the forward pass.
        chat_template: ``True`` reproduces FLUX's own text path
            (``<|im_start|>user … <|im_end|> … <think>\\n\\n</think>``) padded to
            ``max_sequence_length``; ``False`` tokenises the raw string with
            dynamic padding — useful to measure how much the template shifts
            the residual stream (pattern **F** in the token cards).
        max_sequence_length: FLUX uses 512.

    Returns:
        :class:`HiddenStates` with everything on CPU in float32.
    """
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)

    if chat_template:
        ids, masks = [], []
        for p in prompts:
            enc = tokenizer(
                apply_flux_chat_template(tokenizer, p),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            ids.append(enc["input_ids"])
            masks.append(enc["attention_mask"])
        input_ids = torch.cat(ids, dim=0)
        attention_mask = torch.cat(masks, dim=0)
    else:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    out = text_encoder(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    return HiddenStates(
        layers=_snapshot(out.hidden_states),
        input_ids=input_ids.cpu(),
        attention_mask=attention_mask.cpu(),
        chat_template=chat_template,
        prompts=tuple(prompts),
    )


def fingerprint(hs: HiddenStates, b: int = 0, pos: int = 0) -> Dict[str, float]:
    """Cheap checksum of the cache at one position.

    Call before and after a lens pass; if the two dicts differ, something is
    still writing into the cached activations.
    """
    h0 = hs.layers[0][b, pos].float()
    hL = hs.layers[-1][b, pos].float()
    return {
        "n_layers": hs.n_layers,
        "L0_sum": round(float(h0.sum()), 4),
        "L0_std": round(float(h0.std()), 4),
        "Llast_sum": round(float(hL.sum()), 4),
        "Llast_std": round(float(hL.std()), 4),
    }
