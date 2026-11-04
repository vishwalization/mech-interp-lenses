"""
Unembedding (``W_U``) and final RMSNorm as plain CPU float32 tensors.

Reading ``module.weight`` on an offloaded component returns a **meta** tensor
(the real data lives in accelerate's ``weights_map``) and ``.cpu()`` on it
raises ``NotImplementedError: Cannot copy out of meta tensor; no data!``.
:func:`materialize_weight` goes through accelerate's public
``align_module_device`` to on-load the real parameter, snapshot it, and
re-offload on exit.

Everything downstream is *functional*: :func:`rms_norm` is a pure function and
:class:`Unembed` holds only tensors, so no hook, no in-place kernel, and no
autograd graph can touch the cached activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def materialize_weight(module: torch.nn.Module, name: str = "weight") -> torch.Tensor:
    """CPU float32 clone of ``module.<name>``, correct whether or not it is offloaded."""
    from accelerate.utils import align_module_device, has_offloaded_params

    if has_offloaded_params(module):
        with align_module_device(module, execution_device="cpu"):
            p = getattr(module, name)
            return p.detach().float().cpu().contiguous().clone()
    p = getattr(module, name)
    return p.detach().float().cpu().contiguous().clone()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen-style RMSNorm with no module, no hooks, no in-place writes.

    ``y = x / sqrt(mean(x²) + eps) * weight`` computed in float32.
    """
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return x32 * torch.rsqrt(var + eps) * weight.float()


def _find_final_norm(text_encoder) -> torch.nn.Module:
    norm = getattr(text_encoder, "norm", None)
    if norm is None:
        inner = getattr(text_encoder, "model", None)
        norm = getattr(inner, "norm", None) if inner is not None else None
    if norm is None:
        raise RuntimeError("Could not find the final RMSNorm on the text encoder")
    return norm


@dataclass
class Unembed:
    """Cached unembedding matrix + final norm for logit-lens style read-outs.

    Attributes:
        W_U: ``(vocab, d_model)`` float32 CPU. Logits are ``v @ W_U.T``.
        norm_weight: ``(d_model,)`` float32 CPU, final RMSNorm scale.
        norm_eps: RMSNorm epsilon.
        tokenizer: used only for decoding ids in :meth:`topk`.
        tied: ``True`` if there was no LM head and the input embedding was used.
    """

    W_U: torch.Tensor
    norm_weight: torch.Tensor
    norm_eps: float
    tokenizer: object = None
    tied: bool = False

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_text_encoder(cls, text_encoder, tokenizer=None, verbose: bool = True) -> "Unembed":
        out_emb = text_encoder.get_output_embeddings()
        if out_emb is not None:
            W_U, tied = materialize_weight(out_emb), False
        else:  # tied embeddings — standard logit-lens fallback
            W_U, tied = materialize_weight(text_encoder.get_input_embeddings()), True

        norm_mod = _find_final_norm(text_encoder)
        norm_weight = materialize_weight(norm_mod)
        norm_eps = float(getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6)))

        if verbose:
            head = "tied input embeddings" if tied else "LM head"
            print(
                f"[unembed] {head} W_U={tuple(W_U.shape)}  "
                f"final_norm={norm_mod.__class__.__name__} eps={norm_eps}"
            )
        return cls(W_U=W_U, norm_weight=norm_weight, norm_eps=norm_eps, tokenizer=tokenizer, tied=tied)

    # -- maths -----------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(self.W_U.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_U.shape[1])

    @torch.inference_mode()
    def logits(self, v: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """``(d_model,)`` residual → ``(vocab,)`` logits. ``v`` is never modified."""
        v = v.float()
        if apply_norm:
            v = rms_norm(v, self.norm_weight, self.norm_eps)
        return v @ self.W_U.T

    @torch.inference_mode()
    def probs(self, v: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        return self.logits(v, apply_norm).softmax(dim=-1)

    @torch.inference_mode()
    def topk(self, v: torch.Tensor, k: int = 5, apply_norm: bool = True, ndigits: int = 3) -> List[Tuple[str, float]]:
        """Top-``k`` ``(decoded_token, prob)`` pairs for one residual vector."""
        if self.tokenizer is None:
            raise ValueError("Unembed.topk needs a tokenizer")
        p, idx = self.probs(v, apply_norm).topk(k)
        return [
            (self.tokenizer.decode([int(i)]).strip(), round(float(pp), ndigits))
            for pp, i in zip(p, idx)
        ]

    @torch.inference_mode()
    def target_mass_and_rank(
        self, v: torch.Tensor, vocab_ids: List[int], apply_norm: bool = True
    ) -> Tuple[float, Optional[int]]:
        """Probability mass on a set of vocab ids and the rank of the best one (1 = top)."""
        if not vocab_ids:
            return 0.0, None
        probs = self.probs(v, apply_norm)
        ids = torch.as_tensor(vocab_ids, dtype=torch.long)
        mass = float(probs[ids].sum())
        best = ids[probs[ids].argmax()]
        rank = int((probs > probs[best]).sum().item()) + 1
        return mass, rank
