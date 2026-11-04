"""
Pipeline loading for FLUX.2 Klein 4B.

The text encoder inside the pipeline is a ``Qwen3ForCausalLM`` (36 layers,
d_model = 2560, vocab = 151 936). FLUX builds its prompt embedding by stacking
three of its hidden-state layers, which is what makes the diffusion lens
possible without touching the DiT.

Memory footprint on a 16 GB card is only workable with
``enable_sequential_cpu_offload()``. That setting is the source of every
"garbage after re-run" bug this project hit — see ``docs/notes/offload-safe-activations.md``
and the helpers in :mod:`lenses.activations` / :mod:`lenses.unembed` that
work around it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Must precede CUDA initialisation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def hf_login(token: Optional[str] = None) -> bool:
    """Log in to the Hugging Face Hub if a token is available.

    Resolution order: explicit argument → ``HF_TOKEN`` env var → Kaggle secret
    ``HF_TOKEN`` (when running on Kaggle). Returns ``True`` if a login happened.
    """
    from huggingface_hub import login

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        try:  # Kaggle notebooks expose secrets through this client
            from kaggle_secrets import UserSecretsClient  # type: ignore

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        login(token=token)
        return True
    return False


@dataclass
class LoadedPipeline:
    """A loaded FLUX.2 Klein pipeline plus the handles the lenses need."""

    pipe: object
    device: str

    @property
    def text_encoder(self):
        return self.pipe.text_encoder

    @property
    def tokenizer(self):
        return self.pipe.tokenizer


def load_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
    sequential_cpu_offload: bool = True,
    vae_tiling: bool = True,
    login: bool = True,
) -> LoadedPipeline:
    """Load ``Flux2KleinPipeline`` with the memory settings used throughout.

    Args:
        model_id: Hub id of the pipeline.
        dtype: weights dtype (bf16 is the only one tested).
        sequential_cpu_offload: keep weights on CPU/meta and stream them to the
            GPU per sub-module. Required on ≤16 GB GPUs. When ``True``, never
            read ``module.weight`` directly — use :func:`lenses.unembed.materialize_weight`.
        vae_tiling: cheap decode-time VRAM savings.
        login: attempt :func:`hf_login` first (gated model).
    """
    if login:
        hf_login()

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if vae_tiling:
        pipe.vae.enable_tiling()
    if sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LoadedPipeline(pipe=pipe, device=device)


def free_cuda() -> None:
    """``gc.collect(); torch.cuda.empty_cache()`` — call between image generations."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
