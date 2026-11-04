"""
Diffusion lens (Toker et al., ACL 2024) for FLUX.2 Klein.

Question answered: *if I hand an intermediate text-encoder layer to the image
model, what picture do I get?*

FLUX.2 Klein builds its prompt embedding by stacking **three** Qwen3 hidden
states along the channel axis — natively ``(i, i+9, i+18)`` — and flattening to
``(B, S, 3·D)``. :func:`make_prompt_embeds` reproduces that stacking exactly
from cached :class:`HiddenStates`, so we can feed *any* triple:

* ``(L, L, L)`` — a true single-layer diffusion lens: the image is driven by
  depth ``L`` alone (:func:`sample_depths`).
* ``(i, i+9, i+18)`` — the pipeline's native stack, swept over ``i``
  (:func:`native_triples`).
* ``(c−k, c, c+k)`` — "palindrome" combos around a centre layer
  (:func:`palindrome_triples`).

Running the encoder once and re-slicing its hidden states is *N× cheaper* than
calling ``pipe._get_qwen3_prompt_embeds`` once per layer combination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .activations import HiddenStates
from .model import free_cuda

LayerTriple = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# layer selections
# ---------------------------------------------------------------------------

def sample_depths(n_layers: int, stride: int = 3) -> List[int]:
    """``0, stride, 2·stride, …`` plus the last index, for the pure ``(L,L,L)`` lens."""
    depths = list(range(0, n_layers, max(1, stride)))
    if depths[-1] != n_layers - 1:
        depths.append(n_layers - 1)
    return depths


def native_triples(n_layers: int, spacing: int = 9) -> List[LayerTriple]:
    """FLUX's own ``(i, i+9, i+18)`` stack for every ``i`` that fits."""
    return [
        (i, i + spacing, i + 2 * spacing)
        for i in range(n_layers)
        if i + 2 * spacing < n_layers
    ]


def palindrome_triples(center: int = 18, n_layers: int = 37, max_offset: int = 9) -> List[LayerTriple]:
    """``(center−k, center, center+k)`` for ``k = max_offset … 1`` — squeeze toward the centre."""
    return [
        (center - k, center, center + k)
        for k in range(max_offset, 0, -1)
        if center - k >= 0 and center + k < n_layers
    ]


# ---------------------------------------------------------------------------
# embeds + generation
# ---------------------------------------------------------------------------

def make_prompt_embeds(
    hs: HiddenStates,
    layers: Sequence[int],
    batch_index: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Stack the chosen hidden-state layers exactly like ``_get_qwen3_prompt_embeds``.

    Returns ``(B or 1, S, len(layers)·D)`` in ``dtype`` (CPU; the pipeline
    moves it under offload).
    """
    tensors = [hs.layers[k] for k in layers]
    if batch_index is not None:
        tensors = [t[batch_index : batch_index + 1] for t in tensors]
    out = torch.stack(tensors, dim=1).to(dtype)  # (B, n_ch, S, D)
    b, n_ch, seq, hid = out.shape
    return out.permute(0, 2, 1, 3).reshape(b, seq, n_ch * hid)


@torch.inference_mode()
def generate(
    pipe,
    prompt_embeds: torch.Tensor,
    *,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
):
    """One image from pre-built prompt embeds (Klein is a few-step model: 4 steps is fine)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(
        prompt_embeds=prompt_embeds,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=gen,
    ).images[0]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def run_pure_lens(
    pipe,
    hs: HiddenStates,
    items: Sequence[str],
    depths: Sequence[int],
    out_dir: Path,
    *,
    seed: int = 42,
    gen_kwargs: Optional[dict] = None,
    log=print,
) -> Dict[Tuple[int, int], Path]:
    """Pure ``(L,L,L)`` lens: one image per (prompt, depth). Skips files that exist."""
    gen_kwargs = gen_kwargs or {}
    out_dir = Path(out_dir)
    results: Dict[Tuple[int, int], Path] = {}
    for b, item in enumerate(items):
        sub = out_dir / "pure_LLL" / f"{b:02d}_{_slug(item)}"
        sub.mkdir(parents=True, exist_ok=True)
        log(f"[{b}] {hs.prompts[b] if hs.prompts else item}")
        for L in depths:
            path = sub / f"L{L:02d}.png"
            results[(b, L)] = path
            if path.exists():
                continue
            embeds = make_prompt_embeds(hs, (L, L, L), batch_index=b)
            img = generate(pipe, embeds, seed=seed, **gen_kwargs)
            img.save(path)
            del embeds, img
            free_cuda()
            log(f"    L={L:02d} -> {path}")
    return results


def run_combo_sweep(
    pipe,
    hs: HiddenStates,
    items: Sequence[str],
    combos: Iterable[LayerTriple],
    out_dir: Path,
    *,
    tag: str = "triples",
    seed: int = 42,
    gen_kwargs: Optional[dict] = None,
    log=print,
) -> Dict[Tuple[int, LayerTriple], Path]:
    """Arbitrary layer-combination sweep (native triples, palindromes, …)."""
    gen_kwargs = gen_kwargs or {}
    out_dir = Path(out_dir)
    combos = list(combos)
    results: Dict[Tuple[int, LayerTriple], Path] = {}
    for b, item in enumerate(items):
        sub = out_dir / tag / f"{b:02d}_{_slug(item)}"
        sub.mkdir(parents=True, exist_ok=True)
        for layers in combos:
            path = sub / ("layers_" + "_".join(f"{k:02d}" for k in layers) + ".png")
            results[(b, tuple(layers))] = path
            if path.exists():
                continue
            embeds = make_prompt_embeds(hs, layers, batch_index=b)
            img = generate(pipe, embeds, seed=seed, **gen_kwargs)
            img.save(path)
            del embeds, img
            free_cuda()
            log(f"[{b}] {tuple(layers)} -> {path}")
    return results


# ---------------------------------------------------------------------------
# optional CLIP proxy for "is the object there yet?"
# ---------------------------------------------------------------------------

def clip_scores(
    results: Dict[Tuple[int, int], Path],
    items: Sequence[str],
    depths: Sequence[int],
    template: str = "a photo of a {item} on a white table",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> Dict[int, Dict[int, float]]:
    """CLIP cosine(image, text) per (item, depth). Requires ``open_clip_torch``."""
    import open_clip  # noqa: WPS433 (optional dependency)
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model = model.eval()
    scores: Dict[int, Dict[int, float]] = {b: {} for b in range(len(items))}
    with torch.inference_mode():
        for b, item in enumerate(items):
            t = model.encode_text(tok([template.format(item=item)]))
            t = t / t.norm(dim=-1, keepdim=True)
            for L in depths:
                p = results.get((b, L))
                if p is None or not Path(p).exists():
                    continue
                im = preprocess(Image.open(p)).unsqueeze(0)
                f = model.encode_image(im)
                f = f / f.norm(dim=-1, keepdim=True)
                scores[b][L] = float((f @ t.T).squeeze())
    return scores


def lstar_from_scores(series: Dict[int, float], depths: Sequence[int], threshold: Optional[float] = None) -> Optional[int]:
    """Earliest depth crossing ``threshold`` (or the arg-max depth if ``None``)."""
    if not series:
        return None
    if threshold is None:
        return max(series, key=series.get)
    for L in depths:
        if series.get(L, -1e9) >= threshold:
            return L
    return None


def write_card_partial(
    out_dir: Path,
    *,
    items: Sequence[str],
    prompts: Sequence[str],
    depths: Sequence[int],
    seed: int,
    lstar_human: Dict[str, Optional[int]],
    clip: Optional[Dict[int, Dict[int, float]]] = None,
    model: str = "FLUX.2-klein-4B",
) -> Path:
    """``diffusion_card_partial.json`` consumed by :mod:`lenses.token_cards`."""
    payload = {
        "method": "diffusion_lens",
        "model": model,
        "seed": seed,
        "depths": list(depths),
        "prompts": list(prompts),
        "items": list(items),
        "Lstar_human": lstar_human,
        "clip_scores": {str(k): v for k, v in (clip or {}).items()},
    }
    out = Path(out_dir) / "diffusion_card_partial.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")
