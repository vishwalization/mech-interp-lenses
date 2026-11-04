"""
Jacobian lens ("J-lens") on the Qwen3 text encoder.

Question answered: *which words is this activation **poised** to let the model
say* — not merely what it decodes to right now?

Anthropic's recipe: fit an average Jacobian ``J_ℓ = E[∂h_final / ∂h_ℓ]`` on
generic text, map ``h ↦ J_ℓ h``, then unembed. Two paths are provided:

* **Path A — official** (:func:`fit_official_jlens`): the ``jlens`` package
  from ``anthropics/jacobian-lens``. It needs real forward+backward passes, so
  it is run on a *separate* non-offloaded copy of the encoder (an offloaded
  model's parameters live on ``meta`` and autograd cannot touch them).
* **Path B — DIY diagonal transport** (:class:`DiagonalTransport`): a
  per-dimension least-squares scale ``h_final ≈ h_ℓ ⊙ s_ℓ`` fitted on the
  prompts' own real tokens. Not a Jacobian, but it answers the same practical
  question and runs on CPU in milliseconds. Use it for exploration; switch to
  Path A for paper-grade claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .activations import HiddenStates
from .logit_lens import TopK, first_emergence_layer, rank_curve
from .unembed import Unembed


# ---------------------------------------------------------------------------
# Path B — DIY diagonal transport
# ---------------------------------------------------------------------------

@dataclass
class DiagonalTransport:
    """``h_final ≈ h_L ⊙ scale[L]`` with identity at the final layer.

    Callable as ``transport(v, L)`` so it plugs into
    :func:`lenses.logit_lens.rank_curve`.
    """

    scales: Dict[int, torch.Tensor]

    def __call__(self, v: torch.Tensor, L: int) -> torch.Tensor:
        return v.float() * self.scales[L].float()

    @classmethod
    @torch.inference_mode()
    def fit(cls, hs: HiddenStates, min_pos: int = 1) -> "DiagonalTransport":
        """Per-layer, per-dimension ridge-stable scale from masked real tokens.

        ``s_d = Σ h_d·y_d / (Σ h_d² + 1e-8)`` over every unmasked position
        (skipping the first ``min_pos`` specials) and every prompt in the batch.
        """
        last = hs.layers[-1]  # already post-final-norm
        B, S, D = last.shape
        mask = hs.attention_mask.bool().clone()
        mask[:, :min_pos] = False

        scales: Dict[int, torch.Tensor] = {}
        for L in range(hs.n_layers - 1):
            h = hs.layers[L]
            num = torch.zeros(D)
            den = torch.full((D,), 1e-8)
            for b in range(B):
                m = mask[b]
                if m.sum() == 0:
                    continue
                x, y = h[b, m].float(), last[b, m].float()
                num += (x * y).sum(0)
                den += (x * x).sum(0)
            scales[L] = (num / den).contiguous()
        scales[hs.last] = torch.ones(D)
        return cls(scales=scales)

    def save(self, path: Path) -> None:
        torch.save({int(k): v for k, v in self.scales.items()}, path)

    @classmethod
    def load(cls, path: Path) -> "DiagonalTransport":
        return cls(scales=torch.load(path, map_location="cpu", weights_only=True))


@torch.inference_mode()
def j_lens(
    hs: HiddenStates,
    unembed: Unembed,
    transport: DiagonalTransport,
    b: int,
    pos: int,
    topk: int = 5,
) -> Dict[int, TopK]:
    """Top-``k`` after transporting each layer into the final basis."""
    out: Dict[int, TopK] = {}
    for L in range(hs.n_layers):
        v = transport(hs.vector(L, b, pos), L)
        out[L] = unembed.topk(v, k=topk, apply_norm=(L != hs.last), ndigits=4)
    return out


def j_vs_logit_curves(
    hs: HiddenStates,
    unembed: Unembed,
    transport: DiagonalTransport,
    positions: Dict[int, int],
    vocab_ids: Dict[int, List[int]],
) -> Dict[str, Dict[int, list]]:
    """Probability-mass and rank curves under both lenses for every prompt."""
    out = {"j_prob": {}, "j_rank": {}, "logit_prob": {}, "logit_rank": {}}
    for b, pos in positions.items():
        out["j_prob"][b], out["j_rank"][b] = rank_curve(hs, unembed, b, pos, vocab_ids[b], transport=transport)
        out["logit_prob"][b], out["logit_rank"][b] = rank_curve(hs, unembed, b, pos, vocab_ids[b])
    return out


def lstar_table(
    curves: Dict[str, Dict[int, list]],
    items: Sequence[str],
    *,
    easy_count: int,
    prob_thresh: float = 0.05,
    rank_thresh: int = 50,
) -> List[dict]:
    """Rows with ``Lstar_J`` and ``Lstar_logit`` for the token cards."""
    rows = []
    for b, item in enumerate(items):
        Lj = first_emergence_layer(curves["j_prob"][b], curves["j_rank"][b], prob_thresh=prob_thresh, rank_thresh=rank_thresh)
        Ll = first_emergence_layer(curves["logit_prob"][b], curves["logit_rank"][b], prob_thresh=prob_thresh, rank_thresh=rank_thresh)
        rows.append(
            {
                "idx": b,
                "item": item,
                "bucket": "easy" if b < easy_count else "hard",
                "Lstar_J_diy": Lj,
                "Lstar_logit": Ll,
                "final_prob_J": curves["j_prob"][b][-1],
                "final_prob_logit": curves["logit_prob"][b][-1],
                "final_rank_J": curves["j_rank"][b][-1],
            }
        )
    return rows


def write_card_partial(
    out_dir: Path,
    *,
    prompts: Sequence[str],
    items: Sequence[str],
    positions: Dict[int, int],
    rows: List[dict],
    curves: Dict[str, Dict[int, list]],
    model: str = "FLUX.2-klein-4B / Qwen3 text encoder",
) -> Path:
    payload = {
        "method": "j_lens_diy_diagonal_transport",
        "model": model,
        "prompts": list(prompts),
        "items": list(items),
        "b_pos": {str(k): v for k, v in positions.items()},
        "rows": rows,
        "diy_prob": {str(k): v for k, v in curves["j_prob"].items()},
        "diy_rank": {str(k): v for k, v in curves["j_rank"].items()},
        "raw_prob": {str(k): v for k, v in curves["logit_prob"].items()},
        "raw_rank": {str(k): v for k, v in curves["logit_rank"].items()},
    }
    out = Path(out_dir) / "j_lens_card_partial.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


# ---------------------------------------------------------------------------
# Path A — official Anthropic jlens
# ---------------------------------------------------------------------------

def fit_official_jlens(
    tokenizer,
    out_dir: Path,
    *,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    subfolder: str = "text_encoder",
    n_prompts: int = 8,
    max_seq_len: int = 128,
    dim_batch: int = 4,
    min_tokens: int = 32,
    device: Optional[str] = None,
    log=print,
):
    """Fit Anthropic's Jacobian lens on a dedicated, non-offloaded Qwen3 copy.

    Install first::

        pip install "git+https://github.com/anthropics/jacobian-lens.git#egg=jlens"

    Returns ``(lens, lens_model)`` or raises. A checkpoint is written every
    prompt so an interrupted fit can resume; an empty/poisoned checkpoint from
    a failed run is deleted automatically.

    Memory: the full encoder (~8 GB bf16) plus autograd — needs a ≥24 GB GPU
    for comfort. Free the offloaded pipeline's GPU scraps before calling.
    """
    import jlens  # type: ignore
    import transformers

    from .model import free_cuda

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / "jacobian_lens.pt"
    ckpt_path = out_dir / "ckpt.pt"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    free_cuda()
    log(f"[jlens] loading dedicated CausalLM {model_id}/{subfolder} on {device}")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, subfolder=subfolder, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()

    try:
        lens_model = jlens.from_hf(hf, tokenizer)
    except Exception as e:  # tokenizer mismatch → load the pipeline's own
        log(f"[jlens] from_hf with pipeline tokenizer failed ({e}); loading tokenizer from hub")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        lens_model = jlens.from_hf(hf, tokenizer)
    log(f"[jlens] {lens_model}")

    if lens_path.exists():
        for loader in ("load", "from_pretrained"):
            fn = getattr(jlens.JacobianLens, loader, None)
            if fn is None:
                continue
            try:
                lens = fn(str(lens_path)) if loader == "load" else fn(str(lens_path.parent), filename=lens_path.name)
                log(f"[jlens] loaded existing lens from {lens_path}")
                return lens, lens_model
            except Exception as e:
                log(f"[jlens] {loader} failed: {e}")

    # fitting prompts — jlens skips the first 16 + last positions, so they must be long
    try:
        from jlens.examples import load_wikitext_prompts  # type: ignore

        prompts = load_wikitext_prompts(n_prompts, min_chars=600)
    except Exception as e:
        log(f"[jlens] WikiText unavailable ({type(e).__name__}); using built-in paragraphs")
        prompts = _FALLBACK_FIT_PROMPTS[:n_prompts]

    def n_tok(t: str) -> int:
        return int(lens_model.encode(t, max_length=max_seq_len).shape[1])

    prompts = [p for p in prompts if n_tok(p) > min_tokens]
    if not prompts:
        raise ValueError(f"all fit prompts tokenised to ≤{min_tokens} tokens")
    log(f"[jlens] fitting on {len(prompts)} prompts, max_seq_len={max_seq_len}, dim_batch={dim_batch}")

    resume = False
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if int(state.get("n_done", 0)) == 0:
                ckpt_path.unlink()
                log("[jlens] deleted empty checkpoint from a failed run")
            else:
                resume = True
        except Exception:
            ckpt_path.unlink(missing_ok=True)

    lens = jlens.fit(
        lens_model,
        prompts=prompts,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=1,
        max_seq_len=max_seq_len,
        dim_batch=dim_batch,
        resume=resume,
    )
    lens.save(str(lens_path))
    log(f"[jlens] saved {lens_path}")
    return lens, lens_model


def apply_official_jlens(lens, lens_model, tokenizer, prompt: str, positions=(-2,), topk: int = 5) -> Dict[int, List[str]]:
    """Top-``k`` tokens per layer from a fitted official lens for one prompt."""
    lens_logits, _model_logits, _ = lens.apply(lens_model, prompt, positions=list(positions))
    return {
        int(layer): [tokenizer.decode([t]) for t in logits[0].topk(topk).indices]
        for layer, logits in sorted(lens_logits.items())
    }


_FALLBACK_FIT_PROMPTS = [
    "A bright red apple sits on a wooden table next to a ceramic bowl, a folded blue napkin, and a small "
    "glass of water. Morning light comes through the kitchen window and makes the fruit look shiny. Someone "
    "has already sliced bread and left the knife on the board.",
    "The capital of France is Paris, a city famous for museums, cafes, bridges over the Seine, and the Eiffel "
    "Tower at night. Tourists walk along the river, take photographs, and visit bookshops on narrow streets "
    "filled with bakeries and flower stalls.",
    "Once upon a time there was a curious fox who wandered through an ancient forest under silver moonlight. "
    "Owls called from the trees, streams ran over smooth stones, and the fox followed a faint path until it "
    "found a clearing with soft moss and wildflowers.",
    "In machine learning, a residual stream is the running hidden state that each transformer layer reads "
    "from and writes back into. Early layers often encode local syntax, while later layers mix in more "
    "semantic and task-specific information useful for prediction.",
    "The animal that spins webs is the spider, which builds sticky traps to catch flying insects for food. "
    "Many species live in gardens, forests, and houses. Their silk is strong for its weight, and young spiders "
    "sometimes travel by ballooning on long threads of silk.",
    "Scientists studied a rare marsupial that lives near coastal scrub. It eats leaves at night, rests during "
    "the day, and avoids open ground when predators are nearby. Field notes describe its tracks, diet, and the "
    "quiet calls it makes after rain.",
    "A floating Japanese temple above a sea of cherry blossoms at sunset, with a giant glowing koi fish swimming "
    "through the clouds, is an example of a vivid image caption. Such sentences still need many tokens so the "
    "Jacobian average can skip early sink positions.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n"
    "        a, b = b, a + b\n    return b\n# classic dynamic-programming rewrite of the recursive example",
]
