# Offload-safe activations: the bug that shaped this codebase

> Status: resolved. Everything in `lenses/activations.py` and `lenses/unembed.py`
> exists because of what is described here. Keep this note if you ever touch
> how hidden states or weights are read.

## Setup

- `Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=bf16)`
- `pipe.enable_sequential_cpu_offload()` — required on a 16 GB GPU
- `pipe.vae.enable_tiling()`
- Text encoder: `Qwen3ForCausalLM`, residual width **2560**, **36 layers + embedding** → 37 hidden states, vocab **151 936**

## Symptom

Running the logit lens on a cached residual stream:

- **First run:** early layers decode sensibly, later layers come back as
  near-uniform softmax garbage —
  `[('"', 0.0), ('!', 0.0), ('#', 0.0), ('%', 0.0), ('$', 0.0)]`
  (top indices are just early vocab punctuation; probabilities round to zero).
- **Second run of the same cell, no recompute:** garbage on *every* layer.
- **Recovery:** re-running the text-encoder forward pass made it work again — once.

So the cached activations (or their storage) were being damaged between
reads. Model weights were fine.

## Root causes

Two stacked issues:

1. **Views + in-place / kernel RMSNorm.** `hs[b, pos]` is a *view* into the
   cached tensor. Passing it through the live `final_norm` module (Qwen3
   `RMSNorm`, possibly a hub kernel via `@use_kernel_forward_from_hub`) can
   write through storage that aliases the cache. The first pass looks fine
   while computing; the second pass reads already-mutated vectors.

2. **`enable_sequential_cpu_offload()` + activations left on the GPU.** Under
   sequential offload, later kernels (the norm, the 150k-wide unembed matmul,
   anything else in the pipeline) reuse GPU memory the cached hidden states
   were sitting in. That matches "first run partly garbage, second run fully
   garbage".

`clone()` on the GPU was **not** enough in practice. What held up:

## Fix 1 — CPU float32 snapshots

```python
hs = tuple(h.detach().float().cpu().contiguous().clone() for h in output.hidden_states)
```

→ `lenses.activations.capture_hidden_states`. Nothing on the GPU can clobber
them, and everything downstream runs on CPU in float32.

## Fix 2 — never call the live norm; materialise weights

Reading `module.weight` on an offloaded component returns a **meta** tensor
(the real data lives in accelerate's `weights_map`), so `.cpu()` on it raises

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

The weights are on-loaded through accelerate's public API and snapshotted once:

```python
from accelerate.utils import align_module_device, has_offloaded_params

def materialize_weight(module, name="weight"):
    if has_offloaded_params(module):
        with align_module_device(module, execution_device="cpu"):
            return getattr(module, name).detach().float().cpu().contiguous().clone()
    return getattr(module, name).detach().float().cpu().contiguous().clone()
```

Pass the **module** (`te.get_output_embeddings()`, `te.model.norm`), not
`.weight`. → `lenses.unembed.materialize_weight`. The norm itself is a pure
function (`lenses.unembed.rms_norm`): no `nn.Module`, no hooks, no in-place ops.

Verified by reproducing the exact meta-offload on a tiny module: the old
helper raised the identical error; the new one returns weights matching ground
truth, re-offloads cleanly, and is stable across repeated reads.

## Fix 3 — don't double-norm the last layer

HF `Qwen*Model.forward` with `output_hidden_states=True` appends the pre-norm
input *before* each block and, after the loop, appends `self.norm(hidden_states)`.
So `hidden_states[-1]` is **already post-final-norm**. Every lens applies the
final RMSNorm to all layers **except the last** (`apply_norm = L != hs.last`).

## How to tell the difference now

`lenses.activations.fingerprint(hs, b, pos)` is a cheap checksum. The CLIs
print it before and after a pass:

- fingerprints **differ** between two runs → something is still writing into
  the cache; it is a lifetime/mutation bug.
- fingerprints **identical** but early layers still look odd → that is the
  method (raw logit lens is not a tuned lens), not corruption.

## Also learned

- `_get_qwen3_prompt_embeds()` runs the text encoder **once per layer
  combination**. `output_hidden_states=True` already returns *every* layer for
  the whole batch in one forward pass, so capture once and re-slice —
  `N×` cheaper for `N` combinations (`lenses.diffusion_lens.make_prompt_embeds`).
- Fitting Anthropic's `jlens` needs real forward+backward passes; it cannot
  wrap the offloaded encoder (meta tensors). Load a separate, non-offloaded
  `AutoModelForCausalLM.from_pretrained(model_id, subfolder="text_encoder")`
  for it (`lenses.j_lens.fit_official_jlens`).
