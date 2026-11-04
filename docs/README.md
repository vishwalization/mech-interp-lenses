# docs/

| File | What it is |
|------|------------|
| [`three_lenses.md`](three_lenses.md) | How the three lenses relate, run order, pattern cheat-sheet, where results land |
| [`notes/offload-safe-activations.md`](notes/offload-safe-activations.md) | The `enable_sequential_cpu_offload()` activation-corruption bug and the fixes baked into `lenses/activations.py` + `lenses/unembed.py` |
| [`experiment_loop.md`](experiment_loop.md) | Setting up the push-to-run loop: self-hosted GitHub runner on a RunPod GPU, `experiments/` layout, workflow |

## Related reading

- nostalgebraist, *interpreting GPT: the logit lens* (2020) — the original logit lens.
- Toker et al., *Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines* (ACL 2024).
- Kaplan, Toker et al., *Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models* (2025).
- Toker et al., *Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models* (2025).
- Anthropic, *Jacobian lens* — [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens).
- Neel Nanda, *A Comprehensive Mechanistic Interpretability Explainer & Glossary*.
