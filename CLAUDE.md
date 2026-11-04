# CLAUDE.md

Guidance for AI coding assistants working in this repo. Static file — edit it when the setup changes.

## What this repo is
Mechanistic-interpretability lenses (logit, diffusion, Jacobian) on the Qwen3 text encoder inside
FLUX.2 Klein 4B, plus a token-card classifier that fuses the three per-item emergence depths.
`README.md` is the front door; `docs/three_lenses.md` is the method overview.

## Layout
- `src/lenses/` — the package. All model access goes through `model.py` (load) → `activations.py`
  (capture to CPU fp32) → `unembed.py` (meta-safe weights, functional RMSNorm). The lenses
  (`logit_lens.py`, `diffusion_lens.py`, `j_lens.py`) only ever read `HiddenStates` + `Unembed`.
- `scripts/run_*.py` — one CLI per lens; they share `scripts/_common.py`. Results go to
  `results/<lens>/<config-name>/` (gitignored).
- `configs/*.yaml` — prompt families. `lenses.config.ExperimentConfig.load` is the schema.
- `tests/` — CPU-only pytest; uses a fake tokenizer and synthetic hidden states. Run `pytest -q`.
- `experiments/` + `.github/workflows/run-experiment.yml` — push-to-run GPU loop on a self-hosted
  RunPod runner; `docs/experiment_loop.md`. `_template/run.py` dispatches on `method:` to `scripts/`.
- `assets/` — curated figures for the README. Never commit `results/`.

## Invariants (do not break)
1. Hidden states are snapshotted `detach().float().cpu().contiguous().clone()` immediately after the
   forward pass. Never keep GPU activations across calls under sequential CPU offload.
2. Never read `.weight` on an offloaded module; use `lenses.unembed.materialize_weight(module)`.
3. The last hidden state is already post-final-norm: apply RMSNorm to layers `0 … n-2` only
   (`apply_norm = L != hs.last`).
4. No notebooks in the repo (`*.ipynb` is gitignored). Scripts are the source of truth.
Full story: `docs/notes/offload-safe-activations.md`.

## GPU loop gotchas
- The workflow only runs on `main` for `experiments/**` changes, excluding `_template/`. Editing
  `src/` or `scripts/` does not trigger a GPU run.
- Pod venv is `/workspace/venv-mech`; runner label `runpod-gpu`; runner log `/workspace/logs/runner.log`.
- When giving commands to paste into the pod web terminal, never split a long command or URL across lines.

## Git identity
This repo uses a **local** git identity (`vishwalization` / GitHub noreply email), distinct from the
machine's global work identity. Commits are authored by the repo owner only — no co-author trailers.
