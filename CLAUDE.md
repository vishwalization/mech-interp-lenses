# CLAUDE.md

Guidance for Claude Code when working in this repo. This is a **static file** — it is not auto-updated; edit it when the setup changes.

## Project purpose
Mechanistic-interpretability "lenses" for text-to-image diffusion. Applies **diffusion lens**, **logit lens**, and **J-matrix** analysis to **FLUX.2 Klein** (with **Qwen** as a helper model). Each experiment renders images + attention maps and records metrics.

## Repo layout
- `experiments/` — one folder per experiment. `_template/` is the scaffold (`config.yaml`, `run.py`); real experiments are copies like `exp001_test/`.
- `src/` — lens implementations: `diffusion_lens.py`, `logit_lens.py`, `j_matrix.py`. **Currently stubs** (raise `NotImplementedError`) — the real model code goes here.
- `scripts/new_experiment.sh` — scaffolds a new experiment from `_template`.
- `.github/workflows/run-experiment.yml` — the CI workflow that runs experiments on the GPU.
- `requirements.txt` — Python deps (torch, diffusers, transformers, Pillow, matplotlib, pyyaml, …).
- Results/logs are **gitignored** — they live on the RunPod volume, not in git.

## The experiment loop
1. Push to `main` touching `experiments/**` triggers `run-experiment.yml`.
2. The workflow detects the changed experiment folder (ignoring `_template`).
3. It runs on the self-hosted GPU runner:
   `python experiments/<exp>/run.py --config experiments/<exp>/config.yaml --output /workspace/results/<exp>`
4. Only one GPU job runs at a time (concurrency group `gpu-runner`).

## Creating an experiment
1. `./scripts/new_experiment.sh exp002_whatever`
2. Edit the new `config.yaml` and `run.py` (wire in `src/` lens code).
3. `git push` to `main` — the runner does the rest.
4. Results appear on the pod at `/workspace/results/<exp>/` (`images/`, `attention_maps/`, `metrics.json`, `run.log`).

## RunPod environment
- Persistent volume `/workspace` holds everything: `venv-mech/` (the venv), `mech-interp-lenses/` (cloned repo), `actions-runner/`, `results/`, `logs/`.
- **The venv is `/workspace/venv-mech`** (not `venv`). The workflow activates it via `source /workspace/venv-mech/bin/activate`.
- Runner auto-starts on pod boot via the pod **start command**:
  `bash -c "( sleep 20 && cd /workspace/actions-runner && RUNNER_ALLOW_RUNASROOT=1 ./run.sh > /workspace/logs/runner.log 2>&1 & ) ; exec /start.sh"`
  (backgrounds the runner after the volume mounts, then execs `/start.sh` so SSH stays up and the container doesn't restart-loop).
- Runner label: `runpod-gpu`. Runner log: `/workspace/logs/runner.log`.
- After starting a stopped pod, the runner shows **Idle** (green) on GitHub in ~30s.

## Gotchas
- When giving the user commands to paste into the pod terminal, **never split a long command or URL across lines** — the web terminal breaks them and they fail. Use shell variables for long URLs/tokens.
- The workflow only runs on `main` (and only for `experiments/**` changes). Root-level or `src/` changes do not trigger a GPU run.
- This repo uses a **local** git identity (`vishwalization` / GitHub noreply email), distinct from the machine's global `vishwa-eve` work identity. Don't assume the global identity here.
