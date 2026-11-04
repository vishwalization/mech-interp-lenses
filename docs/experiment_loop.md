# Push-to-run GPU loop — setup guide

> Original planning notes for the RunPod + GitHub Actions loop. Paths and names below match this repo's `.github/workflows/run-experiment.yml`.

This guide turns your 11-step table into a working system. Once it's set up, your daily
routine becomes:

> **Describe the experiment → let Claude write the code → push to GitHub → the GPU runs it
> by itself → you look at the results later.**

You do this once (the setup). After that, every experiment is just "write code, push, wait."

---

## The big picture (how the pieces talk to each other)

```
   You (phone/laptop)                    GitHub                        RunPod (GPU pod)
 ┌────────────────────┐          ┌──────────────────┐          ┌──────────────────────┐
 │ Describe experiment │  push    │  Your repo       │  notify  │  Self-hosted runner   │
 │ in Claude / Cursor  ├─────────►│  + GitHub Actions├─────────►│  (lives in tmux)      │
 │ Edit code in Cursor │          │  workflow file   │          │  runs your code on GPU│
 └────────────────────┘          └──────────────────┘          │  saves to /workspace  │
          ▲                                                     └──────────┬───────────┘
          │                      review via SFTP / Remote-SSH              │
          └─────────────────────────────────────────────────────────────-┘
```

Three machines, one loop:
- **Your laptop/phone** = where you think and write code.
- **GitHub** = the messenger. A `git push` is the button that starts everything.
- **RunPod pod** = the worker with the GPU. A small program on it (the "runner") waits for
  GitHub to say "go," then runs your experiment and saves everything.

---

## What you need before starting

- A **GitHub account** and one **repository** for your research.
- A **RunPod account** with a **GPU pod** that has a **persistent network volume** attached
  (this is important — the volume is the part that never gets erased).
- **Cursor** installed on your laptop (VS Code with Claude built in).
- Your model code ready to run (the FLUX.2 Klein + Qwen loading, plus your Diffusion Lens /
  Logit Lens / J-Matrix code).

> Rule of thumb: anything you want to *keep* must live on the persistent volume, which on
> RunPod is usually mounted at **`/workspace`**. The rest of the pod's disk can be wiped when
> the pod restarts.

---

# PART A — One-time setup on RunPod (the "background worker")

This is Steps 5–8 in your table. You do this once.

### A1. Open a terminal on your pod and make a tmux session

`tmux` keeps a program alive even after you close your browser/SSH window.

```bash
# on the RunPod pod
tmux new -s runner
```

(To leave tmux later without killing it: press **Ctrl+B**, then **D**. To come back:
`tmux attach -t runner`.)

### A2. Install your Python environment ON the persistent volume

Put the environment on `/workspace` so it survives pod restarts.

```bash
cd /workspace
python -m venv venv-mech
source venv-mech/bin/activate
pip install -r /workspace/<your-repo>/requirements.txt   # torch, diffusers, transformers, pyyaml, etc.
mkdir -p /workspace/results /workspace/logs
```

### A3. Install the self-hosted GitHub runner (also on the volume)

Go to your repo on GitHub → **Settings → Actions → Runners → New self-hosted runner →
Linux**. GitHub shows you the exact commands with a **token**. Use them here, but install
into `/workspace` so it isn't wiped:

```bash
cd /workspace
mkdir actions-runner && cd actions-runner

# GitHub gives you the exact version + URL on that page — copy those:
curl -o runner.tar.gz -L https://github.com/actions/runner/releases/download/vX.X.X/actions-runner-linux-x64-X.X.X.tar.gz
tar xzf runner.tar.gz

# Register it with YOUR repo. The label "runpod-gpu" is how the workflow finds this machine.
./config.sh --url https://github.com/USERNAME/REPO --token YOUR_TOKEN --labels runpod-gpu
```

> The token from GitHub expires in about an hour, so run `config.sh` soon after copying it.
> If it expires, just refresh that GitHub page for a new one.

### A4. Start the runner (inside your tmux session)

```bash
cd /workspace/actions-runner
./run.sh
```

You'll see "Listening for Jobs." Now press **Ctrl+B** then **D** to detach. The runner keeps
running in the background 24/7 — this is your "background agent."

**Important restart note:** `tmux` survives you disconnecting, but if the *pod itself* is
stopped and started again, all running programs stop. After a pod restart you'll need to
reattach and start the runner again:

```bash
tmux new -s runner
cd /workspace/actions-runner && source /workspace/venv-mech/bin/activate && ./run.sh
```

(You can automate this with a RunPod **start command / startup script** so it happens on its
own — optional, but nice for true 24/7.)

---

# PART B — Set up your GitHub repo (structure + templates)

This is Steps 3–4 and 7 in your table. Do this on your laptop in Cursor.

### B1. Use a clear folder layout

```
mech-interp-lenses/
├── .github/workflows/run-experiment.yml   # the automation rule (Part C)
├── experiments/
│   ├── _template/                         # copy this to start a new experiment
│   │   ├── config.yaml                    #   lens config + `method:` key
│   │   └── run.py                         #   dispatches to scripts/run_<method>.py
│   └── exp005_diffusion_lens/             # one folder per experiment
├── src/lenses/                            # the reusable lens code (see README)
├── scripts/run_*.py                       # one CLI per lens
├── scripts/new_experiment.sh
├── requirements.txt
└── README.md
```

Why this shape: **one folder per experiment** keeps things tidy, and pushing a new folder
under `experiments/` is what will trigger a run.

### B2. An experiment's config file (`config.yaml`)

The config is an ordinary lens config (same schema as `configs/*.yaml`) plus a
`method:` key naming the script to run and an optional `args:` list of extra
flags. See `experiments/_template/config.yaml`.

```yaml
name: exp005_diffusion_lens
method: diffusion_lens         # logit_lens | emergence_curve | diffusion_lens | j_lens | token_cards | layer_ablation
args: ["--stride", "2"]
prompts: [...]
generation: {seed: 42, num_inference_steps: 4}
```

### B3. The experiment runner (`run.py`)

`experiments/_template/run.py` reads `method`, runs
`scripts/run_<method>.py --config <config> --results-dir <output>` and writes
`metrics.json` with status and timing. You normally don't edit it — put new
behaviour in `src/lenses/` and `scripts/`.

### B4. A helper to start a new experiment quickly (`scripts/new_experiment.sh`)

```bash
#!/usr/bin/env bash
# usage: ./scripts/new_experiment.sh exp006_logit_lens
NAME="$1"
DIR="experiments/$NAME"
cp -r experiments/_template "$DIR"
# set the name inside the copied config
sed -i "s/^name:.*/name: $NAME/" "$DIR/config.yaml"
echo "Created $DIR — now edit its config.yaml and run.py"
```

---

# PART C — The automation rule (GitHub Actions)

This is Steps 5–6 in your table: the file that says "when experiment code is pushed, run it
on the GPU machine."

Create `.github/workflows/run-experiment.yml`:

```yaml
name: Run Experiment

on:
  push:
    paths:
      - 'experiments/**'        # only runs when experiment files change

# Only one GPU job at a time, so two pushes don't fight over the GPU
concurrency:
  group: gpu-runner
  cancel-in-progress: false

jobs:
  run-experiment:
    runs-on: [self-hosted, runpod-gpu]   # <-- matches the label from A3
    steps:
      - name: Get the code
        uses: actions/checkout@v4
        with:
          fetch-depth: 2                 # need last commit to see what changed

      - name: Figure out which experiment changed
        id: detect
        shell: bash
        run: |
          BEFORE="${{ github.event.before }}"
          if [ "$BEFORE" = "0000000000000000000000000000000000000000" ] || [ -z "$BEFORE" ]; then
            FILES=$(git show --name-only --pretty="" HEAD)
          else
            FILES=$(git diff --name-only "$BEFORE" "${{ github.sha }}")
          fi
          EXP=$(echo "$FILES" | grep '^experiments/' | grep -v '^experiments/_template' \
                | cut -d/ -f2 | sort -u | head -n1)
          echo "Detected experiment: $EXP"
          echo "exp=$EXP" >> "$GITHUB_OUTPUT"

      - name: Run it on the GPU
        if: steps.detect.outputs.exp != ''
        shell: bash
        run: |
          source /workspace/venv-mech/bin/activate
          export PYTHONPATH="$PWD/src:$PWD"              # so `from src...` works
          EXP="${{ steps.detect.outputs.exp }}"
          OUT="/workspace/results/$EXP"
          mkdir -p "$OUT" /workspace/logs
          python "experiments/$EXP/run.py" \
            --config "experiments/$EXP/config.yaml" \
            --output "$OUT" 2>&1 | tee "/workspace/logs/$EXP.log"
```

What this does, in plain words:
1. Waits for a push that touches `experiments/`.
2. Sends the job **only** to your RunPod machine (because of the `runpod-gpu` label).
3. Works out which experiment folder you just added/changed.
4. Runs that experiment on the GPU and saves everything to `/workspace/results/<name>/`,
   with a copy of the log in `/workspace/logs/`.

> Good news: self-hosted runners are **free and unmetered** on GitHub — only GitHub's own
> cloud runners cost minutes. Your GPU time is billed by RunPod, not GitHub.

---

# PART D — Your daily loop (the fun part)

Now that setup is done, here's the everyday flow (Steps 1–4 and 11):

1. **Describe it** — In the Claude app or in Cursor, say something like:
   *"Run Diffusion Lens on Qwen + FLUX.2 Klein using a prompt about a red apple, save the
   attention maps. Make it experiment exp005."*
2. **Claude plans + writes code** — In Cursor, Claude can read your whole repo, so it knows
   your folder layout and helper functions. It fills in `config.yaml` and `run.py`. Laptop is
   best here because Cursor keeps the full project in mind.
3. **Create the folder** — Either let Claude create `experiments/exp005_diffusion_lens/`, or
   run `./scripts/new_experiment.sh exp005_diffusion_lens` and edit the files.
4. **Push** — this is the trigger:
   ```bash
   git add experiments/exp005_diffusion_lens
   git commit -m "exp005: diffusion lens, red apple"
   git push
   ```
5. **Wait** — the runner picks it up automatically and runs it on the GPU. You can watch
   progress on GitHub → **Actions** tab, live.
6. **Review later** (Part E).

---

# PART E — Looking at your results

Everything lands in `/workspace/results/exp005_diffusion_lens/` on the pod:

```
results/exp005_diffusion_lens/
├── images/            # generated images
├── attention_maps/    # attention / lens outputs
├── metrics.json       # numbers / summary
└── run.log            # full log
```

Three easy ways to see them:

- **VS Code / Cursor Remote-SSH (nicest):** connect to the pod over SSH, then open the
  `/workspace/results` folder like it's a local folder. You see images and files right there.
- **SFTP from a file browser:** use FileZilla, Cyberduck, or `sftp`. RunPod gives you an IP
  and a port on the pod's Connect page:
  ```bash
  sftp -P <PORT> root@<POD_IP>
  # then: cd /workspace/results  ->  get -r exp005_diffusion_lens
  ```
- **Pull a copy to your laptop with scp:**
  ```bash
  scp -P <PORT> -r root@<POD_IP>:/workspace/results/exp005_diffusion_lens ./
  ```

On your phone, an SFTP app (e.g. any Files app that supports SFTP) works the same way for a
quick look.

---

# Common gotchas (worth reading once)

- **Keep everything important on `/workspace`.** The venv, the runner, and all results must
  live there, or a pod restart wipes them.
- **After a pod restart, restart the runner** (Part A4). Consider a RunPod startup script so
  it comes back on its own.
- **Gated models / API tokens** (e.g. Hugging Face): don't commit them to GitHub. Put them in
  the pod's environment (a `.env` on the volume, or `export HF_TOKEN=...` in the venv
  activation), since the code runs on the pod anyway.
- **One GPU, one job at a time:** the `concurrency` block in the workflow makes pushes queue
  up instead of crashing into each other.
- **If a run fails,** the log is in two places: the GitHub **Actions** page (click the failed
  run) and `/workspace/logs/<name>.log` on the pod. So you can always see what happened.
- **First push edge case:** the workflow already handles brand-new branches (the
  "0000..." check), so it won't break on the very first push.

---

## How this maps back to your table

| Your step | Where it lives in this guide |
|---|---|
| 1. Describe experiment | Part D, step 1 |
| 2. Claude suggests a plan | Part D, step 2 (Cursor reads the repo) |
| 3. Create/edit code file | Part B (structure + templates) |
| 4. Push to GitHub (the trigger) | Part D, step 4 |
| 5. GitHub detects push | Part C (the workflow `on: push`) |
| 6. Runner picks up the job | Part A (self-hosted runner in tmux) |
| 7. Folder structure created | Part B3 `run.py` + Part C (`mkdir`) |
| 8. Experiment runs on GPU | Part C "Run it on the GPU" step |
| 9. Outputs saved automatically | Part B3 + `/workspace/results/...` |
| 10. Logs saved (volume + Actions) | Part B3 logging + `tee` in Part C |
| 11. You review results | Part E |
