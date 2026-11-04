#!/usr/bin/env python
"""
Experiment entry point for the push-to-run GPU loop (see docs/experiment_loop.md).

The GitHub Actions workflow calls::

    python experiments/<exp>/run.py --config experiments/<exp>/config.yaml --output /workspace/results/<exp>

``config.yaml`` is an ordinary lens config (same schema as ``configs/*.yaml``)
plus a ``method`` key naming which script to run. Everything else is delegated
to ``scripts/run_<method>.py`` with ``--results-dir`` pointed at ``--output``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
METHODS = {
    "logit_lens": "run_logit_lens.py",
    "emergence_curve": "run_emergence_curve.py",
    "diffusion_lens": "run_diffusion_lens.py",
    "j_lens": "run_j_lens.py",
    "token_cards": "run_token_cards.py",
    "layer_ablation": "run_layer_ablation.py",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    method = cfg.get("method", "emergence_curve")
    if method not in METHODS:
        sys.exit(f"unknown method {method!r}; choose from {sorted(METHODS)}")

    extra = cfg.get("args", [])  # optional list of extra CLI flags, e.g. ["--stride", "2"]
    cmd = [sys.executable, str(REPO / "scripts" / METHODS[method]),
           "--config", str(Path(args.config).resolve()), "--results-dir", str(out), *map(str, extra)]
    print("running:", " ".join(cmd), flush=True)
    started = datetime.now(timezone.utc)
    rc = subprocess.call(cmd, cwd=REPO)

    (out / "metrics.json").write_text(json.dumps({
        "name": cfg.get("name"), "method": method, "status": "ok" if rc == 0 else f"failed({rc})",
        "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
