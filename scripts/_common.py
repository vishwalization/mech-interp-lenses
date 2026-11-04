"""Shared CLI plumbing for the ``scripts/run_*.py`` entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# allow `python scripts/run_x.py` without `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lenses.config import ExperimentConfig  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, help="YAML experiment config (see configs/)")
    p.add_argument("--results-dir", default=None, help="override results root")
    p.add_argument("--name", default=None, help="override experiment name (sub-folder under results/<lens>/)")
    p.add_argument("--model-id", default=None, help="override model id")
    p.add_argument("--no-offload", action="store_true", help="disable sequential CPU offload (needs ≥24 GB VRAM)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def setup_logging(out_dir: Path, verbose: bool = False) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("lenses")


def load_config(args) -> ExperimentConfig:
    return ExperimentConfig.load(args.config, results_dir=args.results_dir, name=args.name, model_id=args.model_id)


def load_model_and_states(cfg: ExperimentConfig, args, log, chat_template: bool = True, build_unembed: bool = True):
    """Load the pipeline, capture hidden states for ``cfg.prompts``, and (optionally) build the unembed.

    ``build_unembed=False`` skips materialising the 151k × 2560 float32 W_U (~1.5 GB RAM)
    for scripts that only generate images.
    """
    from lenses.activations import capture_hidden_states
    from lenses.model import load_pipeline
    from lenses.unembed import Unembed

    log.info("loading %s (sequential_cpu_offload=%s)", cfg.model_id, not args.no_offload)
    lp = load_pipeline(cfg.model_id, sequential_cpu_offload=not args.no_offload)
    log.info("capturing hidden states for %d prompts (chat_template=%s)", len(cfg.prompts), chat_template)
    hs = capture_hidden_states(
        lp.text_encoder, lp.tokenizer, cfg.prompts, device=lp.device,
        chat_template=chat_template, max_sequence_length=cfg.max_sequence_length,
    )
    log.info("n_layers=%d batch=%d seq=%d d=%d", hs.n_layers, hs.batch_size, hs.seq_len, hs.d_model)
    unembed = Unembed.from_text_encoder(lp.text_encoder, lp.tokenizer) if build_unembed else None
    return lp, hs, unembed


def dump_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
