#!/usr/bin/env python
"""
Token cards: fuse L*_logit, L*_J and L*_diff per item and classify patterns A–G.

CPU only — reads the ``*_card_partial.json`` files the other three scripts
wrote under ``results/<lens>/<name>/``. Manual overrides (after looking at the
diffusion grid) go in a YAML passed with ``--manual``::

    dog:     {Lstar_diff: 15, notes: "clear dog from L15"}
    axolotl: {Lstar_diff: null, notes: "salamander-ish, never axolotl"}

Writes ``results/token_cards/<name>/token_cards.{csv,json}`` and
``emergence_scatter.png``.

Example::

    python scripts/run_token_cards.py --config configs/animals.yaml --manual configs/lstar_human_animals.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from _common import base_parser, load_config, setup_logging


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--manual", type=Path, default=None, help="YAML of per-item overrides")
    p.add_argument("--early", type=int, default=12)
    p.add_argument("--late-gap", type=int, default=8)
    p.add_argument("--disagree", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("token_cards")
    log = setup_logging(out, args.verbose)

    from lenses.plotting import plot_emergence_scatter
    from lenses.token_cards import RESEARCH_PATHS, Thresholds, build_cards, leading_pattern

    loaded = {}
    for lens, fname in (("logit", "logit_card_partial.json"), ("diffusion", "diffusion_card_partial.json"), ("j_lens", "j_lens_card_partial.json")):
        path = cfg.out / {"logit": "logit_lens", "diffusion": "diffusion_lens", "j_lens": "j_lens"}[lens] / cfg.name / fname
        if path.exists():
            loaded[lens] = json.loads(path.read_text())
            log.info("loaded %s", path)
        else:
            log.info("missing (ok if manual): %s", path)

    manual = yaml.safe_load(args.manual.read_text()) if args.manual and args.manual.exists() else None
    th = Thresholds(early=args.early, late_gap=args.late_gap, disagree=args.disagree)
    df = build_cards(loaded, items=cfg.items, easy_count=cfg.easy_count, manual=manual, th=th)

    df.to_csv(out / "token_cards.csv", index=False)
    df.to_json(out / "token_cards.json", orient="records", indent=2)
    plot_emergence_scatter(df, out=out / "emergence_scatter.png")

    print("\n" + df.to_string(index=False))
    lead = leading_pattern(df)
    if lead:
        print(f"\nPattern histogram: {lead['histogram']}")
        print(f"→ Leading pattern: {lead['pattern']} — {lead['name']} (n={lead['count']}, items={lead['items']})")
        print("  Repeats ≥3 → strong enough to pursue as the main research path." if lead["strong"]
              else "  Fewer than 3 repeats — collect more tokens before locking a path.")
        if lead["pattern"] in RESEARCH_PATHS:
            claim, first = RESEARCH_PATHS[lead["pattern"]]
            print(f"  Claim:            {claim}\n  First experiment: {first}")
    else:
        print("\nNo classified patterns yet — fill in L* values.")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
