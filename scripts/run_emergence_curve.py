#!/usr/bin/env python
"""
Emergence curves: P(target word) per layer under the logit lens, easy vs. hard items.

Writes to ``results/logit_lens/<name>/``:

* ``emergence_curves.json``   – curve per item + L*_logit rows
* ``logit_card_partial.json`` – consumed by run_token_cards.py
* ``top1_table.csv``          – layer × item top-1 token and probability
* ``emergence_curve.png``, ``emergence_small_multiples.png``

Example::

    python scripts/run_emergence_curve.py --config configs/fruits.yaml
"""

from __future__ import annotations

from _common import base_parser, dump_json, load_config, load_model_and_states, setup_logging


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--own-token-only", action="store_true",
                   help="measure P(the exact sub-token at the read position) instead of all word variants / fallback")
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("logit_lens")
    log = setup_logging(out, args.verbose)
    log.info("\n%s", cfg.summary())

    from lenses.logit_lens import first_emergence_layer, rank_curve, top1_table
    from lenses.plotting import plot_emergence_curves, plot_small_multiples
    from lenses.tokens import find_item_positions, resolve_targets

    lp, hs, unembed = load_model_and_states(cfg, args, log)
    positions = find_item_positions(lp.tokenizer, hs.input_ids, cfg.items, which=cfg.position)

    if args.own_token_only:
        vocab_ids = {b: [int(hs.input_ids[b, positions[b]])] for b in positions}
        own = {b: True for b in positions}
    else:
        vocab_ids, own = resolve_targets(lp.tokenizer, cfg.items, fallback=cfg.fallback_word)

    for b, item in enumerate(cfg.items):
        toks = [lp.tokenizer.decode([i]) for i in vocab_ids[b][:6]]
        log.info("%-14s pos=%2d own_token=%-5s targets(%d)=%s", item, positions[b], own[b], len(vocab_ids[b]), toks)

    curves, rows = {}, []
    for b, item in enumerate(cfg.items):
        probs, ranks = rank_curve(hs, unembed, b, positions[b], vocab_ids[b])
        curves[b] = probs
        rows.append({
            "idx": b, "item": item, "bucket": "easy" if b < cfg.easy_count else "hard",
            "own_token": own[b], "pos": positions[b],
            "Lstar_logit": first_emergence_layer(probs, ranks, prob_thresh=cfg.prob_thresh, rank_thresh=cfg.rank_thresh),
            "peak_layer": max(range(1, len(probs)), key=lambda L: probs[L]),
            "peak_prob": max(probs[1:]),
            "final_prob": probs[-1], "final_rank": ranks[-1],
        })

    dump_json({"items": cfg.items, "prompts": cfg.prompts, "curves": {str(k): v for k, v in curves.items()}, "rows": rows},
              out / "emergence_curves.json")
    dump_json({"method": "logit_lens", "model": cfg.model_id, "items": cfg.items, "prompts": cfg.prompts, "rows": rows},
              out / "logit_card_partial.json")

    df = top1_table(hs, unembed, positions, cfg.items)
    df.to_csv(out / "top1_table.csv")

    known = [own[b] for b in range(len(cfg.items))]
    label = cfg.extra.get("concept_label", "concept")
    plot_emergence_curves(curves, cfg.items, known, title=f"Emergence of the {label} across layers",
                          ylabel=f"P(target {label} token)", out=out / "emergence_curve.png")
    plot_small_multiples(curves, cfg.items, known, title=f"Emergence of each {label} across layers",
                         ylabel=f"P(target {label} token)", out=out / "emergence_small_multiples.png")

    print(f"\n{'item':<14}{'bucket':<7}{'own':<6}{'L*_logit':>9}{'peak L':>8}{'peak P':>8}{'P_final':>9}")
    for r in rows:
        print(f"{r['item']:<14}{r['bucket']:<7}{str(r['own_token']):<6}{str(r['Lstar_logit']):>9}{r['peak_layer']:>8}{r['peak_prob']:>8.3f}{r['final_prob']:>9.3f}")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
