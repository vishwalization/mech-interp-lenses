#!/usr/bin/env python
"""
J-lens: transport each layer into the final residual basis, then unembed.

Default is the DIY diagonal transport (CPU, instant). ``--official`` fits
Anthropic's Jacobian lens on a separate non-offloaded copy of the encoder
(needs ``pip install "git+https://github.com/anthropics/jacobian-lens.git#egg=jlens"``
and a large GPU).

Writes to ``results/j_lens/<name>/``:

* ``j_lens_top5.json``        – per-layer top-k under J-lens *and* raw logit lens, every prompt
* ``j_lens_card_partial.json``– L*_J / L*_logit rows, consumed by run_token_cards.py
* ``emergence_j_vs_logit.png``, ``rank_curves_diy_j.png``
* ``diagonal_transport.pt``    – the fitted scales

Example::

    python scripts/run_j_lens.py --config configs/animals.yaml --print-idx 0 5
"""

from __future__ import annotations

from _common import base_parser, dump_json, load_config, load_model_and_states, setup_logging


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--print-idx", type=int, nargs="*", default=[0], help="prompts to print side-by-side")
    p.add_argument("--every", type=int, default=3, help="print every k-th layer in the console table")
    p.add_argument("--official", action="store_true", help="also fit/apply Anthropic's jlens (Path A)")
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("j_lens")
    log = setup_logging(out, args.verbose)
    log.info("\n%s", cfg.summary())

    from lenses.j_lens import DiagonalTransport, j_lens, j_vs_logit_curves, lstar_table, write_card_partial
    from lenses.logit_lens import format_side_by_side, logit_lens
    from lenses.plotting import plot_j_vs_logit, plot_rank_curves
    from lenses.tokens import find_item_positions, resolve_targets

    lp, hs, unembed = load_model_and_states(cfg, args, log)
    positions = find_item_positions(lp.tokenizer, hs.input_ids, cfg.items, which=cfg.position)
    vocab_ids, own = resolve_targets(lp.tokenizer, cfg.items, fallback=cfg.fallback_word)

    transport = DiagonalTransport.fit(hs)
    transport.save(out / "diagonal_transport.pt")
    log.info("fitted diagonal transport for %d layers (L0 mean scale %.3f)", len(transport.scales), float(transport.scales[0].mean()))

    # per-prompt tables
    tables = {}
    for b, item in enumerate(cfg.items):
        tables[item] = {
            "prompt": cfg.prompts[b], "pos": positions[b],
            "j_lens": j_lens(hs, unembed, transport, b, positions[b], topk=cfg.topk),
            "logit_lens": logit_lens(hs, unembed, b, positions[b], topk=cfg.topk),
        }
    dump_json(tables, out / f"j_lens_top{cfg.topk}.json")

    for b in args.print_idx:
        t = tables[cfg.items[b]]
        layers = list(range(0, hs.n_layers, max(1, args.every)))
        print(f"\nprompt[{b}] = {cfg.prompts[b]!r}   item = {cfg.items[b]!r}   pos = {positions[b]}")
        print(format_side_by_side({L: t["j_lens"][L] for L in layers}, {L: t["logit_lens"][L] for L in layers},
                                  "DIY J-lens top-k", "raw logit lens top-k"))

    # curves + L*
    curves = j_vs_logit_curves(hs, unembed, transport, positions, vocab_ids)
    rows = lstar_table(curves, cfg.items, easy_count=cfg.easy_count, prob_thresh=cfg.prob_thresh, rank_thresh=cfg.rank_thresh)
    plot_j_vs_logit(curves, cfg.items, cfg.easy_count, out=out / "emergence_j_vs_logit.png")
    plot_rank_curves(curves["j_rank"], cfg.items, cfg.easy_count, out=out / "rank_curves_diy_j.png")

    print(f"\n{'idx':>3} {'item':<14}{'bucket':<7}{'L*_J':>5}{'L*_logit':>9}{'P_J∞':>7}{'rank_J∞':>9}")
    for r in rows:
        print(f"{r['idx']:3d} {r['item']:<14}{r['bucket']:<7}{str(r['Lstar_J_diy']):>5}{str(r['Lstar_logit']):>9}{r['final_prob_J']:>7.3f}{str(r['final_rank_J']):>9}")

    path = write_card_partial(out, prompts=cfg.prompts, items=cfg.items, positions=positions, rows=rows, curves=curves, model=cfg.model_id)
    log.info("wrote %s", path)

    if args.official:
        from lenses.j_lens import apply_official_jlens, fit_official_jlens

        try:
            lens, lens_model = fit_official_jlens(lp.tokenizer, out, model_id=cfg.model_id, log=log.info)
            official = {item: apply_official_jlens(lens, lens_model, lp.tokenizer, cfg.prompts[b], topk=cfg.topk)
                        for b, item in enumerate(cfg.items)}
            dump_json(official, out / "official_jlens_top5.json")
            log.info("wrote %s", out / "official_jlens_top5.json")
        except Exception as e:  # keep the DIY results even if Path A fails
            log.error("official jlens failed: %s: %s", type(e).__name__, e)


if __name__ == "__main__":
    main()
