#!/usr/bin/env python
"""
Layer-combination ablation of FLUX's three-layer prompt-embed stack.

FLUX.2 Klein feeds the DiT a stack of three encoder layers, natively
``(i, i+9, i+18)``. This sweeps alternative triples on long scene prompts and
saves one image strip per prompt — the experiment that originally motivated the
three lenses.

Modes:

* ``triples``    – native ``(i, i+9, i+18)`` for every ``i``
* ``palindrome`` – ``(c−k, c, c+k)`` squeezing toward a centre layer (default 18)
* ``custom``     – ``--combos 9,18,27 12,18,24 …``

Example::

    python scripts/run_layer_ablation.py --config configs/scenes.yaml --mode palindrome
"""

from __future__ import annotations

from _common import base_parser, load_config, load_model_and_states, setup_logging


def _parse_combo(s: str):
    parts = tuple(int(x) for x in s.replace("(", "").replace(")", "").split(","))
    if len(parts) != 3:
        raise ValueError(f"combo must have 3 layers: {s}")
    return parts


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=["triples", "palindrome", "custom"], default="triples")
    p.add_argument("--center", type=int, default=18, help="palindrome centre layer")
    p.add_argument("--every", type=int, default=1, help="take every k-th combo")
    p.add_argument("--combos", nargs="*", default=None, help='custom combos, e.g. "9,18,27" "12,18,24"')
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("layer_ablation")
    log = setup_logging(out, args.verbose)
    log.info("\n%s", cfg.summary())

    from lenses.diffusion_lens import native_triples, palindrome_triples, run_combo_sweep
    from lenses.plotting import plot_image_strip

    lp, hs, _ = load_model_and_states(cfg, args, log, build_unembed=False)

    if args.mode == "triples":
        combos = native_triples(hs.n_layers)
    elif args.mode == "palindrome":
        combos = palindrome_triples(center=args.center, n_layers=hs.n_layers)
    else:
        combos = [_parse_combo(c) for c in (args.combos or [])]
    combos = combos[:: max(1, args.every)]
    log.info("%d combos: %s … %s", len(combos), combos[0], combos[-1])

    results = run_combo_sweep(lp.pipe, hs, cfg.items, combos, out, tag=args.mode, seed=cfg.generation.seed,
                              gen_kwargs=cfg.generation.kwargs(), log=log.info)

    for b, item in enumerate(cfg.items):
        paths = [results[(b, tuple(c))] for c in combos]
        strip = out / f"strip_{args.mode}_{b:02d}_{item.replace(' ', '_')[:30]}.png"
        plot_image_strip(paths, [str(c) for c in combos], title=cfg.prompts[b][:90], out=strip)
        log.info("wrote %s", strip)


if __name__ == "__main__":
    main()
