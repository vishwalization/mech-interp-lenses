#!/usr/bin/env python
"""
Diffusion lens: one image per (prompt, encoder depth) using pure ``(L, L, L)`` embeds.

Writes to ``results/diffusion_lens/<name>/``:

* ``pure_LLL/<idx>_<item>/L##.png`` – the images (skips existing files, so re-runs resume)
* ``grid_pure_LLL.png``             – prompt × depth grid
* ``diffusion_card_partial.json``   – consumed by run_token_cards.py
* optional CLIP curves with ``--clip``

After inspecting the grid, record the earliest layer where each object is
clearly present in a YAML file and pass it with ``--lstar-human``::

    dog: 15
    axolotl: null      # never a clear axolotl

Example::

    python scripts/run_diffusion_lens.py --config configs/animals.yaml --stride 3
"""

from __future__ import annotations

from pathlib import Path

import yaml

from _common import base_parser, dump_json, load_config, load_model_and_states, setup_logging


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--stride", type=int, default=None, help="sample every k-th depth (default: config.layer_stride)")
    p.add_argument("--depths", type=int, nargs="*", default=None, help="explicit list of depths (overrides --stride)")
    p.add_argument("--clip", action="store_true", help="score images with CLIP (pip install open_clip_torch)")
    p.add_argument("--lstar-human", type=Path, default=None, help="YAML of item → earliest clear layer")
    p.add_argument("--no-grid", action="store_true")
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("diffusion_lens")
    log = setup_logging(out, args.verbose)
    log.info("\n%s", cfg.summary())

    from lenses.diffusion_lens import clip_scores, lstar_from_scores, run_pure_lens, sample_depths, write_card_partial
    from lenses.plotting import plot_image_grid

    lp, hs, _unembed = load_model_and_states(cfg, args, log, build_unembed=False)
    depths = args.depths or sample_depths(hs.n_layers, args.stride or cfg.layer_stride)
    log.info("depths (L,L,L): %s", depths)

    results = run_pure_lens(lp.pipe, hs, cfg.items, depths, out, seed=cfg.generation.seed,
                            gen_kwargs=cfg.generation.kwargs(), log=log.info)
    log.info("%d images", len(results))

    if not args.no_grid:
        plot_image_grid(results, cfg.items, depths, out=out / "grid_pure_LLL.png")
        log.info("wrote %s", out / "grid_pure_LLL.png")

    scores = None
    if args.clip:
        scores = clip_scores(results, cfg.items, depths)
        dump_json({str(k): v for k, v in scores.items()}, out / "clip_scores.json")

    human = {item: None for item in cfg.items}
    if args.lstar_human and args.lstar_human.exists():
        human.update(yaml.safe_load(args.lstar_human.read_text()) or {})

    print(f"\n{'idx':>3}  {'item':<14}{'bucket':<7}{'L*_human':>9}{'L*_clip':>8}")
    for b, item in enumerate(cfg.items):
        Lc = lstar_from_scores(scores[b], depths) if scores else None
        print(f"{b:3d}  {item:<14}{('easy' if b < cfg.easy_count else 'hard'):<7}{str(human.get(item)):>9}{str(Lc):>8}")

    path = write_card_partial(out, items=cfg.items, prompts=cfg.prompts, depths=depths,
                              seed=cfg.generation.seed, lstar_human=human, clip=scores, model=cfg.model_id)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
