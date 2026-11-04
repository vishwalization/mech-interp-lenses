#!/usr/bin/env python
"""
Logit lens at the item-token position, with vs. without FLUX's chat template.

Prints a per-layer top-k table side by side for one prompt and writes the full
tables for every prompt to ``results/logit_lens/<name>/``. Also prints cache
fingerprints before/after so a corrupted-activation bug cannot go unnoticed.

Example::

    python scripts/run_logit_lens.py --config configs/animals.yaml --prompt-idx 0
"""

from __future__ import annotations

from _common import base_parser, dump_json, load_config, load_model_and_states, setup_logging


def main() -> None:
    p = base_parser(__doc__.split("\n\n")[0])
    p.add_argument("--prompt-idx", type=int, default=0, help="which prompt to print in full")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--skip-no-chat", action="store_true", help="only run the chat-templated path")
    args = p.parse_args()

    cfg = load_config(args)
    out = cfg.lens_dir("logit_lens")
    log = setup_logging(out, args.verbose)
    log.info("\n%s", cfg.summary())
    topk = args.topk or cfg.topk

    from lenses.activations import capture_hidden_states, fingerprint
    from lenses.logit_lens import format_side_by_side, logit_lens
    from lenses.tokens import find_item_positions

    lp, hs_chat, unembed = load_model_and_states(cfg, args, log, chat_template=True)
    pos_chat = find_item_positions(lp.tokenizer, hs_chat.input_ids, cfg.items, which=cfg.position)

    hs_plain, pos_plain = None, None
    if not args.skip_no_chat:
        hs_plain = capture_hidden_states(lp.text_encoder, lp.tokenizer, cfg.prompts, device=lp.device, chat_template=False)
        pos_plain = find_item_positions(lp.tokenizer, hs_plain.input_ids, cfg.items, which=cfg.position)

    # --- all prompts → json ---------------------------------------------------
    tables = {}
    for b, item in enumerate(cfg.items):
        entry = {
            "item": item,
            "prompt": cfg.prompts[b],
            "pos_with_chat": pos_chat[b],
            "tokens_with_chat": hs_chat.tokens(b, lp.tokenizer, 24),
            "with_chat": logit_lens(hs_chat, unembed, b, pos_chat[b], topk=topk),
        }
        if hs_plain is not None:
            entry["pos_without_chat"] = pos_plain[b]
            entry["tokens_without_chat"] = hs_plain.tokens(b, lp.tokenizer)
            entry["without_chat"] = logit_lens(hs_plain, unembed, b, pos_plain[b], topk=topk)
        tables[item] = entry
    dump_json(tables, out / f"logit_lens_top{topk}.json")

    # --- one prompt → console -----------------------------------------------------
    b = args.prompt_idx
    fp_before = fingerprint(hs_chat, b, pos_chat[b])
    left = logit_lens(hs_chat, unembed, b, pos_chat[b], topk=topk)
    right = logit_lens(hs_plain, unembed, b, pos_plain[b], topk=topk) if hs_plain is not None else {}
    fp_after = fingerprint(hs_chat, b, pos_chat[b])

    print(f"\nprompt[{b}] = {cfg.prompts[b]!r}   item = {cfg.items[b]!r}")
    print(f"  tokens (chat):    {hs_chat.tokens(b, lp.tokenizer, 20)}  → reading position {pos_chat[b]}")
    if hs_plain is not None:
        print(f"  tokens (no chat): {hs_plain.tokens(b, lp.tokenizer)}  → reading position {pos_plain[b]}")
    print(f"  cache fingerprint unchanged across the pass: {fp_before == fp_after}\n")
    print(format_side_by_side(left, right))
    log.info("wrote %s", out / f"logit_lens_top{topk}.json")


if __name__ == "__main__":
    main()
