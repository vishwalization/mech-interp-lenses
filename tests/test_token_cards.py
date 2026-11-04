import json

import pytest

from lenses.config import ExperimentConfig
from lenses.token_cards import Thresholds, build_cards, classify, leading_pattern

TH = Thresholds(early=12, late_gap=8, disagree=8, n_layers=37)


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"Lstar_logit": 8, "Lstar_J": 7, "Lstar_diff": 22, "bucket": "easy"}, "A"),   # verbal early, visual late
        ({"Lstar_logit": 30, "Lstar_J": 28, "Lstar_diff": 6, "bucket": "easy"}, "B"),  # visual first
        ({"Lstar_logit": 5, "Lstar_J": 6, "Lstar_diff": 9, "bucket": "easy"}, "C"),    # easy triad
        ({"Lstar_logit": None, "Lstar_J": 34, "Lstar_diff": None, "bucket": "hard"}, "D"),
        ({"Lstar_logit": 4, "Lstar_J": 20, "Lstar_diff": 15, "bucket": "easy"}, "G"),  # logit ≠ J
        # None = lens ran, target never emerged → counts as late
        ({"Lstar_logit": None, "Lstar_J": None, "Lstar_diff": 6, "bucket": "easy"}, "B"),
        ({"Lstar_logit": 5, "Lstar_J": 6, "Lstar_diff": None, "bucket": "easy"}, "A"),
        ({"Lstar_logit": None, "Lstar_J": None, "Lstar_diff": None, "bucket": "hard"}, "D"),
        # fewer than two lenses actually run → incomplete
        ({"Lstar_diff": 15, "bucket": "hard"}, "?"),
        ({"Lstar_logit": None, "Lstar_J": None, "Lstar_diff": 15, "bucket": "hard", "measured": "diff"}, "?"),
    ],
)
def test_classify(row, expected):
    assert classify(row, TH) == expected


def test_build_cards_tracks_which_lenses_ran():
    loaded = {"j_lens": {"items": ["dog"], "rows": [{"item": "dog", "Lstar_J_diy": 7, "Lstar_logit": 8}]}}
    df = build_cards(loaded, easy_count=1, th=TH)
    assert df.loc[0, "measured"] == "logit,J"
    assert df.loc[0, "pattern"] == "C"                       # two verbal lenses, both early, easy bucket
    df2 = build_cards(loaded, easy_count=1, manual={"dog": {"Lstar_diff": 22}}, th=TH)
    assert df2.loc[0, "measured"] == "logit,J,diff"
    assert df2.loc[0, "pattern"] == "A"


def test_build_cards_merges_partials_and_manual(tmp_path):
    loaded = {
        "j_lens": {"items": ["dog", "axolotl"], "rows": [
            {"item": "dog", "Lstar_J_diy": 7, "Lstar_logit": 8},
            {"item": "axolotl", "Lstar_J_diy": 20, "Lstar_logit": None},
        ]},
        "diffusion": {"items": ["dog", "axolotl"], "Lstar_human": {"dog": 22, "axolotl": None}},
    }
    df = build_cards(loaded, easy_count=1, manual={"axolotl": {"notes": "never clear"}}, th=TH)
    assert list(df["item"]) == ["dog", "axolotl"]
    assert df.loc[0, "pattern"] == "A"
    assert df.loc[1, "pattern"] == "D"
    assert df.loc[1, "notes"] == "never clear"
    lead = leading_pattern(df, min_repeats=3)
    assert lead["count"] == 1 and lead["strong"] is False


def test_config_loads_and_parses_items(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(json.dumps({
        "prompts": ["a dog over a white table", "a quokka over a white table"],
        "fallback_word": "animal", "easy_count": 1,
        "generation": {"seed": 7}, "custom_knob": 3,
    }))
    cfg = ExperimentConfig.load(cfg_path, results_dir=str(tmp_path / "res"))
    assert cfg.items == ["dog", "quokka"]
    assert cfg.name == "c"
    assert cfg.generation.seed == 7 and cfg.generation.num_inference_steps == 4
    assert cfg.extra == {"custom_knob": 3}
    assert cfg.lens_dir("logit_lens") == tmp_path / "res" / "logit_lens" / "c"
