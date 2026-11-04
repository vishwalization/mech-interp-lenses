import pytest

from lenses.tokens import find_item_position, find_item_positions, parse_item


@pytest.mark.parametrize(
    "prompt, item",
    [
        ("a dog over a white table", "dog"),
        ("an apple over a white table", "apple"),
        ("a quokka over a white table", "quokka"),          # naive replace("a ", "") gave "quokkover white table"
        ("an axolotl over a white table", "axolotl"),
        ("a platypus over a white table", "platypus"),
        ("A spaceship parked near a vegetable shop in desert", "spaceship"),
    ],
)
def test_parse_item(prompt, item):
    assert parse_item(prompt) == item


def test_single_token_position(tok, hs):
    # ['<|im_start|>','user','Ċ','a','Ġdog', ...] → 4
    assert find_item_position(tok, hs.input_ids[0], "dog") == 4


def test_multi_piece_last_and_first(tok, hs):
    # ['<|im_start|>','user','Ċ','an','Ġax','ol','otl','Ġover', ...]
    assert find_item_position(tok, hs.input_ids[1], "axolotl", which="last") == 6
    assert find_item_position(tok, hs.input_ids[1], "axolotl", which="first") == 4


def test_multi_piece_does_not_fall_back_to_next_word(tok, hs):
    # the old heuristic returned the position of 'Ġover' for every multi-piece animal
    pos = find_item_position(tok, hs.input_ids[2], "quokka")
    assert tok.convert_ids_to_tokens([hs.input_ids[2][pos]])[0] != "Ġover"
    assert pos == 6


def test_positions_batch(tok, hs):
    assert find_item_positions(tok, hs.input_ids, ["dog", "axolotl", "quokka"]) == {0: 4, 1: 6, 2: 6}


def test_missing_item_raises(tok, hs):
    with pytest.raises(ValueError):
        find_item_position(tok, hs.input_ids[0], "giraffe")
