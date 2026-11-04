"""Shared fixtures: a tiny fake tokenizer + synthetic HiddenStates (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lenses.activations import HiddenStates  # noqa: E402
from lenses.unembed import Unembed  # noqa: E402


class FakeTokenizer:
    """Word-level tokenizer with a Ġ prefix, enough for position/vocab tests.

    vocab: 0 pad, 1 <|im_start|>, 2 user, 3 Ċ, then words. Multi-piece items
    are declared in ``pieces`` so ``"axolotl"`` → ``Ġax ol otl``.
    """

    pieces = {"axolotl": ["ax", "ol", "otl"], "quokka": ["qu", "ok", "ka"], "okapi": ["ok", "api"]}

    def __init__(self):
        self.id2tok = ["<pad>", "<|im_start|>", "user", "Ċ"]
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def _add(self, t):
        if t not in self.tok2id:
            self.tok2id[t] = len(self.id2tok)
            self.id2tok.append(t)
        return self.tok2id[t]

    def _split(self, text):
        out = []
        for i, w in enumerate(text.split()):
            if w in self.pieces:
                ps = self.pieces[w]
                out += [("Ġ" if i else "") + ps[0]] + ps[1:]
            else:
                out.append(("Ġ" if i else "") + w)
        return out

    def encode_row(self, text, chat=True, pad_to=16):
        toks = (["<|im_start|>", "user", "Ċ"] if chat else []) + self._split(text)
        ids = [self._add(t) for t in toks]
        ids += [0] * (pad_to - len(ids))
        return torch.tensor(ids)

    def __len__(self):
        return len(self.id2tok)

    def convert_ids_to_tokens(self, ids):
        return [self.id2tok[int(i)] for i in ids]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.id2tok[int(i)].replace("Ġ", " ") for i in ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, **kw):
        toks = self._split(text)
        ids = [self._add(t) for t in toks]
        offs, cur = [], 0
        for t in toks:
            s = t.replace("Ġ", "")
            start = text.find(s, cur)
            offs.append((start, start + len(s)))
            cur = start + len(s)
        return {"input_ids": ids, "offset_mapping": offs}


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def prompts():
    return ["a dog over a white table", "an axolotl over a white table", "a quokka over a white table"]


@pytest.fixture
def hs(tok, prompts):
    torch.manual_seed(0)
    ids = torch.stack([tok.encode_row(p) for p in prompts])
    mask = (ids != 0).long()
    B, S, D, n_layers = ids.shape[0], ids.shape[1], 8, 5
    layers = tuple(torch.randn(B, S, D) * (1 + L) for L in range(n_layers))
    return HiddenStates(layers=layers, input_ids=ids, attention_mask=mask, prompts=tuple(prompts))


@pytest.fixture
def unembed(tok, hs):
    torch.manual_seed(1)
    V, D = len(tok), hs.d_model          # one unembed row per vocab id
    return Unembed(W_U=torch.randn(V, D), norm_weight=torch.ones(D), norm_eps=1e-6, tokenizer=tok)
