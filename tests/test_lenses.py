import torch

from lenses.activations import fingerprint
from lenses.diffusion_lens import make_prompt_embeds, native_triples, palindrome_triples, sample_depths
from lenses.j_lens import DiagonalTransport, j_lens
from lenses.logit_lens import emergence_curve, first_emergence_layer, logit_lens, rank_curve
from lenses.unembed import rms_norm


# --- unembed / rms_norm ------------------------------------------------------

def test_rms_norm_matches_reference():
    torch.manual_seed(0)
    x, w, eps = torch.randn(2560), torch.rand(2560) + 0.5, 1e-6
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    assert torch.allclose(rms_norm(x, w, eps), ref, atol=1e-6)


def test_rms_norm_is_pure():
    x = torch.randn(16)
    x0 = x.clone()
    rms_norm(x, torch.ones(16), 1e-6)
    assert torch.equal(x, x0)


def test_unembed_probs_sum_to_one_and_topk_decodes(unembed, hs):
    v = hs.vector(1, 0, 4)
    p = unembed.probs(v)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    top = unembed.topk(v, k=3)
    assert len(top) == 3 and all(isinstance(t, str) and 0 <= pr <= 1 for t, pr in top)


# --- logit lens ----------------------------------------------------------------

def test_logit_lens_shapes_and_no_mutation(unembed, hs):
    fp0 = fingerprint(hs, 0, 4)
    rows = logit_lens(hs, unembed, b=0, pos=4, topk=5)
    rows2 = logit_lens(hs, unembed, b=0, pos=4, topk=5)
    assert set(rows) == set(range(hs.n_layers))
    assert rows == rows2                           # deterministic on re-run
    assert fingerprint(hs, 0, 4) == fp0            # cache untouched


def test_last_layer_is_not_double_normed(unembed, hs):
    # last layer: apply_norm=False → logits == raw v @ W_U.T
    v = hs.vector(hs.last, 0, 4)
    expected = (v @ unembed.W_U.T).softmax(-1).topk(1).indices
    got_tok = logit_lens(hs, unembed, 0, 4, topk=1)[hs.last][0][0]
    assert got_tok == unembed.tokenizer.decode([int(expected)]).strip()


def test_emergence_and_rank_curves_consistent(unembed, hs):
    ids = [5, 6]
    probs = emergence_curve(hs, unembed, 0, 4, ids)
    masses, ranks = rank_curve(hs, unembed, 0, 4, ids)
    assert len(probs) == hs.n_layers
    assert all(abs(a - b) < 1e-6 for a, b in zip(probs, masses))
    assert all(r is None or r >= 1 for r in ranks)


def test_first_emergence_layer_rules():
    probs = [1.0, 0.0, 0.01, 0.2, 0.9]
    assert first_emergence_layer(probs) == 3                      # skips L0 by default
    assert first_emergence_layer(probs, skip_embedding_layer=False) == 0
    assert first_emergence_layer(probs, rank_curve_=[None, 10, 99, 99, 1], rank_thresh=20) == 1
    assert first_emergence_layer([0.0] * 5) is None


# --- diffusion lens --------------------------------------------------------------

def test_make_prompt_embeds_matches_flux_stacking(hs):
    layers = (1, 2, 3)
    out = make_prompt_embeds(hs, layers, batch_index=0, dtype=torch.float32)
    B, S, D = 1, hs.seq_len, hs.d_model
    assert out.shape == (B, S, 3 * D)
    # channel-major within the feature dim: [layer1 | layer2 | layer3] per position
    for k, L in enumerate(layers):
        assert torch.allclose(out[0, :, k * D:(k + 1) * D], hs.layers[L][0].float())


def test_layer_selections():
    assert sample_depths(37, 3) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    assert sample_depths(37, 1)[-1] == 36 and len(sample_depths(37, 1)) == 37
    nt = native_triples(37)
    assert nt[0] == (0, 9, 18) and nt[-1] == (18, 27, 36) and len(nt) == 19
    assert palindrome_triples(18, 37)[0] == (9, 18, 27) and palindrome_triples(18, 37)[-1] == (17, 18, 19)


# --- j lens ------------------------------------------------------------------------

def test_diagonal_transport_recovers_scale():
    """If h_final = h_L * s exactly, the fit must recover s."""
    torch.manual_seed(0)
    B, S, D = 2, 10, 6
    s = torch.linspace(0.5, 2.0, D)
    h0 = torch.randn(B, S, D)
    from lenses.activations import HiddenStates

    hs = HiddenStates(layers=(h0, h0 * s), input_ids=torch.ones(B, S, dtype=torch.long),
                      attention_mask=torch.ones(B, S, dtype=torch.long))
    t = DiagonalTransport.fit(hs, min_pos=0)
    assert torch.allclose(t.scales[0], s, atol=1e-4)
    assert torch.equal(t.scales[1], torch.ones(D))


def test_j_lens_identity_at_last_layer(unembed, hs):
    t = DiagonalTransport.fit(hs)
    j = j_lens(hs, unembed, t, 0, 4, topk=3)
    raw = logit_lens(hs, unembed, 0, 4, topk=3)
    assert [tok for tok, _ in j[hs.last]] == [tok for tok, _ in raw[hs.last]]


def test_transport_roundtrip(tmp_path, hs):
    t = DiagonalTransport.fit(hs)
    t.save(tmp_path / "t.pt")
    t2 = DiagonalTransport.load(tmp_path / "t.pt")
    assert all(torch.equal(t.scales[k], t2.scales[k]) for k in t.scales)
