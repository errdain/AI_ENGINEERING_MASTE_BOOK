import numpy as np, pytest, torch
import torch.nn.functional as F
from attention_np import (softmax, causal_mask, padding_mask, sliding_window_mask, scaled_dot_product_attention,
                          split_heads, merge_heads, MultiHeadAttention as NpMHA, kv_cache_bytes, score_matrix_bytes)
from attention_torch import sdpa_manual, MultiHeadAttention as TMHA

R = np.random.default_rng(0)


def test_softmax_rows_sum_to_one_and_stable():
    x = R.normal(size=(4, 7)) * 1000          # would overflow a naive exp
    p = softmax(x)
    assert np.isfinite(p).all() and np.allclose(p.sum(-1), 1)


def test_sqrt_dk_scaling_keeps_score_variance_near_one():
    for d in (16, 256):
        q, k = R.normal(size=(5000, d)), R.normal(size=(5000, d))
        raw = (q * k).sum(-1)
        assert abs(raw.var() / d - 1) < 0.15              # unscaled variance ~ d
        assert abs((raw / np.sqrt(d)).var() - 1) < 0.15   # scaled variance ~ 1


def test_unscaled_softmax_saturates_more():
    q, k = R.normal(size=(1, 256)), R.normal(size=(64, 256))
    s = (q @ k.T)[0]
    assert softmax(s).max() > softmax(s / 16).max()


def test_numpy_sdpa_matches_torch_sdpa():
    q, k, v = (R.normal(size=(2, 3, 6, 8)) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(*(torch.tensor(a) for a in (q, k, v))).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)
    np.testing.assert_allclose(w.sum(-1), 1)


def test_causal_matches_torch_and_no_future_leak():
    q, k, v = (R.normal(size=(1, 2, 5, 4)) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    ref = F.scaled_dot_product_attention(*(torch.tensor(a) for a in (q, k, v)), is_causal=True).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)
    assert np.allclose(np.triu(w[0, 0], 1), 0)
    v2 = v.copy(); v2[..., 4, :] += 100                   # change the LAST token's value
    out2, _ = scaled_dot_product_attention(q, k, v2, causal_mask(5))
    np.testing.assert_allclose(out[..., :4, :], out2[..., :4, :])   # earlier outputs unchanged


def test_padding_mask_ignores_pad_tokens():
    q, k, v = (R.normal(size=(2, 1, 6, 4)) for _ in range(3))
    pm = padding_mask([3, 6], 6)[:, None, None, :]        # (B,1,1,S)
    out, w = scaled_dot_product_attention(q, k, v, pm)
    assert np.allclose(w[0, :, :, 3:], 0)
    v2 = v.copy(); v2[0, :, 3:] = 999                      # garbage in PAD positions
    out2, _ = scaled_dot_product_attention(q, k, v2, pm)
    np.testing.assert_allclose(out[0], out2[0])
    ref = F.scaled_dot_product_attention(*(torch.tensor(a) for a in (q, k, v)), attn_mask=torch.tensor(pm)).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_fully_masked_row_is_finite_in_numpy():
    q, k, v = (R.normal(size=(1, 3, 4)) for _ in range(3))
    out, _ = scaled_dot_product_attention(q, k, v, np.zeros((3, 3), dtype=bool))
    assert np.isfinite(out).all()


def test_split_merge_roundtrip():
    x = R.normal(size=(2, 5, 12))
    assert split_heads(x, 3).shape == (2, 3, 5, 4)
    np.testing.assert_array_equal(merge_heads(split_heads(x, 3)), x)


def test_sliding_window_and_sinks():
    m = sliding_window_mask(8, 3)
    assert m[5].tolist() == [0, 0, 0, 1, 1, 1] + [0, 0]
    m2 = sliding_window_mask(8, 3, sinks=1)
    assert m2[5, 0] and not m2[5, 1] and not m2[2, 5]
    assert sliding_window_mask(8, 99).tolist() == causal_mask(8).tolist()


def _copy_weights(npm, tm):
    for name in ("q", "k", "v", "o"):
        getattr(tm, "w" + name).weight.data = torch.tensor(getattr(npm, "W" + name).T.copy())


@pytest.mark.parametrize("hk", [4, 2, 1])   # MHA, GQA, MQA
def test_numpy_mha_matches_torch_module(hk):
    npm = NpMHA(16, 4, hk, seed=3)
    tm = TMHA(16, 4, hk).double(); _copy_weights(npm, tm)
    x = R.normal(size=(2, 6, 16))
    a = npm(x, mask=causal_mask(6))
    b = tm(torch.tensor(x), is_causal=True).detach().numpy()
    np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("hk", [4, 2, 1])
def test_manual_equals_sdpa_backend(hk):
    torch.manual_seed(0)
    m = TMHA(32, 4, hk)
    x = torch.randn(3, 7, 32)
    pad = torch.tensor([[1] * 7, [1] * 5 + [0] * 2, [1] * 3 + [0] * 4], dtype=torch.bool)[:, None, None, :]
    y1 = m(x, attn_mask=pad)
    m.use_sdpa = True
    y2 = m(x, attn_mask=pad)
    torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)
    m.use_sdpa = False
    c1 = m(x, is_causal=True)
    m.use_sdpa = True
    torch.testing.assert_close(c1, m(x, is_causal=True), atol=1e-5, rtol=1e-5)


def test_sdpa_manual_matches_functional_incl_float_mask_and_scale():
    q, k, v = (torch.randn(2, 2, 5, 8) for _ in range(3))
    bias = torch.randn(5, 5)
    torch.testing.assert_close(sdpa_manual(q, k, v, attn_mask=bias), F.scaled_dot_product_attention(q, k, v, attn_mask=bias), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sdpa_manual(q, k, v, scale=0.3), F.scaled_dot_product_attention(q, k, v, scale=0.3), atol=1e-5, rtol=1e-5)


def test_cross_attention_shapes_and_independence_of_decoder_length():
    torch.manual_seed(0)
    m = TMHA(16, 2)
    dec, enc = torch.randn(2, 4, 16), torch.randn(2, 9, 16)
    assert m(dec, context=enc).shape == (2, 4, 16)
    a = NpMHA(16, 2)(R.normal(size=(1, 3, 16)), context=R.normal(size=(1, 7, 16)))
    assert a.shape == (1, 3, 16)


def test_head_symmetry_permutation_equivariance():
    """Self-attention without positions: permuting tokens permutes outputs (why Tier 6 adds positions)."""
    m = NpMHA(16, 4); x = R.normal(size=(1, 6, 16)); p = R.permutation(6)
    np.testing.assert_allclose(m(x)[:, p], m(x[:, p]), atol=1e-10)


def test_mqa_gqa_parameter_and_cache_savings():
    h, hk = TMHA(64, 8, 8), TMHA(64, 8, 2)
    nkv = lambda m: sum(p.numel() for p in (*m.wk.parameters(), *m.wv.parameters()))
    assert nkv(hk) * 4 == nkv(h)
    full = kv_cache_bytes(32, 32, 128, 4096); g = kv_cache_bytes(32, 8, 128, 4096)
    assert full == 4 * g == 2 * 32 * 32 * 128 * 4096 * 2
    assert score_matrix_bytes(4096, 32) == 32 * 4096 * 4096 * 2
