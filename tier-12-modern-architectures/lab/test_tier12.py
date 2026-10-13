import math, numpy as np, pytest
import arch_matrix as am
import ssm_scan as ss

# Toy config used for hand-computed checks:
# d=8, L=2, H=4 (dh=2), Hkv=2, d_ff=16 (gated), V=10, untied, 2 bytes
TOY = am.ArchSpec(name="toy", kind="decoder", d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=16, vocab=10)


def test_attention_params_hand_computed():
    # Q 8*8=64, K 8*4=32, V 32, O 8*8=64  -> 192
    assert am.attn_params(TOY) == 192


def test_total_params_hand_computed():
    # per layer: attn 192 + ffn 3*8*16=384 + norms 16 = 592 ; x2 = 1184
    # embedding 80 + head 80 + final norm 8 = 168 -> 1352
    assert am.total_params(TOY) == 1352
    assert am.active_params(TOY) == 1352            # dense: active == total


def test_tied_embeddings_remove_head():
    assert am.total_params(am.replace(TOY, tie_embeddings=True)) == 1352 - 80


def test_gqa_mqa_mha_attention_params_and_cache():
    mha = am.replace(TOY, n_kv_heads=4); mqa = am.replace(TOY, n_kv_heads=1)
    assert am.attn_params(mha) == 4 * 64            # 64+64+64+64
    assert am.attn_params(mqa) == 64 + 16 + 16 + 64
    # kv bytes/token = 2 * L * Hkv * dh * 2B
    assert am.kv_bytes_per_token(mha) == 2 * 2 * 4 * 2 * 2 == 64
    assert am.kv_bytes_per_token(TOY) == 32
    assert am.kv_bytes_per_token(mqa) == 16
    assert am.kv_bytes_per_token(mha) / am.kv_bytes_per_token(TOY) == 4 / 2   # ratio H/Hkv rule


def test_moe_total_vs_active():
    moe = am.replace(TOY, n_experts=4, top_k=2)
    ffn = 384
    router = 8 * 4
    assert am.ffn_params_layer(moe, active=False) == 4 * ffn + router
    assert am.ffn_params_layer(moe, active=True) == 2 * ffn + router
    delta_total = 2 * (3 * ffn + 0 + router)        # per layer extra: (4-1)*ffn + router, x2 layers
    assert am.total_params(moe) == 1352 + delta_total
    assert am.active_params(moe) == 1352 + 2 * (ffn + router)
    assert am.total_params(moe) > am.active_params(moe) > 0


def test_sliding_window_caps_cache():
    swa = am.replace(TOY, pattern="LL", window=16)
    per_tok_layer = 2 * 2 * 2 * 2               # 2*Hkv*dh*bytes = 16
    assert am.cache_bytes(swa, 10) == 2 * 10 * per_tok_layer
    assert am.cache_bytes(swa, 1000) == 2 * 16 * per_tok_layer      # capped at the window
    assert am.cache_bytes(TOY, 1000) == 2 * 1000 * per_tok_layer    # full attention grows linearly
    assert am.kv_bytes_per_token(swa) == 0                          # no marginal growth


def test_ssm_cache_is_constant_in_context():
    s = am.replace(TOY, pattern="SS", ssm_state=4, ssm_expand=2, ssm_conv=4)
    di = 16
    assert am.ssm_state_bytes(s) == 2 * (di * 4 + di * 3) * 2
    assert am.cache_bytes(s, 10) == am.cache_bytes(s, 10**6)


def test_flops_per_token_hand_computed():
    n = 100
    # matmul params: head 80 + 2 layers*(192+384) = 1232 ; flops = 2*1232 = 2464
    # attention: per layer 4*n*H*dh = 4*100*4*2 = 3200 ; x2 = 6400
    assert am.flops_per_token(TOY, n) == 2464 + 6400
    # flops grow linearly in n for global attention, flat for SSM
    s = am.replace(TOY, pattern="SS")
    assert am.flops_per_token(s, 10) == am.flops_per_token(s, 10**5)
    assert am.flops_per_token(TOY, 2 * n) - am.flops_per_token(TOY, n) == 6400


def test_moe_flops_use_active_params():
    moe = am.replace(TOY, n_experts=4, top_k=1)
    dense = am.replace(TOY, n_experts=1)
    extra_router = 2 * 2 * (8 * 4)            # 2 flops * 2 layers * router matrix
    assert am.flops_per_token(moe, 0) == am.flops_per_token(dense, 0) + extra_router


def test_pattern_length_validated():
    with pytest.raises(ValueError):
        am.replace(TOY, pattern="A").layers()


def test_tile_pattern():
    assert am.tile_pattern("LA", 5) == "LALAL"


def test_encoder_has_no_kv_cache_and_encdec_has_more_params():
    enc = am.replace(TOY, kind="encoder")
    assert am.kv_bytes_per_token(enc) == 0 and am.cache_bytes(enc, 100) == 0
    ed = am.replace(TOY, kind="encdec", n_enc_layers=2)
    assert am.total_params(ed) > am.total_params(TOY)


def test_max_concurrent_sequences():
    s = am.PRESETS["Dense GQA-8 (illustrative)"]
    k = am.max_concurrent_sequences(s, 80, 8192)
    assert k == int((80e9 - am.weight_bytes(s)) // am.cache_bytes(s, 8192))
    assert am.max_concurrent_sequences(am.PRESETS["Dense MQA (illustrative)"], 80, 8192) > k


@pytest.mark.parametrize("name,target_b,tol", [
    ("Llama-2-7B-shape (case study)", 6.74, 0.01),      # widely quoted figures; REVIEW_REQUIRED
    ("Llama-3-8B-shape (case study)", 8.03, 0.01),
    ("BERT-base (case study)", 0.110, 0.02),
    ("GPT-2 small (case study)", 0.124, 0.02),
])
def test_case_study_param_sanity(name, target_b, tol):
    got = am.total_params(am.PRESETS[name]) / 1e9
    assert abs(got - target_b) / target_b < tol


def test_mixtral_shape_active_fraction():
    m = am.PRESETS["Mixtral-8x7B-shape (case study)"]
    assert 0.25 < am.active_params(m) / am.total_params(m) < 0.30


def test_matrix_and_renderers():
    rows = am.default_matrix(4096)
    assert len(rows) == len(am.PRESETS)
    md = am.to_markdown(rows)
    assert md.count("\n") == len(rows) + 1
    html = am.to_html(rows, 4096)
    assert "<table>" in html and "REVIEW_REQUIRED" in html


# ---------------------------------------------------------------- SSM <-> attention
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_four_views_agree_selective(seed):
    x, a, B, C = ss.make_inputs(T=23, N=5, selective=True, seed=seed)   # T not a power of two
    y = ss.recurrent_scan(x, a, B, C)
    np.testing.assert_allclose(ss.matrix_form(a, B, C) @ x, y, atol=1e-10)
    np.testing.assert_allclose(ss.parallel_scan(x, a, B, C), y, atol=1e-10)


def test_conv_form_time_invariant():
    x, a, B, C = ss.make_inputs(T=20, N=4, selective=False, seed=3)
    np.testing.assert_allclose(ss.conv_form(x, a, B, C), ss.recurrent_scan(x, a, B, C), atol=1e-10)


def test_conv_form_rejects_selective():
    x, a, B, C = ss.make_inputs(T=8, N=3, selective=True)
    with pytest.raises(ValueError):
        ss.conv_form(x, a, B, C)


def test_matrix_is_causal_and_decays():
    _, a, B, C = ss.make_inputs(T=12, N=3, selective=True)
    M = ss.matrix_form(a, B, C)
    assert np.allclose(np.triu(M, 1), 0)
    # hand check of one entry: M[2,0] = C2 . (a1*a2*B0)
    np.testing.assert_allclose(M[2, 0], C[2] @ (a[1] * a[2] * B[0]))
    np.testing.assert_allclose(M[1, 1], C[1] @ B[1])


def test_state_is_constant_size():
    x, a, B, C = ss.make_inputs(T=50, N=4)
    _, hs = ss.recurrent_scan(x, a, B, C, return_states=True)
    assert hs.shape == (50, 4)                        # O(N) per step, independent of T


def test_linear_attention_is_a_equals_one_ssm():
    rng = np.random.default_rng(0)
    q, k, v = rng.standard_normal((3, 10, 4))
    np.testing.assert_allclose(ss.linear_attention_recurrent(q, k, v), ss.linear_attention_matrix(q, k, v), atol=1e-10)


def test_softmax_attention_has_no_finite_state():
    # last-token output of softmax attention changes if an EARLY token changes -> needs all T tokens cached
    rng = np.random.default_rng(1)
    q, k, v = rng.standard_normal((3, 8, 4))
    v2 = v.copy(); v2[0] += 1
    assert not np.allclose(ss.softmax_attention(q, k, v)[-1], ss.softmax_attention(q, k, v2)[-1])


def test_cost_crossover():
    d, N = 4096, 16
    n_cross = ss.decode_cost_ssm(d, N) / (4 * d)      # attention cost == ssm cost at n = 6N/4
    assert n_cross == 1.5 * N
    assert ss.decode_cost_attention(10 * n_cross, d) > ss.decode_cost_ssm(d, N)
