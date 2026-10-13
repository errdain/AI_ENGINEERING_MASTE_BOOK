"""Tier 8 tests: shapes, causality, loss decreases, parameter-count formula, norms, activations."""
import itertools
import pytest
import torch
from mini_gpt import GPTConfig, MiniGPT, RMSNorm, count_params, count_params_formula
from train_mini_gpt import build_data, train

SMALL = dict(vocab_size=23, block_size=16, n_layer=2, n_head=2, d_model=32)


def test_output_shapes_and_loss_at_init():
    cfg = GPTConfig(**SMALL); m = MiniGPT(cfg)
    x = torch.randint(0, 23, (3, 10)); y = torch.randint(0, 23, (3, 10))
    logits, loss = m(x, y)
    assert logits.shape == (3, 10, 23)
    # untrained model ~ uniform: loss close to ln(V)
    assert abs(loss.item() - torch.log(torch.tensor(23.0)).item()) < 0.3


def test_block_preserves_shape_residual_stream():
    cfg = GPTConfig(**SMALL); m = MiniGPT(cfg)
    x = torch.randn(2, 7, 32)
    for b in m.blocks:
        assert b(x).shape == x.shape


@pytest.mark.parametrize("prenorm", [True, False])
def test_causal_mask_no_future_leak(prenorm):
    torch.manual_seed(0)
    m = MiniGPT(GPTConfig(**SMALL, prenorm=prenorm)).eval()
    a = torch.randint(0, 23, (1, 12)); b = a.clone()
    b[0, 6:] = torch.randint(0, 23, (6,))            # change only positions >= 6
    la, _ = m(a); lb, _ = m(b)
    assert torch.allclose(la[:, :6], lb[:, :6], atol=1e-5), "future tokens leaked into the past"
    assert not torch.allclose(la[:, 6:], lb[:, 6:], atol=1e-5)   # sanity: the test can detect changes


def test_attention_weights_upper_triangle_zero_and_rows_sum_to_one():
    m = MiniGPT(GPTConfig(**SMALL)).eval(); m(torch.randint(0, 23, (2, 9)))
    att = m.blocks[0].attn.last_attn                  # (B, h, T, T)
    assert torch.all(torch.triu(att, diagonal=1) == 0)
    assert torch.allclose(att.sum(-1), torch.ones(2, 2, 9), atol=1e-5)


def test_loss_decreases_on_char_corpus():
    chars, _, tr, va = build_data()
    cfg = GPTConfig(vocab_size=len(chars), block_size=32, n_layer=2, n_head=2, d_model=48)
    _, h = train(cfg, tr, va, max_steps=80, B=16, eval_every=40, log=False)
    first, last = sum(h["train"][:5]) / 5, sum(h["train"][-5:]) / 5
    assert last < first - 0.8, (first, last)
    assert h["val"][-1] < h["val"][0]


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = MiniGPT(GPTConfig(**SMALL)); opt = torch.optim.Adam(m.parameters(), 3e-3)
    x = torch.randint(0, 23, (4, 12)); y = torch.randint(0, 23, (4, 12))
    for _ in range(150):
        _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3


@pytest.mark.parametrize("norm,prenorm,act,tie,bias", list(itertools.product(
    ["layernorm", "rmsnorm"], [True, False], ["gelu", "swiglu"], [True, False], [True, False])))
def test_param_count_formula_matches(norm, prenorm, act, tie, bias):
    cfg = GPTConfig(**SMALL, norm=norm, prenorm=prenorm, act=act, tie_weights=tie, bias=bias,
                    d_ff=88 if act == "swiglu" else None)
    assert count_params(MiniGPT(cfg)) == count_params_formula(cfg)


def test_gpt2_style_closed_form_12Ld2():
    """For gelu/4d/bias/tied: total = V*d + T*d + L*(12d^2+13d) + 2d."""
    c = GPTConfig(vocab_size=100, block_size=32, n_layer=3, n_head=4, d_model=64)
    d, L = 64, 3
    assert count_params_formula(c) == 100 * d + 32 * d + L * (12 * d * d + 13 * d) + 2 * d
    assert count_params(MiniGPT(c)) == count_params_formula(c)


def test_weight_tying_shares_storage_and_saves_V_times_d():
    t = MiniGPT(GPTConfig(**SMALL, tie_weights=True)); u = MiniGPT(GPTConfig(**SMALL, tie_weights=False))
    assert t.head.weight.data_ptr() == t.tok.weight.data_ptr()
    assert count_params(u) - count_params(t) == 23 * 32


def test_rmsnorm_unit_rms_and_layernorm_zero_mean():
    x = torch.randn(4, 8, 32) * 5 + 3
    r = RMSNorm(32)(x)
    assert torch.allclose(r.pow(2).mean(-1), torch.ones(4, 8), atol=1e-3)
    ln = torch.nn.LayerNorm(32)(x)
    assert torch.allclose(ln.mean(-1), torch.zeros(4, 8), atol=1e-4)


def test_head_divisibility_and_block_size_errors():
    with pytest.raises(AssertionError):
        GPTConfig(**{**SMALL, "n_head": 5})
    m = MiniGPT(GPTConfig(**SMALL))
    with pytest.raises(AssertionError):
        m(torch.zeros(1, 17, dtype=torch.long))


def test_generate_length_and_range():
    m = MiniGPT(GPTConfig(**SMALL))
    out = m.generate(torch.zeros(1, 3, dtype=torch.long), 40, top_k=5)   # longer than block_size: windows
    assert out.shape == (1, 43) and out.max() < 23
