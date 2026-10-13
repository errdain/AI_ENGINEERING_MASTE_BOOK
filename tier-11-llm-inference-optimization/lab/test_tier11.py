import copy, random
import pytest
import torch
torch.set_num_threads(2)
from tiny_gpt import TinyGPT
from decoding import (top_k_filter, top_p_filter, min_p_filter, apply_temperature, generate, beam_search,
                      repetition_penalty, hit_stop_sequence, sample_next)
from quant import quantize_sym, dequantize, quant_error, quantize_model
from scheduler import Req, simulate, metrics
from kv_memory import kv_bytes, kv_bytes_per_token


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TinyGPT(vocab=32, d=32, n_layer=2, n_head=4, n_kv_head=2, max_len=64).eval()


def test_kv_cache_logits_equal_full_recompute(model):
    ids = torch.randint(0, 32, (2, 12))
    full, _ = model(ids)
    logits, caches = model(ids[:, :7])                 # prefill
    outs = [logits]
    for t in range(7, 12):                             # decode one token at a time
        l, caches = model(ids[:, t:t + 1], caches)
        outs.append(l)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)


def test_cached_and_uncached_generation_identical(model):
    a = generate(model, [1, 2, 3], 15, use_cache=True, temperature=0)
    b = generate(model, [1, 2, 3], 15, use_cache=False, temperature=0)
    assert a == b


def test_cache_shape_uses_kv_heads(model):
    _, c = model(torch.randint(0, 32, (1, 5)))
    assert c[0][0].shape == (1, 2, 5, 8)               # n_kv_head=2, head_dim=8


def test_top_k_property():
    l = torch.randn(100)
    out = top_k_filter(l, 7)
    assert torch.isfinite(out).sum() == 7
    assert set(torch.topk(l, 7).indices.tolist()) == set(torch.isfinite(out).nonzero().flatten().tolist())


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_properties(p):
    torch.manual_seed(1)
    l = torch.randn(50) * 2
    out = top_p_filter(l, p)
    keep = torch.isfinite(out)
    probs = l.softmax(-1)
    assert keep.sum() >= 1
    assert probs[keep].sum() >= p - 1e-6                         # nucleus covers >= p
    kept_sorted = probs[keep].sort().values
    assert probs[keep].sum() - kept_sorted[0] < p + 1e-6         # and is minimal (drop the smallest -> below p)
    assert probs[keep].min() >= probs[~keep].max() if (~keep).any() else True   # kept tokens are the most probable


def test_top_p_one_token_and_full():
    l = torch.tensor([10., 0., 0.])
    assert torch.isfinite(top_p_filter(l, 0.01)).sum() == 1
    assert torch.isfinite(top_p_filter(torch.randn(10), 1.0)).all()


def test_temperature_limits():
    l = torch.tensor([2., 1., 0.])
    assert apply_temperature(l, 1e-3).softmax(-1)[0] > 0.999
    assert apply_temperature(l, 1e3).softmax(-1).std() < 1e-3
    assert sample_next(l, temperature=0) == 0


def test_sampled_token_always_in_nucleus():
    torch.manual_seed(2)
    l = torch.randn(30)
    allowed = set(torch.isfinite(top_p_filter(top_k_filter(l, 8), 0.7)).nonzero().flatten().tolist())
    g = torch.Generator().manual_seed(3)
    for _ in range(200):
        assert sample_next(l, 1.0, top_k=8, top_p=0.7, generator=g) in allowed


def test_min_p_keeps_argmax():
    l = torch.randn(40)
    assert torch.isfinite(min_p_filter(l, 0.99))[l.argmax()]


def test_repetition_penalty_lowers_seen_tokens():
    l = torch.tensor([2., -2., 1.])
    out = repetition_penalty(l, [0, 1], penalty=2.0)
    assert out[0] < l[0] and out[1] < l[1] and out[2] == l[2]
    assert repetition_penalty(l, [0, 0], 1.0, freq=0.5)[0] == l[0] - 1.0


def test_stop_sequence():
    assert hit_stop_sequence([1, 2, 3, 4], [[3, 4]]) and not hit_stop_sequence([1, 2, 3], [[3, 4]])


def test_beam_search_score_at_least_greedy(model):
    greedy = generate(model, [1, 2], 6, temperature=0)
    beams = beam_search(model, [1, 2], 6, beams=4)

    def lp(ids):
        l = model(torch.tensor([ids]))[0][0].log_softmax(-1)
        return sum(float(l[i - 1, ids[i]]) for i in range(2, len(ids)))
    assert beams[0][1] >= lp(greedy) - 1e-5
    assert abs(beams[0][1] - lp(beams[0][0])) < 1e-4


@pytest.mark.parametrize("dim", [None, 0])
def test_int8_error_bound(dim):
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q, s = quantize_sym(w, 8, dim)
    err = (w - dequantize(q, s)).abs()
    assert (err <= s / 2 + 1e-7).all()               # rounding error <= half a step
    assert q.min() >= -128 and q.max() <= 127


def test_per_channel_beats_per_tensor_with_outlier_row():
    torch.manual_seed(0)
    w = torch.randn(32, 64); w[0] *= 50
    assert quant_error(w, 8, dim=0)["rmse"] < quant_error(w, 8)["rmse"]


def test_int4_worse_than_int8_and_group_helps():
    torch.manual_seed(0)
    w = torch.randn(64, 128); w[:, 0] *= 30
    assert quant_error(w, 4, group=32)["rmse"] > quant_error(w, 8, group=32)["rmse"]
    assert quant_error(w, 4, group=32)["rmse"] < quant_error(w, 4)["rmse"]


def test_quantized_model_logits_close(model):
    x = torch.randint(0, 32, (1, 10))
    ref = model(x)[0]
    qm = quantize_model(copy.deepcopy(model), 8)
    assert (qm(x)[0] - ref).abs().max() < 0.05


def test_kv_formula():
    assert kv_bytes(32, 8, 128, 4096, 1, "fp16") == 2 * 32 * 8 * 128 * 4096 * 2
    assert kv_bytes(32, 32, 128, 1, 1) == 4 * kv_bytes(32, 8, 128, 1, 1)      # GQA 4x saving
    assert kv_bytes_per_token(2, 2, 8, "fp32") == 2 * 2 * 2 * 8 * 4


def _reqs(seed, n=30):
    rnd = random.Random(seed)
    return [Req(i, rnd.randint(0, 40), rnd.randint(5, 50), rnd.randint(1, 60)) for i in range(n)]


@pytest.mark.parametrize("mode", ["static", "continuous"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("reserve", [True, False])
def test_scheduler_never_exceeds_capacity_and_finishes(mode, seed, reserve):
    cap, mb = 300, 6
    reqs = _reqs(seed)
    fin, rej, log = simulate(reqs, cap, mb, mode, reserve_full=reserve)
    if reserve:                                      # with full reservation actual KV can never exceed capacity
        assert all(kv <= cap for _, kv in log)
    assert all(len(ids) <= mb for ids, _ in log)
    assert len(fin) + len(rej) == len(reqs)
    assert all(r.produced == r.out for r in fin)
    assert all(r.rid in {x.rid for x in rej} for r in reqs if r.prompt + r.out > cap)


def test_continuous_beats_static():
    reqs = [Req(i, 0, 10, 5 if i % 2 else 60) for i in range(16)]
    fs, _, ls = simulate(reqs, 10_000, 4, "static")
    fc, _, lc = simulate(reqs, 10_000, 4, "continuous")
    assert metrics(fc, lc)["makespan"] < metrics(fs, ls)["makespan"]


def test_admission_control_sheds_load():
    reqs = [Req(i, 0, 10, 20) for i in range(20)]
    fin, rej, _ = simulate(reqs, 10_000, 2, "continuous", max_queue=3)
    assert len(rej) > 0 and len(fin) + len(rej) == 20
