import json

import numpy as np
import pytest
import torch

import data_prep as dp
import run_lab
from model import ModelConfig, MoEFFN, TinyLM
from pretrain import TrainConfig, flops_6nd, get_batch, lr_at, train
from tok_bpe import ByteBPE


@pytest.fixture(scope="module")
def prepared():
    return run_lab.prepare(seed=0, n_docs=300, merges=60)


def small_cfg(tok, steps=30, **kw):
    return TrainConfig(steps=steps, eval_every=10, ckpt_every=10, eval_batches=2, batch_size=8,
                       model=ModelConfig(vocab_size=tok.vocab_size, ctx=32, d_model=32, n_layer=1, n_head=2, **kw))


# ---------------- data pipeline ----------------
def test_clean_text_strips_html_and_normalises():
    t = dp.clean_text("<p>Café ​ok</p><script>x=1</script>")
    assert t == "Café ok"


def test_quality_filter_reasons():
    assert dp.quality_filter("ok")[1] == "too_short"
    assert dp.quality_filter("### $$$ %%% ^^^ &&& *** ((( ))) !!! @@@ ###")[1] in ("symbol_ratio", "too_short")
    assert dp.quality_filter("the cat sees a red boat in the morning today ok")[0]


def test_exact_dedup_ignores_case_and_whitespace():
    out, n = dp.exact_dedup([{"text": "Hello  World"}, {"text": "hello world"}, {"text": "other"}])
    assert n == 1 and len(out) == 2


def test_minhash_estimates_jaccard():
    a = "the quick brown fox jumps over the lazy dog near the old river bank today " * 3
    b = a.replace("fox", "cat")
    ja = dp.jaccard(dp.shingles(a), dp.shingles(b))
    est = dp.estimated_jaccard(dp.minhash_signature(dp.shingles(a), 256), dp.minhash_signature(dp.shingles(b), 256))
    assert abs(ja - est) < 0.15
    assert dp.estimated_jaccard(dp.minhash_signature(dp.shingles(a)), dp.minhash_signature(dp.shingles(a))) == 1.0


def test_near_dedup_removes_near_copies_keeps_distinct():
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    docs = [{"text": base}, {"text": base.replace("pi", "PI")}, {"text": "completely different words appear in this sentence about boats rivers and bridges"}]
    out, dropped = dp.near_dedup(docs, 0.6)
    assert dropped == 1 and len(out) == 2


def test_contamination_detect_and_remove():
    bench = ["one two three four five six seven eight nine ten"]
    docs = [{"text": "x " + bench[0] + " y"}, {"text": "unrelated text about something else entirely here with many words"}]
    assert dp.contamination_rate(docs, bench) == 1.0
    out, n = dp.decontaminate(docs, bench)
    assert n == 1 and dp.contamination_rate(out, bench) == 0.0


def test_pipeline_funnel_monotonic_and_removes_leak(prepared):
    _, _, _, rep = prepared
    counts = [c for _, c in rep["stages"]]
    assert counts == sorted(counts, reverse=True) and counts[-1] < counts[0]
    assert rep["contamination_before"] == 1.0 and rep["contamination_after"] == 0.0
    assert rep["dropped"]["license_filter"] > 0 and rep["dropped"]["near_dedup"] > 0


def test_mixture_follows_weights():
    docs = [{"source": s, "text": f"{s}{i}"} for s in "abc" for i in range(200)]
    m = dp.mix_sources(docs, {"a": 0.5, "b": 0.3, "c": 0.2}, 100, seed=1)
    c = {s: sum(d["source"] == s for d in m) for s in "abc"}
    assert c == {"a": 50, "b": 30, "c": 20}


def test_split_is_stable_and_disjoint(prepared):
    docs = dp.make_synthetic_crawl(1, 100)
    a1, b1 = dp.split_train_val(docs); a2, b2 = dp.split_train_val(list(reversed(docs)))
    assert {d["id"] for d in b1} == {d["id"] for d in b2} and not ({d["id"] for d in a1} & {d["id"] for d in b1})


def test_tokenizer_roundtrip_and_stream(prepared):
    tok, tr, va, rep = prepared
    s = "the old sailor 🙂 naïve"
    assert tok.decode(tok.encode(s)) == s
    assert tr.dtype == np.uint16 and int(tr.max()) < tok.vocab_size and rep["train_tokens"] == len(tr)


# ---------------- model ----------------
def test_model_shapes_and_initial_loss():
    c = ModelConfig(vocab_size=100, ctx=16, d_model=32, n_layer=1, n_head=2)
    m = TinyLM(c); x = torch.randint(0, 100, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 100)
    assert abs(loss.item() - np.log(100)) < 0.5  # untrained ~ ln(V)
    assert m.head.weight is m.tok.weight          # tied


def test_causality_future_tokens_do_not_change_past_logits():
    m = TinyLM(ModelConfig(vocab_size=50, ctx=16, d_model=32, n_layer=2, n_head=2)).eval()
    a = torch.randint(0, 50, (1, 16)); b = a.clone(); b[0, 10:] = (b[0, 10:] + 1) % 50
    la, _ = m(a); lb, _ = m(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)


def test_moe_router_topk_and_balance_loss():
    torch.manual_seed(0)
    moe = MoEFFN(16, 4, 2)
    y = moe(torch.randn(2, 32, 16))
    assert y.shape == (2, 32, 16)
    assert abs(float(moe.last_load.sum()) - 2.0) < 1e-5     # top-2 => 2 expert slots per token
    assert float(moe.aux_loss) >= 1.0 - 1e-5                 # E*sum f*P >= 1, equality when balanced
    # a collapsed router (all tokens -> expert 0) must score worse than the random one
    with torch.no_grad():
        moe.router.weight.zero_(); moe.router.weight[0] += 5.0
    moe(torch.ones(1, 32, 16)); assert float(moe.aux_loss) > 2.0


# ---------------- training / reproducibility ----------------
def test_lr_schedule_shape():
    c = TrainConfig(steps=100, warmup=10, lr=1.0, min_lr_frac=0.1)
    assert lr_at(0, c) < lr_at(9, c) <= 1.0 and abs(lr_at(10, c) - 1.0) < 1e-6 and abs(lr_at(100, c) - 0.1) < 1e-6


def test_batches_are_pure_function_of_seed_and_step(prepared):
    _, tr, _, _ = prepared
    x1, y1 = get_batch(tr, 16, 4, 7, 3); x2, y2 = get_batch(tr, 16, 4, 7, 3); x3, _ = get_batch(tr, 16, 4, 7, 4)
    assert torch.equal(x1, x2) and not torch.equal(x1, x3) and torch.equal(x1[:, 1:], y1[:, :-1])


def test_loss_decreases_and_metrics_json(prepared, tmp_path):
    tok, tr, va, _ = prepared
    m = train(small_cfg(tok, 60), tr, va, tmp_path)
    h = m["history"]["train"]
    assert np.mean([r["loss"] for r in h[-5:]]) < np.mean([r["loss"] for r in h[:5]]) - 0.5
    j = json.loads((tmp_path / "metrics.json").read_text())
    assert j["steps_done"] == 60 and j["train_flops_6nd"] == flops_6nd(j["n_params"], j["tokens_seen"])
    assert j["final_val"]["val_ppl"] > 1 and (tmp_path / "latest.pt").exists()


def test_same_seed_same_result(prepared, tmp_path):
    tok, tr, va, _ = prepared
    a = train(small_cfg(tok, 15), tr, va, tmp_path / "a"); b = train(small_cfg(tok, 15), tr, va, tmp_path / "b")
    assert a["final_train_loss"] == b["final_train_loss"]
    c = small_cfg(tok, 15); c.seed = 99
    assert train(c, tr, va, tmp_path / "c")["final_train_loss"] != a["final_train_loss"]


def test_resume_matches_uninterrupted_run(prepared, tmp_path):
    tok, tr, va, _ = prepared
    full = train(small_cfg(tok, 30), tr, va, tmp_path / "full")
    train(small_cfg(tok, 30), tr, va, tmp_path / "res", stop_at=20)           # "crash" at step 20
    res = train(small_cfg(tok, 30), tr, va, tmp_path / "res", resume=True)    # continue to 30
    assert res["steps_done"] == 30 and any(e["event"] == "resumed" for e in res["history"]["events"])
    assert abs(res["final_train_loss"] - full["final_train_loss"]) < 1e-5
    s1 = torch.load(tmp_path / "full" / "latest.pt", weights_only=False)["model"]
    s2 = torch.load(tmp_path / "res" / "latest.pt", weights_only=False)["model"]
    assert all(torch.allclose(s1[k], s2[k], atol=1e-5) for k in s1)


def test_moe_model_trains(prepared, tmp_path):
    tok, tr, va, _ = prepared
    m = train(small_cfg(tok, 40, moe_experts=4), tr, va, tmp_path)
    h = m["history"]["train"]
    assert h[-1]["loss"] < h[0]["loss"] and m["n_params"] > 0


def test_6nd_rule():
    assert flops_6nd(1_000_000_000, 20_000_000_000) == 1.2e20


# ---------------- scaling laws ----------------
def test_compute_optimal_respects_budget_and_is_a_minimum():
    from scaling import compute_optimal, loss
    C = 5.76e23
    N, D = compute_optimal(C)
    assert abs(6 * N * D / C - 1) < 1e-9
    for f in (0.5, 0.8, 1.25, 2.0):                      # any other split of the same budget is worse
        n2 = N * f; d2 = C / (6 * n2)
        assert loss(n2, d2) > loss(N, D)


def test_fit_power_law_recovers_parameters():
    from scaling import fit_power_law
    N = np.array([1e5, 3e5, 1e6, 3e6, 1e7]); L = 1.5 + 50.0 / N ** 0.3
    f = fit_power_law(N, L)
    assert abs(f["alpha"] - 0.3) < 0.03 and abs(f["E"] - 1.5) < 0.1
