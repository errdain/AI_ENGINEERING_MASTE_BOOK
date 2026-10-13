"""pytest for Tier 15. Metric formulas are hand-checked against small worked examples."""
import json, math
import numpy as np
import pytest
from metrics import *
from toy import *
from evalharness import *
from demo import make_ds, model_v1, model_v2


# ----- hand-checked metric formulas -----
def test_accuracy_hand():
    assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75

def test_prf1_hand():
    # tp=2 fp=1 fn=1 tn=1 -> P=2/3 R=2/3 F1=2/3
    y, p = [1, 1, 1, 0, 0], [1, 1, 0, 1, 0]
    m = prf1(y, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)

def test_f1_is_harmonic_not_arithmetic():
    # 1 TP, 0 FP, 9 FN -> P=1, R=0.1, F1=2*0.1/1.1
    m = prf1([1] * 10, [1] + [0] * 9)
    assert m["f1"] == pytest.approx(0.2 / 1.1) and m["f1"] < 0.55

def test_prf1_zero_division():
    assert prf1([0, 0], [0, 0])["f1"] == 0.0

def test_perplexity_hand():
    assert perplexity_from_probs([0.5, 0.5, 0.5]) == pytest.approx(2.0)      # uniform over 2 -> PPL 2
    assert perplexity_from_probs([1.0, 1.0]) == pytest.approx(1.0)
    assert perplexity_from_probs([0.25, 1.0]) == pytest.approx(2.0)           # geometric mean of 1/p = sqrt(4)

def test_uniform_model_ppl_equals_vocab():
    V = 50
    assert perplexity([math.log(V)] * 10) == pytest.approx(V)

def test_bigram_lm_memorised_text_has_lower_ppl():
    lm = BigramLM("the cat sat on the mat the cat ran")
    assert perplexity(lm.token_nlls("the cat sat")) < perplexity(lm.token_nlls("mat the on ran sat"))

# ----- calibration -----
def test_ece_hand():
    # bin 0.9-1.0: conf .9,.9 acc 1 -> gap .1 ; bin .5-.6: conf .6 acc 0 -> gap .6 ; ECE=(2/3)*.1+(1/3)*.6
    conf, ok = [0.9, 0.9, 0.6], [1, 1, 0]
    assert ece(conf, ok, 10) == pytest.approx(2 / 3 * 0.1 + 1 / 3 * 0.6)

def test_ece_perfect_zero_and_bounds():
    rng = np.random.default_rng(0)
    c = rng.random(200_000); ok = rng.random(200_000) < c          # correct with prob == confidence
    assert ece(c, ok) < 0.01
    assert 0 <= ece([1.0] * 5, [0] * 5) <= 1 and ece([1.0] * 5, [0] * 5) == pytest.approx(1.0)

def test_confidence_one_edge_lands_in_last_bin():
    assert reliability_bins([1.0], [1], 10)[-1][2] == 1

def test_overconfident_model_and_temperature_scaling():
    lg, y = make_classifier_logits(4000, signal=1.5, sharpen=3.0, seed=1)
    p = softmax(lg); conf = p.max(1); ok = p.argmax(1) == y
    e0 = ece(conf, ok)
    T = fit_temperature(lg, y)
    p2 = softmax(lg, T); e1 = ece(p2.max(1), p2.argmax(1) == y)
    assert T > 1.5 and e1 < e0 / 2
    assert (p2.argmax(1) == p.argmax(1)).all()                     # accuracy unchanged by T

def test_brier_hand():
    assert brier([[1, 0], [0.5, 0.5]], [[1, 0], [1, 0]]) == pytest.approx((0 + 0.5) / 2)

# ----- bootstrap -----
def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = (rng.random(100) < .7).astype(float); big = (rng.random(4000) < .7).astype(float)
    lo, hi = bootstrap_ci(small); lo2, hi2 = bootstrap_ci(big)
    assert lo < small.mean() < hi and (hi2 - lo2) < (hi - lo) / 3

def test_bootstrap_ci_coverage():
    """~95% of CIs from repeated samples of a p=0.6 benchmark should contain the truth (allow 88-100%)."""
    rng = np.random.default_rng(42); hits = 0; R = 200
    for r in range(R):
        x = (rng.random(150) < 0.6).astype(float)
        lo, hi = bootstrap_ci(x, n_boot=300, seed=r)
        hits += lo <= 0.6 <= hi
    assert 0.88 <= hits / R <= 1.0

def test_wilson_matches_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)

def test_paired_bootstrap_detects_real_gap_not_noise():
    rng = np.random.default_rng(1)
    base = (rng.random(500) < .6).astype(float)
    better = np.where(rng.random(500) < .15, 1.0, base)             # strictly >= base
    assert paired_bootstrap(better, base)["ci_low"] > 0
    same = paired_bootstrap(base, base.copy())
    assert same["diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]

# ----- kappa, fairness, contamination -----
def test_kappa_hand():
    # po=.8, pe=.5*.5+.5*.5=.5 -> kappa=.6
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]; b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert cohen_kappa(a, b) == pytest.approx(0.6)
    assert cohen_kappa(a, a) == 1.0

def test_kappa_chance_agreement_is_zero():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

def test_fairness_hand():
    y = [1, 1, 0, 0, 1, 1, 0, 0]; p = [1, 1, 1, 0, 1, 0, 0, 0]; g = ["a"] * 4 + ["b"] * 4
    r = group_rates(y, p, g)
    assert r["a"]["selection"] == .75 and r["b"]["selection"] == .25
    assert demographic_parity_diff(p, g) == pytest.approx(.5)
    # TPR a=1 b=.5 ; FPR a=.5 b=0 -> max gap .5
    assert equalized_odds_diff(y, p, g) == pytest.approx(.5)

def test_parity_and_equalized_odds_can_disagree():
    # base rates differ (a: 50% positive, b: 10%); a perfect classifier has EO diff 0 but DP diff > 0
    y = np.array([1] * 50 + [0] * 50 + [1] * 10 + [0] * 90); g = np.array(["a"] * 100 + ["b"] * 100)
    assert equalized_odds_diff(y, y, g) == 0 and demographic_parity_diff(y, g) == pytest.approx(.4)

def test_ngram_overlap():
    train = ["the quick brown fox jumps over the lazy dog"]
    assert ngram_overlap("the quick brown fox jumps", train, 4) == 1.0
    assert ngram_overlap("a totally different sentence here", train, 4) == 0.0

# ----- judge bias -----
def test_unbiased_judge_is_swap_consistent_and_biased_one_is_not():
    c0, f0 = swap_consistency(MockJudge(0, 0, seed=1), 0.9, 0.3, 100, 100, n=1500)
    c1, f1 = swap_consistency(MockJudge(2.0, 0, seed=1), 0.6, 0.55, 100, 100, n=1500)
    assert f0 == pytest.approx(0.5, abs=0.05) and f1 > 0.7 and c1 < c0

def test_verbosity_bias_flips_close_pairs():
    j = MockJudge(0, 3.0, seed=2)
    assert j.p_first(0.5, 0.5, 300, 100) > 0.99 > j.p_first(0.5, 0.5, 100, 300) + .98

# ----- harness: datasets, reproducibility, versioning -----
def test_dataset_version_changes_with_content_not_order():
    ds = make_ds(); v = ds.version
    ds.examples.reverse(); assert ds.version == v
    ds.examples[0].reference += "!"; assert ds.version != v

def test_dataset_roundtrip(tmp_path):
    ds = make_ds(); ds.save(tmp_path / "d.jsonl")
    ds2 = Dataset.load(tmp_path / "d.jsonl", ds.name, ds.kind)
    assert ds2.version == ds.version

def test_regression_dataset_requires_unique_id_and_source():
    ds = make_ds(); ds.add_regression("r1", "what is 7+7?", "14", source="incident-1234", tags=["math"])
    assert ds.examples[-1].source == "incident-1234"
    with pytest.raises(ValueError): ds.add_regression("r1", "x", "y", "z")

def test_reproducible_same_seed_same_run_id():
    ds = make_ds()
    a = run_eval(model_v1, "v1", ds, seed=3); b = run_eval(model_v1, "v1", ds, seed=3)
    assert a.run_id == b.run_id and a.to_json() == b.to_json()

def test_stochastic_model_is_seeded_per_example():
    import random
    def flaky(x): return "ok" if random.random() < 0.5 else "bad"
    ds = Dataset("t", "golden", [Example(f"e{i}", "x", "ok") for i in range(50)])
    a = run_eval(flaky, "f", ds, seed=7)
    ds.examples.reverse(); b = run_eval(flaky, "f", ds, seed=7)
    assert a.scores == b.scores                                  # order-independent
    assert run_eval(flaky, "f", ds, seed=8).scores != a.scores   # other seed -> other draw

def test_result_json_roundtrip_and_versioned_filename(tmp_path):
    r = run_eval(model_v1, "v1", make_ds(), seed=1)
    p = r.save(tmp_path)
    assert r.run_id in p.name
    r2 = EvalResult.from_json(p.read_text())
    assert r2.run_id == r.run_id and json.loads(p.read_text())["harness_version"] == HARNESS_VERSION

def test_run_id_changes_when_model_output_changes():
    ds = make_ds()
    assert run_eval(model_v1, "m", ds).run_id != run_eval(model_v2, "m", ds).run_id

def test_compare_refuses_different_dataset_versions():
    ds = make_ds(); a = run_eval(model_v1, "a", ds)
    ds.examples[0].reference = "changed"; b = run_eval(model_v1, "b", ds)
    with pytest.raises(ValueError): compare(a, b)

def test_slice_metrics_present():
    r = run_eval(model_v1, "v1", make_ds())
    assert r.metrics["slice:safety"] == 1.0 and r.metrics["slice:math"] < 1.0
    assert r.metrics["ci_low"] <= r.metrics["mean"] <= r.metrics["ci_high"]

# ----- human review -----
def test_human_review_import():
    r = run_eval(model_v1, "v1", make_ds())
    rows = []
    for i in ["q0", "q1", "q7", "q14"]:                      # q7, q14 are wrong (multiples of 7)
        rows += [{"example_id": i, "rater": "ann", "label": int(r.scores[i])},
                 {"example_id": i, "rater": "bob", "label": int(r.scores[i])}]
    rows += [{"example_id": "q7", "rater": "ann", "label": 1}, {"example_id": "q7", "rater": "bob", "label": 1}]  # humans override auto
    import_human_reviews(r, rows)
    assert r.human["n_reviewed"] == 4 and r.human["auto_vs_human_disagreements"] == ["q7"]
    assert r.human["kappa"] is not None
    with pytest.raises(KeyError): import_human_reviews(r, [{"example_id": "nope", "rater": "a", "label": 1}])

# ----- gates -----
def _res(mean, safety=1.0, ci_low=None):
    return EvalResult("m", "d", "v", "exact_match", 0, "1", {"a": 1.0}, {"a": []},
                      {"mean": mean, "slice:safety": safety, "ci_low": mean - .05 if ci_low is None else ci_low, "n": 1})

def test_gate_pass_and_fail_with_reasons():
    gates = [Gate("acc", "mean", min=0.8), Gate("safety", "slice:safety", min=0.99)]
    assert evaluate_gates(_res(0.85), gates).passed
    rep = evaluate_gates(_res(0.85, safety=0.95), gates)
    assert not rep.passed and "safety" in rep.reasons[0] and "0.950" in rep.reasons[0]

def test_gate_uses_ci_lower_bound_conservatively():
    g = [Gate("acc-lcb", "ci_low", min=0.8)]
    assert not evaluate_gates(_res(0.82), g).passed and evaluate_gates(_res(0.9), g).passed

def test_soft_gate_warns_but_passes():
    rep = evaluate_gates(_res(0.5), [Gate("nice", "mean", min=0.8, hard=False)])
    assert rep.passed and rep.warnings

def test_missing_metric_fails_closed():
    assert not evaluate_gates(_res(0.9), [Gate("x", "slice:privacy", min=0.5)]).passed

def test_regression_gate_needs_baseline_and_threshold():
    g = [Gate("reg", "mean", max_regression=0.02)]
    assert not evaluate_gates(_res(0.9), g).passed                       # no baseline -> fail closed
    assert evaluate_gates(_res(0.89), g, baseline=_res(0.90)).passed     # drop .01 ok
    assert not evaluate_gates(_res(0.85), g, baseline=_res(0.90)).passed

def test_max_gate():
    assert not evaluate_gates(_res(0.5), [Gate("harm", "mean", max=0.3)]).passed

def test_significant_regression_gate_ignores_noise():
    ds = make_ds()
    a = run_eval(model_v2, "base", ds); b = run_eval(model_v1, "cand", ds)
    g = [Gate("reg", "mean", max_regression=0.02, require_significant=True)]
    assert not evaluate_gates(b, g, baseline=a).passed                   # real drop (CI excludes 0)
    assert evaluate_gates(a, g, baseline=a).passed
