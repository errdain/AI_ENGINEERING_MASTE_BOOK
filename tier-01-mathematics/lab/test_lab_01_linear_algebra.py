import numpy as np
import pytest

import lab_01_linear_algebra as L


def test_scalars_vectors_matrices_tensors():
    assert L.ex_1_1() == {"a_shape": (), "b_shape": (1,), "sum_shape": (1,)}
    r = L.ex_1_2()
    assert r["float32_GB"] == pytest.approx(3.072) and r["float16_GB"] == pytest.approx(1.536)
    assert L.ex_1_3()["actual"] == (3,)
    assert L.ex_1_4() == {"float32_MiB": 128.0, "float16_MiB": 64.0}


def test_attention_shapes_and_softmax_axis():
    assert L.ex_1_5() == {"out_shape": (2, 5, 8), "rows_sum_to_one": True}
    r = L.ex_1_6()
    assert r["axis_last_ok"] and not r["axis_first_ok"]


def test_broadcasting_bug_and_guard():
    r = L.ex_1_7(B=8)
    assert r["buggy_shape"] == (8, 8) and r["guard_raises"]
    assert r["buggy_mse"] != pytest.approx(r["correct_mse"])


def test_vector_ops_and_dot_variance():
    r = L.ex_1_8(); assert r["commutative"] and r["associative"]
    t = L.ex_1_9()
    assert t[0.25].max() > t[1.0].max() > t[4.0].max()  # lower T = peakier
    for d, (raw, scaled) in L.ex_1_10().items():
        assert raw == pytest.approx(d, rel=0.1) and scaled == pytest.approx(1.0, rel=0.1)


def test_matmul_transpose_identity():
    r = L.ex_1_11(sizes=(16, 48))
    assert all(v["match"] for v in r.values())
    assert r[48]["slowdown"] > 5  # timing is noisy: only the largest size is asserted, with a loose bound
    r = L.ex_1_12()
    assert r["A_contig"] and not r["AT_contig"] and r["ascontig_contig"] and r["shares_memory"]
    r = L.ex_1_13()
    assert r["rank_before"] == 2 and r["full_rank_after"] and r["cond_after"] < r["cond_before"] * 1e-6


def test_inverse_conditioning_amplifies_perturbation():
    r = L.ex_1_14()
    assert r["cond"] == pytest.approx(1e10, rel=0.05)
    assert r["amplification"] > 1e9  # a 1e-8 change in b moves x by ~100


def test_norms_clipping_and_distances():
    r = L.ex_1_15()
    assert r["before"] > 5 and r["after"] == pytest.approx(5.0, abs=1e-6) and r["small_unchanged"]
    d = L.ex_1_16()
    assert d[2] > d[20] > d[500]  # relative contrast shrinks with dimension


def test_cosine_dot_l2_rankings():
    r = L.ex_1_17()
    assert r["cos"] == r["unit_dot"] == r["unit_l2"]  # identical on unit vectors
    assert r["dot"] != r["cos"]  # norms change the dot-product ranking


def test_eigen_and_svd_labs():
    r = L.ex_1_18()
    assert r["major_axis_angle_deg"] == pytest.approx(30.0, abs=2.0)
    assert r["explained_variance_ratio"][0] > 0.95
    s = L.ex_1_19()
    assert s["k95"] == 5 and s["errors"][5] < s["errors"][1] and s["errors"][20] < s["errors"][5]
