import pytest

torch = pytest.importorskip("torch")
import lab_04_pytorch as T


def test_tensor_basics_and_views():
    r = T.t_1_4_tensor_basics(); assert r["shape"] == (32, 128, 768) and r["MiB"] == 12.0
    v = T.t_1_12_transpose_view()
    assert v["is_view"] and not v["contiguous"] and not v["view_after_transpose_ok"] and v["reshape_ok"] and v["contiguous_then_view_ok"]


def test_autograd_matches_manual_backprop():
    r = T.t_1_25_autograd_vs_manual(); assert r["dW1_max_diff"] < 1e-12 and r["dW2_max_diff"] < 1e-12


def test_variance_default_differs_between_numpy_and_torch():
    r = T.t_1_30_var_defaults()
    assert r["numpy_default"] == pytest.approx(4.0) and r["torch_default"] == pytest.approx(32 / 7) and r["torch_correction0"] == pytest.approx(4.0)


def test_cross_entropy_and_kl_conventions():
    r = T.t_1_37_cross_entropy_pitfall()
    assert r["F.cross_entropy"] == pytest.approx(r["manual"], abs=1e-5)
    assert r["double_softmax"] != pytest.approx(r["F.cross_entropy"], abs=1e-3)  # passing probabilities is silently wrong
    k = T.t_1_38_kl_convention(); assert k["F.kl_div(log q, p)"] == pytest.approx(k["manual_nats"], abs=1e-6)


def test_dtype_table_matches_concept_file():
    t = T.t_1_40_dtype_table()
    assert t["torch.float16"]["max"] == 65504 and t["torch.float16"]["eps"] == pytest.approx(9.765625e-4)
    assert t["torch.bfloat16"]["eps"] == pytest.approx(0.0078125) and t["torch.bfloat16"]["max"] == pytest.approx(3.3895e38, rel=1e-3)
    assert t["torch.float32"]["max"] == pytest.approx(3.4028e38, rel=1e-3)
    b = T.t_1_40_bf16_vs_fp16()
    assert b["fp16_70000"] == float("inf") and b["bf16_70000"] > 6e4  # bf16 keeps range
    assert b["fp16_1.001"] != 1.0 and b["bf16_1.001"] == 1.0  # ...but loses precision


def test_logsumexp_and_numerical_failure_modes():
    r = T.t_1_43_logsumexp(); assert r["torch"] == pytest.approx(r["ours"])
    s = T.t_1_44_loss_scaling()
    assert s["fp16_unscaled"] == 0.0 and s["recovered"] == pytest.approx(1e-8, rel=0.01)
    assert T.t_1_44_int8_wrap()["int8_127_plus_1"] == -128
    n = T.t_1_44_nan_propagation()
    assert n["inf-inf"] != n["inf-inf"] and n["naive_softmax_nan"] and n["stable_softmax_finite"]


def test_safe_checkpoint_loading_behaviour():
    r = T.t_1_4_safe_load()
    assert r["tensors_ok"] and r["object_blocked_with_weights_only"]
    # default_blocks_objects is version-dependent (True on recent PyTorch releases); recorded, not asserted
    assert isinstance(r["default_blocks_objects"], bool)
