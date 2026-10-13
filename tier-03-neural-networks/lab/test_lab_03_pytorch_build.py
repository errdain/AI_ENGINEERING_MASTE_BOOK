import pytest

torch = pytest.importorskip("torch")
import lab_03_pytorch_build as T


def test_numpy_and_pytorch_training_trajectories_are_identical():
    r = T.t_3_trajectory_match(); assert r["max_weight_difference_after_steps"] < 1e-12 and r["final_loss_numpy"] == pytest.approx(r["final_loss_torch"], abs=1e-12)


def test_autograd_graph_structure_and_values():
    r = T.t_3_autograd_graph(); assert r["grad_fn_chain"][0] == "TanhBackward0" and r["grad_fn_chain"][-1] == "AccumulateGrad" and r["is_leaf(x,w,z)"] == (True, True, False)
    assert r["dx"] == pytest.approx(r["manual_dx"], abs=1e-12) and r["result_under_no_grad_requires_grad"] is False


def test_standard_pytorch_loop_learns_circle():
    r = T.t_3_standard_training_loop(); assert r["params"] == 82 and r["loss_start_end"][1] < 0.1 and r["train_acc"] > 0.98


def test_layers_match_torch_modules():
    r = T.t_3_layers_vs_torch(); assert r["layernorm"] < 1e-12 and r["rmsnorm"] < 1e-12 and r["dropout_eval_identity"] and r["param_counts"] == {"LayerNorm(6)": 12, "RMSNorm(6)": 6, "BatchNorm1d(6)": 12}


def test_init_helpers_match_formulas():
    r = T.t_3_init_helpers()
    a, b = r["xavier_normal_std"]; assert a == pytest.approx(b, rel=0.02)
    a, b = r["kaiming_normal_std(relu)"]; assert a == pytest.approx(b, rel=0.02)
    a, b = r["xavier_uniform_bound"]; assert a <= b + 1e-12 and a == pytest.approx(b, rel=0.01)
    g = r["calculate_gain"]; assert g["tanh"] == pytest.approx(5 / 3) and g["relu"] == pytest.approx(2 ** 0.5) and g["sigmoid"] == 1


def test_clip_grad_norm_matches_torch():
    r = T.t_3_clip_grad_matches(); assert r["norm_before"][0] == pytest.approx(r["norm_before"][1]) and r["max_grad_difference_after_clip"] < 1e-6
