import warnings

import numpy as np
import pytest

import lab_02_training_internals as L

warnings.simplefilter("ignore", RuntimeWarning)


def test_forward_propagation_hand_trace():
    r = L.ex_3_18(); assert r["z1"] == [[-1.5, 3.2, 1.1]] and r["a1"] == [[0.0, 3.2, 1.1]] and r["z2"][0][0] == pytest.approx(-2.55) and r["library_output"][0][0] == pytest.approx(-2.55)
    assert r["activation_memory_floats_for_batch32(784-128-64-10)"] == 31552 and r["flops_per_sample"] == 218368 and r["deterministic"]


def test_computational_graph_and_gradient_accumulation():
    r = L.ex_3_19(); assert r["f"] == -8 and (r["df/da"], r["df/db"], r["df/dc"]) == (6, -4, -2) and r["shared_node_gradient(x^2+2x at 3)"] == 8


def test_backprop_matches_finite_differences_and_cost_argument():
    r = L.ex_3_20(); assert r["max_error_vs_finite_differences"] < 1e-8 and r["forward_passes_needed_by_finite_differences"] == 218772


def test_autodiff_properties():
    r = L.ex_3_21(); assert r["reverse_mode_max_error_1000_inputs"] < 1e-12 and r["gradient_through_round"] == 0
    e = r["numerical_error_by_h(central)"]; assert e[1e-5] < e[1e-3] < e[1e-1] and e[1e-12] > e[1e-5]               # truncation vs round-off sweet spot
    assert r["branch_gradients(relu or 0.1x)"] == (1.0, 0.1)


def test_initialization_scale_matters():
    s = L.ex_3_22()["activation_std_layers_1_10_30"]
    assert s["zeros"][2] == 0 and s["normal_0.01"][2] < 1e-6 and s["normal_1"][2] > 0.9 and s["xavier_normal"][2] > 0.05 and s["he_normal"][2] > 0.3
    assert L.ex_3_22()["constant_init_rows_identical_after_training"]                                                  # symmetry never breaks


def test_xavier_and_he_variance_preservation():
    x = L.ex_3_23()["tanh_30_layers_std"]; assert x["normal_0.01"] < 1e-6 and 0.05 < x["xavier_normal"] < 0.3 and x["xavier_uniform"] == pytest.approx(x["xavier_normal"], rel=0.2)
    h = L.ex_3_24()
    r = h["relu_30_layers_final_std"]; assert 0.2 < r["he_normal"] < 2 and r["xavier_normal"] < 1e-3 and r["normal_0.01"] < 1e-20
    assert r["xavier_normal"] == pytest.approx(h["predicted_decay_xavier_on_relu"], rel=1.0)                         # within an order of magnitude of 0.5^(L/2)


def test_vanishing_gradients_and_fixes():
    r = L.ex_3_25()["first_over_last_layer_gradient_norm"]
    assert r["sigmoid_xavier_20L"] < 1e-10 and r["sigmoid_xavier_20L+residual"] > 1e6 * r["sigmoid_xavier_20L"] and r["sigmoid_xavier_20L+layernorm"] > 1e3 * r["sigmoid_xavier_20L"]
    assert 0.01 < r["relu_he_20L"] < 100


def test_exploding_gradients_and_clipping():
    r = L.ex_3_26(); g = r["global_gradient_norm_at_init_by_weight_gain(20 layers)"]
    assert g[1.0] < g[1.25] < g[1.5] < g[2.0] and g[2.0] > 1e5 * g[1.0]
    assert r["gain1.3_no_clip"]["final_loss"] == float("inf") and np.isfinite(r["gain1.3_clip_norm1"]["final_loss"]) and r["gain1.3_clip_norm1"]["max_grad_norm_before_clip"] > 10
    assert r["stable_final_loss(gain=1.0)"] < 1.0


def test_residual_connections_restore_gradient_flow_and_training():
    r = L.ex_3_27(); p, s = r["plain_30L"], r["residual_30L"]
    assert s["first_layer_grad_norm"] > 10 * p["first_layer_grad_norm"] and s["loss_after_150_steps"] < 0.1 and p["loss_after_150_steps"] > 0.9 * r["chance_loss"]


def test_dropout_behaviour():
    r = L.ex_3_28(); assert r["expected_activation_preserved(mean over masks)"] == pytest.approx(1.0, abs=0.02) and r["train_mode_outputs_vary"] and r["eval_mode_outputs_identical"]
    assert r["test_acc_mean"]["dropout0.5"] >= r["test_acc_mean"]["none"] - 0.01                                      # helps or at least does not hurt (effect is small here)


def test_weight_decay_shrinks_weights_and_helps_on_small_data():
    r = L.ex_3_29()["by_weight_decay"]; assert r[0.1]["weight_norm"] < 0.3 * r[0.0]["weight_norm"] and r[0.1]["test_acc"] > r[0.0]["test_acc"] + 0.01
    assert L.ex_3_29()["after_100_steps_without_gradient"] == pytest.approx(0.999 ** 100)


def test_batchnorm_properties():
    r = L.ex_3_30(); s = r["output_of_one_sample_std_across_random_batches_by_batch_size"]
    assert r["running_mean_error_after_40_passes"] < 0.3 and abs(r["eval_output_mean_std"][0]) < 0.05 and r["eval_output_mean_std"][1] == pytest.approx(1.0, abs=0.05)
    assert s[2] > s[8] > s[32] > s[128] and s[2] / s[128] > 5 and r["batch_of_one_output_all_zero"]
    assert r["final_loss_lr0.5"]["no_norm"] == float("inf") and r["final_loss_lr0.5"]["batchnorm"] < 0.1


def test_layernorm_properties():
    r = L.ex_3_31(); assert r["sample_independent_of_batch"] and r["works_for_batch_of_one"] and r["sequence_input_shape"] == (2, 5, 16) and r["no_running_stats_needed"]
    assert r["per_sample_mean_std"][0] < 1e-12 and r["per_sample_mean_std"][1] < 1e-4 and r["batchnorm_train_vs_eval_output_gap_on_same_input"] > 0.5


def test_rmsnorm_properties():
    r = L.ex_3_32(); assert r["scale_invariance_rms(x vs 10x)"] < 1e-5 and r["shift_invariance_layernorm"] < 1e-10 and r["shift_invariance_rmsnorm(not invariant)"] > 1
    assert r["max_diff_when_inputs_have_mean_5"] > 10 * r["max_diff_when_inputs_are_zero_mean"] and r["params_ln_vs_rms"] == (2048, 1024) and r["time_ratio_ln_over_rms"] > 1.2
