import numpy as np
import pytest

import lab_01_neurons_and_mlp as L


def test_single_neuron_logic_gates_and_two_layer_xor():
    r = L.ex_3_1(); assert r["AND"] == [0, 0, 0, 1] and r["OR"] == [0, 1, 1, 1] and r["NAND"] == [1, 1, 1, 0] and r["XOR_with_two_layers"] == [0, 1, 1, 0]


def test_perceptron_convergence_and_xor_failure():
    r = L.ex_3_2(); lg = r["logic"]
    assert lg["AND"]["converged"] and lg["OR"]["converged"] and not lg["XOR"]["converged"] and lg["XOR"]["mistakes"] > 100
    for m, v in r["mistakes_vs_margin"].items(): assert v["converged"] and v["mistakes"] <= v["novikoff_bound"] + 1


def test_shapes_and_parameter_counts():
    r = L.ex_3_3_to_3_5()
    assert r["shapes"][0] == (32, 784) and r["shapes"][-1] == (32, 10) and r["params_per_dense_layer"] == [100480, 8256, 650] and r["total_params"] == 109386
    assert r["params_of_final_layer_for_50k_vocab_with_d_model_768"] == 38_450_000


def test_bias_shifts_the_boundary_and_weights_reflect_relevance():
    r = L.ex_3_6_3_7()
    assert r["acc_with_bias"] > 0.9 and r["acc_without_bias"] < 0.75
    w0, w1 = r["learned_weight_signs_follow_feature_relevance(|w0|,|w1|)"]; assert w0 > 20 * w1


def test_weighted_sum_matrix_form_equals_loops():
    r = L.ex_3_8(); assert r["equal"] and r["row_independence"] and r["shape"] == (256, 32) and r["speedup"] > 50


def test_activation_table_facts():
    r = L.ex_3_9_to_3_14()
    assert r["sigmoid"]["max_derivative"] == pytest.approx(0.25, abs=1e-3) and r["tanh"]["max_derivative"] == pytest.approx(1.0, abs=1e-3)
    assert r["relu"]["min"] == 0 and r["leaky_relu"]["flat_fraction(|f'|<0.01 on [-6,6])"] == 0 and r["silu"]["min"] == pytest.approx(-0.278, abs=2e-3)
    assert r["gelu"]["min"] == pytest.approx(-0.17, abs=5e-3) and r["gelu"]["max_derivative"] > 1.0
    assert r["relu_dead_fraction(pre-activation shifted -1)"] > 0.5 and r["sigmoid_chain_gradient_bound_10_layers"] < 1e-6
    assert r["tanh"]["flat_fraction(|f'|<0.01 on [-6,6])"] > r["sigmoid"]["flat_fraction(|f'|<0.01 on [-6,6])"]


def test_softmax_lab():
    r = L.ex_3_15(); e = r["mean_entropy_by_temperature"]
    assert r["rows_sum_to_one"] and r["shift_invariant"] and r["stable_for_1000"] and r["argmax_unchanged_by_temperature"] and r["sigmoid_equals_2class_softmax"]
    assert e[0.1] < e[1.0] < e[10.0]


def test_dense_layers_collapse_without_nonlinearity():
    r = L.ex_3_16(); assert r["two_linear_equals_one"] and r["with_relu_differs"] and r["params"] == (54, 21) and r["collapsed_params"] == 27


def test_mlp_xor_width_and_nonlinear_boundary():
    r = L.ex_3_17(); m = r["sin_mse_by_hidden_width"]
    assert r["xor_by_hand"] == [0.0, 1.0, 1.0, 0.0]
    assert m[1] > m[2] > m[4] > m[16] and m[16] < 0.01                         # more hidden units fit sin(x) better (64 units is not monotone at this fixed budget)
    assert r["circle_acc_linear"] < 0.7 and r["circle_acc_mlp"] > 0.97
