import pytest

import lab_01_paradigms_and_data as L


def test_ai_ml_dl_ladder_on_xor_task():
    r = L.ex_2_1()
    assert r["rule_based"] < 0.65 and r["linear_ml"] < 0.65  # wrong rule / linear model can't represent XOR
    assert r["ml_with_engineered_feature"] > 0.92 and r["neural_net"] > 0.92  # engineered feature or learned feature both solve it


def test_supervised_beats_baseline_and_generalizes():
    r = L.ex_2_2()
    assert r["test_acc"] > r["baseline_acc"] + 0.3 and abs(r["train_acc"] - r["test_acc"]) < 0.1


def test_unsupervised_clusters_and_elbow():
    r = L.ex_2_3(); i = r["inertia_by_k"]
    assert r["agreement_with_hidden_labels"] > 0.99
    assert i[1] > i[2] > i[3] > i[4] > i[6] and (i[1] - i[2]) > 10 * (i[2] - i[3])  # elbow at the true k = 2


def test_self_training_uses_confident_pseudolabels():
    r = L.ex_2_4()
    assert r["pseudo_labels_used"] > 100 and r["pseudo_label_accuracy"] > 0.9
    assert r["self_training_acc"] >= r["labeled_only_acc"] - 0.02  # helps or at least doesn't hurt on well-separated data


def test_self_supervised_bigram_learns_from_unlabeled_text():
    r = L.ex_2_5()
    assert r["bigram_nll"] < 0.5 * r["uniform_nll"] and r["most_likely_after_t"] == "h" and r["n_training_pairs"] > 1000


def test_bandit_needs_exploration():
    r = L.ex_2_6()
    assert r[0.0]["best_arm_share"] == 0.0 and r[0.1]["avg_reward"] > 0.7 > r[0.0]["avg_reward"] + 0.4


def test_features_labels_samples_effects():
    r = L.ex_2_7_to_2_9()
    assert r["X_shape"] == (1000, 2) and r["y_shape"] == (1000,)
    assert r["junk60"] < r["clean"] - 0.05 and r["noise40"] < r["clean"] - 0.1
    assert r["n=5"] < r["n=20"] < r["n=400"] + 0.01 and r["n=400"] > 0.9


def test_group_split_removes_optimistic_bias():
    r = L.ex_2_10()
    assert r["shared_users_in_row_split"] > 0 and r["shared_users_in_group_split"] == 0
    assert r["random_row_split_acc"] > 0.95 and r["group_split_acc"] < 0.7


def test_peeking_at_test_set_inflates_score():
    r = L.ex_2_11_to_2_13()
    assert r["best_test_acc_after_peeking"] > 0.58 and abs(r["fresh_data_acc_of_winner"] - 0.5) < 0.03
