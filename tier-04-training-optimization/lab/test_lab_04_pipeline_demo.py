import lab_04_pipeline_demo as L


def test_training_loop_learns():
    r = L.ex_4_1_training_loop()
    assert r["last_train_loss"] < r["first_train_loss"] and r["final_val_acc"] > 0.75


def test_schedule_shape():
    r = L.ex_4_10_schedule()
    assert r[19] > r[0] and r[20] >= r[100] > r[199]


def test_accumulation_is_equivalent():
    assert L.ex_4_14_accumulation_equals_bigger_batch()["max_weight_difference"] < 1e-5


def test_resume_is_exact():
    assert L.ex_4_20_resume_is_exact()["max_weight_difference_after_resume"] < 1e-6


def test_profile_has_all_phases():
    r = L.ex_4_25_profile()
    assert set(r["phase_shares"]) == {"data", "forward", "backward", "optimizer"} and r["flops_per_step"] > 0
