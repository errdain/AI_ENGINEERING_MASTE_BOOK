import lab_05_sequence_comparison as L


def test_all_recurrent_char_models_beat_the_bigram_baseline_and_the_uniform_guess():
    r = L.ex_5_10_char_language_models(steps=400)
    assert all(r[k] < 0.6 * r["bigram_baseline"] for k in ("rnn", "lstm", "gru")) and r["bigram_baseline"] < r["uniform_guess"]


def test_gradient_signal_decays_for_small_weights_and_grows_for_large_ones():
    r = L.ex_5_13_gradient_signal()
    assert r[0.2][0] < 1e-3 and r[1.2][10] > 10                                # vanishing vs exploding
    assert r[0.2][0] < r[0.5][0] or r[0.5][0] < 1e-1


def test_memory_gets_harder_with_delay_for_the_vanilla_rnn_but_not_for_the_lstm():
    r = L.ex_5_14_memory_vs_delay(delays=(5, 25), steps=1500)
    assert r["rnn"][5] > 0.9 and r["rnn"][25] < 0.3 and r["lstm"][25] > 0.9


def test_attention_beats_the_plain_encoder_decoder_on_long_inputs():
    r = L.ex_5_21_seq2seq_vs_length(lengths=(16,), steps=1200)
    assert r["none"][16] < 0.5 and r["bahdanau"][16] > 0.8 and r["luong"][16] > 0.8


def test_teacher_forcing_probability_is_not_the_main_factor_on_this_easy_task():
    r = L.ex_5_22_teacher_forcing(steps=800)
    assert max(r.values()) - min(r.values()) < 0.25 and min(r.values()) > 0.5


def test_exposure_bias_free_running_is_worse_than_teacher_forced():
    r = L.ex_5_23_exposure_bias(steps=800)
    assert r["per_token_accuracy_with_own_history"] < r["per_token_accuracy_with_true_history"]


def test_sequential_depth_table():
    r = L.ex_5_26_sequential_depth()
    assert r[1000]["rnn_sequential_steps"] == 1000 and r[1000]["attention_sequential_steps"] == 1 and r[100]["attention_pairwise_scores"] == 10000
