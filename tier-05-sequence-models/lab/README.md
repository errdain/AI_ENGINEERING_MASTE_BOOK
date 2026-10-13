# Tier 5 Lab — character RNN → LSTM → attention Seq2Seq comparison

The roadmap's build task: *"character RNN → LSTM → attention Seq2Seq comparison."* `code/seqmodels.py` holds NumPy cells (RNN, LSTM, GRU) checked against `torch.nn`, a NumPy RNN with manual BPTT, a tiny NumPy word2vec (skip-gram and CBOW), and small PyTorch models: a character language model (RNN / LSTM / GRU), a copy-after-delay memory task, and a GRU Seq2Seq with optional Bahdanau or Luong attention on a digit-reversal task.

```sh
cd tier-05-sequence-models
pip install -r requirements.txt
python -m pytest -q code lab                        # 17 tests (about 3 minutes: the tests train small models)
PYTHONPATH=code python lab/lab_05_sequence_comparison.py     # print every comparison
```

| Function (lab/lab_05_sequence_comparison.py) | Concepts | What it compares |
|---|---|---|
| `ex_5_10_char_language_models` | 5.10, 5.14, 5.19 | held-out next-character loss of RNN, LSTM, GRU vs a bigram baseline (all about equal on short-range text) |
| `ex_5_13_gradient_signal` | 5.12, 5.13 | learning signal vs distance for four recurrent-weight scales |
| `ex_5_14_memory_vs_delay` | 5.14–5.19 | memory-task accuracy vs delay: RNN fails around 10–20 steps, GRU around 30, LSTM (forget bias 1) solves 40 |
| `ex_5_21_seq2seq_vs_length` | 5.20, 5.21, 5.24, 5.25 | exact-match accuracy reversing 4/8/16 digits: no attention vs Bahdanau vs Luong |
| `ex_5_22_teacher_forcing` | 5.22 | teacher-forcing ratio 1.0 / 0.5 / 0.0 (little difference on this easy task) |
| `ex_5_23_exposure_bias` | 5.23 | per-token accuracy with true vs own history (gap ≈ 0.09 at length 12) |
| `ex_5_26_sequential_depth` | 5.26 | serial steps and path lengths of RNN vs attention |

All runs are seeded and CPU-only. The memory-task break points are single-seed results and move by a few steps between seeds; the tests assert the ordering (vanilla RNN fails where the LSTM succeeds), not exact numbers.
