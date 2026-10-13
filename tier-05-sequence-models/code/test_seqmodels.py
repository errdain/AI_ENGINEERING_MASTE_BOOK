import numpy as np
import pytest
import torch
from torch import nn

import seqmodels as S


def torch_weights(mod):
    return [p.detach().numpy()[...] for p in (mod.weight_ih_l0, mod.weight_hh_l0, mod.bias_ih_l0, mod.bias_hh_l0)]


def test_rnn_cell_matches_torch():
    torch.manual_seed(0); m = nn.RNN(3, 5, batch_first=True); x = torch.randn(2, 4, 3)
    ref, _ = m(x); Wih, Whh, bih, bhh = torch_weights(m)
    h = np.zeros((2, 5)); outs = []
    for t in range(4):
        h = S.rnn_cell(x[:, t].numpy(), h, Wih, Whh, bih, bhh); outs.append(h)
    assert np.allclose(np.stack(outs, 1), ref.detach().numpy(), atol=1e-6)


def test_lstm_cell_matches_torch():
    torch.manual_seed(0); m = nn.LSTM(3, 5, batch_first=True); x = torch.randn(2, 4, 3)
    ref, (hn, cn) = m(x); Wih, Whh, bih, bhh = torch_weights(m)
    h = c = np.zeros((2, 5)); outs = []
    for t in range(4):
        h, c, _ = S.lstm_cell(x[:, t].numpy(), h, c, Wih, Whh, bih, bhh); outs.append(h)
    assert np.allclose(np.stack(outs, 1), ref.detach().numpy(), atol=1e-6) and np.allclose(c, cn[0].detach().numpy(), atol=1e-6)


def test_gru_cell_matches_torch():
    torch.manual_seed(0); m = nn.GRU(3, 5, batch_first=True); x = torch.randn(2, 4, 3)
    ref, _ = m(x); Wih, Whh, bih, bhh = torch_weights(m)
    h = np.zeros((2, 5)); outs = []
    for t in range(4):
        h, _ = S.gru_cell(x[:, t].numpy(), h, Wih, Whh, bih, bhh); outs.append(h)
    assert np.allclose(np.stack(outs, 1), ref.detach().numpy(), atol=1e-6)


def test_bptt_matches_finite_differences():
    net = S.NumpyRNN(vocab=5, hidden=6, seed=1); xs, ys = [0, 3, 1, 4, 2, 2], [3, 1, 4, 2, 2, 0]
    loss, hs, probs = net.forward(xs, ys); grads, _ = net.backward(xs, ys, hs, probs)
    for name in ("Wxh", "Whh", "Why"):
        W = getattr(net, name); idx = (1, 2); h = 1e-6
        W[idx] += h; lp = net.forward(xs, ys)[0]; W[idx] -= 2 * h; lm = net.forward(xs, ys)[0]; W[idx] += h
        assert grads[name][idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-8)


def test_rnn_gradient_signal_decays_with_distance_when_the_recurrent_weights_are_small():
    net = S.NumpyRNN(vocab=4, hidden=16, seed=0, scale=0.2)
    xs = list(np.random.default_rng(0).integers(0, 4, 40)); ys = xs[1:] + [0]
    norms = net.signal_from_last_step(xs, ys)
    assert norms[0] < 1e-3 * norms[-1]                                     # the signal reaching step 0 is tiny


def test_lstm_remembers_longer_than_a_vanilla_rnn():
    rnn, lstm = S.train_memory("rnn", 20, steps=800), S.train_memory("lstm", 20, steps=1500)
    assert rnn < 0.3 and lstm > 0.9


def test_attention_fixes_long_reversal():
    plain = S.train_seq2seq(None, 16, steps=1200); att = S.train_seq2seq("luong", 16, steps=1200)
    assert S.seq2seq_accuracy(plain, 16) < 0.5 and S.seq2seq_accuracy(att, 16) > 0.9


def test_attention_weights_are_a_distribution_and_peak_on_the_mirrored_position():
    m = S.train_seq2seq("luong", 8, steps=1200); src, _, _ = S.reverse_batch(50, 8, torch.Generator().manual_seed(3))
    _, att = m.greedy(src, 9)
    assert torch.allclose(att.sum(-1), torch.ones(50, 9), atol=1e-5)
    assert (att[:, :8].argmax(-1) == torch.arange(7, -1, -1)).float().mean() > 0.8   # step t attends to source position 7 - t


def toy_sentences(n=300, seed=0):
    rng = np.random.default_rng(seed); A = ["cat", "dog", "bird", "fox"]; V = ["saw", "chased", "liked"]; C = ["red", "blue", "green"]; O = ["car", "ball", "box"]
    out = []
    for _ in range(n):
        out.append(["the", A[rng.integers(4)], V[rng.integers(3)], "the", A[rng.integers(4)]])
        out.append(["the", C[rng.integers(3)], O[rng.integers(3)], "was", "here"])
    return out


@pytest.mark.parametrize("cbow", [False, True])
def test_word2vec_puts_words_with_the_same_role_closer_together(cbow):
    vocab, E = S.skipgram(toy_sentences(120), dim=8, epochs=30, cbow=cbow); idx = {w: i for i, w in enumerate(vocab)}
    E = E / np.linalg.norm(E, axis=1, keepdims=True); sim = lambda a, b: float(E[idx[a]] @ E[idx[b]])
    same = np.mean([sim("cat", "dog"), sim("red", "blue"), sim("saw", "chased"), sim("car", "ball")])
    diff = np.mean([sim("cat", "red"), sim("saw", "car"), sim("dog", "ball"), sim("chased", "blue")])
    assert same > diff + 0.2
