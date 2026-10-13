import math

import numpy as np
import pytest

import nn as N
from autodiff import Dual, Value

torch = pytest.importorskip("torch")
RNG = np.random.default_rng(0)


def _check_layer_vs_torch(layer, tmod, X, dY, params):
    xt = torch.tensor(X, requires_grad=True); yt = tmod(xt); yt.backward(torch.tensor(dY))
    out = layer.forward(X); dX = layer.backward(dY)
    np.testing.assert_allclose(out, yt.detach().numpy(), atol=1e-10); np.testing.assert_allclose(dX, xt.grad.numpy(), atol=1e-9)
    for name, (tp) in params.items(): np.testing.assert_allclose(getattr(layer, name), tp.grad.numpy(), atol=1e-9)


def test_batchnorm_forward_backward_and_eval_match_torch():
    X, dY = RNG.standard_normal((16, 5)) * 3 + 2, RNG.standard_normal((16, 5))
    bn = N.BatchNorm1d(5); bn.gamma, bn.beta = RNG.standard_normal(5), RNG.standard_normal(5)
    tbn = torch.nn.BatchNorm1d(5, eps=1e-5, momentum=0.1).double(); tbn.weight.data = torch.tensor(bn.gamma); tbn.bias.data = torch.tensor(bn.beta)
    xt = torch.tensor(X, requires_grad=True); yt = tbn(xt); yt.backward(torch.tensor(dY))
    out = bn.forward(X); dX = bn.backward(dY)
    np.testing.assert_allclose(out, yt.detach().numpy(), atol=1e-10); np.testing.assert_allclose(dX, xt.grad.numpy(), atol=1e-9)
    np.testing.assert_allclose(bn.dgamma, tbn.weight.grad.numpy(), atol=1e-9); np.testing.assert_allclose(bn.dbeta, tbn.bias.grad.numpy(), atol=1e-9)
    np.testing.assert_allclose(bn.running_mean, tbn.running_mean.numpy(), atol=1e-12); np.testing.assert_allclose(bn.running_var, tbn.running_var.numpy(), atol=1e-12)
    bn.training = False; tbn.eval(); np.testing.assert_allclose(bn.forward(X), tbn(torch.tensor(X)).detach().numpy(), atol=1e-10)


def test_batchnorm_train_output_has_zero_mean_unit_var_and_depends_on_batch():
    bn = N.BatchNorm1d(4); X = RNG.standard_normal((32, 4)) * 5 + 7; y = bn.forward(X)
    np.testing.assert_allclose(y.mean(0), 0, atol=1e-10); np.testing.assert_allclose(y.var(0), 1, atol=1e-3)
    row_alone = N.BatchNorm1d(4).forward(np.vstack([X[:1], RNG.standard_normal((31, 4))]))[0]
    assert not np.allclose(row_alone, y[0])                                                     # output of a sample depends on the other samples


def test_layernorm_matches_torch_and_is_batch_independent():
    X, dY = RNG.standard_normal((6, 8)) * 4 + 1, RNG.standard_normal((6, 8)); ln = N.LayerNorm(8); ln.gamma, ln.beta = RNG.standard_normal(8), RNG.standard_normal(8)
    tln = torch.nn.LayerNorm(8, eps=1e-5).double(); tln.weight.data = torch.tensor(ln.gamma); tln.bias.data = torch.tensor(ln.beta)
    xt = torch.tensor(X, requires_grad=True); yt = tln(xt); yt.backward(torch.tensor(dY)); out = ln.forward(X); dX = ln.backward(dY)
    np.testing.assert_allclose(out, yt.detach().numpy(), atol=1e-10); np.testing.assert_allclose(dX, xt.grad.numpy(), atol=1e-9)
    np.testing.assert_allclose(ln.dgamma, tln.weight.grad.numpy(), atol=1e-9); np.testing.assert_allclose(ln.dbeta, tln.bias.grad.numpy(), atol=1e-9)
    np.testing.assert_allclose(N.LayerNorm(8).forward(X[:1]), N.LayerNorm(8).forward(X)[:1], atol=1e-12)     # one sample alone = same result
    np.testing.assert_allclose(N.LayerNorm(8).forward(X).mean(-1), 0, atol=1e-10)


def test_rmsnorm_matches_torch_and_has_no_mean_subtraction():
    X, dY = RNG.standard_normal((6, 8)) * 2 + 3, RNG.standard_normal((6, 8)); rn = N.RMSNorm(8); rn.gamma = RNG.standard_normal(8)
    trn = torch.nn.RMSNorm(8, eps=1e-6).double(); trn.weight.data = torch.tensor(rn.gamma)
    xt = torch.tensor(X, requires_grad=True); yt = trn(xt); yt.backward(torch.tensor(dY)); out = rn.forward(X); dX = rn.backward(dY)
    np.testing.assert_allclose(out, yt.detach().numpy(), atol=1e-10); np.testing.assert_allclose(dX, xt.grad.numpy(), atol=1e-9); np.testing.assert_allclose(rn.dgamma, trn.weight.grad.numpy(), atol=1e-9)
    y = N.RMSNorm(8).forward(X); np.testing.assert_allclose(np.sqrt((y**2).mean(-1)), 1, atol=1e-5); assert abs(y.mean(-1)).max() > 0.1     # unit RMS, mean NOT removed
    assert N.RMSNorm(8).n_params() == 8 and N.LayerNorm(8).n_params() == 16 and N.BatchNorm1d(8).n_params() == 16


def test_dropout_inverted_scaling_eval_identity_and_backward():
    d = N.Dropout(0.3, seed=1); X = np.ones((20000, 10)); y = d.forward(X)
    assert y.mean() == pytest.approx(1.0, abs=0.02) and (y == 0).mean() == pytest.approx(0.3, abs=0.01) and set(np.unique(y.round(6))) == {0.0, round(1 / 0.7, 6)}
    np.testing.assert_allclose(d.backward(np.ones_like(X)), d.mask)
    d.training = False; np.testing.assert_array_equal(d.forward(X), X)
    with pytest.raises(ValueError): N.Dropout(1.0)


def test_residual_block_gradient_has_identity_path():
    rng = np.random.default_rng(0); blk = N.Sequential([N.Dense(6, 6, "xavier_normal", rng), N.Activation("tanh")]); res = N.Residual(blk)
    X, dY = RNG.standard_normal((4, 6)), RNG.standard_normal((4, 6)); out = res.forward(X); np.testing.assert_allclose(out, X + blk.forward(X))
    dX = res.backward(dY); np.testing.assert_allclose(dX, dY + blk.backward(dY))


def test_deep_net_with_all_pieces_gradients_match_finite_differences():
    net = N.build_deep_net(5, 2, "tanh", "xavier_normal", norm="layer", residual=True, n_out=3, seed=1); net.set_training(False)
    X, y = RNG.standard_normal((6, 5)), RNG.integers(0, 3, 6)
    loss, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d)
    for l, name, arr, g in net.parameters():
        num = N.numerical_grad(lambda: N.softmax_cross_entropy(net.forward(X), y)[0], arr); np.testing.assert_allclose(getattr(l, g), num, atol=1e-6)


def test_batchnorm_deep_net_gradient_check_in_train_mode():
    net = N.build_deep_net(4, 2, "tanh", "xavier_normal", norm="batch", n_out=2, seed=2); X, y = RNG.standard_normal((8, 4)), RNG.integers(0, 2, 8)
    loss, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d)
    for l, name, arr, g in net.parameters():
        grad = getattr(l, g).copy(); num = N.numerical_grad(lambda: N.softmax_cross_entropy(net.forward(X), y)[0], arr); np.testing.assert_allclose(grad, num, atol=1e-5)


def test_weight_decay_equals_l2_gradient_for_sgd_and_shrinks_weights():
    rng = np.random.default_rng(0); a, b = N.Dense(3, 2, rng=rng), N.Dense(3, 2, rng=np.random.default_rng(0)); a.dW = b.dW = RNG.standard_normal((2, 3)); a.db = b.db = np.zeros(2)
    N.Sequential([a]).step(0.1, weight_decay=0.01); b.W -= 0.1 * (b.dW + 0.01 * b.W)
    np.testing.assert_allclose(a.W, b.W)
    w = N.Dense(3, 2, rng=np.random.default_rng(1)); w.dW = np.zeros((2, 3)); w.db = np.zeros(2); n0 = np.linalg.norm(w.W)
    for _ in range(100): N.Sequential([w]).step(0.1, weight_decay=0.1)
    assert np.linalg.norm(w.W) == pytest.approx(n0 * (1 - 0.01) ** 100, rel=1e-9)


def test_clip_grad_norm():
    net = N.build_deep_net(4, 1, "relu", n_out=2); X, y = RNG.standard_normal((8, 4)) * 50, RNG.integers(0, 2, 8)
    _, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d); before = N.global_grad_norm(net); returned = N.clip_grad_norm(net, 0.1)
    assert returned == pytest.approx(before) and N.global_grad_norm(net) == pytest.approx(min(before, 0.1))


# ---------------------------------------------------------------- autodiff
def _f_value(x, y):
    return ((x * y + x.tanh()) ** 2 + (y * y + 1).log() - (x / (y * y + 2)) + (x - y).exp() * 0.1 + x.relu() + (x * 0.5).sigmoid())


def test_reverse_mode_matches_torch_and_numerical():
    x, y = Value(0.7), Value(-1.3); out = _f_value(x, y); out.backward()
    xt, yt = torch.tensor(0.7, dtype=torch.float64, requires_grad=True), torch.tensor(-1.3, dtype=torch.float64, requires_grad=True)
    ot = ((xt * yt + torch.tanh(xt)) ** 2 + torch.log(yt * yt + 1) - xt / (yt * yt + 2) + torch.exp(xt - yt) * 0.1 + torch.relu(xt) + torch.sigmoid(xt * 0.5)); ot.backward()
    assert out.data == pytest.approx(ot.item()) and x.grad == pytest.approx(xt.grad.item()) and y.grad == pytest.approx(yt.grad.item())
    h = 1e-6; num = (_f_value(Value(0.7 + h), Value(-1.3)).data - _f_value(Value(0.7 - h), Value(-1.3)).data) / (2 * h); assert x.grad == pytest.approx(num, rel=1e-6)


def test_gradient_accumulates_over_multiple_uses():
    x = Value(3.0); y = x * x + x * 2; y.backward(); assert x.grad == pytest.approx(2 * 3 + 2)       # d/dx(x^2 + 2x) = 2x + 2: paths are summed


def test_forward_mode_equals_reverse_mode_for_one_input():
    f = lambda x: (x * x + x.tanh()).exp() - x * 3
    d = f(Dual(0.4, 1.0)); v = Value(0.4); r = (v * v + v.tanh()).exp() - v * 3; r.backward()
    assert d.v == pytest.approx(r.data) and d.d == pytest.approx(v.grad)


def test_reverse_mode_one_pass_gives_all_gradients_while_forward_needs_one_per_input():
    n = 5; xs = [Value(0.1 * (i + 1)) for i in range(n)]; out = Value(0.0)
    for x in xs: out = out + x * x * x
    out.backward(); rev = [x.grad for x in xs]; fwd = []
    for i in range(n):
        d = Dual(0.0)
        for j in range(n): d = d + Dual(0.1 * (j + 1), 1.0 if i == j else 0.0) ** 3
        fwd.append(d.d)
    np.testing.assert_allclose(rev, fwd); np.testing.assert_allclose(rev, [3 * (0.1 * (i + 1)) ** 2 for i in range(n)])
