import numpy as np
import pytest

import nn as N

torch = pytest.importorskip("torch")
F = torch.nn.functional
RNG = np.random.default_rng(0)
XS = np.linspace(-8, 8, 161)


def _torch_act(name, x):
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu, "gelu": F.gelu, "silu": F.silu, "leaky_relu": lambda z: F.leaky_relu(z, 0.01)}[name](t)
    y.sum().backward(); return y.detach().numpy(), t.grad.numpy()


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu", "gelu", "silu", "leaky_relu"])
def test_activation_values_and_derivatives_match_torch(name):
    f, df = N.ACTIVATIONS[name]; y, g = _torch_act(name, XS)
    np.testing.assert_allclose(f(XS), y, atol=1e-12)
    mask = np.abs(XS) > 1e-9 if name in ("relu", "leaky_relu") else slice(None)   # kink at 0: conventions differ
    np.testing.assert_allclose(df(XS)[mask], g[mask], atol=1e-10)


def test_activation_properties():
    assert N.sigmoid(XS).min() > 0 and N.sigmoid(XS).max() < 1 and d_max(N.d_sigmoid) == pytest.approx(0.25, abs=1e-3)
    assert d_max(N.d_tanh) == pytest.approx(1.0, abs=1e-3) and np.all(N.relu(XS) >= 0)
    assert np.isfinite(N.sigmoid(np.array([-1000.0, 1000.0]))).all()
    assert N.gelu(np.array([0.0]))[0] == 0 and N.gelu(np.array([-10.0]))[0] > -1e-6          # GELU: ~0 for very negative x
    assert N.silu(np.array([-1.278]))[0] == pytest.approx(-0.2785, abs=1e-3)               # SiLU dips below zero (min ~ -0.278)
    assert N.d_sigmoid(np.array([8.0]))[0] < 4e-4                                           # saturation


def d_max(df): return float(df(XS).max())


def test_softmax_properties_and_torch():
    z = RNG.standard_normal((5, 7)) * 5
    p = N.softmax(z); np.testing.assert_allclose(p.sum(1), 1); np.testing.assert_allclose(p, N.softmax(z + 100))
    np.testing.assert_allclose(p, F.softmax(torch.tensor(z), -1).numpy(), atol=1e-12)
    np.testing.assert_allclose(N.log_softmax(z), F.log_softmax(torch.tensor(z), -1).numpy(), atol=1e-12)
    assert np.isfinite(N.softmax(np.array([[1000.0, 0.0]]))).all()
    hot, cold = N.softmax(z[:1], temperature=0.1), N.softmax(z[:1], temperature=10)
    assert hot.max() > p[:1].max() > cold.max()


def test_perceptron_learns_and_or_not_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for y in (np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1])):
        p = N.Perceptron().fit(X, y); assert p.converged_ and (p.predict(X) == y).all()
    p = N.Perceptron(max_epochs=200).fit(X, np.array([0, 1, 1, 0])); assert not p.converged_ and (p.predict(X) == [0, 1, 1, 0]).mean() < 1


def test_perceptron_mistake_bound():
    rng = np.random.default_rng(1); w = np.array([1.0, -2.0]); X = rng.uniform(-1, 1, (300, 2)); s = X @ w
    keep = np.abs(s) > 0.3; X, y = X[keep], (s[keep] > 0).astype(int)
    p = N.Perceptron(max_epochs=1000).fit(X, y)
    R = np.max(np.linalg.norm(X, axis=1)); gamma = np.min(np.abs(X @ w)) / np.linalg.norm(w)    # margin of the true separator
    assert p.converged_ and sum(p.mistakes_per_epoch_) <= (R / gamma) ** 2 + 1                  # Novikoff bound


def test_dense_forward_and_gradients_match_torch():
    d = N.Dense(4, 3, rng=np.random.default_rng(0)); d.b = RNG.standard_normal(3)
    X = RNG.standard_normal((6, 4)); out = d.forward(X); dY = RNG.standard_normal((6, 3)); dX = d.backward(dY)
    lin = torch.nn.Linear(4, 3).double(); lin.weight.data = torch.tensor(d.W); lin.bias.data = torch.tensor(d.b)
    xt = torch.tensor(X, requires_grad=True); yt = lin(xt); yt.backward(torch.tensor(dY))
    np.testing.assert_allclose(out, yt.detach().numpy()); np.testing.assert_allclose(dX, xt.grad.numpy())
    np.testing.assert_allclose(d.dW, lin.weight.grad.numpy()); np.testing.assert_allclose(d.db, lin.bias.grad.numpy())


def test_parameter_counts_match_torch():
    m = N.MLP([784, 128, 64, 10]); assert m.n_params() == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10 == 109386
    t = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
    assert sum(p.numel() for p in t.parameters()) == m.n_params()
    assert N.MLP([5, 3, 2], bias=False).n_params() == 5 * 3 + 3 * 2


def test_two_dense_layers_without_activation_collapse():
    a, b = N.Dense(5, 4, rng=np.random.default_rng(1)), N.Dense(4, 3, rng=np.random.default_rng(2)); X = RNG.standard_normal((7, 5))
    np.testing.assert_allclose(b.forward(a.forward(X)), X @ (b.W @ a.W).T)                      # one linear map


@pytest.mark.parametrize("act", ["tanh", "relu", "gelu", "sigmoid", "silu"])
def test_mlp_backprop_matches_numerical_and_torch(act):
    m = N.MLP([4, 6, 3], activation=act, init="xavier_normal", seed=3); X = RNG.standard_normal((8, 4)); y = RNG.integers(0, 3, 8)
    loss, d = N.softmax_cross_entropy(m.forward(X), y); m.backward(d)
    for layer, name, arr, g in m.parameters():
        num = N.numerical_grad(lambda: N.softmax_cross_entropy(m.forward(X), y)[0], arr)
        np.testing.assert_allclose(getattr(layer, g), num, atol=1e-6)
    # torch
    ts = [torch.tensor(l.W, requires_grad=True) for l in m.layers if isinstance(l, N.Dense)]; tb = [torch.tensor(l.b, requires_grad=True) for l in m.layers if isinstance(l, N.Dense)]
    f = {"tanh": torch.tanh, "relu": torch.relu, "gelu": F.gelu, "sigmoid": torch.sigmoid, "silu": F.silu}[act]
    h = f(torch.tensor(X) @ ts[0].T + tb[0]); lt = F.cross_entropy(h @ ts[1].T + tb[1], torch.tensor(y)); lt.backward()
    assert loss == pytest.approx(float(lt.detach()))
    np.testing.assert_allclose(m.layers[0].dW, ts[0].grad.numpy(), atol=1e-10); np.testing.assert_allclose(m.layers[-1].dW, ts[1].grad.numpy(), atol=1e-10)


def test_losses_match_torch_and_are_stable():
    z = RNG.standard_normal((6, 4)); y = RNG.integers(0, 4, 6)
    assert N.softmax_cross_entropy(z, y)[0] == pytest.approx(float(F.cross_entropy(torch.tensor(z), torch.tensor(y))))
    assert np.isfinite(N.softmax_cross_entropy(np.array([[1000.0, -1000.0]]), np.array([1]))[0])
    zb, yb = RNG.standard_normal(9), RNG.integers(0, 2, 9)
    assert N.sigmoid_bce(zb, yb)[0] == pytest.approx(float(F.binary_cross_entropy_with_logits(torch.tensor(zb), torch.tensor(yb, dtype=torch.float64))))


def test_mlp_solves_xor_and_regression_improves():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); y = np.array([0, 1, 1, 0])
    m = N.MLP([2, 4, 2], activation="tanh", seed=1).fit(X, y, lr=0.5, epochs=2000)
    assert (m.predict_classes(X) == y).all() and m.history_[-1] < m.history_[0] / 10
    lin = N.MLP([2, 2], seed=1).fit(X, y, lr=0.5, epochs=2000)                                 # no hidden layer = linear classifier
    assert (lin.predict_classes(X) == y).mean() < 1.0


def test_init_scheme_scales():
    rng = np.random.default_rng(0); n = 400
    assert N.init_weights(n, n, "he_normal", rng).std() == pytest.approx(np.sqrt(2 / n), rel=0.05)
    assert N.init_weights(n, n, "xavier_normal", rng).std() == pytest.approx(np.sqrt(1 / n), rel=0.05)
    assert np.abs(N.init_weights(n, n, "xavier_uniform", rng)).max() <= np.sqrt(6 / (2 * n)) + 1e-12
    with pytest.raises(ValueError): N.init_weights(2, 2, "bogus")
