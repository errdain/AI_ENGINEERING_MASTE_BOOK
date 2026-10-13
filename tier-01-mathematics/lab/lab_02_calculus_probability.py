"""Tier 1 Lab, part 2: solutions to the hands-on exercises of concepts 1.20-1.31."""
from __future__ import annotations

import numpy as np

from calculus_prob import Welford, derivative, numerical_gradient, numerical_jacobian, softmax


def ex_1_20() -> dict:
    """Activation ranges and where each is (nearly) flat (|f'| < 0.01) on [-5, 5]."""
    x = np.linspace(-5, 5, 1001)
    sig = 1 / (1 + np.exp(-x))
    acts = {"sigmoid": sig, "tanh": np.tanh(x), "relu": np.maximum(x, 0)}
    grads = {"sigmoid": sig * (1 - sig), "tanh": 1 - np.tanh(x) ** 2, "relu": (x > 0).astype(float)}
    return {n: {"min": float(a.min()), "max": float(a.max()), "flat_fraction": float((np.abs(grads[n]) < 0.01).mean())}
            for n, a in acts.items()}


def ex_1_21() -> dict:
    """Forward vs central difference error of sin'(1) = cos(1) over h = 1e-1 .. 1e-16."""
    true = np.cos(1.0)
    hs = [10.0 ** -k for k in range(1, 17)]
    fwd = [abs((np.sin(1 + h) - np.sin(1)) / h - true) for h in hs]
    cen = [abs((np.sin(1 + h) - np.sin(1 - h)) / (2 * h) - true) for h in hs]
    return {"best_h_forward": hs[int(np.argmin(fwd))], "best_h_central": hs[int(np.argmin(cen))],
            "best_err_forward": float(min(fwd)), "best_err_central": float(min(cen))}


def ex_1_22() -> dict:
    """L(w, b) = mean((w*x + b - y)^2): analytic partials vs numerical."""
    x, y = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.1, 5.9])
    L = lambda p: float(np.mean((p[0] * x + p[1] - y) ** 2))
    p = np.array([1.5, 0.3])
    r = p[0] * x + p[1] - y
    analytic = np.array([2 * np.mean(r * x), 2 * np.mean(r)])
    return {"analytic": analytic, "numerical": numerical_gradient(L, p)}


def ex_1_23(etas=(0.01, 0.05, 0.09, 0.11), steps: int = 300) -> dict:
    """GD on f = x^2 + 10 y^2: stable iff eta < 0.1."""
    out = {}
    for eta in etas:
        v = np.array([1.0, 1.0])
        for _ in range(steps):
            v = v - eta * np.array([2 * v[0], 20 * v[1]])
        out[eta] = float(np.linalg.norm(v))
    return out


def ex_1_24() -> dict:
    """Jacobian of f(x, y) = (x^2 y, 5x + sin y) at (1, 2)."""
    f = lambda v: np.array([v[0] ** 2 * v[1], 5 * v[0] + np.sin(v[1])])
    p = np.array([1.0, 2.0])
    by_hand = np.array([[2 * p[0] * p[1], p[0] ** 2], [5.0, np.cos(p[1])]])
    return {"by_hand": by_hand, "numerical": numerical_jacobian(f, p)}


def ex_1_25(seed: int = 0) -> dict:
    """Two-layer net with bias and selectable activation; backprop verified by gradient check."""
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(3), 0.5
    sizes = {"W1": (4, 3), "b1": (4,), "W2": (4,)}

    def unpack(theta):
        out, i = {}, 0
        for k, s in sizes.items():
            n = int(np.prod(s)); out[k] = theta[i:i + n].reshape(s); i += n
        return out

    def forward_backward(theta, act):
        P = unpack(theta)
        z1 = P["W1"] @ x + P["b1"]
        a1, da = (np.tanh(z1), 1 - np.tanh(z1) ** 2) if act == "tanh" else (np.maximum(z1, 0), (z1 > 0).astype(float))
        yhat = P["W2"] @ a1
        loss = (yhat - y) ** 2
        d_y = 2 * (yhat - y)
        dz1 = d_y * P["W2"] * da
        grads = np.concatenate([np.outer(dz1, x).ravel(), dz1, d_y * a1])
        return loss, grads

    theta = rng.standard_normal(sum(int(np.prod(s)) for s in sizes.values()))
    res = {}
    for act in ("tanh", "relu"):
        _, g = forward_backward(theta, act)
        gn = numerical_gradient(lambda t: forward_backward(t, act)[0], theta)
        res[act] = float(np.max(np.abs(g - gn)))
    return res


def ex_1_26(n: int = 400_000, seed: int = 0) -> dict:
    d = np.random.default_rng(seed).integers(1, 7, size=(n, 2))
    return {"estimate": float((d.sum(1) == 7).mean()), "exact": 6 / 36}


def ex_1_27(seeds=range(5)) -> dict:
    """Toy 'model' accuracy across seeds: report mean +- std and judge a 0.4% difference."""
    accs_a, accs_b = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        accs_a.append(float((rng.random(2000) < 0.870).mean()))
        accs_b.append(float((rng.random(2000) < 0.874).mean()))
    a, b = np.array(accs_a), np.array(accs_b)
    return {"A": (float(a.mean()), float(a.std(ddof=1))), "B": (float(b.mean()), float(b.std(ddof=1))),
            "diff_within_noise": bool(abs(a.mean() - b.mean()) < 2 * np.sqrt(a.var(ddof=1) + b.var(ddof=1)))}


def ex_1_28(n: int = 200_000, seed: int = 0) -> dict:
    """Normal vs log-normal with the same mean: percentiles show the tail the mean hides."""
    rng = np.random.default_rng(seed)
    sigma = 1.0
    mu = np.log(200) - sigma ** 2 / 2  # lognormal mean = exp(mu + sigma^2/2) = 200
    ln = rng.lognormal(mu, sigma, n)
    nm = rng.normal(200, ln.std(), n)
    q = lambda a: {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99))}
    return {"normal": q(nm), "lognormal": q(ln)}


def ex_1_29(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 5, 1000)
    x[:10] = 1e5  # 1% extreme outliers
    s = np.sort(x); k = int(0.1 * len(x))
    return {"mean": float(x.mean()), "median": float(np.median(x)), "trimmed_mean": float(s[k:-k].mean())}


def ex_1_30(trials: int = 20000, n: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((trials, n))
    return {"mean_var_ddof0": float(X.var(axis=1, ddof=0).mean()), "mean_var_ddof1": float(X.var(axis=1, ddof=1).mean())}


class StandardScaler:
    """fit on train only; zero-variance guard."""

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, float)
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)  # constant feature: leave unscaled
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) / self.std_


def ex_1_31(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    train = np.column_stack([rng.normal(10, 3, 500), np.full(500, 7.0)])
    test = np.column_stack([rng.normal(12, 3, 100), np.full(100, 7.0)])
    sc = StandardScaler().fit(train)
    return {"train_mean": sc.transform(train).mean(0), "train_std": sc.transform(train).std(0),
            "test_mean_col0": float(sc.transform(test)[:, 0].mean()), "const_col_finite": bool(np.isfinite(sc.transform(test)).all())}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
