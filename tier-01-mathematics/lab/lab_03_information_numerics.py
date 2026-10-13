"""Tier 1 Lab, part 3: solutions to the hands-on exercises of concepts 1.32-1.47."""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np

from calculus_prob import Welford, numerical_gradient
from info_numerics import (
    bayes_posterior, cross_entropy_from_logits, double_well_grad, entropy, kl_divergence, log_softmax, logsumexp, descend_1d,
)


def ex_1_32(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal([0, 0], [[4.0, 3.0], [3.0, 4.0]], 20000)
    w, Q = np.linalg.eigh(np.cov(X, rowvar=False))
    major = Q[:, -1]
    return {"eigvals": w.tolist(), "major_axis_deg": float(np.rad2deg(np.arctan2(major[1], major[0])) % 180),
            "explained_ratio": float(w[-1] / w.sum())}


def ex_1_33() -> dict:
    """Chain rule on a bigram table, in log space."""
    vocab = ["<s>", "the", "cat", "sat", "</s>"]
    idx = {w: i for i, w in enumerate(vocab)}
    P = np.zeros((5, 5))
    P[idx["<s>"], idx["the"]] = 1.0
    P[idx["the"], idx["cat"]] = 0.6; P[idx["the"], idx["sat"]] = 0.4
    P[idx["cat"], idx["sat"]] = 0.7; P[idx["cat"], idx["</s>"]] = 0.3
    P[idx["sat"], idx["</s>"]] = 1.0
    sent = ["<s>", "the", "cat", "sat", "</s>"]
    logp = sum(np.log(P[idx[a], idx[b]]) for a, b in zip(sent, sent[1:]))
    return {"log_prob": float(logp), "prob": float(np.exp(logp)), "direct": 1.0 * 0.6 * 0.7 * 1.0}


def ex_1_34() -> dict:
    prev = np.logspace(-3, np.log10(0.5), 400)
    post = np.array([bayes_posterior(p, 0.99, 0.05) for p in prev])
    cross = float(prev[np.argmax(post >= 0.5)])
    return {"posterior_at_1pct": float(bayes_posterior(0.01, 0.99, 0.05)), "prevalence_for_50pct": cross,
            "analytic_prevalence": 0.05 / (0.99 + 0.05)}  # solve p*0.99 = (1-p)*0.05


def ex_1_35(seed: int = 0) -> dict:
    """Exponential distribution: MLE of the rate is 1/mean; confirmed by minimizing the NLL on a grid."""
    x = np.random.default_rng(seed).exponential(1 / 2.5, 5000)
    nll = lambda lam: -(len(x) * np.log(lam) - lam * x.sum())
    grid = np.linspace(0.5, 5, 4501)
    return {"closed_form": float(1 / x.mean()), "grid": float(grid[np.argmin([nll(l) for l in grid])])}


def ex_1_36() -> dict:
    z = np.array([3.0, 1.0, 0.5, -1.0])
    out = {}
    for T in (0.1, 0.5, 1.0, 2.0, 10.0):
        p = np.exp(log_softmax(z / T))
        out[T] = entropy(p, base=np.e)
    return out


def ex_1_37(seed: int = 0) -> dict:
    """Cross-entropy from logits: analytic gradient (softmax - onehot) vs numerical."""
    rng = np.random.default_rng(seed)
    z, y = rng.standard_normal(6), 2
    p = np.exp(log_softmax(z)); g = p.copy(); g[y] -= 1
    gn = numerical_gradient(lambda v: cross_entropy_from_logits(v, y), z)
    return {"max_abs_diff": float(np.max(np.abs(g - gn)))}


def ex_1_38() -> dict:
    """Fit a single Gaussian to a bimodal mixture by forward vs reverse KL (grid search)."""
    x = np.linspace(-10, 10, 2001)
    pdf = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p = 0.5 * pdf(-3, 1) + 0.5 * pdf(3, 1); p /= p.sum()
    best = {"forward": (np.inf, None), "reverse": (np.inf, None)}
    for m in np.linspace(-5, 5, 41):
        for s in np.linspace(0.5, 5, 19):
            q = pdf(m, s); q = np.maximum(q / q.sum(), 1e-300)
            f = float(np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0])))
            r = float(np.sum(q * np.log(q / np.maximum(p, 1e-300))))
            if f < best["forward"][0]: best["forward"] = (f, (float(m), float(s)))
            if r < best["reverse"][0]: best["reverse"] = (r, (float(m), float(s)))
    return {"forward_fit": best["forward"][1], "reverse_fit": best["reverse"][1]}


def ex_1_39(seed: int = 0, n: int = 5000) -> dict:
    """Variance of 1e9 + N(0,1): naive vs two-pass vs Welford vs exact Decimal reference."""
    x = np.random.default_rng(seed).standard_normal(n) + 1e9
    getcontext().prec = 60
    ds = [Decimal(float(v)) for v in x]
    mean = sum(ds) / n
    ref = float(sum((d - mean) ** 2 for d in ds) / (n - 1))
    naive = float((np.mean(x ** 2) - np.mean(x) ** 2) * n / (n - 1))
    two_pass = float(np.sum((x - x.mean()) ** 2) / (n - 1))
    w = Welford(); [w.update(float(v)) for v in x]
    return {"reference": ref, "naive": naive, "two_pass": two_pass, "welford": w.var}


def ex_1_40(n: int = 1_000_000, seed: int = 0) -> dict:
    x = np.random.default_rng(seed).random(n).astype(np.float32)
    sub = x[:200_000]
    exact = float(np.sum(sub.astype(np.float64)))
    naive = np.float32(0)
    for v in sub:
        naive = np.float32(naive + v)
    return {"pairwise_f32_err": abs(float(np.sum(sub)) - exact), "naive_f32_err": abs(float(naive) - exact)}


def ex_1_41(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, 1001)
    zipf = (1 / ranks) / (1 / ranks).sum()
    toks = rng.choice(ranks - 1, size=200, p=zipf)
    return {"direct_product": float(np.prod(zipf[toks])), "log_prob": float(np.sum(np.log(zipf[toks])))}


def ex_1_42() -> dict:
    out = {}
    for dt in (np.float16, np.float32, np.float64):
        lo, hi = 0.0, 1000.0
        with np.errstate(all="ignore"):
            for _ in range(60):
                mid = (lo + hi) / 2
                if np.isfinite(dt(np.exp(dt(mid)))): lo = mid
                else: hi = mid
        out[dt.__name__] = {"largest_finite_arg": lo, "ln_max": float(np.log(np.finfo(dt).max))}
    return out


class OnlineLogSumExp:
    """Streaming log-sum-exp with a running max (the idea inside fused attention)."""

    def __init__(self) -> None:
        self.m, self.s = -np.inf, 0.0

    def update(self, chunk: np.ndarray) -> None:
        cm = float(np.max(chunk))
        new_m = max(self.m, cm)
        if np.isfinite(new_m):
            self.s = self.s * np.exp(self.m - new_m) if np.isfinite(self.m) else 0.0
            self.s += float(np.sum(np.exp(chunk - new_m)))
        self.m = new_m

    @property
    def value(self) -> float:
        return float(self.m + np.log(self.s)) if self.s > 0 else float("-inf")


def ex_1_43(seed: int = 0) -> dict:
    x = np.random.default_rng(seed).standard_normal(1000) * 50
    o = OnlineLogSumExp()
    for chunk in np.array_split(x, 7):
        o.update(chunk)
    return {"online": o.value, "batch": float(logsumexp(x))}


class NonFiniteError(RuntimeError):
    pass


def guarded_step(loss: float, grad_norm: float, step: int) -> None:
    """Training-loop guard: raise with context when loss or grad-norm is non-finite."""
    if not np.isfinite(loss) or not np.isfinite(grad_norm):
        raise NonFiniteError(f"step {step}: loss={loss}, grad_norm={grad_norm}")


def ex_1_44() -> dict:
    losses = [2.3, 2.1, 1.9, float("nan"), 1.5]
    for i, l in enumerate(losses):
        try:
            guarded_step(l, 1.0, i)
        except NonFiniteError as e:
            return {"stopped_at": i, "message": str(e)}
    return {"stopped_at": None}


def ex_1_45(seed: int = 0) -> dict:
    """Logistic regression (convex): same optimum from 10 inits. Tiny MLP (non-convex): losses vary."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 2)); y = (X[:, 0] * X[:, 1] > 0).astype(float)  # XOR-like: not linearly separable
    Xl = np.hstack([X, np.ones((300, 1))])
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    lr_losses, mlp_losses = [], []
    for s in range(10):
        r = np.random.default_rng(100 + s)
        w = r.standard_normal(3) * 3
        for _ in range(3000):
            w -= 0.5 * Xl.T @ (sigmoid(Xl @ w) - y) / 300
        p = np.clip(sigmoid(Xl @ w), 1e-12, 1 - 1e-12)
        lr_losses.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        W1, b1, w2, b2 = r.standard_normal((2, 3)), np.zeros(3), r.standard_normal(3), 0.0
        for _ in range(3000):
            h = np.tanh(X @ W1 + b1); p = sigmoid(h @ w2 + b2); d = (p - y) / 300
            dh = np.outer(d, w2) * (1 - h ** 2)
            W1 -= 1.0 * X.T @ dh; b1 -= 1.0 * dh.sum(0); w2 -= 1.0 * h.T @ d; b2 -= 1.0 * d.sum()
        p = np.clip(sigmoid(np.tanh(X @ W1 + b1) @ w2 + b2), 1e-12, 1 - 1e-12)
        mlp_losses.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return {"logistic_spread": float(np.ptp(lr_losses)), "logistic_mean": float(np.mean(lr_losses)),
            "mlp_spread": float(np.ptp(mlp_losses)), "mlp_mean": float(np.mean(mlp_losses))}


def ex_1_46() -> dict:
    """x^2 + 10 y^2: iterations to reach ||v|| < 1e-6 with plain GD and heavy-ball momentum."""
    grad = lambda v: np.array([2 * v[0], 20 * v[1]])

    def run(lr, beta):
        v, m = np.array([1.0, 1.0]), np.zeros(2)
        for i in range(1, 100_000):
            m = beta * m + grad(v); v = v - lr * m
            n = np.linalg.norm(v)
            if n < 1e-6: return i
            if not np.isfinite(n) or n > 1e6: return None  # diverged
        return None

    return {"gd_best_lr_0.099": run(0.099, 0.0), "gd_unstable_lr_0.11": run(0.11, 0.0), "momentum_lr_0.05_beta_0.8": run(0.05, 0.8)}


def ex_1_47(starts: int = 50, seed: int = 0) -> dict:
    xs = np.random.default_rng(seed).uniform(-2.5, 2.5, starts)
    finals = np.array([descend_1d(double_well_grad, float(x0)) for x0 in xs])
    left = int((finals < 0).sum())
    boundary = float(np.roots([4, 0, -6, 1]).real[np.argsort(np.roots([4, 0, -6, 1]).real)][1])
    return {"to_global_min": left, "to_local_min": starts - left, "basin_boundary": boundary,
            "start_rule_holds": bool(np.all((xs < boundary - 0.01) <= (finals < 0)) and np.all((xs > boundary + 0.01) <= (finals > 0)))}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
