"""Tier 3 Lab, part 1: concepts 3.1-3.17 (neuron, perceptron, layers, weights/bias, weighted sum, activations, softmax, dense layer, MLP)."""
from __future__ import annotations

import time

import numpy as np

import nn as N


def ex_3_1() -> dict:
    """A single artificial neuron (weighted sum + threshold) implements logic gates; XOR needs more than one neuron."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    gate = lambda w, b: (X @ np.array(w) + b > 0).astype(int).tolist()
    return {"AND": gate([1, 1], -1.5), "OR": gate([1, 1], -0.5), "NAND": gate([-1, -1], 1.5),
            "XOR_with_two_layers": (lambda h: (h @ np.array([1, -1]) - 0.5 > 0).astype(int).tolist())(np.c_[X @ [1, 1] - 0.5 > 0, X @ [1, 1] - 1.5 > 0].astype(float))}


def ex_3_2(seed: int = 0) -> dict:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); out = {}
    for name, y in (("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])):
        p = N.Perceptron(max_epochs=100).fit(X, np.array(y)); out[name] = {"converged": bool(p.converged_), "epochs": len(p.mistakes_per_epoch_), "mistakes": int(sum(p.mistakes_per_epoch_))}
    rng = np.random.default_rng(seed); w = np.array([1.0, -2.0]); bound = {}
    for margin in (0.5, 0.2, 0.05):
        X2 = rng.uniform(-1, 1, (4000, 2)); s = X2 @ w; keep = np.abs(s) > margin; X2, y2 = X2[keep][:300], (s[keep][:300] > 0).astype(int)
        p = N.Perceptron(max_epochs=5000).fit(X2, y2); R = np.linalg.norm(X2, axis=1).max(); g = np.abs(X2 @ w).min() / np.linalg.norm(w)
        bound[margin] = {"mistakes": int(sum(p.mistakes_per_epoch_)), "novikoff_bound": float((R / g) ** 2), "converged": bool(p.converged_)}
    return {"logic": out, "mistakes_vs_margin": bound}


def ex_3_3_to_3_5() -> dict:
    sizes = [784, 128, 64, 10]; m = N.MLP(sizes); x = np.zeros((32, 784)); shapes = [x.shape]
    for l in m.layers: x = l.forward(x); shapes.append(x.shape)
    per_layer = [(sizes[i] * sizes[i + 1] + sizes[i + 1]) for i in range(3)]
    return {"shapes": shapes, "params_per_dense_layer": per_layer, "total_params": m.n_params(),
            "output_layer_sizes": {"regression(1 value)": 1, "binary(1 logit)": 1, "10-class": 10, "vocab-50k next-token": 50000},
            "params_of_final_layer_for_50k_vocab_with_d_model_768": 768 * 50000 + 50000}


def ex_3_6_3_7(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X = rng.standard_normal((600, 2)) + 3.0; y = (X[:, 0] + X[:, 1] > 6.5).astype(int)    # boundary does not pass through the origin
    fit = lambda bias: N.MLP([2, 1], bias=bias, seed=0).fit(X, y.astype(float).reshape(-1, 1), loss="bce", lr=0.1, epochs=800)
    acc = lambda m: float((m.predict_classes(X) == y).mean())
    Xs = (X - X.mean(0)) / X.std(0); w_true = np.array([2.0, 0.0]); ys = (Xs @ w_true > 0).astype(int)
    m = N.MLP([2, 1], seed=0).fit(Xs, ys.astype(float).reshape(-1, 1), loss="bce", lr=0.3, epochs=500); w = m.layers[0].W.ravel()
    return {"acc_with_bias": acc(fit(True)), "acc_without_bias": acc(fit(False)), "learned_weight_signs_follow_feature_relevance(|w0|,|w1|)": (abs(w[0]), abs(w[1]))}


def ex_3_8(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X, W, b = rng.standard_normal((256, 64)), rng.standard_normal((32, 64)), rng.standard_normal(32)
    def loops():
        out = np.zeros((256, 32))
        for i in range(256):
            for j in range(32):
                s = b[j]
                for k in range(64): s += X[i, k] * W[j, k]
                out[i, j] = s
        return out
    t0 = time.perf_counter(); a = loops(); t1 = time.perf_counter(); v = X @ W.T + b; t2 = time.perf_counter()
    return {"equal": bool(np.allclose(a, v)), "shape": v.shape, "row_independence": bool(np.allclose((X[:1] @ W.T + b), v[:1])), "speedup": (t1 - t0) / max(t2 - t1, 1e-9)}


def ex_3_9_to_3_14() -> dict:
    x = np.linspace(-8, 8, 1601); xs = np.linspace(-6, 6, 1201); out = {}
    for name, (f, df) in N.ACTIVATIONS.items():
        if name == "identity": continue
        d = df(xs)
        out[name] = {"min": round(float(f(x).min()), 3), "max": round(float(f(x).max()), 3), "max_derivative": round(float(d.max()), 3), "flat_fraction(|f'|<0.01 on [-6,6])": round(float((np.abs(d) < 0.01).mean()), 3)}
    rng = np.random.default_rng(0); h = rng.standard_normal((10000, 50)) @ rng.standard_normal((50, 50)).T * 0.5 - 1.0      # pre-activations with a negative shift
    out["relu_dead_fraction(pre-activation shifted -1)"] = float((h <= 0).mean())
    out["gelu_minus_relu_max_abs_diff"] = float(np.abs(N.gelu(x) - N.relu(x)).max())
    out["sigmoid_chain_gradient_bound_10_layers"] = 0.25 ** 10
    return out


def ex_3_15(seed: int = 0) -> dict:
    z = np.random.default_rng(seed).standard_normal((4, 6)) * 3; p = N.softmax(z)
    ent = lambda q: float(-(q * np.log(q + 1e-300)).sum(1).mean())
    return {"rows_sum_to_one": bool(np.allclose(p.sum(1), 1)), "shift_invariant": bool(np.allclose(p, N.softmax(z + 50))), "stable_for_1000": bool(np.isfinite(N.softmax(np.array([[1000.0, 0.0]]))).all()),
            "mean_entropy_by_temperature": {T: ent(N.softmax(z, temperature=T)) for T in (0.1, 1.0, 10.0)},
            "argmax_unchanged_by_temperature": bool((N.softmax(z, temperature=0.1).argmax(1) == p.argmax(1)).all()), "sigmoid_equals_2class_softmax": bool(np.allclose(N.sigmoid(z[:, 0] - z[:, 1]), N.softmax(z[:, :2])[:, 0]))}


def ex_3_16(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X = rng.standard_normal((5, 8)); a, b = N.Dense(8, 6, rng=rng), N.Dense(6, 3, rng=rng)
    lin = b.forward(a.forward(X)); one = X @ (b.W @ a.W).T; nonlin = b.forward(N.relu(a.forward(X)))
    return {"two_linear_equals_one": bool(np.allclose(lin, one)), "with_relu_differs": bool(not np.allclose(nonlin, one)), "params": (a.n_params(), b.n_params()), "collapsed_params": 8 * 3 + 3}


def ex_3_17(seed: int = 0) -> dict:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    relu = lambda z: np.maximum(0, z); W1 = np.array([[1, 1], [1, 1]], float); b1 = np.array([0, -1.0]); w2 = np.array([1, -2.0])
    by_hand = (relu(X @ W1.T + b1) @ w2).tolist()                                                    # exact XOR with 2 hidden ReLU units
    xs = np.linspace(-3, 3, 200)[:, None]; ys = np.sin(xs)
    mse_by_width = {}
    for k in (1, 2, 4, 16, 64):
        m = N.MLP([1, k, 1], activation="tanh", init="xavier_normal", seed=seed).fit(xs, ys, loss="mse", lr=0.1, epochs=6000); mse_by_width[k] = float(m.history_[-1])
    rng = np.random.default_rng(seed); P = rng.standard_normal((800, 2)); yc = (np.linalg.norm(P, axis=1) > 1.0).astype(int)    # circle: not linearly separable
    lin = N.MLP([2, 2], seed=0).fit(P, yc, lr=0.3, epochs=400); mlp = N.MLP([2, 16, 2], activation="tanh", init="xavier_normal", seed=0).fit(P, yc, lr=0.3, epochs=1500)
    return {"xor_by_hand": by_hand, "sin_mse_by_hidden_width": mse_by_width, "circle_acc_linear": float((lin.predict_classes(P) == yc).mean()), "circle_acc_mlp": float((mlp.predict_classes(P) == yc).mean())}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
