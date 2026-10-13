"""Tier 3 Lab, part 2: concepts 3.18-3.32 (forward/backward machinery, initialization, gradient problems, residuals, dropout, weight decay, normalization)."""
from __future__ import annotations

import numpy as np

import nn as N
from autodiff import Dual, Value


# ---- 3.18 forward propagation ------------------------------------------------------------------------------
def ex_3_18() -> dict:
    m = N.MLP([2, 3, 1], activation="relu", bias=True)
    W1 = np.array([[0.5, -1.0], [2.0, 0.1], [-0.3, 0.8]]); b1 = np.array([0.0, 1.0, -0.2]); W2 = np.array([[1.0, -1.0, 0.5]]); b2 = np.array([0.1])
    m.layers[0].W, m.layers[0].b, m.layers[2].W, m.layers[2].b = W1, b1, W2, b2
    x = np.array([[1.0, 2.0]]); z1 = x @ W1.T + b1; a1 = np.maximum(0, z1); z2 = a1 @ W2.T + b2
    B, widths = 32, [784, 128, 64, 10]
    return {"z1": z1.tolist(), "a1": a1.tolist(), "z2": z2.tolist(), "library_output": m.forward(x).tolist(),
            "activation_memory_floats_for_batch32(784-128-64-10)": B * sum(widths), "flops_per_sample": 2 * (784 * 128 + 128 * 64 + 64 * 10),
            "deterministic": bool(np.array_equal(m.forward(x), m.forward(x)))}


# ---- 3.19 computational graph --------------------------------------------------------------------------------
def ex_3_19() -> dict:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0); e = a * b; d = e + c; f = d * Value(-2.0)
    f.backward(); x = Value(3.0); y = x * x + x * 2; y.backward()
    w = [Value(0.5), Value(-0.4)]; xs = [1.0, 2.0]; z = w[0] * xs[0] + w[1] * xs[1]; out = z.tanh()
    return {"f": f.data, "df/da": a.grad, "df/db": b.grad, "df/dc": c.grad, "graph_nodes_f": f.graph_size(), "shared_node_gradient(x^2+2x at 3)": x.grad,
            "graph_nodes_single_neuron": out.graph_size()}


# ---- 3.20 backpropagation ---------------------------------------------------------------------------------------
def ex_3_20(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); m = N.MLP([4, 5, 3], activation="tanh", init="xavier_normal", seed=seed); X, y = rng.standard_normal((6, 4)), rng.integers(0, 3, 6)
    loss, d = N.softmax_cross_entropy(m.forward(X), y); m.backward(d)
    err = max(float(np.abs(getattr(l, g) - N.numerical_grad(lambda: N.softmax_cross_entropy(m.forward(X), y)[0], arr)).max()) for l, _, arr, g in m.parameters())
    P = N.MLP([784, 128, 64, 10]).n_params(); fwd = 2 * P                                         # forward FLOPs per sample ~ 2P
    return {"max_error_vs_finite_differences": err, "backward_flops_per_sample_approx": 2 * fwd, "ratio_backward_to_forward": 2.0,
            "forward_passes_needed_by_finite_differences": 2 * P, "forward_passes_needed_by_backprop": 1}


# ---- 3.21 automatic differentiation -----------------------------------------------------------------------------------
def ex_3_21() -> dict:
    n = 1000; xs = np.linspace(0.1, 1, n)
    def sum_cubes_rev():
        vs = [Value(x) for x in xs]; out = Value(0.0)
        for v in vs: out = out + v * v * v
        out.backward(); return np.array([v.grad for v in vs])
    rev = sum_cubes_rev(); exact = 3 * xs ** 2
    h_err = {}
    for h in (1e-1, 1e-3, 1e-5, 1e-8, 1e-12):
        h_err[h] = float(abs(((0.7 + h) ** 3 - (0.7 - h) ** 3) / (2 * h) - 3 * 0.7 ** 2))
    f = lambda v: v.relu() if v.data > 0 else v * 0.1                                              # data-dependent Python branch
    a, b = Value(2.0), Value(-2.0); fa, fb = f(a), f(b); fa.backward(); fb.backward()
    r = Value(2.7); rr = Value(float(np.round(r.data)))                                                 # rounding breaks the graph: no gradient flows
    return {"reverse_mode_max_error_1000_inputs": float(np.abs(rev - exact).max()), "numerical_error_by_h(central)": h_err,
            "branch_gradients(relu or 0.1x)": (a.grad, b.grad), "gradient_through_round": r.grad, "forward_passes_forward_mode": n, "forward_passes_reverse_mode": 1}


# ---- 3.22-3.24 initialization -------------------------------------------------------------------------------------------
def activation_std_by_layer(width: int, depth: int, act: str, init: str, seed: int = 0, n: int = 512) -> list[float]:
    rng = np.random.default_rng(seed); x = rng.standard_normal((n, width)); f = N.ACTIVATIONS[act][0]; out = []
    for _ in range(depth):
        x = f(x @ N.init_weights(width, width, init, rng).T); out.append(float(x.std()))
    return out


def ex_3_22(seed: int = 0) -> dict:
    stds = {init: activation_std_by_layer(256, 30, "tanh", init, seed) for init in ("zeros", "normal_0.01", "normal_1", "xavier_normal", "he_normal")}
    # symmetry: with identical (constant) initial weights all hidden units stay identical
    m = N.MLP([3, 4, 2], activation="tanh", seed=0)
    for l in m.layers:
        if isinstance(l, N.Dense): l.W[:] = 0.3
    rng = np.random.default_rng(seed); X, y = rng.standard_normal((20, 3)), rng.integers(0, 2, 20); m.fit(X, y, lr=0.2, epochs=50)
    rows = m.layers[0].W
    return {"activation_std_layers_1_10_30": {k: (round(v[0], 4), round(v[9], 4), round(v[29], 4)) for k, v in stds.items()}, "constant_init_rows_identical_after_training": bool(np.allclose(rows, rows[0]))}


def ex_3_23(seed: int = 0) -> dict:
    return {"tanh_30_layers_std": {init: round(activation_std_by_layer(256, 30, "tanh", init, seed)[-1], 4) for init in ("xavier_normal", "xavier_uniform", "he_normal", "normal_0.01")},
            "xavier_variance_formula": {"n_in=n_out=256": 2 / (256 + 256), "n_in=784,n_out=128": 2 / (784 + 128)}}


def ex_3_24(seed: int = 0) -> dict:
    return {"relu_30_layers_final_std": {init: activation_std_by_layer(256, 30, "relu", init, seed)[-1] for init in ("he_normal", "xavier_normal", "normal_0.01")},
            "predicted_decay_xavier_on_relu": float(0.5 ** (30 / 2)), "he_variance_formula(n_in=256)": 2 / 256}


# ---- 3.25 / 3.26 vanishing and exploding gradients ----------------------------------------------------------------------------
def grad_norms_by_layer(act: str, init: str, depth: int = 20, width: int = 64, gain: float = 1.0, norm: str | None = None, residual: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed); layers = []
    for i in range(depth):
        blk = [N.Dense(width, width, init, rng)]
        blk[0].W *= gain
        if norm == "batch": blk.append(N.BatchNorm1d(width))
        if norm == "layer": blk.append(N.LayerNorm(width))
        blk.append(N.Activation(act)); seq = N.Sequential(blk); layers.append(N.Residual(seq) if residual else seq)
    layers.append(N.Dense(width, 10, init, rng)); net = N.Sequential(layers)
    X, y = rng.standard_normal((128, width)), rng.integers(0, 10, 128); _, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d)
    return [float(np.linalg.norm(l.dW)) for blk in net.layers for l in (blk.block.layers if isinstance(blk, N.Residual) else blk.layers if isinstance(blk, N.Sequential) else [blk]) if isinstance(l, N.Dense)]


def ex_3_25(seed: int = 0) -> dict:
    ratio = lambda g: g[0] / g[-1]
    return {"first_over_last_layer_gradient_norm": {"sigmoid_xavier_20L": ratio(grad_norms_by_layer("sigmoid", "xavier_normal", seed=seed)), "tanh_xavier_20L": ratio(grad_norms_by_layer("tanh", "xavier_normal", seed=seed)),
            "relu_he_20L": ratio(grad_norms_by_layer("relu", "he_normal", seed=seed)), "sigmoid_xavier_20L+layernorm": ratio(grad_norms_by_layer("sigmoid", "xavier_normal", norm="layer", seed=seed)),
            "sigmoid_xavier_20L+residual": ratio(grad_norms_by_layer("sigmoid", "xavier_normal", residual=True, seed=seed))}, "sigmoid_bound_per_layer": 0.25}


def train_deep(gain: float, clip: float | None, steps: int = 60, lr: float = 0.1, depth: int = 12, width: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed); net = N.build_deep_net(width, depth, "relu", "he_normal", n_out=3, seed=seed)
    for blk in net.layers[:-1]: blk.layers[0].W *= gain
    X = rng.standard_normal((200, width)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int); losses, norms = [], []
    for _ in range(steps):
        out = net.forward(X); loss, d = N.softmax_cross_entropy(out, y)
        if not np.isfinite(loss): losses.append(float("inf")); break
        net.backward(d); norms.append(N.global_grad_norm(net) if clip is None else N.clip_grad_norm(net, clip)); net.step(lr); losses.append(loss)
    return losses, norms


def ex_3_26(seed: int = 0) -> dict:
    global_norm = lambda gain: float(np.sqrt(sum(x**2 for x in grad_norms_by_layer("relu", "he_normal", depth=20, gain=gain, seed=seed))))
    stable, _ = train_deep(1.0, None, seed=seed); big, big_norms = train_deep(1.3, None, seed=seed); clipped, clipped_norms = train_deep(1.3, 1.0, seed=seed)
    return {"global_gradient_norm_at_init_by_weight_gain(20 layers)": {g: global_norm(g) for g in (1.0, 1.25, 1.5, 2.0)}, "stable_final_loss(gain=1.0)": stable[-1],
            "gain1.3_no_clip": {"final_loss": big[-1], "steps_survived": len(big_norms)}, "gain1.3_clip_norm1": {"max_grad_norm_before_clip": max(clipped_norms), "final_loss": clipped[-1]}}


# ---- 3.27 residual connections ---------------------------------------------------------------------------------------------------
def ex_3_27(seed: int = 0) -> dict:
    def run(residual: bool, depth: int = 30):
        net = N.build_deep_net(32, depth, "tanh", "xavier_normal", residual=residual, n_out=3, seed=seed)
        for blk in net.layers[:-1]:
            blk_layers = (blk.block if residual else blk).layers; blk_layers[0].W *= (0.5 if residual else 1.0)
        rng = np.random.default_rng(seed); X = rng.standard_normal((200, 32)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
        _, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d)
        first = net.layers[0].block.layers[0].dW if residual else net.layers[0].layers[0].dW
        for _ in range(150):
            out = net.forward(X); loss, d = N.softmax_cross_entropy(out, y); net.backward(d); net.step(0.1)
        return float(np.linalg.norm(first)), float(loss)
    plain, res = run(False), run(True)
    return {"plain_30L": {"first_layer_grad_norm": plain[0], "loss_after_150_steps": plain[1]}, "residual_30L": {"first_layer_grad_norm": res[0], "loss_after_150_steps": res[1]}, "chance_loss": float(np.log(3))}


# ---- 3.28 dropout ------------------------------------------------------------------------------------------------------------------------
def _small_data(seed, n=60, p=40, sig=3):
    rng = np.random.default_rng(seed); w = np.zeros(p); w[:sig] = 1.0
    gen = lambda m: (lambda X: (X, ((X @ w + 1.2 * rng.standard_normal(m)) > 0).astype(int)))(rng.standard_normal((m, p)))
    return gen(n), gen(2000)


def ex_3_28(seeds=range(6)) -> dict:
    res = {"none": [], "dropout0.5": []}
    for s in seeds:
        (X, y), (Xt, yt) = _small_data(s)
        for name, p in (("none", 0.0), ("dropout0.5", 0.5)):
            rng = np.random.default_rng(s); net = N.Sequential([N.Dense(40, 128, "he_normal", rng), N.Activation("relu"), N.Dropout(p, s), N.Dense(128, 2, "he_normal", rng)])
            for _ in range(400):
                loss, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d); net.step(0.05)
            net.set_training(False); res[name].append(float((net.forward(Xt).argmax(1) == yt).mean()))
    d = N.Dropout(0.5, 0); x = np.ones((1, 1000)); outs = np.array([d.forward(x).mean() for _ in range(200)])
    net_eval = N.Sequential([N.Dense(4, 4, "he_normal", np.random.default_rng(0)), N.Dropout(0.5, 1)]); xx = np.ones((1, 4))
    return {"test_acc_mean": {k: float(np.mean(v)) for k, v in res.items()}, "expected_activation_preserved(mean over masks)": float(outs.mean()),
            "train_mode_outputs_vary": bool(not np.allclose(net_eval.forward(xx), net_eval.forward(xx))), "eval_mode_outputs_identical": bool(np.allclose(net_eval.set_training(False).forward(xx), net_eval.forward(xx)))}


# ---- 3.29 weight decay -------------------------------------------------------------------------------------------------------------------------
def ex_3_29(seeds=range(6)) -> dict:
    out = {}
    for wd in (0.0, 1e-3, 1e-2, 1e-1):
        accs, norms = [], []
        for s in seeds:
            (X, y), (Xt, yt) = _small_data(s); rng = np.random.default_rng(s)
            net = N.Sequential([N.Dense(40, 128, "he_normal", rng), N.Activation("relu"), N.Dense(128, 2, "he_normal", rng)])
            for _ in range(400):
                loss, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d); net.step(0.05, weight_decay=wd)
            accs.append(float((net.forward(Xt).argmax(1) == yt).mean())); norms.append(float(np.sqrt(sum((a**2).sum() for l, n, a, g in net.parameters() if n == "W"))))
        out[wd] = {"test_acc": float(np.mean(accs)), "weight_norm": float(np.mean(norms))}
    return {"by_weight_decay": out, "decay_factor_per_step(lr=0.1,wd=0.01)": 1 - 0.1 * 0.01, "after_100_steps_without_gradient": (1 - 0.1 * 0.01) ** 100}


# ---- 3.30 batch normalization ---------------------------------------------------------------------------------------------------------------------
def ex_3_30(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X = rng.standard_normal((256, 8)) * 5 + 3; bn = N.BatchNorm1d(8)
    for _ in range(40):
        for i in range(0, 256, 32): bn.forward(X[i:i + 32])
    bn.training = False; ev = bn.forward(X)
    x0 = X[0:1]; spread = {}
    for b in (2, 8, 32, 128):                                                  # the same sample, placed in random batches of size b
        outs = [N.BatchNorm1d(8).forward(np.vstack([x0, X[rng.choice(np.arange(1, 256), b - 1, replace=False)]]))[0] for _ in range(300)]
        spread[b] = float(np.std(outs, axis=0).mean())
    one = N.BatchNorm1d(8).forward(X[:1])
    def final_loss(norm, lr):
        net = N.build_deep_net(32, 8, "relu", "he_normal", norm=norm, n_out=3, seed=seed); r = np.random.default_rng(seed); Xd = r.standard_normal((256, 32)); y = (Xd[:, 0] > 0).astype(int) + (Xd[:, 1] > 0).astype(int)
        for _ in range(60):
            loss, d = N.softmax_cross_entropy(net.forward(Xd), y)
            if not np.isfinite(loss) or loss > 1e6: return float("inf")
            net.backward(d); net.step(lr)
        return float(loss)
    return {"running_mean_error_after_40_passes": float(np.abs(bn.running_mean - X.mean(0)).max()), "eval_output_mean_std": (float(ev.mean()), float(ev.std())),
            "output_of_one_sample_std_across_random_batches_by_batch_size": spread, "batch_of_one_output_all_zero": bool(np.allclose(one, 0)),
            "final_loss_lr0.5": {"no_norm": final_loss(None, 0.5), "batchnorm": final_loss("batch", 0.5)}}


def bn_gap(X) -> float:
    bn = N.BatchNorm1d(16); batch = X[:4]; tr = bn.forward(batch); bn.training = False; return float(np.abs(tr - bn.forward(batch)).max())


# ---- 3.31 layer normalization --------------------------------------------------------------------------------------------------------------------------
def ex_3_31(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X = rng.standard_normal((64, 16)) * 4 + 2; ln = N.LayerNorm(16)
    full, alone = ln.forward(X), N.LayerNorm(16).forward(X[:1])
    seq = rng.standard_normal((2, 5, 16)); y = N.LayerNorm(16).forward(seq)                              # works on (B, T, D) — normalizes the last axis
    return {"sample_independent_of_batch": bool(np.allclose(full[:1], alone)), "per_sample_mean_std": (float(abs(full.mean(-1)).max()), float(abs(full.std(-1) - 1).max())), "works_for_batch_of_one": bool(np.isfinite(alone).all() and alone.std() > 0.9),
            "sequence_input_shape": y.shape, "no_running_stats_needed": not hasattr(ln, "running_mean"), "batchnorm_train_vs_eval_output_gap_on_same_input": bn_gap(X)}


# ---- 3.32 RMSNorm ---------------------------------------------------------------------------------------------------------------------------------------------
def ex_3_32(seed: int = 0) -> dict:
    import time
    rng = np.random.default_rng(seed); D = 1024; X = rng.standard_normal((4096, D)) * 3; Xs = X + 5.0
    ln, rn = N.LayerNorm(D), N.RMSNorm(D)
    centered, shifted = np.abs(ln.forward(X) - rn.forward(X)).max(), np.abs(ln.forward(Xs) - rn.forward(Xs)).max()
    scale_inv = float(np.abs(rn.forward(X) - rn.forward(10 * X)).max()); ln_shift = float(np.abs(ln.forward(X) - ln.forward(X + 7)).max()); rn_shift = float(np.abs(rn.forward(X) - rn.forward(X + 7)).max())
    t = {}
    for name, layer in (("layernorm", ln), ("rmsnorm", rn)):
        t0 = time.perf_counter()
        for _ in range(20): layer.forward(X)
        t[name] = (time.perf_counter() - t0) / 20
    return {"max_diff_when_inputs_are_zero_mean": float(centered), "max_diff_when_inputs_have_mean_5": float(shifted), "scale_invariance_rms(x vs 10x)": scale_inv,
            "shift_invariance_layernorm": ln_shift, "shift_invariance_rmsnorm(not invariant)": rn_shift, "params_ln_vs_rms": (ln.n_params(), rn.n_params()), "time_ratio_ln_over_rms": t["layernorm"] / t["rmsnorm"]}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
