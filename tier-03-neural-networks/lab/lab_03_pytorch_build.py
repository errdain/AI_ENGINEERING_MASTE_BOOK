"""Tier 3 Lab, part 3: the same networks built a second time in PyTorch ("manually first, then in PyTorch").

The NumPy library (code/nn.py) is the reference implementation; here PyTorch reproduces it exactly, which verifies both and
shows what the framework automates (graph construction, autograd, optimizers, init helpers).
Requires torch. Verified with torch 2.14.1 (CPU, float64 for exact comparison).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

import nn as N

torch.set_default_dtype(torch.float64)


def _copy_mlp_to_torch(m: N.MLP, act=tnn.Tanh) -> tnn.Sequential:
    layers = []
    for l in m.layers:
        if isinstance(l, N.Dense):
            lin = tnn.Linear(l.W.shape[1], l.W.shape[0]); lin.weight.data = torch.tensor(l.W.copy()); lin.bias.data = torch.tensor(l.b.copy()); layers.append(lin)
        else: layers.append(act())
    return tnn.Sequential(*layers)


def t_3_trajectory_match(steps: int = 100, wd: float = 0.01, lr: float = 0.1, seed: int = 0) -> dict:
    """Full-batch SGD with weight decay for the same initial weights: NumPy and PyTorch weights stay identical."""
    rng = np.random.default_rng(seed); X, y = rng.standard_normal((50, 4)), rng.integers(0, 3, 50)
    m = N.MLP([4, 8, 3], activation="tanh", init="xavier_normal", seed=seed); tm = _copy_mlp_to_torch(m)
    opt = torch.optim.SGD(tm.parameters(), lr=lr, weight_decay=wd)                       # note: torch applies weight decay to biases as well
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(steps):
        loss, d = N.softmax_cross_entropy(m.forward(X), y); m.backward(d)
        for l in m.layers:                                                                   # same rule as torch.optim.SGD(weight_decay=wd) (decay on W and b)
            if isinstance(l, N.Dense): l.W -= lr * (l.dW + wd * l.W); l.b -= lr * (l.db + wd * l.b)
        opt.zero_grad(); F.cross_entropy(tm(Xt), yt).backward(); opt.step()
    diffs = [float(np.abs(l.W - tl.weight.detach().numpy()).max()) for l, tl in zip([l for l in m.layers if isinstance(l, N.Dense)], [t for t in tm if isinstance(t, tnn.Linear)])]
    return {"max_weight_difference_after_steps": max(diffs), "final_loss_numpy": N.softmax_cross_entropy(m.forward(X), y)[0], "final_loss_torch": F.cross_entropy(tm(Xt), yt).item()}


def t_3_autograd_graph() -> dict:
    x = torch.tensor([2.0, -3.0], requires_grad=True); w = torch.tensor([0.5, -0.4], requires_grad=True)
    z = (w * x).sum(); a = torch.tanh(z); a.backward()
    chain, fn = [], a.grad_fn
    while fn is not None:
        chain.append(type(fn).__name__); fn = fn.next_functions[0][0] if fn.next_functions else None
    with torch.no_grad(): v = w * 2
    return {"grad_fn_chain": chain, "is_leaf(x,w,z)": (x.is_leaf, w.is_leaf, z.is_leaf), "dx": x.grad.tolist(),
            "manual_dx": ((1 - np.tanh(0.5 * 2 + -0.4 * -3) ** 2) * np.array([0.5, -0.4])).tolist(), "result_under_no_grad_requires_grad": v.requires_grad}


def t_3_standard_training_loop(seed: int = 0) -> dict:
    torch.manual_seed(seed); rng = np.random.default_rng(seed); P = rng.standard_normal((800, 2)); yc = (np.linalg.norm(P, axis=1) > 1.0).astype(int)
    X, y = torch.tensor(P), torch.tensor(yc)
    model = tnn.Sequential(tnn.Linear(2, 16), tnn.Tanh(), tnn.Linear(16, 2)); opt = torch.optim.SGD(model.parameters(), lr=0.3); loss_fn = tnn.CrossEntropyLoss(); losses = []
    for _ in range(1500):
        opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step(); losses.append(loss.item())
    acc = float((model(X).argmax(1) == y).double().mean())
    return {"params": sum(p.numel() for p in model.parameters()), "loss_start_end": (losses[0], losses[-1]), "train_acc": acc}


def t_3_layers_vs_torch(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X = rng.standard_normal((12, 6)) * 3 + 1
    ln, tln = N.LayerNorm(6), tnn.LayerNorm(6); rn, trn = N.RMSNorm(6), tnn.RMSNorm(6, eps=1e-6)
    xt = torch.tensor(X); d = lambda a, b: float(np.abs(a - b.detach().numpy()).max())
    dr = tnn.Dropout(0.5); dr.eval()
    return {"layernorm": d(ln.forward(X), tln(xt)), "rmsnorm": d(rn.forward(X), trn(xt)), "dropout_eval_identity": bool(torch.equal(dr(xt), xt)),
            "param_counts": {"LayerNorm(6)": sum(p.numel() for p in tln.parameters()), "RMSNorm(6)": sum(p.numel() for p in trn.parameters()), "BatchNorm1d(6)": sum(p.numel() for p in tnn.BatchNorm1d(6).parameters())}}


def t_3_init_helpers(seed: int = 0) -> dict:
    torch.manual_seed(seed); w = torch.empty(512, 256)
    tnn.init.xavier_normal_(w); xs = float(w.std()); tnn.init.kaiming_normal_(w, nonlinearity="relu"); ks = float(w.std()); tnn.init.xavier_uniform_(w); xu = float(w.abs().max())
    return {"xavier_normal_std": (xs, float(np.sqrt(2 / (256 + 512)))), "kaiming_normal_std(relu)": (ks, float(np.sqrt(2 / 256))), "xavier_uniform_bound": (xu, float(np.sqrt(6 / (256 + 512)))),
            "calculate_gain": {"tanh": tnn.init.calculate_gain("tanh"), "relu": tnn.init.calculate_gain("relu"), "sigmoid": tnn.init.calculate_gain("sigmoid")}}


def t_3_clip_grad_matches() -> dict:
    rng = np.random.default_rng(0); net = N.build_deep_net(6, 2, "relu", "he_normal", n_out=3, seed=0); X, y = rng.standard_normal((10, 6)) * 20, rng.integers(0, 3, 10)
    _, d = N.softmax_cross_entropy(net.forward(X), y); net.backward(d)
    ps = [torch.nn.Parameter(torch.tensor(getattr(l, g).copy())) for l, _, _, g in net.parameters()]
    for p, (l, _, _, g) in zip(ps, net.parameters()): p.grad = torch.tensor(getattr(l, g).copy())
    tn = float(torch.nn.utils.clip_grad_norm_(ps, 0.5)); ours = N.clip_grad_norm(net, 0.5)
    return {"norm_before": (ours, tn), "max_grad_difference_after_clip": max(float(np.abs(getattr(l, g) - p.grad.numpy()).max()) for p, (l, _, _, g) in zip(ps, net.parameters()))}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("t_"):
            print(name, fn())
