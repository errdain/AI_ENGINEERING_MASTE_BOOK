"""Tier 1 Lab, part 4: PyTorch checks for claims made in the concept files.

Requires `torch`. Verified with torch 2.14.1 (CPU). Behaviour of defaults (e.g. `torch.load`) is
version-sensitive: re-run these tests when upgrading PyTorch.
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F

from calculus_prob import tiny_net_forward_backward


def t_1_4_tensor_basics() -> dict:
    x = torch.zeros(32, 128, 768, dtype=torch.float32, device="cpu")
    return {"ndim": x.ndim, "shape": tuple(x.shape), "numel": x.numel(), "MiB": x.numel() * x.element_size() / 2**20}


def t_1_12_transpose_view() -> dict:
    A = torch.arange(12.0).reshape(3, 4)
    At = A.T
    try:
        At.view(12)
        view_ok = True
    except RuntimeError:
        view_ok = False
    return {"is_view": At.data_ptr() == A.data_ptr(), "contiguous": At.is_contiguous(), "view_after_transpose_ok": view_ok,
            "reshape_ok": At.reshape(12).shape == (12,), "contiguous_then_view_ok": At.contiguous().view(12).shape == (12,)}


def t_1_25_autograd_vs_manual(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(3), 0.7
    W1, W2 = rng.standard_normal((4, 3)), rng.standard_normal(4)
    _, dW1, dW2 = tiny_net_forward_backward(x, W1, W2, y)
    tW1 = torch.tensor(W1, requires_grad=True); tW2 = torch.tensor(W2, requires_grad=True)
    loss = (tW2 @ torch.tanh(tW1 @ torch.tensor(x)) - y) ** 2
    loss.backward()
    return {"dW1_max_diff": float(np.abs(tW1.grad.numpy() - dW1).max()), "dW2_max_diff": float(np.abs(tW2.grad.numpy() - dW2).max())}


def t_1_30_var_defaults() -> dict:
    x = np.array([2.0, 4, 4, 4, 5, 5, 7, 9])
    t = torch.tensor(x)
    return {"numpy_default": float(np.var(x)), "torch_default": float(t.var()), "torch_correction0": float(t.var(correction=0))}


def t_1_37_cross_entropy_pitfall(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 5, generator=g); target = torch.tensor([0, 2, 4, 1])
    ours = -torch.log_softmax(logits, -1)[torch.arange(4), target].mean()
    return {"F.cross_entropy": float(F.cross_entropy(logits, target)), "manual": float(ours),
            "double_softmax": float(F.cross_entropy(torch.softmax(logits, -1), target))}  # wrong on purpose


def t_1_38_kl_convention() -> dict:
    p = torch.tensor([0.5, 0.5]); q = torch.tensor([0.9, 0.1])
    kl_pq = F.kl_div(q.log(), p, reduction="sum")  # input = log q, target = p  => KL(p || q) in nats
    return {"F.kl_div(log q, p)": float(kl_pq), "manual_nats": float((p * (p / q).log()).sum())}


def t_1_40_dtype_table() -> dict:
    out = {}
    for dt in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        fi = torch.finfo(dt)
        out[str(dt)] = {"bits": fi.bits, "eps": float(fi.eps), "max": float(fi.max), "tiny": float(fi.tiny)}
    return out


def t_1_40_bf16_vs_fp16() -> dict:
    big = torch.tensor(70000.0)
    small_step = torch.tensor(1.0) + torch.tensor(0.001)
    return {"fp16_70000": float(big.half()), "bf16_70000": float(big.bfloat16()),
            "fp16_1.001": float(small_step.half()), "bf16_1.001": float(small_step.bfloat16())}


def t_1_43_logsumexp(seed: int = 0) -> dict:
    x = torch.randn(1000, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 50
    from info_numerics import logsumexp
    return {"torch": float(torch.logsumexp(x, 0)), "ours": float(logsumexp(x.numpy()))}


def t_1_44_loss_scaling() -> dict:
    """A gradient of 1e-8 underflows in fp16; scaling the loss by 2**15 keeps it, unscaling restores it."""
    g = torch.tensor(1e-8)
    S = 2.0 ** 15
    return {"fp16_unscaled": float(g.half()), "fp16_scaled": float((g * S).half()), "recovered": float((g * S).half().float() / S)}


def t_1_44_int8_wrap() -> dict:
    return {"int8_127_plus_1": int(torch.tensor([127], dtype=torch.int8)[0] + torch.tensor(1, dtype=torch.int8))}


def t_1_44_nan_propagation() -> dict:
    inf = torch.tensor(float("inf"))
    return {"inf-inf": float(inf - inf), "0*inf": float(torch.tensor(0.0) * inf), "exp(1000)": float(torch.exp(torch.tensor(1000.0))),
            "naive_softmax_nan": bool(torch.isnan(torch.exp(torch.tensor([1000.0, 1001.0])) / torch.exp(torch.tensor([1000.0, 1001.0])).sum()).any()),
            "stable_softmax_finite": bool(torch.isfinite(torch.softmax(torch.tensor([1000.0, 1001.0]), 0)).all())}


class _Evil:  # stands in for an arbitrary pickled object
    pass


def t_1_4_safe_load() -> dict:
    """Plain tensors load with weights_only=True; an arbitrary pickled object does not."""
    buf = io.BytesIO(); torch.save({"w": torch.ones(2)}, buf); buf.seek(0)
    tensors_ok = torch.load(buf, weights_only=True)["w"].tolist() == [1.0, 1.0]
    buf2 = io.BytesIO(); torch.save({"obj": _Evil()}, buf2); buf2.seek(0)
    try:
        torch.load(buf2, weights_only=True)
        blocked = False
    except Exception:
        blocked = True
    buf3 = io.BytesIO(); torch.save({"obj": _Evil()}, buf3); buf3.seek(0)
    try:
        torch.load(buf3)  # default behaviour of the installed version
        default_blocks = False
    except Exception:
        default_blocks = True
    return {"tensors_ok": tensors_ok, "object_blocked_with_weights_only": blocked, "default_blocks_objects": default_blocks,
            "torch_version": torch.__version__}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("t_"):
            print(name, fn())
