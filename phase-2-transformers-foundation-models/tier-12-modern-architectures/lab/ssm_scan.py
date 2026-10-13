"""Tiny NumPy demo: a (selective) diagonal state-space layer computed FOUR equivalent ways.

Model, per channel-free toy (scalar input x_t, state h_t in R^N, scalar output y_t):
    h_t = a_t * h_{t-1} + B_t * x_t        (a_t, B_t in R^N; elementwise; a_t in (0,1) = decay)
    y_t = <C_t, h_t>

  1. recurrent_scan   : O(T) loop, O(N) state     -- how SSMs/Mamba DECODE
  2. matrix_form      : y = M x, M[t,s] = <C_t, (prod_{r=s+1..t} a_r) * B_s>, lower-triangular
                        -> a *masked, decayed attention* matrix (the "SSM = attention" view)
  3. parallel_scan    : log-depth associative scan over affine maps (how SSMs TRAIN on GPUs)
  4. conv_form        : only when a, B, C are time-invariant: y = K * x with K_k = <C, a^k B>

All four agree to float tolerance (see test_ssm_scan.py).  Not Mamba itself -- no gating,
no convolution, no discretization -- just the equivalence at the heart of Tier 12.17-12.19.
"""
from __future__ import annotations
import numpy as np


def make_inputs(T=16, N=4, selective=True, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    if selective:
        a = 1 / (1 + np.exp(-rng.standard_normal((T, N))))   # sigmoid -> (0,1)
        B = rng.standard_normal((T, N))
        C = rng.standard_normal((T, N))
    else:
        a = np.tile(1 / (1 + np.exp(-rng.standard_normal(N))), (T, 1))
        B = np.tile(rng.standard_normal(N), (T, 1))
        C = np.tile(rng.standard_normal(N), (T, 1))
    return x, a, B, C


def recurrent_scan(x, a, B, C, return_states=False):
    T, N = a.shape
    h = np.zeros(N)
    y = np.empty(T)
    hs = np.empty((T, N))
    for t in range(T):
        h = a[t] * h + B[t] * x[t]
        y[t] = C[t] @ h
        hs[t] = h
    return (y, hs) if return_states else y


def matrix_form(a, B, C):
    """Return the T x T lower-triangular mixing matrix M (y = M @ x)."""
    T, N = a.shape
    M = np.zeros((T, T))
    for t in range(T):
        decay = np.ones(N)                      # prod_{r=s+1..t} a_r, built backwards from s=t
        for s in range(t, -1, -1):
            M[t, s] = C[t] @ (decay * B[s])
            if s > 0:
                decay = decay * a[s]
    return M


def parallel_scan(x, a, B, C):
    """Hillis-Steele inclusive scan over affine maps (A, b): h -> A*h + b.
    Composition (A2,b2)o(A1,b1) = (A2*A1, A2*b1 + b2).  ceil(log2 T) rounds."""
    A = a.copy()
    b = B * x[:, None]
    T = len(x)
    d = 1
    while d < T:
        A_new, b_new = A.copy(), b.copy()
        A_new[d:] = A[d:] * A[:-d]
        b_new[d:] = A[d:] * b[:-d] + b[d:]
        A, b = A_new, b_new
        d *= 2
    return np.einsum("tn,tn->t", C, b)          # h_t = b_t (initial state is zero)


def conv_form(x, a, B, C):
    """Time-invariant case only: y_t = sum_{k<=t} K_k x_{t-k}, K_k = <C, a^k * B>."""
    if not (np.allclose(a, a[0]) and np.allclose(B, B[0]) and np.allclose(C, C[0])):
        raise ValueError("conv_form needs time-invariant a, B, C (selectivity breaks the convolution view)")
    T = len(x)
    K = np.array([C[0] @ (a[0] ** k * B[0]) for k in range(T)])
    return np.array([K[: t + 1][::-1] @ x[: t + 1] for t in range(T)])


# ---- bonus: causal LINEAR attention is the a == 1 SSM with an N x d_v matrix state
def linear_attention_recurrent(q, k, v):
    T, dk = q.shape
    S = np.zeros((dk, v.shape[1]))
    out = np.empty((T, v.shape[1]))
    for t in range(T):
        S = S + np.outer(k[t], v[t])
        out[t] = q[t] @ S
    return out


def linear_attention_matrix(q, k, v):
    return np.tril(q @ k.T) @ v


def softmax_attention(q, k, v):
    T, d = q.shape
    s = q @ k.T / np.sqrt(d)
    s[np.triu_indices(T, 1)] = -np.inf
    w = np.exp(s - s.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)) @ v


# ---- cost model (per generated token, per layer) used by figures / explorers
def decode_cost_attention(n, d):
    """FLOPs for one new token vs n cached tokens (QK and AV)."""
    return 4 * n * d


def decode_cost_ssm(d, N):
    return 6 * d * N


if __name__ == "__main__":
    x, a, B, C = make_inputs(16, 4)
    y1 = recurrent_scan(x, a, B, C)
    print("matrix  max err:", np.abs(matrix_form(a, B, C) @ x - y1).max())
    print("pscan   max err:", np.abs(parallel_scan(x, a, B, C) - y1).max())
    xi, ai, Bi, Ci = make_inputs(16, 4, selective=False)
    print("conv    max err:", np.abs(conv_form(xi, ai, Bi, Ci) - recurrent_scan(xi, ai, Bi, Ci)).max())
