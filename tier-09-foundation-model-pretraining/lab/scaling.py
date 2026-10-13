"""Tier 9 lab, part 4: scaling-law helpers (9.14, 9.15, 9.20, 9.21).

Parametric loss (Hoffmann et al. 2022, "Chinchilla"):   L(N, D) = E + A / N^alpha + B / D^beta
 N = parameters, D = training tokens.  Compute C ~ 6 N D FLOPs.

The default constants below are the values I recall from that paper's parametric fit
(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28).  REVIEW_REQUIRED: verify against the paper before
quoting; they describe THEIR data/tokenizer/architecture, not yours -- always refit on your own runs.
"""
from __future__ import annotations

import numpy as np

CHINCHILLA = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)  # REVIEW_REQUIRED


def loss(N: float, D: float, k: dict = CHINCHILLA) -> float:
    return k["E"] + k["A"] / N ** k["alpha"] + k["B"] / D ** k["beta"]


def compute_optimal(C: float, k: dict = CHINCHILLA) -> tuple[float, float]:
    """Minimise L(N, D) subject to 6*N*D = C  (Lagrange): returns (N_opt, D_opt)."""
    a, b = k["alpha"], k["beta"]
    G = (a * k["A"] / (b * k["B"])) ** (1.0 / (a + b))
    N = G * (C / 6.0) ** (b / (a + b))
    return N, C / (6.0 * N)


def fit_power_law(N: np.ndarray, L: np.ndarray, E_grid: np.ndarray | None = None) -> dict:
    """Fit L = E + A * N^-alpha: grid-search the irreducible E, least squares on log(L-E) vs log N."""
    N, L = np.asarray(N, float), np.asarray(L, float)
    E_grid = np.linspace(0, 0.98 * L.min(), 200) if E_grid is None else E_grid
    best = None
    for E in E_grid:
        y = np.log(L - E)
        slope, icpt = np.polyfit(np.log(N), y, 1)
        err = float(np.sum((np.polyval([slope, icpt], np.log(N)) - y) ** 2))
        if best is None or err < best[0]:
            best = (err, E, -slope, float(np.exp(icpt)))
    return {"E": float(best[1]), "alpha": float(best[2]), "A": best[3], "sse": best[0]}
