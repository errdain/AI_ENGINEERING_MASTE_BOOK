"""Figure 3.9-1 (SVG): the activation functions of 3.10-3.14 and their derivatives, computed from nn.ACTIVATIONS."""
import numpy as np
import matplotlib.pyplot as plt
from _common import OUT, figstyle as fs
import nn as N

x = np.linspace(-6, 6, 1201)
names = ["sigmoid", "tanh", "relu", "gelu", "silu"]; cols = [fs.BLUE, fs.TEAL, fs.RED, fs.YELLOW, fs.PURPLE]
fig, (a, b) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, gridspec_kw=dict(hspace=0.12))
for n, c in zip(names, cols):
    f, df = N.ACTIVATIONS[n]; a.plot(x, f(x), color=c, lw=2.4, label={"relu": "ReLU", "gelu": "GELU", "silu": "SiLU (Swish)"}.get(n, n)); b.plot(x, df(x), color=c, lw=2.4)
a.set_ylim(-1.5, 3.2); b.set_ylim(-0.25, 1.28)
a.axhline(0, color=fs.GRID, lw=1); a.axvline(0, color=fs.GRID, lw=1); b.axhline(0, color=fs.GRID, lw=1); b.axvline(0, color=fs.GRID, lw=1)
a.set_ylabel("activation  φ(z)"); b.set_ylabel("derivative  φ′(z)"); b.set_xlabel("pre-activation  z = w·x + b")
a.legend(loc="upper left", ncol=5, frameon=True, fontsize=10)
a.set_title("Activation functions (top) and their derivatives (bottom)", pad=12, fontsize=14, fontweight="bold")
b.axhline(0.25, color=fs.BLUE, lw=1, ls=(0, (4, 4))); b.text(4.2, 0.28, "sigmoid′ ≤ 0.25", color=fs.BLUE, fontsize=10)
b.text(-5.9, 0.07, "ReLU: derivative is exactly 0 for z < 0 (dead zone)", color=fs.RED, fontsize=10)
b.text(1.6, 1.16, "GELU/SiLU derivative briefly exceeds 1", color=fs.YELLOW, fontsize=10)
a.annotate("SiLU minimum ≈ −0.278", xy=(-1.28, N.silu(np.array(-1.28))), xytext=(1.0, -1.05), color=fs.PURPLE, arrowprops=dict(arrowstyle="->", color=fs.PURPLE), fontsize=10)
a.annotate("GELU minimum ≈ −0.17", xy=(-0.75, N.gelu(np.array(-0.75))), xytext=(1.0, -0.6), color=fs.YELLOW, arrowprops=dict(arrowstyle="->", color=fs.YELLOW), fontsize=10)
a.text(3.6, 1.12, "sigmoid and tanh saturate at 1", color=fs.BLUE, fontsize=10); a.text(-5.9, -0.88, "tanh saturates at −1", color=fs.TEAL, fontsize=10)
for ax in (a, b): ax.grid(True, alpha=0.4)
fig.savefig(OUT / "fig_3_09_activations.svg"); print("ok")
