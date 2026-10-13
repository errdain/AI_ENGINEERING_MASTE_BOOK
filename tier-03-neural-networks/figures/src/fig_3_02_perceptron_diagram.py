"""Figure 3.2-1 (SVG): one perceptron computing the AND gate, with the arithmetic written on the picture."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from _common import OUT, fs

w = np.array([1.0, 1.0]); b = -1.5
fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.2))
for ax, x in zip(axs, (np.array([1.0, 1.0]), np.array([1.0, 0.0]))):
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off"); z = float(w @ x + b); out = int(z > 0)
    for i, (xi, y_) in enumerate(zip(x, (4.4, 1.8))):
        ax.add_patch(Circle((1.0, y_), 0.55, fc=fs.TEAL + "55", ec=fs.TEAL, lw=2)); ax.text(1.0, y_, f"x{i + 1}={xi:g}", ha="center", va="center", color=fs.FG, fontsize=10.5, fontweight="bold")
        ax.annotate("", xy=(5.0, 3.1 + (0.35 if i == 0 else -0.35)), xytext=(1.6, y_), arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=2.4)); ax.text(3.0, y_ + (-0.55 if i == 0 else 0.45) + (0.3 if i == 0 else -0.2), f"weight w{i + 1} = {w[i]:g}", color=fs.BLUE, fontsize=10.5)
    ax.add_patch(FancyBboxPatch((5.0, 2.5), 2.6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.3", fc=fs.PURPLE + "33", ec=fs.PURPLE, lw=2.2)); ax.text(6.3, 3.1, "add up", ha="center", va="center", color=fs.FG, fontsize=11.5); ax.text(6.3, 1.7, f"z = {w[0]:g}·{x[0]:g} + {w[1]:g}·{x[1]:g} + ({b:g}) = {z:g}", ha="center", color=fs.YELLOW, fontsize=11.5)
    ax.text(6.3, 5.0, f"bias b = {b:g}  (the threshold)", ha="center", color=fs.PURPLE, fontsize=10.5); ax.annotate("", xy=(6.3, 3.75), xytext=(6.3, 4.8), arrowprops=dict(arrowstyle="-|>", color=fs.PURPLE, lw=2))
    ax.annotate("", xy=(9.0, 3.1), xytext=(7.6, 3.1), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2.4)); ax.text(8.3, 3.45, "z > 0 ?", ha="center", color=fs.FG, fontsize=10.5)
    ax.add_patch(Circle((9.9, 3.1), 0.8, fc=(fs.RED if out else fs.BLUE) + "55", ec=(fs.RED if out else fs.BLUE), lw=2.4)); ax.text(9.9, 3.1, f"output\n{out}", ha="center", va="center", color=fs.FG, fontsize=12, fontweight="bold")
    ax.set_title(f"inputs ({x[0]:g}, {x[1]:g}):  {z:g} is {'above' if out else 'not above'} 0, so the output is {out}", fontsize=11.5, color=(fs.RED if out else fs.BLUE))
fig.suptitle("One perceptron computing AND: multiply each input by its weight, add the bias, output 1 if the total is above 0", fontsize=13, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_3_02_perceptron_diagram.svg", bbox_inches="tight"); plt.close(fig)
