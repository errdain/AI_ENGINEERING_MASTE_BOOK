"""Figure 1.1-1 (SVG): the ladder of containers for numbers: scalar, vector, matrix, tensor, with the NumPy shape of each."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from _common import OUT, fs

fig, axs = plt.subplots(1, 4, figsize=(15, 4.8))
for ax in axs: ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
def cell(ax, x, y, text, c, w=1.5, h=1.5): ax.add_patch(Rectangle((x, y), w * 0.94, h * 0.94, fc=c + "55", ec=c, lw=1.6)); ax.text(x + w * 0.47, y + h * 0.47, text, ha="center", va="center", color=fs.FG, fontsize=11, fontweight="bold")
# scalar
cell(axs[0], 3.4, 3.2, "21.5", fs.TEAL, 3.2, 2.6); axs[0].set_title("scalar\none number", fontsize=13, color=fs.TEAL, fontweight="bold")
axs[0].text(5, 1.9, "shape  ()", ha="center", color=fs.YELLOW, fontsize=12); axs[0].text(5, 0.8, "e.g. today's temperature", ha="center", color=fs.MUTED, fontsize=9.5)
# vector
for i, v in enumerate(["21.5", "40", "1013"]): cell(axs[1], 0.3 + i * 3.1, 3.4, v, fs.BLUE, 3.0, 2.2)
axs[1].set_title("vector\na list of numbers", fontsize=13, color=fs.BLUE, fontweight="bold"); axs[1].text(5, 1.9, "shape  (3,)", ha="center", color=fs.YELLOW, fontsize=12); axs[1].text(5, 0.8, "e.g. temperature, humidity, pressure", ha="center", color=fs.MUTED, fontsize=9.5)
# matrix
vals = [["21", "40", "1013"], ["19", "55", "1009"]]
for r in range(2):
    for c in range(3): cell(axs[2], 0.3 + c * 3.1, 4.9 - r * 2.1, vals[r][c], fs.PURPLE, 3.0, 2.0)
axs[2].set_title("matrix\na grid of numbers", fontsize=13, color=fs.PURPLE, fontweight="bold"); axs[2].text(5, 1.9, "shape  (2, 3)", ha="center", color=fs.YELLOW, fontsize=12); axs[2].text(5, 0.8, "e.g. 2 days × 3 measurements", ha="center", color=fs.MUTED, fontsize=9.5)
# tensor (three stacked grids)
for k, col in enumerate([fs.ORANGE, fs.ORANGE, fs.ORANGE]):
    ox, oy = 1.0 + k * 1.5, 1.0 + k * 1.3
    for r in range(2):
        for c in range(3): axs[3].add_patch(Rectangle((ox + c * 2.0, oy + 3.0 - r * 1.4), 1.9, 1.3, fc=col + ("33" if k < 2 else "66"), ec=col, lw=1.2))
axs[3].set_title("tensor\na block of numbers", fontsize=13, color=fs.ORANGE, fontweight="bold"); axs[3].text(5, 0.1, "shape  (3, 2, 3)", ha="center", color=fs.YELLOW, fontsize=12); axs[3].text(5, -0.9, "e.g. 3 weeks × 2 days × 3 measurements", ha="center", color=fs.MUTED, fontsize=9.5)
fig.suptitle("Everything in AI is built from numbers: first one number, then lists, grids and blocks of them", fontsize=14, fontweight="bold", y=1.0)
fig.text(0.5, 0.03, "a scalar has 0 dimensions, a vector 1, a matrix 2, a tensor 3 or more  (the number of dimensions is the number of numbers in the shape)", ha="center", color=fs.MUTED, fontsize=10)
fig.savefig(OUT / "fig_1_01_dimension_ladder.svg", bbox_inches="tight"); plt.close(fig)
