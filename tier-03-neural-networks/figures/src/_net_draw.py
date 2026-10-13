"""Shared drawing code for the 2-3-1 network animations (forward pass, backpropagation)."""
import numpy as np
import matplotlib.pyplot as plt
from _common import figstyle as fs
import nn as N

W1 = np.array([[0.5, -1.0], [2.0, 0.1], [-0.3, 0.8]]); B1 = np.array([0.0, 1.0, -0.2]); W2 = np.array([[1.0, -1.0, 0.5]]); B2 = np.array([0.1]); X = np.array([[1.0, 2.0]])
POS = {"in": [(0.0, 0.7), (0.0, -0.7)], "h": [(3.0, 1.3), (3.0, 0.0), (3.0, -1.3)], "out": [(6.0, 0.0)]}
LABEL_X = {"in": "input", "h": "hidden (ReLU)", "out": "output"}


def build_net():
    m = N.MLP([2, 3, 1], activation="relu"); m.layers[0].W, m.layers[0].b, m.layers[2].W, m.layers[2].b = W1.copy(), B1.copy(), W2.copy(), B2.copy(); return m


def new_axes():
    fig = plt.figure(figsize=(10, 5.6)); ax = fig.add_axes([0.02, 0.2, 0.96, 0.62]); ax.set_xlim(-1.6, 7.6); ax.set_ylim(-1.65, 1.75); ax.set_aspect("equal"); ax.axis("off"); return fig, ax


def draw_edges(ax, layer_a, layer_b, weights, colors, widths, alphas, labels=True, label_fs=9):
    for j, pb in enumerate(POS[layer_b]):
        for i, pa in enumerate(POS[layer_a]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=colors[j][i], lw=widths[j][i], alpha=alphas[j][i], solid_capstyle="round", zorder=1)
            if labels:
                t = 0.33 if (i + j) % 2 == 0 else 0.62; x = pa[0] + t * (pb[0] - pa[0]); y = pa[1] + t * (pb[1] - pa[1])
                ax.text(x, y + 0.07, f"{weights[j][i]:.2f}" if abs(weights[j][i]) >= 0.005 else "0", color=fs.FG, fontsize=label_fs, ha="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.12", fc=fs.BG, ec="none", alpha=0.85))


def draw_nodes(ax, values, fills, edge_colors=None, texts=None, r=0.3):
    for layer in ("in", "h", "out"):
        for k, (x, y) in enumerate(POS[layer]):
            c = plt.Circle((x, y), r, fc=fills[layer][k][0], ec=(edge_colors or {}).get(layer, [fs.FG] * 3)[k], lw=2.0, alpha=fills[layer][k][1], zorder=3); ax.add_patch(c)
            v = values[layer][k]
            ax.text(x, y, "" if v is None else v, color=fs.BG if fills[layer][k][1] > 0.6 else fs.FG, fontsize=11, ha="center", va="center", fontweight="bold", zorder=5)
    for layer, name in LABEL_X.items(): ax.text(POS[layer][0][0], 1.78, name, color=fs.MUTED, ha="center", fontsize=11)


def banner(fig, title, step, eq, note=""):
    fig.text(0.5, 0.94, title, ha="center", color=fs.FG, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.88, step, ha="center", color=fs.YELLOW, fontsize=12)
    fig.text(0.5, 0.115, eq, ha="center", color=fs.FG, fontsize=13)
    if note: fig.text(0.5, 0.05, note, ha="center", color=fs.MUTED, fontsize=10)
