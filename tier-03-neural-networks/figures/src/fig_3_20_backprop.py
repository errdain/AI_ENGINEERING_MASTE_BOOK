"""Figure 3.20-1 (GIF): backpropagation through the same 2-3-1 ReLU network; every gradient shown is computed by nn.py (MSE loss, target 0)."""
import numpy as np
from _common import OUT, figstyle as fs
from _net_draw import *

m = build_net(); out = m.forward(X); loss, d = N.mse_loss(out, np.zeros_like(out)); m.backward(d)
z1 = X @ W1.T + B1; a1 = np.maximum(0, z1); dW2, dW1 = m.layers[2].dW, m.layers[0].dW
dyhat = float(d[0, 0]); da1 = (d @ W2)[0]; dz1 = da1 * (z1[0] > 0)                  # manual chain rule must agree with the library
assert np.allclose(dW2, d.T @ a1) and np.allclose(dW1, np.outer(dz1, X[0])) and np.isclose(dyhat, 2 * out[0, 0])
f = lambda v: "0" if abs(v) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")
frames, PH = [], 9

def style(G, thr):
    g = np.abs(G).max() or 1
    return [[(fs.BLUE if G[j][i] > 0 else fs.RED) if abs(G[j][i]) > 1e-9 else fs.MUTED for i in range(G.shape[1])] for j in range(G.shape[0])], \
           [[1.2 + 4.5 * abs(G[j][i]) / g for i in range(G.shape[1])] for j in range(G.shape[0])], [[0.9 if abs(G[j][i]) > 1e-9 else 0.3 for i in range(G.shape[1])] for j in range(G.shape[0])]

def render(phase, p):
    fig, ax = new_axes(); e = fs.ease(p)
    ec1 = [[fs.MUTED] * 2 for _ in range(3)]; ew1 = [[1.2] * 2 for _ in range(3)]; ea1 = [[0.3] * 2 for _ in range(3)]; lab1 = W1
    ec2 = [[fs.MUTED] * 3]; ew2 = [[1.2] * 3]; ea2 = [[0.3] * 3]; lab2 = W2
    vals = {"in": [f(1), f(2)], "h": [f(v) for v in a1[0]], "out": [f(out[0, 0])]}
    fills = {"in": [(fs.TEAL, 0.6)] * 2, "h": [(fs.TEAL, 0.6) if a1[0][j] > 0 else (fs.RED, 0.35) for j in range(3)], "out": [(fs.ORANGE, 0.6)]}
    title = "Backpropagation: the blame flows from the loss back to every weight"
    step, eq = "Forward pass done: ŷ = −2.55, target 0  →  loss L = (ŷ − 0)² = " + f(loss), "L = (ŷ − t)² = " + f(loss)
    if phase >= 1:
        step, eq = "Step 1 — start at the output:  ∂L/∂ŷ = 2(ŷ − t)", "∂L/∂ŷ = 2·(−2.55) = " + f(dyhat)
        vals["out"] = ["δ=" + f(dyhat)]; fills["out"] = [(fs.PURPLE, 0.95)]
    if phase >= 2:
        ec2, ew2, ea2 = style(dW2, 0); lab2 = dW2
        step, eq = "Step 2 — weight gradients of the last layer:  ∂L/∂W₂ = (∂L/∂ŷ)·a₁", "∂L/∂W₂ = " + f(dyhat) + " · [0, 3.2, 1.1] = [" + ", ".join(f(v) for v in dW2[0]) + "]"
    if phase >= 3:
        step, eq = "Step 3 — pass the gradient down:  ∂L/∂a₁ = (∂L/∂ŷ)·W₂, then through ReLU (blocked where z₁ ≤ 0)", "∂L/∂a₁ = [" + ", ".join(f(v) for v in da1) + "]   →   ∂L/∂z₁ = [" + ", ".join(f(v) for v in dz1) + "]"
        k = e if phase == 3 else 1.0
        vals["h"] = ["δ=" + f(dz1[j] if (k > 0.5) else da1[j]) for j in range(3)]
        fills["h"] = [(fs.RED, 0.45) if z1[0][j] <= 0 and k > 0.5 else (fs.PURPLE, 0.95) for j in range(3)]
    if phase >= 4:
        ec1, ew1, ea1 = style(dW1, 0); lab1 = dW1
        step, eq = "Step 4 — weight gradients of the first layer:  ∂L/∂W₁ = (∂L/∂z₁)·xᵀ", "∂L/∂W₁ = [" + "; ".join("[" + ", ".join(f(v) for v in row) + "]" for row in dW1) + "]"
    if phase >= 5:
        step, eq = "Every weight now has a gradient — the optimizer will use them:  w ← w − η·∂L/∂w", "unit 1 got zero gradient: its ReLU was off (dead for this input)"
    draw_edges(ax, "in", "h", lab1, ec1, ew1, ea1); draw_edges(ax, "h", "out", lab2, ec2, ew2, ea2)
    if phase in (2, 4):                                                                  # gradient dots travelling backward
        if phase == 2:
            for i in range(3): pa, pb = POS["out"][0], POS["h"][i]; ax.scatter([pa[0] + e * (pb[0] - pa[0])], [pa[1] + e * (pb[1] - pa[1])], s=60, color=fs.YELLOW, zorder=6)
        else:
            for j in range(3):
                for i in range(2): pa, pb = POS["h"][j], POS["in"][i]; ax.scatter([pa[0] + e * (pb[0] - pa[0])], [pa[1] + e * (pb[1] - pa[1])], s=60, color=fs.YELLOW, zorder=6)
    draw_nodes(ax, vals, fills)
    banner(fig, title, step, eq, "edge labels now show gradients: blue = increase the loss if the weight grows, red = decrease; thicker = larger |gradient|")
    img = fs.fig_to_image(fig); plt.close(fig); return img

for ph in range(6):
    for t in range(PH if ph in (2, 3, 4) else 5): frames.append(render(ph, t / max(PH - 1, 1)))
size = fs.save_gif(frames, OUT / "fig_3_20_backprop.gif", fps=9, colors=64, hold_last=12); print("frames", len(frames), "bytes", size, "loss", loss)
