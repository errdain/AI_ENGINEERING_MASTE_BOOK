"""Figure 3.18-1 (GIF): the forward pass of the 2-3-1 ReLU network from the concept file, with values computed by nn.py."""
import numpy as np
from _common import OUT, figstyle as fs
from _net_draw import *

m = build_net(); x = X.copy(); z1 = x @ W1.T + B1; a1 = np.maximum(0, z1); z2 = a1 @ W2.T + B2
assert np.allclose(m.forward(x), z2)                                             # the picture uses the library's numbers
f = lambda v: "0" if abs(v) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")
frames, PH = [], 9

def render(phase, p):
    fig, ax = new_axes(); e = fs.ease(p)
    ew = [[1.2] * 2 for _ in range(3)]; ec = [[fs.MUTED] * 2 for _ in range(3)]; ea = [[0.35] * 2 for _ in range(3)]
    ew2 = [[1.2] * 3]; ec2 = [[fs.MUTED] * 3]; ea2 = [[0.35] * 3]
    vals = {"in": [f(x[0, 0]), f(x[0, 1])], "h": [None] * 3, "out": [None]}
    fills = {"in": [(fs.TEAL, 1.0)] * 2, "h": [("#2a2f4e", 1.0)] * 3, "out": [("#2a2f4e", 1.0)]}
    title = "Forward propagation: x → z₁ → a₁ → z₂"
    if phase == 0: step, eq = "Step 0 — the input x = [1, 2] enters the network", "x = [1, 2]"
    if phase >= 1:
        for j in range(3):
            for i in range(2): ec[j][i] = fs.BLUE; ew[j][i] = 1.5 + 2.0 * abs(W1[j, i]); ea[j][i] = 0.35 + 0.55 * (e if phase == 1 else 1)
        step, eq = "Step 1 — weighted sums  z₁ = W₁x + b₁", "z₁ = W₁x + b₁ = [" + ", ".join(f(v) for v in z1[0]) + "]"
        if phase == 1:
            for j in range(3):
                for i in range(2):                                                          # a signal dot travelling along each edge
                    pa, pb = POS["in"][i], POS["h"][j]; ax.scatter([pa[0] + e * (pb[0] - pa[0])], [pa[1] + e * (pb[1] - pa[1])], s=60, color=fs.YELLOW, zorder=6)
        vals["h"] = [f(v) for v in z1[0]] if (phase > 1 or p > 0.8) else [None] * 3
        fills["h"] = [(fs.BLUE, 0.9)] * 3 if vals["h"][0] is not None else fills["h"]
    if phase >= 2:
        step, eq = "Step 2 — ReLU keeps positives, zeroes negatives:  a₁ = max(0, z₁)", "a₁ = [" + ", ".join(f(v) for v in a1[0]) + "]   (unit 1 is switched off)"
        k = e if phase == 2 else 1.0
        vals["h"] = [f(a1[0][j] if k > 0.5 else z1[0][j]) for j in range(3)]
        fills["h"] = [(fs.RED if a1[0][j] == 0 and k > 0.5 else fs.TEAL, 0.95 if a1[0][j] > 0 or k < 0.5 else 0.45) for j in range(3)]
    if phase >= 3:
        for i in range(3): ec2[0][i] = fs.ORANGE; ew2[0][i] = 1.5 + 2.0 * abs(W2[0, i]); ea2[0][i] = 0.35 + 0.55 * (e if phase == 3 else 1)
        step, eq = "Step 3 — output  z₂ = W₂a₁ + b₂", "z₂ = 1·0 + (−1)·3.2 + 0.5·1.1 + 0.1 = " + f(z2[0, 0])
        if phase == 3:
            for i in range(3):
                pa, pb = POS["h"][i], POS["out"][0]; ax.scatter([pa[0] + e * (pb[0] - pa[0])], [pa[1] + e * (pb[1] - pa[1])], s=60, color=fs.YELLOW, zorder=6)
        vals["out"] = [f(z2[0, 0])] if (phase > 3 or p > 0.8) else [None]; fills["out"] = [(fs.ORANGE, 0.95)] if vals["out"][0] else fills["out"]
    if phase >= 4: step, eq = "Done — the prediction is ŷ = −2.55 (the library output matches)", "ŷ = " + f(z2[0, 0])
    draw_edges(ax, "in", "h", W1, ec, ew, ea); draw_edges(ax, "h", "out", W2, ec2, ew2, ea2)
    draw_nodes(ax, vals, fills)
    banner(fig, title, step, eq, "edge labels = weights; thicker edge = larger |weight|; node text = value computed at that step")
    img = fs.fig_to_image(fig); plt.close(fig); return img

for ph in range(5):
    for t in range(PH if ph < 4 else 6): frames.append(render(ph, t / max(PH - 1, 1)))
size = fs.save_gif(frames, OUT / "fig_3_18_forward_pass.gif", fps=9, colors=64, hold_last=10); print("frames", len(frames), "bytes", size)
