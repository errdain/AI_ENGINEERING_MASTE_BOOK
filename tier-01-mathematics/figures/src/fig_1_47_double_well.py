"""Figure 1.47-1 (GIF, 3D): gradient descent on a double-well surface f(x, y) = x^4 - 3x^2 + x + y^2 from two starting points."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, figstyle as fs
from info_numerics import double_well, double_well_grad, descend_1d

f = lambda x, y: double_well(x) + y**2
left, right = descend_1d(double_well_grad, -2.0), descend_1d(double_well_grad, 2.0)                  # the minima reported in the concept file (-1.30, 1.13)
def run(x0, y0, lr=0.02, steps=400):
    p = np.array([x0, y0]); out = [p.copy()]
    for _ in range(steps): p = p - lr * np.array([double_well_grad(p[0]), 2 * p[1]]); out.append(p.copy())
    return np.array(out)
A, B = run(-2.0, 1.4), run(2.0, 1.4); assert abs(A[-1, 0] - left) < 1e-2 and abs(B[-1, 0] - right) < 1e-2
sub = lambda P: P[::5]; A, B = sub(A), sub(B)
g = np.linspace(-2.3, 2.3, 60); gy = np.linspace(-1.6, 1.6, 44); X, Y = np.meshgrid(g, gy); Z = np.minimum(f(X, Y), 9.0)
cmap = LinearSegmentedColormap.from_list("dw", [fs.BLUE, "#34406e", fs.ORANGE])
frames = []; NF = 40; LIFT = 0.5
for k in range(NF):
    fig = plt.figure(figsize=(8.4, 5.4)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.55, linewidth=0, antialiased=False, rstride=1, cstride=1, vmin=-3.6, vmax=9)
    n = int(len(A) * fs.ease(min(1.0, k / (NF * 0.62)))) + 1
    for P, c in ((A, fs.TEAL), (B, fs.PINK)):
        ax.plot(P[:n, 0], P[:n, 1], np.minimum(f(P[:n, 0], P[:n, 1]), 9) + LIFT, color=c, lw=3.6, zorder=9); ax.scatter(P[:1, 0], P[:1, 1], f(P[:1, 0], P[:1, 1]) + LIFT, s=55, color=c, depthshade=False, zorder=10)
        ax.scatter(P[n - 1:n, 0], P[n - 1:n, 1], f(P[n - 1:n, 0], P[n - 1:n, 1]) + LIFT, s=110, color=c, edgecolor="white", depthshade=False, zorder=11)
    ax.scatter([left], [0], [double_well(left)], marker="*", s=260, color=fs.TEAL, edgecolor="white", depthshade=False, zorder=12); ax.scatter([right], [0], [double_well(right)], marker="*", s=260, color=fs.PINK, edgecolor="white", depthshade=False, zorder=12)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.set_pane_color((0.06, 0.07, 0.12, 1.0)); pane._axinfo["grid"]["color"] = fs.GRID
    ax.set_zlim(-4, 9.5); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("loss f(x, y)"); ax.view_init(elev=30, azim=-68 + 36 * k / NF); ax.set_position([0.02, 0.1, 0.96, 0.76])
    fig.text(0.5, 0.945, "Gradient descent finds a minimum — which one depends on where you start", ha="center", color=fs.FG, fontsize=11.5, fontweight="bold")
    fig.text(0.49, 0.06, f"★ teal: start x = −2 → global minimum, f = {double_well(left):.2f}", ha="right", color=fs.TEAL, fontsize=10.5)
    fig.text(0.51, 0.06, f"★ pink: start x = +2 → local minimum, f = {double_well(right):.2f}", ha="left", color=fs.PINK, fontsize=10.5)
    fig.text(0.5, 0.015, "both runs end with gradient ≈ 0 and look converged; only one is the global minimum", ha="center", color=fs.MUTED, fontsize=9.5)
    frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("bytes", fs.save_gif(frames, OUT / "fig_1_47_double_well.gif", fps=12, colors=48, hold_last=10), "minima", left, right)
