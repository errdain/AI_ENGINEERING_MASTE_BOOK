"""Figure 1.46-1 (PNG, 3D): the saddle f(x, y) = x^2 - y^2. Zero gradient at the origin, but it is a minimum along x and a maximum along y."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, figstyle as fs
from info_numerics import saddle

assert saddle(np.array([0.0, 0.1])) < saddle(np.zeros(2)) < saddle(np.array([0.1, 0.0]))        # the same facts the tests check
g = np.linspace(-1.6, 1.6, 90); X, Y = np.meshgrid(g, g); Z = X**2 - Y**2
cmap = LinearSegmentedColormap.from_list("sd", [fs.RED, "#2a2f4e", fs.BLUE])
fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.82, linewidth=0, antialiased=True, rstride=1, cstride=1)
t = np.linspace(-1.6, 1.6, 200)
ax.plot(t, np.zeros_like(t), t**2, color=fs.BLUE, lw=3.2, label="along x: curves UP (a minimum)")
ax.plot(np.zeros_like(t), t, -t**2, color=fs.RED, lw=3.2, label="along y: curves DOWN (a maximum)")
ax.scatter([0], [0], [0], s=120, color=fs.YELLOW, zorder=10, depthshade=False); ax.text(0.1, 0.1, 0.35, "saddle point\n∇f = 0", color=fs.YELLOW, fontsize=11)
p = np.array([0.9, 1e-3]); path = [p.copy()]                                                      # gradient descent starting almost on the saddle's stable axis
while abs(p[1]) < 1.5 and len(path) < 200: p = p - 0.12 * np.array([2 * p[0], -2 * p[1]]); path.append(p.copy())
path = np.array(path); ax.plot(path[:, 0], path[:, 1], path[:, 0]**2 - path[:, 1]**2 + 0.03, color=fs.TEAL, lw=2.4, marker="o", ms=3.5, label="gradient descent: stalls near the saddle, then escapes along y")
for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.set_pane_color((0.06, 0.07, 0.12, 1.0)); pane._axinfo["grid"]["color"] = fs.GRID
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x, y) = x² − y²"); ax.set_zlim(-2.6, 2.7); ax.view_init(elev=24, azim=-58)
ax.set_title("A saddle point: the gradient is zero, but it is not a minimum", fontsize=14, fontweight="bold", pad=4)
ax.legend(loc="upper left", fontsize=9.5, bbox_to_anchor=(0.0, 0.93))
fig.savefig(OUT / "fig_1_46_saddle.png", dpi=150); print("ok")
