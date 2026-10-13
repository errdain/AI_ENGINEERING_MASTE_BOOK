"""Figure 2.2-1 (SVG): supervised learning in three steps with the fruit example used on the concept page."""
import numpy as np
import matplotlib.pyplot as plt
from _common import OUT, fs

apples = np.array([1.0, 2.0, 3.0, 4.0, 4.8]); oranges = np.array([5.2, 6.0, 7.0, 8.0, 9.0]); x = np.r_[apples, oranges]; y = np.r_[np.zeros(5), np.ones(5)].astype(int)
ts = np.arange(0.0, 10.01, 0.1); mistakes = np.array([int(((x > t).astype(int) != y).sum()) for t in ts]); best = ts[mistakes == mistakes.min()]; t_best = float((apples.max() + oranges.min()) / 2)
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9))
for ax in axs: ax.grid(True, alpha=0.3)
a = axs[0]; a.scatter(apples, np.zeros(5), s=170, color=fs.TEAL, edgecolor=fs.BG, label="apple (label 0)"); a.scatter(oranges, np.zeros(5), s=170, color=fs.ORANGE, edgecolor=fs.BG, label="orange (label 1)"); a.set_ylim(-1, 1); a.set_yticks([]); a.set_xlim(0, 10); a.set_xlabel("feature: fruit size (0–10)"); a.legend(loc="upper center", ncol=2, fontsize=9.5)
a.set_title("① Labeled examples: each fruit comes\nwith its correct answer", fontsize=11.5)
b = axs[1]; b.plot(ts, mistakes, color=fs.BLUE, lw=3); b.scatter([t_best], [mistakes.min()], s=150, color=fs.YELLOW, edgecolor="white", zorder=5); b.set_ylim(-0.3, 7.2); b.set_xlabel("threshold: call it an orange if size > threshold"); b.set_ylabel("mistakes on the 10 training fruits"); b.annotate(f"training = pick the threshold\nwith the fewest mistakes\n(here {t_best:.1f}: {mistakes.min()} mistakes)", xy=(t_best, mistakes.min()), xytext=(5.4, 5.6), color=fs.YELLOW, arrowprops=dict(arrowstyle="->", color=fs.YELLOW), fontsize=10)
b.set_title("② Learn: try settings, keep the one\nthat makes the fewest mistakes", fontsize=11.5)
c = axs[2]; c.scatter(apples, np.zeros(5), s=120, color=fs.TEAL, alpha=0.5); c.scatter(oranges, np.zeros(5), s=120, color=fs.ORANGE, alpha=0.5); c.axvline(t_best, color=fs.YELLOW, lw=2.6); c.axvspan(0, t_best, color=fs.TEAL, alpha=0.1); c.axvspan(t_best, 10, color=fs.ORANGE, alpha=0.1)
for nv in (2.4, 7.2): c.scatter([nv], [0], marker="*", s=420, color=fs.FG, edgecolor=fs.BG, zorder=6); c.text(nv, 0.35, f"new fruit\n{nv} → {'orange' if nv > t_best else 'apple'}", ha="center", color=fs.FG, fontsize=10.5)
c.set_ylim(-1, 1); c.set_yticks([]); c.set_xlim(0, 10); c.set_xlabel("feature: fruit size (0–10)"); c.set_title("③ Predict: apply the learned rule\nto fruits it has never seen", fontsize=11.5)
fig.suptitle("Supervised learning: learn a rule from labeled examples, then use it on new ones", fontsize=14, fontweight="bold", y=1.04); fig.savefig(OUT / "fig_2_02_supervised_learning.svg", bbox_inches="tight"); plt.close(fig); print("best threshold", t_best, "mistakes", mistakes.min())
