"""Tier 1 tensor figures: 3-D tensor (1.4), axes and reductions (1.6), broadcasting animation (1.7), matrix multiplication animation (1.11)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, fs
from linalg_basics import broadcast_shape, matmul_manual

# ---- 1.4 a 3-D tensor (batch, sequence, features) as a block of voxels
B, T, D = 4, 5, 6; filled = np.ones((B, T, D), bool); colors = np.empty((B, T, D), dtype=object)
base = [fs.BLUE, fs.TEAL, fs.PURPLE, fs.ORANGE]
for b in range(B): colors[b] = base[b] + "30"
colors[1, 2, :] = fs.YELLOW + "ff"                                                       # x[1, 2, :]  -> one token's feature vector
fig = plt.figure(figsize=(11, 6.6)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
ax.voxels(filled, facecolors=colors, edgecolor=(1, 1, 1, 0.18), linewidth=0.5); ax.set_box_aspect((B, T, D)); ax.view_init(elev=22, azim=-58)
ax.set_xlabel("batch (4)", labelpad=8); ax.set_ylabel("sequence (5)", labelpad=8); ax.set_zlabel("features (6)", labelpad=6); ax.set_position([0.08, 0.2, 0.84, 0.7])
for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.set_pane_color((0.06, 0.07, 0.12, 1.0)); pane._axinfo["grid"]["color"] = fs.GRID
ax.set_xticks(np.arange(B) + 0.5, [str(i) for i in range(B)]); ax.set_yticks(np.arange(T) + 0.5, [str(i) for i in range(T)]); ax.set_zticks(np.arange(D) + 0.5, [str(i) for i in range(D)])
fig.suptitle("A 3-D tensor of shape (batch, sequence, features) = (4, 5, 6): 120 numbers", fontsize=14, fontweight="bold", y=0.96)
fig.text(0.5, 0.075, "axis 0 = batch (one colour per sentence), axis 1 = sequence (token position), axis 2 = features\nthe yellow column is x[1, 2, :] — the 6-number vector of token 2 in sentence 1", ha="center", color=fs.YELLOW, fontsize=10.5)
fig.text(0.5, 0.02, f"x.shape = {filled.shape},  x.ndim = 3,  x.size = {filled.size};  in an LLM the features axis is d_model (e.g. 768)", ha="center", color=fs.MUTED, fontsize=9.5)
fig.savefig(OUT / "fig_1_04_tensor_3d.png", dpi=150); plt.close(fig)

def draw_mat(ax, M, x0, y0, cell=0.9, fmt="{:g}", fc=None, text_c=None, alpha=None, title=None, tsize=13, fsz=12):
    r, c = M.shape
    for i in range(r):
        for j in range(c):
            col = (fc[i][j] if fc is not None else "#2b3050"); a = 1.0 if alpha is None else alpha[i][j]
            ax.add_patch(Rectangle((x0 + j * cell, y0 - i * cell), cell * 0.94, cell * 0.94, fc=col, ec="none", alpha=a))
            if M[i, j] is not None and not (isinstance(M[i, j], float) and np.isnan(M[i, j])): ax.text(x0 + j * cell + cell * 0.47, y0 - i * cell + cell * 0.47, fmt.format(M[i, j]), ha="center", va="center", color=(text_c or fs.FG), fontsize=fsz, fontweight="bold")
    if title: ax.text(x0 + c * cell / 2, y0 + cell * 1.2, title, ha="center", color=fs.MUTED, fontsize=tsize)

# ---- 1.6 axes and reductions
M = np.arange(1, 13).reshape(3, 4); fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
for ax in axs: ax.set_xlim(-1.2, 6.2); ax.set_ylim(-3.6, 3.4); ax.set_aspect("equal"); ax.axis("off")
draw_mat(axs[0], M, 0.2, 1.8, title="x with shape (3, 4)"); axs[0].set_title("the matrix  (axis 0 = rows ↓, axis 1 = columns →)", fontsize=11)
axs[0].annotate("", xy=(-0.5, -0.9), xytext=(-0.5, 2.6), arrowprops=dict(arrowstyle="-|>", color=fs.TEAL, lw=2.4)); axs[0].text(-1.15, 0.85, "axis 0", color=fs.TEAL, rotation=90, fontsize=11, va="center")
axs[0].annotate("", xy=(4.2, -1.3), xytext=(0.2, -1.3), arrowprops=dict(arrowstyle="-|>", color=fs.ORANGE, lw=2.4)); axs[0].text(2.2, -1.85, "axis 1", color=fs.ORANGE, fontsize=11, ha="center")
s0, s1 = M.sum(axis=0), M.sum(axis=1)
fc = [[fs.TEAL + "88"] * 4 for _ in range(3)]; draw_mat(axs[1], M, 0.2, 1.8, fc=fc, title="x.sum(axis=0)  → shape (4,)"); draw_mat(axs[1], s0[None, :], 0.2, -1.7, fc=[[fs.TEAL] * 4], text_c=fs.BG); axs[1].set_title("collapse the ROWS: one total per column", fontsize=11)
fc = [[fs.ORANGE + "88"] * 4 for _ in range(3)]; draw_mat(axs[2], M, 0.2, 1.8, fc=fc, title="x.sum(axis=1)  → shape (3,)"); draw_mat(axs[2], s1[:, None], 4.4, 1.8, fc=[[fs.ORANGE]] * 3, text_c=fs.BG); axs[2].set_title("collapse the COLUMNS: one total per row", fontsize=11)
fig.suptitle("Axes: the reduction removes the axis you name (softmax, LayerNorm and pooling all choose an axis)", fontsize=13, fontweight="bold", y=1.03)
assert s0.tolist() == [15, 18, 21, 24] and s1.tolist() == [10, 26, 42]; fig.savefig(OUT / "fig_1_06_axes_reduction.svg", bbox_inches="tight"); plt.close(fig)

# ---- 1.7 broadcasting animation: (4,1) + (1,5) -> (4,5)
A = np.array([[10], [20], [30], [40]]); Bv = np.array([[1, 2, 3, 4, 5]]); R = A + Bv; assert R.shape == (4, 5) == broadcast_shape((4, 1), (1, 5))
frames = []; cs = 0.85
def frame(phase, p):
    fig, ax = plt.subplots(figsize=(10.4, 5.8)); ax.set_xlim(-1.4, 11.2); ax.set_ylim(-4.2, 5.4); ax.set_aspect("equal"); ax.axis("off"); e = fs.ease(p)
    # A (column) at left, B (row) on top, result region at right
    ncols = 5 if phase >= 1 else 1; k = 1 + int(round(e * 4)) if phase == 1 else (5 if phase > 1 else 1)
    fcA = [[fs.BLUE if (j == 0) else fs.BLUE + "55" for j in range(5)] for _ in range(4)]
    for i in range(4):
        for j in range(5):
            if j < (k if phase >= 1 else 1):
                ax.add_patch(Rectangle((j * cs, 3.4 - i * cs), cs * 0.92, cs * 0.92, fc=fs.BLUE if j == 0 else fs.BLUE + "66", ec="none")); ax.text(j * cs + cs * 0.46, 3.4 - i * cs + cs * 0.46, str(A[i, 0]), ha="center", va="center", color=fs.BG if j == 0 else fs.FG, fontsize=11, fontweight="bold")
    ax.text(5 * cs / 2, 4.55, "A  (4, 1)" + (" → (4, 5)" if phase >= 1 else ""), ha="center", color=fs.BLUE, fontsize=12)
    kb = 1 + int(round(e * 3)) if phase == 2 else (4 if phase > 2 else 1)
    for j in range(5):
        for i in range(4):
            if i < (kb if phase >= 2 else 1):
                ax.add_patch(Rectangle((6.2 + j * cs, 3.4 - i * cs), cs * 0.92, cs * 0.92, fc=fs.TEAL if i == 0 else fs.TEAL + "66", ec="none")); ax.text(6.2 + j * cs + cs * 0.46, 3.4 - i * cs + cs * 0.46, str(Bv[0, j]), ha="center", va="center", color=fs.BG if i == 0 else fs.FG, fontsize=11, fontweight="bold")
    ax.text(6.2 + 5 * cs / 2, 4.55, "B  (1, 5)" + (" → (4, 5)" if phase >= 2 else ""), ha="center", color=fs.TEAL, fontsize=12)
    ax.text(5.1, 2.75, "+", color=fs.FG, fontsize=22, ha="center")
    if phase >= 3:
        shown = R if phase >= 4 else np.where(np.arange(20).reshape(4, 5) < int(e * 20) + 1, R, 0)
        for i in range(4):
            for j in range(5):
                if phase >= 4 or (i * 5 + j) <= int(e * 20):
                    ax.add_patch(Rectangle((2.9 + j * cs, -1.3 - i * cs), cs * 0.92, cs * 0.92, fc=fs.YELLOW, ec="none")); ax.text(2.9 + j * cs + cs * 0.46, -1.3 - i * cs + cs * 0.46, str(R[i, j]), ha="center", va="center", color=fs.BG, fontsize=11, fontweight="bold")
        ax.text(2.9 + 5 * cs / 2, -0.55 + 0.05, "A + B  (4, 5)", ha="center", color=fs.YELLOW, fontsize=12)
    msg = {0: "Two arrays of different shapes: (4, 1) and (1, 5)", 1: "Rule: a size-1 axis is stretched (virtually, no copy) to match the other array — A is repeated across the columns",
           2: "…and B is repeated down the rows. Shapes compare from the right: (4, 1) vs (1, 5) → (4, 5)", 3: "Add element by element on the stretched arrays", 4: "Result: every A[i] + B[j] combination — a (4, 5) array"}[phase]
    fig.text(0.5, 0.94, "Broadcasting", ha="center", color=fs.FG, fontsize=16, fontweight="bold"); fig.text(0.5, 0.075, msg, ha="center", color=fs.YELLOW, fontsize=11.5)
    img = fs.fig_to_image(fig); plt.close(fig); return img
for ph, n in ((0, 6), (1, 8), (2, 8), (3, 10), (4, 6)):
    for t in range(n): frames.append(frame(ph, t / max(n - 1, 1)))
print("broadcast bytes", fs.save_gif(frames, OUT / "fig_1_07_broadcasting.gif", fps=9, colors=48, hold_last=10))

# ---- 1.11 matrix multiplication animation: C[i, j] = row i of A . column j of B
A = np.array([[1, 2, 0, 1], [0, 3, 1, 2], [2, 1, 1, 0]]); Bm = np.array([[1, 0], [2, 1], [0, 3], [1, 1]]); C = np.array(matmul_manual(A.tolist(), Bm.tolist())); assert np.array_equal(C, A @ Bm) and C.shape == (3, 2)
frames = []; order = [(i, j) for j in range(2) for i in range(3)]
def mframe(step, p, final=False):
    fig, ax = plt.subplots(figsize=(10.4, 5.6)); ax.set_xlim(-0.8, 12.2); ax.set_ylim(-4.2, 4.6); ax.set_aspect("equal"); ax.axis("off")
    ax_, ay_, bx_, by_, cx_, cy_ = 0.0, 2.6, 5.6, 2.6, 5.6, -1.0
    done = order[:step] + ([order[step]] if (not final and p > 0.7 and step < len(order)) else [])
    i, j = order[min(step, 5)]
    fa = [[(fs.BLUE if (r == i and not final) else "#2b3050") for _ in range(4)] for r in range(3)]; fb = [[(fs.TEAL if (c == j and not final) else "#2b3050") for c in range(2)] for _ in range(4)]
    fc = [[("#1f2440") for _ in range(2)] for _ in range(3)]
    Cm = np.full((3, 2), np.nan)
    for (a, b) in done: fc[a][b] = fs.YELLOW; Cm[a, b] = C[a, b]
    if not final: fc[i][j] = fs.ORANGE if (i, j) not in done else fs.YELLOW
    draw_mat(ax, A, ax_, ay_, cell=0.95, fc=fa, title="A  (3×4)"); draw_mat(ax, Bm, bx_, by_ + 0.95, cell=0.95, fc=fb, title="B  (4×2)")
    draw_mat(ax, Cm, cx_ + 0.0, cy_, cell=0.95, fc=fc, text_c=fs.BG, title="C = A @ B  (3×2)")
    ax.text(4.6, 1.45, "@", color=fs.FG, fontsize=22, ha="center"); ax.text(2.2, -3.6, "", color=fs.FG)
    if not final:
        terms = " + ".join(f"{A[i, k]}·{Bm[k, j]}" for k in range(4)); val = int(C[i, j])
        eq = f"C[{i},{j}] = {terms} = {val}" if p > 0.45 else f"C[{i},{j}] = (row {i} of A) · (column {j} of B)"
        fig.text(0.5, 0.075, eq, ha="center", color=fs.YELLOW, fontsize=13)
    else: fig.text(0.5, 0.075, "Every entry is one dot product; inner dimensions (4 and 4) must match, the result is (3, 2)", ha="center", color=fs.YELLOW, fontsize=12.5)
    fig.text(0.5, 0.94, "Matrix multiplication = many dot products", ha="center", color=fs.FG, fontsize=16, fontweight="bold")
    img = fs.fig_to_image(fig); plt.close(fig); return img
for s in range(6):
    for t in range(4): frames.append(mframe(s, t / 3))
for t in range(8): frames.append(mframe(6, 1.0, final=True))
print("matmul bytes", fs.save_gif(frames, OUT / "fig_1_11_matrix_multiplication.gif", fps=6, colors=48, hold_last=8))
