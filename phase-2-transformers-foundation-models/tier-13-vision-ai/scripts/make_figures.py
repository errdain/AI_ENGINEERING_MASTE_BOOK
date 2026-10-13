"""Generate Tier 13 figures + GIFs.  Run:  python scripts/make_figures.py"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
from vision import *
from models import *
import torch

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def sample_image(shape=0, color=0, size=16, seed=5):
    rng = np.random.default_rng(seed)
    m = draw_shape(shape, size, rng)
    return (RGB[color][:, None, None] * m[None]).astype(np.float32)


def gray_circle(size=16):
    return draw_shape(0, size, np.random.default_rng(2))


def box(ax, xy, w, h, text, fc, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec="white"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, weight="bold")


def arrow(ax, a, b, color="#556"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4))


# 01 ------------------------------------------------------------------
def fig_tensor():
    img = sample_image(1, 2)
    img = np.clip(img + np.random.default_rng(0).normal(0, .08, img.shape), 0, 1).astype(np.float32)
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1, 1, 1.5]})
    ax[0].imshow(img.transpose(1, 2, 0), interpolation="nearest"); ax[0].set_title("RGB image\nshape (3,16,16)")
    for c, (n, cm) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
        ax[c + 1].imshow(img[c], cmap=cm, vmin=0, vmax=1, interpolation="nearest"); ax[c + 1].set_title(f"channel {n} (H x W)")
    crop = img[2, 5:11, 5:11]
    ax[4].imshow(crop, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for i in range(6):
        for j in range(6):
            ax[4].text(j, i, f"{crop[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if crop[i, j] < .6 else "w")
    ax[4].set_title("6x6 crop of channel B:\nnumbers are the pixels")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("An image is a tensor: channels x height x width of numbers (float in [0,1] after /255)", y=1.04)
    fig.savefig(OUT / "01_image_as_tensor.png"); plt.close(fig)


# 02 GIF conv slide ---------------------------------------------------
def gif_conv():
    img = gray_circle(12); k = KERNELS["sobel-x"]; s = 1
    out = conv2d_single(img, k, pad=0); ho = out.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    ax[1].imshow(k, cmap="coolwarm", vmin=-2, vmax=2)
    for (i, j), v in np.ndenumerate(k): ax[1].text(j, i, int(v), ha="center", va="center", fontsize=12)
    ax[1].set_title("kernel (Sobel-x)")
    n = ho * ho

    def upd(f):
        f = min(f, n - 1); i, j = divmod(f, ho)
        ax[0].clear(); ax[0].imshow(img, cmap="gray", interpolation="nearest")
        ax[0].add_patch(Rectangle((j - .5, i - .5), 3, 3, fill=False, ec="#E45756", lw=2.5))
        ax[0].set_title(f"input 12x12, window at (row {i}, col {j})")
        m = np.full_like(out, np.nan); flat = out.ravel().copy(); m.ravel()[:f + 1] = flat[:f + 1]
        ax[2].clear(); ax[2].imshow(m, cmap="coolwarm", vmin=-4, vmax=4, interpolation="nearest")
        ax[2].add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=2))
        ax[2].set_title(f"output {ho}x{ho} = (12-3)/1+1;  value={out[i, j]:.1f}")
        for a in (ax[0], ax[2]): a.set_xticks([]); a.set_yticks([])
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle("Convolution: slide the kernel, multiply-and-sum, write one output pixel (vertical edges light up)")
    step = 3
    anim = FuncAnimation(fig, upd, frames=list(range(0, n, step)) + [n - 1] * 4, interval=150)
    anim.save(OUT / "02_convolution_sliding.gif", writer=PillowWriter(fps=7), dpi=60); plt.close(fig)


# 03 kernels ----------------------------------------------------------
def fig_kernels():
    img = gray_circle(24)
    img = np.clip(img + np.random.default_rng(0).normal(0, .1, img.shape), 0, 1)
    names = ["identity", "box blur", "sharpen", "sobel-x", "sobel-y"]
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
    for a, n in zip(ax, names):
        o = conv2d_single(img, KERNELS[n], pad=1)
        a.imshow(o, cmap="gray" if n in ("identity", "box blur", "sharpen") else "coolwarm", interpolation="nearest")
        a.set_title(n); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Same noisy circle, five 3x3 kernels (padding=1 keeps 24x24): blur smooths noise, Sobel finds edges, sharpen amplifies noise", y=1.04)
    fig.savefig(OUT / "03_kernel_gallery.png"); plt.close(fig)


# 04 receptive field --------------------------------------------------
def fig_rf():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    L = 8
    for lab, layers, c in [("3x3 convs, stride 1", [(3, 1)] * L, C[0]),
                           ("3x3 conv + 2x2 pool, repeated", [(3, 1), (2, 2)] * (L // 2), C[1]),
                           ("3x3 convs, stride 2", [(3, 2)] * L, C[3])]:
        rf = [r for r, _ in receptive_field(layers)]
        ax[0].plot(range(1, len(rf) + 1), rf, "o-", label=lab, color=c)
    ax[0].set_yscale("log"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("receptive field (input pixels, log)")
    ax[0].set_title("Receptive field growth"); ax[0].legend(fontsize=8)
    img = gray_circle(16); mp = max_pool2d(img)
    ax[1].axis("off")
    a1 = ax[1].inset_axes([0, 0.05, 0.42, 0.85]); a2 = ax[1].inset_axes([0.55, 0.2, 0.42, 0.55])
    a1.imshow(img, cmap="gray", interpolation="nearest"); a1.set_title("16x16"); a2.imshow(mp, cmap="gray", interpolation="nearest"); a2.set_title("max-pool 2x2 -> 8x8")
    for a in (a1, a2): a.set_xticks([]); a.set_yticks([])
    ax[1].set_title("Pooling halves resolution, keeps the strongest response")
    fig.savefig(OUT / "04_receptive_field_pooling.png"); plt.close(fig)


# 05 resnet + gradient flow ------------------------------------------
def grad_profile(residual, depth=40, width=32, seed=0):
    torch.manual_seed(seed)
    layers = [torch.nn.Linear(width, width) for _ in range(depth)]
    x = torch.randn(64, width, requires_grad=False); h = x.clone().requires_grad_(True); hs = [h]
    for l in layers:
        h = (h + torch.tanh(l(h))) if residual else torch.tanh(l(h)); h.retain_grad(); hs.append(h)
    h.pow(2).mean().backward()
    return [hh.grad.norm().item() for hh in hs[1:]]


def fig_resnet():
    fig = plt.figure(figsize=(13, 4.2)); a = fig.add_axes([0, 0, .38, 1]); a.axis("off"); a.set_xlim(0, 6); a.set_ylim(0, 8)
    box(a, (1.6, 6.8), 1.6, .7, "x", C[0]); box(a, (1.6, 5.2), 1.6, .8, "conv3x3 + ReLU", C[1]); box(a, (1.6, 3.6), 1.6, .8, "conv3x3", C[1])
    box(a, (1.6, 1.6), 1.6, .8, "+  then ReLU", C[2]); box(a, (1.6, .2), 1.6, .7, "y = relu(x + F(x))", C[0], fs=8)
    for y0, y1 in [(6.8, 6.0), (5.2, 4.4), (3.6, 2.4), (1.6, .9)]: arrow(a, (2.4, y0), (2.4, y1))
    a.add_patch(FancyArrowPatch((1.5, 7.1), (1.5, 2.0), connectionstyle="arc3,rad=.45", arrowstyle="-|>", mutation_scale=12, color=C[3], lw=2.2))
    a.text(.05, 4.6, "identity\nshortcut", color=C[3], fontsize=9, weight="bold"); a.text(3.5, 4.6, "F(x): learned\nresidual", color=C[1], fontsize=9, weight="bold")
    a.set_title("Residual block (Tier-13.12)")
    ax = fig.add_axes([.46, .15, .52, .72])
    gp, gr = grad_profile(False), grad_profile(True)
    ax.semilogy(range(1, 41), gp, color=C[3], label="plain 40-layer tanh net"); ax.semilogy(range(1, 41), gr, color=C[2], label="same net + identity shortcuts")
    ax.set_xlabel("layer index (1 = near input, 40 = near loss)"); ax.set_ylabel("|dLoss/d activation| (log)")
    ax.set_title("Measured gradient flow: shortcuts keep early-layer gradients alive"); ax.legend()
    fig.savefig(OUT / "05_resnet_gradient_flow.png"); plt.close(fig)


# 06 ViT patchify ----------------------------------------------------
def fig_patchify():
    img = sample_image(2, 1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 1.7]})
    ax[0].imshow(img.transpose(1, 2, 0), interpolation="nearest")
    for t in range(0, 17, 4): ax[0].axhline(t - .5, color="w", lw=1.5); ax[0].axvline(t - .5, color="w", lw=1.5)
    ax[0].set_title("16x16 image, patch=4 -> 16 patches")
    pt = patchify(img, 4)
    ax[1].axis("off"); ax[1].set_title("patch tokens, raster order")
    for k in range(16):
        s = ax[1].inset_axes([(k % 4) * .25, 1 - (k // 4 + 1) * .25, .22, .22]); s.imshow(unpatchify(pt[k], 4, 3, 4, 4).transpose(1, 2, 0), interpolation="nearest"); s.set_xticks([]); s.set_yticks([])
        s.text(0, 0, k, color="w", fontsize=7, va="top", ha="left", bbox=dict(fc="k", pad=.5, lw=0))
    ax[2].axis("off"); ax[2].set_xlim(0, 10); ax[2].set_ylim(0, 4)
    box(ax[2], (0, 2.8), 1.4, .8, "[CLS]", C[3]); 
    for k in range(4): box(ax[2], (1.6 + k * 1.2, 2.8), 1.0, .8, f"p{k}", C[0])
    ax[2].text(6.5, 3.15, "... p15", fontsize=10)
    box(ax[2], (0, 1.5), 9.2, .8, "linear: 48 numbers -> d   (+ learned position vector per token)", C[1], fs=9)
    box(ax[2], (0, .2), 9.2, .8, "Transformer encoder (Tier 7-8): 17 tokens attend to each other", C[2], fs=9)
    ax[2].set_title("17 tokens = CLS + 16 patches\n(each patch = 4*4*3 = 48 values)")
    fig.savefig(OUT / "06_vit_patchification.png"); plt.close(fig)


# 07 CLIP ------------------------------------------------------------
def clip_data():
    Xtr, ys, yc = make_dataset(900, seed=0); Xte, ys2, yc2 = make_dataset(200, seed=1)
    img, txt = TinyCNN(), TextTower(); return img, txt, Xtr, ys, yc, Xte, ys2, yc2


def fig_clip():
    img, txt, Xtr, ys, yc, Xte, ys2, yc2 = clip_data()
    idx = [int(np.where((yc2 == c) & (ys2 == s))[0][0]) for c in (0, 1, 2) for s in range(4)]
    cap = [f"{COLORS[c]} {SHAPES[s]}" for c in range(3) for s in range(4)]

    def sim():
        with torch.no_grad():
            zi = F.normalize(img.embed(torch.as_tensor(Xte[idx])), dim=-1)
            zt = F.normalize(txt(torch.arange(3).repeat_interleave(4), torch.arange(4).repeat(3)), dim=-1)
            return (zi @ zt.T).numpy()
    s0 = sim(); fit_clip(img, txt, Xtr, yc, ys, epochs=15); s1 = sim()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for a, s, t in [(ax[0], s0, "before training"), (ax[1], s1, "after contrastive training")]:
        im = a.imshow(s, cmap="viridis", vmin=-1, vmax=1); a.set_title(f"cosine similarity, {t}")
        a.set_xticks(range(12)); a.set_xticklabels(cap, rotation=90, fontsize=7); a.set_yticks(range(12)); a.set_yticklabels(cap, fontsize=7)
        a.set_xlabel("caption (text tower)"); a.set_ylabel("image (shown by its true caption)")
    fig.colorbar(im, ax=ax[:2], fraction=.025, pad=.02, label="cosine similarity")
    ax[2].set_position([.76, .2, .22, .62])
    d = np.linspace(0, 1, 5)
    z = s1
    for T, c in [(1.0, C[0]), (0.1, C[1]), (0.02, C[3])]:
        p = np.exp(z / T); p = p / p.sum(1, keepdims=True)
        ax[2].plot(range(12), np.diag(p), "o-", color=c, label=f"T={T}")
    ax[2].set_xticks(range(12)); ax[2].set_xlabel("image index"); ax[2].set_ylabel("P(own caption)")
    ax[2].set_title("Temperature sharpens the softmax\n(same trained similarities)"); ax[2].legend()
    fig.savefig(OUT / "07_clip_similarity_matrix.png"); plt.close(fig)


def gif_clip_training():
    img, txt, Xtr, ys, yc, Xte, ys2, yc2 = clip_data()
    idx = [int(np.where((yc2 == c) & (ys2 == s))[0][0]) for c in (0, 1, 2) for s in range(4)]
    mats = []

    def sim():
        with torch.no_grad():
            zi = F.normalize(img.embed(torch.as_tensor(Xte[idx])), dim=-1)
            zt = F.normalize(txt(torch.arange(3).repeat_interleave(4), torch.arange(4).repeat(3)), dim=-1)
            return (zi @ zt.T).numpy()
    mats.append(sim())
    for e in range(8):
        fit_clip(img, txt, Xtr, yc, ys, epochs=1, seed=e); mats.append(sim())
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    im = ax.imshow(mats[0], cmap="viridis", vmin=-1, vmax=1)
    ax.set_xlabel("caption index (12 colour x shape)"); ax.set_ylabel("image index (same order)"); fig.colorbar(im, label="cosine sim")

    def upd(f):
        im.set_data(mats[f]); ax.set_title(f"epoch {f}: diagonal = matched pairs", fontsize=9)
    anim = FuncAnimation(fig, upd, frames=len(mats), interval=500)
    anim.save(OUT / "07b_clip_training.gif", writer=PillowWriter(fps=2), dpi=70); plt.close(fig)


# 08 detection ------------------------------------------------------
def fig_nms():
    boxes = [(2, 2, 10, 10), (3, 3, 11, 11), (2.5, 1.5, 10.5, 9.5), (14, 12, 20, 18), (15, 12.5, 21, 18.5), (8, 12, 12, 17)]
    sc = [.9, .85, .6, .8, .7, .5]; keep = nms(boxes, sc, .5)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, (b, s) in zip(ax[:2], [(range(6), "before NMS: all 6 candidate boxes"), (keep, f"after NMS (IoU thr 0.5): keep {keep}")]):
        a.set_xlim(0, 23); a.set_ylim(20, 0); a.set_aspect("equal"); a.set_title(s, fontsize=10)
        for i in b:
            x1, y1, x2, y2 = boxes[i]; a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=C[i % 8], lw=2)); a.text(x1, y1 - .2, f"#{i} s={sc[i]}", color=C[i % 8], fontsize=8)
    t = np.linspace(0, 8, 50); ious = [iou((0, 0, 8, 8), (d, 0, 8 + d, 8)) for d in t]
    ax[2].plot(t, ious, color=C[0]); ax[2].axhline(.5, color=C[3], ls="--", label="NMS threshold 0.5")
    ax[2].set_xlabel("horizontal shift of 8x8 box (px)"); ax[2].set_ylabel("IoU with original"); ax[2].set_title("IoU decays as boxes separate"); ax[2].legend()
    fig.savefig(OUT / "08_iou_nms.png"); plt.close(fig)


# 09 diffusion GIF --------------------------------------------------
def gif_diffusion():
    x0 = (sample_image(0, 0, 24).transpose(1, 2, 0) * 2 - 1); eps = np.random.default_rng(0).normal(size=x0.shape)
    T = 100; betas = np.linspace(1e-4, 0.05, T); ab = np.cumprod(1 - betas)
    ts = [0, 5, 10, 20, 35, 50, 75, 99]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6, 1.6]})
    ax[1].plot(range(T), np.sqrt(ab), label="signal  sqrt(abar_t)", color=C[0]); ax[1].plot(range(T), np.sqrt(1 - ab), label="noise  sqrt(1-abar_t)", color=C[3])
    ax[1].set_xlabel("timestep t"); ax[1].set_title("linear beta schedule"); ax[1].legend(fontsize=8); ln = ax[1].axvline(0, color="k", ls=":")
    ims = ax[0].imshow(np.clip((x0 + 1) / 2, 0, 1), interpolation="nearest"); ax[0].set_xticks([]); ax[0].set_yticks([])
    row = np.linspace(0, 1, 24); snr = ab / (1 - ab); ax[2].semilogy(range(T), snr, color=C[2]); ax[2].set_xlabel("timestep t"); ax[2].set_title("signal-to-noise ratio (log)"); ln2 = ax[2].axvline(0, color="k", ls=":")

    def upd(f):
        t = ts[f]; xt = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
        ims.set_data(np.clip((xt + 1) / 2, 0, 1)); ax[0].set_title(f"x_t at t={t}"); ln.set_xdata([t, t]); ln2.set_xdata([t, t])
    fig.suptitle("Forward diffusion: x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps", y=1.02)
    anim = FuncAnimation(fig, lambda f: upd(min(f, len(ts) - 1)), frames=len(ts) + 2, interval=600)
    anim.save(OUT / "09_diffusion_forward.gif", writer=PillowWriter(fps=2), dpi=65); plt.close(fig)


# 10 pipelines ------------------------------------------------------
def fig_pipelines():
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.2))
    for a in ax: a.axis("off"); a.set_xlim(0, 13); a.set_ylim(0, 2)
    a = ax[0]; a.set_title("VLM architecture (LLaVA-style pattern; one common design, not the only one)", loc="left")
    box(a, (0, .6), 1.6, .8, "image", C[7]); box(a, (2.1, .6), 2.0, .8, "vision encoder\n(ViT, often frozen)", C[0], 8)
    box(a, (4.6, .6), 2.0, .8, "projector\n(MLP / Q-Former)", C[1], 8); box(a, (7.1, 1.1), 2.0, .6, "visual tokens", C[4], 8)
    box(a, (7.1, .2), 2.0, .6, "text prompt tokens", C[5], 8); box(a, (9.6, .6), 1.8, .8, "LLM decoder", C[2], 9); box(a, (11.7, .6), 1.2, .8, "answer", C[3])
    for x0, x1 in [(1.6, 2.1), (4.1, 4.6), (6.6, 7.1), (9.1, 9.6), (11.4, 11.7)]: arrow(a, (x0, 1.0), (x1, 1.0))
    a = ax[1]; a.set_title("OCR / document pipeline", loc="left")
    for k, (t, c) in enumerate([("scan / photo", C[7]), ("preprocess\n(deskew, binarize)", C[0]), ("text detection\n(boxes)", C[1]), ("recognition\n(crop -> text)", C[2]), ("layout + reading\norder", C[5]), ("structured output\n(JSON / table)", C[3])]):
        box(a, (k * 2.15, .5), 1.85, .9, t, c, 8)
        if k: arrow(a, (k * 2.15 - .3, .95), (k * 2.15, .95))
    fig.savefig(OUT / "10_vlm_ocr_pipelines.png"); plt.close(fig)


# 11 lab results ---------------------------------------------------
def fig_lab():
    Xtr, ys, yc = make_dataset(1200, seed=0); Xte, ys2, yc2 = make_dataset(400, seed=1)
    names, accs, pars, curves = [], [], [], {}
    for n, m in [("CNN", TinyCNN()), ("ResNet", TinyResNet()), ("ViT", TinyViT())]:
        cur = []
        for ep in (3, 6, 10, 20, 30):
            torch.manual_seed(0); mm = type(m)()
            cur.append(fit_classifier(mm, Xtr[:ep * 0 + len(Xtr)], ys, Xte, ys2, epochs=ep))
        curves[n] = cur; pars.append(count_params(m))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (n, cur), c in zip(curves.items(), C): ax[0].plot([3, 6, 10, 20, 30], cur, "o-", label=f"{n} ({pars[list(curves).index(n)]:,} params)", color=c)
    ax[0].axhline(.25, color="gray", ls=":", label="chance"); ax[0].set_xlabel("training epochs"); ax[0].set_ylabel("test accuracy"); ax[0].set_title("Tiny CNN vs ResNet vs ViT (1,200 synthetic images)"); ax[0].legend(fontsize=8)
    fs = []
    for frac in (.1, .25, .5, 1.0):
        n = int(1200 * frac); row = []
        for T in (TinyCNN, TinyViT): torch.manual_seed(0); row.append(fit_classifier(T(), Xtr[:n], ys[:n], Xte, ys2, epochs=30))
        fs.append(row)
    fs = np.array(fs); x = np.arange(4)
    ax[1].bar(x - .18, fs[:, 0], .36, label="CNN", color=C[0]); ax[1].bar(x + .18, fs[:, 1], .36, label="ViT", color=C[1])
    ax[1].set_xticks(x); ax[1].set_xticklabels(["10%", "25%", "50%", "100%"]); ax[1].set_xlabel("fraction of training data"); ax[1].set_ylabel("test accuracy")
    ax[1].set_title("Data efficiency: convolution's locality prior helps when data is scarce"); ax[1].title.set_fontsize(9); ax[1].legend()
    fig.savefig(OUT / "11_cnn_vs_vit_lab.png"); plt.close(fig)
    print("lab curves", curves, "data-frac", fs.tolist())


if __name__ == "__main__":
    for f in [fig_tensor, gif_conv, fig_kernels, fig_rf, fig_resnet, fig_patchify, fig_clip, gif_clip_training, fig_nms, gif_diffusion, fig_pipelines, fig_lab]:
        print(f.__name__); f()
