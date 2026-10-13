"""Tier 13 lab (numpy part): synthetic images, convolution, pooling, receptive fields,
patchify, IoU / NMS.  Everything here is dependency-light and tested in test_tier13.py."""
import numpy as np

SHAPES = ["circle", "square", "cross", "stripes"]
COLORS = ["red", "green", "blue"]
RGB = np.array([[1.0, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.2, 1.0]], dtype=np.float32)


# ----------------------------------------------------------------- synthetic data
def draw_shape(shape: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (size,size) float mask in [0,1] for one of SHAPES, randomly placed/scaled."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = rng.uniform(0.22, 0.34) * size
    cy = rng.uniform(0.38, 0.62) * size
    cx = rng.uniform(0.38, 0.62) * size
    if shape == 0:    # circle
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    elif shape == 1:  # square
        m = (np.abs(yy - cy) <= r * 0.85) & (np.abs(xx - cx) <= r * 0.85)
    elif shape == 2:  # cross (plus sign)
        t = max(1.0, r * 0.3)
        m = ((np.abs(yy - cy) <= t) & (np.abs(xx - cx) <= r)) | ((np.abs(xx - cx) <= t) & (np.abs(yy - cy) <= r))
    else:             # diagonal stripes inside a box
        box = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
        m = box & (((yy + xx) // 2) % 2 == 0)
    return m.astype(np.float32)


def make_dataset(n: int, size: int = 16, seed: int = 0, noise: float = 0.15):
    """n synthetic RGB images (n,3,size,size), shape labels, colour labels. Deterministic in seed."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3, size, size), np.float32)
    ys = rng.integers(0, len(SHAPES), n)
    yc = rng.integers(0, len(COLORS), n)
    for i in range(n):
        m = draw_shape(int(ys[i]), size, rng)
        X[i] = RGB[yc[i]][:, None, None] * m[None]
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return np.clip(X, 0, 1), ys.astype(np.int64), yc.astype(np.int64)


# ----------------------------------------------------------------- preprocessing
def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    """Per-channel (x-mean)/std for x of shape (...,C,H,W)."""
    mean = np.asarray(mean, np.float32)[:, None, None]
    std = np.asarray(std, np.float32)[:, None, None]
    return (x - mean) / std


# ----------------------------------------------------------------- convolution
def conv_out_size(n: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    """floor((n + 2p - d(k-1) - 1)/s) + 1"""
    return (n + 2 * pad - dilation * (k - 1) - 1) // stride + 1


def conv2d_naive(x, w, b=None, stride=1, pad=0):
    """Explicit-loop cross-correlation (what deep-learning libraries call convolution).
    x: (C_in,H,W), w: (C_out,C_in,kh,kw) -> (C_out,H_out,W_out)."""
    c_in, h, wd = x.shape
    c_out, c_in2, kh, kw = w.shape
    assert c_in == c_in2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = conv_out_size(h, kh, stride, pad), conv_out_size(wd, kw, stride, pad)
    out = np.zeros((c_out, ho, wo), np.float64)
    for o in range(c_out):
        for i in range(ho):
            for j in range(wo):
                patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[o, i, j] = (patch * w[o]).sum() + (0 if b is None else b[o])
    return out


def conv2d_single(img, kernel, stride=1, pad=0):
    """Convenience: one-channel image, one kernel -> 2-D map."""
    return conv2d_naive(img[None], kernel[None, None], stride=stride, pad=pad)[0]


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "box blur": np.ones((3, 3)) / 9,
    "sobel-x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel-y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
}


def max_pool2d(x, k=2, stride=None):
    """x: (H,W) -> pooled map; stride defaults to k."""
    stride = stride or k
    ho, wo = conv_out_size(x.shape[0], k, stride), conv_out_size(x.shape[1], k, stride)
    out = np.zeros((ho, wo), x.dtype)
    for i in range(ho):
        for j in range(wo):
            out[i, j] = x[i * stride:i * stride + k, j * stride:j * stride + k].max()
    return out


def receptive_field(layers):
    """layers: list of (kernel, stride). Returns (rf, jump) after each layer.
    rf_l = rf_{l-1} + (k-1)*jump_{l-1};  jump_l = jump_{l-1}*s."""
    rf, jump, hist = 1, 1, []
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
        hist.append((rf, jump))
    return hist


# ----------------------------------------------------------------- patches (ViT)
def patchify(img, p):
    """img: (C,H,W) -> (N, p*p*C) with N=(H/p)(W/p); raster order of patches, (row,col,channel) flatten."""
    c, h, w = img.shape
    assert h % p == 0 and w % p == 0, "image size must be divisible by patch size"
    t = img.reshape(c, h // p, p, w // p, p).transpose(1, 3, 2, 4, 0)  # (gh,gw,p,p,C)
    return t.reshape((h // p) * (w // p), p * p * c)


def unpatchify(patches, p, c, h, w):
    gh, gw = h // p, w // p
    t = patches.reshape(gh, gw, p, p, c).transpose(4, 0, 2, 1, 3)  # (C,gh,p,gw,p)
    return t.reshape(c, h, w)


# ----------------------------------------------------------------- detection
def iou(a, b):
    """Boxes as (x1,y1,x2,y2). Returns intersection-over-union (0 if no overlap)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, scores, iou_thr=0.5):
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if iou(boxes[i], boxes[j]) <= iou_thr]
    return keep
