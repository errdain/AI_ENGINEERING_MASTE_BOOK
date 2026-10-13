"""Tier 14 lab - synthetic video, frame sampling, shot detection, temporal attention masks, cost model."""
from __future__ import annotations
import numpy as np


def synth_video(n_frames: int = 90, size: int = 32, shots: tuple[int, ...] = (30, 60), seed: int = 0) -> np.ndarray:
    """Synthetic grayscale video [T, H, W] in [0,1]: a moving square on a shot-specific background.
    Hard cuts occur at the frame indices in `shots` (background level + noise change abruptly)."""
    rng = np.random.default_rng(seed)
    bounds = [0, *shots, n_frames]
    bg_levels = rng.uniform(0.1, 0.9, len(bounds) - 1)
    v = np.zeros((n_frames, size, size))
    for s in range(len(bounds) - 1):
        for t in range(bounds[s], bounds[s + 1]):
            f = np.full((size, size), bg_levels[s])
            f += 0.15 * np.sin(np.linspace(0, 3 + 4 * s, size))[None, :]   # shot-specific texture
            k = t - bounds[s]; x = int((k * 1.0 + 3 * s) % (size - 8)); y = int(size / 2 - 4 + 6 * np.sin(k / 6))
            f[y:y + 6, x:x + 6] = 1.0 - bg_levels[s]
            v[t] = f
    return np.clip(v + 0.01 * rng.standard_normal(v.shape), 0, 1)


def sample_uniform(n_frames: int, k: int) -> np.ndarray:
    """k frame indices at the centres of k equal segments (TSN / video-LLM style)."""
    seg = n_frames / k
    return np.minimum(((np.arange(k) + 0.5) * seg).astype(int), n_frames - 1)


def sample_stride(n_frames: int, stride: int, offset: int = 0) -> np.ndarray:
    return np.arange(offset, n_frames, stride)


def sample_fps(n_frames: int, src_fps: float, target_fps: float) -> np.ndarray:
    """Pick frames so the output approximates target_fps (nearest-frame, no blending)."""
    dur = n_frames / src_fps
    return np.minimum(np.round(np.arange(0, dur, 1 / target_fps) * src_fps).astype(int), n_frames - 1)


def frame_hist(frame: np.ndarray, bins: int = 16) -> np.ndarray:
    h, _ = np.histogram(frame, bins=bins, range=(0, 1)); return h / h.sum()


def hist_distance(v: np.ndarray, bins: int = 16) -> np.ndarray:
    """L1 histogram distance between consecutive frames, length T-1; peaks at hard cuts."""
    H = np.stack([frame_hist(f, bins) for f in v])
    return np.abs(np.diff(H, axis=0)).sum(1)


def detect_shots(v: np.ndarray, threshold: float | None = None, min_gap: int = 5) -> list[int]:
    """Indices t where a new shot starts (frame t differs from t-1). Adaptive threshold = median + 6*MAD."""
    d = hist_distance(v)
    if threshold is None:
        med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6
        threshold = med + 6 * mad + 0.2
    cuts, last = [], -10 ** 9
    for i in np.where(d > threshold)[0]:
        if i + 1 - last >= min_gap: cuts.append(int(i + 1)); last = i + 1
    return cuts


def patchify_video(v: np.ndarray, tubelet: int = 2, patch: int = 8) -> np.ndarray:
    """ViViT-style tubelets: [T,H,W] -> [N_tokens, tubelet*patch*patch]."""
    T, H, W = v.shape; T2 = T // tubelet * tubelet
    x = v[:T2].reshape(T2 // tubelet, tubelet, H // patch, patch, W // patch, patch)
    return x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, tubelet * patch * patch)


def video_token_count(frames: int, size: int, patch: int, tubelet: int = 1) -> int:
    return (frames // tubelet) * (size // patch) ** 2


def temporal_attention_mask(T: int, kind: str = "full", window: int = 3, causal: bool = False) -> np.ndarray:
    """Boolean [T, T] mask (True = may attend) over frames: full / local window / causal."""
    i = np.arange(T)[:, None]; j = np.arange(T)[None, :]
    m = np.ones((T, T), bool) if kind == "full" else np.abs(i - j) <= window
    return m & (j <= i) if causal else m


def factorized_attention_cost(T: int, S: int) -> tuple[int, int]:
    """(joint, factorized) number of attention score pairs for T frames x S spatial tokens."""
    return (T * S) ** 2, T * S * S + S * T * T


def av_offset_estimate(audio_env: np.ndarray, video_env: np.ndarray, max_lag: int) -> int:
    """Lag (samples of the shared rate) maximizing cross-correlation of audio[t+lag] with video[t].
    Negative lag = the video track is delayed relative to the audio; positive = video leads."""
    a = (audio_env - audio_env.mean()) / (audio_env.std() + 1e-9)
    b = (video_env - video_env.mean()) / (video_env.std() + 1e-9)
    lags = list(range(-max_lag, max_lag + 1))
    sc = []
    for L in lags:
        x, y = (a[L:], b[:len(b) - L]) if L >= 0 else (a[:L], b[-L:])
        n = min(len(x), len(y)); sc.append(float(np.mean(x[:n] * y[:n])))
    return lags[int(np.argmax(sc))]


def video_cost(minutes: float, fps_sampled: float, tokens_per_frame: int, usd_per_mtok: float) -> dict:
    """Input-token cost of feeding sampled frames to a video-language model (price is a parameter)."""
    frames = minutes * 60 * fps_sampled; toks = frames * tokens_per_frame
    return {"frames": frames, "tokens": toks, "usd": toks / 1e6 * usd_per_mtok}


def audio_cost(minutes: float, usd_per_min: float, rtf: float) -> dict:
    return {"usd": minutes * usd_per_min, "compute_minutes": minutes * rtf}


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU of two time intervals (start, end) - the metric for temporal grounding / segment detection."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def recall_at_k(sim: np.ndarray, k: int) -> float:
    """Retrieval recall@k for a square similarity matrix whose correct match is on the diagonal."""
    ranks = (-sim).argsort(1)[:, :k]
    return float(np.mean([i in ranks[i] for i in range(len(sim))]))


def shot_embedding(frames: np.ndarray, bins: int = 16) -> np.ndarray:
    """Toy video embedding: mean of per-frame intensity histograms (a stand-in for a learned encoder)."""
    return np.mean([frame_hist(f, bins) for f in frames], axis=0)
