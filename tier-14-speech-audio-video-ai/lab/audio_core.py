"""Tier 14 lab - audio fundamentals in numpy only (no downloads, no audio files).

Covers: synthetic signals, sampling/aliasing/quantization, STFT, mel filterbank, preprocessing,
toy residual-vector-quantization codec, energy VAD, toy diarization (k-means), WER/CER.
Everything is deterministic given a seed.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- synthesis (14.1-14.3)

def tone(freq: float, dur: float, sr: int, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Pure sine tone: x[n] = amp * sin(2*pi*f*n/sr + phase)."""
    t = np.arange(int(round(dur * sr))) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def chirp(f0: float, f1: float, dur: float, sr: int, amp: float = 1.0) -> np.ndarray:
    """Linear chirp from f0 to f1 Hz (phase is the integral of instantaneous frequency)."""
    n = int(round(dur * sr)); t = np.arange(n) / sr
    k = (f1 - f0) / dur
    return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def noise(dur: float, sr: int, amp: float = 1.0, seed: int = 0) -> np.ndarray:
    return amp * np.random.default_rng(seed).standard_normal(int(round(dur * sr)))


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform mid-rise quantizer on [-1, 1] with 2**bits levels (returns dequantized values)."""
    levels = 2 ** bits
    xc = np.clip(x, -1, 1 - 1e-9)
    return (np.floor((xc + 1) / 2 * levels) + 0.5) / levels * 2 - 1


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    err = clean - noisy
    return float(10 * np.log10((clean ** 2).sum() / max((err ** 2).sum(), 1e-20)))


def alias_frequency(f: float, sr: float) -> float:
    """Apparent frequency of a tone at f Hz when sampled at sr Hz (folding around sr/2)."""
    r = f % sr
    return r if r <= sr / 2 else sr - r


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def dominant_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * sr / len(x))


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int, lowpass: bool = True) -> np.ndarray:
    """Naive resampler: (optional) FFT low-pass at the new Nyquist, then linear interpolation."""
    if lowpass and sr_out < sr_in:
        X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr_in)
        X[f > sr_out / 2] = 0; x = np.fft.irfft(X, n=len(x))
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.arange(n_out) * sr_in / sr_out, np.arange(len(x)), x)

# ---------------------------------------------------------------- STFT / mel (14.4-14.5)

def frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice into overlapping frames [n_frames, win]; pads the tail with zeros so no sample is dropped."""
    n_frames = 1 + max(0, int(np.ceil((len(x) - win) / hop)))
    pad = (n_frames - 1) * hop + win - len(x)
    xp = np.pad(x, (0, max(pad, 0)))
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return xp[idx]


def stft(x: np.ndarray, win: int = 256, hop: int = 64, window: str = "hann") -> np.ndarray:
    """Complex STFT [n_frames, win//2+1]."""
    w = np.hanning(win + 1)[:-1] if window == "hann" else np.ones(win)
    return np.fft.rfft(frame_signal(x, win, hop) * w, axis=1)


def power_spectrogram(x, win=256, hop=64):
    return np.abs(stft(x, win, hop)) ** 2


def hz_to_mel(f):  # O'Malley/HTK mel scale
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank [n_mels, n_fft//2+1]."""
    fmax = fmax or sr / 2
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, ce, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9); down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[m] = np.maximum(0, np.minimum(up, down))
    return fb


def log_mel(x: np.ndarray, sr: int, n_mels: int = 24, win: int = 256, hop: int = 80, eps: float = 1e-10) -> np.ndarray:
    """Log-mel spectrogram [n_frames, n_mels] (natural log of power)."""
    P = power_spectrogram(x, win, hop)
    return np.log(P @ mel_filterbank(n_mels, win, sr).T + eps)


def istft(S: np.ndarray, win: int = 256, hop: int = 64) -> np.ndarray:
    """Overlap-add inverse of `stft` (Hann analysis window, window-squared normalisation)."""
    w = np.hanning(win + 1)[:-1]
    frames = np.fft.irfft(S, n=win, axis=1) * w
    n = hop * (len(frames) - 1) + win
    out = np.zeros(n); norm = np.zeros(n)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + win] += f; norm[i * hop:i * hop + win] += w ** 2
    return out / np.maximum(norm, 1e-8)

def griffin_lim(mag: np.ndarray, win: int = 256, hop: int = 64, iters: int = 60, seed: int = 0) -> np.ndarray:
    """Griffin-Lim: estimate a waveform from an STFT *magnitude* by alternating projections
    (a classical, non-neural 'vocoder' used to explain 14.17)."""
    rng = np.random.default_rng(seed)
    S = mag * np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(iters):
        y = istft(S, win, hop)
        A = stft(y, win, hop)[: len(mag)]
        S = mag * np.exp(1j * np.angle(A))
    return istft(S, win, hop)


def spectral_convergence(mag_ref: np.ndarray, x: np.ndarray, win: int = 256, hop: int = 64) -> float:
    """||ref - |STFT(x)|||_F / ||ref||_F  (0 = perfect magnitude match)."""
    m = np.abs(stft(x, win, hop))[: len(mag_ref)]
    return float(np.linalg.norm(mag_ref - m) / np.linalg.norm(mag_ref))


# ---------------------------------------------------------------- preprocessing (14.6)

def pre_emphasis(x: np.ndarray, a: float = 0.97) -> np.ndarray:
    return np.append(x[:1], x[1:] - a * x[:-1])


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(x))
    return x if m == 0 else x * (peak / m)


def cmvn(feats: np.ndarray) -> np.ndarray:
    """Per-utterance cepstral mean/variance normalisation over time."""
    return (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-8)


def pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))

# ---------------------------------------------------------------- neural-codec idea: RVQ (14.8)

def kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # farthest-first initialisation (deterministic given seed): avoids picking two near-identical centres
    C = [X[rng.integers(len(X))]]
    while len(C) < k:
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0); C.append(X[int(d.argmax())])
    C = np.stack(C).astype(float)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1); a = d.argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return C


class ResidualVQ:
    """Residual vector quantization: stage q quantizes the residual left by stages < q.
    This is the core of SoundStream/EnCodec-style codecs (without the encoder/decoder network)."""

    def __init__(self, n_stages: int = 4, codebook_size: int = 16):
        self.n_stages, self.k = n_stages, codebook_size
        self.books: list[np.ndarray] = []

    def fit(self, X: np.ndarray, seed: int = 0) -> "ResidualVQ":
        R = X.copy(); self.books = []
        for q in range(self.n_stages):
            C = kmeans(R, self.k, seed=seed + q); self.books.append(C)
            idx = ((R[:, None, :] - C[None]) ** 2).sum(-1).argmin(1); R = R - C[idx]
        return self

    def encode(self, X: np.ndarray, n_stages: int | None = None) -> np.ndarray:
        R = X.copy(); codes = []
        for C in self.books[: n_stages or self.n_stages]:
            idx = ((R[:, None, :] - C[None]) ** 2).sum(-1).argmin(1); codes.append(idx); R = R - C[idx]
        return np.stack(codes, 1)  # [N, stages] integer tokens

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return sum(self.books[q][codes[:, q]] for q in range(codes.shape[1]))

    def bits_per_vector(self, n_stages: int | None = None) -> float:
        return (n_stages or self.n_stages) * float(np.log2(self.k))

# ---------------------------------------------------------------- VAD / segmentation (14.13)

def energy_vad(x: np.ndarray, sr: int, frame_ms: float = 20, threshold_db: float = -30, hangover: int = 2) -> np.ndarray:
    """Per-frame speech flag: frame energy within `threshold_db` of the loudest frame, plus hangover frames."""
    win = int(sr * frame_ms / 1000)
    e = (frame_signal(x, win, win) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e / e.max())
    flag = db > threshold_db
    out = flag.copy()
    for i in np.where(flag)[0]:
        out[i:i + hangover + 1] = True
    return out


def flags_to_segments(flags: np.ndarray, frame_s: float, min_len: int = 1) -> list[tuple[float, float]]:
    segs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None: start = i
        if not f and start is not None:
            if i - start >= min_len: segs.append((start * frame_s, i * frame_s))
            start = None
    return segs

# ---------------------------------------------------------------- diarization (14.14)

def speaker_features(x: np.ndarray, sr: int, n_mels: int = 20) -> np.ndarray:
    """Tiny 'voice print': mean log-mel spectral shape over the loudest half of the frames
    (so silence at segment edges does not matter). Not a real x-vector."""
    lm = log_mel(x, sr, n_mels=n_mels, win=256, hop=128)
    loud = lm[lm.sum(1) >= np.median(lm.sum(1))]
    v = loud.mean(0)
    return v - v.mean()


def cluster_speakers(F: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    Z = (F - F.mean(0)) / (F.std(0) + 1e-8)
    C = kmeans(Z, k, seed=seed)
    return ((Z[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)

# ---------------------------------------------------------------- metrics (14.20)

def edit_distance(a, b) -> tuple[int, int, int, int]:
    """Levenshtein distance with (substitutions, deletions, insertions) counts via backtrace."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, S, De, In = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            S += int(a[i - 1] != b[j - 1]); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1: De += 1; i -= 1
        else: In += 1; j -= 1
    return int(D[n, m]), S, De, In


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h)[0] / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    return edit_distance(list(ref), list(hyp))[0] / max(len(ref), 1)


def real_time_factor(process_seconds: float, audio_seconds: float) -> float:
    """RTF < 1 means faster than real time."""
    return process_seconds / audio_seconds


def streaming_latency(chunk_s: float, lookahead_s: float, compute_s: float) -> float:
    """Worst-case algorithmic+compute latency for a chunked streaming recognizer (see 14.18)."""
    return chunk_s + lookahead_s + compute_s
