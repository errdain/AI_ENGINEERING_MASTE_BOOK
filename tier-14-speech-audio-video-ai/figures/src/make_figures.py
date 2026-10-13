"""Generate Tier 14 figures + GIFs.  Run:  python figures/src/make_figures.py   (about 1 minute; trains a tiny CTC model)"""
import sys, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lab"))
torch.set_num_threads(2)
from audio_core import (tone, chirp, noise, quantize, snr_db, alias_frequency, stft, log_mel, mel_filterbank, pre_emphasis,
                        cmvn, ResidualVQ, energy_vad, flags_to_segments, speaker_features, cluster_speakers, streaming_latency)
from ctc_asr import (FramewiseCTC, make_dataset, train, forced_align, ctc_collapse, VOCAB, HOP, SR, N_MELS)
from video_core import (synth_video, sample_uniform, sample_stride, hist_distance, detect_shots, temporal_attention_mask,
                        factorized_attention_cost)

OUT = HERE.parents[1] / "figures"; OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10.5})


def save(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / name, dpi=90); plt.close(fig)


def fig_sampling():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    f = 5.0; tt = np.linspace(0, 1, 2000)
    for a, sr, ttl in [(ax[0], 40, "sr=40 Hz  (8 samples/cycle)"), (ax[1], 8, "sr=8 Hz  (below Nyquist 2f=10)")]:
        a.plot(tt, np.sin(2 * np.pi * f * tt), color="#bbb", label="true 5 Hz tone")
        n = np.arange(int(sr) + 1) / sr; a.stem(n, np.sin(2 * np.pi * f * n), basefmt=" ", linefmt=C[0], markerfmt="o", label="samples")
        if sr == 8:
            fa = alias_frequency(f, sr); a.plot(tt, -np.sin(2 * np.pi * fa * tt) * np.sign(1), color=C[3], lw=2,
                                                label=f"aliased: looks like {fa:.0f} Hz")
        a.set_title(ttl); a.set_xlabel("time (s)"); a.set_ylabel("amplitude"); a.legend(fontsize=8, loc="lower left")
    x = tone(220, 0.01, 16000, 0.99); t = np.arange(len(x)) / 16000
    for b, c in [(3, C[3]), (8, C[0])]:
        ax[2].step(t * 1000, quantize(x, b), where="mid", color=c, label=f"{b} bits: SNR {snr_db(x, quantize(x, b)):.0f} dB")
    ax[2].plot(t * 1000, x, color="#999", lw=1, label="original"); ax[2].set_title("Quantization: fewer bits = coarser levels")
    ax[2].set_xlabel("time (ms)"); ax[2].set_ylabel("amplitude"); ax[2].legend(fontsize=8)
    fig.suptitle("Figure 1 - Sampling, aliasing and quantization (14.1, 14.2)", x=0.01, ha="left", weight="bold")
    save(fig, "01_sampling_aliasing_quantization.png")


def fig_amp_freq():
    sr = 8000; fig, ax = plt.subplots(2, 3, figsize=(13, 5.4)); t = np.arange(400) / sr * 1000
    for a, (x, ttl) in zip(ax[0], [(tone(200, 0.05, sr)[:400], "base: 200 Hz, amp 1"), (tone(200, 0.05, sr, 0.4)[:400], "quieter: amp 0.4 (same pitch)"),
                                  (tone(500, 0.05, sr)[:400], "higher pitch: 500 Hz (same loudness)")]):
        a.plot(t, x, color=C[0]); a.set_ylim(-1.1, 1.1); a.set_title(ttl); a.set_xlabel("time (ms)"); a.set_ylabel("amplitude")
    mix = tone(200, 0.5, sr) + 0.5 * tone(600, 0.5, sr) + 0.25 * tone(1500, 0.5, sr)
    ax[1, 0].plot(np.arange(400) / sr * 1000, mix[:400], color=C[1]); ax[1, 0].set_title("mixture of 3 tones (time domain)")
    ax[1, 0].set_xlabel("time (ms)"); ax[1, 0].set_ylabel("amplitude")
    sp = np.abs(np.fft.rfft(mix * np.hanning(len(mix)))) / len(mix) * 4; fr = np.fft.rfftfreq(len(mix), 1 / sr)
    ax[1, 1].plot(fr, sp, color=C[2]); ax[1, 1].set_xlim(0, 2500); ax[1, 1].set_title("same signal (frequency domain): 3 peaks, heights = amplitudes")
    ax[1, 1].set_xlabel("frequency (Hz)"); ax[1, 1].set_ylabel("magnitude")
    ax[1, 2].plot(t, tone(200, 0.05, sr)[:400], color=C[0], label="phase 0"); ax[1, 2].plot(t, tone(200, 0.05, sr, phase=np.pi / 2)[:400], color=C[3], label="phase +90 deg")
    ax[1, 2].set_title("phase shifts the wave, spectrum magnitude unchanged"); ax[1, 2].legend(fontsize=8); ax[1, 2].set_xlabel("time (ms)"); ax[1, 2].set_ylabel("amplitude")
    fig.suptitle("Figure 2 - Amplitude, frequency, phase, and the Fourier view (14.3)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "02_amplitude_frequency_phase.png")


def fig_spectrogram():
    sr = 8000; x = np.concatenate([chirp(300, 3000, 1.0, sr), tone(1000, 0.5, sr), np.zeros(sr // 4)])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].plot(np.arange(len(x)) / sr, x, lw=.5, color=C[0]); ax[0].set_title("waveform: chirp, then 1 kHz tone, then silence"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("amplitude")
    for a, (w, h) in zip(ax[1:], [(64, 16), (1024, 256)]):
        S = 10 * np.log10(np.abs(stft(x, w, h)) ** 2 + 1e-10).T
        im = a.imshow(S, origin="lower", aspect="auto", extent=[0, len(x) / sr, 0, sr / 2], vmin=S.max() - 70, cmap="magma")
        a.set_title(f"window={w} ({w / sr * 1000:.0f} ms), hop={h}: " + ("sharp in time, blurry in freq" if w == 64 else "sharp in freq, smeared in time"), fontsize=9)
        a.set_xlabel("time (s)"); a.set_ylabel("frequency (Hz)"); fig.colorbar(im, ax=a, label="power (dB)")
    fig.suptitle("Figure 3 - STFT spectrogram: the time/frequency resolution trade-off (14.4)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "03_spectrogram_window_tradeoff.png")


def fig_mel():
    sr = 8000; fb = mel_filterbank(16, 512, sr); fr = np.fft.rfftfreq(512, 1 / sr)
    f2 = plt.figure(figsize=(13, 3.9)); ax = f2.subplots(1, 3)
    for m in range(16): ax[0].plot(fr, fb[m], color=C[m % 8])
    ax[0].set_title("16 triangular mel filters: narrow at low f, wide at high f"); ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("filter weight")
    hz = np.linspace(0, 4000, 200); ax[1].plot(hz, 2595 * np.log10(1 + hz / 700), color=C[1]); ax[1].set_title("mel(f) = 2595 log10(1+f/700): compresses highs")
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("mel")
    data = make_dataset(1, seed=5)[0]; x, text = data[2], data[1]
    lm = log_mel(x, SR, n_mels=N_MELS, win=256, hop=HOP)
    im = ax[2].imshow(lm.T, origin="lower", aspect="auto", cmap="magma", extent=[0, len(x) / SR, 0, N_MELS]); ax[2].set_title(f'log-mel (24 bands) of synthetic utterance "{text}"')
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("mel band"); f2.colorbar(im, ax=ax[2], label="log power")
    f2.suptitle("Figure 4 - Mel filterbank and the log-mel features our CTC model eats (14.5)", x=0.01, ha="left", weight="bold"); f2.tight_layout()
    save(f2, "04_mel_filterbank_logmel.png")


def fig_preproc():
    sr = 8000; x = tone(100, 0.6, sr, 1.0) + 0.1 * tone(2500, 0.6, sr) + 0.02 * noise(0.6, sr, seed=1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); f = np.fft.rfftfreq(len(x), 1 / sr)
    for sig, lab, c in [(x, "raw", C[0]), (pre_emphasis(x), "pre-emphasis a=0.97", C[3])]:
        ax[0].semilogy(f, np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) + 1e-6, color=c, label=lab, lw=1)
    ax[0].set_title("pre-emphasis boosts highs (flattens tilt)"); ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("magnitude"); ax[0].legend(fontsize=8)
    d = make_dataset(1, seed=7)[0]; lm = log_mel(d[2], SR, n_mels=N_MELS, win=256, hop=HOP)
    ax[1].hist(lm.ravel(), 40, color=C[0], alpha=.7, label="raw log-mel"); ax[1].hist(cmvn(lm).ravel(), 40, color=C[1], alpha=.7, label="after CMVN")
    ax[1].set_title("normalisation: zero-mean, unit-variance features"); ax[1].set_xlabel("feature value"); ax[1].set_ylabel("count"); ax[1].legend(fontsize=8)
    n = [10, 20, 30, 45, 60]; ax[2].bar([f"{k}s" for k in n], [k * 16000 * 2 / 1e6 for k in n], color=C[2]); ax[2].set_title("16 kHz 16-bit mono = 32 KB/s of raw audio")
    ax[2].set_xlabel("duration"); ax[2].set_ylabel("size (MB)")
    fig.suptitle("Figure 5 - Audio preprocessing (14.6)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "05_audio_preprocessing.png")


def fig_rvq():
    X = np.random.default_rng(0).standard_normal((2000, 8)); q = ResidualVQ(6, 32).fit(X)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    errs = [np.mean((X - q.decode(q.encode(X, s))) ** 2) for s in range(1, 7)]
    ax[0].plot(range(1, 7), errs, "o-", color=C[3]); ax[0].set_title("reconstruction MSE vs #RVQ stages (used)"); ax[0].set_xlabel("stages (codebooks) used"); ax[0].set_ylabel("MSE")
    ax[1].plot(range(1, 7), [q.bits_per_vector(s) for s in range(1, 7)], "s-", color=C[0]); ax[1].set_title("bits per vector = stages x log2(32)")
    ax[1].set_xlabel("stages used"); ax[1].set_ylabel("bits per frame vector")
    for fps_ in (50, 75):
        ax[2].plot(range(1, 7), [fps_ * q.bits_per_vector(s) / 1000 for s in range(1, 7)], "o-", label=f"{fps_} frames/s")
    ax[2].set_title("bitrate (kbps) = frames/s x bits (toy numbers)"); ax[2].set_xlabel("stages used"); ax[2].set_ylabel("kbps"); ax[2].legend(fontsize=8)
    fig.suptitle("Figure 6 - Residual vector quantization: drop codebooks to trade quality for bitrate (14.8)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "06_rvq_codec_bitrate.png")


_MODEL = {}


def trained():
    if not _MODEL:
        tr = make_dataset(200, seed=1); m = FramewiseCTC(); _MODEL["l"] = train(m, tr, epochs=30); _MODEL["m"] = m
        _MODEL["ex"] = make_dataset(8, seed=2)
    return _MODEL["m"], _MODEL["ex"], _MODEL["l"]


def fig_ctc():
    m, ex, losses = trained(); f, text, x, lab = ex[0]
    with torch.no_grad(): lp = m(torch.from_numpy(f)[None])[0].numpy()
    path = lp.argmax(-1); spans = forced_align(lp, text)
    fig = plt.figure(figsize=(13, 5.8)); gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    a = fig.add_subplot(gs[0, :]); im = a.imshow(np.exp(lp).T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    a.set_yticks(range(len(VOCAB))); a.set_yticklabels(VOCAB); a.set_xlabel("frame (10 ms)"); a.set_ylabel("label")
    for ch, s, e in spans: a.axvspan(s - .5, e - .5, color="white", alpha=.18); a.text((s + e) / 2 - .5, len(VOCAB) - .3, ch, color="white", ha="center", weight="bold")
    a.set_title(f'CTC posteriors P(label | frame) for "{text}"; white bands = forced alignment spans; decoded: "{ctc_collapse(path)}"'); fig.colorbar(im, ax=a, label="probability", pad=0.01)
    b = fig.add_subplot(gs[1, 0]); b.plot(losses, color=C[3]); b.set_yscale("log"); b.set_title("CTC training loss (toy model)"); b.set_xlabel("epoch"); b.set_ylabel("loss (log)")
    c = fig.add_subplot(gs[1, 1]); c.axis("off")
    c.text(0, .9, "Frame path (argmax):", weight="bold"); c.text(0, .5, " ".join(VOCAB[p] if p else "_" for p in path[:36]) + " ...", family="monospace", fontsize=8)
    c.text(0, .1, f"collapse repeats, drop blank (_)  ->  {ctc_collapse(path)}", family="monospace", fontsize=9)
    fig.suptitle("Figure 7 - CTC: peaky frame posteriors, blanks, and forced alignment (14.10, 14.15)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "07_ctc_posteriors_alignment.png")


def fig_vad_diar():
    sr = 8000; rng = np.random.default_rng(0); parts, truth = [], []
    for i, (spk, f) in enumerate([(0, 180), (1, 520), (0, 180), (1, 520), (0, 180)]):
        parts += [noise(0.3, sr, 0.003, seed=i), tone(f, 0.6, sr, 0.8) + 0.4 * tone(2 * f, 0.6, sr) + noise(0.6, sr, 0.02, seed=50 + i)]; truth.append(spk)
    x = np.concatenate(parts); flags = energy_vad(x, sr, 20, -30); segs = flags_to_segments(flags, 0.02)
    F = np.stack([speaker_features(x[int(a * sr):int(b * sr)], sr) for a, b in segs]); lab = cluster_speakers(F, 2)
    fig, ax = plt.subplots(3, 1, figsize=(12, 5.4), sharex=True); t = np.arange(len(x)) / sr
    ax[0].plot(t, x, lw=.4, color=C[0]); ax[0].set_ylabel("amplitude"); ax[0].set_title("waveform: 5 voiced segments from 2 synthetic 'speakers' separated by noise")
    ax[1].step(np.arange(len(flags)) * .02, flags, where="post", color=C[2]); ax[1].set_ylabel("speech?"); ax[1].set_title("energy VAD (-30 dB threshold, 2-frame hangover)")
    for (a, b), l in zip(segs, lab): ax[2].barh(0, b - a, left=a, color=C[l], height=.6); ax[2].text((a + b) / 2, 0, f"spk {l}", ha="center", va="center", color="white", weight="bold")
    ax[2].set_yticks([]); ax[2].set_xlabel("time (s)"); ax[2].set_title("diarization: k-means (k=2) on mean log-mel shape of each segment"); ax[2].set_ylim(-.5, .5)
    fig.suptitle("Figure 8 - Segmentation and diarization timeline (14.13, 14.14)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "08_vad_diarization_timeline.png")


def fig_latency():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    chunks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
    for look, c in [(0, C[0]), (0.3, C[1])]:
        ax[0].plot(chunks, [streaming_latency(ch, look, 0.05) for ch in chunks], "o-", color=c, label=f"lookahead {look}s")
    ax[0].set_xscale("log"); ax[0].set_xlabel("chunk length (s)"); ax[0].set_ylabel("worst-case latency (s)"); ax[0].legend(fontsize=8)
    ax[0].set_title("streaming latency = chunk + lookahead + compute (compute 50 ms)")
    stages = ["VAD/endpoint", "ASR", "LLM first token", "TTS first audio", "network"]
    casc = [0.4, 0.3, 0.5, 0.25, 0.15]; left = 0
    for s, v, c in zip(stages, casc, C): ax[1].barh(0, v, left=left, color=c, label=f"{s} {v * 1000:.0f} ms"); left += v
    ax[1].set_yticks([]); ax[1].set_xlabel("seconds from end of user speech"); ax[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.25))
    ax[1].set_title(f"cascaded voice-agent budget = {left:.2f}s (ILLUSTRATIVE numbers)")
    fig.suptitle("Figure 9 - Streaming and realtime latency (14.18, 14.19)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "09_streaming_latency_budget.png")


def fig_video():
    v = synth_video(90, 32, (30, 60)); idx = sample_uniform(90, 8); d = hist_distance(v); cuts = detect_shots(v)
    fig = plt.figure(figsize=(13, 5.6)); gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.4])
    for k, i in enumerate(idx):
        a = fig.add_subplot(gs[0, k]); a.imshow(v[i], cmap="gray", vmin=0, vmax=1); a.set_title(f"frame {i}", fontsize=8); a.axis("off")
    b = fig.add_subplot(gs[1, :]); b.plot(np.arange(1, 90), d, color=C[0], label="L1 histogram distance (t-1 -> t)")
    for c in cuts: b.axvline(c, color=C[3], ls="--", label="detected cut" if c == cuts[0] else None)
    for i in idx: b.plot([i], [-0.05], "^", color=C[1], clip_on=False, label="uniform sample (k=8)" if i == idx[0] else None)
    b.set_xlabel("frame index"); b.set_ylabel("distance"); b.legend(fontsize=8); b.set_title("shot boundaries = spikes in frame-to-frame histogram difference; uniform sampling covers all 3 shots")
    fig.suptitle("Figure 10 - Frame sampling and shot detection on a synthetic video (14.22 - 14.24)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "10_frame_sampling_shot_detection.png")


def fig_temporal():
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
    for a, (m, t) in zip(ax[:3], [(temporal_attention_mask(12), "full (bidirectional)"), (temporal_attention_mask(12, "local", 2), "local window +-2"),
                                  (temporal_attention_mask(12, "full", causal=True), "causal (streaming)")]):
        a.imshow(m, cmap="Blues", origin="upper", vmin=0, vmax=1.3); a.set_title(t, fontsize=9); a.set_xlabel("key frame"); a.set_ylabel("query frame")
    S = 196; Ts = np.array([4, 8, 16, 32, 64]); j = [factorized_attention_cost(int(T), S)[0] for T in Ts]; f = [factorized_attention_cost(int(T), S)[1] for T in Ts]
    ax[3].loglog(Ts, j, "o-", color=C[3], label="joint space-time"); ax[3].loglog(Ts, f, "s-", color=C[2], label="factorized (space then time)")
    ax[3].set_xlabel("frames T (196 patches/frame)"); ax[3].set_ylabel("attention score pairs"); ax[3].legend(fontsize=8); ax[3].set_title("why factorize", fontsize=9)
    fig.suptitle("Figure 11 - Temporal attention patterns and cost (14.26)", x=0.01, ha="left", weight="bold"); fig.tight_layout()
    save(fig, "11_temporal_attention_patterns.png")

# ------------------------------------------------------------------ GIFs

def gif_stft():
    sr = 8000; x = chirp(300, 3000, 1.0, sr); win, hop = 256, 128; S = 10 * np.log10(np.abs(stft(x, win, hop)) ** 2 + 1e-10).T
    nf = S.shape[1]; sel = list(range(0, nf, 3)); fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1.4]})
    t = np.arange(len(x)) / sr
    def draw(k):
        i = sel[k]
        for a in ax: a.clear()
        ax[0].plot(t, x, lw=.4, color="#bbb"); s0 = i * hop; ax[0].axvspan(s0 / sr, (s0 + win) / sr, color=C[3], alpha=.5); ax[0].set_title("waveform + sliding window"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("amplitude")
        fr = x[s0:s0 + win] * np.hanning(win + 1)[:-1]; fr = np.pad(fr, (0, win - len(fr)))
        ax[1].plot(np.fft.rfftfreq(win, 1 / sr), 10 * np.log10(np.abs(np.fft.rfft(fr)) ** 2 + 1e-10), color=C[0]); ax[1].set_ylim(-20, 50)
        ax[1].set_title("FFT of the windowed slice"); ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("power (dB)")
        img = np.full_like(S, np.nan); img[:, :i + 1] = S[:, :i + 1]
        ax[2].imshow(img, origin="lower", aspect="auto", extent=[0, nf * hop / sr, 0, sr / 2], cmap="magma", vmin=-20, vmax=S.max()); ax[2].set_title("spectrogram builds one column per hop"); ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("frequency (Hz)")
        fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.35); fig.suptitle("Figure 12 (GIF) - STFT: slide, window, FFT, stack (14.4)", x=0.01, ha="left", weight="bold")
    anim = FuncAnimation(fig, draw, frames=len(sel), interval=120); anim.save(OUT / "12_stft_sliding_window.gif", writer=PillowWriter(fps=8), dpi=60); plt.close(fig)


def gif_ctc():
    m, ex, _ = trained(); f, text, x, lab = ex[1]
    with torch.no_grad(): lp = m(torch.from_numpy(f)[None])[0].numpy()
    path = lp.argmax(-1); T = len(path); steps = list(range(0, T, 2)) + [T - 1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), gridspec_kw={"height_ratios": [2, 1]})
    def draw(k):
        t = steps[k] + 1
        for a in ax: a.clear()
        img = np.full((len(VOCAB), T), np.nan); img[:, :t] = np.exp(lp[:t]).T
        ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[0, T, -.5, len(VOCAB) - .5]); ax[0].axvline(t, color="white")
        ax[0].set_yticks(range(len(VOCAB))); ax[0].set_yticklabels(VOCAB); ax[0].set_xlabel("frame (10 ms)"); ax[0].set_ylabel("label"); ax[0].set_title("per-frame posteriors revealed left to right")
        ax[1].axis("off"); pth = " ".join(VOCAB[p] if p else "_" for p in path[:t])
        ax[1].text(0, .75, "argmax path ('_' = blank):", weight="bold", fontsize=9); ax[1].text(0, .35, pth[-90:], family="monospace", fontsize=8)
        ax[1].text(0, -.1, f'collapsed so far: "{ctc_collapse(path[:t])}"   (target: "{text}")', family="monospace", fontsize=10, color=C[3], weight="bold")
        fig.subplots_adjust(top=0.86, bottom=0.08, hspace=0.3); fig.suptitle("Figure 13 (GIF) - CTC greedy decoding: collapse repeats, drop blanks (14.10)", x=0.01, ha="left", weight="bold")
    anim = FuncAnimation(fig, draw, frames=len(steps), interval=140); anim.save(OUT / "13_ctc_greedy_decoding.gif", writer=PillowWriter(fps=7), dpi=60); plt.close(fig)


def gif_video():
    v = synth_video(90, 32, (30, 60)); fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 2]}); d = hist_distance(v); cuts = detect_shots(v)
    ks = list(range(0, 90, 3))
    def draw(k):
        i = ks[k]
        for a in ax: a.clear()
        ax[0].imshow(v[i], cmap="gray", vmin=0, vmax=1); ax[0].set_title(f"frame {i}"); ax[0].axis("off")
        ax[1].plot(np.arange(1, 90), d, color="#ccc"); ax[1].plot(np.arange(1, i + 1), d[:i], color=C[0])
        for c in cuts:
            if c <= i: ax[1].axvline(c, color=C[3], ls="--")
        ax[1].set_xlim(0, 90); ax[1].set_ylim(0, d.max() * 1.1); ax[1].set_xlabel("frame"); ax[1].set_ylabel("histogram L1 distance"); ax[1].set_title("shot detector: cut = spike")
        fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.2); fig.suptitle("Figure 14 (GIF) - Scanning a synthetic video for shot boundaries (14.24)", x=0.01, ha="left", weight="bold")
    anim = FuncAnimation(fig, draw, frames=len(ks), interval=120); anim.save(OUT / "14_shot_detection_scan.gif", writer=PillowWriter(fps=8), dpi=60); plt.close(fig)


if __name__ == "__main__":
    for fn in [fig_sampling, fig_amp_freq, fig_spectrogram, fig_mel, fig_preproc, fig_rvq, fig_ctc, fig_vad_diar, fig_latency, fig_video, fig_temporal, gif_stft, gif_ctc, gif_video]:
        fn(); print("done", fn.__name__)
