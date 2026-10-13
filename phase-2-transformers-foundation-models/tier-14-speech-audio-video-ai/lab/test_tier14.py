import numpy as np
import pytest
import torch

torch.set_num_threads(2)
from audio_core import (ResidualVQ, alias_frequency, cer, chirp, cmvn, dominant_freq, edit_distance, energy_vad,
                        flags_to_segments, frame_signal, hz_to_mel, istft, log_mel, mel_filterbank, mel_to_hz,
                        noise, pre_emphasis, quantize, resample_linear, snr_db, stft, tone, wer, griffin_lim, spectral_convergence, cluster_speakers,
                        speaker_features, streaming_latency)
from ctc_asr import (FramewiseCTC, VOCAB, collate, ctc_collapse, forced_align, greedy_decode, make_dataset,
                     prefix_beam_search, train, transcribe, streaming_greedy, N_MELS)
from video_core import (av_offset_estimate, detect_shots, factorized_attention_cost, hist_distance, patchify_video,
                        sample_fps, sample_stride, sample_uniform, synth_video, temporal_attention_mask,
                        video_cost, video_token_count, temporal_iou, recall_at_k, shot_embedding)

# ------------------------------------------------------------------ audio basics

def test_tone_frequency_recovered():
    x = tone(440, 1.0, 8000)
    assert abs(dominant_freq(x, 8000) - 440) < 2


def test_aliasing_folding():
    assert alias_frequency(3000, 8000) == 3000
    assert alias_frequency(5000, 8000) == 3000           # above Nyquist folds down
    x = tone(5000, 1.0, 8000)
    assert abs(dominant_freq(x, 8000) - 3000) < 2         # measured, not just formula


def test_quantization_snr_about_6db_per_bit():
    x = tone(220, 1.0, 16000, amp=0.99)
    s8, s12 = snr_db(x, quantize(x, 8)), snr_db(x, quantize(x, 12))
    assert 18 < s12 - s8 < 30                              # ~6.02 dB/bit * 4 bits
    assert s8 > 40


def test_resample_preserves_frequency():
    x = tone(300, 1.0, 16000); y = resample_linear(x, 16000, 8000)
    assert len(y) == 8000 and abs(dominant_freq(y, 8000) - 300) < 2


def test_lowpass_prevents_alias_energy():
    x = tone(6000, 1.0, 16000)                             # above new Nyquist (4000)
    assert np.abs(resample_linear(x, 16000, 8000, lowpass=True)).max() < 0.1
    assert np.abs(resample_linear(x, 16000, 8000, lowpass=False)).max() > 0.5


def test_chirp_instantaneous_frequency_increases():
    x = chirp(200, 2000, 1.0, 8000); a, b = x[:2000], x[-2000:]
    assert dominant_freq(b, 8000) > 3 * dominant_freq(a, 8000)

# ------------------------------------------------------------------ STFT / mel

def test_frame_signal_covers_all_samples():
    fr = frame_signal(np.arange(1000.0), 256, 64)
    assert fr.shape[1] == 256 and fr[-1].max() == 999


def test_stft_peak_bin_matches_frequency():
    sr, win = 8000, 256; S = np.abs(stft(tone(1000, 0.5, sr), win, 64))
    assert abs(S[5].argmax() * sr / win - 1000) <= sr / win


def test_istft_reconstructs_middle():
    x = noise(0.5, 8000, seed=3); y = istft(stft(x, 256, 64), 256, 64)
    assert np.allclose(x[256:3000], y[256:3000], atol=1e-6)


def test_mel_roundtrip_and_monotone():
    assert np.allclose(mel_to_hz(hz_to_mel(np.array([100., 1000., 4000.]))), [100, 1000, 4000])
    assert hz_to_mel(1000) - hz_to_mel(500) > hz_to_mel(4000) - hz_to_mel(3500)   # mel compresses highs
    fb = mel_filterbank(20, 256, 8000)
    assert fb.shape == (20, 129) and (fb >= 0).all() and (fb.max(1) <= 1 + 1e-9).all()
    centers = fb.argmax(1); assert (np.diff(centers) >= 0).all()


def test_mel_resolution_finer_at_low_freq():
    fb = mel_filterbank(20, 512, 8000); width = (fb > 0).sum(1)
    assert width[-1] > width[0]                            # high-frequency filters are wider


def test_log_mel_shape_and_tone_localised():
    lm = log_mel(tone(500, 0.5, 8000), 8000, n_mels=24, win=256, hop=80)
    assert lm.shape[1] == 24
    lo = lm.mean(0).argmax(); lm2 = log_mel(tone(3000, 0.5, 8000), 8000, n_mels=24).mean(0).argmax()
    assert lm2 > lo


def test_preprocessing():
    x = np.array([1.0, 1.0, 1.0]); assert pre_emphasis(x)[1] == pytest.approx(0.03)
    f = cmvn(np.random.default_rng(0).standard_normal((50, 4)) * 5 + 3)
    assert np.allclose(f.mean(0), 0, atol=1e-6) and np.allclose(f.std(0), 1, atol=1e-3)

# ------------------------------------------------------------------ codec / vad / diarization / metrics

def test_rvq_error_decreases_with_stages():
    X = np.random.default_rng(0).standard_normal((600, 8))
    q = ResidualVQ(4, 16).fit(X)
    errs = [np.mean((X - q.decode(q.encode(X, s))) ** 2) for s in (1, 2, 3, 4)]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert q.bits_per_vector() == 16 and q.bits_per_vector(2) == 8


def test_vad_finds_burst():
    sr = 8000; x = np.concatenate([noise(0.5, sr, 0.001), tone(500, 0.5, sr), noise(0.5, sr, 0.001)])
    segs = flags_to_segments(energy_vad(x, sr), 0.02)
    assert len(segs) == 1 and abs(segs[0][0] - 0.5) < 0.06 and abs(segs[0][1] - 1.0) < 0.12


def test_diarization_separates_two_synthetic_voices():
    sr = 8000; segs = [tone(f, 0.4, sr) + 0.3 * tone(2 * f, 0.4, sr) for f in (200, 200, 600, 600, 200, 600)]
    F = np.stack([speaker_features(s + noise(0.4, sr, 0.01, seed=i), sr) for i, s in enumerate(segs)])
    lab = cluster_speakers(F, 2)
    assert lab[0] == lab[1] == lab[4] and lab[2] == lab[3] == lab[5] and lab[0] != lab[2]


def test_edit_distance_and_wer_cer():
    assert edit_distance("kitten", "sitting")[0] == 3
    assert edit_distance(list("abc"), list("abc")) == (0, 0, 0, 0)
    d, s, de, ins = edit_distance("a b c".split(), "a x c d".split()); assert (d, s, de, ins) == (2, 1, 0, 1)
    assert wer("the cat sat", "the cat sat") == 0
    assert wer("the cat sat", "the dog sat down") == pytest.approx(2 / 3)
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert wer("a b", "") == 1.0                             # deletions only


def test_wer_can_exceed_one():
    assert wer("a", "a b c d") == 3.0


def test_latency_budget():
    assert streaming_latency(0.2, 0.1, 0.05) == pytest.approx(0.35)

# ------------------------------------------------------------------ CTC

def test_ctc_collapse():
    i = {c: VOCAB.index(c) for c in "ab"}
    assert ctc_collapse([0, i["a"], i["a"], 0, i["a"], i["b"], i["b"], 0]) == "aab"
    assert ctc_collapse([0, 0, 0]) == ""
    assert ctc_collapse([i["a"], i["a"], i["a"]]) == "a"       # repeats collapse; blank needed for 'aa'


def _peaky(path, V=len(VOCAB)):
    lp = np.full((len(path), V), np.log(0.01)); lp[np.arange(len(path)), path] = np.log(0.94)
    return lp


def test_greedy_and_beam_agree_on_peaky_posteriors():
    path = [0, 1, 1, 0, 1, 2, 2, 0]; lp = _peaky(path)
    assert greedy_decode(lp) == "aab" == prefix_beam_search(lp)


def test_beam_beats_greedy_on_ambiguous_case():
    # classic: blank slightly most likely each frame, but 'a' mass summed over alignments wins overall
    V = len(VOCAB); p = np.full((2, V), 1e-6); p[:, 0] = 0.55; p[:, 1] = 0.45 - 1e-6 * (V - 2)
    lp = np.log(p / p.sum(1, keepdims=True))
    assert greedy_decode(lp) == ""                           # blank, blank
    assert prefix_beam_search(lp) == "a"                     # 1-0.55^2 = 0.70 > 0.30


def test_forced_align_recovers_order_and_repeats():
    path = [0, 1, 1, 0, 1, 2, 2, 0]; spans = forced_align(_peaky(path), "aab")
    assert [s[0] for s in spans] == ["a", "a", "b"]
    assert spans[0][1] == 1 and spans[1][1] == 4 and spans[2][1] == 5


def test_torch_ctc_loss_matches_perfect_alignment_ordering():
    import torch.nn.functional as F
    good = torch.from_numpy(_peaky([0, 1, 0, 2, 0])).float()[:, None, :]
    bad = torch.from_numpy(_peaky([0, 2, 0, 1, 0])).float()[:, None, :]
    tgt = torch.tensor([[1, 2]]); il = torch.tensor([5]); tl = torch.tensor([2])
    assert F.ctc_loss(good, tgt, il, tl) < F.ctc_loss(bad, tgt, il, tl)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    tr, te = make_dataset(200, seed=1), make_dataset(30, seed=2)
    m = FramewiseCTC(); losses = train(m, tr, epochs=24)
    return m, te, losses


def test_training_reduces_loss_and_decodes(trained):
    m, te, losses = trained
    assert losses[-1] < 0.3 * losses[0]
    c = np.mean([cer(t, transcribe(m, f)) for f, t, *_ in te])
    assert c < 0.2, c


def test_forced_alignment_on_trained_model(trained):
    m, te, _ = trained; f, text, x, lab = te[0]
    with torch.no_grad(): lp = m(torch.from_numpy(f)[None])[0].numpy()
    spans = forced_align(lp, text)
    assert "".join(s[0] for s in spans) == text
    assert all(a < b for _, a, b in spans)
    assert all(spans[i][2] <= spans[i + 1][1] + 1 for i in range(len(spans) - 1))


def test_streaming_matches_offline_mostly(trained):
    m, te, _ = trained
    c = np.mean([cer(t, streaming_greedy(m, f, 20, 15)[0]) for f, t, *_ in te[:10]])
    assert c < 0.35

# ------------------------------------------------------------------ video

def test_samplers():
    assert len(sample_uniform(100, 8)) == 8 and sample_uniform(100, 4).tolist() == [12, 37, 62, 87]
    assert sample_stride(10, 3).tolist() == [0, 3, 6, 9]
    assert sample_fps(300, 30, 1).tolist() == list(range(0, 300, 30))
    assert (np.diff(sample_uniform(37, 5)) > 0).all()


def test_shot_detector_finds_cuts():
    v = synth_video(90, 32, (30, 60)); assert detect_shots(v) == [30, 60]
    d = hist_distance(v); assert d[29] > 5 * np.median(d)


def test_shot_detector_no_false_positive_single_shot():
    assert detect_shots(synth_video(60, 32, ())) == []


def test_tubelets_and_token_counts():
    v = synth_video(16, 32, ()); p = patchify_video(v, tubelet=2, patch=8)
    assert p.shape == ((16 // 2) * 16, 2 * 64)
    assert video_token_count(16, 32, 8, 2) == 128


def test_temporal_masks():
    full = temporal_attention_mask(6); assert full.all()
    loc = temporal_attention_mask(6, "local", 1); assert loc.sum() == 6 + 2 * 5
    c = temporal_attention_mask(6, "full", causal=True); assert c.sum() == 21 and not c[0, 1]
    j, f = factorized_attention_cost(16, 196); assert f < j / 10


def test_av_offset_estimate():
    rng = np.random.default_rng(0); a = rng.random(200); b = np.roll(a, 7)
    assert av_offset_estimate(a, b, 20) == -7          # b is a delayed by 7
    assert av_offset_estimate(b, a, 20) == 7


def test_video_cost_linear_in_fps():
    a = video_cost(10, 1, 256, 1.0); b = video_cost(10, 2, 256, 1.0)
    assert b["tokens"] == 2 * a["tokens"] and a["frames"] == 600


def test_griffin_lim_improves_with_iterations():
    x = chirp(300, 2500, 0.5, 8000) + 0.5 * tone(1000, 0.5, 8000)
    mag = np.abs(stft(x, 256, 64))
    e5 = spectral_convergence(mag, griffin_lim(mag, 256, 64, iters=5))
    e60 = spectral_convergence(mag, griffin_lim(mag, 256, 64, iters=60))
    assert e60 < e5 < 1.0 and e60 < 0.35, (e5, e60)


def test_temporal_iou_and_recall():
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert temporal_iou((0, 1), (2, 3)) == 0.0
    sim = np.eye(4) + 0.1 * np.random.default_rng(0).random((4, 4))
    assert recall_at_k(sim, 1) == 1.0
    assert recall_at_k(-np.eye(4), 1) < 1.0


def test_shot_embedding_groups_by_shot():
    v = synth_video(90, 32, (30, 60)); e = [shot_embedding(v[a:b]) for a, b in [(0, 15), (15, 30), (30, 45), (60, 75)]]
    d = lambda x, y: np.abs(x - y).sum()
    assert d(e[0], e[1]) < d(e[0], e[2]) and d(e[0], e[1]) < d(e[0], e[3])
