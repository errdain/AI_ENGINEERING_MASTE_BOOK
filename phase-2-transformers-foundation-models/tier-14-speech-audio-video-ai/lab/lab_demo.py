"""End-to-end Tier 14 lab: train the toy CTC recogniser, then report quality AND latency numbers.
Run:  python lab_demo.py        (about 20-40 s on 2 CPU threads)"""
from __future__ import annotations
import time
import numpy as np
import torch
from audio_core import wer, cer, real_time_factor, log_mel, stft, ResidualVQ, snr_db, quantize, tone, streaming_latency
from ctc_asr import (FramewiseCTC, make_dataset, train, transcribe, forced_align, streaming_greedy, HOP, SR, N_MELS)
from video_core import synth_video, sample_uniform, detect_shots, video_cost

torch.set_num_threads(2)


def main(n_train: int = 200, epochs: int = 30) -> dict:
    t0 = time.perf_counter()
    train_set, test_set = make_dataset(n_train, seed=1), make_dataset(60, seed=2)
    model = FramewiseCTC()
    losses = train(model, train_set, epochs=epochs)
    t_train = time.perf_counter() - t0
    res = {"loss_first": losses[0], "loss_last": losses[-1], "train_s": t_train}
    for dec in ("greedy", "beam"):
        t1 = time.perf_counter(); hyp = [transcribe(model, f, dec) for f, *_ in test_set]
        dt = time.perf_counter() - t1; audio_s = sum(len(x) for *_, x, _ in test_set) / SR
        refs = [t for _, t, *_ in test_set]
        res[dec] = {"CER": float(np.mean([cer(r, h) for r, h in zip(refs, hyp)])),
                    "utt_error_rate": float(np.mean([r != h for r, h in zip(refs, hyp)])),
                    "RTF": real_time_factor(dt, audio_s)}
    f, text, x, lab = test_set[0]
    with torch.no_grad(): lp = model(torch.from_numpy(f)[None])[0].numpy()
    res["alignment_example"] = (text, forced_align(lp, text))
    stream_text, _ = streaming_greedy(model, f, chunk=20, context=10)
    res["streaming_example"] = (text, stream_text)
    res["stream_latency_s"] = streaming_latency(20 * HOP / SR, 0.0, 0.01)
    v = synth_video(); res["shots_found"] = detect_shots(v); res["sampled_frames"] = sample_uniform(len(v), 8).tolist()
    res["video_cost"] = video_cost(10, 1.0, 256, 1.0)
    return res


if __name__ == "__main__":
    r = main()
    for k, v in r.items(): print(f"{k:20s} {v}")
