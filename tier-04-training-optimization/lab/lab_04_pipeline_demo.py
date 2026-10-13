"""Tier 4 lab: run the complete pipeline (code/pipeline.py) and print what each concept contributes.

    cd tier-04-training-optimization && PYTHONPATH=code python lab/lab_04_pipeline_demo.py
Each ex_* function returns a dict, like the lab files of earlier tiers; numbers use fixed seeds.
"""
import os
import tempfile

import torch

import pipeline as P


def datasets():
    return P.make_dataset(2000, 1), P.make_dataset(1000, 2)


def ex_4_1_training_loop():
    tr, va = datasets()
    t = P.Trainer(P.Config(steps=300, eval_every=20, patience=0, lr=1e-2, hidden=64), tr, va); t.fit()
    evals = [r for r in t.history if "val_acc" in r]
    return {"first_train_loss": t.history[0]["train_loss"], "last_train_loss": t.history[-1]["train_loss"], "final_val_acc": evals[-1]["val_acc"]}


def ex_4_10_schedule():
    return {s: round(P.warmup_cosine(s, 200, 20, 1e-2, 0.1), 5) for s in (0, 9, 19, 20, 100, 199)}


def ex_4_14_accumulation_equals_bigger_batch():
    tr, va = datasets()
    class Fixed(P.Stream):
        def indices(self, j):
            return torch.arange(j * self.mb, (j + 1) * self.mb) % self.n
    big = P.Trainer(P.Config(micro_batch=64, accum=1, steps=20, clip_norm=None, patience=0), tr, va); big.stream = Fixed(2000, 64, 0); big.fit()
    acc = P.Trainer(P.Config(micro_batch=16, accum=4, steps=20, clip_norm=None, patience=0), tr, va); acc.stream = Fixed(2000, 16, 0); acc.fit()
    w = lambda t: torch.cat([p.flatten() for p in t.model.parameters()])
    return {"max_weight_difference": (w(big) - w(acc)).abs().max().item()}


def ex_4_20_resume_is_exact():
    tr, va = datasets(); cfg = P.Config(steps=100, eval_every=20, patience=0, dropout=0.1)
    full = P.Trainer(cfg, tr, va); full.fit()
    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    a = P.Trainer(cfg, tr, va); a.fit(until=50); a.save(path)
    b = P.Trainer(cfg, tr, va); b.resume(path); b.fit()
    w = lambda t: torch.cat([p.flatten() for p in t.model.parameters()])
    return {"max_weight_difference_after_resume": (w(full) - w(b)).abs().max().item()}


def ex_4_25_profile():
    tr, va = datasets()
    t = P.Trainer(P.Config(steps=100, patience=0, hidden=64, micro_batch=64, accum=2), tr, va); t.fit()
    return {"phase_shares": {k: round(v, 2) for k, v in t.timer.shares().items()}, "flops_per_step": P.step_flops(t.cfg)}


if __name__ == "__main__":
    for fn in (ex_4_1_training_loop, ex_4_10_schedule, ex_4_14_accumulation_equals_bigger_batch, ex_4_20_resume_is_exact, ex_4_25_profile):
        print(fn.__name__, fn())
