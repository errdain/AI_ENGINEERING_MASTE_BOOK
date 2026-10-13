"""Tier 9 lab, part 3: the pretraining loop -- seeds, checkpoints, resume, metrics JSON.

Reproducibility design (9.22, 9.28):
  * `seed_everything` seeds python / numpy / torch and enables deterministic algorithms.
  * Batches are a pure function of (seed, step):  rng = default_rng([seed, step]).  So a run that is
    stopped and resumed sees EXACTLY the same data as an uninterrupted run -- no sampler state to save.
  * Checkpoints hold model + optimizer + step + config + the metric history; written atomically
    (tmp file then os.replace) so a crash mid-write never leaves a corrupt "latest".
  * Dropout is 0, so a resumed run matches an uninterrupted run (tested to ~1e-6).
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, TinyLM


@dataclass
class TrainConfig:
    seed: int = 1337
    steps: int = 200
    batch_size: int = 16
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 8
    ckpt_every: int = 50
    model: ModelConfig = field(default_factory=ModelConfig)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)  # avoids rare CPU slow-downs on denormal floats
    torch.set_num_threads(1)  # tiny model: threads only add sync overhead (10x faster on a busy box) and noise


def lr_at(step: int, c: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr_frac * lr (the standard LLM schedule)."""
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    p = (step - c.warmup) / max(1, c.steps - c.warmup)
    return c.lr * (c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def get_batch(data: np.ndarray, ctx: int, bs: int, seed: int, step: int, split: str = "train"):
    """Deterministic batch: random windows of length ctx(+1 for the shifted target)."""
    rng = np.random.default_rng([seed, step, 0 if split == "train" else 1])
    ix = rng.integers(0, len(data) - ctx - 1, size=bs)
    x = np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model: TinyLM, data: np.ndarray, c: TrainConfig, seed_offset: int = 10**6) -> dict:
    model.eval()
    losses = []
    for i in range(c.eval_batches):
        x, y = get_batch(data, c.model.ctx, c.batch_size, c.seed, seed_offset + i, "val")
        model(x, y); losses.append(model.last_ce)
    model.train()
    v = float(np.mean(losses))
    return {"val_loss": v, "val_ppl": math.exp(v)}


def save_checkpoint(path: Path, model, opt, step: int, c: TrainConfig, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": asdict(c), "history": history}, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def flops_6nd(n_params: int, n_tokens: int) -> float:
    """Training FLOPs ~ 6 * N * D  (2ND forward + 4ND backward), the standard first-order rule."""
    return 6.0 * n_params * n_tokens


def train(c: TrainConfig, train_data: np.ndarray, val_data: np.ndarray, out_dir: str | Path,
          resume: bool = False, stop_at: int | None = None, extra_meta: dict | None = None,
          loss_spike_factor: float = 3.0) -> dict:
    """Train up to `stop_at` (default c.steps).  Returns the metrics dict (also written to metrics.json)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_everything(c.seed)
    model = TinyLM(c.model)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=c.lr, betas=(0.9, 0.95))
    step, hist = 0, {"train": [], "eval": [], "events": []}
    ck = out / "latest.pt"
    if resume and ck.exists():
        s = load_checkpoint(ck)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step, hist = s["step"], s["history"]
        hist["events"].append({"step": step, "event": "resumed"})
    end = min(stop_at or c.steps, c.steps)
    t0, tok_seen0 = time.time(), step * c.batch_size * c.model.ctx
    ema = None
    model.train()
    while step < end:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, c)
        x, y = get_batch(train_data, c.model.ctx, c.batch_size, c.seed, step)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):
            hist["events"].append({"step": step, "event": "non_finite_loss_abort"}); break
        opt.step()
        step += 1
        ce = model.last_ce
        if ema is not None and ce > loss_spike_factor * ema and step > c.warmup:
            hist["events"].append({"step": step, "event": "loss_spike", "loss": ce, "ema": ema})
        ema = ce if ema is None else 0.95 * ema + 0.05 * ce
        hist["train"].append({"step": step, "loss": ce, "lr": opt.param_groups[0]["lr"], "grad_norm": gn,
                              "tokens": step * c.batch_size * c.model.ctx})
        if step % c.eval_every == 0 or step == end:
            hist["eval"].append({"step": step, **evaluate(model, val_data, c)})
        if step % c.ckpt_every == 0 or step == end:
            save_checkpoint(ck, model, opt, step, c, hist)
            save_checkpoint(out / f"step_{step:06d}.pt", model, opt, step, c, hist) if step % (4 * c.ckpt_every) == 0 else None
    dt = time.time() - t0
    n_params = model.num_params()
    tokens = step * c.batch_size * c.model.ctx
    metrics = {
        "config": asdict(c), "n_params": n_params, "n_params_non_embedding": model.num_params(True),
        "steps_done": step, "tokens_seen": tokens, "train_flops_6nd": flops_6nd(n_params, tokens),
        "final_train_loss": hist["train"][-1]["loss"] if hist["train"] else None,
        "final_val": hist["eval"][-1] if hist["eval"] else None,
        "tokens_per_sec_this_session": (tokens - tok_seen0) / max(dt, 1e-9),
        "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads()},
        "history": hist, **(extra_meta or {}),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics
