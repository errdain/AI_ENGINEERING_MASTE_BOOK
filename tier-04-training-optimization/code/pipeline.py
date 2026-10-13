"""Tier 4 build: a complete, small PyTorch training pipeline with a profiler.

Everything the concept pages teach, in one readable module (CPU, deterministic, ~250 lines):
seeds (4.22), warm-up + cosine schedule (4.10-4.12), AdamW with no-decay groups (4.9, 4.29), gradient accumulation (4.14),
gradient clipping and a non-finite guard (4.15, 4.38), optional bf16 autocast (4.18-4.19), validation with early stopping (4.36),
atomic checkpoints with exact resume (4.20-4.21), gradient / learning-rate logging (4.35, 4.37) and a phase profiler (4.25).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------------------------------- config, seeds
@dataclasses.dataclass
class Config:
    seed: int = 0
    steps: int = 200              # optimizer steps
    micro_batch: int = 16
    accum: int = 1                # micro-batches per optimizer step (effective batch = micro_batch * accum)
    lr: float = 3e-3
    warmup: int = 20
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    clip_norm: float | None = 1.0
    bf16: bool = False
    eval_every: int = 20
    patience: int = 5             # evaluations without improvement before stopping (0 = never stop)
    hidden: int = 32
    dropout: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ----------------------------------------------------------------------------------------------- data
def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two noisy interleaved classes in 2-D plus 6 irrelevant features (needs a hidden layer)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)
    y = ((X[:, 0] * X[:, 1] + 0.2 * torch.randn(n, generator=g)) > 0).long()
    return X, y


class Stream:
    """Deterministic batch stream: micro-batch number j is a pure function of (seed, j), so a resumed run sees the same data."""

    def __init__(self, n: int, micro_batch: int, seed: int):
        self.n, self.mb, self.seed = n, micro_batch, seed
        self.per_epoch = n // micro_batch

    def indices(self, j: int) -> torch.Tensor:
        epoch, k = divmod(j, self.per_epoch)
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 1000 + epoch))
        return perm[k * self.mb:(k + 1) * self.mb]


# ----------------------------------------------------------------------------------------------- schedule, groups, model
def warmup_cosine(step: int, total: int, warmup: int, peak: float, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = peak * floor_frac
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def build_model(cfg: Config) -> nn.Module:
    return nn.Sequential(nn.Linear(8, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 2))


def param_groups(model: nn.Module, weight_decay: float):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]          # biases and normalization gains
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ----------------------------------------------------------------------------------------------- profiler
class PhaseTimer:
    def __init__(self):
        self.seconds: dict[str, float] = {}

    def run(self, phase: str, fn):
        start = time.perf_counter(); out = fn()
        self.seconds[phase] = self.seconds.get(phase, 0.0) + time.perf_counter() - start
        return out

    def shares(self) -> dict[str, float]:
        total = sum(self.seconds.values()) or 1.0
        return {k: v / total for k, v in self.seconds.items()}


def step_flops(cfg: Config) -> int:
    """Rough FLOPs of one optimizer step: forward = 2 * weights * samples, backward ~ 2x forward."""
    weights = 8 * cfg.hidden + cfg.hidden * 2
    samples = cfg.micro_batch * cfg.accum
    return 3 * 2 * weights * samples


# ----------------------------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)                                              # atomic


def load_checkpoint(path: str) -> dict:
    return torch.load(path, weights_only=False)


# ----------------------------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Config, train_data, val_data):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.Xtr, self.ytr = train_data
        self.Xva, self.yva = val_data
        self.model = build_model(cfg)
        self.opt = torch.optim.AdamW(param_groups(self.model, cfg.weight_decay), lr=cfg.lr)
        self.stream = Stream(len(self.Xtr), cfg.micro_batch, cfg.seed)
        self.step = 0
        self.history: list[dict] = []
        self.best = {"val_loss": float("inf"), "step": 0, "state": None}
        self.bad_evals = 0
        self.skipped = 0
        self.timer = PhaseTimer()

    # -- one optimizer step ----------------------------------------------------------------------------
    def train_step(self) -> dict:
        cfg = self.cfg
        lr = warmup_cosine(self.step, cfg.steps, cfg.warmup, cfg.lr, cfg.min_lr_frac)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for a in range(cfg.accum):
            idx = self.timer.run("data", lambda: self.stream.indices(self.step * cfg.accum + a))
            x, y = self.Xtr[idx], self.ytr[idx]
            def forward():
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
                    out = self.model(x)
                return nn.functional.cross_entropy(out.float(), y)
            loss = self.timer.run("forward", forward)
            self.timer.run("backward", lambda: (loss / cfg.accum).backward())
            total_loss += loss.item() / cfg.accum
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_norm if cfg.clip_norm else 1e12).item()
        if not math.isfinite(norm):
            self.skipped += 1                                            # non-finite guard: skip the update, keep the weights
            self.opt.zero_grad(set_to_none=True)
        else:
            self.timer.run("optimizer", self.opt.step)
        self.step += 1
        return {"step": self.step, "train_loss": total_loss, "lr": lr, "grad_norm": norm}

    # -- validation -------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        out = self.model(self.Xva)
        return {"val_loss": nn.functional.cross_entropy(out, self.yva).item(), "val_acc": (out.argmax(1) == self.yva).float().mean().item()}

    # -- main loop --------------------------------------------------------------------------------------
    def fit(self, until: int | None = None, ckpt_path: str | None = None, ckpt_every: int = 0) -> list[dict]:
        until = self.cfg.steps if until is None else until
        while self.step < until:
            row = self.train_step()
            if self.step % self.cfg.eval_every == 0 or self.step == self.cfg.steps:
                row.update(self.evaluate())
                if row["val_loss"] < self.best["val_loss"] - 1e-6:
                    self.best = {"val_loss": row["val_loss"], "step": self.step, "state": copy.deepcopy(self.model.state_dict())}; self.bad_evals = 0
                else:
                    self.bad_evals += 1
            self.history.append(row)
            if ckpt_path and ckpt_every and self.step % ckpt_every == 0:
                self.save(ckpt_path)
            if self.cfg.patience and self.bad_evals >= self.cfg.patience:
                break
        return self.history

    # -- checkpoints ------------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        save_checkpoint(path, {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": self.step, "history": self.history,
                               "best": self.best, "bad_evals": self.bad_evals, "skipped": self.skipped, "torch_rng": torch.get_rng_state(), "cfg": dataclasses.asdict(self.cfg)})

    def resume(self, path: str) -> None:
        c = load_checkpoint(path)
        self.model.load_state_dict(c["model"]); self.opt.load_state_dict(c["opt"])
        self.step, self.history, self.best, self.bad_evals, self.skipped = c["step"], c["history"], c["best"], c["bad_evals"], c["skipped"]
        torch.set_rng_state(c["torch_rng"])
