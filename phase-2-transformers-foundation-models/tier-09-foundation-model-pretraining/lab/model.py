"""Tier 9 lab, part 2: a tiny decoder-only language model (dense or Mixture-of-Experts FFN).

The transformer block itself is taught in Tier 8 and attention in Tier 7 -- this file is only
the *vehicle* we pretrain.  It is intentionally small (pre-norm, learned positions, GELU).
The new Tier 9 piece is `MoEFFN`: top-k routing + the Switch-style load-balancing loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 512
    ctx: int = 64
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    moe_experts: int = 0      # 0 = dense FFN; >0 = sparse MoE with this many experts
    moe_top_k: int = 2
    moe_aux_weight: float = 0.01


class MoEFFN(nn.Module):
    """Token-choice top-k Mixture-of-Experts feed-forward layer (9.25-9.27).

    router: logits = x W_r  -> softmax -> top-k experts per token -> weighted sum of expert outputs.
    aux loss (Switch Transformer): E * sum_i f_i * P_i  where f_i = fraction of tokens whose top-1 is
    expert i and P_i = mean router probability of expert i.  It is minimised (=1) by a uniform load.
    No capacity limit / token dropping here (kept simple; real systems add both -- REVIEW_REQUIRED)."""

    def __init__(self, d: int, n_exp: int, top_k: int):
        super().__init__()
        self.n_exp, self.top_k = n_exp, top_k
        self.router = nn.Linear(d, n_exp, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(n_exp)])
        self.aux_loss = torch.zeros(())
        self.last_load = torch.zeros(n_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = F.softmax(self.router(flat), dim=-1)                  # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)                 # (N, k)
        top_p = top_p / top_p.sum(-1, keepdim=True)                   # renormalise gates
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, expert(flat[tok]) * top_p[tok, slot].unsqueeze(-1))
        f = F.one_hot(top_i[:, 0], self.n_exp).float().mean(0)       # top-1 load fraction
        self.aux_loss = self.n_exp * (f * probs.mean(0)).sum()
        self.last_load = F.one_hot(top_i, self.n_exp).float().sum(1).mean(0).detach()  # tokens/expert share
        return out.reshape(B, T, D)


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv, self.proj = nn.Linear(c.d_model, 3 * c.d_model), nn.Linear(c.d_model, c.d_model)
        self.n_head = c.n_head
        self.ffn = (MoEFFN(c.d_model, c.moe_experts, c.moe_top_k) if c.moe_experts else
                    nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # Tier 7 owns this math
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok, self.pos = nn.Embedding(c.vocab_size, c.d_model), nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight                            # weight tying (Tier 8)
        self.apply(self._init)
        self.last_ce = float('nan')

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        # next-token prediction (9.17): position t predicts token t+1 (targets are pre-shifted)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.last_ce = float(loss.detach())                           # pure cross-entropy (no aux term)
        aux = sum(b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoEFFN)) if self.c.moe_experts else 0.0
        return logits, loss + (self.c.moe_aux_weight * aux if self.c.moe_experts else 0.0)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - (self.pos.weight.numel() + self.tok.weight.numel() if non_embedding else 0)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int, temperature: float = 1.0, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.ctx:])
            p = F.softmax(logits[:, -1] / max(temperature, 1e-6), dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], dim=1)
        return idx
