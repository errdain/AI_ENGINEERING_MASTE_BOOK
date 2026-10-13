"""Mini-GPT: a decoder-only Transformer in ~150 lines of PyTorch (Tier 8 build).

Configurable so every Tier-8 design choice can be toggled and its parameter count checked:
  norm  : "layernorm" | "rmsnorm"          (8.11, 8.14)
  prenorm: True (Pre-LN) | False (Post-LN) (8.12, 8.13)
  act   : "gelu" | "swiglu"                (8.15, 8.16)
  tie_weights                              (8.22)
Attention is written out by hand (Tier 7 owns the maths); here it is just a layer inside the block.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64      # max context length (learned absolute positions)
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None   # None -> 4*d_model (gelu) ; swiglu uses the value as given
    dropout: float = 0.0
    bias: bool = True
    norm: str = "layernorm"
    prenorm: bool = True
    act: str = "gelu"
    tie_weights: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.norm in ("layernorm", "rmsnorm") and self.act in ("gelu", "swiglu")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class RMSNorm(nn.Module):
    """x / rms(x) * g  -- no mean subtraction, no bias (8.14)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    return nn.LayerNorm(cfg.d_model) if cfg.norm == "layernorm" else RMSNorm(cfg.d_model)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (8.6). Input/output: (B, T, d_model)."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.h, self.dk = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
                             persistent=False)
        self.last_attn = None  # kept for inspection / tests

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))   # (B,h,T,dk)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)                          # (B,h,T,T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))                       # causal: hide the future
        att = att.softmax(-1)
        self.last_attn = att.detach()
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)                      # concat heads
        return self.drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network (8.8, 8.9): applied to every token independently."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.swiglu = cfg.act == "swiglu"
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias) if self.swiglu else None
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        h = F.silu(self.gate(x)) * self.up(x) if self.swiglu else F.gelu(self.up(x))
        return self.drop(self.down(h))


class Block(nn.Module):
    """Transformer block (8.5) with residual connections (8.10), Pre-LN or Post-LN."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.prenorm = cfg.prenorm
        self.n1, self.attn, self.n2, self.mlp = make_norm(cfg), CausalSelfAttention(cfg), make_norm(cfg), MLP(cfg)

    def forward(self, x):
        if self.prenorm:                      # x + F(Norm(x)) : clean residual highway
            x = x + self.attn(self.n1(x))
            return x + self.mlp(self.n2(x))
        x = self.n1(x + self.attn(x))         # Norm(x + F(x)) : original 2017 layout
        return self.n2(x + self.mlp(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))   # layer stacking (8.17)
        self.final_norm = make_norm(cfg) if cfg.prenorm else nn.Identity()    # Pre-LN needs a last norm
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight                                 # weight tying (8.22)
        self.apply(self._init)
        for n, p in self.named_parameters():                                   # GPT-2 style residual scaling
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))   # (B,T,d)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.final_norm(x))                                         # (B,T,V)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = logits.topk(top_k); logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        return idx


def count_params_formula(cfg: GPTConfig) -> int:
    """Closed-form parameter count (8.21). Must equal sum(p.numel()) for MiniGPT(cfg)."""
    d, V, L, ff, b = cfg.d_model, cfg.vocab_size, cfg.n_layer, cfg.d_ff, int(cfg.bias)
    norm = d * (2 if cfg.norm == "layernorm" else 1)
    attn = 3 * d * d + 3 * d * b + d * d + d * b
    n_mat = 3 if cfg.act == "swiglu" else 2
    mlp = n_mat * d * ff + (2 * ff + d if cfg.act == "swiglu" else ff + d) * b
    block = 2 * norm + attn + mlp
    total = V * d + cfg.block_size * d + L * block
    total += norm if cfg.prenorm else 0          # final norm only in Pre-LN
    total += 0 if cfg.tie_weights else V * d     # untied output head
    return total


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())   # tied weights counted once by .parameters()
