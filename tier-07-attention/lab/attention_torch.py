"""Attention in PyTorch: raw-math version and an nn.Module (MHA/GQA/MQA) (Tier 7 build, step 3-4).
Framework equivalent: torch.nn.functional.scaled_dot_product_attention (SDPA)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sdpa_manual(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """Raw-PyTorch SDPA. attn_mask: bool (True = keep) or additive float, broadcastable to (...,T,S)."""
    t, s, d = q.size(-2), k.size(-2), q.size(-1)
    scale = 1 / math.sqrt(d) if scale is None else scale
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        keep = torch.ones(t, s, dtype=torch.bool, device=q.device).tril(s - t)
        scores = scores.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return torch.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    """From-scratch MHA. n_kv_heads=None -> MHA, 1 -> MQA, other divisor -> GQA.
    use_sdpa=True swaps the hand-written core for F.scaled_dot_product_attention."""

    def __init__(self, d_model, n_heads, n_kv_heads=None, bias=False, use_sdpa=False):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.hk, self.d, self.use_sdpa = n_heads, n_kv_heads, d_model // n_heads, use_sdpa
        self.wq = nn.Linear(d_model, n_heads * self.d, bias=bias)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d, bias=bias)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d, bias=bias)
        self.wo = nn.Linear(n_heads * self.d, d_model, bias=bias)

    def forward(self, x, context=None, attn_mask=None, is_causal=False):
        context = x if context is None else context
        b, t, _ = x.shape
        s = context.size(1)
        q = self.wq(x).view(b, t, self.h, self.d).transpose(1, 2)          # (B,H,T,D)
        k = self.wk(context).view(b, s, self.hk, self.d).transpose(1, 2)   # (B,Hk,S,D)
        v = self.wv(context).view(b, s, self.hk, self.d).transpose(1, 2)
        rep = self.h // self.hk
        k = k.repeat_interleave(rep, dim=1)       # query head i uses KV head i // rep
        v = v.repeat_interleave(rep, dim=1)
        if self.use_sdpa:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        else:
            o = sdpa_manual(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.wo(o.transpose(1, 2).reshape(b, t, self.h * self.d))
