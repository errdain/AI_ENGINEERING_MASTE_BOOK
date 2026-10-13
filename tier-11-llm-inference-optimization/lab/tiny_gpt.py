"""Minimal decoder-only GPT with an optional KV cache (self-contained; see Tier 7/8 for the math).
Pre-LN blocks, learned positions, grouped-query attention (n_kv_head <= n_head)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, n_head, n_kv_head):
        super().__init__()
        assert d % n_head == 0 and n_head % n_kv_head == 0
        self.h, self.kv, self.hd = n_head, n_kv_head, d // n_head
        self.q = nn.Linear(d, n_head * self.hd, bias=False)
        self.k = nn.Linear(d, n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(d, n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        past = 0
        if cache is not None:                      # cache = (K, V) of shape (B, kv, S, hd)
            past = cache[0].shape[2]
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        rep = self.h // self.kv                    # GQA: share each KV head across `rep` query heads
        kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        S = kk.shape[2]
        att = (q @ kk.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=past)   # query i sees keys <= past+i
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ vv).transpose(1, 2).reshape(B, T, -1)
        return self.o(y), new_cache


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, n_head, n_kv_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x, cache=None):
        a, c = self.attn(self.ln1(x), cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), c


class TinyGPT(nn.Module):
    def __init__(self, vocab=64, d=64, n_layer=2, n_head=4, n_kv_head=2, max_len=256):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, max_len=max_len)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head, n_kv_head) for _ in range(n_layer))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, caches=None):
        """idx: (B,T) new tokens. caches: list of per-layer (K,V) or None. Returns logits (B,T,V), new caches."""
        past = 0 if caches is None else caches[0][0].shape[2]
        pos = torch.arange(past, past + idx.shape[1])
        x = self.tok(idx) + self.pos(pos)
        new = []
        for i, b in enumerate(self.blocks):
            x, c = b(x, None if caches is None else caches[i])
            new.append(c)
        return self.head(self.lnf(x)), new


def kv_cache_bytes(caches):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in caches)
