"""A ~60k-parameter char-level decoder-only transformer, written for hackability (not speed).
Linear layers are named q_proj/k_proj/v_proj/o_proj/fc1/fc2 so LoRA can target them by name (10.14).
Block internals are Tier 8's topic; here the model is just the thing we fine-tune."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import VOCAB, PAD


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        y = self.ln1(x)
        sp = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(self.q_proj(y)), sp(self.k_proj(y)), sp(self.v_proj(y)), is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, d=48, layers=2, heads=4, ctx=48):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def n_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def lm_loss(model, x, y, mask=None):
    """Mean next-token cross-entropy (optionally only where mask==1)."""
    ce = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none")
    if mask is None:
        return ce.mean()
    return (ce * mask).sum() / mask.sum().clamp(min=1)


def seq_logprob(model, ids, mask):
    """Sum of log p(token) over masked (response) positions -> one number per sequence."""
    x, y, m = ids[:, :-1], ids[:, 1:], mask[:, 1:]
    lp = torch.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (lp * m).sum(-1)


@torch.no_grad()
def generate(model, prompt_ids, max_new=24, greedy=True, end_id=3, gen=None):
    """Batch-1 generation, no KV cache (fine at this size). Returns generated ids (without the end token)."""
    ids = torch.tensor([prompt_ids])
    out = []
    for _ in range(max_new):
        logits = model(ids[:, -model.ctx:])[0, -1]
        nxt = int(logits.argmax()) if greedy else int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        if nxt == end_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
    return out
