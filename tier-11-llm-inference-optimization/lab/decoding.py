"""Decoding strategies on logits + generation loops (with / without KV cache)."""
import torch


def apply_temperature(logits, t):
    assert t > 0
    return logits / t


def top_k_filter(logits, k):
    """Keep the k largest logits (ties at the threshold are kept); others -> -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus: smallest set of tokens (by prob, descending) whose cumulative prob >= p. Always keeps >=1 token."""
    if p >= 1.0:
        return logits
    sl, si = torch.sort(logits, descending=True, dim=-1)
    probs = sl.softmax(-1)
    before = probs.cumsum(-1) - probs               # mass strictly before each token
    drop_sorted = before >= p                       # token is outside the nucleus once mass-before already >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, si, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def min_p_filter(logits, min_p):
    probs = logits.softmax(-1)
    return logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, float("-inf"))


def repetition_penalty(logits, generated, penalty=1.2, freq=0.0, presence=0.0):
    """CTRL-style multiplicative penalty + OpenAI-style additive frequency/presence penalties. logits: (V,)"""
    out = logits.clone()
    if len(generated) == 0:
        return out
    ids, counts = torch.unique(torch.as_tensor(generated), return_counts=True)
    l = out[ids]
    l = torch.where(l > 0, l / penalty, l * penalty)
    out[ids] = l - freq * counts.float() - presence
    return out


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, generator=None):
    """logits (V,). temperature==0 -> greedy."""
    if temperature == 0:
        return int(logits.argmax())
    l = apply_temperature(logits, temperature)
    l = top_k_filter(l, top_k)
    l = top_p_filter(l, top_p)
    if min_p > 0:
        l = min_p_filter(l, min_p)
    return int(torch.multinomial(l.softmax(-1), 1, generator=generator))


@torch.no_grad()
def generate(model, prompt, max_new, use_cache=True, stop_ids=(), rep_penalty=1.0, **sample_kw):
    """Single-sequence generation. Returns list of token ids (prompt + new). Stops at any id in stop_ids."""
    model.eval()
    ids = list(prompt)
    caches = None
    x = torch.tensor([ids])
    for _ in range(max_new):
        if use_cache:
            logits, caches = model(x, caches)          # first call = prefill; later calls feed one token
        else:
            logits, _ = model(torch.tensor([ids]))     # recompute everything every step
        l = logits[0, -1]
        if rep_penalty != 1.0:
            l = repetition_penalty(l, ids, rep_penalty)
        nxt = sample_next(l, **sample_kw)
        ids.append(nxt)
        x = torch.tensor([[nxt]])
        if nxt in stop_ids:
            break
    return ids


def hit_stop_sequence(text_ids, stop_seqs):
    """True if the generated ids end with any stop sequence (list of lists of ids)."""
    return any(len(s) > 0 and list(text_ids[-len(s):]) == list(s) for s in stop_seqs)


@torch.no_grad()
def beam_search(model, prompt, max_new, beams=3, length_penalty=0.0):
    """Beam search by summed log-prob (re-runs the model per beam; clarity over speed). Returns (ids, score) best first."""
    model.eval()
    beam = [(0.0, list(prompt))]
    for _ in range(max_new):
        cand = []
        for sc, ids in beam:
            lp = model(torch.tensor([ids]))[0][0, -1].log_softmax(-1)
            top = torch.topk(lp, beams)
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((sc + v, ids + [i]))
        cand.sort(key=lambda c: -c[0] / (len(c[1]) ** length_penalty if length_penalty else 1))
        beam = cand[:beams]
    return [(ids, sc) for sc, ids in beam]
