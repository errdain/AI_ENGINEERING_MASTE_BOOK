"""Tier 10 lab: pretrain a tiny char-LM, then compare  full fine-tuning  vs  LoRA  vs  QLoRA-lite  on an
instruction task, then run an SFT -> DPO workflow, then adapter merging.   python run_lab.py   (~30-50 s, CPU)"""
import copy
import random
import time
import torch
from data import (base_batch, split_combos, sft_batch, sft_example, collate, encode, decode, render_chat,
                  answer, sloppy)
from tiny_lm import TinyLM, lm_loss, n_params, seq_logprob, generate
from lora import apply_lora, quantize_model_inplace
from alignment import dpo_loss
from merge import lerp_state

torch.set_num_threads(1)
ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")


def train(model, batch_fn, steps, lr, log=None, every=10):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    for s in range(steps):
        x, y, m = batch_fn()
        loss = lm_loss(model, x, y, m)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if log is not None and s % every == 0:
            log.append((s, loss.item()))
    return model


def pretrain(seed=0, steps=500, lr=3e-3):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyLM()

    def bf():
        x, y = base_batch(rng)
        return x, y, None
    train(model, bf, steps, lr)
    return model


@torch.no_grad()
def base_loss(model, seed=123, n=8):
    rng = random.Random(seed)
    xs = [base_batch(rng) for _ in range(n)]
    return sum(lm_loss(model, x, y).item() for x, y in xs) / n


def task_accuracy(model, task, combos, limit=33):
    ok = 0
    for w in combos[:limit]:
        p = encode(render_chat([{"role": "user", "content": f"{task} " + " ".join(w)}], True))
        ok += decode(generate(model, p, max_new=14)) == answer(task, w)
    return ok / min(limit, len(combos))


def sft_run(base, mode, task="up", steps=250, lr=None, seed=1, r=8, alpha=16, targets=ALL, replay=0.0):
    """mode: 'full' | 'lora' | 'qlora'.  replay = fraction of steps that use plain base text (mitigates forgetting).
    Returns (model, info)."""
    torch.manual_seed(seed)  # fixes the LoRA A init
    m = copy.deepcopy(base)
    rng = random.Random(seed)
    tr, te = split_combos()
    if mode == "qlora":
        quantize_model_inplace(m)
    if mode in ("lora", "qlora"):
        apply_lora(m, targets, r, alpha)
    lr = lr or (1e-3 if mode == "full" else 1e-2)

    def bf():
        if replay and rng.random() < replay:
            x, y = base_batch(rng)
            return x, y, torch.ones_like(y, dtype=torch.float)
        return sft_batch(rng, tr, (task,))
    t0 = time.time()
    log = []
    train(m, bf, steps, lr, log)
    info = dict(mode=mode, trainable=n_params(m, True), total=n_params(m), secs=time.time() - t0,
                acc=task_accuracy(m, task, te), base_loss=base_loss(m), log=log)
    return m, info


def full_batch(examples):
    """(ids, mask) both (B, L) aligned, mask=1 on response tokens."""
    L = 40
    ids = torch.zeros(len(examples), L, dtype=torch.long)
    mask = torch.zeros(len(examples), L)
    for i, (x, m) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x); mask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids, mask


def greedy_outcomes(model, te):
    """Fractions of greedy answers that are: correct (UPPER) / sloppy (echo, lower-case) / other."""
    up = low = 0
    for w in te:
        p = encode(render_chat([{"role": "user", "content": "up " + " ".join(w)}], True))
        o = decode(generate(model, p, max_new=14))
        up += o == answer("up", w)
        low += o == sloppy("up", w)
    n = len(te)
    return up / n, low / n, 1 - (up + low) / n


@torch.no_grad()
def pref_accuracy(model, te):
    ic, mc = full_batch([sft_example("up", w) for w in te])
    ir, mr = full_batch([sft_example("up", w, sloppy("up", w)) for w in te])
    return (seq_logprob(model, ic, mc) > seq_logprob(model, ir, mr)).float().mean().item()


def dpo_run(noisy, tr, steps=60, beta=0.1, lr=5e-5, seed=7, verbose=False, nll_weight=0.0):
    rng = random.Random(seed)
    ref = copy.deepcopy(noisy).requires_grad_(False)
    pol = copy.deepcopy(noisy)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    hist = []
    for s in range(steps):
        ws = [rng.choice(tr) for _ in range(32)]
        ic, mc = full_batch([sft_example("up", w) for w in ws])
        ir, mr = full_batch([sft_example("up", w, sloppy("up", w)) for w in ws])
        pc, pr = seq_logprob(pol, ic, mc), seq_logprob(pol, ir, mr)
        with torch.no_grad():
            rc, rr = seq_logprob(ref, ic, mc), seq_logprob(ref, ir, mr)
        loss, st = dpo_loss(pc, pr, rc, rr, beta)
        if nll_weight:  # optional SFT anchor on the chosen answer (RPO-style regulariser)
            loss = loss - nll_weight * (pc / mc[:, 1:].sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((s, loss.item(), st["margin"], st["acc"]))
        if verbose and s % 20 == 0:
            print(f"   dpo step {s:2d}  loss={loss.item():.3f}  margin={st['margin']:.2f}  pref-acc={st['acc']:.2f}")
    return pol, hist


def main():
    t00 = time.time()
    base = pretrain(steps=250)
    tr, te = split_combos()
    print(f"[base] params={n_params(base):,}  base-text loss={base_loss(base):.3f}  "
          f"up-acc={task_accuracy(base, 'up', te):.2f}  (a base model does not follow instructions)")
    runs = [("full", 150, dict(mode="full")), ("full + replay", 150, dict(mode="full", replay=0.3)),
            ("lora q,v r4", 200, dict(mode="lora", r=4, alpha=8, targets=("q_proj", "v_proj"))),
            ("lora all r8", 200, dict(mode="lora")), ("lora all r8 + replay", 200, dict(mode="lora", replay=0.3)),
            ("qlora-lite r8", 200, dict(mode="qlora"))]
    keep = {}
    for name, steps, kw in runs:
        m, info = sft_run(base, steps=steps, **kw)
        keep[name] = (m, info)
        print(f"[{name:20s}] trainable={info['trainable']:>7,} ({100*info['trainable']/info['total']:5.1f}%)  "
              f"up-acc={info['acc']:.2f}  base-text-loss={info['base_loss']:.2f}  {info['secs']:.1f}s")
    # ---- SFT on noisy data (half the answers sloppy), then DPO
    rng = random.Random(5)
    noisy = copy.deepcopy(base)
    train(noisy, lambda: sft_batch(rng, tr, ("up",), noise=0.5), 400, 1e-3)
    f = lambda m: "correct/sloppy/other = %.2f/%.2f/%.2f  pref-acc=%.2f" % (*greedy_outcomes(m, te), pref_accuracy(m, te))
    print(f"[noisy SFT       ] {f(noisy)}")
    pol, _ = dpo_run(noisy, tr, verbose=True)
    print(f"[+ DPO           ] {f(pol)}")
    pol2, _ = dpo_run(noisy, tr, nll_weight=1.0)
    print(f"[+ DPO + NLL     ] {f(pol2)}")
    # ---- merging: interpolate base -> fully fine-tuned weights (10.31): behaviour changes smoothly with t
    m_full = keep["full"][0]
    sb, sf = base.state_dict(), m_full.state_dict()
    for t in (0.0, 0.5, 1.0):
        mm = copy.deepcopy(base)
        mm.load_state_dict(lerp_state(sb, sf, t))
        print(f"[lerp t={t}] up-acc={task_accuracy(mm, 'up', te):.2f}  base-text-loss={base_loss(mm):.2f}")
    print(f"total {time.time()-t00:.1f}s")


if __name__ == "__main__":
    main()
