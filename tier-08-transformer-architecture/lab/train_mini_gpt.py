"""Train Mini-GPT on the tiny char corpus for a fixed wall-clock budget on CPU.

    python train_mini_gpt.py --seconds 60       # writes outputs/train_log.json and outputs/sample.txt
"""
from __future__ import annotations
import argparse, json, pathlib, time
import torch
from mini_gpt import GPTConfig, MiniGPT, count_params, count_params_formula
from corpus import CORPUS

OUT = pathlib.Path(__file__).resolve().parent / "outputs"


def build_data(text: str = CORPUS, val_frac: float = 0.1):
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return chars, stoi, data[:n], data[n:]


def get_batch(data, B, T, gen):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, B, T, gen, iters=10):
    model.eval()
    ls = [model(*get_batch(data, B, T, gen))[1].item() for _ in range(iters)]
    model.train(); return sum(ls) / len(ls)


def train(cfg: GPTConfig, train_data, val_data, seconds=None, max_steps=None, B=32, lr=3e-3, seed=0, eval_every=25, log=True, callback=None):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    hist = {"step": [], "time": [], "train": [], "val_step": [], "val": []}
    t0 = time.time(); step = 0
    while True:
        el = time.time() - t0
        if (seconds is not None and el >= seconds) or (max_steps is not None and step >= max_steps):
            break
        frac = el / seconds if seconds else step / max_steps
        for g in opt.param_groups:                    # warmup (first 20 steps) then cosine decay
            g["lr"] = lr * min(1.0, (step + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(3.14159 * min(frac, 1))))
        x, y = get_batch(train_data, B, cfg.block_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist["step"].append(step); hist["time"].append(time.time() - t0); hist["train"].append(loss.item())
        if step % eval_every == 0:
            hist["val_step"].append(step); hist["val"].append(eval_loss(model, val_data, B, cfg.block_size, gen))
            if callback: callback(model, step, time.time() - t0)
            if log: print(f"step {step:5d}  t={time.time()-t0:5.1f}s  train {loss.item():.3f}  val {hist['val'][-1]:.3f}", flush=True)
        step += 1
    return model, hist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--d_model", type=int, default=96); ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--block", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(2)   # >2 threads oversubscribes tiny matmuls on small CPUs
    chars, stoi, tr, va = build_data()
    cfg = GPTConfig(vocab_size=len(chars), block_size=a.block, n_layer=a.n_layer, n_head=a.n_head,
                    d_model=a.d_model, dropout=0.1)
    print(f"corpus {len(tr)+len(va)} chars, vocab {len(chars)}")
    samples = []

    def snap(m, step, t):                     # a short sample every ~12 s: watch babble become language
        if not samples or t - samples[-1]["time"] >= 12 or step == 0:
            g = torch.Generator().manual_seed(1); torch.manual_seed(1)
            o = m.generate(torch.tensor([[stoi[c] for c in "The "]]), 110, temperature=0.8, top_k=8)[0].tolist()
            m.train(); samples.append({"step": step, "time": t, "text": "".join(chars[i] for i in o)})

    model, hist = train(cfg, tr, va, seconds=a.seconds, callback=snap)
    n = count_params(model); assert n == count_params_formula(cfg)
    print(f"params {n:,} (formula matches)  steps {len(hist['step'])}")
    prompt = "The "; idx = torch.tensor([[stoi[c] for c in prompt]])
    torch.manual_seed(1)
    out = model.generate(idx, 300, temperature=0.8, top_k=8)[0].tolist()
    sample = "".join(chars[i] for i in out); print("--- sample ---\n" + sample)
    OUT.mkdir(exist_ok=True)
    (OUT / "sample.txt").write_text(sample)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, OUT / "mini_gpt_fp16.pt")
    (OUT / "chars.json").write_text(json.dumps(chars))
    hist.update(samples=samples, params=n, vocab=len(chars), d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head,
                block=cfg.block_size, seconds=a.seconds, final_val=hist["val"][-1], steps=len(hist["step"]))
    (OUT / "train_log.json").write_text(json.dumps(hist))


if __name__ == "__main__":
    main()
