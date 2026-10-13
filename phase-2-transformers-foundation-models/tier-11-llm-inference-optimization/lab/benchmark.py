"""Inference benchmark on a tiny random-init GPT (CPU, 2 threads). Numbers are relative, not representative of GPUs."""
import time, copy
import torch
from tiny_gpt import TinyGPT, kv_cache_bytes
from decoding import generate, beam_search
from quant import quantize_model, quant_error
from scheduler import Req, simulate, metrics
from kv_memory import kv_bytes

torch.set_num_threads(2)


def bench(fn, reps=3):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    m = TinyGPT(vocab=64, d=128, n_layer=4, n_head=8, n_kv_head=2, max_len=256).eval()
    prompt = list(range(1, 33))
    print("== KV cache vs full recompute (greedy, 32 prompt + N new) ==")
    for n in (16, 64, 128):
        a = bench(lambda: generate(m, prompt, n, use_cache=True, temperature=0), 2)
        b = bench(lambda: generate(m, prompt, n, use_cache=False, temperature=0), 2)
        print(f"  N={n:4d} cache {a*1e3:7.1f} ms | no cache {b*1e3:7.1f} ms | speedup {b/a:4.1f}x")
    print("== decoding strategies (64 new tokens) ==")
    for name, kw in [("greedy", dict(temperature=0)), ("temp 0.8", dict(temperature=.8)),
                     ("top-k 10", dict(temperature=1, top_k=10)), ("top-p 0.9", dict(temperature=1, top_p=.9))]:
        t = bench(lambda: generate(m, prompt, 64, **kw), 2)
        print(f"  {name:10s} {64/t:7.1f} tok/s")
    t = bench(lambda: beam_search(m, prompt, 16, beams=3), 1)
    print(f"  beam(3)    {16/t:7.1f} steps/s (re-runs model per beam, no cache: slow by design)")
    print("== int8 weight-only quantization ==")
    x = torch.tensor([prompt])
    ref = m(x)[0]
    for bits, grp in ((8, None), (4, 32)):
        qm = quantize_model(copy.deepcopy(m), bits, grp)
        out = qm(x)[0]
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"  int{bits}{' g'+str(grp) if grp else '   '} max|dlogit|={float((out-ref).abs().max()):.4f} top1-agree={agree:.2f}")
    w = m.blocks[0].mlp[0].weight.data
    print("  per-tensor", quant_error(w, 8), "\n  per-channel", quant_error(w, 8, dim=0))
    print("== KV memory ==")
    _, c = m(x)
    print(f"  measured cache for 32 tok: {kv_cache_bytes(c)} B; formula: {kv_bytes(4, 2, 16, 32, 1, 'fp32'):.0f} B")
    print("== scheduler (24 requests, capacity 600 KV tokens) ==")
    import random
    rnd = random.Random(1)
    reqs = [Req(i, i * 2, rnd.randint(10, 40), rnd.choice([5, 10, 40, 60])) for i in range(24)]
    for mode in ("static", "continuous"):
        f, r, log = simulate(reqs, 600, 8, mode)
        print(f"  {mode:10s}", {k: round(v, 2) for k, v in metrics(f, log).items()})


if __name__ == "__main__":
    main()
