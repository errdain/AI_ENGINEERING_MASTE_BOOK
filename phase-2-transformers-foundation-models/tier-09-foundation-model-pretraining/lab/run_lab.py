"""Tier 9 build: run the whole miniature pretraining lab end-to-end (CPU, ~30 s).

    python run_lab.py [--out runs/demo] [--moe 0] [--steps 200]

crawl -> pipeline (clean/filter/dedup/decontaminate) -> mixture -> tokenizer -> stream -> pretrain -> metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import data_prep as dp
from model import ModelConfig, TinyLM
from pretrain import TrainConfig, train
from tok_bpe import ByteBPE

BENCH = ["the old sailor describes the quiet river near the hill and the pilot finds a heavy box at the harbor today again"]


def prepare(seed: int = 0, n_docs: int = 400, merges: int = 120, leak_benchmark: bool = True):
    raw = dp.make_synthetic_crawl(seed, n_docs)
    if leak_benchmark:  # simulate a benchmark item leaking into the crawl (9.11/9.12)
        raw.append({"id": "leak", "text": BENCH[0] + " " + BENCH[0], "source": "web", "license": "cc0", "url": "x"})
    docs, report = dp.run_pipeline(raw, BENCH)
    report["contamination_before"] = dp.contamination_rate([{"text": d["text"]} for d in raw if d["id"] == "leak"], BENCH)
    report["contamination_after"] = dp.contamination_rate(docs, BENCH)
    mixed = dp.mix_sources(docs, {"wiki": 0.4, "web": 0.2, "code": 0.2, "chat": 0.2}, total_docs=len(docs), seed=seed)
    tr_docs, va_docs = dp.split_train_val(mixed)
    tok = ByteBPE.train("\n".join(d["text"] for d in tr_docs), merges)
    tr, va = dp.docs_to_stream(tok, tr_docs), dp.docs_to_stream(tok, va_docs)
    report.update(n_train_docs=len(tr_docs), n_val_docs=len(va_docs), train_tokens=int(len(tr)), val_tokens=int(len(va)),
                  vocab_size=tok.vocab_size, train_fingerprint=dp.stream_fingerprint(tr))
    return tok, tr, va, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo"); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--moe", type=int, default=0); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tok, tr, va, rep = prepare()
    cfg = TrainConfig(steps=a.steps, model=ModelConfig(vocab_size=tok.vocab_size, moe_experts=a.moe))
    m = train(cfg, tr, va, a.out, resume=a.resume, extra_meta={"data_report": rep})
    Path(a.out, "tokenizer.json").write_text(json.dumps({"merges": tok.merges}))
    print("funnel:", rep["stages"])
    print(f"params={m['n_params']:,} tokens={m['tokens_seen']:,} 6ND={m['train_flops_6nd']:.2e} FLOPs")
    print("val:", m["final_val"], f"| {time.time() - t0:.1f}s total")
    model = TinyLM(cfg.model)
    ck = torch.load(Path(a.out) / "latest.pt", weights_only=False); model.load_state_dict(ck["model"])
    out = model.generate(torch.tensor([tok.encode(" the old")]), 40, temperature=0.8)
    print("sample:", tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
