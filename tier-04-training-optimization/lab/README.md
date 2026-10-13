# Tier 4 Lab — a complete PyTorch training pipeline with a profiler

The roadmap's build task: *"complete PyTorch training pipeline + profiler."* `code/pipeline.py` is one readable module (CPU, deterministic) that combines what the concept pages teach: seeds (4.22), warm-up + cosine schedule (4.10–4.12), AdamW with no-decay groups (4.9, 4.29), gradient accumulation (4.14), clipping and a non-finite guard (4.15, 4.38), optional BF16 autocast (4.19), validation with early stopping (4.36), atomic checkpoints with exact resume (4.20–4.21), gradient/learning-rate logging (4.35, 4.37) and a phase profiler with FLOP counting (4.25).

```sh
cd tier-04-training-optimization
pip install -r requirements.txt
python -m pytest -q code lab                          # 16 tests
PYTHONPATH=code python lab/lab_04_pipeline_demo.py    # print the demo results
```

| File | Content |
|---|---|
| `code/pipeline.py` | `Config`, `set_seed`, deterministic `Stream`, `warmup_cosine`, `param_groups`, `PhaseTimer`, `step_flops`, `Trainer` (train_step / evaluate / fit / save / resume) |
| `code/test_pipeline.py` | 11 tests: schedule shape, no-decay groups, accumulation = bigger batch, exact resume (with dropout), resume without optimizer state differs, clipping bound, seed reproducibility, NaN guard, learning + early stopping, BF16 close to FP32, profiler phases |
| `lab/lab_04_pipeline_demo.py` | `ex_*` demos: training loop, schedule values, accumulation equivalence (max weight difference ~1e-7), exact resume (difference 0.0), phase shares and FLOPs |
| `lab/test_lab_04_pipeline_demo.py` | 5 tests on those demos |

Not covered (needs GPUs / several processes): real CUDA mixed precision with `GradScaler`, tensor/pipeline parallelism, FSDP/ZeRO sharding, GPU memory counters. The concept pages show CPU simulations of the mathematics (all-reduce, shard/gather, pipeline schedule) and mark GPU behaviour as NOT_TESTED.
