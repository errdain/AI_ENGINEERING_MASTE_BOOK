# Tier 11 — LLM Inference & Optimization — Revision Sheet

> Status: DRAFT. Generated from the concept files `11.01-autoregressive-inference.md` … `11.47-inference-observability.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 11.1 Autoregressive inference | One forward pass per generated token; each token depends on the previous one, so output length sets latency. | *One token out, one pass in.* |
| 11.2 Logits | Logits are unnormalised log-probabilities per vocabulary token; softmax makes them a distribution and all decoding controls act on them. | *Logits are scores; softmax makes them odds.* |
| 11.3 Decoding overview | Decoding rules convert logits to tokens: deterministic, stochastic or constrained, applied in a defined order. | *Logits in, one token out; choose the rule to fit the task.* |
| 11.4 Greedy decoding | Greedy = argmax each step; same as temperature 0 or top-k 1. | *Greedy: always the favourite, never a surprise.* |
| 11.5 Temperature | Divide logits by T; low T sharpens, high T flattens, T=0 is greedy. | *Temperature bends the whole curve.* |
| 11.6 Top-k sampling | Keep the k highest logits, mask the rest, renormalise, sample. | *Top-k counts.* |
| 11.7 Top-p sampling | Keep the smallest top set with cumulative probability >= p, then sample. | *Top-p weighs.* |
| 11.8 Repetition/frequency penalties | Lower seen-token logits (multiplicative or additive) to reduce loops, at the risk of hurting legitimate repetition. | *Penalties punish parrots, and sometimes poets.* |
| 11.9 Beam search | Track the B best partial sequences by summed log-prob, expand and prune; B x cost. | *Beam: keep several roads open.* |
| 11.10 Stop sequences | Strings/token patterns that end generation; pair with a token cap and finish-reason logging. | *Stop where you meant to, and cap where you didn't.* |
| 11.11 Prefill phase | Prefill = parallel pass over the prompt that builds the KV cache and yields the first token; compute-bound; determines TTFT. | *Prefill pays once.* |
| 11.12 Decode phase | Decode = one token per step per sequence; memory-bandwidth-bound; sets ITL and tokens/s. | *Decode pays per token.* |
| 11.13 Time to first token (TTFT) | TTFT = queue + prefill (+ network) until the first token; report percentiles. | *TTFT: how long until it starts talking.* |
| 11.14 Inter-token latency | Time between consecutive output tokens; set by memory bandwidth, batch and KV size. | *ITL: the rhythm of the stream.* |
| 11.15 Tokens per second | Per-user TPS ~ 1/ITL; system TPS = total tokens/s across users. | *Always say whose tokens per second.* |
| 11.16 KV cache | Cached per-layer K/V of past tokens so each step computes only one token; memory = 2 x layers x kv_heads x head_dim x tokens x bytes. | *Cache the past, compute the present.* |
| 11.17 KV-cache memory calculation | `2 x layers x kv_heads x head_dim x tokens x batch x dtype_bytes`. | *Two tensors, every layer, every token.* |
| 11.18 Prefix caching | Reuse the KV cache of an identical prompt prefix to skip its prefill; needs exact match; static content first. | *Cache the shared opening, not the unique ending.* |
| 11.19 Batching | Run multiple sequences per pass to amortise weight reads; static batching idles slots as requests finish. | *One weight read, many readers.* |
| 11.20 Continuous batching | Per-iteration scheduling: finished sequences leave and new ones join immediately; keeps GPU batch full. | *Refill every step.* |
| 11.21 PagedAttention | Block-based, on-demand KV allocation with block tables: near-zero fragmentation, more concurrency. | *Virtual memory for the KV cache.* |
| 11.22 Request scheduling | The policy that orders and admits requests (FCFS, priority, fair queues) and handles preemption. | *Schedule by SLO, not by arrival alone.* |
| 11.23 Backpressure | Signal overload upstream (bounded queue, 429/503, Retry-After) so load is shed early. | *Reject before you drown.* |
| 11.24 Admission control | Decide admit/queue/reject using size, quota, priority, queue depth and KV headroom. | *Decide at the door.* |
| 11.25 Speculative decoding | Draft k tokens cheaply, verify them in one target pass, keep what agrees; output distribution is unchanged. Helps least at large batch or low acceptance. | "Draft cheap, verify in parallel, keep what agrees." |
| 11.26 Quantization | Fewer bits per weight means fewer bytes read per token and less memory, at the cost of rounding error; group scales limit the damage from outliers. | "Fewer bits, fewer bytes, faster tokens — if you can afford the rounding." |
| 11.27 INT8 inference | INT8 halves weight memory/traffic vs FP16 with minimal loss; activation outliers are the hard part. | "INT8: half the bytes, almost the same model." |
| 11.28 INT4 inference | INT4 = 16 levels, group scales, ~4x smaller than FP16, with measurable accuracy cost. | "Sixteen levels, one scale per group: a quarter of the bytes." |
| 11.29 GGUF | GGUF = quantized weights + metadata + tokenizer in one memory-mappable file for llama.cpp. | "GGUF: the whole model in one portable box." |
| 11.30 llama.cpp | llama.cpp: portable C++ runtime for quantized GGUF models, optimized for local single-user use. | "Small C++, big reach: LLMs on your laptop." |
| 11.31 Ollama | Ollama = easy local LLM manager and API; vLLM = high-throughput GPU serving. | "Ollama: `docker run` for local LLMs." |
| 11.32 vLLM | vLLM = PagedAttention + continuous batching + OpenAI-compatible server for multi-user GPU inference. | "vLLM: paged memory, constant batching, many users." |
| 11.33 Local vs hosted inference | Hosted: no ops and pay-per-token; self-hosted: control and cheaper at high steady utilization; hybrid is common. | "Own the base load, rent the peaks." |
| 11.34 Model replicas | Replicas scale throughput and availability, not latency or model size. | "Shard to fit, replicate to scale." |
| 11.35 Tensor-parallel inference | TP shards each layer across GPUs with an all-reduce per block; fast links needed. | "Split each layer, sum the pieces." |
| 11.36 Pipeline-parallel inference | PP splits by layers; low communication; no latency gain; needs micro-batches. | "Layers on a line: cheap hops, no speedup for one request." |
| 11.37 Serving autoscaling | Autoscale on queue/tokens/KV pressure with a warm floor and graceful drain. | "Scale before the spike; warm to be fast." |
| 11.38 Cold starts | Cold start = scheduling + weight transfer + load + warm-up, often minutes; hide it with warm capacity. | "Big weights, slow warm-up: keep some replicas hot." |
| 11.39 Memory bandwidth | Decode is limited by bytes/bandwidth; ITL ≥ weight bytes ÷ bandwidth. | "Decode is a library trip for one book." |
| 11.40 Compute-bound vs memory-bound inference | Roofline: min(peak, bandwidth × intensity); decode left of the ridge, long prefill right. | "Left of the ridge: bytes; right: FLOPs." |
| 11.41 Prefill vs decode bottlenecks | Prefill: compute-bound, TTFT; decode: memory-bound, ITL. | "Read fast, write slow." |
| 11.42 Concurrency | Concurrency = arrival rate × latency; raise it until the SLO knee. | "Seats at the table are the limit." |
| 11.43 Throughput | Throughput = tokens/s system-wide; batching trades per-user speed for it. | "More riders per bus." |
| 11.44 GPU-memory budgeting | Free-for-KV = usable − weights − workspace; divide by bytes/token. | "Weights first, KV gets the rest." |
| 11.45 Inference cost | Cost = $/hour ÷ tokens/hour; utilization is the lever. | "Dollars per hour over tokens per hour." |
| 11.46 Serving reliability | Timeouts, limits, real health checks, safe retries, fallbacks, canaries. | "Assume it breaks; degrade gracefully." |
| 11.47 Inference observability | Percentiles, tokens, finish reasons, KV saturation. | "Percentiles, tokens and finish reasons." |
