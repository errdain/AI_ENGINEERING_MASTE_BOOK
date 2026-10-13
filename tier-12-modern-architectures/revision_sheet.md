# Tier 12 — Modern Foundation-Model Architectures — Revision Sheet

> Status: DRAFT. Generated from the concept files `12.01-architectural-evolution-after-the-original-transformer.md` … `12.30-comparing-modern-model-families-without-depending-on-vendor-.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 12.1 Architectural evolution after the original Transformer | Same Transformer block; better norm/FFN/positions, fewer KV heads, sparse FFN, and sometimes SSM layers. | *Same block, new bottleneck each time.* |
| 12.2 Dense Transformers | Dense: all weights active every token; cost ≈ 2·params FLOPs. | *Dense: every weight works on every token.* |
| 12.3 Encoder-only models as an architectural family | Bidirectional, understanding-oriented, no generation. | *Encoders read everything, write nothing.* |
| 12.4 Decoder-only models as an architectural family | Causal attention + next-token loop + KV cache. | *Predict the next token, remember the past in a cache.* |
| 12.5 Encoder-decoder models as an architectural family | Encoder reads, decoder writes, cross-attention connects them. | *Read once, write with a glance back.* |
| 12.6 Multi-Query Attention (MQA) in modern models | One shared K/V head; cache ÷ H. | *One dictionary for all readers.* |
| 12.7 Grouped-Query Attention (GQA) in modern models | Groups share K/V heads; cache ÷ H/H_kv. | *Share K/V in groups; keep queries independent.* |
| 12.8 RoPE variants and extensions | Rescale RoPE frequencies (PI/base/NTK/YaRN) to extend context. | *Stretch the slow dials, keep the fast ones.* |
| 12.9 Long-context architectures | Positions + cache + sparse/linear compute + evaluation. | *Long context is a budget, not a switch.* |
| 12.10 Sliding-window attention | Attend to last w tokens; cache ≤ w. | *See nearby directly, far indirectly.* |
| 12.11 Sparse attention | Compute a structured subset of attention pairs. | *Compute the pairs that matter.* |
| 12.12 Efficient-attention architecture patterns | Match the efficiency trick to the bottleneck. | *Name the bottleneck, then choose the trick.* |
| 12.13 Mixture of Experts (MoE) | Many FFN experts, few active per token. | *Pay memory for all experts, compute for a few.* |
| 12.14 Sparse MoE | Top-k of E experts run per token. | *k of E: sparse compute, dense memory.* |
| 12.15 Expert routing | Router picks experts under a capacity limit. | *The router is the real model inside the model.* |
| 12.16 Expert load balancing | Auxiliary loss/noise to even out expert loads. | *Spread the work or experts die.* |
| 12.17 State Space Models (SSMs) | Fixed state + linear recurrence = O(1) memory per token. | *A summary that never grows.* |
| 12.18 Mamba | Selective SSM with a scan kernel. | *An SSM that decides what to remember.* |
| 12.19 Transformer-SSM hybrid architectures | Few attention layers + many SSM layers. | *A few attention layers buy back recall.* |
| 12.20 Multimodal foundation-model architectures | Encoder + projector + LLM; modality tokens cost context. | *Turn pictures into tokens, pay in context.* |
| 12.21 Retrieval-enhanced model architectures | Consult external text at inference instead of only weights. | *Open-book inference.* |
| 12.22 Reasoning-oriented training/inference patterns | More tokens at inference, trained to use them. | *Think longer, pay per token.* |
| 12.23 Architecture selection trade-offs | Filter by memory/latency math, then evaluate on your task. | *Compute the cache before you pick the model.* |
| 12.24 Architecture case study: BERT family | BERT: a case study; compute from the config. | *BERT: bidirectional reader, no generator.* |
| 12.25 Architecture case study: GPT family | GPT: a case study; compute from the config. | *GPT: the decoder-only scaling template.* |
| 12.26 Architecture case study: T5/BART family | T5/BART: a case study; compute from the config. | *T5/BART: encode, then write with cross-attention.* |
| 12.27 Architecture case study: Llama family | Llama: a case study; compute from the config. | *Llama: the open template of the modern dense decoder.* |
| 12.28 Architecture case study: Qwen family | Qwen: a case study; compute from the config. | *Qwen: few KV heads, big vocabulary, long context.* |
| 12.29 Architecture case study: Gemma family | Gemma: a case study; compute from the config. | *Gemma: alternate local and global layers.* |
| 12.30 Comparing modern model families without depending on vendor-specific APIs | Compare by config-derived costs plus your own eval, not vendor APIs. | *Compare the machine, not the badge.* |
