# Tier 7 — Attention — Revision Sheet

> Status: DRAFT. Generated from the concept files `07.01-why-attention.md` … `07.30-sliding-window-attention-introduction-only-full-architecture.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 7.1 Why attention | Content-based soft lookup: weights from query·key, output = weighted values. | *Ask with a query, match on keys, take the values.* |
| 7.2 Query | Query = X·W_q: what this token is looking for. | *The query is the question a token carries.* |
| 7.3 Key | Key = X·W_k: how a token advertises itself for matching. | *The key is the label that gets you found.* |
| 7.4 Value | Value = X·W_v: the payload that gets averaged. | *The value is the payload you actually receive.* |
| 7.5 Q/K/V projections | Q=XW_q, K=XW_k, V=XW_v: three learned linear maps. | *Q asks, K answers "am I relevant", V is the payload.* |
| 7.6 Dot-product similarity | score = q·k = ‖q‖‖k‖cos θ. | *Dot product: aligned and long means a high score.* |
| 7.7 Attention scores | Scores = Q Kᵀ (scaled, masked) before softmax. | *Scores are pre-softmax; weights are post-softmax.* |
| 7.8 Scaling by √dk | Divide by √d_k to keep score variance ≈ 1. | *Dot, then divide by root so softmax stays soft.* |
| 7.9 Softmax | Row-wise exp-normalise over keys; subtract the max first. | *Exp, normalise, over keys; subtract the max first.* |
| 7.10 Weighted values | Out = W·V: weighted average of value rows. | *Weights say how much, values say what.* |
| 7.11 Scaled dot-product attention | `softmax(QKᵀ/√d_k + mask)V`. | *Dot, divide by root, softmax, blend.* |
| 7.12 Self-attention | Q, K, V from the same sequence. | *Self: ask yourself.* |
| 7.13 Cross-attention | Q from one sequence, K and V from another. | *Cross: ask someone else.* |
| 7.14 Causal attention | Mask j>i with −∞ so no token sees the future. | *Causal: the past only.* |
| 7.15 Bidirectional attention | No causal mask: everyone sees everyone. | *Bidirectional: everyone talks to everyone.* |
| 7.16 Attention masks | Mask scores with −∞ before softmax; never after. | *Mask before softmax, never after.* |
| 7.17 Padding masks | Key-side mask hiding PAD tokens. | *Pad tokens are invisible to attention.* |
| 7.18 Multi-head attention | H parallel attentions on d_model/H slices, concatenated and mixed by W_o. | *Many small attentions, one output mix.* |
| 7.19 Head dimensions | d_h = d_model / H. | *Width divided by heads.* |
| 7.20 Projection matrices | W_q, W_k, W_v in; W_o out; 4·d² params for MHA. | *Three matrices in, one matrix out.* |
| 7.21 MQA | One shared K/V head for all query heads. | *Many askers, one answer-book.* |
| 7.22 GQA | Fewer KV heads than query heads: smaller KV cache, little quality loss. | *Many askers, few answer-books.* |
| 7.23 Attention complexity | ≈4·B·H·n²·d_h FLOPs per layer; quadratic in n. | *Every token talks to every token: n squared.* |
| 7.24 Attention memory requirements | KV cache = 2·L·H_kv·d_h·n·B·bytes; scores = B·H·n²·bytes. | *Cache grows with n, scores with n squared.* |
| 7.25 FlashAttention | Exact, tiled, IO-aware attention: O(n) extra memory, O(n²) compute. | *Same answer, never write the big matrix.* |
| 7.26 Attention visualization | Heatmap of weights: where it looked, not why it answered. | *A heatmap of where it looked, not why it answered.* |
| 7.27 Attention patterns | Heads specialise: previous token, local, sink, induction, diffuse. | *Heads develop habits.* |
| 7.28 Attention sinks | Sinks: first tokens absorb surplus attention mass; keep them when evicting. | *Park the spare attention on token zero — don't evict it.* |
| 7.29 Long-context attention limitations | Capacity ≠ use: cost, position extrapolation, middle-loss, dilution. | *A big window is not a big memory.* |
| 7.30 Sliding-window attention (introduction only) | Attend only to the last w tokens: O(n·w). | *Look only at the recent window.* |
