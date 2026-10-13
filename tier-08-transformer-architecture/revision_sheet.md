# Tier 8 — Transformer Architecture — Revision Sheet

> Status: DRAFT. Generated from the concept files `08.01-original-transformer.md` … `08.30-masked-objective.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 8.1 Original Transformer | An encoder-decoder network built only from attention + feed-forward layers, with residuals and layer norm, that replaced RNNs for sequence modelling. | Attention mixes tokens, the MLP thinks per token, residuals keep the road open. |
| 8.2 Transformer encoder | A stack of blocks with unmasked (padding-only) self-attention that turns a sequence into contextual vectors. | Encoder: everyone sees everyone. |
| 8.3 Transformer decoder | A stack of blocks with causal self-attention (and optionally cross-attention) that predicts the next token. | Decoder: write forward, never peek ahead. |
| 8.4 Encoder-decoder architecture | Two stacks, one bridge: decoder queries read encoder keys and values. | Encoder reads once, decoder writes while peeking at the notes. |
| 8.5 Transformer block | Block = (norm -> attention -> add) then (norm -> FFN -> add). | Attention mixes tokens, the MLP thinks per token, residuals keep the road open. |
| 8.6 Self-attention layer | The sub-layer where tokens read from each other through Q, K, V and a mask. | Self-attention: tokens ask, tokens answer, the mask decides who may. |
| 8.7 Cross-attention layer | Attention whose queries come from the decoder and whose keys and values come from the encoder. | Cross-attention: ask with your own query, answer from someone else's notes. |
| 8.8 Feed-forward network | Per-token two-layer MLP, `d -> 4d -> d`, holding about two-thirds of block parameters. | Attention = communicate, MLP = compute. |
| 8.9 Position-wise MLP | Same MLP, every token, no talking between positions. | The MLP never looks sideways. |
| 8.10 Residual connections | `x + F(x)`: an identity path for gradients and a shared workspace for layers. | Never replace, always add. |
| 8.11 LayerNorm | Per-token standardisation over features with a learned scale and shift. | LayerNorm: re-centre and rescale every token on its own. |
| 8.12 Pre-LN | Pre-LN: `x + F(Norm(x))`, norm the branch not the road. | Pre-LN: norm the branch, not the road. |
| 8.13 Post-LN | Post-LN: `Norm(x + F(x))`, the original layout. | Post-LN: add first, then normalise everything. |
| 8.14 RMSNorm | RMSNorm: scale by the RMS only, no mean, no bias. | RMSNorm: only the size matters, not the centre. |
| 8.15 GELU | GELU: a soft, smooth ReLU, `x * Phi(x)`. | GELU softly switches. |
| 8.16 SwiGLU | SwiGLU: a learned gate decides how much of each hidden unit passes. | SwiGLU lets the token decide how open the switch is. |
| 8.17 Layer stacking | Stacking = `L` independent blocks in a row on one residual stream. | Stack many blocks, keep one stream. |
| 8.18 Hidden dimension | `d_model`: stream width; parameters scale as `d^2`. | Double the width, quadruple the weights. |
| 8.19 Number of heads | `h` heads of width `d/h`; parameters unchanged. | More heads split the same width; they do not add weights. |
| 8.20 Number of layers | `L` blocks: linear parameters, linear latency. | Each layer costs twelve d-squared and one more step of latency. |
| 8.21 Parameter counting | `N ~ 12 L d^2 + V d` (plus positions); fp16 memory `2N` bytes. | Twelve d-squared per layer, plus the vocabulary table. |
| 8.22 Weight tying | Tying: one matrix for reading tokens and for writing them. | One table, two uses: read in, write out. |
| 8.23 Encoder-only architecture | Bidirectional stack for understanding tasks. | Encoder-only: read everything, understand, don't write. |
| 8.24 BERT | BERT: masked-language-model encoder, fine-tuned for tasks. | BERT: fill in the blanks, looking both ways. |
| 8.25 Decoder-only architecture | Causal blocks, next-token objective, generate by appending. | Decoder-only: write forward, one token at a time. |
| 8.26 GPT | GPT: decoder-only, next-token pre-training, scaled. | GPT: predict the next token, at scale. |
| 8.27 Encoder-decoder architecture | Pick by output type: encoder for labels, decoder for open text, enc-dec for bounded rewrites. | Match the family to the output: label, text, or rewrite. |
| 8.28 T5/BART | T5/BART: corrupt text in, reconstruct text out. | T5/BART: tear the text, then rewrite it. |
| 8.29 Causal objective | Predict the next token from the past; the mask makes it fair. | Predict the next token; never peek. |
| 8.30 Masked objective | Mask some tokens, predict them from both sides. | Masked: fill in the blank using both sides. |
