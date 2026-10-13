# Tier 5 — Sequence Models: Revision Sheet

> Status: DRAFT (Author: Claude). Condensed from concept files `05.01`–`05.27`; the concept file is the authoritative source. Section 2 is generated from sections 18 and 23 of each file.
> Code evidence: `code/` (10 tests) and `lab/` (7 tests) compare RNN, LSTM, GRU and attention Seq2Seq on small tasks; all page examples run under `tests/test_md_snippets.py`.

## 1. One-page rule sheet

| Topic | Rule | Note |
|---|---|---|
| Language model | `P(w₁…w_T) = Π P(w_t | w_{<t})`; train with cross-entropy; perplexity = exp(loss) | compare perplexity only under the same tokenization |
| One-hot → embedding | `one-hot · W = W[id]` | embeddings are dense, similarity-aware, `V × d` |
| Word2Vec | predict context words; negative sampling `−log σ(u_o·v_c) − Σ log σ(−u_n·v_c)` | CBOW: centre from mean of context; skip-gram: each context from centre |
| GloVe | `w_i·w̃_j + b_i + b̃_j ≈ log X_ij`, weight `f(X_ij)` | global counts; final vector = sum of the two |
| FastText | word vector = sum of character n-gram vectors | handles OOV and typos |
| RNN | `hₜ = tanh(W_xh xₜ + W_hh h(t−1) + b)` | shared weights; parameters independent of length |
| BPTT | unroll; sum shared-weight gradients over steps; truncate with `detach` | memory O(T·H) |
| Gradient problem | gradient ∝ ρⁿ over n steps | vanishing → short memory; exploding → clip |
| LSTM | `c = f⊙c + i⊙g`, `h = o⊙tanh(c)` | 4× RNN parameters; forget-bias 1 trick |
| GRU | `h = (1−z)⊙n + z⊙h_prev` | 3/4 of the LSTM parameters; no output gate |
| Encoder-decoder | context vector → autoregressive decoder with `<bos>`/`<eos>` | fixed-size context is the bottleneck |
| Decoding | greedy, beam search (length-normalized), sampling | beam width trades quality and cost |
| Teacher forcing | feed true previous tokens in training | fast, parallel; causes exposure bias |
| Exposure bias | train on correct prefixes, test on own prefixes | measure the free-running vs teacher-forced gap |
| Bahdanau attention | `e = vᵀtanh(W_s s_{t−1} + W_h hⱼ)`, softmax, weighted sum | previous state; additive |
| Luong attention | scores: dot / general / concat; current state | dot-product → Transformer attention |
| Why RNNs bottleneck | serial steps T, path T−1, fixed state | attention: parallel, path 1, T² cost |
| Transformer | `softmax(QKᵀ/√d)V` + positions + causal mask | recurrence removed |

## 2. Rapid-fire answers and memory hooks

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 5.1 Sequential data | Order matters; tasks: next-item, classification, labelling, seq2seq; variable length → padding/masks; split by time; baseline = repeat last. | **Sequential data: the order is part of the data.** |
| 5.2 Language as a sequence | Text → tokens → ids; model `P(next / previous)`; chain rule; train with cross-entropy; perplexity = exp(loss). | **Language modelling: guess the next token, billions of times.** |
| 5.3 One-hot encoding | Single 1 at the word's index; `e_i·W = W[i]`; huge and sparse; all words orthogonal; embeddings are the dense alternative. | **One-hot: a name tag with nothing written on it but the seat number.** |
| 5.4 Word embeddings | Dense vector per token from a `V × d` table; trained so that words in similar contexts are close; cosine similarity; static vs contextual. | **Embedding: a word's coordinates in meaning-space, learned from the company it keeps.** |
| 5.5 Word2Vec | Predict context words; two matrices (in/out); negative sampling loss `−log σ(u_o·v_c) − Σ log σ(−u_n·v_c)`; CBOW vs skip-gram; fast. | **Word2Vec: learn a word by guessing its neighbours, and tell real neighbours from random ones.** |
| 5.6 CBOW (continuous bag of words) | Predict centre from the mean of context embeddings; fast; ignores order; skip-gram is the mirror image. | **CBOW: fill in the blank from the bag of neighbours.** |
| 5.7 Skip-gram | Centre → each context word; ~2w pairs per token; better for rare words; negative sampling or hierarchical softmax for speed. | **Skip-gram: from one word, guess all its neighbours.** |
| 5.8 GloVe | Fit `w_i·w̃_j + b_i + b̃_j ≈ log X_ij` with weights `f(X_ij)`; global counts; final vector = sum of the two; similar quality to Word2Vec. | **GloVe: factorize the log of the global co-occurrence counts.** |
| 5.9 FastText | Word vector = sum of character n-gram vectors (hashed); skip-gram objective; handles OOV and typos; ancestor of sub-word tokenization. | **FastText: a word is the sum of its spelled pieces, so new words are never strangers.** |
| 5.10 Recurrent neural network (RNN) | `hₜ = tanh(W_xh xₜ + W_hh h(t−1) + b)`; shared weights over time; any length; sequential; unroll to train; vanishing gradients limit memory. | **RNN: one cell, reused at every step, carrying a memory forward.** |
| 5.11 Hidden state | Fixed-size learned memory `hₜ`; only channel from past to future; fades; reset per sequence; detach between chunks. | **Hidden state: a notepad of fixed size that is rewritten at every step.** |
| 5.12 Backpropagation through time (BPTT) | Backprop on the unrolled graph; sum shared-weight gradients over time; `dh_t = local + W_hhᵀ(tanh' ∘ dh_{t+1})`; truncate with `detach`; clip. | **BPTT: unroll, backprop, and add up the shared weights' share from every step.** |
| 5.13 RNN gradient problems | Gradient ∝ ρⁿ over n steps; vanishing → short memory; exploding → clip; fix with LSTM/GRU, good init, attention. | **RNN gradients: a long chain of multiplications that either fades out or blows up.** |
| 5.14 LSTM (long short-term memory) | `c = f⊙c + i⊙g`, `h = o⊙tanh(c)`; gates = sigmoids; additive cell path; forget-bias init; 4× RNN parameters; GRU is the lighter cousin. | **LSTM: a notebook with an eraser, a pen and a window, all controlled by learned gates.** |
| 5.15 Cell state | `c = f⊙c_prev + i⊙g`; gradient = product of forget gates; unbounded; `h = o⊙tanh(c)` reads it; return `(h_n, c_n)`. | **Cell state: the conveyor belt that carries memory unchanged unless a gate acts.** |
| 5.16 Forget gate | `f = σ(W[x,h] + b)`; scales `c_prev`; `∂c/∂c_prev = f`; init bias ≈ 1; learned resets. | **Forget gate: the eraser, with a learned rule for when to use it.** |
| 5.17 Input gate | `i = σ(W[x,h]+b)`, candidate `g = tanh(…)`, write `i⊙g`; protects memory from irrelevant input; with forget gate: hold / overwrite / accumulate / clear. | **Input gate: the filter that decides which of today's news makes it into the notebook.** |
| 5.18 Output gate | `h = o⊙tanh(c)`; separates storing from using; `h` bounded, `c` not; GRU has no output gate. | **Output gate: keep the memory in the safe, open the window only when asked.** |
| 5.19 GRU (gated recurrent unit) | One state; reset `r`, update `z`; `h = (1−z)·n + z·h_prev`; ~25% fewer parameters than LSTM; no output gate. | **GRU: an LSTM on a diet: one state, two gates, and a blend of old and new.** |
| 5.20 Encoder-decoder | Encoder → context vector → decoder generates autoregressively; any lengths; bottleneck; attention fixes it; teacher forcing in training. | **Encoder-decoder: read it all, keep the gist, then write the answer word by word.** |
| 5.21 Seq2Seq (sequence-to-sequence learning) | Encoder-decoder + token cross-entropy + teacher forcing; decode greedy/beam (with length normalization); attention fixes long inputs; exposure bias. | **Seq2Seq: train with the answer key beside you, decode on your own.** |
| 5.22 Teacher forcing | Train with true previous tokens (shifted target); fast and parallel; mismatch with inference → exposure bias; fixes: scheduled sampling, sequence-level objectives; evaluate free-running. | **Teacher forcing: practise with the answer key; take the exam without it.** |
| 5.23 Exposure bias | Train on true prefixes, test on own prefixes → errors compound; measure the free-running vs teacher-forced gap; fixes: scheduled sampling, sequence-level/RL training, scale. | **Exposure bias: trained on perfect prefixes, tested on its own messy ones.** |
| 5.24 Bahdanau attention | `e = vᵀtanh(W_s s + W_h h_j)`, `α = softmax(e)`, `c = Σ α h`; uses the previous decoder state; additive attention; learns alignment; fixes the bottleneck. | **Bahdanau attention: at every word you write, glance back at the whole source and weigh it.** |
| 5.25 Luong attention | After the RNN step; scores: dot / general / concat; `α = softmax`, `c = Σ α h`, `h̃ = tanh(W[c;s])`; global vs local; ancestor of dot-product attention. | **Luong attention: think first, then look things up with a simple dot product.** |
| 5.26 Why RNNs became a bottleneck | RNN: `T` serial steps, path `T−1`, fixed state. Attention: parallel, path 1, `T²` cost. That trade drove the move to Transformers. | **RNNs read one word at a time through a narrow door; attention opens every door at once.** |
| 5.27 Transition to Transformers | Self-attention `softmax(QKᵀ/√d)V` instead of recurrence; add positions; causal mask for generation; feed-forward + residual + LayerNorm; parallel training; `O(T²)`. | **Transformer: attention without the chain: everyone looks at everyone, all at once.** |
