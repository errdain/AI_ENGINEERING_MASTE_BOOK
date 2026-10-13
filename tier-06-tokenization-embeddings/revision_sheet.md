# Tier 6 — Tokenization & Embeddings — Revision Sheet

> Status: DRAFT. Generated from the concept files `06.01-text-preprocessing.md` … `06.32-tokens-per-character-compression-intuition.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 6.1 Text preprocessing | Canonicalize text (normalize, clean, pre-split) before the vocabulary is applied. | *Same pixels, different bytes: canonicalize before you count.* |
| 6.2 Character tokenization | One token per character: no OOV, but long sequences. | *Characters never surprise you, but they take forever to say anything.* |
| 6.3 Word tokenization | One token per word: short sequences, huge vocab, OOV. | *Words read nicely until the first word you've never seen.* |
| 6.4 Byte tokenization | Base alphabet of 256 bytes: no UNK, but non-Latin text costs 2-4x tokens before merges. | *Bytes cover every language, but not every language equally cheaply.* |
| 6.5 Subword tokenization | Frequent words whole, rare words in reusable pieces. | *Common words whole, rare words in pieces.* |
| 6.6 Vocabulary | The frozen token-to-ID table; its size V sets the embedding shape. | *The vocabulary is a contract between tokenizer and weights.* |
| 6.7 Token IDs | Index of a token in the vocabulary; selects an embedding row. | *An ID is an address, not a quantity.* |
| 6.8 UNK | Catch-all for unseen input; lossy; mostly obsolete with byte-level tokenizers. | *UNK is a shrug in token form.* |
| 6.9 PAD | Filler to make batches rectangular; always masked. | *Pad to fit the tensor, mask to forget the pad.* |
| 6.10 BOS | Start-of-sequence marker; follow the model card. | *BOS says 'start here'; user text must never get to say it.* |
| 6.11 EOS | Learned stop signal; also cap max tokens. | *No EOS, no end.* |
| 6.12 BPE | Greedy frequency-based merges of adjacent symbols produce a subword vocabulary. | *Merge the most common pair, again and again.* |
| 6.13 WordPiece | Likelihood-scored merges, `##` continuations, longest-match encoding (BERT). | *WordPiece bites the longest known chunk first.* |
| 6.14 SentencePiece | Library for BPE/Unigram on raw text with `▁` whitespace. | *SentencePiece sees one long string, spaces and all.* |
| 6.15 Unigram LM tokenizer | Top-down pruned vocabulary; Viterbi best segmentation. | *Unigram prunes down to the cheapest segmentation.* |
| 6.16 Attention masks | Mask entries set to -inf on scores before softmax (padding + causal). | *Mask the pad, mask the future, set scores to minus infinity.* |
| 6.17 Token embedding | Learned vector per token ID; static at input. | *An embedding is a trainable row per token.* |
| 6.18 Embedding matrix | V x d table; lookup = row index = one-hot @ E. | *One row per token, V rows tall.* |
| 6.19 Embedding dimensions | Width of token vectors / residual stream; parameters ~ V*d and ~d^2 per layer. | *d is the width of the highway every token rides.* |
| 6.20 Positional information | Attention is order-blind; positions must be injected (absolute, sinusoidal, RoPE, ALiBi). | *Attention sees a bag; position tells it the line.* |
| 6.21 Learned positional embeddings | Trainable position table added to token embeddings; fixed max length. | *A numbered seat for each position, and no seat 17 in a 16-seat hall.* |
| 6.22 Sinusoidal positional encoding | Fixed sin/cos vectors at geometric frequencies, added to embeddings. | *A clock face with a hand per frequency.* |
| 6.23 Relative positional encoding | Attention depends on offset j-i, not on absolute positions. | *Relative position asks 'how far apart?', not 'where am I?'.* |
| 6.24 RoPE | Rotate Q and K by position-dependent angles so scores depend on relative offset. | *Position is an angle; distance is the angle between.* |
| 6.25 ALiBi | Per-head linear distance penalty on attention logits. | *ALiBi taxes distance.* |
| 6.26 Token vs semantic embeddings | Token embedding = table row; semantic embedding = pooled encoder output for a whole text. | *One word, two meanings: lookup rows versus whole-text vectors.* |
| 6.27 Tokenizer training | Learn the vocabulary from representative data, measure per domain, freeze, version. | *Train the tokenizer on the text you will actually pay for.* |
| 6.28 Vocabulary-size trade-offs | Bigger V: shorter sequences but larger embedding/softmax and rarer tokens. | *Bigger vocab, shorter text, heavier table.* |
| 6.29 Byte fallback | Unknown characters become raw byte tokens instead of UNK. | *Byte fallback: when the dictionary fails, spell it in bytes.* |
| 6.30 Unicode normalization | Canonicalize equivalent Unicode (NFC default; NFKC lossy) before tokenization. | *Same pixels, different bytes - normalize before you count.* |
| 6.31 Tokenization efficiency | Fewer tokens per text = cheaper, faster, more context; measure per language. | *Count in tokens, with the model's own tokenizer, per language.* |
| 6.32 Tokens-per-character / compression intuition | ~4 chars/token for English; worse elsewhere; always verify. | *Four characters a token, until it isn't.* |
