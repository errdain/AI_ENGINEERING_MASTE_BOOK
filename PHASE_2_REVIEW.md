# Phase 2 review (Tiers 6–15) and restructuring

Reviewed by Claude. Phase 2 was authored by ChatGPT on `chatgpt-bgc-1012` and merged into `main` as `phase-2-transformers-foundation-models/`. This review covers (1) what was found, (2) what was changed to put it in the same layout as Tiers 1–5, and (3) what is still open. Nothing in the *content* of the concept pages was rewritten, only moved and re-linked.

## 1. Layout: before and after

| | Before | After (same as Tiers 1–5) |
|---|---|---|
| Location | `phase-2-transformers-foundation-models/tier-NN-*/` | `tier-NN-*/` at the repository root |
| Concept files | `tier-NN-*/concepts/NN.CC-slug.md` | `tier-NN-*/NN.CC-slug.md` |
| Figure scripts | `scripts/make_figures.py` | `figures/src/make_figures.py` (paths updated, regeneration tested on Tiers 6 and 15) |
| Tier overview | `README.md` (kept) | `README.md` (kept) + new `revision_sheet.md`, `requirements.txt`, `reviews/author_open_items.md` |
| Phase overview | `phase-2-.../README.md` | `PHASE_2_OVERVIEW.md` |
| Assets kept | `interactive/` (HTML explorers), `lab/` (code + tests), `figures/` (PNG + GIF) | unchanged (these are extra to Tiers 1–5 and stay) |

Tooling updated for the new layout: `tools/check_concepts.py`, `tools/concept_list.py`, `tools/mark_tier.py`, `tools/TIER_BRIEF.md`. All relative links were rewritten; a repository-wide link check finds **0 broken links**, and `check_concepts.py` reports OK for all ten tiers.

## 2. Inventory

| Tier | Topic | Concepts | Avg page KB | Figures | GIFs | Explorers | Lab tests | REVIEW_REQUIRED marks | Page code blocks run standalone |
|---|---|---|---|---|---|---|---|---|---|
| 06 | Tokenization & Embeddings | 32 | 5.2 | 7 | 2 | 6 | 14 | 49 | 31/32 |
| 07 | Attention | 30 | 4.9 | 10 | 3 | 5 | 23 | 44 | 30/33 |
| 08 | Transformer Architecture | 30 | 5.6 | 10 | 2 | 6 | 44 | 37 | 29/30 |
| 09 | Foundation Model Pretraining | 31 | 6.9 | 10 | 2 | 6 | 22 | 51 | 31/31 |
| 10 | Fine-Tuning & Alignment | 35 | 4.5 | 10 | 3 | 5 | 21 | 33 | 16/33 |
| 11 | LLM Inference & Optimization | 47 | 5.3 | 7 | 4 | 5 | 43 | 88 | 44/46 |
| 12 | Modern Foundation-Model Architectures | 30 | 4.2 | 9 | 2 | 5 | 29 | 49 | 30/30 |
| 13 | Vision AI | 32 | 5.2 | 9 | 3 | 5 | 21 | 41 | 32/32 |
| 14 | Speech, Audio & Video AI | 33 | 7.4 | 11 | 3 | 5 | 37 | 91 | 34/34 |
| 15 | Foundation-Model Evaluation & Safety | 35 | 5.4 | 8 | 2 | 5 | 40 | 24 | 35/35 |
| | **total** | **335** | | **91** | **26** | **53** | **294** | **507** | **312/336** |

Counts by script: `python tools/check_concepts.py tier-NN-*` (headings, metadata, links), `pytest --collect-only lab`, and a standalone run of every Python block in section 7 (*Code*) of every page.

## 3. Findings

**Good**
- Complete: all 335 roadmap concepts of Tiers 6–15 exist, one file each, all with the 23 standard sections in order and a metadata block (validated against the frozen roadmap).
- Substantial and consistent: pages average 3–7 KB, none empty; every tier has a README with index, figures (PNG/GIF), self-contained interactive explorers and a tested lab. No orphan figure or explorer files; no TODO/TBD markers.
- The labs are real: 294 tests pass in CI together with the rest of the repository.

**Differences from Tiers 1–5 (not yet harmonized, except Tier 6)**
- Update: **Tier 6 now has the learner-first layer** (see `tier-06-tokenization-embeddings/reviews/author_open_items.md`). Tiers 7–15 still do not.
- Pages (Tiers 7–15) use the original 23-section standard (uppercase headings, metadata list). They do **not** have the learner-first additions of Tiers 1–5 (plain-words box, glossary, three-step code ladder with outputs verified by `tests/test_md_snippets.py`), so a beginner entry point is missing. Tiers 6–15 are also harder than 1–5 and would benefit most from it.
- Page code is not executed by CI. A standalone run found 312 of 336 blocks runnable and 24 not:

  *Missing optional package or service (3):*
- `tier-06-tokenization-embeddings/06.14-sentencepiece.md`: ModuleNotFoundError: No module named 'sentencepiece'
- `tier-11-llm-inference-optimization/11.31-ollama.md`: urllib.error.HTTPError: HTTP Error 404: Not Found
- `tier-11-llm-inference-optimization/11.32-vllm.md`: ModuleNotFoundError: No module named 'openai'

  *Excerpts that need the surrounding lab code or earlier state (21):*
- `tier-07-attention/07.11-scaled-dot-product-attention.md`: NameError: name 'torch' is not defined
- `tier-07-attention/07.20-projection-matrices.md`: SyntaxError: 'return' outside function
- `tier-07-attention/07.22-gqa.md`: NameError: name 'self' is not defined
- `tier-08-transformer-architecture/08.25-decoder-only-architecture.md`: FileNotFoundError: [Errno 2] No such file or directory: 'outputs/train_log.json'
- `tier-10-fine-tuning-alignment/10.01-base-models-vs-instruction-tuned-models.md`: NameError: name 'base_model' is not defined
- `tier-10-fine-tuning-alignment/10.03-supervised-fine-tuning-sft.md`: NameError: name 'sft_example' is not defined
- `tier-10-fine-tuning-alignment/10.06-sft-data-curation.md`: NameError: name 'random' is not defined. Did you forget to import 'random'?
- `tier-10-fine-tuning-alignment/10.10-parameter-efficient-fine-tuning-peft.md`: NameError: name 'apply_lora' is not defined
- `tier-10-fine-tuning-alignment/10.11-lora.md`: NameError: name 'nn' is not defined
- `tier-10-fine-tuning-alignment/10.13-lora-alpha.md`: NameError: name 'LoRALinear' is not defined
- `tier-10-fine-tuning-alignment/10.14-lora-target-modules.md`: NameError: name 'apply_lora' is not defined
- `tier-10-fine-tuning-alignment/10.15-qlora.md`: NameError: name 'quantize_4bit' is not defined
- `tier-10-fine-tuning-alignment/10.16-adapters.md`: NameError: name 'nn' is not defined
- `tier-10-fine-tuning-alignment/10.17-prompt-tuning.md`: NameError: name 'nn' is not defined
- `tier-10-fine-tuning-alignment/10.22-rlhf.md`: NameError: name 'rm' is not defined
- `tier-10-fine-tuning-alignment/10.28-catastrophic-forgetting.md`: NameError: name 'base_batch' is not defined
- `tier-10-fine-tuning-alignment/10.29-catastrophic-forgetting-evaluation.md`: NameError: name 'before' is not defined
- `tier-10-fine-tuning-alignment/10.30-adapter-merging.md`: NameError: name 'merge_all' is not defined
- `tier-10-fine-tuning-alignment/10.31-model-merging.md`: NameError: name 'task_vector' is not defined
- `tier-10-fine-tuning-alignment/10.32-checkpoint-merging.md`: NameError: name 'average_states' is not defined
- `tier-10-fine-tuning-alignment/10.33-fine-tuning-evaluation.md`: NameError: name 'task_accuracy' is not defined

  Most failures are deliberate excerpts, but they are not marked as such; two (`07.11`, `07.20`) look like real omissions (missing import or a stray `return` at top level). `08.25` needs the lab's training output (`lab/train_mini_gpt.py`) to exist first.
- There is no `code/` library folder: the code lives in `lab/` next to its tests (kept as authored).
- Status words differ: tier READMEs say `AUTHOR_DRAFT`, `CHAPTER_STATUS.md` now says `DRAFT COMPLETE`.

**Overlap with Tiers 1–5 (duplicated or near-duplicated concepts, partly dictated by the frozen roadmap)**
- 7.9 Softmax ↔ 3.15; 8.10 Residual connections ↔ 3.27; 8.11 LayerNorm ↔ 3.31; 8.14 RMSNorm ↔ 3.32; 8.15 GELU ↔ 3.13; 15.9 Accuracy ↔ 2.37; 15.11 Calibration ↔ 2.43.
- Tier 6 embeddings ↔ 5.3–5.9; Tier 7 attention ↔ 5.24–5.27; Tier 10 optimizers/PEFT mentions ↔ 4.8–4.9; Tier 11 memory/quantization ↔ 4.24 and 4.41.
- Suggested rule: the Tier 1–5 page teaches the idea from scratch, the Tier 6–15 page cross-links to it and covers only what is specific to LLMs. This is not yet done; pages do not cross-reference the early tiers.

**Unverified**
- `REVIEW_REQUIRED` marks (507 in total) flag claims, attributions and numbers that still need a source check by the Critic.
- The interactive HTML explorers were not re-tested here (`tools/check_interactive.py` needs Playwright and a browser); they were smoke-tested by their author.
- GPU behaviour described in Tiers 9–12 (distributed pre-training, inference kernels, quantization speed-ups) cannot be exercised on CPU CI.

## 4. Recommended next steps (not done in this change)
1. Add the learner-first layer to Tiers 6–15 (plain-words box, glossary, executed code ladder), reusing the lab code, starting with 6, 7, 8.
2. Mark excerpts explicitly and make the standalone snippets of §3 run (or hide them from the snippet test list).
3. Add cross-links between the overlapping concepts and trim duplicated explanations.
4. Critic review of the `REVIEW_REQUIRED` marks; run the explorer smoke test in CI.
