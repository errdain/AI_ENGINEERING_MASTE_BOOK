# Phase 2 — Transformers & Foundation Models

Every tier folder contains:

| Folder/file | Purpose |
|---|---|
| `README.md` | the chapter (theory → mechanics → code → enterprise → interview) |
| `figures/` | labelled PNGs and animated GIFs (regenerate with `scripts/make_figures.py`) |
| `interactive/` | self-contained HTML explorers with sliders — open in any browser, no install |
| `lab/` | runnable Python + pytest tests for the tier's "Build" item |
| `scripts/` | figure generators |

| Tier | Topic | Status |
|---|---|---|
| 06 | [Tokenization & Embeddings](tier-06-tokenization-embeddings/README.md) | AUTHOR_DRAFT |
| 07 | [Attention](tier-07-attention/README.md) | AUTHOR_DRAFT |
| 08 | [Transformer Architecture](tier-08-transformer-architecture/README.md) | AUTHOR_DRAFT |
| 09 | [Foundation Model Pretraining](tier-09-foundation-model-pretraining/README.md) | AUTHOR_DRAFT |
| 10 | [Fine-Tuning & Alignment](tier-10-fine-tuning-alignment/README.md) | AUTHOR_DRAFT |
| 11 | [LLM Inference & Optimization](tier-11-llm-inference-optimization/README.md) | AUTHOR_DRAFT |
| 12 | [Modern Foundation-Model Architectures](tier-12-modern-architectures/README.md) | AUTHOR_DRAFT |
| 13 | [Vision AI](tier-13-vision-ai/README.md) | AUTHOR_DRAFT |

Smoke-test all interactive pages: `python tools/check_interactive.py phase-2-transformers-foundation-models`.
