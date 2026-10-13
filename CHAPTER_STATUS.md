# CHAPTER STATUS

Source of truth for tier ownership and progress. Curriculum: `ROADMAP_V1.0_FROZEN.md` (FROZEN V1.0, 40 tiers, 7 phases).
Rules: `BOOK_SPECIFICATION.md`, `AUTHOR_REVIEW_GUIDE.md`, `GLOSSARY.md`.

## Pipeline per tier

Author → Critic → Code Review → Final Review → FINAL

A tier moves forward only when the previous stage is complete and its report is committed on the reviewer's branch.
A chapter cannot be FINAL until the Book Quality Standard checklist in `BOOK_SPECIFICATION.md` is fully ticked.

## Branches

| Agent | Role | Branch |
|---|---|---|
| Claude | Author / Critic | `bgc-10122026` |
| ChatGPT | Author / Critic / Final Review | `chatgpt-bgc-1012` |
| Codex | Code Review | `codex-bgc1012` |

Each agent commits only to its own branch. Merge into `main` only after Final Review passes.
## File layout (one file per concept)

Each tier gets its own folder, and **each concept gets its own `.md` file**. Never one file for a whole tier.

```
tier-NN-<slug>/
  NN.CC-<concept-slug>.md     one concept, full 23-part format (e.g. 01.10-dot-product.md)
  code/                       runnable code + tests for the tier
  reviews/                    critic.md, code_review.md, final_review.md, author_open_items.md
  revision_sheet.md           condensed sheet, written after the tier is complete
```

`CC` is the two-digit concept number from the roadmap, so files sort in curriculum order. Do not renumber concepts.

## Status values

`NOT STARTED` · `IN PROGRESS` · `DRAFT COMPLETE` · `CRITIC DONE` · `CODE REVIEW DONE` · `FINAL REVIEW DONE` · `FINAL`

Code results use only: `TESTED` · `PARTIALLY_TESTED` · `NOT_TESTED`.

## Tier assignments

Source column: **Reference** = the original table (Tiers 1–8); **Proposed** = assigned by Claude from the same pattern (Tiers 9–40), pending owner confirmation.
Rule used for proposals: author and critic are always different models (Claude ↔ ChatGPT); Codex reviews code for every tier; ChatGPT does Final Review. Claude authors hands-on engineering/agent/architecture tiers; ChatGPT authors model-theory, vision/audio, RAG, evaluation, cloud and data tiers.

| Tier | Phase | Topic | Concepts | Author | Critic | Code Review | Final Review | Status | Source |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 1 | Mathematics for AI | 47 | Claude | ChatGPT | Codex | ChatGPT | IN PROGRESS (1.1–1.19 drafted) | Reference |
| 02 | 1 | Machine Learning Fundamentals | 51 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Reference |
| 03 | 1 | Neural Networks | 32 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Reference |
| 04 | 1 | Deep Learning Training & Optimization | 41 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Reference |
| 05 | 1 | Sequence Models | 27 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Reference |
| 06 | 2 | Tokenization & Embeddings | 32 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Reference |
| 07 | 2 | Attention | 30 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Reference |
| 08 | 2 | Transformer Architecture | 30 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Reference |
| 09 | 2 | Foundation Model Pretraining | 31 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 10 | 2 | Fine-Tuning & Alignment | 35 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 11 | 2 | LLM Inference & Optimization | 47 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 12 | 2 | Modern Foundation-Model Architectures | 30 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 13 | 2 | Vision AI | 32 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 14 | 2 | Speech, Audio & Video AI | 33 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 15 | 2 | Foundation-Model Evaluation & Safety | 35 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 16 | 3 | Raw LLM Engineering & Multi-Provider Systems | 70 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 17 | 3 | Prompt & Context Engineering | 40 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 18 | 3 | Memory Systems | 32 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 19 | 3 | Structured Outputs & Tool Calling | 45 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 20 | 4 | RAG Fundamentals | 52 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 21 | 4 | Advanced RAG | 37 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 22 | 4 | GraphRAG & Agentic RAG | 33 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 23 | 4 | Multimodal AI & Multimodal RAG | 44 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 24 | 5 | LangChain & LlamaIndex | 52 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 25 | 5 | LangGraph & Stateful AI Workflows | 54 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 26 | 5 | AI Agents | 53 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 27 | 5 | Multi-Agent Systems | 54 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 28 | 5 | Model Context Protocol (MCP) | 51 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 29 | 5 | Agent-to-Agent Interoperability | 33 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 30 | 6 | AI System Evaluation | 56 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 31 | 6 | AI Observability & LLMOps | 40 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 32 | 6 | AI Security, Guardrails & Governance | 50 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 33 | 6 | Production AI Backend Engineering | 43 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 34 | 6 | Containers, Testing, CI/CD & Infrastructure as Code | 46 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 35 | 6 | Cloud AI Engineering | 19 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 36 | 6 | AI Performance, Scalability, Reliability & Cost Engineering | 44 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 37 | 7 | AI System Design | 44 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 38 | 7 | AI Data Engineering | 50 | ChatGPT | Claude | Codex | ChatGPT | NOT STARTED | Proposed |
| 39 | 7 | Enterprise AI Architecture & Design Patterns | 38 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |
| 40 | 7 | Enterprise Capstones & Interview Mastery | 20 | Claude | ChatGPT | Codex | ChatGPT | NOT STARTED | Proposed |

## Progress notes

- Tier 1: concepts 1.1–1.19 drafted as files `tier-01-mathematics/01.01-scalars.md` … `01.19-singular-value-decomposition.md`; 30 tests passing in `code/`. Remaining: 1.20–1.47, then Tier 1 lab and condensed revision sheet.
- Open proposal: add "number of axes" vs "vector dimension" to `GLOSSARY.md` (needs Critic decision).
- `AUTHOR_REVIEW_GUIDE.md` refers to `ROADMAP.md`; the file is `ROADMAP_V1.0_FROZEN.md`.
