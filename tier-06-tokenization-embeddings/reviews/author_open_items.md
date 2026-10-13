# tier-06-tokenization-embeddings — review notes (added during the Phase 2 restructuring)

These notes come from an automated structural review by Claude of the tier as authored on `chatgpt-bgc-1012`; they are not an author's report.

- **Structure:** 32 concept files `NN.CC-slug.md` directly in the tier folder; headings, metadata block and relative links validated by `python tools/check_concepts.py tier-06-tokenization-embeddings` (OK). Overview: `README.md`.
- **Assets:** 7 static figures, 2 GIFs, 6 interactive explorers (all referenced from the pages or README); figure generator `figures/src/make_figures.py`.
- **Code:** `lab/` with 14 collected tests (run in CI together with the rest of the repository).
- **Open markers:** 49 `REVIEW_REQUIRED` and 0 `NOT_TESTED` markers inside the concept files; they still need a source check by the Critic.
- **Format:** pages use the original 23-section standard (uppercase headings, metadata list). They do not yet have the learner-first additions of Tiers 1–5 (plain-words box, glossary, executed three-step code ladder with verified outputs); code blocks in the pages are not executed by `tests/test_md_snippets.py`.
