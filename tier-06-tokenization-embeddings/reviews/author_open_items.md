# tier-06-tokenization-embeddings — review notes (added during the Phase 2 restructuring)

These notes come from an automated structural review by Claude of the tier as authored on `chatgpt-bgc-1012`; they are not an author's report.

- **Structure:** 32 concept files `NN.CC-slug.md` directly in the tier folder; headings, metadata block and relative links validated by `python tools/check_concepts.py tier-06-tokenization-embeddings` (OK). Overview: `README.md`.
- **Assets:** 7 static figures, 2 GIFs, 6 interactive explorers (all referenced from the pages or README); figure generator `figures/src/make_figures.py`.
- **Code:** `lab/` with 14 collected tests (run in CI together with the rest of the repository).
- **Open markers:** 49 `REVIEW_REQUIRED` and 0 `NOT_TESTED` markers inside the concept files; they still need a source check by the Critic.
- **Format:** the learner-first layer was added by Claude in a later change: each page now has the level/reading-time line, a plain-words box, "You need first", a figure (existing PNG/GIF or a Mermaid diagram) with a *How to read it* note, a glossary table, a worked "By hand" example, an executed code ladder (① plain Python, ② NumPy or PyTorch, ③ PyTorch or the tier library) whose outputs are verified by `tests/test_md_snippets.py`, an exercise solution, sections 9–16 collapsed, and reviewer notes collapsed. The original ChatGPT text of sections 1–23 is kept; the original code excerpts are kept as *Lab excerpt* (not executed).
