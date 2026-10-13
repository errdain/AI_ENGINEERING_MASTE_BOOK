# Tier 1 code-review preflight

This is preparation, not the formal code-review report or stage approval.

## Assignment and prerequisite

Codex performs code review on `codex-bgc1012`, per `CHAPTER_STATUS.md`.
Read: BOOK_SPECIFICATION.md, AUTHOR_REVIEW_GUIDE.md,
ROADMAP_V1.0_FROZEN.md, GLOSSARY.md, and CHAPTER_STATUS.md.
There is no preceding tier summary for Tier 1.

The tracker requires Author → Critic → Code Review → Final Review.
Tier 1 is still IN PROGRESS; the fetched critic branch contains no critic
report. Do not advance the tracker or claim CODE REVIEW DONE.

## Snapshot

- Main: `883c33cda1646da075be348cc5a131a819a5aa41`.
- Author branch: `04dc7bb2a3439fa931fd3704a69287ea0de28829`.
- Critic branch: `e6cc5a6d62ea723e9d8abee2509bb3c7f46bb069`.
- Executed code: main's `tier-01-mathematics/code/`.
- The author branch has identical code files at this snapshot and has already
  split concepts into `01.01-scalars.md` through `01.10-dot-product.md`.
- The author branch and main diverge; no merge of unreviewed author text was
  made. The final review intake must use the exact critic-reviewed revision.

## Preliminary execution

Environment: Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, isolated workspace venv.

From the repository root, using a Python environment with these dependencies:

```sh
python -m pytest -q tier-01-mathematics/code
python tier-01-mathematics/code/linalg_basics.py
```

TESTED: existing suite, 11 passed. Demo executed successfully:

- Addition: `[5, 7, 9]` in manual and NumPy versions.
- Scaling: `[2, 4, 6]` in both versions.
- Dot product: 32 in both versions.
- Tensor: 3 axes, shape `(32, 128, 768)`, size 3145728, float64.
- Broadcast result: `(32, 128, 768)`.

Overall manuscript code status: PARTIALLY_TESTED. This run covers the existing
code files only, not all Markdown snippets or all edge cases. No external API
or secret handling is exercised by the demo. PyTorch has not been installed
or tested in this environment.

## Intake checklist when critic stage completes

1. Fetch the completed author material and committed critic report.
2. Read each concept file and record the reviewed commit and outstanding findings.
3. Preserve one Markdown file per concept in the tier folder.
4. Execute applicable Markdown examples as well as standalone code.
5. Expand checks for empty/ragged matrices, invalid dimensions, zero-length
   axes, broadcasting failures, numerical edge cases, and tensor shapes.
6. Verify applicable PyTorch examples and version-sensitive claims.
7. Write `reviews/code_review.md` with evidence, severities, and explicit
   TESTED / PARTIALLY_TESTED / NOT_TESTED scope.
8. Advance status only if the required review stage is actually complete.

Earlier preliminary Codex authoring is preserved in a named Git stash and
is not applied over the assigned author's work.
