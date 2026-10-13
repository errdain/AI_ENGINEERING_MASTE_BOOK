# Brief for authoring one tier (visual edition)

Repo: /home/user/AI_ENGINEERING_MASTE_BOOK. Read first: BOOK_SPECIFICATION.md, AUTHOR_REVIEW_GUIDE.md, GLOSSARY.md,
the roadmap section for YOUR tier in ROADMAP_V1.0_FROZEN.md (do NOT change/renumber concepts), and the finished template
phase-2-transformers-foundation-models/tier-06-tokenization-embeddings/ (README.md, scripts/make_figures.py, interactive/*.html, lab/).
Match that template's structure, tone, depth and quality.

Deliver ONLY inside your tier folder: phase-2-transformers-foundation-models/tier-XX-<slug>/
  README.md            chapter: header+status, "how to use" table (read/see/play/build), big picture, concept map table
                       (EVERY roadmap item -> section), concept sections following the 23-section standard (compress for small
                       concepts, full for major ones: what/why/how/example/code/enterprise/AI-connection/trade-offs/mistakes/
                       cost/security/interview 30-60s/rapid-fire/questions/exercise/memory hook), build section, interview kit,
                       revision questions, related concepts, review-status table.
  figures/             labelled PNG + animated GIF (matplotlib FuncAnimation+PillowWriter) -- aim for 6-9 figures, >=2 GIFs, each
                       embedded in README with a caption saying what to look at. Titles/axes/legends must be labelled. No manim.
  interactive/         3-5 self-contained HTML explorers (vanilla JS, canvas/SVG, NO CDN/network), with ADJUSTABLE sliders/inputs,
                       light+dark CSS vars, responsive, copy the CSS pattern from tier-06. Each linked from README with its controls.
  lab/                 the roadmap "Build" item as runnable Python (numpy and/or torch CPU, both installed) + pytest tests that pass.
  scripts/make_figures.py   regenerates every figure.
Rules: Python 3.13; keep lab runs fast (<60s total). Never claim TESTED unless you ran it. Mark unverifiable facts/numbers
(model sizes, benchmark numbers, fast-moving APIs) REVIEW_REQUIRED. Be accurate; no invented facts. One authoritative teaching
location per concept (reference other tiers instead of re-teaching; Tier 7 owns attention math, Tier 8 owns transformer blocks).
Do not touch other tiers' folders or top-level files. Do NOT run git commands (the parent commits).
Verify: run pytest in lab/; run `python tools/check_interactive.py <your interactive dir> --shots /tmp/shots_XX` (must report ok)
and LOOK at 2-3 screenshots + 2-3 figures (Read the png) to fix overlapping/unreadable labels; run make_figures.py.
Keep each GIF < 600KB, PNG < 250KB (use fewer frames / lower dpi). Final reply: <=10 lines: files made, test results, caveats.
