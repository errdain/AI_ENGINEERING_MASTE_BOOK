# Brief for authoring one tier (visual edition)

Repo: /home/user/AI_ENGINEERING_MASTE_BOOK. Read first: BOOK_SPECIFICATION.md, AUTHOR_REVIEW_GUIDE.md, GLOSSARY.md,
the roadmap section for YOUR tier in ROADMAP_V1.0_FROZEN.md (do NOT change/renumber concepts), and the finished template
tier-06-tokenization-embeddings/ (README.md, NN.CC-*.md concept files, figures/src/make_figures.py, interactive/*.html, lab/).
Match that template's structure, tone, depth and quality.

Deliver ONLY inside your tier folder: tier-XX-<slug>/ at the repository root (concept files NN.CC-slug.md directly inside; figure scripts in figures/src/)
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


# >>> OVERRIDING FORMAT RULE (from the book owner; supersedes the README.md bullet above) <<<
The owner requires EACH roadmap concept to have its OWN .md file, strictly following BOOK_SPECIFICATION.md
("STANDARD CONCEPT FORMAT", "DEPTH RULE", "CODE RULES", "ACCURACY RULE", "DUPLICATION RULE"). Do NOT put concept teaching in one unified file.
* Canonical list + filenames: `python tools/concept_list.py <tier>` -> `concepts/{TT}.{NN}-{slug}.md` (use EXACTLY these names; one file per roadmap item, none skipped, none added/renumbered).
* Each concept file = exactly:
  `## X.Y Concept Name`, then `Metadata:` (Tier, Phase, Priority, Difficulty, Interview Importance, Hands-on Importance, Prerequisites, Leads To; prerequisites/leads-to are links to other concept files or tier folders),
  then the 23 sections with EXACTLY these headings, in order: `### 1. WHAT IS IT?` `### 2. WHY DOES IT EXIST?` `### 3. DETAILED EXPLANATION` `### 4. HOW DOES IT WORK?` `### 5. SIMPLE EXAMPLE` `### 6. TECHNICAL EXAMPLE` `### 7. CODE` `### 8. INPUT → PROCESS → OUTPUT` `### 9. ENTERPRISE USE CASE` `### 10. AI ENGINEERING CONNECTION` `### 11. ADVANTAGES` `### 12. LIMITATIONS` `### 13. TRADE-OFFS` `### 14. COMMON MISTAKES` `### 15. PERFORMANCE / COST CONSIDERATIONS` `### 16. SECURITY CONSIDERATIONS` `### 17. INTERVIEW EXPLANATION` `### 18. RAPID-FIRE ANSWER` `### 19. INTERVIEW QUESTIONS` (with Basic / Intermediate / Advanced / Scenario) `### 20. HANDS-ON EXERCISE` `### 21. REVISION QUESTIONS` `### 22. RELATED CONCEPTS` `### 23. ONE-LINE MEMORY HOOK`.
  Per the DEPTH RULE do NOT pad: tiny concepts are short (a sentence or a few lines per section; where a section truly does not apply write one line saying why, never delete the heading). Major concepts get full depth (the ones the old README treated in depth).
* VISUALS ARE REQUIRED IN CONCEPT FILES: embed the relevant labelled figure/GIF (`![caption](../figures/xx.png)`) and link the relevant interactive page (`[explore](../interactive/xx.html)`) and lab code (`../lab/xx.py`) in the concept file where it teaches that concept (sections 3/4/7). Concepts with no suited figure may use an ASCII/Mermaid diagram in section 8. Add NEW figures/pages if an important concept has no visual.
* Tier `README.md` becomes the TIER OVERVIEW only: goal, prerequisites/leads-to, big picture, how-to-use table, visual gallery (all figures/GIFs embedded), interactive explorer table, concept index table (every concept -> link to its file), the Build/lab description + results, tier-level interview kit, tier summary, review-status table. NO per-concept teaching there.
* Write the concept files in several batches; keep each coherent. Reuse (and fix/extend) content from any existing tier README/lab; keep facts consistent with the lab code. Mark uncertain facts REVIEW_REQUIRED. Code snippets in concept files must be correct (run non-trivial ones; keep them consistent with lab/).
* Finally run `python tools/check_concepts.py <tier-folder>` (validates filenames vs roadmap, headings, and relative links/images) and fix every error before reporting.
