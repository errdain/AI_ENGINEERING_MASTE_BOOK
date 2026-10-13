# Tier 2 — author open items

## PROPOSE_ADDITION (not inserted; needs Critic / owner decision)

Tier 1 concept files (written earlier) cross-reference Tier 2 for topics that are **not concepts in the frozen roadmap's Tier 2 list (2.1–2.51)**:

| Referenced from Tier 1 | Where it appears | Status in `ROADMAP_V1.0_FROZEN.md` |
|---|---|---|
| PCA | 1.18, 1.19, 1.32 ("Tier 2 PCA") | not a listed concept anywhere (grep: no "PCA") |
| Naive Bayes | 1.34 | not listed |
| Decision trees | 1.36 | not listed |
| Linear regression / logistic regression | 1.14, 1.45, 1.35 | not listed as concepts; only "Regression" (2.15) and "Classification" (2.14) |
| Feature scaling | 1.31 | listed (2.47) ✔ |
| Calibration | 1.26, 1.34 | listed (2.43) ✔ |

Reason: the cross-references would point readers to chapters that do not exist.
Affected tiers: 1 and 2 (possibly 38 for dimensionality reduction).
Dependency impact: PCA and tree/NB models are used in later tiers only as background; no later tier depends on them as prerequisites.
Duplication impact: none if taught inside 2.14–2.16 / 2.48 as worked examples.
Terminology impact: none.
Author's handling for now: Tier 2 uses linear regression, logistic regression and k-means as **worked examples inside 2.14, 2.15, 2.16** (they are the implementations in `code/mlkit.py`), and does **not** add PCA, Naive Bayes or decision trees as concepts. The Tier 1 wording "Tier 2 PCA / Naive Bayes / Decision trees" should be changed to "see 2.x" or removed after the owner decides: (a) approve adding concepts (roadmap V1.1), or (b) keep frozen and rewrite the Tier 1 references.

## Batch 1 (2.1–2.13)

- TESTED: `code/mlkit.py` (32 tests, scikit-learn 1.9.1 used as independent reference for linear/ridge/logistic regression, confusion matrix, precision/recall/F1/F-beta, ROC/AUC, Brier, KS), `lab/lab_01_paradigms_and_data.py` (9 tests). NumPy 2.5.4, Python 3.14.
- Several demonstration numbers in the concept files come from fixed seeds. Tests assert directions and thresholds, not the exact digits. Digits quoted in the text were produced by running the lab on this environment.
- Semi-supervised gain in 2.4 is small (96.0% → 97.25%) and seed-dependent: the text says so.
- REVIEW_REQUIRED: 2.8 mentions "agreement measures" without computing one (no Cohen's kappa implemented); decide whether to add to the lab.
- REVIEW_REQUIRED: 2.5/2.6 statements about BERT-style masked LMs, RLHF and 'most LLM agents are not trained with RL' are general background claims without citations.
- REVIEW_REQUIRED: 2.13 'benchmark contamination' paragraph needs a citation.
- Forward references to later tiers: Tier 3, 6, 8, 9, 10, 13, 14, 15, 19, 21, 26, 30, 32, 38.
