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

## Batch 2 (2.14–2.25)

- TESTED: `lab/lab_02_models_and_training.py` (10 tests in `lab/test_lab_02_models_and_training.py`). Silhouette implementation checked against scikit-learn. Whole tier so far: 32 `code/` + 19 `lab/` (9 + 10) = 51 tests.
- Numbers in the text come from fixed seeds on NumPy 2.5.4; tests assert directions/thresholds.
- DUPLICATION to adjudicate (roadmap lists both): SGD = 2.20 and 4.6; batch size = 2.21/2.23 and 4.13; training loop vocabulary (epoch/iteration) vs 4.1; loss definition 2.17 vs 4.3; learning rate 2.25 vs LR scheduling 4.10. Author handling: Tier 2 is the conceptual and NumPy-mechanics authority; each file points to the Tier 4 concept for PyTorch/engineering detail.
- TIER 1 CROSS-REFERENCES to fix after Critic review: Tier 1 files say "Tier 4 Loss functions" and "Tier 4 SGD/Optimizers"; the loss definition is authoritative in 2.17 (Tier 4 only has 4.3 Loss calculation). Also see the PROPOSE_ADDITION above.
- REVIEW_REQUIRED: 2.18 claim that weight decay and L2 regularization differ under adaptive optimizers (AdamW); 2.21 batch-size/learning-rate scaling heuristics and large-batch generalization statement; 2.22 "1–5 epochs for fine-tuning" rule of thumb and "LLM pretraining often <1 epoch" statement; 2.20 Robbins–Monro conditions are stated without derivation.
- NOT_TESTED: the exercises in 2.14–2.25 (quantile loss, concentric rings, throughput vs batch size, LR-range test) are described, not shipped as code.
- Lab limitation: the multiclass model (`fit_softmax_regression`) lives in the lab file, not `mlkit.py`; binary logistic regression is the library-verified model.

## Batch 3 (2.26–2.36)

- TESTED: `lab/lab_03_generalization.py` (14 test cases in `lab/test_lab_03_generalization.py`, incl. lasso vs scikit-learn). Whole tier: 32 `code/` + 33 `lab/` (9 + 10 + 14) = 65 tests.
- Demonstrations changed during authoring because the first version did not show the claimed effect (kept honest, documented): (a) gradient-descent early stopping on a degree-9 polynomial showed no validation minimum, so the early-stopping demo uses a 60-feature/30-sample linear problem; (b) bias–variance with *random* training inputs showed degree-1 variance above degree-3 (extra variance from sampling x), so `bias_variance_poly` now defaults to a fixed design and 2.29 explains the difference; (c) a first temporal-leakage demo (random walk + linear fit) showed no effect, replaced by 1-NN on the time index.
- Early-stopping demo effect is small (≈2–15% validation increase depending on seed); the test asserts ≥2% for four seeds.
- TERMINOLOGY to approve in `GLOSSARY.md` (not inserted): "bias" has five meanings (2.28), "variance" (model) vs variance (1.30) (2.29), "data leakage" vs "overfitting", "generalization gap", "domain classifier", PSI.
- DUPLICATION: Dropout/weight decay (3.28/3.29), AdamW (4.9), drift monitoring (31), benchmark contamination (15/30) are taught in later tiers; Tier 2 gives the concept and points forward.
- REVIEW_REQUIRED: 2.27/2.30 double-descent / benign overfitting statements (need citations); 2.31 statistical-learning-theory bounds statement; 2.32 weight decay vs L2 under Adam; 2.34 PSI thresholds (convention); 2.36 Bayesian optimization / Hyperband descriptions (no code here); 2.35 statement that one or two hyperparameters usually dominate (general observation, cite Bergstra & Bengio before FINAL).
- NOT_TESTED: nested cross-validation is described in 2.36 but not implemented; importance weighting in 2.34's exercise; exercises in 2.26–2.36 are described, not shipped.

## Batch 4 (2.37–2.51)

- TESTED: `code/mlkit.py` extended (multiclass confusion matrix, macro/micro F1, average precision, time-series splits; 35 tests incl. scikit-learn comparisons) and `lab/lab_04_metrics_and_data.py` (15 tests). Whole tier: 35 `code/` + 48 `lab/` (9 + 10 + 14 + 15) → run `python -m pytest -q code lab` for the current total.
- DUPLICATION in the frozen roadmap: **2.45 Class imbalance vs 2.50 Dataset imbalance**. Author split: 2.45 = training methods and metrics for rare classes; 2.50 = dataset-level aspects (evaluation-set construction, prevalence vs production, downsampling and prior correction, calibration). Critic should confirm or request merge via PROPOSE_ADDITION. Also overlapping: 2.47 Feature scaling vs Tier 3 normalization layers (3.30–3.32); 2.48 vs Tier 6 embeddings; 2.43 Calibration vs Tier 15/30; 2.46 baselines vs Tier 30.
- Honest findings during authoring (kept in the text): class weights and oversampling *reduced* F1 at the default threshold while leaving PR-AUC unchanged (2.45); a fitted linear trend was worse than predicting the mean on the forecasting example and the naive forecast won (2.46); the plain and class-weighted models tie once the threshold is tuned to the cost ratio (2.51).
- A one-line outlier-clipping effect in 2.49 (78.4% → 81.9%) depends on the clip percentiles (5–95); with 1–99 the effect disappeared in the first run.
- REVIEW_REQUIRED: 2.44 statement that CV folds' errors are correlated so naive standard errors understate uncertainty (standard result, needs citation); 2.41 AUC interpretation (Mann–Whitney equivalence is standard); 2.43 claims on LLM probability calibration; 2.51 metric table (NDCG and MAPE are mentioned but not implemented); 2.47 skewness/log claim is shown for one lognormal sample.
- NOT_TESTED: nested CV, isotonic regression, SMOTE, Hyperband, NDCG, precision@k/recall@k (described only); exercises in 2.37–2.51.
- SECURITY/PRIVACY statements throughout Tier 2 are general guidance, not verified against a specific framework.
- Tier 1 cross-references to fix after the Critic stage: "Tier 2 PCA / Naive Bayes / Decision trees / Regression" (see PROPOSE_ADDITION), "Tier 4 Loss functions" → 2.17.
