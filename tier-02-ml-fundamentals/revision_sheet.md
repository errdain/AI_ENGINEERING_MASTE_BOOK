# Tier 2 — Machine Learning Fundamentals: Revision Sheet

> Status: DRAFT (Author: Claude). Condensed from concept files `02.01`–`02.51`. The concept file is the authoritative source; this sheet only summarises it. Rapid-fire answers and memory hooks in section 2 are copied from sections 18 and 23 of each concept file.
> Code evidence: `code/mlkit.py` (35 tests) and `lab/` (48 tests). Run: `cd tier-02-ml-fundamentals && python -m pytest -q code lab` (83 tests; scikit-learn is used only as a test reference).

## 1. One-page formula and rule sheet

| Topic | Formula / rule | Note |
|---|---|---|
| Pipeline | split → fit preprocessing on train → fit model → tune on validation → test once | any statistic computed from test rows is leakage (2.33) |
| Roles of data | train fits parameters; validation picks hyperparameters/early-stop; test estimates final performance once | peeking: best of 200 coin-flip models scores 62% on test, 47.7% on fresh data |
| Split unit | split by what must generalize: user/group, time, stratified class | random row split on grouped data: 99.5% vs 44% (group split) |
| Gradient descent | `θ ← θ − η∇J(θ)` | full batch, mini-batch (B), pure SGD (B=1) |
| Learning-rate limit | quadratic: stable iff `η < 2/λ_max` | one feature ×30 → limit ÷840; standardize (2.47) |
| Epoch / iteration | iterations per epoch `= ⌈n/B⌉` (`⌊n/B⌋` with drop_last); steps = epochs × that | n=1050, B=100, 3 epochs → 33 |
| Mini-batch gradient | unbiased; variance ∝ `1/B` | constant η leaves a noise floor; decay η |
| Objective | `J = (1/n)Σℓ + λR(θ)` | loss ≠ objective ≠ metric |
| Losses | MSE (Gaussian MLE), MAE (robust), Huber, BCE/cross-entropy from logits | MSE on sigmoid saturates (gradient ~200× smaller when confidently wrong) |
| Linear regression | `ŷ = w·x + b`; solve with `lstsq`; `R² = 1 − SSE/SST` | R² ≤ 0 = no better than the mean |
| Ridge / lasso | `+λ‖w‖₂²` shrinks; `+α‖w‖₁` zeroes weights | scale features first; never penalize the intercept |
| Bias–variance | `E[err] = Bias² + Variance + noise` | variance ≈ `σ²p/n` for least squares with fixed design |
| Under/overfit | underfit: train & val both bad; overfit: train ↓, val ↑ | degree-15 poly: train 0.016, val 23.5 |
| Generalization gap | `test error − train error` | shrinks ~ with n; random labels: 100% train, 51% test (1-NN) |
| Accuracy | `(TP+TN)/N` | 98.9% for "always negative" at 1% prevalence |
| Precision / recall | `TP/(TP+FP)`, `TP/(TP+FN)` | PPV `= TPR·π/(TPR·π + FPR(1−π))` |
| F1 / F-beta | `2TP/(2TP+FP+FN)`; `Fβ = (1+β²)PR/(β²P+R)` | harmonic mean: P=0.9, R=0.1 → 0.18 |
| ROC-AUC | `P(score_pos > score_neg)` | 0.5 chance, rank-only; use average precision (PR-AUC) for rare positives |
| Calibration | predicted p ≈ observed frequency | ECE, Brier; fix with temperature/Platt on held-out data; AUC unchanged |
| Cross-validation | k folds; stratified / grouped / expanding-window | everything fitted inside the fold; nested CV for honest tuning |
| Imbalance | weights/resampling move the threshold, not the ranking | PR-AUC unchanged (≈0.81) while F1@0.5 fell 0.71 → 0.35 |
| Prior correction | `logit += log(π_prod/(1−π_prod)) − log(π_train/(1−π_train))` | ECE 0.108 → 0.001 after 50/50 downsampling |
| Scaling | `(x−μ)/σ`, min–max, `(x−median)/IQR`; `log` for skew | fit on train only; trees and OLS predictions don't need it |
| Distribution shift | covariate `P(x)`, label `P(y)`, concept `P(y\|x)` | PSI <0.1 stable, >0.25 large (convention); domain-classifier AUC |
| Hyperparameter search | random > grid when few params matter; log scale for rates | best-of-many is optimistic: 62.5% CV vs 48.9% fresh |
| Cost-based threshold | calibrated p: predict positive if `p > c_FP/(c_FP + c_FN)` | cost ratio, not 0.5 |

## 2. Concept-by-concept revision

### A. Paradigms, data and splits (2.1–2.13)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 2.1 | [Artificial Intelligence vs Machine Learning vs Deep Learning](02.01-ai-vs-ml-vs-dl.md) | AI ⊃ ML ⊃ DL: AI = intelligent behaviour, ML = learned from data, DL = learned with deep neural nets. | Rules write behaviour by hand; ML fits it from examples; DL also learns the features. |
| 2.2 | [Supervised learning](02.02-supervised-learning.md) | Learn f(x)→y from labeled pairs by minimizing a loss; evaluate on unseen data. | Supervised: learn from answered questions, then take the exam on new ones. |
| 2.3 | [Unsupervised learning](02.03-unsupervised-learning.md) | Structure discovery without labels: clustering, density, representation; validate by usefulness. | Unsupervised: find the shape of the data when nobody gave you the answers. |
| 2.4 | [Semi-supervised learning](02.04-semi-supervised-learning.md) | Few labels + many unlabeled; self-training with confident pseudo-labels; risk = confirmation bias. | Teach yourself from the cases you're sure about — and keep an outside examiner. |
| 2.5 | [Self-supervised learning](02.05-self-supervised-learning.md) | Labels come from the data itself (next/masked token); trained with cross-entropy; basis of foundation models. | Self-supervised: hide a piece, guess it, and let the data grade itself. |
| 2.6 | [Reinforcement learning overview](02.06-reinforcement-learning-overview.md) | Agent–environment loop maximizing cumulative reward; exploration vs exploitation; RLHF adapts LLMs. | RL: act, get scored, adjust — and keep exploring. |
| 2.7 | [Features](02.07-features.md) | Features = model inputs (columns of X); relevance and prediction-time availability matter most. | The model only knows what the features tell it. |
| 2.8 | [Labels](02.08-labels.md) | Labels = targets y; quality caps performance; watch noise, proxies and delay. | Garbage labels in, confident garbage out. |
| 2.9 | [Samples](02.09-samples.md) | Sample = one (x, y) row; representativeness and independence beat raw count. | Quantity is a start; representativeness is the point. |
| 2.10 | [Dataset construction](02.10-dataset-construction.md) | Define → collect → clean → dedupe → label → split by group/time → document & version. | Split by the thing you must generalize to. |
| 2.11 | [Training dataset](02.11-training-dataset.md) | Training set fits parameters and preprocessing statistics; never used to judge generalization. | Everything learned comes from the training set — and only from it. |
| 2.12 | [Validation dataset](02.12-validation-dataset.md) | Validation = for tuning and selection; reuse makes it optimistic; final estimate uses the test set. | Validation steers; test judges. |
| 2.13 | [Test dataset](02.13-test-dataset.md) | Use once at the end; peeking inflates; size sets precision; keep a private set. | Test set: open the envelope once. |

### B. Task types and training mechanics (2.14–2.25)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 2.14 | [Classification](02.14-classification.md) | Predict a discrete class via logits → softmax/sigmoid; train with cross-entropy; set the threshold by cost. | Classification: scores in, probabilities out, threshold chosen by cost. |
| 2.15 | [Regression](02.15-regression.md) | Predict a number; least squares = MLE under Gaussian noise; R² compares to the mean baseline. | Regression: fit the best line (or curve) and always compare with 'just predict the mean'. |
| 2.16 | [Clustering](02.16-clustering.md) | k-means minimizes within-cluster squared distance; scale features, use multiple inits, choose k by silhouette/stability. | k-means: scale first, restart often, and treat clusters as hypotheses. |
| 2.17 | [Loss functions](02.17-loss-functions.md) | Loss = per-sample error score; MSE/MAE/Huber for regression; cross-entropy for classification; not the same as the metric. | The loss is the teacher's grading rule — students optimize exactly what it rewards. |
| 2.18 | [Objective functions](02.18-objective-functions.md) | Objective = data loss + penalties; differs from the evaluation metric; λ trades fit for simplicity. | Objective = what you minimize; metric = what you report. Keep them distinct. |
| 2.19 | [Gradient descent](02.19-gradient-descent.md) | θ ← θ − η∇J; converges to least squares on linear regression; speed depends on conditioning and η. | Look at the slope, step the other way, repeat. |
| 2.20 | [Stochastic gradient descent (SGD)](02.20-stochastic-gradient-descent.md) | Noisy unbiased gradient from random samples; cheap steps; needs decaying LR to converge exactly. | SGD: a cheap, noisy compass that still points downhill on average. |
| 2.21 | [Mini-batch training](02.21-mini-batch-training.md) | Average gradient over B samples: variance ~1/B, parallel efficient; retune LR when B changes. | Mini-batch: the Goldilocks gradient — not too noisy, not too slow. |
| 2.22 | [Epoch](02.22-epoch.md) | Epoch = one full pass over the training set; number of epochs is a tuned hyperparameter. | One epoch = every sample seen once. |
| 2.23 | [Batch](02.23-batch.md) | Batch = samples processed together per step; global batch = micro × accumulation × replicas. | Batch: the tray you push through the oven together. |
| 2.24 | [Iteration](02.24-iteration.md) | Iteration = one optimizer update; steps/epoch = ⌈n/B⌉; schedules use global steps. | An iteration is one update — count the steps, not just the epochs. |
| 2.25 | [Learning rate](02.25-learning-rate.md) | η scales each step; stable iff η < 2/λ_max (quadratic); sweep on a log scale; retune with batch size. | Learning rate: big enough to move, small enough not to explode — and stay under 2/λ_max. |

### C. Generalization, leakage, shift and tuning (2.26–2.36)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 2.26 | [Underfitting](02.26-underfitting.md) | Underfit = high train and validation error (high bias); fix capacity/features/training, not data volume. | Underfit: bad on the homework and bad on the exam — the model is too simple. |
| 2.27 | [Overfitting](02.27-overfitting.md) | Overfit = train error ↓, validation error ↑; fix with data, simpler model, regularization, early stopping. | Overfit: a perfect memory and a poor understanding. |
| 2.28 | [Bias](02.28-bias.md) | Bias = systematic error of the model class (underfitting); distinct from dataset, fairness and inductive bias. | Bias: the same miss in the same direction, however many times you throw. |
| 2.29 | [Variance](02.29-variance.md) | Model variance = prediction spread across training sets; grows with flexibility, shrinks ~1/n with data. | Variance: how much the model changes its mind when the sample changes. |
| 2.30 | [Bias–variance trade-off](02.30-bias-variance-trade-off.md) | Error = bias² + variance + noise; complexity trades bias for variance; validate to find the sweet spot. | Pick the complexity where bias and variance cross — and respect the noise floor. |
| 2.31 | [Generalization](02.31-generalization.md) | Performance on unseen data; gap = test − train error; estimate on independent data matching deployment. | Generalization: can it do the new exam, not just recite the old one. |
| 2.32 | [Regularization](02.32-regularization.md) | Constrain complexity: L2 shrinks, L1 sparsifies; tune λ on validation; scale features first. | Regularize: pay a little bias to buy a lot of stability. |
| 2.33 | [Data leakage](02.33-data-leakage.md) | Information from outside the training/prediction-time boundary inflates scores; split first, fit inside folds, split by time/group. | If the model could not know it at prediction time, it must not know it at training time. |
| 2.34 | [Distribution shift](02.34-distribution-shift.md) | Train ≠ production distribution: covariate, label, concept shift; detect with PSI/KS/domain classifier; retrain or recalibrate. | The world moves; the model doesn't — so measure the gap. |
| 2.35 | [Hyperparameters](02.35-hyperparameters.md) | Pre-set controls (lr, batch, λ, depth, threshold) tuned on validation; parameters are learned. | Hyperparameters: the dials you set; parameters: the numbers the model learns. |
| 2.36 | [Hyperparameter tuning](02.36-hyperparameter-tuning.md) | Random > grid when few parameters matter; tune on validation; the best-of-many is optimistic, so confirm on a held-out test. | Tune on validation, confirm on test — the winner of many trials is always a little lucky. |

### D. Metrics and evaluation (2.37–2.44)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 2.37 | [Accuracy](02.37-accuracy.md) | Accuracy = correct / total; misleading under imbalance; compare to the majority baseline. | High accuracy can be a model that never finds the thing you're looking for. |
| 2.38 | [Precision](02.38-precision.md) | Precision = TP/(TP+FP) = P(positive \| flagged); falls with lower prevalence and lower thresholds. | Precision: when it shouts 'positive', how often is it right? |
| 2.39 | [Recall](02.39-recall.md) | Recall (TPR, sensitivity) = TP/(TP+FN) = P(flagged \| positive); rises as the threshold falls. | Recall: of everything that was really there, how much did we find? |
| 2.40 | [F1 score](02.40-f1.md) | F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN); harmonic mean; macro vs micro averaging; threshold-dependent. | F1: you can't hide a terrible recall behind a perfect precision. |
| 2.41 | [ROC and AUC](02.41-roc-auc.md) | AUC = P(score_pos > score_neg); 0.5 chance; rank-only; use PR-AUC for rare positives. | AUC measures ranking, not trust: it can't tell you the threshold or the probability. |
| 2.42 | [Confusion matrix](02.42-confusion-matrix.md) | Counts of (true, predicted) pairs; derives accuracy/precision/recall/F1; rows=true here. | Don't just count the errors — see which kind they are. |
| 2.43 | [Calibration](02.43-calibration.md) | Calibrated: predicted p ≈ observed frequency; check reliability diagram/ECE/Brier; fix with temperature/Platt on held-out data. | A probability is a promise — calibration checks whether it's kept. |
| 2.44 | [Cross-validation](02.44-cross-validation.md) | k-fold: train on k−1, validate on 1, rotate; stratify/group/time-aware; preprocess inside folds; nested CV for honest tuning. | Every row gets to be the exam once — but never while it is also in the textbook. |

### E. Imbalance, baselines, data preparation and metric choice (2.45–2.51)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 2.45 | [Class imbalance](02.45-class-imbalance.md) | Rare class: use PR-AUC/recall/precision; weights & resampling move the threshold, not the ranking; resample after splitting. | Imbalance changes the metric and the threshold, not the model's intelligence. |
| 2.46 | [Baseline models](02.46-baseline-models.md) | Majority/mean, naive heuristic, simple model, incumbent; every model must beat them on identical splits. | If you can't beat the dumb baseline, you don't have a model yet. |
| 2.47 | [Feature scaling and normalization](02.47-feature-scaling-normalization.md) | Standardize/min-max/robust (+log for skew); fit on train only; matters for GD, distances, penalties; not for trees/OLS predictions. | Put every feature on a comparable ruler — measured with the training set's ruler only. |
| 2.48 | [Feature engineering overview](02.48-feature-engineering-overview.md) | Create informative inputs (transforms, interactions, aggregates, encodings); beware leakage; ensure serving parity. | Better inputs beat fancier models — as long as they exist at prediction time. |
| 2.49 | [Data preprocessing](02.49-data-preprocessing.md) | Validate, impute (train stats) + indicators, clip outliers, dedupe, encode, scale; fit on train; ship as a pipeline. | Clean once, fit on train, ship the same recipe to production. |
| 2.50 | [Dataset imbalance](02.50-dataset-imbalance.md) | Dataset prevalence ≠ production prevalence ⇒ prior-correct logits, reweight eval metrics; AUC unaffected. | Rebalance the data if you must — then remember to rebalance the probabilities. |
| 2.51 | [Metric selection](02.51-metric-selection.md) | Metric = decision + costs + operating point; ranking metrics vs threshold metrics vs probability metrics; costs flip the winner; report slices & uncertainty. | Measure what the decision costs, not what is easiest to compute. |

## 3. Top pitfalls (each has a test or lab demonstration)

1. **Evaluating on training data, or peeking at the test set** (2.11–2.13): scores inflate; best-of-200 noise models → 62% on test, chance on fresh data.
2. **Fitting scalers/selectors/imputers before splitting** (2.33, 2.47, 2.49): 90% CV accuracy on pure noise vs 41.7% when selection is inside the fold.
3. **Random splits on grouped or time-ordered data** (2.10, 2.44): 99.5% vs 44%; MSE 0.19 vs 0.40.
4. **Target-derived features and naive target encoding** (2.33, 2.48): 100% vs 50%; correlation 0.55 vs 0.03 out-of-fold.
5. **Accuracy on imbalanced data** (2.37, 2.45): 98.9% for a model that finds nothing.
6. **ROC-AUC alone for rare positives** (2.41): AUC ≈ 0.98 throughout while average precision falls 0.98 → 0.46 and recall at 0.5 → 4%.
7. **Default 0.5 threshold** (2.14, 2.51): cost 48 vs 27 at the cost-optimal 0.1; the "winning" model flips with the cost ratio.
8. **Treating weights/oversampling as accuracy boosters** (2.45): they shift the operating point; PR-AUC does not change.
9. **Forgetting prevalence** (2.38, 2.50): precision 0.96 → 0.54 when prevalence drops 50% → 5%; downsampled training inflates probabilities.
10. **Trusting softmax scores as probabilities** (2.43): overconfident models; fit temperature/Platt on held-out data.
11. **Learning rate above 2/λ_max, or unscaled features** (2.25, 2.47): divergence (loss 3×10³¹); limit shrinks ~840×.
12. **Choosing hyperparameters on the test set, or reporting the best of many trials** (2.35, 2.36).
13. **Skipping the baseline** (2.46): naive forecast (0.188) beat the fitted trend (1.213) and the mean (0.840).
14. **Using "bias" and "variance" loosely** (2.28, 2.29): five meanings of bias; model variance ≠ data variance.
15. **No monitoring for shift** (2.34): linear model MSE 2.0 → 36.2 under covariate shift; PSI 3.37; domain AUC 0.92.

## 4. Interview drills (30–60 s each; answers are in the concept files)

- Why three data splits, and what goes wrong if you reuse the test set? → 2.11–2.13.
- Walk through bias–variance with a polynomial example and a learning curve. → 2.26–2.30.
- Your CV score is 0.99 but production is chance: list leakage suspects in order. → 2.33, 2.10, 2.44.
- Fraud with 0.2% positives: metrics, threshold, calibration, resampling? → 2.37–2.43, 2.45, 2.50, 2.51.
- Why does SGD work, and why decay the learning rate? → 2.19–2.21, 2.25.
- Loss vs objective vs metric: give an example where they disagree. → 2.17, 2.18, 2.51.
- Precision dropped from 90% to 20% in production with the same model. → 2.38, 2.34, 2.50.
- When does feature scaling matter and when not? → 2.47.
- Grid vs random search; why is the best CV score optimistic? → 2.35, 2.36.
- Design the evaluation plan for an LLM/RAG feature: baselines, metrics, slices, shift monitoring. → 2.46, 2.51, 2.34.

## 5. Status of claims

TESTED: every quantitative claim above that cites a lab demonstration or test (NumPy 2.5.4, scikit-learn 1.9.1, Python 3.14; fixed seeds, tests assert directions and thresholds). REVIEW_REQUIRED items (double descent, weight decay vs L2 under Adam, PSI thresholds as convention, LLM-probability calibration, hyperparameter-importance citation, nested CV and other items described but not implemented) are listed in `reviews/author_open_items.md`. Duplication to adjudicate: 2.45 vs 2.50; Tier 2 vs Tier 4 for SGD/batch/loss/learning rate.
