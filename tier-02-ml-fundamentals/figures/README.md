# Tier 2 figures

Original figures generated from this tier's own code (see [`../../figures-common/README.md`](../../figures-common/README.md) for the conventions).

| Figure | Concept | Type | What it shows | File | Script |
|---|---|---|---|---|---|
| 2.2-1 | [2.2](../02.02-supervised-learning.md) | SVG | learn from labelled examples, then predict new ones | `fig_2_02_supervised_learning.svg` | `src/fig_2_02_supervised_learning.py` |
| 2.1-1 | [2.1](../02.01-ai-vs-ml-vs-dl.md) | SVG | AI contains ML contains DL, and the XOR ladder | `fig_2_01_ai_ml_dl.svg` | `src/t2_basics.py` |
| 2.9-1 | [2.9](../02.09-samples.md) | SVG | a learning curve and what slows it | `fig_2_09_learning_curve.svg` | `src/t2_basics.py` |
| 2.10-1 | [2.10](../02.10-dataset-construction.md) | SVG | random row split versus group split | `fig_2_10_split_leakage.svg` | `src/t2_basics.py` |
| 2.13-1 | [2.13](../02.13-test-dataset.md) | SVG | what peeking at the test set does | `fig_2_13_test_peeking.svg` | `src/t2_basics.py` |
| 2.14-1 | [2.14](../02.14-classification.md) | PNG | probabilities, boundary and threshold | `fig_2_14_decision_boundary.png` | `src/t2_basics.py` |
| 2.16-1 | [2.16](../02.16-clustering.md) | GIF | k-means step by step (animated) | `fig_2_16_kmeans.gif` | `src/t2_training.py` |
| 2.17-1 | [2.17](../02.17-loss-functions.md) | SVG | regression losses and the gradient of classification losses | `fig_2_17_loss_functions.svg` | `src/t2_basics.py` |
| 2.20-1 | [2.20](../02.20-stochastic-gradient-descent.md) | PNG | full batch, mini-batch and pure SGD paths | `fig_2_20_sgd_paths.png` | `src/t2_training.py` |
| 2.25-1 | [2.25](../02.25-learning-rate.md) | GIF | three learning rates on a narrow valley (animated) | `fig_2_25_learning_rate.gif` | `src/t2_training.py` |
| 2.27-1 | [2.27](../02.27-overfitting.md) | SVG | the same 30 points fitted three ways | `fig_2_27_under_over_fitting.svg` | `src/t2_generalization.py` |
| 2.30-1 | [2.30](../02.30-bias-variance-trade-off.md) | SVG | bias², variance and noise against complexity | `fig_2_30_bias_variance.svg` | `src/t2_generalization.py` |
| 2.32-1 | [2.32](../02.32-regularization.md) | SVG | ridge and lasso coefficient paths | `fig_2_32_regularization_paths.svg` | `src/t2_generalization.py` |
| 2.34-1 | [2.34](../02.34-distribution-shift.md) | SVG | covariate shift and its detectors | `fig_2_34_distribution_shift.svg` | `src/t2_generalization.py` |
| 2.38-1 | [2.38](../02.38-precision.md) | SVG | precision, recall and the effect of prevalence | `fig_2_38_precision_recall.svg` | `src/t2_metrics.py` |
| 2.41-1 | [2.41](../02.41-roc-auc.md) | SVG | ROC versus precision–recall under imbalance | `fig_2_41_roc_vs_pr.svg` | `src/t2_metrics.py` |
| 2.42-1 | [2.42](../02.42-confusion-matrix.md) | SVG | a 3-class confusion matrix as counts and as recall | `fig_2_42_confusion_matrix.svg` | `src/t2_metrics.py` |
| 2.43-1 | [2.43](../02.43-calibration.md) | SVG | reliability diagrams before and after temperature scaling | `fig_2_43_calibration.svg` | `src/t2_metrics.py` |
| 2.44-1 | [2.44](../02.44-cross-validation.md) | SVG | k-fold, stratified and time-series splits | `fig_2_44_cross_validation.svg` | `src/t2_metrics.py` |
| 2.45-1 | [2.45](../02.45-class-imbalance.md) | SVG | imbalance methods move the operating point | `fig_2_45_imbalance_methods.svg` | `src/t2_metrics.py` |
| 2.47-1 | [2.47](../02.47-feature-scaling-normalization.md) | SVG | scaling reshapes the loss surface | `fig_2_47_scaling_conditioning.svg` | `src/t2_training.py` |
| 2.51-1 | [2.51](../02.51-metric-selection.md) | SVG | the best threshold moves with the costs | `fig_2_51_cost_vs_threshold.svg` | `src/t2_metrics.py` |

Regenerate everything: `python tier-02-ml-fundamentals/figures/src/make_all.py` (needs `matplotlib` and `pillow`; see `requirements.txt`).
