# Tier 2 Lab — small ML training / evaluation laboratory

Runnable solutions and demonstrations for the concepts of Tier 2. Library code is in `../code/mlkit.py`; scikit-learn is used **only in tests** as an independent reference.

```sh
cd tier-02-ml-fundamentals
pip install -r requirements.txt
python -m pytest -q code lab                    # 83 tests (35 in code/, 48 in lab/)
PYTHONPATH=code python lab/lab_01_paradigms_and_data.py     # print every demo result
```

| File | Concepts | Demos |
|---|---|---|
| `lab_01_paradigms_and_data.py` | 2.1–2.13 | rule vs linear vs engineered vs neural net on XOR, supervised baseline, k-means elbow, self-training, bigram self-supervision, bandit exploration, junk features / label noise / learning curve, group-vs-row split leakage, test-set peeking |
| `lab_02_models_and_training.py` | 2.14–2.25 | multiclass softmax, cost-based threshold, R², k-means scaling and initialization, silhouette, robust losses, BCE vs MSE saturation, ridge objective, GD = closed form, mini-batch gradient variance, epoch/batch/iteration accounting, learning-rate stability limit |
| `lab_03_generalization.py` | 2.26–2.36 | under/overfitting by degree, early stopping, bias–variance decomposition, generalization gap, ridge/lasso, leakage (selection, target, temporal), distribution shift (PSI, KS, domain classifier, label shift), random vs grid search, selection optimism |
| `lab_04_metrics_and_data.py` | 2.37–2.51 | accuracy paradox, precision/recall vs prevalence, F1/macro/micro, ROC-AUC vs PR-AUC under imbalance, confusion matrix + cost, calibration (temperature, Platt), CV vs hold-out variance and time-series CV, imbalance methods, baselines, scalers, feature engineering and target-encoding leakage, preprocessing pipeline, prior correction, metric selection by cost |

Conventions: every `ex_*` function returns a dict; numbers quoted in the concept files come from these functions with fixed seeds on NumPy 2.5.4 / scikit-learn 1.9.1 / Python 3.14; tests assert directions and thresholds, not exact digits.
