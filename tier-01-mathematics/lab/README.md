# Tier 1 Lab — NumPy / PyTorch Mathematics Lab

Solutions to the hands-on exercise (section 20) of every Tier 1 concept, plus PyTorch checks for the claims the concept files make about PyTorch.

```sh
cd tier-01-mathematics
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q code lab          # 105 tests (77 in code/, 28 in lab/)
python lab/lab_01_linear_algebra.py   # print every exercise result (needs PYTHONPATH=code)
```

Run the demos with `PYTHONPATH=code python lab/<file>.py`. Verified with Python 3.14.5, NumPy 2.5.4, torch 2.14.1 (CPU).

| File | Concepts | Exercises (function) |
|---|---|---|
| `lab_01_linear_algebra.py` | 1.1–1.19 | `ex_1_1` … `ex_1_19`: shapes, memory sizing, attention shape asserts, softmax axis, broadcasting bug + guard, dot-product variance, matmul timing, transpose contiguity, ridge `+λI`, conditioning (×1e10 amplification), global-norm clipping, distance concentration, cosine/dot/L2 rankings, covariance axes (PCA), SVD energy |
| `lab_02_calculus_probability.py` | 1.20–1.31 | `ex_1_20` … `ex_1_31`: activation ranges, finite-difference sweet spot, partials, GD stability threshold, Jacobian, backprop with bias/ReLU, dice simulation, seeds mean±std, normal vs log-normal tails, robust means, Bessel's correction, `StandardScaler` |
| `lab_03_information_numerics.py` | 1.32–1.47 | `ex_1_32` … `ex_1_47`: covariance axes, bigram chain rule, prevalence sweep, exponential MLE, entropy vs temperature, CE gradient check, forward vs reverse KL, variance with a `Decimal` reference, float32 summation error, log-prob of a sequence, max finite `exp` per dtype, online log-sum-exp, NaN guard, convex vs non-convex restarts, momentum vs GD, basins of a double well |
| `lab_04_pytorch.py` | 1.4, 1.12, 1.25, 1.30, 1.37, 1.38, 1.40, 1.43, 1.44 | `t_*`: tensor basics, view-after-transpose error, autograd vs manual backprop, `var` default, `F.cross_entropy` + double-softmax pitfall, `F.kl_div` convention, `torch.finfo` dtype table, fp16 vs bf16, `torch.logsumexp`, fp16 loss scaling, int8 wraparound, NaN propagation, `torch.load(weights_only)` |

Conventions: every `ex_*`/`t_*` function returns a dict so results can be asserted; timing-based checks use best-of-3 and loose bounds; PyTorch behaviour that is version-sensitive (e.g. `torch.load` defaults) is recorded and should be re-verified when upgrading.
