# Tier 1 figures

Original figures generated from this tier's own code (see [`../../figures-common/README.md`](../../figures-common/README.md) for the conventions).

| Figure | Concept | Type | What it shows | File | Script |
|---|---|---|---|---|---|
| 1.1-1 | [1.1](../01.01-scalars.md) | SVG | from one number to blocks of numbers | `fig_1_01_dimension_ladder.svg` | `src/fig_1_01_dimension_ladder.py` |
| 1.4-1 | [1.4](../01.04-tensors.md) | PNG | a batch of sequences is a 3-D tensor | `fig_1_04_tensor_3d.png` | `src/t1_tensors.py` |
| 1.6-1 | [1.6](../01.06-dimensions-and-axes.md) | SVG | which axis does a reduction remove? | `fig_1_06_axes_reduction.svg` | `src/t1_tensors.py` |
| 1.7-1 | [1.7](../01.07-broadcasting.md) | GIF | broadcasting (4, 1) + (1, 5), animated | `fig_1_07_broadcasting.gif` | `src/t1_tensors.py` |
| 1.8-1 | [1.8](../01.08-vector-addition.md) | SVG | adding and scaling vectors | `fig_1_08_vector_add_scale.svg` | `src/t1_geometry.py` |
| 1.10-1 | [1.10](../01.10-dot-product.md) | SVG | the dot product as projection and angle | `fig_1_10_dot_product.svg` | `src/t1_geometry.py` |
| 1.11-1 | [1.11](../01.11-matrix-multiplication.md) | GIF | matrix multiplication as many dot products (animated) | `fig_1_11_matrix_multiplication.gif` | `src/t1_tensors.py` |
| 1.15-1 | [1.15](../01.15-norms.md) | SVG | unit balls of different norms | `fig_1_15_norm_balls.svg` | `src/t1_geometry.py` |
| 1.16-1 | [1.16](../01.16-distance-metrics.md) | SVG | distance concentration in high dimension | `fig_1_16_distance_concentration.svg` | `src/t1_geometry.py` |
| 1.17-1 | [1.17](../01.17-cosine-similarity.md) | SVG | cosine similarity looks only at direction | `fig_1_17_cosine.svg` | `src/t1_geometry.py` |
| 1.18-1 | [1.18](../01.18-eigenvalues-eigenvectors.md) | GIF | eigenvectors under the matrix [[2, 1], [1, 2]] (animated) | `fig_1_18_eigenvectors.gif` | `src/t1_linalg_calculus.py` |
| 1.19-1 and 1.19-2 | [1.19](../01.19-singular-value-decomposition.md) | SVG | SVD as rotate-stretch-rotate, and low-rank approximation | `fig_1_19_svd_steps.svg` | `src/t1_linalg_calculus.py` |
| 1.19-1 and 1.19-2 | [1.19](../01.19-singular-value-decomposition.md) | PNG | SVD as rotate-stretch-rotate, and low-rank approximation | `fig_1_19_lowrank_image.png` | `src/t1_linalg_calculus.py` |
| 1.21-1 | [1.21](../01.21-derivatives.md) | GIF | the derivative as the limit of secant slopes (animated) | `fig_1_21_derivative_limit.gif` | `src/t1_linalg_calculus.py` |
| 1.23-1 | [1.23](../01.23-gradients.md) | PNG | the gradient points uphill (3-D) | `fig_1_23_gradient_surface.png` | `src/t1_linalg_calculus.py` |
| 1.27-1 | [1.27](../01.27-random-variables.md) | SVG | the law of large numbers | `fig_1_27_law_of_large_numbers.svg` | `src/t1_prob.py` |
| 1.28-1 | [1.28](../01.28-probability-distributions.md) | SVG | PMF, categorical, PDF and CDF | `fig_1_28_distributions.svg` | `src/t1_prob.py` |
| 1.31-1 | [1.31](../01.31-standard-deviation.md) | SVG | the 68–95–99.7 rule | `fig_1_31_empirical_rule.svg` | `src/t1_prob.py` |
| 1.32-1 | [1.32](../01.32-covariance.md) | SVG | a covariance matrix as an ellipse | `fig_1_32_covariance_ellipse.svg` | `src/t1_geometry.py` |
| 1.34-1 | [1.34](../01.34-bayes-theorem.md) | PNG | why most positives are false: 1,000 people | `fig_1_34_bayes_waffle.png` | `src/t1_prob.py` |
| 1.36-1 | [1.36](../01.36-entropy.md) | SVG | entropy of a coin | `fig_1_36_entropy.svg` | `src/t1_prob.py` |
| 1.38-1 | [1.38](../01.38-kl-divergence.md) | SVG | forward versus reverse KL | `fig_1_38_kl_directions.svg` | `src/t1_prob.py` |
| 1.40-1 | [1.40](../01.40-floating-point-precision.md) | SVG | the gap between representable numbers grows with the number | `fig_1_40_float_spacing.svg` | `src/t1_prob.py` |
| 1.45-1 | [1.45](../01.45-convexity-intuition.md) | SVG | the chord test for convexity | `fig_1_45_convexity.svg` | `src/t1_geometry.py` |
| 1.46-1 | [1.46](../01.46-optimization-landscapes.md) | PNG | a saddle point in 3D | `fig_1_46_saddle.png` | `src/fig_1_46_saddle.py` |
| 1.47-1 | [1.47](../01.47-local-global-minima.md) | GIF | two gradient-descent runs on a double-well surface (animated, rotating) | `fig_1_47_double_well.gif` | `src/fig_1_47_double_well.py` |

Regenerate everything: `python tier-01-mathematics/figures/src/make_all.py` (needs `matplotlib` and `pillow`; see `requirements.txt`).
