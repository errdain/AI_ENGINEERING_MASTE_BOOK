# Tier 1 — Mathematics for AI: Revision Sheet

> Status: DRAFT (Author: Claude). Condensed from concept files `01.01`–`01.47`. The concept file is the authoritative source; this sheet only summarises it. Rapid-fire answers and memory hooks in the tables below are copied from sections 18 and 23 of each concept file.
> Code evidence: `code/` (77 tests) and `lab/` (28 tests). Run: `cd tier-01-mathematics && python -m pytest -q code lab`.

## 1. One-page formula sheet

| Topic | Formula | Note |
|---|---|---|
| Dot product | `a·b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ` | scalar; basis of attention scores |
| Matmul | `(m,n)@(n,p) → (m,p)`, ≈ `2mnp` FLOPs | associative, not commutative; `(AB)ᵀ = BᵀAᵀ` |
| Broadcasting | right-align shapes; axes equal or 1 | silent `(n,)` vs `(n,1)` → `(n,n)` bug |
| Norms | `‖x‖₁=Σ\|xᵢ\|`, `‖x‖₂=√Σxᵢ²`, `‖x‖∞=max\|xᵢ\|` | `p<1` is not a norm |
| Cosine | `a·b / (‖a‖‖b‖)` | = dot product on unit vectors; `‖a−b‖² = 2−2cos` |
| Eigen | `Av = λv`; symmetric: `S = QΛQᵀ` | use `eigh` for symmetric matrices |
| SVD | `A = UΣVᵀ`; best rank-k = top-k terms | error `√Σ_{i>k} σᵢ²` (Eckart–Young) |
| Condition number | `κ = σ_max/σ_min` | error amplification factor |
| Derivative | `f′(x) = lim (f(x+h)−f(x))/h` | central difference error `O(h²)`, h≈1e-5 (float64) |
| Gradient | `∇f = [∂f/∂xᵢ]` | same shape as params; descent: `x ← x − η∇f` |
| Chain rule | `dy/dx = (dy/du)(du/dx)`; `J_{g∘f} = J_g J_f` | sum over paths, multiply along paths |
| Softmax Jacobian | `diag(p) − ppᵀ` | CE+softmax gradient wrt logits: `p − onehot` |
| Expectation | `E[X] = Σ x p(x)`; `E[aX+b] = aE[X]+b` | `E[g(X)] ≠ g(E[X])` |
| Variance | `E[(X−μ)²] = E[X²]−μ²`; `Var(aX+b) = a²Var(X)` | sample: ÷(n−1); NumPy default ÷n |
| Std error | `σ/√n` | 4× data → 2× less error |
| Conditional | `P(A\|B) = P(A∩B)/P(B)` | chain rule: `P(x₁..x_T) = Π P(x_t\|x_<t)` |
| Bayes | `P(H\|E) ∝ P(E\|H)P(H)` | base rate matters (99%-sensitive test, 1% prevalence → 1/6) |
| MLE | `argmax Σ log p(xᵢ\|θ)` | NLL = loss; Gaussian→MSE, categorical→cross-entropy |
| Entropy | `H = −Σ p log p` | max `log K` at uniform; perplexity `= exp(H)` |
| Cross-entropy / KL | `H(p,q) = H(p) + KL(p‖q)`; `KL ≥ 0`, asymmetric | `q=0` where `p>0` → ∞ |
| Log-sum-exp | `m + log Σ exp(xᵢ−m)`, `m = max x` | gradient = softmax; CE = `−z_y + LSE(z)` |
| Floating point | float16 max 65504 (ε≈9.8e-4); bf16 ε≈7.8e-3, fp32 range; float32 ε≈1.2e-7; float64 ε≈2.2e-16 | exponent=range, mantissa=precision |
| exp overflow | float16 ≈ e^11.09, float32 ≈ e^88.72, float64 ≈ e^709.78 | subtract max before exp |
| Convex | `f(λa+(1−λ)b) ≤ λf(a)+(1−λ)f(b)`; Hessian PSD | local min = global min |
| GD on quadratic | stable iff `η < 2/λ_max` | `x²+10y²`: `η < 0.1` |

## 2. Concept-by-concept revision

### A. Linear algebra foundations (1.1–1.19)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.1 | [Scalars](01.01-scalars.md) | A scalar is a 0-D value; the loss is the scalar training minimizes. | A scalar is one number with zero axes. |
| 1.2 | [Vectors](01.02-vectors.md) | A vector is a 1-D ordered list of numbers representing a point/direction in ℝⁿ. | A vector is a point with a name for each axis. |
| 1.3 | [Matrices](01.03-matrices.md) | A matrix is a 2-D array; it stores data or represents a linear map. | A matrix is a table of data, or a machine that turns vectors into vectors. |
| 1.4 | [Tensors](01.04-tensors.md) | A tensor is an n-D array with a dtype (and device); the universal data container of deep learning. | Tensor = the n-dimensional box that all deep learning data lives in. |
| 1.5 | [Tensor shapes](01.05-tensor-shapes.md) | Shape = per-axis sizes; matmul needs matching inner dims; reshape keeps element count. | Write the shape first; the code follows. |
| 1.6 | [Dimensions and axes](01.06-dimensions-and-axes.md) | Axis = a tensor direction; reductions remove it; softmax/LayerNorm axis choice is critical. | Name the axis, or the axis picks you. |
| 1.7 | [Broadcasting](01.07-broadcasting.md) | Right-align shapes; axes must be equal or 1; size-1 axes stretch virtually. | Right-align, match or 1, else error. |
| 1.8 | [Vector addition](01.08-vector-addition.md) | Elementwise sum of equal-length vectors; basis of residual connections. | Add vectors entry by entry; same length or stop. |
| 1.9 | [Scalar multiplication](01.09-scalar-multiplication.md) | Multiply every entry by a scalar; changes length, not direction (unless negative). | A scalar is a volume knob for a vector. |
| 1.10 | [Dot product](01.10-dot-product.md) | Dot product = Σ aᵢbᵢ = ‖a‖‖b‖cosθ; a scalar alignment score. | Dot product = multiply pairs, add up: how much do two vectors agree? |
| 1.11 | [Matrix multiplication](01.11-matrix-multiplication.md) | `(m,n)@(n,p)→(m,p)`; each entry is a row–column dot product; associative, not commutative; ~2mnp FLOPs. | Row meets column, dot-product, repeat — inner dimensions must agree. |
| 1.12 | [Matrix transpose](01.12-matrix-transpose.md) | Swap rows/columns; `(AB)ᵀ = BᵀAᵀ`; `.T` is a view. | Flip over the diagonal; products flip their order. |
| 1.13 | [Identity matrices](01.13-identity-matrices.md) | Ones on the diagonal; `AI = IA = A`; defines inverses; `+λI` stabilizes. | Identity: the matrix that changes nothing. |
| 1.14 | [Matrix inverse](01.14-matrix-inverse.md) | `AA⁻¹ = I`; needs square + full rank; prefer `solve`; watch the condition number. | Undo button for a matrix — exists only if nothing was squashed, and use `solve`, not `inv`. |
| 1.15 | [Norms](01.15-norms.md) | Norm = size; L1 sparse, L2 smooth, L∞ worst-case; clip gradients by global L2 norm. | Norm = ruler for vectors: L1 counts blocks, L2 uses a string, L∞ takes the biggest step. |
| 1.16 | [Distance metrics](01.16-distance-metrics.md) | Metric = non-negative, symmetric, triangle inequality; on unit vectors L2² = 2−2cos. | Distance = norm of the difference; keep the metric the same end to end. |
| 1.17 | [Cosine similarity](01.17-cosine-similarity.md) | `cos = a·b / (‖a‖‖b‖)`, range [−1,1], scale-invariant; equals dot product on unit vectors. | Cosine = the angle, not the length: normalize once, then dot. |
| 1.18 | [Eigenvalues and eigenvectors](01.18-eigenvalues-eigenvectors.md) | `Av = λv`; symmetric ⇒ real λ, orthogonal v; largest /λ/ dominates repeated products. | Eigenvectors are the matrix's natural axes; eigenvalues say how hard it pulls along each. |
| 1.19 | [Singular Value Decomposition](01.19-singular-value-decomposition.md) | `A = UΣVᵀ`; top-k truncation is the best rank-k approximation; PCA is SVD of centered data. | SVD: rotate, stretch, rotate — and keep only the biggest stretches. |

### B. Calculus (1.20–1.25)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.20 | [Functions](01.20-functions.md) | A function maps each input to one output; networks are compositions; nonlinearity prevents collapse. | A network is functions stacked; nonlinearity keeps the stack from collapsing. |
| 1.21 | [Derivatives](01.21-derivatives.md) | Derivative = instantaneous rate of change = slope; training follows it downhill. | Derivative = slope: which way and how steeply the output moves when you nudge the input. |
| 1.22 | [Partial derivatives](01.22-partial-derivatives.md) | ∂f/∂xᵢ = slope along axis i with others fixed; all of them make the gradient. | Partial derivative = wiggle one knob, freeze the rest. |
| 1.23 | [Gradients](01.23-gradients.md) | ∇L points uphill; step −η∇L; same shape as parameters; clip when it explodes. | Gradient = compass pointing uphill; walk the other way, in small steps. |
| 1.24 | [Jacobians](01.24-jacobians.md) | Jacobian = m×n derivative matrix; backprop uses vector–Jacobian products; softmax J = diag(p) − ppᵀ. | Jacobian: every output's sensitivity to every input, in one matrix. |
| 1.25 | [Chain rule](01.25-chain-rule.md) | d(g∘f) = g′(f)·f′; backprop = chain rule backwards; sum over paths, multiply along them. | Multiply along the path, add across the paths, and check with finite differences. |

### C. Probability and statistics (1.26–1.35)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.26 | [Probability](01.26-probability.md) | P ∈ [0,1], P(Ω)=1, additive for disjoint events; softmax ≠ guaranteed calibration. | Probability: numbers between 0 and 1 that must add up to 1 over everything that can happen. |
| 1.27 | [Random variables](01.27-random-variables.md) | A random variable maps outcomes to numbers; discrete or continuous; i.i.d. is an assumption. | A random variable turns "what happened" into "a number we can average". |
| 1.28 | [Probability distributions](01.28-probability-distributions.md) | PMF/PDF/CDF; Bernoulli, categorical, normal; densities can exceed 1; temperature reshapes softmax. | A distribution is the shape of uncertainty: heights are probabilities for PMFs, areas for PDFs. |
| 1.29 | [Mean and expectation](01.29-mean-expectation.md) | E[X] = Σ x·p(x); linear; sample mean estimates it; E[g(X)] ≠ g(E[X]). | Expectation = the balance point; losses are expectations estimated by batches. |
| 1.30 | [Variance](01.30-variance.md) | Var = E[(X−μ)²] = E[X²]−μ²; Var(aX+b)=a²Var; sample variance uses n−1; compute it stably. | Variance: average squared surprise; shifting is free, scaling is squared. |
| 1.31 | [Standard deviation](01.31-standard-deviation.md) | σ = √Var; z = (x−μ)/σ; 68-95-99.7 for normal data; SE = σ/√n. | Std = typical distance from the mean, in the data's own units. |
| 1.32 | [Covariance](01.32-covariance.md) | Cov = E[(X−μx)(Y−μy)]; matrix is symmetric PSD; PCA diagonalizes it; uncorrelated ≠ independent. | Covariance: do two variables wiggle together — in a straight line only. |
| 1.33 | [Conditional probability](01.33-conditional-probability.md) | P(A/B)=P(A∩B)/P(B); chain rule factorizes sequences; not symmetric. | Conditioning: shrink the world to where B is true, then ask how often A is also true. |
| 1.34 | [Bayes' theorem](01.34-bayes-theorem.md) | Posterior ∝ likelihood × prior; base rates matter; MLE = MAP with a flat prior. | Posterior = prior × likelihood, then normalize — and never forget the base rate. |
| 1.35 | [Maximum likelihood](01.35-maximum-likelihood.md) | argmax Σ log p(x/θ); NLL = loss; cross-entropy = categorical NLL; MSE = Gaussian NLL. | MLE: pick the parameters that would have been least surprised by the data. |

### D. Information theory (1.36–1.38)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.36 | [Entropy](01.36-entropy.md) | H = −Σ p log p; max at uniform (log K); perplexity = exp(H). | Entropy: how surprised you are on average — flat distributions surprise most. |
| 1.37 | [Cross-entropy](01.37-cross-entropy.md) | H(p,q) = H(p) + KL(p‖q); one-hot case = −log q(y); gradient = softmax − onehot; compute from logits. | Cross-entropy: how many bits you pay for the truth when you bet with the wrong distribution. |
| 1.38 | [KL divergence](01.38-kl-divergence.md) | KL(p‖q)=Σp log(p/q) ≥ 0; asymmetric; ∞ if q=0 where p>0; CE = H + KL. | KL: the extra bits you pay for believing q when the truth is p — and it matters which is which. |

### E. Numerical computing (1.39–1.44)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.39 | [Numerical stability](01.39-numerical-stability.md) | Shift before exp, log-space for products, fuse ops, add ε, solve don't invert, check for NaN. | Same math, different arithmetic: pick the form that keeps numbers small and subtractions safe. |
| 1.40 | [Floating-point precision](01.40-floating-point-precision.md) | Exponent=range, mantissa=precision; fp16 max 65504; bf16 = fp32 range; accumulate in fp32; never `==`. | Floats are a ruler whose marks spread apart as numbers grow — never trust the last digits. |
| 1.41 | [Logarithms](01.41-logarithms.md) | log(ab)=log a+log b; monotonic; domain x>0; use log-probs; log1p for tiny x. | Log: the ruler that turns multiplication into addition. |
| 1.42 | [Exponentials](01.42-exponentials.md) | eˣ>0, own derivative, inverse of ln; softmax/sigmoid use it; overflows fast. | exp: makes everything positive, makes gaps explode — handle with a max-shift. |
| 1.43 | [Log-sum-exp](01.43-log-sum-exp.md) | LSE = m + log Σ exp(x−m); smooth max; gradient = softmax; cross-entropy = −z_y + LSE(z). | Factor out the biggest number, then add up what's left. |
| 1.44 | [Numerical overflow and underflow](01.44-numerical-overflow-underflow.md) | Overflow→inf, underflow→0, inf−inf=NaN; use log space, max-shift, ε, clipping, loss scaling. | Too big becomes inf, too small becomes zero, and inf or zero together make NaN. |

### F. Optimization intuition (1.45–1.47)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 1.45 | [Convexity intuition](01.45-convexity-intuition.md) | Convex = bowl; Hessian PSD; local min = global min; deep nets are non-convex. | Convex: a bowl with one bottom; deep learning is a mountain range. |
| 1.46 | [Optimization landscapes](01.46-optimization-landscapes.md) | Saddles > bad minima in high-D; ravines need momentum/Adam; cliffs need clipping; flat minima ≈ better generalization (contested). | Training is hiking in fog: you feel the slope, not the map — saddles, ravines and plateaus decide the journey. |
| 1.47 | [Local and global minima (intuition)](01.47-local-global-minima.md) | Local vs global; init decides basin; zero gradient ≠ best; deep nets: use seeds/ensembles and validate. | Gradient descent rolls downhill into the nearest valley, not necessarily the deepest one. |

## 3. Top pitfalls (each one has a test or lab exercise)

1. **Shape bugs hidden by broadcasting** — `(B,1)` vs `(B,)` gives `(B,B)` (1.7, lab `ex_1_7`).
2. **Softmax/loss from probabilities instead of logits** — use log-sum-exp; naive softmax of `[1000,1001,1002]` is NaN (1.37, 1.43, lab `t_1_37`).
3. **Double softmax before `CrossEntropyLoss`** — runs silently with a wrong loss (lab `t_1_37`).
4. **Explicit matrix inverse** — use `solve`; check the condition number (1.14, lab `ex_1_14`: 1e-8 change in b → ×1e10 amplification).
5. **Wrong `ddof`** — NumPy `var` ÷n, PyTorch `var` ÷(n−1) (1.30, lab `t_1_30`).
6. **Naive `E[X²]−E[X]²`** — catastrophic cancellation at large offsets; use Welford (1.30, lab `ex_1_39`).
7. **Zero correlation ≠ independence** (1.32).
8. **Scaler fit on test data** — leakage (1.31, lab `ex_1_31`).
9. **Treating softmax scores as calibrated probabilities / ignoring base rates** (1.26, 1.34).
10. **Cosine vs dot vs L2 mismatch between index and query** (1.16, 1.17, lab `ex_1_17`).
11. **KL argument order, and ∞ from a zero in q** (1.38).
12. **Perplexity compared across tokenizers, or bits mixed with nats** (1.36, 1.37).
13. **fp16 without loss scaling; summing in low precision** (1.40, 1.44, lab `t_1_44`).
14. **Learning rate above the stability limit** — diverges (1.23, lab `ex_1_23`).
15. **Zero gradient ≠ minimum; zero training loss ≠ goal** (1.46, 1.47, lab `ex_1_47`).

## 4. Interview drills (30–60 s each, answers are in the concept files)

- Why does attention divide scores by `√d`? → 1.10 (variance of `q·k` grows as `d`; lab `ex_1_10`).
- Why LoRA / low-rank? → 1.19.
- Why is cross-entropy's gradient `softmax − onehot`? → 1.24, 1.37.
- Why are products of probabilities computed in log space? → 1.41, 1.44.
- Why does a 99%-accurate test give mostly false alarms? → 1.34.
- Why does bf16 train without loss scaling but fp16 doesn't? → 1.40, 1.44.
- Why is the minibatch gradient unbiased? → 1.27, 1.29.
- Why do deep nets get stuck at saddles/plateaus more than at bad local minima? → 1.46 (REVIEW_REQUIRED citations).
- Estimate memory of a 7B model in fp16 / 4-bit. → 1.40.
- Loss turns NaN at step 8,000: debugging plan. → 1.39, 1.44 (lab `ex_1_44` guard).

## 5. Status of claims

TESTED: all numeric claims above that cite a test, lab exercise, or `t_` PyTorch check (torch 2.14.1, NumPy 2.5.4). REVIEW_REQUIRED items remain in `reviews/author_open_items.md` (citations for landscape/overparameterization claims; hardware training practice for fp8/bf16).
