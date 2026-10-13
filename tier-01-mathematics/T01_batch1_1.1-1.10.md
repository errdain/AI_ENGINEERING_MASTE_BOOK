# TIER 1 — MATHEMATICS FOR AI
## Batch 1: Concepts 1.1 – 1.10 (Scalars → Dot Product)

Status: DRAFT (Author: Claude). Next: Critic (ChatGPT) → Code Review (Codex) → Final Review (ChatGPT).
Code: `code/linalg_basics.py`, `code/test_linalg_basics.py` — TESTED (11 passed, NumPy 2.5.4, Python 3.14).
PyTorch snippets are marked NOT_TESTED in this batch (torch not installed in the authoring environment).

---

## 1.1 Scalars

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 1/5 · Interview: Low · Hands-on: Medium
Prerequisites: none · Leads to: 1.2 Vectors

### 1. What is it?
A single number: a 0-dimensional quantity such as `3`, `-0.5`, or `1e-4`.

### 2. Why does it exist?
Everything bigger (vectors, matrices, tensors) is a collection of scalars. Many model controls are scalars too: learning rate, temperature, loss value.

### 3. Detailed explanation
A scalar has magnitude and no direction or axes. In code, "scalar" still has a *dtype* (`int64`, `float32`, `bfloat16`), and that dtype decides range and precision (see 1.40).

### 4. How does it work?
Scalars combine through ordinary arithmetic. A scalar times a vector scales every entry (1.9). A scalar-valued function is the usual shape of a training loss: many numbers in, one number out.

### 5. Simple example
Your age, a temperature reading, a price.

### 6. Technical example
Loss `L = 0.42`. Learning rate `η = 3e-4`. Sampling temperature `T = 0.7`.

### 7. Code
```python
import numpy as np
x = np.float32(0.7)
print(x.ndim, x.shape)   # 0 ()
```

### 8. Input → Process → Output
`numbers in → loss function → one scalar out → used to compute gradients`

### 9. Enterprise use case
Monitoring dashboards track scalar metrics (latency, cost per request, eval score) over time.

### 10. AI engineering connection
Training optimizes a scalar loss. LLM APIs expose scalar knobs (temperature, top_p, max_tokens).

### 11. Advantages
Simplest object; trivial to log and compare.

### 12. Limitations
One number cannot represent structure. A single score hides *why* something is good or bad.

### 13. Trade-offs
Scalar summaries (average accuracy) are easy to track but can mask per-slice failures.

### 14. Common mistakes
Confusing shape `()` with shape `(1,)`. Calling `.item()`-less tensors "floats" and getting shape bugs.

### 15. Performance / cost
Tracking per-step scalars on GPU and calling `.item()` every step forces a device sync; log every N steps.

### 16. Security
Scalar settings from user input (e.g., `max_tokens`) must be validated and capped to bound cost.

### 17. Interview explanation
"A scalar is a single number with no axes. In ML the most important one is the loss, because training is just nudging parameters to make that one number smaller."

### 18. Rapid-fire
A scalar is a 0-D value; the loss is the scalar training minimizes.

### 19. Interview questions
- Basic: What is the shape of a scalar?
- Intermediate: Difference between shape `()` and `(1,)`?
- Advanced: Why must a loss be scalar for `backward()`?
- Scenario: Your eval score rose 2 points but users complain. What might the scalar hide?

### 20. Hands-on exercise
Create a 0-D and a `(1,)` array; show that `a + b` broadcasts to shape `(1,)`.

### 21. Revision questions
1) ndim of a scalar? 2) Name three scalars in an LLM API call.

### 22. Related concepts
1.2 Vectors · 1.21 Derivatives · 1.30 Variance

### 23. Memory hook
**A scalar is one number with zero axes.**

---

## 1.2 Vectors

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 1/5 · Interview: High · Hands-on: High
Prerequisites: 1.1 · Leads to: 1.3, 1.8–1.10, 1.15–1.17, Tier 6 Embeddings

### 1. What is it?
An ordered list of numbers, a 1-D array: `v = [v₁, …, vₙ] ∈ ℝⁿ`.

### 2. Why does it exist?
Real things have several measurable properties at once. A vector packages them into one object we can do arithmetic on.

### 3. Detailed explanation
Two views, both useful: (a) a *list of features* (house = [area, rooms, age]); (b) a *point or arrow in n-dimensional space*. Order matters: `[1,2]` ≠ `[2,1]`. The length `n` is the dimension.

### 4. How does it work?
Vectors support addition and scaling (1.8, 1.9); together with dot products (1.10) they give lengths, angles and similarity.

### 5. Simple example
GPS coordinates `[lat, lon]`; an RGB color `[255, 128, 0]`.

### 6. Technical example
A token embedding is a vector, e.g. 768 or 4096 floats. A "row" of a dataset is a feature vector.

### 7. Code
```python
import numpy as np
v = np.array([2.0, -1.0, 0.5])
print(v.shape, v.ndim)   # (3,) 1
```

### 8. Input → Process → Output
`raw object → featurize/encode → vector → model`

### 9. Enterprise use case
Vector databases store millions of embedding vectors for semantic search.

### 10. AI engineering connection
Embeddings, hidden states, and attention queries/keys/values are vectors. Retrieval (Tier 20) is nearest-neighbor search over vectors.

### 11. Advantages
Uniform representation; fast hardware support.

### 12. Limitations
Fixed length; meaning of each coordinate is arbitrary for learned vectors (not human interpretable).

### 13. Trade-offs
Higher dimension → more expressiveness but more memory/compute and distance concentration.

### 14. Common mistakes
Mixing up row vs column vector; treating shape `(n,)` as `(n,1)`; comparing vectors from different embedding models.

### 15. Performance / cost
Memory = n × bytes per element. 1M vectors × 1536 × 4 B ≈ 6.1 GB at float32.

### 16. Security
Embeddings can leak information about source text; treat vector stores as sensitive data.

### 17. Interview explanation
"A vector is just an ordered list of numbers. In AI we use them to represent things, like words or images, so that 'similar' becomes 'close' and we can compute it."

### 18. Rapid-fire
A vector is a 1-D ordered list of numbers representing a point/direction in ℝⁿ.

### 19. Interview questions
- Basic: What is a vector? What is its dimension?
- Intermediate: Shape `(n,)` vs `(n,1)` vs `(1,n)`?
- Advanced: Why can't you compare embeddings from two different models?
- Scenario: Estimate storage for 50M 1024-d float16 vectors.

### 20. Hands-on exercise
Compute memory for 1M vectors of dim 768 in float32 and float16; verify with `ndarray.nbytes`.

### 21. Revision questions
1) Is order important? 2) What does dimension mean?

### 22. Related concepts
1.3 Matrices · 1.15 Norms · Tier 6 Embeddings

### 23. Memory hook
**A vector is a point with a name for each axis.**

---

## 1.3 Matrices

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 2/5 · Interview: High · Hands-on: High
Prerequisites: 1.2 · Leads to: 1.4, 1.11–1.14, 1.18–1.19, Tier 3

### 1. What is it?
A 2-D grid of numbers with `m` rows and `n` columns, shape `(m, n)`.

### 2. Why does it exist?
It stores a batch of vectors, or a linear transformation (a rule that maps vectors to vectors), in one object.

### 3. Detailed explanation
Two readings: **data** (rows = examples, columns = features) and **transformation** (a matrix `W` maps an `n`-vector to an `m`-vector via `Wx`). Neural-network layers are mostly matrices of learned weights.

### 4. How does it work?
`A[i, j]` is row `i`, column `j`. Matrix–vector and matrix–matrix products (1.11) apply the transformation.

### 5. Simple example
A spreadsheet; a grayscale image (height × width).

### 6. Technical example
A linear layer `y = xWᵀ + b` with `W` of shape `(out_features, in_features)`. An embedding table of shape `(vocab_size, d_model)`.

### 7. Code
```python
import numpy as np
A = np.arange(6.0).reshape(2, 3)   # shape (2, 3)
print(A[1, 2], A.T.shape)          # 5.0 (3, 2)
```

### 8. Input → Process → Output
`vector (n,) → matrix (m,n) → vector (m,)`

### 9. Enterprise use case
Tabular data feeding fraud models; feature matrices in batch scoring pipelines.

### 10. AI engineering connection
Weights, attention score matrices, and embedding tables are matrices. Most FLOPs in a Transformer are matrix multiplies.

### 11. Advantages
Maps cleanly onto GPU/TPU matmul hardware.

### 12. Limitations
Only 2 axes: batches of sequences need 3-D tensors (1.4).

### 13. Trade-offs
Dense matrices are simple and fast; sparse formats save memory only when sparsity is high.

### 14. Common mistakes
Swapping rows/columns convention (PyTorch `Linear.weight` is `(out, in)`); forgetting row-major layout affects speed.

### 15. Performance / cost
A dense `(m, n)` float32 matrix costs `4mn` bytes. A 4096×4096 float32 matrix is 64 MiB.

### 16. Security
Model weight files are matrices with commercial value; protect and verify integrity (e.g., checksums, safe formats).

### 17. Interview explanation
"A matrix is a grid of numbers. I think of it two ways: a table of data, or a function that turns one vector into another. Neural network weights are matrices."

### 18. Rapid-fire
A matrix is a 2-D array; it stores data or represents a linear map.

### 19. Interview questions
- Basic: What is shape `(m, n)`?
- Intermediate: Why is `Linear.weight` `(out, in)` in PyTorch?
- Advanced: Parameter count of a 4096→11008 layer?
- Scenario: Weights are 64 MiB per layer; how many fit on a 24 GB GPU after reserving 8 GB?

### 20. Hands-on exercise
Build `W (3,4)` and `x (4,)`, compute `W @ x`, and state the output shape before running.

### 21. Revision questions
1) What does an `(m,n)` matrix map from and to? 2) Rows vs columns in a dataset?

### 22. Related concepts
1.11 Matrix multiplication · 1.12 Transpose · Tier 3 Neural networks

### 23. Memory hook
**A matrix is a table of data, or a machine that turns vectors into vectors.**

---

## 1.4 Tensors

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 2/5 · Interview: High · Hands-on: Very high
Prerequisites: 1.1–1.3 · Leads to: 1.5–1.7, Tier 3, Tier 8

### 1. What is it?
An n-dimensional array of numbers sharing one dtype. Scalar = 0-D, vector = 1-D, matrix = 2-D, and higher.

### 2. Why does it exist?
Real data has more axes than two: batch × time × features, or batch × channels × height × width.

### 3. Detailed explanation
In ML, "tensor" means a multi-dimensional array (as in NumPy/PyTorch), not the full physics definition involving transformation laws. Frameworks add device placement (CPU/GPU) and, in training libraries, gradient tracking.

### 4. How does it work?
A tensor has `shape`, `dtype`, `device`. Operations are vectorized across all elements.

### 5. Simple example
A stack of spreadsheets: sheets × rows × columns.

### 6. Technical example
LLM activations: `(batch, seq_len, d_model)`, e.g. `(32, 128, 768)`.

### 7. Code
```python
import numpy as np
x = np.zeros((32, 128, 768), dtype=np.float32)
print(x.ndim, x.size, x.nbytes / 2**20)  # 3 3145728 12.0 (MiB)
```
```python
# NOT_TESTED (torch not installed here)
import torch
x = torch.zeros(32, 128, 768, dtype=torch.float32, device="cpu")
```

### 8. Input → Process → Output
`data → tensor (batch, ...) → tensor ops → tensor out`

### 9. Enterprise use case
Batch inference services stack requests into tensors to use the GPU efficiently.

### 10. AI engineering connection
Every input, weight, activation and gradient in deep learning is a tensor. KV cache is a tensor whose size drives serving cost (Tier 11).

### 11. Advantages
One abstraction for all data; fast vectorized compute.

### 12. Limitations
Needs rectangular (non-ragged) data; padding/masking is required for variable lengths.

### 13. Trade-offs
Bigger batch → better throughput, more memory.

### 14. Common mistakes
Wrong axis order (`NCHW` vs `NHWC`); mixed dtypes/devices; thinking "tensor" implies GPU.

### 15. Performance / cost
Memory = `product(shape) × bytes(dtype)`. Halving dtype width halves memory.

### 16. Security
Untrusted serialized tensors (e.g., pickle-based checkpoints) can execute code on load; prefer safe formats such as safetensors. REVIEW_REQUIRED: verify current loader defaults in the PyTorch version used.

### 17. Interview explanation
"A tensor is an n-dimensional array. Scalars, vectors and matrices are just 0, 1 and 2-D cases. In deep learning everything flows as tensors, usually shaped batch, sequence, features."

### 18. Rapid-fire
A tensor is an n-D array with a dtype (and device); the universal data container of deep learning.

### 19. Interview questions
- Basic: Tensor vs matrix?
- Intermediate: Shape of a batch of 32 RGB 224×224 images in NCHW?
- Advanced: Memory of KV cache for given layers/heads/seq?
- Scenario: OOM at batch 64: which tensors dominate and what are your options?

### 20. Hands-on exercise
Compute `nbytes` for a `(8, 1024, 4096)` tensor in float32/float16 and check with NumPy.

### 21. Revision questions
1) ndim of vector/matrix/image batch? 2) Which three attributes define a tensor in a framework?

### 22. Related concepts
1.5 Tensor shapes · 1.6 Axes · 1.7 Broadcasting

### 23. Memory hook
**Tensor = the n-dimensional box that all deep learning data lives in.**

---

## 1.5 Tensor shapes

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 2/5 · Interview: High · Hands-on: Very high
Prerequisites: 1.4 · Leads to: 1.6, 1.7, 1.11

### 1. What is it?
The tuple listing each axis's size, e.g. `(32, 128, 768)`.

### 2. Why does it exist?
Shape is the contract between operations. Most deep learning bugs are shape bugs.

### 3. Detailed explanation
`size = product(shape)`. `reshape` changes the shape without changing element order or count; `transpose`/`permute` reorders axes; `squeeze`/`unsqueeze` remove/add size-1 axes.

### 4. How does it work?
Rules: matmul needs inner dims equal `(m,n)@(n,p)→(m,p)`; elementwise ops need equal or broadcast-compatible shapes (1.7).

### 5. Simple example
A bookshelf: 5 shelves × 20 books → shape `(5, 20)`.

### 6. Technical example
Multi-head attention reshapes `(B, T, D)` → `(B, H, T, D/H)`.

### 7. Code
```python
import numpy as np
x = np.arange(24).reshape(2, 3, 4)
print(x.shape, x.reshape(6, 4).shape, x.transpose(0, 2, 1).shape)  # (2,3,4) (6,4) (2,4,3)
```

### 8. Input → Process → Output
`shape in → op rule → shape out` (predict it *before* running).

### 9. Enterprise use case
Input validation at model-serving boundaries rejects malformed payload shapes before reaching the GPU.

### 10. AI engineering connection
Shape annotations are how Transformer code is read and debugged.

### 11. Advantages
Predictable, checkable contracts.

### 12. Limitations
Dynamic shapes (variable sequence length) complicate compilation and batching.

### 13. Trade-offs
Padding to fixed shapes simplifies code but wastes compute.

### 14. Common mistakes
`reshape` vs `transpose` confusion (reshape does **not** reorder data semantically); silent broadcasting hiding a shape bug.

### 15. Performance / cost
Non-contiguous tensors after transpose may require copies before `view`/reshape.

### 16. Security
Validate shapes/dtypes of untrusted inputs to avoid resource exhaustion (huge allocations).

### 17. Interview explanation
"Shape is the list of axis sizes. I always write the expected shape next to each tensor operation, because most bugs in deep learning code are shape mismatches."

### 18. Rapid-fire
Shape = per-axis sizes; matmul needs matching inner dims; reshape keeps element count.

### 19. Interview questions
- Basic: Shape of `x[0]` for `x` of `(2,3,4)`?
- Intermediate: `reshape` vs `transpose`?
- Advanced: Why is `view` after `transpose` an error in PyTorch?
- Scenario: Debug `mat1 and mat2 shapes cannot be multiplied`.

### 20. Hands-on exercise
Annotate each line of a small attention head with shapes, then verify with asserts.

### 21. Revision questions
1) `size` from shape? 2) Output shape of `(4,5) @ (5,2)`?

### 22. Related concepts
1.6 Axes · 1.7 Broadcasting · 1.11 Matrix multiplication

### 23. Memory hook
**Write the shape first; the code follows.**

---

## 1.6 Dimensions and axes

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 2/5 · Interview: Medium · Hands-on: Very high
Prerequisites: 1.5 · Leads to: 1.7, Tier 3, Tier 7

### 1. What is it?
An **axis** is one direction (index position) of a tensor; **ndim** is the number of axes. Operations like `sum(axis=…)` collapse a chosen axis.

### 2. Why does it exist?
You must say *which* axis an operation acts on (average over batch? over features?).

### 3. Detailed explanation
Terminology warning: "dimension" is overloaded. *Tensor ndim* (3 axes) differs from *vector dimension* (768 entries). This book writes "number of axes" for the former and "vector dimension" for the latter.

### 4. How does it work?
Reducing over axis `k` removes that axis (unless `keepdims=True`). Negative indices count from the end (`axis=-1` = last).

### 5. Simple example
Table of scores: `sum(axis=0)` totals each column; `sum(axis=1)` totals each row.

### 6. Technical example
Softmax over the last axis (keys) makes attention weights for each query sum to 1. LayerNorm normalizes over the feature axis.

### 7. Code
```python
import numpy as np
x = np.arange(12.0).reshape(3, 4)
print(x.sum(axis=0).shape, x.sum(axis=1).shape, x.sum(axis=-1, keepdims=True).shape)  # (4,) (3,) (3, 1)
```

### 8. Input → Process → Output
`tensor + axis → reduce/apply → tensor with that axis removed or kept`

### 9. Enterprise use case
Metrics aggregation over request, time and model-version axes in analytics pipelines.

### 10. AI engineering connection
Softmax, normalization and pooling are defined by their axis; wrong axis = silently wrong model.

### 11. Advantages
Precise control of aggregation.

### 12. Limitations
Easy to apply to the wrong axis without any error.

### 13. Trade-offs
`keepdims=True` keeps broadcast-friendliness at the cost of an extra size-1 axis.

### 14. Common mistakes
Softmax over batch axis; mean over the wrong axis; mixing the "dimension" meanings.

### 15. Performance / cost
Reductions along the contiguous axis are typically faster.

### 16. Security
N/A beyond shape validation.

### 17. Interview explanation
"An axis is a direction in the tensor. Every reduction or normalization needs the axis specified; for example softmax goes over the key axis so each query's weights add to one."

### 18. Rapid-fire
Axis = a tensor direction; reductions remove it; softmax/LayerNorm axis choice is critical.

### 19. Interview questions
- Basic: What does `sum(axis=0)` do on `(3,4)`?
- Intermediate: Why `keepdims=True`?
- Advanced: Which axis does LayerNorm normalize over versus BatchNorm?
- Scenario: Attention outputs depend on other items in the batch; find the bug.

### 20. Hands-on exercise
Write a test proving softmax rows sum to 1 along the chosen axis and not the other.

### 21. Revision questions
1) Shape after `mean(axis=1)` on `(5,6,7)`? 2) Two meanings of "dimension"?

### 22. Related concepts
1.5 · 1.7 · Tier 7 Attention

### 23. Memory hook
**Name the axis, or the axis picks you.**

---

## 1.7 Broadcasting

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 3/5 · Interview: High · Hands-on: Very high
Prerequisites: 1.5, 1.6 · Leads to: 1.8, 1.9, Tier 3, Tier 7

### 1. What is it?
A rule that lets elementwise operations work on tensors of different shapes by virtually stretching size-1 (or missing) axes.

### 2. Why does it exist?
To avoid writing loops and avoid copying data (e.g., add one bias vector to every row).

### 3. Detailed explanation
Compare shapes **right-aligned**. For each axis pair: sizes are compatible if equal, or one of them is 1 (or missing). The result axis is the larger. Otherwise: error. The stretching is virtual; data is not copied.

### 4. How does it work?
Example `(32,1,768) + (128,1)`: right-align → `(32,1,768)` vs `(1,128,1)` → axis-wise max → `(32,128,768)`.

### 5. Simple example
Adding the same tax to every price in a list.

### 6. Technical example
Adding a bias `(d,)` to activations `(B,T,d)`; applying a causal mask `(T,T)` to scores `(B,H,T,T)`.

### 7. Code (manual rule, tested against NumPy)
```python
def broadcast_shape(s1, s2):
    out = []
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        if d1 == d2 or d2 == 1: out.append(d1)
        elif d1 == 1:           out.append(d2)
        else: raise ValueError(f"cannot broadcast {s1} with {s2}")
    longer = s1 if len(s1) > len(s2) else s2
    out.extend(reversed(longer[: abs(len(s1) - len(s2))]))
    return tuple(reversed(out))
```
Full version + parametrized tests vs `np.broadcast_shapes`: `code/linalg_basics.py` — TESTED.

### 8. Input → Process → Output
`two shapes → right-align → compare per axis → result shape or error`

### 9. Enterprise use case
Vectorized feature scaling (subtract per-feature mean) over millions of rows without Python loops.

### 10. AI engineering connection
Bias addition, masking, normalization, attention scaling and sampling temperature all rely on broadcasting.

### 11. Advantages
Concise, fast, memory-light.

### 12. Limitations
Silent bugs: `(n,)` with `(n,1)` yields `(n,n)` instead of an error.

### 13. Trade-offs
Brevity vs readability; consider explicit `unsqueeze`/`reshape` and shape asserts.

### 14. Common mistakes
`(n,) - (n,1)` accidental outer operation; loss `(B,1)` vs target `(B,)` producing `(B,B)`.

### 15. Performance / cost
Broadcasting avoids materializing the expanded tensor, but the *result* can be large: `(32,128,768)` float32 ≈ 12 MiB; a careless `(N,N)` from `(N,)` and `(N,1)` with N=100k is ~40 GB.

### 16. Security
Untrusted shapes + accidental broadcasting can trigger huge allocations (denial of service); validate shapes.

### 17. Interview explanation
"Broadcasting lets tensors of different shapes combine elementwise. Shapes are aligned from the right, and axes must match or be 1. It's how we add a bias vector to a whole batch without a loop — and it's also a classic source of silent bugs."

### 18. Rapid-fire
Right-align shapes; axes must be equal or 1; size-1 axes stretch virtually.

### 19. Interview questions
- Basic: Can `(3,)` + `(4,3)` work? Result?
- Intermediate: What is `(5,)` - `(5,1)`?
- Advanced: Why can a `(B,1)` vs `(B,)` loss mismatch silently train wrongly?
- Scenario: Memory spikes after a harmless-looking subtraction; diagnose.

### 20. Hands-on exercise
Write a test that fails if `pred.shape != target.shape` before computing MSE; show the silent `(B,B)` bug without it.

### 21. Revision questions
1) Result of `(4,1)` with `(1,5)`? 2) Is data copied when broadcasting?

### 22. Related concepts
1.5 · 1.6 · 1.8 · Tier 7

### 23. Memory hook
**Right-align, match or 1, else error.**

---

## 1.8 Vector addition

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 1/5 · Interview: Medium · Hands-on: High
Prerequisites: 1.2 · Leads to: Tier 8 residual connections

### 1. What is it?
Elementwise sum of two same-length vectors: `(a+b)ᵢ = aᵢ + bᵢ`.

### 2. Why does it exist?
Combine effects/information: displacements, features, signals.

### 3. Detailed explanation
Geometrically: place the arrows tip-to-tail. Properties: commutative, associative, with a zero vector identity.

### 4. How does it work?
Entry by entry. Length mismatch is an error (or a broadcast in tensor libraries, so check shapes).

### 5. Simple example
Walk 3 east + 4 north: `[3,0]+[0,4]=[3,4]`.

### 6. Technical example
Residual connection `y = x + F(x)`; positional encoding added to token embeddings.

### 7. Code
```python
from linalg_basics import vec_add_manual
print(vec_add_manual([1, 2, 3], [4, 5, 6]))  # [5, 7, 9]
```
TESTED (`test_add_and_scale_match_numpy`, `test_add_length_mismatch_raises`).

### 8. Input → Process → Output
`a (n,), b (n,) → elementwise + → (n,)`

### 9. Enterprise use case
Combining feature vectors from different sources in a feature-store pipeline.

### 10. AI engineering connection
Residual streams in Transformers; "king − man + woman" analogies are famous but approximate and not a guarantee for modern embeddings.

### 11. Advantages
Linear time, trivially parallel.

### 12. Limitations
Requires aligned semantics: adding unrelated spaces is meaningless.

### 13. Trade-offs
Addition mixes information irreversibly; concatenation keeps it separate but grows dimension.

### 14. Common mistakes
Adding vectors from different embedding spaces; relying on silent broadcasting.

### 15. Performance / cost
O(n) memory-bound; fused kernels avoid extra memory passes.

### 16. Security
N/A.

### 17. Interview explanation
"Vector addition is elementwise. In Transformers it shows up as residual connections: the layer output is added to its input so information and gradients have a direct path."

### 18. Rapid-fire
Elementwise sum of equal-length vectors; basis of residual connections.

### 19. Interview questions
- Basic: Compute `[1,2]+[3,4]`.
- Intermediate: Add vs concatenate?
- Advanced: Why do residual connections help deep networks train?
- Scenario: Positional encodings added to embeddings: what must match?

### 20. Hands-on exercise
Verify commutativity and associativity numerically with random float64 vectors using `allclose`.

### 21. Revision questions
1) Identity element? 2) What if lengths differ?

### 22. Related concepts
1.9 · 1.7 · Tier 8

### 23. Memory hook
**Add vectors entry by entry; same length or stop.**

---

## 1.9 Scalar multiplication

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 1/5 · Interview: Medium · Hands-on: High
Prerequisites: 1.1, 1.2 · Leads to: 1.15 Norms, gradient descent

### 1. What is it?
Multiplying every entry of a vector (or tensor) by one scalar: `(c·a)ᵢ = c·aᵢ`.

### 2. Why does it exist?
To change magnitude (and flip direction if `c<0`) without changing the line along which the vector points.

### 3. Detailed explanation
`c>1` stretches, `0<c<1` shrinks, `c<0` reverses, `c=0` gives the zero vector. Distributes over addition: `c(a+b)=ca+cb`.

### 4. How does it work?
Each entry is scaled independently.

### 5. Simple example
Doubling a recipe.

### 6. Technical example
Gradient descent step: `w ← w − η·∇L`. Attention scores are scaled by `1/√dₖ`.

### 7. Code
```python
from linalg_basics import vec_scale_manual
print(vec_scale_manual(-2, [1, 2, 3]))  # [-2, -4, -6]
```
TESTED.

### 8. Input → Process → Output
`scalar c, vector a → c·aᵢ per entry → vector`

### 9. Enterprise use case
Normalizing features (scale to unit range) before model training.

### 10. AI engineering connection
Learning rate, temperature scaling of logits (`logits / T`), and attention scaling are scalar multiplications.

### 11. Advantages
Cheap, interpretable control knob.

### 12. Limitations
Cannot change direction (except flip); a poor global scale can hurt precision.

### 13. Trade-offs
Large scale → overflow risk (1.44); small → underflow.

### 14. Common mistakes
Dividing logits by `T=0`; applying temperature after softmax instead of before.

### 15. Performance / cost
Memory-bound, fused with neighbors in compiled kernels.

### 16. Security
Clamp user-provided scale factors/temperature.

### 17. Interview explanation
"Scalar multiplication stretches or shrinks a vector. The learning rate in gradient descent and the temperature on logits are both scalar multiplications — one scales the update, the other the sharpness of the output distribution."

### 18. Rapid-fire
Multiply every entry by a scalar; changes length, not direction (unless negative).

### 19. Interview questions
- Basic: `3·[1,-2]`?
- Intermediate: Effect of temperature <1 on softmax?
- Advanced: Why scale attention scores by `1/√dₖ`?
- Scenario: Training diverges after raising the learning rate; explain via scaling the update.

### 20. Hands-on exercise
Plot softmax outputs of fixed logits for T in {0.25, 1, 4} (numbers only if no plotting).

### 21. Revision questions
1) Effect of `c=-1`? 2) Is `c(a+b)=ca+cb`?

### 22. Related concepts
1.8 · 1.15 · 1.23 Gradients

### 23. Memory hook
**A scalar is a volume knob for a vector.**

---

## 1.10 Dot product

**Metadata** — Tier: 1 · Phase: 1 · Priority: P0 · Difficulty: 2/5 · Interview: Very high · Hands-on: Very high
Prerequisites: 1.2, 1.8, 1.9 · Leads to: 1.11, 1.15, 1.17, Tier 7 Attention

### 1. What is it?
For equal-length vectors, `a·b = Σ aᵢbᵢ` — a single scalar.

### 2. Why does it exist?
It measures how aligned two vectors are. It is the elementary operation behind matrix multiplication, similarity and attention.

### 3. Detailed explanation
Geometric form: `a·b = ‖a‖‖b‖cos θ`. Positive → pointing roughly the same way; zero → orthogonal; negative → opposing. It grows with both lengths, so on its own it mixes *direction* and *magnitude* (normalize to get cosine similarity, 1.17).

### 4. How does it work?
Multiply matching entries, then sum. Properties: commutative, distributes over addition, `a·a = ‖a‖²`.

### 5. Simple example
Shopping: `[2 apples, 3 bread] · [price₁, price₂]` = total cost.

### 6. Technical example
Attention score between a query and a key: `q·k / √dₖ`. A linear neuron: `w·x + b`.

### 7. Code
```python
import numpy as np
from linalg_basics import dot_manual
a, b = np.array([1., 2., 3.]), np.array([4., 5., 6.])
print(dot_manual(a.tolist(), b.tolist()), a @ b, np.dot(a, b))  # 32.0 32.0 32.0
```
TESTED (value, orthogonality, length-mismatch error).

### 8. Input → Process → Output
`a (n,), b (n,) → elementwise product → sum → scalar`

### 9. Enterprise use case
Semantic search ranks documents by dot-product (or cosine) between query and document embeddings; many vector DBs support inner-product metrics.

### 10. AI engineering connection
Attention scores, embedding similarity, and every neuron are dot products. `QKᵀ` is just all pairwise dot products at once.

### 11. Advantages
Cheap, differentiable, hardware-optimized.

### 12. Limitations
Sensitive to vector length; only captures linear (aligned) relationships.

### 13. Trade-offs
Dot product (magnitude-aware) vs cosine (direction only) vs L2 distance. Choose the metric the embedding model was trained for.

### 14. Common mistakes
Assuming dot product ∈ [−1, 1] (that's cosine); using it on unnormalized embeddings of varying norm; confusing dot product (scalar) with elementwise product (vector).

### 15. Performance / cost
O(n). Dominates retrieval cost: brute-force search over N vectors is O(N·n); ANN indexes trade recall for speed.

### 16. Security
Similarity scores can leak membership/info about the stored vectors; apply access control at the vector-store layer, not only the app layer.

### 17. Interview explanation
"The dot product multiplies matching entries and sums them into one number that says how aligned two vectors are. Attention uses it to score a query against keys, and retrieval uses it to compare embeddings. If I need direction only, I normalize first, which gives cosine similarity."

### 18. Rapid-fire
Dot product = Σ aᵢbᵢ = ‖a‖‖b‖cosθ; a scalar alignment score.

### 19. Interview questions
- Basic: Compute `[1,2,3]·[4,5,6]`.
- Intermediate: What does dot product zero mean? Relationship to cosine?
- Advanced: Why divide attention scores by `√dₖ`? (variance grows with dₖ → softmax saturation)
- Scenario: Retrieval favors long documents' chunks; relate to embedding norms and metric choice.

### 20. Hands-on exercise
Show numerically that for random N(0,1) vectors, variance of `q·k` ≈ `d`, and that dividing by `√d` brings it ≈ 1 (d = 16, 256, 4096).

### 21. Revision questions
1) Dot product result type? 2) Geometric meaning? 3) Dot vs elementwise product?

### 22. Related concepts
1.11 Matrix multiplication · 1.15 Norms · 1.17 Cosine similarity · Tier 7

### 23. Memory hook
**Dot product = multiply pairs, add up: how much do two vectors agree?**

---

## Open items for Critic / Reviewers

- PROPOSE_ADDITION: none in this batch.
- REVIEW_REQUIRED: Section 1.4 §16 (safe checkpoint loading defaults vary by PyTorch version); verify against current docs before FINAL.
- NOT_TESTED: PyTorch snippet in 1.4 (torch not installed).
- Hands-on exercises 1.7 and 1.10 describe tests but do not yet ship solution code; to be added in the Tier 1 lab (`Build: NumPy/PyTorch Mathematics Lab`).
- Terminology: this batch introduces the distinction "number of axes" vs "vector dimension" (1.6); Glossary has no entry. Reviewer to approve or reject adding one (not inserted silently).
