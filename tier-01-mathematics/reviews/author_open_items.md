# Open items (batch 1, concepts 1.1–1.10)

- PROPOSE_ADDITION: none in this batch.
- REVIEW_REQUIRED: Section 1.4 §16 (safe checkpoint loading defaults vary by PyTorch version); verify against current docs before FINAL.
- NOT_TESTED: PyTorch snippet in 1.4 (torch not installed).
- Hands-on exercises 1.7 and 1.10 describe tests but do not yet ship solution code; to be added in the Tier 1 lab (`Build: NumPy/PyTorch Mathematics Lab`).
- Terminology: this batch introduces the distinction "number of axes" vs "vector dimension" (1.6); Glossary has no entry. Reviewer to approve or reject adding one (not inserted silently).

## Batch 2 (1.11–1.19)

- TESTED: all code in `code/linalg_core.py` (19 tests in `test_linalg_core.py`; 30 total with batch 1). NumPy 2.5.4.
- NOT_TESTED: exercises in 1.15 (global-norm clipping), 1.16 (distance concentration), 1.17 (ranking comparison), 1.18 (PCA axes), 1.19 (image rank) are described but not shipped as code; planned for the Tier 1 lab.
- Forward references to concepts not yet drafted: 1.32 Covariance, 1.40 Floating-point precision, 1.46 Optimization landscapes; Tier 2 PCA, Tier 5 RNN gradients, Tier 10 LoRA. Critic should confirm these cross-references match the frozen roadmap.
- REVIEW_REQUIRED: 1.19 §16 claim about low-rank structure aiding extraction/membership attacks needs a literature citation.
- REVIEW_REQUIRED: 1.11 §10 "≈2×parameters FLOPs/token" holds for the dense forward pass only (excludes attention score cost and MoE sparsity).
- Matmul ordering example in 1.11 (`(x@W1)@W2` vs `x@(W1@W2)`) is explained by operation count, not benchmarked.
