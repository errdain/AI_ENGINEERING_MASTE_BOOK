# Tier 3 — author open items

## Batch 1 (3.1–3.17)

- TESTED: `code/nn.py` (21 tests; every activation value+derivative, softmax/log-softmax, Dense forward+backward, MLP parameter counts, whole-network backprop vs finite differences AND vs PyTorch autograd for 5 activations, losses vs PyTorch) and `lab/lab_01_neurons_and_mlp.py` (9 tests). PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, Python 3.14.
- Bug found by the tests while authoring: `sigmoid_bce` broadcast a `(n,1)` label array against a raveled logit array into `(n,n)`; fixed by raveling labels.
- Demonstration honesty: in 3.17 the 64-unit network fits `sin(x)` *worse* than the 16-unit network at the same fixed budget; the text reports this as an optimization effect. The ordering 1 > 2 > 4 > 16 is asserted, 64 is not.
- NOT_TESTED / REVIEW_REQUIRED: biological statements in 3.1 (spiking, plasticity, McCulloch–Pitts 1943, Rosenblatt 1958, Minsky–Papert 1969 history simplified); which published models use GELU / SiLU / SwiGLU / bias-free layers (3.7, 3.13, 3.14); the universal approximation theorem statement (3.17); "hierarchy of features" in 3.4; tanh-GELU constants (3.13).
- DUPLICATION / ORDER: 3.15 Softmax is used before Tier 7 (attention) and overlaps with 1.43/1.37 (stable softmax, cross-entropy); 3.25–3.26 (vanishing/exploding gradients) are previewed in 3.9–3.12 only through derivative bounds; initialization is implemented (`init_weights`) in this batch but taught in 3.22–3.24 (later batch).
- Terminology: "pre-activation", "logit", "dense/fully connected/linear layer" are used synonymously where stated; weight convention `(out, in)` follows PyTorch (Keras differs).
- Forward references: 3.18–3.32, Tier 4 (training loop, optimizers), Tier 5, 7, 8, 10, 11, 12, 13, 16, 19, 23.
