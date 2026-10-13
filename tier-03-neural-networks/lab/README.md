# Tier 3 Lab — a neural network built manually first, then in PyTorch

The roadmap's build task: *"neural network first manually, then in PyTorch."* `code/nn.py` is a NumPy library with hand-written forward and backward passes; `code/autodiff.py` is a tiny autodiff engine. PyTorch is used here (a) as an independent reference in tests and (b) to rebuild the same networks so the two implementations can be compared.

```sh
cd tier-03-neural-networks
pip install -r requirements.txt
python -m pytest -q code lab                  # 64 tests (35 in code/, 29 in lab/)
PYTHONPATH=code python lab/lab_02_training_internals.py     # print every demo result
```

| File | Concepts | Content |
|---|---|---|
| `lab_01_neurons_and_mlp.py` | 3.1–3.17 | logic gates, perceptron convergence bound and XOR failure, shapes and parameter counts, bias effect, weighted-sum loops vs matmul, activation table, softmax temperature, layer collapse, XOR by hand, width vs fit, circle dataset |
| `lab_02_training_internals.py` | 3.18–3.32 | forward-pass hand trace, computational graph, backprop vs finite differences, autodiff modes, initialization (symmetry, scale, Xavier, He), vanishing and exploding gradients, clipping, residual connections, dropout, weight decay, BatchNorm, LayerNorm, RMSNorm |
| `lab_03_pytorch_build.py` | 3.17–3.32 | identical NumPy/PyTorch training trajectories (weight difference ~1e-16), autograd graph inspection, standard `nn.Module` training loop, layers vs `torch.nn`, init helpers vs formulas, gradient clipping vs `clip_grad_norm_` |

Conventions: weight matrices are `(out, in)` like PyTorch; every `ex_*`/`t_*` function returns a dict; numbers in the concept files come from these functions with fixed seeds (NumPy 2.5.4, PyTorch 2.14.1, Python 3.14); tests assert directions/thresholds, not exact digits. Behaviour that depends on framework defaults (`torch.nn.Linear` initialization, `weight_decay` applying to biases) is version-sensitive: re-run when upgrading.
