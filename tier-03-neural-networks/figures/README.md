# Tier 3 figures (pilot)

| Figure | Concept | Type | What it shows | Script | Data source |
|---|---|---|---|---|---|
| 3.9-1 | [3.9 Activation functions](../03.09-activation-functions.md) | SVG | five activations and their derivatives; saturation and dead zones | `src/fig_3_09_activations.py` | `nn.ACTIVATIONS` |
| 3.17-1 | [3.17 MLP](../03.17-mlp.md) | PNG (3-D) | a trained 4-6-6-3 network; edge colour = weight sign, brightness = size | `src/fig_3_17_mlp_3d.py` | `nn.MLP` trained in the script |
| 3.18-1 | [3.18 Forward propagation](../03.18-forward-propagation.md) | GIF | the worked 2-3-1 forward pass, step by step | `src/fig_3_18_forward_pass.py` | `nn.MLP.forward` |
| 3.20-1 | [3.20 Backpropagation](../03.20-backpropagation.md) | GIF | gradients flowing backward through the same network | `src/fig_3_20_backprop.py` | `nn.MLP.backward`, `nn.mse_loss` |
| 3.25-1 | [3.25 Vanishing gradients](../03.25-vanishing-gradients.md) | SVG | per-layer gradient norm for five 20-layer networks | `src/fig_3_25_gradient_flow.py` | `lab_02_training_internals.grad_norms_by_layer` |
| 3.25-2 | [3.25 Vanishing gradients](../03.25-vanishing-gradients.md) | PNG (3-D) | gradient norm over depth × initial weight gain | `src/fig_3_25_gradient_flow.py` | same |

Regenerate: `python tier-03-neural-networks/figures/src/make_all.py` (needs `matplotlib` and `pillow`; see `requirements.txt`). Conventions: [`../../figures-common/README.md`](../../figures-common/README.md).
