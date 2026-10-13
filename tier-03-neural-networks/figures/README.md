# Tier 3 figures

Original figures generated from this tier's own code (see [`../../figures-common/README.md`](../../figures-common/README.md) for the conventions).

| Figure | Concept | Type | What it shows | File | Script |
|---|---|---|---|---|---|
| 3.9-1 | [3.9](../03.09-activation-functions.md) | SVG | activation functions and their derivatives | `fig_3_09_activations.svg` | `src/fig_3_09_activations.py` |
| 3.17-1 | [3.17](../03.17-mlp.md) | PNG | a trained MLP drawn in 3D | `fig_3_17_mlp_3d.png` | `src/fig_3_17_mlp_3d.py` |
| 3.18-1 | [3.18](../03.18-forward-propagation.md) | GIF | the forward pass, step by step (animated) | `fig_3_18_forward_pass.gif` | `src/fig_3_18_forward_pass.py` |
| 3.20-1 | [3.20](../03.20-backpropagation.md) | GIF | backpropagation on the same network (animated) | `fig_3_20_backprop.gif` | `src/fig_3_20_backprop.py` |
| 3.25-1 and 3.25-2 | [3.25](../03.25-vanishing-gradients.md) | SVG | how much learning signal reaches each layer, and how it depends on depth and initial scale | `fig_3_25_gradient_norms.svg` | `src/fig_3_25_gradient_flow.py` |
| 3.25-1 and 3.25-2 | [3.25](../03.25-vanishing-gradients.md) | PNG | how much learning signal reaches each layer, and how it depends on depth and initial scale | `fig_3_25_gain_depth_surface.png` | `src/fig_3_25_gradient_flow.py` |

Regenerate everything: `python tier-03-neural-networks/figures/src/make_all.py` (needs `matplotlib` and `pillow`; see `requirements.txt`).
