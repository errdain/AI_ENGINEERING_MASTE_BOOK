# Tier 3 figures

Original figures generated from this tier's own code (see [`../../figures-common/README.md`](../../figures-common/README.md) for the conventions).

| Figure | Concept | Type | What it shows | File | Script |
|---|---|---|---|---|---|
| 3.2-1 | [3.2](../03.02-perceptron.md) | SVG | one perceptron computing AND | `fig_3_02_perceptron_diagram.svg` | `src/fig_3_02_perceptron_diagram.py` |
| 3.2-2 | [3.2](../03.02-perceptron.md) | GIF | perceptron learning, step by step (animated) | `fig_3_02_perceptron.gif` | `src/t3_concepts.py` |
| 3.4-1 | [3.4](../03.04-hidden-layers.md) | PNG | why a hidden layer helps: the circle problem | `fig_3_04_hidden_layers_boundary.png` | `src/t3_concepts.py` |
| 3.7-1 | [3.7](../03.07-bias.md) | SVG | what the bias does to the decision boundary | `fig_3_07_bias_boundary.svg` | `src/t3_concepts.py` |
| 3.9-1 | [3.9](../03.09-activation-functions.md) | SVG | activation functions and their derivatives | `fig_3_09_activations.svg` | `src/fig_3_09_activations.py` |
| 3.15-1 | [3.15](../03.15-softmax.md) | SVG | softmax and temperature | `fig_3_15_softmax_temperature.svg` | `src/t3_concepts.py` |
| 3.17-1 | [3.17](../03.17-mlp.md) | PNG | a trained MLP drawn in 3D | `fig_3_17_mlp_3d.png` | `src/fig_3_17_mlp_3d.py` |
| 3.18-1 | [3.18](../03.18-forward-propagation.md) | GIF | the forward pass, step by step (animated) | `fig_3_18_forward_pass.gif` | `src/fig_3_18_forward_pass.py` |
| 3.19-1 | [3.19](../03.19-computational-graph.md) | SVG | a computational graph with forward values and gradients | `fig_3_19_computational_graph.svg` | `src/t3_concepts.py` |
| 3.20-1 | [3.20](../03.20-backpropagation.md) | GIF | backpropagation on the same network (animated) | `fig_3_20_backprop.gif` | `src/fig_3_20_backprop.py` |
| 3.22-1 | [3.22](../03.22-parameter-initialization.md) | SVG | activation scale through 30 layers for different initializations | `fig_3_22_initialization.svg` | `src/t3_training.py` |
| 3.25-1 and 3.25-2 | [3.25](../03.25-vanishing-gradients.md) | SVG | how much learning signal reaches each layer, and how it depends on depth and initial scale | `fig_3_25_gradient_norms.svg` | `src/fig_3_25_gradient_flow.py` |
| 3.25-1 and 3.25-2 | [3.25](../03.25-vanishing-gradients.md) | PNG | how much learning signal reaches each layer, and how it depends on depth and initial scale | `fig_3_25_gain_depth_surface.png` | `src/fig_3_25_gradient_flow.py` |
| 3.26-1 | [3.26](../03.26-exploding-gradients.md) | SVG | exploding gradients and global-norm clipping | `fig_3_26_exploding_clipping.svg` | `src/t3_training.py` |
| 3.27-1 | [3.27](../03.27-residual-connections.md) | SVG | a residual block and what it does to training | `fig_3_27_residual_connections.svg` | `src/t3_training.py` |
| 3.28-1 | [3.28](../03.28-dropout.md) | GIF | dropout masks over training steps (animated) | `fig_3_28_dropout.gif` | `src/t3_norm.py` |
| 3.29-1 | [3.29](../03.29-weight-decay.md) | SVG | weight decay shrinks weights and can help on small data | `fig_3_29_weight_decay.svg` | `src/t3_training.py` |
| 3.30-1 | [3.30](../03.30-batch-normalization.md) | SVG | what batch normalization does and its catch | `fig_3_30_batch_normalization.svg` | `src/t3_norm.py` |
| 3.31-1 | [3.31](../03.31-layer-normalization.md) | SVG | which numbers each normalization layer groups together | `fig_3_31_normalization_axes.svg` | `src/t3_norm.py` |
| 3.32-1 | [3.32](../03.32-rmsnorm.md) | SVG | LayerNorm versus RMSNorm outputs | `fig_3_32_layernorm_vs_rmsnorm.svg` | `src/t3_norm.py` |

Regenerate everything: `python tier-03-neural-networks/figures/src/make_all.py` (needs `matplotlib` and `pillow`; see `requirements.txt`).
