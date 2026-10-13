"""Regenerate every figure of this tier (run from anywhere): python make_all.py"""
import pathlib, runpy, sys
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for name in ['t3_concepts.py', 't3_norm.py', 't3_training.py', 'fig_3_09_activations.py', 'fig_3_17_mlp_3d.py', 'fig_3_18_forward_pass.py', 'fig_3_20_backprop.py', 'fig_3_25_gradient_flow.py']:
    print("building", name); runpy.run_path(str(here / name), run_name="__main__")
