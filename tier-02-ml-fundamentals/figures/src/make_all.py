"""Regenerate every figure of this tier (run from anywhere): python make_all.py"""
import pathlib, runpy, sys
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for name in ['fig_2_02_supervised_learning.py', 't2_basics.py', 't2_generalization.py', 't2_metrics.py', 't2_training.py']:
    print("building", name); runpy.run_path(str(here / name), run_name="__main__")
