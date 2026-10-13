"""Regenerate every figure of this tier (run from anywhere): python make_all.py"""
import pathlib, runpy, sys
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for name in ['fig_1_01_dimension_ladder.py', 't1_geometry.py', 't1_linalg_calculus.py', 't1_prob.py', 't1_tensors.py', 'fig_1_46_saddle.py', 'fig_1_47_double_well.py']:
    print("building", name); runpy.run_path(str(here / name), run_name="__main__")
