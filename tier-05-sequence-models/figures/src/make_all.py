"""Regenerate every figure of this tier (run from anywhere): python make_all.py   (takes a few minutes: it trains the small models)"""
import pathlib, runpy, sys
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for name in ['t5_figures.py']:
    print("building", name); runpy.run_path(str(here / name), run_name="__main__")
