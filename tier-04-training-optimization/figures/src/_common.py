import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures-common"))
OUT = pathlib.Path(__file__).resolve().parents[1]
import figstyle
figstyle.apply()

import matplotlib.pyplot as plt
import numpy as np
fs = figstyle
