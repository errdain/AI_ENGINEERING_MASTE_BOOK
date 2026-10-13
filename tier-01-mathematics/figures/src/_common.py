import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures-common"))
sys.path.insert(0, str(ROOT / "tier-01-mathematics" / "code"))
OUT = pathlib.Path(__file__).resolve().parents[1]
import figstyle
figstyle.apply()

import matplotlib.pyplot as plt
import numpy as np
fs = figstyle


def arrow(ax, p0, p1, color, lw=2.6, label=None, ls="-", head=14, z=5, fs_label=11, label_off=(0.05, 0.05), alpha=1.0):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=f"-|>,head_width=0.25,head_length=0.5", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, alpha=alpha), zorder=z)
    if label: ax.text(p1[0] + label_off[0], p1[1] + label_off[1], label, color=color, fontsize=fs_label, zorder=z + 1)


def clean2d(ax, lim=None, grid=True, equal=True):
    if equal: ax.set_aspect("equal")
    if lim: ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    if grid: ax.grid(True, alpha=0.35)
    ax.axhline(0, color=fs.GRID, lw=1); ax.axvline(0, color=fs.GRID, lw=1)
