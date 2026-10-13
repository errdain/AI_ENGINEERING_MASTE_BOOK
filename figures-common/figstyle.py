"""Shared visual style for the book's figures: a dark, Manim-like look (deep blue-black canvas, saturated accents, smooth easing).

Every figure script in tier-NN-*/figures/src/ imports this module. Figures are generated from the SAME code and data that the
concept files use (tier-NN-*/code/), so numbers shown in a picture are computed, not typed in.
"""
from __future__ import annotations

import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BG = "#0f1220"
FG = "#e8e8f2"
MUTED = "#8a8fa8"
GRID = "#262b45"
BLUE = "#58c4dd"
TEAL = "#83c167"
YELLOW = "#f4d03f"
RED = "#fc6255"
ORANGE = "#ff862f"
PURPLE = "#b08fd8"
PINK = "#ff8cc6"
PALETTE = [BLUE, TEAL, YELLOW, RED, ORANGE, PURPLE, PINK]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.titlecolor": FG, "font.family": "DejaVu Sans", "mathtext.fontset": "cm",
        "axes.titlesize": 13, "axes.labelsize": 11, "legend.facecolor": "#161a2e", "legend.edgecolor": GRID, "legend.labelcolor": FG,
        "svg.fonttype": "none", "savefig.dpi": 200, "figure.dpi": 100,
    })


def ease(t: float) -> float:
    """Smooth ease-in-out on [0, 1] (Manim's `smooth` rate function, simplified)."""
    t = float(np.clip(t, 0, 1)); return t * t * (3 - 2 * t)


def fig_to_image(fig) -> Image.Image:
    fig.canvas.draw(); return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def save_gif(frames: list[Image.Image], path: str | pathlib.Path, fps: int = 12, colors: int = 96, hold_last: int = 6) -> int:
    """Write an optimized looping GIF; returns the file size in bytes. The last frame is held for `hold_last` frames."""
    q = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    q += [q[-1]] * hold_last
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return pathlib.Path(path).stat().st_size


def caption_text(ax, text: str, y: float = -0.02, **kw):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=9, **kw)
