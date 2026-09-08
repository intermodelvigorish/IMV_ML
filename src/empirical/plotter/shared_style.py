"""Panel layout helpers using the project's canonical publication style.

Panels from different examples end up side by side in the paper, so their size,
palette, weights, sizes and edge colour have to agree. Importing this module and
calling :func:`apply` is the whole contract::

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from empirical.plotter import shared_style

    shared_style.apply()
    figure, axes = plt.subplots(2, 3, figsize=shared_style.figure_size(2, 3),
                                layout="constrained")

Every setting is also a module constant, so a notebook that needs to deviate can
read the constant and override one call instead of restyling from scratch.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


if __package__:
    from .figure_utils import (COLORMAP, PALETTE_COLORS, PAPER_STYLE,
                               spectral_colors)
else:
    from figure_utils import (COLORMAP, PALETTE_COLORS, PAPER_STYLE,
                              spectral_colors)

__all__ = [
    "apply", "figure_size", "categorical_colors", "sequential_cmap",
    "bar_style", "heatmap_style", "rc_params", "COLORS",
]

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

# One panel, in inches. The example notebooks were written against this size and
# the multi-panel figures are built as whole multiples of it, so a single panel
# lifted out of a grid still matches the standalone figures.
PANEL_WIDTH = 5.4
PANEL_HEIGHT = 4.6
FIGURE_DPI = 110              # on-screen only; publication exports are PDF

# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #

# Keep palette and typography in sync with all other publication figures.
PALETTE = COLORMAP
# The six named colours behind the ramp, for a mark that needs one on its own
# ("navy", "steel_blue", "blue", "light_blue", "cream", "red").
COLORS = PALETTE_COLORS
EDGE_COLOR = "#1f2a30"        # near-black with the palette's blue-green cast
EDGE_WIDTH = 0.6
GRID_COLOR = "#d7dcdf"
GRID_LINESTYLE = "--"
AXIS_COLOR = "#3f484d"
TEXT_COLOR = "#1f2a30"
ERROR_COLOR = "#3f484d"
CAPSIZE = 3

# --------------------------------------------------------------------------- #
# Type
# --------------------------------------------------------------------------- #

FONT_FAMILY = PAPER_STYLE["font.family"]
BASE_FONT_SIZE = PAPER_STYLE["font.size"]
TICK_FONT_SIZE = PAPER_STYLE["xtick.labelsize"]
LABEL_FONT_SIZE = PAPER_STYLE["axes.labelsize"]
TITLE_FONT_SIZE = PAPER_STYLE["axes.titlesize"]
SUPTITLE_FONT_SIZE = 13
LEGEND_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 9

# Panel titles sit flush with the left edge of the axes, so a1/a2 and the rest
# of the panel labels line up down the figure.
TITLE_LOCATION = "left"
BODY_FONT_WEIGHT = "normal"
LABEL_FONT_WEIGHT = "medium"
TITLE_FONT_WEIGHT = "semibold"
SUPTITLE_FONT_WEIGHT = "bold"

TITLE_KWARGS = {"fontsize": TITLE_FONT_SIZE, "fontweight": TITLE_FONT_WEIGHT,
                "color": TEXT_COLOR}
SUPTITLE_KWARGS = {"fontsize": SUPTITLE_FONT_SIZE, "fontweight": SUPTITLE_FONT_WEIGHT,
                   "color": TEXT_COLOR}
LABEL_KWARGS = {"fontsize": LABEL_FONT_SIZE, "fontweight": LABEL_FONT_WEIGHT,
                "color": TEXT_COLOR}


def rc_params():
    """The style as a plain rcParams mapping, for callers that prefer a context."""
    return {
        **PAPER_STYLE,
        "axes.prop_cycle": plt.cycler(color=spectral_colors(10)),
        "figure.figsize": (PANEL_WIDTH, PANEL_HEIGHT),
        "figure.dpi": FIGURE_DPI,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "font.weight": BODY_FONT_WEIGHT,
        "text.color": TEXT_COLOR,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": TITLE_FONT_WEIGHT,
        "axes.titlelocation": TITLE_LOCATION,
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.labelweight": LABEL_FONT_WEIGHT,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": EDGE_WIDTH,
        "axes.facecolor": "white",
        # Bars are read against their neighbours, so the frame is dropped and a
        # faint horizontal rule carries the comparison instead.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": GRID_LINESTYLE,
        "grid.linewidth": 0.6,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "xtick.labelcolor": TEXT_COLOR,
        "ytick.labelcolor": TEXT_COLOR,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.frameon": False,
    }


def apply(**overrides):
    """Install the style globally; keyword overrides are applied on top."""
    settings = {**rc_params(), **overrides}
    plt.rcParams.update(settings)
    return settings


def figure_size(n_rows=1, n_cols=1, *, width=PANEL_WIDTH, height=PANEL_HEIGHT):
    """Figure size for a grid of panels, in inches."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError("a figure needs at least one row and one column")
    return (n_cols * width, n_rows * height)


def categorical_colors(n_colors):
    """`n_colors` well-separated colours drawn from the shared palette."""
    return sns.color_palette(PALETTE, n_colors)


def sequential_cmap():
    """The shared palette as a continuous colormap, for heatmaps and ramps."""
    return sns.color_palette(PALETTE, as_cmap=True)


def bar_style(n_bars):
    """Keyword arguments for a categorical bar chart with error bars."""
    return {
        "color": categorical_colors(n_bars),
        "edgecolor": EDGE_COLOR,
        "linewidth": EDGE_WIDTH,
        "capsize": CAPSIZE,
        "error_kw": {"ecolor": ERROR_COLOR, "elinewidth": EDGE_WIDTH,
                     "capthick": EDGE_WIDTH},
    }


def heatmap_style(n_classes):
    """Keyword arguments for an annotated square heatmap of `n_classes` classes.

    The annotation shrinks and loses a decimal as the matrix grows, which is the
    point at which three decimals in a seven-class grid stop being readable.
    """
    crowded = n_classes > 5
    return {
        "cmap": sequential_cmap(),
        "annot": True,
        "fmt": ".2f" if crowded else ".3f",
        "annot_kws": {"fontsize": ANNOTATION_FONT_SIZE - (2 if crowded else 0),
                      "fontweight": BODY_FONT_WEIGHT},
        "linewidths": EDGE_WIDTH,
        "linecolor": EDGE_COLOR,
        "square": True,
    }
