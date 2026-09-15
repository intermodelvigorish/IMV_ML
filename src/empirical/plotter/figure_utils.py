"""Shared publication styles, panel layouts, PDF exports and saved-result plots.

Use configure_plotting() for the base font/palette, or rc_params() inside
mpl.rc_context() for the multiclass and SHAP panel style. Keeping these profiles
separate preserves each figure family's layout without leaking style changes.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
import os
import shutil
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class AblationExample:
    dataset: str
    title: str
    task: str
    variants: tuple[str, ...]

    @property
    def notebook_stem(self):
        return f"ablation_imv_{self.dataset}"


ABLATION_EXAMPLES = (
    AblationExample(
        "mnist", "MNIST / CNN", "Odd versus even digits",
        ("FullCNN", "NoConv2", "NoHidden", "NoDropout", "Linear"),
    ),
    AblationExample(
        "har", "UCI HAR / BiGRU", "Walking upstairs versus downstairs",
        ("FullBiGRU", "UniGRU", "OneLayer", "NoAttention", "MeanPoolMLP"),
    ),
    AblationExample(
        "magic_gamma", "MAGIC Gamma / MLP", "Gamma versus hadron events",
        ("FullFeatures", "NoGeometry", "NoConcentration", "NoMoments", "NoOrientation"),
    ),
)
ABLATION_SEEDS = tuple(range(42, 52))
# The project palette, warm end to cool end. Every figure draws from these six
# colours, either as the continuous ramp below or sampled for categorical marks.
PALETTE_COLORS = {
    "red": "#E66859",
    "cream": "#FEE7BA",
    "light_blue": "#75BBD4",
    "blue": "#74ADD1",
    "steel_blue": "#416FA0",
    "navy": "#274668",
    "green":"#74D1AF"
}
COLORMAP = "imv"
# Stops run cool to warm, so a low value reads navy and a high one red, which is
# the direction every published figure already uses. Cream sits at the midpoint:
# a diverging norm centred on zero (the ablation matrix) then puts zero on the
# neutral colour rather than part-way up a blue.
COLORMAP_STOPS = (
    (0.00, PALETTE_COLORS["navy"]),
    (0.16, PALETTE_COLORS["steel_blue"]),
    (0.30, PALETTE_COLORS["blue"]),
    (0.38, PALETTE_COLORS["light_blue"]),
    (0.50, PALETTE_COLORS["cream"]),
    (1.00, PALETTE_COLORS["red"]),
)


def _register_colormap():
    """Register the project ramp under COLORMAP so any cmap= name resolves it."""
    colormap = LinearSegmentedColormap.from_list(COLORMAP, COLORMAP_STOPS)
    # force, because a notebook that re-imports this module in a live kernel
    # would otherwise fail on the second registration.
    mpl.colormaps.register(colormap, name=COLORMAP, force=True)
    mpl.colormaps.register(colormap.reversed(), name=f"{COLORMAP}_r", force=True)
    return colormap


_register_colormap()
# Helvetica on macOS, its metric-compatible URW clone where a Linux box has it,
# and matplotlib's own font as the guaranteed last resort.
FONT_CANDIDATES = ("Helvetica", "Nimbus Sans", "DejaVu Sans")


def _installed_font_families(candidates=FONT_CANDIDATES):
    """The candidates this machine actually has, in order of preference.

    matplotlib logs `findfont: Font family 'X' not found.` once per lookup for
    every name in the chain it cannot resolve, even when a later name does
    resolve, so a machine missing one of these would print the warning for every
    figure it draws. Filtering here keeps the chain portable and the log clean.
    """
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in candidates if name in installed] or [candidates[-1]]


FONT_FAMILY = _installed_font_families()
GRID_LINESTYLE = "--"
PAPER_STYLE = {
    "font.family": FONT_FAMILY,
    "font.sans-serif": FONT_FAMILY,
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.7,
    "grid.linestyle": GRID_LINESTYLE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "image.cmap": COLORMAP,
}

# Panel geometry and styling for multiclass and SHAP figures. These are layered
# over PAPER_STYLE by rc_params(), not applied when this module is imported.
PANEL_WIDTH = 5.4
PANEL_HEIGHT = 4.6
COMPACT_PANEL_HEIGHT = 3.4   # shorter bar-chart rows in multi-dataset overviews
FIGURE_DPI = 110              # on-screen only; publication exports are PDF

PALETTE = COLORMAP
COLORS = PALETTE_COLORS
EDGE_COLOR = "#1f2a30"
EDGE_WIDTH = 0.6
BAR_EDGE_COLOR = "k"
GRID_COLOR = "#d7dcdf"
AXIS_COLOR = "#3f484d"
TEXT_COLOR = "#1f2a30"
ERROR_COLOR = "k"
CAPSIZE = 3
BAR_LABEL_FORMAT = ".3f"
BAR_LABEL_PADDING = 4
BAR_LIMIT_PADDING = 0.18
COLORBAR_WIDTH = 0.08       # fraction of the heatmap's width; no subplot resizing
COLORBAR_PADDING = 0.04
TIGHT_LAYOUT_PAD = 1.08

BASE_FONT_SIZE = PAPER_STYLE["font.size"]
TICK_FONT_SIZE = PAPER_STYLE["xtick.labelsize"]
LABEL_FONT_SIZE = PAPER_STYLE["axes.labelsize"]
TITLE_FONT_SIZE = PAPER_STYLE["axes.titlesize"]
SUPTITLE_FONT_SIZE = 13
LEGEND_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 9
MULTICLASS_ANNOTATION_FONT_SIZE = 10

TITLE_LOCATION = "left"
BODY_FONT_WEIGHT = "normal"
LABEL_FONT_WEIGHT = "medium"
TITLE_FONT_WEIGHT = "bold"
SUPTITLE_FONT_WEIGHT = "bold"

TITLE_KWARGS = {"fontsize": TITLE_FONT_SIZE, "fontweight": TITLE_FONT_WEIGHT,
                "color": TEXT_COLOR}
SUPTITLE_KWARGS = {"fontsize": SUPTITLE_FONT_SIZE, "fontweight": SUPTITLE_FONT_WEIGHT,
                   "color": TEXT_COLOR}
LABEL_KWARGS = {"fontsize": LABEL_FONT_SIZE, "fontweight": LABEL_FONT_WEIGHT,
                "color": TEXT_COLOR}


def spectral_colors(count):
    """Sample the project colormap consistently for categorical bars/lines.

    Named for the ramp this project used to carry; it samples whatever COLORMAP
    names, so the callers across the example notebooks did not have to change.
    """
    return mpl.colormaps[COLORMAP](np.linspace(0.08, 0.92, count))


def configure_plotting():
    """Apply the shared font and palette before creating notebook figures."""
    mpl.rcParams.update(PAPER_STYLE)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=spectral_colors(10))


def rc_params():
    """Panel style as an rcParams mapping, for use inside mpl.rc_context()."""
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
    """Install the panel style globally; keyword overrides are applied on top."""
    settings = {**rc_params(), **overrides}
    plt.rcParams.update(settings)
    return settings


def label_panels(axes, *, start=0):
    """Replace axes titles with bold, left-aligned letters in row-major order.

    Pass the data axes explicitly, not figure.axes, to exclude colourbars.
    Axis labels, legends, annotations and plotted values are left unchanged.
    """
    panels = np.asarray(axes, dtype=object).ravel()
    if not isinstance(start, int) or start < 0 or start + len(panels) > 26:
        raise ValueError("panel labels must fall between a. and z.")
    for index, axis in enumerate(panels, start=start):
        axis.set_title("", loc="center")
        axis.set_title("", loc="right")
        axis.set_title(f"{chr(97 + index)}.", loc="left", fontweight="bold", pad=8)


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


def bar_style(n_bars, *, colors=None):
    """Keyword arguments for a categorical bar chart with error bars."""
    return {
        "color": categorical_colors(n_bars) if colors is None else colors,
        "edgecolor": BAR_EDGE_COLOR,
        "linewidth": EDGE_WIDTH,
        "capsize": CAPSIZE,
        "error_kw": {"ecolor": ERROR_COLOR, "elinewidth": EDGE_WIDTH,
                     "capthick": EDGE_WIDTH},
    }


def annotate_bars(axis, bars, *, fontsize=ANNOTATION_FONT_SIZE):
    """Label vertical bar values beyond their error caps, including negatives."""
    if bars.orientation != "vertical":
        raise ValueError("annotate_bars expects vertical bars")
    segments = []
    if bars.errorbar is not None and bars.errorbar.has_yerr:
        segments = bars.errorbar.lines[2][-1].get_segments()
    annotations = []
    for index, rectangle in enumerate(bars.patches):
        value = rectangle.get_height()
        if not np.isfinite(value):
            continue
        positive = value >= 0
        endpoint = rectangle.get_y() + value
        if index < len(segments) and len(segments[index]):
            endpoints = segments[index][:, 1]
            endpoint = endpoints.max() if positive else endpoints.min()
        label = format(value, BAR_LABEL_FORMAT)
        if label == "-0.000":
            label = "0.000"
        annotation = axis.annotate(
            label, (rectangle.get_x() + rectangle.get_width() / 2, endpoint),
            textcoords="offset points", xytext=(0, BAR_LABEL_PADDING if positive else -BAR_LABEL_PADDING),
            ha="center", va="bottom" if positive else "top",
            fontsize=fontsize, fontweight=BODY_FONT_WEIGHT, color=TEXT_COLOR,
        )
        annotation.set_gid("bar-value")
        annotations.append(annotation)
    return annotations


def plot_bars(axis, x, values, *, yerr=None, colors=None, width=0.8,
              annotation_fontsize=ANNOTATION_FONT_SIZE):
    """Draw black-edged, annotated bars with shared error-bar and axis styling.

    Call set_bar_limits() after drawing the whole row to reserve annotation room
    across shared y axes. Passing colors preserves any meaningful sign encoding.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bar values must be one-dimensional")
    bars = axis.bar(x, values, yerr=yerr, width=width,
                    **bar_style(len(values), colors=colors))
    annotate_bars(axis, bars, fontsize=annotation_fontsize)
    axis.set_axisbelow(True)
    axis.grid(False, axis="x")
    axis.grid(True, axis="y", color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=EDGE_WIDTH)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axhline(0, color=AXIS_COLOR, linewidth=EDGE_WIDTH)
    return bars


def set_bar_limits(axes, *, padding=BAR_LIMIT_PADDING):
    """Add common headroom beyond bar/error extents without moving any axes.

    Shared y-axis groups are handled together. Negative error extents remain
    visible; an all-zero group still gets a finite, non-degenerate scale.
    """
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("bar-limit padding must be finite and positive")
    visited = set()
    for axis in np.asarray(axes, dtype=object).ravel():
        if axis in visited:
            continue
        group = axis.get_shared_y_axes().get_siblings(axis)
        visited.update(group)
        bounds = np.array([sibling.dataLim.intervaly for sibling in group])
        bounds = bounds[np.isfinite(bounds).all(axis=1)]
        if not len(bounds):
            continue
        low, high = min(0, bounds[:, 0].min()), max(0, bounds[:, 1].max())
        # axhline(0) can leave a few ulps below zero after a canvas draw. Do not
        # give an otherwise positive chart a full negative annotation margin.
        if low < 0 and abs(low) <= 8 * np.finfo(float).eps * max(abs(low), abs(high)):
            low = 0.0
        margin = padding * (high - low if high > low else 1.0)
        axis.set_ylim(low - margin if low < 0 else 0, high + margin)


def style_heatmap_axes(axis, *, xlabels=None, ylabels=None):
    """Frame a heatmap in black and repeat its class labels on opposing edges.

    Existing tick locations are retained, supporting both seaborn's cell centres
    and explicitly positioned meshes. Apply after setting ticks and labels.
    """
    if xlabels is not None:
        axis.set_xticks(axis.get_xticks(), labels=xlabels)
    if ylabels is not None:
        axis.set_yticks(axis.get_yticks(), labels=ylabels)
    axis.grid(False)
    axis.tick_params(axis="both", which="major", top=True, bottom=True,
                     left=True, right=True, labeltop=True, labelbottom=True,
                     labelleft=True, labelright=True, direction="out", length=3,
                     width=EDGE_WIDTH, color="k", labelsize=TICK_FONT_SIZE)
    for tick in axis.xaxis.get_major_ticks():
        for label, alignment in ((tick.label1, "right"), (tick.label2, "left")):
            label.set_rotation(45)
            label.set_rotation_mode("anchor")
            label.set_horizontalalignment(alignment)
    for tick in axis.yaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label2.set_rotation(0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("k")
        spine.set_linewidth(EDGE_WIDTH)


def add_heatmap_colorbar(mappable, axis, label, *, width=COLORBAR_WIDTH, pad=COLORBAR_PADDING):
    """Add a full-height vector colourbar, clearing any mirrored class labels."""
    if not np.isfinite(width) or width <= 0 or not np.isfinite(pad) or pad < 0:
        raise ValueError("colourbar width must be positive and padding nonnegative")
    colorbar_axis = axis.inset_axes([1 + pad, 0, width, 1])

    def locate_colorbar(child, renderer):
        # Measure on each draw: tight_layout changes axes widths, but text sizes
        # stay fixed. A fixed fractional gap would collide with long class names.
        bounds = axis.get_window_extent(renderer)
        right_labels = [tick.label2 for tick in axis.yaxis.get_major_ticks()
                        if tick.label2.get_visible()]
        decorations = right_labels + axis.get_xticklabels() if right_labels else []
        right = max([bounds.x1] + [text.get_window_extent(renderer).x1
                                  for text in decorations if text.get_visible()])
        return Bbox.from_bounds(right + pad * bounds.width, bounds.y0,
                                width * bounds.width, bounds.height).transformed(
                                    axis.figure.transFigure.inverted())

    colorbar_axis.set_axes_locator(locate_colorbar)
    colorbar = axis.figure.colorbar(
        mappable, cax=colorbar_axis, orientation="vertical",
    )
    colorbar.set_label(label, **LABEL_KWARGS)
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE, width=EDGE_WIDTH, length=3, colors=AXIS_COLOR)
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor("k")
    colorbar.outline.set_linewidth(EDGE_WIDTH)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    return colorbar


def heatmap_style(n_classes, *, annotation_fontsize=MULTICLASS_ANNOTATION_FONT_SIZE):
    """Keyword arguments for an annotated square heatmap of `n_classes` classes.

    Larger matrices lose a decimal, not font size, so seven-class annotations
    remain legible at manuscript scale.
    """
    crowded = n_classes > 5
    return {
        "cmap": sequential_cmap(),
        "annot": True,
        "fmt": ".2f" if crowded else ".3f",
        "annot_kws": {"fontsize": annotation_fontsize,
                      "fontweight": BODY_FONT_WEIGHT},
        "linewidths": EDGE_WIDTH,
        "linecolor": EDGE_COLOR,
        "square": True,
    }


def plot_multiclass_overview(results, model_names, *, figsize=None):
    """Stack dataset blocks: pairwise heatmaps above one-vs-rest bars.

    Columns retain model order. Each dataset keeps its own shared heatmap scale,
    including negative IMV, and larger groups use compact rows for the supplement.
    Values and seed standard deviations are read directly from the saved results.
    """
    results, model_names = tuple(results), tuple(model_names)
    if not results or not model_names:
        raise ValueError("multiclass figures need at least one dataset and model")
    if 2 * len(results) * len(model_names) > 26:
        raise ValueError("multiclass panel labels must fall between a. and z.")
    if figsize is None:
        height = PANEL_HEIGHT if len(results) == 1 else COMPACT_PANEL_HEIGHT
        figsize = figure_size(2 * len(results), len(model_names), height=height)
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(2 * len(results), len(model_names),
                               height_ratios=[1, 0.85] * len(results))
    axes = np.empty((2 * len(results), len(model_names)), dtype=object)
    for block, result in enumerate(results):
        row = 2 * block
        labels = result["class_names"]
        matrices = [result["pairwise"][name].to_numpy(dtype=float) for name in model_names]
        low = min(0.0, min(float(np.nanmin(matrix)) for matrix in matrices))
        high = max(float(np.nanmax(matrix)) for matrix in matrices)
        style = {**heatmap_style(len(labels)), "square": False}
        for column, name in enumerate(model_names):
            heat = figure.add_subplot(grid[row, column])
            bars = figure.add_subplot(grid[row + 1, column])
            axes[row, column], axes[row + 1, column] = heat, bars
            sns.heatmap(result["pairwise"][name], ax=heat, vmin=low, vmax=high,
                        cbar=False, **style)
            style_heatmap_axes(heat, xlabels=labels, ylabels=labels)
            part = result["ova_summary"].loc[result["ova_summary"].model == name].sort_values("class")
            plot_bars(bars, labels, part["mean"], yerr=part["std"].fillna(0),
                      annotation_fontsize=MULTICLASS_ANNOTATION_FONT_SIZE)
            bars.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        add_heatmap_colorbar(axes[row, 0].collections[0], axes[row, -1], "Pairwise IMV")
        axes[row + 1, 0].set_ylabel(f"One-vs-rest IMV (mean of {result['n_seeds']} seeds)",
                                    **LABEL_KWARGS)
        set_bar_limits(axes[row + 1])
    label_panels(axes)
    return figure


def plot_ablation_matrix(matrix, *, figsize=(6, 6), cmap=COLORMAP):
    """Style imvpy's heatmap without changing its values or normalization."""
    from imvpy.utils import plot_ablation_matrix as _imvpy_ablation_matrix

    # The pinned imvpy plot helper hardcodes its palette and has no cmap argument.
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = _imvpy_ablation_matrix(matrix, figsize=figsize, title="")
        label_panels(ax)
        mesh = ax.collections[0]
        mesh.set_cmap(cmap)
        for annotation, value in zip(ax.texts, np.asarray(matrix).flat):
            rgb = mesh.cmap(mesh.norm(value))[:3]
            luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            annotation.set_color("white" if luminance < 0.48 else "#202020")
        colorbar_label = mesh.colorbar.ax.get_ylabel()
        mesh.colorbar.remove()
        add_heatmap_colorbar(mesh, ax, colorbar_label or "Directional IMV")
    return fig, ax


def find_project_root(start=None):
    """Find this repository from the root or any nested notebook directory."""
    start = Path.cwd() if start is None else Path(start)
    start = start.expanduser().resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if ((directory / "requirements.txt").is_file()
                and (directory / "src" / "empirical" / "plotter" / "figure_utils.py").is_file()):
            return directory
    raise FileNotFoundError(f"Cannot locate the IMV_ML project above {start}")


def artifact_directory():
    """Honor the same cache settings as the source example notebooks."""
    cache = Path(os.environ.get("IMV_CACHE_HOME", Path.home() / ".cache" / "imv"))
    return Path(os.environ.get("IMV_ARTIFACT_CACHE", cache / "notebook_artifacts")).expanduser()


def figure_directory(project_root):
    """Resolve relative output overrides against the repository, not kernel cwd."""
    directory = Path(os.environ.get("IMV_FIGURE_DIR", "output/figures")).expanduser()
    return directory if directory.is_absolute() else Path(project_root) / directory


@dataclass
class AblationResults:
    example: AblationExample
    seeds: tuple[int, ...]
    pairwise_mean: pd.DataFrame
    pairwise_std: pd.DataFrame
    null_by_seed: pd.DataFrame
    null_summary: pd.DataFrame
    source_paths: tuple[Path, Path]


def _require_complete(table, columns, keys, expected_index, path):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError(f"{path.name}: missing or duplicate keys {keys}")
    actual = pd.MultiIndex.from_frame(table[keys])
    if len(actual) != len(expected_index) or set(actual) != set(expected_index):
        raise ValueError(
            f"{path.name}: incomplete or unexpected seed/variant coverage; "
            "rerun the corresponding source notebook to completion"
        )


def load_ablation_results(example, artifact_root=None, *, project_root=None, seeds=ABLATION_SEEDS):
    """Load a complete paired experiment and compute each model's null IMV.

    Section 4.4, Eq. (10) of the paper uses constant p=0.5. The source
    diagnostics already store each seed/model's geometric mean likelihood,
    so imvpy can calculate this score without loading or refitting a model.
    Transform each seed before averaging: IMV is nonlinear in likelihood.
    With project_root, prefer the published CSV pair. An incomplete published
    pair is an error, never an invitation to mix it with cached results.
    """
    from imvpy import AblationIMV, imv_from_likelihoods

    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required for sample SD")
    root = artifact_directory() if artifact_root is None else Path(artifact_root).expanduser()
    results = root / example.notebook_stem / "results"
    if project_root is not None:
        published = Path(project_root).expanduser() / "output/examples/ablation_imv"
        if any((published / f"{example.dataset}_{suffix}.csv").is_file()
               for suffix in ("ablation_imv_by_seed", "variant_diagnostics")):
            results = published
    pair_path = results / f"{example.dataset}_ablation_imv_by_seed.csv"
    diag_path = results / f"{example.dataset}_variant_diagnostics.csv"
    for path in (pair_path, diag_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing result: {path}\nRun src/empirical/ablation_imv/"
                f"{example.notebook_stem}.ipynb first, restore its complete published CSV pair, "
                "or set IMV_ARTIFACT_CACHE when no published pair exists."
            )
    pairwise = pd.read_csv(pair_path)
    diagnostics = pd.read_csv(diag_path)
    variants = example.variants
    pair_index = pd.MultiIndex.from_product(
        [seeds, variants, variants], names=["seed", "enhanced", "basic"]
    )
    diag_index = pd.MultiIndex.from_product([seeds, variants], names=["seed", "variant"])
    _require_complete(
        pairwise, [*pair_index.names, "ablation_imv"],
        list(pair_index.names), pair_index, pair_path,
    )
    _require_complete(
        diagnostics, [*diag_index.names, "geometric_mean_likelihood"],
        list(diag_index.names), diag_index, diag_path,
    )
    pair_values = pairwise.set_index(list(pair_index.names)).loc[pair_index, "ablation_imv"]
    likelihoods = diagnostics.set_index(list(diag_index.names)).loc[
        diag_index, "geometric_mean_likelihood"
    ]
    if not np.isfinite(pair_values.to_numpy(dtype=float)).all():
        raise ValueError(f"{pair_path.name}: non-finite IMV scores")
    if not (np.isfinite(likelihoods.to_numpy(dtype=float)).all()
            and likelihoods.gt(0).all() and likelihoods.le(1).all()):
        raise ValueError(f"{diag_path.name}: likelihoods must be finite and in (0, 1]")

    matrices, null_rows = [], []
    for seed in seeds:
        matrix = pair_values.loc[seed].unstack("basic").reindex(index=variants, columns=variants)
        if not np.allclose(np.diag(matrix), 0, atol=1e-12):
            raise ValueError(f"{pair_path.name}: seed {seed} has a nonzero diagonal")
        a = likelihoods.loc[seed]
        # Cross-check both CSVs to avoid mixing results from different runs.
        expected = np.array([
            [imv_from_likelihoods(a[basic], a[enhanced]) for basic in variants]
            for enhanced in variants
        ])
        if not np.allclose(matrix, expected, rtol=1e-7, atol=1e-8):
            raise ValueError(
                f"{example.notebook_stem}: pairwise scores and likelihoods disagree "
                f"for seed {seed}; check result files and the imvpy version"
            )
        matrices.append(matrix)
        null_rows.extend(
            {"seed": seed, "variant": variant,
             "imv_vs_null": imv_from_likelihoods(0.5, a[variant])}
            for variant in variants
        )

    mean = AblationIMV.average_imv_matrices(matrices)
    std = pd.DataFrame(
        np.std(np.stack([matrix.to_numpy() for matrix in matrices]), axis=0, ddof=1),
        index=variants, columns=variants,
    )
    null_by_seed = pd.DataFrame(null_rows)
    null_summary = (null_by_seed.groupby("variant")["imv_vs_null"]
                    .agg(["mean", "std", "count"]).reindex(variants))
    return AblationResults(example, seeds, mean, std, null_by_seed, null_summary,
                           (pair_path, diag_path))


def export_ablation_results(example, artifact_root=None, *, project_root=None, seeds=ABLATION_SEEDS):
    """Publish compact, validated figure inputs; leave prediction traces cached."""
    result = load_ablation_results(example, artifact_root, seeds=seeds)
    project_root = find_project_root() if project_root is None else Path(project_root).expanduser()
    destination = project_root / "output/examples/ablation_imv"
    full = example.variants[0]
    full_comparison = pd.DataFrame({
        "mean": result.pairwise_mean.loc[full], "std": result.pairwise_std.loc[full],
    }).reindex(example.variants[1:])
    full_comparison.index.name = "basic"
    summaries = {
        "ablation_imv_directional": (result.pairwise_mean, True),
        "ablation_imv_directional_std": (result.pairwise_std, True),
        "full_vs_ablation": (full_comparison, True),
        "imv_vs_null_by_seed": (result.null_by_seed, False),
        "imv_vs_null_summary": (result.null_summary, True),
    }
    destination.mkdir(parents=True, exist_ok=True)
    # Prepare every file before replacing any existing published result. Each
    # replacement is atomic; the loader detects an interrupted mixed CSV pair.
    with tempfile.TemporaryDirectory(prefix=f".{example.dataset}-export-", dir=destination) as temporary:
        stage = Path(temporary)
        for source in result.source_paths:
            shutil.copyfile(source, stage / source.name)
        for suffix, (table, index) in summaries.items():
            table.to_csv(stage / f"{example.dataset}_{suffix}.csv", index=index)
        names = sorted(path.name for path in stage.iterdir())
        for name in names:
            os.replace(stage / name, destination / name)
    return tuple(destination / name for name in names)


def plot_ablation_overview(results, *, figsize=(15.5, 9.0)):
    """A 2x3 paper-style comparison: directional heatmaps above null-IMV bars.

    Color normalization is shared across the heatmaps and centered on zero.
    Bars retain the same variant order and y limits in all three examples.
    Error bars are sample SD across seeds, not a confidence interval.
    """
    results = tuple(results)
    if len(results) != 3:
        raise ValueError("The 2x3 overview requires exactly three ablation examples")
    if any(result.seeds != results[0].seeds for result in results):
        raise ValueError("All three examples must use the same seeds")
    magnitude = max(float(np.abs(result.pairwise_mean.to_numpy()).max()) for result in results)
    limit = max(0.05, np.ceil(magnitude / 0.05) * 0.05)
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = mpl.colormaps[COLORMAP]

    with mpl.rc_context(PAPER_STYLE):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2, 3, height_ratios=[1, 0.78],
        )
        axes = np.empty((2, 3), dtype=object)
        for column, result in enumerate(results):
            example = result.example
            labels = example.variants
            x = np.arange(len(labels))
            heatmap = fig.add_subplot(grid[0, column])
            axes[0, column] = heatmap
            values = result.pairwise_mean.to_numpy()
            # Keep cells as vector paths: embedded imshow rasters can break in PDF/SVG.
            edges = np.arange(len(labels) + 1) - 0.5
            mesh = heatmap.pcolormesh(
                edges, edges, values, cmap=cmap, norm=norm, shading="flat",
                edgecolors="face", linewidth=0.2, antialiased=False, rasterized=False,
            )
            mesh.set_gid(f"ablation_heatmap_{column}")
            heatmap.set_xlim(edges[0], edges[-1])
            heatmap.set_ylim(edges[-1], edges[0])
            for (row, col), value in np.ndenumerate(values):
                rgb = cmap(norm(value))[:3]
                luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
                text = "0.000" if abs(value) < 0.0005 else f"{value:.3f}"
                heatmap.text(col, row, text, ha="center", va="center", fontsize=9,
                             color="white" if luminance < 0.48 else "#202020")
            heatmap.set_xticks(x, labels, rotation=42, ha="right", rotation_mode="anchor")
            heatmap.set_yticks(x, labels)
            heatmap.tick_params(length=0, pad=4)
            heatmap.set_xlabel("Basic model (column)")
            if column == 0:
                heatmap.set_ylabel("Enhanced model (row)")
            for spine in heatmap.spines.values():
                spine.set_visible(False)

            bars = fig.add_subplot(grid[1, column], sharey=axes[1, 0] if column else None)
            axes[1, column] = bars
            plot_bars(bars, x, result.null_summary["mean"], yerr=result.null_summary["std"],
                      colors=spectral_colors(len(labels)), width=0.73)
            bars.set_xticks(x, labels, rotation=42, ha="right", rotation_mode="anchor")
            bars.set_xlim(-0.6, len(labels) - 0.4)
            bars.yaxis.set_major_locator(MultipleLocator(0.2))
            bars.set_xlabel("Model variant")
            if column == 0:
                bars.set_ylabel("Mean IMV versus 0.5 baseline")
            else:
                bars.tick_params(labelleft=False)

        set_bar_limits(axes[1])
        add_heatmap_colorbar(axes[0, 0].collections[0], axes[0, -1], "Mean directional IMV")
        fig.align_xlabels(axes[0, :])
        fig.align_xlabels(axes[1, :])
        label_panels(axes)
    return fig, axes


def apply_tight_layout(figure):
    """Fit panels and inset colourbars, reserving room for bottom figure legends.

    Matplotlib includes inset colourbars in their parent axes' tight bounds, but
    not figure-level legends. Measure those before fitting the subplot area.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legends = [legend for legend in figure.legends
               if legend.get_visible() and legend.get_in_layout()]
    bottom = 0.0
    if legends:
        bottom = max(legend.get_window_extent(renderer).transformed(
            figure.transFigure.inverted()).y1 for legend in legends)
        bottom += TIGHT_LAYOUT_PAD * mpl.rcParams["font.size"] / 72 / figure.get_figheight()
    figure.tight_layout(pad=TIGHT_LAYOUT_PAD, rect=(0, bottom, 1, 1))
    return figure


def save_publication_figure(figure, destination, *, dpi=800):
    """Apply tight layout and write only a PDF with the expected path mapping.

    imvpy's save_figure always writes three formats, so use Matplotlib directly
    for export; all IMV calculations still use imvpy.
    """
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    path = Path(destination).expanduser()
    if path.suffix.lower() in {".png", ".svg", ".pdf"}:
        path = path.with_suffix("")
    path = Path(f"{path}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Embedding macOS's Helvetica subsets a .ttc whose post table and creation
    # date trip two fontTools warnings on every export; neither says anything
    # about the figure, and they would otherwise land in every notebook output.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    with mpl.rc_context(PAPER_STYLE):
        apply_tight_layout(figure)
        figure.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    # Retire only this figure's legacy exports, and only after the PDF succeeds.
    for suffix in (".png", ".svg"):
        path.with_suffix(suffix).unlink(missing_ok=True)
    return {"pdf": path}
