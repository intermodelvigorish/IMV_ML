"""Publication plots built from saved IMV results, without training models."""

from dataclasses import dataclass
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from imvpy import AblationIMV, imv_from_likelihoods
from imvpy.utils import plot_ablation_matrix as _imvpy_ablation_matrix


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
COLORMAP = "Spectral_r"
PAPER_STYLE = {
    "font.family": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "image.cmap": COLORMAP,
}


def spectral_colors(count):
    """Sample the project colormap consistently for categorical bars/lines."""
    return mpl.colormaps[COLORMAP](np.linspace(0.08, 0.92, count))


def configure_plotting():
    """Apply the shared font and palette before creating notebook figures."""
    mpl.rcParams.update(PAPER_STYLE)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=spectral_colors(10))


def plot_ablation_matrix(matrix, *, figsize=(6, 6), title="Ablation IMV matrix",
                         cmap=COLORMAP):
    """Style imvpy's heatmap without changing its values or normalization."""
    # The pinned imvpy plot helper hardcodes its palette and has no cmap argument.
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = _imvpy_ablation_matrix(matrix, figsize=figsize, title=title)
        mesh = ax.collections[0]
        mesh.set_cmap(cmap)
        for annotation, value in zip(ax.texts, np.asarray(matrix).flat):
            rgb = mesh.cmap(mesh.norm(value))[:3]
            luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            annotation.set_color("white" if luminance < 0.48 else "#202020")
        mesh.colorbar.solids.set_rasterized(False)
        mesh.colorbar.solids.set_edgecolor("face")
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


def load_ablation_results(example, artifact_root=None, *, seeds=ABLATION_SEEDS):
    """Load a complete paired experiment and compute each model's null IMV.

    Section 4.4, Eq. (10) of the paper uses constant p=0.5. The source
    diagnostics already store each seed/model's geometric mean likelihood,
    so imvpy can calculate this score without loading or refitting a model.
    Transform each seed before averaging: IMV is nonlinear in likelihood.
    """
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required for sample SD")
    root = artifact_directory() if artifact_root is None else Path(artifact_root).expanduser()
    results = root / example.notebook_stem / "results"
    pair_path = results / f"{example.dataset}_ablation_imv_by_seed.csv"
    diag_path = results / f"{example.dataset}_variant_diagnostics.csv"
    for path in (pair_path, diag_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing result: {path}\nRun src/empirical/ablation_imv/"
                f"{example.notebook_stem}.ipynb first, or set IMV_ARTIFACT_CACHE."
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
    lower = min(float((r.null_summary["mean"] - r.null_summary["std"]).min()) for r in results)
    upper = max(float((r.null_summary["mean"] + r.null_summary["std"]).max()) for r in results)
    y_limits = (min(0.0, lower - 0.04), max(1.04, upper + 0.04))

    with mpl.rc_context(PAPER_STYLE):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2, 4, width_ratios=[1, 1, 1, 0.045], height_ratios=[1, 0.78],
            left=0.073, right=0.954, bottom=0.21, top=0.91, wspace=0.72, hspace=0.83,
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
            heatmap.set_title(f"({chr(97 + column)})  {example.title}", pad=26)
            heatmap.text(0.5, 1.045, example.task, transform=heatmap.transAxes,
                         ha="center", fontsize=10, color="0.3")
            for spine in heatmap.spines.values():
                spine.set_visible(False)

            bars = fig.add_subplot(grid[1, column], sharey=axes[1, 0] if column else None)
            axes[1, column] = bars
            bars.bar(x, result.null_summary["mean"], yerr=result.null_summary["std"],
                     color=spectral_colors(len(labels)), width=0.73, capsize=3,
                     error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"})
            bars.set_xticks(x, labels, rotation=42, ha="right", rotation_mode="anchor")
            bars.set_ylim(*y_limits)
            bars.set_xlim(-0.6, len(labels) - 0.4)
            bars.yaxis.set_major_locator(MultipleLocator(0.2))
            bars.set_axisbelow(True)
            bars.grid(axis="y", linestyle="-.", color="0.72", linewidth=0.6)
            bars.axhline(0, linewidth=0.7, color="0.25")
            bars.set_title(f"({chr(100 + column)})  Full and ablated models", pad=10)
            bars.set_xlabel("Model variant")
            if column == 0:
                bars.set_ylabel("Mean IMV versus 0.5 baseline")
            else:
                bars.tick_params(labelleft=False)
            bars.spines[["top", "right"]].set_visible(False)

        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                               cax=fig.add_subplot(grid[0, 3]))
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.set_label("Mean directional IMV", fontsize=10)
        colorbar.outline.set_visible(False)
        fig.align_xlabels(axes[0, :])
        fig.align_xlabels(axes[1, :])
        fig.text(
            0.5, 0.028,
            f"Cells: row model relative to column model.  "
            f"Bars: constant p = 0.5 baseline.  "
            f"Whiskers: +/- 1 SD across {len(results[0].seeds)} seeds.",
            ha="center", fontsize=10,
        )
    return fig, axes


def save_publication_figure(figure, destination, *, dpi=800):
    """Write only a PDF, retaining the format-to-path mapping notebooks expect.

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
    with mpl.rc_context(PAPER_STYLE):
        figure.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    return {"pdf": path}
