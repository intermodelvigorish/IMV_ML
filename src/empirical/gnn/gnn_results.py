"""Validate and plot saved GNN learning curves, without loading graph data."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotter.figure_utils import PAPER_STYLE, COLORMAP, artifact_directory, label_panels


DATASET_ORDER = ("PROTEINS", "NCI1", "NCI109", "Mutagenicity", "AIDS", "DD")
CURVE_METRICS = ("imv", "accuracy", "precision", "recall")
CURVE_STYLE = {
    **PAPER_STYLE,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "path.simplify": False,
}


def summarize_curves(table, *, seeds=tuple(range(42, 52))):
    """Means and observed seed extrema at each epoch, not confidence intervals."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least two distinct seeds")
    keys = ["dataset", "split", "epoch", "seed"]
    if table.duplicated(keys).any() or set(table.seed) != set(seeds):
        raise ValueError("Unexpected or duplicate seeds in learning curves")
    metrics = [*CURVE_METRICS, "balanced_accuracy", "roc_auc", "f1", "brier", "log_loss"]
    rows = []
    for key, group in table.groupby(keys[:-1], sort=False):
        if set(group.seed) != set(seeds):
            raise ValueError(f"Incomplete seed coverage at {key}")
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            # Do not silently condition an IMV average on only the successful seeds.
            mean = float(values.mean()) if finite.all() else np.nan
            sd = float(values.std(ddof=1)) if finite.all() else np.nan
            low = float(values.min()) if finite.all() else np.nan
            high = float(values.max()) if finite.all() else np.nan
            rows.append(dict(zip(keys[:-1], key)) | {
                "metric": metric, "mean": mean, "std": sd, "range_low": low,
                "range_high": high, "n_seeds": len(seeds), "n_valid": int(finite.sum()),
            })
    return pd.DataFrame(rows)


@dataclass
class GNNResults:
    epochs: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict
    directory: Path


def load_gnn_results(artifact_root=None):
    root = artifact_directory() if artifact_root is None else Path(artifact_root)
    directory = root / "gnn_training" / "results"
    manifest_path, table_path = directory / "manifest.json", directory / "epoch_metrics.csv"
    if not manifest_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(
            f"Missing completed GNN results in {directory}. "
            "Execute src/empirical/gnn/gnn_training.ipynb first (six datasets, ten seeds, 200 epochs)."
        )
    manifest = json.loads(manifest_path.read_text())
    if (not manifest.get("production") or manifest["datasets"] != list(DATASET_ORDER)
            or manifest["seeds"] != list(range(42, 52)) or manifest["config"]["epochs"] != 200):
        raise ValueError("The publication plot requires the full six-dataset, ten-seed, 200-epoch run")
    if hashlib.sha256(table_path.read_bytes()).hexdigest() != manifest["metrics_sha256"]:
        raise ValueError("GNN metric CSV does not match its completion manifest")
    table = pd.read_csv(table_path)
    keys = ["dataset", "seed", "epoch", "split"]
    required = keys + list(CURVE_METRICS) + ["imv_status", "n_graphs"]
    if not set(required) <= set(table):
        raise ValueError(f"Missing GNN metric columns: {sorted(set(required) - set(table))}")
    expected = pd.MultiIndex.from_product(
        [DATASET_ORDER, range(42, 52), range(201), ("validation", "test")], names=keys
    )
    if table.duplicated(keys).any() or set(pd.MultiIndex.from_frame(table[keys])) != set(expected):
        raise ValueError("Incomplete or duplicated GNN dataset/seed/epoch/split coverage")
    for metric in CURVE_METRICS[1:]:
        if not table[metric].between(0, 1).all():
            raise ValueError(f"Non-finite or out-of-range {metric}")
    if not np.array_equal(table.imv.isna(), table.imv_status.eq("undefined")):
        raise ValueError("Undefined IMV values must be marked explicitly")
    if np.isinf(table.imv.to_numpy()).any() or not table.n_graphs.gt(0).all():
        raise ValueError("Invalid IMV values or evaluation sizes")
    return GNNResults(table, summarize_curves(table), manifest, directory)


def plot_gnn_learning_curves(results, *, figsize=(15.5, 8.5)):
    """Mean curves with shaded per-epoch min-to-max seed ranges."""
    summary = results.summary.query("split == 'test'")
    palette = mpl.colormaps[COLORMAP]
    styles = {
        "imv": (palette(0.96), "-", 1.25, "Mean IMV"),
        "accuracy": (palette(0.05), "--", 0.85, "Mean accuracy"),
        "precision": (palette(0.23), "-.", 0.85, "Mean precision"),
        "recall": (palette(0.78), ":", 1.0, "Mean recall"),
    }
    plotted = summary[summary.metric.isin(CURVE_METRICS)]
    low = min(-0.02, float(plotted.range_low.min()) - 0.025)
    high = max(1.02, float(plotted.range_high.max()) + 0.025)
    with mpl.rc_context(CURVE_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        for index, (name, ax) in enumerate(zip(DATASET_ORDER, axes.flat)):
            for metric, (color, linestyle, width, label) in styles.items():
                curve = summary[(summary.dataset == name) & (summary.metric == metric)].sort_values("epoch")
                x = curve.epoch.to_numpy()
                ax.fill_between(x, curve.range_low.to_numpy(), curve.range_high.to_numpy(), color=color,
                                alpha=0.18 if metric == "imv" else 0.09, linewidth=0)
                ax.plot(x, curve["mean"].to_numpy(), color=color, linestyle=linestyle,
                        linewidth=width, label=label, zorder=4 if metric == "imv" else 3)
            ax.set_xlim(0, 200)
            ax.set_ylim(low, high)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.axhline(0, color="0.65", linewidth=0.6)
            ax.grid(True, linewidth=0.6, color="0.8")
            ax.spines[["top", "right"]].set_visible(False)
            if index % 3 == 0:
                ax.set_ylabel("Held-out test metric")
            if index >= 3:
                ax.set_xlabel("Epoch")
        label_panels(axes)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4,
                   frameon=True, edgecolor="k", facecolor="white", framealpha=1,
                   fancybox=False, handlelength=3)
    return fig, axes


def plot_gnn_seed_curves(results, *, metric="imv", figsize=(15.5, 8.5)):
    """Expose individual trajectories without averaging away seed variation."""
    if metric not in CURVE_METRICS:
        raise ValueError(f"metric must be one of {CURVE_METRICS}")
    raw = results.epochs.query("split == 'test'")
    seeds = results.manifest["seeds"]
    palette = mpl.colormaps[COLORMAP]
    # Avoid the near-white centre of Spectral_r for thin lines on white axes.
    positions = np.r_[np.linspace(0.02, 0.30, (len(seeds) + 1) // 2),
                      np.linspace(0.70, 0.98, len(seeds) // 2)]
    ylabel = "IMV" if metric == "imv" else metric.capitalize()
    with mpl.rc_context(CURVE_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=False)
        for index, (name, ax) in enumerate(zip(DATASET_ORDER, axes.flat)):
            for seed, position in zip(seeds, positions):
                trajectory = raw[(raw.dataset == name) & (raw.seed == seed)].sort_values("epoch")
                ax.plot(trajectory.epoch, trajectory[metric], color=palette(position),
                        linewidth=0.7, alpha=0.85, label=f"Seed {seed}")
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(True, linewidth=0.6, color="0.8")
            ax.spines[["top", "right"]].set_visible(False)
            if index >= 3:
                ax.set_xlabel("Epoch")
        label_panels(axes)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=5, frameon=True, edgecolor="k", facecolor="white", framealpha=1,
                   fancybox=False, handlelength=3)
    return fig, axes
