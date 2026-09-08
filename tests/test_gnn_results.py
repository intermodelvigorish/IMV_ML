"""Publication completeness and figure checks on small generated test fixtures."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from empirical.plotter.figure_utils import save_publication_figure
from empirical.gnn.gnn_results import (
    CURVE_METRICS, DATASET_ORDER, load_gnn_results, plot_gnn_learning_curves,
    plot_gnn_seed_curves,
)


class GNNResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "gnn_training" / "results"
        self.directory.mkdir(parents=True)
        self.table = pd.MultiIndex.from_product(
            [DATASET_ORDER, range(42, 52), range(201), ["validation", "test"]],
            names=["dataset", "seed", "epoch", "split"],
        ).to_frame(index=False)
        self.table["imv"] = .1 + self.table.epoch / 1000 + (self.table.seed - 42) / 1000
        for index, metric in enumerate(("accuracy", "precision", "recall")):
            self.table[metric] = .6 + self.table.epoch / 1000 + (self.table.seed - 42) ** 2 / (1000 * (index + 1))
        self.table["imv_status"] = "defined"
        self.table["n_graphs"] = 150
        self.manifest = {"production": True, "datasets": list(DATASET_ORDER), "seeds": list(range(42, 52)),
                         "config": {"epochs": 200},
                         "inventory": [{"dataset": name, "graphs": 1000} for name in DATASET_ORDER]}
        self.write()

    def write(self):
        path = self.directory / "epoch_metrics.csv"
        self.table.to_csv(path, index=False)
        self.manifest["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def assert_clean_panel_layout(self, fig, axes):
        self.assertFalse(fig.texts)
        for index, ax in enumerate(axes.flat):
            self.assertEqual(ax.get_title(loc="left"), f"{chr(97 + index)}.")
            self.assertEqual(ax._left_title.get_fontweight(), "bold")
            self.assertEqual(ax._left_title.get_fontsize(), 15)
            self.assertEqual(ax.xaxis.label.get_fontsize(), 13)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 13)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                self.assertEqual(tick.get_fontsize(), 12)
            self.assertEqual(ax.get_title(loc="center"), "")
            self.assertEqual(ax.get_title(loc="right"), "")
            self.assertFalse(ax.texts)
        self.assertEqual(len(fig.legends), 1)
        self.assertTrue(fig.legends[0].get_frame_on())
        np.testing.assert_array_equal(fig.legends[0].get_frame().get_edgecolor(), [0, 0, 0, 1])
        for label in fig.legends[0].get_texts():
            self.assertEqual(label.get_fontsize(), 12)

    def test_complete_curves_plot_six_panels_and_only_pdf(self):
        results = load_gnn_results(self.root)
        fig, axes = plot_gnn_learning_curves(results)
        self.addCleanup(plt.close, fig)
        self.assertEqual(axes.shape, (2, 3))
        for name, ax in zip(DATASET_ORDER, axes.flat):
            self.assertEqual(len(ax.lines), 5)  # Four means and the zero reference.
            self.assertEqual(len(ax.collections), 4)
            imv_line = next(line for line in ax.lines if line.get_label() == "Mean IMV")
            expected = results.summary.query("dataset == @name and split == 'test' and metric == 'imv'")
            np.testing.assert_allclose(imv_line.get_ydata(), expected["mean"])
            self.assertEqual(ax.title.get_fontfamily()[0], "Helvetica")
            self.assertEqual(ax.get_xlim(), (0, 200))
            self.assertFalse(imv_line.get_path().should_simplify)
            for metric, band in zip(CURVE_METRICS, ax.collections):
                raw = self.table.query("dataset == @name and split == 'test'")
                vertices = band.get_paths()[0].vertices
                for epoch, values in raw.groupby("epoch")[metric]:
                    bounds = vertices[vertices[:, 0] == epoch, 1]
                    self.assertAlmostEqual(bounds.min(), values.min())
                    self.assertAlmostEqual(bounds.max(), values.max())
        self.assert_clean_panel_layout(fig, axes)
        paths = save_publication_figure(fig, self.root / "overview")
        self.assertEqual(set(paths), {"pdf"})
        self.assertNotIn(b"/Subtype /Image", paths["pdf"].read_bytes())
        for ax in axes.flat:
            for line in ax.lines:
                if len(line.get_xdata()) == 201:
                    self.assertFalse(line.get_path().should_simplify)

    def test_every_mean_epoch_is_drawn_without_path_simplification(self):
        # Alternating spikes and a missing value must survive as-is in the plot.
        mask = (self.table.seed == 42) & (self.table.epoch % 2 == 0)
        self.table.loc[mask, "imv"] -= .09
        self.table.loc[1, ["imv", "imv_status"]] = [np.nan, "undefined"]
        self.write()
        results = load_gnn_results(self.root)
        with matplotlib.rc_context({"path.simplify": True}):
            fig, axes = plot_gnn_learning_curves(results)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        for name, ax in zip(DATASET_ORDER, axes.flat):
            for metric in CURVE_METRICS:
                label = "Mean IMV" if metric == "imv" else f"Mean {metric}"
                line = next(line for line in ax.lines if line.get_label() == label)
                trajectories = self.table.query("dataset == @name and split == 'test'").pivot(
                    index="epoch", columns="seed", values=metric,
                ).sort_index()
                np.testing.assert_array_equal(line.get_xdata(), trajectories.index)
                np.testing.assert_allclose(line.get_ydata(), trajectories.to_numpy().mean(axis=1), equal_nan=True)
                self.assertEqual(len(line.get_path().vertices), 201)
                self.assertFalse(line.get_path().should_simplify)

    def test_seed_detail_preserves_individual_trajectories_and_axis_labels(self):
        results = load_gnn_results(self.root)
        for metric in CURVE_METRICS:
            fig, axes = plot_gnn_seed_curves(results, metric=metric)
            self.addCleanup(plt.close, fig)
            for name, ax in zip(DATASET_ORDER, axes.flat):
                self.assertEqual(len(ax.lines), 10)
                self.assertEqual(len(ax.get_shared_y_axes().get_siblings(ax)), 1)
                for seed, line in zip(self.manifest["seeds"], ax.lines):
                    expected = self.table.query("dataset == @name and seed == @seed and split == 'test'")
                    np.testing.assert_array_equal(line.get_xdata(), expected.epoch)
                    np.testing.assert_allclose(line.get_ydata(), expected[metric])
                    self.assertFalse(line.get_path().should_simplify)
            self.assert_clean_panel_layout(fig, axes)
        with self.assertRaises(ValueError):
            plot_gnn_seed_curves(results, metric="not_a_metric")

    def test_missing_epoch_rejected_even_with_matching_hash(self):
        self.table = self.table.iloc[:-1]
        self.write()
        with self.assertRaisesRegex(ValueError, "coverage"):
            load_gnn_results(self.root)

    def test_smoke_run_cannot_be_a_publication_figure(self):
        self.manifest["production"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "publication"):
            load_gnn_results(self.root)

    def test_changed_csv_requires_matching_completion_manifest(self):
        with (self.directory / "epoch_metrics.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_gnn_results(self.root)

    def test_undefined_imv_is_retained_as_a_gap(self):
        self.table.loc[0, "imv"] = np.nan
        self.table.loc[0, "imv_status"] = "undefined"
        self.write()
        result = load_gnn_results(self.root)
        row = result.summary.query("dataset == 'PROTEINS' and epoch == 0 and split == 'validation' and metric == 'imv'").iloc[0]
        self.assertEqual(row.n_valid, 9)
        self.assertTrue(np.isnan(row["mean"]))
        self.assertTrue(np.isnan(row.range_low))
        self.assertTrue(np.isnan(row.range_high))


if __name__ == "__main__":
    unittest.main()
