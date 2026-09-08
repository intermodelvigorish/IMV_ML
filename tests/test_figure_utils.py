"""Scientific plotting checks using small saved-result fixtures, without training."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from imvpy import imv_from_likelihoods

from src.figure_utils import (
    COLORMAP, AblationExample, configure_plotting, load_ablation_results,
    plot_ablation_matrix, plot_ablation_overview, save_publication_figure, spectral_colors,
)


class AblationPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.example = AblationExample("fixture", "Test model", "Test task", ("Full", "Ablated"))
        self.directory = self.root / self.example.notebook_stem / "results"
        self.directory.mkdir(parents=True)
        self.diag_path = self.directory / "fixture_variant_diagnostics.csv"
        self.pair_path = self.directory / "fixture_ablation_imv_by_seed.csv"
        likelihoods = {42: {"Full": 0.6, "Ablated": 0.7},
                       43: {"Full": 0.95, "Ablated": 0.55}}
        pd.DataFrame([
            {"seed": seed, "variant": variant, "geometric_mean_likelihood": value}
            for seed, variants in likelihoods.items() for variant, value in variants.items()
        ]).to_csv(self.diag_path, index=False)
        pd.DataFrame([
            {"seed": seed, "enhanced": enhanced, "basic": basic,
             "ablation_imv": imv_from_likelihoods(variants[basic], variants[enhanced])}
            for seed, variants in likelihoods.items()
            for enhanced in variants for basic in variants
        ]).to_csv(self.pair_path, index=False)

    def load(self):
        return load_ablation_results(self.example, self.root, seeds=(42, 43))

    def test_transform_before_averaging_and_sample_sd(self):
        result = self.load()
        scores = [imv_from_likelihoods(0.5, a) for a in (0.6, 0.95)]
        self.assertAlmostEqual(result.null_summary.loc["Full", "mean"], np.mean(scores))
        self.assertAlmostEqual(result.null_summary.loc["Full", "std"], np.std(scores, ddof=1))
        self.assertNotAlmostEqual(np.mean(scores), imv_from_likelihoods(0.5, 0.775), places=4)
        self.assertEqual(tuple(result.pairwise_mean.index), self.example.variants)
        self.assertEqual(result.null_summary.loc["Full", "count"], 2)

    def test_direction_is_preserved(self):
        matrix = self.load().pairwise_mean
        expected = np.mean([imv_from_likelihoods(0.7, 0.6), imv_from_likelihoods(0.55, 0.95)])
        self.assertAlmostEqual(matrix.loc["Full", "Ablated"], expected)
        self.assertNotAlmostEqual(matrix.loc["Full", "Ablated"], matrix.loc["Ablated", "Full"])
        np.testing.assert_array_equal(np.diag(matrix), [0, 0])

    def test_row_order_does_not_affect_pairing(self):
        expected = self.load()
        for path in (self.diag_path, self.pair_path):
            pd.read_csv(path).sample(frac=1, random_state=7).to_csv(path, index=False)
        actual = self.load()
        pd.testing.assert_frame_equal(expected.pairwise_mean, actual.pairwise_mean)
        pd.testing.assert_frame_equal(expected.null_summary, actual.null_summary)

    def test_incomplete_or_duplicate_results_are_rejected(self):
        original = pd.read_csv(self.pair_path)
        for broken in (original.iloc[:-1], pd.concat([original, original.iloc[:1]])):
            with self.subTest(rows=len(broken)):
                broken.to_csv(self.pair_path, index=False)
                with self.assertRaises(ValueError):
                    self.load()

    def test_mismatched_diagnostics_are_rejected(self):
        diagnostics = pd.read_csv(self.diag_path)
        diagnostics.loc[0, "geometric_mean_likelihood"] = 0.8
        diagnostics.to_csv(self.diag_path, index=False)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.load()

    def test_missing_files_have_upstream_notebook_hint(self):
        with self.assertRaisesRegex(FileNotFoundError, "ablation_imv_fixture.ipynb"):
            load_ablation_results(self.example, self.root / "missing")

    def test_six_panels_share_scales_and_bars_use_saved_scores(self):
        result = self.load()
        fig, axes = plot_ablation_overview([result, result, result])
        self.addCleanup(plt.close, fig)
        self.assertEqual(axes.shape, (2, 3))
        fig.canvas.draw()
        for column in range(3):
            heatmap = axes[0, column]
            mesh = heatmap.collections[0]
            self.assertEqual(mesh.cmap.name, "Spectral_r")
            self.assertEqual(heatmap.title.get_fontfamily()[0], "Helvetica")
            self.assertIs(axes[0, 0].collections[0].norm, mesh.norm)
            np.testing.assert_allclose(
                mesh.get_array().reshape(result.pairwise_mean.shape), result.pairwise_mean
            )
            np.testing.assert_allclose(
                mesh.get_facecolors(), mesh.cmap(mesh.norm(result.pairwise_mean)).reshape(-1, 4)
            )
            self.assertEqual(heatmap.get_ylim(), (len(result.example.variants) - 0.5, -0.5))
            self.assertEqual(axes[1, 0].get_ylim(), axes[1, column].get_ylim())
            np.testing.assert_allclose(
                [bar.get_height() for bar in axes[1, column].patches], result.null_summary["mean"]
            )
            np.testing.assert_allclose(
                [bar.get_facecolor() for bar in axes[1, column].patches],
                spectral_colors(len(result.example.variants)),
            )

    def test_exports_only_pdf_without_raster_heatmaps(self):
        result = self.load()
        fig, _ = plot_ablation_overview([result, result, result])
        self.addCleanup(plt.close, fig)
        for suffix in ("", ".pdf", ".png", ".svg"):
            with self.subTest(suffix=suffix):
                paths = save_publication_figure(fig, self.root / f"overview{suffix}", dpi=80)
                self.assertEqual(paths, {"pdf": self.root / "overview.pdf"})
                self.assertNotIn(b"/Subtype /Image", paths["pdf"].read_bytes())
        self.assertEqual(list(self.root.glob("overview.*")), [self.root / "overview.pdf"])

    def test_configure_plotting_sets_shared_defaults(self):
        with matplotlib.rc_context():
            configure_plotting()
            self.assertEqual(matplotlib.rcParams["font.family"][0], "Helvetica")
            self.assertEqual(matplotlib.rcParams["image.cmap"], COLORMAP)
            np.testing.assert_allclose(
                matplotlib.rcParams["axes.prop_cycle"].by_key()["color"], spectral_colors(10)
            )

    def test_imvpy_heatmap_wrapper_uses_project_palette(self):
        matrix = self.load().pairwise_mean
        fig, ax = plot_ablation_matrix(matrix)
        self.addCleanup(plt.close, fig)
        mesh = ax.collections[0]
        self.assertEqual(mesh.cmap.name, "Spectral_r")
        self.assertEqual(ax.title.get_fontfamily()[0], "Helvetica")
        np.testing.assert_allclose(mesh.get_array().reshape(matrix.shape), matrix)
        self.assertFalse(mesh.colorbar.solids.get_rasterized())
        paths = save_publication_figure(fig, self.root / "single_matrix")
        self.assertNotIn(b"/Subtype /Image", paths["pdf"].read_bytes())

    @unittest.skipUnless(shutil.which("pdftoppm"), "PDF renderer check requires pdftoppm")
    def test_pdf_renders_each_cell_with_the_correct_color(self):
        result = self.load()
        fig, axes = plot_ablation_overview([result, result, result])
        self.addCleanup(plt.close, fig)
        paths = save_publication_figure(fig, self.root / "render", dpi=80)
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(
            matplotlib.rcParams["savefig.pad_inches"]
        )
        image_path = self.root / "pdf_render.png"
        command = ["pdftoppm", "-scale-to", "1600", "-singlefile", "-png",
                   str(paths["pdf"]), str(image_path.with_suffix(""))]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
        height, width = pixels.shape[:2]
        for ax in axes[0]:
            mesh = ax.collections[0]
            for (row, column), value in np.ndenumerate(result.pairwise_mean.to_numpy()):
                # Sample inside each cell, away from its text and boundaries.
                point = ax.transData.transform((column - 0.25, row - 0.25))
                x, y = fig.dpi_scale_trans.inverted().transform(point)
                px = round((x - bbox.x0) / bbox.width * width)
                py = round((bbox.y1 - y) / bbox.height * height)
                expected = np.array(mesh.cmap(mesh.norm(value))[:3]) * 255
                np.testing.assert_allclose(pixels[py, px], expected, atol=3)


if __name__ == "__main__":
    unittest.main()
