"""Shared bar annotations and colourbars, using small figures only."""

import unittest
import warnings
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.empirical.plotter import figure_utils as figures


class PlottingPrimitiveTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_both_style_profiles_use_dashed_gridlines(self):
        for configure in (figures.configure_plotting, figures.apply):
            with self.subTest(profile=configure.__name__), mpl.rc_context({"grid.linestyle": ":"}):
                configure()
                self.assertEqual(mpl.rcParams["grid.linestyle"], "--")
                _, axis = plt.subplots()
                axis.grid(True, axis="both")
                for line in axis.get_xgridlines() + axis.get_ygridlines():
                    self.assertTrue(line.get_visible())
                    self.assertEqual(line.get_linestyle(), "--")

    def test_bar_edges_and_labels_include_asymmetric_errors_and_negative_values(self):
        _, axis = plt.subplots()
        values = np.array([.25, -.125, 0, -.0001])
        errors = np.array([[.02, .03, 0, 0], [.04, .06, 0, 0]])
        colors = figures.spectral_colors(len(values))
        bars = figures.plot_bars(axis, range(4), values, yerr=errors, colors=colors)
        np.testing.assert_allclose([bar.get_height() for bar in bars], values)
        np.testing.assert_allclose([bar.get_facecolor() for bar in bars], colors)
        for bar in bars:
            np.testing.assert_array_equal(bar.get_edgecolor(), [0, 0, 0, 1])
            self.assertEqual(bar.get_linewidth(), figures.EDGE_WIDTH)
        labels = [text for text in axis.texts if text.get_gid() == "bar-value"]
        self.assertEqual([text.get_text() for text in labels], ["0.250", "-0.125", "0.000", "0.000"])
        np.testing.assert_allclose([text.xy[1] for text in labels], [.29, -.155, 0, -.0001])
        self.assertEqual([text.get_va() for text in labels], ["bottom", "top", "bottom", "top"])
        for label, direction in zip(labels, [1, -1, 1, -1]):
            self.assertEqual(label.get_position(), (0, direction * figures.BAR_LABEL_PADDING))
            self.assertEqual(label.get_fontsize(), figures.ANNOTATION_FONT_SIZE)
        np.testing.assert_array_equal(bars.errorbar.lines[2][0].get_colors(), [[0, 0, 0, 1]])
        figures.set_bar_limits(axis)
        self.assertLess(axis.get_ylim()[0], -.155)
        self.assertGreater(axis.get_ylim()[1], .29)

    def test_missing_values_are_not_labelled_and_zero_bars_get_valid_limits(self):
        _, axis = plt.subplots()
        bars = figures.plot_bars(axis, ["A", "B", "C"], [0, np.nan, 0])
        self.assertTrue(np.isnan(bars[1].get_height()))
        self.assertEqual([text.get_text() for text in axis.texts], ["0.000", "0.000"])
        figures.set_bar_limits(axis)
        low, high = axis.get_ylim()
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        self.assertTrue(np.isfinite(high))

    def test_bar_headroom_is_shared_without_resizing_panels(self):
        figure, axes = plt.subplots(1, 2, sharey=True)
        figures.plot_bars(axes[0], ["A"], [1], yerr=[.1])
        figures.plot_bars(axes[1], ["B"], [3], yerr=[.3])
        before = [axis.get_position().bounds for axis in axes]
        figures.set_bar_limits(axes)
        figure.canvas.draw()
        self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertAlmostEqual(axes[0].get_ylim()[1], 3.3 * (1 + figures.BAR_LIMIT_PADDING))
        np.testing.assert_allclose([axis.get_position().bounds for axis in axes], before)
        for value in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                figures.set_bar_limits(axes, padding=value)

    def test_wider_colourbars_keep_every_panel_and_vertical_alignment(self):
        figure, axes = plt.subplots(2, 3)
        mesh = axes[0, 0].pcolormesh([[0, .2], [.8, 1]], cmap=figures.COLORMAP)
        figure.canvas.draw()
        before = np.array([axis.get_position().bounds for axis in axes.flat])
        colorbar = figures.add_heatmap_colorbar(mesh, axes[0, -1], "IMV")
        figure.canvas.draw()
        np.testing.assert_allclose([axis.get_position().bounds for axis in axes.flat], before)
        parent_box, box = axes[0, -1].get_position(), colorbar.ax.get_position()
        self.assertAlmostEqual(box.y0, parent_box.y0)
        self.assertAlmostEqual(box.height, parent_box.height)
        self.assertAlmostEqual(box.width / parent_box.width, figures.COLORBAR_WIDTH)
        self.assertGreater(box.width / parent_box.width, .045)
        self.assertAlmostEqual((box.x0 - parent_box.x1) / parent_box.width, figures.COLORBAR_PADDING)
        self.assertIs(colorbar.norm, mesh.norm)
        self.assertEqual(colorbar.cmap.name, figures.COLORMAP)
        self.assertEqual(colorbar.ax.get_ylabel(), "IMV")
        self.assertFalse(colorbar.solids.get_rasterized())
        self.assertTrue(colorbar.outline.get_visible())
        np.testing.assert_array_equal(colorbar.outline.get_edgecolor(), [0, 0, 0, 1])
        self.assertEqual(colorbar.outline.get_linewidth(), figures.EDGE_WIDTH)
        self.assertEqual(colorbar.ax.yaxis.label.get_fontsize(), figures.LABEL_FONT_SIZE)

    def test_bar_limits_ignore_only_machine_roundoff_below_zero(self):
        _, axis = plt.subplots()
        figures.plot_bars(axis, ["A"], [.5])
        axis.dataLim.y0 = -np.finfo(float).eps
        figures.set_bar_limits(axis)
        self.assertEqual(axis.get_ylim()[0], 0)
        axis.dataLim.y0 = -1e-8
        figures.set_bar_limits(axis)
        self.assertLess(axis.get_ylim()[0], -1e-8)

    def test_colourbar_validation(self):
        _, axis = plt.subplots()
        mesh = axis.pcolormesh([[0, 1]])
        for kwargs in ({"width": 0}, {"width": np.nan}, {"pad": -1}):
            with self.assertRaises(ValueError):
                figures.add_heatmap_colorbar(mesh, axis, "IMV", **kwargs)

    def test_mirrored_heatmap_labels_clear_colourbar_after_layout_and_resize(self):
        with plt.rc_context(figures.rc_params()):
            figure, axis = plt.subplots(figsize=(6, 5))
            mesh = axis.pcolormesh([[0, .5], [.5, 0]], cmap=figures.COLORMAP)
            axis.set_xticks([.5, 1.5], ["Barbunya", "Dermason"])
            axis.set_yticks([.5, 1.5], ["Barbunya", "Dermason"])
            figures.style_heatmap_axes(axis)
            figures.label_panels(axis)
            colorbar = figures.add_heatmap_colorbar(mesh, axis, "Pairwise IMV")
            for size in ((6, 5), (8, 4)):
                figure.set_size_inches(*size)
                figures.apply_tight_layout(figure)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                self.assertAlmostEqual(colorbar.ax.bbox.height, axis.bbox.height)
                self.assertAlmostEqual(colorbar.ax.bbox.width / axis.bbox.width,
                                       figures.COLORBAR_WIDTH)
                for tick in axis.yaxis.get_major_ticks():
                    self.assertLess(tick.label2.get_window_extent(renderer).x1, colorbar.ax.bbox.x0)
                title = axis._left_title.get_window_extent(renderer)
                for tick in axis.xaxis.get_major_ticks():
                    self.assertGreaterEqual(title.y0, tick.label2.get_window_extent(renderer).y1)
                box = axis.get_tightbbox(renderer)
                self.assertGreaterEqual(box.x0, 0)
                self.assertGreaterEqual(box.y0, 0)
                self.assertLessEqual(box.x1, figure.bbox.width)
                self.assertLessEqual(box.y1, figure.bbox.height)

    def test_tight_layout_fits_colourbar_and_keeps_columns_aligned(self):
        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis in axes.flat:
            axis.set_xlabel("Long category label")
            axis.set_ylabel("Metric label")
        figures.label_panels(axes)
        mesh = axes[0, -1].pcolormesh([[0, 1], [1, 0]], cmap=figures.COLORMAP)
        colorbar = figures.add_heatmap_colorbar(mesh, axes[0, -1], "Directional IMV")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with patch.object(figure, "tight_layout", wraps=figure.tight_layout) as layout:
                self.assertIs(figures.apply_tight_layout(figure), figure)
                layout.assert_called_once()
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for upper, lower in zip(axes[0], axes[1]):
            self.assertAlmostEqual(upper.get_position().x0, lower.get_position().x0)
            self.assertAlmostEqual(upper.get_position().width, lower.get_position().width)
            self.assertGreater(upper.get_tightbbox(renderer).y0, lower.get_tightbbox(renderer).y1)
        for axis in [*axes.flat, colorbar.ax]:
            box = axis.get_tightbbox(renderer)
            self.assertGreaterEqual(box.x0, 0)
            self.assertGreaterEqual(box.y0, 0)
            self.assertLessEqual(box.x1, figure.bbox.width)
            self.assertLessEqual(box.y1, figure.bbox.height)
        self.assertAlmostEqual(colorbar.ax.get_position().height, axes[0, -1].get_position().height)
        self.assertAlmostEqual(colorbar.ax.get_position().width / axes[0, -1].get_position().width,
                               figures.COLORBAR_WIDTH)

    def test_tight_layout_reserves_space_for_bottom_figure_legend(self):
        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis in axes.flat:
            axis.plot([0, 1], [0, 1], label="Mean IMV")
            axis.set_xlabel("Epoch")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend = figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015))
        figures.apply_tight_layout(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in axes[1]:
            self.assertGreater(axis.get_tightbbox(renderer).y0, legend.get_window_extent(renderer).y1)


if __name__ == "__main__":
    unittest.main()
