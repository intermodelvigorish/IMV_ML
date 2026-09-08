"""The incoming panel helpers must not override the publication font or palette."""

import unittest

import matplotlib as mpl

from src.figure_utils import COLORMAP, PAPER_STYLE
from src import shared_style


class SharedStyleTests(unittest.TestCase):
    def test_panel_style_uses_canonical_font_and_palette(self):
        with mpl.rc_context():
            shared_style.apply()
            self.assertEqual(mpl.rcParams["font.family"], PAPER_STYLE["font.family"])
            self.assertEqual(mpl.rcParams["font.sans-serif"], PAPER_STYLE["font.sans-serif"])
            self.assertEqual(mpl.rcParams["image.cmap"], COLORMAP)
            self.assertEqual(mpl.rcParams["pdf.fonttype"], PAPER_STYLE["pdf.fonttype"])
            self.assertEqual(shared_style.PALETTE, COLORMAP)
            self.assertEqual(shared_style.heatmap_style(3)["cmap"].name, COLORMAP)
            self.assertEqual(len(shared_style.bar_style(3)["color"]), 3)

    def test_panel_geometry_is_preserved(self):
        self.assertEqual(shared_style.figure_size(2, 3), (3 * 5.4, 2 * 4.6))
        with self.assertRaises(ValueError):
            shared_style.figure_size(0, 3)


if __name__ == "__main__":
    unittest.main()
