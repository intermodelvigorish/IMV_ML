"""Check that the combined notebook keeps its sections independent."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from IPython.core.inputtransformer2 import TransformerManager
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.empirical.plotter import shared_style


NOTEBOOK = Path(__file__).resolve().parents[1] / "src/empirical/plotter/plotter.ipynb"


class CombinedPlotterTests(unittest.TestCase):
    def setUp(self):
        self.cells = json.loads(NOTEBOOK.read_text())["cells"]
        transformer = TransformerManager()
        self.trees = [ast.parse(transformer.transform_cell("".join(cell["source"])))
                      for cell in self.cells if cell["cell_type"] == "code"]

    def test_combined_notebook_retains_sections_and_one_shared_bootstrap(self):
        ids = [cell["id"] for cell in self.cells]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({"plot-export", "rerun-caption", "gnn-export", "gnn-caption",
                         "gnn-seed-detail-export", "821b4ba7", "8817b8f5", "3960f2bf"}
                        <= set(ids))
        roots = [node for tree in self.trees for node in ast.walk(tree)
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                 and node.id == "PROJECT_ROOT"]
        self.assertEqual(len(roots), 1)
        names = {node.id for tree in self.trees for node in ast.walk(tree)
                 if isinstance(node, ast.Name)}
        self.assertTrue({"ablation_results", "multiclass_results", "gnn"} <= names)
        self.assertNotIn("results", names)

    def test_multiclass_export_restores_style_for_other_sections(self):
        definition = next(node for tree in self.trees for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "show_example")
        figure_styles = []

        def example_figure(result):
            figure_styles.append(mpl.rcParams["axes.grid"])
            self.assertEqual(mpl.rcParams["font.family"], shared_style.FONT_FAMILY)
            return plt.figure()

        def save_figure(figure, path):
            return {"pdf": path.with_suffix(".pdf")}

        namespace = {
            "plt": plt, "shared_style": shared_style,
            "multiclass_results": {"fixture": {"title": "Fixture", "n_seeds": 10,
                                               "table": None, "global_summary": None}},
            "display": lambda table: None, "format_global_metrics": lambda table: "",
            "example_figure": example_figure, "save_publication_figure": save_figure,
            "relative_path": str,
        }
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(NOTEBOOK), "exec"),
             namespace)
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({"axes.grid": False}):
            namespace["FIGURES"] = Path(directory)
            original_figures = plt.get_fignums()
            with patch.object(plt, "show"), patch("builtins.print"):
                paths = namespace["show_example"]("fixture")
            self.assertEqual(set(paths), {"pdf"})
            self.assertEqual(Path(paths["pdf"]).name, "multi_imv_results__fixture.pdf")
            self.assertFalse(mpl.rcParams["axes.grid"])
            self.assertEqual(plt.get_fignums(), original_figures)
        self.assertEqual(figure_styles, [True])


if __name__ == "__main__":
    unittest.main()
