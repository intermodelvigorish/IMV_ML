"""Path and discovery checks that do not execute the empirical experiments."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from IPython.core.inputtransformer2 import TransformerManager

from src.figure_utils import find_project_root


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def notebooks():
    return sorted(path for path in SOURCE.rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in path.parts)


class SourceLayoutTests(unittest.TestCase):
    def test_examples_live_in_one_of_the_two_groups(self):
        for group in ("ablation_imv", "gnn", "multi_imv", "shap_imv"):
            self.assertTrue((SOURCE / "empirical" / group).is_dir())
            self.assertFalse((SOURCE / group).exists())
        self.assertFalse((SOURCE / "simulated").exists())
        for filename in ("ablate_imv.ipynb", "multi_imv.ipynb", "shap_imv.ipynb",
                         "complexity.ipynb", "vanilla_imv.py"):
            self.assertTrue((SOURCE / "simulations" / filename).is_file())
        for notebook in notebooks():
            self.assertIn(notebook.relative_to(SOURCE).parts[0],
                          ("empirical", "simulations", "plotter"))

    def test_root_finder_works_from_all_nested_notebooks(self):
        for notebook in notebooks():
            with self.subTest(notebook=notebook.relative_to(ROOT)):
                self.assertEqual(find_project_root(notebook.parent), ROOT)
                self.assertEqual(find_project_root(notebook), ROOT)
        with tempfile.TemporaryDirectory() as unrelated:
            with self.assertRaises(FileNotFoundError):
                find_project_root(unrelated)

    def test_notebook_bootstraps_resolve_root_from_root_and_notebook_directory(self):
        transformer = TransformerManager()
        checked = set()
        for notebook in notebooks():
            data = json.loads(notebook.read_text())
            for cell in data["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse(transformer.transform_cell("".join(cell["source"])))
                for statement in tree.body:
                    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                        continue
                    target = statement.targets[0]
                    if not isinstance(target, ast.Name) or target.id not in ("PROJECT_ROOT", "REPO_ROOT"):
                        continue
                    # Evaluate only the root expression, never data loading or training.
                    code = compile(ast.Expression(statement.value), str(notebook), "eval")
                    for cwd in (ROOT, notebook.parent):
                        with self.subTest(notebook=notebook.relative_to(ROOT), cwd=cwd):
                            with patch("os.getcwd", return_value=str(cwd)):
                                self.assertEqual(eval(code, {"Path": Path}), ROOT)
                    checked.add(notebook)
        expected = {path for path in notebooks() if path.relative_to(SOURCE).parts[0]
                    in ("empirical", "plotter")}
        expected.add(SOURCE / "simulations" / "complexity.ipynb")
        self.assertEqual(checked, expected)

    def test_notebook_sources_use_current_paths_and_imports(self):
        legacy_path = re.compile(r"src/(?:ablation_imv|multi_imv|shap_imv|gnn|simulated)/")
        legacy_import = re.compile(r"\b(?:from|import)\s+gnn(?:\.|\s)")
        for notebook in notebooks():
            data = json.loads(notebook.read_text())
            source = "\n".join("".join(cell["source"]) for cell in data["cells"])
            with self.subTest(notebook=notebook.relative_to(ROOT)):
                self.assertIsNone(legacy_path.search(source))
                self.assertIsNone(legacy_import.search(source))
                for reference in re.findall(r"src/[A-Za-z0-9_./-]+\.ipynb", source):
                    self.assertTrue((ROOT / reference).is_file(), reference)

    def test_runner_discovers_every_notebook_once_with_plotters_last(self):
        env = {**os.environ, "IMV_N_JOBS": "2", "IMV_TORCH_JOBS": "2"}
        command = ["bash", str(ROOT / "run_all.sh"), "--resume", "--dry-run"]
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(command, cwd=cwd, env=env, check=True,
                                    capture_output=True, text=True, timeout=30)
        discovered = re.findall(r"^\[\d+/\d+\] (src/.*\.ipynb)$", result.stdout, re.MULTILINE)
        expected = [str(path.relative_to(ROOT)) for path in notebooks()]
        producers = sorted(path for path in expected if not path.startswith("src/plotter/"))
        plotters = sorted(path for path in expected if path.startswith("src/plotter/"))
        self.assertEqual(discovered, producers + plotters)
        self.assertEqual(len(discovered), len(set(discovered)))

    def test_empirical_notebooks_include_metric_and_feature_comparisons(self):
        expected = {
            "multi_imv": ("car_evaluation", "dry_bean", "nursery"),
            "shap_imv": ("adult_income", "breast_cancer", "titanic"),
        }
        transformer = TransformerManager()
        for group, datasets in expected.items():
            for dataset in datasets:
                notebook = SOURCE / "empirical" / group / f"{group}_{dataset}.ipynb"
                data = json.loads(notebook.read_text())
                definitions = set()
                for cell in data["cells"]:
                    if cell["cell_type"] == "code":
                        tree = ast.parse(transformer.transform_cell("".join(cell["source"])))
                        definitions.update(node.name for node in tree.body
                                           if isinstance(node, ast.FunctionDef))
                with self.subTest(notebook=notebook.name):
                    required = ({"classification_metrics", "format_cell"}
                                if group == "multi_imv"
                                else {"shap_importance", "lime_importance",
                                      "anchor_importance", "evaluate"})
                    self.assertTrue(required <= definitions, required - definitions)
                    ids = [cell["id"] for cell in data["cells"]]
                    self.assertEqual(len(ids), len(set(ids)))

    def test_feature_helper_finds_project_when_invoked_outside_repository(self):
        script = SOURCE / "empirical" / "shap_imv" / "top_k_features.py"
        tree = ast.parse(script.read_text())
        root_assignment = next(node for node in tree.body
                               if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name)
                                       and target.id == "PROJECT_ROOT"
                                       for target in node.targets))
        code = compile(ast.Expression(root_assignment.value), str(script), "eval")
        with tempfile.TemporaryDirectory() as cwd:
            with patch("os.getcwd", return_value=cwd):
                root = eval(code, {"Path": Path, "__file__": str(script)})
        self.assertEqual(root, ROOT)

    def test_multiclass_plotter_uses_pdf_export_and_root_anchored_directory(self):
        notebook = SOURCE / "plotter" / "multi_imv_results.ipynb"
        data = json.loads(notebook.read_text())
        source = "\n".join("".join(cell["source"]) for cell in data["cells"])
        self.assertIn("save_publication_figure as save_figure", source)
        self.assertIn("FIGURES = figure_directory(PROJECT_ROOT)", source)
        self.assertNotIn("from imvpy.utils import save_figure", source)


if __name__ == "__main__":
    unittest.main()
