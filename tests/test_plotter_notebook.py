"""Check that the combined notebook keeps its sections independent."""

import ast
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from IPython.core.inputtransformer2 import TransformerManager
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.empirical.plotter import figure_utils


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
                         "gnn-seed-detail-export", "821b4ba7", "8817b8f5", "3960f2bf",
                         "bd49eaec", "619d9a88", "1a56605e", "a70321b1"}
                        <= set(ids))
        roots = [node for tree in self.trees for node in ast.walk(tree)
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                 and node.id == "PROJECT_ROOT"]
        self.assertEqual(len(roots), 1)
        names = {node.id for tree in self.trees for node in ast.walk(tree)
                 if isinstance(node, ast.Name)}
        self.assertTrue({"ablation_results", "multiclass_results", "shap_results", "gnn"} <= names)
        self.assertNotIn("results", names)
        definitions = [node.name for tree in self.trees for node in tree.body
                       if isinstance(node, ast.FunctionDef)]
        self.assertEqual(len(definitions), len(set(definitions)))

    def declarations(self):
        """Load pure helpers and literal settings, not result loading or exports."""
        namespace = {"np": np, "pd": pd, "plt": plt, "sns": sns, "figure_utils": figure_utils}
        for tree in self.trees:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"),
                         namespace)
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            namespace[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        return namespace

    def test_figure_two_caption_matches_panels_and_attribution_protocol(self):
        caption = "".join(next(cell for cell in self.cells if cell["id"] == "5854528d")["source"])
        for label in ("Figure 2", "Titanic", "Wisconsin Diagnostic", "Adult Income",
                      "logistic regression", "XGBoost", "LightGBM", "eight predictors",
                      "five-fold stratified cross-validation", "ten random seeds",
                      "one sample standard deviation", "not confidence intervals"):
            self.assertIn(label, caption)
        for panel in ("a.", "c.", "d.", "f.", "g.", "i."):
            self.assertIn(f"**{panel}**", caption)
        self.assertIn("N_SPLITS = 5", caption)
        self.assertIn("EVAL_SPLITS = 10", caption)

    def test_section_order_and_numbering_follow_shap_multi_ablate_gnn(self):
        expected = [
            ("92817dba", 1, "SHAP-IMV"),
            ("e6d180f0", 2, "Multi-IMV"),
            ("interpretation", 3, "Ablate-IMV"),
            ("gnn-curves-protocol", 4, "GNN training"),
        ]
        self.assertEqual(self.cells[1]["id"], "imports-paths")
        sections = []
        current_number = None
        for cell in self.cells[2:]:
            if cell["cell_type"] != "markdown":
                continue
            for line in "".join(cell["source"]).splitlines():
                section = re.match(r"^## (\d+)\. ([^:]+?)(?::| \u2014|$)", line)
                if section:
                    current_number = int(section[1])
                    sections.append((cell["id"], current_number, section[2]))
                subsection = re.match(r"^### (\d+)\.(\d+)\.", line)
                if subsection:
                    self.assertEqual(int(subsection[1]), current_number)
        self.assertEqual(sections, expected)
        intro = "".join(self.cells[0]["source"])
        for _, number, label in expected:
            self.assertIn(f"{number}. **{label}:**", intro)

    def test_shap_table_captions_distinguish_attribution_from_refit_folds(self):
        namespace = self.declarations()
        transformer = TransformerManager()
        for example in namespace["SHAP_EXAMPLES"]:
            path = NOTEBOOK.parents[1] / "shap_imv" / f"{example}.ipynb"
            settings = {}
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse(transformer.transform_cell("".join(cell["source"])))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        if name in ("N_SPLITS", "EVAL_SPLITS", "SEEDS"):
                            settings[name] = ast.literal_eval(node.value)
            self.assertEqual(settings["N_SPLITS"], namespace["SHAP_ATTRIBUTION_SPLITS"])
            self.assertEqual(settings["EVAL_SPLITS"], 10)
            self.assertEqual(len(settings["SEEDS"]), 10)
        note = "".join(next(c for c in self.cells if c["id"] == "a8660909")["source"])
        self.assertIn("eight specified predictors", note)
        self.assertIn("not unbiased estimates", note)
        self.assertNotIn("Six features", note)

    def test_multiclass_manuscript_mapping_and_captions(self):
        namespace = self.declarations()
        self.assertEqual(namespace["MULTI_FIGURE_FILES"], {"dry_bean": "figure_3"})
        self.assertEqual(namespace["MULTI_SUPPLEMENTARY_EXAMPLES"], ("car_evaluation", "nursery"))
        self.assertEqual(namespace["MULTI_SUPPLEMENTARY_FIGURE"], "figure_A4")
        for cell_id, figure in (("003c702e", "Figure 3"), ("5929c66e", "Figure A4")):
            caption = "".join(next(cell for cell in self.cells if cell["id"] == cell_id)["source"])
            for phrase in (figure, "five-fold stratified cross-validation", "ten random seeds",
                           "one sample standard deviation", "not confidence intervals",
                           "constant input only", "**a.**", "**f.**"):
                self.assertIn(phrase, caption)
            self.assertNotIn("0.5", caption)
        self.assertIn("Car Evaluation", caption)
        self.assertIn("Nursery", caption)
        self.assertIn("**g.**", caption)
        self.assertIn("**l.**", caption)
        source = "".join(next(cell for cell in self.cells if cell["id"] == "3960f2bf")["source"])
        self.assertIn("show_multi_supplement()", source)

    def test_table_three_mapping_and_caption_match_the_car_evaluation_protocol(self):
        namespace = self.declarations()
        text = "".join(next(c for c in self.cells if c["id"] == "c6b61bdf")["source"])
        self.assertTrue(text.startswith("### 2.3. Table 3: Car Evaluation"))
        for phrase in ("multi_imv_car_evaluation.tex", "tab:MultiIMVCarEvaluation",
                       "ten seeds", "five-fold stratified cross-validation",
                       "not confidence intervals", "pooled out-of-fold predictions"):
            self.assertIn(phrase, text)
        caption = namespace["multi_table_caption"]({
            "dataset": "car_evaluation", "title": "Car Evaluation", "n_seeds": 10,
        })
        for phrase in ("across 10 seeds", "5-fold stratified cross-validation",
                       "one sample standard deviation", "not confidence intervals",
                       "constant input only", "averaged over folds within each seed",
                       "pooled out-of-fold predictions", "multiclass argmax",
                       "percentages", "Unacceptable and Acceptable"):
            self.assertIn(phrase, caption)
        self.assertNotIn("Table 3:", caption)  # LaTeX owns the displayed number.
        notebook = NOTEBOOK.parents[1] / "multi_imv/multi_imv_car_evaluation.ipynb"
        transformer = TransformerManager()
        settings = {}
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(transformer.transform_cell("".join(cell["source"])))
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ("SEEDS", "N_SPLITS"):
                        settings[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(settings["N_SPLITS"], namespace["MULTI_N_SPLITS"])
        self.assertEqual(settings["SEEDS"], list(range(42, 52)))

    def test_ablation_figure_four_mapping_and_caption(self):
        self.assertEqual(self.declarations()["FIGURE_NAME"], "figure_4")
        caption = "".join(next(cell for cell in self.cells if cell["id"] == "rerun-caption")["source"])
        for phrase in ("Figure 4", "figure_4.pdf", "MNIST", "UCI HAR", "MAGIC Gamma",
                       "directional pairwise", "rows are enhanced", "columns are basic",
                       "constant 0.5-probability", "ten seed-level scores",
                       "one sample standard deviation", "not confidence intervals"):
            self.assertIn(phrase, caption)
        for label in "abcdef":
            self.assertIn(f"**{label}.**", caption)
        self.assertNotIn("IMDB", caption)
        self.assertNotIn("CIFAR", caption)

    def test_gnn_figure_five_mapping_and_caption(self):
        self.assertEqual(self.declarations()["GNN_FIGURE_NAME"], "figure_5")
        caption = "".join(next(cell for cell in self.cells if cell["id"] == "gnn-caption")["source"])
        for phrase in ("Figure 5", "figure_5.pdf", "PROTEINS", "NCI1", "NCI109",
                       "Mutagenicity", "AIDS", "DD", "ten independent runs",
                       "minimum and maximum seed values at each epoch",
                       "rather than confidence intervals", "70/15/15",
                       "Epoch 0", "training-set prevalence", "without temporal smoothing"):
            self.assertIn(phrase, caption)
        for label in "abcdef":
            self.assertIn(f"**{label}.**", caption)

    def test_gnn_export_preserves_old_pdf_on_failure_and_closes_figures(self):
        namespace = self.declarations()
        namespace.update(gnn=None, relative_path=str,
                         plot_gnn_learning_curves=lambda results: (plt.figure(), []))
        with tempfile.TemporaryDirectory() as directory:
            namespace["FIGURES"] = Path(directory)
            obsolete = Path(directory) / "gnn_training__six_datasets__learning_curves.pdf"
            obsolete.touch()
            unrelated = Path(directory) / "gnn_training__six_datasets__individual_seed_imv.pdf"
            unrelated.touch()
            before = plt.get_fignums()
            with patch.object(plt, "show"):
                def failed_save(*args):
                    raise OSError("export failed")
                namespace["save_publication_figure"] = failed_save
                with self.assertRaisesRegex(OSError, "export failed"):
                    namespace["show_gnn_figure"]()
                self.assertTrue(obsolete.exists())
                self.assertEqual(plt.get_fignums(), before)
                namespace["save_publication_figure"] = lambda figure, path: {"pdf": path.with_suffix(".pdf")}
                paths = namespace["show_gnn_figure"]()
            self.assertEqual(Path(paths["pdf"]).name, "figure_5.pdf")
            self.assertFalse(obsolete.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(plt.get_fignums(), before)

    def test_ablation_export_preserves_old_pdf_on_failure_and_closes_figures(self):
        namespace = self.declarations()
        namespace.update(ablation_results=[], relative_path=str,
                         plot_ablation_overview=lambda results: (plt.figure(), []))
        with tempfile.TemporaryDirectory() as directory:
            namespace["FIGURES"] = Path(directory)
            obsolete = Path(directory) / "ablation_imv__mnist_har_magic_gamma__overview.pdf"
            obsolete.touch()
            unrelated = Path(directory) / "figure_3.pdf"
            unrelated.touch()
            before = plt.get_fignums()
            with patch.object(plt, "show"):
                def failed_save(*args):
                    raise OSError("export failed")
                namespace["save_publication_figure"] = failed_save
                with self.assertRaisesRegex(OSError, "export failed"):
                    namespace["show_ablation_figure"]()
                self.assertTrue(obsolete.exists())
                self.assertEqual(plt.get_fignums(), before)
                namespace["save_publication_figure"] = lambda figure, path: {"pdf": path.with_suffix(".pdf")}
                paths = namespace["show_ablation_figure"]()
            self.assertEqual(Path(paths["pdf"]).name, "figure_4.pdf")
            self.assertFalse(obsolete.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(plt.get_fignums(), before)

    def test_combined_multiclass_supplement_preserves_both_datasets(self):
        namespace = self.declarations()
        models = namespace["MULTI_MODEL_NAMES"]
        results = []
        for count, offset in ((4, -.1), (3, .2)):
            results.append({
                "class_names": [f"Class {i}" for i in range(count)], "n_seeds": 10,
                "pairwise": {model: pd.DataFrame(np.full((count, count), offset + j * .1))
                             for j, model in enumerate(models)},
                "ova_summary": pd.DataFrame([
                    {"model": model, "class": i, "mean": .1 + i * .1, "std": .01}
                    for model in models for i in range(count)
                ]),
            })
        with plt.rc_context(figure_utils.rc_params()):
            figure = figure_utils.plot_multiclass_overview(results, models)
            figure_utils.apply_tight_layout(figure)
        self.addCleanup(plt.close, figure)
        axes = sorted(figure.axes, key=lambda axis: axis.get_subplotspec().num1)
        self.assertEqual(len(axes), 12)
        axes = np.asarray(axes).reshape(4, 3)
        self.assertFalse(figure.texts)
        for index, axis in enumerate(axes.flat):
            self.assertEqual(axis.get_title(loc="left"), f"{chr(97 + index)}.")
            self.assertEqual(axis._left_title.get_fontweight(), "bold")
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for block, result in enumerate(results):
            for column, model in enumerate(models):
                heat, bars = axes[2 * block, column], axes[2 * block + 1, column]
                np.testing.assert_allclose(np.asarray(heat.collections[0].get_array()).ravel(),
                                           result["pairwise"][model].to_numpy().ravel())
                part = result["ova_summary"].query("model == @model").sort_values("class")
                np.testing.assert_allclose([bar.get_height() for bar in bars.patches], part["mean"])
                segments = bars.containers[0].lines[2][0].get_segments()
                np.testing.assert_allclose([line[:, 1] for line in segments],
                                           np.column_stack((part["mean"] - part["std"],
                                                            part["mean"] + part["std"])))
                self.assertTrue(all(t.get_fontsize() == figure_utils.MULTICLASS_ANNOTATION_FONT_SIZE
                                    for t in heat.texts + bars.texts))
            norms = [axis.collections[0].norm for axis in axes[2 * block]]
            self.assertEqual(len({(norm.vmin, norm.vmax) for norm in norms}), 1)
            colorbar = axes[2 * block, 0].collections[0].colorbar
            self.assertAlmostEqual(colorbar.ax.bbox.height, axes[2 * block, -1].bbox.height)
        self.assertLess(axes[0, 0].collections[0].norm.vmin, 0)
        self.assertEqual(axes[2, 0].collections[0].norm.vmin, 0)
        for upper, lower in zip(axes[:-1], axes[1:]):
            for above, below in zip(upper, lower):
                self.assertGreater(above.get_tightbbox(renderer).y0, below.get_tightbbox(renderer).y1)

    def test_supplement_export_retires_separate_files_only_after_success(self):
        namespace = self.declarations()
        namespace.update(multiclass_results={name: {} for name in namespace["MULTI_SUPPLEMENTARY_EXAMPLES"]},
                         relative_path=str)
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({"axes.grid": False}):
            namespace["FIGURES"] = Path(directory)
            obsolete = [Path(directory) / name for name in
                        ("figure_A5.pdf", "multi_imv_results__car_evaluation.pdf",
                         "multi_imv_results__nursery.pdf")]
            for path in obsolete:
                path.touch()
            unrelated = Path(directory) / "figure_3.pdf"
            unrelated.touch()
            before = plt.get_fignums()
            with patch.object(figure_utils, "plot_multiclass_overview", side_effect=lambda *args: plt.figure()), \
                    patch.object(plt, "show"):
                namespace["save_publication_figure"] = lambda *args: (_ for _ in ()).throw(OSError("save failed"))
                with self.assertRaisesRegex(OSError, "save failed"):
                    namespace["show_multi_supplement"]()
                self.assertTrue(all(path.exists() for path in obsolete))
                self.assertEqual(plt.get_fignums(), before)
                namespace["save_publication_figure"] = lambda figure, path: {"pdf": path.with_suffix(".pdf")}
                paths = namespace["show_multi_supplement"]()
            self.assertEqual(Path(paths["pdf"]).name, "figure_A4.pdf")
            self.assertFalse(any(path.exists() for path in obsolete))
            self.assertTrue(unrelated.exists())
            self.assertEqual(plt.get_fignums(), before)
            self.assertFalse(mpl.rcParams["axes.grid"])

    def test_multiclass_and_shap_tables_keep_their_own_protocols_and_labels(self):
        namespace = self.declarations()
        classes = []
        globals_ = []
        for model in namespace["MULTI_MODEL_NAMES"]:
            for index in range(2):
                row = {"model": model, "class": index}
                for metric in namespace["MULTI_CLASS_METRIC_LABELS"]:
                    row.update({f"{metric}_mean": .5, f"{metric}_std": .01})
                classes.append(row)
            row = {"model": model}
            for metric in namespace["MULTI_GLOBAL_LABELS"]:
                row.update({f"{metric}_mean": .5, f"{metric}_std": .01})
            globals_.append(row)
        multi = {"dataset": "fixture", "title": "Fixture", "class_names": ["A", "B"],
                 "class_summary": pd.DataFrame(classes), "global_summary": pd.DataFrame(globals_),
                 "n_seeds": 10}
        before = namespace["multi_latex_table"](multi)
        self.assertIn("5-fold stratified cross-validation", before)
        self.assertEqual(before.count(r"\caption{"), 1)
        self.assertEqual(before.count(r"\caption*{"), 1)
        self.assertLess(before.index(r"\label{"), before.index(r"\begin{tabular}"))
        self.assertIn("support-weighted class averages", before)
        self.assertIn("Cohen's kappa are multiplied by 100", before)

        shap_results, top_k = {}, []
        for example, (dataset, title) in namespace["SHAP_EXAMPLES"].items():
            ranking = pd.DataFrame([
                {"model": model, "feature": f"feature_{rank}", "rank": rank,
                 "mean": .1 / rank, "std": .001}
                for model in namespace["SHAP_MODEL_NAMES"] for rank in range(1, 9)
            ])
            shap_results[example] = {"title": title, "ranking": ranking}
            for model in namespace["SHAP_MODEL_NAMES"]:
                for index, method in enumerate(namespace["SHAP_METHOD_NAMES"]):
                    top_k.append({"example": example, "model": model, "method": method,
                                  "accuracy_mean": .8 + index * .01,
                                  "precision_mean": .7 + index * .01})
        namespace.update(shap_results=shap_results, SHAP_N_SEEDS=10, SHAP_N_SPLITS=10,
                         shap_combined_top_k=pd.DataFrame(top_k),
                         shap_combined_ranking=pd.concat([r["ranking"] for r in shap_results.values()]))
        matrix = namespace["shap_metrics_matrix"]()
        comparison = namespace["shap_metrics_latex"](matrix)
        ranking_table = namespace["shap_ranking_latex"]()
        self.assertIn("10 folds for each of 10 seeds", comparison)
        self.assertIn("positive-class precision are percentages", comparison)
        self.assertIn("not nested within its folds", comparison)
        self.assertIn("weights the 3 datasets equally", comparison)
        self.assertIn("Attribution uses 5-fold stratified cross-validation", ranking_table)
        self.assertIn("distinct from the 10-fold refit", ranking_table)
        self.assertIn(r"\multicolumn{8}{c}{Rank by mean SHAP-IMV}", ranking_table)
        self.assertIn(r"\shortstack{feature\_8\\(0.013)}", ranking_table)
        self.assertIn(r"\label{SHAPIMVRankingtable}", ranking_table)
        self.assertIn("fixed 8-variable specification", ranking_table)
        for text, label in ((comparison, "SHAPIMVTabulartable"),
                            (ranking_table, "SHAPIMVRankingtable")):
            self.assertEqual(text.count(r"\caption{"), 1)
            self.assertEqual(text.count(r"\label{" + label + "}"), 1)
            self.assertLess(text.index(r"\label{"), text.index(r"\begin{tabular}"))
            self.assertNotIn(r"\vspace{-", text)
        self.assertEqual(namespace["multi_latex_table"](multi), before)
        examples = ("shap_imv_titanic", "shap_imv_breast_cancer", "shap_imv_adult_income")
        self.assertEqual(namespace["SHAP_FIGURE_GROUPS"],
                         (("figure_2", examples),))
        with plt.rc_context(figure_utils.rc_params()):
            figure = namespace["shap_ranking_figure"](examples)
            figure_utils.apply_tight_layout(figure)
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 9)
        np.testing.assert_allclose(figure.get_size_inches(),
                                   [3 * figure_utils.PANEL_WIDTH, 3 * figure_utils.COMPACT_PANEL_HEIGHT])
        self.assertLess(figure_utils.COMPACT_PANEL_HEIGHT, figure_utils.PANEL_HEIGHT)
        self.assertFalse(figure.texts)
        figure.canvas.draw()
        axes = np.asarray(figure.axes).reshape(3, 3)
        renderer = figure.canvas.get_renderer()
        for row, example in zip(axes, examples):
            for axis, model in zip(row, namespace["SHAP_MODEL_NAMES"]):
                self.assertEqual(set(axis.get_shared_y_axes().get_siblings(axis)), set(row))
                ranked = shap_results[example]["ranking"].query("model == @model").sort_values("rank")
                np.testing.assert_allclose([bar.get_height() for bar in axis.patches], ranked["mean"])
                self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                 ranked["feature"].tolist())
                for text in axis.texts:
                    box = text.get_window_extent(renderer)
                    self.assertGreaterEqual(box.y0, axis.bbox.y0)
                    self.assertLessEqual(box.y1, axis.bbox.y1)
        for upper_row, lower_row in zip(axes[:-1], axes[1:]):
            for upper, lower in zip(upper_row, lower_row):
                self.assertGreater(upper.get_tightbbox(renderer).y0, lower.get_tightbbox(renderer).y1)
        for index, axis in enumerate(figure.axes):
            self.assertEqual(len(axis.patches), 8)
            self.assertEqual(len([text for text in axis.texts if text.get_gid() == "bar-value"]), 8)
            for bar in axis.patches:
                np.testing.assert_array_equal(bar.get_edgecolor(), [0, 0, 0, 1])
            self.assertEqual(axis.get_title(loc="left"), f"{chr(97 + index)}.")
            self.assertEqual(axis._left_title.get_fontweight(), "bold")
            self.assertEqual(axis.get_title(loc="center"), "")
            self.assertEqual(axis.get_title(loc="right"), "")

    def test_multiclass_panels_have_only_row_major_letters(self):
        namespace = self.declarations()
        models = namespace["MULTI_MODEL_NAMES"]
        result = {
            "class_names": ["A", "B"], "n_seeds": 10,
            "pairwise": {model: pd.DataFrame([[0, .2], [.2, 0]]) for model in models},
            "ova_summary": pd.DataFrame([
                {"model": model, "class": index, "mean": .2 + index / 10, "std": .01}
                for model in models for index in range(2)
            ]),
        }
        with plt.rc_context(figure_utils.rc_params()):
            figure = namespace["example_figure"](result)
        self.addCleanup(plt.close, figure)
        self.assertFalse(figure.texts)
        axes = sorted(figure.axes, key=lambda axis: axis.get_subplotspec().num1)
        self.assertEqual(len(axes), 6)
        figure.canvas.draw()
        grid = axes[0].get_subplotspec().get_gridspec()
        self.assertFalse(grid.locally_modified_subplot_params())
        self.assertEqual(grid.get_height_ratios(), [1, 0.85])
        colorbar = axes[0].collections[0].colorbar
        self.assertAlmostEqual(colorbar.ax.get_position().height, axes[2].get_position().height)
        for axis in axes[3:]:
            self.assertEqual(len([text for text in axis.texts if text.get_gid() == "bar-value"]), 2)
            self.assertTrue(all(text.get_fontsize() == figure_utils.MULTICLASS_ANNOTATION_FONT_SIZE
                                for text in axis.texts))
        for axis in axes[:3]:
            for spine in axis.spines.values():
                self.assertTrue(spine.get_visible())
                np.testing.assert_array_equal(spine.get_edgecolor(), [0, 0, 0, 1])
            for tick in [*axis.xaxis.get_major_ticks(), *axis.yaxis.get_major_ticks()]:
                self.assertTrue(tick.label1.get_visible() and tick.label2.get_visible())
                self.assertEqual(tick.label1.get_text(), tick.label2.get_text())
            self.assertTrue(all(text.get_fontsize() == figure_utils.MULTICLASS_ANNOTATION_FONT_SIZE
                                for text in axis.texts))
        for index, axis in enumerate(axes):
            self.assertEqual(axis.get_title(loc="left"), f"{chr(97 + index)}.")
            self.assertEqual(axis._left_title.get_fontweight(), "bold")
            self.assertEqual(axis.get_title(loc="center"), "")
            self.assertEqual(axis.get_title(loc="right"), "")

    def test_shap_export_restores_style_and_closes_figures(self):
        namespace = self.declarations()
        observed = []

        def ranking_figure(examples):
            observed.append(mpl.rcParams["axes.grid"])
            return plt.figure()

        namespace.update(shap_ranking_figure=ranking_figure, relative_path=str,
                         save_publication_figure=lambda figure, path: {"pdf": path.with_suffix(".pdf")})
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({"axes.grid": False}):
            namespace["FIGURES"] = Path(directory)
            obsolete = [Path(directory) / f"shap_imv_results__{stem}.pdf" for stem in
                        ("titanic", "breast_cancer_adult_income", "titanic_breast_cancer_adult_income")]
            for path in obsolete:
                path.touch()
            unrelated = Path(directory) / "another_figure.pdf"
            unrelated.touch()
            before = plt.get_fignums()
            with patch.object(plt, "show"):
                paths = namespace["show_shap_figures"]()
            self.assertEqual(set(paths), {"figure_2"})
            self.assertEqual(Path(paths["figure_2"]).name, "figure_2.pdf")
            self.assertFalse(any(path.exists() for path in obsolete))
            self.assertTrue(unrelated.exists())
            self.assertTrue(all(path.endswith(".pdf") for path in paths.values()))
            self.assertFalse(mpl.rcParams["axes.grid"])
            self.assertEqual(plt.get_fignums(), before)
        self.assertEqual(observed, [True])

    def test_multiclass_export_restores_style_for_other_sections(self):
        definition = next(node for tree in self.trees for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "show_example")
        figure_styles = []

        def example_figure(result):
            figure_styles.append(mpl.rcParams["axes.grid"])
            self.assertEqual(mpl.rcParams["font.family"], figure_utils.FONT_FAMILY)
            return plt.figure()

        def save_figure(figure, path):
            return {"pdf": path.with_suffix(".pdf")}

        namespace = {
            "plt": plt, "figure_utils": figure_utils,
            "multiclass_results": {"fixture": {"title": "Fixture", "n_seeds": 10,
                                               "table": None, "global_summary": None}},
            "display": lambda table: None, "format_global_metrics": lambda table: "",
            "example_figure": example_figure, "save_publication_figure": save_figure,
            "relative_path": str, "MULTI_FIGURE_FILES": {"fixture": "figure_3"},
        }
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(NOTEBOOK), "exec"),
             namespace)
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({"axes.grid": False}):
            namespace["FIGURES"] = Path(directory)
            obsolete = Path(directory) / "multi_imv_results__fixture.pdf"
            obsolete.touch()
            original_figures = plt.get_fignums()
            with patch.object(plt, "show"), patch("builtins.print"):
                paths = namespace["show_example"]("fixture")
            self.assertEqual(set(paths), {"pdf"})
            self.assertEqual(Path(paths["pdf"]).name, "figure_3.pdf")
            self.assertFalse(obsolete.exists())
            self.assertFalse(mpl.rcParams["axes.grid"])
            self.assertEqual(plt.get_fignums(), original_figures)
        self.assertEqual(figure_styles, [True])


if __name__ == "__main__":
    unittest.main()
