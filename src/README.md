# Source Layout

- `empirical/`: real-data examples, grouped into `ablation_imv/`, `gnn/`,
  `multi_imv/`, and `shap_imv/`, with results-only plotting in `plotter/`.
- `simulations/`: synthetic ablation, multiclass, and SHAP examples, the
  complexity benchmark, and `vanilla_imv.py`.
- `empirical/plotter/plotter.ipynb`: the single results-only notebook, ordered
  SHAP-IMV, Multi-IMV, Ablate-IMV, then GNN training; figures, CSV summaries, manuscript LaTeX tables,
  and suggested captions. Runs after all producers.
- `empirical/plotter/figure_utils.py`: shared plotting styles, panel layouts,
  PDF export, and result loaders, used by all example notebooks.

From the repository root, preview the execution order without computing:

```bash
bash run_all.sh --resume --dry-run
```

Run all notebooks, reusing valid cached results where supported:

```bash
bash run_all.sh --resume
```

Rebuild figures from completed empirical results without retraining:

```bash
python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=imv-ml src/empirical/plotter/plotter.ipynb
```

Figures remain in `output/figures` (PDF only), with tables in `output/tables`
and empirical metrics in `output/examples`. Dataset caches,
training artifacts, and output filenames do not depend on the example folders.
The GNN modules are imported as `empirical.gnn.gnn_training` and
`empirical.gnn.gnn_results`, with `src` on the Python path.
All plotting helpers are imported from `empirical.plotter.figure_utils`.
Use `configure_plotting()` for the base font/palette, or
`plt.rc_context(figure_utils.rc_params())` for multiclass and SHAP panel styling.
`label_panels(axes)` applies bold, left-aligned `a.`, `b.`, ... labels in row-major
order; pass only the data axes, excluding colourbars. Put descriptions in the
caption, not panel titles or a figure-wide title. PDF export removes any old
PNG/SVG files with the same stem.

`save_publication_figure()` calls the shared `apply_tight_layout(figure)` before
export, fitting labels and inset colourbars while reserving space for bottom
figure legends. Use the same helper for previews without exporting. Keep
gridspec margins unset so tight layout can size the rows and columns.

Use `plot_bars(axis, labels, values, yerr=spread)` for black outlines, consistent
three-decimal annotations beyond the error caps, and common grid/error-bar
styling. `set_bar_limits(axes)` reserves room for these annotations across shared
y-axis groups, including negative values. `colors=` preserves any sign encoding.
`add_heatmap_colorbar(mappable, axis, label)` adds a wider, full-height vector
colourbar as an inset without resizing the panels or changing row spacing.
`style_heatmap_axes(axis)` adds black spines and matching class labels on all four
edges; the colourbar automatically clears the right-hand labels after layout.
The shared constants in `figure_utils.py` control annotation size/padding, border
widths, and colourbar width/gap for every caller.

The SHAP ranking overview combines Titanic, Breast Cancer, and Adult Income
as three rows, with logistic regression, XGBoost, and LightGBM as columns.
It uses `figure_size(..., height=COMPACT_PANEL_HEIGHT)` for shorter bar panels
and exports `figure_2.pdf`, retiring the superseded split/combined filenames
only after that PDF saves successfully.

The multiclass overviews export Dry Bean as `figure_3.pdf` and combine Car
Evaluation and Nursery into `figure_A4.pdf`. The shared
`plot_multiclass_overview(results, model_names)` stacks one heatmap/bar block per
dataset. The supplement has four rows by three columns, with Car Evaluation in
panels a.-f. and Nursery in g.-l. Its export and caption follow section 2.5;
section 2.4 retains the Figure 3 caption. Larger annotations are controlled by
`MULTICLASS_ANNOTATION_FONT_SIZE`. The old separate `figure_A5.pdf` is retired only
after the combined PDF saves successfully.

Section 3 exports the complete MNIST, UCI HAR and MAGIC Gamma ablation overview
as `figure_4.pdf`, replacing the manuscript's former Figures 4 and 6 with one
2 x 3 figure. The accompanying caption distinguishes directional pairwise IMV
from IMV against the constant 0.5 baseline and reports seed standard deviations,
not confidence intervals. The previous descriptive overview filename is retired
only after the numbered PDF saves successfully.

Section 4 exports all six GNN datasets as `figure_5.pdf`, with a matching caption
in the notebook. Means and minimum-to-maximum seed bands include every epoch
from 0 to 200, without smoothing. The individual-seed diagnostic remains a
separate export; the old descriptive learning-curves filename is retired only
after Figure 5 saves successfully.
