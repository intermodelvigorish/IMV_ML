# Source Layout

- `empirical/`: real-data examples, grouped into `ablation_imv/`, `gnn/`,
  `multi_imv/`, and `shap_imv/`, with results-only plotting in `plotter/`.
- `simulations/`: synthetic ablation, multiclass, and SHAP examples, the
  complexity benchmark, and `vanilla_imv.py`.
- `empirical/plotter/plotter.ipynb`: combined ablation and GNN figures, multiclass
  metric tables and figures, and suggested captions. Runs after all producers.
- `empirical/plotter/figure_utils.py`: shared plotting style, PDF export, and
  result loaders, used by the example notebooks.
- `empirical/plotter/shared_style.py`: panel layout helpers using the same font
  and colormap.

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
jupyter nbconvert --execute --to notebook --inplace src/empirical/plotter/plotter.ipynb
```

Figures remain in `output/figures` (PDF only), with tables in `output/tables`
and empirical metrics in `output/examples`. Dataset caches,
training artifacts, and output filenames do not depend on the example folders.
The GNN modules are imported as `empirical.gnn.gnn_training` and
`empirical.gnn.gnn_results`, with `src` on the Python path.
Plotting helpers are imported from `empirical.plotter.figure_utils` and
`empirical.plotter.shared_style`.
