# Source Layout

- `empirical/`: real-data examples, grouped into `ablation_imv/`, `gnn/`,
  `multi_imv/`, and `shap_imv/`.
- `simulations/`: synthetic ablation, multiclass, and SHAP examples, the
  complexity benchmark, and `vanilla_imv.py`.
- `plotter/plotter.ipynb`: combined results-only plotting and suggested captions.
- `plotter/multi_imv_results.ipynb`: multiclass metric tables and summary figures.
- `figure_utils.py`: project-wide plotting style, figure export, and result loaders.
- `shared_style.py`: panel layout helpers using the same font and colormap.

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
jupyter nbconvert --execute --to notebook --inplace src/plotter/*.ipynb
```

Figures remain in `output/figures` (PDF only), with tables in `output/tables`
and empirical metrics in `output/examples`. Dataset caches,
training artifacts, and output filenames do not depend on the example folders.
The GNN modules are imported as `empirical.gnn.gnn_training` and
`empirical.gnn.gnn_results`, with `src` on the Python path.
