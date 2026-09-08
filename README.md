# InterModel Vigorish for Interpretable Machine Learning

This repository contains the research code, worked examples, and publication
figures for *Mastering Model Fit: Applications of the InterModel Vigorish for
Interpretable Machine Learning*.

The examples use the [imvpy](https://github.com/intermodelvigorish/imvpy) library
to measure predictive improvement, compare full and ablated models, attribute
predictive gains to features, and track model performance during training.
This repository provides the experiments and plotting workflows; `imvpy`
provides the IMV calculations.

The workflow produces:

- Synthetic examples of vanilla IMV, Ablate-IMV, Multi-IMV, and SHAP-IMV.
- Three real-data ablation examples and their combined comparison figure.
- Multiclass and feature-attribution examples with classification metric tables.
- GNN learning curves on six graph datasets, with ten independent seeds per dataset.
- Publication figures, seed-level result tables, and a runtime benchmark.

All commands below are intended to be run from the repository root.

## Repository Layout

| Location | Contents |
| --- | --- |
| [src/empirical/ablation_imv/](src/empirical/ablation_imv/) | Full-versus-ablated model experiments. |
| [src/empirical/multi_imv/](src/empirical/multi_imv/) | Multiclass IMV, classification metrics, and tables. |
| [src/empirical/shap_imv/](src/empirical/shap_imv/) | SHAP-IMV and top-five feature-selection comparisons. |
| [src/empirical/gnn/](src/empirical/gnn/) | Graph-classification training, checkpoints, and epoch-level metrics. |
| [src/empirical/plotter/](src/empirical/plotter/) | One results-only notebook ordered SHAP, Multi-IMV, Ablate-IMV, then GNNs, plus shared plotting helpers. |
| [src/simulations/](src/simulations/) | Synthetic examples and the vanilla IMV complexity benchmark. |
| [output/](output/) | Saved PDF figures, empirical metric tables, and benchmark outputs. |
| [tests/](tests/) | Plotting, result-validation, training-restart, and source-layout checks. |
| [run_all.sh](run_all.sh) | Notebook execution order, installation option, fresh/resume modes, and process cleanup. |

The runner executes **15 notebooks**: ten empirical producers, four simulation
notebooks, and the combined plotter last. The standalone
[`vanilla_imv.py`](src/simulations/vanilla_imv.py) example is not a notebook and
is not executed by `run_all.sh`.

## Example Map

| Example family | Datasets | Models or comparisons |
| --- | --- | --- |
| Ablate-IMV | MNIST; UCI Human Activity Recognition; MAGIC Gamma | CNN, bidirectional GRU, and MLP, respectively, with architectural or feature-group ablations. |
| Multi-IMV | Car Evaluation; Dry Bean; Nursery | Logistic regression, XGBoost, and LightGBM; pairwise and one-vs-rest IMV. |
| SHAP-IMV | Adult Income; Breast Cancer; Titanic | Logistic regression, XGBoost, and LightGBM; feature attributions and top-five comparisons with SHAP, LIME, and Anchors. |
| GNN training | PROTEINS; NCI1; NCI109; Mutagenicity; AIDS; DD | Three-layer graph convolutional networks, evaluated at every epoch. |
| Simulations | Generated classification data | Minimal IMV examples and timings of the complete `imvpy.vanilla_imv` call. |

Dataset filtering, feature engineering, and model settings are documented in
the individual notebooks. The GNN study is a documented reconstruction of the
paper's protocol, not a claim of exact numerical replication.

## Setup

Use Linux or WSL with Bash, Git, and Python. The runner uses GNU command-line
tools, including `find`, `sort`, and `realpath`, plus `flock` and `setsid`.
The current development checks use Python 3.12.

Create an isolated environment outside the repository, install the dependencies,
and register its project-specific notebook kernel:

```bash
python3 -m venv "$HOME/.venvs/imv-ml"
source "$HOME/.venvs/imv-ml/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name imv-ml --display-name "Python (IMV_ML)"
```

Keep this environment active when running the commands below. Registering the
kernel lets the runner explicitly select the environment containing the installed
packages. Use `python -m nbconvert`, not a potentially unrelated `jupyter`
executable from Anaconda base. For interactive notebooks, launch Jupyter from
this environment and select **Python (IMV_ML)**.

[`requirements.txt`](requirements.txt) installs `imvpy[deep-learning]` from commit
`de7fe7a80300ae849fcdc37519651509b380a8d1`, rather than an unpinned latest release.
Other dependencies use version ranges or individual pins; this is not a fully
locked environment.

Public datasets are fetched by the source notebooks. No API keys are required,
but an internet connection is needed for uncached downloads, and some UCI
loaders may contact the source again during a resumed run.

## One-Command Workflows

### Preview the Run

```bash
bash run_all.sh --resume --dry-run
```

Prints the notebook order, worker settings, and output location without
installing dependencies, deleting outputs, or executing notebooks.

### Resume Existing Work

```bash
bash run_all.sh --resume
```

Reuses valid checkpoints where supported and preserves existing outputs.
Every notebook is still executed: simulations and uncached work can recompute,
and dataset loading can require network access. This is **not an offline mode**.
Add `--install` to install the requirements before execution.

### Clean Recompute

```bash
bash run_all.sh --fresh
```

**This deletes everything inside `output/`, including any manually added files.**
It bypasses experiment checkpoints and replaces results as computation completes.
Downloaded datasets and artifact-cache directories are not deleted.

**Running `bash run_all.sh` without a mode also selects `--fresh`.** Use an
explicit mode to avoid accidentally starting a destructive, expensive rerun.

### Rebuild Publication Figures Only

```bash
python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=imv-ml src/empirical/plotter/plotter.ipynb
```

Rebuilds the combined 3 x 3 SHAP ranking (`figure_2.pdf`), two multiclass figures,
the ablation overview, and GNN mean/range and individual-seed figures, in that order. It also writes
the multiclass and SHAP CSV summaries and five manuscript LaTeX tables: three
multiclass tables, the SHAP top-five comparison, and the SHAP ranking. This
notebook does not download datasets or train models. It requires completed ablation, GNN,
multiclass, and SHAP results, including files in the artifact cache; the committed
PDFs and summary tables alone are not sufficient.

### Recompute Only the SHAP Examples

After completing Setup, this uses the project interpreter and kernel explicitly:

```bash
(
  set -e
  source "$HOME/.venvs/imv-ml/bin/activate"
  export IMV_FORCE_RECOMPUTE=1
  for notebook in src/empirical/shap_imv/shap_imv_*.ipynb \
                  src/empirical/plotter/plotter.ipynb
  do
    python -m nbconvert --execute --to notebook --inplace \
      --ExecutePreprocessor.kernel_name=imv-ml \
      --ExecutePreprocessor.timeout=-1 "$notebook"
  done
)
```

This replaces the eight-feature SHAP results and then rebuilds the combined
figures and tables, using existing results for the other experiments. It does
not wipe `output/` or retrain the other experiment families. Use
`IMV_FORCE_RECOMPUTE=0` to reuse valid checkpoints after an interruption.

### Run a Small Worked Example

```bash
python src/simulations/vanilla_imv.py
```

Simulates a binary outcome, fits logistic regression, and prints held-out IMV
relative to the training-prevalence baseline. It does not launch the full suite.

## Current Defaults and Parallelism

Notebooks run **sequentially**; independent seed/model jobs run in parallel
inside the empirical notebooks. The combined plotter always runs last.

| Setting | Default | Purpose |
| --- | --- | --- |
| `IMV_N_JOBS` | `15` in the runner | Requested worker count for tabular seed/model jobs. |
| `IMV_TORCH_JOBS` | `6` in the runner | Requested ablation workers; actual concurrency is bounded by CPU resources, or one worker on GPU. |
| `IMV_GNN_DEVICE` | `auto` | Select `cpu`, `cuda`, or automatic CUDA detection for GNN training. |
| `IMV_GNN_JOBS` | Up to 6 on CPU; one per visible GPU | Limit GNN worker processes. CPU workers use two numerical threads each by default. |
| `CUDA_VISIBLE_DEVICES` | Visible devices supplied by the environment | Select which GPUs GNN workers can use. |

For a smaller CPU allocation:

```bash
IMV_N_JOBS=4 IMV_TORCH_JOBS=2 IMV_GNN_DEVICE=cpu IMV_GNN_JOBS=2 bash run_all.sh --resume
```

GPU execution requires CUDA-enabled PyTorch and a working driver. The GNN
scheduler assigns one worker per visible GPU rather than running multiple
training workers on the same device. Exact SHAP-IMV subset evaluation and
LIME/Anchors comparisons can be expensive even with parallel workers.

## Data and Caching

| Setting or directory | Default location | Contents |
| --- | --- | --- |
| `IMV_CACHE_HOME` | `~/.cache/imv` | Base directory for caches used by the experiments. |
| `IMV_DATA_CACHE` | `$IMV_CACHE_HOME/datasets` | Dataset files for loaders that support this cache. |
| `IMV_ARTIFACT_CACHE` | `$IMV_CACHE_HOME/notebook_artifacts` | Per-example checkpoints, predictions, manifests, and detailed result CSVs. |
| `IMV_FIGURE_DIR` | `output/figures` | Exported publication PDFs; use an absolute path for overrides across all notebooks. |
| `output/examples/` | Within the repository | Multiclass metric tables and SHAP feature-selection summaries. |
| `output/tables/` | Within the repository | Complexity measurements, metadata, and the LaTeX table. |

Keep artifact caches if you want to rebuild figures without retraining.
In fresh mode, `IMV_FIGURE_DIR` must remain inside `output/`.

The complexity notebook also copies its generated table to a locally detected
`Apps/Overleaf/IMV_ML_paper` manuscript, if present. `IMV_MANUSCRIPT_DIR` can
select a manuscript directory explicitly. For an Overleaf path, the notebook
requires `overleaf-sync-now` and refreshes the manuscript before copying.
Without a manuscript directory, it only exports the benchmark files locally.

## Metrics and Reproducibility

- The main empirical experiments use ten seeds, **42-51**. The minimal synthetic examples use a fixed seed instead.
- The ablation overview averages seed-level scores and uses sample standard deviations for whiskers. Its bar-chart null predicts a constant probability of **0.5**.
- GNN training uses a seed-specific stratified 70/15/15 train/validation/test split and 200 epochs. Epoch 0 and every training epoch are evaluated: **60 runs and 12,060 test evaluations** in the full experiment.
- GNN mean curves retain every epoch. Shading spans the **minimum and maximum across seeds**, not confidence intervals. There is no temporal smoothing or renderer path simplification; undefined IMV values remain gaps.
- GNN IMV uses a null estimated from training-set prevalence. Accuracy, precision, recall, and additional diagnostics are saved alongside IMV. Null definitions differ between examples and are documented rather than silently equated.
- GNN checkpoints preserve model, optimizer, random-number state, and history. Result loaders check completeness and consistency before drawing publication figures. Bitwise equality across hardware or library versions is not promised.

## Key Outputs

Figures are saved as **PDF only**, using Helvetica with font fallbacks and the
shared project colormap. Suggested ablation and GNN captions are included in the
[combined plotter notebook](src/empirical/plotter/plotter.ipynb).
Panels are labelled **a.**, **b.**, and so on, across rows and then down, with no
descriptive titles or figure-wide titles. Dataset/model descriptions belong in
the captions. Exporting a PDF removes older PNG/SVG copies of that same figure.
All bars use black outlines and three-decimal value labels placed beyond their
error caps. Shared helpers standardize error bars, annotation headroom and grids;
wider heatmap colourbars stay aligned to their panels without changing row spacing.

| Output | File or directory |
| --- | --- |
| Three-example ablation overview | [Figure 4: MNIST, UCI HAR and MAGIC Gamma](output/figures/figure_4.pdf) |
| Six-dataset GNN means and seed ranges | [Figure 5: GNN training](output/figures/figure_5.pdf) |
| Individual GNN seed trajectories | [gnn_training__six_datasets__individual_seed_imv.pdf](output/figures/gnn_training__six_datasets__individual_seed_imv.pdf) |
| Multiclass overview figures | [Figure 3: Dry Bean](output/figures/figure_3.pdf), [Figure A4: Car Evaluation and Nursery](output/figures/figure_A4.pdf) |
| Standalone ablation, multiclass, and SHAP-IMV figures | [output/figures/](output/figures/) |
| Multiclass metrics and top-five feature tables | [output/examples/](output/examples/) |
| Complexity table and timing audit files | [output/tables/](output/tables/) |

## Recovery Workflows

- **Interrupted run:** restart with `bash run_all.sh --resume`. Completed work is reused where supported; GNN jobs restart from their latest valid checkpoint.
- **Missing plotter inputs:** run the producer named in the error, with `python -m nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=imv-ml PATH_TO_NOTEBOOK`, then rerun the combined plotter. Point `IMV_ARTIFACT_CACHE` at the original results if they are stored elsewhere.
- **Another invocation is active:** the runner uses a process lock to prevent overlapping full runs. Wait for that invocation or interrupt its original terminal with `Ctrl+C`; the runner cleans up its active notebook process group.
- **Import errors in notebooks:** reactivate the environment, reinstall `requirements.txt` if necessary, and register its `imv-ml` kernel using the setup command above. `numpy.core.multiarray failed to import` with NumPy 2 indicates an incompatible compiled dependency; this project requires `numpy>=1.26,<2`. Do not repair Anaconda base or upgrade NumPy alone. Restart any interactive kernel after installing dependencies.

The runner checks imports and kernel availability before clearing outputs or
starting experiments. `IMV_KERNEL_NAME` overrides its default `imv-ml` kernel.
It reports each notebook's completion and stops on the first failure;
a quick resumed run can indicate restored checkpoints rather than new training.

## Validation Checks

```bash
python -m unittest discover -s tests -v
bash run_all.sh --resume --dry-run
```

The tests use small fixtures rather than the full empirical datasets. They
cover IMV result validation, PDF exports, epoch/seed coverage, exact checkpoint
restart, plotting-style isolation, and notebook discovery. PDF pixel checks
use `pdftoppm` from Poppler when it is available.

## License

GNU General Public License, version 3.0. See [LICENSE](LICENSE).
Third-party datasets and dependencies retain their own licenses and terms.
