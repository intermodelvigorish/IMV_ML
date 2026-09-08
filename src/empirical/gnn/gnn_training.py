"""Reproducible, checkpointed TU graph classification learning curves.

The native sparse GCN implements the same normalized adjacency operation as
GCNConv. Sparse matrix multiplication avoids allocating an edge-by-hidden
message tensor, which is particularly expensive for DD.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import random
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
import imvpy
from imvpy import get_w, imv_from_likelihoods, information_deficit, ll


DATASETS = ("PROTEINS", "NCI1", "NCI109", "Mutagenicity", "AIDS", "DD")
SEEDS = tuple(range(42, 52))
METRICS = ("imv", "accuracy", "precision", "recall")
EXTRA_METRICS = ("balanced_accuracy", "roc_auc", "f1", "brier", "log_loss")


@dataclass(frozen=True)
class GNNConfig:
    epochs: int = 200
    hidden: int = 64
    learning_rate: float = 0.001
    batch_size: int = 512
    dropout: float = 0.5
    weight_decay: float = 0.0
    feature_mode: str = "native"
    max_nodes: int = 20000
    cpu_threads: int = 2
    checkpoint_every: int = 10

    def __post_init__(self):
        for key in ("epochs", "hidden", "batch_size", "max_nodes", "cpu_threads", "checkpoint_every"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (not all(np.isfinite(v) for v in (self.dropout, self.learning_rate, self.weight_decay))
                or not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0):
            raise ValueError("Invalid dropout, learning rate, or weight decay")
        if self.feature_mode not in ("native", "native+degree"):
            raise ValueError("feature_mode must be native or native+degree")


def cache_paths():
    cache = Path(os.environ.get("IMV_CACHE_HOME", Path.home() / ".cache/imv")).expanduser()
    data = Path(os.environ.get("IMV_DATA_CACHE", cache / "datasets")).expanduser() / "tu"
    artifacts = Path(os.environ.get("IMV_ARTIFACT_CACHE", cache / "notebook_artifacts")).expanduser()
    return data.resolve(), (artifacts / "gnn_training").resolve()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_csv(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        table.to_csv(handle, index=False)
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_torch(path, value):
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as handle:
        temporary = handle.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _run_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A GNN run is already active in {directory}") from error
        yield


@dataclass
class Graph:
    x: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    y: int


def normalized_edges(edge_index, nodes):
    """Undirected, unweighted A+I with duplicate edges and self-loops removed."""
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    loops = torch.arange(nodes)
    edges = torch.cat((edges, edges.flip(0), torch.stack((loops, loops))), dim=1)
    sparse = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (nodes, nodes)).coalesce()
    indices = sparse.indices()
    degree = torch.bincount(indices[0], minlength=nodes).float()
    weights = (degree[indices[0]] * degree[indices[1]]).rsqrt()
    return indices, weights


@lru_cache(maxsize=2)
def load_graphs(name, data_root):
    from torch_geometric.datasets import TUDataset
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    dataset = TUDataset(str(data_root), name, use_node_attr=True, cleaned=False)
    labels = np.array([int(graph.y.item()) for graph in dataset])
    if set(labels) != {0, 1}:
        raise ValueError(f"{name} must have exactly two graph classes")
    graphs = []
    for graph in dataset:
        if graph.x is None or not torch.isfinite(graph.x).all() or graph.num_nodes < 1:
            raise ValueError(f"Invalid node features in {name}")
        indices, weights = normalized_edges(graph.edge_index, graph.num_nodes)
        graphs.append(Graph(graph.x.float(), indices, weights, int(graph.y.item())))
    raw_hash = hashlib.sha256()
    for path in sorted(Path(dataset.raw_dir).glob("*.txt")):
        raw_hash.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                raw_hash.update(block)
    metadata = {
        "dataset": name, "graphs": len(graphs), "class_counts": np.bincount(labels).tolist(),
        "node_features": dataset.num_node_features,
        "continuous_attributes": dataset.num_node_attributes,
        "raw_sha256": raw_hash.hexdigest(),
        "source": f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip",
        "label_encoding": "TU/PyG label order mapped to 0 and 1; no semantic relabeling",
        "raw_label_order": np.unique(np.loadtxt(
            Path(dataset.raw_dir) / f"{name}_graph_labels.txt", dtype=int)).tolist(),
    }
    return graphs, metadata


def split_graphs(labels, seed):
    """Stratified 70/15/15 split, with a different partition for each seed."""
    ids = np.arange(len(labels))
    train, remainder = train_test_split(ids, test_size=0.30, stratify=labels, random_state=seed)
    validation, test = train_test_split(
        remainder, test_size=0.5, stratify=np.asarray(labels)[remainder], random_state=seed
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def prepare_features(graphs, train_ids, n_attributes, mode):
    """Keep categorical one-hot columns; standardize continuous columns on train nodes only."""
    features = []
    for graph in graphs:
        x = graph.x.clone()
        if mode == "native+degree":
            degree = torch.bincount(graph.indices[0], minlength=len(x)).float() - 1
            x = torch.cat((x, degree.log1p().unsqueeze(1)), dim=1)
        features.append(x)
    columns = list(range(n_attributes))
    if mode == "native+degree":
        columns.append(features[0].shape[1] - 1)
    if columns:
        values = torch.cat([features[i][:, columns] for i in train_ids]).double()
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False)
        scale[scale < 1e-12] = 1
        for x in features:
            x[:, columns] = ((x[:, columns].double() - mean) / scale).float()
    else:
        mean = scale = torch.empty(0)
    prepared = [Graph(x, g.indices, g.weights, g.y) for x, g in zip(features, graphs)]
    return prepared, {"columns": columns, "mean": mean.tolist(), "scale": scale.tolist()}


def microbatches(ids, graphs, max_nodes):
    batch, nodes = [], 0
    for index in ids:
        count = len(graphs[index].x)
        if batch and nodes + count > max_nodes:
            yield batch
            batch, nodes = [], 0
        batch.append(int(index))
        nodes += count
    if batch:
        yield batch


def pack_graphs(graphs, ids, device):
    xs, indices, weights, graph_ids, labels = [], [], [], [], []
    offset = 0
    for batch_id, index in enumerate(ids):
        graph = graphs[index]
        xs.append(graph.x)
        indices.append(graph.indices + offset)
        weights.append(graph.weights)
        graph_ids.append(torch.full((len(graph.x),), batch_id, dtype=torch.long))
        labels.append(graph.y)
        offset += len(graph.x)
    x = torch.cat(xs).to(device)
    adjacency = torch.sparse_coo_tensor(
        torch.cat(indices, dim=1), torch.cat(weights), (offset, offset)
    ).coalesce().to(device)
    membership = torch.cat(graph_ids)
    sizes = torch.bincount(membership).float()
    pooling = torch.sparse_coo_tensor(
        torch.stack((membership, torch.arange(offset))), 1 / sizes[membership], (len(ids), offset)
    ).coalesce().to(device)
    return x, adjacency, pooling, torch.tensor(labels, dtype=torch.float32, device=device)


class SparseGCN(nn.Module):
    """Three GCN layers, ReLU, global mean pooling, dropout, binary logit."""
    def __init__(self, in_features, hidden=64, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, hidden, bias=False),
            nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False),
        ])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(3)])
        self.classifier = nn.Linear(hidden, 1)
        self.dropout = dropout
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, adjacency, pooling):
        for layer, bias in zip(self.layers, self.biases):
            x = F.relu(torch.sparse.mm(adjacency, layer(x)) + bias)
        x = torch.sparse.mm(pooling, x)
        return self.classifier(F.dropout(x, p=self.dropout, training=self.training)).squeeze(-1)


def score_predictions(y, probabilities, null_probability):
    """IMV is primary; preserve the package's below-chance behavior explicitly."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    started = time.perf_counter()
    likelihood = ll(y, probabilities)
    null_likelihood = ll(y, null_probability)
    imv = imv_from_likelihoods(null_likelihood, likelihood)
    imv_seconds = time.perf_counter() - started
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "imv": imv, "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, probabilities),
        "f1": f1_score(y, predicted, zero_division=0),
        "precision_weighted": precision_score(y, predicted, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, predicted, average="weighted", zero_division=0),
        "precision_macro": precision_score(y, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(y, predicted, average="macro", zero_division=0),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, np.clip(probabilities, 1e-9, 1 - 1e-9), labels=[0, 1]),
        "likelihood": likelihood, "null_likelihood": null_likelihood,
        "null_probability": float(null_probability),
        "information_deficit": information_deficit(likelihood),
        "imv_status": ("undefined" if not np.isfinite(imv) else
                       "chance_boundary" if likelihood < 0.5 else "defined"),
        "imv_fair_coin": imv_from_likelihoods(0.5, likelihood),
        "imv_ceiling": 1 / get_w(null_likelihood) - 1,
        "imv_seconds": imv_seconds, "n_graphs": len(y),
    }


def _predict(model, batches):
    probabilities, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, adjacency, pooling, y in batches:
            probabilities.append(model(x, adjacency, pooling).sigmoid().cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _device_fingerprint(device):
    return {"type": torch.device(device).type,
            "name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu"}


def job_metadata(name, seed, config, data_metadata, device):
    import sklearn
    import torch_geometric
    return {
        "dataset": name, "seed": seed, "config": asdict(config),
        "raw_sha256": data_metadata["raw_sha256"],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "pyg": torch_geometric.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__, "imvpy": imvpy.__version__},
        "device": _device_fingerprint(device),
    }


def run_seed(name, seed, config, data_root, artifact_root, device="cpu", fresh=False):
    """Resume exactly at the last saved epoch, including optimizer and RNG state."""
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw, data_metadata = load_graphs(name, str(data_root))
    metadata = job_metadata(name, seed, config, data_metadata, device)
    job_id = _hash(metadata)
    directory = Path(artifact_root) / "runs" / name / f"seed_{seed}_{job_id[:16]}"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.pt"
    if not fresh and (directory / "complete.json").exists():
        marker = json.loads((directory / "complete.json").read_text())
        metrics_path = directory / "metrics.csv"
        predictions_path = directory / "test_predictions.npz"
        if (marker["job_id"] == job_id and metrics_path.is_file()
                and predictions_path.is_file()
                and hashlib.sha256(metrics_path.read_bytes()).hexdigest() == marker["metrics_sha256"]
                and hashlib.sha256(predictions_path.read_bytes()).hexdigest() == marker.get("predictions_sha256")):
            print(f"Restored {name} seed {seed}: {config.epochs} epochs", flush=True)
            return {"dataset": name, "seed": seed, "job_id": job_id, "directory": str(directory)}
    splits = split_graphs([g.y for g in raw], seed)
    graphs, scaling = prepare_features(raw, splits["train"], data_metadata["continuous_attributes"],
                                       config.feature_mode)
    null_probability = float(np.mean([graphs[i].y for i in splits["train"]]))
    model = SparseGCN(graphs[0].x.shape[1], config.hidden, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    evaluation = {
        split: [pack_graphs(graphs, ids, device) for ids in microbatches(
            splits[split], graphs, config.max_nodes)]
        for split in ("validation", "test")
    }
    _atomic_json(directory / "manifest.json", {
        **metadata, "job_id": job_id, "data": data_metadata, "scaling": scaling,
        "splits": {key: ids.tolist() for key, ids in splits.items()}, "null_probability": null_probability,
    })
    history, predictions, start = [], [], 0
    if not fresh and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["job_id"] != job_id:
            raise ValueError(f"Checkpoint fingerprint mismatch: {state_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        history, predictions, start = state["history"], state["predictions"], state["epoch"] + 1
        print(f"Resuming {name} seed {seed} at epoch {start}", flush=True)
    for epoch in range(start, config.epochs + 1):
        started = time.perf_counter()
        training_loss = float("nan")
        if epoch:
            model.train()
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(splits["train"])
            losses = []
            for offset in range(0, len(order), config.batch_size):
                logical_batch = order[offset:offset + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for ids in microbatches(logical_batch, graphs, config.max_nodes):
                    x, adjacency, pooling, y = pack_graphs(graphs, ids, device)
                    loss = F.binary_cross_entropy_with_logits(model(x, adjacency, pooling), y, reduction="sum")
                    (loss / len(logical_batch)).backward()
                    losses.append(float(loss.detach()))
                optimizer.step()
            training_loss = sum(losses) / len(order)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        for split, batches in evaluation.items():
            started = time.perf_counter()
            y, p = _predict(model, batches)
            evaluation_seconds = time.perf_counter() - started
            history.append({"dataset": name, "seed": seed, "epoch": epoch, "split": split,
                            **score_predictions(y, p, null_probability), "train_loss": training_loss,
                            "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds})
            if split == "test":
                predictions.append(p.astype(np.float32))
        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            _atomic_torch(state_path, {
                "job_id": job_id, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
                "history": history, "predictions": predictions,
            })
        if epoch % 25 == 0 or epoch == config.epochs:
            last = history[-1]
            print(f"{name} seed {seed} epoch {epoch}/{config.epochs}: "
                  f"IMV={last['imv']:.4f}, accuracy={last['accuracy']:.4f}", flush=True)
    table = pd.DataFrame(history)
    _atomic_csv(directory / "metrics.csv", table)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as handle:
        temporary = handle.name
    np.savez_compressed(temporary, probabilities=np.stack(predictions),
                        y=np.array([graphs[i].y for i in splits["test"]]), graph_ids=splits["test"],
                        epochs=np.arange(config.epochs + 1))
    os.replace(temporary, directory / "test_predictions.npz")
    _atomic_json(directory / "complete.json", {
        "job_id": job_id, "metrics_sha256": hashlib.sha256((directory / "metrics.csv").read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((directory / "test_predictions.npz").read_bytes()).hexdigest(),
    })
    return {"dataset": name, "seed": seed, "job_id": job_id, "directory": str(directory)}


def execution_plan(config, jobs=None, device="auto"):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if jobs is not None and (not isinstance(jobs, int) or jobs < 1):
        raise ValueError("GNN jobs must be positive integers")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA-enabled PyTorch and a working driver")
    if use_cuda:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if jobs is not None:
            devices = devices[:max(1, jobs)]
        return devices
    maximum = max(1, cpus // config.cpu_threads)
    requested = int(jobs if jobs is not None else os.environ.get("IMV_GNN_JOBS", min(6, maximum)))
    if requested < 1:
        raise ValueError("GNN jobs must be positive")
    return ["cpu"] * min(requested, maximum)


def _run_lane(tasks, config, data_root, artifact_root, device, fresh):
    return [run_seed(name, seed, config, data_root, artifact_root, device, fresh) for name, seed in tasks]


def run_experiments(config=None, *, datasets=DATASETS, seeds=SEEDS, jobs=None, device="auto",
                    data_root=None, artifact_root=None, fresh=False):
    """One process per GPU, or bounded CPU seed workers; never share a GPU implicitly."""
    config = config or GNNConfig()
    datasets, seeds = tuple(datasets), tuple(seeds)
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
        raise ValueError("datasets must be distinct supported TU datasets")
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) != s or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    default_data, default_artifact = cache_paths()
    data_root = Path(data_root or default_data)
    artifact_root = Path(artifact_root or default_artifact)
    devices = execution_plan(config, jobs, device)
    tasks = [(name, seed) for name in datasets for seed in seeds]
    devices = devices[:len(tasks)]
    print(f"GNN: {len(tasks)} runs x {config.epochs} epochs; devices={devices}; "
          f"{config.cpu_threads} CPU threads/worker; features={config.feature_mode}", flush=True)
    with _run_lock(artifact_root):
        torch.set_num_threads(config.cpu_threads)
        inventory = [load_graphs(name, str(data_root))[1] for name in datasets]
        records = []
        if len(devices) == 1:
            records = _run_lane(tasks, config, data_root, artifact_root, devices[0], fresh)
        else:
            pool = ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn"))
            try:
                futures = [pool.submit(_run_lane, tasks[i::len(devices)], config, data_root,
                                       artifact_root, dev, fresh) for i, dev in enumerate(devices)]
                for future in as_completed(futures):
                    records.extend(future.result())
            except BaseException:
                # Python 3.12 has no public terminate_workers API. Do not leave
                # sibling lanes training after one fails or the user interrupts.
                for process in list(pool._processes.values()):
                    process.terminate()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        records.sort(key=lambda r: (datasets.index(r["dataset"]), seeds.index(r["seed"])))
        tables = [pd.read_csv(Path(r["directory"]) / "metrics.csv") for r in records]
        combined = pd.concat(tables, ignore_index=True)
        # Short smoke runs have separate result directories; never masquerade as the paper run.
        production = datasets == DATASETS and seeds == SEEDS and config == GNNConfig()
        result_dir = artifact_root / ("results" if production else "checks")
        if not production:
            result_dir /= _hash({"config": asdict(config), "datasets": datasets, "seeds": seeds})[:16]
        _atomic_csv(result_dir / "epoch_metrics.csv", combined)
        _atomic_json(result_dir / "manifest.json", {
            "config": asdict(config), "datasets": list(datasets), "seeds": list(seeds),
            "production": production, "inventory": inventory, "runs": records,
            "metrics_sha256": hashlib.sha256((result_dir / "epoch_metrics.csv").read_bytes()).hexdigest(),
            "interval": "pointwise Student-t 95% CI across seeds; not simultaneous or population uncertainty",
            "null": "training prevalence evaluated on the same held-out graphs as the GCN",
            "classification_average": "binary positive class 1; weighted and macro diagnostics also retained",
        })
        print(f"Completed results: {result_dir}", flush=True)
        return combined, result_dir
