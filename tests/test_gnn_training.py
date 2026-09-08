"""Small, offline scientific and checkpoint regression tests for the GNN study."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from imvpy import calculate_imv
from torch_geometric.nn import GCNConv, global_mean_pool

import gnn.gnn_training as training
from gnn.gnn_training import (
    GNNConfig, Graph, SparseGCN, execution_plan, normalized_edges, pack_graphs,
    prepare_features, run_seed, score_predictions, split_graphs,
)
from gnn.gnn_results import summarize_curves


class GNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def graphs(self, count=40):
        edges, weights = normalized_edges(torch.tensor([[0, 1], [1, 0]]), 2)
        return [Graph(torch.tensor([[float(i % 2), 1.0], [1.0, 0.5]]), edges, weights, i % 2)
                for i in range(count)]

    def test_normalization_handles_isolates_duplicates_and_loops(self):
        edges = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 0]])
        indices, weights = normalized_edges(edges, 3)
        matrix = torch.sparse_coo_tensor(indices, weights, (3, 3)).to_dense()
        torch.testing.assert_close(matrix, torch.tensor([[.5, .5, 0], [.5, .5, 0], [0, 0, 1.]]))

    def test_sparse_gcn_matches_pyg_forward_and_gradients(self):
        graphs = self.graphs(2)
        x, adjacency, pooling, _ = pack_graphs(graphs, [0, 1], "cpu")
        model = SparseGCN(2, hidden=4, dropout=0)
        convs = [GCNConv(2, 4), GCNConv(4, 4), GCNConv(4, 4)]
        for conv, layer, bias in zip(convs, model.layers, model.biases):
            conv.lin.weight.data.copy_(layer.weight.data)
            conv.bias.data.copy_(bias.data)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        h = x.clone()
        for conv in convs:
            h = conv(h, edge_index).relu()
        expected = model.classifier(global_mean_pool(h, torch.tensor([0, 0, 1, 1]))).squeeze(-1)
        actual = model(x, adjacency, pooling)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        for conv, layer in zip(convs, model.layers):
            torch.testing.assert_close(layer.weight.grad, conv.lin.weight.grad)

    def test_splits_are_stratified_disjoint_repeatable_and_seed_specific(self):
        y = np.tile([0, 1], 100)
        splits = split_graphs(y, 42)
        self.assertEqual([len(v) for v in splits.values()], [140, 30, 30])
        self.assertEqual(len(set(np.concatenate(list(splits.values())))), len(y))
        for name, ids in splits.items():
            np.testing.assert_array_equal(ids, split_graphs(y, 42)[name])
            self.assertEqual(set(y[ids]), {0, 1})
        self.assertFalse(np.array_equal(splits["train"], split_graphs(y, 43)["train"]))

    def test_scaling_does_not_use_held_out_nodes(self):
        graphs = self.graphs(4)
        first, scaling = prepare_features(graphs, [0, 1], 1, "native")
        graphs[3].x[:, 0] = 1000000
        second, changed = prepare_features(graphs, [0, 1], 1, "native")
        self.assertEqual(scaling, changed)
        torch.testing.assert_close(first[0].x, second[0].x)
        torch.testing.assert_close(first[0].x[:, 1], graphs[0].x[:, 1])
        self.assertTrue(torch.equal(graphs[3].x[:, 0], torch.full((2,), 1000000.)))

    def test_metrics_use_probability_imv_and_binary_precision_recall(self):
        y = np.array([0, 0, 0, 1])
        p = np.array([.1, .2, .6, .8])
        metrics = score_predictions(y, p, .25)
        self.assertAlmostEqual(metrics["imv"], calculate_imv(.25, p, y))
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["precision"], .5)
        self.assertEqual(metrics["recall"], 1.)
        self.assertNotEqual(metrics["precision"], metrics["precision_weighted"])
        self.assertLess(metrics["imv_ceiling"], 1)

    def test_undefined_imv_is_not_replaced_with_zero(self):
        with self.assertWarns(Warning):
            result = score_predictions([0, 1], [.999, .001], .5)
        self.assertTrue(np.isnan(result["imv"]))
        self.assertEqual(result["imv_status"], "undefined")
        self.assertLess(result["information_deficit"], 0)

    def test_range_uses_seed_extremes_without_dropping_failed_seeds(self):
        table = pd.DataFrame({"dataset": ["PROTEINS"] * 10, "split": ["test"] * 10,
                              "epoch": [1] * 10, "seed": list(range(42, 52)),
                              "imv": [-.2, .1, .1, .2, .2, .2, .3, .3, .4, .9]})
        row = summarize_curves(table).iloc[0]
        self.assertAlmostEqual(row["mean"], table.imv.mean())
        self.assertAlmostEqual(row["std"], table.imv.std(ddof=1))
        self.assertEqual(row.range_low, -.2)
        self.assertEqual(row.range_high, .9)
        table.loc[0, "imv"] = np.nan
        row = summarize_curves(table).iloc[0]
        self.assertEqual(row.n_valid, 9)
        self.assertTrue(np.isnan(row["mean"]))
        self.assertTrue(np.isnan(row.range_low))
        self.assertTrue(np.isnan(row.range_high))
        with self.assertRaises(ValueError):
            summarize_curves(table.iloc[:-1])

    def test_resume_restores_optimizer_and_rng_exactly(self):
        graphs = self.graphs()
        metadata = {"raw_sha256": "fixture", "continuous_attributes": 0}
        config = GNNConfig(epochs=4, hidden=4, batch_size=8, checkpoint_every=1, cpu_threads=1)
        with tempfile.TemporaryDirectory() as root, patch.object(training, "load_graphs", return_value=(graphs, metadata)):
            baseline = run_seed("PROTEINS", 42, config, root, Path(root) / "baseline")
            original_save = training._atomic_torch

            def interrupt_after_checkpoint(path, state):
                original_save(path, state)
                if state["epoch"] == 2:
                    raise InterruptedError("simulated stop after durable checkpoint")

            with patch.object(training, "_atomic_torch", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(InterruptedError):
                    run_seed("PROTEINS", 42, config, root, Path(root) / "resume")
            resumed = run_seed("PROTEINS", 42, config, root, Path(root) / "resume")
            with np.load(Path(baseline["directory"]) / "test_predictions.npz") as a:
                with np.load(Path(resumed["directory"]) / "test_predictions.npz") as b:
                    np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
            with patch.object(training, "SparseGCN", side_effect=AssertionError("Must not retrain")):
                cached = run_seed("PROTEINS", 42, config, root, Path(root) / "resume")
            self.assertEqual(cached["job_id"], resumed["job_id"])

    def test_execution_plan_bounds_cpu_jobs_and_rejects_unavailable_gpu(self):
        config = GNNConfig(cpu_threads=2)
        with patch.object(training.os, "sched_getaffinity", return_value=set(range(4))):
            self.assertEqual(execution_plan(config, jobs=50, device="cpu"), ["cpu", "cpu"])
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                execution_plan(config, device="cuda")
        with self.assertRaises(ValueError):
            execution_plan(config, jobs=0)

    def test_each_epoch_is_evaluated_even_between_checkpoints(self):
        graphs = self.graphs()
        metadata = {"raw_sha256": "fixture", "continuous_attributes": 0}
        config = GNNConfig(epochs=5, hidden=4, batch_size=8, checkpoint_every=3, cpu_threads=1)
        with tempfile.TemporaryDirectory() as root, patch.object(training, "load_graphs", return_value=(graphs, metadata)):
            with patch.object(training, "_predict", wraps=training._predict) as predict:
                with patch.object(training, "_atomic_torch", wraps=training._atomic_torch) as checkpoint:
                    result = run_seed("PROTEINS", 42, config, root, Path(root) / "results")
            self.assertEqual(predict.call_count, 2 * (config.epochs + 1))
            self.assertEqual(checkpoint.call_count, 3)  # Epochs 0, 3 and 5 only.
            table = pd.read_csv(Path(result["directory"]) / "metrics.csv")
            for split in ("validation", "test"):
                np.testing.assert_array_equal(table.query("split == @split").epoch, np.arange(6))
            with np.load(Path(result["directory"]) / "test_predictions.npz") as archive:
                np.testing.assert_array_equal(archive["epochs"], np.arange(6))
                self.assertFalse(np.any(np.all(np.diff(archive["probabilities"], axis=0) == 0, axis=1)))

    def test_spawned_worker_imports_package_and_unpickles_config(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            result = pool.submit(execution_plan, GNNConfig(cpu_threads=1), jobs=1, device="cpu")
            self.assertEqual(result.result(timeout=30), ["cpu"])


if __name__ == "__main__":
    unittest.main()
