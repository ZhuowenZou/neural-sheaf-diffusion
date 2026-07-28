# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

import exp.temporal_benchmark_utils as benchmark_utils


def _tkg_spec():
    return benchmark_utils.DatasetSpec(
        requested_name="tkgl-smallpedia",
        loader_name="tkgl-smallpedia",
        task_family="tkg",
        dataset_module="tgb.linkproppred.dataset_pyg",
        evaluator_module="tgb.linkproppred.evaluate",
        metric_name="mrr",
        train_metric_supported=False,
    )


def test_make_static_edge_index_builds_bidirectional_edges():
    dataset = SimpleNamespace(
        static_data={
            "head": torch.tensor([0, 1], dtype=torch.long),
            "tail": torch.tensor([1, 2], dtype=torch.long),
        }
    )

    edge_index = benchmark_utils.make_static_edge_index(dataset)

    assert edge_index.shape == (2, 4)
    assert set(map(tuple, edge_index.t().tolist())) == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }


def test_build_supervised_snapshots_augments_tkg_snapshots_with_static_edges():
    temporal_data = SimpleNamespace(
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([2], dtype=torch.long),
        t=torch.tensor([5], dtype=torch.long),
        edge_type=torch.tensor([7], dtype=torch.long),
    )
    node_features = torch.randn(3, 4)
    static_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    snapshots, selected_edge_ids = benchmark_utils.build_supervised_snapshots(
        _tkg_spec(),
        dataset=SimpleNamespace(),
        temporal_data=temporal_data,
        edge_ids=torch.tensor([0], dtype=torch.long),
        node_features=node_features,
        static_edge_index=static_edge_index,
    )

    assert torch.equal(selected_edge_ids, torch.tensor([0], dtype=torch.long))
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert torch.equal(snapshot.edge_types, torch.tensor([7], dtype=torch.long))
    assert set(map(tuple, snapshot.edge_index.t().tolist())) == {
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
    }


def test_evaluate_model_streaming_for_tkg_passes_edge_types_to_negative_sampler(monkeypatch):
    calls = {}

    class DummyEvaluator:
        def __init__(self, name):
            calls["evaluator_name"] = name

        def eval(self, payload):
            calls["payload"] = payload
            return {"mrr": 0.5}

    class DummyNegativeSampler:
        def query_batch(self, src, dst, ts, edge_type=None, split_mode=None):
            calls["query_src"] = src.clone()
            calls["query_dst"] = dst.clone()
            calls["query_ts"] = ts.clone()
            calls["query_edge_type"] = None if edge_type is None else edge_type.clone()
            calls["query_split_mode"] = split_mode
            return [np.asarray([0, 1], dtype=np.int64)]

    class DummyDataset:
        eval_metric = "mrr"
        negative_sampler = DummyNegativeSampler()

        def load_val_ns(self):
            calls["loaded_val_ns"] = True

    class DummyModel(torch.nn.Module):
        def forward_sequence(self, snapshots, initial_state=None):
            logits = torch.log_softmax(torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float), dim=-1)
            state = SimpleNamespace(memory=torch.ones(1, 1), spatial=torch.ones(1, 1))
            return [logits], state

    monkeypatch.setattr(
        benchmark_utils,
        "pytest_importorskip",
        lambda module_name: SimpleNamespace(Evaluator=DummyEvaluator),
    )

    snapshot = SimpleNamespace(
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([2], dtype=torch.long),
        edge_timestamps=torch.tensor([9], dtype=torch.long),
        edge_types=torch.tensor([4], dtype=torch.long),
        timestamp=torch.tensor(9, dtype=torch.long),
    )

    metric, loss, state = benchmark_utils.evaluate_model_streaming(
        _tkg_spec(),
        DummyDataset(),
        [snapshot],
        DummyModel(),
        initial_state=None,
        split_mode="val",
        compute_metric=True,
        compute_loss=True,
    )

    assert calls["evaluator_name"] == "tkgl-smallpedia"
    assert calls["loaded_val_ns"] is True
    assert calls["query_split_mode"] == "val"
    assert torch.equal(calls["query_edge_type"], torch.tensor([4], dtype=torch.long))
    assert metric == 0.5
    assert loss >= 0.0
    assert state is not None


def test_evaluate_model_streaming_uses_event_scoring_branch(monkeypatch):
    calls = {}

    class DummyEvaluator:
        def __init__(self, name):
            calls["evaluator_name"] = name

        def eval(self, payload):
            calls["payload"] = payload
            return {"mrr": 0.25}

    class DummyNegativeSampler:
        def query_batch(self, src, dst, ts, edge_type=None, split_mode=None):
            calls["query_edge_type"] = None if edge_type is None else edge_type.clone()
            calls["query_split_mode"] = split_mode
            return [np.asarray([0, 1], dtype=np.int64)]

    class DummyDataset:
        eval_metric = "mrr"
        negative_sampler = DummyNegativeSampler()

        def load_test_ns(self):
            calls["loaded_test_ns"] = True

    class DummyEventModel(torch.nn.Module):
        supports_event_scoring = True
        graph_size = 3
        train_negatives_per_pos = 2

        def forward_sequence(self, snapshots, initial_state=None):
            output = {
                "spatial": torch.ones(3, 4),
                "node_signal": torch.ones(3, 4) * 2,
            }
            state = SimpleNamespace(memory=torch.ones(1, 1), spatial=torch.ones(1, 1))
            return [output], state

        def score_event_pairs(self, output, src, dst, edge_type=None):
            calls["score_pairs_edge_type"] = None if edge_type is None else edge_type.clone()
            return torch.tensor([0.5], dtype=torch.float, device=src.device)

        def score_event_candidates(self, output, src, dst_candidates, edge_type=None):
            calls["score_candidates_shape"] = tuple(dst_candidates.shape)
            calls["score_candidates_edge_type"] = None if edge_type is None else edge_type.clone()
            return torch.tensor([[0.1, -0.2]], dtype=torch.float, device=src.device)

    monkeypatch.setattr(
        benchmark_utils,
        "pytest_importorskip",
        lambda module_name: SimpleNamespace(Evaluator=DummyEvaluator),
    )

    snapshot = SimpleNamespace(
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([2], dtype=torch.long),
        edge_timestamps=torch.tensor([11], dtype=torch.long),
        edge_types=torch.tensor([1], dtype=torch.long),
        timestamp=torch.tensor(11, dtype=torch.long),
    )

    metric, loss, state = benchmark_utils.evaluate_model_streaming(
        _tkg_spec(),
        DummyDataset(),
        [snapshot],
        DummyEventModel(),
        initial_state=None,
        split_mode="test",
        compute_metric=True,
        compute_loss=True,
    )

    assert calls["evaluator_name"] == "tkgl-smallpedia"
    assert calls["loaded_test_ns"] is True
    assert calls["query_split_mode"] == "test"
    assert torch.equal(calls["query_edge_type"], torch.tensor([1], dtype=torch.long))
    assert calls["score_candidates_shape"] == (1, 2)
    assert torch.equal(calls["score_pairs_edge_type"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(calls["score_candidates_edge_type"], torch.tensor([1], dtype=torch.long))
    assert metric == 0.25
    assert loss >= 0.0
    assert state is not None
