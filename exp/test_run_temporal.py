# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import torch

import exp.run_temporal as run_temporal
from models.mamba_models import (
    TemporalMambaState,
    TemporalMambaSheafSSMOnlyDiffusion,
    TemporalMambaSheafSheafOnlyDiffusion,
)


def test_make_model_respects_output_dim():
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    x = torch.randn(2, 4)
    args = SimpleNamespace(
        stateful_temporal=True,
        closure_hops=1,
        temporal_d_model=8,
    )

    model = run_temporal._make_model(
        edge_index=edge_index,
        x=x,
        num_nodes=2,
        output_dim=3,
        device=torch.device("cpu"),
        args=args,
    )

    assert model.lin2.out_features == 3


def test_make_model_resolves_ablation_variants():
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    x = torch.randn(2, 4)

    sheaf_only_args = SimpleNamespace(
        model="TemporalMambaSheafSheafOnly",
        stateful_temporal=True,
        closure_hops=1,
        temporal_d_model=8,
    )
    sheaf_only_model = run_temporal._make_model(
        edge_index=edge_index,
        x=x,
        num_nodes=2,
        output_dim=3,
        device=torch.device("cpu"),
        args=sheaf_only_args,
    )

    ssm_only_args = SimpleNamespace(
        model="TemporalMambaSheafSSMOnly",
        stateful_temporal=True,
        closure_hops=1,
        temporal_d_model=8,
    )
    ssm_only_model = run_temporal._make_model(
        edge_index=edge_index,
        x=x,
        num_nodes=2,
        output_dim=3,
        device=torch.device("cpu"),
        args=ssm_only_args,
    )

    assert isinstance(sheaf_only_model, TemporalMambaSheafSheafOnlyDiffusion)
    assert isinstance(ssm_only_model, TemporalMambaSheafSSMOnlyDiffusion)


def test_run_epoch_uses_temporal_chunks_and_detaches_state(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward_sequence(self, snapshots, initial_state=None):
            self.calls.append({
                "chunk_len": len(snapshots),
                "initial_requires_grad": (
                    None if initial_state is None or initial_state.memory is None
                    else bool(initial_state.memory.requires_grad)
                ),
            })
            outputs = [self.weight.reshape(1, 1) for _ in snapshots]
            next_memory = self.weight.reshape(1, 1) + len(self.calls)
            next_state = TemporalMambaState(memory=next_memory, spatial=next_memory.clone())
            return outputs, next_state

    def fake_node_property_loss(outputs, dataset, snapshots):
        if not snapshots[0]["supervised"]:
            return torch.tensor(0.0)
        return outputs[0].sum()

    monkeypatch.setattr(run_temporal, "_node_property_loss", fake_node_property_loss)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshots = [
        {"supervised": True},
        {"supervised": False},
        {"supervised": True},
    ]

    loss = run_temporal._run_epoch(
        model,
        optimizer,
        snapshots,
        dataset=None,
        bptt_steps=2,
    )

    assert loss > 0.0
    assert [call["chunk_len"] for call in model.calls] == [2, 1]
    assert model.calls[0]["initial_requires_grad"] is None
    assert model.calls[1]["initial_requires_grad"] is False
    assert model.weight.item() != 1.0


def test_evaluate_model_streaming_returns_loss_and_state():
    class DummyDataset:
        eval_metric = "ndcg"

        def reset_label_time(self):
            self.calls = 0

        def get_node_label(self, ts):
            self.calls += 1
            if ts == 1:
                return None
            label_srcs = torch.tensor([0], dtype=torch.long)
            labels = torch.tensor([[1.0, 0.0]], dtype=torch.float)
            return None, label_srcs, labels

    class DummyModel(torch.nn.Module):
        def forward_sequence(self, snapshots, initial_state=None):
            logits = torch.log_softmax(torch.tensor([[2.0, 1.0]], dtype=torch.float), dim=-1)
            state = TemporalMambaState(
                memory=torch.ones(1, 1),
                spatial=torch.ones(1, 1) * len(snapshots),
            )
            return [logits], state

    snapshots = [
        SimpleNamespace(timestamp=torch.tensor(1)),
        SimpleNamespace(timestamp=torch.tensor(2)),
    ]

    metric, loss, state = run_temporal._evaluate_model_streaming(
        dataset_name="tgbn-trade",
        dataset=DummyDataset(),
        snapshots=snapshots,
        model=DummyModel(),
        initial_state=None,
        compute_metric=False,
        compute_loss=True,
    )

    assert metric != metric  # nan when metric computation is disabled
    assert loss >= 0.0
    assert isinstance(state, TemporalMambaState)


def test_evaluate_model_streaming_uses_tgb_evaluator(monkeypatch):
    calls = {}

    class DummyDataset:
        eval_metric = "ndcg"

        def reset_label_time(self):
            self.calls = 0

        def get_node_label(self, ts):
            self.calls += 1
            label_srcs = torch.tensor([0], dtype=torch.long)
            labels = torch.tensor([[1.0, 0.0]], dtype=torch.float)
            return None, label_srcs, labels

    class DummyModel(torch.nn.Module):
        def forward_sequence(self, snapshots, initial_state=None):
            logits = torch.log_softmax(torch.tensor([[2.0, 1.0]], dtype=torch.float), dim=-1)
            state = TemporalMambaState(
                memory=torch.ones(1, 1),
                spatial=torch.ones(1, 1),
            )
            return [logits], state

    class DummyEvaluator:
        def __init__(self, name):
            calls["name"] = name

        def eval(self, payload):
            calls["payload"] = payload
            return {"ndcg": 0.75}

    tgb_module = ModuleType("tgb")
    nodeproppred_module = ModuleType("tgb.nodeproppred")
    evaluate_module = ModuleType("tgb.nodeproppred.evaluate")
    evaluate_module.Evaluator = DummyEvaluator
    tgb_module.nodeproppred = nodeproppred_module
    nodeproppred_module.evaluate = evaluate_module

    monkeypatch.setitem(sys.modules, "tgb", tgb_module)
    monkeypatch.setitem(sys.modules, "tgb.nodeproppred", nodeproppred_module)
    monkeypatch.setitem(sys.modules, "tgb.nodeproppred.evaluate", evaluate_module)

    metric, loss, _ = run_temporal._evaluate_model_streaming(
        dataset_name="tgbn-trade",
        dataset=DummyDataset(),
        snapshots=[SimpleNamespace(timestamp=torch.tensor(2))],
        model=DummyModel(),
        initial_state=None,
        compute_metric=True,
        compute_loss=True,
    )

    assert calls["name"] == "tgbn-trade"
    assert set(calls["payload"].keys()) == {"y_pred", "y_true", "eval_metric"}
    assert metric == 0.75
    assert loss >= 0.0
