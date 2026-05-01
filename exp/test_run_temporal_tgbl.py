# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import exp.run_temporal_tgbl as run_temporal_tgbl
from models.mamba_models import TemporalMambaState


def test_resolve_dataset_name_maps_v2_aliases():
    assert run_temporal_tgbl._resolve_dataset_name("tgbl-wiki-v2") == "tgbl-wiki"
    assert run_temporal_tgbl._resolve_dataset_name("tgbl-review-v2") == "tgbl-review"
    assert run_temporal_tgbl._resolve_dataset_name("tgbl-wiki") == "tgbl-wiki"


def test_global_to_local_destination_uses_destination_offset():
    destination_spec = {"offset": 7, "size": 4}
    dst = torch.tensor([7, 8, 10], dtype=torch.long)
    dst_local = run_temporal_tgbl._global_to_local_destination(dst, destination_spec)
    assert torch.equal(dst_local, torch.tensor([0, 1, 3], dtype=torch.long))


def test_global_to_local_destination_rejects_out_of_range_ids():
    destination_spec = {"offset": 5, "size": 3}
    dst = torch.tensor([4], dtype=torch.long)
    try:
        run_temporal_tgbl._global_to_local_destination(dst, destination_spec)
    except IndexError as exc:
        assert "destination vocabulary" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected destination remapping to reject out-of-range ids.")


def test_edge_prediction_loss_remaps_global_destinations():
    logits = torch.log_softmax(
        torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
            ],
            dtype=torch.float,
        ),
        dim=-1,
    )
    snapshot = SimpleNamespace(
        src=torch.tensor([0, 1], dtype=torch.long),
        dst=torch.tensor([5, 7], dtype=torch.long),
    )
    loss = run_temporal_tgbl._edge_prediction_loss(
        [logits],
        [snapshot],
        {"offset": 5, "size": 3},
    )
    assert loss.item() >= 0.0
def test_run_epoch_uses_temporal_chunks_and_detaches_state():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward_sequence(self, snapshots, initial_state=None):
            self.calls.append(
                {
                    "chunk_len": len(snapshots),
                    "initial_requires_grad": (
                        None if initial_state is None or initial_state.memory is None
                        else bool(initial_state.memory.requires_grad)
                    ),
                }
            )
            logits = torch.stack(
                [
                    torch.stack([self.weight, -self.weight]),
                    torch.stack([-self.weight, self.weight]),
                ],
                dim=0,
            )
            outputs = [torch.log_softmax(logits, dim=-1) for _ in snapshots]
            next_memory = self.weight.reshape(1, 1) + len(self.calls)
            next_state = TemporalMambaState(memory=next_memory, spatial=next_memory.clone())
            return outputs, next_state

    def fake_edge_prediction_loss(outputs, snapshots, destination_spec):
        if not snapshots[0]["supervised"]:
            return torch.tensor(0.0)
        return outputs[0].sum()

    original_loss = run_temporal_tgbl._edge_prediction_loss
    run_temporal_tgbl._edge_prediction_loss = fake_edge_prediction_loss
    try:
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        snapshots = [
            {"supervised": True},
            {"supervised": False},
            {"supervised": True},
        ]

        loss = run_temporal_tgbl._run_epoch(
            model,
            optimizer,
            snapshots,
            {"offset": 0, "size": 1},
            bptt_steps=2,
        )
    finally:
        run_temporal_tgbl._edge_prediction_loss = original_loss

    assert isinstance(loss, float)
    assert [call["chunk_len"] for call in model.calls] == [2, 1]
    assert model.calls[0]["initial_requires_grad"] is None
    assert model.calls[1]["initial_requires_grad"] is False
    assert model.weight.item() != 1.0


def test_evaluate_model_streaming_returns_loss_and_state_for_train_split():
    class DummyDataset:
        eval_metric = "mrr"

    class DummyModel(torch.nn.Module):
        def forward_sequence(self, snapshots, initial_state=None):
            logits = torch.log_softmax(torch.tensor([[2.0, 1.0]], dtype=torch.float), dim=-1)
            state = TemporalMambaState(
                memory=torch.ones(1, 1),
                spatial=torch.ones(1, 1) * len(snapshots),
            )
            return [logits], state

    snapshot = SimpleNamespace(
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([5], dtype=torch.long),
        edge_timestamps=torch.tensor([1], dtype=torch.long),
        edge_types=None,
    )

    metric, loss, state = run_temporal_tgbl._evaluate_model_streaming(
        dataset_name="tgbl-wiki",
        dataset=DummyDataset(),
        snapshots=[snapshot],
        model=DummyModel(),
        destination_spec={"offset": 5, "size": 2},
        initial_state=None,
        split_mode="train",
        compute_metric=False,
        compute_loss=True,
    )

    assert metric != metric
    assert loss >= 0.0
    assert isinstance(state, TemporalMambaState)
