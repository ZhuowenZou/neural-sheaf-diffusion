# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from exp.temporal_utils import build_temporal_snapshots, sequence_loss


def test_build_temporal_snapshots_groups_by_timestamp():
    src = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    timestamps = torch.tensor([5, 5, 7, 7], dtype=torch.long)
    node_features = torch.randn(4, 3)

    snapshots = build_temporal_snapshots(src, dst, timestamps, node_features)

    assert len(snapshots) == 2
    assert torch.equal(snapshots[0].src, torch.tensor([0, 1]))
    assert torch.equal(snapshots[1].src, torch.tensor([2, 1]))
    assert list(snapshots[0].edge_index.shape) == [2, 4]
    assert list(snapshots[1].edge_index.shape) == [2, 4]
    assert torch.equal(snapshots[0].active_nodes, torch.tensor([0, 1, 2]))


def test_build_temporal_snapshots_groups_by_time_window():
    src = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
    timestamps = torch.tensor([10, 11, 14, 15, 21], dtype=torch.long)
    node_features = torch.randn(5, 3)

    snapshots = build_temporal_snapshots(
        src,
        dst,
        timestamps,
        node_features,
        time_window=5,
    )

    assert len(snapshots) == 3
    assert torch.equal(snapshots[0].src, torch.tensor([0, 1, 2]))
    assert torch.equal(snapshots[1].src, torch.tensor([3]))
    assert torch.equal(snapshots[2].src, torch.tensor([4]))
    assert int(snapshots[0].timestamp.item()) == 14
    assert int(snapshots[1].timestamp.item()) == 15
    assert int(snapshots[2].timestamp.item()) == 21


def test_build_temporal_snapshots_rejects_non_positive_time_window():
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 0], dtype=torch.long)
    timestamps = torch.tensor([5, 7], dtype=torch.long)
    node_features = torch.randn(2, 3)

    try:
        build_temporal_snapshots(src, dst, timestamps, node_features, time_window=0)
    except ValueError as exc:
        assert "time_window must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive time_window")


def test_sequence_loss_uses_matching_snapshot_order():
    logits_0 = torch.log_softmax(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), dim=-1)
    logits_1 = torch.log_softmax(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), dim=-1)
    snapshots = [
        type('Snapshot', (), {'src': torch.tensor([0, 1]), 'dst': torch.tensor([0, 1])}),
        type('Snapshot', (), {'src': torch.tensor([0, 1]), 'dst': torch.tensor([1, 0])}),
    ]

    loss = sequence_loss([logits_0, logits_1], snapshots)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
