# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch


@dataclass
class TemporalSnapshot:
    x: torch.Tensor
    edge_index: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    active_nodes: torch.Tensor
    timestamp: torch.Tensor
    edge_timestamps: Optional[torch.Tensor] = None
    edge_types: Optional[torch.Tensor] = None


def _make_bidirectional_edge_index(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    edge_index = torch.stack([src, dst], dim=0)
    reverse_edge_index = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, reverse_edge_index], dim=1).contiguous()


def build_temporal_snapshots(
    src: torch.Tensor,
    dst: torch.Tensor,
    timestamps: torch.Tensor,
    node_features: torch.Tensor,
    edge_ids: Optional[torch.Tensor] = None,
    max_edges: Optional[int] = None,
    time_window: Optional[int] = None,
    edge_types: Optional[torch.Tensor] = None,
) -> List[TemporalSnapshot]:
    if edge_ids is None:
        edge_ids = torch.arange(src.numel(), device=src.device)

    if max_edges is not None:
        edge_ids = edge_ids[:max_edges]

    if edge_ids.numel() == 0:
        return []

    if time_window is not None and time_window <= 0:
        raise ValueError("time_window must be positive when provided.")

    ordered = edge_ids[torch.argsort(timestamps[edge_ids])]
    ordered_ts = timestamps[ordered]

    snapshots: List[TemporalSnapshot] = []
    if time_window is None:
        bucket_keys = ordered_ts
    else:
        window_start = ordered_ts[0]
        bucket_keys = torch.div(ordered_ts - window_start, time_window, rounding_mode='floor')

    _, bucket_counts = torch.unique_consecutive(bucket_keys, return_counts=True)
    start = 0
    for count in bucket_counts.tolist():
        stop = start + count
        snapshot_edge_ids = ordered[start:stop]
        snapshot_src = src[snapshot_edge_ids].long()
        snapshot_dst = dst[snapshot_edge_ids].long()
        active_nodes = torch.unique(torch.cat([snapshot_src, snapshot_dst], dim=0))
        snapshot_ts = ordered_ts[stop - 1]
        snapshot_edge_types = None if edge_types is None else edge_types[snapshot_edge_ids].long()
        snapshots.append(
            TemporalSnapshot(
                x=node_features,
                edge_index=_make_bidirectional_edge_index(snapshot_src, snapshot_dst),
                src=snapshot_src,
                dst=snapshot_dst,
                active_nodes=active_nodes,
                timestamp=snapshot_ts,
                edge_timestamps=ordered_ts[start:stop],
                edge_types=snapshot_edge_types,
            )
        )
        start = stop

    return snapshots


def sequence_loss(outputs: Sequence[torch.Tensor], snapshots: Sequence[TemporalSnapshot]) -> torch.Tensor:
    if len(outputs) != len(snapshots):
        raise ValueError("Outputs and snapshots must have the same length.")

    losses = []
    for logits, snapshot in zip(outputs, snapshots):
        losses.append(torch.nn.functional.nll_loss(logits[snapshot.src], snapshot.dst))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)
