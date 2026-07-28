# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import exp.run_temporal_tkgl as run_temporal_tkgl


def test_infer_num_nodes_accounts_for_static_edges():
    dataset = SimpleNamespace(
        num_nodes=4,
        static_data={
            "head": torch.tensor([1, 6], dtype=torch.long),
            "tail": torch.tensor([2, 3], dtype=torch.long),
        },
    )
    temporal_data = SimpleNamespace(
        src=torch.tensor([0, 1], dtype=torch.long),
        dst=torch.tensor([2, 3], dtype=torch.long),
    )

    assert run_temporal_tkgl._infer_num_nodes(dataset, temporal_data) == 7


def test_validate_edge_index_bounds_rejects_out_of_range_ids():
    edge_index = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="outside the configured graph size"):
        run_temporal_tkgl._validate_edge_index_bounds(edge_index, num_nodes=5, label="edge_index")
