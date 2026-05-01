# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.mamba_models import (
    MambaSheafDiffusion,
    TemporalEdgeMambaSheafDiffusion,
    TemporalMambaState,
)


def _make_args(num_nodes, input_dim, output_dim):
    return {
        'hidden_channels': 4,
        'device': torch.device('cpu'),
        'graph_size': num_nodes,
        'layers': 2,
        'normalised': True,
        'deg_normalised': False,
        'linear': False,
        'input_dropout': 0.0,
        'dropout': 0.0,
        'left_weights': True,
        'right_weights': True,
        'use_act': True,
        'second_linear': False,
        'add_lp': False,
        'add_hp': False,
        'max_t': 1.0,
        'sheaf_act': 'tanh',
        'edge_weights': True,
        'orth': 'householder',
        'sparse_learner': False,
        'd': 2,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'stateful_temporal': True,
        'closure_hops': 1,
        'temporal_d_model': 8,
    }


def _make_edge_index():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2], 
            [1, 0, 2, 1, 3, 2, 0, 1, 1, 2, 2, 3],
        ],
        dtype=torch.long,
    )
    return edge_index


def test_mamba_sheaf_diffusion_forward_and_temporal_step():
    torch.manual_seed(0)
    edge_index = _make_edge_index()
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=3)

    model = MambaSheafDiffusion(edge_index, args)
    model.eval()

    logits = model(x)
    assert list(logits.shape) == [4, 3]
    assert torch.isfinite(logits).all()

    model.reset_temporal_state()
    logits_1, state_1 = model.step(x, return_state=True)
    assert isinstance(state_1, TemporalMambaState)
    assert list(logits_1.shape) == [4, 3]
    assert list(state_1.memory.shape) == [4, model.hidden_dim]
    assert list(state_1.spatial.shape) == [4, model.hidden_dim]

    active_nodes = torch.tensor([0], dtype=torch.long)
    previous_state = TemporalMambaState(memory=state_1.memory.clone(), spatial=state_1.spatial.clone())
    _, state_2 = model.step(x, active_nodes=active_nodes, state=previous_state, return_state=True)

    assert torch.allclose(state_2.memory[2:], previous_state.memory[2:])
    assert not torch.allclose(state_2.spatial, state_1.spatial, atol=1e-5)


def test_mamba_sheaf_diffusion_forward_sequence():
    torch.manual_seed(1)
    edge_index = _make_edge_index()
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=3)

    model = MambaSheafDiffusion(edge_index, args)
    model.eval()

    snapshots = [
        {'x': x, 'edge_index': edge_index, 'active_nodes': torch.tensor([0, 1], dtype=torch.long)},
        {'x': x + 0.1, 'edge_index': edge_index, 'active_nodes': torch.tensor([2, 3], dtype=torch.long)},
    ]

    outputs, state = model.forward_sequence(snapshots)
    assert len(outputs) == 2
    assert list(outputs[0].shape) == [4, 3]
    assert list(outputs[1].shape) == [4, 3]
    assert state is not None
    assert list(state.memory.shape) == [4, model.hidden_dim]


def test_mamba_sheaf_diffusion_supports_disabling_left_and_right_weights():
    torch.manual_seed(2)
    edge_index = _make_edge_index()
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=3)
    args["left_weights"] = False
    args["right_weights"] = False

    model = MambaSheafDiffusion(edge_index, args)
    model.eval()

    logits = model(x)
    assert list(logits.shape) == [4, 3]
    assert torch.isfinite(logits).all()


def test_temporal_edge_mamba_sheaf_diffusion_tracks_destination_vocab():
    torch.manual_seed(3)
    edge_index = _make_edge_index()
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=7)
    args["destination_offset"] = 10
    args["destination_size"] = 5

    model = TemporalEdgeMambaSheafDiffusion(edge_index, args)
    model.eval()

    logits = model(x)
    assert list(logits.shape) == [4, 5]
    assert model.destination_offset == 10
    assert model.destination_size == 5

    dst_global = torch.tensor([10, 12, 14], dtype=torch.long)
    dst_local = model.global_to_local_destination(dst_global)
    assert torch.equal(dst_local, torch.tensor([0, 2, 4], dtype=torch.long))
    assert torch.equal(model.local_to_global_destination(dst_local), dst_global)
