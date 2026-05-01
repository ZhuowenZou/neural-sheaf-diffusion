import torch

from models.mamba_models import TemporalMambaState
from models.sparse_temporal_mamba import (
    SparseTemporalEdgeMambaSheafDiffusion,
    SparseTemporalMambaSheafDiffusion,
)


def _make_args(num_nodes, input_dim, output_dim):
    return {
        "hidden_channels": 4,
        "device": torch.device("cpu"),
        "graph_size": num_nodes,
        "layers": 2,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "use_act": True,
        "second_linear": False,
        "add_lp": False,
        "add_hp": False,
        "max_t": 1.0,
        "sheaf_act": "tanh",
        "edge_weights": True,
        "orth": "householder",
        "sparse_learner": False,
        "d": 2,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "stateful_temporal": True,
        "closure_hops": 1,
        "temporal_d_model": 8,
    }


def _make_base_edge_index():
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2],
            [1, 0, 2, 1, 3, 2, 0, 1, 1, 2, 2, 3],
        ],
        dtype=torch.long,
    )


def _make_snapshot_edge_index(src, dst):
    edge_index = torch.tensor([[src, dst], [dst, src]], dtype=torch.long)
    return edge_index


def test_sparse_temporal_mamba_uses_snapshot_edge_index_in_sequence():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=3)
    model = SparseTemporalMambaSheafDiffusion(_make_base_edge_index(), args)
    model.eval()

    snapshot = {
        "x": x,
        "edge_index": _make_snapshot_edge_index(0, 1),
        "active_nodes": torch.tensor([0], dtype=torch.long),
    }
    direct_logits, direct_state = model.step(
        x,
        edge_index=snapshot["edge_index"],
        active_nodes=snapshot["active_nodes"],
        return_state=True,
    )

    model.reset_temporal_state()
    outputs, sequence_state = model.forward_sequence([snapshot])

    assert torch.allclose(outputs[0], direct_logits)
    assert torch.allclose(sequence_state.memory, direct_state.memory)
    assert torch.allclose(sequence_state.spatial, direct_state.spatial)


def test_sparse_temporal_mamba_preserves_inactive_global_state():
    torch.manual_seed(1)
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=3)
    model = SparseTemporalMambaSheafDiffusion(_make_base_edge_index(), args)
    model.eval()

    initial_memory = torch.randn(4, model.hidden_dim)
    initial_spatial = torch.randn(4, model.hidden_dim)
    state = TemporalMambaState(memory=initial_memory.clone(), spatial=initial_spatial.clone())

    _, next_state = model.step(
        x,
        edge_index=_make_snapshot_edge_index(0, 1),
        active_nodes=torch.tensor([0], dtype=torch.long),
        state=state,
        return_state=True,
    )

    assert torch.allclose(next_state.memory[2:], initial_memory[2:])
    assert torch.allclose(next_state.spatial[2:], initial_spatial[2:])
    assert not torch.allclose(next_state.memory[0], initial_memory[0])


def test_sparse_temporal_edge_mamba_tracks_destination_vocab():
    torch.manual_seed(2)
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=7)
    args["destination_offset"] = 10
    args["destination_size"] = 5

    model = SparseTemporalEdgeMambaSheafDiffusion(_make_base_edge_index(), args)
    model.eval()

    logits = model.step(
        x,
        edge_index=_make_snapshot_edge_index(0, 1),
        active_nodes=torch.tensor([0, 1], dtype=torch.long),
    )
    assert list(logits.shape) == [4, 5]

    dst_global = torch.tensor([10, 12, 14], dtype=torch.long)
    dst_local = model.global_to_local_destination(dst_global)
    assert torch.equal(dst_local, torch.tensor([0, 2, 4], dtype=torch.long))
    assert torch.equal(model.local_to_global_destination(dst_local), dst_global)
