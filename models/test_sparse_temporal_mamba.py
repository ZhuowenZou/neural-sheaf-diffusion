import torch

from models.mamba_models import TemporalMambaState
from models.sparse_temporal_mamba import (
    EventTemporalMambaSheafDiffusion,
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


def test_event_temporal_mamba_returns_event_outputs_and_scores_candidates():
    torch.manual_seed(3)
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=4)
    args["num_relations"] = 5
    args["train_negatives_per_pos"] = 3
    args["candidate_chunk_size"] = 2

    model = EventTemporalMambaSheafDiffusion(_make_base_edge_index(), args)
    model.eval()

    event_output, state = model.step(
        x,
        edge_index=_make_snapshot_edge_index(0, 1),
        active_nodes=torch.tensor([0, 1], dtype=torch.long),
        return_state=True,
    )

    assert set(event_output.keys()) == {"spatial", "x", "node_signal"}
    assert list(event_output["spatial"].shape) == [4, model.hidden_dim]
    assert event_output["node_signal"] is None
    assert event_output["x"] is x
    assert isinstance(state, TemporalMambaState)

    pos_scores = model.score_event_pairs(
        event_output,
        src=torch.tensor([0, 1], dtype=torch.long),
        dst=torch.tensor([1, 2], dtype=torch.long),
        edge_type=torch.tensor([0, 3], dtype=torch.long),
    )
    neg_scores = model.score_event_candidates(
        event_output,
        src=torch.tensor([0, 1], dtype=torch.long),
        dst_candidates=torch.tensor([[1, 2, 3], [0, 2, 3]], dtype=torch.long),
        edge_type=torch.tensor([0, 3], dtype=torch.long),
    )

    assert list(pos_scores.shape) == [2]
    assert list(neg_scores.shape) == [2, 3]
    assert torch.isfinite(pos_scores).all()
    assert torch.isfinite(neg_scores).all()


def test_event_temporal_mamba_forward_sequence_uses_snapshot_edge_index():
    torch.manual_seed(4)
    x = torch.randn(4, 6)
    args = _make_args(num_nodes=4, input_dim=6, output_dim=4)
    args["num_relations"] = 3

    model = EventTemporalMambaSheafDiffusion(_make_base_edge_index(), args)
    model.eval()

    snapshots = [
        {
            "x": x,
            "edge_index": _make_snapshot_edge_index(0, 1),
            "active_nodes": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "x": x + 0.2,
            "edge_index": _make_snapshot_edge_index(2, 3),
            "active_nodes": torch.tensor([2, 3], dtype=torch.long),
        },
    ]

    outputs, state = model.forward_sequence(snapshots)

    assert len(outputs) == 2
    assert list(outputs[0]["spatial"].shape) == [4, model.hidden_dim]
    assert outputs[1]["node_signal"] is None
    repr_out = model._event_node_repr(outputs[1], torch.tensor([2, 3], dtype=torch.long))
    assert list(repr_out.shape) == [2, model.hidden_dim]
    assert isinstance(state, TemporalMambaState)
