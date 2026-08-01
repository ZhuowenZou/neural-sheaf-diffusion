"""Validation for FaithfulEventTemporalSheafDiffusion (sparse event path)."""

import torch

from models.faithful_event_model import FaithfulEventTemporalSheafDiffusion

torch.manual_seed(0)


def make_model(num_nodes=20, input_dim=8, num_relations=5, **overrides):
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    args = {
        "d": 2,
        "add_lp": False,
        "add_hp": False,
        "device": "cpu",
        "graph_size": num_nodes,
        "layers": 2,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "sparse_learner": False,
        "use_act": True,
        "input_dim": input_dim,
        "hidden_channels": 8,
        "output_dim": num_nodes,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
        "stateful_temporal": False,
        "closure_hops": 1,
        "temporal_d_model": 16,
        "feedback_dim": 8,
        "num_relations": num_relations,
        "train_negatives_per_pos": 4,
    }
    args.update(overrides)
    return FaithfulEventTemporalSheafDiffusion(edge_index, args), num_nodes


def event_snapshot(n, input_dim, n_events, ts, seed):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (n_events,), generator=g)
    dst = torch.randint(0, n, (n_events,), generator=g)
    return {
        "x": torch.randn(n, input_dim, generator=g),
        "edge_index": torch.stack([src, dst]),
        "active_nodes": torch.unique(torch.cat([src, dst])),
        "timestamp": torch.tensor(float(ts)),
    }, src, dst


def test_step_shapes_and_scoring_api():
    model, n = make_model()
    model.eval()
    snap, src, dst = event_snapshot(n, 8, 6, 0, seed=1)
    with torch.no_grad():
        outputs, state = model.forward_sequence([snap])
    out = outputs[0]
    assert out["spatial"].shape == (n, model.hidden_dim)
    assert state.memory.shape == (n, model.d_h)
    with torch.no_grad():
        pos = model.score_event_pairs(out, src, dst, edge_type=torch.zeros_like(src))
        assert pos.shape == src.shape
        cands = torch.randint(0, n, (src.numel(), 7))
        scores = model.score_event_candidates(out, src, cands, edge_type=torch.zeros_like(src))
        assert scores.shape == (src.numel(), 7)


def test_delta_t_changes_second_step():
    model, n = make_model()
    model.eval()
    outs = {}
    for gap in [1, 40]:
        model.reset_temporal_state()
        s0, _, _ = event_snapshot(n, 8, 6, 0, seed=2)
        s1, _, _ = event_snapshot(n, 8, 6, gap, seed=3)
        with torch.no_grad():
            outputs, _ = model.forward_sequence([s0, s1])
        outs[gap] = outputs[1]["spatial"]
    assert not torch.allclose(outs[1], outs[40], atol=1e-5)


def test_memory_persists_and_is_causal():
    model, n = make_model()
    model.eval()
    snaps = [event_snapshot(n, 8, 6, k, seed=10 + k)[0] for k in range(4)]
    with torch.no_grad():
        model.reset_temporal_state()
        out_a, _ = model.forward_sequence(snaps)
        perturbed = [dict(snaps[0]), *snaps[1:]]
        perturbed[0]["x"] = snaps[0]["x"] + 5.0
        model.reset_temporal_state()
        out_b, _ = model.forward_sequence(perturbed)
    assert not torch.allclose(out_a[-1]["spatial"], out_b[-1]["spatial"], atol=1e-6)


def test_untouched_nodes_keep_state():
    model, n = make_model(closure_hops=0)
    model.eval()
    s0, _, _ = event_snapshot(n, 8, 12, 0, seed=4)
    with torch.no_grad():
        model.reset_temporal_state()
        _, st0 = model.forward_sequence([s0])
        s1 = {
            "x": torch.randn(n, 8),
            "edge_index": torch.tensor([[0], [1]]),
            "active_nodes": torch.tensor([0, 1]),
            "timestamp": torch.tensor(1.0),
        }
        _, st1 = model.forward_sequence([s1], initial_state=st0)
    untouched = torch.tensor([k for k in range(n) if k > 1])
    assert torch.allclose(st1.memory[untouched], st0.memory[untouched], atol=1e-7)
    assert not torch.allclose(st1.memory[:2], st0.memory[:2], atol=1e-6)


def test_sheaf_decoded_once_per_event():
    model, n = make_model(layers=3)
    calls = {"n": 0}
    orig = model.sheaf_learner.forward

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.sheaf_learner.forward = counting
    model.eval()
    snap, _, _ = event_snapshot(n, 8, 6, 0, seed=5)
    with torch.no_grad():
        model.forward_sequence([snap])
    assert calls["n"] == 1


def test_current_only_control_isolated_to_sheaf():
    for mode, expect_same in [("current_only", True), ("history", False)]:
        torch.manual_seed(6)
        model, n = make_model(sheaf_conditioning=mode)
        model.eval()
        final, _, _ = event_snapshot(n, 8, 6, 2, seed=7)
        maps = {}
        for tag, seed in [("a", 20), ("b", 21)]:
            model.reset_temporal_state()
            hist, _, _ = event_snapshot(n, 8, 6, 0, seed=seed)
            with torch.no_grad():
                _, st = model.forward_sequence([hist])
                model.forward_sequence([final], initial_state=st)
            maps[tag] = model.sheaf_learner.L.clone()
        same = torch.allclose(maps["a"], maps["b"], atol=1e-6)
        assert same == expect_same, f"{mode}: same={same}"


def test_gradients_flow_end_to_end():
    model, n = make_model()
    snap0, _, _ = event_snapshot(n, 8, 8, 0, seed=9)
    snap, src, dst = event_snapshot(n, 8, 8, 1, seed=8)
    outputs, _ = model.forward_sequence([snap0, snap])
    pos = model.score_event_pairs(outputs[1], src, dst, edge_type=torch.zeros_like(src))
    loss = -pos.mean()
    loss.backward()
    for name in ["ssm.A", "ssm.B_selector.weight", "sheaf_learner.linear1.weight",
                 "P_z.weight", "W1.weight", "W2.weight", "log_tau",
                 "event_source_proj.weight", "feedback_proj.weight"]:
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad at {name}"


def test_original_modules_removed():
    model, _ = make_model()
    names = {n for n, _ in model.named_parameters()}
    for banned in ["nodewise_learner", "sheaf_learners.", "lin_left_weights",
                   "lin_right_weights", "epsilons", "lin2"]:
        assert not any(banned in n for n in names), f"stale module params: {banned}"
