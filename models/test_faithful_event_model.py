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
    for name in ["ssm.A_lower", "ssm.A_neg_diag", "ssm.B_selector.weight", "sheaf_learner.linear1.weight",
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


def test_node_type_embedding_changes_output():
    types = torch.arange(20) % 4
    model, n = make_model(node_types=types)
    model.eval()
    snap, _, _ = event_snapshot(n, 8, 6, 0, seed=30)
    with torch.no_grad():
        out_a, _ = model.forward_sequence([snap])
        model.reset_temporal_state()
        model.node_type_embedding.weight.data.mul_(5.0)
        out_b, _ = model.forward_sequence([snap])
    assert not torch.allclose(out_a[0]["spatial"], out_b[0]["spatial"], atol=1e-6)


def test_relation_in_input_reaches_memory():
    model, n = make_model(relation_in_input=True, num_relations=5)
    assert model.ssm.d_input == model.d_q + model.relation_input_dim
    model.eval()
    snap, src, dst = event_snapshot(n, 8, 6, 0, seed=31)
    snap["edge_types"] = torch.randint(0, 5, (src.numel(),))
    with torch.no_grad():
        _, st_a = model.forward_sequence([snap])
        model.reset_temporal_state()
        snap2 = dict(snap)
        snap2["edge_types"] = (snap["edge_types"] + 1) % 5
        _, st_b = model.forward_sequence([snap2])
    assert not torch.allclose(st_a.memory, st_b.memory, atol=1e-7)


def test_recurrency_decoder_causal_and_effective():
    model, n = make_model(recurrency_decoder=True, num_relations=5)
    with torch.no_grad():
        model.recurrency_mlp[-1].weight.fill_(1.0)
        model.recurrency_mlp[-1].bias.fill_(1.0)
    model.eval()
    src = torch.tensor([1, 2]); dst = torch.tensor([3, 4])
    et = torch.tensor([0, 1])
    snap = {"x": torch.randn(n, 8), "edge_index": torch.stack([src, dst]),
            "active_nodes": torch.unique(torch.cat([src, dst])),
            "timestamp": torch.tensor(0.0), "edge_types": et}
    with torch.no_grad():
        out0, st = model.forward_sequence([snap])
        # scoring the SAME snapshot: its events are pending, not committed
        b0 = model._recurrency_bonus(src, et, dst)
        assert torch.allclose(b0, torch.zeros(2)), b0
        snap1 = dict(snap); snap1["timestamp"] = torch.tensor(2.0)
        out1, _ = model.forward_sequence([snap1], initial_state=st)
        # now the first snapshot is committed: seen triples get a bonus
        b1 = model._recurrency_bonus(src, et, dst)
        assert (b1 > 0).all(), b1
        # unseen triple gets none
        b_unseen = model._recurrency_bonus(torch.tensor([5]), torch.tensor([2]), torch.tensor([6]))
        assert torch.allclose(b_unseen, torch.zeros(1))
        # wrong relation for a seen pair gets none
        b_wrongrel = model._recurrency_bonus(src[:1], torch.tensor([3]), dst[:1])
        assert torch.allclose(b_wrongrel, torch.zeros(1))
    model.reset_temporal_state()
    assert model._rec_keys is None and model._rec_pending is None


def test_dense_vocab_scoring_matches_sparse_path():
    torch.manual_seed(3)
    model, n = make_model(recurrency_decoder=True, num_relations=5, closure_hops=0)
    with torch.no_grad():
        model.recurrency_mlp[-1].weight.normal_(); model.recurrency_mlp[-1].bias.fill_(0.5)
    model.eval()
    s0, src0, dst0 = event_snapshot(n, 8, 10, 0, seed=40)
    s0["edge_types"] = torch.randint(0, 5, (src0.numel(),))
    s1, src1, dst1 = event_snapshot(n, 8, 6, 3, seed=41)
    s1["edge_types"] = torch.randint(0, 5, (src1.numel(),))
    with torch.no_grad():
        _, st = model.forward_sequence([s0])
        outs, _ = model.forward_sequence([s1], initial_state=st)
        out = outs[0]
        # full-vocabulary candidates (dense path) vs per-candidate path + bonus
        cands = torch.arange(n).unsqueeze(0).expand(src1.numel(), n).clone()
        dense = model.score_event_candidates(out, src1, cands, edge_type=s1["edge_types"])
        sparse = (super(type(model), model).score_event_candidates(out, src1, cands, edge_type=s1["edge_types"])
                  + model._recurrency_bonus(src1, s1["edge_types"], cands))
    assert dense.shape == sparse.shape
    assert torch.allclose(dense, sparse, atol=1e-5), (dense - sparse).abs().max()
    # bonus actually fired for a seen triple
    assert (model._recurrency_bonus_dense_rows(src0[:1], s0["edge_types"][:1], n, dense.device, dense.dtype) != 0).any()


def test_intra_snapshot_recurrency_is_event_exact_and_causal():
    model, n = make_model(recurrency_decoder=True, num_relations=3, closure_hops=0)
    with torch.no_grad():
        model.recurrency_mlp[-1].weight.fill_(1.0); model.recurrency_mlp[-1].bias.fill_(1.0)
    model.eval()
    # three identical triples in ONE snapshot at t = 0, 5, 5 (+ one distinct)
    src = torch.tensor([1, 1, 1, 4]); dst = torch.tensor([2, 2, 2, 5]); et = torch.tensor([0, 0, 0, 1])
    ts = torch.tensor([0.0, 5.0, 5.0, 7.0])
    snap = {"x": torch.randn(n, 8), "edge_index": torch.stack([src, dst]), "active_nodes": torch.unique(torch.cat([src, dst])),
            "timestamp": torch.tensor(7.0), "edge_types": et, "edge_timestamps": ts}
    with torch.no_grad():
        outs, st = model.forward_sequence([snap])
        b = model._recurrency_bonus(src, et, dst, query_time=ts)
        assert b[0] == 0, "first occurrence must not see itself or later events"
        assert b[1] > 0 and b[2] > 0, "later occurrences see the t=0 event"
        assert torch.isclose(b[1], b[2]), "ties at t=5 are mutually invisible (strict)"
        assert b[3] == 0
        # without query_time the old snapshot-level behaviour holds (nothing committed yet)
        assert (model._recurrency_bonus(src, et, dst) == 0).all()
        # dense path agrees with sparse path under query_time
        cands = torch.arange(n).unsqueeze(0).expand(4, n).clone()
        dense = model.score_event_candidates(outs[0], src, cands, edge_type=et, query_time=ts)
        sparse = super(type(model), model).score_event_candidates(outs[0], src, cands, edge_type=et) \
            + model._recurrency_bonus(src, et, cands, query_time=ts)
        assert torch.allclose(dense, sparse, atol=1e-5)
        # after commit the cache holds count 3 with last_t = 5 for the repeated triple
        nxt = dict(snap); nxt["timestamp"] = torch.tensor(9.0); nxt["edge_timestamps"] = ts + 9
        model.forward_sequence([nxt], initial_state=st)
        key = model._rec_pack(src[:1], et[:1], dst[:1])
        pos = torch.searchsorted(model._rec_keys, key)
        assert model._rec_count[pos].item() == 3 and model._rec_last_t[pos].item() == 5.0


def test_memory_dim_independent_of_spatial_dim():
    # d_h (64) != hidden_dim (16): guards against the parent's spatial-sized
    # initial memory silently replacing the faithful d_h-sized one.
    model, n = make_model(temporal_d_model=64, recurrency_decoder=True, num_relations=3)
    assert model.d_h == 64 and model.hidden_dim == 16
    s0, _, _ = event_snapshot(n, 8, 8, 0, seed=1)
    s0["edge_types"] = torch.zeros(8, dtype=torch.long); s0["edge_timestamps"] = torch.arange(8.0)
    s1, _, _ = event_snapshot(n, 8, 8, 10, seed=2)
    s1["edge_types"] = torch.zeros(8, dtype=torch.long); s1["edge_timestamps"] = torch.arange(8.0) + 10
    _, st = model.forward_sequence([s0])
    _, st = model.forward_sequence([s1], initial_state=st)
    assert st.memory.shape == (n, 64) and st.spatial.shape == (n, 16)


def test_recurrency_history_warm_start_survives_reset():
    model, n = make_model(recurrency_decoder=True, num_relations=3)
    with torch.no_grad():
        model.recurrency_mlp[-1].weight.fill_(1.0); model.recurrency_mlp[-1].bias.fill_(1.0)
    model.set_recurrency_history(torch.tensor([1]), torch.tensor([2]), torch.tensor([0]), torch.tensor([-5.0]))
    model.reset_temporal_state()
    b = model._recurrency_bonus(torch.tensor([1]), torch.tensor([0]), torch.tensor([2]))
    assert b.item() > 0, "warm history must be visible without any forward"
    assert model._recurrency_bonus(torch.tensor([1]), torch.tensor([1]), torch.tensor([2])).item() == 0
    model.reset_temporal_state()
    assert model._rec_keys.numel() == 1


def test_untyped_channel_sees_repeats_across_event_types():
    model, n = make_model(recurrency_decoder=True, recurrency_untyped=True, num_relations=4)
    assert model.recurrency_mlp[0].in_features == 6
    with torch.no_grad():
        model.recurrency_mlp[-1].weight.fill_(1.0); model.recurrency_mlp[-1].bias.fill_(0.0)
    model.eval()
    src = torch.tensor([1]); dst = torch.tensor([2])
    snap = {"x": torch.randn(n, 8), "edge_index": torch.stack([src, dst]), "active_nodes": torch.tensor([1, 2]),
            "timestamp": torch.tensor(0.0), "edge_types": torch.tensor([0]), "edge_timestamps": torch.tensor([0.0])}
    with torch.no_grad():
        _, st = model.forward_sequence([snap])
        nxt = dict(snap); nxt["timestamp"] = torch.tensor(3.0); nxt["edge_timestamps"] = torch.tensor([3.0])
        model.forward_sequence([nxt], initial_state=st)   # commits the (1, r0, 2) event
        same_type = model._recurrency_bonus(src, torch.tensor([0]), dst)
        other_type = model._recurrency_bonus(src, torch.tensor([3]), dst)
        unseen_pair = model._recurrency_bonus(torch.tensor([5]), torch.tensor([0]), torch.tensor([6]))
    assert same_type.item() > other_type.item() > 0, (same_type, other_type)
    assert unseen_pair.item() == 0
    # typed-only model gives the other-type query nothing
    m2, _ = make_model(recurrency_decoder=True, num_relations=4)
    with torch.no_grad():
        m2.recurrency_mlp[-1].weight.fill_(1.0)
    m2.eval()
    with torch.no_grad():
        _, st2 = m2.forward_sequence([snap]); m2.forward_sequence([nxt], initial_state=st2)
        assert m2._recurrency_bonus(src, torch.tensor([3]), dst).item() == 0


def test_symmetric_channel_sees_reverse_direction_repeats():
    model, n = make_model(recurrency_decoder=True, recurrency_untyped=True, recurrency_symmetric=True, num_relations=2)
    assert model.recurrency_mlp[0].in_features == 9
    with torch.no_grad():  # deterministic, monotone gate: bonus = sum of features
        model.recurrency_mlp[0].weight.fill_(1.0); model.recurrency_mlp[0].bias.fill_(0.0)
        model.recurrency_mlp[-1].weight.fill_(1.0); model.recurrency_mlp[-1].bias.fill_(0.0)
    model.eval()
    snap = {"x": torch.randn(n, 8), "edge_index": torch.tensor([[1], [2]]), "active_nodes": torch.tensor([1, 2]),
            "timestamp": torch.tensor(0.0), "edge_types": torch.tensor([0]), "edge_timestamps": torch.tensor([0.0])}
    with torch.no_grad():
        _, st = model.forward_sequence([snap])
        nxt = dict(snap); nxt["timestamp"] = torch.tensor(2.0); nxt["edge_timestamps"] = torch.tensor([2.0])
        model.forward_sequence([nxt], initial_state=st)
        fwd = model._recurrency_bonus(torch.tensor([1]), torch.tensor([0]), torch.tensor([2]))
        rev = model._recurrency_bonus(torch.tensor([2]), torch.tensor([0]), torch.tensor([1]))
        # dense path agrees for the reverse query
        out = model.forward_sequence([dict(nxt, timestamp=torch.tensor(3.0), edge_timestamps=torch.tensor([3.0]))], initial_state=st)[0][0]
        cands = torch.arange(n).unsqueeze(0)
        dense = model.score_event_candidates(out, torch.tensor([2]), cands, edge_type=torch.tensor([0]))
        sparse = super(type(model), model).score_event_candidates(out, torch.tensor([2]), cands, edge_type=torch.tensor([0])) \
            + model._recurrency_bonus(torch.tensor([2]), torch.tensor([0]), cands)
    assert fwd.item() > rev.item() > 0, (fwd, rev)   # reverse pair seen only through the symmetric channel
    assert torch.allclose(dense, sparse, atol=1e-5)


def test_ablation_flags_isolate_ingredients():
    # no_delta_t: outputs invariant to the physical gap; identity sheaf: runs and differs from learned sheaf
    torch.manual_seed(5)
    m_nodt, n = make_model(no_delta_t=True)
    assert not m_nodt.ssm.dt_time_weight.requires_grad and m_nodt.ssm.dt_time_weight.item() == 0.0
    m_nodt.eval()
    outs = {}
    for gap in [1, 40]:
        m_nodt.reset_temporal_state()
        s0, _, _ = event_snapshot(n, 8, 6, 0, seed=2)
        s1, _, _ = event_snapshot(n, 8, 6, gap, seed=3)
        with torch.no_grad():
            o, _ = m_nodt.forward_sequence([s0, s1])
        outs[gap] = o[1]["spatial"]
    assert torch.allclose(outs[1], outs[40], atol=1e-6), "no_delta_t model must ignore the gap"
    torch.manual_seed(5)
    m_id, _ = make_model(sheaf_identity=True)
    torch.manual_seed(5)
    m_full, _ = make_model()
    m_id.eval(); m_full.eval()
    snap, src, dst = event_snapshot(n, 8, 8, 0, seed=7)
    with torch.no_grad():
        o_id, st_id = m_id.forward_sequence([snap]); o_full, st_full = m_full.forward_sequence([snap])
    assert not torch.allclose(o_id[0]["spatial"], o_full[0]["spatial"], atol=1e-6)
    # residual helper runs for both and returns one value per pair
    with torch.no_grad():
        r_full = m_full.sheaf_residual(st_full.memory, o_full[0]["spatial"], src, dst)
        r_id = m_id.sheaf_residual(st_id.memory, o_id[0]["spatial"], src, dst)
    assert r_full.shape == (src.numel(),) and r_id.shape == (src.numel(),) and torch.isfinite(r_full).all()

