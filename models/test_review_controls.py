"""Deciding tests for the review controls (server handoff 2026-09-22).

* new arms (GRU, diagonal SSM, node-frame, attention) run forward/backward
  with the shared head/REC and change the output when history changes;
* the fast core-off bypass produces outputs identical to the reference
  core-off path (no_memory + layers 0 without the bypass);
* node-frame transport telescopes to identity around cycles while the
  incidence-specific decoder need not;
* per-node clocks: closure-only nodes get different gaps under node_update
  and node_interaction; the global clock ignores node inactivity; vector
  gaps are sliced consistently through SSM chunks;
* isolated negative-sampling RNG yields identical negatives across arms;
* the query-validity audit counts NaN / +inf negatives (not only positives)
  and its three metrics agree when everything is finite.
"""
import numpy as np
import torch

from exp import temporal_benchmark_utils as bu
from models.faithful_event_model import FaithfulEventTemporalSheafDiffusion
from models.temporal_sheaf_ssm import SelectiveZOHSSM
from models.test_faithful_event_model import event_snapshot, make_model


def _stream(n, n_snap, seed, n_events=6, gap=1.0):
    out = []
    for k in range(n_snap):
        snap, src, dst = event_snapshot(n, 8, n_events, ts=k * gap, seed=seed * 100 + k)
        out.append((snap, src, dst))
    return out


def _run(model, stream, state=None):
    model.eval()
    model.reset_temporal_state()
    with torch.no_grad():
        for snap, _, _ in stream:
            outs, state = model.forward_sequence([snap], initial_state=state)
    return outs[0], state


def test_new_arms_run_and_depend_on_history():
    for kw in ({"backbone": "gru"}, {"backbone": "diag_ssm"}, {"spatial": "node_frame"}, {"spatial": "attention"}):
        torch.manual_seed(0)
        model, n = make_model(recurrency_decoder=True, **kw)
        a = _stream(n, 3, seed=1)
        b = _stream(n, 3, seed=2)
        out_a, _ = _run(model, a)
        out_b, _ = _run(model, b)
        assert torch.isfinite(out_a["spatial"]).all()
        assert not torch.allclose(out_a["spatial"], out_b["spatial"]), kw
        # backward through the shared head
        model.train()
        model.reset_temporal_state()
        outs, state = model.forward_sequence([a[0][0]])
        src, dst = a[1][1], a[1][2]
        pos = model.score_event_pairs(outs[0], src, dst, edge_type=torch.zeros_like(src))
        loss = torch.nn.functional.softplus(-pos).mean()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.ssm.parameters()), kw


def test_fast_core_off_matches_reference_outputs():
    torch.manual_seed(0)
    ref, n = make_model(no_memory=True, layers=0, recurrency_decoder=True)
    fast, _ = make_model(no_memory=True, layers=0, recurrency_decoder=True, fast_core_off=True)
    fast.load_state_dict(ref.state_dict())
    stream = _stream(n, 4, seed=3)
    out_ref, _ = _run(ref, stream)
    out_fast, _ = _run(fast, stream)
    assert torch.allclose(out_ref["spatial"], out_fast["spatial"], atol=1e-6)
    src, dst = stream[-1][1], stream[-1][2]
    with torch.no_grad():
        s_ref = ref.score_event_pairs(out_ref, src, dst, edge_type=torch.zeros_like(src), query_time=torch.full(src.shape, 3.5))
        s_fast = fast.score_event_pairs(out_fast, src, dst, edge_type=torch.zeros_like(src), query_time=torch.full(src.shape, 3.5))
    assert torch.allclose(s_ref, s_fast, atol=1e-6)


def _cycle_product(model, h, tri):
    """Transport product T_01 T_12 T_20 with T_uv = R_{e<-u}^T R_{e<-v}."""
    d = model.final_d
    und = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    builder = model._make_local_builder(und, 3)
    if model.spatial_variant == "node_frame":
        maps = torch.tanh(model.frame_decoder(h)).index_select(0, und[0])
    else:
        maps = model.sheaf_learner(h, und)
    R = builder.orth_transform(maps).reshape(-1, d, d)
    idx = {(int(und[0, e]), int(und[1, e])): e for e in range(und.size(1))}
    T = lambda u, v: R[idx[(u, v)]].t() @ R[idx[(v, u)]]
    return T(0, 1) @ T(1, 2) @ T(2, 0)


def test_node_frame_transport_is_flat_and_incidence_maps_need_not_be():
    torch.manual_seed(0)
    frame, _ = make_model(spatial="node_frame")
    sheaf, _ = make_model()
    h = torch.randn(3, frame.d_h)
    P_frame = _cycle_product(frame, h, None)
    assert torch.allclose(P_frame, torch.eye(frame.final_d), atol=1e-5)
    # the incidence decoder generically produces a non-identity cycle product
    # (unsaturated tanh: saturated +/-1 parameters give angles that are multiples of pi/2)
    with torch.no_grad():
        sheaf.sheaf_learner.linear1.weight.normal_(0, 0.3)
    P_sheaf = _cycle_product(sheaf, h, None)
    assert not torch.allclose(P_sheaf, torch.eye(sheaf.final_d), atol=1e-3)


def test_attention_gate_at_one_equals_identity_control():
    torch.manual_seed(0)
    att, n = make_model(spatial="attention")
    ident, _ = make_model(spatial="identity")
    ident.load_state_dict({k: v for k, v in att.state_dict().items() if k in ident.state_dict()}, strict=False)
    with torch.no_grad():
        att.gate_decoder.weight.zero_(); att.gate_decoder.bias.fill_(50.0)   # sigmoid(100) == 1
    stream = _stream(n, 3, seed=5)
    out_a, _ = _run(att, stream)
    out_i, _ = _run(ident, stream)
    assert torch.allclose(out_a["spatial"], out_i["spatial"], atol=1e-5)


def test_node_clocks_differ_for_inactive_and_closure_nodes():
    torch.manual_seed(0)
    n = 12
    seen = {}
    for clock in ("global", "node_update", "node_interaction"):
        model, _ = make_model(num_nodes=n, clock=clock)
        model.eval(); model.reset_temporal_state()
        # snapshot 0 touches nodes 0-1 (t=0); snapshot 1 touches 5-6 (t=10); snapshot 2 touches 0,5 (t=13)
        def snap(pairs, t):
            src = torch.tensor([p[0] for p in pairs]); dst = torch.tensor([p[1] for p in pairs])
            return {"x": torch.ones(n, 8), "edge_index": torch.stack([src, dst]),
                    "active_nodes": torch.unique(torch.cat([src, dst])), "timestamp": torch.tensor(float(t))}
        gaps = []
        orig = model.ssm.step_size

        def spy(q, delta_t):
            gaps.append(delta_t.detach().clone())
            return orig(q, delta_t)
        model.ssm.step_size = spy
        with torch.no_grad():
            state = None
            for s in (snap([(0, 1)], 0), snap([(5, 6)], 10), snap([(0, 5)], 13)):
                _, state = model.forward_sequence([s], initial_state=state)
        seen[clock] = gaps[-1]
    assert seen["global"].dim() == 0 and float(seen["global"]) == 3.0
    # node_update: node 0 last updated at t=0 -> 13; node 5 at t=10 -> 3; closure nodes updated at their last step
    g_upd = seen["node_update"]; g_int = seen["node_interaction"]
    assert g_upd.dim() == 1 and g_int.dim() == 1
    assert float(g_upd.max()) == 13.0 and float(g_upd.min()) >= 0.0
    # node 1 is a closure node of node 0 in snapshot 2 (ring context graph): last interaction at t=0 -> 13,
    # last update also at t=0 -> 13; node 11 (closure of 0) never interacted -> gap 0 under node_interaction
    assert float(g_int.max()) == 13.0
    assert not torch.equal(g_upd, g_int)
    assert (g_int == 0).sum() >= 1   # first-observation nodes get gap 0, not a sentinel


def test_vector_gap_is_sliced_through_ssm_chunks():
    torch.manual_seed(0)
    ssm = SelectiveZOHSSM(8, 5)
    ssm.chunk_size = 7
    ssm.eval()
    h = torch.randn(20, 8); q = torch.randn(20, 5); gap = torch.rand(20) * 5
    with torch.no_grad():
        chunked = ssm(h, q, gap)
        ssm.chunk_size = 10_000
        whole = ssm(h, q, gap)
    assert torch.allclose(chunked, whole, atol=1e-6)


def test_isolated_negative_rng_is_architecture_independent():
    pos = torch.arange(10)
    bu.install_negative_rng(43, 0, torch.device("cpu"))
    torch.manual_seed(1); a = bu._sample_uniform_negative_destinations(pos, num_nodes=50, negatives_per_positive=4)
    bu.install_negative_rng(43, 0, torch.device("cpu"))
    torch.manual_seed(999); _ = torch.randn(1000)   # different global-RNG consumption
    b = bu._sample_uniform_negative_destinations(pos, num_nodes=50, negatives_per_positive=4)
    bu.NEGATIVE_RNG["generator"] = None
    assert torch.equal(a, b)
    assert not (a == pos.view(-1, 1)).any()


def test_query_audit_counts_invalid_negatives_and_parity():
    class Snap:
        src = torch.tensor([1, 2, 3]); dst = torch.tensor([4, 5, 6]); timestamp = torch.tensor(7.0)
        edge_timestamps = torch.tensor([7.0, 7.0, 7.0]); edge_types = None; edge_ids = torch.tensor([10, 11, 12])
    pos = torch.tensor([1.0, 2.0, 3.0])
    neg = torch.tensor([[0.5, float("nan")], [2.5, float("inf")], [0.0, 1.0]])
    mask = np.ones((3, 2), dtype=bool)
    audit = bu.QueryValidityAudit()
    gp, gn = bu._guard_nonfinite_scores(pos.clone(), neg.clone())
    audit.record_batch("test", 0, Snap(), pos, neg, mask, gp, gn)
    r = audit.summary_rows()[0]
    assert r["neg_nan"] == 1 and r["neg_posinf"] == 1 and r["queries_with_invalid_neg"] == 2
    assert r["queries_affected"] == 2 and r["pos_nonfinite"] == 0
    # conservative diagnostic zeroes the two affected queries; guarded keeps them
    assert abs(r["mrr_conservative"] - (1.0 / 3)) < 1e-9
    assert r["mrr_guarded"] > r["mrr_conservative"]
    assert not r["parity_guarded_vs_conservative"]
    assert len(audit.failures) == 2 and audit.failures[0]["edge_id"] == 10
    # all-finite batch: the three metrics agree
    audit2 = bu.QueryValidityAudit()
    neg2 = torch.tensor([[0.5, 0.2], [2.5, 1.0], [0.0, 1.0]])
    audit2.record_batch("test", 0, Snap(), pos, neg2, mask, pos, neg2)
    r2 = audit2.summary_rows()[0]
    assert r2["parity_raw_vs_guarded"] and r2["parity_guarded_vs_conservative"] and r2["queries_affected"] == 0
