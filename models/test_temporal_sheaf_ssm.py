"""Faithfulness validation for FaithfulTemporalSheafDiffusion (paper Sec. 3.2).

Each test checks one clause of the discrete event-time algorithm:
  - HiPPO-LegS initialization of the shared generator A (Eq. 5 / Init section)
  - exact zero-order-hold discretization (Eqs. 16-17)
  - the recurrence is affine in h_{k-1} (Eq. 18), not a nonlinear pseudo-seq
  - physical event gaps Delta_k condition the selector (Eq. 11 / 15)
  - input-dependent selectivity of Delta_{u,k} and B_{u,k} (Eq. 15)
  - causality and true long-range memory over the event index
  - the sheaf is decoded once per event time and held fixed (Sec. 3.2)
  - lagged feedback reads the PREVIOUS graph G_{k-1} (Eq. 13)
  - active-node gating (Eq. 11: u in A_k)
  - the model can learn a task requiring event-history memory
"""

import math

import pytest
import torch

from models.temporal_sheaf_ssm import (
    FaithfulTemporalSheafDiffusion,
    SelectiveZOHSSM,
    TemporalSheafState,
    hippo_legs_b,
    hippo_legs_matrix,
)

torch.manual_seed(0)


def make_args(num_nodes=12, input_dim=8, output_dim=12, **overrides):
    args = {
        "graph_size": num_nodes,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "d": 2,
        "hidden_channels": 8,
        "layers": 2,
        "dropout": 0.0,
        "input_dropout": 0.0,
        "use_act": True,
        "sheaf_act": "tanh",
        "orth": "householder",
        "add_lp": False,
        "add_hp": False,
        "closure_hops": 1,
        "stateful_temporal": False,
        "temporal_d_model": 16,
        "feedback_dim": 8,
    }
    args.update(overrides)
    return args


def ring_edge_index(n):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def make_model(**overrides):
    n = overrides.pop("num_nodes", 12)
    args = make_args(num_nodes=n, **overrides)
    return FaithfulTemporalSheafDiffusion(ring_edge_index(n), args), n


def snapshot(x, edge_index, ts, active=None):
    return {
        "x": x,
        "edge_index": edge_index,
        "active_nodes": active,
        "timestamp": torch.tensor(float(ts)),
    }


# ---------------------------------------------------------------------------
# HiPPO-LegS initialization
# ---------------------------------------------------------------------------

def test_hippo_legs_matrix_matches_formula():
    n = 6
    A = hippo_legs_matrix(n)
    for i in range(n):
        for j in range(n):
            if i > j:
                expected = -math.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                expected = -(i + 1)
            else:
                expected = 0.0
            assert A[i, j].item() == pytest.approx(expected, rel=1e-5)
    b = hippo_legs_b(n)
    for i in range(n):
        assert b[i].item() == pytest.approx(math.sqrt(2 * i + 1), rel=1e-6)


def test_model_A_initialized_to_hippo_legs():
    model, _ = make_model()
    assert torch.allclose(model.ssm.A.detach(), hippo_legs_matrix(model.d_h), atol=1e-6)


# ---------------------------------------------------------------------------
# Exact ZOH discretization (Eqs. 16-17)
# ---------------------------------------------------------------------------

def test_zoh_matches_continuous_solution_scalar():
    """For d_h = 1, ZOH must reproduce the exact solution of
    h' = a h + b q with constant input:  h(dt) = e^{a dt} h0 + (e^{a dt}-1)/a * b q."""
    ssm = SelectiveZOHSSM(d_state=1, d_input=1, dt_cap=10.0).double()
    with torch.no_grad():
        ssm.A.fill_(-0.7)
        ssm.B_selector.weight.zero_()
        ssm.B_selector.bias.fill_(2.0)
        ssm.dt_proj.weight.zero_()
        ssm.dt_proj.bias.fill_(0.5)  # softplus(0.5 + time term) = dt
        ssm.dt_time_weight.zero_()  # freeze physical term for closed-form check

    h0 = torch.tensor([[1.5]], dtype=torch.float64)
    q = torch.tensor([[3.0]], dtype=torch.float64)
    h1 = ssm(h0, q, torch.tensor(0.0, dtype=torch.float64))

    dt = math.log(1 + math.exp(0.5))
    a, b = -0.7, 2.0
    expected = math.exp(a * dt) * 1.5 + (math.exp(a * dt) - 1.0) / a * (b * 3.0)
    assert h1.item() == pytest.approx(expected, rel=1e-10)


def test_zoh_matches_phi1_series_matrix():
    """Batched solve-based B_bar q must equal dt*phi1(dt A) B q computed from
    the Taylor series of phi1(M) = sum_{k>=0} M^k / (k+1)!."""
    torch.manual_seed(1)
    d_state, d_input, n = 5, 3, 4
    ssm = SelectiveZOHSSM(d_state, d_input).double()
    q = torch.randn(n, d_input, dtype=torch.float64)
    h0 = torch.zeros(n, d_state, dtype=torch.float64)
    delta = torch.tensor(2.0, dtype=torch.float64)

    h1 = ssm(h0, q, delta)

    dt = ssm.step_size(q, delta)
    A = ssm.A
    B = ssm.B_selector(q).view(n, d_state, d_input)
    Bq = torch.bmm(B, q.unsqueeze(-1)).squeeze(-1)
    expected = torch.zeros_like(h1)
    for i in range(n):
        M = dt[i] * A
        phi1 = torch.zeros_like(M)
        term = torch.eye(d_state, dtype=torch.float64)
        for k in range(30):
            phi1 = phi1 + term / float(math.factorial(k + 1))
            term = term @ M
        expected[i] = dt[i] * (phi1 @ Bq[i])
    assert torch.allclose(h1, expected, atol=1e-8)


def test_recurrence_is_affine_in_previous_state():
    """Eq. 18 is h_k = A_bar h_{k-1} + B_bar q: affine in h_{k-1} with a
    state-independent Jacobian (rules out MambaBlock-style nonlinear mixing)."""
    ssm = SelectiveZOHSSM(d_state=6, d_input=4).double()
    q = torch.randn(3, 4, dtype=torch.float64)
    delta = torch.tensor(1.0, dtype=torch.float64)

    h_a = torch.randn(3, 6, dtype=torch.float64)
    h_b = torch.randn(3, 6, dtype=torch.float64)
    lam = 0.3
    out_mix = ssm(lam * h_a + (1 - lam) * h_b, q, delta)
    mix_out = lam * ssm(h_a, q, delta) + (1 - lam) * ssm(h_b, q, delta)
    assert torch.allclose(out_mix, mix_out, atol=1e-10)


# ---------------------------------------------------------------------------
# Physical Delta_k conditioning and selectivity (Eq. 15)
# ---------------------------------------------------------------------------

def test_physical_delta_changes_update():
    ssm = SelectiveZOHSSM(d_state=8, d_input=4)
    h0 = torch.randn(5, 8)
    q = torch.randn(5, 4)
    out_small = ssm(h0, q, torch.tensor(1.0))
    out_large = ssm(h0, q, torch.tensor(50.0))
    assert not torch.allclose(out_small, out_large, atol=1e-5)


def test_model_output_depends_on_event_gap():
    """Same two snapshots, different physical gap => different second output."""
    model, n = make_model()
    model.eval()
    x = torch.randn(n, 8)
    ei = ring_edge_index(n)
    with torch.no_grad():
        out_close, _ = model.forward_sequence([snapshot(x, ei, 0), snapshot(x, ei, 1)])
        out_far, _ = model.forward_sequence([snapshot(x, ei, 0), snapshot(x, ei, 100)])
    assert torch.allclose(out_close[0], out_far[0], atol=1e-6)  # first step identical
    assert not torch.allclose(out_close[1], out_far[1], atol=1e-5)


def test_step_size_is_input_selective():
    ssm = SelectiveZOHSSM(d_state=4, d_input=3)
    with torch.no_grad():
        ssm.dt_proj.weight.normal_(std=1.0)
    q = torch.randn(6, 3)
    dt = ssm.step_size(q, torch.tensor(1.0))
    assert (dt > 0).all()
    assert dt.std() > 1e-6  # different inputs -> different step sizes


# ---------------------------------------------------------------------------
# Causality and true event-history memory
# ---------------------------------------------------------------------------

def test_causality_future_does_not_affect_past():
    model, n = make_model()
    model.eval()
    ei = ring_edge_index(n)
    xs = [torch.randn(n, 8) for _ in range(3)]
    with torch.no_grad():
        out_a, _ = model.forward_sequence([snapshot(xs[i], ei, i) for i in range(3)])
        xs_perturbed = [xs[0], xs[1], xs[2] + 10.0]
        out_b, _ = model.forward_sequence([snapshot(x, ei, i) for i, x in enumerate(xs_perturbed)])
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)
    assert not torch.allclose(out_a[2], out_b[2], atol=1e-4)


def test_memory_propagates_across_many_steps():
    """The step-0 input must influence the step-K output through the explicit
    recurrence (this is exactly what the length-2 pseudo-sequence lost)."""
    model, n = make_model()
    model.eval()
    ei = ring_edge_index(n)
    K = 6
    base = [torch.randn(n, 8) for _ in range(K)]
    with torch.no_grad():
        out_a, _ = model.forward_sequence([snapshot(base[i], ei, i) for i in range(K)])
        perturbed = [base[0] + 5.0] + base[1:]
        out_b, _ = model.forward_sequence([snapshot(x, ei, i) for i, x in enumerate(perturbed)])
    diff = (out_a[-1] - out_b[-1]).abs().max().item()
    assert diff > 1e-6, f"step-0 perturbation vanished by step {K}: diff={diff}"


# ---------------------------------------------------------------------------
# Sheaf decoded once per event and held fixed (Sec. 3.2)
# ---------------------------------------------------------------------------

def test_sheaf_decoded_once_per_event():
    model, n = make_model(layers=4)
    calls = {"n": 0}
    orig = model.sheaf_learner.forward

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.sheaf_learner.forward = counting
    model.eval()
    with torch.no_grad():
        model.step(torch.randn(n, 8), ring_edge_index(n), timestamp=torch.tensor(0.0))
    assert calls["n"] == 1, f"sheaf decoded {calls['n']} times for a 4-layer interval"


# ---------------------------------------------------------------------------
# Lagged feedback uses the previous graph G_{k-1} (Eq. 13)
# ---------------------------------------------------------------------------

def test_lagged_feedback_uses_previous_graph():
    model, n = make_model()
    model.eval()
    z_prev = torch.randn(n, model.hidden_dim)
    ei_prev = ring_edge_index(n)
    state_ring = TemporalSheafState(memory=None, spatial=z_prev, prev_edge_index=ei_prev)
    state_empty = TemporalSheafState(memory=None, spatial=z_prev, prev_edge_index=torch.zeros(2, 0, dtype=torch.long))
    with torch.no_grad():
        fb_ring = model._lagged_feedback(state_ring, z_prev.device)
        fb_empty = model._lagged_feedback(state_empty, z_prev.device)
    assert not torch.allclose(fb_ring, fb_empty, atol=1e-6)
    # No previous interval => zero feedback (paper: k = 0 case).
    fb_none = model._lagged_feedback(None, z_prev.device)
    assert torch.all(fb_none == 0)


def test_state_detach_preserves_graph_and_timestamp():
    model, n = make_model()
    x = torch.randn(n, 8)
    _, state = model.step(x, ring_edge_index(n), timestamp=torch.tensor(7.0), return_state=True)
    d = state.detach()
    assert d.prev_edge_index is not None and torch.equal(d.prev_edge_index, state.prev_edge_index)
    assert d.prev_timestamp is not None and d.prev_timestamp.item() == 7.0
    assert not d.memory.requires_grad and not d.spatial.requires_grad


# ---------------------------------------------------------------------------
# Active-node gating (Eq. 11)
# ---------------------------------------------------------------------------

def test_inactive_node_memory_unchanged():
    model, n = make_model(closure_hops=0)
    model.eval()
    x = torch.randn(n, 8)
    ei = ring_edge_index(n)
    with torch.no_grad():
        _, s1 = model.step(x, ei, timestamp=torch.tensor(0.0), return_state=True)
        active = torch.tensor([0, 1])
        _, s2 = model.step(
            x, ei, active_nodes=active, timestamp=torch.tensor(1.0), state=s1, return_state=True
        )
    inactive = torch.tensor([k for k in range(n) if k not in (0, 1)])
    assert torch.allclose(s2.memory[inactive], s1.memory[inactive], atol=1e-7)
    assert not torch.allclose(s2.memory[:2], s1.memory[:2], atol=1e-6)


# ---------------------------------------------------------------------------
# Learning: task solvable only through event-history memory
# ---------------------------------------------------------------------------

def test_learns_delayed_recall_task():
    """At step k the label of node u is the identity of the feature pattern
    shown at step k-2 (uniform across steps otherwise). A model without
    cross-event memory cannot beat chance; the recurrence should learn it."""
    torch.manual_seed(3)
    n, dim, n_classes = 10, 8, 4
    model, _ = make_model(num_nodes=n, input_dim=dim, output_dim=n_classes, layers=1)
    ei = ring_edge_index(n)
    patterns = torch.randn(n_classes, dim) * 3.0

    def make_sequence(K=6):
        cls = torch.randint(0, n_classes, (K,))
        snaps, labels = [], []
        for k in range(K):
            x = patterns[cls[k]].unsqueeze(0).expand(n, dim) + 0.01 * torch.randn(n, dim)
            snaps.append(snapshot(x, ei, k))
            labels.append(cls[k - 2] if k >= 2 else None)
        return snaps, labels

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for it in range(150):
        snaps, labels = make_sequence()
        outputs, _ = model.forward_sequence(snaps)
        loss_terms = []
        for out, lab in zip(outputs, labels):
            if lab is None:
                continue
            target = torch.full((n,), int(lab), dtype=torch.long)
            loss_terms.append(torch.nn.functional.nll_loss(out, target))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    chance = math.log(n_classes)
    assert late < early - 0.1, f"no learning: early={early:.3f} late={late:.3f}"
    assert late < chance * 0.75, f"failed delayed recall: late={late:.3f} vs chance={chance:.3f}"


def test_gradients_flow_to_all_components():
    model, n = make_model()
    ei = ring_edge_index(n)
    outputs, _ = model.forward_sequence(
        [snapshot(torch.randn(n, 8), ei, k) for k in range(3)]
    )
    loss = torch.stack([o.sum() for o in outputs]).sum()
    loss.backward()
    for name in ["ssm.A", "ssm.dt_proj.bias", "ssm.B_selector.weight", "sheaf_learner.linear1.weight",
                 "P_z.weight", "W1.weight", "W2.weight", "log_tau", "lin2.weight", "feedback_proj.weight"]:
        param = dict(model.named_parameters())[name]
        assert param.grad is not None and param.grad.abs().sum() > 0, f"no gradient at {name}"


# ---------------------------------------------------------------------------
# Reviewer control: no-history-to-sheaf (sheaf_conditioning='current_only')
# ---------------------------------------------------------------------------

def test_current_only_sheaf_ignores_history():
    """Under current_only, the decoded restriction maps at step k must be
    identical regardless of history; under history they must differ."""
    for mode, expect_same in [("current_only", True), ("history", False)]:
        torch.manual_seed(5)
        model, n = make_model(sheaf_conditioning=mode)
        model.eval()
        ei = ring_edge_index(n)
        x_hist_a = torch.randn(n, 8)
        x_hist_b = torch.randn(n, 8) + 3.0
        x_final = torch.randn(n, 8)
        maps = {}
        for name, hist in [("a", x_hist_a), ("b", x_hist_b)]:
            with torch.no_grad():
                _, s = model.step(hist, ei, timestamp=torch.tensor(0.0), return_state=True)
                model.step(x_final, ei, timestamp=torch.tensor(1.0), state=s)
            maps[name] = model.sheaf_learner.L.clone()
        same = torch.allclose(maps["a"], maps["b"], atol=1e-6)
        assert same == expect_same, f"mode={mode}: maps same={same}, expected {expect_same}"


def test_current_only_memory_still_drives_features():
    """The control removes history only from the sheaf: logits must still
    depend on history through P_z / lagged feedback."""
    torch.manual_seed(5)
    model, n = make_model(sheaf_conditioning="current_only")
    model.eval()
    ei = ring_edge_index(n)
    x_final = torch.randn(n, 8)
    outs = []
    for shift in [0.0, 3.0]:
        with torch.no_grad():
            _, s = model.step(torch.randn(n, 8) + shift, ei, timestamp=torch.tensor(0.0), return_state=True)
            out = model.step(x_final, ei, timestamp=torch.tensor(1.0), state=s)
        outs.append(out)
    assert not torch.allclose(outs[0], outs[1], atol=1e-5)
