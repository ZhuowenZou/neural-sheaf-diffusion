"""Deciding tests for the hidden assumptions found in the protocol/model audit.

Each test pins one assumption that previously went unverified (see
results/analytic/audit/AUDIT.md):
  * TGB's node-label cursor is continuous across splits (per-split reset
    paired tgbn-trade val 2010 with the 1987 labels).
  * TGB's link evaluator ranks a NaN positive FIRST; the harness must not
    report MRR 1.0 for an invalid prediction.
  * The physical-gap selector must not saturate at dt_cap for typical gaps.
  * The identity-sheaf ablation must use the builder's normalization.
  * The dense-vocabulary recurrency lookup must be deterministic on CUDA.
"""
import os

import numpy as np
import pytest
import torch

from exp.temporal_benchmark_utils import _guard_nonfinite_scores, seek_label_cursor


class _Snap:
    def __init__(self, ts):
        self.timestamp = torch.tensor(ts)


class _FakeLabelDataset:
    """Mimics tgb.nodeproppred.dataset.NodePropPredDataset's cursor."""

    def __init__(self, label_ts):
        self.label_ts = np.asarray(label_ts)
        self.label_ts_idx = 0

    def reset_label_time(self):
        self.label_ts_idx = 0

    def get_node_label(self, cur_t):
        if self.label_ts_idx >= len(self.label_ts):
            return None
        ts = self.label_ts[self.label_ts_idx]
        if cur_t >= ts:
            self.label_ts_idx += 1
            return (np.full(1, ts), np.zeros(1, dtype=int), np.zeros((1, 3)))
        return None


def _fired(ds, snaps):
    out = []
    for s in snaps:
        lab = ds.get_node_label(int(s.timestamp))
        out.append(None if lab is None else int(lab[0][0]))
    return out


def test_label_cursor_matches_continuous_cursor():
    labels = list(range(1987, 2017))
    train = [_Snap(t) for t in range(1986, 2010)]
    val = [_Snap(t) for t in range(2010, 2014)]
    test = [_Snap(t) for t in range(2014, 2017)]
    # reference: TGB example (reset once, run train->val->test)
    ref = _FakeLabelDataset(labels)
    ref.reset_label_time()
    _fired(ref, train)
    ref_val = _fired(ref, val)
    ref_test = _fired(ref, test)
    # harness: per-split seek
    ds = _FakeLabelDataset(labels)
    seek_label_cursor(ds, val, after_ts=int(train[-1].timestamp))
    assert _fired(ds, val) == ref_val == [2010, 2011, 2012, 2013]
    seek_label_cursor(ds, test, after_ts=int(val[-1].timestamp))
    assert _fired(ds, test) == ref_test == [2014, 2015, 2016]
    # the old behaviour (plain reset) is the bug being pinned
    ds.reset_label_time()
    assert _fired(ds, val) == [1987, 1988, 1989, 1990]


def test_label_cursor_fires_label_between_splits():
    # a label timestamp strictly between the last train edge and the first val
    # edge must fire at the first val snapshot (as in TGB's continuous cursor)
    labels = [10, 20, 30, 40]
    train = [_Snap(5), _Snap(15)]
    val = [_Snap(25), _Snap(35)]
    ref = _FakeLabelDataset(labels)
    _fired(ref, train)
    ref_val = _fired(ref, val)
    ds = _FakeLabelDataset(labels)
    seek_label_cursor(ds, val, after_ts=15)
    assert _fired(ds, val) == ref_val == [20, 30]


@pytest.mark.skipif(not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "datasets")), reason="needs TGB datasets")
def test_tgbn_trade_split_pairing_with_real_dataset():
    pytest.importorskip("tgb")
    from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    ds = PyGNodePropPredDataset(name="tgbn-trade", root=root)
    data = ds.get_TemporalData()
    years = lambda mask: [_Snap(int(t)) for t in np.unique(data.t[mask].numpy())]
    train, val, test = years(ds.train_mask), years(ds.val_mask), years(ds.test_mask)
    seek_label_cursor(ds, val, after_ts=int(train[-1].timestamp))
    assert _fired(ds, val) == [2010, 2011, 2012, 2013]
    seek_label_cursor(ds, test, after_ts=int(val[-1].timestamp))
    assert _fired(ds, test) == [2014, 2015, 2016]


def test_nonfinite_positive_is_ranked_last_not_first():
    pytest.importorskip("tgb")
    from tgb.linkproppred.evaluate import Evaluator

    ev = Evaluator(name="tgbl-wiki")
    pos = torch.tensor([float("nan"), 0.5])
    neg = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.9]])
    raw = ev.eval({"y_pred_pos": pos, "y_pred_neg": neg, "eval_metric": ["mrr"]})
    assert float(np.mean(raw["mrr"])) == pytest.approx((1.0 + 0.5) / 2)  # TGB ranks NaN first
    gpos, gneg = _guard_nonfinite_scores(pos, neg)
    guarded = ev.eval({"y_pred_pos": gpos, "y_pred_neg": gneg, "eval_metric": ["mrr"]})
    assert float(np.mean(guarded["mrr"])) == pytest.approx((0.25 + 0.5) / 2)  # -inf -> last


def test_step_size_does_not_saturate_after_delta_scaling():
    from models.temporal_sheaf_ssm import SelectiveZOHSSM

    ssm = SelectiveZOHSSM(d_state=16, d_input=8)
    q = torch.randn(32, 8)
    gap = torch.tensor(600.0, dtype=torch.float64)  # tgbl-wiki window in seconds
    dt_raw = ssm.step_size(q, gap)
    assert torch.allclose(dt_raw, torch.full_like(dt_raw, ssm.dt_cap))  # the pre-audit saturation
    ssm.set_delta_scale(600.0)  # median inter-snapshot gap -> typical gap is 1 unit
    dt = ssm.step_size(q, gap)
    assert bool((dt < ssm.dt_cap).all())
    dt.sum().backward()
    assert ssm.dt_time_weight.grad is not None and float(ssm.dt_time_weight.grad.abs()) > 0
    assert ssm.dt_proj.bias.grad is not None and float(ssm.dt_proj.bias.grad.abs()) > 0
    with pytest.raises(ValueError):
        ssm.set_delta_scale(0.0)


def _tiny_model(**extra):
    from models.test_faithful_event_model import make_model

    overrides = dict(recurrency_decoder=True, recurrency_untyped=True, recurrency_symmetric=True)
    overrides.update(extra)
    model, _ = make_model(num_nodes=12, input_dim=4, num_relations=3, **overrides)
    return model


def test_identity_ablation_uses_builder_normalization():
    torch.manual_seed(0)
    model = _tiny_model(sheaf_identity=True)
    d = model.final_d
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    n = 4
    builder = model._make_local_builder(ei, n)
    sig = torch.randn(n, model.sheaf_learner_in_dim if hasattr(model, "sheaf_learner_in_dim") else model.hidden_dim * 1)
    try:
        maps = model.sheaf_learner(sig, ei)
    except Exception:
        maps = torch.zeros(ei.size(1), d * (d - 1) // 2)
    L, _ = builder(torch.zeros_like(maps))
    dense = torch.sparse_coo_tensor(L[0], L[1], (n * d, n * d)).to_dense()
    # augmented normalization: D~ = D + I ; L = D~^-1/2 (D - A) D~^-1/2, kron I_d
    A = torch.zeros(n, n)
    A[ei[0], ei[1]] = 1.0
    deg = A.sum(1)
    dn = 1.0 / torch.sqrt(deg + 1.0)
    Lg = torch.diag(deg) - A
    Lg = dn.view(-1, 1) * Lg * dn.view(1, -1)
    expected = torch.kron(Lg, torch.eye(d))
    assert torch.allclose(dense, expected, atol=1e-5), (dense - expected).abs().max()
    # and it is NOT the unit-diagonal I - D^-1/2 A D^-1/2 the old closed form produced
    old = torch.kron(torch.eye(n) - (1 / torch.sqrt(deg)).view(-1, 1) * A * (1 / torch.sqrt(deg)).view(1, -1), torch.eye(d))
    assert not torch.allclose(dense, old, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA determinism test")
def test_dense_recurrency_recency_is_deterministic_on_cuda():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = _tiny_model().to(dev)
    model.reset_temporal_state()
    src = torch.tensor([1] * 6, device=dev)
    dst = torch.tensor([7] * 6, device=dev)
    rel = torch.tensor([0] * 6, device=dev)
    # one committed event and five earlier-in-snapshot repeats of the same key
    for name, store in model._rec_stores:
        store.stash(model._rec_channel_keys(name, src[:1], rel[:1], dst[:1]), torch.tensor([1.0], device=dev, dtype=torch.float64), build_intra=False)
        store.advance()
    ts = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=dev, dtype=torch.float64)
    model._prev_timestamp = torch.tensor(7.0, dtype=torch.float64)
    for name, store in model._rec_stores:
        store.stash(model._rec_channel_keys(name, src[1:], rel[1:], dst[1:]), ts, build_intra=True)
    q_time = torch.tensor([6.5], device=dev, dtype=torch.float64)
    outs = []
    for _ in range(20):
        outs.append(model._recurrency_bonus_dense_rows(src[:1], rel[:1], model.graph_size, dev, torch.float32, query_time=q_time).detach().clone())
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


# ---------------------------------------------------------------------------
# Leak-free predict-then-update protocol (audit rows 21 / 32)
# ---------------------------------------------------------------------------
def _link_spec():
    from exp.temporal_benchmark_utils import DatasetSpec

    return DatasetSpec(requested_name="tkgl-smallpedia", loader_name="tkgl-smallpedia", task_family="tkg",
                       dataset_module="tgb.linkproppred.dataset_pyg", evaluator_module="tgb.linkproppred.evaluate",
                       metric_name="mrr", train_metric_supported=False)


def _stream(num_nodes=20, n_snap=6, events=5, seed=0, x=None):
    from exp.temporal_utils import TemporalSnapshot

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, 4, generator=torch.Generator().manual_seed(123)) if x is None else x
    snaps = []
    for k in range(n_snap):
        s = torch.randint(0, num_nodes, (events,), generator=g)
        d = torch.randint(0, num_nodes, (events,), generator=g)
        keep = s != d
        s, d = s[keep], d[keep]
        r = torch.randint(0, 3, (s.numel(),), generator=g)
        snaps.append(TemporalSnapshot(
            x=x, edge_index=torch.stack([torch.cat([s, d]), torch.cat([d, s])]), src=s, dst=d,
            active_nodes=torch.unique(torch.cat([s, d])), timestamp=torch.tensor(float(k + 1)),
            edge_timestamps=torch.full((s.numel(),), float(k + 1)), edge_types=r))
    return snaps


def test_predict_from_previous_training_updates_backbone(monkeypatch):
    from exp import temporal_benchmark_utils as bu

    monkeypatch.setenv("TSD_SCORE_FROM_PREVIOUS_STATE", "1")
    torch.manual_seed(0)
    model = _tiny_model()
    snaps = _stream(num_nodes=12)
    backbone = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("ssm.") or "sheaf_learner" in n}
    assert backbone, "no backbone parameters found"
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    model.reset_temporal_state()
    loss = bu.run_epoch(_link_spec(), model, opt, snaps, dataset=None, bptt_steps=1)
    assert np.isfinite(loss)
    changed = [n for n, p in model.named_parameters() if n in backbone and not torch.equal(p.detach(), backbone[n])]
    assert changed, "backbone received no gradient under the leak-free protocol (detached-history bug)"


def test_predict_from_previous_eval_is_independent_of_current_events(monkeypatch):
    from types import SimpleNamespace
    from exp import temporal_benchmark_utils as bu

    monkeypatch.setenv("TSD_SCORE_FROM_PREVIOUS_STATE", "1")
    captured = []

    class Ev:
        def __init__(self, name): pass
        def eval(self, payload):
            captured.append(payload["y_pred_pos"].detach().clone())
            return {"hits@10": np.ones(len(payload["y_pred_pos"])), "mrr": np.ones(len(payload["y_pred_pos"]))}

    class NS:
        def query_batch(self, src, dst, ts, edge_type=None, split_mode=None):
            return [np.asarray([1, 2, 3], dtype=np.int64) for _ in range(src.numel())]

    class DS:
        eval_metric = "mrr"
        negative_sampler = NS()
        def load_test_ns(self): pass

    monkeypatch.setattr(bu, "pytest_importorskip", lambda m: SimpleNamespace(Evaluator=Ev))
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    hist = _stream(num_nodes=12, n_snap=3, seed=1)
    last_a = _stream(num_nodes=12, n_snap=1, seed=2)[0]
    # same first query (src, dst, rel, t); the OTHER events of the snapshot differ
    last_b = _stream(num_nodes=12, n_snap=1, seed=3)[0]
    last_b.src[0], last_b.dst[0], last_b.edge_types[0] = last_a.src[0], last_a.dst[0], last_a.edge_types[0]
    last_b.timestamp = last_a.timestamp
    last_b.edge_timestamps = torch.full_like(last_b.edge_timestamps, float(last_a.timestamp))

    def run(last):
        captured.clear()
        model.reset_temporal_state()
        state = bu.advance_context(model, hist)
        bu.evaluate_model_streaming(_link_spec(), DS(), [last], model, initial_state=state, split_mode="test")
        return captured[-1][0].item()

    assert run(last_a) == pytest.approx(run(last_b), abs=1e-6)


def test_embeddings_in_head_reach_the_scoring_head():
    torch.manual_seed(0)
    snaps = _stream(num_nodes=12, n_snap=2, seed=5)
    base = _tiny_model(learnable_node_features=True)
    head = _tiny_model(learnable_node_features=True, embeddings_in_head=True)
    head.load_state_dict(base.state_dict())
    for m in (base, head):
        m.eval(); m.reset_temporal_state()
    ob, _ = base.forward_sequence([snaps[0]], initial_state=None)
    oh, _ = head.forward_sequence([snaps[0]], initial_state=None)
    assert torch.equal(ob[0]["x"], snaps[0].x)                       # default: raw features
    assert not torch.equal(oh[0]["x"], snaps[0].x)                   # flag: x + embedding
    assert torch.allclose(oh[0]["x"], snaps[0].x + head.node_feature_embedding.weight)
    assert torch.equal(ob[0]["spatial"], oh[0]["spatial"])          # the temporal core is untouched
    s = torch.tensor([1, 2]); d = torch.tensor([5, 7]); r = torch.tensor([0, 1])
    sb = base.score_event_pairs(ob[0], s, d, edge_type=r); sh = head.score_event_pairs(oh[0], s, d, edge_type=r)
    assert not torch.allclose(sb, sh)


# ---------------------------------------------------------------------------
# Node-property label pairing (audit row 34): drain every label a snapshot passes,
# edge-based split seek, and the runners' own import paths
# ---------------------------------------------------------------------------
class _LabelDS:
    """Fake TGB node-property dataset: one label per timestamp in `label_ts`."""
    eval_metric = "ndcg"

    def __init__(self, label_ts, num_nodes=6, num_classes=4):
        self.label_ts = np.asarray(label_ts)
        self.label_ts_idx = 0
        self.n, self.c = num_nodes, num_classes

    def reset_label_time(self):
        self.label_ts_idx = 0

    def get_node_label(self, cur_t):
        if self.label_ts_idx >= len(self.label_ts):
            return None
        ts = int(self.label_ts[self.label_ts_idx])
        if cur_t >= ts:
            self.label_ts_idx += 1
            g = torch.Generator().manual_seed(ts)
            return (torch.full((self.n,), ts), torch.arange(self.n), torch.rand(self.n, self.c, generator=g))
        return None


def _snap_with_edges(t0, t1, num_nodes=6):
    from exp.temporal_utils import TemporalSnapshot

    e = torch.arange(t0, t1 + 1).float()
    s = torch.zeros(len(e), dtype=torch.long); d = torch.ones(len(e), dtype=torch.long)
    return TemporalSnapshot(x=torch.zeros(num_nodes, 3), edge_index=torch.stack([s, d]), src=s, dst=d,
                            active_nodes=torch.tensor([0, 1]), timestamp=torch.tensor(float(t1)), edge_timestamps=e)


class _ConstModel(torch.nn.Module):
    def __init__(self, num_nodes=6, num_classes=4):
        super().__init__(); self.logits = torch.randn(num_nodes, num_classes)

    def forward_sequence(self, snapshots, initial_state=None):
        return [self.logits for _ in snapshots], initial_state


def test_drain_consumes_every_label_a_snapshot_passes():
    from exp.temporal_benchmark_utils import drain_snapshot_labels, node_label_batches, seek_label_cursor

    ds = _LabelDS(range(1, 31))                              # daily labels 1..30
    weeks = [_snap_with_edges(1, 7), _snap_with_edges(8, 14), _snap_with_edges(15, 21)]
    seek_label_cursor(ds, weeks)
    assert [ts for ts, _, _ in drain_snapshot_labels(ds, weeks[0])] == list(range(1, 8))
    assert [ts for ts, _, _ in drain_snapshot_labels(ds, weeks[1])] == list(range(8, 15))
    # node_label_batches: 7 pairs per weekly snapshot, and chunked calls with seek=False continue the cursor
    assert len(node_label_batches(ds, weeks)) == 21
    seek_label_cursor(ds, weeks)
    assert len(node_label_batches(ds, weeks[:1], seek=False)) == 7
    assert len(node_label_batches(ds, weeks[1:], seek=False)) == 14


def test_seek_skips_labels_in_the_gap_left_by_a_capped_training_split():
    from exp.temporal_benchmark_utils import drain_snapshot_labels, seek_label_cursor

    ds = _LabelDS(range(1, 31))
    val = [_snap_with_edges(20, 25), _snap_with_edges(26, 27)]     # training ended at t=5; 6..19 never streamed
    seek_label_cursor(ds, val)
    assert [ts for ts, _, _ in drain_snapshot_labels(ds, val[0])] == [20, 21, 22, 23, 24, 25]
    assert [ts for ts, _, _ in drain_snapshot_labels(ds, val[1])] == [26, 27]


def test_runner_evaluation_paths_score_every_label_per_timestamp():
    pytest.importorskip("tgb")
    import exp.faithful_temporal_studies as fts
    import exp.temporal_mamba_studies as tms
    from exp import temporal_benchmark_utils as bu

    for evaluate in (tms._evaluate_model_streaming, fts._evaluate_model_streaming):
        ds = _LabelDS(range(1, 31))
        val = [_snap_with_edges(20, 25), _snap_with_edges(26, 27)]
        metric, loss, _ = evaluate("tgbn-trade", ds, val, _ConstModel(), initial_state=None)
        assert bu.evaluate_node_property_streaming.last_label_timestamps == list(range(20, 28))
        assert np.isfinite(metric) and np.isfinite(loss)


@pytest.mark.skipif(not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "datasets")), reason="needs TGB datasets")
def test_tgbn_trade_windows_pair_every_label_year():
    pytest.importorskip("tgb")
    from exp.faithful_temporal_studies import make_faithful_trade_config, prepare_temporal_experiment_context
    from exp.temporal_benchmark_utils import drain_snapshot_labels, seek_label_cursor

    os.environ.setdefault("TGB_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets")))
    cfg = make_faithful_trade_config()

    def fired(ds, snaps):
        seek_label_cursor(ds, snaps)
        return [[ts for ts, _, _ in drain_snapshot_labels(ds, s)] for s in snaps]

    for cap, win, exp_val, exp_test in [
        (2048, cfg["time_window"], [[2010, 2011, 2012], [2013]], [[2014, 2015, 2016]]),
        (None, 1, [[2010], [2011], [2012], [2013]], [[2014], [2015], [2016]]),
    ]:
        ctx = prepare_temporal_experiment_context("tgbn-trade", split_caps={"train": cap, "val": None, "test": None},
                                                  device=torch.device("cpu"), preload_time_windows=[win])
        b = ctx.get_snapshot_bundle(win)
        assert fired(ctx.dataset, b["val_snapshots"]) == exp_val
        assert fired(ctx.dataset, b["test_snapshots"]) == exp_test


def test_no_memory_output_is_independent_of_history_and_current_only_is_not():
    """`--no-memory` must make the scored state independent of the past stream;
    `sheaf_conditioning=current_only` alone must NOT (it only changes what the
    restriction maps are decoded from; the memory still enters Z0)."""
    torch.manual_seed(0)
    hist_a = _stream(num_nodes=12, n_snap=3, seed=11)
    hist_b = _stream(num_nodes=12, n_snap=3, seed=12)
    last = _stream(num_nodes=12, n_snap=1, seed=13)[0]

    def final_spatial(model, hist):
        model.eval(); model.reset_temporal_state()
        with torch.no_grad():
            state = None
            for s in hist:
                _, state = model.forward_sequence([s], initial_state=state)
            outs, _ = model.forward_sequence([last], initial_state=state)
        return outs[0]["spatial"].clone()

    nomem = _tiny_model(no_memory=True)
    assert nomem.sheaf_conditioning == "current_only"
    assert torch.allclose(final_spatial(nomem, hist_a), final_spatial(nomem, hist_b), atol=1e-6)
    cur = _tiny_model(sheaf_conditioning="current_only")
    assert not torch.allclose(final_spatial(cur, hist_a), final_spatial(cur, hist_b), atol=1e-6)
