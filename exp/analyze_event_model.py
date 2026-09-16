"""Per-event mechanistic analysis of a trained faithful event model (Arm B).

Loads a checkpoint (from run_event_benchmark --save-checkpoint), replays
train -> val -> test under the official protocol, and records for every TEST
event: reciprocal rank, whether the (s,r,o) triple / (s,o) pair recurred,
the physical gap since the subject's last event, the relation frequency, and
the transport residual of the positive vs. the mean over 64 sampled
negatives. Optional --time-scale multiplies all timestamps at inference (a
Delta-aware core degrades under rescaling; an order-only one is invariant).

Usage: python -m exp.analyze_event_model --dataset tkgl-smallpedia --checkpoint
       results/event_bench/sp_untyped_sym_s43/best.pt --out results/analytic/sp_full
       [same model flags as the training run] [--time-scale 0.25]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exp import temporal_benchmark_utils as bu
from exp.run_event_benchmark import _make_model, _context_edge_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="faithful"); ap.add_argument("--train-edges-cap", type=int, default=None)
    ap.add_argument("--time-window", type=float, default=None); ap.add_argument("--context-edges", type=int, default=50000)
    ap.add_argument("--val-edges-cap", type=int, default=None); ap.add_argument("--test-edges-cap", type=int, default=None)
    ap.add_argument("--time-scale", type=float, default=1.0)
    for f in ["feedback-dim", "d", "layers", "hidden-channels", "temporal-d-model", "closure-hops", "train-negatives-per-pos",
              "candidate-chunk-size", "max-score-elements"]:
        ap.add_argument(f"--{f}", type=int, default={"feedback-dim": 16, "d": 2, "layers": 2, "hidden-channels": 8, "temporal-d-model": 64,
                                                      "closure-hops": 1, "train-negatives-per-pos": 32, "candidate-chunk-size": 1024,
                                                      "max-score-elements": 4_000_000}[f])
    for f in ["no-memory-readout", "learn-node-emb", "node-type-emb", "relation-in-input", "recurrency-decoder", "recurrency-untyped",
              "recurrency-symmetric", "sheaf-identity", "no-delta-t", "no-dst-range", "no-memory", "emb-in-head"]:
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--sheaf-conditioning", default="history"); ap.add_argument("--train-loss", default="softplus")
    ap.add_argument("--no-residual", action="store_true", help="skip the transport-residual diagnostic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    spec, dataset, td = bu.load_temporal_data(args.dataset)
    import dataclasses

    def scaled(snap):
        """Model-side copy with all timestamps multiplied by --time-scale; the
        original snapshot keeps feeding the official negative sampler."""
        if args.time_scale == 1.0:
            return snap
        ts = snap.timestamp.double() * args.time_scale
        et = None if snap.edge_timestamps is None else snap.edge_timestamps.double() * args.time_scale
        if dataclasses.is_dataclass(snap):
            return dataclasses.replace(snap, timestamp=ts, edge_timestamps=et)
        return snap._replace(timestamp=ts, edge_timestamps=et)
    num_nodes = max(int(td.src.max()), int(td.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    feats = bu.make_node_features(dataset, num_nodes).to(device)
    train_ids, val_ids, test_ids, _ = bu.split_edge_ids(dataset, td)
    if args.train_edges_cap is not None:
        train_ids = train_ids[-args.train_edges_cap:]
    ctx = _context_edge_index(td, train_ids, args.context_edges)
    val_ids = val_ids if args.val_edges_cap is None else val_ids[: args.val_edges_cap]
    test_ids = test_ids if args.test_edges_cap is None else test_ids[: args.test_edges_cap]
    snaps = lambda ids: bu.build_snapshots(td, ids, feats, time_window=args.time_window)
    train_s, val_s, test_s = snaps(train_ids), snaps(val_ids), snaps(test_ids)
    dataset.load_val_ns(); dataset.load_val_ns = lambda: None
    dataset.load_test_ns(); dataset.load_test_ns = lambda: None

    node_types = None
    if args.node_type_emb:
        nt = getattr(dataset, "node_type", None); node_types = None if nt is None else torch.as_tensor(nt)[:num_nodes]
    model = _make_model(ctx, feats, num_nodes, device, args, num_relations, node_types=node_types)
    if not args.no_dst_range:
        lo, hi = int(td.dst.min()), int(td.dst.max())
        if (lo, hi) != (0, num_nodes - 1): model.event_destination_range = (lo, hi)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)); model.eval()

    # history bookkeeping for recurrence / gap / relation-frequency features
    R = max(num_relations, 1)
    src_all, dst_all, t_all = td.src.numpy(), td.dst.numpy(), td.t.numpy().astype(np.float64)
    rel_all = td.edge_type.numpy() if getattr(td, "edge_type", None) is not None else np.zeros_like(src_all)
    hist_ids = np.concatenate([train_ids.numpy(), val_ids.numpy()])
    seen_triple = set((src_all[hist_ids].astype(np.int64) * R + rel_all[hist_ids]) * num_nodes + dst_all[hist_ids])
    seen_pair = set(src_all[hist_ids].astype(np.int64) * num_nodes + dst_all[hist_ids])
    rel_count = np.bincount(rel_all[hist_ids], minlength=R)
    last_seen = {}
    for i in hist_ids: last_seen[int(src_all[i])] = t_all[i]

    from tgb.linkproppred.evaluate import Evaluator
    evaluator = Evaluator(name=spec.loader_name)
    model.reset_temporal_state()
    state = bu.advance_context(model, [scaled(s) for s in train_s])
    state = bu.advance_context(model, [scaled(s) for s in val_s], initial_state=state)
    # Cross-check: score the identical test stream through the harness's own
    # streaming evaluator from the same post-val state (deep-copied model so
    # the recurrency caches are not consumed twice).
    import copy
    if args.time_scale == 1.0:
        m2 = copy.deepcopy(model)
        for (_, s2), (_, s1) in zip(m2._rec_stores, model._rec_stores):
            s2.keys, s2.count, s2.last_t = (None if s1.keys is None else s1.keys.clone()), (None if s1.count is None else s1.count.clone()), (None if s1.last_t is None else s1.last_t.clone())
            s2.pending = None if s1.pending is None else (s1.pending[0].clone(), s1.pending[1].clone()); s2.intra = None if s1.intra is None else tuple(x.clone() if torch.is_tensor(x) else x for x in s1.intra)
        m2._prev_timestamp = model._prev_timestamp; m2._prev_event_edge_index = model._prev_event_edge_index
        h_mrr, _, _ = bu.evaluate_model_streaming(spec, dataset, test_s, m2, initial_state=state, split_mode="test")
        print(f"harness cross-check test MRR: {h_mrr:.4f}", flush=True)
        del m2
    rows = []; t0 = time.perf_counter(); off_mrr, off_n = 0.0, 0
    prev_out = None
    with torch.no_grad():
        for snap in test_s:
            msnap = scaled(snap)
            deferred = os.environ.get("TSD_SCORE_FROM_PREVIOUS_STATE") == "1"
            if deferred:
                # leak-free protocol (audit rows 21/32): score snapshot k with the
                # previous forward's output (history < k), forward k afterwards
                if prev_out is None:
                    out = {"spatial": state.spatial, "x": msnap.x.to(device) if getattr(msnap, "x", None) is not None else None, "node_signal": None}
                else:
                    out = dict(prev_out, node_signal=None)
            else:
                outs, state = model.forward_sequence([msnap], initial_state=state)
                out = outs[0]
            src, dst = snap.src.to(device), snap.dst.to(device)
            et = snap.edge_types.to(device) if snap.edge_types is not None else None
            qt = msnap.edge_timestamps.to(device) if getattr(model, "supports_query_time", False) and msnap.edge_timestamps is not None else None
            pos = model.score_event_pairs(out, src, dst, edge_type=et, **({"query_time": qt} if qt is not None else {}))
            neg = dataset.negative_sampler.query_batch(snap.src, snap.dst, snap.edge_timestamps, edge_type=snap.edge_types, split_mode="test")
            padded, mask = bu._pad_negative_samples(neg)
            neg_dst = torch.as_tensor(padded, dtype=torch.long, device=device)
            ns = model.score_event_candidates(out, src, neg_dst, edge_type=et, **({"query_time": qt} if qt is not None else {}))
            ns = ns.masked_fill(~torch.as_tensor(mask, device=device), float("-inf"))
            rank = 1 + (ns > pos.unsqueeze(-1)).sum(-1) + 0.5 * (ns == pos.unsqueeze(-1)).sum(-1)
            m_off = evaluator.eval({"y_pred_pos": pos.detach().cpu().numpy(), "y_pred_neg": ns.detach().cpu().numpy(), "eval_metric": ["mrr"]})["mrr"]
            off_mrr += float(np.mean(m_off)) * pos.numel(); off_n += pos.numel()
            # transport residual: positive vs 64 sampled negatives per event
            if args.no_residual:
                r_pos = torch.full((src.numel(),), float("nan"), device=device); r_neg = r_pos
            else:
                k = min(64, neg_dst.size(1))
                samp = neg_dst[:, torch.randperm(neg_dst.size(1), device=device)[:k]]
                r_pos = model.sheaf_residual(state.memory, out["spatial"], src, dst)
                r_neg = model.sheaf_residual(state.memory, out["spatial"], src.repeat_interleave(k), samp.reshape(-1)).view(-1, k).mean(-1)
            s_np, d_np, t_np = snap.src.numpy(), snap.dst.numpy(), snap.edge_timestamps.numpy().astype(np.float64)
            r_np = snap.edge_types.numpy() if snap.edge_types is not None else np.zeros_like(s_np)
            for i in range(len(s_np)):
                key_t = (int(s_np[i]) * R + int(r_np[i])) * num_nodes + int(d_np[i]); key_p = int(s_np[i]) * num_nodes + int(d_np[i])
                gap = t_np[i] - last_seen.get(int(s_np[i]), np.nan)
                rows.append(dict(rr=1.0 / float(rank[i]), rec_triple=key_t in seen_triple, rec_pair=key_p in seen_pair,
                                 gap=gap, rel=int(r_np[i]), rel_freq=int(rel_count[int(r_np[i])]),
                                 res_pos=float(r_pos[i]), res_neg=float(r_neg[i])))
            for i in range(len(s_np)):  # commit this snapshot's facts for later events (causal, snapshot-granular)
                seen_triple.add((int(s_np[i]) * R + int(r_np[i])) * num_nodes + int(d_np[i])); seen_pair.add(int(s_np[i]) * num_nodes + int(d_np[i]))
                last_seen[int(s_np[i])] = t_np[i]
            if deferred:
                outs, state = model.forward_sequence([msnap], initial_state=state); prev_out = outs[0]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "events.csv"), index=False)
    summary = {"n_events": len(df), "mrr": df.rr.mean(), "mrr_official_evaluator": off_mrr / max(off_n, 1),
               "mrr_harness_crosscheck": float(h_mrr) if args.time_scale == 1.0 else None, "mrr_recurrent_triple": df[df.rec_triple].rr.mean(), "mrr_novel_triple": df[~df.rec_triple].rr.mean(),
               "mrr_novel_pair": df[~df.rec_pair].rr.mean(), "frac_novel_pair": float((~df.rec_pair).mean()),
               "residual_pos_mean": df.res_pos.mean(), "residual_neg_mean": df.res_neg.mean(),
               "residual_pos_lt_neg_frac": float((df.res_pos < df.res_neg).mean()), "time_scale": args.time_scale, "sec": time.perf_counter() - t0}
    g = df.copy(); g["gap_bucket"] = pd.qcut(g.gap.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    summary["mrr_by_gap_quartile"] = g.groupby("gap_bucket", observed=True).rr.mean().to_dict()
    g["rel_bucket"] = pd.qcut(g.rel_freq.rank(method="first"), 4, labels=["rare", "q2", "q3", "frequent"])
    summary["mrr_by_rel_freq_quartile"] = g.groupby("rel_bucket", observed=True).rr.mean().to_dict()
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
