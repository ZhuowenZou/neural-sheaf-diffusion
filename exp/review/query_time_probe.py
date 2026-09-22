"""Fixed-state future-query diagnostic (server handoff 2026-09-22, section 2).

Replays a trained checkpoint through the retained training stream, clones the
state, and scores the SAME (source, relation, candidate) tuples of the first
validation batches at their original query time and at shifted future times
WITHOUT ingesting anything.  Reports, separately for the neural score, the REC
bonus and the total, the fraction of scores that change and the mean |delta|,
plus the rank changes of the positives.  Verifies exactly which paths respond to
query time (expected: only REC through its recency feature).

    python -m exp.review.query_time_probe --run results/event_bench/leakfree2/wiki_f_s43 --out results/review_2026_09_22/clock/probe_wiki_s43 --shifts 3600 86400 2592000
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from exp import temporal_benchmark_utils as bu
from exp.run_event_benchmark import _context_edge_index, _make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", type=float, nargs="+", default=[3600, 86400, 2592000])
    ap.add_argument("--batches", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pd.read_csv(os.path.join(a.run, "results.csv")).iloc[0]
    cfg = json.loads(r["config_json"])
    args = argparse.Namespace(**cfg)
    for k, v in dict(backbone="tsd", spatial=None, clock="global", fast_core_off=False).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec, dataset, td = bu.load_temporal_data(args.dataset)
    num_nodes = max(int(td.src.max()), int(td.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    feats = bu.make_node_features(dataset, num_nodes).to(device)
    train_ids, val_ids, test_ids, _ = bu.split_edge_ids(dataset, td)
    if args.train_edges_cap is not None:
        train_ids = train_ids[-args.train_edges_cap:]
    ctx = _context_edge_index(td, train_ids, args.context_edges)
    train_s = bu.build_snapshots(td, train_ids, feats, time_window=args.time_window)
    val_s = bu.build_snapshots(td, val_ids, feats, time_window=args.time_window)
    dataset.load_val_ns(); dataset.load_val_ns = lambda: None
    node_types = None
    if getattr(args, "node_type_emb", False):
        nt = getattr(dataset, "node_type", None); node_types = None if nt is None else torch.as_tensor(nt)[:num_nodes]
    model = _make_model(ctx, feats, num_nodes, device, args, num_relations, node_types=node_types)
    if not args.no_dst_range:
        lo, hi = int(td.dst.min()), int(td.dst.max())
        if (lo, hi) != (0, num_nodes - 1):
            model.event_destination_range = (lo, hi)
    if hasattr(model, "ssm") and hasattr(model.ssm, "set_delta_scale"):
        ts = np.array([float(s.timestamp) for s in train_s]); g = np.diff(ts); g = g[g > 0]
        model.ssm.set_delta_scale(float(np.median(g)) if g.size else 1.0)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=device)); model.eval()
    os.environ["TSD_SCORE_FROM_PREVIOUS_STATE"] = "1"
    model.reset_temporal_state()
    state = bu.advance_context(model, train_s)
    out = {"spatial": state.spatial, "x": model._head_input_features(feats) if hasattr(model, "_head_input_features") else feats, "node_signal": None}
    rows = []
    with torch.no_grad():
        for k, snap in enumerate(val_s[: a.batches]):
            if snap.src.numel() == 0:
                continue
            src, dst = snap.src.to(device), snap.dst.to(device)
            et = snap.edge_types.to(device) if snap.edge_types is not None else None
            neg = dataset.negative_sampler.query_batch(snap.src.cpu(), snap.dst.cpu(), snap.edge_timestamps.cpu(),
                                                       edge_type=snap.edge_types.cpu() if snap.edge_types is not None else None, split_mode="val")
            padded, mask = bu._pad_negative_samples(neg)
            if padded is None:
                continue
            negd = torch.as_tensor(padded, dtype=torch.long, device=device)
            qt0 = snap.edge_timestamps.to(device)
            base_pos_neural = super(type(model), model).score_event_pairs(out, src, dst, edge_type=et)
            base_neg_neural = super(type(model), model).score_event_candidates(out, src, negd, edge_type=et)
            for shift in [0.0] + list(a.shifts):
                qt = qt0 + shift
                pos_total = model.score_event_pairs(out, src, dst, edge_type=et, query_time=qt)
                neg_total = model.score_event_candidates(out, src, negd, edge_type=et, query_time=qt)
                pos_neural = super(type(model), model).score_event_pairs(out, src, dst, edge_type=et)
                neg_neural = super(type(model), model).score_event_candidates(out, src, negd, edge_type=et)
                pos_rec = pos_total - pos_neural; neg_rec = neg_total - neg_neural
                negm = neg_total.masked_fill(~torch.as_tensor(mask, device=device), float("-inf"))
                rank = 0.5 * ((negm > pos_total[:, None]).sum(1) + (negm >= pos_total[:, None]).sum(1)).float() + 1
                rows.append(dict(batch=k, shift_sec=shift, queries=int(src.numel()),
                                 neural_changed_frac=float((pos_neural != base_pos_neural).float().mean()),
                                 neural_mean_abs_delta=float((pos_neural - base_pos_neural).abs().mean()),
                                 rec_changed_frac=float(((pos_rec - (rows[-1]["_rec0"] if False else 0)) != 0).float().mean()) if False else float("nan"),
                                 pos_rec_mean=float(pos_rec.mean()), neg_rec_mean=float(neg_rec[torch.as_tensor(mask, device=device)].mean()),
                                 pos_total_mean=float(pos_total.mean()), mrr=float((1.0 / rank).mean()),
                                 rec_nonzero_frac=float((pos_rec != 0).float().mean())))
    df = pd.DataFrame(rows)
    # REC/total response relative to shift 0 per batch
    base = df[df.shift_sec == 0].set_index("batch")
    df["pos_rec_delta_vs_t0"] = df.apply(lambda r: r.pos_rec_mean - base.loc[r.batch, "pos_rec_mean"], axis=1)
    df["mrr_delta_vs_t0"] = df.apply(lambda r: r.mrr - base.loc[r.batch, "mrr"], axis=1)
    df.to_csv(os.path.join(a.out, "query_time_probe.csv"), index=False)
    summ = df.groupby("shift_sec").agg(queries=("queries", "sum"), neural_changed_frac=("neural_changed_frac", "mean"),
                                       neural_mean_abs_delta=("neural_mean_abs_delta", "mean"), pos_rec_delta_vs_t0=("pos_rec_delta_vs_t0", "mean"),
                                       mrr=("mrr", "mean"), mrr_delta_vs_t0=("mrr_delta_vs_t0", "mean"), rec_nonzero_frac=("rec_nonzero_frac", "mean")).reset_index()
    summ.to_csv(os.path.join(a.out, "query_time_probe_summary.csv"), index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
