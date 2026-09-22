"""Gap-feature intervention with fixed batch membership (server handoff 2026-09-22, section 2).

Replays a trained checkpoint and evaluates validation MRR while the physical
gap supplied to the step selector is multiplied by a factor (through the
non-learned `delta_scale` buffer: gap/scale), WITHOUT changing the batches,
the events, the REC recency features or the query times.  This isolates the
model's sensitivity to the timing channel from re-bucketing (the existing
synthetic --time-scale re-buckets and is not this intervention).

    python -m exp.review.gap_intervention --run results/event_bench/leakfree2/wiki_f_s43 --out results/review_2026_09_22/clock/gap_wiki_s43 --factors 0.25 0.5 1 2 4 --val-edges 30000
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
    ap.add_argument("--factors", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--val-edges", type=int, default=30000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pd.read_csv(os.path.join(a.run, "results.csv")).iloc[0]
    cfg = json.loads(r["config_json"]); args = argparse.Namespace(**cfg)
    for k, v in dict(backbone="tsd", spatial=None, clock="global", fast_core_off=False, no_memory=False, emb_in_head=False,
                     node_type_emb=False, learn_node_emb=False, sheaf_identity=False, no_delta_t=False, relation_in_input=False,
                     recurrency_decoder=False, recurrency_untyped=False, recurrency_symmetric=False, no_memory_readout=False,
                     no_dst_range=False, train_edges_cap=None, time_window=None, context_edges=50000).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec, dataset, td = bu.load_temporal_data(args.dataset)
    num_nodes = max(int(td.src.max()), int(td.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    feats = bu.make_node_features(dataset, num_nodes).to(device)
    train_ids, val_ids, _, _ = bu.split_edge_ids(dataset, td)
    if args.train_edges_cap is not None:
        train_ids = train_ids[-args.train_edges_cap:]
    ctx = _context_edge_index(td, train_ids, args.context_edges)
    train_s = bu.build_snapshots(td, train_ids, feats, time_window=args.time_window)
    val_s = bu.build_snapshots(td, val_ids[: a.val_edges], feats, time_window=args.time_window)
    dataset.load_val_ns(); dataset.load_val_ns = lambda: None
    node_types = None
    if getattr(args, "node_type_emb", False):
        nt = getattr(dataset, "node_type", None); node_types = None if nt is None else torch.as_tensor(nt)[:num_nodes]
    model = _make_model(ctx, feats, num_nodes, device, args, num_relations, node_types=node_types)
    if not args.no_dst_range:
        lo, hi = int(td.dst.min()), int(td.dst.max())
        if (lo, hi) != (0, num_nodes - 1):
            model.event_destination_range = (lo, hi)
    ts = np.array([float(s.timestamp) for s in train_s]); g = np.diff(ts); g = g[g > 0]
    base_scale = float(np.median(g)) if g.size else 1.0
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=device)); model.eval()
    os.environ["TSD_SCORE_FROM_PREVIOUS_STATE"] = "1"
    rows = []
    for f in a.factors:
        model.ssm.set_delta_scale(base_scale / f)   # gap/scale = f * (gap/base)  -> gaps appear f times longer
        model.reset_temporal_state()
        state = bu.advance_context(model, train_s)
        mrr, loss, _ = bu.evaluate_model_streaming(spec, dataset, val_s, model, initial_state=state, split_mode="val",
                                                   label_cursor_after_ts=bu.last_snapshot_timestamp(train_s))
        rows.append(dict(run=os.path.basename(a.run.rstrip("/")), gap_factor=f, delta_scale=base_scale / f, val_mrr=float(mrr),
                         val_hits10=float(getattr(bu.evaluate_model_streaming, "last_hits10", float("nan"))), val_loss=float(loss),
                         val_edges=int(sum(s.src.numel() for s in val_s)),
                         note="batches, events, REC recency and query times unchanged; only the selector's gap input is scaled"))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "gap_intervention.csv"), index=False)


if __name__ == "__main__":
    main()
