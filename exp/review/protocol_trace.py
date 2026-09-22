"""Node-property protocol trace (server handoff 2026-09-22, section 1).

For tgbn-trade / tgbn-genre on the INSTALLED TGB package and data:
  * traces every label our runner scores (split seek + drain after ingesting a
    whole snapshot) with the exact observation frontier before scoring;
  * traces the official TGN example's deferred-cursor schedule (batch 200,
    `query_t > label_t` -> fire one label, ingest current-batch edges with
    t < next label time, score, ingest the rest) with the same fields;
  * compares scored label sets per split (boundary assignments, unscored
    terminal labels);
  * reconstructs label vectors from raw edges for candidate generating
    windows (max abs error) to pin the label semantics;
  * evaluates three diagnostic predictors on the exact scored labels under
    both schedules: last released label, forecast from previous completed
    periods (1 and 3), and copy/reconstruction from currently available edges.
Outputs: protocol_trace.csv, protocol_examples.csv, protocol_label_sets.csv,
label_reconstruction.csv, diagnostic_predictors.csv, diagnostic_scores.csv.gz,
protocol_verdict.md (facts only; interpretation is left to FINDINGS.md).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from exp import temporal_benchmark_utils as bu
from exp.temporal_mamba_studies import prepare_temporal_experiment_context


def _ndcg(evaluator, y_true, y_pred):
    return float(evaluator.eval({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "eval_metric": ["ndcg"]})["ndcg"])


def _hash_labels(srcs, labels):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(srcs)).tobytes() + np.ascontiguousarray(np.asarray(labels)).tobytes()).hexdigest()[:16]


class Frontier:
    """Dummy state: only records which raw edges were ingested."""

    def __init__(self, t_all):
        self.t_all = t_all
        self.n = 0
        self.t_min = None
        self.t_max = None
        self.ingested_ids = []

    def ingest(self, ids):
        ids = np.asarray(ids)
        if ids.size == 0:
            return
        t = self.t_all[ids]
        self.n += ids.size
        self.t_min = int(t.min()) if self.t_min is None else min(self.t_min, int(t.min()))
        self.t_max = int(t.max()) if self.t_max is None else max(self.t_max, int(t.max()))
        self.ingested_ids.append(ids)

    def counts_relative_to(self, label_ts):
        if not self.ingested_ids:
            return 0, 0, 0
        ids = np.concatenate(self.ingested_ids)
        t = self.t_all[ids]
        return int((t < label_ts).sum()), int((t == label_ts).sum()), int((t > label_ts).sum())


def trace_ours(ds, td, context, time_window, train_cap, t_all):
    inner = getattr(ds, "dataset", ds)
    bundle = context.get_snapshot_bundle(time_window)
    rows, examples = [], []
    fr = Frontier(t_all)
    ingested_before = {}   # label_ts -> frontier snapshot (for the diagnostic predictors)
    for split in ("train", "val", "test"):
        snaps = bundle[f"{split}_snapshots"]
        bu.seek_label_cursor(ds, snaps)
        cursor_start = int(inner.label_ts_idx)
        n_lab = 0
        for k, snap in enumerate(snaps):
            fr.ingest(snap.edge_ids.numpy())
            pending_before = int(inner.label_ts[inner.label_ts_idx]) if inner.label_ts_idx < len(inner.label_ts) else -1
            drained = bu.drain_snapshot_labels(ds, snap)
            for j, (lts, srcs, labels) in enumerate(drained):
                lt, eq, gt = fr.counts_relative_to(lts)
                pending_after = int(inner.label_ts[inner.label_ts_idx]) if inner.label_ts_idx < len(inner.label_ts) else -1
                rows.append(dict(runner="ours", dataset=context.dataset_name, split=split, label_ts=int(lts),
                                 snapshot_index=k, snapshot_ts=int(snap.timestamp), snapshot_edges=int(snap.src.numel()),
                                 snapshot_t_min=int(snap.edge_timestamps.min()), snapshot_t_max=int(snap.edge_timestamps.max()),
                                 frontier_n=fr.n, frontier_t_min=fr.t_min, frontier_t_max=fr.t_max,
                                 ingested_t_lt_label=lt, ingested_t_eq_label=eq, ingested_t_gt_label=gt,
                                 label_nodes=int(len(srcs)), label_hash=_hash_labels(srcs, labels),
                                 pending_label_before=pending_before, pending_label_after=pending_after,
                                 labels_drained_in_snapshot=len(drained), drain_position=j,
                                 first_in_split=(n_lab == 0), last_in_split=False))
                ingested_before[(split, int(lts))] = np.concatenate(fr.ingested_ids) if fr.ingested_ids else np.zeros(0, dtype=np.int64)
                n_lab += 1
        if rows:
            for r in reversed(rows):
                if r["split"] == split and r["runner"] == "ours":
                    r["last_in_split"] = True
                    break
        examples.append(dict(runner="ours", split=split, cursor_index_at_seek=cursor_start,
                             split_first_edge_ts=int(snaps[0].edge_timestamps.min()) if snaps else None,
                             split_last_edge_ts=int(snaps[-1].edge_timestamps.max()) if snaps else None,
                             snapshots=len(snaps), labels_scored=n_lab,
                             note=f"time_window={time_window}, train_cap={train_cap} (prefix of the training split)"))
    return pd.DataFrame(rows), examples, ingested_before


def trace_official(ds, td, t_all, batch_size=200):
    """Symbolic replay of examples/nodeproppred/tgbn-*/tgn.py with a dummy state."""
    inner = getattr(ds, "dataset", ds)
    masks = {s: np.asarray(getattr(ds, f"{s}_mask")) for s in ("train", "val", "test")}
    ds.reset_label_time()
    fr = Frontier(t_all)
    rows, examples = [], []
    ingested_before = {}
    for split in ("train", "val", "test"):
        ids = np.nonzero(masks[split])[0]
        ids = ids[np.argsort(t_all[ids], kind="stable")]
        label_t = int(inner.return_label_ts())
        n_lab, n_batches, unscored_break = 0, 0, False
        for b0 in range(0, ids.size, batch_size):
            bid = ids[b0:b0 + batch_size]
            n_batches += 1
            query_t = int(t_all[bid[-1]])
            rest = bid
            if query_t > label_t:
                tup = ds.get_node_label(query_t)
                if tup is None:
                    unscored_break = True
                    fr.ingest(bid)
                    continue
                lts, srcs, labels = tup
                lts = int(np.asarray(lts).reshape(-1)[0])
                label_t = int(inner.return_label_ts())
                prev_mask = t_all[bid] < label_t
                fr.ingest(bid[prev_mask])
                rest = bid[~prev_mask]
                lt, eq, gt = fr.counts_relative_to(lts)
                rows.append(dict(runner="official_tgn_example", dataset=ds.name if hasattr(ds, "name") else "", split=split,
                                 label_ts=lts, snapshot_index=n_batches - 1, snapshot_ts=query_t, snapshot_edges=int(bid.size),
                                 snapshot_t_min=int(t_all[bid].min()), snapshot_t_max=int(t_all[bid].max()),
                                 frontier_n=fr.n, frontier_t_min=fr.t_min, frontier_t_max=fr.t_max,
                                 ingested_t_lt_label=lt, ingested_t_eq_label=eq, ingested_t_gt_label=gt,
                                 label_nodes=int(len(srcs)), label_hash=_hash_labels(srcs, labels),
                                 pending_label_before=lts, pending_label_after=label_t,
                                 labels_drained_in_snapshot=1, drain_position=0, first_in_split=(n_lab == 0), last_in_split=False))
                ingested_before[(split, lts)] = np.concatenate(fr.ingested_ids)
                n_lab += 1
            fr.ingest(rest)
        for r in reversed(rows):
            if r["split"] == split:
                r["last_in_split"] = True
                break
        examples.append(dict(runner="official_tgn_example", split=split, batches=n_batches, labels_scored=n_lab,
                             split_first_edge_ts=int(t_all[ids].min()), split_last_edge_ts=int(t_all[ids].max()),
                             pending_label_at_split_end=int(inner.return_label_ts()), cursor_exhausted_break=unscored_break,
                             note=f"batch_size={batch_size}; label fires when batch last t > pending label ts; one label per batch"))
    return pd.DataFrame(rows), examples, ingested_before


def period_aggregates(src, dst, t, w, n_nodes, n_classes, period_of):
    """{period: {u: normalised weight vector over classes}} computed with pandas."""
    df = pd.DataFrame({"p": period_of(t), "u": src, "v": dst, "w": w})
    df = df[df.v < n_classes]
    g = df.groupby(["p", "u", "v"], sort=False)["w"].sum().reset_index()
    out = {}
    for p, sub in g.groupby("p", sort=False):
        vec = {}
        for u, s2 in sub.groupby("u", sort=False):
            v = np.zeros(n_classes)
            v[s2.v.values.astype(int)] = s2.w.values
            tot = v.sum()
            vec[int(u)] = v / tot if tot > 0 else v
        out[int(p)] = vec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbn-trade")
    ap.add_argument("--time-window", type=float, default=1.0)
    ap.add_argument("--train-cap", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--max-label-days-recon", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    tw = int(args.time_window)
    context = prepare_temporal_experiment_context(
        args.dataset, split_caps={"train": (None if args.train_cap < 0 else args.train_cap), "val": None, "test": None},
        device=torch.device("cpu"), preload_time_windows=[tw])
    ds, td = context.dataset, context.temporal_data
    inner = getattr(ds, "dataset", ds)
    t_all = td.t.numpy().astype(np.int64)
    src, dst = td.src.numpy().astype(np.int64), td.dst.numpy().astype(np.int64)
    w = td.msg.numpy().reshape(len(t_all), -1)[:, 0].astype(np.float64) if getattr(td, "msg", None) is not None else np.ones(len(t_all))
    label_ts = np.asarray(inner.label_ts).astype(np.int64)
    n_classes = int(next(iter(next(iter(inner.label_dict.values())).values())).shape[0])
    print(f"{args.dataset}: {len(t_all)} edges, {len(label_ts)} label timestamps, {n_classes} classes; "
          f"splits train<= {int(t_all[np.asarray(ds.train_mask)].max())} val<= {int(t_all[np.asarray(ds.val_mask)].max())}", flush=True)

    ours, ex_ours, ing_ours = trace_ours(ds, td, context, tw, args.train_cap, t_all)
    official, ex_off, ing_off = trace_official(ds, td, t_all, args.batch_size)
    trace = pd.concat([ours, official], ignore_index=True)
    trace.to_csv(os.path.join(args.out, "protocol_trace.csv"), index=False)
    pd.DataFrame(ex_ours + ex_off).to_csv(os.path.join(args.out, "protocol_examples.csv"), index=False)

    # label-set comparison per split
    sets = []
    for split in ("train", "val", "test"):
        a = set(ours[ours.split == split].label_ts); b = set(official[official.split == split].label_ts)
        sets.append(dict(split=split, ours_n=len(a), official_n=len(b), ours_only=sorted(a - b), official_only=sorted(b - a),
                         ours_first=min(a) if a else None, ours_last=max(a) if a else None,
                         official_first=min(b) if b else None, official_last=max(b) if b else None))
    all_scored_ours = set(ours.label_ts); all_scored_off = set(official.label_ts)
    sets.append(dict(split="never_scored", ours_n=len(set(label_ts) - all_scored_ours), official_n=len(set(label_ts) - all_scored_off),
                     ours_only=sorted(set(label_ts) - all_scored_ours)[:20], official_only=sorted(set(label_ts) - all_scored_off)[:20]))
    pd.DataFrame(sets).to_csv(os.path.join(args.out, "protocol_label_sets.csv"), index=False)
    print("label sets:", json.dumps(sets, default=str)[:1500], flush=True)

    # ---- label reconstruction: which edge window generates label(ts, u)? ----
    unit = tw if args.dataset == "tgbn-trade" else 86400
    cands = [("same_period", 0, 1), ("next_period", 1, 1), ("prev_period", -1, 1)]
    if args.dataset != "tgbn-trade":
        cands += [("next_7", 0, 7), ("next_8_incl", 0, 8), ("prev_7", -7, 7), ("next_7_from_+1", 1, 7), ("window_-1_+7", -1, 8)]
    recon = []
    chosen = np.linspace(0, len(label_ts) - 1, min(args.max_label_days_recon, len(label_ts))).astype(int)
    for name, off, length in cands:
        errs, n_pairs = [], 0
        for i in chosen:
            lts = int(label_ts[i]); lo = lts + off * unit; hi = lo + length * unit
            m = (t_all >= lo) & (t_all < hi)
            agg = period_aggregates(src[m], dst[m], t_all[m], w[m], context.num_nodes, n_classes, lambda t: np.zeros_like(t))
            vecs = agg.get(0, {})
            for u, lab in inner.label_dict[lts].items():
                lab = np.asarray(lab, dtype=np.float64); s = lab.sum(); labn = lab / s if s > 0 else lab
                pred = vecs.get(int(u), np.zeros(n_classes))
                errs.append(float(np.abs(pred - labn).max())); n_pairs += 1
        recon.append(dict(candidate=name, offset_periods=off, length_periods=length, unit_sec_or_years=unit,
                          label_days_checked=len(chosen), pairs=n_pairs, max_abs_err_mean=float(np.mean(errs)) if errs else np.nan,
                          max_abs_err_max=float(np.max(errs)) if errs else np.nan,
                          frac_pairs_exact_1e6=float(np.mean(np.asarray(errs) < 1e-6)) if errs else np.nan))
        print(f"reconstruction {name}: mean max-abs-err {recon[-1]['max_abs_err_mean']:.4g}, exact frac {recon[-1]['frac_pairs_exact_1e-6']:.3f}", flush=True)
    recon_df = pd.DataFrame(recon); recon_df.to_csv(os.path.join(args.out, "label_reconstruction.csv"), index=False)
    best = recon_df.sort_values("max_abs_err_mean").iloc[0]

    # ---- diagnostic predictors on the exact scored labels (val/test) under both schedules ----
    from tgb.nodeproppred.evaluate import Evaluator
    evaluator = Evaluator(name=args.dataset)
    gen_off, gen_len = int(best.offset_periods), int(best.length_periods)
    period_of = lambda t: (t // unit).astype(np.int64)
    per_period = period_aggregates(src, dst, t_all, w, context.num_nodes, n_classes, period_of)   # completed-period aggregates
    label_sorted = np.sort(label_ts)
    scores, summary = [], []
    for runner, trace_df, ing in (("ours", ours, ing_ours), ("official_tgn_example", official, ing_off)):
        for split in ("val", "test"):
            sub = trace_df[trace_df.split == split]
            per_ts = {k: [] for k in ("last_label", "prev_1_period", "prev_3_periods", "copy_available_edges")}
            for _, r in sub.iterrows():
                lts = int(r.label_ts)
                labels = inner.label_dict[lts]; users = np.array(sorted(labels.keys())); Y = np.stack([np.asarray(labels[u]) for u in users])
                # (a) last released label of the same node (previous label timestamp)
                prev_idx = np.searchsorted(label_sorted, lts) - 1
                P_last = np.zeros_like(Y)
                if prev_idx >= 0:
                    prev = inner.label_dict[int(label_sorted[prev_idx])]
                    for i, u in enumerate(users):
                        if int(u) in prev:
                            P_last[i] = np.asarray(prev[int(u)])
                # (b) forecast from previous COMPLETED periods relative to the generating period of this label
                gen_p = int((lts + gen_off * unit) // unit)
                P1 = np.zeros_like(Y); P3 = np.zeros_like(Y)
                for i, u in enumerate(users):
                    v1 = per_period.get(gen_p - 1, {}).get(int(u)); P1[i] = v1 if v1 is not None else 0
                    acc = [per_period.get(gen_p - k, {}).get(int(u)) for k in (1, 2, 3)]
                    acc = [a for a in acc if a is not None]; P3[i] = np.mean(acc, axis=0) if acc else 0
                # (c) copy from the edges actually ingested before scoring, restricted to the generating window
                ids = ing.get((split, lts), np.zeros(0, dtype=np.int64))
                lo = lts + gen_off * unit; hi = lo + gen_len * unit
                m = (t_all[ids] >= lo) & (t_all[ids] < hi)
                agg = period_aggregates(src[ids][m], dst[ids][m], t_all[ids][m], w[ids][m], context.num_nodes, n_classes, lambda t: np.zeros_like(t)).get(0, {})
                Pc = np.stack([agg.get(int(u), np.zeros(n_classes)) for u in users])
                for name, P in (("last_label", P_last), ("prev_1_period", P1), ("prev_3_periods", P3), ("copy_available_edges", Pc)):
                    s = _ndcg(evaluator, Y, P)
                    per_ts[name].append(s)
                    scores.append(dict(runner=runner, split=split, label_ts=lts, predictor=name, ndcg=s, nodes=len(users),
                                       edges_available_in_window=int(m.sum())))
            for name, v in per_ts.items():
                summary.append(dict(runner=runner, split=split, predictor=name, label_timestamps=len(v), ndcg_mean=float(np.mean(v)) if v else np.nan))
            print(f"[{runner} {split}] " + ", ".join(f"{k}={np.mean(v):.4f}" for k, v in per_ts.items() if v), flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(args.out, "diagnostic_predictors.csv"), index=False)
    pd.DataFrame(scores).to_csv(os.path.join(args.out, "diagnostic_scores.csv.gz"), index=False, compression="gzip")

    with open(os.path.join(args.out, "protocol_verdict.md"), "w") as fh:
        fh.write(f"# Protocol trace: {args.dataset} (time_window={tw}, train_cap={args.train_cap}; installed py-tgb)\n\n")
        fh.write(f"Generated {time.strftime('%Y-%m-%d %H:%M')} in {time.time() - t0:.0f}s. Facts only.\n\n")
        fh.write("## Label semantics (reconstruction from raw edges)\n\n" + recon_df.to_markdown(index=False) + "\n\n")
        fh.write(f"Best-matching generating window: **{best.candidate}** (offset {gen_off} period(s), length {gen_len}), "
                 f"mean max-abs error {best.max_abs_err_mean:.3g}.\n\n")
        fh.write("## Scored label sets per split\n\n" + pd.DataFrame(sets).to_markdown(index=False) + "\n\n")
        fh.write("## Split-level examples\n\n" + pd.DataFrame(ex_ours + ex_off).to_markdown(index=False) + "\n\n")
        fh.write("## Boundary labels (first/last of every split, both runners)\n\n")
        bd = trace[(trace.first_in_split) | (trace.last_in_split)][["runner", "split", "label_ts", "snapshot_ts", "frontier_t_max", "ingested_t_lt_label", "ingested_t_eq_label", "ingested_t_gt_label", "label_nodes", "label_hash", "first_in_split", "last_in_split"]]
        fh.write(bd.to_markdown(index=False) + "\n\n")
        fh.write("## Diagnostic predictors (NDCG@10, mean over scored label timestamps)\n\n" + pd.DataFrame(summary).to_markdown(index=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
