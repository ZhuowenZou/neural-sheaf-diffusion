"""Stratified wiki analysis (server handoff, section 3): per-query reciprocal ranks of every matched run
(query_ranks.csv.gz, keyed by global edge id) joined with (a) pair recurrence -- whether (src, dst) occurred
earlier in the stream -- and (b) the source's inactivity gap -- time since its previous event -- computed
from the raw tgbl-wiki stream; MRR per arm per stratum and paired deltas vs TSD on identical queries.

    python -m exp.review.wiki_strata --runs "results/review_2026_09_22/matched/wiki_*_lr1e-3" --out results/review_2026_09_22/matched/wiki_strata
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from exp import temporal_benchmark_utils as bu
from exp.review.collect_review import arm_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec, ds, td = bu.load_temporal_data("tgbl-wiki")
    src = td.src.numpy(); dst = td.dst.numpy(); t = td.t.numpy().astype(np.int64)
    n = src.size
    seen = set(); recurring = np.zeros(n, dtype=bool); last_src = {}; gap = np.full(n, np.nan)
    for i in range(n):
        key = (int(src[i]), int(dst[i]))
        recurring[i] = key in seen; seen.add(key)
        u = int(src[i])
        if u in last_src:
            gap[i] = t[i] - last_src[u]
        last_src[u] = t[i]
    side = pd.DataFrame({"edge_id": np.arange(n), "recurring": recurring, "gap": gap})
    test_ids = np.nonzero(np.asarray(ds.test_mask))[0]
    q = np.nanquantile(side.gap.values[test_ids], [0.25, 0.5, 0.75])
    side["gap_q"] = np.where(np.isnan(side.gap), "first", np.where(side.gap <= q[0], "q1", np.where(side.gap <= q[1], "q2", np.where(side.gap <= q[2], "q3", "q4"))))
    rows = []
    for d in sorted(glob.glob(a.runs)):
        f = os.path.join(d, "query_ranks.csv.gz"); r = os.path.join(d, "results.csv")
        if not (os.path.exists(f) and os.path.exists(r)):
            continue
        res = pd.read_csv(r).iloc[0]; cfg = json.loads(res["config_json"])
        qr = pd.read_csv(f); qr = qr[qr.split == 2].merge(side, on="edge_id", how="left")
        rec = dict(run=os.path.basename(d), arm=arm_of(cfg), seed=int(res.seed), n=len(qr), mrr=qr.rr.mean(),
                   mrr_recurring=qr[qr.recurring].rr.mean(), mrr_novel=qr[~qr.recurring].rr.mean(),
                   frac_recurring=qr.recurring.mean())
        for g in ("first", "q1", "q2", "q3", "q4"):
            rec[f"mrr_gap_{g}"] = qr[qr.gap_q == g].rr.mean()
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "per_run_strata.csv"), index=False)
    cols = ["mrr", "mrr_recurring", "mrr_novel", "mrr_gap_first", "mrr_gap_q1", "mrr_gap_q2", "mrr_gap_q3", "mrr_gap_q4"]
    per_arm = df.groupby("arm")[cols].mean().round(4); per_arm["n"] = df.groupby("arm").seed.count()
    per_arm.to_csv(os.path.join(a.out, "per_arm_strata.csv"))
    base = df[df.arm == "tsd"].set_index("seed")
    pairs = []
    for arm, g in df[df.arm != "tsd"].groupby("arm"):
        g = g.set_index("seed"); common = sorted(set(g.index) & set(base.index))
        rec = dict(arm=arm, n=len(common))
        for c in cols:
            dlt = np.array([g.loc[s, c] - base.loc[s, c] for s in common])
            rec[f"d_{c}"] = round(dlt.mean(), 4); rec[f"signs_{c}"] = f"{(dlt>0).sum()}+ {(dlt<0).sum()}-"
        pairs.append(rec)
    pd.DataFrame(pairs).to_csv(os.path.join(a.out, "paired_strata_vs_tsd.csv"), index=False)
    print("gap quartile edges (s):", q.tolist(), "| test recurring fraction:", round(float(side.recurring.values[test_ids].mean()), 3))
    print(per_arm.to_string()); print(pd.DataFrame(pairs)[["arm", "n", "d_mrr_recurring", "signs_mrr_recurring", "d_mrr_novel", "signs_mrr_novel", "d_mrr_gap_q4", "signs_mrr_gap_q4"]].to_string(index=False))


if __name__ == "__main__":
    main()
