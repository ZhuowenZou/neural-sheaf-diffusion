"""Stratified analysis of the synthetic history task (server handoff 2026-09-22, section 4).

Joins every finished run's per-query reciprocal ranks (query_ranks.csv.gz, keyed by global edge id) with the
generator sidecar (novel / recurring pair, cue event or interaction, cue-known-in-permitted-history) and the
oracle reciprocal ranks on the SAME queries, and writes per-run and per-arm tables.

    python -m exp.review.synthetic_analysis --gen results/review_2026_09_22/synthetic/gen_s1 --runs "results/review_2026_09_22/synthetic/runs/synth_gen_s1_*"
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from exp.review.collect_review import arm_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.gen.rstrip("/")), "synthetic_analysis")
    os.makedirs(out, exist_ok=True)
    z = np.load(os.path.join(a.gen, "data.npz"), allow_pickle=False)
    side = pd.DataFrame({"edge_id": np.arange(z["src"].size), "novel": z["novel"], "kind": z["kind"], "split_gen": z["split"]})
    orc = pd.read_csv(os.path.join(a.gen, "oracle_per_query.csv.gz")).rename(columns={"event": "edge_id"})
    rows, per_arm = [], []
    for d in sorted(glob.glob(a.runs)):
        res = os.path.join(d, "results.csv"); qr = os.path.join(d, "query_ranks.csv.gz")
        if not (os.path.exists(res) and os.path.exists(qr)):
            continue
        r = pd.read_csv(res).iloc[0]; cfg = json.loads(r["config_json"]); arm = arm_of(cfg); rec = "on" if cfg.get("recurrency_decoder") else "off"
        q = pd.read_csv(qr).merge(side, on="edge_id", how="left").merge(orc[["edge_id", "rr_permitted", "rr_latent", "rr_chance", "cue_known"]], on="edge_id", how="left")
        q = q[q.split == 2]   # test split
        inter = q[q.kind == 0]
        rec_ = dict(run=os.path.basename(d), arm=arm, rec=rec, seed=int(r.seed), test_mrr_all=float(q.rr.mean()), n_all=len(q),
                    mrr_interactions=float(inter.rr.mean()), n_interactions=len(inter),
                    mrr_cue_events=float(q[q.kind == 1].rr.mean()) if (q.kind == 1).any() else np.nan,
                    mrr_novel=float(inter[inter.novel].rr.mean()), n_novel=int(inter.novel.sum()),
                    mrr_recurring=float(inter[~inter.novel].rr.mean()), n_recurring=int((~inter.novel).sum()),
                    mrr_cue_known=float(inter[inter.cue_known == True].rr.mean()), mrr_cue_unknown=float(inter[inter.cue_known == False].rr.mean()) if (inter.cue_known == False).any() else np.nan,
                    oracle_permitted=float(inter.rr_permitted.mean()), oracle_latent=float(inter.rr_latent.mean()), chance=float(inter.rr_chance.mean()),
                    oracle_permitted_novel=float(inter[inter.novel].rr_permitted.mean()), oracle_permitted_recurring=float(inter[~inter.novel].rr_permitted.mean()))
        rows.append(rec_)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_run_stratified.csv"), index=False)
    if len(df):
        g = df.groupby(["arm", "rec"]).agg(n=("seed", "count"), mrr=("mrr_interactions", "mean"), sd=("mrr_interactions", "std"),
                                           novel=("mrr_novel", "mean"), recurring=("mrr_recurring", "mean"), cue_known=("mrr_cue_known", "mean"),
                                           oracle_permitted=("oracle_permitted", "mean"), oracle_latent=("oracle_latent", "mean"), chance=("chance", "mean")).reset_index()
        g.to_csv(os.path.join(out, "per_arm_stratified.csv"), index=False)
        print(g.to_string(index=False))
        # paired vs tsd on common seeds
        pairs = []
        for rec, gg in df.groupby("rec"):
            base = gg[gg.arm == "tsd"].set_index("seed").mrr_interactions
            for arm, ga in gg[gg.arm != "tsd"].groupby("arm"):
                s = ga.set_index("seed").mrr_interactions; common = sorted(set(s.index) & set(base.index))
                if common:
                    d = np.array([s[k] - base[k] for k in common])
                    pairs.append(dict(rec=rec, arm=arm, n=len(common), delta_mean=d.mean(), delta_sd=d.std(ddof=1) if len(d) > 1 else np.nan, signs=f"{(d>0).sum()}+ {(d<0).sum()}-"))
        pd.DataFrame(pairs).to_csv(os.path.join(out, "paired_vs_tsd.csv"), index=False)
        if pairs:
            print(pd.DataFrame(pairs).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
