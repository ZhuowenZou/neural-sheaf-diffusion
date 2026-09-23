"""Collect the review campaign (server handoff 2026-09-22, section 8).

Reads every results/review_2026_09_22/{matched,audit,forum,...}/*/results.csv and writes, under
results/review_2026_09_22/:
  per_seed_results.csv   one row per finished run (arm derived from the resolved config, never from the name)
  paired_contrasts.csv   per dataset x lr-locked arm: per-seed paired deltas vs TSD, mean, SD, t-interval, signs
  costs.csv              timers / memory / storage / parameter counts per run
  numerical_events.csv   training and evaluation safeguard counters with denominators per run
  query_validity.csv     concatenated per-split score-validity audits (new runs + checkpoint replays)
  REVIEW_SUMMARY.md      tables of the above plus unfinished / crashed runs
"""
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

RV = "results/review_2026_09_22"


def arm_of(cfg):
    if cfg.get("no_memory") and int(cfg.get("layers", 2)) == 0:
        return "core-off"
    b = cfg.get("backbone", "tsd"); sp = cfg.get("spatial") or ("identity" if cfg.get("sheaf_identity") else "sheaf")
    cond = cfg.get("sheaf_conditioning", "history")
    if b == "gru":
        return "gru-ordinary" if sp == "identity" else f"gru-{sp}"
    if b == "diag_ssm":
        return "diagssm-ordinary" if sp == "identity" else f"diagssm-{sp}"
    if sp == "identity":
        return "identity-maps"
    if sp == "node_frame":
        return "node-frame"
    if sp == "attention":
        return "attention-gates"
    if cond == "current_only":
        return "current-only-maps"
    if cfg.get("no_delta_t"):
        return "tsd-no-gap"
    if cfg.get("no_memory"):
        return "tsd-no-memory"
    return "tsd"


def collect():
    rows, unfinished = [], []
    for res in sorted(glob.glob(f"{RV}/*/*/results.csv")):
        d = pd.read_csv(res).iloc[0].to_dict()
        try:
            cfg = json.loads(d.get("config_json", "{}"))
        except Exception:
            cfg = {}
        run = res.split("/")[-2]; group = res.split("/")[-3]
        d.update(run=run, group=group, arm=arm_of(cfg), lr=cfg.get("lr"), rec="on" if cfg.get("recurrency_decoder") else "off",
                 clock=cfg.get("clock", "global"), is_audit=bool(cfg.get("audit_eval_only")), reused_retained=False,
                 finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(res))),
                 _proto=tuple(cfg.get(k) for k in ("lr", "train_edges_cap", "time_window", "temporal_d_model", "recurrency_decoder",
                                                    "recurrency_untyped", "recurrency_symmetric", "relation_in_input", "node_type_emb",
                                                    "epochs", "patience", "min_epochs", "track_val_edges")))
        d.pop("config_json", None)
        if cfg.get("audit_eval_only"):
            # parity with the retained metric of the replayed run (P0 gate)
            old = f"results/event_bench/leakfree2/{run}/results.csv"
            if os.path.exists(old):
                o = pd.read_csv(old).iloc[0]
                d["retained_test_mrr"] = float(o.get("test_mrr", np.nan))
                d["retained_val_mrr"] = float(o.get("validation_mrr", np.nan))
                d["replay_minus_retained_test"] = float(d.get("test_mrr", np.nan)) - d["retained_test_mrr"]
        rows.append(d)
    # retained runs (results/event_bench/leakfree2) reused under the provenance rule: checkpoint replays reproduce
    # the stored test MRR (audit/), the protocol is identical (matched on the flags below against the NEW runs of
    # the same dataset group), code equivalence pinned by tests.  wiki is excluded (the review re-tuned its lr).
    GROUP = {"thgl-forum": "forum", "tkgl-smallpedia": "sp", "thgl-software": "sw", "tkgl-polecat": "polecat",
             "tkgl-wikidata": "wd", "tkgl-icews": "icews"}
    KEYS = ("lr", "train_edges_cap", "time_window", "temporal_d_model", "recurrency_decoder", "recurrency_untyped",
            "recurrency_symmetric", "relation_in_input", "node_type_emb", "epochs", "patience", "min_epochs", "track_val_edges")
    def proto(cfg):
        return tuple(cfg.get(k) for k in KEYS)
    new_protos = {}
    for d in rows:
        if d.get("is_audit") or d.get("group") not in GROUP.values():
            continue
        new_protos.setdefault(d["group"], set()).add(d.get("_proto"))
    for res in sorted(glob.glob("results/event_bench/leakfree2/*/results.csv")):
        d = pd.read_csv(res).iloc[0].to_dict()
        try:
            cfg = json.loads(d.get("config_json", "{}"))
        except Exception:
            cfg = {}
        ds = cfg.get("dataset"); run = res.split("/")[-2]
        if ds not in GROUP or cfg.get("model", "faithful") != "faithful" or pd.isna(d.get("test_mrr", np.nan)) or cfg.get("eval_only") and not run.startswith("icews_eval"):
            continue
        if run.startswith("icews_eval"):   # metrics live in the eval-only run; the budget is the training run's
            tr = f"results/event_bench/leakfree2/icews_f_s{cfg.get('seed')}/results.csv"
            if os.path.exists(tr):
                tcfg = json.loads(pd.read_csv(tr).iloc[0]["config_json"])
                for k in ("epochs", "patience", "min_epochs"):
                    cfg[k] = tcfg.get(k)
        group = GROUP[ds]
        if new_protos.get(group) and proto(cfg) not in new_protos[group]:
            continue   # different budget/protocol than the review runs of this group (e.g. 40-epoch smallpedia)
        d.update(run=run, group=group, arm=arm_of(cfg), lr=cfg.get("lr"), rec="on" if cfg.get("recurrency_decoder") else "off",
                 clock="global", is_audit=False, reused_retained=True,
                 finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(res))))
        d.pop("config_json", None); d.pop("_proto", None)
        rows.append(d)
    for d in rows:
        d.pop("_proto", None)
    for log in sorted(glob.glob(f"{RV}/queue/*.log")):
        n = os.path.basename(log)[:-4]
        if re.search(r"\.fail\d+$", n):
            continue
        cmd = log[:-4] + ".cmd"
        if not os.path.exists(cmd):
            continue
        out = re.search(r"--out (\S+)", open(cmd).read())
        if out and os.path.exists(os.path.join(out.group(1), "results.csv")):
            continue
        txt = open(log, errors="replace").read()
        state = "crashed" if ("Traceback" in txt or "OutOfMemoryError" in txt) else "running/pending"
        ep = re.findall(r"'epoch': (\d+)", txt)
        unfinished.append(dict(run=n, state=state, last_epoch=int(ep[-1]) if ep else 0,
                               attempts=len(glob.glob(log[:-4] + ".fail*.log")) + 1))
    return pd.DataFrame(rows), pd.DataFrame(unfinished)


def paired(df):
    out = []
    d = df[(~df.is_audit) & df.test_mrr.notna() & (df.group != "smoke")]
    d = d.drop_duplicates(subset=["dataset", "rec", "group", "arm", "lr", "seed"], keep="last")
    for (ds, rec, grp), g in d.groupby(["dataset", "rec", "group"]):
        # lock: the lr used by the majority of each arm's seeds (wave 2); wave-1 alternatives stay separate rows
        base = g[g.arm == "tsd"]
        if base.empty:
            continue
        for lr_b, gb in base.groupby("lr"):
            b = gb.set_index("seed").test_mrr
            for arm, ga in g[g.arm != "tsd"].groupby("arm"):
                for lr_a, gaa in ga.groupby("lr"):
                    a = gaa.set_index("seed").test_mrr
                    common = sorted(set(a.index) & set(b.index))
                    if not common:
                        continue
                    delta = np.array([a[s] - b[s] for s in common])   # arm minus TSD
                    n = len(delta); mean = delta.mean(); sd = delta.std(ddof=1) if n > 1 else np.nan
                    from scipy import stats
                    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
                    out.append(dict(dataset=ds, group=grp, rec=rec, tsd_lr=lr_b, arm=arm, arm_lr=lr_a, seeds=common, n=n,
                                    tsd_mean=float(b[common].mean()), arm_mean=float(a[common].mean()),
                                    delta_arm_minus_tsd_mean=float(mean), delta_sd=float(sd) if n > 1 else np.nan,
                                    t95_halfwidth=float(half) if n > 1 else np.nan,
                                    deltas=" / ".join(f"{x:+.4f}" for x in delta), signs=f"{int((delta > 0).sum())}+ {int((delta < 0).sum())}-",
                                    method="paired per-seed differences; mean, sample SD, 95% t-interval (n-1 df); descriptive only"))
    return pd.DataFrame(out)


def main():
    runs, unfinished = collect()
    os.makedirs(RV, exist_ok=True)
    if len(runs):
        keep = ["run", "group", "dataset", "arm", "lr", "rec", "clock", "seed", "is_audit", "validation_mrr", "test_mrr", "test_hits10",
                "best_track_epoch", "train_epochs_run", "params_total", "params_active", "query_audit_affected_total",
                "query_audit_parity", "retained_test_mrr", "replay_minus_retained_test", "reused_retained", "finished"]
        runs[[c for c in keep if c in runs.columns]].to_csv(f"{RV}/per_seed_results.csv", index=False)
        cost_cols = [c for c in runs.columns if c.startswith(("hw_", "params_", "final_", "prep_", "negatives_", "train_sec",
                                                              "selection_", "end_to_end", "rec_", "core_state", "checkpoint_bytes",
                                                              "cpu_rss", "reserved_buffer", "tracked_nodes"))]
        runs[["run", "group", "dataset", "arm", "seed", "lr"] + cost_cols].to_csv(f"{RV}/costs.csv", index=False)
        num_cols = [c for c in runs.columns if c.startswith(("train_steps", "train_nonfinite", "train_clipped", "train_skipped",
                                                             "val_nonfinite", "val_skipped", "val_metric_examples", "test_nonfinite",
                                                             "test_skipped", "test_metric_examples", "query_audit"))]
        runs[["run", "group", "dataset", "arm", "seed"] + num_cols].to_csv(f"{RV}/numerical_events.csv", index=False)
        pc = paired(runs)
        pc.to_csv(f"{RV}/paired_contrasts.csv", index=False)
    else:
        pc = pd.DataFrame()
    qv = []
    for f in sorted(glob.glob(f"{RV}/*/*/query_validity.csv")):
        q = pd.read_csv(f); q["run"] = f.split("/")[-2]; q["group"] = f.split("/")[-3]; qv.append(q)
    qv = pd.concat(qv, ignore_index=True) if qv else pd.DataFrame()
    if len(qv):
        qv.to_csv(f"{RV}/query_validity.csv", index=False)
    # clock diagnostics (checkpoint replays + clock runs) and protocol traces, concatenated at the root
    cd = []
    for f in sorted(glob.glob(f"{RV}/*/*/clock_diagnostics.csv")):
        try:
            c = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if not len(c):
            continue
        c["run"] = f.split("/")[-2]; c["group"] = f.split("/")[-3]
        tj = os.path.join(os.path.dirname(f), "clock_learned_timing.json")
        if os.path.exists(tj):
            t = json.load(open(tj)); c["clock"] = t.get("clock"); c["delta_scale"] = t.get("delta_scale")
            for k, v in t.get("dt_time_params", {}).items():
                c[k] = v
        cd.append(c)
    if cd:
        pd.concat(cd, ignore_index=True).to_csv(f"{RV}/clock_diagnostics.csv", index=False)
    tr = []
    for f in sorted(glob.glob(f"{RV}/provenance/trace_*/protocol_trace.csv")):
        t = pd.read_csv(f); t["trace"] = f.split("/")[-2]; tr.append(t)
    if tr:
        pd.concat(tr, ignore_index=True).to_csv(f"{RV}/protocol_trace.csv", index=False)
        with open(f"{RV}/protocol_verdict.md", "w") as fh:
            for f in sorted(glob.glob(f"{RV}/provenance/trace_*/protocol_verdict.md")):
                fh.write(open(f).read() + "\n\n---\n\n")
    lines = [f"# Review campaign summary (auto-generated {time.strftime('%Y-%m-%d %H:%M')})", "",
             f"{len(runs)} finished runs under `{RV}`; arms derived from resolved configs.", ""]
    if len(runs):
        g = runs[~runs.is_audit].groupby(["dataset", "group", "rec", "arm", "lr"]).test_mrr.agg(
            n="count", mean="mean", std="std", values=lambda v: " / ".join(f"{x:.4f}" for x in v)).reset_index()
        lines += ["## Test MRR by dataset x arm x lr", "", g.to_markdown(index=False), ""]
        if len(pc):
            lines += ["## Paired contrasts (arm minus TSD, per-seed)", "",
                      pc[["dataset", "rec", "arm", "arm_lr", "tsd_lr", "n", "tsd_mean", "arm_mean", "delta_arm_minus_tsd_mean", "delta_sd", "t95_halfwidth", "deltas", "signs"]].to_markdown(index=False), ""]
        aud = runs[runs.is_audit]
        if len(aud):
            cols = [c for c in ["run", "dataset", "seed", "validation_mrr", "test_mrr", "retained_test_mrr", "replay_minus_retained_test",
                                "query_audit_affected_total", "query_audit_parity"] if c in aud.columns]
            lines += ["## Checkpoint-replay audits (P0)", "", aud[cols].to_markdown(index=False), ""]
    if len(qv):
        lines += ["## Score-validity audit (per split)", "",
                  qv[["group", "run", "split", "queries", "queries_affected", "pos_nonfinite", "neg_nan", "neg_posinf", "neg_neginf",
                      "mrr_tgb_raw", "mrr_guarded", "mrr_conservative", "state_nonfinite_snapshots", "rec_nonfinite_snapshots"]].to_markdown(index=False), ""]
    if cd:
        cdf = pd.concat(cd, ignore_index=True)
        cols = [c for c in ["group", "run", "clock", "split", "activity", "n", "frac_capped", "frac_zero_gap_used", "n_first_update",
                            "n_first_interaction", "mean_dt", "mean_dt_uncapped", "mean_log10_gap_used_pos", "delta_scale"] if c in cdf.columns]
        lines += ["## Clock / saturation diagnostics (per run x split x activity class)", "", cdf[cols].to_markdown(index=False), ""]
    if len(unfinished):
        lines += ["## Not finished", "", unfinished.to_markdown(index=False), ""]
    open(f"{RV}/REVIEW_SUMMARY.md", "w").write("\n".join(lines) + "\n")
    import hashlib
    fp = int(hashlib.md5("|".join(sorted(runs.run)).encode()).hexdigest()[:8], 16) if len(runs) else 0
    print(f"collected {len(runs)} finished runs, {len(unfinished)} unfinished; fingerprint {fp}")


if __name__ == "__main__":
    main()
