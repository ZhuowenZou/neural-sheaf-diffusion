"""Analysis: tune alpha per (cell, scenario, method) on held-out draws (criterion: whole-window error),
then paired differences on the evaluation draws with bootstrap 95% CIs and sign counts, per segment.

  python -m exp.histgeom.analyze --out results/histgeom_2026_09_24
"""
import argparse
import os

import numpy as np
import pandas as pd

PAIRS = [("hist", "cur"), ("hist", "in"), ("in", "cur"), ("hist", "legs"), ("cur", "legs")]
LABEL = {"hist": "H_hist", "cur": "H_cur", "in": "H_in", "legs": "LegS (alpha=0)", "point": "pointwise filter (no compression)"}


def boot_ci(d, rng, B=4000):
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.out, "raw_errors.csv.gz"))
    tune = df[(df.split == "tune") & (df.segment == "all")]
    tm = tune.groupby(["cell", "scenario", "method", "alpha"]).rel_err.mean().reset_index()
    best = tm.loc[tm.groupby(["cell", "scenario", "method"]).rel_err.idxmin(), ["cell", "scenario", "method", "alpha"]]
    best.to_csv(os.path.join(a.out, "tuned_alpha.csv"), index=False)
    ev = df[df.split == "eval"].merge(best, on=["cell", "scenario", "method", "alpha"])
    # alpha-sweep curves (eval draws, every alpha) for the appendix
    df[df.split == "eval"].groupby(["cell", "scenario", "method", "alpha", "segment"]).rel_err.agg(["mean", "std", "count"]).reset_index() \
        .to_csv(os.path.join(a.out, "alpha_sweep_curves.csv"), index=False)
    summ = ev.groupby(["cell", "scenario", "method", "segment"]).rel_err.agg(["mean", "std", "count"]).reset_index()
    summ = summ.merge(best, on=["cell", "scenario", "method"])
    summ.to_csv(os.path.join(a.out, "tuned_errors.csv"), index=False)
    rng = np.random.default_rng(0); rows = []
    piv = ev.pivot_table(index=["cell", "scenario", "segment", "seed"], columns="method", values="rel_err")
    for (cell, sc, seg), g in piv.groupby(level=[0, 1, 2]):
        for m1, m2 in PAIRS:
            d = (g[m1] - g[m2]).to_numpy()
            lo, hi = boot_ci(d, rng)
            rows.append(dict(cell=cell, scenario=sc, segment=seg, pair=f"{m1}-{m2}", n=len(d), mean_diff=d.mean(),
                             ci_lo=lo, ci_hi=hi, n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()),
                             rel_mean_diff=d.mean() / g[m2].mean()))
    pc = pd.DataFrame(rows); pc.to_csv(os.path.join(a.out, "paired_differences.csv"), index=False)
    print(f"tuned {len(best)} (cell,scenario,method) alphas; {len(pc)} paired contrasts")


if __name__ == "__main__":
    main()
