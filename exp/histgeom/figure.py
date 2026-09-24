"""Main-text figure: pre-change error vs rotation magnitude, Scenario A | Scenario B, tuned alpha per method."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"hist": ("H_hist (historical geometry)", "#1f5aa6", "-", "o"),
         "in": ("H_in (filter, then compress)", "#e08a1e", "--", "s"),
         "cur": ("H_cur (current geometry)", "#b8322a", "-", "^"),
         "legs": ("LegS, α = 0", "#7a7a7a", ":", "")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--suffix", default="_snr0")
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(a.out, "raw_errors.csv.gz")); best = pd.read_csv(os.path.join(a.out, "tuned_alpha.csv"))
    ev = ev[(ev.split == "eval") & (ev.segment == "pre")].merge(best, on=["cell", "scenario", "method", "alpha"])
    thetas = [0, 5, 10, 20, 30, 45, 60, 90]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, sc, title in zip(axes, ("A", "B"), ("A: physical re-mount (history in old frame)", "B: calibration correction (history in new frame)")):
        for m, (lab, col, ls, mk) in STYLE.items():
            mu, lo, hi = [], [], []
            for t in thetas:
                x = ev[(ev.cell == f"theta{t}{a.suffix}") & (ev.scenario == sc) & (ev.method == m)].rel_err.to_numpy()
                mu.append(x.mean()); se = x.std(ddof=1) / np.sqrt(len(x)); lo.append(x.mean() - 1.96 * se); hi.append(x.mean() + 1.96 * se)
            ax.plot(thetas, mu, ls, color=col, marker=mk, ms=4, lw=1.6, label=lab)
            ax.fill_between(thetas, lo, hi, color=col, alpha=0.12, lw=0)
        ax.set_title(title, fontsize=9); ax.set_xlabel("rotation of re-mounted sensors (degrees)")
        ax.grid(alpha=0.25); ax.set_xticks(thetas[::1] if False else [0, 15, 30, 45, 60, 75, 90])
    axes[0].set_ylabel("pre-change relative error\n(clean history, tuned α)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_pre_change_vs_rotation{a.suffix}.{ext}"), dpi=200)
    print("figure written")


if __name__ == "__main__":
    main()
