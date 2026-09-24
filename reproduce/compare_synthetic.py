"""Compare regenerated synthetic-study records with the archived ones at the record revision, and re-derive every
number quoted in Section 5 / Appendix F from the regenerated records.  Exit code 1 on any failed check."""
import io, subprocess, sys
import numpy as np, pandas as pd

new_dir, rev = sys.argv[1], sys.argv[2]
ARCH = "results/histgeom_2026_09_24"
def archived(name):
    raw = subprocess.check_output(["git", "show", f"{rev}:{ARCH}/{name}"])
    return pd.read_csv(io.BytesIO(raw), compression="gzip" if name.endswith(".gz") else None)
fails = []; out = ["# Clean-environment reproduction: comparison with archived records", ""]
def check(name, ok, detail):
    out.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: fails.append(name)

# 1. raw per-draw errors
a, n = archived("raw_errors.csv.gz"), pd.read_csv(f"{new_dir}/raw_errors.csv.gz")
key = ["cell", "seed", "split", "scenario", "method", "alpha", "segment"]
m = a.merge(n, on=key, suffixes=("_a", "_n"), how="outer", indicator=True)
d = (m.rel_err_n - m.rel_err_a).abs(); r = d / m.rel_err_a.abs().clip(lower=1e-300)
check("raw per-draw errors: identical row set", (m._merge == "both").all() and len(a) == len(n), f"{len(a)} archived rows, {len(n)} regenerated")
check("raw per-draw errors: values", r.max() < 1e-8, f"max abs diff {d.max():.2e}, max rel diff {r.max():.2e}")
# 2. selected alphas
ta, tn = archived("tuned_alpha.csv"), pd.read_csv(f"{new_dir}/tuned_alpha.csv")
mm = ta.merge(tn, on=["cell", "scenario", "method"], suffixes=("_a", "_n"))
check("selected regularisation (alpha) per cell/scenario/method", len(mm) == len(ta) and (mm.alpha_a == mm.alpha_n).all(),
      f"{int((mm.alpha_a == mm.alpha_n).sum())}/{len(ta)} identical")
# 3. tuned summaries and paired bootstrap contrasts
for f, vals, k in (("tuned_errors.csv", ["mean", "std"], ["cell", "scenario", "method", "segment"]),
                   ("paired_differences.csv", ["mean_diff", "ci_lo", "ci_hi"], ["cell", "scenario", "segment", "pair"])):
    x = archived(f).merge(pd.read_csv(f"{new_dir}/{f}"), on=k, suffixes=("_a", "_n"))
    dd = max(float((x[v + "_n"] - x[v + "_a"]).abs().max()) for v in vals)
    same_signs = True
    if f == "paired_differences.csv":
        same_signs = bool(((x.n_neg_a == x.n_neg_n) & (x.n_pos_a == x.n_pos_n)).all())
    check(f, dd < 1e-8 and same_signs, f"{len(x)} rows, max abs diff {dd:.2e}" + ("" if f != "paired_differences.csv" else f", sign counts identical: {same_signs}"))
# 4. numbers quoted in the paper, re-derived from the REGENERATED records
te = pd.read_csv(f"{new_dir}/tuned_errors.csv"); pc = pd.read_csv(f"{new_dir}/paired_differences.csv")
g = lambda c, s, m_, seg: te[(te.cell == c) & (te.scenario == s) & (te.method == m_) & (te.segment == seg)].iloc[0]
table15 = {("theta45", "A", "pre"): (0.0901, 0.0901, 0.0920, 0.0914), ("theta45", "A", "all"): (0.0854, 0.0854, 0.0861, 0.0865),
           ("theta45", "B", "pre"): (0.0666, 0.0666, 0.0650, 0.0669), ("theta45", "B", "all"): (0.0630, 0.0630, 0.0619, 0.0636),
           ("theta45_snr0", "A", "pre"): (0.1807, 0.1807, 0.2156, 0.2204), ("theta45_snr0", "A", "all"): (0.1718, 0.1717, 0.1939, 0.2089),
           ("theta45_snr0", "B", "pre"): (0.2024, 0.2024, 0.1681, 0.2106), ("theta45_snr0", "B", "all"): (0.1842, 0.1842, 0.1605, 0.2002)}
bad = []
for (c, s, seg), want in table15.items():
    got = tuple(round(float(g(c, s, m_, seg)["mean"]), 4) for m_ in ("hist", "in", "cur", "legs"))
    if got != want: bad.append(f"{c}/{s}/{seg}: paper {want} vs regenerated {got}")
check("Table 15 means (32 entries)", not bad, "all match to 4 decimals" if not bad else "; ".join(bad))
p = pc.set_index(["cell", "scenario", "segment", "pair"])
qa, qb = p.loc[("theta45_snr0", "A", "pre", "hist-cur")], p.loc[("theta45_snr0", "B", "pre", "hist-cur")]
txt = f"A {qa.mean_diff:+.4f} [{qa.ci_lo:+.4f}, {qa.ci_hi:+.4f}], B {qb.mean_diff:+.4f} [{qb.ci_lo:+.4f}, {qb.ci_hi:+.4f}]"
check("Section 5 paired differences (paper: -0.0349 [-0.0393,-0.0305]; +0.0343 [0.0299,0.0386])",
      txt == "A -0.0349 [-0.0393, -0.0305], B +0.0343 [+0.0299, +0.0386]", txt)
x = pc[(pc.pair == "hist-cur") & (pc.segment == "pre") & (~pc.cell.str.startswith("R1")) & (~pc.cell.str.startswith("theta0"))]
ok = x.apply(lambda r: r.ci_hi < 0 if r.scenario == "A" else r.ci_lo > 0, axis=1)
cnt = x.assign(ok=ok).groupby("cell").ok.all()
cnt = cnt.drop(index="theta45_snr0", errors="ignore")      # duplicated 0 dB/45 deg configuration counted once (= snr0)
check("Appendix F.2 crossover count (paper: 28 of 31)", (int(cnt.sum()), len(cnt)) == (28, 31), f"{int(cnt.sum())} of {len(cnt)}")
# 5. exactness bounds quoted in F.3 and CG iteration counts
ex = pd.read_csv(f"{new_dir}/exactness_table.csv")
bounds = {"paper_eq8_vs_batch_C": 2.0e-14, "paper_eq8_vs_batch_K": 2.0e-14, "moments_eq56_vs_batch_K": 3.9e-13,
          "recursion_vs_batch_C": 8.0e-13, "recursion_vs_batch_K": 8.0e-13, "cg_moments_vs_dense_H": 1.1e-12, "hcur_eq59_vs_resolvent": 4.4e-14}
det = [f"{c} {ex[c].max():.1e} (<{b:.1e})" for c, b in bounds.items()]
check("Appendix F.3 numerical-equivalence bounds", all(ex[c].max() < b for c, b in bounds.items()) and
      bool(((ex.K_min_eig > 1 - 1e-9) & (ex.K_max_eig < 7 + 1e-9)).all()), "; ".join(det))
ca, cn = archived("cost_benchmark.csv"), pd.read_csv(f"{new_dir}/cost_benchmark.csv")
check("CG iteration counts (cost benchmark)", (ca.cg_iters.values == cn.cg_iters.values).all(), f"{cn.cg_iters.tolist()}")
out += ["", f"**{'ALL CHECKS PASS' if not fails else str(len(fails)) + ' CHECK(S) FAILED: ' + ', '.join(fails)}**",
        "", "Timing and memory in cost_benchmark.csv are hardware-dependent and are not compared."]
print("\n".join(out)); sys.exit(1 if fails else 0)
