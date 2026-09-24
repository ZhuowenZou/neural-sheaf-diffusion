"""Appendix exactness table: max relative errors between equivalent constructions on random sensor paths."""
import argparse, os
import numpy as np, pandas as pd, scipy.sparse as sp
from dataclasses import replace
from exp.histgeom import core as hc
from exp.histgeom.sweep import Cfg, build


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = []
    for seed, (N, pert) in enumerate([(8, "none"), (16, "none"), (16, "drift"), (16, "many"), (32, "none")]):
        cfg = replace(Cfg(), N=N, T=200, perturb=pert, regime="R3" if pert != "none" else "R2")
        d = build(cfg, 777 + seed)
        edges = d["edges"].copy(); edges[0] = 0.0
        Xs, Ls = list(d["X"]["A"]), d["Ls"]; p = Ls[0].shape[0]; alpha = 3.0
        Qs = [np.eye(p) + alpha * L for L in Ls]
        C, G = hc.gram_recursion(edges, Xs, Ls, N)
        Cb, Gb = hc.batch_G_C(edges, Xs, Ls, N)
        Cp, Kp = hc.gram_recursion_paper(edges, Xs, Qs, N)
        K = alpha * G; K[np.arange(N), np.arange(N)] += np.eye(p)
        Kb = alpha * Gb; Kb[np.arange(N), np.arange(N)] += np.eye(p)
        Mp = hc.moment_recursion_paper(edges, Qs, N)
        Hd = hc.dense_solve(Gb, Cb, alpha)
        Ms = [sp.csr_matrix(np.where(np.abs(m) > 1e-14, m, 0.0)) for m in hc.moment_recursion(edges, Ls, N)]
        Hcg, it = hc.cg_moment_solve(Ms, hc.gamma_coeffs(N), C, alpha)
        rows.append(dict(N=N, path=pert, p=p, T=cfg.T,
                         recursion_vs_batch_C=rel(C, Cb), recursion_vs_batch_K=rel(K, Kb),
                         paper_eq8_vs_batch_C=rel(Cp, Cb), paper_eq8_vs_batch_K=rel(Kp, Kb),
                         moments_eq56_vs_batch_K=rel(hc.G_from_moments(Mp, N), Kb),
                         cg_moments_vs_dense_H=rel(Hcg, Hd), cg_iterations=it,
                         hcur_eq59_vs_resolvent=rel(hc.hcur_recursion_paper(edges, Xs, Qs, N), np.linalg.solve(Qs[-1], Cb)),
                         K_min_eig=float(np.linalg.eigvalsh(hc.big(Kb)).min()), K_max_eig=float(np.linalg.eigvalsh(hc.big(Kb)).max()),
                         bound_1_plus_2alpha=1 + 2 * alpha))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "exactness_table.csv"), index=False)


if __name__ == "__main__":
    main()
