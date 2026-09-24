"""Cost benchmark: dense Gram state + dense solve vs operator moments + CG (sparse M_r) vs H_cur / H_in,
as p (= n d_s) and N grow.  Single-threaded BLAS; T held intervals; alpha = 1 and 10.

  OMP_NUM_THREADS=1 python -m exp.histgeom.cost --out results/histgeom_2026_09_24
"""
import argparse
import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import scipy.sparse as sp

from exp.histgeom import core as hc


def path(n, d, T, seed=0):
    rng = np.random.default_rng(seed)
    E = hc.knn_edges(rng.random((n, 2)), 4)
    R0 = [hc.rot2(a) if d == 2 else hc.rot3(rng.normal(size=3), rng.uniform(0, 3)) for a in rng.uniform(0, 6.3, n)]
    R1 = [(hc.rot2(0.8) if d == 2 else hc.rot3(rng.normal(size=3), 0.8)) @ R if u % 2 else R for u, R in enumerate(R0)]
    L0, L1 = hc.sheaf_laplacian(R0, E), hc.sheaf_laplacian(R1, E)
    edges = np.linspace(0, 1, T + 1)
    Ls = [L0 if k < T // 2 else L1 for k in range(T)]
    Xs = [rng.normal(size=n * d) for _ in range(T)]
    return edges, Xs, Ls


def timed(f):
    tracemalloc.start(); t = time.perf_counter(); out = f(); dt = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return out, dt, peak / 2 ** 20


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--T", type=int, default=100)
    a = ap.parse_args(); rows = []
    for n, d in ((20, 2), (40, 2), (50, 3)):
        for N in (8, 16, 32):
            p = n * d
            edges, Xs, Ls = path(n, d, a.T)
            (C, G), t_gram, m_gram = timed(lambda: hc.gram_recursion(edges, Xs, Ls, N))
            M, t_mom, m_mom = timed(lambda: hc.moment_recursion(edges, Ls, N))
            gam = hc.gamma_coeffs(N)
            Ms = [sp.csr_matrix(np.where(np.abs(m) > 1e-14, m, 0.0)) for m in M]
            for alpha in (1.0, 10.0):
                Hd, t_dense, m_dense = timed(lambda: hc.dense_solve(G, C, alpha))
                (Hc, it), t_cg, m_cg = timed(lambda: hc.cg_moment_solve(Ms, gam, C, alpha))
                _, t_cur, _ = timed(lambda: np.linalg.solve(np.eye(p) + alpha * Ls[-1], C))
                _, t_in, _ = timed(lambda: hc.filtered_compress(edges, Xs, Ls, N, [alpha]))
                rows.append(dict(p=p, n=n, d=d, N=N, T=a.T, alpha=alpha,
                                 gram_state_MiB=G.nbytes / 2 ** 20, moment_state_MiB=M.nbytes / 2 ** 20,
                                 moment_state_nnz=int(sum(m.nnz for m in Ms)),
                                 t_gram_recursion_s=t_gram, t_moment_recursion_s=t_mom,
                                 t_dense_solve_s=t_dense, peak_dense_solve_MiB=m_dense,
                                 t_cg_solve_s=t_cg, peak_cg_solve_MiB=m_cg, cg_iters=it,
                                 cg_vs_dense_max_abs_diff=float(np.max(np.abs(Hc - Hd))),
                                 t_cur_s=t_cur, t_in_s=t_in))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cost_benchmark.csv"), index=False)


if __name__ == "__main__":
    main()
