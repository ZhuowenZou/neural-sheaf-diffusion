import numpy as np
import scipy.sparse as sp
from exp.histgeom import core as hc


def _random_path(rng, n=6, d=2, T=40, N=6, switch=20):
    pos = rng.random((n, 2)); E = hc.knn_edges(pos, 3)
    R0 = [hc.rot2(a) for a in rng.uniform(0, 2 * np.pi, n)]
    R1 = [R @ hc.rot2(0.7) if u % 2 else R for u, R in enumerate(R0)]
    L0, L1 = hc.sheaf_laplacian(R0, E), hc.sheaf_laplacian(R1, E)
    edges = np.sort(np.concatenate([[0.0], rng.uniform(0, 1, T - 1), [1.0]]))   # irregular held intervals
    Ls = [L0 if k < switch else L1 for k in range(T)]
    Xs = [rng.normal(size=n * d) for _ in range(T)]
    return edges, Xs, Ls, N


def test_paper_example_hist_and_cur():
    """Theory-audit example: scalar mode, L(s) = s on [0,1], X = 1, alpha = 1, N = 2
    -> K = [[3/2, sqrt3/6],[sqrt3/6, 3/2]], C = (1, 0), H_hist = (9/13, -sqrt3/13), H_cur = (1/2, 0)."""
    s, w = hc.gl(0, 1, 8); P = hc.basis(2, s, 1.0)
    G = ((P.T * (w * s)) @ P)[:, :, None, None]
    C = (P.T @ w)[None, :]
    assert np.allclose(np.eye(2) + G[:, :, 0, 0], [[1.5, np.sqrt(3) / 6], [np.sqrt(3) / 6, 1.5]], atol=1e-14)
    assert np.allclose(hc.dense_solve(G, C, 1.0), [[9 / 13, -np.sqrt(3) / 13]], atol=1e-14)
    assert np.allclose(C / (1 + 1.0 * 1.0), [[0.5, 0.0]], atol=1e-14)            # Q(1) = 1 + alpha * L(1)


def test_paper_example_via_recursion_converges():
    """Same example through the online recursion with L held on T intervals: O(1/T) convergence to 9/13."""
    errs = []
    for T in (50, 400, 3200):
        edges = np.linspace(0, 1, T + 1)
        Ls = [np.array([[0.5 * (edges[k] + edges[k + 1])]]) for k in range(T)]
        C, G = hc.gram_recursion(edges, [np.ones(1)] * T, Ls, 2)
        errs.append(abs(hc.dense_solve(G, C, 1.0)[0, 0] - 9 / 13))
    assert errs[2] < errs[1] < errs[0] and errs[2] < 1e-6


def test_recursion_equals_batch_and_moments():
    rng = np.random.default_rng(0)
    edges, Xs, Ls, N = _random_path(rng)
    C, G = hc.gram_recursion(edges, Xs, Ls, N)
    Cb, Gb = hc.batch_G_C(edges, Xs, Ls, N)
    assert np.max(np.abs(C - Cb)) / np.max(np.abs(Cb)) < 1e-11
    assert np.max(np.abs(G - Gb)) / np.max(np.abs(Gb)) < 1e-11
    Gm = hc.G_from_moments(hc.moment_recursion(edges, Ls, N), N)
    assert np.max(np.abs(Gm - Gb)) / np.max(np.abs(Gb)) < 1e-11


def test_step_identity_B_equals_I_minus_JJT():
    A, e, B = hc.step_mats(7, 0.3, 0.45)
    assert np.allclose(B, np.eye(7) - (0.3 / 0.45) * A @ A.T, atol=1e-13)


def test_solvers_agree_and_cg_iterations_bounded():
    rng = np.random.default_rng(1)
    edges, Xs, Ls, N = _random_path(rng)
    C, G = hc.gram_recursion(edges, Xs, Ls, N)
    M = hc.moment_recursion(edges, Ls, N)
    Ms = [sp.csr_matrix(np.where(np.abs(m) > 1e-15, m, 0.0)) for m in M]
    for a in (0.1, 3.0, 30.0):
        Hd = hc.dense_solve(G, C, a)
        He = hc.HistSolver(G).solve(C, a)
        Hc, it = hc.cg_moment_solve(Ms, hc.gamma_coeffs(N), C, a)
        assert np.allclose(Hd, He, atol=1e-10) and np.allclose(Hd, Hc, atol=1e-9)
    lam = np.linalg.eigvalsh(hc.big(G))
    assert lam.min() > -1e-12 and lam.max() < 2 + 1e-12                        # I <= K <= (1+2 alpha) I


def test_constant_geometry_all_constructions_coincide():
    rng = np.random.default_rng(2)
    edges, Xs, Ls, N = _random_path(rng, switch=10 ** 6)
    C, G = hc.gram_recursion(edges, Xs, Ls, N)
    a = 2.0
    Hh = hc.dense_solve(G, C, a)
    Hc = np.linalg.solve(np.eye(len(Xs[0])) + a * Ls[-1], C)
    Hi = hc.filtered_compress(edges, Xs, Ls, N, [a])[0]
    assert np.allclose(Hh, Hc, atol=1e-11) and np.allclose(Hh, Hi, atol=1e-11)


def test_global_sections_are_in_the_kernel():
    rng = np.random.default_rng(3)
    n = 8; R = [hc.rot2(a) for a in rng.uniform(0, 6.3, n)]
    L = hc.sheaf_laplacian(R, hc.knn_edges(rng.random((n, 2)), 3))
    g = rng.normal(size=2)
    x = np.concatenate([Ru.T @ g for Ru in R])
    assert np.linalg.norm(L @ x) < 1e-12 * np.linalg.norm(x)


def test_paper_three_node_example():
    """Eq. (11): path 1-2-3 on [0,1/2), triangle on [1/2,1], sym-normalised L, alpha=1, N=1, X = v
    -> H_hist(1) = 4/9 v, H_cur(1) = 2/5 v, H_in(1) = 9/20 v."""
    v = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    Lp = hc.graph_laplacian_sym(3, [(0, 1), (1, 2)]); Lt = hc.graph_laplacian_sym(3, [(0, 1), (1, 2), (0, 2)])
    edges = np.array([0.0, 0.5, 1.0]); Ls = [Lp, Lt]; Xs = [v, v]
    C, G = hc.gram_recursion(edges, Xs, Ls, 1)
    assert np.allclose(hc.dense_solve(G, C, 1.0)[:, 0], 4 / 9 * v, atol=1e-14)
    assert np.allclose(np.linalg.solve(np.eye(3) + Lt, C)[:, 0], 2 / 5 * v, atol=1e-14)
    assert np.allclose(hc.filtered_compress(edges, Xs, Ls, 1, [1.0])[0][:, 0], 9 / 20 * v, atol=1e-14)
    Cp, Kp = hc.gram_recursion_paper(edges, Xs, [np.eye(3) + L for L in Ls], 1)
    assert np.allclose(hc.dense_solve(Kp - np.eye(1)[:, :, None, None] * np.eye(3), Cp, 1.0)[:, 0], 4 / 9 * v, atol=1e-14)
    assert np.allclose(hc.hcur_recursion_paper(edges, Xs, [np.eye(3) + L for L in Ls], 1)[:, 0], 2 / 5 * v, atol=1e-14)


def test_paper_literal_recursions_match():
    """Eq. (8) (expm log-time), Eq. (56) and Eq. (59) reproduce the quadrature forms on an irregular path."""
    rng = np.random.default_rng(4)
    edges, Xs, Ls, N = _random_path(rng)
    edges = np.concatenate([[0.0], np.sort(rng.uniform(0.01, 1, len(Xs) - 1)), [1.0]])   # t_k > 0 for k >= 1
    a = 1.7; Qs = [np.eye(len(Xs[0])) + a * L for L in Ls]
    C, G = hc.gram_recursion(edges, Xs, Ls, N)
    Cp, Kp = hc.gram_recursion_paper(edges, Xs, Qs, N)
    K = a * G; K[np.arange(N), np.arange(N)] += np.eye(len(Xs[0]))
    assert np.max(np.abs(Cp - C)) < 1e-10 and np.max(np.abs(Kp - K)) < 1e-10
    Mp = hc.moment_recursion_paper(edges, Qs, N)
    assert np.max(np.abs(hc.G_from_moments(Mp, N) - K)) < 1e-10
    assert np.max(np.abs(hc.hcur_recursion_paper(edges, Xs, Qs, N) - np.linalg.solve(Qs[-1], C))) < 1e-10


def test_hist_solver_on_degenerate_path_that_broke_openblas_syevd():
    """Regression: a delayed-switch draw whose Gram matrix made numpy.linalg.eigh fail to converge under OpenBLAS."""
    from dataclasses import replace
    from exp.histgeom.sweep import Cfg, build
    d = build(replace(Cfg(), regime="R3", perturb="drift"), 24)
    p = d["Ls"][0].shape[0]; Xs = list(d["X"]["A"])
    C, G = hc.gram_recursion(d["edges"], Xs, d["Ls"], 16)
    S = hc.HistSolver(G)
    for a in (0.1, 3.0, 100.0):
        assert np.allclose(S.solve(C, a), hc.dense_solve(G, C, a), atol=1e-9)
