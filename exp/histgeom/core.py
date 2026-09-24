"""Historical-geometry polynomial memory: exact constructions and a sensor-array generator.

Definitions (reconstructed from the paper's worked example, pinned by unit tests):
  basis     pi_i(s) = sqrt(2i+1) P_i(2s/t - 1), orthonormal w.r.t. mu_t = ds/t on [0, t]
  objective J_t(H) = int ||X - H pi||^2 dmu_t + alpha int (H pi)^T Ltil(s) (H pi) dmu_t
  normal eq K H = C,  K = I + alpha G,  G = int Ltil(s) (x) pi pi^T dmu_t,  C = int X pi^T dmu_t
  H_hist = K^{-1} C (historical geometry); H_cur = Q(t)^{-1} C with Q = I + alpha Ltil(t) (current geometry)
  H_in   = int Q(s)^{-1} X(s) pi^T dmu_t (filter at observation time, then compress); LegS = C (alpha = 0)
X is held piecewise constant on the observation intervals; Ltil is the sheaf Laplacian of
the prescribed restriction maps R_{e<-u} = R_u (scaled by 1/max degree, spectrum in [0,2]), held on the same intervals.  All arrays float64.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

# ----------------------------------------------------------------------------- basis / quadrature
def basis(N, s, t):
    x = 2.0 * np.asarray(s, dtype=np.float64) / t - 1.0
    return legvander(x, N - 1) * np.sqrt(2.0 * np.arange(N) + 1.0)


def gl(a, b, n):
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def step_mats(N, t, t2):
    """A_ia = (1/t) int_0^t pi^{t2}_i pi^t_a ds ;  e_i = (1/t2) int_t^t2 pi^{t2}_i ;  B = (1/t2) int_t^t2 pi pi^T.
    Gauss-Legendre with N+1 nodes is exact for these degree <= 2N-2 integrands."""
    s, w = gl(0.0, t, N + 1)
    A = (basis(N, s, t2).T * w) @ basis(N, s, t) / t
    s2, w2 = gl(t, t2, N + 1)
    P2 = basis(N, s2, t2)
    return A, P2.T @ w2 / t2, (P2.T * w2) @ P2 / t2


def gamma_coeffs(N):
    """gamma_ijr = int pi_i pi_j P_r dmu (orthonormal P_r, r <= 2N-2); 2N nodes exact for degree 3N-3."""
    s, w = gl(0.0, 1.0, 2 * N)
    P, R = basis(N, s, 1.0), basis(2 * N - 1, s, 1.0)
    return np.einsum("q,qi,qj,qr->ijr", w, P, P, R)


# ----------------------------------------------------------------------------- recursions
def gram_recursion(edges, Xs, Ls, N, return_trace=False):
    """Online Gram recursion over held intervals (edges[k-1], edges[k]] with inputs Xs[k] (p,), Ls[k] (p,p).
    Start: C = X0 e0^T, G = I_N (x) L0 over the first interval.  Update (t -> t2):
      C <- (t/t2) C A^T + X e^T
      G <- (t/t2) A G A^T + (I - (t/t2) A A^T) (x) L     [the paper's K <- J K J^T + (I - J J^T) (x) Q, J = sqrt(t/t2) A]"""
    p = Xs[0].shape[0]
    C = np.zeros((p, N)); C[:, 0] = Xs[0]
    G = np.zeros((N, N, p, p)); G[np.arange(N), np.arange(N)] = Ls[0]
    t = edges[1]
    for k in range(1, len(Xs)):
        t2 = edges[k + 1]
        A, e, _ = step_mats(N, t, t2)
        r = t / t2
        C = r * C @ A.T + np.outer(Xs[k], e)
        G = r * np.einsum("ia,abxy,jb->ijxy", A, G, A, optimize=True) + (np.eye(N) - r * A @ A.T)[:, :, None, None] * Ls[k]
        t = t2
    return C, G


def moment_recursion(edges, Ls, N):
    """Operator moments M_r = int Ltil(s) P_r(s) dmu_t, r = 0..2N-2 (Prop. B.7 form): M <- (t/t2) A_M M + v (x) L."""
    R = 2 * N - 1
    p = Ls[0].shape[0]
    M = np.zeros((R, p, p)); M[0] = Ls[0]
    t = edges[1]
    for k in range(1, len(Ls)):
        t2 = edges[k + 1]
        A, v, _ = step_mats(R, t, t2)
        M = (t / t2) * np.einsum("rj,jxy->rxy", A, M) + v[:, None, None] * Ls[k]
        t = t2
    return M


def batch_G_C(edges, Xs, Ls, N):
    """Direct batch construction by per-interval Gauss-Legendre quadrature (exact for held inputs)."""
    t = edges[-1]; p = Xs[0].shape[0]
    C = np.zeros((p, N)); G = np.zeros((N, N, p, p))
    for k in range(len(Xs)):
        s, w = gl(edges[k], edges[k + 1], N + 1)
        P = basis(N, s, t)
        C += np.outer(Xs[k], P.T @ w / t)
        G += ((P.T * w) @ P / t)[:, :, None, None] * Ls[k]
    return C, G


def G_from_moments(M, N, gam=None):
    gam = gamma_coeffs(N) if gam is None else gam
    return np.einsum("ijr,rxy->ijxy", gam, M)


# ----------------------------------------------------------------------------- solvers
def big(G):
    N, _, p, _ = G.shape
    return G.transpose(0, 2, 1, 3).reshape(N * p, N * p)


class HistSolver:
    """All-alpha solver for H_hist = (I + alpha G)^{-1} C via one eigendecomposition of G."""
    def __init__(self, G):
        self.N, _, self.p, _ = G.shape
        B = big(G)
        B = 0.5 * (B + B.T)          # exact symmetry (the recursion leaves ~1e-16 asymmetry)
        # G is highly degenerate (consistent readings span a large kernel).  numpy's eigh (LAPACK *syevd) fails to
        # converge on a few such matrices under OpenBLAS (7 of 2,160 sweep draws in a clean pip environment);
        # the MRRR driver (*syevr) is robust.  Fall back to numpy only if SciPy's driver itself fails.
        try:
            from scipy.linalg import eigh as _eigh
            self.lam, self.V = _eigh(B, driver="evr")
        except Exception:
            self.lam, self.V = np.linalg.eigh(B)

    def solve(self, C, alpha):
        c = C.T.reshape(-1)
        h = self.V @ ((self.V.T @ c) / (1.0 + alpha * self.lam))
        return h.reshape(self.N, self.p).T


def dense_solve(G, C, alpha):
    N, _, p, _ = G.shape
    h = np.linalg.solve(np.eye(N * p) + alpha * big(G), C.T.reshape(-1))
    return h.reshape(N, p).T


def cg_moment_solve(M_sparse, gam, C, alpha, tol=1e-12, maxit=500):
    """Conjugate gradient on K H = C with the matvec (K H)_i = H_i + alpha sum_j sum_r gamma_ijr M_r H_j,
    using only the sparse operator moments (never forms K).  cond(K) <= 1 + 2 alpha."""
    def mv(H):
        Y = np.stack([Mr @ H for Mr in M_sparse])          # (R, p, N)
        return H + alpha * np.einsum("ijr,rxj->xi", gam, Y, optimize=True)
    H = C.copy(); r = C - mv(H); z = r.copy(); rs = np.sum(r * r); it = 0
    nb = np.sqrt(np.sum(C * C))
    while np.sqrt(rs) > tol * nb and it < maxit:
        Az = mv(z); a = rs / np.sum(z * Az)
        H += a * z; r -= a * Az; rs2 = np.sum(r * r); z = r + (rs2 / rs) * z; rs = rs2; it += 1
    return H, it


def filtered_compress(edges, Xs, Ls, N, alphas, L_cache=None):
    """H_in(alpha) = sum_k (I + alpha L_k)^{-1} X_k (x) (1/t) int_{I_k} pi, for all alphas (eigh per distinct L)."""
    t = edges[-1]; p = Xs[0].shape[0]
    out = np.zeros((len(alphas), p, N))
    eig = {}
    for k in range(len(Xs)):
        key = id(Ls[k])
        if key not in eig:
            eig[key] = np.linalg.eigh(Ls[k])
        lam, V = eig[key]
        s, w = gl(edges[k], edges[k + 1], N + 1)
        e = basis(N, s, t).T @ w / t
        z = V.T @ Xs[k]
        for ai, a in enumerate(alphas):
            out[ai] += np.outer(V @ (z / (1.0 + a * lam)), e)
    return out


# ----------------------------------------------------------------------------- sheaf / generator
def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def rot3(axis, th):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def knn_edges(pos, k):
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1); np.fill_diagonal(D, np.inf)
    E = set()
    for u in range(len(pos)):
        for v in np.argsort(D[u])[:k]:
            E.add((min(u, v), max(u, v)))
    return sorted(E)


def sheaf_laplacian(Rs, edges_uv):
    """Sheaf Laplacian with R_{e<-u} = R_u:  L_uu = deg I, L_uv = -R_u^T R_v, scaled by 1 / max degree."""
    n, d = len(Rs), Rs[0].shape[0]
    L = np.zeros((n * d, n * d)); deg = np.zeros(n)
    for u, v in edges_uv:
        deg[u] += 1; deg[v] += 1
        L[u*d:(u+1)*d, v*d:(v+1)*d] -= Rs[u].T @ Rs[v]
        L[v*d:(v+1)*d, u*d:(u+1)*d] -= Rs[v].T @ Rs[u]
    for u in range(n):
        L[u*d:(u+1)*d, u*d:(u+1)*d] += deg[u] * np.eye(d)
    # Scale by the maximum degree (not D^{-1/2} L D^{-1/2}): keeps the global sections (physically consistent
    # readings) exactly in the kernel while bounding the spectrum by 2, so I <= K <= (1 + 2 alpha) I still holds.
    return L / max(deg.max(), 1.0)


# ----------------------------------------------------------------------------- paper-literal forms (App. B)
def legs_AB(N):
    """Eq. (5): (A_H)_ij = -sqrt((2i+1)(2j+1)) (i>j), -(i+1) (i=j), 0 (i<j); (b_H)_i = sqrt(2i+1)."""
    i = np.arange(N)
    A = -np.sqrt(np.outer(2 * i + 1, 2 * i + 1)); A = np.tril(A, -1) - np.diag(i + 1.0)
    return A, np.sqrt(2 * i + 1.0)


def _legs_step(N, eta):
    from scipy.linalg import expm
    A, b = legs_AB(N)
    E = expm(eta * A); J = expm(eta * (A + 0.5 * np.eye(N)))
    v = np.linalg.solve(A, (E - np.eye(N)) @ b)
    return E, J, v


def gram_recursion_paper(edges, Xs, Qs, N):
    """Prop. B.6 / Eq. (8), verbatim: C <- C E^T + X v^T ;  K <- (J (x) I) K (J^T (x) I) + (I - J J^T) (x) Q,
    with history-consistent start C = X0 e0^T, K = I_N (x) Q0 on (0, t0].  K stored as (N, N, p, p)."""
    p = Xs[0].shape[0]
    C = np.zeros((p, N)); C[:, 0] = Xs[0]
    K = np.zeros((N, N, p, p)); K[np.arange(N), np.arange(N)] = Qs[0]
    for k in range(1, len(Xs)):
        E, J, v = _legs_step(N, np.log(edges[k + 1] / edges[k]))
        C = C @ E.T + np.outer(Xs[k], v)
        K = np.einsum("ia,abxy,jb->ijxy", J, K, J, optimize=True) + (np.eye(N) - J @ J.T)[:, :, None, None] * Qs[k]
    return C, K


def moment_recursion_paper(edges, Qs, N):
    """Prop. B.7 / Eq. (56): M_r = t^-1 int Q(s) pi_r ds (r < 2N-1), M <- E^(M) M + v^(M) (x) Q; start M_0 = Q0."""
    R = 2 * N - 1; p = Qs[0].shape[0]
    M = np.zeros((R, p, p)); M[0] = Qs[0]
    for k in range(1, len(Qs)):
        E, _, v = _legs_step(R, np.log(edges[k + 1] / edges[k]))
        M = np.einsum("rj,jxy->rxy", E, M) + v[:, None, None] * Qs[k]
    return M


def hcur_recursion_paper(edges, Xs, Qs, N):
    """Prop. B.8 / Eq. (59): H <- Q_{k+1}^{-1} { Q_k H E^T + X_k v^T } (current geometry with jump transfer)."""
    p = Xs[0].shape[0]
    C0 = np.zeros((p, N)); C0[:, 0] = Xs[0]
    H = np.linalg.solve(Qs[0], C0)
    for k in range(1, len(Xs)):
        E, _, v = _legs_step(N, np.log(edges[k + 1] / edges[k]))
        H = np.linalg.solve(Qs[k], Qs[k - 1] @ H @ E.T + np.outer(Xs[k], v))
    return H


def graph_laplacian_sym(n, edges_uv):
    """Symmetric normalised graph Laplacian without self-loops (the paper's ordinary-graph example)."""
    A = np.zeros((n, n))
    for u, v in edges_uv:
        A[u, v] = A[v, u] = 1.0
    d = A.sum(1); s = 1 / np.sqrt(d)
    return np.eye(n) - s[:, None] * A * s[None, :]
