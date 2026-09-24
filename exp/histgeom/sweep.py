"""Sensor array with changing mounts: Scenario A (physical re-mount) vs B (calibration correction).

Both scenarios receive the SAME prescribed geometry path (old maps before the change, new after); only the
ground truth differs.  Methods are scored on recovery of the clean history y (never on the objective).

  python -m exp.histgeom.sweep --out results/histgeom_2026_09_24 --procs 96
"""
import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from exp.histgeom import core as hc

ALPHAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
METHODS = ["hist", "cur", "in", "legs", "point"]


@dataclass(frozen=True)
class Cfg:
    name: str = "base"
    regime: str = "R2"          # R1 well-specified prior, R2 sensor array (main), R3 = R2 + perturbation
    perturb: str = "none"       # none | noisy_maps | tau_shift | drift | many
    n: int = 20
    d: int = 2
    k: int = 4
    N: int = 16
    T: int = 400
    frac: float = 0.5
    theta: float = 45.0
    tau: float = 0.5
    snr_db: float = 10.0
    dev: float = 0.2
    n_sin: int = 5
    alpha0: float = 1.0          # Regime 1: strength of the (correct) historical prior
    fmax: float = 4.0            # calibrated: noise-free LegS error ~6% at N=16 (target 5-10%, fixed before comparing methods)


# ----------------------------------------------------------------------------- generator
def _rot(d, rng, th):
    if d == 2:
        return hc.rot2(th * rng.choice([-1.0, 1.0]))
    return hc.rot3(rng.normal(size=3), th)


def _rand_rot(d, rng):
    if d == 2:
        return hc.rot2(rng.uniform(0, 2 * np.pi))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _sines(rng, dims, n_sin, fmax, scale):
    f = rng.uniform(0.5, fmax, (dims, n_sin)); ph = rng.uniform(0, 2 * np.pi, (dims, n_sin))
    a = rng.normal(size=(dims, n_sin)) * scale / np.sqrt(n_sin)
    return lambda s: np.einsum("dj,sdj->sd", a, np.sin(2 * np.pi * f[None] * np.asarray(s)[:, None, None] + ph[None]))


def build(cfg, seed):
    """Returns the prescribed path (edges, Ls), the scenario truths on a fine grid, and observations."""
    rng = np.random.default_rng(seed)
    n, d, T = cfg.n, cfg.d, cfg.T
    E = hc.knn_edges(rng.random((n, 2)), cfg.k)
    R0 = [_rand_rot(d, rng) for _ in range(n)]
    S = rng.choice(n, max(1, int(round(cfg.frac * n))), replace=False)
    th = np.deg2rad(cfg.theta)
    tau, w_drift, m_events = cfg.tau, 0.3, 5
    if cfg.perturb == "many":
        ev = np.sort(rng.uniform(0.1, 0.9, m_events))
        steps = {u: [_rot(d, rng, th / m_events) for _ in range(m_events)] for u in S}
    else:
        D = {u: _rot(d, rng, th) for u in S}

    def orient(s):
        """Physical (scenario A) orientation of every sensor at time s."""
        out = []
        for u in range(n):
            R = R0[u]
            if u in S:
                if cfg.perturb == "many":
                    for j, e in enumerate(ev):
                        if s >= e:
                            R = steps[u][j] @ R
                elif cfg.perturb == "drift":
                    frac_ = np.clip((s - (tau - w_drift / 2)) / w_drift, 0.0, 1.0)
                    if d == 2:
                        ang = np.arctan2(D[u][1, 0], D[u][0, 0]) * frac_
                        R = hc.rot2(ang) @ R
                    else:
                        from scipy.linalg import expm, logm
                        R = np.real(expm(frac_ * logm(D[u]))) @ R
                elif s >= tau:
                    R = D[u] @ R
            out.append(R)
        return out

    final = orient(1.0)
    change_end = {"many": ev[-1] if cfg.perturb == "many" else None, "drift": tau + w_drift / 2}.get(cfg.perturb) or tau
    edges = np.linspace(0.0, 1.0, T + 1); mids = 0.5 * (edges[1:] + edges[:-1])
    # prescribed path (what the methods receive): the believed orientation at each observation time
    noise_rot = {}
    if cfg.perturb == "noisy_maps":
        for u in range(n):
            for epoch in (0, 1):
                noise_rot[(u, epoch)] = _rot(d, rng, np.deg2rad(5.0) * abs(rng.normal()))
    presc_tau = tau + 0.1 if cfg.perturb == "tau_shift" else tau
    Ls, cache = [], {}
    for k, m in enumerate(mids):
        if cfg.perturb == "tau_shift":
            key = ("ts", m >= presc_tau)
            Rp = orient(1.0 if m >= presc_tau else 0.0)
        elif cfg.perturb in ("drift", "many"):
            Rp = orient(m); key = ("t", round(float(m), 9)) if cfg.perturb == "drift" else ("ev", int(np.sum(ev <= m)))
        else:
            ep = int(m >= tau); Rp = orient(m); key = ("ep", ep)
            if cfg.perturb == "noisy_maps":
                Rp = [noise_rot[(u, ep)] @ Rp[u] for u in range(n)]
        if key not in cache:
            cache[key] = hc.sheaf_laplacian(Rp, E)
        Ls.append(cache[key])
    # ground truth and observations (shared noise between A and B)
    F = 2000
    fine = (np.arange(F) + 0.5) / F
    g = _sines(rng, d, cfg.n_sin, cfg.fmax, 1.0)
    dv = _sines(rng, n * d, 3, cfg.fmax, cfg.dev)
    def readings(times, scen):
        G_ = g(times); Dv = dv(times)
        out = np.empty((len(times), n * d))
        for i, s in enumerate(times):
            Rs = orient(s) if scen == "A" else final
            out[i] = np.concatenate([Rs[u].T @ G_[i] for u in range(n)]) + Dv[i]
        return out
    y = {sc: readings(fine, sc) for sc in ("A", "B")}
    x_clean = {sc: readings(mids, sc) for sc in ("A", "B")}
    sigma = np.sqrt(np.mean(y["A"] ** 2) / 10 ** (cfg.snr_db / 10))
    eps = rng.normal(size=(T, n * d)) * sigma
    X = {sc: x_clean[sc] + eps for sc in ("A", "B")}
    pre = fine < change_end
    return dict(edges=edges, Ls=Ls, fine=fine, y=y, X=X, pre=pre, sigma=sigma)


def build_R1(cfg, seed, alpha0=1.0, ridge=0.01):
    """Regime 1: truth drawn from the Gaussian prior whose MAP estimate is H_hist:
    precision (M alpha0 / sigma^2) G + ridge (M / sigma^2) I (ridge makes it proper on ker G)."""
    base = build(replace(cfg, regime="R2"), seed)
    rng = np.random.default_rng(seed + 7919)
    edges, Ls = base["edges"], base["Ls"]
    p = Ls[0].shape[0]
    _, G = hc.gram_recursion(edges, [np.zeros(p)] * len(Ls), Ls, cfg.N)
    lam, V = np.linalg.eigh(hc.big(G))
    M, sigma = cfg.T, 0.3
    prec = (M / sigma ** 2) * (alpha0 * lam + ridge)
    h = V @ (rng.normal(size=len(lam)) / np.sqrt(prec))
    H = h.reshape(cfg.N, p).T
    fine = base["fine"]; mids = 0.5 * (edges[1:] + edges[:-1])
    y = H @ hc.basis(cfg.N, fine, 1.0).T
    X = (H @ hc.basis(cfg.N, mids, 1.0).T).T + rng.normal(size=(cfg.T, p)) * sigma
    return dict(edges=edges, Ls=Ls, fine=fine, y={"A": y.T}, X={"A": X}, pre=base["pre"], sigma=sigma, G=G)


# ----------------------------------------------------------------------------- evaluation
def c_recursion(edges, Xs, N):
    p = Xs[0].shape[0]; C = np.zeros((p, N)); C[:, 0] = Xs[0]; t = edges[1]
    for k in range(1, len(Xs)):
        t2 = edges[k + 1]; A, e, _ = hc.step_mats(N, t, t2)
        C = (t / t2) * C @ A.T + np.outer(Xs[k], e); t = t2
    return C


def rel_err(y, yh, mask):
    return float(np.linalg.norm((y - yh)[mask]) / np.linalg.norm(y[mask]))


def run_task(args):
    cfg, seed, split = args
    t0 = time.perf_counter()
    data = build_R1(cfg, seed, alpha0=cfg.alpha0) if cfg.regime == "R1" else build(cfg, seed)
    edges, Ls, fine, pre = data["edges"], data["Ls"], data["fine"], data["pre"]
    N, T = cfg.N, cfg.T
    p = Ls[0].shape[0]
    G = data.get("G")
    if G is None:
        _, G = hc.gram_recursion(edges, [np.zeros(p)] * T, Ls, N)
    solver = hc.HistSolver(G)
    Pf = hc.basis(N, fine, 1.0)
    idx = np.minimum((fine * T).astype(int), T - 1)
    eigL = {}
    for L in Ls:
        if id(L) not in eigL:
            eigL[id(L)] = np.linalg.eigh(L)
    lamT, VT = np.linalg.eigh(Ls[-1])
    segs = {"pre": pre, "post": ~pre, "all": np.ones_like(pre), "end": fine >= 0.98}   # end = current value
    rows = []
    for sc, Xm in data["X"].items():
        y = data["y"][sc]
        Xs = list(Xm)
        C = c_recursion(edges, Xs, N)
        Hin = hc.filtered_compress(edges, Xs, Ls, N, ALPHAS)
        for ai, a in enumerate(ALPHAS):
            est = {
                "hist": (solver.solve(C, a) @ Pf.T).T,
                "cur": ((VT @ ((VT.T @ C) / (1 + a * lamT)[:, None])) @ Pf.T).T,
                "in": (Hin[ai] @ Pf.T).T,
                "legs": (C @ Pf.T).T,
            }
            pt = np.empty_like(y)
            for k in np.unique(idx):
                lam, V = eigL[id(Ls[k])]
                pt[idx == k] = V @ ((V.T @ Xs[k]) / (1 + a * lam))
            est["point"] = pt
            for m, yh in est.items():
                for sg, mk in segs.items():
                    rows.append((cfg.name, seed, split, sc, m, a, sg, rel_err(y, yh, mk)))
    return rows, time.perf_counter() - t0


def cells_stress():
    """Stress test towards the Gram memory's ideal regime (E1 well-specified strong prior; E2 physical ideal)."""
    b = Cfg()
    out = [replace(b, name=f"E1_a{int(a)}_N{N}", regime="R1", alpha0=a, N=N) for a in (1, 10, 100) for N in (4, 8, 16)]
    out += [replace(b, name=f"E2_N{N}_snr{s}", dev=0.0, theta=90.0, frac=1.0, N=N, snr_db=s) for N in (4, 8, 16) for s in (-10, 0, 10)]
    return out


def cells():
    b = Cfg()
    out = [replace(b, name=f"theta{int(t)}", theta=t) for t in (0, 5, 10, 20, 30, 45, 60, 90)]
    out += [replace(b, name=f"frac{f}", frac=f) for f in (0.1, 0.25, 1.0)]
    out += [replace(b, name=f"tau{t}", tau=t) for t in (0.2, 0.8)]
    out += [replace(b, name=f"N{N}", N=N) for N in (4, 8, 32)]
    out += [replace(b, name=f"snr{s}", snr_db=s) for s in (-10, -5, 0, 20, 30)]
    out += [replace(b, name=f"R3_{p}", regime="R3", perturb=p) for p in ("noisy_maps", "tau_shift", "drift", "many")]
    out += [replace(b, name=f"R1_theta{int(t)}", regime="R1", theta=t) for t in (0, 45)]
    out += [replace(b, name="d3_theta45", d=3)]
    out += [replace(b, name=f"theta{int(t)}_snr0", theta=t, snr_db=0.0) for t in (0, 5, 10, 20, 30, 45, 60, 90)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=50); ap.add_argument("--n-tune", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    ap.add_argument("--stress", action="store_true", help="run the stress-test cells (E1, E2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ap_cells = cells_stress() if a.stress else cells()
    cs = ap_cells
    if a.only:
        cs = [c for c in cs if c.name in a.only.split(",")]
    json.dump([asdict(c) for c in cs], open(os.path.join(a.out, "cells.json"), "w"), indent=1)
    tasks = [(c, s, "eval") for c in cs for s in range(a.n_eval)] + [(c, 100000 + s, "tune") for c in cs for s in range(a.n_tune)]
    tasks.sort(key=lambda x: -x[0].N)          # longest first
    t0 = time.perf_counter(); rows, secs = [], []
    with Pool(a.procs) as pool:
        for i, (r, dt) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
            rows += r; secs.append(dt)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} tasks, {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["cell", "seed", "split", "scenario", "method", "alpha", "segment", "rel_err"])
    path = os.path.join(a.out, "raw_errors.csv.gz")
    if a.only and os.path.exists(path):          # append a partial re-run, replacing the same cells
        old = pd.read_csv(path); df = pd.concat([old[~old.cell.isin(df.cell.unique())], df], ignore_index=True)
    df.to_csv(path, index=False, compression="gzip")
    print(f"done: {len(tasks)} tasks in {time.perf_counter() - t0:.0f}s (median task {np.median(secs):.1f}s)")


if __name__ == "__main__":
    main()
