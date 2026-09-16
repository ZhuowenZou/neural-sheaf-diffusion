"""Synthetic temporal sheaf process (Arm A of the analytic experiment).

Nodes live in K communities; a latent state x_u(t) in R^d evolves under a
community-specific continuous-time linear system (rotation + decay) ONLY
through physical time, so event order alone cannot recover it. Each
community pair carries a planted orthogonal transport O_{ab} (the sheaf).
An event at time t picks a source u and a destination v with probability
proportional to exp(-||O_{c(u)c(v)} x_u(t) - x_v(t)||^2 / tau), mixed with a
recurrence component (repeat a past partner with prob rho). Knobs turn each
generative factor off so the ablations can be matched to the factors.

Usage: python -m exp.synthetic_temporal_sheaf --out results/analytic/synth_full
       --transport rotations --dynamics evolving --gaps bursty --rho 0.3
"""
import argparse, json, os
import numpy as np, torch


def random_orthogonal(d, rng):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def make_dataset(n=300, k=6, d=4, n_events=60000, transport="rotations", dynamics="evolving",
                 gaps="bursty", rho=0.3, tau=0.1, seed=0, oracle_frac=0.15, rate_range=(0.2, 1.0), oracle_bucket=10.0, noise=0.02):
    """v2 generator (2026-09-05). The v1 generator multiplied EVERY node state by
    exp(-0.02*dt) at every event without renormalising, so all states decayed to
    ~0 within a few thousand events and the destination distribution became
    uniform: the generator's own oracle scored MRR 0.02 (chance 0.0033).  v2
    keeps states on the unit sphere (rotation only), uses a sharper tau, and
    records oracle ceilings on the last `oracle_frac` of events so that the
    signal each variant could exploit is known before any model is trained."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, k, size=n)
    # community dynamics: per 2x2 block an angular rate w_c (no norm decay)
    rates = rng.uniform(rate_range[0], rate_range[1], size=(k, d // 2)) if dynamics == "evolving" else np.zeros((k, d // 2))
    decay = 0.0
    O = np.zeros((k, k, d, d))
    for a in range(k):
        for b in range(a, k):
            Q = random_orthogonal(d, rng) if (transport == "rotations" and a != b) else np.eye(d)
            O[a, b], O[b, a] = Q, Q.T
    x = rng.normal(size=(n, d)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x.copy()   # initial latent states, exposed as observable node features (v3 runner)
    last_t = np.zeros(n)
    partners = [[] for _ in range(n)]

    def evolve_all(t):
        dt = t - last_t
        w = rates[comm]  # (n, d//2)
        ang = w * dt[:, None]
        c, s = np.cos(ang), np.sin(ang)
        xr = x.reshape(n, d // 2, 2)
        x0, x1 = xr[..., 0].copy(), xr[..., 1].copy()
        xr[..., 0] = c * x0 - s * x1
        xr[..., 1] = s * x0 + c * x1
        x[:] = xr.reshape(n, d) * np.exp(-decay * dt)[:, None]
        last_t[:] = t

    x_static = x.copy()

    def rank_of(scores, v, u):
        sc = scores.copy(); sc[u] = -np.inf
        return 1 + (sc > sc[v]).sum() + 0.5 * ((sc == sc[v]).sum() - 1)

    t = 0.0
    src, dst, ts = [], [], []
    oracle = {"dynamics_transport": [], "static_x": [], "no_transport": [], "stale_bucket": [], "recurrent": []}
    start_oracle = int((1.0 - oracle_frac) * n_events)
    x_bucket = x.copy(); bucket_id = -1   # state as of the previous snapshot boundary (predict-then-update view)
    for i in range(n_events):
        t += (rng.pareto(1.5) * 2.0) if gaps == "bursty" else 1.0
        b = int(t // oracle_bucket)
        if b != bucket_id:
            x_bucket = x.copy(); bucket_id = b
        evolve_all(t)
        u = int(rng.integers(0, n))
        rec = bool(partners[u]) and rng.random() < rho
        if rec:
            v = int(rng.choice(partners[u]))
        else:
            tu = np.einsum("bij,j->bi", O[comm[u], comm], x[u])  # transported source per candidate
            dist = ((tu - x) ** 2).sum(1)
            p = np.exp(-dist / tau); p[u] = 0.0; p /= p.sum()
            v = int(rng.choice(n, p=p))
        if i >= start_oracle:
            s_dyn = -((np.einsum("bij,j->bi", O[comm[u], comm], x[u]) - x) ** 2).sum(1)
            s_sta = -((np.einsum("bij,j->bi", O[comm[u], comm], x_static[u]) - x_static) ** 2).sum(1)
            s_not = -((x[u] - x) ** 2).sum(1)
            oracle["dynamics_transport"].append(1.0 / rank_of(s_dyn, v, u))
            oracle["static_x"].append(1.0 / rank_of(s_sta, v, u))
            oracle["no_transport"].append(1.0 / rank_of(s_not, v, u))
            s_stale = -((np.einsum("bij,j->bi", O[comm[u], comm], x_bucket[u]) - x_bucket) ** 2).sum(1)
            oracle["stale_bucket"].append(1.0 / rank_of(s_stale, v, u))
            oracle["recurrent"].append(rec)
        src.append(u); dst.append(v); ts.append(t)
        partners[u].append(v)
        x[v] += rng.normal(scale=noise, size=d)   # per-hit state kick (random-walk drift; 0 = exactly static states)
        x[v] /= np.linalg.norm(x[v])
    is_rec = np.array(oracle.pop("recurrent"), dtype=bool)
    ceilings = {}
    for key, vals in oracle.items():
        a = np.array(vals)
        ceilings[key] = dict(all=float(a.mean()), non_recurrent=float(a[~is_rec].mean()) if (~is_rec).any() else None,
                             recurrent=float(a[is_rec].mean()) if is_rec.any() else None)
    ceilings["recurrent_fraction"] = float(is_rec.mean()); ceilings["chance"] = 1.0 / (n - 1)
    return dict(src=np.array(src), dst=np.array(dst), t=np.array(ts), comm=comm, n=n, x0=x0, rates=rates,
                meta=dict(n=n, k=k, d=d, transport=transport, dynamics=dynamics, gaps=gaps, rho=rho, tau=tau, seed=seed,
                          generator="v2", rate_range=list(rate_range), oracle_bucket=oracle_bucket, noise=noise,
                          oracle_mrr_last_fraction=oracle_frac, oracle=ceilings))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--transport", choices=["rotations", "identity"], default="rotations")
    ap.add_argument("--dynamics", choices=["evolving", "static"], default="evolving")
    ap.add_argument("--gaps", choices=["bursty", "uniform"], default="bursty")
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.1, help="softmax temperature of the transported-distance kernel")
    ap.add_argument("--n", type=int, default=300, help="number of nodes")
    ap.add_argument("--k", type=int, default=6, help="number of communities")
    ap.add_argument("--noise", type=float, default=0.02, help="per-hit state kick scale (random-walk drift of destination states)")
    ap.add_argument("--n-events", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-range", type=float, nargs=2, default=[0.2, 1.0], help="angular rates (rad per time unit) drawn per community/block")
    ap.add_argument("--oracle-bucket", type=float, default=10.0, help="snapshot width used by the stale-bucket oracle")
    a = ap.parse_args()
    ds = make_dataset(n=a.n, k=a.k, n_events=a.n_events, transport=a.transport, dynamics=a.dynamics, gaps=a.gaps, rho=a.rho, tau=a.tau, seed=a.seed,
                      rate_range=tuple(a.rate_range), oracle_bucket=a.oracle_bucket, noise=a.noise)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "events.npz"), src=ds["src"], dst=ds["dst"], t=ds["t"], comm=ds["comm"], x0=ds["x0"], rates=ds["rates"])
    json.dump(ds["meta"], open(os.path.join(a.out, "meta.json"), "w"))
    # recurrency degree of the last 20% (test) w.r.t. the first 80%
    m = int(0.8 * len(ds["src"])); key = ds["src"] * ds["n"] + ds["dst"]
    print(json.dumps(dict(ds["meta"], events=len(ds["src"]), rec_pair=float(np.isin(key[m:], np.unique(key[:m])).mean()),
                          median_gap=float(np.median(np.diff(ds["t"]))))))
