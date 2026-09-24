"""E3: the learned TSD model on its ideal hypothesis -- vector signals on a graph are related through
edge-specific, possibly NON-FLAT transports that switch at an unannounced time and must be inferred from history.

  y_u(k) = mean_{v~u} T_uv(k) s_v(k) + 0.1 eps,  s(k) ~ N(0, I) fresh and observed at k,  y_u(k) revealed after
  predicting it;  model input x_u(k) = [s_u(k), y_u(k-1)].   T_uv in SO(2).

Variants: nonflat (independent rotation per edge; cycle products != I), flat (T_uv = R_u^T R_v; node frames suffice).
Transports are resampled per episode (so static parameters cannot memorise them) and a random half of the edges
(flat: of the nodes) switch at a random time in the episode.  Node inputs are the current readings only.
Arms (same head, loss, data, budget; one flag each): TSD, current-only maps, identity maps, node-frame geometry,
history-conditioned edge gates, GRU + identity propagation, diagonal SSM + identity, core-off.
References: persistence, true-law oracle (knows T), permitted-history oracle (knows the law, estimates T by
sliding-window least squares on past data only).

  python -m exp.histgeom.transport_task --arm tsd --variant nonflat --seed 43 --out <dir>
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from exp.histgeom import core as hc

ARMS = {"tsd": {}, "curonly": {"sheaf_conditioning": "current_only"}, "identity": {"spatial": "identity"},
        "nodeframe": {"spatial": "node_frame"}, "attention": {"spatial": "attention"},
        "gru": {"backbone": "gru", "spatial": "identity"}, "diag": {"backbone": "diag_ssm", "spatial": "identity"},
        "coreoff": {"no_memory": True, "layers": 0, "fast_core_off": True},
        "tsd_mlpdec": {},
        "oracle_maps": {}}     # upper bound: the TRUE transports are supplied as restriction maps (no inference)


class OracleSheafLearner(nn.Module):
    """Returns Householder parameters realising the true transport T_uv = R_{e<-u}^T R_{e<-v}:
    R(p) = Rot(pi + 2 atan p); for u < v set p(u->v) = 0 (Rot(pi)) and p(v->u) = tan(theta_uv / 2)."""
    def __init__(self, n_total):
        super().__init__(); self.n = n_total; self.table = None; self.L = None
    def set_angles(self, theta_by_pair):          # dict (u, v) with u < v -> angle of T_uv, global node ids
        self.table = theta_by_pair
    def forward(self, x, edge_index):
        r, c = edge_index.tolist()
        out = []
        for u, v in zip(r, c):
            out.append(0.0 if u < v else float(np.tan(self.table[(v, u)] / 2.0)))
        return torch.tensor(out, dtype=x.dtype, device=x.device).view(-1, 1)
    def set_L(self, w):
        self.L = w.detach()      # TSD with a two-layer MLP map decoder on [h_u; h_v] (capacity for pairwise correlation)


class MLPConcatSheafLearner(nn.Module):
    def __init__(self, d_in, n_out, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2 * d_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out)); self.L = None
    def forward(self, x, edge_index):
        r, c = edge_index
        return torch.tanh(self.net(torch.cat([x[r], x[c]], dim=1)))
    def set_L(self, w):
        self.L = w.detach()


# ----------------------------------------------------------------------------- generator
def graph(rng, n=24, k=4):
    E = hc.knn_edges(rng.random((n, 2)), k)
    return E


def episode(rng, E, n, K=200, variant="nonflat", sig=0.1, switch_frac=0.5):
    """y_u(k) = mean_{v~u} T_uv(k) s_v(k) + sig*eps,  s_v(k) ~ N(0, I) fresh each step (observed at k);
    y_u(k) is revealed only after the prediction at step k.  Returns s (K,n,2), y (K,n,2), switch time, T, nbr."""
    def rots(m):
        return np.stack([hc.rot2(a) for a in rng.uniform(0, 2 * np.pi, m)])
    tsw = int(rng.integers(K // 4, 3 * K // 4))
    if variant == "nonflat":
        T0 = rots(len(E)); T1 = T0.copy(); sw = rng.random(len(E)) < switch_frac; T1[sw] = rots(int(sw.sum()))
    else:
        R0 = rots(n); R1 = R0.copy(); sw = rng.random(n) < switch_frac; R1[sw] = rots(int(sw.sum()))
        T0 = np.stack([R0[u].T @ R0[v] for u, v in E]); T1 = np.stack([R1[u].T @ R1[v] for u, v in E])
    nbr = [[] for _ in range(n)]
    for i, (u, v) in enumerate(E):
        nbr[u].append((v, i, False)); nbr[v].append((u, i, True))   # T_vu = T_uv^T
    S = rng.normal(size=(K, n, 2)); Y = np.zeros((K, n, 2))
    for k in range(K):
        T = T0 if k < tsw else T1
        for u in range(n):
            Y[k, u] = np.mean([(T[i].T if tr else T[i]) @ S[k, v] for v, i, tr in nbr[u]], axis=0)
    Y += sig * rng.normal(size=Y.shape)
    return S, Y, tsw, (T0, T1), nbr


def oracle_preds(S, Y, tsw, Ts, nbr, window=30, ridge=1e-3):
    """true-law oracle (knows T) and permitted-history oracle (window least squares on past (s, y) pairs only)."""
    K, n = S.shape[:2]
    true = np.zeros((K, n, 2)); hist = np.zeros((K, n, 2))
    for k in range(K):
        T = Ts[0] if k < tsw else Ts[1]
        for u in range(n):
            true[k, u] = np.mean([(T[i].T if tr else T[i]) @ S[k, v] for v, i, tr in nbr[u]], axis=0)
            lo = max(0, k - window)
            if k - lo < 3:
                continue
            Xr = np.concatenate([S[lo:k, v] for v, _, _ in nbr[u]], axis=1)
            W = np.linalg.solve(Xr.T @ Xr + ridge * np.eye(Xr.shape[1]), Xr.T @ Y[lo:k, u])
            hist[k, u] = np.concatenate([S[k, v] for v, _, _ in nbr[u]]) @ W
    return true, hist


# ----------------------------------------------------------------------------- model
def make_model(E, n_per, B, arm, device, d_h=32):
    from models.faithful_event_model import FaithfulEventTemporalSheafDiffusion
    src = []; dst = []
    for b in range(B):
        for u, v in E:
            src += [u + b * n_per, v + b * n_per]; dst += [v + b * n_per, u + b * n_per]
    ei = torch.tensor([src, dst])
    args = dict(d=2, add_lp=False, add_hp=False, device=device, graph_size=n_per * B, layers=2, normalised=True,
                deg_normalised=False, linear=False, input_dropout=0.0, dropout=0.0, left_weights=True, right_weights=True,
                sparse_learner=False, use_act=True, input_dim=4, hidden_channels=4, output_dim=2, sheaf_act="tanh",
                second_linear=False, orth="householder", edge_weights=False, max_t=1.0, stateful_temporal=False,
                closure_hops=1, temporal_d_model=d_h, feedback_dim=8, num_relations=0, train_negatives_per_pos=4)
    args.update(ARMS[arm])
    m = FaithfulEventTemporalSheafDiffusion(ei, {k: v for k, v in args.items()}).to(device)
    if arm == "tsd_mlpdec":
        m.sheaf_learner = MLPConcatSheafLearner(m.d_h, m.get_param_size()).to(device)
    if arm == "oracle_maps":
        m.sheaf_learner = OracleSheafLearner(n_per * B)
    m.ssm.set_delta_scale(1.0)
    return m, ei.to(device)


def angle_tables(eps, E, n):
    """per episode-batch: (table_before, table_after, switch times) with global node ids."""
    before, after, tsws = {}, {}, []
    for b, e in enumerate(eps):
        T0, T1 = e[3]; tsws.append(e[2])
        for i, (u, v) in enumerate(E):
            before[(u + b * n, v + b * n)] = np.arctan2(T0[i][1, 0], T0[i][0, 0])
            after[(u + b * n, v + b * n)] = np.arctan2(T1[i][1, 0], T1[i][0, 0])
    return before, after, tsws


def oracle_table_at(k, before, after, tsws, n):
    return {key: (after[key] if k >= tsws[key[0] // n] else before[key]) for key in before}


class Net(nn.Module):
    def __init__(self, core):
        super().__init__(); self.core = core; self.head = nn.Linear(core.hidden_dim, 2)

    def step(self, x, ei, k, state):
        snap = {"x": x, "edge_index": ei, "active_nodes": None, "timestamp": torch.tensor(float(k), device=x.device)}
        out, state = self.core.forward_sequence([snap], initial_state=state)
        return self.head(out[0]["spatial"]), state


def batch_episodes(rng, E, n, B, K, variant):
    """inputs x(k) = [s(k), y(k-1)] (y(-1) = 0), targets y(k); stacked over B disjoint graph copies."""
    eps = [episode(rng, E, n, K, variant) for _ in range(B)]
    S = np.concatenate([e[0] for e in eps], axis=1); Y = np.concatenate([e[1] for e in eps], axis=1)
    Yprev = np.concatenate([np.zeros_like(Y[:1]), Y[:-1]], axis=0)
    return np.concatenate([S, Yprev], axis=2), Y, eps


def run(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grng = np.random.default_rng(12345)                    # one fixed graph for all arms/seeds
    n = 24; E = graph(grng, n)
    core, ei = make_model(E, n, a.batch, a.arm, dev)
    net = Net(core).to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    params = sum(p.numel() for p in net.parameters())
    t0 = time.time(); hist = []
    data_rng = np.random.default_rng(1000 + a.seed)       # training data stream per seed (identical across arms)
    for it in range(a.iters):
        x, y, eps_ = batch_episodes(data_rng, E, n, a.batch, a.K, a.variant)
        X = torch.tensor(x, dtype=torch.float32, device=dev); Y = torch.tensor(y, dtype=torch.float32, device=dev)
        if a.arm == "oracle_maps":
            tabs = angle_tables(eps_, E, n)
        core.reset_temporal_state(); state = None; losses = []
        opt.zero_grad(); acc = 0.0; cnt = 0
        for k in range(a.K):
            if a.arm == "oracle_maps":
                core.sheaf_learner.set_angles(oracle_table_at(k, *tabs, n))
            pred, state = net.step(X[k], ei, k, state)
            if k >= a.burn:
                acc = acc + ((pred - Y[k]) ** 2).mean(); cnt += 1
            if (k + 1) % a.bptt == 0 or k == a.K - 1:
                if cnt:
                    loss = acc / cnt; loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
                    losses.append(float(loss.detach())); acc = 0.0; cnt = 0
                state = type(state)(memory=state.memory.detach(), spatial=state.spatial.detach())
        hist.append(float(np.mean(losses)))
        if it % 10 == 0:
            print(f"it {it} train_mse {hist[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # evaluation on fixed test episodes (same across arms and seeds)
    test_rng = np.random.default_rng(777)
    net.eval(); rows = []
    with torch.no_grad():
        for tb in range(a.test_batches):
            x, y, eps = batch_episodes(test_rng, E, n, a.batch, a.K, a.variant)
            X = torch.tensor(x, dtype=torch.float32, device=dev)
            core.reset_temporal_state(); state = None; P = []
            if a.arm == "oracle_maps":
                tabs = angle_tables(eps, E, n)
            for k in range(a.K):
                if a.arm == "oracle_maps":
                    core.sheaf_learner.set_angles(oracle_table_at(k, *tabs, n))
                pred, state = net.step(X[k], ei, k, state); P.append(pred.cpu().numpy())
            P = np.stack(P)
            for b, (Sb, Yb, tsw, Ts, nbr) in enumerate(eps):
                sl = slice(b * n, (b + 1) * n)
                tru, his = oracle_preds(Sb, Yb, tsw, Ts, nbr) if a.oracles else (None, None)
                for k in range(a.burn, a.K):
                    phase = "pre" if k < tsw else ("adapt" if k < tsw + 30 else "post")
                    r = dict(episode=tb * a.batch + b, k=k, phase=phase, mse_model=float(np.mean((P[k, sl] - Yb[k]) ** 2)),
                             mse_zero=float(np.mean(Yb[k] ** 2)), mse_persist=float(np.mean((Yb[k - 1] - Yb[k]) ** 2)))
                    if a.oracles:
                        r.update(mse_true=float(np.mean((tru[k] - Yb[k]) ** 2)), mse_hist=float(np.mean((his[k] - Yb[k]) ** 2)))
                    rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "test_steps.csv.gz"), index=False, compression="gzip")
    summ = dict(arm=a.arm, variant=a.variant, seed=a.seed, params=params, train_secs=time.time() - t0,
                final_train_mse=hist[-1], **{f"{c}_{ph}": float(df[df.phase == ph][c].mean()) for c in df.columns if c.startswith("mse")
                                            for ph in ("pre", "adapt", "post")},
                **{f"{c}_all": float(df[c].mean()) for c in df.columns if c.startswith("mse")})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(hist, open(os.path.join(a.out, "train_curve.json"), "w"))
    print(json.dumps(summ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tsd", choices=list(ARMS)); ap.add_argument("--variant", default="nonflat", choices=["nonflat", "flat"])
    ap.add_argument("--seed", type=int, default=43); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--K", type=int, default=200); ap.add_argument("--burn", type=int, default=10)
    ap.add_argument("--bptt", type=int, default=10); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--test-batches", type=int, default=4); ap.add_argument("--oracles", action="store_true")
    run(ap.parse_args())
