"""History-dependent synthetic link task (server handoff 2026-09-22, section 4).

Generative law ("cued target drift"):
  * N nodes in K balanced hidden communities; K designated CUE nodes (one per community).
  * Every node u holds a hidden target community g_u(t).  It is set by u's most
    recent CUE event: an interaction u -> cue_k sets g_u := k.  Cues are ordinary
    observable events (with probability p_cue per event); the initial targets are
    random and are NOT announced (nodes without a cue yet are unpredictable).
  * All other events: source u uniform; destination v drawn from community g_u(t)
    with probability proportional to exp(beta * pop_v), pop_v ~ N(0,1) a hidden
    per-node popularity observable only through in-degree history.
  * Event times: i.i.d. exponential gaps (irregular); batches are fixed-width time
    windows in the runner (--time-window), independent of the event process.
  * Node features are fixed random Gaussians (never encode community, popularity
    or targets).  The current batch is scored from the state BEFORE ingestion
    (predict-from-previous), so a model without memory sees only the identical
    node features / context and cannot infer g_u.
  * Reserved novelty: for val/test events, with probability p_novel the destination
    is drawn (popularity-weighted) among target-community nodes NEVER paired with u
    before; realised novel fractions are reported per split.
  * Negative candidates for val/test: 32 uniform nodes != positive, fixed per
    query, stored with the data (query ids = event rows).

Oracles on the same queries/candidates (TGB tie handling: mean of optimistic
and pessimistic rank):
  * permitted-history oracle: knows the law; uses only events with t < batch start
    (cue -> target community; in-degree counts -> popularity ranking);
  * latent oracle: true g_u at the event and true pop_v (separate ceiling);
  * current-only reference: no history -> uniform scores (chance).

The dataset shim exposes the interface `exp.run_event_benchmark` uses so all arms,
audits and counters run unchanged:  --dataset synth-history:<npz path>.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch


def generate(n_nodes=400, n_comm=8, n_events=60000, p_cue=0.03, beta=1.0, gap_mean=1.0, p_novel=0.5,
             val_frac=0.15, test_frac=0.15, n_neg=32, feat_dim=32, seed=1, typed_cues=False):
    """typed_cues: cue events carry an observable relation id 1+k (interactions: 0), so the cue is an
    explicit event attribute rather than only the identity of the cue node (variant "r")."""
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(n_comm), n_nodes // n_comm)
    comm = np.concatenate([comm, rng.integers(0, n_comm, n_nodes - comm.size)])
    rng.shuffle(comm)
    cue_nodes = np.array([int(np.nonzero(comm == k)[0][0]) for k in range(n_comm)])   # one cue node per community
    is_cue = np.zeros(n_nodes, dtype=bool); is_cue[cue_nodes] = True
    pop = rng.normal(size=n_nodes)
    members = [np.nonzero((comm == k) & ~is_cue)[0] for k in range(n_comm)]
    target = rng.integers(0, n_comm, n_nodes)          # initial hidden targets (unannounced)
    gaps = rng.exponential(gap_mean, n_events)
    t = np.cumsum(gaps)
    t_val = t[int((1 - val_frac - test_frac) * n_events)]
    t_test = t[int((1 - test_frac) * n_events)]
    src = rng.integers(0, n_nodes, n_events)
    dst = np.zeros(n_events, dtype=np.int64)
    kind = np.zeros(n_events, dtype=np.int8)           # 0 = interaction, 1 = cue
    novel = np.zeros(n_events, dtype=bool)
    seen = set()
    target_at = np.zeros(n_events, dtype=np.int64)
    for i in range(n_events):
        u = int(src[i])
        if rng.random() < p_cue:
            k = int(rng.integers(0, n_comm)); dst[i] = cue_nodes[k]; kind[i] = 1; target[u] = k
            target_at[i] = k
            continue
        g = int(target[u]); target_at[i] = g
        cand = members[g]
        cand = cand[cand != u]
        split_is_eval = t[i] >= t_val
        if split_is_eval and rng.random() < p_novel:
            unseen = np.array([v for v in cand if (u, int(v)) not in seen])
            if unseen.size:
                cand = unseen
        w = np.exp(beta * pop[cand]); w /= w.sum()
        v = int(rng.choice(cand, p=w)); dst[i] = v
        novel[i] = (u, v) not in seen
        seen.add((u, v))
    x = rng.normal(size=(n_nodes, feat_dim)).astype(np.float32)
    split = np.where(t < t_val, 0, np.where(t < t_test, 1, 2)).astype(np.int8)
    neg = np.full((n_events, n_neg), -1, dtype=np.int64)
    ev = np.nonzero(split > 0)[0]
    for i in ev:
        c = rng.integers(0, n_nodes - 1, n_neg)
        c = c + (c >= dst[i])
        neg[i] = c
    edge_type = np.where(kind == 1, 1 + target_at, 0).astype(np.int64) if typed_cues else np.zeros(n_events, dtype=np.int64)
    data = dict(src=src, dst=dst, t=t, kind=kind, novel=novel, split=split, neg=neg, x=x, comm=comm, cue_nodes=cue_nodes,
                pop=pop, target_at=target_at, edge_type=edge_type, typed_cues=np.array(bool(typed_cues)),
                config=json.dumps(dict(n_nodes=n_nodes, n_comm=n_comm, n_events=n_events, p_cue=p_cue, beta=beta, gap_mean=gap_mean,
                                       p_novel=p_novel, val_frac=val_frac, test_frac=test_frac, n_neg=n_neg, feat_dim=feat_dim, seed=seed,
                                       typed_cues=bool(typed_cues))))
    return data


def data_sha(data):
    h = hashlib.sha256()
    for k in ("src", "dst", "t", "kind", "split", "neg", "x"):
        h.update(np.ascontiguousarray(data[k]).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- runner shim
class _NegSampler:
    def __init__(self, neg, t):
        self.neg = neg
        self._lookup = {}
        self.t = t

    def query_batch(self, pos_src, pos_dst, pos_timestamp, edge_type=None, split_mode="test"):
        ids = self._ids_for(np.asarray(pos_src), np.asarray(pos_dst), np.asarray(pos_timestamp))
        return [self.neg[i] for i in ids]

    def _ids_for(self, s, d, ts):
        if not self._lookup:
            for i, (a, b, c) in enumerate(zip(self.src, self.dst, self.t)):
                self._lookup[(int(a), int(b), float(c))] = i
        return [self._lookup[(int(a), int(b), float(c))] for a, b, c in zip(s, d, ts)]


class SyntheticHistoryDataset:
    """Duck-types the pieces of PyGLinkPropPredDataset that the event runner uses."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.src = z["src"]; self.dst = z["dst"]; self.t = z["t"]; self.kind = z["kind"]; self.novel = z["novel"]
        self.split = z["split"]; self.neg = z["neg"]; self.x = z["x"]; self.comm = z["comm"]; self.cue_nodes = z["cue_nodes"]
        self.pop = z["pop"]; self.target_at = z["target_at"]; self.config = json.loads(str(z["config"]))
        n = self.src.size
        # integer timestamps (the runner and REC caches use ints in places): milliseconds
        self.t_int = np.round(self.t * 1000).astype(np.int64)
        self.train_mask = torch.as_tensor(self.split == 0); self.val_mask = torch.as_tensor(self.split == 1); self.test_mask = torch.as_tensor(self.split == 2)
        self.edge_type = z["edge_type"] if "edge_type" in z.files else np.zeros(n, dtype=np.int64)
        self.typed = bool(z["typed_cues"]) if "typed_cues" in z.files else False
        self.eval_metric = "mrr"; self.num_rels = (int(self.edge_type.max()) + 1) if self.typed else 0
        self.node_feat = torch.as_tensor(self.x)
        self.negative_sampler = _NegSampler(self.neg, self.t_int); self.negative_sampler.src = self.src; self.negative_sampler.dst = self.dst
        self.name = "synth-history"

    def get_TemporalData(self):
        from torch_geometric.data import TemporalData
        td = TemporalData(src=torch.as_tensor(self.src), dst=torch.as_tensor(self.dst), t=torch.as_tensor(self.t_int),
                          msg=torch.ones(self.src.size, 1))
        if self.typed:
            td.edge_type = torch.as_tensor(self.edge_type)
        return td

    def load_val_ns(self):
        return None

    def load_test_ns(self):
        return None


# ----------------------------------------------------------------------------- oracles
def _rr(pos, neg):
    opt = (neg > pos[:, None]).sum(1); pes = (neg >= pos[:, None]).sum(1)
    return 1.0 / (0.5 * (opt + pes) + 1.0)


def oracles(ds, time_window):
    """Per val/test query: reciprocal rank of the permitted-history oracle, the latent oracle and chance,
    stratified by novel/recurring and event kind.  Batches = the runner's fixed-width windows anchored at
    the split's first event (identical to exp.temporal_utils.build_temporal_snapshots)."""
    import pandas as pd
    n = ds.src.size; N = ds.x.shape[0]; K = ds.config["n_comm"]
    cue_of = {int(c): k for k, c in enumerate(ds.cue_nodes)}
    rows = []
    for split_id, split in ((1, "val"), (2, "test")):
        ids = np.nonzero(ds.split == split_id)[0]
        t0 = ds.t_int[ids[0]]
        bucket = (ds.t_int[ids] - t0) // int(time_window)
        # permitted history = all events with bucket < this query's bucket (previous batches; earlier splits fully)
        # state trackers replayed in time order over the whole stream
        last_cue = np.full(N, -1, dtype=np.int64); indeg = np.zeros(N)
        ptr = 0
        order = np.arange(n)   # already time sorted
        for b in np.unique(bucket):
            q = ids[bucket == b]
            t_start = ds.t_int[q[0]] if b == 0 else t0 + b * int(time_window)
            # ingest everything strictly before this bucket's window start (previous batches)
            while ptr < n and ds.t_int[order[ptr]] < t_start and order[ptr] < q[0]:
                i = order[ptr]; u, v = int(ds.src[i]), int(ds.dst[i])
                if ds.kind[i] == 1:
                    last_cue[u] = cue_of[v]
                else:
                    indeg[v] += 1
                ptr += 1
            for i in q:
                if ds.kind[i] == 1:
                    continue   # cue events are not link-prediction targets of interest but ARE scored by the runner; keep them separate
                u, v = int(ds.src[i]), int(ds.dst[i]); cands = np.concatenate([[v], ds.neg[i]])
                g_perm = last_cue[u]
                if g_perm >= 0:
                    s_perm = (ds.comm[cands] == g_perm) * 1000.0 + indeg[cands] + 1e-6 * np.arange(cands.size)[::-1] * 0
                else:
                    s_perm = indeg[cands].astype(float)   # no cue seen: popularity only
                s_lat = (ds.comm[cands] == ds.target_at[i]) * 1000.0 + ds.pop[cands]
                rows.append(dict(split=split, event=int(i), novel=bool(ds.novel[i]), cue_known=bool(g_perm >= 0),
                                 rr_permitted=float(_rr(s_perm[:1], s_perm[None, 1:])[0]),
                                 rr_latent=float(_rr(s_lat[:1], s_lat[None, 1:])[0]),
                                 rr_chance=float(1.0 / (1 + 0.5 * cands.size - 0.5))))
    df = pd.DataFrame(rows)
    summ = df.groupby(["split", "novel"]).agg(n=("event", "count"), mrr_permitted=("rr_permitted", "mean"),
                                              mrr_latent=("rr_latent", "mean"), mrr_chance=("rr_chance", "mean"),
                                              cue_known=("cue_known", "mean")).reset_index()
    overall = df.groupby("split").agg(n=("event", "count"), mrr_permitted=("rr_permitted", "mean"), mrr_latent=("rr_latent", "mean"),
                                      mrr_chance=("rr_chance", "mean"), novel_fraction=("novel", "mean"), cue_known=("cue_known", "mean")).reset_index()
    return df, summ, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-nodes", type=int, default=400); ap.add_argument("--n-comm", type=int, default=8)
    ap.add_argument("--n-events", type=int, default=60000); ap.add_argument("--p-cue", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--p-novel", type=float, default=0.5)
    ap.add_argument("--time-window", type=int, default=20000, help="runner bucket width in ms (gap mean = 1000 ms)")
    ap.add_argument("--typed-cues", action="store_true", help="cue events carry relation id 1+k (variant r)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = generate(n_nodes=args.n_nodes, n_comm=args.n_comm, n_events=args.n_events, p_cue=args.p_cue, beta=args.beta,
                    p_novel=args.p_novel, seed=args.seed, typed_cues=args.typed_cues)
    path = os.path.join(args.out, "data.npz")
    np.savez(path, **data)
    sha = data_sha(data)
    ds = SyntheticHistoryDataset(path)
    per_query, summ, overall = oracles(ds, args.time_window)
    per_query.to_csv(os.path.join(args.out, "oracle_per_query.csv.gz"), index=False, compression="gzip")
    summ.to_csv(os.path.join(args.out, "oracle_by_novelty.csv"), index=False)
    overall.to_csv(os.path.join(args.out, "oracle_summary.csv"), index=False)
    meta = dict(config=data["config"], data_sha256=sha, time_window_ms=args.time_window,
                events_per_split={s: int((data["split"] == i).sum()) for i, s in enumerate(("train", "val", "test"))},
                cue_events=int(data["kind"].sum()), novel_fraction={s: float(data["novel"][data["split"] == i].mean()) for i, s in enumerate(("train", "val", "test"))},
                oracle_summary=overall.to_dict("records"), oracle_by_novelty=summ.to_dict("records"),
                permitted_information="events with timestamp < the query batch's window start (previous batches only)",
                hidden_sidecar="comm, pop, target_at, cue_nodes, novel (npz keys; used only by oracles/analysis)")
    json.dump(meta, open(os.path.join(args.out, "generator_meta.json"), "w"), indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("data_sha256", "events_per_split", "cue_events", "novel_fraction")}))
    print(overall.to_string(index=False)); print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
