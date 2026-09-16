"""Train/evaluate event models on the synthetic temporal sheaf process (Arm A).

Snapshots = one per unit of generator time (events bucketed by floor(t)),
train = first 70% of events, val = next 10%, test = last 20%. Loss = pairwise
softplus vs 32 uniform negatives; eval = MRR against ALL other nodes,
stratified by gap-since-source's-last-event quartile and novel/recurrent.

Usage: python -m exp.run_synthetic_sheaf --data results/analytic/synth_full
       --variant full|identity|nodelta|current_only|original --out ...
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.faithful_event_model import FaithfulEventTemporalSheafDiffusion
from models.sparse_temporal_mamba import EventTemporalMambaSheafDiffusion

VARIANTS = {
    "full": {}, "identity": {"sheaf_identity": True}, "nodelta": {"no_delta_t": True},
    "current_only": {"sheaf_conditioning": "current_only"}, "original": None,
}


def build(args, n, device, ctx, input_dim=16):
    margs = {"d": 2, "add_lp": False, "add_hp": False, "device": device, "graph_size": n, "layers": 2, "normalised": True,
             "deg_normalised": False, "linear": False, "input_dropout": 0.0, "dropout": 0.0, "left_weights": True,
             "right_weights": True, "sparse_learner": False, "use_act": True, "input_dim": input_dim, "hidden_channels": getattr(args, "hidden", 8),
             "output_dim": n, "sheaf_act": "tanh", "second_linear": False, "orth": "householder", "edge_weights": False,
             "max_t": 1.0, "stateful_temporal": False, "closure_hops": 1, "temporal_d_model": 32, "num_relations": 0,
             "train_negatives_per_pos": 32, "candidate_chunk_size": 1024, "max_score_elements": 4_000_000,
             "feedback_dim": 16, "memory_readout": True, "recurrency_decoder": args.recurrency, "learnable_node_features": True,
             "embeddings_in_head": True}
    v = VARIANTS[args.variant]
    if v is None:
        return EventTemporalMambaSheafDiffusion(ctx, margs).to(device)
    margs.update(v)
    return FaithfulEventTemporalSheafDiffusion(ctx, margs).to(device)


def snapshots(src, dst, t, x, n, device, bucket=10.0):
    b = np.floor(t / bucket).astype(np.int64)
    out = []
    for key in np.unique(b):
        m = b == key
        s, d = torch.as_tensor(src[m]), torch.as_tensor(dst[m])
        out.append({"x": x, "edge_index": torch.stack([torch.cat([s, d]), torch.cat([d, s])]).to(device),
                    "active_nodes": torch.unique(torch.cat([s, d])).to(device), "timestamp": torch.tensor(float(t[m].max())),
                    "edge_types": torch.zeros(len(s), dtype=torch.long, device=device),
                    "edge_timestamps": torch.as_tensor(t[m], dtype=torch.float64, device=device), "src": s.to(device), "dst": d.to(device)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--recurrency", action="store_true"); ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--hidden", type=int, default=8, help="hidden channels of the backbone/head representations")
    ap.add_argument("--features", choices=["observed", "random"], default="observed",
                    help="observed = x(0) + community one-hot from the generator (v3); random = 16-d Gaussian (v1/v2)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="rescale timestamps at TEST time only")
    ap.add_argument("--leaky", action="store_true", help="pre-audit protocol: score snapshot k from its own output (for the correction note only)")
    ap.add_argument("--bucket", type=float, default=10.0, help="snapshot width in generator time units")
    ap.add_argument("--ctx-events", type=int, default=300, help="train-event prefix used as the static context graph")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(args.data, "events.npz")); src, dst, t = z["src"], z["dst"], z["t"]; n = int(max(src.max(), dst.max())) + 1
    n_tr, n_va = int(0.7 * len(src)), int(0.8 * len(src))
    if args.features == "observed" and "x0" in z.files:
        # v3: observable node attributes = initial latent state x(0) + community one-hot.
        # The model must learn the rotation rates / transports from events to track
        # x(t); a static model knows only x(0).  Random features hide the state entirely.
        comm = torch.as_tensor(z["comm"]).long(); onehot = torch.nn.functional.one_hot(comm, int(comm.max()) + 1).float()
        x = torch.cat([torch.as_tensor(z["x0"]).float(), onehot], dim=1).to(device)
        print(f"node features: observed x0 ({z['x0'].shape[1]}) + community one-hot ({onehot.size(1)})", flush=True)
    else:
        x = torch.randn(n, 16, generator=torch.Generator().manual_seed(0)).to(device)
    from exp import temporal_benchmark_utils as bu
    # the Laplacian builder needs an undirected, coalesced context graph
    # Context graph = a SMALL prefix of train events: with 60k events on 300
    # nodes the full train graph is near-complete, so every event's local set
    # (closure over the context) would be the whole graph and the sheaf
    # learner would decode ~35k maps per step. The core only needs the
    # current event neighbourhood; the context is a runner convention.
    ctx = bu._normalize_sheaf_edge_index(torch.stack([torch.as_tensor(src[:args.ctx_events]), torch.as_tensor(dst[:args.ctx_events])])).to(device)
    model = build(args, n, device, ctx, input_dim=int(x.size(1)))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = snapshots(src[:n_tr], dst[:n_tr], t[:n_tr], x, n, device, args.bucket)
    if hasattr(model, "ssm") and hasattr(model.ssm, "set_delta_scale"):
        # Delta_k in units of the median training inter-snapshot gap (audit row 25)
        _ts = np.array([float(s["timestamp"]) for s in tr]); _g = np.diff(_ts); _g = _g[_g > 0]
        model.ssm.set_delta_scale(float(np.median(_g)) if _g.size else 1.0)
        print(f"delta_time_scale={float(model.ssm.delta_scale):g}", flush=True)
    va = snapshots(src[n_tr:n_va], dst[n_tr:n_va], t[n_tr:n_va], x, n, device, args.bucket)
    tt = t.copy()
    if args.time_scale != 1.0:  # stretch/compress gaps within the test period only
        tt[n_va:] = t[n_va - 1] + (t[n_va:] - t[n_va - 1]) * args.time_scale
    te = snapshots(src[n_va:], dst[n_va:], tt[n_va:], x, n, device, args.bucket)
    all_nodes = torch.arange(n, device=device)

    leaky = bool(getattr(args, "leaky", False))
    # Leak-free protocol (audit rows 21/32): snapshot k's events are scored with
    # the output of the forward on snapshot k-1 (attached during training, so
    # the backbone is trained), and snapshot k is ingested only afterwards.

    def loss_fn(out, s):
        pos = model.score_event_pairs(out, s["src"], s["dst"], edge_type=s["edge_types"], query_time=s["edge_timestamps"]) if hasattr(model, "supports_query_time") \
            else model.score_event_pairs(out, s["src"], s["dst"], edge_type=s["edge_types"])
        neg = torch.randint(0, n, (s["src"].numel(), 32), device=device)
        ns = model.score_event_candidates(out, s["src"], neg, edge_type=s["edge_types"], query_time=s["edge_timestamps"]) if hasattr(model, "supports_query_time") \
            else model.score_event_candidates(out, s["src"], neg, edge_type=s["edge_types"])
        return torch.nn.functional.softplus(ns - pos.unsqueeze(-1)).mean()

    def evaluate(snaps, state, record=False):
        model.eval(); rrs, rows = [], []
        seen = set(); last = {}
        for i in range(n_va): seen.add(int(src[i]) * n + int(dst[i])); last[int(src[i])] = float(t[i])
        with torch.no_grad():
            prev_out = None
            for s in snaps:
                if leaky:
                    outs, state = model.forward_sequence([s], initial_state=state); out = outs[0]
                elif prev_out is None:
                    out = {"spatial": state.spatial, "x": s["x"], "node_signal": None}
                else:
                    out = dict(prev_out, node_signal=None)
                cand = all_nodes.view(1, n).expand(s["src"].numel(), n)
                kw = {"query_time": s["edge_timestamps"]} if hasattr(model, "supports_query_time") else {}
                sc = model.score_event_candidates(out, s["src"], cand, edge_type=s["edge_types"], **kw)
                pos = sc.gather(1, s["dst"].view(-1, 1)); sc = sc.scatter(1, s["src"].view(-1, 1), float("-inf"))
                rank = 1 + (sc > pos).sum(-1) + 0.5 * ((sc == pos).sum(-1) - 1)
                rr = (1.0 / rank.float()).cpu().numpy(); rrs.append(rr)
                if record:
                    for j in range(len(rr)):
                        su, dv, tj = int(s["src"][j]), int(s["dst"][j]), float(s["edge_timestamps"][j])
                        rows.append(dict(rr=float(rr[j]), rec=(su * n + dv) in seen, gap=tj - last.get(su, np.nan)))
                        seen.add(su * n + dv); last[su] = tj
                if not leaky:
                    outs, state = model.forward_sequence([s], initial_state=state); prev_out = outs[0]
        model.train()
        return float(np.concatenate(rrs).mean()), state, rows

    best, best_state_dict = -1, None
    for ep in range(args.epochs):
        model.reset_temporal_state(); state = None; t0 = time.perf_counter(); losses = []
        for i in range(0 if leaky else 1, len(tr)):
            outs, new_state = model.forward_sequence([tr[i] if leaky else tr[i - 1]], initial_state=state)
            loss = loss_fn(outs[0] if leaky else dict(outs[0], node_signal=None), tr[i]); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
            state = type(new_state)(memory=new_state.memory.detach(), spatial=new_state.spatial.detach())
        model.reset_temporal_state(); st = None
        for s in tr:  # replay for a clean state
            with torch.no_grad(): _, st = model.forward_sequence([s], initial_state=st)
        vm, st, _ = evaluate(va, st)
        print(f"[{args.variant}] epoch {ep+1} loss {np.mean(losses):.4f} val_mrr {vm:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
        if vm > best: best, best_state_dict = vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state_dict); model.reset_temporal_state(); st = None
    for s in tr + va:
        with torch.no_grad(): _, st = model.forward_sequence([s], initial_state=st)
    tm, _, rows = evaluate(te, st, record=True)
    import pandas as pd
    df = pd.DataFrame(rows); df["gap_q"] = pd.qcut(df.gap.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    summary = dict(variant=args.variant, recurrency=args.recurrency, time_scale=args.time_scale, best_val=best, test_mrr=tm,
                   mrr_novel=float(df[~df.rec].rr.mean()), mrr_recurrent=float(df[df.rec].rr.mean()), frac_novel=float((~df.rec).mean()),
                   mrr_by_gap=df.groupby("gap_q", observed=True).rr.mean().to_dict())
    df.to_csv(os.path.join(args.out, "events.csv"), index=False)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
