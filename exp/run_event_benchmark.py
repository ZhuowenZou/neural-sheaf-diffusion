# Generalized event-scoring benchmark for TGB link-style datasets
# (tgbl-*, tkgl-*, thgl-*): original EventTemporalMambaSheafDiffusion vs
# FaithfulEventTemporalSheafDiffusion under a raw-data protocol.
#
# Generalizes exp/run_wikidata_benchmark.py (kept intact for provenance):
#   - --dataset picks any mrr-metric TGB dataset;
#   - --time-window buckets snapshots in native time units (None = one
#     snapshot per distinct timestamp, i.e. fully raw);
#   - the model's static context graph is the union of the first
#     --context-edges train edges (exact-timestamp datasets have near-empty
#     first snapshots, so "first snapshot" is not a usable context);
#   - training negatives are restricted to the observed destination id range
#     (bipartite datasets like tgbl-wiki never link outside it);
#   - temporal state is reset before every context replay so the module-level
#     previous-timestamp cache cannot leak across streams;
#   - optional --patience early stopping on the tracking-val MRR.
#
# Run:
#   python -m exp.run_event_benchmark --dataset tkgl-smallpedia --model faithful \
#       --epochs 30 --out results/event_bench/smallpedia_faithful

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp import temporal_benchmark_utils as bu


def _repo_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).strip()
    except Exception:
        return "unknown"


def _seed_everything(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _context_edge_index(temporal_data, train_ids, num_edges):
    ids = train_ids[:num_edges] if num_edges and num_edges > 0 else train_ids
    src = temporal_data.src[ids]
    dst = temporal_data.dst[ids]
    return torch.stack([src, dst]).long()


def _make_model(edge_index, node_features, num_nodes, device, args, num_relations, node_types=None):
    from models.faithful_event_model import FaithfulEventTemporalSheafDiffusion
    from models.sparse_temporal_mamba import EventTemporalMambaSheafDiffusion

    model_args = {
        "d": args.d,
        "add_lp": False,
        "add_hp": False,
        "device": device,
        "graph_size": num_nodes,
        "layers": args.layers,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "sparse_learner": False,
        "use_act": True,
        "input_dim": node_features.size(1),
        "hidden_channels": args.hidden_channels,
        "output_dim": num_nodes,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
        "stateful_temporal": False,
        "closure_hops": args.closure_hops,
        "temporal_d_model": args.temporal_d_model,
        "num_relations": num_relations,
        "train_negatives_per_pos": args.train_negatives_per_pos,
        "candidate_chunk_size": args.candidate_chunk_size,
        "max_score_elements": args.max_score_elements,
    }
    edge_index = bu._normalize_sheaf_edge_index(edge_index).to(device)
    if args.model == "faithful":
        model_args.update(
            {
                "feedback_dim": args.feedback_dim,
                "memory_readout": not args.no_memory_readout,
                "sheaf_conditioning": args.sheaf_conditioning,
                "learnable_node_features": args.learn_node_emb,
                "relation_in_input": args.relation_in_input,
                "recurrency_decoder": args.recurrency_decoder,
                "recurrency_untyped": args.recurrency_untyped,
                "recurrency_symmetric": args.recurrency_symmetric,
                "sheaf_identity": args.sheaf_identity,
                "no_delta_t": args.no_delta_t,
                "no_memory": args.no_memory,
                "embeddings_in_head": args.emb_in_head,
                "node_types": node_types,
            }
        )
        return FaithfulEventTemporalSheafDiffusion(edge_index, model_args).to(device)
    return EventTemporalMambaSheafDiffusion(edge_index, model_args).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Any mrr-metric TGB dataset (tgbl-*, tkgl-*, thgl-*)")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--model", choices=["original", "faithful"], default="original")
    parser.add_argument("--feedback-dim", type=int, default=16)
    parser.add_argument("--no-memory-readout", action="store_true")
    parser.add_argument("--sheaf-conditioning", choices=["history", "current_only"], default="history")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=None,
                        help="Stop after this many epochs without tracking-val improvement.")
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--time-window", type=float, default=None,
                        help="Snapshot bucket width in native time units (None = exact timestamps).")
    parser.add_argument("--context-edges", type=int, default=50_000,
                        help="Model context graph = union of the first N train edges (-1 = all).")
    parser.add_argument("--train-edges-cap", type=int, default=None,
                        help="Use only the most recent N train edges (suffix cap). None = all.")
    parser.add_argument("--val-edges-cap", type=int, default=None)
    parser.add_argument("--test-edges-cap", type=int, default=None)
    parser.add_argument("--track-val-edges", type=int, default=100_000,
                        help="Prefix cap on validation edges for per-epoch tracking/selection.")
    parser.add_argument("--train-loss", choices=["softplus", "ce"], default="softplus",
                        help="Training objective: pairwise softplus or sampled-softmax CE.")
    parser.add_argument("--learn-node-emb", action="store_true",
                        help="Faithful model only: add a learnable per-node input embedding.")
    parser.add_argument("--emb-in-head", action="store_true",
                        help="EMB-head/TYPE-head: the scoring head encodes x + learnable node/node-type embeddings for every node "
                             "(by default embeddings reach the head only via the state of nodes that were active).")
    parser.add_argument("--node-type-emb", action="store_true",
                        help="Faithful model: add learned node-type embeddings (thgl-*).")
    parser.add_argument("--predict-from-previous", action="store_true",
                        help="Leak-free protocol (train + eval): score snapshot k from the state after snapshot k-1, then update.")
    parser.add_argument("--reserve-gpu-mb", type=int, default=None,
                        help="Pre-reserve this much GPU memory at start-up (default: env TSD_RESERVE_GPU_MB, set by wait_launch.sh).")
    parser.add_argument("--delta-time-scale", default="auto",
                        help="Units of the physical gap Delta_k: 'auto' = median inter-snapshot gap of the training stream; "
                             "a positive float otherwise (1.0 = raw timestamps, the pre-audit behaviour).")
    parser.add_argument("--sheaf-identity", action="store_true",
                        help="Ablation: restriction maps fixed to identity (plain diffusion).")
    parser.add_argument("--no-delta-t", action="store_true",
                        help="Ablation: step-size selector ignores the physical gap Delta_k.")
    parser.add_argument("--no-memory", action="store_true",
                        help="Ablation: the persistent per-node SSM memory is never read (Z0 = P_z[x; 0], sheaf "
                             "conditioned on the current input). Combine with --layers 0 for a core-off model.")
    parser.add_argument("--recurrency-symmetric", action="store_true",
                        help="Add an order-agnostic (min,max) pair recurrency channel.")
    parser.add_argument("--recurrency-untyped", action="store_true",
                        help="Add an untyped (s,o) recurrency channel (repeat memory across event types).")
    parser.add_argument("--cache-warm-history", action="store_true",
                        help="With --train-edges-cap: pre-commit the pre-suffix train facts to the recurrency cache.")
    parser.add_argument("--recurrency-decoder", action="store_true",
                        help="Faithful model: add a causal (s,r,o) recurrency cache bonus to event scores.")
    parser.add_argument("--relation-in-input", action="store_true",
                        help="Faithful model: append incident-relation aggregate to the SSM input q.")
    parser.add_argument("--no-dst-range", action="store_true",
                        help="Do not restrict training negatives to the observed destination range.")
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--temporal-d-model", type=int, default=64)
    parser.add_argument("--closure-hops", type=int, default=1)
    parser.add_argument("--train-negatives-per-pos", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    parser.add_argument("--max-score-elements", type=int, default=4_000_000)
    parser.add_argument("--bptt-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max grad norm (<=0 disables). The original dense tgbl runs diverged without it.")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Save the best (tracking-val) state_dict to <out>/best.pt.")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip training; load this state_dict and run the final evaluation only.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bu.reserve_gpu_memory(device, mib=args.reserve_gpu_mb)
    sha = _repo_sha()
    print(f"device={device} sha={sha[:12]} config={vars(args)}", flush=True)

    t0 = time.perf_counter()
    spec, dataset, temporal_data = bu.load_temporal_data(args.dataset)
    assert spec.metric_name == "mrr", f"event benchmark needs an mrr dataset, got {spec.metric_name}"
    num_nodes = max(int(temporal_data.src.max()), int(temporal_data.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    node_features = bu.make_node_features(dataset, num_nodes).to(device)

    train_ids, val_ids, test_ids, split_source = bu.split_edge_ids(dataset, temporal_data)
    print(f"split_source={split_source}", flush=True)

    dst_range = None
    if not args.no_dst_range:
        dst_lo = int(temporal_data.dst.min())
        dst_hi = int(temporal_data.dst.max())
        if (dst_lo, dst_hi) != (0, num_nodes - 1):
            dst_range = (dst_lo, dst_hi)
    print(f"destination_range={dst_range}", flush=True)

    history_ids = None
    if args.train_edges_cap is not None:
        history_ids = train_ids[:-args.train_edges_cap]
        train_ids = train_ids[-args.train_edges_cap:]
    context_edge_index = _context_edge_index(temporal_data, train_ids, args.context_edges)
    final_val_ids = val_ids if args.val_edges_cap is None else val_ids[: args.val_edges_cap]
    final_test_ids = test_ids if args.test_edges_cap is None else test_ids[: args.test_edges_cap]
    track_val_ids = val_ids[: args.track_val_edges]

    def snapshots_for(ids):
        return bu.build_snapshots(temporal_data, ids, node_features, time_window=args.time_window)

    train_snapshots = snapshots_for(train_ids)
    track_val_snapshots = snapshots_for(track_val_ids)
    final_val_snapshots = snapshots_for(final_val_ids)
    final_test_snapshots = snapshots_for(final_test_ids)
    print(
        f"snapshots: train={len(train_snapshots)} (edges={train_ids.numel()}) "
        f"track_val={len(track_val_snapshots)} final_val={len(final_val_snapshots)} "
        f"final_test={len(final_test_snapshots)} | prep_sec={time.perf_counter() - t0:.1f}",
        flush=True,
    )

    # Load negative-sample pickles once; evaluate_model_streaming would
    # otherwise re-read them from disk on every evaluation call.
    t_ns = time.perf_counter()
    dataset.load_val_ns()
    dataset.load_val_ns = lambda: None
    dataset.load_test_ns()
    dataset.load_test_ns = lambda: None
    print(f"negative-sample sets loaded in {time.perf_counter() - t_ns:.1f}s", flush=True)

    _seed_everything(args.seed)
    node_types = None
    if args.node_type_emb:
        node_types = getattr(dataset, "node_type", None)
        if node_types is not None:
            node_types = torch.as_tensor(node_types)[:num_nodes]
        print(f"node_types: {'present' if node_types is not None else 'MISSING on dataset'}", flush=True)
    model = _make_model(context_edge_index, node_features, num_nodes, device, args, num_relations, node_types=node_types)
    if dst_range is not None:
        model.event_destination_range = dst_range
    model.event_loss_type = args.train_loss
    if hasattr(model, "ssm") and hasattr(model.ssm, "set_delta_scale"):
        if str(args.delta_time_scale).lower() == "auto":
            _ts = np.array([float(sn.timestamp) for sn in train_snapshots], dtype=np.float64)
            _gaps = np.diff(_ts)
            _gaps = _gaps[_gaps > 0]
            delta_scale = float(np.median(_gaps)) if _gaps.size else 1.0
        else:
            delta_scale = float(args.delta_time_scale)
        model.ssm.set_delta_scale(delta_scale)
        print(f"delta_time_scale={delta_scale:g} (Delta_k is measured in units of the median training inter-snapshot gap)", flush=True)
    if args.predict_from_previous:
        os.environ["TSD_SCORE_FROM_PREVIOUS_STATE"] = "1"
        print("protocol: predict-from-previous-state (leak-free) for training and evaluation", flush=True)
    if args.cache_warm_history and history_ids is not None and history_ids.numel() > 0 and args.model == "faithful":
        et = getattr(temporal_data, "edge_type", None)
        model.set_recurrency_history(
            temporal_data.src[history_ids], temporal_data.dst[history_ids],
            None if et is None else et[history_ids], temporal_data.t[history_ids],
        )
        model.reset_temporal_state()
        print(f"recurrency cache warmed with {history_ids.numel()} pre-suffix facts "
              f"({model._rec_keys.numel() if model._rec_keys is not None else 0} distinct triples)", flush=True)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"parameter_count={param_count} num_relations={num_relations} num_nodes={num_nodes}", flush=True)

    if args.grad_clip and args.grad_clip > 0:
        original_step = optimizer.step

        def clipped_step(*step_args, **step_kwargs):
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gnorm):
                optimizer.zero_grad()
                return None
            return original_step(*step_args, **step_kwargs)

        optimizer.step = clipped_step

    history = []
    best = {"val": float("-inf"), "epoch": -1, "state_dict": None}
    epochs_since_best = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.eval_only:
        state = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(state)
        print(f"eval-only: loaded {args.eval_only}", flush=True)
    for epoch in range(0 if args.eval_only else args.epochs):
        model.reset_temporal_state()
        t_epoch = time.perf_counter()
        train_loss = bu.run_epoch(
            spec, model, optimizer, train_snapshots, dataset,
            bptt_steps=args.bptt_steps, show_progress=False,
        )
        train_sec = time.perf_counter() - t_epoch

        t_eval = time.perf_counter()
        model.reset_temporal_state()
        train_state = bu.advance_context(model, train_snapshots)
        track_val_mrr, track_val_loss, _ = bu.evaluate_model_streaming(
            spec, dataset, track_val_snapshots, model,
            initial_state=train_state, split_mode="val",
            label_cursor_after_ts=bu.last_snapshot_timestamp(train_snapshots),
        )
        eval_sec = time.perf_counter() - t_eval
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "track_val_mrr": track_val_mrr,
            "track_val_loss": track_val_loss,
            "train_sec": round(train_sec, 1),
            "track_eval_sec": round(eval_sec, 1),
            "peak_gpu_mem_mb": round(peak_mb, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(args.out, "history.csv"), index=False)
        print(f"[epoch {epoch + 1}/{args.epochs}] {row}", flush=True)

        # A diverged model scores NaN on every candidate and the TGB evaluator
        # then pins MRR near 1.0 — require a finite eval loss before trusting
        # the metric for checkpoint selection.
        if np.isfinite(track_val_mrr) and np.isfinite(track_val_loss) and track_val_mrr > best["val"]:
            best = {
                "val": float(track_val_mrr),
                "epoch": epoch + 1,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if (
            args.patience is not None
            and epoch + 1 >= args.min_epochs
            and epochs_since_best >= args.patience
        ):
            print(f"early stop at epoch {epoch + 1} (patience {args.patience})", flush=True)
            break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
        if args.save_checkpoint:
            torch.save(best["state_dict"], os.path.join(args.out, "best.pt"))
            print(f"checkpoint saved: {os.path.join(args.out, 'best.pt')}", flush=True)

    result = {
        "dataset": args.dataset,
        "model": ("FaithfulEventTemporalSheaf" if args.model == "faithful"
                  else "EventTemporalMambaSheaf (memory-conditioned)"),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_track_epoch": best["epoch"],
        "best_track_val_mrr": best["val"],
        "parameter_count": param_count,
        "commit_hash": sha,
        "train_edges": int(train_ids.numel()),
        "final_val_edges": int(final_val_ids.numel()),
        "final_test_edges": int(final_test_ids.numel()),
        "config_json": json.dumps(vars(args), sort_keys=True),
    }

    if not args.skip_final_eval:
        model.reset_temporal_state()
        t_final = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_state = bu.advance_context(model, train_snapshots)
        val_mrr, val_loss, val_state = bu.evaluate_model_streaming(
            spec, dataset, final_val_snapshots, model,
            initial_state=train_state, split_mode="val",
            label_cursor_after_ts=bu.last_snapshot_timestamp(train_snapshots),
        )
        print(f"final val_mrr={val_mrr:.4f} ({time.perf_counter() - t_final:.1f}s so far)", flush=True)
        test_mrr, test_loss, _ = bu.evaluate_model_streaming(
            spec, dataset, final_test_snapshots, model,
            initial_state=val_state, split_mode="test",
            label_cursor_after_ts=bu.last_snapshot_timestamp(final_val_snapshots),
        )
        final_sec = time.perf_counter() - t_final
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
        result.update({
            "validation_mrr": float(val_mrr),
            "test_mrr": float(test_mrr),
            "final_val_loss": float(val_loss),
            "final_test_loss": float(test_loss),
            "final_eval_sec": round(final_sec, 1),
            "final_eval_peak_gpu_mem_mb": round(peak_mb, 1),
        })
        test_hits = getattr(bu.evaluate_model_streaming, "last_hits10", float("nan"))
        result["test_hits10"] = float(test_hits)
        print(f"FINAL val_mrr={val_mrr:.4f} test_mrr={test_mrr:.4f} test_hits10={test_hits:.4f} eval_sec={final_sec:.1f}", flush=True)

    pd.DataFrame([result]).to_csv(os.path.join(args.out, "results.csv"), index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "config_json"}, indent=2))


if __name__ == "__main__":
    main()
