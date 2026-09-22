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


def _hardware_record(device):
    import platform
    import subprocess as _sp
    rec = {"hostname": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
           "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    try:
        import torch_geometric; rec["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    try:
        from importlib.metadata import version as _v
        rec["py_tgb"] = _v("py-tgb")
    except Exception:
        pass
    try:
        cpu = [l for l in open("/proc/cpuinfo").read().splitlines() if l.startswith("model name")]
        rec["cpu_model"] = cpu[0].split(":", 1)[1].strip() if cpu else "unknown"
        rec["cpu_cores"] = os.cpu_count()
        mem = [l for l in open("/proc/meminfo").read().splitlines() if l.startswith("MemTotal")]
        rec["ram_gib"] = round(int(mem[0].split()[1]) / 2**20, 1) if mem else float("nan")
    except Exception:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        rec.update({"gpu_model": props.name, "gpu_vram_gib": round(props.total_memory / 2**30, 1),
                    "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "gpu_count_visible": torch.cuda.device_count(),
                    "precision": "float32", "sync": "torch.cuda.synchronize-free wall clock (perf_counter)"})
        try:
            rec["gpu_driver"] = _sp.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                 text=True, timeout=10).split()[0]
            rec["gpu_other_processes_at_start"] = int(_sp.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10).count("\n"))
        except Exception:
            pass
    return {f"hw_{k}": v for k, v in rec.items()}


def _storage_record(model):
    """Persistent-state / cache / process storage after the final evaluation (section 6)."""
    rec = {}
    try:
        import resource
        rec["cpu_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        pass
    keys, bytes_ = 0, 0
    for name, store in getattr(model, "_rec_stores", []):
        n = store.size
        keys += n
        bytes_ += n * (8 + 8 + 8)
        rec[f"rec_keys_{name}"] = n
    rec["rec_keys_total"] = keys
    rec["rec_cache_bytes"] = bytes_
    rec["rec_cache_device"] = str(model._rec_stores[0][1].keys.device) if getattr(model, "_rec_stores", None) and model._rec_stores[0][1].keys is not None else "n/a"
    st = getattr(model, "_temporal_state", None)
    rec["tracked_nodes"] = int(getattr(model, "graph_size", 0))
    rec["core_state_bytes"] = int(model.graph_size * (getattr(model, "d_h", 0) + getattr(model, "hidden_dim", 0)) * 4)
    rec["checkpoint_bytes"] = int(sum(p.numel() * p.element_size() for p in model.state_dict().values()))
    return rec


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
                "backbone": getattr(args, "backbone", "tsd"),
                "spatial": getattr(args, "spatial", None) or ("identity" if args.sheaf_identity else "sheaf"),
                "clock": getattr(args, "clock", "global"),
                "fast_core_off": bool(getattr(args, "fast_core_off", False)),
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
    # ---- review controls (handoff 2026-09-22) ----
    parser.add_argument("--backbone", choices=["tsd", "gru", "diag_ssm"], default="tsd",
                        help="Temporal backbone: the selective ZOH SSM (tsd), a GRU with the same inputs + gap feature, "
                             "or a stable diagonal SSM with fixed B and the same step selector.")
    parser.add_argument("--spatial", choices=["sheaf", "identity", "node_frame", "attention"], default=None,
                        help="Spatial operator: learned incidence maps (sheaf), identity maps, one orthogonal frame per "
                             "node (node_frame), or identity transport with history-conditioned edge gates (attention).")
    parser.add_argument("--clock", choices=["global", "node_update", "node_interaction"], default="global",
                        help="Gap fed to the step selector: global batch gap, time since the node's last memory update, "
                             "or time since the node's last observed interaction (first observation -> gap 0).")
    parser.add_argument("--fast-core-off", action="store_true",
                        help="With --no-memory --layers 0: skip the unused SSM transition / map decoder (verified identical outputs).")
    parser.add_argument("--rng-isolation", action="store_true",
                        help="Draw training negatives from a dedicated generator seeded by (seed, epoch) so paired arms "
                             "see identical negative samples regardless of architecture-dependent RNG consumption.")
    parser.add_argument("--no-query-audit", action="store_true",
                        help="Disable the per-query score-validity audit (query_validity*.csv in --out).")
    parser.add_argument("--clock-diagnostics", action="store_true",
                        help="Record bounded clock/saturation diagnostics during the final evaluation (clock_*.csv in --out).")
    parser.add_argument("--audit-eval-only", action="store_true",
                        help="With --eval-only: label the outputs as a checkpoint replay audit (no training).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_start_all = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reserved_mib = bu.reserve_gpu_memory(device, mib=args.reserve_gpu_mb)
    sha = _repo_sha()
    print(f"device={device} sha={sha[:12]} config={vars(args)}", flush=True)
    with open(os.path.join(args.out, "resolved_config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "args": vars(args), "commit": sha, "cwd": os.getcwd(),
                   "env": {k: v for k, v in os.environ.items() if k.startswith(("TSD_", "TGB_", "CUDA_", "PYTORCH_"))}},
                  fh, indent=1, sort_keys=True)
    hardware = _hardware_record(device)
    bu.reset_train_counters()

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
    prep_sec = time.perf_counter() - t0
    t_ns = time.perf_counter()
    dataset.load_val_ns()
    dataset.load_val_ns = lambda: None
    dataset.load_test_ns()
    dataset.load_test_ns = lambda: None
    negatives_sec = time.perf_counter() - t_ns
    print(f"negative-sample sets loaded in {negatives_sec:.1f}s", flush=True)

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
    component_counts = model.component_parameter_counts() if hasattr(model, "component_parameter_counts") else {}
    print(f"parameters_by_component={component_counts}", flush=True)
    if not args.no_query_audit:
        bu.QUERY_AUDIT["audit"] = bu.QueryValidityAudit()
    if args.clock_diagnostics:
        from models.diagnostics import ClockDiagnostics
        clock_diag = ClockDiagnostics()
    else:
        clock_diag = None

    if args.grad_clip and args.grad_clip > 0:
        original_step = optimizer.step

        def clipped_step(*step_args, **step_kwargs):
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gnorm):
                # counted (review handoff, section 6); the loop records the skipped step
                bu.TRAIN_COUNTERS["nonfinite_grad"] += 1
                optimizer.zero_grad()
                return False
            if float(gnorm) > args.grad_clip:
                bu.TRAIN_COUNTERS["clipped"] += 1
            original_step(*step_args, **step_kwargs)
            return True

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
    epoch_counters = []
    for epoch in range(0 if args.eval_only else args.epochs):
        model.reset_temporal_state()
        if args.rng_isolation:
            bu.install_negative_rng(args.seed, epoch, device)
        before = dict(bu.TRAIN_COUNTERS)
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
            "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1) if device.type == "cuda" else float("nan"),
        }
        row.update({f"epoch_{k}": bu.TRAIN_COUNTERS[k] - before[k] for k in bu.TRAIN_COUNTERS})
        row["track_nonfinite_positives"] = getattr(bu.evaluate_model_streaming, "last_nonfinite_positives", 0)
        row["track_nonfinite_negatives"] = getattr(bu.evaluate_model_streaming, "last_nonfinite_negatives", 0)
        row["track_skipped_queries"] = getattr(bu.evaluate_model_streaming, "last_skipped_queries", 0)
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
        "prep_sec": round(prep_sec, 1),
        "negatives_load_sec": round(negatives_sec, 1),
        "train_epochs_run": len(history),
        "train_sec_total": round(sum(r["train_sec"] for r in history), 1),
        "selection_eval_sec_total": round(sum(r["track_eval_sec"] for r in history), 1),
        "reserved_buffer_mib": int(reserved_mib),
        "backbone": getattr(args, "backbone", "tsd"),
        "spatial": getattr(args, "spatial", None) or ("identity" if args.sheaf_identity else "sheaf"),
        "clock": getattr(args, "clock", "global"),
        "rng_isolation": bool(args.rng_isolation),
        **{f"params_{k}": v for k, v in component_counts.items()},
        **{f"train_{k}": v for k, v in bu.TRAIN_COUNTERS.items()},
        **hardware,
    }

    if not args.skip_final_eval:
        model.reset_temporal_state()
        if clock_diag is not None:
            model.diag = clock_diag
            model.ssm.diag = clock_diag
            clock_diag.split = "train_replay"
        t_final = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_state = bu.advance_context(model, train_snapshots)
        replay_sec = time.perf_counter() - t_final
        if clock_diag is not None:
            clock_diag.split = "val"
        t_val = time.perf_counter()
        val_mrr, val_loss, val_state = bu.evaluate_model_streaming(
            spec, dataset, final_val_snapshots, model,
            initial_state=train_state, split_mode="val",
            label_cursor_after_ts=bu.last_snapshot_timestamp(train_snapshots),
        )
        val_sec = time.perf_counter() - t_val
        val_audit = {k: getattr(bu.evaluate_model_streaming, f"last_{k}", 0)
                     for k in ("nonfinite_positives", "nonfinite_negatives", "skipped_queries", "metric_examples")}
        print(f"final val_mrr={val_mrr:.4f} ({time.perf_counter() - t_final:.1f}s so far)", flush=True)
        if clock_diag is not None:
            clock_diag.split = "test"
        t_test = time.perf_counter()
        test_mrr, test_loss, _ = bu.evaluate_model_streaming(
            spec, dataset, final_test_snapshots, model,
            initial_state=val_state, split_mode="test",
            label_cursor_after_ts=bu.last_snapshot_timestamp(final_val_snapshots),
        )
        test_sec = time.perf_counter() - t_test
        test_audit = {k: getattr(bu.evaluate_model_streaming, f"last_{k}", 0)
                      for k in ("nonfinite_positives", "nonfinite_negatives", "skipped_queries", "metric_examples")}
        final_sec = time.perf_counter() - t_final
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
        result.update({
            "validation_mrr": float(val_mrr),
            "test_mrr": float(test_mrr),
            "final_val_loss": float(val_loss),
            "final_test_loss": float(test_loss),
            "final_eval_sec": round(final_sec, 1),
            "final_replay_sec": round(replay_sec, 1),
            "final_val_sec": round(val_sec, 1),
            "final_test_sec": round(test_sec, 1),
            "final_eval_peak_gpu_mem_mb": round(peak_mb, 1),
            "final_eval_peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1) if device.type == "cuda" else float("nan"),
            **{f"val_{k}": v for k, v in val_audit.items()},
            **{f"test_{k}": v for k, v in test_audit.items()},
            **_storage_record(model),
        })
        test_hits = getattr(bu.evaluate_model_streaming, "last_hits10", float("nan"))
        result["test_hits10"] = float(test_hits)
        print(f"FINAL val_mrr={val_mrr:.4f} test_mrr={test_mrr:.4f} test_hits10={test_hits:.4f} eval_sec={final_sec:.1f}", flush=True)
        if clock_diag is not None:
            model.diag = None
            model.ssm.diag = None
            agg = clock_diag.write(args.out)
            timing = {n: float(p.detach().cpu()) for n, p in model.ssm.named_parameters() if n.startswith("dt_time")}
            with open(os.path.join(args.out, "clock_learned_timing.json"), "w") as fh:
                json.dump({"dt_time_params": timing, "delta_scale": float(model.ssm.delta_scale), "clock": model.clock}, fh, indent=1)
            print(f"clock diagnostics written ({len(agg)} aggregate rows)", flush=True)
    audit = bu.QUERY_AUDIT.get("audit")
    if audit is not None:
        audit.write(args.out, tag=os.path.basename(args.out.rstrip("/")))
        rows = audit.summary_rows()
        for r in rows:
            print(f"[query-audit] split={r['split']} queries={r['queries']} affected={r['queries_affected']} "
                  f"pos_nonfinite={r['pos_nonfinite']} neg_nan={r['neg_nan']} neg_posinf={r['neg_posinf']} "
                  f"neg_neginf={r['neg_neginf']} mrr_raw={r['mrr_tgb_raw']:.6f} mrr_guarded={r['mrr_guarded']:.6f} "
                  f"mrr_conservative={r['mrr_conservative']:.6f} parity={r['parity_raw_vs_guarded'] and r['parity_guarded_vs_conservative']}", flush=True)
        result["query_audit_affected_total"] = int(sum(r["queries_affected"] for r in rows))
        result["query_audit_parity"] = bool(all(r["parity_raw_vs_guarded"] and r["parity_guarded_vs_conservative"] for r in rows))
        bu.QUERY_AUDIT["audit"] = None
    result["end_to_end_sec"] = round(time.perf_counter() - t_start_all, 1)

    pd.DataFrame([result]).to_csv(os.path.join(args.out, "results.csv"), index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "config_json"}, indent=2))


if __name__ == "__main__":
    main()
