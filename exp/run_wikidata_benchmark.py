# Benchmark of the memory-conditioned Temporal Sheaf Diffusion model
# (EventTemporalMambaSheafDiffusion) on tkgl-wikidata, the largest TGB 2.0
# temporal knowledge graph (1.23M nodes, 19.7M temporal edges, 1192 relations).
#
# Scale-aware protocol:
#   - yearly snapshots (native timestamps), sparse local diffusion per snapshot;
#   - training with bptt_steps=1 so at most one snapshot's graph is alive;
#   - optional train-suffix cap (most recent years) and val/test prefix caps;
#   - quick capped-val tracking during training; one final full official
#     val/test evaluation (streaming, dst-time-filtered negatives, TGB MRR).
#
# Run:
#   python exp/run_wikidata_benchmark.py --epochs 3 --out results/wikidata_benchmark

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

DATASET = "tkgl-wikidata"


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


def _make_model(edge_index, node_features, num_nodes, device, args, num_relations):
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
    return EventTemporalMambaSheafDiffusion(edge_index, model_args).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-edges-cap", type=int, default=None,
                        help="Use only the most recent N train edges (suffix cap). None = all 13.97M.")
    parser.add_argument("--val-edges-cap", type=int, default=None,
                        help="Prefix cap on validation edges for the FINAL eval. None = full official split.")
    parser.add_argument("--test-edges-cap", type=int, default=None,
                        help="Prefix cap on test edges for the FINAL eval. None = full official split.")
    parser.add_argument("--track-val-edges", type=int, default=100_000,
                        help="Prefix cap on validation edges for per-epoch tracking/selection.")
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
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--out", default="results/wikidata_benchmark")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sha = _repo_sha()
    print(f"device={device} sha={sha[:12]} config={vars(args)}")

    t0 = time.perf_counter()
    spec, dataset, temporal_data = bu.load_temporal_data(DATASET)
    assert spec.task_family == "tkg" and spec.metric_name == "mrr"
    num_nodes = max(int(temporal_data.src.max()), int(temporal_data.dst.max())) + 1
    num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    node_features = bu.make_node_features(dataset, num_nodes).to(device)

    train_ids, val_ids, test_ids, split_source = bu.split_edge_ids(dataset, temporal_data)
    assert split_source == "official_tgb_masks", split_source

    if args.train_edges_cap is not None:
        train_ids = train_ids[-args.train_edges_cap:]
    final_val_ids = val_ids if args.val_edges_cap is None else val_ids[: args.val_edges_cap]
    final_test_ids = test_ids if args.test_edges_cap is None else test_ids[: args.test_edges_cap]
    track_val_ids = val_ids[: args.track_val_edges]

    def snapshots_for(ids):
        return bu.build_snapshots(temporal_data, ids, node_features)

    train_snapshots = snapshots_for(train_ids)
    track_val_snapshots = snapshots_for(track_val_ids)
    final_val_snapshots = snapshots_for(final_val_ids)
    final_test_snapshots = snapshots_for(final_test_ids)
    print(
        f"snapshots: train={len(train_snapshots)} (edges={train_ids.numel()}) "
        f"track_val={len(track_val_snapshots)} final_val={len(final_val_snapshots)} "
        f"final_test={len(final_test_snapshots)} | prep_sec={time.perf_counter() - t0:.1f}"
    )

    # Load the (14GB each) negative-sample pickles once; evaluate_model_streaming
    # would otherwise re-read them from disk on every evaluation call.
    t_ns = time.perf_counter()
    dataset.load_val_ns()
    dataset.load_val_ns = lambda: None
    dataset.load_test_ns()
    dataset.load_test_ns = lambda: None
    print(f"negative-sample sets loaded in {time.perf_counter() - t_ns:.1f}s", flush=True)

    _seed_everything(args.seed)
    model = _make_model(train_snapshots[0].edge_index, node_features, num_nodes, device, args, num_relations)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"parameter_count={param_count} num_relations={num_relations} num_nodes={num_nodes}")

    history = []
    best = {"val": float("-inf"), "epoch": -1, "state_dict": None}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):
        model.reset_temporal_state()
        t_epoch = time.perf_counter()
        train_loss = bu.run_epoch(
            spec, model, optimizer, train_snapshots, dataset,
            bptt_steps=args.bptt_steps, show_progress=False,
        )
        train_sec = time.perf_counter() - t_epoch

        t_eval = time.perf_counter()
        train_state = bu.advance_context(model, train_snapshots)
        track_val_mrr, track_val_loss, _ = bu.evaluate_model_streaming(
            spec, dataset, track_val_snapshots, model,
            initial_state=train_state, split_mode="val",
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

        if np.isfinite(track_val_mrr) and track_val_mrr > best["val"]:
            best = {
                "val": float(track_val_mrr),
                "epoch": epoch + 1,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    result = {
        "dataset": DATASET,
        "model": "EventTemporalMambaSheaf (memory-conditioned)",
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
        )
        print(f"final val_mrr={val_mrr:.4f} ({time.perf_counter() - t_final:.1f}s so far)", flush=True)
        test_mrr, test_loss, _ = bu.evaluate_model_streaming(
            spec, dataset, final_test_snapshots, model,
            initial_state=val_state, split_mode="test",
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
        print(f"FINAL val_mrr={val_mrr:.4f} test_mrr={test_mrr:.4f} eval_sec={final_sec:.1f}", flush=True)

    pd.DataFrame([result]).to_csv(os.path.join(args.out, "results.csv"), index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "config_json"}, indent=2))


if __name__ == "__main__":
    main()
