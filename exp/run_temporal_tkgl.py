#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp.parser import get_parser, str2bool
from exp import temporal_benchmark_utils as benchmark_utils

try:
    import git
except Exception:  # pragma: no cover - optional dependency at runtime.
    git = None


def parse_args():
    parser = get_parser()
    parser.add_argument(
        "--temporal_include_static_context",
        dest="temporal_include_static_context",
        type=str2bool,
        default=True,
        help=(
            "Append dataset static edges to every temporal snapshot when available. "
            "This is especially useful for tkgl-smallpedia."
        ),
    )
    return parser.parse_args()


def _repo_sha():
    if git is not None:
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except Exception:
            pass

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _infer_num_nodes(dataset, temporal_data):
    candidates = []

    if hasattr(dataset, "num_nodes") and dataset.num_nodes is not None:
        candidates.append(int(dataset.num_nodes))

    if getattr(temporal_data, "src", None) is not None and temporal_data.src.numel() > 0:
        candidates.append(int(temporal_data.src.max().item()) + 1)
    if getattr(temporal_data, "dst", None) is not None and temporal_data.dst.numel() > 0:
        candidates.append(int(temporal_data.dst.max().item()) + 1)

    static_data = getattr(dataset, "static_data", None)
    if static_data:
        head = static_data.get("head")
        tail = static_data.get("tail")
        if head is not None and head.numel() > 0:
            candidates.append(int(head.max().item()) + 1)
        if tail is not None and tail.numel() > 0:
            candidates.append(int(tail.max().item()) + 1)

    if not candidates:
        raise ValueError("Could not infer num_nodes from the temporal or static dataset contents.")

    return max(candidates)


def _validate_edge_index_bounds(edge_index, num_nodes, label):
    if edge_index is None or edge_index.numel() == 0:
        return
    edge_min = int(edge_index.min().item())
    edge_max = int(edge_index.max().item())
    if edge_min < 0 or edge_max >= int(num_nodes):
        raise ValueError(
            f"{label} contains node ids outside the configured graph size. "
            f"graph_size={num_nodes} edge_min={edge_min} edge_max={edge_max}"
        )


def _state_norm(state, field):
    if state is None:
        return float("nan")
    value = getattr(state, field, None)
    if value is None:
        return float("nan")
    return float(value.norm().detach().cpu())


def _snapshot_summary(split_name, snapshots, edge_ids, dataset_spec, dataset, temporal_data):
    supervision = benchmark_utils.summarize_supervision(dataset_spec, dataset, snapshots)
    active_counts = [int(snapshot.active_nodes.numel()) for snapshot in snapshots]
    edge_counts = [int(snapshot.src.numel()) for snapshot in snapshots]
    timestamps = [int(snapshot.timestamp.item()) for snapshot in snapshots]
    selected_ts = temporal_data.t[edge_ids]
    raw_unique_timestamps = int(torch.unique(selected_ts).numel()) if edge_ids.numel() > 0 else 0

    edge_type_values = []
    for snapshot in snapshots:
        if snapshot.edge_types is not None:
            edge_type_values.append(snapshot.edge_types)
    unique_edge_types = int(torch.unique(torch.cat(edge_type_values)).numel()) if edge_type_values else 0

    row = {
        "split": split_name,
        "edge_ids_used": int(edge_ids.numel()),
        "raw_unique_timestamps": raw_unique_timestamps,
        "snapshots": len(snapshots),
        "compression_ratio": (len(snapshots) / raw_unique_timestamps) if raw_unique_timestamps else float("nan"),
        "edges_per_snapshot_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "edges_per_snapshot_max": int(max(edge_counts)) if edge_counts else 0,
        "active_nodes_mean": float(np.mean(active_counts)) if active_counts else float("nan"),
        "active_nodes_max": int(max(active_counts)) if active_counts else 0,
        "timestamp_start": min(timestamps) if timestamps else None,
        "timestamp_end": max(timestamps) if timestamps else None,
        "unique_edge_types": unique_edge_types,
    }
    row.update(supervision)
    return row


def main():
    args = parse_args()

    if args.model not in ("MambaSheaf", "TemporalMambaSheaf", "SparseTemporalMambaSheaf", "EventTemporalMambaSheaf"):
        raise ValueError("run_temporal_tkgl.py only supports temporal Mamba sheaf variants.")

    requested_dataset_name = args.temporal_dataset
    dataset_spec, dataset, temporal_data = benchmark_utils.load_temporal_data(requested_dataset_name)
    if dataset_spec.task_family != "tkg":
        raise ValueError(
            f"run_temporal_tkgl.py only supports tkgl-* datasets, got {requested_dataset_name!r} "
            f"(task_family={dataset_spec.task_family!r})."
        )

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    num_nodes = _infer_num_nodes(dataset, temporal_data)
    output_dim = num_nodes
    args.temporal_num_relations = int(getattr(dataset, "num_rels", 0) or 0)
    node_features = benchmark_utils.make_node_features(dataset, num_nodes).to(device)
    static_edge_index = (
        benchmark_utils.make_static_edge_index(dataset)
        if args.temporal_include_static_context
        else None
    )
    _validate_edge_index_bounds(static_edge_index, num_nodes, "static_edge_index")

    use_event_model = args.model == "EventTemporalMambaSheaf"
    snapshot_static_edge_index = None if use_event_model else static_edge_index

    train_edge_ids, val_edge_ids, test_edge_ids, split_source = benchmark_utils.split_edge_ids(
        dataset,
        temporal_data,
    )

    train_snapshots, train_edge_ids = benchmark_utils.build_supervised_snapshots(
        dataset_spec,
        dataset,
        temporal_data,
        train_edge_ids,
        node_features,
        args.temporal_train_edges,
        args.temporal_snapshot_time_window,
        split_name="train",
        static_edge_index=snapshot_static_edge_index,
    )
    val_snapshots, val_edge_ids = benchmark_utils.build_supervised_snapshots(
        dataset_spec,
        dataset,
        temporal_data,
        val_edge_ids,
        node_features,
        args.temporal_val_edges,
        args.temporal_snapshot_time_window,
        split_name="validation",
        static_edge_index=snapshot_static_edge_index,
    )
    test_snapshots, test_edge_ids = benchmark_utils.build_supervised_snapshots(
        dataset_spec,
        dataset,
        temporal_data,
        test_edge_ids,
        node_features,
        args.temporal_test_edges,
        args.temporal_snapshot_time_window,
        split_name="test",
        static_edge_index=snapshot_static_edge_index,
    )

    if not train_snapshots or not val_snapshots:
        raise RuntimeError("Temporal dataset did not produce non-empty train/validation snapshot sequences.")

    _validate_edge_index_bounds(train_snapshots[0].edge_index, num_nodes, "train_snapshots[0].edge_index")

    model_edge_index = train_snapshots[0].edge_index
    if use_event_model and static_edge_index is not None and static_edge_index.numel() > 0:
        model_edge_index = static_edge_index

    model = benchmark_utils.make_model(
        model_edge_index,
        node_features,
        num_nodes,
        output_dim,
        device,
        args,
    )
    model_output_dim = int(getattr(model, "output_dim", output_dim))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    metric_name = getattr(dataset, "eval_metric", dataset_spec.metric_name)
    history = []
    best_val_metric = float("-inf")
    best_test_metric = float("nan")
    best_epoch = -1
    best_state_dict = None

    start = time.perf_counter()
    for epoch in tqdm(range(args.temporal_epochs), desc="Temporal tkgl training"):
        model.reset_temporal_state()
        train_loss = benchmark_utils.run_epoch(
            dataset_spec,
            model,
            optimizer,
            train_snapshots,
            dataset,
            bptt_steps=args.temporal_bptt_steps,
            show_progress=args.temporal_epoch_progress_bar,
            epoch_label=f"Epoch {epoch + 1}/{args.temporal_epochs}",
            tqdm_factory=tqdm,
        )

        should_eval = ((epoch + 1) % max(args.temporal_eval_every, 1) == 0) or (epoch == args.temporal_epochs - 1)
        train_metric = float("nan")
        val_metric = float("nan")
        test_metric = float("nan")
        train_label_loss = float("nan")
        val_label_loss = float("nan")
        test_label_loss = float("nan")
        train_state_memory_norm = float("nan")
        train_state_spatial_norm = float("nan")
        val_state_memory_norm = float("nan")
        val_state_spatial_norm = float("nan")
        test_state_memory_norm = float("nan")
        test_state_spatial_norm = float("nan")

        if should_eval:
            if args.temporal_skip_train_eval:
                train_state = benchmark_utils.advance_context(model, train_snapshots)
            else:
                train_metric, train_label_loss, train_state = benchmark_utils.evaluate_model_streaming(
                    dataset_spec,
                    dataset,
                    train_snapshots,
                    model,
                    initial_state=None,
                    split_mode="train",
                )

            val_metric, val_label_loss, val_state = benchmark_utils.evaluate_model_streaming(
                dataset_spec,
                dataset,
                val_snapshots,
                model,
                initial_state=train_state,
                split_mode="val",
            )
            test_metric, test_label_loss, test_state = benchmark_utils.evaluate_model_streaming(
                dataset_spec,
                dataset,
                test_snapshots,
                model,
                initial_state=val_state,
                split_mode="test",
            )

            train_state_memory_norm = _state_norm(train_state, "memory")
            train_state_spatial_norm = _state_norm(train_state, "spatial")
            val_state_memory_norm = _state_norm(val_state, "memory")
            val_state_spatial_norm = _state_norm(val_state, "spatial")
            test_state_memory_norm = _state_norm(test_state, "memory")
            test_state_spatial_norm = _state_norm(test_state, "spatial")

            if np.isfinite(val_metric) and val_metric > best_val_metric:
                best_val_metric = float(val_metric)
                best_test_metric = float(test_metric)
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "evaluated": bool(should_eval),
                f"train_{metric_name}": float(train_metric),
                f"val_{metric_name}": float(val_metric),
                f"test_{metric_name}": float(test_metric),
                "train_label_loss_eval": train_label_loss,
                "val_label_loss_eval": val_label_loss,
                "test_label_loss_eval": test_label_loss,
                "train_state_memory_norm": train_state_memory_norm,
                "train_state_spatial_norm": train_state_spatial_norm,
                "val_state_memory_norm": val_state_memory_norm,
                "val_state_spatial_norm": val_state_spatial_norm,
                "test_state_memory_norm": test_state_memory_norm,
                "test_state_spatial_norm": test_state_spatial_norm,
            }
        )

    elapsed = time.perf_counter() - start
    history_df = pd.DataFrame(history)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.reset_temporal_state()
    if args.temporal_skip_train_eval:
        final_train_metric = float("nan")
        final_train_loss = float("nan")
        final_train_state = benchmark_utils.advance_context(model, train_snapshots)
    else:
        final_train_metric, final_train_loss, final_train_state = benchmark_utils.evaluate_model_streaming(
            dataset_spec,
            dataset,
            train_snapshots,
            model,
            initial_state=None,
            split_mode="train",
        )

    final_val_metric, final_val_loss, final_val_state = benchmark_utils.evaluate_model_streaming(
        dataset_spec,
        dataset,
        val_snapshots,
        model,
        initial_state=final_train_state,
        split_mode="val",
    )
    final_test_metric, final_test_loss, final_test_state = benchmark_utils.evaluate_model_streaming(
        dataset_spec,
        dataset,
        test_snapshots,
        model,
        initial_state=final_val_state,
        split_mode="test",
    )

    split_summary_df = pd.DataFrame(
        [
            _snapshot_summary("train", train_snapshots, train_edge_ids, dataset_spec, dataset, temporal_data),
            _snapshot_summary("val", val_snapshots, val_edge_ids, dataset_spec, dataset, temporal_data),
            _snapshot_summary("test", test_snapshots, test_edge_ids, dataset_spec, dataset, temporal_data),
        ]
    )

    num_rels = getattr(dataset, "num_rels", None)
    edge_type = getattr(temporal_data, "edge_type", None)
    config_df = pd.DataFrame(
        [
            {"field": "requested_dataset_name", "value": requested_dataset_name},
            {"field": "dataset_name", "value": dataset_spec.loader_name},
            {"field": "task_family", "value": dataset_spec.task_family},
            {"field": "eval_metric", "value": metric_name},
            {"field": "device", "value": str(device)},
            {"field": "seed", "value": args.seed},
            {"field": "repo_sha", "value": _repo_sha()},
            {"field": "split_source", "value": split_source},
            {"field": "num_nodes", "value": num_nodes},
            {"field": "output_dim", "value": model_output_dim},
            {"field": "num_relations", "value": int(num_rels) if num_rels is not None else 0},
            {"field": "edge_type_present", "value": bool(edge_type is not None)},
            {
                "field": "temporal_include_static_context",
                "value": bool(args.temporal_include_static_context),
            },
            {
                "field": "static_edge_count",
                "value": int(static_edge_index.size(1) // 2) if static_edge_index is not None else 0,
            },
            {"field": "temporal_snapshot_time_window", "value": args.temporal_snapshot_time_window},
            {"field": "temporal_bptt_steps", "value": args.temporal_bptt_steps},
            {"field": "temporal_eval_every", "value": args.temporal_eval_every},
            {"field": "temporal_skip_train_eval", "value": args.temporal_skip_train_eval},
            {"field": "temporal_train_negatives_per_pos", "value": args.temporal_train_negatives_per_pos},
            {"field": "temporal_candidate_chunk_size", "value": args.temporal_candidate_chunk_size},
            {"field": "temporal_epochs", "value": args.temporal_epochs},
            {"field": "train_edges_selected", "value": int(train_edge_ids.numel())},
            {"field": "val_edges_selected", "value": int(val_edge_ids.numel())},
            {"field": "test_edges_selected", "value": int(test_edge_ids.numel())},
        ]
    )

    model_config_df = pd.DataFrame(
        [
            {"field": "hidden_channels", "value": benchmark_utils._env_int("TGB_HIDDEN_CHANNELS", 8)},
            {"field": "layers", "value": benchmark_utils._env_int("TGB_LAYERS", 2)},
            {"field": "dropout", "value": float(os.environ.get("TGB_DROPOUT", 0.0))},
            {
                "field": "temporal_d_model",
                "value": args.temporal_d_model or benchmark_utils._env_int("TGB_TEMPORAL_D_MODEL", 64),
            },
            {"field": "input_dim", "value": int(node_features.size(1))},
            {"field": "graph_size", "value": int(num_nodes)},
            {"field": "output_dim", "value": int(model_output_dim)},
            {"field": "stateful_temporal", "value": bool(args.stateful_temporal)},
            {"field": "closure_hops", "value": int(args.closure_hops)},
            {"field": "train_negatives_per_pos", "value": int(getattr(model, "train_negatives_per_pos", 0))},
            {"field": "candidate_chunk_size", "value": int(getattr(model, "candidate_chunk_size", 0))},
            {"field": "optimizer", "value": "Adam"},
            {"field": "optimizer_lr", "value": optimizer.param_groups[0]["lr"]},
            {"field": "optimizer_weight_decay", "value": optimizer.param_groups[0]["weight_decay"]},
        ]
    )

    final_diagnostics_df = pd.DataFrame(
        [
            {
                "split": "train",
                metric_name: final_train_metric,
                "label_loss": final_train_loss,
                "num_snapshots": len(train_snapshots),
                "state_memory_norm": _state_norm(final_train_state, "memory"),
                "state_spatial_norm": _state_norm(final_train_state, "spatial"),
            },
            {
                "split": "val",
                metric_name: final_val_metric,
                "label_loss": final_val_loss,
                "num_snapshots": len(val_snapshots),
                "state_memory_norm": _state_norm(final_val_state, "memory"),
                "state_spatial_norm": _state_norm(final_val_state, "spatial"),
            },
            {
                "split": "test",
                metric_name: final_test_metric,
                "label_loss": final_test_loss,
                "num_snapshots": len(test_snapshots),
                "state_memory_norm": _state_norm(final_test_state, "memory"),
                "state_spatial_norm": _state_norm(final_test_state, "spatial"),
            },
        ]
    )

    results_dir = Path("results") / f"{dataset_spec.loader_name}_{args.model}_{int(time.time())}"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(results_dir / "config.csv", index=False)
    model_config_df.to_csv(results_dir / "model_config.csv", index=False)
    split_summary_df.to_csv(results_dir / "split_summary.csv", index=False)
    history_df.to_csv(results_dir / "history.csv", index=False)
    final_diagnostics_df.to_csv(results_dir / "final_diagnostics.csv", index=False)

    print(
        f"{args.model} on {dataset_spec.loader_name} | split={split_source} "
        f"| best_val_{metric_name}={best_val_metric:.4f} "
        f"| best_test_{metric_name}={best_test_metric:.4f} "
        f"| final_val_{metric_name}={final_val_metric:.4f} "
        f"| final_test_{metric_name}={final_test_metric:.4f} "
        f"| elapsed_sec={elapsed:.2f}"
    )
    print(f"Saved diagnostics to {results_dir}")


if __name__ == "__main__":
    main()
