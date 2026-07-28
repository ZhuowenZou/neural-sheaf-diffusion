#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp.parser import get_parser
from exp.temporal_utils import build_temporal_snapshots
from models.mamba_models import TemporalEdgeMambaSheafDiffusion
from models.sparse_temporal_mamba import SparseTemporalEdgeMambaSheafDiffusion

try:
    import git
except Exception:  # pragma: no cover - optional dependency at runtime.
    git = None


DATASET_ALIASES = {
    "tgbl-wiki-v2": "tgbl-wiki",
    "tgbl-review-v2": "tgbl-review",
}


def _env_int(name, default):
    return int(os.environ.get(name, default))


def pytest_importorskip(module_name):
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError(
            f"Required temporal dependency {module_name!r} is unavailable. "
            "Install the optional TGB dependencies before running this entry point."
        ) from exc
    return sys.modules[module_name]


def _resolve_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def _load_tgbl_dataset(dataset_name):
    loader_name = _resolve_dataset_name(dataset_name)
    if not loader_name.startswith("tgbl-"):
        raise ValueError(
            f"run_temporal_tgbl.py only supports tgbl-* datasets, got {dataset_name!r}."
        )

    root = os.environ.get("TGB_ROOT", "datasets")
    answer = "y" if os.environ.get("TGB_DOWNLOAD") == "1" else "n"
    tgb_dataset = pytest_importorskip("tgb.linkproppred.dataset_pyg")
    pytest_importorskip("tgb.linkproppred.evaluate")
    with patch("builtins.input", return_value=answer):
        dataset = tgb_dataset.PyGLinkPropPredDataset(name=loader_name, root=root)
    return loader_name, dataset


def _load_temporal_data(dataset_name):
    loader_name, dataset = _load_tgbl_dataset(dataset_name)
    temporal_data = dataset.get_TemporalData()
    return loader_name, dataset, temporal_data


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


def _make_node_features(dataset, num_nodes):
    try:
        if hasattr(dataset, "node_feat") and dataset.node_feat is not None and dataset.node_feat.size(0) == num_nodes:
            return dataset.node_feat.float()
    except (AttributeError, RuntimeError):
        pass

    feature_dim = _env_int("TGB_NODE_FEATURE_DIM", 64)
    generator = torch.Generator().manual_seed(_env_int("TGB_SEED", 43))
    return torch.randn(num_nodes, feature_dim, generator=generator)


def _normalize_sheaf_edge_index(edge_index):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None)
    return edge_index.contiguous()


def _infer_graph_size(temporal_data):
    max_node_id = torch.stack([temporal_data.src.max(), temporal_data.dst.max()]).max()
    return int(max_node_id.item()) + 1


def _destination_spec(temporal_data):
    dst_min = int(temporal_data.dst.min().item())
    dst_max = int(temporal_data.dst.max().item())
    return {
        "offset": dst_min,
        "max": dst_max,
        "size": dst_max - dst_min + 1,
    }


def _make_model(edge_index, x, num_nodes, destination_spec, device, args):
    model_args = {
        "d": _env_int("TGB_SHEAF_D", 2),
        "add_lp": False,
        "add_hp": False,
        "device": device,
        "graph_size": num_nodes,
        "layers": _env_int("TGB_LAYERS", 2),
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": float(os.environ.get("TGB_DROPOUT", 0.0)),
        "left_weights": True,
        "right_weights": True,
        "sparse_learner": False,
        "use_act": True,
        "input_dim": x.size(1),
        "hidden_channels": _env_int("TGB_HIDDEN_CHANNELS", 8),
        "output_dim": int(destination_spec["size"]),
        "destination_offset": int(destination_spec["offset"]),
        "destination_size": int(destination_spec["size"]),
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
        "stateful_temporal": args.stateful_temporal,
        "closure_hops": args.closure_hops,
        "temporal_d_model": args.temporal_d_model or _env_int("TGB_TEMPORAL_D_MODEL", 64),
    }
    model_cls = (
        SparseTemporalEdgeMambaSheafDiffusion
        if args.model == "SparseTemporalMambaSheaf"
        else TemporalEdgeMambaSheafDiffusion
    )
    return model_cls(
        _normalize_sheaf_edge_index(edge_index).to(device),
        model_args,
    ).to(device)


def _split_edge_ids(dataset, temporal_data):
    if all(hasattr(dataset, name) for name in ("train_mask", "val_mask", "test_mask")):
        train_edge_ids = dataset.train_mask.nonzero(as_tuple=False).view(-1)
        val_edge_ids = dataset.val_mask.nonzero(as_tuple=False).view(-1)
        test_edge_ids = dataset.test_mask.nonzero(as_tuple=False).view(-1)
        split_source = "official_tgb_masks"
    else:
        total_edges = len(temporal_data.src)
        train_edge_ids = torch.arange(0, total_edges // 2)
        val_edge_ids = torch.arange(total_edges // 2, (total_edges * 3) // 4)
        test_edge_ids = torch.arange((total_edges * 3) // 4, total_edges)
        split_source = "fallback_contiguous_50_25_25"

    return train_edge_ids, val_edge_ids, test_edge_ids, split_source


def _build_snapshots(temporal_data, edge_ids, node_features, max_edges=None, time_window=None):
    edge_types = getattr(temporal_data, "edge_type", None)
    return build_temporal_snapshots(
        temporal_data.src,
        temporal_data.dst,
        temporal_data.t,
        node_features,
        edge_ids=edge_ids,
        max_edges=max_edges,
        time_window=time_window,
        edge_types=edge_types,
    )


def _trim_negative_samples(neg_samples):
    if neg_samples is None:
        return None
    arrays = [np.asarray(sample, dtype=np.int64) for sample in neg_samples]
    if not arrays:
        return None
    lengths = [len(arr) for arr in arrays]
    if min(lengths) == 0:
        return None
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        arrays = [arr[:min_len] for arr in arrays]
    return np.stack(arrays, axis=0)


def _global_to_local_destination(dst, destination_spec):
    dst_local = dst.long() - int(destination_spec["offset"])
    if dst_local.numel() == 0:
        return dst_local
    if int(dst_local.min().item()) < 0 or int(dst_local.max().item()) >= int(destination_spec["size"]):
        raise IndexError(
            "Encountered destination ids outside the configured destination vocabulary. "
            f"Offset={destination_spec['offset']} size={destination_spec['size']} "
            f"dst_min={int(dst.min().item())} dst_max={int(dst.max().item())}."
        )
    return dst_local


def _assert_finite_tensor(tensor, label):
    if tensor is None:
        return
    if torch.isfinite(tensor).all():
        return

    bad_mask = ~torch.isfinite(tensor)
    bad_count = int(bad_mask.sum().item())
    sample_value = tensor[bad_mask].reshape(-1)[0].item()
    raise ValueError(
        f"Encountered non-finite values in {label}. "
        f"bad_values={bad_count} sample={sample_value!r}"
    )


def _edge_prediction_loss(outputs, snapshots, destination_spec):
    from torch.nn import functional as F

    losses = []
    for logits, snapshot in zip(outputs, snapshots):
        if snapshot.src.numel() == 0:
            continue
        _assert_finite_tensor(logits, "training logits")
        target = _global_to_local_destination(snapshot.dst.to(logits.device), destination_spec)
        losses.append(F.nll_loss(logits[snapshot.src.to(logits.device)], target))
    if not losses:
        return torch.tensor(0.0, device=outputs[0].device if outputs else "cpu")
    mean_loss = torch.stack(losses).mean()
    _assert_finite_tensor(mean_loss, "training loss")
    return mean_loss


def _detach_temporal_state(state):
    if state is None:
        return None
    return type(state)(
        memory=state.memory.detach() if state.memory is not None else None,
        spatial=state.spatial.detach() if state.spatial is not None else None,
    )


def _advance_context(model, snapshots, initial_state=None):
    state = initial_state
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            _, state = model.forward_sequence([snapshot], initial_state=state)
    return state


def _evaluate_model_streaming(
    dataset_name,
    dataset,
    snapshots,
    model,
    destination_spec,
    initial_state=None,
    split_mode=None,
    compute_metric=True,
    compute_loss=True,
):
    from torch.nn import functional as F
    from tgb.linkproppred.evaluate import Evaluator

    evaluator = Evaluator(name=dataset_name) if compute_metric else None
    losses = []
    metric_sum = 0.0
    metric_examples = 0
    state = initial_state

    if compute_metric and split_mode == "val":
        dataset.load_val_ns()
    elif compute_metric and split_mode == "test":
        dataset.load_test_ns()

    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            outputs, state = model.forward_sequence([snapshot], initial_state=state)
            logits = outputs[0]
            split_label = split_mode or "unknown"
            snapshot_timestamp = getattr(snapshot, "timestamp", None)
            timestamp_label = int(snapshot_timestamp.item()) if snapshot_timestamp is not None else "unknown"
            _assert_finite_tensor(logits, f"{split_label} logits at timestamp {timestamp_label}")

            if snapshot.src.numel() == 0:
                continue

            pos_src = snapshot.src.to(logits.device)
            pos_dst_global = snapshot.dst.to(logits.device)
            pos_dst_local = _global_to_local_destination(pos_dst_global, destination_spec)

            if compute_loss:
                losses.append(F.nll_loss(logits[pos_src], pos_dst_local).detach().cpu())

            if not compute_metric or split_mode not in ("val", "test"):
                continue

            pos_ts = snapshot.edge_timestamps
            if pos_ts is None:
                pos_ts = snapshot.timestamp.expand(snapshot.src.numel())
            edge_type = snapshot.edge_types
            neg_samples = dataset.negative_sampler.query_batch(
                snapshot.src.cpu(),
                snapshot.dst.cpu(),
                pos_ts.cpu(),
                edge_type=edge_type.cpu() if edge_type is not None else None,
                split_mode=split_mode,
            )
            neg_samples = _trim_negative_samples(neg_samples)
            if neg_samples is None:
                continue

            neg_dst_global = torch.as_tensor(neg_samples, dtype=torch.long, device=logits.device)
            neg_dst_local = _global_to_local_destination(neg_dst_global, destination_spec)

            y_pred_pos = logits[pos_src, pos_dst_local]
            y_pred_neg = logits[pos_src.view(-1, 1).expand_as(neg_dst_local), neg_dst_local]
            _assert_finite_tensor(y_pred_pos, f"{split_label} positive predictions at timestamp {timestamp_label}")
            _assert_finite_tensor(y_pred_neg, f"{split_label} negative predictions at timestamp {timestamp_label}")
            score = evaluator.eval(
                {
                    "y_pred_pos": y_pred_pos,
                    "y_pred_neg": y_pred_neg,
                    "eval_metric": [dataset.eval_metric],
                }
            )
            metric_val = list(score.values())[0] if isinstance(score, dict) else score
            metric_sum += float(metric_val) * int(pos_src.numel())
            metric_examples += int(pos_src.numel())

    metric = float("nan")
    if compute_metric and metric_examples > 0:
        metric = metric_sum / metric_examples

    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    return metric, mean_loss, state


def _run_epoch(
    model,
    optimizer,
    train_snapshots,
    destination_spec,
    bptt_steps=None,
    show_progress=False,
    epoch_label=None,
):
    model.train()
    if bptt_steps is None or bptt_steps <= 0:
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = _edge_prediction_loss(outputs, train_snapshots, destination_spec)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    chunk_losses = []
    state = None
    chunk_starts = range(0, len(train_snapshots), bptt_steps)
    if show_progress:
        chunk_starts = tqdm(
            chunk_starts,
            total=(len(train_snapshots) + bptt_steps - 1) // bptt_steps,
            desc=epoch_label or "Epoch chunks",
            leave=True,
        )

    for start in chunk_starts:
        chunk = train_snapshots[start:start + bptt_steps]
        optimizer.zero_grad()
        outputs, state = model.forward_sequence(chunk, initial_state=state)
        loss = _edge_prediction_loss(outputs, chunk, destination_spec)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            chunk_losses.append(float(loss.detach().cpu()))

        state = _detach_temporal_state(state)

    if not chunk_losses:
        return 0.0
    return float(sum(chunk_losses) / len(chunk_losses))


def _snapshot_summary(split_name, snapshots, edge_ids, temporal_data, destination_spec):
    edge_counts = [int(snapshot.src.numel()) for snapshot in snapshots]
    active_counts = [int(snapshot.active_nodes.numel()) for snapshot in snapshots]
    timestamps = [int(snapshot.timestamp.item()) for snapshot in snapshots]
    selected_ts = temporal_data.t[edge_ids]
    raw_unique_timestamps = int(torch.unique(selected_ts).numel()) if edge_ids.numel() > 0 else 0
    return {
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
        "destination_offset": int(destination_spec["offset"]),
        "destination_size": int(destination_spec["size"]),
    }


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.model not in ("MambaSheaf", "TemporalMambaSheaf", "SparseTemporalMambaSheaf"):
        raise ValueError("run_temporal_tgbl.py only supports the temporal Mamba sheaf model.")

    requested_dataset_name = args.temporal_dataset
    dataset_name, dataset, temporal_data = _load_temporal_data(requested_dataset_name)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    num_nodes = _infer_graph_size(temporal_data)
    destination_spec = _destination_spec(temporal_data)
    node_features = _make_node_features(dataset, num_nodes).to(device)

    train_edge_ids, val_edge_ids, test_edge_ids, split_source = _split_edge_ids(dataset, temporal_data)
    train_edge_ids = train_edge_ids[:args.temporal_train_edges]
    val_edge_ids = val_edge_ids[:args.temporal_val_edges]
    test_edge_ids = test_edge_ids[:args.temporal_test_edges]

    train_snapshots = _build_snapshots(
        temporal_data,
        train_edge_ids,
        node_features,
        max_edges=args.temporal_max_edges_per_snapshot,
        time_window=args.temporal_snapshot_time_window,
    )
    val_snapshots = _build_snapshots(
        temporal_data,
        val_edge_ids,
        node_features,
        max_edges=args.temporal_max_edges_per_snapshot,
        time_window=args.temporal_snapshot_time_window,
    )
    test_snapshots = _build_snapshots(
        temporal_data,
        test_edge_ids,
        node_features,
        max_edges=args.temporal_max_edges_per_snapshot,
        time_window=args.temporal_snapshot_time_window,
    )

    if not train_snapshots or not val_snapshots:
        raise RuntimeError("Temporal dataset did not produce non-empty train/validation snapshot sequences.")

    model = _make_model(
        train_snapshots[0].edge_index,
        node_features,
        num_nodes,
        destination_spec,
        device,
        args,
    )
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

    metric_name = getattr(dataset, "eval_metric", "mrr")
    history = []
    best_val_metric = float("-inf")
    best_test_metric = float("nan")
    best_epoch = -1
    best_state_dict = None

    start = time.perf_counter()
    for epoch in tqdm(range(args.temporal_epochs), desc="Temporal tgbl training"):
        model.reset_temporal_state()
        train_loss = _run_epoch(
            model,
            optimizer,
            train_snapshots,
            destination_spec,
            bptt_steps=args.temporal_bptt_steps,
            show_progress=args.temporal_epoch_progress_bar,
            epoch_label=f"Epoch {epoch + 1}/{args.temporal_epochs}",
        )

        should_eval = ((epoch + 1) % max(args.temporal_eval_every, 1) == 0) or (epoch == args.temporal_epochs - 1)
        train_metric = float("nan")
        val_metric = float("nan")
        test_metric = float("nan")
        train_eval_loss = float("nan")
        val_eval_loss = float("nan")
        test_eval_loss = float("nan")
        train_state_memory_norm = float("nan")
        val_state_memory_norm = float("nan")
        test_state_memory_norm = float("nan")

        if should_eval:
            if args.temporal_skip_train_eval:
                train_state = _advance_context(model, train_snapshots)
            else:
                train_metric, train_eval_loss, train_state = _evaluate_model_streaming(
                    dataset_name,
                    dataset,
                    train_snapshots,
                    model,
                    destination_spec,
                    initial_state=None,
                    split_mode="train",
                    compute_metric=False,
                    compute_loss=True,
                )

            val_metric, val_eval_loss, val_state = _evaluate_model_streaming(
                dataset_name,
                dataset,
                val_snapshots,
                model,
                destination_spec,
                initial_state=train_state,
                split_mode="val",
            )
            test_metric, test_eval_loss, test_state = _evaluate_model_streaming(
                dataset_name,
                dataset,
                test_snapshots,
                model,
                destination_spec,
                initial_state=val_state,
                split_mode="test",
            )

            train_state_memory_norm = (
                float(train_state.memory.norm().detach().cpu())
                if train_state is not None and train_state.memory is not None
                else float("nan")
            )
            val_state_memory_norm = (
                float(val_state.memory.norm().detach().cpu())
                if val_state is not None and val_state.memory is not None
                else float("nan")
            )
            test_state_memory_norm = (
                float(test_state.memory.norm().detach().cpu())
                if test_state is not None and test_state.memory is not None
                else float("nan")
            )

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
                "train_eval_loss": train_eval_loss,
                "val_eval_loss": val_eval_loss,
                "test_eval_loss": test_eval_loss,
                "train_state_memory_norm": train_state_memory_norm,
                "val_state_memory_norm": val_state_memory_norm,
                "test_state_memory_norm": test_state_memory_norm,
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
        final_train_state = _advance_context(model, train_snapshots)
    else:
        final_train_metric, final_train_loss, final_train_state = _evaluate_model_streaming(
            dataset_name,
            dataset,
            train_snapshots,
            model,
            destination_spec,
            initial_state=None,
            split_mode="train",
            compute_metric=False,
            compute_loss=True,
        )

    final_val_metric, final_val_loss, final_val_state = _evaluate_model_streaming(
        dataset_name,
        dataset,
        val_snapshots,
        model,
        destination_spec,
        initial_state=final_train_state,
        split_mode="val",
    )
    final_test_metric, final_test_loss, final_test_state = _evaluate_model_streaming(
        dataset_name,
        dataset,
        test_snapshots,
        model,
        destination_spec,
        initial_state=final_val_state,
        split_mode="test",
    )

    split_summary_df = pd.DataFrame(
        [
            _snapshot_summary("train", train_snapshots, train_edge_ids, temporal_data, destination_spec),
            _snapshot_summary("val", val_snapshots, val_edge_ids, temporal_data, destination_spec),
            _snapshot_summary("test", test_snapshots, test_edge_ids, temporal_data, destination_spec),
        ]
    )

    config_df = pd.DataFrame(
        [
            {"field": "requested_dataset_name", "value": requested_dataset_name},
            {"field": "dataset_name", "value": dataset_name},
            {"field": "task_family", "value": "tgbl_linkprop"},
            {"field": "eval_metric", "value": metric_name},
            {"field": "device", "value": str(device)},
            {"field": "seed", "value": args.seed},
            {"field": "repo_sha", "value": _repo_sha()},
            {"field": "split_source", "value": split_source},
            {"field": "num_nodes", "value": num_nodes},
            {"field": "destination_offset", "value": int(destination_spec["offset"])},
            {"field": "destination_size", "value": int(destination_spec["size"])},
            {"field": "temporal_snapshot_time_window", "value": args.temporal_snapshot_time_window},
            {"field": "temporal_bptt_steps", "value": args.temporal_bptt_steps},
            {"field": "temporal_eval_every", "value": args.temporal_eval_every},
            {"field": "temporal_skip_train_eval", "value": args.temporal_skip_train_eval},
            {"field": "temporal_epochs", "value": args.temporal_epochs},
            {"field": "train_edges_selected", "value": int(train_edge_ids.numel())},
            {"field": "val_edges_selected", "value": int(val_edge_ids.numel())},
            {"field": "test_edges_selected", "value": int(test_edge_ids.numel())},
        ]
    )

    model_config_df = pd.DataFrame(
        [
            {"field": "hidden_channels", "value": _env_int("TGB_HIDDEN_CHANNELS", 8)},
            {"field": "layers", "value": _env_int("TGB_LAYERS", 2)},
            {"field": "dropout", "value": float(os.environ.get("TGB_DROPOUT", 0.0))},
            {"field": "temporal_d_model", "value": args.temporal_d_model or _env_int("TGB_TEMPORAL_D_MODEL", 64)},
            {"field": "input_dim", "value": int(node_features.size(1))},
            {"field": "graph_size", "value": int(num_nodes)},
            {"field": "output_dim", "value": int(destination_spec["size"])},
            {"field": "destination_offset", "value": int(destination_spec["offset"])},
            {"field": "stateful_temporal", "value": bool(args.stateful_temporal)},
            {"field": "closure_hops", "value": int(args.closure_hops)},
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
                "loss": final_train_loss,
                "state_memory_norm": (
                    float(final_train_state.memory.norm().detach().cpu())
                    if final_train_state and final_train_state.memory is not None
                    else float("nan")
                ),
            },
            {
                "split": "val",
                metric_name: final_val_metric,
                "loss": final_val_loss,
                "state_memory_norm": (
                    float(final_val_state.memory.norm().detach().cpu())
                    if final_val_state and final_val_state.memory is not None
                    else float("nan")
                ),
            },
            {
                "split": "test",
                metric_name: final_test_metric,
                "loss": final_test_loss,
                "state_memory_norm": (
                    float(final_test_state.memory.norm().detach().cpu())
                    if final_test_state and final_test_state.memory is not None
                    else float("nan")
                ),
            },
        ]
    )

    results_dir = Path("results") / f"{dataset_name}_{args.model}_{int(time.time())}"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(results_dir / "config.csv", index=False)
    model_config_df.to_csv(results_dir / "model_config.csv", index=False)
    split_summary_df.to_csv(results_dir / "split_summary.csv", index=False)
    history_df.to_csv(results_dir / "history.csv", index=False)
    final_diagnostics_df.to_csv(results_dir / "final_diagnostics.csv", index=False)

    print(
        f"{args.model} on {dataset_name} | split={split_source} "
        f"| best_val_{metric_name}={best_val_metric:.4f} "
        f"| best_test_{metric_name}={best_test_metric:.4f} "
        f"| final_val_{metric_name}={final_val_metric:.4f} "
        f"| final_test_{metric_name}={final_test_metric:.4f} "
        f"| elapsed_sec={elapsed:.2f}"
    )
    print(f"Saved diagnostics to {results_dir}")


if __name__ == "__main__":
    main()
