import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from exp.temporal_utils import TemporalSnapshot, build_temporal_snapshots
from models.mamba_models import MambaSheafDiffusion, RolloutTemporalMambaSheafDiffusion


DATASET_ALIASES = {
    "tgbl-wiki-v2": "tgbl-wiki",
    "tgbl-review-v2": "tgbl-review",
}


@dataclass(frozen=True)
class DatasetSpec:
    requested_name: str
    loader_name: str
    task_family: str
    dataset_module: str
    evaluator_module: str
    metric_name: str
    train_metric_supported: bool
    notes: str = ""


def _env_int(name, default):
    return int(os.environ.get(name, default))


def pytest_importorskip(module_name):
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Required temporal dependency {module_name!r} is unavailable. "
            "Install the optional TGB dependencies before running this entry point."
        ) from exc
    return sys.modules[module_name]


def normalize_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    loader_name = normalize_dataset_name(dataset_name)
    if loader_name.startswith("tgbn-"):
        return DatasetSpec(
            requested_name=dataset_name,
            loader_name=loader_name,
            task_family="nodeprop",
            dataset_module="tgb.nodeproppred.dataset_pyg",
            evaluator_module="tgb.nodeproppred.evaluate",
            metric_name="ndcg",
            train_metric_supported=True,
        )
    if loader_name.startswith("tgbl-"):
        notes = ""
        if dataset_name != loader_name:
            notes = (
                f"TGB exposes {dataset_name!r} via the loader name {loader_name!r}; "
                "the Python package maps that to the v2 dataset files internally."
            )
        return DatasetSpec(
            requested_name=dataset_name,
            loader_name=loader_name,
            task_family="linkprop",
            dataset_module="tgb.linkproppred.dataset_pyg",
            evaluator_module="tgb.linkproppred.evaluate",
            metric_name="mrr",
            train_metric_supported=False,
            notes=notes,
        )
    if loader_name.startswith("tkgl-"):
        return DatasetSpec(
            requested_name=dataset_name,
            loader_name=loader_name,
            task_family="tkg",
            dataset_module="tgb.linkproppred.dataset_pyg",
            evaluator_module="tgb.linkproppred.evaluate",
            metric_name="mrr",
            train_metric_supported=False,
            notes=(
                "This notebook evaluates TKG datasets as temporal destination-ranking tasks "
                "with TGB negative samples. Relation types are surfaced in diagnostics and "
                "negative-sampler queries, but the current backbone does not score relations explicitly."
            ),
        )
    if loader_name.startswith("thgl-"):
        return DatasetSpec(
            requested_name=dataset_name,
            loader_name=loader_name,
            task_family="thg",
            dataset_module="tgb.linkproppred.dataset_pyg",
            evaluator_module="tgb.linkproppred.evaluate",
            metric_name="mrr",
            train_metric_supported=False,
            notes=(
                "This notebook evaluates THG datasets as temporal destination-ranking tasks "
                "with TGB negative samples. Node and edge types are exposed in diagnostics, "
                "but the current backbone is still homogeneous."
            ),
        )
    raise ValueError(f"Unsupported temporal TGB dataset {dataset_name!r}.")


def load_tgb_dataset(dataset_name: str):
    spec = resolve_dataset_spec(dataset_name)
    root = os.environ.get("TGB_ROOT", "datasets")
    answer = "y" if os.environ.get("TGB_DOWNLOAD") == "1" else "n"

    tgb_dataset = pytest_importorskip(spec.dataset_module)
    pytest_importorskip(spec.evaluator_module)
    dataset_cls = (
        tgb_dataset.PyGNodePropPredDataset
        if spec.task_family == "nodeprop"
        else tgb_dataset.PyGLinkPropPredDataset
    )
    with patch("builtins.input", return_value=answer):
        dataset = dataset_cls(name=spec.loader_name, root=root)
    return spec, dataset


def load_temporal_data(dataset_name: str):
    spec, dataset = load_tgb_dataset(dataset_name)
    temporal_data = dataset.get_TemporalData()
    return spec, dataset, temporal_data


def make_node_features(dataset, num_nodes):
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


def make_model(edge_index, x, num_nodes, output_dim, device, args):
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
        "output_dim": output_dim,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
        "stateful_temporal": args.stateful_temporal,
        "closure_hops": args.closure_hops,
        "temporal_d_model": args.temporal_d_model or _env_int("TGB_TEMPORAL_D_MODEL", 64),
    }
    model_name = getattr(args, "model", None)
    model_cls = RolloutTemporalMambaSheafDiffusion if model_name == "RolloutTemporalMambaSheaf" else MambaSheafDiffusion
    return model_cls(_normalize_sheaf_edge_index(edge_index).to(device), model_args).to(device)


def split_edge_ids(dataset, temporal_data):
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


def make_static_edge_index(dataset):
    if not hasattr(dataset, "static_data"):
        return None
    try:
        static_data = dataset.static_data
    except Exception:
        return None
    if not static_data:
        return None
    head = static_data.get("head")
    tail = static_data.get("tail")
    if head is None or tail is None or head.numel() == 0:
        return None
    edge_index = torch.stack([head.long(), tail.long()], dim=0)
    reverse_edge_index = torch.stack([tail.long(), head.long()], dim=0)
    return _normalize_sheaf_edge_index(torch.cat([edge_index, reverse_edge_index], dim=1))


def _augment_snapshot_with_static_edges(snapshot: TemporalSnapshot, static_edge_index: Optional[torch.Tensor]):
    if static_edge_index is None:
        return snapshot
    combined_edge_index = torch.cat([snapshot.edge_index, static_edge_index.to(snapshot.edge_index.device)], dim=1)
    return TemporalSnapshot(
        x=snapshot.x,
        edge_index=_normalize_sheaf_edge_index(combined_edge_index),
        src=snapshot.src,
        dst=snapshot.dst,
        active_nodes=snapshot.active_nodes,
        timestamp=snapshot.timestamp,
        edge_timestamps=snapshot.edge_timestamps,
        edge_types=snapshot.edge_types,
    )


def build_snapshots(
    temporal_data,
    edge_ids,
    node_features,
    max_edges=None,
    time_window=None,
    static_edge_index=None,
):
    edge_types = getattr(temporal_data, "edge_type", None)
    snapshots = build_temporal_snapshots(
        temporal_data.src,
        temporal_data.dst,
        temporal_data.t,
        node_features,
        edge_ids=edge_ids,
        max_edges=max_edges,
        time_window=time_window,
        edge_types=edge_types,
    )
    return [_augment_snapshot_with_static_edges(snapshot, static_edge_index) for snapshot in snapshots]


def normalize_window_candidates(window_candidates: Iterable[Optional[int]]):
    normalized = []
    seen = set()
    for window in window_candidates:
        value = None if window is None else int(window)
        if value is not None and value <= 0:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    numeric = sorted(window for window in normalized if window is not None)
    return ([None] if None in seen else []) + numeric


def make_window_candidates(timestamps, edge_ids, manual_candidates=None, limit=12):
    if manual_candidates is not None:
        return normalize_window_candidates(manual_candidates)
    if edge_ids.numel() == 0:
        return [None]

    selected_ts = timestamps[edge_ids]
    span = int((selected_ts.max() - selected_ts.min()).item()) if selected_ts.numel() > 0 else 0
    if span <= 1:
        return [None, 1]

    raw = np.geomspace(1, span, num=limit)
    candidates = [None] + [int(x) for x in np.unique(np.round(raw).astype(int)).tolist() if int(x) > 0]
    return normalize_window_candidates(candidates)


def estimate_snapshot_bucket_stats(timestamps, edge_ids, time_window=None, max_edges=None):
    selected_edge_ids = edge_ids if max_edges is None else edge_ids[:max_edges]
    if selected_edge_ids.numel() == 0:
        return {
            "edge_prefix": 0,
            "raw_unique_timestamps": 0,
            "estimated_snapshots": 0,
            "estimated_mean_edges": float("nan"),
            "estimated_max_edges": 0,
            "estimated_min_edges": 0,
            "compression_ratio": float("nan"),
            "timestamp_start": None,
            "timestamp_end": None,
            "representative_timestamps": torch.empty(0, dtype=timestamps.dtype),
        }

    selected_ts = timestamps[selected_edge_ids]
    ordered_ts = selected_ts[torch.argsort(selected_ts)]
    raw_unique_timestamps = int(torch.unique_consecutive(ordered_ts).numel())

    if time_window is None:
        bucket_keys = ordered_ts
    else:
        if time_window <= 0:
            raise ValueError("time_window must be positive when provided.")
        bucket_keys = torch.div(ordered_ts - ordered_ts[0], time_window, rounding_mode="floor")

    _, bucket_counts = torch.unique_consecutive(bucket_keys, return_counts=True)
    bucket_end_indices = torch.cumsum(bucket_counts, dim=0) - 1
    representative_timestamps = ordered_ts[bucket_end_indices]

    return {
        "edge_prefix": int(selected_edge_ids.numel()),
        "raw_unique_timestamps": raw_unique_timestamps,
        "estimated_snapshots": int(bucket_counts.numel()),
        "estimated_mean_edges": float(bucket_counts.float().mean().item()),
        "estimated_max_edges": int(bucket_counts.max().item()),
        "estimated_min_edges": int(bucket_counts.min().item()),
        "compression_ratio": float(bucket_counts.numel() / raw_unique_timestamps) if raw_unique_timestamps else float("nan"),
        "timestamp_start": int(ordered_ts[0].item()),
        "timestamp_end": int(ordered_ts[-1].item()),
        "representative_timestamps": representative_timestamps,
    }


def _has_label_at_snapshot_timestamps(dataset, snapshot_timestamps):
    dataset.reset_label_time()
    first_label_ts = None
    for ts in snapshot_timestamps.tolist():
        label_tuple = dataset.get_node_label(int(ts))
        if label_tuple is not None:
            first_label_ts = int(ts)
            break
    return first_label_ts is not None, first_label_ts


def estimate_supervised_prefix(spec: DatasetSpec, dataset, temporal_data, edge_ids, max_edges=None, time_window=None):
    selected_count = int(edge_ids.numel()) if max_edges is None else min(int(max_edges), int(edge_ids.numel()))
    if selected_count == 0:
        stats = estimate_snapshot_bucket_stats(temporal_data.t, edge_ids, time_window=time_window, max_edges=0)
        stats.update({"has_label_supervision": False, "first_label_timestamp": None})
        return stats

    while True:
        stats = estimate_snapshot_bucket_stats(
            temporal_data.t,
            edge_ids,
            time_window=time_window,
            max_edges=selected_count,
        )
        if spec.task_family == "nodeprop":
            has_label_supervision, first_label_timestamp = _has_label_at_snapshot_timestamps(
                dataset,
                stats["representative_timestamps"],
            )
        else:
            has_label_supervision = stats["estimated_snapshots"] > 0
            first_label_timestamp = stats["timestamp_start"]
        stats.update({
            "has_label_supervision": has_label_supervision,
            "first_label_timestamp": first_label_timestamp,
        })
        if has_label_supervision or selected_count >= int(edge_ids.numel()):
            return stats
        selected_count = min(int(edge_ids.numel()), max(selected_count * 2, selected_count + 1))


def estimate_time_window_grid(spec: DatasetSpec, dataset, temporal_data, edge_ids, max_edges, candidate_windows):
    rows = []
    for time_window in candidate_windows:
        stats = estimate_supervised_prefix(
            spec,
            dataset,
            temporal_data,
            edge_ids,
            max_edges=max_edges,
            time_window=time_window,
        )
        rows.append({
            "time_window": time_window,
            "window_label": "exact timestamps" if time_window is None else str(time_window),
            "edge_prefix": stats["edge_prefix"],
            "raw_unique_timestamps": stats["raw_unique_timestamps"],
            "estimated_snapshots": stats["estimated_snapshots"],
            "estimated_mean_edges": stats["estimated_mean_edges"],
            "estimated_max_edges": stats["estimated_max_edges"],
            "estimated_min_edges": stats["estimated_min_edges"],
            "compression_ratio": stats["compression_ratio"],
            "timestamp_start": stats["timestamp_start"],
            "timestamp_end": stats["timestamp_end"],
            "has_label_supervision": stats["has_label_supervision"],
            "first_label_timestamp": stats["first_label_timestamp"],
        })
    return pd.DataFrame(rows)


def recommend_time_window(estimate_df, target_max_snapshots=None, target_max_mean_edges=None):
    if estimate_df.empty:
        return None

    feasible = estimate_df.copy()
    feasible = feasible[feasible["has_label_supervision"]]
    if target_max_snapshots is not None:
        feasible = feasible[feasible["estimated_snapshots"] <= target_max_snapshots]
    if target_max_mean_edges is not None:
        feasible = feasible[feasible["estimated_mean_edges"] <= target_max_mean_edges]
    if feasible.empty:
        return None

    order = feasible["time_window"].map(lambda x: -1 if x is None else int(x))
    return feasible.iloc[order.argsort(kind="stable")].iloc[0].to_dict()


def node_label_batches(dataset, snapshots):
    dataset.reset_label_time()
    batches = []
    for snapshot in snapshots:
        cur_t = int(snapshot.timestamp.item()) if torch.is_tensor(snapshot.timestamp) else int(snapshot.timestamp)
        label_tuple = dataset.get_node_label(cur_t)
        if label_tuple is None:
            continue
        _, label_srcs, labels = label_tuple
        batches.append((snapshot, label_srcs.long(), labels.float()))
    return batches


def summarize_supervision(spec: DatasetSpec, dataset, snapshots):
    if spec.task_family == "nodeprop":
        label_batches = node_label_batches(dataset, snapshots)
        label_snapshot_ids = {id(snapshot) for snapshot, _, _ in label_batches}
        label_counts = [int(labels.size(0)) for _, _, labels in label_batches]
        return {
            "snapshots_with_labels": len(label_batches),
            "snapshot_label_coverage": len(label_batches) / len(snapshots) if snapshots else float("nan"),
            "label_rows_total": int(sum(label_counts)),
            "label_rows_mean": float(np.mean(label_counts)) if label_counts else float("nan"),
            "snapshot_ids_with_labels": len(label_snapshot_ids),
        }

    edge_counts = [int(snapshot.src.numel()) for snapshot in snapshots]
    return {
        "snapshots_with_labels": len(snapshots),
        "snapshot_label_coverage": 1.0 if snapshots else float("nan"),
        "label_rows_total": int(sum(edge_counts)),
        "label_rows_mean": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "snapshot_ids_with_labels": len(snapshots),
    }


def build_supervised_snapshots(
    spec: DatasetSpec,
    dataset,
    temporal_data,
    edge_ids,
    node_features,
    max_edges=None,
    time_window=None,
    split_name="split",
    static_edge_index=None,
):
    if max_edges is None:
        selected_edge_ids = edge_ids
    else:
        selected_edge_ids = edge_ids[:max_edges]

    if selected_edge_ids.numel() == 0:
        return [], selected_edge_ids

    if spec.task_family != "nodeprop":
        snapshots = build_snapshots(
            temporal_data,
            selected_edge_ids,
            node_features,
            max_edges=None,
            time_window=time_window,
            static_edge_index=static_edge_index,
        )
        return snapshots, selected_edge_ids

    while True:
        snapshots = build_snapshots(
            temporal_data,
            selected_edge_ids,
            node_features,
            max_edges=None,
            time_window=time_window,
            static_edge_index=static_edge_index,
        )
        if node_label_batches(dataset, snapshots):
            return snapshots, selected_edge_ids

        if selected_edge_ids.numel() >= edge_ids.numel():
            raise RuntimeError(
                f"{split_name} split produced no label-supervised snapshots even after "
                f"expanding to all {edge_ids.numel()} available edges."
            )

        next_count = min(edge_ids.numel(), max(selected_edge_ids.numel() * 2, selected_edge_ids.numel() + 1))
        selected_edge_ids = edge_ids[:next_count]


def node_property_loss(outputs: Sequence[torch.Tensor], dataset, snapshots: Sequence[TemporalSnapshot]):
    from torch.nn import functional as F

    losses = []
    batches = node_label_batches(dataset, snapshots)
    batch_index = 0
    for snapshot_index, snapshot in enumerate(snapshots):
        if batch_index >= len(batches):
            break
        batch_snapshot, label_srcs, labels = batches[batch_index]
        if batch_snapshot is not snapshot:
            continue
        logits = outputs[snapshot_index]
        pred = logits.index_select(0, label_srcs.to(logits.device))
        target = labels.to(logits.device)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        losses.append(-(target * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean())
        batch_index += 1

    if not losses:
        return torch.tensor(0.0, device=outputs[0].device if outputs else "cpu")
    return torch.stack(losses).mean()


def link_prediction_loss(outputs: Sequence[torch.Tensor], snapshots: Sequence[TemporalSnapshot]):
    from torch.nn import functional as F

    losses = []
    for logits, snapshot in zip(outputs, snapshots):
        if snapshot.src.numel() == 0:
            continue
        losses.append(F.nll_loss(logits[snapshot.src.to(logits.device)], snapshot.dst.to(logits.device)))
    if not losses:
        return torch.tensor(0.0, device=outputs[0].device if outputs else "cpu")
    return torch.stack(losses).mean()


def sequence_loss_for_task(spec: DatasetSpec, outputs, dataset, snapshots):
    if spec.task_family == "nodeprop":
        return node_property_loss(outputs, dataset, snapshots)
    return link_prediction_loss(outputs, snapshots)


def _next_snapshot_labels(dataset, snapshot):
    cur_t = int(snapshot.timestamp.item()) if torch.is_tensor(snapshot.timestamp) else int(snapshot.timestamp)
    label_tuple = dataset.get_node_label(cur_t)
    if label_tuple is None:
        return None
    _, label_srcs, labels = label_tuple
    return label_srcs.long(), labels.float()


def detach_temporal_state(state):
    if state is None:
        return None
    return type(state)(
        memory=state.memory.detach() if state.memory is not None else None,
        spatial=state.spatial.detach() if state.spatial is not None else None,
    )


def forward_snapshot_sequence(model, snapshots, initial_state=None):
    if getattr(model, "prefers_rollout_sequence", False):
        return model.forward_sequence(snapshots, initial_state=initial_state)

    outputs = []
    state = initial_state
    for snapshot in snapshots:
        chunk_outputs, state = model.forward_sequence([snapshot], initial_state=state)
        outputs.extend(chunk_outputs)
    return outputs, state


def advance_context(model, snapshots, initial_state=None):
    model.eval()
    with torch.no_grad():
        _, state = forward_snapshot_sequence(model, snapshots, initial_state=initial_state)
    return state


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


def evaluate_model_streaming(
    spec: DatasetSpec,
    dataset,
    snapshots,
    model,
    initial_state=None,
    split_mode=None,
    compute_metric=True,
    compute_loss=True,
):
    from torch.nn import functional as F

    evaluator = None
    if compute_metric:
        evaluator_module = pytest_importorskip(spec.evaluator_module)
        evaluator = evaluator_module.Evaluator(name=spec.loader_name)

    all_y_pred = []
    all_y_true = []
    losses = []
    metric_sum = 0.0
    metric_examples = 0
    state = initial_state

    if spec.task_family == "nodeprop":
        dataset.reset_label_time()

    if spec.task_family != "nodeprop":
        if split_mode == "val" and hasattr(dataset, "load_val_ns"):
            dataset.load_val_ns()
        elif split_mode == "test" and hasattr(dataset, "load_test_ns"):
            dataset.load_test_ns()

    model.eval()
    with torch.no_grad():
        logits_sequence, state = forward_snapshot_sequence(model, snapshots, initial_state=state)
        for snapshot, logits in zip(snapshots, logits_sequence):

            if spec.task_family == "nodeprop":
                label_batch = _next_snapshot_labels(dataset, snapshot)
                if label_batch is None:
                    continue

                label_srcs, labels = label_batch
                pred = logits.index_select(0, label_srcs.to(logits.device))
                target = labels.to(logits.device)
                normalised_target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                if compute_loss:
                    losses.append((-(normalised_target * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean()).detach().cpu())
                if compute_metric:
                    all_y_pred.append(pred.detach().cpu())
                    all_y_true.append(labels.detach().cpu())
                continue

            pos_src = snapshot.src.to(logits.device)
            pos_dst = snapshot.dst.to(logits.device)
            if pos_src.numel() == 0:
                continue

            if compute_loss:
                losses.append(F.nll_loss(logits[pos_src], pos_dst).detach().cpu())

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

            neg_dst = torch.as_tensor(neg_samples, dtype=torch.long, device=logits.device)
            y_pred_pos = logits[pos_src, pos_dst]
            y_pred_neg = logits[pos_src.view(-1, 1).expand_as(neg_dst), neg_dst]
            score = evaluator.eval({
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
                "eval_metric": [dataset.eval_metric],
            })
            metric_val = list(score.values())[0] if isinstance(score, dict) else score
            metric_sum += float(metric_val) * int(pos_src.numel())
            metric_examples += int(pos_src.numel())

    metric = float("nan")
    if spec.task_family == "nodeprop" and compute_metric and all_y_pred:
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)
        try:
            score = evaluator.eval({
                "y_pred": all_y_pred,
                "y_true": all_y_true,
                "eval_metric": [dataset.eval_metric],
            })
            metric_val = list(score.values())[0] if isinstance(score, dict) else score
            metric = float(metric_val)
        except Exception:
            metric = float("nan")
    elif spec.task_family != "nodeprop" and compute_metric and metric_examples > 0:
        metric = metric_sum / metric_examples

    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    return metric, mean_loss, state


def run_epoch(
    spec: DatasetSpec,
    model,
    optimizer,
    train_snapshots,
    dataset,
    bptt_steps=None,
    show_progress=False,
    epoch_label=None,
    tqdm_factory=None,
):
    model.train()
    if bptt_steps is None or bptt_steps <= 0:
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = sequence_loss_for_task(spec, outputs, dataset, train_snapshots)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    chunk_losses = []
    state = None
    chunk_starts = range(0, len(train_snapshots), bptt_steps)
    if show_progress and tqdm_factory is not None:
        chunk_starts = tqdm_factory(
            chunk_starts,
            total=(len(train_snapshots) + bptt_steps - 1) // bptt_steps,
            desc=epoch_label or "Epoch chunks",
            leave=True,
        )

    for start in chunk_starts:
        chunk = train_snapshots[start:start + bptt_steps]
        optimizer.zero_grad()
        outputs, state = model.forward_sequence(chunk, initial_state=state)
        loss = sequence_loss_for_task(spec, outputs, dataset, chunk)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            chunk_losses.append(float(loss.detach().cpu()))

        state = detach_temporal_state(state)

    if not chunk_losses:
        return 0.0
    return float(sum(chunk_losses) / len(chunk_losses))
