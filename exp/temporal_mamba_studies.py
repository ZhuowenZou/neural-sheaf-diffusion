# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected
from tqdm.auto import tqdm

from exp.temporal_utils import build_temporal_snapshots
from models.mamba_models import MambaSheafDiffusion


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pytest_importorskip(module_name: str):
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Required temporal dependency {module_name!r} is unavailable. "
            "Install the optional TGB dependencies before running these studies."
        ) from exc
    return sys.modules[module_name]


def _load_tgb_dataset(dataset_name: str):
    root = os.environ.get("TGB_ROOT", "datasets")
    answer = "y" if os.environ.get("TGB_DOWNLOAD") == "1" else "n"
    tgb_dataset = pytest_importorskip("tgb.nodeproppred.dataset_pyg")
    pytest_importorskip("tgb.nodeproppred.evaluate")
    with patch("builtins.input", return_value=answer):
        return tgb_dataset.PyGNodePropPredDataset(name=dataset_name, root=root)


def _load_temporal_data(dataset_name: str):
    dataset = _load_tgb_dataset(dataset_name)
    temporal_data = dataset.get_TemporalData()
    return dataset, temporal_data


def _make_node_features(dataset, num_nodes: int, seed: int = 43) -> torch.Tensor:
    try:
        if hasattr(dataset, "node_feat") and dataset.node_feat is not None and dataset.node_feat.size(0) == num_nodes:
            return dataset.node_feat.float()
    except (AttributeError, RuntimeError):
        pass

    feature_dim = int(os.environ.get("TGB_NODE_FEATURE_DIM", 64))
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, feature_dim, generator=generator)


def _safe_output_dim(dataset, num_nodes: int) -> int:
    try:
        output_dim = getattr(dataset, "num_classes")
    except Exception:
        output_dim = None
    if output_dim is None:
        return int(num_nodes)
    return int(output_dim)


def _normalize_sheaf_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None)
    return edge_index.contiguous()


def _canonical_time_window(window):
    if window is None or pd.isna(window):
        return None
    return int(window)


def _format_value(parameter: str, value: Any) -> str:
    if parameter == "time_window":
        value = _canonical_time_window(value)
        return "exact timestamps" if value is None else str(value)
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def _make_model_from_config(edge_index, x, num_nodes, output_dim, device, config):
    model_args = {
        "d": int(config["d"]),
        "add_lp": bool(config.get("add_lp", False)),
        "add_hp": bool(config.get("add_hp", False)),
        "device": device,
        "graph_size": num_nodes,
        "layers": int(config["layers"]),
        "normalised": bool(config.get("normalised", True)),
        "deg_normalised": bool(config.get("deg_normalised", False)),
        "linear": bool(config.get("linear", False)),
        "input_dropout": float(config.get("input_dropout", 0.0)),
        "dropout": float(config["dropout"]),
        "left_weights": bool(config.get("left_weights", True)),
        "right_weights": bool(config.get("right_weights", True)),
        "sparse_learner": bool(config.get("sparse_learner", False)),
        "use_act": bool(config.get("use_act", True)),
        "input_dim": int(x.size(1)),
        "hidden_channels": int(config["hidden_channels"]),
        "output_dim": int(output_dim),
        "sheaf_act": str(config.get("sheaf_act", "tanh")),
        "second_linear": bool(config.get("second_linear", False)),
        "orth": str(config.get("orth", "householder")),
        "edge_weights": bool(config.get("edge_weights", False)),
        "max_t": float(config.get("max_t", 1.0)),
        "stateful_temporal": bool(config["stateful_temporal"]),
        "closure_hops": int(config["closure_hops"]),
        "temporal_d_model": int(config["temporal_d_model"]),
    }
    return MambaSheafDiffusion(_normalize_sheaf_edge_index(edge_index).to(device), model_args).to(device)


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
    return build_temporal_snapshots(
        temporal_data.src,
        temporal_data.dst,
        temporal_data.t,
        node_features,
        edge_ids=edge_ids,
        max_edges=max_edges,
        time_window=_canonical_time_window(time_window),
    )


def _node_label_batches(dataset, snapshots):
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


def _build_supervised_snapshots(
    dataset,
    temporal_data,
    edge_ids,
    node_features,
    max_edges=None,
    time_window=None,
    split_name="split",
):
    if max_edges is None:
        selected_edge_ids = edge_ids
    else:
        selected_edge_ids = edge_ids[:max_edges]

    if selected_edge_ids.numel() == 0:
        return [], selected_edge_ids

    while True:
        snapshots = _build_snapshots(
            temporal_data,
            selected_edge_ids,
            node_features,
            max_edges=None,
            time_window=time_window,
        )
        if _node_label_batches(dataset, snapshots):
            return snapshots, selected_edge_ids

        if selected_edge_ids.numel() >= edge_ids.numel():
            raise RuntimeError(
                f"{split_name} split produced no label-supervised snapshots even after "
                f"expanding to all {edge_ids.numel()} available edges."
            )

        next_count = min(edge_ids.numel(), max(selected_edge_ids.numel() * 2, selected_edge_ids.numel() + 1))
        selected_edge_ids = edge_ids[:next_count]


def _summarize_snapshots(dataset, snapshots):
    label_batches = _node_label_batches(dataset, snapshots)
    label_counts = [int(labels.size(0)) for _, _, labels in label_batches]
    edge_counts = [int(snapshot.src.numel()) for snapshot in snapshots]
    active_counts = [int(snapshot.active_nodes.numel()) for snapshot in snapshots]
    timestamps = [int(snapshot.timestamp.item()) for snapshot in snapshots]
    return {
        "snapshots": len(snapshots),
        "snapshots_with_labels": len(label_batches),
        "mean_edges": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "max_edges": int(max(edge_counts)) if edge_counts else 0,
        "mean_active_nodes": float(np.mean(active_counts)) if active_counts else float("nan"),
        "mean_labels": float(np.mean(label_counts)) if label_counts else float("nan"),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def _node_property_loss(outputs, dataset, snapshots):
    from torch.nn import functional as F

    losses = []
    batches = _node_label_batches(dataset, snapshots)
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


def _next_snapshot_labels(dataset, snapshot):
    cur_t = int(snapshot.timestamp.item()) if torch.is_tensor(snapshot.timestamp) else int(snapshot.timestamp)
    label_tuple = dataset.get_node_label(cur_t)
    if label_tuple is None:
        return None
    _, label_srcs, labels = label_tuple
    return label_srcs.long(), labels.float()


def _detach_temporal_state(state):
    if state is None:
        return None
    return type(state)(
        memory=state.memory.detach() if state.memory is not None else None,
        spatial=state.spatial.detach() if state.spatial is not None else None,
    )


def _run_epoch(model, optimizer, train_snapshots, dataset, bptt_steps=None, show_progress=False, epoch_label=None):
    model.train()
    if bptt_steps is None or bptt_steps <= 0:
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = _node_property_loss(outputs, dataset, train_snapshots)
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
        loss = _node_property_loss(outputs, dataset, chunk)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            chunk_losses.append(float(loss.detach().cpu()))

        state = _detach_temporal_state(state)

    if not chunk_losses:
        return 0.0
    return float(sum(chunk_losses) / len(chunk_losses))


def _advance_context(model, snapshots, initial_state=None):
    state = initial_state
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            _, state = model.forward_sequence([snapshot], initial_state=state)
    return state


def _evaluate_model_streaming(dataset_name, dataset, snapshots, model, initial_state=None, compute_metric=True, compute_loss=True):
    from torch.nn import functional as F

    evaluator = None
    if compute_metric:
        from tgb.nodeproppred.evaluate import Evaluator

        evaluator = Evaluator(name=dataset_name)

    all_y_pred = []
    all_y_true = []
    losses = []
    state = initial_state

    dataset.reset_label_time()
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            outputs, state = model.forward_sequence([snapshot], initial_state=state)
            logits = outputs[0]
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

    metric = float("nan")
    if compute_metric and all_y_pred:
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)
        try:
            score = evaluator.eval(
                {
                    "y_pred": all_y_pred,
                    "y_true": all_y_true,
                    "eval_metric": [dataset.eval_metric],
                }
            )
            metric_val = list(score.values())[0] if isinstance(score, dict) else score
            metric = float(metric_val)
        except Exception:
            metric = float("nan")

    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    return metric, mean_loss, state


def train_single_config(
    *,
    dataset_name,
    dataset,
    snapshot_bundle,
    node_features,
    num_nodes,
    output_dim,
    device,
    config,
    seed,
    keep_history=False,
):
    if not snapshot_bundle["train_snapshots"] or not snapshot_bundle["val_snapshots"]:
        raise RuntimeError("Snapshot bundle must include non-empty train and validation sequences.")

    _seed_everything(seed)

    model = _make_model_from_config(
        snapshot_bundle["train_snapshots"][0].edge_index,
        node_features,
        num_nodes,
        output_dim,
        device,
        config,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metric_name = getattr(dataset, "eval_metric", "ndcg")
    history = []
    best_val_metric = float("-inf")
    best_epoch = -1
    best_state_dict = None
    best_test_metric = float("nan")
    best_val_loss = float("nan")
    epochs_since_improvement = 0
    eval_every = max(int(config.get("eval_every", 1)), 1)
    min_delta = float(config.get("min_delta", 0.0))
    patience = int(config.get("early_stopping_patience", 10))
    min_epochs_before_stopping = int(config.get("min_epochs_before_stopping", eval_every))
    bptt_steps = config.get("temporal_bptt_steps")
    if bptt_steps is not None:
        bptt_steps = min(int(bptt_steps), len(snapshot_bundle["train_snapshots"]))
        if bptt_steps <= 0:
            bptt_steps = None

    for epoch in range(int(config["epochs"])):
        model.reset_temporal_state()
        train_loss = _run_epoch(
            model,
            optimizer,
            snapshot_bundle["train_snapshots"],
            dataset,
            bptt_steps=bptt_steps,
            show_progress=False,
            epoch_label=f"Epoch {epoch + 1}/{config['epochs']}",
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
        }

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == int(config["epochs"]) - 1)
        if should_eval:
            if config.get("skip_train_eval", True):
                train_metric = float("nan")
                train_eval_loss = float("nan")
                train_state = _advance_context(model, snapshot_bundle["train_snapshots"])
            else:
                train_metric, train_eval_loss, train_state = _evaluate_model_streaming(
                    dataset_name,
                    dataset,
                    snapshot_bundle["train_snapshots"],
                    model,
                    initial_state=None,
                )
                row[f"train_{metric_name}"] = train_metric
                row["train_eval_loss"] = train_eval_loss

            val_metric, val_loss, val_state = _evaluate_model_streaming(
                dataset_name,
                dataset,
                snapshot_bundle["val_snapshots"],
                model,
                initial_state=train_state,
            )
            test_metric = float("nan")
            test_loss = float("nan")
            if snapshot_bundle["test_snapshots"]:
                test_metric, test_loss, _ = _evaluate_model_streaming(
                    dataset_name,
                    dataset,
                    snapshot_bundle["test_snapshots"],
                    model,
                    initial_state=val_state,
                )

            row.update(
                {
                    f"val_{metric_name}": val_metric,
                    "val_loss": val_loss,
                    f"test_{metric_name}": test_metric,
                    "test_loss": test_loss,
                }
            )

            improved = (not math.isnan(val_metric)) and (val_metric > best_val_metric + min_delta)
            if improved:
                best_val_metric = float(val_metric)
                best_test_metric = float(test_metric)
                best_val_loss = float(val_loss)
                best_epoch = epoch + 1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += eval_every

            if (epoch + 1) >= min_epochs_before_stopping and epochs_since_improvement >= patience:
                row["early_stopped"] = True
                history.append(row)
                break

        history.append(row)

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    model.reset_temporal_state()
    if config.get("skip_train_eval", True):
        final_train_metric = float("nan")
        final_train_loss = float("nan")
        final_train_state = _advance_context(model, snapshot_bundle["train_snapshots"])
    else:
        final_train_metric, final_train_loss, final_train_state = _evaluate_model_streaming(
            dataset_name,
            dataset,
            snapshot_bundle["train_snapshots"],
            model,
            initial_state=None,
        )

    final_val_metric, final_val_loss, final_val_state = _evaluate_model_streaming(
        dataset_name,
        dataset,
        snapshot_bundle["val_snapshots"],
        model,
        initial_state=final_train_state,
    )
    final_test_metric = float("nan")
    final_test_loss = float("nan")
    if snapshot_bundle["test_snapshots"]:
        final_test_metric, final_test_loss, _ = _evaluate_model_streaming(
            dataset_name,
            dataset,
            snapshot_bundle["test_snapshots"],
            model,
            initial_state=final_val_state,
        )

    result = {
        "metric_name": metric_name,
        "best_epoch": best_epoch,
        "best_val_metric": final_val_metric,
        "best_val_loss": final_val_loss,
        "best_test_metric": final_test_metric,
        "best_test_loss": final_test_loss,
        "best_train_metric": final_train_metric,
        "best_train_loss": final_train_loss,
        "search_best_test_metric": best_test_metric,
        "search_best_val_loss": best_val_loss,
        "history": pd.DataFrame(history) if keep_history else None,
    }

    del model
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@dataclass
class TemporalExperimentContext:
    dataset_name: str
    device: torch.device
    dataset: Any
    temporal_data: Any
    node_features: torch.Tensor
    num_nodes: int
    output_dim: int
    split_edge_ids: Dict[str, torch.Tensor]
    split_caps: Dict[str, int]
    split_source: str
    metric_name: str
    snapshot_cache: Dict[Optional[int], Dict[str, Any]] = field(default_factory=dict)

    def get_snapshot_bundle(self, time_window=None):
        key = _canonical_time_window(time_window)
        if key not in self.snapshot_cache:
            train_snapshots, train_ids = _build_supervised_snapshots(
                self.dataset,
                self.temporal_data,
                self.split_edge_ids["train"],
                self.node_features,
                max_edges=self.split_caps["train"],
                time_window=key,
                split_name="train",
            )
            val_snapshots, val_ids = _build_supervised_snapshots(
                self.dataset,
                self.temporal_data,
                self.split_edge_ids["val"],
                self.node_features,
                max_edges=self.split_caps["val"],
                time_window=key,
                split_name="validation",
            )
            test_snapshots, test_ids = _build_supervised_snapshots(
                self.dataset,
                self.temporal_data,
                self.split_edge_ids["test"],
                self.node_features,
                max_edges=self.split_caps["test"],
                time_window=key,
                split_name="test",
            )
            self.snapshot_cache[key] = {
                "time_window": key,
                "train_snapshots": train_snapshots,
                "val_snapshots": val_snapshots,
                "test_snapshots": test_snapshots,
                "train_edge_count": int(train_ids.numel()),
                "val_edge_count": int(val_ids.numel()),
                "test_edge_count": int(test_ids.numel()),
            }
        return self.snapshot_cache[key]

    def snapshot_table(self) -> pd.DataFrame:
        rows = []
        for time_window, bundle in sorted(self.snapshot_cache.items(), key=lambda item: (-1 if item[0] is None else item[0])):
            train_summary = _summarize_snapshots(self.dataset, bundle["train_snapshots"])
            val_summary = _summarize_snapshots(self.dataset, bundle["val_snapshots"])
            test_summary = _summarize_snapshots(self.dataset, bundle["test_snapshots"])
            rows.append(
                {
                    "time_window": time_window,
                    "window_label": _format_value("time_window", time_window),
                    "train_edges": bundle["train_edge_count"],
                    "train_snapshots": train_summary["snapshots"],
                    "train_mean_edges": train_summary["mean_edges"],
                    "train_labelled_snapshots": train_summary["snapshots_with_labels"],
                    "val_edges": bundle["val_edge_count"],
                    "val_snapshots": val_summary["snapshots"],
                    "val_labelled_snapshots": val_summary["snapshots_with_labels"],
                    "test_edges": bundle["test_edge_count"],
                    "test_snapshots": test_summary["snapshots"],
                    "test_labelled_snapshots": test_summary["snapshots_with_labels"],
                }
            )
        return pd.DataFrame(rows)


def prepare_temporal_experiment_context(
    dataset_name: str,
    split_caps: Dict[str, int],
    *,
    device: Optional[torch.device] = None,
    seed: int = 43,
    preload_time_windows: Optional[Sequence[Optional[int]]] = None,
) -> TemporalExperimentContext:
    dataset, temporal_data = _load_temporal_data(dataset_name)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_nodes = int(temporal_data.num_nodes)
    output_dim = _safe_output_dim(dataset, num_nodes)
    node_features = _make_node_features(dataset, num_nodes, seed=seed).to(device)
    train_edge_ids, val_edge_ids, test_edge_ids, split_source = _split_edge_ids(dataset, temporal_data)
    context = TemporalExperimentContext(
        dataset_name=dataset_name,
        device=device,
        dataset=dataset,
        temporal_data=temporal_data,
        node_features=node_features,
        num_nodes=num_nodes,
        output_dim=output_dim,
        split_edge_ids={
            "train": train_edge_ids,
            "val": val_edge_ids,
            "test": test_edge_ids,
        },
        split_caps={key: int(value) for key, value in split_caps.items()},
        split_source=split_source,
        metric_name=getattr(dataset, "eval_metric", "ndcg"),
    )
    if preload_time_windows:
        for time_window in preload_time_windows:
            context.get_snapshot_bundle(time_window)
    return context


def make_trade_baseline_config() -> Dict[str, Any]:
    return {
        "time_window": 3,
        "lr": 0.012487940706903825,
        "weight_decay": 1.98417640446958e-07,
        "hidden_channels": 64,
        "layers": 3,
        "dropout": 0.0,
        "d": 4,
        "temporal_d_model": 32,
        "stateful_temporal": True,
        "closure_hops": 2,
        "temporal_bptt_steps": 8,
        "epochs": 200,
        "eval_every": 2,
        "early_stopping_patience": 24,
        "min_epochs_before_stopping": 30,
        "skip_train_eval": True,
        "min_delta": 0.0,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "sparse_learner": False,
        "use_act": True,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "add_lp": False,
        "add_hp": False,
        "max_t": 1.0,
    }


def _dedupe_key(parameter: str, value: Any):
    if parameter == "time_window":
        return _canonical_time_window(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _with_baseline(parameter: str, base_value, values: Iterable[Any]) -> List[Any]:
    ordered = [base_value, *values]
    normalized = []
    seen = set()
    for value in ordered:
        key = _dedupe_key(parameter, value)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(_canonical_time_window(value) if parameter == "time_window" else value)
    return normalized


def build_sensitivity_experiments(base_config: Dict[str, Any], parameter_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    experiments = []
    for parameter, values in parameter_grid.items():
        baseline_value = base_config.get(parameter)
        for value in _with_baseline(parameter, baseline_value, values):
            config = dict(base_config)
            config[parameter] = _canonical_time_window(value) if parameter == "time_window" else value
            experiments.append(
                {
                    "study": "sensitivity",
                    "parameter": parameter,
                    "value": config[parameter],
                    "value_label": _format_value(parameter, config[parameter]),
                    "experiment_name": f"{parameter}={_format_value(parameter, config[parameter])}",
                    "config": config,
                }
            )
    return experiments


def build_ablation_experiments(base_config: Dict[str, Any], flags: Sequence[str]) -> List[Dict[str, Any]]:
    experiments = []
    for flag in flags:
        for value in [False, True]:
            config = dict(base_config)
            config[flag] = bool(value)
            experiments.append(
                {
                    "study": "ablation",
                    "parameter": flag,
                    "value": bool(value),
                    "value_label": str(bool(value)),
                    "experiment_name": f"{flag}={bool(value)}",
                    "config": config,
                }
            )
    return experiments


def run_experiments(
    context: TemporalExperimentContext,
    experiments: Sequence[Dict[str, Any]],
    *,
    repeats: int = 5,
    seed_base: int = 43,
    progress: bool = True,
) -> pd.DataFrame:
    rows = []
    total_runs = len(experiments) * int(repeats)
    progress_bar = tqdm(total=total_runs, disable=not progress, desc="Temporal Mamba study runs")

    try:
        for experiment in experiments:
            config = dict(experiment["config"])
            snapshot_bundle = context.get_snapshot_bundle(config.get("time_window"))
            for repeat_idx in range(int(repeats)):
                seed = int(seed_base + repeat_idx)
                result = train_single_config(
                    dataset_name=context.dataset_name,
                    dataset=context.dataset,
                    snapshot_bundle=snapshot_bundle,
                    node_features=context.node_features,
                    num_nodes=context.num_nodes,
                    output_dim=context.output_dim,
                    device=context.device,
                    config=config,
                    seed=seed,
                    keep_history=False,
                )
                rows.append(
                    {
                        "study": experiment["study"],
                        "parameter": experiment["parameter"],
                        "value": experiment["value"],
                        "value_label": experiment["value_label"],
                        "experiment_name": experiment["experiment_name"],
                        "repeat": repeat_idx + 1,
                        "seed": seed,
                        "metric_name": result["metric_name"],
                        "best_epoch": result["best_epoch"],
                        "best_train_metric": result["best_train_metric"],
                        "best_val_metric": result["best_val_metric"],
                        "best_test_metric": result["best_test_metric"],
                        "best_train_loss": result["best_train_loss"],
                        "best_val_loss": result["best_val_loss"],
                        "best_test_loss": result["best_test_loss"],
                        "search_best_test_metric": result["search_best_test_metric"],
                        "search_best_val_loss": result["search_best_val_loss"],
                        "time_window": _canonical_time_window(config.get("time_window")),
                        "config_json": json.dumps(config, sort_keys=True),
                    }
                )
                progress_bar.update(1)
    finally:
        progress_bar.close()

    return pd.DataFrame(rows)


def summarize_experiments(raw_df: pd.DataFrame, group_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    if group_cols is None:
        group_cols = ["study", "parameter", "value", "value_label", "experiment_name"]

    metric_cols = [
        "best_epoch",
        "best_train_metric",
        "best_val_metric",
        "best_test_metric",
        "best_train_loss",
        "best_val_loss",
        "best_test_loss",
        "search_best_test_metric",
        "search_best_val_loss",
    ]
    summary = raw_df.groupby(list(group_cols), dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary.sort_values(by=["study", "parameter", "experiment_name"]).reset_index(drop=True)


def plot_sensitivity_results(
    summary_df: pd.DataFrame,
    *,
    metric_col: str = "best_test_metric_mean",
    error_col: str = "best_test_metric_std",
    parameter_order: Optional[Sequence[str]] = None,
    max_cols: int = 3,
):
    sensitivity_df = summary_df[summary_df["study"] == "sensitivity"].copy()
    if sensitivity_df.empty:
        raise ValueError("No sensitivity rows found in the provided summary dataframe.")

    if parameter_order is None:
        parameter_order = list(dict.fromkeys(sensitivity_df["parameter"].tolist()))
    parameters = [parameter for parameter in parameter_order if parameter in set(sensitivity_df["parameter"])]
    ncols = min(max_cols, max(len(parameters), 1))
    nrows = math.ceil(len(parameters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, parameter in zip(flat_axes, parameters):
        param_df = sensitivity_df[sensitivity_df["parameter"] == parameter].copy()
        if parameter == "time_window":
            sort_key = param_df["value"].map(lambda x: -1 if pd.isna(x) else int(x))
        elif pd.api.types.is_numeric_dtype(param_df["value"]):
            sort_key = param_df["value"]
        else:
            sort_key = param_df["value_label"]
        param_df = param_df.iloc[np.argsort(sort_key.to_numpy(), kind="stable")]
        x = np.arange(len(param_df))
        y = param_df[metric_col].to_numpy(dtype=float)
        yerr = param_df[error_col].fillna(0.0).to_numpy(dtype=float)
        labels = param_df["value_label"].tolist()

        ax.errorbar(x, y, yerr=yerr, fmt="o-", linewidth=1.8, capsize=4)
        ax.set_title(parameter)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("Value")
        ax.set_ylabel(metric_col.replace("_mean", ""))

    for ax in flat_axes[len(parameters):]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes
