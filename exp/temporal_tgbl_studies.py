# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import exp.run_temporal_tgbl as tgbl_runner


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def _normalize_window_candidates(window_candidates: Iterable[Optional[int]]) -> List[Optional[int]]:
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


def make_window_candidates(timestamps, edge_ids, manual_candidates=None, limit=12) -> List[Optional[int]]:
    if manual_candidates is not None:
        return _normalize_window_candidates(manual_candidates)
    if edge_ids.numel() == 0:
        return [None]

    selected_ts = timestamps[edge_ids]
    span = int((selected_ts.max() - selected_ts.min()).item()) if selected_ts.numel() > 0 else 0
    if span <= 1:
        return [None, 1]

    raw = np.geomspace(1, span, num=limit)
    candidates = [None] + [int(x) for x in np.unique(np.round(raw).astype(int)).tolist() if int(x) > 0]
    return _normalize_window_candidates(candidates)


def estimate_snapshot_bucket_stats(timestamps, edge_ids, time_window=None, max_edges=None) -> Dict[str, Any]:
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
    }


def estimate_time_window_grid(temporal_data, edge_ids, max_edges, candidate_windows) -> pd.DataFrame:
    rows = []
    for time_window in candidate_windows:
        stats = estimate_snapshot_bucket_stats(
            temporal_data.t,
            edge_ids,
            time_window=time_window,
            max_edges=max_edges,
        )
        rows.append(
            {
                "time_window": time_window,
                "window_label": "exact timestamps" if time_window is None else str(time_window),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def recommend_time_window(estimate_df, target_max_snapshots=None, target_max_mean_edges=None):
    if estimate_df.empty:
        return None

    feasible = estimate_df.copy()
    if target_max_snapshots is not None:
        feasible = feasible[feasible["estimated_snapshots"] <= target_max_snapshots]
    if target_max_mean_edges is not None:
        feasible = feasible[feasible["estimated_mean_edges"] <= target_max_mean_edges]
    if feasible.empty:
        return None

    order = feasible["time_window"].map(lambda x: -1 if _canonical_time_window(x) is None else _canonical_time_window(x))
    return feasible.iloc[order.argsort(kind="stable")].iloc[0].to_dict()


def choose_search_windows(estimate_df, recommendation=None, max_windows=4) -> List[Optional[int]]:
    if estimate_df.empty:
        return [None]

    feasible = estimate_df.sort_values(
        by=["estimated_snapshots", "estimated_mean_edges", "time_window"],
        key=lambda series: (
            series.map(lambda x: -1 if _canonical_time_window(x) is None else _canonical_time_window(x))
            if series.name == "time_window"
            else series
        ),
    )

    picked = []
    if recommendation is not None:
        picked.append(_canonical_time_window(recommendation["time_window"]))

    exact_rows = feasible[feasible["time_window"].isna()]
    if not exact_rows.empty:
        picked.append(None)

    for value in feasible["time_window"].tolist():
        picked.append(_canonical_time_window(value))
        if len(_normalize_window_candidates(picked)) >= max_windows:
            break

    return _normalize_window_candidates(picked)[:max_windows]


def summarize_snapshots(snapshots) -> Dict[str, Any]:
    edge_counts = [int(snapshot.src.numel()) for snapshot in snapshots]
    active_counts = [int(snapshot.active_nodes.numel()) for snapshot in snapshots]
    timestamps = [int(snapshot.timestamp.item()) for snapshot in snapshots]
    return {
        "snapshots": len(snapshots),
        "mean_edges": float(np.mean(edge_counts)) if edge_counts else float("nan"),
        "max_edges": int(max(edge_counts)) if edge_counts else 0,
        "mean_active_nodes": float(np.mean(active_counts)) if active_counts else float("nan"),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


@dataclass
class TgblTemporalExperimentContext:
    dataset_name_requested: str
    dataset_name: str
    device: torch.device
    dataset: Any
    temporal_data: Any
    node_features: torch.Tensor
    num_nodes: int
    destination_spec: Dict[str, int]
    split_edge_ids: Dict[str, torch.Tensor]
    split_caps: Dict[str, int]
    split_source: str
    metric_name: str
    snapshot_cache: Dict[Optional[int], Dict[str, Any]] = field(default_factory=dict)

    def get_snapshot_bundle(self, time_window=None):
        key = _canonical_time_window(time_window)
        if key not in self.snapshot_cache:
            train_ids = self.split_edge_ids["train"][: self.split_caps["train"]]
            val_ids = self.split_edge_ids["val"][: self.split_caps["val"]]
            test_ids = self.split_edge_ids["test"][: self.split_caps["test"]]
            self.snapshot_cache[key] = {
                "time_window": key,
                "train_snapshots": tgbl_runner._build_snapshots(
                    self.temporal_data,
                    train_ids,
                    self.node_features,
                    max_edges=None,
                    time_window=key,
                ),
                "val_snapshots": tgbl_runner._build_snapshots(
                    self.temporal_data,
                    val_ids,
                    self.node_features,
                    max_edges=None,
                    time_window=key,
                ),
                "test_snapshots": tgbl_runner._build_snapshots(
                    self.temporal_data,
                    test_ids,
                    self.node_features,
                    max_edges=None,
                    time_window=key,
                ),
                "train_edge_count": int(train_ids.numel()),
                "val_edge_count": int(val_ids.numel()),
                "test_edge_count": int(test_ids.numel()),
            }
        return self.snapshot_cache[key]

    def snapshot_table(self) -> pd.DataFrame:
        rows = []
        for time_window, bundle in sorted(self.snapshot_cache.items(), key=lambda item: (-1 if item[0] is None else item[0])):
            train_summary = summarize_snapshots(bundle["train_snapshots"])
            val_summary = summarize_snapshots(bundle["val_snapshots"])
            test_summary = summarize_snapshots(bundle["test_snapshots"])
            rows.append(
                {
                    "time_window": time_window,
                    "window_label": _format_value("time_window", time_window),
                    "train_edges": bundle["train_edge_count"],
                    "train_snapshots": train_summary["snapshots"],
                    "train_mean_edges": train_summary["mean_edges"],
                    "val_edges": bundle["val_edge_count"],
                    "val_snapshots": val_summary["snapshots"],
                    "test_edges": bundle["test_edge_count"],
                    "test_snapshots": test_summary["snapshots"],
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
) -> TgblTemporalExperimentContext:
    dataset_name_resolved, dataset, temporal_data = tgbl_runner._load_temporal_data(dataset_name)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_nodes = tgbl_runner._infer_graph_size(temporal_data)
    destination_spec = tgbl_runner._destination_spec(temporal_data)
    node_features = tgbl_runner._make_node_features(dataset, num_nodes).to(device)
    train_edge_ids, val_edge_ids, test_edge_ids, split_source = tgbl_runner._split_edge_ids(dataset, temporal_data)

    context = TgblTemporalExperimentContext(
        dataset_name_requested=dataset_name,
        dataset_name=dataset_name_resolved,
        device=device,
        dataset=dataset,
        temporal_data=temporal_data,
        node_features=node_features,
        num_nodes=num_nodes,
        destination_spec=destination_spec,
        split_edge_ids={
            "train": train_edge_ids,
            "val": val_edge_ids,
            "test": test_edge_ids,
        },
        split_caps={key: int(value) for key, value in split_caps.items()},
        split_source=split_source,
        metric_name=getattr(dataset, "eval_metric", "mrr"),
    )
    if preload_time_windows:
        for time_window in preload_time_windows:
            context.get_snapshot_bundle(time_window)
    return context


def make_tgbl_baseline_config() -> Dict[str, Any]:
    return {
        "model": "TemporalMambaSheaf",
        "time_window": None,
        "lr": 3e-3,
        "weight_decay": 1e-6,
        "hidden_channels": 8,
        "layers": 2,
        "dropout": 0.0,
        "d": 2,
        "temporal_d_model": 32,
        "stateful_temporal": True,
        "closure_hops": 1,
        "temporal_bptt_steps": 8,
        "epochs": 40,
        "eval_every": 4,
        "early_stopping_patience": 12,
        "min_epochs_before_stopping": 12,
        "skip_train_eval": True,
        "evaluate_test": False,
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


def sample_trial_config(
    trial,
    *,
    search_windows,
    search_epochs,
    eval_every,
    patience,
    min_epochs_before_stopping,
    evaluate_test=False,
):
    return {
        "model": "TemporalMambaSheaf",
        "time_window": trial.suggest_categorical("time_window", list(search_windows)),
        "lr": trial.suggest_float("lr", 1e-4, 2e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "hidden_channels": trial.suggest_categorical("hidden_channels", [4, 8, 16, 24, 32]),
        "layers": trial.suggest_categorical("layers", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.05),
        "d": trial.suggest_categorical("d", [2, 4]),
        "temporal_d_model": trial.suggest_categorical("temporal_d_model", [16, 32, 64]),
        "stateful_temporal": True,
        "closure_hops": trial.suggest_categorical("closure_hops", [0, 1, 2]),
        "temporal_bptt_steps": trial.suggest_categorical("temporal_bptt_steps", [2, 4, 8, 16]),
        "epochs": int(search_epochs),
        "eval_every": int(eval_every),
        "early_stopping_patience": int(patience),
        "min_epochs_before_stopping": int(min_epochs_before_stopping),
        "skip_train_eval": True,
        "evaluate_test": bool(evaluate_test),
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


def _find_metric_column(df, split_name, metric_name):
    preferred = [
        f"{split_name}_{metric_name}",
        f"{split_name}_{str(metric_name).lower()}",
        f"{split_name}_{str(metric_name).upper()}",
    ]
    for col in preferred:
        if col in df.columns and not df[col].isna().all():
            return col
    prefix = f"{split_name}_"
    candidates = [
        col
        for col in df.columns
        if col.startswith(prefix)
        and col not in {f"{split_name}_loss", f"{split_name}_eval_loss"}
        and not df[col].isna().all()
    ]
    return candidates[0] if candidates else None


def train_single_config(
    *,
    context: TgblTemporalExperimentContext,
    snapshot_bundle: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
    keep_history: bool = False,
    trial=None,
):
    if not snapshot_bundle["train_snapshots"] or not snapshot_bundle["val_snapshots"]:
        raise RuntimeError("Snapshot bundle must include non-empty train and validation sequences.")

    _seed_everything(seed)

    model = tgbl_runner._make_model(
        snapshot_bundle["train_snapshots"][0].edge_index,
        context.node_features,
        context.num_nodes,
        context.destination_spec,
        context.device,
        type("Args", (), {
            "model": config.get("model", "TemporalMambaSheaf"),
            "stateful_temporal": config.get("stateful_temporal", True),
            "closure_hops": config.get("closure_hops", 1),
            "temporal_d_model": config.get("temporal_d_model"),
        })(),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metric_name = context.metric_name
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
    evaluate_test = bool(config.get("evaluate_test", True))

    for epoch in range(int(config["epochs"])):
        model.reset_temporal_state()
        train_loss = tgbl_runner._run_epoch(
            model,
            optimizer,
            snapshot_bundle["train_snapshots"],
            context.destination_spec,
            bptt_steps=bptt_steps,
            show_progress=False,
            epoch_label=f"Epoch {epoch + 1}/{config['epochs']}",
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss}

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == int(config["epochs"]) - 1)
        if should_eval:
            if config.get("skip_train_eval", True):
                train_metric = float("nan")
                train_eval_loss = float("nan")
                train_state = tgbl_runner._advance_context(model, snapshot_bundle["train_snapshots"])
            else:
                train_metric, train_eval_loss, train_state = tgbl_runner._evaluate_model_streaming(
                    context.dataset_name,
                    context.dataset,
                    snapshot_bundle["train_snapshots"],
                    model,
                    context.destination_spec,
                    initial_state=None,
                    split_mode="train",
                    compute_metric=False,
                    compute_loss=True,
                )
                row[f"train_{metric_name}"] = train_metric
                row["train_eval_loss"] = train_eval_loss

            val_metric, val_loss, val_state = tgbl_runner._evaluate_model_streaming(
                context.dataset_name,
                context.dataset,
                snapshot_bundle["val_snapshots"],
                model,
                context.destination_spec,
                initial_state=train_state,
                split_mode="val",
                compute_metric=True,
                compute_loss=True,
            )

            test_metric = float("nan")
            test_loss = float("nan")
            if evaluate_test and snapshot_bundle["test_snapshots"]:
                test_metric, test_loss, _ = tgbl_runner._evaluate_model_streaming(
                    context.dataset_name,
                    context.dataset,
                    snapshot_bundle["test_snapshots"],
                    model,
                    context.destination_spec,
                    initial_state=val_state,
                    split_mode="test",
                    compute_metric=True,
                    compute_loss=True,
                )

            row.update(
                {
                    f"val_{metric_name}": val_metric,
                    "val_loss": val_loss,
                    f"test_{metric_name}": test_metric,
                    "test_loss": test_loss,
                }
            )

            if trial is not None and not math.isnan(val_metric):
                trial.report(float(val_metric), step=epoch + 1)
                if trial.should_prune():
                    raise __import__("optuna").exceptions.TrialPruned()

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
        final_train_state = tgbl_runner._advance_context(model, snapshot_bundle["train_snapshots"])
    else:
        final_train_metric, final_train_loss, final_train_state = tgbl_runner._evaluate_model_streaming(
            context.dataset_name,
            context.dataset,
            snapshot_bundle["train_snapshots"],
            model,
            context.destination_spec,
            initial_state=None,
            split_mode="train",
            compute_metric=False,
            compute_loss=True,
        )

    final_val_metric, final_val_loss, final_val_state = tgbl_runner._evaluate_model_streaming(
        context.dataset_name,
        context.dataset,
        snapshot_bundle["val_snapshots"],
        model,
        context.destination_spec,
        initial_state=final_train_state,
        split_mode="val",
        compute_metric=True,
        compute_loss=True,
    )
    final_test_metric = float("nan")
    final_test_loss = float("nan")
    if evaluate_test and snapshot_bundle["test_snapshots"]:
        final_test_metric, final_test_loss, _ = tgbl_runner._evaluate_model_streaming(
            context.dataset_name,
            context.dataset,
            snapshot_bundle["test_snapshots"],
            model,
            context.destination_spec,
            initial_state=final_val_state,
            split_mode="test",
            compute_metric=True,
            compute_loss=True,
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
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def top_completed_trials(study, top_k=5):
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    completed_df = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if completed_df.empty:
        return completed_df, []
    completed_df = completed_df.sort_values(by="value", ascending=False).reset_index(drop=True)
    configs = []
    for _, row in completed_df.head(int(top_k)).iterrows():
        config_json = row.get("user_attrs_config_json")
        if isinstance(config_json, str) and config_json:
            config = json.loads(config_json)
            if "time_window" in config:
                config["time_window"] = _canonical_time_window(config["time_window"])
        else:
            config = {}
            for column, value in row.items():
                if column.startswith("params_"):
                    key = column[len("params_") :]
                    config[key] = _canonical_time_window(value) if key == "time_window" else value
        configs.append(config)
    return completed_df, configs


def merge_final_config(best_params, *, final_epochs, eval_every, patience, min_epochs_before_stopping, skip_train_eval=True, evaluate_test=True):
    config = dict(best_params)
    config.update(
        {
            "epochs": int(final_epochs),
            "eval_every": int(eval_every),
            "early_stopping_patience": int(patience),
            "min_epochs_before_stopping": int(min_epochs_before_stopping),
            "skip_train_eval": bool(skip_train_eval),
            "evaluate_test": bool(evaluate_test),
            "min_delta": 0.0,
        }
    )
    return config


def config_frame(config):
    rows = []
    for key, value in config.items():
        normalized_value = _canonical_time_window(value) if key == "time_window" else value
        rows.append(
            {
                "field": key,
                "value": "exact timestamps" if key == "time_window" and normalized_value is None else normalized_value,
            }
        )
    return pd.DataFrame(rows)


def render_reproduction_command(config, dataset_name, split_caps):
    cli = [
        "PYTHONPATH=. python -m exp.run_temporal_tgbl",
        f"    --dataset={dataset_name}",
        f"    --temporal_dataset={dataset_name}",
        "    --model=TemporalMambaSheaf",
        f"    --lr={config['lr']}",
        f"    --weight_decay={config['weight_decay']}",
        f"    --stateful_temporal={str(config['stateful_temporal'])}",
        f"    --closure_hops={config['closure_hops']}",
        f"    --temporal_d_model={config['temporal_d_model']}",
        f"    --temporal_epochs={config['epochs']}",
        f"    --temporal_train_edges={split_caps['train']}",
        f"    --temporal_val_edges={split_caps['val']}",
        f"    --temporal_test_edges={split_caps['test']}",
        f"    --temporal_eval_every={config['eval_every']}",
        f"    --temporal_skip_train_eval={str(config.get('skip_train_eval', True))}",
    ]
    if config.get("temporal_bptt_steps") is not None:
        cli.append(f"    --temporal_bptt_steps={config['temporal_bptt_steps']}")
    time_window = _canonical_time_window(config.get("time_window"))
    if time_window is not None:
        cli.append(f"    --temporal_snapshot_time_window={time_window}")

    env_lines = [
        f"export TGB_SHEAF_D={config['d']}",
        f"export TGB_LAYERS={config['layers']}",
        f"export TGB_HIDDEN_CHANNELS={config['hidden_channels']}",
        f"export TGB_DROPOUT={config['dropout']}",
    ]
    return "\n".join(env_lines + ["\\"] + [" \\\n".join(cli)])


def rerank_configs(
    context: TgblTemporalExperimentContext,
    configs: Sequence[Dict[str, Any]],
    *,
    seed_base: int = 43,
    progress: bool = True,
) -> pd.DataFrame:
    rows = []
    progress_bar = tqdm(total=len(configs), disable=not progress, desc="tgbl rerank runs")
    try:
        for idx, config in enumerate(configs):
            config = dict(config)
            snapshot_bundle = context.get_snapshot_bundle(config.get("time_window"))
            result = train_single_config(
                context=context,
                snapshot_bundle=snapshot_bundle,
                config=config,
                seed=seed_base + idx,
                keep_history=False,
                trial=None,
            )
            rows.append(
                {
                    "rank_candidate": idx + 1,
                    "metric_name": result["metric_name"],
                    "best_epoch": result["best_epoch"],
                    "best_train_metric": result["best_train_metric"],
                    "best_val_metric": result["best_val_metric"],
                    "best_test_metric": result["best_test_metric"],
                    "best_train_loss": result["best_train_loss"],
                    "best_val_loss": result["best_val_loss"],
                    "best_test_loss": result["best_test_loss"],
                    "time_window": _canonical_time_window(config.get("time_window")),
                    "config_json": json.dumps(config, sort_keys=True),
                }
            )
            progress_bar.update(1)
    finally:
        progress_bar.close()
    return pd.DataFrame(rows).sort_values(by="best_val_metric", ascending=False).reset_index(drop=True)
