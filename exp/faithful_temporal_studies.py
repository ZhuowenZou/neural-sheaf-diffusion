"""Training/benchmark runner for FaithfulTemporalSheafDiffusion on tgbn-* tasks.

Reuses the data preparation and TGB evaluation of ``temporal_mamba_studies``;
only the model factory and the BPTT epoch differ (the faithful model's state
carries the previous graph and timestamp, so detaching must go through
``TemporalSheafState.detach()`` rather than rebuilding memory/spatial only).
"""

import gc
import math
from typing import Any, Dict, Optional

import pandas as pd
import torch

from exp.temporal_mamba_studies import (  # noqa: F401 (re-exported helpers)
    _advance_context,
    _evaluate_model_streaming,
    _node_property_loss,
    _normalize_sheaf_edge_index,
    _seed_everything,
    make_trade_baseline_config,
    prepare_temporal_experiment_context,
)
from models.temporal_sheaf_ssm import FaithfulTemporalSheafDiffusion


def make_faithful_model(edge_index, x, num_nodes, output_dim, device, config):
    model_args = {
        "graph_size": num_nodes,
        "input_dim": int(x.size(1)),
        "output_dim": int(output_dim),
        "d": int(config["d"]),
        "hidden_channels": int(config["hidden_channels"]),
        "layers": int(config["layers"]),
        "dropout": float(config.get("dropout", 0.0)),
        "input_dropout": float(config.get("input_dropout", 0.0)),
        "use_act": bool(config.get("use_act", True)),
        "sheaf_act": str(config.get("sheaf_act", "tanh")),
        "orth": str(config.get("orth", "householder")),
        "add_lp": bool(config.get("add_lp", False)),
        "add_hp": bool(config.get("add_hp", False)),
        "closure_hops": int(config.get("closure_hops", 1)),
        "stateful_temporal": bool(config.get("stateful_temporal", True)),
        "temporal_d_model": int(config.get("temporal_d_model", 64)),
        "feedback_dim": int(config.get("feedback_dim", 32)),
        "diffusion_update": str(config.get("diffusion_update", "euler")),
        "dt_min": float(config.get("dt_min", 1e-3)),
        "dt_max": float(config.get("dt_max", 0.1)),
        "dt_cap": float(config.get("dt_cap", 0.25)),
        "psi_log_degree": bool(config.get("psi_log_degree", False)),
        "memory_readout": bool(config.get("memory_readout", False)),
        "sheaf_conditioning": str(config.get("sheaf_conditioning", "history")),
    }
    return FaithfulTemporalSheafDiffusion(
        _normalize_sheaf_edge_index(edge_index).to(device), model_args
    ).to(device)


def _run_epoch(model, optimizer, train_snapshots, dataset, bptt_steps=None, grad_clip=None):
    model.train()
    if bptt_steps is None or bptt_steps <= 0:
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = _node_property_loss(outputs, dataset, train_snapshots)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach().cpu())

    chunk_losses = []
    state = None
    for start in range(0, len(train_snapshots), bptt_steps):
        chunk = train_snapshots[start:start + bptt_steps]
        optimizer.zero_grad()
        outputs, state = model.forward_sequence(chunk, initial_state=state)
        loss = _node_property_loss(outputs, dataset, chunk)
        if loss.requires_grad:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            chunk_losses.append(float(loss.detach().cpu()))
        state = state.detach() if state is not None else None

    return float(sum(chunk_losses) / len(chunk_losses)) if chunk_losses else 0.0


def train_single_faithful(
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

    model = make_faithful_model(
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
    epochs_since_improvement = 0
    eval_every = max(int(config.get("eval_every", 1)), 1)
    min_delta = float(config.get("min_delta", 0.0))
    patience = int(config.get("early_stopping_patience", 10))
    min_epochs_before_stopping = int(config.get("min_epochs_before_stopping", eval_every))
    grad_clip = config.get("grad_clip")
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
            grad_clip=grad_clip,
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss}

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == int(config["epochs"]) - 1)
        if should_eval:
            train_state = _advance_context(model, snapshot_bundle["train_snapshots"])
            val_metric, val_loss, val_state = _evaluate_model_streaming(
                dataset_name, dataset, snapshot_bundle["val_snapshots"], model,
                initial_state=train_state,
            )
            test_metric, test_loss = float("nan"), float("nan")
            if snapshot_bundle["test_snapshots"]:
                test_metric, test_loss, _ = _evaluate_model_streaming(
                    dataset_name, dataset, snapshot_bundle["test_snapshots"], model,
                    initial_state=val_state,
                )
            row.update({
                f"val_{metric_name}": val_metric,
                "val_loss": val_loss,
                f"test_{metric_name}": test_metric,
                "test_loss": test_loss,
            })

            improved = (not math.isnan(val_metric)) and (val_metric > best_val_metric + min_delta)
            if improved:
                best_val_metric = float(val_metric)
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
    final_train_state = _advance_context(model, snapshot_bundle["train_snapshots"])
    final_val_metric, final_val_loss, final_val_state = _evaluate_model_streaming(
        dataset_name, dataset, snapshot_bundle["val_snapshots"], model,
        initial_state=final_train_state,
    )
    final_test_metric, final_test_loss = float("nan"), float("nan")
    if snapshot_bundle["test_snapshots"]:
        final_test_metric, final_test_loss, _ = _evaluate_model_streaming(
            dataset_name, dataset, snapshot_bundle["test_snapshots"], model,
            initial_state=final_val_state,
        )

    result = {
        "metric_name": metric_name,
        "best_epoch": best_epoch,
        "best_val_metric": final_val_metric,
        "best_val_loss": final_val_loss,
        "best_test_metric": final_test_metric,
        "best_test_loss": final_test_loss,
        "n_params": sum(p.numel() for p in model.parameters()),
        "history": pd.DataFrame(history) if keep_history else None,
    }

    del model
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_faithful_trade_config() -> Dict[str, Any]:
    """Trade baseline hyperparameters adapted for the faithful model."""
    config = make_trade_baseline_config()
    config.update(
        {
            "temporal_d_model": 64,
            "feedback_dim": 32,
            "diffusion_update": "euler",
            "grad_clip": 1.0,
            # ZOH training can transiently degrade before recovering (best
            # epochs up to ~90); the baseline's patience of 24 truncates that
            # recovery, so the faithful model gets a longer leash.
            "early_stopping_patience": 60,
            "min_epochs_before_stopping": 80,
        }
    )
    return config
