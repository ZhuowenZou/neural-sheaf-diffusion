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


def _guarded_step(model, optimizer, loss, grad_clip):
    """Backward + clip + step, skipping the update when the loss or the
    gradient norm is non-finite (the same safeguard the event harness applies;
    a single NaN step would otherwise poison the parameters for the rest of
    the run).  Returns True when a step was taken."""
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return False
    loss.backward()
    if grad_clip:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(norm):
            optimizer.zero_grad()
            return False
    optimizer.step()
    return True


def _run_epoch(model, optimizer, train_snapshots, dataset, bptt_steps=None, grad_clip=None):
    model.train()
    if bptt_steps is None or bptt_steps <= 0:
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = _node_property_loss(outputs, dataset, train_snapshots)
        if not _guarded_step(model, optimizer, loss, grad_clip):
            print("  non-finite loss/gradient: step skipped", flush=True)
            return float("nan")
        return float(loss.detach().cpu())

    chunk_losses = []
    skipped = 0
    n_chunks = 0
    state = None
    from exp.temporal_benchmark_utils import seek_label_cursor
    seek_label_cursor(dataset, train_snapshots)   # once per epoch; chunks continue the cursor
    for start in range(0, len(train_snapshots), bptt_steps):
        chunk = train_snapshots[start:start + bptt_steps]
        optimizer.zero_grad()
        outputs, state = model.forward_sequence(chunk, initial_state=state)
        loss = _node_property_loss(outputs, dataset, chunk, seek=False)
        if loss.requires_grad:
            n_chunks += 1
            if _guarded_step(model, optimizer, loss, grad_clip):
                chunk_losses.append(float(loss.detach().cpu()))
            else:
                skipped += 1
        state = state.detach() if state is not None else None
    if skipped:
        print(f"  non-finite chunks skipped this epoch: {skipped}/{n_chunks}", flush=True)

    return float(sum(chunk_losses) / len(chunk_losses)) if chunk_losses else float("nan")


def _capture_rng():
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    import random

    import numpy as np

    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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

    # Periodic checkpoint + resume: long runs on the shared node die to events
    # outside our control (co-location memory pressure, node-wide GPU resets;
    # audit ledger 2026-09-10).  config["checkpoint_path"] enables it; the file
    # is written atomically every config["checkpoint_every"] epochs and, if it
    # exists at start, training resumes from it (model, optimizer, history,
    # best state, early-stopping counters and RNG states).
    import os as _os

    checkpoint_path = config.get("checkpoint_path")
    checkpoint_every = max(int(config.get("checkpoint_every", 1)), 1)
    start_epoch = 0
    if checkpoint_path and _os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = list(ckpt["history"])
        best_val_metric = float(ckpt["best_val_metric"])
        best_epoch = int(ckpt["best_epoch"])
        best_state_dict = ckpt["best_state_dict"]
        epochs_since_improvement = int(ckpt["epochs_since_improvement"])
        start_epoch = int(ckpt["epoch"])
        _restore_rng(ckpt.get("rng"))
        print(f"resumed from checkpoint {checkpoint_path} at epoch {start_epoch} "
              f"(best epoch {best_epoch}, best val {best_val_metric:.4f})", flush=True)

    def _save_checkpoint(next_epoch):
        if not checkpoint_path:
            return
        payload = {
            "epoch": int(next_epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "best_state_dict": best_state_dict,
            "epochs_since_improvement": epochs_since_improvement,
            "rng": _capture_rng(),
            "seed": seed,
        }
        tmp = checkpoint_path + ".tmp"
        torch.save(payload, tmp)
        _os.replace(tmp, checkpoint_path)

    for epoch in range(start_epoch, int(config["epochs"])):
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
                print(f"[epoch {epoch + 1}] {row} (early stop; best epoch {best_epoch})", flush=True)
                _save_checkpoint(int(config["epochs"]))  # nothing left to train on resume
                break

        history.append(row)
        # per-epoch progress line (flushed) so long runs on shared GPUs are observable
        print(f"[epoch {epoch + 1}] {row} (best epoch {best_epoch}, best val {best_val_metric:.4f})", flush=True)
        if (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(epoch + 1)

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
    if device.type == "cuda" and not int(_os.environ.get("TSD_RESERVE_GPU_MB", "0") or 0):
        # keep the start-up reservation (see temporal_benchmark_utils.reserve_gpu_memory)
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
