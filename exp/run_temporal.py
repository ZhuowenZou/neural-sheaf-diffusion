#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from exp.temporal_utils import build_temporal_snapshots, sequence_loss
from models.mamba_models import (
    MambaSheafDiffusion,
    TemporalMambaSheafSheafOnlyDiffusion,
    TemporalMambaSheafSSMOnlyDiffusion,
)
from models.sparse_temporal_mamba import SparseTemporalMambaSheafDiffusion

try:
    import git
except Exception:  # pragma: no cover - optional dependency at runtime.
    git = None


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _load_tgb_dataset(dataset_name):
    # Load NodePropPred dataset (which includes message/edge property prediction)
    root = os.environ.get("TGB_ROOT", "datasets")
    answer = "y" if os.environ.get("TGB_DOWNLOAD") == "1" else "n"
    
    tgb_dataset = pytest_importorskip("tgb.nodeproppred.dataset_pyg")
    pytest_importorskip("tgb.nodeproppred.evaluate")
    with patch("builtins.input", return_value=answer):
        return tgb_dataset.PyGNodePropPredDataset(name=dataset_name, root=root)


def pytest_importorskip(module_name):
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - only used in runtime environments.
        raise RuntimeError(
            f"Required temporal dependency {module_name!r} is unavailable. "
            "Install the optional TGB dependencies before running this entry point."
        ) from exc
    return sys.modules[module_name]


def _make_node_features(dataset, num_nodes):
    # Try to get node features from dataset if available
    try:
        if hasattr(dataset, 'node_feat') and dataset.node_feat is not None and dataset.node_feat.size(0) == num_nodes:
            return dataset.node_feat.float()
    except (AttributeError, RuntimeError):
        pass

    # Fall back to random features
    feature_dim = _env_int("TGB_NODE_FEATURE_DIM", 64)
    generator = torch.Generator().manual_seed(_env_int("TGB_SEED", 43))
    return torch.randn(num_nodes, feature_dim, generator=generator)


def _normalize_sheaf_edge_index(edge_index):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None)
    return edge_index.contiguous()


def _make_model(edge_index, x, num_nodes, output_dim, device, args):
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
    model_cls = {
        "TemporalMambaSheafSheafOnly": TemporalMambaSheafSheafOnlyDiffusion,
        "TemporalMambaSheafSSMOnly": TemporalMambaSheafSSMOnlyDiffusion,
        "SparseTemporalMambaSheaf": SparseTemporalMambaSheafDiffusion,
    }.get(getattr(args, "model", None), MambaSheafDiffusion)
    return model_cls(_normalize_sheaf_edge_index(edge_index).to(device), model_args).to(device)


def _load_temporal_data(dataset_name):
    dataset = _load_tgb_dataset(dataset_name)
    temporal_data = dataset.get_TemporalData()
    return dataset, temporal_data


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


def _edge_indices(mask, max_edges):
    idx = mask.nonzero(as_tuple=False).view(-1)
    return idx[:max_edges]


def _split_edge_ids(dataset, temporal_data, args):
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
        time_window=time_window,
    )


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


def _evaluate_snapshot_sequence(dataset_name, dataset, snapshots, logits_sequence, device, split_mode, temporal_data=None):
    from tgb.nodeproppred.evaluate import Evaluator

    evaluator = Evaluator(name=dataset_name)
    batches = _node_label_batches(dataset, snapshots)
    all_y_pred = []
    all_y_true = []

    batch_index = 0
    for snapshot_index, snapshot in enumerate(snapshots):
        if batch_index >= len(batches):
            break
        batch_snapshot, label_srcs, labels = batches[batch_index]
        if batch_snapshot is not snapshot:
            continue
        logits = logits_sequence[snapshot_index]
        all_y_pred.append(logits.index_select(0, label_srcs.to(logits.device)).detach().cpu())
        all_y_true.append(labels.detach().cpu())
        batch_index += 1

    if not all_y_pred:
        return float("nan")

    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    try:
        score = evaluator.eval({
            "y_pred": all_y_pred,
            "y_true": all_y_true,
            "eval_metric": [dataset.eval_metric],
        })
        metric_val = list(score.values())[0] if isinstance(score, dict) else score
        return float(metric_val)
    except Exception:
        return float("nan")


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
    initial_state=None,
    compute_metric=True,
    compute_loss=True,
):
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
                losses.append(-(normalised_target * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean().detach().cpu())
            if compute_metric:
                all_y_pred.append(pred.detach().cpu())
                all_y_true.append(labels.detach().cpu())

    metric = float("nan")
    if compute_metric and all_y_pred:
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

    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    return metric, mean_loss, state


def _run_epoch(
    model,
    optimizer,
    train_snapshots,
    dataset,
    bptt_steps=None,
    show_progress=False,
    epoch_label=None,
):
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


def _context_sequence(model, snapshots, initial_state=None):
    model.eval()
    with torch.no_grad():
        return model.forward_sequence(snapshots, initial_state=initial_state)


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.model not in (
        "MambaSheaf",
        "TemporalMambaSheaf",
        "TemporalMambaSheafSheafOnly",
        "TemporalMambaSheafSSMOnly",
        "SparseTemporalMambaSheaf",
    ):
        raise ValueError("run_temporal.py only supports the temporal Mamba sheaf model.")

    sha = _repo_sha()

    dataset_name = args.temporal_dataset
    dataset, temporal_data = _load_temporal_data(dataset_name)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    num_nodes = int(temporal_data.num_nodes)
    output_dim = int(getattr(dataset, "num_classes", num_nodes))
    node_features = _make_node_features(dataset, num_nodes).to(device)

    train_edge_ids, val_edge_ids, test_edge_ids, split_source = _split_edge_ids(dataset, temporal_data, args)

    train_snapshots, train_edge_ids = _build_supervised_snapshots(
        dataset,
        temporal_data,
        train_edge_ids,
        node_features,
        args.temporal_max_edges_per_snapshot,
        args.temporal_snapshot_time_window,
        split_name="train",
    )
    val_snapshots, val_edge_ids = _build_supervised_snapshots(
        dataset,
        temporal_data,
        val_edge_ids,
        node_features,
        args.temporal_max_edges_per_snapshot,
        args.temporal_snapshot_time_window,
        split_name="validation",
    )
    test_snapshots, test_edge_ids = _build_supervised_snapshots(
        dataset,
        temporal_data,
        test_edge_ids,
        node_features,
        args.temporal_max_edges_per_snapshot,
        args.temporal_snapshot_time_window,
        split_name="test",
    )

    if not train_snapshots or not val_snapshots:
        raise RuntimeError("Temporal dataset did not produce non-empty train/validation snapshot sequences.")

    model = _make_model(train_snapshots[0].edge_index, node_features, num_nodes, output_dim, device, args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    wandb.init(project="sheaf-temporal", config=vars(args), entity=args.entity)
    best_val_metric = float("-inf")
    best_test_metric = float("nan")
    best_epoch = -1
    
    metric_name = getattr(dataset, "eval_metric", "ndcg")

    start = time.perf_counter()
    for epoch in tqdm(range(args.temporal_epochs)):
        model.reset_temporal_state()
        last_loss = _run_epoch(
            model,
            optimizer,
            train_snapshots,
            dataset,
            bptt_steps=args.temporal_bptt_steps,
            show_progress=args.temporal_epoch_progress_bar,
            epoch_label=f"Epoch {epoch + 1}/{args.temporal_epochs}",
        )

        should_eval = ((epoch + 1) % max(args.temporal_eval_every, 1) == 0) or (epoch == args.temporal_epochs - 1)
        log_payload = {
            "epoch": epoch,
            "train_loss": last_loss,
        }

        if should_eval:
            train_metric = float("nan")
            train_eval_loss = float("nan")
            if args.temporal_skip_train_eval:
                train_state = _advance_context(model, train_snapshots)
            else:
                train_metric, train_eval_loss, train_state = _evaluate_model_streaming(
                    dataset_name,
                    dataset,
                    train_snapshots,
                    model,
                    initial_state=None,
                )
                log_payload[f"train_{metric_name}"] = train_metric
                log_payload["train_eval_loss"] = train_eval_loss

            val_metric, val_eval_loss, val_state = _evaluate_model_streaming(
                dataset_name,
                dataset,
                val_snapshots,
                model,
                initial_state=train_state,
            )
            test_metric, test_eval_loss, _ = _evaluate_model_streaming(
                dataset_name,
                dataset,
                test_snapshots,
                model,
                initial_state=val_state,
            )

            if np.isfinite(val_metric) and val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
                best_epoch = epoch

            log_payload.update({
                f"val_{metric_name}": val_metric,
                f"test_{metric_name}": test_metric,
                "val_eval_loss": val_eval_loss,
                "test_eval_loss": test_eval_loss,
            })

        wandb.log(log_payload, step=epoch)

    elapsed = time.perf_counter() - start
    wandb.log(
        {
            f"best_val_{metric_name}": best_val_metric,
            f"best_test_{metric_name}": best_test_metric,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
        }
    )
    wandb.finish()

    model_name = args.model
    print(
        f"{model_name} on {dataset_name} | split={split_source} | SHA: {sha} | "
        f"best_val_{metric_name}={best_val_metric:.4f} best_test_{metric_name}={best_test_metric:.4f} "
        f"best_epoch={best_epoch} elapsed_sec={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
