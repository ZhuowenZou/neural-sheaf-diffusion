# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.mamba_models import MambaSheafDiffusion


DATASET_NAME = "tgbl-wiki-vs"


pytestmark = pytest.mark.benchmark


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _load_tgbl_wiki_vs():
    if os.environ.get("RUN_TGB_BENCHMARK") != "1":
        pytest.skip("Set RUN_TGB_BENCHMARK=1 to run the tgbl-wiki-vs benchmark.")

    tgb_dataset = pytest.importorskip("tgb.linkproppred.dataset_pyg")
    pytest.importorskip("tgb.linkproppred.evaluate")

    root = os.environ.get("TGB_ROOT", "datasets")
    answer = "y" if os.environ.get("TGB_DOWNLOAD") == "1" else "n"
    try:
        with patch("builtins.input", return_value=answer):
            return tgb_dataset.PyGLinkPropPredDataset(name=DATASET_NAME, root=root)
    except Exception as exc:
        pytest.skip(
            f"{DATASET_NAME} is not available under TGB_ROOT={root!r}. "
            "Pre-download it or set TGB_DOWNLOAD=1. "
            f"Original error: {exc}"
        )


def _make_static_edge_index(src, dst, num_edges):
    src = src[:num_edges].long()
    dst = dst[:num_edges].long()
    edge_index = torch.stack([src, dst], dim=0)
    reverse_edge_index = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, reverse_edge_index], dim=1).contiguous()


def _make_node_features(dataset, num_nodes):
    if dataset.node_feat is not None and dataset.node_feat.size(0) == num_nodes:
        return dataset.node_feat.float()

    feature_dim = _env_int("TGB_NODE_FEATURE_DIM", 64)
    generator = torch.Generator().manual_seed(_env_int("TGB_SEED", 43))
    return torch.randn(num_nodes, feature_dim, generator=generator)


def _make_model(edge_index, x, num_nodes, device):
    args = {
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
        "output_dim": num_nodes,
        "sheaf_act": "tanh",
        "second_linear": False,
        "orth": "householder",
        "edge_weights": False,
        "max_t": 1.0,
    }
    return MambaSheafDiffusion(edge_index.to(device), args).to(device)


def _edge_indices(mask, max_edges):
    idx = mask.nonzero(as_tuple=False).view(-1)
    return idx[:max_edges]


def _evaluate_mrr(dataset, logits, temporal_data, eval_idx, device):
    from tgb.linkproppred.evaluate import Evaluator

    dataset.load_val_ns()
    pos_src = temporal_data.src[eval_idx].cpu()
    pos_dst = temporal_data.dst[eval_idx].cpu()
    pos_ts = temporal_data.t[eval_idx].cpu()
    neg_samples = dataset.negative_sampler.query_batch(
        pos_src, pos_dst, pos_ts, split_mode="val"
    )

    neg_lengths = [len(neg) for neg in neg_samples]
    if min(neg_lengths) == 0:
        pytest.skip("TGB returned an empty validation negative set.")
    if len(set(neg_lengths)) > 1:
        min_negatives = min(neg_lengths)
        neg_samples = [neg[:min_negatives] for neg in neg_samples]

    pos_src = pos_src.to(device)
    pos_dst = pos_dst.to(device)
    neg_dst = torch.as_tensor(np.stack(neg_samples), dtype=torch.long, device=device)

    with torch.no_grad():
        y_pred_pos = logits[pos_src, pos_dst]
        y_pred_neg = logits[pos_src.view(-1, 1).expand_as(neg_dst), neg_dst]

    evaluator = Evaluator(name=DATASET_NAME)
    return evaluator.eval(
        {
            "y_pred_pos": y_pred_pos,
            "y_pred_neg": y_pred_neg,
            "eval_metric": dataset.eval_metric,
        }
    )


def test_mamba_sheaf_diffusion_tgbl_wiki_vs_benchmark():
    seed = _env_int("TGB_SEED", 43)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _load_tgbl_wiki_vs()
    temporal_data = dataset.get_TemporalData()
    num_nodes = int(dataset.num_nodes)

    train_idx = _edge_indices(dataset.train_mask, _env_int("TGB_MAX_TRAIN_EDGES", 2048))
    val_idx = _edge_indices(dataset.val_mask, _env_int("TGB_MAX_EVAL_EDGES", 256))
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        pytest.skip("tgbl-wiki-vs did not expose non-empty train/validation splits.")

    edge_index = _make_static_edge_index(
        temporal_data.src[train_idx], temporal_data.dst[train_idx], train_idx.numel()
    )
    x = _make_node_features(dataset, num_nodes).to(device)
    model = _make_model(edge_index, x, num_nodes, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(os.environ.get("TGB_LR", 1e-3)),
        weight_decay=float(os.environ.get("TGB_WEIGHT_DECAY", 0.0)),
    )

    train_src = temporal_data.src[train_idx].to(device)
    train_dst = temporal_data.dst[train_idx].to(device)

    start = time.perf_counter()
    epochs = _env_int("TGB_EPOCHS", 2)
    last_loss = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.nll_loss(logits[train_src], train_dst)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        logits = model(x)
    metrics = _evaluate_mrr(dataset, logits, temporal_data, val_idx, device)
    elapsed = time.perf_counter() - start

    mrr = float(metrics["mrr"])
    assert np.isfinite(last_loss)
    assert np.isfinite(mrr)
    assert 0.0 <= mrr <= 1.0
    print(
        f"{DATASET_NAME} MambaSheafDiffusion benchmark | "
        f"epochs={epochs} train_edges={train_idx.numel()} "
        f"val_edges={val_idx.numel()} loss={last_loss:.4f} "
        f"mrr={mrr:.4f} elapsed_sec={elapsed:.2f}"
    )
