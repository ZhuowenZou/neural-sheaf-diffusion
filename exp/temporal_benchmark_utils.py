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
from models.mamba_models import (
    MambaSheafDiffusion,
    TemporalMambaSheafSheafOnlyDiffusion,
    TemporalMambaSheafSSMOnlyDiffusion,
)
from models.sparse_temporal_mamba import (
    EventTemporalMambaSheafDiffusion,
    SparseTemporalMambaSheafDiffusion,
)


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
    if dataset_name.startswith("synth-history:"):
        # review handoff, section 4: the history-dependent synthetic task, served through the
        # same interface as a TGB link dataset (MRR with fixed per-query negatives)
        from exp.review.synthetic_history import SyntheticHistoryDataset
        dataset = SyntheticHistoryDataset(dataset_name.split(":", 1)[1])
        spec = DatasetSpec(requested_name=dataset_name, loader_name="tgbl-wiki", task_family="linkprop",
                           dataset_module="tgb.linkproppred.dataset_pyg", evaluator_module="tgb.linkproppred.evaluate",
                           metric_name="mrr", train_metric_supported=False,
                           notes="synthetic history task; evaluator name tgbl-wiki only selects the MRR/hits@10 metric code")
        return spec, dataset, dataset.get_TemporalData()
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
        "num_relations": int(getattr(args, "temporal_num_relations", 0) or 0),
        "train_negatives_per_pos": int(getattr(args, "temporal_train_negatives_per_pos", 32)),
        "candidate_chunk_size": int(getattr(args, "temporal_candidate_chunk_size", 2048)),
        "max_score_elements": int(getattr(args, "temporal_max_score_elements", 4_000_000)),
    }
    model_cls = {
        "TemporalMambaSheafSheafOnly": TemporalMambaSheafSheafOnlyDiffusion,
        "TemporalMambaSheafSSMOnly": TemporalMambaSheafSSMOnlyDiffusion,
        "SparseTemporalMambaSheaf": SparseTemporalMambaSheafDiffusion,
        "EventTemporalMambaSheaf": EventTemporalMambaSheafDiffusion,
    }.get(getattr(args, "model", None), MambaSheafDiffusion)
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
        edge_ids=getattr(snapshot, "edge_ids", None),
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


_NONFINITE_POSITIVES = [0]
_NONFINITE_NEGATIVES = [0]
_SKIPPED_QUERIES = [0]

# Training-side numerical counters (serialised into results.csv by the runners;
# review handoff 2026-09-22, section 6).  Reset by `reset_train_counters`.
TRAIN_COUNTERS = {"steps": 0, "nonfinite_loss": 0, "nonfinite_grad": 0, "clipped": 0, "skipped_steps": 0}


def reset_train_counters():
    for k in TRAIN_COUNTERS:
        TRAIN_COUNTERS[k] = 0


# Isolated RNG stream for training negatives (review handoff, section 3): when a
# generator is installed here, negative destinations are drawn from it instead
# of the global torch RNG, so architecture-dependent draws (initialisation,
# dropout) cannot change the sampled training task between paired arms.
NEGATIVE_RNG = {"generator": None}


def install_negative_rng(seed, epoch, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1_000_003 + int(epoch) * 7919 + 17)
    NEGATIVE_RNG["generator"] = gen
    return gen


class QueryValidityAudit:
    """Per-query score-validity audit (review handoff, section 0).

    Records, BEFORE numerical substitution and before the pad mask is applied,
    the validity of every positive score and the number of NaN / +inf / -inf
    negative logits per query, together with the candidate count, the intended
    (pad) mask count and the action taken.  Aggregates are exact per split; a
    bounded list of failing queries keeps replay information (global edge id,
    split, snapshot index, timestamp, relation).  Three reciprocal ranks are
    accumulated per query: TGB's own semantics on the RAW scores (a NaN positive
    ranks first), the guarded scores the harness reports, and a conservative
    diagnostic that assigns reciprocal rank 0 to any query with an invalid
    positive or an invalid negative (full denominator kept).  Parity of the
    three establishes that no substitution changed the reported metric."""

    def __init__(self, max_failures=20000, k=10):
        self.max_failures = int(max_failures)
        self.k = int(k)
        self.splits = {}
        self.failures = []
        self.state_checks = []

    def _agg(self, split):
        if split not in self.splits:
            self.splits[split] = dict(
                queries=0, snapshots=0, candidates=0, pad_masked=0,
                pos_nonfinite=0, pos_nan=0, pos_posinf=0, pos_neginf=0,
                neg_nan=0, neg_posinf=0, neg_neginf=0, queries_with_invalid_neg=0,
                queries_affected=0, transformed_scores=0,
                rr_tgb_raw=0.0, rr_guarded=0.0, rr_conservative=0.0, hits_guarded=0.0,
                state_nonfinite_snapshots=0, rec_nonfinite_snapshots=0,
            )
        return self.splits[split]

    @staticmethod
    def _ranks(pos, neg):
        # TGB semantics (numpy comparisons; NaN compares False)
        pos = pos.reshape(-1, 1)
        opt = (neg > pos).sum(axis=1)
        pes = (neg >= pos).sum(axis=1)
        return 0.5 * (opt + pes) + 1.0

    def record_batch(self, split, snapshot_index, snapshot, pos_raw, neg_raw, pad_mask,
                     pos_guarded, neg_guarded):
        a = self._agg(split)
        pos_np = pos_raw.detach().cpu().numpy().astype(np.float64)
        neg_np = neg_raw.detach().cpu().numpy().astype(np.float64)
        mask = np.asarray(pad_mask, dtype=bool) if pad_mask is not None else np.ones_like(neg_np, dtype=bool)
        n, kc = neg_np.shape
        a["queries"] += n
        a["snapshots"] += 1
        a["candidates"] += int(mask.sum())
        a["pad_masked"] += int((~mask).sum())
        pos_nan = np.isnan(pos_np); pos_pinf = pos_np == np.inf; pos_ninf = pos_np == -np.inf
        neg_nan = np.isnan(neg_np) & mask; neg_pinf = (neg_np == np.inf) & mask; neg_ninf = (neg_np == -np.inf) & mask
        a["pos_nan"] += int(pos_nan.sum()); a["pos_posinf"] += int(pos_pinf.sum()); a["pos_neginf"] += int(pos_ninf.sum())
        a["pos_nonfinite"] += int((pos_nan | pos_pinf | pos_ninf).sum())
        a["neg_nan"] += int(neg_nan.sum()); a["neg_posinf"] += int(neg_pinf.sum()); a["neg_neginf"] += int(neg_ninf.sum())
        bad_neg_q = (neg_nan | neg_pinf | neg_ninf).any(axis=1)
        bad_pos_q = pos_nan | pos_pinf | pos_ninf
        affected = bad_neg_q | bad_pos_q
        a["queries_with_invalid_neg"] += int(bad_neg_q.sum())
        a["queries_affected"] += int(affected.sum())
        a["transformed_scores"] += int((pos_nan | pos_pinf).sum() + (neg_nan | neg_pinf).sum())
        # raw TGB semantics on the intended candidate set (pads excluded by -inf, as the harness does)
        neg_raw_masked = np.where(mask, neg_np, -np.inf)
        rr_raw = 1.0 / self._ranks(pos_np, neg_raw_masked)
        rr_g = 1.0 / self._ranks(pos_guarded.detach().cpu().numpy().astype(np.float64),
                                 neg_guarded.detach().cpu().numpy().astype(np.float64))
        rr_c = np.where(affected, 0.0, rr_g)
        a["rr_tgb_raw"] += float(rr_raw.sum()); a["rr_guarded"] += float(rr_g.sum()); a["rr_conservative"] += float(rr_c.sum())
        a["hits_guarded"] += float((self._ranks(pos_guarded.detach().cpu().numpy().astype(np.float64),
                                                neg_guarded.detach().cpu().numpy().astype(np.float64)) <= self.k).sum())
        if affected.any() and len(self.failures) < self.max_failures:
            ids = getattr(snapshot, "edge_ids", None)
            ts = getattr(snapshot, "edge_timestamps", None)
            et = getattr(snapshot, "edge_types", None)
            for i in np.nonzero(affected)[0][: self.max_failures - len(self.failures)]:
                self.failures.append(dict(
                    split=split, snapshot_index=int(snapshot_index), row=int(i),
                    edge_id=int(ids[i]) if ids is not None else -1,
                    src=int(snapshot.src[i]), dst=int(snapshot.dst[i]),
                    timestamp=float(ts[i]) if ts is not None else float(snapshot.timestamp),
                    relation=int(et[i]) if et is not None else -1,
                    positive_raw=float(pos_np[i]), positive_valid=bool(~bad_pos_q[i]),
                    neg_nan=int(neg_nan[i].sum()), neg_posinf=int(neg_pinf[i].sum()), neg_neginf=int(neg_ninf[i].sum()),
                    candidates=int(mask[i].sum()), pad_masked=int((~mask[i]).sum()),
                    rr_tgb_raw=float(rr_raw[i]), rr_guarded=float(rr_g[i]), rr_conservative=float(rr_c[i]),
                    action="positive->-inf" if bad_pos_q[i] else "negatives(nan,+inf)->-inf",
                ))

    def record_state(self, split, snapshot_index, model, state):
        """Finite checks of the persistent state and the recurrency caches after
        the snapshot was ingested (locates the first failure; nothing is repaired)."""
        a = self._agg(split)
        bad_state = False
        for name in ("memory", "spatial"):
            t = getattr(state, name, None) if state is not None else None
            if t is not None and not bool(torch.isfinite(t).all()):
                bad_state = True
        bad_rec = False
        for _, store in getattr(model, "_rec_stores", []):
            if store.size:
                seen = store.count > 0
                if not bool(torch.isfinite(store.count).all()) or not bool(torch.isfinite(store.last_t[seen]).all()):
                    bad_rec = True
        if bad_state:
            a["state_nonfinite_snapshots"] += 1
        if bad_rec:
            a["rec_nonfinite_snapshots"] += 1
        if (bad_state or bad_rec) and len(self.state_checks) < self.max_failures:
            self.state_checks.append(dict(split=split, snapshot_index=int(snapshot_index),
                                          state_nonfinite=bad_state, rec_nonfinite=bad_rec))

    def summary_rows(self, tag=None):
        rows = []
        for split, a in self.splits.items():
            q = max(a["queries"], 1)
            rows.append(dict(tag=tag, split=split, **a,
                             mrr_tgb_raw=a["rr_tgb_raw"] / q, mrr_guarded=a["rr_guarded"] / q,
                             mrr_conservative=a["rr_conservative"] / q, hits10_guarded=a["hits_guarded"] / q,
                             parity_raw_vs_guarded=abs(a["rr_tgb_raw"] - a["rr_guarded"]) < 1e-9,
                             parity_guarded_vs_conservative=abs(a["rr_guarded"] - a["rr_conservative"]) < 1e-9))
        return rows

    def write(self, out_dir, tag=None):
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(self.summary_rows(tag)).to_csv(os.path.join(out_dir, "query_validity.csv"), index=False)
        pd.DataFrame(self.failures).to_csv(os.path.join(out_dir, "query_validity_failures.csv"), index=False)
        pd.DataFrame(self.state_checks).to_csv(os.path.join(out_dir, "state_validity_failures.csv"), index=False)


# The active audit collector (None = disabled).  Runners install one with
# `QUERY_AUDIT["audit"] = QueryValidityAudit()` and write it out afterwards.
QUERY_AUDIT = {"audit": None}


def _guard_nonfinite_scores(y_pred_pos, y_pred_neg):
    """TGB's link evaluator ranks a non-finite positive FIRST (NaN comparisons
    are False -> optimistic rank 1 -> MRR 1.0).  A non-finite model score is an
    invalid prediction, so it is ranked LAST here (-inf) and counted; NaN/+inf
    negatives are set to -inf (they cannot outrank anything) and are counted too
    (review handoff, section 0: a removed invalid negative can improve the
    positive's rank, so negative failures must be visible).  Audit finding #3."""
    pos_bad = ~torch.isfinite(y_pred_pos)
    if bool(pos_bad.any()):
        _NONFINITE_POSITIVES[0] += int(pos_bad.sum().item())
        y_pred_pos = y_pred_pos.masked_fill(pos_bad, float("-inf"))
    neg_bad = torch.isnan(y_pred_neg) | (y_pred_neg == float("inf"))
    if bool(neg_bad.any()):
        _NONFINITE_NEGATIVES[0] += int(neg_bad.sum().item())
        y_pred_neg = y_pred_neg.masked_fill(neg_bad, float("-inf"))
    return y_pred_pos, y_pred_neg


def reserve_gpu_memory(device, mib=None, min_fraction=0.25, retries_per_size=6, retry_sleep=30.0):
    """Pre-reserve GPU memory in PyTorch's caching allocator.

    On a shared node a job that starts on a card with enough free memory can
    be squeezed minutes later by another user's arrival (a MAGMA workspace
    allocation then aborts the process; audit ledger 2026-09-09).  Allocating
    and freeing one large block at start-up keeps that memory reserved by the
    caching allocator for the life of the process (nothing in the training
    path calls ``torch.cuda.empty_cache``), so later allocations are served
    from it.  Amount: ``mib`` or the ``TSD_RESERVE_GPU_MB`` environment
    variable (set by ``wait_launch.sh`` from the job's memory request).  If
    the card fills up between the launcher's check and this call, the same
    amount is retried a few times, then halved down to ``min_fraction``.
    Returns the MiB actually reserved (0 when disabled or on CPU)."""
    import time as _time

    mib = int(os.environ.get("TSD_RESERVE_GPU_MB", "0") or 0) if mib is None else int(mib)
    if mib <= 0 or device is None or torch.device(device).type != "cuda":
        return 0
    target = mib
    while target >= max(int(mib * min_fraction), 1):
        for attempt in range(retries_per_size):
            try:
                block = torch.empty(int(target) * 2 ** 20, dtype=torch.uint8, device=device)
                del block
                print(f"reserved {target} MiB of GPU memory on {device} in the caching allocator "
                      f"(requested {mib}; now reserved {torch.cuda.memory_reserved(device) / 2**20:.0f} MiB)", flush=True)
                return target
            except torch.cuda.OutOfMemoryError:
                if attempt < retries_per_size - 1:
                    _time.sleep(retry_sleep)
        target //= 2
    print(f"WARNING: could not reserve GPU memory on {device} (requested {mib} MiB); continuing unreserved", flush=True)
    return 0


def _label_ts_array(dataset):
    inner = getattr(dataset, "dataset", dataset)
    label_ts = getattr(inner, "label_ts", None)
    return None if label_ts is None else np.asarray(label_ts), inner


def seek_label_cursor(dataset, snapshots, after_ts=None):
    """Position TGB's node-label cursor for a split.

    TGB's label stream is a monotone cursor (labels fire when cur_t >= label_ts)
    that is reset once per epoch and carried across train -> val -> test.  Our
    splits may be evaluated separately and may leave gaps in the edge stream
    (capped training), so the cursor is placed at the first label whose
    timestamp is >= the split's first EDGE timestamp: labels that lie in a gap
    between splits never "pass" in the stream and are skipped, labels inside
    the split are all consumed by `drain_snapshot_labels`.  `after_ts` is kept
    for callers without edge timestamps (first label > after_ts)."""
    dataset.reset_label_time()
    label_ts, inner = _label_ts_array(dataset)
    if label_ts is None or not snapshots:
        return
    first = snapshots[0]
    ets = getattr(first, "edge_timestamps", None)
    if ets is not None and torch.is_tensor(ets) and ets.numel():
        idx = int(np.searchsorted(label_ts, int(ets.min().item()), side="left"))
    elif after_ts is not None:
        idx = int(np.searchsorted(label_ts, int(after_ts), side="right"))
    else:
        first_ts = int(first.timestamp.item()) if torch.is_tensor(first.timestamp) else int(first.timestamp)
        idx = int(np.searchsorted(label_ts, first_ts, side="left"))
    inner.label_ts_idx = idx


def _snapshot_time(snapshot):
    ts = snapshot.timestamp
    return int(ts.item()) if torch.is_tensor(ts) else int(ts)


def drain_snapshot_labels(dataset, snapshot):
    """All labels with label_ts <= snapshot time, in order, each as
    (label_ts, label_srcs, labels) -- exactly the labels TGB's loop would have
    fired while streaming the edges of this snapshot (TGB checks the cursor on
    every batch; a coarse snapshot may cover several label timestamps, e.g.
    seven daily labels per weekly snapshot, three yearly labels per 3-year
    window).  The cursor advances past them."""
    cur_t = _snapshot_time(snapshot)
    out = []
    while True:
        label_tuple = dataset.get_node_label(cur_t)
        if label_tuple is None:
            break
        ts, label_srcs, labels = label_tuple
        ts0 = ts[0] if hasattr(ts, "__len__") else ts
        ts0 = int(ts0.item()) if torch.is_tensor(ts0) else int(ts0)
        label_srcs = torch.as_tensor(label_srcs).long()
        labels = torch.as_tensor(labels).float()
        out.append((ts0, label_srcs, labels))
    return out


def last_snapshot_timestamp(snapshots):
    if not snapshots:
        return None
    ts = snapshots[-1].timestamp
    return int(ts.item()) if torch.is_tensor(ts) else int(ts)


def node_label_batches(dataset, snapshots, seek=True):
    """(snapshot, label_srcs, labels) for EVERY label consumed while streaming
    `snapshots` (see `drain_snapshot_labels`); `seek=False` continues from the
    cursor's current position (sequential chunks of one epoch)."""
    if seek:
        seek_label_cursor(dataset, snapshots)
    batches = []
    for snapshot in snapshots:
        for _, label_srcs, labels in drain_snapshot_labels(dataset, snapshot):
            batches.append((snapshot, label_srcs, labels))
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


def node_property_loss(outputs: Sequence[torch.Tensor], dataset, snapshots: Sequence[TemporalSnapshot], seek=True):
    from torch.nn import functional as F

    losses = []
    index_of = {id(snapshot): k for k, snapshot in enumerate(snapshots)}
    for batch_snapshot, label_srcs, labels in node_label_batches(dataset, snapshots, seek=seek):
        logits = outputs[index_of[id(batch_snapshot)]]
        pred = logits.index_select(0, label_srcs.to(logits.device))
        target = labels.to(logits.device)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        losses.append(-(target * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean())

    if not losses:
        return torch.tensor(0.0, device=outputs[0].device if outputs else "cpu")
    return torch.stack(losses).mean()

def _uses_event_scoring(model) -> bool:
    return bool(getattr(model, "supports_event_scoring", False))


def _sample_uniform_negative_destinations(
    positive_dst: torch.Tensor,
    *,
    num_nodes: int,
    negatives_per_positive: int,
    destination_range: Optional[Tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    if positive_dst.numel() == 0 or negatives_per_positive <= 0:
        return None
    # Bipartite datasets (e.g. tgbl-wiki) only ever link into a contiguous
    # destination id block; sampling outside it trains against impossible
    # candidates, so restrict negatives to [lo, hi] when the range is known.
    lo, hi = (0, num_nodes - 1) if destination_range is None else destination_range
    vocab = hi - lo + 1
    if vocab <= 1:
        return None

    positive_dst = positive_dst.long()
    gen = NEGATIVE_RNG.get("generator")
    if gen is not None and str(gen.device) != str(positive_dst.device):
        gen = None   # generator devices must match; fall back to the global RNG
    sampled = torch.randint(
        low=0,
        high=max(vocab - 1, 1),
        size=(positive_dst.numel(), negatives_per_positive),
        device=positive_dst.device,
        generator=gen,
    )
    sampled = sampled + (sampled >= (positive_dst.view(-1, 1) - lo)).long()
    return (sampled + lo).long()


def _event_pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: Optional[torch.Tensor],
):
    from torch.nn import functional as F

    if negative_scores is None or negative_scores.numel() == 0:
        return F.softplus(-positive_scores).mean()
    return F.softplus(negative_scores - positive_scores.unsqueeze(-1)).mean()


def _query_time_kwargs(model, snapshot, device):
    """Per-event timestamps for event-exact scoring, when the model supports it."""
    if getattr(model, "supports_query_time", False) and getattr(snapshot, "edge_timestamps", None) is not None:
        return {"query_time": snapshot.edge_timestamps.to(device)}
    return {}


def event_link_prediction_loss(model, outputs, snapshots: Sequence[TemporalSnapshot]):
    losses = []
    for output, snapshot in zip(outputs, snapshots):
        if snapshot.src.numel() == 0:
            continue

        device = output["spatial"].device
        pos_src = snapshot.src.to(device)
        pos_dst = snapshot.dst.to(device)
        edge_type = snapshot.edge_types.to(device) if snapshot.edge_types is not None else None

        qt = _query_time_kwargs(model, snapshot, device)
        positive_scores = model.score_event_pairs(output, pos_src, pos_dst, edge_type=edge_type, **qt)
        negative_dst = _sample_uniform_negative_destinations(
            pos_dst,
            num_nodes=model.graph_size,
            negatives_per_positive=getattr(model, "train_negatives_per_pos", 32),
            destination_range=getattr(model, "event_destination_range", None),
        )
        negative_scores = None
        if negative_dst is not None:
            negative_scores = model.score_event_candidates(output, pos_src, negative_dst, edge_type=edge_type, **qt)
        loss_type = getattr(model, "event_loss_type", "softplus")
        if loss_type == "ce" and negative_scores is not None and negative_scores.numel() > 0:
            # Sampled softmax: the positive competes against its sampled
            # negatives directly, aligning the training signal with ranking.
            from torch.nn import functional as F
            logits = torch.cat([positive_scores.unsqueeze(-1), negative_scores], dim=-1)
            target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits, target))
        else:
            losses.append(_event_pairwise_ranking_loss(positive_scores, negative_scores))

    if not losses:
        output = outputs[0] if outputs else None
        device = output["spatial"].device if output is not None else "cpu"
        return torch.tensor(0.0, device=device)
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


def sequence_loss_for_task(spec: DatasetSpec, outputs, dataset, snapshots, model=None, seek_labels=True):
    if spec.task_family == "nodeprop":
        return node_property_loss(outputs, dataset, snapshots, seek=seek_labels)
    if _uses_event_scoring(model):
        return event_link_prediction_loss(model, outputs, snapshots)
    return link_prediction_loss(outputs, snapshots)


def _next_snapshot_labels(dataset, snapshot):
    """Compat shim: first label of the snapshot (prefer `drain_snapshot_labels`)."""
    drained = drain_snapshot_labels(dataset, snapshot)
    return None if not drained else (drained[0][1], drained[0][2])


def evaluate_node_property_streaming(dataset_name, dataset, snapshots, model, initial_state=None,
                                     compute_metric=True, compute_loss=True, evaluator=None):
    """TGB node-property evaluation for a split: stream the snapshots, and for
    each snapshot score EVERY label whose timestamp it has passed
    (`drain_snapshot_labels`) with that snapshot's output; NDCG@10 per label
    timestamp, unweighted mean over label timestamps (TGB's
    `total_score / num_label_ts`).  Returns (metric, mean_loss, state)."""
    from torch.nn import functional as F

    if compute_metric and evaluator is None:
        from tgb.nodeproppred.evaluate import Evaluator

        evaluator = Evaluator(name=dataset_name)
    metric_name = getattr(dataset, "eval_metric", "ndcg")
    seek_label_cursor(dataset, snapshots)
    per_ts, losses, label_times = [], [], []
    nonfinite = 0
    state = initial_state
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            outputs, state = model.forward_sequence([snapshot], initial_state=state)
            logits = outputs[0]
            for label_ts, label_srcs, labels in drain_snapshot_labels(dataset, snapshot):
                pred = logits.index_select(0, label_srcs.to(logits.device))
                target = labels.to(logits.device)
                normalised_target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                if compute_loss:
                    losses.append((-(normalised_target * F.log_softmax(pred, dim=-1)).sum(dim=-1).mean()).detach().cpu())
                if compute_metric:
                    if not bool(torch.isfinite(pred).all()):
                        # a non-finite prediction is an invalid answer: it scores 0 for this
                        # label timestamp (sklearn would raise; the old code silently returned
                        # NaN for the whole split) and is counted for the caller
                        nonfinite += 1
                        per_ts.append(0.0)
                    else:
                        score = evaluator.eval({"y_pred": pred.detach().cpu(), "y_true": labels.detach().cpu(), "eval_metric": [metric_name]})
                        per_ts.append(float(score[metric_name] if isinstance(score, dict) else score))
                    label_times.append(label_ts)
    if nonfinite:
        print(f"  [eval-audit] {nonfinite}/{len(per_ts)} label timestamps had non-finite predictions (scored 0)", flush=True)
    evaluate_node_property_streaming.last_label_timestamps = label_times
    evaluate_node_property_streaming.last_nonfinite_labels = nonfinite
    metric = float(np.mean(per_ts)) if per_ts else float("nan")
    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    return metric, mean_loss, state

def detach_temporal_state(state):
    if state is None:
        return None
    return type(state)(
        memory=state.memory.detach() if state.memory is not None else None,
        spatial=state.spatial.detach() if state.spatial is not None else None,
    )


def advance_context(model, snapshots, initial_state=None):
    state = initial_state
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            _, state = model.forward_sequence([snapshot], initial_state=state)
    return state


def _pad_negative_samples(neg_samples):
    """Exact per-query negatives: pad ragged official lists to the max
    length and return a validity mask; padded slots must be scored -inf so
    every query is ranked against ITS OWN full negative list (trimming to the
    min length discards valid negatives and slightly inflates MRR)."""
    if neg_samples is None:
        return None, None
    arrays = [np.asarray(sample, dtype=np.int64) for sample in neg_samples]
    if not arrays:
        return None, None
    lengths = [len(arr) for arr in arrays]
    if min(lengths) == 0:
        return None, None
    max_len = max(lengths)
    padded = np.zeros((len(arrays), max_len), dtype=np.int64)
    mask = np.zeros((len(arrays), max_len), dtype=bool)
    for i, arr in enumerate(arrays):
        padded[i, : len(arr)] = arr
        mask[i, : len(arr)] = True
    return padded, mask


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
    label_cursor_after_ts=None,
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
    hits_sum = [0.0]
    state = initial_state

    if spec.task_family == "nodeprop":
        return evaluate_node_property_streaming(spec.loader_name, dataset, snapshots, model, initial_state=initial_state,
                                                compute_metric=compute_metric, compute_loss=compute_loss, evaluator=evaluator)

    if spec.task_family != "nodeprop":
        if split_mode == "val" and hasattr(dataset, "load_val_ns"):
            dataset.load_val_ns()
        elif split_mode == "test" and hasattr(dataset, "load_test_ns"):
            dataset.load_test_ns()

    use_event_scoring = spec.task_family != "nodeprop" and _uses_event_scoring(model)
    model.eval()
    # Leak-free protocol (audit rows 21/32): score snapshot k with the output of
    # the PREVIOUS forward (history < k; the recurrency stores hold only earlier
    # events), and only then forward snapshot k to update the state.
    score_from_previous = os.environ.get("TSD_SCORE_FROM_PREVIOUS_STATE") == "1"
    prev_output = None
    with torch.no_grad():
        for snapshot_index, snapshot in enumerate(snapshots):
            deferred = score_from_previous and use_event_scoring
            if deferred:
                if prev_output is None:
                    sp = getattr(initial_state, "spatial", None) if initial_state is not None else None
                    if sp is None:
                        # no history at all: this snapshot can only seed the state
                        outputs, state = model.forward_sequence([snapshot], initial_state=state)
                        prev_output = outputs[0]
                        _SKIPPED_QUERIES[0] += int(snapshot.src.numel())
                        continue
                    x_dev = snapshot.x.to(sp.device) if getattr(snapshot, "x", None) is not None else None
                    model_output = {"spatial": sp, "x": x_dev, "node_signal": None}
                else:
                    model_output = dict(prev_output, node_signal=None)
            else:
                outputs, state = model.forward_sequence([snapshot], initial_state=state)
                model_output = outputs[0]

            if use_event_scoring:
                device = model_output["spatial"].device
                pos_src = snapshot.src.to(device)
                pos_dst = snapshot.dst.to(device)
                edge_type = snapshot.edge_types.to(device) if snapshot.edge_types is not None else None
                if pos_src.numel() == 0:
                    if deferred:
                        outputs, state = model.forward_sequence([snapshot], initial_state=state)
                        prev_output = outputs[0]
                    continue

                qt = _query_time_kwargs(model, snapshot, device)
                positive_scores = model.score_event_pairs(
                    model_output,
                    pos_src,
                    pos_dst,
                    edge_type=edge_type,
                    **qt,
                )

                negative_scores = None
                if split_mode in ("val", "test"):
                    pos_ts = snapshot.edge_timestamps
                    if pos_ts is None:
                        pos_ts = snapshot.timestamp.expand(snapshot.src.numel())
                    neg_samples = dataset.negative_sampler.query_batch(
                        snapshot.src.cpu(),
                        snapshot.dst.cpu(),
                        pos_ts.cpu(),
                        edge_type=snapshot.edge_types.cpu() if snapshot.edge_types is not None else None,
                        split_mode=split_mode,
                    )
                    padded, neg_mask = _pad_negative_samples(neg_samples)
                    if padded is None:
                        _SKIPPED_QUERIES[0] += int(snapshot.src.numel())
                    if padded is not None:
                        neg_dst = torch.as_tensor(padded, dtype=torch.long, device=device)
                        negative_scores = model.score_event_candidates(
                            model_output,
                            pos_src,
                            neg_dst,
                            edge_type=edge_type,
                            **qt,
                        )
                        pad_mask = torch.as_tensor(neg_mask, device=device)
                        negative_scores = negative_scores.masked_fill(~pad_mask, float("-inf"))

                if compute_loss:
                    loss_negatives = negative_scores
                    if loss_negatives is None:
                        sampled_neg_dst = _sample_uniform_negative_destinations(
                            pos_dst,
                            num_nodes=model.graph_size,
                            negatives_per_positive=getattr(model, "train_negatives_per_pos", 32),
                        )
                        if sampled_neg_dst is not None:
                            loss_negatives = model.score_event_candidates(
                                model_output,
                                pos_src,
                                sampled_neg_dst,
                                edge_type=edge_type,
                                **qt,
                            )
                    losses.append(_event_pairwise_ranking_loss(positive_scores, loss_negatives).detach().cpu())

                if compute_metric and split_mode in ("val", "test") and negative_scores is not None:
                    raw_pos, raw_neg = positive_scores, negative_scores
                    positive_scores, negative_scores = _guard_nonfinite_scores(positive_scores, negative_scores)
                    audit = QUERY_AUDIT.get("audit")
                    if audit is not None:
                        audit.record_batch(split_mode, snapshot_index, snapshot, raw_pos, raw_neg, neg_mask,
                                           positive_scores, negative_scores)
                    score = evaluator.eval({
                        "y_pred_pos": positive_scores,
                        "y_pred_neg": negative_scores,
                        "eval_metric": [dataset.eval_metric],
                    })
                    # TGB's link evaluator returns {'hits@10', 'mrr'} (hits first);
                    # select the dataset's metric by NAME - the previous
                    # positional pick silently reported Hits@10 as MRR.
                    metric_val = score[dataset.eval_metric] if isinstance(score, dict) else score
                    metric_sum += float(np.mean(metric_val)) * int(pos_src.numel())
                    metric_examples += int(pos_src.numel())
                    if isinstance(score, dict) and "hits@10" in score:
                        hits_sum[0] += float(np.mean(score["hits@10"])) * int(pos_src.numel())
                if deferred:
                    # update: ingest snapshot k after its events were scored
                    outputs, state = model.forward_sequence([snapshot], initial_state=state)
                    prev_output = outputs[0]
                audit = QUERY_AUDIT.get("audit")
                if audit is not None and split_mode in ("val", "test"):
                    audit.record_state(split_mode, snapshot_index, model, state)
                continue

            logits = model_output
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
                _SKIPPED_QUERIES[0] += int(snapshot.src.numel())
                continue

            neg_dst = torch.as_tensor(neg_samples, dtype=torch.long, device=logits.device)
            y_pred_pos = logits[pos_src, pos_dst]
            y_pred_neg = logits[pos_src.view(-1, 1).expand_as(neg_dst), neg_dst]
            y_pred_pos, y_pred_neg = _guard_nonfinite_scores(y_pred_pos, y_pred_neg)
            score = evaluator.eval({
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
                "eval_metric": [dataset.eval_metric],
            })
            metric_val = score[dataset.eval_metric] if isinstance(score, dict) else score
            metric_sum += float(metric_val) * int(pos_src.numel())
            metric_examples += int(pos_src.numel())

    metric = float("nan")
    if False:
        pass
    elif spec.task_family != "nodeprop" and compute_metric and metric_examples > 0:
        metric = metric_sum / metric_examples
        evaluate_model_streaming.last_hits10 = hits_sum[0] / metric_examples if metric_examples else float("nan")

    mean_loss = float(torch.stack(losses).mean().item()) if losses else float("nan")
    # Audit accounting: TGB scores every query and never ranks a NaN first.
    evaluate_model_streaming.last_nonfinite_positives = _NONFINITE_POSITIVES[0]
    evaluate_model_streaming.last_nonfinite_negatives = _NONFINITE_NEGATIVES[0]
    evaluate_model_streaming.last_skipped_queries = _SKIPPED_QUERIES[0]
    evaluate_model_streaming.last_metric_examples = metric_examples
    if _NONFINITE_POSITIVES[0] or _NONFINITE_NEGATIVES[0] or _SKIPPED_QUERIES[0]:
        print(f"[eval-audit] split={split_mode} non_finite_positive_scores={_NONFINITE_POSITIVES[0]} "
              f"non_finite_negative_scores={_NONFINITE_NEGATIVES[0]} "
              f"skipped_queries={_SKIPPED_QUERIES[0]} scored_queries={metric_examples}", flush=True)
    _NONFINITE_POSITIVES[0] = 0
    _NONFINITE_NEGATIVES[0] = 0
    _SKIPPED_QUERIES[0] = 0
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
    use_event_scoring = _uses_event_scoring(model)
    if use_event_scoring and os.environ.get("TSD_SCORE_FROM_PREVIOUS_STATE") == "1" and spec.task_family != "nodeprop":
        # Leak-free predict-then-update training (audit rows 21/32): iteration k
        # forwards snapshot k-1 (graph attached, so the backbone is trained) and
        # scores snapshot k's events with that output; the recurrency stores then
        # hold only events < k.  The state is detached only afterwards (bptt=1).
        state = None
        losses = []
        iterator = range(1, len(train_snapshots))
        if show_progress and tqdm_factory is not None:
            iterator = tqdm_factory(iterator, total=max(len(train_snapshots) - 1, 0),
                                    desc=epoch_label or "Epoch snapshots", leave=True)
        for k in iterator:
            optimizer.zero_grad()
            outputs, new_state = model.forward_sequence([train_snapshots[k - 1]], initial_state=state)
            out = outputs[0]
            if isinstance(out, dict):
                out = dict(out, node_signal=None)
            loss = sequence_loss_for_task(spec, [out], dataset, [train_snapshots[k]], model=model)
            if loss.requires_grad:
                if not bool(torch.isfinite(loss)):
                    # a non-finite loss is skipped explicitly and COUNTED (it used to
                    # reach backward and be dropped silently by the gradient guard)
                    TRAIN_COUNTERS["nonfinite_loss"] += 1
                    TRAIN_COUNTERS["skipped_steps"] += 1
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    stepped = optimizer.step()
                    TRAIN_COUNTERS["steps"] += 1
                    if stepped is False:
                        TRAIN_COUNTERS["skipped_steps"] += 1
                    losses.append(float(loss.detach().cpu()))
            state = detach_temporal_state(new_state)
        return float(sum(losses) / len(losses)) if losses else 0.0
    if bptt_steps is None or bptt_steps <= 0:
        if use_event_scoring:
            optimizer.zero_grad()
            state = None
            losses = []
            for snapshot in train_snapshots:
                outputs, state = model.forward_sequence([snapshot], initial_state=state)
                losses.append(sequence_loss_for_task(spec, outputs, dataset, [snapshot], model=model))
            if not losses:
                return 0.0
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            return float(loss.detach().cpu())
        optimizer.zero_grad()
        outputs, _ = model.forward_sequence(train_snapshots)
        loss = sequence_loss_for_task(spec, outputs, dataset, train_snapshots, model=model)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    chunk_losses = []
    state = None
    if spec.task_family == "nodeprop":
        seek_label_cursor(dataset, train_snapshots)   # once per epoch; chunks continue the cursor
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
        if use_event_scoring:
            chunk_state = state
            loss_terms = []
            for snapshot in chunk:
                outputs, chunk_state = model.forward_sequence([snapshot], initial_state=chunk_state)
                loss_terms.append(sequence_loss_for_task(spec, outputs, dataset, [snapshot], model=model))
            state = chunk_state
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
        else:
            outputs, state = model.forward_sequence(chunk, initial_state=state)
            loss = sequence_loss_for_task(spec, outputs, dataset, chunk, model=model, seek_labels=False)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            chunk_losses.append(float(loss.detach().cpu()))

        state = detach_temporal_state(state)

    if not chunk_losses:
        return 0.0
    return float(sum(chunk_losses) / len(chunk_losses))
