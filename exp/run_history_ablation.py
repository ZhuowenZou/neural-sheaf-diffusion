# Paired "no-history-to-sheaf" ablation on tgbn-trade.
#
# Trains the full history-conditioned model (sheaf_conditioning=history) and the
# current-only control (sheaf_conditioning=current_only) with identical seeds,
# hyperparameters (make_trade_baseline_config), snapshot bundles, and the
# official TGB evaluator, then writes machine-readable per-run results.
#
# Run:
#   python exp/run_history_ablation.py --seeds 43 44 45 46 47 \
#       --out results/history_ablation_tgbn-trade
#   python exp/run_history_ablation.py --smoke   # quick 4-epoch sanity pass

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp.temporal_mamba_studies import (
    _make_model_from_config,
    _node_property_loss,
    make_trade_baseline_config,
    prepare_temporal_experiment_context,
    train_single_config,
)

DATASET_NAME = "tgbn-trade"
SPLIT_CAPS = {"train": 2048, "val": None, "test": None}
VARIANTS = ("history", "current_only")


def _repo_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).strip()
    except Exception:
        return "unknown"


def _variant_config(base_config, variant):
    config = dict(base_config)
    config["sheaf_conditioning"] = variant
    return config


def _parameter_count(context, config):
    bundle = context.get_snapshot_bundle(config.get("time_window"))
    model = _make_model_from_config(
        bundle["train_snapshots"][0].edge_index,
        context.node_features,
        context.num_nodes,
        context.output_dim,
        context.device,
        config,
    )
    count = sum(p.numel() for p in model.parameters())
    del model
    return count


def _preflight_grad_check(context, config, variant):
    """One forward/backward pass; every trainable tensor must receive a gradient."""
    bundle = context.get_snapshot_bundle(config.get("time_window"))
    torch.manual_seed(0)
    model = _make_model_from_config(
        bundle["train_snapshots"][0].edge_index,
        context.node_features,
        context.num_nodes,
        context.output_dim,
        context.device,
        config,
    )
    model.train()
    chunk = bundle["train_snapshots"][:4]
    outputs, _ = model.forward_sequence(chunk)
    loss = _node_property_loss(outputs, context.dataset, chunk)
    loss.backward()
    # The unused CPU-fallback branch of MambaBlock gets no grad on CUDA; only
    # require that every gradient that exists is finite and that the sheaf and
    # temporal learners both receive signal.
    nonfinite = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    sheaf_grad_norm = sum(
        float(param.grad.norm())
        for name, param in model.named_parameters()
        if "sheaf_learner" in name and param.grad is not None
    )
    temporal_grad_norm = sum(
        float(param.grad.norm())
        for name, param in model.named_parameters()
        if "nodewise_learner" in name and param.grad is not None
    )
    del model
    if nonfinite:
        raise RuntimeError(f"[{variant}] parameters with non-finite gradients: {nonfinite}")
    if sheaf_grad_norm == 0.0:
        raise RuntimeError(f"[{variant}] sheaf learner received zero gradient.")
    if temporal_grad_norm == 0.0:
        raise RuntimeError(f"[{variant}] temporal learner received zero gradient.")
    print(
        f"[preflight {variant}] loss={float(loss):.4f} sheaf_grad_norm={sheaf_grad_norm:.3e} "
        f"temporal_grad_norm={temporal_grad_norm:.3e} all present grads finite"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46, 47])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs (smoke testing only).")
    parser.add_argument("--smoke", action="store_true", help="4-epoch, single-seed smoke run.")
    parser.add_argument("--out", default="results/history_ablation_tgbn-trade")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.epochs = args.epochs or 4

    os.makedirs(args.out, exist_ok=True)
    sha = _repo_sha()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_config = make_trade_baseline_config()
    if args.epochs is not None:
        base_config["epochs"] = int(args.epochs)
        base_config["min_epochs_before_stopping"] = min(
            int(base_config["min_epochs_before_stopping"]), int(args.epochs)
        )

    context = prepare_temporal_experiment_context(
        DATASET_NAME,
        SPLIT_CAPS,
        device=device,
        seed=43,
        preload_time_windows=[base_config["time_window"]],
    )
    # Validation checks 6-7: official evaluator + official chronological splits,
    # shared bit-for-bit between variants via the cached snapshot bundle.
    assert context.split_source == "official_tgb_masks", context.split_source
    assert context.metric_name == "ndcg", context.metric_name

    param_counts = {}
    for variant in args.variants:
        config = _variant_config(base_config, variant)
        param_counts[variant] = _parameter_count(context, config)
        _preflight_grad_check(context, config, variant)
    print(f"parameter counts: {param_counts}")

    results_path = os.path.join(args.out, "results.csv")
    rows = []
    if os.path.exists(results_path):
        rows = pd.read_csv(results_path).to_dict("records")
    done = {(row["variant"], row["seed"]) for row in rows}

    for seed in args.seeds:
        for variant in args.variants:
            if (variant, seed) in done:
                print(f"skip {variant} seed={seed} (already recorded)")
                continue
            config = _variant_config(base_config, variant)
            bundle = context.get_snapshot_bundle(config.get("time_window"))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = train_single_config(
                dataset_name=DATASET_NAME,
                dataset=context.dataset,
                snapshot_bundle=bundle,
                node_features=context.node_features,
                num_nodes=context.num_nodes,
                output_dim=context.output_dim,
                device=device,
                config=config,
                seed=int(seed),
                keep_history=False,
            )
            runtime = time.perf_counter() - start
            peak_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
            )
            row = {
                "variant": variant,
                "seed": int(seed),
                "best_epoch": result["best_epoch"],
                "validation_ndcg": result["best_val_metric"],
                "test_ndcg": result["best_test_metric"],
                "runtime_seconds": round(runtime, 2),
                "parameter_count": param_counts[variant],
                "commit_hash": sha,
                "peak_gpu_mem_mb": round(peak_mem_mb, 1),
                "config_json": json.dumps(config, sort_keys=True),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(
                f"[done] {variant} seed={seed} best_epoch={row['best_epoch']} "
                f"val_ndcg={row['validation_ndcg']:.4f} test_ndcg={row['test_ndcg']:.4f} "
                f"runtime={runtime:.1f}s"
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("variant")[["validation_ndcg", "test_ndcg"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("\n=== summary ===")
    print(summary)
    summary.to_csv(os.path.join(args.out, "summary.csv"))

    pivot = df.pivot(index="seed", columns="variant", values="test_ndcg")
    if set(VARIANTS).issubset(pivot.columns):
        pivot["full_minus_control"] = pivot["history"] - pivot["current_only"]
        pivot.to_csv(os.path.join(args.out, "paired_differences.csv"))
        print("\n=== paired per-seed test NDCG ===")
        print(pivot.round(4))
        print(f"\nmean paired difference (history - current_only): {pivot['full_minus_control'].mean():.4f}")


if __name__ == "__main__":
    main()
