"""CLI runner: original MambaSheafDiffusion baseline on tgbn-trade (reference)."""

import argparse
import json
import time

import pandas as pd
import torch

from exp.temporal_mamba_studies import (
    make_trade_baseline_config,
    prepare_temporal_experiment_context,
    train_single_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--train-cap", type=int, default=2048,
                        help="Max train edges (-1 for uncapped)")
    args = parser.parse_args()

    config = make_trade_baseline_config()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.override:
        config.update(json.loads(args.override))

    context = prepare_temporal_experiment_context(
        "tgbn-trade",
        split_caps={"train": (None if args.train_cap < 0 else args.train_cap), "val": None, "test": None},
        preload_time_windows=[config["time_window"]],
    )
    bundle = context.get_snapshot_bundle(config["time_window"])

    rows = []
    for seed in args.seeds:
        start = time.perf_counter()
        result = train_single_config(
            dataset_name=context.dataset_name,
            dataset=context.dataset,
            snapshot_bundle=bundle,
            node_features=context.node_features,
            num_nodes=context.num_nodes,
            output_dim=context.output_dim,
            device=context.device,
            config=config,
            seed=seed,
        )
        elapsed = time.perf_counter() - start
        row = {
            "seed": seed,
            "val_ndcg": result["best_val_metric"],
            "test_ndcg": result["best_test_metric"],
            "best_epoch": result["best_epoch"],
            "minutes": elapsed / 60.0,
        }
        rows.append(row)
        print(f"seed {seed}: val={row['val_ndcg']:.4f} test={row['test_ndcg']:.4f} "
              f"best_epoch={row['best_epoch']} ({row['minutes']:.1f} min)")

    df = pd.DataFrame(rows)
    print("\n== summary ==")
    print(df.to_string(index=False))
    print(f"test NDCG mean±std: {df.test_ndcg.mean():.4f} ± {df.test_ndcg.std():.4f}")
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
