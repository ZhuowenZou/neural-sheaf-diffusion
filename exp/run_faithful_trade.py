"""CLI runner: benchmark FaithfulTemporalSheafDiffusion on tgbn-trade.

Usage:
    TGB_ROOT=$PWD/datasets python -m exp.run_faithful_trade \
        --seeds 43 44 45 46 47 --epochs 200 --out results/faithful_trade.csv
"""

import argparse
import json
import os
import time

import pandas as pd
import torch

from exp.faithful_temporal_studies import (
    make_faithful_trade_config,
    prepare_temporal_experiment_context,
    train_single_faithful,
)
from exp.temporal_benchmark_utils import reserve_gpu_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[43])
    parser.add_argument("--dataset", type=str, default="tgbn-trade",
                        help="Any TGB nodeprop (tgbn-*) dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", type=str, default=None,
                        help="JSON dict of config overrides")
    parser.add_argument("--train-cap", type=int, default=2048,
                        help="Max train edges (-1 for uncapped)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Per-seed checkpoint/resume directory (default: <out>.ckpt when --out is given)")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint every N epochs")
    parser.add_argument("--reserve-gpu-mb", type=int, default=None,
                        help="Pre-reserve GPU memory at start-up (default: env TSD_RESERVE_GPU_MB from wait_launch.sh)")
    args = parser.parse_args()

    config = make_faithful_trade_config()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.override:
        config.update(json.loads(args.override))

    device = torch.device(args.device) if args.device else None
    context = prepare_temporal_experiment_context(
        args.dataset,
        split_caps={"train": (None if args.train_cap < 0 else args.train_cap), "val": None, "test": None},
        device=device,
        preload_time_windows=[config["time_window"]],
    )
    bundle = context.get_snapshot_bundle(config["time_window"])
    print(
        f"snapshots: train={len(bundle['train_snapshots'])} "
        f"val={len(bundle['val_snapshots'])} test={len(bundle['test_snapshots'])} "
        f"nodes={context.num_nodes} output_dim={context.output_dim}"
    )

    reserve_gpu_memory(context.device, mib=args.reserve_gpu_mb)

    ckpt_dir = args.checkpoint_dir or (f"{args.out}.ckpt" if args.out else None)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    rows = []
    for seed in args.seeds:
        done_file = os.path.join(ckpt_dir, f"seed{seed}.done.json") if ckpt_dir else None
        if done_file and os.path.exists(done_file):
            with open(done_file) as fh:
                row = json.load(fh)
            rows.append(row)
            print(f"seed {seed}: already finished ({done_file}) val={row['val_ndcg']:.4f} "
                  f"test={row['test_ndcg']:.4f}", flush=True)
            continue
        seed_config = dict(config)
        if ckpt_dir:
            seed_config["checkpoint_path"] = os.path.join(ckpt_dir, f"seed{seed}.pt")
            seed_config["checkpoint_every"] = args.checkpoint_every
        start = time.perf_counter()
        result = train_single_faithful(
            dataset_name=context.dataset_name,
            dataset=context.dataset,
            snapshot_bundle=bundle,
            node_features=context.node_features,
            num_nodes=context.num_nodes,
            output_dim=context.output_dim,
            device=context.device,
            config=seed_config,
            seed=seed,
        )
        elapsed = time.perf_counter() - start
        row = {
            "seed": seed,
            "val_ndcg": result["best_val_metric"],
            "test_ndcg": result["best_test_metric"],
            "best_epoch": result["best_epoch"],
            "n_params": result["n_params"],
            "minutes": elapsed / 60.0,
        }
        rows.append(row)
        print(f"seed {seed}: val={row['val_ndcg']:.4f} test={row['test_ndcg']:.4f} "
              f"best_epoch={row['best_epoch']} ({row['minutes']:.1f} min)", flush=True)
        if done_file:
            with open(done_file, "w") as fh:
                json.dump(row, fh)
            ckpt = seed_config.get("checkpoint_path")
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)

    df = pd.DataFrame(rows)
    print("\n== summary ==")
    print(df.to_string(index=False))
    print(f"test NDCG mean±std: {df.test_ndcg.mean():.4f} ± {df.test_ndcg.std():.4f}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
