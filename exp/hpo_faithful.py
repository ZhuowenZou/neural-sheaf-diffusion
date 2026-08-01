"""Random-search HPO for FaithfulTemporalSheafDiffusion on tgbn-trade.

Stage 1: N sampled configs, screening seed(s), selection on val NDCG.
Stage 2 (rerun with --configs-from): top-K configs on extra seeds.
Final evaluation of the winner runs through exp.run_faithful_trade.

Trials are appended to the output CSV as they finish, so shards can be
inspected mid-flight and interrupted runs lose at most one trial.

Usage (3-way GPU shard):
    CUDA_VISIBLE_DEVICES=0 python -m exp.hpo_faithful --shard 0/3 --out hpo.csv
    CUDA_VISIBLE_DEVICES=1 python -m exp.hpo_faithful --shard 1/3 --out hpo.csv
    CUDA_VISIBLE_DEVICES=5 python -m exp.hpo_faithful --shard 2/3 --out hpo.csv
"""

import argparse
import json
import math
import os
import random
import time

import pandas as pd

from exp.faithful_temporal_studies import (
    make_faithful_trade_config,
    prepare_temporal_experiment_context,
    train_single_faithful,
)

SEARCH_SPACE_SEED = 7


def sample_trial(rng: random.Random) -> dict:
    return {
        "lr": 10 ** rng.uniform(math.log10(1e-3), math.log10(3e-2)),
        "weight_decay": 10 ** rng.uniform(-8, -4),
        "temporal_d_model": rng.choice([64, 128, 256]),
        "hidden_channels": rng.choice([64, 128]),
        "d": rng.choice([4, 5, 6]),
        "layers": rng.choice([2, 3, 4, 5]),
        "feedback_dim": rng.choice([16, 32, 64]),
        "diffusion_update": rng.choice(["euler", "direct"]),
        "dropout": rng.choice([0.0, 0.1]),
        # Deterministic dt_init per trial so trial quality is not confounded
        # by the model-seed dt draw.
        "dt_init": rng.choice([0.003, 0.01, 0.03]),
    }


def trial_config(trial: dict, epochs: int, patience: int, min_epochs: int) -> dict:
    config = make_faithful_trade_config()
    config.update(trial)
    config["dt_min"] = trial["dt_init"]
    config["dt_max"] = trial["dt_init"]
    config["epochs"] = epochs
    config["early_stopping_patience"] = patience
    config["min_epochs_before_stopping"] = min_epochs
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43])
    parser.add_argument("--shard", type=str, default="0/1", help="i/n trial sharding")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--configs-from", type=str, default=None,
                        help="JSON list of trial dicts to run instead of sampling")
    args = parser.parse_args()

    shard_i, shard_n = (int(v) for v in args.shard.split("/"))

    if args.configs_from:
        with open(args.configs_from) as fh:
            trials = [(t.pop("trial_id"), t) for t in json.load(fh)]
    else:
        rng = random.Random(SEARCH_SPACE_SEED)
        trials = [(i, sample_trial(rng)) for i in range(args.n_trials)]
    trials = [(tid, t) for tid, t in trials if tid % shard_n == shard_i]

    context = prepare_temporal_experiment_context(
        "tgbn-trade",
        split_caps={"train": 2048, "val": None, "test": None},
        preload_time_windows=[make_faithful_trade_config()["time_window"]],
    )
    bundle = context.get_snapshot_bundle(make_faithful_trade_config()["time_window"])

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev.trial_id, prev.seed))

    for trial_id, trial in trials:
        config = trial_config(trial, args.epochs, args.patience, args.min_epochs)
        for seed in args.seeds:
            if (trial_id, seed) in done:
                continue
            start = time.perf_counter()
            try:
                result = train_single_faithful(
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
                row = {
                    "trial_id": trial_id,
                    "seed": seed,
                    "val_ndcg": result["best_val_metric"],
                    "test_ndcg": result["best_test_metric"],
                    "best_epoch": result["best_epoch"],
                    "n_params": result["n_params"],
                    "minutes": (time.perf_counter() - start) / 60.0,
                    "error": "",
                    **{f"hp_{k}": v for k, v in trial.items()},
                }
            except Exception as exc:  # noqa: BLE001 - a failed trial must not kill the shard
                row = {
                    "trial_id": trial_id,
                    "seed": seed,
                    "val_ndcg": float("nan"),
                    "test_ndcg": float("nan"),
                    "best_epoch": -1,
                    "n_params": -1,
                    "minutes": (time.perf_counter() - start) / 60.0,
                    "error": repr(exc)[:200],
                    **{f"hp_{k}": v for k, v in trial.items()},
                }
            header = not os.path.exists(args.out)
            pd.DataFrame([row]).to_csv(args.out, mode="a", header=header, index=False)
            print(f"trial {trial_id} seed {seed}: val={row['val_ndcg']:.4f} "
                  f"test={row['test_ndcg']:.4f} ({row['minutes']:.1f} min) {row['error']}")


if __name__ == "__main__":
    main()
