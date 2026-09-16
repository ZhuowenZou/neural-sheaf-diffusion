#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
exec results/event_bench/queues/wait_launch.sh 6000 results/leakfree_nodeprop/genre_faithful_weekly_s43.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -u -m exp.run_faithful_trade --dataset tgbn-genre --seeds 43 --train-cap -1 --override '{"time_window": 604800, "eval_every": 2, "memory_readout": true, "epochs": 40, "early_stopping_patience": 12, "min_epochs_before_stopping": 16, "lr": 0.005}' --out results/leakfree_nodeprop/genre_faithful_weekly_s43.csv
