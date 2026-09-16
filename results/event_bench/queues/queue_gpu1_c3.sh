#!/bin/bash
# Waits for the genre faithful seed-47 lane, then genre ORIGINAL finals at lr 0.005 (the sweep winner)
cd /home/zhuowez1/project/neural-sheaf-diffusion
while pgrep -f "run_faithful_trade --dataset tgbn-genre --seeds 47" > /dev/null; do sleep 60; done
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset TGB_ROOT
O=results/event_bench
$PY -m exp.run_baseline_trade --dataset tgbn-genre --seeds 46 47 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "epochs": 40, "early_stopping_patience": 16, "min_epochs_before_stopping": 16, "lr": 0.005}' \
  --out $O/genre_o_lr005_s4647.csv > $O/genre_o_lr005_s4647.log 2>&1
echo "QUEUE_GPU1_C3_DONE"
