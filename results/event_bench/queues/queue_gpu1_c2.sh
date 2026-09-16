#!/bin/bash
# Waits for the running genre lane, then: genre faithful seed 47, genre original finals seeds 46/47
cd /home/zhuowez1/project/neural-sheaf-diffusion
while pgrep -f "genre_o_lr005" > /dev/null; do sleep 60; done
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset TGB_ROOT
O=results/event_bench
$PY -m exp.run_faithful_trade --dataset tgbn-genre --seeds 47 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "memory_readout": true, "epochs": 40, "early_stopping_patience": 12, "min_epochs_before_stopping": 16, "lr": 0.005}' \
  --out $O/genre_f_lr005_s47.csv > $O/genre_f_lr005_s47.log 2>&1
$PY -m exp.run_baseline_trade --dataset tgbn-genre --seeds 46 47 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "epochs": 40, "early_stopping_patience": 16, "min_epochs_before_stopping": 16}' \
  --out $O/genre_o_lr0125_s4647.csv > $O/genre_o_lr0125_s4647.log 2>&1
echo "QUEUE_GPU1_C2_DONE"
