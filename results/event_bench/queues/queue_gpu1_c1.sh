#!/bin/bash
# Phase C-1: smallpedia lr 1e-2 probes (both models) + genre original @ lr 0.005
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset TGB_ROOT
O=results/event_bench
sp="--dataset tkgl-smallpedia --epochs 25 --patience 8 --min-epochs 10 --track-val-edges 50000 --seed 43"
$PY -m exp.run_event_benchmark $sp --model original --lr 1e-2 --out $O/sp_o_lr1e2 > $O/sp_o_lr1e2.log 2>&1
$PY -m exp.run_event_benchmark $sp --model faithful --lr 1e-2 --out $O/sp_f_lr1e2 > $O/sp_f_lr1e2.log 2>&1
$PY -m exp.run_baseline_trade --dataset tgbn-genre --seeds 43 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "epochs": 40, "early_stopping_patience": 16, "min_epochs_before_stopping": 16, "lr": 0.005}' \
  --out $O/genre_o_lr005.csv > $O/genre_o_lr005.log 2>&1
echo "QUEUE_GPU1_C1_DONE"
