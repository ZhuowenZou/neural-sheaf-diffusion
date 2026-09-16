#!/bin/bash
# tgbn-genre tuning (weekly windows, uncapped) then wikidata original lr sweep
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset TGB_ROOT
O=results/event_bench
$PY -m exp.run_faithful_trade --dataset tgbn-genre --seeds 43 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "memory_readout": true, "epochs": 40, "early_stopping_patience": 12, "min_epochs_before_stopping": 16}' \
  --out $O/genre_f_lr0125.csv > $O/genre_f_lr0125.log 2>&1
$PY -m exp.run_baseline_trade --dataset tgbn-genre --seeds 43 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "epochs": 40, "early_stopping_patience": 16, "min_epochs_before_stopping": 16}' \
  --out $O/genre_o_lr0125.csv > $O/genre_o_lr0125.log 2>&1
$PY -m exp.run_faithful_trade --dataset tgbn-genre --seeds 43 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "memory_readout": true, "epochs": 40, "early_stopping_patience": 12, "min_epochs_before_stopping": 16, "lr": 0.005}' \
  --out $O/genre_f_lr005.csv > $O/genre_f_lr005.log 2>&1
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
wd="--dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 100000 --temporal-d-model 32 --seed 43"
$PY -m exp.run_event_benchmark $wd --model original --lr 3e-4 --out $O/wd_o_lr3e4 > $O/wd_o_lr3e4.log 2>&1
$PY -m exp.run_event_benchmark $wd --model original --lr 3e-3 --out $O/wd_o_lr3e3 > $O/wd_o_lr3e3.log 2>&1
echo "QUEUE_GPU1_DONE"
