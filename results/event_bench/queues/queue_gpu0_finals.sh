#!/bin/bash
# Finals: wiki both models seeds 46/47, genre faithful seed 46
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset TGB_ROOT
O=results/event_bench
w="--dataset tgbl-wiki --time-window 600 --epochs 15 --patience 6 --min-epochs 8 --track-val-edges 8000"
$PY -m exp.run_event_benchmark $w --model faithful --lr 3e-4 --seed 46 --out $O/wiki_f_lr3e4_s46 > $O/wiki_f_lr3e4_s46.log 2>&1
$PY -m exp.run_event_benchmark $w --model faithful --lr 3e-4 --seed 47 --out $O/wiki_f_lr3e4_s47 > $O/wiki_f_lr3e4_s47.log 2>&1
$PY -m exp.run_event_benchmark $w --model original --lr 3e-3 --seed 46 --out $O/wiki_o_lr3e3_s46 > $O/wiki_o_lr3e3_s46.log 2>&1
$PY -m exp.run_event_benchmark $w --model original --lr 3e-3 --seed 47 --out $O/wiki_o_lr3e3_s47 > $O/wiki_o_lr3e3_s47.log 2>&1
$PY -m exp.run_faithful_trade --dataset tgbn-genre --seeds 46 --train-cap -1 \
  --override '{"time_window": 604800, "eval_every": 2, "memory_readout": true, "epochs": 40, "early_stopping_patience": 12, "min_epochs_before_stopping": 16, "lr": 0.005}' \
  --out $O/genre_f_lr005_s46.csv > $O/genre_f_lr005_s46.log 2>&1
echo "QUEUE_GPU0_FINALS_DONE"
