#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
for s in 46 47; do
  $PY -m exp.run_event_benchmark --dataset tkgl-wikidata --model faithful --epochs 4 --train-edges-cap 2000000 --track-val-edges 300000 --temporal-d-model 32 --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --seed $s --out results/event_bench/wd_f_untyped_sym_s$s > results/event_bench/wd_f_untyped_sym_s$s.log 2>&1
done
echo QUEUE_WD_SYM_DONE
