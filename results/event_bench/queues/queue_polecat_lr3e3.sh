#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
while kill -0 1896501 2>/dev/null; do sleep 120; done
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -m exp.run_event_benchmark --dataset tkgl-polecat --model faithful --lr 3e-3 --epochs 10 --patience 5 --min-epochs 4 --track-val-edges 30000 --relation-in-input --recurrency-decoder --recurrency-untyped --seed 43 --out results/event_bench/polecat_untyped_lr3e3 > results/event_bench/polecat_untyped_lr3e3.log 2>&1
echo QUEUE_POLECAT_LR3E3_DONE
