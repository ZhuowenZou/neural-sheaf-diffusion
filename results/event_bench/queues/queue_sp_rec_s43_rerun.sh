#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
while pgrep -f "sp_rec_final_s47" > /dev/null; do sleep 60; done
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
$PY -m exp.run_event_benchmark --dataset tkgl-smallpedia --model faithful --lr 1e-2 \
  --relation-in-input --recurrency-decoder --epochs 15 --patience 6 --min-epochs 6 \
  --track-val-edges 50000 --seed 43 --out results/event_bench/sp_rec_final_s43_v2 > results/event_bench/sp_rec_final_s43_v2.log 2>&1
echo QUEUE_S43_RERUN_DONE
