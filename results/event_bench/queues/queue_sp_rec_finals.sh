#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
for s in 43 46 47; do
  $PY -m exp.run_event_benchmark --dataset tkgl-smallpedia --model faithful --lr 1e-2 \
    --relation-in-input --recurrency-decoder --epochs 15 --patience 6 --min-epochs 6 \
    --track-val-edges 50000 --seed $s --out results/event_bench/sp_rec_final_s$s > results/event_bench/sp_rec_final_s$s.log 2>&1
done
echo QUEUE_SP_REC_DONE
