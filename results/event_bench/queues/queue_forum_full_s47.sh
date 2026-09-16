#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
while kill -0 3955680 2>/dev/null; do sleep 300; done
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
$PY -m exp.run_event_benchmark --dataset thgl-forum --model faithful --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --train-negatives-per-pos 32 --epochs 4 --patience 3 --min-epochs 2 --track-val-edges 20000 --save-checkpoint --seed 47 --out results/event_bench/forum_full_s47 > results/event_bench/forum_full_s47.log 2>&1
echo QUEUE_FORUM_FULL_S47_DONE
