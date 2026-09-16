#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 14000 results/event_bench/leakfree2/polecat_abl_nodelta.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-polecat --epochs 10 --patience 5 --min-epochs 4 --track-val-edges 30000 --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --predict-from-previous --save-checkpoint --seed 43 --no-delta-t --out results/event_bench/leakfree2/polecat_abl_nodelta
