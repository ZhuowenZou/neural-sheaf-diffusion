#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 16000 results/event_bench/leakfree2/icews_norec_identity.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-icews --epochs 6 --patience 3 --min-epochs 3 --train-edges-cap 2000000 --track-val-edges 30000 --temporal-d-model 64 --lr 3e-3 --relation-in-input --model faithful --predict-from-previous --save-checkpoint --seed 43 --sheaf-identity --out results/event_bench/leakfree2/icews_norec_identity
