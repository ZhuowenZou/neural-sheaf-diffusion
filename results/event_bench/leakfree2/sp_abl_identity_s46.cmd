#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 12000 results/event_bench/leakfree2/sp_abl_identity_s46.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-smallpedia --track-val-edges 50000 --epochs 15 --patience 6 --min-epochs 6 --model faithful --lr 1e-2 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --predict-from-previous --save-checkpoint --sheaf-identity --seed 46 --out results/event_bench/leakfree2/sp_abl_identity_s46
