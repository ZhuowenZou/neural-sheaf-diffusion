#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 12000 results/event_bench/leakfree2/wd_abl_identity.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 300000 --temporal-d-model 32 --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --predict-from-previous --save-checkpoint --sheaf-identity --seed 43 --out results/event_bench/leakfree2/wd_abl_identity
