#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 16000 results/event_bench/leakfree2/icews_eval_s43.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-icews --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --temporal-d-model 64 --train-edges-cap 2000000 --track-val-edges 30000 --predict-from-previous --eval-only results/event_bench/leakfree2/icews_f_s43/best.pt --out results/event_bench/leakfree2/icews_eval_s43
