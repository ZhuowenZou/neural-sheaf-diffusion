#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
until [ -f results/event_bench/leakfree2/icews_f_s47/best.pt ] && [ -f results/event_bench/leakfree2/icews_f_s47/results.csv ]; do sleep 600; done
exec results/event_bench/queues/wait_launch.sh 16000 results/event_bench/leakfree2/icews_eval_s47.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-icews --epochs 6 --patience 3 --min-epochs 3 --train-edges-cap 2000000 --track-val-edges 30000 --temporal-d-model 64 --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --model faithful --predict-from-previous --save-checkpoint --eval-only results/event_bench/leakfree2/icews_f_s47/best.pt --seed 47 --out results/event_bench/leakfree2/icews_eval_s47
