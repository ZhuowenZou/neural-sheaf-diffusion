#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="2"
exec results/event_bench/queues/wait_launch.sh 16000 results/review_2026_09_22/queue/icews_curonly_recon_s45.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tkgl-icews --model faithful --epochs 6 --patience 3 --min-epochs 3 --train-edges-cap 2000000 --track-val-edges 30000 --temporal-d-model 64 --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --predict-from-previous --save-checkpoint --sheaf-conditioning current_only --seed 45 --out results/review_2026_09_22/icews/icews_curonly_recon_s45 > results/review_2026_09_22/queue/icews_curonly_recon_s45.log 2>&1
