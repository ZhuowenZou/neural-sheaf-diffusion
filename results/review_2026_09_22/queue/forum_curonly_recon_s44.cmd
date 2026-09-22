#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="7 6"
exec results/event_bench/queues/wait_launch.sh 14000 results/review_2026_09_22/queue/forum_curonly_recon_s44.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset thgl-forum --model faithful --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --train-negatives-per-pos 32 --epochs 4 --patience 3 --min-epochs 2 --track-val-edges 20000 --predict-from-previous --save-checkpoint --dump-query-ranks --recurrency-decoder --recurrency-untyped --sheaf-conditioning current_only --seed 44 --out results/review_2026_09_22/forum/forum_curonly_recon_s44 > results/review_2026_09_22/queue/forum_curonly_recon_s44.log 2>&1
