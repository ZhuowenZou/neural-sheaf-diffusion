#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="7"
exec results/event_bench/queues/wait_launch.sh 2000 results/review_2026_09_22/queue/wiki_diag_s45_lr1e-3.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tgbl-wiki --model faithful --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4 --recurrency-decoder --predict-from-previous --save-checkpoint --rng-isolation --dump-query-ranks --backbone diag_ssm --spatial identity --lr 1e-3 --seed 45 --out results/review_2026_09_22/matched/wiki_diag_s45_lr1e-3 > results/review_2026_09_22/queue/wiki_diag_s45_lr1e-3.log 2>&1
