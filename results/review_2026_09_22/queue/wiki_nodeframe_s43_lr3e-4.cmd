#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="3 6"
exec results/event_bench/queues/wait_launch.sh 2000 results/review_2026_09_22/queue/wiki_nodeframe_s43_lr3e-4.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tgbl-wiki --model faithful --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4 --recurrency-decoder --predict-from-previous --save-checkpoint --rng-isolation --spatial node_frame --lr 3e-4 --seed 43 --out results/review_2026_09_22/matched/wiki_nodeframe_s43_lr3e-4 > results/review_2026_09_22/queue/wiki_nodeframe_s43_lr3e-4.log 2>&1
