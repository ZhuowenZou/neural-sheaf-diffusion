#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="2"
exec results/event_bench/queues/wait_launch.sh 2500 results/review_2026_09_22/queue/wiki_tsd_clock-node_update_s47.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tgbl-wiki --model faithful --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4 --recurrency-decoder --predict-from-previous --save-checkpoint --rng-isolation --dump-query-ranks --clock node_update --clock-diagnostics --lr 1e-3 --seed 47 --out results/review_2026_09_22/clock/wiki_tsd_clock-node_update_s47 > results/review_2026_09_22/queue/wiki_tsd_clock-node_update_s47.log 2>&1
