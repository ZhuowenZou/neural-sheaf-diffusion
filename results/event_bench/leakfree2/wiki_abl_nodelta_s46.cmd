#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 2500 results/event_bench/leakfree2/wiki_abl_nodelta_s46.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset tgbl-wiki --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4 --model faithful --lr 3e-4 --recurrency-decoder --predict-from-previous --save-checkpoint --no-delta-t --seed 46 --out results/event_bench/leakfree2/wiki_abl_nodelta_s46
