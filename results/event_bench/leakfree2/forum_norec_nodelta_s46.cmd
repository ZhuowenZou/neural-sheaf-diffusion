#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
exec results/event_bench/queues/wait_launch.sh 22000 results/event_bench/leakfree2/forum_norec_nodelta_s46.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset thgl-forum --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --train-negatives-per-pos 32 --epochs 4 --patience 3 --min-epochs 2 --track-val-edges 20000 --model faithful --predict-from-previous --save-checkpoint --seed 46 --no-delta-t --out results/event_bench/leakfree2/forum_norec_nodelta_s46
