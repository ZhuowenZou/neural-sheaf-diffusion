#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_ONLY_GPUS="3"
exec results/event_bench/queues/wait_launch.sh 1500 results/review_2026_09_22/queue/synth_gen_s1_tsd_recoff_s46.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.run_event_benchmark --dataset synth-history:results/review_2026_09_22/synthetic/gen_s1/data.npz --model faithful --time-window 20000 --context-edges 2000 --track-val-edges 3000 --train-negatives-per-pos 32 --epochs 6 --patience 3 --min-epochs 3 --lr 1e-3 --predict-from-previous --save-checkpoint --rng-isolation --dump-query-ranks  --seed 46 --out results/review_2026_09_22/synthetic/runs/synth_gen_s1_tsd_recoff_s46 > results/review_2026_09_22/queue/synth_gen_s1_tsd_recoff_s46.log 2>&1
