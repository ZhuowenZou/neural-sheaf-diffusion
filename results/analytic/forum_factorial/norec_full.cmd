#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TSD_SCORE_FROM_PREVIOUS_STATE=1
exec results/event_bench/queues/wait_launch.sh 22000 results/analytic/forum_factorial/norec_full.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -m exp.analyze_event_model --dataset thgl-forum --time-window 600 --train-edges-cap 4000000 --test-edges-cap 1000000 --node-type-emb --relation-in-input --no-residual --checkpoint results/event_bench/leakfree2/forum_norec_full/best.pt  --out results/analytic/forum_factorial/norec_full
