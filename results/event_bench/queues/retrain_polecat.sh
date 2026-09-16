#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; T=results/event_bench/true_mrr
[ -f $T/polecat_f_s43/results.csv ] || $PY -m exp.run_event_benchmark --dataset tkgl-polecat --model faithful --lr 3e-3 --epochs 10 --patience 5 --min-epochs 4 --track-val-edges 30000 --relation-in-input --recurrency-decoder --recurrency-untyped --save-checkpoint --seed 43 --out $T/polecat_f_s43 > $T/polecat_f_s43.log 2>&1
echo RETRAIN_POLECAT_DONE
