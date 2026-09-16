#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; T=results/event_bench/true_mrr
S="--dataset thgl-software --time-window 3600 --track-val-edges 20000 --lr 1e-3 --train-negatives-per-pos 32 --save-checkpoint"
for s in 43 46 47; do [ -f $T/sw_f_s$s/results.csv ] || $PY -m exp.run_event_benchmark $S --model faithful --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --epochs 20 --patience 8 --min-epochs 8 --seed $s --out $T/sw_f_s$s > $T/sw_f_s$s.log 2>&1; done
[ -f $T/sw_o_s43/results.csv ] || $PY -m exp.run_event_benchmark $S --model original --epochs 12 --patience 5 --min-epochs 6 --seed 43 --out $T/sw_o_s43 > $T/sw_o_s43.log 2>&1
echo RETRAIN_SOFTWARE_DONE
