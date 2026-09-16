#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; T=results/event_bench/true_mrr
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
W="--dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 300000 --temporal-d-model 32 --save-checkpoint"
for s in 43 46 47; do [ -f $T/wd_f_s$s/results.csv ] || $PY -m exp.run_event_benchmark $W --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --seed $s --out $T/wd_f_s$s > $T/wd_f_s$s.log 2>&1; done
[ -f $T/wd_o_s43/results.csv ] || $PY -m exp.run_event_benchmark $W --model original --lr 3e-3 --seed 43 --out $T/wd_o_s43 > $T/wd_o_s43.log 2>&1
echo RETRAIN_WIKIDATA_DONE
