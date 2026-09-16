#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; T=results/event_bench/true_mrr
W="--dataset tgbl-wiki --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4 --save-checkpoint"
for s in 43 46 47; do [ -f $T/wiki_f_s$s/results.csv ] || $PY -m exp.run_event_benchmark $W --model faithful --lr 3e-4 --recurrency-decoder --seed $s --out $T/wiki_f_s$s > $T/wiki_f_s$s.log 2>&1; done
for s in 43 46 47; do [ -f $T/wiki_o_s$s/results.csv ] || $PY -m exp.run_event_benchmark $W --model original --lr 3e-3 --seed $s --out $T/wiki_o_s$s > $T/wiki_o_s$s.log 2>&1; done
echo RETRAIN_WIKI_DONE
