#!/bin/bash
# tkgl-smallpedia lr sweep both models, then wikidata faithful (stable-A) lr sweep
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
unset TGB_ROOT
O=results/event_bench
sp="--dataset tkgl-smallpedia --epochs 25 --patience 8 --min-epochs 10 --track-val-edges 50000 --seed 43"
$PY -m exp.run_event_benchmark $sp --model faithful --lr 1e-3 --out $O/sp_f_lr1e3 > $O/sp_f_lr1e3.log 2>&1
$PY -m exp.run_event_benchmark $sp --model original --lr 1e-3 --out $O/sp_o_lr1e3 > $O/sp_o_lr1e3.log 2>&1
$PY -m exp.run_event_benchmark $sp --model faithful --lr 3e-3 --out $O/sp_f_lr3e3 > $O/sp_f_lr3e3.log 2>&1
$PY -m exp.run_event_benchmark $sp --model original --lr 3e-3 --out $O/sp_o_lr3e3 > $O/sp_o_lr3e3.log 2>&1
$PY -m exp.run_event_benchmark $sp --model faithful --lr 3e-4 --out $O/sp_f_lr3e4 > $O/sp_f_lr3e4.log 2>&1
$PY -m exp.run_event_benchmark $sp --model original --lr 3e-4 --out $O/sp_o_lr3e4 > $O/sp_o_lr3e4.log 2>&1
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
wd="--dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 100000 --temporal-d-model 32 --seed 43"
$PY -m exp.run_event_benchmark $wd --model faithful --lr 1e-3 --out $O/wd_f_lr1e3 > $O/wd_f_lr1e3.log 2>&1
$PY -m exp.run_event_benchmark $wd --model faithful --lr 3e-4 --out $O/wd_f_lr3e4 > $O/wd_f_lr3e4.log 2>&1
$PY -m exp.run_event_benchmark $wd --model faithful --lr 3e-3 --out $O/wd_f_lr3e3 > $O/wd_f_lr3e3.log 2>&1
echo "QUEUE_GPU4_DONE"
