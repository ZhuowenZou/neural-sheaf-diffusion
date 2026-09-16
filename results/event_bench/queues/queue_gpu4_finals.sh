#!/bin/bash
# Finals: smallpedia both models seeds 46/47, then wikidata seed 46 both models
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
unset TGB_ROOT
O=results/event_bench
sp="--dataset tkgl-smallpedia --epochs 25 --patience 8 --min-epochs 10 --track-val-edges 50000"
$PY -m exp.run_event_benchmark $sp --model original --lr 3e-3 --seed 46 --out $O/sp_o_lr3e3_s46 > $O/sp_o_lr3e3_s46.log 2>&1
$PY -m exp.run_event_benchmark $sp --model original --lr 3e-3 --seed 47 --out $O/sp_o_lr3e3_s47 > $O/sp_o_lr3e3_s47.log 2>&1
$PY -m exp.run_event_benchmark $sp --model faithful --lr 1e-2 --seed 46 --out $O/sp_f_lr1e2_s46 > $O/sp_f_lr1e2_s46.log 2>&1
$PY -m exp.run_event_benchmark $sp --model faithful --lr 1e-2 --seed 47 --out $O/sp_f_lr1e2_s47 > $O/sp_f_lr1e2_s47.log 2>&1
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
wd="--dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 100000 --temporal-d-model 32"
$PY -m exp.run_event_benchmark $wd --model original --lr 3e-3 --seed 46 --out $O/wd_o_lr3e3_s46 > $O/wd_o_lr3e3_s46.log 2>&1
$PY -m exp.run_event_benchmark $wd --model faithful --lr 1e-3 --seed 46 --out $O/wd_f_lr1e3_s46 > $O/wd_f_lr1e3_s46.log 2>&1
echo "QUEUE_GPU4_FINALS_DONE"
