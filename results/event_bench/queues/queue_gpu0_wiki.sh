#!/bin/bash
# tgbl-wiki lr sweep, both models, window 600s, seed 43
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset TGB_ROOT
O=results/event_bench
common="--dataset tgbl-wiki --time-window 600 --epochs 15 --patience 6 --min-epochs 8 --track-val-edges 8000 --seed 43"
$PY -m exp.run_event_benchmark $common --model faithful --lr 3e-4  --out $O/wiki_f_lr3e4  > $O/wiki_f_lr3e4.log  2>&1
$PY -m exp.run_event_benchmark $common --model faithful --lr 1e-3  --out $O/wiki_f_lr1e3  > $O/wiki_f_lr1e3.log  2>&1
$PY -m exp.run_event_benchmark $common --model faithful --lr 3e-3  --out $O/wiki_f_lr3e3  > $O/wiki_f_lr3e3.log  2>&1
$PY -m exp.run_event_benchmark $common --model original --lr 1e-3  --out $O/wiki_o_lr1e3  > $O/wiki_o_lr1e3.log  2>&1
$PY -m exp.run_event_benchmark $common --model original --lr 3e-3  --out $O/wiki_o_lr3e3  > $O/wiki_o_lr3e3.log  2>&1
$PY -m exp.run_event_benchmark $common --model original --lr 1e-2  --out $O/wiki_o_lr1e2  > $O/wiki_o_lr1e2.log  2>&1
echo "QUEUE_GPU0_DONE"
