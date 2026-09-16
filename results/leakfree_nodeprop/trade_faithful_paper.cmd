#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
export TGB_ROOT=/home/zhuowez1/project/neural-sheaf-diffusion/datasets
exec results/event_bench/queues/wait_launch.sh 6000 results/leakfree_nodeprop/trade_faithful_paper.log /home/zhuowez1/miniconda3/envs/nsd/bin/python -u -m exp.run_faithful_trade --dataset tgbn-trade --seeds 43 44 45 46 47 --out results/leakfree_nodeprop/trade_faithful_paper.csv
