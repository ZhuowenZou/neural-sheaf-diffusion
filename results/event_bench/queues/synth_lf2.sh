#!/bin/bash
# Leak-free synthetic analytic campaign (audit 2026-09-04): run_synthetic_sheaf.py now
# scores snapshot k from the state after k-1 by default (train + eval).
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; WL=results/event_bench/queues/wait_launch.sh
A=results/analytic; O=$A/synth_lf2; mkdir -p $O
job() { local out=$1; shift; [ -f $out/summary.json ] && return 0; mkdir $out.lock 2>/dev/null || return 0
        $WL 8000 $out.log $PY -m exp.run_synthetic_sheaf "$@" --epochs 4 --seed 43 --out $out > $out.log 2>&1; }
lane() { for regime in "$@"; do for variant in full identity nodelta current_only original; do
           job $O/synth_${regime}_${variant} --data $A/synth_$regime --variant $variant; done; done; }
( lane full ) &
sleep 15
( lane notrans static ) &
sleep 15
( lane uniform; for variant in full nodelta original; do for ts in 0.25 4; do
    job $O/synth_full_${variant}_ts$ts --data $A/synth_full --variant $variant --time-scale $ts; done; done ) &
wait; echo "synth_lf2 finished $(date)"
