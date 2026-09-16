#!/bin/bash
# Leak-free smallpedia ablation lane + per-event analyses (audit 2026-09-04).
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; WL=results/event_bench/queues/wait_launch.sh
L=results/event_bench/leakfree2; A=results/analytic/lf2; mkdir -p $A
H="--relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric"
SP="--dataset tkgl-smallpedia --model faithful --lr 1e-2 --track-val-edges 50000 --epochs 15 --patience 6 --min-epochs 6 $H --predict-from-previous --save-checkpoint"
run() { local name=$1; shift; [ -f $L/$name/results.csv ] && return 0; $WL 30000 $L/$name.log $PY -m exp.run_event_benchmark $SP "$@" --seed 43 --out $L/$name > $L/$name.log 2>&1; }
run sp_abl_identity --sheaf-identity
run sp_abl_nodelta --no-delta-t
run sp_abl_curonly --sheaf-conditioning current_only
# per-event analyses on the leak-free checkpoints (waits for the champion from the main campaign)
export TSD_SCORE_FROM_PREVIOUS_STATE=1
for name in sp_f_s43 sp_abl_identity sp_abl_nodelta sp_abl_curonly; do
  until [ -f $L/$name/best.pt ] && [ -f $L/$name/results.csv ]; do sleep 600; done
  for ts in 1.0 0.25 4.0; do
    [ -f $A/analysis_${name}_ts$ts/summary.json ] && continue
    $WL 30000 $A/analysis_${name}_ts$ts.log $PY -m exp.analyze_event_model --dataset tkgl-smallpedia --checkpoint $L/$name/best.pt $H --time-scale $ts --out $A/analysis_${name}_ts$ts > $A/analysis_${name}_ts$ts.log 2>&1
  done
done
echo "LEAKFREE SP ABLATIONS+ANALYSES DONE $(date)"
