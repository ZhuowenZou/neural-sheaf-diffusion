#!/bin/bash
# Per-event analysis of each smallpedia ablation checkpoint (run after the
# analysis runner is validated against the harness cross-check).
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; A=results/analytic
head="--relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric"
declare -A FLAGS=( [sp_abl_identity]="$head --sheaf-identity" [sp_abl_nodelta]="$head --no-delta-t" [sp_abl_curonly]="$head --sheaf-conditioning current_only" )
for name in sp_abl_identity sp_abl_nodelta sp_abl_curonly; do
  ck=$A/$name/best.pt; [ -f $ck ] || { echo "missing $ck"; continue; }
  for ts in 1.0 0.25 4.0; do
    out=$A/analysis_${name}_ts$ts; [ -f $out/summary.json ] && continue
    $PY -m exp.analyze_event_model --dataset tkgl-smallpedia --checkpoint $ck ${FLAGS[$name]} --time-scale $ts --out $out > $out.log 2>&1
  done
done
echo ANALYSIS_ABLATIONS_DONE
