#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
A=results/analytic
for ts in 1.0 0.25 4.0; do
  $PY -m exp.analyze_event_model --dataset tkgl-smallpedia --checkpoint results/event_bench/sp_untyped_sym_s43/best.pt \
    --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --time-scale $ts \
    --out $A/analysis_sp_full_ts$ts > $A/analysis_sp_full_ts$ts.log 2>&1
done
echo ANALYSIS_SP_FULL_DONE
