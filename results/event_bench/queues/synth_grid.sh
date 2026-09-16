#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
A=results/analytic
for regime in full notrans static uniform; do
  for variant in full identity nodelta current_only original; do
    out=$A/synth_${regime}_${variant}
    [ -f $out/summary.json ] && continue
    $PY -m exp.run_synthetic_sheaf --data $A/synth_$regime --variant $variant --epochs 4 --seed 43 --out $out > $out.log 2>&1
  done
done
# time-perturbation on the full regime (full core vs no-delta): test gaps x0.25 and x4
for variant in full nodelta original; do for ts in 0.25 4; do
  out=$A/synth_full_${variant}_ts$ts
  [ -f $out/summary.json ] && continue
  $PY -m exp.run_synthetic_sheaf --data $A/synth_full --variant $variant --epochs 4 --seed 43 --time-scale $ts --out $out > $out.log 2>&1
done; done
echo SYNTH_GRID_DONE
