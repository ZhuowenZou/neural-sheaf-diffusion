#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; A=results/analytic
for regime in uniform static; do for variant in full identity nodelta current_only original; do
  out=$A/synth_${regime}_${variant}; [ -f $out/summary.json ] && continue; mkdir $out.lock 2>/dev/null || continue
  $PY -m exp.run_synthetic_sheaf --data $A/synth_$regime --variant $variant --epochs 4 --seed 43 --out $out > $out.log 2>&1
done; done
echo SYNTH_GRID_B_DONE
