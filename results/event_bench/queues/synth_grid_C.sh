#!/bin/bash
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; A=results/analytic
for variant in full nodelta original; do for ts in 0.25 4; do
  out=$A/synth_full_${variant}_ts$ts; [ -f $out/summary.json ] && continue; mkdir $out.lock 2>/dev/null || continue
  $PY -m exp.run_synthetic_sheaf --data $A/synth_full --variant $variant --epochs 4 --seed 43 --time-scale $ts --out $out > $out.log 2>&1
done; done
echo SYNTH_GRID_C_DONE
