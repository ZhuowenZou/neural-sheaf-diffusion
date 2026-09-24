#!/bin/bash
# E3 matrix: 10 arms x {nonflat, flat} x seeds {43,44,45}; 300 iterations; GPUs 0 and 7 (ours). Oracles computed once per variant.
cd /home/zhuowez1/project/neural-sheaf-diffusion
OUT=results/histgeom_e3_2026_09_24; PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; i=0
for v in nonflat flat; do for seed in 43 44 45; do
  for arm in tsd tsd_mlpdec curonly identity nodeframe attention gru diag coreoff oracle_maps; do
    g=$([ $((i % 2)) = 0 ] && echo 0 || echo 7); i=$((i+1))
    extra=""; [ "$arm" = tsd ] && [ $seed = 43 ] && extra="--oracles"
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nohup $PY -m exp.histgeom.transport_task --arm $arm --variant $v --seed $seed --iters 300 $extra \
      --out $OUT/$v/${arm}_s$seed > $OUT/logs/${v}_${arm}_s$seed.log 2>&1 &
    sleep 1
  done
done; done
wait
