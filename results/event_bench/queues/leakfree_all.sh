#!/bin/bash
# Leak-free re-measurement campaign (audit 2026-09-04).
# Every run: predict-from-previous protocol (score snapshot k from the state
# after k-1, train + eval), Delta_k in units of the median training gap
# (--delta-time-scale auto, default), checkpoint saved, true MRR by name.
# Champion configs are the pre-audit per-dataset champions; nothing else changed.
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
WL=results/event_bench/queues/wait_launch.sh
L=results/event_bench/leakfree; mkdir -p $L
C="--predict-from-previous --save-checkpoint"
run() { # run <gpu_free_mb> <name> <args...>
  local mb=$1 name=$2; shift 2
  [ -f $L/$name/results.csv ] && return 0
  $WL $mb $L/$name.log $PY -m exp.run_event_benchmark "$@" $C --out $L/$name > $L/$name.log 2>&1
}
W="--dataset tgbl-wiki --time-window 600 --track-val-edges 8000 --train-negatives-per-pos 32 --epochs 8 --patience 5 --min-epochs 4"
S="--dataset thgl-software --time-window 3600 --track-val-edges 20000 --lr 1e-3 --train-negatives-per-pos 32"
WD="--dataset tkgl-wikidata --epochs 4 --train-edges-cap 2000000 --track-val-edges 300000 --temporal-d-model 32"
SP="--dataset tkgl-smallpedia --track-val-edges 50000 --epochs 40 --patience 10 --min-epochs 15"
PC="--dataset tkgl-polecat --epochs 10 --patience 5 --min-epochs 4 --track-val-edges 30000"
FO="--dataset thgl-forum --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --train-negatives-per-pos 32 --epochs 4 --patience 3 --min-epochs 2 --track-val-edges 20000"
IC="--dataset tkgl-icews --epochs 6 --patience 3 --min-epochs 3 --train-edges-cap 2000000 --track-val-edges 30000 --temporal-d-model 64 --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --skip-final-eval"

# lane 1: wiki (faithful REC, original), then extra seeds
( run 30000 wiki_f_s43 $W --model faithful --lr 3e-4 --recurrency-decoder --seed 43
  run 30000 wiki_o_s43 $W --model original --lr 3e-3 --seed 43
  for s in 46 47; do run 30000 wiki_f_s$s $W --model faithful --lr 3e-4 --recurrency-decoder --seed $s; done ) &
sleep 20
# lane 2: software
( run 40000 sw_f_s43 $S --model faithful --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --epochs 20 --patience 8 --min-epochs 8 --seed 43
  run 40000 sw_o_s43 $S --model original --epochs 12 --patience 5 --min-epochs 6 --seed 43
  for s in 46 47; do run 40000 sw_f_s$s $S --model faithful --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --epochs 20 --patience 8 --min-epochs 8 --seed $s; done ) &
sleep 20
# lane 3: smallpedia (full champion: REL + REC typed/untyped/sym), original
( run 30000 sp_f_s43 $SP --model faithful --lr 1e-2 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --seed 43
  run 30000 sp_o_s43 $SP --model original --lr 1e-2 --seed 43
  for s in 46 47; do run 30000 sp_f_s$s $SP --model faithful --lr 1e-2 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --seed $s; done ) &
sleep 20
# lane 4: polecat, then wikidata (both dense-vocabulary; 45 GB)
( run 40000 polecat_f_s43 $PC --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --seed 43
  run 40000 polecat_o_s43 $PC --model original --lr 3e-3 --seed 43
  run 45000 wd_f_s43 $WD --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric --seed 43
  run 45000 wd_o_s43 $WD --model original --lr 3e-3 --seed 43 ) &
sleep 20
# lane 5: forum (60 GB), then icews training (45 GB; full eval afterwards via --eval-only)
( run 60000 forum_f_s43 $FO --model faithful --seed 43
  run 45000 icews_f_s43 $IC --model faithful --seed 43 ) &
wait
echo "leak-free campaign finished: $(date)"
