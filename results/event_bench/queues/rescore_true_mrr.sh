#!/bin/bash
# Re-score every existing checkpoint with the corrected (by-name) MRR.
cd /home/zhuowez1/project/neural-sheaf-diffusion
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; A=results/analytic; R=results/event_bench
sp="--dataset tkgl-smallpedia --model faithful --lr 1e-2 --track-val-edges 50000 --seed 43"
head="--relation-in-input --recurrency-decoder --recurrency-untyped --recurrency-symmetric"
run() { name=$1; ck=$2; shift 2; [ -f $ck ] || { echo "missing $ck"; return; }; [ -f $A/rescore_$name/results.csv ] && return; $PY -m exp.run_event_benchmark "$@" --eval-only $ck --out $A/rescore_$name > $A/rescore_$name.log 2>&1; }
run sp_sym_s43 $R/sp_untyped_sym_s43/best.pt $sp $head
run sp_sym_s46 $R/sp_untyped_sym_s46/best.pt $sp $head
run sp_sym_s47 $R/sp_untyped_sym_s47/best.pt $sp $head
run sp_abl_identity $A/sp_abl_identity/best.pt $sp $head --sheaf-identity
run sp_abl_curonly $A/sp_abl_curonly/best.pt $sp $head --sheaf-conditioning current_only
pc="--dataset tkgl-polecat --model faithful --lr 1e-2 --track-val-edges 30000 --relation-in-input --recurrency-decoder --recurrency-untyped"
run polecat_s46 $R/polecat_untyped_s46/best.pt $pc --seed 46
run polecat_s47 $R/polecat_untyped_s47/best.pt $pc --seed 47
fo="--dataset thgl-forum --model faithful --time-window 600 --train-edges-cap 4000000 --lr 1e-3 --node-type-emb --relation-in-input --recurrency-decoder --recurrency-untyped --train-negatives-per-pos 32 --track-val-edges 20000"
run forum_s43 $R/forum_full_s43/best.pt $fo --seed 43
run forum_s46 $R/forum_full_s46/best.pt $fo --seed 46
echo RESCORE_DONE
