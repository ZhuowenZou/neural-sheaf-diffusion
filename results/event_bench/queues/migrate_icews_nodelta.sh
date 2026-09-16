#!/bin/bash
# Vacate colleague GPU 2 (compensation budget went negative when colleagues left our cards):
# once icews_abl_nodelta has finished TRAINING (best.pt written; the in-run final eval would take
# ~10 h on GPU 2), stop it, and run the identical final evaluation on our own GPUs via --eval-only.
cd /home/zhuowez1/project/neural-sheaf-diffusion
until grep -q "checkpoint saved" results/event_bench/leakfree2/icews_abl_nodelta.log 2>/dev/null && [ -f results/event_bench/leakfree2/icews_abl_nodelta/best.pt ]; do sleep 120; done
mv -f results/event_bench/leakfree2/icews_abl_nodelta.cmd results/event_bench/leakfree2/icews_abl_nodelta.cmd.migrated 2>/dev/null   # daemon must not relaunch training
P=$(pgrep -f "exp.run_event_benchmark.*--out results/event_bench/leakfree2/icews_abl_nodelta$" | while read q; do [ "$(ps -o comm= -p $q)" = python ] && echo $q; done | head -1)
[ -n "$P" ] && kill $P && sleep 5
echo "$(date '+%F %T') icews_abl_nodelta: training done on gpu2, process $P stopped after best.pt; final eval migrated to GPU 0/7 (eval-only)" >> results/monitor/actions.log
printf '#!/bin/bash\ncd %s\nexec results/event_bench/queues/wait_launch.sh 16000 %s/icews_abl_nodelta_eval.log %s -m exp.run_event_benchmark %s --eval-only %s/icews_abl_nodelta/best.pt --out %s/icews_abl_nodelta_eval\n' "/home/zhuowez1/project/neural-sheaf-diffusion" "results/event_bench/leakfree2" "/home/zhuowez1/miniconda3/envs/nsd/bin/python" "--dataset tkgl-icews --epochs 6 --patience 3 --min-epochs 3 --train-edges-cap 2000000 --track-val-edges 30000 --temporal-d-model 64 --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --model faithful --predict-from-previous --save-checkpoint --no-delta-t --seed 43" "results/event_bench/leakfree2" "results/event_bench/leakfree2" > results/event_bench/leakfree2/icews_abl_nodelta_eval.cmd; chmod +x results/event_bench/leakfree2/icews_abl_nodelta_eval.cmd
