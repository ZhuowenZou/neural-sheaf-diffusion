#!/bin/bash
# Session-independent GPU + campaign monitor (run under `setsid nohup`; survives closed remote sessions).
#  * every INTERVAL seconds: append per-GPU memory/utilisation to results/monitor/gpu_log.csv,
#    rewrite results/monitor/STATUS.md (campaign runs DONE/RUNNING/CRASHED/WAITING, node-property, synthetic v4)
#  * actions: launch the icews full evaluation once its training checkpoint exists (once);
#             retry a faithful benchmark run that died with CUDA OOM (once per run), reconstructing the
#             command from leakfree2_all.sh; everything is appended to results/monitor/actions.log
cd /home/zhuowez1/project/neural-sheaf-diffusion || exit 1
M=results/monitor; mkdir -p $M
L=results/event_bench/leakfree2; Q=results/event_bench/queues
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python; WL=$Q/wait_launch.sh
INTERVAL=${INTERVAL:-180}   # 3-minute polling (user request 2026-09-11)
OWN=${TSD_OWN_GPUS:-"0 7"}; TAKEOVER=${TSD_IDLE_TAKEOVER_SEC:-3600}; C=$M/claims; mkdir -p $C
touch $M/retried.txt $M/actions.log
echo "monitor started $(date) pid $$" >> $M/actions.log

gpu_line() { CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; }

while true; do
  ts=$(date '+%F %T')
  gpu_line | awk -v t="$ts" -F', ' '{printf "%s,%s,%d,%d,%s\n", t, $1, $2, $3, $4}' >> $M/gpu_log.csv
  # Colleague-occupancy tracking (policy 2026-09-11): for every GPU record the epoch since
  # which NO other user's process has been on it (0 = occupied by a colleague now).  The
  # launcher takes a colleague GPU (1-6) only after TAKEOVER seconds of such idleness.
  now=$(date +%s)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader > $M/.gpus.$$ 2>/dev/null
  declare -A busy=(); declare -A since=()
  [ -f $M/gpu_idle.txt ] && while read -r g s; do since[$g]=$s; done < $M/gpu_idle.txt
  while IFS=, read -r p u; do p=${p// /}; u=${u// /}
    [ "$(ps -o user= -p $p 2>/dev/null)" = "zhuowez1" ] && continue
    g=$(awk -F', ' -v u="$u" '$2==u{print $1}' $M/.gpus.$$); [ -n "$g" ] && busy[$g]=1
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
  { while IFS=, read -r g u; do g=${g// /}
      if [ -n "${busy[$g]}" ]; then echo "$g 0"
      elif [ "${since[$g]:-0}" -gt 0 ] 2>/dev/null; then echo "$g ${since[$g]}"
      else echo "$g $now"; fi
    done < $M/.gpus.$$; } > $M/gpu_idle.txt.tmp && mv -f $M/gpu_idle.txt.tmp $M/gpu_idle.txt
  rm -f $M/.gpus.$$
  ours=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read p m; do u=$(ps -o user= -p $p 2>/dev/null); [ "$u" = "zhuowez1" ] && echo "$p:$(echo $m | tr -d ' ')"; done | tr '\n' ' ')
  {
    echo "# Monitor status ($ts)"; echo
    echo "## GPU free (GiB) and utilisation"
    gpu_line | awk -F', ' '{printf "%s:%.0fGB(%s%%) ",$1,($3-$2)/1024,$4} END{print ""}'
    echo "our GPU processes (pid:MiB): ${ours:-none}"; echo
    echo "## placement policy: ours = GPUs ${OWN}; colleagues' GPUs become eligible after ${TAKEOVER}s without any other user's process"
    for f in $C/[0-9]*; do [ -f "$f" ] || continue; kill -0 $(basename $f) 2>/dev/null || rm -f $f; done
    while read -r g s; do
      own=0; for o in $OWN; do [ "$o" = "$g" ] && own=1; done
      if [ $own = 1 ]; then st="OURS"
      elif [ "${s:-0}" -eq 0 ] 2>/dev/null; then st="colleague busy"
      elif [ $((now - s)) -ge $TAKEOVER ]; then st="ELIGIBLE (idle $(( (now - s) / 60 )) min)"
      else st="idle $(( (now - s) / 60 )) min (eligible in $(( (TAKEOVER - now + s) / 60 )) min)"; fi
      printf -- "- gpu%s: %s\n" "$g" "$st"
    done < $M/gpu_idle.txt
    # compensation budget (policy 2026-09-12): colleagues' memory on OUR GPUs may be matched by ours on THEIRS
    budget=0; spent=0
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader > $M/.gpus2.$$ 2>/dev/null
    while IFS=, read -r p m u; do p=${p// /}; m=${m// /}; u=${u// /}; g=$(awk -F', ' -v u="$u" '$2==u{print $1}' $M/.gpus2.$$); [ -n "$g" ] || continue
      own=0; for o in $OWN; do [ "$o" = "$g" ] && own=1; done
      if [ "$(ps -o user= -p $p 2>/dev/null)" = "zhuowez1" ]; then [ $own = 0 ] && spent=$((spent + m)); else [ $own = 1 ] && budget=$((budget + m)); fi
    done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null); rm -f $M/.gpus2.$$
    echo "compensation: colleagues hold ${budget} MiB on our GPUs; we hold ${spent} MiB on theirs; budget left $((budget - spent)) MiB (a job may go to a colleague GPU while it fits in this budget and in that card's free memory)"
    echo "our launcher claims (pid:gpu/MiB): $(for f in $C/[0-9]*; do [ -f "$f" ] && printf '%s:%s ' $(basename $f) "$(tr ' ' '/' < $f)"; done)"; echo
    echo "## leakfree2 benchmark runs"
    for f in $L/*.log; do n=$(basename $f .log); case $n in queue*|*_queue|*.retry) continue;; esac
      if grep -q "FINAL val_mrr" $f 2>/dev/null; then echo "- DONE     $n: $(grep -m1 -h 'FINAL val_mrr' $f | cut -c1-90)"
      elif grep -q "OutOfMemoryError\|Traceback" $f 2>/dev/null; then echo "- CRASHED  $n: $(grep -m1 -h 'OutOfMemoryError\|Error' $f | cut -c1-80)"
      elif [ -f $L/$n/results.csv ]; then echo "- TRAINED  $n (no final evaluation in this run): $(grep -m1 -h 'best_track_val_mrr' $f | tr -d ' ,' | cut -c1-60)"
      elif pgrep -f -- "python -m exp.run_event_benchmark.*--out $L/$n " >/dev/null 2>&1 || pgrep -f -- "python -m exp.run_event_benchmark.*--out $L/$n\$" >/dev/null 2>&1; then echo "- RUNNING  $n: $(grep -E 'epoch' $f | tail -1 | sed 's/.*epoch/epoch/' | cut -c1-70)"
      elif pgrep -f -- "wait_launch.sh [0-9]* $L/$n.log" >/dev/null 2>&1; then echo "- WAITING  $n (launcher pending GPU memory)"
      elif [ -s $f ]; then echo "- STOPPED? $n: $(tail -c 120 $f | tr '\n' ' ' | cut -c1-90)"
      else echo "- PENDING  $n (not started)"; fi
    done
    for cmdf in $L/*.cmd; do [ -f "$cmdf" ] || continue; n=$(basename $cmdf .cmd); [ -f $L/$n.log ] && continue
      if pgrep -f -- "wait_launch.sh [0-9]* $L/$n.log" >/dev/null 2>&1; then echo "- WAITING  $n (launcher pending GPU memory; attempts so far: $(ls $L/$n.fail*.log 2>/dev/null | wc -l))"
      else echo "- QUEUED   $n (.cmd present; daemon will launch)"; fi; done
    echo; echo "## node-property reruns"
    for f in results/leakfree_nodeprop/*.log; do n=$(basename $f .log); [ "$n" = lane ] && continue; echo "- $n: $(grep -E 'mean|seed 4' $f 2>/dev/null | tail -1 | cut -c1-90)"; done
    echo; echo "## synthetic v4"
    for f in results/analytic/synth_v4/runs/*.log; do n=$(basename $f .log)
      if [ -f results/analytic/synth_v4/runs/$n/summary.json ]; then
        echo "- DONE $n: $($PY -c "import json; s=json.load(open('results/analytic/synth_v4/runs/$n/summary.json')); print('test %.3f novel %.3f rec %.3f' % (s['test_mrr'], s['mrr_novel'], s['mrr_recurrent']))" 2>/dev/null)"
      else echo "- $n: $(grep -h epoch $f 2>/dev/null | tail -1 | sed 's/.*epoch/epoch/' | cut -c1-40)"; fi
    done
    echo; echo "## recent actions"; tail -5 $M/actions.log
  } > $M/STATUS.md.tmp 2>/dev/null && mv -f $M/STATUS.md.tmp $M/STATUS.md

  # action (a): icews full evaluation once training finished (once)
  if [ -f $L/icews_f_s43/best.pt ] && [ -f $L/icews_f_s43/results.csv ] && ! grep -q "^icews_eval" $M/retried.txt \
     && ! pgrep -f "eval-only $L/icews_f_s43/best.pt" >/dev/null 2>&1 && ! grep -q "FINAL val_mrr" $L/icews_eval_s43.log 2>/dev/null; then
    echo "icews_eval $ts" >> $M/retried.txt; echo "$ts launching icews full evaluation" >> $M/actions.log
    nohup $WL 16000 $L/icews_eval_s43.log $PY -m exp.run_event_benchmark --dataset tkgl-icews --model faithful --lr 3e-3 --relation-in-input --recurrency-decoder --recurrency-untyped --temporal-d-model 64 --train-edges-cap 2000000 --track-val-edges 30000 --predict-from-previous --eval-only $L/icews_f_s43/best.pt --out $L/icews_eval_s43 > $L/icews_eval_s43.log 2>&1 &
  fi
  # action (b): retry faithful runs that died with CUDA OOM (once per run)
  for f in $L/*.log; do n=$(basename $f .log); case $n in *_o_*|queue*|*_queue|icews_eval*|*.retry) continue;; esac
    [ -f $L/$n/results.csv ] && continue
    grep -q "OutOfMemoryError" $f 2>/dev/null || continue
    grep -q "^$n " $M/retried.txt && continue
    pgrep -f -- "--out $L/$n " >/dev/null 2>&1 && continue
    line=$(grep -h "run [0-9]* $n " $Q/leakfree2_all.sh 2>/dev/null | head -1)
    if [ -z "$line" ]; then echo "$n $ts no-queue-line" >> $M/retried.txt; echo "$ts $n crashed (OOM) - no single-run queue line, not retried" >> $M/actions.log; continue; fi
    echo "$n $ts" >> $M/retried.txt; echo "$ts retrying $n after OOM" >> $M/actions.log
    header=$(sed -n '1,/^# lane 1/p' $Q/leakfree2_all.sh | grep -v '^# lane 1')
    nohup bash -c "$header
mv -f $L/$n.log $L/$n.oom.log 2>/dev/null; $line" > $L/$n.retry.log 2>&1 &
  done
  # action (c): generic bounded retry for any run that has a command file <dir>/<name>.cmd
  # (an executable script that execs wait_launch.sh with the full command). A run is
  # (re)launched when it is neither finished nor alive (no python process, no pending
  # launcher); the crashed log is kept as <name>.failN.log; at most 4 attempts.
  for cmdf in $L/*.cmd results/leakfree_nodeprop/*.cmd; do
    [ -f "$cmdf" ] || continue
    dir=$(dirname $cmdf); n=$(basename $cmdf .cmd); log=$dir/$n.log
    if [ -f $dir/$n/results.csv ] || grep -q "FINAL val_mrr\|test NDCG mean" $log 2>/dev/null; then continue; fi
    if pgrep -f -- "--out $dir/$n\b" >/dev/null 2>&1 || pgrep -f -- "wait_launch.sh [0-9]* $log" >/dev/null 2>&1 || pgrep -f -- "bash $cmdf" >/dev/null 2>&1; then continue; fi
    tries=$(ls $dir/$n.fail*.log 2>/dev/null | wc -l)
    [ $tries -ge 4 ] && continue
    [ -s $log ] && mv -f $log $dir/$n.fail$((tries+1)).log
    echo "$ts (re)launching $n via $cmdf (attempt $((tries+1)))" >> $M/actions.log
    nohup bash $cmdf > /dev/null 2>&1 &
  done
  # stop file / self-reload (so an upgraded script takes effect without killing the daemon)
  [ -f $M/STOP ] && { echo "$(date '+%F %T') stop file found, exiting" >> $M/actions.log; rm -f $M/STOP; exit 0; }
  if [ -n "$SCRIPT_MTIME" ] && [ "$(stat -c %Y "$0")" != "$SCRIPT_MTIME" ]; then echo "$(date '+%F %T') script changed, reloading" >> $M/actions.log; exec bash "$0"; fi
  SCRIPT_MTIME=$(stat -c %Y "$0")
  sleep $INTERVAL
done
