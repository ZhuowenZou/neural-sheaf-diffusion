#!/bin/bash
# Usage: wait_launch.sh <min_free_MiB> <logfile> <command...>
#
# Placement policy (2026-09-11/12, agreed with the colleagues sharing the node):
#   * OUR GPUs are TSD_OWN_GPUS (default "0 7"): always eligible, no cap;
#   * GPUs 1-6 belong to the colleagues; a colleague GPU is eligible when EITHER
#       (a) the monitor daemon has seen no other user's process on it for
#           TSD_IDLE_TAKEOVER_SEC (default 3600 s; state in results/monitor/gpu_idle.txt,
#           "<gpu> <epoch since idle>", 0 = occupied now; file older than 10 min => rule off), OR
#       (b) COMPENSATION: the memory colleagues currently occupy on OUR GPUs may be matched
#           by our usage on THEIR GPUs -- budget = sum of other users' memory on 0/7,
#           spent = our processes' memory on 1-6 plus our pending claims there; a job of
#           <min_free_MiB> may go to a colleague GPU while spent + request <= budget;
#   * among eligible cards with >= <min_free_MiB> free (after subtracting memory claimed by
#     our other launchers whose jobs have not allocated yet) the one with the most free
#     memory wins; the decision is serialised with flock;
#   * the job runs with TSD_RESERVE_GPU_MB so it pre-reserves its memory in the caching
#     allocator and cannot be squeezed by a later arrival on the same card.
# Polling interval: TSD_POLL_SEC (default 180 s).
MIN=$1; LOG=$2; shift 2
ROOT=/home/zhuowez1/project/neural-sheaf-diffusion; cd $ROOT
C=$ROOT/results/monitor/claims; mkdir -p $C
IDLE=$ROOT/results/monitor/gpu_idle.txt
OWN=${TSD_OWN_GPUS:-"0 7"}; TAKEOVER=${TSD_IDLE_TAKEOVER_SEC:-3600}; INTERVAL=${TSD_POLL_SEC:-180}; ME=$(id -un)
trap 'rm -f $C/$$' EXIT
sleep $((RANDOM % 20))
while true; do
  DECISION=$( (
    flock -w 90 9 || exit 0
    for f in $C/[0-9]*; do [ -f "$f" ] || continue; kill -0 $(basename $f) 2>/dev/null || rm -f $f; done
    now=$(date +%s)
    declare -A isown=(); for g in $OWN; do isown[$g]=1; done
    declare -A eligible=(); for g in $OWN; do eligible[$g]=own; done
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits > $C/.gpus.$$ 2>/dev/null
    declare -A gidx=(); while IFS=, read -r idx uuid used total; do idx=${idx// /}; uuid=${uuid// /}; gidx[$uuid]=$idx; done < $C/.gpus.$$
    # (a) idle takeover
    if [ -f $IDLE ] && [ $((now - $(stat -c %Y $IDLE))) -le 600 ]; then
      while read -r g since; do
        [ -n "${eligible[$g]}" ] && continue
        if [ "${since:-0}" -gt 0 ] 2>/dev/null && [ $((now - since)) -ge $TAKEOVER ]; then eligible[$g]="idle$(( (now - since) / 60 ))m"; fi
      done < $IDLE
    fi
    # (b) compensation budget
    budget=0; spent=0
    while IFS=, read -r p m u; do p=${p// /}; m=${m// /}; u=${u// /}; g=${gidx[$u]}; [ -n "$g" ] || continue
      if [ "$(ps -o user= -p $p 2>/dev/null)" = "$ME" ]; then [ -z "${isown[$g]}" ] && spent=$((spent + m))
      else [ -n "${isown[$g]}" ] && budget=$((budget + m)); fi
    done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null)
    declare -A claimed=(); for f in $C/[0-9]*; do [ -f "$f" ] || continue; read -r g m < $f; claimed[$g]=$(( ${claimed[$g]:-0} + m )); [ -z "${isown[$g]}" ] && spent=$((spent + m)); done
    remaining=$((budget - spent))
    if [ $remaining -ge $MIN ]; then
      while IFS=, read -r idx uuid used total; do idx=${idx// /}; [ -n "${eligible[$idx]}" ] && continue; eligible[$idx]="comp(${remaining}MiB-left)"; done < $C/.gpus.$$
    fi
    best=""; bestfree=0
    while IFS=, read -r idx uuid used total; do
      idx=${idx// /}; used=${used// /}; total=${total// /}
      [ -n "${eligible[$idx]}" ] || continue
      free=$((total - used - ${claimed[$idx]:-0}))
      [ $free -ge $MIN ] || continue
      [ $free -gt $bestfree ] && { bestfree=$free; best=$idx; }
    done < $C/.gpus.$$
    rm -f $C/.gpus.$$
    why=""; [ -n "$best" ] && { echo "$best $MIN" > $C/$$; why=${eligible[$best]}; }
    echo "$best|$why|$bestfree|budget=${budget} spent=${spent} $(for g in $(printf '%s\n' "${!eligible[@]}" | sort -n); do printf '%s=%s ' $g ${eligible[$g]}; done)"
  ) 9>$C/.lock )
  GPU=${DECISION%%|*}
  if [ -n "$GPU" ]; then
    if [ $MIN -gt 4096 ]; then RES=$((MIN - 1024)); else RES=$((MIN * 3 / 4)); fi
    echo "$(date) launching on gpu$GPU (request ${MIN} MiB, reserve ${RES} MiB; card=$(echo "$DECISION" | cut -d'|' -f2) free=$(echo "$DECISION" | cut -d'|' -f3) $(echo "$DECISION" | cut -d'|' -f4))" >> "$LOG.launch"
    ( sleep 300; rm -f $C/$$ ) &   # the job has reserved its memory by then; stop double-counting
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TSD_RESERVE_GPU_MB=$RES "$@" > "$LOG" 2>&1
    exit $?
  fi
  sleep $INTERVAL
done
