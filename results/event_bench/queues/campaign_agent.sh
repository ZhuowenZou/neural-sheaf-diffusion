#!/bin/bash
# Campaign agent: unattended operation of the experiment campaign for a fixed lifetime (default 24 h).
# Runs detached (setsid nohup) so it survives closed remote sessions.
#
#   1. MONITOR  - keeps the GPU monitor daemon (gpu_monitor.sh) alive; logs our / others' memory per GPU
#                 every tick to results/monitor/agent_usage.csv (the daemon itself logs totals every 3 min).
#   2. LOAD     - task queue: any <name>.cmd dropped into results/monitor/inbox/ is syntax-checked and moved
#                 into the campaign directory, where the daemon launches it through wait_launch.sh under the
#                 agreed GPU policy (GPUs 0/7 ours; 1-6 only after 1 h idle or within the compensation
#                 budget; memory pre-reserved; up to 4 retries).  No new tasks are accepted in the last
#                 ACCEPT_UNTIL seconds of the lifetime.
#   3. COLLECT  - runs exp/collect_results.py (results_summary.csv + RESULTS_SUMMARY.md).
#   4. COMMIT   - commits and pushes whenever the set of finished runs changed (and at shutdown); never
#                 commits a file larger than 50 MB; push failures are retried on the next tick.
#   5. SHUTDOWN - after LIFETIME: final collect + report (results/monitor/AGENT_REPORT.md) + commit + push,
#                 moves un-launched tasks back to hold/, stops the monitor daemon, exits.  Jobs still running
#                 at that moment are left to finish (their results stay on disk, uncommitted) and are listed.
#   Stop early:  touch results/monitor/AGENT_STOP
ROOT=/home/zhuowez1/project/neural-sheaf-diffusion; cd $ROOT || exit 1
M=results/monitor; L=results/event_bench/leakfree2; Q=results/event_bench/queues; RV=results/review_2026_09_22
BRANCH=$(git branch --show-current)   # pushes go to the CURRENT branch (review campaign 2026-09-22)
PY=/home/zhuowez1/miniconda3/envs/nsd/bin/python
LIFETIME=${AGENT_LIFETIME:-86400}; TICK=${AGENT_TICK:-600}; ACCEPT_UNTIL=${AGENT_ACCEPT_UNTIL:-43200}
mkdir -p $M/inbox $L/hold
LOG=$M/agent.log; START=$(date +%s); END=$((START + LIFETIME))
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20"
if [ -f $M/agent.pid ] && kill -0 "$(cat $M/agent.pid)" 2>/dev/null; then echo "agent already running (pid $(cat $M/agent.pid))"; exit 0; fi
echo $$ > $M/agent.pid; rm -f $M/AGENT_STOP
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "agent started pid $$, lifetime ${LIFETIME}s (until $(date -d @$END '+%F %T')), tick ${TICK}s"

ensure_daemon() {
  if ! ps -eo args | grep -q "^bash $Q/gpu_monitor.sh"; then
    setsid nohup bash $Q/gpu_monitor.sh > $M/daemon.log 2>&1 < /dev/null &
    say "monitor daemon was not running -> relaunched"
  fi
}
log_usage() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader > $M/.ag_gpus.$$ 2>/dev/null
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r p m u; do
    p=${p// /}; m=${m// /}; u=${u// /}; g=$(awk -F', ' -v u="$u" '$2==u{print $1}' $M/.ag_gpus.$$); [ -n "$g" ] || continue
    if [ "$(ps -o user= -p $p 2>/dev/null)" = "$(id -un)" ]; then echo "$g ours $m"; else echo "$g others $m"; fi
  done | awk -v t="$(date '+%F %T')" '{a[$1" "$2]+=$3} END{for(g=0;g<8;g++) printf "%s,%d,%d,%d\n", t, g, a[g" ours"], a[g" others"]}' >> $M/agent_usage.csv
  rm -f $M/.ag_gpus.$$
}
load_inbox() {
  local now=$(date +%s)
  for f in $M/inbox/*.cmd; do [ -f "$f" ] || continue
    n=$(basename $f)
    if [ $((END - now)) -lt $ACCEPT_UNTIL ]; then say "inbox: $n NOT accepted (less than ${ACCEPT_UNTIL}s of lifetime left) -> hold/"; mv -f $f $L/hold/$n; continue; fi
    if bash -n "$f" 2>/dev/null && grep -q "wait_launch.sh" "$f"; then chmod +x $f; mv -f $f $RV/queue/$n; say "inbox: accepted $n -> $RV/queue (the daemon launches it under the GPU policy)"
    else say "inbox: REJECTED $n (syntax error or does not go through wait_launch.sh)"; mv -f $f $f.rejected; fi
  done
}
collect_and_commit() {   # $1 = "force" to commit even when the set of finished runs is unchanged
  $PY -m exp.collect_results >/dev/null 2>&1
  out=$($PY -m exp.review.collect_review 2>&1 | tail -1); fp=$(echo "$out" | grep -o "fingerprint [0-9]*" | awk '{print $2}')
  [ -z "$fp" ] && { say "collect FAILED: $out"; return; }
  last=$(cat $M/.agent_fp 2>/dev/null)
  if [ "$fp" != "$last" ] || [ "$1" = force ]; then
    git add -A >/dev/null 2>&1
    for big in $(git diff --cached --name-only --diff-filter=AM | while read p; do [ -f "$p" ] && [ $(stat -c %s "$p") -gt 52428800 ] && echo "$p"; done); do git reset -q HEAD -- "$big"; say "skipped large file $big"; done
    if ! git diff --cached --quiet; then
      n=$(git diff --cached --name-only | wc -l)
      git commit -q -m "agent: results snapshot $(date '+%F %H:%M') ($out)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && say "committed $n paths ($out)"
    fi
    echo "$fp" > $M/.agent_fp
  fi
  if [ -n "$(git log origin/$BRANCH..HEAD --oneline 2>/dev/null)" ] || ! git rev-parse --verify -q origin/$BRANCH >/dev/null; then
    if timeout 300 git push -q -u origin $BRANCH 2>>$LOG; then say "pushed to origin/$BRANCH ($(git log --oneline -1 | cut -c1-60))"; else say "push FAILED (will retry next tick)"; fi
  fi
}
our_jobs() { ps -eo pid,user,args | awk -v u="$(id -un)" '$2==u' | grep -E "python -m exp\.(run_event_benchmark|run_faithful_trade|run_synthetic_sheaf|analyze_event_model)" | grep -v grep | grep -o "\-\-out [^ ]*" | sed 's#--out ##; s#.*/##' | sort -u | tr '\n' ' '; }

[ -f $M/.agent_fp ] || { $PY -m exp.collect_results 2>&1 | tail -1 | grep -o "fingerprint [0-9]*" | awk '{print $2}' > $M/.agent_fp; }
while true; do
  now=$(date +%s)
  [ -f $M/AGENT_STOP ] && { say "AGENT_STOP found -> shutting down early"; break; }
  [ $now -ge $END ] && { say "lifetime reached -> shutting down"; break; }
  ensure_daemon; load_inbox; log_usage; collect_and_commit
  say "tick: running [$(our_jobs)] pending-launchers $(ps -eo args | grep -c '[w]ait_launch.sh [0-9]') free-GiB $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "%.0f ", ($2-$1)/1024}')"
  sleep $TICK
done

# ---- shutdown ----
for f in $M/inbox/*.cmd; do [ -f "$f" ] && mv -f $f $L/hold/; done
for f in $RV/queue/*.cmd; do [ -f "$f" ] || continue; n=$(basename $f .cmd)   # tasks never launched go back to hold/
  if [ ! -f $RV/queue/$n.log.launch ]; then
    lp=$(ps -eo pid,args | grep "[w]ait_launch.sh [0-9]* $RV/queue/$n.log" | awk '{print $1}'); [ -n "$lp" ] && kill $lp 2>/dev/null
    mv -f $f $RV/hold/; say "shutdown: un-launched task $n moved to $RV/hold/"; fi
done
still=$(our_jobs)
{
  echo "# Campaign agent report"; echo; echo "window: $(date -d @$START '+%F %T') -> $(date '+%F %T')"; echo
  echo "## Runs finished during the window"; $PY - <<PYEOF
import pandas as pd, time
d = pd.read_csv("results/review_2026_09_22/per_seed_results.csv"); start = time.strftime("%Y-%m-%d %H:%M", time.localtime($START))
d = d[d.finished >= start]
print(d[["run", "dataset", "arm", "seed", "lr", "validation_mrr", "test_mrr", "query_audit_parity", "finished"]].to_string(index=False) if len(d) else "(none)")
PYEOF
  echo; echo "## Still running at shutdown (left to finish; results NOT committed by the agent)"; echo "${still:-none}"
  echo; echo "## GPU usage during the window (mean MiB per GPU: ours / others)"; awk -F, -v s="$(date -d @$START '+%F %T')" '$1>=s{o[$2]+=$3; x[$2]+=$4; n[$2]++} END{for(g=0;g<8;g++) if(n[g]) printf "gpu%d: %.0f / %.0f\n", g, o[g]/n[g], x[g]/n[g]}' $M/agent_usage.csv
  echo; echo "## Agent log (last 40 lines)"; tail -40 $LOG
} > $M/AGENT_REPORT.md 2>&1
collect_and_commit force
# stop the monitor daemon (explicit PID; it is our own process)
dp=$(ps -eo pid,args | awk '$2=="bash" && $3=="'$Q'/gpu_monitor.sh"{print $1}'); [ -n "$dp" ] && kill $dp 2>/dev/null && say "monitor daemon (pid $dp) stopped"
say "agent exiting"; rm -f $M/agent.pid
git add -A >/dev/null 2>&1; git diff --cached --quiet || { git commit -q -m "agent: final log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"; timeout 300 git push -q -u origin $BRANCH 2>>$LOG; }
