#!/bin/bash
exec 9>/tmp/pool3.lock; flock -n 9 || { echo "already running"; exit 1; }
cd ~/vakra || exit 1
source .venv/bin/activate && set -a && source .env && set +a
export AGENT_TIMEOUT_SECONDS=300 SAMPLE_FILE=results/tune191_uuids.json MAX_OTHER_ERR=8
LOG=pool3.out
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
health () { [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/models -H "Authorization: Bearer $LITELLM_API_KEY")" = "200" ]; }
if env | grep -q -E "^(PLAN_AGENT|STEP_RETRIEVAL|VERIFY|GROUND_PROMPT|RECORD_FAILED_RETRIES|BASELINE_GROUND_ANSWER|BASELINE_ANSWER_ON_LIMIT|RETRIEVER|SAVE_TRAJECTORY)="; then say "ABORT: flag set in environment"; exit 1; fi
[ -n "$(git status --short | grep -v '^??' | grep -v runs_log.csv)" ] && { say "ABORT: uncommitted tracked changes"; exit 1; }
health || { say "ABORT: model server not answering"; exit 1; }
[ "$(docker ps -q | wc -l)" -ge 4 ] || { say "ABORT: fewer than 4 containers"; exit 1; }
HASH=$(git rev-parse --short HEAD); N=191
DOMS_CSV=$(ls tmp_gt_tune191 | sed 's/.json//' | paste -sd, -)
say "start: commit $HASH, sample $N"
run () {   # name, notes, top-k (0 = none), VAR=value settings
  name=$1; notes=$2; topk=$3; shift 3
  [ -e STOP_POOL3 ] && { say "stopped by STOP_POOL3"; exit 1; }
  health || { say "ABORT: model server down before $name"; exit 1; }
  ARGS=""; [ "$topk" != "0" ] && ARGS="--top-k $topk"
  say "running $name"
  env SAVE_TRAJECTORY=1 "$@" python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model openai/qwen3.5-4b \
    --max-iter 12 --temperature 0 --thinking off --build b11960 $ARGS \
    --uuids-file results/tune191_uuids.json --notes "$notes; commit $HASH; server ctx 32768" > $name.log 2>&1
  health || { say "ABORT: server down after $name; its results are invalid"; exit 1; }
  res=$(python3 validate_run.py $name $N); say "$name: $res"
  case "$res" in OK*) ;; *) say "ABORT"; exit 1;; esac
  say "$name: schema errors $(grep -a -c 'is a required property' $name.log), context-window lines $(grep -a -c 'ContextWindowExceeded' $name.log)"
}
run pool191_hyb_plan             "tune191: hybrid plan agent, all flags"        0  PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid
run pool191_hyb_base_ground_k30  "tune191: ReAct + hybrid + grounded, k=30"     30 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1
run pool191_hyb_base_ground_k20  "tune191: ReAct + hybrid + grounded, k=20"     20 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1
say "all three finished"; echo done > pool3.done
