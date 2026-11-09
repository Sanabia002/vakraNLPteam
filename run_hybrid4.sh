#!/bin/bash
# Four end-to-end runs on the tune sample, one at a time. Stops at the first problem.
cd ~/vakra || exit 1
source .venv/bin/activate && set -a && source .env && set +a
export AGENT_TIMEOUT_SECONDS=300
EXTRA_ENV="SAVE_TRAJECTORY=1"     # e.g. EXTRA_ENV="SAVE_TRAJECTORY=1" once you know the flag name
LOG=hyb4.out
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
health () { [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/v1/models -H "Authorization: Bearer $LITELLM_API_KEY")" = "200" ]; }

# ---- pre-flight ----
if env | grep -q -E "^(PLAN_AGENT|STEP_RETRIEVAL|VERIFY|GROUND_PROMPT|RECORD_FAILED_RETRIES|BASELINE_GROUND_ANSWER|RETRIEVER)="; then say "ABORT: a flag is set in the environment"; exit 1; fi
if [ -n "$(git status --short | grep -v '^??' | grep -v runs_log.csv)" ]; then say "ABORT: uncommitted changes to tracked files"; exit 1; fi
[ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/v1/models -H "Authorization: Bearer $LITELLM_API_KEY")" = "200" ] || { say "ABORT: model server not answering"; exit 1; }
[ "$(docker ps -q | wc -l)" -ge 4 ] || { say "ABORT: fewer than 4 containers running"; exit 1; }
HASH=$(git rev-parse --short HEAD)
DOMS_CSV=$(ls tmp_gt_tune | sed 's/.json//' | paste -sd, -)
N=$(python3 -c "import json; print(len(json.load(open('results/tune_sample_uuids.json'))))")
say "start: commit $HASH, sample $N, domains $(echo $DOMS_CSV | tr ',' '\n' | wc -l)"

check () {
  python3 validate_run.py "$1" "$N"
}

run () {   # name, notes, top-k (0 = none), then VAR=value settings
  name=$1; notes=$2; topk=$3; shift 3
  [ -e STOP_HYB4 ] && { say "stopped by STOP_HYB4"; exit 1; }
  ARGS=""; [ "$topk" != "0" ] && ARGS="--top-k $topk"
  health || { say "ABORT: model server down before $name"; exit 1; }
  say "running $name"
  env $EXTRA_ENV "$@" python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model openai/qwen3.5-4b \
    --max-iter 12 --temperature 0 --thinking off --build b11960 $ARGS \
    --uuids-file results/tune_sample_uuids.json --notes "$notes; commit $HASH" > $name.log 2>&1
  health || { say "ABORT: model server down after $name; its results are invalid"; exit 1; }
  res=$(check $name); say "$name: $res"
  case "$res" in OK*) ;; *) say "ABORT"; exit 1;; esac
  s=$(grep -a -c 'is a required property' $name.log); c=$(grep -a -c 'ContextWindowExceeded' $name.log)
  say "$name: schema errors $s, context errors $c"
  [ "$s" -eq 0 ] && [ "$c" -eq 0 ] || { say "ABORT: schema/context errors"; exit 1; }
}

run tune68_hyb_plan        "full agent + RETRIEVER=hybrid"                       0  PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid
run tune68_hyb_plan_nostep "full agent + hybrid, STEP_RETRIEVAL=0"               0  PLAN_AGENT=1 STEP_RETRIEVAL=0 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid
run tune68_hyb_base        "ReAct k=20 + RETRIEVER=hybrid"                       20 RETRIEVER=hybrid
run tune68_hyb_base_ground "ReAct k=20 + hybrid + BASELINE_GROUND_ANSWER"        20 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1
say "all four finished"; echo done > hyb4.done
