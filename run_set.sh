#!/bin/bash
# usage: ./run_set.sh dev|test  -> runs D (Groq), C (local plan agent), merges F, then the plain baseline
SET=$1; case "$SET" in dev|test) ;; *) echo "usage: ./run_set.sh dev|test"; exit 1;; esac
exec 9>/tmp/runset.lock; flock -n 9 || { echo "already running"; exit 1; }
cd ~/vakra || exit 1
source .venv/bin/activate && set -a && source .env && set +a
LOCAL_KEY=$LITELLM_API_KEY; GT=tmp_gt_$SET; SAMPLE=results/${SET}_uuids.json
N=$(python3 -c "import json; print(len(json.load(open('$SAMPLE'))))")
export SAMPLE_FILE=$SAMPLE MAX_OTHER_ERR=8 AGENT_TIMEOUT_SECONDS=300
[ "$SET" = test ] && export ALLOW_FROZEN=1
LOG=runset_$SET.out; say () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
hl () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:8080/v1/models -H "Authorization: Bearer $LOCAL_KEY")" = "200" ]; }
hg () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://api.groq.com/openai/v1/models -H "Authorization: Bearer $(cat ~/.groq_pilot_key)")" = "200" ]; }
if env | grep -q -E "^(PLAN_AGENT|STEP_RETRIEVAL|VERIFY|GROUND_PROMPT|RECORD_FAILED_RETRIES|BASELINE_|RETRIEVER|SAVE_TRAJECTORY|LLM_EXTRA_KWARGS)="; then say "ABORT: flag set"; exit 1; fi
git rev-parse -q --verify refs/tags/final-config >/dev/null || { say "ABORT: no final-config tag"; exit 1; }
git diff --quiet final-config HEAD -- agents benchmark_runner.py || { say "ABORT: agents/ or runner changed since the tag"; exit 1; }
[ -n "$(git status --short | grep -v '^??' | grep -v runs_log.csv)" ] && { say "ABORT: uncommitted tracked changes"; exit 1; }
[ -d $GT ] && [ -f $SAMPLE ] || { say "ABORT: run make_eval_sets.py"; exit 1; }
grep -a -q "n_ctx_slot = 32768" ~/llama_server.log || { say "ABORT: local server not at 32768 context"; exit 1; }
hl || { say "ABORT: local server not answering"; exit 1; }; hg || { say "ABORT: Groq pilot key/connection"; exit 1; }
[ "$(docker ps -q | wc -l)" -ge 4 ] || { say "ABORT: fewer than 4 containers"; exit 1; }
HASH=$(git rev-parse --short HEAD); DOMS_CSV=$(ls $GT | sed 's/.json//' | paste -sd, -)
say "start $SET: $N questions, tag final-config, commit $HASH"
run () {   # name backend model topk notes [VAR=value ...]
  name=$1; be=$2; model=$3; topk=$4; notes=$5; shift 5
  [ -e STOP_RUNSET ] && { say "stopped by STOP_RUNSET"; exit 1; }
  if [ "$be" = groq ]; then hg; else hl; fi || { say "ABORT: $be backend down before $name"; exit 1; }
  ARGS=""; [ "$topk" != "0" ] && ARGS="--top-k $topk"
  say "running $name"
  if [ "$be" = groq ]; then
    env LITELLM_BASE_URL=https://api.groq.com/openai/v1 LITELLM_API_KEY="$(cat ~/.groq_pilot_key)" LLM_EXTRA_KWARGS='{"extra_body":{"reasoning_effort":"none"},"max_tokens":1024}' "$@" \
      python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model $model --max-iter 12 --temperature 0 --thinking off --build groq-qwen3.8-27b $ARGS --uuids-file $SAMPLE --notes "$notes; tag final-config; commit $HASH" > $name.log 2>&1
  else
    env "$@" python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model $model --max-iter 12 --temperature 0 --thinking off --build b11960 $ARGS --uuids-file $SAMPLE --notes "$notes; tag final-config; commit $HASH" > $name.log 2>&1
  fi
  if [ "$be" = groq ]; then hg; else hl; fi || { say "ABORT: $be backend down after $name; results invalid"; exit 1; }
  res=$(python3 validate_run.py $name $N); say "$name: $res"
  case "$res" in OK*) ;; *) say "ABORT: validation"; exit 1;; esac
  say "$name: schema errors $(grep -a -c 'is a required property' $name.log), context lines $(grep -a -c 'ContextWindowExceeded' $name.log), 429 lines $(grep -a -c ' 429 \|RateLimit' $name.log)"
}
run final_${SET}_D groq openai/qwen/qwen3.8-27b 30 "FINAL D: ReAct+hybrid+grounded k=30, qwen3.8-27b on Groq" SAVE_TRAJECTORY=1 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1
run final_${SET}_C local openai/qwen3.5-4b 0 "FINAL C: hybrid plan agent (fallback component)" SAVE_TRAJECTORY=1 PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid
python3 merge_fallback.py final_${SET}_D final_${SET}_C final_${SET}_F | tee -a $LOG
res=$(python3 validate_run.py final_${SET}_F $N); say "final_${SET}_F: $res"
case "$res" in OK*) ;; *) say "ABORT: F validation"; exit 1;; esac
run final_${SET}_baseline local openai/qwen3.5-4b 20 "FINAL baseline: provided ReAct agent, k=20, no flags"
say "all done"; echo done > runset_$SET.done
