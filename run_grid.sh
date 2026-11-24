#!/bin/bash
# usage: ./run_grid.sh test  -> supplementary uniform model-by-agent grid (results/freeze_rule.txt, "Supplementary uniform grid").
# Stages, once each: D4 (local 4B), C27 (Groq 27B), baseline27 (Groq 27B); then merges E = D4+C and F27 = D+C27 (no model calls).
# Copied from run_set.sh: same pre-flight checks, run() body, flags, k, timeout, iterations and LLM settings. Never touches final_test_{D,C,baseline,F}.
SET=$1; case "$SET" in test) ;; *) echo "usage: ./run_grid.sh test"; exit 1;; esac
exec 9>/tmp/runset.lock; flock -n 9 || { echo "already running (run_set.sh or run_grid.sh holds the lock)"; exit 1; }
cd ~/vakra || exit 1
source .venv/bin/activate && set -a && source .env && set +a
LOCAL_KEY=$LITELLM_API_KEY; GT=tmp_gt_$SET; SAMPLE=results/${SET}_uuids.json
N=$(python3 -c "import json; print(len(json.load(open('$SAMPLE'))))")
export SAMPLE_FILE=$SAMPLE MAX_OTHER_ERR=${MAX_OTHER_ERR:-8} AGENT_TIMEOUT_SECONDS=300
[ "$SET" = test ] && export ALLOW_FROZEN=1
LOG=rungrid_$SET.out; say () { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a $LOG; }
hl () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:8080/v1/models -H "Authorization: Bearer $LOCAL_KEY")" = "200" ]; }
hg () { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://api.groq.com/openai/v1/models -H "Authorization: Bearer $(cat ~/.groq_pilot_key)")" = "200" ]; }
if env | grep -q -E "^(PLAN_AGENT|STEP_RETRIEVAL|VERIFY|GROUND_PROMPT|RECORD_FAILED_RETRIES|BASELINE_|RETRIEVER|SAVE_TRAJECTORY|LLM_EXTRA_KWARGS)="; then say "ABORT: flag set"; exit 1; fi
git rev-parse -q --verify refs/tags/final-config >/dev/null || { say "ABORT: no final-config tag"; exit 1; }
git diff --quiet final-config HEAD -- agents benchmark_runner.py || { say "ABORT: agents/ or runner changed since the tag"; exit 1; }
git diff --quiet final-config -- agents benchmark_runner.py || { say "ABORT: agents/ or runner differ from the tag in the working tree"; exit 1; }
[ -n "$(git status --short | grep -v '^??' | grep -v runs_log.csv)" ] && { say "ABORT: uncommitted tracked changes"; exit 1; }
grep -q "^Supplementary uniform grid" results/freeze_rule.txt && git log --format=%H -1 -- results/freeze_rule.txt | grep -q . \
  && git diff --quiet HEAD -- results/freeze_rule.txt || { say "ABORT: grid protocol note not committed"; exit 1; }
[ -d $GT ] && [ -f $SAMPLE ] || { say "ABORT: run make_eval_sets.py"; exit 1; }
grep -a -q "n_ctx_slot = 32768" ~/llama_server.log || { say "ABORT: local server not at 32768 context"; exit 1; }
hl || { say "ABORT: local server not answering"; exit 1; }; hg || { say "ABORT: Groq pilot key/connection"; exit 1; }
[ "$(docker ps -q | wc -l)" -ge 4 ] || { say "ABORT: fewer than 4 containers"; exit 1; }
for r in D C; do python3 validate_run.py final_${SET}_$r $N | grep -q '^OK' || { say "ABORT: existing final_${SET}_$r not valid (needed for merges)"; exit 1; }; done
for r in E F27; do [ -e output/final_${SET}_$r ] || [ -e results/merge_log_final_${SET}_$r.json ] && { say "ABORT: final_${SET}_$r already exists; not overwriting"; exit 1; }; done
HASH=$(git rev-parse --short HEAD); DOMS_CSV=$(ls $GT | sed 's/.json//' | paste -sd, -)
say "start grid $SET: $N questions, tag final-config, commit $HASH"
run () {   # name backend model topk notes [VAR=value ...]   (body identical to run_set.sh, plus timing)
  name=$1; be=$2; model=$3; topk=$4; notes=$5; shift 5
  if [ -d output/$name ]; then
    if python3 validate_run.py $name $N | grep -q '^OK'; then say "skip $name: already valid"; return 0; fi
    say "ABORT: output/$name exists but is not valid; inspect the error, do not rerun automatically"; exit 1
  fi
  [ -e STOP_RUNSET ] && { say "stopped by STOP_RUNSET"; exit 1; }
  if [ "$be" = groq ]; then hg; else hl; fi || { say "ABORT: $be backend down before $name"; exit 1; }
  ARGS=""; [ "$topk" != "0" ] && ARGS="--top-k $topk"
  say "running $name"; t0=$(date +%s)
  if [ "$be" = groq ]; then
    env LITELLM_BASE_URL=https://api.groq.com/openai/v1 LITELLM_API_KEY="$(cat ~/.groq_pilot_key)" LLM_EXTRA_KWARGS='{"extra_body":{"reasoning_effort":"none"},"max_tokens":1024}' "$@" \
      python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model $model --max-iter 12 --temperature 0 --thinking off --build groq-qwen3.8-27b $ARGS --uuids-file $SAMPLE --notes "$notes; tag final-config; commit $HASH" > $name.log 2>&1
  else
    env "$@" python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model $model --max-iter 12 --temperature 0 --thinking off --build b11960 $ARGS --uuids-file $SAMPLE --notes "$notes; tag final-config; commit $HASH" > $name.log 2>&1
  fi
  say "$name: finished in $(( ($(date +%s) - t0) / 60 )) min"
  if [ "$be" = groq ]; then hg; else hl; fi || { say "ABORT: $be backend down after $name; results invalid"; exit 1; }
  res=$(python3 validate_run.py $name $N); say "$name: $res"
  case "$res" in OK*) ;; *) say "ABORT: validation"; exit 1;; esac
  say "$name: schema errors $(grep -a -c 'is a required property' $name.log), context lines $(grep -a -c 'ContextWindowExceeded' $name.log), 429 lines $(grep -a -c ' 429 \|RateLimit' $name.log)"
}
# Stage order fixed by the task. Flags copied from run_set.sh's D, C and baseline lines; only backend/model swapped.
run final_${SET}_D4 local openai/qwen3.5-4b 30 "GRID D4: ReAct+hybrid+grounded k=30, local 4B (supplementary)" SAVE_TRAJECTORY=1 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1
run final_${SET}_C27 groq openai/qwen/qwen3.8-27b 0 "GRID C27: hybrid plan agent, qwen3.8-27b on Groq (supplementary)" SAVE_TRAJECTORY=1 PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid
run final_${SET}_baseline27 groq openai/qwen/qwen3.8-27b 20 "GRID baseline27: provided ReAct agent, k=20, no flags, qwen3.8-27b on Groq (supplementary)"
for m in "E final_${SET}_D4 final_${SET}_C" "F27 final_${SET}_D final_${SET}_C27"; do
  set -- $m; python3 merge_fallback.py $2 $3 final_${SET}_$1 | tee -a $LOG
  res=$(python3 validate_run.py final_${SET}_$1 $N); say "final_${SET}_$1: $res"
  case "$res" in OK*) ;; *) say "ABORT: $1 validation"; exit 1;; esac
done
say "all done"; echo done > rungrid_$SET.done
