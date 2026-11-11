#!/bin/bash
# usage: ./run_groq27.sh NAME SAMPLE_FILE N
exec 9>/tmp/groq27.lock; flock -n 9 || { echo "already running"; exit 1; }
NAME=$1; SAMPLE=$2; N=$3
cd ~/vakra || exit 1
[ -f ~/.groq_pilot_key ] || { echo "ABORT: ~/.groq_pilot_key missing"; exit 1; }
source .venv/bin/activate && set -a && source .env && set +a
export LITELLM_BASE_URL=https://api.groq.com/openai/v1 LITELLM_API_KEY=$(cat ~/.groq_pilot_key)
export LLM_EXTRA_KWARGS='{"reasoning_effort":"none","max_tokens":1024}'
export AGENT_TIMEOUT_SECONDS=300 SAMPLE_FILE=$SAMPLE MAX_OTHER_ERR=8
if env | grep -q -E "^(PLAN_AGENT|STEP_RETRIEVAL|VERIFY|GROUND_PROMPT|BASELINE_|RETRIEVER|SAVE_TRAJECTORY)="; then echo "ABORT: flag set"; exit 1; fi
[ -n "$(git status --short | grep -v '^??' | grep -v runs_log.csv)" ] && { echo "ABORT: uncommitted tracked changes"; exit 1; }
[ "$(curl -s -o /dev/null -w '%{http_code}' https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LITELLM_API_KEY")" = "200" ] || { echo "ABORT: Groq key/connection"; exit 1; }
[ "$(docker ps -q | wc -l)" -ge 4 ] || { echo "ABORT: fewer than 4 containers"; exit 1; }
HASH=$(git rev-parse --short HEAD); DOMS_CSV=$(ls tmp_gt_tune191 | sed 's/.json//' | paste -sd, -)
env SAVE_TRAJECTORY=1 RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1 python run_logged.py --out output/$NAME --cap 3 --domain "$DOMS_CSV" \
  --model openai/qwen/qwen3.8-27b --top-k 30 --max-iter 12 --temperature 0 --thinking off --build groq-qwen3.8-27b \
  --uuids-file $SAMPLE --notes "D: ReAct+hybrid+grounded k=30, qwen3.8-27b on Groq, extra kwargs set; commit $HASH" > $NAME.log 2>&1
python3 validate_run.py $NAME $N
echo "429 lines: $(grep -a -c ' 429 \|RateLimit' $NAME.log) | schema errors: $(grep -a -c 'is a required property' $NAME.log) | context lines: $(grep -a -c 'ContextWindowExceeded' $NAME.log)"
