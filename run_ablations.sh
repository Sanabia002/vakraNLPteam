#!/bin/bash
cd ~/vakra && source .venv/bin/activate && set -a && source .env && set +a
export AGENT_TIMEOUT_SECONDS=300
DOMS_CSV=$(ls tmp_gt_tune | sed 's/.json//' | paste -sd, -)
run () {  # name, notes, then VAR=value settings
  name=$1; notes=$2; shift 2
  env "$@" python run_logged.py --out output/$name --cap 3 --domain "$DOMS_CSV" --model openai/qwen3.5-4b \
    --max-iter 12 --temperature 0 --thinking off --build b11960 \
    --uuids-file results/tune_sample_uuids.json --notes "$notes" > $name.log 2>&1
}
run tune68_abl_noverify    "ablation: VERIFY=0"                PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=0 GROUND_PROMPT=1
run tune68_abl_nostepretr  "ablation: STEP_RETRIEVAL=0"        PLAN_AGENT=1 STEP_RETRIEVAL=0 VERIFY=1 GROUND_PROMPT=1
run tune68_abl_noground    "ablation: GROUND_PROMPT=0"         PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=0
run tune68_abl_norecord    "ablation: RECORD_FAILED_RETRIES=0" PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RECORD_FAILED_RETRIES=0
echo done > ablations.done
