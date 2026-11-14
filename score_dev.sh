#!/bin/bash
exec 9>/tmp/score_dev.lock; flock -n 9 || { echo "another scorer is already running"; exit 1; }
cd ~/vakra && source .venv/bin/activate && set -a && source .env && set +a
export API_KEY=$(cat ~/.groq_key); export JUDGE_BACKEND=groq
DOMS=$(ls tmp_gt_dev | sed 's/.json//' | tr '\n' ' ')
for run in "$@"; do
  out=results/eval_$run.json
  if [ -f "$out" ] && [ "$(python check_eval.py $out | tail -1)" = "VALID" ]; then echo "$run: already scored and VALID"; continue; fi
  rm -f "$out"
  python evaluator/evaluator.py --capability_name capability_multihop_reasoning --gt_root tmp_gt_dev \
    --pred_root output/$run --domains $DOMS --output $out > eval_$run.log 2>&1
  res=$(python check_eval.py $out | tail -3); echo "== $run"; echo "$res"
  if ! echo "$res" | grep -q "^VALID"; then
    echo "judge errors: re-judging only the failed items in 60 s"; sleep 60
    python rejudge_dev.py $run
    [ "$(python check_eval.py results/eval_${run}_merged.json | tail -1)" = "VALID" ] || { echo "STOP: $run still has unjudged items"; exit 1; }
    python check_eval.py results/eval_${run}_merged.json | tail -2
  fi
  sleep 10
done
echo "all scored"
