#!/bin/bash
# usage: ./score_grid.sh RUN...  -> like score_test.sh, but writes results/eval_gridtest_<RUN>.json (never eval_final_*).
# Shares score_test.sh's lock, so only one folder is scored at a time.
exec 9>/tmp/score_test.lock; flock -n 9 || { echo "another scorer is already running"; exit 1; }
cd ~/vakra && source .venv/bin/activate && set -a && source .env && set +a
export API_KEY=$(cat ~/.groq_key); export JUDGE_BACKEND=groq
DOMS=$(ls tmp_gt_test | sed 's/.json//' | tr '\n' ' ')
for run in "$@"; do
  case "$run" in final_test_D4|final_test_C27|final_test_baseline27|final_test_E|final_test_F27) ;; *) echo "STOP: $run is not a grid run"; exit 1;; esac
  out=results/eval_gridtest_$run.json
  if [ -f "$out" ] && [ "$(python check_eval.py $out | tail -1)" = "VALID" ]; then echo "$run: already scored and VALID"; continue; fi
  [ -f results/eval_gridtest_${run}_merged.json ] && { echo "STOP: $run already has a merged re-judge file; not rescoring"; exit 1; }
  rm -f "$out"
  python evaluator/evaluator.py --capability_name capability_multihop_reasoning --gt_root tmp_gt_test \
    --pred_root output/$run --domains $DOMS --output $out > eval_gridtest_$run.log 2>&1
  res=$(python check_eval.py $out | tail -3); echo "== $run"; echo "$res"
  if ! echo "$res" | grep -q "^VALID"; then
    echo "judge errors: re-judging only the failed items in 60 s"; sleep 60
    python rejudge_grid.py $run
    [ "$(python check_eval.py results/eval_gridtest_${run}_merged.json | tail -1)" = "VALID" ] || { echo "STOP: $run still has unjudged items"; exit 1; }
    python check_eval.py results/eval_gridtest_${run}_merged.json | tail -2
  fi
  sleep 10
done
echo "all scored"
