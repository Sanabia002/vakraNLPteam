#!/usr/bin/env bash
# Run, judge and score configurations one after another on the GPU machine, from the repository root.
# Usage: bash subtask4/deploy/queue.sh LOG CONFIG:SPLIT [CONFIG:SPLIT ...]
#   e.g. bash subtask4/deploy/queue.sh queue_system.log system_parity:tune system:tune system_parity:dev system:dev
# Each run needs a clean commit (run.py refuses otherwise). Already finished runs are skipped, so a stopped queue can
# be started again with the same arguments.
set -uo pipefail
LOG="$1"; shift
MCP=subtask4/deploy/mcp_native.yaml
PY=.venv/bin/python
stamp() { date -u +%H:%M; }
for item in "$@"; do
  config="subtask4/configs/${item%%:*}.yaml"; split="${item##*:}"
  run_id=$($PY -c "from subtask4.config import RunConfig; print(RunConfig.load('$config', split='$split').run_id)")
  out="output/subtask4/$run_id"
  if [ ! -f "$out/score_local.json" ]; then
    resume=""; [ -d "$out" ] && resume="--resume"
    echo "[$(stamp)] run $run_id" >> "$LOG"
    $PY -m subtask4 run "$config" --split "$split" --mcp-config "$MCP" $resume >> "$out.run.out" 2>&1 \
      || { echo "[$(stamp)] run FAILED $run_id (see $out.run.out)" >> "$LOG"; continue; }
    echo "[$(stamp)] eval $run_id" >> "$LOG"
    $PY -m subtask4 eval "$run_id" --judge local --mcp-config "$MCP" >> "$out.eval.out" 2>&1 \
      || { echo "[$(stamp)] eval FAILED $run_id" >> "$LOG"; continue; }
    $PY -m subtask4 score "$run_id" >> "$LOG" 2>&1
  else
    echo "[$(stamp)] done already: $run_id" >> "$LOG"
  fi
done
echo "[$(stamp)] QUEUE DONE" >> "$LOG"
