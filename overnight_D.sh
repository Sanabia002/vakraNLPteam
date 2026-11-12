#!/bin/bash
exec 9>/tmp/overnightD.lock; flock -n 9 || { echo "already running"; exit 1; }
cd ~/vakra || exit 1
source .venv/bin/activate && set -a && source .env && set +a
export SAMPLE_FILE=results/tune191_uuids.json MAX_OTHER_ERR=8
LOG=overnightD.out; say () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "start D pool run"
./run_groq27.sh pool191_groq27 results/tune191_uuids.json 191 >> $LOG 2>&1
res=$(python3 validate_run.py pool191_groq27 191); say "pool191_groq27: $res"
case "$res" in OK*) ;; *) say "ABORT: D failed validation"; exit 1;; esac
./score_pool.sh pool191_groq27 2>&1 | tee -a $LOG
python3 merge_fallback.py pool191_groq27 pool191_hyb_plan pool191_F | tee -a $LOG
res=$(python3 validate_run.py pool191_F 191); say "pool191_F: $res"
case "$res" in OK*) ;; *) say "ABORT: F failed validation"; exit 1;; esac
./score_pool.sh pool191_F 2>&1 | tee -a $LOG
say "all done"; echo done > overnightD.done
