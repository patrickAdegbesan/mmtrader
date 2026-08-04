#!/bin/bash
# Continuous learning loop. Runs until stopped, not for a fixed number of
# rounds -- each pass warm-starts from the current production version,
# retrains, evaluates old against new on the same held-out slice, and
# promotes only on improvement. A round that learns nothing changes nothing.
#
# This is the same loop docker-compose's `retrainer` service and
# deploy/cognition-retrain.timer run; this script is the bare form for
# running it in a shell.
#
#   scripts/learn_forever.sh [data.parquet] [episodes_per_round]
#
# Progress goes to logs/learn_forever.log; every promotion is recorded in
# models/<agent>/ as a new version, so the history is auditable after.

set -u
cd "$(dirname "$0")/.."
DATA="${1:-data/processed/BTC_USDT_1m.parquet}"
EPISODES="${2:-60}"
LOG="logs/learn_forever.log"
mkdir -p logs

round=0
while true; do
  round=$((round + 1))
  echo "=== round $round started $(date -u +%FT%TZ) ===" | tee -a "$LOG"

  python3 scripts/retrain.py --data "$DATA" --episodes "$EPISODES" --seed "$round" \
    2>&1 | tee -a "$LOG" | grep -E "promoted|PROMOTE|KEEP|eval total R|Traceback|Error"

  echo "=== round $round finished $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done
