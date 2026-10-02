#!/bin/bash
set -u

ROOT="/home/mwynn/nfl_data_engine"
LOG="$ROOT/logs/coaching_schedule_refresh.log"
LOCK="/tmp/wfs_schedule_aware_coaching_refresh.lock"

cd "$ROOT" || exit 1
source "$ROOT/venv/bin/activate"

mkdir -p "$ROOT/logs"

exec 9>"$LOCK"

if ! flock -n 9; then
    exit 0
fi

env PYTHONPATH=. \
  "$ROOT/venv/bin/python" \
  "$ROOT/scripts/run_schedule_aware_coaching_refresh_v1.py" \
  >> "$LOG" 2>&1

RC=$?

if [ "$RC" -ne 0 ]; then
    echo "FAIL: schedule-aware coaching refresh rc=$RC" >> "$LOG"
    exit "$RC"
fi

"$ROOT/venv/bin/python" \
  "$ROOT/scripts/report_wfs_prediction_accuracy_v1.py" \
  >> "$LOG" 2>&1

RC=$?

if [ "$RC" -ne 0 ]; then
    echo "FAIL: WFS prediction accuracy refresh rc=$RC" >> "$LOG"
    exit "$RC"
fi

echo "PASS: AUTOMATED WFS PREDICTION ACCURACY REFRESH" >> "$LOG"
exit 0
