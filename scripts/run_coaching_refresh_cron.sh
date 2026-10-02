#!/bin/bash
set -u

ROOT="/home/mwynn/nfl_data_engine"
LOG="$ROOT/logs/coaching_intelligence_refresh.log"
LOCK="/tmp/wfs_coaching_intelligence_refresh.lock"

cd "$ROOT" || exit 1
source "$ROOT/venv/bin/activate"

mkdir -p "$ROOT/logs"

exec flock -n "$LOCK" \
  env PYTHONPATH=. \
  python scripts/run_coaching_intelligence_refresh_v1.py \
  >> "$LOG" 2>&1
