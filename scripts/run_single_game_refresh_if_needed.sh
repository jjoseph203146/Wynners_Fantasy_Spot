#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
RAW_DIR="$ROOT/data/fanduel/single_game/raw"
FINAL="$ROOT/data/fanduel/single_game/derived/single_game_projection_pool.parquet"
WRAPPER="$ROOT/scripts/run_single_game_derived_refresh.sh"
LOCK="$ROOT/data/fanduel/single_game/.derived_refresh.lock"

exec 9>"$LOCK"

if ! flock -n 9; then
    echo "[$(date -Is)] SINGLE_GAME_REFRESH=SKIP_ALREADY_RUNNING"
    exit 0
fi

if [ ! -d "$RAW_DIR" ]; then
    echo "[$(date -Is)] SINGLE_GAME_REFRESH=FAIL_RAW_DIR_MISSING"
    exit 2
fi

NEWEST_RAW="$(
    find "$RAW_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*.csv' \
        -printf '%T@\n' \
        2>/dev/null \
        | sort -nr \
        | head -n 1
)"

if [ -z "$NEWEST_RAW" ]; then
    echo "[$(date -Is)] SINGLE_GAME_REFRESH=SKIP_NO_RAW_FILES"
    exit 0
fi

if [ -f "$FINAL" ]; then
    FINAL_MTIME="$(stat -c %Y "$FINAL")"
else
    FINAL_MTIME="0"
fi

NEWEST_RAW_INT="${NEWEST_RAW%%.*}"

echo "[$(date -Is)] RAW_MTIME=$NEWEST_RAW_INT FINAL_MTIME=$FINAL_MTIME"

if [ "$NEWEST_RAW_INT" -le "$FINAL_MTIME" ]; then
    echo "[$(date -Is)] SINGLE_GAME_REFRESH=NO_CHANGE"
    exit 0
fi

echo "[$(date -Is)] SINGLE_GAME_REFRESH=START"

"$WRAPPER"

echo "[$(date -Is)] SINGLE_GAME_REFRESH=PASS"
