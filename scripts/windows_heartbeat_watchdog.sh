#!/usr/bin/env bash
set -euo pipefail

HB="/home/mwynn/nfl_data_engine/data/heartbeat/windows_heartbeat.txt"
MAX_AGE=360

NOW="$(date +%s)"
NOW_ISO="$(date -Is)"

if [ ! -s "$HB" ]; then
    echo "[$NOW_ISO] WINDOWS_HEARTBEAT=FAIL reason=MISSING"
    exit 2
fi

EPOCH="$(awk -F= '$1=="epoch"{print $2; exit}' "$HB")"
TASK_STATE="$(awk -F= '$1=="single_game_task_state"{print $2; exit}' "$HB")"
TASK_RESULT="$(awk -F= '$1=="single_game_last_result"{print $2; exit}' "$HB")"

if ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
    echo "[$NOW_ISO] WINDOWS_HEARTBEAT=FAIL reason=INVALID_EPOCH"
    exit 2
fi

AGE=$((NOW - EPOCH))

if [ "$AGE" -lt 0 ] || [ "$AGE" -gt "$MAX_AGE" ]; then
    echo "[$NOW_ISO] WINDOWS_HEARTBEAT=FAIL reason=STALE age_seconds=$AGE"
    exit 2
fi

if [ "$TASK_RESULT" != "0" ]; then
    echo "[$NOW_ISO] WINDOWS_HEARTBEAT=FAIL reason=SINGLE_GAME_TASK_RESULT result=$TASK_RESULT age_seconds=$AGE"
    exit 2
fi

if [ "$TASK_STATE" != "Ready" ] && [ "$TASK_STATE" != "Running" ]; then
    echo "[$NOW_ISO] WINDOWS_HEARTBEAT=FAIL reason=SINGLE_GAME_TASK_STATE state=$TASK_STATE age_seconds=$AGE"
    exit 2
fi

echo "[$NOW_ISO] WINDOWS_HEARTBEAT=PASS age_seconds=$AGE task_state=$TASK_STATE task_result=$TASK_RESULT"
