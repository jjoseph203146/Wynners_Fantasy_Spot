#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
HB_DIR="$ROOT/data/heartbeat"
OUT="$HB_DIR/r610_heartbeat.txt"
TMP="$HB_DIR/.r610_heartbeat.tmp.$$"

mkdir -p "$HB_DIR"

NOW_EPOCH="$(date +%s)"
NOW_ISO="$(date -Is)"
HOST="$(hostname)"

CRON_ACTIVE="NO"
if systemctl is-active --quiet cron; then
    CRON_ACTIVE="YES"
fi

SINGLE_GAME_CRON="NO"
if crontab -l 2>/dev/null |
   grep -Fq '/home/mwynn/nfl_data_engine/scripts/run_single_game_refresh_if_needed.sh'; then
    SINGLE_GAME_CRON="YES"
fi

FINAL="$ROOT/data/fanduel/single_game/derived/single_game_projection_pool.parquet"

if [ -f "$FINAL" ]; then
    FINAL_MTIME="$(stat -c %Y "$FINAL")"
else
    FINAL_MTIME="0"
fi

{
    echo "version=1"
    echo "host=$HOST"
    echo "epoch=$NOW_EPOCH"
    echo "timestamp=$NOW_ISO"
    echo "cron_active=$CRON_ACTIVE"
    echo "single_game_cron_installed=$SINGLE_GAME_CRON"
    echo "single_game_final_mtime=$FINAL_MTIME"
} > "$TMP"

mv -f "$TMP" "$OUT"

echo "R610_HEARTBEAT_WRITE=PASS"
echo "HEARTBEAT=$OUT"
echo "EPOCH=$NOW_EPOCH"
echo "CRON_ACTIVE=$CRON_ACTIVE"
echo "SINGLE_GAME_CRON_INSTALLED=$SINGLE_GAME_CRON"
