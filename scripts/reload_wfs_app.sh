#!/usr/bin/env bash

# WFS_STREAMLIT_SAFE_RELOAD_V1
#
# Purpose:
#   Safely reload the Wynners Fantasy Spot Streamlit application after
#   validated Python source changes.
#
# Scope:
#   - validates Python source before restart
#   - restarts ONLY wfs.service
#   - verifies systemd health
#   - verifies the expected Streamlit app process
#
# Explicitly does NOT:
#   - run the NFL updater
#   - restart NFL Live
#   - restart the API
#   - modify data/model artifacts
#   - modify projections/forecasts/solver state

set -u

ROOT="/home/mwynn/nfl_data_engine"
PYTHON="${ROOT}/venv/bin/python"
SERVICE="wfs.service"

cd "$ROOT" || {
    echo "WFS_APP_RELOAD=REFUSED"
    echo "REASON=PROJECT_ROOT_UNAVAILABLE"
    return 1 2>/dev/null || true
}

echo "===== WFS SAFE APP RELOAD ====="

if [ ! -x "$PYTHON" ]; then
    echo "WFS_APP_RELOAD=REFUSED"
    echo "REASON=PYTHON_ENVIRONMENT_UNAVAILABLE"
    return 1 2>/dev/null || true
fi

echo
echo "===== PRE-RESTART COMPILE ====="

"$PYTHON" -m py_compile \
    app.py \
    player_outlook_data.py \
    player_outlook_relevance.py \
    player_outlook_capture_reference_v1.py \
    wfs_player_outlook_rollover_v1.py \
    components/player_outlook.py \
    wfs_pages/data_center.py

COMPILE_RC=$?

if [ "$COMPILE_RC" -ne 0 ]; then
    echo
    echo "WFS_APP_RELOAD=REFUSED"
    echo "REASON=PYTHON_COMPILE_FAILED"
    echo "COMPILE_RC=$COMPILE_RC"
    return 1 2>/dev/null || true
fi

echo "COMPILE=PASS"

echo
echo "===== RESTART STREAMLIT ONLY ====="

if ! sudo systemctl restart "$SERVICE"; then
    echo "WFS_APP_RELOAD=FAIL"
    echo "REASON=SYSTEMD_RESTART_FAILED"
    return 1 2>/dev/null || true
fi

sleep 3

if ! systemctl is-active --quiet "$SERVICE"; then
    echo "WFS_APP_RELOAD=FAIL"
    echo "REASON=WFS_SERVICE_NOT_ACTIVE"
    systemctl --no-pager --full status "$SERVICE" | head -25
    return 1 2>/dev/null || true
fi

echo "SERVICE_ACTIVE=YES"

PROCESS_LINE="$(
    ps -ef |
    grep -E '[s]treamlit.*nfl_data_engine/app.py' |
    head -1
)"

if [ -z "$PROCESS_LINE" ]; then
    echo "WFS_APP_RELOAD=FAIL"
    echo "REASON=STREAMLIT_PROCESS_NOT_FOUND"
    return 1 2>/dev/null || true
fi

echo "STREAMLIT_PROCESS=PASS"

echo
echo "===== SERVICE ====="
systemctl --no-pager --full status "$SERVICE" | head -15

echo
echo "===== CONTRACT ====="
echo "WFS_APP_RELOAD=PASS"
echo "SERVICE_RESTARTED=wfs.service"
echo "NFL_LIVE_RESTARTED=NO"
echo "API_RESTARTED=NO"
echo "UPDATER_EXECUTED=NO"
echo "DATA_ARTIFACTS_CHANGED=NO"
