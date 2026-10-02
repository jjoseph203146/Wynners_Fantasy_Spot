#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
cd "$ROOT"

EXPECTED_UPDATER_SHA="46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a"
EXPECTED_FEEDBACK_SHA="02320fca4a6db2592329c77152f2c1abb6212d4eb46e5e26de391ed5d2ca0049"
EXPECTED_PROJECTION_SHA="2ab7b98663e16d0a5747e244173b7b407c0bfb9975868000f7cba6627738439d"

echo "=== NFL-2026-FEEDBACK-1R — PRE-INSTALL AUDIT ==="
echo

echo "=== FILE HASHES ==="

ACTUAL_UPDATER_SHA="$(sha256sum run_updater.sh | awk '{print $1}')"
ACTUAL_FEEDBACK_SHA="$(sha256sum nfl_2026_feedback_refresh.py | awk '{print $1}')"
ACTUAL_PROJECTION_SHA="$(sha256sum production_projection.py | awk '{print $1}')"

echo "RUN_UPDATER_SHA256=$ACTUAL_UPDATER_SHA"
echo "FEEDBACK_ORCHESTRATOR_SHA256=$ACTUAL_FEEDBACK_SHA"
echo "PRODUCTION_PROJECTION_SHA256=$ACTUAL_PROJECTION_SHA"

if [ "$ACTUAL_UPDATER_SHA" != "$EXPECTED_UPDATER_SHA" ]; then
    echo "FAIL_CLOSED: run_updater.sh baseline mismatch"
    exit 1
fi

if [ "$ACTUAL_FEEDBACK_SHA" != "$EXPECTED_FEEDBACK_SHA" ]; then
    echo "FAIL_CLOSED: feedback orchestrator baseline mismatch"
    exit 1
fi

if [ "$ACTUAL_PROJECTION_SHA" != "$EXPECTED_PROJECTION_SHA" ]; then
    echo "FAIL_CLOSED: production_projection.py baseline mismatch"
    exit 1
fi

echo
echo "HASH_GATE=PASS"

echo
echo "=== CURRENT UPDATER ==="
nl -ba run_updater.sh

echo
echo "=== LOCK CONTRACT ==="
grep -nE \
'nfl_updater\.lock|flock|exec 9|FD9|python|injury|updater\.py' \
run_updater.sh || true

echo
echo "=== FEEDBACK ORCHESTRATOR LOCK CHECK ==="

if grep -nE 'flock|nfl_updater\.lock' nfl_2026_feedback_refresh.py; then
    echo
    echo "FAIL_CLOSED: feedback orchestrator contains its own updater lock"
    exit 1
fi

echo "NESTED_LOCK_FOUND=FALSE"

echo
echo "=== CRON CONTRACT ==="
crontab -l | grep -F '/home/mwynn/nfl_data_engine/run_updater.sh'

COUNT="$(
    crontab -l |
    grep -Fc '/home/mwynn/nfl_data_engine/run_updater.sh'
)"

echo "UPDATER_CRON_COUNT=$COUNT"

if [ "$COUNT" -ne 1 ]; then
    echo "FAIL_CLOSED: expected exactly one updater cron entry"
    exit 1
fi

echo
echo "=== FEEDBACK SCRIPT COMPILE ==="
./venv/bin/python -m py_compile nfl_2026_feedback_refresh.py

echo "FEEDBACK_COMPILE=PASS"

echo
echo "=== SHELL SYNTAX ==="
bash -n run_updater.sh

echo "RUN_UPDATER_SHELL_SYNTAX=PASS"

echo
echo "NFL_2026_FEEDBACK_1R_READ_ONLY=TRUE"
echo "NFL_2026_FEEDBACK_1R_STATUS=PASS"
