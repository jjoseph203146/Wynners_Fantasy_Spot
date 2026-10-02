#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
TARGET="$ROOT/run_updater.sh"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_OLD_RUN_UPDATER_SHA="6315c6c700d04a537cb0751c7f015f6b59fe9b77da5763c6a05d23767cdd2d4c"
EXPECTED_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_SOLVER_READY_SHA="35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"
EXPECTED_NEW_RUN_UPDATER_SHA="fc60e5bc6d1ebe61251923e7810df1d5cfada5ed24c867838a56c30e94cf407b"

cd "$ROOT"
mkdir -p "$BACKUP_DIR"

require_hash() {
    local file="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    printf '%s  %s\n' "$actual" "$file"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL | hash mismatch: $file"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 1
    fi
}

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-21D"
echo "PRODUCTION AUTOMATIC INJURY PIPELINE DEPLOYMENT"
echo "======================================================================"

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$TARGET" "$EXPECTED_OLD_RUN_UPDATER_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "=== CRON BEFORE ==="
CRON_LINE="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_LINE"
CRON_COUNT="$(printf '%s\n' "$CRON_LINE" | sed '/^$/d' | wc -l)"
if [[ "$CRON_COUNT" -ne 1 ]]; then
    echo "FAIL | expected exactly one run_updater.sh cron entry; found $CRON_COUNT"
    exit 1
fi

if flock -n "$ROOT/nfl_updater.lock" -c true; then
    echo "UPDATER_LOCK_PREFLIGHT=FREE"
else
    echo "FAIL | updater lock is currently busy; retry after current updater finishes"
    exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$BACKUP_DIR/run_updater_before_injury21d_${TS}.sh"
CANDIDATE="$ROOT/.run_updater_injury21d.$$"

cp -a "$TARGET" "$BACKUP"
echo "BACKUP=$BACKUP"

cat > "$CANDIDATE" <<'RUN_UPDATER_EOF'
#!/bin/bash

set -u

PROJECT_DIR="/home/mwynn/nfl_data_engine"
PYTHON="$PROJECT_DIR/venv/bin/python"
UPDATER="$PROJECT_DIR/updater.py"
INJURY_INGEST="$PROJECT_DIR/fanduel_injury_ingest.py"
INJURY_CONSENSUS="$PROJECT_DIR/injury_consensus.py"
SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
LOCK_FILE="$PROJECT_DIR/nfl_updater.lock"
CRON_LOG="$PROJECT_DIR/logs/cron.log"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$PROJECT_DIR/logs"

log() {
    echo "$*" >> "$CRON_LOG"
}

wait_for_live_quiet() {
    local stage="$1"
    local waited=0

    log "Waiting for LIVE quiet window before $stage..."

    while pgrep -f '[l]ive_concurrency_gate.py' >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))

        if [ "$waited" -ge 120 ]; then
            log "LIVE quiet-window timeout before $stage."
            return 1
        fi
    done

    log "LIVE quiet window acquired for $stage after ${waited}s."
    return 0
}

run_stage() {
    local name="$1"
    local script="$2"
    local failure_code="$3"

    log "------------------------------------------------------------"
    log "Starting $name: $(date)"

    "$PYTHON" "$script" >> "$CRON_LOG" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        log "$name FAILED with exit code $rc."
        return "$failure_code"
    fi

    log "$name completed successfully: $(date)"
    return 0
}

run_pipeline() {
    log "------------------------------------------------------------"
    log "Starting core updater: $(date)"

    "$PYTHON" "$UPDATER" >> "$CRON_LOG" 2>&1
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        log "Core updater FAILED with exit code $rc."
        return 10
    fi

    log "Core updater completed successfully: $(date)"

    if ! wait_for_live_quiet "FanDuel injury ingest"; then
        return 31
    fi
    if ! run_stage "FanDuel injury ingest" "$INJURY_INGEST" 21; then
        return $?
    fi

    if ! wait_for_live_quiet "injury consensus"; then
        return 32
    fi
    if ! run_stage "Injury consensus" "$INJURY_CONSENSUS" 22; then
        return $?
    fi

    if ! wait_for_live_quiet "solver-ready injury rebuild"; then
        return 33
    fi
    if ! run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23; then
        return $?
    fi

    return 0
}

log "============================================================"
log "NFL hourly pipeline launcher started: $(date)"

# Hold the existing updater lock for the ENTIRE core + injury chain.
exec 9>"$LOCK_FILE"

if ! flock -n 9; then
    log "Updater skipped because another instance is running."
    log "Finished: $(date)"
    log ""
    exit 1
fi

run_pipeline
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    log "NFL hourly pipeline completed successfully."
else
    log "NFL hourly pipeline exited with code: $EXIT_CODE"
fi

log "Finished: $(date)"
log ""

exit "$EXIT_CODE"
RUN_UPDATER_EOF

chmod --reference="$TARGET" "$CANDIDATE"

echo
echo "=== CANDIDATE PROOF ==="
bash -n "$CANDIDATE"
echo "CANDIDATE_SYNTAX=PASS"
require_hash "$CANDIDATE" "$EXPECTED_NEW_RUN_UPDATER_SHA"

python3 - "$CANDIDATE" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

required = [
    'exec 9>"$LOCK_FILE"',
    'flock -n 9',
    '"$PYTHON" "$UPDATER"',
    '"$PYTHON" "$script"',
    'wait_for_live_quiet "FanDuel injury ingest"',
    'run_stage "FanDuel injury ingest" "$INJURY_INGEST" 21',
    'wait_for_live_quiet "injury consensus"',
    'run_stage "Injury consensus" "$INJURY_CONSENSUS" 22',
    'wait_for_live_quiet "solver-ready injury rebuild"',
    'run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23',
]

missing = [x for x in required if x not in text]
if missing:
    raise SystemExit("FAIL | missing structural tokens: " + repr(missing))

order = [
    text.index('"$PYTHON" "$UPDATER"'),
    text.index('run_stage "FanDuel injury ingest"'),
    text.index('run_stage "Injury consensus"'),
    text.index('run_stage "Solver-ready injury rebuild"'),
]

if order != sorted(order) or len(set(order)) != len(order):
    raise SystemExit("FAIL | pipeline stage ordering proof")

print("PIPELINE_ORDER=UPDATER>INGEST>CONSENSUS>SOLVER_READY")
print("ONE_LOCK_WHOLE_CHAIN=PASS")
print("STRUCTURAL_PROOF=PASS")
PY

echo
echo "=== INSTALL ==="
mv "$CANDIDATE" "$TARGET"
trap - EXIT

require_hash "$TARGET" "$EXPECTED_NEW_RUN_UPDATER_SHA"
bash -n "$TARGET"
echo "INSTALLED_SYNTAX=PASS"

echo
echo "=== PROTECTED HASH CONTRACT AFTER ==="
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== CRON AFTER ==="
CRON_AFTER="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_AFTER"
if [[ "$CRON_AFTER" != "$CRON_LINE" ]]; then
    echo "FAIL | cron entry changed unexpectedly"
    cp -a "$BACKUP" "$TARGET"
    echo "ROLLBACK_RUN_UPDATER=PASS"
    exit 1
fi
echo "CRON_UNCHANGED=PASS"

echo
echo "=== SERVICES AFTER ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "NEW_RUN_UPDATER_SHA=$EXPECTED_NEW_RUN_UPDATER_SHA"
echo "NFL_INJURY_21D_STATUS=PASS"
echo "UPDATER_PY_MODIFIED=FALSE"
echo "INJURY_MODULES_MODIFIED=FALSE"
echo "CRON_MODIFIED=FALSE"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
