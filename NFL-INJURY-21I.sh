#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
TARGET="$ROOT/run_updater.sh"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_CURRENT_SHA="4baf0e8328f3ab78c8058b0c9721bd26b8790187d39027a6e041c9778f20767b"
EXPECTED_NEW_SHA="46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a"
EXPECTED_LIVE_SAFE_SHA="458d057bca746fd6d2a64a1a10bcd392c9395f345cc8bf655e06f905ea362165"
EXPECTED_GATE_SHA="0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a"
EXPECTED_ORCH_SHA="7767f05398677febfc303eb41626d23b8f7e11aee24b2144d5a0bb6294aab816"
EXPECTED_LIVE_INGEST_SHA="f92c910722e05910dfc47b9e38821a7027d245a6b4d412037559c58e48a4acd4"
EXPECTED_FD_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_SOLVER_SHA="35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"

cd "$ROOT"
mkdir -p "$BACKUP_DIR"

require_hash() {
    local file="$1" expected="$2" actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    printf '%s  %s\n' "$actual" "$file"
    [[ "$actual" == "$expected" ]] || {
        echo "FAIL | hash mismatch: $file"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 1
    }
}

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-21I"
echo "BOUNDED FANDUEL INGEST RETRY DEPLOYMENT"
echo "======================================================================"

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$TARGET" "$EXPECTED_CURRENT_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_FD_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== SERVICES ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

CRON_BEFORE="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "CRON_BEFORE=$CRON_BEFORE"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$BACKUP_DIR/run_updater_before_injury21i_${TS}.sh"
CANDIDATE="$ROOT/.run_updater_injury21i.$$"

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

run_fanduel_ingest_with_retry() {
    local attempt=1
    local max_attempts=3
    local rc=0

    while [ "$attempt" -le "$max_attempts" ]; do
        log "------------------------------------------------------------"
        log "Starting FanDuel injury ingest attempt $attempt/$max_attempts: $(date)"

        "$PYTHON" "$INJURY_INGEST" >> "$CRON_LOG" 2>&1
        rc=$?

        if [ "$rc" -eq 0 ]; then
            log "FanDuel injury ingest completed successfully on attempt $attempt/$max_attempts: $(date)"
            return 0
        fi

        log "FanDuel injury ingest attempt $attempt/$max_attempts FAILED with exit code $rc."

        if [ "$attempt" -eq "$max_attempts" ]; then
            log "FanDuel injury ingest exhausted $max_attempts attempts."
            return 21
        fi

        if [ "$attempt" -eq 1 ]; then
            log "Retrying FanDuel injury ingest in 15 seconds."
            sleep 15
        else
            log "Retrying FanDuel injury ingest in 30 seconds."
            sleep 30
        fi

        attempt=$((attempt + 1))
    done

    return 21
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
    run_fanduel_ingest_with_retry
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "injury consensus"; then
        return 32
    fi
    run_stage "Injury consensus" "$INJURY_CONSENSUS" 22
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "solver-ready injury rebuild"; then
        return 33
    fi
    run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    return 0
}

log "============================================================"
log "NFL hourly pipeline launcher started: $(date)"

# Hold the existing updater lock for the ENTIRE core + injury chain.
exec 9>"$LOCK_FILE"

# Wait up to 60 seconds for an in-flight LIVE read cycle to release
# its shared nfl_updater.lock. Once acquired exclusively, this lock
# protects the complete updater + injury chain from new LIVE DB reads.
if ! flock -w 60 9; then
    log "Updater skipped because shared updater/LIVE lock remained busy for 60 seconds."
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
require_hash "$CANDIDATE" "$EXPECTED_NEW_SHA"

python3 - "$CANDIDATE" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()

required = [
    'run_fanduel_ingest_with_retry()',
    'local max_attempts=3',
    'Retrying FanDuel injury ingest in 15 seconds.',
    'Retrying FanDuel injury ingest in 30 seconds.',
    'FanDuel injury ingest exhausted $max_attempts attempts.',
    'return 21',
    'run_fanduel_ingest_with_retry',
    'run_stage "Injury consensus" "$INJURY_CONSENSUS" 22',
    'run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23',
    'flock -w 60 9',
]
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit("FAIL | missing structural tokens: " + repr(missing))

if 'run_stage "FanDuel injury ingest"' in text:
    raise SystemExit("FAIL | old single-attempt FanDuel call remains")

order = [
    text.index('"$PYTHON" "$UPDATER"'),
    text.index('run_fanduel_ingest_with_retry', text.index('run_pipeline()')),
    text.index('run_stage "Injury consensus"'),
    text.index('run_stage "Solver-ready injury rebuild"'),
]
if order != sorted(order):
    raise SystemExit("FAIL | stage ordering")

print("PIPELINE_ORDER=UPDATER>FANDUEL_RETRY>CONSENSUS>SOLVER_READY")
print("FANDUEL_MAX_ATTEMPTS=3")
print("FANDUEL_RETRY_BACKOFF=15,30")
print("LOCAL_STAGES_RETRY=FALSE")
print("FAIL_CLOSED_EXIT_CODE=21")
print("STRUCTURAL_PROOF=PASS")
PY

echo
echo "=== SYNTHETIC RETRY POLICY PROOF ==="
python3 <<'PY'
def policy(results):
    waits = []
    for attempt, rc in enumerate(results[:3], start=1):
        if rc == 0:
            return 0, attempt, waits
        if attempt < 3:
            waits.append(15 if attempt == 1 else 30)
    return 21, min(3, len(results)), waits

cases = [
    ([0], (0, 1, [])),
    ([1, 0], (0, 2, [15])),
    ([1, 1, 0], (0, 3, [15, 30])),
    ([1, 1, 1], (21, 3, [15, 30])),
]
for inputs, expected in cases:
    actual = policy(inputs)
    if actual != expected:
        raise SystemExit(f"FAIL | retry policy {inputs} -> {actual} != {expected}")
print("SYNTHETIC_RETRY_POLICY=PASS")
PY

echo
echo "=== INSTALL ==="
mv "$CANDIDATE" "$TARGET"
require_hash "$TARGET" "$EXPECTED_NEW_SHA"
bash -n "$TARGET"
echo "INSTALLED_SYNTAX=PASS"

echo
echo "=== PROTECTED HASHES AFTER ==="
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_FD_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

CRON_AFTER="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "CRON_AFTER=$CRON_AFTER"
[[ "$CRON_AFTER" == "$CRON_BEFORE" ]] || {
    cp -a "$BACKUP" "$TARGET"
    echo "ROLLBACK_RUN_UPDATER=PASS"
    echo "FAIL | cron changed unexpectedly"
    exit 1
}
echo "CRON_UNCHANGED=PASS"

echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "NEW_RUN_UPDATER_SHA=$EXPECTED_NEW_SHA"
echo "NFL_INJURY_21I_STATUS=PASS"
echo "FANDUEL_INGEST_MODULE_MODIFIED=FALSE"
echo "LIVE_MODULES_MODIFIED=FALSE"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
