#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
LAUNCHER="$ROOT/run_updater.sh"
CRON_LOG="$ROOT/logs/cron.log"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_LAUNCHER_SHA="ec0941b870192f0f8fa47ac76c93f0c667e903616ca362546ffd7dc4591cf1d2"
EXPECTED_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_SOLVER_READY_SHA="35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"

cd "$ROOT"
mkdir -p "$BACKUP_DIR" "$ROOT/logs"

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
echo "WFS NFL — NFL-INJURY-21E"
echo "END-TO-END PRODUCTION LAUNCHER PROOF"
echo "======================================================================"

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "=== CRON CONTRACT ==="
CRON_LINE="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_LINE"
CRON_COUNT="$(printf '%s\n' "$CRON_LINE" | sed '/^$/d' | wc -l)"
if [[ "$CRON_COUNT" -ne 1 ]]; then
    echo "FAIL | expected exactly one launcher cron entry; found $CRON_COUNT"
    exit 1
fi

if flock -n "$ROOT/nfl_updater.lock" -c true; then
    echo "UPDATER_LOCK_PREFLIGHT=FREE"
else
    echo "FAIL | updater lock is currently busy; retry after current cycle finishes"
    exit 1
fi

echo
echo "=== PRE-RUN DATABASE PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    print("PRE_DB_INTEGRITY=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | pre-run DB integrity")
    for table in (
        "fanduel_slate_projection_pool",
        "fanduel_solver_ready_pool",
        "injury_consensus_current",
        "injury_news_signals",
    ):
        n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"PRE_{table.upper()}={n}")
PY

TS="$(date -u +%Y%m%dT%H%M%SZ)"
DB_BACKUP="$BACKUP_DIR/nfl_before_injury21e_${TS}.db"

echo
echo "=== SAFETY BACKUP ==="
"$PY" - "$DB" "$DB_BACKUP" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1:3]
with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
    s.backup(d)
with sqlite3.connect(f"file:{dst}?mode=ro", uri=True) as c:
    result = c.execute("PRAGMA integrity_check").fetchone()[0]
if result != "ok":
    raise SystemExit("FAIL | backup integrity")
print("SQLITE_BACKUP=PASS")
PY
echo "DB_BACKUP=$DB_BACKUP"

LOG_SIZE=0
if [[ -f "$CRON_LOG" ]]; then
    LOG_SIZE="$(stat -c '%s' "$CRON_LOG")"
fi

echo
echo "=== EXECUTE INSTALLED PRODUCTION LAUNCHER ONCE ==="
echo "LAUNCHER_START=$(date --iso-8601=seconds)"

set +e
"$LAUNCHER"
LAUNCHER_RC=$?
set -e

echo "LAUNCHER_END=$(date --iso-8601=seconds)"
echo "LAUNCHER_EXIT_CODE=$LAUNCHER_RC"

NEW_LOG="$BACKUP_DIR/injury21e_launcher_log_${TS}.txt"
if [[ -f "$CRON_LOG" ]]; then
    tail -c +$((LOG_SIZE + 1)) "$CRON_LOG" > "$NEW_LOG"
else
    : > "$NEW_LOG"
fi

echo
echo "=== THIS RUN'S LAUNCHER LOG ==="
cat "$NEW_LOG"

if [[ "$LAUNCHER_RC" -ne 0 ]]; then
    echo "FAIL | production launcher exited nonzero"
    echo "NO_AUTOMATIC_ROLLBACK_PERFORMED=TRUE"
    echo "Backup retained for controlled recovery: $DB_BACKUP"
    exit "$LAUNCHER_RC"
fi

echo
echo "=== LOG CONTRACT PROOF ==="
"$PY" - "$NEW_LOG" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(errors="replace")

required = [
    "NFL hourly pipeline launcher started:",
    "Starting core updater:",
    "Core updater completed successfully:",
    "Starting FanDuel injury ingest:",
    "FanDuel injury ingest completed successfully:",
    "Starting Injury consensus:",
    "Injury consensus completed successfully:",
    "Starting Solver-ready injury rebuild:",
    "Solver-ready injury rebuild completed successfully:",
    "NFL hourly pipeline completed successfully.",
]

missing = [s for s in required if s not in text]
if missing:
    raise SystemExit("FAIL | missing launcher success markers: " + repr(missing))

ordered = [
    text.index("Starting core updater:"),
    text.index("Starting FanDuel injury ingest:"),
    text.index("Starting Injury consensus:"),
    text.index("Starting Solver-ready injury rebuild:"),
]

if ordered != sorted(ordered):
    raise SystemExit("FAIL | launcher log stage ordering")

bad = [
    "Core updater FAILED",
    "FanDuel injury ingest FAILED",
    "Injury consensus FAILED",
    "Solver-ready injury rebuild FAILED",
    "LIVE quiet-window timeout",
    "NFL hourly pipeline exited with code:",
]
found_bad = [s for s in bad if s in text]
if found_bad:
    raise SystemExit("FAIL | failure markers found: " + repr(found_bad))

print("LAUNCHER_STAGE_ORDER=PASS")
print("LAUNCHER_SUCCESS_MARKERS=PASS")
print("LAUNCHER_FAILURE_MARKERS=NONE")
PY

echo
echo "=== POST-RUN DATABASE PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys

db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    print("POST_DB_INTEGRITY=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | post-run DB integrity")

    projection_rows = c.execute(
        "SELECT COUNT(*) FROM fanduel_slate_projection_pool"
    ).fetchone()[0]
    solver_rows = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool"
    ).fetchone()[0]
    consensus_rows = c.execute(
        "SELECT COUNT(*) FROM injury_consensus_current"
    ).fetchone()[0]
    signal_rows = c.execute(
        "SELECT COUNT(*) FROM injury_news_signals"
    ).fetchone()[0]

    print("POST_PROJECTION_ROWS=" + str(projection_rows))
    print("POST_SOLVER_READY_ROWS=" + str(solver_rows))
    print("POST_CONSENSUS_ROWS=" + str(consensus_rows))
    print("POST_INJURY_SIGNAL_ROWS=" + str(signal_rows))

    if projection_rows <= 0:
        raise SystemExit("FAIL | projection pool empty")
    if solver_rows != projection_rows:
        raise SystemExit(
            f"FAIL | solver-ready rows {solver_rows} != projection rows {projection_rows}"
        )
    if consensus_rows <= 0:
        raise SystemExit("FAIL | consensus empty")
    if signal_rows <= 0:
        raise SystemExit("FAIL | injury signal table empty")

    required = {
        "injury_consensus_found",
        "injury_gsis_id",
        "injury_gate",
        "injury_status",
        "injury_gate_reason",
        "injury_risk",
        "solver_status",
        "solver_eligible",
    }
    cols = {
        r[1]
        for r in c.execute('PRAGMA table_info("fanduel_solver_ready_pool")')
    }
    missing = sorted(required - cols)
    print("MISSING_REQUIRED_COLUMNS=" + repr(missing))
    if missing:
        raise SystemExit("FAIL | required injury audit columns missing")

    attached = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool "
        "WHERE injury_consensus_found=1"
    ).fetchone()[0]
    blocked = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool "
        "WHERE solver_status='BLOCKED_INJURY'"
    ).fetchone()[0]
    escapes = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool "
        "WHERE injury_gate='BLOCK' AND solver_eligible=1"
    ).fetchone()[0]
    blank_gsis = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool "
        "WHERE injury_consensus_found=1 "
        "AND (injury_gsis_id IS NULL OR trim(injury_gsis_id)='')"
    ).fetchone()[0]

    print("CONSENSUS_ATTACHED_ROWS=" + str(attached))
    print("BLOCKED_INJURY_ROWS=" + str(blocked))
    print("BLOCK_ESCAPE_ROWS=" + str(escapes))
    print("ATTACHED_BLANK_GSIS_ROWS=" + str(blank_gsis))

    if attached <= 0:
        raise SystemExit("FAIL | no consensus attachments")
    if escapes != 0:
        raise SystemExit("FAIL | injury BLOCK escape")
    if blank_gsis != 0:
        raise SystemExit("FAIL | attached row missing GSIS")

    print("NFL_INJURY_21E_DB_PROOF=PASS")
PY

echo
echo "=== FROZEN HASH CONTRACT AFTER ==="
require_hash "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== CRON AFTER ==="
CRON_AFTER="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_AFTER"
if [[ "$CRON_AFTER" != "$CRON_LINE" ]]; then
    echo "FAIL | cron changed unexpectedly"
    exit 1
fi
echo "CRON_UNCHANGED=PASS"

echo
echo "=== SERVICES AFTER ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "NFL_INJURY_21E_STATUS=PASS"
echo "RUN_UPDATER_SHA=$EXPECTED_LAUNCHER_SHA"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
echo "NO_AUTOMATIC_ROLLBACK_NEEDED=TRUE"
