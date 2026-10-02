#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
LAUNCHER="$ROOT/run_updater.sh"
CRON_LOG="$ROOT/logs/cron.log"
LIVE_LOG="$ROOT/logs/wfs_live_safe_poll.log"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_LAUNCHER_SHA="46ffe95fb09585538bc01efceba6c24a111bfe85072f660a6f9a1e1f380ef96a"
EXPECTED_LIVE_SAFE_SHA="458d057bca746fd6d2a64a1a10bcd392c9395f345cc8bf655e06f905ea362165"
EXPECTED_LIVE_GATE_SHA="0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a"
EXPECTED_LIVE_ORCH_SHA="7767f05398677febfc303eb41626d23b8f7e11aee24b2144d5a0bb6294aab816"
EXPECTED_LIVE_INGEST_SHA="f92c910722e05910dfc47b9e38821a7027d245a6b4d412037559c58e48a4acd4"
EXPECTED_INJURY_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_SOLVER_READY_SHA="35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"

cd "$ROOT"
mkdir -p "$BACKUP_DIR" "$ROOT/logs"

hash_of() { sha256sum "$1" | awk '{print $1}'; }

require_hash() {
    local file="$1" expected="$2" actual
    actual="$(hash_of "$file")"
    printf '%s  %s\n' "$actual" "$file"
    [[ "$actual" == "$expected" ]] || {
        echo "FAIL | hash mismatch: $file"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 1
    }
}

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-21J"
echo "FINAL END-TO-END AUTOMATIC INJURY PIPELINE PROOF"
echo "======================================================================"

echo
echo "=== CRON COLLISION GUARD ==="
MINUTE="$(date +%M)"
MINUTE_NUM=$((10#$MINUTE))
echo "CURRENT_MINUTE=$MINUTE"
if (( MINUTE_NUM >= 5 && MINUTE_NUM <= 10 )); then
    echo "FAIL | too close to minute-7 cron window; rerun outside minutes 05-10"
    exit 1
fi
echo "CRON_COLLISION_GUARD=PASS"

echo
echo "=== FROZEN HASH CONTRACT BEFORE ==="
require_hash "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INJURY_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "=== CRON CONTRACT ==="
CRON_BEFORE="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_BEFORE"
CRON_COUNT="$(printf '%s\n' "$CRON_BEFORE" | sed '/^$/d' | wc -l)"
[[ "$CRON_COUNT" -eq 1 ]] || {
    echo "FAIL | expected exactly one launcher cron entry; found $CRON_COUNT"
    exit 1
}

if flock -n "$ROOT/nfl_updater.lock" -c true; then
    echo "UPDATER_LOCK_PREFLIGHT=FREE"
else
    echo "UPDATER_LOCK_PREFLIGHT=BUSY_ACCEPTED"
    echo "INFO | LIVE may hold shared lock; launcher will wait up to 60 seconds"
fi

echo
echo "=== PRE-RUN DB HEALTH ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    fk = len(c.execute("PRAGMA foreign_key_check").fetchall())
    print("PRE_DB_INTEGRITY=" + integrity)
    print("PRE_FOREIGN_KEY_VIOLATIONS=" + str(fk))
    if integrity != "ok" or fk != 0:
        raise SystemExit("FAIL | pre-run database health")
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
DB_BACKUP="$BACKUP_DIR/nfl_before_injury21j_${TS}.db"

echo
echo "=== SAFETY BACKUP ==="
"$PY" - "$DB" "$DB_BACKUP" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1:3]
with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
    s.backup(d)
with sqlite3.connect(f"file:{dst}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    fk = len(c.execute("PRAGMA foreign_key_check").fetchall())
print("BACKUP_INTEGRITY=" + integrity)
print("BACKUP_FOREIGN_KEY_VIOLATIONS=" + str(fk))
if integrity != "ok" or fk != 0:
    raise SystemExit("FAIL | backup health")
print("SQLITE_BACKUP=PASS")
PY
echo "DB_BACKUP=$DB_BACKUP"

CRON_SIZE=0
LIVE_SIZE=0
[[ -f "$CRON_LOG" ]] && CRON_SIZE="$(stat -c '%s' "$CRON_LOG")"
[[ -f "$LIVE_LOG" ]] && LIVE_SIZE="$(stat -c '%s' "$LIVE_LOG")"

echo
echo "=== EXECUTE PRODUCTION LAUNCHER ONCE ==="
echo "LAUNCHER_START=$(date --iso-8601=seconds)"

set +e
"$LAUNCHER"
RC=$?
set -e

echo "LAUNCHER_END=$(date --iso-8601=seconds)"
echo "LAUNCHER_EXIT_CODE=$RC"

NEW_CRON_LOG="$BACKUP_DIR/injury21j_cron_segment_${TS}.txt"
NEW_LIVE_LOG="$BACKUP_DIR/injury21j_live_segment_${TS}.txt"

if [[ -f "$CRON_LOG" ]]; then
    tail -c +$((CRON_SIZE + 1)) "$CRON_LOG" > "$NEW_CRON_LOG"
else
    : > "$NEW_CRON_LOG"
fi

if [[ -f "$LIVE_LOG" ]]; then
    tail -c +$((LIVE_SIZE + 1)) "$LIVE_LOG" > "$NEW_LIVE_LOG"
else
    : > "$NEW_LIVE_LOG"
fi

echo
echo "=== THIS RUN'S PIPELINE LOG ==="
cat "$NEW_CRON_LOG"

if [[ "$RC" -ne 0 ]]; then
    echo "FAIL | production launcher exited nonzero"
    echo "NO_AUTOMATIC_ROLLBACK_PERFORMED=TRUE"
    echo "DB_BACKUP_RETAINED=$DB_BACKUP"
    exit "$RC"
fi

echo
echo "=== PIPELINE LOG CONTRACT ==="
"$PY" - "$NEW_CRON_LOG" <<'PY'
from pathlib import Path
import re, sys

text = Path(sys.argv[1]).read_text(errors="replace")

required = [
    "NFL hourly pipeline launcher started:",
    "Starting core updater:",
    "Core updater completed successfully:",
    "Starting FanDuel injury ingest attempt ",
    "FanDuel injury ingest completed successfully on attempt ",
    "Starting Injury consensus:",
    "Injury consensus completed successfully:",
    "Starting Solver-ready injury rebuild:",
    "Solver-ready injury rebuild completed successfully:",
    "NFL hourly pipeline completed successfully.",
]
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit("FAIL | missing success markers: " + repr(missing))

core = text.index("Starting core updater:")
fd = text.index("Starting FanDuel injury ingest attempt ")
cons = text.index("Starting Injury consensus:")
solver = text.index("Starting Solver-ready injury rebuild:")
if not core < fd < cons < solver:
    raise SystemExit("FAIL | pipeline stage ordering")

forbidden = [
    "database is locked",
    "Core updater FAILED",
    "Injury consensus FAILED",
    "Solver-ready injury rebuild FAILED",
    "LIVE quiet-window timeout",
    "NFL hourly pipeline exited with code:",
    "FanDuel injury ingest exhausted 3 attempts.",
]
found = [x for x in forbidden if x in text]
if found:
    raise SystemExit("FAIL | forbidden failure markers: " + repr(found))

attempts = [int(x) for x in re.findall(
    r"Starting FanDuel injury ingest attempt ([123])/3", text
)]
if not attempts:
    raise SystemExit("FAIL | no FanDuel attempts detected")
if attempts != list(range(1, max(attempts) + 1)):
    raise SystemExit("FAIL | FanDuel attempts not sequential: " + repr(attempts))
if len(attempts) > 3:
    raise SystemExit("FAIL | more than 3 FanDuel attempts")

success = re.search(
    r"FanDuel injury ingest completed successfully on attempt ([123])/3",
    text,
)
if not success:
    raise SystemExit("FAIL | no FanDuel success attempt")
success_attempt = int(success.group(1))
if success_attempt != max(attempts):
    raise SystemExit(
        f"FAIL | success attempt {success_attempt} != final attempt {max(attempts)}"
    )

print("PIPELINE_STAGE_ORDER=PASS")
print("PIPELINE_SUCCESS_MARKERS=PASS")
print("DATABASE_LOCK_ERRORS=0")
print("PIPELINE_FAILURE_MARKERS=NONE")
print("FANDUEL_ATTEMPTS_USED=" + str(len(attempts)))
print("FANDUEL_SUCCESS_ATTEMPT=" + str(success_attempt))
print("FANDUEL_BOUNDED_RETRY_RUNTIME_PROOF=PASS")
PY

echo
echo "=== LIVE LOCK CONTRACT OBSERVATION ==="
if [[ -s "$NEW_LIVE_LOG" ]]; then
    grep -E \
      'WAITING_FOR_UPDATER_SHARED_LOCK=1|UPDATER_SHARED_LOCK_ACQUIRED=1|UPDATER_SHARED_LOCK_RELEASED=1|POLL_RESULT|LIVE18R_EXCEPTION' \
      "$NEW_LIVE_LOG" || true
else
    echo "LIVE_LOG_SEGMENT=<EMPTY>"
fi

if grep -q 'LIVE18R_EXCEPTION' "$NEW_LIVE_LOG" 2>/dev/null; then
    echo "FAIL | LIVE exception observed during production proof"
    exit 1
fi
echo "LIVE_EXCEPTION_MARKERS=NONE"

echo
echo "=== POST-RUN DB PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys

db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    fk = len(c.execute("PRAGMA foreign_key_check").fetchall())
    print("POST_DB_INTEGRITY=" + integrity)
    print("POST_FOREIGN_KEY_VIOLATIONS=" + str(fk))
    if integrity != "ok" or fk != 0:
        raise SystemExit("FAIL | post-run database health")

    projection = c.execute(
        "SELECT COUNT(*) FROM fanduel_slate_projection_pool"
    ).fetchone()[0]
    solver = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool"
    ).fetchone()[0]
    consensus = c.execute(
        "SELECT COUNT(*) FROM injury_consensus_current"
    ).fetchone()[0]
    signals = c.execute(
        "SELECT COUNT(*) FROM injury_news_signals"
    ).fetchone()[0]

    print("POST_PROJECTION_ROWS=" + str(projection))
    print("POST_SOLVER_READY_ROWS=" + str(solver))
    print("POST_CONSENSUS_ROWS=" + str(consensus))
    print("POST_INJURY_SIGNAL_ROWS=" + str(signals))

    if projection <= 0:
        raise SystemExit("FAIL | projection pool empty")
    if solver != projection:
        raise SystemExit(
            f"FAIL | solver rows {solver} != projection rows {projection}"
        )
    if consensus <= 0 or signals <= 0:
        raise SystemExit("FAIL | injury tables empty")

    required_cols = {
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
    missing = sorted(required_cols - cols)
    print("MISSING_REQUIRED_COLUMNS=" + repr(missing))
    if missing:
        raise SystemExit("FAIL | missing injury audit columns")

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

print("NFL_INJURY_21J_DB_PROOF=PASS")
PY

echo
echo "=== FROZEN HASH CONTRACT AFTER ==="
require_hash "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INJURY_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"

echo
echo "=== CRON AFTER ==="
CRON_AFTER="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "$CRON_AFTER"
[[ "$CRON_AFTER" == "$CRON_BEFORE" ]] || {
    echo "FAIL | cron changed unexpectedly"
    exit 1
}
echo "CRON_UNCHANGED=PASS"

echo
echo "=== SERVICES AFTER ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "NFL_INJURY_21J_STATUS=PASS"
echo "RUN_UPDATER_SHA=$EXPECTED_LAUNCHER_SHA"
echo "LIVE_SAFE_POLL_SHA=$EXPECTED_LIVE_SAFE_SHA"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
echo "NO_AUTOMATIC_ROLLBACK_NEEDED=TRUE"
