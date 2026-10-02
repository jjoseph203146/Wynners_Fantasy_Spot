#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
LOCK="$ROOT/nfl_updater.lock"
BACKUP_ROOT="$ROOT/backups/injury_consensus"

EXPECTED_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_SOLVER_READY_SHA="35d955e868b205a6851564d085353e23fb79e240ea3bace064277f95a6f5ac2a"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"
EXPECTED_RUN_UPDATER_SHA="6315c6c700d04a537cb0751c7f015f6b59fe9b77da5763c6a05d23767cdd2d4c"

cd "$ROOT"
mkdir -p "$BACKUP_ROOT"

require_hash() {
    local file="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    printf '%s  %s\n' "$actual" "$file"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL | protected hash mismatch: $file"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 1
    fi
}

wait_for_live_quiet() {
    local label="$1"
    local quiet=0

    echo
    echo "=== WAIT FOR LIVE QUIET WINDOW: $label ==="

    for _ in $(seq 1 120); do
        if ! pgrep -f '[l]ive_concurrency_gate.py --season 2026' >/dev/null 2>&1; then
            quiet=1
            break
        fi
        sleep 1
    done

    if [[ "$quiet" -ne 1 ]]; then
        echo "FAIL | no LIVE gate quiet window observed within 120 seconds"
        return 1
    fi

    echo "LIVE_GATE_QUIET_WINDOW_${label}=PASS"
}

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-21C"
echo "STANDALONE AUTOMATIC INJURY CHAIN VALIDATION"
echo "NO CRON OR MODULE EDITS"
echo "======================================================================"

echo
echo "=== FROZEN HASH CONTRACT ==="
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/run_updater.sh" "$EXPECTED_RUN_UPDATER_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FAIL | updater lock is busy"
    exit 1
fi
echo "PASS | acquired existing nfl_updater.lock"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$BACKUP_ROOT/injury21c_${TS}"
DB_BACKUP="$BACKUP_DIR/nfl.db"
mkdir -p "$BACKUP_DIR/files"

FILES=(
    "$ROOT/data/csv/fanduel_injury_ingest_current.csv"
    "$ROOT/data/csv/injury_consensus_current.csv"
    "$ROOT/data/parquet/injury_consensus_current.parquet"
    "$ROOT/data/csv/injury_consensus_audit.csv"
    "$ROOT/data/csv/fanduel_solver_ready_pool.csv"
    "$ROOT/data/parquet/fanduel_solver_ready_pool.parquet"
    "$ROOT/data/csv/fanduel_solver_ready_manifest.csv"
    "$ROOT/data/csv/audit_fanduel_solver_ready_gaps.csv"
)

echo
echo "=== BACKUP ==="
"$PY" - "$DB" "$DB_BACKUP" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1:3]
with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
    s.backup(d)
print("SQLITE_BACKUP=PASS")
PY

for f in "${FILES[@]}"; do
    key="$(printf '%s' "$f" | sha256sum | awk '{print $1}')"
    if [[ -e "$f" ]]; then
        cp -a "$f" "$BACKUP_DIR/files/$key"
        printf '%s\t%s\t%s\n' "PRESENT" "$key" "$f" >> "$BACKUP_DIR/file_manifest.tsv"
    else
        printf '%s\t%s\t%s\n' "ABSENT" "$key" "$f" >> "$BACKUP_DIR/file_manifest.tsv"
    fi
done

echo "BACKUP_DIR=$BACKUP_DIR"

rollback_needed=1
rollback() {
    local rc=$?
    if [[ "$rollback_needed" -eq 1 ]]; then
        echo
        echo "!!! NFL-INJURY-21C FAILURE — RESTORING PRE-RUN STATE !!!"
        cp -a "$DB_BACKUP" "$DB"

        while IFS=$'\t' read -r state key path; do
            if [[ "$state" == "PRESENT" ]]; then
                mkdir -p "$(dirname "$path")"
                cp -a "$BACKUP_DIR/files/$key" "$path"
            else
                rm -f "$path"
            fi
        done < "$BACKUP_DIR/file_manifest.tsv"

        echo "ROLLBACK=PASS"
        echo "NFL_INJURY_21C_STATUS=ROLLED_BACK"
    fi
    exit "$rc"
}
trap rollback ERR INT TERM

echo
echo "=== PRE-RUN DB SHAPE ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    for table in (
        "fanduel_slate_projection_pool",
        "fanduel_solver_ready_pool",
        "injury_consensus_current",
        "injury_news_signals",
    ):
        row = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row:
            print(f"{table}=MISSING")
        else:
            n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"{table}={n}")
PY

wait_for_live_quiet "INGEST"

echo
echo "=== STEP 1 — FANDUEL INJURY INGEST ==="
set +e
"$PY" "$ROOT/fanduel_injury_ingest.py"
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
    echo "FAIL | FanDuel injury ingest exited $rc"
    exit 21
fi
echo "FANDUEL_INJURY_INGEST=PASS"

wait_for_live_quiet "CONSENSUS"

echo
echo "=== STEP 2 — INJURY CONSENSUS ==="
set +e
"$PY" "$ROOT/injury_consensus.py"
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
    echo "FAIL | injury consensus exited $rc"
    exit 22
fi
echo "INJURY_CONSENSUS=PASS"

wait_for_live_quiet "SOLVER_READY"

echo
echo "=== STEP 3 — SOLVER-READY INJURY GATE ==="
set +e
"$PY" "$ROOT/fanduel_solver_ready_pool.py"
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
    echo "FAIL | solver-ready rebuild exited $rc"
    exit 23
fi
echo "SOLVER_READY_REBUILD=PASS"

echo
echo "=== POST-RUN PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys

db = sys.argv[1]
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    print("DB_INTEGRITY=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | database integrity")

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

    print("PROJECTION_ROWS=" + str(projection_rows))
    print("SOLVER_READY_ROWS=" + str(solver_rows))
    print("CONSENSUS_ROWS=" + str(consensus_rows))
    print("INJURY_SIGNAL_ROWS=" + str(signal_rows))

    if projection_rows <= 0:
        raise SystemExit("FAIL | projection pool empty")
    if solver_rows != projection_rows:
        raise SystemExit(
            f"FAIL | solver-ready row count {solver_rows} != projection rows {projection_rows}"
        )
    if consensus_rows <= 0:
        raise SystemExit("FAIL | consensus empty")
    if signal_rows <= 0:
        raise SystemExit("FAIL | injury news signals empty")

    cols = {
        r[1]
        for r in c.execute('PRAGMA table_info("fanduel_solver_ready_pool")')
    }
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
    missing = sorted(required - cols)
    print("MISSING_REQUIRED_COLUMNS=" + repr(missing))
    if missing:
        raise SystemExit("FAIL | missing solver-ready injury columns")

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
    mismatch = c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool "
        "WHERE injury_consensus_found=1 "
        "AND (injury_gsis_id IS NULL OR trim(injury_gsis_id)='')"
    ).fetchone()[0]

    print("CONSENSUS_ATTACHED_ROWS=" + str(attached))
    print("BLOCKED_INJURY_ROWS=" + str(blocked))
    print("BLOCK_ESCAPE_ROWS=" + str(escapes))
    print("ATTACHED_BLANK_GSIS_ROWS=" + str(mismatch))

    if attached <= 0:
        raise SystemExit("FAIL | no consensus rows attached to solver-ready pool")
    if escapes != 0:
        raise SystemExit("FAIL | injury BLOCK escape")
    if mismatch != 0:
        raise SystemExit("FAIL | attached consensus row missing exact GSIS")

    statuses = c.execute(
        "SELECT solver_status, COUNT(*) "
        "FROM fanduel_solver_ready_pool "
        "GROUP BY solver_status ORDER BY solver_status"
    ).fetchall()
    print("SOLVER_STATUS_COUNTS=" + repr(statuses))

    print("NFL_INJURY_21C_DB_PROOF=PASS")
PY

echo
echo "=== FROZEN HASH CONTRACT AFTER ==="
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_SOLVER_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/run_updater.sh" "$EXPECTED_RUN_UPDATER_SHA"

echo
echo "=== SERVICES AFTER ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

rollback_needed=0
trap - ERR INT TERM

echo
echo "NFL_INJURY_21C_STATUS=PASS"
echo "CRON_MODIFIED=FALSE"
echo "UPDATER_MODIFIED=FALSE"
echo "RUN_UPDATER_MODIFIED=FALSE"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
