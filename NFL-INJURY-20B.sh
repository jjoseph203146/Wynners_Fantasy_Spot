#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
TARGET="$ROOT/fanduel_solver_ready_pool.py"
STAGED="$ROOT/fanduel_solver_ready_pool_injury20b.py"
LOCK="$ROOT/nfl_updater.lock"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_OLD_SHA="9ebcf4731cf4e853d7c5af04d0a2d802e79204ad268cce3e423445e6f128e370"
EXPECTED_NEW_SHA="e385c3acf42026a3c704a92162c406b19f310df0982190e2916a348f540a0617"
EXPECTED_CONSENSUS_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"
EXPECTED_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_APP_SHA="07cde83abf3904ad30b14d0917eeaaba312528ef546fbb1a0dc50064bacf3a7c"
EXPECTED_SOLVER_SHA="753a85e8efd73407d88dbf501b4bdad281fb5241a85c00a8b690b78cbd51dd95"
EXPECTED_V5_SHA="4a0b1e864cc13678bdcbb4df3480b94f83a9c999b935b3a72d4bb95949ca9793"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"
EXPECTED_LIVE_INGEST_SHA="f92c910722e05910dfc47b9e38821a7027d245a6b4d412037559c58e48a4acd4"
EXPECTED_LIVE_SAFE_SHA="96d3fffbc3cbbc51f2675e0c231cfa528bcba1dde23f807fcca0ea7f81057673"
EXPECTED_LIVE_ORCH_SHA="7767f05398677febfc303eb41626d23b8f7e11aee24b2144d5a0bb6294aab816"
EXPECTED_LIVE_GATE_SHA="0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a"
EXPECTED_LIVE_VIEW_SHA="11ae0beabf6de740ee920c15cf0f50be4219d9a7155553f050c7093b24bb67b6"

cd "$ROOT"
mkdir -p "$BACKUP_DIR"

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-20B CONTROLLED WRITE"
echo "SOLVER-READY INJURY GATE INTEGRATION"
echo "======================================================================"

require_hash() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  printf '%s  %s\n' "$actual" "$(basename "$file")"
  [[ "$actual" == "$expected" ]] || {
    echo "FAIL | hash mismatch: $file"
    echo "EXPECTED=$expected"
    echo "ACTUAL=$actual"
    exit 1
  }
}

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$TARGET" "$EXPECTED_OLD_SHA"
require_hash "$STAGED" "$EXPECTED_NEW_SHA"
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/app.py" "$EXPECTED_APP_SHA"
require_hash "$ROOT/fanduel_nfl_ui_solver.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/fanduel_slate_projection_attach_v5.py" "$EXPECTED_V5_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/wfs_live_view.py" "$EXPECTED_LIVE_VIEW_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

exec 9>"$LOCK"
flock -n 9 || { echo "FAIL | updater lock busy"; exit 1; }
echo "PASS | acquired nfl updater lock"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
MODULE_BACKUP="$BACKUP_DIR/fanduel_solver_ready_pool_before_injury20b_${TS}.py"
DB_BACKUP="$BACKUP_DIR/nfl_before_injury20b_${TS}.db"

cp -a "$TARGET" "$MODULE_BACKUP"
"$PY" - "$DB" "$DB_BACKUP" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as s, sqlite3.connect(sys.argv[2]) as d:
    s.backup(d)
print("SQLITE_BACKUP=PASS")
PY

echo "MODULE_BACKUP=$MODULE_BACKUP"
echo "DB_BACKUP=$DB_BACKUP"

rollback_needed=1
rollback() {
  rc=$?
  if [[ "$rollback_needed" -eq 1 ]]; then
    echo "!!! FAILURE — ROLLING BACK NFL-INJURY-20B !!!"
    cp -a "$MODULE_BACKUP" "$TARGET"
    cp -a "$DB_BACKUP" "$DB"
    echo "ROLLBACK_TARGET_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
  fi
  exit "$rc"
}
trap rollback ERR INT TERM

echo
echo "=== INSTALL + COMPILE ==="
cp -a "$STAGED" "$TARGET"
require_hash "$TARGET" "$EXPECTED_NEW_SHA"
"$PY" -m py_compile "$TARGET"
echo "PY_COMPILE=PASS"

echo
echo "=== RUN 1 ==="
"$PY" "$TARGET"

HASH1="$("$PY" - "$DB" <<'PY'
import hashlib, json, sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute(
        "SELECT * FROM fanduel_solver_ready_pool "
        "ORDER BY slate_slug_solver,player_solver,team_solver,salary_solver"
    )]
print(hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest())
PY
)"
echo "RUN1_SEMANTIC_HASH=$HASH1"

echo
echo "=== RUN 1 PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    c.row_factory=sqlite3.Row

    esc=c.execute("""
        SELECT COUNT(*)
        FROM fanduel_solver_ready_pool
        WHERE injury_gate='BLOCK'
          AND solver_eligible=1
    """).fetchone()[0]
    print("BLOCK_ESCAPE_ROWS=" + str(esc))
    if esc:
        raise SystemExit("FAIL | BLOCK injury escaped solver eligibility")

    blocked=c.execute("""
        SELECT COUNT(*)
        FROM fanduel_solver_ready_pool
        WHERE solver_status='BLOCKED_INJURY'
    """).fetchone()[0]
    print("BLOCKED_INJURY_ROWS=" + str(blocked))
    if blocked < 1:
        raise SystemExit("FAIL | no injury-blocked solver rows found")

    h=c.execute("""
        SELECT
            player_solver,team_solver,solver_position,position_identity,
            solver_status,solver_eligible,
            injury_gate,injury_status,injury_gate_reason,injury_risk
        FROM fanduel_solver_ready_pool
        WHERE position_identity='00-0040734'
    """).fetchall()
    print("HENDERSON_ROWS=" + str(len(h)))
    for r in h:
        print("HENDERSON_PROOF=" + repr(tuple(r)))
        if r["solver_status"] != "BLOCKED_INJURY" or r["solver_eligible"] != 0:
            raise SystemExit("FAIL | TreVeyon Henderson not hard-blocked")

    if not h:
        raise SystemExit("FAIL | TreVeyon Henderson not found in solver-ready pool")

    missing_identity_block=c.execute("""
        SELECT COUNT(*)
        FROM fanduel_solver_ready_pool
        WHERE solver_status='BLOCKED_INJURY'
          AND (position_identity IS NULL OR TRIM(position_identity)='')
    """).fetchone()[0]
    print("INJURY_BLOCK_WITHOUT_IDENTITY_ROWS=" + str(missing_identity_block))
    if missing_identity_block:
        raise SystemExit("FAIL | injury block without exact GSIS identity")

    print("INJURY_GATE_INTEGRATION_PROOF=PASS")
PY

echo
echo "=== RUN 2 / DETERMINISM ==="
"$PY" "$TARGET"

HASH2="$("$PY" - "$DB" <<'PY'
import hashlib, json, sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute(
        "SELECT * FROM fanduel_solver_ready_pool "
        "ORDER BY slate_slug_solver,player_solver,team_solver,salary_solver"
    )]
print(hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest())
PY
)"
echo "RUN2_SEMANTIC_HASH=$HASH2"
[[ "$HASH1" == "$HASH2" ]]
echo "SEMANTIC_DETERMINISM=PASS"

echo
echo "=== FINAL DB PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    integrity=c.execute("PRAGMA integrity_check").fetchone()[0]
    print("DB_INTEGRITY_AFTER=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | DB integrity")

    print("SOLVER_ELIGIBLE_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool WHERE solver_eligible=1"
    ).fetchone()[0]))
    print("BLOCKED_INJURY_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool WHERE solver_status='BLOCKED_INJURY'"
    ).fetchone()[0]))
PY

echo
echo "=== PROTECTED HASHES AFTER ==="
require_hash "$ROOT/injury_consensus.py" "$EXPECTED_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/app.py" "$EXPECTED_APP_SHA"
require_hash "$ROOT/fanduel_nfl_ui_solver.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/fanduel_slate_projection_attach_v5.py" "$EXPECTED_V5_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/wfs_live_view.py" "$EXPECTED_LIVE_VIEW_SHA"

echo
echo "=== SERVICES AFTER ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

rollback_needed=0
trap - ERR INT TERM

echo
echo "SOLVER_READY_INJURY20B_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
echo "NFL_INJURY_20B_STATUS=PASS"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
