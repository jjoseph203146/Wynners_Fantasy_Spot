#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
TARGET="$ROOT/injury_consensus.py"
STAGED="$ROOT/injury_consensus_v2_compat.py"
LOCK="$ROOT/nfl_updater.lock"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_OLD_CONSENSUS_SHA="705a5bb91572f3cca5e3fc79d0a8cc378eb11e3021c8fdafcf8a7937dfe5fc9c"
EXPECTED_INGEST_SHA="8b8cd6f463c1ef354d3e20b54ffa53175e2ef51d221ba9bfe75a0598b6f006c2"
EXPECTED_STAGED_SHA="e4bb68e5b839cb091ca7ef6e8c25968796443f9f7c1b9cdc0a9d69609cb77c09"

EXPECTED_APP_SHA="07cde83abf3904ad30b14d0917eeaaba312528ef546fbb1a0dc50064bacf3a7c"
EXPECTED_SOLVER_SHA="753a85e8efd73407d88dbf501b4bdad281fb5241a85c00a8b690b78cbd51dd95"
EXPECTED_V5_SHA="4a0b1e864cc13678bdcbb4df3480b94f83a9c999b935b3a72d4bb95949ca9793"
EXPECTED_READY_SHA="9ebcf4731cf4e853d7c5af04d0a2d802e79204ad268cce3e423445e6f128e370"
EXPECTED_UPDATER_SHA="a9637bcb077d62c35cfabf48c5a17aa293c7176f35302f7020d5a0aaef41096a"
EXPECTED_LIVE_INGEST_SHA="f92c910722e05910dfc47b9e38821a7027d245a6b4d412037559c58e48a4acd4"
EXPECTED_LIVE_SAFE_SHA="96d3fffbc3cbbc51f2675e0c231cfa528bcba1dde23f807fcca0ea7f81057673"
EXPECTED_LIVE_ORCH_SHA="7767f05398677febfc303eb41626d23b8f7e11aee24b2144d5a0bb6294aab816"
EXPECTED_LIVE_GATE_SHA="0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a"
EXPECTED_LIVE_VIEW_SHA="11ae0beabf6de740ee920c15cf0f50be4219d9a7155553f050c7093b24bb67b6"

cd "$ROOT"
mkdir -p "$BACKUP_DIR"

echo "======================================================================"
echo "WFS NFL — NFL-INJURY-19 CONTROLLED WRITE"
echo "CONSENSUS V2 COMPATIBILITY MIGRATION"
echo "======================================================================"

require_hash() {
    local file="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    printf '%s  %s\n' "$actual" "$(basename "$file")"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL | hash mismatch: $file"
        echo "EXPECTED=$expected"
        echo "ACTUAL=$actual"
        exit 1
    fi
}

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$TARGET" "$EXPECTED_OLD_CONSENSUS_SHA"
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$STAGED" "$EXPECTED_STAGED_SHA"
require_hash "$ROOT/app.py" "$EXPECTED_APP_SHA"
require_hash "$ROOT/fanduel_nfl_ui_solver.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/fanduel_slate_projection_attach_v5.py" "$EXPECTED_V5_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/wfs_live_view.py" "$EXPECTED_LIVE_VIEW_SHA"

echo
echo "=== SERVICES BEFORE ==="
printf 'wfs.service='
systemctl is-active wfs.service
printf 'wfs-nfl-live.service='
systemctl is-active wfs-nfl-live.service

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FAIL | nfl updater lock is busy: $LOCK"
    exit 1
fi
echo "PASS | acquired nfl updater lock"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
MODULE_BACKUP="$BACKUP_DIR/injury_consensus_before_injury19_${TS}.py"
DB_BACKUP="$BACKUP_DIR/nfl_before_injury19_${TS}.db"

cp -a "$TARGET" "$MODULE_BACKUP"

"$PY" - "$DB" "$DB_BACKUP" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
    s.backup(d)
print("SQLITE_BACKUP=PASS")
PY

echo "MODULE_BACKUP=$MODULE_BACKUP"
echo "DB_BACKUP=$DB_BACKUP"

rollback_needed=1
rollback() {
    local rc=$?
    if [[ "$rollback_needed" -eq 1 ]]; then
        echo
        echo "!!! NFL-INJURY-19 FAILURE — RESTORING PRE-STAGE BASELINE !!!"
        cp -a "$MODULE_BACKUP" "$TARGET"
        cp -a "$DB_BACKUP" "$DB"
        echo "ROLLBACK_MODULE_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
        "$PY" - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    print("ROLLBACK_DB_INTEGRITY=" + c.execute("PRAGMA integrity_check").fetchone()[0])
PY
    fi
    exit "$rc"
}
trap rollback ERR INT TERM

echo
echo "=== DATABASE HEALTH BEFORE ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    print("DB_INTEGRITY_BEFORE=" + c.execute("PRAGMA integrity_check").fetchone()[0])
    cols = c.execute("PRAGMA table_info(injury_consensus_current)").fetchall()
    print("CONSENSUS_COLUMNS_BEFORE=" + ",".join(r[1] for r in cols))
    print("CONSENSUS_ROWS_BEFORE=" + str(c.execute(
        "SELECT COUNT(*) FROM injury_consensus_current"
    ).fetchone()[0]))
PY

echo
echo "=== INSTALL STAGED CONSENSUS ==="
cp -a "$STAGED" "$TARGET"
require_hash "$TARGET" "$EXPECTED_STAGED_SHA"
"$PY" -m py_compile "$TARGET"
echo "PY_COMPILE=PASS"

semantic_hash() {
"$PY" - "$DB" <<'PY'
import hashlib, json, sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(db) as c:
    c.row_factory=sqlite3.Row
    cols=[r[1] for r in c.execute("PRAGMA table_info(injury_consensus_current)")]
    ignore={"updated_at","last_updated"}
    use=[x for x in cols if x not in ignore]
    sql="SELECT " + ",".join(f'"{x}"' for x in use) + \
        " FROM injury_consensus_current ORDER BY season,week,game_type,team,gsis_id"
    rows=[dict(r) for r in c.execute(sql)]
payload=json.dumps(rows, sort_keys=True, separators=(",",":"), default=str).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

echo
echo "=== RUN 1 ==="
"$PY" "$TARGET"

HASH1="$(semantic_hash)"
echo "SEMANTIC_HASH_RUN1=$HASH1"

echo
echo "=== RUN 1 DATABASE PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
db=sys.argv[1]
with sqlite3.connect(db) as c:
    c.row_factory=sqlite3.Row
    cols={r[1] for r in c.execute("PRAGMA table_info(injury_consensus_current)")}
    required={
        "secondary_injury","injury_risk","last_updated",
        "structured_source","availability_risk","updated_at"
    }
    missing=sorted(required-cols)
    print("MISSING_COMPAT_COLUMNS=" + (",".join(missing) if missing else "<NONE>"))
    if missing:
        raise SystemExit("FAIL | compatibility columns missing")

    null_required=c.execute("""
        SELECT COUNT(*)
        FROM injury_consensus_current
        WHERE structured_source IS NULL
           OR availability_risk IS NULL
           OR updated_at IS NULL
           OR injury_risk IS NULL
           OR last_updated IS NULL
    """).fetchone()[0]
    print("NULL_REQUIRED_COMPAT_ROWS=" + str(null_required))
    if null_required:
        raise SystemExit("FAIL | required compatibility field is NULL")

    risk_mismatch=c.execute("""
        SELECT COUNT(*)
        FROM injury_consensus_current
        WHERE COALESCE(availability_risk,'') <> COALESCE(injury_risk,'')
    """).fetchone()[0]
    print("LEGACY_V2_RISK_MISMATCH_ROWS=" + str(risk_mismatch))
    if risk_mismatch:
        raise SystemExit("FAIL | availability_risk/injury_risk mismatch")

    hard_escape=c.execute("""
        SELECT COUNT(*)
        FROM injury_consensus_current
        WHERE UPPER(COALESCE(consensus_status,'')) IN ('OUT','DOUBTFUL','INACTIVE')
          AND injury_gate <> 'BLOCK'
    """).fetchone()[0]
    print("HARD_STATUS_ESCAPE_ROWS=" + str(hard_escape))
    if hard_escape:
        raise SystemExit("FAIL | hard status escaped BLOCK")

    secondary_block=c.execute("""
        SELECT COUNT(*)
        FROM injury_consensus_current
        WHERE structured_present=0
          AND injury_gate='BLOCK'
    """).fetchone()[0]
    print("SECONDARY_ONLY_BLOCK_ROWS=" + str(secondary_block))
    if secondary_block:
        raise SystemExit("FAIL | secondary-only signal created BLOCK")

    q_block=c.execute("""
        SELECT COUNT(*)
        FROM injury_consensus_current
        WHERE consensus_status='QUESTIONABLE'
          AND injury_gate <> 'ALLOW'
    """).fetchone()[0]
    print("QUESTIONABLE_BLOCK_ROWS=" + str(q_block))
    if q_block:
        raise SystemExit("FAIL | QUESTIONABLE not ALLOW")

    block_count=c.execute("""
        SELECT COUNT(*) FROM injury_consensus_current
        WHERE injury_gate='BLOCK'
    """).fetchone()[0]
    print("BLOCK_ROWS=" + str(block_count))

    secondary_only=c.execute("""
        SELECT COUNT(*) FROM injury_consensus_current
        WHERE structured_present=0 AND secondary_present=1
    """).fetchone()[0]
    print("SECONDARY_ONLY_ROWS=" + str(secondary_only))

    latest_mismatch=c.execute("""
        WITH latest AS (
            SELECT gsis_id, MAX(source_timestamp) AS max_ts
            FROM injury_news_signals
            WHERE source='FANDUEL_RESEARCH'
              AND source_timestamp IS NOT NULL
              AND TRIM(source_timestamp) <> ''
            GROUP BY gsis_id
        )
        SELECT COUNT(*)
        FROM injury_consensus_current c
        JOIN latest l ON l.gsis_id=c.gsis_id
        WHERE c.secondary_source='FANDUEL_RESEARCH'
          AND COALESCE(c.secondary_source_timestamp,'') <> COALESCE(l.max_ts,'')
    """).fetchone()[0]
    print("LATEST_FANDUEL_TIMESTAMP_MISMATCH_ROWS=" + str(latest_mismatch))
    if latest_mismatch:
        raise SystemExit("FAIL | consensus did not select latest FanDuel event")

    tua=c.execute("""
        SELECT player_name,gsis_id,team,position,
               structured_present,secondary_present,
               secondary_practice_status,
               secondary_availability_status,
               injury_gate,injury_risk,injury_gate_reason,
               secondary_source_timestamp
        FROM injury_consensus_current
        WHERE gsis_id='00-0036212'
    """).fetchall()
    print("TUA_ROWS=" + str(len(tua)))
    for row in tua:
        print("TUA_PROOF=" + repr(tuple(row)))
        if row["secondary_present"] != 1:
            raise SystemExit("FAIL | Tua missing secondary signal")
        if row["structured_present"] == 0 and row["injury_gate"] != "ALLOW":
            raise SystemExit("FAIL | Tua secondary-only row not ALLOW")
        if row["structured_present"] == 0:
            print("TUA_SECONDARY_ONLY_PROOF=PASS")
        else:
            print("TUA_SECONDARY_ONLY_PROOF=SUPERSEDED_BY_CURRENT_STRUCTURED_STATUS")

    print("CONSENSUS_SCHEMA_COMPAT_PROOF=PASS")
PY

echo
echo "=== RUN 2 / DETERMINISM ==="
"$PY" "$TARGET"

HASH2="$(semantic_hash)"
echo "SEMANTIC_HASH_RUN2=$HASH2"

if [[ "$HASH1" != "$HASH2" ]]; then
    echo "FAIL | semantic consensus changed between identical runs"
    exit 1
fi
echo "SEMANTIC_DETERMINISM=PASS"

echo
echo "=== FINAL DATABASE PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as c:
    integrity=c.execute("PRAGMA integrity_check").fetchone()[0]
    print("DB_INTEGRITY_AFTER=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | DB integrity")
    fk=c.execute("PRAGMA foreign_key_check").fetchall()
    print("DB_FOREIGN_KEY_VIOLATIONS=" + str(len(fk)))
    if fk:
        raise SystemExit("FAIL | foreign key violations")
    print("CONSENSUS_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM injury_consensus_current"
    ).fetchone()[0]))
    print("GATE_DISTRIBUTION=" + repr(c.execute("""
        SELECT injury_gate,COUNT(*)
        FROM injury_consensus_current
        GROUP BY injury_gate
        ORDER BY injury_gate
    """).fetchall()))
    print("SOURCE_DISTRIBUTION=" + repr(c.execute("""
        SELECT structured_present,secondary_present,COUNT(*)
        FROM injury_consensus_current
        GROUP BY structured_present,secondary_present
        ORDER BY structured_present,secondary_present
    """).fetchall()))
PY

echo
echo "=== PROTECTED HASHES AFTER ==="
require_hash "$ROOT/fanduel_injury_ingest.py" "$EXPECTED_INGEST_SHA"
require_hash "$ROOT/app.py" "$EXPECTED_APP_SHA"
require_hash "$ROOT/fanduel_nfl_ui_solver.py" "$EXPECTED_SOLVER_SHA"
require_hash "$ROOT/fanduel_slate_projection_attach_v5.py" "$EXPECTED_V5_SHA"
require_hash "$ROOT/fanduel_solver_ready_pool.py" "$EXPECTED_READY_SHA"
require_hash "$ROOT/updater.py" "$EXPECTED_UPDATER_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_LIVE_INGEST_SHA"
require_hash "$ROOT/live_safe_poll.py" "$EXPECTED_LIVE_SAFE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_LIVE_ORCH_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_LIVE_GATE_SHA"
require_hash "$ROOT/wfs_live_view.py" "$EXPECTED_LIVE_VIEW_SHA"

echo
echo "=== SERVICES AFTER ==="
WFS_STATUS="$(systemctl is-active wfs.service)"
LIVE_STATUS="$(systemctl is-active wfs-nfl-live.service)"
echo "wfs.service=$WFS_STATUS"
echo "wfs-nfl-live.service=$LIVE_STATUS"
[[ "$WFS_STATUS" == "active" ]]
[[ "$LIVE_STATUS" == "active" ]]

rollback_needed=0
trap - ERR INT TERM

echo
echo "CONSENSUS_V2_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
echo "NFL_INJURY_19_STATUS=PASS"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
