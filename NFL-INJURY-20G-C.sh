#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
DB="$ROOT/data/nfl.db"
TARGET="$ROOT/fanduel_solver_ready_pool.py"
STAGED="$ROOT/fanduel_solver_ready_pool_injury20e.py"
LOCK="$ROOT/nfl_updater.lock"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_OLD_SHA="9ebcf4731cf4e853d7c5af04d0a2d802e79204ad268cce3e423445e6f128e370"
EXPECTED_NEW_SHA="1b533cf42117f83688d9cbb18839c70b1c7202faef5ec03dc5d3cf6b63bf7c6f"
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
echo "WFS NFL — NFL-INJURY-20G-C CONTROLLED DEPLOYMENT"
echo "ONE PRODUCTION WRITE + WRITE-FREE DETERMINISM PROOF"
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
MODULE_BACKUP="$BACKUP_DIR/fanduel_solver_ready_pool_before_injury20g_${TS}.py"
DB_BACKUP="$BACKUP_DIR/nfl_before_injury20g_${TS}.db"
BASELINE_SNAPSHOT="$BACKUP_DIR/solver_ready_semantics_before_injury20g_${TS}.csv"

cp -a "$TARGET" "$MODULE_BACKUP"

"$PY" - "$DB" "$DB_BACKUP" "$BASELINE_SNAPSHOT" <<'PY'
import sqlite3, sys
import pandas as pd

db, backup, snapshot = sys.argv[1:4]

with sqlite3.connect(db) as s, sqlite3.connect(backup) as d:
    s.backup(d)

with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    df = pd.read_sql_query("""
        SELECT
            slate_slug_solver,
            player_solver,
            team_solver,
            salary_solver,
            solver_status,
            solver_eligible
        FROM fanduel_solver_ready_pool
        ORDER BY slate_slug_solver, player_solver, team_solver, salary_solver
    """, c)

df.to_csv(snapshot, index=False)
print("SQLITE_BACKUP=PASS")
print("BASELINE_SEMANTIC_ROWS=" + str(len(df)))
PY

echo "MODULE_BACKUP=$MODULE_BACKUP"
echo "DB_BACKUP=$DB_BACKUP"
echo "BASELINE_SNAPSHOT=$BASELINE_SNAPSHOT"

rollback_needed=1
rollback() {
  rc=$?
  if [[ "$rollback_needed" -eq 1 ]]; then
    echo "!!! FAILURE — ROLLING BACK NFL-INJURY-20G !!!"
    cp -a "$MODULE_BACKUP" "$TARGET"
    cp -a "$DB_BACKUP" "$DB"
    echo "ROLLBACK_TARGET_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
    echo "NFL_INJURY_20G_C_STATUS=ROLLED_BACK"
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
echo "=== SYNTHETIC PURE-FUNCTION INJURY GATE PROOF ==="
"$PY" - "$TARGET" <<'PY'
import importlib.util
import sys
import pandas as pd

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("inj20g_candidate", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

solver = pd.DataFrame([
    {
        "_is_dst": 0,
        "position_identity": "GSIS:00-TEST-BLOCK",
        "solver_status": "READY",
        "solver_eligible": 1,
    },
    {
        "_is_dst": 0,
        "position_identity": "GSIS:00-TEST-ALLOW",
        "solver_status": "READY",
        "solver_eligible": 1,
    },
    {
        "_is_dst": 0,
        "position_identity": "GSIS:00-TEST-NONE",
        "solver_status": "READY",
        "solver_eligible": 1,
    },
    {
        "_is_dst": 0,
        "position_identity": "GSIS:00-TEST-ALLOW",
        "solver_status": "BLOCKED_PROJECTION",
        "solver_eligible": 0,
    },
])

consensus = pd.DataFrame([
    {
        "gsis_id": "00-TEST-BLOCK",
        "injury_gate": "BLOCK",
        "consensus_status": "OUT",
        "injury_gate_reason": "SYNTHETIC_BLOCK",
        "injury_risk": "HIGH",
    },
    {
        "gsis_id": "00-TEST-ALLOW",
        "injury_gate": "ALLOW",
        "consensus_status": "QUESTIONABLE",
        "injury_gate_reason": "SYNTHETIC_ALLOW",
        "injury_risk": "ELEVATED",
    },
])

out = m.apply_injury_gate(solver, consensus)

assert out.loc[0, "injury_gsis_id"] == "00-TEST-BLOCK"
assert out.loc[0, "injury_gate"] == "BLOCK"
assert out.loc[0, "solver_status"] == "BLOCKED_INJURY"
assert int(out.loc[0, "solver_eligible"]) == 0

assert out.loc[1, "injury_gate"] == "ALLOW"
assert out.loc[1, "solver_status"] == "READY"
assert int(out.loc[1, "solver_eligible"]) == 1

assert int(out.loc[2, "injury_consensus_found"]) == 0
assert out.loc[2, "solver_status"] == "READY"
assert int(out.loc[2, "solver_eligible"]) == 1

assert out.loc[3, "injury_gate"] == "ALLOW"
assert out.loc[3, "solver_status"] == "BLOCKED_PROJECTION"
assert int(out.loc[3, "solver_eligible"]) == 0

assert m.canonical_gsis_from_identity("GSIS:00-1234567") == "00-1234567"
assert m.canonical_gsis_from_identity("00-1234567") == ""
assert m.canonical_gsis_from_identity("OTHER:00-1234567") == ""

print("SYNTHETIC_BLOCK_CASE=PASS")
print("SYNTHETIC_ALLOW_CASE=PASS")
print("SYNTHETIC_NO_CONSENSUS_CASE=PASS")
print("SYNTHETIC_PREBLOCKED_CASE=PASS")
print("STRICT_GSIS_PREFIX_CASE=PASS")
print("SYNTHETIC_INJURY_GATE_PROOF=PASS")
PY

echo
echo "=== WAIT FOR LIVE GATE QUIET WINDOW ==="
quiet=0
for i in $(seq 1 120); do
  if ! pgrep -f '[l]ive_concurrency_gate.py --season 2026' >/dev/null 2>&1; then
    quiet=1
    echo "LIVE_GATE_QUIET_WINDOW=PASS"
    break
  fi
  sleep 1
done

[[ "$quiet" -eq 1 ]] || {
  echo "FAIL | no LIVE gate quiet window observed within 120 seconds"
  exit 1
}

echo
echo "=== SINGLE PRODUCTION RUN ==="
"$PY" "$TARGET"

echo
echo "=== PRODUCTION DB PROOF ==="
PROD_HASH="$("$PY" - "$DB" "$BASELINE_SNAPSHOT" <<'PY'
import hashlib
import sqlite3
import sys
import pandas as pd

db, baseline_path = sys.argv[1:3]

KEY = [
    "slate_slug_solver",
    "player_solver",
    "team_solver",
    "salary_solver",
]

SEMANTIC = KEY + ["solver_status", "solver_eligible"]

def canonical_hash(df):
    x = df.copy()
    x = x.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    for c in x.columns:
        x[c] = x[c].fillna("").astype(str)
    payload = x.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()

baseline = pd.read_csv(baseline_path, dtype=str).fillna("")

with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
    post = pd.read_sql_query(
        "SELECT * FROM fanduel_solver_ready_pool",
        c,
    )

required = {
    "injury_consensus_found",
    "injury_gsis_id",
    "injury_gate",
    "injury_status",
    "injury_gate_reason",
    "injury_risk",
}
missing = sorted(required - set(post.columns))
print("MISSING_INJURY_AUDIT_COLUMNS=" + repr(missing))
if missing:
    raise SystemExit("FAIL | missing injury audit columns")

escapes = post[
    post["injury_gate"].fillna("").eq("BLOCK")
    & post["solver_eligible"].eq(1)
]
print("BLOCK_ESCAPE_ROWS=" + str(len(escapes)))
if len(escapes):
    raise SystemExit("FAIL | BLOCK injury escape")

blocked = post["solver_status"].eq("BLOCKED_INJURY").sum()
print("BLOCKED_INJURY_ROWS=" + str(int(blocked)))

attached = post["injury_consensus_found"].eq(1).sum()
print("CONSENSUS_ATTACHED_ROWS=" + str(int(attached)))
if attached < 1:
    raise SystemExit("FAIL | no injury consensus attachments")

malformed = post[
    post["injury_consensus_found"].eq(1)
    & (
        post["injury_gsis_id"].fillna("").eq("")
        | ~post["position_identity"].fillna("").str.startswith("GSIS:")
    )
]
print("ATTACHED_WITHOUT_CANONICAL_GSIS_ROWS=" + str(len(malformed)))
if len(malformed):
    raise SystemExit("FAIL | malformed attached GSIS")

mismatch = post[
    post["injury_consensus_found"].eq(1)
    & (
        post["injury_gsis_id"].fillna("")
        != post["position_identity"].fillna("").str[5:]
    )
]
print("GSIS_CANONICALIZATION_MISMATCH_ROWS=" + str(len(mismatch)))
if len(mismatch):
    raise SystemExit("FAIL | GSIS canonicalization mismatch")

tua = post[post["injury_gsis_id"].fillna("").eq("00-0036212")]
print("TUA_SOLVER_ROWS=" + str(len(tua)))
if tua.empty:
    raise SystemExit("FAIL | Tua overlap missing")

for _, r in tua.iterrows():
    if int(r["injury_consensus_found"]) != 1:
        raise SystemExit("FAIL | Tua consensus attach")
    if str(r["injury_gate"]) != "ALLOW":
        raise SystemExit("FAIL | Tua gate")
    if str(r["injury_gsis_id"]) != "00-0036212":
        raise SystemExit("FAIL | Tua GSIS")

print("TUA_OVERLAP_PROOF=PASS")

post_sem = post[SEMANTIC].copy()
post_sem["salary_solver"] = post_sem["salary_solver"].astype(str)

base_keys = baseline[KEY].astype(str)
post_keys = post_sem[KEY].astype(str)

b = baseline.copy()
p = post_sem.copy()
for c in SEMANTIC:
    b[c] = b[c].fillna("").astype(str)
    p[c] = p[c].fillna("").astype(str)

merged = b.merge(
    p,
    on=KEY,
    how="outer",
    suffixes=("_before", "_after"),
    indicator=True,
)

missing_rows = merged[merged["_merge"] != "both"]
print("BASELINE_KEY_DRIFT_ROWS=" + str(len(missing_rows)))
if len(missing_rows):
    print(missing_rows.head(20).to_string(index=False))
    raise SystemExit("FAIL | row/key drift against baseline")

status_drift = merged[
    (merged["solver_status_before"] != merged["solver_status_after"])
    | (merged["solver_eligible_before"] != merged["solver_eligible_after"])
]

# Any status change is allowed only for a real exact BLOCK attachment.
post_block_keys = post.loc[
    post["injury_gate"].fillna("").eq("BLOCK"),
    KEY,
].copy()
for c in KEY:
    post_block_keys[c] = post_block_keys[c].fillna("").astype(str)

if not status_drift.empty:
    allowed = status_drift.merge(post_block_keys.drop_duplicates(), on=KEY, how="left", indicator="block_merge")
    illegal = allowed[allowed["block_merge"] != "both"]
else:
    illegal = status_drift

print("NON_BLOCK_SEMANTIC_DRIFT_ROWS=" + str(len(illegal)))
if len(illegal):
    print(illegal.head(20).to_string(index=False))
    raise SystemExit("FAIL | non-BLOCK solver semantics changed")

print("BASELINE_REGRESSION_PROOF=PASS")
print("PRODUCTION_SEMANTIC_HASH=" + canonical_hash(post_sem))
PY
)"

echo "$PROD_HASH"
PROD_HASH_VALUE="$(printf '%s\n' "$PROD_HASH" | awk -F= '/^PRODUCTION_SEMANTIC_HASH=/{print $2}')"
[[ -n "$PROD_HASH_VALUE" ]] || {
  echo "FAIL | production semantic hash missing"
  exit 1
}

echo
echo "=== WRITE-FREE DETERMINISM RUN ==="
DRY_HASH="$("$PY" - "$TARGET" "$DB" <<'PY'
import hashlib
import importlib.util
import sqlite3
import sys

import pandas as pd

module_path, db = sys.argv[1:3]

spec = importlib.util.spec_from_file_location("inj20g_candidate", module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

KEY = [
    "slate_slug_solver",
    "player_solver",
    "team_solver",
    "salary_solver",
]
SEMANTIC = KEY + ["solver_status", "solver_eligible"]

def canonical_hash(df):
    x = df.copy()
    x = x.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    for c in x.columns:
        x[c] = x[c].fillna("").astype(str)
    payload = x.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()

with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
    source = pd.read_sql_query(
        f'SELECT * FROM "{m.SOURCE_TABLE}"',
        conn,
    )
    offense, offense_source = m.load_offense_source(conn)
    source_schema = m.resolve_source_schema(source)
    offense_schema = m.resolve_offense_schema(offense)
    offense_lookup = m.prepare_offense_lookup(offense, offense_schema)
    injury_consensus = m.load_injury_consensus(conn)

solver = m.build_solver_pool(source, source_schema, offense_lookup)
solver = m.apply_injury_gate(solver, injury_consensus)
exported = m.clean_export(solver)

print("WRITE_FREE_SOURCE_ROWS=" + str(len(source)))
print("WRITE_FREE_OFFENSE_ROWS=" + str(len(offense)))
print("WRITE_FREE_SOLVER_ROWS=" + str(len(exported)))
print("WRITE_FREE_CONSENSUS_ATTACHMENTS=" + str(
    int(exported["injury_consensus_found"].eq(1).sum())
))
print("WRITE_FREE_BLOCK_ESCAPES=" + str(
    int((
        exported["injury_gate"].fillna("").eq("BLOCK")
        & exported["solver_eligible"].eq(1)
    ).sum())
))
print("WRITE_FREE_SEMANTIC_HASH=" + canonical_hash(exported[SEMANTIC]))
PY
)"

echo "$DRY_HASH"
DRY_HASH_VALUE="$(printf '%s\n' "$DRY_HASH" | awk -F= '/^WRITE_FREE_SEMANTIC_HASH=/{print $2}')"
[[ -n "$DRY_HASH_VALUE" ]] || {
  echo "FAIL | write-free semantic hash missing"
  exit 1
}

[[ "$PROD_HASH_VALUE" == "$DRY_HASH_VALUE" ]] || {
  echo "FAIL | write-free determinism mismatch"
  echo "PRODUCTION=$PROD_HASH_VALUE"
  echo "WRITE_FREE=$DRY_HASH_VALUE"
  exit 1
}
echo "WRITE_FREE_DETERMINISM=PASS"

echo
echo "=== FINAL DB PROOF ==="
"$PY" - "$DB" <<'PY'
import sqlite3, sys

with sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True) as c:
    integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
    print("DB_INTEGRITY_AFTER=" + integrity)
    if integrity != "ok":
        raise SystemExit("FAIL | DB integrity")

    fk = c.execute("PRAGMA foreign_key_check").fetchall()
    print("DB_FOREIGN_KEY_VIOLATIONS=" + str(len(fk)))
    if fk:
        raise SystemExit("FAIL | foreign key violations")

    print("SOLVER_READY_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool"
    ).fetchone()[0]))
    print("SOLVER_ELIGIBLE_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool WHERE solver_eligible=1"
    ).fetchone()[0]))
    print("BLOCKED_INJURY_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool WHERE solver_status='BLOCKED_INJURY'"
    ).fetchone()[0]))
    print("CONSENSUS_ATTACHED_ROWS=" + str(c.execute(
        "SELECT COUNT(*) FROM fanduel_solver_ready_pool WHERE injury_consensus_found=1"
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
echo "SOLVER_READY_INJURY20G_SHA=$(sha256sum "$TARGET" | awk '{print $1}')"
echo "NFL_INJURY_20G_C_STATUS=PASS"
echo "NO_SERVICE_RESTART_PERFORMED=TRUE"
