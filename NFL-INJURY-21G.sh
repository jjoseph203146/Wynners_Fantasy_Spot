#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mwynn/nfl_data_engine"
PY="$ROOT/venv/bin/python"
LIVE="$ROOT/live_safe_poll.py"
LAUNCHER="$ROOT/run_updater.sh"
BACKUP_DIR="$ROOT/backups/injury_consensus"

EXPECTED_LIVE_SHA="96d3fffbc3cbbc51f2675e0c231cfa528bcba1dde23f807fcca0ea7f81057673"
EXPECTED_LAUNCHER_SHA="ec0941b870192f0f8fa47ac76c93f0c667e903616ca362546ffd7dc4591cf1d2"
EXPECTED_GATE_SHA="0d329611d7c628cafccf5339f2ecaac1fd2c0ff68760ba05cf4ec6c018e5c65a"
EXPECTED_ORCH_SHA="7767f05398677febfc303eb41626d23b8f7e11aee24b2144d5a0bb6294aab816"
EXPECTED_INGEST_SHA="f92c910722e05910dfc47b9e38821a7027d245a6b4d412037559c58e48a4acd4"

cd "$ROOT"
mkdir -p "$BACKUP_DIR"

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
echo "WFS NFL — NFL-INJURY-21G"
echo "SHARED UPDATER/LIVE LOCK CONTRACT DEPLOYMENT"
echo "======================================================================"

echo
echo "=== PREFLIGHT HASH CONTRACT ==="
require_hash "$LIVE" "$EXPECTED_LIVE_SHA"
require_hash "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_INGEST_SHA"

echo
echo "=== SERVICES BEFORE ==="
echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

CRON_BEFORE="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "CRON_BEFORE=$CRON_BEFORE"

if flock -n "$ROOT/nfl_updater.lock" -c true; then
    echo "UPDATER_LOCK_PREFLIGHT=FREE"
else
    echo "FAIL | updater lock busy; retry after current updater finishes"
    exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LIVE_BACKUP="$BACKUP_DIR/live_safe_poll_before_injury21g_${TS}.py"
LAUNCHER_BACKUP="$BACKUP_DIR/run_updater_before_injury21g_${TS}.sh"
LIVE_CAND="$ROOT/.live_safe_poll_injury21g.$$"
LAUNCHER_CAND="$ROOT/.run_updater_injury21g.$$"

cp -a "$LIVE" "$LIVE_BACKUP"
cp -a "$LAUNCHER" "$LAUNCHER_BACKUP"
cp -a "$LIVE" "$LIVE_CAND"
cp -a "$LAUNCHER" "$LAUNCHER_CAND"

echo "LIVE_BACKUP=$LIVE_BACKUP"
echo "LAUNCHER_BACKUP=$LAUNCHER_BACKUP"

"$PY" - "$LIVE_CAND" "$LAUNCHER_CAND" <<'PY'
from pathlib import Path
import sys

live = Path(sys.argv[1])
launcher = Path(sys.argv[2])

s = live.read_text()

old = '''LOCK_FILE = (
    ROOT / ".wfs_live_safe_poll.lock"
)
'''
new = '''LOCK_FILE = (
    ROOT / ".wfs_live_safe_poll.lock"
)

UPDATER_LOCK_FILE = (
    ROOT / "nfl_updater.lock"
)
'''
if s.count(old) != 1:
    raise SystemExit(f"FAIL | LIVE path anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = '''    completed = subprocess.run(
        command,
        cwd=str(
            ROOT
        ),
        capture_output=True,
        text=True,
        check=False,
    )
'''
new = '''    updater_lock_handle = (
        UPDATER_LOCK_FILE.open(
            "a+"
        )
    )

    emit(
        "WAITING_FOR_UPDATER_SHARED_LOCK=1"
        f" | path={UPDATER_LOCK_FILE}"
    )

    fcntl.flock(
        updater_lock_handle.fileno(),
        fcntl.LOCK_SH,
    )

    emit(
        "UPDATER_SHARED_LOCK_ACQUIRED=1"
        f" | path={UPDATER_LOCK_FILE}"
    )

    try:
        completed = subprocess.run(
            command,
            cwd=str(
                ROOT
            ),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        fcntl.flock(
            updater_lock_handle.fileno(),
            fcntl.LOCK_UN,
        )
        updater_lock_handle.close()

        emit(
            "UPDATER_SHARED_LOCK_RELEASED=1"
            f" | path={UPDATER_LOCK_FILE}"
        )
'''
if s.count(old) != 1:
    raise SystemExit(f"FAIL | LIVE subprocess anchor count={s.count(old)}")
s = s.replace(old, new, 1)
live.write_text(s)

t = launcher.read_text()
old = '''if ! flock -n 9; then
    log "Updater skipped because another instance is running."
    log "Finished: $(date)"
    log ""
    exit 1
fi
'''
new = '''# Wait up to 60 seconds for an in-flight LIVE read cycle to release
# its shared nfl_updater.lock. Once acquired exclusively, this lock
# protects the complete updater + injury chain from new LIVE DB reads.
if ! flock -w 60 9; then
    log "Updater skipped because shared updater/LIVE lock remained busy for 60 seconds."
    log "Finished: $(date)"
    log ""
    exit 1
fi
'''
if t.count(old) != 1:
    raise SystemExit(f"FAIL | launcher flock anchor count={t.count(old)}")
t = t.replace(old, new, 1)
launcher.write_text(t)

print("CANDIDATE_BUILD=PASS")
PY

chmod --reference="$LIVE" "$LIVE_CAND"
chmod --reference="$LAUNCHER" "$LAUNCHER_CAND"

echo
echo "=== CANDIDATE SYNTAX ==="
"$PY" -m py_compile "$LIVE_CAND"
bash -n "$LAUNCHER_CAND"
echo "LIVE_CANDIDATE_COMPILE=PASS"
echo "LAUNCHER_CANDIDATE_SYNTAX=PASS"

NEW_LIVE_SHA="$(hash_of "$LIVE_CAND")"
NEW_LAUNCHER_SHA="$(hash_of "$LAUNCHER_CAND")"
echo "NEW_LIVE_SAFE_POLL_SHA=$NEW_LIVE_SHA"
echo "NEW_RUN_UPDATER_SHA=$NEW_LAUNCHER_SHA"

echo
echo "=== STRUCTURAL PROOF ==="
"$PY" - "$LIVE_CAND" "$LAUNCHER_CAND" <<'PY'
from pathlib import Path
import sys

live = Path(sys.argv[1]).read_text()
launcher = Path(sys.argv[2]).read_text()

required_live = [
    'ROOT / "nfl_updater.lock"',
    'fcntl.LOCK_SH',
    'WAITING_FOR_UPDATER_SHARED_LOCK=1',
    'UPDATER_SHARED_LOCK_ACQUIRED=1',
    'UPDATER_SHARED_LOCK_RELEASED=1',
]
missing = [x for x in required_live if x not in live]
if missing:
    raise SystemExit("FAIL | LIVE structural tokens missing: " + repr(missing))

acq = live.index('UPDATER_SHARED_LOCK_ACQUIRED=1')
run = live.index('completed = subprocess.run(', acq)
rel = live.index('UPDATER_SHARED_LOCK_RELEASED=1', run)
if not acq < run < rel:
    raise SystemExit("FAIL | LIVE lock ordering")

if 'flock -w 60 9' not in launcher:
    raise SystemExit("FAIL | launcher bounded exclusive wait missing")

print("LIVE_SHARED_LOCK_WRAPS_GATE=PASS")
print("UPDATER_EXCLUSIVE_WAIT_CONTRACT=PASS")
print("LOCK_ORDERING_PROOF=PASS")
PY

echo
echo "=== SYNTHETIC LOCK CONTRACT ==="
TEST_LOCK="$ROOT/.injury21g_lock_test.$$"
: > "$TEST_LOCK"

(
    exec 8>"$TEST_LOCK"
    flock -s 8
    : > "$TEST_LOCK.ready"
    sleep 2
) &
HOLDER=$!

for _ in $(seq 1 50); do
    [[ -f "$TEST_LOCK.ready" ]] && break
    sleep 0.05
done

if flock -n "$TEST_LOCK" -c true; then
    kill "$HOLDER" 2>/dev/null || true
    wait "$HOLDER" 2>/dev/null || true
    rm -f "$TEST_LOCK" "$TEST_LOCK.ready"
    echo "FAIL | exclusive escaped shared holder"
    exit 1
else
    echo "SHARED_BLOCKS_EXCLUSIVE=PASS"
fi
wait "$HOLDER"
rm -f "$TEST_LOCK.ready"

if flock -n "$TEST_LOCK" -c true; then
    echo "EXCLUSIVE_AFTER_SHARED_RELEASE=PASS"
else
    rm -f "$TEST_LOCK"
    echo "FAIL | exclusive unavailable after shared release"
    exit 1
fi

"$PY" - "$TEST_LOCK" <<'PY' &
import fcntl, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
h = p.open("a+")
fcntl.flock(h.fileno(), fcntl.LOCK_EX)
pathlib.Path(str(p) + ".ready").write_text("ready")
time.sleep(2)
fcntl.flock(h.fileno(), fcntl.LOCK_UN)
h.close()
PY
HOLDER=$!

for _ in $(seq 1 50); do
    [[ -f "$TEST_LOCK.ready" ]] && break
    sleep 0.05
done

"$PY" - "$TEST_LOCK" <<'PY'
import fcntl, pathlib, sys
p = pathlib.Path(sys.argv[1])
h = p.open("a+")
try:
    fcntl.flock(h.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
except BlockingIOError:
    print("EXCLUSIVE_BLOCKS_SHARED=PASS")
else:
    fcntl.flock(h.fileno(), fcntl.LOCK_UN)
    raise SystemExit("FAIL | shared escaped exclusive holder")
finally:
    h.close()
PY

wait "$HOLDER"
rm -f "$TEST_LOCK" "$TEST_LOCK.ready"
echo "SYNTHETIC_LOCK_PROOF=PASS"

echo
echo "=== INSTALL ==="
systemctl stop wfs-nfl-live.service
echo "LIVE_STOP_STATE=$(systemctl is-active wfs-nfl-live.service || true)"

cp -a "$LIVE_CAND" "$LIVE"
cp -a "$LAUNCHER_CAND" "$LAUNCHER"
rm -f "$LIVE_CAND" "$LAUNCHER_CAND"

"$PY" -m py_compile "$LIVE"
bash -n "$LAUNCHER"

echo
echo "=== RESTART LIVE ==="
if ! systemctl start wfs-nfl-live.service; then
    echo "FAIL | LIVE start failed; rolling back"
    systemctl stop wfs-nfl-live.service || true
    cp -a "$LIVE_BACKUP" "$LIVE"
    cp -a "$LAUNCHER_BACKUP" "$LAUNCHER"
    "$PY" -m py_compile "$LIVE"
    bash -n "$LAUNCHER"
    systemctl start wfs-nfl-live.service || true
    echo "ROLLBACK=PERFORMED"
    exit 1
fi

sleep 3
if [[ "$(systemctl is-active wfs-nfl-live.service)" != "active" ]]; then
    echo "FAIL | LIVE not active after restart; rolling back"
    systemctl stop wfs-nfl-live.service || true
    cp -a "$LIVE_BACKUP" "$LIVE"
    cp -a "$LAUNCHER_BACKUP" "$LAUNCHER"
    "$PY" -m py_compile "$LIVE"
    bash -n "$LAUNCHER"
    systemctl start wfs-nfl-live.service || true
    echo "ROLLBACK=PERFORMED"
    exit 1
fi

echo
echo "=== POST-INSTALL ==="
echo "$(hash_of "$LIVE")  $LIVE"
echo "$(hash_of "$LAUNCHER")  $LAUNCHER"
require_hash "$ROOT/live_concurrency_gate.py" "$EXPECTED_GATE_SHA"
require_hash "$ROOT/live_orchestrator.py" "$EXPECTED_ORCH_SHA"
require_hash "$ROOT/live_ingest.py" "$EXPECTED_INGEST_SHA"

CRON_AFTER="$(crontab -l | grep '/home/mwynn/nfl_data_engine/run_updater.sh' || true)"
echo "CRON_AFTER=$CRON_AFTER"
[[ "$CRON_AFTER" == "$CRON_BEFORE" ]] || {
    echo "FAIL | cron changed unexpectedly"
    exit 1
}
echo "CRON_UNCHANGED=PASS"

echo "wfs.service=$(systemctl is-active wfs.service)"
echo "wfs-nfl-live.service=$(systemctl is-active wfs-nfl-live.service)"

echo
echo "NFL_INJURY_21G_STATUS=PASS"
echo "NEW_LIVE_SAFE_POLL_SHA=$NEW_LIVE_SHA"
echo "NEW_RUN_UPDATER_SHA=$NEW_LAUNCHER_SHA"
echo "LIVE_SERVICE_RESTARTED=TRUE"
echo "WFS_SERVICE_RESTARTED=FALSE"
