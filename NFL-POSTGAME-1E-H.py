#!/usr/bin/env python3
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import tempfile
import difflib

ROOT = Path("/home/mwynn/nfl_data_engine")
MODULE = ROOT / "wfs_postgame_feedback.py"
APP = ROOT / "app.py"
BACKUP = ROOT / "backups" / "wfs_postgame_feedback.py.NFL-POSTGAME-1E-H.prechange.bak"

EXPECTED_MODULE_SHA = "701707c22178521519216dbb470d748c6542da5a293b2c6ddf684d859e2567c5"
EXPECTED_APP_SHA = "81d7eb2b1b82a3adcecdc13384e923df2c86f5fc85816fc41b185621c3437325"

OLD = '''            cand=f.execute("SELECT * FROM forecast_predictions WHERE game_id=? AND historical_proof_status='PASS' AND forecast_status LIKE 'READY%' ORDER BY captured_at_utc",(g["game_id"],)).fetchall()
'''

NEW = '''            cand=f.execute(
                "SELECT p.*, s.captured_at_utc, s.historical_proof_status, s.snapshot_status "
                "FROM forecast_predictions p "
                "JOIN forecast_snapshots s ON s.snapshot_id = p.snapshot_id "
                "WHERE p.game_id=? "
                "AND s.historical_proof_status='PASS' "
                "AND p.forecast_status LIKE 'READY%' "
                "ORDER BY s.captured_at_utc, p.snapshot_id",
                (g["game_id"],),
            ).fetchall()
'''

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(msg):
    print(f"FAIL={msg}")
    sys.exit(1)

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-H")
    print("POSTGAME FORECAST LEDGER QUERY FIX — FAIL CLOSED")
    print("=" * 76)

    if not MODULE.is_file():
        fail(f"missing module: {MODULE}")
    if not APP.is_file():
        fail(f"missing app.py: {APP}")

    module_sha = sha(MODULE)
    app_sha = sha(APP)
    print(f"PRE_MODULE_SHA256={module_sha}")
    print(f"APP_SHA256={app_sha}")

    if module_sha != EXPECTED_MODULE_SHA:
        fail("current feedback module hash mismatch")
    if app_sha != EXPECTED_APP_SHA:
        fail("app.py hash mismatch; refusing module-only change")

    original = MODULE.read_text()
    count = original.count(OLD)
    print(f"BAD_QUERY_ANCHOR_COUNT={count}")
    if count != 1:
        fail(f"bad query anchor count={count}")

    candidate = original.replace(OLD, NEW, 1)

    checks = {
        "FORECAST_JOIN_PRESENT": "JOIN forecast_snapshots s" in candidate,
        "EXACT_SNAPSHOT_JOIN": "s.snapshot_id = p.snapshot_id" in candidate,
        "PROOF_FROM_SNAPSHOT": "s.historical_proof_status='PASS'" in candidate,
        "STATUS_FROM_PREDICTION": "p.forecast_status LIKE 'READY%'" in candidate,
        "CAPTURE_FROM_SNAPSHOT": "s.captured_at_utc" in candidate,
        "EXACT_GAME_ID": "WHERE p.game_id=?" in candidate,
        "OLD_BAD_QUERY_REMOVED": "FROM forecast_predictions WHERE game_id=? AND historical_proof_status" not in candidate,
    }

    for name, ok in checks.items():
        print(f"{name}={str(ok).upper()}")
        if not ok:
            fail(f"semantic assertion failed: {name}")

    with tempfile.TemporaryDirectory(prefix="wfs_1eh_") as td:
        tmp = Path(td) / "wfs_postgame_feedback.py"
        tmp.write_text(candidate)
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(tmp)],
            text=True,
            capture_output=True,
        )
        print(f"CANDIDATE_COMPILE_EXIT={proc.returncode}")
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            fail("candidate compile failed")

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP.exists():
        backup_sha = sha(BACKUP)
        if backup_sha != EXPECTED_MODULE_SHA:
            fail("existing backup hash mismatch")
        print("BACKUP_ACTION=EXISTING_VALID")
    else:
        shutil.copy2(MODULE, BACKUP)
        print("BACKUP_ACTION=CREATED")

    print(f"BACKUP_PATH={BACKUP}")
    print(f"BACKUP_SHA256={sha(BACKUP)}")

    staged = MODULE.with_name("wfs_postgame_feedback.py.NFL-POSTGAME-1E-H.tmp")
    staged.write_text(candidate)
    staged.replace(MODULE)

    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(MODULE)],
        text=True,
        capture_output=True,
    )
    print(f"PRODUCTION_COMPILE_EXIT={proc.returncode}")
    if proc.returncode != 0:
        shutil.copy2(BACKUP, MODULE)
        print("ROLLBACK=PERFORMED")
        print(proc.stdout)
        print(proc.stderr)
        fail("production compile failed; rolled back")

    print(f"POST_MODULE_SHA256={sha(MODULE)}")
    print(f"APP_SHA256_UNCHANGED={sha(APP)}")

    print("\n=== UNIFIED DIFF ===")
    for line in difflib.unified_diff(
        original.splitlines(),
        candidate.splitlines(),
        fromfile="wfs_postgame_feedback.py.before",
        tofile="wfs_postgame_feedback.py.after",
        lineterm="",
        n=5,
    ):
        print(line)

    print("\nNFL_POSTGAME_1E_H_STATUS=PASS")
    print("GAME_GRADER_AUTHORITY_ALIGNED=TRUE")
    print("APP_WRITES=0")
    print("DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
