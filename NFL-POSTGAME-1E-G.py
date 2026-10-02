#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1E-G
Preserve Data Center Admin workspace selection through page-routing block.

Root cause:
- mode defaults to Public
- Data Center workspace radio can set Admin
- later non-Build branch unconditionally reset mode to Public

Fix:
- remove only the redundant later `mode = "Public"` reset
- all non-Build/non-Data-Center pages remain Public because mode already defaults to Public

Fail-closed:
- exact current app.py hash required
- exact reset anchor required once
- candidate compiles before write
- rollback backup created
- production compiles after write
- automatic rollback on compile failure

No service restart. No database/solver/updater/LIVE/injury/cron changes.
"""

from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import tempfile
import difflib

ROOT = Path("/home/mwynn/nfl_data_engine")
APP = ROOT / "app.py"
BACKUP = ROOT / "backups" / "app.py.NFL-POSTGAME-1E-G.prechange.bak"

EXPECTED_APP_SHA = "e6c63c80d4fa75ae0ec568ed269f5d9274757950d9418a082b2d2e0507055aed"

OLD = """    else:
        mode = "Public"
        selected_slate = next(
            (s for s in slates if normalize_slate(s) == "main"),
            slates[0],
        )
"""

NEW = """    else:
        selected_slate = next(
            (s for s in slates if normalize_slate(s) == "main"),
            slates[0],
        )
"""

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(msg):
    print(f"FAIL={msg}")
    sys.exit(1)

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-G")
    print("PRESERVE DATA CENTER ADMIN MODE — FAIL CLOSED")
    print("=" * 76)

    if not APP.is_file():
        fail(f"missing app.py: {APP}")

    pre_sha = sha(APP)
    print(f"PRE_APP_SHA256={pre_sha}")
    if pre_sha != EXPECTED_APP_SHA:
        fail("current app.py hash mismatch")

    original = APP.read_text()
    anchor_count = original.count(OLD)
    print(f"RESET_BLOCK_ANCHOR_COUNT={anchor_count}")
    if anchor_count != 1:
        fail(f"reset block anchor count={anchor_count}")

    candidate = original.replace(OLD, NEW, 1)

    checks = {
        "INITIAL_PUBLIC_DEFAULT_PRESENT": '    mode = "Public"' in candidate,
        "DATA_CENTER_WORKSPACE_SELECTOR_PRESENT": 'if page in {"🏗 Build Lineups", "🏈 NFL Data Center"}:' in candidate,
        "DATA_CENTER_ROUTE_PRESENT": 'render_data_center(mode)' in candidate,
        "ADMIN_FEEDBACK_GATE_PRESENT": 'if workspace_mode == "Admin":' in candidate,
        "LATE_PUBLIC_RESET_REMOVED": OLD not in candidate,
        "BUILD_LINEUPS_BRANCH_PRESENT": 'if page == "🏗 Build Lineups":' in candidate,
    }

    for name, ok in checks.items():
        print(f"{name}={str(ok).upper()}")
        if not ok:
            fail(f"semantic assertion failed: {name}")

    with tempfile.TemporaryDirectory(prefix="wfs_1eg_") as td:
        tmp = Path(td) / "app.py"
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
        if backup_sha != EXPECTED_APP_SHA:
            fail("existing 1E-G backup hash mismatch")
        print("BACKUP_ACTION=EXISTING_VALID")
    else:
        shutil.copy2(APP, BACKUP)
        print("BACKUP_ACTION=CREATED")

    print(f"BACKUP_PATH={BACKUP}")
    print(f"BACKUP_SHA256={sha(BACKUP)}")

    staged = APP.with_name("app.py.NFL-POSTGAME-1E-G.tmp")
    staged.write_text(candidate)
    staged.replace(APP)

    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(APP)],
        text=True,
        capture_output=True,
    )
    print(f"PRODUCTION_COMPILE_EXIT={proc.returncode}")
    if proc.returncode != 0:
        shutil.copy2(BACKUP, APP)
        print("ROLLBACK=PERFORMED")
        print(proc.stdout)
        print(proc.stderr)
        fail("production compile failed; rolled back")

    print(f"POST_APP_SHA256={sha(APP)}")

    print("\n=== UNIFIED DIFF ===")
    for line in difflib.unified_diff(
        original.splitlines(),
        candidate.splitlines(),
        fromfile="app.py.before",
        tofile="app.py.after",
        lineterm="",
        n=5,
    ):
        print(line)

    print("\nNFL_POSTGAME_1E_G_STATUS=PASS")
    print("DATA_CENTER_ADMIN_MODE_PRESERVED=TRUE")
    print("NON_DATA_CENTER_DEFAULT_PUBLIC=TRUE")
    print("PUBLIC_SOLVER_CHANGE=FALSE")
    print("DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
