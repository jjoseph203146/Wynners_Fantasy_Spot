#!/usr/bin/env python3
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import tempfile
import difflib

ROOT = Path("/home/mwynn/nfl_data_engine")
APP = ROOT / "app.py"
BACKUP = ROOT / "backups" / "app.py.NFL-POSTGAME-1E-E.prechange.bak"

EXPECTED_APP_SHA = "a98229b3c7832ad38ae064b3b7fcf0b4cb801b17c23427720a8494f405a8a301"

OLD = """    if page == "🏗 Build Lineups":
        admin_emails = {
            str(email).strip().lower()
            for email in st.secrets.get("admin", {}).get("emails", [])
        }
        current_email = str(st.user.get("email") or "").strip().lower()
        is_admin = current_email in admin_emails

        workspace_options = ["Public", "Admin"] if is_admin else ["Public"]

        mode = st.radio(
            "Workspace",
            workspace_options,
            horizontal=True,
        )

        if mode == "Admin" and not is_admin:
            st.error("Admin access is not authorized for this account.")
            st.stop()

        default_main = next(
"""

NEW = """    admin_emails = {
        str(email).strip().lower()
        for email in st.secrets.get("admin", {}).get("emails", [])
    }
    current_email = str(st.user.get("email") or "").strip().lower()
    is_admin = current_email in admin_emails

    mode = "Public"

    if page in {"🏗 Build Lineups", "🏈 NFL Data Center"}:
        workspace_options = ["Public", "Admin"] if is_admin else ["Public"]

        mode = st.radio(
            "Workspace",
            workspace_options,
            horizontal=True,
        )

        if mode == "Admin" and not is_admin:
            st.error("Admin access is not authorized for this account.")
            st.stop()

    if page == "🏗 Build Lineups":
        default_main = next(
"""

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(msg):
    print(f"FAIL={msg}")
    sys.exit(1)

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-E")
    print("DATA CENTER ADMIN MODE FIX — FAIL CLOSED")
    print("=" * 76)

    if not APP.is_file():
        fail(f"missing app.py: {APP}")

    pre_sha = sha(APP)
    print(f"PRE_APP_SHA256={pre_sha}")
    if pre_sha != EXPECTED_APP_SHA:
        fail("current app.py hash mismatch")

    original = APP.read_text()
    anchor_count = original.count(OLD)
    print(f"WORKSPACE_BLOCK_ANCHOR_COUNT={anchor_count}")
    if anchor_count != 1:
        fail(f"workspace block anchor count={anchor_count}")

    candidate = original.replace(OLD, NEW, 1)

    checks = {
        "SHARED_PAGE_GATE": 'if page in {"🏗 Build Lineups", "🏈 NFL Data Center"}:' in candidate,
        "FAIL_CLOSED_DEFAULT": 'mode = "Public"' in candidate,
        "ADMIN_GUARD": 'if mode == "Admin" and not is_admin:' in candidate,
        "BUILD_BRANCH_PRESERVED": 'if page == "🏗 Build Lineups":\n        default_main = next(' in candidate,
        "DATA_CENTER_ROUTE_PRESERVED": 'render_data_center(mode)' in candidate,
        "POSTGAME_ADMIN_GATE_PRESERVED": 'if workspace_mode == "Admin" and postgame_tab is not None:' in candidate,
    }
    for name, ok in checks.items():
        print(f"{name}={str(ok).upper()}")
        if not ok:
            fail(f"semantic assertion failed: {name}")

    with tempfile.TemporaryDirectory(prefix="wfs_1ee_") as td:
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
            fail("existing 1E-E backup hash mismatch")
        print("BACKUP_ACTION=EXISTING_VALID")
    else:
        shutil.copy2(APP, BACKUP)
        print("BACKUP_ACTION=CREATED")

    print(f"BACKUP_PATH={BACKUP}")
    print(f"BACKUP_SHA256={sha(BACKUP)}")

    staged = APP.with_name("app.py.NFL-POSTGAME-1E-E.tmp")
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

    print("\nNFL_POSTGAME_1E_E_STATUS=PASS")
    print("DATA_CENTER_WORKSPACE_SELECTOR=TRUE")
    print("AUTHORIZED_ADMIN_CAN_SELECT_ADMIN=TRUE")
    print("NON_ADMIN_PUBLIC_ONLY=TRUE")
    print("DEFAULT_MODE_PUBLIC=TRUE")
    print("DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
