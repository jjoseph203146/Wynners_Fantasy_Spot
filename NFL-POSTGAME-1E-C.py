#!/usr/bin/env python3
# WFS NFL — NFL-POSTGAME-1E-C
# Fail-closed Admin-only Data Center postgame integration.

from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import tempfile
import difflib

ROOT = Path("/home/mwynn/nfl_data_engine")
APP = ROOT / "app.py"
MODULE = ROOT / "wfs_postgame_feedback.py"
BACKUP = ROOT / "backups" / "app.py.NFL-POSTGAME-1E-C.prechange.bak"

EXPECTED_APP_SHA = "07cde83abf3904ad30b14d0917eeaaba312528ef546fbb1a0dc50064bacf3a7c"
EXPECTED_MODULE_SHA = "701707c22178521519216dbb470d748c6542da5a293b2c6ddf684d859e2567c5"

OLD_TABS = '    schedule_tab, box_tab = st.tabs(["📅 Schedule", "📊 Box Scores"])\n'

NEW_TABS = '''    dc_tab_labels = ["📅 Schedule", "📊 Box Scores"]
    if workspace_mode == "Admin":
        dc_tab_labels.append("🧪 Postgame Feedback")

    dc_tabs = st.tabs(dc_tab_labels)
    schedule_tab, box_tab = dc_tabs[0], dc_tabs[1]
    postgame_tab = dc_tabs[2] if workspace_mode == "Admin" else None
'''

ANCHOR_AFTER_DC = '''                else:
                    st.info("No games match the selected filters.")



SLEEPER_API_BASE = "https://api.sleeper.app/v1"
'''

REPLACEMENT_AFTER_DC = '''                else:
                    st.info("No games match the selected filters.")

    if workspace_mode == "Admin" and postgame_tab is not None:
        with postgame_tab:
            from wfs_postgame_feedback import render_postgame_feedback_admin
            render_postgame_feedback_admin()



SLEEPER_API_BASE = "https://api.sleeper.app/v1"
'''

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(msg: str) -> None:
    print(f"FAIL={msg}")
    sys.exit(1)

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-C")
    print("ADMIN-ONLY DATA CENTER POSTGAME INTEGRATION — FAIL CLOSED")
    print("=" * 76)

    if not APP.is_file():
        fail(f"missing app.py: {APP}")
    if not MODULE.is_file():
        fail(f"missing module: {MODULE}")

    app_sha = sha(APP)
    mod_sha = sha(MODULE)
    print(f"PRE_APP_SHA256={app_sha}")
    print(f"MODULE_SHA256={mod_sha}")

    if app_sha != EXPECTED_APP_SHA:
        fail("frozen app.py hash mismatch")
    if mod_sha != EXPECTED_MODULE_SHA:
        fail("postgame module hash mismatch")

    original = APP.read_text()
    tabs_count = original.count(OLD_TABS)
    anchor_count = original.count(ANCHOR_AFTER_DC)
    print(f"TABS_ANCHOR_COUNT={tabs_count}")
    print(f"INSERTION_ANCHOR_COUNT={anchor_count}")

    if tabs_count != 1:
        fail(f"tabs anchor count={tabs_count}")
    if anchor_count != 1:
        fail(f"postgame insertion anchor count={anchor_count}")

    candidate = original.replace(OLD_TABS, NEW_TABS, 1)
    candidate = candidate.replace(ANCHOR_AFTER_DC, REPLACEMENT_AFTER_DC, 1)

    with tempfile.TemporaryDirectory(prefix="wfs_1ec_") as td:
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
        existing_backup_sha = sha(BACKUP)
        if existing_backup_sha != EXPECTED_APP_SHA:
            fail("existing backup hash mismatch")
        print("BACKUP_ACTION=EXISTING_VALID")
    else:
        shutil.copy2(APP, BACKUP)
        print("BACKUP_ACTION=CREATED")

    print(f"BACKUP_PATH={BACKUP}")
    print(f"BACKUP_SHA256={sha(BACKUP)}")

    staged = APP.with_name("app.py.NFL-POSTGAME-1E-C.tmp")
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

    new_sha = sha(APP)
    print(f"POST_APP_SHA256={new_sha}")

    print("\n=== UNIFIED DIFF ===")
    diff = difflib.unified_diff(
        original.splitlines(),
        candidate.splitlines(),
        fromfile="app.py.before",
        tofile="app.py.after",
        lineterm="",
        n=4,
    )
    for line in diff:
        print(line)

    print("\nNFL_POSTGAME_1E_C_STATUS=PASS")
    print("ADMIN_ONLY_POSTGAME_TAB=TRUE")
    print("PUBLIC_DATA_CENTER_TABS_UNCHANGED=TRUE")
    print("DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
