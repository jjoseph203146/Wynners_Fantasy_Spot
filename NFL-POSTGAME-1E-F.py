#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1E-F
Make Admin Postgame Feedback unmistakably visible in NFL Data Center.

This removes the mobile-sensitive third-tab presentation and renders the
Admin-only feedback panel directly above the normal Schedule/Box Score tabs.

Fail-closed:
- exact current app.py hash required
- exact source anchors required once
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
BACKUP = ROOT / "backups" / "app.py.NFL-POSTGAME-1E-F.prechange.bak"

EXPECTED_APP_SHA = "af30f4c5928d1280acbfb980f6982a7e3a02265aca3fb29126d9e23c8c138ef6"

OLD_TABS = """    dc_tab_labels = ["📅 Schedule", "📊 Box Scores"]
    if workspace_mode == "Admin":
        dc_tab_labels.append("🧪 Postgame Feedback")

    dc_tabs = st.tabs(dc_tab_labels)
    schedule_tab, box_tab = dc_tabs[0], dc_tabs[1]
    postgame_tab = dc_tabs[2] if workspace_mode == "Admin" else None

    with schedule_tab:
"""

NEW_TABS = """    if workspace_mode == "Admin":
        st.markdown("### 🔒 Admin Postgame Feedback")
        from wfs_postgame_feedback import render_postgame_feedback_admin
        render_postgame_feedback_admin()
        st.divider()

    schedule_tab, box_tab = st.tabs(["📅 Schedule", "📊 Box Scores"])

    with schedule_tab:
"""

OLD_BOTTOM = """    if workspace_mode == "Admin" and postgame_tab is not None:
        with postgame_tab:
            from wfs_postgame_feedback import render_postgame_feedback_admin
            render_postgame_feedback_admin()



SLEEPER_API_BASE = "https://api.sleeper.app/v1"
"""

NEW_BOTTOM = """SLEEPER_API_BASE = "https://api.sleeper.app/v1"
"""

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(msg):
    print(f"FAIL={msg}")
    sys.exit(1)

def main():
    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1E-F")
    print("VISIBLE ADMIN POSTGAME PANEL — FAIL CLOSED")
    print("=" * 76)

    if not APP.is_file():
        fail(f"missing app.py: {APP}")

    pre_sha = sha(APP)
    print(f"PRE_APP_SHA256={pre_sha}")
    if pre_sha != EXPECTED_APP_SHA:
        fail("current app.py hash mismatch")

    original = APP.read_text()

    tabs_count = original.count(OLD_TABS)
    bottom_count = original.count(OLD_BOTTOM)
    print(f"TAB_BLOCK_ANCHOR_COUNT={tabs_count}")
    print(f"OLD_BOTTOM_BLOCK_ANCHOR_COUNT={bottom_count}")

    if tabs_count != 1:
        fail(f"tab block anchor count={tabs_count}")
    if bottom_count != 1:
        fail(f"old bottom block anchor count={bottom_count}")

    candidate = original.replace(OLD_TABS, NEW_TABS, 1)
    candidate = candidate.replace(OLD_BOTTOM, NEW_BOTTOM, 1)

    checks = {
        "ADMIN_GATE_PRESENT": 'if workspace_mode == "Admin":' in candidate,
        "VISIBLE_ADMIN_HEADING": 'st.markdown("### 🔒 Admin Postgame Feedback")' in candidate,
        "POSTGAME_RENDER_PRESENT": 'render_postgame_feedback_admin()' in candidate,
        "PUBLIC_TABS_EXACT": 'schedule_tab, box_tab = st.tabs(["📅 Schedule", "📊 Box Scores"])' in candidate,
        "OLD_POSTGAME_TAB_REMOVED": 'postgame_tab' not in candidate,
        "DATA_CENTER_ROUTE_PRESERVED": 'render_data_center(mode)' in candidate,
        "WORKSPACE_SELECTOR_PRESERVED": 'if page in {"🏗 Build Lineups", "🏈 NFL Data Center"}:' in candidate,
    }

    for name, ok in checks.items():
        print(f"{name}={str(ok).upper()}")
        if not ok:
            fail(f"semantic assertion failed: {name}")

    with tempfile.TemporaryDirectory(prefix="wfs_1ef_") as td:
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
            fail("existing 1E-F backup hash mismatch")
        print("BACKUP_ACTION=EXISTING_VALID")
    else:
        shutil.copy2(APP, BACKUP)
        print("BACKUP_ACTION=CREATED")

    print(f"BACKUP_PATH={BACKUP}")
    print(f"BACKUP_SHA256={sha(BACKUP)}")

    staged = APP.with_name("app.py.NFL-POSTGAME-1E-F.tmp")
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

    print("\nNFL_POSTGAME_1E_F_STATUS=PASS")
    print("ADMIN_FEEDBACK_VISIBLE_ABOVE_TABS=TRUE")
    print("PUBLIC_DATA_CENTER_UNCHANGED=TRUE")
    print("MOBILE_THIRD_TAB_DEPENDENCY_REMOVED=TRUE")
    print("DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
