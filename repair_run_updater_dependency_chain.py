#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "run_updater.sh"
BACKUP_ROOT = ROOT / "backups"

REQUIRED_FILES = [
    ROOT / "fanduel_slate_ingest_v2.py",
    ROOT / "current_slate_features.py",
    ROOT / "production_projection.py",
    ROOT / "fanduel_player_pool.py",
    ROOT / "fanduel_slate_projection_attach_v5.py",
    ROOT / "fanduel_solver_ready_pool.py",
]

OLD_VARS = '''INJURY_CONSENSUS="$PROJECT_DIR/injury_consensus.py"
SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
'''

NEW_VARS = '''INJURY_CONSENSUS="$PROJECT_DIR/injury_consensus.py"
FANDUEL_SLATE_INGEST="$PROJECT_DIR/fanduel_slate_ingest_v2.py"
CURRENT_SLATE_FEATURES="$PROJECT_DIR/current_slate_features.py"
PRODUCTION_PROJECTION="$PROJECT_DIR/production_projection.py"
FANDUEL_PLAYER_POOL="$PROJECT_DIR/fanduel_player_pool.py"
FANDUEL_PROJECTION_ATTACH="$PROJECT_DIR/fanduel_slate_projection_attach_v5.py"
SOLVER_READY="$PROJECT_DIR/fanduel_solver_ready_pool.py"
FEEDBACK_REFRESH="$PROJECT_DIR/nfl_2026_feedback_refresh.py"
'''

OLD_PIPE = '''    if ! wait_for_live_quiet "solver-ready injury rebuild"; then
        return 33
    fi
    run_stage "Solver-ready injury rebuild" "$SOLVER_READY" 23
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "2026 feedback refresh"; then
'''

NEW_PIPE = '''    if ! wait_for_live_quiet "FanDuel schedule-aware slate ingest"; then
        return 37
    fi
    run_stage "FanDuel schedule-aware slate ingest" "$FANDUEL_SLATE_INGEST" 30
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "current slate feature rebuild"; then
        return 38
    fi
    run_stage "Current slate feature rebuild" "$CURRENT_SLATE_FEATURES" 31
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "production projection rebuild"; then
        return 39
    fi
    run_stage "Production projection rebuild" "$PRODUCTION_PROJECTION" 32
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "FanDuel player pool rebuild"; then
        return 40
    fi
    run_stage "FanDuel player pool rebuild" "$FANDUEL_PLAYER_POOL" 33
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "FanDuel slate projection attach"; then
        return 41
    fi
    run_stage "FanDuel slate projection attach" "$FANDUEL_PROJECTION_ATTACH" 34
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "solver-ready rebuild"; then
        return 42
    fi
    run_stage "Solver-ready rebuild" "$SOLVER_READY" 35
    rc=$?
    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if ! wait_for_live_quiet "2026 feedback refresh"; then
'''

def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise RuntimeError("Required pipeline file(s) missing:\n" + "\n".join(missing))

    text = TARGET.read_text(encoding="utf-8")

    if NEW_VARS in text or "FanDuel schedule-aware slate ingest" in text:
        raise RuntimeError("Updater dependency repair appears to be already installed.")

    if text.count(OLD_VARS) != 1:
        raise RuntimeError(
            f"Expected exactly one variable anchor; found {text.count(OLD_VARS)}. "
            "No changes were written."
        )

    if text.count(OLD_PIPE) != 1:
        raise RuntimeError(
            f"Expected exactly one pipeline anchor; found {text.count(OLD_PIPE)}. "
            "No changes were written."
        )

    patched = text.replace(OLD_VARS, NEW_VARS, 1)
    patched = patched.replace(OLD_PIPE, NEW_PIPE, 1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"run_updater_dependency_chain_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    TARGET.write_text(patched, encoding="utf-8")

    syntax = subprocess.run(
        ["bash", "-n", str(TARGET)],
        text=True,
        capture_output=True,
    )
    if syntax.returncode != 0:
        shutil.copy2(backup, TARGET)
        raise RuntimeError(
            "bash -n failed; original restored.\n"
            + syntax.stdout
            + syntax.stderr
        )

    print("=" * 96)
    print("NFL HOURLY UPDATER DEPENDENCY REPAIR: PASS")
    print("=" * 96)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print()
    print("Inserted dependency chain:")
    print("  injury consensus")
    print("    -> fanduel_slate_ingest_v2.py")
    print("    -> current_slate_features.py")
    print("    -> production_projection.py")
    print("    -> fanduel_player_pool.py")
    print("    -> fanduel_slate_projection_attach_v5.py")
    print("    -> fanduel_solver_ready_pool.py")
    print("    -> 2026 feedback refresh")
    print()
    print("Unchanged:")
    print("  core updater")
    print("  player identity")
    print("  injury ingest / retry logic")
    print("  injury consensus")
    print("  forecast stages")
    print("  lock / quiet-window behavior")
    print("  cron schedule")
    print()
    print("Validation: bash -n PASS")
    print()
    print("NOTE:")
    print("  nfl_2026_feedback_refresh.py is still expected to fail closed until its")
    print("  frozen production_projection.py baseline/hash is intentionally reconciled.")
    print("  This repair does not weaken or bypass that safety check.")

if __name__ == "__main__":
    main()
