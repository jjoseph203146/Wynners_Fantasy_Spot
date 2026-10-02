#!/usr/bin/env python3
from __future__ import annotations

import ast
import py_compile
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "app.py"
BACKUP_ROOT = ROOT / "backups"

OLD = """    settings = LateSwapSettings(
        qb_stack=2,
        bring_back=1,
        projection_loss_pct=0.01,
        salary_floor=0,
        max_team=0,
    )
"""

NEW = """    settings = LateSwapSettings(
        qb_stack=1,
        bring_back=0,
        projection_loss_pct=0.015,
        salary_floor=0,
        max_team=0,
    )
"""

def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    ast.parse(text)

    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one LateSwapSettings app block; found {count}. "
            "No changes were written."
        )

    if NEW in text:
        raise RuntimeError("Validated Late Swap settings already appear installed.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"late_swap_app_settings_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    patched = text.replace(OLD, NEW, 1)
    TARGET.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        shutil.copy2(backup, TARGET)
        raise

    print("=" * 88)
    print("LATE SWAP APP SETTINGS REPAIR: PASS")
    print("=" * 88)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print("Changed:")
    print("  qb_stack            : 2 -> 1")
    print("  bring_back          : 1 -> 0")
    print("  projection_loss_pct : 0.01 -> 0.015")
    print("Unchanged:")
    print("  salary_floor        : 0")
    print("  max_team            : 0")
    print("  Stage 2 solver code")
    print("  normal production solver")
    print("Compile: PASS")
    print()
    print("Reason:")
    print("  Aligns Streamlit Late Swap with the exact settings that just passed")
    print("  diagnose_exact_late_swap_runtime.py for this 4-locked / 5-swappable lineup.")

if __name__ == "__main__":
    main()
