#!/usr/bin/env python3
from __future__ import annotations

import ast
import py_compile
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "fanduel_nfl_late_swap_solver_stage2_v1.py"
BACKUP_ROOT = ROOT / "backups"

OLD_BLOCK = '''    if not out["solver_position"].astype(str).isin(prod.VALID_POSITIONS).all():
        bad = out.loc[
            ~out["solver_position"].astype(str).isin(prod.VALID_POSITIONS),
            ["player_solver", "team_solver", "solver_position"],
        ]
        raise ValueError(
            "Late-swap source pool contains invalid position:\\n"
            + bad.to_string(index=False)
        )
'''

FILTER_BLOCK = '''    out = out.loc[
        out["_late_swap_candidate"]
        | out["_late_swap_preserve_only"]
    ].copy()
'''

NEW_BLOCK = '''    # Validate positions only after narrowing to rows that can actually
    # participate in Late Swap:
    #   1) normal solver-eligible candidates, or
    #   2) exact LOCKED/UNKNOWN current occupants retained preserve-only.
    #
    # Unresolved-position rows that are neither candidates nor current locked
    # occupants remain quarantined upstream and must not block the whole slate.
    bad_working_position = ~out["solver_position"].astype(str).isin(
        prod.VALID_POSITIONS
    )
    if bad_working_position.any():
        bad = out.loc[
            bad_working_position,
            [
                "player_solver",
                "team_solver",
                "solver_position",
                "_late_swap_candidate",
                "_late_swap_preserve_only",
            ],
        ]
        raise ValueError(
            "Late-swap working pool contains invalid position:\\n"
            + bad.to_string(index=False)
        )
'''


def validate_source(text: str) -> None:
    tree = ast.parse(text)

    funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prepare_pool"
    ]
    if len(funcs) != 1:
        raise RuntimeError(
            f"Expected exactly one _prepare_pool(); found {len(funcs)}. "
            "No changes were written."
        )

    if text.count(OLD_BLOCK) != 1:
        raise RuntimeError(
            "Expected exactly one early source-position validation block. "
            f"Found {text.count(OLD_BLOCK)}. No changes were written."
        )

    if text.count(FILTER_BLOCK) != 1:
        raise RuntimeError(
            "Expected exactly one candidate/preserve-only filter block. "
            f"Found {text.count(FILTER_BLOCK)}. No changes were written."
        )

    required_markers = [
        "_late_swap_candidate",
        "_late_swap_preserve_only",
        "_projection_display",
        "current_locked_keys",
    ]
    missing = [m for m in required_markers if m not in text]
    if missing:
        raise RuntimeError(
            "Installed solver does not match expected late-swap v2 structure. "
            "Missing marker(s): " + ", ".join(missing)
        )


def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")

    if "Late-swap working pool contains invalid position:" in original:
        raise RuntimeError(
            "The working-pool position fix already appears to be installed. "
            "No changes were written."
        )

    validate_source(original)

    patched = original.replace(OLD_BLOCK, "", 1)

    patched = patched.replace(
        FILTER_BLOCK,
        FILTER_BLOCK + NEW_BLOCK,
        1,
    )

    if "Late-swap source pool contains invalid position:" in patched:
        raise RuntimeError(
            "Old source-wide position validation still exists after patch."
        )

    if patched.count(
        "Late-swap working pool contains invalid position:"
    ) != 1:
        raise RuntimeError(
            "Expected exactly one working-pool position validation after patch."
        )

    ast.parse(patched)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"late_swap_position_scope_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    TARGET.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        shutil.copy2(backup, TARGET)
        raise

    print("=" * 88)
    print("LATE SWAP POSITION-SCOPE REPAIR: PASS")
    print("=" * 88)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print()
    print("Changed:")
    print("  raw 65-row source position validation : REMOVED")
    print("  candidate/preserve-only filtering     : UNCHANGED")
    print("  working-pool position validation      : ADDED")
    print()
    print("Unchanged:")
    print("  solver_eligible candidate rules")
    print("  preserve-only locked-player rules")
    print("  salary validation")
    print("  projection validation")
    print("  FanDuel salary/team/position legality")
    print("  QB stack / bring-back rules")
    print("  normal production solver")
    print()
    print("Compile: PASS")
    print()
    print("Expected effect:")
    print("  Patrick Ricard / Hunter Luepke / Adam Prentice remain quarantined")
    print("  but no longer block the entire SNF-MNF Late Swap solve.")


if __name__ == "__main__":
    main()
