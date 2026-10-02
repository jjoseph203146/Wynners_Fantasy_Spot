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

OLD_BLOCK = '''    out = prod.attach_validated_gpp_fields(out)
    out = prod.add_opponents(out)
    out = prod.build_gpp_objective_fields(out)

    # Preserve-only locked rows are not part of candidate ranking.
    # Their GPP contribution is also a fixed constant, so neutralize it.
    if "gpp_objective_rank_points" in out.columns:
        out.loc[
            out["_late_swap_preserve_only"],
            "gpp_objective_rank_points",
        ] = 0
'''

NEW_BLOCK = '''    # Build validated GPP objective fields only for rows that can actually
    # compete for SWAPPABLE slots. Preserve-only locked rows are fixed constants
    # and must not be required to have current GPP-score coverage.
    candidate_rows = out.loc[
        out["_late_swap_candidate"]
    ].copy()
    preserve_rows = out.loc[
        out["_late_swap_preserve_only"]
    ].copy()

    if not candidate_rows.empty:
        candidate_rows = prod.attach_validated_gpp_fields(candidate_rows)
        candidate_rows = prod.add_opponents(candidate_rows)
        candidate_rows = prod.build_gpp_objective_fields(candidate_rows)

    if not preserve_rows.empty:
        preserve_rows = prod.add_opponents(preserve_rows)
        preserve_rows["gpp_objective_signal"] = 0.0
        preserve_rows["gpp_objective_method"] = "LOCKED_PRESERVE_ONLY"
        preserve_rows["gpp_objective_rank_points"] = 0

    out = pd.concat(
        [candidate_rows, preserve_rows],
        ignore_index=True,
        sort=False,
    )

    if out.empty:
        raise ValueError("Late-swap working pool is empty after preparation.")
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
            "Expected exactly one current full-pool GPP objective block; "
            f"found {text.count(OLD_BLOCK)}. No changes were written."
        )

    required = [
        "_late_swap_candidate",
        "_late_swap_preserve_only",
        "_projection_display",
        "Late-swap working pool contains invalid position:",
    ]
    missing = [m for m in required if m not in text]
    if missing:
        raise RuntimeError(
            "Installed Late Swap solver does not match expected repaired structure. "
            "Missing marker(s): " + ", ".join(missing)
        )


def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")

    if 'LOCKED_PRESERVE_ONLY' in original:
        raise RuntimeError(
            "Candidate-only GPP repair already appears installed. "
            "No changes were written."
        )

    validate_source(original)

    patched = original.replace(
        OLD_BLOCK,
        NEW_BLOCK,
        1,
    )

    if patched.count("LOCKED_PRESERVE_ONLY") != 1:
        raise RuntimeError(
            "Expected exactly one preserve-only GPP marker after patch."
        )

    if "out = prod.build_gpp_objective_fields(out)" in patched:
        raise RuntimeError(
            "Full working-pool GPP objective call still exists after patch."
        )

    ast.parse(patched)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"late_swap_candidate_gpp_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    TARGET.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        shutil.copy2(backup, TARGET)
        raise

    print("=" * 92)
    print("LATE SWAP CANDIDATE-ONLY GPP REPAIR: PASS")
    print("=" * 92)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print()
    print("Changed:")
    print("  candidate GPP attachment/ranking : candidate rows only")
    print("  preserve-only GPP contribution   : fixed zero constant")
    print("  preserve-only model presence     : retained")
    print()
    print("Unchanged:")
    print("  exact locked-slot preservation")
    print("  candidate solver_eligible rules")
    print("  candidate current-projection requirement")
    print("  FanDuel salary cap")
    print("  FanDuel max-4/team")
    print("  positional legality")
    print("  QB stack / bring-back rules")
    print("  normal production solver")
    print()
    print("Compile: PASS")
    print()
    print("Expected effect:")
    print("  locked DAL/NYG RB/WR/TE rows no longer require GPP score coverage")
    print("  while all DEN/KC candidates keep the validated production GPP objective.")


if __name__ == "__main__":
    main()
