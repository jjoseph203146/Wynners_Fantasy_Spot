#!/usr/bin/env python3
from __future__ import annotations

import py_compile
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
TARGET = ROOT / "app.py"
BACKUP_ROOT = ROOT / "backups"

SLATE_POOL_BLOCK = r"""def slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    target = normalize_slate(slate_name)
    mask = (
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    )
    out = df.loc[mask].copy()
    eligible = (
        pd.to_numeric(out["solver_eligible"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    out = out.loc[eligible].copy()
    out["salary_solver"] = pd.to_numeric(out["salary_solver"], errors="coerce")
    out["projection_solver"] = pd.to_numeric(
        out["projection_solver"], errors="coerce"
    )
    out = out.dropna(subset=["salary_solver", "projection_solver"])
    out["salary_solver"] = out["salary_solver"].astype(int)
    out["_ui_key"] = out.apply(ui_player_key, axis=1)
    out["_label"] = out.apply(
        lambda r: (
            f'{r["player_solver"]} — {r["team_solver"]} '
            f'{r["solver_position"]} — ${int(r["salary_solver"]):,} '
            f'— {float(r["projection_solver"]):.2f}'
        ),
        axis=1,
    )
    return out.sort_values(
        ["solver_position", "projection_solver", "salary_solver", "player_solver"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
"""

LATE_SWAP_HELPER = r"""

def late_swap_slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    # Full selected-slate view for Late Swap only.
    # Retains blocked rows so already-started roster occupants can be preserved.
    target = normalize_slate(slate_name)
    mask = (
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    )

    out = df.loc[mask].copy()

    out["salary_solver"] = pd.to_numeric(
        out["salary_solver"],
        errors="coerce",
    )
    out["projection_solver"] = pd.to_numeric(
        out["projection_solver"],
        errors="coerce",
    )

    out = out.dropna(
        subset=["salary_solver"]
    ).copy()

    out = out.loc[
        out["salary_solver"].gt(0)
    ].copy()

    out["salary_solver"] = (
        out["salary_solver"].astype(int)
    )

    out["_ui_key"] = out.apply(
        ui_player_key,
        axis=1,
    )

    def _late_swap_label(row):
        projection = row.get("projection_solver")
        projection_text = (
            f"{float(projection):.2f}"
            if pd.notna(projection)
            else "locked/no current projection"
        )

        position = str(
            row.get("solver_position") or ""
        ).strip() or "—"

        return (
            f'{row["player_solver"]} — {row["team_solver"]} '
            f'{position} — ${int(row["salary_solver"]):,} '
            f'— {projection_text}'
        )

    out["_label"] = out.apply(
        _late_swap_label,
        axis=1,
    )

    return out.sort_values(
        [
            "solver_position",
            "projection_solver",
            "salary_solver",
            "player_solver",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
        kind="mergesort",
        na_position="last",
    )
"""

SELECTED_POOL_ANCHOR = "selected_pool = slate_pool(pool, selected_slate)\n"
SELECTED_POOL_REPLACEMENT = """selected_pool = slate_pool(pool, selected_slate)
late_swap_selected_pool = late_swap_slate_pool(
    pool,
    selected_slate,
)
"""

LATE_SWAP_CALL_OLD = "render_late_swap_stage1(selected_pool, selected_slate)"
LATE_SWAP_CALL_NEW = """render_late_swap_stage1(
            late_swap_selected_pool,
            selected_slate,
        )"""


def require_exactly_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 source anchor, found {count}. "
            "No changes were written."
        )


def main() -> None:
    if not TARGET.exists():
        raise RuntimeError(f"Missing target: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")

    if "def late_swap_slate_pool(" in original:
        raise RuntimeError(
            "late_swap_slate_pool() already exists in app.py. "
            "No changes were written."
        )

    require_exactly_once(original, SLATE_POOL_BLOCK, "slate_pool function")
    require_exactly_once(original, SELECTED_POOL_ANCHOR, "selected_pool assignment")
    require_exactly_once(original, LATE_SWAP_CALL_OLD, "Late Swap render call")

    patched = original.replace(
        SLATE_POOL_BLOCK,
        SLATE_POOL_BLOCK + LATE_SWAP_HELPER,
        1,
    )
    patched = patched.replace(
        SELECTED_POOL_ANCHOR,
        SELECTED_POOL_REPLACEMENT,
        1,
    )
    patched = patched.replace(
        LATE_SWAP_CALL_OLD,
        LATE_SWAP_CALL_NEW,
        1,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"late_swap_full_pool_app_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    backup = backup_dir / "app.py"
    shutil.copy2(TARGET, backup)

    TARGET.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        shutil.copy2(backup, TARGET)
        raise

    print("=" * 78)
    print("LATE SWAP FULL-POOL APP REPAIR: PASS")
    print("=" * 78)
    print(f"Target: {TARGET}")
    print(f"Backup: {backup}")
    print()
    print("Normal selected_pool behavior: UNCHANGED")
    print("Late Swap pool behavior:       FULL selected slate, including blocked rows")
    print("Compile:                       PASS")
    print()
    print("Expected SNF-MNF behavior:")
    print("  normal lineup pool : 27 solver-eligible rows")
    print("  Late Swap pool     : 65 retained slate rows")
    print("  DAL/NYG            : preserve-only when already locked in current lineup")
    print("  DEN/KC             : only solver-eligible SWAPPABLE rows may be introduced")


if __name__ == "__main__":
    main()
