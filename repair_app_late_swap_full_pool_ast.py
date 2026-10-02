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

HELPER = r'''

def late_swap_slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    # Full selected-slate view for Late Swap only.
    # Retains solver-blocked rows so already-started roster occupants can be preserved.
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

    out = out.dropna(subset=["salary_solver"]).copy()
    out = out.loc[out["salary_solver"].gt(0)].copy()
    out["salary_solver"] = out["salary_solver"].astype(int)

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

        position_value = row.get("solver_position")
        position = (
            str(position_value).strip()
            if pd.notna(position_value)
            else ""
        ) or "—"

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
'''


def is_name(node, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def find_slate_pool_function(tree: ast.AST):
    matches = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "slate_pool"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one slate_pool() function; found {len(matches)}. "
            "No changes were written."
        )
    return matches[0]


def find_selected_pool_assignment(tree: ast.AST):
    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1 or not is_name(node.targets[0], "selected_pool"):
            continue

        value = node.value
        if (
            isinstance(value, ast.Call)
            and is_name(value.func, "slate_pool")
            and len(value.args) >= 2
            and is_name(value.args[0], "pool")
            and is_name(value.args[1], "selected_slate")
        ):
            matches.append(node)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one selected_pool = slate_pool(pool, selected_slate) "
            f"assignment; found {len(matches)}. No changes were written."
        )

    return matches[0]


def find_late_swap_call(tree: ast.AST):
    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not is_name(node.func, "render_late_swap_stage1"):
            continue

        if (
            len(node.args) >= 2
            and is_name(node.args[0], "selected_pool")
            and is_name(node.args[1], "selected_slate")
        ):
            matches.append(node)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one render_late_swap_stage1(selected_pool, selected_slate) "
            f"call; found {len(matches)}. No changes were written."
        )

    return matches[0]


def insert_after_line(lines: list[str], line_no: int, text: str) -> list[str]:
    idx = int(line_no)
    return lines[:idx] + text.splitlines(keepends=True) + lines[idx:]


def replace_line_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    replacement: str,
) -> list[str]:
    start_idx = int(start_line) - 1
    end_idx = int(end_line)
    return (
        lines[:start_idx]
        + replacement.splitlines(keepends=True)
        + lines[end_idx:]
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

    tree = ast.parse(original)

    slate_pool_fn = find_slate_pool_function(tree)
    selected_assign = find_selected_pool_assignment(tree)
    late_swap_call = find_late_swap_call(tree)

    lines = original.splitlines(keepends=True)

    call_start = late_swap_call.lineno
    call_end = late_swap_call.end_lineno
    call_line = lines[call_start - 1]
    call_indent = call_line[: len(call_line) - len(call_line.lstrip())]

    replacement_call = (
        f"{call_indent}render_late_swap_stage1(\\n"
        f"{call_indent}    late_swap_selected_pool,\\n"
        f"{call_indent}    selected_slate,\\n"
        f"{call_indent})\\n"
    )

    lines = replace_line_span(
        lines,
        call_start,
        call_end,
        replacement_call,
    )

    interim = "".join(lines)
    tree2 = ast.parse(interim)
    selected_assign2 = find_selected_pool_assignment(tree2)

    assign_line = lines[selected_assign2.lineno - 1]
    assignment_indent = assign_line[: len(assign_line) - len(assign_line.lstrip())]

    late_pool_assignment = (
        f"{assignment_indent}late_swap_selected_pool = late_swap_slate_pool(\\n"
        f"{assignment_indent}    pool,\\n"
        f"{assignment_indent}    selected_slate,\\n"
        f"{assignment_indent})\\n"
    )

    lines = insert_after_line(
        lines,
        selected_assign2.end_lineno,
        late_pool_assignment,
    )

    interim2 = "".join(lines)
    tree3 = ast.parse(interim2)
    slate_pool_fn3 = find_slate_pool_function(tree3)

    lines = insert_after_line(
        lines,
        slate_pool_fn3.end_lineno,
        HELPER,
    )

    patched = "".join(lines)
    final_tree = ast.parse(patched)

    helper_defs = [
        n for n in ast.walk(final_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "late_swap_slate_pool"
    ]
    if len(helper_defs) != 1:
        raise RuntimeError(
            f"Patched source validation failed: late_swap_slate_pool count={len(helper_defs)}."
        )

    late_pool_assignments = [
        n for n in ast.walk(final_tree)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and is_name(n.targets[0], "late_swap_selected_pool")
    ]
    if len(late_pool_assignments) != 1:
        raise RuntimeError(
            "Patched source validation failed: expected one "
            "late_swap_selected_pool assignment."
        )

    routed_calls = []
    old_calls = []
    for node in ast.walk(final_tree):
        if not isinstance(node, ast.Call):
            continue
        if not is_name(node.func, "render_late_swap_stage1"):
            continue

        if len(node.args) >= 2:
            if is_name(node.args[0], "late_swap_selected_pool"):
                routed_calls.append(node)
            if is_name(node.args[0], "selected_pool"):
                old_calls.append(node)

    if len(routed_calls) != 1 or old_calls:
        raise RuntimeError(
            "Patched source validation failed: Late Swap routing is not exact."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"late_swap_full_pool_app_ast_{stamp}"
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
    print("Late Swap routing:             late_swap_selected_pool")
    print("Compile:                       PASS")
    print()
    print("Expected SNF-MNF behavior:")
    print("  normal lineup pool : 27 solver-eligible rows")
    print("  Late Swap pool     : 65 retained slate rows")
    print("  DAL/NYG            : preserve-only when locked in current lineup")
    print("  DEN/KC             : solver-eligible SWAPPABLE candidates only")


if __name__ == "__main__":
    main()
