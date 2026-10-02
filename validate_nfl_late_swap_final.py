#!/usr/bin/env python3
from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
APP = ROOT / "app.py"
LATE_SOLVER = ROOT / "fanduel_nfl_late_swap_solver_stage2_v1.py"
DB = ROOT / "data" / "nfl.db"
TABLE = "fanduel_solver_ready_pool"
TARGET_SLUG = "snf_mnf"


def normalize_slate(value) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_name(node, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def validate_app_routing() -> dict:
    text = APP.read_text(encoding="utf-8")
    tree = ast.parse(text)

    helper_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "late_swap_slate_pool"
    ]

    normal_assignments = []
    late_assignments = []
    routed_calls = []
    old_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if is_name(node.targets[0], "selected_pool"):
                normal_assignments.append(node)
            elif is_name(node.targets[0], "late_swap_selected_pool"):
                late_assignments.append(node)

        if (
            isinstance(node, ast.Call)
            and is_name(node.func, "render_late_swap_stage1")
            and len(node.args) >= 2
        ):
            if is_name(node.args[0], "late_swap_selected_pool"):
                routed_calls.append(node)
            if is_name(node.args[0], "selected_pool"):
                old_calls.append(node)

    return {
        "helper_defs": len(helper_defs),
        "normal_selected_pool_assignments": len(normal_assignments),
        "late_swap_selected_pool_assignments": len(late_assignments),
        "late_swap_routed_calls": len(routed_calls),
        "old_selected_pool_late_swap_calls": len(old_calls),
        "pass": (
            len(helper_defs) == 1
            and len(normal_assignments) == 1
            and len(late_assignments) == 1
            and len(routed_calls) == 1
            and len(old_calls) == 0
        ),
    }


def validate_solver_module() -> dict:
    text = LATE_SOLVER.read_text(encoding="utf-8")

    checks = {
        "preserve_only_marker": "_late_swap_preserve_only" in text,
        "candidate_marker": "_late_swap_candidate" in text,
        "current_slot_keys_prepare_arg": (
            "current_slot_keys: Mapping[str, str]" in text
        ),
        "slot_status_prepare_arg": (
            "slot_status: Mapping[str, str]" in text
        ),
        "preserve_projection_display": "_projection_display" in text,
    }

    checks["pass"] = all(checks.values())
    return checks


def load_target_pool() -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query(
            f'SELECT * FROM "{TABLE}"',
            conn,
        )

    if "slate_slug_solver" not in df.columns:
        raise RuntimeError(
            "Missing slate_slug_solver from fanduel_solver_ready_pool."
        )

    mask = df["slate_slug_solver"].map(normalize_slate).eq(TARGET_SLUG)

    if "slate_name_solver" in df.columns:
        mask = mask | df["slate_name_solver"].map(normalize_slate).eq(
            TARGET_SLUG
        )

    return df.loc[mask].copy()


def validate_pool(df: pd.DataFrame) -> dict:
    required = {
        "player_solver",
        "team_solver",
        "solver_position",
        "salary_solver",
        "projection_solver",
        "solver_eligible",
        "game_solver",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Solver-ready pool missing required columns: "
            + ", ".join(missing)
        )

    eligible_mask = (
        pd.to_numeric(df["solver_eligible"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    total = int(len(df))
    eligible = int(eligible_mask.sum())
    blocked = int(total - eligible)

    game_summary = (
        df.assign(_eligible=eligible_mask.astype(int))
        .groupby("game_solver", dropna=False)
        .agg(
            rows=("player_solver", "size"),
            eligible_rows=("_eligible", "sum"),
        )
        .reset_index()
        .sort_values("game_solver", kind="mergesort")
    )

    return {
        "total": total,
        "eligible": eligible,
        "blocked": blocked,
        "games": int(df["game_solver"].nunique(dropna=True)),
        "game_summary": game_summary,
        "pass": (
            total == 65
            and eligible == 27
            and blocked == 38
        ),
    }


def main() -> None:
    print("=" * 92)
    print("NFL LATE SWAP FINAL READ-ONLY VALIDATION")
    print("=" * 92)

    app_result = validate_app_routing()
    solver_result = validate_solver_module()
    pool = load_target_pool()
    pool_result = validate_pool(pool)

    print()
    print("APP ROUTING")
    print("-" * 92)
    for key, value in app_result.items():
        if key != "pass":
            print(f"{key:40s} {value}")
    print(f"{'APP ROUTING STATUS':40s} {'PASS' if app_result['pass'] else 'BLOCKED'}")

    print()
    print("LATE SWAP SOLVER BOUNDARY")
    print("-" * 92)
    for key, value in solver_result.items():
        if key != "pass":
            print(f"{key:40s} {value}")
    print(
        f"{'LATE SWAP SOLVER STATUS':40s} "
        f"{'PASS' if solver_result['pass'] else 'BLOCKED'}"
    )

    print()
    print("SNF-MNF SOLVER-READY POOL")
    print("-" * 92)
    print(f"{'total retained rows':40s} {pool_result['total']}")
    print(f"{'solver eligible rows':40s} {pool_result['eligible']}")
    print(f"{'solver blocked rows':40s} {pool_result['blocked']}")
    print(f"{'distinct games':40s} {pool_result['games']}")
    print()
    print(pool_result["game_summary"].to_string(index=False))
    print()
    print(
        f"{'POOL STATUS':40s} "
        f"{'PASS' if pool_result['pass'] else 'BLOCKED'}"
    )

    overall = (
        app_result["pass"]
        and solver_result["pass"]
        and pool_result["pass"]
    )

    print()
    print("=" * 92)
    print(
        "FINAL LATE SWAP STATIC VALIDATION: "
        + ("PASS" if overall else "BLOCKED")
    )
    print("=" * 92)

    if overall:
        print()
        print("Static pipeline is ready for one real FanDuel late-swap entry test.")
        print("Expected behavior:")
        print("  - DAL/NYG current occupants stay frozen in exact original slots.")
        print("  - DEN/KC solver-eligible players may fill only SWAPPABLE slots.")
        print("  - Locked/blocked DAL/NYG players cannot be newly introduced.")
        print("  - Normal Classic lineup generation remains eligible-only.")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
