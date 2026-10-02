#!/usr/bin/env python3
"""
fanduel_nfl_lineup_solver.py

Deterministic baseline FanDuel NFL Classic lineup solver.

Consumes ONLY:
    SQLite table: fanduel_solver_ready_pool

FanDuel Classic roster:
    1 QB
    2 RB
    3 WR
    1 TE
    1 FLEX (RB/WR/TE)
    1 DST
    9 total players
    $60,000 salary cap

Design goals
------------
- Deterministic CP-SAT optimization.
- Exact slot assignment; displayed slot is the actual assigned FanDuel slot.
- No fuzzy matching.
- No salary inference.
- No unprojected players.
- No mutation of upstream tables/files.
- Generates top-N distinct lineups by projection.
- Optional salary floor, max lineup overlap, and max players per team.
- Exports lineups and long-form lineup players for downstream GPP logic.

This is intentionally a BASELINE solver. Correlation/stacking/GPP leverage rules
should be layered on only after this legal-lineup engine passes structural audit.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from config import DATABASE_PATH, CSV_DIR


SOURCE_TABLE = "fanduel_solver_ready_pool"

SALARY_CAP = 60000
ROSTER_SLOTS = [
    "QB",
    "RB1",
    "RB2",
    "WR1",
    "WR2",
    "WR3",
    "TE",
    "FLEX",
    "DST",
]

SLOT_ELIGIBILITY = {
    "QB": {"QB"},
    "RB1": {"RB"},
    "RB2": {"RB"},
    "WR1": {"WR"},
    "WR2": {"WR"},
    "WR3": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "DST": {"DST"},
}


def section(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic FanDuel NFL Classic lineup solver."
    )
    parser.add_argument(
        "--slate",
        required=False,
        help=(
            "Slate name or slug, e.g. 'Main', 'main', '1pm only', "
            "'sun-mon'. If omitted, available slates are printed."
        ),
    )
    parser.add_argument(
        "--lineups",
        type=int,
        default=20,
        help="Number of distinct lineups to generate. Default: 20.",
    )
    parser.add_argument(
        "--salary-floor",
        type=int,
        default=0,
        help="Optional minimum lineup salary. Default: 0.",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=8,
        help=(
            "Maximum players shared with each previously generated lineup. "
            "Default: 8 (only exact duplicates banned). Use 7 or lower for "
            "stronger diversification."
        ),
    )
    parser.add_argument(
        "--max-team",
        type=int,
        default=0,
        help=(
            "Optional maximum players from one NFL team. 0 disables this "
            "extra strategy constraint. Default: 0."
        ),
    )
    return parser.parse_args()


def normalize_slate(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_pool():
    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, SOURCE_TABLE):
            raise RuntimeError(
                f"Required table not found: {SOURCE_TABLE}. "
                "Run fanduel_solver_ready_pool.py first."
            )
        df = pd.read_sql_query(f'SELECT * FROM "{SOURCE_TABLE}"', conn)

    required = [
        "slate_name_solver",
        "slate_slug_solver",
        "player_solver",
        "team_solver",
        "salary_solver",
        "game_solver",
        "projection_solver",
        "solver_position",
        "solver_eligible",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{SOURCE_TABLE} missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["salary_solver"] = pd.to_numeric(df["salary_solver"], errors="coerce")
    df["projection_solver"] = pd.to_numeric(
        df["projection_solver"], errors="coerce"
    )
    df["solver_eligible"] = pd.to_numeric(
        df["solver_eligible"], errors="coerce"
    ).fillna(0).astype(int)

    return df


def list_slates(df: pd.DataFrame):
    slates = (
        df[["slate_name_solver", "slate_slug_solver"]]
        .drop_duplicates()
        .sort_values("slate_name_solver")
    )
    section("AVAILABLE SOLVER-READY SLATES")
    print(slates.to_string(index=False))


def select_slate(df: pd.DataFrame, requested: str):
    target = normalize_slate(requested)

    matches = df[
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    ].copy()

    if matches.empty:
        available = (
            df[["slate_name_solver", "slate_slug_solver"]]
            .drop_duplicates()
            .sort_values("slate_name_solver")
        )
        raise RuntimeError(
            f"Slate not found: {requested}\n\nAvailable slates:\n"
            + available.to_string(index=False)
        )

    unique_slates = matches["slate_slug_solver"].dropna().unique()
    if len(unique_slates) != 1:
        raise RuntimeError(
            f"Slate request '{requested}' matched multiple slates: "
            f"{list(unique_slates)}"
        )

    eligible = matches[matches["solver_eligible"].eq(1)].copy()

    eligible = eligible[
        eligible["salary_solver"].notna()
        & eligible["salary_solver"].gt(0)
        & eligible["projection_solver"].notna()
        & np.isfinite(eligible["projection_solver"])
        & eligible["solver_position"].isin({"QB", "RB", "WR", "TE", "DST"})
    ].copy()

    eligible = eligible.reset_index(drop=True)
    eligible["player_index"] = eligible.index.astype(int)

    return matches, eligible


def validate_pool(df: pd.DataFrame):
    if df.empty:
        raise RuntimeError("No solver-eligible rows remain after validation.")

    dup = df.duplicated(
        ["player_solver", "team_solver", "salary_solver"], keep=False
    )
    if dup.any():
        bad = df.loc[
            dup,
            [
                "player_solver",
                "team_solver",
                "salary_solver",
                "solver_position",
            ],
        ]
        raise RuntimeError(
            "Duplicate solver players detected within selected slate:\n"
            + bad.to_string(index=False)
        )

    counts = df["solver_position"].value_counts().to_dict()
    minimums = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    missing = {
        pos: need - counts.get(pos, 0)
        for pos, need in minimums.items()
        if counts.get(pos, 0) < need
    }
    if missing:
        raise RuntimeError(
            f"Selected slate cannot fill required FanDuel slots: {missing}"
        )


def build_and_solve(
    pool: pd.DataFrame,
    previous_lineups,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
):
    model = cp_model.CpModel()

    # x[(player_index, slot)] = 1 iff player occupies that exact slot.
    x = {}

    for i, row in pool.iterrows():
        pos = row["solver_position"]
        for slot in ROSTER_SLOTS:
            if pos in SLOT_ELIGIBILITY[slot]:
                x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")

    # Exactly one player in every FanDuel slot.
    for slot in ROSTER_SLOTS:
        vars_for_slot = [var for (i, s), var in x.items() if s == slot]
        if not vars_for_slot:
            raise RuntimeError(f"No eligible players available for slot {slot}.")
        model.Add(sum(vars_for_slot) == 1)

    # Each player can occupy at most one slot.
    selected = {}
    for i in pool.index:
        player_vars = [var for (j, _), var in x.items() if j == i]
        if not player_vars:
            continue
        y = model.NewBoolVar(f"selected_{i}")
        model.Add(sum(player_vars) == y)
        selected[i] = y

    # Exactly nine unique roster selections.
    model.Add(sum(selected.values()) == 9)

    salary_expr = sum(
        int(round(float(pool.at[i, "salary_solver"]))) * y
        for i, y in selected.items()
    )
    model.Add(salary_expr <= SALARY_CAP)

    if salary_floor > 0:
        model.Add(salary_expr >= salary_floor)

    # Optional extra team concentration cap.
    if max_team > 0:
        for team, group in pool.groupby("team_solver"):
            idxs = [i for i in group.index if i in selected]
            if idxs:
                model.Add(sum(selected[i] for i in idxs) <= max_team)

    # Deterministic portfolio diversification.
    for lineup in previous_lineups:
        overlap_vars = [
            selected[i]
            for i in lineup["player_indices"]
            if i in selected
        ]
        if overlap_vars:
            model.Add(sum(overlap_vars) <= max_overlap)

    # Maximize internal projection. CP-SAT requires integer objective.
    projection_scale = 1000
    objective = sum(
        int(round(float(pool.at[i, "projection_solver"]) * projection_scale)) * y
        for i, y in selected.items()
    )
    model.Maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.cp_model_presolve = True

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    slot_rows = []
    player_indices = []

    for slot in ROSTER_SLOTS:
        chosen = None
        for (i, s), var in x.items():
            if s == slot and solver.Value(var) == 1:
                chosen = i
                break

        if chosen is None:
            raise RuntimeError(f"Solver returned lineup without slot {slot}.")

        row = pool.loc[chosen]
        player_indices.append(chosen)
        slot_rows.append(
            {
                "slot": slot,
                "player": row["player_solver"],
                "position": row["solver_position"],
                "team": row["team_solver"],
                "game": row["game_solver"],
                "salary": int(round(float(row["salary_solver"]))),
                "projection": float(row["projection_solver"]),
                "player_index": int(chosen),
            }
        )

    lineup_df = pd.DataFrame(slot_rows)
    total_salary = int(lineup_df["salary"].sum())
    total_projection = float(lineup_df["projection"].sum())

    return {
        "player_indices": tuple(sorted(player_indices)),
        "slots": lineup_df,
        "total_salary": total_salary,
        "total_projection": total_projection,
    }


def generate_lineups(
    pool: pd.DataFrame,
    n_lineups: int,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
):
    if n_lineups < 1:
        raise RuntimeError("--lineups must be >= 1.")
    if salary_floor < 0 or salary_floor > SALARY_CAP:
        raise RuntimeError(
            f"--salary-floor must be between 0 and {SALARY_CAP}."
        )
    if max_overlap < 0 or max_overlap > 8:
        raise RuntimeError("--max-overlap must be between 0 and 8.")
    if max_team < 0 or max_team > 9:
        raise RuntimeError("--max-team must be between 0 and 9.")

    results = []

    for _ in range(n_lineups):
        lineup = build_and_solve(
            pool=pool,
            previous_lineups=results,
            salary_floor=salary_floor,
            max_overlap=max_overlap,
            max_team=max_team,
        )
        if lineup is None:
            break
        results.append(lineup)

    return results


def audit_lineup(lineup):
    df = lineup["slots"]

    errors = []

    if len(df) != 9:
        errors.append(f"row_count={len(df)}")

    if df["player_index"].nunique() != 9:
        errors.append("duplicate_player")

    if lineup["total_salary"] > SALARY_CAP:
        errors.append("salary_cap")

    if list(df["slot"]) != ROSTER_SLOTS:
        errors.append("slot_order_or_fill")

    for _, row in df.iterrows():
        if row["position"] not in SLOT_ELIGIBILITY[row["slot"]]:
            errors.append(
                f"invalid_slot:{row['player']}:{row['position']}->{row['slot']}"
            )

    return errors


def export_results(
    slate_name: str,
    slate_slug: str,
    lineups,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
):
    safe_slug = normalize_slate(slate_slug)

    lineup_rows = []
    long_rows = []

    for rank, lineup in enumerate(lineups, start=1):
        errors = audit_lineup(lineup)

        wide = {
            "lineup_rank": rank,
            "slate_name": slate_name,
            "slate_slug": slate_slug,
            "total_salary": lineup["total_salary"],
            "salary_left": SALARY_CAP - lineup["total_salary"],
            "total_projection": round(lineup["total_projection"], 6),
            "audit_status": "PASS" if not errors else "FAIL",
            "audit_errors": ";".join(errors),
        }

        for _, r in lineup["slots"].iterrows():
            wide[r["slot"]] = r["player"]

            long_rows.append(
                {
                    "lineup_rank": rank,
                    "slate_name": slate_name,
                    "slate_slug": slate_slug,
                    "slot": r["slot"],
                    "player": r["player"],
                    "position": r["position"],
                    "team": r["team"],
                    "game": r["game"],
                    "salary": r["salary"],
                    "projection": r["projection"],
                    "lineup_total_salary": lineup["total_salary"],
                    "lineup_salary_left": SALARY_CAP - lineup["total_salary"],
                    "lineup_total_projection": lineup["total_projection"],
                    "audit_status": "PASS" if not errors else "FAIL",
                }
            )

        lineup_rows.append(wide)

    summary = pd.DataFrame(lineup_rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = Path(CSV_DIR) / f"fanduel_lineups_{safe_slug}.csv"
    long_path = Path(CSV_DIR) / f"fanduel_lineup_players_{safe_slug}.csv"
    audit_path = Path(CSV_DIR) / f"audit_fanduel_lineups_{safe_slug}.csv"

    summary.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)

    audit = pd.DataFrame(
        [
            {
                "slate_name": slate_name,
                "slate_slug": slate_slug,
                "requested_lineups": len(lineups),
                "generated_lineups": len(lineups),
                "salary_cap": SALARY_CAP,
                "salary_floor": salary_floor,
                "max_overlap": max_overlap,
                "max_team": max_team,
                "all_lineups_pass": bool(
                    len(summary) > 0
                    and summary["audit_status"].eq("PASS").all()
                ),
                "duplicate_lineups": int(
                    summary[ROSTER_SLOTS].duplicated().sum()
                )
                if len(summary) else 0,
            }
        ]
    )
    audit.to_csv(audit_path, index=False)

    return summary, long_df, summary_path, long_path, audit_path


def print_lineups(summary: pd.DataFrame):
    section("GENERATED LINEUPS")

    display_cols = [
        "lineup_rank",
        "total_salary",
        "salary_left",
        "total_projection",
        *ROSTER_SLOTS,
        "audit_status",
    ]
    print(summary[display_cols].to_string(index=False))


def main():
    args = parse_args()

    section("FANDUEL NFL CLASSIC BASELINE SOLVER")
    print(f"Database: {DATABASE_PATH}")
    print(f"Player pool: {SOURCE_TABLE}")
    print(f"Salary cap: ${SALARY_CAP:,}")
    print("Roster: QB / RB / RB / WR / WR / WR / TE / FLEX / DST")
    print("Projection objective: internal model projection")
    print("Deterministic CP-SAT: enabled")
    print("Randomness: disabled")

    all_rows = load_pool()

    if not args.slate:
        list_slates(all_rows)
        print()
        print(
            "Run with --slate, for example:\n"
            "  python fanduel_nfl_lineup_solver.py --slate Main --lineups 20"
        )
        return

    source_slate, pool = select_slate(all_rows, args.slate)
    validate_pool(pool)

    slate_name = str(source_slate["slate_name_solver"].iloc[0])
    slate_slug = str(source_slate["slate_slug_solver"].iloc[0])

    section("SELECTED SLATE")
    print(f"Slate name:            {slate_name}")
    print(f"Slate slug:            {slate_slug}")
    print(f"Contest rows:          {len(source_slate)}")
    print(f"Solver eligible rows:  {len(pool)}")
    print(f"Requested lineups:     {args.lineups}")
    print(f"Salary floor:          ${args.salary_floor:,}")
    print(f"Maximum overlap:       {args.max_overlap}")
    print(
        "Maximum per team:     "
        + ("disabled" if args.max_team == 0 else str(args.max_team))
    )
    print()
    print("Eligible position counts:")
    print(
        pool["solver_position"]
        .value_counts()
        .rename_axis("position")
        .reset_index(name="players")
        .to_string(index=False)
    )

    lineups = generate_lineups(
        pool=pool,
        n_lineups=args.lineups,
        salary_floor=args.salary_floor,
        max_overlap=args.max_overlap,
        max_team=args.max_team,
    )

    if not lineups:
        raise RuntimeError(
            "No feasible lineup found with the requested constraints."
        )

    summary, long_df, summary_path, long_path, audit_path = export_results(
        slate_name=slate_name,
        slate_slug=slate_slug,
        lineups=lineups,
        salary_floor=args.salary_floor,
        max_overlap=args.max_overlap,
        max_team=args.max_team,
    )

    print_lineups(summary)

    section("STRUCTURAL AUDIT")
    print(f"Generated lineups:       {len(summary)}")
    print(f"All lineup audits PASS:  {summary['audit_status'].eq('PASS').all()}")
    print(f"Duplicate lineups:       {int(summary[ROSTER_SLOTS].duplicated().sum())}")
    print(f"Minimum salary used:     ${int(summary['total_salary'].min()):,}")
    print(f"Maximum salary used:     ${int(summary['total_salary'].max()):,}")
    print(
        "Projection range:      "
        f"{summary['total_projection'].min():.3f} - "
        f"{summary['total_projection'].max():.3f}"
    )

    section("EXPORTS")
    print(f"Lineup summary: {summary_path}")
    print(f"Lineup players: {long_path}")
    print(f"Audit:          {audit_path}")

    section("BASELINE SOLVER COMPLETE")
    print(
        "This validates legal roster construction from the solver-ready pool. "
        "Correlation, stacking, leverage, exposure, ceiling, and tournament "
        "portfolio rules should be added as the next layer rather than "
        "changing the upstream data pipeline."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 118)
        print("FANDUEL NFL LINEUP SOLVER FAILED")
        print("=" * 118)
        print(f"{type(exc).__name__}: {exc}")
        raise
