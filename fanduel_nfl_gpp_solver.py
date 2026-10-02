#!/usr/bin/env python3
"""
fanduel_nfl_gpp_solver.py

Deterministic FanDuel NFL GPP correlation solver.

Consumes ONLY:
    SQLite table: fanduel_solver_ready_pool

This file is a NEW layer above the frozen legal-lineup baseline.
It does not modify upstream projections, slate ingestion, or the baseline solver.

FanDuel Classic roster:
    1 QB
    2 RB
    3 WR
    1 TE
    1 FLEX (RB/WR/TE)
    1 DST
    9 total players
    $60,000 salary cap

GPP correlation rules in this first layer:
    - Require the selected QB to be paired with N teammate pass catchers
      (WR/TE only), where N is configurable as 1 or 2.
    - Optional opponent bring-back requiring at least one RB/WR/TE from
      the QB's game opponent.
    - Deterministic CP-SAT optimization.
    - No fuzzy matching.
    - No salary inference.
    - No ownership assumptions.
    - No randomness.
    - No changes to internal projections.
    - Existing legal roster construction remains enforced.

Examples:
    python fanduel_nfl_gpp_solver.py --slate Main --lineups 20

    python fanduel_nfl_gpp_solver.py \
        --slate Main \
        --lineups 20 \
        --qb-stack 2 \
        --bring-back 1 \
        --max-overlap 7

Outputs:
    data/csv/fanduel_gpp_lineups_<slate>.csv
    data/csv/fanduel_gpp_lineup_players_<slate>.csv
    data/csv/audit_fanduel_gpp_lineups_<slate>.csv
"""

from __future__ import annotations

import argparse
import re
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

PASS_CATCHER_POSITIONS = {"WR", "TE"}
BRING_BACK_POSITIONS = {"RB", "WR", "TE"}
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "DST"}


def section(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def normalize_slate(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_team(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z]", "", str(value).upper())


def parse_game_teams(game_value):
    """
    Parse a game key such as:
        ARI@LAC
        ARI @ LAC
        ARI-LAC
        ARI vs LAC
        ARI v LAC

    Returns (away_or_left_team, home_or_right_team) when exactly two
    team tokens can be resolved. Otherwise returns (None, None).
    """
    if pd.isna(game_value):
        return None, None

    raw = str(game_value).upper().strip()

    # Preferred explicit separators.
    patterns = [
        r"^\s*([A-Z]{2,4})\s*@\s*([A-Z]{2,4})\s*$",
        r"^\s*([A-Z]{2,4})\s+VS\.?\s+([A-Z]{2,4})\s*$",
        r"^\s*([A-Z]{2,4})\s+V\s+([A-Z]{2,4})\s*$",
        r"^\s*([A-Z]{2,4})\s*-\s*([A-Z]{2,4})\s*$",
    ]

    for pattern in patterns:
        match = re.match(pattern, raw)
        if match:
            return normalize_team(match.group(1)), normalize_team(match.group(2))

    # Conservative fallback: exactly two uppercase team-like tokens.
    tokens = re.findall(r"\b[A-Z]{2,4}\b", raw)
    if len(tokens) == 2:
        return normalize_team(tokens[0]), normalize_team(tokens[1])

    return None, None


def opponent_from_game(team, game_value):
    team = normalize_team(team)
    left, right = parse_game_teams(game_value)

    if not team or not left or not right:
        return None

    if team == left:
        return right
    if team == right:
        return left

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic FanDuel NFL GPP QB/game-stack solver."
    )
    parser.add_argument(
        "--slate",
        required=False,
        help=(
            "Slate name or slug, e.g. Main, main, '1pm only', "
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
        "--qb-stack",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "Minimum teammate WR/TE pass catchers paired with selected QB. "
            "Allowed: 1 or 2. Default: 1."
        ),
    )
    parser.add_argument(
        "--bring-back",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Require at least one opponent RB/WR/TE from the QB's game. "
            "1=yes, 0=no. Default: 1."
        ),
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
        default=7,
        help=(
            "Maximum players shared with every previously generated lineup. "
            "Default: 7."
        ),
    )
    parser.add_argument(
        "--max-team",
        type=int,
        default=0,
        help=(
            "Optional maximum selected players from one NFL team. "
            "0 disables this extra constraint. Default: 0."
        ),
    )
    return parser.parse_args()


def load_pool():
    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, SOURCE_TABLE):
            raise RuntimeError(
                f"Required table not found: {SOURCE_TABLE}. "
                "Run fanduel_solver_ready_pool.py first."
            )

        df = pd.read_sql_query(
            f'SELECT * FROM "{SOURCE_TABLE}"',
            conn,
        )

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

    df["salary_solver"] = pd.to_numeric(
        df["salary_solver"], errors="coerce"
    )
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

    unique_slugs = matches["slate_slug_solver"].dropna().unique()
    if len(unique_slugs) != 1:
        raise RuntimeError(
            f"Slate request '{requested}' matched multiple slates: "
            f"{list(unique_slugs)}"
        )

    eligible = matches[matches["solver_eligible"].eq(1)].copy()

    eligible = eligible[
        eligible["salary_solver"].notna()
        & eligible["salary_solver"].gt(0)
        & eligible["projection_solver"].notna()
        & np.isfinite(eligible["projection_solver"])
        & eligible["solver_position"].isin(VALID_POSITIONS)
    ].copy()

    eligible = eligible.reset_index(drop=True)
    eligible["player_index"] = eligible.index.astype(int)
    eligible["team_norm"] = eligible["team_solver"].map(normalize_team)
    eligible["opponent_team"] = eligible.apply(
        lambda r: opponent_from_game(
            r["team_solver"],
            r["game_solver"],
        ),
        axis=1,
    )

    return matches, eligible


def validate_pool(df: pd.DataFrame, qb_stack: int, bring_back: int):
    if df.empty:
        raise RuntimeError("No solver-eligible rows remain after validation.")

    dup = df.duplicated(
        ["player_solver", "team_solver", "salary_solver"],
        keep=False,
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
    minimums = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "DST": 1,
    }

    missing = {
        pos: need - counts.get(pos, 0)
        for pos, need in minimums.items()
        if counts.get(pos, 0) < need
    }

    if missing:
        raise RuntimeError(
            f"Selected slate cannot fill required FanDuel slots: {missing}"
        )

    # Validate every eligible QB can be interpreted inside its game.
    bad_qbs = df[
        df["solver_position"].eq("QB")
        & df["opponent_team"].isna()
    ]

    if not bad_qbs.empty:
        cols = [
            "player_solver",
            "team_solver",
            "game_solver",
        ]
        raise RuntimeError(
            "Cannot resolve opponent from game key for eligible QB rows:\n"
            + bad_qbs[cols].to_string(index=False)
        )

    # Audit how many QBs can actually support the requested stack structure.
    viable_qbs = []

    for _, qb in df[df["solver_position"].eq("QB")].iterrows():
        teammates = df[
            df["team_norm"].eq(qb["team_norm"])
            & df["solver_position"].isin(PASS_CATCHER_POSITIONS)
        ]

        opponents = df[
            df["team_norm"].eq(qb["opponent_team"])
            & df["solver_position"].isin(BRING_BACK_POSITIONS)
        ]

        if len(teammates) < qb_stack:
            continue

        if bring_back == 1 and len(opponents) < 1:
            continue

        viable_qbs.append(qb["player_solver"])

    if not viable_qbs:
        raise RuntimeError(
            "No eligible QB can satisfy the requested stack/bring-back "
            "structure on this slate."
        )

    return viable_qbs


def build_and_solve(
    pool: pd.DataFrame,
    previous_lineups,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: int,
):
    model = cp_model.CpModel()

    # x[(player_index, slot)] = 1 iff player occupies exact slot.
    x = {}

    for i, row in pool.iterrows():
        position = row["solver_position"]

        for slot in ROSTER_SLOTS:
            if position in SLOT_ELIGIBILITY[slot]:
                x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")

    # Exactly one player per slot.
    for slot in ROSTER_SLOTS:
        slot_vars = [
            var
            for (i, s), var in x.items()
            if s == slot
        ]

        if not slot_vars:
            raise RuntimeError(
                f"No eligible players available for slot {slot}."
            )

        model.Add(sum(slot_vars) == 1)

    # At most one slot per player, represented by selected[i].
    selected = {}

    for i in pool.index:
        player_vars = [
            var
            for (j, _), var in x.items()
            if j == i
        ]

        if not player_vars:
            continue

        y = model.NewBoolVar(f"selected_{i}")
        model.Add(sum(player_vars) == y)
        selected[i] = y

    model.Add(sum(selected.values()) == 9)

    salary_expr = sum(
        int(round(float(pool.at[i, "salary_solver"]))) * y
        for i, y in selected.items()
    )

    model.Add(salary_expr <= SALARY_CAP)

    if salary_floor > 0:
        model.Add(salary_expr >= salary_floor)

    # Optional team concentration cap.
    if max_team > 0:
        for team, group in pool.groupby("team_norm"):
            indices = [
                i
                for i in group.index
                if i in selected
            ]

            if indices:
                model.Add(
                    sum(selected[i] for i in indices) <= max_team
                )

    # ------------------------------------------------------------------
    # QB correlation engine
    # ------------------------------------------------------------------
    #
    # For each QB:
    #   if QB selected:
    #       selected teammate WR/TE >= qb_stack
    #       selected opponent RB/WR/TE >= 1 when bring_back == 1
    #
    # Exactly one QB is selected by roster construction, so these
    # implications apply only to the active lineup QB.
    # ------------------------------------------------------------------

    for qb_idx, qb in pool[
        pool["solver_position"].eq("QB")
    ].iterrows():
        if qb_idx not in selected:
            continue

        teammate_indices = [
            i
            for i, row in pool.iterrows()
            if (
                i in selected
                and row["team_norm"] == qb["team_norm"]
                and row["solver_position"] in PASS_CATCHER_POSITIONS
            )
        ]

        if teammate_indices:
            model.Add(
                sum(selected[i] for i in teammate_indices)
                >= qb_stack * selected[qb_idx]
            )
        else:
            # If the QB cannot satisfy the stack, prohibit him.
            model.Add(selected[qb_idx] == 0)

        if bring_back == 1:
            opponent_indices = [
                i
                for i, row in pool.iterrows()
                if (
                    i in selected
                    and row["team_norm"] == qb["opponent_team"]
                    and row["solver_position"] in BRING_BACK_POSITIONS
                )
            ]

            if opponent_indices:
                model.Add(
                    sum(selected[i] for i in opponent_indices)
                    >= selected[qb_idx]
                )
            else:
                model.Add(selected[qb_idx] == 0)

    # Deterministic lineup diversification against every prior lineup.
    for lineup in previous_lineups:
        overlap_vars = [
            selected[i]
            for i in lineup["player_indices"]
            if i in selected
        ]

        if overlap_vars:
            model.Add(
                sum(overlap_vars) <= max_overlap
            )

    # Maximize internal mean projection.
    projection_scale = 1000

    objective = sum(
        int(
            round(
                float(pool.at[i, "projection_solver"])
                * projection_scale
            )
        ) * y
        for i, y in selected.items()
    )

    model.Maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.cp_model_presolve = True

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
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
            raise RuntimeError(
                f"Solver returned lineup without slot {slot}."
            )

        row = pool.loc[chosen]
        player_indices.append(chosen)

        slot_rows.append(
            {
                "slot": slot,
                "player": row["player_solver"],
                "position": row["solver_position"],
                "team": row["team_solver"],
                "team_norm": row["team_norm"],
                "opponent_team": row["opponent_team"],
                "game": row["game_solver"],
                "salary": int(
                    round(float(row["salary_solver"]))
                ),
                "projection": float(
                    row["projection_solver"]
                ),
                "player_index": int(chosen),
            }
        )

    lineup_df = pd.DataFrame(slot_rows)

    return {
        "player_indices": tuple(sorted(player_indices)),
        "slots": lineup_df,
        "total_salary": int(lineup_df["salary"].sum()),
        "total_projection": float(
            lineup_df["projection"].sum()
        ),
    }


def analyze_stack(lineup, required_qb_stack: int, required_bring_back: int):
    df = lineup["slots"].copy()

    qb_rows = df[df["slot"].eq("QB")]

    if len(qb_rows) != 1:
        return {
            "qb": "",
            "qb_team": "",
            "qb_opponent": "",
            "qb_game": "",
            "stack_pass_catchers": "",
            "stack_pass_catcher_count": 0,
            "bring_back_players": "",
            "bring_back_count": 0,
            "stack_type": "INVALID_QB_COUNT",
            "stack_audit": "FAIL",
        }

    qb = qb_rows.iloc[0]
    qb_team = normalize_team(qb["team"])
    qb_opponent = qb["opponent_team"]

    teammate_pc = df[
        df["team_norm"].eq(qb_team)
        & df["position"].isin(PASS_CATCHER_POSITIONS)
    ]

    opponent_skill = df[
        df["team_norm"].eq(qb_opponent)
        & df["position"].isin(BRING_BACK_POSITIONS)
    ]

    teammate_names = sorted(
        teammate_pc["player"].astype(str).tolist()
    )
    opponent_names = sorted(
        opponent_skill["player"].astype(str).tolist()
    )

    pc_count = len(teammate_names)
    bb_count = len(opponent_names)

    stack_type = (
        f"QB+{pc_count}"
        + (f"+{bb_count}BR" if bb_count > 0 else "")
    )

    passes = (
        pc_count >= required_qb_stack
        and (
            required_bring_back == 0
            or bb_count >= 1
        )
    )

    return {
        "qb": qb["player"],
        "qb_team": qb["team"],
        "qb_opponent": qb_opponent,
        "qb_game": qb["game"],
        "stack_pass_catchers": " | ".join(teammate_names),
        "stack_pass_catcher_count": pc_count,
        "bring_back_players": " | ".join(opponent_names),
        "bring_back_count": bb_count,
        "stack_type": stack_type,
        "stack_audit": "PASS" if passes else "FAIL",
    }


def audit_lineup(
    lineup,
    required_qb_stack: int,
    required_bring_back: int,
):
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

    stack = analyze_stack(
        lineup,
        required_qb_stack=required_qb_stack,
        required_bring_back=required_bring_back,
    )

    if stack["stack_audit"] != "PASS":
        errors.append("stack_correlation")

    return errors, stack


def generate_lineups(
    pool: pd.DataFrame,
    n_lineups: int,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: int,
):
    if n_lineups < 1:
        raise RuntimeError("--lineups must be >= 1.")

    if salary_floor < 0 or salary_floor > SALARY_CAP:
        raise RuntimeError(
            f"--salary-floor must be between 0 and {SALARY_CAP}."
        )

    if max_overlap < 0 or max_overlap > 8:
        raise RuntimeError(
            "--max-overlap must be between 0 and 8."
        )

    if max_team < 0 or max_team > 9:
        raise RuntimeError(
            "--max-team must be between 0 and 9."
        )

    results = []

    for _ in range(n_lineups):
        lineup = build_and_solve(
            pool=pool,
            previous_lineups=results,
            salary_floor=salary_floor,
            max_overlap=max_overlap,
            max_team=max_team,
            qb_stack=qb_stack,
            bring_back=bring_back,
        )

        if lineup is None:
            break

        results.append(lineup)

    return results


def export_results(
    slate_name: str,
    slate_slug: str,
    lineups,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: int,
):
    safe_slug = normalize_slate(slate_slug)

    lineup_rows = []
    long_rows = []

    for rank, lineup in enumerate(lineups, start=1):
        errors, stack = audit_lineup(
            lineup,
            required_qb_stack=qb_stack,
            required_bring_back=bring_back,
        )

        wide = {
            "lineup_rank": rank,
            "slate_name": slate_name,
            "slate_slug": slate_slug,
            "total_salary": lineup["total_salary"],
            "salary_left": SALARY_CAP - lineup["total_salary"],
            "total_projection": round(
                lineup["total_projection"],
                6,
            ),
            "qb": stack["qb"],
            "qb_team": stack["qb_team"],
            "qb_opponent": stack["qb_opponent"],
            "qb_game": stack["qb_game"],
            "stack_pass_catchers": stack["stack_pass_catchers"],
            "stack_pass_catcher_count": stack["stack_pass_catcher_count"],
            "bring_back_players": stack["bring_back_players"],
            "bring_back_count": stack["bring_back_count"],
            "stack_type": stack["stack_type"],
            "stack_audit": stack["stack_audit"],
            "audit_status": "PASS" if not errors else "FAIL",
            "audit_errors": ";".join(errors),
        }

        for _, row in lineup["slots"].iterrows():
            wide[row["slot"]] = row["player"]

            long_rows.append(
                {
                    "lineup_rank": rank,
                    "slate_name": slate_name,
                    "slate_slug": slate_slug,
                    "slot": row["slot"],
                    "player": row["player"],
                    "position": row["position"],
                    "team": row["team"],
                    "game": row["game"],
                    "salary": row["salary"],
                    "projection": row["projection"],
                    "lineup_total_salary": lineup["total_salary"],
                    "lineup_salary_left": SALARY_CAP - lineup["total_salary"],
                    "lineup_total_projection": lineup["total_projection"],
                    "lineup_qb": stack["qb"],
                    "lineup_qb_team": stack["qb_team"],
                    "lineup_qb_opponent": stack["qb_opponent"],
                    "lineup_stack_type": stack["stack_type"],
                    "lineup_stack_pass_catchers": stack[
                        "stack_pass_catchers"
                    ],
                    "lineup_bring_back_players": stack[
                        "bring_back_players"
                    ],
                    "audit_status": "PASS" if not errors else "FAIL",
                }
            )

        lineup_rows.append(wide)

    summary = pd.DataFrame(lineup_rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = (
        Path(CSV_DIR)
        / f"fanduel_gpp_lineups_{safe_slug}.csv"
    )
    long_path = (
        Path(CSV_DIR)
        / f"fanduel_gpp_lineup_players_{safe_slug}.csv"
    )
    audit_path = (
        Path(CSV_DIR)
        / f"audit_fanduel_gpp_lineups_{safe_slug}.csv"
    )

    summary.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)

    duplicate_lineups = (
        int(summary[ROSTER_SLOTS].duplicated().sum())
        if len(summary)
        else 0
    )

    audit = pd.DataFrame(
        [
            {
                "slate_name": slate_name,
                "slate_slug": slate_slug,
                "generated_lineups": len(lineups),
                "salary_cap": SALARY_CAP,
                "salary_floor": salary_floor,
                "max_overlap": max_overlap,
                "max_team": max_team,
                "required_qb_stack": qb_stack,
                "required_bring_back": bring_back,
                "all_lineups_pass": bool(
                    len(summary) > 0
                    and summary["audit_status"].eq("PASS").all()
                ),
                "all_stack_audits_pass": bool(
                    len(summary) > 0
                    and summary["stack_audit"].eq("PASS").all()
                ),
                "duplicate_lineups": duplicate_lineups,
                "minimum_salary_used": (
                    int(summary["total_salary"].min())
                    if len(summary)
                    else None
                ),
                "maximum_salary_used": (
                    int(summary["total_salary"].max())
                    if len(summary)
                    else None
                ),
                "minimum_projection": (
                    float(summary["total_projection"].min())
                    if len(summary)
                    else None
                ),
                "maximum_projection": (
                    float(summary["total_projection"].max())
                    if len(summary)
                    else None
                ),
            }
        ]
    )

    audit.to_csv(audit_path, index=False)

    return (
        summary,
        long_df,
        summary_path,
        long_path,
        audit_path,
    )


def print_lineups(summary: pd.DataFrame):
    section("GENERATED GPP LINEUPS")

    display_cols = [
        "lineup_rank",
        "total_salary",
        "salary_left",
        "total_projection",
        "QB",
        "stack_pass_catchers",
        "bring_back_players",
        "stack_type",
        "RB1",
        "RB2",
        "WR1",
        "WR2",
        "WR3",
        "TE",
        "FLEX",
        "DST",
        "audit_status",
    ]

    print(
        summary[display_cols].to_string(index=False)
    )


def print_stack_distribution(summary: pd.DataFrame):
    section("STACK DISTRIBUTION")

    distribution = (
        summary.groupby(
            ["stack_type"],
            dropna=False,
        )
        .size()
        .rename("lineups")
        .reset_index()
        .sort_values(
            ["lineups", "stack_type"],
            ascending=[False, True],
        )
    )

    print(distribution.to_string(index=False))

    section("QB EXPOSURE")

    qb_exp = (
        summary["QB"]
        .value_counts()
        .rename_axis("QB")
        .reset_index(name="lineups")
    )

    qb_exp["exposure_pct"] = (
        qb_exp["lineups"]
        / len(summary)
        * 100.0
    ).round(2)

    print(qb_exp.to_string(index=False))


def main():
    args = parse_args()

    section("FANDUEL NFL GPP CORRELATION SOLVER")
    print(f"Database: {DATABASE_PATH}")
    print(f"Player pool: {SOURCE_TABLE}")
    print(f"Salary cap: ${SALARY_CAP:,}")
    print("Roster: QB / RB / RB / WR / WR / WR / TE / FLEX / DST")
    print("Projection objective: internal model projection")
    print("QB stack positions: WR/TE")
    print("Bring-back positions: RB/WR/TE")
    print("Deterministic CP-SAT: enabled")
    print("Randomness: disabled")
    print("Fuzzy matching: disabled")

    all_rows = load_pool()

    if not args.slate:
        list_slates(all_rows)
        print()
        print(
            "Example:\n"
            "  python fanduel_nfl_gpp_solver.py "
            "--slate Main --lineups 20 "
            "--qb-stack 1 --bring-back 1"
        )
        return

    source_slate, pool = select_slate(
        all_rows,
        args.slate,
    )

    viable_qbs = validate_pool(
        pool,
        qb_stack=args.qb_stack,
        bring_back=args.bring_back,
    )

    slate_name = str(
        source_slate["slate_name_solver"].iloc[0]
    )
    slate_slug = str(
        source_slate["slate_slug_solver"].iloc[0]
    )

    section("SELECTED SLATE")
    print(f"Slate name:              {slate_name}")
    print(f"Slate slug:              {slate_slug}")
    print(f"Contest rows:            {len(source_slate)}")
    print(f"Solver eligible rows:    {len(pool)}")
    print(f"Requested lineups:       {args.lineups}")
    print(f"Required QB stack:       {args.qb_stack} WR/TE")
    print(
        "Required bring-back:    "
        + ("YES" if args.bring_back == 1 else "NO")
    )
    print(f"Salary floor:            ${args.salary_floor:,}")
    print(f"Maximum overlap:         {args.max_overlap}")
    print(
        "Maximum per team:       "
        + (
            "disabled"
            if args.max_team == 0
            else str(args.max_team)
        )
    )
    print(f"Stack-viable QBs:        {len(viable_qbs)}")

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
        qb_stack=args.qb_stack,
        bring_back=args.bring_back,
    )

    if not lineups:
        raise RuntimeError(
            "No feasible GPP lineup found with the requested "
            "stack/correlation constraints."
        )

    (
        summary,
        long_df,
        summary_path,
        long_path,
        audit_path,
    ) = export_results(
        slate_name=slate_name,
        slate_slug=slate_slug,
        lineups=lineups,
        salary_floor=args.salary_floor,
        max_overlap=args.max_overlap,
        max_team=args.max_team,
        qb_stack=args.qb_stack,
        bring_back=args.bring_back,
    )

    print_lineups(summary)
    print_stack_distribution(summary)

    section("STRUCTURAL + CORRELATION AUDIT")
    print(f"Generated lineups:          {len(summary)}")
    print(
        "All lineup audits PASS:     "
        f"{summary['audit_status'].eq('PASS').all()}"
    )
    print(
        "All stack audits PASS:      "
        f"{summary['stack_audit'].eq('PASS').all()}"
    )
    print(
        "Duplicate lineups:          "
        f"{int(summary[ROSTER_SLOTS].duplicated().sum())}"
    )
    print(
        "Minimum salary used:        "
        f"${int(summary['total_salary'].min()):,}"
    )
    print(
        "Maximum salary used:        "
        f"${int(summary['total_salary'].max()):,}"
    )
    print(
        "Projection range:           "
        f"{summary['total_projection'].min():.3f} - "
        f"{summary['total_projection'].max():.3f}"
    )
    print(
        "Minimum stack catchers:     "
        f"{int(summary['stack_pass_catcher_count'].min())}"
    )

    if args.bring_back == 1:
        print(
            "Minimum bring-back count:  "
            f"{int(summary['bring_back_count'].min())}"
        )

    section("EXPORTS")
    print(f"GPP lineup summary: {summary_path}")
    print(f"GPP lineup players: {long_path}")
    print(f"GPP audit:          {audit_path}")

    section("GPP CORRELATION LAYER COMPLETE")
    print(
        "This layer adds deterministic QB/pass-catcher and opponent "
        "bring-back correlation while preserving the frozen legal-lineup "
        "baseline. Exposure, ceiling weighting, leverage, and portfolio "
        "construction remain separate future layers."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 122)
        print("FANDUEL NFL GPP SOLVER FAILED")
        print("=" * 122)
        print(f"{type(exc).__name__}: {exc}")
        raise
