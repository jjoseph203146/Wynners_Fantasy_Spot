#!/usr/bin/env python3
"""
fanduel_nfl_gpp_objective_solver.py

Deterministic FanDuel NFL GPP objective + portfolio solver.

NEW LAYER ONLY.
Does not modify frozen baseline solvers, projections, slate ingestion,
or the solver-ready pool.

Consumes:
    SQLite: fanduel_solver_ready_pool
    Parquet: nfl_fanduel_player_pool.parquet

Objective architecture:
    1. Preserve legal FanDuel roster and salary constraints.
    2. Preserve QB stacking / bring-back / overlap / exposure constraints.
    3. Use historically promoted, position-specific GPP rankings for
       offensive tournament differentiation.
    4. Use internal FanDuel projection as deterministic secondary tiebreak.
    5. QB and DST remain projection-driven.
    6. No fuzzy matching, salary inference, randomness, ownership assumptions,
       or invented ceiling values.

Promoted GPP paths:
    QB  -> ridge / internal projection
    RB  -> gpp_score_20
    WR  -> gpp_score_15 and gpp_score_25, combined by average percentile rank
    TE  -> gpp_score_15 and gpp_score_20, combined by average percentile rank
           (TE10 remains audit/research because Main coverage was 61/62)
    DST -> internal projection

Important:
    GPP scores are normalized/ranking signals, NOT FanDuel point values.
    Therefore this solver does not add raw GPP scores to fantasy projections.
    It converts validated GPP signals to within-position integer ranks and
    optimizes those ranks lexicographically ahead of projection.

Run:
    python fanduel_nfl_gpp_objective_solver.py \
        --slate Main \
        --lineups 20 \
        --qb-stack 2 \
        --bring-back 1 \
        --max-overlap 7
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise SystemExit(
        "OR-Tools is required. Activate the project venv and run: pip install ortools"
    ) from exc

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


SOLVER_TABLE = "fanduel_solver_ready_pool"
OFFENSE_PARQUET = Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet"

SALARY_CAP = 60000
ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "DST"}

# Large enough that one GPP-rank point always dominates the maximum possible
# projection tiebreak difference across a nine-player lineup.
GPP_PRIMARY_MULTIPLIER = 1_000_000
PROJECTION_SCALE = 1_000


def section(title: str) -> None:
    print()
    print("=" * 126)
    print(title)
    print("=" * 126)


def normalize_slate(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_team(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z]", "", str(value).upper())


def safe_slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return out or "slate"


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_args():
    p = argparse.ArgumentParser(
        description="Deterministic FanDuel NFL GPP objective portfolio solver."
    )
    p.add_argument("--slate", default="Main")
    p.add_argument("--lineups", type=int, default=20)
    p.add_argument("--salary-floor", type=int, default=0)
    p.add_argument("--max-overlap", type=int, default=7)
    p.add_argument("--max-team", type=int, default=0)
    p.add_argument("--qb-stack", type=int, choices=[1, 2], default=2)
    p.add_argument("--bring-back", type=int, choices=[0, 1], default=1)
    p.add_argument("--max-player-exposure", type=float, default=0.60)
    p.add_argument("--max-qb-exposure", type=float, default=0.40)
    p.add_argument("--max-dst-exposure", type=float, default=0.40)
    return p.parse_args()


def validate_args(args) -> None:
    if args.lineups < 1:
        raise ValueError("--lineups must be >= 1")
    if not 0 <= args.salary_floor <= SALARY_CAP:
        raise ValueError(f"--salary-floor must be between 0 and {SALARY_CAP}")
    if not 0 <= args.max_overlap <= 8:
        raise ValueError("--max-overlap must be between 0 and 8")
    if args.max_team < 0:
        raise ValueError("--max-team must be >= 0")
    for name in [
        "max_player_exposure",
        "max_qb_exposure",
        "max_dst_exposure",
    ]:
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0 and <= 1")


def load_solver_pool() -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (SOLVER_TABLE,),
        ).fetchone()
        if not exists:
            raise RuntimeError(f"Missing SQLite table: {SOLVER_TABLE}")
        return pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)


def select_slate(df: pd.DataFrame, requested: str) -> pd.DataFrame:
    target = normalize_slate(requested)
    mask = (
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    )
    out = df.loc[mask].copy()
    if out.empty:
        available = sorted(df["slate_name_solver"].dropna().astype(str).unique())
        raise RuntimeError(f"Slate not found: {requested}. Available: {available}")
    return out


def attach_validated_gpp_fields(pool: pd.DataFrame) -> pd.DataFrame:
    if not OFFENSE_PARQUET.exists():
        raise RuntimeError(f"Missing offensive parquet: {OFFENSE_PARQUET}")

    offense = pd.read_parquet(OFFENSE_PARQUET)

    required = [
        "fd_name",
        "team",
        "fd_position",
        "ridge_projection",
        "gpp_score_15",
        "gpp_score_20",
        "gpp_score_25",
    ]
    missing = [c for c in required if c not in offense.columns]
    if missing:
        raise RuntimeError(f"Offensive parquet missing fields: {missing}")

    offense = offense.copy()
    offense["_name_key"] = offense["fd_name"].map(normalize_name)
    offense["_team_key"] = offense["team"].map(normalize_team)
    offense["_pos_key"] = offense["fd_position"].astype(str).str.upper().str.strip()

    dup = offense.duplicated(["_name_key", "_team_key", "_pos_key"], keep=False)
    if dup.any():
        raise RuntimeError(
            "Duplicate exact offensive name+team+position keys detected."
        )

    keep = [
        "_name_key",
        "_team_key",
        "_pos_key",
        "ridge_projection",
        "gpp_score_15",
        "gpp_score_20",
        "gpp_score_25",
    ]
    offense = offense[keep].rename(
        columns={
            "ridge_projection": "gpp_ridge_projection",
            "gpp_score_15": "gpp_score_15_attached",
            "gpp_score_20": "gpp_score_20_attached",
            "gpp_score_25": "gpp_score_25_attached",
        }
    )

    out = pool.copy()
    out["_name_key"] = out["player_solver"].map(normalize_name)
    out["_team_key"] = out["team_solver"].map(normalize_team)
    out["_pos_key"] = out["solver_position"].astype(str).str.upper().str.strip()

    offense_mask = out["solver_position"].isin(["QB", "RB", "WR", "TE"])
    off = out.loc[offense_mask].merge(
        offense,
        on=["_name_key", "_team_key", "_pos_key"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not off["_merge"].eq("both").all():
        bad = off.loc[
            off["_merge"].ne("both"),
            ["player_solver", "team_solver", "solver_position"],
        ]
        raise RuntimeError(
            "Incomplete exact offensive GPP attachment:\n"
            + bad.to_string(index=False)
        )

    dst = out.loc[~offense_mask].copy()
    for col in [
        "gpp_ridge_projection",
        "gpp_score_15_attached",
        "gpp_score_20_attached",
        "gpp_score_25_attached",
    ]:
        dst[col] = np.nan
    dst["_merge"] = "DST"

    combined = pd.concat([off, dst], ignore_index=True, sort=False)
    return combined


def percentile_points(series: pd.Series) -> pd.Series:
    """
    Deterministic within-position percentile points in [0, 1].
    Higher raw score -> higher percentile.
    """
    s = finite_numeric(series)
    if s.isna().any():
        raise RuntimeError("Validated GPP field contains missing/nonfinite values.")
    if len(s) == 1:
        return pd.Series([1.0], index=s.index)
    ranks = s.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(s) - 1.0)


def build_gpp_objective_fields(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["gpp_objective_signal"] = np.nan
    out["gpp_objective_method"] = ""
    out["gpp_objective_rank_points"] = 0

    for pos in ["QB", "RB", "WR", "TE", "DST"]:
        mask = out["solver_position"].eq(pos)
        sub = out.loc[mask].copy()

        if sub.empty:
            continue

        if pos == "QB":
            raw = finite_numeric(sub["projection_solver"])
            method = "QB_PROJECTION"
            signal = percentile_points(raw)

        elif pos == "RB":
            raw = finite_numeric(sub["gpp_score_20_attached"])
            if raw.isna().any():
                raise RuntimeError("RB gpp_score_20 coverage is incomplete.")
            method = "RB_GPP20_PROMOTED"
            signal = percentile_points(raw)

        elif pos == "WR":
            s15 = finite_numeric(sub["gpp_score_15_attached"])
            s25 = finite_numeric(sub["gpp_score_25_attached"])
            if s15.isna().any() or s25.isna().any():
                raise RuntimeError("WR gpp_score_15/25 coverage is incomplete.")
            method = "WR_GPP15_GPP25_PROMOTED"
            signal = (percentile_points(s15) + percentile_points(s25)) / 2.0

        elif pos == "TE":
            s15 = finite_numeric(sub["gpp_score_15_attached"])
            s20 = finite_numeric(sub["gpp_score_20_attached"])
            if s15.isna().any() or s20.isna().any():
                raise RuntimeError("TE gpp_score_15/20 coverage is incomplete.")
            method = "TE_GPP15_GPP20_PROMOTED"
            signal = (percentile_points(s15) + percentile_points(s20)) / 2.0

        else:  # DST
            raw = finite_numeric(sub["projection_solver"])
            method = "DST_PROJECTION"
            signal = percentile_points(raw)

        # Integer rank points, preserving only deterministic ordering.
        rank_points = np.rint(signal * 100_000).astype(int)

        out.loc[sub.index, "gpp_objective_signal"] = signal
        out.loc[sub.index, "gpp_objective_method"] = method
        out.loc[sub.index, "gpp_objective_rank_points"] = rank_points

    if out["gpp_objective_signal"].isna().any():
        raise RuntimeError("GPP objective signal creation is incomplete.")

    return out


def parse_game(game_value: str):
    """
    Derive teams from a game key/string such as ARI@LAC.
    Returns (away, home) or (None, None).
    """
    if pd.isna(game_value):
        return None, None
    text = str(game_value).upper().strip()
    m = re.search(r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b", text)
    if not m:
        return None, None
    return normalize_team(m.group(1)), normalize_team(m.group(2))


def add_opponents(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    opponents = []

    for _, row in out.iterrows():
        team = normalize_team(row["team_solver"])
        away, home = parse_game(row["game_solver"])
        if not away or not home:
            opponents.append("")
        elif team == away:
            opponents.append(home)
        elif team == home:
            opponents.append(away)
        else:
            opponents.append("")

    out["opponent_solver"] = opponents

    qbs = out[out["solver_position"].eq("QB")]
    bad = qbs[qbs["opponent_solver"].eq("")]
    if not bad.empty:
        raise RuntimeError(
            "Could not resolve opponent for eligible QB rows:\n"
            + bad[["player_solver", "team_solver", "game_solver"]].to_string(index=False)
        )
    return out


def eligible_for_slot(position: str, slot: str) -> bool:
    if slot == "QB":
        return position == "QB"
    if slot in {"RB1", "RB2"}:
        return position == "RB"
    if slot in {"WR1", "WR2", "WR3"}:
        return position == "WR"
    if slot == "TE":
        return position == "TE"
    if slot == "FLEX":
        return position in {"RB", "WR", "TE"}
    if slot == "DST":
        return position == "DST"
    return False


def cap_count(n_lineups: int, exposure: float) -> int:
    return max(1, int(math.floor(n_lineups * exposure + 1e-12)))


def player_cap(row, args) -> int:
    pos = row["solver_position"]
    if pos == "QB":
        return cap_count(args.lineups, args.max_qb_exposure)
    if pos == "DST":
        return cap_count(args.lineups, args.max_dst_exposure)
    return cap_count(args.lineups, args.max_player_exposure)


def solve_one(
    pool: pd.DataFrame,
    args,
    exposure_counts: dict[int, int],
    prior_lineups: list[set[int]],
):
    model = cp_model.CpModel()

    # x[(row_index, slot)] = selected in that exact roster slot.
    x = {}
    for i, row in pool.iterrows():
        pos = row["solver_position"]
        for slot in ROSTER_SLOTS:
            if eligible_for_slot(pos, slot):
                x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")

    # Exactly one player per roster slot.
    for slot in ROSTER_SLOTS:
        vars_for_slot = [var for (i, s), var in x.items() if s == slot]
        if not vars_for_slot:
            raise RuntimeError(f"No eligible candidates for roster slot {slot}")
        model.Add(sum(vars_for_slot) == 1)

    # A player may occupy at most one slot.
    selected = {}
    for i in pool.index:
        vars_for_player = [var for (j, _), var in x.items() if j == i]
        if vars_for_player:
            y = model.NewBoolVar(f"selected_{i}")
            model.Add(sum(vars_for_player) == y)
            selected[i] = y

    model.Add(sum(selected.values()) == 9)

    # Salary.
    salary_expr = sum(
        int(pool.at[i, "salary_solver"]) * y for i, y in selected.items()
    )
    model.Add(salary_expr <= SALARY_CAP)
    if args.salary_floor > 0:
        model.Add(salary_expr >= args.salary_floor)

    # Optional team concentration cap (strategy only; disabled by default).
    if args.max_team > 0:
        for team in sorted(pool["team_solver"].dropna().astype(str).unique()):
            idxs = pool.index[pool["team_solver"].eq(team)].tolist()
            vars_team = [selected[i] for i in idxs if i in selected]
            if vars_team:
                model.Add(sum(vars_team) <= args.max_team)

    # Portfolio exposure hard caps.
    for i, y in selected.items():
        used = exposure_counts.get(i, 0)
        cap = player_cap(pool.loc[i], args)
        if used >= cap:
            model.Add(y == 0)

    # QB stack + bring-back implications.
    qb_indices = pool.index[pool["solver_position"].eq("QB")].tolist()

    for qi in qb_indices:
        if qi not in selected:
            continue

        qteam = pool.at[qi, "team_solver"]
        opp = pool.at[qi, "opponent_solver"]

        stack_idxs = pool.index[
            pool["team_solver"].eq(qteam)
            & pool["solver_position"].isin(["WR", "TE"])
        ].tolist()
        stack_vars = [selected[i] for i in stack_idxs if i in selected]

        if len(stack_vars) < args.qb_stack:
            model.Add(selected[qi] == 0)
        else:
            model.Add(sum(stack_vars) >= args.qb_stack * selected[qi])

        if args.bring_back:
            br_idxs = pool.index[
                pool["team_solver"].eq(opp)
                & pool["solver_position"].isin(["RB", "WR", "TE"])
            ].tolist()
            br_vars = [selected[i] for i in br_idxs if i in selected]
            if not br_vars:
                model.Add(selected[qi] == 0)
            else:
                model.Add(sum(br_vars) >= selected[qi])

    # Diversification / duplicate prevention.
    for prior in prior_lineups:
        overlap_vars = [selected[i] for i in prior if i in selected]
        model.Add(sum(overlap_vars) <= args.max_overlap)

    # Lexicographic-style integer objective:
    # validated within-position GPP rank points dominate projection;
    # projection remains deterministic secondary tiebreak.
    gpp_expr = sum(
        int(pool.at[i, "gpp_objective_rank_points"]) * y
        for i, y in selected.items()
    )
    projection_expr = sum(
        int(round(float(pool.at[i, "projection_solver"]) * PROJECTION_SCALE)) * y
        for i, y in selected.items()
    )

    model.Maximize(
        GPP_PRIMARY_MULTIPLIER * gpp_expr + projection_expr
    )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = []
    for (i, slot), var in x.items():
        if solver.Value(var) == 1:
            chosen.append((i, slot))

    if len(chosen) != 9:
        raise RuntimeError(f"Solver returned {len(chosen)} players instead of 9.")

    return chosen


def audit_stack(lineup_df: pd.DataFrame, args):
    qb = lineup_df[lineup_df["slot"].eq("QB")]
    if len(qb) != 1:
        return {
            "qb": "",
            "qb_team": "",
            "qb_opponent": "",
            "stack_pass_catchers": "",
            "stack_count": 0,
            "bring_back_players": "",
            "bring_back_count": 0,
            "stack_type": "",
            "stack_audit": "FAIL_QB_COUNT",
        }

    q = qb.iloc[0]
    qteam = q["team_solver"]
    opp = q["opponent_solver"]

    stack = lineup_df[
        lineup_df["team_solver"].eq(qteam)
        & lineup_df["solver_position"].isin(["WR", "TE"])
    ]
    bring = lineup_df[
        lineup_df["team_solver"].eq(opp)
        & lineup_df["solver_position"].isin(["RB", "WR", "TE"])
    ]

    stack_count = len(stack)
    bring_count = len(bring)

    ok = stack_count >= args.qb_stack
    if args.bring_back:
        ok = ok and bring_count >= 1

    return {
        "qb": q["player_solver"],
        "qb_team": qteam,
        "qb_opponent": opp,
        "stack_pass_catchers": " | ".join(stack["player_solver"].astype(str)),
        "stack_count": stack_count,
        "bring_back_players": " | ".join(bring["player_solver"].astype(str)),
        "bring_back_count": bring_count,
        "stack_type": f"QB+{args.qb_stack}" + ("+1BR" if args.bring_back else ""),
        "stack_audit": "PASS" if ok else "FAIL",
    }


def main():
    args = parse_args()
    validate_args(args)

    section("FANDUEL NFL GPP OBJECTIVE PORTFOLIO SOLVER")
    print(f"Database: {DATABASE_PATH}")
    print(f"Solver table: {SOLVER_TABLE}")
    print(f"Offense parquet: {OFFENSE_PARQUET}")
    print(f"Slate: {args.slate}")
    print(f"Requested lineups: {args.lineups}")
    print(f"QB stack: {args.qb_stack}")
    print(f"Bring-back: {args.bring_back}")
    print(f"Max overlap: {args.max_overlap}")
    print(f"Salary floor: {args.salary_floor}")
    print(f"Max team: {args.max_team} (0 = disabled)")
    print(f"General exposure: {args.max_player_exposure:.0%}")
    print(f"QB exposure: {args.max_qb_exposure:.0%}")
    print(f"DST exposure: {args.max_dst_exposure:.0%}")
    print("Objective: validated GPP rank FIRST, internal projection SECOND")
    print("Frozen baseline solver modifications: NONE")

    all_pool = load_solver_pool()
    slate = select_slate(all_pool, args.slate)

    eligible = slate[
        pd.to_numeric(slate["solver_eligible"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    ].copy()

    eligible["salary_solver"] = finite_numeric(eligible["salary_solver"])
    eligible["projection_solver"] = finite_numeric(eligible["projection_solver"])

    if eligible["salary_solver"].isna().any():
        raise RuntimeError("Eligible pool contains invalid salary.")
    if eligible["projection_solver"].isna().any():
        raise RuntimeError("Eligible pool contains invalid projection.")
    if not eligible["solver_position"].isin(VALID_POSITIONS).all():
        raise RuntimeError("Eligible pool contains invalid position.")

    eligible["salary_solver"] = eligible["salary_solver"].astype(int)

    # Reject duplicate contest-player records.
    dup = eligible.duplicated(
        ["player_solver", "team_solver", "salary_solver"],
        keep=False,
    )
    if dup.any():
        raise RuntimeError(
            "Duplicate eligible player/team/salary rows detected:\n"
            + eligible.loc[
                dup,
                ["player_solver", "team_solver", "solver_position", "salary_solver"],
            ].to_string(index=False)
        )

    counts = eligible["solver_position"].value_counts().to_dict()
    minimums = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    for pos, minimum in minimums.items():
        if counts.get(pos, 0) < minimum:
            raise RuntimeError(
                f"Insufficient {pos}: found {counts.get(pos, 0)}, need {minimum}"
            )

    eligible = attach_validated_gpp_fields(eligible)
    eligible = add_opponents(eligible)
    eligible = build_gpp_objective_fields(eligible)

    # Stable integer index for CP-SAT and portfolio exposure accounting.
    eligible = eligible.sort_values(
        ["solver_position", "team_solver", "player_solver", "salary_solver"],
        kind="mergesort",
    ).reset_index(drop=True)

    section("OBJECTIVE ATTACHMENT")
    print(f"Contest rows: {len(slate)}")
    print(f"Solver eligible: {len(eligible)}")
    print()
    print(
        eligible.groupby(
            ["solver_position", "gpp_objective_method"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )

    exposure_counts = {i: 0 for i in eligible.index}
    prior_lineups: list[set[int]] = []
    lineup_rows = []
    player_rows = []
    audit_rows = []

    section("GENERATING LINEUPS")

    for lineup_no in range(1, args.lineups + 1):
        solved = solve_one(
            eligible,
            args,
            exposure_counts,
            prior_lineups,
        )

        if solved is None:
            print(
                f"Stopped at lineup {lineup_no}: constraints/exposure caps "
                "made the remaining portfolio infeasible."
            )
            break

        chosen_indices = {i for i, _ in solved}
        prior_lineups.append(chosen_indices)

        for i in chosen_indices:
            exposure_counts[i] += 1

        slot_order = {slot: n for n, slot in enumerate(ROSTER_SLOTS)}
        chosen_sorted = sorted(solved, key=lambda z: slot_order[z[1]])

        rows = []
        for i, slot in chosen_sorted:
            r = eligible.loc[i].copy()
            r["slot"] = slot
            rows.append(r)

        lineup_df = pd.DataFrame(rows)

        salary = int(lineup_df["salary_solver"].sum())
        projection = float(lineup_df["projection_solver"].sum())
        gpp_rank_total = int(lineup_df["gpp_objective_rank_points"].sum())
        stack = audit_stack(lineup_df, args)

        lineup_key = "|".join(
            sorted(
                f"{r.player_solver}::{r.team_solver}"
                for r in lineup_df.itertuples()
            )
        )

        lineup_record = {
            "lineup": lineup_no,
            "salary": salary,
            "projection": projection,
            "gpp_rank_total": gpp_rank_total,
            "lineup_key": lineup_key,
            **stack,
        }

        for slot in ROSTER_SLOTS:
            player = lineup_df.loc[lineup_df["slot"].eq(slot), "player_solver"]
            lineup_record[slot] = player.iloc[0] if len(player) else ""

        lineup_rows.append(lineup_record)

        for r in lineup_df.itertuples():
            player_rows.append({
                "lineup": lineup_no,
                "slot": r.slot,
                "player": r.player_solver,
                "team": r.team_solver,
                "opponent": r.opponent_solver,
                "position": r.solver_position,
                "salary": int(r.salary_solver),
                "projection": float(r.projection_solver),
                "gpp_objective_signal": float(r.gpp_objective_signal),
                "gpp_objective_rank_points": int(r.gpp_objective_rank_points),
                "gpp_objective_method": r.gpp_objective_method,
                "game": r.game_solver,
            })

        roster_counts = lineup_df["slot"].value_counts().to_dict()
        roster_ok = all(roster_counts.get(slot, 0) == 1 for slot in ROSTER_SLOTS)
        salary_ok = salary <= SALARY_CAP and (
            args.salary_floor == 0 or salary >= args.salary_floor
        )
        unique_ok = lineup_df["player_solver"].nunique() == 9
        stack_ok = stack["stack_audit"] == "PASS"

        audit_rows.append({
            "lineup": lineup_no,
            "roster_ok": roster_ok,
            "salary_ok": salary_ok,
            "unique_players_ok": unique_ok,
            "stack_ok": stack_ok,
            "salary": salary,
            "projection": projection,
            "gpp_rank_total": gpp_rank_total,
            "audit": "PASS"
            if roster_ok and salary_ok and unique_ok and stack_ok
            else "FAIL",
        })

        print(
            f"{lineup_no:>3}: salary={salary:>5} "
            f"projection={projection:>8.3f} "
            f"gpp_rank={gpp_rank_total:>7} "
            f"QB={stack['qb']} "
            f"{stack['stack_type']}"
        )

    if not lineup_rows:
        raise RuntimeError("No feasible lineups generated.")

    lineups_df = pd.DataFrame(lineup_rows)
    players_df = pd.DataFrame(player_rows)
    audit_df = pd.DataFrame(audit_rows)

    # Exposure report.
    exposure_rows = []
    generated = len(lineups_df)

    for i, row in eligible.iterrows():
        count = exposure_counts.get(i, 0)
        if count == 0:
            continue

        cap = player_cap(row, args)
        exposure_rows.append({
            "player": row["player_solver"],
            "team": row["team_solver"],
            "position": row["solver_position"],
            "salary": int(row["salary_solver"]),
            "projection": float(row["projection_solver"]),
            "gpp_objective_signal": float(row["gpp_objective_signal"]),
            "gpp_objective_method": row["gpp_objective_method"],
            "lineups": count,
            "exposure_pct": 100.0 * count / generated,
            "cap_lineups": cap,
            "cap_pct_requested": (
                args.max_qb_exposure * 100.0
                if row["solver_position"] == "QB"
                else args.max_dst_exposure * 100.0
                if row["solver_position"] == "DST"
                else args.max_player_exposure * 100.0
            ),
            "cap_ok": count <= cap,
        })

    exposure_df = pd.DataFrame(exposure_rows).sort_values(
        ["lineups", "projection", "player"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    duplicate_lineups = int(lineups_df["lineup_key"].duplicated().sum())

    section("PORTFOLIO AUDIT")
    print(f"Requested lineups: {args.lineups}")
    print(f"Generated lineups: {generated}")
    print(f"Duplicate lineups: {duplicate_lineups}")
    print(f"All lineup audits PASS: {bool(audit_df['audit'].eq('PASS').all())}")
    print(f"All exposure caps PASS: {bool(exposure_df['cap_ok'].all())}")
    print(
        f"Salary range: {int(lineups_df['salary'].min())} - "
        f"{int(lineups_df['salary'].max())}"
    )
    print(
        f"Projection range: {lineups_df['projection'].min():.3f} - "
        f"{lineups_df['projection'].max():.3f}"
    )
    print(
        f"GPP rank range: {int(lineups_df['gpp_rank_total'].min())} - "
        f"{int(lineups_df['gpp_rank_total'].max())}"
    )

    section("TOP EXPOSURES")
    print(
        exposure_df.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    slug = safe_slug(args.slate)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    lineup_path = Path(CSV_DIR) / f"fanduel_gpp_objective_portfolio_{slug}.csv"
    player_path = Path(CSV_DIR) / f"fanduel_gpp_objective_portfolio_players_{slug}.csv"
    exposure_path = Path(CSV_DIR) / f"fanduel_gpp_objective_exposure_{slug}.csv"
    audit_path = Path(CSV_DIR) / f"audit_fanduel_gpp_objective_portfolio_{slug}.csv"

    lineups_df.to_csv(lineup_path, index=False)
    players_df.to_csv(player_path, index=False)
    exposure_df.to_csv(exposure_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    section("EXPORTS")
    print(lineup_path)
    print(player_path)
    print(exposure_path)
    print(audit_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        section("FANDUEL NFL GPP OBJECTIVE SOLVER FAILED")
        print(f"{type(exc).__name__}: {exc}")
        raise
