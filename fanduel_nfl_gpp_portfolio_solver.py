#!/usr/bin/env python3
"""
fanduel_nfl_gpp_portfolio_solver.py

Deterministic FanDuel NFL GPP portfolio solver.

NEW layer above the frozen:
    - fanduel_nfl_lineup_solver.py
    - fanduel_nfl_gpp_solver.py

Consumes ONLY:
    SQLite table: fanduel_solver_ready_pool

Adds portfolio-level hard exposure controls while preserving:
    - $60,000 FanDuel salary cap
    - QB/RB/RB/WR/WR/WR/TE/FLEX/DST roster
    - exact slot assignment
    - QB + WR/TE stacking
    - optional opponent RB/WR/TE bring-back
    - deterministic CP-SAT
    - no fuzzy matching
    - no salary inference
    - no randomness
    - no projection changes

Exposure controls:
    --max-player-exposure   default 0.60
    --max-qb-exposure       default 0.40
    --max-dst-exposure      default 0.40

For N requested lineups, a percentage cap is converted to an integer hard cap
using floor(N * exposure). Example: 20 lineups at 0.40 => maximum 8 lineups.

If the requested portfolio becomes infeasible before N lineups are generated,
the script stops and reports the shortfall rather than silently violating caps.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from config import DATABASE_PATH, CSV_DIR


SOURCE_TABLE = "fanduel_solver_ready_pool"
SALARY_CAP = 60000

ROSTER_SLOTS = [
    "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"
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


def section(title):
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def normalize_slate(value):
    return (
        str(value).strip().lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_team(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z]", "", str(value).upper())


def parse_game_teams(game_value):
    if pd.isna(game_value):
        return None, None

    raw = str(game_value).upper().strip()
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


def exposure_cap(n_lineups, exposure):
    if not (0 < exposure <= 1):
        raise RuntimeError("Exposure values must be > 0 and <= 1.")
    return max(1, int(math.floor(n_lineups * exposure + 1e-9)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Deterministic FanDuel NFL GPP portfolio solver."
    )
    p.add_argument("--slate", required=False)
    p.add_argument("--lineups", type=int, default=20)
    p.add_argument("--qb-stack", type=int, choices=[1, 2], default=2)
    p.add_argument("--bring-back", type=int, choices=[0, 1], default=1)
    p.add_argument("--salary-floor", type=int, default=0)
    p.add_argument("--max-overlap", type=int, default=7)
    p.add_argument("--max-team", type=int, default=0)
    p.add_argument("--max-player-exposure", type=float, default=0.60)
    p.add_argument("--max-qb-exposure", type=float, default=0.40)
    p.add_argument("--max-dst-exposure", type=float, default=0.40)
    return p.parse_args()


def load_pool():
    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, SOURCE_TABLE):
            raise RuntimeError(f"Required table not found: {SOURCE_TABLE}")
        df = pd.read_sql_query(f'SELECT * FROM "{SOURCE_TABLE}"', conn)

    required = [
        "slate_name_solver", "slate_slug_solver", "player_solver",
        "team_solver", "salary_solver", "game_solver",
        "projection_solver", "solver_position", "solver_eligible",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["salary_solver"] = pd.to_numeric(df["salary_solver"], errors="coerce")
    df["projection_solver"] = pd.to_numeric(df["projection_solver"], errors="coerce")
    df["solver_eligible"] = pd.to_numeric(
        df["solver_eligible"], errors="coerce"
    ).fillna(0).astype(int)
    return df


def list_slates(df):
    section("AVAILABLE SLATES")
    print(
        df[["slate_name_solver", "slate_slug_solver"]]
        .drop_duplicates()
        .sort_values("slate_name_solver")
        .to_string(index=False)
    )


def select_slate(df, requested):
    target = normalize_slate(requested)
    rows = df[
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    ].copy()

    if rows.empty:
        raise RuntimeError(f"Slate not found: {requested}")

    pool = rows[rows["solver_eligible"].eq(1)].copy()
    pool = pool[
        pool["salary_solver"].notna()
        & pool["salary_solver"].gt(0)
        & pool["projection_solver"].notna()
        & np.isfinite(pool["projection_solver"])
        & pool["solver_position"].isin(VALID_POSITIONS)
    ].copy().reset_index(drop=True)

    pool["player_index"] = pool.index.astype(int)
    pool["team_norm"] = pool["team_solver"].map(normalize_team)
    pool["opponent_team"] = pool.apply(
        lambda r: opponent_from_game(r["team_solver"], r["game_solver"]), axis=1
    )
    return rows, pool


def validate_pool(pool):
    if pool.empty:
        raise RuntimeError("No solver-eligible rows.")

    dup = pool.duplicated(
        ["player_solver", "team_solver", "salary_solver"], keep=False
    )
    if dup.any():
        raise RuntimeError(
            "Duplicate players detected:\n"
            + pool.loc[dup, [
                "player_solver", "team_solver", "salary_solver"
            ]].to_string(index=False)
        )

    bad_qbs = pool[
        pool["solver_position"].eq("QB") & pool["opponent_team"].isna()
    ]
    if not bad_qbs.empty:
        raise RuntimeError(
            "QB opponent resolution failed:\n"
            + bad_qbs[["player_solver", "team_solver", "game_solver"]]
            .to_string(index=False)
        )


def player_cap_for_row(row, total_lineups, args):
    general = exposure_cap(total_lineups, args.max_player_exposure)

    if row["solver_position"] == "QB":
        return min(general, exposure_cap(total_lineups, args.max_qb_exposure))
    if row["solver_position"] == "DST":
        return min(general, exposure_cap(total_lineups, args.max_dst_exposure))
    return general


def build_and_solve(pool, previous, exposure_counts, total_lineups, args):
    model = cp_model.CpModel()
    x = {}

    for i, row in pool.iterrows():
        pos = row["solver_position"]
        for slot in ROSTER_SLOTS:
            if pos in SLOT_ELIGIBILITY[slot]:
                x[(i, slot)] = model.NewBoolVar(f"x_{i}_{slot}")

    for slot in ROSTER_SLOTS:
        vars_ = [v for (i, s), v in x.items() if s == slot]
        if not vars_:
            return None
        model.Add(sum(vars_) == 1)

    selected = {}
    for i in pool.index:
        vars_ = [v for (j, s), v in x.items() if j == i]
        if vars_:
            y = model.NewBoolVar(f"selected_{i}")
            model.Add(sum(vars_) == y)
            selected[i] = y

    model.Add(sum(selected.values()) == 9)

    salary = sum(
        int(round(float(pool.at[i, "salary_solver"]))) * y
        for i, y in selected.items()
    )
    model.Add(salary <= SALARY_CAP)
    if args.salary_floor > 0:
        model.Add(salary >= args.salary_floor)

    if args.max_team > 0:
        for team, group in pool.groupby("team_norm"):
            idx = [i for i in group.index if i in selected]
            if idx:
                model.Add(sum(selected[i] for i in idx) <= args.max_team)

    # Hard portfolio exposure gates. Once a player has reached the integer
    # portfolio cap, that player is prohibited from subsequent lineups.
    for i, y in selected.items():
        row = pool.loc[i]
        cap = player_cap_for_row(row, total_lineups, args)
        used = exposure_counts.get(i, 0)
        if used >= cap:
            model.Add(y == 0)

    # QB correlation.
    for qb_idx, qb in pool[pool["solver_position"].eq("QB")].iterrows():
        if qb_idx not in selected:
            continue

        teammate_idx = [
            i for i, row in pool.iterrows()
            if (
                i in selected
                and row["team_norm"] == qb["team_norm"]
                and row["solver_position"] in PASS_CATCHER_POSITIONS
            )
        ]

        if len(teammate_idx) >= args.qb_stack:
            model.Add(
                sum(selected[i] for i in teammate_idx)
                >= args.qb_stack * selected[qb_idx]
            )
        else:
            model.Add(selected[qb_idx] == 0)

        if args.bring_back == 1:
            opp_idx = [
                i for i, row in pool.iterrows()
                if (
                    i in selected
                    and row["team_norm"] == qb["opponent_team"]
                    and row["solver_position"] in BRING_BACK_POSITIONS
                )
            ]
            if opp_idx:
                model.Add(
                    sum(selected[i] for i in opp_idx) >= selected[qb_idx]
                )
            else:
                model.Add(selected[qb_idx] == 0)

    # Pairwise overlap with all prior portfolio lineups.
    for lineup in previous:
        overlap = [
            selected[i] for i in lineup["player_indices"] if i in selected
        ]
        if overlap:
            model.Add(sum(overlap) <= args.max_overlap)

    scale = 1000
    objective = sum(
        int(round(float(pool.at[i, "projection_solver"]) * scale)) * y
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

    slots = []
    indices = []

    for slot in ROSTER_SLOTS:
        chosen = None
        for (i, s), var in x.items():
            if s == slot and solver.Value(var) == 1:
                chosen = i
                break
        if chosen is None:
            return None

        row = pool.loc[chosen]
        indices.append(chosen)
        slots.append({
            "slot": slot,
            "player": row["player_solver"],
            "position": row["solver_position"],
            "team": row["team_solver"],
            "team_norm": row["team_norm"],
            "opponent_team": row["opponent_team"],
            "game": row["game_solver"],
            "salary": int(round(float(row["salary_solver"]))),
            "projection": float(row["projection_solver"]),
            "player_index": int(chosen),
        })

    sdf = pd.DataFrame(slots)
    return {
        "player_indices": tuple(sorted(indices)),
        "slots": sdf,
        "total_salary": int(sdf["salary"].sum()),
        "total_projection": float(sdf["projection"].sum()),
    }


def analyze_stack(lineup, args):
    df = lineup["slots"]
    qb = df[df["slot"].eq("QB")].iloc[0]
    qbt = normalize_team(qb["team"])
    opp = qb["opponent_team"]

    pc = df[
        df["team_norm"].eq(qbt)
        & df["position"].isin(PASS_CATCHER_POSITIONS)
    ]
    br = df[
        df["team_norm"].eq(opp)
        & df["position"].isin(BRING_BACK_POSITIONS)
    ]

    pc_names = sorted(pc["player"].astype(str).tolist())
    br_names = sorted(br["player"].astype(str).tolist())

    passed = (
        len(pc_names) >= args.qb_stack
        and (args.bring_back == 0 or len(br_names) >= 1)
    )

    return {
        "qb": qb["player"],
        "qb_team": qb["team"],
        "qb_opponent": opp,
        "stack_pass_catchers": " | ".join(pc_names),
        "stack_pass_catcher_count": len(pc_names),
        "bring_back_players": " | ".join(br_names),
        "bring_back_count": len(br_names),
        "stack_type": f"QB+{len(pc_names)}"
        + (f"+{len(br_names)}BR" if br_names else ""),
        "stack_audit": "PASS" if passed else "FAIL",
    }


def generate_portfolio(pool, args):
    previous = []
    counts = Counter()

    for _ in range(args.lineups):
        lineup = build_and_solve(
            pool=pool,
            previous=previous,
            exposure_counts=counts,
            total_lineups=args.lineups,
            args=args,
        )
        if lineup is None:
            break

        previous.append(lineup)
        for i in lineup["player_indices"]:
            counts[i] += 1

    return previous, counts


def build_exports(slate_name, slate_slug, pool, lineups, counts, args):
    summary_rows = []
    long_rows = []

    for rank, lineup in enumerate(lineups, 1):
        stack = analyze_stack(lineup, args)
        errors = []

        if len(lineup["slots"]) != 9:
            errors.append("row_count")
        if lineup["slots"]["player_index"].nunique() != 9:
            errors.append("duplicate_player")
        if lineup["total_salary"] > SALARY_CAP:
            errors.append("salary_cap")
        if stack["stack_audit"] != "PASS":
            errors.append("stack")

        wide = {
            "lineup_rank": rank,
            "slate_name": slate_name,
            "slate_slug": slate_slug,
            "total_salary": lineup["total_salary"],
            "salary_left": SALARY_CAP - lineup["total_salary"],
            "total_projection": round(lineup["total_projection"], 6),
            **stack,
            "audit_status": "PASS" if not errors else "FAIL",
            "audit_errors": ";".join(errors),
        }

        for _, row in lineup["slots"].iterrows():
            wide[row["slot"]] = row["player"]
            long_rows.append({
                "lineup_rank": rank,
                "slot": row["slot"],
                "player": row["player"],
                "position": row["position"],
                "team": row["team"],
                "game": row["game"],
                "salary": row["salary"],
                "projection": row["projection"],
                "lineup_total_salary": lineup["total_salary"],
                "lineup_total_projection": lineup["total_projection"],
                "lineup_qb": stack["qb"],
                "stack_type": stack["stack_type"],
            })

        summary_rows.append(wide)

    summary = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)

    exposure_rows = []
    for i, row in pool.iterrows():
        used = counts.get(i, 0)
        if used == 0:
            continue

        cap = player_cap_for_row(row, args.lineups, args)
        exposure_rows.append({
            "player": row["player_solver"],
            "position": row["solver_position"],
            "team": row["team_solver"],
            "lineups": used,
            "exposure_pct": round(100.0 * used / len(lineups), 2)
            if lineups else 0.0,
            "hard_cap_lineups": cap,
            "requested_portfolio_pct_cap": round(
                100.0 * cap / args.lineups, 2
            ),
            "cap_status": "PASS" if used <= cap else "FAIL",
        })

    exposure = pd.DataFrame(exposure_rows)
    if not exposure.empty:
        exposure = exposure.sort_values(
            ["lineups", "position", "player"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    slug = normalize_slate(slate_slug)
    summary_path = Path(CSV_DIR) / f"fanduel_gpp_portfolio_{slug}.csv"
    long_path = Path(CSV_DIR) / f"fanduel_gpp_portfolio_players_{slug}.csv"
    exposure_path = Path(CSV_DIR) / f"fanduel_gpp_exposure_{slug}.csv"
    audit_path = Path(CSV_DIR) / f"audit_fanduel_gpp_portfolio_{slug}.csv"

    summary.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)
    exposure.to_csv(exposure_path, index=False)

    complete = len(lineups) == args.lineups
    caps_pass = exposure.empty or exposure["cap_status"].eq("PASS").all()
    lineup_pass = (
        len(summary) > 0 and summary["audit_status"].eq("PASS").all()
    )

    audit = pd.DataFrame([{
        "slate_name": slate_name,
        "slate_slug": slate_slug,
        "requested_lineups": args.lineups,
        "generated_lineups": len(lineups),
        "portfolio_complete": complete,
        "all_lineup_audits_pass": lineup_pass,
        "all_exposure_caps_pass": bool(caps_pass),
        "max_player_exposure": args.max_player_exposure,
        "max_qb_exposure": args.max_qb_exposure,
        "max_dst_exposure": args.max_dst_exposure,
        "player_cap_lineups": exposure_cap(
            args.lineups, args.max_player_exposure
        ),
        "qb_cap_lineups": min(
            exposure_cap(args.lineups, args.max_player_exposure),
            exposure_cap(args.lineups, args.max_qb_exposure),
        ),
        "dst_cap_lineups": min(
            exposure_cap(args.lineups, args.max_player_exposure),
            exposure_cap(args.lineups, args.max_dst_exposure),
        ),
        "qb_stack": args.qb_stack,
        "bring_back": args.bring_back,
        "max_overlap": args.max_overlap,
        "salary_floor": args.salary_floor,
    }])
    audit.to_csv(audit_path, index=False)

    return (
        summary, long_df, exposure, audit,
        summary_path, long_path, exposure_path, audit_path
    )


def main():
    args = parse_args()

    if args.lineups < 1:
        raise RuntimeError("--lineups must be >= 1.")
    if not (0 < args.max_player_exposure <= 1):
        raise RuntimeError("--max-player-exposure must be > 0 and <= 1.")
    if not (0 < args.max_qb_exposure <= 1):
        raise RuntimeError("--max-qb-exposure must be > 0 and <= 1.")
    if not (0 < args.max_dst_exposure <= 1):
        raise RuntimeError("--max-dst-exposure must be > 0 and <= 1.")
    if args.max_overlap < 0 or args.max_overlap > 8:
        raise RuntimeError("--max-overlap must be between 0 and 8.")
    if args.salary_floor < 0 or args.salary_floor > SALARY_CAP:
        raise RuntimeError("--salary-floor outside valid range.")

    section("FANDUEL NFL GPP PORTFOLIO SOLVER")
    print(f"Database: {DATABASE_PATH}")
    print(f"Player pool: {SOURCE_TABLE}")
    print(f"Requested lineups: {args.lineups}")
    print(f"QB stack: {args.qb_stack} WR/TE")
    print(f"Bring-back required: {'YES' if args.bring_back else 'NO'}")
    print(f"Max player exposure: {args.max_player_exposure:.0%}")
    print(f"Max QB exposure: {args.max_qb_exposure:.0%}")
    print(f"Max DST exposure: {args.max_dst_exposure:.0%}")
    print(f"General hard cap: {exposure_cap(args.lineups, args.max_player_exposure)} lineups")
    print(
        "QB hard cap: "
        f"{min(exposure_cap(args.lineups, args.max_player_exposure), exposure_cap(args.lineups, args.max_qb_exposure))} lineups"
    )
    print(
        "DST hard cap: "
        f"{min(exposure_cap(args.lineups, args.max_player_exposure), exposure_cap(args.lineups, args.max_dst_exposure))} lineups"
    )
    print("Deterministic CP-SAT: enabled")
    print("Randomness: disabled")

    all_rows = load_pool()

    if not args.slate:
        list_slates(all_rows)
        return

    source, pool = select_slate(all_rows, args.slate)
    validate_pool(pool)

    slate_name = str(source["slate_name_solver"].iloc[0])
    slate_slug = str(source["slate_slug_solver"].iloc[0])

    section("SELECTED SLATE")
    print(f"Slate: {slate_name} ({slate_slug})")
    print(f"Contest rows: {len(source)}")
    print(f"Solver eligible rows: {len(pool)}")
    print()
    print(
        pool["solver_position"]
        .value_counts()
        .rename_axis("position")
        .reset_index(name="players")
        .to_string(index=False)
    )

    lineups, counts = generate_portfolio(pool, args)

    if not lineups:
        raise RuntimeError("No feasible portfolio lineup could be generated.")

    (
        summary, long_df, exposure, audit,
        summary_path, long_path, exposure_path, audit_path
    ) = build_exports(
        slate_name, slate_slug, pool, lineups, counts, args
    )

    section("PORTFOLIO LINEUPS")
    cols = [
        "lineup_rank", "total_salary", "salary_left", "total_projection",
        "QB", "stack_pass_catchers", "bring_back_players", "stack_type",
        "DST", "audit_status",
    ]
    print(summary[cols].to_string(index=False))

    section("TOP PLAYER EXPOSURES")
    if exposure.empty:
        print("No exposure rows.")
    else:
        print(exposure.head(30).to_string(index=False))

    section("QB EXPOSURE")
    qbexp = exposure[exposure["position"].eq("QB")] if not exposure.empty else exposure
    print(qbexp.to_string(index=False) if not qbexp.empty else "No QB exposure rows.")

    section("DST EXPOSURE")
    dstexp = exposure[exposure["position"].eq("DST")] if not exposure.empty else exposure
    print(dstexp.to_string(index=False) if not dstexp.empty else "No DST exposure rows.")

    section("PORTFOLIO AUDIT")
    print(f"Requested lineups:          {args.lineups}")
    print(f"Generated lineups:          {len(lineups)}")
    print(f"Portfolio complete:         {len(lineups) == args.lineups}")
    print(
        "All lineup audits PASS:    "
        f"{summary['audit_status'].eq('PASS').all()}"
    )
    print(
        "All exposure caps PASS:    "
        f"{exposure.empty or exposure['cap_status'].eq('PASS').all()}"
    )
    print(
        "Duplicate lineups:         "
        f"{int(summary[ROSTER_SLOTS].duplicated().sum())}"
    )
    print(
        "Projection range:          "
        f"{summary['total_projection'].min():.3f} - "
        f"{summary['total_projection'].max():.3f}"
    )

    section("EXPORTS")
    print(f"Portfolio: {summary_path}")
    print(f"Players:   {long_path}")
    print(f"Exposure:  {exposure_path}")
    print(f"Audit:     {audit_path}")

    section("GPP PORTFOLIO LAYER COMPLETE")
    if len(lineups) == args.lineups:
        print(
            "Full deterministic portfolio generated within hard exposure, "
            "correlation, salary, roster, and overlap constraints."
        )
    else:
        print(
            "Portfolio stopped before the requested lineup count because the "
            "remaining hard constraints were infeasible. Exposure caps were "
            "not violated."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 124)
        print("FANDUEL NFL GPP PORTFOLIO SOLVER FAILED")
        print("=" * 124)
        print(f"{type(exc).__name__}: {exc}")
        raise
