#!/usr/bin/env python3
"""
nfl_gpp_objective_scale_audit.py

READ-ONLY scale audit for validated NFL GPP objective candidates.

Purpose:
    Determine whether existing GPP/ceiling fields are directly comparable
    to FanDuel projection points before any of them are used as a solver
    objective.

Consumes:
    SQLite: fanduel_solver_ready_pool
    Parquet: nfl_fanduel_player_pool.parquet

This script DOES NOT modify:
    - SQLite
    - parquet files
    - projections
    - frozen lineup solvers

Deterministic merge:
    exact normalized player name + team + position only.

Run:
    cd /home/mwynn/nfl_data_engine
    source venv/bin/activate
    python nfl_gpp_objective_scale_audit.py --slate Main
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, PARQUET_DIR


SOLVER_TABLE = "fanduel_solver_ready_pool"
OFFENSE_PARQUET = Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet"

# Only fields that matter to the historically retained GPP research paths,
# plus supporting volatility/probability fields for scale inspection.
FIELDS = [
    "ridge_projection",
    "fd_std_5",
    "ceiling_prob_10",
    "ceiling_prob_15",
    "ceiling_prob_20",
    "ceiling_prob_25",
    "ceiling_prob_30",
    "gpp_score_10",
    "gpp_score_15",
    "gpp_score_20",
    "gpp_score_25",
    "gpp_score_30",
    "gpp_percentile_10",
    "gpp_percentile_15",
    "gpp_percentile_20",
    "gpp_percentile_25",
    "gpp_percentile_30",
]

PROMOTED_FIELDS_BY_POSITION = {
    "QB": ["ridge_projection"],
    "RB": ["ridge_projection", "gpp_score_20", "ceiling_prob_20", "fd_std_5"],
    "WR": [
        "ridge_projection",
        "gpp_score_15",
        "gpp_score_25",
        "ceiling_prob_15",
        "ceiling_prob_25",
        "fd_std_5",
    ],
    "TE": [
        "ridge_projection",
        "gpp_score_10",
        "gpp_score_15",
        "gpp_score_20",
        "ceiling_prob_10",
        "ceiling_prob_15",
        "ceiling_prob_20",
        "fd_std_5",
    ],
}


def section(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def normalize_slate(value) -> str:
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


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def parse_args():
    p = argparse.ArgumentParser(
        description="Read-only NFL GPP objective scale audit."
    )
    p.add_argument("--slate", default="Main")
    return p.parse_args()


def load_solver() -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (SOLVER_TABLE,),
        ).fetchone()
        if not exists:
            raise RuntimeError(f"Missing SQLite table: {SOLVER_TABLE}")

        return pd.read_sql_query(
            f'SELECT * FROM "{SOLVER_TABLE}"',
            conn,
        )


def choose_slate(df: pd.DataFrame, requested: str) -> pd.DataFrame:
    target = normalize_slate(requested)
    out = df[
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    ].copy()

    if out.empty:
        raise RuntimeError(f"Slate not found: {requested}")
    return out


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def describe_fields(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []

    for col in fields:
        if col not in df.columns:
            rows.append({
                "field": col,
                "n": 0,
                "min": np.nan,
                "p10": np.nan,
                "median": np.nan,
                "mean": np.nan,
                "p90": np.nan,
                "max": np.nan,
                "std": np.nan,
            })
            continue

        s = finite_series(df, col).dropna()

        if s.empty:
            rows.append({
                "field": col,
                "n": 0,
                "min": np.nan,
                "p10": np.nan,
                "median": np.nan,
                "mean": np.nan,
                "p90": np.nan,
                "max": np.nan,
                "std": np.nan,
            })
            continue

        rows.append({
            "field": col,
            "n": len(s),
            "min": float(s.min()),
            "p10": float(s.quantile(0.10)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p90": float(s.quantile(0.90)),
            "max": float(s.max()),
            "std": float(s.std(ddof=0)),
        })

    return pd.DataFrame(rows)


def corr_rows(df: pd.DataFrame, position: str, fields: list[str]) -> pd.DataFrame:
    rows = []
    base = finite_series(df, "projection_solver")

    for col in fields:
        if col not in df.columns:
            continue

        s = finite_series(df, col)
        mask = base.notna() & s.notna()

        corr = np.nan
        if mask.sum() >= 2:
            corr = float(base[mask].corr(s[mask]))

        rows.append({
            "position": position,
            "field": col,
            "paired_rows": int(mask.sum()),
            "corr_vs_projection_solver": corr,
        })

    return pd.DataFrame(rows)


def rank_shift_rows(df: pd.DataFrame, position: str, fields: list[str]) -> pd.DataFrame:
    rows = []
    base = finite_series(df, "projection_solver")

    for col in fields:
        if col not in df.columns:
            continue

        s = finite_series(df, col)
        mask = base.notna() & s.notna()

        if mask.sum() < 2:
            continue

        temp = df.loc[mask, ["player_solver", "team_solver"]].copy()
        temp["projection_solver"] = base[mask].values
        temp[col] = s[mask].values

        temp["projection_rank"] = temp["projection_solver"].rank(
            method="min", ascending=False
        )
        temp["field_rank"] = temp[col].rank(
            method="min", ascending=False
        )
        temp["rank_change"] = (
            temp["projection_rank"] - temp["field_rank"]
        )

        top_up = temp.sort_values(
            ["rank_change", col],
            ascending=[False, False],
        ).head(8)

        top_down = temp.sort_values(
            ["rank_change", col],
            ascending=[True, True],
        ).head(8)

        for direction, subset in [
            ("UP", top_up),
            ("DOWN", top_down),
        ]:
            for _, r in subset.iterrows():
                rows.append({
                    "position": position,
                    "field": col,
                    "direction": direction,
                    "player": r["player_solver"],
                    "team": r["team_solver"],
                    "projection": round(float(r["projection_solver"]), 4),
                    "field_value": round(float(r[col]), 6),
                    "projection_rank": int(r["projection_rank"]),
                    "field_rank": int(r["field_rank"]),
                    "rank_change": int(r["rank_change"]),
                })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    section("NFL GPP OBJECTIVE SCALE AUDIT")
    print(f"Database: {DATABASE_PATH}")
    print(f"Solver table: {SOLVER_TABLE}")
    print(f"Offense parquet: {OFFENSE_PARQUET}")
    print("Mode: READ ONLY")
    print("Frozen solver modifications: NONE")

    solver_all = load_solver()
    slate = choose_slate(solver_all, args.slate)

    eligible = slate[
        pd.to_numeric(
            slate["solver_eligible"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
    ].copy()

    offense_solver = eligible[
        eligible["solver_position"].isin(["QB", "RB", "WR", "TE"])
    ].copy()

    if not OFFENSE_PARQUET.exists():
        raise RuntimeError(
            f"Missing offensive parquet: {OFFENSE_PARQUET}"
        )

    offense = pd.read_parquet(OFFENSE_PARQUET)

    required = ["fd_name", "team", "fd_position"] + FIELDS
    missing = [c for c in required if c not in offense.columns]
    if missing:
        raise RuntimeError(
            f"Offensive parquet missing required audit fields: {missing}"
        )

    offense["_name_key"] = offense["fd_name"].map(normalize_name)
    offense["_team_key"] = offense["team"].map(normalize_team)
    offense["_pos_key"] = (
        offense["fd_position"].astype(str).str.upper().str.strip()
    )

    offense_solver["_name_key"] = offense_solver["player_solver"].map(
        normalize_name
    )
    offense_solver["_team_key"] = offense_solver["team_solver"].map(
        normalize_team
    )
    offense_solver["_pos_key"] = (
        offense_solver["solver_position"].astype(str).str.upper().str.strip()
    )

    dup = offense.duplicated(
        ["_name_key", "_team_key", "_pos_key"],
        keep=False,
    )
    if dup.any():
        raise RuntimeError(
            "Duplicate offensive exact name+team+position keys detected."
        )

    rename_map = {c: f"offense__{c}" for c in FIELDS}
    attach = offense[
        ["_name_key", "_team_key", "_pos_key"] + FIELDS
    ].rename(columns=rename_map)

    merged = offense_solver.merge(
        attach,
        on=["_name_key", "_team_key", "_pos_key"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    section("ATTACHMENT AUDIT")
    print(f"Eligible offensive rows: {len(offense_solver)}")
    print(f"Exact matches: {int(merged['_merge'].eq('both').sum())}")
    print(f"Unmatched: {int(merged['_merge'].ne('both').sum())}")

    if merged["_merge"].ne("both").any():
        print(
            merged.loc[
                merged["_merge"].ne("both"),
                ["player_solver", "team_solver", "solver_position"],
            ].to_string(index=False)
        )
        raise RuntimeError("Exact objective-field attachment is incomplete.")

    for original, renamed in rename_map.items():
        merged[original] = merged[renamed]

    section("POSITION-SPECIFIC SCALE SUMMARY")
    all_corr = []
    all_shifts = []

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = merged[merged["solver_position"].eq(pos)].copy()
        fields = PROMOTED_FIELDS_BY_POSITION[pos]

        print()
        print("-" * 124)
        print(f"{pos} | rows={len(sub)}")
        print("-" * 124)

        display_fields = ["projection_solver"] + fields
        display_fields = list(dict.fromkeys(display_fields))

        stats = describe_fields(sub, display_fields)
        print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        corr = corr_rows(sub, pos, fields)
        if not corr.empty:
            print()
            print("Correlation vs projection_solver:")
            print(
                corr.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}",
                )
            )
            all_corr.append(corr)

        shifts = rank_shift_rows(sub, pos, fields)
        if not shifts.empty:
            all_shifts.append(shifts)

    section("PROMOTED-FIELD RANK SHIFTS")
    if all_shifts:
        shifts = pd.concat(all_shifts, ignore_index=True)

        # Only show GPP-score rank changes; ridge is expected to be close
        # to the production projection and is not the diversification target.
        gpp_shifts = shifts[
            shifts["field"].str.startswith("gpp_score_")
        ].copy()

        if gpp_shifts.empty:
            print("No GPP-score rank shifts available.")
        else:
            print(gpp_shifts.to_string(index=False))
    else:
        print("No rank-shift rows available.")

    section("DECISION GATES")
    print("Use a GPP field directly as a lineup objective ONLY if:")
    print("  1. exact deterministic attachment is complete,")
    print("  2. coverage is complete for its intended position,")
    print("  3. its numeric scale is understood,")
    print("  4. its ranking behavior matches the intended GPP effect, and")
    print("  5. it comes from an already validated/promoted research path.")
    print()
    print(
        "This audit intentionally does not create a new objective or "
        "modify any production/frozen solver."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 124)
        print("NFL GPP OBJECTIVE SCALE AUDIT FAILED")
        print("=" * 124)
        print(f"{type(exc).__name__}: {exc}")
        raise
