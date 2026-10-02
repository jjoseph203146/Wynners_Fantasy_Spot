#!/usr/bin/env python3
"""
audit_gpp_objective_fields.py

READ-ONLY audit for the next NFL GPP objective layer.

Purpose:
    Inspect the stable solver-ready slate pool and the stable offensive
    FanDuel player-pool parquet to determine which existing ceiling/GPP/
    volatility fields can be attached deterministically.

This script DOES NOT:
    - modify SQLite
    - modify projections
    - modify any frozen solver
    - impute values
    - fuzzy match players
    - create new model scores

It only reports field availability, deterministic identity coverage,
and finite-value coverage for potential GPP objective inputs.

Run:
    cd /home/mwynn/nfl_data_engine
    source venv/bin/activate
    python audit_gpp_objective_fields.py --slate Main
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

KEYWORDS = (
    "ceiling",
    "gpp",
    "vol",
    "std",
    "upside",
    "prob",
    "ridge",
    "projection",
)


def section(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


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
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def parse_args():
    p = argparse.ArgumentParser(
        description="Read-only audit of available NFL GPP objective fields."
    )
    p.add_argument("--slate", default="Main")
    return p.parse_args()


def load_solver():
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


def choose_slate(df, requested):
    target = normalize_slate(requested)

    name_col = "slate_name_solver"
    slug_col = "slate_slug_solver"

    if name_col not in df.columns or slug_col not in df.columns:
        raise RuntimeError(
            f"Solver table lacks {name_col}/{slug_col}."
        )

    out = df[
        df[name_col].map(normalize_slate).eq(target)
        | df[slug_col].map(normalize_slate).eq(target)
    ].copy()

    if out.empty:
        raise RuntimeError(f"Slate not found: {requested}")

    return out


def candidate_columns(df):
    return [
        c for c in df.columns
        if any(k in c.lower() for k in KEYWORDS)
    ]


def coverage_table(df, columns):
    rows = []

    for col in columns:
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        finite = numeric.notna() & np.isfinite(numeric)

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "finite_numeric": int(finite.sum()),
            "rows": len(df),
            "finite_pct": round(100.0 * finite.sum() / len(df), 2)
            if len(df) else 0.0,
            "nunique": int(s.nunique(dropna=True)),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    section("NFL GPP OBJECTIVE FIELD AUDIT")
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

    section("SOLVER-READY SLATE")
    print(f"Slate: {args.slate}")
    print(f"Contest rows: {len(slate)}")
    print(f"Solver eligible: {len(eligible)}")
    print()
    print(
        eligible["solver_position"]
        .value_counts()
        .rename_axis("position")
        .reset_index(name="rows")
        .to_string(index=False)
    )

    section("CANDIDATE FIELDS ALREADY IN SOLVER TABLE")
    solver_candidates = candidate_columns(eligible)

    if solver_candidates:
        cov = coverage_table(eligible, solver_candidates)
        print(cov.to_string(index=False))
    else:
        print("No ceiling/GPP/volatility/probability candidate fields found.")

    if not OFFENSE_PARQUET.exists():
        raise RuntimeError(
            f"Stable offensive parquet not found: {OFFENSE_PARQUET}"
        )

    offense = pd.read_parquet(OFFENSE_PARQUET)

    section("OFFENSIVE PARQUET CANDIDATE FIELDS")
    offense_candidates = candidate_columns(offense)

    if offense_candidates:
        print(
            coverage_table(offense, offense_candidates)
            .to_string(index=False)
        )
    else:
        print("No candidate fields found in offensive parquet.")

    required_offense = ["fd_name", "team", "fd_position"]
    missing = [c for c in required_offense if c not in offense.columns]

    if missing:
        raise RuntimeError(
            f"Offensive parquet missing deterministic keys: {missing}"
        )

    off = offense.copy()
    off["_name_key"] = off["fd_name"].map(normalize_name)
    off["_team_key"] = off["team"].map(normalize_team)
    off["_pos_key"] = off["fd_position"].astype(str).str.upper().str.strip()

    sol = eligible[
        eligible["solver_position"].isin(["QB", "RB", "WR", "TE"])
    ].copy()

    sol["_name_key"] = sol["player_solver"].map(normalize_name)
    sol["_team_key"] = sol["team_solver"].map(normalize_team)
    sol["_pos_key"] = sol["solver_position"].astype(str).str.upper().str.strip()

    # Deterministic exact normalized name + team + position only.
    duplicate_keys = off.duplicated(
        ["_name_key", "_team_key", "_pos_key"],
        keep=False,
    )

    section("DETERMINISTIC OFFENSIVE KEY AUDIT")
    print(f"Offensive parquet rows: {len(off)}")
    print(
        "Duplicate exact name+team+position keys: "
        f"{int(duplicate_keys.sum())}"
    )

    if duplicate_keys.any():
        print()
        print("Duplicate-key sample:")
        print(
            off.loc[
                duplicate_keys,
                ["fd_name", "team", "fd_position"]
            ].head(30).to_string(index=False)
        )

    usable_off = off.loc[~duplicate_keys].copy()

    attach_cols = [
        "_name_key",
        "_team_key",
        "_pos_key",
    ] + offense_candidates

    attach_cols = list(dict.fromkeys(attach_cols))

    merged = sol.merge(
        usable_off[attach_cols],
        how="left",
        on=["_name_key", "_team_key", "_pos_key"],
        validate="many_to_one",
        indicator=True,
    )

    section("CURRENT SLATE OFFENSIVE ATTACHMENT COVERAGE")
    print(f"Eligible offensive rows: {len(sol)}")
    print(
        "Exact deterministic matches: "
        f"{int(merged['_merge'].eq('both').sum())}"
    )
    print(
        "Unmatched: "
        f"{int(merged['_merge'].ne('both').sum())}"
    )

    unmatched = merged[merged["_merge"].ne("both")]

    if not unmatched.empty:
        print()
        print("Unmatched rows:")
        print(
            unmatched[
                ["player_solver", "team_solver", "solver_position"]
            ].to_string(index=False)
        )

    section("CURRENT SLATE CANDIDATE FIELD COVERAGE")
    if offense_candidates:
        matched = merged[merged["_merge"].eq("both")].copy()
        cov = coverage_table(matched, offense_candidates)
        print(cov.to_string(index=False))
    else:
        print("No offensive candidate fields to audit.")

    section("FIELD NAMES")
    print("Solver candidate fields:")
    for c in solver_candidates:
        print(f"  {c}")

    print()
    print("Offensive parquet candidate fields:")
    for c in offense_candidates:
        print(f"  {c}")

    section("AUDIT COMPLETE")
    print(
        "No scoring field was promoted. Use this output to choose only "
        "existing, deterministic, sufficiently covered GPP objective inputs."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 122)
        print("GPP OBJECTIVE FIELD AUDIT FAILED")
        print("=" * 122)
        print(f"{type(exc).__name__}: {exc}")
        raise
