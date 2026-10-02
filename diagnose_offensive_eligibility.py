#!/usr/bin/env python3
"""
diagnose_offensive_eligibility.py

READ-ONLY diagnostic for FanDuel NFL offensive projection eligibility.

Purpose
-------
Explain exactly why salary-bearing slate players are not READY in
fanduel_slate_projection_pool, especially MATCHED_NOT_MODEL_ELIGIBLE and
MATCHED_NO_PROJECTION rows.

This script does NOT modify SQLite, parquet, CSV source files, projections,
eligibility, or the solver.

Inputs
------
SQLite:
    fanduel_slate_projection_pool

Stable offensive pool:
    SQLite fanduel_player_pool if present, otherwise:
    data/parquet/nfl_fanduel_player_pool.parquet
    data/csv/nfl_fanduel_player_pool.csv
    data/parquet/fanduel_player_pool.parquet
    data/csv/fanduel_player_pool.csv

Outputs
-------
data/csv/audit_offensive_eligibility_unique_players.csv
data/csv/audit_offensive_eligibility_by_team.csv
data/csv/audit_offensive_eligibility_by_status.csv
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


ATTACH_TABLE = "fanduel_slate_projection_pool"
OFFENSE_TABLE = "fanduel_player_pool"

OFFENSE_FILE_CANDIDATES = [
    Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet",
    Path(CSV_DIR) / "nfl_fanduel_player_pool.csv",
    Path(PARQUET_DIR) / "fanduel_player_pool.parquet",
    Path(CSV_DIR) / "fanduel_player_pool.csv",
]

OUT_UNIQUE = Path(CSV_DIR) / "audit_offensive_eligibility_unique_players.csv"
OUT_TEAM = Path(CSV_DIR) / "audit_offensive_eligibility_by_team.csv"
OUT_STATUS = Path(CSV_DIR) / "audit_offensive_eligibility_by_status.csv"


TEAM_ALIASES = {
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BLT":"BAL",
    "BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "CLV":"CLE","DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB",
    "GNB":"GB","HOU":"HOU","IND":"IND","JAC":"JAX","JAX":"JAX",
    "KC":"KC","KAN":"KC","LV":"LV","LVR":"LV","OAK":"LV",
    "LAC":"LAC","SD":"LAC","SDG":"LAC","LA":"LA","LAR":"LA",
    "STL":"LA","MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE",
    "NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI",
    "PIT":"PIT","SF":"SF","SFO":"SF","SEA":"SEA","TB":"TB",
    "TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS",
}


def section(title):
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def norm_text(value):
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_team(value):
    if pd.isna(value):
        return None
    s = re.sub(r"[^A-Z]", "", str(value).upper())
    return TEAM_ALIASES.get(s)


def table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def load_offense(conn):
    if table_exists(conn, OFFENSE_TABLE):
        return pd.read_sql_query(f'SELECT * FROM "{OFFENSE_TABLE}"', conn), f"sqlite:{OFFENSE_TABLE}"

    for path in OFFENSE_FILE_CANDIDATES:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if not df.empty:
            return df, str(path)

    raise RuntimeError(
        "Stable offensive pool not found. Searched:\n"
        + "\n".join(f"  - {p}" for p in OFFENSE_FILE_CANDIDATES)
    )


def first_col(df, candidates, required=False):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(
            f"Required column not found. Tried {candidates}. "
            f"Available: {list(df.columns)}"
        )
    return None


def boolish(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "ready", "eligible"})
        .astype(int)
    )


def main():
    section("OFFENSIVE ELIGIBILITY DIAGNOSTIC — READ ONLY")
    print(f"Database: {DATABASE_PATH}")
    print("No source files or database tables will be modified.")

    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, ATTACH_TABLE):
            raise RuntimeError(
                f"Required table not found: {ATTACH_TABLE}. "
                "Run fanduel_slate_projection_attach_v4.py first."
            )

        attached = pd.read_sql_query(
            f'SELECT * FROM "{ATTACH_TABLE}"',
            conn,
        )
        offense, offense_source = load_offense(conn)

    print(f"Attachment rows: {len(attached)}")
    print(f"Stable offensive source: {offense_source}")
    print(f"Stable offensive rows: {len(offense)}")

    # Resolve attachment fields.
    a_player = first_col(attached, ["player", "fd_name", "player_name"], True)
    a_team = first_col(attached, ["team_internal", "team"], True)
    a_salary = first_col(attached, ["salary"], True)
    a_dst = first_col(attached, ["is_dst"], True)
    a_status = first_col(attached, ["projection_status"], True)
    a_slate = first_col(attached, ["slate_name"], True)

    # Resolve stable offensive fields.
    o_player = first_col(offense, ["fd_name", "player_display_name", "player"], True)
    o_team = first_col(offense, ["team", "model_team", "team_internal"], True)
    o_pos = first_col(offense, ["fd_position", "model_position", "position"], True)
    o_salary = first_col(offense, ["salary", "fd_salary"])
    o_identity = first_col(offense, ["identity_key", "player_id", "gsis_id"])
    o_eligible = first_col(offense, ["optimizer_eligible"], True)
    o_pool_status = first_col(offense, ["player_pool_status"])
    o_prod_status = first_col(offense, ["production_status"])
    o_ridge = first_col(offense, ["ridge_projection"])
    o_game_id = first_col(offense, ["game_id"])
    o_week = first_col(offense, ["week"])
    o_game_date = first_col(offense, ["game_date"])
    o_game_time = first_col(offense, ["game_time"])
    o_model_team = first_col(offense, ["model_team"])
    o_model_pos = first_col(offense, ["model_position"])
    o_history = first_col(offense, ["player_history_games"])

    section("RESOLVED STABLE OFFENSIVE FIELDS")
    fields = {
        "player": o_player,
        "team": o_team,
        "position": o_pos,
        "salary": o_salary,
        "identity": o_identity,
        "optimizer_eligible": o_eligible,
        "player_pool_status": o_pool_status,
        "production_status": o_prod_status,
        "ridge_projection": o_ridge,
        "game_id": o_game_id,
        "week": o_week,
        "game_date": o_game_date,
        "game_time": o_game_time,
        "model_team": o_model_team,
        "model_position": o_model_pos,
        "player_history_games": o_history,
    }
    for k, v in fields.items():
        print(f"{k:<22} -> {v}")

    # Build deterministic normalized stable lookup.
    off = offense.copy()
    off["_player_norm"] = off[o_player].map(norm_text)
    off["_team_norm"] = off[o_team].map(norm_team)
    off["_eligible_norm"] = boolish(off[o_eligible])

    if o_ridge:
        off["_ridge_norm"] = pd.to_numeric(off[o_ridge], errors="coerce")
    else:
        off["_ridge_norm"] = np.nan

    key_counts = (
        off.groupby(["_player_norm", "_team_norm"], dropna=False)
        .size()
        .rename("_stable_match_count")
        .reset_index()
    )
    off = off.merge(key_counts, on=["_player_norm", "_team_norm"], how="left")

    # Only offensive gaps from attachment table.
    gaps = attached[
        (pd.to_numeric(attached[a_dst], errors="coerce").fillna(0).eq(0))
        & attached[a_status].ne("READY")
    ].copy()

    gaps["_player_norm"] = gaps[a_player].map(norm_text)
    gaps["_team_norm"] = gaps[a_team].map(norm_team)

    unique_gap_keys = (
        gaps[
            [a_player, "_player_norm", "_team_norm", a_status]
        ]
        .drop_duplicates(["_player_norm", "_team_norm", a_status])
        .copy()
    )

    records = []

    for gap in unique_gap_keys.to_dict("records"):
        player_norm = gap["_player_norm"]
        team_norm = gap["_team_norm"]

        matches = off[
            off["_player_norm"].eq(player_norm)
            & off["_team_norm"].eq(team_norm)
        ].copy()

        base = {
            "player": gap[a_player],
            "team": team_norm,
            "attachment_status": gap[a_status],
            "stable_match_rows": len(matches),
        }

        if matches.empty:
            base["diagnosis"] = "NO_STABLE_OFFENSIVE_MATCH"
            records.append(base)
            continue

        # Attachment v4 only accepted unique exact name+team matches.
        # Preserve every relevant upstream state in the diagnostic.
        for _, m in matches.iterrows():
            rec = dict(base)
            rec["diagnosis"] = (
                "UPSTREAM_NOT_ELIGIBLE"
                if int(m["_eligible_norm"]) != 1
                else (
                    "UPSTREAM_ELIGIBLE_BUT_NO_RIDGE"
                    if pd.isna(m["_ridge_norm"])
                    else "UPSTREAM_READY"
                )
            )
            rec["stable_player"] = m.get(o_player)
            rec["stable_position"] = m.get(o_pos)
            rec["stable_salary"] = m.get(o_salary) if o_salary else np.nan
            rec["identity_key"] = m.get(o_identity) if o_identity else ""
            rec["optimizer_eligible"] = int(m["_eligible_norm"])
            rec["ridge_projection"] = m["_ridge_norm"]
            rec["player_pool_status"] = m.get(o_pool_status) if o_pool_status else ""
            rec["production_status"] = m.get(o_prod_status) if o_prod_status else ""
            rec["game_id"] = m.get(o_game_id) if o_game_id else ""
            rec["week"] = m.get(o_week) if o_week else np.nan
            rec["game_date"] = m.get(o_game_date) if o_game_date else ""
            rec["game_time"] = m.get(o_game_time) if o_game_time else ""
            rec["model_team"] = m.get(o_model_team) if o_model_team else ""
            rec["model_position"] = m.get(o_model_pos) if o_model_pos else ""
            rec["player_history_games"] = m.get(o_history) if o_history else np.nan
            records.append(rec)

    detail = pd.DataFrame(records)

    # Add slate membership and salary variants for each unique gap player.
    memberships = (
        gaps.groupby(["_player_norm", "_team_norm"], dropna=False)
        .agg(
            slate_count=(a_slate, "nunique"),
            slates=(a_slate, lambda s: ";".join(sorted(set(map(str, s))))),
            salary_variants=(a_salary, lambda s: ";".join(
                map(str, sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
            )),
        )
        .reset_index()
    )

    detail["_player_norm"] = detail["player"].map(norm_text)
    detail["_team_norm"] = detail["team"].map(norm_team)
    detail = detail.merge(
        memberships,
        on=["_player_norm", "_team_norm"],
        how="left",
    )

    # Team/status summaries are based on unique player-team diagnoses.
    summary_base = (
        detail.sort_values(
            ["player", "team", "diagnosis"],
            na_position="last",
        )
        .drop_duplicates(
            ["_player_norm", "_team_norm", "attachment_status", "diagnosis"]
        )
    )

    by_team = (
        summary_base.groupby(
            ["team", "attachment_status", "diagnosis"],
            dropna=False,
        )
        .size()
        .rename("unique_players")
        .reset_index()
        .sort_values(
            ["unique_players", "team"],
            ascending=[False, True],
        )
    )

    by_status = (
        summary_base.groupby(
            [
                "attachment_status",
                "diagnosis",
                "production_status",
                "player_pool_status",
                "optimizer_eligible",
            ],
            dropna=False,
        )
        .size()
        .rename("unique_players")
        .reset_index()
        .sort_values("unique_players", ascending=False)
    )

    section("GAP SUMMARY")
    print(f"Offensive gap slate-rows:       {len(gaps)}")
    print(
        "Unique gap player/team pairs:   "
        f"{gaps[['_player_norm','_team_norm']].drop_duplicates().shape[0]}"
    )
    print(
        "Matched-not-model-eligible rows: "
        f"{gaps[a_status].eq('MATCHED_NOT_MODEL_ELIGIBLE').sum()}"
    )
    print(
        "Matched-no-projection rows:       "
        f"{gaps[a_status].eq('MATCHED_NO_PROJECTION').sum()}"
    )

    section("UPSTREAM DIAGNOSIS BY TEAM")
    print(by_team.to_string(index=False))

    section("UPSTREAM STATUS COMBINATIONS")
    display_status = [
        c for c in [
            "attachment_status",
            "diagnosis",
            "production_status",
            "player_pool_status",
            "optimizer_eligible",
            "unique_players",
        ]
        if c in by_status.columns
    ]
    print(by_status[display_status].to_string(index=False))

    section("KEY NIGHT-SLATE / HIGH-VALUE GAP PLAYERS")
    focus_teams = {"DAL", "DEN", "KC", "NYG"}
    focus = summary_base[
        summary_base["team"].isin(focus_teams)
    ].copy()

    display = [
        c for c in [
            "player", "team", "stable_position",
            "attachment_status", "diagnosis",
            "optimizer_eligible", "ridge_projection",
            "production_status", "player_pool_status",
            "game_id", "week", "game_date",
            "model_team", "model_position",
            "player_history_games", "slates",
        ]
        if c in focus.columns
    ]

    if focus.empty:
        print("No DAL/DEN/KC/NYG gap players found.")
    else:
        print(
            focus[display]
            .sort_values(["team", "player"])
            .to_string(index=False)
        )

    OUT_UNIQUE.parent.mkdir(parents=True, exist_ok=True)
    detail.drop(columns=["_player_norm", "_team_norm"], errors="ignore").to_csv(
        OUT_UNIQUE,
        index=False,
    )
    by_team.to_csv(OUT_TEAM, index=False)
    by_status.to_csv(OUT_STATUS, index=False)

    section("EXPORTS")
    print(f"Unique player diagnostic: {OUT_UNIQUE}")
    print(f"Team summary:             {OUT_TEAM}")
    print(f"Status summary:           {OUT_STATUS}")

    section("DIAGNOSTIC COMPLETE")
    print("READ ONLY: no eligibility or projection values were changed.")
    print(
        "Use the production_status / player_pool_status / game_id / week fields "
        "above to identify the exact upstream gate before changing any stable file."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 112)
        print("OFFENSIVE ELIGIBILITY DIAGNOSTIC FAILED")
        print("=" * 112)
        print(f"{type(exc).__name__}: {exc}")
        raise
