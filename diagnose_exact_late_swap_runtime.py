#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sqlite3
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data" / "nfl.db"
TABLE = "fanduel_solver_ready_pool"

import fanduel_nfl_ui_solver as prod
import fanduel_nfl_late_swap_solver_stage2_v1 as late

TEST_LINEUP = {
    "QB": ("Bo Nix", "DEN", "SWAPPABLE"),
    "RB1": ("Kenneth Walker III", "KC", "SWAPPABLE"),
    "RB2": ("J.K. Dobbins", "DEN", "SWAPPABLE"),
    "WR1": ("Malik Nabers", "NYG", "LOCKED"),
    "WR2": ("George Pickens", "DAL", "LOCKED"),
    "WR3": ("Jaylen Waddle", "DEN", "SWAPPABLE"),
    "TE": ("Isaiah Likely", "NYG", "LOCKED"),
    "FLEX": ("Tyrone Tracy Jr.", "NYG", "LOCKED"),
    "DST": ("Denver D/ST", "DEN", "SWAPPABLE"),
}

def norm(v):
    return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")

def simple_name(v):
    return (
        str(v).lower().replace(".", "").replace("'", "")
        .replace("-", " ").replace("  ", " ").strip()
    )

def find_one(df, player, team):
    names = df["player_solver"].astype(str).str.strip()
    teams = df["team_solver"].astype(str).str.strip().str.upper()
    exact = df.loc[names.eq(player) & teams.eq(team)].copy()
    if exact.empty:
        exact = df.loc[
            df["player_solver"].map(simple_name).eq(simple_name(player))
            & teams.eq(team)
        ].copy()
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected exactly one row for {player}/{team}; found {len(exact)}"
        )
    return exact.iloc[0]

def main():
    print("=" * 100)
    print("READ-ONLY EXACT LATE SWAP RUNTIME DIAGNOSTIC")
    print("=" * 100)
    print("Module:", Path(late.__file__).resolve())
    print("_prepare_pool signature:", inspect.signature(late._prepare_pool))
    print("optimize_late_swap signature:", inspect.signature(late.optimize_late_swap))
    print()

    with sqlite3.connect(DB) as conn:
        all_rows = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', conn)

    mask = all_rows["slate_slug_solver"].map(norm).eq("snf_mnf")
    if "slate_name_solver" in all_rows.columns:
        mask = mask | all_rows["slate_name_solver"].map(norm).eq("snf_mnf")
    pool = all_rows.loc[mask].copy()

    elig = pd.to_numeric(pool["solver_eligible"], errors="coerce").fillna(0).astype(int).eq(1)
    print(f"Slate rows: {len(pool)} | Eligible: {int(elig.sum())} | Blocked: {int((~elig).sum())}")
    print()

    current_slot_keys = {}
    slot_status = {}
    salary_total = 0
    team_counts = {}

    print("CURRENT TEST LINEUP")
    print("-" * 100)
    for slot, (player, team, status) in TEST_LINEUP.items():
        row = find_one(pool, player, team)
        key = str(prod.player_key_from_row(row))
        current_slot_keys[slot] = key
        slot_status[slot] = status
        salary = int(pd.to_numeric(row["salary_solver"], errors="raise"))
        salary_total += salary
        team_counts[team] = team_counts.get(team, 0) + 1
        print(
            f"{slot:5s} {player:22s} {team:3s} ${salary:>5,d} "
            f"eligible={int(pd.to_numeric(row['solver_eligible'], errors='coerce') or 0)} "
            f"proj={row.get('projection_solver')} game={row.get('game_solver')} status={status}"
        )

    print()
    print("Salary total:", salary_total)
    print("Team counts:", team_counts)
    print()

    player_status = {}
    for _, row in pool.iterrows():
        key = str(prod.player_key_from_row(row))
        game = str(row.get("game_solver") or "").upper().strip()
        if game == "DAL@NYG":
            player_status[key] = "LOCKED"
        elif game == "DEN@KC":
            player_status[key] = "SWAPPABLE"
        else:
            player_status[key] = "UNKNOWN"

    settings = late.LateSwapSettings()
    print("Settings:", settings)
    print()

    try:
        result = late.optimize_late_swap(
            pool=pool,
            current_slot_keys=current_slot_keys,
            slot_status=slot_status,
            player_status=player_status,
            settings=settings,
        )
    except Exception as exc:
        print("RESULT: FAILED")
        print("Exception type:", type(exc).__name__)
        print("Exception text:", exc)
        print()
        traceback.print_exc()
        print()
        print("DIAGNOSTIC COMPLETE — READ ONLY")
        raise SystemExit(2)

    print("RESULT: PASS")
    for key in [
        "exact_lock_ok", "salary_total", "max_team_count",
        "preserve_only_locked_rows", "projection_ready_candidate_rows",
        "projection_optimum", "projection_floor"
    ]:
        if key in result:
            print(f"{key}: {result[key]}")

    lineup = result.get("lineup")
    if isinstance(lineup, pd.DataFrame):
        print()
        print(lineup.to_string(index=False))

    print()
    print("DIAGNOSTIC COMPLETE — READ ONLY")

if __name__ == "__main__":
    main()
