#!/usr/bin/env python3
"""RP-2B: load and independently verify FanDuel offensive performance."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DB = ROOT / "data" / "nfl.db"
TARGET_DB = ROOT / "data" / "fanduel_player_role_performance.db"
CONTRACT = "WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
POSITIONS = {"QB", "RB", "WR", "TE"}
TOLERANCE = 0.011


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def num(value: object) -> float:
    if value is None or clean(value) == "":
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"NONFINITE_NUMERIC_VALUE:{value}")
    return result


def calculate(row: sqlite3.Row) -> tuple[float, int, int, int]:
    pass_yards = num(row["passing_yards"])
    rush_yards = num(row["rushing_yards"])
    rec_yards = num(row["receiving_yards"])
    pass_bonus = 0
    rush_bonus = 0
    rec_bonus = 0
    points = (
        pass_yards * 0.04 + num(row["passing_tds"]) * 4
        - num(row["passing_interceptions"])
        + rush_yards * 0.10 + num(row["rushing_tds"]) * 6
        + num(row["receptions"]) * 0.50
        + rec_yards * 0.10 + num(row["receiving_tds"]) * 6
        + (num(row["passing_2pt_conversions"])
           + num(row["rushing_2pt_conversions"])
           + num(row["receiving_2pt_conversions"])) * 2
        + num(row["special_teams_tds"]) * 6
        - num(row["total_fumbles_lost"]) * 2
    )
    return round(points, 2), pass_bonus, rush_bonus, rec_bonus


def main() -> int:
    print("RP-2B FANDUEL OFFENSIVE PERFORMANCE")
    print(f"SOURCE={SOURCE_DB}")
    print(f"TARGET={TARGET_DB}")
    print("SOURCE_DATABASE_MODE=READ_ONLY")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("SOLVER_INTEGRATION_ALLOWED=FALSE")
    if not SOURCE_DB.is_file() or not TARGET_DB.is_file():
        raise RuntimeError("REQUIRED_DATABASE_MISSING")

    source = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source.execute("PRAGMA query_only = ON")
    rows = source.execute(
        """
        SELECT season, week, game_id, player_id, position, team, opponent_team,
               passing_yards, passing_tds, passing_interceptions,
               rushing_yards, rushing_tds, receptions, receiving_yards,
               receiving_tds, passing_2pt_conversions,
               rushing_2pt_conversions, receiving_2pt_conversions,
               special_teams_tds, total_fumbles_lost, carries, targets,
               target_share, receiving_air_yards, air_yards_share, wopr,
               fanduel_points, fanduel_points_verified, fanduel_scored_at
        FROM player_game_stats
        WHERE position IN ('QB','RB','WR','TE')
          AND game_id IS NOT NULL AND TRIM(game_id) <> ''
          AND player_id IS NOT NULL AND TRIM(player_id) <> ''
        ORDER BY season, week, game_id, player_id
        """
    ).fetchall()

    target = sqlite3.connect(TARGET_DB)
    target.row_factory = sqlite3.Row
    target.execute("PRAGMA foreign_keys = ON")
    contract = target.execute(
        "SELECT contract FROM schema_contract WHERE singleton_id=1"
    ).fetchone()
    if contract is None or contract["contract"] != CONTRACT:
        raise RuntimeError("TARGET_CONTRACT_MISMATCH")
    known_players = {r[0] for r in target.execute("SELECT gsis_id FROM dim_player")}
    known_games = {r[0] for r in target.execute("SELECT game_id FROM dim_game")}

    facts, opportunities, mismatches = [], [], []
    team_weeks: dict[tuple[int, int, str], tuple] = {}
    seen, unverified = set(), 0
    for row in rows:
        season, week = int(row["season"]), int(row["week"])
        game_id, player_id = clean(row["game_id"]), clean(row["player_id"])
        position = clean(row["position"]).upper()
        team, opponent = clean(row["team"]).upper(), clean(row["opponent_team"]).upper()
        key = (season, week, game_id, player_id)
        if key in seen:
            raise RuntimeError(f"DUPLICATE_PLAYER_GAME:{key}")
        seen.add(key)
        if player_id not in known_players:
            raise RuntimeError(f"UNMATCHED_GSIS_ID:{player_id}")
        if game_id not in known_games:
            raise RuntimeError(f"UNMATCHED_GAME_ID:{game_id}")
        if position not in POSITIONS or not team or not opponent:
            raise RuntimeError(f"INVALID_PLAYER_GAME_CONTEXT:{key}")
        calculated, pass_bonus, rush_bonus, rec_bonus = calculate(row)
        official = num(row["fanduel_points"])
        difference = round(calculated - official, 4)
        verified = int(row["fanduel_points_verified"] or 0) == 1
        unverified += int(not verified)
        if abs(difference) > TOLERANCE:
            mismatches.append((key, calculated, official, difference))
        facts.append((
            season, week, game_id, player_id, team, opponent, position,
            num(row["passing_yards"]), num(row["passing_tds"]),
            num(row["passing_interceptions"]), num(row["rushing_yards"]),
            num(row["rushing_tds"]), num(row["receptions"]),
            num(row["receiving_yards"]), num(row["receiving_tds"]),
            num(row["passing_2pt_conversions"]),
            num(row["rushing_2pt_conversions"]),
            num(row["receiving_2pt_conversions"]),
            num(row["special_teams_tds"]), None,
            num(row["total_fumbles_lost"]), pass_bonus, rush_bonus, rec_bonus,
            calculated, official, difference,
            int(verified and abs(difference) <= TOLERANCE),
            "OWN_FUMBLE_RECOVERY_TD_SOURCE_UNAVAILABLE",
            clean(row["fanduel_scored_at"]) or None,
        ))
        opportunities.append((
            season, week, game_id, player_id, team, position,
            num(row["carries"]), num(row["targets"]), num(row["receptions"]),
            num(row["target_share"]), num(row["receiving_air_yards"]),
            num(row["air_yards_share"]), num(row["receiving_air_yards"]),
            None, num(row["wopr"]), "NFL.DB.PLAYER_GAME_STATS",
        ))
        team_key = (season, week, player_id)
        team_value = (season, week, player_id, team, opponent, game_id,
                      "NFL.DB.PLAYER_GAME_STATS")
        if team_key in team_weeks and team_weeks[team_key] != team_value:
            raise RuntimeError(f"CONFLICTING_PLAYER_TEAM_WEEK:{team_key}")
        team_weeks[team_key] = team_value
    source.close()

    if unverified:
        raise RuntimeError(f"UNVERIFIED_SOURCE_FANDUEL_ROWS:{unverified}")
    if mismatches:
        for item in mismatches[:10]:
            print(f"MISMATCH={item}")
        raise RuntimeError(f"FANDUEL_SCORING_MISMATCHES:{len(mismatches)}")

    build_id = "RP2B_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = utc_now()
    try:
        target.execute("BEGIN IMMEDIATE")
        target.execute("DELETE FROM fact_player_game_fanduel")
        target.execute("DELETE FROM fact_player_opportunity")
        target.execute("DELETE FROM bridge_player_team_week")
        target.executemany("INSERT INTO bridge_player_team_week VALUES (?,?,?,?,?,?,?)",
                           list(team_weeks.values()))
        target.executemany("INSERT INTO fact_player_opportunity VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           opportunities)
        target.executemany("INSERT INTO fact_player_game_fanduel VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           facts)
        target.execute(
            """INSERT INTO build_manifest
               (build_id,stage,contract,source_nfl_db_path,source_nfl_db_sha256,
                source_cutoff_at,build_started_at,build_completed_at,status,
                production_tables_modified,solver_integration_allowed,notes)
               VALUES (?,'RP-2B',?,?,?,?,?,?,'OK',0,0,?)""",
            (build_id, CONTRACT, str(SOURCE_DB), sha256_file(SOURCE_DB),
             started, started, utc_now(), f"offense_rows={len(facts)}"),
        )
        foreign_keys = target.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise RuntimeError(f"FOREIGN_KEY_ERRORS:{len(foreign_keys)}")
        target.commit()
    except Exception:
        target.rollback()
        raise

    by_season = target.execute(
        "SELECT season,COUNT(*) FROM fact_player_game_fanduel GROUP BY season ORDER BY season"
    ).fetchall()
    quick = target.execute("PRAGMA quick_check").fetchone()[0]
    target.close()
    print(f"PLAYER_FACT_ROWS={len(facts)}")
    print(f"OPPORTUNITY_ROWS={len(opportunities)}")
    print(f"PLAYER_TEAM_WEEK_ROWS={len(team_weeks)}")
    for season, count in by_season:
        print(f"SEASON={season}|PLAYER_FACT_ROWS={count}")
    print("SCORING_MISMATCHES=0")
    print("UNVERIFIED_SOURCE_ROWS=0")
    print("OWN_FUMBLE_RECOVERY_TD_STATUS=SOURCE_UNAVAILABLE")
    print(f"QUICK_CHECK={quick}")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("STATUS=OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}:{exc}")
        print("STATUS=FAILED")
        sys.exit(1)
