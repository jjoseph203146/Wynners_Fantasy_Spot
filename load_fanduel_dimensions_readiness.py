#!/usr/bin/env python3
"""RP-2A: load exact dimensions and game-readiness status.

Source data/nfl.db is opened read-only. Writes are restricted to the new
data/fanduel_player_role_performance.db analytical database.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DB = ROOT / "data" / "nfl.db"
TARGET_DB = ROOT / "data" / "fanduel_player_role_performance.db"
CONTRACT = "WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
POSITIONS = {"QB", "RB", "WR", "TE"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def target_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(TARGET_DB)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def validate_files() -> None:
    if not SOURCE_DB.is_file():
        raise RuntimeError(f"SOURCE_DB_MISSING:{SOURCE_DB}")
    if not TARGET_DB.is_file():
        raise RuntimeError(
            "TARGET_DB_MISSING:run create_fanduel_performance_db.py first"
        )


def validate_target(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT contract, fan_duel_only, production_integration_allowed "
        "FROM schema_contract WHERE singleton_id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("TARGET_SCHEMA_CONTRACT_MISSING")
    if row["contract"] != CONTRACT:
        raise RuntimeError(f"TARGET_CONTRACT_MISMATCH:{row['contract']}")
    if row["fan_duel_only"] != 1 or row["production_integration_allowed"] != 0:
        raise RuntimeError("TARGET_SAFETY_CONTRACT_FAILED")


def validate_source(connection: sqlite3.Connection) -> None:
    required = {
        "games",
        "player_identity",
        "player_game_stats",
        "team_game_stats",
        "team_defense_game_stats",
        "team_defense_fanduel_scoring",
    }
    present = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"SOURCE_TABLES_MISSING:{missing}")


def clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def canonical_position(value: object) -> str:
    return clean_text(value).upper()


def load_games(source: sqlite3.Connection) -> list[tuple]:
    rows = source.execute(
        """
        SELECT game_id, season, week, game_type, game_date, gametime,
               away_team, home_team, away_score, home_score,
               completed, updated_at
        FROM games
        WHERE game_id IS NOT NULL
          AND TRIM(game_id) <> ''
          AND season IS NOT NULL
          AND week IS NOT NULL
          AND away_team IS NOT NULL
          AND home_team IS NOT NULL
        ORDER BY season, week, game_id
        """
    ).fetchall()
    output = []
    seen: set[str] = set()
    for row in rows:
        game_id = clean_text(row["game_id"])
        if game_id in seen:
            raise RuntimeError(f"DUPLICATE_SOURCE_GAME_ID:{game_id}")
        seen.add(game_id)
        game_date = clean_text(row["game_date"])
        gametime = clean_text(row["gametime"])
        kickoff = f"{game_date} {gametime}".strip() or None
        away_team = clean_text(row["away_team"]).upper()
        home_team = clean_text(row["home_team"]).upper()
        if not away_team or not home_team or away_team == home_team:
            raise RuntimeError(f"INVALID_GAME_TEAMS:{game_id}")
        output.append(
            (
                game_id,
                int(row["season"]),
                int(row["week"]),
                clean_text(row["game_type"]) or None,
                kickoff,
                away_team,
                home_team,
                row["away_score"],
                row["home_score"],
                1 if row["completed"] else 0,
                clean_text(row["updated_at"]) or None,
            )
        )
    return output


def load_players(source: sqlite3.Connection) -> tuple[list[tuple], int]:
    stat_history: dict[str, dict[str, object]] = {}
    stat_rows = source.execute(
        """
        SELECT player_id, season, week, game_id, position
        FROM player_game_stats
        WHERE player_id IS NOT NULL AND TRIM(player_id) <> ''
          AND position IN ('QB', 'RB', 'WR', 'TE')
        ORDER BY player_id, season, week, game_id, position
        """
    ).fetchall()
    for stat in stat_rows:
        player_id = clean_text(stat["player_id"])
        season = int(stat["season"])
        week = int(stat["week"])
        game_id = clean_text(stat["game_id"])
        position = canonical_position(stat["position"])
        state = stat_history.setdefault(
            player_id,
            {
                "first_season": season,
                "last_season": season,
                "latest_key": (season, week, game_id),
                "latest_position": position,
            },
        )
        state["first_season"] = min(int(state["first_season"]), season)
        state["last_season"] = max(int(state["last_season"]), season)
        current_key = (season, week, game_id)
        if current_key > state["latest_key"]:
            state["latest_key"] = current_key
            state["latest_position"] = position
        elif current_key == state["latest_key"] and position != state["latest_position"]:
            raise RuntimeError(
                f"CONFLICTING_LATEST_FANDUEL_POSITION:{player_id}:"
                f"{state['latest_position']}:{position}"
            )
    rows = source.execute(
        """
        SELECT gsis_id, full_name, football_name, position
        FROM player_identity
        WHERE gsis_id IS NOT NULL AND TRIM(gsis_id) <> ''
        ORDER BY gsis_id
        """
    ).fetchall()
    output: list[tuple] = []
    excluded_positions = 0
    seen: dict[str, tuple[str, str]] = {}
    for row in rows:
        gsis_id = clean_text(row["gsis_id"])
        stats = stat_history.get(gsis_id)
        position = (
            canonical_position(stats["latest_position"])
            if stats is not None
            else canonical_position(row["position"])
        )
        if position not in POSITIONS:
            excluded_positions += 1
            continue
        name = clean_text(row["full_name"]) or clean_text(row["football_name"])
        if not name:
            raise RuntimeError(f"BLANK_PLAYER_NAME:{gsis_id}")
        identity = (name, position)
        if gsis_id in seen and seen[gsis_id] != identity:
            raise RuntimeError(
                f"CONFLICTING_GSIS_IDENTITY:{gsis_id}:{seen[gsis_id]}:{identity}"
            )
        if gsis_id in seen:
            continue
        seen[gsis_id] = identity
        first_season = stats["first_season"] if stats is not None else None
        last_season = stats["last_season"] if stats is not None else None
        output.append(
            (gsis_id, name, position, first_season, last_season, "EXACT_GSIS")
        )
    return output, excluded_positions


def build_readiness(source: sqlite3.Connection) -> list[tuple]:
    games = source.execute(
        """
        SELECT game_id, completed
        FROM games
        WHERE game_id IS NOT NULL AND TRIM(game_id) <> ''
        ORDER BY game_id
        """
    ).fetchall()
    output = []
    audited_at = utc_now()
    for game in games:
        game_id = clean_text(game["game_id"])
        completed = 1 if game["completed"] else 0
        player_rows, verified_player_rows = source.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN fanduel_points_verified = 1 THEN 1 ELSE 0 END), 0)
            FROM player_game_stats WHERE game_id = ?
            """,
            (game_id,),
        ).fetchone()
        team_rows = source.execute(
            "SELECT COUNT(*) FROM team_game_stats WHERE game_id = ?",
            (game_id,),
        ).fetchone()[0]
        dst_rows = source.execute(
            "SELECT COUNT(*) FROM team_defense_game_stats WHERE game_id = ?",
            (game_id,),
        ).fetchone()[0]
        dst_fd_rows = source.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN fanduel_scoring_verified = 1 THEN 1 ELSE 0 END), 0)
            FROM team_defense_fanduel_scoring WHERE game_id = ?
            """,
            (game_id,),
        ).fetchone()[0]

        reasons = []
        if not completed:
            reasons.append("NOT_COMPLETED")
        if completed and player_rows == 0:
            reasons.append("NO_PLAYER_RESULTS")
        elif completed and verified_player_rows != player_rows:
            reasons.append("PLAYER_FD_NOT_FULLY_VERIFIED")
        if completed and team_rows != 2:
            reasons.append(f"TEAM_GAME_ROWS_{team_rows}")
        if completed and dst_rows != 2:
            reasons.append(f"DST_STAT_ROWS_{dst_rows}")
        if completed and dst_fd_rows != 2:
            reasons.append(f"DST_FD_VERIFIED_ROWS_{dst_fd_rows}")
        ready = int(completed and player_rows > 0 and verified_player_rows == player_rows
                    and team_rows == 2 and dst_rows == 2 and dst_fd_rows == 2)
        reason_code = "READY" if ready else "|".join(reasons)
        output.append(
            (
                game_id,
                completed,
                int(player_rows),
                int(verified_player_rows),
                int(team_rows),
                int(dst_rows),
                int(dst_fd_rows),
                ready,
                reason_code,
                audited_at,
            )
        )
    return output


def main() -> int:
    print("RP-2A FANDUEL DIMENSIONS AND GAME READINESS")
    print(f"SOURCE={SOURCE_DB}")
    print(f"TARGET={TARGET_DB}")
    print("SOURCE_DATABASE_MODE=READ_ONLY")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("SOLVER_INTEGRATION_ALLOWED=FALSE")
    validate_files()
    source_hash = sha256_file(SOURCE_DB)
    started = utc_now()
    build_id = "RP2A_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with source_connection() as source:
        validate_source(source)
        games = load_games(source)
        players, excluded_positions = load_players(source)
        readiness = build_readiness(source)

    with target_connection() as target:
        validate_target(target)
        try:
            target.execute("BEGIN IMMEDIATE")
            target.executemany(
                """
                INSERT INTO dim_game (
                    game_id, season, week, game_type, kickoff_at,
                    away_team, home_team, away_score, home_score,
                    completed, source_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    season=excluded.season,
                    week=excluded.week,
                    game_type=excluded.game_type,
                    kickoff_at=excluded.kickoff_at,
                    away_team=excluded.away_team,
                    home_team=excluded.home_team,
                    away_score=excluded.away_score,
                    home_score=excluded.home_score,
                    completed=excluded.completed,
                    source_updated_at=excluded.source_updated_at
                """,
                games,
            )
            target.executemany(
                """
                INSERT INTO dim_player (
                    gsis_id, player_name, canonical_position,
                    first_seen_season, last_seen_season, identity_status
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(gsis_id) DO UPDATE SET
                    player_name=excluded.player_name,
                    canonical_position=excluded.canonical_position,
                    first_seen_season=excluded.first_seen_season,
                    last_seen_season=excluded.last_seen_season,
                    identity_status=excluded.identity_status
                """,
                players,
            )
            target.executemany(
                """
                INSERT INTO game_readiness (
                    game_id, completed, player_rows, verified_player_fd_rows,
                    team_game_rows, dst_stat_rows, verified_dst_fd_rows,
                    performance_ready, reason_code, audited_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    completed=excluded.completed,
                    player_rows=excluded.player_rows,
                    verified_player_fd_rows=excluded.verified_player_fd_rows,
                    team_game_rows=excluded.team_game_rows,
                    dst_stat_rows=excluded.dst_stat_rows,
                    verified_dst_fd_rows=excluded.verified_dst_fd_rows,
                    performance_ready=excluded.performance_ready,
                    reason_code=excluded.reason_code,
                    audited_at=excluded.audited_at
                """,
                readiness,
            )
            target.execute(
                """
                INSERT INTO build_manifest (
                    build_id, stage, contract, source_nfl_db_path,
                    source_nfl_db_sha256, source_cutoff_at,
                    build_started_at, build_completed_at, status,
                    production_tables_modified, solver_integration_allowed, notes
                ) VALUES (?, 'RP-2A', ?, ?, ?, ?, ?, ?, 'OK', 0, 0, ?)
                """,
                (
                    build_id,
                    CONTRACT,
                    str(SOURCE_DB),
                    source_hash,
                    started,
                    started,
                    utc_now(),
                    f"games={len(games)};players={len(players)};readiness={len(readiness)}",
                ),
            )
            foreign_keys = target.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise RuntimeError(f"FOREIGN_KEY_ERRORS:{len(foreign_keys)}")
            target.commit()
        except Exception:
            target.rollback()
            raise

        ready_count = target.execute(
            "SELECT COUNT(*) FROM game_readiness WHERE performance_ready = 1"
        ).fetchone()[0]
        completed_not_ready = target.execute(
            """
            SELECT COUNT(*) FROM game_readiness
            WHERE completed = 1 AND performance_ready = 0
            """
        ).fetchone()[0]
        current_2026 = target.execute(
            """
            SELECT COUNT(*) FROM game_readiness r
            JOIN dim_game g ON g.game_id = r.game_id
            WHERE g.season = 2026 AND r.performance_ready = 1
            """
        ).fetchone()[0]
        reason_rows = target.execute(
            """
            SELECT reason_code, COUNT(*) AS games
            FROM game_readiness
            WHERE completed = 1 AND performance_ready = 0
            GROUP BY reason_code ORDER BY games DESC, reason_code
            """
        ).fetchall()
        quick_check = target.execute("PRAGMA quick_check").fetchone()[0]

    print(f"GAMES_LOADED={len(games)}")
    print(f"PLAYERS_LOADED={len(players)}")
    print(f"NON_OFFENSE_IDENTITIES_EXCLUDED={excluded_positions}")
    print(f"READINESS_ROWS={len(readiness)}")
    print(f"ALL_SEASONS_READY_GAMES={ready_count}")
    print(f"COMPLETED_NOT_READY={completed_not_ready}")
    print(f"2026_READY_GAMES={current_2026}")
    for row in reason_rows:
        print(f"NOT_READY|GAMES={row['games']}|REASON={row['reason_code']}")
    print(f"QUICK_CHECK={quick_check}")
    print("PERFORMANCE_ROWS_LOADED=FALSE")
    print("STATUS=OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}:{exc}")
        print("STATUS=FAILED")
        sys.exit(1)
