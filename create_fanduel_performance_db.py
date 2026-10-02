#!/usr/bin/env python3
"""RP-1: create the empty FanDuel player/role performance database.

This script writes only to data/fanduel_player_role_performance.db.
It does not attach, update, or modify data/nfl.db or any solver database.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "fanduel_player_role_performance.db"
CONTRACT = "WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
SCHEMA_VERSION = 3
APPLICATION_ID = 0x57465346  # WFSF


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_contract (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    contract TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    fan_duel_only INTEGER NOT NULL CHECK (fan_duel_only = 1),
    offense_and_dst_only INTEGER NOT NULL CHECK (offense_and_dst_only = 1),
    scoring_contract TEXT NOT NULL,
    production_integration_allowed INTEGER NOT NULL DEFAULT 0
        CHECK (production_integration_allowed = 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_manifest (
    build_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    contract TEXT NOT NULL,
    source_nfl_db_path TEXT,
    source_nfl_db_sha256 TEXT,
    source_cutoff_at TEXT,
    build_started_at TEXT NOT NULL,
    build_completed_at TEXT,
    status TEXT NOT NULL,
    production_tables_modified INTEGER NOT NULL DEFAULT 0
        CHECK (production_tables_modified = 0),
    solver_integration_allowed INTEGER NOT NULL DEFAULT 0
        CHECK (solver_integration_allowed = 0),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS source_file_manifest (
    source_file_id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL CHECK (source_size_bytes >= 0),
    slate_date TEXT,
    slate_type TEXT,
    captured_at TEXT NOT NULL,
    imported_at TEXT,
    import_status TEXT NOT NULL,
    UNIQUE (source_sha256, source_type)
);

CREATE TABLE IF NOT EXISTS dim_game (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT,
    kickoff_at TEXT,
    game_date TEXT,
    kickoff_local_time TEXT,
    kickoff_timezone TEXT,
    kickoff_utc TEXT,
    away_team TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_score INTEGER,
    home_score INTEGER,
    completed INTEGER NOT NULL CHECK (completed IN (0, 1)),
    source_updated_at TEXT,
    CHECK (away_team <> home_team)
);

CREATE TABLE IF NOT EXISTS dim_player (
    gsis_id TEXT PRIMARY KEY,
    player_name TEXT NOT NULL,
    canonical_position TEXT NOT NULL
        CHECK (canonical_position IN ('QB', 'RB', 'WR', 'TE')),
    first_seen_season INTEGER,
    last_seen_season INTEGER,
    identity_status TEXT NOT NULL DEFAULT 'EXACT_GSIS'
        CHECK (identity_status = 'EXACT_GSIS')
);

CREATE TABLE IF NOT EXISTS bridge_player_team_week (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    team TEXT NOT NULL,
    opponent_team TEXT,
    game_id TEXT REFERENCES dim_game(game_id),
    source_authority TEXT NOT NULL,
    PRIMARY KEY (season, week, gsis_id)
);

CREATE TABLE IF NOT EXISTS game_readiness (
    game_id TEXT PRIMARY KEY REFERENCES dim_game(game_id),
    completed INTEGER NOT NULL CHECK (completed IN (0, 1)),
    player_rows INTEGER NOT NULL DEFAULT 0 CHECK (player_rows >= 0),
    verified_player_fd_rows INTEGER NOT NULL DEFAULT 0
        CHECK (verified_player_fd_rows >= 0),
    team_game_rows INTEGER NOT NULL DEFAULT 0 CHECK (team_game_rows >= 0),
    dst_stat_rows INTEGER NOT NULL DEFAULT 0 CHECK (dst_stat_rows >= 0),
    verified_dst_fd_rows INTEGER NOT NULL DEFAULT 0
        CHECK (verified_dst_fd_rows >= 0),
    performance_ready INTEGER NOT NULL DEFAULT 0
        CHECK (performance_ready IN (0, 1)),
    reason_code TEXT NOT NULL,
    audited_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_player_game_fanduel (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    team TEXT NOT NULL,
    opponent_team TEXT NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE')),
    passing_yards REAL NOT NULL DEFAULT 0,
    passing_tds REAL NOT NULL DEFAULT 0,
    passing_interceptions REAL NOT NULL DEFAULT 0,
    rushing_yards REAL NOT NULL DEFAULT 0,
    rushing_tds REAL NOT NULL DEFAULT 0,
    receptions REAL NOT NULL DEFAULT 0,
    receiving_yards REAL NOT NULL DEFAULT 0,
    receiving_tds REAL NOT NULL DEFAULT 0,
    passing_2pt_conversions REAL NOT NULL DEFAULT 0,
    rushing_2pt_conversions REAL NOT NULL DEFAULT 0,
    receiving_2pt_conversions REAL NOT NULL DEFAULT 0,
    special_teams_tds REAL NOT NULL DEFAULT 0,
    own_fumble_recovery_tds REAL,
    total_fumbles_lost REAL NOT NULL DEFAULT 0,
    passing_300_bonus INTEGER NOT NULL DEFAULT 0 CHECK (passing_300_bonus IN (0, 1)),
    rushing_100_bonus INTEGER NOT NULL DEFAULT 0 CHECK (rushing_100_bonus IN (0, 1)),
    receiving_100_bonus INTEGER NOT NULL DEFAULT 0 CHECK (receiving_100_bonus IN (0, 1)),
    calculated_fanduel_points REAL NOT NULL,
    official_fanduel_points REAL,
    points_difference REAL,
    scoring_verified INTEGER NOT NULL DEFAULT 0 CHECK (scoring_verified IN (0, 1)),
    special_scoring_status TEXT NOT NULL,
    source_scored_at TEXT,
    PRIMARY KEY (season, week, game_id, gsis_id)
);

CREATE TABLE IF NOT EXISTS fact_dst_game_fanduel (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    team TEXT NOT NULL,
    opponent_team TEXT NOT NULL,
    sacks REAL NOT NULL DEFAULT 0,
    interceptions REAL NOT NULL DEFAULT 0,
    fumble_recoveries REAL NOT NULL DEFAULT 0,
    safeties REAL NOT NULL DEFAULT 0,
    blocked_kicks REAL NOT NULL DEFAULT 0,
    return_tds REAL NOT NULL DEFAULT 0,
    extra_point_returns REAL NOT NULL DEFAULT 0,
    points_allowed INTEGER NOT NULL,
    points_allowed_score REAL NOT NULL,
    calculated_fanduel_points REAL NOT NULL,
    official_fanduel_points REAL,
    points_difference REAL,
    scoring_verified INTEGER NOT NULL DEFAULT 0 CHECK (scoring_verified IN (0, 1)),
    source_contract TEXT NOT NULL,
    PRIMARY KEY (season, week, game_id, team)
);

CREATE TABLE IF NOT EXISTS fact_player_opportunity (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    team TEXT NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE')),
    carries REAL,
    targets REAL,
    receptions REAL,
    target_share REAL,
    air_yards REAL,
    air_yards_share REAL,
    receiving_air_yards REAL,
    rushing_share REAL,
    weighted_opportunity REAL,
    source_authority TEXT NOT NULL,
    PRIMARY KEY (season, week, game_id, gsis_id)
);

CREATE TABLE IF NOT EXISTS fact_player_participation (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    team TEXT NOT NULL,
    offense_snaps REAL,
    offense_snap_share REAL,
    source_authority TEXT NOT NULL,
    PRIMARY KEY (season, week, game_id, gsis_id)
);

CREATE TABLE IF NOT EXISTS fact_player_role_observed (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    role_contract TEXT NOT NULL,
    primary_role TEXT NOT NULL,
    rushing_role TEXT NOT NULL,
    receiving_role TEXT NOT NULL,
    red_zone_role TEXT NOT NULL,
    role_confidence REAL NOT NULL CHECK (role_confidence BETWEEN 0 AND 1),
    evidence_json TEXT NOT NULL CHECK (json_valid(evidence_json)),
    reason_codes_json TEXT NOT NULL CHECK (json_valid(reason_codes_json)),
    classified_at TEXT NOT NULL,
    PRIMARY KEY (season, week, game_id, gsis_id, role_contract)
);

CREATE TABLE IF NOT EXISTS snapshot_player_role_pregame (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    gsis_id TEXT NOT NULL REFERENCES dim_player(gsis_id),
    snapshot_at TEXT NOT NULL,
    kickoff_at TEXT NOT NULL,
    role_contract TEXT NOT NULL,
    expected_primary_role TEXT NOT NULL,
    expected_role_confidence REAL NOT NULL
        CHECK (expected_role_confidence BETWEEN 0 AND 1),
    evidence_through_game_id TEXT,
    leakage_check_passed INTEGER NOT NULL CHECK (leakage_check_passed = 1),
    PRIMARY KEY (season, week, game_id, gsis_id, snapshot_at, role_contract),
    CHECK (snapshot_at < kickoff_at)
);

CREATE TABLE IF NOT EXISTS fact_projection_vs_actual (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    entity_type TEXT NOT NULL CHECK (entity_type IN ('PLAYER', 'DST')),
    entity_id TEXT NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE', 'DST')),
    projection_snapshot_at TEXT NOT NULL,
    kickoff_at TEXT NOT NULL,
    projection_source TEXT NOT NULL,
    projected_fanduel_points REAL NOT NULL,
    actual_fanduel_points REAL NOT NULL,
    projection_error REAL NOT NULL,
    value_over_projection REAL NOT NULL,
    ceiling_result INTEGER NOT NULL CHECK (ceiling_result IN (0, 1)),
    cutoff_check_passed INTEGER NOT NULL CHECK (cutoff_check_passed = 1),
    PRIMARY KEY (game_id, entity_type, entity_id, projection_snapshot_at, projection_source),
    CHECK (projection_snapshot_at < kickoff_at)
);

CREATE TABLE IF NOT EXISTS fact_fanduel_slate_player (
    source_file_id INTEGER NOT NULL REFERENCES source_file_manifest(source_file_id),
    slate_date TEXT NOT NULL,
    slate_type TEXT NOT NULL,
    fanduel_player_id TEXT NOT NULL,
    gsis_id TEXT REFERENCES dim_player(gsis_id),
    player_name TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent_team TEXT,
    position TEXT NOT NULL CHECK (position IN ('QB', 'RB', 'WR', 'TE', 'DST')),
    salary INTEGER NOT NULL CHECK (salary > 0),
    roster_status TEXT,
    identity_status TEXT NOT NULL,
    PRIMARY KEY (source_file_id, fanduel_player_id)
);

CREATE TABLE IF NOT EXISTS fact_lineup_validation_fixture (
    fixture_id TEXT NOT NULL,
    slate_type TEXT NOT NULL,
    lineup_total REAL NOT NULL,
    slot_number INTEGER NOT NULL CHECK (slot_number BETWEEN 1 AND 9),
    roster_slot TEXT NOT NULL CHECK (roster_slot IN ('QB', 'RB', 'WR', 'TE', 'FLEX', 'DST')),
    entity_name TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent_team TEXT,
    recorded_fanduel_points REAL NOT NULL,
    salary INTEGER,
    ownership REAL,
    source_file_id INTEGER REFERENCES source_file_manifest(source_file_id),
    human_verified INTEGER NOT NULL CHECK (human_verified IN (0, 1)),
    PRIMARY KEY (fixture_id, slot_number)
);


CREATE TABLE IF NOT EXISTS snapshot_fanduel_slate_eligibility (
    source_file_id INTEGER NOT NULL
        REFERENCES source_file_manifest(source_file_id),
    fanduel_player_id TEXT NOT NULL,
    snapshot_at_utc TEXT NOT NULL,
    game_id TEXT NOT NULL REFERENCES dim_game(game_id),
    kickoff_utc TEXT NOT NULL,
    player_name TEXT NOT NULL,
    gsis_id TEXT REFERENCES dim_player(gsis_id),
    team TEXT NOT NULL,
    opponent_team TEXT,
    position TEXT NOT NULL
        CHECK (position IN ('QB', 'RB', 'WR', 'TE', 'DST')),
    salary INTEGER NOT NULL CHECK (salary > 0),
    fanduel_roster_status TEXT,
    nfl_roster_status TEXT,
    depth_position TEXT,
    depth_rank INTEGER,
    injury_gate TEXT,
    availability_risk TEXT,
    eligibility_status TEXT NOT NULL
        CHECK (
            eligibility_status IN (
                'ELIGIBLE',
                'REVIEW',
                'EXCLUDED'
            )
        ),
    reason_codes_json TEXT NOT NULL,
    warning_codes_json TEXT NOT NULL,
    roster_source_updated_at TEXT,
    depth_source_snapshot_at TEXT,
    injury_source_updated_at TEXT,
    cutoff_check_passed INTEGER NOT NULL
        CHECK (cutoff_check_passed = 1),
    PRIMARY KEY (
        source_file_id,
        fanduel_player_id,
        snapshot_at_utc
    )
);
CREATE INDEX IF NOT EXISTS idx_slate_eligibility_status
    ON snapshot_fanduel_slate_eligibility (
        source_file_id,
        snapshot_at_utc,
        eligibility_status,
        position
    );
CREATE TABLE IF NOT EXISTS audit_result (
    audit_id INTEGER PRIMARY KEY,
    build_id TEXT REFERENCES build_manifest(build_id),
    audit_name TEXT NOT NULL,
    entity_key TEXT,
    status TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL', 'WARN')),
    observed_value TEXT,
    expected_value TEXT,
    reason_code TEXT NOT NULL,
    audited_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_game_season_week
    ON dim_game(season, week, game_id);
CREATE INDEX IF NOT EXISTS idx_player_team_week
    ON bridge_player_team_week(season, week, team, gsis_id);
CREATE INDEX IF NOT EXISTS idx_player_fd_position
    ON fact_player_game_fanduel(season, week, position, calculated_fanduel_points);
CREATE INDEX IF NOT EXISTS idx_dst_fd_week
    ON fact_dst_game_fanduel(season, week, calculated_fanduel_points);
CREATE INDEX IF NOT EXISTS idx_role_player_history
    ON fact_player_role_observed(gsis_id, season, week);
CREATE INDEX IF NOT EXISTS idx_slate_salary
    ON fact_fanduel_slate_player(slate_date, slate_type, position, salary);
"""


EXPECTED_TABLES = {
    "schema_contract",
    "build_manifest",
    "source_file_manifest",
    "dim_game",
    "dim_player",
    "bridge_player_team_week",
    "game_readiness",
    "fact_player_game_fanduel",
    "fact_dst_game_fanduel",
    "fact_player_opportunity",
    "fact_player_participation",
    "fact_player_role_observed",
    "snapshot_player_role_pregame",
    "fact_projection_vs_actual",
    "fact_fanduel_slate_player",
    "fact_lineup_validation_fixture",
    "snapshot_fanduel_slate_eligibility",
    "audit_result",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def existing_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def validate_existing_database(connection: sqlite3.Connection) -> None:
    app_id = connection.execute("PRAGMA application_id").fetchone()[0]
    user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = existing_tables(connection)
    if tables and app_id != APPLICATION_ID:
        raise RuntimeError(
            "REFUSING_NON_WFS_DATABASE: target exists but application_id does not match"
        )
    if tables and user_version not in (0, SCHEMA_VERSION):
        raise RuntimeError(
            f"SCHEMA_VERSION_MISMATCH: found={user_version} expected={SCHEMA_VERSION}"
        )


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existed_before = DB_PATH.exists()
    print("RP-1 FANDUEL PLAYER/ROLE DATABASE SKELETON")
    print(f"TARGET={DB_PATH}")
    print("FANDUEL_ONLY=TRUE")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("SOLVER_INTEGRATION_ALLOWED=FALSE")

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            validate_existing_database(connection)
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            timestamp = utc_now()
            connection.execute(
                """
                INSERT INTO schema_contract (
                    singleton_id, contract, schema_version, fan_duel_only,
                    offense_and_dst_only, scoring_contract,
                    production_integration_allowed, created_at, updated_at
                ) VALUES (1, ?, ?, 1, 1, ?, 0, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    updated_at = excluded.updated_at
                WHERE schema_contract.contract = excluded.contract
                  AND schema_contract.schema_version = excluded.schema_version
                """,
                (
                    CONTRACT,
                    SCHEMA_VERSION,
                    "USER_SUPPLIED_FANDUEL_CONTEST_RULES_2026-09-14",
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()

            tables = existing_tables(connection)
            missing = sorted(EXPECTED_TABLES - tables)
            extra = sorted(tables - EXPECTED_TABLES)
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

        print(f"EXISTED_BEFORE={str(existed_before).upper()}")
        print(f"TABLES={len(tables)}")
        print(f"MISSING_TABLES={len(missing)}")
        print(f"EXTRA_TABLES={len(extra)}")
        print(f"QUICK_CHECK={integrity}")
        print(f"FOREIGN_KEY_ERRORS={len(foreign_key_errors)}")

        if missing or integrity != "ok" or foreign_key_errors:
            print(f"MISSING={missing}")
            print("STATUS=REVIEW_REQUIRED")
            return 2

        print("ROW_LOAD_PERFORMED=FALSE")
        print("STATUS=OK")
        return 0
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}: {exc}")
        print("STATUS=FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
