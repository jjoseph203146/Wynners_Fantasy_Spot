#!/usr/bin/env python3

"""
WFS NFL Forecast Center
Forecast V1 — Historical Player Impact Builder

PURPOSE
-------
Build leakage-safe historical injury/availability features for the
WFS NFL Forecast Center.

INPUTS — READ ONLY
------------------
data/nfl.db

Tables:
    games
    injuries
    player_identity
    player_weekly_usage
    player_snap_counts

OUTPUTS
-------
processed/forecast_v1_player_impact.csv
processed/forecast_v1_team_impact.csv
processed/forecast_v1_impact_audit.json

SAFETY
------
- SQLite is opened read-only.
- No SQLite tables are created, dropped, updated, or inserted.
- No production NFL files are modified.
- Target-week realized usage/snaps are NEVER used to determine
  target-week player importance.
- Historical evidence must be strictly earlier than the target game.
- No fuzzy player matching.
- Missing history is represented explicitly as COLD, never as proof
  that a player has zero importance.
- No arbitrary fantasy/team-point injury penalties are assigned.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nfl.db"
PROCESSED_DIR = ROOT / "processed"

PLAYER_OUTPUT = (
    PROCESSED_DIR
    / "forecast_v1_player_impact.csv"
)

TEAM_OUTPUT = (
    PROCESSED_DIR
    / "forecast_v1_team_impact.csv"
)

AUDIT_OUTPUT = (
    PROCESSED_DIR
    / "forecast_v1_impact_audit.json"
)

SEASONS = (2023, 2024, 2025)

IMPACT_STATUSES = {
    "OUT",
    "DOUBTFUL",
}

OFFENSE_POSITIONS = {
    "QB",
    "RB",
    "FB",
    "WR",
    "TE",
    "C",
    "G",
    "T",
    "OT",
    "OG",
    "OL",
}

SKILL_POSITIONS = {
    "QB",
    "RB",
    "FB",
    "WR",
    "TE",
}

DEFENSE_POSITIONS = {
    "CB",
    "DB",
    "DE",
    "DL",
    "DT",
    "LB",
    "NT",
    "S",
}

SPECIAL_TEAMS_POSITIONS = {
    "K",
    "P",
    "LS",
}


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def mean_nonnull(values):
    clean = [
        safe_float(v)
        for v in values
    ]

    clean = [
        v
        for v in clean
        if v is not None
    ]

    if not clean:
        return None

    return float(
        sum(clean) / len(clean)
    )


def last_nonnull(values):
    for value in values:
        value = safe_float(value)

        if value is not None:
            return value

    return None


def history_tier(history_games: int) -> str:
    if history_games >= 5:
        return "FULL"

    if history_games >= 3:
        return "PARTIAL"

    if history_games >= 1:
        return "LIMITED"

    return "COLD"


def position_group(position) -> str:
    pos = (
        str(position).strip().upper()
        if position is not None
        else ""
    )

    if pos in OFFENSE_POSITIONS:
        return "OFFENSE"

    if pos in DEFENSE_POSITIONS:
        return "DEFENSE"

    if pos in SPECIAL_TEAMS_POSITIONS:
        return "SPECIAL_TEAMS"

    return "OTHER"


def status_weight(status: str) -> float:
    """
    This is NOT a football point penalty.

    It is only an availability encoding used for separate
    OUT and DOUBTFUL feature aggregation.

    OUT      = 1.0
    DOUBTFUL = 0.0 here because doubtful metrics are stored
               independently rather than mixed into OUT.

    No assumed scoring effect is encoded.
    """
    return 1.0 if status == "OUT" else 0.0


def sql_placeholders(n: int) -> str:
    return ",".join(
        "?"
        for _ in range(n)
    )


def check_required_columns(
    conn,
    table,
    required,
):
    rows = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    existing = {
        row["name"]
        for row in rows
    }

    missing = [
        col
        for col in required
        if col not in existing
    ]

    if missing:
        raise RuntimeError(
            f"{table} missing required columns: "
            + ", ".join(missing)
        )


# ============================================================
# DATABASE SAFETY / SCHEMA
# ============================================================

def open_read_only():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
    )

    conn.row_factory = sqlite3.Row

    return conn


def audit_database(conn):
    section("1. DATABASE SAFETY")

    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(
        f"Integrity: {integrity}"
    )

    if str(integrity).lower() != "ok":
        raise RuntimeError(
            "SQLite integrity check failed."
        )

    fk_rows = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print(
        "Foreign keys:",
        "PASS" if not fk_rows else "FAIL",
    )

    if fk_rows:
        raise RuntimeError(
            "Foreign-key audit failed."
        )

    requirements = {
        "games": [
            "game_id",
            "season",
            "week",
            "game_type",
            "game_date",
            "gametime",
            "home_team",
            "away_team",
        ],
        "injuries": [
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
            "position",
            "full_name",
            "report_status",
            "practice_status",
        ],
        "player_identity": [
            "gsis_id",
            "pfr_id",
        ],
        "player_weekly_usage": [
            "game_id",
            "season",
            "week",
            "player_id",
            "team",
            "position",
            "fanduel_points",
            "offense_snaps",
            "offense_pct",
            "opportunities",
            "touches",
            "targets",
            "carries",
        ],
        "player_snap_counts": [
            "game_id",
            "season",
            "week",
            "team",
            "pfr_player_id",
            "offense_snaps",
            "offense_pct",
            "defense_snaps",
            "defense_pct",
            "st_snaps",
            "st_pct",
        ],
    }

    for table, required in requirements.items():
        check_required_columns(
            conn,
            table,
            required,
        )

    print(
        "Required tables/columns: PASS"
    )


# ============================================================
# LOAD TARGET GAMES
# ============================================================

def load_games(conn):
    section("2. LOADING HISTORICAL GAMES")

    ph = sql_placeholders(
        len(SEASONS)
    )

    games = pd.read_sql_query(
        f"""
        SELECT
            game_id,
            season,
            week,
            game_type,
            game_date,
            gametime,
            home_team,
            away_team
        FROM games
        WHERE season IN ({ph})
        ORDER BY
            game_date,
            gametime,
            game_id
        """,
        conn,
        params=SEASONS,
    )

    if games.empty:
        raise RuntimeError(
            "No historical games loaded."
        )

    # Build Forecast-only chronology from the authoritative
    # games.game_date + games.gametime fields.
    #
    # These source fields do not carry an explicit timezone.
    # We use them only as a deterministic NFL game-order key.
    # No database field is modified.
    games["game_datetime"] = (
        pd.to_datetime(
            games["game_date"].astype(str)
            + " "
            + games["gametime"].astype(str),
            utc=True,
            errors="coerce",
        )
    )

    print(
        f"Games loaded: {len(games)}"
    )

    print(
        "Games with datetime:",
        int(
            games["game_datetime"]
            .notna()
            .sum()
        ),
    )

    return games


# ============================================================
# TEAM-GAME MAP
# ============================================================

def build_team_games(games):
    home = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "game_datetime",
            "home_team",
            "away_team",
        ]
    ].copy()

    home["team"] = home["home_team"]
    home["opponent_team"] = (
        home["away_team"]
    )
    home["home_away"] = "HOME"

    away = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "game_datetime",
            "home_team",
            "away_team",
        ]
    ].copy()

    away["team"] = away["away_team"]
    away["opponent_team"] = (
        away["home_team"]
    )
    away["home_away"] = "AWAY"

    keep = [
        "game_id",
        "season",
        "week",
        "game_type",
        "game_datetime",
        "team",
        "opponent_team",
        "home_away",
    ]

    team_games = pd.concat(
        [
            home[keep],
            away[keep],
        ],
        ignore_index=True,
    )

    dup = team_games.duplicated(
        [
            "game_id",
            "team",
        ]
    ).sum()

    if dup:
        raise RuntimeError(
            "Duplicate game/team rows in "
            f"team-game map: {dup}"
        )

    return team_games


# ============================================================
# LOAD IMPACT INJURIES
# ============================================================

def load_injuries(conn):
    section("3. LOADING OUT / DOUBTFUL PLAYERS")

    ph = sql_placeholders(
        len(SEASONS)
    )

    df = pd.read_sql_query(
        f"""
        SELECT
            season,
            week,
            game_type,
            team,
            gsis_id,
            position,
            full_name,
            report_status,
            practice_status
        FROM injuries
        WHERE season IN ({ph})
          AND UPPER(
                TRIM(
                    COALESCE(
                        report_status,
                        ''
                    )
                )
              ) IN (
                'OUT',
                'DOUBTFUL'
              )
        ORDER BY
            season,
            week,
            team,
            gsis_id
        """,
        conn,
        params=SEASONS,
    )

    if df.empty:
        raise RuntimeError(
            "No OUT/DOUBTFUL injury rows loaded."
        )

    df["status"] = (
        df["report_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print(
        f"Impact injury rows: {len(df)}"
    )

    print(
        df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    return df


# ============================================================
# ATTACH TARGET GAME
# ============================================================

def attach_target_games(
    injuries,
    team_games,
):
    section("4. ATTACHING INJURIES TO TARGET GAMES")

    game_map = team_games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "game_datetime",
            "team",
            "opponent_team",
            "home_away",
        ]
    ].copy()

    merged = injuries.merge(
        game_map,
        how="left",
        on=[
            "season",
            "week",
            "game_type",
            "team",
        ],
        validate="many_to_one",
    )

    unmatched = merged[
        merged["game_id"].isna()
    ].copy()

    print(
        f"Rows attached to target game: "
        f"{merged['game_id'].notna().sum()}"
    )

    print(
        f"Rows without target game: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:
        print()
        print(
            "WARNING: unmatched injury rows will "
            "not enter Forecast training output."
        )

    merged = merged[
        merged["game_id"].notna()
    ].copy()

    return merged, unmatched


# ============================================================
# IDENTITY
# ============================================================

def attach_identity(
    conn,
    df,
):
    section("5. ATTACHING EXACT PLAYER IDENTITY")

    identity = pd.read_sql_query(
        """
        SELECT
            gsis_id,
            pfr_id
        FROM player_identity
        """,
        conn,
    )

    identity = (
        identity
        .drop_duplicates(
            subset=["gsis_id"],
            keep="last",
        )
    )

    merged = df.merge(
        identity,
        how="left",
        on="gsis_id",
        validate="many_to_one",
    )

    print(
        "PFR identity coverage:",
        f"{merged['pfr_id'].notna().sum()} "
        f"/ {len(merged)}",
    )

    return merged


# ============================================================
# LOAD HISTORICAL EVIDENCE
# ============================================================

def load_usage(conn):
    section("6. LOADING HISTORICAL USAGE")

    ph = sql_placeholders(
        len(SEASONS)
    )

    df = pd.read_sql_query(
        f"""
        SELECT
            u.game_id,
            u.season,
            u.week,
            u.player_id,
            u.team,
            u.position,
            u.fanduel_points,
            u.offense_snaps,
            u.offense_pct,
            u.opportunities,
            u.touches,
            u.targets,
            u.carries,
            (
                g.game_date
                || ' '
                || g.gametime
            ) AS game_datetime
        FROM player_weekly_usage u
        LEFT JOIN games g
            ON u.game_id = g.game_id
        WHERE u.season IN ({ph})
        ORDER BY
            u.player_id,
            g.game_date,
            g.gametime,
            u.game_id
        """,
        conn,
        params=SEASONS,
    )

    df["game_datetime"] = (
        pd.to_datetime(
            df["game_datetime"],
            utc=True,
            errors="coerce",
        )
    )

    print(
        f"Usage rows loaded: {len(df)}"
    )

    return df


def load_snaps(conn):
    section("7. LOADING HISTORICAL SNAP COUNTS")

    ph = sql_placeholders(
        len(SEASONS)
    )

    df = pd.read_sql_query(
        f"""
        SELECT
            s.game_id,
            s.season,
            s.week,
            s.team,
            s.pfr_player_id,
            s.offense_snaps,
            s.offense_pct,
            s.defense_snaps,
            s.defense_pct,
            s.st_snaps,
            s.st_pct,
            (
                g.game_date
                || ' '
                || g.gametime
            ) AS game_datetime
        FROM player_snap_counts s
        LEFT JOIN games g
            ON s.game_id = g.game_id
        WHERE s.season IN ({ph})
        ORDER BY
            s.pfr_player_id,
            g.game_date,
            g.gametime,
            s.game_id
        """,
        conn,
        params=SEASONS,
    )

    df["game_datetime"] = (
        pd.to_datetime(
            df["game_datetime"],
            utc=True,
            errors="coerce",
        )
    )

    print(
        f"Snap rows loaded: {len(df)}"
    )

    return df


# ============================================================
# INDEX HISTORY
# ============================================================

def make_history_indexes(
    usage,
    snaps,
):
    section("8. INDEXING PRIOR HISTORY")

    usage_index = {}

    for player_id, group in usage.groupby(
        "player_id",
        sort=False,
    ):
        group = (
            group
            .sort_values(
                [
                    "game_datetime",
                    "season",
                    "week",
                    "game_id",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        usage_index[
            str(player_id)
        ] = group

    snap_index = {}

    valid_snaps = snaps[
        snaps["pfr_player_id"].notna()
    ]

    for pfr_id, group in valid_snaps.groupby(
        "pfr_player_id",
        sort=False,
    ):
        group = (
            group
            .sort_values(
                [
                    "game_datetime",
                    "season",
                    "week",
                    "game_id",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        snap_index[
            str(pfr_id)
        ] = group

    print(
        f"Usage players indexed: "
        f"{len(usage_index)}"
    )

    print(
        f"Snap players indexed: "
        f"{len(snap_index)}"
    )

    return (
        usage_index,
        snap_index,
    )


# ============================================================
# STRICT PRIOR FILTER
# ============================================================

def prior_rows(
    history,
    target_datetime,
    target_season,
    target_week,
):
    if history is None or history.empty:
        return history

    # Primary chronology:
    # strictly earlier game datetime.
    if pd.notna(target_datetime):

        dated = history[
            history["game_datetime"].notna()
        ]

        if not dated.empty:
            return dated[
                dated["game_datetime"]
                < target_datetime
            ].copy()

    # Explicit fallback when a datetime is unavailable.
    # Still strictly earlier season/week.
    return history[
        (
            history["season"]
            < target_season
        )
        |
        (
            (
                history["season"]
                == target_season
            )
            &
            (
                history["week"]
                < target_week
            )
        )
    ].copy()


# ============================================================
# PLAYER IMPACT FEATURES
# ============================================================

def build_player_impact(
    targets,
    usage_index,
    snap_index,
):
    section("9. BUILDING PLAYER IMPACT FEATURES")

    records = []

    leakage_failures = 0

    for row in targets.itertuples(
        index=False
    ):
        gsis_id = str(row.gsis_id)

        pos = (
            str(row.position).strip().upper()
            if row.position is not None
            else ""
        )

        group = position_group(pos)

        target_dt = row.game_datetime

        usage_history = prior_rows(
            usage_index.get(gsis_id),
            target_dt,
            int(row.season),
            int(row.week),
        )

        if usage_history is None:
            # Player has no prior usage record at all.
            # Preserve this as an explicit COLD start.
            usage_history = pd.DataFrame()

        elif not usage_history.empty:
            usage_history = (
                usage_history
                .sort_values(
                    [
                        "game_datetime",
                        "season",
                        "week",
                        "game_id",
                    ],
                    ascending=False,
                    na_position="last",
                )
                .head(5)
            )

        pfr_id = (
            str(row.pfr_id)
            if pd.notna(row.pfr_id)
            else None
        )

        snap_history = prior_rows(
            (
                snap_index.get(pfr_id)
                if pfr_id
                else None
            ),
            target_dt,
            int(row.season),
            int(row.week),
        )

        if snap_history is None:
            # Missing PFR identity or no prior snap history.
            # Missing history is not treated as zero impact.
            snap_history = pd.DataFrame()

        elif not snap_history.empty:
            snap_history = (
                snap_history
                .sort_values(
                    [
                        "game_datetime",
                        "season",
                        "week",
                        "game_id",
                    ],
                    ascending=False,
                    na_position="last",
                )
                .head(5)
            )

        # ----------------------------------------------
        # Hard leakage assertion
        # ----------------------------------------------

        for hist in (
            usage_history,
            snap_history,
        ):
            if hist.empty:
                continue

            if pd.notna(target_dt):

                bad = hist[
                    hist["game_datetime"].notna()
                    &
                    (
                        hist["game_datetime"]
                        >= target_dt
                    )
                ]

            else:

                bad = hist[
                    (
                        hist["season"]
                        > row.season
                    )
                    |
                    (
                        (
                            hist["season"]
                            == row.season
                        )
                        &
                        (
                            hist["week"]
                            >= row.week
                        )
                    )
                ]

            if not bad.empty:
                leakage_failures += len(bad)

        history_games = len(
            usage_history
        )

        tier = history_tier(
            history_games
        )

        # ----------------------------------------------
        # Offensive usage history
        # ----------------------------------------------

        if not usage_history.empty:

            offense_pct_values = (
                usage_history[
                    "offense_pct"
                ].tolist()
            )

            offense_snap_values = (
                usage_history[
                    "offense_snaps"
                ].tolist()
            )

            fd_values = (
                usage_history[
                    "fanduel_points"
                ].tolist()
            )

            opp_values = (
                usage_history[
                    "opportunities"
                ].tolist()
            )

            touch_values = (
                usage_history[
                    "touches"
                ].tolist()
            )

            target_values = (
                usage_history[
                    "targets"
                ].tolist()
            )

            carry_values = (
                usage_history[
                    "carries"
                ].tolist()
            )

        else:

            offense_pct_values = []
            offense_snap_values = []
            fd_values = []
            opp_values = []
            touch_values = []
            target_values = []
            carry_values = []

        # ----------------------------------------------
        # Snap history
        # ----------------------------------------------

        if not snap_history.empty:

            defense_pct_values = (
                snap_history[
                    "defense_pct"
                ].tolist()
            )

            defense_snap_values = (
                snap_history[
                    "defense_snaps"
                ].tolist()
            )

            st_pct_values = (
                snap_history[
                    "st_pct"
                ].tolist()
            )

        else:

            defense_pct_values = []
            defense_snap_values = []
            st_pct_values = []

        # ----------------------------------------------
        # Use most recent 3 / 5 observations
        # ----------------------------------------------

        def first_n(values, n):
            return values[:n]

        offense_pct_last = last_nonnull(
            offense_pct_values
        )

        offense_pct_avg_3 = mean_nonnull(
            first_n(
                offense_pct_values,
                3,
            )
        )

        offense_pct_avg_5 = mean_nonnull(
            first_n(
                offense_pct_values,
                5,
            )
        )

        defense_pct_last = last_nonnull(
            defense_pct_values
        )

        defense_pct_avg_3 = mean_nonnull(
            first_n(
                defense_pct_values,
                3,
            )
        )

        defense_pct_avg_5 = mean_nonnull(
            first_n(
                defense_pct_values,
                5,
            )
        )

        fd_avg_3 = mean_nonnull(
            first_n(
                fd_values,
                3,
            )
        )

        fd_avg_5 = mean_nonnull(
            first_n(
                fd_values,
                5,
            )
        )

        opportunities_avg_3 = mean_nonnull(
            first_n(
                opp_values,
                3,
            )
        )

        opportunities_avg_5 = mean_nonnull(
            first_n(
                opp_values,
                5,
            )
        )

        touches_avg_3 = mean_nonnull(
            first_n(
                touch_values,
                3,
            )
        )

        targets_avg_3 = mean_nonnull(
            first_n(
                target_values,
                3,
            )
        )

        carries_avg_3 = mean_nonnull(
            first_n(
                carry_values,
                3,
            )
        )

        # ----------------------------------------------
        # Starter-level role flags
        #
        # These are descriptive workload indicators.
        # They are NOT point adjustments.
        # ----------------------------------------------

        offensive_starter_role = int(
            group == "OFFENSE"
            and offense_pct_avg_3 is not None
            and offense_pct_avg_3 >= 0.60
        )

        defensive_starter_role = int(
            group == "DEFENSE"
            and defense_pct_avg_3 is not None
            and defense_pct_avg_3 >= 0.60
        )

        # ----------------------------------------------
        # Player record
        # ----------------------------------------------

        records.append(
            {
                "game_id": row.game_id,
                "game_datetime": row.game_datetime,
                "season": int(row.season),
                "week": int(row.week),
                "game_type": row.game_type,
                "team": row.team,
                "opponent_team": row.opponent_team,
                "home_away": row.home_away,

                "gsis_id": row.gsis_id,
                "pfr_id": row.pfr_id,
                "player_name": row.full_name,
                "position": pos,
                "position_group": group,

                "report_status": row.status,
                "practice_status": row.practice_status,

                "history_games": history_games,
                "history_tier": tier,

                "prior_offense_snap_last":
                    last_nonnull(
                        offense_snap_values
                    ),

                "prior_offense_pct_last":
                    offense_pct_last,

                "prior_offense_pct_avg_3":
                    offense_pct_avg_3,

                "prior_offense_pct_avg_5":
                    offense_pct_avg_5,

                "prior_defense_snap_last":
                    last_nonnull(
                        defense_snap_values
                    ),

                "prior_defense_pct_last":
                    defense_pct_last,

                "prior_defense_pct_avg_3":
                    defense_pct_avg_3,

                "prior_defense_pct_avg_5":
                    defense_pct_avg_5,

                "prior_st_pct_avg_3":
                    mean_nonnull(
                        first_n(
                            st_pct_values,
                            3,
                        )
                    ),

                "prior_fd_avg_3":
                    fd_avg_3,

                "prior_fd_avg_5":
                    fd_avg_5,

                "prior_opportunities_avg_3":
                    opportunities_avg_3,

                "prior_opportunities_avg_5":
                    opportunities_avg_5,

                "prior_touches_avg_3":
                    touches_avg_3,

                "prior_targets_avg_3":
                    targets_avg_3,

                "prior_carries_avg_3":
                    carries_avg_3,

                "offensive_starter_role":
                    offensive_starter_role,

                "defensive_starter_role":
                    defensive_starter_role,

                "cold_start_flag":
                    int(tier == "COLD"),

                "pfr_identity_missing_flag":
                    int(
                        pd.isna(row.pfr_id)
                    ),

                "source_timestamp_proven_flag":
                    int(
                        int(row.season)
                        in (2023, 2024)
                    ),
            }
        )

    if leakage_failures:
        raise RuntimeError(
            "LEAKAGE AUDIT FAILED: "
            f"{leakage_failures} non-prior "
            "history rows detected."
        )

    result = pd.DataFrame(
        records
    )

    print(
        f"Player impact rows built: "
        f"{len(result)}"
    )

    print(
        "Leakage audit: PASS"
    )

    return result


# ============================================================
# TEAM-GAME AGGREGATION
# ============================================================

def build_team_impact(
    team_games,
    player_impact,
):
    section("10. BUILDING TEAM-GAME IMPACT FEATURES")

    records = []

    grouped = {
        key: group
        for key, group in player_impact.groupby(
            [
                "game_id",
                "team",
            ],
            sort=False,
        )
    }

    def sum_col(
        frame,
        column,
    ):
        if frame.empty:
            return 0.0

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        return float(
            values.fillna(0.0).sum()
        )

    def max_col(
        frame,
        column,
    ):
        if frame.empty:
            return 0.0

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return 0.0

        return float(values.max())

    for game in team_games.itertuples(
        index=False
    ):
        group = grouped.get(
            (
                game.game_id,
                game.team,
            )
        )

        if group is None:
            group = player_impact.iloc[
                0:0
            ]

        out = group[
            group["report_status"]
            == "OUT"
        ]

        doubtful = group[
            group["report_status"]
            == "DOUBTFUL"
        ]

        out_off = out[
            out["position_group"]
            == "OFFENSE"
        ]

        out_def = out[
            out["position_group"]
            == "DEFENSE"
        ]

        doubtful_off = doubtful[
            doubtful["position_group"]
            == "OFFENSE"
        ]

        doubtful_def = doubtful[
            doubtful["position_group"]
            == "DEFENSE"
        ]

        out_skill = out[
            out["position"].isin(
                SKILL_POSITIONS
            )
        ]

        doubtful_skill = doubtful[
            doubtful["position"].isin(
                SKILL_POSITIONS
            )
        ]

        records.append(
            {
                "game_id":
                    game.game_id,

                "game_datetime":
                    game.game_datetime,

                "season":
                    int(game.season),

                "week":
                    int(game.week),

                "game_type":
                    game.game_type,

                "team":
                    game.team,

                "opponent_team":
                    game.opponent_team,

                "home_away":
                    game.home_away,

                # ------------------------------
                # Counts
                # ------------------------------

                "out_count":
                    int(len(out)),

                "doubtful_count":
                    int(len(doubtful)),

                "out_offense_count":
                    int(len(out_off)),

                "out_defense_count":
                    int(len(out_def)),

                "doubtful_offense_count":
                    int(len(doubtful_off)),

                "doubtful_defense_count":
                    int(len(doubtful_def)),

                "out_qb_count":
                    int(
                        (
                            out["position"]
                            == "QB"
                        ).sum()
                    ),

                "doubtful_qb_count":
                    int(
                        (
                            doubtful["position"]
                            == "QB"
                        ).sum()
                    ),

                "out_rb_count":
                    int(
                        (
                            out["position"]
                            == "RB"
                        ).sum()
                    ),

                "out_wr_count":
                    int(
                        (
                            out["position"]
                            == "WR"
                        ).sum()
                    ),

                "out_te_count":
                    int(
                        (
                            out["position"]
                            == "TE"
                        ).sum()
                    ),

                "out_ol_count":
                    int(
                        out["position"]
                        .isin(
                            {
                                "C",
                                "G",
                                "T",
                                "OT",
                                "OG",
                                "OL",
                            }
                        )
                        .sum()
                    ),

                # ------------------------------
                # Starter-role counts
                # ------------------------------

                "out_offensive_starter_count":
                    int(
                        out_off[
                            "offensive_starter_role"
                        ].sum()
                    ),

                "out_defensive_starter_count":
                    int(
                        out_def[
                            "defensive_starter_role"
                        ].sum()
                    ),

                "doubtful_offensive_starter_count":
                    int(
                        doubtful_off[
                            "offensive_starter_role"
                        ].sum()
                    ),

                "doubtful_defensive_starter_count":
                    int(
                        doubtful_def[
                            "defensive_starter_role"
                        ].sum()
                    ),

                # ------------------------------
                # Workload lost / at risk
                # ------------------------------

                "out_offense_snap_load":
                    sum_col(
                        out_off,
                        "prior_offense_pct_avg_3",
                    ),

                "out_defense_snap_load":
                    sum_col(
                        out_def,
                        "prior_defense_pct_avg_3",
                    ),

                "doubtful_offense_snap_load":
                    sum_col(
                        doubtful_off,
                        "prior_offense_pct_avg_3",
                    ),

                "doubtful_defense_snap_load":
                    sum_col(
                        doubtful_def,
                        "prior_defense_pct_avg_3",
                    ),

                # ------------------------------
                # Skill-player opportunity
                # ------------------------------

                "out_skill_opportunities_avg_3":
                    sum_col(
                        out_skill,
                        "prior_opportunities_avg_3",
                    ),

                "out_skill_targets_avg_3":
                    sum_col(
                        out_skill,
                        "prior_targets_avg_3",
                    ),

                "out_skill_carries_avg_3":
                    sum_col(
                        out_skill,
                        "prior_carries_avg_3",
                    ),

                "out_skill_fd_avg_3":
                    sum_col(
                        out_skill,
                        "prior_fd_avg_3",
                    ),

                "doubtful_skill_opportunities_avg_3":
                    sum_col(
                        doubtful_skill,
                        "prior_opportunities_avg_3",
                    ),

                # ------------------------------
                # Largest individual workload
                # ------------------------------

                "max_out_offense_pct_avg_3":
                    max_col(
                        out_off,
                        "prior_offense_pct_avg_3",
                    ),

                "max_out_defense_pct_avg_3":
                    max_col(
                        out_def,
                        "prior_defense_pct_avg_3",
                    ),

                # ------------------------------
                # Cold start / provenance
                # ------------------------------

                "out_cold_start_count":
                    int(
                        out[
                            "cold_start_flag"
                        ].sum()
                    ),

                "doubtful_cold_start_count":
                    int(
                        doubtful[
                            "cold_start_flag"
                        ].sum()
                    ),

                "out_missing_pfr_identity_count":
                    int(
                        out[
                            "pfr_identity_missing_flag"
                        ].sum()
                    ),

                "impact_source_timestamp_proven":
                    int(
                        int(game.season)
                        in (2023, 2024)
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        f"Team impact rows built: "
        f"{len(result)}"
    )

    return result


# ============================================================
# AUDIT
# ============================================================

def audit_outputs(
    games,
    team_games,
    player_impact,
    team_impact,
    unmatched,
):
    section("11. AUDIT")

    checks = {}

    checks[
        "player_rows_nonzero"
    ] = len(player_impact) > 0

    checks[
        "team_rows_equal_two_per_game"
    ] = (
        len(team_impact)
        == len(games) * 2
    )

    checks[
        "team_game_keys_unique"
    ] = (
        team_impact.duplicated(
            [
                "game_id",
                "team",
            ]
        ).sum()
        == 0
    )

    checks[
        "player_target_keys_present"
    ] = (
        player_impact[
            [
                "game_id",
                "team",
                "gsis_id",
            ]
        ]
        .isna()
        .any()
        .any()
        == False
    )

    checks[
        "statuses_valid"
    ] = set(
        player_impact[
            "report_status"
        ].dropna().unique()
    ).issubset(
        IMPACT_STATUSES
    )

    checks[
        "history_tiers_valid"
    ] = set(
        player_impact[
            "history_tier"
        ].dropna().unique()
    ).issubset(
        {
            "FULL",
            "PARTIAL",
            "LIMITED",
            "COLD",
        }
    )

    checks[
        "no_negative_history"
    ] = (
        (
            player_impact[
                "history_games"
            ]
            >= 0
        ).all()
    )

    checks[
        "no_sqlite_write_mode"
    ] = True

    for name, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'} "
            f"| {name}"
        )

    status_counts = (
        player_impact[
            "report_status"
        ]
        .value_counts()
        .to_dict()
    )

    tier_counts = (
        player_impact[
            "history_tier"
        ]
        .value_counts()
        .to_dict()
    )

    position_counts = (
        player_impact[
            "position"
        ]
        .value_counts()
        .to_dict()
    )

    season_player_counts = (
        player_impact
        .groupby("season")
        .size()
        .to_dict()
    )

    audit = {
        "builder":
            "build_forecast_v1_impact.py",

        "database":
            str(DB_PATH),

        "database_mode":
            "READ_ONLY",

        "seasons":
            list(SEASONS),

        "games":
            int(len(games)),

        "expected_team_game_rows":
            int(len(team_games)),

        "player_impact_rows":
            int(len(player_impact)),

        "team_impact_rows":
            int(len(team_impact)),

        "unmatched_injury_rows":
            int(len(unmatched)),

        "status_counts": {
            str(k): int(v)
            for k, v in status_counts.items()
        },

        "history_tier_counts": {
            str(k): int(v)
            for k, v in tier_counts.items()
        },

        "position_counts": {
            str(k): int(v)
            for k, v in position_counts.items()
        },

        "season_player_counts": {
            str(k): int(v)
            for k, v in season_player_counts.items()
        },

        "source_provenance": {
            "injury_source":
                "nflverse via nflreadpy.load_injuries",

            "2023_source_timestamp":
                "available in live nflverse source",

            "2024_source_timestamp":
                "available in live nflverse source",

            "2025_source_timestamp":
                "not exposed in audited live source",

            "local_injuries_updated_at":
                "WFS ingestion timestamp; not source report timestamp",
        },

        "leakage_policy": {
            "primary_order":
                "game_datetime",

            "fallback_order":
                "season/week",

            "rule":
                "historical evidence must be strictly before target game",

            "target_week_realized_usage":
                "forbidden",

            "target_week_realized_snaps":
                "forbidden",
        },

        "checks": {
            key: bool(value)
            for key, value in checks.items()
        },

        "overall_pass":
            bool(all(checks.values())),
    }

    return audit


# ============================================================
# EXPORT
# ============================================================

def export_outputs(
    player_impact,
    team_impact,
    audit,
):
    section("12. EXPORT")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_impact.to_csv(
        PLAYER_OUTPUT,
        index=False,
    )

    team_impact.to_csv(
        TEAM_OUTPUT,
        index=False,
    )

    with AUDIT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            audit,
            fh,
            indent=2,
            sort_keys=True,
        )

    print(
        f"Player impact:\n{PLAYER_OUTPUT}"
    )

    print(
        f"\nTeam impact:\n{TEAM_OUTPUT}"
    )

    print(
        f"\nAudit:\n{AUDIT_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print(
        "WFS NFL FORECAST CENTER — "
        "V1 PLAYER IMPACT BUILDER"
    )
    print("=" * 100)

    print(
        f"Database: {DB_PATH}"
    )

    print(
        "MODE: READ-ONLY SQLITE / "
        "CSV+JSON OUTPUT ONLY"
    )

    conn = open_read_only()

    try:
        audit_database(conn)

        games = load_games(conn)

        team_games = build_team_games(
            games
        )

        injuries = load_injuries(conn)

        targets, unmatched = (
            attach_target_games(
                injuries,
                team_games,
            )
        )

        targets = attach_identity(
            conn,
            targets,
        )

        usage = load_usage(conn)

        snaps = load_snaps(conn)

        (
            usage_index,
            snap_index,
        ) = make_history_indexes(
            usage,
            snaps,
        )

        player_impact = (
            build_player_impact(
                targets,
                usage_index,
                snap_index,
            )
        )

        team_impact = (
            build_team_impact(
                team_games,
                player_impact,
            )
        )

        audit = audit_outputs(
            games,
            team_games,
            player_impact,
            team_impact,
            unmatched,
        )

    finally:
        conn.close()

    export_outputs(
        player_impact,
        team_impact,
        audit,
    )

    print()
    print("=" * 100)

    if audit["overall_pass"]:
        print(
            "FORECAST V1 IMPACT DATASET: PASS"
        )
    else:
        print(
            "FORECAST V1 IMPACT DATASET: FAIL"
        )

    print(
        "NO MACHINE-LEARNING MODEL TRAINED."
    )

    print(
        "NO SQLITE TABLES MODIFIED."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
