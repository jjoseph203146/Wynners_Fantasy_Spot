#!/usr/bin/env python3

"""
WFS NFL Forecast Center
Forecast V1 Historical Team-Game Dataset Builder

Purpose
-------
Create a leakage-safe historical modeling dataset with one row per
team per completed NFL game.

Historical training universe:
    2023-2025

Expected:
    855 games
    1,710 team-game rows

Important safety rules
----------------------
1. READ ONLY against data/nfl.db.
2. This script NEVER writes to SQLite.
3. Existing optimizer files/tables are not modified.
4. Existing pregame feature tables are consumed as frozen inputs.
5. Current-game outcomes are included only as modeling targets.
6. Coach historical statistics are calculated strictly from prior games.
7. Injury/impact inputs use pregame player features only.
8. OUT / DOUBTFUL / QUESTIONABLE remain separate.
9. No fuzzy matching.
10. No machine-learning model is trained here.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nfl.db"

OUTPUT_DIR = ROOT / "processed"

DATASET_PATH = (
    OUTPUT_DIR
    / "forecast_v1_team_game_training.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "forecast_v1_team_game_training_audit.json"
)

HISTORICAL_SEASONS = (2023, 2024, 2025)

EXPECTED_GAMES = 855
EXPECTED_ROWS = EXPECTED_GAMES * 2


# ============================================================
# HELPERS
# ============================================================

def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def safe_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def require_table(
    conn: sqlite3.Connection,
    table: str,
):
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table,),
    ).fetchone()

    if not row or row[0] != 1:
        raise RuntimeError(
            f"Required table missing: {table}"
        )


def table_columns(
    conn: sqlite3.Connection,
    table: str,
):
    return [
        row["name"]
        for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    ]


def require_columns(
    conn: sqlite3.Connection,
    table: str,
    required: set[str],
):
    actual = set(
        table_columns(
            conn,
            table,
        )
    )

    missing = sorted(
        required - actual
    )

    if missing:
        raise RuntimeError(
            f"{table} missing required columns: "
            f"{missing}"
        )


def csv_value(value):
    if value is None:
        return ""

    return value


# ============================================================
# LOAD HISTORICAL GAMES
# ============================================================

def load_games(
    conn: sqlite3.Connection,
):
    placeholders = ",".join(
        "?" for _ in HISTORICAL_SEASONS
    )

    sql = f"""
        SELECT
            game_id,
            season,
            game_type,
            week,
            game_date,
            weekday,
            gametime,

            away_team,
            home_team,

            away_score,
            home_score,

            away_rest,
            home_rest,

            away_moneyline,
            home_moneyline,

            spread_line,
            total_line,

            roof,
            surface,
            temp,
            wind,

            away_qb_name,
            home_qb_name,

            away_coach,
            home_coach

        FROM games

        WHERE season IN ({placeholders})
          AND completed = 1
          AND away_score IS NOT NULL
          AND home_score IS NOT NULL

        ORDER BY
            game_date,
            COALESCE(gametime, ''),
            game_id
    """

    return conn.execute(
        sql,
        HISTORICAL_SEASONS,
    ).fetchall()


# ============================================================
# LOAD TEAM PREGAME ENVIRONMENT
# ============================================================

def load_team_environment(
    conn: sqlite3.Connection,
):
    rows = conn.execute(
        """
        SELECT *
        FROM team_pregame_environment
        WHERE season BETWEEN 2023 AND 2025
        """
    ).fetchall()

    lookup = {}

    for row in rows:
        key = (
            row["game_id"],
            row["team"],
        )

        if key in lookup:
            raise RuntimeError(
                "Duplicate team_pregame_environment "
                f"row: {key}"
            )

        lookup[key] = dict(row)

    return lookup


# ============================================================
# INJURY / IMPACT PLAYER AGGREGATES
# ============================================================

def load_impact_context(
    conn: sqlite3.Connection,
):
    """
    Initial V1 injury layer.

    This deliberately does NOT assign an arbitrary "star score".

    Instead it records the amount of established PRIOR usage
    missing or at risk because of a player's pregame injury status.

    OUT, DOUBTFUL and QUESTIONABLE remain separate so a later
    model can determine the actual historical effect.
    """

    rows = conn.execute(
        """
        SELECT
            game_id,
            team,

            COUNT(*) AS player_rows,

            SUM(
                CASE
                    WHEN injury_flag = 1
                    THEN 1 ELSE 0
                END
            ) AS injury_flag_count,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN 1 ELSE 0
                END
            ) AS out_count,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'DOUBTFUL'
                    THEN 1 ELSE 0
                END
            ) AS doubtful_count,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'QUESTIONABLE'
                    THEN 1 ELSE 0
                END
            ) AS questionable_count,

            SUM(
                CASE
                    WHEN position = 'QB'
                     AND UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                     ) = 'OUT'
                    THEN 1 ELSE 0
                END
            ) AS qb_out_count,

            SUM(
                CASE
                    WHEN position = 'QB'
                     AND UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                     ) = 'DOUBTFUL'
                    THEN 1 ELSE 0
                END
            ) AS qb_doubtful_count,

            SUM(
                CASE
                    WHEN position = 'QB'
                     AND UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                     ) = 'QUESTIONABLE'
                    THEN 1 ELSE 0
                END
            ) AS qb_questionable_count,

            -- ------------------------------------------------
            -- OUT prior-usage context
            -- ------------------------------------------------

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        fd_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS out_fd_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        snap_pct_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS out_snap_pct_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        opportunities_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS out_opportunities_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        targets_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS out_targets_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        carries_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS out_carries_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'OUT'
                    THEN COALESCE(
                        established_role_flag,
                        0
                    )
                    ELSE 0
                END
            ) AS out_established_role_count,

            -- ------------------------------------------------
            -- DOUBTFUL prior-usage context
            -- ------------------------------------------------

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'DOUBTFUL'
                    THEN COALESCE(
                        fd_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS doubtful_fd_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'DOUBTFUL'
                    THEN COALESCE(
                        snap_pct_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS doubtful_snap_pct_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'DOUBTFUL'
                    THEN COALESCE(
                        opportunities_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS doubtful_opportunities_avg3_sum,

            -- ------------------------------------------------
            -- QUESTIONABLE prior-usage context
            -- ------------------------------------------------

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'QUESTIONABLE'
                    THEN COALESCE(
                        fd_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS questionable_fd_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'QUESTIONABLE'
                    THEN COALESCE(
                        snap_pct_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS questionable_snap_pct_avg3_sum,

            SUM(
                CASE
                    WHEN UPPER(
                        TRIM(
                            COALESCE(
                                report_status,
                                ''
                            )
                        )
                    ) = 'QUESTIONABLE'
                    THEN COALESCE(
                        opportunities_avg_3,
                        0
                    )
                    ELSE 0
                END
            ) AS questionable_opportunities_avg3_sum

        FROM player_pregame_features

        WHERE season BETWEEN 2023 AND 2025

        GROUP BY
            game_id,
            team
        """
    ).fetchall()

    return {
        (
            row["game_id"],
            row["team"],
        ): dict(row)
        for row in rows
    }


# ============================================================
# COACH PRIOR HISTORY
# ============================================================

class CoachHistory:
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.points_for = 0.0
        self.points_against = 0.0

    def snapshot(self):
        if self.games <= 0:
            return {
                "games": 0,
                "win_pct": None,
                "points_for_avg": None,
                "points_against_avg": None,
            }

        return {
            "games": self.games,
            "win_pct": (
                self.wins
                / self.games
            ),
            "points_for_avg": (
                self.points_for
                / self.games
            ),
            "points_against_avg": (
                self.points_against
                / self.games
            ),
        }

    def add_game(
        self,
        points_for,
        points_against,
    ):
        self.games += 1

        self.points_for += (
            float(points_for)
        )

        self.points_against += (
            float(points_against)
        )

        if points_for > points_against:
            self.wins += 1


# ============================================================
# ROW CREATION
# ============================================================

GAME_METADATA_FIELDS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "game_date",
    "weekday",
    "gametime",

    "team",
    "opponent_team",
    "is_home",

    "coach",
    "opponent_coach",

    "coach_prior_games",
    "coach_prior_win_pct",
    "coach_prior_points_for_avg",
    "coach_prior_points_against_avg",

    "opponent_coach_prior_games",
    "opponent_coach_prior_win_pct",
    "opponent_coach_prior_points_for_avg",
    "opponent_coach_prior_points_against_avg",

    "team_qb_name",
    "opponent_qb_name",

    "team_rest",
    "opponent_rest",

    # Raw market variables.
    # spread_line semantics are intentionally NOT transformed yet.
    "market_home_spread_raw",
    "market_total",
    "team_moneyline",
    "opponent_moneyline",

    "roof",
    "surface",
    "temp",
    "wind",
]


IMPACT_FIELDS = [
    "player_rows",
    "injury_flag_count",

    "out_count",
    "doubtful_count",
    "questionable_count",

    "qb_out_count",
    "qb_doubtful_count",
    "qb_questionable_count",

    "out_fd_avg3_sum",
    "out_snap_pct_avg3_sum",
    "out_opportunities_avg3_sum",
    "out_targets_avg3_sum",
    "out_carries_avg3_sum",
    "out_established_role_count",

    "doubtful_fd_avg3_sum",
    "doubtful_snap_pct_avg3_sum",
    "doubtful_opportunities_avg3_sum",

    "questionable_fd_avg3_sum",
    "questionable_snap_pct_avg3_sum",
    "questionable_opportunities_avg3_sum",
]


TARGET_FIELDS = [
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
]


def empty_impact():
    return {
        field: 0
        for field in IMPACT_FIELDS
    }


def build_dataset(
    conn: sqlite3.Connection,
):
    games = load_games(conn)

    env_lookup = (
        load_team_environment(conn)
    )

    impact_lookup = (
        load_impact_context(conn)
    )

    env_columns = table_columns(
        conn,
        "team_pregame_environment",
    )

    env_feature_columns = [
        col
        for col in env_columns
        if col not in {
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
            "updated_at",
        }
    ]

    coach_history = defaultdict(
        CoachHistory
    )

    output = []

    missing_team_env = 0
    missing_opponent_env = 0

    for game in games:
        game = dict(game)

        away_team = game["away_team"]
        home_team = game["home_team"]

        away_score = safe_float(
            game["away_score"]
        )

        home_score = safe_float(
            game["home_score"]
        )

        if (
            away_score is None
            or home_score is None
        ):
            raise RuntimeError(
                "Completed historical game "
                f"missing score: {game['game_id']}"
            )

        away_coach = normalize_text(
            game["away_coach"]
        )

        home_coach = normalize_text(
            game["home_coach"]
        )

        # ----------------------------------------------------
        # Snapshot coach history BEFORE current game update.
        # ----------------------------------------------------

        away_coach_prior = (
            coach_history[
                away_coach
            ].snapshot()
            if away_coach
            else {
                "games": 0,
                "win_pct": None,
                "points_for_avg": None,
                "points_against_avg": None,
            }
        )

        home_coach_prior = (
            coach_history[
                home_coach
            ].snapshot()
            if home_coach
            else {
                "games": 0,
                "win_pct": None,
                "points_for_avg": None,
                "points_against_avg": None,
            }
        )

        side_specs = [
            {
                "team": away_team,
                "opponent": home_team,
                "is_home": 0,

                "coach": away_coach,
                "opp_coach": home_coach,

                "coach_prior":
                    away_coach_prior,
                "opp_coach_prior":
                    home_coach_prior,

                "qb":
                    game["away_qb_name"],
                "opp_qb":
                    game["home_qb_name"],

                "rest":
                    game["away_rest"],
                "opp_rest":
                    game["home_rest"],

                "team_ml":
                    game["away_moneyline"],
                "opp_ml":
                    game["home_moneyline"],

                "team_points":
                    away_score,
                "opp_points":
                    home_score,
            },
            {
                "team": home_team,
                "opponent": away_team,
                "is_home": 1,

                "coach": home_coach,
                "opp_coach": away_coach,

                "coach_prior":
                    home_coach_prior,
                "opp_coach_prior":
                    away_coach_prior,

                "qb":
                    game["home_qb_name"],
                "opp_qb":
                    game["away_qb_name"],

                "rest":
                    game["home_rest"],
                "opp_rest":
                    game["away_rest"],

                "team_ml":
                    game["home_moneyline"],
                "opp_ml":
                    game["away_moneyline"],

                "team_points":
                    home_score,
                "opp_points":
                    away_score,
            },
        ]

        for side in side_specs:
            team = side["team"]
            opponent = side["opponent"]

            row = {
                "game_id":
                    game["game_id"],

                "season":
                    game["season"],

                "game_type":
                    game["game_type"],

                "week":
                    game["week"],

                "game_date":
                    game["game_date"],

                "weekday":
                    game["weekday"],

                "gametime":
                    game["gametime"],

                "team":
                    team,

                "opponent_team":
                    opponent,

                "is_home":
                    side["is_home"],

                "coach":
                    side["coach"],

                "opponent_coach":
                    side["opp_coach"],

                "coach_prior_games":
                    side[
                        "coach_prior"
                    ]["games"],

                "coach_prior_win_pct":
                    side[
                        "coach_prior"
                    ]["win_pct"],

                "coach_prior_points_for_avg":
                    side[
                        "coach_prior"
                    ]["points_for_avg"],

                "coach_prior_points_against_avg":
                    side[
                        "coach_prior"
                    ]["points_against_avg"],

                "opponent_coach_prior_games":
                    side[
                        "opp_coach_prior"
                    ]["games"],

                "opponent_coach_prior_win_pct":
                    side[
                        "opp_coach_prior"
                    ]["win_pct"],

                "opponent_coach_prior_points_for_avg":
                    side[
                        "opp_coach_prior"
                    ]["points_for_avg"],

                "opponent_coach_prior_points_against_avg":
                    side[
                        "opp_coach_prior"
                    ]["points_against_avg"],

                "team_qb_name":
                    normalize_text(
                        side["qb"]
                    ),

                "opponent_qb_name":
                    normalize_text(
                        side["opp_qb"]
                    ),

                "team_rest":
                    safe_float(
                        side["rest"]
                    ),

                "opponent_rest":
                    safe_float(
                        side["opp_rest"]
                    ),

                "market_home_spread_raw":
                    safe_float(
                        game["spread_line"]
                    ),

                "market_total":
                    safe_float(
                        game["total_line"]
                    ),

                "team_moneyline":
                    safe_float(
                        side["team_ml"]
                    ),

                "opponent_moneyline":
                    safe_float(
                        side["opp_ml"]
                    ),

                "roof":
                    normalize_text(
                        game["roof"]
                    ),

                "surface":
                    normalize_text(
                        game["surface"]
                    ),

                "temp":
                    safe_float(
                        game["temp"]
                    ),

                "wind":
                    safe_float(
                        game["wind"]
                    ),
            }

            # ------------------------------------------------
            # Team pregame environment
            # ------------------------------------------------

            team_env = env_lookup.get(
                (
                    game["game_id"],
                    team,
                )
            )

            if team_env is None:
                missing_team_env += 1

                for column in (
                    env_feature_columns
                ):
                    row[
                        f"team_{column}"
                    ] = None
            else:
                for column in (
                    env_feature_columns
                ):
                    row[
                        f"team_{column}"
                    ] = team_env.get(
                        column
                    )

            # ------------------------------------------------
            # Opponent pregame environment
            # ------------------------------------------------

            opp_env = env_lookup.get(
                (
                    game["game_id"],
                    opponent,
                )
            )

            if opp_env is None:
                missing_opponent_env += 1

                for column in (
                    env_feature_columns
                ):
                    row[
                        f"opp_{column}"
                    ] = None
            else:
                for column in (
                    env_feature_columns
                ):
                    row[
                        f"opp_{column}"
                    ] = opp_env.get(
                        column
                    )

            # ------------------------------------------------
            # Team injury / impact context
            # ------------------------------------------------

            team_impact = dict(
                empty_impact()
            )

            team_impact.update(
                impact_lookup.get(
                    (
                        game["game_id"],
                        team,
                    ),
                    {},
                )
            )

            for field in IMPACT_FIELDS:
                row[
                    f"team_{field}"
                ] = team_impact.get(
                    field,
                    0,
                )

            # ------------------------------------------------
            # Opponent injury / impact context
            # ------------------------------------------------

            opp_impact = dict(
                empty_impact()
            )

            opp_impact.update(
                impact_lookup.get(
                    (
                        game["game_id"],
                        opponent,
                    ),
                    {},
                )
            )

            for field in IMPACT_FIELDS:
                row[
                    f"opp_{field}"
                ] = opp_impact.get(
                    field,
                    0,
                )

            # ------------------------------------------------
            # Targets — CURRENT GAME OUTCOME
            #
            # These are targets only and must never become
            # Forecast V1 predictor inputs.
            # ------------------------------------------------

            team_points = side[
                "team_points"
            ]

            opp_points = side[
                "opp_points"
            ]

            margin = (
                team_points
                - opp_points
            )

            total_points = (
                team_points
                + opp_points
            )

            if team_points > opp_points:
                target_result = "WIN"
                target_win = 1

            elif team_points < opp_points:
                target_result = "LOSS"
                target_win = 0

            else:
                target_result = "TIE"
                target_win = None

            row.update(
                {
                    "target_team_points":
                        team_points,

                    "target_opponent_points":
                        opp_points,

                    "target_margin":
                        margin,

                    "target_total_points":
                        total_points,

                    "target_win":
                        target_win,

                    "target_result":
                        target_result,
                }
            )

            output.append(row)

        # ----------------------------------------------------
        # Update coach histories only AFTER both sides'
        # pregame snapshots were created.
        # ----------------------------------------------------

        if away_coach:
            coach_history[
                away_coach
            ].add_game(
                away_score,
                home_score,
            )

        if home_coach:
            coach_history[
                home_coach
            ].add_game(
                home_score,
                away_score,
            )

    return (
        output,
        env_feature_columns,
        missing_team_env,
        missing_opponent_env,
    )


# ============================================================
# AUDIT
# ============================================================

def audit_dataset(
    rows,
    missing_team_env,
    missing_opponent_env,
):
    row_count = len(rows)

    unique_games = {
        row["game_id"]
        for row in rows
    }

    keys = [
        (
            row["game_id"],
            row["team"],
        )
        for row in rows
    ]

    duplicate_count = (
        len(keys)
        - len(set(keys))
    )

    game_counts = defaultdict(int)

    for row in rows:
        game_counts[
            row["game_id"]
        ] += 1

    bad_game_row_counts = {
        game_id: count
        for game_id, count
        in game_counts.items()
        if count != 2
    }

    missing_targets = sum(
        1
        for row in rows
        if (
            row[
                "target_team_points"
            ] is None
            or row[
                "target_opponent_points"
            ] is None
        )
    )

    missing_spread = sum(
        1
        for row in rows
        if row[
            "market_home_spread_raw"
        ] is None
    )

    missing_total = sum(
        1
        for row in rows
        if row["market_total"] is None
    )

    missing_temp = sum(
        1
        for row in rows
        if row["temp"] is None
    )

    missing_wind = sum(
        1
        for row in rows
        if row["wind"] is None
    )

    coach_zero_history = sum(
        1
        for row in rows
        if row[
            "coach_prior_games"
        ] == 0
    )

    qb_out_rows = sum(
        1
        for row in rows
        if (
            safe_int(
                row.get(
                    "team_qb_out_count"
                )
            )
            or 0
        ) > 0
    )

    out_rows = sum(
        1
        for row in rows
        if (
            safe_int(
                row.get(
                    "team_out_count"
                )
            )
            or 0
        ) > 0
    )

    doubtful_rows = sum(
        1
        for row in rows
        if (
            safe_int(
                row.get(
                    "team_doubtful_count"
                )
            )
            or 0
        ) > 0
    )

    questionable_rows = sum(
        1
        for row in rows
        if (
            safe_int(
                row.get(
                    "team_questionable_count"
                )
            )
            or 0
        ) > 0
    )

    critical_checks = {
        "row_count":
            row_count == EXPECTED_ROWS,

        "unique_games":
            len(unique_games)
            == EXPECTED_GAMES,

        "duplicate_team_game_keys":
            duplicate_count == 0,

        "exactly_two_rows_per_game":
            not bad_game_row_counts,

        "team_environment_complete":
            missing_team_env == 0,

        "opponent_environment_complete":
            missing_opponent_env == 0,

        "targets_complete":
            missing_targets == 0,

        "market_spread_complete":
            missing_spread == 0,

        "market_total_complete":
            missing_total == 0,
    }

    overall_pass = all(
        critical_checks.values()
    )

    return {
        "created_at_utc":
            now_utc(),

        "database":
            str(DB_PATH),

        "historical_seasons":
            list(
                HISTORICAL_SEASONS
            ),

        "expected_games":
            EXPECTED_GAMES,

        "expected_rows":
            EXPECTED_ROWS,

        "actual_games":
            len(unique_games),

        "actual_rows":
            row_count,

        "duplicate_team_game_keys":
            duplicate_count,

        "bad_game_row_counts":
            bad_game_row_counts,

        "missing_team_environment_rows":
            missing_team_env,

        "missing_opponent_environment_rows":
            missing_opponent_env,

        "missing_target_rows":
            missing_targets,

        "missing_market_spread_rows":
            missing_spread,

        "missing_market_total_rows":
            missing_total,

        "missing_temperature_rows":
            missing_temp,

        "missing_wind_rows":
            missing_wind,

        "coach_zero_prior_history_rows":
            coach_zero_history,

        "team_games_with_out_players":
            out_rows,

        "team_games_with_doubtful_players":
            doubtful_rows,

        "team_games_with_questionable_players":
            questionable_rows,

        "team_games_with_qb_out":
            qb_out_rows,

        "critical_checks":
            critical_checks,

        "overall_pass":
            overall_pass,

        "notes": [
            (
                "Weather missingness is allowed "
                "and must be handled later."
            ),
            (
                "Coach cold-start history is allowed."
            ),
            (
                "OUT, DOUBTFUL and QUESTIONABLE "
                "remain separate."
            ),
            (
                "No arbitrary star-player weight "
                "has been introduced."
            ),
            (
                "Impact variables measure prior "
                "usage associated with unavailable "
                "or at-risk players."
            ),
            (
                "Defensive individual-player "
                "impact is not yet fully modeled."
            ),
            (
                "No model training occurs here."
            ),
            (
                "No SQLite writes occur here."
            ),
        ],
    }


# ============================================================
# EXPORT
# ============================================================

def export_dataset(
    rows,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No dataset rows produced."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with DATASET_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: csv_value(
                        row.get(key)
                    )
                    for key in fieldnames
                }
            )


def export_audit(
    audit,
):
    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            audit,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 88)
    print(
        "WFS NFL FORECAST CENTER — "
        "V1 HISTORICAL DATASET BUILDER"
    )
    print("=" * 88)

    print("Database:")
    print(DB_PATH)

    print()
    print(
        "MODE: READ-ONLY SQLITE / "
        "CSV+JSON OUTPUT ONLY"
    )

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Database not found: {DB_PATH}"
        )

    uri = (
        f"file:{DB_PATH}"
        "?mode=ro"
    )

    conn = sqlite3.connect(
        uri,
        uri=True,
    )

    conn.row_factory = sqlite3.Row

    try:
        print()
        print("=" * 88)
        print("1. DATABASE SAFETY")
        print("=" * 88)

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        print(
            "Integrity:",
            integrity,
        )

        print(
            "Foreign keys:",
            (
                "PASS"
                if not fk
                else f"FAIL ({len(fk)})"
            ),
        )

        if integrity != "ok":
            raise RuntimeError(
                "Database integrity check failed."
            )

        if fk:
            raise RuntimeError(
                "Foreign-key check failed."
            )

        required_tables = [
            "games",
            "team_pregame_environment",
            "player_pregame_features",
        ]

        for table in required_tables:
            require_table(
                conn,
                table,
            )

        require_columns(
            conn,
            "games",
            {
                "game_id",
                "season",
                "game_type",
                "week",
                "game_date",
                "gametime",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
                "away_rest",
                "home_rest",
                "away_moneyline",
                "home_moneyline",
                "spread_line",
                "total_line",
                "roof",
                "surface",
                "temp",
                "wind",
                "away_qb_name",
                "home_qb_name",
                "away_coach",
                "home_coach",
                "completed",
            },
        )

        require_columns(
            conn,
            "player_pregame_features",
            {
                "game_id",
                "season",
                "team",
                "position",
                "report_status",
                "injury_flag",
                "fd_avg_3",
                "snap_pct_avg_3",
                "opportunities_avg_3",
                "targets_avg_3",
                "carries_avg_3",
                "established_role_flag",
            },
        )

        print(
            "Required tables/columns: PASS"
        )

        print()
        print("=" * 88)
        print("2. BUILDING DATASET")
        print("=" * 88)

        (
            rows,
            env_features,
            missing_team_env,
            missing_opponent_env,
        ) = build_dataset(conn)

        print(
            "Rows built:",
            len(rows),
        )

        print(
            "Team-environment features:",
            len(env_features),
        )

        print()
        print("=" * 88)
        print("3. AUDIT")
        print("=" * 88)

        audit = audit_dataset(
            rows,
            missing_team_env,
            missing_opponent_env,
        )

        for name, passed in (
            audit[
                "critical_checks"
            ].items()
        ):
            print(
                f"{'PASS' if passed else 'FAIL':8s}"
                f" | {name}"
            )

        print("-" * 88)

        print(
            "Games:",
            audit["actual_games"],
        )

        print(
            "Rows:",
            audit["actual_rows"],
        )

        print(
            "Team-games with OUT players:",
            audit[
                "team_games_with_out_players"
            ],
        )

        print(
            "Team-games with DOUBTFUL players:",
            audit[
                "team_games_with_doubtful_players"
            ],
        )

        print(
            "Team-games with QUESTIONABLE players:",
            audit[
                "team_games_with_questionable_players"
            ],
        )

        print(
            "Team-games with QB OUT:",
            audit[
                "team_games_with_qb_out"
            ],
        )

        print(
            "Missing temperature rows:",
            audit[
                "missing_temperature_rows"
            ],
        )

        print(
            "Missing wind rows:",
            audit[
                "missing_wind_rows"
            ],
        )

        print(
            "Coach zero-prior-history rows:",
            audit[
                "coach_zero_prior_history_rows"
            ],
        )

        print()
        print("=" * 88)
        print("4. EXPORT")
        print("=" * 88)

        export_dataset(
            rows
        )

        export_audit(
            audit
        )

        print(
            "Dataset:"
        )
        print(
            DATASET_PATH
        )

        print()
        print(
            "Audit:"
        )
        print(
            AUDIT_PATH
        )

        print()
        print("=" * 88)

        if audit["overall_pass"]:
            print(
                "FORECAST V1 DATASET: PASS"
            )
        else:
            print(
                "FORECAST V1 DATASET: FAIL"
            )

        print("=" * 88)

        print()
        print(
            "NO MACHINE-LEARNING MODEL TRAINED."
        )

        print(
            "NO SQLITE TABLES MODIFIED."
        )

        if not audit["overall_pass"]:
            raise RuntimeError(
                "Forecast dataset audit failed."
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
