from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection


# =========================================================
# OUTPUTS
# =========================================================

TEAM_ENV_CSV = (
    CSV_DIR / "nfl_team_pregame_environment.csv"
)

TEAM_ENV_PARQUET = (
    PARQUET_DIR / "nfl_team_pregame_environment.parquet"
)

DVP_CSV = (
    CSV_DIR / "nfl_team_position_dvp.csv"
)

DVP_PARQUET = (
    PARQUET_DIR / "nfl_team_position_dvp.parquet"
)


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def now_utc():

    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)


def clean_numeric(series):

    return (
        series
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
    )


# =========================================================
# DATABASE TABLES
# =========================================================

def rebuild_tables():

    section(
        "REBUILDING TEAM ENVIRONMENT TABLES"
    )

    with get_connection() as conn:

        conn.execute(
            """
            DROP TABLE IF EXISTS
            team_pregame_environment
            """
        )

        conn.execute(
            """
            DROP TABLE IF EXISTS
            team_position_dvp
            """
        )

        conn.execute(
            """
            CREATE TABLE
            team_pregame_environment (

                game_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,

                team TEXT NOT NULL,
                opponent_team TEXT NOT NULL,

                history_games INTEGER,

                points_for_last REAL,
                points_for_avg_3 REAL,
                points_for_avg_5 REAL,

                points_against_last REAL,
                points_against_avg_3 REAL,
                points_against_avg_5 REAL,

                offensive_plays_avg_3 REAL,
                offensive_plays_avg_5 REAL,

                pass_attempts_avg_3 REAL,
                pass_attempts_avg_5 REAL,

                rush_attempts_avg_3 REAL,
                rush_attempts_avg_5 REAL,

                pass_rate_avg_3 REAL,
                pass_rate_avg_5 REAL,

                rush_rate_avg_3 REAL,
                rush_rate_avg_5 REAL,

                passing_yards_avg_3 REAL,
                rushing_yards_avg_3 REAL,

                passing_tds_avg_3 REAL,
                rushing_tds_avg_3 REAL,

                opponent_points_allowed_avg_3 REAL,
                opponent_points_allowed_avg_5 REAL,

                opponent_pass_yards_allowed_avg_3 REAL,
                opponent_rush_yards_allowed_avg_3 REAL,

                opponent_pass_tds_allowed_avg_3 REAL,
                opponent_rush_tds_allowed_avg_3 REAL,

                team_scoring_trend REAL,
                opponent_scoring_trend REAL,

                pace_trend REAL,
                pass_rate_trend REAL,

                updated_at TEXT,

                PRIMARY KEY (
                    game_id,
                    team
                )
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE
            team_position_dvp (

                game_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,

                defense_team TEXT NOT NULL,
                position TEXT NOT NULL,

                history_games INTEGER,

                fd_allowed_last REAL,
                fd_allowed_avg_3 REAL,
                fd_allowed_avg_5 REAL,

                targets_allowed_avg_3 REAL,
                carries_allowed_avg_3 REAL,

                receptions_allowed_avg_3 REAL,

                receiving_yards_allowed_avg_3 REAL,
                rushing_yards_allowed_avg_3 REAL,

                receiving_tds_allowed_avg_3 REAL,
                rushing_tds_allowed_avg_3 REAL,

                opportunities_allowed_avg_3 REAL,

                fd_allowed_trend REAL,
                opportunity_allowed_trend REAL,

                updated_at TEXT,

                PRIMARY KEY (
                    game_id,
                    defense_team,
                    position
                )
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_team_env_team

            ON team_pregame_environment(
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_team_env_game

            ON team_pregame_environment(
                game_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_team_env_week

            ON team_pregame_environment(
                season,
                week
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_dvp_team_position

            ON team_position_dvp(
                defense_team,
                position
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_dvp_game

            ON team_position_dvp(
                game_id
            )
            """
        )

    print(
        "Fresh team environment tables created."
    )


# =========================================================
# LOAD TEAM GAME STATS
# =========================================================

def load_team_stats():

    section(
        "LOADING TEAM GAME STATS"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM team_game_stats

            WHERE
                game_id IS NOT NULL
            """,
            conn,
        )

    print(
        f"Team-game rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# LOAD PLAYER USAGE
# =========================================================

def load_player_usage():

    section(
        "LOADING PLAYER USAGE FOR DVP"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,

                team,
                opponent_team,

                position,

                fanduel_points,

                carries,
                targets,
                receptions,

                rushing_yards,
                receiving_yards,

                rushing_tds,
                receiving_tds,

                opportunities

            FROM player_weekly_usage
            """,
            conn,
        )

    print(
        f"Player usage rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# NORMALIZE TEAM STATS
# =========================================================

def prepare_team_stats(
    df,
):

    section(
        "PREPARING TEAM GAME METRICS"
    )

    # -----------------------------------------------------
    # REQUIRED COLUMN DISCOVERY
    #
    # nflverse team stats may expose slightly different
    # naming over time. We explicitly map known fields.
    # -----------------------------------------------------

    rename_map = {}

    candidates = {
        "attempts": [
            "attempts",
            "passing_attempts",
        ],

        "passing_yards": [
            "passing_yards",
        ],

        "passing_tds": [
            "passing_tds",
        ],

        "carries": [
            "carries",
            "rushing_attempts",
        ],

        "rushing_yards": [
            "rushing_yards",
        ],

        "rushing_tds": [
            "rushing_tds",
        ],
    }

    for target, possible_columns in candidates.items():

        for column in possible_columns:

            if column in df.columns:

                rename_map[
                    column
                ] = target

                break

    df = df.rename(
        columns=rename_map
    )

    required = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Required team stat column "
                f"missing: {column}"
            )

    numeric_defaults = [
        "attempts",
        "passing_yards",
        "passing_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ]

    for column in numeric_defaults:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = safe_numeric(
            df[column]
        )

    # -----------------------------------------------------
    # SCORE FROM GAMES TABLE
    # -----------------------------------------------------

    with get_connection() as conn:

        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                home_team,
                away_team,
                home_score,
                away_score

            FROM games

            WHERE
                home_score IS NOT NULL
                AND
                away_score IS NOT NULL
            """,
            conn,
        )

    home = games[
        [
            "game_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent_team",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )

    away = games[
        [
            "game_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent_team",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )

    score_df = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    df = df.merge(
        score_df,
        how="left",
        on=[
            "game_id",
            "team",
            "opponent_team",
        ],
        validate="one_to_one",
    )

    if df[
        "points_for"
    ].isna().any():

        missing = int(
            df[
                "points_for"
            ].isna().sum()
        )

        raise RuntimeError(
            f"{missing} team-game rows "
            f"failed score attachment."
        )

    df[
        "points_for"
    ] = safe_numeric(
        df["points_for"]
    )

    df[
        "points_against"
    ] = safe_numeric(
        df["points_against"]
    )

    df[
        "offensive_plays"
    ] = (
        df["attempts"]
        +
        df["carries"]
    )

    denominator = (
        df[
            "offensive_plays"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    df[
        "pass_rate"
    ] = (
        df["attempts"]
        /
        denominator
    ).fillna(0.0)

    df[
        "rush_rate"
    ] = (
        df["carries"]
        /
        denominator
    ).fillna(0.0)

    print(
        "Team game metrics prepared."
    )

    return df


# =========================================================
# ROLLING HELPERS
# =========================================================

def lagged_last(
    group,
    column,
):

    return group[
        column
    ].shift(1)


def lagged_mean(
    group,
    column,
    window,
):

    return group[
        column
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                window,
                min_periods=1,
            )
            .mean()
    )


# =========================================================
# BUILD TEAM PREGAME ENVIRONMENT
# =========================================================

def build_team_environment(
    df,
):

    section(
        "BUILDING TEAM PREGAME ENVIRONMENT"
    )

    df = (
        df.sort_values(
            [
                "team",
                "season",
                "week",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    group = df.groupby(
        "team",
        group_keys=False,
    )

    df[
        "history_games"
    ] = group.cumcount()

    # -----------------------------------------------------
    # OFFENSE HISTORY
    # -----------------------------------------------------

    df[
        "points_for_last"
    ] = lagged_last(
        group,
        "points_for",
    )

    df[
        "points_for_avg_3"
    ] = lagged_mean(
        group,
        "points_for",
        3,
    )

    df[
        "points_for_avg_5"
    ] = lagged_mean(
        group,
        "points_for",
        5,
    )

    df[
        "points_against_last"
    ] = lagged_last(
        group,
        "points_against",
    )

    df[
        "points_against_avg_3"
    ] = lagged_mean(
        group,
        "points_against",
        3,
    )

    df[
        "points_against_avg_5"
    ] = lagged_mean(
        group,
        "points_against",
        5,
    )

    df[
        "offensive_plays_avg_3"
    ] = lagged_mean(
        group,
        "offensive_plays",
        3,
    )

    df[
        "offensive_plays_avg_5"
    ] = lagged_mean(
        group,
        "offensive_plays",
        5,
    )

    df[
        "pass_attempts_avg_3"
    ] = lagged_mean(
        group,
        "attempts",
        3,
    )

    df[
        "pass_attempts_avg_5"
    ] = lagged_mean(
        group,
        "attempts",
        5,
    )

    df[
        "rush_attempts_avg_3"
    ] = lagged_mean(
        group,
        "carries",
        3,
    )

    df[
        "rush_attempts_avg_5"
    ] = lagged_mean(
        group,
        "carries",
        5,
    )

    df[
        "pass_rate_avg_3"
    ] = lagged_mean(
        group,
        "pass_rate",
        3,
    )

    df[
        "pass_rate_avg_5"
    ] = lagged_mean(
        group,
        "pass_rate",
        5,
    )

    df[
        "rush_rate_avg_3"
    ] = lagged_mean(
        group,
        "rush_rate",
        3,
    )

    df[
        "rush_rate_avg_5"
    ] = lagged_mean(
        group,
        "rush_rate",
        5,
    )

    df[
        "passing_yards_avg_3"
    ] = lagged_mean(
        group,
        "passing_yards",
        3,
    )

    df[
        "rushing_yards_avg_3"
    ] = lagged_mean(
        group,
        "rushing_yards",
        3,
    )

    df[
        "passing_tds_avg_3"
    ] = lagged_mean(
        group,
        "passing_tds",
        3,
    )

    df[
        "rushing_tds_avg_3"
    ] = lagged_mean(
        group,
        "rushing_tds",
        3,
    )

    # -----------------------------------------------------
    # DEFENSIVE HISTORY FOR EACH TEAM
    #
    # points_against / opponent production
    # -----------------------------------------------------

    defense = df[
        [
            "game_id",
            "season",
            "week",
            "team",
            "points_against",
        ]
    ].copy()

    # -----------------------------------------------------
    # Opponent yards / TDs are simply the opposing team's
    # offense from the same game.
    # -----------------------------------------------------

    opponent_offense = df[
        [
            "game_id",
            "team",
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
        ]
    ].copy()

    opponent_offense = opponent_offense.rename(
        columns={
            "team": "opponent_team",

            "passing_yards":
                "opponent_passing_yards",

            "rushing_yards":
                "opponent_rushing_yards",

            "passing_tds":
                "opponent_passing_tds",

            "rushing_tds":
                "opponent_rushing_tds",
        }
    )

    df = df.merge(
        opponent_offense,
        how="left",
        on=[
            "game_id",
            "opponent_team",
        ],
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Now compute each defense's historical allowed stats.
    # -----------------------------------------------------

    defense_source = df[
        [
            "game_id",
            "season",
            "week",
            "team",
            "points_against",

            "opponent_passing_yards",
            "opponent_rushing_yards",

            "opponent_passing_tds",
            "opponent_rushing_tds",
        ]
    ].copy()

    defense_source = (
        defense_source
        .sort_values(
            [
                "team",
                "season",
                "week",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    defense_group = (
        defense_source.groupby(
            "team",
            group_keys=False,
        )
    )

    defense_source[
        "opponent_points_allowed_avg_3"
    ] = lagged_mean(
        defense_group,
        "points_against",
        3,
    )

    defense_source[
        "opponent_points_allowed_avg_5"
    ] = lagged_mean(
        defense_group,
        "points_against",
        5,
    )

    defense_source[
        "opponent_pass_yards_allowed_avg_3"
    ] = lagged_mean(
        defense_group,
        "opponent_passing_yards",
        3,
    )

    defense_source[
        "opponent_rush_yards_allowed_avg_3"
    ] = lagged_mean(
        defense_group,
        "opponent_rushing_yards",
        3,
    )

    defense_source[
        "opponent_pass_tds_allowed_avg_3"
    ] = lagged_mean(
        defense_group,
        "opponent_passing_tds",
        3,
    )

    defense_source[
        "opponent_rush_tds_allowed_avg_3"
    ] = lagged_mean(
        defense_group,
        "opponent_rushing_tds",
        3,
    )

    opponent_defense = defense_source[
        [
            "game_id",
            "team",

            "opponent_points_allowed_avg_3",
            "opponent_points_allowed_avg_5",

            "opponent_pass_yards_allowed_avg_3",
            "opponent_rush_yards_allowed_avg_3",

            "opponent_pass_tds_allowed_avg_3",
            "opponent_rush_tds_allowed_avg_3",
        ]
    ].copy()

    opponent_defense = opponent_defense.rename(
        columns={
            "team": "opponent_team",
        }
    )

    df = df.merge(
        opponent_defense,
        how="left",
        on=[
            "game_id",
            "opponent_team",
        ],
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # TRENDS
    # -----------------------------------------------------

    df[
        "team_scoring_trend"
    ] = (
        df["points_for_avg_3"]
        -
        df["points_for_avg_5"]
    )

    df[
        "opponent_scoring_trend"
    ] = (
        df[
            "opponent_points_allowed_avg_3"
        ]
        -
        df[
            "opponent_points_allowed_avg_5"
        ]
    )

    df[
        "pace_trend"
    ] = (
        df["offensive_plays_avg_3"]
        -
        df["offensive_plays_avg_5"]
    )

    df[
        "pass_rate_trend"
    ] = (
        df["pass_rate_avg_3"]
        -
        df["pass_rate_avg_5"]
    )

    numeric_output = [
        "points_for_last",
        "points_for_avg_3",
        "points_for_avg_5",

        "points_against_last",
        "points_against_avg_3",
        "points_against_avg_5",

        "offensive_plays_avg_3",
        "offensive_plays_avg_5",

        "pass_attempts_avg_3",
        "pass_attempts_avg_5",

        "rush_attempts_avg_3",
        "rush_attempts_avg_5",

        "pass_rate_avg_3",
        "pass_rate_avg_5",

        "rush_rate_avg_3",
        "rush_rate_avg_5",

        "passing_yards_avg_3",
        "rushing_yards_avg_3",

        "passing_tds_avg_3",
        "rushing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_points_allowed_avg_5",

        "opponent_pass_yards_allowed_avg_3",
        "opponent_rush_yards_allowed_avg_3",

        "opponent_pass_tds_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",

        "team_scoring_trend",
        "opponent_scoring_trend",

        "pace_trend",
        "pass_rate_trend",
    ]

    for column in numeric_output:

        df[column] = clean_numeric(
            df[column]
        )

    print(
        "Team pregame environment calculated."
    )

    return df


# =========================================================
# BUILD POSITION DVP
# =========================================================

def build_position_dvp(
    player_df,
):

    section(
        "BUILDING POSITION DVP"
    )

    skill_positions = {
        "QB",
        "RB",
        "WR",
        "TE",
    }

    player_df = player_df[
        player_df[
            "position"
        ].isin(
            skill_positions
        )
    ].copy()

    # -----------------------------------------------------
    # Aggregate offensive production against each defense
    # by game and position.
    # -----------------------------------------------------

    game_dvp = (
        player_df.groupby(
            [
                "game_id",
                "season",
                "week",
                "opponent_team",
                "position",
            ],
            as_index=False,
        )
        .agg(
            fanduel_points=(
                "fanduel_points",
                "sum",
            ),

            targets=(
                "targets",
                "sum",
            ),

            carries=(
                "carries",
                "sum",
            ),

            receptions=(
                "receptions",
                "sum",
            ),

            receiving_yards=(
                "receiving_yards",
                "sum",
            ),

            rushing_yards=(
                "rushing_yards",
                "sum",
            ),

            receiving_tds=(
                "receiving_tds",
                "sum",
            ),

            rushing_tds=(
                "rushing_tds",
                "sum",
            ),

            opportunities=(
                "opportunities",
                "sum",
            ),
        )
    )

    game_dvp = game_dvp.rename(
        columns={
            "opponent_team":
                "defense_team",
        }
    )

    game_dvp = (
        game_dvp
        .sort_values(
            [
                "defense_team",
                "position",
                "season",
                "week",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    group = game_dvp.groupby(
        [
            "defense_team",
            "position",
        ],
        group_keys=False,
    )

    game_dvp[
        "history_games"
    ] = group.cumcount()

    game_dvp[
        "fd_allowed_last"
    ] = lagged_last(
        group,
        "fanduel_points",
    )

    game_dvp[
        "fd_allowed_avg_3"
    ] = lagged_mean(
        group,
        "fanduel_points",
        3,
    )

    game_dvp[
        "fd_allowed_avg_5"
    ] = lagged_mean(
        group,
        "fanduel_points",
        5,
    )

    game_dvp[
        "targets_allowed_avg_3"
    ] = lagged_mean(
        group,
        "targets",
        3,
    )

    game_dvp[
        "carries_allowed_avg_3"
    ] = lagged_mean(
        group,
        "carries",
        3,
    )

    game_dvp[
        "receptions_allowed_avg_3"
    ] = lagged_mean(
        group,
        "receptions",
        3,
    )

    game_dvp[
        "receiving_yards_allowed_avg_3"
    ] = lagged_mean(
        group,
        "receiving_yards",
        3,
    )

    game_dvp[
        "rushing_yards_allowed_avg_3"
    ] = lagged_mean(
        group,
        "rushing_yards",
        3,
    )

    game_dvp[
        "receiving_tds_allowed_avg_3"
    ] = lagged_mean(
        group,
        "receiving_tds",
        3,
    )

    game_dvp[
        "rushing_tds_allowed_avg_3"
    ] = lagged_mean(
        group,
        "rushing_tds",
        3,
    )

    game_dvp[
        "opportunities_allowed_avg_3"
    ] = lagged_mean(
        group,
        "opportunities",
        3,
    )

    game_dvp[
        "fd_allowed_trend"
    ] = (
        game_dvp[
            "fd_allowed_avg_3"
        ]
        -
        game_dvp[
            "fd_allowed_avg_5"
        ]
    )

    opportunities_avg_5 = (
        group[
            "opportunities"
        ].transform(
            lambda s:
                s.shift(1)
                .rolling(
                    5,
                    min_periods=1,
                )
                .mean()
        )
    )

    game_dvp[
        "opportunity_allowed_trend"
    ] = (
        game_dvp[
            "opportunities_allowed_avg_3"
        ]
        -
        opportunities_avg_5
    )

    numeric_columns = [
        "fd_allowed_last",
        "fd_allowed_avg_3",
        "fd_allowed_avg_5",

        "targets_allowed_avg_3",
        "carries_allowed_avg_3",

        "receptions_allowed_avg_3",

        "receiving_yards_allowed_avg_3",
        "rushing_yards_allowed_avg_3",

        "receiving_tds_allowed_avg_3",
        "rushing_tds_allowed_avg_3",

        "opportunities_allowed_avg_3",

        "fd_allowed_trend",
        "opportunity_allowed_trend",
    ]

    for column in numeric_columns:

        game_dvp[column] = clean_numeric(
            game_dvp[column]
        )

    print(
        f"Position DvP rows built: "
        f"{len(game_dvp)}"
    )

    return game_dvp


# =========================================================
# STORE TEAM ENVIRONMENT
# =========================================================

def store_team_environment(
    df,
):

    section(
        "STORING TEAM PREGAME ENVIRONMENT"
    )

    timestamp = now_utc()

    columns = [
        "game_id",
        "season",
        "week",

        "team",
        "opponent_team",

        "history_games",

        "points_for_last",
        "points_for_avg_3",
        "points_for_avg_5",

        "points_against_last",
        "points_against_avg_3",
        "points_against_avg_5",

        "offensive_plays_avg_3",
        "offensive_plays_avg_5",

        "pass_attempts_avg_3",
        "pass_attempts_avg_5",

        "rush_attempts_avg_3",
        "rush_attempts_avg_5",

        "pass_rate_avg_3",
        "pass_rate_avg_5",

        "rush_rate_avg_3",
        "rush_rate_avg_5",

        "passing_yards_avg_3",
        "rushing_yards_avg_3",

        "passing_tds_avg_3",
        "rushing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_points_allowed_avg_5",

        "opponent_pass_yards_allowed_avg_3",
        "opponent_rush_yards_allowed_avg_3",

        "opponent_pass_tds_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",

        "team_scoring_trend",
        "opponent_scoring_trend",

        "pace_trend",
        "pass_rate_trend",

        "updated_at",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    sql = f"""
        INSERT INTO
        team_pregame_environment (
            {", ".join(columns)}
        )

        VALUES (
            {placeholders}
        )
    """

    rows = []

    for _, row in df.iterrows():

        values = []

        for column in columns:

            if column == "updated_at":

                value = timestamp

            else:

                value = row[column]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                np.integer,
            ):

                value = int(value)

            elif isinstance(
                value,
                np.floating,
            ):

                value = float(value)

            values.append(
                value
            )

        rows.append(
            tuple(values)
        )

    with get_connection() as conn:

        conn.executemany(
            sql,
            rows,
        )

    print(
        f"Team environment rows inserted: "
        f"{len(rows)}"
    )


# =========================================================
# STORE DVP
# =========================================================

def store_dvp(
    df,
):

    section(
        "STORING POSITION DVP"
    )

    timestamp = now_utc()

    columns = [
        "game_id",
        "season",
        "week",

        "defense_team",
        "position",

        "history_games",

        "fd_allowed_last",
        "fd_allowed_avg_3",
        "fd_allowed_avg_5",

        "targets_allowed_avg_3",
        "carries_allowed_avg_3",

        "receptions_allowed_avg_3",

        "receiving_yards_allowed_avg_3",
        "rushing_yards_allowed_avg_3",

        "receiving_tds_allowed_avg_3",
        "rushing_tds_allowed_avg_3",

        "opportunities_allowed_avg_3",

        "fd_allowed_trend",
        "opportunity_allowed_trend",

        "updated_at",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    sql = f"""
        INSERT INTO
        team_position_dvp (
            {", ".join(columns)}
        )

        VALUES (
            {placeholders}
        )
    """

    rows = []

    for _, row in df.iterrows():

        values = []

        for column in columns:

            if column == "updated_at":

                value = timestamp

            else:

                value = row[column]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                np.integer,
            ):

                value = int(value)

            elif isinstance(
                value,
                np.floating,
            ):

                value = float(value)

            values.append(
                value
            )

        rows.append(
            tuple(values)
        )

    with get_connection() as conn:

        conn.executemany(
            sql,
            rows,
        )

    print(
        f"DvP rows inserted: "
        f"{len(rows)}"
    )


# =========================================================
# AUDIT
# =========================================================

def audit_tables():

    section(
        "TEAM ENVIRONMENT AUDIT"
    )

    with get_connection() as conn:

        team_rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM team_pregame_environment
            """
        ).fetchone()[0]

        dvp_rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM team_position_dvp
            """
        ).fetchone()[0]

        bad_team_games = conn.execute(
            """
            SELECT COUNT(*)

            FROM team_pregame_environment e

            LEFT JOIN games g
                ON e.game_id = g.game_id

            WHERE
                g.game_id IS NULL
            """
        ).fetchone()[0]

        duplicate_team_rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    game_id,
                    team,
                    COUNT(*) AS n

                FROM team_pregame_environment

                GROUP BY
                    game_id,
                    team

                HAVING
                    COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        duplicate_dvp_rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    game_id,
                    defense_team,
                    position,
                    COUNT(*) AS n

                FROM team_position_dvp

                GROUP BY
                    game_id,
                    defense_team,
                    position

                HAVING
                    COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        leakage_team = conn.execute(
            """
            SELECT COUNT(*)

            FROM team_pregame_environment

            WHERE
                history_games = 0
                AND
                (
                    points_for_avg_3 != 0
                    OR
                    offensive_plays_avg_3 != 0
                )
            """
        ).fetchone()[0]

        leakage_dvp = conn.execute(
            """
            SELECT COUNT(*)

            FROM team_position_dvp

            WHERE
                history_games = 0
                AND
                (
                    fd_allowed_avg_3 != 0
                    OR
                    opportunities_allowed_avg_3 != 0
                )
            """
        ).fetchone()[0]

    print(
        f"Team environment rows: "
        f"{team_rows}"
    )

    print(
        f"DvP rows: "
        f"{dvp_rows}"
    )

    print(
        f"Bad game relationships: "
        f"{bad_team_games}"
    )

    print(
        f"Duplicate team-game rows: "
        f"{duplicate_team_rows}"
    )

    print(
        f"Duplicate DvP rows: "
        f"{duplicate_dvp_rows}"
    )

    print(
        f"Team first-game leakage: "
        f"{leakage_team}"
    )

    print(
        f"DvP first-game leakage: "
        f"{leakage_dvp}"
    )

    problems = (
        bad_team_games
        +
        duplicate_team_rows
        +
        duplicate_dvp_rows
        +
        leakage_team
        +
        leakage_dvp
    )

    if problems:

        raise RuntimeError(
            "Team environment audit failed."
        )

    print()
    print(
        "PASS: team environment and DvP "
        "passed structural/leakage audits."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary():

    section(
        "TEAM ENVIRONMENT SUMMARY"
    )

    with get_connection() as conn:

        team_summary = pd.read_sql_query(
            """
            SELECT
                season,

                COUNT(*) AS rows,

                ROUND(
                    AVG(points_for_avg_3),
                    2
                ) AS avg_points_3,

                ROUND(
                    AVG(
                        offensive_plays_avg_3
                    ),
                    2
                ) AS avg_plays_3,

                ROUND(
                    AVG(pass_rate_avg_3),
                    3
                ) AS avg_pass_rate_3

            FROM team_pregame_environment

            GROUP BY season

            ORDER BY season
            """,
            conn,
        )

    print(
        team_summary.to_string(
            index=False
        )
    )


# =========================================================
# EXPORT
# =========================================================

def export_tables():

    section(
        "EXPORTING TEAM ENVIRONMENT"
    )

    with get_connection() as conn:

        team_df = pd.read_sql_query(
            """
            SELECT *

            FROM team_pregame_environment

            ORDER BY
                season,
                week,
                game_id,
                team
            """,
            conn,
        )

        dvp_df = pd.read_sql_query(
            """
            SELECT *

            FROM team_position_dvp

            ORDER BY
                season,
                week,
                game_id,
                defense_team,
                position
            """,
            conn,
        )

    team_df.to_csv(
        TEAM_ENV_CSV,
        index=False,
    )

    team_df.to_parquet(
        TEAM_ENV_PARQUET,
        index=False,
    )

    dvp_df.to_csv(
        DVP_CSV,
        index=False,
    )

    dvp_df.to_parquet(
        DVP_PARQUET,
        index=False,
    )

    print(
        f"Team rows exported: "
        f"{len(team_df)}"
    )

    print(
        f"DvP rows exported: "
        f"{len(dvp_df)}"
    )

    print()
    print(
        f"Team CSV:     "
        f"{TEAM_ENV_CSV}"
    )

    print(
        f"Team Parquet: "
        f"{TEAM_ENV_PARQUET}"
    )

    print(
        f"DvP CSV:      "
        f"{DVP_CSV}"
    )

    print(
        f"DvP Parquet:  "
        f"{DVP_PARQUET}"
    )


# =========================================================
# MAIN
# =========================================================

def run_team_environment_build():

    section(
        "NFL TEAM ENVIRONMENT ENGINE"
    )

    rebuild_tables()

    team_df = load_team_stats()

    player_df = load_player_usage()

    team_df = prepare_team_stats(
        team_df
    )

    team_df = build_team_environment(
        team_df
    )

    dvp_df = build_position_dvp(
        player_df
    )

    store_team_environment(
        team_df
    )

    store_dvp(
        dvp_df
    )

    audit_tables()

    print_summary()

    export_tables()

    section(
        "TEAM ENVIRONMENT BUILD SUCCESSFUL"
    )


if __name__ == "__main__":

    run_team_environment_build()
