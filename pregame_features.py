import numpy as np
import pandas as pd

from datetime import datetime, timezone

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection


# =========================================================
# OUTPUT PATHS
# =========================================================

PREGAME_CSV = (
    CSV_DIR / "nfl_player_pregame_features.csv"
)

PREGAME_PARQUET = (
    PARQUET_DIR / "nfl_player_pregame_features.parquet"
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
# DATABASE TABLE
# =========================================================

def rebuild_pregame_table():

    section(
        "REBUILDING PLAYER PREGAME FEATURES TABLE"
    )

    with get_connection() as conn:

        conn.execute(
            """
            DROP TABLE IF EXISTS
            player_pregame_features
            """
        )

        conn.execute(
            """
            CREATE TABLE
            player_pregame_features (

                game_id TEXT NOT NULL,

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                season_type TEXT,

                identity_key TEXT NOT NULL,
                player_id TEXT NOT NULL,

                player_name TEXT,
                player_display_name TEXT,

                position TEXT,
                position_group TEXT,

                team TEXT NOT NULL,
                opponent_team TEXT,

                report_status TEXT,
                practice_status TEXT,
                primary_injury TEXT,

                active_flag INTEGER,
                injury_flag INTEGER,

                history_games INTEGER,

                fd_last REAL,

                fd_avg_3 REAL,
                fd_avg_5 REAL,
                fd_max_5 REAL,
                fd_std_5 REAL,

                opportunities_last REAL,
                opportunities_avg_3 REAL,
                opportunities_avg_5 REAL,
                opportunities_max_5 REAL,

                touches_last REAL,
                touches_avg_3 REAL,
                touches_avg_5 REAL,

                snaps_last REAL,
                snaps_avg_3 REAL,
                snaps_avg_5 REAL,

                snap_pct_last REAL,
                snap_pct_avg_3 REAL,
                snap_pct_avg_5 REAL,

                targets_last REAL,
                targets_avg_3 REAL,
                targets_avg_5 REAL,

                carries_last REAL,
                carries_avg_3 REAL,
                carries_avg_5 REAL,

                receptions_avg_3 REAL,

                yards_last REAL,
                yards_avg_3 REAL,
                yards_avg_5 REAL,

                target_share_last REAL,
                target_share_avg_3 REAL,

                air_yards_share_last REAL,
                air_yards_share_avg_3 REAL,

                wopr_last REAL,
                wopr_avg_3 REAL,

                fd_per_snap_avg_3 REAL,
                fd_per_touch_avg_3 REAL,
                yards_per_opportunity_avg_3 REAL,

                role_expansion_last INTEGER,
                role_decline_last INTEGER,

                role_expansions_3 INTEGER,
                role_declines_3 INTEGER,

                high_usage_games_3 INTEGER,
                starter_usage_games_3 INTEGER,

                opportunity_trend REAL,
                snap_trend REAL,
                target_trend REAL,
                carry_trend REAL,
                fanduel_trend REAL,

                established_role_flag INTEGER,
                rising_role_flag INTEGER,
                declining_role_flag INTEGER,

                updated_at TEXT,

                PRIMARY KEY (
                    game_id,
                    player_id,
                    team
                )
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_player

            ON player_pregame_features(
                player_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_identity

            ON player_pregame_features(
                identity_key
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_game

            ON player_pregame_features(
                game_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_season_week

            ON player_pregame_features(
                season,
                week
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_team

            ON player_pregame_features(
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_pregame_position

            ON player_pregame_features(
                position
            )
            """
        )

    print(
        "Fresh player_pregame_features "
        "table created."
    )


# =========================================================
# LOAD USAGE
# =========================================================

def load_usage():

    section(
        "LOADING PLAYER WEEKLY USAGE"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                game_id,

                season,
                week,
                season_type,

                identity_key,
                player_id,

                player_name,
                player_display_name,

                position,
                position_group,

                team,
                opponent_team,

                fanduel_points,

                offense_snaps,
                offense_pct,

                carries,
                targets,
                receptions,

                target_share,
                air_yards_share,
                wopr,

                touches,
                opportunities,

                yards_from_scrimmage,

                fanduel_per_snap,
                fanduel_per_touch,
                yards_per_opportunity,

                report_status,
                practice_status,
                primary_injury,

                active_flag,
                injury_flag,

                role_expansion_flag,
                role_decline_flag,

                high_usage_flag,
                starter_usage_flag

            FROM player_weekly_usage

            ORDER BY
                player_id,
                season,
                week,
                game_id
            """,
            conn,
        )

    print(
        f"Usage rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# NUMERIC CLEANUP
# =========================================================

def normalize_numeric_columns(
    df,
):

    columns = [
        "fanduel_points",

        "offense_snaps",
        "offense_pct",

        "carries",
        "targets",
        "receptions",

        "target_share",
        "air_yards_share",
        "wopr",

        "touches",
        "opportunities",

        "yards_from_scrimmage",

        "fanduel_per_snap",
        "fanduel_per_touch",
        "yards_per_opportunity",

        "active_flag",
        "injury_flag",

        "role_expansion_flag",
        "role_decline_flag",

        "high_usage_flag",
        "starter_usage_flag",
    ]

    for column in columns:

        df[column] = safe_numeric(
            df[column]
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


def lagged_max(
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
            .max()
    )


def lagged_std(
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
                min_periods=2,
            )
            .std(
                ddof=0
            )
    )


def lagged_sum(
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
            .sum()
    )


# =========================================================
# BUILD PREGAME FEATURES
# =========================================================

def calculate_pregame_features(
    df,
):

    section(
        "CALCULATING STRICTLY PREGAME FEATURES"
    )

    df = (
        df.sort_values(
            [
                "player_id",
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
        "player_id",
        group_keys=False,
    )

    # -----------------------------------------------------
    # NUMBER OF PRIOR OBSERVATIONS
    # -----------------------------------------------------

    df[
        "history_games"
    ] = group.cumcount()

    # =====================================================
    # FANDUEL HISTORY
    # =====================================================

    df[
        "fd_last"
    ] = lagged_last(
        group,
        "fanduel_points",
    )

    df[
        "fd_avg_3"
    ] = lagged_mean(
        group,
        "fanduel_points",
        3,
    )

    df[
        "fd_avg_5"
    ] = lagged_mean(
        group,
        "fanduel_points",
        5,
    )

    df[
        "fd_max_5"
    ] = lagged_max(
        group,
        "fanduel_points",
        5,
    )

    df[
        "fd_std_5"
    ] = lagged_std(
        group,
        "fanduel_points",
        5,
    )

    # =====================================================
    # OPPORTUNITY
    # =====================================================

    df[
        "opportunities_last"
    ] = lagged_last(
        group,
        "opportunities",
    )

    df[
        "opportunities_avg_3"
    ] = lagged_mean(
        group,
        "opportunities",
        3,
    )

    df[
        "opportunities_avg_5"
    ] = lagged_mean(
        group,
        "opportunities",
        5,
    )

    df[
        "opportunities_max_5"
    ] = lagged_max(
        group,
        "opportunities",
        5,
    )

    # =====================================================
    # TOUCHES
    # =====================================================

    df[
        "touches_last"
    ] = lagged_last(
        group,
        "touches",
    )

    df[
        "touches_avg_3"
    ] = lagged_mean(
        group,
        "touches",
        3,
    )

    df[
        "touches_avg_5"
    ] = lagged_mean(
        group,
        "touches",
        5,
    )

    # =====================================================
    # SNAP VOLUME
    # =====================================================

    df[
        "snaps_last"
    ] = lagged_last(
        group,
        "offense_snaps",
    )

    df[
        "snaps_avg_3"
    ] = lagged_mean(
        group,
        "offense_snaps",
        3,
    )

    df[
        "snaps_avg_5"
    ] = lagged_mean(
        group,
        "offense_snaps",
        5,
    )

    # =====================================================
    # SNAP PERCENT
    # =====================================================

    df[
        "snap_pct_last"
    ] = lagged_last(
        group,
        "offense_pct",
    )

    df[
        "snap_pct_avg_3"
    ] = lagged_mean(
        group,
        "offense_pct",
        3,
    )

    df[
        "snap_pct_avg_5"
    ] = lagged_mean(
        group,
        "offense_pct",
        5,
    )

    # =====================================================
    # TARGETS
    # =====================================================

    df[
        "targets_last"
    ] = lagged_last(
        group,
        "targets",
    )

    df[
        "targets_avg_3"
    ] = lagged_mean(
        group,
        "targets",
        3,
    )

    df[
        "targets_avg_5"
    ] = lagged_mean(
        group,
        "targets",
        5,
    )

    # =====================================================
    # CARRIES
    # =====================================================

    df[
        "carries_last"
    ] = lagged_last(
        group,
        "carries",
    )

    df[
        "carries_avg_3"
    ] = lagged_mean(
        group,
        "carries",
        3,
    )

    df[
        "carries_avg_5"
    ] = lagged_mean(
        group,
        "carries",
        5,
    )

    # =====================================================
    # RECEPTIONS
    # =====================================================

    df[
        "receptions_avg_3"
    ] = lagged_mean(
        group,
        "receptions",
        3,
    )

    # =====================================================
    # SCRIMMAGE YARDS
    # =====================================================

    df[
        "yards_last"
    ] = lagged_last(
        group,
        "yards_from_scrimmage",
    )

    df[
        "yards_avg_3"
    ] = lagged_mean(
        group,
        "yards_from_scrimmage",
        3,
    )

    df[
        "yards_avg_5"
    ] = lagged_mean(
        group,
        "yards_from_scrimmage",
        5,
    )

    # =====================================================
    # RECEIVING ROLE
    # =====================================================

    df[
        "target_share_last"
    ] = lagged_last(
        group,
        "target_share",
    )

    df[
        "target_share_avg_3"
    ] = lagged_mean(
        group,
        "target_share",
        3,
    )

    df[
        "air_yards_share_last"
    ] = lagged_last(
        group,
        "air_yards_share",
    )

    df[
        "air_yards_share_avg_3"
    ] = lagged_mean(
        group,
        "air_yards_share",
        3,
    )

    df[
        "wopr_last"
    ] = lagged_last(
        group,
        "wopr",
    )

    df[
        "wopr_avg_3"
    ] = lagged_mean(
        group,
        "wopr",
        3,
    )

    # =====================================================
    # EFFICIENCY HISTORY
    # =====================================================

    df[
        "fd_per_snap_avg_3"
    ] = lagged_mean(
        group,
        "fanduel_per_snap",
        3,
    )

    df[
        "fd_per_touch_avg_3"
    ] = lagged_mean(
        group,
        "fanduel_per_touch",
        3,
    )

    df[
        "yards_per_opportunity_avg_3"
    ] = lagged_mean(
        group,
        "yards_per_opportunity",
        3,
    )

    # =====================================================
    # PRIOR ROLE SIGNALS
    # =====================================================

    df[
        "role_expansion_last"
    ] = lagged_last(
        group,
        "role_expansion_flag",
    )

    df[
        "role_decline_last"
    ] = lagged_last(
        group,
        "role_decline_flag",
    )

    df[
        "role_expansions_3"
    ] = lagged_sum(
        group,
        "role_expansion_flag",
        3,
    )

    df[
        "role_declines_3"
    ] = lagged_sum(
        group,
        "role_decline_flag",
        3,
    )

    df[
        "high_usage_games_3"
    ] = lagged_sum(
        group,
        "high_usage_flag",
        3,
    )

    df[
        "starter_usage_games_3"
    ] = lagged_sum(
        group,
        "starter_usage_flag",
        3,
    )

    # =====================================================
    # TREND FEATURES
    #
    # Positive means recent role is stronger than the
    # broader five-game baseline.
    # =====================================================

    df[
        "opportunity_trend"
    ] = (
        df[
            "opportunities_avg_3"
        ]
        -
        df[
            "opportunities_avg_5"
        ]
    )

    df[
        "snap_trend"
    ] = (
        df[
            "snap_pct_avg_3"
        ]
        -
        df[
            "snap_pct_avg_5"
        ]
    )

    df[
        "target_trend"
    ] = (
        df[
            "targets_avg_3"
        ]
        -
        df[
            "targets_avg_5"
        ]
    )

    df[
        "carry_trend"
    ] = (
        df[
            "carries_avg_3"
        ]
        -
        df[
            "carries_avg_5"
        ]
    )

    df[
        "fanduel_trend"
    ] = (
        df[
            "fd_avg_3"
        ]
        -
        df[
            "fd_avg_5"
        ]
    )

    # =====================================================
    # ROLE CLASSIFICATIONS
    #
    # These use ONLY information prior to the game.
    # =====================================================

    df[
        "established_role_flag"
    ] = (
        (
            df[
                "snap_pct_avg_3"
            ] >= 0.60
        )
        |
        (
            df[
                "opportunities_avg_3"
            ] >= 12.0
        )
    ).astype(int)

    df[
        "rising_role_flag"
    ] = (
        (
            df[
                "opportunity_trend"
            ] >= 2.0
        )
        |
        (
            df[
                "snap_trend"
            ] >= 0.08
        )
        |
        (
            df[
                "target_trend"
            ] >= 1.5
        )
    ).astype(int)

    df[
        "declining_role_flag"
    ] = (
        (
            df[
                "opportunity_trend"
            ] <= -2.0
        )
        |
        (
            df[
                "snap_trend"
            ] <= -0.08
        )
        |
        (
            df[
                "target_trend"
            ] <= -1.5
        )
    ).astype(int)

    # =====================================================
    # CLEAN FIRST-GAME / NULL HISTORY
    # =====================================================

    numeric_feature_columns = [
        "fd_last",

        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",

        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "opportunities_max_5",

        "touches_last",
        "touches_avg_3",
        "touches_avg_5",

        "snaps_last",
        "snaps_avg_3",
        "snaps_avg_5",

        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",

        "targets_last",
        "targets_avg_3",
        "targets_avg_5",

        "carries_last",
        "carries_avg_3",
        "carries_avg_5",

        "receptions_avg_3",

        "yards_last",
        "yards_avg_3",
        "yards_avg_5",

        "target_share_last",
        "target_share_avg_3",

        "air_yards_share_last",
        "air_yards_share_avg_3",

        "wopr_last",
        "wopr_avg_3",

        "fd_per_snap_avg_3",
        "fd_per_touch_avg_3",
        "yards_per_opportunity_avg_3",

        "role_expansion_last",
        "role_decline_last",

        "role_expansions_3",
        "role_declines_3",

        "high_usage_games_3",
        "starter_usage_games_3",

        "opportunity_trend",
        "snap_trend",
        "target_trend",
        "carry_trend",
        "fanduel_trend",
    ]

    for column in numeric_feature_columns:

        df[column] = clean_numeric(
            df[column]
        )

    integer_columns = [
        "history_games",

        "role_expansion_last",
        "role_decline_last",

        "role_expansions_3",
        "role_declines_3",

        "high_usage_games_3",
        "starter_usage_games_3",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "active_flag",
        "injury_flag",
    ]

    for column in integer_columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    print(
        "Pregame features calculated."
    )

    print(
        "PASS: all rolling performance "
        "features use prior games only."
    )

    return df


# =========================================================
# STORE
# =========================================================

def store_pregame_features(
    df,
):

    section(
        "STORING PLAYER PREGAME FEATURES"
    )

    updated_at = now_utc()

    columns = [
        "game_id",

        "season",
        "week",
        "season_type",

        "identity_key",
        "player_id",

        "player_name",
        "player_display_name",

        "position",
        "position_group",

        "team",
        "opponent_team",

        "report_status",
        "practice_status",
        "primary_injury",

        "active_flag",
        "injury_flag",

        "history_games",

        "fd_last",

        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",

        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "opportunities_max_5",

        "touches_last",
        "touches_avg_3",
        "touches_avg_5",

        "snaps_last",
        "snaps_avg_3",
        "snaps_avg_5",

        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",

        "targets_last",
        "targets_avg_3",
        "targets_avg_5",

        "carries_last",
        "carries_avg_3",
        "carries_avg_5",

        "receptions_avg_3",

        "yards_last",
        "yards_avg_3",
        "yards_avg_5",

        "target_share_last",
        "target_share_avg_3",

        "air_yards_share_last",
        "air_yards_share_avg_3",

        "wopr_last",
        "wopr_avg_3",

        "fd_per_snap_avg_3",
        "fd_per_touch_avg_3",
        "yards_per_opportunity_avg_3",

        "role_expansion_last",
        "role_decline_last",

        "role_expansions_3",
        "role_declines_3",

        "high_usage_games_3",
        "starter_usage_games_3",

        "opportunity_trend",
        "snap_trend",
        "target_trend",
        "carry_trend",
        "fanduel_trend",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "updated_at",
    ]

    placeholders = (
        ", ".join(
            ["?"] * len(columns)
        )
    )

    sql = f"""
        INSERT INTO
        player_pregame_features (
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

                value = updated_at

            else:

                value = row[
                    column
                ]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                np.integer,
            ):

                value = int(
                    value
                )

            elif isinstance(
                value,
                np.floating,
            ):

                value = float(
                    value
                )

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
        f"Rows inserted: "
        f"{len(rows)}"
    )


# =========================================================
# STRUCTURAL AUDIT
# =========================================================

def audit_pregame_table():

    section(
        "PREGAME FEATURE STRUCTURAL AUDIT"
    )

    with get_connection() as conn:

        total = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features
            """
        ).fetchone()[0]

        missing_identity = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features

            WHERE
                identity_key IS NULL
            """
        ).fetchone()[0]

        unmatched_game = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features p

            LEFT JOIN games g
                ON p.game_id = g.game_id

            WHERE
                g.game_id IS NULL
            """
        ).fetchone()[0]

        duplicates = conn.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    game_id,
                    player_id,
                    team,
                    COUNT(*) AS n

                FROM player_pregame_features

                GROUP BY
                    game_id,
                    player_id,
                    team

                HAVING
                    COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        negative_history = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features

            WHERE
                history_games < 0
            """
        ).fetchone()[0]

    print(
        f"Pregame rows: "
        f"{total}"
    )

    print(
        f"Missing identity: "
        f"{missing_identity}"
    )

    print(
        f"Unmatched game_id: "
        f"{unmatched_game}"
    )

    print(
        f"Duplicate player-game rows: "
        f"{duplicates}"
    )

    print(
        f"Negative history counts: "
        f"{negative_history}"
    )

    problems = (
        missing_identity
        +
        unmatched_game
        +
        duplicates
        +
        negative_history
    )

    if problems:

        raise RuntimeError(
            "Pregame structural audit failed."
        )

    print()
    print(
        "PASS: pregame feature table "
        "passed structural audits."
    )


# =========================================================
# LEAKAGE AUDIT
# =========================================================

def audit_first_game_history():

    section(
        "PREGAME LEAKAGE AUDIT"
    )

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features

            WHERE
                history_games = 0
                AND
                (
                    fd_avg_3 != 0
                    OR
                    opportunities_avg_3 != 0
                    OR
                    snap_pct_avg_3 != 0
                    OR
                    targets_avg_3 != 0
                    OR
                    carries_avg_3 != 0
                )
            """
        ).fetchone()[0]

        first_games = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_pregame_features

            WHERE
                history_games = 0
            """
        ).fetchone()[0]

    print(
        f"First observed player games: "
        f"{first_games}"
    )

    print(
        f"First games with leaked history: "
        f"{rows}"
    )

    if rows:

        raise RuntimeError(
            "Leakage audit failed."
        )

    print()
    print(
        "PASS: first observed games contain "
        "no prior-game performance history."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary():

    section(
        "PREGAME FEATURE SUMMARY"
    )

    with get_connection() as conn:

        summary = pd.read_sql_query(
            """
            SELECT
                season,

                COUNT(*) AS rows,

                COUNT(
                    DISTINCT player_id
                ) AS players,

                ROUND(
                    AVG(history_games),
                    2
                ) AS avg_history,

                ROUND(
                    AVG(fd_avg_3),
                    2
                ) AS avg_fd_3,

                ROUND(
                    AVG(
                        opportunities_avg_3
                    ),
                    2
                ) AS avg_opp_3,

                SUM(
                    established_role_flag
                ) AS established_roles,

                SUM(
                    rising_role_flag
                ) AS rising_roles,

                SUM(
                    declining_role_flag
                ) AS declining_roles

            FROM player_pregame_features

            GROUP BY season

            ORDER BY season
            """,
            conn,
        )

    print(
        summary.to_string(
            index=False
        )
    )


# =========================================================
# EXAMPLE HIGH-USAGE PLAYERS
# =========================================================

def print_role_examples():

    section(
        "TOP PREGAME ROLE PROFILES"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                season,
                week,

                player_display_name,
                position,

                team,
                opponent_team,

                history_games,

                snap_pct_avg_3,
                opportunities_avg_3,

                targets_avg_3,
                carries_avg_3,

                fd_avg_3,
                fd_max_5,

                opportunity_trend,
                snap_trend,

                rising_role_flag

            FROM player_pregame_features

            WHERE
                position IN (
                    'RB',
                    'WR',
                    'TE'
                )

                AND
                history_games >= 3

            ORDER BY
                opportunities_avg_3 DESC,
                snap_pct_avg_3 DESC

            LIMIT 25
            """,
            conn,
        )

    print(
        df.to_string(
            index=False
        )
    )


# =========================================================
# EXPORT
# =========================================================

def export_pregame():

    section(
        "PREGAME FEATURE EXPORT"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM player_pregame_features

            ORDER BY
                season,
                week,
                game_id,
                team,
                position,
                player_display_name
            """,
            conn,
        )

    df.to_csv(
        PREGAME_CSV,
        index=False,
    )

    df.to_parquet(
        PREGAME_PARQUET,
        index=False,
    )

    print(
        f"Rows exported: "
        f"{len(df)}"
    )

    print(
        f"CSV:     "
        f"{PREGAME_CSV}"
    )

    print(
        f"Parquet: "
        f"{PREGAME_PARQUET}"
    )


# =========================================================
# MAIN
# =========================================================

def run_pregame_feature_build():

    section(
        "NFL PREGAME FEATURE ENGINE"
    )

    rebuild_pregame_table()

    df = load_usage()

    df = normalize_numeric_columns(
        df
    )

    df = calculate_pregame_features(
        df
    )

    store_pregame_features(
        df
    )

    audit_pregame_table()

    audit_first_game_history()

    print_summary()

    print_role_examples()

    export_pregame()

    section(
        "PREGAME FEATURE BUILD SUCCESSFUL"
    )


if __name__ == "__main__":

    run_pregame_feature_build()
