from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection


# =========================================================
# OUTPUT PATHS
# =========================================================

FEATURE_MATRIX_CSV = (
    CSV_DIR / "nfl_dfs_feature_matrix.csv"
)

FEATURE_MATRIX_PARQUET = (
    PARQUET_DIR / "nfl_dfs_feature_matrix.parquet"
)

UNMATCHED_TEAM_CSV = (
    CSV_DIR / "audit_feature_matrix_unmatched_team_environment.csv"
)

UNMATCHED_DVP_CSV = (
    CSV_DIR / "audit_feature_matrix_unmatched_dvp.csv"
)

UNMATCHED_TARGET_CSV = (
    CSV_DIR / "audit_feature_matrix_unmatched_target.csv"
)


# =========================================================
# CONSTANTS
# =========================================================

OFFENSIVE_POSITIONS = {
    "QB",
    "RB",
    "WR",
    "TE",
}


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
        pd.to_numeric(
            series,
            errors="coerce",
        )
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

def rebuild_feature_matrix_table():

    section(
        "REBUILDING DFS FEATURE MATRIX TABLE"
    )

    with get_connection() as conn:

        conn.execute(
            """
            DROP TABLE IF EXISTS
            dfs_feature_matrix
            """
        )

        conn.execute(
            """
            CREATE TABLE
            dfs_feature_matrix (

                game_id TEXT NOT NULL,

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                season_type TEXT,

                identity_key TEXT NOT NULL,
                player_id TEXT NOT NULL,

                player_name TEXT,
                player_display_name TEXT,

                position TEXT NOT NULL,
                position_group TEXT,

                team TEXT NOT NULL,
                opponent_team TEXT NOT NULL,

                target_fanduel_points REAL NOT NULL,

                report_status TEXT,
                practice_status TEXT,
                primary_injury TEXT,

                active_flag INTEGER,
                injury_flag INTEGER,

                player_history_games INTEGER,

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

                team_history_games INTEGER,

                team_points_for_last REAL,
                team_points_for_avg_3 REAL,
                team_points_for_avg_5 REAL,

                team_points_against_last REAL,
                team_points_against_avg_3 REAL,
                team_points_against_avg_5 REAL,

                team_offensive_plays_avg_3 REAL,
                team_offensive_plays_avg_5 REAL,

                team_pass_attempts_avg_3 REAL,
                team_pass_attempts_avg_5 REAL,

                team_rush_attempts_avg_3 REAL,
                team_rush_attempts_avg_5 REAL,

                team_pass_rate_avg_3 REAL,
                team_pass_rate_avg_5 REAL,

                team_rush_rate_avg_3 REAL,
                team_rush_rate_avg_5 REAL,

                team_passing_yards_avg_3 REAL,
                team_rushing_yards_avg_3 REAL,

                team_passing_tds_avg_3 REAL,
                team_rushing_tds_avg_3 REAL,

                opponent_points_allowed_avg_3 REAL,
                opponent_points_allowed_avg_5 REAL,

                opponent_pass_yards_allowed_avg_3 REAL,
                opponent_rush_yards_allowed_avg_3 REAL,

                opponent_pass_tds_allowed_avg_3 REAL,
                opponent_rush_tds_allowed_avg_3 REAL,

                team_scoring_trend REAL,
                opponent_scoring_trend REAL,

                team_pace_trend REAL,
                team_pass_rate_trend REAL,

                dvp_history_games INTEGER,

                dvp_fd_allowed_last REAL,
                dvp_fd_allowed_avg_3 REAL,
                dvp_fd_allowed_avg_5 REAL,

                dvp_targets_allowed_avg_3 REAL,
                dvp_carries_allowed_avg_3 REAL,

                dvp_receptions_allowed_avg_3 REAL,

                dvp_receiving_yards_allowed_avg_3 REAL,
                dvp_rushing_yards_allowed_avg_3 REAL,

                dvp_receiving_tds_allowed_avg_3 REAL,
                dvp_rushing_tds_allowed_avg_3 REAL,

                dvp_opportunities_allowed_avg_3 REAL,

                dvp_fd_allowed_trend REAL,
                dvp_opportunity_allowed_trend REAL,

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
            idx_feature_matrix_player

            ON dfs_feature_matrix(
                player_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_feature_matrix_game

            ON dfs_feature_matrix(
                game_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_feature_matrix_week

            ON dfs_feature_matrix(
                season,
                week
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_feature_matrix_position

            ON dfs_feature_matrix(
                position
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_feature_matrix_team

            ON dfs_feature_matrix(
                team
            )
            """
        )

    print(
        "Fresh dfs_feature_matrix table created."
    )


# =========================================================
# LOAD PLAYER PREGAME FEATURES
# =========================================================

def load_player_pregame():

    section(
        "LOADING PLAYER PREGAME FEATURES"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM player_pregame_features

            WHERE
                position IN (
                    'QB',
                    'RB',
                    'WR',
                    'TE'
                )
            """,
            conn,
        )

    print(
        f"Offensive player rows loaded: "
        f"{len(df)}"
    )

    print()

    for position in [
        "QB",
        "RB",
        "WR",
        "TE",
    ]:

        count = int(
            (
                df["position"]
                ==
                position
            ).sum()
        )

        print(
            f"{position}: {count}"
        )

    return df


# =========================================================
# LOAD TARGET
# =========================================================

def load_target():

    section(
        "LOADING ACTUAL FANDUEL RESULTS"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                player_id,
                team,
                fanduel_points

            FROM player_weekly_usage

            WHERE
                position IN (
                    'QB',
                    'RB',
                    'WR',
                    'TE'
                )
            """,
            conn,
        )

    df = df.rename(
        columns={
            "fanduel_points":
                "target_fanduel_points",
        }
    )

    print(
        f"Target rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# LOAD TEAM ENVIRONMENT
# =========================================================

def load_team_environment():

    section(
        "LOADING TEAM PREGAME ENVIRONMENT"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM team_pregame_environment
            """,
            conn,
        )

    print(
        f"Team environment rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# LOAD DVP
# =========================================================

def load_dvp():

    section(
        "LOADING POSITION DVP"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM team_position_dvp

            WHERE
                position IN (
                    'QB',
                    'RB',
                    'WR',
                    'TE'
                )
            """,
            conn,
        )

    print(
        f"DvP rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# TARGET ATTACHMENT
# =========================================================

def attach_target(
    player_df,
    target_df,
):

    section(
        "ATTACHING FANDUEL TARGET"
    )

    target_df = (
        target_df
        .drop_duplicates(
            subset=[
                "game_id",
                "player_id",
                "team",
            ]
        )
    )

    merged = player_df.merge(
        target_df,
        how="left",
        on=[
            "game_id",
            "player_id",
            "team",
        ],
        validate="one_to_one",
    )

    unmatched = merged[
        merged[
            "target_fanduel_points"
        ].isna()
    ].copy()

    unmatched.to_csv(
        UNMATCHED_TARGET_CSV,
        index=False,
    )

    print(
        f"Rows with target: "
        f"{merged['target_fanduel_points'].notna().sum()}"
    )

    print(
        f"Rows without target: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:

        raise RuntimeError(
            "Feature matrix build stopped: "
            "FanDuel target attachment failed."
        )

    print(
        "PASS: every offensive player row "
        "has an actual FanDuel target."
    )

    return merged


# =========================================================
# TEAM ENVIRONMENT ATTACHMENT
# =========================================================

def attach_team_environment(
    df,
    team_df,
):

    section(
        "ATTACHING TEAM ENVIRONMENT"
    )

    team_columns = [
        "game_id",
        "team",

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
    ]

    team_small = team_df[
        team_columns
    ].copy()

    rename_map = {

        "history_games":
            "team_history_games",

        "points_for_last":
            "team_points_for_last",

        "points_for_avg_3":
            "team_points_for_avg_3",

        "points_for_avg_5":
            "team_points_for_avg_5",

        "points_against_last":
            "team_points_against_last",

        "points_against_avg_3":
            "team_points_against_avg_3",

        "points_against_avg_5":
            "team_points_against_avg_5",

        "offensive_plays_avg_3":
            "team_offensive_plays_avg_3",

        "offensive_plays_avg_5":
            "team_offensive_plays_avg_5",

        "pass_attempts_avg_3":
            "team_pass_attempts_avg_3",

        "pass_attempts_avg_5":
            "team_pass_attempts_avg_5",

        "rush_attempts_avg_3":
            "team_rush_attempts_avg_3",

        "rush_attempts_avg_5":
            "team_rush_attempts_avg_5",

        "pass_rate_avg_3":
            "team_pass_rate_avg_3",

        "pass_rate_avg_5":
            "team_pass_rate_avg_5",

        "rush_rate_avg_3":
            "team_rush_rate_avg_3",

        "rush_rate_avg_5":
            "team_rush_rate_avg_5",

        "passing_yards_avg_3":
            "team_passing_yards_avg_3",

        "rushing_yards_avg_3":
            "team_rushing_yards_avg_3",

        "passing_tds_avg_3":
            "team_passing_tds_avg_3",

        "rushing_tds_avg_3":
            "team_rushing_tds_avg_3",

        "pace_trend":
            "team_pace_trend",

        "pass_rate_trend":
            "team_pass_rate_trend",
    }

    team_small = team_small.rename(
        columns=rename_map
    )

    merged = df.merge(
        team_small,
        how="left",
        on=[
            "game_id",
            "team",
        ],
        validate="many_to_one",
    )

    unmatched = merged[
        merged[
            "team_history_games"
        ].isna()
    ].copy()

    unmatched.to_csv(
        UNMATCHED_TEAM_CSV,
        index=False,
    )

    print(
        f"Rows with team environment: "
        f"{merged['team_history_games'].notna().sum()}"
    )

    print(
        f"Rows without team environment: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:

        raise RuntimeError(
            "Feature matrix build stopped: "
            "team environment join failed."
        )

    print(
        "PASS: every player row received "
        "team/opponent environment."
    )

    return merged


# =========================================================
# DVP ATTACHMENT
# =========================================================

def attach_dvp(
    df,
    dvp_df,
):

    section(
        "ATTACHING POSITION DVP"
    )

    dvp_columns = [
        "game_id",
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
    ]

    dvp_small = dvp_df[
        dvp_columns
    ].copy()

    dvp_small = dvp_small.rename(
        columns={
            "history_games":
                "dvp_history_games",

            "fd_allowed_last":
                "dvp_fd_allowed_last",

            "fd_allowed_avg_3":
                "dvp_fd_allowed_avg_3",

            "fd_allowed_avg_5":
                "dvp_fd_allowed_avg_5",

            "targets_allowed_avg_3":
                "dvp_targets_allowed_avg_3",

            "carries_allowed_avg_3":
                "dvp_carries_allowed_avg_3",

            "receptions_allowed_avg_3":
                "dvp_receptions_allowed_avg_3",

            "receiving_yards_allowed_avg_3":
                "dvp_receiving_yards_allowed_avg_3",

            "rushing_yards_allowed_avg_3":
                "dvp_rushing_yards_allowed_avg_3",

            "receiving_tds_allowed_avg_3":
                "dvp_receiving_tds_allowed_avg_3",

            "rushing_tds_allowed_avg_3":
                "dvp_rushing_tds_allowed_avg_3",

            "opportunities_allowed_avg_3":
                "dvp_opportunities_allowed_avg_3",

            "fd_allowed_trend":
                "dvp_fd_allowed_trend",

            "opportunity_allowed_trend":
                "dvp_opportunity_allowed_trend",
        }
    )

    merged = df.merge(
        dvp_small,
        how="left",
        left_on=[
            "game_id",
            "opponent_team",
            "position",
        ],
        right_on=[
            "game_id",
            "defense_team",
            "position",
        ],
        validate="many_to_one",
    )

    unmatched = merged[
        merged[
            "dvp_history_games"
        ].isna()
    ].copy()

    unmatched.to_csv(
        UNMATCHED_DVP_CSV,
        index=False,
    )

    print(
        f"Rows with DvP: "
        f"{merged['dvp_history_games'].notna().sum()}"
    )

    print(
        f"Rows without DvP: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:

        print()
        print(
            "WARNING: some offensive rows do "
            "not have a position DvP record."
        )

        print(
            f"Audit: {UNMATCHED_DVP_CSV}"
        )

        print()
        print(
            "Missing DvP will be initialized "
            "to zero-history values rather than "
            "dropping valid player games."
        )

    # -----------------------------------------------------
    # Missing DvP is allowed only because an opponent may
    # have no previous/source observation for that exact
    # position/game combination.
    #
    # Never discard the actual player-game target.
    # -----------------------------------------------------

    dvp_numeric = [
        "dvp_history_games",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",

        "dvp_targets_allowed_avg_3",
        "dvp_carries_allowed_avg_3",

        "dvp_receptions_allowed_avg_3",

        "dvp_receiving_yards_allowed_avg_3",
        "dvp_rushing_yards_allowed_avg_3",

        "dvp_receiving_tds_allowed_avg_3",
        "dvp_rushing_tds_allowed_avg_3",

        "dvp_opportunities_allowed_avg_3",

        "dvp_fd_allowed_trend",
        "dvp_opportunity_allowed_trend",
    ]

    for column in dvp_numeric:

        merged[column] = clean_numeric(
            merged[column]
        )

    merged["dvp_history_games"] = (
        merged[
            "dvp_history_games"
        ]
        .astype(int)
    )

    if "defense_team" in merged.columns:

        merged = merged.drop(
            columns=[
                "defense_team",
            ]
        )

    return merged


# =========================================================
# FINAL CLEANUP
# =========================================================

def finalize_matrix(
    df,
):

    section(
        "FINALIZING FEATURE MATRIX"
    )

    # -----------------------------------------------------
    # Rename player history count so all history domains
    # have explicit meaning.
    # -----------------------------------------------------

    df = df.rename(
        columns={
            "history_games":
                "player_history_games",
        }
    )

    # -----------------------------------------------------
    # Target must remain actual FanDuel result.
    # Never modify or normalize the target here.
    # -----------------------------------------------------

    df[
        "target_fanduel_points"
    ] = clean_numeric(
        df[
            "target_fanduel_points"
        ]
    )

    numeric_columns = [
        column
        for column in df.columns
        if column not in {
            "game_id",
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
        }
    ]

    for column in numeric_columns:

        if column in {
            "season",
            "week",
        }:

            continue

        df[column] = clean_numeric(
            df[column]
        )

    integer_columns = [
        "season",
        "week",

        "active_flag",
        "injury_flag",

        "player_history_games",

        "role_expansion_last",
        "role_decline_last",

        "role_expansions_3",
        "role_declines_3",

        "high_usage_games_3",
        "starter_usage_games_3",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "team_history_games",
        "dvp_history_games",
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

    df = (
        df.sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
                "position",
                "player_display_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Final matrix rows: "
        f"{len(df)}"
    )

    print(
        f"Final matrix columns: "
        f"{len(df.columns)}"
    )

    return df


# =========================================================
# STORAGE COLUMN ORDER
# =========================================================

def matrix_columns():

    return [
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

        "target_fanduel_points",

        "report_status",
        "practice_status",
        "primary_injury",

        "active_flag",
        "injury_flag",

        "player_history_games",

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

        "team_history_games",

        "team_points_for_last",
        "team_points_for_avg_3",
        "team_points_for_avg_5",

        "team_points_against_last",
        "team_points_against_avg_3",
        "team_points_against_avg_5",

        "team_offensive_plays_avg_3",
        "team_offensive_plays_avg_5",

        "team_pass_attempts_avg_3",
        "team_pass_attempts_avg_5",

        "team_rush_attempts_avg_3",
        "team_rush_attempts_avg_5",

        "team_pass_rate_avg_3",
        "team_pass_rate_avg_5",

        "team_rush_rate_avg_3",
        "team_rush_rate_avg_5",

        "team_passing_yards_avg_3",
        "team_rushing_yards_avg_3",

        "team_passing_tds_avg_3",
        "team_rushing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_points_allowed_avg_5",

        "opponent_pass_yards_allowed_avg_3",
        "opponent_rush_yards_allowed_avg_3",

        "opponent_pass_tds_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",

        "team_scoring_trend",
        "opponent_scoring_trend",

        "team_pace_trend",
        "team_pass_rate_trend",

        "dvp_history_games",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",

        "dvp_targets_allowed_avg_3",
        "dvp_carries_allowed_avg_3",

        "dvp_receptions_allowed_avg_3",

        "dvp_receiving_yards_allowed_avg_3",
        "dvp_rushing_yards_allowed_avg_3",

        "dvp_receiving_tds_allowed_avg_3",
        "dvp_rushing_tds_allowed_avg_3",

        "dvp_opportunities_allowed_avg_3",

        "dvp_fd_allowed_trend",
        "dvp_opportunity_allowed_trend",
    ]


# =========================================================
# STORE MATRIX
# =========================================================

def store_matrix(
    df,
):

    section(
        "STORING DFS FEATURE MATRIX"
    )

    timestamp = now_utc()

    columns = matrix_columns()

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Feature matrix is missing "
            "required columns: "
            +
            ", ".join(
                missing_columns
            )
        )

    insert_columns = (
        columns
        +
        [
            "updated_at",
        ]
    )

    placeholders = ", ".join(
        ["?"] * len(insert_columns)
    )

    sql = f"""
        INSERT INTO
        dfs_feature_matrix (
            {", ".join(insert_columns)}
        )

        VALUES (
            {placeholders}
        )
    """

    rows = []

    for _, row in df.iterrows():

        values = []

        for column in columns:

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

        values.append(
            timestamp
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
        f"Feature rows inserted: "
        f"{len(rows)}"
    )


# =========================================================
# STRUCTURAL AUDIT
# =========================================================

def audit_matrix():

    section(
        "DFS FEATURE MATRIX AUDIT"
    )

    with get_connection() as conn:

        total = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix
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

                FROM dfs_feature_matrix

                GROUP BY
                    game_id,
                    player_id,
                    team

                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        bad_games = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix f

            LEFT JOIN games g
                ON f.game_id = g.game_id

            WHERE
                g.game_id IS NULL
            """
        ).fetchone()[0]

        missing_identity = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix

            WHERE
                identity_key IS NULL
            """
        ).fetchone()[0]

        invalid_position = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix

            WHERE
                position NOT IN (
                    'QB',
                    'RB',
                    'WR',
                    'TE'
                )
            """
        ).fetchone()[0]

        null_target = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix

            WHERE
                target_fanduel_points
                IS NULL
            """
        ).fetchone()[0]

        negative_history = conn.execute(
            """
            SELECT COUNT(*)

            FROM dfs_feature_matrix

            WHERE
                player_history_games < 0
                OR
                team_history_games < 0
                OR
                dvp_history_games < 0
            """
        ).fetchone()[0]

    print(
        f"Feature rows: "
        f"{total}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicates}"
    )

    print(
        f"Bad game relationships: "
        f"{bad_games}"
    )

    print(
        f"Missing identities: "
        f"{missing_identity}"
    )

    print(
        f"Invalid positions: "
        f"{invalid_position}"
    )

    print(
        f"NULL targets: "
        f"{null_target}"
    )

    print(
        f"Negative history values: "
        f"{negative_history}"
    )

    problems = (
        duplicates
        +
        bad_games
        +
        missing_identity
        +
        invalid_position
        +
        null_target
        +
        negative_history
    )

    if problems:

        raise RuntimeError(
            "DFS feature matrix "
            "structural audit failed."
        )

    print()
    print(
        "PASS: DFS feature matrix "
        "passed structural audits."
    )


# =========================================================
# LEAKAGE AUDIT
# =========================================================

def audit_leakage():

    section(
        "DFS FEATURE MATRIX LEAKAGE AUDIT"
    )

    with get_connection() as conn:

        player_first_game_leaks = (
            conn.execute(
                """
                SELECT COUNT(*)

                FROM dfs_feature_matrix

                WHERE
                    player_history_games = 0

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
        )

        team_first_game_leaks = (
            conn.execute(
                """
                SELECT COUNT(*)

                FROM dfs_feature_matrix

                WHERE
                    team_history_games = 0

                    AND
                    (
                        team_points_for_avg_3 != 0
                        OR
                        team_offensive_plays_avg_3 != 0
                    )
                """
            ).fetchone()[0]
        )

        dvp_first_game_leaks = (
            conn.execute(
                """
                SELECT COUNT(*)

                FROM dfs_feature_matrix

                WHERE
                    dvp_history_games = 0

                    AND
                    (
                        dvp_fd_allowed_avg_3 != 0
                        OR
                        dvp_opportunities_allowed_avg_3 != 0
                    )
                """
            ).fetchone()[0]
        )

    print(
        f"Player-history leakage: "
        f"{player_first_game_leaks}"
    )

    print(
        f"Team-history leakage: "
        f"{team_first_game_leaks}"
    )

    print(
        f"DvP-history leakage: "
        f"{dvp_first_game_leaks}"
    )

    problems = (
        player_first_game_leaks
        +
        team_first_game_leaks
        +
        dvp_first_game_leaks
    )

    if problems:

        raise RuntimeError(
            "DFS feature matrix "
            "leakage audit failed."
        )

    print()
    print(
        "PASS: feature matrix contains "
        "no detected first-observation leakage."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary():

    section(
        "DFS FEATURE MATRIX SUMMARY"
    )

    with get_connection() as conn:

        season_summary = (
            pd.read_sql_query(
                """
                SELECT
                    season,

                    COUNT(*) AS rows,

                    COUNT(
                        DISTINCT player_id
                    ) AS players,

                    ROUND(
                        AVG(
                            target_fanduel_points
                        ),
                        2
                    ) AS avg_actual_fd,

                    ROUND(
                        AVG(fd_avg_3),
                        2
                    ) AS avg_fd_history,

                    ROUND(
                        AVG(
                            opportunities_avg_3
                        ),
                        2
                    ) AS avg_opportunity_history,

                    ROUND(
                        AVG(
                            dvp_fd_allowed_avg_3
                        ),
                        2
                    ) AS avg_dvp_fd_allowed

                FROM dfs_feature_matrix

                GROUP BY season

                ORDER BY season
                """,
                conn,
            )
        )

        position_summary = (
            pd.read_sql_query(
                """
                SELECT
                    position,

                    COUNT(*) AS rows,

                    COUNT(
                        DISTINCT player_id
                    ) AS players,

                    ROUND(
                        AVG(
                            target_fanduel_points
                        ),
                        2
                    ) AS avg_fd,

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

                    ROUND(
                        AVG(
                            snap_pct_avg_3
                        ),
                        3
                    ) AS avg_snap_3

                FROM dfs_feature_matrix

                GROUP BY position

                ORDER BY position
                """,
                conn,
            )
        )

    print(
        season_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "BY POSITION"
    )

    print()

    print(
        position_summary.to_string(
            index=False
        )
    )


# =========================================================
# HIGH-VALUE EXAMPLES
# =========================================================

def print_top_profiles():

    section(
        "TOP HISTORICAL PREGAME DFS PROFILES"
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

                player_history_games,

                fd_avg_3,
                fd_max_5,

                opportunities_avg_3,
                snap_pct_avg_3,

                target_share_avg_3,
                wopr_avg_3,

                team_points_for_avg_3,

                dvp_fd_allowed_avg_3,
                dvp_opportunities_allowed_avg_3,

                rising_role_flag,

                target_fanduel_points

            FROM dfs_feature_matrix

            WHERE
                player_history_games >= 3

            ORDER BY
                fd_avg_3 DESC,
                opportunities_avg_3 DESC

            LIMIT 30
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

def export_matrix():

    section(
        "DFS FEATURE MATRIX EXPORT"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM dfs_feature_matrix

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
        FEATURE_MATRIX_CSV,
        index=False,
    )

    df.to_parquet(
        FEATURE_MATRIX_PARQUET,
        index=False,
    )

    print(
        f"Rows exported: "
        f"{len(df)}"
    )

    print(
        f"Columns exported: "
        f"{len(df.columns)}"
    )

    print()

    print(
        f"CSV:     "
        f"{FEATURE_MATRIX_CSV}"
    )

    print(
        f"Parquet: "
        f"{FEATURE_MATRIX_PARQUET}"
    )


# =========================================================
# MAIN
# =========================================================

def run_feature_matrix_build():

    section(
        "NFL DFS FEATURE MATRIX ENGINE"
    )

    rebuild_feature_matrix_table()

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    player_df = load_player_pregame()

    target_df = load_target()

    team_df = load_team_environment()

    dvp_df = load_dvp()

    # -----------------------------------------------------
    # JOIN ACTUAL RESULT
    # -----------------------------------------------------

    df = attach_target(
        player_df,
        target_df,
    )

    # -----------------------------------------------------
    # JOIN TEAM / GAME ENVIRONMENT
    # -----------------------------------------------------

    df = attach_team_environment(
        df,
        team_df,
    )

    # -----------------------------------------------------
    # JOIN DEFENSE VS POSITION
    # -----------------------------------------------------

    df = attach_dvp(
        df,
        dvp_df,
    )

    # -----------------------------------------------------
    # FINALIZE
    # -----------------------------------------------------

    df = finalize_matrix(
        df
    )

    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    store_matrix(
        df
    )

    # -----------------------------------------------------
    # AUDIT
    # -----------------------------------------------------

    audit_matrix()

    audit_leakage()

    print_summary()

    print_top_profiles()

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    export_matrix()

    section(
        "DFS FEATURE MATRIX BUILD SUCCESSFUL"
    )


if __name__ == "__main__":

    run_feature_matrix_build()
