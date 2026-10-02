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

USAGE_CSV = (
    CSV_DIR / "nfl_player_weekly_usage.csv"
)

USAGE_PARQUET = (
    PARQUET_DIR / "nfl_player_weekly_usage.parquet"
)

UNMATCHED_IDENTITY_CSV = (
    CSV_DIR / "audit_usage_unmatched_identity.csv"
)

UNMATCHED_SNAPS_CSV = (
    CSV_DIR / "audit_usage_unmatched_snaps.csv"
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


def safe_divide(
    numerator,
    denominator,
):

    denominator = denominator.replace(
        0,
        np.nan,
    )

    result = (
        numerator
        / denominator
    )

    return (
        result
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

def rebuild_usage_table():

    section(
        "REBUILDING PLAYER WEEKLY USAGE TABLE"
    )

    with get_connection() as conn:

        conn.execute(
            """
            DROP TABLE IF EXISTS
            player_weekly_usage
            """
        )

        conn.execute(
            """
            CREATE TABLE
            player_weekly_usage (

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

                fanduel_points REAL,

                offense_snaps REAL,
                offense_pct REAL,

                carries REAL,
                targets REAL,
                receptions REAL,

                rushing_yards REAL,
                receiving_yards REAL,

                rushing_tds REAL,
                receiving_tds REAL,

                target_share REAL,
                air_yards_share REAL,
                wopr REAL,

                touches REAL,
                opportunities REAL,

                yards_from_scrimmage REAL,

                carries_per_snap REAL,
                targets_per_snap REAL,
                touches_per_snap REAL,

                fanduel_per_snap REAL,
                fanduel_per_touch REAL,
                yards_per_opportunity REAL,

                report_status TEXT,
                practice_status TEXT,
                primary_injury TEXT,

                active_flag INTEGER,
                injury_flag INTEGER,

                usage_3g REAL,
                snap_pct_3g REAL,
                target_3g REAL,
                carry_3g REAL,
                fanduel_3g REAL,

                usage_delta REAL,
                snap_delta REAL,
                target_delta REAL,
                carry_delta REAL,

                role_expansion_flag INTEGER,
                role_decline_flag INTEGER,

                high_usage_flag INTEGER,
                starter_usage_flag INTEGER,

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
            idx_usage_player

            ON player_weekly_usage(
                player_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_usage_identity

            ON player_weekly_usage(
                identity_key
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_usage_game

            ON player_weekly_usage(
                game_id
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_usage_season_week

            ON player_weekly_usage(
                season,
                week
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_usage_team

            ON player_weekly_usage(
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX
            idx_usage_position

            ON player_weekly_usage(
                position
            )
            """
        )

    print(
        "Fresh player_weekly_usage table created."
    )


# =========================================================
# PLAYER GAME STATS
# =========================================================

def load_player_stats():

    section(
        "LOADING PLAYER GAME STATS"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                season_type,

                player_id,
                player_name,
                player_display_name,

                position,
                position_group,

                team,
                opponent_team,

                fanduel_points,

                carries,
                rushing_yards,
                rushing_tds,

                targets,
                receptions,
                receiving_yards,
                receiving_tds,

                target_share,
                air_yards_share,
                wopr

            FROM player_game_stats

            WHERE
                game_id IS NOT NULL
            """,
            conn,
        )

    print(
        f"Player-game rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# IDENTITY
# =========================================================

def load_identity():

    section(
        "LOADING PLAYER IDENTITY"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                identity_key,
                gsis_id,
                pfr_id

            FROM player_identity
            """,
            conn,
        )

    print(
        f"Identity rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# SNAP COUNTS
# =========================================================

def load_snaps():

    section(
        "LOADING SNAP COUNTS"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                team,
                pfr_player_id,

                offense_snaps,
                offense_pct

            FROM player_snap_counts
            """,
            conn,
        )

    print(
        f"Snap rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# INJURIES
# =========================================================

def load_injuries():

    section(
        "LOADING INJURY CONTEXT"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                game_type,
                team,
                gsis_id,

                report_primary_injury,
                report_status,
                practice_status

            FROM injuries
            """,
            conn,
        )

    print(
        f"Injury rows loaded: "
        f"{len(df)}"
    )

    return df


# =========================================================
# IDENTITY AUDIT
# =========================================================

def attach_identity(
    stats_df,
    identity_df,
):

    section(
        "ATTACHING PLAYER IDENTITY"
    )

    identity_small = (
        identity_df[
            [
                "identity_key",
                "gsis_id",
                "pfr_id",
            ]
        ]
        .drop_duplicates(
            subset=[
                "gsis_id",
            ]
        )
    )

    merged = stats_df.merge(
        identity_small,
        how="left",
        left_on="player_id",
        right_on="gsis_id",
        validate="many_to_one",
    )

    unmatched = merged[
        merged[
            "identity_key"
        ].isna()
    ].copy()

    unmatched.to_csv(
        UNMATCHED_IDENTITY_CSV,
        index=False,
    )

    print(
        f"Rows with identity: "
        f"{merged['identity_key'].notna().sum()}"
    )

    print(
        f"Rows without identity: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:

        raise RuntimeError(
            "Usage build stopped because "
            "player-game rows failed identity "
            "resolution."
        )

    print(
        "PASS: every player-game row "
        "resolved to player_identity."
    )

    return merged


# =========================================================
# SNAP ATTACHMENT
# =========================================================

def attach_snaps(
    df,
    snap_df,
):

    section(
        "ATTACHING SNAP COUNTS"
    )

    snaps = (
        snap_df[
            [
                "game_id",
                "team",
                "pfr_player_id",
                "offense_snaps",
                "offense_pct",
            ]
        ]
        .drop_duplicates(
            subset=[
                "game_id",
                "team",
                "pfr_player_id",
            ]
        )
    )

    merged = df.merge(
        snaps,
        how="left",
        left_on=[
            "game_id",
            "team",
            "pfr_id",
        ],
        right_on=[
            "game_id",
            "team",
            "pfr_player_id",
        ],
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # A statistical player can legitimately have no
    # offensive snap record:
    #
    # kickers
    # returners
    # defenders with a recorded stat
    # special-teams-only players
    #
    # Therefore this is an audit, not a fatal error.
    # -----------------------------------------------------

    unmatched = merged[
        merged[
            "offense_snaps"
        ].isna()
    ].copy()

    unmatched[
        [
            "season",
            "week",
            "game_id",
            "player_id",
            "player_display_name",
            "position",
            "team",
            "pfr_id",
        ]
    ].to_csv(
        UNMATCHED_SNAPS_CSV,
        index=False,
    )

    print(
        f"Rows with offensive snap data: "
        f"{merged['offense_snaps'].notna().sum()}"
    )

    print(
        f"Rows without offensive snap data: "
        f"{len(unmatched)}"
    )

    print(
        f"Snap audit: "
        f"{UNMATCHED_SNAPS_CSV}"
    )

    return merged


# =========================================================
# INJURY ATTACHMENT
# =========================================================

def attach_injuries(
    df,
    injury_df,
):

    section(
        "ATTACHING INJURY CONTEXT"
    )

    injuries = (
        injury_df[
            [
                "season",
                "week",
                "game_type",
                "team",
                "gsis_id",
                "report_primary_injury",
                "report_status",
                "practice_status",
            ]
        ]
        .drop_duplicates(
            subset=[
                "season",
                "week",
                "game_type",
                "team",
                "gsis_id",
            ],
            keep="last",
        )
    )

    merged = df.merge(
        injuries,
        how="left",
        left_on=[
            "season",
            "week",
            "season_type",
            "team",
            "player_id",
        ],
        right_on=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
        ],
        validate="many_to_one",
        suffixes=(
            "",
            "_injury",
        ),
    )

    print(
        f"Rows with injury reports: "
        f"{merged['report_status'].notna().sum()}"
    )

    return merged


# =========================================================
# CORE USAGE METRICS
# =========================================================

def calculate_usage_metrics(
    df,
):

    section(
        "CALCULATING USAGE METRICS"
    )

    numeric_columns = [
        "fanduel_points",
        "offense_snaps",
        "offense_pct",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "target_share",
        "air_yards_share",
        "wopr",
    ]

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = safe_numeric(
            df[column]
        )

    # -----------------------------------------------------
    # RAW ROLE
    # -----------------------------------------------------

    df["touches"] = (
        df["carries"]
        +
        df["receptions"]
    )

    df["opportunities"] = (
        df["carries"]
        +
        df["targets"]
    )

    df[
        "yards_from_scrimmage"
    ] = (
        df["rushing_yards"]
        +
        df["receiving_yards"]
    )

    # -----------------------------------------------------
    # PER-SNAP
    # -----------------------------------------------------

    df[
        "carries_per_snap"
    ] = safe_divide(
        df["carries"],
        df["offense_snaps"],
    )

    df[
        "targets_per_snap"
    ] = safe_divide(
        df["targets"],
        df["offense_snaps"],
    )

    df[
        "touches_per_snap"
    ] = safe_divide(
        df["touches"],
        df["offense_snaps"],
    )

    # -----------------------------------------------------
    # EFFICIENCY
    # -----------------------------------------------------

    df[
        "fanduel_per_snap"
    ] = safe_divide(
        df["fanduel_points"],
        df["offense_snaps"],
    )

    df[
        "fanduel_per_touch"
    ] = safe_divide(
        df["fanduel_points"],
        df["touches"],
    )

    df[
        "yards_per_opportunity"
    ] = safe_divide(
        df[
            "yards_from_scrimmage"
        ],
        df["opportunities"],
    )

    # -----------------------------------------------------
    # STATUS FLAGS
    #
    # A missing injury row is not treated as injured.
    # -----------------------------------------------------

    report_upper = (
        df["report_status"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    df[
        "injury_flag"
    ] = (
        report_upper
        != ""
    ).astype(int)

    inactive_statuses = {
        "OUT",
        "DOUBTFUL",
        "INACTIVE",
    }

    df[
        "active_flag"
    ] = (
        ~report_upper.isin(
            inactive_statuses
        )
    ).astype(int)

    print(
        "Core usage metrics calculated."
    )

    return df


# =========================================================
# ROLLING ROLE FEATURES
# =========================================================

def calculate_rolling_features(
    df,
):

    section(
        "CALCULATING ROLLING ROLE FEATURES"
    )

    # -----------------------------------------------------
    # We need chronological ordering.
    # game_id already identifies the specific game, but
    # season/week gives us deterministic historical order.
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "player_id",
            "season",
            "week",
            "game_id",
        ]
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Composite opportunity signal.
    #
    # This intentionally measures workload rather than
    # fantasy production.
    # -----------------------------------------------------

    df[
        "usage_value"
    ] = (
        df["carries"]
        +
        df["targets"]
    )

    group = df.groupby(
        "player_id",
        group_keys=False,
    )

    # -----------------------------------------------------
    # PRIOR THREE GAMES ONLY
    #
    # shift(1) is critical.
    # The current game's outcome cannot be allowed into
    # its own pre-game historical trend.
    # -----------------------------------------------------

    df[
        "usage_3g"
    ] = group[
        "usage_value"
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
    )

    df[
        "snap_pct_3g"
    ] = group[
        "offense_pct"
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
    )

    df[
        "target_3g"
    ] = group[
        "targets"
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
    )

    df[
        "carry_3g"
    ] = group[
        "carries"
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
    )

    df[
        "fanduel_3g"
    ] = group[
        "fanduel_points"
    ].transform(
        lambda s:
            s.shift(1)
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
    )

    # -----------------------------------------------------
    # PREVIOUS GAME VALUES
    # -----------------------------------------------------

    df[
        "previous_usage"
    ] = group[
        "usage_value"
    ].shift(1)

    df[
        "previous_snap_pct"
    ] = group[
        "offense_pct"
    ].shift(1)

    df[
        "previous_targets"
    ] = group[
        "targets"
    ].shift(1)

    df[
        "previous_carries"
    ] = group[
        "carries"
    ].shift(1)

    # -----------------------------------------------------
    # CURRENT GAME VS PRIOR 3-GAME BASELINE
    #
    # These are post-game descriptive role-change metrics.
    # When we later build predictive features, we will use
    # their lagged versions only.
    # -----------------------------------------------------

    df[
        "usage_delta"
    ] = (
        df["usage_value"]
        -
        df["usage_3g"]
    )

    df[
        "snap_delta"
    ] = (
        df["offense_pct"]
        -
        df["snap_pct_3g"]
    )

    df[
        "target_delta"
    ] = (
        df["targets"]
        -
        df["target_3g"]
    )

    df[
        "carry_delta"
    ] = (
        df["carries"]
        -
        df["carry_3g"]
    )

    # -----------------------------------------------------
    # FLAGS
    #
    # These are descriptive game-role classifications,
    # not projections.
    # -----------------------------------------------------

    df[
        "role_expansion_flag"
    ] = (
        (
            df["usage_delta"] >= 4.0
        )
        |
        (
            df["snap_delta"] >= 0.15
        )
    ).astype(int)

    df[
        "role_decline_flag"
    ] = (
        (
            df["usage_delta"] <= -4.0
        )
        |
        (
            df["snap_delta"] <= -0.15
        )
    ).astype(int)

    df[
        "high_usage_flag"
    ] = (
        df["opportunities"] >= 15.0
    ).astype(int)

    df[
        "starter_usage_flag"
    ] = (
        (
            df["offense_pct"] >= 0.60
        )
        |
        (
            df["offense_snaps"] >= 40.0
        )
    ).astype(int)

    # -----------------------------------------------------
    # First career observation has no history.
    # Keep prior metrics at zero for storage simplicity.
    # -----------------------------------------------------

    rolling_columns = [
        "usage_3g",
        "snap_pct_3g",
        "target_3g",
        "carry_3g",
        "fanduel_3g",
        "usage_delta",
        "snap_delta",
        "target_delta",
        "carry_delta",
    ]

    for column in rolling_columns:

        df[column] = (
            df[column]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
        )

    print(
        "Rolling three-game role "
        "features calculated."
    )

    return df


# =========================================================
# INSERT DATABASE
# =========================================================

def store_usage(
    df,
):

    section(
        "STORING PLAYER WEEKLY USAGE"
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

        "fanduel_points",

        "offense_snaps",
        "offense_pct",

        "carries",
        "targets",
        "receptions",

        "rushing_yards",
        "receiving_yards",

        "rushing_tds",
        "receiving_tds",

        "target_share",
        "air_yards_share",
        "wopr",

        "touches",
        "opportunities",

        "yards_from_scrimmage",

        "carries_per_snap",
        "targets_per_snap",
        "touches_per_snap",

        "fanduel_per_snap",
        "fanduel_per_touch",
        "yards_per_opportunity",

        "report_status",
        "practice_status",
        "primary_injury",

        "active_flag",
        "injury_flag",

        "usage_3g",
        "snap_pct_3g",
        "target_3g",
        "carry_3g",
        "fanduel_3g",

        "usage_delta",
        "snap_delta",
        "target_delta",
        "carry_delta",

        "role_expansion_flag",
        "role_decline_flag",

        "high_usage_flag",
        "starter_usage_flag",

        "updated_at",
    ]

    insert_sql = f"""
        INSERT INTO player_weekly_usage (
            {", ".join(columns)}
        )

        VALUES (
            {", ".join(["?"] * len(columns))}
        )
    """

    rows = []

    for _, row in df.iterrows():

        values = [

            row["game_id"],
            int(row["season"]),
            int(row["week"]),
            row["season_type"],

            row["identity_key"],
            row["player_id"],

            row["player_name"],
            row["player_display_name"],

            row["position"],
            row["position_group"],

            row["team"],
            row["opponent_team"],

            float(
                row["fanduel_points"]
            ),

            float(
                row["offense_snaps"]
            ),

            float(
                row["offense_pct"]
            ),

            float(
                row["carries"]
            ),

            float(
                row["targets"]
            ),

            float(
                row["receptions"]
            ),

            float(
                row["rushing_yards"]
            ),

            float(
                row["receiving_yards"]
            ),

            float(
                row["rushing_tds"]
            ),

            float(
                row["receiving_tds"]
            ),

            float(
                row["target_share"]
            ),

            float(
                row["air_yards_share"]
            ),

            float(
                row["wopr"]
            ),

            float(
                row["touches"]
            ),

            float(
                row["opportunities"]
            ),

            float(
                row[
                    "yards_from_scrimmage"
                ]
            ),

            float(
                row["carries_per_snap"]
            ),

            float(
                row["targets_per_snap"]
            ),

            float(
                row["touches_per_snap"]
            ),

            float(
                row["fanduel_per_snap"]
            ),

            float(
                row["fanduel_per_touch"]
            ),

            float(
                row[
                    "yards_per_opportunity"
                ]
            ),

            row["report_status"],
            row["practice_status"],
            row[
                "report_primary_injury"
            ],

            int(
                row["active_flag"]
            ),

            int(
                row["injury_flag"]
            ),

            float(
                row["usage_3g"]
            ),

            float(
                row["snap_pct_3g"]
            ),

            float(
                row["target_3g"]
            ),

            float(
                row["carry_3g"]
            ),

            float(
                row["fanduel_3g"]
            ),

            float(
                row["usage_delta"]
            ),

            float(
                row["snap_delta"]
            ),

            float(
                row["target_delta"]
            ),

            float(
                row["carry_delta"]
            ),

            int(
                row[
                    "role_expansion_flag"
                ]
            ),

            int(
                row[
                    "role_decline_flag"
                ]
            ),

            int(
                row[
                    "high_usage_flag"
                ]
            ),

            int(
                row[
                    "starter_usage_flag"
                ]
            ),

            updated_at,
        ]

        rows.append(
            tuple(values)
        )

    with get_connection() as conn:

        conn.executemany(
            insert_sql,
            rows,
        )

    print(
        f"Rows inserted: {len(rows)}"
    )


# =========================================================
# DATABASE AUDIT
# =========================================================

def audit_usage():

    section(
        "PLAYER USAGE DATABASE AUDIT"
    )

    with get_connection() as conn:

        total = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_weekly_usage
            """
        ).fetchone()[0]

        missing_identity = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_weekly_usage

            WHERE identity_key IS NULL
            """
        ).fetchone()[0]

        missing_game = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_weekly_usage u

            LEFT JOIN games g
                ON u.game_id = g.game_id

            WHERE g.game_id IS NULL
            """
        ).fetchone()[0]

        duplicate_rows = conn.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    game_id,
                    player_id,
                    team,
                    COUNT(*) AS n

                FROM player_weekly_usage

                GROUP BY
                    game_id,
                    player_id,
                    team

                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        negative_snaps = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_weekly_usage

            WHERE offense_snaps < 0
            """
        ).fetchone()[0]

        invalid_snap_pct = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_weekly_usage

            WHERE
                offense_pct < 0
                OR offense_pct > 1.01
            """
        ).fetchone()[0]

    print(
        f"Usage rows: {total}"
    )

    print(
        f"Missing identity: "
        f"{missing_identity}"
    )

    print(
        f"Unmatched game_id: "
        f"{missing_game}"
    )

    print(
        f"Duplicate player-game rows: "
        f"{duplicate_rows}"
    )

    print(
        f"Negative snap rows: "
        f"{negative_snaps}"
    )

    print(
        f"Invalid snap-percent rows: "
        f"{invalid_snap_pct}"
    )

    problems = (
        missing_identity
        +
        missing_game
        +
        duplicate_rows
        +
        negative_snaps
        +
        invalid_snap_pct
    )

    if problems == 0:

        print()
        print(
            "PASS: player usage table "
            "passed structural audits."
        )

    else:

        raise RuntimeError(
            "Player usage structural "
            "audit failed."
        )


# =========================================================
# SUMMARY
# =========================================================

def print_usage_summary():

    section(
        "PLAYER USAGE SUMMARY"
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
                    AVG(fanduel_points),
                    2
                ) AS avg_fd,

                ROUND(
                    AVG(offense_pct),
                    3
                ) AS avg_snap_pct,

                ROUND(
                    AVG(opportunities),
                    2
                ) AS avg_opportunities,

                SUM(
                    role_expansion_flag
                ) AS role_expansions,

                SUM(
                    high_usage_flag
                ) AS high_usage_games

            FROM player_weekly_usage

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
# TOP ROLE GAMES
# =========================================================

def print_top_usage_games():

    section(
        "TOP HISTORICAL USAGE GAMES"
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

                offense_pct,
                carries,
                targets,
                opportunities,

                fanduel_points

            FROM player_weekly_usage

            WHERE
                position IN (
                    'RB',
                    'WR',
                    'TE'
                )

            ORDER BY
                opportunities DESC,
                fanduel_points DESC

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

def export_usage():

    section(
        "PLAYER USAGE EXPORT"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM player_weekly_usage

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
        USAGE_CSV,
        index=False,
    )

    df.to_parquet(
        USAGE_PARQUET,
        index=False,
    )

    print(
        f"Rows exported: "
        f"{len(df)}"
    )

    print(
        f"CSV:     {USAGE_CSV}"
    )

    print(
        f"Parquet: {USAGE_PARQUET}"
    )


# =========================================================
# MAIN
# =========================================================

def run_player_usage_build():

    section(
        "NFL PLAYER USAGE ENGINE"
    )

    # -----------------------------------------------------
    # DERIVED TABLE
    # -----------------------------------------------------

    rebuild_usage_table()

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    stats_df = load_player_stats()

    identity_df = load_identity()

    snap_df = load_snaps()

    injury_df = load_injuries()

    # -----------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------

    df = attach_identity(
        stats_df,
        identity_df,
    )

    df = attach_snaps(
        df,
        snap_df,
    )

    df = attach_injuries(
        df,
        injury_df,
    )

    # -----------------------------------------------------
    # MODELING FEATURES
    # -----------------------------------------------------

    df = calculate_usage_metrics(
        df
    )

    df = calculate_rolling_features(
        df
    )

    # -----------------------------------------------------
    # STORE
    # -----------------------------------------------------

    store_usage(
        df
    )

    # -----------------------------------------------------
    # AUDIT
    # -----------------------------------------------------

    audit_usage()

    print_usage_summary()

    print_top_usage_games()

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    export_usage()

    section(
        "PLAYER USAGE BUILD SUCCESSFUL"
    )


if __name__ == "__main__":
    run_player_usage_build()
