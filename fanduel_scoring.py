from datetime import datetime, timezone

import pandas as pd
import nflreadpy as nfl

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import (
    get_connection,
    initialize_database,
)


# =========================================================
# SETTINGS
# =========================================================

HISTORICAL_SEASONS = [
    2023,
    2024,
    2025,
]

PLAYER_CSV = CSV_DIR / "nfl_player_game_stats.csv"
PLAYER_PARQUET = PARQUET_DIR / "nfl_player_game_stats.parquet"

SCORING_AUDIT_CSV = CSV_DIR / "audit_fanduel_scoring.csv"
TOP_SCORES_CSV = CSV_DIR / "fanduel_top_scores.csv"


# =========================================================
# FANDUEL SCORING CONFIGURATION
# =========================================================
#
# Project scoring:
#
# Passing yards              0.04
# Passing TD                 4
# Passing interception      -1
#
# Rushing yards              0.10
# Rushing TD                 6
#
# Receiving yards            0.10
# Reception                  0.50
# Receiving TD               6
#
# Two-point conversion       2
# Special teams TD           6
# Fumble lost               -2
#
# Yardage bonuses
# 300+ passing yards          +3
# 100+ rushing yards          +3
# 100+ receiving yards        +3
#
# =========================================================

PASS_YARD_POINTS = 0.04
PASS_TD_POINTS = 4.0
PASS_INT_POINTS = -1.0
PASSING_BONUS_YARDS = 300.0

RUSH_YARD_POINTS = 0.10
RUSH_TD_POINTS = 6.0
RUSHING_BONUS_YARDS = 100.0

RECEPTION_POINTS = 0.50
RECEIVING_YARD_POINTS = 0.10
RECEIVING_TD_POINTS = 6.0
RECEIVING_BONUS_YARDS = 100.0

YARDAGE_BONUS_POINTS = 3.0

TWO_POINT_POINTS = 2.0

SPECIAL_TEAMS_TD_POINTS = 6.0

FUMBLE_LOST_POINTS = -2.0


# =========================================================
# REQUIRED SOURCE COLUMNS
# =========================================================

REQUIRED_SCORING_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "player_id",
    "team",

    "passing_yards",
    "passing_tds",
    "passing_interceptions",

    "rushing_yards",
    "rushing_tds",

    "receptions",
    "receiving_yards",
    "receiving_tds",

    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",

    "special_teams_tds",

    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]


# =========================================================
# HELPERS
# =========================================================

def safe_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def safe_int(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def safe_float(value, default=0.0):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def get_value(row, column, default=None):
    if column not in row.index:
        return default

    value = row[column]

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def get_table_columns(conn, table_name):
    rows = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return {
        row["name"]
        for row in rows
    }


# =========================================================
# DATABASE MIGRATION
# =========================================================

def initialize_scoring_columns():
    initialize_database()

    with get_connection() as conn:

        columns = get_table_columns(
            conn,
            "player_game_stats",
        )

        migrations = {
            "passing_interceptions": "REAL",
            "sack_fumbles_lost": "REAL",
            "total_fumbles_lost": "REAL",
            "fanduel_points": "REAL",
            "fanduel_points_verified": "INTEGER DEFAULT 0",
            "fanduel_scored_at": "TEXT",
        }

        for column, datatype in migrations.items():

            if column not in columns:

                print(
                    f"Migrating player_game_stats: "
                    f"adding {column}"
                )

                conn.execute(
                    f"""
                    ALTER TABLE player_game_stats
                    ADD COLUMN {column} {datatype}
                    """
                )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_player_fanduel_points
            ON player_game_stats(fanduel_points)
            """
        )


# =========================================================
# DOWNLOAD SOURCE STATS
# =========================================================

def download_scoring_source(seasons):
    print()
    print("=" * 70)
    print("DOWNLOADING FANDUEL SCORING SOURCE")
    print("=" * 70)

    stats = nfl.load_player_stats(
        seasons=seasons,
        summary_level="week",
    )

    df = stats.to_pandas()

    print(
        f"Downloaded {len(df)} source rows "
        f"for seasons {seasons}"
    )

    missing_columns = [
        column
        for column in REQUIRED_SCORING_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Required scoring columns are missing "
            "from nflverse:\n"
            + "\n".join(missing_columns)
        )

    print(
        "All required FanDuel scoring "
        "source columns are present."
    )

    return df


# =========================================================
# FANDUEL CALCULATION
# =========================================================

def calculate_fanduel_points(row):

    passing_yards = safe_float(
        get_value(row, "passing_yards")
    )

    passing_tds = safe_float(
        get_value(row, "passing_tds")
    )

    passing_interceptions = safe_float(
        get_value(
            row,
            "passing_interceptions"
        )
    )

    rushing_yards = safe_float(
        get_value(row, "rushing_yards")
    )

    rushing_tds = safe_float(
        get_value(row, "rushing_tds")
    )

    receptions = safe_float(
        get_value(row, "receptions")
    )

    receiving_yards = safe_float(
        get_value(row, "receiving_yards")
    )

    receiving_tds = safe_float(
        get_value(row, "receiving_tds")
    )

    passing_2pt = safe_float(
        get_value(
            row,
            "passing_2pt_conversions"
        )
    )

    rushing_2pt = safe_float(
        get_value(
            row,
            "rushing_2pt_conversions"
        )
    )

    receiving_2pt = safe_float(
        get_value(
            row,
            "receiving_2pt_conversions"
        )
    )

    special_teams_tds = safe_float(
        get_value(
            row,
            "special_teams_tds"
        )
    )

    sack_fumbles_lost = safe_float(
        get_value(
            row,
            "sack_fumbles_lost"
        )
    )

    rushing_fumbles_lost = safe_float(
        get_value(
            row,
            "rushing_fumbles_lost"
        )
    )

    receiving_fumbles_lost = safe_float(
        get_value(
            row,
            "receiving_fumbles_lost"
        )
    )

    total_fumbles_lost = (
        sack_fumbles_lost
        + rushing_fumbles_lost
        + receiving_fumbles_lost
    )

    points = 0.0

    # Passing
    points += (
        passing_yards
        * PASS_YARD_POINTS
    )

    points += (
        passing_tds
        * PASS_TD_POINTS
    )

    points += (
        passing_interceptions
        * PASS_INT_POINTS
    )

    # Rushing
    points += (
        rushing_yards
        * RUSH_YARD_POINTS
    )

    points += (
        rushing_tds
        * RUSH_TD_POINTS
    )

    # Receiving
    points += (
        receptions
        * RECEPTION_POINTS
    )

    points += (
        receiving_yards
        * RECEIVING_YARD_POINTS
    )

    points += (
        receiving_tds
        * RECEIVING_TD_POINTS
    )

    # Two-point conversions
    points += (
        passing_2pt
        + rushing_2pt
        + receiving_2pt
    ) * TWO_POINT_POINTS

    # Special teams touchdowns
    points += (
        special_teams_tds
        * SPECIAL_TEAMS_TD_POINTS
    )

    # Lost fumbles
    points += (
        total_fumbles_lost
        * FUMBLE_LOST_POINTS
    )

    return (
        round(points, 2),
        passing_interceptions,
        sack_fumbles_lost,
        total_fumbles_lost,
    )


def run_scoring_regression_tests():
    print()
    print("=" * 70)
    print("FANDUEL SCORING REGRESSION TESTS")
    print("=" * 70)

    cases = [
        (
            "299 pass yards",
            {"passing_yards": 299},
            11.96,
        ),
        (
            "300 pass yards: no bonus",
            {"passing_yards": 300},
            12.00,
        ),
        (
            "99 rush yards",
            {"rushing_yards": 99},
            9.90,
        ),
        (
            "100 rush yards: no bonus",
            {"rushing_yards": 100},
            10.00,
        ),
        (
            "99 receiving yards + 5 catches",
            {"receiving_yards": 99, "receptions": 5},
            12.40,
        ),
        (
            "100 receiving yards + 5 catches: no bonus",
            {"receiving_yards": 100, "receptions": 5},
            12.50,
        ),
        (
            "yardage thresholds do not award bonuses",
            {
                "passing_yards": 300,
                "rushing_yards": 100,
                "receiving_yards": 100,
            },
            32.00,
        ),
    ]

    failures = []

    for label, values, expected in cases:
        row = pd.Series(values, dtype=object)
        actual = calculate_fanduel_points(row)[0]

        if abs(actual - expected) > 1e-9:
            failures.append(
                f"{label}: expected {expected:.2f}, got {actual:.2f}"
            )
        else:
            print(f"PASS: {label} -> {actual:.2f}")

    if failures:
        raise RuntimeError(
            "FanDuel scoring regression failure:\n"
            + "\n".join(failures)
        )

    print()
    print("PASS: FanDuel offensive scoring regression tests passed.")


# =========================================================
# DATABASE UPDATE
# =========================================================

def update_fanduel_scores(df):
    print()
    print("=" * 70)
    print("CALCULATING FANDUEL POINTS")
    print("=" * 70)

    scored_at = datetime.now(
        timezone.utc
    ).isoformat()

    sql = """
        UPDATE player_game_stats

        SET
            interceptions = ?,
            passing_interceptions = ?,
            sack_fumbles_lost = ?,
            total_fumbles_lost = ?,
            fanduel_points = ?,
            fanduel_points_verified = 1,
            fanduel_scored_at = ?

        WHERE
            game_id = ?
            AND player_id = ?
            AND team = ?
    """

    updates = []

    source_rows_skipped = 0

    for _, row in df.iterrows():

        game_id = safe_text(
            get_value(row, "game_id")
        )

        player_id = safe_text(
            get_value(row, "player_id")
        )

        team = safe_text(
            get_value(row, "team")
        )

        if (
            game_id is None
            or player_id is None
            or team is None
        ):
            source_rows_skipped += 1
            continue

        (
            fanduel_points,
            passing_interceptions,
            sack_fumbles_lost,
            total_fumbles_lost,
        ) = calculate_fanduel_points(row)

        updates.append(
            (
                passing_interceptions,
                passing_interceptions,
                sack_fumbles_lost,
                total_fumbles_lost,
                fanduel_points,
                scored_at,
                game_id,
                player_id,
                team,
            )
        )

    with get_connection() as conn:

        before_changes = conn.total_changes

        conn.executemany(
            sql,
            updates,
        )

        rows_updated = (
            conn.total_changes
            - before_changes
        )

    print(
        f"Scoring source rows prepared: "
        f"{len(updates)}"
    )

    print(
        f"Source rows skipped: "
        f"{source_rows_skipped}"
    )

    print(
        f"Database rows updated: "
        f"{rows_updated}"
    )


# =========================================================
# SCORING AUDIT
# =========================================================

def audit_scoring():
    print()
    print("=" * 70)
    print("FANDUEL SCORING AUDIT")
    print("=" * 70)

    with get_connection() as conn:

        total_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_game_stats
            """
        ).fetchone()[0]

        verified_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_game_stats
            WHERE fanduel_points_verified = 1
            """
        ).fetchone()[0]

        unverified_df = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                player_id,
                player_name,
                position,
                team,
                opponent_team,
                fanduel_points,
                fanduel_points_verified

            FROM player_game_stats

            WHERE
                fanduel_points_verified IS NULL
                OR fanduel_points_verified != 1

            ORDER BY
                season,
                week,
                game_id,
                team,
                player_name
            """,
            conn,
        )

        null_scores = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_game_stats
            WHERE fanduel_points IS NULL
            """
        ).fetchone()[0]

    unverified_df.to_csv(
        SCORING_AUDIT_CSV,
        index=False,
    )

    print(
        f"Total stored player rows: "
        f"{total_rows}"
    )

    print(
        f"Verified FanDuel rows: "
        f"{verified_rows}"
    )

    print(
        f"NULL FanDuel scores: "
        f"{null_scores}"
    )

    print(
        f"Unverified rows: "
        f"{len(unverified_df)}"
    )

    if (
        total_rows == verified_rows
        and null_scores == 0
        and len(unverified_df) == 0
    ):

        print()
        print(
            "PASS: every stored player-game row "
            "has a verified FanDuel score."
        )

    else:

        print()
        print(
            "WARNING: some stored rows were "
            "not scored."
        )

        print(
            f"Audit file: "
            f"{SCORING_AUDIT_CSV}"
        )


# =========================================================
# SANITY CHECK
# =========================================================

def print_scoring_sanity_check():

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                season,
                COUNT(*) AS players,

                ROUND(
                    AVG(fanduel_points),
                    2
                ) AS avg_points,

                ROUND(
                    MAX(fanduel_points),
                    2
                ) AS max_points

            FROM player_game_stats

            WHERE fanduel_points_verified = 1

            GROUP BY season

            ORDER BY season
            """
        ).fetchall()

    print()
    print("=" * 70)
    print("FANDUEL SCORING SUMMARY")
    print("=" * 70)

    for row in rows:

        print(
            f"{row['season']}: "
            f"{row['players']} rows | "
            f"AVG {row['avg_points']} | "
            f"MAX {row['max_points']}"
        )


# =========================================================
# TOP SCORES
# =========================================================

def export_top_scores():

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT
                g.game_date,

                p.game_id,
                p.season,
                p.week,

                p.player_id,
                p.player_name,
                p.player_display_name,
                p.position,

                p.team,
                p.opponent_team,

                p.passing_yards,
                p.passing_tds,
                p.passing_interceptions,

                p.carries,
                p.rushing_yards,
                p.rushing_tds,

                p.targets,
                p.receptions,
                p.receiving_yards,
                p.receiving_tds,

                p.total_fumbles_lost,
                p.special_teams_tds,

                p.fanduel_points

            FROM player_game_stats p

            JOIN games g
                ON p.game_id = g.game_id

            WHERE
                p.fanduel_points_verified = 1

            ORDER BY
                p.fanduel_points DESC,
                g.game_date DESC
            """,
            conn,
        )

    df.to_csv(
        TOP_SCORES_CSV,
        index=False,
    )

    print()
    print(
        f"Top-score export: "
        f"{TOP_SCORES_CSV}"
    )

    print()
    print("=" * 70)
    print("TOP 20 FANDUEL PERFORMANCES")
    print("=" * 70)

    display_columns = [
        "season",
        "week",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "fanduel_points",
    ]

    print(
        df[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )


# =========================================================
# REFRESH MAIN EXPORTS
# =========================================================

def refresh_player_exports():

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM player_game_stats

            ORDER BY
                season,
                week,
                game_id,
                team,
                player_name
            """,
            conn,
        )

    df.to_csv(
        PLAYER_CSV,
        index=False,
    )

    df.to_parquet(
        PLAYER_PARQUET,
        index=False,
    )

    print()
    print("=" * 70)
    print("PLAYER EXPORT REFRESH")
    print("=" * 70)

    print(
        f"Rows exported: {len(df)}"
    )

    print(
        f"CSV:     {PLAYER_CSV}"
    )

    print(
        f"Parquet: {PLAYER_PARQUET}"
    )


# =========================================================
# MAIN
# =========================================================

def run_fanduel_scoring(
    seasons=None,
):

    if seasons is None:
        seasons = HISTORICAL_SEASONS

    print()
    print("=" * 70)
    print("NFL FANDUEL SCORING ENGINE")
    print("=" * 70)

    run_scoring_regression_tests()

    initialize_scoring_columns()

    df = download_scoring_source(
        seasons
    )

    update_fanduel_scores(
        df
    )

    audit_scoring()

    print_scoring_sanity_check()

    export_top_scores()

    refresh_player_exports()

    print()
    print("=" * 70)
    print("FANDUEL SCORING UPDATE SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    run_fanduel_scoring()
