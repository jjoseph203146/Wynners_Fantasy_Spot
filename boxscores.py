from datetime import datetime, timezone

import pandas as pd
import nflreadpy as nfl

from config import (
    PARQUET_DIR,
    CSV_DIR,
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

PLAYER_PARQUET = PARQUET_DIR / "nfl_player_game_stats.parquet"
PLAYER_CSV = CSV_DIR / "nfl_player_game_stats.csv"

TEAM_PARQUET = PARQUET_DIR / "nfl_team_game_stats.parquet"
TEAM_CSV = CSV_DIR / "nfl_team_game_stats.csv"

PLAYER_SKIPPED_AUDIT = CSV_DIR / "audit_player_stats_skipped.csv"
PLAYER_GAME_AUDIT = CSV_DIR / "audit_player_game_id_unmatched.csv"
TEAM_GAME_AUDIT = CSV_DIR / "audit_team_game_id_unmatched.csv"


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


def safe_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
# DATABASE TABLES / MIGRATION
# =========================================================

def initialize_boxscore_tables():
    initialize_database()

    with get_connection() as conn:

        # -------------------------------------------------
        # PLAYER GAME STATS
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS player_game_stats (

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                season_type TEXT,

                player_id TEXT NOT NULL,
                player_name TEXT,

                player_display_name TEXT,
                position TEXT,
                position_group TEXT,

                team TEXT,
                opponent_team TEXT,

                completions REAL,
                attempts REAL,
                passing_yards REAL,
                passing_tds REAL,
                interceptions REAL,
                sacks_suffered REAL,
                sack_yards_lost REAL,
                passing_air_yards REAL,
                passing_yards_after_catch REAL,
                passing_first_downs REAL,
                passing_epa REAL,
                passing_2pt_conversions REAL,

                carries REAL,
                rushing_yards REAL,
                rushing_tds REAL,
                rushing_fumbles REAL,
                rushing_fumbles_lost REAL,
                rushing_first_downs REAL,
                rushing_epa REAL,
                rushing_2pt_conversions REAL,

                receptions REAL,
                targets REAL,
                receiving_yards REAL,
                receiving_tds REAL,
                receiving_fumbles REAL,
                receiving_fumbles_lost REAL,
                receiving_air_yards REAL,
                receiving_yards_after_catch REAL,
                receiving_first_downs REAL,
                receiving_epa REAL,
                receiving_2pt_conversions REAL,

                racr REAL,
                target_share REAL,
                air_yards_share REAL,
                wopr REAL,

                special_teams_tds REAL,

                fantasy_points REAL,
                fantasy_points_ppr REAL,

                updated_at TEXT,

                PRIMARY KEY (
                    season,
                    week,
                    season_type,
                    player_id,
                    team
                )
            )
            """
        )

        # -------------------------------------------------
        # TEAM GAME STATS
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS team_game_stats (

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                season_type TEXT,

                team TEXT NOT NULL,
                opponent_team TEXT,

                completions REAL,
                attempts REAL,
                passing_yards REAL,
                passing_tds REAL,
                interceptions REAL,
                sacks_suffered REAL,
                sack_yards_lost REAL,
                passing_air_yards REAL,
                passing_yards_after_catch REAL,
                passing_first_downs REAL,
                passing_epa REAL,

                carries REAL,
                rushing_yards REAL,
                rushing_tds REAL,
                rushing_fumbles REAL,
                rushing_fumbles_lost REAL,
                rushing_first_downs REAL,
                rushing_epa REAL,

                receptions REAL,
                targets REAL,
                receiving_yards REAL,
                receiving_tds REAL,
                receiving_fumbles REAL,
                receiving_fumbles_lost REAL,
                receiving_air_yards REAL,
                receiving_yards_after_catch REAL,
                receiving_first_downs REAL,
                receiving_epa REAL,

                special_teams_tds REAL,

                fantasy_points REAL,
                fantasy_points_ppr REAL,

                updated_at TEXT,

                PRIMARY KEY (
                    season,
                    week,
                    season_type,
                    team
                )
            )
            """
        )

        # -------------------------------------------------
        # SAFE MIGRATION: ADD GAME_ID
        # -------------------------------------------------

        player_columns = get_table_columns(
            conn,
            "player_game_stats"
        )

        if "game_id" not in player_columns:
            print(
                "Migrating player_game_stats: "
                "adding game_id"
            )

            conn.execute(
                """
                ALTER TABLE player_game_stats
                ADD COLUMN game_id TEXT
                """
            )

        team_columns = get_table_columns(
            conn,
            "team_game_stats"
        )

        if "game_id" not in team_columns:
            print(
                "Migrating team_game_stats: "
                "adding game_id"
            )

            conn.execute(
                """
                ALTER TABLE team_game_stats
                ADD COLUMN game_id TEXT
                """
            )

        # -------------------------------------------------
        # INDEXES
        # -------------------------------------------------

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_game
            ON player_game_stats(game_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_player
            ON player_game_stats(player_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_name
            ON player_game_stats(player_name)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_team
            ON player_game_stats(team)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_stats_season_week
            ON player_game_stats(season, week)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_team_stats_game
            ON team_game_stats(game_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_team_stats_team
            ON team_game_stats(team)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_team_stats_season_week
            ON team_game_stats(season, week)
            """
        )


# =========================================================
# DOWNLOAD
# =========================================================

def download_player_stats(seasons):
    print()
    print("=" * 70)
    print("DOWNLOADING PLAYER GAME STATS")
    print("=" * 70)

    stats = nfl.load_player_stats(
        seasons=seasons,
        summary_level="week",
    )

    df = stats.to_pandas()

    print(
        f"Downloaded {len(df)} player-game rows "
        f"for seasons {seasons}"
    )

    return df


def download_team_stats(seasons):
    print()
    print("=" * 70)
    print("DOWNLOADING TEAM GAME STATS")
    print("=" * 70)

    stats = nfl.load_team_stats(
        seasons=seasons,
        summary_level="week",
    )

    df = stats.to_pandas()

    print(
        f"Downloaded {len(df)} team-game rows "
        f"for seasons {seasons}"
    )

    return df


# =========================================================
# PLAYER DATABASE LOAD
# =========================================================

def update_player_stats(df):
    updated_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "season",
        "week",
        "season_type",
        "player_id",
        "player_name",
        "player_display_name",
        "position",
        "position_group",
        "team",
        "opponent_team",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks_suffered",
        "sack_yards_lost",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_epa",
        "passing_2pt_conversions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "rushing_epa",
        "rushing_2pt_conversions",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles",
        "receiving_fumbles_lost",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_first_downs",
        "receiving_epa",
        "receiving_2pt_conversions",
        "racr",
        "target_share",
        "air_yards_share",
        "wopr",
        "special_teams_tds",
        "fantasy_points",
        "fantasy_points_ppr",
        "updated_at",
        "game_id",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    update_columns = [
        column
        for column in columns
        if column not in {
            "season",
            "week",
            "season_type",
            "player_id",
            "team",
        }
    ]

    update_clause = ", ".join(
        f"{column}=excluded.{column}"
        for column in update_columns
    )

    sql = f"""
        INSERT INTO player_game_stats (
            {", ".join(columns)}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT(
            season,
            week,
            season_type,
            player_id,
            team
        )
        DO UPDATE SET
            {update_clause}
    """

    rows = []
    skipped = []

    for _, row in df.iterrows():

        season = safe_int(
            get_value(row, "season")
        )

        week = safe_int(
            get_value(row, "week")
        )

        season_type = safe_text(
            get_value(row, "season_type")
        )

        player_id = safe_text(
            get_value(row, "player_id")
        )

        game_id = safe_text(
            get_value(row, "game_id")
        )

        team = safe_text(
            get_value(row, "team")
        )

        if team is None:
            team = safe_text(
                get_value(row, "recent_team")
            )

        missing = []

        if season is None:
            missing.append("season")

        if week is None:
            missing.append("week")

        if player_id is None:
            missing.append("player_id")

        if team is None:
            missing.append("team")

        if game_id is None:
            missing.append("game_id")

        if missing:

            skipped.append(
                {
                    "season": season,
                    "week": week,
                    "season_type": season_type,
                    "game_id": game_id,
                    "player_id": player_id,
                    "player_name": safe_text(
                        get_value(
                            row,
                            "player_name"
                        )
                    ),
                    "team": team,
                    "opponent_team": safe_text(
                        get_value(
                            row,
                            "opponent_team"
                        )
                    ),
                    "reason": (
                        "missing:"
                        + ",".join(missing)
                    ),
                }
            )

            continue

        values = (
            season,
            week,
            season_type,
            player_id,

            safe_text(
                get_value(
                    row,
                    "player_name"
                )
            ),

            safe_text(
                get_value(
                    row,
                    "player_display_name"
                )
            ),

            safe_text(
                get_value(
                    row,
                    "position"
                )
            ),

            safe_text(
                get_value(
                    row,
                    "position_group"
                )
            ),

            team,

            safe_text(
                get_value(
                    row,
                    "opponent_team"
                )
            ),

            safe_float(get_value(row, "completions")),
            safe_float(get_value(row, "attempts")),
            safe_float(get_value(row, "passing_yards")),
            safe_float(get_value(row, "passing_tds")),
            safe_float(get_value(row, "interceptions")),
            safe_float(get_value(row, "sacks_suffered")),
            safe_float(get_value(row, "sack_yards_lost")),
            safe_float(get_value(row, "passing_air_yards")),
            safe_float(
                get_value(
                    row,
                    "passing_yards_after_catch"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "passing_first_downs"
                )
            ),
            safe_float(get_value(row, "passing_epa")),
            safe_float(
                get_value(
                    row,
                    "passing_2pt_conversions"
                )
            ),

            safe_float(get_value(row, "carries")),
            safe_float(get_value(row, "rushing_yards")),
            safe_float(get_value(row, "rushing_tds")),
            safe_float(get_value(row, "rushing_fumbles")),
            safe_float(
                get_value(
                    row,
                    "rushing_fumbles_lost"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "rushing_first_downs"
                )
            ),
            safe_float(get_value(row, "rushing_epa")),
            safe_float(
                get_value(
                    row,
                    "rushing_2pt_conversions"
                )
            ),

            safe_float(get_value(row, "receptions")),
            safe_float(get_value(row, "targets")),
            safe_float(get_value(row, "receiving_yards")),
            safe_float(get_value(row, "receiving_tds")),
            safe_float(get_value(row, "receiving_fumbles")),
            safe_float(
                get_value(
                    row,
                    "receiving_fumbles_lost"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_air_yards"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_yards_after_catch"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_first_downs"
                )
            ),
            safe_float(get_value(row, "receiving_epa")),
            safe_float(
                get_value(
                    row,
                    "receiving_2pt_conversions"
                )
            ),

            safe_float(get_value(row, "racr")),
            safe_float(get_value(row, "target_share")),
            safe_float(get_value(row, "air_yards_share")),
            safe_float(get_value(row, "wopr")),

            safe_float(
                get_value(
                    row,
                    "special_teams_tds"
                )
            ),

            safe_float(
                get_value(
                    row,
                    "fantasy_points"
                )
            ),

            safe_float(
                get_value(
                    row,
                    "fantasy_points_ppr"
                )
            ),

            updated_at,
            game_id,
        )

        rows.append(values)

    with get_connection() as conn:
        conn.executemany(
            sql,
            rows,
        )

    skipped_df = pd.DataFrame(skipped)

    skipped_df.to_csv(
        PLAYER_SKIPPED_AUDIT,
        index=False,
    )

    print(
        f"Inserted/updated {len(rows)} "
        f"player-game rows."
    )

    print(
        f"Skipped {len(skipped)} "
        f"player rows."
    )

    print(
        f"Skipped audit: "
        f"{PLAYER_SKIPPED_AUDIT}"
    )


# =========================================================
# TEAM DATABASE LOAD
# =========================================================

def update_team_stats(df):
    updated_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "season",
        "week",
        "season_type",
        "team",
        "opponent_team",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks_suffered",
        "sack_yards_lost",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_epa",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "rushing_epa",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles",
        "receiving_fumbles_lost",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_first_downs",
        "receiving_epa",
        "special_teams_tds",
        "fantasy_points",
        "fantasy_points_ppr",
        "updated_at",
        "game_id",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    update_columns = [
        column
        for column in columns
        if column not in {
            "season",
            "week",
            "season_type",
            "team",
        }
    ]

    update_clause = ", ".join(
        f"{column}=excluded.{column}"
        for column in update_columns
    )

    sql = f"""
        INSERT INTO team_game_stats (
            {", ".join(columns)}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT(
            season,
            week,
            season_type,
            team
        )
        DO UPDATE SET
            {update_clause}
    """

    rows = []

    for _, row in df.iterrows():

        season = safe_int(
            get_value(row, "season")
        )

        week = safe_int(
            get_value(row, "week")
        )

        season_type = safe_text(
            get_value(row, "season_type")
        )

        team = safe_text(
            get_value(row, "team")
        )

        game_id = safe_text(
            get_value(row, "game_id")
        )

        if (
            season is None
            or week is None
            or team is None
            or game_id is None
        ):
            continue

        values = (
            season,
            week,
            season_type,
            team,

            safe_text(
                get_value(
                    row,
                    "opponent_team"
                )
            ),

            safe_float(get_value(row, "completions")),
            safe_float(get_value(row, "attempts")),
            safe_float(get_value(row, "passing_yards")),
            safe_float(get_value(row, "passing_tds")),
            safe_float(get_value(row, "interceptions")),
            safe_float(get_value(row, "sacks_suffered")),
            safe_float(get_value(row, "sack_yards_lost")),
            safe_float(get_value(row, "passing_air_yards")),
            safe_float(
                get_value(
                    row,
                    "passing_yards_after_catch"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "passing_first_downs"
                )
            ),
            safe_float(get_value(row, "passing_epa")),

            safe_float(get_value(row, "carries")),
            safe_float(get_value(row, "rushing_yards")),
            safe_float(get_value(row, "rushing_tds")),
            safe_float(get_value(row, "rushing_fumbles")),
            safe_float(
                get_value(
                    row,
                    "rushing_fumbles_lost"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "rushing_first_downs"
                )
            ),
            safe_float(get_value(row, "rushing_epa")),

            safe_float(get_value(row, "receptions")),
            safe_float(get_value(row, "targets")),
            safe_float(get_value(row, "receiving_yards")),
            safe_float(get_value(row, "receiving_tds")),
            safe_float(get_value(row, "receiving_fumbles")),
            safe_float(
                get_value(
                    row,
                    "receiving_fumbles_lost"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_air_yards"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_yards_after_catch"
                )
            ),
            safe_float(
                get_value(
                    row,
                    "receiving_first_downs"
                )
            ),
            safe_float(get_value(row, "receiving_epa")),

            safe_float(
                get_value(
                    row,
                    "special_teams_tds"
                )
            ),

            safe_float(
                get_value(
                    row,
                    "fantasy_points"
                )
            ),

            safe_float(
                get_value(
                    row,
                    "fantasy_points_ppr"
                )
            ),

            updated_at,
            game_id,
        )

        rows.append(values)

    with get_connection() as conn:
        conn.executemany(
            sql,
            rows,
        )

    print(
        f"Inserted/updated {len(rows)} "
        f"team-game rows."
    )


# =========================================================
# GAME-ID AUDIT
# =========================================================

def audit_game_ids():
    print()
    print("=" * 70)
    print("GAME ID RELATIONSHIP AUDIT")
    print("=" * 70)

    with get_connection() as conn:

        player_unmatched = pd.read_sql_query(
            """
            SELECT
                p.game_id,
                p.season,
                p.week,
                p.player_id,
                p.player_name,
                p.team,
                p.opponent_team
            FROM player_game_stats p
            LEFT JOIN games g
                ON p.game_id = g.game_id
            WHERE
                p.game_id IS NOT NULL
                AND g.game_id IS NULL
            ORDER BY
                p.season,
                p.week,
                p.game_id,
                p.player_name
            """,
            conn,
        )

        team_unmatched = pd.read_sql_query(
            """
            SELECT
                t.game_id,
                t.season,
                t.week,
                t.team,
                t.opponent_team
            FROM team_game_stats t
            LEFT JOIN games g
                ON t.game_id = g.game_id
            WHERE
                t.game_id IS NOT NULL
                AND g.game_id IS NULL
            ORDER BY
                t.season,
                t.week,
                t.game_id,
                t.team
            """,
            conn,
        )

        player_null_game_ids = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_game_stats
            WHERE game_id IS NULL
            """
        ).fetchone()[0]

        team_null_game_ids = conn.execute(
            """
            SELECT COUNT(*)
            FROM team_game_stats
            WHERE game_id IS NULL
            """
        ).fetchone()[0]

    player_unmatched.to_csv(
        PLAYER_GAME_AUDIT,
        index=False,
    )

    team_unmatched.to_csv(
        TEAM_GAME_AUDIT,
        index=False,
    )

    print(
        f"Player rows with NULL game_id: "
        f"{player_null_game_ids}"
    )

    print(
        f"Team rows with NULL game_id: "
        f"{team_null_game_ids}"
    )

    print(
        f"Player rows with unmatched game_id: "
        f"{len(player_unmatched)}"
    )

    print(
        f"Team rows with unmatched game_id: "
        f"{len(team_unmatched)}"
    )

    if (
        len(player_unmatched) == 0
        and len(team_unmatched) == 0
        and player_null_game_ids == 0
        and team_null_game_ids == 0
    ):
        print()
        print(
            "PASS: all stored box-score rows "
            "have valid game relationships."
        )
    else:
        print()
        print(
            "WARNING: relationship audit "
            "found issues."
        )

        print(
            f"Player audit: "
            f"{PLAYER_GAME_AUDIT}"
        )

        print(
            f"Team audit: "
            f"{TEAM_GAME_AUDIT}"
        )


# =========================================================
# EXPORTS
# =========================================================

def export_boxscores():

    with get_connection() as conn:

        player_df = pd.read_sql_query(
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

        team_df = pd.read_sql_query(
            """
            SELECT *
            FROM team_game_stats
            ORDER BY
                season,
                week,
                game_id,
                team
            """,
            conn,
        )

    player_df.to_parquet(
        PLAYER_PARQUET,
        index=False,
    )

    player_df.to_csv(
        PLAYER_CSV,
        index=False,
    )

    team_df.to_parquet(
        TEAM_PARQUET,
        index=False,
    )

    team_df.to_csv(
        TEAM_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print("BOXSCORE EXPORTS")
    print("=" * 70)

    print(
        f"Player rows: {len(player_df)}"
    )

    print(
        f"Team rows:   {len(team_df)}"
    )

    print()
    print(f"Player parquet: {PLAYER_PARQUET}")
    print(f"Team parquet:   {TEAM_PARQUET}")


# =========================================================
# SUMMARY
# =========================================================

def print_boxscore_summary():

    with get_connection() as conn:

        player_rows = conn.execute(
            """
            SELECT
                season,
                COUNT(*) AS rows,
                COUNT(DISTINCT game_id) AS games
            FROM player_game_stats
            GROUP BY season
            ORDER BY season
            """
        ).fetchall()

        team_rows = conn.execute(
            """
            SELECT
                season,
                COUNT(*) AS rows,
                COUNT(DISTINCT game_id) AS games
            FROM team_game_stats
            GROUP BY season
            ORDER BY season
            """
        ).fetchall()

    print()
    print("=" * 70)
    print("PLAYER GAME STATS")
    print("=" * 70)

    for row in player_rows:

        print(
            f"{row['season']}: "
            f"{row['rows']} rows | "
            f"{row['games']} games"
        )

    print()
    print("=" * 70)
    print("TEAM GAME STATS")
    print("=" * 70)

    for row in team_rows:

        print(
            f"{row['season']}: "
            f"{row['rows']} rows | "
            f"{row['games']} games"
        )


# =========================================================
# RUN
# =========================================================

def run_boxscore_update(
    seasons=None,
):

    if seasons is None:
        seasons = HISTORICAL_SEASONS

    initialize_boxscore_tables()

    player_df = download_player_stats(
        seasons
    )

    update_player_stats(
        player_df
    )

    team_df = download_team_stats(
        seasons
    )

    update_team_stats(
        team_df
    )

    audit_game_ids()

    export_boxscores()

    print_boxscore_summary()


if __name__ == "__main__":
    run_boxscore_update()
