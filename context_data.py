from datetime import datetime, timezone

import pandas as pd
import nflreadpy as nfl

from config import (
    CURRENT_SEASON,
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

ROSTER_CSV = CSV_DIR / "nfl_weekly_rosters.csv"
ROSTER_PARQUET = PARQUET_DIR / "nfl_weekly_rosters.parquet"

SNAPS_CSV = CSV_DIR / "nfl_snap_counts.csv"
SNAPS_PARQUET = PARQUET_DIR / "nfl_snap_counts.parquet"

INJURY_CSV = CSV_DIR / "nfl_injuries.csv"
INJURY_PARQUET = PARQUET_DIR / "nfl_injuries.parquet"

DEPTH_CSV = CSV_DIR / "nfl_depth_charts.csv"
DEPTH_PARQUET = PARQUET_DIR / "nfl_depth_charts.parquet"


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


def now_utc():
    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# DATABASE
# =========================================================

def initialize_context_tables():

    initialize_database()

    with get_connection() as conn:

        # -------------------------------------------------
        # WEEKLY ROSTERS
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_rosters (

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                game_type TEXT,

                team TEXT NOT NULL,

                gsis_id TEXT NOT NULL,

                full_name TEXT,
                first_name TEXT,
                last_name TEXT,
                football_name TEXT,

                position TEXT,
                depth_chart_position TEXT,

                jersey_number INTEGER,

                status TEXT,
                status_description_abbr TEXT,

                birth_date TEXT,
                height INTEGER,
                weight INTEGER,
                college TEXT,

                years_exp INTEGER,

                espn_id TEXT,
                pfr_id TEXT,
                pff_id TEXT,
                sleeper_id TEXT,

                rookie_year INTEGER,
                entry_year INTEGER,
                draft_club TEXT,
                draft_number INTEGER,

                updated_at TEXT,

                PRIMARY KEY (
                    season,
                    week,
                    game_type,
                    team,
                    gsis_id
                )
            )
            """
        )

        # -------------------------------------------------
        # SNAP COUNTS
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS player_snap_counts (

                game_id TEXT NOT NULL,

                pfr_game_id TEXT,

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                game_type TEXT,

                player_name TEXT,
                pfr_player_id TEXT NOT NULL,

                position TEXT,

                team TEXT NOT NULL,
                opponent_team TEXT,

                offense_snaps REAL,
                offense_pct REAL,

                defense_snaps REAL,
                defense_pct REAL,

                st_snaps REAL,
                st_pct REAL,

                updated_at TEXT,

                PRIMARY KEY (
                    game_id,
                    team,
                    pfr_player_id
                )
            )
            """
        )

        # -------------------------------------------------
        # INJURIES
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injuries (

                season INTEGER NOT NULL,
                week INTEGER NOT NULL,

                season_type TEXT,
                game_type TEXT,

                team TEXT NOT NULL,

                gsis_id TEXT NOT NULL,

                position TEXT,

                full_name TEXT,
                first_name TEXT,
                last_name TEXT,

                report_primary_injury TEXT,
                report_secondary_injury TEXT,

                report_status TEXT,

                practice_primary_injury TEXT,
                practice_secondary_injury TEXT,

                practice_status TEXT,

                updated_at TEXT,

                PRIMARY KEY (
                    season,
                    week,
                    game_type,
                    team,
                    gsis_id
                )
            )
            """
        )

        # -------------------------------------------------
        # DEPTH CHART SNAPSHOTS
        # -------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS depth_charts (

                snapshot_dt TEXT NOT NULL,

                team TEXT NOT NULL,

                player_name TEXT,

                espn_id TEXT,
                gsis_id TEXT,

                pos_grp_id TEXT,
                pos_grp TEXT,

                pos_id TEXT,
                pos_name TEXT,
                pos_abb TEXT,

                pos_slot INTEGER,
                pos_rank INTEGER,

                updated_at TEXT,

                PRIMARY KEY (
                    snapshot_dt,
                    team,
                    pos_id,
                    pos_slot,
                    pos_rank
                )
            )
            """
        )

        # -------------------------------------------------
        # INDEXES
        # -------------------------------------------------

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_roster_player
            ON weekly_rosters(gsis_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_roster_team_week
            ON weekly_rosters(
                season,
                week,
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_snaps_game
            ON player_snap_counts(game_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_snaps_player
            ON player_snap_counts(pfr_player_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_snaps_team
            ON player_snap_counts(
                season,
                week,
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_injury_player
            ON injuries(gsis_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_injury_team_week
            ON injuries(
                season,
                week,
                team
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_depth_gsis
            ON depth_charts(gsis_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_depth_team_date
            ON depth_charts(
                team,
                snapshot_dt
            )
            """
        )


# =========================================================
# GENERIC UPSERT
# =========================================================

def upsert_rows(
    table,
    columns,
    conflict_columns,
    rows,
):

    if not rows:
        return 0

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    update_columns = [
        column
        for column in columns
        if column not in conflict_columns
    ]

    update_clause = ", ".join(
        f"{column}=excluded.{column}"
        for column in update_columns
    )

    sql = f"""
        INSERT INTO {table} (
            {", ".join(columns)}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT(
            {", ".join(conflict_columns)}
        )
        DO UPDATE SET
            {update_clause}
    """

    with get_connection() as conn:
        conn.executemany(
            sql,
            rows,
        )

    return len(rows)


# =========================================================
# WEEKLY ROSTERS
# =========================================================

def update_weekly_rosters(seasons):

    print()
    print("=" * 70)
    print("WEEKLY ROSTERS")
    print("=" * 70)

    try:
        data = nfl.load_rosters_weekly(
            seasons=seasons
        )

    except Exception as exc:
        print(
            f"Roster data unavailable: {exc}"
        )
        return False

    df = data.to_pandas()

    print(
        f"Downloaded {len(df)} roster rows."
    )

    columns = [
        "season",
        "week",
        "game_type",
        "team",
        "gsis_id",
        "full_name",
        "first_name",
        "last_name",
        "football_name",
        "position",
        "depth_chart_position",
        "jersey_number",
        "status",
        "status_description_abbr",
        "birth_date",
        "height",
        "weight",
        "college",
        "years_exp",
        "espn_id",
        "pfr_id",
        "pff_id",
        "sleeper_id",
        "rookie_year",
        "entry_year",
        "draft_club",
        "draft_number",
        "updated_at",
    ]

    rows = []
    skipped = 0
    updated_at = now_utc()

    for _, row in df.iterrows():

        season = safe_int(
            get_value(row, "season")
        )

        week = safe_int(
            get_value(row, "week")
        )

        game_type = safe_text(
            get_value(row, "game_type")
        )

        team = safe_text(
            get_value(row, "team")
        )

        gsis_id = safe_text(
            get_value(row, "gsis_id")
        )

        if (
            season is None
            or week is None
            or game_type is None
            or team is None
            or gsis_id is None
        ):
            skipped += 1
            continue

        rows.append(
            (
                season,
                week,
                game_type,
                team,
                gsis_id,

                safe_text(
                    get_value(row, "full_name")
                ),

                safe_text(
                    get_value(row, "first_name")
                ),

                safe_text(
                    get_value(row, "last_name")
                ),

                safe_text(
                    get_value(row, "football_name")
                ),

                safe_text(
                    get_value(row, "position")
                ),

                safe_text(
                    get_value(
                        row,
                        "depth_chart_position"
                    )
                ),

                safe_int(
                    get_value(row, "jersey_number")
                ),

                safe_text(
                    get_value(row, "status")
                ),

                safe_text(
                    get_value(
                        row,
                        "status_description_abbr"
                    )
                ),

                safe_text(
                    get_value(row, "birth_date")
                ),

                safe_int(
                    get_value(row, "height")
                ),

                safe_int(
                    get_value(row, "weight")
                ),

                safe_text(
                    get_value(row, "college")
                ),

                safe_int(
                    get_value(row, "years_exp")
                ),

                safe_text(
                    get_value(row, "espn_id")
                ),

                safe_text(
                    get_value(row, "pfr_id")
                ),

                safe_text(
                    get_value(row, "pff_id")
                ),

                safe_text(
                    get_value(row, "sleeper_id")
                ),

                safe_int(
                    get_value(row, "rookie_year")
                ),

                safe_int(
                    get_value(row, "entry_year")
                ),

                safe_text(
                    get_value(row, "draft_club")
                ),

                safe_int(
                    get_value(row, "draft_number")
                ),

                updated_at,
            )
        )

    inserted = upsert_rows(
        table="weekly_rosters",
        columns=columns,
        conflict_columns=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
        ],
        rows=rows,
    )

    print(
        f"Inserted/updated {inserted} "
        f"roster rows."
    )

    print(
        f"Skipped {skipped} roster rows."
    )

    return True


# =========================================================
# SNAP COUNTS
# =========================================================

def update_snap_counts(seasons):

    print()
    print("=" * 70)
    print("SNAP COUNTS")
    print("=" * 70)

    try:
        data = nfl.load_snap_counts(
            seasons=seasons
        )

    except Exception as exc:
        print(
            f"Snap-count data unavailable: {exc}"
        )
        return False

    df = data.to_pandas()

    print(
        f"Downloaded {len(df)} snap rows."
    )

    columns = [
        "game_id",
        "pfr_game_id",
        "season",
        "week",
        "game_type",
        "player_name",
        "pfr_player_id",
        "position",
        "team",
        "opponent_team",
        "offense_snaps",
        "offense_pct",
        "defense_snaps",
        "defense_pct",
        "st_snaps",
        "st_pct",
        "updated_at",
    ]

    rows = []
    skipped = 0
    updated_at = now_utc()

    for _, row in df.iterrows():

        game_id = safe_text(
            get_value(row, "game_id")
        )

        pfr_player_id = safe_text(
            get_value(row, "pfr_player_id")
        )

        team = safe_text(
            get_value(row, "team")
        )

        if (
            game_id is None
            or pfr_player_id is None
            or team is None
        ):
            skipped += 1
            continue

        rows.append(
            (
                game_id,

                safe_text(
                    get_value(row, "pfr_game_id")
                ),

                safe_int(
                    get_value(row, "season")
                ),

                safe_int(
                    get_value(row, "week")
                ),

                safe_text(
                    get_value(row, "game_type")
                ),

                safe_text(
                    get_value(row, "player")
                ),

                pfr_player_id,

                safe_text(
                    get_value(row, "position")
                ),

                team,

                safe_text(
                    get_value(row, "opponent")
                ),

                safe_float(
                    get_value(row, "offense_snaps")
                ),

                safe_float(
                    get_value(row, "offense_pct")
                ),

                safe_float(
                    get_value(row, "defense_snaps")
                ),

                safe_float(
                    get_value(row, "defense_pct")
                ),

                safe_float(
                    get_value(row, "st_snaps")
                ),

                safe_float(
                    get_value(row, "st_pct")
                ),

                updated_at,
            )
        )

    inserted = upsert_rows(
        table="player_snap_counts",
        columns=columns,
        conflict_columns=[
            "game_id",
            "team",
            "pfr_player_id",
        ],
        rows=rows,
    )

    print(
        f"Inserted/updated {inserted} "
        f"snap rows."
    )

    print(
        f"Skipped {skipped} snap rows."
    )

    return True


# =========================================================
# INJURIES
# =========================================================

def update_injuries(seasons):

    print()
    print("=" * 70)
    print("INJURIES")
    print("=" * 70)

    try:
        data = nfl.load_injuries(
            seasons=seasons
        )

    except Exception as exc:
        print(
            f"Injury data unavailable: {exc}"
        )
        return False

    df = data.to_pandas()

    print(
        f"Downloaded {len(df)} injury rows."
    )

    columns = [
        "season",
        "week",
        "season_type",
        "game_type",
        "team",
        "gsis_id",
        "position",
        "full_name",
        "first_name",
        "last_name",
        "report_primary_injury",
        "report_secondary_injury",
        "report_status",
        "practice_primary_injury",
        "practice_secondary_injury",
        "practice_status",
        "updated_at",
    ]

    rows = []
    skipped = 0
    updated_at = now_utc()

    for _, row in df.iterrows():

        season = safe_int(
            get_value(row, "season")
        )

        week = safe_int(
            get_value(row, "week")
        )

        game_type = safe_text(
            get_value(row, "game_type")
        )

        team = safe_text(
            get_value(row, "team")
        )

        gsis_id = safe_text(
            get_value(row, "gsis_id")
        )

        if (
            season is None
            or week is None
            or game_type is None
            or team is None
            or gsis_id is None
        ):
            skipped += 1
            continue

        rows.append(
            (
                season,
                week,

                safe_text(
                    get_value(row, "season_type")
                ),

                game_type,
                team,
                gsis_id,

                safe_text(
                    get_value(row, "position")
                ),

                safe_text(
                    get_value(row, "full_name")
                ),

                safe_text(
                    get_value(row, "first_name")
                ),

                safe_text(
                    get_value(row, "last_name")
                ),

                safe_text(
                    get_value(
                        row,
                        "report_primary_injury"
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "report_secondary_injury"
                    )
                ),

                safe_text(
                    get_value(row, "report_status")
                ),

                safe_text(
                    get_value(
                        row,
                        "practice_primary_injury"
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "practice_secondary_injury"
                    )
                ),

                safe_text(
                    get_value(row, "practice_status")
                ),

                updated_at,
            )
        )

    inserted = upsert_rows(
        table="injuries",
        columns=columns,
        conflict_columns=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
        ],
        rows=rows,
    )

    print(
        f"Inserted/updated {inserted} "
        f"injury rows."
    )

    print(
        f"Skipped {skipped} injury rows."
    )

    return True


# =========================================================
# DEPTH CHARTS
# =========================================================

def update_depth_charts(seasons):

    print()
    print("=" * 70)
    print("DEPTH CHARTS")
    print("=" * 70)

    try:
        data = nfl.load_depth_charts(
            seasons=seasons
        )

    except Exception as exc:
        print(
            f"Depth-chart data unavailable: {exc}"
        )
        return False

    df = data.to_pandas()

    print(
        f"Downloaded {len(df)} "
        f"depth-chart rows."
    )

    columns = [
        "snapshot_dt",
        "team",
        "player_name",
        "espn_id",
        "gsis_id",
        "pos_grp_id",
        "pos_grp",
        "pos_id",
        "pos_name",
        "pos_abb",
        "pos_slot",
        "pos_rank",
        "updated_at",
    ]

    rows = []
    skipped = 0
    updated_at = now_utc()

    for _, row in df.iterrows():

        snapshot_dt = safe_text(
            get_value(row, "dt")
        )

        team = safe_text(
            get_value(row, "team")
        )

        pos_id = safe_text(
            get_value(row, "pos_id")
        )

        pos_slot = safe_int(
            get_value(row, "pos_slot")
        )

        pos_rank = safe_int(
            get_value(row, "pos_rank")
        )

        if (
            snapshot_dt is None
            or team is None
            or pos_id is None
            or pos_slot is None
            or pos_rank is None
        ):
            skipped += 1
            continue

        rows.append(
            (
                snapshot_dt,
                team,

                safe_text(
                    get_value(row, "player_name")
                ),

                safe_text(
                    get_value(row, "espn_id")
                ),

                safe_text(
                    get_value(row, "gsis_id")
                ),

                safe_text(
                    get_value(row, "pos_grp_id")
                ),

                safe_text(
                    get_value(row, "pos_grp")
                ),

                pos_id,

                safe_text(
                    get_value(row, "pos_name")
                ),

                safe_text(
                    get_value(row, "pos_abb")
                ),

                pos_slot,
                pos_rank,

                updated_at,
            )
        )

    inserted = upsert_rows(
        table="depth_charts",
        columns=columns,
        conflict_columns=[
            "snapshot_dt",
            "team",
            "pos_id",
            "pos_slot",
            "pos_rank",
        ],
        rows=rows,
    )

    print(
        f"Inserted/updated {inserted} "
        f"depth-chart rows."
    )

    print(
        f"Skipped {skipped} "
        f"depth-chart rows."
    )

    return True


# =========================================================
# EXPORTS
# =========================================================

def export_context_data():

    print()
    print("=" * 70)
    print("CONTEXT DATA EXPORTS")
    print("=" * 70)

    exports = [
        (
            "weekly_rosters",
            ROSTER_CSV,
            ROSTER_PARQUET,
        ),
        (
            "player_snap_counts",
            SNAPS_CSV,
            SNAPS_PARQUET,
        ),
        (
            "injuries",
            INJURY_CSV,
            INJURY_PARQUET,
        ),
        (
            "depth_charts",
            DEPTH_CSV,
            DEPTH_PARQUET,
        ),
    ]

    with get_connection() as conn:

        for (
            table,
            csv_path,
            parquet_path,
        ) in exports:

            df = pd.read_sql_query(
                f"SELECT * FROM {table}",
                conn,
            )

            df.to_csv(
                csv_path,
                index=False,
            )

            df.to_parquet(
                parquet_path,
                index=False,
            )

            print(
                f"{table}: {len(df)} rows"
            )


# =========================================================
# AUDIT
# =========================================================

def audit_context_data():

    print()
    print("=" * 70)
    print("CONTEXT DATABASE AUDIT")
    print("=" * 70)

    with get_connection() as conn:

        tables = [
            "weekly_rosters",
            "player_snap_counts",
            "injuries",
            "depth_charts",
        ]

        for table in tables:

            count = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {table}
                """
            ).fetchone()[0]

            print(
                f"{table}: {count} rows"
            )

        unmatched_snaps = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_snap_counts s

            LEFT JOIN games g
                ON s.game_id = g.game_id

            WHERE g.game_id IS NULL
            """
        ).fetchone()[0]

        print()

        print(
            "Snap rows with unmatched "
            f"game_id: {unmatched_snaps}"
        )

        if unmatched_snaps == 0:

            print(
                "PASS: all snap-count game IDs "
                "match the games table."
            )

        else:

            print(
                "WARNING: unmatched snap-count "
                "game IDs were found."
            )


# =========================================================
# HISTORICAL LOAD
# =========================================================

def load_historical_context():

    print()
    print("=" * 70)
    print("NFL HISTORICAL CONTEXT LOADER")
    print("=" * 70)

    initialize_context_tables()

    update_weekly_rosters(
        HISTORICAL_SEASONS
    )

    update_snap_counts(
        HISTORICAL_SEASONS
    )

    update_injuries(
        HISTORICAL_SEASONS
    )

    update_depth_charts(
        HISTORICAL_SEASONS
    )

    audit_context_data()

    export_context_data()

    print()
    print("=" * 70)
    print(
        "HISTORICAL CONTEXT LOAD SUCCESSFUL"
    )
    print("=" * 70)


# =========================================================
# CURRENT DEPTH CHART
# =========================================================

def load_current_depth_charts():

    print()
    print("=" * 70)
    print(
        f"{CURRENT_SEASON} DEPTH CHART UPDATE"
    )
    print("=" * 70)

    initialize_context_tables()

    success = update_depth_charts(
        [CURRENT_SEASON]
    )

    if success:

        audit_context_data()

        export_context_data()

    return success


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    load_historical_context()

    if CURRENT_SEASON not in HISTORICAL_SEASONS:

        load_current_depth_charts()

