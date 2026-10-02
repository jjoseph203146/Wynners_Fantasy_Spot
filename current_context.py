from datetime import datetime, timezone

import pandas as pd
import nflreadpy as nfl

from config import (
    CURRENT_SEASON,
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection

from context_data import (
    initialize_context_tables,
    update_weekly_rosters,
    update_snap_counts,
    update_injuries,
    upsert_rows,
    safe_text,
    safe_int,
    get_value,
)


# =========================================================
# EXPORT PATHS
# =========================================================

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

def now_utc():
    return datetime.now(
        timezone.utc
    ).isoformat()


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# INDIVIDUAL TABLE EXPORT
# =========================================================

def export_table(
    table_name,
    csv_path,
    parquet_path,
):

    with get_connection() as conn:

        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {table_name}
            """,
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
        f"{table_name}: {len(df)} rows exported"
    )


# =========================================================
# DEPTH CHART STATE
# =========================================================

def get_latest_depth_snapshot_by_team():

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                team,
                MAX(snapshot_dt)

            FROM depth_charts

            GROUP BY team
            """
        ).fetchall()

    result = {}

    for row in rows:

        team = row[0]
        latest_dt = row[1]

        if (
            team is not None
            and latest_dt is not None
        ):
            result[
                str(team)
            ] = str(latest_dt)

    return result


# =========================================================
# CURRENT DEPTH CHARTS
# =========================================================

def update_current_depth_charts():

    section(
        f"{CURRENT_SEASON} DEPTH CHARTS"
    )

    try:

        data = nfl.load_depth_charts(
            seasons=[CURRENT_SEASON]
        )

    except Exception as exc:

        print(
            f"{CURRENT_SEASON} depth charts "
            f"not available."
        )

        print(
            f"Source response: {exc}"
        )

        return {
            "available": False,
            "changed": False,
            "rows": 0,
        }

    df = data.to_pandas()

    if df.empty:

        print(
            "No depth-chart rows returned."
        )

        return {
            "available": False,
            "changed": False,
            "rows": 0,
        }

    print(
        f"Downloaded {len(df)} source rows."
    )

    # -----------------------------------------------------
    # CURRENT DATABASE STATE
    # -----------------------------------------------------

    latest_by_team = (
        get_latest_depth_snapshot_by_team()
    )

    # -----------------------------------------------------
    # FAST VECTORIZED FILTER
    # -----------------------------------------------------

    team_series = (
        df["team"]
        .astype("string")
    )

    snapshot_series = (
        df["dt"]
        .astype("string")
    )

    stored_latest = (
        team_series.map(
            latest_by_team
        )
    )

    new_mask = (
        stored_latest.isna()
        |
        (
            snapshot_series
            >
            stored_latest.fillna("")
        )
    )

    new_df = df.loc[
        new_mask
    ].copy()

    print(
        f"New snapshot rows: {len(new_df)}"
    )

    if new_df.empty:

        print(
            "Depth charts already current."
        )

        print(
            "Depth-chart export unchanged."
        )

        return {
            "available": True,
            "changed": False,
            "rows": 0,
        }

    # -----------------------------------------------------
    # PREPARE ROWS
    # -----------------------------------------------------

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

    for _, row in new_df.iterrows():

        snapshot_dt = safe_text(
            get_value(
                row,
                "dt",
            )
        )

        team = safe_text(
            get_value(
                row,
                "team",
            )
        )

        pos_id = safe_text(
            get_value(
                row,
                "pos_id",
            )
        )

        pos_slot = safe_int(
            get_value(
                row,
                "pos_slot",
            )
        )

        pos_rank = safe_int(
            get_value(
                row,
                "pos_rank",
            )
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
                    get_value(
                        row,
                        "player_name",
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "espn_id",
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "gsis_id",
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "pos_grp_id",
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "pos_grp",
                    )
                ),

                pos_id,

                safe_text(
                    get_value(
                        row,
                        "pos_name",
                    )
                ),

                safe_text(
                    get_value(
                        row,
                        "pos_abb",
                    )
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
        f"Inserted {inserted} new "
        f"depth-chart rows."
    )

    print(
        f"Skipped {skipped} rows."
    )

    # -----------------------------------------------------
    # EXPORT ONLY BECAUSE DATA CHANGED
    # -----------------------------------------------------

    if inserted > 0:

        section(
            "DEPTH CHART EXPORT REFRESH"
        )

        export_table(
            table_name="depth_charts",
            csv_path=DEPTH_CSV,
            parquet_path=DEPTH_PARQUET,
        )

    return {
        "available": True,
        "changed": inserted > 0,
        "rows": inserted,
    }


# =========================================================
# AUDIT
# =========================================================

def audit_current_context():

    section(
        "CURRENT CONTEXT AUDIT"
    )

    with get_connection() as conn:

        roster_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM weekly_rosters
            """
        ).fetchone()[0]

        snap_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_snap_counts
            """
        ).fetchone()[0]

        injury_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM injuries
            """
        ).fetchone()[0]

        depth_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM depth_charts
            """
        ).fetchone()[0]

        unmatched_snaps = conn.execute(
            """
            SELECT COUNT(*)

            FROM player_snap_counts s

            LEFT JOIN games g
                ON s.game_id = g.game_id

            WHERE g.game_id IS NULL
            """
        ).fetchone()[0]

    print(
        f"Weekly rosters: {roster_count}"
    )

    print(
        f"Snap counts:    {snap_count}"
    )

    print(
        f"Injuries:       {injury_count}"
    )

    print(
        f"Depth charts:   {depth_count}"
    )

    print()

    print(
        f"Unmatched snap game IDs: "
        f"{unmatched_snaps}"
    )

    if unmatched_snaps == 0:

        print(
            "PASS: all snap-count game IDs "
            "match games."
        )

    else:

        print(
            "WARNING: unmatched snap "
            "game IDs detected."
        )


# =========================================================
# CURRENT ROSTERS
# =========================================================

def run_current_rosters():

    success = update_weekly_rosters(
        [CURRENT_SEASON]
    )

    if success:

        section(
            "ROSTER EXPORT REFRESH"
        )

        export_table(
            table_name="weekly_rosters",
            csv_path=ROSTER_CSV,
            parquet_path=ROSTER_PARQUET,
        )

    return success


# =========================================================
# CURRENT SNAP COUNTS
# =========================================================

def run_current_snaps():

    success = update_snap_counts(
        [CURRENT_SEASON]
    )

    if success:

        section(
            "SNAP COUNT EXPORT REFRESH"
        )

        export_table(
            table_name="player_snap_counts",
            csv_path=SNAPS_CSV,
            parquet_path=SNAPS_PARQUET,
        )

    return success


# =========================================================
# CURRENT INJURIES
# =========================================================

def run_current_injuries():

    success = update_injuries(
        [CURRENT_SEASON]
    )

    if success:

        section(
            "INJURY EXPORT REFRESH"
        )

        export_table(
            table_name="injuries",
            csv_path=INJURY_CSV,
            parquet_path=INJURY_PARQUET,
        )

    return success


# =========================================================
# PRODUCTION CONTEXT UPDATE
# =========================================================

def run_current_context_update():

    section(
        "NFL CURRENT CONTEXT UPDATE"
    )

    print(
        f"Season: {CURRENT_SEASON}"
    )

    initialize_context_tables()

    # -----------------------------------------------------
    # ROSTERS
    # -----------------------------------------------------

    roster_ok = run_current_rosters()

    # -----------------------------------------------------
    # SNAP COUNTS
    # -----------------------------------------------------

    snaps_ok = run_current_snaps()

    # -----------------------------------------------------
    # INJURIES
    # -----------------------------------------------------

    injury_ok = run_current_injuries()

    # -----------------------------------------------------
    # DEPTH CHARTS
    # -----------------------------------------------------

    depth_result = (
        update_current_depth_charts()
    )

    # -----------------------------------------------------
    # AUDIT
    # -----------------------------------------------------

    audit_current_context()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    section(
        "CURRENT CONTEXT UPDATE COMPLETE"
    )

    print(
        f"Rosters: "
        f"{'AVAILABLE' if roster_ok else 'NOT AVAILABLE'}"
    )

    print(
        f"Snaps: "
        f"{'AVAILABLE' if snaps_ok else 'NOT AVAILABLE'}"
    )

    print(
        f"Injuries: "
        f"{'AVAILABLE' if injury_ok else 'NOT AVAILABLE'}"
    )

    print(
        f"Depth charts: "
        f"{'AVAILABLE' if depth_result['available'] else 'NOT AVAILABLE'}"
    )

    print(
        f"New depth-chart rows: "
        f"{depth_result['rows']}"
    )

    return {
        "rosters_available": roster_ok,
        "snaps_available": snaps_ok,
        "injuries_available": injury_ok,

        "depth_available":
            depth_result[
                "available"
            ],

        "depth_changed":
            depth_result[
                "changed"
            ],

        "depth_new_rows":
            depth_result[
                "rows"
            ],
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_current_context_update()
