import sqlite3
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH, PARQUET_DIR


TARGETS = {
    "GSIS:00-0040142": "Kaleb Johnson",
    "GSIS:00-0038809": "Ben Sims",
}

CANDIDATE_CURRENT_FEATURE_FILES = [
    PARQUET_DIR / "nfl_current_slate_features.parquet",
    PARQUET_DIR / "current_slate_features.parquet",
    PARQUET_DIR / "nfl_current_features.parquet",
]

PROJECTION_FILE = PARQUET_DIR / "nfl_production_projection.parquet"
POOL_FILE = PARQUET_DIR / "nfl_fanduel_player_pool.parquet"

OUTPUT_FILE = Path(
    "/home/mwynn/nfl_data_engine/data/csv/"
    "audit_current_feature_gap_targets.csv"
)


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def table_columns(connection, table):
    rows = connection.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return [row[1] for row in rows]


def pick_column(columns, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def resolve_current_feature_file():
    for path in CANDIDATE_CURRENT_FEATURE_FILES:
        if path.exists():
            return path

    matches = sorted(
        path
        for path in PARQUET_DIR.glob("*.parquet")
        if "current" in path.name.lower()
        and "feature" in path.name.lower()
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise RuntimeError(
            "Could not locate current-slate feature parquet. "
            f"Checked: {[str(p) for p in CANDIDATE_CURRENT_FEATURE_FILES]}"
        )

    raise RuntimeError(
        "Multiple possible current feature files found: "
        + ", ".join(str(path) for path in matches)
    )


def gsis_from_identity(identity_key):
    if pd.isna(identity_key):
        return None

    value = str(identity_key).strip()

    if value.startswith("GSIS:"):
        return value.split("GSIS:", 1)[1]

    return None


def load_latest_depth_for_targets():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        columns = table_columns(
            connection,
            "depth_charts",
        )

        section("DEPTH-CHART SCHEMA")

        print("Available columns:")
        for column in columns:
            print(f"  {column}")

        snapshot_col = pick_column(
            columns,
            [
                "snapshot_dt",
                "dt",
                "snapshot",
                "date",
            ],
        )

        team_col = pick_column(
            columns,
            ["team"],
        )

        gsis_col = pick_column(
            columns,
            [
                "gsis_id",
                "gsis",
            ],
        )

        name_col = pick_column(
            columns,
            [
                "player_name",
                "full_name",
                "name",
            ],
        )

        position_col = pick_column(
            columns,
            [
                "position",
                "position_group",
                "pos_abb",
                "pos_abbr",
                "pos_name",
                "pos_group",
                "pos_grp",
            ],
        )

        rank_col = pick_column(
            columns,
            [
                "pos_rank",
                "rank",
                "position_rank",
                "depth_rank",
            ],
        )

        slot_col = pick_column(
            columns,
            [
                "pos_slot",
                "slot",
                "position_slot",
                "depth_slot",
            ],
        )

        required = {
            "snapshot": snapshot_col,
            "team": team_col,
            "gsis": gsis_col,
        }

        missing = [
            label
            for label, column in required.items()
            if column is None
        ]

        if missing:
            raise RuntimeError(
                "Could not resolve required depth-chart columns: "
                + ", ".join(missing)
                + f". Available columns: {columns}"
            )

        print()
        print("Resolved depth columns:")
        print(f"  snapshot -> {snapshot_col}")
        print(f"  team     -> {team_col}")
        print(f"  gsis     -> {gsis_col}")
        print(f"  name     -> {name_col}")
        print(f"  position -> {position_col}")
        print(f"  rank     -> {rank_col}")
        print(f"  slot     -> {slot_col}")

        select_parts = [
            f"d.{snapshot_col} AS snapshot_dt",
            f"d.{team_col} AS depth_team",
            f"d.{gsis_col} AS gsis_id",
        ]

        if name_col:
            select_parts.append(
                f"d.{name_col} AS depth_player_name"
            )
        else:
            select_parts.append(
                "NULL AS depth_player_name"
            )

        if position_col:
            select_parts.append(
                f"d.{position_col} AS depth_position"
            )
        else:
            select_parts.append(
                "NULL AS depth_position"
            )

        if rank_col:
            select_parts.append(
                f"d.{rank_col} AS depth_rank"
            )
        else:
            select_parts.append(
                "NULL AS depth_rank"
            )

        if slot_col:
            select_parts.append(
                f"d.{slot_col} AS depth_slot"
            )
        else:
            select_parts.append(
                "NULL AS depth_slot"
            )

        query = f"""
        WITH latest AS (
            SELECT
                {team_col} AS team,
                MAX({snapshot_col}) AS max_snapshot
            FROM depth_charts
            GROUP BY {team_col}
        )
        SELECT
            {", ".join(select_parts)}
        FROM depth_charts d
        INNER JOIN latest l
            ON d.{team_col} = l.team
           AND d.{snapshot_col} = l.max_snapshot
        WHERE d.{gsis_col} IS NOT NULL
        """

        depth = pd.read_sql_query(
            query,
            connection,
        )

    finally:
        connection.close()

    target_gsis = {
        gsis_from_identity(identity_key)
        for identity_key in TARGETS
    }

    return depth[
        depth["gsis_id"].astype(str).isin(
            {str(value) for value in target_gsis}
        )
    ].copy()


def values_from_rows(df, column):
    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def main():
    section("TARGETED CURRENT-FEATURE GAP AUDIT")

    feature_file = resolve_current_feature_file()

    print(f"Current feature file: {feature_file}")
    print(f"Projection file: {PROJECTION_FILE}")
    print(f"FanDuel pool file: {POOL_FILE}")

    current = pd.read_parquet(feature_file)
    projection = pd.read_parquet(PROJECTION_FILE)
    pool = pd.read_parquet(POOL_FILE)
    depth = load_latest_depth_for_targets()

    rows = []

    for identity_key, expected_name in TARGETS.items():
        current_rows = current[
            current["identity_key"].astype(str)
            ==
            identity_key
        ].copy()

        projection_rows = projection[
            projection["identity_key"].astype(str)
            ==
            identity_key
        ].copy()

        pool_rows = pool[
            pool["identity_key"].astype(str)
            ==
            identity_key
        ].copy()

        gsis_id = gsis_from_identity(identity_key)

        depth_rows = depth[
            depth["gsis_id"].astype(str)
            ==
            str(gsis_id)
        ].copy()

        current_teams = values_from_rows(
            current_rows,
            "team",
        )

        current_positions = values_from_rows(
            current_rows,
            "position",
        )

        current_statuses = values_from_rows(
            current_rows,
            "feature_status",
        )

        projection_teams = values_from_rows(
            projection_rows,
            "team",
        )

        projection_statuses = values_from_rows(
            projection_rows,
            "production_status",
        )

        pool_teams = values_from_rows(
            pool_rows,
            "team",
        )

        pool_positions = values_from_rows(
            pool_rows,
            "fd_position",
        )

        depth_teams = values_from_rows(
            depth_rows,
            "depth_team",
        )

        depth_positions = values_from_rows(
            depth_rows,
            "depth_position",
        )

        depth_ranks = values_from_rows(
            depth_rows,
            "depth_rank",
        )

        depth_slots = values_from_rows(
            depth_rows,
            "depth_slot",
        )

        if current_rows.empty:
            diagnosis = "MISSING_FROM_CURRENT_FEATURES"
        elif projection_rows.empty:
            diagnosis = "PRESENT_IN_FEATURES_MISSING_FROM_PROJECTION"
        else:
            diagnosis = "PRESENT_IN_BOTH_CHECK_JOIN_KEYS"

        rows.append(
            {
                "identity_key": identity_key,
                "expected_name": expected_name,
                "pool_rows": len(pool_rows),
                "pool_teams": ",".join(pool_teams),
                "pool_positions": ",".join(pool_positions),
                "depth_rows": len(depth_rows),
                "depth_teams": ",".join(depth_teams),
                "depth_positions": ",".join(depth_positions),
                "depth_ranks": ",".join(depth_ranks),
                "depth_slots": ",".join(depth_slots),
                "current_feature_rows": len(current_rows),
                "current_feature_teams": ",".join(current_teams),
                "current_feature_positions": ",".join(current_positions),
                "current_feature_statuses": ",".join(current_statuses),
                "projection_rows": len(projection_rows),
                "projection_teams": ",".join(projection_teams),
                "projection_statuses": ",".join(projection_statuses),
                "diagnosis": diagnosis,
            }
        )

    audit = pd.DataFrame(rows)

    section("TARGET DETAILS")

    print(
        audit.to_string(index=False)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    section("AUDIT COMPLETE")

    print(f"CSV: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
