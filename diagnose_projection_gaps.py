import sqlite3
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH, PARQUET_DIR


PLAYER_POOL_PATH = PARQUET_DIR / "nfl_fanduel_player_pool.parquet"
PROJECTION_PATH = PARQUET_DIR / "nfl_production_projection.parquet"

OUTPUT_PATH = Path(
    "/home/mwynn/nfl_data_engine/data/csv/"
    "audit_fanduel_projection_gaps.csv"
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
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def identity_to_gsis(identity_key):
    if pd.isna(identity_key):
        return None

    value = str(identity_key).strip()

    if value.startswith("GSIS:"):
        return value.split("GSIS:", 1)[1]

    return None


def load_gap_rows():
    if not PLAYER_POOL_PATH.exists():
        raise RuntimeError(
            f"Missing player-pool parquet: {PLAYER_POOL_PATH}"
        )

    pool = pd.read_parquet(
        PLAYER_POOL_PATH
    ).copy()

    gaps = pool[
        pool["player_pool_status"]
        ==
        "NO_CURRENT_PROJECTION"
    ].copy()

    gaps["gsis_id"] = gaps[
        "identity_key"
    ].apply(identity_to_gsis)

    return gaps


def load_latest_depth():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        columns = table_columns(
            connection,
            "depth_charts",
        )

        snapshot_col = pick_column(
            columns,
            [
                "snapshot_dt",
                "dt",
            ],
        )

        team_col = pick_column(
            columns,
            ["team"],
        )

        gsis_col = pick_column(
            columns,
            ["gsis_id"],
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
                "pos_abb",
                "pos_abbr",
                "pos_name",
                "position_group",
            ],
        )

        rank_col = pick_column(
            columns,
            [
                "pos_rank",
                "rank",
                "position_rank",
            ],
        )

        slot_col = pick_column(
            columns,
            [
                "pos_slot",
                "slot",
                "position_slot",
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

    return depth


def load_projection():
    if not PROJECTION_PATH.exists():
        raise RuntimeError(
            f"Missing production projection: {PROJECTION_PATH}"
        )

    return pd.read_parquet(
        PROJECTION_PATH
    ).copy()


def diagnose():
    section("FANDUEL NO-CURRENT-PROJECTION DIAGNOSTIC")

    gaps = load_gap_rows()

    print(
        f"Salary-bearing NO_CURRENT_PROJECTION rows: "
        f"{len(gaps)}"
    )

    if gaps.empty:
        print("Nothing to diagnose.")
        return pd.DataFrame()

    projection = load_projection()
    depth = load_latest_depth()

    print(
        f"Production projection rows: {len(projection)}"
    )
    print(
        f"Latest depth-chart rows: {len(depth)}"
    )

    projection_ids = set(
        projection["identity_key"]
        .dropna()
        .astype(str)
    )

    records = []

    for _, row in gaps.iterrows():
        identity_key = row.get("identity_key")
        gsis_id = row.get("gsis_id")
        fd_team = row.get("team")

        depth_rows = depth[
            depth["gsis_id"].astype(str)
            ==
            str(gsis_id)
        ].copy()

        depth_teams = sorted(
            depth_rows["depth_team"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        depth_positions = sorted(
            depth_rows["depth_position"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        depth_ranks = sorted(
            depth_rows["depth_rank"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        depth_slots = sorted(
            depth_rows["depth_slot"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        in_projection = (
            str(identity_key)
            in projection_ids
        )

        in_latest_depth = (
            len(depth_rows) > 0
        )

        team_match = (
            fd_team in depth_teams
            if in_latest_depth
            else False
        )

        if not in_latest_depth:
            classification = (
                "NOT_IN_LATEST_DEPTH"
            )
        elif not team_match:
            classification = (
                "LATEST_DEPTH_TEAM_MISMATCH"
            )
        elif not in_projection:
            classification = (
                "IN_LATEST_DEPTH_BUT_NO_PROJECTION"
            )
        else:
            classification = (
                "UNEXPECTED_PROJECTION_JOIN_GAP"
            )

        records.append(
            {
                "fd_position": row.get(
                    "fd_position"
                ),
                "fd_name": row.get(
                    "fd_name"
                ),
                "fd_team": fd_team,
                "opponent_team": row.get(
                    "opponent_team"
                ),
                "salary": row.get(
                    "salary"
                ),
                "identity_key": identity_key,
                "gsis_id": gsis_id,
                "identity_match_method": row.get(
                    "identity_match_method"
                ),
                "in_production_projection": int(
                    in_projection
                ),
                "in_latest_depth": int(
                    in_latest_depth
                ),
                "latest_depth_team_match": int(
                    team_match
                ),
                "latest_depth_teams": ",".join(
                    depth_teams
                ),
                "latest_depth_positions": ",".join(
                    depth_positions
                ),
                "latest_depth_ranks": ",".join(
                    depth_ranks
                ),
                "latest_depth_slots": ",".join(
                    depth_slots
                ),
                "classification": classification,
            }
        )

    audit = pd.DataFrame(
        records
    )

    section("CLASSIFICATION SUMMARY")

    print(
        audit["classification"]
        .value_counts()
        .to_string()
    )

    section("PLAYER DETAILS")

    display_columns = [
        "fd_position",
        "fd_name",
        "fd_team",
        "salary",
        "latest_depth_teams",
        "latest_depth_positions",
        "latest_depth_ranks",
        "classification",
    ]

    print(
        audit[
            display_columns
        ].to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    section("DIAGNOSTIC COMPLETE")

    print(f"Audit CSV: {OUTPUT_PATH}")

    return audit


if __name__ == "__main__":
    diagnose()
