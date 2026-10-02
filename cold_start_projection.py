#!/usr/bin/env python3

"""
WFS Cold-Start Projection Engine v1

Purpose
-------
Generate deterministic fantasy-point estimates for players who have:

    production_status == COLD_START

The estimator is intentionally separate from the frozen WFS production
projection model and FanDuel optimizer.

Method
------
Historical calibration cohort:
    2025 REG Week 1
    QB/RB/WR/TE
    zero prior NFL games

Role authority:
    latest offensive depth-chart rank available before Week 1 kickoff

Estimator:
    Position + Depth Rank empirical mean, shrunk toward position mean.

    estimate = (n * rank_mean + K * position_mean) / (n + K)

K = 1

K=1 was selected by leave-one-out MAE on the historical cold-start cohort.

Movement/promotion history is retained for audit/intelligence only and does
NOT modify the projection.

No fuzzy matching.
No invented projection for an unresolved player.
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from config import DATABASE_PATH


APP_DIR = Path(__file__).resolve().parent

PRODUCTION_PROJECTION_PATH = (
    APP_DIR / "data" / "parquet" / "nfl_production_projection.parquet"
)

OUTPUT_PARQUET = (
    APP_DIR / "data" / "parquet" / "nfl_cold_start_projection.parquet"
)

OUTPUT_CSV = (
    APP_DIR / "data" / "csv" / "nfl_cold_start_projection.csv"
)

AUDIT_CSV = (
    APP_DIR / "data" / "csv" / "audit_cold_start_projection.csv"
)

SHRINK_K = 1.0
OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def load_historical_cohort(conn):
    wk1 = pd.read_sql_query(
        """
        SELECT
            p.player_id,
            p.player_display_name,
            p.position,
            p.team,
            p.game_id,
            p.fanduel_points,
            g.game_date,
            g.gametime
        FROM player_game_stats p
        JOIN games g
          ON p.game_id = g.game_id
        WHERE p.season = 2025
          AND p.week = 1
          AND p.season_type = 'REG'
          AND p.position IN ('QB','RB','WR','TE')
        """,
        conn,
    )

    prior = pd.read_sql_query(
        """
        SELECT
            player_id,
            COUNT(*) AS prior_games
        FROM player_game_stats
        WHERE season < 2025
        GROUP BY player_id
        """,
        conn,
    )

    depth = pd.read_sql_query(
        """
        SELECT
            snapshot_dt,
            gsis_id,
            team,
            pos_abb,
            pos_slot,
            pos_rank
        FROM depth_charts
        WHERE substr(snapshot_dt,1,4) = '2025'
          AND pos_abb IN ('QB','RB','WR','TE')
        """,
        conn,
    )

    wk1["player_id"] = wk1["player_id"].map(normalize_id)
    prior["player_id"] = prior["player_id"].map(normalize_id)
    depth["gsis_id"] = depth["gsis_id"].map(normalize_id)

    wk1 = wk1.merge(
        prior,
        on="player_id",
        how="left",
    )

    wk1["prior_games"] = (
        wk1["prior_games"]
        .fillna(0)
        .astype(int)
    )

    cold = wk1[
        wk1["prior_games"] == 0
    ].copy()

    cold["kickoff"] = pd.to_datetime(
        cold["game_date"].astype(str)
        + " "
        + cold["gametime"].astype(str),
        utc=True,
        errors="coerce",
    )

    depth["snapshot_dt"] = pd.to_datetime(
        depth["snapshot_dt"],
        utc=True,
        errors="coerce",
    )

    rows = []

    for _, player in cold.iterrows():

        player_depth = depth[
            (depth["gsis_id"] == player["player_id"])
            & (depth["pos_abb"] == player["position"])
            & (depth["snapshot_dt"] < player["kickoff"])
        ].copy()

        if player_depth.empty:
            continue

        player_depth = (
            player_depth
            .sort_values("snapshot_dt")
            .drop_duplicates(
                subset=[
                    "snapshot_dt",
                    "pos_slot",
                    "pos_rank",
                ],
                keep="last",
            )
        )

        latest = player_depth.iloc[-1]

        rows.append(
            {
                "player_id": player["player_id"],
                "player_display_name": player["player_display_name"],
                "position": player["position"],
                "team": player["team"],
                "pos_rank": int(latest["pos_rank"]),
                "fanduel_points": float(player["fanduel_points"]),
            }
        )

    cohort = pd.DataFrame(rows)

    if cohort.empty:
        raise RuntimeError(
            "Historical cold-start cohort is empty."
        )

    return cohort


def build_calibration_table(cohort):
    position_stats = (
        cohort
        .groupby("position")["fanduel_points"]
        .agg(
            position_n="count",
            position_mean="mean",
        )
        .reset_index()
    )

    rank_stats = (
        cohort
        .groupby(
            ["position", "pos_rank"]
        )["fanduel_points"]
        .agg(
            rank_n="count",
            rank_mean="mean",
            rank_median="median",
        )
        .reset_index()
    )

    calibration = rank_stats.merge(
        position_stats,
        on="position",
        how="left",
        validate="many_to_one",
    )

    calibration["cold_start_estimate"] = (
        (
            calibration["rank_n"]
            * calibration["rank_mean"]
        )
        + (
            SHRINK_K
            * calibration["position_mean"]
        )
    ) / (
        calibration["rank_n"]
        + SHRINK_K
    )

    return calibration


def load_current_cold_starts():
    if not PRODUCTION_PROJECTION_PATH.exists():
        raise FileNotFoundError(
            f"Production projection not found: "
            f"{PRODUCTION_PROJECTION_PATH}"
        )

    production = pd.read_parquet(
        PRODUCTION_PROJECTION_PATH
    )

    required = {
        "player_display_name",
        "position",
        "team",
        "production_status",
    }

    missing = required - set(production.columns)

    if missing:
        raise RuntimeError(
            "Production projection missing required columns: "
            + ", ".join(sorted(missing))
        )

    current = production[
        production["production_status"]
        .astype(str)
        .str.upper()
        .eq("COLD_START")
    ].copy()

    current = current[
        current["position"].isin(
            OFFENSIVE_POSITIONS
        )
    ].copy()

    return current


def load_current_depth(conn):
    depth = pd.read_sql_query(
        """
        SELECT
            snapshot_dt,
            team,
            player_name,
            gsis_id,
            pos_abb,
            pos_slot,
            pos_rank
        FROM depth_charts
        WHERE pos_abb IN ('QB','RB','WR','TE')
        """,
        conn,
    )

    depth["snapshot_dt"] = pd.to_datetime(
        depth["snapshot_dt"],
        utc=True,
        errors="coerce",
    )

    depth["gsis_id"] = depth["gsis_id"].map(
        normalize_id
    )

    return depth


def resolve_player_id(row):
    for col in [
        "player_id",
        "gsis_id",
    ]:
        if col in row.index:
            value = normalize_id(row[col])
            if value:
                return value

    identity_key = row.get("identity_key")

    if pd.notna(identity_key):
        identity_key = str(identity_key)

        if identity_key.startswith("GSIS:"):
            return normalize_id(
                identity_key.split(":", 1)[1]
            )

    return None


def get_current_role(row, depth):
    player_id = resolve_player_id(row)

    if not player_id:
        return None

    matches = depth[
        (depth["gsis_id"] == player_id)
        & (depth["pos_abb"] == row["position"])
    ].copy()

    if matches.empty:
        return None

    # Prefer current team when available.
    team_matches = matches[
        matches["team"] == row["team"]
    ]

    if not team_matches.empty:
        matches = team_matches

    matches = matches.sort_values(
        "snapshot_dt"
    )

    latest = matches.iloc[-1]

    final_rank = int(latest["pos_rank"])

    # Deduplicate snapshots before role-history analysis.
    matches = matches.drop_duplicates(
        subset=[
            "snapshot_dt",
            "pos_slot",
            "pos_rank",
        ],
        keep="last",
    ).sort_values("snapshot_dt")

    ranks = matches["pos_rank"].tolist()

    start_idx = len(ranks) - 1

    while (
        start_idx > 0
        and ranks[start_idx - 1] == final_rank
    ):
        start_idx -= 1

    current_run = matches.iloc[start_idx:]

    previous_rank = None

    if start_idx > 0:
        previous_rank = int(
            matches.iloc[start_idx - 1]["pos_rank"]
        )

    if previous_rank is None:
        movement = "STABLE"
    elif final_rank < previous_rank:
        movement = "PROMOTED"
    elif final_rank > previous_rank:
        movement = "DEMOTED"
    else:
        movement = "STABLE"

    return {
        "gsis_id": player_id,
        "depth_snapshot_dt": latest["snapshot_dt"],
        "pos_slot": int(latest["pos_slot"]),
        "pos_rank": final_rank,
        "previous_rank": previous_rank,
        "movement": movement,
        "current_rank_snapshots": len(current_run),
        "rank_change_dt": current_run.iloc[0]["snapshot_dt"],
    }


def build_current_estimates(
    current,
    depth,
    calibration,
):
    rows = []

    for _, player in current.iterrows():

        role = get_current_role(
            player,
            depth,
        )

        base = {
            "player_display_name":
                player["player_display_name"],
            "team":
                player["team"],
            "position":
                player["position"],
            "production_status":
                player["production_status"],
        }

        if role is None:
            rows.append(
                {
                    **base,
                    "cold_start_status":
                        "NO_EXACT_DEPTH_ROLE",
                    "cold_start_estimate":
                        np.nan,
                }
            )
            continue

        bucket = calibration[
            (calibration["position"]
                == player["position"])
            & (calibration["pos_rank"]
                == role["pos_rank"])
        ]

        if bucket.empty:
            rows.append(
                {
                    **base,
                    **role,
                    "cold_start_status":
                        "NO_HISTORICAL_BUCKET",
                    "cold_start_estimate":
                        np.nan,
                }
            )
            continue

        bucket = bucket.iloc[0]

        rows.append(
            {
                **base,
                **role,
                "historical_rank_n":
                    int(bucket["rank_n"]),
                "historical_rank_mean":
                    float(bucket["rank_mean"]),
                "historical_rank_median":
                    float(bucket["rank_median"]),
                "historical_position_n":
                    int(bucket["position_n"]),
                "historical_position_mean":
                    float(bucket["position_mean"]),
                "shrink_k":
                    SHRINK_K,
                "cold_start_estimate":
                    float(
                        bucket[
                            "cold_start_estimate"
                        ]
                    ),
                "cold_start_status":
                    "COLD_START ESTIMATE",
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "=== WFS COLD-START PROJECTION v1 ==="
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:

        cohort = load_historical_cohort(
            conn
        )

        calibration = build_calibration_table(
            cohort
        )

        current = load_current_cold_starts()

        depth = load_current_depth(conn)

        output = build_current_estimates(
            current,
            depth,
            calibration,
        )

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nHistorical cohort:",
        len(cohort),
    )

    print(
        "Current cold starts:",
        len(current),
    )

    print(
        "Estimated:",
        int(
            output[
                "cold_start_estimate"
            ].notna().sum()
        ),
    )

    print(
        "Unresolved:",
        int(
            output[
                "cold_start_estimate"
            ].isna().sum()
        ),
    )

    output.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    calibration.to_csv(
        AUDIT_CSV,
        index=False,
    )

    print(
        f"\nParquet: {OUTPUT_PARQUET}"
    )

    print(
        f"CSV:     {OUTPUT_CSV}"
    )

    print(
        f"Audit:   {AUDIT_CSV}"
    )

    print(
        "\n=== TARGET PLAYERS ==="
    )

    targets = {
        "Jadarian Price",
        "Caleb Douglas",
        "Ja'Kobi Lane",
    }

    target_rows = output[
        output[
            "player_display_name"
        ].isin(targets)
    ].copy()

    columns = [
        "player_display_name",
        "team",
        "position",
        "pos_rank",
        "previous_rank",
        "movement",
        "historical_rank_n",
        "historical_rank_mean",
        "historical_position_mean",
        "cold_start_estimate",
        "cold_start_status",
    ]

    columns = [
        c for c in columns
        if c in target_rows.columns
    ]

    print(
        target_rows[columns]
        .sort_values(
            "player_display_name"
        )
        .to_string(index=False)
    )

    print(
        "\n=== CALIBRATION TABLE ==="
    )

    print(
        calibration[
            [
                "position",
                "pos_rank",
                "rank_n",
                "rank_mean",
                "rank_median",
                "position_mean",
                "cold_start_estimate",
            ]
        ]
        .sort_values(
            ["position", "pos_rank"]
        )
        .to_string(index=False)
    )

    print(
        "\nCold-Start v1 build complete."
    )


if __name__ == "__main__":
    main()
