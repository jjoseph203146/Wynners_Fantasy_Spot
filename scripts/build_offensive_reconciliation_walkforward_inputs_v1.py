#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_inputs_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_inputs_v1_audit.json"
)

POSITIONS = {"QB", "RB", "FB", "HB", "WR", "TE"}

HARD_OUT = {
    "OUT",
    "INACTIVE",
    "RESERVE",
    "IR",
    "PUP",
    "SUSPENDED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_position(value: object) -> str:
    text = str(value or "").strip().upper()

    if text in {"FB", "HB"}:
        return "RB"

    return text


def num(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return float(default)

    return float(parsed)


def recent_values(
    prior: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    values = pd.to_numeric(
        prior[column],
        errors="coerce",
    ).fillna(0.0)

    if values.empty:
        return 0.0, 0.0, 0.0

    return (
        float(values.iloc[-1]),
        float(values.tail(3).mean()),
        float(values.tail(5).mean()),
    )


def main() -> None:
    if not DB.is_file():
        raise FileNotFoundError(DB)

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                away_team,
                home_team,
                away_qb_id,
                home_qb_id,
                away_qb_name,
                home_qb_name
            FROM games
            WHERE (
                season = 2025
                AND week BETWEEN 8 AND 18
            )
            OR (
                season = 2026
                AND week = 1
            )
            ORDER BY season, week, game_id
            """,
            conn,
        )

        rosters = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                full_name,
                position,
                depth_chart_position,
                status,
                status_description_abbr
            FROM weekly_rosters
            WHERE (
                season = 2025
                AND week BETWEEN 8 AND 18
            )
            OR (
                season = 2026
                AND week = 1
            )
            """,
            conn,
        )

        injuries = pd.read_sql_query(
            """
            SELECT
                season,
                week,
                team,
                gsis_id,
                report_status,
                practice_status,
                report_primary_injury
            FROM injuries
            WHERE (
                season = 2025
                AND week BETWEEN 8 AND 18
            )
            OR (
                season = 2026
                AND week = 1
            )
            """,
            conn,
        )

        depth = pd.read_sql_query(
            """
            SELECT
                snapshot_dt,
                team,
                player_name,
                gsis_id,
                pos_abb,
                pos_rank
            FROM depth_charts
            WHERE pos_abb IN (
                'QB', 'RB', 'WR', 'TE'
            )
            """,
            conn,
        )

        player_stats = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                team,
                opponent_team,
                player_id,
                player_display_name,
                position,
                attempts,
                completions,
                passing_yards,
                passing_tds,
                passing_interceptions,
                carries,
                rushing_yards,
                rushing_tds,
                targets,
                receptions,
                receiving_yards,
                receiving_tds,
                fanduel_points
            FROM player_game_stats
            WHERE player_id IS NOT NULL
            """,
            conn,
        )

    games["game_date"] = pd.to_datetime(
        games["game_date"],
        utc=True,
        errors="coerce",
    )

    depth["snapshot_dt"] = pd.to_datetime(
        depth["snapshot_dt"],
        utc=True,
        errors="coerce",
    )

    for frame in (
        games,
        rosters,
        injuries,
        player_stats,
    ):
        if "season" in frame.columns:
            frame["season"] = pd.to_numeric(
                frame["season"],
                errors="coerce",
            )

        if "week" in frame.columns:
            frame["week"] = pd.to_numeric(
                frame["week"],
                errors="coerce",
            )

    rosters["position"] = [
        normalize_position(
            depth_position
            if pd.notna(depth_position)
            else position
        )
        for depth_position, position
        in zip(
            rosters["depth_chart_position"],
            rosters["position"],
        )
    ]

    rosters = rosters[
        rosters["position"].isin(
            {"QB", "RB", "WR", "TE"}
        )
        & rosters["status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("ACT")
    ].copy()

    injuries = injuries.sort_values(
        ["season", "week", "team", "gsis_id"]
    ).drop_duplicates(
        ["season", "week", "team", "gsis_id"],
        keep="last",
    )

    injuries["report_status_normalized"] = (
        injuries["report_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    team_games = pd.concat([
        games[
            [
                "game_id",
                "season",
                "week",
                "game_date",
                "away_team",
                "home_team",
                "away_qb_id",
                "away_qb_name",
            ]
        ].rename(columns={
            "away_team": "team",
            "home_team": "opponent_team",
            "away_qb_id": "primary_qb_id",
            "away_qb_name": "primary_qb_name",
        }),
        games[
            [
                "game_id",
                "season",
                "week",
                "game_date",
                "home_team",
                "away_team",
                "home_qb_id",
                "home_qb_name",
            ]
        ].rename(columns={
            "home_team": "team",
            "away_team": "opponent_team",
            "home_qb_id": "primary_qb_id",
            "home_qb_name": "primary_qb_name",
        }),
    ], ignore_index=True)

    pool = team_games.merge(
        rosters,
        on=["season", "week", "team"],
        how="left",
    )

    pool = pool.merge(
        injuries[
            [
                "season",
                "week",
                "team",
                "gsis_id",
                "report_status",
                "practice_status",
                "report_primary_injury",
                "report_status_normalized",
            ]
        ],
        on=["season", "week", "team", "gsis_id"],
        how="left",
    )

    pool["hard_out_flag"] = (
        pool["report_status_normalized"]
        .fillna("")
        .isin(HARD_OUT)
        .astype(int)
    )

    pool = pool[
        pool["hard_out_flag"].eq(0)
    ].copy()

    pool["player_id"] = (
        pool["gsis_id"]
        .fillna("")
        .astype(str)
    )

    pool["player_name"] = (
        pool["full_name"]
        .fillna("")
        .astype(str)
    )

    # Fail-safe: insert the authoritative scheduled QB when the
    # weekly roster snapshot does not contain that QB.
    existing_keys = set(
        zip(
            pool["game_id"].astype(str),
            pool["team"].astype(str),
            pool["player_id"].astype(str),
        )
    )

    authority_rows = []

    for row in team_games.itertuples(index=False):
        key = (
            str(row.game_id),
            str(row.team),
            str(row.primary_qb_id),
        )

        if key in existing_keys:
            continue

        authority_rows.append({
            "game_id": row.game_id,
            "season": row.season,
            "week": row.week,
            "game_date": row.game_date,
            "team": row.team,
            "opponent_team": row.opponent_team,
            "primary_qb_id": row.primary_qb_id,
            "primary_qb_name": row.primary_qb_name,
            "gsis_id": row.primary_qb_id,
            "full_name": row.primary_qb_name,
            "position": "QB",
            "depth_chart_position": "QB",
            "status": "QB_AUTHORITY",
            "status_description_abbr": "",
            "report_status": "",
            "practice_status": "",
            "report_primary_injury": "",
            "report_status_normalized": "",
            "hard_out_flag": 0,
            "player_id": str(row.primary_qb_id),
            "player_name": str(row.primary_qb_name),
        })

    if authority_rows:
        pool = pd.concat(
            [
                pool,
                pd.DataFrame(authority_rows),
            ],
            ignore_index=True,
            sort=False,
        )

    pool["position"] = pool["position"].map(
        normalize_position
    )

    # Attach the latest depth snapshot that existed before the game.
    depth_rows = []

    for team_game in team_games.itertuples(index=False):
        eligible = depth[
            depth["team"].astype(str).eq(
                str(team_game.team)
            )
            & depth["snapshot_dt"].le(
                team_game.game_date
            )
        ]

        if eligible.empty:
            continue

        snapshot_dt = eligible["snapshot_dt"].max()

        latest = eligible[
            eligible["snapshot_dt"].eq(snapshot_dt)
        ].copy()

        # A hybrid player can occupy multiple depth slots in the
        # same snapshot. Keep exactly one pregame record, preferring
        # the smallest rank (the highest-priority listed role).
        latest["pos_rank"] = pd.to_numeric(
            latest["pos_rank"],
            errors="coerce",
        )

        latest = latest.sort_values(
            ["gsis_id", "pos_rank", "pos_abb"],
            na_position="last",
        ).drop_duplicates(
            ["gsis_id"],
            keep="first",
        )

        latest["game_id"] = team_game.game_id
        latest["depth_snapshot_dt"] = snapshot_dt

        depth_rows.append(
            latest[
                [
                    "game_id",
                    "team",
                    "gsis_id",
                    "pos_abb",
                    "pos_rank",
                    "depth_snapshot_dt",
                ]
            ]
        )

    depth_asof = pd.concat(
        depth_rows,
        ignore_index=True,
    )

    pool = pool.merge(
        depth_asof,
        on=["game_id", "team", "gsis_id"],
        how="left",
    )

    pool["depth_position"] = (
        pool["pos_abb"]
        .fillna(pool["position"])
        .map(normalize_position)
    )

    pool["depth_rank"] = pd.to_numeric(
        pool["pos_rank"],
        errors="coerce",
    )

    pool.loc[
        pool["depth_rank"].isna(),
        "depth_rank",
    ] = (
        pool[
            pool["depth_rank"].isna()
        ]
        .groupby(
            ["game_id", "team", "position"]
        )
        .cumcount()
        + 5
    )

    stat_columns = [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fanduel_points",
    ]

    for column in stat_columns:
        player_stats[column] = pd.to_numeric(
            player_stats[column],
            errors="coerce",
        ).fillna(0.0)

    player_stats = player_stats.sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)

    history_by_player = {
        str(player_id): block.copy()
        for player_id, block
        in player_stats.groupby(
            player_stats["player_id"].astype(str)
        )
    }

    feature_rows = []

    for row in pool.itertuples(index=False):
        history = history_by_player.get(
            str(row.player_id),
            player_stats.iloc[0:0],
        )

        prior = history[
            (history["season"] < row.season)
            | (
                history["season"].eq(row.season)
                & history["week"].lt(row.week)
            )
        ].sort_values(
            ["season", "week", "game_id"]
        )

        record = {
            "game_id": row.game_id,
            "season": int(row.season),
            "week": int(row.week),
            "game_date": row.game_date,
            "team": row.team,
            "opponent_team": row.opponent_team,
            "player_id": row.player_id,
            "player_name": row.player_name,
            "position": row.position,
            "primary_qb_id": row.primary_qb_id,
            "primary_qb_flag": int(
                str(row.player_id)
                == str(row.primary_qb_id)
            ),
            "roster_status": row.status,
            "report_status": row.report_status,
            "practice_status": row.practice_status,
            "primary_injury": row.report_primary_injury,
            "depth_snapshot_dt": row.depth_snapshot_dt,
            "depth_rank": num(row.depth_rank, 5.0),
            "history_games": int(len(prior)),
        }

        if len(prior) == 0:
            record["history_class"] = "COLD_START_0"
        elif len(prior) <= 4:
            record["history_class"] = "SPARSE_1_TO_4"
        else:
            record["history_class"] = "ESTABLISHED_5_PLUS"

        for column in stat_columns:
            last, avg_3, avg_5 = recent_values(
                prior,
                column,
            )

            record[f"{column}_last"] = last
            record[f"{column}_avg_3"] = avg_3
            record[f"{column}_avg_5"] = avg_5

        feature_rows.append(record)

    result = pd.DataFrame(feature_rows)

    actual = player_stats[
        player_stats["game_id"].isin(
            games["game_id"]
        )
    ][
        [
            "game_id",
            "team",
            "player_id",
            *stat_columns,
        ]
    ].copy()

    actual = actual.rename(
        columns={
            column: f"actual_{column}"
            for column in stat_columns
        }
    )

    result = result.merge(
        actual,
        on=["game_id", "team", "player_id"],
        how="left",
    )

    actual_columns = [
        f"actual_{column}"
        for column in stat_columns
    ]

    result[actual_columns] = (
        result[actual_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["actual_participant_flag"] = (
        result[actual_columns]
        .abs()
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    # A zero-stat player may still have appeared. Preserve that fact.
    actual_keys = set(
        zip(
            actual["game_id"].astype(str),
            actual["team"].astype(str),
            actual["player_id"].astype(str),
        )
    )

    result["actual_participant_flag"] = [
        int(
            (
                str(game_id),
                str(team),
                str(player_id),
            ) in actual_keys
        )
        for game_id, team, player_id
        in zip(
            result["game_id"],
            result["team"],
            result["player_id"],
        )
    ]

    duplicate_keys = int(
        result.duplicated(
            ["game_id", "team", "player_id"]
        ).sum()
    )

    primary_counts = result.groupby(
        ["game_id", "team"]
    )["primary_qb_flag"].sum()

    target_keys = set(
        games["game_id"].astype(str)
    )

    leakage_checks = {}

    for column in (
        "attempts",
        "carries",
        "targets",
        "fanduel_points",
    ):
        leakage_checks[column] = int(
            result[f"{column}_last"].isna().sum()
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = result.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
            "position",
            "depth_rank",
            "player_name",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT,
        index=False,
    )

    actual_target_rows = player_stats[
        player_stats["game_id"].astype(str).isin(
            target_keys
        )
        & player_stats["position"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(POSITIONS)
    ].drop_duplicates(
        ["game_id", "team", "player_id"]
    )

    found_actual_keys = set(
        zip(
            result["game_id"].astype(str),
            result["team"].astype(str),
            result["player_id"].astype(str),
        )
    )

    actual_key_set = set(
        zip(
            actual_target_rows["game_id"].astype(str),
            actual_target_rows["team"].astype(str),
            actual_target_rows["player_id"].astype(str),
        )
    )

    participant_coverage = (
        len(found_actual_keys & actual_key_set)
        / max(len(actual_key_set), 1)
    )

    hard_failures = {
        "duplicate_player_keys": duplicate_keys,
        "team_games_missing_primary_qb": int(
            primary_counts.ne(1).sum()
        ),
        "team_games": int(
            result.groupby(
                ["game_id", "team"]
            ).ngroups
        ),
        "games": int(
            result["game_id"].nunique()
        ),
    }

    status = (
        "PASS"
        if (
            hard_failures["duplicate_player_keys"] == 0
            and hard_failures[
                "team_games_missing_primary_qb"
            ] == 0
            and hard_failures["team_games"] == 360
            and hard_failures["games"] == 180
            and participant_coverage >= 0.99
        )
        else "FAIL"
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_WALKFORWARD_INPUTS_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "target_games": int(
            result["game_id"].nunique()
        ),
        "team_games": int(
            result.groupby(
                ["game_id", "team"]
            ).ngroups
        ),
        "rows": int(len(result)),
        "players": int(
            result["player_id"].nunique()
        ),
        "participant_coverage":
            float(participant_coverage),
        "history_classes": {
            str(key): int(value)
            for key, value
            in result[
                "history_class"
            ].value_counts().items()
        },
        "hard_failures": hard_failures,
        "output": {
            "path": str(OUTPUT.resolve()),
            "sha256": sha256(OUTPUT),
        },
    }

    AUDIT_OUTPUT.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "WFS OFFENSIVE RECONCILIATION "
        "WALK-FORWARD INPUTS V1"
    )
    print("=" * 80)
    print(f"ROWS={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(
        "TEAM_GAMES="
        f"{result.groupby(['game_id','team']).ngroups}"
    )
    print(
        "PARTICIPANT_COVERAGE="
        f"{100 * participant_coverage:.4f}%"
    )
    print(
        "PRIMARY_QB_MIN="
        f"{int(primary_counts.min())}"
    )
    print(
        "PRIMARY_QB_MAX="
        f"{int(primary_counts.max())}"
    )
    print(
        "DUPLICATE_PLAYER_KEYS="
        f"{duplicate_keys}"
    )

    print("\n=== HISTORY CLASSES ===")
    print(
        result[
            "history_class"
        ].value_counts().to_string()
    )

    print("\n=== POOL SIZE BY TEAM-GAME ===")
    sizes = result.groupby(
        ["game_id", "team"]
    ).size()

    print(
        f"MIN={sizes.min()} "
        f"P50={sizes.median():.1f} "
        f"P95={sizes.quantile(0.95):.1f} "
        f"MAX={sizes.max()}"
    )

    print(f"\nOUTPUT={OUTPUT}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(f"WALKFORWARD_INPUT_STATUS={status}")

    if status != "PASS":
        raise RuntimeError(
            "Walk-forward input contract failed"
        )


if __name__ == "__main__":
    main()
