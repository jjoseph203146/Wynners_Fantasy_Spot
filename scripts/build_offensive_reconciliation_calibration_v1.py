#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"
CURRENT = (
    ROOT
    / "data"
    / "parquet"
    / "current_unified_stat_forecasts.parquet"
)

TEAM_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_team_history_v1.csv"
)

PLAYER_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_player_roles_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_calibration_v1_audit.json"
)

SEASONS = (2023, 2024, 2025, 2026)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_table(
    conn: sqlite3.Connection,
    table: str,
) -> pd.DataFrame:
    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table,),
    ).fetchone()

    if not exists:
        raise RuntimeError(
            f"Missing required table: {table}"
        )

    return pd.read_sql_query(
        f'SELECT * FROM "{table}"',
        conn,
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )


def numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)


def quantiles(
    series: pd.Series,
) -> dict[str, float]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    return {
        key: float(value)
        for key, value in {
            "p01": values.quantile(0.01),
            "p05": values.quantile(0.05),
            "p10": values.quantile(0.10),
            "p25": values.quantile(0.25),
            "p50": values.quantile(0.50),
            "p75": values.quantile(0.75),
            "p90": values.quantile(0.90),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
        }.items()
    }


def main() -> None:
    for required_path in (
        DB,
        CURRENT,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    current = pd.read_parquet(CURRENT)

    require_columns(
        current,
        {
            "entity_type",
            "game_id",
        },
        "current_unified_stat_forecasts",
    )

    current_offense = current[
        current["entity_type"]
        .astype(str)
        .eq("OFFENSE_PLAYER")
    ].copy()

    target_game_ids = {
        str(value).strip()
        for value in current_offense[
            "game_id"
        ].dropna()
        if str(value).strip()
    }

    if not target_game_ids:
        raise RuntimeError(
            "No current offensive target games were found."
        )

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:
        team = read_table(
            conn,
            "team_game_stats",
        )

        player = read_table(
            conn,
            "player_game_stats",
        )

    require_columns(
        team,
        {
            "game_id",
            "season",
            "week",
            "team",
            "attempts",
            "completions",
            "carries",
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
        },
        "team_game_stats",
    )

    require_columns(
        player,
        {
            "game_id",
            "season",
            "week",
            "team",
            "player_id",
            "player_display_name",
            "position",
            "attempts",
            "completions",
            "carries",
            "passing_yards",
            "passing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
            "rushing_yards",
            "rushing_tds",
        },
        "player_game_stats",
    )

    team = team[
        pd.to_numeric(
            team["season"],
            errors="coerce",
        ).isin(SEASONS)
    ].copy()

    player = player[
        pd.to_numeric(
            player["season"],
            errors="coerce",
        ).isin(SEASONS)
    ].copy()

    team_target_overlap = (
        team["game_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(target_game_ids)
    )

    player_target_overlap = (
        player["game_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(target_game_ids)
    )

    excluded_team_rows = int(
        team_target_overlap.sum()
    )
    excluded_player_rows = int(
        player_target_overlap.sum()
    )

    excluded_target_games = sorted(
        set(
            team.loc[
                team_target_overlap,
                "game_id",
            ].astype(str)
        )
        | set(
            player.loc[
                player_target_overlap,
                "game_id",
            ].astype(str)
        )
    )

    team = team[
        ~team_target_overlap
    ].copy()

    player = player[
        ~player_target_overlap
    ].copy()

    remaining_team_overlap = (
        set(
            team["game_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        & target_game_ids
    )

    remaining_player_overlap = (
        set(
            player["game_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        & target_game_ids
    )

    if (
        remaining_team_overlap
        or remaining_player_overlap
    ):
        raise RuntimeError(
            "Target-game exclusion failed. "
            f"team={sorted(remaining_team_overlap)} "
            f"player={sorted(remaining_player_overlap)}"
        )

    team_numeric = [
        "attempts",
        "completions",
        "carries",
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
    ]

    player_numeric = [
        "attempts",
        "completions",
        "carries",
        "passing_yards",
        "passing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "rushing_yards",
        "rushing_tds",
    ]

    numeric(
        team,
        team_numeric,
    )

    numeric(
        player,
        player_numeric,
    )

    team["offensive_plays"] = (
        team["attempts"]
        + team["carries"]
    )

    team["completion_rate"] = (
        team["completions"]
        / team["attempts"].replace(0, np.nan)
    )

    team["yards_per_attempt"] = (
        team["passing_yards"]
        / team["attempts"].replace(0, np.nan)
    )

    team["yards_per_carry"] = (
        team["rushing_yards"]
        / team["carries"].replace(0, np.nan)
    )

    team["rush_rate"] = (
        team["carries"]
        / team["offensive_plays"].replace(
            0,
            np.nan,
        )
    )

    player["opportunities"] = (
        player["carries"]
        + player["targets"]
    )

    player["team_carries"] = player.groupby(
        ["game_id", "team"]
    )["carries"].transform("sum")

    player["team_targets"] = player.groupby(
        ["game_id", "team"]
    )["targets"].transform("sum")

    player["team_receptions"] = player.groupby(
        ["game_id", "team"]
    )["receptions"].transform("sum")

    player["team_receiving_yards"] = (
        player.groupby(
            ["game_id", "team"]
        )["receiving_yards"].transform("sum")
    )

    player["carry_share"] = (
        player["carries"]
        / player["team_carries"].replace(
            0,
            np.nan,
        )
    )

    player["target_share_calculated"] = (
        player["targets"]
        / player["team_targets"].replace(
            0,
            np.nan,
        )
    )

    player["reception_share"] = (
        player["receptions"]
        / player["team_receptions"].replace(
            0,
            np.nan,
        )
    )

    player["receiving_yard_share"] = (
        player["receiving_yards"]
        / player[
            "team_receiving_yards"
        ].replace(
            0,
            np.nan,
        )
    )

    player["yards_per_carry"] = (
        player["rushing_yards"]
        / player["carries"].replace(
            0,
            np.nan,
        )
    )

    player["yards_per_target"] = (
        player["receiving_yards"]
        / player["targets"].replace(
            0,
            np.nan,
        )
    )

    player["catch_rate"] = (
        player["receptions"]
        / player["targets"].replace(
            0,
            np.nan,
        )
    )

    team = team.sort_values(
        ["season", "week", "game_id", "team"]
    ).reset_index(drop=True)

    player = player.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
            "position",
            "player_id",
        ]
    ).reset_index(drop=True)

    TEAM_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    team.to_csv(
        TEAM_OUTPUT,
        index=False,
    )

    player.to_csv(
        PLAYER_OUTPUT,
        index=False,
    )

    team_metrics = [
        "offensive_plays",
        "attempts",
        "completions",
        "carries",
        "passing_yards",
        "rushing_yards",
        "completion_rate",
        "yards_per_attempt",
        "yards_per_carry",
        "rush_rate",
        "passing_tds",
        "rushing_tds",
    ]

    role_metrics = [
        "opportunities",
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "carry_share",
        "target_share_calculated",
        "reception_share",
        "receiving_yard_share",
        "yards_per_carry",
        "yards_per_target",
        "catch_rate",
    ]

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_CALIBRATION_V1",
        "status": "PASS",
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_forecast_modified": False,
        "target_exclusion": {
            "status": "PASS",
            "current_target_games": int(
                len(target_game_ids)
            ),
            "excluded_games": (
                excluded_target_games
            ),
            "excluded_team_rows": (
                excluded_team_rows
            ),
            "excluded_player_rows": (
                excluded_player_rows
            ),
            "remaining_team_overlap": 0,
            "remaining_player_overlap": 0,
        },
        "seasons": list(SEASONS),
        "team_rows": int(len(team)),
        "player_rows": int(len(player)),
        "games": int(team["game_id"].nunique()),
        "team_quantiles": {
            metric: quantiles(team[metric])
            for metric in team_metrics
        },
        "position_quantiles": {
            str(position): {
                metric: quantiles(
                    block[metric]
                )
                for metric in role_metrics
            }
            for position, block
            in player[
                player["position"].isin(
                    ["QB", "RB", "FB", "WR", "TE"]
                )
            ].groupby("position")
        },
        "week1_2026": {
            "team_rows": int(
                len(
                    team[
                        team["season"].eq(2026)
                        & team["week"].eq(1)
                    ]
                )
            ),
            "player_rows": int(
                len(
                    player[
                        player["season"].eq(2026)
                        & player["week"].eq(1)
                    ]
                )
            ),
        },
        "outputs": {
            "team": {
                "path": str(
                    TEAM_OUTPUT.resolve()
                ),
                "sha256": sha256(
                    TEAM_OUTPUT
                ),
            },
            "player": {
                "path": str(
                    PLAYER_OUTPUT.resolve()
                ),
                "sha256": sha256(
                    PLAYER_OUTPUT
                ),
            },
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
        "WFS ALL-OFFENSE RECONCILIATION "
        "CALIBRATION V1"
    )
    print("=" * 80)
    print(f"TEAM_ROWS={len(team)}")
    print(f"PLAYER_ROWS={len(player)}")
    print(
        "GAMES="
        f"{team['game_id'].nunique()}"
    )
    print(
        "WEEK1_2026_TEAM_ROWS="
        f"{audit['week1_2026']['team_rows']}"
    )
    print(
        "WEEK1_2026_PLAYER_ROWS="
        f"{audit['week1_2026']['player_rows']}"
    )

    for metric in (
        "offensive_plays",
        "attempts",
        "carries",
        "yards_per_attempt",
        "yards_per_carry",
        "rush_rate",
    ):
        values = audit[
            "team_quantiles"
        ][metric]

        print(
            f"{metric.upper()} "
            f"P05={values['p05']:.4f} "
            f"P50={values['p50']:.4f} "
            f"P95={values['p95']:.4f}"
        )

    print(f"TEAM_OUTPUT={TEAM_OUTPUT}")
    print(f"PLAYER_OUTPUT={PLAYER_OUTPUT}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_FORECAST_MODIFIED=FALSE")
    print("CALIBRATION_STATUS=PASS")


if __name__ == "__main__":
    main()
