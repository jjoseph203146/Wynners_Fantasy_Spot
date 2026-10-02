#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from database import get_connection
from team_environment import (
    build_team_environment,
    load_team_stats,
    prepare_team_stats,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_team_environment.parquet"
)

REG_GAME_TYPE = "REG"


def section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def load_schedule() -> pd.DataFrame:
    with get_connection() as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                game_type,
                week,
                game_date,
                away_team,
                home_team,
                completed
            FROM games
            WHERE game_id IS NOT NULL
            """,
            conn,
        )

    if games.empty:
        raise RuntimeError("No schedule rows available.")

    games["season"] = pd.to_numeric(
        games["season"], errors="raise"
    ).astype(int)

    games["week"] = pd.to_numeric(
        games["week"], errors="raise"
    ).astype(int)

    games["completed"] = pd.to_numeric(
        games["completed"], errors="coerce"
    ).fillna(0).astype(int)

    return games


def resolve_active_target_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    regular = games[
        games["game_type"].astype(str).str.upper().eq(
            REG_GAME_TYPE
        )
    ].copy()

    if regular.empty:
        raise RuntimeError(
            "No regular-season schedule rows available."
        )

    current_season = int(regular["season"].max())

    unfinished = regular[
        (regular["season"] == current_season)
        & (regular["completed"] == 0)
    ].copy()

    if unfinished.empty:
        raise RuntimeError(
            f"No unfinished REG games for season "
            f"{current_season}."
        )

    active_week = int(unfinished["week"].min())

    target = unfinished[
        unfinished["week"] == active_week
    ].copy()

    if target.empty:
        raise RuntimeError(
            "Active target resolution produced zero games."
        )

    if target["game_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate game_id in active target schedule."
        )

    print(
        f"Active target: season={current_season} "
        f"week={active_week} games={len(target)}"
    )

    return target.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)


def resolve_regression_target_games(
    games: pd.DataFrame,
    season: int,
    week: int,
    as_of_date: str,
) -> pd.DataFrame:
    regular = games[
        games["game_type"].astype(str).str.upper().eq(
            REG_GAME_TYPE
        )
    ].copy()

    target = regular[
        (regular["season"] == int(season))
        & (regular["week"] == int(week))
        & (
            pd.to_datetime(
                regular["game_date"],
                errors="coerce",
            )
            >= pd.Timestamp(as_of_date)
        )
    ].copy()

    if target.empty:
        raise RuntimeError(
            "Regression target produced zero games."
        )

    if target["game_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate game_id in regression target."
        )

    print(
        f"Regression target: season={season} "
        f"week={week} as_of_date={as_of_date} "
        f"games={len(target)}"
    )

    return target.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)


def build_target_rows(
    target_games: pd.DataFrame,
    prepared_history: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, game in target_games.iterrows():
        game_id = str(game["game_id"])
        season = int(game["season"])
        week = int(game["week"])
        away = str(game["away_team"]).strip()
        home = str(game["home_team"]).strip()

        if not away or not home:
            raise RuntimeError(
                f"Blank team identity for {game_id}."
            )

        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "team": away,
                "opponent_team": home,
            }
        )

        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "team": home,
                "opponent_team": away,
            }
        )

    targets = pd.DataFrame(rows)

    if targets.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate game/team target identity."
        )

    # build_team_environment() expects the prepared
    # historical metric columns to exist on every row.
    # Current target rows carry no outcome information.
    # NaN is intentional: lagged calculations use only
    # rows before the target game.
    for column in prepared_history.columns:
        if column not in targets.columns:
            targets[column] = np.nan

    targets = targets[
        prepared_history.columns
    ].copy()

    return targets


def build_current_environment(
    target_games: pd.DataFrame,
) -> pd.DataFrame:
    section("LOAD HISTORICAL TEAM AUTHORITY")

    raw_history = load_team_stats()
    prepared = prepare_team_stats(raw_history)

    target_ids = set(
        target_games["game_id"].astype(str)
    )

    # Critical leakage protection:
    # if target games are already represented in historical
    # team stats, remove those rows before appending blank
    # inference targets.
    history = prepared[
        ~prepared["game_id"].astype(str).isin(
            target_ids
        )
    ].copy()

    target_rows = build_target_rows(
        target_games,
        history,
    )

    combined = pd.concat(
        [history, target_rows],
        ignore_index=True,
        sort=False,
    )

    section("BUILD CURRENT TEAM ENVIRONMENT")

    environment = build_team_environment(
        combined
    )

    current = environment[
        environment["game_id"]
        .astype(str)
        .isin(target_ids)
    ].copy()

    expected_rows = len(target_games) * 2

    if len(current) != expected_rows:
        raise RuntimeError(
            f"Current environment row mismatch: "
            f"expected={expected_rows} "
            f"actual={len(current)}"
        )

    if current.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate game/team in current environment."
        )

    expected_identity = set()

    for _, game in target_games.iterrows():
        gid = str(game["game_id"])
        expected_identity.add(
            (gid, str(game["away_team"]).strip())
        )
        expected_identity.add(
            (gid, str(game["home_team"]).strip())
        )

    actual_identity = set(
        zip(
            current["game_id"].astype(str),
            current["team"].astype(str),
        )
    )

    if actual_identity != expected_identity:
        missing = sorted(
            expected_identity - actual_identity
        )
        extra = sorted(
            actual_identity - expected_identity
        )

        raise RuntimeError(
            "Current environment identity mismatch. "
            f"missing={missing} extra={extra}"
        )

    output_columns = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
        "history_games",
        "points_for_last",
        "points_for_avg_3",
        "points_for_avg_5",
        "points_against_last",
        "points_against_avg_3",
        "points_against_avg_5",
        "offensive_plays_avg_3",
        "offensive_plays_avg_5",
        "pass_attempts_avg_3",
        "pass_attempts_avg_5",
        "rush_attempts_avg_3",
        "rush_attempts_avg_5",
        "pass_rate_avg_3",
        "pass_rate_avg_5",
        "rush_rate_avg_3",
        "rush_rate_avg_5",
        "passing_yards_avg_3",
        "rushing_yards_avg_3",
        "passing_tds_avg_3",
        "rushing_tds_avg_3",
        "opponent_points_allowed_avg_3",
        "opponent_points_allowed_avg_5",
        "opponent_pass_yards_allowed_avg_3",
        "opponent_rush_yards_allowed_avg_3",
        "opponent_pass_tds_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",
        "team_scoring_trend",
        "opponent_scoring_trend",
        "pace_trend",
        "pass_rate_trend",
    ]

    missing_columns = [
        c
        for c in output_columns
        if c not in current.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Current environment missing columns: "
            + ", ".join(missing_columns)
        )

    current = current[
        output_columns
    ].copy()

    numeric_columns = [
        c
        for c in output_columns
        if c
        not in {
            "game_id",
            "team",
            "opponent_team",
        }
    ]

    numeric = current[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        bad = numeric.columns[
            numeric.isna().any()
        ].tolist()

        raise RuntimeError(
            "Current environment contains null numeric "
            f"values: {bad}"
        )

    values = numeric.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Current environment contains "
            "nonfinite numeric values."
        )

    return current.sort_values(
        ["game_id", "team"]
    ).reset_index(drop=True)


def compare_frozen(
    current: pd.DataFrame,
    frozen_path: Path,
) -> None:
    section("FROZEN REGRESSION")

    frozen = pd.read_parquet(
        frozen_path
    ).copy()

    if "history_games_team" in frozen.columns:
        frozen = frozen.rename(
            columns={
                "history_games_team":
                    "history_games"
            }
        )

    common = [
        c
        for c in current.columns
        if c in frozen.columns
    ]

    identity = [
        "game_id",
        "team",
    ]

    left = current[
        common
    ].copy()

    right = frozen[
        common
    ].copy()

    merged = left.merge(
        right,
        on=identity,
        how="outer",
        suffixes=("_new", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )

    identity_failures = int(
        (merged["_merge"] != "both").sum()
    )

    print(
        "IDENTITY_FAILURES =",
        identity_failures,
    )

    if identity_failures:
        print(
            merged[
                merged["_merge"] != "both"
            ][
                identity + ["_merge"]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "Frozen identity regression failed."
        )

    numeric_compare = [
        c
        for c in common
        if c not in {
            "game_id",
            "team",
            "opponent_team",
        }
    ]

    max_delta = 0.0
    mismatch_columns = []

    for column in numeric_compare:
        a = pd.to_numeric(
            merged[f"{column}_new"],
            errors="coerce",
        )

        b = pd.to_numeric(
            merged[f"{column}_frozen"],
            errors="coerce",
        )

        delta = (
            a - b
        ).abs()

        column_max = float(
            delta.max()
        ) if len(delta) else 0.0

        max_delta = max(
            max_delta,
            column_max,
        )

        if (delta > 1e-10).any():
            mismatch_columns.append(
                (column, column_max)
            )

    print(
        "MAX_ABS_NUMERIC_DELTA =",
        max_delta,
    )

    print(
        "MISMATCH_COLUMNS_GT_1E-10 =",
        mismatch_columns,
    )

    if mismatch_columns:
        raise RuntimeError(
            "Frozen numeric regression failed."
        )

    print(
        "FROZEN_REGRESSION=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--regression-week1",
        action="store_true",
    )

    parser.add_argument(
        "--frozen",
        default=str(
            ROOT
            / "data"
            / "model_candidates"
            / "stat_forecast"
            / "stage24j_c_r7_r12_20260912T185444Z"
            / "current_team_environment_28.parquet"
        ),
    )

    args = parser.parse_args()

    games = load_schedule()

    if args.regression_week1:
        target_games = resolve_regression_target_games(
            games=games,
            season=2026,
            week=1,
            as_of_date="2026-09-12",
        )
    else:
        target_games = resolve_active_target_games(
            games
        )

    current = build_current_environment(
        target_games
    )

    section("CURRENT TEAM ENVIRONMENT SUMMARY")

    print(
        "ROWS =",
        len(current),
    )
    print(
        "GAMES =",
        current["game_id"].nunique(),
    )
    print(
        "TEAMS =",
        current["team"].nunique(),
    )
    print(
        "SEASONS =",
        sorted(
            current["season"]
            .astype(int)
            .unique()
            .tolist()
        ),
    )
    print(
        "WEEKS =",
        sorted(
            current["week"]
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    if args.regression_week1:
        compare_frozen(
            current,
            Path(args.frozen),
        )
        return

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output.with_suffix(
        output.suffix + ".tmp"
    )

    current.to_parquet(
        temp,
        index=False,
    )

    temp.replace(output)

    print()
    print(
        "PUBLISHED =",
        output,
    )


if __name__ == "__main__":
    main()
