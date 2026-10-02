#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import dst_history as hist
import dst_fanduel_scoring as scoring


ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"

OUTPUT = (
    ROOT
    / "data/parquet/"
    / "nfl_current_dst_postgame_actuals.parquet"
)

CURRENT_SEASON = 2026

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "team",
    "opponent_team",
    "home_away",
    "team_score",
    "opponent_score",
    "completed",
    "sacks",
    "interceptions",
    "fumble_recoveries",
    "safeties",
    "blocked_punts",
    "blocked_field_goals",
    "blocked_extra_points",
    "blocked_kicks",
    "defensive_tds",
    "special_teams_tds",
    "return_tds",
    "defensive_two_point_returns",
    "offensive_tds_allowed",
    "field_goals_allowed",
    "extra_points_allowed",
    "two_point_conversions_allowed",
    "fanduel_points_allowed",
    "sack_points",
    "interception_points",
    "fumble_recovery_points",
    "safety_points",
    "blocked_kick_points",
    "return_td_points",
    "defensive_two_point_return_points",
    "points_allowed_tier_points",
    "fanduel_dst_points",
    "fanduel_scoring_verified",
    "source",
    "updated_at",
    "fanduel_scored_at",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)
    tmp = Path(tmp_name)

    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_completed_games(
    conn: sqlite3.Connection,
    season: int,
) -> pd.DataFrame:

    df = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            game_type,
            home_team,
            away_team,
            home_score,
            away_score,
            completed
        FROM games
        WHERE season = ?
          AND UPPER(game_type) = 'REG'
          AND completed = 1
        ORDER BY week, game_id
        """,
        conn,
        params=[season],
    )

    if df.empty:
        raise RuntimeError(
            f"NO_COMPLETED_REGULAR_GAMES:{season}"
        )

    if df["game_id"].isna().any():
        raise RuntimeError("NULL_GAME_ID")

    if df["game_id"].duplicated().any():
        raise RuntimeError("DUPLICATE_GAME_ID")

    for c in [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]:
        if df[c].isna().any():
            raise RuntimeError(
                f"COMPLETED_GAME_MISSING_{c.upper()}"
            )

    return df


def build_current_actuals(
    season: int,
) -> pd.DataFrame:

    with sqlite3.connect(DB) as conn:
        games = load_completed_games(
            conn,
            season,
        )

    print(
        "COMPLETED_GAMES =",
        len(games),
    )

    print(
        "COMPLETED_WEEKS =",
        sorted(
            pd.to_numeric(
                games["week"],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    # Reuse the validated historical population builder.
    team_games = hist.build_expected_team_games(
        games
    )

    expected_rows = len(games) * 2

    if len(team_games) != expected_rows:
        raise RuntimeError(
            "TEAM_GAME_POPULATION_FAILURE:"
            f"{len(team_games)}:{expected_rows}"
        )

    # Reuse the validated PBP loader and event classifier.
    pbp, _schema = hist.load_pbp_season(
        season
    )

    completed_ids = set(
        games["game_id"].astype(str)
    )

    pbp = pbp[
        pbp["game_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    pbp_ids = set(
        pbp["game_id"]
        .dropna()
        .astype(str)
    )

    missing_pbp_games = sorted(
        completed_ids - pbp_ids
    )

    if missing_pbp_games:
        raise RuntimeError(
            "COMPLETED_GAMES_MISSING_PBP:"
            + ",".join(missing_pbp_games)
        )

    events = hist.classify_pbp_events(
        pbp
    )

    raw_dst = hist.aggregate_events(
        team_games,
        events,
    )

    if len(raw_dst) != expected_rows:
        raise RuntimeError(
            "RAW_DST_ROW_FAILURE:"
            f"{len(raw_dst)}:{expected_rows}"
        )

    # FanDuel PA requires the scoring-specific PBP surface.
    pa_pbp = scoring.load_pbp_season(
        season
    )

    pa_pbp = pa_pbp[
        pa_pbp["game_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    pa_ids = set(
        pa_pbp["game_id"]
        .dropna()
        .astype(str)
    )

    missing_pa_games = sorted(
        completed_ids - pa_ids
    )

    if missing_pa_games:
        raise RuntimeError(
            "COMPLETED_GAMES_MISSING_PA_PBP:"
            + ",".join(missing_pa_games)
        )

    scoring_plays = (
        scoring.classify_scoring_plays(
            pa_pbp
        )
    )

    with_pa, _pa_by_team = (
        scoring.aggregate_points_allowed(
            raw_dst,
            scoring_plays,
        )
    )

    scored = scoring.score_dst(
        with_pa
    )

    # --------------------------------------------------------------
    # Current-season validation
    # --------------------------------------------------------------

    if len(scored) != expected_rows:
        raise RuntimeError(
            "SCORED_ROW_FAILURE:"
            f"{len(scored)}:{expected_rows}"
        )

    duplicates = int(
        scored.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"DUPLICATE_GAME_TEAM:{duplicates}"
        )

    expected_game_ids = set(
        games["game_id"].astype(str)
    )

    actual_game_ids = set(
        scored["game_id"].astype(str)
    )

    if actual_game_ids != expected_game_ids:
        missing = sorted(
            expected_game_ids - actual_game_ids
        )
        extra = sorted(
            actual_game_ids - expected_game_ids
        )

        raise RuntimeError(
            "GAME_ID_COVERAGE_FAILURE:"
            f"missing={missing}:extra={extra}"
        )

    game_team_counts = (
        scored.groupby("game_id")["team"]
        .nunique()
    )

    if game_team_counts.ne(2).any():
        bad = (
            game_team_counts[
                game_team_counts.ne(2)
            ]
            .to_dict()
        )

        raise RuntimeError(
            f"GAME_TEAM_COVERAGE_FAILURE:{bad}"
        )

    if scored["team"].isna().any():
        raise RuntimeError("NULL_TEAM")

    if scored["opponent_team"].isna().any():
        raise RuntimeError(
            "NULL_OPPONENT_TEAM"
        )

    if scored["team"].eq(
        scored["opponent_team"]
    ).any():
        raise RuntimeError(
            "TEAM_EQUALS_OPPONENT"
        )

    required_numeric = [
        "team_score",
        "opponent_score",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
        "defensive_two_point_returns",
        "fanduel_points_allowed",
        "fanduel_dst_points",
    ]

    for c in required_numeric:
        x = pd.to_numeric(
            scored[c],
            errors="coerce",
        )

        if x.isna().any():
            raise RuntimeError(
                f"NULL_NUMERIC:{c}"
            )

        if not np.isfinite(
            x.to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                f"NONFINITE_NUMERIC:{c}"
            )

    # FanDuel PA cannot exceed scoreboard points.
    scoreboard = pd.to_numeric(
        scored["opponent_score"],
        errors="raise",
    )

    fd_pa = pd.to_numeric(
        scored["fanduel_points_allowed"],
        errors="raise",
    )

    if fd_pa.lt(0).any():
        raise RuntimeError(
            "NEGATIVE_FANDUEL_PA"
        )

    if fd_pa.gt(scoreboard).any():
        bad = scored.loc[
            fd_pa.gt(scoreboard),
            [
                "game_id",
                "team",
                "opponent_team",
                "opponent_score",
                "fanduel_points_allowed",
            ],
        ]

        raise RuntimeError(
            "FANDUEL_PA_EXCEEDS_SCOREBOARD:\n"
            + bad.to_string(index=False)
        )

    # Component reconstruction.
    component_cols = [
        "sack_points",
        "interception_points",
        "fumble_recovery_points",
        "safety_points",
        "blocked_kick_points",
        "return_td_points",
        "defensive_two_point_return_points",
        "points_allowed_tier_points",
    ]

    reconstructed = (
        scored[component_cols]
        .sum(axis=1)
        .round(2)
    )

    actual_fd = pd.to_numeric(
        scored["fanduel_dst_points"],
        errors="raise",
    ).round(2)

    if not reconstructed.eq(
        actual_fd
    ).all():
        raise RuntimeError(
            "FANDUEL_COMPONENT_RECONSTRUCTION_FAILURE"
        )

    if scored[
        "fanduel_scoring_verified"
    ].ne(1).any():
        raise RuntimeError(
            "UNVERIFIED_FANDUEL_SCORING"
        )

    scored["source"] = (
        "nflverse_pbp_current_season"
    )

    missing_output = [
        c
        for c in OUTPUT_COLUMNS
        if c not in scored.columns
    ]

    if missing_output:
        raise RuntimeError(
            "MISSING_OUTPUT_COLUMNS:"
            + ",".join(missing_output)
        )

    scored = (
        scored[OUTPUT_COLUMNS]
        .sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return scored


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=CURRENT_SEASON,
    )

    args = parser.parse_args()

    if args.season != CURRENT_SEASON:
        raise RuntimeError(
            "CURRENT_BRIDGE_SEASON_LOCK:"
            f"{args.season}"
        )

    scored = build_current_actuals(
        args.season
    )

    atomic_parquet(
        scored,
        OUTPUT,
    )

    print()
    print("ROWS =", len(scored))
    print(
        "GAMES =",
        scored["game_id"].nunique(),
    )
    print(
        "TEAMS =",
        scored["team"].nunique(),
    )
    print(
        "WEEKS =",
        sorted(
            scored["week"]
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    print(
        "DUP_GAME_TEAM =",
        int(
            scored.duplicated(
                ["game_id", "team"]
            ).sum()
        ),
    )

    print(
        "FD_PA_SCOREBOARD_DIFF_ROWS =",
        int(
            (
                pd.to_numeric(
                    scored["opponent_score"],
                    errors="raise",
                )
                !=
                pd.to_numeric(
                    scored[
                        "fanduel_points_allowed"
                    ],
                    errors="raise",
                )
            ).sum()
        ),
    )

    print("OUTPUT =", OUTPUT)
    print(
        "SHA256 =",
        sha256_file(OUTPUT),
    )

    print()
    print(
        "CURRENT_DST_POSTGAME_ACTUALS=PASS"
    )


if __name__ == "__main__":
    main()
