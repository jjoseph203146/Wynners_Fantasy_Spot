#!/usr/bin/env python3

"""
Current Offensive Model Matrix
==============================

Purpose
-------
Build the current-game offensive inference matrix for the frozen
Stage23E14 football-stat models.

This module does NOT:
- retrain models
- run model inference
- calculate FanDuel expectations
- modify historical pregame_features.py
- modify historical team_environment.py
- modify feature_matrix.py
- modify current_slate_features.py
- modify solver inputs
- write to SQLite

It reuses the validated current-player feature engine and joins it to
the validated current-team-environment artifact.

Production output
-----------------
data/parquet/nfl_current_offensive_model_matrix.parquet

Regression mode
---------------
--regression-week1

Regression mode rebuilds the historical Week 1 target population
using the same September 12 target cutoff used by the frozen Stage24J
R12 artifacts and compares the reconstructed matrix against:

data/model_candidates/stat_forecast/
stage24j_c_r7_r12_20260912T185444Z/
offensive_model_matrix_79_current_479.parquet

No production artifact is written in regression mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from database import get_connection

import current_slate_features as csf
import current_team_environment as cte
import pregame_features as pf


ROOT = Path(__file__).resolve().parent

E14_DIR = (
    ROOT
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage23e14_20260912T154412Z"
)

FEATURE_CONTRACT_PATH = (
    E14_DIR
    / "feature_contract.json"
)

CURRENT_ENV_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_team_environment.parquet"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_offensive_model_matrix.parquet"
)

FROZEN_R12_MATRIX = (
    ROOT
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage24j_c_r7_r12_20260912T185444Z"
    / "offensive_model_matrix_79_current_479.parquet"
)

FROZEN_R12_ENV = (
    ROOT
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage24j_c_r7_r12_20260912T185444Z"
    / "current_team_environment_28.parquet"
)

REGRESSION_SEASON = 2026
REGRESSION_WEEK = 1
REGRESSION_AS_OF_DATE = "2026-09-12"

IDENTITY_COLUMNS = [
    "game_id",
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
]


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_feature_contract() -> list[str]:
    section("LOADING E14 FEATURE CONTRACT")

    if not FEATURE_CONTRACT_PATH.is_file():
        raise RuntimeError(
            "Missing E14 feature contract: "
            f"{FEATURE_CONTRACT_PATH}"
        )

    with FEATURE_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        contract = json.load(handle)

    features = contract.get("features")

    if not isinstance(features, list):
        raise RuntimeError(
            "E14 feature contract does not contain "
            "a valid features list."
        )

    if len(features) != 79:
        raise RuntimeError(
            "E14 feature contract expected 79 features; "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise RuntimeError(
            "Duplicate features found in E14 contract."
        )

    if contract.get("feature_count") != 79:
        raise RuntimeError(
            "E14 feature_count is not 79."
        )

    if contract.get("football_only") is not True:
        raise RuntimeError(
            "E14 contract is not football_only=true."
        )

    if contract.get("fantasy_derived_features") is not False:
        raise RuntimeError(
            "E14 contract unexpectedly permits "
            "fantasy-derived features."
        )

    print("FEATURES =", len(features))
    print("FOOTBALL_ONLY = PASS")
    print("FANTASY_DERIVED_FEATURES = FALSE")

    return features


def load_player_authorities():
    section("LOADING CURRENT PLAYER AUTHORITIES")

    core_df = csf.load_core_features()

    with get_connection() as conn:
        games = csf.load_games(conn)

        (
            history,
            usage_table,
        ) = csf.load_player_history(
            conn,
            games,
        )

    mapping = csf.resolve_history_columns(
        history,
        usage_table,
    )

    history = csf.normalize_history(
        history,
        mapping,
    )

    print(
        "HISTORICAL PLAYER ROWS =",
        len(history),
    )

    return (
        core_df,
        games,
        history,
    )


def build_player_rows_for_targets(
    targets: pd.DataFrame,
    core_df: pd.DataFrame,
    games: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    section("BUILDING CURRENT PLAYER FEATURES")

    if targets.empty:
        raise RuntimeError(
            "No inference targets were resolved."
        )

    groups = csf.target_groups(
        targets
    )

    all_teams = sorted(
        set(
            targets[
                "away_team"
            ].dropna()
        )
        |
        set(
            targets[
                "home_team"
            ].dropna()
        )
    )

    with get_connection() as conn:
        depth = csf.load_current_depth(
            conn,
            all_teams,
        )

        depth = csf.attach_identity(
            conn,
            depth,
        )

    output_frames = []

    for (
        target_season,
        target_week,
        slate,
        teams,
    ) in groups:

        target_depth = depth[
            depth[
                "team"
            ].isin(
                teams
            )
        ].copy()

        target_depth = csf.attach_current_roster_position(
            target_depth,
            target_season,
            target_week,
        )

        (
            players,
            output,
            audit,
        ) = csf.build_current_rows(
            target_depth,
            slate,
            target_season,
            target_week,
            history,
            core_df,
        )

        if output.empty:
            raise RuntimeError(
                "Current player feature builder returned "
                f"zero rows for {target_season} "
                f"Week {target_week}."
            )

        problem_audit = audit[
            ~audit[
                "status"
            ].isin(
                [
                    "READY",
                    "COLD_START",
                ]
            )
        ]

        if not problem_audit.empty:
            raise RuntimeError(
                "Current player feature builder produced "
                f"{len(problem_audit)} invalid audit rows."
            )

        output_frames.append(
            output
        )

    if not output_frames:
        raise RuntimeError(
            "No current player feature frames were built."
        )

    output = pd.concat(
        output_frames,
        ignore_index=True,
    )

    print(
        "CURRENT PLAYER ROWS =",
        len(output),
    )

    return output


def load_production_targets(
    games: pd.DataFrame,
) -> pd.DataFrame:
    section("RESOLVING PRODUCTION TARGETS")

    with get_connection() as conn:
        targets = csf.load_inference_targets(
            conn,
            games,
        )

    if targets.empty:
        raise RuntimeError(
            "Production inference target set is empty."
        )

    return targets


def load_regression_targets(
    games: pd.DataFrame,
) -> pd.DataFrame:
    section("RESOLVING WEEK 1 REGRESSION TARGETS")

    schedule = cte.load_schedule()

    targets = cte.resolve_regression_target_games(
        schedule,
        season=REGRESSION_SEASON,
        week=REGRESSION_WEEK,
        as_of_date=REGRESSION_AS_OF_DATE,
    )

    if targets.empty:
        raise RuntimeError(
            "Week 1 regression target set is empty."
        )

    required = {
        "season",
        "week",
        "game_id",
        "game_date",
        "away_team",
        "home_team",
    }

    missing = sorted(
        required
        - set(targets.columns)
    )

    if missing:
        raise RuntimeError(
            "Regression targets missing columns: "
            + ", ".join(missing)
        )

    #
    # current_slate_features.build_game_map() requires
    # game_time. Recover it from the authoritative games
    # dataframe by exact game_id. Never synthesize kickoff
    # times.
    #
    if "game_time" not in games.columns:
        raise RuntimeError(
            "Authoritative games dataframe is missing "
            "game_time."
        )

    game_times = (
        games[
            [
                "game_id",
                "game_time",
            ]
        ]
        .copy()
    )

    duplicate_game_ids = (
        game_times[
            "game_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_game_ids.any():
        bad = (
            game_times.loc[
                duplicate_game_ids,
                [
                    "game_id",
                    "game_time",
                ],
            ]
            .sort_values(
                [
                    "game_id",
                    "game_time",
                ]
            )
        )

        print()
        print(
            "DUPLICATE AUTHORITATIVE GAME IDS:"
        )
        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Duplicate authoritative game_id values "
            "while resolving regression game_time."
        )

    targets = targets.merge(
        game_times,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    missing_game_time = (
        targets[
            "game_time"
        ]
        .isna()
        |
        targets[
            "game_time"
        ]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if missing_game_time.any():
        bad = targets.loc[
            missing_game_time,
            [
                "game_id",
                "game_date",
                "away_team",
                "home_team",
            ],
        ]

        print()
        print(
            "REGRESSION TARGETS MISSING "
            "AUTHORITATIVE GAME TIME:"
        )
        print(
            bad.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Unable to attach authoritative game_time "
            "to every Week 1 regression target."
        )

    print(
        "REGRESSION TARGET GAMES =",
        targets["game_id"].nunique(),
    )

    print(
        "REGRESSION GAME_TIME COVERAGE =",
        int(
            targets[
                "game_time"
            ].notna().sum()
        ),
        "/",
        len(targets),
    )

    return targets

def load_environment(
    regression: bool,
) -> pd.DataFrame:
    section("LOADING CURRENT TEAM ENVIRONMENT")

    path = (
        FROZEN_R12_ENV
        if regression
        else CURRENT_ENV_PATH
    )

    if not path.is_file():
        raise RuntimeError(
            "Missing team environment artifact: "
            f"{path}"
        )

    env = pd.read_parquet(
        path
    )

    required = {
        "game_id",
        "team",
        "opponent_team",
        "history_games",
    }

    missing = sorted(
        required
        - set(env.columns)
    )

    if missing:
        raise RuntimeError(
            "Team environment missing columns: "
            + ", ".join(missing)
        )

    duplicate_count = int(
        env.duplicated(
            [
                "game_id",
                "team",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate team environment identity rows: "
            f"{duplicate_count}"
        )

    print(
        "ENVIRONMENT ROWS =",
        len(env),
    )

    print(
        "ENVIRONMENT GAMES =",
        env["game_id"].nunique(),
    )

    return env


def load_current_injury_consensus() -> pd.DataFrame:
    """Load current injury authority keyed only by exact GSIS id."""
    with get_connection() as conn:
        injury = pd.read_sql_query(
            """
            SELECT
                gsis_id,
                report_status,
                consensus_status,
                injury_gate,
                injury_gate_reason
            FROM injury_consensus_current
            """,
            conn,
        )

    if injury.empty:
        return pd.DataFrame(
            columns=[
                "gsis_id",
                "report_status",
                "consensus_status",
                "injury_gate",
                "injury_gate_reason",
            ]
        )

    injury["gsis_id"] = injury["gsis_id"].fillna("").astype(str).str.strip()
    injury = injury[injury["gsis_id"] != ""].copy()

    dup = injury["gsis_id"].duplicated(keep=False)
    if dup.any():
        bad = injury.loc[dup].sort_values("gsis_id")
        print("DUPLICATE CURRENT INJURY GSIS IDS:")
        print(bad.head(100).to_string(index=False))
        raise RuntimeError("Duplicate GSIS ids in injury_consensus_current.")

    return injury


def derive_current_status_flags(player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Map current injury authority to the historical E14 active/injury contract.

    Absence from injury_consensus_current is healthy-by-absence.  Identity is
    exact GSIS only; no player-name matching is permitted.
    """
    df = player_features.copy()

    if "player_id" not in df.columns:
        raise RuntimeError("Current player rows missing player_id.")

    df["player_id"] = df["player_id"].fillna("").astype(str).str.strip()
    if df["player_id"].eq("").any():
        raise RuntimeError("Blank current player_id before injury attachment.")

    injury = load_current_injury_consensus().rename(columns={"gsis_id": "player_id"})
    df = df.merge(
        injury,
        on="player_id",
        how="left",
        validate="many_to_one",
        indicator="_injury_merge",
    )

    df["active_flag"] = 1
    df["injury_flag"] = 0

    report = df["report_status"].fillna("").astype(str).str.strip().str.upper()
    consensus = df["consensus_status"].fillna("").astype(str).str.strip().str.upper()
    gate = df["injury_gate"].fillna("").astype(str).str.strip().str.upper()

    injured = (
        report.isin(["QUESTIONABLE", "DOUBTFUL", "OUT", "NOTE"])
        | consensus.isin(["QUESTIONABLE", "DOUBTFUL", "OUT", "NOTE"])
        | gate.eq("BLOCK")
    )
    unavailable = (
        report.isin(["DOUBTFUL", "OUT"])
        | consensus.isin(["DOUBTFUL", "OUT"])
        | gate.eq("BLOCK")
    )

    df.loc[injured, "injury_flag"] = 1
    df.loc[unavailable, "active_flag"] = 0

    print("CURRENT INJURY EXACT-GSIS MATCHES =", int((df["_injury_merge"] == "both").sum()))
    print("CURRENT INJURY HEALTHY-BY-ABSENCE =", int((df["_injury_merge"] == "left_only").sum()))
    print("CURRENT ACTIVE_FLAG=0 =", int((df["active_flag"] == 0).sum()))
    print("CURRENT INJURY_FLAG=1 =", int((df["injury_flag"] == 1).sum()))

    return df.drop(columns=["_injury_merge"])


def build_full_player_pregame_features(
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the complete historical E14 player feature surface for targets."""
    section("BUILDING FULL E14 PLAYER PREGAME FEATURES")

    current = derive_current_status_flags(player_features)

    required_current = {
        "season", "week", "game_id", "player_id", "player_display_name",
        "position", "team", "opponent_team",
    }
    missing = sorted(required_current - set(current.columns))
    if missing:
        raise RuntimeError("Current target rows missing columns: " + ", ".join(missing))

    # Historical player_weekly_usage is the validated source used by
    # pregame_features.py.  Remove every target game before appending blank
    # target rows so completed games cannot leak into a historical regression.
    usage = pf.load_usage()
    usage = pf.normalize_numeric_columns(usage)

    target_game_ids = set(current["game_id"].astype(str))
    usage = usage[~usage["game_id"].astype(str).isin(target_game_ids)].copy()

    # pregame_features.calculate_pregame_features uses player_id as its
    # grouping key. Current player_id is the exact resolved GSIS authority.
    usage_columns = list(usage.columns)
    target_rows = []

    for row in current.itertuples(index=False):
        record = {column: np.nan for column in usage_columns}
        values = {
            "game_id": row.game_id,
            "season": row.season,
            "week": row.week,
            "season_type": "REG",
            "identity_key": getattr(row, "identity_key", np.nan),
            "player_id": row.player_id,
            "player_name": getattr(row, "player_name", row.player_display_name),
            "player_display_name": row.player_display_name,
            "position": row.position,
            "position_group": row.position,
            "team": row.team,
            "opponent_team": row.opponent_team,
            "active_flag": int(row.active_flag),
            "injury_flag": int(row.injury_flag),
        }
        for key, value in values.items():
            if key in record:
                record[key] = value
        # Target-game realized football values are intentionally zero. They are
        # never consumed for the target because every rolling feature is lagged.
        for column in usage_columns:
            if column in {
                "fanduel_points", "offense_snaps", "offense_pct", "carries",
                "targets", "receptions", "target_share", "air_yards_share",
                "wopr", "touches", "opportunities", "yards_from_scrimmage",
                "fanduel_per_snap", "fanduel_per_touch", "yards_per_opportunity",
                "role_expansion_flag", "role_decline_flag", "high_usage_flag",
                "starter_usage_flag",
            }:
                record[column] = 0.0
        target_rows.append(record)

    target_usage = pd.DataFrame(target_rows, columns=usage_columns)
    combined = pd.concat([usage, target_usage], ignore_index=True)
    combined = pf.normalize_numeric_columns(combined)
    pregame = pf.calculate_pregame_features(combined)

    target_keys = current[["game_id", "player_id", "team"]].astype(str)
    target_key_set = set(map(tuple, target_keys.to_numpy().tolist()))
    mask = [
        (str(g), str(p), str(t)) in target_key_set
        for g, p, t in pregame[["game_id", "player_id", "team"]].itertuples(index=False, name=None)
    ]
    target_pregame = pregame.loc[mask].copy()

    dup = target_pregame.duplicated(["game_id", "player_id", "team"], keep=False)
    if dup.any():
        raise RuntimeError("Duplicate target rows after E14 pregame reconstruction.")

    current_keys = set(map(tuple, target_keys.to_numpy().tolist()))
    built_keys = set(map(tuple, target_pregame[["game_id", "player_id", "team"]].astype(str).to_numpy().tolist()))
    missing_keys = current_keys - built_keys
    extra_keys = built_keys - current_keys
    print("E14 PLAYER TARGET ROWS =", len(target_pregame))
    print("E14 PLAYER TARGET MISSING =", len(missing_keys))
    print("E14 PLAYER TARGET EXTRA =", len(extra_keys))
    if missing_keys or extra_keys:
        raise RuntimeError("E14 player pregame target identity mismatch.")

    # Preserve current authoritative identity/display metadata while importing
    # only the historical player-feature columns from pregame_features.py.
    meta = current.copy()
    drop_from_pregame = [
        c for c in [
            "player_name", "player_display_name", "position", "opponent_team",
            "season", "week", "identity_key",
        ] if c in target_pregame.columns
    ]
    target_pregame = target_pregame.drop(columns=drop_from_pregame)

    merged = meta.merge(
        target_pregame,
        on=["game_id", "player_id", "team"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_pregame"),
    )

    if "history_games" not in merged.columns:
        raise RuntimeError("Reconstructed pregame rows missing history_games.")
    merged = merged.rename(columns={"history_games": "history_games_player"})

    return merged


def prepare_player_features(
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    df = build_full_player_pregame_features(player_features)

    if "player_name" not in df.columns:
        if "player_display_name" not in df.columns:
            raise RuntimeError(
                "Current player features contain neither player_name nor player_display_name."
            )
        df["player_name"] = df["player_display_name"]

    if "player_display_name" not in df.columns:
        df["player_display_name"] = df["player_name"]

    return df

def join_environment(
    player_features: pd.DataFrame,
    env: pd.DataFrame,
) -> pd.DataFrame:
    section("JOINING TEAM ENVIRONMENT")

    player_features = (
        prepare_player_features(
            player_features
        )
    )

    env_join = env.copy()

    # season/week belong to both authorities. Require exact equality later,
    # then keep the player target values so the E14 contract retains plain
    # season/week column names.

    env_join = env_join.rename(
        columns={
            "history_games":
                "history_games_team",
        }
    )

    joined = player_features.merge(
        env_join,
        on=[
            "game_id",
            "team",
        ],
        how="left",
        suffixes=(
            "_player",
            "_env",
        ),
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int(
        (
            joined["_merge"]
            != "both"
        ).sum()
    )

    print(
        "PLAYER ROWS =",
        len(player_features),
    )

    print(
        "ENV MATCHED =",
        len(joined) - unmatched,
    )

    print(
        "ENV UNMATCHED =",
        unmatched,
    )

    if unmatched:
        bad = joined.loc[
            joined["_merge"] != "both",
            [
                "game_id",
                "team",
                "player_id",
                "player_display_name",
            ],
        ]

        print()
        print(
            bad.head(100)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Current player rows failed exact "
            "team-environment attachment."
        )

    for dimension in ["season", "week"]:
        player_col = f"{dimension}_player"
        env_col = f"{dimension}_env"
        if player_col in joined.columns and env_col in joined.columns:
            mismatch = (
                pd.to_numeric(joined[player_col], errors="coerce")
                != pd.to_numeric(joined[env_col], errors="coerce")
            )
            if mismatch.any():
                raise RuntimeError(
                    f"Player/environment {dimension} mismatch: {int(mismatch.sum())}"
                )
            joined[dimension] = joined[player_col]
            joined = joined.drop(columns=[player_col, env_col])

    player_opp = (
        "opponent_team_player"
        if "opponent_team_player" in joined.columns
        else "opponent_team"
    )

    env_opp = (
        "opponent_team_env"
        if "opponent_team_env" in joined.columns
        else None
    )

    if env_opp is not None:
        mismatch = (
            joined[player_opp]
            .astype(str)
            !=
            joined[env_opp]
            .astype(str)
        )

        mismatch_count = int(
            mismatch.sum()
        )

        print(
            "OPPONENT MISMATCH =",
            mismatch_count,
        )

        if mismatch_count:
            raise RuntimeError(
                "Player/environment opponent identity "
                "mismatch."
            )

        joined[
            "opponent_team"
        ] = joined[
            player_opp
        ]

    joined = joined.drop(
        columns=[
            "_merge",
        ],
        errors="ignore",
    )

    return joined


def build_matrix(
    joined: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    section("BUILDING E14 MATRIX")

    required = (
        IDENTITY_COLUMNS
        + features
    )

    missing = [
        column
        for column in required
        if column not in joined.columns
    ]

    if missing:
        print(
            "MISSING REQUIRED COLUMNS:"
        )

        for column in missing:
            print(
                " ",
                column,
            )

        raise RuntimeError(
            "Current E14 matrix source is missing "
            f"{len(missing)} required columns."
        )

    matrix = joined[
        required
    ].copy()

    expected_columns = (
        IDENTITY_COLUMNS
        + features
    )

    if (
        list(matrix.columns)
        != expected_columns
    ):
        raise RuntimeError(
            "E14 matrix column order mismatch."
        )

    duplicate_count = int(
        matrix.duplicated(
            [
                "game_id",
                "player_id",
                "team",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate E14 player identities: "
            f"{duplicate_count}"
        )

    blank_identity = pd.Series(
        False,
        index=matrix.index,
    )

    for column in [
        "game_id",
        "player_id",
        "position",
        "team",
        "opponent_team",
    ]:
        blank_identity |= (
            matrix[column]
            .isna()
            |
            matrix[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

    identity_failures = int(
        blank_identity.sum()
    )

    if identity_failures:
        raise RuntimeError(
            "Blank E14 identity rows: "
            f"{identity_failures}"
        )

    for feature in features:
        matrix[feature] = pd.to_numeric(
            matrix[feature],
            errors="coerce",
        )

    feature_frame = matrix[
        features
    ]

    null_cells = int(
        feature_frame
        .isna()
        .sum()
        .sum()
    )

    nonfinite_cells = int(
        (
            ~np.isfinite(
                feature_frame
                .to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    print(
        "ROWS =",
        len(matrix),
    )

    print(
        "GAMES =",
        matrix["game_id"].nunique(),
    )

    print(
        "TEAMS =",
        matrix["team"].nunique(),
    )

    print(
        "COLUMNS =",
        len(matrix.columns),
    )

    print(
        "FEATURES =",
        len(features),
    )

    print(
        "IDENTITY_FAILURES =",
        identity_failures,
    )

    print(
        "DUPLICATES =",
        duplicate_count,
    )

    print(
        "NULL_FEATURE_CELLS =",
        null_cells,
    )

    print(
        "NONFINITE_FEATURE_CELLS =",
        nonfinite_cells,
    )

    print(
        "COLUMN_ORDER_MATCH =",
        list(matrix.columns)
        == expected_columns,
    )

    if null_cells:
        raise RuntimeError(
            "Null cells found in E14 model features."
        )

    if nonfinite_cells:
        raise RuntimeError(
            "Nonfinite cells found in E14 model features."
        )

    return matrix


def compare_frozen(
    matrix: pd.DataFrame,
    features: list[str],
) -> None:
    section("WEEK 1 FROZEN R12 REGRESSION")

    if not FROZEN_R12_MATRIX.is_file():
        raise RuntimeError(f"Missing frozen R12 matrix: {FROZEN_R12_MATRIX}")

    frozen = pd.read_parquet(FROZEN_R12_MATRIX)

    print("CURRENT ROWS =", len(matrix))
    print("FROZEN ROWS =", len(frozen))
    print("CURRENT GAMES =", matrix["game_id"].nunique())
    print("FROZEN GAMES =", frozen["game_id"].nunique())
    print("CURRENT TEAMS =", matrix["team"].nunique())
    print("FROZEN TEAMS =", frozen["team"].nunique())

    if matrix["game_id"].nunique() != 14:
        raise RuntimeError("Week 1 regression expected 14 games.")
    if matrix["team"].nunique() != 28:
        raise RuntimeError("Week 1 regression expected 28 teams.")
    if list(matrix.columns) != list(frozen.columns):
        raise RuntimeError("Frozen R12 column contract mismatch.")

    keys = ["game_id", "player_id", "team"]
    current_keys = set(map(tuple, matrix[keys].astype(str).to_numpy().tolist()))
    frozen_keys = set(map(tuple, frozen[keys].astype(str).to_numpy().tolist()))
    missing_from_current = frozen_keys - current_keys
    extra_in_current = current_keys - frozen_keys

    print("IDENTITY_MISSING_FROM_CURRENT =", len(missing_from_current))
    print("IDENTITY_EXTRA_IN_CURRENT =", len(extra_in_current))

    if missing_from_current:
        print("\nMISSING IDENTITIES (population drift diagnostic):")
        for row in sorted(missing_from_current)[:100]:
            print(row)
    if extra_in_current:
        print("\nEXTRA IDENTITIES (population drift diagnostic):")
        for row in sorted(extra_in_current)[:100]:
            print(row)

    common_keys = current_keys & frozen_keys
    if not common_keys:
        raise RuntimeError("Week 1 R12 regression has zero common identities.")

    current_common = matrix[
        matrix[keys].astype(str).apply(tuple, axis=1).isin(common_keys)
    ].sort_values(keys).reset_index(drop=True)
    frozen_common = frozen[
        frozen[keys].astype(str).apply(tuple, axis=1).isin(common_keys)
    ].sort_values(keys).reset_index(drop=True)

    if len(current_common) != len(frozen_common):
        raise RuntimeError("Common-identity regression alignment failed.")

    identity_mismatch = 0
    for column in IDENTITY_COLUMNS:
        mismatch = (
            current_common[column].fillna("").astype(str)
            != frozen_common[column].fillna("").astype(str)
        )
        identity_mismatch += int(mismatch.sum())

    print("COMMON_IDENTITIES =", len(current_common))
    print("IDENTITY_VALUE_MISMATCH_CELLS =", identity_mismatch)
    if identity_mismatch:
        raise RuntimeError("Week 1 common identity/meta regression failed.")

    drift_rows = []
    max_abs_delta = 0.0
    for feature in features:
        a = pd.to_numeric(current_common[feature], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(frozen_common[feature], errors="coerce").to_numpy(dtype=float)
        delta = np.abs(a - b)
        feature_max = float(np.nanmax(delta)) if len(delta) else 0.0
        mismatch_count = int((delta > 1e-10).sum())
        max_abs_delta = max(max_abs_delta, feature_max)
        if mismatch_count:
            drift_rows.append((feature, mismatch_count, feature_max))

    print("NUMERIC_DRIFT_COLUMNS =", len(drift_rows))
    print("MAX_ABS_NUMERIC_DELTA =", max_abs_delta)
    if drift_rows:
        print("\nNUMERIC DRIFT (diagnostic; historical data may have advanced):")
        for feature, count, maximum in drift_rows:
            print(f"{feature:45s} rows={count:4d} max_abs_delta={maximum:.12g}")

    print("\nPOPULATION_DRIFT_DIAGNOSTIC=", "YES" if (missing_from_current or extra_in_current) else "NO")
    print("FROZEN_COMMON_IDENTITY_REGRESSION=PASS")

def publish(
    matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    section("PUBLISHING CURRENT E14 MATRIX")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    matrix.to_parquet(
        temp_path,
        index=False,
    )

    check = pd.read_parquet(
        temp_path
    )

    if (
        list(check.columns)
        != list(matrix.columns)
    ):
        temp_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Published matrix column verification failed."
        )

    if len(check) != len(matrix):
        temp_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Published matrix row verification failed."
        )

    temp_path.replace(
        output_path
    )

    print(
        "PUBLISHED =",
        output_path,
    )

    print(
        "ROWS =",
        len(matrix),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--regression-week1",
        action="store_true",
        help=(
            "Rebuild the frozen September 12 Week 1 "
            "target population and compare structurally "
            "against Stage24J R12. No production output "
            "is written."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    features = load_feature_contract()

    (
        core_df,
        games,
        history,
    ) = load_player_authorities()

    if args.regression_week1:
        targets = load_regression_targets(
            games
        )
    else:
        targets = load_production_targets(
            games
        )

    player_features = (
        build_player_rows_for_targets(
            targets,
            core_df,
            games,
            history,
        )
    )

    env = load_environment(
        regression=args.regression_week1
    )

    joined = join_environment(
        player_features,
        env,
    )

    matrix = build_matrix(
        joined,
        features,
    )

    if args.regression_week1:
        compare_frozen(
            matrix,
            features,
        )

        print()
        print(
            "REGRESSION MODE: "
            "NO PRODUCTION ARTIFACT WRITTEN"
        )

        return

    publish(
        matrix,
        Path(
            args.output
        ).resolve(),
    )

    print()
    print(
        "CURRENT_OFFENSIVE_MODEL_MATRIX=PASS"
    )


if __name__ == "__main__":
    main()
