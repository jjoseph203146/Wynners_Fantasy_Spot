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

MATRIX_PATH = (
    ROOT
    / "data/parquet"
    / "nfl_current_offensive_model_matrix.parquet"
)

CONTROL_PATH = (
    ROOT
    / "processed"
    / "offensive_reconciliation_team_history_v1.csv"
)

CURRENT_CONTROL_PATH = (
    ROOT
    / "processed"
    / "offensive_team_reconciliation_team_audit_v3.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_current_adaptive_team_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_current_adaptive_team_v1_audit.json"
)

METRICS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
]

FEATURES = [
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

SELECTED_ALPHAS = {
    "attempts": 100.0,
    "completions": 1000.0,
    "passing_yards": 1000.0,
    "passing_tds": 0.1,
    "carries": 100.0,
    "rushing_yards": 100.0,
    "rushing_tds": 1000.0,
}

from wfs_schedule_context import resolve_schedule_week_context

SCHEDULE_CONTEXT = resolve_schedule_week_context()
TARGET_SEASON = int(SCHEDULE_CONTEXT.season)
TARGET_WEEK = int(SCHEDULE_CONTEXT.planning_week)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()

def numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    clean = frame[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return clean.fillna(0.0).to_numpy(
        dtype=float
    )

def fit_ridge(
    train: pd.DataFrame,
    metric: str,
    alpha: float,
) -> dict[str, object]:
    x = numeric_frame(
        train,
        FEATURES,
    )

    y = pd.to_numeric(
        train[metric],
        errors="coerce",
    ).fillna(0.0).to_numpy(
        dtype=float
    )

    mean = x.mean(axis=0)
    std = x.std(axis=0)

    std[std < 1e-9] = 1.0

    xs = (x - mean) / std

    y_mean = float(y.mean())
    yc = y - y_mean

    identity = np.eye(
        xs.shape[1]
    )

    beta = np.linalg.solve(
        xs.T @ xs
        + alpha * identity,
        xs.T @ yc,
    )

    return {
        "mean": mean,
        "std": std,
        "y_mean": y_mean,
        "beta": beta,
        "lower": float(
            np.quantile(y, 0.01)
        ),
        "upper": float(
            np.quantile(y, 0.99)
        ),
    }

def predict(
    model: dict[str, object],
    frame: pd.DataFrame,
) -> np.ndarray:
    x = numeric_frame(
        frame,
        FEATURES,
    )

    xs = (
        x - model["mean"]
    ) / model["std"]

    raw = (
        model["y_mean"]
        + xs @ model["beta"]
    )

    return np.clip(
        raw,
        max(
            0.0,
            model["lower"],
        ),
        model["upper"],
    )

def main() -> None:
    for required in (
        DB,
        MATRIX_PATH,
        CONTROL_PATH,
        CURRENT_CONTROL_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                required
            )

    if OUTPUT.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {OUTPUT}"
        )

    uri = (
        f"file:{DB.resolve()}?mode=ro"
    )

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:
        environment = pd.read_sql_query(
            """
            SELECT *
            FROM team_pregame_environment
            """,
            conn,
        )

        actual = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                team,
                attempts,
                completions,
                passing_yards,
                passing_tds,
                carries,
                rushing_yards,
                rushing_tds
            FROM team_game_stats
            """,
            conn,
        )

    required_environment = (
        set(FEATURES)
        - set(environment.columns)
    )

    if required_environment:
        raise RuntimeError(
            "Missing historical features: "
            f"{sorted(required_environment)}"
        )

    data = environment.merge(
        actual,
        on=[
            "game_id",
            "season",
            "week",
            "team",
        ],
        how="inner",
        validate="one_to_one",
    )

    for column in (
        "season",
        "week",
        *FEATURES,
        *METRICS,
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    train = data.loc[
        (data["season"] < TARGET_SEASON)
        | (
            data["season"].eq(
                TARGET_SEASON
            )
            & data["week"].lt(
                TARGET_WEEK
            )
        )
    ].copy()

    leakage_rows = int(
        (
            train["season"].gt(
                TARGET_SEASON
            )
            | (
                train["season"].eq(
                    TARGET_SEASON
                )
                & train["week"].ge(
                    TARGET_WEEK
                )
            )
        ).sum()
    )

    current = pd.read_parquet(
        MATRIX_PATH
    )

    key_columns = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    ]

    current_features = (
        current[
            key_columns
            + [
                "history_games_team",
                *[
                    feature
                    for feature in FEATURES
                    if feature
                    != "history_games"
                ],
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "history_games_team":
                    "history_games",
            }
        )
        .reset_index(drop=True)
    )

    current_features = current_features.loc[
        current_features[
            "season"
        ].eq(TARGET_SEASON)
        & current_features[
            "week"
        ].eq(TARGET_WEEK)
    ].copy()

    duplicate_keys = int(
        current_features.duplicated(
            [
                "game_id",
                "team",
            ]
        ).sum()
    )

    missing_feature_rows = int(
        current_features[
            FEATURES
        ].isna().all(axis=1).sum()
    )

    records = current_features[
        key_columns
    ].copy()

    model_audit = {}

    for metric in METRICS:
        model = fit_ridge(
            train,
            metric,
            SELECTED_ALPHAS[metric],
        )

        records[
            f"adaptive_{metric}"
        ] = predict(
            model,
            current_features,
        )

        model_audit[metric] = {
            "alpha": float(
                SELECTED_ALPHAS[
                    metric
                ]
            ),
            "training_rows": int(
                len(train)
            ),
            "training_lower": float(
                model["lower"]
            ),
            "training_upper": float(
                model["upper"]
            ),
        }

    records[
        "adaptive_completions"
    ] = np.minimum(
        records[
            "adaptive_completions"
        ],
        records[
            "adaptive_attempts"
        ],
    )

    current_control = pd.read_csv(
        CURRENT_CONTROL_PATH
    )

    control_columns = [
        "game_id",
        "team",
        "team_pass_attempts",
        "team_completions",
        "team_passing_yards",
        "team_passing_tds",
        "team_carries",
        "team_rushing_yards",
        "team_rushing_tds",
    ]

    missing_control = [
        column
        for column in control_columns
        if column
        not in current_control.columns
    ]

    if missing_control:
        raise RuntimeError(
            "Missing control columns: "
            f"{missing_control}"
        )

    result = records.merge(
        current_control[
            control_columns
        ],
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    control_map = {
        "attempts":
            "team_pass_attempts",
        "completions":
            "team_completions",
        "passing_yards":
            "team_passing_yards",
        "passing_tds":
            "team_passing_tds",
        "carries":
            "team_carries",
        "rushing_yards":
            "team_rushing_yards",
        "rushing_tds":
            "team_rushing_tds",
    }

    for metric, control_column in (
        control_map.items()
    ):
        result[
            f"control_{metric}"
        ] = pd.to_numeric(
            result[control_column],
            errors="coerce",
        )

        result[
            f"adaptive_minus_control_{metric}"
        ] = (
            result[
                f"adaptive_{metric}"
            ]
            - result[
                f"control_{metric}"
            ]
        )

    projection_columns = [
        f"adaptive_{metric}"
        for metric in METRICS
    ]

    hard_failures = {
        "target_team_games_not_expected": int(
            len(result) != len(current_features)
        ),
        "duplicate_target_keys":
            duplicate_keys,
        "missing_feature_rows":
            missing_feature_rows,
        "future_training_rows":
            leakage_rows,
        "missing_control_rows": int(
            result[
                [
                    f"control_{metric}"
                    for metric in METRICS
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        ),
        "nonfinite_predictions": int(
            (
                ~np.isfinite(
                    result[
                        projection_columns
                    ].to_numpy(
                        dtype=float
                    )
                )
            ).sum()
        ),
        "negative_predictions": int(
            result[
                projection_columns
            ]
            .lt(0)
            .sum()
            .sum()
        ),
        "completion_over_attempt": int(
            (
                result[
                    "adaptive_completions"
                ]
                > result[
                    "adaptive_attempts"
                ]
                + 1e-9
            ).sum()
        ),
    }

    status = (
        "PASS_HARD_CONTRACTS"
        if not any(
            hard_failures.values()
        )
        else "FAIL_HARD_CONTRACTS"
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    audit = {
        "version": (
            "WFS_CURRENT_ADAPTIVE_TEAM_V1"
        ),
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "target_season": TARGET_SEASON,
        "target_week": TARGET_WEEK,
        "rows": int(len(result)),
        "games": int(
            result[
                "game_id"
            ].nunique()
        ),
        "training_rows": int(
            len(train)
        ),
        "training_max_season": int(
            train["season"].max()
        ),
        "training_max_week_2026": int(
            train.loc[
                train["season"].eq(
                    TARGET_SEASON
                ),
                "week",
            ].max()
        ),
        "model_contract": model_audit,
        "hard_failures":
            hard_failures,
        "source_hashes": {
            str(MATRIX_PATH):
                sha256(MATRIX_PATH),
            str(CONTROL_PATH):
                sha256(CONTROL_PATH),
            str(CURRENT_CONTROL_PATH):
                sha256(
                    CURRENT_CONTROL_PATH
                ),
        },
        "sqlite_modified": False,
        "production_modified": False,
        "status": status,
    }

    AUDIT_OUTPUT.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "WFS CURRENT ADAPTIVE TEAM "
        "ENVIRONMENT V1"
    )
    print("=" * 80)

    print("=== SUMMARY ===")
    print(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
    )

    comparison_columns = [
        "game_id",
        "team",
        *[
            column
            for metric in METRICS
            for column in (
                f"control_{metric}",
                f"adaptive_{metric}",
                (
                    "adaptive_minus_control_"
                    f"{metric}"
                ),
            )
        ],
    ]

    print("=== LARGEST ENVIRONMENT CHANGES ===")

    result[
        "environment_change_score"
    ] = sum(
        result[
            f"adaptive_minus_control_{metric}"
        ].abs()
        / max(
            1.0,
            float(
                result[
                    f"control_{metric}"
                ]
                .abs()
                .median()
            ),
        )
        for metric in METRICS
    )

    print(
        result.sort_values(
            "environment_change_score",
            ascending=False,
        )[
            comparison_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    print("OUTPUT=", OUTPUT)
    print(
        "AUDIT_OUTPUT=",
        AUDIT_OUTPUT,
    )
    print(
        f"CURRENT_ADAPTIVE_TEAM_STATUS="
        f"{status}"
    )
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")

    if status.startswith("FAIL"):
        raise RuntimeError(
            "Current adaptive team "
            "contract failed"
        )

if __name__ == "__main__":
    main()
