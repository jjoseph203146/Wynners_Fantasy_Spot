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

V1_PATH = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v2.csv"
)

WEEK_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_week_v2.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v2_audit.json"
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

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


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
    x = numeric_frame(train, FEATURES)

    y = pd.to_numeric(
        train[metric],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    mean = x.mean(axis=0)
    std = x.std(axis=0)

    std[std < 1e-9] = 1.0

    xs = (x - mean) / std
    y_mean = float(y.mean())
    yc = y - y_mean

    identity = np.eye(xs.shape[1])

    beta = np.linalg.solve(
        xs.T @ xs + alpha * identity,
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
    x = numeric_frame(frame, FEATURES)

    xs = (
        x - model["mean"]
    ) / model["std"]

    raw = (
        model["y_mean"]
        + xs @ model["beta"]
    )

    return np.clip(
        raw,
        max(0.0, model["lower"]),
        model["upper"],
    )


def mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                predicted - actual
            )
        )
    )


def main() -> None:
    for required in (DB, V1_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
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

    required_features = set(FEATURES) - set(
        environment.columns
    )

    if required_features:
        raise RuntimeError(
            "Missing features: "
            f"{sorted(required_features)}"
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

    data = data.sort_values(
        ["season", "week", "game_id", "team"]
    ).reset_index(drop=True)

    # Lock ridge strength using development seasons only:
    # train on 2023, validate on 2024.
    tuning_train = data[
        data["season"].eq(2023)
    ]

    tuning_validation = data[
        data["season"].eq(2024)
    ]

    if tuning_train.empty or tuning_validation.empty:
        raise RuntimeError(
            "Missing 2023/2024 tuning data"
        )

    selected_alphas = {}
    tuning_rows = []

    for metric in METRICS:
        validation_actual = pd.to_numeric(
            tuning_validation[metric],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)

        candidates = []

        for alpha in ALPHAS:
            model = fit_ridge(
                tuning_train,
                metric,
                alpha,
            )

            validation_prediction = predict(
                model,
                tuning_validation,
            )

            score = mae(
                validation_actual,
                validation_prediction,
            )

            candidates.append(
                (score, alpha)
            )

            tuning_rows.append({
                "metric": metric,
                "alpha": alpha,
                "validation_mae": score,
            })

        candidates.sort()
        selected_alphas[metric] = float(
            candidates[0][1]
        )

    control = pd.read_csv(V1_PATH)

    target_keys = control[
        [
            "game_id",
            "season",
            "week",
            "team",
        ]
    ].drop_duplicates()

    target = target_keys.merge(
        data,
        on=[
            "game_id",
            "season",
            "week",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    if target[FEATURES].isna().all(axis=1).any():
        raise RuntimeError(
            "Missing target environment rows"
        )

    prediction_rows = []
    leakage_rows = 0

    for row in target.itertuples(index=False):
        train = data[
            (data["season"] < row.season)
            | (
                data["season"].eq(row.season)
                & data["week"].lt(row.week)
            )
        ]

        if train[
            train["game_id"].astype(str).eq(
                str(row.game_id)
            )
        ].shape[0]:
            leakage_rows += 1

        target_frame = pd.DataFrame([
            {
                feature: getattr(row, feature)
                for feature in FEATURES
            }
        ])

        record = {
            "game_id": row.game_id,
            "season": int(row.season),
            "week": int(row.week),
            "team": row.team,
            "opponent_team": row.opponent_team,
            "training_rows": int(len(train)),
        }

        for metric in METRICS:
            model = fit_ridge(
                train,
                metric,
                selected_alphas[metric],
            )

            record[
                f"adaptive_{metric}"
            ] = float(
                predict(
                    model,
                    target_frame,
                )[0]
            )

            record[
                f"actual_{metric}"
            ] = float(
                getattr(row, metric)
            )

        prediction_rows.append(record)

    result = pd.DataFrame(prediction_rows)

    result["adaptive_completions"] = np.minimum(
        result["adaptive_completions"],
        result["adaptive_attempts"],
    )

    comparison_columns = [
        "game_id",
        "team",
        *[
            column
            for metric in METRICS
            for column in (
                f"baseline_{metric}",
                f"projected_{metric}",
            )
        ],
    ]

    result = result.merge(
        control[comparison_columns],
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    summary_rows = []

    for metric in METRICS:
        actual_values = result[
            f"actual_{metric}"
        ].to_numpy(dtype=float)

        baseline = result[
            f"baseline_{metric}"
        ].to_numpy(dtype=float)

        fixed = result[
            f"projected_{metric}"
        ].to_numpy(dtype=float)

        adaptive = result[
            f"adaptive_{metric}"
        ].to_numpy(dtype=float)

        baseline_mae = mae(
            actual_values,
            baseline,
        )

        fixed_mae = mae(
            actual_values,
            fixed,
        )

        adaptive_mae = mae(
            actual_values,
            adaptive,
        )

        summary_rows.append({
            "metric": metric,
            "baseline_mae": baseline_mae,
            "fixed_v1_mae": fixed_mae,
            "adaptive_v2_mae": adaptive_mae,
            "adaptive_vs_baseline":
                baseline_mae - adaptive_mae,
            "adaptive_vs_fixed":
                fixed_mae - adaptive_mae,
            "adaptive_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            adaptive
                            - actual_values
                        )
                    )
                )
            ),
            "adaptive_bias": float(
                np.mean(
                    adaptive
                    - actual_values
                )
            ),
            "selected_alpha":
                selected_alphas[metric],
        })

    summary = pd.DataFrame(summary_rows)

    week_rows = []

    for (
        season,
        week,
    ), block in result.groupby(
        ["season", "week"]
    ):
        for metric in METRICS:
            actual_values = block[
                f"actual_{metric}"
            ].to_numpy(dtype=float)

            for model_name, column in (
                (
                    "BASELINE_AVG5",
                    f"baseline_{metric}",
                ),
                (
                    "FIXED_V1",
                    f"projected_{metric}",
                ),
                (
                    "ADAPTIVE_V2",
                    f"adaptive_{metric}",
                ),
            ):
                predicted = block[
                    column
                ].to_numpy(dtype=float)

                week_rows.append({
                    "season": int(season),
                    "week": int(week),
                    "metric": metric,
                    "model": model_name,
                    "team_games": int(
                        len(block)
                    ),
                    "mae": mae(
                        actual_values,
                        predicted,
                    ),
                })

    week_report = pd.DataFrame(week_rows)

    adaptive_week = week_report[
        week_report["model"].eq(
            "ADAPTIVE_V2"
        )
    ].merge(
        week_report[
            week_report["model"].eq(
                "BASELINE_AVG5"
            )
        ],
        on=[
            "season",
            "week",
            "metric",
        ],
        suffixes=(
            "_adaptive",
            "_baseline",
        ),
    )

    adaptive_week["adaptive_win"] = (
        adaptive_week["mae_adaptive"]
        < adaptive_week["mae_baseline"]
    )

    duplicate_keys = int(
        result.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    prediction_columns = [
        f"adaptive_{metric}"
        for metric in METRICS
    ]

    nonfinite = int(
        (~np.isfinite(
            result[
                prediction_columns
            ].to_numpy(dtype=float)
        )).sum()
    )

    negative = int(
        (
            result[
                prediction_columns
            ] < 0
        ).sum().sum()
    )

    hard_failures = {
        "target_leakage_rows":
            leakage_rows,
        "duplicate_team_keys":
            duplicate_keys,
        "wrong_team_game_count":
            int(len(result) != 360),
        "nonfinite_predictions":
            nonfinite,
        "negative_predictions":
            negative,
        "completions_over_attempts":
            int(
                (
                    result[
                        "adaptive_completions"
                    ]
                    > result[
                        "adaptive_attempts"
                    ] + 1e-9
                ).sum()
            ),
    }

    status = (
        "PASS_HARD_CONTRACTS"
        if sum(
            hard_failures.values()
        ) == 0
        else "FAIL_HARD_CONTRACTS"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    week_report.to_csv(
        WEEK_OUTPUT,
        index=False,
    )

    tuning_output = OUTPUT.with_name(
        "offensive_reconciliation_"
        "walkforward_team_v2_tuning.csv"
    )

    pd.DataFrame(
        tuning_rows
    ).to_csv(
        tuning_output,
        index=False,
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_"
            "WALKFORWARD_TEAM_V2",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "tuning_contract":
            "TRAIN_2023_VALIDATE_2024",
        "evaluation_contract":
            "EXPANDING_STRICTLY_PRIOR",
        "games": int(
            result["game_id"].nunique()
        ),
        "team_games": int(len(result)),
        "features": FEATURES,
        "selected_alphas":
            selected_alphas,
        "hard_failures":
            hard_failures,
        "summary": summary.to_dict(
            orient="records"
        ),
        "outputs": {
            "team": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
            "week": {
                "path": str(
                    WEEK_OUTPUT.resolve()
                ),
                "sha256": sha256(
                    WEEK_OUTPUT
                ),
            },
            "tuning": {
                "path": str(
                    tuning_output.resolve()
                ),
                "sha256": sha256(
                    tuning_output
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
        "WFS OFFENSIVE RECONCILIATION "
        "WALK-FORWARD TEAM V2"
    )
    print("=" * 80)
    print(f"TEAM_GAMES={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(
        "TARGET_LEAKAGE_ROWS="
        f"{leakage_rows}"
    )

    print("\n=== LOCKED ALPHAS ===")
    for metric, alpha in selected_alphas.items():
        print(
            f"{metric.upper()}={alpha}"
        )

    print("\n=== TEAM MODEL COMPARISON ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n=== ADAPTIVE WEEKLY WINS VS BASELINE ===")
    print(
        adaptive_week.groupby(
            "metric"
        )["adaptive_win"].agg(
            ["sum", "count"]
        ).to_string()
    )

    print(f"\nTEAM_OUTPUT={OUTPUT}")
    print(f"WEEK_OUTPUT={WEEK_OUTPUT}")
    print(f"TUNING_OUTPUT={tuning_output}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(
        "WALKFORWARD_TEAM_V2_STATUS="
        f"{status}"
    )

    if status != "PASS_HARD_CONTRACTS":
        raise RuntimeError(
            "Adaptive team contract failed"
        )


if __name__ == "__main__":
    main()
