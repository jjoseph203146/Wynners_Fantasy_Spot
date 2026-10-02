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

INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_inputs_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v1_audit.json"
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

WEIGHTS = {
    "last": 0.45,
    "avg_3": 0.25,
    "avg_5": 0.10,
    "opponent": 0.15,
    "league": 0.05,
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


def values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(0.0)


def recent(
    frame: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    series = values(frame, column)

    if series.empty:
        return 0.0, 0.0, 0.0

    return (
        float(series.iloc[-1]),
        float(series.tail(3).mean()),
        float(series.tail(5).mean()),
    )


def weighted(
    last: float,
    avg_3: float,
    avg_5: float,
    opponent: float,
    league: float,
) -> float:
    return float(
        last * WEIGHTS["last"]
        + avg_3 * WEIGHTS["avg_3"]
        + avg_5 * WEIGHTS["avg_5"]
        + opponent * WEIGHTS["opponent"]
        + league * WEIGHTS["league"]
    )


def bounded(
    value: float,
    prior: pd.Series,
) -> float:
    clean = pd.to_numeric(
        prior,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return max(0.0, float(value))

    lower = float(clean.quantile(0.01))
    upper = float(clean.quantile(0.99))

    return float(
        np.clip(
            value,
            lower,
            upper,
        )
    )


def error_summary(
    frame: pd.DataFrame,
    prediction_prefix: str,
) -> dict[str, dict[str, float]]:
    result = {}

    for metric in METRICS:
        actual = values(
            frame,
            f"actual_{metric}",
        )

        predicted = values(
            frame,
            f"{prediction_prefix}_{metric}",
        )

        error = predicted - actual

        result[metric] = {
            "mae": float(error.abs().mean()),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(error)
                    )
                )
            ),
            "bias": float(error.mean()),
        }

    return result


def main() -> None:
    for required in (DB, INPUT):
        if not required.is_file():
            raise FileNotFoundError(required)

    target = pd.read_csv(INPUT)

    target_team_games = (
        target[
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent_team",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "week",
                "game_id",
                "team",
            ]
        )
        .reset_index(drop=True)
    )

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        history = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                team,
                opponent_team,
                attempts,
                completions,
                passing_yards,
                passing_tds,
                carries,
                rushing_yards,
                rushing_tds
            FROM team_game_stats
            ORDER BY season, week, game_id, team
            """,
            conn,
        )

    for column in (
        "season",
        "week",
        *METRICS,
    ):
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    rows = []
    future_leakage_rows = 0

    for target_row in target_team_games.itertuples(
        index=False
    ):
        strict_prior = history[
            (history["season"] < target_row.season)
            | (
                history["season"].eq(
                    target_row.season
                )
                & history["week"].lt(
                    target_row.week
                )
            )
        ].copy()

        if strict_prior[
            strict_prior["game_id"].astype(str).eq(
                str(target_row.game_id)
            )
        ].shape[0]:
            future_leakage_rows += 1

        offense = strict_prior[
            strict_prior["team"].astype(str).eq(
                str(target_row.team)
            )
        ].sort_values(
            ["season", "week", "game_id"]
        )

        defense_allowed = strict_prior[
            strict_prior[
                "opponent_team"
            ].astype(str).eq(
                str(target_row.opponent_team)
            )
        ].sort_values(
            ["season", "week", "game_id"]
        )

        actual_rows = history[
            history["game_id"].astype(str).eq(
                str(target_row.game_id)
            )
            & history["team"].astype(str).eq(
                str(target_row.team)
            )
        ]

        if len(actual_rows) != 1:
            raise RuntimeError(
                "Missing or duplicate actual team row: "
                f"{target_row.game_id} "
                f"{target_row.team}"
            )

        actual = actual_rows.iloc[0]

        record = {
            "game_id": target_row.game_id,
            "season": int(target_row.season),
            "week": int(target_row.week),
            "team": target_row.team,
            "opponent_team":
                target_row.opponent_team,
            "team_prior_games": int(
                len(offense)
            ),
            "opponent_allowed_prior_games": int(
                len(defense_allowed)
            ),
        }

        for metric in METRICS:
            last, avg_3, avg_5 = recent(
                offense,
                metric,
            )

            allowed_3 = (
                float(
                    values(
                        defense_allowed,
                        metric,
                    ).tail(3).mean()
                )
                if not defense_allowed.empty
                else 0.0
            )

            league_median = float(
                values(
                    strict_prior,
                    metric,
                ).median()
            )

            raw_projection = weighted(
                last,
                avg_3,
                avg_5,
                allowed_3,
                league_median,
            )

            projection = bounded(
                raw_projection,
                strict_prior[metric],
            )

            record[f"team_last_{metric}"] = last
            record[f"team_avg_3_{metric}"] = avg_3
            record[f"team_avg_5_{metric}"] = avg_5
            record[
                f"opponent_allowed_avg_3_{metric}"
            ] = allowed_3
            record[f"league_median_{metric}"] = (
                league_median
            )
            record[f"raw_{metric}"] = (
                raw_projection
            )
            record[f"projected_{metric}"] = (
                projection
            )
            record[f"actual_{metric}"] = float(
                actual[metric]
            )

        # Keep passing outputs internally feasible.
        record["projected_completions"] = min(
            record["projected_attempts"],
            record["projected_completions"],
        )

        # Honest simple baseline: five-game team average.
        for metric in METRICS:
            record[f"baseline_{metric}"] = (
                record[f"team_avg_5_{metric}"]
            )

        rows.append(record)

    result = pd.DataFrame(rows)

    duplicate_keys = int(
        result.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    nonfinite = int(
        (~np.isfinite(
            result[
                [
                    f"projected_{metric}"
                    for metric in METRICS
                ]
            ].to_numpy(dtype=float)
        )).sum()
    )

    negative = int(
        (
            result[
                [
                    f"projected_{metric}"
                    for metric in METRICS
                ]
            ] < 0
        ).sum().sum()
    )

    projected_metrics = error_summary(
        result,
        "projected",
    )

    baseline_metrics = error_summary(
        result,
        "baseline",
    )

    comparison_rows = []

    for metric in METRICS:
        projected_mae = projected_metrics[
            metric
        ]["mae"]

        baseline_mae = baseline_metrics[
            metric
        ]["mae"]

        comparison_rows.append({
            "metric": metric,
            "baseline_mae": baseline_mae,
            "projected_mae": projected_mae,
            "mae_improvement":
                baseline_mae - projected_mae,
            "projected_rmse":
                projected_metrics[
                    metric
                ]["rmse"],
            "projected_bias":
                projected_metrics[
                    metric
                ]["bias"],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    week_rows = []

    for (
        season,
        week,
    ), block in result.groupby(
        ["season", "week"]
    ):
        for metric in METRICS:
            actual = values(
                block,
                f"actual_{metric}",
            )

            projected = values(
                block,
                f"projected_{metric}",
            )

            baseline = values(
                block,
                f"baseline_{metric}",
            )

            week_rows.append({
                "season": int(season),
                "week": int(week),
                "metric": metric,
                "team_games": int(
                    len(block)
                ),
                "projected_mae": float(
                    (projected - actual)
                    .abs()
                    .mean()
                ),
                "baseline_mae": float(
                    (baseline - actual)
                    .abs()
                    .mean()
                ),
            })

    week_report = pd.DataFrame(week_rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    week_output = OUTPUT.with_name(
        "offensive_reconciliation_"
        "walkforward_team_week_v1.csv"
    )

    week_report.to_csv(
        week_output,
        index=False,
    )

    hard_failures = {
        "duplicate_team_keys":
            duplicate_keys,
        "future_target_rows_in_prior":
            future_leakage_rows,
        "nonfinite_projections":
            nonfinite,
        "negative_projections":
            negative,
        "wrong_team_game_count":
            int(len(result) != 360),
        "completions_over_attempts":
            int(
                (
                    result[
                        "projected_completions"
                    ]
                    > result[
                        "projected_attempts"
                    ] + 1e-9
                ).sum()
            ),
    }

    status = (
        "PASS"
        if sum(
            hard_failures.values()
        ) == 0
        else "FAIL"
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_"
            "WALKFORWARD_TEAM_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "games": int(
            result["game_id"].nunique()
        ),
        "team_games": int(len(result)),
        "weights": WEIGHTS,
        "hard_failures": hard_failures,
        "baseline_metrics":
            baseline_metrics,
        "projected_metrics":
            projected_metrics,
        "outputs": {
            "team": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
            "week": {
                "path": str(
                    week_output.resolve()
                ),
                "sha256": sha256(
                    week_output
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
        "WALK-FORWARD TEAM V1"
    )
    print("=" * 80)
    print(f"TEAM_GAMES={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(
        "FUTURE_TARGET_ROWS_IN_PRIOR="
        f"{future_leakage_rows}"
    )
    print(
        "DUPLICATE_TEAM_KEYS="
        f"{duplicate_keys}"
    )
    print(
        "NONFINITE_PROJECTIONS="
        f"{nonfinite}"
    )
    print(
        "NEGATIVE_PROJECTIONS="
        f"{negative}"
    )

    print("\n=== TEAM BUDGET ERROR ===")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n=== WEEKLY WIN COUNTS ===")

    weekly_pivot = week_report.copy()
    weekly_pivot["projected_win"] = (
        weekly_pivot["projected_mae"]
        < weekly_pivot["baseline_mae"]
    )

    print(
        weekly_pivot.groupby(
            "metric"
        )["projected_win"].agg(
            ["sum", "count"]
        ).to_string()
    )

    print(f"\nTEAM_OUTPUT={OUTPUT}")
    print(f"WEEK_OUTPUT={week_output}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(
        "WALKFORWARD_TEAM_STATUS="
        f"{status}"
    )

    if status != "PASS":
        raise RuntimeError(
            "Walk-forward team contract failed"
        )


if __name__ == "__main__":
    main()
