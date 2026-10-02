#!/usr/bin/env python3

"""
dst_production_projection.py

Train the promoted NFL FanDuel D/ST ridge model on all completed
2023-2025 historical data and score current 2026 D/ST feature rows.

Inputs
------
SQLite:
    dst_pregame_features
    current_dst_features

CSV:
    data/csv/dst_projection_benchmark_selected_features.csv

Model
-----
- Exact feature set promoted by dst_projection_benchmark.py
- Exact promoted ridge alpha from benchmark output
- Train rows: all 2023-2025 historical D/ST rows
- Train-only median imputation
- Train-only standardization
- NumPy ridge regression (no scipy/sklearn dependency)

Outputs
-------
SQLite:
    dst_production_projection

CSV:
    nfl_dst_production_projection.csv
    audit_dst_production_projection_summary.csv

Parquet:
    nfl_dst_production_projection.parquet

The historical frozen D/ST source tables are read-only.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


HISTORICAL_TABLE = "dst_pregame_features"
CURRENT_TABLE = "current_dst_features"
OUTPUT_TABLE = "dst_production_projection"

BENCHMARK_FEATURES_PATH = (
    Path(CSV_DIR)
    / "dst_projection_benchmark_selected_features.csv"
)

CSV_OUTPUT = (
    Path(CSV_DIR)
    / "nfl_dst_production_projection.csv"
)

PARQUET_OUTPUT = (
    Path(PARQUET_DIR)
    / "nfl_dst_production_projection.parquet"
)

AUDIT_OUTPUT = (
    Path(CSV_DIR)
    / "audit_dst_production_projection_summary.csv"
)

TARGET = "fanduel_dst_points"
TRAIN_SEASONS = [2023, 2024, 2025]
CURRENT_SEASON = 2026


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_benchmark_selection() -> tuple[list[str], float, str]:
    if not BENCHMARK_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Benchmark-selected feature file not found: "
            f"{BENCHMARK_FEATURES_PATH}"
        )

    df = pd.read_csv(BENCHMARK_FEATURES_PATH)

    required = {
        "feature",
        "research_rank",
        "selected_feature_count",
        "selected_alpha",
        "benchmark_decision",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Benchmark-selected feature file missing columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise RuntimeError(
            "Benchmark-selected feature file contains zero rows."
        )

    decisions = (
        df["benchmark_decision"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(decisions) != 1:
        raise RuntimeError(
            "Expected exactly one benchmark decision, found: "
            f"{decisions}"
        )

    decision = decisions[0]

    if decision != "PROMOTE_CANDIDATE":
        raise RuntimeError(
            "Benchmark model has not earned promotion. "
            f"Decision was: {decision}"
        )

    counts = (
        pd.to_numeric(
            df["selected_feature_count"],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    if len(counts) != 1:
        raise RuntimeError(
            "Expected one selected feature count, found: "
            f"{counts}"
        )

    selected_count = int(counts[0])

    alphas = (
        pd.to_numeric(
            df["selected_alpha"],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    if len(alphas) != 1:
        raise RuntimeError(
            "Expected one selected alpha, found: "
            f"{alphas}"
        )

    alpha = float(alphas[0])

    ordered = (
        df.assign(
            research_rank_num=pd.to_numeric(
                df["research_rank"],
                errors="coerce",
            )
        )
        .sort_values(
            ["research_rank_num", "feature"],
            ascending=[True, True],
        )
        ["feature"]
        .astype(str)
        .tolist()
    )

    if len(ordered) != selected_count:
        raise RuntimeError(
            "Selected feature file row count does not match "
            f"selected_feature_count: rows={len(ordered)}, "
            f"expected={selected_count}"
        )

    return ordered, alpha, decision


def load_historical(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(HISTORICAL_TABLE)}
        WHERE season IN (2023, 2024, 2025)
        ORDER BY season, week, game_id, team
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            f"{HISTORICAL_TABLE} returned zero rows."
        )

    return df


def load_current(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(CURRENT_TABLE)}
        WHERE season = 2026
        ORDER BY week, game_datetime, game_id, team
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            f"{CURRENT_TABLE} returned zero current rows."
        )

    return df


def prepare_train_current(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    missing_hist = sorted(
        set(features) - set(historical.columns)
    )
    if missing_hist:
        raise RuntimeError(
            "Historical source missing promoted features: "
            + ", ".join(missing_hist)
        )

    missing_current = sorted(
        set(features) - set(current.columns)
    )
    if missing_current:
        raise RuntimeError(
            "Current source missing promoted features: "
            + ", ".join(missing_current)
        )

    train_x = pd.DataFrame(
        {
            feature: pd.to_numeric(
                historical[feature],
                errors="coerce",
            )
            for feature in features
        }
    )

    current_x = pd.DataFrame(
        {
            feature: pd.to_numeric(
                current[feature],
                errors="coerce",
            )
            for feature in features
        }
    )

    train_y = pd.to_numeric(
        historical[TARGET],
        errors="coerce",
    )

    if train_y.isna().any():
        raise RuntimeError(
            "Historical target contains NULL rows."
        )

    medians = train_x.median(
        axis=0,
        skipna=True,
    )

    bad = medians[
        medians.isna()
    ].index.tolist()

    if bad:
        raise RuntimeError(
            "Training features entirely NULL: "
            + ", ".join(bad)
        )

    train_x = train_x.fillna(medians)
    current_x = current_x.fillna(medians)

    means = train_x.mean(axis=0)

    stds = train_x.std(
        axis=0,
        ddof=0,
    )

    stds = stds.mask(
        stds.eq(0),
        1.0,
    )

    train_z = (
        (train_x - means)
        / stds
    ).to_numpy(dtype=float)

    current_z = (
        (current_x - means)
        / stds
    ).to_numpy(dtype=float)

    return (
        train_z,
        train_y.to_numpy(dtype=float),
        current_z,
        medians,
        means,
        stds,
    )


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    if len(y) == 0:
        raise RuntimeError(
            "Cannot fit ridge on zero rows."
        )

    y_mean = float(
        np.mean(y)
    )
    y_centered = (
        y - y_mean
    )

    x_mean = np.mean(
        x,
        axis=0,
    )
    x_centered = (
        x - x_mean
    )

    xtx = (
        x_centered.T
        @ x_centered
    )

    penalty = (
        alpha
        * np.eye(
            xtx.shape[0],
            dtype=float,
        )
    )

    rhs = (
        x_centered.T
        @ y_centered
    )

    try:
        coef = np.linalg.solve(
            xtx + penalty,
            rhs,
        )
    except np.linalg.LinAlgError:
        coef = (
            np.linalg.pinv(
                xtx + penalty
            )
            @ rhs
        )

    intercept = (
        y_mean
        - float(
            x_mean @ coef
        )
    )

    return (
        float(intercept),
        coef,
    )


def build_projection_output(
    current: pd.DataFrame,
    predictions: np.ndarray,
    feature_count: int,
    alpha: float,
) -> pd.DataFrame:
    keep = [
        col
        for col in [
            "game_id",
            "season",
            "week",
            "game_type",
            "game_date",
            "gametime",
            "game_datetime",
            "team",
            "opponent_team",
            "is_home",
            "def_history_games",
            "opp_off_history_games",
        ]
        if col in current.columns
    ]

    out = current[
        keep
    ].copy()

    out[
        "dst_projection"
    ] = predictions.astype(float)

    # Keep a raw model score for auditability.
    out[
        "dst_projection_raw"
    ] = predictions.astype(float)

    # Production projections should not become negative fantasy values.
    # The raw value remains available for diagnostics.
    out[
        "dst_projection"
    ] = out[
        "dst_projection"
    ].clip(lower=0.0)

    out[
        "model_feature_count"
    ] = int(feature_count)

    out[
        "ridge_alpha"
    ] = float(alpha)

    out[
        "model_train_seasons"
    ] = "2023-2025"

    out[
        "model_status"
    ] = "PROMOTED"

    return out


def build_audit(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    output: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []

    def add(
        item,
        value,
        expected="",
        status="INFO",
    ):
        rows.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    add(
        "historical_rows",
        len(historical),
        1710,
        (
            "PASS"
            if len(historical) == 1710
            else "CHECK"
        ),
    )

    add(
        "current_feature_rows",
        len(current),
        544,
        (
            "PASS"
            if len(current) == 544
            else "CHECK"
        ),
    )

    add(
        "projection_rows",
        len(output),
        len(current),
        (
            "PASS"
            if len(output) == len(current)
            else "FAIL"
        ),
    )

    duplicate_rows = int(
        output.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    add(
        "duplicate_game_team_rows",
        duplicate_rows,
        0,
        (
            "PASS"
            if duplicate_rows == 0
            else "FAIL"
        ),
    )

    null_projection_rows = int(
        output[
            "dst_projection"
        ].isna().sum()
    )

    add(
        "null_projection_rows",
        null_projection_rows,
        0,
        (
            "PASS"
            if null_projection_rows == 0
            else "FAIL"
        ),
    )

    nonfinite_rows = int(
        (
            ~np.isfinite(
                pd.to_numeric(
                    output[
                        "dst_projection"
                    ],
                    errors="coerce",
                )
            )
        ).sum()
    )

    add(
        "nonfinite_projection_rows",
        nonfinite_rows,
        0,
        (
            "PASS"
            if nonfinite_rows == 0
            else "FAIL"
        ),
    )

    negative_production_rows = int(
        (
            pd.to_numeric(
                output[
                    "dst_projection"
                ],
                errors="coerce",
            )
            < 0
        ).sum()
    )

    add(
        "negative_production_projection_rows",
        negative_production_rows,
        0,
        (
            "PASS"
            if negative_production_rows == 0
            else "FAIL"
        ),
    )

    add(
        "selected_feature_count",
        len(features),
        20,
        (
            "PASS"
            if len(features) == 20
            else "CHECK"
        ),
    )

    add(
        "ridge_alpha",
        alpha,
        100.0,
        (
            "PASS"
            if np.isclose(alpha, 100.0)
            else "CHECK"
        ),
    )

    complete_current = int(
        current[
            features
        ].notna().all(
            axis=1
        ).sum()
    )

    add(
        "current_rows_complete_all_features",
        complete_current,
        len(current),
        (
            "PASS"
            if complete_current == len(current)
            else "CHECK"
        ),
    )

    week1_rows = int(
        output[
            "week"
        ].eq(1).sum()
    )

    add(
        "week1_projection_rows",
        week1_rows,
        32,
        (
            "PASS"
            if week1_rows == 32
            else "CHECK"
        ),
    )

    return pd.DataFrame(
        rows
    )


def write_sqlite(
    conn: sqlite3.Connection,
    output: pd.DataFrame,
) -> None:
    db_out = output.copy()

    if (
        "game_datetime"
        in db_out.columns
        and pd.api.types.is_datetime64_any_dtype(
            db_out["game_datetime"]
        )
    ):
        db_out[
            "game_datetime"
        ] = db_out[
            "game_datetime"
        ].astype(str)

    db_out.to_sql(
        OUTPUT_TABLE,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_game_team
        ON {OUTPUT_TABLE}(game_id, team)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_season_week
        ON {OUTPUT_TABLE}(season, week)
        """
    )

    conn.commit()


def main() -> None:
    section(
        "NFL D/ST PRODUCTION PROJECTION"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )
    print(
        f"Historical source: {HISTORICAL_TABLE}"
    )
    print(
        f"Current source: {CURRENT_TABLE}"
    )
    print(
        f"Benchmark selection: {BENCHMARK_FEATURES_PATH}"
    )
    print(
        "Production train seasons: 2023-2025"
    )
    print(
        "Current scoring season: 2026"
    )
    print(
        "Frozen historical tables: READ ONLY"
    )

    if not Path(
        DATABASE_PATH
    ).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    Path(
        CSV_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        PARQUET_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        features,
        alpha,
        decision,
    ) = load_benchmark_selection()

    section(
        "PROMOTED MODEL CONFIGURATION"
    )

    print(
        f"Benchmark decision: {decision}"
    )
    print(
        f"Feature count: {len(features)}"
    )
    print(
        f"Ridge alpha: {alpha}"
    )

    print()
    print(
        "Features:"
    )

    for i, feature in enumerate(
        features,
        start=1,
    ):
        print(
            f"  {i:2d}. {feature}"
        )

    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:
        historical = load_historical(
            conn
        )
        current = load_current(
            conn
        )

        section(
            "TRAINING / CURRENT INPUT AUDIT"
        )

        print(
            f"Historical train rows: {len(historical)}"
        )
        print(
            f"Current feature rows: {len(current)}"
        )

        (
            train_x,
            train_y,
            current_x,
            medians,
            means,
            stds,
        ) = prepare_train_current(
            historical,
            current,
            features,
        )

        intercept, coef = ridge_fit(
            train_x,
            train_y,
            alpha,
        )

        predictions = (
            intercept
            + current_x @ coef
        )

        output = build_projection_output(
            current,
            predictions,
            feature_count=len(features),
            alpha=alpha,
        )

        section(
            "PRODUCTION PROJECTION AUDIT"
        )

        summary = build_audit(
            historical,
            current,
            output,
            features,
            alpha,
        )

        print(
            summary.to_string(
                index=False
            )
        )

        failures = summary[
            summary[
                "status"
            ].eq("FAIL")
        ]

        if not failures.empty:
            print()
            print(
                "STRUCTURAL AUDIT: FAIL"
            )
            raise RuntimeError(
                "D/ST production projection audit failed. "
                "No output table written."
            )

        print()
        print(
            "STRUCTURAL AUDIT: PASS"
        )

        section(
            "WRITING D/ST PRODUCTION PROJECTIONS"
        )

        write_sqlite(
            conn,
            output,
        )

    output.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    output.to_parquet(
        PARQUET_OUTPUT,
        index=False,
    )

    summary.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    section(
        "2026 WEEK 1 D/ST PROJECTIONS"
    )

    week1 = (
        output[
            output["week"].eq(1)
        ]
        .sort_values(
            [
                "dst_projection",
                "team",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    display_cols = [
        col
        for col in [
            "team",
            "opponent_team",
            "is_home",
            "dst_projection",
            "dst_projection_raw",
        ]
        if col in week1.columns
    ]

    print(
        week1[
            display_cols
        ].to_string(
            index=False
        )
    )

    section(
        "MODEL COEFFICIENTS"
    )

    coef_df = pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": coef,
            "train_median": [
                medians[f]
                for f in features
            ],
            "train_mean": [
                means[f]
                for f in features
            ],
            "train_std": [
                stds[f]
                for f in features
            ],
        }
    )

    print(
        coef_df.sort_values(
            "standardized_coefficient",
            key=lambda s: s.abs(),
            ascending=False,
        ).to_string(
            index=False
        )
    )

    section(
        "EXPORTS"
    )

    print(
        f"CSV: {CSV_OUTPUT}"
    )
    print(
        f"Parquet: {PARQUET_OUTPUT}"
    )
    print(
        f"Audit: {AUDIT_OUTPUT}"
    )
    print(
        f"SQLite table: {OUTPUT_TABLE}"
    )

    section(
        "D/ST PRODUCTION PROJECTION COMPLETE"
    )

    print(
        f"Projected current rows: {len(output)}"
    )
    print(
        f"Week 1 rows: {len(week1)}"
    )
    print(
        "Next layer after audit review: merge D/ST projections "
        "into the FanDuel player pool using source-file salaries only."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 96)
        print(
            "D/ST PRODUCTION PROJECTION FAILED"
        )
        print("=" * 96)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)
