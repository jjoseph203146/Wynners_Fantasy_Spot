#!/usr/bin/env python3

"""
dst_projection_benchmark.py

Chronological benchmark for NFL FanDuel D/ST projections.

Source
------
SQLite:
    dst_pregame_features

Research shortlist
------------------
CSV:
    data/csv/dst_feature_shortlist.csv

Target
------
    fanduel_dst_points

Benchmark design
----------------
Primary production decision:
    Train: 2023-2024
    Test : 2025

Models:
    1. Train-mean baseline
    2. Recent D/ST baseline (def_fd_avg_5, fallback def_fd_avg_3,
       fallback train mean)
    3. Ridge regression using a deterministic compact feature set chosen
       from the already-completed chronological feature research.

Feature selection:
    - Reads only rows with shortlist == 1.
    - Ranks by the existing research_rank.
    - Evaluates compact top-N sets: 5, 8, 10, 12, 15, 20.
    - Feature selection is therefore based on the prior research layer,
      not on 2025 target outcomes inside this script.
    - Missing values are median-imputed using TRAIN data only.
    - Standardization uses TRAIN mean/std only.
    - Ridge coefficients are solved with NumPy; no sklearn/scipy required.
    - The intercept is not penalized because predictors and target are
      centered before solving.

Outputs
-------
CSV:
    audit_dst_projection_benchmark_summary.csv
    dst_projection_benchmark_results.csv
    dst_projection_benchmark_predictions.csv
    dst_projection_benchmark_selected_features.csv

This is a benchmark/research layer only.
It does not modify frozen D/ST source tables and does not create the
current-season production projection.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR


SOURCE_TABLE = "dst_pregame_features"
TARGET = "fanduel_dst_points"

SHORTLIST_PATH = Path(CSV_DIR) / "dst_feature_shortlist.csv"

SUMMARY_OUTPUT = (
    Path(CSV_DIR) / "audit_dst_projection_benchmark_summary.csv"
)
RESULTS_OUTPUT = (
    Path(CSV_DIR) / "dst_projection_benchmark_results.csv"
)
PREDICTIONS_OUTPUT = (
    Path(CSV_DIR) / "dst_projection_benchmark_predictions.csv"
)
FEATURES_OUTPUT = (
    Path(CSV_DIR) / "dst_projection_benchmark_selected_features.csv"
)

TRAIN_SEASONS = [2023, 2024]
TEST_SEASON = 2025

FEATURE_COUNTS = [5, 8, 10, 12, 15, 20]

# Fixed ridge grid. Hyperparameter selection is performed only through
# expanding chronological validation inside the training period:
# train 2023 -> validate 2024.
RIDGE_ALPHAS = [
    0.0,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

RECENT_BASELINE_FEATURES = [
    "def_fd_avg_5",
    "def_fd_avg_3",
]


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_source(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(SOURCE_TABLE)}
        WHERE season IN (2023, 2024, 2025)
        ORDER BY season, week, game_id, team
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(f"{SOURCE_TABLE} returned zero rows.")

    required = {
        "game_id",
        "season",
        "week",
        "team",
        TARGET,
        "def_fd_avg_5",
        "def_fd_avg_3",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Missing required source columns: " + ", ".join(missing)
        )

    duplicates = int(
        df.duplicated(["game_id", "team"]).sum()
    )
    if duplicates:
        raise RuntimeError(
            f"{SOURCE_TABLE} contains {duplicates} duplicate game/team rows."
        )

    return df


def load_shortlist() -> pd.DataFrame:
    if not SHORTLIST_PATH.exists():
        raise FileNotFoundError(
            f"Research shortlist not found: {SHORTLIST_PATH}"
        )

    df = pd.read_csv(SHORTLIST_PATH)

    required = {
        "feature",
        "research_rank",
        "shortlist",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Shortlist missing required columns: "
            + ", ".join(missing)
        )

    df["shortlist"] = pd.to_numeric(
        df["shortlist"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["research_rank"] = pd.to_numeric(
        df["research_rank"],
        errors="coerce",
    )

    selected = (
        df[df["shortlist"].eq(1)]
        .dropna(subset=["feature", "research_rank"])
        .sort_values(
            ["research_rank", "feature"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    if selected.empty:
        raise RuntimeError(
            "Research shortlist contains zero selected features."
        )

    return selected


def prepare_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_x = pd.DataFrame(
        {
            f: pd.to_numeric(train[f], errors="coerce")
            for f in features
        }
    )
    test_x = pd.DataFrame(
        {
            f: pd.to_numeric(test[f], errors="coerce")
            for f in features
        }
    )

    train_y = pd.to_numeric(
        train[TARGET],
        errors="coerce",
    ).to_numpy(dtype=float)

    test_y = pd.to_numeric(
        test[TARGET],
        errors="coerce",
    ).to_numpy(dtype=float)

    medians = train_x.median(axis=0, skipna=True)

    # If a training feature is entirely null, it cannot be used safely.
    bad = medians[medians.isna()].index.tolist()
    if bad:
        raise RuntimeError(
            "Training features entirely NULL: " + ", ".join(bad)
        )

    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0, ddof=0)

    # Zero-variance features are standardized with denominator 1.0, which
    # leaves them at zero after centering and prevents numerical failure.
    stds = stds.mask(stds.eq(0), 1.0)

    train_z = (
        (train_x - means) / stds
    ).to_numpy(dtype=float)

    test_z = (
        (test_x - means) / stds
    ).to_numpy(dtype=float)

    meta = {
        "medians": medians,
        "means": means,
        "stds": stds,
    }

    return train_z, train_y, test_z, test_y, meta


def ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    if len(y) == 0:
        raise RuntimeError("Cannot fit ridge on zero rows.")

    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    # x is already standardized, but recenter defensively.
    x_mean = np.mean(x, axis=0)
    x_centered = x - x_mean

    xtx = x_centered.T @ x_centered
    penalty = alpha * np.eye(xtx.shape[0], dtype=float)

    rhs = x_centered.T @ y_centered

    try:
        coef = np.linalg.solve(
            xtx + penalty,
            rhs,
        )
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(
            xtx + penalty
        ) @ rhs

    intercept = (
        y_mean - float(x_mean @ coef)
    )

    return intercept, coef


def ridge_predict(
    x: np.ndarray,
    intercept: float,
    coef: np.ndarray,
) -> np.ndarray:
    return intercept + x @ coef


def metrics(
    actual: np.ndarray,
    pred: np.ndarray,
) -> dict:
    mask = (
        np.isfinite(actual)
        & np.isfinite(pred)
    )

    y = actual[mask]
    p = pred[mask]

    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
        }

    errors = p - y

    mae = float(
        np.mean(np.abs(errors))
    )
    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    corr = np.nan
    if (
        len(y) >= 3
        and np.std(y) > 0
        and np.std(p) > 0
    ):
        corr = float(
            np.corrcoef(y, p)[0, 1]
        )

    return {
        "n": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
    }


def recent_baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    train_mean = float(
        pd.to_numeric(
            train[TARGET],
            errors="coerce",
        ).mean()
    )

    pred = pd.Series(
        np.nan,
        index=test.index,
        dtype=float,
    )

    for feature in RECENT_BASELINE_FEATURES:
        values = pd.to_numeric(
            test[feature],
            errors="coerce",
        )
        pred = pred.fillna(values)

    pred = pred.fillna(train_mean)

    return pred.to_numpy(dtype=float)


def choose_alpha_and_count(
    train_pool: pd.DataFrame,
    ranked_features: list[str],
) -> tuple[int, float, pd.DataFrame]:
    """
    Tune feature count and ridge alpha only within the training period.

    Inner chronological validation:
        train 2023 -> validate 2024
    """

    inner_train = train_pool[
        train_pool["season"].eq(2023)
    ].copy()

    inner_valid = train_pool[
        train_pool["season"].eq(2024)
    ].copy()

    rows = []

    valid_counts = [
        n for n in FEATURE_COUNTS
        if n <= len(ranked_features)
    ]

    if not valid_counts:
        valid_counts = [len(ranked_features)]

    for count in valid_counts:
        features = ranked_features[:count]

        (
            train_x,
            train_y,
            valid_x,
            valid_y,
            _,
        ) = prepare_matrix(
            inner_train,
            inner_valid,
            features,
        )

        for alpha in RIDGE_ALPHAS:
            intercept, coef = ridge_fit(
                train_x,
                train_y,
                alpha,
            )

            pred = ridge_predict(
                valid_x,
                intercept,
                coef,
            )

            m = metrics(
                valid_y,
                pred,
            )

            rows.append(
                {
                    "feature_count": count,
                    "alpha": alpha,
                    "validation_n": m["n"],
                    "validation_mae": m["mae"],
                    "validation_rmse": m["rmse"],
                    "validation_corr": m["corr"],
                }
            )

    validation = pd.DataFrame(rows)

    validation = validation.sort_values(
        [
            "validation_rmse",
            "validation_mae",
            "validation_corr",
            "feature_count",
            "alpha",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    if validation.empty:
        raise RuntimeError(
            "No inner validation benchmark rows were created."
        )

    winner = validation.iloc[0]

    return (
        int(winner["feature_count"]),
        float(winner["alpha"]),
        validation,
    )


def build_predictions_frame(
    test: pd.DataFrame,
    actual: np.ndarray,
    mean_pred: np.ndarray,
    recent_pred: np.ndarray,
    ridge_pred: np.ndarray,
) -> pd.DataFrame:
    keep = [
        c
        for c in [
            "game_id",
            "season",
            "week",
            "game_date",
            "team",
            "opponent_team",
            "is_home",
        ]
        if c in test.columns
    ]

    out = test[keep].copy()

    out["actual_fanduel_dst_points"] = actual
    out["train_mean_prediction"] = mean_pred
    out["recent_dst_prediction"] = recent_pred
    out["ridge_prediction"] = ridge_pred

    out["ridge_error"] = (
        out["ridge_prediction"]
        - out["actual_fanduel_dst_points"]
    )

    out["ridge_abs_error"] = (
        out["ridge_error"].abs()
    )

    return out


def main() -> None:
    section("NFL D/ST CHRONOLOGICAL PROJECTION BENCHMARK")

    print(f"Database: {DATABASE_PATH}")
    print(f"Source: {SOURCE_TABLE}")
    print(f"Research shortlist: {SHORTLIST_PATH}")
    print(f"Target: {TARGET}")
    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Primary held-out test season: {TEST_SEASON}")
    print("Inner tuning fold: 2023 -> 2024")
    print("Frozen D/ST source tables: NOT modified.")
    print("Current-season production model: NOT created here.")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    Path(CSV_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as conn:
        df = load_source(conn)

    shortlist = load_shortlist()

    ranked_features = shortlist[
        "feature"
    ].astype(str).tolist()

    missing_features = sorted(
        set(ranked_features)
        - set(df.columns)
    )

    if missing_features:
        raise RuntimeError(
            "Shortlisted features missing from source table: "
            + ", ".join(missing_features)
        )

    train = df[
        df["season"].isin(TRAIN_SEASONS)
    ].copy()

    test = df[
        df["season"].eq(TEST_SEASON)
    ].copy()

    section("SOURCE AUDIT")

    source_audit = [
        {
            "item": "source_rows",
            "value": len(df),
            "expected": 1710,
            "status": (
                "PASS"
                if len(df) == 1710
                else "CHECK"
            ),
        },
        {
            "item": "train_rows",
            "value": len(train),
            "expected": 1140,
            "status": (
                "PASS"
                if len(train) == 1140
                else "CHECK"
            ),
        },
        {
            "item": "test_rows",
            "value": len(test),
            "expected": 570,
            "status": (
                "PASS"
                if len(test) == 570
                else "CHECK"
            ),
        },
        {
            "item": "duplicate_game_team_rows",
            "value": int(
                df.duplicated(
                    ["game_id", "team"]
                ).sum()
            ),
            "expected": 0,
            "status": (
                "PASS"
                if int(
                    df.duplicated(
                        ["game_id", "team"]
                    ).sum()
                ) == 0
                else "FAIL"
            ),
        },
        {
            "item": "null_target_rows",
            "value": int(
                df[TARGET].isna().sum()
            ),
            "expected": 0,
            "status": (
                "PASS"
                if int(
                    df[TARGET].isna().sum()
                ) == 0
                else "FAIL"
            ),
        },
        {
            "item": "research_shortlist_rows",
            "value": len(shortlist),
            "expected": "",
            "status": "INFO",
        },
    ]

    source_audit_df = pd.DataFrame(
        source_audit
    )

    print(
        source_audit_df.to_string(
            index=False
        )
    )

    failures = source_audit_df[
        source_audit_df["status"].eq("FAIL")
    ]

    if not failures.empty:
        raise RuntimeError(
            "D/ST projection benchmark source audit failed."
        )

    section("INNER CHRONOLOGICAL MODEL SELECTION")

    (
        selected_count,
        selected_alpha,
        validation,
    ) = choose_alpha_and_count(
        train,
        ranked_features,
    )

    selected_features = (
        ranked_features[:selected_count]
    )

    print(
        validation.head(15).to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected feature count: {selected_count}"
    )
    print(
        f"Selected ridge alpha: {selected_alpha}"
    )

    print()
    print("Selected features:")
    for i, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(f"  {i:2d}. {feature}")

    section("PRIMARY HELD-OUT 2025 BENCHMARK")

    (
        train_x,
        train_y,
        test_x,
        test_y,
        _,
    ) = prepare_matrix(
        train,
        test,
        selected_features,
    )

    intercept, coef = ridge_fit(
        train_x,
        train_y,
        selected_alpha,
    )

    ridge_pred = ridge_predict(
        test_x,
        intercept,
        coef,
    )

    train_mean_value = float(
        np.mean(train_y)
    )

    mean_pred = np.full(
        len(test_y),
        train_mean_value,
        dtype=float,
    )

    recent_pred = recent_baseline_predictions(
        train,
        test,
    )

    mean_metrics = metrics(
        test_y,
        mean_pred,
    )

    recent_metrics = metrics(
        test_y,
        recent_pred,
    )

    ridge_metrics = metrics(
        test_y,
        ridge_pred,
    )

    result_rows = []

    for name, m in [
        ("train_mean_baseline", mean_metrics),
        ("recent_dst_baseline", recent_metrics),
        ("ridge_compact", ridge_metrics),
    ]:
        result_rows.append(
            {
                "model": name,
                "train_seasons": "2023-2024",
                "test_season": 2025,
                "feature_count": (
                    selected_count
                    if name == "ridge_compact"
                    else 0
                ),
                "ridge_alpha": (
                    selected_alpha
                    if name == "ridge_compact"
                    else np.nan
                ),
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "corr": m["corr"],
            }
        )

    results = pd.DataFrame(
        result_rows
    )

    results["mae_vs_recent"] = (
        recent_metrics["mae"]
        - results["mae"]
    )

    results["rmse_vs_recent"] = (
        recent_metrics["rmse"]
        - results["rmse"]
    )

    results["corr_vs_recent"] = (
        results["corr"]
        - recent_metrics["corr"]
    )

    print(
        results.to_string(
            index=False
        )
    )

    section("BENCHMARK DECISION")

    ridge_beats_recent_mae = (
        ridge_metrics["mae"]
        < recent_metrics["mae"]
    )

    ridge_beats_recent_rmse = (
        ridge_metrics["rmse"]
        < recent_metrics["rmse"]
    )

    ridge_beats_recent_corr = (
        pd.notna(ridge_metrics["corr"])
        and (
            pd.isna(recent_metrics["corr"])
            or ridge_metrics["corr"]
            > recent_metrics["corr"]
        )
    )

    wins = sum(
        [
            ridge_beats_recent_mae,
            ridge_beats_recent_rmse,
            ridge_beats_recent_corr,
        ]
    )

    if (
        ridge_beats_recent_rmse
        and ridge_beats_recent_corr
    ):
        decision = "PROMOTE_CANDIDATE"
        rationale = (
            "Compact ridge beats recent-DST baseline "
            "on held-out RMSE and correlation."
        )
    elif wins >= 2:
        decision = "REVIEW"
        rationale = (
            "Compact ridge wins at least two held-out "
            "metrics but does not satisfy the preferred "
            "RMSE + correlation promotion gate."
        )
    else:
        decision = "DO_NOT_PROMOTE"
        rationale = (
            "Compact ridge does not demonstrate enough "
            "held-out improvement over recent D/ST form."
        )

    print(
        f"Ridge beats recent MAE:  "
        f"{ridge_beats_recent_mae}"
    )
    print(
        f"Ridge beats recent RMSE: "
        f"{ridge_beats_recent_rmse}"
    )
    print(
        f"Ridge beats recent Corr: "
        f"{ridge_beats_recent_corr}"
    )
    print()
    print(f"Decision: {decision}")
    print(f"Reason: {rationale}")

    predictions = build_predictions_frame(
        test,
        test_y,
        mean_pred,
        recent_pred,
        ridge_pred,
    )

    feature_rows = []

    coef_map = dict(
        zip(
            selected_features,
            coef,
        )
    )

    shortlist_rank = dict(
        zip(
            shortlist["feature"].astype(str),
            shortlist["research_rank"],
        )
    )

    for feature in selected_features:
        feature_rows.append(
            {
                "feature": feature,
                "research_rank": int(
                    shortlist_rank[feature]
                ),
                "standardized_ridge_coefficient": float(
                    coef_map[feature]
                ),
                "selected_feature_count": selected_count,
                "selected_alpha": selected_alpha,
                "benchmark_decision": decision,
            }
        )

    selected_feature_df = pd.DataFrame(
        feature_rows
    )

    summary_rows = list(source_audit)

    summary_rows.extend(
        [
            {
                "item": "selected_feature_count",
                "value": selected_count,
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "selected_ridge_alpha",
                "value": selected_alpha,
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "recent_baseline_mae",
                "value": recent_metrics["mae"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "recent_baseline_rmse",
                "value": recent_metrics["rmse"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "recent_baseline_corr",
                "value": recent_metrics["corr"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "ridge_mae",
                "value": ridge_metrics["mae"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "ridge_rmse",
                "value": ridge_metrics["rmse"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "ridge_corr",
                "value": ridge_metrics["corr"],
                "expected": "",
                "status": "INFO",
            },
            {
                "item": "benchmark_decision",
                "value": decision,
                "expected": "",
                "status": "INFO",
            },
        ]
    )

    summary = pd.DataFrame(
        summary_rows
    )

    results.to_csv(
        RESULTS_OUTPUT,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    selected_feature_df.to_csv(
        FEATURES_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    section("EXPORTS")

    print(f"Results: {RESULTS_OUTPUT}")
    print(f"Predictions: {PREDICTIONS_OUTPUT}")
    print(f"Selected features: {FEATURES_OUTPUT}")
    print(f"Audit: {SUMMARY_OUTPUT}")

    section("D/ST PROJECTION BENCHMARK COMPLETE")

    print(
        f"Decision: {decision}"
    )
    print(
        "No frozen D/ST source table was modified."
    )
    print(
        "Next step after benchmark review: "
        "freeze the benchmark or revise the compact "
        "model before building current 2026 D/ST projections."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 96)
        print("D/ST PROJECTION BENCHMARK FAILED")
        print("=" * 96)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)
