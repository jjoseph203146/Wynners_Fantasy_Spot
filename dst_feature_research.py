#!/usr/bin/env python3

"""
dst_feature_research.py

Deterministic historical research / pruning for leakage-safe NFL D/ST
pregame features.

Input
-----
SQLite:
    dst_pregame_features

Target
------
    fanduel_dst_points

Research design
---------------
1. Uses only the already leakage-safe pregame feature table.
2. Never modifies any frozen D/ST history/scoring/feature source table.
3. Evaluates numeric candidate predictors with chronological out-of-sample
   folds:
       Fold A: train 2023 -> test 2024
       Fold B: train 2023-2024 -> test 2025
4. Fits one-feature ordinary least squares models using TRAIN data only.
5. Compares each feature against the train-mean baseline on each test fold.
6. Computes full-sample Pearson/Spearman only as descriptive research
   diagnostics, never as the sole selection criterion.
7. Detects redundant feature pairs using absolute feature-feature
   correlation >= 0.85.
8. Produces a deterministic shortlist by ranking chronological performance,
   coverage, and redundancy.

Outputs
-------
CSV:
    audit_dst_feature_research_summary.csv
    dst_feature_research.csv
    dst_feature_redundancy.csv
    dst_feature_shortlist.csv

No production model is created here.
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

RESEARCH_OUTPUT = Path(CSV_DIR) / "dst_feature_research.csv"
REDUNDANCY_OUTPUT = Path(CSV_DIR) / "dst_feature_redundancy.csv"
SHORTLIST_OUTPUT = Path(CSV_DIR) / "dst_feature_shortlist.csv"
AUDIT_OUTPUT = Path(CSV_DIR) / "audit_dst_feature_research_summary.csv"

HISTORICAL_SEASONS = [2023, 2024, 2025]

# High-correlation threshold is used only to identify redundant pairs.
REDUNDANCY_CORR_THRESHOLD = 0.85

# These fields are current-game outcomes / targets and must never become
# predictors in the research set.
OUTCOME_COLUMNS = {
    "fanduel_dst_points",
    "fanduel_points_allowed",
    "sacks",
    "interceptions",
    "fumble_recoveries",
    "safeties",
    "blocked_kicks",
    "return_tds",
}

# Metadata / keys, not model predictors.
METADATA_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "game_date",
    "gametime",
    "game_datetime",
    "team",
    "opponent_team",
    "home_away",
    "feature_built_at",
}


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
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

    required = {"game_id", "season", "week", "team", TARGET}
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            "Missing required source columns: " + ", ".join(missing)
        )

    duplicates = int(df.duplicated(["game_id", "team"]).sum())
    if duplicates:
        raise RuntimeError(
            f"{SOURCE_TABLE} contains {duplicates} duplicate game/team rows."
        )

    return df


def candidate_features(df: pd.DataFrame) -> list[str]:
    candidates = []

    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        if col in OUTCOME_COLUMNS:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.notna().sum() == 0:
            continue

        # Require actual variability somewhere in the historical sample.
        if numeric.dropna().nunique() <= 1:
            continue

        candidates.append(col)

    return sorted(candidates)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """
    Correlation helper with no SciPy dependency.

    Pearson uses NumPy directly.
    Spearman is Pearson correlation of pandas average ranks, which is the
    standard Spearman rank-correlation definition and avoids pandas trying
    to import scipy.stats.
    """
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(frame) < 3:
        return np.nan

    if frame["x"].nunique() <= 1 or frame["y"].nunique() <= 1:
        return np.nan

    if method == "pearson":
        x_values = frame["x"].to_numpy(dtype=float)
        y_values = frame["y"].to_numpy(dtype=float)

    elif method == "spearman":
        x_values = (
            frame["x"]
            .rank(method="average")
            .to_numpy(dtype=float)
        )
        y_values = (
            frame["y"]
            .rank(method="average")
            .to_numpy(dtype=float)
        )

    else:
        raise ValueError(
            f"Unsupported correlation method: {method}"
        )

    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return np.nan

    return float(
        np.corrcoef(x_values, y_values)[0, 1]
    )


def fit_univariate_ols(
    train_x: pd.Series,
    train_y: pd.Series,
) -> tuple[float, float] | None:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(train_x, errors="coerce"),
            "y": pd.to_numeric(train_y, errors="coerce"),
        }
    ).dropna()

    if len(frame) < 10:
        return None

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)

    if np.nanstd(x) == 0:
        return None

    x_mean = x.mean()
    y_mean = y.mean()

    denom = np.sum((x - x_mean) ** 2)

    if denom <= 0:
        return None

    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean

    return float(intercept), float(slope)


def evaluate_fold(
    df: pd.DataFrame,
    feature: str,
    train_seasons: list[int],
    test_season: int,
) -> dict:
    train = df[df["season"].isin(train_seasons)].copy()
    test = df[df["season"].eq(test_season)].copy()

    train_x = pd.to_numeric(train[feature], errors="coerce")
    train_y = pd.to_numeric(train[TARGET], errors="coerce")
    test_x = pd.to_numeric(test[feature], errors="coerce")
    test_y = pd.to_numeric(test[TARGET], errors="coerce")

    fit = fit_univariate_ols(train_x, train_y)

    train_mean = float(train_y.dropna().mean())

    test_mask = test_x.notna() & test_y.notna()
    n_test = int(test_mask.sum())

    result = {
        "n_test": n_test,
        "mae": np.nan,
        "rmse": np.nan,
        "corr": np.nan,
        "baseline_mae": np.nan,
        "baseline_rmse": np.nan,
        "mae_improvement": np.nan,
        "rmse_improvement": np.nan,
        "slope": np.nan,
    }

    if fit is None or n_test < 5:
        return result

    intercept, slope = fit

    x = test_x[test_mask].to_numpy(dtype=float)
    y = test_y[test_mask].to_numpy(dtype=float)

    pred = intercept + slope * x
    baseline = np.full(len(y), train_mean, dtype=float)

    errors = pred - y
    base_errors = baseline - y

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    base_mae = float(np.mean(np.abs(base_errors)))
    base_rmse = float(np.sqrt(np.mean(base_errors ** 2)))

    corr = np.nan
    if len(y) >= 3 and np.std(pred) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(pred, y)[0, 1])

    result.update(
        {
            "mae": mae,
            "rmse": rmse,
            "corr": corr,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "mae_improvement": base_mae - mae,
            "rmse_improvement": base_rmse - rmse,
            "slope": slope,
        }
    )

    return result


def research_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    target = pd.to_numeric(df[TARGET], errors="coerce")

    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce")

        coverage = float(x.notna().mean())
        non_null = int(x.notna().sum())
        unique_values = int(x.dropna().nunique())

        pearson = safe_corr(x, target, "pearson")
        spearman = safe_corr(x, target, "spearman")

        fold_2024 = evaluate_fold(
            df,
            feature,
            train_seasons=[2023],
            test_season=2024,
        )

        fold_2025 = evaluate_fold(
            df,
            feature,
            train_seasons=[2023, 2024],
            test_season=2025,
        )

        valid_corrs = [
            v
            for v in [
                fold_2024["corr"],
                fold_2025["corr"],
            ]
            if pd.notna(v)
        ]

        mean_oos_corr = (
            float(np.mean(valid_corrs))
            if valid_corrs
            else np.nan
        )

        valid_rmse_improvements = [
            v
            for v in [
                fold_2024["rmse_improvement"],
                fold_2025["rmse_improvement"],
            ]
            if pd.notna(v)
        ]

        mean_rmse_improvement = (
            float(np.mean(valid_rmse_improvements))
            if valid_rmse_improvements
            else np.nan
        )

        valid_mae_improvements = [
            v
            for v in [
                fold_2024["mae_improvement"],
                fold_2025["mae_improvement"],
            ]
            if pd.notna(v)
        ]

        mean_mae_improvement = (
            float(np.mean(valid_mae_improvements))
            if valid_mae_improvements
            else np.nan
        )

        positive_rmse_folds = int(
            sum(
                pd.notna(v) and v > 0
                for v in [
                    fold_2024["rmse_improvement"],
                    fold_2025["rmse_improvement"],
                ]
            )
        )

        positive_corr_folds = int(
            sum(
                pd.notna(v) and v > 0
                for v in [
                    fold_2024["corr"],
                    fold_2025["corr"],
                ]
            )
        )

        direction_consistent = int(
            pd.notna(fold_2024["slope"])
            and pd.notna(fold_2025["slope"])
            and np.sign(fold_2024["slope"]) == np.sign(fold_2025["slope"])
        )

        rows.append(
            {
                "feature": feature,
                "coverage": coverage,
                "non_null_rows": non_null,
                "unique_values": unique_values,
                "pearson_full": pearson,
                "spearman_full": spearman,

                "test_2024_n": fold_2024["n_test"],
                "test_2024_mae": fold_2024["mae"],
                "test_2024_rmse": fold_2024["rmse"],
                "test_2024_corr": fold_2024["corr"],
                "test_2024_baseline_mae": fold_2024["baseline_mae"],
                "test_2024_baseline_rmse": fold_2024["baseline_rmse"],
                "test_2024_mae_improvement": fold_2024["mae_improvement"],
                "test_2024_rmse_improvement": fold_2024["rmse_improvement"],
                "test_2024_slope": fold_2024["slope"],

                "test_2025_n": fold_2025["n_test"],
                "test_2025_mae": fold_2025["mae"],
                "test_2025_rmse": fold_2025["rmse"],
                "test_2025_corr": fold_2025["corr"],
                "test_2025_baseline_mae": fold_2025["baseline_mae"],
                "test_2025_baseline_rmse": fold_2025["baseline_rmse"],
                "test_2025_mae_improvement": fold_2025["mae_improvement"],
                "test_2025_rmse_improvement": fold_2025["rmse_improvement"],
                "test_2025_slope": fold_2025["slope"],

                "mean_oos_corr": mean_oos_corr,
                "mean_oos_rmse_improvement": mean_rmse_improvement,
                "mean_oos_mae_improvement": mean_mae_improvement,
                "positive_rmse_folds": positive_rmse_folds,
                "positive_corr_folds": positive_corr_folds,
                "direction_consistent": direction_consistent,
            }
        )

    result = pd.DataFrame(rows)

    # Deterministic research ranking:
    # 1) beats RMSE baseline in both chronological folds
    # 2) positive correlation in both folds
    # 3) stable direction
    # 4) average OOS RMSE improvement
    # 5) average OOS correlation
    # 6) coverage
    result = result.sort_values(
        [
            "positive_rmse_folds",
            "positive_corr_folds",
            "direction_consistent",
            "mean_oos_rmse_improvement",
            "mean_oos_corr",
            "coverage",
            "feature",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    result["research_rank"] = np.arange(1, len(result) + 1)

    return result


def build_redundancy(
    df: pd.DataFrame,
    research: pd.DataFrame,
) -> pd.DataFrame:
    features = research["feature"].tolist()

    numeric = pd.DataFrame(
        {
            f: pd.to_numeric(df[f], errors="coerce")
            for f in features
        }
    )

    corr = numeric.corr(method="pearson")

    rank_map = dict(
        zip(
            research["feature"],
            research["research_rank"],
        )
    )

    rows = []

    for i, left in enumerate(features):
        for right in features[i + 1:]:
            value = corr.loc[left, right]

            if pd.isna(value):
                continue

            if abs(float(value)) < REDUNDANCY_CORR_THRESHOLD:
                continue

            left_rank = int(rank_map[left])
            right_rank = int(rank_map[right])

            preferred = left if left_rank < right_rank else right
            redundant = right if preferred == left else left

            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "pearson_corr": float(value),
                    "abs_corr": abs(float(value)),
                    "preferred_feature": preferred,
                    "redundant_feature": redundant,
                    "preferred_research_rank": int(rank_map[preferred]),
                    "redundant_research_rank": int(rank_map[redundant]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "pearson_corr",
                "abs_corr",
                "preferred_feature",
                "redundant_feature",
                "preferred_research_rank",
                "redundant_research_rank",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["abs_corr", "preferred_research_rank"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_shortlist(
    research: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conservative deterministic shortlist.

    A feature is considered chronologically supported if:
      - it beats the train-mean RMSE baseline in BOTH OOS folds, OR
      - it has positive OOS correlation in BOTH folds and improves RMSE
        in at least one fold.

    Then redundant lower-ranked fields are removed only when correlated
    >= 0.85 with a higher-ranked supported field.

    This avoids selecting features merely because of in-sample correlation.
    """

    supported = research[
        (
            research["positive_rmse_folds"].eq(2)
        )
        |
        (
            research["positive_corr_folds"].eq(2)
            & research["positive_rmse_folds"].ge(1)
        )
    ].copy()

    supported_names = set(supported["feature"])

    redundant_drop = set()

    if not redundancy.empty:
        for row in redundancy.itertuples(index=False):
            if (
                row.preferred_feature in supported_names
                and row.redundant_feature in supported_names
            ):
                redundant_drop.add(row.redundant_feature)

    supported["redundant_drop"] = (
        supported["feature"].isin(redundant_drop).astype(int)
    )

    supported["shortlist"] = (
        supported["redundant_drop"].eq(0).astype(int)
    )

    supported = supported.sort_values(
        ["shortlist", "research_rank"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return supported


def audit(
    df: pd.DataFrame,
    features: list[str],
    research: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    section("D/ST FEATURE RESEARCH AUDIT")

    rows = []

    def add(item, value, expected="", status="INFO"):
        rows.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    add(
        "source_rows",
        len(df),
        1710,
        "PASS" if len(df) == 1710 else "CHECK",
    )

    add(
        "duplicate_game_team_rows",
        int(df.duplicated(["game_id", "team"]).sum()),
        0,
        (
            "PASS"
            if int(df.duplicated(["game_id", "team"]).sum()) == 0
            else "FAIL"
        ),
    )

    add(
        "null_target_rows",
        int(df[TARGET].isna().sum()),
        0,
        "PASS" if int(df[TARGET].isna().sum()) == 0 else "FAIL",
    )

    for season in HISTORICAL_SEASONS:
        count = int(df["season"].eq(season).sum())
        add(
            f"rows_{season}",
            count,
            570,
            "PASS" if count == 570 else "CHECK",
        )

    add(
        "candidate_feature_count",
        len(features),
        "",
        "INFO",
    )

    add(
        "research_rows",
        len(research),
        len(features),
        "PASS" if len(research) == len(features) else "FAIL",
    )

    forbidden_candidates = sorted(
        set(features)
        & (OUTCOME_COLUMNS | METADATA_COLUMNS)
    )

    add(
        "forbidden_candidate_features",
        len(forbidden_candidates),
        0,
        "PASS" if not forbidden_candidates else "FAIL",
    )

    add(
        "chronologically_supported_features",
        len(shortlist),
        "",
        "INFO",
    )

    final_count = (
        int(shortlist["shortlist"].sum())
        if not shortlist.empty
        else 0
    )

    add(
        "final_shortlist_features",
        final_count,
        "",
        "INFO",
    )

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    failures = summary[summary["status"].eq("FAIL")]

    print()
    if failures.empty:
        print("STRUCTURAL RESEARCH AUDIT: PASS")
    else:
        print("STRUCTURAL RESEARCH AUDIT: FAIL")

    return summary


def print_top_research(research: pd.DataFrame) -> None:
    section("TOP D/ST FEATURE RESEARCH RESULTS")

    columns = [
        "research_rank",
        "feature",
        "coverage",
        "pearson_full",
        "spearman_full",
        "test_2024_corr",
        "test_2024_rmse_improvement",
        "test_2025_corr",
        "test_2025_rmse_improvement",
        "positive_rmse_folds",
        "direction_consistent",
    ]

    print(
        research[columns]
        .head(30)
        .to_string(index=False)
    )


def print_shortlist(shortlist: pd.DataFrame) -> None:
    section("D/ST CHRONOLOGICAL FEATURE SHORTLIST")

    if shortlist.empty:
        print("No features met chronological support criteria.")
        return

    columns = [
        "research_rank",
        "feature",
        "mean_oos_corr",
        "mean_oos_rmse_improvement",
        "positive_rmse_folds",
        "positive_corr_folds",
        "direction_consistent",
        "redundant_drop",
        "shortlist",
    ]

    print(shortlist[columns].to_string(index=False))


def main() -> None:
    section("NFL D/ST FEATURE RESEARCH / PRUNING")

    print(f"Database: {DATABASE_PATH}")
    print(f"Source: {SOURCE_TABLE}")
    print(f"Target: {TARGET}")
    print("Chronological folds:")
    print("  2023 -> 2024")
    print("  2023-2024 -> 2025")
    print("Production model: NOT created here.")
    print("Frozen source tables: NOT modified.")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        df = load_features(conn)

    features = candidate_features(df)

    section("RESEARCHING CANDIDATE FEATURES")

    print(f"Candidate numeric predictors: {len(features)}")

    research = research_features(
        df,
        features,
    )

    redundancy = build_redundancy(
        df,
        research,
    )

    shortlist = build_shortlist(
        research,
        redundancy,
    )

    summary = audit(
        df,
        features,
        research,
        shortlist,
    )

    failures = summary[
        summary["status"].eq("FAIL")
    ]

    research.to_csv(
        RESEARCH_OUTPUT,
        index=False,
    )

    redundancy.to_csv(
        REDUNDANCY_OUTPUT,
        index=False,
    )

    shortlist.to_csv(
        SHORTLIST_OUTPUT,
        index=False,
    )

    summary.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print_top_research(research)
    print_shortlist(shortlist)

    section("RESEARCH EXPORTS")

    print(f"Research: {RESEARCH_OUTPUT}")
    print(f"Redundancy: {REDUNDANCY_OUTPUT}")
    print(f"Shortlist: {SHORTLIST_OUTPUT}")
    print(f"Audit: {AUDIT_OUTPUT}")

    if not failures.empty:
        raise RuntimeError(
            "D/ST feature research structural audit failed."
        )

    section("D/ST FEATURE RESEARCH COMPLETE")

    final_count = (
        int(shortlist["shortlist"].sum())
        if not shortlist.empty
        else 0
    )

    print(f"Final non-redundant shortlist count: {final_count}")
    print(
        "Next layer after review: chronological D/ST "
        "projection benchmark."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 92)
        print("D/ST FEATURE RESEARCH FAILED")
        print("=" * 92)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
