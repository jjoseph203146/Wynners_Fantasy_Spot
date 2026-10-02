#!/usr/bin/env python3

"""
WFS NFL SEASON & PLAYOFF SIMULATOR
Simulator V1 — Experiment 001

Purpose
-------
Test a market-independent direct game-outcome model.

Historical contract:
- source: processed/forecast_v1_team_game_training.csv
- one HOME-perspective row per game
- train: 2023 + 2024
- holdout: 2025
- no market/spread/total/moneyline predictors
- no target/result predictors
- no production writes
- no production forecast changes

Models:
- Ridge regression: home margin
- Ridge regression: game total
- Logistic regression: home win probability

The experiment also reports simple non-model baselines.
"""

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/home/mwynn/nfl_data_engine")

SOURCE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training.csv"
)

OUTDIR = (
    ROOT
    / "processed"
    / "simulator_v1"
)

PREDICTIONS = (
    OUTDIR
    / "simulator_v1_experiment_001_predictions.csv"
)

AUDIT = (
    OUTDIR
    / "simulator_v1_experiment_001_audit.json"
)

TRAIN_SEASONS = {2023, 2024}
HOLDOUT_SEASON = 2025

MARKET_COLUMNS = {
    "market_home_spread_raw",
    "market_total",
    "team_moneyline",
    "opponent_moneyline",
}

TARGET_COLUMNS = {
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
}

IDENTITY_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "game_date",
    "weekday",
    "gametime",
    "team",
    "opponent_team",
}

# Do not allow direct identity memorization in EXP001.
DIRECT_IDENTITY_EXCLUSIONS = {
    "coach",
    "opponent_coach",
    "team_qb_name",
    "opponent_qb_name",
}

# Weather is intentionally deferred in EXP001 because historical
# availability is incomplete.
DEFERRED_COLUMNS = {
    "temp",
    "wind",
}

EXPECTED_CONSTANT_EXCLUSIONS = {
    "team_qb_out_count",
    "team_qb_doubtful_count",
    "team_out_fd_avg3_sum",
    "team_out_snap_pct_avg3_sum",
    "team_out_opportunities_avg3_sum",
    "team_out_targets_avg3_sum",
    "team_out_carries_avg3_sum",
    "team_out_established_role_count",
    "opp_qb_out_count",
    "opp_qb_doubtful_count",
    "opp_out_fd_avg3_sum",
    "opp_out_snap_pct_avg3_sum",
    "opp_out_opportunities_avg3_sum",
    "opp_out_targets_avg3_sum",
    "opp_out_carries_avg3_sum",
    "opp_out_established_role_count",
}


def fail(message):
    print(f"FAIL | {message}")
    raise SystemExit(1)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def finite(value):
    return value is not None and math.isfinite(float(value))


def rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def mae(y_true, y_pred):
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


def ece(y_true, prob, bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    total = len(y)

    if total == 0:
        return float("nan")

    score = 0.0

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == bins - 1:
            mask = (
                (p >= lo)
                & (p <= hi)
            )
        else:
            mask = (
                (p >= lo)
                & (p < hi)
            )

        n = int(mask.sum())

        if n == 0:
            continue

        observed = float(y[mask].mean())
        predicted = float(p[mask].mean())

        score += (
            n / total
        ) * abs(
            observed - predicted
        )

    return float(score)


def validate_mirror_contract(df):
    failures = []

    for gid, g in df.groupby(
        "game_id",
        sort=False,
    ):
        if len(g) != 2:
            failures.append(
                f"{gid}:rows={len(g)}"
            )
            continue

        a = g.iloc[0]
        b = g.iloc[1]

        if not (
            str(a["team"])
            == str(b["opponent_team"])
            and str(b["team"])
            == str(a["opponent_team"])
        ):
            failures.append(
                f"{gid}:team_mirror"
            )

        if int(a["is_home"]) + int(
            b["is_home"]
        ) != 1:
            failures.append(
                f"{gid}:home_mirror"
            )

        if not np.isclose(
            float(a["target_margin"]),
            -float(b["target_margin"]),
        ):
            failures.append(
                f"{gid}:margin_mirror"
            )

        if not np.isclose(
            float(a["target_total_points"]),
            float(b["target_total_points"]),
        ):
            failures.append(
                f"{gid}:total_mirror"
            )

    if failures:
        fail(
            "mirror contract failed | "
            + ";".join(failures[:10])
        )


def build_model(alpha=10.0):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def build_classifier():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


print(
    "===== WFS SIMULATOR V1 — "
    "EXPERIMENT 001 ====="
)

if not SOURCE.exists():
    fail(
        f"source missing: {SOURCE}"
    )

source_sha = sha256(SOURCE)

df = pd.read_csv(SOURCE)

print(
    f"SOURCE_ROWS={len(df)}"
)
print(
    f"SOURCE_SHA256={source_sha}"
)

required = (
    IDENTITY_COLUMNS
    | MARKET_COLUMNS
    | TARGET_COLUMNS
    | {"is_home"}
)

missing = sorted(
    required - set(df.columns)
)

if missing:
    fail(
        "missing required columns: "
        + ",".join(missing)
    )

# ------------------------------------------------------------
# Structural validation
# ------------------------------------------------------------

validate_mirror_contract(df)

print(
    "GAME_MIRROR_CONTRACT=PASS"
)

# ------------------------------------------------------------
# Collapse to one home-perspective row per game.
# ------------------------------------------------------------

home = (
    df[
        pd.to_numeric(
            df["is_home"],
            errors="coerce",
        )
        == 1
    ]
    .copy()
    .sort_values(
        [
            "season",
            "week",
            "game_date",
            "game_id",
        ]
    )
    .reset_index(drop=True)
)

if len(home) != df["game_id"].nunique():
    fail(
        "home-row collapse did not produce "
        "exactly one row per game"
    )

if home["game_id"].duplicated().any():
    fail(
        "duplicate game_id after home collapse"
    )

print(
    f"HOME_GAME_ROWS={len(home)}"
)

# ------------------------------------------------------------
# Train / holdout contract
# ------------------------------------------------------------

train = home[
    home["season"].isin(
        TRAIN_SEASONS
    )
].copy()

test = home[
    home["season"]
    == HOLDOUT_SEASON
].copy()

if len(train) != 570:
    fail(
        f"unexpected training games: {len(train)}"
    )

if len(test) != 285:
    fail(
        f"unexpected holdout games: {len(test)}"
    )

if set(train["game_id"]) & set(
    test["game_id"]
):
    fail(
        "train/holdout game overlap"
    )

print(
    f"TRAIN_GAMES={len(train)}"
)
print(
    f"HOLDOUT_GAMES={len(test)}"
)

# ------------------------------------------------------------
# Feature contract
# ------------------------------------------------------------

excluded = (
    MARKET_COLUMNS
    | TARGET_COLUMNS
    | IDENTITY_COLUMNS
    | DIRECT_IDENTITY_EXCLUSIONS
    | DEFERRED_COLUMNS
)

candidate_features = []

for col in home.columns:
    if col in excluded:
        continue

    if not pd.api.types.is_numeric_dtype(
        home[col]
    ):
        continue

    # Determine variance from TRAIN only.
    x = pd.to_numeric(
        train[col],
        errors="coerce",
    )

    if x.notna().sum() == 0:
        continue

    if x.dropna().nunique() <= 1:
        continue

    candidate_features.append(col)

candidate_features = sorted(
    candidate_features
)

# Explicit leakage guard.
for col in candidate_features:
    low = col.lower()

    if col in MARKET_COLUMNS:
        fail(
            f"market leakage: {col}"
        )

    if col in TARGET_COLUMNS:
        fail(
            f"target leakage: {col}"
        )

    if (
        "moneyline" in low
        or "market_" in low
    ):
        fail(
            f"market-like feature escaped: {col}"
        )

if not candidate_features:
    fail(
        "zero candidate features"
    )

print(
    f"MODEL_FEATURES={len(candidate_features)}"
)

print(
    "MARKET_PREDICTORS_USED=0"
)
print(
    "TARGET_PREDICTORS_USED=0"
)

print(
    "\n===== FEATURE CONTRACT ====="
)

for col in candidate_features:
    print(
        f"ALLOW|{col}"
    )

for col in sorted(
    MARKET_COLUMNS
):
    print(
        f"EXCLUDE_MARKET|{col}"
    )

for col in sorted(
    TARGET_COLUMNS
):
    print(
        f"EXCLUDE_TARGET|{col}"
    )

for col in sorted(
    DIRECT_IDENTITY_EXCLUSIONS
):
    print(
        f"EXCLUDE_IDENTITY|{col}"
    )

for col in sorted(
    DEFERRED_COLUMNS
):
    print(
        f"DEFER|{col}"
    )

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

y_margin_train = pd.to_numeric(
    train["target_margin"],
    errors="coerce",
)

y_margin_test = pd.to_numeric(
    test["target_margin"],
    errors="coerce",
)

y_total_train = pd.to_numeric(
    train["target_total_points"],
    errors="coerce",
)

y_total_test = pd.to_numeric(
    test["target_total_points"],
    errors="coerce",
)

# target_win has NULL for ties.
# For probability training/evaluation, exclude ties.
train_win_mask = (
    train["target_win"].notna()
)

test_win_mask = (
    test["target_win"].notna()
)

y_win_train = pd.to_numeric(
    train.loc[
        train_win_mask,
        "target_win",
    ],
    errors="coerce",
)

y_win_test = pd.to_numeric(
    test.loc[
        test_win_mask,
        "target_win",
    ],
    errors="coerce",
)

if not (
    y_margin_train.notna().all()
    and y_margin_test.notna().all()
    and y_total_train.notna().all()
    and y_total_test.notna().all()
):
    fail(
        "missing regression target"
    )

# ------------------------------------------------------------
# Baselines
# ------------------------------------------------------------

baseline_margin = float(
    y_margin_train.mean()
)

baseline_total = float(
    y_total_train.mean()
)

baseline_home_win_prob = float(
    y_win_train.mean()
)

baseline_margin_pred = np.full(
    len(test),
    baseline_margin,
)

baseline_total_pred = np.full(
    len(test),
    baseline_total,
)

baseline_win_prob = np.full(
    int(test_win_mask.sum()),
    baseline_home_win_prob,
)

baseline_win_class = (
    baseline_win_prob >= 0.5
).astype(int)

# ------------------------------------------------------------
# Direct margin model
# ------------------------------------------------------------

X_train = train[
    candidate_features
].copy()

X_test = test[
    candidate_features
].copy()

margin_model = build_model(
    alpha=10.0
)

margin_model.fit(
    X_train,
    y_margin_train,
)

pred_margin = margin_model.predict(
    X_test
)

# ------------------------------------------------------------
# Direct total model
# ------------------------------------------------------------

total_model = build_model(
    alpha=10.0
)

total_model.fit(
    X_train,
    y_total_train,
)

pred_total = total_model.predict(
    X_test
)

# ------------------------------------------------------------
# Home win probability model
# ------------------------------------------------------------

win_model = build_classifier()

win_model.fit(
    X_train.loc[
        train_win_mask
    ],
    y_win_train,
)

pred_win_prob_full = (
    win_model.predict_proba(
        X_test
    )[:, 1]
)

pred_win_prob_eval = (
    pred_win_prob_full[
        test_win_mask.to_numpy()
    ]
)

pred_win_class = (
    pred_win_prob_eval >= 0.5
).astype(int)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

metrics = {
    "baseline": {
        "margin_mae": mae(
            y_margin_test,
            baseline_margin_pred,
        ),
        "margin_rmse": rmse(
            y_margin_test,
            baseline_margin_pred,
        ),
        "total_mae": mae(
            y_total_test,
            baseline_total_pred,
        ),
        "total_rmse": rmse(
            y_total_test,
            baseline_total_pred,
        ),
        "home_win_accuracy": float(
            accuracy_score(
                y_win_test,
                baseline_win_class,
            )
        ),
        "home_win_brier": float(
            brier_score_loss(
                y_win_test,
                baseline_win_prob,
            )
        ),
        "home_win_logloss": float(
            log_loss(
                y_win_test,
                baseline_win_prob,
                labels=[0, 1],
            )
        ),
        "home_win_ece": ece(
            y_win_test,
            baseline_win_prob,
        ),
    },
    "simulator_v1_exp001": {
        "margin_mae": mae(
            y_margin_test,
            pred_margin,
        ),
        "margin_rmse": rmse(
            y_margin_test,
            pred_margin,
        ),
        "total_mae": mae(
            y_total_test,
            pred_total,
        ),
        "total_rmse": rmse(
            y_total_test,
            pred_total,
        ),
        "home_win_accuracy": float(
            accuracy_score(
                y_win_test,
                pred_win_class,
            )
        ),
        "home_win_brier": float(
            brier_score_loss(
                y_win_test,
                pred_win_prob_eval,
            )
        ),
        "home_win_logloss": float(
            log_loss(
                y_win_test,
                pred_win_prob_eval,
                labels=[0, 1],
            )
        ),
        "home_win_ece": ece(
            y_win_test,
            pred_win_prob_eval,
        ),
    },
}

# ------------------------------------------------------------
# Prediction artifact
# ------------------------------------------------------------

pred = test[
    [
        "game_id",
        "season",
        "week",
        "game_date",
        "team",
        "opponent_team",
        "target_team_points",
        "target_opponent_points",
        "target_margin",
        "target_total_points",
        "target_win",
        "target_result",
    ]
].copy()

pred = pred.rename(
    columns={
        "team": "home_team",
        "opponent_team": "away_team",
        "target_team_points":
            "actual_home_points",
        "target_opponent_points":
            "actual_away_points",
        "target_margin":
            "actual_home_margin",
        "target_total_points":
            "actual_total_points",
        "target_win":
            "actual_home_win",
        "target_result":
            "actual_home_result",
    }
)

pred[
    "baseline_home_margin"
] = baseline_margin

pred[
    "baseline_total_points"
] = baseline_total

pred[
    "baseline_home_win_probability"
] = baseline_home_win_prob

pred[
    "pred_home_margin"
] = pred_margin

pred[
    "pred_total_points"
] = pred_total

pred[
    "pred_home_win_probability"
] = pred_win_prob_full

# Coherent score decomposition from margin + total.
pred[
    "pred_home_points"
] = (
    pred["pred_total_points"]
    + pred["pred_home_margin"]
) / 2.0

pred[
    "pred_away_points"
] = (
    pred["pred_total_points"]
    - pred["pred_home_margin"]
) / 2.0

pred[
    "pred_winner"
] = np.where(
    pred[
        "pred_home_win_probability"
    ] >= 0.5,
    pred["home_team"],
    pred["away_team"],
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

pred.to_csv(
    PREDICTIONS,
    index=False,
)

# ------------------------------------------------------------
# Audit artifact
# ------------------------------------------------------------

audit = {
    "status": "PASS",
    "experiment":
        "SIMULATOR_V1_EXPERIMENT_001",
    "purpose":
        "market-independent direct game outcome model",
    "source": {
        "path": str(SOURCE),
        "sha256": source_sha,
        "source_rows": int(len(df)),
        "unique_games": int(
            df["game_id"].nunique()
        ),
    },
    "grain": {
        "training_grain":
            "one home-perspective row per game",
        "mirror_contract":
            "PASS",
    },
    "split": {
        "train_seasons":
            sorted(TRAIN_SEASONS),
        "holdout_season":
            HOLDOUT_SEASON,
        "train_games":
            int(len(train)),
        "holdout_games":
            int(len(test)),
    },
    "feature_contract": {
        "feature_count":
            len(candidate_features),
        "features":
            candidate_features,
        "market_predictors_used":
            [],
        "target_predictors_used":
            [],
        "direct_identity_excluded":
            sorted(
                DIRECT_IDENTITY_EXCLUSIONS
            ),
        "deferred":
            sorted(DEFERRED_COLUMNS),
        "constant_exclusions_expected":
            sorted(
                EXPECTED_CONSTANT_EXCLUSIONS
            ),
    },
    "model": {
        "margin":
            "Ridge(alpha=10)",
        "total":
            "Ridge(alpha=10)",
        "home_win_probability":
            "LogisticRegression(C=0.25)",
    },
    "metrics_2025_holdout":
        metrics,
    "production": {
        "production_forecast_modified":
            False,
        "optimizer_modified":
            False,
        "production_database_written":
            False,
    },
    "output": {
        "predictions_csv":
            str(PREDICTIONS),
    },
}

AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print(
    "\n===== 2025 HOLDOUT RESULTS ====="
)

for name, m in metrics.items():
    print(f"\nMODEL={name}")

    for key, value in m.items():
        print(
            f"{key}={value:.6f}"
        )

print(
    "\n===== DELTAS VS BASELINE ====="
)

b = metrics["baseline"]
m = metrics[
    "simulator_v1_exp001"
]

print(
    "MARGIN_MAE_DELTA="
    f"{m['margin_mae'] - b['margin_mae']:+.6f}"
)

print(
    "MARGIN_RMSE_DELTA="
    f"{m['margin_rmse'] - b['margin_rmse']:+.6f}"
)

print(
    "TOTAL_MAE_DELTA="
    f"{m['total_mae'] - b['total_mae']:+.6f}"
)

print(
    "TOTAL_RMSE_DELTA="
    f"{m['total_rmse'] - b['total_rmse']:+.6f}"
)

print(
    "WIN_ACCURACY_DELTA="
    f"{m['home_win_accuracy'] - b['home_win_accuracy']:+.6f}"
)

print(
    "WIN_BRIER_DELTA="
    f"{m['home_win_brier'] - b['home_win_brier']:+.6f}"
)

print(
    "WIN_LOGLOSS_DELTA="
    f"{m['home_win_logloss'] - b['home_win_logloss']:+.6f}"
)

print(
    "WIN_ECE_DELTA="
    f"{m['home_win_ece'] - b['home_win_ece']:+.6f}"
)

print(
    "\n===== OUTPUT ====="
)
print(
    f"PREDICTIONS={PREDICTIONS}"
)
print(
    f"AUDIT={AUDIT}"
)
print(
    f"PREDICTIONS_SHA256={sha256(PREDICTIONS)}"
)
print(
    f"AUDIT_SHA256={sha256(AUDIT)}"
)

print(
    "\nMARKET_PREDICTORS_USED=0"
)
print(
    "PRODUCTION_FORECAST_MODIFIED=FALSE"
)
print(
    "OPTIMIZER_MODIFIED=FALSE"
)
print(
    "PRODUCTION_DATABASE_WRITTEN=FALSE"
)
print(
    "SIMULATOR_V1_EXPERIMENT_001=PASS"
)
