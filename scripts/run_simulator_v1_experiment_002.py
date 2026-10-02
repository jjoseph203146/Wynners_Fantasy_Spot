#!/usr/bin/env python3

"""
WFS NFL SEASON & PLAYOFF SIMULATOR
Simulator V1 — Experiment 002

Purpose
-------
Improve the market-independent margin/probability architecture without
using the 2025 holdout for model selection or calibration.

Contracts
---------
- source: processed/forecast_v1_team_game_training.csv
- one HOME-perspective row per game
- development: 2023 + 2024
- untouched final holdout: 2025
- no market/spread/total/moneyline predictors
- no target/result predictors
- no production writes
- no production forecast changes
- EXP001 artifacts untouched

EXP002
------
1. Rebuild the EXP001 numeric feature contract.
2. Select Ridge alpha using TRAINING-ONLY season-forward validation:
       train 2023 -> validate 2024
3. Generate out-of-fold development margin predictions:
       train 2023 -> predict 2024
       train 2024 -> predict 2023
   These predictions are used ONLY to estimate residual scale and
   probability calibration.
4. Convert predicted margins to raw home-win probabilities using the
   Normal residual CDF.
5. Fit a one-dimensional logistic calibration model using ONLY
   development OOF predicted margins and realized development outcomes.
6. Fit the selected margin model on all 2023+2024 games.
7. Evaluate exactly once on untouched 2025.
8. Re-run the EXP001 logistic architecture for apples-to-apples
   comparison.
9. Report confidence buckets and totals diagnostics.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

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
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/mwynn/nfl_data_engine")

SOURCE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training.csv"
)

EXP001_AUDIT = (
    ROOT
    / "processed"
    / "simulator_v1"
    / "simulator_v1_experiment_001_audit.json"
)

EXP001_PREDICTIONS = (
    ROOT
    / "processed"
    / "simulator_v1"
    / "simulator_v1_experiment_001_predictions.csv"
)

OUTDIR = (
    ROOT
    / "processed"
    / "simulator_v1"
)

PREDICTIONS = (
    OUTDIR
    / "simulator_v1_experiment_002_predictions.csv"
)

AUDIT = (
    OUTDIR
    / "simulator_v1_experiment_002_audit.json"
)

BUCKETS = (
    OUTDIR
    / "simulator_v1_experiment_002_probability_buckets.csv"
)

DEV_SEASONS = {2023, 2024}
HOLDOUT_SEASON = 2025

ALPHAS = [
    0.1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]

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

DIRECT_IDENTITY_EXCLUSIONS = {
    "coach",
    "opponent_coach",
    "team_qb_name",
    "opponent_qb_name",
}

DEFERRED_COLUMNS = {
    "temp",
    "wind",
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


def mae(y_true, y_pred):
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


def rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
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

    result = 0.0

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

        result += (
            n / total
        ) * abs(
            float(y[mask].mean())
            - float(p[mask].mean())
        )

    return float(result)


def normal_cdf_array(x):
    x = np.asarray(x, dtype=float)

    return np.array(
        [
            0.5
            * (
                1.0
                + math.erf(
                    float(v)
                    / math.sqrt(2.0)
                )
            )
            for v in x
        ],
        dtype=float,
    )


def make_ridge(alpha):
    return Pipeline(
        [
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
                    alpha=float(alpha),
                ),
            ),
        ]
    )


def make_logistic():
    return Pipeline(
        [
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


def probability_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    p = np.clip(
        p,
        1e-6,
        1.0 - 1e-6,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "logloss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
        "ece": ece(
            y,
            p,
        ),
    }


print(
    "===== WFS SIMULATOR V1 — "
    "EXPERIMENT 002 ====="
)

if not SOURCE.exists():
    fail(
        f"missing source: {SOURCE}"
    )

if not EXP001_AUDIT.exists():
    fail(
        "EXP001 audit missing"
    )

if not EXP001_PREDICTIONS.exists():
    fail(
        "EXP001 predictions missing"
    )

source_sha = sha256(SOURCE)
exp001_audit_sha_before = sha256(
    EXP001_AUDIT
)
exp001_pred_sha_before = sha256(
    EXP001_PREDICTIONS
)

print(
    f"SOURCE_SHA256={source_sha}"
)
print(
    "EXP001_AUDIT_SHA256_BEFORE="
    f"{exp001_audit_sha_before}"
)
print(
    "EXP001_PREDICTIONS_SHA256_BEFORE="
    f"{exp001_pred_sha_before}"
)

df = pd.read_csv(SOURCE)

# ------------------------------------------------------------
# One home row per game
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

if len(home) != 855:
    fail(
        f"expected 855 games, got {len(home)}"
    )

if home["game_id"].duplicated().any():
    fail(
        "duplicate game_id after home collapse"
    )

dev = home[
    home["season"].isin(
        DEV_SEASONS
    )
].copy()

test = home[
    home["season"]
    == HOLDOUT_SEASON
].copy()

if len(dev) != 570:
    fail(
        f"expected 570 development games, got {len(dev)}"
    )

if len(test) != 285:
    fail(
        f"expected 285 holdout games, got {len(test)}"
    )

print(
    f"DEVELOPMENT_GAMES={len(dev)}"
)
print(
    f"HOLDOUT_GAMES={len(test)}"
)

# ------------------------------------------------------------
# Rebuild EXP001 feature contract
# ------------------------------------------------------------

excluded = (
    MARKET_COLUMNS
    | TARGET_COLUMNS
    | IDENTITY_COLUMNS
    | DIRECT_IDENTITY_EXCLUSIONS
    | DEFERRED_COLUMNS
)

features = []

for col in home.columns:
    if col in excluded:
        continue

    if not pd.api.types.is_numeric_dtype(
        home[col]
    ):
        continue

    x = pd.to_numeric(
        dev[col],
        errors="coerce",
    )

    if x.notna().sum() == 0:
        continue

    if x.dropna().nunique() <= 1:
        continue

    features.append(col)

features = sorted(features)

if len(features) != 92:
    fail(
        "EXP001 feature-contract drift | "
        f"expected=92 actual={len(features)}"
    )

for col in features:
    low = col.lower()

    if (
        col in MARKET_COLUMNS
        or col in TARGET_COLUMNS
        or "moneyline" in low
        or "market_" in low
    ):
        fail(
            f"leakage feature escaped: {col}"
        )

print(
    f"MODEL_FEATURES={len(features)}"
)
print(
    "MARKET_PREDICTORS_USED=0"
)
print(
    "TARGET_PREDICTORS_USED=0"
)

# ------------------------------------------------------------
# Training-only alpha selection:
# train 2023 -> validate 2024.
# ------------------------------------------------------------

print(
    "\n===== TRAINING-ONLY RIDGE SELECTION ====="
)

train23 = dev[
    dev["season"] == 2023
].copy()

valid24 = dev[
    dev["season"] == 2024
].copy()

if len(train23) != 285:
    fail(
        f"2023 count unexpected: {len(train23)}"
    )

if len(valid24) != 285:
    fail(
        f"2024 count unexpected: {len(valid24)}"
    )

X23 = train23[features]
X24 = valid24[features]

y23 = pd.to_numeric(
    train23["target_margin"],
    errors="coerce",
)

y24 = pd.to_numeric(
    valid24["target_margin"],
    errors="coerce",
)

alpha_results = []

for alpha in ALPHAS:
    model = make_ridge(alpha)

    model.fit(
        X23,
        y23,
    )

    p = model.predict(
        X24
    )

    row = {
        "alpha": float(alpha),
        "mae": mae(
            y24,
            p,
        ),
        "rmse": rmse(
            y24,
            p,
        ),
    }

    alpha_results.append(row)

    print(
        f"ALPHA={alpha:8.1f}|"
        f"MAE={row['mae']:.6f}|"
        f"RMSE={row['rmse']:.6f}"
    )

# Primary selection criterion = RMSE.
# MAE then alpha provide deterministic tie breaks.
best = min(
    alpha_results,
    key=lambda r: (
        r["rmse"],
        r["mae"],
        r["alpha"],
    ),
)

selected_alpha = float(
    best["alpha"]
)

print(
    f"SELECTED_ALPHA={selected_alpha}"
)
print(
    "ALPHA_SELECTION_USED_2025=FALSE"
)

# ------------------------------------------------------------
# Development OOF predictions for calibration.
#
# Fold A: 2023 -> 2024
# Fold B: 2024 -> 2023
#
# No row predicts itself.
# ------------------------------------------------------------

print(
    "\n===== DEVELOPMENT OOF CALIBRATION SET ====="
)

oof_parts = []

for train_season, pred_season in [
    (2023, 2024),
    (2024, 2023),
]:
    tr = dev[
        dev["season"]
        == train_season
    ].copy()

    va = dev[
        dev["season"]
        == pred_season
    ].copy()

    model = make_ridge(
        selected_alpha
    )

    model.fit(
        tr[features],
        pd.to_numeric(
            tr["target_margin"],
            errors="coerce",
        ),
    )

    margin_pred = model.predict(
        va[features]
    )

    part = va[
        [
            "game_id",
            "season",
            "target_margin",
            "target_win",
            "target_result",
        ]
    ].copy()

    part[
        "oof_margin_prediction"
    ] = margin_pred

    oof_parts.append(part)

oof = (
    pd.concat(
        oof_parts,
        ignore_index=True,
    )
    .sort_values(
        [
            "season",
            "game_id",
        ]
    )
    .reset_index(drop=True)
)

if len(oof) != 570:
    fail(
        f"OOF count unexpected: {len(oof)}"
    )

if oof["game_id"].duplicated().any():
    fail(
        "OOF duplicate game"
    )

oof_residual = (
    pd.to_numeric(
        oof["target_margin"],
        errors="coerce",
    )
    - pd.to_numeric(
        oof["oof_margin_prediction"],
        errors="coerce",
    )
)

residual_sigma = float(
    np.sqrt(
        np.mean(
            np.square(
                oof_residual
            )
        )
    )
)

if (
    not math.isfinite(
        residual_sigma
    )
    or residual_sigma <= 0
):
    fail(
        "invalid OOF residual sigma"
    )

print(
    f"OOF_GAMES={len(oof)}"
)
print(
    f"OOF_RESIDUAL_SIGMA={residual_sigma:.6f}"
)

# Raw margin probability:
# P(home margin > 0) under Normal residual assumption.
oof[
    "raw_margin_probability"
] = normal_cdf_array(
    oof[
        "oof_margin_prediction"
    ].to_numpy()
    / residual_sigma
)

# Ties have target_win=NULL and are not probability-training rows.
oof_win = oof[
    oof["target_win"].notna()
].copy()

if len(oof_win) < 500:
    fail(
        "unexpectedly small OOF win set"
    )

# Calibrator uses ONLY OOF predicted margin.
calibrator = LogisticRegression(
    C=1.0,
    max_iter=5000,
    solver="lbfgs",
)

calibrator.fit(
    oof_win[
        ["oof_margin_prediction"]
    ],
    pd.to_numeric(
        oof_win["target_win"],
        errors="coerce",
    ).astype(int),
)

print(
    "CALIBRATION_SOURCE="
    "DEVELOPMENT_OOF_ONLY"
)
print(
    "CALIBRATION_USED_2025=FALSE"
)

# ------------------------------------------------------------
# Final selected margin model:
# all 2023+2024 -> 2025.
# ------------------------------------------------------------

final_margin_model = make_ridge(
    selected_alpha
)

final_margin_model.fit(
    dev[features],
    pd.to_numeric(
        dev["target_margin"],
        errors="coerce",
    ),
)

holdout_margin_pred = (
    final_margin_model.predict(
        test[features]
    )
)

raw_margin_prob = normal_cdf_array(
    holdout_margin_pred
    / residual_sigma
)

calibrated_margin_prob = (
    calibrator.predict_proba(
        pd.DataFrame(
            {
                "oof_margin_prediction":
                    holdout_margin_pred
            }
        )
    )[:, 1]
)

# ------------------------------------------------------------
# EXP001 logistic architecture, rebuilt independently.
# ------------------------------------------------------------

logistic = make_logistic()

dev_win_mask = (
    dev["target_win"].notna()
)

logistic.fit(
    dev.loc[
        dev_win_mask,
        features,
    ],
    pd.to_numeric(
        dev.loc[
            dev_win_mask,
            "target_win",
        ],
        errors="coerce",
    ).astype(int),
)

exp001_logistic_prob = (
    logistic.predict_proba(
        test[features]
    )[:, 1]
)

# ------------------------------------------------------------
# EXP001 alpha=10 margin architecture rebuilt for comparison.
# ------------------------------------------------------------

exp001_margin_model = make_ridge(
    10.0
)

exp001_margin_model.fit(
    dev[features],
    pd.to_numeric(
        dev["target_margin"],
        errors="coerce",
    ),
)

exp001_margin_pred = (
    exp001_margin_model.predict(
        test[features]
    )
)

# ------------------------------------------------------------
# Holdout metrics.
# 2025 is evaluated now, after all selections/calibration.
# ------------------------------------------------------------

print(
    "\n===== UNTOUCHED 2025 HOLDOUT ====="
)

actual_margin = pd.to_numeric(
    test["target_margin"],
    errors="coerce",
)

holdout_win_mask = (
    test["target_win"].notna()
)

actual_win = pd.to_numeric(
    test.loc[
        holdout_win_mask,
        "target_win",
    ],
    errors="coerce",
).astype(int)

prob_mask_np = (
    holdout_win_mask.to_numpy()
)

margin_metrics = {
    "exp001_alpha10": {
        "mae": mae(
            actual_margin,
            exp001_margin_pred,
        ),
        "rmse": rmse(
            actual_margin,
            exp001_margin_pred,
        ),
    },
    "exp002_selected_alpha": {
        "alpha":
            selected_alpha,
        "mae": mae(
            actual_margin,
            holdout_margin_pred,
        ),
        "rmse": rmse(
            actual_margin,
            holdout_margin_pred,
        ),
    },
}

probability_results = {
    "exp001_logistic": (
        probability_metrics(
            actual_win,
            exp001_logistic_prob[
                prob_mask_np
            ],
        )
    ),
    "exp002_raw_margin_normal": (
        probability_metrics(
            actual_win,
            raw_margin_prob[
                prob_mask_np
            ],
        )
    ),
    "exp002_calibrated_margin": (
        probability_metrics(
            actual_win,
            calibrated_margin_prob[
                prob_mask_np
            ],
        )
    ),
}

for name, m in margin_metrics.items():
    print(
        f"\nMARGIN_MODEL={name}"
    )

    if "alpha" in m:
        print(
            f"alpha={m['alpha']}"
        )

    print(
        f"margin_mae={m['mae']:.6f}"
    )
    print(
        f"margin_rmse={m['rmse']:.6f}"
    )

for name, m in probability_results.items():
    print(
        f"\nPROBABILITY_MODEL={name}"
    )
    print(
        f"accuracy={m['accuracy']:.6f}"
    )
    print(
        f"brier={m['brier']:.6f}"
    )
    print(
        f"logloss={m['logloss']:.6f}"
    )
    print(
        f"ece={m['ece']:.6f}"
    )

# ------------------------------------------------------------
# Totals diagnostic ONLY.
# Preserve EXP001 alpha=10 architecture.
# ------------------------------------------------------------

total_model = make_ridge(
    10.0
)

total_model.fit(
    dev[features],
    pd.to_numeric(
        dev["target_total_points"],
        errors="coerce",
    ),
)

holdout_total_pred = (
    total_model.predict(
        test[features]
    )
)

actual_total = pd.to_numeric(
    test["target_total_points"],
    errors="coerce",
)

train_total_mean = float(
    pd.to_numeric(
        dev["target_total_points"],
        errors="coerce",
    ).mean()
)

baseline_total_pred = np.full(
    len(test),
    train_total_mean,
)

total_metrics = {
    "baseline": {
        "mae": mae(
            actual_total,
            baseline_total_pred,
        ),
        "rmse": rmse(
            actual_total,
            baseline_total_pred,
        ),
    },
    "exp001_alpha10": {
        "mae": mae(
            actual_total,
            holdout_total_pred,
        ),
        "rmse": rmse(
            actual_total,
            holdout_total_pred,
        ),
    },
}

print(
    "\n===== TOTALS DIAGNOSTIC — NO ARCHITECTURE CHANGE ====="
)

for name, m in total_metrics.items():
    print(
        f"TOTAL_MODEL={name}|"
        f"MAE={m['mae']:.6f}|"
        f"RMSE={m['rmse']:.6f}"
    )

# ------------------------------------------------------------
# Confidence bucket diagnostic for calibrated probability.
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
    "exp001_margin_prediction"
] = exp001_margin_pred

pred[
    "exp002_margin_prediction"
] = holdout_margin_pred

pred[
    "exp001_logistic_home_win_probability"
] = exp001_logistic_prob

pred[
    "exp002_raw_margin_home_win_probability"
] = raw_margin_prob

pred[
    "exp002_calibrated_home_win_probability"
] = calibrated_margin_prob

pred[
    "exp001_total_prediction"
] = holdout_total_pred

pred[
    "exp002_implied_home_points"
] = (
    pred[
        "exp001_total_prediction"
    ]
    + pred[
        "exp002_margin_prediction"
    ]
) / 2.0

pred[
    "exp002_implied_away_points"
] = (
    pred[
        "exp001_total_prediction"
    ]
    - pred[
        "exp002_margin_prediction"
    ]
) / 2.0

bucket_source = pred[
    pred["actual_home_win"].notna()
].copy()

bucket_source[
    "probability_bucket"
] = pd.cut(
    bucket_source[
        "exp002_calibrated_home_win_probability"
    ],
    bins=[
        0.0,
        0.35,
        0.45,
        0.55,
        0.65,
        1.0,
    ],
    labels=[
        "0.00-0.35",
        "0.35-0.45",
        "0.45-0.55",
        "0.55-0.65",
        "0.65-1.00",
    ],
    include_lowest=True,
)

bucket_rows = []

for bucket, g in bucket_source.groupby(
    "probability_bucket",
    observed=False,
):
    if len(g) == 0:
        continue

    bucket_rows.append(
        {
            "bucket": str(bucket),
            "games": int(len(g)),
            "mean_predicted_home_win_probability":
                float(
                    g[
                        "exp002_calibrated_home_win_probability"
                    ].mean()
                ),
            "actual_home_win_rate":
                float(
                    pd.to_numeric(
                        g["actual_home_win"],
                        errors="coerce",
                    ).mean()
                ),
            "absolute_calibration_gap":
                abs(
                    float(
                        g[
                            "exp002_calibrated_home_win_probability"
                        ].mean()
                    )
                    - float(
                        pd.to_numeric(
                            g["actual_home_win"],
                            errors="coerce",
                        ).mean()
                    )
                ),
        }
    )

buckets_df = pd.DataFrame(
    bucket_rows
)

print(
    "\n===== CALIBRATED PROBABILITY BUCKETS ====="
)

if len(buckets_df):
    print(
        buckets_df.to_string(
            index=False
        )
    )

# ------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

pred.to_csv(
    PREDICTIONS,
    index=False,
)

buckets_df.to_csv(
    BUCKETS,
    index=False,
)

audit = {
    "status":
        "PASS",
    "experiment":
        "SIMULATOR_V1_EXPERIMENT_002",
    "source": {
        "path":
            str(SOURCE),
        "sha256":
            source_sha,
    },
    "exp001_preservation": {
        "audit_sha256_before":
            exp001_audit_sha_before,
        "predictions_sha256_before":
            exp001_pred_sha_before,
    },
    "contract": {
        "development_seasons":
            sorted(DEV_SEASONS),
        "holdout_season":
            HOLDOUT_SEASON,
        "development_games":
            int(len(dev)),
        "holdout_games":
            int(len(test)),
        "feature_count":
            int(len(features)),
        "market_predictors_used":
            [],
        "target_predictors_used":
            [],
        "holdout_used_for_alpha_selection":
            False,
        "holdout_used_for_calibration":
            False,
    },
    "ridge_selection": {
        "candidate_alphas":
            ALPHAS,
        "selection_fold":
            "train_2023_validate_2024",
        "selection_metric":
            "RMSE",
        "results":
            alpha_results,
        "selected_alpha":
            selected_alpha,
    },
    "calibration": {
        "method":
            "logistic calibration on OOF predicted margin",
        "oof_design":
            [
                "train_2023_predict_2024",
                "train_2024_predict_2023",
            ],
        "oof_games":
            int(len(oof)),
        "oof_residual_sigma":
            residual_sigma,
        "holdout_used":
            False,
    },
    "holdout_2025": {
        "margin":
            margin_metrics,
        "probability":
            probability_results,
        "totals_diagnostic":
            total_metrics,
    },
    "production": {
        "production_forecast_modified":
            False,
        "optimizer_modified":
            False,
        "production_database_written":
            False,
    },
    "outputs": {
        "predictions":
            str(PREDICTIONS),
        "probability_buckets":
            str(BUCKETS),
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
# EXP001 immutability check
# ------------------------------------------------------------

exp001_audit_sha_after = sha256(
    EXP001_AUDIT
)

exp001_pred_sha_after = sha256(
    EXP001_PREDICTIONS
)

if (
    exp001_audit_sha_after
    != exp001_audit_sha_before
):
    fail(
        "EXP001 audit artifact changed"
    )

if (
    exp001_pred_sha_after
    != exp001_pred_sha_before
):
    fail(
        "EXP001 prediction artifact changed"
    )

print(
    "\n===== EXP001 IMMUTABILITY ====="
)
print(
    "EXP001_AUDIT_UNCHANGED=TRUE"
)
print(
    "EXP001_PREDICTIONS_UNCHANGED=TRUE"
)

print(
    "\n===== OUTPUT ====="
)
print(
    f"PREDICTIONS={PREDICTIONS}"
)
print(
    f"BUCKETS={BUCKETS}"
)
print(
    f"AUDIT={AUDIT}"
)
print(
    f"PREDICTIONS_SHA256={sha256(PREDICTIONS)}"
)
print(
    f"BUCKETS_SHA256={sha256(BUCKETS)}"
)
print(
    f"AUDIT_SHA256={sha256(AUDIT)}"
)

print(
    "\nMARKET_PREDICTORS_USED=0"
)
print(
    "HOLDOUT_USED_FOR_SELECTION=FALSE"
)
print(
    "HOLDOUT_USED_FOR_CALIBRATION=FALSE"
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
    "SIMULATOR_V1_EXPERIMENT_002=PASS"
)
