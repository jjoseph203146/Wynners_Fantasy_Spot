#!/usr/bin/env python3

"""
WFS NFL SEASON & PLAYOFF SIMULATOR
Simulator V1 — Experiment 003

Purpose
-------
A. Extend margin regularization beyond EXP002's boundary alpha=1000.
B. Preserve EXP002 probability architecture:
      Ridge margin -> development OOF residual sigma -> Normal CDF.
C. Test a target-specific totals feature architecture.
D. Keep 2025 completely outside model/feature/hyperparameter selection.

No market predictors.
No target predictors.
No production writes.
EXP001/EXP002 artifacts remain untouched.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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

SOURCE = ROOT / "processed" / "forecast_v1_team_game_training.csv"
OUTDIR = ROOT / "processed" / "simulator_v1"

EXP001_AUDIT = OUTDIR / "simulator_v1_experiment_001_audit.json"
EXP001_PRED = OUTDIR / "simulator_v1_experiment_001_predictions.csv"

EXP002_AUDIT = OUTDIR / "simulator_v1_experiment_002_audit.json"
EXP002_PRED = OUTDIR / "simulator_v1_experiment_002_predictions.csv"
EXP002_BUCKETS = OUTDIR / "simulator_v1_experiment_002_probability_buckets.csv"

PREDICTIONS = OUTDIR / "simulator_v1_experiment_003_predictions.csv"
AUDIT = OUTDIR / "simulator_v1_experiment_003_audit.json"
BUCKETS = OUTDIR / "simulator_v1_experiment_003_probability_buckets.csv"

DEV_SEASONS = {2023, 2024}
HOLDOUT_SEASON = 2025

# EXP002 landed at the edge of its search grid.
MARGIN_ALPHAS = [
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
]

TOTAL_ALPHAS = [
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def ece(y_true, prob, bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)

    total = len(y)

    if total == 0:
        return float("nan")

    result = 0.0

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

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

    return np.asarray(
        [
            0.5 * (
                1.0
                + math.erf(
                    float(v) / math.sqrt(2.0)
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
                SimpleImputer(strategy="median"),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=float(alpha)),
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

    return {
        "accuracy": float(
            accuracy_score(
                y,
                (p >= 0.5).astype(int),
            )
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
        "logloss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
        "ece": ece(y, p),
    }


def fit_predict_ridge(train, valid, features, target, alpha):
    model = make_ridge(alpha)

    model.fit(
        train[features],
        pd.to_numeric(
            train[target],
            errors="coerce",
        ),
    )

    return model.predict(
        valid[features]
    )


print(
    "===== WFS SIMULATOR V1 — EXPERIMENT 003 ====="
)

# ============================================================
# PRESERVATION CONTRACT
# ============================================================

required_artifacts = [
    SOURCE,
    EXP001_AUDIT,
    EXP001_PRED,
    EXP002_AUDIT,
    EXP002_PRED,
    EXP002_BUCKETS,
]

for path in required_artifacts:
    if not path.exists():
        fail(f"missing required artifact: {path}")

preserve_before = {
    str(path): sha256(path)
    for path in required_artifacts
}

print(f"SOURCE_SHA256={preserve_before[str(SOURCE)]}")
print(f"EXP001_AUDIT_SHA256_BEFORE={preserve_before[str(EXP001_AUDIT)]}")
print(f"EXP001_PRED_SHA256_BEFORE={preserve_before[str(EXP001_PRED)]}")
print(f"EXP002_AUDIT_SHA256_BEFORE={preserve_before[str(EXP002_AUDIT)]}")
print(f"EXP002_PRED_SHA256_BEFORE={preserve_before[str(EXP002_PRED)]}")
print(f"EXP002_BUCKETS_SHA256_BEFORE={preserve_before[str(EXP002_BUCKETS)]}")

# ============================================================
# LOAD / GRAIN
# ============================================================

df = pd.read_csv(SOURCE)

home = (
    df[
        pd.to_numeric(
            df["is_home"],
            errors="coerce",
        ) == 1
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
    fail(f"expected 855 home-game rows, got {len(home)}")

if home["game_id"].duplicated().any():
    fail("duplicate game_id after home-row collapse")

dev = home[
    home["season"].isin(DEV_SEASONS)
].copy()

test = home[
    home["season"] == HOLDOUT_SEASON
].copy()

train23 = dev[
    dev["season"] == 2023
].copy()

valid24 = dev[
    dev["season"] == 2024
].copy()

if len(dev) != 570:
    fail(f"development games expected=570 actual={len(dev)}")

if len(test) != 285:
    fail(f"holdout games expected=285 actual={len(test)}")

if len(train23) != 285 or len(valid24) != 285:
    fail("2023/2024 development split invalid")

print(f"DEVELOPMENT_GAMES={len(dev)}")
print(f"HOLDOUT_GAMES={len(test)}")

# ============================================================
# EXP001/EXP002 NUMERIC FEATURE CONTRACT
# ============================================================

excluded = (
    MARKET_COLUMNS
    | TARGET_COLUMNS
    | IDENTITY_COLUMNS
    | DIRECT_IDENTITY_EXCLUSIONS
    | DEFERRED_COLUMNS
)

all_features = []

for col in home.columns:
    if col in excluded:
        continue

    if not pd.api.types.is_numeric_dtype(home[col]):
        continue

    x = pd.to_numeric(
        dev[col],
        errors="coerce",
    )

    if x.notna().sum() == 0:
        continue

    if x.dropna().nunique() <= 1:
        continue

    all_features.append(col)

all_features = sorted(all_features)

if len(all_features) != 92:
    fail(
        "feature-contract drift | "
        f"expected=92 actual={len(all_features)}"
    )

for col in all_features:
    low = col.lower()

    if (
        col in MARKET_COLUMNS
        or col in TARGET_COLUMNS
        or "moneyline" in low
        or "market_" in low
    ):
        fail(f"leakage feature escaped: {col}")

print(f"FULL_FEATURE_COUNT={len(all_features)}")
print("MARKET_PREDICTORS_USED=0")
print("TARGET_PREDICTORS_USED=0")

# ============================================================
# TARGET-SPECIFIC TOTAL FEATURE CONTRACT
#
# EXP003 deliberately excludes:
# - injury/status aggregates
# - coach history
# - rest
# - history-count bookkeeping
#
# It retains direct prior scoring, defense, play volume,
# passing/rushing production and pace/style signals.
# ============================================================

TOTAL_ALLOWED_TOKENS = (
    "points_for_",
    "points_against_",
    "offensive_plays_",
    "pass_attempts_",
    "rush_attempts_",
    "pass_rate_",
    "rush_rate_",
    "passing_yards_",
    "rushing_yards_",
    "passing_tds_",
    "rushing_tds_",
    "opponent_points_allowed_",
    "opponent_pass_yards_allowed_",
    "opponent_rush_yards_allowed_",
    "opponent_pass_tds_allowed_",
    "opponent_rush_tds_allowed_",
    "scoring_trend",
    "pace_trend",
)

total_features = []

for col in all_features:
    low = col.lower()

    if any(
        token in low
        for token in TOTAL_ALLOWED_TOKENS
    ):
        total_features.append(col)

total_features = sorted(set(total_features))

if len(total_features) < 20:
    fail(
        "target-specific total feature set unexpectedly small | "
        f"count={len(total_features)}"
    )

print(
    f"TOTAL_TARGET_SPECIFIC_FEATURE_COUNT={len(total_features)}"
)

print("\n===== TOTAL TARGET-SPECIFIC FEATURES =====")

for col in total_features:
    print(f"TOTAL_ALLOW|{col}")

# ============================================================
# MARGIN ALPHA SELECTION — DEVELOPMENT ONLY
# ============================================================

print(
    "\n===== MARGIN REGULARIZATION — TRAIN 2023 / VALIDATE 2024 ====="
)

margin_selection = []

for alpha in MARGIN_ALPHAS:
    pred24 = fit_predict_ridge(
        train23,
        valid24,
        all_features,
        "target_margin",
        alpha,
    )

    row = {
        "alpha": float(alpha),
        "mae": mae(
            valid24["target_margin"],
            pred24,
        ),
        "rmse": rmse(
            valid24["target_margin"],
            pred24,
        ),
    }

    margin_selection.append(row)

    print(
        f"ALPHA={alpha:10.1f}|"
        f"MAE={row['mae']:.6f}|"
        f"RMSE={row['rmse']:.6f}"
    )

best_margin = min(
    margin_selection,
    key=lambda r: (
        r["rmse"],
        r["mae"],
        r["alpha"],
    ),
)

selected_margin_alpha = float(
    best_margin["alpha"]
)

print(
    f"SELECTED_MARGIN_ALPHA={selected_margin_alpha}"
)
print(
    "MARGIN_SELECTION_USED_2025=FALSE"
)

# ============================================================
# TOTAL ALPHA SELECTION — DEVELOPMENT ONLY
# ============================================================

print(
    "\n===== TOTAL REGULARIZATION — TRAIN 2023 / VALIDATE 2024 ====="
)

total_selection = []

for alpha in TOTAL_ALPHAS:
    pred24 = fit_predict_ridge(
        train23,
        valid24,
        total_features,
        "target_total_points",
        alpha,
    )

    row = {
        "alpha": float(alpha),
        "mae": mae(
            valid24["target_total_points"],
            pred24,
        ),
        "rmse": rmse(
            valid24["target_total_points"],
            pred24,
        ),
    }

    total_selection.append(row)

    print(
        f"ALPHA={alpha:10.1f}|"
        f"MAE={row['mae']:.6f}|"
        f"RMSE={row['rmse']:.6f}"
    )

best_total = min(
    total_selection,
    key=lambda r: (
        r["rmse"],
        r["mae"],
        r["alpha"],
    ),
)

selected_total_alpha = float(
    best_total["alpha"]
)

print(
    f"SELECTED_TOTAL_ALPHA={selected_total_alpha}"
)
print(
    "TOTAL_SELECTION_USED_2025=FALSE"
)

# ============================================================
# MARGIN OOF RESIDUAL SIGMA
#
# Preserve EXP002 architecture:
# 2023 -> 2024
# 2024 -> 2023
# ============================================================

print(
    "\n===== EXP003 DEVELOPMENT OOF MARGIN SCALE ====="
)

oof_parts = []

for train_season, pred_season in [
    (2023, 2024),
    (2024, 2023),
]:
    tr = dev[
        dev["season"] == train_season
    ].copy()

    va = dev[
        dev["season"] == pred_season
    ].copy()

    p = fit_predict_ridge(
        tr,
        va,
        all_features,
        "target_margin",
        selected_margin_alpha,
    )

    part = va[
        [
            "game_id",
            "season",
            "target_margin",
            "target_win",
        ]
    ].copy()

    part["pred_margin"] = p

    oof_parts.append(part)

oof = pd.concat(
    oof_parts,
    ignore_index=True,
)

if len(oof) != 570:
    fail(f"OOF games expected=570 actual={len(oof)}")

if oof["game_id"].duplicated().any():
    fail("duplicate OOF game")

oof_residual = (
    pd.to_numeric(
        oof["target_margin"],
        errors="coerce",
    )
    - pd.to_numeric(
        oof["pred_margin"],
        errors="coerce",
    )
)

residual_sigma = float(
    np.sqrt(
        np.mean(
            np.square(oof_residual)
        )
    )
)

if (
    not math.isfinite(residual_sigma)
    or residual_sigma <= 0
):
    fail("invalid OOF residual sigma")

print(f"OOF_GAMES={len(oof)}")
print(f"OOF_RESIDUAL_SIGMA={residual_sigma:.6f}")
print("PROBABILITY_ARCHITECTURE=RIDGE_MARGIN_NORMAL_CDF")
print("EXPLICIT_CALIBRATOR_USED=FALSE")
print("PROBABILITY_SCALE_USED_2025=FALSE")

# ============================================================
# FINAL MODELS — ONLY NOW FIT ALL 2023+2024
# ============================================================

margin_model = make_ridge(
    selected_margin_alpha
)

margin_model.fit(
    dev[all_features],
    pd.to_numeric(
        dev["target_margin"],
        errors="coerce",
    ),
)

pred_margin = margin_model.predict(
    test[all_features]
)

pred_probability = normal_cdf_array(
    pred_margin / residual_sigma
)

total_model = make_ridge(
    selected_total_alpha
)

total_model.fit(
    dev[total_features],
    pd.to_numeric(
        dev["target_total_points"],
        errors="coerce",
    ),
)

pred_total = total_model.predict(
    test[total_features]
)

# ============================================================
# HOLDOUT — EVALUATED ONLY AFTER SELECTION
# ============================================================

actual_margin = pd.to_numeric(
    test["target_margin"],
    errors="coerce",
)

actual_total = pd.to_numeric(
    test["target_total_points"],
    errors="coerce",
)

win_mask = test["target_win"].notna()
win_mask_np = win_mask.to_numpy()

actual_win = pd.to_numeric(
    test.loc[
        win_mask,
        "target_win",
    ],
    errors="coerce",
).astype(int)

exp003_margin_metrics = {
    "mae": mae(
        actual_margin,
        pred_margin,
    ),
    "rmse": rmse(
        actual_margin,
        pred_margin,
    ),
}

exp003_probability_metrics = probability_metrics(
    actual_win,
    pred_probability[
        win_mask_np
    ],
)

train_total_mean = float(
    pd.to_numeric(
        dev["target_total_points"],
        errors="coerce",
    ).mean()
)

baseline_total = np.full(
    len(test),
    train_total_mean,
)

baseline_total_metrics = {
    "mae": mae(
        actual_total,
        baseline_total,
    ),
    "rmse": rmse(
        actual_total,
        baseline_total,
    ),
}

exp003_total_metrics = {
    "mae": mae(
        actual_total,
        pred_total,
    ),
    "rmse": rmse(
        actual_total,
        pred_total,
    ),
}

print(
    "\n===== UNTOUCHED 2025 HOLDOUT ====="
)

print(
    f"MARGIN_ALPHA={selected_margin_alpha}"
)
print(
    f"MARGIN_MAE={exp003_margin_metrics['mae']:.6f}"
)
print(
    f"MARGIN_RMSE={exp003_margin_metrics['rmse']:.6f}"
)

print(
    f"WIN_ACCURACY={exp003_probability_metrics['accuracy']:.6f}"
)
print(
    f"WIN_BRIER={exp003_probability_metrics['brier']:.6f}"
)
print(
    f"WIN_LOGLOSS={exp003_probability_metrics['logloss']:.6f}"
)
print(
    f"WIN_ECE={exp003_probability_metrics['ece']:.6f}"
)

print(
    f"TOTAL_ALPHA={selected_total_alpha}"
)
print(
    f"TOTAL_BASELINE_MAE={baseline_total_metrics['mae']:.6f}"
)
print(
    f"TOTAL_EXP003_MAE={exp003_total_metrics['mae']:.6f}"
)
print(
    f"TOTAL_MAE_DELTA="
    f"{exp003_total_metrics['mae'] - baseline_total_metrics['mae']:+.6f}"
)
print(
    f"TOTAL_BASELINE_RMSE={baseline_total_metrics['rmse']:.6f}"
)
print(
    f"TOTAL_EXP003_RMSE={exp003_total_metrics['rmse']:.6f}"
)
print(
    f"TOTAL_RMSE_DELTA="
    f"{exp003_total_metrics['rmse'] - baseline_total_metrics['rmse']:+.6f}"
)

# ============================================================
# EXP002 REFERENCE METRICS
# ============================================================

with EXP002_AUDIT.open() as f:
    exp002_audit = json.load(f)

exp002_holdout = exp002_audit[
    "holdout_2025"
]

exp002_margin = exp002_holdout[
    "margin"
]["exp002_selected_alpha"]

exp002_prob = exp002_holdout[
    "probability"
]["exp002_raw_margin_normal"]

print(
    "\n===== EXP003 VS EXP002 ====="
)

print(
    f"MARGIN_MAE_DELTA_VS_EXP002="
    f"{exp003_margin_metrics['mae'] - float(exp002_margin['mae']):+.6f}"
)

print(
    f"MARGIN_RMSE_DELTA_VS_EXP002="
    f"{exp003_margin_metrics['rmse'] - float(exp002_margin['rmse']):+.6f}"
)

print(
    f"WIN_ACCURACY_DELTA_VS_EXP002="
    f"{exp003_probability_metrics['accuracy'] - float(exp002_prob['accuracy']):+.6f}"
)

print(
    f"WIN_BRIER_DELTA_VS_EXP002="
    f"{exp003_probability_metrics['brier'] - float(exp002_prob['brier']):+.6f}"
)

print(
    f"WIN_LOGLOSS_DELTA_VS_EXP002="
    f"{exp003_probability_metrics['logloss'] - float(exp002_prob['logloss']):+.6f}"
)

print(
    f"WIN_ECE_DELTA_VS_EXP002="
    f"{exp003_probability_metrics['ece'] - float(exp002_prob['ece']):+.6f}"
)

# ============================================================
# PREDICTIONS
# ============================================================

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
        "target_team_points": "actual_home_points",
        "target_opponent_points": "actual_away_points",
        "target_margin": "actual_home_margin",
        "target_total_points": "actual_total_points",
        "target_win": "actual_home_win",
        "target_result": "actual_home_result",
    }
)

pred["pred_home_margin"] = pred_margin
pred["pred_total_points"] = pred_total
pred["pred_home_win_probability"] = pred_probability

pred["pred_home_points"] = (
    pred["pred_total_points"]
    + pred["pred_home_margin"]
) / 2.0

pred["pred_away_points"] = (
    pred["pred_total_points"]
    - pred["pred_home_margin"]
) / 2.0

pred["pred_winner"] = np.where(
    pred["pred_home_win_probability"] >= 0.5,
    pred["home_team"],
    pred["away_team"],
)

# ============================================================
# PROBABILITY BUCKETS
# ============================================================

bucket_source = pred[
    pred["actual_home_win"].notna()
].copy()

bucket_source["probability_bucket"] = pd.cut(
    bucket_source["pred_home_win_probability"],
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

    mean_prob = float(
        g["pred_home_win_probability"].mean()
    )

    actual_rate = float(
        pd.to_numeric(
            g["actual_home_win"],
            errors="coerce",
        ).mean()
    )

    bucket_rows.append(
        {
            "bucket": str(bucket),
            "games": int(len(g)),
            "mean_predicted_home_win_probability": mean_prob,
            "actual_home_win_rate": actual_rate,
            "absolute_calibration_gap": abs(
                mean_prob - actual_rate
            ),
        }
    )

buckets_df = pd.DataFrame(bucket_rows)

print(
    "\n===== EXP003 PROBABILITY BUCKETS ====="
)

if len(buckets_df):
    print(
        buckets_df.to_string(
            index=False
        )
    )

# ============================================================
# WRITE ISOLATED ARTIFACTS
# ============================================================

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
    "status": "PASS",
    "experiment": "SIMULATOR_V1_EXPERIMENT_003",
    "source": {
        "path": str(SOURCE),
        "sha256": preserve_before[str(SOURCE)],
    },
    "contract": {
        "development_seasons": sorted(DEV_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "development_games": int(len(dev)),
        "holdout_games": int(len(test)),
        "full_feature_count": int(len(all_features)),
        "total_target_specific_feature_count": int(
            len(total_features)
        ),
        "total_target_specific_features": total_features,
        "market_predictors_used": [],
        "target_predictors_used": [],
        "holdout_used_for_selection": False,
        "holdout_used_for_probability_scale": False,
    },
    "margin": {
        "candidate_alphas": MARGIN_ALPHAS,
        "selection_fold": "train_2023_validate_2024",
        "selection_metric": "RMSE",
        "selection_results": margin_selection,
        "selected_alpha": selected_margin_alpha,
        "oof_residual_sigma": residual_sigma,
        "probability_architecture":
            "ridge_margin_normal_cdf",
        "explicit_calibrator_used": False,
    },
    "totals": {
        "candidate_alphas": TOTAL_ALPHAS,
        "selection_fold": "train_2023_validate_2024",
        "selection_metric": "RMSE",
        "selection_results": total_selection,
        "selected_alpha": selected_total_alpha,
    },
    "holdout_2025": {
        "margin": exp003_margin_metrics,
        "probability": exp003_probability_metrics,
        "totals_baseline": baseline_total_metrics,
        "totals_exp003": exp003_total_metrics,
    },
    "production": {
        "production_forecast_modified": False,
        "optimizer_modified": False,
        "production_database_written": False,
    },
    "outputs": {
        "predictions": str(PREDICTIONS),
        "probability_buckets": str(BUCKETS),
    },
}

AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    ) + "\n"
)

# ============================================================
# IMMUTABILITY
# ============================================================

for path in required_artifacts:
    after = sha256(path)
    before = preserve_before[str(path)]

    if after != before:
        fail(
            "protected artifact changed | "
            f"{path}"
        )

print(
    "\n===== PRIOR ARTIFACT IMMUTABILITY ====="
)
print("SOURCE_UNCHANGED=TRUE")
print("EXP001_UNCHANGED=TRUE")
print("EXP002_UNCHANGED=TRUE")

print(
    "\n===== OUTPUT ====="
)
print(f"PREDICTIONS={PREDICTIONS}")
print(f"BUCKETS={BUCKETS}")
print(f"AUDIT={AUDIT}")
print(f"PREDICTIONS_SHA256={sha256(PREDICTIONS)}")
print(f"BUCKETS_SHA256={sha256(BUCKETS)}")
print(f"AUDIT_SHA256={sha256(AUDIT)}")

print()
print("MARKET_PREDICTORS_USED=0")
print("TARGET_PREDICTORS_USED=0")
print("HOLDOUT_USED_FOR_SELECTION=FALSE")
print("HOLDOUT_USED_FOR_PROBABILITY_SCALE=FALSE")
print("EXPLICIT_CALIBRATOR_USED=FALSE")
print("PRODUCTION_FORECAST_MODIFIED=FALSE")
print("OPTIMIZER_MODIFIED=FALSE")
print("PRODUCTION_DATABASE_WRITTEN=FALSE")
print("SIMULATOR_V1_EXPERIMENT_003=PASS")
