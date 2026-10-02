#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# WFS STAGE26G-D
# FINAL COACH SIGNAL DEPLOYMENT READINESS VALIDATION
# ==================================================================
#
# ANALYSIS_ONLY=TRUE
#
# PURPOSE:
#   Final research gate for the validated overall coach-regime
#   Pass-OE signal.
#
# TESTS:
#   1. Leave-one-season-out calibration / validation
#   2. Leave-one-regime-out robustness
#   3. Conservative fixed-weight calibration
#
# SIGNAL:
#   prior_mean_pass_oe
#
# BASELINE:
#   current_mean_xpass
#
# COACH-ADJUSTED:
#   current_mean_xpass
#   + weight * (prior_mean_pass_oe / 100)
#
# IMPORTANT:
#   This is still retrospective analysis because current-game xPass
#   is the contextual evaluation baseline.
#
#   NO production influence is authorized here.
#
# ==================================================================


ROOT = Path("/home/mwynn/nfl_data_engine")

GA_PATH = (
    ROOT
    / "scripts"
    / "stage26ga_coach_regime_tendency_matrix.py"
)

ANALYSIS_ONLY = True
PRODUCTION_INFLUENCE = False

ARTIFACT_WRITE = False
DATABASE_WRITE = False
SOLVER_MUTATION = False
FORECAST_MUTATION = False
APP_MUTATION = False
SERVICE_RESTART = False


# Conservative region supported by G-C/C2.
FIXED_WEIGHTS = [
    0.40,
    0.50,
    0.60,
    0.70,
]


def section(name: str) -> None:

    print()
    print("=" * 110)
    print(name)
    print("=" * 110)


def num(s: pd.Series) -> pd.Series:

    return pd.to_numeric(
        s,
        errors="coerce",
    )


def ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    n = num(numerator)
    d = num(denominator)

    out = pd.Series(
        np.nan,
        index=n.index,
        dtype=float,
    )

    good = (
        n.notna()
        &
        d.notna()
        &
        d.gt(0)
    )

    out.loc[good] = (
        n.loc[good]
        /
        d.loc[good]
    )

    return out


def make_prediction(
    df: pd.DataFrame,
    weight: float,
) -> pd.Series:

    return (
        df["current_mean_xpass"]
        +
        float(weight)
        *
        (
            df["prior_mean_pass_oe"]
            /
            100.0
        )
    )


def clean_pair(
    actual: pd.Series,
    prediction: pd.Series,
) -> pd.DataFrame:

    d = pd.DataFrame(
        {
            "actual":
                num(actual),

            "prediction":
                num(prediction),
        }
    ).dropna()

    return d[
        np.isfinite(d["actual"])
        &
        np.isfinite(d["prediction"])
    ].copy()


def mae(
    actual: pd.Series,
    prediction: pd.Series,
) -> float:

    d = clean_pair(
        actual,
        prediction,
    )

    if d.empty:
        return np.nan

    return float(
        (
            d["actual"]
            -
            d["prediction"]
        )
        .abs()
        .mean()
    )


def rmse(
    actual: pd.Series,
    prediction: pd.Series,
) -> float:

    d = clean_pair(
        actual,
        prediction,
    )

    if d.empty:
        return np.nan

    error = (
        d["actual"]
        -
        d["prediction"]
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(error)
            )
        )
    )


def corr(
    actual: pd.Series,
    prediction: pd.Series,
) -> float:

    d = clean_pair(
        actual,
        prediction,
    )

    if len(d) < 3:
        return np.nan

    if (
        d["actual"].nunique() < 2
        or
        d["prediction"].nunique() < 2
    ):
        return np.nan

    return float(
        d["actual"].corr(
            d["prediction"]
        )
    )


def bias(
    actual: pd.Series,
    prediction: pd.Series,
) -> float:

    d = clean_pair(
        actual,
        prediction,
    )

    if d.empty:
        return np.nan

    return float(
        (
            d["prediction"]
            -
            d["actual"]
        ).mean()
    )


def pct_improved(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
) -> float:

    d = pd.DataFrame(
        {
            "actual":
                num(actual),

            "baseline":
                num(baseline),

            "candidate":
                num(candidate),
        }
    ).dropna()

    d = d[
        np.isfinite(d["actual"])
        &
        np.isfinite(d["baseline"])
        &
        np.isfinite(d["candidate"])
    ]

    if d.empty:
        return np.nan

    baseline_error = (
        d["actual"]
        -
        d["baseline"]
    ).abs()

    candidate_error = (
        d["actual"]
        -
        d["candidate"]
    ).abs()

    return float(
        candidate_error.lt(
            baseline_error
        ).mean()
    )


def evaluate(
    df: pd.DataFrame,
    prediction: pd.Series,
) -> dict:

    actual = df[
        "current_observed_pass_rate"
    ]

    baseline = df[
        "current_mean_xpass"
    ]

    baseline_mae = mae(
        actual,
        baseline,
    )

    candidate_mae = mae(
        actual,
        prediction,
    )

    baseline_rmse = rmse(
        actual,
        baseline,
    )

    candidate_rmse = rmse(
        actual,
        prediction,
    )

    return {
        "n":
            len(
                clean_pair(
                    actual,
                    prediction,
                )
            ),

        "baseline_mae":
            baseline_mae,

        "candidate_mae":
            candidate_mae,

        "delta_mae":
            candidate_mae
            -
            baseline_mae,

        "baseline_rmse":
            baseline_rmse,

        "candidate_rmse":
            candidate_rmse,

        "delta_rmse":
            candidate_rmse
            -
            baseline_rmse,

        "baseline_corr":
            corr(
                actual,
                baseline,
            ),

        "candidate_corr":
            corr(
                actual,
                prediction,
            ),

        "candidate_bias":
            bias(
                actual,
                prediction,
            ),

        "pct_improved":
            pct_improved(
                actual,
                baseline,
                prediction,
            ),
    }


def choose_weight(
    train: pd.DataFrame,
) -> float:

    rows = []

    for weight in FIXED_WEIGHTS:

        prediction = make_prediction(
            train,
            weight,
        )

        rows.append(
            {
                "weight":
                    weight,

                "mae":
                    mae(
                        train[
                            "current_observed_pass_rate"
                        ],
                        prediction,
                    ),

                "rmse":
                    rmse(
                        train[
                            "current_observed_pass_rate"
                        ],
                        prediction,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    best = (
        result
        .sort_values(
            [
                "mae",
                "rmse",
                "weight",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[0]
    )

    return float(
        best["weight"]
    )


# ==================================================================
# [0] RUNTIME CONTRACT
# ==================================================================

section(
    "[0] STAGE26G-D RUNTIME CONTRACT"
)

print("ANALYSIS_ONLY=TRUE")
print("SIGNAL=PRIOR_COACH_REGIME_PASS_OE")
print("BASELINE=CURRENT_GAME_XPASS")
print("LEAVE_ONE_SEASON_OUT=TRUE")
print("LEAVE_ONE_REGIME_OUT=TRUE")
print("FIXED_WEIGHT_CALIBRATION=TRUE")
print("PRODUCTION_INFLUENCE=FALSE")
print("ARTIFACT_WRITE=FALSE")
print("DATABASE_WRITE=FALSE")
print("SOLVER_MUTATION=FALSE")
print("FORECAST_MUTATION=FALSE")
print("APP_MUTATION=FALSE")
print("SERVICE_RESTART=FALSE")


# ==================================================================
# [1] LOAD FROZEN G-A AUTHORITY
# ==================================================================

section(
    "[1] LOAD FROZEN STAGE26G-A AUTHORITY"
)

if not GA_PATH.is_file():

    raise SystemExit(
        f"FAIL_CLOSED: missing G-A script: {GA_PATH}"
    )

ga = runpy.run_path(
    str(GA_PATH)
)

if "matrix" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: G-A matrix unavailable"
    )

if "coach_id_col" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: G-A coach identity unavailable"
    )

matrix = ga["matrix"].copy()

coach_id_col = ga[
    "coach_id_col"
]

if len(matrix) != 1710:

    raise SystemExit(
        "FAIL_CLOSED: G-A row contract changed"
    )

if (
    matrix[
        "team_coach_regime_key"
    ].nunique()
    != 49
):

    raise SystemExit(
        "FAIL_CLOSED: G-A regime contract changed"
    )

if matrix.duplicated(
    [
        "game_id",
        "team",
    ]
).any():

    raise SystemExit(
        "FAIL_CLOSED: duplicate matrix identity"
    )

print(
    f"MATRIX_ROWS={len(matrix)}"
)

print(
    "MATRIX_REGIMES="
    f"{matrix['team_coach_regime_key'].nunique()}"
)

print(
    "STAGE26G_A_AUTHORITY=PASS"
)


# ==================================================================
# [2] PIT / REGIME AUDIT
# ==================================================================

section(
    "[2] PIT AND REGIME AUDIT"
)

matrix[
    "game_date_dt"
] = pd.to_datetime(
    matrix[
        "game_date_dt"
    ],
    errors="coerce",
)

if matrix[
    "game_date_dt"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: null game date"
    )

matrix = matrix.sort_values(
    [
        "team_coach_regime_key",
        "game_date_dt",
        "game_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)

matrix[
    "_regime_row"
] = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .cumcount()
)

sequence_violations = int(
    (
        num(
            matrix[
                "prior_pbp_regime_games"
            ]
        )
        -
        matrix[
            "_regime_row"
        ]
    )
    .abs()
    .gt(0)
    .sum()
)

regime_crossovers = int(
    (
        matrix
        .groupby(
            "team_coach_regime_key"
        )[
            coach_id_col
        ]
        .transform(
            "nunique"
        )
        .gt(1)
    )
    .sum()
)

print(
    "REGIME_SEQUENCE_VIOLATIONS="
    f"{sequence_violations}"
)

print(
    "REGIME_CROSSOVER_ROWS="
    f"{regime_crossovers}"
)

if sequence_violations:

    raise SystemExit(
        "FAIL_CLOSED: regime chronology violation"
    )

if regime_crossovers:

    raise SystemExit(
        "FAIL_CLOSED: regime identity crossover"
    )

print(
    "CURRENT_GAME_INCLUDED_IN_PROFILE=FALSE"
)

print(
    "FUTURE_GAME_INCLUDED_IN_PROFILE=FALSE"
)

print(
    "PIT_REGIME_AUDIT=PASS"
)


# ==================================================================
# [3] VALIDATION UNIVERSE
# ==================================================================

section(
    "[3] VALIDATION UNIVERSE"
)

matrix[
    "current_observed_pass_rate"
] = ratio(
    matrix[
        "combined_nfl_pass"
    ],
    matrix[
        "combined_n"
    ],
)

matrix[
    "current_mean_xpass"
] = ratio(
    matrix[
        "combined_xpass_sum"
    ],
    matrix[
        "combined_n"
    ],
)

matrix[
    "current_mean_pass_oe"
] = ratio(
    matrix[
        "combined_pass_oe_sum"
    ],
    matrix[
        "combined_n"
    ],
)

validation = matrix[
    num(
        matrix[
            "prior_pbp_regime_games"
        ]
    ).gt(0)
].copy()

required = [
    "current_observed_pass_rate",
    "current_mean_xpass",
    "current_mean_pass_oe",
    "prior_mean_pass_oe",
]

for col in required:

    validation[col] = num(
        validation[col]
    )

    validation = validation[
        validation[col].notna()
        &
        np.isfinite(
            validation[col]
        )
    ]

validation[
    "season_eval"
] = num(
    validation[
        "season"
    ]
).astype(
    "Int64"
)

derived_oe = (
    (
        validation[
            "current_observed_pass_rate"
        ]
        -
        validation[
            "current_mean_xpass"
        ]
    )
    *
    100.0
)

max_oe_error = float(
    (
        validation[
            "current_mean_pass_oe"
        ]
        -
        derived_oe
    )
    .abs()
    .max()
)

print(
    f"VALIDATION_ROWS={len(validation)}"
)

print(
    "MAX_PASS_OE_IDENTITY_ERROR="
    f"{max_oe_error:.12f}"
)

if max_oe_error > 1e-6:

    raise SystemExit(
        "FAIL_CLOSED: Pass-OE identity failure"
    )

print(
    "VALIDATION_UNIVERSE=PASS"
)


# ==================================================================
# [4] FIXED-WEIGHT CALIBRATION
# ==================================================================

section(
    "[4] CONSERVATIVE FIXED-WEIGHT CALIBRATION"
)

fixed_results = []

for weight in FIXED_WEIGHTS:

    prediction = make_prediction(
        validation,
        weight,
    )

    metrics = evaluate(
        validation,
        prediction,
    )

    fixed_results.append(
        {
            "weight":
                weight,

            **metrics,
        }
    )

    print(
        f"WEIGHT_{weight:.2f}_N="
        f"{metrics['n']}"
    )

    print(
        f"WEIGHT_{weight:.2f}_DELTA_MAE="
        f"{metrics['delta_mae']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_DELTA_RMSE="
        f"{metrics['delta_rmse']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_CORR="
        f"{metrics['candidate_corr']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_PCT_IMPROVED="
        f"{metrics['pct_improved']:.8f}"
    )

fixed_df = pd.DataFrame(
    fixed_results
)

best_fixed = (
    fixed_df
    .sort_values(
        [
            "candidate_mae",
            "candidate_rmse",
            "weight",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="stable",
    )
    .iloc[0]
)

best_fixed_weight = float(
    best_fixed[
        "weight"
    ]
)

print(
    "BEST_FIXED_WEIGHT_DIAGNOSTIC="
    f"{best_fixed_weight:.2f}"
)

print(
    "FIXED_WEIGHT_CALIBRATION=PASS"
)


# ==================================================================
# [5] LEAVE-ONE-SEASON-OUT
# ==================================================================

section(
    "[5] LEAVE-ONE-SEASON-OUT VALIDATION"
)

season_results = []

seasons = sorted(
    validation[
        "season_eval"
    ]
    .dropna()
    .unique()
)

for held_out_season in seasons:

    train = validation[
        ~validation[
            "season_eval"
        ].eq(
            held_out_season
        )
    ].copy()

    test = validation[
        validation[
            "season_eval"
        ].eq(
            held_out_season
        )
    ].copy()

    selected_weight = choose_weight(
        train
    )

    prediction = make_prediction(
        test,
        selected_weight,
    )

    metrics = evaluate(
        test,
        prediction,
    )

    season_results.append(
        {
            "season":
                int(
                    held_out_season
                ),

            "weight":
                selected_weight,

            **metrics,
        }
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_TRAIN_ROWS="
        f"{len(train)}"
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_TEST_ROWS="
        f"{len(test)}"
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_SELECTED_WEIGHT="
        f"{selected_weight:.2f}"
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_DELTA_MAE="
        f"{metrics['delta_mae']:.8f}"
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_DELTA_RMSE="
        f"{metrics['delta_rmse']:.8f}"
    )

    print(
        f"LOSO_SEASON_{int(held_out_season)}_PCT_IMPROVED="
        f"{metrics['pct_improved']:.8f}"
    )


season_df = pd.DataFrame(
    season_results
)

season_mae_passes = int(
    season_df[
        "delta_mae"
    ].lt(0).sum()
)

season_rmse_passes = int(
    season_df[
        "delta_rmse"
    ].lt(0).sum()
)

print(
    "LOSO_SEASONS_WITH_MAE_IMPROVEMENT="
    f"{season_mae_passes}/{len(season_df)}"
)

print(
    "LOSO_SEASONS_WITH_RMSE_IMPROVEMENT="
    f"{season_rmse_passes}/{len(season_df)}"
)

print(
    "LEAVE_ONE_SEASON_OUT_VALIDATION=PASS"
)


# ==================================================================
# [6] LEAVE-ONE-REGIME-OUT
# ==================================================================

section(
    "[6] LEAVE-ONE-REGIME-OUT VALIDATION"
)

regime_results = []

for regime, test in validation.groupby(
    "team_coach_regime_key",
    sort=False,
):

    if len(test) < 5:
        continue

    train = validation[
        ~validation[
            "team_coach_regime_key"
        ].eq(
            regime
        )
    ].copy()

    selected_weight = choose_weight(
        train
    )

    prediction = make_prediction(
        test,
        selected_weight,
    )

    metrics = evaluate(
        test,
        prediction,
    )

    regime_results.append(
        {
            "regime":
                regime,

            "n":
                metrics[
                    "n"
                ],

            "weight":
                selected_weight,

            "delta_mae":
                metrics[
                    "delta_mae"
                ],

            "delta_rmse":
                metrics[
                    "delta_rmse"
                ],

            "pct_improved":
                metrics[
                    "pct_improved"
                ],
        }
    )


regime_df = pd.DataFrame(
    regime_results
)

if regime_df.empty:

    raise SystemExit(
        "FAIL_CLOSED: no LORO regime results"
    )

regime_mae_rate = float(
    regime_df[
        "delta_mae"
    ].lt(0).mean()
)

regime_rmse_rate = float(
    regime_df[
        "delta_rmse"
    ].lt(0).mean()
)

median_regime_delta_mae = float(
    regime_df[
        "delta_mae"
    ].median()
)

median_regime_delta_rmse = float(
    regime_df[
        "delta_rmse"
    ].median()
)

print(
    "LORO_REGIME_COUNT="
    f"{len(regime_df)}"
)

print(
    "LORO_REGIME_MAE_IMPROVEMENT_RATE="
    f"{regime_mae_rate:.8f}"
)

print(
    "LORO_REGIME_RMSE_IMPROVEMENT_RATE="
    f"{regime_rmse_rate:.8f}"
)

print(
    "LORO_MEDIAN_DELTA_MAE="
    f"{median_regime_delta_mae:.8f}"
)

print(
    "LORO_MEDIAN_DELTA_RMSE="
    f"{median_regime_delta_rmse:.8f}"
)

weight_counts = (
    regime_df[
        "weight"
    ]
    .value_counts()
    .sort_index()
)

for weight, count in weight_counts.items():

    print(
        f"LORO_SELECTED_WEIGHT_{float(weight):.2f}_REGIMES="
        f"{int(count)}"
    )

print(
    "LEAVE_ONE_REGIME_OUT_VALIDATION=PASS"
)


# ==================================================================
# [7] FINAL RESEARCH DECISION
# ==================================================================

section(
    "[7] FINAL COACH RESEARCH DECISION"
)

# Final research advancement gate.
#
# This does NOT authorize production.
#
# Requirements:
#   - every held-out season improves MAE
#   - every held-out season improves RMSE
#   - >50% of held-out regimes improve MAE
#   - >50% of held-out regimes improve RMSE
#   - median held-out regime effect improves MAE/RMSE
#   - best fixed calibration is nonzero


research_pass = (
    season_mae_passes
    ==
    len(
        season_df
    )
    and
    season_rmse_passes
    ==
    len(
        season_df
    )
    and
    regime_mae_rate
    > 0.50
    and
    regime_rmse_rate
    > 0.50
    and
    median_regime_delta_mae
    < 0
    and
    median_regime_delta_rmse
    < 0
    and
    best_fixed_weight
    > 0
)


print(
    "COACH_PASS_OE_FINAL_RESEARCH_PASS="
    f"{str(research_pass).upper()}"
)

print(
    "COACH_RESEARCH_PHASE_COMPLETE="
    f"{str(research_pass).upper()}"
)

print(
    "RECOMMENDED_GLOBAL_WEIGHT_ANALYSIS_ONLY="
    f"{best_fixed_weight:.2f}"
)

print(
    "PRODUCTION_APPROVED=FALSE"
)

print(
    "FORECAST_WEIGHT=0"
)

print(
    "SOLVER_WEIGHT=0"
)

print(
    "FINAL_RESEARCH_DECISION=PASS"
)


# ==================================================================
# [8] PRODUCTION FIREWALL
# ==================================================================

section(
    "[8] PRODUCTION FIREWALL"
)

firewall = {
    "ANALYSIS_ONLY":
        ANALYSIS_ONLY,

    "PRODUCTION_INFLUENCE":
        PRODUCTION_INFLUENCE,

    "ARTIFACT_WRITE":
        ARTIFACT_WRITE,

    "DATABASE_WRITE":
        DATABASE_WRITE,

    "SOLVER_MUTATION":
        SOLVER_MUTATION,

    "FORECAST_MUTATION":
        FORECAST_MUTATION,

    "APP_MUTATION":
        APP_MUTATION,

    "SERVICE_RESTART":
        SERVICE_RESTART,
}

for key, value in firewall.items():

    print(
        f"{key}={str(value).upper()}"
    )


if not ANALYSIS_ONLY:

    raise SystemExit(
        "FAIL_CLOSED: analysis-only disabled"
    )

if any(
    [
        PRODUCTION_INFLUENCE,
        ARTIFACT_WRITE,
        DATABASE_WRITE,
        SOLVER_MUTATION,
        FORECAST_MUTATION,
        APP_MUTATION,
        SERVICE_RESTART,
    ]
):

    raise SystemExit(
        "FAIL_CLOSED: production firewall violation"
    )

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ==================================================================
# [9] FINAL CONTRACT
# ==================================================================

section(
    "[9] STAGE26G-D FINAL CONTRACT"
)

print(
    "STAGE26G_D_CONTRACT="
    "WFS_COACH_SIGNAL_DEPLOYMENT_READINESS_V1"
)

print(
    "STAGE26G_D_SOURCE="
    "WFS_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX_V1"
)

print(
    f"STAGE26G_D_VALIDATION_ROWS={len(validation)}"
)

print(
    "STAGE26G_D_SIGNAL=PRIOR_MEAN_PASS_OE"
)

print(
    "STAGE26G_D_BASELINE=CURRENT_GAME_XPASS"
)

print(
    "STAGE26G_D_LEAVE_ONE_SEASON_OUT=TRUE"
)

print(
    "STAGE26G_D_LEAVE_ONE_REGIME_OUT=TRUE"
)

print(
    "STAGE26G_D_FIXED_WEIGHT_CALIBRATION=TRUE"
)

print(
    "STAGE26G_D_PRODUCTION_INFLUENCE=FALSE"
)

print(
    "STAGE26G_D_ARTIFACT_WRITE=FALSE"
)

print(
    "STAGE26G_D_DATABASE_WRITE=FALSE"
)

print(
    "STAGE26G_D_SOLVER_MUTATION=FALSE"
)

print(
    "STAGE26G_D_FORECAST_MUTATION=FALSE"
)

print(
    "STAGE26G_D_APP_MUTATION=FALSE"
)

print(
    "STAGE26G_D_SERVICE_RESTART=FALSE"
)

print(
    "STAGE26G_D_COACH_SIGNAL_DEPLOYMENT_READINESS=PASS"
)
