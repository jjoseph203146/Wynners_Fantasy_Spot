#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# WFS STAGE26G-C
# INCREMENTAL COACH SIGNAL VALUE + SHRINKAGE VALIDATION
# ==================================================================
#
# ANALYSIS_ONLY=TRUE
#
# PURPOSE:
#   Test whether point-in-time historical coach-regime Pass-OE
#   improves next-game pass-rate prediction beyond current-game
#   contextual xPass.
#
# BASELINE:
#   current_mean_xpass
#
# COACH-ADJUSTED:
#   current_mean_xpass
#   + weight * (prior_mean_pass_oe / 100)
#
# IMPORTANT:
#   - Stage26G-A remains the source authority.
#   - G-C does not reconstruct coach tendencies independently.
#   - Current game is excluded from historical predictor.
#   - Future games may not select earlier weights.
#   - Coach-regime boundaries remain hard resets.
#   - This stage has NO production influence.
#
# ARTIFACT_WRITE=FALSE
# DATABASE_WRITE=FALSE
# SOLVER_MUTATION=FALSE
# FORECAST_MUTATION=FALSE
# APP_MUTATION=FALSE
# SERVICE_RESTART=FALSE
#
# ==================================================================


ROOT = Path("/home/mwynn/nfl_data_engine")

STAGE26GA_PATH = (
    ROOT
    / "scripts"
    / "stage26ga_coach_regime_tendency_matrix.py"
)

ANALYSIS_ONLY = True

ARTIFACT_WRITE = False
DATABASE_WRITE = False
SOLVER_MUTATION = False
FORECAST_MUTATION = False
APP_MUTATION = False
SERVICE_RESTART = False


# Candidate coach-signal weights.
#
# 0.00 is the required pure xPass control.
# Larger values allow progressively more historical coach Pass-OE.
#
# These are evaluated, not promoted.
CANDIDATE_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
    1.00,
]


# Minimum number of chronological training rows before
# a walk-forward row is allowed to select a nonzero weight.
MIN_TRAIN_ROWS = 200


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


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    n = num(
        numerator
    )

    d = num(
        denominator
    )

    result = pd.Series(
        np.nan,
        index=n.index,
        dtype=float,
    )

    valid = (
        n.notna()
        &
        d.notna()
        &
        d.gt(0)
    )

    result.loc[
        valid
    ] = (
        n.loc[
            valid
        ]
        /
        d.loc[
            valid
        ]
    )

    return result


def finite_df(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:

    out = df.copy()

    for col in cols:

        out[
            col
        ] = num(
            out[
                col
            ]
        )

    mask = pd.Series(
        True,
        index=out.index,
    )

    for col in cols:

        mask &= (
            out[
                col
            ].notna()
            &
            np.isfinite(
                out[
                    col
                ]
            )
        )

    return out[
        mask
    ].copy()


def metric_mae(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    pair = pd.DataFrame(
        {
            "actual":
                num(actual),

            "pred":
                num(pred),
        }
    ).dropna()

    pair = pair[
        np.isfinite(
            pair[
                "actual"
            ]
        )
        &
        np.isfinite(
            pair[
                "pred"
            ]
        )
    ]

    if pair.empty:

        return np.nan

    return float(
        (
            pair[
                "actual"
            ]
            -
            pair[
                "pred"
            ]
        )
        .abs()
        .mean()
    )


def metric_rmse(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    pair = pd.DataFrame(
        {
            "actual":
                num(actual),

            "pred":
                num(pred),
        }
    ).dropna()

    pair = pair[
        np.isfinite(
            pair[
                "actual"
            ]
        )
        &
        np.isfinite(
            pair[
                "pred"
            ]
        )
    ]

    if pair.empty:

        return np.nan

    err = (
        pair[
            "actual"
        ]
        -
        pair[
            "pred"
        ]
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    err
                )
            )
        )
    )


def metric_bias(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    pair = pd.DataFrame(
        {
            "actual":
                num(actual),

            "pred":
                num(pred),
        }
    ).dropna()

    pair = pair[
        np.isfinite(
            pair[
                "actual"
            ]
        )
        &
        np.isfinite(
            pair[
                "pred"
            ]
        )
    ]

    if pair.empty:

        return np.nan

    return float(
        (
            pair[
                "pred"
            ]
            -
            pair[
                "actual"
            ]
        )
        .mean()
    )


def metric_corr(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    pair = pd.DataFrame(
        {
            "actual":
                num(actual),

            "pred":
                num(pred),
        }
    ).dropna()

    pair = pair[
        np.isfinite(
            pair[
                "actual"
            ]
        )
        &
        np.isfinite(
            pair[
                "pred"
            ]
        )
    ]

    if len(
        pair
    ) < 3:

        return np.nan

    if (
        pair[
            "actual"
        ].nunique()
        < 2
        or
        pair[
            "pred"
        ].nunique()
        < 2
    ):

        return np.nan

    return float(
        pair[
            "actual"
        ].corr(
            pair[
                "pred"
            ],
            method="pearson",
        )
    )


def pct_improved(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual":
                num(actual),

            "baseline":
                num(baseline),

            "candidate":
                num(candidate),
        }
    ).dropna()

    df = df[
        np.isfinite(
            df[
                "actual"
            ]
        )
        &
        np.isfinite(
            df[
                "baseline"
            ]
        )
        &
        np.isfinite(
            df[
                "candidate"
            ]
        )
    ]

    if df.empty:

        return np.nan

    baseline_err = (
        df[
            "actual"
        ]
        -
        df[
            "baseline"
        ]
    ).abs()

    candidate_err = (
        df[
            "actual"
        ]
        -
        df[
            "candidate"
        ]
    ).abs()

    return float(
        candidate_err.lt(
            baseline_err
        ).mean()
    )


def evaluate_predictions(
    df: pd.DataFrame,
    actual_col: str,
    baseline_col: str,
    candidate_col: str,
) -> dict[str, float]:

    return {
        "n":
            int(
                finite_df(
                    df,
                    [
                        actual_col,
                        baseline_col,
                        candidate_col,
                    ],
                ).shape[
                    0
                ]
            ),

        "baseline_mae":
            metric_mae(
                df[
                    actual_col
                ],
                df[
                    baseline_col
                ],
            ),

        "candidate_mae":
            metric_mae(
                df[
                    actual_col
                ],
                df[
                    candidate_col
                ],
            ),

        "baseline_rmse":
            metric_rmse(
                df[
                    actual_col
                ],
                df[
                    baseline_col
                ],
            ),

        "candidate_rmse":
            metric_rmse(
                df[
                    actual_col
                ],
                df[
                    candidate_col
                ],
            ),

        "baseline_corr":
            metric_corr(
                df[
                    actual_col
                ],
                df[
                    baseline_col
                ],
            ),

        "candidate_corr":
            metric_corr(
                df[
                    actual_col
                ],
                df[
                    candidate_col
                ],
            ),

        "baseline_bias":
            metric_bias(
                df[
                    actual_col
                ],
                df[
                    baseline_col
                ],
            ),

        "candidate_bias":
            metric_bias(
                df[
                    actual_col
                ],
                df[
                    candidate_col
                ],
            ),

        "pct_games_improved":
            pct_improved(
                df[
                    actual_col
                ],
                df[
                    baseline_col
                ],
                df[
                    candidate_col
                ],
            ),
    }


# ==================================================================
# [0] RUNTIME CONTRACT
# ==================================================================

section(
    "[0] STAGE26G-C RUNTIME CONTRACT"
)

print(
    "ANALYSIS_ONLY=TRUE"
)

print(
    "SOURCE_STAGE26G_A=TRUE"
)

print(
    "BASELINE=CURRENT_GAME_XPASS"
)

print(
    "INCREMENTAL_SIGNAL=PRIOR_COACH_REGIME_PASS_OE"
)

print(
    "POINT_IN_TIME=TRUE"
)

print(
    "CURRENT_GAME_EXCLUDED_FROM_COACH_PROFILE=TRUE"
)

print(
    "FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "CHRONOLOGICAL_WEIGHT_SELECTION=TRUE"
)

print(
    "TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "PRODUCTION_INFLUENCE=FALSE"
)

print(
    "ARTIFACT_WRITE=FALSE"
)

print(
    "DATABASE_WRITE=FALSE"
)

print(
    "SOLVER_MUTATION=FALSE"
)

print(
    "FORECAST_MUTATION=FALSE"
)

print(
    "APP_MUTATION=FALSE"
)

print(
    "SERVICE_RESTART=FALSE"
)


# ==================================================================
# [1] LOAD VALIDATED STAGE26G-A MATRIX
# ==================================================================

section(
    "[1] LOAD VALIDATED STAGE26G-A MATRIX"
)

if not STAGE26GA_PATH.is_file():

    raise SystemExit(
        f"FAIL_CLOSED: Stage26G-A missing: {STAGE26GA_PATH}"
    )

print(
    f"STAGE26G_A_PATH={STAGE26GA_PATH}"
)

ga = runpy.run_path(
    str(
        STAGE26GA_PATH
    )
)

if "matrix" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A did not expose matrix"
    )

if "coach_id_col" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A did not expose coach identity column"
    )

matrix = ga[
    "matrix"
].copy()

coach_id_col = ga[
    "coach_id_col"
]

print(
    f"STAGE26G_A_MATRIX_ROWS={len(matrix)}"
)

print(
    "STAGE26G_A_MATRIX_REGIMES="
    f"{matrix['team_coach_regime_key'].nunique()}"
)

if len(
    matrix
) != 1710:

    raise SystemExit(
        "FAIL_CLOSED: G-A matrix row contract changed"
    )

if matrix[
    "team_coach_regime_key"
].nunique() != 49:

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
        "FAIL_CLOSED: duplicate G-A matrix identity"
    )

print(
    "STAGE26G_A_MATRIX_IMPORT=PASS"
)


# ==================================================================
# [2] REASSERT PIT / REGIME INTEGRITY
# ==================================================================

section(
    "[2] POINT-IN-TIME AND REGIME INTEGRITY"
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
    "_gc_regime_row"
] = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .cumcount()
)

regime_sequence_violations = int(
    (
        num(
            matrix[
                "prior_pbp_regime_games"
            ]
        )
        -
        num(
            matrix[
                "_gc_regime_row"
            ]
        )
    )
    .abs()
    .gt(0)
    .sum()
)

regime_identity_crossover = int(
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

first_rows = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)
)

first_history_nonzero = int(
    num(
        first_rows[
            "prior_pbp_regime_games"
        ]
    )
    .ne(0)
    .sum()
)

print(
    "REGIME_SEQUENCE_VIOLATIONS="
    f"{regime_sequence_violations}"
)

print(
    "REGIME_IDENTITY_CROSSOVER_ROWS="
    f"{regime_identity_crossover}"
)

print(
    "FIRST_REGIME_ROW_HISTORY_NONZERO="
    f"{first_history_nonzero}"
)

if regime_sequence_violations:

    raise SystemExit(
        "FAIL_CLOSED: PIT regime sequence violation"
    )

if regime_identity_crossover:

    raise SystemExit(
        "FAIL_CLOSED: coach-regime crossover"
    )

if first_history_nonzero:

    raise SystemExit(
        "FAIL_CLOSED: cold-start regime inherited history"
    )

print(
    "CURRENT_GAME_INCLUDED_IN_COACH_PROFILE=FALSE"
)

print(
    "FUTURE_GAME_INCLUDED_IN_COACH_PROFILE=FALSE"
)

print(
    "POINT_IN_TIME_AND_REGIME_INTEGRITY=PASS"
)


# ==================================================================
# [3] CURRENT-GAME TARGET + CONTEXT BASELINE
# ==================================================================

section(
    "[3] CURRENT-GAME TARGET AND CONTEXT BASELINE"
)

matrix[
    "current_observed_pass_rate"
] = safe_ratio(
    matrix[
        "combined_nfl_pass"
    ],
    matrix[
        "combined_n"
    ],
)

matrix[
    "current_mean_xpass"
] = safe_ratio(
    matrix[
        "combined_xpass_sum"
    ],
    matrix[
        "combined_n"
    ],
)

matrix[
    "current_mean_pass_oe"
] = safe_ratio(
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

validation = finite_df(
    validation,
    [
        "current_observed_pass_rate",
        "current_mean_xpass",
        "current_mean_pass_oe",
        "prior_mean_pass_oe",
        "prior_pbp_regime_games",
    ],
)

print(
    f"VALIDATION_ROWS={len(validation)}"
)

if validation.empty:

    raise SystemExit(
        "FAIL_CLOSED: no usable G-C validation rows"
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

oe_identity_error = (
    validation[
        "current_mean_pass_oe"
    ]
    -
    derived_oe
).abs()

max_oe_identity_error = float(
    oe_identity_error.max()
)

print(
    "MAX_CURRENT_PASS_OE_IDENTITY_ERROR="
    f"{max_oe_identity_error:.12f}"
)

if max_oe_identity_error > 1e-6:

    raise SystemExit(
        "FAIL_CLOSED: current Pass-OE identity violation"
    )

print(
    "CURRENT_GAME_TARGET_CONTEXT_BASELINE=PASS"
)


# ==================================================================
# [4] ZERO-WEIGHT CONTROL
# ==================================================================

section(
    "[4] ZERO-WEIGHT CONTROL"
)

validation[
    "baseline_prediction"
] = validation[
    "current_mean_xpass"
]

validation[
    "candidate_weight_0"
] = (
    validation[
        "current_mean_xpass"
    ]
    +
    0.0
    *
    (
        validation[
            "prior_mean_pass_oe"
        ]
        /
        100.0
    )
)

zero_control_error = (
    validation[
        "candidate_weight_0"
    ]
    -
    validation[
        "baseline_prediction"
    ]
).abs()

max_zero_control_error = float(
    zero_control_error.max()
)

print(
    "ZERO_WEIGHT_CONTROL_MAX_ERROR="
    f"{max_zero_control_error:.12f}"
)

if max_zero_control_error > 1e-12:

    raise SystemExit(
        "FAIL_CLOSED: zero-weight candidate does not reproduce baseline"
    )

print(
    "ZERO_WEIGHT_CONTROL=PASS"
)


# ==================================================================
# [5] FIXED-WEIGHT GRID EVALUATION
# ==================================================================

section(
    "[5] FIXED-WEIGHT GRID EVALUATION"
)

grid_results = []

for weight in CANDIDATE_WEIGHTS:

    col = (
        "pred_weight_"
        +
        str(
            weight
        ).replace(
            ".",
            "_"
        )
    )

    validation[
        col
    ] = (
        validation[
            "current_mean_xpass"
        ]
        +
        weight
        *
        (
            validation[
                "prior_mean_pass_oe"
            ]
            /
            100.0
        )
    )

    metrics = evaluate_predictions(
        validation,
        actual_col="current_observed_pass_rate",
        baseline_col="baseline_prediction",
        candidate_col=col,
    )

    delta_mae = (
        metrics[
            "candidate_mae"
        ]
        -
        metrics[
            "baseline_mae"
        ]
    )

    delta_rmse = (
        metrics[
            "candidate_rmse"
        ]
        -
        metrics[
            "baseline_rmse"
        ]
    )

    print(
        f"WEIGHT_{weight:.2f}_N="
        f"{metrics['n']}"
    )

    print(
        f"WEIGHT_{weight:.2f}_MAE="
        f"{metrics['candidate_mae']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_DELTA_MAE="
        f"{delta_mae:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_RMSE="
        f"{metrics['candidate_rmse']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_DELTA_RMSE="
        f"{delta_rmse:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_CORR="
        f"{metrics['candidate_corr']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_BIAS="
        f"{metrics['candidate_bias']:.8f}"
    )

    print(
        f"WEIGHT_{weight:.2f}_PCT_GAMES_IMPROVED="
        f"{metrics['pct_games_improved']:.8f}"
    )

    grid_results.append(
        {
            "weight":
                weight,

            **metrics,

            "delta_mae":
                delta_mae,

            "delta_rmse":
                delta_rmse,
        }
    )

grid_df = pd.DataFrame(
    grid_results
)

print(
    "FIXED_WEIGHT_GRID_EVALUATION=PASS"
)


# ==================================================================
# [6] BEST FULL-SAMPLE WEIGHT — DIAGNOSTIC ONLY
# ==================================================================

section(
    "[6] FULL-SAMPLE BEST WEIGHT — DIAGNOSTIC ONLY"
)

best_full = (
    grid_df
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
    .iloc[
        0
    ]
)

print(
    "FULL_SAMPLE_BEST_WEIGHT="
    f"{float(best_full['weight']):.2f}"
)

print(
    "FULL_SAMPLE_BEST_MAE="
    f"{float(best_full['candidate_mae']):.8f}"
)

print(
    "FULL_SAMPLE_BEST_DELTA_MAE="
    f"{float(best_full['delta_mae']):.8f}"
)

print(
    "FULL_SAMPLE_BEST_WEIGHT_PRODUCTION_AUTHORIZED=FALSE"
)

print(
    "FULL_SAMPLE_BEST_WEIGHT_USED_FOR_OOS_SELECTION=FALSE"
)

print(
    "FULL_SAMPLE_BEST_WEIGHT_DIAGNOSTIC=PASS"
)


# ==================================================================
# [7] CHRONOLOGICAL WALK-FORWARD WEIGHT SELECTION
# ==================================================================

section(
    "[7] CHRONOLOGICAL WALK-FORWARD WEIGHT SELECTION"
)

walk = validation.sort_values(
    [
        "game_date_dt",
        "game_id",
        "team",
    ],
    kind="stable",
).reset_index(
    drop=True
)

walk[
    "selected_weight"
] = np.nan

walk[
    "walk_forward_prediction"
] = np.nan

walk[
    "walk_forward_train_rows"
] = 0

weight_selection_counts = {
    weight: 0
    for weight in CANDIDATE_WEIGHTS
}

for idx in range(
    len(
        walk
    )
):

    train = walk.iloc[
        :idx
    ].copy()

    walk.at[
        idx,
        "walk_forward_train_rows",
    ] = len(
        train
    )

    if len(
        train
    ) < MIN_TRAIN_ROWS:

        selected_weight = 0.0

    else:

        train_results = []

        for weight in CANDIDATE_WEIGHTS:

            pred = (
                train[
                    "current_mean_xpass"
                ]
                +
                weight
                *
                (
                    train[
                        "prior_mean_pass_oe"
                    ]
                    /
                    100.0
                )
            )

            this_mae = metric_mae(
                train[
                    "current_observed_pass_rate"
                ],
                pred,
            )

            this_rmse = metric_rmse(
                train[
                    "current_observed_pass_rate"
                ],
                pred,
            )

            train_results.append(
                {
                    "weight":
                        weight,

                    "mae":
                        this_mae,

                    "rmse":
                        this_rmse,
                }
            )

        train_result_df = pd.DataFrame(
            train_results
        )

        selected = (
            train_result_df
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
            .iloc[
                0
            ]
        )

        selected_weight = float(
            selected[
                "weight"
            ]
        )

    row = walk.iloc[
        idx
    ]

    prediction = (
        float(
            row[
                "current_mean_xpass"
            ]
        )
        +
        selected_weight
        *
        (
            float(
                row[
                    "prior_mean_pass_oe"
                ]
            )
            /
            100.0
        )
    )

    walk.at[
        idx,
        "selected_weight",
    ] = selected_weight

    walk.at[
        idx,
        "walk_forward_prediction",
    ] = prediction

    weight_selection_counts[
        selected_weight
    ] += 1


print(
    f"WALK_FORWARD_ROWS={len(walk)}"
)

print(
    f"WALK_FORWARD_MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}"
)

for weight in CANDIDATE_WEIGHTS:

    print(
        f"WALK_FORWARD_SELECTED_WEIGHT_{weight:.2f}_ROWS="
        f"{weight_selection_counts[weight]}"
    )

if walk[
    "selected_weight"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: walk-forward row missing selected weight"
    )

if walk[
    "walk_forward_prediction"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: walk-forward row missing prediction"
    )

print(
    "CHRONOLOGICAL_WALK_FORWARD_WEIGHT_SELECTION=PASS"
)


# ==================================================================
# [8] WALK-FORWARD OUT-OF-SAMPLE VALUE
# ==================================================================

section(
    "[8] WALK-FORWARD OUT-OF-SAMPLE VALUE"
)

oos = walk[
    walk[
        "walk_forward_train_rows"
    ].ge(
        MIN_TRAIN_ROWS
    )
].copy()

print(
    f"OOS_EVALUATION_ROWS={len(oos)}"
)

if oos.empty:

    raise SystemExit(
        "FAIL_CLOSED: no walk-forward OOS rows"
    )

oos_metrics = evaluate_predictions(
    oos,
    actual_col="current_observed_pass_rate",
    baseline_col="baseline_prediction",
    candidate_col="walk_forward_prediction",
)

oos_delta_mae = (
    oos_metrics[
        "candidate_mae"
    ]
    -
    oos_metrics[
        "baseline_mae"
    ]
)

oos_delta_rmse = (
    oos_metrics[
        "candidate_rmse"
    ]
    -
    oos_metrics[
        "baseline_rmse"
    ]
)

print(
    "OOS_BASELINE_MAE="
    f"{oos_metrics['baseline_mae']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_MAE="
    f"{oos_metrics['candidate_mae']:.8f}"
)

print(
    "OOS_DELTA_MAE="
    f"{oos_delta_mae:.8f}"
)

print(
    "OOS_BASELINE_RMSE="
    f"{oos_metrics['baseline_rmse']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_RMSE="
    f"{oos_metrics['candidate_rmse']:.8f}"
)

print(
    "OOS_DELTA_RMSE="
    f"{oos_delta_rmse:.8f}"
)

print(
    "OOS_BASELINE_CORR="
    f"{oos_metrics['baseline_corr']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_CORR="
    f"{oos_metrics['candidate_corr']:.8f}"
)

print(
    "OOS_BASELINE_BIAS="
    f"{oos_metrics['baseline_bias']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_BIAS="
    f"{oos_metrics['candidate_bias']:.8f}"
)

print(
    "OOS_PCT_GAMES_IMPROVED="
    f"{oos_metrics['pct_games_improved']:.8f}"
)

print(
    "OOS_COACH_ADJUSTMENT_BEATS_BASELINE_MAE="
    f"{str(oos_delta_mae < 0).upper()}"
)

print(
    "OOS_COACH_ADJUSTMENT_BEATS_BASELINE_RMSE="
    f"{str(oos_delta_rmse < 0).upper()}"
)

print(
    "WALK_FORWARD_OUT_OF_SAMPLE_VALUE=PASS"
)


# ==================================================================
# [9] SEASON-BY-SEASON ROBUSTNESS
# ==================================================================

section(
    "[9] SEASON-BY-SEASON ROBUSTNESS"
)

oos[
    "season_eval"
] = num(
    oos[
        "season"
    ]
).astype(
    "Int64"
)

for season in sorted(
    oos[
        "season_eval"
    ]
    .dropna()
    .unique()
):

    sub = oos[
        oos[
            "season_eval"
        ].eq(
            season
        )
    ]

    metrics = evaluate_predictions(
        sub,
        actual_col="current_observed_pass_rate",
        baseline_col="baseline_prediction",
        candidate_col="walk_forward_prediction",
    )

    delta_mae = (
        metrics[
            "candidate_mae"
        ]
        -
        metrics[
            "baseline_mae"
        ]
    )

    delta_rmse = (
        metrics[
            "candidate_rmse"
        ]
        -
        metrics[
            "baseline_rmse"
        ]
    )

    print(
        f"SEASON_{int(season)}_N="
        f"{metrics['n']}"
    )

    print(
        f"SEASON_{int(season)}_DELTA_MAE="
        f"{delta_mae:.8f}"
    )

    print(
        f"SEASON_{int(season)}_DELTA_RMSE="
        f"{delta_rmse:.8f}"
    )

    print(
        f"SEASON_{int(season)}_PCT_GAMES_IMPROVED="
        f"{metrics['pct_games_improved']:.8f}"
    )

print(
    "SEASON_BY_SEASON_ROBUSTNESS=PASS"
)


# ==================================================================
# [10] MATURITY-BUCKET ROBUSTNESS
# ==================================================================

section(
    "[10] MATURITY-BUCKET ROBUSTNESS"
)

oos[
    "maturity_bucket"
] = pd.cut(
    num(
        oos[
            "prior_pbp_regime_games"
        ]
    ),
    bins=[
        0,
        3,
        7,
        15,
        np.inf,
    ],
    labels=[
        "01_1_TO_3",
        "02_4_TO_7",
        "03_8_TO_15",
        "04_16_PLUS",
    ],
    right=True,
)

if oos[
    "maturity_bucket"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: OOS maturity bucket missing"
    )

for bucket in [
    "01_1_TO_3",
    "02_4_TO_7",
    "03_8_TO_15",
    "04_16_PLUS",
]:

    sub = oos[
        oos[
            "maturity_bucket"
        ].astype(
            str
        ).eq(
            bucket
        )
    ]

    metrics = evaluate_predictions(
        sub,
        actual_col="current_observed_pass_rate",
        baseline_col="baseline_prediction",
        candidate_col="walk_forward_prediction",
    )

    delta_mae = (
        metrics[
            "candidate_mae"
        ]
        -
        metrics[
            "baseline_mae"
        ]
    )

    delta_rmse = (
        metrics[
            "candidate_rmse"
        ]
        -
        metrics[
            "baseline_rmse"
        ]
    )

    print(
        f"MATURITY_{bucket}_N="
        f"{metrics['n']}"
    )

    print(
        f"MATURITY_{bucket}_DELTA_MAE="
        + (
            "NA"
            if not np.isfinite(
                delta_mae
            )
            else f"{delta_mae:.8f}"
        )
    )

    print(
        f"MATURITY_{bucket}_DELTA_RMSE="
        + (
            "NA"
            if not np.isfinite(
                delta_rmse
            )
            else f"{delta_rmse:.8f}"
        )
    )

    print(
        f"MATURITY_{bucket}_PCT_GAMES_IMPROVED="
        + (
            "NA"
            if not np.isfinite(
                metrics[
                    "pct_games_improved"
                ]
            )
            else f"{metrics['pct_games_improved']:.8f}"
        )
    )

print(
    "MATURITY_BUCKET_ROBUSTNESS=PASS"
)


# ==================================================================
# [11] REGIME-LEVEL ROBUSTNESS
# ==================================================================

section(
    "[11] REGIME-LEVEL ROBUSTNESS"
)

regime_results = []

for regime, sub in oos.groupby(
    "team_coach_regime_key",
    sort=False,
):

    metrics = evaluate_predictions(
        sub,
        actual_col="current_observed_pass_rate",
        baseline_col="baseline_prediction",
        candidate_col="walk_forward_prediction",
    )

    if metrics[
        "n"
    ] < 5:

        continue

    delta_mae = (
        metrics[
            "candidate_mae"
        ]
        -
        metrics[
            "baseline_mae"
        ]
    )

    regime_results.append(
        {
            "team_coach_regime_key":
                regime,

            "n":
                metrics[
                    "n"
                ],

            "delta_mae":
                delta_mae,
        }
    )

regime_df = pd.DataFrame(
    regime_results
)

if regime_df.empty:

    raise SystemExit(
        "FAIL_CLOSED: no eligible regime-level validation groups"
    )

regimes_improved = int(
    regime_df[
        "delta_mae"
    ].lt(
        0
    ).sum()
)

regime_improvement_rate = float(
    regime_df[
        "delta_mae"
    ].lt(
        0
    ).mean()
)

median_regime_delta_mae = float(
    regime_df[
        "delta_mae"
    ].median()
)

print(
    f"REGIME_EVALUATION_COUNT={len(regime_df)}"
)

print(
    f"REGIMES_WITH_MAE_IMPROVEMENT={regimes_improved}"
)

print(
    "REGIME_MAE_IMPROVEMENT_RATE="
    f"{regime_improvement_rate:.8f}"
)

print(
    "MEDIAN_REGIME_DELTA_MAE="
    f"{median_regime_delta_mae:.8f}"
)

print(
    "REGIME_LEVEL_ROBUSTNESS=PASS"
)


# ==================================================================
# [12] CHRONOLOGICAL LEAKAGE AUDIT
# ==================================================================

section(
    "[12] CHRONOLOGICAL LEAKAGE AUDIT"
)

chronology = walk[
    [
        "game_date_dt",
        "game_id",
        "team",
        "walk_forward_train_rows",
    ]
].copy()

chronology[
    "game_date_dt"
] = pd.to_datetime(
    chronology[
        "game_date_dt"
    ],
    errors="coerce",
)

if chronology[
    "game_date_dt"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: G-C chronology contains null dates"
    )

chronology_violations = int(
    chronology[
        "game_date_dt"
    ]
    .diff()
    .dt.total_seconds()
    .fillna(
        0
    )
    .lt(
        0
    )
    .sum()
)

training_count_violations = int(
    (
        chronology[
            "walk_forward_train_rows"
        ]
        -
        np.arange(
            len(
                chronology
            )
        )
    )
    .astype(
        float
    )
    .round(
        12
    )
    .ne(
        0
    )
    .sum()
)

print(
    "GLOBAL_CHRONOLOGY_VIOLATIONS="
    f"{chronology_violations}"
)

print(
    "WALK_FORWARD_TRAIN_COUNT_VIOLATIONS="
    f"{training_count_violations}"
)

if chronology_violations:

    raise SystemExit(
        "FAIL_CLOSED: global walk-forward chronology violation"
    )

if training_count_violations:

    raise SystemExit(
        "FAIL_CLOSED: walk-forward training set includes non-prior rows"
    )

print(
    "FUTURE_ROWS_USED_TO_SELECT_EARLIER_WEIGHT=FALSE"
)

print(
    "CURRENT_ROW_USED_TO_SELECT_OWN_WEIGHT=FALSE"
)

print(
    "CHRONOLOGICAL_LEAKAGE_AUDIT=PASS"
)


# ==================================================================
# [13] DESCRIPTIVE DECISION GATE
# ==================================================================

section(
    "[13] DESCRIPTIVE DECISION GATE"
)

# --------------------------------------------------------------
# This gate is ANALYTIC ONLY.
#
# It does not authorize forecast or solver use.
#
# ADVANCES_FOR_FURTHER_STUDY requires:
#
#   1. OOS MAE improvement
#   2. OOS RMSE improvement
#   3. > 50% of games improved
#   4. > 50% of evaluated regimes improved
#
# Even a PASS here remains analysis-only.
# --------------------------------------------------------------

advance_for_further_study = (
    oos_delta_mae < 0
    and
    oos_delta_rmse < 0
    and
    oos_metrics[
        "pct_games_improved"
    ] > 0.50
    and
    regime_improvement_rate > 0.50
)

print(
    "COACH_PASS_OE_ADVANCES_FOR_FURTHER_STUDY="
    f"{str(advance_for_further_study).upper()}"
)

print(
    "COACH_PASS_OE_PRODUCTION_APPROVED=FALSE"
)

print(
    "COACH_PASS_OE_FORECAST_WEIGHT=0"
)

print(
    "COACH_PASS_OE_SOLVER_WEIGHT=0"
)

print(
    "DESCRIPTIVE_DECISION_GATE=PASS"
)


# ==================================================================
# [14] PRODUCTION FIREWALL
# ==================================================================

section(
    "[14] PRODUCTION FIREWALL"
)

firewall = {
    "ANALYSIS_ONLY":
        ANALYSIS_ONLY,

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
        "FAIL_CLOSED: G-C analysis-only firewall disabled"
    )

if any(
    [
        ARTIFACT_WRITE,
        DATABASE_WRITE,
        SOLVER_MUTATION,
        FORECAST_MUTATION,
        APP_MUTATION,
        SERVICE_RESTART,
    ]
):

    raise SystemExit(
        "FAIL_CLOSED: G-C production firewall violation"
    )

print(
    "PRODUCTION_INFLUENCE=FALSE"
)

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ==================================================================
# [15] FINAL CONTRACT
# ==================================================================

section(
    "[15] STAGE26G-C FINAL CONTRACT"
)

print(
    "STAGE26G_C_CONTRACT="
    "WFS_COACH_INCREMENTAL_VALUE_SHRINKAGE_VALIDATION_V1"
)

print(
    "STAGE26G_C_SOURCE="
    "WFS_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX_V1"
)

print(
    f"STAGE26G_C_MATRIX_ROWS={len(matrix)}"
)

print(
    f"STAGE26G_C_VALIDATION_ROWS={len(validation)}"
)

print(
    f"STAGE26G_C_OOS_ROWS={len(oos)}"
)

print(
    "STAGE26G_C_BASELINE=CURRENT_GAME_XPASS"
)

print(
    "STAGE26G_C_INCREMENTAL_SIGNAL=PRIOR_COACH_REGIME_PASS_OE"
)

print(
    "STAGE26G_C_POINT_IN_TIME=TRUE"
)

print(
    "STAGE26G_C_CURRENT_GAME_EXCLUDED_FROM_COACH_PROFILE=TRUE"
)

print(
    "STAGE26G_C_FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "STAGE26G_C_CHRONOLOGICAL_WEIGHT_SELECTION=TRUE"
)

print(
    "STAGE26G_C_TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "STAGE26G_C_PRODUCTION_INFLUENCE=FALSE"
)

print(
    "STAGE26G_C_ARTIFACT_WRITE=FALSE"
)

print(
    "STAGE26G_C_DATABASE_WRITE=FALSE"
)

print(
    "STAGE26G_C_SOLVER_MUTATION=FALSE"
)

print(
    "STAGE26G_C_FORECAST_MUTATION=FALSE"
)

print(
    "STAGE26G_C_APP_MUTATION=FALSE"
)

print(
    "STAGE26G_C_SERVICE_RESTART=FALSE"
)

print(
    "STAGE26G_C_INCREMENTAL_COACH_SIGNAL_VALIDATION=PASS"
)
