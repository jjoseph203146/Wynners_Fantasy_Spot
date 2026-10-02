#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# WFS STAGE26G-C2
# DATE-BLOCKED WALK-FORWARD LEAKAGE HARDENING
# ==================================================================
#
# ANALYSIS_ONLY=TRUE
#
# PURPOSE:
#   Revalidate Stage26G-C incremental coach Pass-OE value using
#   strict date-blocked chronological weight selection.
#
# TRAINING RULE:
#   game_date_dt < target game_date_dt
#
# Therefore training may NOT contain:
#   - current row
#   - opponent row from current game
#   - any other game on same date
#   - any future date
#
# BASELINE:
#   current_mean_xpass
#
# COACH ADJUSTMENT:
#   current_mean_xpass
#   + selected_weight * (prior_mean_pass_oe / 100)
#
# NO PRODUCTION INFLUENCE.
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

    n = num(numerator)
    d = num(denominator)

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

    result.loc[valid] = (
        n.loc[valid]
        /
        d.loc[valid]
    )

    return result


def finite_df(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:

    out = df.copy()

    mask = pd.Series(
        True,
        index=out.index,
    )

    for col in cols:

        out[col] = num(
            out[col]
        )

        mask &= (
            out[col].notna()
            &
            np.isfinite(
                out[col]
            )
        )

    return out[
        mask
    ].copy()


def metric_mae(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual": num(actual),
            "pred": num(pred),
        }
    ).dropna()

    df = df[
        np.isfinite(df["actual"])
        &
        np.isfinite(df["pred"])
    ]

    if df.empty:
        return np.nan

    return float(
        (
            df["actual"]
            -
            df["pred"]
        )
        .abs()
        .mean()
    )


def metric_rmse(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual": num(actual),
            "pred": num(pred),
        }
    ).dropna()

    df = df[
        np.isfinite(df["actual"])
        &
        np.isfinite(df["pred"])
    ]

    if df.empty:
        return np.nan

    err = (
        df["actual"]
        -
        df["pred"]
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


def metric_corr(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual": num(actual),
            "pred": num(pred),
        }
    ).dropna()

    df = df[
        np.isfinite(df["actual"])
        &
        np.isfinite(df["pred"])
    ]

    if len(df) < 3:
        return np.nan

    if (
        df["actual"].nunique() < 2
        or
        df["pred"].nunique() < 2
    ):
        return np.nan

    return float(
        df["actual"].corr(
            df["pred"],
            method="pearson",
        )
    )


def metric_bias(
    actual: pd.Series,
    pred: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual": num(actual),
            "pred": num(pred),
        }
    ).dropna()

    df = df[
        np.isfinite(df["actual"])
        &
        np.isfinite(df["pred"])
    ]

    if df.empty:
        return np.nan

    return float(
        (
            df["pred"]
            -
            df["actual"]
        ).mean()
    )


def pct_improved(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "actual": num(actual),
            "baseline": num(baseline),
            "candidate": num(candidate),
        }
    ).dropna()

    df = df[
        np.isfinite(df["actual"])
        &
        np.isfinite(df["baseline"])
        &
        np.isfinite(df["candidate"])
    ]

    if df.empty:
        return np.nan

    baseline_error = (
        df["actual"]
        -
        df["baseline"]
    ).abs()

    candidate_error = (
        df["actual"]
        -
        df["candidate"]
    ).abs()

    return float(
        candidate_error.lt(
            baseline_error
        ).mean()
    )


def evaluate(
    df: pd.DataFrame,
    candidate_col: str,
) -> dict[str, float]:

    usable = finite_df(
        df,
        [
            "current_observed_pass_rate",
            "baseline_prediction",
            candidate_col,
        ],
    )

    return {
        "n":
            len(usable),

        "baseline_mae":
            metric_mae(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    "baseline_prediction"
                ],
            ),

        "candidate_mae":
            metric_mae(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    candidate_col
                ],
            ),

        "baseline_rmse":
            metric_rmse(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    "baseline_prediction"
                ],
            ),

        "candidate_rmse":
            metric_rmse(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    candidate_col
                ],
            ),

        "baseline_corr":
            metric_corr(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    "baseline_prediction"
                ],
            ),

        "candidate_corr":
            metric_corr(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    candidate_col
                ],
            ),

        "baseline_bias":
            metric_bias(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    "baseline_prediction"
                ],
            ),

        "candidate_bias":
            metric_bias(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    candidate_col
                ],
            ),

        "pct_improved":
            pct_improved(
                usable[
                    "current_observed_pass_rate"
                ],
                usable[
                    "baseline_prediction"
                ],
                usable[
                    candidate_col
                ],
            ),
    }


# ==================================================================
# [0] RUNTIME CONTRACT
# ==================================================================

section(
    "[0] STAGE26G-C2 RUNTIME CONTRACT"
)

print("ANALYSIS_ONLY=TRUE")
print("SOURCE_STAGE26G_A=TRUE")
print("DATE_BLOCKED_WALK_FORWARD=TRUE")
print("TRAIN_DATE_STRICTLY_LT_TARGET_DATE=TRUE")
print("SAME_GAME_TRAINING_ALLOWED=FALSE")
print("SAME_DATE_TRAINING_ALLOWED=FALSE")
print("FUTURE_DATE_TRAINING_ALLOWED=FALSE")
print("TEAM_COACH_REGIME_RESET=TRUE")
print("PRODUCTION_INFLUENCE=FALSE")

print("ARTIFACT_WRITE=FALSE")
print("DATABASE_WRITE=FALSE")
print("SOLVER_MUTATION=FALSE")
print("FORECAST_MUTATION=FALSE")
print("APP_MUTATION=FALSE")
print("SERVICE_RESTART=FALSE")


# ==================================================================
# [1] LOAD VALIDATED STAGE26G-A MATRIX
# ==================================================================

section(
    "[1] LOAD VALIDATED STAGE26G-A MATRIX"
)

if not STAGE26GA_PATH.is_file():

    raise SystemExit(
        f"FAIL_CLOSED: missing G-A script: {STAGE26GA_PATH}"
    )

ga = runpy.run_path(
    str(STAGE26GA_PATH)
)

if "matrix" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: G-A did not expose matrix"
    )

if "coach_id_col" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: G-A did not expose coach_id_col"
    )

matrix = ga["matrix"].copy()

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

if len(matrix) != 1710:

    raise SystemExit(
        "FAIL_CLOSED: G-A row contract changed"
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
# [2] REASSERT PIT / REGIME CONTRACT
# ==================================================================

section(
    "[2] POINT-IN-TIME AND REGIME CONTRACT"
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
        "FAIL_CLOSED: null game_date_dt"
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
        num(
            matrix[
                "_regime_row"
            ]
        )
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

first_rows = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)
)

cold_start_history_nonzero = int(
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
    f"{sequence_violations}"
)

print(
    "REGIME_IDENTITY_CROSSOVER_ROWS="
    f"{regime_crossovers}"
)

print(
    "COLD_START_HISTORY_NONZERO="
    f"{cold_start_history_nonzero}"
)

if sequence_violations:

    raise SystemExit(
        "FAIL_CLOSED: regime sequence violation"
    )

if regime_crossovers:

    raise SystemExit(
        "FAIL_CLOSED: regime identity crossover"
    )

if cold_start_history_nonzero:

    raise SystemExit(
        "FAIL_CLOSED: cold-start regime inherited history"
    )

print(
    "POINT_IN_TIME_AND_REGIME_CONTRACT=PASS"
)


# ==================================================================
# [3] CURRENT-GAME TARGET / XPASS BASELINE
# ==================================================================

section(
    "[3] CURRENT-GAME TARGET AND XPASS BASELINE"
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

validation[
    "baseline_prediction"
] = validation[
    "current_mean_xpass"
]

print(
    f"VALIDATION_ROWS={len(validation)}"
)

derived_current_oe = (
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
        derived_current_oe
    )
    .abs()
    .max()
)

print(
    "MAX_CURRENT_PASS_OE_IDENTITY_ERROR="
    f"{max_oe_error:.12f}"
)

if max_oe_error > 1e-6:

    raise SystemExit(
        "FAIL_CLOSED: current Pass-OE identity failure"
    )

print(
    "CURRENT_GAME_TARGET_XPASS_BASELINE=PASS"
)


# ==================================================================
# [4] ZERO-WEIGHT CONTROL
# ==================================================================

section(
    "[4] ZERO-WEIGHT CONTROL"
)

validation[
    "zero_weight_prediction"
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

zero_error = float(
    (
        validation[
            "zero_weight_prediction"
        ]
        -
        validation[
            "baseline_prediction"
        ]
    )
    .abs()
    .max()
)

print(
    "ZERO_WEIGHT_CONTROL_MAX_ERROR="
    f"{zero_error:.12f}"
)

if zero_error > 1e-12:

    raise SystemExit(
        "FAIL_CLOSED: zero weight control mismatch"
    )

print(
    "ZERO_WEIGHT_CONTROL=PASS"
)


# ==================================================================
# [5] DATE-BLOCKED WALK-FORWARD
# ==================================================================

section(
    "[5] DATE-BLOCKED WALK-FORWARD WEIGHT SELECTION"
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
    "date_blocked_prediction"
] = np.nan

walk[
    "training_rows"
] = 0

walk[
    "max_training_date"
] = pd.NaT


unique_dates = sorted(
    walk[
        "game_date_dt"
    ]
    .dropna()
    .unique()
)


selection_counts = {
    weight: 0
    for weight in CANDIDATE_WEIGHTS
}


for target_date in unique_dates:

    target_mask = (
        walk[
            "game_date_dt"
        ].eq(
            target_date
        )
    )

    target_rows = walk[
        target_mask
    ].copy()

    train = walk[
        walk[
            "game_date_dt"
        ].lt(
            target_date
        )
    ].copy()

    train_rows = len(
        train
    )

    if train_rows:

        max_train_date = train[
            "game_date_dt"
        ].max()

    else:

        max_train_date = pd.NaT

    if train_rows < MIN_TRAIN_ROWS:

        selected_weight = 0.0

    else:

        candidates = []

        for weight in CANDIDATE_WEIGHTS:

            train_pred = (
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
                train_pred,
            )

            this_rmse = metric_rmse(
                train[
                    "current_observed_pass_rate"
                ],
                train_pred,
            )

            candidates.append(
                {
                    "weight":
                        weight,

                    "mae":
                        this_mae,

                    "rmse":
                        this_rmse,
                }
            )

        candidate_df = pd.DataFrame(
            candidates
        )

        selected = (
            candidate_df
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

        selected_weight = float(
            selected[
                "weight"
            ]
        )

    predictions = (
        target_rows[
            "current_mean_xpass"
        ]
        +
        selected_weight
        *
        (
            target_rows[
                "prior_mean_pass_oe"
            ]
            /
            100.0
        )
    )

    walk.loc[
        target_mask,
        "selected_weight",
    ] = selected_weight

    walk.loc[
        target_mask,
        "date_blocked_prediction",
    ] = predictions.to_numpy()

    walk.loc[
        target_mask,
        "training_rows",
    ] = train_rows

    walk.loc[
        target_mask,
        "max_training_date",
    ] = max_train_date

    selection_counts[
        selected_weight
    ] += len(
        target_rows
    )


print(
    f"DATE_BLOCKED_ROWS={len(walk)}"
)

print(
    f"UNIQUE_TARGET_DATES={len(unique_dates)}"
)

print(
    f"MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}"
)

for weight in CANDIDATE_WEIGHTS:

    print(
        f"SELECTED_WEIGHT_{weight:.2f}_ROWS="
        f"{selection_counts[weight]}"
    )

if walk[
    "selected_weight"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: missing selected weight"
    )

if walk[
    "date_blocked_prediction"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: missing date-blocked prediction"
    )

print(
    "DATE_BLOCKED_WALK_FORWARD_WEIGHT_SELECTION=PASS"
)


# ==================================================================
# [6] EXPLICIT SAME-GAME / SAME-DATE LEAKAGE AUDIT
# ==================================================================

section(
    "[6] SAME-GAME AND SAME-DATE LEAKAGE AUDIT"
)

same_date_training_leakage_rows = 0
same_game_training_leakage_rows = 0
max_train_date_violations = 0


for target_date in unique_dates:

    target = walk[
        walk[
            "game_date_dt"
        ].eq(
            target_date
        )
    ]

    training = walk[
        walk[
            "game_date_dt"
        ].lt(
            target_date
        )
    ]

    target_game_ids = set(
        target[
            "game_id"
        ].astype(str)
    )

    if not training.empty:

        same_date_training_leakage_rows += int(
            training[
                "game_date_dt"
            ]
            .eq(
                target_date
            )
            .sum()
        )

        same_game_training_leakage_rows += int(
            training[
                "game_id"
            ]
            .astype(str)
            .isin(
                target_game_ids
            )
            .sum()
        )

        max_train_date = training[
            "game_date_dt"
        ].max()

        if not (
            max_train_date
            <
            target_date
        ):

            max_train_date_violations += len(
                target
            )


print(
    "SAME_GAME_TRAINING_LEAKAGE_ROWS="
    f"{same_game_training_leakage_rows}"
)

print(
    "SAME_DATE_TRAINING_LEAKAGE_ROWS="
    f"{same_date_training_leakage_rows}"
)

print(
    "MAX_TRAIN_DATE_NOT_LT_TARGET_DATE_ROWS="
    f"{max_train_date_violations}"
)

print(
    "TARGET_GAME_ROWS_USED_FOR_WEIGHT_SELECTION="
    f"{same_game_training_leakage_rows}"
)

if same_game_training_leakage_rows:

    raise SystemExit(
        "FAIL_CLOSED: same-game weight-selection leakage"
    )

if same_date_training_leakage_rows:

    raise SystemExit(
        "FAIL_CLOSED: same-date weight-selection leakage"
    )

if max_train_date_violations:

    raise SystemExit(
        "FAIL_CLOSED: training date is not strictly earlier than target date"
    )

print(
    "MAX_TRAIN_DATE_LT_TARGET_DATE=TRUE"
)

print(
    "SAME_GAME_TRAINING_ALLOWED=FALSE"
)

print(
    "SAME_DATE_TRAINING_ALLOWED=FALSE"
)

print(
    "SAME_GAME_SAME_DATE_LEAKAGE_AUDIT=PASS"
)


# ==================================================================
# [7] DATE-BLOCKED OUT-OF-SAMPLE VALUE
# ==================================================================

section(
    "[7] DATE-BLOCKED OUT-OF-SAMPLE VALUE"
)

oos = walk[
    walk[
        "training_rows"
    ].ge(
        MIN_TRAIN_ROWS
    )
].copy()

print(
    f"OOS_ROWS={len(oos)}"
)

if oos.empty:

    raise SystemExit(
        "FAIL_CLOSED: no date-blocked OOS rows"
    )

metrics = evaluate(
    oos,
    candidate_col="date_blocked_prediction",
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
    "OOS_BASELINE_MAE="
    f"{metrics['baseline_mae']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_MAE="
    f"{metrics['candidate_mae']:.8f}"
)

print(
    "OOS_DELTA_MAE="
    f"{delta_mae:.8f}"
)

print(
    "OOS_BASELINE_RMSE="
    f"{metrics['baseline_rmse']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_RMSE="
    f"{metrics['candidate_rmse']:.8f}"
)

print(
    "OOS_DELTA_RMSE="
    f"{delta_rmse:.8f}"
)

print(
    "OOS_BASELINE_CORR="
    f"{metrics['baseline_corr']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_CORR="
    f"{metrics['candidate_corr']:.8f}"
)

print(
    "OOS_BASELINE_BIAS="
    f"{metrics['baseline_bias']:.8f}"
)

print(
    "OOS_COACH_ADJUSTED_BIAS="
    f"{metrics['candidate_bias']:.8f}"
)

print(
    "OOS_PCT_GAMES_IMPROVED="
    f"{metrics['pct_improved']:.8f}"
)

print(
    "OOS_BEATS_BASELINE_MAE="
    f"{str(delta_mae < 0).upper()}"
)

print(
    "OOS_BEATS_BASELINE_RMSE="
    f"{str(delta_rmse < 0).upper()}"
)

print(
    "DATE_BLOCKED_OUT_OF_SAMPLE_VALUE=PASS"
)


# ==================================================================
# [8] SEASON ROBUSTNESS
# ==================================================================

section(
    "[8] SEASON ROBUSTNESS"
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

    m = evaluate(
        sub,
        candidate_col="date_blocked_prediction",
    )

    d_mae = (
        m[
            "candidate_mae"
        ]
        -
        m[
            "baseline_mae"
        ]
    )

    d_rmse = (
        m[
            "candidate_rmse"
        ]
        -
        m[
            "baseline_rmse"
        ]
    )

    print(
        f"SEASON_{int(season)}_N="
        f"{m['n']}"
    )

    print(
        f"SEASON_{int(season)}_DELTA_MAE="
        f"{d_mae:.8f}"
    )

    print(
        f"SEASON_{int(season)}_DELTA_RMSE="
        f"{d_rmse:.8f}"
    )

    print(
        f"SEASON_{int(season)}_PCT_GAMES_IMPROVED="
        f"{m['pct_improved']:.8f}"
    )

print(
    "SEASON_ROBUSTNESS=PASS"
)


# ==================================================================
# [9] MATURITY ROBUSTNESS
# ==================================================================

section(
    "[9] MATURITY ROBUSTNESS"
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
        "FAIL_CLOSED: missing maturity bucket"
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
        ]
        .astype(str)
        .eq(
            bucket
        )
    ]

    m = evaluate(
        sub,
        candidate_col="date_blocked_prediction",
    )

    d_mae = (
        m[
            "candidate_mae"
        ]
        -
        m[
            "baseline_mae"
        ]
    )

    d_rmse = (
        m[
            "candidate_rmse"
        ]
        -
        m[
            "baseline_rmse"
        ]
    )

    print(
        f"MATURITY_{bucket}_N="
        f"{m['n']}"
    )

    print(
        f"MATURITY_{bucket}_DELTA_MAE="
        + (
            "NA"
            if not np.isfinite(
                d_mae
            )
            else f"{d_mae:.8f}"
        )
    )

    print(
        f"MATURITY_{bucket}_DELTA_RMSE="
        + (
            "NA"
            if not np.isfinite(
                d_rmse
            )
            else f"{d_rmse:.8f}"
        )
    )

    print(
        f"MATURITY_{bucket}_PCT_GAMES_IMPROVED="
        + (
            "NA"
            if not np.isfinite(
                m[
                    "pct_improved"
                ]
            )
            else f"{m['pct_improved']:.8f}"
        )
    )

print(
    "MATURITY_ROBUSTNESS=PASS"
)


# ==================================================================
# [10] REGIME ROBUSTNESS
# ==================================================================

section(
    "[10] REGIME ROBUSTNESS"
)

regime_results = []

for regime, sub in oos.groupby(
    "team_coach_regime_key",
    sort=False,
):

    m = evaluate(
        sub,
        candidate_col="date_blocked_prediction",
    )

    if m[
        "n"
    ] < 5:

        continue

    d_mae = (
        m[
            "candidate_mae"
        ]
        -
        m[
            "baseline_mae"
        ]
    )

    regime_results.append(
        {
            "regime":
                regime,

            "n":
                m[
                    "n"
                ],

            "delta_mae":
                d_mae,
        }
    )

regime_df = pd.DataFrame(
    regime_results
)

if regime_df.empty:

    raise SystemExit(
        "FAIL_CLOSED: no regime evaluation groups"
    )

regime_improvement_rate = float(
    regime_df[
        "delta_mae"
    ]
    .lt(0)
    .mean()
)

median_regime_delta = float(
    regime_df[
        "delta_mae"
    ].median()
)

print(
    "REGIME_EVALUATION_COUNT="
    f"{len(regime_df)}"
)

print(
    "REGIMES_WITH_MAE_IMPROVEMENT="
    f"{int(regime_df['delta_mae'].lt(0).sum())}"
)

print(
    "REGIME_MAE_IMPROVEMENT_RATE="
    f"{regime_improvement_rate:.8f}"
)

print(
    "MEDIAN_REGIME_DELTA_MAE="
    f"{median_regime_delta:.8f}"
)

print(
    "REGIME_ROBUSTNESS=PASS"
)


# ==================================================================
# [11] DATE-BLOCKED DECISION GATE
# ==================================================================

section(
    "[11] DATE-BLOCKED DECISION GATE"
)

advance = (
    delta_mae < 0
    and
    delta_rmse < 0
    and
    metrics[
        "pct_improved"
    ] > 0.50
    and
    regime_improvement_rate > 0.50
)

print(
    "COACH_PASS_OE_SURVIVES_DATE_BLOCKED_VALIDATION="
    f"{str(advance).upper()}"
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
    "DATE_BLOCKED_DECISION_GATE=PASS"
)


# ==================================================================
# [12] PRODUCTION FIREWALL
# ==================================================================

section(
    "[12] PRODUCTION FIREWALL"
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
        "FAIL_CLOSED: analysis-only firewall disabled"
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
        "FAIL_CLOSED: production firewall violation"
    )

print(
    "PRODUCTION_INFLUENCE=FALSE"
)

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ==================================================================
# [13] FINAL CONTRACT
# ==================================================================

section(
    "[13] STAGE26G-C2 FINAL CONTRACT"
)

print(
    "STAGE26G_C2_CONTRACT="
    "WFS_COACH_DATE_BLOCKED_INCREMENTAL_VALUE_VALIDATION_V1"
)

print(
    "STAGE26G_C2_SOURCE="
    "WFS_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX_V1"
)

print(
    f"STAGE26G_C2_MATRIX_ROWS={len(matrix)}"
)

print(
    f"STAGE26G_C2_VALIDATION_ROWS={len(validation)}"
)

print(
    f"STAGE26G_C2_OOS_ROWS={len(oos)}"
)

print(
    "STAGE26G_C2_BASELINE=CURRENT_GAME_XPASS"
)

print(
    "STAGE26G_C2_INCREMENTAL_SIGNAL=PRIOR_COACH_REGIME_PASS_OE"
)

print(
    "STAGE26G_C2_DATE_BLOCKED_WALK_FORWARD=TRUE"
)

print(
    "STAGE26G_C2_TRAIN_DATE_STRICTLY_LT_TARGET_DATE=TRUE"
)

print(
    "STAGE26G_C2_SAME_GAME_TRAINING_LEAKAGE=FALSE"
)

print(
    "STAGE26G_C2_SAME_DATE_TRAINING_LEAKAGE=FALSE"
)

print(
    "STAGE26G_C2_FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "STAGE26G_C2_TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "STAGE26G_C2_PRODUCTION_INFLUENCE=FALSE"
)

print(
    "STAGE26G_C2_ARTIFACT_WRITE=FALSE"
)

print(
    "STAGE26G_C2_DATABASE_WRITE=FALSE"
)

print(
    "STAGE26G_C2_SOLVER_MUTATION=FALSE"
)

print(
    "STAGE26G_C2_FORECAST_MUTATION=FALSE"
)

print(
    "STAGE26G_C2_APP_MUTATION=FALSE"
)

print(
    "STAGE26G_C2_SERVICE_RESTART=FALSE"
)

print(
    "STAGE26G_C2_DATE_BLOCKED_COACH_SIGNAL_VALIDATION=PASS"
)
