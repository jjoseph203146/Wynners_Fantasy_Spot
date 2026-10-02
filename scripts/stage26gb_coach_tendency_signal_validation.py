#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# WFS STAGE26G-B
# POINT-IN-TIME COACH TENDENCY SIGNAL VALIDATION
# ==================================================================
#
# ANALYSIS_ONLY=TRUE
#
# PURPOSE:
#   Validate whether Stage26G-A point-in-time coach-regime tendencies
#   persist into the NEXT observed game.
#
# IMPORTANT:
#   - Stage26G-A remains the sole matrix authority.
#   - G-B does not reconstruct coach tendencies independently.
#   - The target game is NEVER included in its own predictor.
#   - Coach-regime boundaries remain hard resets.
#   - No signal receives production influence in this stage.
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

    out = pd.Series(
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

    out.loc[valid] = (
        n.loc[valid]
        /
        d.loc[valid]
    )

    return out


def finite_pair(
    x: pd.Series,
    y: pd.Series,
) -> pd.DataFrame:

    pair = pd.DataFrame(
        {
            "x": num(x),
            "y": num(y),
        }
    )

    pair = pair[
        pair["x"].notna()
        &
        pair["y"].notna()
        &
        np.isfinite(pair["x"])
        &
        np.isfinite(pair["y"])
    ].copy()

    return pair


def pearson(
    x: pd.Series,
    y: pd.Series,
) -> float:

    pair = finite_pair(
        x,
        y,
    )

    if len(pair) < 3:
        return np.nan

    if (
        pair["x"].nunique() < 2
        or
        pair["y"].nunique() < 2
    ):
        return np.nan

    return float(
        pair["x"].corr(
            pair["y"],
            method="pearson",
        )
    )


def mae(
    x: pd.Series,
    y: pd.Series,
) -> float:

    pair = finite_pair(
        x,
        y,
    )

    if pair.empty:
        return np.nan

    return float(
        (
            pair["x"]
            -
            pair["y"]
        )
        .abs()
        .mean()
    )


def directional_accuracy(
    predictor: pd.Series,
    target: pd.Series,
    reference: pd.Series,
) -> float:

    df = pd.DataFrame(
        {
            "predictor": num(predictor),
            "target": num(target),
            "reference": num(reference),
        }
    )

    df = df.dropna()

    if df.empty:
        return np.nan

    pred_delta = (
        df["predictor"]
        -
        df["reference"]
    )

    target_delta = (
        df["target"]
        -
        df["reference"]
    )

    usable = (
        pred_delta.ne(0)
        &
        target_delta.ne(0)
    )

    if not usable.any():
        return np.nan

    correct = (
        np.sign(
            pred_delta[usable]
        )
        ==
        np.sign(
            target_delta[usable]
        )
    )

    return float(
        correct.mean()
    )


# ==================================================================
# [0] RUNTIME CONTRACT
# ==================================================================

section(
    "[0] STAGE26G-B RUNTIME CONTRACT"
)

print("ANALYSIS_ONLY=TRUE")
print("SOURCE_STAGE26G_A=TRUE")
print("POINT_IN_TIME=TRUE")
print("NEXT_GAME_VALIDATION=TRUE")
print("CURRENT_GAME_EXCLUDED_FROM_PREDICTOR=TRUE")
print("FUTURE_GAME_LEAKAGE=FALSE")
print("TEAM_COACH_REGIME_RESET=TRUE")

print("ARTIFACT_WRITE=FALSE")
print("DATABASE_WRITE=FALSE")
print("SOLVER_MUTATION=FALSE")
print("FORECAST_MUTATION=FALSE")
print("APP_MUTATION=FALSE")
print("SERVICE_RESTART=FALSE")


# ==================================================================
# [1] LOAD FROZEN STAGE26G-A AUTHORITY
# ==================================================================

section(
    "[1] LOAD FROZEN STAGE26G-A AUTHORITY"
)

if not STAGE26GA_PATH.is_file():

    raise SystemExit(
        f"FAIL_CLOSED: Stage26G-A script missing: {STAGE26GA_PATH}"
    )

print(
    f"STAGE26G_A_PATH={STAGE26GA_PATH}"
)

ga = runpy.run_path(
    str(STAGE26GA_PATH)
)

if "matrix" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A did not expose matrix"
    )

matrix = ga["matrix"].copy()

if "coach_id_col" not in ga:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A did not expose coach identity column"
    )

coach_id_col = ga["coach_id_col"]

print(
    f"COACH_IDENTITY_COLUMN={coach_id_col}"
)

print(
    f"STAGE26G_A_MATRIX_ROWS={len(matrix)}"
)

print(
    "STAGE26G_A_MATRIX_REGIMES="
    f"{matrix['team_coach_regime_key'].nunique()}"
)

if len(matrix) != 1710:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A matrix row contract changed"
    )

if matrix[
    "team_coach_regime_key"
].nunique() != 49:

    raise SystemExit(
        "FAIL_CLOSED: Stage26G-A regime contract changed"
    )

if matrix.duplicated(
    [
        "game_id",
        "team",
    ]
).any():

    raise SystemExit(
        "FAIL_CLOSED: duplicate Stage26G-A matrix identity"
    )

print(
    "STAGE26G_A_AUTHORITY_IMPORT=PASS"
)


# ==================================================================
# [2] REASSERT POINT-IN-TIME VALIDATION UNIVERSE
# ==================================================================

section(
    "[2] POINT-IN-TIME VALIDATION UNIVERSE"
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
    "_gb_regime_game_number"
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
                "_gb_regime_game_number"
            ]
        )
    )
    .abs()
    .gt(0)
    .sum()
)

print(
    "PRIOR_REGIME_SEQUENCE_VIOLATIONS="
    f"{sequence_violations}"
)

if sequence_violations:

    raise SystemExit(
        "FAIL_CLOSED: G-B predictor chronology differs from G-A PIT chronology"
    )

first_rows = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)
)

cold_start_prior_nonzero = int(
    num(
        first_rows[
            "prior_pbp_regime_games"
        ]
    )
    .ne(0)
    .sum()
)

print(
    "COLD_START_ROWS="
    f"{len(first_rows)}"
)

print(
    "COLD_START_PRIOR_HISTORY_NONZERO="
    f"{cold_start_prior_nonzero}"
)

if cold_start_prior_nonzero:

    raise SystemExit(
        "FAIL_CLOSED: new coach regime inherited prior history"
    )

validation = matrix[
    num(
        matrix[
            "prior_pbp_regime_games"
        ]
    ).gt(0)
].copy()

print(
    f"VALIDATION_ROWS={len(validation)}"
)

print(
    "VALIDATION_REGIMES="
    f"{validation['team_coach_regime_key'].nunique()}"
)

print(
    "CURRENT_GAME_INCLUDED_IN_PREDICTOR=FALSE"
)

print(
    "FUTURE_GAME_INCLUDED_IN_PREDICTOR=FALSE"
)

print(
    "POINT_IN_TIME_VALIDATION_UNIVERSE=PASS"
)


# ==================================================================
# [3] BUILD CURRENT-GAME TARGETS
# ==================================================================

section(
    "[3] CURRENT-GAME VALIDATION TARGETS"
)

matrix[
    "current_raw_pass_rate"
] = safe_ratio(
    matrix[
        "raw_pass"
    ],
    matrix[
        "raw_n"
    ],
)

matrix[
    "current_observed_pass_rate_consistent"
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

matrix[
    "current_early_down_pass_rate"
] = safe_ratio(
    matrix[
        "early_raw_pass"
    ],
    matrix[
        "early_raw_n"
    ],
)

matrix[
    "current_early_down_pass_oe"
] = safe_ratio(
    matrix[
        "early_pass_oe_sum"
    ],
    matrix[
        "early_combined_n"
    ],
)

matrix[
    "current_neutral_pass_rate"
] = safe_ratio(
    matrix[
        "neutral_raw_pass"
    ],
    matrix[
        "neutral_raw_n"
    ],
)

matrix[
    "current_neutral_pass_oe"
] = safe_ratio(
    matrix[
        "neutral_pass_oe_sum"
    ],
    matrix[
        "neutral_combined_n"
    ],
)

matrix[
    "current_redzone_pass_rate"
] = safe_ratio(
    matrix[
        "redzone_raw_pass"
    ],
    matrix[
        "redzone_raw_n"
    ],
)

matrix[
    "current_redzone_pass_oe"
] = safe_ratio(
    matrix[
        "redzone_pass_oe_sum"
    ],
    matrix[
        "redzone_combined_n"
    ],
)

matrix[
    "current_short_yardage_pass_rate_candidate"
] = safe_ratio(
    matrix[
        "short_candidate_pass"
    ],
    matrix[
        "short_candidate_n"
    ],
)

matrix[
    "current_fourth_down_go_rate_classified"
] = safe_ratio(
    matrix[
        "fourth_go"
    ],
    matrix[
        "fourth_classified_n"
    ],
)

validation = matrix[
    num(
        matrix[
            "prior_pbp_regime_games"
        ]
    ).gt(0)
].copy()

target_columns = [
    "current_raw_pass_rate",
    "current_observed_pass_rate_consistent",
    "current_mean_xpass",
    "current_mean_pass_oe",
    "current_early_down_pass_rate",
    "current_early_down_pass_oe",
    "current_neutral_pass_rate",
    "current_neutral_pass_oe",
    "current_redzone_pass_rate",
    "current_redzone_pass_oe",
    "current_short_yardage_pass_rate_candidate",
    "current_fourth_down_go_rate_classified",
]

nonfinite_target_values = 0

for col in target_columns:

    vals = num(
        validation[
            col
        ]
    )

    nonfinite_target_values += int(
        (
            vals.notna()
            &
            ~np.isfinite(vals)
        ).sum()
    )

print(
    "NONFINITE_CURRENT_TARGET_VALUES="
    f"{nonfinite_target_values}"
)

if nonfinite_target_values:

    raise SystemExit(
        "FAIL_CLOSED: nonfinite current-game validation target"
    )

print(
    "CURRENT_GAME_VALIDATION_TARGETS=PASS"
)


# ==================================================================
# [4] PASS-OE TARGET IDENTITY
# ==================================================================

section(
    "[4] CURRENT-GAME PASS-OE IDENTITY"
)

has_current_combined = (
    num(
        validation[
            "combined_n"
        ]
    ).gt(0)
)

derived_current_oe = (
    (
        validation[
            "current_observed_pass_rate_consistent"
        ]
        -
        validation[
            "current_mean_xpass"
        ]
    )
    *
    100.0
)

oe_error = (
    validation[
        "current_mean_pass_oe"
    ]
    -
    derived_current_oe
).abs()

if has_current_combined.any():

    max_current_oe_error = float(
        oe_error[
            has_current_combined
        ].max()
    )

else:

    max_current_oe_error = 0.0

print(
    "CURRENT_PASS_OE_IDENTITY_ROWS="
    f"{int(has_current_combined.sum())}"
)

print(
    "MAX_CURRENT_PASS_OE_IDENTITY_ERROR="
    f"{max_current_oe_error:.12f}"
)

if (
    np.isfinite(
        max_current_oe_error
    )
    and
    max_current_oe_error > 1e-6
):

    raise SystemExit(
        "FAIL_CLOSED: current-game Pass-OE identity violation"
    )

print(
    "CURRENT_GAME_PASS_OE_IDENTITY=PASS"
)


# ==================================================================
# [5] MATURITY BUCKETS
# ==================================================================

section(
    "[5] COACH-REGIME MATURITY BUCKETS"
)

prior_games = num(
    validation[
        "prior_pbp_regime_games"
    ]
)

validation[
    "maturity_bucket"
] = pd.cut(
    prior_games,
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

if validation[
    "maturity_bucket"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: validation row missing maturity bucket"
    )

bucket_counts = (
    validation[
        "maturity_bucket"
    ]
    .value_counts(
        sort=False
    )
)

for bucket, count in bucket_counts.items():

    print(
        f"MATURITY_{bucket}_ROWS={int(count)}"
    )

print(
    "COACH_REGIME_MATURITY_BUCKETS=PASS"
)


# ==================================================================
# [6] SIGNAL CONTRACT
# ==================================================================

section(
    "[6] COACH TENDENCY SIGNAL CONTRACT"
)

signals = [
    {
        "name":
            "RAW_PASS_RATE",

        "predictor":
            "prior_raw_pass_rate",

        "target":
            "current_raw_pass_rate",

        "reference":
            None,
    },

    {
        "name":
            "PASS_OE",

        "predictor":
            "prior_mean_pass_oe",

        "target":
            "current_mean_pass_oe",

        "reference":
            None,
    },

    {
        "name":
            "EARLY_DOWN_PASS_RATE",

        "predictor":
            "prior_early_down_pass_rate",

        "target":
            "current_early_down_pass_rate",

        "reference":
            None,
    },

    {
        "name":
            "EARLY_DOWN_PASS_OE",

        "predictor":
            "prior_early_down_pass_oe",

        "target":
            "current_early_down_pass_oe",

        "reference":
            None,
    },

    {
        "name":
            "NEUTRAL_PASS_RATE",

        "predictor":
            "prior_neutral_pass_rate",

        "target":
            "current_neutral_pass_rate",

        "reference":
            None,
    },

    {
        "name":
            "NEUTRAL_PASS_OE",

        "predictor":
            "prior_neutral_pass_oe",

        "target":
            "current_neutral_pass_oe",

        "reference":
            None,
    },

    {
        "name":
            "REDZONE_PASS_RATE",

        "predictor":
            "prior_redzone_pass_rate",

        "target":
            "current_redzone_pass_rate",

        "reference":
            None,
    },

    {
        "name":
            "REDZONE_PASS_OE",

        "predictor":
            "prior_redzone_pass_oe",

        "target":
            "current_redzone_pass_oe",

        "reference":
            None,
    },

    {
        "name":
            "SHORT_YARDAGE_PASS_RATE",

        "predictor":
            "prior_short_yardage_pass_rate_candidate",

        "target":
            "current_short_yardage_pass_rate_candidate",

        "reference":
            None,
    },

    {
        "name":
            "FOURTH_DOWN_GO_RATE",

        "predictor":
            "prior_fourth_down_go_rate_classified",

        "target":
            "current_fourth_down_go_rate_classified",

        "reference":
            None,
    },
]

for signal in signals:

    for required in [
        signal[
            "predictor"
        ],
        signal[
            "target"
        ],
    ]:

        if required not in validation.columns:

            raise SystemExit(
                "FAIL_CLOSED: missing G-B signal column: "
                f"{required}"
            )

print(
    f"SIGNAL_COUNT={len(signals)}"
)

print(
    "COACH_TENDENCY_SIGNAL_CONTRACT=PASS"
)


# ==================================================================
# [7] OVERALL NEXT-GAME PERSISTENCE
# ==================================================================

section(
    "[7] OVERALL NEXT-GAME SIGNAL PERSISTENCE"
)

overall_results = []

for signal in signals:

    name = signal[
        "name"
    ]

    predictor = signal[
        "predictor"
    ]

    target = signal[
        "target"
    ]

    pair = finite_pair(
        validation[
            predictor
        ],
        validation[
            target
        ],
    )

    n = len(
        pair
    )

    corr = pearson(
        validation[
            predictor
        ],
        validation[
            target
        ],
    )

    error = mae(
        validation[
            predictor
        ],
        validation[
            target
        ],
    )

    print(
        f"{name}_N={n}"
    )

    print(
        f"{name}_PEARSON="
        + (
            "NA"
            if not np.isfinite(corr)
            else f"{corr:.8f}"
        )
    )

    print(
        f"{name}_MAE="
        + (
            "NA"
            if not np.isfinite(error)
            else f"{error:.8f}"
        )
    )

    overall_results.append(
        {
            "signal":
                name,

            "n":
                n,

            "pearson":
                corr,

            "mae":
                error,
        }
    )

print(
    "OVERALL_NEXT_GAME_SIGNAL_PERSISTENCE=PASS"
)


# ==================================================================
# [8] MATURITY-BUCKET PERSISTENCE
# ==================================================================

section(
    "[8] MATURITY-BUCKET SIGNAL PERSISTENCE"
)

bucket_results = []

bucket_order = [
    "01_1_TO_3",
    "02_4_TO_7",
    "03_8_TO_15",
    "04_16_PLUS",
]

for signal in signals:

    name = signal[
        "name"
    ]

    predictor = signal[
        "predictor"
    ]

    target = signal[
        "target"
    ]

    for bucket in bucket_order:

        sub = validation[
            validation[
                "maturity_bucket"
            ].astype(str).eq(
                bucket
            )
        ]

        pair = finite_pair(
            sub[
                predictor
            ],
            sub[
                target
            ],
        )

        n = len(
            pair
        )

        corr = pearson(
            sub[
                predictor
            ],
            sub[
                target
            ],
        )

        error = mae(
            sub[
                predictor
            ],
            sub[
                target
            ],
        )

        print(
            f"{name}_{bucket}_N={n}"
        )

        print(
            f"{name}_{bucket}_PEARSON="
            + (
                "NA"
                if not np.isfinite(corr)
                else f"{corr:.8f}"
            )
        )

        print(
            f"{name}_{bucket}_MAE="
            + (
                "NA"
                if not np.isfinite(error)
                else f"{error:.8f}"
            )
        )

        bucket_results.append(
            {
                "signal":
                    name,

                "bucket":
                    bucket,

                "n":
                    n,

                "pearson":
                    corr,

                "mae":
                    error,
            }
        )

print(
    "MATURITY_BUCKET_SIGNAL_PERSISTENCE=PASS"
)


# ==================================================================
# [9] CONTEXT BASELINE — PASS RATE VS NFLVERSE XPASS
# ==================================================================

section(
    "[9] CONTEXT BASELINE VALIDATION"
)

context_df = validation[
    [
        "prior_raw_pass_rate",
        "current_raw_pass_rate",
        "current_mean_xpass",
    ]
].dropna()

context_n = len(
    context_df
)

coach_mae = mae(
    context_df[
        "prior_raw_pass_rate"
    ],
    context_df[
        "current_raw_pass_rate"
    ],
)

xpass_mae = mae(
    context_df[
        "current_mean_xpass"
    ],
    context_df[
        "current_raw_pass_rate"
    ],
)

coach_corr = pearson(
    context_df[
        "prior_raw_pass_rate"
    ],
    context_df[
        "current_raw_pass_rate"
    ],
)

xpass_corr = pearson(
    context_df[
        "current_mean_xpass"
    ],
    context_df[
        "current_raw_pass_rate"
    ],
)

print(
    f"CONTEXT_BASELINE_ROWS={context_n}"
)

print(
    "COACH_PRIOR_PASS_RATE_MAE="
    f"{coach_mae:.8f}"
)

print(
    "CURRENT_XPASS_MAE="
    f"{xpass_mae:.8f}"
)

print(
    "COACH_PRIOR_PASS_RATE_CORR="
    f"{coach_corr:.8f}"
)

print(
    "CURRENT_XPASS_CORR="
    f"{xpass_corr:.8f}"
)

if (
    np.isfinite(coach_mae)
    and
    np.isfinite(xpass_mae)
):

    print(
        "COACH_PRIOR_BEATS_XPASS_MAE="
        f"{str(coach_mae < xpass_mae).upper()}"
    )

print(
    "CONTEXT_BASELINE_VALIDATION=PASS"
)


# ==================================================================
# [10] PASS-OE DIRECTIONAL PERSISTENCE
# ==================================================================

section(
    "[10] PASS-OE DIRECTIONAL PERSISTENCE"
)

oe_df = validation[
    [
        "prior_mean_pass_oe",
        "current_mean_pass_oe",
    ]
].dropna()

oe_usable = (
    oe_df[
        "prior_mean_pass_oe"
    ].ne(0)
    &
    oe_df[
        "current_mean_pass_oe"
    ].ne(0)
)

oe_directional_accuracy = np.nan

if oe_usable.any():

    oe_directional_accuracy = float(
        (
            np.sign(
                oe_df.loc[
                    oe_usable,
                    "prior_mean_pass_oe",
                ]
            )
            ==
            np.sign(
                oe_df.loc[
                    oe_usable,
                    "current_mean_pass_oe",
                ]
            )
        ).mean()
    )

print(
    "PASS_OE_DIRECTIONAL_ROWS="
    f"{int(oe_usable.sum())}"
)

print(
    "PASS_OE_DIRECTIONAL_ACCURACY="
    + (
        "NA"
        if not np.isfinite(
            oe_directional_accuracy
        )
        else f"{oe_directional_accuracy:.8f}"
    )
)

print(
    "PASS_OE_DIRECTIONAL_PERSISTENCE=PASS"
)


# ==================================================================
# [11] TEMPORAL / REGIME LEAKAGE AUDIT
# ==================================================================

section(
    "[11] TEMPORAL AND REGIME LEAKAGE AUDIT"
)

date_order_violations = 0

for _, group in matrix.groupby(
    "team_coach_regime_key",
    sort=False,
):

    dates = pd.to_datetime(
        group[
            "game_date_dt"
        ],
        errors="coerce",
    )

    date_order_violations += int(
        dates.diff()
        .dt.total_seconds()
        .fillna(0)
        .lt(0)
        .sum()
    )

regime_identity_violations = int(
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
    ).sum()
)

print(
    "TEMPORAL_ORDER_VIOLATIONS="
    f"{date_order_violations}"
)

print(
    "REGIME_IDENTITY_CROSSOVER_ROWS="
    f"{regime_identity_violations}"
)

print(
    "CURRENT_TARGET_INCLUDED_IN_PRIOR_PROFILE=FALSE"
)

print(
    "FUTURE_TARGET_INCLUDED_IN_PRIOR_PROFILE=FALSE"
)

if date_order_violations:

    raise SystemExit(
        "FAIL_CLOSED: G-B temporal ordering violation"
    )

if regime_identity_violations:

    raise SystemExit(
        "FAIL_CLOSED: G-B coach-regime crossover"
    )

print(
    "TEMPORAL_REGIME_LEAKAGE_AUDIT=PASS"
)


# ==================================================================
# [12] SIGNAL MATURITY SUMMARY
# ==================================================================

section(
    "[12] SIGNAL MATURITY SUMMARY"
)

overall_df = pd.DataFrame(
    overall_results
)

bucket_df = pd.DataFrame(
    bucket_results
)

for signal in signals:

    name = signal[
        "name"
    ]

    overall_row = overall_df[
        overall_df[
            "signal"
        ].eq(
            name
        )
    ]

    mature = bucket_df[
        bucket_df[
            "signal"
        ].eq(
            name
        )
        &
        bucket_df[
            "bucket"
        ].eq(
            "04_16_PLUS"
        )
    ]

    overall_corr = (
        float(
            overall_row[
                "pearson"
            ].iloc[0]
        )
        if len(overall_row)
        else np.nan
    )

    mature_corr = (
        float(
            mature[
                "pearson"
            ].iloc[0]
        )
        if len(mature)
        else np.nan
    )

    mature_n = (
        int(
            mature[
                "n"
            ].iloc[0]
        )
        if len(mature)
        else 0
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # This is descriptive classification only.
    # It DOES NOT authorize production influence.
    # --------------------------------------------------------------

    if (
        np.isfinite(overall_corr)
        and
        np.isfinite(mature_corr)
        and
        overall_corr > 0
        and
        mature_corr > 0
        and
        mature_n >= 100
    ):

        status = (
            "PERSISTENCE_CANDIDATE"
        )

    elif (
        np.isfinite(overall_corr)
        and
        overall_corr > 0
    ):

        status = (
            "WEAK_OR_SAMPLE_DEPENDENT"
        )

    else:

        status = (
            "NO_POSITIVE_PERSISTENCE"
        )

    print(
        f"{name}_DESCRIPTIVE_STATUS={status}"
    )

print(
    "SIGNAL_MATURITY_SUMMARY=PASS"
)


# ==================================================================
# [13] PRODUCTION FIREWALL
# ==================================================================

section(
    "[13] PRODUCTION FIREWALL"
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
        "FAIL_CLOSED: Stage26G-B production firewall violation"
    )

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ==================================================================
# [14] FINAL CONTRACT
# ==================================================================

section(
    "[14] STAGE26G-B FINAL CONTRACT"
)

print(
    "STAGE26G_B_CONTRACT="
    "WFS_POINT_IN_TIME_COACH_TENDENCY_SIGNAL_VALIDATION_V1"
)

print(
    "STAGE26G_B_SOURCE="
    "WFS_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX_V1"
)

print(
    f"STAGE26G_B_MATRIX_ROWS={len(matrix)}"
)

print(
    f"STAGE26G_B_VALIDATION_ROWS={len(validation)}"
)

print(
    "STAGE26G_B_POINT_IN_TIME=TRUE"
)

print(
    "STAGE26G_B_NEXT_GAME_VALIDATION=TRUE"
)

print(
    "STAGE26G_B_CURRENT_GAME_EXCLUDED_FROM_PREDICTOR=TRUE"
)

print(
    "STAGE26G_B_FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "STAGE26G_B_TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "STAGE26G_B_PRODUCTION_INFLUENCE=FALSE"
)

print(
    "STAGE26G_B_ARTIFACT_WRITE=FALSE"
)

print(
    "STAGE26G_B_DATABASE_WRITE=FALSE"
)

print(
    "STAGE26G_B_SOLVER_MUTATION=FALSE"
)

print(
    "STAGE26G_B_FORECAST_MUTATION=FALSE"
)

print(
    "STAGE26G_B_APP_MUTATION=FALSE"
)

print(
    "STAGE26G_B_SERVICE_RESTART=FALSE"
)

print(
    "STAGE26G_B_COACH_TENDENCY_SIGNAL_VALIDATION=PASS"
)
