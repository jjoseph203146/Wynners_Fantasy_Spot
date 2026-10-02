import math

import numpy as np
import pandas as pd

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection


# =========================================================
# CONFIGURATION
# =========================================================

POSITIONS = [
    "QB",
    "RB",
    "WR",
    "TE",
]

TRAIN_SEASON = 2023
TUNE_SEASON = 2024
TEST_SEASON = 2025

L2_VALUES = [
    0.0,
    0.01,
    0.10,
    1.0,
    10.0,
    100.0,
]

LEARNING_RATE = 0.05
MAX_ITERATIONS = 3000
TOLERANCE = 1e-8

# ---------------------------------------------------------
# Position-specific ceiling thresholds.
#
# These give us useful event counts while still measuring
# tournament-level upside.
# ---------------------------------------------------------

POSITION_THRESHOLDS = {

    "QB": [
        20.0,
        25.0,
        30.0,
    ],

    "RB": [
        15.0,
        20.0,
        25.0,
        30.0,
    ],

    "WR": [
        15.0,
        20.0,
        25.0,
        30.0,
    ],

    "TE": [
        10.0,
        15.0,
        20.0,
        25.0,
    ],
}


# =========================================================
# INPUTS
# =========================================================

CORE_FEATURES_CSV = (
    CSV_DIR / "nfl_core_projection_features.csv"
)

RIDGE_PREDICTIONS_CSV = (
    CSV_DIR / "nfl_projection_predictions_2025.csv"
)


# =========================================================
# OUTPUTS
# =========================================================

CEILING_TUNING_CSV = (
    CSV_DIR / "nfl_ceiling_probability_tuning.csv"
)

CEILING_TUNING_PARQUET = (
    PARQUET_DIR / "nfl_ceiling_probability_tuning.parquet"
)

CEILING_TEST_CSV = (
    CSV_DIR / "nfl_ceiling_probability_2025.csv"
)

CEILING_TEST_PARQUET = (
    PARQUET_DIR / "nfl_ceiling_probability_2025.parquet"
)

CEILING_SUMMARY_CSV = (
    CSV_DIR / "nfl_ceiling_probability_benchmark.csv"
)

CEILING_SUMMARY_PARQUET = (
    PARQUET_DIR / "nfl_ceiling_probability_benchmark.parquet"
)

CEILING_COEFFICIENTS_CSV = (
    CSV_DIR / "nfl_ceiling_probability_coefficients.csv"
)

CEILING_COEFFICIENTS_PARQUET = (
    PARQUET_DIR / "nfl_ceiling_probability_coefficients.parquet"
)


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sigmoid(values):

    values = np.clip(
        values,
        -35.0,
        35.0,
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -values
            )
        )
    )


def log_loss(
    actual,
    probability,
):

    probability = np.clip(
        probability,
        1e-12,
        1.0 - 1e-12,
    )

    return float(
        -np.mean(
            (
                actual
                *
                np.log(
                    probability
                )
            )
            +
            (
                (
                    1.0
                    -
                    actual
                )
                *
                np.log(
                    1.0
                    -
                    probability
                )
            )
        )
    )


def brier_score(
    actual,
    probability,
):

    return float(
        np.mean(
            (
                probability
                -
                actual
            )
            **
            2
        )
    )


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# =========================================================
# AUC
# =========================================================

def auc_score(
    actual,
    probability,
):

    actual = np.asarray(
        actual,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    positives = (
        actual
        ==
        1
    )

    negatives = (
        actual
        ==
        0
    )

    n_positive = int(
        positives.sum()
    )

    n_negative = int(
        negatives.sum()
    )

    if (
        n_positive == 0
        or
        n_negative == 0
    ):

        return 0.5

    ranks = pd.Series(
        probability
    ).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = float(
        ranks[
            positives
        ].sum()
    )

    auc = (
        positive_rank_sum
        -
        (
            n_positive
            *
            (
                n_positive
                +
                1
            )
            /
            2.0
        )
    ) / (
        n_positive
        *
        n_negative
    )

    return float(
        auc
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_matrix():

    section(
        "LOADING DFS FEATURE MATRIX"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM dfs_feature_matrix

            WHERE
                season IN (
                    2023,
                    2024,
                    2025
                )

                AND

                position IN (
                    'QB',
                    'RB',
                    'WR',
                    'TE'
                )

            ORDER BY
                season,
                week,
                game_id,
                position,
                player_id
            """,
            conn,
        )

    print(
        f"Rows loaded: "
        f"{len(df)}"
    )

    return df


def load_core_features():

    section(
        "LOADING FROZEN CORE FEATURES"
    )

    if not CORE_FEATURES_CSV.exists():

        raise RuntimeError(
            f"Missing: "
            f"{CORE_FEATURES_CSV}"
        )

    df = pd.read_csv(
        CORE_FEATURES_CSV
    )

    print(
        f"Core features loaded: "
        f"{len(df)}"
    )

    return df


def load_ridge_predictions():

    section(
        "LOADING 2025 RIDGE BENCHMARK"
    )

    if not RIDGE_PREDICTIONS_CSV.exists():

        raise RuntimeError(
            f"Missing: "
            f"{RIDGE_PREDICTIONS_CSV}"
        )

    df = pd.read_csv(
        RIDGE_PREDICTIONS_CSV
    )

    print(
        f"Ridge prediction rows: "
        f"{len(df)}"
    )

    return df


# =========================================================
# FEATURE PREPARATION
# =========================================================

def position_features(
    core_df,
    position,
):

    return (
        core_df[
            core_df[
                "position"
            ]
            ==
            position
        ]
        .sort_values(
            "core_rank"
        )[
            "feature"
        ]
        .tolist()
    )


def prepare_x(
    df,
    features,
):

    return (
        df[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )


def fit_standardizer(x):

    means = np.mean(
        x,
        axis=0,
    )

    stds = np.std(
        x,
        axis=0,
    )

    stds = np.where(
        stds < 1e-12,
        1.0,
        stds,
    )

    return (
        means,
        stds,
    )


def transform_standardizer(
    x,
    means,
    stds,
):

    return (
        x
        -
        means
    ) / stds


# =========================================================
# LOGISTIC MODEL
# =========================================================

def fit_logistic(
    x,
    y,
    l2_value,
):

    rows = x.shape[0]
    features = x.shape[1]

    beta = np.zeros(
        features,
        dtype=float,
    )

    positive_rate = float(
        np.mean(
            y
        )
    )

    positive_rate = min(
        max(
            positive_rate,
            1e-6,
        ),
        1.0 - 1e-6,
    )

    intercept = math.log(
        positive_rate
        /
        (
            1.0
            -
            positive_rate
        )
    )

    previous_loss = None

    for iteration in range(
        MAX_ITERATIONS
    ):

        linear = (
            intercept
            +
            x
            @
            beta
        )

        probability = sigmoid(
            linear
        )

        error = (
            probability
            -
            y
        )

        intercept_gradient = float(
            np.mean(
                error
            )
        )

        beta_gradient = (
            (
                x.T
                @
                error
            )
            /
            rows
        )

        beta_gradient = (
            beta_gradient
            +
            (
                l2_value
                /
                rows
            )
            *
            beta
        )

        intercept -= (
            LEARNING_RATE
            *
            intercept_gradient
        )

        beta -= (
            LEARNING_RATE
            *
            beta_gradient
        )

        if (
            iteration
            %
            25
            ==
            0
        ):

            probability = sigmoid(
                intercept
                +
                x
                @
                beta
            )

            current_loss = (
                log_loss(
                    y,
                    probability,
                )
                +
                (
                    l2_value
                    *
                    float(
                        beta
                        @
                        beta
                    )
                    /
                    (
                        2.0
                        *
                        rows
                    )
                )
            )

            if previous_loss is not None:

                if (
                    abs(
                        previous_loss
                        -
                        current_loss
                    )
                    <
                    TOLERANCE
                ):

                    break

            previous_loss = (
                current_loss
            )

    return (
        beta,
        intercept,
        iteration + 1,
    )


def predict_probability(
    x,
    beta,
    intercept,
):

    return sigmoid(
        intercept
        +
        x
        @
        beta
    )


# =========================================================
# TUNE MODEL
# =========================================================

def tune_model(
    matrix_df,
    core_df,
    position,
    threshold,
):

    features = position_features(
        core_df,
        position,
    )

    train = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            TRAIN_SEASON
        )
        &
        (
            matrix_df[
                "position"
            ]
            ==
            position
        )
    ].copy()

    validation = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            TUNE_SEASON
        )
        &
        (
            matrix_df[
                "position"
            ]
            ==
            position
        )
    ].copy()

    x_train = prepare_x(
        train,
        features,
    )

    x_validation = prepare_x(
        validation,
        features,
    )

    y_train = (
        pd.to_numeric(
            train[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
        >=
        threshold
    ).astype(float)

    y_validation = (
        pd.to_numeric(
            validation[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
        >=
        threshold
    ).astype(float)

    if (
        y_train.sum()
        ==
        0
        or
        y_validation.sum()
        ==
        0
    ):

        raise RuntimeError(
            f"Insufficient ceiling events: "
            f"{position} {threshold}"
        )

    (
        means,
        stds,
    ) = fit_standardizer(
        x_train
    )

    x_train = transform_standardizer(
        x_train,
        means,
        stds,
    )

    x_validation = transform_standardizer(
        x_validation,
        means,
        stds,
    )

    rows = []

    for l2_value in L2_VALUES:

        (
            beta,
            intercept,
            iterations,
        ) = fit_logistic(
            x_train,
            y_train,
            l2_value,
        )

        probability = (
            predict_probability(
                x_validation,
                beta,
                intercept,
            )
        )

        rows.append(
            {
                "position":
                    position,

                "threshold":
                    threshold,

                "l2_value":
                    l2_value,

                "train_rows":
                    len(
                        train
                    ),

                "validation_rows":
                    len(
                        validation
                    ),

                "train_events":
                    int(
                        y_train.sum()
                    ),

                "validation_events":
                    int(
                        y_validation.sum()
                    ),

                "validation_event_rate":
                    float(
                        y_validation.mean()
                    ),

                "log_loss":
                    log_loss(
                        y_validation,
                        probability,
                    ),

                "brier_score":
                    brier_score(
                        y_validation,
                        probability,
                    ),

                "auc":
                    auc_score(
                        y_validation,
                        probability,
                    ),

                "iterations":
                    iterations,
            }
        )

    result = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # Tune on LOG LOSS.
    #
    # Brier then smaller penalty break ties.
    # -----------------------------------------------------

    result = (
        result.sort_values(
            [
                "log_loss",
                "brier_score",
                "l2_value",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "selected"
    ] = 0

    result.loc[
        0,
        "selected",
    ] = 1

    best_l2 = float(
        result.iloc[0][
            "l2_value"
        ]
    )

    return (
        best_l2,
        result,
    )


# =========================================================
# FINAL MODEL
# =========================================================

def fit_final_model(
    matrix_df,
    core_df,
    position,
    threshold,
    l2_value,
):

    features = position_features(
        core_df,
        position,
    )

    development = matrix_df[
        (
            matrix_df[
                "season"
            ].isin(
                [
                    TRAIN_SEASON,
                    TUNE_SEASON,
                ]
            )
        )
        &
        (
            matrix_df[
                "position"
            ]
            ==
            position
        )
    ].copy()

    test = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            TEST_SEASON
        )
        &
        (
            matrix_df[
                "position"
            ]
            ==
            position
        )
    ].copy()

    x_dev = prepare_x(
        development,
        features,
    )

    x_test = prepare_x(
        test,
        features,
    )

    y_dev = (
        pd.to_numeric(
            development[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
        >=
        threshold
    ).astype(float)

    y_test = (
        pd.to_numeric(
            test[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
        >=
        threshold
    ).astype(float)

    (
        means,
        stds,
    ) = fit_standardizer(
        x_dev
    )

    x_dev_scaled = (
        transform_standardizer(
            x_dev,
            means,
            stds,
        )
    )

    x_test_scaled = (
        transform_standardizer(
            x_test,
            means,
            stds,
        )
    )

    (
        beta,
        intercept,
        iterations,
    ) = fit_logistic(
        x_dev_scaled,
        y_dev,
        l2_value,
    )

    probability = (
        predict_probability(
            x_test_scaled,
            beta,
            intercept,
        )
    )

    output = test[
        [
            "game_id",
            "season",
            "week",
            "player_id",
            "player_display_name",
            "position",
            "team",
            "opponent_team",
            "target_fanduel_points",
        ]
    ].copy()

    output[
        "threshold"
    ] = threshold

    output[
        "actual_ceiling"
    ] = y_test.astype(int)

    output[
        "ceiling_probability"
    ] = probability

    output[
        "l2_value"
    ] = l2_value

    coefficient_rows = []

    for index, feature in enumerate(
        features
    ):

        coefficient_rows.append(
            {
                "position":
                    position,

                "threshold":
                    threshold,

                "feature":
                    feature,

                "l2_value":
                    l2_value,

                "standardized_coefficient":
                    float(
                        beta[
                            index
                        ]
                    ),

                "training_mean":
                    float(
                        means[
                            index
                        ]
                    ),

                "training_std":
                    float(
                        stds[
                            index
                        ]
                    ),

                "intercept":
                    float(
                        intercept
                    ),

                "iterations":
                    iterations,
            }
        )

    return (
        output,
        pd.DataFrame(
            coefficient_rows
        ),
    )


# =========================================================
# JOIN RIDGE PROJECTION
# =========================================================

def attach_ridge(
    ceiling_df,
    ridge_df,
):

    ridge_small = ridge_df[
        [
            "game_id",
            "player_id",
            "team",
            "model_projection",
            "baseline_projection",
        ]
    ].copy()

    return ceiling_df.merge(
        ridge_small,
        how="left",
        on=[
            "game_id",
            "player_id",
            "team",
        ],
        validate="many_to_one",
    )


# =========================================================
# RANKING EVALUATION
# =========================================================

def evaluate_ranker(
    data,
    score_column,
    model_name,
):

    rows = []

    for (
        position,
        threshold
    ), group in data.groupby(
        [
            "position",
            "threshold",
        ]
    ):

        working = (
            group.copy()
        )

        working[
            "score_percentile"
        ] = (
            working.groupby(
                [
                    "season",
                    "week",
                ]
            )[
                score_column
            ]
            .rank(
                method="first",
                pct=True,
            )
        )

        top = working[
            working[
                "score_percentile"
            ]
            >
            0.80
        ]

        overall_rate = float(
            working[
                "actual_ceiling"
            ].mean()
        )

        top_rate = float(
            top[
                "actual_ceiling"
            ].mean()
        )

        if overall_rate > 0:

            lift = (
                top_rate
                /
                overall_rate
            )

        else:

            lift = 0.0

        ceiling_games = working[
            working[
                "actual_ceiling"
            ]
            ==
            1
        ]

        if len(
            ceiling_games
        ):

            capture = float(
                (
                    ceiling_games[
                        "score_percentile"
                    ]
                    >
                    0.80
                ).mean()
            )

        else:

            capture = 0.0

        rows.append(
            {
                "position":
                    position,

                "threshold":
                    threshold,

                "model":
                    model_name,

                "rows":
                    len(
                        working
                    ),

                "ceiling_events":
                    int(
                        working[
                            "actual_ceiling"
                        ].sum()
                    ),

                "overall_ceiling_rate":
                    overall_rate,

                "top_quintile_ceiling_rate":
                    top_rate,

                "ceiling_lift":
                    lift,

                "ceiling_capture_rate":
                    capture,

                "auc":
                    auc_score(
                        working[
                            "actual_ceiling"
                        ].to_numpy(),
                        working[
                            score_column
                        ].to_numpy(),
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# AUDIT
# =========================================================

def audit_predictions(
    df,
):

    section(
        "CEILING PROBABILITY AUDIT"
    )

    invalid_probability = int(
        (
            (
                df[
                    "ceiling_probability"
                ]
                <
                0
            )
            |
            (
                df[
                    "ceiling_probability"
                ]
                >
                1
            )
        ).sum()
    )

    null_probability = int(
        df[
            "ceiling_probability"
        ]
        .isna()
        .sum()
    )

    null_ridge = int(
        df[
            "model_projection"
        ]
        .isna()
        .sum()
    )

    duplicates = (
        df.groupby(
            [
                "game_id",
                "player_id",
                "team",
                "threshold",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    duplicates = duplicates[
        duplicates[
            "n"
        ]
        >
        1
    ]

    print(
        f"Prediction rows: "
        f"{len(df)}"
    )

    print(
        f"Invalid probabilities: "
        f"{invalid_probability}"
    )

    print(
        f"NULL probabilities: "
        f"{null_probability}"
    )

    print(
        f"NULL ridge projections: "
        f"{null_ridge}"
    )

    print(
        f"Duplicate player-threshold rows: "
        f"{len(duplicates)}"
    )

    problems = (
        invalid_probability
        +
        null_probability
        +
        null_ridge
        +
        len(
            duplicates
        )
    )

    if problems:

        raise RuntimeError(
            "Ceiling probability audit failed."
        )

    print()
    print(
        "PASS: ceiling probabilities "
        "passed structural audits."
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    summary_df,
):

    section(
        "2025 CEILING MODEL BENCHMARK"
    )

    print(
        summary_df[
            [
                "position",
                "threshold",
                "model",
                "ceiling_events",
                "top_quintile_ceiling_rate",
                "ceiling_lift",
                "ceiling_capture_rate",
                "auc",
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# EXPORT
# =========================================================

def export_results(
    tuning_df,
    predictions_df,
    summary_df,
    coefficients_df,
):

    section(
        "EXPORTING CEILING MODEL RESULTS"
    )

    tuning_df.to_csv(
        CEILING_TUNING_CSV,
        index=False,
    )

    tuning_df.to_parquet(
        CEILING_TUNING_PARQUET,
        index=False,
    )

    predictions_df.to_csv(
        CEILING_TEST_CSV,
        index=False,
    )

    predictions_df.to_parquet(
        CEILING_TEST_PARQUET,
        index=False,
    )

    summary_df.to_csv(
        CEILING_SUMMARY_CSV,
        index=False,
    )

    summary_df.to_parquet(
        CEILING_SUMMARY_PARQUET,
        index=False,
    )

    coefficients_df.to_csv(
        CEILING_COEFFICIENTS_CSV,
        index=False,
    )

    coefficients_df.to_parquet(
        CEILING_COEFFICIENTS_PARQUET,
        index=False,
    )

    print(
        f"Tuning rows: "
        f"{len(tuning_df)}"
    )

    print(
        f"Prediction rows: "
        f"{len(predictions_df)}"
    )

    print(
        f"Benchmark rows: "
        f"{len(summary_df)}"
    )

    print(
        f"Coefficient rows: "
        f"{len(coefficients_df)}"
    )

    print()

    print(
        f"Benchmark CSV: "
        f"{CEILING_SUMMARY_CSV}"
    )

    print(
        f"Predictions CSV: "
        f"{CEILING_TEST_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_ceiling_probability_engine():

    section(
        "NFL DFS CEILING PROBABILITY ENGINE"
    )

    matrix_df = load_matrix()

    core_df = load_core_features()

    ridge_df = load_ridge_predictions()

    tuning_frames = []
    prediction_frames = []
    coefficient_frames = []

    selected_l2 = {}

    # =====================================================
    # TUNE 2023 -> 2024
    # =====================================================

    section(
        "TUNING CEILING MODELS"
    )

    for position in POSITIONS:

        for threshold in (
            POSITION_THRESHOLDS[
                position
            ]
        ):

            (
                best_l2,
                tuning_df,
            ) = tune_model(
                matrix_df,
                core_df,
                position,
                threshold,
            )

            selected_l2[
                (
                    position,
                    threshold,
                )
            ] = best_l2

            tuning_frames.append(
                tuning_df
            )

            print(
                f"{position} "
                f"{threshold:.0f}+ "
                f"selected L2: "
                f"{best_l2}"
            )

    # =====================================================
    # REFIT 2023+2024 -> TEST 2025
    # =====================================================

    section(
        "FITTING FINAL CEILING MODELS"
    )

    for position in POSITIONS:

        for threshold in (
            POSITION_THRESHOLDS[
                position
            ]
        ):

            (
                prediction_df,
                coefficient_df,
            ) = fit_final_model(
                matrix_df,
                core_df,
                position,
                threshold,
                selected_l2[
                    (
                        position,
                        threshold,
                    )
                ],
            )

            prediction_frames.append(
                prediction_df
            )

            coefficient_frames.append(
                coefficient_df
            )

            print(
                f"{position} "
                f"{threshold:.0f}+: "
                f"{len(prediction_df)} "
                f"2025 predictions"
            )

    tuning_df = pd.concat(
        tuning_frames,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    coefficients_df = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    predictions_df = attach_ridge(
        predictions_df,
        ridge_df,
    )

    audit_predictions(
        predictions_df
    )

    # =====================================================
    # DIRECT CEILING MODEL
    # =====================================================

    direct_summary = evaluate_ranker(
        predictions_df,
        "ceiling_probability",
        "DIRECT_CEILING",
    )

    # =====================================================
    # EXISTING RIDGE PROJECTION
    # =====================================================

    ridge_summary = evaluate_ranker(
        predictions_df,
        "model_projection",
        "RIDGE_PROJECTION",
    )

    # =====================================================
    # SIMPLE BASELINE
    # =====================================================

    baseline_summary = evaluate_ranker(
        predictions_df,
        "baseline_projection",
        "RECENT_FD_BASELINE",
    )

    summary_df = pd.concat(
        [
            direct_summary,
            ridge_summary,
            baseline_summary,
        ],
        ignore_index=True,
    )

    summary_df = (
        summary_df.sort_values(
            [
                "position",
                "threshold",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print_summary(
        summary_df
    )

    export_results(
        tuning_df,
        predictions_df,
        summary_df,
        coefficients_df,
    )

    section(
        "CEILING PROBABILITY BUILD SUCCESSFUL"
    )

    print()
    print(
        "Chronology:"
    )

    print(
        "2023 -> train candidate ceiling models"
    )

    print(
        "2024 -> choose L2 penalty"
    )

    print(
        "2023+2024 -> refit"
    )

    print(
        "2025 -> untouched final ceiling test"
    )

    print()

    print(
        "Next decision:"
    )

    print(
        "Compare DIRECT_CEILING against "
        "RIDGE_PROJECTION for GPP ranking."
    )


if __name__ == "__main__":

    run_ceiling_probability_engine()
