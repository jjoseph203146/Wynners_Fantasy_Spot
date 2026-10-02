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

TUNING_TRAIN_SEASON = 2023
TUNING_VALIDATION_SEASON = 2024
FINAL_TEST_SEASON = 2025

RIDGE_LAMBDAS = [
    0.0,
    0.01,
    0.10,
    1.0,
    10.0,
    100.0,
]

CEILING_THRESHOLDS = [
    15.0,
    20.0,
    25.0,
    30.0,
]

MIN_HISTORY_ESTABLISHED = 3


# =========================================================
# INPUT PATH
# =========================================================

CORE_FEATURES_CSV = (
    CSV_DIR / "nfl_core_projection_features.csv"
)


# =========================================================
# OUTPUT PATHS
# =========================================================

BENCHMARK_CSV = (
    CSV_DIR / "nfl_projection_benchmark.csv"
)

BENCHMARK_PARQUET = (
    PARQUET_DIR / "nfl_projection_benchmark.parquet"
)

RIDGE_TUNING_CSV = (
    CSV_DIR / "nfl_projection_ridge_tuning.csv"
)

RIDGE_TUNING_PARQUET = (
    PARQUET_DIR / "nfl_projection_ridge_tuning.parquet"
)

PREDICTIONS_CSV = (
    CSV_DIR / "nfl_projection_predictions_2025.csv"
)

PREDICTIONS_PARQUET = (
    PARQUET_DIR / "nfl_projection_predictions_2025.parquet"
)

CEILING_CSV = (
    CSV_DIR / "nfl_projection_ceiling_benchmark.csv"
)

CEILING_PARQUET = (
    PARQUET_DIR / "nfl_projection_ceiling_benchmark.parquet"
)

MODEL_COEFFICIENTS_CSV = (
    CSV_DIR / "nfl_projection_model_coefficients.csv"
)

MODEL_COEFFICIENTS_PARQUET = (
    PARQUET_DIR / "nfl_projection_model_coefficients.parquet"
)


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_corr(
    actual,
    predicted,
):

    temp = pd.DataFrame(
        {
            "actual": safe_numeric(
                actual
            ),
            "predicted": safe_numeric(
                predicted
            ),
        }
    ).dropna()

    if len(temp) < 3:

        return 0.0

    if (
        temp["actual"].nunique() <= 1
        or
        temp["predicted"].nunique() <= 1
    ):

        return 0.0

    value = temp[
        "actual"
    ].corr(
        temp[
            "predicted"
        ]
    )

    if pd.isna(value):

        return 0.0

    return float(value)


def mae(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual
                -
                predicted
            )
        )
    )


def rmse(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    actual
                    -
                    predicted
                )
                **
                2
            )
        )
    )


def bias(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            predicted
            -
            actual
        )
    )


# =========================================================
# LOAD MATRIX
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

    print()

    print(
        df.groupby(
            [
                "season",
                "position",
            ]
        ).size().to_string()
    )

    return df


# =========================================================
# LOAD CORE FEATURES
# =========================================================

def load_core_features():

    section(
        "LOADING FROZEN CORE FEATURES"
    )

    if not CORE_FEATURES_CSV.exists():

        raise RuntimeError(
            "Missing core feature file: "
            f"{CORE_FEATURES_CSV}"
        )

    df = pd.read_csv(
        CORE_FEATURES_CSV
    )

    required = [
        "position",
        "core_rank",
        "feature",
        "development_class",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Core feature file missing: "
            +
            ", ".join(
                missing
            )
        )

    print(
        f"Core feature rows: "
        f"{len(df)}"
    )

    print()

    for position in POSITIONS:

        features = (
            df[
                df[
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

        print(
            f"{position}: "
            f"{len(features)} features"
        )

        for feature in features:

            print(
                f"  {feature}"
            )

    return df


# =========================================================
# VALIDATE CORE FEATURES
# =========================================================

def validate_core_features(
    matrix_df,
    core_df,
):

    section(
        "VALIDATING FROZEN CORE FEATURES"
    )

    missing = []

    for _, row in core_df.iterrows():

        feature = row[
            "feature"
        ]

        if feature not in matrix_df.columns:

            missing.append(
                (
                    row[
                        "position"
                    ],
                    feature,
                )
            )

    if missing:

        for position, feature in missing:

            print(
                f"MISSING: "
                f"{position} | "
                f"{feature}"
            )

        raise RuntimeError(
            "Projection benchmark stopped "
            "because frozen features are missing."
        )

    print(
        "PASS: all frozen core features "
        "exist in the matrix."
    )


# =========================================================
# BASELINE FEATURE
# =========================================================

def choose_baseline_feature(
    core_df,
    position,
):

    position_df = (
        core_df[
            core_df[
                "position"
            ]
            ==
            position
        ]
        .sort_values(
            "core_rank"
        )
    )

    preferred = [
        "fd_avg_3",
        "fd_avg_5",
        "fd_last",
    ]

    available = set(
        position_df[
            "feature"
        ]
    )

    for feature in preferred:

        if feature in available:

            return feature

    raise RuntimeError(
        f"No recent-FD baseline feature "
        f"available for {position}."
    )


# =========================================================
# MATRIX PREPARATION
# =========================================================

def prepare_xy(
    df,
    features,
):

    x = (
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

    y = (
        pd.to_numeric(
            df[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    return (
        x,
        y,
    )


# =========================================================
# STANDARDIZATION
# =========================================================

def fit_standardizer(
    x,
):

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
# RIDGE MODEL
# =========================================================

def fit_ridge(
    x,
    y,
    ridge_lambda,
):

    # -----------------------------------------------------
    # X is already standardized.
    #
    # y is centered so the intercept is not penalized.
    # -----------------------------------------------------

    y_mean = float(
        np.mean(
            y
        )
    )

    centered_y = (
        y
        -
        y_mean
    )

    n_features = (
        x.shape[1]
    )

    xtx = (
        x.T
        @
        x
    )

    penalty = (
        ridge_lambda
        *
        np.eye(
            n_features
        )
    )

    beta = (
        np.linalg.pinv(
            xtx
            +
            penalty
        )
        @
        x.T
        @
        centered_y
    )

    return (
        beta,
        y_mean,
    )


def predict_ridge(
    x,
    beta,
    intercept,
):

    return (
        intercept
        +
        x
        @
        beta
    )


# =========================================================
# RIDGE TUNING
# =========================================================

def tune_position_model(
    matrix_df,
    core_df,
    position,
):

    section(
        f"TUNING {position} RIDGE MODEL"
    )

    features = (
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

    train_df = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            TUNING_TRAIN_SEASON
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

    validation_df = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            TUNING_VALIDATION_SEASON
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

    (
        x_train,
        y_train,
    ) = prepare_xy(
        train_df,
        features,
    )

    (
        x_validation,
        y_validation,
    ) = prepare_xy(
        validation_df,
        features,
    )

    (
        means,
        stds,
    ) = fit_standardizer(
        x_train
    )

    x_train_scaled = (
        transform_standardizer(
            x_train,
            means,
            stds,
        )
    )

    x_validation_scaled = (
        transform_standardizer(
            x_validation,
            means,
            stds,
        )
    )

    rows = []

    for ridge_lambda in RIDGE_LAMBDAS:

        (
            beta,
            intercept,
        ) = fit_ridge(
            x_train_scaled,
            y_train,
            ridge_lambda,
        )

        predictions = predict_ridge(
            x_validation_scaled,
            beta,
            intercept,
        )

        rows.append(
            {
                "position":
                    position,

                "train_season":
                    TUNING_TRAIN_SEASON,

                "validation_season":
                    TUNING_VALIDATION_SEASON,

                "ridge_lambda":
                    ridge_lambda,

                "features":
                    len(
                        features
                    ),

                "validation_rows":
                    len(
                        validation_df
                    ),

                "mae":
                    mae(
                        y_validation,
                        predictions,
                    ),

                "rmse":
                    rmse(
                        y_validation,
                        predictions,
                    ),

                "correlation":
                    safe_corr(
                        y_validation,
                        predictions,
                    ),

                "bias":
                    bias(
                        y_validation,
                        predictions,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # Primary selection metric = RMSE.
    #
    # MAE is deterministic tiebreaker, followed by the
    # smaller ridge penalty.
    # -----------------------------------------------------

    result = (
        result.sort_values(
            [
                "rmse",
                "mae",
                "ridge_lambda",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "selected"
    ] = 0

    result.loc[
        result.index[0],
        "selected",
    ] = 1

    best_lambda = float(
        result.iloc[0][
            "ridge_lambda"
        ]
    )

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        f"Selected {position} lambda: "
        f"{best_lambda}"
    )

    return (
        best_lambda,
        result,
    )


# =========================================================
# FINAL MODEL
# =========================================================

def fit_final_position_model(
    matrix_df,
    core_df,
    position,
    ridge_lambda,
):

    features = (
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

    development_df = matrix_df[
        (
            matrix_df[
                "season"
            ].isin(
                [
                    TUNING_TRAIN_SEASON,
                    TUNING_VALIDATION_SEASON,
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

    test_df = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            FINAL_TEST_SEASON
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

    (
        x_dev,
        y_dev,
    ) = prepare_xy(
        development_df,
        features,
    )

    (
        x_test,
        y_test,
    ) = prepare_xy(
        test_df,
        features,
    )

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
    ) = fit_ridge(
        x_dev_scaled,
        y_dev,
        ridge_lambda,
    )

    predictions = predict_ridge(
        x_test_scaled,
        beta,
        intercept,
    )

    coefficient_rows = []

    for index, feature in enumerate(
        features
    ):

        coefficient_rows.append(
            {
                "position":
                    position,

                "feature":
                    feature,

                "ridge_lambda":
                    ridge_lambda,

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

                "model_intercept":
                    intercept,
            }
        )

    coefficient_df = pd.DataFrame(
        coefficient_rows
    )

    prediction_df = test_df[
        [
            "game_id",
            "season",
            "week",

            "player_id",
            "player_display_name",

            "position",
            "team",
            "opponent_team",

            "player_history_games",

            "target_fanduel_points",
        ]
    ].copy()

    prediction_df[
        "model_projection"
    ] = predictions

    baseline_feature = (
        choose_baseline_feature(
            core_df,
            position,
        )
    )

    prediction_df[
        "baseline_feature"
    ] = baseline_feature

    prediction_df[
        "baseline_projection"
    ] = (
        pd.to_numeric(
            test_df[
                baseline_feature
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
    )

    prediction_df[
        "ridge_lambda"
    ] = ridge_lambda

    return (
        prediction_df,
        coefficient_df,
    )


# =========================================================
# METRICS
# =========================================================

def evaluate_predictions(
    prediction_df,
    position,
    model_name,
    prediction_column,
    sample_name,
):

    if sample_name == "ALL":

        sample = (
            prediction_df.copy()
        )

    elif sample_name == "ESTABLISHED":

        sample = prediction_df[
            prediction_df[
                "player_history_games"
            ]
            >=
            MIN_HISTORY_ESTABLISHED
        ].copy()

    else:

        raise RuntimeError(
            f"Unknown sample: "
            f"{sample_name}"
        )

    actual = (
        pd.to_numeric(
            sample[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
    )

    predicted = (
        pd.to_numeric(
            sample[
                prediction_column
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy()
    )

    return {
        "season":
            FINAL_TEST_SEASON,

        "position":
            position,

        "sample":
            sample_name,

        "model":
            model_name,

        "rows":
            len(
                sample
            ),

        "mae":
            mae(
                actual,
                predicted,
            ),

        "rmse":
            rmse(
                actual,
                predicted,
            ),

        "correlation":
            safe_corr(
                actual,
                predicted,
            ),

        "bias":
            bias(
                actual,
                predicted,
            ),

        "actual_mean":
            float(
                np.mean(
                    actual
                )
            ),

        "prediction_mean":
            float(
                np.mean(
                    predicted
                )
            ),
    }


# =========================================================
# CEILING RANKING ANALYSIS
# =========================================================

def evaluate_ceiling_ranking(
    prediction_df,
    position,
    model_name,
    prediction_column,
    sample_name,
):

    if sample_name == "ALL":

        sample = (
            prediction_df.copy()
        )

    else:

        sample = prediction_df[
            prediction_df[
                "player_history_games"
            ]
            >=
            MIN_HISTORY_ESTABLISHED
        ].copy()

    if sample.empty:

        return []

    sample[
        prediction_column
    ] = pd.to_numeric(
        sample[
            prediction_column
        ],
        errors="coerce",
    ).fillna(0.0)

    sample[
        "target_fanduel_points"
    ] = pd.to_numeric(
        sample[
            "target_fanduel_points"
        ],
        errors="coerce",
    ).fillna(0.0)

    # -----------------------------------------------------
    # Top prediction quintile within POSITION + WEEK.
    #
    # This is closer to DFS selection than evaluating one
    # global threshold across an entire season.
    # -----------------------------------------------------

    sample[
        "prediction_percentile"
    ] = (
        sample.groupby(
            [
                "season",
                "week",
            ]
        )[
            prediction_column
        ]
        .rank(
            method="first",
            pct=True,
        )
    )

    top_group = sample[
        sample[
            "prediction_percentile"
        ]
        >
        0.80
    ].copy()

    rows = []

    for threshold in CEILING_THRESHOLDS:

        overall_hit = float(
            (
                sample[
                    "target_fanduel_points"
                ]
                >=
                threshold
            ).mean()
        )

        top_hit = float(
            (
                top_group[
                    "target_fanduel_points"
                ]
                >=
                threshold
            ).mean()
        )

        if overall_hit > 0:

            lift = (
                top_hit
                /
                overall_hit
            )

        else:

            lift = 0.0

        actual_ceiling_players = sample[
            sample[
                "target_fanduel_points"
            ]
            >=
            threshold
        ]

        if len(
            actual_ceiling_players
        ) > 0:

            ceiling_capture_rate = float(
                (
                    actual_ceiling_players[
                        "prediction_percentile"
                    ]
                    >
                    0.80
                ).mean()
            )

        else:

            ceiling_capture_rate = 0.0

        rows.append(
            {
                "season":
                    FINAL_TEST_SEASON,

                "position":
                    position,

                "sample":
                    sample_name,

                "model":
                    model_name,

                "threshold":
                    threshold,

                "sample_rows":
                    len(
                        sample
                    ),

                "top_quintile_rows":
                    len(
                        top_group
                    ),

                "overall_ceiling_rate":
                    overall_hit,

                "top_prediction_ceiling_rate":
                    top_hit,

                "ceiling_lift":
                    lift,

                "ceiling_capture_rate":
                    ceiling_capture_rate,

                "top_prediction_actual_fd_mean":
                    float(
                        top_group[
                            "target_fanduel_points"
                        ].mean()
                    ),
            }
        )

    return rows


# =========================================================
# MODEL IMPROVEMENT AUDIT
# =========================================================

def compare_models(
    benchmark_df,
):

    section(
        "2025 MODEL IMPROVEMENT SUMMARY"
    )

    rows = []

    for position in POSITIONS:

        for sample in [
            "ALL",
            "ESTABLISHED",
        ]:

            subset = benchmark_df[
                (
                    benchmark_df[
                        "position"
                    ]
                    ==
                    position
                )
                &
                (
                    benchmark_df[
                        "sample"
                    ]
                    ==
                    sample
                )
            ]

            baseline = subset[
                subset[
                    "model"
                ]
                ==
                "BASELINE"
            ]

            ridge = subset[
                subset[
                    "model"
                ]
                ==
                "RIDGE_CORE"
            ]

            if (
                baseline.empty
                or
                ridge.empty
            ):

                continue

            baseline = baseline.iloc[0]
            ridge = ridge.iloc[0]

            rows.append(
                {
                    "position":
                        position,

                    "sample":
                        sample,

                    "baseline_mae":
                        baseline[
                            "mae"
                        ],

                    "model_mae":
                        ridge[
                            "mae"
                        ],

                    "mae_improvement":
                        (
                            baseline[
                                "mae"
                            ]
                            -
                            ridge[
                                "mae"
                            ]
                        ),

                    "baseline_rmse":
                        baseline[
                            "rmse"
                        ],

                    "model_rmse":
                        ridge[
                            "rmse"
                        ],

                    "rmse_improvement":
                        (
                            baseline[
                                "rmse"
                            ]
                            -
                            ridge[
                                "rmse"
                            ]
                        ),

                    "baseline_corr":
                        baseline[
                            "correlation"
                        ],

                    "model_corr":
                        ridge[
                            "correlation"
                        ],

                    "correlation_improvement":
                        (
                            ridge[
                                "correlation"
                            ]
                            -
                            baseline[
                                "correlation"
                            ]
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    return result


# =========================================================
# STRUCTURAL AUDIT
# =========================================================

def audit_predictions(
    prediction_df,
):

    section(
        "PROJECTION PREDICTION AUDIT"
    )

    duplicates = (
        prediction_df.groupby(
            [
                "game_id",
                "player_id",
                "team",
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

    null_model = int(
        prediction_df[
            "model_projection"
        ].isna().sum()
    )

    null_baseline = int(
        prediction_df[
            "baseline_projection"
        ].isna().sum()
    )

    invalid_season = int(
        (
            prediction_df[
                "season"
            ]
            !=
            FINAL_TEST_SEASON
        ).sum()
    )

    print(
        f"2025 prediction rows: "
        f"{len(prediction_df)}"
    )

    print(
        f"Duplicate player-game rows: "
        f"{len(duplicates)}"
    )

    print(
        f"NULL model projections: "
        f"{null_model}"
    )

    print(
        f"NULL baseline projections: "
        f"{null_baseline}"
    )

    print(
        f"Non-2025 rows: "
        f"{invalid_season}"
    )

    problems = (
        len(
            duplicates
        )
        +
        null_model
        +
        null_baseline
        +
        invalid_season
    )

    if problems:

        raise RuntimeError(
            "Projection prediction audit failed."
        )

    print()
    print(
        "PASS: 2025 projection predictions "
        "passed structural audits."
    )


# =========================================================
# EXPORT
# =========================================================

def export_results(
    benchmark_df,
    tuning_df,
    predictions_df,
    ceiling_df,
    coefficients_df,
):

    section(
        "EXPORTING PROJECTION BENCHMARK"
    )

    benchmark_df.to_csv(
        BENCHMARK_CSV,
        index=False,
    )

    benchmark_df.to_parquet(
        BENCHMARK_PARQUET,
        index=False,
    )

    tuning_df.to_csv(
        RIDGE_TUNING_CSV,
        index=False,
    )

    tuning_df.to_parquet(
        RIDGE_TUNING_PARQUET,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    predictions_df.to_parquet(
        PREDICTIONS_PARQUET,
        index=False,
    )

    ceiling_df.to_csv(
        CEILING_CSV,
        index=False,
    )

    ceiling_df.to_parquet(
        CEILING_PARQUET,
        index=False,
    )

    coefficients_df.to_csv(
        MODEL_COEFFICIENTS_CSV,
        index=False,
    )

    coefficients_df.to_parquet(
        MODEL_COEFFICIENTS_PARQUET,
        index=False,
    )

    print(
        f"Benchmark rows: "
        f"{len(benchmark_df)}"
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
        f"Ceiling rows: "
        f"{len(ceiling_df)}"
    )

    print(
        f"Coefficient rows: "
        f"{len(coefficients_df)}"
    )

    print()

    print(
        f"Benchmark CSV: "
        f"{BENCHMARK_CSV}"
    )

    print(
        f"Predictions CSV: "
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"Ceiling CSV: "
        f"{CEILING_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_projection_benchmark():

    section(
        "NFL DFS PROJECTION BENCHMARK ENGINE"
    )

    matrix_df = load_matrix()

    core_df = load_core_features()

    validate_core_features(
        matrix_df,
        core_df,
    )

    tuning_frames = []

    prediction_frames = []

    coefficient_frames = []

    selected_lambdas = {}

    # =====================================================
    # TUNE USING 2023 -> 2024 ONLY
    # =====================================================

    for position in POSITIONS:

        (
            best_lambda,
            tuning_df,
        ) = tune_position_model(
            matrix_df,
            core_df,
            position,
        )

        selected_lambdas[
            position
        ] = best_lambda

        tuning_frames.append(
            tuning_df
        )

    # =====================================================
    # REFIT 2023+2024, TEST 2025
    # =====================================================

    section(
        "FITTING FINAL 2023-2024 MODELS"
    )

    for position in POSITIONS:

        (
            prediction_df,
            coefficient_df,
        ) = fit_final_position_model(
            matrix_df,
            core_df,
            position,
            selected_lambdas[
                position
            ],
        )

        prediction_frames.append(
            prediction_df
        )

        coefficient_frames.append(
            coefficient_df
        )

        print(
            f"{position}: "
            f"{len(prediction_df)} "
            f"2025 predictions"
        )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    coefficients_df = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    tuning_df = pd.concat(
        tuning_frames,
        ignore_index=True,
    )

    audit_predictions(
        predictions_df
    )

    # =====================================================
    # POINT-PREDICTION BENCHMARK
    # =====================================================

    section(
        "2025 POINT PROJECTION BENCHMARK"
    )

    benchmark_rows = []

    ceiling_rows = []

    for position in POSITIONS:

        position_df = predictions_df[
            predictions_df[
                "position"
            ]
            ==
            position
        ].copy()

        for sample_name in [
            "ALL",
            "ESTABLISHED",
        ]:

            benchmark_rows.append(
                evaluate_predictions(
                    position_df,
                    position,
                    "BASELINE",
                    "baseline_projection",
                    sample_name,
                )
            )

            benchmark_rows.append(
                evaluate_predictions(
                    position_df,
                    position,
                    "RIDGE_CORE",
                    "model_projection",
                    sample_name,
                )
            )

            ceiling_rows.extend(
                evaluate_ceiling_ranking(
                    position_df,
                    position,
                    "BASELINE",
                    "baseline_projection",
                    sample_name,
                )
            )

            ceiling_rows.extend(
                evaluate_ceiling_ranking(
                    position_df,
                    position,
                    "RIDGE_CORE",
                    "model_projection",
                    sample_name,
                )
            )

    benchmark_df = pd.DataFrame(
        benchmark_rows
    )

    ceiling_df = pd.DataFrame(
        ceiling_rows
    )

    print(
        benchmark_df.to_string(
            index=False
        )
    )

    improvement_df = compare_models(
        benchmark_df
    )

    # =====================================================
    # CEILING SUMMARY
    # =====================================================

    section(
        "2025 CEILING RANKING SUMMARY"
    )

    ceiling_summary = ceiling_df[
        (
            ceiling_df[
                "sample"
            ]
            ==
            "ESTABLISHED"
        )
        &
        (
            ceiling_df[
                "threshold"
            ].isin(
                [
                    20.0,
                    25.0,
                    30.0,
                ]
            )
        )
    ].copy()

    print(
        ceiling_summary[
            [
                "position",
                "model",
                "threshold",
                "overall_ceiling_rate",
                "top_prediction_ceiling_rate",
                "ceiling_lift",
                "ceiling_capture_rate",
                "top_prediction_actual_fd_mean",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # COEFFICIENT SUMMARY
    # =====================================================

    section(
        "FINAL STANDARDIZED MODEL COEFFICIENTS"
    )

    coefficient_print = (
        coefficients_df.copy()
    )

    coefficient_print[
        "absolute_coefficient"
    ] = (
        coefficient_print[
            "standardized_coefficient"
        ]
        .abs()
    )

    coefficient_print = (
        coefficient_print.sort_values(
            [
                "position",
                "absolute_coefficient",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        coefficient_print[
            [
                "position",
                "feature",
                "ridge_lambda",
                "standardized_coefficient",
            ]
        ].to_string(
            index=False
        )
    )

    export_results(
        benchmark_df,
        tuning_df,
        predictions_df,
        ceiling_df,
        coefficients_df,
    )

    section(
        "PROJECTION BENCHMARK BUILD SUCCESSFUL"
    )

    print()
    print(
        "Model selection chronology:"
    )

    print()
    print(
        "2023 -> train candidate ridge models"
    )

    print(
        "2024 -> select ridge penalty"
    )

    print(
        "2023+2024 -> refit final model"
    )

    print(
        "2025 -> untouched final evaluation"
    )

    print()
    print(
        "No 2025 result was used to choose "
        "features or ridge penalties."
    )


if __name__ == "__main__":

    run_projection_benchmark()
