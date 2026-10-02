import numpy as np
import pandas as pd

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection

from projection_benchmark import (
    fit_standardizer as ridge_fit_standardizer,
    transform_standardizer as ridge_transform_standardizer,
    fit_ridge,
    predict_ridge,
)

from ceiling_probability import (
    fit_standardizer as ceiling_fit_standardizer,
    transform_standardizer as ceiling_transform_standardizer,
    fit_logistic,
    predict_probability,
    auc_score,
)


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
BLEND_TUNE_SEASON = 2024
FINAL_TEST_SEASON = 2025

BLEND_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

# ---------------------------------------------------------
# Weight interpretation:
#
# 0.00 = pure ridge projection rank
# 1.00 = pure direct ceiling probability rank
#
# Intermediate values blend the WEEKLY POSITIONAL
# percentile ranks of the two signals.
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
# INPUT FILES
# =========================================================

CORE_FEATURES_CSV = (
    CSV_DIR / "nfl_core_projection_features.csv"
)

RIDGE_TUNING_CSV = (
    CSV_DIR / "nfl_projection_ridge_tuning.csv"
)

CEILING_TUNING_CSV = (
    CSV_DIR / "nfl_ceiling_probability_tuning.csv"
)


# =========================================================
# OUTPUT FILES
# =========================================================

BLEND_TUNING_CSV = (
    CSV_DIR / "nfl_gpp_rank_blend_tuning.csv"
)

BLEND_TUNING_PARQUET = (
    PARQUET_DIR / "nfl_gpp_rank_blend_tuning.parquet"
)

GPP_BENCHMARK_CSV = (
    CSV_DIR / "nfl_gpp_rank_benchmark_2025.csv"
)

GPP_BENCHMARK_PARQUET = (
    PARQUET_DIR / "nfl_gpp_rank_benchmark_2025.parquet"
)

GPP_PREDICTIONS_CSV = (
    CSV_DIR / "nfl_gpp_rank_predictions_2025.csv"
)

GPP_PREDICTIONS_PARQUET = (
    PARQUET_DIR / "nfl_gpp_rank_predictions_2025.parquet"
)

SELECTED_BLEND_CSV = (
    CSV_DIR / "nfl_gpp_rank_selected_blends.csv"
)

SELECTED_BLEND_PARQUET = (
    PARQUET_DIR / "nfl_gpp_rank_selected_blends.parquet"
)


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def actual_fd_array(df):

    return (
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


def percentile_rank(
    df,
    score_column,
):

    return (
        df.groupby(
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
        f"Rows loaded: {len(df)}"
    )

    return df


def load_core_features():

    section(
        "LOADING CORE FEATURES"
    )

    if not CORE_FEATURES_CSV.exists():

        raise RuntimeError(
            f"Missing: {CORE_FEATURES_CSV}"
        )

    df = pd.read_csv(
        CORE_FEATURES_CSV
    )

    print(
        f"Core feature rows: {len(df)}"
    )

    return df


def load_selected_ridge_lambdas():

    section(
        "LOADING LOCKED RIDGE PENALTIES"
    )

    if not RIDGE_TUNING_CSV.exists():

        raise RuntimeError(
            f"Missing: {RIDGE_TUNING_CSV}"
        )

    df = pd.read_csv(
        RIDGE_TUNING_CSV
    )

    selected = df[
        df[
            "selected"
        ]
        ==
        1
    ].copy()

    values = {}

    for _, row in selected.iterrows():

        position = row[
            "position"
        ]

        values[
            position
        ] = float(
            row[
                "ridge_lambda"
            ]
        )

    for position in POSITIONS:

        if position not in values:

            raise RuntimeError(
                f"No ridge lambda for {position}"
            )

        print(
            f"{position}: "
            f"{values[position]}"
        )

    return values


def load_selected_ceiling_l2():

    section(
        "LOADING LOCKED CEILING PENALTIES"
    )

    if not CEILING_TUNING_CSV.exists():

        raise RuntimeError(
            f"Missing: {CEILING_TUNING_CSV}"
        )

    df = pd.read_csv(
        CEILING_TUNING_CSV
    )

    selected = df[
        df[
            "selected"
        ]
        ==
        1
    ].copy()

    values = {}

    for _, row in selected.iterrows():

        key = (
            row[
                "position"
            ],
            float(
                row[
                    "threshold"
                ]
            ),
        )

        values[
            key
        ] = float(
            row[
                "l2_value"
            ]
        )

    for position in POSITIONS:

        for threshold in POSITION_THRESHOLDS[
            position
        ]:

            key = (
                position,
                threshold,
            )

            if key not in values:

                raise RuntimeError(
                    "Missing ceiling L2: "
                    f"{position} "
                    f"{threshold}"
                )

    print(
        f"Locked ceiling models: "
        f"{len(values)}"
    )

    return values


# =========================================================
# FEATURE LIST
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


# =========================================================
# RIDGE COMPONENT
# =========================================================

def fit_predict_ridge_component(
    train_df,
    predict_df,
    features,
    ridge_lambda,
):

    x_train = prepare_x(
        train_df,
        features,
    )

    x_predict = prepare_x(
        predict_df,
        features,
    )

    y_train = actual_fd_array(
        train_df
    )

    (
        means,
        stds,
    ) = ridge_fit_standardizer(
        x_train
    )

    x_train_scaled = (
        ridge_transform_standardizer(
            x_train,
            means,
            stds,
        )
    )

    x_predict_scaled = (
        ridge_transform_standardizer(
            x_predict,
            means,
            stds,
        )
    )

    (
        beta,
        intercept,
    ) = fit_ridge(
        x_train_scaled,
        y_train,
        ridge_lambda,
    )

    return predict_ridge(
        x_predict_scaled,
        beta,
        intercept,
    )


# =========================================================
# CEILING COMPONENT
# =========================================================

def fit_predict_ceiling_component(
    train_df,
    predict_df,
    features,
    threshold,
    l2_value,
):

    x_train = prepare_x(
        train_df,
        features,
    )

    x_predict = prepare_x(
        predict_df,
        features,
    )

    y_train = (
        actual_fd_array(
            train_df
        )
        >=
        threshold
    ).astype(float)

    if (
        y_train.sum()
        ==
        0
    ):

        raise RuntimeError(
            f"No ceiling events for "
            f"threshold {threshold}"
        )

    (
        means,
        stds,
    ) = ceiling_fit_standardizer(
        x_train
    )

    x_train_scaled = (
        ceiling_transform_standardizer(
            x_train,
            means,
            stds,
        )
    )

    x_predict_scaled = (
        ceiling_transform_standardizer(
            x_predict,
            means,
            stds,
        )
    )

    (
        beta,
        intercept,
        _,
    ) = fit_logistic(
        x_train_scaled,
        y_train,
        l2_value,
    )

    return predict_probability(
        x_predict_scaled,
        beta,
        intercept,
    )


# =========================================================
# BUILD COMPONENT PREDICTIONS
# =========================================================

def build_component_predictions(
    matrix_df,
    core_df,
    ridge_lambdas,
    ceiling_l2,
    training_seasons,
    prediction_season,
    position,
    threshold,
):

    features = position_features(
        core_df,
        position,
    )

    train_df = matrix_df[
        (
            matrix_df[
                "season"
            ].isin(
                training_seasons
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

    prediction_df = matrix_df[
        (
            matrix_df[
                "season"
            ]
            ==
            prediction_season
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

    ridge_prediction = (
        fit_predict_ridge_component(
            train_df,
            prediction_df,
            features,
            ridge_lambdas[
                position
            ],
        )
    )

    ceiling_probability = (
        fit_predict_ceiling_component(
            train_df,
            prediction_df,
            features,
            threshold,
            ceiling_l2[
                (
                    position,
                    threshold,
                )
            ],
        )
    )

    output = prediction_df[
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
    ] = (
        pd.to_numeric(
            output[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        >=
        threshold
    ).astype(int)

    output[
        "ridge_projection"
    ] = ridge_prediction

    output[
        "ceiling_probability"
    ] = ceiling_probability

    output[
        "ridge_rank"
    ] = percentile_rank(
        output,
        "ridge_projection",
    )

    output[
        "ceiling_rank"
    ] = percentile_rank(
        output,
        "ceiling_probability",
    )

    return output


# =========================================================
# EVALUATE SCORE
# =========================================================

def evaluate_score(
    df,
    score_column,
):

    working = df.copy()

    working[
        "final_rank"
    ] = percentile_rank(
        working,
        score_column,
    )

    top = working[
        working[
            "final_rank"
        ]
        >
        0.80
    ]

    ceiling_events = int(
        working[
            "actual_ceiling"
        ].sum()
    )

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

    ceiling_players = working[
        working[
            "actual_ceiling"
        ]
        ==
        1
    ]

    if len(
        ceiling_players
    ):

        capture = float(
            (
                ceiling_players[
                    "final_rank"
                ]
                >
                0.80
            ).mean()
        )

    else:

        capture = 0.0

    auc = auc_score(
        working[
            "actual_ceiling"
        ].to_numpy(),
        working[
            score_column
        ].to_numpy(),
    )

    return {
        "rows":
            len(
                working
            ),

        "ceiling_events":
            ceiling_events,

        "overall_ceiling_rate":
            overall_rate,

        "top_quintile_ceiling_rate":
            top_rate,

        "ceiling_lift":
            lift,

        "ceiling_capture_rate":
            capture,

        "auc":
            auc,
    }


# =========================================================
# TUNE BLEND ON 2024
# =========================================================

def tune_blend(
    component_df,
    position,
    threshold,
):

    rows = []

    for ceiling_weight in BLEND_WEIGHTS:

        ridge_weight = (
            1.0
            -
            ceiling_weight
        )

        working = (
            component_df.copy()
        )

        working[
            "blend_score"
        ] = (
            ridge_weight
            *
            working[
                "ridge_rank"
            ]
            +
            ceiling_weight
            *
            working[
                "ceiling_rank"
            ]
        )

        metrics = evaluate_score(
            working,
            "blend_score",
        )

        rows.append(
            {
                "position":
                    position,

                "threshold":
                    threshold,

                "tuning_season":
                    BLEND_TUNE_SEASON,

                "ridge_weight":
                    ridge_weight,

                "ceiling_weight":
                    ceiling_weight,

                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # GPP selection hierarchy:
    #
    # 1. Top-quintile ceiling hit rate
    # 2. AUC
    # 3. Capture rate
    # 4. Prefer more ridge weight if otherwise tied
    #
    # The final tie-break intentionally favors the simpler
    # established ridge signal.
    # -----------------------------------------------------

    result = (
        result.sort_values(
            [
                "top_quintile_ceiling_rate",
                "auc",
                "ceiling_capture_rate",
                "ridge_weight",
            ],
            ascending=[
                False,
                False,
                False,
                False,
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
        0,
        "selected",
    ] = 1

    selected = (
        result.iloc[0]
    )

    return (
        float(
            selected[
                "ceiling_weight"
            ]
        ),
        result,
    )


# =========================================================
# FINAL 2025 EVALUATION
# =========================================================

def evaluate_final_models(
    component_df,
    position,
    threshold,
    ceiling_weight,
):

    ridge_weight = (
        1.0
        -
        ceiling_weight
    )

    working = (
        component_df.copy()
    )

    working[
        "ridge_only_score"
    ] = working[
        "ridge_rank"
    ]

    working[
        "ceiling_only_score"
    ] = working[
        "ceiling_rank"
    ]

    working[
        "blend_score"
    ] = (
        ridge_weight
        *
        working[
            "ridge_rank"
        ]
        +
        ceiling_weight
        *
        working[
            "ceiling_rank"
        ]
    )

    rows = []

    for (
        model_name,
        score_column,
    ) in [
        (
            "RIDGE_ONLY",
            "ridge_only_score",
        ),
        (
            "DIRECT_CEILING_ONLY",
            "ceiling_only_score",
        ),
        (
            "LOCKED_BLEND",
            "blend_score",
        ),
    ]:

        metrics = evaluate_score(
            working,
            score_column,
        )

        rows.append(
            {
                "season":
                    FINAL_TEST_SEASON,

                "position":
                    position,

                "threshold":
                    threshold,

                "model":
                    model_name,

                "ridge_weight":
                    (
                        1.0
                        if model_name
                        ==
                        "RIDGE_ONLY"
                        else
                        0.0
                        if model_name
                        ==
                        "DIRECT_CEILING_ONLY"
                        else
                        ridge_weight
                    ),

                "ceiling_weight":
                    (
                        0.0
                        if model_name
                        ==
                        "RIDGE_ONLY"
                        else
                        1.0
                        if model_name
                        ==
                        "DIRECT_CEILING_ONLY"
                        else
                        ceiling_weight
                    ),

                **metrics,
            }
        )

    working[
        "selected_ceiling_weight"
    ] = ceiling_weight

    working[
        "selected_ridge_weight"
    ] = ridge_weight

    working[
        "gpp_rank_score"
    ] = working[
        "blend_score"
    ]

    working[
        "gpp_rank_percentile"
    ] = percentile_rank(
        working,
        "gpp_rank_score",
    )

    return (
        pd.DataFrame(
            rows
        ),
        working,
    )


# =========================================================
# AUDIT
# =========================================================

def audit_final_predictions(
    df,
):

    section(
        "GPP RANK PREDICTION AUDIT"
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

    null_scores = int(
        df[
            "gpp_rank_score"
        ]
        .isna()
        .sum()
    )

    invalid_rank = int(
        (
            (
                df[
                    "gpp_rank_percentile"
                ]
                <
                0
            )
            |
            (
                df[
                    "gpp_rank_percentile"
                ]
                >
                1
            )
        ).sum()
    )

    wrong_season = int(
        (
            df[
                "season"
            ]
            !=
            FINAL_TEST_SEASON
        ).sum()
    )

    print(
        f"Prediction rows: "
        f"{len(df)}"
    )

    print(
        f"Duplicate player-threshold rows: "
        f"{len(duplicates)}"
    )

    print(
        f"NULL GPP scores: "
        f"{null_scores}"
    )

    print(
        f"Invalid percentiles: "
        f"{invalid_rank}"
    )

    print(
        f"Non-2025 rows: "
        f"{wrong_season}"
    )

    problems = (
        len(
            duplicates
        )
        +
        null_scores
        +
        invalid_rank
        +
        wrong_season
    )

    if problems:

        raise RuntimeError(
            "GPP rank prediction audit failed."
        )

    print()
    print(
        "PASS: final GPP rank predictions "
        "passed structural audits."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_selected_blends(
    selected_df,
):

    section(
        "LOCKED 2024 BLEND WEIGHTS"
    )

    print(
        selected_df[
            [
                "position",
                "threshold",
                "ridge_weight",
                "ceiling_weight",
                "top_quintile_ceiling_rate",
                "ceiling_capture_rate",
                "auc",
            ]
        ].to_string(
            index=False
        )
    )


def print_final_benchmark(
    benchmark_df,
):

    section(
        "2025 GPP RANK BENCHMARK"
    )

    print(
        benchmark_df[
            [
                "position",
                "threshold",
                "model",
                "ridge_weight",
                "ceiling_weight",
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
    selected_df,
    benchmark_df,
    predictions_df,
):

    section(
        "EXPORTING GPP RANK RESULTS"
    )

    tuning_df.to_csv(
        BLEND_TUNING_CSV,
        index=False,
    )

    tuning_df.to_parquet(
        BLEND_TUNING_PARQUET,
        index=False,
    )

    selected_df.to_csv(
        SELECTED_BLEND_CSV,
        index=False,
    )

    selected_df.to_parquet(
        SELECTED_BLEND_PARQUET,
        index=False,
    )

    benchmark_df.to_csv(
        GPP_BENCHMARK_CSV,
        index=False,
    )

    benchmark_df.to_parquet(
        GPP_BENCHMARK_PARQUET,
        index=False,
    )

    predictions_df.to_csv(
        GPP_PREDICTIONS_CSV,
        index=False,
    )

    predictions_df.to_parquet(
        GPP_PREDICTIONS_PARQUET,
        index=False,
    )

    print(
        f"Blend tuning rows: "
        f"{len(tuning_df)}"
    )

    print(
        f"Selected blend rows: "
        f"{len(selected_df)}"
    )

    print(
        f"Benchmark rows: "
        f"{len(benchmark_df)}"
    )

    print(
        f"Prediction rows: "
        f"{len(predictions_df)}"
    )

    print()

    print(
        f"Selected blends CSV: "
        f"{SELECTED_BLEND_CSV}"
    )

    print(
        f"Benchmark CSV: "
        f"{GPP_BENCHMARK_CSV}"
    )

    print(
        f"Predictions CSV: "
        f"{GPP_PREDICTIONS_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_gpp_rank_benchmark():

    section(
        "NFL DFS GPP RANK BENCHMARK ENGINE"
    )

    matrix_df = load_matrix()

    core_df = load_core_features()

    ridge_lambdas = (
        load_selected_ridge_lambdas()
    )

    ceiling_l2 = (
        load_selected_ceiling_l2()
    )

    tuning_frames = []

    selected_rows = []

    final_benchmark_frames = []

    final_prediction_frames = []

    # =====================================================
    # 2024 BLEND TUNING
    #
    # Models are trained using 2023 only.
    # =====================================================

    section(
        "TUNING BLENDS ON 2024"
    )

    for position in POSITIONS:

        for threshold in (
            POSITION_THRESHOLDS[
                position
            ]
        ):

            component_2024 = (
                build_component_predictions(
                    matrix_df,
                    core_df,
                    ridge_lambdas,
                    ceiling_l2,
                    [
                        TRAIN_SEASON,
                    ],
                    BLEND_TUNE_SEASON,
                    position,
                    threshold,
                )
            )

            (
                ceiling_weight,
                tuning_df,
            ) = tune_blend(
                component_2024,
                position,
                threshold,
            )

            tuning_frames.append(
                tuning_df
            )

            selected = tuning_df[
                tuning_df[
                    "selected"
                ]
                ==
                1
            ].iloc[0].to_dict()

            selected_rows.append(
                selected
            )

            print(
                f"{position} "
                f"{threshold:.0f}+: "
                f"ridge "
                f"{1.0 - ceiling_weight:.2f} | "
                f"ceiling "
                f"{ceiling_weight:.2f}"
            )

    tuning_output = pd.concat(
        tuning_frames,
        ignore_index=True,
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    print_selected_blends(
        selected_df
    )

    # =====================================================
    # FINAL TEST
    #
    # Components are refit on 2023 + 2024.
    # Locked blend weight is then tested on 2025.
    # =====================================================

    section(
        "RUNNING UNTOUCHED 2025 GPP TEST"
    )

    for _, selected in (
        selected_df.iterrows()
    ):

        position = selected[
            "position"
        ]

        threshold = float(
            selected[
                "threshold"
            ]
        )

        ceiling_weight = float(
            selected[
                "ceiling_weight"
            ]
        )

        component_2025 = (
            build_component_predictions(
                matrix_df,
                core_df,
                ridge_lambdas,
                ceiling_l2,
                [
                    TRAIN_SEASON,
                    BLEND_TUNE_SEASON,
                ],
                FINAL_TEST_SEASON,
                position,
                threshold,
            )
        )

        (
            benchmark_df,
            predictions_df,
        ) = evaluate_final_models(
            component_2025,
            position,
            threshold,
            ceiling_weight,
        )

        final_benchmark_frames.append(
            benchmark_df
        )

        final_prediction_frames.append(
            predictions_df
        )

    final_benchmark_df = pd.concat(
        final_benchmark_frames,
        ignore_index=True,
    )

    final_predictions_df = pd.concat(
        final_prediction_frames,
        ignore_index=True,
    )

    audit_final_predictions(
        final_predictions_df
    )

    print_final_benchmark(
        final_benchmark_df
    )

    export_results(
        tuning_output,
        selected_df,
        final_benchmark_df,
        final_predictions_df,
    )

    section(
        "GPP RANK BENCHMARK BUILD SUCCESSFUL"
    )

    print()
    print(
        "Chronology:"
    )

    print(
        "2023 -> fit ridge + ceiling components"
    )

    print(
        "2024 -> select blend weights"
    )

    print(
        "2023+2024 -> refit components"
    )

    print(
        "2025 -> untouched final blend evaluation"
    )

    print()
    print(
        "Promotion rule:"
    )

    print(
        "Use LOCKED_BLEND only where it "
        "outperforms RIDGE_ONLY out of sample."
    )


if __name__ == "__main__":

    run_gpp_rank_benchmark()
