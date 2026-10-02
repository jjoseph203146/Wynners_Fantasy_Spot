from live_injury_reforecast import *
import numpy as np
import pandas as pd

from config import (
    CSV_DIR,
    PARQUET_DIR,
)

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

BASE_TRAINING_SEASONS = [
    2023,
    2024,
    2025,
]

ROLLING_TRAINING_START_SEASON = 2026

RIDGE_LAMBDAS = {
    "QB": 10.0,
    "RB": 100.0,
    "WR": 10.0,
    "TE": 100.0,
}

CEILING_THRESHOLDS = {
    "QB": [
        20,
        25,
        30,
    ],
    "RB": [
        15,
        20,
        25,
        30,
    ],
    "WR": [
        15,
        20,
        25,
        30,
    ],
    "TE": [
        10,
        15,
        20,
        25,
    ],
}

ALL_THRESHOLDS = sorted(
    {
        threshold
        for thresholds in CEILING_THRESHOLDS.values()
        for threshold in thresholds
    }
)

CEILING_L2 = {
    ("QB", 20): 1.0,
    ("QB", 25): 10.0,
    ("QB", 30): 1.0,

    ("RB", 15): 10.0,
    ("RB", 20): 10.0,
    ("RB", 25): 10.0,
    ("RB", 30): 10.0,

    ("WR", 15): 0.0,
    ("WR", 20): 10.0,
    ("WR", 25): 0.0,
    ("WR", 30): 0.0,

    ("TE", 10): 100.0,
    ("TE", 15): 10.0,
    ("TE", 20): 1.0,
    ("TE", 25): 10.0,
}


# =========================================================
# LOCKED GPP POLICY
# =========================================================

GPP_POLICY = {
    ("QB", 20): (1.0, 0.0),
    ("QB", 25): (1.0, 0.0),
    ("QB", 30): (1.0, 0.0),

    ("RB", 15): (1.0, 0.0),
    ("RB", 20): (0.2, 0.8),
    ("RB", 25): (1.0, 0.0),
    ("RB", 30): (1.0, 0.0),

    ("WR", 15): (0.1, 0.9),
    ("WR", 20): (1.0, 0.0),
    ("WR", 25): (0.6, 0.4),
    ("WR", 30): (1.0, 0.0),

    ("TE", 10): (0.9, 0.1),
    ("TE", 15): (0.9, 0.1),
    ("TE", 20): (0.7, 0.3),
    ("TE", 25): (1.0, 0.0),
}


# =========================================================
# FILES
# =========================================================

MATRIX_PARQUET = (
    PARQUET_DIR
    / "nfl_dfs_feature_matrix.parquet"
)

MATRIX_CSV = (
    CSV_DIR
    / "nfl_dfs_feature_matrix.csv"
)

CORE_PARQUET = (
    PARQUET_DIR
    / "nfl_core_projection_features.parquet"
)

CORE_CSV = (
    CSV_DIR
    / "nfl_core_projection_features.csv"
)

CURRENT_PARQUET = (
    PARQUET_DIR
    / "nfl_current_slate_features.parquet"
)

CURRENT_CSV = (
    CSV_DIR
    / "nfl_current_slate_features.csv"
)

OUTPUT_PARQUET = (
    PARQUET_DIR
    / "nfl_production_projection.parquet"
)

OUTPUT_CSV = (
    CSV_DIR
    / "nfl_production_projection.csv"
)

AUDIT_PARQUET = (
    PARQUET_DIR
    / "audit_production_projection.parquet"
)

AUDIT_CSV = (
    CSV_DIR
    / "audit_production_projection.csv"
)

MODEL_SUMMARY_PARQUET = (
    PARQUET_DIR
    / "nfl_production_model_summary.parquet"
)

MODEL_SUMMARY_CSV = (
    CSV_DIR
    / "nfl_production_model_summary.csv"
)


# =========================================================
# DISPLAY
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =========================================================
# LOAD FILE
# =========================================================

def load_dataframe(
    parquet_path,
    csv_path,
    label,
):

    if parquet_path.exists():

        print(
            f"{label}: "
            f"{parquet_path}"
        )

        return pd.read_parquet(
            parquet_path
        )

    if csv_path.exists():

        print(
            f"{label}: "
            f"{csv_path}"
        )

        return pd.read_csv(
            csv_path,
            low_memory=False,
        )

    raise RuntimeError(
        f"Could not find {label}."
    )


# =========================================================
# NUMERIC PREPARATION
# =========================================================

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
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
    )


def prepare_y(df):

    return (
        pd.to_numeric(
            df[
                "target_fanduel_points"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
    )


# =========================================================
# POSITIONAL PERCENTILE
# =========================================================

def percentile_rank(
    df,
    score_column,
):

    return (
        df.groupby(
            [
                "season",
                "week",
                "position",
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
# CORE FEATURES
# =========================================================

def load_core_features():

    section(
        "LOADING FROZEN CORE FEATURES"
    )

    core = load_dataframe(
        CORE_PARQUET,
        CORE_CSV,
        "Core feature source",
    )

    required = [
        "position",
        "core_rank",
        "feature",
    ]

    missing = [
        column
        for column in required
        if column not in core.columns
    ]

    if missing:

        raise RuntimeError(
            "Core feature source missing: "
            +
            ", ".join(
                missing
            )
        )

    core = core[
        core[
            "position"
        ].isin(
            POSITIONS
        )
    ].copy()

    core[
        "core_rank"
    ] = pd.to_numeric(
        core[
            "core_rank"
        ],
        errors="coerce",
    )

    core = core.sort_values(
        [
            "position",
            "core_rank",
        ]
    )

    print(
        f"Frozen feature rows: "
        f"{len(core)}"
    )

    for position in POSITIONS:

        count = len(
            core[
                core[
                    "position"
                ]
                ==
                position
            ]
        )

        print(
            f"{position}: "
            f"{count}"
        )

    if len(core) != 42:

        raise RuntimeError(
            "Expected exactly 42 frozen "
            "position-feature rows."
        )

    return core


def position_features(
    core,
    position,
):

    return (
        core[
            core[
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
# TRAINING MATRIX
# =========================================================

def load_training_matrix(
    core,
    target_season,
    target_week,
):

    section(
        "LOADING PRODUCTION TRAINING MATRIX"
    )

    matrix = load_dataframe(
        MATRIX_PARQUET,
        MATRIX_CSV,
        "Historical matrix",
    )

    required = [
        "season",
        "week",
        "position",
        "target_fanduel_points",
    ]

    missing = [
        column
        for column in required
        if column not in matrix.columns
    ]

    if missing:

        raise RuntimeError(
            "Historical matrix missing: "
            +
            ", ".join(
                missing
            )
        )

    matrix[
        "season"
    ] = pd.to_numeric(
        matrix[
            "season"
        ],
        errors="coerce",
    )

    matrix[
        "week"
    ] = pd.to_numeric(
        matrix[
            "week"
        ],
        errors="coerce",
    )

    base_mask = matrix[
        "season"
    ].isin(
        BASE_TRAINING_SEASONS
    )

    rolling_mask = (
        (
            matrix[
                "season"
            ]
            >=
            ROLLING_TRAINING_START_SEASON
        )
        &
        (
            (
                matrix[
                    "season"
                ]
                <
                target_season
            )
            |
            (
                (
                    matrix[
                        "season"
                    ]
                    ==
                    target_season
                )
                &
                (
                    matrix[
                        "week"
                    ]
                    <
                    target_week
                )
            )
        )
    )

    matrix = matrix[
        base_mask
        |
        rolling_mask
    ].copy()

    matrix = matrix[
        matrix[
            "position"
        ].isin(
            POSITIONS
        )
    ].copy()

    all_features = (
        core[
            "feature"
        ]
        .drop_duplicates()
        .tolist()
    )

    missing_features = [
        feature
        for feature in all_features
        if feature not in matrix.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Historical matrix missing frozen "
            "features: "
            +
            ", ".join(
                missing_features
            )
        )

    print(
        f"Base training seasons: "
        f"{BASE_TRAINING_SEASONS}"
    )

    print(
        f"Rolling training start season: "
        f"{ROLLING_TRAINING_START_SEASON}"
    )

    print(
        f"Target season/week: "
        f"{target_season}/{target_week}"
    )

    print(
        f"Training rows: "
        f"{len(matrix)}"
    )

    print()

    print(
        matrix.groupby(
            [
                "season",
                "position",
            ]
        )
        .size()
        .to_string()
    )

    return matrix


# =========================================================
# CURRENT FEATURES
# =========================================================

def load_current_features(
    core,
):

    section(
        "LOADING CURRENT SLATE FEATURES"
    )

    current = load_dataframe(
        CURRENT_PARQUET,
        CURRENT_CSV,
        "Current slate feature source",
    )

    required = [
        "season",
        "week",
        "game_id",
        "identity_key",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "player_history_games",
    ]

    missing = [
        column
        for column in required
        if column not in current.columns
    ]

    if missing:

        raise RuntimeError(
            "Current slate feature file missing: "
            +
            ", ".join(
                missing
            )
        )

    all_features = (
        core[
            "feature"
        ]
        .drop_duplicates()
        .tolist()
    )

    missing_features = [
        feature
        for feature in all_features
        if feature not in current.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Current slate missing frozen "
            "features: "
            +
            ", ".join(
                missing_features
            )
        )

    current[
        "player_history_games"
    ] = (
        pd.to_numeric(
            current[
                "player_history_games"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    current[
        "production_status"
    ] = np.where(
        current[
            "player_history_games"
        ]
        >
        0,
        "MODEL_READY",
        "COLD_START",
    )

    print(
        f"Current rows: "
        f"{len(current)}"
    )

    print()

    print(
        current.groupby(
            [
                "position",
                "production_status",
            ]
        )
        .size()
        .to_string()
    )

    return current


def resolve_target_season_week(
    current,
):

    required = [
        "season",
        "week",
    ]

    missing = [
        column
        for column in required
        if column not in current.columns
    ]

    if missing:

        raise RuntimeError(
            "Current inference pool missing "
            "target authority columns: "
            +
            ", ".join(
                missing
            )
        )

    pairs = (
        current[
            [
                "season",
                "week",
            ]
        ]
        .copy()
    )

    pairs[
        "season"
    ] = pd.to_numeric(
        pairs[
            "season"
        ],
        errors="coerce",
    )

    pairs[
        "week"
    ] = pd.to_numeric(
        pairs[
            "week"
        ],
        errors="coerce",
    )

    if (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Current inference pool contains "
            "NULL/non-numeric season or week."
        )

    pairs = (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        pairs
    ) != 1:

        raise RuntimeError(
            "Current inference pool must contain "
            "exactly one season/week pair; "
            f"found {len(pairs)}."
        )

    target_season = int(
        pairs.iloc[
            0
        ][
            "season"
        ]
    )

    target_week = int(
        pairs.iloc[
            0
        ][
            "week"
        ]
    )

    if target_week < 1:

        raise RuntimeError(
            "Target week must be >= 1."
        )

    return (
        target_season,
        target_week,
    )


# =========================================================
# RIDGE MODEL
# =========================================================

def score_ridge_models(
    historical,
    current,
    core,
):

    section(
        "FITTING PRODUCTION RIDGE MODELS"
    )

    current = current.copy()

    current[
        "ridge_projection"
    ] = np.nan

    summary_rows = []

    for position in POSITIONS:

        features = position_features(
            core,
            position,
        )

        ridge_lambda = (
            RIDGE_LAMBDAS[
                position
            ]
        )

        train = historical[
            historical[
                "position"
            ]
            ==
            position
        ].copy()

        score = current[
            (
                current[
                    "position"
                ]
                ==
                position
            )
            &
            (
                current[
                    "production_status"
                ]
                ==
                "MODEL_READY"
            )
        ].copy()

        x_train = prepare_x(
            train,
            features,
        )

        y_train = prepare_y(
            train
        )

        means, stds = (
            ridge_fit_standardizer(
                x_train
            )
        )

        x_train = (
            ridge_transform_standardizer(
                x_train,
                means,
                stds,
            )
        )

        beta, intercept = fit_ridge(
            x_train,
            y_train,
            ridge_lambda,
        )

        if not score.empty:

            x_score = prepare_x(
                score,
                features,
            )

            x_score = (
                ridge_transform_standardizer(
                    x_score,
                    means,
                    stds,
                )
            )

            predictions = predict_ridge(
                x_score,
                beta,
                intercept,
            )

            current.loc[
                score.index,
                "ridge_projection",
            ] = predictions

        print(
            f"{position}: "
            f"train={len(train)} "
            f"lambda={ridge_lambda} "
            f"score={len(score)}"
        )

        summary_rows.append(
            {
                "model_type":
                    "RIDGE",
                "position":
                    position,
                "threshold":
                    np.nan,
                "regularization":
                    ridge_lambda,
                "training_rows":
                    len(train),
                "positive_events":
                    np.nan,
                "positive_rate":
                    np.nan,
                "iterations":
                    np.nan,
                "feature_count":
                    len(features),
            }
        )

    return (
        current,
        summary_rows,
    )


# =========================================================
# CEILING MODELS
# =========================================================

def score_ceiling_models(
    historical,
    current,
    core,
    summary_rows,
):

    section(
        "FITTING PRODUCTION CEILING MODELS"
    )

    current = current.copy()

    # -----------------------------------------------------
    # IMPORTANT:
    # Initialize every shared threshold column ONCE.
    # Do not reset inside a position loop.
    # -----------------------------------------------------

    for threshold in ALL_THRESHOLDS:

        current[
            f"ceiling_prob_{threshold}"
        ] = np.nan

    for position in POSITIONS:

        features = position_features(
            core,
            position,
        )

        train = historical[
            historical[
                "position"
            ]
            ==
            position
        ].copy()

        score = current[
            (
                current[
                    "position"
                ]
                ==
                position
            )
            &
            (
                current[
                    "production_status"
                ]
                ==
                "MODEL_READY"
            )
        ].copy()

        x_train_raw = prepare_x(
            train,
            features,
        )

        actual = prepare_y(
            train
        )

        means, stds = (
            ceiling_fit_standardizer(
                x_train_raw
            )
        )

        x_train = (
            ceiling_transform_standardizer(
                x_train_raw,
                means,
                stds,
            )
        )

        if not score.empty:

            x_score_raw = prepare_x(
                score,
                features,
            )

            x_score = (
                ceiling_transform_standardizer(
                    x_score_raw,
                    means,
                    stds,
                )
            )

        else:

            x_score = np.empty(
                (
                    0,
                    len(features),
                )
            )

        for threshold in (
            CEILING_THRESHOLDS[
                position
            ]
        ):

            l2_value = CEILING_L2[
                (
                    position,
                    threshold,
                )
            ]

            y_train = (
                actual
                >=
                float(
                    threshold
                )
            ).astype(
                float
            )

            positives = int(
                y_train.sum()
            )

            if positives == 0:

                raise RuntimeError(
                    f"{position} "
                    f"{threshold}+ "
                    f"has no positives."
                )

            (
                beta,
                intercept,
                iterations,
            ) = fit_logistic(
                x_train,
                y_train,
                l2_value,
            )

            if len(score):

                probability = (
                    predict_probability(
                        x_score,
                        beta,
                        intercept,
                    )
                )

                current.loc[
                    score.index,
                    f"ceiling_prob_{threshold}",
                ] = probability

            positive_rate = (
                positives
                /
                len(train)
            )

            print(
                f"{position} "
                f"{threshold}+: "
                f"train={len(train)} "
                f"positive={positives} "
                f"rate={positive_rate:.4f} "
                f"L2={l2_value} "
                f"iterations={iterations}"
            )

            summary_rows.append(
                {
                    "model_type":
                        "DIRECT_CEILING",
                    "position":
                        position,
                    "threshold":
                        threshold,
                    "regularization":
                        l2_value,
                    "training_rows":
                        len(train),
                    "positive_events":
                        positives,
                    "positive_rate":
                        positive_rate,
                    "iterations":
                        iterations,
                    "feature_count":
                        len(features),
                }
            )

    return (
        current,
        summary_rows,
    )


# =========================================================
# RIDGE RANK
# =========================================================

def attach_ridge_rank(
    current,
):

    section(
        "BUILDING RIDGE POSITIONAL RANKS"
    )

    current = current.copy()

    current[
        "ridge_percentile"
    ] = np.nan

    current[
        "ridge_rank"
    ] = np.nan

    ready_mask = (
        current[
            "production_status"
        ]
        ==
        "MODEL_READY"
    )

    ready = current.loc[
        ready_mask
    ].copy()

    ready[
        "ridge_percentile"
    ] = percentile_rank(
        ready,
        "ridge_projection",
    )

    ready[
        "ridge_rank"
    ] = (
        ready.groupby(
            [
                "season",
                "week",
                "position",
            ]
        )[
            "ridge_projection"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    current.loc[
        ready.index,
        "ridge_percentile",
    ] = ready[
        "ridge_percentile"
    ]

    current.loc[
        ready.index,
        "ridge_rank",
    ] = ready[
        "ridge_rank"
    ]

    return current


# =========================================================
# CEILING PERCENTILES
# =========================================================

def attach_ceiling_percentiles(
    current,
):

    section(
        "BUILDING DIRECT CEILING POSITIONAL RANKS"
    )

    current = current.copy()

    # -----------------------------------------------------
    # Initialize shared columns once.
    # -----------------------------------------------------

    for threshold in ALL_THRESHOLDS:

        current[
            f"ceiling_percentile_{threshold}"
        ] = np.nan

        current[
            f"ceiling_rank_{threshold}"
        ] = np.nan

    for position in POSITIONS:

        mask = (
            (
                current[
                    "position"
                ]
                ==
                position
            )
            &
            (
                current[
                    "production_status"
                ]
                ==
                "MODEL_READY"
            )
        )

        subset = current.loc[
            mask
        ].copy()

        if subset.empty:

            continue

        for threshold in (
            CEILING_THRESHOLDS[
                position
            ]
        ):

            probability_column = (
                f"ceiling_prob_{threshold}"
            )

            percentile_column = (
                f"ceiling_percentile_{threshold}"
            )

            rank_column = (
                f"ceiling_rank_{threshold}"
            )

            subset[
                percentile_column
            ] = percentile_rank(
                subset,
                probability_column,
            )

            subset[
                rank_column
            ] = (
                subset.groupby(
                    [
                        "season",
                        "week",
                        "position",
                    ]
                )[
                    probability_column
                ]
                .rank(
                    method="first",
                    ascending=False,
                )
            )

            current.loc[
                subset.index,
                percentile_column,
            ] = subset[
                percentile_column
            ]

            current.loc[
                subset.index,
                rank_column,
            ] = subset[
                rank_column
            ]

    return current


# =========================================================
# LOCKED GPP POLICY
# =========================================================

def apply_gpp_policy(
    current,
):

    section(
        "APPLYING LOCKED GPP POLICY"
    )

    current = current.copy()

    # -----------------------------------------------------
    # Initialize shared threshold columns ONCE.
    #
    # This is the bug fix.
    # -----------------------------------------------------

    for threshold in ALL_THRESHOLDS:

        current[
            f"gpp_score_{threshold}"
        ] = np.nan

        current[
            f"gpp_percentile_{threshold}"
        ] = np.nan

        current[
            f"gpp_rank_{threshold}"
        ] = np.nan

        current[
            f"gpp_method_{threshold}"
        ] = None

    for position in POSITIONS:

        for threshold in (
            CEILING_THRESHOLDS[
                position
            ]
        ):

            (
                ridge_weight,
                ceiling_weight,
            ) = GPP_POLICY[
                (
                    position,
                    threshold,
                )
            ]

            score_column = (
                f"gpp_score_{threshold}"
            )

            percentile_column = (
                f"gpp_percentile_{threshold}"
            )

            rank_column = (
                f"gpp_rank_{threshold}"
            )

            method_column = (
                f"gpp_method_{threshold}"
            )

            ceiling_percentile_column = (
                f"ceiling_percentile_{threshold}"
            )

            mask = (
                (
                    current[
                        "position"
                    ]
                    ==
                    position
                )
                &
                (
                    current[
                        "production_status"
                    ]
                    ==
                    "MODEL_READY"
                )
            )

            subset = current.loc[
                mask
            ].copy()

            if subset.empty:

                continue

            subset[
                score_column
            ] = (
                ridge_weight
                *
                subset[
                    "ridge_percentile"
                ]
                +
                ceiling_weight
                *
                subset[
                    ceiling_percentile_column
                ]
            )

            subset[
                percentile_column
            ] = percentile_rank(
                subset,
                score_column,
            )

            subset[
                rank_column
            ] = (
                subset.groupby(
                    [
                        "season",
                        "week",
                        "position",
                    ]
                )[
                    score_column
                ]
                .rank(
                    method="first",
                    ascending=False,
                )
            )

            if ceiling_weight > 0:

                method = (
                    f"BLEND_"
                    f"{ridge_weight:.1f}_RIDGE_"
                    f"{ceiling_weight:.1f}_CEILING"
                )

            else:

                method = (
                    "RIDGE_ONLY"
                )

            current.loc[
                subset.index,
                score_column,
            ] = subset[
                score_column
            ]

            current.loc[
                subset.index,
                percentile_column,
            ] = subset[
                percentile_column
            ]

            current.loc[
                subset.index,
                rank_column,
            ] = subset[
                rank_column
            ]

            current.loc[
                subset.index,
                method_column,
            ] = method

            print(
                f"{position} "
                f"{threshold}+: "
                f"ridge={ridge_weight:.1f} "
                f"ceiling={ceiling_weight:.1f} "
                f"{method}"
            )

    return current


# =========================================================
# AUDIT
# =========================================================

def audit_output(
    current,
):

    section(
        "PRODUCTION PROJECTION AUDIT"
    )

    duplicate_rows = (
        current.groupby(
            [
                "game_id",
                "identity_key",
                "team",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    duplicate_rows = duplicate_rows[
        duplicate_rows[
            "n"
        ]
        >
        1
    ]

    ready = current[
        current[
            "production_status"
        ]
        ==
        "MODEL_READY"
    ].copy()

    cold = current[
        current[
            "production_status"
        ]
        ==
        "COLD_START"
    ].copy()

    ready_null_ridge = int(
        ready[
            "ridge_projection"
        ]
        .isna()
        .sum()
    )

    ridge_values = pd.to_numeric(
        ready[
            "ridge_projection"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    ready_nonfinite_ridge = int(
        (
            ~np.isfinite(
                ridge_values
            )
        ).sum()
    )

    ready_null_ridge_percentile = int(
        ready[
            "ridge_percentile"
        ]
        .isna()
        .sum()
    )

    ready_null_ridge_rank = int(
        ready[
            "ridge_rank"
        ]
        .isna()
        .sum()
    )

    invalid_ridge_percentile = int(
        (
            (
                pd.to_numeric(
                    ready[
                        "ridge_percentile"
                    ],
                    errors="coerce",
                )
                <
                0.0
            )
            |
            (
                pd.to_numeric(
                    ready[
                        "ridge_percentile"
                    ],
                    errors="coerce",
                )
                >
                1.0
            )
        ).sum()
    )

    invalid_ridge_rank = int(
        (
            pd.to_numeric(
                ready[
                    "ridge_rank"
                ],
                errors="coerce",
            )
            <
            1.0
        ).sum()
    )

    cold_scored_ridge = int(
        cold[
            "ridge_projection"
        ]
        .notna()
        .sum()
    )

    cold_ridge_percentile_count = int(
        cold[
            "ridge_percentile"
        ]
        .notna()
        .sum()
    )

    cold_ridge_rank_count = int(
        cold[
            "ridge_rank"
        ]
        .notna()
        .sum()
    )

    null_ready_probabilities = 0
    null_ready_ceiling_percentiles = 0
    null_ready_ceiling_ranks = 0
    invalid_probabilities = 0
    invalid_ceiling_percentiles = 0
    invalid_ceiling_ranks = 0

    null_ready_gpp_scores = 0
    null_ready_gpp_percentiles = 0
    null_ready_gpp_ranks = 0
    null_ready_gpp_methods = 0
    invalid_gpp_scores = 0
    invalid_gpp_percentiles = 0
    invalid_gpp_ranks = 0

    cold_probability_count = 0
    cold_ceiling_percentile_count = 0
    cold_ceiling_rank_count = 0
    cold_gpp_score_count = 0
    cold_gpp_percentile_count = 0
    cold_gpp_rank_count = 0
    cold_gpp_method_count = 0

    nonapplicable_output_count = 0

    for position in POSITIONS:

        position_ready = ready[
            ready[
                "position"
            ]
            ==
            position
        ].copy()

        position_cold = cold[
            cold[
                "position"
            ]
            ==
            position
        ].copy()

        applicable_thresholds = set(
            CEILING_THRESHOLDS[
                position
            ]
        )

        for threshold in ALL_THRESHOLDS:

            probability_column = (
                f"ceiling_prob_{threshold}"
            )

            ceiling_percentile_column = (
                f"ceiling_percentile_{threshold}"
            )

            ceiling_rank_column = (
                f"ceiling_rank_{threshold}"
            )

            gpp_score_column = (
                f"gpp_score_{threshold}"
            )

            gpp_percentile_column = (
                f"gpp_percentile_{threshold}"
            )

            gpp_rank_column = (
                f"gpp_rank_{threshold}"
            )

            gpp_method_column = (
                f"gpp_method_{threshold}"
            )

            output_columns = [
                probability_column,
                ceiling_percentile_column,
                ceiling_rank_column,
                gpp_score_column,
                gpp_percentile_column,
                gpp_rank_column,
                gpp_method_column,
            ]

            if threshold in applicable_thresholds:

                probability = pd.to_numeric(
                    position_ready[
                        probability_column
                    ],
                    errors="coerce",
                )

                ceiling_percentile = pd.to_numeric(
                    position_ready[
                        ceiling_percentile_column
                    ],
                    errors="coerce",
                )

                ceiling_rank = pd.to_numeric(
                    position_ready[
                        ceiling_rank_column
                    ],
                    errors="coerce",
                )

                gpp_score = pd.to_numeric(
                    position_ready[
                        gpp_score_column
                    ],
                    errors="coerce",
                )

                gpp_percentile = pd.to_numeric(
                    position_ready[
                        gpp_percentile_column
                    ],
                    errors="coerce",
                )

                gpp_rank = pd.to_numeric(
                    position_ready[
                        gpp_rank_column
                    ],
                    errors="coerce",
                )

                gpp_method = position_ready[
                    gpp_method_column
                ]

                null_ready_probabilities += int(
                    probability.isna().sum()
                )

                null_ready_ceiling_percentiles += int(
                    ceiling_percentile.isna().sum()
                )

                null_ready_ceiling_ranks += int(
                    ceiling_rank.isna().sum()
                )

                null_ready_gpp_scores += int(
                    gpp_score.isna().sum()
                )

                null_ready_gpp_percentiles += int(
                    gpp_percentile.isna().sum()
                )

                null_ready_gpp_ranks += int(
                    gpp_rank.isna().sum()
                )

                null_ready_gpp_methods += int(
                    gpp_method.isna().sum()
                )

                valid_probability = (
                    probability.dropna()
                )

                valid_ceiling_percentile = (
                    ceiling_percentile.dropna()
                )

                valid_ceiling_rank = (
                    ceiling_rank.dropna()
                )

                valid_gpp_score = (
                    gpp_score.dropna()
                )

                valid_gpp_percentile = (
                    gpp_percentile.dropna()
                )

                valid_gpp_rank = (
                    gpp_rank.dropna()
                )

                invalid_probabilities += int(
                    (
                        (
                            valid_probability < 0.0
                        )
                        |
                        (
                            valid_probability > 1.0
                        )
                    ).sum()
                )

                invalid_ceiling_percentiles += int(
                    (
                        (
                            valid_ceiling_percentile < 0.0
                        )
                        |
                        (
                            valid_ceiling_percentile > 1.0
                        )
                    ).sum()
                )

                invalid_ceiling_ranks += int(
                    (
                        valid_ceiling_rank < 1.0
                    ).sum()
                )

                invalid_gpp_scores += int(
                    (
                        (
                            valid_gpp_score < 0.0
                        )
                        |
                        (
                            valid_gpp_score > 1.0
                        )
                    ).sum()
                )

                invalid_gpp_percentiles += int(
                    (
                        (
                            valid_gpp_percentile < 0.0
                        )
                        |
                        (
                            valid_gpp_percentile > 1.0
                        )
                    ).sum()
                )

                invalid_gpp_ranks += int(
                    (
                        valid_gpp_rank < 1.0
                    ).sum()
                )

                cold_probability_count += int(
                    position_cold[
                        probability_column
                    ]
                    .notna()
                    .sum()
                )

                cold_ceiling_percentile_count += int(
                    position_cold[
                        ceiling_percentile_column
                    ]
                    .notna()
                    .sum()
                )

                cold_ceiling_rank_count += int(
                    position_cold[
                        ceiling_rank_column
                    ]
                    .notna()
                    .sum()
                )

                cold_gpp_score_count += int(
                    position_cold[
                        gpp_score_column
                    ]
                    .notna()
                    .sum()
                )

                cold_gpp_percentile_count += int(
                    position_cold[
                        gpp_percentile_column
                    ]
                    .notna()
                    .sum()
                )

                cold_gpp_rank_count += int(
                    position_cold[
                        gpp_rank_column
                    ]
                    .notna()
                    .sum()
                )

                cold_gpp_method_count += int(
                    position_cold[
                        gpp_method_column
                    ]
                    .notna()
                    .sum()
                )

            else:

                for column in output_columns:

                    nonapplicable_output_count += int(
                        current.loc[
                            current[
                                "position"
                            ]
                            ==
                            position,
                            column,
                        ]
                        .notna()
                        .sum()
                    )

    invalid_status = int(
        (
            ~current[
                "production_status"
            ].isin(
                [
                    "MODEL_READY",
                    "COLD_START",
                ]
            )
        ).sum()
    )

    print(
        f"Total rows: "
        f"{len(current)}"
    )

    print(
        f"Model-ready rows: "
        f"{len(ready)}"
    )

    print(
        f"Cold-start rows: "
        f"{len(cold)}"
    )

    print(
        f"Duplicate player-game rows: "
        f"{len(duplicate_rows)}"
    )

    print(
        f"MODEL_READY NULL ridge: "
        f"{ready_null_ridge}"
    )

    print(
        f"MODEL_READY non-finite ridge: "
        f"{ready_nonfinite_ridge}"
    )

    print(
        f"MODEL_READY NULL ridge percentile: "
        f"{ready_null_ridge_percentile}"
    )

    print(
        f"MODEL_READY NULL ridge rank: "
        f"{ready_null_ridge_rank}"
    )

    print(
        f"Invalid ridge percentiles: "
        f"{invalid_ridge_percentile}"
    )

    print(
        f"Invalid ridge ranks: "
        f"{invalid_ridge_rank}"
    )

    print(
        f"COLD_START ridge outputs: "
        f"{cold_scored_ridge}"
    )

    print(
        f"COLD_START ridge percentile outputs: "
        f"{cold_ridge_percentile_count}"
    )

    print(
        f"COLD_START ridge rank outputs: "
        f"{cold_ridge_rank_count}"
    )

    print(
        "MODEL_READY NULL applicable "
        "ceiling probabilities: "
        f"{null_ready_probabilities}"
    )

    print(
        "MODEL_READY NULL applicable "
        "ceiling percentiles: "
        f"{null_ready_ceiling_percentiles}"
    )

    print(
        "MODEL_READY NULL applicable "
        "ceiling ranks: "
        f"{null_ready_ceiling_ranks}"
    )

    print(
        f"Invalid ceiling probabilities: "
        f"{invalid_probabilities}"
    )

    print(
        f"Invalid ceiling percentiles: "
        f"{invalid_ceiling_percentiles}"
    )

    print(
        f"Invalid ceiling ranks: "
        f"{invalid_ceiling_ranks}"
    )

    print(
        "MODEL_READY NULL applicable "
        "GPP scores: "
        f"{null_ready_gpp_scores}"
    )

    print(
        "MODEL_READY NULL applicable "
        "GPP percentiles: "
        f"{null_ready_gpp_percentiles}"
    )

    print(
        "MODEL_READY NULL applicable "
        "GPP ranks: "
        f"{null_ready_gpp_ranks}"
    )

    print(
        "MODEL_READY NULL applicable "
        "GPP methods: "
        f"{null_ready_gpp_methods}"
    )

    print(
        f"Invalid GPP scores: "
        f"{invalid_gpp_scores}"
    )

    print(
        f"Invalid GPP percentiles: "
        f"{invalid_gpp_percentiles}"
    )

    print(
        f"Invalid GPP ranks: "
        f"{invalid_gpp_ranks}"
    )

    print(
        f"COLD_START ceiling outputs: "
        f"{cold_probability_count}"
    )

    print(
        f"COLD_START ceiling percentile outputs: "
        f"{cold_ceiling_percentile_count}"
    )

    print(
        f"COLD_START ceiling rank outputs: "
        f"{cold_ceiling_rank_count}"
    )

    print(
        f"COLD_START GPP score outputs: "
        f"{cold_gpp_score_count}"
    )

    print(
        f"COLD_START GPP percentile outputs: "
        f"{cold_gpp_percentile_count}"
    )

    print(
        f"COLD_START GPP rank outputs: "
        f"{cold_gpp_rank_count}"
    )

    print(
        f"COLD_START GPP method outputs: "
        f"{cold_gpp_method_count}"
    )

    print(
        f"Non-applicable threshold outputs: "
        f"{nonapplicable_output_count}"
    )

    print(
        f"Invalid production statuses: "
        f"{invalid_status}"
    )

    audit_rows = [
        {
            "audit":
                "total_rows",
            "value":
                len(current),
        },
        {
            "audit":
                "model_ready_rows",
            "value":
                len(ready),
        },
        {
            "audit":
                "cold_start_rows",
            "value":
                len(cold),
        },
        {
            "audit":
                "duplicate_player_game_rows",
            "value":
                len(duplicate_rows),
        },
        {
            "audit":
                "ready_null_ridge",
            "value":
                ready_null_ridge,
        },
        {
            "audit":
                "ready_nonfinite_ridge",
            "value":
                ready_nonfinite_ridge,
        },
        {
            "audit":
                "ready_null_ridge_percentile",
            "value":
                ready_null_ridge_percentile,
        },
        {
            "audit":
                "ready_null_ridge_rank",
            "value":
                ready_null_ridge_rank,
        },
        {
            "audit":
                "invalid_ridge_percentile",
            "value":
                invalid_ridge_percentile,
        },
        {
            "audit":
                "invalid_ridge_rank",
            "value":
                invalid_ridge_rank,
        },
        {
            "audit":
                "cold_start_ridge_outputs",
            "value":
                cold_scored_ridge,
        },
        {
            "audit":
                "cold_start_ridge_percentile_outputs",
            "value":
                cold_ridge_percentile_count,
        },
        {
            "audit":
                "cold_start_ridge_rank_outputs",
            "value":
                cold_ridge_rank_count,
        },
        {
            "audit":
                "ready_null_ceiling_probabilities",
            "value":
                null_ready_probabilities,
        },
        {
            "audit":
                "ready_null_ceiling_percentiles",
            "value":
                null_ready_ceiling_percentiles,
        },
        {
            "audit":
                "ready_null_ceiling_ranks",
            "value":
                null_ready_ceiling_ranks,
        },
        {
            "audit":
                "invalid_ceiling_probabilities",
            "value":
                invalid_probabilities,
        },
        {
            "audit":
                "invalid_ceiling_percentiles",
            "value":
                invalid_ceiling_percentiles,
        },
        {
            "audit":
                "invalid_ceiling_ranks",
            "value":
                invalid_ceiling_ranks,
        },
        {
            "audit":
                "ready_null_gpp_scores",
            "value":
                null_ready_gpp_scores,
        },
        {
            "audit":
                "ready_null_gpp_percentiles",
            "value":
                null_ready_gpp_percentiles,
        },
        {
            "audit":
                "ready_null_gpp_ranks",
            "value":
                null_ready_gpp_ranks,
        },
        {
            "audit":
                "ready_null_gpp_methods",
            "value":
                null_ready_gpp_methods,
        },
        {
            "audit":
                "invalid_gpp_scores",
            "value":
                invalid_gpp_scores,
        },
        {
            "audit":
                "invalid_gpp_percentiles",
            "value":
                invalid_gpp_percentiles,
        },
        {
            "audit":
                "invalid_gpp_ranks",
            "value":
                invalid_gpp_ranks,
        },
        {
            "audit":
                "cold_start_ceiling_outputs",
            "value":
                cold_probability_count,
        },
        {
            "audit":
                "cold_start_ceiling_percentile_outputs",
            "value":
                cold_ceiling_percentile_count,
        },
        {
            "audit":
                "cold_start_ceiling_rank_outputs",
            "value":
                cold_ceiling_rank_count,
        },
        {
            "audit":
                "cold_start_gpp_score_outputs",
            "value":
                cold_gpp_score_count,
        },
        {
            "audit":
                "cold_start_gpp_percentile_outputs",
            "value":
                cold_gpp_percentile_count,
        },
        {
            "audit":
                "cold_start_gpp_rank_outputs",
            "value":
                cold_gpp_rank_count,
        },
        {
            "audit":
                "cold_start_gpp_method_outputs",
            "value":
                cold_gpp_method_count,
        },
        {
            "audit":
                "nonapplicable_threshold_outputs",
            "value":
                nonapplicable_output_count,
        },
        {
            "audit":
                "invalid_production_status",
            "value":
                invalid_status,
        },
    ]

    audit = pd.DataFrame(
        audit_rows
    )

    problems = (
        len(
            duplicate_rows
        )
        +
        ready_null_ridge
        +
        ready_nonfinite_ridge
        +
        ready_null_ridge_percentile
        +
        ready_null_ridge_rank
        +
        invalid_ridge_percentile
        +
        invalid_ridge_rank
        +
        cold_scored_ridge
        +
        cold_ridge_percentile_count
        +
        cold_ridge_rank_count
        +
        null_ready_probabilities
        +
        null_ready_ceiling_percentiles
        +
        null_ready_ceiling_ranks
        +
        invalid_probabilities
        +
        invalid_ceiling_percentiles
        +
        invalid_ceiling_ranks
        +
        null_ready_gpp_scores
        +
        null_ready_gpp_percentiles
        +
        null_ready_gpp_ranks
        +
        null_ready_gpp_methods
        +
        invalid_gpp_scores
        +
        invalid_gpp_percentiles
        +
        invalid_gpp_ranks
        +
        cold_probability_count
        +
        cold_ceiling_percentile_count
        +
        cold_ceiling_rank_count
        +
        cold_gpp_score_count
        +
        cold_gpp_percentile_count
        +
        cold_gpp_rank_count
        +
        cold_gpp_method_count
        +
        nonapplicable_output_count
        +
        invalid_status
    )

    if problems:

        raise RuntimeError(
            "Production projection audit failed."
        )

    print()

    print(
        "PASS: production projections "
        "passed structural audits."
    )

    return audit


# =========================================================
# LEADERS
# =========================================================

def print_leaders(
    current,
):

    section(
        "CURRENT PRODUCTION LEADERS"
    )

    ready = current[
        current[
            "production_status"
        ]
        ==
        "MODEL_READY"
    ].copy()

    for position in POSITIONS:

        rows = ready[
            ready[
                "position"
            ]
            ==
            position
        ].copy()

        rows = rows.sort_values(
            [
                "ridge_rank",
                "player_display_name",
            ]
        ).head(
            15
        )

        print()
        print(
            position
        )
        print(
            "-" * 78
        )

        columns = [
            "player_display_name",
            "team",
            "opponent_team",
            "player_history_games",
            "ridge_projection",
            "ridge_percentile",
            "ridge_rank",
        ]

        for threshold in (
            CEILING_THRESHOLDS[
                position
            ]
        ):

            columns.extend(
                [
                    f"ceiling_prob_{threshold}",
                    f"ceiling_percentile_{threshold}",
                    f"ceiling_rank_{threshold}",
                    f"gpp_score_{threshold}",
                    f"gpp_percentile_{threshold}",
                    f"gpp_rank_{threshold}",
                    f"gpp_method_{threshold}",
                ]
            )

        print(
            rows[
                columns
            ]
            .to_string(
                index=False
            )
        )


# =========================================================
# EXPORT
# =========================================================

def export_results(
    current,
    audit,
    model_summary,
):

    section(
        "EXPORTING PRODUCTION PROJECTIONS"
    )

    current.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    current.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    audit.to_csv(
        AUDIT_CSV,
        index=False,
    )

    audit.to_parquet(
        AUDIT_PARQUET,
        index=False,
    )

    model_summary.to_csv(
        MODEL_SUMMARY_CSV,
        index=False,
    )

    model_summary.to_parquet(
        MODEL_SUMMARY_PARQUET,
        index=False,
    )

    print(
        f"Projection rows: "
        f"{len(current)}"
    )

    print(
        f"Model summary rows: "
        f"{len(model_summary)}"
    )

    print()

    print(
        f"Projection CSV: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Projection Parquet: "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"Audit CSV: "
        f"{AUDIT_CSV}"
    )

    print(
        f"Model summary CSV: "
        f"{MODEL_SUMMARY_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_production_projection():

    section(
        "NFL PRODUCTION PROJECTION ENGINE"
    )

    print(
        "Training window: 2023-2025 base "
        "+ rolling completed prior weeks "
        "from 2026 onward."
    )

    print(
        "Current-season realized rows are "
        "eligible only when their season/week "
        "is strictly before each target "
        "season/week."
    )

    core = load_core_features()

    current_all = load_current_features(
        core
    )

    pairs = (
        current_all[
            [
                "season",
                "week",
            ]
        ]
        .copy()
    )

    pairs[
        "season"
    ] = pd.to_numeric(
        pairs[
            "season"
        ],
        errors="coerce",
    )

    pairs[
        "week"
    ] = pd.to_numeric(
        pairs[
            "week"
        ],
        errors="coerce",
    )

    if (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Current inference pool contains "
            "NULL/non-numeric season or week."
        )

    pairs = (
        pairs[
            [
                "season",
                "week",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "week",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if pairs.empty:

        raise RuntimeError(
            "Current inference pool contains "
            "no season/week targets."
        )

    scored_frames = []
    summary_frames = []

    for _, pair in pairs.iterrows():

        target_season = int(
            pair[
                "season"
            ]
        )

        target_week = int(
            pair[
                "week"
            ]
        )

        current = current_all[
            (
                pd.to_numeric(
                    current_all[
                        "season"
                    ],
                    errors="coerce",
                )
                ==
                target_season
            )
            &
            (
                pd.to_numeric(
                    current_all[
                        "week"
                    ],
                    errors="coerce",
                )
                ==
                target_week
            )
        ].copy()

        # Preserve the existing one-target invariant as an explicit
        # safety gate for every independent scoring batch.
        (
            resolved_season,
            resolved_week,
        ) = resolve_target_season_week(
            current
        )

        if (
            resolved_season
            !=
            target_season
            or
            resolved_week
            !=
            target_week
        ):

            raise RuntimeError(
                "Target slice authority mismatch."
            )

        section(
            f"SCORING TARGET "
            f"{target_season} WEEK {target_week}"
        )

        print(
            f"Target rows: "
            f"{len(current)}"
        )

        historical = load_training_matrix(
            core,
            target_season,
            target_week,
        )

        (
            current,
            summary_rows,
        ) = score_ridge_models(
            historical,
            current,
            core,
        )

        injuries = load_injury_consensus(
            target_season, target_week
        )
        replacement_profiles = build_replacement_profiles(
            load_replacement_history()
        )
        current, injury_audit = apply_live_injury_reforecast(
            current,
            injuries,
            replacement_profiles,
        )
        (
            current,
            summary_rows,
        ) = score_ceiling_models(
            historical,
            current,
            core,
            summary_rows,
        )

        current = attach_ridge_rank(
            current
        )

        current = attach_ceiling_percentiles(
            current
        )

        current = apply_gpp_policy(
            current
        )

        target_audit = audit_output(
            current
        )

        target_summary = pd.DataFrame(
            summary_rows
        )

        if not target_summary.empty:

            target_summary.insert(
                0,
                "week",
                target_week,
            )

            target_summary.insert(
                0,
                "season",
                target_season,
            )

        print_leaders(
            current
        )

        scored_frames.append(
            current
        )

        summary_frames.append(
            target_summary
        )

    current = pd.concat(
        scored_frames,
        ignore_index=True,
    )

    current = current.sort_values(
        [
            "season",
            "week",
            "game_id",
            "position",
            "team",
            "player_display_name",
        ]
    ).reset_index(
        drop=True
    )

    # Re-audit the combined export as a final structural gate.
    audit = audit_output(
        current
    )

    if summary_frames:

        model_summary = pd.concat(
            summary_frames,
            ignore_index=True,
        )

    else:

        model_summary = pd.DataFrame()

    section(
        "COMBINED INFERENCE TARGET SUMMARY"
    )

    print(
        current.groupby(
            [
                "season",
                "week",
                "production_status",
            ]
        )
        .size()
        .to_string()
    )

    export_results(
        current,
        audit,
        model_summary,
    )

    section(
        "PRODUCTION PROJECTION BUILD SUCCESSFUL"
    )

    ready_count = int(
        (
            current[
                "production_status"
            ]
            ==
            "MODEL_READY"
        ).sum()
    )

    cold_count = int(
        (
            current[
                "production_status"
            ]
            ==
            "COLD_START"
        ).sum()
    )

    print()

    print(
        f"Model-ready players: "
        f"{ready_count}"
    )

    print(
        f"Cold-start players: "
        f"{cold_count}"
    )

    print()

    print(
        "Historical fit = 2023-2025 base "
        "plus rolling prior-week results "
        "from 2026 onward."
    )

    print(
        "Live inference targets:"
    )

    for _, pair in pairs.iterrows():

        print(
            f"  {int(pair['season'])} "
            f"week {int(pair['week'])}"
        )

    print(
        "Each target is trained/scored "
        "independently with the existing "
        "one-season/week safety invariant."
    )

    print(
        "Ridge projection = "
        "point-projection component."
    )

    print(
        "Direct ceiling = "
        "secondary threshold component."
    )

    print(
        "Promoted blends only = "
        "RB20, WR15, WR25, "
        "TE10, TE15, TE20."
    )

    print(
        "All other GPP thresholds = "
        "ridge-only."
    )

    print(
        "No arbitrary primary GPP threshold "
        "is designated."
    )

    print(
        "Cold starts are retained but "
        "not model-scored."
    )


if __name__ == "__main__":

    run_production_projection()
