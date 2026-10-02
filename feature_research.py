from pathlib import Path

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

DEVELOPMENT_SEASONS = [
    2023,
    2024,
]

VALIDATION_SEASONS = [
    2025,
]

POSITIONS = [
    "QB",
    "RB",
    "WR",
    "TE",
]

CEILING_THRESHOLDS = [
    15.0,
    20.0,
    25.0,
    30.0,
]


# =========================================================
# OUTPUT PATHS
# =========================================================

RESEARCH_SUMMARY_CSV = (
    CSV_DIR / "nfl_feature_research_summary.csv"
)

BUCKET_RESEARCH_CSV = (
    CSV_DIR / "nfl_feature_research_buckets.csv"
)

CEILING_RESEARCH_CSV = (
    CSV_DIR / "nfl_feature_research_ceilings.csv"
)

SELECTED_FEATURES_CSV = (
    CSV_DIR / "nfl_feature_research_selected.csv"
)

VALIDATION_REPORT_CSV = (
    CSV_DIR / "nfl_feature_validation_2025.csv"
)

RESEARCH_SUMMARY_PARQUET = (
    PARQUET_DIR / "nfl_feature_research_summary.parquet"
)

BUCKET_RESEARCH_PARQUET = (
    PARQUET_DIR / "nfl_feature_research_buckets.parquet"
)

CEILING_RESEARCH_PARQUET = (
    PARQUET_DIR / "nfl_feature_research_ceilings.parquet"
)

SELECTED_FEATURES_PARQUET = (
    PARQUET_DIR / "nfl_feature_research_selected.parquet"
)

VALIDATION_REPORT_PARQUET = (
    PARQUET_DIR / "nfl_feature_validation_2025.parquet"
)


# =========================================================
# POSITION-SPECIFIC CANDIDATES
# =========================================================

POSITION_FEATURES = {

    # -----------------------------------------------------
    # QB
    #
    # Generic "opportunities" is NOT used heavily here
    # because player_weekly_usage defines opportunities as
    # carries + targets, which is not QB total involvement.
    # -----------------------------------------------------

    "QB": [
        "fd_last",
        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",
        "fanduel_trend",

        "snaps_last",
        "snaps_avg_3",
        "snap_pct_last",
        "snap_pct_avg_3",

        "carries_last",
        "carries_avg_3",
        "carries_avg_5",
        "carry_trend",

        "team_points_for_last",
        "team_points_for_avg_3",
        "team_points_for_avg_5",
        "team_scoring_trend",

        "team_offensive_plays_avg_3",
        "team_offensive_plays_avg_5",
        "team_pace_trend",

        "team_pass_attempts_avg_3",
        "team_pass_attempts_avg_5",

        "team_pass_rate_avg_3",
        "team_pass_rate_avg_5",
        "team_pass_rate_trend",

        "team_passing_yards_avg_3",
        "team_passing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_points_allowed_avg_5",

        "opponent_pass_yards_allowed_avg_3",
        "opponent_pass_tds_allowed_avg_3",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",
        "dvp_fd_allowed_trend",
    ],

    # -----------------------------------------------------
    # RB
    # -----------------------------------------------------

    "RB": [
        "fd_last",
        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",
        "fanduel_trend",

        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "opportunities_max_5",
        "opportunity_trend",

        "touches_last",
        "touches_avg_3",
        "touches_avg_5",

        "snaps_last",
        "snaps_avg_3",
        "snaps_avg_5",

        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",
        "snap_trend",

        "targets_last",
        "targets_avg_3",
        "targets_avg_5",
        "target_trend",

        "carries_last",
        "carries_avg_3",
        "carries_avg_5",
        "carry_trend",

        "receptions_avg_3",

        "yards_last",
        "yards_avg_3",
        "yards_avg_5",

        "target_share_last",
        "target_share_avg_3",

        "wopr_last",
        "wopr_avg_3",

        "fd_per_snap_avg_3",
        "fd_per_touch_avg_3",
        "yards_per_opportunity_avg_3",

        "role_expansion_last",
        "role_expansions_3",
        "high_usage_games_3",
        "starter_usage_games_3",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "team_points_for_avg_3",
        "team_points_for_avg_5",
        "team_scoring_trend",

        "team_offensive_plays_avg_3",
        "team_pace_trend",

        "team_rush_attempts_avg_3",
        "team_rush_attempts_avg_5",

        "team_rush_rate_avg_3",
        "team_rush_rate_avg_5",

        "team_rushing_yards_avg_3",
        "team_rushing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_rush_yards_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",

        "dvp_targets_allowed_avg_3",
        "dvp_carries_allowed_avg_3",
        "dvp_receptions_allowed_avg_3",

        "dvp_receiving_yards_allowed_avg_3",
        "dvp_rushing_yards_allowed_avg_3",

        "dvp_receiving_tds_allowed_avg_3",
        "dvp_rushing_tds_allowed_avg_3",

        "dvp_opportunities_allowed_avg_3",

        "dvp_fd_allowed_trend",
        "dvp_opportunity_allowed_trend",
    ],

    # -----------------------------------------------------
    # WR
    # -----------------------------------------------------

    "WR": [
        "fd_last",
        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",
        "fanduel_trend",

        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "opportunity_trend",

        "snaps_last",
        "snaps_avg_3",
        "snaps_avg_5",

        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",
        "snap_trend",

        "targets_last",
        "targets_avg_3",
        "targets_avg_5",
        "target_trend",

        "receptions_avg_3",

        "yards_last",
        "yards_avg_3",
        "yards_avg_5",

        "target_share_last",
        "target_share_avg_3",

        "air_yards_share_last",
        "air_yards_share_avg_3",

        "wopr_last",
        "wopr_avg_3",

        "fd_per_snap_avg_3",
        "fd_per_touch_avg_3",
        "yards_per_opportunity_avg_3",

        "role_expansion_last",
        "role_expansions_3",
        "high_usage_games_3",
        "starter_usage_games_3",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "team_points_for_avg_3",
        "team_points_for_avg_5",
        "team_scoring_trend",

        "team_offensive_plays_avg_3",
        "team_pace_trend",

        "team_pass_attempts_avg_3",
        "team_pass_attempts_avg_5",

        "team_pass_rate_avg_3",
        "team_pass_rate_avg_5",

        "team_passing_yards_avg_3",
        "team_passing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_pass_yards_allowed_avg_3",
        "opponent_pass_tds_allowed_avg_3",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",

        "dvp_targets_allowed_avg_3",
        "dvp_receptions_allowed_avg_3",

        "dvp_receiving_yards_allowed_avg_3",
        "dvp_receiving_tds_allowed_avg_3",

        "dvp_opportunities_allowed_avg_3",

        "dvp_fd_allowed_trend",
        "dvp_opportunity_allowed_trend",
    ],

    # -----------------------------------------------------
    # TE
    # -----------------------------------------------------

    "TE": [
        "fd_last",
        "fd_avg_3",
        "fd_avg_5",
        "fd_max_5",
        "fd_std_5",
        "fanduel_trend",

        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "opportunity_trend",

        "snaps_last",
        "snaps_avg_3",
        "snaps_avg_5",

        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",
        "snap_trend",

        "targets_last",
        "targets_avg_3",
        "targets_avg_5",
        "target_trend",

        "receptions_avg_3",

        "yards_last",
        "yards_avg_3",
        "yards_avg_5",

        "target_share_last",
        "target_share_avg_3",

        "air_yards_share_last",
        "air_yards_share_avg_3",

        "wopr_last",
        "wopr_avg_3",

        "fd_per_snap_avg_3",
        "fd_per_touch_avg_3",
        "yards_per_opportunity_avg_3",

        "role_expansion_last",
        "role_expansions_3",
        "high_usage_games_3",
        "starter_usage_games_3",

        "established_role_flag",
        "rising_role_flag",
        "declining_role_flag",

        "team_points_for_avg_3",
        "team_points_for_avg_5",
        "team_scoring_trend",

        "team_offensive_plays_avg_3",
        "team_pace_trend",

        "team_pass_attempts_avg_3",
        "team_pass_attempts_avg_5",

        "team_pass_rate_avg_3",
        "team_pass_rate_avg_5",

        "team_passing_yards_avg_3",
        "team_passing_tds_avg_3",

        "opponent_points_allowed_avg_3",
        "opponent_pass_yards_allowed_avg_3",
        "opponent_pass_tds_allowed_avg_3",

        "dvp_fd_allowed_last",
        "dvp_fd_allowed_avg_3",
        "dvp_fd_allowed_avg_5",

        "dvp_targets_allowed_avg_3",
        "dvp_receptions_allowed_avg_3",

        "dvp_receiving_yards_allowed_avg_3",
        "dvp_receiving_tds_allowed_avg_3",

        "dvp_opportunities_allowed_avg_3",

        "dvp_fd_allowed_trend",
        "dvp_opportunity_allowed_trend",
    ],
}


# =========================================================
# HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_corr(
    x,
    y,
):

    temp = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(temp) < 20:

        return 0.0

    if (
        temp["x"].nunique() <= 1
        or
        temp["y"].nunique() <= 1
    ):

        return 0.0

    value = temp[
        "x"
    ].corr(
        temp["y"]
    )

    if pd.isna(value):

        return 0.0

    return float(value)


def direction_label(
    value,
):

    if value > 0.01:

        return "POSITIVE"

    if value < -0.01:

        return "NEGATIVE"

    return "FLAT"


def safe_mean(series):

    value = pd.to_numeric(
        series,
        errors="coerce",
    ).mean()

    if pd.isna(value):

        return 0.0

    return float(value)


def safe_rate(series):

    if len(series) == 0:

        return 0.0

    value = pd.to_numeric(
        series,
        errors="coerce",
    ).mean()

    if pd.isna(value):

        return 0.0

    return float(value)


def create_quantile_bucket(
    series,
):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.notna()

    output = pd.Series(
        index=series.index,
        dtype="object",
    )

    output[:] = None

    if valid.sum() < 5:

        return output

    ranks = (
        numeric[
            valid
        ]
        .rank(
            method="first",
            pct=True,
        )
    )

    labels = pd.cut(
        ranks,
        bins=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        labels=[
            "Q1_LOW",
            "Q2",
            "Q3",
            "Q4",
            "Q5_HIGH",
        ],
        include_lowest=True,
    )

    output.loc[
        valid
    ] = labels.astype(str)

    return output


def favorable_bucket(
    correlation,
):

    if correlation >= 0:

        return "Q5_HIGH"

    return "Q1_LOW"


def unfavorable_bucket(
    correlation,
):

    if correlation >= 0:

        return "Q1_LOW"

    return "Q5_HIGH"


# =========================================================
# LOAD MATRIX
# =========================================================

def load_feature_matrix():

    section(
        "LOADING DFS FEATURE MATRIX"
    )

    with get_connection() as conn:

        df = pd.read_sql_query(
            """
            SELECT *

            FROM dfs_feature_matrix

            ORDER BY
                season,
                week,
                game_id,
                team,
                position,
                player_id
            """,
            conn,
        )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print()

    print(
        "Season counts:"
    )

    print(
        df.groupby(
            "season"
        ).size().to_string()
    )

    return df


# =========================================================
# FEATURE VALIDATION
# =========================================================

def validate_feature_configuration(
    df,
):

    section(
        "VALIDATING FEATURE CONFIGURATION"
    )

    missing = []

    for position in POSITIONS:

        for feature in POSITION_FEATURES[
            position
        ]:

            if feature not in df.columns:

                missing.append(
                    (
                        position,
                        feature,
                    )
                )

    if missing:

        print(
            "Missing configured features:"
        )

        for position, feature in missing:

            print(
                f"{position}: {feature}"
            )

        raise RuntimeError(
            "Feature research stopped because "
            "configured matrix columns are missing."
        )

    print(
        "PASS: all configured candidate "
        "features exist in the matrix."
    )


# =========================================================
# SAMPLE SUMMARY
# =========================================================

def print_sample_summary(
    df,
):

    section(
        "RESEARCH SAMPLE"
    )

    development = df[
        df[
            "season"
        ].isin(
            DEVELOPMENT_SEASONS
        )
    ]

    validation = df[
        df[
            "season"
        ].isin(
            VALIDATION_SEASONS
        )
    ]

    print(
        "DEVELOPMENT"
    )

    print(
        f"Seasons: {DEVELOPMENT_SEASONS}"
    )

    print(
        f"Rows: {len(development)}"
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        f"Seasons: {VALIDATION_SEASONS}"
    )

    print(
        f"Rows: {len(validation)}"
    )

    print()

    for position in POSITIONS:

        dev_count = len(
            development[
                development[
                    "position"
                ]
                ==
                position
            ]
        )

        val_count = len(
            validation[
                validation[
                    "position"
                ]
                ==
                position
            ]
        )

        print(
            f"{position}: "
            f"DEV {dev_count} | "
            f"VAL {val_count}"
        )


# =========================================================
# BUCKET ANALYSIS
# =========================================================

def analyze_buckets(
    data,
    position,
    feature,
    split_name,
):

    working = data[
        [
            feature,
            "target_fanduel_points",
        ]
    ].copy()

    working[
        feature
    ] = pd.to_numeric(
        working[
            feature
        ],
        errors="coerce",
    )

    working[
        "target_fanduel_points"
    ] = pd.to_numeric(
        working[
            "target_fanduel_points"
        ],
        errors="coerce",
    )

    working = working.dropna()

    if len(working) < 20:

        return []

    working[
        "bucket"
    ] = create_quantile_bucket(
        working[
            feature
        ]
    )

    rows = []

    for bucket in [
        "Q1_LOW",
        "Q2",
        "Q3",
        "Q4",
        "Q5_HIGH",
    ]:

        bucket_df = working[
            working[
                "bucket"
            ]
            ==
            bucket
        ]

        if bucket_df.empty:

            continue

        row = {
            "split":
                split_name,

            "position":
                position,

            "feature":
                feature,

            "bucket":
                bucket,

            "sample_size":
                len(
                    bucket_df
                ),

            "feature_mean":
                safe_mean(
                    bucket_df[
                        feature
                    ]
                ),

            "actual_fd_mean":
                safe_mean(
                    bucket_df[
                        "target_fanduel_points"
                    ]
                ),

            "actual_fd_median":
                float(
                    bucket_df[
                        "target_fanduel_points"
                    ]
                    .median()
                ),
        }

        for threshold in (
            CEILING_THRESHOLDS
        ):

            row[
                f"hit_rate_{int(threshold)}"
            ] = safe_rate(
                (
                    bucket_df[
                        "target_fanduel_points"
                    ]
                    >=
                    threshold
                ).astype(int)
            )

        rows.append(
            row
        )

    return rows


# =========================================================
# CEILING ANALYSIS
# =========================================================

def analyze_ceilings(
    data,
    position,
    feature,
    split_name,
    correlation,
):

    working = data[
        [
            feature,
            "target_fanduel_points",
        ]
    ].copy()

    working[
        feature
    ] = pd.to_numeric(
        working[
            feature
        ],
        errors="coerce",
    )

    working[
        "target_fanduel_points"
    ] = pd.to_numeric(
        working[
            "target_fanduel_points"
        ],
        errors="coerce",
    )

    working = working.dropna()

    if len(working) < 20:

        return []

    working[
        "bucket"
    ] = create_quantile_bucket(
        working[
            feature
        ]
    )

    good_bucket = favorable_bucket(
        correlation
    )

    bad_bucket = unfavorable_bucket(
        correlation
    )

    favorable = working[
        working[
            "bucket"
        ]
        ==
        good_bucket
    ]

    unfavorable = working[
        working[
            "bucket"
        ]
        ==
        bad_bucket
    ]

    rows = []

    for threshold in (
        CEILING_THRESHOLDS
    ):

        overall_rate = safe_rate(
            (
                working[
                    "target_fanduel_points"
                ]
                >=
                threshold
            ).astype(int)
        )

        favorable_rate = safe_rate(
            (
                favorable[
                    "target_fanduel_points"
                ]
                >=
                threshold
            ).astype(int)
        )

        unfavorable_rate = safe_rate(
            (
                unfavorable[
                    "target_fanduel_points"
                ]
                >=
                threshold
            ).astype(int)
        )

        if overall_rate > 0:

            favorable_lift = (
                favorable_rate
                /
                overall_rate
            )

        else:

            favorable_lift = 0.0

        rows.append(
            {
                "split":
                    split_name,

                "position":
                    position,

                "feature":
                    feature,

                "threshold":
                    threshold,

                "correlation":
                    correlation,

                "signal_direction":
                    direction_label(
                        correlation
                    ),

                "favorable_bucket":
                    good_bucket,

                "sample_size":
                    len(
                        working
                    ),

                "overall_hit_rate":
                    overall_rate,

                "favorable_hit_rate":
                    favorable_rate,

                "unfavorable_hit_rate":
                    unfavorable_rate,

                "favorable_vs_overall_lift":
                    favorable_lift,

                "favorable_minus_unfavorable":
                    (
                        favorable_rate
                        -
                        unfavorable_rate
                    ),
            }
        )

    return rows


# =========================================================
# DEVELOPMENT FEATURE ANALYSIS
# =========================================================

def analyze_development_feature(
    dev_df,
    position,
    feature,
):

    target = (
        dev_df[
            "target_fanduel_points"
        ]
    )

    feature_values = (
        dev_df[
            feature
        ]
    )

    correlation = safe_corr(
        feature_values,
        target,
    )

    working = dev_df[
        [
            feature,
            "target_fanduel_points",
        ]
    ].copy()

    working[
        feature
    ] = pd.to_numeric(
        working[
            feature
        ],
        errors="coerce",
    )

    working[
        "target_fanduel_points"
    ] = pd.to_numeric(
        working[
            "target_fanduel_points"
        ],
        errors="coerce",
    )

    working = working.dropna()

    working[
        "bucket"
    ] = create_quantile_bucket(
        working[
            feature
        ]
    )

    good_bucket = favorable_bucket(
        correlation
    )

    bad_bucket = unfavorable_bucket(
        correlation
    )

    favorable = working[
        working[
            "bucket"
        ]
        ==
        good_bucket
    ]

    unfavorable = working[
        working[
            "bucket"
        ]
        ==
        bad_bucket
    ]

    overall_fd = safe_mean(
        working[
            "target_fanduel_points"
        ]
    )

    favorable_fd = safe_mean(
        favorable[
            "target_fanduel_points"
        ]
    )

    unfavorable_fd = safe_mean(
        unfavorable[
            "target_fanduel_points"
        ]
    )

    if overall_fd != 0:

        favorable_mean_lift = (
            favorable_fd
            -
            overall_fd
        ) / abs(
            overall_fd
        )

    else:

        favorable_mean_lift = 0.0

    # -----------------------------------------------------
    # 20+ ceiling is used as a common cross-position
    # development ceiling component.
    #
    # Other thresholds remain available in the ceiling
    # research output.
    # -----------------------------------------------------

    threshold = 20.0

    overall_20 = safe_rate(
        (
            working[
                "target_fanduel_points"
            ]
            >=
            threshold
        ).astype(int)
    )

    favorable_20 = safe_rate(
        (
            favorable[
                "target_fanduel_points"
            ]
            >=
            threshold
        ).astype(int)
    )

    unfavorable_20 = safe_rate(
        (
            unfavorable[
                "target_fanduel_points"
            ]
            >=
            threshold
        ).astype(int)
    )

    if overall_20 > 0:

        ceiling_lift_20 = (
            favorable_20
            /
            overall_20
        )

    else:

        ceiling_lift_20 = 0.0

    # =====================================================
    # DETERMINISTIC DEVELOPMENT SCORE
    #
    # IMPORTANT:
    # This score uses development data only.
    # 2025 is not involved.
    # =====================================================

    correlation_score = (
        min(
            abs(
                correlation
            )
            /
            0.30,
            1.0,
        )
        *
        40.0
    )

    positive_mean_lift = max(
        favorable_mean_lift,
        0.0,
    )

    mean_lift_score = (
        min(
            positive_mean_lift
            /
            0.50,
            1.0,
        )
        *
        25.0
    )

    positive_ceiling_lift = max(
        ceiling_lift_20
        -
        1.0,
        0.0,
    )

    ceiling_score = (
        min(
            positive_ceiling_lift
            /
            1.0,
            1.0,
        )
        *
        25.0
    )

    sample_score = (
        min(
            len(
                working
            )
            /
            2000.0,
            1.0,
        )
        *
        10.0
    )

    development_score = (
        correlation_score
        +
        mean_lift_score
        +
        ceiling_score
        +
        sample_score
    )

    # =====================================================
    # DEVELOPMENT CLASSIFICATION ONLY
    #
    # The holdout season cannot change this classification.
    # =====================================================

    if (
        abs(
            correlation
        )
        >=
        0.10

        and

        favorable_fd
        >
        unfavorable_fd

        and

        development_score
        >=
        40.0
    ):

        development_class = (
            "PASS"
        )

    elif (
        abs(
            correlation
        )
        >=
        0.05

        and

        favorable_fd
        >
        unfavorable_fd
    ):

        development_class = (
            "WEAK"
        )

    else:

        development_class = (
            "FAIL"
        )

    return {
        "position":
            position,

        "feature":
            feature,

        "development_seasons":
            ",".join(
                str(
                    season
                )
                for season in (
                    DEVELOPMENT_SEASONS
                )
            ),

        "development_sample":
            len(
                working
            ),

        "development_feature_mean":
            safe_mean(
                working[
                    feature
                ]
            ),

        "development_target_mean":
            overall_fd,

        "development_correlation":
            correlation,

        "development_direction":
            direction_label(
                correlation
            ),

        "favorable_bucket":
            good_bucket,

        "unfavorable_bucket":
            bad_bucket,

        "favorable_fd_mean":
            favorable_fd,

        "unfavorable_fd_mean":
            unfavorable_fd,

        "favorable_mean_lift_pct":
            favorable_mean_lift,

        "overall_20_hit_rate":
            overall_20,

        "favorable_20_hit_rate":
            favorable_20,

        "unfavorable_20_hit_rate":
            unfavorable_20,

        "ceiling_lift_20":
            ceiling_lift_20,

        "development_score":
            development_score,

        "development_class":
            development_class,
    }


# =========================================================
# VALIDATION FEATURE ANALYSIS
# =========================================================

def analyze_validation_feature(
    val_df,
    dev_result,
):

    position = dev_result[
        "position"
    ]

    feature = dev_result[
        "feature"
    ]

    dev_corr = dev_result[
        "development_correlation"
    ]

    working = val_df[
        [
            feature,
            "target_fanduel_points",
        ]
    ].copy()

    working[
        feature
    ] = pd.to_numeric(
        working[
            feature
        ],
        errors="coerce",
    )

    working[
        "target_fanduel_points"
    ] = pd.to_numeric(
        working[
            "target_fanduel_points"
        ],
        errors="coerce",
    )

    working = working.dropna()

    val_corr = safe_corr(
        working[
            feature
        ],
        working[
            "target_fanduel_points"
        ],
    )

    working[
        "bucket"
    ] = create_quantile_bucket(
        working[
            feature
        ]
    )

    # -----------------------------------------------------
    # We preserve DEVELOPMENT direction.
    #
    # This is critical.
    # We do NOT look at 2025 and then decide which side of
    # the feature is favorable.
    # -----------------------------------------------------

    good_bucket = favorable_bucket(
        dev_corr
    )

    bad_bucket = unfavorable_bucket(
        dev_corr
    )

    favorable = working[
        working[
            "bucket"
        ]
        ==
        good_bucket
    ]

    unfavorable = working[
        working[
            "bucket"
        ]
        ==
        bad_bucket
    ]

    validation_fd_mean = safe_mean(
        working[
            "target_fanduel_points"
        ]
    )

    favorable_fd = safe_mean(
        favorable[
            "target_fanduel_points"
        ]
    )

    unfavorable_fd = safe_mean(
        unfavorable[
            "target_fanduel_points"
        ]
    )

    threshold = 20.0

    overall_20 = safe_rate(
        (
            working[
                "target_fanduel_points"
            ]
            >=
            threshold
        ).astype(int)
    )

    favorable_20 = safe_rate(
        (
            favorable[
                "target_fanduel_points"
            ]
            >=
            threshold
        ).astype(int)
    )

    if overall_20 > 0:

        ceiling_lift_20 = (
            favorable_20
            /
            overall_20
        )

    else:

        ceiling_lift_20 = 0.0

    dev_direction = direction_label(
        dev_corr
    )

    val_direction = direction_label(
        val_corr
    )

    if (
        dev_direction
        ==
        "FLAT"
    ):

        direction_consistent = 0

    elif (
        dev_direction
        ==
        val_direction
    ):

        direction_consistent = 1

    else:

        direction_consistent = 0

    # -----------------------------------------------------
    # VALIDATION STATUS
    #
    # This does NOT alter development_class.
    # It only tells us whether the locked development
    # signal survived 2025.
    # -----------------------------------------------------

    if (
        dev_result[
            "development_class"
        ]
        ==
        "PASS"

        and

        direction_consistent
        ==
        1

        and

        abs(
            val_corr
        )
        >=
        0.05

        and

        favorable_fd
        >
        unfavorable_fd
    ):

        validation_status = (
            "CONFIRMED"
        )

    elif (
        dev_result[
            "development_class"
        ]
        in {
            "PASS",
            "WEAK",
        }

        and

        direction_consistent
        ==
        1

        and

        favorable_fd
        >=
        unfavorable_fd
    ):

        validation_status = (
            "PARTIAL"
        )

    else:

        validation_status = (
            "FAILED_VALIDATION"
        )

    return {
        "position":
            position,

        "feature":
            feature,

        "development_class":
            dev_result[
                "development_class"
            ],

        "development_score":
            dev_result[
                "development_score"
            ],

        "development_correlation":
            dev_corr,

        "development_direction":
            dev_direction,

        "validation_seasons":
            ",".join(
                str(
                    season
                )
                for season in (
                    VALIDATION_SEASONS
                )
            ),

        "validation_sample":
            len(
                working
            ),

        "validation_target_mean":
            validation_fd_mean,

        "validation_correlation":
            val_corr,

        "validation_direction":
            val_direction,

        "direction_consistent":
            direction_consistent,

        "locked_favorable_bucket":
            good_bucket,

        "validation_favorable_fd_mean":
            favorable_fd,

        "validation_unfavorable_fd_mean":
            unfavorable_fd,

        "validation_overall_20_hit_rate":
            overall_20,

        "validation_favorable_20_hit_rate":
            favorable_20,

        "validation_ceiling_lift_20":
            ceiling_lift_20,

        "validation_status":
            validation_status,
    }


# =========================================================
# RUN RESEARCH
# =========================================================

def run_feature_research(
    df,
):

    section(
        "RUNNING DEVELOPMENT FEATURE RESEARCH"
    )

    development = df[
        df[
            "season"
        ].isin(
            DEVELOPMENT_SEASONS
        )
    ].copy()

    validation = df[
        df[
            "season"
        ].isin(
            VALIDATION_SEASONS
        )
    ].copy()

    summary_rows = []
    bucket_rows = []
    ceiling_rows = []
    validation_rows = []

    for position in POSITIONS:

        print()
        print(
            f"Analyzing {position}..."
        )

        dev_position = development[
            development[
                "position"
            ]
            ==
            position
        ].copy()

        val_position = validation[
            validation[
                "position"
            ]
            ==
            position
        ].copy()

        for feature in POSITION_FEATURES[
            position
        ]:

            dev_result = (
                analyze_development_feature(
                    dev_position,
                    position,
                    feature,
                )
            )

            summary_rows.append(
                dev_result
            )

            bucket_rows.extend(
                analyze_buckets(
                    dev_position,
                    position,
                    feature,
                    "DEVELOPMENT",
                )
            )

            ceiling_rows.extend(
                analyze_ceilings(
                    dev_position,
                    position,
                    feature,
                    "DEVELOPMENT",
                    dev_result[
                        "development_correlation"
                    ],
                )
            )

            # ---------------------------------------------
            # 2025 validation comes only AFTER the
            # development result has been locked.
            # ---------------------------------------------

            validation_result = (
                analyze_validation_feature(
                    val_position,
                    dev_result,
                )
            )

            validation_rows.append(
                validation_result
            )

            bucket_rows.extend(
                analyze_buckets(
                    val_position,
                    position,
                    feature,
                    "VALIDATION",
                )
            )

            ceiling_rows.extend(
                analyze_ceilings(
                    val_position,
                    position,
                    feature,
                    "VALIDATION",
                    dev_result[
                        "development_correlation"
                    ],
                )
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    bucket_df = pd.DataFrame(
        bucket_rows
    )

    ceiling_df = pd.DataFrame(
        ceiling_rows
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    return (
        summary_df,
        bucket_df,
        ceiling_df,
        validation_df,
    )


# =========================================================
# RANKING
# =========================================================

def rank_development_features(
    summary_df,
):

    section(
        "RANKING DEVELOPMENT FEATURES"
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "position",
                "development_score",
                "development_correlation",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary_df[
        "development_rank"
    ] = (
        summary_df.groupby(
            "position"
        )[
            "development_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return summary_df


def rank_validation_features(
    validation_df,
):

    # -----------------------------------------------------
    # Validation rank is descriptive only.
    # It never feeds back into development selection.
    # -----------------------------------------------------

    validation_df = (
        validation_df.copy()
    )

    validation_df[
        "validation_abs_corr"
    ] = (
        validation_df[
            "validation_correlation"
        ]
        .abs()
    )

    validation_df[
        "validation_rank"
    ] = (
        validation_df.groupby(
            "position"
        )[
            "validation_abs_corr"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return validation_df


# =========================================================
# SELECT LOCKED DEVELOPMENT FEATURES
# =========================================================

def build_selected_features(
    summary_df,
    validation_df,
):

    section(
        "LOCKING DEVELOPMENT FEATURE SET"
    )

    selected = summary_df[
        summary_df[
            "development_class"
        ].isin(
            [
                "PASS",
                "WEAK",
            ]
        )
    ].copy()

    validation_small = validation_df[
        [
            "position",
            "feature",

            "validation_correlation",
            "validation_direction",
            "direction_consistent",

            "validation_favorable_fd_mean",
            "validation_unfavorable_fd_mean",

            "validation_ceiling_lift_20",

            "validation_rank",
            "validation_status",
        ]
    ].copy()

    selected = selected.merge(
        validation_small,
        how="left",
        on=[
            "position",
            "feature",
        ],
        validate="one_to_one",
    )

    selected[
        "rank_change"
    ] = (
        selected[
            "validation_rank"
        ]
        -
        selected[
            "development_rank"
        ]
    )

    selected = (
        selected.sort_values(
            [
                "position",
                "development_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Locked PASS/WEAK features: "
        f"{len(selected)}"
    )

    print()

    for position in POSITIONS:

        position_df = selected[
            selected[
                "position"
            ]
            ==
            position
        ]

        passes = int(
            (
                position_df[
                    "development_class"
                ]
                ==
                "PASS"
            ).sum()
        )

        weak = int(
            (
                position_df[
                    "development_class"
                ]
                ==
                "WEAK"
            ).sum()
        )

        confirmed = int(
            (
                position_df[
                    "validation_status"
                ]
                ==
                "CONFIRMED"
            ).sum()
        )

        print(
            f"{position}: "
            f"PASS {passes} | "
            f"WEAK {weak} | "
            f"2025 confirmed {confirmed}"
        )

    return selected


# =========================================================
# PRINT TOP RESULTS
# =========================================================

def print_top_results(
    summary_df,
    validation_df,
):

    section(
        "TOP DEVELOPMENT SIGNALS"
    )

    merged = summary_df.merge(
        validation_df[
            [
                "position",
                "feature",

                "validation_correlation",
                "direction_consistent",
                "validation_status",
            ]
        ],
        how="left",
        on=[
            "position",
            "feature",
        ],
        validate="one_to_one",
    )

    for position in POSITIONS:

        print()
        print(
            f"{position}"
        )

        print(
            "-" * 78
        )

        position_df = (
            merged[
                merged[
                    "position"
                ]
                ==
                position
            ]
            .sort_values(
                "development_rank"
            )
            .head(
                15
            )
        )

        print(
            position_df[
                [
                    "development_rank",
                    "feature",

                    "development_class",
                    "development_score",

                    "development_correlation",
                    "development_direction",

                    "favorable_fd_mean",
                    "unfavorable_fd_mean",

                    "ceiling_lift_20",

                    "validation_correlation",
                    "direction_consistent",

                    "validation_status",
                ]
            ].to_string(
                index=False
            )
        )


# =========================================================
# POSITION VALIDATION SUMMARY
# =========================================================

def print_validation_summary(
    summary_df,
    validation_df,
):

    section(
        "2025 HOLDOUT VALIDATION SUMMARY"
    )

    combined = summary_df[
        [
            "position",
            "feature",
            "development_class",
        ]
    ].merge(
        validation_df[
            [
                "position",
                "feature",
                "validation_status",
            ]
        ],
        on=[
            "position",
            "feature",
        ],
        validate="one_to_one",
    )

    selected = combined[
        combined[
            "development_class"
        ].isin(
            [
                "PASS",
                "WEAK",
            ]
        )
    ]

    rows = []

    for position in POSITIONS:

        position_df = selected[
            selected[
                "position"
            ]
            ==
            position
        ]

        rows.append(
            {
                "position":
                    position,

                "development_selected":
                    len(
                        position_df
                    ),

                "confirmed":
                    int(
                        (
                            position_df[
                                "validation_status"
                            ]
                            ==
                            "CONFIRMED"
                        ).sum()
                    ),

                "partial":
                    int(
                        (
                            position_df[
                                "validation_status"
                            ]
                            ==
                            "PARTIAL"
                        ).sum()
                    ),

                "failed_validation":
                    int(
                        (
                            position_df[
                                "validation_status"
                            ]
                            ==
                            "FAILED_VALIDATION"
                        ).sum()
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "confirmation_rate"
    ] = np.where(
        result[
            "development_selected"
        ]
        >
        0,

        result[
            "confirmed"
        ]
        /
        result[
            "development_selected"
        ],

        0.0,
    )

    print(
        result.to_string(
            index=False
        )
    )


# =========================================================
# EXPORT
# =========================================================

def export_research(
    summary_df,
    bucket_df,
    ceiling_df,
    selected_df,
    validation_df,
):

    section(
        "EXPORTING FEATURE RESEARCH"
    )

    summary_df.to_csv(
        RESEARCH_SUMMARY_CSV,
        index=False,
    )

    summary_df.to_parquet(
        RESEARCH_SUMMARY_PARQUET,
        index=False,
    )

    bucket_df.to_csv(
        BUCKET_RESEARCH_CSV,
        index=False,
    )

    bucket_df.to_parquet(
        BUCKET_RESEARCH_PARQUET,
        index=False,
    )

    ceiling_df.to_csv(
        CEILING_RESEARCH_CSV,
        index=False,
    )

    ceiling_df.to_parquet(
        CEILING_RESEARCH_PARQUET,
        index=False,
    )

    selected_df.to_csv(
        SELECTED_FEATURES_CSV,
        index=False,
    )

    selected_df.to_parquet(
        SELECTED_FEATURES_PARQUET,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_REPORT_CSV,
        index=False,
    )

    validation_df.to_parquet(
        VALIDATION_REPORT_PARQUET,
        index=False,
    )

    print(
        f"Summary rows: "
        f"{len(summary_df)}"
    )

    print(
        f"Bucket rows: "
        f"{len(bucket_df)}"
    )

    print(
        f"Ceiling rows: "
        f"{len(ceiling_df)}"
    )

    print(
        f"Selected rows: "
        f"{len(selected_df)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df)}"
    )

    print()

    print(
        f"Summary CSV: "
        f"{RESEARCH_SUMMARY_CSV}"
    )

    print(
        f"Selected CSV: "
        f"{SELECTED_FEATURES_CSV}"
    )

    print(
        f"Validation CSV: "
        f"{VALIDATION_REPORT_CSV}"
    )

    print(
        f"Buckets CSV: "
        f"{BUCKET_RESEARCH_CSV}"
    )

    print(
        f"Ceilings CSV: "
        f"{CEILING_RESEARCH_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_feature_research_engine():

    section(
        "NFL DFS FEATURE RESEARCH ENGINE"
    )

    # -----------------------------------------------------
    # LOAD LOCKED HISTORICAL FEATURE MATRIX
    # -----------------------------------------------------

    df = load_feature_matrix()

    validate_feature_configuration(
        df
    )

    print_sample_summary(
        df
    )

    # -----------------------------------------------------
    # DEVELOPMENT FIRST
    #
    # Feature classification is determined entirely from
    # 2023 + 2024.
    # -----------------------------------------------------

    (
        summary_df,
        bucket_df,
        ceiling_df,
        validation_df,
    ) = run_feature_research(
        df
    )

    summary_df = (
        rank_development_features(
            summary_df
        )
    )

    validation_df = (
        rank_validation_features(
            validation_df
        )
    )

    # -----------------------------------------------------
    # LOCK DEVELOPMENT SELECTION
    # -----------------------------------------------------

    selected_df = (
        build_selected_features(
            summary_df,
            validation_df,
        )
    )

    # -----------------------------------------------------
    # REPORT HOLDOUT PERFORMANCE
    # -----------------------------------------------------

    print_top_results(
        summary_df,
        validation_df,
    )

    print_validation_summary(
        summary_df,
        validation_df,
    )

    # -----------------------------------------------------
    # EXPORT EVERYTHING
    # -----------------------------------------------------

    export_research(
        summary_df,
        bucket_df,
        ceiling_df,
        selected_df,
        validation_df,
    )

    section(
        "FEATURE RESEARCH BUILD SUCCESSFUL"
    )

    print()
    print(
        "Development feature selection:"
    )

    print(
        f"{DEVELOPMENT_SEASONS}"
    )

    print()

    print(
        "Untouched holdout validation:"
    )

    print(
        f"{VALIDATION_SEASONS}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "2025 validation did not alter "
        "development PASS / WEAK / FAIL "
        "classification."
    )


if __name__ == "__main__":

    run_feature_research_engine()
