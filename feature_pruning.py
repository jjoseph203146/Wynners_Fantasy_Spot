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

POSITIONS = [
    "QB",
    "RB",
    "WR",
    "TE",
]

# ---------------------------------------------------------
# Only development PASS signals are eligible for the
# first projection model.
#
# WEAK signals remain available in research outputs but
# do not enter the first modeling feature set.
# ---------------------------------------------------------

ALLOWED_DEVELOPMENT_CLASSES = {
    "PASS",
}

# ---------------------------------------------------------
# If two features have absolute correlation at or above
# this threshold in 2023-24, the lower-ranked feature is
# removed.
# ---------------------------------------------------------

CORRELATION_THRESHOLD = 0.85

# ---------------------------------------------------------
# Maximum number of surviving features per position.
#
# The correlation filter normally produces fewer than
# this, but the hard cap prevents uncontrolled expansion.
# ---------------------------------------------------------

MAX_FEATURES_PER_POSITION = 12


# =========================================================
# INPUT PATH
# =========================================================

RESEARCH_SUMMARY_CSV = (
    CSV_DIR / "nfl_feature_research_summary.csv"
)


# =========================================================
# OUTPUT PATHS
# =========================================================

CORE_FEATURES_CSV = (
    CSV_DIR / "nfl_core_projection_features.csv"
)

CORE_FEATURES_PARQUET = (
    PARQUET_DIR / "nfl_core_projection_features.parquet"
)

PRUNING_AUDIT_CSV = (
    CSV_DIR / "audit_feature_pruning.csv"
)

PRUNING_AUDIT_PARQUET = (
    PARQUET_DIR / "audit_feature_pruning.parquet"
)

FEATURE_CORRELATION_CSV = (
    CSV_DIR / "nfl_feature_development_correlations.csv"
)

FEATURE_CORRELATION_PARQUET = (
    PARQUET_DIR / "nfl_feature_development_correlations.parquet"
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


def feature_family(
    feature,
):

    # -----------------------------------------------------
    # These families are descriptive only.
    #
    # They do NOT determine selection.
    # Selection is still development rank + correlation.
    # -----------------------------------------------------

    if feature.startswith(
        "fd_"
    ) or feature.startswith(
        "fanduel_"
    ):

        return "RECENT_FANDUEL"

    if (
        "opportunit" in feature
        or
        "touches" in feature
    ):

        return "OPPORTUNITY"

    if (
        "target" in feature
        or
        "wopr" in feature
        or
        "air_yards" in feature
        or
        "receptions" in feature
    ):

        return "RECEIVING_ROLE"

    if (
        "carr" in feature
        or
        "rush" in feature
    ):

        return "RUSHING_ROLE"

    if (
        "snap" in feature
    ):

        return "SNAP_ROLE"

    if (
        feature.startswith(
            "team_"
        )
    ):

        return "TEAM_ENVIRONMENT"

    if (
        feature.startswith(
            "opponent_"
        )
    ):

        return "OPPONENT_ENVIRONMENT"

    if (
        feature.startswith(
            "dvp_"
        )
    ):

        return "DVP"

    if (
        "role_" in feature
        or
        "usage_" in feature
    ):

        return "ROLE_SIGNAL"

    if (
        "yards" in feature
    ):

        return "YARDAGE"

    return "OTHER"


# =========================================================
# LOAD FEATURE MATRIX
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
                    2024
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
        f"Development rows loaded: "
        f"{len(df)}"
    )

    print()

    for position in POSITIONS:

        rows = len(
            df[
                df[
                    "position"
                ]
                ==
                position
            ]
        )

        print(
            f"{position}: {rows}"
        )

    return df


# =========================================================
# LOAD DEVELOPMENT RESEARCH
# =========================================================

def load_research():

    section(
        "LOADING DEVELOPMENT FEATURE RESEARCH"
    )

    if not (
        RESEARCH_SUMMARY_CSV.exists()
    ):

        raise RuntimeError(
            "Missing research file: "
            f"{RESEARCH_SUMMARY_CSV}"
        )

    df = pd.read_csv(
        RESEARCH_SUMMARY_CSV
    )

    required = [
        "position",
        "feature",
        "development_class",
        "development_score",
        "development_correlation",
        "development_rank",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Research summary is missing "
            "required columns: "
            +
            ", ".join(
                missing
            )
        )

    eligible = df[
        df[
            "development_class"
        ].isin(
            ALLOWED_DEVELOPMENT_CLASSES
        )
    ].copy()

    print(
        f"Total research rows: "
        f"{len(df)}"
    )

    print(
        f"Development PASS rows eligible: "
        f"{len(eligible)}"
    )

    print()

    for position in POSITIONS:

        count = len(
            eligible[
                eligible[
                    "position"
                ]
                ==
                position
            ]
        )

        print(
            f"{position}: {count}"
        )

    return eligible


# =========================================================
# CONFIGURATION AUDIT
# =========================================================

def validate_features_exist(
    matrix_df,
    research_df,
):

    section(
        "VALIDATING CANDIDATE FEATURES"
    )

    missing = []

    for _, row in (
        research_df.iterrows()
    ):

        feature = row[
            "feature"
        ]

        if feature not in (
            matrix_df.columns
        ):

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
            "Feature pruning stopped because "
            "candidate columns are missing."
        )

    print(
        "PASS: every development PASS feature "
        "exists in the feature matrix."
    )


# =========================================================
# CORRELATION MATRIX
# =========================================================

def build_position_correlations(
    position_df,
    candidate_features,
    position,
):

    section(
        f"{position} DEVELOPMENT CORRELATIONS"
    )

    data = (
        position_df[
            candidate_features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    correlation_matrix = (
        data.corr(
            method="pearson"
        )
    )

    rows = []

    for index_a, feature_a in enumerate(
        candidate_features
    ):

        for index_b in range(
            index_a + 1,
            len(
                candidate_features
            ),
        ):

            feature_b = (
                candidate_features[
                    index_b
                ]
            )

            value = (
                correlation_matrix.loc[
                    feature_a,
                    feature_b,
                ]
            )

            if pd.isna(value):

                value = 0.0

            rows.append(
                {
                    "position":
                        position,

                    "feature_a":
                        feature_a,

                    "feature_b":
                        feature_b,

                    "correlation":
                        float(
                            value
                        ),

                    "absolute_correlation":
                        abs(
                            float(
                                value
                            )
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = (
            result.sort_values(
                "absolute_correlation",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print(
            "Highest feature-feature correlations:"
        )

        print()

        print(
            result.head(
                15
            ).to_string(
                index=False
            )
        )

    return (
        correlation_matrix,
        result,
    )


# =========================================================
# GREEDY CORRELATION PRUNING
# =========================================================

def prune_position(
    matrix_df,
    research_df,
    position,
):

    section(
        f"PRUNING {position}"
    )

    position_research = (
        research_df[
            research_df[
                "position"
            ]
            ==
            position
        ]
        .sort_values(
            [
                "development_rank",
                "development_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    position_matrix = (
        matrix_df[
            matrix_df[
                "position"
            ]
            ==
            position
        ]
        .copy()
    )

    candidates = (
        position_research[
            "feature"
        ]
        .tolist()
    )

    if not candidates:

        raise RuntimeError(
            f"No PASS development "
            f"features for {position}."
        )

    correlation_matrix, correlation_rows = (
        build_position_correlations(
            position_matrix,
            candidates,
            position,
        )
    )

    selected = []

    audit_rows = []

    for _, research_row in (
        position_research.iterrows()
    ):

        feature = (
            research_row[
                "feature"
            ]
        )

        blocking_feature = None
        blocking_correlation = 0.0

        for selected_feature in selected:

            correlation = (
                correlation_matrix.loc[
                    feature,
                    selected_feature,
                ]
            )

            if pd.isna(
                correlation
            ):

                correlation = 0.0

            if (
                abs(
                    correlation
                )
                >=
                CORRELATION_THRESHOLD
            ):

                blocking_feature = (
                    selected_feature
                )

                blocking_correlation = (
                    float(
                        correlation
                    )
                )

                break

        # -------------------------------------------------
        # Keep feature if it is not redundant and we have
        # not hit the deterministic position cap.
        # -------------------------------------------------

        if blocking_feature is not None:

            decision = (
                "REMOVED_CORRELATED"
            )

            selected_flag = 0

        elif (
            len(
                selected
            )
            >=
            MAX_FEATURES_PER_POSITION
        ):

            decision = (
                "REMOVED_POSITION_CAP"
            )

            selected_flag = 0

        else:

            selected.append(
                feature
            )

            decision = (
                "SELECTED"
            )

            selected_flag = 1

        audit_rows.append(
            {
                "position":
                    position,

                "feature":
                    feature,

                "feature_family":
                    feature_family(
                        feature
                    ),

                "development_rank":
                    int(
                        research_row[
                            "development_rank"
                        ]
                    ),

                "development_score":
                    float(
                        research_row[
                            "development_score"
                        ]
                    ),

                "development_correlation":
                    float(
                        research_row[
                            "development_correlation"
                        ]
                    ),

                "development_class":
                    research_row[
                        "development_class"
                    ],

                "selected":
                    selected_flag,

                "decision":
                    decision,

                "blocking_feature":
                    blocking_feature,

                "blocking_correlation":
                    blocking_correlation,
            }
        )

    print()
    print(
        f"Candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Selected: "
        f"{len(selected)}"
    )

    print()

    print(
        "Selected features:"
    )

    print()

    for number, feature in enumerate(
        selected,
        start=1,
    ):

        print(
            f"{number:2d}. "
            f"{feature}"
        )

    return (
        selected,
        pd.DataFrame(
            audit_rows
        ),
        correlation_rows,
    )


# =========================================================
# BUILD CORE FEATURE TABLE
# =========================================================

def build_core_feature_table(
    research_df,
    selected_by_position,
):

    section(
        "BUILDING CORE PROJECTION FEATURE SET"
    )

    rows = []

    for position in POSITIONS:

        selected_features = (
            selected_by_position[
                position
            ]
        )

        position_research = (
            research_df[
                research_df[
                    "position"
                ]
                ==
                position
            ]
            .set_index(
                "feature"
            )
        )

        for core_rank, feature in enumerate(
            selected_features,
            start=1,
        ):

            source = (
                position_research.loc[
                    feature
                ]
            )

            rows.append(
                {
                    "position":
                        position,

                    "core_rank":
                        core_rank,

                    "feature":
                        feature,

                    "feature_family":
                        feature_family(
                            feature
                        ),

                    "development_rank":
                        int(
                            source[
                                "development_rank"
                            ]
                        ),

                    "development_score":
                        float(
                            source[
                                "development_score"
                            ]
                        ),

                    "development_correlation":
                        float(
                            source[
                                "development_correlation"
                            ]
                        ),

                    "development_class":
                        source[
                            "development_class"
                        ],

                    "selection_seasons":
                        "2023,2024",

                    "correlation_threshold":
                        CORRELATION_THRESHOLD,
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        f"Core features selected: "
        f"{len(result)}"
    )

    print()

    summary = (
        result.groupby(
            [
                "position",
                "feature_family",
            ]
        )
        .size()
        .reset_index(
            name="features"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return result


# =========================================================
# VALIDATION AUDIT
# =========================================================

def audit_core_features(
    core_df,
):

    section(
        "CORE FEATURE AUDIT"
    )

    duplicates = (
        core_df.groupby(
            [
                "position",
                "feature",
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

    invalid_positions = (
        ~core_df[
            "position"
        ].isin(
            POSITIONS
        )
    ).sum()

    over_cap = 0

    for position in POSITIONS:

        count = len(
            core_df[
                core_df[
                    "position"
                ]
                ==
                position
            ]
        )

        if (
            count
            >
            MAX_FEATURES_PER_POSITION
        ):

            over_cap += 1

    missing_positions = [
        position
        for position in POSITIONS
        if len(
            core_df[
                core_df[
                    "position"
                ]
                ==
                position
            ]
        )
        ==
        0
    ]

    print(
        f"Duplicate position/features: "
        f"{len(duplicates)}"
    )

    print(
        f"Invalid positions: "
        f"{invalid_positions}"
    )

    print(
        f"Positions over cap: "
        f"{over_cap}"
    )

    print(
        f"Positions with zero features: "
        f"{len(missing_positions)}"
    )

    problems = (
        len(
            duplicates
        )
        +
        int(
            invalid_positions
        )
        +
        over_cap
        +
        len(
            missing_positions
        )
    )

    if problems:

        raise RuntimeError(
            "Core feature audit failed."
        )

    print()
    print(
        "PASS: core projection feature "
        "set passed structural audits."
    )


# =========================================================
# REDUNDANCY AUDIT
# =========================================================

def audit_selected_correlations(
    matrix_df,
    core_df,
):

    section(
        "SELECTED FEATURE REDUNDANCY AUDIT"
    )

    violations = []

    for position in POSITIONS:

        features = (
            core_df[
                core_df[
                    "position"
                ]
                ==
                position
            ][
                "feature"
            ]
            .tolist()
        )

        data = (
            matrix_df[
                matrix_df[
                    "position"
                ]
                ==
                position
            ][
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        correlations = (
            data.corr()
        )

        for i in range(
            len(
                features
            )
        ):

            for j in range(
                i + 1,
                len(
                    features
                ),
            ):

                feature_a = (
                    features[i]
                )

                feature_b = (
                    features[j]
                )

                value = (
                    correlations.loc[
                        feature_a,
                        feature_b,
                    ]
                )

                if pd.isna(
                    value
                ):

                    continue

                if (
                    abs(
                        value
                    )
                    >=
                    CORRELATION_THRESHOLD
                ):

                    violations.append(
                        {
                            "position":
                                position,

                            "feature_a":
                                feature_a,

                            "feature_b":
                                feature_b,

                            "correlation":
                                float(
                                    value
                                ),
                        }
                    )

    print(
        f"Remaining correlation violations: "
        f"{len(violations)}"
    )

    if violations:

        violation_df = pd.DataFrame(
            violations
        )

        print()
        print(
            violation_df.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Selected features still contain "
            "correlation violations."
        )

    print()
    print(
        "PASS: no selected feature pair exceeds "
        f"|r| >= {CORRELATION_THRESHOLD:.2f}."
    )


# =========================================================
# PRINT FINAL FEATURES
# =========================================================

def print_final_features(
    core_df,
):

    section(
        "FINAL CORE PROJECTION FEATURES"
    )

    for position in POSITIONS:

        print()
        print(
            position
        )

        print(
            "-" * 78
        )

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

        print(
            position_df[
                [
                    "core_rank",
                    "feature",
                    "feature_family",
                    "development_rank",
                    "development_score",
                    "development_correlation",
                ]
            ].to_string(
                index=False
            )
        )


# =========================================================
# EXPORT
# =========================================================

def export_results(
    core_df,
    audit_df,
    correlation_df,
):

    section(
        "EXPORTING FEATURE PRUNING RESULTS"
    )

    core_df.to_csv(
        CORE_FEATURES_CSV,
        index=False,
    )

    core_df.to_parquet(
        CORE_FEATURES_PARQUET,
        index=False,
    )

    audit_df.to_csv(
        PRUNING_AUDIT_CSV,
        index=False,
    )

    audit_df.to_parquet(
        PRUNING_AUDIT_PARQUET,
        index=False,
    )

    correlation_df.to_csv(
        FEATURE_CORRELATION_CSV,
        index=False,
    )

    correlation_df.to_parquet(
        FEATURE_CORRELATION_PARQUET,
        index=False,
    )

    print(
        f"Core feature rows: "
        f"{len(core_df)}"
    )

    print(
        f"Pruning audit rows: "
        f"{len(audit_df)}"
    )

    print(
        f"Correlation rows: "
        f"{len(correlation_df)}"
    )

    print()

    print(
        f"Core CSV: "
        f"{CORE_FEATURES_CSV}"
    )

    print(
        f"Audit CSV: "
        f"{PRUNING_AUDIT_CSV}"
    )

    print(
        f"Correlation CSV: "
        f"{FEATURE_CORRELATION_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_feature_pruning():

    section(
        "NFL DFS FEATURE PRUNING ENGINE"
    )

    # -----------------------------------------------------
    # Load 2023-24 development data only.
    # -----------------------------------------------------

    matrix_df = load_matrix()

    research_df = load_research()

    validate_features_exist(
        matrix_df,
        research_df,
    )

    selected_by_position = {}

    audit_frames = []

    correlation_frames = []

    # -----------------------------------------------------
    # Position-specific pruning.
    # -----------------------------------------------------

    for position in POSITIONS:

        (
            selected,
            audit_df,
            correlation_df,
        ) = prune_position(
            matrix_df,
            research_df,
            position,
        )

        selected_by_position[
            position
        ] = selected

        audit_frames.append(
            audit_df
        )

        correlation_frames.append(
            correlation_df
        )

    # -----------------------------------------------------
    # Combine audits.
    # -----------------------------------------------------

    pruning_audit = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    nonempty_correlations = [
        frame
        for frame in correlation_frames
        if not frame.empty
    ]

    if nonempty_correlations:

        correlation_output = pd.concat(
            nonempty_correlations,
            ignore_index=True,
        )

    else:

        correlation_output = pd.DataFrame(
            columns=[
                "position",
                "feature_a",
                "feature_b",
                "correlation",
                "absolute_correlation",
            ]
        )

    # -----------------------------------------------------
    # Build final locked feature table.
    # -----------------------------------------------------

    core_df = build_core_feature_table(
        research_df,
        selected_by_position,
    )

    audit_core_features(
        core_df
    )

    audit_selected_correlations(
        matrix_df,
        core_df,
    )

    print_final_features(
        core_df
    )

    export_results(
        core_df,
        pruning_audit,
        correlation_output,
    )

    section(
        "FEATURE PRUNING BUILD SUCCESSFUL"
    )

    print()
    print(
        "Selection basis:"
    )

    print(
        "2023-2024 development PASS signals only."
    )

    print()

    print(
        "2025 validation status was NOT used "
        "to select or remove features."
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Build position-specific projection "
        "benchmarks using this frozen core "
        "feature set."
    )


if __name__ == "__main__":

    run_feature_pruning()
