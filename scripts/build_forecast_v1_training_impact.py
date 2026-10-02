#!/usr/bin/env python3

"""
WFS NFL Forecast Center
Forecast V1 — Clean Training Dataset + Impact V1

PURPOSE
-------
Create a new historical team-game training artifact that:

1. Preserves the validated non-injury Forecast V1 features.
2. Removes the legacy injury/player-population feature block.
3. Adds validated Impact V1 features for BOTH:
       - the team
       - the opponent
4. Preserves all target columns.
5. Does not modify SQLite.
6. Does not overwrite the original training dataset.

INPUTS
------
processed/forecast_v1_team_game_training.csv
processed/forecast_v1_team_impact.csv

OUTPUTS
-------
processed/forecast_v1_team_game_training_impact_v1.csv
processed/forecast_v1_team_game_training_impact_v1_audit.json
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

TRAIN_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training.csv"
)

IMPACT_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_impact.csv"
)

OUTPUT_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_v1.csv"
)

AUDIT_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_v1_audit.json"
)


# ==========================================================
# LEGACY FEATURES TO QUARANTINE
# ==========================================================

LEGACY_COLUMNS = [
    "team_player_rows",
    "team_injury_flag_count",
    "team_out_count",
    "team_doubtful_count",
    "team_questionable_count",
    "team_qb_out_count",
    "team_qb_doubtful_count",
    "team_qb_questionable_count",
    "team_out_fd_avg3_sum",
    "team_out_snap_pct_avg3_sum",
    "team_out_opportunities_avg3_sum",
    "team_out_targets_avg3_sum",
    "team_out_carries_avg3_sum",
    "team_out_established_role_count",
    "team_doubtful_fd_avg3_sum",
    "team_doubtful_snap_pct_avg3_sum",
    "team_doubtful_opportunities_avg3_sum",
    "team_questionable_fd_avg3_sum",
    "team_questionable_snap_pct_avg3_sum",
    "team_questionable_opportunities_avg3_sum",

    "opp_player_rows",
    "opp_injury_flag_count",
    "opp_out_count",
    "opp_doubtful_count",
    "opp_questionable_count",
    "opp_qb_out_count",
    "opp_qb_doubtful_count",
    "opp_qb_questionable_count",
    "opp_out_fd_avg3_sum",
    "opp_out_snap_pct_avg3_sum",
    "opp_out_opportunities_avg3_sum",
    "opp_out_targets_avg3_sum",
    "opp_out_carries_avg3_sum",
    "opp_out_established_role_count",
    "opp_doubtful_fd_avg3_sum",
    "opp_doubtful_snap_pct_avg3_sum",
    "opp_doubtful_opportunities_avg3_sum",
    "opp_questionable_fd_avg3_sum",
    "opp_questionable_snap_pct_avg3_sum",
    "opp_questionable_opportunities_avg3_sum",
]


# ==========================================================
# APPROVED IMPACT V1 MODEL FEATURES
# ==========================================================

IMPACT_FEATURES = [
    "out_count",
    "doubtful_count",

    "out_offense_count",
    "out_defense_count",
    "doubtful_offense_count",
    "doubtful_defense_count",

    "out_qb_count",
    "doubtful_qb_count",

    "out_rb_count",
    "out_wr_count",
    "out_te_count",
    "out_ol_count",

    "out_offensive_starter_count",
    "out_defensive_starter_count",
    "doubtful_offensive_starter_count",
    "doubtful_defensive_starter_count",

    "out_offense_snap_load",
    "out_defense_snap_load",
    "doubtful_offense_snap_load",
    "doubtful_defense_snap_load",

    "out_skill_opportunities_avg_3",
    "out_skill_targets_avg_3",
    "out_skill_carries_avg_3",
    "out_skill_fd_avg_3",

    "doubtful_skill_opportunities_avg_3",

    "max_out_offense_pct_avg_3",
    "max_out_defense_pct_avg_3",

    "out_cold_start_count",
    "doubtful_cold_start_count",

    "out_missing_pfr_identity_count",
]


TARGET_COLUMNS = [
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
]


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require_columns(df, columns, label):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: "
            f"{missing}"
        )


def main():

    print("=" * 100)
    print(
        "WFS NFL FORECAST CENTER — "
        "V1 CLEAN TRAINING + IMPACT BUILDER"
    )
    print("=" * 100)
    print("CSV INPUT / CSV+JSON OUTPUT ONLY")
    print("NO SQLITE ACCESS")
    print("ORIGINAL TRAINING FILE IS NOT OVERWRITTEN")

    # ------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------

    print_section("1. LOAD INPUTS")

    if not TRAIN_FILE.exists():
        raise RuntimeError(
            f"Missing training file: {TRAIN_FILE}"
        )

    if not IMPACT_FILE.exists():
        raise RuntimeError(
            f"Missing impact file: {IMPACT_FILE}"
        )

    train = pd.read_csv(
        TRAIN_FILE,
        low_memory=False,
    )

    impact = pd.read_csv(
        IMPACT_FILE,
        low_memory=False,
    )

    print(
        f"Training: {len(train)} rows "
        f"x {len(train.columns)} cols"
    )

    print(
        f"Impact  : {len(impact)} rows "
        f"x {len(impact.columns)} cols"
    )

    # ------------------------------------------------------
    # 2. STRUCTURAL VALIDATION
    # ------------------------------------------------------

    print_section("2. STRUCTURAL VALIDATION")

    require_columns(
        train,
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "team",
            "opponent_team",
            *LEGACY_COLUMNS,
            *TARGET_COLUMNS,
        ],
        "training",
    )

    require_columns(
        impact,
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "team",
            "opponent_team",
            "impact_source_timestamp_proven",
            *IMPACT_FEATURES,
        ],
        "impact",
    )

    if len(train) != 1710:
        raise RuntimeError(
            f"Expected 1710 training rows; "
            f"found {len(train)}"
        )

    if len(impact) != 1710:
        raise RuntimeError(
            f"Expected 1710 impact rows; "
            f"found {len(impact)}"
        )

    if train.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Training contains duplicate "
            "(game_id, team) keys."
        )

    if impact.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Impact contains duplicate "
            "(game_id, team) keys."
        )

    print("PASS | row counts")
    print("PASS | unique training keys")
    print("PASS | unique impact keys")
    print("PASS | required columns")

    # ------------------------------------------------------
    # 3. VERIFY MATCHUP METADATA
    # ------------------------------------------------------

    print_section("3. MATCHUP METADATA VALIDATION")

    metadata = train[
        [
            "game_id",
            "team",
            "season",
            "week",
            "game_type",
            "opponent_team",
        ]
    ].merge(
        impact[
            [
                "game_id",
                "team",
                "season",
                "week",
                "game_type",
                "opponent_team",
            ]
        ],
        on=[
            "game_id",
            "team",
        ],
        how="left",
        suffixes=(
            "_train",
            "_impact",
        ),
        validate="one_to_one",
    )

    comparisons = [
        "season",
        "week",
        "game_type",
        "opponent_team",
    ]

    for col in comparisons:

        left = metadata[
            f"{col}_train"
        ]

        right = metadata[
            f"{col}_impact"
        ]

        mismatch = (
            left.astype(str)
            != right.astype(str)
        )

        count = int(
            mismatch.sum()
        )

        if count:
            raise RuntimeError(
                f"Metadata mismatch for {col}: "
                f"{count} rows"
            )

        print(
            f"PASS | {col}"
        )

    # ------------------------------------------------------
    # 4. REMOVE LEGACY INJURY BLOCK
    # ------------------------------------------------------

    print_section("4. QUARANTINE LEGACY INJURY FEATURES")

    clean = train.drop(
        columns=LEGACY_COLUMNS
    ).copy()

    print(
        f"Removed legacy columns: "
        f"{len(LEGACY_COLUMNS)}"
    )

    print(
        f"Columns remaining: "
        f"{len(clean.columns)}"
    )

    legacy_remaining = [
        c
        for c in LEGACY_COLUMNS
        if c in clean.columns
    ]

    if legacy_remaining:
        raise RuntimeError(
            "Legacy columns remain after drop: "
            f"{legacy_remaining}"
        )

    print(
        "PASS | legacy injury/player-population "
        "block removed"
    )

    # ------------------------------------------------------
    # 5. TEAM IMPACT SIDE
    # ------------------------------------------------------

    print_section("5. BUILD TEAM IMPACT SIDE")

    team_side = impact[
        [
            "game_id",
            "team",
            *IMPACT_FEATURES,
        ]
    ].copy()

    team_side = team_side.rename(
        columns={
            col: f"team_impact_{col}"
            for col in IMPACT_FEATURES
        }
    )

    clean = clean.merge(
        team_side,
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    team_impact_cols = [
        f"team_impact_{c}"
        for c in IMPACT_FEATURES
    ]

    missing_team = int(
        clean[
            team_impact_cols
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    if missing_team:
        raise RuntimeError(
            "Missing complete team impact rows: "
            f"{missing_team}"
        )

    print(
        "PASS | team impact attached"
    )

    # ------------------------------------------------------
    # 6. OPPONENT IMPACT SIDE
    # ------------------------------------------------------

    print_section("6. BUILD OPPONENT IMPACT SIDE")

    opp_side = impact[
        [
            "game_id",
            "team",
            "opponent_team",
            *IMPACT_FEATURES,
        ]
    ].copy()

    # In the impact row:
    #   team          = training opponent
    #   opponent_team = training team
    #
    # Rename accordingly before joining.
    opp_side = opp_side.rename(
        columns={
            "team":
                "opponent_team",

            "opponent_team":
                "_impact_opponent_check",

            **{
                col:
                    f"opp_impact_{col}"
                for col in IMPACT_FEATURES
            },
        }
    )

    clean = clean.merge(
        opp_side,
        on=[
            "game_id",
            "opponent_team",
        ],
        how="left",
        validate="one_to_one",
    )

    mismatch = (
        clean[
            "_impact_opponent_check"
        ].astype(str)
        !=
        clean["team"].astype(str)
    )

    mismatch_count = int(
        mismatch.sum()
    )

    if mismatch_count:
        raise RuntimeError(
            "Opponent reverse-match validation "
            f"failed for {mismatch_count} rows."
        )

    clean = clean.drop(
        columns=[
            "_impact_opponent_check",
        ]
    )

    opp_impact_cols = [
        f"opp_impact_{c}"
        for c in IMPACT_FEATURES
    ]

    missing_opp = int(
        clean[
            opp_impact_cols
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    if missing_opp:
        raise RuntimeError(
            "Missing complete opponent impact rows: "
            f"{missing_opp}"
        )

    print(
        "PASS | opponent impact attached"
    )

    # ------------------------------------------------------
    # 7. PROVENANCE AUDIT
    # ------------------------------------------------------

    print_section("7. IMPACT PROVENANCE AUDIT")

    provenance = (
        impact.groupby("season")[
            "impact_source_timestamp_proven"
        ]
        .agg(
            [
                "count",
                "min",
                "max",
            ]
        )
    )

    print(
        provenance.to_string()
    )

    print()
    print(
        "NOTE: impact_source_timestamp_proven "
        "is NOT included as a model feature."
    )

    # ------------------------------------------------------
    # 8. FINAL DATASET VALIDATION
    # ------------------------------------------------------

    print_section("8. FINAL DATASET VALIDATION")

    audits = {}

    audits["row_count_1710"] = (
        len(clean) == 1710
    )

    audits["unique_team_game_keys"] = (
        clean.duplicated(
            [
                "game_id",
                "team",
            ]
        ).sum()
        == 0
    )

    audits["legacy_columns_removed"] = (
        not any(
            c in clean.columns
            for c in LEGACY_COLUMNS
        )
    )

    audits["team_impact_columns_present"] = all(
        c in clean.columns
        for c in team_impact_cols
    )

    audits["opp_impact_columns_present"] = all(
        c in clean.columns
        for c in opp_impact_cols
    )

    audits["targets_present"] = all(
        c in clean.columns
        for c in TARGET_COLUMNS
    )

    audits["team_impact_complete"] = (
        clean[
            team_impact_cols
        ]
        .isna()
        .all(axis=1)
        .sum()
        == 0
    )

    audits["opp_impact_complete"] = (
        clean[
            opp_impact_cols
        ]
        .isna()
        .all(axis=1)
        .sum()
        == 0
    )

    expected_columns = (
        len(train.columns)
        - len(LEGACY_COLUMNS)
        + len(IMPACT_FEATURES)
        + len(IMPACT_FEATURES)
    )

    audits["expected_column_count"] = (
        len(clean.columns)
        == expected_columns
    )

    for name, passed in audits.items():

        print(
            f"{'PASS' if passed else 'FAIL'} "
            f"| {name}"
        )

    if not all(audits.values()):
        raise RuntimeError(
            "Final training dataset audit FAILED."
        )

    print()
    print(
        f"Final rows   : {len(clean)}"
    )

    print(
        f"Final columns: {len(clean.columns)}"
    )

    print(
        f"Expected cols: {expected_columns}"
    )

    # ------------------------------------------------------
    # 9. EXPORT
    # ------------------------------------------------------

    print_section("9. EXPORT")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit_payload = {
        "input_training_file":
            str(TRAIN_FILE),

        "input_impact_file":
            str(IMPACT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "input_training_rows":
            int(len(train)),

        "input_training_columns":
            int(len(train.columns)),

        "legacy_columns_removed":
            LEGACY_COLUMNS,

        "legacy_column_count":
            int(len(LEGACY_COLUMNS)),

        "impact_features_per_side":
            IMPACT_FEATURES,

        "impact_feature_count_per_side":
            int(len(IMPACT_FEATURES)),

        "impact_source_timestamp_proven":
            "AUDIT_ONLY_NOT_MODEL_FEATURE",

        "questionable_status":
            (
                "NOT INCLUDED IN IMPACT V1; "
                "legacy questionable features quarantined"
            ),

        "output_rows":
            int(len(clean)),

        "output_columns":
            int(len(clean.columns)),

        "audits":
            {
                key: bool(value)
                for key, value
                in audits.items()
            },
    }

    with AUDIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            audit_payload,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print(
        "Training dataset:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Audit:"
    )
    print(
        AUDIT_FILE
    )

    print()
    print("=" * 100)
    print(
        "FORECAST V1 CLEAN TRAINING + IMPACT: PASS"
    )
    print(
        "NO SQLITE TABLES MODIFIED."
    )
    print(
        "ORIGINAL TRAINING CSV NOT MODIFIED."
    )
    print(
        "NO MACHINE-LEARNING MODEL TRAINED."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
