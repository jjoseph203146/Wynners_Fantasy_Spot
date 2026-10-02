#!/usr/bin/env python3

"""
WFS Team Formula Learning V1

Purpose
-------
Create strictly as-of multi-game team formula evidence from the frozen
Postgame Formula Evidence V1 artifact.

For each game_id + team row, all learned features are constructed only
from that team's PRIOR games.

The current game's formula/result fields are retained only as validation
targets and are NEVER used to construct the prior features.

Input
-----
processed/postgame_formula_evidence_v1.parquet

Output
------
processed/team_formula_learning_v1.parquet

Canonical grain
---------------
Exactly one row per game_id + team.

Safety
------
Shadow/research only.
No production, solver, forecast, player projection, or coaching-prior
influence.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "processed" / "postgame_formula_evidence_v1.parquet"
OUTPUT_FILE = ROOT / "processed" / "team_formula_learning_v1.parquet"

VERSION = "WFS_TEAM_FORMULA_LEARNING_V1"

PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
FORECAST_MUTATION = False
PLAYER_PROJECTION_MUTATION = False
PERSISTENT_COACH_PRIOR_MUTATION = False


def require_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def assert_unique(df, keys, label):
    dupes = df.duplicated(keys, keep=False)

    if dupes.any():
        sample = (
            df.loc[dupes, keys]
            .sort_values(keys)
            .head(20)
            .to_dict("records")
        )
        raise RuntimeError(
            f"{label} duplicate keys {keys}: {sample}"
        )


def bool_from_label(series, label):
    """
    Return 1/0 where the source label is available.
    Missing source remains NaN.
    """
    return np.where(
        series.isna(),
        np.nan,
        (series == label).astype(float),
    )


def prior_rolling_mean(series, window):
    """
    Strictly prior-only rolling mean.

    shift(1) prevents the current game's value from entering
    its own learned feature.
    """
    return (
        series.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )


def prior_rolling_count(series, window):
    """
    Number of non-null prior observations in the rolling window.
    """
    return (
        series.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .count()
    )


def add_prior_rates(group):
    g = group.copy()

    binary_sources = {
        # Approach
        "more_pass_game":
            ("approach_pass_rate",
             "MORE_PASS_ORIENTED_THAN_BASELINE"),

        "more_run_game":
            ("approach_pass_rate",
             "MORE_RUN_ORIENTED_THAN_BASELINE"),

        # Efficiency
        "positive_pass_epa_game":
            ("pass_efficiency",
             "POSITIVE_PASS_EPA"),

        "positive_rush_epa_game":
            ("rush_efficiency",
             "POSITIVE_RUSH_EPA"),

        # Yardage production
        "above_pass_yards_game":
            ("pass_yardage_production",
             "ABOVE_BASELINE_PASS_YARDS"),

        "above_rush_yards_game":
            ("rush_yardage_production",
             "ABOVE_BASELINE_RUSH_YARDS"),

        # TD production
        "above_pass_tds_game":
            ("pass_td_production",
             "ABOVE_BASELINE_PASS_TDS"),

        "above_rush_tds_game":
            ("rush_td_production",
             "ABOVE_BASELINE_RUSH_TDS"),

        # Scoring
        "above_scoring_game":
            ("scoring_production",
             "ABOVE_BASELINE_SCORING"),
    }

    for new_col, (source_col, label) in binary_sources.items():
        g[new_col] = bool_from_label(
            g[source_col],
            label,
        )

    # ----------------------------------------------------------
    # Combination evidence
    #
    # These describe whether an approach coincided with a
    # successful offensive characteristic. They remain
    # descriptive and non-causal.
    # ----------------------------------------------------------

    g["run_plus_positive_rush_epa"] = (
        g["more_run_game"]
        * g["positive_rush_epa_game"]
    )

    g["run_plus_above_rush_yards"] = (
        g["more_run_game"]
        * g["above_rush_yards_game"]
    )

    g["run_plus_above_rush_tds"] = (
        g["more_run_game"]
        * g["above_rush_tds_game"]
    )

    g["run_plus_above_scoring"] = (
        g["more_run_game"]
        * g["above_scoring_game"]
    )

    g["pass_plus_positive_pass_epa"] = (
        g["more_pass_game"]
        * g["positive_pass_epa_game"]
    )

    g["pass_plus_above_pass_yards"] = (
        g["more_pass_game"]
        * g["above_pass_yards_game"]
    )

    g["pass_plus_above_pass_tds"] = (
        g["more_pass_game"]
        * g["above_pass_tds_game"]
    )

    g["pass_plus_above_scoring"] = (
        g["more_pass_game"]
        * g["above_scoring_game"]
    )

    rate_sources = [
        "more_pass_game",
        "more_run_game",

        "positive_pass_epa_game",
        "positive_rush_epa_game",

        "above_pass_yards_game",
        "above_rush_yards_game",

        "above_pass_tds_game",
        "above_rush_tds_game",

        "above_scoring_game",

        "run_plus_positive_rush_epa",
        "run_plus_above_rush_yards",
        "run_plus_above_rush_tds",
        "run_plus_above_scoring",

        "pass_plus_positive_pass_epa",
        "pass_plus_above_pass_yards",
        "pass_plus_above_pass_tds",
        "pass_plus_above_scoring",
    ]

    for source in rate_sources:
        for window in (3, 5):
            g[f"prior_{window}_{source}_rate"] = (
                prior_rolling_mean(
                    g[source],
                    window,
                )
            )

    # ----------------------------------------------------------
    # Prior continuous performance
    # ----------------------------------------------------------

    continuous_sources = [
        "actual_passing_epa",
        "actual_rushing_epa",

        "pass_rate_delta_vs_baseline",

        "passing_yards_delta_vs_baseline",
        "rushing_yards_delta_vs_baseline",

        "passing_tds_delta_vs_baseline",
        "rushing_tds_delta_vs_baseline",

        "points_delta_vs_baseline",
    ]

    for source in continuous_sources:
        for window in (3, 5):
            g[f"prior_{window}_{source}_avg"] = (
                prior_rolling_mean(
                    pd.to_numeric(
                        g[source],
                        errors="coerce",
                    ),
                    window,
                )
            )

    # ----------------------------------------------------------
    # Prior result evidence
    #
    # Result is secondary context, not the definition of
    # offensive formula success.
    # ----------------------------------------------------------

    g["win_game"] = np.where(
        g["game_result"].isna(),
        np.nan,
        (g["game_result"] == "WIN").astype(float),
    )

    for window in (3, 5):
        g[f"prior_{window}_win_rate"] = (
            prior_rolling_mean(
                g["win_game"],
                window,
            )
        )

        g[f"prior_{window}_result_games"] = (
            prior_rolling_count(
                g["win_game"],
                window,
            )
        )

    # Formula history count is based on core formula evidence.
    core = pd.to_numeric(
        g["has_formula_core_evidence"],
        errors="coerce",
    )

    for window in (3, 5):
        g[f"prior_{window}_formula_games"] = (
            prior_rolling_count(
                core.where(core == 1),
                window,
            )
        )

    return g


def build():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    src = pd.read_parquet(INPUT_FILE)

    required = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",

        "approach_pass_rate",
        "pass_efficiency",
        "rush_efficiency",

        "pass_yardage_production",
        "rush_yardage_production",

        "pass_td_production",
        "rush_td_production",

        "scoring_production",
        "game_result",

        "actual_passing_epa",
        "actual_rushing_epa",

        "pass_rate_delta_vs_baseline",

        "passing_yards_delta_vs_baseline",
        "rushing_yards_delta_vs_baseline",

        "passing_tds_delta_vs_baseline",
        "rushing_tds_delta_vs_baseline",

        "points_delta_vs_baseline",

        "has_formula_core_evidence",
        "has_result_evidence",
    ]

    require_columns(
        src,
        required,
        "postgame formula evidence",
    )

    assert_unique(
        src,
        ["game_id", "team"],
        "postgame formula evidence",
    )

    # ----------------------------------------------------------
    # Deterministic chronological ordering.
    #
    # season/week is sufficient for team-level weekly NFL
    # progression in this artifact.
    # ----------------------------------------------------------

    src = src.sort_values(
        [
            "team",
            "season",
            "week",
            "game_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    learned_parts = []

    for _, group in src.groupby(
        "team",
        sort=False,
        dropna=False,
    ):
        learned_parts.append(
            add_prior_rates(group)
        )

    learned = pd.concat(
        learned_parts,
        ignore_index=True,
    )

    # ----------------------------------------------------------
    # Output identity + learned features + validation targets.
    # ----------------------------------------------------------

    out = pd.DataFrame()

    out["version"] = VERSION

    for c in [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    ]:
        out[c] = learned[c]

    # Current-game fields retained ONLY for validation.
    validation_fields = [
        "approach_pass_rate",
        "pass_efficiency",
        "rush_efficiency",
        "pass_yardage_production",
        "rush_yardage_production",
        "pass_td_production",
        "rush_td_production",
        "scoring_production",
        "game_result",

        "actual_passing_epa",
        "actual_rushing_epa",
        "pass_rate_delta_vs_baseline",

        "passing_yards_delta_vs_baseline",
        "rushing_yards_delta_vs_baseline",

        "passing_tds_delta_vs_baseline",
        "rushing_tds_delta_vs_baseline",

        "points_delta_vs_baseline",

        "has_formula_core_evidence",
        "has_result_evidence",
    ]

    for c in validation_fields:
        out[f"current_{c}"] = learned[c]

    # ----------------------------------------------------------
    # Copy ONLY prior/as-of learning features.
    # ----------------------------------------------------------

    prior_cols = [
        c
        for c in learned.columns
        if c.startswith("prior_")
    ]

    for c in prior_cols:
        out[c] = learned[c]

    # ----------------------------------------------------------
    # Evidence maturity
    # ----------------------------------------------------------

    out["formula_history_games"] = (
        learned.groupby(
            "team",
            sort=False,
        )
        .cumcount()
    )

    out["formula_history_tier"] = np.select(
        [
            out["formula_history_games"] == 0,
            out["formula_history_games"] == 1,
            out["formula_history_games"] == 2,
            out["formula_history_games"] >= 3,
        ],
        [
            "NO_PRIOR",
            "ONE_PRIOR",
            "TWO_PRIOR",
            "THREE_PLUS_PRIOR",
        ],
        default="UNKNOWN",
    )

    # Explicit provenance contract.
    out["current_game_used_in_prior_features"] = False
    out["future_game_used_in_prior_features"] = False
    out["causal_claim_allowed"] = False

    # Safety contract.
    out["production_influence"] = PRODUCTION_INFLUENCE
    out["solver_influence"] = SOLVER_INFLUENCE
    out["forecast_mutation"] = FORECAST_MUTATION
    out["player_projection_mutation"] = (
        PLAYER_PROJECTION_MUTATION
    )
    out["persistent_coach_prior_mutation"] = (
        PERSISTENT_COACH_PRIOR_MUTATION
    )

    assert_unique(
        out,
        ["game_id", "team"],
        "team formula learning",
    )

    return out


def print_validation(df):
    print("=== TEAM FORMULA LEARNING V1 ===")
    print("OUTPUT:", OUTPUT_FILE)

    print("\n=== KEY INTEGRITY ===")
    print("rows:", len(df))

    print(
        "unique_game_team:",
        df[
            ["game_id", "team"]
        ].drop_duplicates().shape[0],
    )

    print(
        "duplicate_game_team:",
        int(
            df.duplicated(
                ["game_id", "team"]
            ).sum()
        ),
    )

    print("\n=== HISTORY TIERS ===")
    print(
        df["formula_history_tier"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== AS-OF SAFETY ===")

    for c in [
        "current_game_used_in_prior_features",
        "future_game_used_in_prior_features",
        "causal_claim_allowed",
        "production_influence",
        "solver_influence",
        "forecast_mutation",
        "player_projection_mutation",
        "persistent_coach_prior_mutation",
    ]:
        print(
            c,
            sorted(
                df[c]
                .dropna()
                .unique()
                .tolist()
            ),
        )

    print("\n=== FIRST GAME PER TEAM PRIOR CHECK ===")

    first = (
        df.sort_values(
            [
                "team",
                "season",
                "week",
                "game_id",
            ],
            kind="stable",
        )
        .groupby(
            "team",
            as_index=False,
        )
        .head(1)
    )

    prior_feature_cols = [
        c
        for c in df.columns
        if c.startswith("prior_")
        and not c.endswith("_games")
    ]

    non_null_first = (
        first[prior_feature_cols]
        .notna()
        .sum()
        .sum()
    )

    print(
        "non_null_prior_values_on_first_team_games:",
        int(non_null_first),
    )

    print("\n=== 2026 SAMPLE ===")

    sample_cols = [
        "game_id",
        "team",
        "opponent_team",
        "formula_history_games",
        "formula_history_tier",

        "prior_3_more_pass_game_rate",
        "prior_3_more_run_game_rate",

        "prior_3_positive_pass_epa_game_rate",
        "prior_3_positive_rush_epa_game_rate",

        "prior_3_run_plus_positive_rush_epa_rate",
        "prior_3_run_plus_above_rush_yards_rate",
        "prior_3_run_plus_above_rush_tds_rate",
        "prior_3_run_plus_above_scoring_rate",

        "prior_3_pass_plus_positive_pass_epa_rate",
        "prior_3_pass_plus_above_pass_yards_rate",
        "prior_3_pass_plus_above_pass_tds_rate",
        "prior_3_pass_plus_above_scoring_rate",

        "prior_3_actual_passing_epa_avg",
        "prior_3_actual_rushing_epa_avg",

        "current_approach_pass_rate",
        "current_pass_efficiency",
        "current_rush_efficiency",
        "current_scoring_production",
        "current_game_result",
    ]

    sample_cols = [
        c
        for c in sample_cols
        if c in df.columns
    ]

    sample = (
        df[df["season"] == 2026][sample_cols]
        .sort_values(
            ["game_id", "team"],
            kind="stable",
        )
        .head(16)
    )

    if sample.empty:
        print("NO 2026 ROWS")
    else:
        print(
            sample.to_string(
                index=False
            )
        )

    print("\n=== 2026 PRIOR COVERAGE ===")

    x = df[df["season"] == 2026]

    for c in [
        "prior_3_more_run_game_rate",
        "prior_3_more_pass_game_rate",
        "prior_3_positive_rush_epa_game_rate",
        "prior_3_positive_pass_epa_game_rate",
        "prior_3_run_plus_positive_rush_epa_rate",
        "prior_3_pass_plus_positive_pass_epa_rate",
    ]:
        if c in x.columns:
            print(
                c,
                int(x[c].notna().sum()),
                "/",
                len(x),
            )


def main():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = build()

    print_validation(df)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== WRITE COMPLETE ===")
    print(OUTPUT_FILE)

    print("\nSHADOW CONTRACT:")
    print("current_game_used_in_prior_features=False")
    print("future_game_used_in_prior_features=False")
    print("causal_claim_allowed=False")
    print("production_influence=False")
    print("solver_influence=False")
    print("forecast_mutation=False")
    print("player_projection_mutation=False")
    print("persistent_coach_prior_mutation=False")


if __name__ == "__main__":
    main()
