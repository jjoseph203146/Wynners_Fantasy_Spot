#!/usr/bin/env python3

"""
WFS Formula Component Signal Validation V1

Purpose
-------
Test individual strictly pregame/as-of formula and opponent-vulnerability
components against logically corresponding current-game outcomes.

This validator intentionally decomposes Formula Matchup Alignment V1.
It does NOT fit weights, optimize thresholds, create a replacement
alignment score, or authorize production use.

Important contextual limitation
--------------------------------
Weather and injury context are NOT present in the frozen Formula Matchup
Alignment V1 artifact. Therefore this validator does not claim that a
component caused an outcome or that an apparent miss invalidates the
component.

Weather/injury evidence must be evaluated separately using authoritative
pregame/as-of sources before production conclusions are made.

Input
-----
processed/formula_matchup_alignment_v1.parquet

Outputs
-------
processed/validation/formula_component_signal_validation_v1.parquet
processed/validation/formula_component_signal_validation_by_season_v1.parquet
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "processed"
    / "formula_matchup_alignment_v1.parquet"
)

OUTPUT_DIR = ROOT / "processed" / "validation"

SUMMARY_FILE = (
    OUTPUT_DIR
    / "formula_component_signal_validation_v1.parquet"
)

SEASON_FILE = (
    OUTPUT_DIR
    / "formula_component_signal_validation_by_season_v1.parquet"
)

VERSION = "WFS_FORMULA_COMPONENT_SIGNAL_VALIDATION_V1"


TESTS = [
    # ----------------------------------------------------------
    # RUN: prior team formula components
    # ----------------------------------------------------------
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_positive_rush_epa_game_rate",
        "target": "current_actual_rushing_epa",
        "target_kind": "EPA",
    },
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_run_plus_positive_rush_epa_rate",
        "target": "current_actual_rushing_epa",
        "target_kind": "EPA",
    },
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_above_rush_yards_game_rate",
        "target": "current_rushing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_run_plus_above_rush_yards_rate",
        "target": "current_rushing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_above_rush_tds_game_rate",
        "target": "current_rushing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },
    {
        "phase": "RUN",
        "family": "FORMULA",
        "component": "prior_3_run_plus_above_rush_tds_rate",
        "target": "current_rushing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },

    # ----------------------------------------------------------
    # RUN: opponent vulnerability components
    # ----------------------------------------------------------
    {
        "phase": "RUN",
        "family": "MATCHUP",
        "component": "opponent_rush_yards_allowed_vs_league",
        "target": "current_rushing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "RUN",
        "family": "MATCHUP",
        "component": "opponent_rush_tds_allowed_vs_league",
        "target": "current_rushing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },
    {
        "phase": "RUN",
        "family": "MATCHUP",
        "component": "rb_fd_allowed_vs_league",
        "target": "current_rushing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA_PROXY",
    },

    # ----------------------------------------------------------
    # PASS: prior team formula components
    # ----------------------------------------------------------
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_positive_pass_epa_game_rate",
        "target": "current_actual_passing_epa",
        "target_kind": "EPA",
    },
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_pass_plus_positive_pass_epa_rate",
        "target": "current_actual_passing_epa",
        "target_kind": "EPA",
    },
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_above_pass_yards_game_rate",
        "target": "current_passing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_pass_plus_above_pass_yards_rate",
        "target": "current_passing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_above_pass_tds_game_rate",
        "target": "current_passing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },
    {
        "phase": "PASS",
        "family": "FORMULA",
        "component": "prior_3_pass_plus_above_pass_tds_rate",
        "target": "current_passing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },

    # ----------------------------------------------------------
    # PASS: opponent vulnerability components
    # ----------------------------------------------------------
    {
        "phase": "PASS",
        "family": "MATCHUP",
        "component": "opponent_pass_yards_allowed_vs_league",
        "target": "current_passing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA",
    },
    {
        "phase": "PASS",
        "family": "MATCHUP",
        "component": "opponent_pass_tds_allowed_vs_league",
        "target": "current_passing_tds_delta_vs_baseline",
        "target_kind": "TD_DELTA",
    },
    {
        "phase": "PASS",
        "family": "MATCHUP",
        "component": "qb_fd_allowed_vs_league",
        "target": "current_actual_passing_epa",
        "target_kind": "EPA_PROXY",
    },
    {
        "phase": "PASS",
        "family": "MATCHUP",
        "component": "wr_fd_allowed_vs_league",
        "target": "current_passing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA_PROXY",
    },
    {
        "phase": "PASS",
        "family": "MATCHUP",
        "component": "te_fd_allowed_vs_league",
        "target": "current_passing_yards_delta_vs_baseline",
        "target_kind": "YARDS_DELTA_PROXY",
    },
]


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]

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


def pearson_safe(x, y):
    z = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(z) < 3:
        return np.nan

    if z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan

    return float(z["x"].corr(z["y"], method="pearson"))


def spearman_safe(x, y):
    z = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(z) < 3:
        return np.nan

    if z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan

    return float(z["x"].corr(z["y"], method="spearman"))


def quantile_groups(component):
    x = pd.to_numeric(component, errors="coerce")

    valid = x.dropna()

    out = pd.Series(
        pd.NA,
        index=x.index,
        dtype="object",
    )

    if valid.empty or valid.nunique() < 2:
        return out

    low_cut = valid.quantile(1.0 / 3.0)
    high_cut = valid.quantile(2.0 / 3.0)

    out.loc[x <= low_cut] = "LOW"
    out.loc[x >= high_cut] = "HIGH"

    middle = (
        x.notna()
        & (x > low_cut)
        & (x < high_cut)
    )

    out.loc[middle] = "MID"

    return out


def summarize_test(df, spec, season=None):
    component = spec["component"]
    target = spec["target"]

    x = pd.to_numeric(
        df[component],
        errors="coerce",
    )

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid = pd.DataFrame(
        {
            "component": x,
            "target": y,
        }
    ).dropna()

    base = {
        "version": VERSION,
        "season": season if season is not None else pd.NA,
        "phase": spec["phase"],
        "family": spec["family"],
        "component": component,
        "target": target,
        "target_kind": spec["target_kind"],
        "games": int(len(valid)),
        "component_mean": np.nan,
        "component_std": np.nan,
        "target_mean": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
        "low_games": 0,
        "low_target_mean": np.nan,
        "mid_games": 0,
        "mid_target_mean": np.nan,
        "high_games": 0,
        "high_target_mean": np.nan,
        "high_minus_low": np.nan,
        "high_positive_target_rate": np.nan,
        "low_positive_target_rate": np.nan,
        "positive_rate_spread": np.nan,
        "weather_controlled": False,
        "injury_controlled": False,
        "causal_claim_allowed": False,
        "weights_fitted": False,
        "thresholds_optimized": False,
        "production_authorized": False,
    }

    if valid.empty:
        return base

    base["component_mean"] = float(
        valid["component"].mean()
    )

    base["component_std"] = float(
        valid["component"].std()
    )

    base["target_mean"] = float(
        valid["target"].mean()
    )

    base["pearson"] = pearson_safe(
        valid["component"],
        valid["target"],
    )

    base["spearman"] = spearman_safe(
        valid["component"],
        valid["target"],
    )

    groups = quantile_groups(
        valid["component"]
    )

    valid = valid.copy()
    valid["bucket"] = groups

    for bucket in ("LOW", "MID", "HIGH"):
        g = valid[
            valid["bucket"] == bucket
        ]

        base[
            f"{bucket.lower()}_games"
        ] = int(len(g))

        if len(g):
            base[
                f"{bucket.lower()}_target_mean"
            ] = float(
                g["target"].mean()
            )

    high = valid[
        valid["bucket"] == "HIGH"
    ]

    low = valid[
        valid["bucket"] == "LOW"
    ]

    if len(high) and len(low):
        base["high_minus_low"] = float(
            high["target"].mean()
            - low["target"].mean()
        )

        base[
            "high_positive_target_rate"
        ] = float(
            (high["target"] > 0).mean()
        )

        base[
            "low_positive_target_rate"
        ] = float(
            (low["target"] > 0).mean()
        )

        base[
            "positive_rate_spread"
        ] = float(
            base["high_positive_target_rate"]
            - base["low_positive_target_rate"]
        )

    return base


def build():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    required = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
        "formula_history_games",
        "current_game_actuals_used_in_alignment",
        "future_game_used_in_alignment",
        "causal_claim_allowed",
        "production_influence",
        "solver_influence",
        "forecast_mutation",
        "player_projection_mutation",
        "persistent_coach_prior_mutation",
    ]

    for spec in TESTS:
        required.extend(
            [
                spec["component"],
                spec["target"],
            ]
        )

    require_columns(
        df,
        sorted(set(required)),
        "Formula Matchup Alignment V1",
    )

    assert_unique(
        df,
        ["game_id", "team"],
        "Formula Matchup Alignment V1",
    )

    safety_cols = [
        "current_game_actuals_used_in_alignment",
        "future_game_used_in_alignment",
        "causal_claim_allowed",
        "production_influence",
        "solver_influence",
        "forecast_mutation",
        "player_projection_mutation",
        "persistent_coach_prior_mutation",
    ]

    for c in safety_cols:
        bad = df[c].fillna(False).astype(bool)

        if bad.any():
            raise RuntimeError(
                f"Upstream safety contract violation: {c}"
            )

    eligible = df[
        pd.to_numeric(
            df["formula_history_games"],
            errors="coerce",
        ) > 0
    ].copy()

    overall_rows = []

    for spec in TESTS:
        overall_rows.append(
            summarize_test(
                eligible,
                spec,
            )
        )

    overall = pd.DataFrame(overall_rows)

    season_rows = []

    for season, g in eligible.groupby(
        "season",
        sort=True,
    ):
        for spec in TESTS:
            season_rows.append(
                summarize_test(
                    g,
                    spec,
                    season=int(season),
                )
            )

    by_season = pd.DataFrame(
        season_rows
    )

    return df, eligible, overall, by_season


def print_table(df, family):
    x = df[
        df["family"] == family
    ][
        [
            "phase",
            "component",
            "target_kind",
            "games",
            "pearson",
            "spearman",
            "low_target_mean",
            "high_target_mean",
            "high_minus_low",
            "positive_rate_spread",
        ]
    ]

    print(
        x.to_string(index=False)
    )


def print_validation(
    source,
    eligible,
    overall,
    by_season,
):
    print(
        "=== FORMULA COMPONENT SIGNAL VALIDATION V1 ==="
    )

    print("INPUT:", INPUT_FILE)

    print(
        "\n=== SOURCE INTEGRITY ==="
    )

    print("rows:", len(source))

    print(
        "unique_game_team:",
        source[
            ["game_id", "team"]
        ].drop_duplicates().shape[0],
    )

    print(
        "eligible_with_prior_history:",
        len(eligible),
    )

    print(
        "\n=== FORMULA COMPONENTS ==="
    )

    print_table(
        overall,
        "FORMULA",
    )

    print(
        "\n=== MATCHUP COMPONENTS ==="
    )

    print_table(
        overall,
        "MATCHUP",
    )

    print(
        "\n=== BY-SEASON HIGH-MINUS-LOW ==="
    )

    season_view = by_season[
        [
            "season",
            "phase",
            "family",
            "component",
            "target_kind",
            "games",
            "spearman",
            "high_minus_low",
            "positive_rate_spread",
        ]
    ]

    print(
        season_view.to_string(
            index=False
        )
    )

    print(
        "\n=== CONTEXT LIMITATION ==="
    )

    print(
        "weather_present_in_alignment_artifact=False"
    )

    print(
        "injury_present_in_alignment_artifact=False"
    )

    print(
        "weather_controlled=False"
    )

    print(
        "injury_controlled=False"
    )

    print(
        "causal_claim_allowed=False"
    )

    print(
        "\n=== VALIDATION CONTRACT ==="
    )

    print("weights_fitted=False")
    print("thresholds_optimized=False")
    print("production_authorized=False")
    print("solver_authorized=False")
    print("forecast_mutation=False")
    print("player_projection_mutation=False")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        source,
        eligible,
        overall,
        by_season,
    ) = build()

    print_validation(
        source,
        eligible,
        overall,
        by_season,
    )

    overall.to_parquet(
        SUMMARY_FILE,
        index=False,
    )

    by_season.to_parquet(
        SEASON_FILE,
        index=False,
    )

    print(
        "\n=== WRITE COMPLETE ==="
    )

    print(SUMMARY_FILE)
    print(SEASON_FILE)


if __name__ == "__main__":
    main()
