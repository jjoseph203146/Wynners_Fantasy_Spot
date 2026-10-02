#!/usr/bin/env python3

"""
WFS Postgame Formula Evidence V1

Purpose
-------
Consume the frozen Postgame Team Learning Evidence V1 artifact and create
deterministic, descriptive evidence about:

1. How the team's actual approach differed from its own pregame baseline.
2. Whether passing/rushing generated positive or negative EPA.
3. Whether passing/rushing production exceeded recent pregame baselines.
4. Whether scoring exceeded recent pregame expectations.
5. The actual game result.

This artifact DOES NOT claim causation.

Example:
    MORE_RUN_ORIENTED_THAN_BASELINE
    POSITIVE_RUSH_EPA
    ABOVE_BASELINE_RUSH_YARDS
    ABOVE_BASELINE_RUSH_TDS
    WIN

does NOT mean:
    "Rushing caused the win."

Canonical grain
---------------
Exactly one row per game_id + team.

Input
-----
processed/postgame_team_learning_evidence_v1.parquet

Output
------
processed/postgame_formula_evidence_v1.parquet

Safety
------
Shadow/descriptive only.
No forecast, projection, solver, coaching-prior, or production influence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "processed"
    / "postgame_team_learning_evidence_v1.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "processed"
    / "postgame_formula_evidence_v1.parquet"
)

VERSION = "WFS_POSTGAME_FORMULA_EVIDENCE_V1"


PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
FORECAST_MUTATION = False
PLAYER_PROJECTION_MUTATION = False
PERSISTENT_COACH_PRIOR_MUTATION = False


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def assert_unique(
    df: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
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


def directional_label(
    value: object,
    positive_label: str,
    negative_label: str,
    zero_label: str,
) -> object:
    """
    Classify sign only.

    No arbitrary magnitude threshold is introduced in V1.
    """
    if pd.isna(value):
        return pd.NA

    value = float(value)

    if value > 0:
        return positive_label

    if value < 0:
        return negative_label

    return zero_label


def epa_label(
    value: object,
    phase: str,
) -> object:
    """
    EPA sign classification only.

    Positive EPA means the phase generated positive expected-points value.
    Negative EPA means the phase generated negative expected-points value.

    V1 does not introduce arbitrary strong/moderate/weak thresholds.
    """
    if pd.isna(value):
        return pd.NA

    value = float(value)

    if value > 0:
        return f"POSITIVE_{phase}_EPA"

    if value < 0:
        return f"NEGATIVE_{phase}_EPA"

    return f"NEUTRAL_{phase}_EPA"


def result_label(
    team_score: object,
    opponent_score: object,
) -> object:
    if pd.isna(team_score) or pd.isna(opponent_score):
        return pd.NA

    team_score = float(team_score)
    opponent_score = float(opponent_score)

    if team_score > opponent_score:
        return "WIN"

    if team_score < opponent_score:
        return "LOSS"

    return "TIE"


def build_evidence_labels(row: pd.Series) -> str:
    """
    Construct a pipe-delimited evidence set.

    Labels remain descriptive and non-causal.
    """
    labels: list[str] = []

    candidates = [
        row.get("approach_pass_rate"),
        row.get("rush_efficiency"),
        row.get("pass_efficiency"),
        row.get("rush_yardage_production"),
        row.get("pass_yardage_production"),
        row.get("rush_td_production"),
        row.get("pass_td_production"),
        row.get("scoring_production"),
        row.get("game_result"),
    ]

    for value in candidates:
        if pd.notna(value):
            labels.append(str(value))

    return "|".join(labels)


def build() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    src = pd.read_parquet(INPUT_FILE)

    required = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",

        "team_score",
        "opponent_score",

        "actual_pass_rate",
        "actual_rush_rate",

        "pregame_pass_rate_avg_3",
        "pregame_rush_rate_avg_3",

        "actual_passing_yards",
        "actual_rushing_yards",

        "pregame_passing_yards_avg_3",
        "pregame_rushing_yards_avg_3",

        "actual_passing_tds",
        "actual_rushing_tds",

        "pregame_passing_tds_avg_3",
        "pregame_rushing_tds_avg_3",

        "actual_passing_epa",
        "actual_rushing_epa",

        "pregame_points_for_avg_3",

        "actual_pass_attempts",
        "actual_rush_attempts",
        "actual_offensive_plays",

        "rb_carries",
        "rb_targets",
        "wr_targets",
        "te_targets",

        "has_actual_team_stats",
        "has_pregame_environment",
        "has_coaching_evidence",
        "has_expected_pass_rate",
        "has_player_opportunity_evidence",
    ]

    require_columns(
        src,
        required,
        "postgame team learning evidence",
    )

    assert_unique(
        src,
        ["game_id", "team"],
        "postgame team learning evidence",
    )

    out = pd.DataFrame()

    # --------------------------------------------------------------
    # Identity
    # --------------------------------------------------------------

    out["version"] = VERSION

    for c in [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    ]:
        out[c] = src[c]

    # --------------------------------------------------------------
    # Actual result
    # --------------------------------------------------------------

    out["team_score"] = src["team_score"]
    out["opponent_score"] = src["opponent_score"]

    out["actual_margin"] = (
        pd.to_numeric(
            src["team_score"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["opponent_score"],
            errors="coerce",
        )
    )

    out["game_result"] = [
        result_label(a, b)
        for a, b in zip(
            src["team_score"],
            src["opponent_score"],
        )
    ]

    # --------------------------------------------------------------
    # Volume / approach
    # --------------------------------------------------------------

    out["actual_pass_attempts"] = (
        src["actual_pass_attempts"]
    )
    out["actual_rush_attempts"] = (
        src["actual_rush_attempts"]
    )
    out["actual_offensive_plays"] = (
        src["actual_offensive_plays"]
    )

    out["actual_pass_rate"] = src["actual_pass_rate"]
    out["actual_rush_rate"] = src["actual_rush_rate"]

    out["pregame_pass_rate_avg_3"] = (
        src["pregame_pass_rate_avg_3"]
    )

    out["pregame_rush_rate_avg_3"] = (
        src["pregame_rush_rate_avg_3"]
    )

    out["pass_rate_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_pass_rate"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_pass_rate_avg_3"],
            errors="coerce",
        )
    )

    out["rush_rate_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_rush_rate"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_rush_rate_avg_3"],
            errors="coerce",
        )
    )

    out["approach_pass_rate"] = [
        directional_label(
            x,
            "MORE_PASS_ORIENTED_THAN_BASELINE",
            "MORE_RUN_ORIENTED_THAN_BASELINE",
            "SAME_PASS_RATE_AS_BASELINE",
        )
        for x in out["pass_rate_delta_vs_baseline"]
    ]

    # --------------------------------------------------------------
    # Efficiency
    # --------------------------------------------------------------

    out["actual_passing_epa"] = src["actual_passing_epa"]
    out["actual_rushing_epa"] = src["actual_rushing_epa"]

    out["pass_efficiency"] = [
        epa_label(x, "PASS")
        for x in src["actual_passing_epa"]
    ]

    out["rush_efficiency"] = [
        epa_label(x, "RUSH")
        for x in src["actual_rushing_epa"]
    ]

    # --------------------------------------------------------------
    # Yardage production vs own recent baseline
    # --------------------------------------------------------------

    out["actual_passing_yards"] = (
        src["actual_passing_yards"]
    )

    out["pregame_passing_yards_avg_3"] = (
        src["pregame_passing_yards_avg_3"]
    )

    out["passing_yards_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_passing_yards"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_passing_yards_avg_3"],
            errors="coerce",
        )
    )

    out["pass_yardage_production"] = [
        directional_label(
            x,
            "ABOVE_BASELINE_PASS_YARDS",
            "BELOW_BASELINE_PASS_YARDS",
            "AT_BASELINE_PASS_YARDS",
        )
        for x in out["passing_yards_delta_vs_baseline"]
    ]

    out["actual_rushing_yards"] = (
        src["actual_rushing_yards"]
    )

    out["pregame_rushing_yards_avg_3"] = (
        src["pregame_rushing_yards_avg_3"]
    )

    out["rushing_yards_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_rushing_yards"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_rushing_yards_avg_3"],
            errors="coerce",
        )
    )

    out["rush_yardage_production"] = [
        directional_label(
            x,
            "ABOVE_BASELINE_RUSH_YARDS",
            "BELOW_BASELINE_RUSH_YARDS",
            "AT_BASELINE_RUSH_YARDS",
        )
        for x in out["rushing_yards_delta_vs_baseline"]
    ]

    # --------------------------------------------------------------
    # TD production vs own recent baseline
    # --------------------------------------------------------------

    out["actual_passing_tds"] = src["actual_passing_tds"]

    out["pregame_passing_tds_avg_3"] = (
        src["pregame_passing_tds_avg_3"]
    )

    out["passing_tds_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_passing_tds"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_passing_tds_avg_3"],
            errors="coerce",
        )
    )

    out["pass_td_production"] = [
        directional_label(
            x,
            "ABOVE_BASELINE_PASS_TDS",
            "BELOW_BASELINE_PASS_TDS",
            "AT_BASELINE_PASS_TDS",
        )
        for x in out["passing_tds_delta_vs_baseline"]
    ]

    out["actual_rushing_tds"] = src["actual_rushing_tds"]

    out["pregame_rushing_tds_avg_3"] = (
        src["pregame_rushing_tds_avg_3"]
    )

    out["rushing_tds_delta_vs_baseline"] = (
        pd.to_numeric(
            src["actual_rushing_tds"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_rushing_tds_avg_3"],
            errors="coerce",
        )
    )

    out["rush_td_production"] = [
        directional_label(
            x,
            "ABOVE_BASELINE_RUSH_TDS",
            "BELOW_BASELINE_RUSH_TDS",
            "AT_BASELINE_RUSH_TDS",
        )
        for x in out["rushing_tds_delta_vs_baseline"]
    ]

    # --------------------------------------------------------------
    # Team scoring vs own recent baseline
    # --------------------------------------------------------------

    out["pregame_points_for_avg_3"] = (
        src["pregame_points_for_avg_3"]
    )

    out["points_delta_vs_baseline"] = (
        pd.to_numeric(
            src["team_score"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            src["pregame_points_for_avg_3"],
            errors="coerce",
        )
    )

    out["scoring_production"] = [
        directional_label(
            x,
            "ABOVE_BASELINE_SCORING",
            "BELOW_BASELINE_SCORING",
            "AT_BASELINE_SCORING",
        )
        for x in out["points_delta_vs_baseline"]
    ]

    # --------------------------------------------------------------
    # Coaching context copied as evidence, not reinterpreted
    # --------------------------------------------------------------

    for c in [
        "expected_pass_rate",
        "final_pass_rate",
        "pass_rate_delta",
        "coaching_confidence",
        "coaching_interpretation",
        "coach_identity",
        "coach_change_flag",
        "coach_team_regime_key",
    ]:
        if c in src.columns:
            out[c] = src[c]

    # --------------------------------------------------------------
    # FanDuel-relevant opportunity consequences
    # --------------------------------------------------------------

    for c in [
        "rb_carries",
        "rb_targets",
        "wr_targets",
        "te_targets",
        "role_expansion_count",
        "role_decline_count",
        "high_usage_count",
        "starter_usage_count",
    ]:
        if c in src.columns:
            out[c] = src[c]

    # --------------------------------------------------------------
    # Evidence availability
    # --------------------------------------------------------------

    for c in [
        "has_actual_team_stats",
        "has_pregame_environment",
        "has_coaching_evidence",
        "has_expected_pass_rate",
        "has_player_opportunity_evidence",
    ]:
        out[c] = src[c]

    # Result evidence exists only where actual scores are available.
    out["has_result_evidence"] = (
        out["team_score"].notna()
        & out["opponent_score"].notna()
    ).astype(int)

    out["has_formula_core_evidence"] = (
        out["actual_pass_rate"].notna()
        & out["pregame_pass_rate_avg_3"].notna()
        & out["actual_passing_epa"].notna()
        & out["actual_rushing_epa"].notna()
        & out["actual_passing_yards"].notna()
        & out["actual_rushing_yards"].notna()
    ).astype(int)

    # --------------------------------------------------------------
    # Combined descriptive evidence labels
    # --------------------------------------------------------------

    out["formula_evidence_labels"] = out.apply(
        build_evidence_labels,
        axis=1,
    )

    # Explicit non-causal contract.
    out["causal_claim_allowed"] = False

    # --------------------------------------------------------------
    # Safety
    # --------------------------------------------------------------

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
        "postgame formula evidence",
    )

    return out


def print_validation(df: pd.DataFrame) -> None:
    print("\n=== POSTGAME FORMULA EVIDENCE V1 ===")
    print("OUTPUT:", OUTPUT_FILE)

    print("\n=== KEY INTEGRITY ===")
    print("rows:", len(df))

    print(
        "unique_game_team:",
        df[["game_id", "team"]]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "duplicate_game_team:",
        int(
            df.duplicated(
                ["game_id", "team"]
            ).sum()
        ),
    )

    print("\n=== COVERAGE ===")

    print(
        "formula_core:",
        int(df["has_formula_core_evidence"].sum()),
        "/",
        len(df),
    )

    print(
        "result_evidence:",
        int(df["has_result_evidence"].sum()),
        "/",
        len(df),
    )

    if "has_coaching_evidence" in df.columns:
        print(
            "coaching_evidence:",
            int(df["has_coaching_evidence"].sum()),
            "/",
            len(df),
        )

    print("\n=== APPROACH DISTRIBUTION ===")
    print(
        df["approach_pass_rate"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== PASS EPA DISTRIBUTION ===")
    print(
        df["pass_efficiency"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== RUSH EPA DISTRIBUTION ===")
    print(
        df["rush_efficiency"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== RESULT DISTRIBUTION ===")
    print(
        df["game_result"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== SAFETY FLAGS ===")

    for c in [
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

    print("\n=== 2026 COMPLETE EXAMPLES ===")

    cols = [
        "game_id",
        "team",
        "opponent_team",
        "team_score",
        "opponent_score",
        "game_result",

        "pregame_pass_rate_avg_3",
        "actual_pass_rate",
        "pass_rate_delta_vs_baseline",
        "approach_pass_rate",

        "actual_passing_epa",
        "pass_efficiency",

        "actual_rushing_epa",
        "rush_efficiency",

        "actual_passing_yards",
        "pregame_passing_yards_avg_3",
        "pass_yardage_production",

        "actual_rushing_yards",
        "pregame_rushing_yards_avg_3",
        "rush_yardage_production",

        "actual_passing_tds",
        "pass_td_production",

        "actual_rushing_tds",
        "rush_td_production",

        "points_delta_vs_baseline",
        "scoring_production",

        "coaching_interpretation",

        "rb_carries",
        "rb_targets",
        "wr_targets",
        "te_targets",

        "formula_evidence_labels",
    ]

    cols = [
        c for c in cols
        if c in df.columns
    ]

    sample = (
        df[
            (df["season"] == 2026)
            & (df["has_result_evidence"] == 1)
        ][cols]
        .sort_values(
            ["game_id", "team"]
        )
        .head(12)
    )

    if sample.empty:
        print("NO COMPLETE 2026 ROWS")
    else:
        print(
            sample.to_string(
                index=False
            )
        )


def main() -> None:
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
    print("causal_claim_allowed=False")
    print("production_influence=False")
    print("solver_influence=False")
    print("forecast_mutation=False")
    print("player_projection_mutation=False")
    print("persistent_coach_prior_mutation=False")


if __name__ == "__main__":
    main()
