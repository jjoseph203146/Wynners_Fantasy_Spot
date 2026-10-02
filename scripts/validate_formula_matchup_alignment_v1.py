#!/usr/bin/env python3

"""
WFS Formula Matchup Alignment Outcome Validation V1

Purpose
-------
Evaluate whether strictly pregame/as-of formula-matchup alignment states
separate CURRENT-game offensive outcomes.

This is validation only.

It does NOT:
- modify Formula Matchup Alignment V1
- fit weights
- optimize thresholds
- mutate forecasts
- mutate projections
- influence solver behavior
- authorize production use

Input
-----
processed/formula_matchup_alignment_v1.parquet

Outputs
-------
processed/validation/formula_matchup_alignment_validation_v1.parquet
processed/validation/formula_matchup_alignment_validation_by_season_v1.parquet

Validation dimensions
---------------------
RUN alignment vs:
- current rushing EPA
- current rushing yards delta vs baseline
- current rushing TD delta vs baseline

PASS alignment vs:
- current passing EPA
- current passing yards delta vs baseline
- current passing TD delta vs baseline

Scoring/result evidence is summarized only where authoritative evidence
exists. It is secondary context, not the definition of offensive success.

Important
---------
Alignment labels are constructed entirely from prior/as-of evidence in
the upstream frozen artifact. Current-game outcomes are validation
targets only.
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

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "validation"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "formula_matchup_alignment_validation_v1.parquet"
)

SEASON_FILE = (
    OUTPUT_DIR
    / "formula_matchup_alignment_validation_by_season_v1.parquet"
)

VERSION = "WFS_FORMULA_MATCHUP_ALIGNMENT_VALIDATION_V1"


def require_columns(df, required, label):
    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def assert_unique(df, keys, label):
    dupes = df.duplicated(
        keys,
        keep=False,
    )

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


def alignment_bucket(value, phase):
    if pd.isna(value):
        return pd.NA

    value = str(value)

    aligned = (
        f"{phase}_FORMULA_VULNERABILITY_ALIGNED"
    )

    mixed = (
        f"{phase}_FORMULA_VULNERABILITY_MIXED"
    )

    unaligned = (
        f"{phase}_FORMULA_VULNERABILITY_UNALIGNED"
    )

    neutral = (
        f"{phase}_FORMULA_VULNERABILITY_NEUTRAL"
    )

    if value == aligned:
        return "ALIGNED"

    if value == mixed:
        return "MIXED"

    if value == unaligned:
        return "UNALIGNED"

    if value == neutral:
        return "NEUTRAL"

    return pd.NA


def safe_rate(series):
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if x.empty:
        return np.nan

    return float(x.mean())


def summarize_group(
    group,
    phase,
    season=None,
):
    if phase == "RUN":
        epa_col = (
            "current_actual_rushing_epa"
        )

        yards_col = (
            "current_rushing_yards_delta_vs_baseline"
        )

        td_col = (
            "current_rushing_tds_delta_vs_baseline"
        )

        bucket_col = "run_alignment_bucket"

    elif phase == "PASS":
        epa_col = (
            "current_actual_passing_epa"
        )

        yards_col = (
            "current_passing_yards_delta_vs_baseline"
        )

        td_col = (
            "current_passing_tds_delta_vs_baseline"
        )

        bucket_col = "pass_alignment_bucket"

    else:
        raise ValueError(phase)

    rows = []

    for bucket, g in group.groupby(
        bucket_col,
        dropna=True,
        sort=False,
    ):
        epa = pd.to_numeric(
            g[epa_col],
            errors="coerce",
        )

        yards = pd.to_numeric(
            g[yards_col],
            errors="coerce",
        )

        tds = pd.to_numeric(
            g[td_col],
            errors="coerce",
        )

        scoring = pd.to_numeric(
            g[
                "current_points_delta_vs_baseline"
            ],
            errors="coerce",
        )

        result_win = np.where(
            g["current_game_result"].isna(),
            np.nan,
            (
                g["current_game_result"]
                == "WIN"
            ).astype(float),
        )

        row = {
            "version": VERSION,
            "season": (
                season
                if season is not None
                else pd.NA
            ),
            "phase": phase,
            "alignment_bucket": bucket,

            "games": int(len(g)),

            "epa_games": int(
                epa.notna().sum()
            ),
            "epa_mean": (
                float(epa.mean())
                if epa.notna().any()
                else np.nan
            ),
            "epa_median": (
                float(epa.median())
                if epa.notna().any()
                else np.nan
            ),
            "positive_epa_rate": (
                float(
                    (epa.dropna() > 0)
                    .mean()
                )
                if epa.notna().any()
                else np.nan
            ),

            "yards_delta_games": int(
                yards.notna().sum()
            ),
            "yards_delta_mean": (
                float(yards.mean())
                if yards.notna().any()
                else np.nan
            ),
            "yards_delta_median": (
                float(yards.median())
                if yards.notna().any()
                else np.nan
            ),
            "above_baseline_yards_rate": (
                float(
                    (yards.dropna() > 0)
                    .mean()
                )
                if yards.notna().any()
                else np.nan
            ),

            "td_delta_games": int(
                tds.notna().sum()
            ),
            "td_delta_mean": (
                float(tds.mean())
                if tds.notna().any()
                else np.nan
            ),
            "td_delta_median": (
                float(tds.median())
                if tds.notna().any()
                else np.nan
            ),
            "above_baseline_td_rate": (
                float(
                    (tds.dropna() > 0)
                    .mean()
                )
                if tds.notna().any()
                else np.nan
            ),

            "scoring_games": int(
                scoring.notna().sum()
            ),
            "points_delta_mean": (
                float(scoring.mean())
                if scoring.notna().any()
                else np.nan
            ),
            "above_baseline_scoring_rate": (
                float(
                    (
                        scoring.dropna()
                        > 0
                    ).mean()
                )
                if scoring.notna().any()
                else np.nan
            ),

            "result_games": int(
                pd.Series(
                    result_win
                ).notna().sum()
            ),
            "win_rate": safe_rate(
                pd.Series(
                    result_win
                )
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def ordered_summary(df):
    if df.empty:
        return df

    order = {
        "ALIGNED": 0,
        "MIXED": 1,
        "UNALIGNED": 2,
        "NEUTRAL": 3,
    }

    x = df.copy()

    x["_order"] = (
        x["alignment_bucket"]
        .map(order)
        .fillna(99)
    )

    sort_cols = []

    if "season" in x.columns:
        sort_cols.append("season")

    sort_cols.extend(
        [
            "phase",
            "_order",
        ]
    )

    x = (
        x.sort_values(
            sort_cols,
            kind="stable",
        )
        .drop(
            columns=["_order"]
        )
        .reset_index(drop=True)
    )

    return x


def build():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    required = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",

        "formula_history_games",
        "formula_history_tier",

        "run_matchup_alignment",
        "pass_matchup_alignment",

        "current_actual_rushing_epa",
        "current_actual_passing_epa",

        "current_rushing_yards_delta_vs_baseline",
        "current_passing_yards_delta_vs_baseline",

        "current_rushing_tds_delta_vs_baseline",
        "current_passing_tds_delta_vs_baseline",

        "current_points_delta_vs_baseline",
        "current_game_result",

        "current_game_actuals_used_in_alignment",
        "future_game_used_in_alignment",
        "causal_claim_allowed",
        "production_influence",
        "solver_influence",
        "forecast_mutation",
        "player_projection_mutation",
        "persistent_coach_prior_mutation",
    ]

    require_columns(
        df,
        required,
        "formula matchup alignment",
    )

    assert_unique(
        df,
        ["game_id", "team"],
        "formula matchup alignment",
    )

    # ----------------------------------------------------------
    # Enforce upstream safety contract before validation.
    # ----------------------------------------------------------

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
        bad = (
            df[c]
            .fillna(False)
            .astype(bool)
        )

        if bad.any():
            raise RuntimeError(
                f"Safety contract violation: {c}"
            )

    df = df.copy()

    df["run_alignment_bucket"] = [
        alignment_bucket(
            x,
            "RUN",
        )
        for x in df[
            "run_matchup_alignment"
        ]
    ]

    df["pass_alignment_bucket"] = [
        alignment_bucket(
            x,
            "PASS",
        )
        for x in df[
            "pass_matchup_alignment"
        ]
    ]

    # Rows without prior history cannot participate.
    eligible = df[
        pd.to_numeric(
            df["formula_history_games"],
            errors="coerce",
        )
        > 0
    ].copy()

    # Overall summaries.
    overall_parts = []

    for phase in [
        "RUN",
        "PASS",
    ]:
        overall_parts.append(
            summarize_group(
                eligible,
                phase,
            )
        )

    overall = pd.concat(
        overall_parts,
        ignore_index=True,
    )

    overall = ordered_summary(
        overall
    )

    # Season-by-season summaries.
    season_parts = []

    for season, season_df in eligible.groupby(
        "season",
        sort=True,
    ):
        for phase in [
            "RUN",
            "PASS",
        ]:
            season_parts.append(
                summarize_group(
                    season_df,
                    phase,
                    season=int(season),
                )
            )

    by_season = pd.concat(
        season_parts,
        ignore_index=True,
    )

    by_season = ordered_summary(
        by_season
    )

    return (
        df,
        eligible,
        overall,
        by_season,
    )


def spread_table(summary, phase):
    x = summary[
        summary["phase"] == phase
    ].copy()

    if x.empty:
        return pd.DataFrame()

    wanted = [
        "alignment_bucket",
        "games",
        "epa_mean",
        "positive_epa_rate",
        "yards_delta_mean",
        "above_baseline_yards_rate",
        "td_delta_mean",
        "above_baseline_td_rate",
        "scoring_games",
        "points_delta_mean",
        "above_baseline_scoring_rate",
        "result_games",
        "win_rate",
    ]

    return x[wanted]


def print_pairwise_differences(
    overall,
    phase,
):
    x = (
        overall[
            overall["phase"] == phase
        ]
        .set_index(
            "alignment_bucket"
        )
    )

    print(
        f"\n=== {phase} ALIGNED VS UNALIGNED ==="
    )

    if (
        "ALIGNED" not in x.index
        or
        "UNALIGNED" not in x.index
    ):
        print(
            "Insufficient buckets for comparison."
        )
        return

    metrics = [
        "epa_mean",
        "positive_epa_rate",
        "yards_delta_mean",
        "above_baseline_yards_rate",
        "td_delta_mean",
        "above_baseline_td_rate",
        "points_delta_mean",
        "above_baseline_scoring_rate",
        "win_rate",
    ]

    for metric in metrics:
        a = x.loc[
            "ALIGNED",
            metric,
        ]

        u = x.loc[
            "UNALIGNED",
            metric,
        ]

        if pd.isna(a) or pd.isna(u):
            diff = np.nan
        else:
            diff = float(a - u)

        print(
            f"{metric}: "
            f"aligned={a} "
            f"unaligned={u} "
            f"diff={diff}"
        )


def print_validation(
    source,
    eligible,
    overall,
    by_season,
):
    print(
        "=== FORMULA MATCHUP ALIGNMENT "
        "OUTCOME VALIDATION V1 ==="
    )

    print(
        "INPUT:",
        INPUT_FILE,
    )

    print(
        "SUMMARY_OUTPUT:",
        SUMMARY_FILE,
    )

    print(
        "SEASON_OUTPUT:",
        SEASON_FILE,
    )

    print(
        "\n=== SOURCE INTEGRITY ==="
    )

    print(
        "rows:",
        len(source),
    )

    print(
        "unique_game_team:",
        source[
            ["game_id", "team"]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "eligible_with_prior_history:",
        len(eligible),
    )

    print(
        "\n=== RUN OVERALL ==="
    )

    print(
        spread_table(
            overall,
            "RUN",
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== PASS OVERALL ==="
    )

    print(
        spread_table(
            overall,
            "PASS",
        ).to_string(
            index=False
        )
    )

    print_pairwise_differences(
        overall,
        "RUN",
    )

    print_pairwise_differences(
        overall,
        "PASS",
    )

    print(
        "\n=== BY-SEASON CORE VIEW ==="
    )

    core = by_season[
        [
            "season",
            "phase",
            "alignment_bucket",
            "games",
            "epa_mean",
            "positive_epa_rate",
            "yards_delta_mean",
            "above_baseline_yards_rate",
            "td_delta_mean",
            "above_baseline_td_rate",
        ]
    ]

    print(
        core.to_string(
            index=False
        )
    )

    print(
        "\n=== VALIDATION CONTRACT ==="
    )

    print(
        "weights_fitted=False"
    )

    print(
        "thresholds_optimized=False"
    )

    print(
        "production_authorized=False"
    )

    print(
        "solver_authorized=False"
    )

    print(
        "forecast_mutation=False"
    )

    print(
        "player_projection_mutation=False"
    )


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

    print(
        SUMMARY_FILE
    )

    print(
        SEASON_FILE
    )


if __name__ == "__main__":
    main()
