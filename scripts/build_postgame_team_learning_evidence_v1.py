#!/usr/bin/env python3

"""
WFS Postgame Team Learning Evidence V1

Purpose
-------
Build a deterministic, shadow-only team-game evidence bridge connecting:

    immutable pregame expectation
        ->
    actual team performance
        ->
    postgame coaching/game-plan evidence
        ->
    objective reconciliation deltas

Canonical grain
---------------
Exactly one row per:

    game_id + team

This artifact is DESCRIPTIVE ONLY.

It MUST NOT:
- mutate forecasts
- mutate player projections
- influence the solver
- mutate coaching priors
- alter production databases
- use future-game information

Output
------
processed/postgame_team_learning_evidence_v1.parquet
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

NFL_DB = ROOT / "data" / "nfl.db"

FORECAST_FILE = (
    ROOT / "processed" / "forecast_prospective_evaluation_v1.csv"
)

COACHING_FILE = (
    ROOT
    / "processed"
    / "coaching_intelligence"
    / "postgame_coach_attached_evidence_v1.csv"
)

OUTPUT_FILE = (
    ROOT / "processed" / "postgame_team_learning_evidence_v1.parquet"
)

VERSION = "WFS_POSTGAME_TEAM_LEARNING_EVIDENCE_V1"


# ---------------------------------------------------------------------
# Safety contract
# ---------------------------------------------------------------------

PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
FORECAST_MUTATION = False
PLAYER_PROJECTION_MUTATION = False
PERSISTENT_COACH_PRIOR_MUTATION = False


def open_ro(path: Path) -> sqlite3.Connection:
    """Open SQLite database strictly read-only."""
    if not path.exists():
        raise FileNotFoundError(path)

    return sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )


def safe_div(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Vectorized division with zero/invalid denominators -> NaN."""
    den = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")

    return np.where(
        den.notna() & (den != 0),
        num / den,
        np.nan,
    )


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
            f"{label} has duplicate keys {keys}: {sample}"
        )


# ---------------------------------------------------------------------
# Authoritative actual team performance
# ---------------------------------------------------------------------

def load_actual_team_stats() -> pd.DataFrame:
    with open_ro(NFL_DB) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                season_type,
                team,
                opponent_team,

                attempts AS actual_pass_attempts,
                carries AS actual_rush_attempts,

                passing_yards AS actual_passing_yards,
                rushing_yards AS actual_rushing_yards,

                passing_tds AS actual_passing_tds,
                rushing_tds AS actual_rushing_tds,

                passing_epa AS actual_passing_epa,
                rushing_epa AS actual_rushing_epa,

                completions AS actual_completions,
                interceptions AS actual_interceptions,
                sacks_suffered AS actual_sacks_suffered,

                updated_at AS actual_stats_updated_at

            FROM team_game_stats
            WHERE game_id IS NOT NULL
              AND team IS NOT NULL
            """,
            conn,
        )

    require_columns(
        df,
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
            "actual_pass_attempts",
            "actual_rush_attempts",
        ],
        "team_game_stats",
    )

    assert_unique(
        df,
        ["game_id", "team"],
        "team_game_stats",
    )

    df["actual_offensive_plays"] = (
        pd.to_numeric(
            df["actual_pass_attempts"],
            errors="coerce",
        )
        +
        pd.to_numeric(
            df["actual_rush_attempts"],
            errors="coerce",
        )
    )

    df["actual_pass_rate"] = safe_div(
        df["actual_pass_attempts"],
        df["actual_offensive_plays"],
    )

    df["actual_rush_rate"] = safe_div(
        df["actual_rush_attempts"],
        df["actual_offensive_plays"],
    )

    return df


# ---------------------------------------------------------------------
# Pregame environment
# ---------------------------------------------------------------------

def load_pregame_environment() -> pd.DataFrame:
    with open_ro(NFL_DB) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                game_id,
                team,
                opponent_team AS pregame_opponent_team,

                history_games AS pregame_history_games,

                points_for_last AS pregame_points_for_last,
                points_for_avg_3 AS pregame_points_for_avg_3,
                points_for_avg_5 AS pregame_points_for_avg_5,

                points_against_last
                    AS pregame_points_against_last,
                points_against_avg_3
                    AS pregame_points_against_avg_3,
                points_against_avg_5
                    AS pregame_points_against_avg_5,

                offensive_plays_avg_3
                    AS pregame_offensive_plays_avg_3,
                offensive_plays_avg_5
                    AS pregame_offensive_plays_avg_5,

                pass_attempts_avg_3
                    AS pregame_pass_attempts_avg_3,
                pass_attempts_avg_5
                    AS pregame_pass_attempts_avg_5,

                rush_attempts_avg_3
                    AS pregame_rush_attempts_avg_3,
                rush_attempts_avg_5
                    AS pregame_rush_attempts_avg_5,

                pass_rate_avg_3
                    AS pregame_pass_rate_avg_3,
                pass_rate_avg_5
                    AS pregame_pass_rate_avg_5,

                rush_rate_avg_3
                    AS pregame_rush_rate_avg_3,
                rush_rate_avg_5
                    AS pregame_rush_rate_avg_5,

                passing_yards_avg_3
                    AS pregame_passing_yards_avg_3,
                rushing_yards_avg_3
                    AS pregame_rushing_yards_avg_3,

                passing_tds_avg_3
                    AS pregame_passing_tds_avg_3,
                rushing_tds_avg_3
                    AS pregame_rushing_tds_avg_3,

                opponent_points_allowed_avg_3,
                opponent_points_allowed_avg_5,

                opponent_pass_yards_allowed_avg_3,
                opponent_rush_yards_allowed_avg_3,

                opponent_pass_tds_allowed_avg_3,
                opponent_rush_tds_allowed_avg_3,

                team_scoring_trend,
                opponent_scoring_trend,
                pace_trend,
                pass_rate_trend,

                updated_at AS pregame_environment_updated_at

            FROM team_pregame_environment
            WHERE game_id IS NOT NULL
              AND team IS NOT NULL
            """,
            conn,
        )

    assert_unique(
        df,
        ["game_id", "team"],
        "team_pregame_environment",
    )

    return df


# ---------------------------------------------------------------------
# Immutable prospective forecast -> team-relative rows
# ---------------------------------------------------------------------

def load_team_forecast() -> pd.DataFrame:
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(FORECAST_FILE)

    src = pd.read_csv(FORECAST_FILE)

    require_columns(
        src,
        [
            "game_id",
            "season",
            "week",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "actual_winner",
            "snapshot_id",
            "captured_at_utc",
            "seconds_before_kickoff",
            "forecast_status",
            "pred_home_margin",
            "pred_total_points",
            "pred_home_points",
            "pred_away_points",
            "pred_winner",
        ],
        "forecast prospective evaluation",
    )

    assert_unique(
        src,
        ["game_id"],
        "forecast prospective evaluation",
    )

    rows: list[dict] = []

    for r in src.to_dict("records"):
        game_id = r["game_id"]

        home = r["home_team"]
        away = r["away_team"]

        home_score = r["home_score"]
        away_score = r["away_score"]

        pred_home = r["pred_home_points"]
        pred_away = r["pred_away_points"]

        common = {
            "game_id": game_id,
            "forecast_snapshot_id": r["snapshot_id"],
            "forecast_captured_at_utc":
                r["captured_at_utc"],
            "forecast_seconds_before_kickoff":
                r["seconds_before_kickoff"],
            "forecast_status": r["forecast_status"],
            "predicted_winner": r["pred_winner"],
            "predicted_total_points":
                r["pred_total_points"],
            "forecast_pred_home_margin":
                r["pred_home_margin"],
        }

        rows.append(
            {
                **common,
                "team": home,
                "forecast_opponent_team": away,

                "team_score": home_score,
                "opponent_score": away_score,

                "actual_winner": r["actual_winner"],

                "predicted_team_points": pred_home,
                "predicted_opponent_points": pred_away,

                "predicted_team_margin":
                    (
                        pred_home - pred_away
                        if pd.notna(pred_home)
                        and pd.notna(pred_away)
                        else np.nan
                    ),
            }
        )

        rows.append(
            {
                **common,
                "team": away,
                "forecast_opponent_team": home,

                "team_score": away_score,
                "opponent_score": home_score,

                "actual_winner": r["actual_winner"],

                "predicted_team_points": pred_away,
                "predicted_opponent_points": pred_home,

                "predicted_team_margin":
                    (
                        pred_away - pred_home
                        if pd.notna(pred_home)
                        and pd.notna(pred_away)
                        else np.nan
                    ),
            }
        )

    df = pd.DataFrame(rows)

    df["actual_margin"] = (
        pd.to_numeric(
            df["team_score"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            df["opponent_score"],
            errors="coerce",
        )
    )

    df["actual_win"] = np.where(
        df["team_score"] > df["opponent_score"],
        1,
        np.where(
            df["team_score"] < df["opponent_score"],
            0,
            np.nan,
        ),
    )

    df["predicted_team_win"] = np.where(
        df["predicted_winner"] == df["team"],
        1,
        np.where(
            df["predicted_winner"].notna(),
            0,
            np.nan,
        ),
    )

    df["winner_prediction_correct"] = np.where(
        df["actual_winner"].notna()
        & df["predicted_winner"].notna(),
        (
            df["actual_winner"]
            == df["predicted_winner"]
        ).astype(int),
        np.nan,
    )

    df["team_points_prediction_error"] = (
        pd.to_numeric(
            df["predicted_team_points"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            df["team_score"],
            errors="coerce",
        )
    )

    df["team_margin_prediction_error"] = (
        pd.to_numeric(
            df["predicted_team_margin"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            df["actual_margin"],
            errors="coerce",
        )
    )

    assert_unique(
        df,
        ["game_id", "team"],
        "team-relative forecast evidence",
    )

    return df


# ---------------------------------------------------------------------
# Existing postgame coaching evidence
# ---------------------------------------------------------------------

def load_coaching() -> pd.DataFrame:
    if not COACHING_FILE.exists():
        raise FileNotFoundError(COACHING_FILE)

    src = pd.read_csv(COACHING_FILE)

    required = [
        "game_id",
        "team",
        "expected_pass_rate",
        "final_pass_rate",
        "pass_rate_delta",
        "early_down_pass_rate",
        "red_zone_pass_rate",
        "third_down_pass_rate",
        "short_yardage_pass_rate",
        "shotgun_rate",
        "no_huddle_rate",
        "scrimmage_plays",
        "pass_calls",
        "rush_calls",
        "target_concentration",
        "carry_concentration",
        "confidence",
        "interpretation",
        "coach_identity",
        "previous_coach_identity",
        "coach_change_flag",
        "coach_team_era_games_prior",
        "coach_career_games_prior",
        "coach_history_tier",
        "coach_team_history_tier",
        "coach_cold_start_flag",
        "coach_team_cold_start_flag",
        "coach_team_regime_key",
    ]

    require_columns(
        src,
        required,
        "postgame coach-attached evidence",
    )

    assert_unique(
        src,
        ["game_id", "team"],
        "postgame coach-attached evidence",
    )

    keep = required.copy()

    if "generated_at_utc" in src.columns:
        keep.append("generated_at_utc")

    df = src[keep].copy()

    df = df.rename(
        columns={
            "confidence": "coaching_confidence",
            "interpretation":
                "coaching_interpretation",
            "generated_at_utc":
                "coaching_generated_at_utc",
        }
    )

    return df


# ---------------------------------------------------------------------
# Player opportunity summaries
# ---------------------------------------------------------------------

def load_player_opportunity_summary() -> pd.DataFrame:
    with open_ro(NFL_DB) as conn:
        src = pd.read_sql_query(
            """
            SELECT
                game_id,
                team,
                position_group,
                carries,
                targets,
                offense_snaps,
                role_expansion_flag,
                role_decline_flag,
                high_usage_flag,
                starter_usage_flag
            FROM player_weekly_usage
            WHERE game_id IS NOT NULL
              AND team IS NOT NULL
              AND position_group IN ('QB', 'RB', 'FB', 'WR', 'TE')
            """,
            conn,
        )

    if src.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "qb_carries",
                "rb_carries",
                "rb_targets",
                "wr_targets",
                "te_targets",
                "role_expansion_count",
                "role_decline_count",
                "high_usage_count",
                "starter_usage_count",
            ]
        )

    for c in [
        "carries",
        "targets",
        "offense_snaps",
        "role_expansion_flag",
        "role_decline_flag",
        "high_usage_flag",
        "starter_usage_flag",
    ]:
        src[c] = pd.to_numeric(
            src[c],
            errors="coerce",
        ).fillna(0)

    src["qb_carries_component"] = np.where(
        src["position_group"] == "QB",
        src["carries"],
        0,
    )

    src["rb_carries_component"] = np.where(
        src["position_group"].isin(["RB", "FB"]),
        src["carries"],
        0,
    )

    src["rb_targets_component"] = np.where(
        src["position_group"].isin(["RB", "FB"]),
        src["targets"],
        0,
    )

    src["wr_targets_component"] = np.where(
        src["position_group"] == "WR",
        src["targets"],
        0,
    )

    src["te_targets_component"] = np.where(
        src["position_group"] == "TE",
        src["targets"],
        0,
    )

    grouped = (
        src.groupby(
            ["game_id", "team"],
            as_index=False,
        )
        .agg(
            qb_carries=(
                "qb_carries_component",
                "sum",
            ),
            rb_carries=(
                "rb_carries_component",
                "sum",
            ),
            rb_targets=(
                "rb_targets_component",
                "sum",
            ),
            wr_targets=(
                "wr_targets_component",
                "sum",
            ),
            te_targets=(
                "te_targets_component",
                "sum",
            ),
            role_expansion_count=(
                "role_expansion_flag",
                "sum",
            ),
            role_decline_count=(
                "role_decline_flag",
                "sum",
            ),
            high_usage_count=(
                "high_usage_flag",
                "sum",
            ),
            starter_usage_count=(
                "starter_usage_flag",
                "sum",
            ),
        )
    )

    assert_unique(
        grouped,
        ["game_id", "team"],
        "player opportunity summary",
    )

    return grouped


# ---------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------

def add_objective_deltas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    pairs = {
        "pass_rate_vs_pregame_avg3":
            (
                "actual_pass_rate",
                "pregame_pass_rate_avg_3",
            ),

        "rush_rate_vs_pregame_avg3":
            (
                "actual_rush_rate",
                "pregame_rush_rate_avg_3",
            ),

        "offensive_plays_vs_pregame_avg3":
            (
                "actual_offensive_plays",
                "pregame_offensive_plays_avg_3",
            ),

        "pass_attempts_vs_pregame_avg3":
            (
                "actual_pass_attempts",
                "pregame_pass_attempts_avg_3",
            ),

        "rush_attempts_vs_pregame_avg3":
            (
                "actual_rush_attempts",
                "pregame_rush_attempts_avg_3",
            ),

        "passing_yards_vs_pregame_avg3":
            (
                "actual_passing_yards",
                "pregame_passing_yards_avg_3",
            ),

        "rushing_yards_vs_pregame_avg3":
            (
                "actual_rushing_yards",
                "pregame_rushing_yards_avg_3",
            ),

        "passing_tds_vs_pregame_avg3":
            (
                "actual_passing_tds",
                "pregame_passing_tds_avg_3",
            ),

        "rushing_tds_vs_pregame_avg3":
            (
                "actual_rushing_tds",
                "pregame_rushing_tds_avg_3",
            ),

        "actual_vs_expected_pass_rate":
            (
                "actual_pass_rate",
                "expected_pass_rate",
            ),
    }

    for new_col, (actual_col, baseline_col) in pairs.items():
        if (
            actual_col in out.columns
            and baseline_col in out.columns
        ):
            out[new_col] = (
                pd.to_numeric(
                    out[actual_col],
                    errors="coerce",
                )
                -
                pd.to_numeric(
                    out[baseline_col],
                    errors="coerce",
                )
            )

    return out


def validate_identity_consistency(
    df: pd.DataFrame,
) -> None:
    problems: list[str] = []

    if "pregame_opponent_team" in df.columns:
        bad = (
            df["pregame_opponent_team"].notna()
            & (
                df["pregame_opponent_team"]
                != df["opponent_team"]
            )
        )

        if bad.any():
            problems.append(
                "pregame opponent identity mismatch: "
                f"{int(bad.sum())}"
            )

    if "forecast_opponent_team" in df.columns:
        bad = (
            df["forecast_opponent_team"].notna()
            & (
                df["forecast_opponent_team"]
                != df["opponent_team"]
            )
        )

        if bad.any():
            problems.append(
                "forecast opponent identity mismatch: "
                f"{int(bad.sum())}"
            )

    if problems:
        raise RuntimeError(
            "; ".join(problems)
        )


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------

def build() -> pd.DataFrame:
    actual = load_actual_team_stats()
    pregame = load_pregame_environment()
    forecast = load_team_forecast()
    coaching = load_coaching()
    player = load_player_opportunity_summary()

    # Canonical base is authoritative team_game_stats.
    out = actual.copy()

    out = out.merge(
        pregame,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        forecast,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        coaching,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        player,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )

    validate_identity_consistency(out)

    out = add_objective_deltas(out)

    out.insert(
        0,
        "version",
        VERSION,
    )

    out["production_influence"] = (
        PRODUCTION_INFLUENCE
    )
    out["solver_influence"] = (
        SOLVER_INFLUENCE
    )
    out["forecast_mutation"] = (
        FORECAST_MUTATION
    )
    out["player_projection_mutation"] = (
        PLAYER_PROJECTION_MUTATION
    )
    out["persistent_coach_prior_mutation"] = (
        PERSISTENT_COACH_PRIOR_MUTATION
    )

    # Evidence-family availability.
    out["has_actual_team_stats"] = 1

    out["has_pregame_environment"] = (
        out["pregame_history_games"].notna()
    ).astype(int)

    out["has_forecast_evidence"] = (
        out["forecast_snapshot_id"].notna()
    ).astype(int)

    out["has_coaching_evidence"] = (
        out["coaching_interpretation"].notna()
    ).astype(int)

    out["has_expected_pass_rate"] = (
        out["expected_pass_rate"].notna()
    ).astype(int)

    out["has_player_opportunity_evidence"] = (
        out["starter_usage_count"].notna()
    ).astype(int)

    assert_unique(
        out,
        ["game_id", "team"],
        "final learning evidence",
    )

    return out


def print_validation(
    df: pd.DataFrame,
) -> None:
    print("\n=== POSTGAME TEAM LEARNING EVIDENCE V1 ===")
    print("OUTPUT:", OUTPUT_FILE)

    print("\n=== KEY INTEGRITY ===")
    print("rows:", len(df))
    print(
        "unique_game_team:",
        df[["game_id", "team"]]
        .drop_duplicates()
        .shape[0],
    )

    dupes = int(
        df.duplicated(
            ["game_id", "team"]
        ).sum()
    )
    print("duplicate_game_team:", dupes)

    print("\n=== COVERAGE ===")
    print(
        "season_min:",
        df["season"].min(),
    )
    print(
        "season_max:",
        df["season"].max(),
    )
    print(
        "week_min:",
        df["week"].min(),
    )
    print(
        "week_max:",
        df["week"].max(),
    )

    for c in [
        "has_actual_team_stats",
        "has_pregame_environment",
        "has_forecast_evidence",
        "has_coaching_evidence",
        "has_expected_pass_rate",
        "has_player_opportunity_evidence",
    ]:
        print(
            f"{c}:",
            int(df[c].sum()),
            "/",
            len(df),
        )

    print("\n=== MAJOR FIELD MISSINGNESS ===")

    major = [
        "team_score",
        "opponent_score",
        "predicted_team_points",
        "actual_pass_attempts",
        "actual_rush_attempts",
        "actual_passing_yards",
        "actual_rushing_yards",
        "actual_passing_epa",
        "actual_rushing_epa",
        "pregame_pass_rate_avg_3",
        "pregame_rush_rate_avg_3",
        "expected_pass_rate",
        "final_pass_rate",
        "coaching_confidence",
        "starter_usage_count",
    ]

    for c in major:
        if c in df.columns:
            print(
                f"{c}:",
                int(df[c].isna().sum()),
            )

    print("\n=== SAFETY FLAGS ===")
    for c in [
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

    print("\n=== 2026 REPRESENTATIVE ROWS ===")

    show = [
        "game_id",
        "team",
        "opponent_team",
        "team_score",
        "opponent_score",
        "actual_win",
        "predicted_team_points",
        "predicted_opponent_points",
        "winner_prediction_correct",
        "pregame_pass_rate_avg_3",
        "actual_pass_rate",
        "pass_rate_vs_pregame_avg3",
        "expected_pass_rate",
        "actual_vs_expected_pass_rate",
        "actual_pass_attempts",
        "actual_rush_attempts",
        "actual_passing_yards",
        "actual_rushing_yards",
        "actual_passing_tds",
        "actual_rushing_tds",
        "actual_passing_epa",
        "actual_rushing_epa",
        "coaching_interpretation",
        "rb_carries",
        "rb_targets",
        "wr_targets",
        "te_targets",
    ]

    show = [
        c for c in show
        if c in df.columns
    ]

    sample = (
        df.loc[
            df["season"] == 2026,
            show,
        ]
        .sort_values(
            ["game_id", "team"]
        )
        .head(10)
    )

    if sample.empty:
        print("NO 2026 ROWS")
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

    # Builder creates only its dedicated shadow artifact.
    # Existing source files/databases are opened read-only.
    df = build()

    print_validation(df)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== WRITE COMPLETE ===")
    print(OUTPUT_FILE)
    print(
        "production_influence=False"
    )
    print(
        "solver_influence=False"
    )
    print(
        "forecast_mutation=False"
    )
    print(
        "player_projection_mutation=False"
    )
    print(
        "persistent_coach_prior_mutation=False"
    )


if __name__ == "__main__":
    main()
