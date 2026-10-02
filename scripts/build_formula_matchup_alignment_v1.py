#!/usr/bin/env python3

"""
WFS Formula Matchup Alignment V1

Purpose
-------
Connect a team's strictly prior demonstrated offensive formula evidence
with the CURRENT opponent's strictly pregame/as-of defensive environment.

This answers questions such as:

- Has this offense recently demonstrated effective rushing?
- Has this offense recently demonstrated effective passing?
- Does the upcoming opponent's pregame environment indicate elevated
  rushing or passing production allowed?
- Do those two evidence families align?

This artifact does NOT claim:
- causation
- predictive validity
- production authorization
- solver authorization

Current-game outcomes are retained ONLY as validation targets and are
never used to construct matchup alignment features.

Inputs
------
processed/team_formula_learning_v1.parquet
data/nfl.db -> team_pregame_environment
data/nfl.db -> team_position_dvp

Output
------
processed/formula_matchup_alignment_v1.parquet

Canonical grain
---------------
Exactly one row per game_id + team.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FORMULA_FILE = (
    ROOT
    / "processed"
    / "team_formula_learning_v1.parquet"
)

NFL_DB = ROOT / "data" / "nfl.db"

OUTPUT_FILE = (
    ROOT
    / "processed"
    / "formula_matchup_alignment_v1.parquet"
)

VERSION = "WFS_FORMULA_MATCHUP_ALIGNMENT_V1"


PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
FORECAST_MUTATION = False
PLAYER_PROJECTION_MUTATION = False
PERSISTENT_COACH_PRIOR_MUTATION = False


def ro_connection(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)

    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


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


def safe_mean(values: list[object]) -> float:
    vals = []

    for value in values:
        if pd.notna(value):
            vals.append(float(value))

    if not vals:
        return np.nan

    return float(np.mean(vals))


def relative_to_league(
    series: pd.Series,
    group_season: pd.Series,
    group_week: pd.Series,
) -> pd.Series:
    """
    Compare each pregame opponent metric to the league mean for the same
    season/week.

    Inputs themselves are already pregame/as-of features.

    Positive result:
        opponent allowed more than league average entering that game.

    Negative result:
        opponent allowed less than league average entering that game.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    league_mean = (
        pd.DataFrame(
            {
                "season": group_season,
                "week": group_week,
                "value": numeric,
            }
        )
        .groupby(
            ["season", "week"],
            dropna=False,
        )["value"]
        .transform("mean")
    )

    return numeric - league_mean


def direction_label(
    value: object,
    positive_label: str,
    negative_label: str,
    neutral_label: str,
) -> object:
    if pd.isna(value):
        return pd.NA

    value = float(value)

    if value > 0:
        return positive_label

    if value < 0:
        return negative_label

    return neutral_label


def alignment_label(
    formula_score: object,
    vulnerability_score: object,
    phase: str,
) -> object:
    """
    Direction-only V1 classification.

    No arbitrary magnitude threshold.

    HIGHER_ALIGNMENT:
        offense has stronger recent evidence in this phase AND opponent
        vulnerability is above league average.

    LOWER_ALIGNMENT:
        one or both directional components are unfavorable.

    MIXED_ALIGNMENT:
        directional evidence conflicts.

    This is descriptive only and not a prediction.
    """

    if pd.isna(formula_score) or pd.isna(vulnerability_score):
        return pd.NA

    f = float(formula_score)
    v = float(vulnerability_score)

    if f > 0 and v > 0:
        return f"{phase}_FORMULA_VULNERABILITY_ALIGNED"

    if f < 0 and v < 0:
        return f"{phase}_FORMULA_VULNERABILITY_UNALIGNED"

    if f == 0 or v == 0:
        return f"{phase}_FORMULA_VULNERABILITY_NEUTRAL"

    return f"{phase}_FORMULA_VULNERABILITY_MIXED"


def load_pregame_environment(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    query = """
        SELECT
            game_id,
            season,
            week,
            team,
            opponent_team,

            opponent_points_allowed_avg_3,
            opponent_points_allowed_avg_5,

            opponent_pass_yards_allowed_avg_3,
            opponent_rush_yards_allowed_avg_3,

            opponent_pass_tds_allowed_avg_3,
            opponent_rush_tds_allowed_avg_3

        FROM team_pregame_environment
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    assert_unique(
        df,
        ["game_id", "team"],
        "team_pregame_environment",
    )

    return df


def load_position_dvp(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    query = """
        SELECT
            game_id,
            season,
            week,
            defense_team,
            position,

            history_games,

            fd_allowed_avg_3,
            targets_allowed_avg_3,
            carries_allowed_avg_3,
            receptions_allowed_avg_3,
            receiving_yards_allowed_avg_3,
            rushing_yards_allowed_avg_3,
            receiving_tds_allowed_avg_3,
            rushing_tds_allowed_avg_3,
            opportunities_allowed_avg_3

        FROM team_position_dvp
        WHERE position IN ('QB', 'RB', 'WR', 'TE')
    """

    raw = pd.read_sql_query(
        query,
        conn,
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team",
            ]
        )

    # Each team row wants the DvP of its opponent defense.
    raw = raw.rename(
        columns={
            "defense_team": "opponent_team",
        }
    )

    # Pivot position rows into one offensive-team matchup row.
    value_cols = [
        "history_games",
        "fd_allowed_avg_3",
        "targets_allowed_avg_3",
        "carries_allowed_avg_3",
        "receptions_allowed_avg_3",
        "receiving_yards_allowed_avg_3",
        "rushing_yards_allowed_avg_3",
        "receiving_tds_allowed_avg_3",
        "rushing_tds_allowed_avg_3",
        "opportunities_allowed_avg_3",
    ]

    parts = []

    for position in ["QB", "RB", "WR", "TE"]:
        p = raw[
            raw["position"] == position
        ].copy()

        prefix = position.lower()

        keep = [
            "game_id",
            "season",
            "week",
            "opponent_team",
        ] + value_cols

        p = p[keep]

        rename = {
            c: f"{prefix}_{c}"
            for c in value_cols
        }

        p = p.rename(
            columns=rename
        )

        assert_unique(
            p,
            [
                "game_id",
                "opponent_team",
            ],
            f"team_position_dvp_{position}",
        )

        parts.append(p)

    # Merge the four position surfaces on game/opponent defense.
    dvp = parts[0]

    for p in parts[1:]:
        dvp = dvp.merge(
            p,
            on=[
                "game_id",
                "season",
                "week",
                "opponent_team",
            ],
            how="outer",
            validate="one_to_one",
        )

    return dvp


def build() -> pd.DataFrame:
    if not FORMULA_FILE.exists():
        raise FileNotFoundError(FORMULA_FILE)

    formula = pd.read_parquet(
        FORMULA_FILE
    )

    required_formula = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",

        "prior_3_more_run_game_rate",
        "prior_3_more_pass_game_rate",

        "prior_3_positive_rush_epa_game_rate",
        "prior_3_positive_pass_epa_game_rate",

        "prior_3_run_plus_positive_rush_epa_rate",
        "prior_3_run_plus_above_rush_yards_rate",
        "prior_3_run_plus_above_rush_tds_rate",

        "prior_3_pass_plus_positive_pass_epa_rate",
        "prior_3_pass_plus_above_pass_yards_rate",
        "prior_3_pass_plus_above_pass_tds_rate",

        "prior_3_actual_rushing_epa_avg",
        "prior_3_actual_passing_epa_avg",

        "formula_history_games",
        "formula_history_tier",

        "current_approach_pass_rate",
        "current_pass_efficiency",
        "current_rush_efficiency",
        "current_scoring_production",
        "current_game_result",

        "current_actual_passing_epa",
        "current_actual_rushing_epa",

        "current_passing_yards_delta_vs_baseline",
        "current_rushing_yards_delta_vs_baseline",

        "current_passing_tds_delta_vs_baseline",
        "current_rushing_tds_delta_vs_baseline",
    ]

    require_columns(
        formula,
        required_formula,
        "team formula learning",
    )

    assert_unique(
        formula,
        ["game_id", "team"],
        "team formula learning",
    )

    with ro_connection(NFL_DB) as conn:
        env = load_pregame_environment(
            conn
        )

        dvp = load_position_dvp(
            conn
        )

    # ----------------------------------------------------------
    # Join canonical pregame environment
    # ----------------------------------------------------------

    env = env.rename(
        columns={
            "opponent_team":
                "environment_opponent_team",
        }
    )

    out = formula.merge(
        env,
        on=[
            "game_id",
            "season",
            "week",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    mismatch = (
        out[
            "environment_opponent_team"
        ].notna()
        &
        (
            out[
                "environment_opponent_team"
            ]
            !=
            out["opponent_team"]
        )
    )

    if mismatch.any():
        sample = out.loc[
            mismatch,
            [
                "game_id",
                "team",
                "opponent_team",
                "environment_opponent_team",
            ],
        ].head(20)

        raise RuntimeError(
            "Opponent identity mismatch in "
            "team_pregame_environment:\n"
            + sample.to_string(index=False)
        )

    # ----------------------------------------------------------
    # Join opponent position DvP
    # ----------------------------------------------------------

    if not dvp.empty:
        out = out.merge(
            dvp,
            on=[
                "game_id",
                "season",
                "week",
                "opponent_team",
            ],
            how="left",
            validate="one_to_one",
        )

    # ----------------------------------------------------------
    # Opponent vulnerability relative to same-week league
    # ----------------------------------------------------------

    out[
        "opponent_pass_yards_allowed_vs_league"
    ] = relative_to_league(
        out[
            "opponent_pass_yards_allowed_avg_3"
        ],
        out["season"],
        out["week"],
    )

    out[
        "opponent_rush_yards_allowed_vs_league"
    ] = relative_to_league(
        out[
            "opponent_rush_yards_allowed_avg_3"
        ],
        out["season"],
        out["week"],
    )

    out[
        "opponent_pass_tds_allowed_vs_league"
    ] = relative_to_league(
        out[
            "opponent_pass_tds_allowed_avg_3"
        ],
        out["season"],
        out["week"],
    )

    out[
        "opponent_rush_tds_allowed_vs_league"
    ] = relative_to_league(
        out[
            "opponent_rush_tds_allowed_avg_3"
        ],
        out["season"],
        out["week"],
    )

    # Position-level FD vulnerability.
    for pos in [
        "qb",
        "rb",
        "wr",
        "te",
    ]:
        c = (
            f"{pos}_"
            "fd_allowed_avg_3"
        )

        if c in out.columns:
            out[
                f"{pos}_fd_allowed_vs_league"
            ] = relative_to_league(
                out[c],
                out["season"],
                out["week"],
            )

    # ----------------------------------------------------------
    # Team demonstrated formula scores
    #
    # V1 uses simple transparent means of prior evidence.
    # No fitted weights.
    # ----------------------------------------------------------

    out["run_formula_score"] = out.apply(
        lambda r: safe_mean(
            [
                r[
                    "prior_3_more_run_game_rate"
                ],
                r[
                    "prior_3_positive_rush_epa_game_rate"
                ],
                r[
                    "prior_3_run_plus_positive_rush_epa_rate"
                ],
                r[
                    "prior_3_run_plus_above_rush_yards_rate"
                ],
                r[
                    "prior_3_run_plus_above_rush_tds_rate"
                ],
            ]
        ),
        axis=1,
    )

    out["pass_formula_score"] = out.apply(
        lambda r: safe_mean(
            [
                r[
                    "prior_3_more_pass_game_rate"
                ],
                r[
                    "prior_3_positive_pass_epa_game_rate"
                ],
                r[
                    "prior_3_pass_plus_positive_pass_epa_rate"
                ],
                r[
                    "prior_3_pass_plus_above_pass_yards_rate"
                ],
                r[
                    "prior_3_pass_plus_above_pass_tds_rate"
                ],
            ]
        ),
        axis=1,
    )

    # Center rate-based formula evidence around 0.5.
    #
    # > 0 = stronger-than-even recent evidence
    # < 0 = weaker-than-even recent evidence
    out[
        "run_formula_direction_score"
    ] = (
        out["run_formula_score"] - 0.5
    )

    out[
        "pass_formula_direction_score"
    ] = (
        out["pass_formula_score"] - 0.5
    )

    # ----------------------------------------------------------
    # Opponent vulnerability composite scores
    #
    # Still transparent, equal-weight, direction-only research.
    # ----------------------------------------------------------

    out[
        "run_vulnerability_score"
    ] = out.apply(
        lambda r: safe_mean(
            [
                r.get(
                    "opponent_rush_yards_allowed_vs_league"
                ),
                r.get(
                    "opponent_rush_tds_allowed_vs_league"
                ),
                r.get(
                    "rb_fd_allowed_vs_league"
                ),
            ]
        ),
        axis=1,
    )

    out[
        "pass_vulnerability_score"
    ] = out.apply(
        lambda r: safe_mean(
            [
                r.get(
                    "opponent_pass_yards_allowed_vs_league"
                ),
                r.get(
                    "opponent_pass_tds_allowed_vs_league"
                ),
                r.get(
                    "qb_fd_allowed_vs_league"
                ),
                r.get(
                    "wr_fd_allowed_vs_league"
                ),
                r.get(
                    "te_fd_allowed_vs_league"
                ),
            ]
        ),
        axis=1,
    )

    # ----------------------------------------------------------
    # Direction labels
    # ----------------------------------------------------------

    out["run_formula_direction"] = [
        direction_label(
            x,
            "STRONGER_RUN_FORMULA_EVIDENCE",
            "WEAKER_RUN_FORMULA_EVIDENCE",
            "NEUTRAL_RUN_FORMULA_EVIDENCE",
        )
        for x in out[
            "run_formula_direction_score"
        ]
    ]

    out["pass_formula_direction"] = [
        direction_label(
            x,
            "STRONGER_PASS_FORMULA_EVIDENCE",
            "WEAKER_PASS_FORMULA_EVIDENCE",
            "NEUTRAL_PASS_FORMULA_EVIDENCE",
        )
        for x in out[
            "pass_formula_direction_score"
        ]
    ]

    out[
        "run_opponent_vulnerability"
    ] = [
        direction_label(
            x,
            "ABOVE_LEAGUE_RUN_VULNERABILITY",
            "BELOW_LEAGUE_RUN_VULNERABILITY",
            "LEAGUE_AVERAGE_RUN_VULNERABILITY",
        )
        for x in out[
            "run_vulnerability_score"
        ]
    ]

    out[
        "pass_opponent_vulnerability"
    ] = [
        direction_label(
            x,
            "ABOVE_LEAGUE_PASS_VULNERABILITY",
            "BELOW_LEAGUE_PASS_VULNERABILITY",
            "LEAGUE_AVERAGE_PASS_VULNERABILITY",
        )
        for x in out[
            "pass_vulnerability_score"
        ]
    ]

    # ----------------------------------------------------------
    # Alignment labels
    # ----------------------------------------------------------

    out["run_matchup_alignment"] = [
        alignment_label(
            f,
            v,
            "RUN",
        )
        for f, v in zip(
            out[
                "run_formula_direction_score"
            ],
            out[
                "run_vulnerability_score"
            ],
        )
    ]

    out["pass_matchup_alignment"] = [
        alignment_label(
            f,
            v,
            "PASS",
        )
        for f, v in zip(
            out[
                "pass_formula_direction_score"
            ],
            out[
                "pass_vulnerability_score"
            ],
        )
    ]

    # ----------------------------------------------------------
    # Evidence availability
    # ----------------------------------------------------------

    out[
        "has_formula_history"
    ] = (
        pd.to_numeric(
            out["formula_history_games"],
            errors="coerce",
        )
        > 0
    ).astype(int)

    out[
        "has_pregame_matchup_environment"
    ] = (
        out[
            "environment_opponent_team"
        ].notna()
    ).astype(int)

    out[
        "has_run_matchup_evidence"
    ] = (
        out[
            "run_formula_direction_score"
        ].notna()
        &
        out[
            "run_vulnerability_score"
        ].notna()
    ).astype(int)

    out[
        "has_pass_matchup_evidence"
    ] = (
        out[
            "pass_formula_direction_score"
        ].notna()
        &
        out[
            "pass_vulnerability_score"
        ].notna()
    ).astype(int)

    # ----------------------------------------------------------
    # Explicit leakage / safety contract
    # ----------------------------------------------------------

    out[
        "current_game_actuals_used_in_alignment"
    ] = False

    out[
        "future_game_used_in_alignment"
    ] = False

    out["causal_claim_allowed"] = False

    out[
        "production_influence"
    ] = PRODUCTION_INFLUENCE

    out[
        "solver_influence"
    ] = SOLVER_INFLUENCE

    out[
        "forecast_mutation"
    ] = FORECAST_MUTATION

    out[
        "player_projection_mutation"
    ] = PLAYER_PROJECTION_MUTATION

    out[
        "persistent_coach_prior_mutation"
    ] = PERSISTENT_COACH_PRIOR_MUTATION

    assert_unique(
        out,
        ["game_id", "team"],
        "formula matchup alignment",
    )

    return out


def print_validation(
    df: pd.DataFrame,
) -> None:

    print(
        "=== FORMULA MATCHUP ALIGNMENT V1 ==="
    )

    print(
        "OUTPUT:",
        OUTPUT_FILE,
    )

    print("\n=== KEY INTEGRITY ===")

    print(
        "rows:",
        len(df),
    )

    print(
        "unique_game_team:",
        df[
            ["game_id", "team"]
        ]
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

    for c in [
        "has_formula_history",
        "has_pregame_matchup_environment",
        "has_run_matchup_evidence",
        "has_pass_matchup_evidence",
    ]:
        print(
            c,
            int(df[c].sum()),
            "/",
            len(df),
        )

    print(
        "\n=== RUN ALIGNMENT DISTRIBUTION ==="
    )

    print(
        df[
            "run_matchup_alignment"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== PASS ALIGNMENT DISTRIBUTION ==="
    )

    print(
        df[
            "pass_matchup_alignment"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\n=== SAFETY FLAGS ===")

    for c in [
        "current_game_actuals_used_in_alignment",
        "future_game_used_in_alignment",
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

    print(
        "\n=== NO-PRIOR ALIGNMENT CHECK ==="
    )

    no_prior = df[
        df["formula_history_games"] == 0
    ]

    bad = no_prior[
        no_prior[
            "run_matchup_alignment"
        ].notna()
        |
        no_prior[
            "pass_matchup_alignment"
        ].notna()
    ]

    print(
        "no_prior_rows:",
        len(no_prior),
    )

    print(
        "no_prior_rows_with_alignment:",
        len(bad),
    )

    print(
        "\n=== 2026 REPRESENTATIVE ROWS ==="
    )

    cols = [
        "game_id",
        "team",
        "opponent_team",

        "formula_history_games",
        "formula_history_tier",

        "run_formula_score",
        "run_formula_direction",
        "run_vulnerability_score",
        "run_opponent_vulnerability",
        "run_matchup_alignment",

        "pass_formula_score",
        "pass_formula_direction",
        "pass_vulnerability_score",
        "pass_opponent_vulnerability",
        "pass_matchup_alignment",

        "opponent_rush_yards_allowed_avg_3",
        "opponent_rush_tds_allowed_avg_3",
        "rb_fd_allowed_avg_3",

        "opponent_pass_yards_allowed_avg_3",
        "opponent_pass_tds_allowed_avg_3",
        "qb_fd_allowed_avg_3",
        "wr_fd_allowed_avg_3",
        "te_fd_allowed_avg_3",

        # Current-game validation targets only.
        "current_actual_rushing_epa",
        "current_rushing_yards_delta_vs_baseline",
        "current_rushing_tds_delta_vs_baseline",

        "current_actual_passing_epa",
        "current_passing_yards_delta_vs_baseline",
        "current_passing_tds_delta_vs_baseline",

        "current_game_result",
    ]

    cols = [
        c
        for c in cols
        if c in df.columns
    ]

    sample = (
        df[
            df["season"] == 2026
        ][cols]
        .sort_values(
            [
                "game_id",
                "team",
            ],
            kind="stable",
        )
        .head(16)
    )

    if sample.empty:
        print(
            "NO 2026 ROWS"
        )
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

    print(
        "\n=== WRITE COMPLETE ==="
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nSHADOW CONTRACT:"
    )

    print(
        "current_game_actuals_used_in_alignment=False"
    )

    print(
        "future_game_used_in_alignment=False"
    )

    print(
        "causal_claim_allowed=False"
    )

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
