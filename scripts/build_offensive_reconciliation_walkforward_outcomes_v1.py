#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PLAYER_INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_opportunity_blend_v1.csv"
)

TEAM_INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v2.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_outcomes_v1.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_outcomes_summary_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_outcomes_v1_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(0.0)


def distribute(
    total: float,
    weights: pd.Series,
) -> pd.Series:
    clean = pd.to_numeric(
        weights,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    if clean.sum() <= 0:
        return pd.Series(
            np.zeros(len(clean)),
            index=clean.index,
            dtype=float,
        )

    return float(total) * clean / clean.sum()


def distribute_with_caps(
    total: float,
    weights: pd.Series,
    caps: pd.Series,
) -> pd.Series:
    result = pd.Series(
        0.0,
        index=weights.index,
        dtype=float,
    )

    remaining = max(0.0, float(total))

    clean_weights = pd.to_numeric(
        weights,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    clean_caps = pd.to_numeric(
        caps,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    available = list(weights.index)

    while remaining > 1e-12 and available:
        current_weights = clean_weights.loc[
            available
        ]

        if current_weights.sum() <= 0:
            current_weights = pd.Series(
                np.ones(len(available)),
                index=available,
                dtype=float,
            )

        proposed = (
            remaining
            * current_weights
            / current_weights.sum()
        )

        capped = False

        for index in list(available):
            room = (
                clean_caps.loc[index]
                - result.loc[index]
            )

            if proposed.loc[index] >= room:
                allocation = max(0.0, room)
                result.loc[index] += allocation
                remaining -= allocation
                available.remove(index)
                capped = True

        if not capped:
            result.loc[available] += proposed
            remaining = 0.0

    return result


def bayesian_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    history_games: pd.Series,
    position: pd.Series,
    priors: dict[str, float],
    pseudo_opportunities: float,
    lower: float,
    upper: float,
) -> pd.Series:
    raw_rate = (
        numerator
        / denominator.replace(0, np.nan)
    )

    prior = (
        position.map(priors)
        .fillna(
            float(
                np.mean(
                    list(priors.values())
                )
            )
        )
    )

    reliability = (
        denominator
        / (
            denominator
            + pseudo_opportunities
        )
    ).clip(lower=0.0, upper=1.0)

    # History count prevents one-game efficiency from
    # receiving established-player authority.
    game_reliability = (
        history_games
        / (history_games + 5.0)
    ).clip(lower=0.0, upper=1.0)

    reliability = np.minimum(
        reliability,
        game_reliability,
    )

    result = (
        reliability
        * raw_rate.fillna(prior)
        + (
            1.0 - reliability
        )
        * prior
    )

    return result.clip(
        lower=lower,
        upper=upper,
    )


def fanduel_points(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.Series:
    return (
        series(
            frame,
            f"{prefix}_passing_yards",
        ) * 0.04
        + series(
            frame,
            f"{prefix}_passing_tds",
        ) * 4.0
        - series(
            frame,
            f"{prefix}_interceptions",
        )
        + series(
            frame,
            f"{prefix}_rushing_yards",
        ) * 0.10
        + series(
            frame,
            f"{prefix}_rushing_tds",
        ) * 6.0
        + series(
            frame,
            f"{prefix}_receiving_yards",
        ) * 0.10
        + series(
            frame,
            f"{prefix}_receiving_tds",
        ) * 6.0
        + series(
            frame,
            f"{prefix}_receptions",
        ) * 0.50
    )


def main() -> None:
    for required in (
        PLAYER_INPUT,
        TEAM_INPUT,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    players = pd.read_csv(PLAYER_INPUT)
    teams = pd.read_csv(TEAM_INPUT)

    team_columns = [
        "game_id",
        "team",
        "adaptive_attempts",
        "adaptive_completions",
        "adaptive_passing_yards",
        "adaptive_passing_tds",
        "adaptive_carries",
        "adaptive_rushing_yards",
        "adaptive_rushing_tds",
    ]

    players = players.merge(
        teams[team_columns],
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_team"),
    )

    players["position"] = (
        players["position"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    history_games = series(
        players,
        "history_games",
    )

    target_avg = (
        0.60 * series(
            players,
            "targets_avg_3",
        )
        + 0.40 * series(
            players,
            "targets_avg_5",
        )
    )

    carry_avg = (
        0.60 * series(
            players,
            "carries_avg_3",
        )
        + 0.40 * series(
            players,
            "carries_avg_5",
        )
    )

    reception_avg = (
        0.60 * series(
            players,
            "receptions_avg_3",
        )
        + 0.40 * series(
            players,
            "receptions_avg_5",
        )
    )

    rushing_yard_avg = (
        0.60 * series(
            players,
            "rushing_yards_avg_3",
        )
        + 0.40 * series(
            players,
            "rushing_yards_avg_5",
        )
    )

    receiving_yard_avg = (
        0.60 * series(
            players,
            "receiving_yards_avg_3",
        )
        + 0.40 * series(
            players,
            "receiving_yards_avg_5",
        )
    )

    rushing_td_avg = (
        0.60 * series(
            players,
            "rushing_tds_avg_3",
        )
        + 0.40 * series(
            players,
            "rushing_tds_avg_5",
        )
    )

    receiving_td_avg = (
        0.60 * series(
            players,
            "receiving_tds_avg_3",
        )
        + 0.40 * series(
            players,
            "receiving_tds_avg_5",
        )
    )

    players["catch_rate"] = bayesian_rate(
        reception_avg,
        target_avg,
        history_games,
        players["position"],
        {
            "RB": 0.72,
            "WR": 0.62,
            "TE": 0.66,
            "QB": 0.00,
        },
        15.0,
        0.20,
        0.92,
    )

    players["yards_per_carry"] = bayesian_rate(
        rushing_yard_avg,
        carry_avg,
        history_games,
        players["position"],
        {
            "QB": 5.0,
            "RB": 4.2,
            "WR": 6.0,
            "TE": 3.0,
        },
        25.0,
        1.5,
        9.0,
    )

    players["yards_per_reception"] = bayesian_rate(
        receiving_yard_avg,
        reception_avg,
        history_games,
        players["position"],
        {
            "RB": 7.5,
            "WR": 12.5,
            "TE": 10.5,
            "QB": 0.0,
        },
        20.0,
        3.0,
        25.0,
    )

    players["rush_td_rate"] = bayesian_rate(
        rushing_td_avg,
        carry_avg,
        history_games,
        players["position"],
        {
            "QB": 0.035,
            "RB": 0.035,
            "WR": 0.015,
            "TE": 0.010,
        },
        20.0,
        0.0,
        0.20,
    )

    players["receiving_td_rate"] = bayesian_rate(
        receiving_td_avg,
        target_avg,
        history_games,
        players["position"],
        {
            "RB": 0.030,
            "WR": 0.050,
            "TE": 0.055,
            "QB": 0.0,
        },
        15.0,
        0.0,
        0.20,
    )

    output_blocks = []
    conservation_rows = []

    for (
        game_id,
        team,
    ), block in players.groupby(
        ["game_id", "team"],
        sort=False,
    ):
        block = block.copy()

        primary = block[
            "primary_qb_flag"
        ].eq(1)

        if primary.sum() != 1:
            raise RuntimeError(
                f"Primary QB contract failed: "
                f"{game_id} {team}"
            )

        attempts = float(
            block[
                "adaptive_attempts"
            ].iloc[0]
        )

        completions = min(
            attempts,
            float(
                block[
                    "adaptive_completions"
                ].iloc[0]
            ),
        )

        passing_yards = float(
            block[
                "adaptive_passing_yards"
            ].iloc[0]
        )

        passing_tds = float(
            block[
                "adaptive_passing_tds"
            ].iloc[0]
        )

        rushing_yards = float(
            block[
                "adaptive_rushing_yards"
            ].iloc[0]
        )

        rushing_tds = float(
            block[
                "adaptive_rushing_tds"
            ].iloc[0]
        )

        for model in ("v1", "hybrid"):
            carries = series(
                block,
                f"{model}_carries",
            )

            targets = series(
                block,
                f"{model}_targets",
            )

            catch_weights = (
                targets
                * series(
                    block,
                    "catch_rate",
                )
            )

            receptions = distribute_with_caps(
                completions,
                catch_weights,
                targets,
            )

            rush_yard_weights = (
                carries
                * series(
                    block,
                    "yards_per_carry",
                )
            )

            receiving_yard_weights = (
                receptions
                * series(
                    block,
                    "yards_per_reception",
                )
            )

            rush_td_weights = (
                carries
                * series(
                    block,
                    "rush_td_rate",
                )
            )

            receiving_td_weights = (
                targets
                * series(
                    block,
                    "receiving_td_rate",
                )
            )

            rush_td_weights.loc[
                carries.lt(0.25)
            ] = 0.0

            receiving_td_weights.loc[
                targets.lt(0.25)
            ] = 0.0

            block[
                f"{model}_receptions"
            ] = receptions

            block[
                f"{model}_rushing_yards"
            ] = distribute(
                rushing_yards,
                rush_yard_weights,
            )

            block[
                f"{model}_receiving_yards"
            ] = distribute(
                passing_yards,
                receiving_yard_weights,
            )

            block[
                f"{model}_rushing_tds"
            ] = distribute_with_caps(
                rushing_tds,
                rush_td_weights,
                carries,
            )

            block[
                f"{model}_receiving_tds"
            ] = distribute_with_caps(
                passing_tds,
                receiving_td_weights,
                targets,
            )

            block[
                f"{model}_attempts"
            ] = 0.0

            block[
                f"{model}_completions"
            ] = 0.0

            block[
                f"{model}_passing_yards"
            ] = 0.0

            block[
                f"{model}_passing_tds"
            ] = 0.0

            block[
                f"{model}_interceptions"
            ] = 0.0

            block.loc[
                primary,
                f"{model}_attempts",
            ] = attempts

            block.loc[
                primary,
                f"{model}_completions",
            ] = completions

            block.loc[
                primary,
                f"{model}_passing_yards",
            ] = passing_yards

            block.loc[
                primary,
                f"{model}_passing_tds",
            ] = passing_tds

            qb_attempt_history = (
                0.60
                * series(
                    block,
                    "attempts_avg_3",
                )
                + 0.40
                * series(
                    block,
                    "attempts_avg_5",
                )
            )

            qb_interception_history = (
                0.60
                * series(
                    block,
                    "passing_interceptions_avg_3",
                )
                + 0.40
                * series(
                    block,
                    "passing_interceptions_avg_5",
                )
            )

            interception_rate = (
                qb_interception_history
                / qb_attempt_history.replace(
                    0,
                    np.nan,
                )
            ).fillna(0.025).clip(
                lower=0.005,
                upper=0.08,
            )

            block.loc[
                primary,
                f"{model}_interceptions",
            ] = (
                attempts
                * interception_rate.loc[
                    primary
                ]
            )

        conservation_rows.append({
            "game_id": game_id,
            "team": team,
            "completion_gap_v1":
                completions
                - block[
                    "v1_receptions"
                ].sum(),
            "completion_gap_hybrid":
                completions
                - block[
                    "hybrid_receptions"
                ].sum(),
            "passing_yard_gap_v1":
                passing_yards
                - block[
                    "v1_receiving_yards"
                ].sum(),
            "passing_yard_gap_hybrid":
                passing_yards
                - block[
                    "hybrid_receiving_yards"
                ].sum(),
            "passing_td_gap_v1":
                passing_tds
                - block[
                    "v1_receiving_tds"
                ].sum(),
            "passing_td_gap_hybrid":
                passing_tds
                - block[
                    "hybrid_receiving_tds"
                ].sum(),
            "rushing_yard_gap_v1":
                rushing_yards
                - block[
                    "v1_rushing_yards"
                ].sum(),
            "rushing_yard_gap_hybrid":
                rushing_yards
                - block[
                    "hybrid_rushing_yards"
                ].sum(),
            "rushing_td_gap_v1":
                rushing_tds
                - block[
                    "v1_rushing_tds"
                ].sum(),
            "rushing_td_gap_hybrid":
                rushing_tds
                - block[
                    "hybrid_rushing_tds"
                ].sum(),
        })

        output_blocks.append(block)

    result = pd.concat(
        output_blocks,
        ignore_index=True,
    )

    for model in ("v1", "hybrid"):
        result[
            f"{model}_fd_points"
        ] = fanduel_points(
            result,
            model,
        )

    result["actual_fd_points"] = series(
        result,
        "actual_fanduel_points",
    )

    conservation = pd.DataFrame(
        conservation_rows
    )

    gap_columns = [
        column
        for column in conservation.columns
        if column.endswith("_v1")
        or column.endswith("_hybrid")
    ]

    max_gap = float(
        conservation[
            gap_columns
        ].abs().max().max()
    )

    evaluation_mask = (
        (
            result["season"].eq(2025)
            & result["week"].between(14, 18)
        )
        | (
            result["season"].eq(2026)
            & result["week"].eq(1)
        )
    )

    summary_rows = []

    evaluation = result[
        evaluation_mask
    ]

    metrics = [
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "fd_points",
    ]

    for history_class, block in [
        ("ALL", evaluation),
        *[
            (str(name), group)
            for name, group
            in evaluation.groupby(
                "history_class"
            )
        ],
    ]:
        for metric in metrics:
            actual_column = (
                "actual_fd_points"
                if metric == "fd_points"
                else f"actual_{metric}"
            )

            actual = series(
                block,
                actual_column,
            )

            for model in ("v1", "hybrid"):
                predicted = series(
                    block,
                    f"{model}_{metric}",
                )

                error = predicted - actual

                summary_rows.append({
                    "history_class":
                        history_class,
                    "metric": metric,
                    "model": model.upper(),
                    "rows": int(len(block)),
                    "mae": float(
                        error.abs().mean()
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                np.square(error)
                            )
                        )
                    ),
                    "bias": float(
                        error.mean()
                    ),
                })

    summary = pd.DataFrame(
        summary_rows
    )

    overall = summary[
        summary["history_class"].eq("ALL")
    ].pivot(
        index="metric",
        columns="model",
        values="mae",
    )

    overall["HYBRID_VS_V1"] = (
        overall["V1"]
        - overall["HYBRID"]
    )

    duplicate_keys = int(
        result.duplicated(
            ["game_id", "team", "player_id"]
        ).sum()
    )

    projection_columns = [
        column
        for column in result.columns
        if column.startswith("v1_")
        or column.startswith("hybrid_")
    ]

    numeric_projection = result[
        projection_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    hard_failures = {
        "duplicate_player_keys":
            duplicate_keys,
        "conservation_failure":
            int(max_gap > 1e-8),
        "nonfinite_predictions":
            int(
                (~np.isfinite(
                    numeric_projection.to_numpy(
                        dtype=float
                    )
                )).sum()
            ),
        "negative_stat_predictions":
            int(
                (
                    numeric_projection[
                        [
                            column
                            for column
                            in numeric_projection.columns
                            if not column.endswith(
                                "_fd_points"
                            )
                        ]
                    ] < 0
                ).sum().sum()
            ),
        "receptions_over_targets":
            int(
                sum(
                    (
                        result[
                            f"{model}_receptions"
                        ]
                        > result[
                            f"{model}_targets"
                        ] + 1e-9
                    ).sum()
                    for model in (
                        "v1",
                        "hybrid",
                    )
                )
            ),
        "rushing_tds_over_carries":
            int(
                sum(
                    (
                        result[
                            f"{model}_rushing_tds"
                        ]
                        > result[
                            f"{model}_carries"
                        ] + 1e-9
                    ).sum()
                    for model in (
                        "v1",
                        "hybrid",
                    )
                )
            ),
        "receiving_tds_over_targets":
            int(
                sum(
                    (
                        result[
                            f"{model}_receiving_tds"
                        ]
                        > result[
                            f"{model}_targets"
                        ] + 1e-9
                    ).sum()
                    for model in (
                        "v1",
                        "hybrid",
                    )
                )
            ),
    }

    status = (
        "PASS_HARD_CONTRACTS"
        if sum(
            hard_failures.values()
        ) == 0
        else "FAIL_HARD_CONTRACTS"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_"
            "WALKFORWARD_OUTCOMES_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "evaluation":
            "2025_WEEKS_14_TO_18_AND_2026_WEEK_1",
        "fanduel_scoring":
            "HALF_PPR_NO_YARDAGE_BONUSES",
        "rows": int(len(result)),
        "games": int(
            result["game_id"].nunique()
        ),
        "max_conservation_gap":
            max_gap,
        "hard_failures":
            hard_failures,
        "holdout_mae":
            overall.reset_index()
            .to_dict(orient="records"),
        "outputs": {
            "player": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
            "summary": {
                "path": str(
                    SUMMARY_OUTPUT.resolve()
                ),
                "sha256": sha256(
                    SUMMARY_OUTPUT
                ),
            },
        },
    }

    AUDIT_OUTPUT.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "WFS OFFENSIVE RECONCILIATION "
        "WALK-FORWARD OUTCOMES V1"
    )
    print("=" * 80)
    print(f"ROWS={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(
        "MAX_CONSERVATION_GAP="
        f"{max_gap:.18e}"
    )

    print("\n=== UNTOUCHED HOLDOUT MAE ===")
    print(
        overall.reset_index().to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n=== HOLDOUT FANDEUL BY HISTORY CLASS ===")
    print(
        summary[
            summary["metric"].eq(
                "fd_points"
            )
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(f"\nPLAYER_OUTPUT={OUTPUT}")
    print(f"SUMMARY_OUTPUT={SUMMARY_OUTPUT}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(
        "WALKFORWARD_OUTCOMES_STATUS="
        f"{status}"
    )

    if status != "PASS_HARD_CONTRACTS":
        raise RuntimeError(
            "Outcome backtest contract failed"
        )


if __name__ == "__main__":
    main()
