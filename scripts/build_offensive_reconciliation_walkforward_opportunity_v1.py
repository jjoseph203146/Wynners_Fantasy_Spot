#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"

PLAYER_INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_inputs_v1.csv"
)

TEAM_INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_team_v2.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_opportunity_v1.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_opportunity_summary_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_opportunity_v1_audit.json"
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


def numeric(
    value: object,
    default: float = 0.0,
) -> float:
    parsed = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return float(default)

    return float(parsed)


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


def recent(
    frame: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(0.0)

    if values.empty:
        return 0.0, 0.0, 0.0

    return (
        float(values.iloc[-1]),
        float(values.tail(3).mean()),
        float(values.tail(5).mean()),
    )


def main() -> None:
    for required in (
        DB,
        PLAYER_INPUT,
        TEAM_INPUT,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    players = pd.read_csv(PLAYER_INPUT)
    teams = pd.read_csv(TEAM_INPUT)

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        player_history = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                team,
                opponent_team,
                player_id,
                position,
                carries,
                targets
            FROM player_game_stats
            WHERE player_id IS NOT NULL
            """,
            conn,
        )

        team_history = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                team,
                opponent_team,
                carries,
                targets,
                attempts
            FROM team_game_stats
            """,
            conn,
        )

    for frame in (
        players,
        teams,
        player_history,
        team_history,
    ):
        for column in ("season", "week"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

    for column in ("carries", "targets"):
        player_history[column] = pd.to_numeric(
            player_history[column],
            errors="coerce",
        ).fillna(0.0)

    for column in (
        "carries",
        "targets",
        "attempts",
    ):
        team_history[column] = pd.to_numeric(
            team_history[column],
            errors="coerce",
        ).fillna(0.0)

    player_history["position"] = (
        player_history["position"]
        .fillna("")
        .astype(str)
        .str.upper()
        .replace({
            "FB": "RB",
            "HB": "RB",
        })
    )

    player_history = player_history.merge(
        team_history[
            [
                "game_id",
                "team",
                "carries",
                "targets",
            ]
        ].rename(columns={
            "carries": "team_carries",
            "targets": "team_targets",
        }),
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    player_history["carry_share"] = (
        player_history["carries"]
        / player_history[
            "team_carries"
        ].replace(0, np.nan)
    ).fillna(0.0)

    player_history["target_share"] = (
        player_history["targets"]
        / player_history[
            "team_targets"
        ].replace(0, np.nan)
    ).fillna(0.0)

    player_history = player_history.sort_values(
        ["season", "week", "game_id"]
    )

    player_groups = {
        str(player_id): block.copy()
        for player_id, block
        in player_history.groupby(
            player_history[
                "player_id"
            ].astype(str)
        )
    }

    position_game = (
        player_history.groupby(
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent_team",
                "position",
            ],
            as_index=False,
        )[
            [
                "carries",
                "targets",
            ]
        ]
        .sum()
        .sort_values(
            ["season", "week", "game_id"]
        )
    )

    team_budgets = teams[
        [
            "game_id",
            "team",
            "adaptive_attempts",
            "adaptive_carries",
        ]
    ].copy()

    players = players.merge(
        team_budgets,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    output_blocks = []
    audit_rows = []

    for (
        game_id,
        team,
    ), block in players.groupby(
        ["game_id", "team"],
        sort=False,
    ):
        block = block.copy()

        season = int(block["season"].iloc[0])
        week = int(block["week"].iloc[0])
        opponent = str(
            block["opponent_team"].iloc[0]
        )

        carry_budget = float(
            block["adaptive_carries"].iloc[0]
        )

        target_budget = float(
            block["adaptive_attempts"].iloc[0]
        ) * 0.94

        strict_position = position_game[
            (position_game["season"] < season)
            | (
                position_game["season"].eq(
                    season
                )
                & position_game["week"].lt(
                    week
                )
            )
        ]

        v1_carry_weights = []
        v1_target_weights = []
        v2_carry_weights = []
        v2_target_weights = []

        # Convert depth/position role priors into team-level shares
        # before blending them with historical player shares.
        carry_role_total = 0.0
        target_role_total = 0.0

        for role_row in block.itertuples():
            role_position = str(
                role_row.position
            ).upper()

            role_depth_rank = max(
                1.0,
                numeric(
                    role_row.depth_rank,
                    5.0,
                ),
            )

            role_depth_factor = (
                1.0
                / np.sqrt(role_depth_rank)
            )

            role_carry_eligible = (
                role_position in {
                    "QB",
                    "RB",
                    "WR",
                }
            )

            if role_position == "QB":
                role_carry_eligible = bool(
                    role_row.primary_qb_flag == 1
                )

            role_target_eligible = (
                role_position in {
                    "RB",
                    "WR",
                    "TE",
                }
            )

            role_carry_base = {
                "QB": 0.20,
                "RB": 1.00,
                "WR": 0.04,
                "TE": 0.00,
            }.get(role_position, 0.0)

            role_target_base = {
                "QB": 0.00,
                "RB": 0.35,
                "WR": 1.00,
                "TE": 0.75,
            }.get(role_position, 0.0)

            if role_carry_eligible:
                carry_role_total += (
                    role_carry_base
                    * role_depth_factor
                )

            if role_target_eligible:
                target_role_total += (
                    role_target_base
                    * role_depth_factor
                )

        carry_role_total = max(
            carry_role_total,
            1e-9,
        )

        target_role_total = max(
            target_role_total,
            1e-9,
        )

        for row in block.itertuples():
            player_id = str(row.player_id)
            position = str(row.position).upper()

            history = player_groups.get(
                player_id,
                player_history.iloc[0:0],
            )

            prior = history[
                (history["season"] < season)
                | (
                    history["season"].eq(season)
                    & history["week"].lt(week)
                )
            ].sort_values(
                ["season", "week", "game_id"]
            )

            games = len(prior)
            reliability = games / (games + 5.0)

            c_last, c_3, c_5 = recent(
                prior,
                "carries",
            )

            t_last, t_3, t_5 = recent(
                prior,
                "targets",
            )

            cs_last, cs_3, cs_5 = recent(
                prior,
                "carry_share",
            )

            ts_last, ts_3, ts_5 = recent(
                prior,
                "target_share",
            )

            depth_rank = max(
                1.0,
                numeric(
                    row.depth_rank,
                    5.0,
                ),
            )

            depth_factor = (
                1.0
                / np.sqrt(depth_rank)
            )

            carry_eligible = (
                position in {
                    "QB",
                    "RB",
                    "WR",
                }
            )

            if position == "QB":
                carry_eligible = bool(
                    row.primary_qb_flag == 1
                )

            target_eligible = (
                position in {
                    "RB",
                    "WR",
                    "TE",
                }
            )

            v1_carry_signal = (
                0.50 * c_last
                + 0.30 * c_3
                + 0.20 * c_5
            )

            v1_target_signal = (
                0.50 * t_last
                + 0.30 * t_3
                + 0.20 * t_5
            )

            carry_position_prior = {
                "QB": 0.20,
                "RB": 1.00,
                "WR": 0.04,
                "TE": 0.00,
            }.get(position, 0.0)

            target_position_prior = {
                "QB": 0.00,
                "RB": 0.35,
                "WR": 1.00,
                "TE": 0.75,
            }.get(position, 0.0)

            carry_role_prior = (
                carry_position_prior
                * depth_factor
                / carry_role_total
            )

            target_role_prior = (
                target_position_prior
                * depth_factor
                / target_role_total
            )

            if games == 0:
                v1_carry_signal = (
                    carry_role_prior
                )
                v1_target_signal = (
                    target_role_prior
                )

            personal_carry_share = (
                0.50 * cs_last
                + 0.30 * cs_3
                + 0.20 * cs_5
            )

            personal_target_share = (
                0.50 * ts_last
                + 0.30 * ts_3
                + 0.20 * ts_5
            )

            allowed = strict_position[
                strict_position[
                    "opponent_team"
                ].astype(str).eq(opponent)
                & strict_position[
                    "position"
                ].eq(position)
            ]

            league_position = (
                strict_position[
                    strict_position[
                        "position"
                    ].eq(position)
                ]
            )

            allowed_carries = pd.to_numeric(
                allowed["carries"],
                errors="coerce",
            ).fillna(0.0).tail(3).mean()

            allowed_targets = pd.to_numeric(
                allowed["targets"],
                errors="coerce",
            ).fillna(0.0).tail(3).mean()

            league_carries = pd.to_numeric(
                league_position["carries"],
                errors="coerce",
            ).fillna(0.0).mean()

            league_targets = pd.to_numeric(
                league_position["targets"],
                errors="coerce",
            ).fillna(0.0).mean()

            carry_matchup = (
                float(
                    allowed_carries
                    / max(
                        league_carries,
                        1e-9,
                    )
                )
                if not allowed.empty
                else 1.0
            )

            target_matchup = (
                float(
                    allowed_targets
                    / max(
                        league_targets,
                        1e-9,
                    )
                )
                if not allowed.empty
                else 1.0
            )

            carry_matchup = float(
                np.clip(
                    carry_matchup,
                    0.70,
                    1.30,
                )
            )

            target_matchup = float(
                np.clip(
                    target_matchup,
                    0.70,
                    1.30,
                )
            )

            v2_carry_signal = (
                reliability
                * personal_carry_share
                + (
                    1.0 - reliability
                )
                * carry_role_prior
            ) * carry_matchup

            v2_target_signal = (
                reliability
                * personal_target_share
                + (
                    1.0 - reliability
                )
                * target_role_prior
            ) * target_matchup

            v1_carry_weights.append(
                max(
                    0.0,
                    v1_carry_signal
                    * depth_factor,
                )
                if carry_eligible
                else 0.0
            )

            v1_target_weights.append(
                max(
                    0.0,
                    v1_target_signal
                    * depth_factor,
                )
                if target_eligible
                else 0.0
            )

            v2_carry_weights.append(
                max(
                    0.0,
                    v2_carry_signal,
                )
                if carry_eligible
                else 0.0
            )

            v2_target_weights.append(
                max(
                    0.0,
                    v2_target_signal,
                )
                if target_eligible
                else 0.0
            )

        block["v1_carries"] = distribute(
            carry_budget,
            pd.Series(
                v1_carry_weights,
                index=block.index,
            ),
        )

        block["v1_targets"] = distribute(
            target_budget,
            pd.Series(
                v1_target_weights,
                index=block.index,
            ),
        )

        block["v2_carries"] = distribute(
            carry_budget,
            pd.Series(
                v2_carry_weights,
                index=block.index,
            ),
        )

        block["v2_targets"] = distribute(
            target_budget,
            pd.Series(
                v2_target_weights,
                index=block.index,
            ),
        )

        audit_rows.append({
            "game_id": game_id,
            "team": team,
            "carry_budget": carry_budget,
            "v1_carries": float(
                block["v1_carries"].sum()
            ),
            "v2_carries": float(
                block["v2_carries"].sum()
            ),
            "target_budget": target_budget,
            "v1_targets": float(
                block["v1_targets"].sum()
            ),
            "v2_targets": float(
                block["v2_targets"].sum()
            ),
        })

        output_blocks.append(block)

    result = pd.concat(
        output_blocks,
        ignore_index=True,
    )

    conservation = pd.DataFrame(
        audit_rows
    )

    conservation["v1_carry_gap"] = (
        conservation["carry_budget"]
        - conservation["v1_carries"]
    )

    conservation["v2_carry_gap"] = (
        conservation["carry_budget"]
        - conservation["v2_carries"]
    )

    conservation["v1_target_gap"] = (
        conservation["target_budget"]
        - conservation["v1_targets"]
    )

    conservation["v2_target_gap"] = (
        conservation["target_budget"]
        - conservation["v2_targets"]
    )

    gap_columns = [
        "v1_carry_gap",
        "v2_carry_gap",
        "v1_target_gap",
        "v2_target_gap",
    ]

    max_gap = float(
        conservation[
            gap_columns
        ].abs().max().max()
    )

    summary_rows = []

    populations = {
        "ALL_ROSTERED": pd.Series(
            True,
            index=result.index,
        ),
        "ACTUAL_PARTICIPANTS":
            result[
                "actual_participant_flag"
            ].eq(1),
    }

    for population, mask in populations.items():
        block = result[mask]

        for history_class, history_block in [
            ("ALL", block),
            *[
                (str(name), group)
                for name, group
                in block.groupby(
                    "history_class"
                )
            ],
        ]:
            for stat in (
                "carries",
                "targets",
            ):
                actual_values = pd.to_numeric(
                    history_block[
                        f"actual_{stat}"
                    ],
                    errors="coerce",
                ).fillna(0.0)

                for model in ("v1", "v2"):
                    predicted = pd.to_numeric(
                        history_block[
                            f"{model}_{stat}"
                        ],
                        errors="coerce",
                    ).fillna(0.0)

                    error = predicted - actual_values

                    summary_rows.append({
                        "population": population,
                        "history_class":
                            history_class,
                        "stat": stat,
                        "model": model.upper(),
                        "rows": int(
                            len(history_block)
                        ),
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

    summary = pd.DataFrame(summary_rows)

    duplicate_keys = int(
        result.duplicated(
            ["game_id", "team", "player_id"]
        ).sum()
    )

    projection_columns = [
        "v1_carries",
        "v1_targets",
        "v2_carries",
        "v2_targets",
    ]

    nonfinite = int(
        (~np.isfinite(
            result[
                projection_columns
            ].to_numpy(dtype=float)
        )).sum()
    )

    negative = int(
        (
            result[
                projection_columns
            ] < 0
        ).sum().sum()
    )

    hard_failures = {
        "duplicate_player_keys":
            duplicate_keys,
        "nonfinite_projections":
            nonfinite,
        "negative_projections":
            negative,
        "conservation_failure":
            int(max_gap > 1e-8),
        "wrong_game_count":
            int(
                result[
                    "game_id"
                ].nunique() != 180
            ),
        "wrong_team_game_count":
            int(
                result.groupby(
                    ["game_id", "team"]
                ).ngroups != 360
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
            "WALKFORWARD_OPPORTUNITY_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "games": int(
            result["game_id"].nunique()
        ),
        "team_games": int(
            result.groupby(
                ["game_id", "team"]
            ).ngroups
        ),
        "rows": int(len(result)),
        "max_conservation_gap":
            max_gap,
        "hard_failures":
            hard_failures,
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
        "WALK-FORWARD OPPORTUNITY V1"
    )
    print("=" * 80)
    print(f"ROWS={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(
        "TEAM_GAMES="
        f"{result.groupby(['game_id','team']).ngroups}"
    )
    print(
        "MAX_CONSERVATION_GAP="
        f"{max_gap:.18e}"
    )

    print("\n=== ALL-ROSTERED V1 VS V2 ===")
    print(
        summary[
            summary["population"].eq(
                "ALL_ROSTERED"
            )
            & summary[
                "history_class"
            ].eq("ALL")
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n=== RESULTS BY HISTORY CLASS ===")
    print(
        summary[
            summary["population"].eq(
                "ALL_ROSTERED"
            )
            & summary[
                "history_class"
            ].ne("ALL")
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
        "WALKFORWARD_OPPORTUNITY_STATUS="
        f"{status}"
    )

    if status != "PASS_HARD_CONTRACTS":
        raise RuntimeError(
            "Opportunity backtest contract failed"
        )


if __name__ == "__main__":
    main()
