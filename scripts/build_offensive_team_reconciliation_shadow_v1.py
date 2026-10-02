#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data" / "nfl.db"

CURRENT = (
    ROOT
    / "data/parquet/"
    / "current_unified_stat_forecasts.parquet"
)

MATRIX = (
    ROOT
    / "data/parquet/"
    / "nfl_current_offensive_model_matrix.parquet"
)

TEAM_HISTORY = (
    ROOT
    / "processed/"
    / "offensive_reconciliation_team_history_v1.csv"
)

PLAYER_HISTORY = (
    ROOT
    / "processed/"
    / "offensive_reconciliation_player_roles_v1.csv"
)

CALIBRATION = (
    ROOT
    / "processed/"
    / "offensive_reconciliation_calibration_v1_audit.json"
)

OUTPUT = (
    ROOT
    / "processed/"
    / "offensive_team_reconciliation_shadow_v1.csv"
)

TEAM_AUDIT = (
    ROOT
    / "processed/"
    / "offensive_team_reconciliation_team_audit_v1.csv"
)

AUDIT = (
    ROOT
    / "processed/"
    / "offensive_team_reconciliation_shadow_v1_audit.json"
)

STAT_COLUMNS = [
    "expected_attempts",
    "expected_carries",
    "expected_completions",
    "expected_passing_yards",
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_receptions",
    "expected_targets",
    "expected_interceptions",
    "expected_passing_tds",
    "expected_rushing_tds",
    "expected_receiving_tds",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def num(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(value):
        return float(default)

    return value


def bounded(
    value: float,
    low: float,
    high: float,
) -> float:
    return min(
        max(float(value), float(low)),
        float(high),
    )


def weighted(values, weights) -> float:
    pairs = [
        (num(value), float(weight))
        for value, weight in zip(
            values,
            weights,
        )
        if math.isfinite(num(value))
        and float(weight) > 0
    ]

    total = sum(weight for _, weight in pairs)

    if total <= 0:
        return 0.0

    return sum(
        value * weight
        for value, weight in pairs
    ) / total


def distribute(
    total: float,
    weights: pd.Series,
) -> pd.Series:
    clean = pd.to_numeric(
        weights,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    if clean.sum() <= 0:
        clean = pd.Series(
            np.ones(len(clean)),
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

    remaining = float(total)
    available = list(weights.index)

    for _ in range(len(available) + 2):
        if remaining <= 1e-10 or not available:
            break

        current_weights = (
            pd.to_numeric(
                weights.loc[available],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        if current_weights.sum() <= 0:
            current_weights[:] = 1.0

        proposed = (
            remaining
            * current_weights
            / current_weights.sum()
        )

        capped_any = False

        for idx in list(available):
            room = max(
                0.0,
                num(caps.loc[idx])
                - num(result.loc[idx]),
            )

            allocation = num(
                proposed.loc[idx]
            )

            if allocation >= room - 1e-10:
                result.loc[idx] += room
                remaining -= room
                available.remove(idx)
                capped_any = True

        if not capped_any:
            result.loc[available] += proposed
            remaining = 0.0
            break

    return result


def recent_team(
    history: pd.DataFrame,
    team: str,
    column: str,
) -> tuple[float, float, float]:
    rows = history[
        history["team"].astype(str).eq(team)
    ].sort_values(
        ["season", "week", "game_id"]
    )

    values = pd.to_numeric(
        rows[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0, 0.0, 0.0

    return (
        float(values.iloc[-1]),
        float(values.tail(3).mean()),
        float(values.tail(5).mean()),
    )


def player_recent(
    history: pd.DataFrame,
    player_id: str,
    column: str,
) -> tuple[float, float, float]:
    rows = history[
        history["player_id"]
        .fillna("")
        .astype(str)
        .eq(str(player_id))
    ].sort_values(
        ["season", "week", "game_id"]
    )

    values = pd.to_numeric(
        rows[column],
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
    for path in (
        DB,
        CURRENT,
        MATRIX,
        TEAM_HISTORY,
        PLAYER_HISTORY,
        CALIBRATION,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    current = pd.read_parquet(CURRENT)

    offense = current[
        current["entity_type"]
        .astype(str)
        .eq("OFFENSE_PLAYER")
    ].copy()

    matrix = pd.read_parquet(MATRIX)
    team_history = pd.read_csv(TEAM_HISTORY)
    player_history = pd.read_csv(PLAYER_HISTORY)

    target_game_ids = {
        str(value).strip()
        for value in offense[
            "game_id"
        ].dropna()
        if str(value).strip()
    }

    if not target_game_ids:
        raise RuntimeError(
            "Shadow reconciliation has no target games."
        )

    team_history_games = {
        str(value).strip()
        for value in team_history[
            "game_id"
        ].dropna()
        if str(value).strip()
    }

    player_history_games = {
        str(value).strip()
        for value in player_history[
            "game_id"
        ].dropna()
        if str(value).strip()
    }

    team_target_overlap = sorted(
        target_game_ids
        & team_history_games
    )

    player_target_overlap = sorted(
        target_game_ids
        & player_history_games
    )

    if (
        team_target_overlap
        or player_target_overlap
    ):
        raise RuntimeError(
            "TARGET_GAME_HISTORY_LEAKAGE: "
            f"team={team_target_overlap} "
            f"player={player_target_overlap}"
        )

    calibration = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8",
        )
    )

    for column in STAT_COLUMNS:
        offense[column] = pd.to_numeric(
            offense.get(column),
            errors="coerce",
        ).fillna(0.0)

        offense[
            "original_" + column.removeprefix(
                "expected_"
            )
        ] = offense[column]

    matrix_columns = [
        "game_id",
        "player_id",
        "team",
        "active_flag",
        "injury_flag",
        "opportunities_last",
        "opportunities_avg_3",
        "opportunities_avg_5",
        "targets_last",
        "targets_avg_3",
        "targets_avg_5",
        "snap_pct_last",
        "snap_pct_avg_3",
        "snap_pct_avg_5",
        "pass_attempts_avg_3",
        "pass_attempts_avg_5",
        "rush_attempts_avg_3",
        "rush_attempts_avg_5",
    ]

    missing = sorted(
        set(matrix_columns)
        - set(matrix.columns)
    )

    if missing:
        raise RuntimeError(
            f"Matrix missing columns: {missing}"
        )

    matrix_small = matrix[
        matrix_columns
    ].copy()

    offense = offense.merge(
        matrix_small,
        on=["game_id", "player_id", "team"],
        how="left",
        validate="one_to_one",
    )

    # Some current unified forecast games may be newer than the
    # frozen-compatible offensive matrix surface. Preserve matrix
    # authority when present; otherwise resolve availability from
    # exact-GSIS current injury consensus. Healthy-by-absence is the
    # established production contract. Explicit BLOCK always wins.
    offense["matrix_authority_present"] = (
        pd.to_numeric(
            offense["active_flag"],
            errors="coerce",
        ).notna()
    )

    offense["active_flag"] = pd.to_numeric(
        offense["active_flag"],
        errors="coerce",
    )

    uri = f"file:{DB.resolve()}?mode=ro"

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:
        injury = pd.read_sql_query(
            """
            SELECT
                gsis_id AS player_id,
                injury_gate,
                injury_gate_reason
            FROM injury_consensus_current
            """,
            conn,
        )

        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                away_team,
                home_team,
                away_qb_id,
                home_qb_id
            FROM games
            """,
            conn,
        )

        depth = pd.read_sql_query(
            """
            SELECT
                snapshot_dt,
                team,
                gsis_id AS player_id,
                pos_abb AS position,
                pos_rank
            FROM depth_charts
            WHERE pos_abb IN (
                'QB', 'RB', 'WR', 'TE'
            )
            """,
            conn,
        )

    injury["player_id"] = (
        injury["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if injury["player_id"].eq("").any():
        injury = injury[
            injury["player_id"].ne("")
        ].copy()

    duplicate_injury = injury[
        "player_id"
    ].duplicated(keep=False)

    if duplicate_injury.any():
        raise RuntimeError(
            "Duplicate exact-GSIS injury authority"
        )

    injury["injury_gate"] = (
        injury["injury_gate"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    gate_by_player = dict(
        zip(
            injury["player_id"],
            injury["injury_gate"],
        )
    )

    # Matrix values remain authoritative when available.
    # Missing matrix rows use current injury consensus:
    # BLOCK=unavailable; otherwise healthy/allowed.
    missing_matrix = offense[
        "active_flag"
    ].isna()

    offense.loc[
        missing_matrix,
        "active_flag",
    ] = (
        offense.loc[
            missing_matrix,
            "player_id",
        ]
        .fillna("")
        .astype(str)
        .map(gate_by_player)
        .fillna("")
        .ne("BLOCK")
        .astype(int)
    )

    # Explicit current BLOCK also overrides an older matrix flag.
    current_block = (
        offense["player_id"]
        .fillna("")
        .astype(str)
        .map(gate_by_player)
        .fillna("")
        .eq("BLOCK")
    )

    offense.loc[
        current_block,
        "active_flag",
    ] = 0

    offense["active_flag"] = (
        pd.to_numeric(
            offense["active_flag"],
            errors="raise",
        )
        .astype(int)
    )

    depth["snapshot_dt"] = pd.to_datetime(
        depth["snapshot_dt"],
        utc=True,
        errors="coerce",
    )

    latest = (
        depth.groupby("team")[
            "snapshot_dt"
        ].transform("max")
    )

    depth = depth[
        depth["snapshot_dt"].eq(latest)
    ].copy()

    depth = (
        depth.sort_values(
            ["team", "player_id", "pos_rank"]
        )
        .drop_duplicates(
            ["team", "player_id"],
            keep="first",
        )
    )

    offense = offense.merge(
        depth[
            ["team", "player_id", "pos_rank"]
        ],
        on=["team", "player_id"],
        how="left",
        validate="many_to_one",
    )

    qb_authority = {}

    for row in games.itertuples(index=False):
        qb_authority[
            (str(row.game_id), str(row.away_team))
        ] = str(row.away_qb_id or "").strip()

        qb_authority[
            (str(row.game_id), str(row.home_team))
        ] = str(row.home_qb_id or "").strip()

    team_q = calibration[
        "team_quantiles"
    ]

    role_q = calibration[
        "position_quantiles"
    ]

    league = {
        column: float(
            pd.to_numeric(
                team_history[column],
                errors="coerce",
            ).median()
        )
        for column in (
            "attempts",
            "carries",
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
            "completion_rate",
        )
    }

    # Construct exact opponent results for defensive allowance history.
    # Each team-game receives the opposing offense's realized values.
    defense_metrics = [
        "attempts",
        "carries",
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
    ]

    offense_side = team_history[
        [
            "game_id",
            "team",
            *defense_metrics,
        ]
    ].copy()

    offense_side = offense_side.rename(
        columns={
            "team": "opposing_offense",
            **{
                metric: f"allowed_{metric}"
                for metric in defense_metrics
            },
        }
    )

    defense_history = team_history[
        [
            "game_id",
            "season",
            "week",
            "team",
        ]
    ].merge(
        offense_side,
        on="game_id",
        how="inner",
    )

    defense_history = defense_history[
        defense_history["team"].astype(str)
        .ne(
            defense_history[
                "opposing_offense"
            ].astype(str)
        )
    ].copy()

    def recent_allowed(
        defense_team: str,
        metric: str,
    ) -> float:
        rows = defense_history[
            defense_history["team"]
            .astype(str)
            .eq(str(defense_team))
        ].sort_values(
            ["season", "week", "game_id"]
        )

        values = pd.to_numeric(
            rows[f"allowed_{metric}"],
            errors="coerce",
        ).dropna()

        if values.empty:
            return float(league[metric])

        return float(values.tail(3).mean())

    output_blocks = []
    team_audits = []

    for (
        game_id,
        team,
    ), block in offense.groupby(
        ["game_id", "team"],
        sort=True,
    ):
        block = block.copy()

        primary_qb = qb_authority.get(
            (str(game_id), str(team)),
            "",
        )

        qb_mask = block[
            "position"
        ].astype(str).eq("QB")

        valid_primary = (
            bool(primary_qb)
            and (
                block["player_id"]
                .astype(str)
                .eq(primary_qb)
                & qb_mask
                & block["active_flag"].eq(1)
            ).any()
        )

        if not valid_primary:
            candidates = block[
                qb_mask
                & block["active_flag"].eq(1)
            ].copy()

            if candidates.empty:
                raise RuntimeError(
                    f"No active QB: {game_id} {team}"
                )

            candidates["rank_sort"] = (
                pd.to_numeric(
                    candidates["pos_rank"],
                    errors="coerce",
                ).fillna(999)
            )

            candidates = candidates.sort_values(
                [
                    "rank_sort",
                    "snap_pct_last",
                    "expected_attempts",
                ],
                ascending=[True, False, False],
            )

            primary_qb = str(
                candidates.iloc[0]["player_id"]
            )

        block["reconciliation_role"] = (
            "ACTIVE_ROTATION"
        )

        block.loc[
            qb_mask,
            "reconciliation_role",
        ] = "CONTINGENCY_QB"

        block.loc[
            block["player_id"]
            .astype(str)
            .eq(primary_qb),
            "reconciliation_role",
        ] = "PRIMARY_QB"

        block.loc[
            block["active_flag"].ne(1),
            "reconciliation_role",
        ] = "UNAVAILABLE"

        active = block[
            block["reconciliation_role"].isin(
                ["PRIMARY_QB", "ACTIVE_ROTATION"]
            )
        ].copy()

        if active.empty:
            raise RuntimeError(
                f"No active offense: {game_id} {team}"
            )

        pass_last, pass_3, pass_5 = recent_team(
            team_history,
            str(team),
            "attempts",
        )

        rush_last, rush_3, rush_5 = recent_team(
            team_history,
            str(team),
            "carries",
        )

        py_last, py_3, py_5 = recent_team(
            team_history,
            str(team),
            "passing_yards",
        )

        ry_last, ry_3, ry_5 = recent_team(
            team_history,
            str(team),
            "rushing_yards",
        )

        ptd_last, ptd_3, ptd_5 = recent_team(
            team_history,
            str(team),
            "passing_tds",
        )

        rtd_last, rtd_3, rtd_5 = recent_team(
            team_history,
            str(team),
            "rushing_tds",
        )

        opponent_team = str(
            block["opponent_team"].iloc[0]
        )

        allowed_attempts = recent_allowed(
            opponent_team,
            "attempts",
        )
        allowed_carries = recent_allowed(
            opponent_team,
            "carries",
        )
        allowed_passing_yards = recent_allowed(
            opponent_team,
            "passing_yards",
        )
        allowed_rushing_yards = recent_allowed(
            opponent_team,
            "rushing_yards",
        )
        allowed_passing_tds = recent_allowed(
            opponent_team,
            "passing_tds",
        )
        allowed_rushing_tds = recent_allowed(
            opponent_team,
            "rushing_tds",
        )

        # Week 1 remains the strongest signal. Recent team history,
        # opponent allowance, and the league baseline stabilize it.
        weights = (
            0.45,
            0.25,
            0.10,
            0.15,
            0.05,
        )

        pass_attempts = weighted(
            (
                pass_last,
                pass_3,
                pass_5,
                allowed_attempts,
                league["attempts"],
            ),
            weights,
        )

        carries = weighted(
            (
                rush_last,
                rush_3,
                rush_5,
                allowed_carries,
                league["carries"],
            ),
            weights,
        )

        passing_yards = weighted(
            (
                py_last,
                py_3,
                py_5,
                allowed_passing_yards,
                league["passing_yards"],
            ),
            weights,
        )

        rushing_yards = weighted(
            (
                ry_last,
                ry_3,
                ry_5,
                allowed_rushing_yards,
                league["rushing_yards"],
            ),
            weights,
        )

        passing_tds = weighted(
            (
                ptd_last,
                ptd_3,
                ptd_5,
                allowed_passing_tds,
                league["passing_tds"],
            ),
            weights,
        )

        rushing_tds = weighted(
            (
                rtd_last,
                rtd_3,
                rtd_5,
                allowed_rushing_tds,
                league["rushing_tds"],
            ),
            weights,
        )

        for name, value in (
            ("attempts", pass_attempts),
            ("carries", carries),
            ("passing_yards", passing_yards),
            ("rushing_yards", rushing_yards),
            ("passing_tds", passing_tds),
            ("rushing_tds", rushing_tds),
        ):
            limits = team_q[name]

            adjusted = bounded(
                value,
                limits["p01"],
                limits["p99"],
            )

            if name == "attempts":
                pass_attempts = adjusted
            elif name == "carries":
                carries = adjusted
            elif name == "passing_yards":
                passing_yards = adjusted
            elif name == "rushing_yards":
                rushing_yards = adjusted
            elif name == "passing_tds":
                passing_tds = adjusted
            elif name == "rushing_tds":
                rushing_tds = adjusted

        completion_rate = weighted(
            (
                num(
                    active.loc[
                        active["player_id"]
                        .astype(str)
                        .eq(primary_qb),
                        "expected_completions",
                    ].sum()
                )
                / max(
                    num(
                        active.loc[
                            active["player_id"]
                            .astype(str)
                            .eq(primary_qb),
                            "expected_attempts",
                        ].sum()
                    ),
                    1e-9,
                ),
                league["completion_rate"],
            ),
            (0.75, 0.25),
        )

        completion_rate = bounded(
            completion_rate,
            team_q["completion_rate"]["p01"],
            team_q["completion_rate"]["p99"],
        )

        completions = min(
            pass_attempts,
            pass_attempts * completion_rate,
        )

        target_budget = pass_attempts * 0.94

        active["depth_factor"] = (
            1.0
            / np.sqrt(
                pd.to_numeric(
                    active["pos_rank"],
                    errors="coerce",
                ).fillna(4.0).clip(lower=1.0)
            )
        )

        carry_weights = []

        target_weights = []

        for row in active.itertuples():
            player_id = str(row.player_id)

            c_last, c_3, c_5 = player_recent(
                player_history,
                player_id,
                "carries",
            )

            t_last, t_3, t_5 = player_recent(
                player_history,
                player_id,
                "targets",
            )

            position = str(row.position)

            carry_eligible = (
                position in {
                    "QB", "RB", "FB", "WR"
                }
            )

            if position == "QB":
                carry_eligible = (
                    player_id == primary_qb
                )

            target_eligible = position in {
                "RB", "FB", "WR", "TE"
            }

            depth_factor = num(
                row.depth_factor,
                0.5,
            )

            carry_signal = weighted(
                (
                    c_last,
                    c_3,
                    c_5,
                    num(row.expected_carries),
                ),
                (0.50, 0.25, 0.10, 0.15),
            )

            target_signal = weighted(
                (
                    t_last,
                    t_3,
                    t_5,
                    num(row.expected_targets),
                ),
                (0.50, 0.25, 0.10, 0.15),
            )

            carry_weights.append(
                max(
                    0.0,
                    carry_signal * depth_factor,
                )
                if carry_eligible
                else 0.0
            )

            target_weights.append(
                max(
                    0.0,
                    target_signal * depth_factor,
                )
                if target_eligible
                else 0.0
            )

        active[
            "reconciled_carries"
        ] = distribute(
            carries,
            pd.Series(
                carry_weights,
                index=active.index,
            ),
        )

        active[
            "reconciled_targets"
        ] = distribute(
            target_budget,
            pd.Series(
                target_weights,
                index=active.index,
            ),
        )

        catch_weights = (
            active["reconciled_targets"]
            * (
                pd.to_numeric(
                    active["expected_receptions"],
                    errors="coerce",
                ).fillna(0.0)
                / pd.to_numeric(
                    active["expected_targets"],
                    errors="coerce",
                ).replace(0, np.nan)
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.60)
            .clip(lower=0.25, upper=0.90)
        )

        active[
            "reconciled_receptions"
        ] = distribute_with_caps(
            completions,
            catch_weights,
            active["reconciled_targets"],
        )

        rush_yard_weights = (
            active["reconciled_carries"]
            * (
                pd.to_numeric(
                    active["expected_rushing_yards"],
                    errors="coerce",
                ).fillna(0.0)
                / pd.to_numeric(
                    active["expected_carries"],
                    errors="coerce",
                ).replace(0, np.nan)
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(4.0)
            .clip(lower=1.5, upper=9.0)
        )

        active[
            "reconciled_rushing_yards"
        ] = distribute(
            rushing_yards,
            rush_yard_weights,
        )

        receiving_weights = (
            active["reconciled_receptions"]
            * (
                pd.to_numeric(
                    active["expected_receiving_yards"],
                    errors="coerce",
                ).fillna(0.0)
                / pd.to_numeric(
                    active["expected_receptions"],
                    errors="coerce",
                ).replace(0, np.nan)
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(8.0)
            .clip(lower=3.0, upper=25.0)
        )

        active[
            "reconciled_receiving_yards"
        ] = distribute(
            passing_yards,
            receiving_weights,
        )

        original_target_denominator = (
            pd.to_numeric(
                active["expected_targets"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.25)
        )

        original_receiving_td_rate = (
            pd.to_numeric(
                active["expected_receiving_tds"],
                errors="coerce",
            ).fillna(0.0)
            / original_target_denominator
        ).clip(lower=0.0, upper=0.35)

        receiving_td_weights = (
            active["reconciled_targets"]
            * (
                0.10
                + original_receiving_td_rate
            )
        )

        receiving_td_weights.loc[
            active["reconciled_targets"].lt(0.25)
        ] = 0.0

        active[
            "reconciled_receiving_tds"
        ] = distribute_with_caps(
            passing_tds,
            receiving_td_weights,
            active["reconciled_targets"],
        )

        original_carry_denominator = (
            pd.to_numeric(
                active["expected_carries"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.25)
        )

        original_rushing_td_rate = (
            pd.to_numeric(
                active["expected_rushing_tds"],
                errors="coerce",
            ).fillna(0.0)
            / original_carry_denominator
        ).clip(lower=0.0, upper=0.35)

        rushing_td_weights = (
            active["reconciled_carries"]
            * (
                0.05
                + original_rushing_td_rate
            )
        )

        active[
            "reconciled_rushing_tds"
        ] = distribute_with_caps(
            rushing_tds,
            rushing_td_weights,
            active["reconciled_carries"],
        )

        active["reconciled_attempts"] = 0.0
        active["reconciled_completions"] = 0.0
        active["reconciled_passing_yards"] = 0.0
        active["reconciled_passing_tds"] = 0.0
        active["reconciled_interceptions"] = 0.0

        primary_mask = (
            active["player_id"]
            .astype(str)
            .eq(primary_qb)
        )

        active.loc[
            primary_mask,
            "reconciled_attempts",
        ] = pass_attempts

        active.loc[
            primary_mask,
            "reconciled_completions",
        ] = completions

        active.loc[
            primary_mask,
            "reconciled_passing_yards",
        ] = passing_yards

        active.loc[
            primary_mask,
            "reconciled_passing_tds",
        ] = passing_tds

        original_primary_attempts = max(
            num(
                active.loc[
                    primary_mask,
                    "expected_attempts",
                ].sum()
            ),
            1e-9,
        )

        interception_rate = (
            num(
                active.loc[
                    primary_mask,
                    "expected_interceptions",
                ].sum()
            )
            / original_primary_attempts
        )

        active.loc[
            primary_mask,
            "reconciled_interceptions",
        ] = pass_attempts * max(
            0.0,
            interception_rate,
        )

        block["reconciled_attempts"] = 0.0
        block["reconciled_completions"] = 0.0
        block["reconciled_passing_yards"] = 0.0
        block["reconciled_passing_tds"] = 0.0
        block["reconciled_interceptions"] = 0.0
        block["reconciled_carries"] = 0.0
        block["reconciled_rushing_yards"] = 0.0
        block["reconciled_rushing_tds"] = 0.0
        block["reconciled_targets"] = 0.0
        block["reconciled_receptions"] = 0.0
        block["reconciled_receiving_yards"] = 0.0
        block["reconciled_receiving_tds"] = 0.0

        reconciled_columns = [
            column
            for column in active.columns
            if column.startswith("reconciled_")
        ]

        block.loc[
            active.index,
            reconciled_columns,
        ] = active[
            reconciled_columns
        ]

        block["primary_qb_id"] = primary_qb

        output_blocks.append(block)

        team_audits.append({
            "game_id": game_id,
            "team": team,
            "primary_qb_id": primary_qb,
            "active_players": int(len(active)),
            "contingency_qbs": int(
                block[
                    "reconciliation_role"
                ].eq("CONTINGENCY_QB").sum()
            ),
            "team_pass_attempts": pass_attempts,
            "player_pass_attempts": float(
                active[
                    "reconciled_attempts"
                ].sum()
            ),
            "team_completions": completions,
            "player_receptions": float(
                active[
                    "reconciled_receptions"
                ].sum()
            ),
            "team_passing_yards": passing_yards,
            "player_receiving_yards": float(
                active[
                    "reconciled_receiving_yards"
                ].sum()
            ),
            "team_passing_tds": passing_tds,
            "player_receiving_tds": float(
                active[
                    "reconciled_receiving_tds"
                ].sum()
            ),
            "team_carries": carries,
            "player_carries": float(
                active[
                    "reconciled_carries"
                ].sum()
            ),
            "team_rushing_yards": rushing_yards,
            "player_rushing_yards": float(
                active[
                    "reconciled_rushing_yards"
                ].sum()
            ),
            "team_rushing_tds": rushing_tds,
            "player_rushing_tds": float(
                active[
                    "reconciled_rushing_tds"
                ].sum()
            ),
        })

    result = pd.concat(
        output_blocks,
        ignore_index=True,
    )

    team_audit = pd.DataFrame(
        team_audits
    )

    identities = [
        (
            "attempt_gap",
            "team_pass_attempts",
            "player_pass_attempts",
        ),
        (
            "completion_reception_gap",
            "team_completions",
            "player_receptions",
        ),
        (
            "pass_receive_yard_gap",
            "team_passing_yards",
            "player_receiving_yards",
        ),
        (
            "pass_receive_td_gap",
            "team_passing_tds",
            "player_receiving_tds",
        ),
        (
            "carry_gap",
            "team_carries",
            "player_carries",
        ),
        (
            "rushing_yard_gap",
            "team_rushing_yards",
            "player_rushing_yards",
        ),
        (
            "rushing_td_gap",
            "team_rushing_tds",
            "player_rushing_tds",
        ),
    ]

    max_gap = 0.0

    for gap, left, right in identities:
        team_audit[gap] = (
            team_audit[left]
            - team_audit[right]
        )

        max_gap = max(
            max_gap,
            float(
                team_audit[gap]
                .abs()
                .max()
            ),
        )

    if max_gap > 1e-8:
        raise RuntimeError(
            f"Conservation failure: {max_gap}"
        )

    if (
        team_audit["primary_qb_id"]
        .fillna("")
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Blank primary QB authority"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    team_audit.to_csv(
        TEAM_AUDIT,
        index=False,
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_TEAM_RECONCILIATION_SHADOW_V1",
        "status": "PASS",
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "target_history_exclusion": {
            "status": "PASS",
            "target_games": int(
                len(target_game_ids)
            ),
            "team_overlap": 0,
            "player_overlap": 0,
        },
        "production_modified": False,
        "rows": int(len(result)),
        "games": int(
            result["game_id"].nunique()
        ),
        "teams": int(len(team_audit)),
        "max_conservation_gap": max_gap,
        "primary_qbs": int(
            team_audit[
                "primary_qb_id"
            ].nunique()
        ),
        "contingency_qb_rows": int(
            result[
                "reconciliation_role"
            ].eq("CONTINGENCY_QB").sum()
        ),
        "unavailable_rows": int(
            result[
                "reconciliation_role"
            ].eq("UNAVAILABLE").sum()
        ),
        "week1_weight": 0.45,
        "recent3_weight": 0.25,
        "recent5_weight": 0.10,
        "opponent_allowed_weight": 0.15,
        "league_weight": 0.05,
        "empirical_bounds": "P01_P99",
        "outputs": {
            "player_shadow": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
            "team_audit": {
                "path": str(
                    TEAM_AUDIT.resolve()
                ),
                "sha256": sha256(
                    TEAM_AUDIT
                ),
            },
        },
    }

    AUDIT.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "WFS ALL-OFFENSE TEAM "
        "RECONCILIATION SHADOW V1"
    )
    print("=" * 80)
    print(f"ROWS={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(f"TEAMS={len(team_audit)}")
    print(
        "CONTINGENCY_QB_ROWS="
        f"{audit['contingency_qb_rows']}"
    )
    print(
        "UNAVAILABLE_ROWS="
        f"{audit['unavailable_rows']}"
    )
    print(
        "MAX_CONSERVATION_GAP="
        f"{max_gap:.18e}"
    )

    sample = result[
        result["game_id"]
        .astype(str)
        .eq("2026_02_SEA_ARI")
        & result[
            "reconciliation_role"
        ].isin(
            ["PRIMARY_QB", "ACTIVE_ROTATION"]
        )
    ][[
        "team",
        "player_id",
        "entity_name",
        "position",
        "reconciliation_role",
        "reconciled_attempts",
        "reconciled_carries",
        "reconciled_passing_yards",
        "reconciled_rushing_yards",
        "reconciled_targets",
        "reconciled_receptions",
        "reconciled_receiving_yards",
        "reconciled_passing_tds",
        "reconciled_rushing_tds",
        "reconciled_receiving_tds",
    ]]

    print("\n=== SEA–ARI SHADOW ===")
    print(
        sample.sort_values(
            ["team", "position", "entity_name"]
        ).to_string(index=False)
    )

    print(f"\nPLAYER_OUTPUT={OUTPUT}")
    print(f"TEAM_AUDIT={TEAM_AUDIT}")
    print(f"AUDIT_OUTPUT={AUDIT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print("SHADOW_STATUS=PASS")


if __name__ == "__main__":
    main()
