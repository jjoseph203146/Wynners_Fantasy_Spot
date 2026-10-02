"""
Primitive Game-State Generator V1 — deterministic scoring intelligence.

SHADOW / VALIDATION ONLY.

Stage 3:
    generated opportunities
        -> passing TD rate
        -> rushing TD rate
        -> passing/rushing TDs
        -> team points reconciliation

No randomness.
No betting market.
No future observations.
No production integration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from simulator_primitive_generator_v1.intelligence import (
    PrimitiveIntelligenceError,
    estimate_team_volume,
    estimate_team_yardage,
)


SCORING_VERSION = "PRIMITIVE_INTELLIGENCE_V1_SCORING_001"


def _number(value: Any) -> Optional[float]:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)

    return None


def _feature_map(
    replay_state: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    rows = replay_state.get("feature_reconstruction")

    if not isinstance(rows, list):
        raise PrimitiveIntelligenceError(
            "MISSING_REPLAY_FEATURES"
        )

    result: Dict[str, Optional[float]] = {}

    for row in rows:
        if not isinstance(row, Mapping):
            raise PrimitiveIntelligenceError(
                "INVALID_REPLAY_FEATURE_ROW"
            )

        name = row.get("name")

        if not isinstance(name, str) or not name:
            raise PrimitiveIntelligenceError(
                "INVALID_REPLAY_FEATURE_NAME"
            )

        result[name] = _number(row.get("value"))

    return result


def _required(
    features: Mapping[str, Optional[float]],
    name: str,
) -> float:
    value = features.get(name)

    if value is None:
        raise PrimitiveIntelligenceError(
            f"REQUIRED_FEATURE_UNAVAILABLE:{name}"
        )

    return value


def _ratio(
    numerator: float,
    denominator: float,
    label: str,
) -> float:
    if denominator <= 0:
        raise PrimitiveIntelligenceError(
            f"INVALID_SCORING_DENOMINATOR:{label}"
        )

    value = numerator / denominator

    if not math.isfinite(value) or value < 0:
        raise PrimitiveIntelligenceError(
            f"INVALID_SCORING_RATE:{label}"
        )

    return value


def _blend_3_5(
    features: Mapping[str, Optional[float]],
    prefix: str,
    metric: str,
) -> float:
    avg3 = _required(
        features,
        f"{prefix}{metric}_avg_3",
    )

    avg5 = _required(
        features,
        f"{prefix}{metric}_avg_5",
    )

    return (avg3 + avg5) / 2.0


def estimate_team_scoring(
    replay_state: Mapping[str, Any],
    volume: Optional[Mapping[str, Any]] = None,
    yardage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate passing TDs, rushing TDs and reconciled team points.

    TD rates use matching 3-game opportunities because the frozen Replay
    contract exposes TD production/allowance as 3-game values.

    Team points use the available 3/5 scoring windows and are reconciled
    against a six-points-per-offensive-TD floor.

    The residual above the TD floor represents scoring not modeled by the
    current primitive set (FG/PAT/defense/ST/safety/etc.). It is preserved
    explicitly rather than falsely attributed to offensive touchdowns.
    """

    if volume is None:
        volume = estimate_team_volume(replay_state)

    if yardage is None:
        yardage = estimate_team_yardage(
            replay_state,
            volume,
        )

    context = replay_state.get(
        "state_manifest", {}
    ).get("context", {})

    team = context.get("team")
    opponent = context.get("opponent")

    if volume.get("team") != team:
        raise PrimitiveIntelligenceError(
            "VOLUME_TEAM_MISMATCH"
        )

    if volume.get("opponent") != opponent:
        raise PrimitiveIntelligenceError(
            "VOLUME_OPPONENT_MISMATCH"
        )

    if yardage.get("team") != team:
        raise PrimitiveIntelligenceError(
            "YARDAGE_TEAM_MISMATCH"
        )

    if yardage.get("opponent") != opponent:
        raise PrimitiveIntelligenceError(
            "YARDAGE_OPPONENT_MISMATCH"
        )

    pass_attempts = volume.get("pass_attempts")
    carries = volume.get("carries")

    if (
        not isinstance(pass_attempts, int)
        or isinstance(pass_attempts, bool)
        or pass_attempts < 0
    ):
        raise PrimitiveIntelligenceError(
            "INVALID_GENERATED_PASS_ATTEMPTS"
        )

    if (
        not isinstance(carries, int)
        or isinstance(carries, bool)
        or carries < 0
    ):
        raise PrimitiveIntelligenceError(
            "INVALID_GENERATED_CARRIES"
        )

    if yardage.get("pass_attempts") != pass_attempts:
        raise PrimitiveIntelligenceError(
            "YARDAGE_PASS_ATTEMPTS_MISMATCH"
        )

    if yardage.get("carries") != carries:
        raise PrimitiveIntelligenceError(
            "YARDAGE_CARRIES_MISMATCH"
        )

    features = _feature_map(replay_state)

    team_pass_tds_3 = _required(
        features,
        "team_passing_tds_avg_3",
    )

    team_rush_tds_3 = _required(
        features,
        "team_rushing_tds_avg_3",
    )

    pass_tds_allowed_3 = _required(
        features,
        "team_opponent_pass_tds_allowed_avg_3",
    )

    rush_tds_allowed_3 = _required(
        features,
        "team_opponent_rush_tds_allowed_avg_3",
    )

    team_pass_attempts_3 = _required(
        features,
        "team_pass_attempts_avg_3",
    )

    team_rush_attempts_3 = _required(
        features,
        "team_rush_attempts_avg_3",
    )

    opponent_pass_attempts_3 = _required(
        features,
        "opp_pass_attempts_avg_3",
    )

    opponent_rush_attempts_3 = _required(
        features,
        "opp_rush_attempts_avg_3",
    )

    offense_pass_td_rate = _ratio(
        team_pass_tds_3,
        team_pass_attempts_3,
        "OFFENSE_PASS_TD",
    )

    defense_pass_td_rate = _ratio(
        pass_tds_allowed_3,
        opponent_pass_attempts_3,
        "DEFENSE_PASS_TD",
    )

    offense_rush_td_rate = _ratio(
        team_rush_tds_3,
        team_rush_attempts_3,
        "OFFENSE_RUSH_TD",
    )

    defense_rush_td_rate = _ratio(
        rush_tds_allowed_3,
        opponent_rush_attempts_3,
        "DEFENSE_RUSH_TD",
    )

    pass_td_rate = (
        offense_pass_td_rate
        + defense_pass_td_rate
    ) / 2.0

    rush_td_rate = (
        offense_rush_td_rate
        + defense_rush_td_rate
    ) / 2.0

    raw_passing_tds = (
        pass_attempts * pass_td_rate
    )

    raw_rushing_tds = (
        carries * rush_td_rate
    )

    passing_tds = max(
        0,
        int(round(raw_passing_tds)),
    )

    rushing_tds = max(
        0,
        int(round(raw_rushing_tds)),
    )

    team_points_recent = _blend_3_5(
        features,
        "team_",
        "points_for",
    )

    opponent_points_allowed = _blend_3_5(
        features,
        "team_",
        "opponent_points_allowed",
    )

    raw_points_expectation = (
        team_points_recent
        + opponent_points_allowed
    ) / 2.0

    offensive_td_points = (
        passing_tds + rushing_tds
    ) * 6

    baseline_points = max(
        0,
        int(round(raw_points_expectation)),
    )

    points = max(
        baseline_points,
        offensive_td_points,
    )

    non_td_scoring_residual = (
        points - offensive_td_points
    )

    return {
        "engine_version": SCORING_VERSION,
        "team": team,
        "opponent": opponent,
        "pass_attempts": pass_attempts,
        "carries": carries,
        "passing_yards": yardage["passing_yards"],
        "rushing_yards": yardage["rushing_yards"],
        "passing_tds": passing_tds,
        "rushing_tds": rushing_tds,
        "points": points,
        "diagnostics": {
            "td_rate_window_games": 3,
            "offense_pass_td_rate":
                offense_pass_td_rate,
            "defense_pass_td_rate":
                defense_pass_td_rate,
            "blended_pass_td_rate":
                pass_td_rate,
            "raw_passing_tds":
                raw_passing_tds,
            "offense_rush_td_rate":
                offense_rush_td_rate,
            "defense_rush_td_rate":
                defense_rush_td_rate,
            "blended_rush_td_rate":
                rush_td_rate,
            "raw_rushing_tds":
                raw_rushing_tds,
            "team_recent_points":
                team_points_recent,
            "opponent_points_allowed":
                opponent_points_allowed,
            "raw_points_expectation":
                raw_points_expectation,
            "offensive_td_points":
                offensive_td_points,
            "non_td_scoring_residual":
                non_td_scoring_residual,
        },
    }


def generate_team_primitives(
    replay_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Produce the complete seven-primitive team result.

    This is deterministic football intelligence only.
    Provenance/lineage packaging remains the responsibility of the
    Primitive Generator V1 contract layer.
    """

    volume = estimate_team_volume(
        replay_state
    )

    yardage = estimate_team_yardage(
        replay_state,
        volume,
    )

    scoring = estimate_team_scoring(
        replay_state,
        volume,
        yardage,
    )

    return {
        "points": scoring["points"],
        "pass_attempts":
            scoring["pass_attempts"],
        "carries":
            scoring["carries"],
        "passing_yards":
            scoring["passing_yards"],
        "passing_tds":
            scoring["passing_tds"],
        "rushing_yards":
            scoring["rushing_yards"],
        "rushing_tds":
            scoring["rushing_tds"],
    }
