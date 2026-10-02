"""
Primitive Game-State Generator V1 — deterministic football intelligence.

SHADOW / VALIDATION ONLY.

Stage 1:
    Replay state -> plays -> pass/rush allocation

Stage 2:
    generated opportunities -> passing/rushing efficiency -> yards

No randomness.
No betting market.
No future observations.
No production integration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


class PrimitiveIntelligenceError(ValueError):
    pass


VOLUME_VERSION = "PRIMITIVE_INTELLIGENCE_V1_VOLUME_001"
YARDAGE_VERSION = "PRIMITIVE_INTELLIGENCE_V1_YARDAGE_002"


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


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(low, min(high, value))


def _ratio(
    numerator: float,
    denominator: float,
    label: str,
) -> float:
    if denominator <= 0:
        raise PrimitiveIntelligenceError(
            f"INVALID_EFFICIENCY_DENOMINATOR:{label}"
        )

    value = numerator / denominator

    if not math.isfinite(value) or value < 0:
        raise PrimitiveIntelligenceError(
            f"INVALID_EFFICIENCY:{label}"
        )

    return value


def estimate_team_volume(
    replay_state: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = replay_state.get("state_manifest", {})
    context = manifest.get("context", {})

    team = context.get("team")
    opponent = context.get("opponent")

    if not isinstance(team, str) or not team:
        raise PrimitiveIntelligenceError("MISSING_TEAM")

    if not isinstance(opponent, str) or not opponent:
        raise PrimitiveIntelligenceError("MISSING_OPPONENT")

    features = _feature_map(replay_state)

    team_plays = _blend_3_5(
        features,
        "team_",
        "offensive_plays",
    )

    opp_plays = _blend_3_5(
        features,
        "opp_",
        "offensive_plays",
    )

    expected_plays_raw = (
        team_plays + opp_plays
    ) / 2.0

    expected_plays = _clamp(
        expected_plays_raw,
        40.0,
        90.0,
    )

    team_pass_rate = _blend_3_5(
        features,
        "team_",
        "pass_rate",
    )

    opp_pass_rate = _blend_3_5(
        features,
        "opp_",
        "pass_rate",
    )

    expected_pass_rate_raw = (
        team_pass_rate + opp_pass_rate
    ) / 2.0

    expected_pass_rate = _clamp(
        expected_pass_rate_raw,
        0.25,
        0.80,
    )

    integer_plays = int(round(expected_plays))

    pass_attempts = int(
        round(integer_plays * expected_pass_rate)
    )

    carries = integer_plays - pass_attempts

    if pass_attempts < 0 or carries < 0:
        raise PrimitiveIntelligenceError(
            "NEGATIVE_VOLUME"
        )

    if pass_attempts + carries != integer_plays:
        raise PrimitiveIntelligenceError(
            "VOLUME_CONSERVATION_FAILURE"
        )

    return {
        "engine_version": VOLUME_VERSION,
        "team": team,
        "opponent": opponent,
        "offensive_plays": integer_plays,
        "pass_attempts": pass_attempts,
        "carries": carries,
        "pass_rate": pass_attempts / integer_plays,
        "rush_rate": carries / integer_plays,
        "diagnostics": {
            "team_recent_plays": team_plays,
            "opponent_recent_plays": opp_plays,
            "raw_expected_plays": expected_plays_raw,
            "team_recent_pass_rate": team_pass_rate,
            "opponent_recent_pass_rate": opp_pass_rate,
            "raw_expected_pass_rate":
                expected_pass_rate_raw,
        },
    }


def estimate_team_yardage(
    replay_state: Mapping[str, Any],
    volume: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert deterministic opportunity volume into passing/rushing yards.

    The frozen Replay contract supplies yardage and yards-allowed only as
    3-game rolling values. Therefore Stage 2 deliberately uses matching
    3-game numerators and denominators.

    Offensive pass efficiency:
        team_passing_yards_avg_3 / team_pass_attempts_avg_3

    Defensive pass efficiency allowed:
        team_opponent_pass_yards_allowed_avg_3 /
        opp_pass_attempts_avg_3

    Offensive rush efficiency:
        team_rushing_yards_avg_3 / team_rush_attempts_avg_3

    Defensive rush efficiency allowed:
        team_opponent_rush_yards_allowed_avg_3 /
        opp_rush_attempts_avg_3

    No nonexistent 5-game yardage field is inferred or fabricated.
    """

    if volume is None:
        volume = estimate_team_volume(replay_state)

    features = _feature_map(replay_state)

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

    team_pass_yards_3 = _required(
        features,
        "team_passing_yards_avg_3",
    )

    team_pass_attempts_3 = _required(
        features,
        "team_pass_attempts_avg_3",
    )

    team_rush_yards_3 = _required(
        features,
        "team_rushing_yards_avg_3",
    )

    team_rush_attempts_3 = _required(
        features,
        "team_rush_attempts_avg_3",
    )

    defense_pass_yards_allowed_3 = _required(
        features,
        "team_opponent_pass_yards_allowed_avg_3",
    )

    defense_rush_yards_allowed_3 = _required(
        features,
        "team_opponent_rush_yards_allowed_avg_3",
    )

    opponent_pass_attempts_3 = _required(
        features,
        "opp_pass_attempts_avg_3",
    )

    opponent_rush_attempts_3 = _required(
        features,
        "opp_rush_attempts_avg_3",
    )

    offense_pass_ypa = _ratio(
        team_pass_yards_3,
        team_pass_attempts_3,
        "OFFENSE_PASS",
    )

    defense_pass_ypa = _ratio(
        defense_pass_yards_allowed_3,
        opponent_pass_attempts_3,
        "DEFENSE_PASS",
    )

    offense_rush_ypc = _ratio(
        team_rush_yards_3,
        team_rush_attempts_3,
        "OFFENSE_RUSH",
    )

    defense_rush_ypc = _ratio(
        defense_rush_yards_allowed_3,
        opponent_rush_attempts_3,
        "DEFENSE_RUSH",
    )

    raw_pass_ypa = (
        offense_pass_ypa + defense_pass_ypa
    ) / 2.0

    raw_rush_ypc = (
        offense_rush_ypc + defense_rush_ypc
    ) / 2.0

    pass_ypa = _clamp(
        raw_pass_ypa,
        3.0,
        12.0,
    )

    rush_ypc = _clamp(
        raw_rush_ypc,
        2.0,
        8.0,
    )

    passing_yards = int(
        round(pass_attempts * pass_ypa)
    )

    rushing_yards = int(
        round(carries * rush_ypc)
    )

    if passing_yards < 0 or rushing_yards < 0:
        raise PrimitiveIntelligenceError(
            "NEGATIVE_GENERATED_YARDAGE"
        )

    return {
        "engine_version": YARDAGE_VERSION,
        "team": team,
        "opponent": opponent,
        "pass_attempts": pass_attempts,
        "carries": carries,
        "passing_yards": passing_yards,
        "rushing_yards": rushing_yards,
        "passing_yards_per_attempt": pass_ypa,
        "rushing_yards_per_carry": rush_ypc,
        "diagnostics": {
            "efficiency_window_games": 3,
            "offense_pass_ypa": offense_pass_ypa,
            "defense_pass_ypa": defense_pass_ypa,
            "raw_pass_ypa": raw_pass_ypa,
            "offense_rush_ypc": offense_rush_ypc,
            "defense_rush_ypc": defense_rush_ypc,
            "raw_rush_ypc": raw_rush_ypc,
        },
    }
