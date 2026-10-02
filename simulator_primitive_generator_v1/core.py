"""
Primitive Game-State Generator V1
=================================

SHADOW / VALIDATION ONLY.

Purpose
-------
Define and validate the deterministic primitive-result contract required by
Recursive State Replay Validator V1.

This module intentionally DOES NOT generate probabilistic football outcomes.
It establishes the safe boundary that a future primitive generator must obey.

No production integration.
No Monte Carlo.
No model training.
No FanDuel influence.
No solver influence.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


CONTRACT_VERSION = "PRIMITIVE_GAME_STATE_GENERATOR_V1"

PROVENANCE_VALUES = {"OBSERVED", "SIMULATED"}

PRIMITIVE_FIELDS = (
    "points",
    "pass_attempts",
    "carries",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
)

INTEGER_PRIMITIVES = {
    "points",
    "pass_attempts",
    "carries",
    "passing_tds",
    "rushing_tds",
}

NONNEGATIVE_PRIMITIVES = set(PRIMITIVE_FIELDS)


class PrimitiveContractError(ValueError):
    """Raised when primitive state violates the V1 contract."""


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """SHA256 over canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrimitiveContractError(message)


def _parse_utc(value: Optional[str], field: str) -> Optional[datetime]:
    if value is None:
        return None

    _require(isinstance(value, str) and value.strip(), f"{field} must be UTC ISO-8601")

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PrimitiveContractError(f"{field} is not valid ISO-8601") from exc

    _require(parsed.tzinfo is not None, f"{field} must include timezone")
    parsed = parsed.astimezone(timezone.utc)

    return parsed


def _validate_number(value: Any, field: str) -> None:
    if value is None:
        return

    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric or null",
    )

    _require(math.isfinite(float(value)), f"{field} must be finite")

    if field in NONNEGATIVE_PRIMITIVES:
        _require(float(value) >= 0.0, f"{field} cannot be negative")

    if field in INTEGER_PRIMITIVES:
        _require(float(value).is_integer(), f"{field} must be integer-valued")


def _validate_team(team: Mapping[str, Any], label: str) -> None:
    _require(isinstance(team, Mapping), f"{label} must be an object")

    team_id = team.get("team")
    _require(isinstance(team_id, str) and team_id.strip(), f"{label}.team required")

    primitives = team.get("primitives")
    _require(isinstance(primitives, Mapping), f"{label}.primitives required")

    _require(
        set(primitives.keys()) == set(PRIMITIVE_FIELDS),
        f"{label}.primitives must contain exactly {list(PRIMITIVE_FIELDS)}",
    )

    for field in PRIMITIVE_FIELDS:
        _validate_number(primitives[field], field)


def validate_primitive_result(data: Mapping[str, Any]) -> None:
    """
    Validate one complete two-team primitive game result.

    Missing primitive values are represented by None, never fabricated zero.
    """

    _require(isinstance(data, Mapping), "result must be an object")

    _require(
        data.get("contract_version") == CONTRACT_VERSION,
        "contract_version mismatch",
    )

    season = data.get("season")
    week = data.get("week")
    simulation_step = data.get("simulation_step")

    _require(isinstance(season, int) and season > 0, "season must be positive integer")
    _require(isinstance(week, int) and week > 0, "week must be positive integer")
    _require(
        isinstance(simulation_step, int) and simulation_step >= 0,
        "simulation_step must be nonnegative integer",
    )

    for field in ("game_id", "scenario_id", "source_version", "source_hash",
                  "transition_version"):
        value = data.get(field)
        _require(isinstance(value, str) and value.strip(), f"{field} required")

    provenance = data.get("provenance")
    _require(
        provenance in PROVENANCE_VALUES,
        f"provenance must be one of {sorted(PROVENANCE_VALUES)}",
    )

    parent_state_id = data.get("parent_state_id")
    parent_state_hash = data.get("parent_state_hash")

    _require(
        isinstance(parent_state_id, str) and parent_state_id.strip(),
        "parent_state_id required",
    )
    _require(
        isinstance(parent_state_hash, str) and parent_state_hash.strip(),
        "parent_state_hash required",
    )

    teams = data.get("teams")
    _require(isinstance(teams, list) and len(teams) == 2, "exactly two teams required")

    _validate_team(teams[0], "teams[0]")
    _validate_team(teams[1], "teams[1]")

    team_keys = [str(t["team"]).strip() for t in teams]
    _require(team_keys[0] != team_keys[1], "teams must be distinct")

    kickoff = _parse_utc(data.get("kickoff_utc"), "kickoff_utc")
    completion = _parse_utc(data.get("completion_utc"), "completion_utc")
    observed = _parse_utc(data.get("observed_at_utc"), "observed_at_utc")
    simulated = _parse_utc(data.get("simulated_at_utc"), "simulated_at_utc")
    cutoff = _parse_utc(data.get("cutoff_utc"), "cutoff_utc")

    _require(kickoff is not None, "kickoff_utc required")
    _require(completion is not None, "completion_utc required")
    _require(cutoff is not None, "cutoff_utc required")

    _require(kickoff < completion, "completion must be after kickoff")
    _require(completion <= cutoff, "completion cannot be after cutoff")

    if provenance == "OBSERVED":
        _require(observed is not None, "OBSERVED result requires observed_at_utc")
        _require(simulated is None, "OBSERVED result cannot have simulated_at_utc")
        _require(
            completion <= observed <= cutoff,
            "OBSERVED chronology requires completion <= observed_at <= cutoff",
        )

    if provenance == "SIMULATED":
        _require(simulated is not None, "SIMULATED result requires simulated_at_utc")
        _require(observed is None, "SIMULATED result cannot have observed_at_utc")
        _require(
            completion <= simulated <= cutoff,
            "SIMULATED chronology requires completion <= simulated_at <= cutoff",
        )

    coach_bindings = data.get("coach_bindings")
    _require(isinstance(coach_bindings, Mapping), "coach_bindings required")

    for team_key in team_keys:
        _require(team_key in coach_bindings, f"missing coach binding for {team_key}")
        coach = coach_bindings[team_key]
        _require(
            coach is None or (isinstance(coach, str) and coach.strip()),
            f"invalid coach binding for {team_key}",
        )

    schedule = data.get("schedule")
    _require(isinstance(schedule, Mapping), "schedule required")
    _require(schedule.get("game_id") == data.get("game_id"), "schedule game_id mismatch")

    schedule_teams = {schedule.get("home_team"), schedule.get("away_team")}
    _require(
        schedule_teams == set(team_keys),
        "schedule teams must match primitive teams",
    )

    _require(
        schedule.get("kickoff_utc") == data.get("kickoff_utc"),
        "schedule kickoff must match result kickoff",
    )


def build_primitive_result(
    *,
    season: int,
    week: int,
    game_id: str,
    scenario_id: str,
    simulation_step: int,
    kickoff_utc: str,
    completion_utc: str,
    cutoff_utc: str,
    provenance: str,
    teams: list,
    coach_bindings: Mapping[str, Optional[str]],
    schedule: Mapping[str, Any],
    source_version: str,
    source_hash: str,
    parent_state_id: str,
    parent_state_hash: str,
    transition_version: str,
    observed_at_utc: Optional[str] = None,
    simulated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a deterministic validated primitive result.

    This function does not simulate values. Callers must explicitly supply
    every primitive, including None for unknown values.
    """

    result: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "season": season,
        "week": week,
        "game_id": game_id,
        "scenario_id": scenario_id,
        "simulation_step": simulation_step,
        "kickoff_utc": kickoff_utc,
        "completion_utc": completion_utc,
        "observed_at_utc": observed_at_utc,
        "simulated_at_utc": simulated_at_utc,
        "cutoff_utc": cutoff_utc,
        "provenance": provenance,
        "source_version": source_version,
        "source_hash": source_hash,
        "parent_state_id": parent_state_id,
        "parent_state_hash": parent_state_hash,
        "transition_version": transition_version,
        "teams": deepcopy(teams),
        "coach_bindings": deepcopy(dict(coach_bindings)),
        "schedule": deepcopy(dict(schedule)),
    }

    validate_primitive_result(result)

    result["primitive_result_hash"] = content_hash(result)

    return result
