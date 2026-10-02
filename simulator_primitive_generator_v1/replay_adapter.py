"""
Primitive Generator V1 -> Recursive State Replay V1 adapter.

SHADOW / VALIDATION ONLY.

Converts one validated primitive game result into the transition-event
schema consumed by simulator_recursive_state_replay_v1.

This module does not modify either source object and does not generate
football outcomes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from .core import validate_primitive_result


PRIMITIVE_TO_REPLAY = {
    "points": "points",
    "pass_attempts": "attempts",
    "carries": "carries",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
}


class ReplayAdapterError(ValueError):
    """Raised when a valid primitive result cannot safely map to Replay V1."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ReplayAdapterError(reason)


def to_replay_transition(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Convert one validated primitive result to one Replay V1 transition event.

    OBSERVED:
        observation_time = observed_at_utc

    SIMULATED:
        observation_time = simulated_at_utc
        scenario_version = scenario_id

    Replay uses observation_time as the timestamp at which the event became
    available to the branch. For simulated evidence this is the simulation
    timestamp, never a fabricated historical observation time.
    """

    validate_primitive_result(result)

    provenance = result["provenance"]

    if provenance == "OBSERVED":
        availability_time = result["observed_at_utc"]
    elif provenance == "SIMULATED":
        availability_time = result["simulated_at_utc"]
    else:
        raise ReplayAdapterError(
            "UNSUPPORTED_PROVENANCE"
        )

    _require(
        availability_time is not None,
        "MISSING_AVAILABILITY_TIME",
    )

    replay_teams = []

    for team_row in result["teams"]:
        team = team_row["team"]
        primitives = team_row["primitives"]

        _require(
            team in result["coach_bindings"],
            "MISSING_COACH_BINDING",
        )

        replay_row = {
            "team": team,
            "coach_id": result["coach_bindings"][team],
        }

        for primitive_name, replay_name in (
            PRIMITIVE_TO_REPLAY.items()
        ):
            replay_row[replay_name] = (
                primitives[primitive_name]
            )

        replay_teams.append(replay_row)

    event: Dict[str, Any] = {
        "game_id": result["game_id"],
        "season": result["season"],
        "week": result["week"],
        "completed_at": result["completion_utc"],
        "observation_time": availability_time,
        "source_ref":
            "PRIMITIVE_GAME_STATE_GENERATOR_V1",
        "source_version": result["source_version"],
        "source_hash":
            result["primitive_result_hash"],
        "provenance": provenance,
        "teams": replay_teams,
    }

    if provenance == "SIMULATED":
        event["scenario_version"] = (
            result["scenario_id"]
        )

    return event


def _validate_parent_lineage(
    replay_input: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """
    Verify the parent identity that Replay V1 actually exposes.

    Replay's authoritative parent identity is:
        parent.state_manifest.state_version

    The target context must reference the same version, and the primitive
    result's parent_state_id must bind to it.

    parent_state_hash is intentionally not compared to state_version because
    Replay V1 does not expose an independent parent hash contract here.
    """

    parent = replay_input.get("parent")

    _require(
        isinstance(parent, Mapping),
        "MISSING_REPLAY_PARENT",
    )

    manifest = parent.get("state_manifest")

    _require(
        isinstance(manifest, Mapping),
        "MISSING_REPLAY_PARENT_MANIFEST",
    )

    parent_version = manifest.get(
        "state_version"
    )

    _require(
        isinstance(parent_version, str)
        and parent_version.strip(),
        "MISSING_REPLAY_PARENT_STATE_VERSION",
    )

    context = replay_input.get("context")

    _require(
        isinstance(context, Mapping),
        "MISSING_REPLAY_CONTEXT",
    )

    context_parent = context.get(
        "parent_state_version"
    )

    _require(
        isinstance(context_parent, str)
        and context_parent.strip(),
        "MISSING_CONTEXT_PARENT_STATE_VERSION",
    )

    _require(
        context_parent == parent_version,
        "REPLAY_PARENT_CONTEXT_MISMATCH",
    )

    _require(
        result.get("parent_state_id")
        == parent_version,
        "RESULT_PARENT_STATE_ID_MISMATCH",
    )


def append_transition_to_replay_input(
    replay_input: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Return a deep-copied Replay input with the generated result appended.

    Parent identity is verified before the generated event is admitted.
    Existing bootstrap, transitions, parent, schedule and context are not
    rewritten.
    """

    # Validate the primitive contract first.
    validate_primitive_result(result)

    # Then prove that its declared parent identity belongs to this Replay
    # branch before constructing/appending the transition.
    _validate_parent_lineage(
        replay_input,
        result,
    )

    out = deepcopy(dict(replay_input))
    event = to_replay_transition(result)

    transitions = list(
        out.get("transitions", [])
    )

    _require(
        event["game_id"] not in {
            e.get("game_id")
            for e in (
                list(out.get("bootstrap", []))
                + transitions
            )
        },
        "DUPLICATE_REPLAY_EVENT",
    )

    transitions.append(event)
    out["transitions"] = transitions

    return out
