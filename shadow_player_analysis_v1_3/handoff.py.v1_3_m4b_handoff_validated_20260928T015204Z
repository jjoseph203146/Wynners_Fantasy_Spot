"""M4B shadow-only orchestration for the V1.3 player analysis pipeline.

This module deliberately contains no claim, revision, identity, temporal, or
corroboration semantics.  It validates the handoff envelope, calls the three
frozen engines, and returns detached audit material.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json

from .claims import build_claims, SAFETY as CLAIM_SAFETY
from .revision_provenance import build_revision_provenance
from .corroboration import build_corroboration, SAFETY as CORROBORATION_SAFETY


ANALYSIS_ONLY = True
PRODUCTION_INFLUENCE = False
SOLVER_INFLUENCE = False
PROJECTION_MUTATION = False
FORECAST_MUTATION = False
AVAILABILITY_AUTHORITY = False
INJURY_AUTHORITY = False
DEPTH_CHART_AUTHORITY = False
AI_ANALYST_INFLUENCE = False
DATABASE_MUTATION = False
CONSENSUS_ENABLED = False

SAFETY = {
    "ANALYSIS_ONLY": ANALYSIS_ONLY,
    "PRODUCTION_INFLUENCE": PRODUCTION_INFLUENCE,
    "SOLVER_INFLUENCE": SOLVER_INFLUENCE,
    "PROJECTION_MUTATION": PROJECTION_MUTATION,
    "FORECAST_MUTATION": FORECAST_MUTATION,
    "AVAILABILITY_AUTHORITY": AVAILABILITY_AUTHORITY,
    "INJURY_AUTHORITY": INJURY_AUTHORITY,
    "DEPTH_CHART_AUTHORITY": DEPTH_CHART_AUTHORITY,
    "AI_ANALYST_INFLUENCE": AI_ANALYST_INFLUENCE,
    "DATABASE_MUTATION": DATABASE_MUTATION,
    "CONSENSUS_ENABLED": CONSENSUS_ENABLED,
}

_REQUIRED = ("source_events", "classified_evidence", "entity_links",
             "quarantine", "source_quality")
_SEQUENCE_FIELDS = frozenset(_REQUIRED[:3] + _REQUIRED[3:4])


def _utc(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MALFORMED_CUTOFF")
    try:
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("MALFORMED_CUTOFF") from exc
    if parsed.tzinfo is None:
        raise ValueError("MALFORMED_CUTOFF")
    return parsed.astimezone(timezone.utc).isoformat()


def _json_copy(value, label):
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("MALFORMED_" + label.upper()) from exc


def _validate_schedule(schedule, cutoff):
    if not isinstance(schedule, dict):
        raise ValueError("MALFORMED_SCHEDULE")
    if schedule.get("authority") != "WFS_SCHEDULE_SNAPSHOT":
        raise ValueError("MALFORMED_SCHEDULE")
    try:
        available = _utc(schedule.get("available_at_utc"))
    except ValueError as exc:
        raise ValueError("MALFORMED_SCHEDULE") from exc
    if available > cutoff:
        raise ValueError("FUTURE_SCHEDULE_AUTHORITY")
    games = schedule.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("MALFORMED_SCHEDULE")
    seen = set()
    for game in games:
        if not isinstance(game, dict):
            raise ValueError("MALFORMED_SCHEDULE")
        game_id = game.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip() or game_id in seen:
            raise ValueError("MALFORMED_SCHEDULE")
        seen.add(game_id)
        try:
            _utc(game.get("kickoff_at_utc"))
        except ValueError as exc:
            raise ValueError("MALFORMED_SCHEDULE") from exc


def _validate_safety(label, value):
    if not isinstance(value, dict):
        raise ValueError("DOWNSTREAM_SAFETY_CONTRACT_VIOLATION:" + label)
    expected = CLAIM_SAFETY if label == "claims" else CORROBORATION_SAFETY
    for key, required in expected.items():
        if value.get(key) is not required:
            raise ValueError("DOWNSTREAM_SAFETY_CONTRACT_VIOLATION:" + label)
    if label == "claims" and "CONSENSUS_ENABLED" in value:
        raise ValueError("DOWNSTREAM_SAFETY_CONTRACT_VIOLATION:claims")


def _validate_v12(value):
    if not isinstance(value, dict):
        raise ValueError("MALFORMED_V1_2_RESULT")
    if any(key not in value for key in _REQUIRED):
        raise ValueError("MISSING_V1_2_COLLECTION")
    for key in _SEQUENCE_FIELDS:
        if not isinstance(value[key], (list, tuple)):
            raise ValueError("WRONG_V1_2_COLLECTION_TYPE:" + key)
    if not isinstance(value["source_quality"], (list, tuple)):
        raise ValueError("WRONG_V1_2_COLLECTION_TYPE:source_quality")
    for key in _REQUIRED:
        if any(not isinstance(row, dict) for row in value[key]):
            raise ValueError("MALFORMED_V1_2_COLLECTION:" + key)
    # Force a complete JSON-compatible detached input before any engine runs.
    return {key: _json_copy(value[key], key) for key in _REQUIRED}


def _claims_shape(result):
    if not isinstance(result, dict) or not isinstance(result.get("claims"), list):
        raise ValueError("UNEXPECTED_CLAIMS_RESULT")
    if result.get("consensus") != []:
        raise ValueError("CONSENSUS_CONTRACT_VIOLATION")
    _validate_safety("claims", result.get("safety"))


def _provenance_shape(result):
    if not isinstance(result, dict):
        raise ValueError("UNEXPECTED_PROVENANCE_RESULT")
    if not isinstance(result.get("revision_provenance"), list) or not isinstance(result.get("event_families"), list):
        raise ValueError("UNEXPECTED_PROVENANCE_RESULT")


def _corroboration_shape(result):
    if not isinstance(result, dict):
        raise ValueError("UNEXPECTED_CORROBORATION_RESULT")
    if not all(isinstance(result.get(key), list) for key in ("groups", "rejected")):
        raise ValueError("UNEXPECTED_CORROBORATION_RESULT")
    if not isinstance(result.get("claim_states"), dict):
        raise ValueError("UNEXPECTED_CORROBORATION_RESULT")
    _validate_safety("corroboration", result.get("safety"))


def build_shadow_handoff(*, v1_2_result, cutoff_utc, schedule):
    """Build one deterministic, detached M4B shadow handoff packet."""
    cutoff = _utc(cutoff_utc)
    inputs = _validate_v12(v1_2_result)
    schedule_copy = _json_copy(schedule, "schedule")
    _validate_schedule(schedule_copy, cutoff)

    claims_result = build_claims(
        source_events=inputs["source_events"],
        classified_evidence=inputs["classified_evidence"],
        entity_links=inputs["entity_links"],
        cutoff_utc=cutoff_utc,
        schedule=schedule_copy,
    )
    _claims_shape(claims_result)
    provenance_result = build_revision_provenance(
        source_events=inputs["source_events"], cutoff_utc=cutoff_utc)
    _provenance_shape(provenance_result)
    corroboration_result = build_corroboration(
        claims_result["claims"], cutoff_utc=cutoff_utc)
    _corroboration_shape(corroboration_result)

    packet = {
        "input_audit": {
            "cutoff_utc": cutoff,
            "source_event_count": len(inputs["source_events"]),
            "classified_evidence_count": len(inputs["classified_evidence"]),
            "entity_link_count": len(inputs["entity_links"]),
            "quarantine_count": len(inputs["quarantine"]),
            "source_quality_count": len(inputs["source_quality"]),
            "schedule_authority": schedule_copy["authority"],
            "schedule_available_at_utc": _utc(schedule_copy["available_at_utc"]),
        },
        "claims": deepcopy(claims_result),
        "revision_provenance": deepcopy(provenance_result),
        "corroboration": deepcopy(corroboration_result),
        "quarantine": deepcopy(inputs["quarantine"]),
        "source_quality": deepcopy(inputs["source_quality"]),
        "safety": dict(SAFETY),
    }
    return _json_copy(packet, "handoff_packet")
