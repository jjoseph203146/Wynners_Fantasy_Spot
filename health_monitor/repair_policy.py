"""
NFL APP Health Monitor V2 repair authorization policy.

IMPORTANT:
This module decides repair AUTHORIZATION only.
It does not execute shell commands, services, builders, publishers,
solvers, database writes, or production mutations.

Unknown conditions always fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


AUTO_FIX = "AUTO_FIX"
GUARDED_FIX = "GUARDED_FIX"
OWNER_ACTION_REQUIRED = "OWNER_ACTION_REQUIRED"

AUTHORIZATIONS = {
    AUTO_FIX,
    GUARDED_FIX,
    OWNER_ACTION_REQUIRED,
}


@dataclass(frozen=True)
class RepairPolicy:
    authorization: str
    repair_id: str | None
    max_attempts: int
    requires_validation: bool
    requires_backup: bool
    description: str


# Explicit allowlist.
#
# Merely appearing here does NOT execute a repair. A later repair runner must
# separately implement and allowlist the corresponding repair_id.
POLICIES: Dict[str, RepairPolicy] = {
    "UPDATER_FAILED": RepairPolicy(
        authorization=GUARDED_FIX,
        repair_id="RUN_UPDATER_ONCE",
        max_attempts=1,
        requires_validation=True,
        requires_backup=False,
        description="Retry the established updater once after safety gates pass.",
    ),

    "UPDATER_MISSED_EXPECTED_RUN": RepairPolicy(
        authorization=GUARDED_FIX,
        repair_id="RUN_UPDATER_ONCE",
        max_attempts=1,
        requires_validation=True,
        requires_backup=False,
        description="Run one missed updater cycle after safety gates pass.",
    ),

    "APP_UNAVAILABLE": RepairPolicy(
        authorization=GUARDED_FIX,
        repair_id="RESTART_WFS_SERVICE_ONCE",
        max_attempts=1,
        requires_validation=True,
        requires_backup=False,
        description="Restart only the allowlisted Streamlit application service once.",
    ),
}


def policy_for(reason_code: str) -> RepairPolicy:
    """
    Return explicit policy or fail closed.

    Anything not positively allowlisted requires the owner.
    """
    return POLICIES.get(
        str(reason_code),
        RepairPolicy(
            authorization=OWNER_ACTION_REQUIRED,
            repair_id=None,
            max_attempts=0,
            requires_validation=True,
            requires_backup=True,
            description="No automatic repair authority exists for this condition.",
        ),
    )


def authorize_incident(incident: dict) -> dict:
    """
    Produce a deterministic authorization decision.

    WARNING incidents never trigger repair.
    OWNER_ACTION_REQUIRED monitor findings never trigger repair even if their
    reason code were accidentally allowlisted.
    """
    status = str(incident.get("status", ""))
    reason = str(incident.get("reason_code", ""))

    policy = policy_for(reason)

    if status == "WARNING":
        return {
            "authorization": "NO_ACTION",
            "repair_id": None,
            "max_attempts": 0,
            "reason": "WARNING_NOT_REPAIRABLE",
        }

    if status == "OWNER_ACTION_REQUIRED":
        return {
            "authorization": OWNER_ACTION_REQUIRED,
            "repair_id": None,
            "max_attempts": 0,
            "reason": "MONITOR_REQUIRES_OWNER",
        }

    if status != "CRITICAL":
        return {
            "authorization": "NO_ACTION",
            "repair_id": None,
            "max_attempts": 0,
            "reason": "STATUS_NOT_REPAIRABLE",
        }

    return {
        "authorization": policy.authorization,
        "repair_id": policy.repair_id,
        "max_attempts": policy.max_attempts,
        "requires_validation": policy.requires_validation,
        "requires_backup": policy.requires_backup,
        "reason": (
            "EXPLICIT_REPAIR_ALLOWLIST"
            if policy.authorization != OWNER_ACTION_REQUIRED
            else "NO_REPAIR_AUTHORITY"
        ),
    }


def repair_allowed(incident: dict) -> bool:
    decision = authorize_incident(incident)

    return decision["authorization"] in {
        AUTO_FIX,
        GUARDED_FIX,
    }


def circuit_state(incident: dict) -> dict:
    """
    Determine whether another repair attempt is permitted.

    Current safety contract:
      - at most one attempt per incident
      - any failed attempt opens the circuit
      - successful repair also prevents another attempt for the same incident
      - recovery/new incident is required before future repair eligibility
    """
    decision = authorize_incident(incident)

    if decision["authorization"] not in {AUTO_FIX, GUARDED_FIX}:
        return {
            "open": True,
            "reason": "REPAIR_NOT_AUTHORIZED",
            "attempts": int(incident.get("repair_attempts", 0)),
        }

    attempts = int(incident.get("repair_attempts", 0))
    max_attempts = int(decision["max_attempts"])

    if incident.get("last_repair_result") == "FAILED":
        return {
            "open": True,
            "reason": "PREVIOUS_REPAIR_FAILED",
            "attempts": attempts,
        }

    if attempts >= max_attempts:
        return {
            "open": True,
            "reason": "REPAIR_BUDGET_EXHAUSTED",
            "attempts": attempts,
        }

    return {
        "open": False,
        "reason": "REPAIR_PERMITTED",
        "attempts": attempts,
    }


def record_repair_attempt(
    incident: dict,
    *,
    result: str,
    attempted_at_utc: str,
) -> dict:
    """
    Mutate incident STATE only.

    This does not perform the repair itself.
    """
    if result not in {"SUCCEEDED", "FAILED"}:
        raise ValueError("Unknown repair result")

    incident["repair_attempts"] = int(
        incident.get("repair_attempts", 0)
    ) + 1
    incident["last_repair_result"] = result
    incident["last_repair_attempt_utc"] = attempted_at_utc

    return incident
