"""
NFL APP Health Monitor V2 controller.

Architecture:

    V1 isolated read-only CLI
        -> immutable health report
        -> persistent incident reconciliation
        -> repair policy / circuit breaker
        -> optional guarded repair
        -> fresh V1 validation
        -> persistent repair result
        -> controller events

Default mode is DRY RUN.

Production repairs require explicit execute_repairs=True from the caller.
Scheduling and notification delivery are intentionally outside this module.
"""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from .incidents import (
    load_state,
    reconcile,
    save_state,
)
from .repair_policy import (
    AUTO_FIX,
    GUARDED_FIX,
    authorize_incident,
    circuit_state,
    record_repair_attempt,
)
from .repair_runner import (
    RepairRejected,
    execute as execute_repair,
)


ROOT = Path("/home/mwynn/nfl_data_engine").resolve()
PYTHON = ROOT / "venv/bin/python"
STATE_DIR = ROOT / "var/health_monitor"
STATE_FILE = STATE_DIR / "incident_state.json"

CONTROLLER_CONTRACT = "NFL_APP_HEALTH_MONITOR_CONTROLLER_V2"


class ObservationFailure(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def observe_v1(
    *,
    runner: Callable = subprocess.run,
) -> Dict:
    """
    Run V1 in its own process.

    V1 installs its existing filesystem write guard before production
    observation. V2 never imports Production into this writable process.
    """
    result = runner(
        [
            str(PYTHON),
            "-B",
            "-m",
            "health_monitor",
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )

    # V1 health exit codes are meaningful health states, not subprocess
    # execution failures.
    if result.returncode not in {0, 10, 20, 30, 40}:
        raise ObservationFailure(
            "V1_UNEXPECTED_EXIT_CODE"
        )

    try:
        report = json.loads(result.stdout)
    except Exception as exc:
        raise ObservationFailure(
            "V1_INVALID_JSON:" + type(exc).__name__
        )

    if report.get("contract") != "NFL_APP_HEALTH_MONITOR_V1":
        raise ObservationFailure(
            "V1_CONTRACT_MISMATCH"
        )

    return report


def incident_validation(
    original_incident: Dict,
    fresh_report: Dict,
) -> bool:
    """
    A repair validates only when the exact incident condition is absent
    from the fresh V1 report.

    We do not require the entire NFL APP to be globally HEALTHY because
    unrelated owner-action incidents may legitimately remain.
    """
    for check in fresh_report.get("checks", []):
        if check.get("status") == "HEALTHY":
            continue

        if (
            str(check.get("check_id", ""))
            == str(original_incident.get("check_id", ""))
            and str(check.get("scope", ""))
            == str(original_incident.get("scope", ""))
            and str(check.get("reason_code", ""))
            == str(original_incident.get("reason_code", ""))
        ):
            return False

    return True


def _active_incidents(state: Dict) -> List[Dict]:
    rows = [
        incident
        for incident in state.get("incidents", {}).values()
        if incident.get("active", False)
    ]

    return sorted(
        rows,
        key=lambda row: (
            row.get("check_id", ""),
            row.get("scope", ""),
            row.get("reason_code", ""),
        ),
    )


def run_cycle(
    report: Dict,
    state: Dict,
    *,
    now: datetime,
    execute_repairs: bool = False,
    repair_executor: Callable = execute_repair,
    fresh_observer: Callable[[], Dict] | None = None,
) -> Dict:
    """
    Pure controller cycle except for an injected repair executor/observer.

    The caller is responsible for loading/saving state.

    Dry-run:
      - reconcile incidents
      - calculate repair eligibility
      - execute nothing

    Execute mode:
      - only authorized active CRITICAL incidents are considered
      - circuit breaker checked
      - guarded runner invoked
      - attempted repairs consume budget
      - successful command must pass fresh V1 incident validation
    """
    working = copy.deepcopy(state)

    reconciled = reconcile(
        report,
        working,
        now=now,
    )

    working = reconciled["state"]
    events = list(reconciled["events"])
    repair_events = []

    for incident in _active_incidents(working):
        decision = authorize_incident(incident)

        if decision["authorization"] not in {
            AUTO_FIX,
            GUARDED_FIX,
        }:
            continue

        circuit = circuit_state(incident)

        if circuit["open"]:
            repair_events.append({
                "event_type": "REPAIR_BLOCKED",
                "incident_id": incident["incident_id"],
                "repair_id": decision.get("repair_id"),
                "reason": circuit["reason"],
            })
            continue

        if not execute_repairs:
            repair_events.append({
                "event_type": "REPAIR_ELIGIBLE_DRY_RUN",
                "incident_id": incident["incident_id"],
                "repair_id": decision.get("repair_id"),
                "authorization": decision["authorization"],
            })
            continue

        if fresh_observer is None:
            raise ValueError(
                "fresh_observer required when repairs are enabled"
            )

        def validate(_repair_id: str) -> bool:
            fresh = fresh_observer()

            return incident_validation(
                incident,
                fresh,
            )

        try:
            outcome = repair_executor(
                incident,
                validate=validate,
            )

        except RepairRejected as exc:
            repair_events.append({
                "event_type": "REPAIR_BLOCKED",
                "incident_id": incident["incident_id"],
                "repair_id": decision.get("repair_id"),
                "reason": str(exc),
            })
            continue

        status = outcome.get("status")

        if status == "DEFERRED":
            repair_events.append({
                "event_type": "REPAIR_DEFERRED",
                "incident_id": incident["incident_id"],
                "repair_id": outcome.get("repair_id"),
                "reason": outcome.get("reason"),
            })
            continue

        if not outcome.get("attempted", False):
            repair_events.append({
                "event_type": "REPAIR_NOT_ATTEMPTED",
                "incident_id": incident["incident_id"],
                "repair_id": outcome.get("repair_id"),
                "reason": outcome.get("reason"),
            })
            continue

        result = (
            "SUCCEEDED"
            if status == "SUCCEEDED"
            else "FAILED"
        )

        record_repair_attempt(
            incident,
            result=result,
            attempted_at_utc=now.isoformat(),
        )

        repair_events.append({
            "event_type": (
                "REPAIR_SUCCEEDED"
                if result == "SUCCEEDED"
                else "REPAIR_FAILED"
            ),
            "incident_id": incident["incident_id"],
            "repair_id": outcome.get("repair_id"),
            "reason": outcome.get("reason"),
        })

    return {
        "contract": CONTROLLER_CONTRACT,
        "generated_at_utc": now.isoformat(),
        "mode": (
            "EXECUTE_REPAIRS"
            if execute_repairs
            else "DRY_RUN"
        ),
        "health_status": report.get("overall_status"),
        "incident_events": events,
        "repair_events": repair_events,
        "state": working,
    }


def production_cycle(
    *,
    execute_repairs: bool = False,
    state_path: Path = STATE_FILE,
    observer: Callable[[], Dict] = observe_v1,
) -> Dict:
    """
    One controller transaction.

    State is saved atomically after the cycle.
    """
    now = utcnow()
    report = observer()
    state = load_state(state_path)

    result = run_cycle(
        report,
        state,
        now=now,
        execute_repairs=execute_repairs,
        fresh_observer=observer,
    )

    save_state(
        state_path,
        result["state"],
    )

    return result
