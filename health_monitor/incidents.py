"""
NFL APP Health Monitor V2 incident-state engine.

This module does NOT inspect or modify production.

Its only writable authority is the caller-supplied Health Monitor state
directory, normally:

    var/health_monitor/

Responsibilities:
- persistent incident state
- deterministic incident identity
- duplicate alert suppression
- severity escalation detection
- one-time recovery detection
- atomic state persistence

Repair execution and notification delivery are intentionally NOT implemented
here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


STATUSES = (
    "HEALTHY",
    "WARNING",
    "CRITICAL",
    "OWNER_ACTION_REQUIRED",
)

SEVERITY = {status: index for index, status in enumerate(STATUSES)}

ALERTABLE = {
    "CRITICAL",
    "OWNER_ACTION_REQUIRED",
}

STATE_CONTRACT = "NFL_APP_HEALTH_MONITOR_INCIDENT_STATE_V1"
EVENT_CONTRACT = "NFL_APP_HEALTH_MONITOR_INCIDENT_EVENTS_V1"


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)

    if value.tzinfo is None:
        raise ValueError("Incident timestamps must be timezone-aware")

    return value.astimezone(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return _utc(value).isoformat()


def incident_key(check: Dict[str, Any]) -> str:
    """
    Stable identity for one monitor condition.

    reason_code is intentionally included. If one scope changes from one
    failure reason to a materially different failure reason, that is a new
    incident rather than silently mutating the old incident.
    """
    payload = {
        "check_id": str(check.get("check_id", "")),
        "scope": str(check.get("scope", "")),
        "reason_code": str(check.get("reason_code", "")),
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:24]


def empty_state() -> Dict[str, Any]:
    return {
        "contract": STATE_CONTRACT,
        "incidents": {},
    }


def load_state(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return empty_state()

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("contract") != STATE_CONTRACT:
        raise ValueError("Unknown Health Monitor incident-state contract")

    incidents = data.get("incidents")

    if not isinstance(incidents, dict):
        raise ValueError("Invalid Health Monitor incident-state payload")

    return data


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """
    Atomically replace the state file.

    This function may write ONLY beneath the explicitly supplied state
    directory. It has no production-data authority.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        state,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"

    fd, temp_name = tempfile.mkstemp(
        prefix=".incident_state.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, path)

    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _new_incident(
    key: str,
    check: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    timestamp = _stamp(now)

    return {
        "incident_id": key,
        "check_id": str(check.get("check_id", "")),
        "scope": str(check.get("scope", "")),
        "reason_code": str(check.get("reason_code", "")),
        "status": str(check["status"]),
        "first_seen_utc": timestamp,
        "last_seen_utc": timestamp,
        "occurrences": 1,
        "active": True,
        "alerted_status": None,
        "resolved_at_utc": None,
    }


def _event(
    event_type: str,
    incident: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    return {
        "event_type": event_type,
        "incident_id": incident["incident_id"],
        "check_id": incident["check_id"],
        "scope": incident["scope"],
        "reason_code": incident["reason_code"],
        "status": incident["status"],
        "occurred_at_utc": _stamp(now),
    }


def _incident_checks(checks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return V1 findings that should create persistent V2 incidents.

    Narrow dependency contract:
    QUALIFIED_STARTER_UNAVAILABLE for TEAM is a downstream symptom when the
    same immutable report already contains ROTOWIRE_STARTER_BLOCKED for TEAM.

    This does not mutate or suppress the V1 health report. It only prevents
    one known root condition from creating two actionable V2 incidents.
    """
    rows = list(checks)

    blocked_teams = {
        str(row.get("scope", "")).strip()
        for row in rows
        if row.get("status") != "HEALTHY"
        and row.get("check_id") == "starter"
        and row.get("reason_code") == "ROTOWIRE_STARTER_BLOCKED"
        and str(row.get("scope", "")).strip()
    }

    result = []

    for row in rows:
        if (
            row.get("status") != "HEALTHY"
            and row.get("check_id") == "qb_invariant"
            and row.get("reason_code") == "QUALIFIED_STARTER_UNAVAILABLE"
        ):
            scope = str(row.get("scope", "")).strip()
            team = scope.rsplit("/", 1)[-1] if "/" in scope else ""

            if team and team in blocked_teams:
                continue

        result.append(row)

    return result


def reconcile(
    report: Dict[str, Any],
    state: Dict[str, Any],
    now: datetime | None = None,
) -> Dict[str, Any]:
    """
    Reconcile one immutable V1 health report against persistent V2 state.

    Returns:
        {
          contract,
          state,
          events
        }

    Event types:
      OPENED
      ESCALATED
      RECOVERED

    WARNING conditions are persisted for history but do not emit alert events.
    HEALTHY checks do not become incidents.
    """
    now = _utc(now)

    if report.get("contract") != "NFL_APP_HEALTH_MONITOR_V1":
        raise ValueError("Unsupported monitor report contract")

    if state.get("contract") != STATE_CONTRACT:
        raise ValueError("Unsupported incident-state contract")

    incidents = state.setdefault("incidents", {})
    seen = set()
    events: List[Dict[str, Any]] = []

    checks: Iterable[Dict[str, Any]] = _incident_checks(
        report.get("checks", [])
    )

    for check in checks:
        status = check.get("status")

        if status not in STATUSES:
            raise ValueError("Unknown monitor status")

        if status == "HEALTHY":
            continue

        key = incident_key(check)
        seen.add(key)

        previous = incidents.get(key)

        if previous is None or not previous.get("active", False):
            incident = _new_incident(key, check, now)
            incidents[key] = incident

            if status in ALERTABLE:
                events.append(_event("OPENED", incident, now))
                incident["alerted_status"] = status

            continue

        old_status = previous["status"]

        previous["last_seen_utc"] = _stamp(now)
        previous["occurrences"] = int(previous.get("occurrences", 0)) + 1
        previous["status"] = status
        previous["active"] = True
        previous["resolved_at_utc"] = None

        old_alerted = previous.get("alerted_status")

        if status in ALERTABLE:
            should_alert = old_alerted is None

            if old_alerted in SEVERITY:
                should_alert = SEVERITY[status] > SEVERITY[old_alerted]

            if should_alert:
                event_type = (
                    "ESCALATED"
                    if old_alerted is not None
                    or SEVERITY[status] > SEVERITY.get(old_status, -1)
                    else "OPENED"
                )
                events.append(_event(event_type, previous, now))
                previous["alerted_status"] = status

    # Any previously active condition absent from the new non-healthy finding
    # set has recovered. Recovery is emitted exactly once because active=False
    # is persisted immediately.
    for key, incident in incidents.items():
        if not incident.get("active", False):
            continue

        if key in seen:
            continue

        was_alerted = incident.get("alerted_status") in ALERTABLE

        incident["active"] = False
        incident["resolved_at_utc"] = _stamp(now)
        incident["last_seen_utc"] = _stamp(now)

        if was_alerted:
            events.append(_event("RECOVERED", incident, now))

    events.sort(
        key=lambda event: (
            event["event_type"],
            event["check_id"],
            event["scope"],
            event["reason_code"],
        )
    )

    return {
        "contract": EVENT_CONTRACT,
        "state": state,
        "events": events,
    }


def process_report(
    report: Dict[str, Any],
    state_path: Path,
    now: datetime | None = None,
) -> Dict[str, Any]:
    """
    Convenience transaction for a controller:

        load -> reconcile -> atomic save

    Notification delivery is deliberately separate. A later outbox layer will
    prevent notification failure from corrupting incident state.
    """
    state_path = Path(state_path)
    state = load_state(state_path)
    result = reconcile(report, state, now=now)
    save_state(state_path, result["state"])
    return result
