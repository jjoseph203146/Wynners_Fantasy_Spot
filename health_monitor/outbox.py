"""
NFL APP Health Monitor V2 durable notification outbox.

Provider-neutral by design.

Responsibilities:
- translate controller/incident events into owner notifications
- deterministic notification identity
- suppress duplicates
- persist pending/delivered/failed notification state
- record delivery attempts
- preserve recovery notifications
- isolate notification failure from repair/production behavior

This module:
- does NOT send email/SMS/push
- does NOT execute repairs
- does NOT invoke production
- does NOT modify cron
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


OUTBOX_CONTRACT = "NFL_APP_HEALTH_MONITOR_OUTBOX_V1"

PENDING = "PENDING"
DELIVERED = "DELIVERED"
FAILED = "FAILED"

VALID_DELIVERY_STATES = {
    PENDING,
    DELIVERED,
    FAILED,
}


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)

    if value.tzinfo is None:
        raise ValueError(
            "Outbox timestamps must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return _utc(value).isoformat()


def empty_outbox() -> Dict[str, Any]:
    return {
        "contract": OUTBOX_CONTRACT,
        "messages": {},
    }


def load_outbox(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return empty_outbox()

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if data.get("contract") != OUTBOX_CONTRACT:
        raise ValueError(
            "Unknown Health Monitor outbox contract"
        )

    messages = data.get("messages")

    if not isinstance(messages, dict):
        raise ValueError(
            "Invalid Health Monitor outbox payload"
        )

    return data


def save_outbox(
    path: Path,
    outbox: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        outbox,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"

    fd, temp_name = tempfile.mkstemp(
        prefix=".outbox.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temp_name,
            path,
        )

    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def notification_key(
    *,
    incident_id: str,
    event_type: str,
    status: str,
) -> str:
    """
    Stable deduplication identity.

    One incident may legitimately produce:
      OPENED
      ESCALATED
      RECOVERED

    Escalation status is part of identity so CRITICAL and later
    OWNER_ACTION_REQUIRED alerts are distinct.
    """
    raw = json.dumps(
        {
            "incident_id": str(incident_id),
            "event_type": str(event_type),
            "status": str(status),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:24]


def _queue(
    outbox: Dict[str, Any],
    *,
    incident_id: str,
    event_type: str,
    status: str,
    check_id: str,
    scope: str,
    reason_code: str,
    now: datetime,
) -> Dict[str, Any] | None:
    key = notification_key(
        incident_id=incident_id,
        event_type=event_type,
        status=status,
    )

    messages = outbox["messages"]

    if key in messages:
        return None

    row = {
        "notification_id": key,
        "incident_id": incident_id,
        "event_type": event_type,
        "status": status,
        "check_id": check_id,
        "scope": scope,
        "reason_code": reason_code,
        "delivery_state": PENDING,
        "created_at_utc": _stamp(now),
        "delivered_at_utc": None,
        "last_attempt_at_utc": None,
        "delivery_attempts": 0,
        "last_error_type": None,
    }

    messages[key] = row

    return row


def enqueue_incident_events(
    outbox: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    *,
    now: datetime,
) -> List[Dict[str, Any]]:
    """
    Queue owner-facing incident events.

    OPENED:
      only CRITICAL / OWNER_ACTION_REQUIRED

    ESCALATED:
      always queue when severity reached an alertable state

    RECOVERED:
      queue once for previously alerted incidents
    """
    queued = []

    for event in events:
        event_type = str(
            event.get("event_type", "")
        )
        status = str(
            event.get("status", "")
        )

        should_queue = (
            (
                event_type == "OPENED"
                and status in {
                    "CRITICAL",
                    "OWNER_ACTION_REQUIRED",
                }
            )
            or event_type == "ESCALATED"
            or event_type == "RECOVERED"
        )

        if not should_queue:
            continue

        row = _queue(
            outbox,
            incident_id=str(
                event.get("incident_id", "")
            ),
            event_type=event_type,
            status=status,
            check_id=str(
                event.get("check_id", "")
            ),
            scope=str(
                event.get("scope", "")
            ),
            reason_code=str(
                event.get("reason_code", "")
            ),
            now=now,
        )

        if row is not None:
            queued.append(row)

    return queued


def enqueue_repair_events(
    outbox: Dict[str, Any],
    events: Iterable[Dict[str, Any]],
    *,
    incident_lookup: Dict[str, Dict[str, Any]],
    now: datetime,
) -> List[Dict[str, Any]]:
    """
    Queue only repair conditions requiring owner attention.

    Successful repair:
        silent

    Deferred repair:
        silent for now; controller will see it again later

    Failed repair:
        alert

    Circuit blocked after an attempted repair:
        alert

    Dry-run eligibility:
        silent
    """
    queued = []

    for event in events:
        event_type = str(
            event.get("event_type", "")
        )

        incident_id = str(
            event.get("incident_id", "")
        )

        incident = incident_lookup.get(
            incident_id,
            {},
        )

        should_queue = False
        status = str(
            incident.get("status", "CRITICAL")
        )

        if event_type == "REPAIR_FAILED":
            should_queue = True

        elif (
            event_type == "REPAIR_BLOCKED"
            and int(
                incident.get(
                    "repair_attempts",
                    0,
                )
            ) > 0
        ):
            should_queue = True

        if not should_queue:
            continue

        row = _queue(
            outbox,
            incident_id=incident_id,
            event_type=event_type,
            status=status,
            check_id=str(
                incident.get("check_id", "")
            ),
            scope=str(
                incident.get("scope", "")
            ),
            reason_code=str(
                event.get(
                    "reason",
                    incident.get(
                        "reason_code",
                        "",
                    ),
                )
            ),
            now=now,
        )

        if row is not None:
            queued.append(row)

    return queued


def pending_messages(
    outbox: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in outbox.get(
            "messages",
            {},
        ).values()
        if row.get("delivery_state") in {
            PENDING,
            FAILED,
        }
    ]

    return sorted(
        rows,
        key=lambda row: (
            row.get("created_at_utc", ""),
            row.get("notification_id", ""),
        ),
    )


def record_delivery(
    message: Dict[str, Any],
    *,
    succeeded: bool,
    attempted_at: datetime,
    error_type: str | None = None,
) -> Dict[str, Any]:
    """
    Record provider result only.

    No exception from a provider is stored verbatim; only sanitized
    exception type may be retained.
    """
    attempted_at = _utc(attempted_at)

    message["delivery_attempts"] = int(
        message.get(
            "delivery_attempts",
            0,
        )
    ) + 1

    message["last_attempt_at_utc"] = (
        attempted_at.isoformat()
    )

    if succeeded:
        message["delivery_state"] = DELIVERED
        message["delivered_at_utc"] = (
            attempted_at.isoformat()
        )
        message["last_error_type"] = None

    else:
        message["delivery_state"] = FAILED
        message["last_error_type"] = (
            str(error_type)
            if error_type
            else "UNKNOWN_DELIVERY_FAILURE"
        )

    return message


def deliver_pending(
    outbox: Dict[str, Any],
    *,
    sender,
    now: datetime,
) -> List[Dict[str, Any]]:
    """
    Attempt pending delivery through an injected provider.

    Provider failure is contained here and returned as delivery state.
    It cannot invoke repair behavior.
    """
    results = []

    for message in pending_messages(outbox):
        try:
            sender(message)

        except Exception as exc:
            record_delivery(
                message,
                succeeded=False,
                attempted_at=now,
                error_type=type(exc).__name__,
            )

            results.append({
                "notification_id":
                    message["notification_id"],
                "delivery_state": FAILED,
                "error_type":
                    type(exc).__name__,
            })

            continue

        record_delivery(
            message,
            succeeded=True,
            attempted_at=now,
        )

        results.append({
            "notification_id":
                message["notification_id"],
            "delivery_state": DELIVERED,
        })

    return results
