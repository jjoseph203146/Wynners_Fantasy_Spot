"""Synthetic tests for Health Monitor V2 durable outbox."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from .outbox import (
    DELIVERED,
    FAILED,
    OUTBOX_CONTRACT,
    empty_outbox,
    enqueue_incident_events,
    enqueue_repair_events,
    deliver_pending,
    load_outbox,
    pending_messages,
    save_outbox,
)


NOW = datetime(
    2030,
    10,
    6,
    16,
    0,
    tzinfo=timezone.utc,
)


def event(
    event_type,
    *,
    status="CRITICAL",
    incident_id="abc",
    reason="UPDATER_FAILED",
):
    return {
        "event_type": event_type,
        "incident_id": incident_id,
        "check_id": "updater",
        "scope": "",
        "reason_code": reason,
        "status": status,
        "occurred_at_utc": NOW.isoformat(),
    }


def incident(
    *,
    incident_id="abc",
    attempts=0,
    status="CRITICAL",
):
    return {
        "incident_id": incident_id,
        "check_id": "updater",
        "scope": "",
        "reason_code": "UPDATER_FAILED",
        "status": status,
        "repair_attempts": attempts,
    }


class OutboxTests(unittest.TestCase):

    def test_critical_opened_queues(self):
        outbox = empty_outbox()

        rows = enqueue_incident_events(
            outbox,
            [event("OPENED")],
            now=NOW,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            len(pending_messages(outbox)),
            1,
        )

    def test_owner_opened_queues(self):
        outbox = empty_outbox()

        rows = enqueue_incident_events(
            outbox,
            [
                event(
                    "OPENED",
                    status="OWNER_ACTION_REQUIRED",
                )
            ],
            now=NOW,
        )

        self.assertEqual(len(rows), 1)

    def test_duplicate_opened_suppressed(self):
        outbox = empty_outbox()
        source = [event("OPENED")]

        first = enqueue_incident_events(
            outbox,
            source,
            now=NOW,
        )

        second = enqueue_incident_events(
            outbox,
            source,
            now=NOW,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(
            len(outbox["messages"]),
            1,
        )

    def test_escalation_is_distinct(self):
        outbox = empty_outbox()

        enqueue_incident_events(
            outbox,
            [event("OPENED")],
            now=NOW,
        )

        rows = enqueue_incident_events(
            outbox,
            [
                event(
                    "ESCALATED",
                    status="OWNER_ACTION_REQUIRED",
                )
            ],
            now=NOW,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            len(outbox["messages"]),
            2,
        )

    def test_recovery_queues_once(self):
        outbox = empty_outbox()

        rows = enqueue_incident_events(
            outbox,
            [event("RECOVERED")],
            now=NOW,
        )

        self.assertEqual(len(rows), 1)

        duplicate = enqueue_incident_events(
            outbox,
            [event("RECOVERED")],
            now=NOW,
        )

        self.assertEqual(duplicate, [])

    def test_successful_repair_is_silent(self):
        outbox = empty_outbox()

        rows = enqueue_repair_events(
            outbox,
            [{
                "event_type": "REPAIR_SUCCEEDED",
                "incident_id": "abc",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_VALIDATED",
            }],
            incident_lookup={
                "abc": incident(attempts=1),
            },
            now=NOW,
        )

        self.assertEqual(rows, [])

    def test_failed_repair_queues(self):
        outbox = empty_outbox()

        rows = enqueue_repair_events(
            outbox,
            [{
                "event_type": "REPAIR_FAILED",
                "incident_id": "abc",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_COMMAND_FAILED",
            }],
            incident_lookup={
                "abc": incident(attempts=1),
            },
            now=NOW,
        )

        self.assertEqual(len(rows), 1)

    def test_deferred_repair_is_silent(self):
        outbox = empty_outbox()

        rows = enqueue_repair_events(
            outbox,
            [{
                "event_type": "REPAIR_DEFERRED",
                "incident_id": "abc",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "LIVE_CONCURRENCY_ACTIVE",
            }],
            incident_lookup={
                "abc": incident(attempts=0),
            },
            now=NOW,
        )

        self.assertEqual(rows, [])

    def test_blocked_after_attempt_queues(self):
        outbox = empty_outbox()

        rows = enqueue_repair_events(
            outbox,
            [{
                "event_type": "REPAIR_BLOCKED",
                "incident_id": "abc",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "PREVIOUS_REPAIR_FAILED",
            }],
            incident_lookup={
                "abc": incident(attempts=1),
            },
            now=NOW,
        )

        self.assertEqual(len(rows), 1)

    def test_blocked_without_attempt_is_silent(self):
        outbox = empty_outbox()

        rows = enqueue_repair_events(
            outbox,
            [{
                "event_type": "REPAIR_BLOCKED",
                "incident_id": "abc",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_NOT_AUTHORIZED",
            }],
            incident_lookup={
                "abc": incident(attempts=0),
            },
            now=NOW,
        )

        self.assertEqual(rows, [])

    def test_delivery_success(self):
        outbox = empty_outbox()

        enqueue_incident_events(
            outbox,
            [event("OPENED")],
            now=NOW,
        )

        sent = []

        results = deliver_pending(
            outbox,
            sender=lambda message:
                sent.append(
                    message["notification_id"]
                ),
            now=NOW,
        )

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            results[0]["delivery_state"],
            DELIVERED,
        )
        self.assertEqual(
            pending_messages(outbox),
            [],
        )

    def test_delivery_failure_is_contained(self):
        outbox = empty_outbox()

        enqueue_incident_events(
            outbox,
            [event("OPENED")],
            now=NOW,
        )

        def fail(_message):
            raise RuntimeError(
                "provider unavailable"
            )

        results = deliver_pending(
            outbox,
            sender=fail,
            now=NOW,
        )

        self.assertEqual(
            results[0]["delivery_state"],
            FAILED,
        )

        pending = pending_messages(outbox)

        self.assertEqual(len(pending), 1)
        self.assertEqual(
            pending[0]["last_error_type"],
            "RuntimeError",
        )

    def test_persistence_survives_restart(self):
        outbox = empty_outbox()

        enqueue_incident_events(
            outbox,
            [event("OPENED")],
            now=NOW,
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outbox.json"

            save_outbox(
                path,
                outbox,
            )

            loaded = load_outbox(path)

        self.assertEqual(
            loaded["contract"],
            OUTBOX_CONTRACT,
        )
        self.assertEqual(
            len(loaded["messages"]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
