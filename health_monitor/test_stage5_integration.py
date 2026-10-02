"""
Synthetic controller -> outbox integration tests.

No production observer, repair command, or external notification provider
is used.
"""

import unittest
from datetime import datetime, timezone

from .controller import run_cycle
from .incidents import empty_state
from .outbox import (
    empty_outbox,
    enqueue_incident_events,
    enqueue_repair_events,
)


NOW = datetime(
    2030,
    10,
    6,
    16,
    0,
    tzinfo=timezone.utc,
)


def check(status, reason):
    return {
        "check_id": "updater",
        "status": status,
        "reason_code": reason,
        "scope": "",
        "summary": reason.lower(),
        "evidence": {},
    }


def report(*checks):
    order = {
        "HEALTHY": 0,
        "WARNING": 1,
        "CRITICAL": 2,
        "OWNER_ACTION_REQUIRED": 3,
    }

    return {
        "contract": "NFL_APP_HEALTH_MONITOR_V1",
        "generated_at_utc": NOW.isoformat(),
        "overall_status": max(
            (
                row["status"]
                for row in checks
            ),
            key=lambda value: order[value],
            default="HEALTHY",
        ),
        "summary_counts": {
            status: sum(
                row["status"] == status
                for row in checks
            )
            for status in order
        },
        "checks": list(checks),
    }


class Stage5IntegrationTests(unittest.TestCase):

    def test_controller_opened_event_enters_outbox(self):
        result = run_cycle(
            report(
                check(
                    "CRITICAL",
                    "UPDATER_FAILED",
                )
            ),
            empty_state(),
            now=NOW,
            execute_repairs=False,
        )

        outbox = empty_outbox()

        queued = enqueue_incident_events(
            outbox,
            result["incident_events"],
            now=NOW,
        )

        self.assertEqual(len(queued), 1)

    def test_successful_repair_generates_no_repair_alert(self):
        def repair(
            incident,
            *,
            validate,
        ):
            self.assertTrue(
                validate("RUN_UPDATER_ONCE")
            )

            return {
                "status": "SUCCEEDED",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_VALIDATED",
                "attempted": True,
            }

        result = run_cycle(
            report(
                check(
                    "CRITICAL",
                    "UPDATER_FAILED",
                )
            ),
            empty_state(),
            now=NOW,
            execute_repairs=True,
            repair_executor=repair,
            fresh_observer=lambda: report(
                check(
                    "HEALTHY",
                    "UPDATER_COMPLETED",
                )
            ),
        )

        outbox = empty_outbox()

        lookup = {
            row["incident_id"]: row
            for row in result[
                "state"
            ]["incidents"].values()
        }

        queued = enqueue_repair_events(
            outbox,
            result["repair_events"],
            incident_lookup=lookup,
            now=NOW,
        )

        self.assertEqual(queued, [])

    def test_failed_repair_generates_owner_alert(self):
        def repair(
            incident,
            *,
            validate,
        ):
            return {
                "status": "FAILED",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_COMMAND_FAILED",
                "attempted": True,
            }

        result = run_cycle(
            report(
                check(
                    "CRITICAL",
                    "UPDATER_FAILED",
                )
            ),
            empty_state(),
            now=NOW,
            execute_repairs=True,
            repair_executor=repair,
            fresh_observer=lambda: report(),
        )

        outbox = empty_outbox()

        lookup = {
            row["incident_id"]: row
            for row in result[
                "state"
            ]["incidents"].values()
        }

        queued = enqueue_repair_events(
            outbox,
            result["repair_events"],
            incident_lookup=lookup,
            now=NOW,
        )

        self.assertEqual(len(queued), 1)


if __name__ == "__main__":
    unittest.main()
