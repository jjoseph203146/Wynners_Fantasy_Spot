"""Synthetic tests for Health Monitor V2 controller."""

import unittest
from datetime import datetime, timezone

from .controller import (
    incident_validation,
    run_cycle,
)
from .incidents import empty_state
from .repair_runner import RepairRejected


NOW = datetime(
    2030,
    10,
    6,
    16,
    0,
    tzinfo=timezone.utc,
)


def check(
    status,
    reason,
    check_id="updater",
    scope="",
):
    return {
        "check_id": check_id,
        "status": status,
        "reason_code": reason,
        "scope": scope,
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

    overall = max(
        (
            row["status"]
            for row in checks
        ),
        key=lambda value: order[value],
        default="HEALTHY",
    )

    return {
        "contract": "NFL_APP_HEALTH_MONITOR_V1",
        "generated_at_utc": NOW.isoformat(),
        "overall_status": overall,
        "summary_counts": {
            status: sum(
                row["status"] == status
                for row in checks
            )
            for status in order
        },
        "checks": list(checks),
    }


class ControllerTests(unittest.TestCase):

    def test_dry_run_never_calls_repair(self):
        calls = []

        def forbidden(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError(
                "repair executed in dry-run"
            )

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
            repair_executor=forbidden,
        )

        self.assertEqual(
            result["mode"],
            "DRY_RUN",
        )

        self.assertEqual(calls, [])

        self.assertEqual(
            result["repair_events"][0]["event_type"],
            "REPAIR_ELIGIBLE_DRY_RUN",
        )

    def test_owner_action_never_repairs(self):
        calls = []

        def forbidden(*args, **kwargs):
            calls.append(1)
            raise AssertionError

        result = run_cycle(
            report(
                check(
                    "OWNER_ACTION_REQUIRED",
                    "STARTER_AUTHORITY_CONFLICT",
                    check_id="starter",
                    scope="WAS",
                )
            ),
            empty_state(),
            now=NOW,
            execute_repairs=True,
            repair_executor=forbidden,
            fresh_observer=lambda: report(),
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            result["repair_events"],
            [],
        )

    def test_successful_repair_recorded(self):
        def repair_executor(
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
            repair_executor=repair_executor,
            fresh_observer=lambda: report(
                check(
                    "HEALTHY",
                    "UPDATER_COMPLETED",
                )
            ),
        )

        event = result["repair_events"][0]

        self.assertEqual(
            event["event_type"],
            "REPAIR_SUCCEEDED",
        )

        incident = next(
            iter(
                result["state"]["incidents"].values()
            )
        )

        self.assertEqual(
            incident["repair_attempts"],
            1,
        )
        self.assertEqual(
            incident["last_repair_result"],
            "SUCCEEDED",
        )

    def test_failed_repair_recorded(self):
        def repair_executor(
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
            repair_executor=repair_executor,
            fresh_observer=lambda: report(),
        )

        self.assertEqual(
            result["repair_events"][0]["event_type"],
            "REPAIR_FAILED",
        )

        incident = next(
            iter(
                result["state"]["incidents"].values()
            )
        )

        self.assertEqual(
            incident["last_repair_result"],
            "FAILED",
        )

    def test_deferred_does_not_consume_budget(self):
        def repair_executor(
            incident,
            *,
            validate,
        ):
            return {
                "status": "DEFERRED",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "LIVE_CONCURRENCY_ACTIVE",
                "attempted": False,
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
            repair_executor=repair_executor,
            fresh_observer=lambda: report(),
        )

        self.assertEqual(
            result["repair_events"][0]["event_type"],
            "REPAIR_DEFERRED",
        )

        incident = next(
            iter(
                result["state"]["incidents"].values()
            )
        )

        self.assertEqual(
            incident.get("repair_attempts", 0),
            0,
        )

    def test_restart_persistence_blocks_second_attempt(self):
        calls = []

        def repair_executor(
            incident,
            *,
            validate,
        ):
            calls.append(1)

            return {
                "status": "FAILED",
                "repair_id": "RUN_UPDATER_ONCE",
                "reason": "REPAIR_COMMAND_FAILED",
                "attempted": True,
            }

        first = run_cycle(
            report(
                check(
                    "CRITICAL",
                    "UPDATER_FAILED",
                )
            ),
            empty_state(),
            now=NOW,
            execute_repairs=True,
            repair_executor=repair_executor,
            fresh_observer=lambda: report(),
        )

        second = run_cycle(
            report(
                check(
                    "CRITICAL",
                    "UPDATER_FAILED",
                )
            ),
            first["state"],
            now=NOW,
            execute_repairs=True,
            repair_executor=repair_executor,
            fresh_observer=lambda: report(),
        )

        self.assertEqual(
            len(calls),
            1,
        )

        self.assertEqual(
            second["repair_events"][0]["event_type"],
            "REPAIR_BLOCKED",
        )

    def test_repair_rejection_is_safe(self):
        def repair_executor(
            incident,
            *,
            validate,
        ):
            raise RepairRejected(
                "SYNTHETIC_REJECTION"
            )

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
            repair_executor=repair_executor,
            fresh_observer=lambda: report(),
        )

        self.assertEqual(
            result["repair_events"][0]["event_type"],
            "REPAIR_BLOCKED",
        )

    def test_validation_exact_incident_absent(self):
        original = {
            "check_id": "updater",
            "scope": "",
            "reason_code": "UPDATER_FAILED",
        }

        fresh = report(
            check(
                "CRITICAL",
                "APP_UNAVAILABLE",
                check_id="app",
            )
        )

        self.assertTrue(
            incident_validation(
                original,
                fresh,
            )
        )

    def test_validation_same_incident_present_fails(self):
        original = {
            "check_id": "updater",
            "scope": "",
            "reason_code": "UPDATER_FAILED",
        }

        fresh = report(
            check(
                "CRITICAL",
                "UPDATER_FAILED",
            )
        )

        self.assertFalse(
            incident_validation(
                original,
                fresh,
            )
        )


if __name__ == "__main__":
    unittest.main()
