"""Synthetic tests for Health Monitor V2 repair authorization."""

import unittest

from .repair_policy import (
    GUARDED_FIX,
    OWNER_ACTION_REQUIRED,
    authorize_incident,
    circuit_state,
    policy_for,
    record_repair_attempt,
    repair_allowed,
)


def incident(reason, status="CRITICAL"):
    return {
        "incident_id": "fixture",
        "check_id": "fixture",
        "scope": "",
        "reason_code": reason,
        "status": status,
        "repair_attempts": 0,
    }


class RepairPolicyTests(unittest.TestCase):

    def test_updater_failure_is_guarded(self):
        row = incident("UPDATER_FAILED")
        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            GUARDED_FIX,
        )
        self.assertEqual(
            decision["repair_id"],
            "RUN_UPDATER_ONCE",
        )
        self.assertTrue(repair_allowed(row))

    def test_missed_updater_is_guarded(self):
        row = incident("UPDATER_MISSED_EXPECTED_RUN")
        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            GUARDED_FIX,
        )
        self.assertEqual(
            decision["repair_id"],
            "RUN_UPDATER_ONCE",
        )

    def test_app_unavailable_is_guarded(self):
        row = incident("APP_UNAVAILABLE")
        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            GUARDED_FIX,
        )
        self.assertEqual(
            decision["repair_id"],
            "RESTART_WFS_SERVICE_ONCE",
        )

    def test_unknown_reason_fails_closed(self):
        row = incident("SOMETHING_NEW")

        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            OWNER_ACTION_REQUIRED,
        )
        self.assertIsNone(decision["repair_id"])
        self.assertFalse(repair_allowed(row))

    def test_identity_problem_requires_owner(self):
        for reason in (
            "LIVE_IDENTITY_UNRESOLVED",
            "STARTER_AUTHORITY_CONFLICT",
            "UNRESOLVED_STARTER_IDENTITY",
            "STARTER_PRIMARY_QB_MISMATCH",
            "PRIMARY_PUBLIC_QB_MISMATCH",
            "QB_NUMERICAL_FORECAST_MISMATCH",
        ):
            with self.subTest(reason=reason):
                decision = authorize_incident(
                    incident(reason)
                )

                self.assertEqual(
                    decision["authorization"],
                    OWNER_ACTION_REQUIRED,
                )

    def test_warning_never_repairs(self):
        row = incident(
            "UPDATER_FAILED",
            status="WARNING",
        )

        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            "NO_ACTION",
        )
        self.assertFalse(repair_allowed(row))

    def test_owner_status_overrides_allowlist(self):
        row = incident(
            "UPDATER_FAILED",
            status="OWNER_ACTION_REQUIRED",
        )

        decision = authorize_incident(row)

        self.assertEqual(
            decision["authorization"],
            OWNER_ACTION_REQUIRED,
        )
        self.assertFalse(repair_allowed(row))

    def test_first_attempt_permitted(self):
        row = incident("UPDATER_FAILED")

        state = circuit_state(row)

        self.assertFalse(state["open"])
        self.assertEqual(
            state["reason"],
            "REPAIR_PERMITTED",
        )

    def test_success_exhausts_budget(self):
        row = incident("UPDATER_FAILED")

        record_repair_attempt(
            row,
            result="SUCCEEDED",
            attempted_at_utc="2030-10-06T16:00:00+00:00",
        )

        state = circuit_state(row)

        self.assertTrue(state["open"])
        self.assertEqual(
            state["reason"],
            "REPAIR_BUDGET_EXHAUSTED",
        )

    def test_failure_opens_circuit(self):
        row = incident("UPDATER_FAILED")

        record_repair_attempt(
            row,
            result="FAILED",
            attempted_at_utc="2030-10-06T16:00:00+00:00",
        )

        state = circuit_state(row)

        self.assertTrue(state["open"])
        self.assertEqual(
            state["reason"],
            "PREVIOUS_REPAIR_FAILED",
        )

    def test_policy_lookup_is_deterministic(self):
        first = policy_for("UPDATER_FAILED")
        second = policy_for("UPDATER_FAILED")

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
