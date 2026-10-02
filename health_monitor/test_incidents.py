"""Synthetic tests for Health Monitor V2 persistent incident state."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .incidents import (
    EVENT_CONTRACT,
    STATE_CONTRACT,
    empty_state,
    incident_key,
    load_state,
    process_report,
    reconcile,
)


NOW = datetime(2030, 10, 6, 16, 0, tzinfo=timezone.utc)


def check(status, reason, check_id="updater", scope=""):
    return {
        "check_id": check_id,
        "status": status,
        "reason_code": reason,
        "scope": scope,
        "summary": reason.lower(),
        "evidence": {},
    }


def report(*checks):
    counts = {
        status: sum(row["status"] == status for row in checks)
        for status in (
            "HEALTHY",
            "WARNING",
            "CRITICAL",
            "OWNER_ACTION_REQUIRED",
        )
    }

    severity = {
        "HEALTHY": 0,
        "WARNING": 1,
        "CRITICAL": 2,
        "OWNER_ACTION_REQUIRED": 3,
    }

    overall = max(
        (row["status"] for row in checks),
        key=lambda value: severity[value],
        default="HEALTHY",
    )

    return {
        "contract": "NFL_APP_HEALTH_MONITOR_V1",
        "generated_at_utc": NOW.isoformat(),
        "overall_status": overall,
        "summary_counts": counts,
        "checks": list(checks),
    }


class IncidentTests(unittest.TestCase):

    def test_healthy_creates_no_incident(self):
        result = reconcile(
            report(check("HEALTHY", "UPDATER_COMPLETED")),
            empty_state(),
            NOW,
        )

        self.assertEqual(result["contract"], EVENT_CONTRACT)
        self.assertEqual(result["events"], [])
        self.assertEqual(result["state"]["incidents"], {})

    def test_critical_opens_once_then_deduplicates(self):
        state = empty_state()
        failure = check("CRITICAL", "UPDATER_FAILED")

        first = reconcile(report(failure), state, NOW)

        self.assertEqual(
            [event["event_type"] for event in first["events"]],
            ["OPENED"],
        )

        second = reconcile(
            report(failure),
            first["state"],
            NOW + timedelta(minutes=5),
        )

        self.assertEqual(second["events"], [])

        key = incident_key(failure)
        incident = second["state"]["incidents"][key]

        self.assertTrue(incident["active"])
        self.assertEqual(incident["occurrences"], 2)
        self.assertEqual(incident["alerted_status"], "CRITICAL")

    def test_critical_to_owner_action_required_escalates(self):
        # Same stable incident identity, but status worsens.
        failure = check(
            "CRITICAL",
            "STARTER_AUTHORITY_CONFLICT",
            check_id="starter",
            scope="WAS",
        )

        state = reconcile(report(failure), empty_state(), NOW)["state"]

        worse = dict(failure)
        worse["status"] = "OWNER_ACTION_REQUIRED"

        result = reconcile(
            report(worse),
            state,
            NOW + timedelta(minutes=1),
        )

        self.assertEqual(
            [event["event_type"] for event in result["events"]],
            ["ESCALATED"],
        )

        key = incident_key(worse)

        self.assertEqual(
            result["state"]["incidents"][key]["alerted_status"],
            "OWNER_ACTION_REQUIRED",
        )

    def test_recovery_emitted_once(self):
        failure = check("CRITICAL", "UPDATER_FAILED")

        state = reconcile(report(failure), empty_state(), NOW)["state"]

        recovered = reconcile(
            report(check("HEALTHY", "UPDATER_COMPLETED")),
            state,
            NOW + timedelta(minutes=10),
        )

        self.assertEqual(
            [event["event_type"] for event in recovered["events"]],
            ["RECOVERED"],
        )

        again = reconcile(
            report(check("HEALTHY", "UPDATER_COMPLETED")),
            recovered["state"],
            NOW + timedelta(minutes=20),
        )

        self.assertEqual(again["events"], [])

    def test_warning_is_tracked_but_not_alerted(self):
        warning = check("WARNING", "UPDATER_RUNNING")

        result = reconcile(report(warning), empty_state(), NOW)

        self.assertEqual(result["events"], [])
        self.assertEqual(len(result["state"]["incidents"]), 1)

    def test_reason_change_closes_old_and_opens_new(self):
        old = check("CRITICAL", "UPDATER_FAILED")

        state = reconcile(report(old), empty_state(), NOW)["state"]

        new = check("CRITICAL", "UPDATER_MISSED_EXPECTED_RUN")

        result = reconcile(
            report(new),
            state,
            NOW + timedelta(minutes=5),
        )

        kinds = sorted(event["event_type"] for event in result["events"])

        self.assertEqual(kinds, ["OPENED", "RECOVERED"])

    def test_persistence_survives_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incident_state.json"
            failure = check("CRITICAL", "UPDATER_FAILED")

            first = process_report(report(failure), path, NOW)

            self.assertTrue(path.is_file())
            self.assertEqual(
                [event["event_type"] for event in first["events"]],
                ["OPENED"],
            )

            # Simulate a new process by loading from disk.
            loaded = load_state(path)

            self.assertEqual(loaded["contract"], STATE_CONTRACT)

            second = process_report(
                report(failure),
                path,
                NOW + timedelta(minutes=5),
            )

            self.assertEqual(second["events"], [])

            persisted = json.loads(path.read_text())

            key = incident_key(failure)

            self.assertEqual(
                persisted["incidents"][key]["occurrences"],
                2,
            )

    def test_unknown_report_contract_fails_closed(self):
        bad = report()
        bad["contract"] = "UNKNOWN"

        with self.assertRaises(ValueError):
            reconcile(bad, empty_state(), NOW)

    def test_unknown_state_contract_fails_closed(self):
        state = empty_state()
        state["contract"] = "UNKNOWN"

        with self.assertRaises(ValueError):
            reconcile(report(), state, NOW)


if __name__ == "__main__":
    unittest.main()


class StarterDependencyIncidentTests(unittest.TestCase):

    def test_qualified_starter_unavailable_suppressed_by_same_team_root(self):
        root = check(
            "OWNER_ACTION_REQUIRED",
            "ROTOWIRE_STARTER_BLOCKED",
            check_id="starter",
            scope="WAS",
        )
        dependent = check(
            "OWNER_ACTION_REQUIRED",
            "QUALIFIED_STARTER_UNAVAILABLE",
            check_id="qb_invariant",
            scope="2026_04_IND_WAS/WAS",
        )

        result = reconcile(
            report(root, dependent),
            empty_state(),
            NOW,
        )

        active = [
            x for x in result["state"]["incidents"].values()
            if x["active"]
        ]

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["check_id"], "starter")
        self.assertEqual(
            active[0]["reason_code"],
            "ROTOWIRE_STARTER_BLOCKED",
        )

    def test_starter_root_does_not_suppress_different_team(self):
        root = check(
            "OWNER_ACTION_REQUIRED",
            "ROTOWIRE_STARTER_BLOCKED",
            check_id="starter",
            scope="WAS",
        )
        dependent = check(
            "OWNER_ACTION_REQUIRED",
            "QUALIFIED_STARTER_UNAVAILABLE",
            check_id="qb_invariant",
            scope="2026_04_CHI_LV/CHI",
        )

        result = reconcile(
            report(root, dependent),
            empty_state(),
            NOW,
        )

        active = [
            x for x in result["state"]["incidents"].values()
            if x["active"]
        ]

        self.assertEqual(len(active), 2)

    def test_existing_dependent_recovers_when_root_appears(self):
        dependent = check(
            "OWNER_ACTION_REQUIRED",
            "QUALIFIED_STARTER_UNAVAILABLE",
            check_id="qb_invariant",
            scope="2026_04_IND_WAS/WAS",
        )

        state = reconcile(
            report(dependent),
            empty_state(),
            NOW,
        )["state"]

        root = check(
            "OWNER_ACTION_REQUIRED",
            "ROTOWIRE_STARTER_BLOCKED",
            check_id="starter",
            scope="WAS",
        )

        result = reconcile(
            report(root, dependent),
            state,
            NOW + timedelta(minutes=5),
        )

        dep = result["state"]["incidents"][incident_key(dependent)]
        self.assertFalse(dep["active"])

        recovered = [
            event for event in result["events"]
            if event["event_type"] == "RECOVERED"
            and event["incident_id"] == incident_key(dependent)
        ]

        self.assertEqual(len(recovered), 1)


