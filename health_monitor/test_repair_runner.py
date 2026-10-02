"""Synthetic tests for Health Monitor V2 guarded repair runner."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .repair_runner import (
    RepairRejected,
    command_for,
    execute,
    preflight,
)


class Result:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def critical(reason):
    return {
        "incident_id": "fixture",
        "check_id": "fixture",
        "scope": "",
        "reason_code": reason,
        "status": "CRITICAL",
        "repair_attempts": 0,
    }


class FakeRunner:
    def __init__(
        self,
        *,
        pgrep_rc=1,
        command_rc=0,
        timeout_command=False,
    ):
        self.pgrep_rc = pgrep_rc
        self.command_rc = command_rc
        self.timeout_command = timeout_command
        self.calls = []

    def __call__(
        self,
        argv,
        cwd=None,
        capture_output=None,
        text=None,
        timeout=None,
        check=None,
    ):
        self.calls.append(list(argv))

        if argv[:2] == ["pgrep", "-f"]:
            return Result(self.pgrep_rc)

        if self.timeout_command:
            raise subprocess.TimeoutExpired(
                argv,
                timeout,
            )

        return Result(self.command_rc)


class RepairRunnerTests(unittest.TestCase):

    def test_unknown_repair_id_rejected(self):
        with self.assertRaises(RepairRejected):
            command_for("ARBITRARY_COMMAND")

    def test_updater_command_is_fixed(self):
        command = command_for("RUN_UPDATER_ONCE")

        self.assertEqual(len(command), 1)
        self.assertTrue(
            command[0].endswith("/run_updater.sh")
        )

    def test_app_restart_command_is_exact(self):
        self.assertEqual(
            command_for("RESTART_WFS_SERVICE_ONCE"),
            [
                "systemctl",
                "restart",
                "wfs.service",
            ],
        )

    def test_live_service_can_never_be_selected(self):
        for repair_id in (
            "RESTART_WFS_NFL_LIVE",
            "STOP_WFS_NFL_LIVE",
            "START_WFS_NFL_LIVE",
        ):
            with self.assertRaises(RepairRejected):
                command_for(repair_id)

    def test_active_live_gate_defers_updater(self):
        runner = FakeRunner(pgrep_rc=0)

        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "lock"

            result = execute(
                critical("UPDATER_FAILED"),
                validate=lambda _: True,
                runner=runner,
                lock_path=lock,
            )

        self.assertEqual(
            result["status"],
            "DEFERRED",
        )
        self.assertFalse(result["attempted"])
        self.assertEqual(
            result["reason"],
            "LIVE_CONCURRENCY_ACTIVE",
        )

        self.assertEqual(
            len(runner.calls),
            1,
        )

    def test_held_updater_lock_defers(self):
        runner = FakeRunner(pgrep_rc=1)

        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "lock"
            lock.touch()

            with patch(
                "health_monitor.repair_runner.updater_lock_available",
                return_value=False,
            ):
                result = execute(
                    critical("UPDATER_FAILED"),
                    validate=lambda _: True,
                    runner=runner,
                    lock_path=lock,
                )

        self.assertEqual(
            result["status"],
            "DEFERRED",
        )
        self.assertFalse(result["attempted"])
        self.assertEqual(
            result["reason"],
            "UPDATER_LOCK_HELD",
        )

    def test_success_requires_validation(self):
        runner = FakeRunner(
            pgrep_rc=1,
            command_rc=0,
        )

        with tempfile.TemporaryDirectory() as temp:
            result = execute(
                critical("UPDATER_FAILED"),
                validate=lambda _: True,
                runner=runner,
                lock_path=Path(temp) / "lock",
            )

        self.assertEqual(
            result["status"],
            "SUCCEEDED",
        )
        self.assertTrue(result["attempted"])

    def test_validation_failure_is_failed_attempt(self):
        runner = FakeRunner(
            pgrep_rc=1,
            command_rc=0,
        )

        with tempfile.TemporaryDirectory() as temp:
            result = execute(
                critical("UPDATER_FAILED"),
                validate=lambda _: False,
                runner=runner,
                lock_path=Path(temp) / "lock",
            )

        self.assertEqual(
            result["status"],
            "FAILED",
        )
        self.assertTrue(result["attempted"])
        self.assertEqual(
            result["reason"],
            "REPAIR_VALIDATION_FAILED",
        )

    def test_command_failure_is_failed_attempt(self):
        runner = FakeRunner(
            pgrep_rc=1,
            command_rc=47,
        )

        with tempfile.TemporaryDirectory() as temp:
            result = execute(
                critical("UPDATER_FAILED"),
                validate=lambda _: True,
                runner=runner,
                lock_path=Path(temp) / "lock",
            )

        self.assertEqual(
            result["status"],
            "FAILED",
        )
        self.assertEqual(
            result["returncode"],
            47,
        )
        self.assertTrue(result["attempted"])

    def test_timeout_is_failed_attempt(self):
        runner = FakeRunner(
            pgrep_rc=1,
            timeout_command=True,
        )

        with tempfile.TemporaryDirectory() as temp:
            result = execute(
                critical("UPDATER_FAILED"),
                validate=lambda _: True,
                runner=runner,
                lock_path=Path(temp) / "lock",
            )

        self.assertEqual(
            result["status"],
            "FAILED",
        )
        self.assertEqual(
            result["reason"],
            "REPAIR_TIMEOUT",
        )
        self.assertTrue(result["attempted"])

    def test_owner_action_never_executes(self):
        runner = FakeRunner()

        row = critical(
            "STARTER_AUTHORITY_CONFLICT"
        )
        row["status"] = "OWNER_ACTION_REQUIRED"

        with self.assertRaises(RepairRejected):
            execute(
                row,
                validate=lambda _: True,
                runner=runner,
            )

        self.assertEqual(runner.calls, [])

    def test_budget_exhaustion_blocks_execution(self):
        runner = FakeRunner()

        row = critical("UPDATER_FAILED")
        row["repair_attempts"] = 1
        row["last_repair_result"] = "FAILED"

        with self.assertRaises(RepairRejected):
            execute(
                row,
                validate=lambda _: True,
                runner=runner,
            )

        self.assertEqual(runner.calls, [])

    def test_preflight_does_not_support_live_service(self):
        with self.assertRaises(RepairRejected):
            preflight(
                "RESTART_WFS_NFL_LIVE"
            )


if __name__ == "__main__":
    unittest.main()
