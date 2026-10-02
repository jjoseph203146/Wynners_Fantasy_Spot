"""
NFL APP Health Monitor V2 guarded repair runner.

This is NOT a generic command runner.

Only explicitly implemented repair IDs may execute:
    RUN_UPDATER_ONCE
    RESTART_WFS_SERVICE_ONCE

Hard safety boundaries:
- never stop/start/restart wfs-nfl-live.service
- never delete or override nfl_updater.lock
- never bypass an active live_concurrency_gate.py
- never modify cron
- never use shell=True
- never accept an arbitrary command from monitor evidence
- one attempt budget is enforced by repair_policy
- validation is mandatory after execution
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
from typing import Callable, Dict, Optional

from .repair_policy import (
    AUTO_FIX,
    GUARDED_FIX,
    authorize_incident,
    circuit_state,
)


ROOT = Path("/home/mwynn/nfl_data_engine").resolve()
UPDATER = ROOT / "run_updater.sh"
UPDATER_LOCK = ROOT / "nfl_updater.lock"

WFS_SERVICE = "wfs.service"
NFL_LIVE_SERVICE = "wfs-nfl-live.service"

IMPLEMENTED_REPAIRS = {
    "RUN_UPDATER_ONCE",
    "RESTART_WFS_SERVICE_ONCE",
}


class RepairDeferred(Exception):
    """Safety condition requires retrying on a later monitor cycle."""


class RepairRejected(Exception):
    """Repair is not authorized or violates the repair contract."""


def _run(
    argv,
    *,
    timeout: int,
    runner: Callable = subprocess.run,
):
    """
    Execute a fixed argv vector.

    shell=True is deliberately impossible through this interface.
    """
    result = runner(
        list(argv),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    return result


def _process_exists(
    pattern: str,
    *,
    runner: Callable = subprocess.run,
) -> bool:
    result = _run(
        ["pgrep", "-f", pattern],
        timeout=5,
        runner=runner,
    )

    return result.returncode == 0


def live_concurrency_active(
    *,
    runner: Callable = subprocess.run,
) -> bool:
    # Same process family already respected by run_updater.sh.
    return _process_exists(
        "[l]ive_concurrency_gate.py",
        runner=runner,
    )


def updater_lock_available(
    lock_path: Path = UPDATER_LOCK,
) -> bool:
    """
    Probe the updater lock without deleting, replacing, truncating or
    overriding it.

    Missing lock file means no established lock inode currently exists.
    Existing lock file is opened read-only and tested non-blocking.
    """
    lock_path = Path(lock_path)

    if not lock_path.exists():
        return True

    fd = os.open(lock_path, os.O_RDONLY)

    try:
        try:
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            return False

        fcntl.flock(fd, fcntl.LOCK_UN)
        return True

    finally:
        os.close(fd)


def preflight(
    repair_id: str,
    *,
    runner: Callable = subprocess.run,
    lock_path: Path = UPDATER_LOCK,
) -> Dict[str, object]:
    if repair_id not in IMPLEMENTED_REPAIRS:
        raise RepairRejected("REPAIR_ID_NOT_IMPLEMENTED")

    if repair_id == "RUN_UPDATER_ONCE":
        if live_concurrency_active(runner=runner):
            raise RepairDeferred("LIVE_CONCURRENCY_ACTIVE")

        if not updater_lock_available(lock_path):
            raise RepairDeferred("UPDATER_LOCK_HELD")

        if not UPDATER.is_file():
            raise RepairRejected("UPDATER_ENTRYPOINT_MISSING")

        if not os.access(UPDATER, os.X_OK):
            raise RepairRejected("UPDATER_ENTRYPOINT_NOT_EXECUTABLE")

        return {
            "status": "PASS",
            "repair_id": repair_id,
            "safety_gate": "UPDATER_SAFE_TO_ATTEMPT",
        }

    if repair_id == "RESTART_WFS_SERVICE_ONCE":
        return {
            "status": "PASS",
            "repair_id": repair_id,
            "safety_gate": "WFS_SERVICE_RESTART_ALLOWLISTED",
        }

    raise RepairRejected("UNREACHABLE_REPAIR_ID")


def command_for(repair_id: str):
    """
    Commands are constants selected solely from an explicit repair ID.
    No command comes from incident text/evidence.
    """
    if repair_id == "RUN_UPDATER_ONCE":
        return [str(UPDATER)]

    if repair_id == "RESTART_WFS_SERVICE_ONCE":
        return [
            "systemctl",
            "restart",
            WFS_SERVICE,
        ]

    raise RepairRejected("REPAIR_ID_NOT_IMPLEMENTED")


def execute(
    incident: dict,
    *,
    validate: Callable[[str], bool],
    runner: Callable = subprocess.run,
    lock_path: Path = UPDATER_LOCK,
) -> Dict[str, object]:
    """
    Execute one policy-authorized repair.

    The caller owns persistent incident-state recording.

    DEFERRED does not consume the repair budget because no mutation was
    attempted.

    Once the command starts, the caller must count the attempt regardless
    of command/validation success.
    """
    decision = authorize_incident(incident)

    if decision["authorization"] not in {
        AUTO_FIX,
        GUARDED_FIX,
    }:
        raise RepairRejected("INCIDENT_NOT_REPAIR_AUTHORIZED")

    repair_id = decision.get("repair_id")

    if repair_id not in IMPLEMENTED_REPAIRS:
        raise RepairRejected("REPAIR_ID_NOT_IMPLEMENTED")

    circuit = circuit_state(incident)

    if circuit["open"]:
        raise RepairRejected(
            "REPAIR_CIRCUIT_OPEN:" + str(circuit["reason"])
        )

    try:
        gate = preflight(
            repair_id,
            runner=runner,
            lock_path=lock_path,
        )
    except RepairDeferred as exc:
        return {
            "status": "DEFERRED",
            "repair_id": repair_id,
            "reason": str(exc),
            "attempted": False,
        }

    argv = command_for(repair_id)

    timeout = (
        3600
        if repair_id == "RUN_UPDATER_ONCE"
        else 30
    )

    try:
        result = _run(
            argv,
            timeout=timeout,
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "FAILED",
            "repair_id": repair_id,
            "reason": "REPAIR_TIMEOUT",
            "attempted": True,
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "repair_id": repair_id,
            "reason": "REPAIR_EXECUTION_EXCEPTION",
            "exception_type": type(exc).__name__,
            "attempted": True,
        }

    if result.returncode != 0:
        return {
            "status": "FAILED",
            "repair_id": repair_id,
            "reason": "REPAIR_COMMAND_FAILED",
            "returncode": int(result.returncode),
            "attempted": True,
        }

    try:
        valid = bool(validate(repair_id))
    except Exception as exc:
        return {
            "status": "FAILED",
            "repair_id": repair_id,
            "reason": "REPAIR_VALIDATION_EXCEPTION",
            "exception_type": type(exc).__name__,
            "attempted": True,
        }

    if not valid:
        return {
            "status": "FAILED",
            "repair_id": repair_id,
            "reason": "REPAIR_VALIDATION_FAILED",
            "attempted": True,
        }

    return {
        "status": "SUCCEEDED",
        "repair_id": repair_id,
        "reason": "REPAIR_VALIDATED",
        "attempted": True,
        "preflight": gate["safety_gate"],
    }
