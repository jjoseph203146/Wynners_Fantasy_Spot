#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PYTHON = ROOT / "venv/bin/python"

DETECTOR = (
    ROOT /
    "scripts/fanduel_classic_stale_detector_v1.py"
)

RECOVERY = (
    ROOT /
    "scripts/wfs_emergency_fanduel_refresh_v1.py"
)

UPDATER_LOCK = ROOT / "nfl_updater.lock"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
    )


def lock_is_free() -> bool:
    with UPDATER_LOCK.open("a+") as fh:
        try:
            fcntl.flock(
                fh.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            return False

        fcntl.flock(
            fh.fileno(),
            fcntl.LOCK_UN,
        )
        return True


def main() -> int:
    print("=" * 72)
    print("FANDUEL CLASSIC AUTO-RECOVERY V1")
    print("=" * 72)

    if not DETECTOR.is_file():
        print("AUTORECOVERY=FAIL_CLOSED")
        print("REASON=DETECTOR_MISSING")
        return 1

    if not RECOVERY.is_file():
        print("AUTORECOVERY=FAIL_CLOSED")
        print("REASON=RECOVERY_CONTROLLER_MISSING")
        return 1

    print("PHASE=PRECHECK_DETECTOR")

    before = run([
        str(PYTHON),
        str(DETECTOR),
    ])

    detector_rc = before.returncode

    print(f"PRECHECK_DETECTOR_RC={detector_rc}")

    if detector_rc == 0:
        print("AUTORECOVERY=NO_ACTION")
        print("RECOVERY_EXECUTED=FALSE")
        return 0

    if detector_rc != 10:
        print("AUTORECOVERY=FAIL_CLOSED")
        print("REASON=DETECTOR_AMBIGUOUS")
        print("RECOVERY_EXECUTED=FALSE")
        return 1

    print("STALE_SOURCE_PROVEN=TRUE")

    # Critical safety contract:
    # V4.3 independently acquires this exact lock.
    # Recovery must never be launched while the caller still owns it.
    if not lock_is_free():
        print("AUTORECOVERY=FAIL_CLOSED")
        print("REASON=UPDATER_LOCK_BUSY")
        print("RECOVERY_EXECUTED=FALSE")
        return 1

    print("UPDATER_LOCK_FREE=TRUE")
    print("RECOVERY_ATTEMPT=1")
    print("PHASE=GUARDED_V43_RECOVERY")

    recovery = run([
        str(PYTHON),
        str(RECOVERY),
        "--execute",
    ])

    print(
        f"RECOVERY_CONTROLLER_RC="
        f"{recovery.returncode}"
    )

    if recovery.returncode != 0:
        print("AUTORECOVERY=RECOVERY_FAILED")
        print("RECOVERY_EXECUTED=TRUE")
        print("RECOVERY_RETRY=FALSE")
        return 1

    print("PHASE=POST_RECOVERY_DETECTOR")

    after = run([
        str(PYTHON),
        str(DETECTOR),
    ])

    print(
        f"POST_RECOVERY_DETECTOR_RC="
        f"{after.returncode}"
    )

    if after.returncode != 0:
        print(
            "AUTORECOVERY="
            "POST_VALIDATION_FAILED"
        )
        print("RECOVERY_EXECUTED=TRUE")
        print("RECOVERY_RETRY=FALSE")
        return 1

    print("AUTORECOVERY=RECOVERED")
    print("RECOVERY_EXECUTED=TRUE")
    print("RECOVERY_ATTEMPTS=1")
    print("POST_VALIDATION=PASS")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("AUTORECOVERY=FAIL_CLOSED")
        print(
            "UNHANDLED_ERROR="
            f"{type(exc).__name__}:{exc}"
        )
        print("RECOVERY_RETRY=FALSE")
        raise SystemExit(1)
