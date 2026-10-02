from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


# ======================================================================
# WFS NFL LIVE — STAGE LIVE-17
#
# CONCURRENCY-SAFE RECURRING POLLER
#
# EXECUTION CHAIN
#
#   live_safe_poll.py
#       ->
#   live_concurrency_gate.py
#       ->
#   live_orchestrator.py
#       ->
#   live_ingest.py
#       ->
#   data/wfs_live.db
#
# DESIGN
#
#   - This file contains NO ESPN parsing.
#   - This file contains NO player identity logic.
#   - This file contains NO SQLite writes.
#   - This file does NOT invoke live_ingest.py directly.
#   - This file does NOT invoke live_orchestrator.py directly.
#
#   Every poll must pass through the LIVE-16R concurrency gate.
#
#   nfl.db may legitimately change because the independent authorized
#   updater owns that database.
#
#   LIVE-16R is responsible for proving that the LIVE process tree made
#   zero writable opens to nfl.db.
#
#   forecast_ledger.db remains strictly protected.
#
#   The frozen LIVE execution files must not change during polling.
#
# ======================================================================


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(
    "/home/mwynn/nfl_data_engine"
).resolve()

PYTHON = (
    ROOT / "venv" / "bin" / "python"
)

CONCURRENCY_GATE = (
    ROOT / "live_concurrency_gate.py"
)

ORCHESTRATOR = (
    ROOT / "live_orchestrator.py"
)

INGEST = (
    ROOT / "live_ingest.py"
)

FORECAST_LEDGER = (
    ROOT / "data" / "forecast_ledger.db"
)

LOG_DIR = (
    ROOT / "logs"
)

LOG_FILE = (
    LOG_DIR / "wfs_live_safe_poll.log"
)

LOCK_FILE = (
    ROOT / ".wfs_live_safe_poll.lock"
)


# ======================================================================
# FROZEN KNOWN BASELINES
# ======================================================================

EXPECTED_ORCHESTRATOR_SHA256 = (
    "7767f05398677febfc303eb41626d23b8"
    "f7e11aee24b2144d5a0bb6294aab816"
)

EXPECTED_INGEST_SHA256 = (
    "472400cadde876fdb79f1b4567fa94f9"
    "d0bf00817b94e63251b6361a8a39d144"
)

EXPECTED_LEDGER_SHA256 = (
    "5cccee8d16cf39e1711bcac748e002f2"
    "f47d53b43776d9510c8659331ba10394"
)


# ======================================================================
# DEFAULTS
# ======================================================================

DEFAULT_SEASON = 2026
DEFAULT_INTERVAL_SECONDS = 30.0
MIN_INTERVAL_SECONDS = 10.0


# ======================================================================
# PROCESS STATE
# ======================================================================

STOP_REQUESTED = False


# ======================================================================
# HELPERS
# ======================================================================

def utc_now() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def emit(
    message: str,
) -> None:

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        f"{utc_now()} | {message}"
    )

    print(
        line,
        flush=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            line + "\n"
        )


def emit_multiline(
    prefix: str,
    text: str,
) -> None:

    if not text:

        emit(
            f"{prefix} | <EMPTY>"
        )

        return

    for line in text.splitlines():

        emit(
            f"{prefix} | {line}"
        )


def signal_handler(
    signum,
    frame,
) -> None:

    global STOP_REQUESTED

    STOP_REQUESTED = True

    emit(
        "STOP_REQUESTED"
        f" | signal={signum}"
    )


def sleep_interruptibly(
    seconds: float,
) -> None:

    deadline = (
        time.monotonic()
        + max(
            0.0,
            seconds,
        )
    )

    while (
        not STOP_REQUESTED
        and time.monotonic()
        < deadline
    ):

        remaining = (
            deadline
            - time.monotonic()
        )

        time.sleep(
            min(
                0.5,
                max(
                    0.0,
                    remaining,
                ),
            )
        )


# ======================================================================
# LOCK
# ======================================================================

def acquire_lock():

    lock_handle = LOCK_FILE.open(
        "a+"
    )

    try:

        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX
            | fcntl.LOCK_NB,
        )

    except BlockingIOError:

        lock_handle.close()

        raise RuntimeError(
            "another LIVE-17 polling process "
            "already holds the lock"
        )

    lock_handle.seek(
        0
    )

    lock_handle.truncate()

    lock_handle.write(
        f"pid={os.getpid()}\n"
    )

    lock_handle.write(
        f"started_utc={utc_now()}\n"
    )

    lock_handle.flush()

    return lock_handle


def release_lock(
    lock_handle,
) -> None:

    try:

        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_UN,
        )

    finally:

        lock_handle.close()


# ======================================================================
# CLI
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "WFS LIVE-17 concurrency-safe "
            "recurring polling wrapper"
        )
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            "Poll start-to-start interval in seconds. "
            f"Minimum {MIN_INTERVAL_SECONDS:.0f}."
        ),
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
    )

    parser.add_argument(
        "--max-polls",
        type=int,
        default=None,
        help=(
            "Stop after this many polls. "
            "Omit for continuous execution."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run exactly one polling cycle."
        ),
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help=(
            "Stop immediately after the first "
            "failed concurrency-gate cycle."
        ),
    )

    parser.add_argument(
        "--exercise-updater-poll",
        type=int,
        default=None,
        help=(
            "TEST ONLY: on this one poll number, pass "
            "--with-updater to LIVE-16R to deliberately "
            "exercise authorized nfl.db concurrency."
        ),
    )

    return parser.parse_args()


# ======================================================================
# PREFLIGHT
# ======================================================================

def verify_required_files():

    required = [
        PYTHON,
        CONCURRENCY_GATE,
        ORCHESTRATOR,
        INGEST,
        FORECAST_LEDGER,
    ]

    for path in required:

        if not path.is_file():

            raise RuntimeError(
                f"missing required file: {path}"
            )

        emit(
            f"PASS | exists | {path}"
        )


def verify_frozen_known_baselines():

    orchestrator_sha = (
        sha256_file(
            ORCHESTRATOR
        )
    )

    ingest_sha = (
        sha256_file(
            INGEST
        )
    )

    ledger_sha = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    emit(
        "LIVE_ORCHESTRATOR_SHA="
        f"{orchestrator_sha}"
    )

    emit(
        "LIVE_INGEST_SHA="
        f"{ingest_sha}"
    )

    emit(
        "FORECAST_LEDGER_SHA="
        f"{ledger_sha}"
    )

    if (
        orchestrator_sha
        != EXPECTED_ORCHESTRATOR_SHA256
    ):

        raise RuntimeError(
            "live_orchestrator.py frozen hash mismatch"
        )

    if (
        ingest_sha
        != EXPECTED_INGEST_SHA256
    ):

        raise RuntimeError(
            "live_ingest.py frozen hash mismatch"
        )

    if (
        ledger_sha
        != EXPECTED_LEDGER_SHA256
    ):

        raise RuntimeError(
            "forecast_ledger.db frozen hash mismatch"
        )

    emit(
        "PASS | frozen known baselines verified"
    )


# ======================================================================
# CHILD EXECUTION
# ======================================================================

def run_concurrency_gate(
    season: int,
    exercise_updater: bool,
):

    command = [
        str(
            PYTHON
        ),
        str(
            CONCURRENCY_GATE
        ),
        "--season",
        str(
            season
        ),
    ]

    if exercise_updater:

        command.append(
            "--with-updater"
        )

    emit(
        "GATE_COMMAND="
        + " ".join(
            command
        )
    )

    started = (
        time.monotonic()
    )

    completed = subprocess.run(
        command,
        cwd=str(
            ROOT
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    elapsed = (
        time.monotonic()
        - started
    )

    return (
        completed,
        elapsed,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    if args.once:

        args.max_polls = 1

    if (
        args.interval
        < MIN_INTERVAL_SECONDS
    ):

        raise SystemExit(
            "FAIL | interval must be at least "
            f"{MIN_INTERVAL_SECONDS:.0f} seconds"
        )

    if (
        args.max_polls is not None
        and args.max_polls < 1
    ):

        raise SystemExit(
            "FAIL | --max-polls must be >= 1"
        )

    if (
        args.exercise_updater_poll is not None
        and args.exercise_updater_poll < 1
    ):

        raise SystemExit(
            "FAIL | --exercise-updater-poll must be >= 1"
        )

    if (
        args.max_polls is not None
        and args.exercise_updater_poll is not None
        and args.exercise_updater_poll
        > args.max_polls
    ):

        raise SystemExit(
            "FAIL | --exercise-updater-poll exceeds "
            "--max-polls"
        )

    signal.signal(
        signal.SIGINT,
        signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler,
    )

    print(
        "=" * 72
    )

    print(
        "WFS NFL LIVE — STAGE LIVE-17"
    )

    print(
        "CONCURRENCY-SAFE RECURRING POLLER"
    )

    print(
        "LIVE-16R GATE ON EVERY CYCLE"
    )

    print(
        "=" * 72
    )

    # ==================================================================
    # PREFLIGHT
    # ==================================================================

    emit(
        "LIVE17_START"
        f" | interval={args.interval:.3f}"
        f" | season={args.season}"
        f" | max_polls={args.max_polls}"
        f" | exercise_updater_poll="
        f"{args.exercise_updater_poll}"
    )

    verify_required_files()

    verify_frozen_known_baselines()

    # The LIVE-16R gate itself becomes frozen for this poller process.
    concurrency_gate_sha_start = (
        sha256_file(
            CONCURRENCY_GATE
        )
    )

    emit(
        "CONCURRENCY_GATE_SHA_START="
        f"{concurrency_gate_sha_start}"
    )

    lock_handle = None

    polls_executed = 0
    successful_polls = 0
    failed_polls = 0

    final_status = (
        "PASS"
    )

    try:

        # ==============================================================
        # LOCK
        # ==============================================================

        lock_handle = (
            acquire_lock()
        )

        emit(
            "POLL_LOCK_ACQUIRED=1"
            f" | path={LOCK_FILE}"
        )

        # ==============================================================
        # POLLING LOOP
        # ==============================================================

        while not STOP_REQUESTED:

            if (
                args.max_polls is not None
                and polls_executed
                >= args.max_polls
            ):

                break

            poll_number = (
                polls_executed
                + 1
            )

            poll_started = (
                time.monotonic()
            )

            exercise_updater = (
                args.exercise_updater_poll
                == poll_number
            )

            emit(
                "=" * 68
            )

            emit(
                f"POLL_START"
                f" | poll={poll_number}"
                f" | exercise_updater="
                f"{int(exercise_updater)}"
            )

            # ----------------------------------------------------------
            # Verify the gate and frozen code have not changed between
            # cycles.
            # ----------------------------------------------------------

            gate_sha_pre = (
                sha256_file(
                    CONCURRENCY_GATE
                )
            )

            orchestrator_sha_pre = (
                sha256_file(
                    ORCHESTRATOR
                )
            )

            ingest_sha_pre = (
                sha256_file(
                    INGEST
                )
            )

            ledger_sha_pre = (
                sha256_file(
                    FORECAST_LEDGER
                )
            )

            if (
                gate_sha_pre
                != concurrency_gate_sha_start
            ):

                raise RuntimeError(
                    "live_concurrency_gate.py changed "
                    "during polling"
                )

            if (
                orchestrator_sha_pre
                != EXPECTED_ORCHESTRATOR_SHA256
            ):

                raise RuntimeError(
                    "live_orchestrator.py changed "
                    "during polling"
                )

            if (
                ingest_sha_pre
                != EXPECTED_INGEST_SHA256
            ):

                raise RuntimeError(
                    "live_ingest.py changed "
                    "during polling"
                )

            if (
                ledger_sha_pre
                != EXPECTED_LEDGER_SHA256
            ):

                raise RuntimeError(
                    "forecast_ledger.db changed "
                    "before polling cycle"
                )

            # ----------------------------------------------------------
            # Every cycle passes through LIVE-16R.
            # ----------------------------------------------------------

            completed, elapsed = (
                run_concurrency_gate(
                    season=args.season,
                    exercise_updater=exercise_updater,
                )
            )

            emit_multiline(
                "GATE_STDOUT",
                completed.stdout,
            )

            emit_multiline(
                "GATE_STDERR",
                completed.stderr,
            )

            # ----------------------------------------------------------
            # Verify frozen surfaces again after gate execution.
            # ----------------------------------------------------------

            gate_sha_post = (
                sha256_file(
                    CONCURRENCY_GATE
                )
            )

            orchestrator_sha_post = (
                sha256_file(
                    ORCHESTRATOR
                )
            )

            ingest_sha_post = (
                sha256_file(
                    INGEST
                )
            )

            ledger_sha_post = (
                sha256_file(
                    FORECAST_LEDGER
                )
            )

            gate_unchanged = (
                gate_sha_post
                == concurrency_gate_sha_start
            )

            orchestrator_unchanged = (
                orchestrator_sha_post
                == EXPECTED_ORCHESTRATOR_SHA256
            )

            ingest_unchanged = (
                ingest_sha_post
                == EXPECTED_INGEST_SHA256
            )

            ledger_unchanged = (
                ledger_sha_post
                == EXPECTED_LEDGER_SHA256
            )

            poll_success = (
                completed.returncode == 0
                and gate_unchanged
                and orchestrator_unchanged
                and ingest_unchanged
                and ledger_unchanged
            )

            polls_executed += 1

            if poll_success:

                successful_polls += 1

                poll_status = (
                    "PASS"
                )

            else:

                failed_polls += 1

                poll_status = (
                    "FAIL"
                )

                final_status = (
                    "FAIL"
                )

            emit(
                "POLL_RESULT"
                f" | poll={poll_number}"
                f" | status={poll_status}"
                f" | gate_rc={completed.returncode}"
                f" | seconds={elapsed:.3f}"
                f" | gate_unchanged="
                f"{int(gate_unchanged)}"
                f" | orchestrator_unchanged="
                f"{int(orchestrator_unchanged)}"
                f" | ingest_unchanged="
                f"{int(ingest_unchanged)}"
                f" | ledger_unchanged="
                f"{int(ledger_unchanged)}"
            )

            if (
                not poll_success
                and args.stop_on_failure
            ):

                emit(
                    "STOP_ON_FAILURE=1"
                    f" | poll={poll_number}"
                )

                break

            if STOP_REQUESTED:

                break

            if (
                args.max_polls is not None
                and polls_executed
                >= args.max_polls
            ):

                break

            # ----------------------------------------------------------
            # Start-to-start cadence.
            # ----------------------------------------------------------

            poll_cycle_elapsed = (
                time.monotonic()
                - poll_started
            )

            sleep_seconds = max(
                0.0,
                args.interval
                - poll_cycle_elapsed,
            )

            emit(
                "POLL_SLEEP"
                f" | poll={poll_number}"
                f" | cycle_seconds="
                f"{poll_cycle_elapsed:.3f}"
                f" | sleep_seconds="
                f"{sleep_seconds:.3f}"
            )

            sleep_interruptibly(
                sleep_seconds
            )

    except Exception as exc:

        final_status = (
            "FAIL"
        )

        emit(
            "LIVE17_EXCEPTION"
            f" | type={type(exc).__name__}"
            f" | message={exc}"
        )

    finally:

        if lock_handle is not None:

            release_lock(
                lock_handle
            )

            emit(
                "POLL_LOCK_RELEASED=1"
            )

    # ==================================================================
    # FINAL FROZEN-SURFACE VERIFY
    # ==================================================================

    concurrency_gate_sha_end = (
        sha256_file(
            CONCURRENCY_GATE
        )
    )

    orchestrator_sha_end = (
        sha256_file(
            ORCHESTRATOR
        )
    )

    ingest_sha_end = (
        sha256_file(
            INGEST
        )
    )

    ledger_sha_end = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    gate_unchanged_final = (
        concurrency_gate_sha_end
        == concurrency_gate_sha_start
    )

    orchestrator_unchanged_final = (
        orchestrator_sha_end
        == EXPECTED_ORCHESTRATOR_SHA256
    )

    ingest_unchanged_final = (
        ingest_sha_end
        == EXPECTED_INGEST_SHA256
    )

    ledger_unchanged_final = (
        ledger_sha_end
        == EXPECTED_LEDGER_SHA256
    )

    if not (
        gate_unchanged_final
        and orchestrator_unchanged_final
        and ingest_unchanged_final
        and ledger_unchanged_final
    ):

        final_status = (
            "FAIL"
        )

    if failed_polls > 0:

        final_status = (
            "FAIL"
        )

    # ==================================================================
    # FINAL OUTPUT
    # ==================================================================

    print()
    print(
        "=" * 72
    )

    print(
        "WFS NFL LIVE — LIVE-17 FINAL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"POLLS_EXECUTED={polls_executed}"
    )

    print(
        f"SUCCESSFUL_POLLS={successful_polls}"
    )

    print(
        f"FAILED_POLLS={failed_polls}"
    )

    print(
        "CONCURRENCY_GATE_UNCHANGED="
        f"{int(gate_unchanged_final)}"
    )

    print(
        "LIVE_ORCHESTRATOR_UNCHANGED="
        f"{int(orchestrator_unchanged_final)}"
    )

    print(
        "LIVE_INGEST_UNCHANGED="
        f"{int(ingest_unchanged_final)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged_final)}"
    )

    print(
        f"CONCURRENCY_GATE_SHA="
        f"{concurrency_gate_sha_end}"
    )

    print(
        f"LIVE17_STATUS={final_status}"
    )

    if final_status == "PASS":

        print(
            "=" * 72
        )

        print(
            "PASS | LIVE-17 CONCURRENCY-SAFE "
            "RECURRING POLLING COMPLETE"
        )

        print(
            "=" * 72
        )

    else:

        print(
            "=" * 72
        )

        print(
            "FAIL | LIVE-17 CONCURRENCY-SAFE "
            "RECURRING POLLING FAILED"
        )

        print(
            "=" * 72
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
