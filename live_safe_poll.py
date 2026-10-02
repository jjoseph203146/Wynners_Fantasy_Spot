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
# WFS NFL LIVE — STAGE LIVE-18R
#
# CONCURRENCY-SAFE RECURRING POLLER
# GRACEFUL SYSTEMD SHUTDOWN CONTRACT
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
# LIVE-18R CHANGE
#
#   An intentional SIGINT/SIGTERM received while the concurrency gate
#   is active is an operational shutdown event, NOT a failed poll.
#
#   Specifically:
#
#       systemctl stop
#           ->
#       SIGTERM received
#           ->
#       STOP_REQUESTED=True
#           ->
#       active child may terminate via SIGTERM
#           ->
#       interrupted poll is NOT counted as failed
#           ->
#       lock released
#           ->
#       process exits 0
#
#   Real child failures when STOP_REQUESTED is false continue to fail
#   closed exactly as before.
#
# IMPORTANT
#
#   - No ESPN parsing exists here.
#   - No identity logic exists here.
#   - No SQLite writes exist here.
#   - nfl.db is not treated as globally immutable here.
#   - LIVE-16R owns nfl.db process-level write provenance.
#   - forecast_ledger.db remains strictly protected.
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

UPDATER_LOCK_FILE = (
    ROOT / "nfl_updater.lock"
)


# ======================================================================
# FROZEN KNOWN BASELINES
# ======================================================================

EXPECTED_ORCHESTRATOR_SHA256 = (
    "af9b6509eb17da199f944651cc27ed5c"
    "dc832fe2592b7db4552296ef89892cd9"
)

EXPECTED_INGEST_SHA256 = (
    '07effd66ace5500984ac02f2f58b41ab3f0ff2fbe3ec539fc823a24eead76fce'
)

# forecast_ledger.db is an authorized append-only forecast surface.
#
# LIVE must never modify it during a polling cycle, but the forecast
# pipeline may legitimately append validated snapshots between LIVE
# cycles. Therefore the ledger is protected by a per-cycle SHA
# baseline rather than one permanent whole-file SHA.


# ======================================================================
# DEFAULTS
# ======================================================================

DEFAULT_SEASON = 2026
DEFAULT_INTERVAL_SECONDS = 45.0
MIN_INTERVAL_SECONDS = 10.0


# ======================================================================
# PROCESS STATE
# ======================================================================

STOP_REQUESTED = False
STOP_SIGNAL = None


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
    global STOP_SIGNAL

    STOP_REQUESTED = True
    STOP_SIGNAL = signum

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
            "another LIVE polling process "
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
            "WFS LIVE-18R concurrency-safe "
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
            "Stop after this many completed polls. "
            "Omit for continuous execution."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run exactly one completed polling cycle."
        ),
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help=(
            "Stop immediately after the first "
            "actual failed concurrency-gate cycle."
        ),
    )

    parser.add_argument(
        "--exercise-updater-poll",
        type=int,
        default=None,
        help=(
            "TEST ONLY: on this poll number, pass "
            "--with-updater to LIVE-16R."
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

    emit(
        "PASS | forecast_ledger.db present"
        " | startup_sha="
        f"{ledger_sha}"
        " | protection=per_cycle_read_only"
    )

    emit(
        "PASS | frozen executable baselines verified"
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

    updater_lock_handle = (
        UPDATER_LOCK_FILE.open(
            "a+"
        )
    )

    emit(
        "WAITING_FOR_UPDATER_SHARED_LOCK=1"
        f" | path={UPDATER_LOCK_FILE}"
    )

    fcntl.flock(
        updater_lock_handle.fileno(),
        fcntl.LOCK_SH,
    )

    emit(
        "UPDATER_SHARED_LOCK_ACQUIRED=1"
        f" | path={UPDATER_LOCK_FILE}"
    )

    try:
        completed = subprocess.run(
            command,
            cwd=str(
                ROOT
            ),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        fcntl.flock(
            updater_lock_handle.fileno(),
            fcntl.LOCK_UN,
        )
        updater_lock_handle.close()

        emit(
            "UPDATER_SHARED_LOCK_RELEASED=1"
            f" | path={UPDATER_LOCK_FILE}"
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
        "WFS NFL LIVE — STAGE LIVE-18R"
    )

    print(
        "CONCURRENCY-SAFE RECURRING POLLER"
    )

    print(
        "GRACEFUL SYSTEMD SHUTDOWN CONTRACT"
    )

    print(
        "=" * 72
    )

    emit(
        "LIVE18R_START"
        f" | interval={args.interval:.3f}"
        f" | season={args.season}"
        f" | max_polls={args.max_polls}"
        f" | exercise_updater_poll="
        f"{args.exercise_updater_poll}"
    )

    verify_required_files()

    verify_frozen_known_baselines()

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

    polls_started = 0
    polls_executed = 0
    successful_polls = 0
    failed_polls = 0
    interrupted_polls = 0

    controlled_stop = False

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

            polls_started += 1

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
                "POLL_START"
                f" | poll={poll_number}"
                f" | exercise_updater="
                f"{int(exercise_updater)}"
            )

            # ----------------------------------------------------------
            # PRE-CYCLE FROZEN SURFACE VERIFY
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

            emit(
                "FORECAST_LEDGER_CYCLE_SHA_START="
                f"{ledger_sha_pre}"
                f" | poll={poll_number}"
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

            # forecast_ledger.db may legitimately change between
            # polling cycles because the forecast pipeline is append-only.
            # ledger_sha_pre is the authoritative baseline for this cycle.

            # ----------------------------------------------------------
            # RUN LIVE-16R GATE
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
            # POST-CYCLE FROZEN SURFACE VERIFY
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
                == ledger_sha_pre
            )

            frozen_cycle_ok = (
                gate_unchanged
                and orchestrator_unchanged
                and ingest_unchanged
                and ledger_unchanged
            )

            # ----------------------------------------------------------
            # LIVE-18R CONTROLLED STOP CLASSIFICATION
            #
            # If SIGINT/SIGTERM arrived while the gate was running,
            # systemd may terminate the whole child process tree.
            #
            # A nonzero child return in this specific state is not an
            # application failure.
            #
            # We still require every frozen surface to remain intact.
            # ----------------------------------------------------------

            if (
                STOP_REQUESTED
                and completed.returncode != 0
            ):

                if not frozen_cycle_ok:

                    failed_polls += 1

                    final_status = (
                        "FAIL"
                    )

                    emit(
                        "POLL_INTERRUPTED_BUT_FROZEN_SURFACE_FAILED"
                        f" | poll={poll_number}"
                        f" | gate_rc={completed.returncode}"
                        f" | gate_unchanged="
                        f"{int(gate_unchanged)}"
                        f" | orchestrator_unchanged="
                        f"{int(orchestrator_unchanged)}"
                        f" | ingest_unchanged="
                        f"{int(ingest_unchanged)}"
                        f" | ledger_unchanged="
                        f"{int(ledger_unchanged)}"
                    )

                else:

                    interrupted_polls += 1

                    controlled_stop = True

                    emit(
                        "POLL_INTERRUPTED_BY_CONTROLLED_STOP"
                        f" | poll={poll_number}"
                        f" | gate_rc={completed.returncode}"
                        f" | signal={STOP_SIGNAL}"
                        f" | seconds={elapsed:.3f}"
                    )

                    emit(
                        "PASS | intentional shutdown "
                        "interruption not counted as poll failure"
                    )

                break

            # ----------------------------------------------------------
            # NORMAL COMPLETED POLL
            # ----------------------------------------------------------

            poll_success = (
                completed.returncode == 0
                and frozen_cycle_ok
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

            # ----------------------------------------------------------
            # Child may have completed successfully at almost the same
            # instant a stop signal arrived.
            #
            # Count the completed PASS normally, then stop cleanly.
            # ----------------------------------------------------------

            if STOP_REQUESTED:

                controlled_stop = True

                emit(
                    "CONTROLLED_STOP_AFTER_COMPLETED_POLL=1"
                    f" | poll={poll_number}"
                    f" | signal={STOP_SIGNAL}"
                )

                break

            if (
                not poll_success
                and args.stop_on_failure
            ):

                emit(
                    "STOP_ON_FAILURE=1"
                    f" | poll={poll_number}"
                )

                break

            if (
                args.max_polls is not None
                and polls_executed
                >= args.max_polls
            ):

                break

            # ----------------------------------------------------------
            # POST-CYCLE QUIET PERIOD
            #
            # The LIVE gate may run longer than the configured interval.
            # Always preserve an unlocked interval after a completed cycle
            # so the exclusive production updater can acquire its lock.
            # ----------------------------------------------------------

            poll_cycle_elapsed = (
                time.monotonic()
                - poll_started
            )

            sleep_seconds = max(
                0.0,
                args.interval,
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

            if STOP_REQUESTED:

                controlled_stop = True

                emit(
                    "CONTROLLED_STOP_DURING_SLEEP=1"
                    f" | signal={STOP_SIGNAL}"
                )

                break

    except Exception as exc:

        final_status = (
            "FAIL"
        )

        emit(
            "LIVE18R_EXCEPTION"
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
    # FINAL FROZEN SURFACE VERIFY
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

    # The forecast ledger is protected inside every LIVE cycle.
    # It may legitimately evolve between cycles through the separate
    # append-only forecast workflow.
    ledger_unchanged_final = True

    final_frozen_surfaces_ok = (
        gate_unchanged_final
        and orchestrator_unchanged_final
        and ingest_unchanged_final
        and ledger_unchanged_final
    )

    if not final_frozen_surfaces_ok:

        final_status = (
            "FAIL"
        )

    if failed_polls > 0:

        final_status = (
            "FAIL"
        )

    # An intentional signal with no actual failure is a successful
    # controlled shutdown.
    if (
        STOP_REQUESTED
        and failed_polls == 0
        and final_frozen_surfaces_ok
    ):

        controlled_stop = True

    # ==================================================================
    # FINAL OUTPUT
    # ==================================================================

    print()
    print(
        "=" * 72
    )

    print(
        "WFS NFL LIVE — LIVE-18R FINAL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"POLLS_STARTED={polls_started}"
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
        f"INTERRUPTED_POLLS={interrupted_polls}"
    )

    print(
        f"STOP_REQUESTED={int(STOP_REQUESTED)}"
    )

    print(
        f"STOP_SIGNAL={STOP_SIGNAL}"
    )

    print(
        f"CONTROLLED_STOP={int(controlled_stop)}"
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
        "FORECAST_LEDGER_CYCLE_PROTECTION="
        f"{int(ledger_unchanged_final)}"
    )

    print(
        "FORECAST_LEDGER_SHA_END="
        f"{ledger_sha_end}"
    )

    print(
        "CONCURRENCY_GATE_SHA="
        f"{concurrency_gate_sha_end}"
    )

    print(
        f"LIVE18R_STATUS={final_status}"
    )

    if (
        final_status == "PASS"
        and controlled_stop
    ):

        print(
            "=" * 72
        )

        print(
            "PASS | LIVE-18R CONTROLLED "
            "SHUTDOWN COMPLETE"
        )

        print(
            "=" * 72
        )

    elif final_status == "PASS":

        print(
            "=" * 72
        )

        print(
            "PASS | LIVE-18R CONCURRENCY-SAFE "
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
            "FAIL | LIVE-18R CONCURRENCY-SAFE "
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
