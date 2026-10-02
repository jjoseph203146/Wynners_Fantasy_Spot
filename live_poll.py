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
# WFS NFL LIVE — STAGE LIVE-14
#
# CONTROLLED RECURRING POLLING WRAPPER
#
# PURPOSE
#   - Run the proven live_orchestrator.py repeatedly.
#   - Prevent overlapping poll processes.
#   - Preserve event-level isolation inside live_orchestrator.py.
#   - Maintain append-only operational logs.
#   - Shut down cleanly on SIGINT / SIGTERM.
#
# THIS FILE:
#   - performs NO direct SQLite writes;
#   - performs NO ESPN parsing;
#   - performs NO player identity resolution;
#   - performs NO play ingestion itself.
#
# Sole persistence writer remains:
#   live_ingest.py
#
# Orchestration remains:
#   live_orchestrator.py
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

ORCHESTRATOR = (
    ROOT / "live_orchestrator.py"
)

LIVE_INGEST = (
    ROOT / "live_ingest.py"
)

DATA_DIR = (
    ROOT / "data"
)

LIVE_DB = (
    DATA_DIR / "wfs_live.db"
)

NFL_DB = (
    DATA_DIR / "nfl.db"
)

FORECAST_LEDGER = (
    DATA_DIR / "forecast_ledger.db"
)

LOG_DIR = (
    ROOT / "logs"
)

LOG_FILE = (
    LOG_DIR / "wfs_live_poll.log"
)

LOCK_FILE = (
    ROOT / ".wfs_live_poll.lock"
)


# ======================================================================
# DEFAULTS
# ======================================================================

DEFAULT_INTERVAL_SECONDS = 30

MIN_INTERVAL_SECONDS = 10

DEFAULT_SEASON = 2026


# ======================================================================
# GLOBAL STOP FLAG
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


def log(
    message: str,
) -> None:

    timestamp = utc_now()

    line = (
        f"{timestamp} | {message}"
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

        handle.flush()


def log_block(
    text: str,
) -> None:

    if not text:
        return

    for line in (
        text.rstrip().splitlines()
    ):

        log(
            f"CHILD | {line}"
        )


def request_stop(
    signum,
    frame,
) -> None:

    global STOP_REQUESTED

    STOP_REQUESTED = True

    log(
        "STOP_REQUESTED"
        f" | signal={signum}"
    )


def sleep_interruptibly(
    seconds: int,
) -> None:

    deadline = (
        time.monotonic()
        + seconds
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
                1.0,
                max(
                    remaining,
                    0.0,
                ),
            )
        )


# ======================================================================
# LOCKING
# ======================================================================

def acquire_process_lock():

    LOCK_FILE.touch(
        exist_ok=True
    )

    handle = LOCK_FILE.open(
        "r+",
        encoding="utf-8",
    )

    try:

        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX
            | fcntl.LOCK_NB,
        )

    except BlockingIOError:

        handle.seek(
            0
        )

        existing = (
            handle.read()
            .strip()
        )

        handle.close()

        raise RuntimeError(
            "another LIVE polling process "
            "already owns the lock"
            + (
                f" | lock_info={existing}"
                if existing
                else ""
            )
        )

    handle.seek(
        0
    )

    handle.truncate(
        0
    )

    handle.write(
        f"pid={os.getpid()}"
        f" started={utc_now()}\n"
    )

    handle.flush()

    return handle


# ======================================================================
# PREFLIGHT
# ======================================================================

def preflight() -> dict:

    required_files = [
        (
            PYTHON,
            "venv Python",
        ),
        (
            ORCHESTRATOR,
            "live_orchestrator.py",
        ),
        (
            LIVE_INGEST,
            "live_ingest.py",
        ),
        (
            LIVE_DB,
            "wfs_live.db",
        ),
        (
            NFL_DB,
            "nfl.db",
        ),
        (
            FORECAST_LEDGER,
            "forecast_ledger.db",
        ),
    ]

    for path, label in (
        required_files
    ):

        if not path.is_file():

            raise RuntimeError(
                "required file missing"
                f" | label={label}"
                f" | path={path}"
            )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "orchestrator_sha":
            sha256_file(
                ORCHESTRATOR
            ),

        "ingest_sha":
            sha256_file(
                LIVE_INGEST
            ),

        "nfl_sha":
            sha256_file(
                NFL_DB
            ),

        "ledger_sha":
            sha256_file(
                FORECAST_LEDGER
            ),
    }


# ======================================================================
# CHILD RUNNER
# ======================================================================

def run_one_poll(
    season: int,
    include_post: bool,
    poll_number: int,
) -> dict:

    command = [
        str(
            PYTHON
        ),
        str(
            ORCHESTRATOR
        ),
        "--season",
        str(
            season
        ),
    ]

    if include_post:

        command.append(
            "--include-post"
        )

    else:

        command.append(
            "--no-include-post"
        )

    log(
        "POLL_BEGIN"
        f" | poll={poll_number}"
        f" | command={' '.join(command)}"
    )

    started = (
        time.monotonic()
    )

    process = subprocess.run(
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

    log_block(
        process.stdout
    )

    if process.stderr:

        for line in (
            process.stderr
            .rstrip()
            .splitlines()
        ):

            log(
                f"CHILD_STDERR | {line}"
            )

    status = (
        "PASS"
        if process.returncode == 0
        else "FAIL"
    )

    log(
        "POLL_END"
        f" | poll={poll_number}"
        f" | status={status}"
        f" | return_code={process.returncode}"
        f" | seconds={elapsed:.3f}"
    )

    return {
        "poll":
            poll_number,

        "return_code":
            process.returncode,

        "status":
            status,

        "elapsed":
            elapsed,
    }


# ======================================================================
# CLI
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "WFS NFL LIVE controlled polling wrapper"
        )
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            "Seconds between poll starts. "
            f"Minimum {MIN_INTERVAL_SECONDS}. "
            f"Default {DEFAULT_INTERVAL_SECONDS}."
        ),
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
    )

    parser.add_argument(
        "--include-post",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward POST-event eligibility "
            "to live_orchestrator.py."
        ),
    )

    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help=(
            "Stop after N polls. "
            "0 means continue until SIGINT/SIGTERM."
        ),
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help=(
            "Stop polling if live_orchestrator.py "
            "returns nonzero."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run exactly one orchestrator poll "
            "and exit."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    global STOP_REQUESTED

    args = parse_args()

    if args.interval < (
        MIN_INTERVAL_SECONDS
    ):

        raise SystemExit(
            "FAIL | interval too aggressive"
            f" | minimum={MIN_INTERVAL_SECONDS}"
        )

    if args.max_polls < 0:

        raise SystemExit(
            "FAIL | max-polls cannot be negative"
        )

    if args.once:

        args.max_polls = 1

    print(
        "=" * 72
    )

    print(
        "WFS NFL LIVE — STAGE LIVE-14"
    )

    print(
        "CONTROLLED RECURRING POLLING WRAPPER"
    )

    print(
        "LOCKED + SEQUENTIAL + LOGGED"
    )

    print(
        "=" * 72
    )

    # ==================================================================
    # 1. PREFLIGHT
    # ==================================================================

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "LIVE14_START"
        f" | pid={os.getpid()}"
    )

    baseline = (
        preflight()
    )

    log(
        "PREFLIGHT_PASS"
        f" | orchestrator_sha="
        f"{baseline['orchestrator_sha']}"
        f" | ingest_sha="
        f"{baseline['ingest_sha']}"
    )

    log(
        "PROTECTED_BASELINE"
        f" | nfl_sha={baseline['nfl_sha']}"
        f" | ledger_sha={baseline['ledger_sha']}"
    )

    # ==================================================================
    # 2. ACQUIRE SINGLE-PROCESS LOCK
    # ==================================================================

    try:

        lock_handle = (
            acquire_process_lock()
        )

    except RuntimeError as exc:

        log(
            f"LOCK_FAIL | {exc}"
        )

        raise SystemExit(
            2
        )

    log(
        f"LOCK_ACQUIRED | path={LOCK_FILE}"
    )

    # ==================================================================
    # 3. SIGNAL HANDLERS
    # ==================================================================

    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    # ==================================================================
    # 4. POLLING LOOP
    # ==================================================================

    poll_number = 0

    success_count = 0

    failure_count = 0

    try:

        while (
            not STOP_REQUESTED
        ):

            poll_number += 1

            cycle_started = (
                time.monotonic()
            )

            result = run_one_poll(
                season=args.season,
                include_post=args.include_post,
                poll_number=poll_number,
            )

            if (
                result[
                    "return_code"
                ]
                == 0
            ):

                success_count += 1

            else:

                failure_count += 1

                log(
                    "POLL_FAILURE_SURFACED"
                    f" | poll={poll_number}"
                )

                if (
                    args.stop_on_failure
                ):

                    log(
                        "STOP_ON_FAILURE_TRIGGERED"
                        f" | poll={poll_number}"
                    )

                    break

            if (
                args.max_polls > 0
                and poll_number
                >= args.max_polls
            ):

                log(
                    "MAX_POLLS_REACHED"
                    f" | max_polls={args.max_polls}"
                )

                break

            if STOP_REQUESTED:

                break

            elapsed = (
                time.monotonic()
                - cycle_started
            )

            sleep_seconds = max(
                0.0,
                args.interval
                - elapsed,
            )

            log(
                "POLL_SLEEP"
                f" | poll={poll_number}"
                f" | requested_interval={args.interval}"
                f" | sleep_seconds={sleep_seconds:.3f}"
            )

            sleep_interruptibly(
                sleep_seconds
            )

    finally:

        # ==============================================================
        # 5. VERIFY FROZEN EXECUTABLE SURFACES
        # ==============================================================

        orchestrator_sha_after = (
            sha256_file(
                ORCHESTRATOR
            )
        )

        ingest_sha_after = (
            sha256_file(
                LIVE_INGEST
            )
        )

        nfl_sha_after = (
            sha256_file(
                NFL_DB
            )
        )

        ledger_sha_after = (
            sha256_file(
                FORECAST_LEDGER
            )
        )

        orchestrator_unchanged = (
            orchestrator_sha_after
            == baseline[
                "orchestrator_sha"
            ]
        )

        ingest_unchanged = (
            ingest_sha_after
            == baseline[
                "ingest_sha"
            ]
        )

        nfl_unchanged = (
            nfl_sha_after
            == baseline[
                "nfl_sha"
            ]
        )

        ledger_unchanged = (
            ledger_sha_after
            == baseline[
                "ledger_sha"
            ]
        )

        log(
            "FINAL_PROTECTED_CHECK"
            f" | orchestrator_unchanged="
            f"{int(orchestrator_unchanged)}"
            f" | ingest_unchanged="
            f"{int(ingest_unchanged)}"
            f" | nfl_db_unchanged="
            f"{int(nfl_unchanged)}"
            f" | forecast_ledger_unchanged="
            f"{int(ledger_unchanged)}"
        )

        log(
            "LIVE14_SUMMARY"
            f" | polls={poll_number}"
            f" | successes={success_count}"
            f" | failures={failure_count}"
        )

        # Release advisory lock by closing handle.
        try:

            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_UN,
            )

        finally:

            lock_handle.close()

        log(
            "LOCK_RELEASED"
        )

    # ==================================================================
    # 6. FINAL CONTRACT
    # ==================================================================

    protected_ok = (
        orchestrator_unchanged
        and ingest_unchanged
        and nfl_unchanged
        and ledger_unchanged
    )

    print()
    print(
        "=" * 72
    )

    if (
        failure_count == 0
        and protected_ok
    ):

        print(
            "PASS | LIVE-14 CONTROLLED "
            "POLLING COMPLETE"
        )

        final_status = (
            "PASS"
        )

    else:

        print(
            "FAIL | LIVE-14 CONTROLLED "
            "POLLING COMPLETED WITH FAILURES"
        )

        final_status = (
            "FAIL"
        )

    print(
        "=" * 72
    )

    print(
        f"POLLS_EXECUTED={poll_number}"
    )

    print(
        f"SUCCESSFUL_POLLS={success_count}"
    )

    print(
        f"FAILED_POLLS={failure_count}"
    )

    print(
        "LIVE_ORCHESTRATOR_UNCHANGED="
        f"{int(orchestrator_unchanged)}"
    )

    print(
        "LIVE_INGEST_UNCHANGED="
        f"{int(ingest_unchanged)}"
    )

    print(
        f"NFL_DB_UNCHANGED={int(nfl_unchanged)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        f"POLL_LOCK_RELEASED=1"
    )

    print(
        f"LOG_FILE={LOG_FILE}"
    )

    print(
        f"LIVE14_STATUS={final_status}"
    )

    if (
        failure_count > 0
        or not protected_ok
    ):

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
