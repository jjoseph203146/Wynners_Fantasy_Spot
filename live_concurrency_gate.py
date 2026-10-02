from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path


# ======================================================================
# WFS NFL LIVE — STAGE LIVE-16R
#
# PROTECTED DB CONCURRENCY CONTRACT
# CORRECTED FAILURE CLASSIFIER
#
# PURPOSE
#   Distinguish:
#
#       legitimate external mutation of data/nfl.db
#
#   from:
#
#       an actual write attempt to data/nfl.db by the LIVE process tree.
#
# IMPORTANT
#   - live_ingest.py is NOT modified.
#   - live_orchestrator.py is NOT modified.
#   - no database schema is modified.
#   - forecast_ledger.db remains strictly immutable.
#   - live_ingest.py and live_orchestrator.py remain byte-for-byte frozen.
#
# LIVE-16R FIX
#   EVENT_OUTCOME=SUCCESS is explicitly treated as success and is never
#   placed into the hard-failure set.
#
# This wrapper may normalize ONLY the proven nfl.db external-hash-drift
# condition when:
#
#   - LIVE made zero writable opens against nfl.db;
#   - all event outcomes were successful;
#   - no explicit failure exists except "FAIL | nfl.db changed";
#   - forecast_ledger.db remained unchanged;
#   - LIVE code remained unchanged;
#   - all database integrity/FK gates remain clean.
#
# Every other condition fails closed.
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

INGEST = (
    ROOT / "live_ingest.py"
)

UPDATER = (
    ROOT / "run_updater.sh"
)

DATA_DIR = (
    ROOT / "data"
)

NFL_DB = (
    DATA_DIR / "nfl.db"
)

LIVE_DB = (
    DATA_DIR / "wfs_live.db"
)

FORECAST_LEDGER = (
    DATA_DIR / "forecast_ledger.db"
)

LOG_DIR = (
    ROOT / "logs"
)

CONCURRENCY_LOG = (
    LOG_DIR / "wfs_live_concurrency.log"
)

EXPECTED_LIVE_SCHEMA_VERSION = (
    "WFS_LIVE_DB_V2"
)

EXPECTED_LIVE_SCHEMA_SHA256 = (
    "ea078eaf3115ea9e3894f51813ebee80"
    "f1168ced48ef9da557db00fc643d9bfa"
)

DEFAULT_SEASON = 2026


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

    with CONCURRENCY_LOG.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            line + "\n"
        )


def schema_fingerprint(
    conn: sqlite3.Connection,
) -> str:

    rows = conn.execute(
        """
        SELECT
            type,
            name,
            tbl_name,
            COALESCE(sql, '')
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY
            type,
            name,
            tbl_name
        """
    ).fetchall()

    normalized = "\n".join(
        "|".join(
            str(value)
            for value in row
        )
        for row in rows
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def sqlite_health(
    path: Path,
):

    uri = (
        f"file:{path}?mode=ro"
    )

    conn = sqlite3.connect(
        uri,
        uri=True,
    )

    try:

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk_errors = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        return (
            integrity,
            len(
                fk_errors
            ),
        )

    finally:

        conn.close()


def live_schema_contract():

    uri = (
        f"file:{LIVE_DB}?mode=ro"
    )

    conn = sqlite3.connect(
        uri,
        uri=True,
    )

    try:

        version_row = conn.execute(
            """
            SELECT schema_version
            FROM live_meta
            """
        ).fetchone()

        if version_row is None:

            return (
                None,
                None,
            )

        version = (
            version_row[0]
        )

        fingerprint = (
            schema_fingerprint(
                conn
            )
        )

        return (
            version,
            fingerprint,
        )

    finally:

        conn.close()


# ======================================================================
# STRACE ANALYSIS
# ======================================================================

WRITE_FLAG_PATTERNS = (
    "O_WRONLY",
    "O_RDWR",
    "O_CREAT",
    "O_TRUNC",
    "O_APPEND",
)


def analyze_trace(
    trace_text: str,
):

    nfl_path = str(
        NFL_DB
    )

    all_nfl_lines = []

    writable_nfl_lines = []

    for raw_line in (
        trace_text.splitlines()
    ):

        if nfl_path not in raw_line:

            continue

        all_nfl_lines.append(
            raw_line
        )

        if any(
            flag in raw_line
            for flag in WRITE_FLAG_PATTERNS
        ):

            writable_nfl_lines.append(
                raw_line
            )

    return {
        "all_nfl_lines":
            all_nfl_lines,

        "writable_nfl_lines":
            writable_nfl_lines,
    }


# ======================================================================
# ORCHESTRATOR OUTPUT CONTRACT
# ======================================================================

def classify_orchestrator_output(
    stdout: str,
    stderr: str,
    return_code: int,
):

    combined = (
        stdout
        + "\n"
        + stderr
    )

    lines = [
        line.strip()
        for line in combined.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------------
    # EXPLICIT FAIL SURFACES
    # --------------------------------------------------------------

    explicit_fail_lines = [
        line
        for line in lines
        if "FAIL |" in line
    ]

    # --------------------------------------------------------------
    # EVENT OUTCOMES
    #
    # LIVE-16R FIX:
    #
    # EVENT_OUTCOME=SUCCESS is success.
    # It must NEVER be classified as a hard failure.
    # --------------------------------------------------------------

    event_outcome_lines = [
        line
        for line in lines
        if line.startswith(
            "EVENT_OUTCOME="
        )
    ]

    # --------------------------------------------------------------
    # EVENT FAILURE SEVERITY
    #
    # SAFE_FAIL means the child rejected the event before commit.
    # It remains visible/auditable but is not a concurrency-layer
    # hard failure.
    #
    # COMMITTED_FAILURE and CONTRACT_FAIL remain hard failures.
    # --------------------------------------------------------------

    event_safe_fail_lines = [
        line
        for line in event_outcome_lines
        if line == "EVENT_OUTCOME=SAFE_FAIL"
    ]

    event_failure_lines = [
        line
        for line in event_outcome_lines
        if line in {
            "EVENT_OUTCOME=COMMITTED_FAILURE",
            "EVENT_OUTCOME=CONTRACT_FAIL",
        }
    ]

    event_success_lines = [
        line
        for line in event_outcome_lines
        if line == "EVENT_OUTCOME=SUCCESS"
    ]

    # --------------------------------------------------------------
    # KNOWN EXTERNAL NFL.DB DRIFT FAILURE
    # --------------------------------------------------------------

    nfl_changed_lines = [
        line
        for line in explicit_fail_lines
        if "FAIL | nfl.db changed"
        in line
    ]

    # --------------------------------------------------------------
    # EVERY OTHER EXPLICIT FAIL REMAINS HARD
    # --------------------------------------------------------------

    safe_prewrite_fail_lines = [
        line
        for line in explicit_fail_lines
        if (
            "FAIL | ambiguous identities remain "
            "— DATABASE UNCHANGED"
            in line
            or
            "FAIL | unresolved identities remain "
            "— DATABASE UNCHANGED"
            in line
        )
    ]

    # LIVE-12 itself returns nonzero when one or more child events
    # SAFE_FAIL. The aggregate summary line is therefore informational
    # to this concurrency layer when every failed child was fail-closed.
    #
    # It is NOT normalized when a COMMITTED_FAILURE or CONTRACT_FAIL
    # event exists; those remain in event_failure_lines and remain hard.
    safe_fail_only_aggregate_lines = [
        line
        for line in explicit_fail_lines
        if (
            "FAIL | LIVE-12 MULTI-EVENT "
            "ORCHESTRATION COMPLETED WITH EVENT FAILURES"
            in line
            and len(event_safe_fail_lines) > 0
            and len(event_failure_lines) == 0
        )
    ]

    other_explicit_fail_lines = [
        line
        for line in explicit_fail_lines
        if (
            "FAIL | nfl.db changed"
            not in line
            and line not in safe_prewrite_fail_lines
            and line not in safe_fail_only_aggregate_lines
        )
    ]

    hard_fail_lines = (
        other_explicit_fail_lines
        + event_failure_lines
    )

    # --------------------------------------------------------------
    # SUPPORTING CONTRACT SIGNALS
    # --------------------------------------------------------------

    has_integrity_ok = (
        "INTEGRITY_CHECK=ok"
        in combined
        or "POST_INTEGRITY=ok"
        in combined
    )

    has_fk_zero = (
        "FOREIGN_KEY_ERRORS=0"
        in combined
        or "POST_FK_ERRORS=0"
        in combined
    )

    forecast_ledger_ok = (
        "FORECAST_LEDGER_UNCHANGED=1"
        in combined
    )

    ingest_unchanged = (
        "LIVE_INGEST_UNCHANGED=1"
        in combined
        or
        "PASS | live_ingest.py unchanged"
        in combined
    )

    all_events_success = (
        len(
            event_failure_lines
        ) == 0
    )

    # --------------------------------------------------------------
    # NATIVE PASS
    # --------------------------------------------------------------

    if return_code == 0:

        classification = (
            "NATIVE_PASS"
        )

    else:

        # ----------------------------------------------------------
        # AUTHORIZED EXTERNAL NFL.DB DRIFT CANDIDATE
        #
        # We require:
        #
        #   - orchestrator nonzero;
        #   - known nfl.db-change failure is present;
        #   - no unrelated explicit FAIL;
        #   - no failed event outcomes;
        #   - database output says integrity clean;
        #   - FK output says clean;
        #   - forecast ledger unchanged;
        #   - live_ingest unchanged.
        #
        # Process-level proof of zero LIVE write opens is checked
        # separately before this result can be normalized.
        # ----------------------------------------------------------

        known_nfl_drift_only = (
            len(
                nfl_changed_lines
            ) > 0
            and len(
                hard_fail_lines
            ) == 0
            and all_events_success
            and has_integrity_ok
            and has_fk_zero
            and forecast_ledger_ok
            and ingest_unchanged
        )

        safe_fail_only = (
            len(
                event_safe_fail_lines
            ) > 0
            and len(
                hard_fail_lines
            ) == 0
            and has_integrity_ok
            and has_fk_zero
            and forecast_ledger_ok
            and ingest_unchanged
        )

        if safe_fail_only:

            classification = (
                "SAFE_FAIL_ONLY"
            )

        elif known_nfl_drift_only:

            classification = (
                "NFL_EXTERNAL_DRIFT_CANDIDATE"
            )

        else:

            classification = (
                "HARD_FAILURE"
            )

    return {
        "classification":
            classification,

        "nfl_changed_lines":
            nfl_changed_lines,

        "event_outcome_lines":
            event_outcome_lines,

        "event_success_lines":
            event_success_lines,

        "event_safe_fail_lines":
            event_safe_fail_lines,

        "event_failure_lines":
            event_failure_lines,

        # Retained for compatibility with original LIVE-16 output.
        "committed_failure_lines":
            event_failure_lines,

        "hard_fail_lines":
            hard_fail_lines,

        "integrity_ok":
            has_integrity_ok,

        "fk_zero":
            has_fk_zero,

        "forecast_ledger_ok":
            forecast_ledger_ok,

        "ingest_unchanged":
            ingest_unchanged,

        "all_events_success":
            all_events_success,
    }


# ======================================================================
# OPTIONAL AUTHORIZED UPDATER OVERLAP
# ======================================================================

def start_authorized_updater():

    if not UPDATER.is_file():

        raise RuntimeError(
            "run_updater.sh missing"
        )

    log(
        "AUTHORIZED_UPDATER_START"
        f" | path={UPDATER}"
    )

    return subprocess.Popen(
        [
            "/bin/bash",
            str(
                UPDATER
            ),
        ],
        cwd=str(
            ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "WFS LIVE-16R protected DB "
            "concurrency contract"
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
    )

    parser.add_argument(
        "--with-updater",
        action="store_true",
        help=(
            "Deliberately start the authorized "
            "NFL updater during the LIVE test."
        ),
    )

    parser.add_argument(
        "--updater-delay",
        type=float,
        default=1.0,
        help=(
            "Seconds after launching the traced "
            "orchestrator to launch run_updater.sh."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    print(
        "=" * 72
    )

    print(
        "WFS NFL LIVE — STAGE LIVE-16R"
    )

    print(
        "PROTECTED DB CONCURRENCY CONTRACT"
    )

    print(
        "CORRECTED FAILURE CLASSIFIER"
    )

    print(
        "=" * 72
    )

    # ==================================================================
    # 1. PREFLIGHT
    # ==================================================================

    print()
    print(
        "=== 1. PREFLIGHT ==="
    )

    required = [
        PYTHON,
        ORCHESTRATOR,
        INGEST,
        NFL_DB,
        LIVE_DB,
        FORECAST_LEDGER,
    ]

    for path in required:

        if not path.is_file():

            raise SystemExit(
                "FAIL | missing required file"
                f" | {path}"
            )

        print(
            f"PASS | exists | {path}"
        )

    strace_path = (
        shutil.which(
            "strace"
        )
    )

    if not strace_path:

        raise SystemExit(
            "FAIL | strace is required for LIVE-16R "
            "but is not installed"
        )

    print(
        f"PASS | strace={strace_path}"
    )

    if (
        args.with_updater
        and not UPDATER.is_file()
    ):

        raise SystemExit(
            "FAIL | --with-updater requested "
            "but run_updater.sh is missing"
        )

    # ==================================================================
    # 2. FROZEN BASELINES
    # ==================================================================

    print()
    print(
        "=== 2. FROZEN BASELINES ==="
    )

    orchestrator_sha_before = (
        sha256_file(
            ORCHESTRATOR
        )
    )

    ingest_sha_before = (
        sha256_file(
            INGEST
        )
    )

    ledger_sha_before = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    nfl_sha_before = (
        sha256_file(
            NFL_DB
        )
    )

    live_schema_version_before, (
        live_schema_sha_before
    ) = live_schema_contract()

    print(
        "LIVE_ORCHESTRATOR_SHA_BEFORE="
        f"{orchestrator_sha_before}"
    )

    print(
        "LIVE_INGEST_SHA_BEFORE="
        f"{ingest_sha_before}"
    )

    print(
        "FORECAST_LEDGER_SHA_BEFORE="
        f"{ledger_sha_before}"
    )

    print(
        f"NFL_DB_SHA_BEFORE={nfl_sha_before}"
    )

    print(
        "LIVE_SCHEMA_VERSION_BEFORE="
        f"{live_schema_version_before}"
    )

    print(
        "LIVE_SCHEMA_SHA_BEFORE="
        f"{live_schema_sha_before}"
    )

    if (
        live_schema_version_before
        != EXPECTED_LIVE_SCHEMA_VERSION
    ):

        raise SystemExit(
            "FAIL | LIVE schema version mismatch"
        )

    if (
        live_schema_sha_before
        != EXPECTED_LIVE_SCHEMA_SHA256
    ):

        raise SystemExit(
            "FAIL | LIVE schema fingerprint mismatch"
        )

    # ==================================================================
    # 3. PRE-HEALTH
    # ==================================================================

    print()
    print(
        "=== 3. PRE-RUN DATABASE HEALTH ==="
    )

    nfl_integrity_before, (
        nfl_fk_before
    ) = sqlite_health(
        NFL_DB
    )

    live_integrity_before, (
        live_fk_before
    ) = sqlite_health(
        LIVE_DB
    )

    ledger_integrity_before, (
        ledger_fk_before
    ) = sqlite_health(
        FORECAST_LEDGER
    )

    print(
        f"NFL_INTEGRITY_BEFORE={nfl_integrity_before}"
    )

    print(
        f"NFL_FK_BEFORE={nfl_fk_before}"
    )

    print(
        f"LIVE_INTEGRITY_BEFORE={live_integrity_before}"
    )

    print(
        f"LIVE_FK_BEFORE={live_fk_before}"
    )

    print(
        "LEDGER_INTEGRITY_BEFORE="
        f"{ledger_integrity_before}"
    )

    print(
        f"LEDGER_FK_BEFORE={ledger_fk_before}"
    )

    if (
        nfl_integrity_before != "ok"
        or nfl_fk_before != 0
        or live_integrity_before != "ok"
        or live_fk_before != 0
        or ledger_integrity_before != "ok"
        or ledger_fk_before != 0
    ):

        raise SystemExit(
            "FAIL | pre-run database health gate"
        )

    # ==================================================================
    # 4. PREPARE TRACE
    # ==================================================================

    print()
    print(
        "=== 4. TRACE LIVE PROCESS TREE ==="
    )

    trace_file_handle = (
        tempfile.NamedTemporaryFile(
            mode="w",
            prefix="wfs_live16r_",
            suffix=".strace",
            delete=False,
            dir="/tmp",
        )
    )

    trace_path = Path(
        trace_file_handle.name
    )

    trace_file_handle.close()

    command = [
        strace_path,
        "-f",
        "-qq",
        "-e",
        (
            "trace=open,openat,creat,"
            "rename,renameat,renameat2,"
            "unlink,unlinkat,truncate,ftruncate"
        ),
        "-o",
        str(
            trace_path
        ),
        str(
            PYTHON
        ),
        str(
            ORCHESTRATOR
        ),
        "--season",
        str(
            args.season
        ),
        "--include-post",
    ]

    print(
        "TRACE_COMMAND="
        + " ".join(
            command
        )
    )

    # ==================================================================
    # 5. RUN
    # ==================================================================

    print()
    print(
        "=== 5. RUN CONTROLLED CONCURRENCY TEST ==="
    )

    started = (
        time.monotonic()
    )

    orchestrator_process = subprocess.Popen(
        command,
        cwd=str(
            ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    updater_process = None

    if args.with_updater:

        time.sleep(
            max(
                0.0,
                args.updater_delay,
            )
        )

        updater_process = (
            start_authorized_updater()
        )

    orchestrator_stdout, (
        orchestrator_stderr
    ) = orchestrator_process.communicate()

    elapsed = (
        time.monotonic()
        - started
    )

    updater_stdout = ""
    updater_stderr = ""
    updater_rc = None

    if updater_process is not None:

        updater_stdout, (
            updater_stderr
        ) = updater_process.communicate()

        updater_rc = (
            updater_process.returncode
        )

    print()
    print(
        "--- LIVE ORCHESTRATOR STDOUT ---"
    )

    print(
        orchestrator_stdout.rstrip()
        if orchestrator_stdout.strip()
        else "<EMPTY>"
    )

    print()
    print(
        "--- LIVE ORCHESTRATOR STDERR ---"
    )

    print(
        orchestrator_stderr.rstrip()
        if orchestrator_stderr.strip()
        else "<EMPTY>"
    )

    print()
    print(
        "ORCHESTRATOR_RETURN_CODE="
        f"{orchestrator_process.returncode}"
    )

    print(
        f"ORCHESTRATOR_SECONDS={elapsed:.3f}"
    )

    if updater_process is not None:

        print()
        print(
            "--- AUTHORIZED UPDATER RESULT ---"
        )

        print(
            f"UPDATER_RETURN_CODE={updater_rc}"
        )

        if updater_stdout.strip():

            print(
                updater_stdout.rstrip()
            )

        if updater_stderr.strip():

            print(
                "--- UPDATER STDERR ---"
            )

            print(
                updater_stderr.rstrip()
            )

    # ==================================================================
    # 6. TRACE ANALYSIS
    # ==================================================================

    print()
    print(
        "=== 6. NFL.DB PROCESS-LEVEL WRITE AUDIT ==="
    )

    trace_text = (
        trace_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    trace_result = (
        analyze_trace(
            trace_text
        )
    )

    nfl_open_count = len(
        trace_result[
            "all_nfl_lines"
        ]
    )

    nfl_write_open_count = len(
        trace_result[
            "writable_nfl_lines"
        ]
    )

    print(
        f"LIVE_NFL_DB_OPEN_EVENTS={nfl_open_count}"
    )

    print(
        "LIVE_NFL_DB_WRITE_OPEN_EVENTS="
        f"{nfl_write_open_count}"
    )

    if (
        trace_result[
            "all_nfl_lines"
        ]
    ):

        print()
        print(
            "--- NFL.DB TRACE EVENTS ---"
        )

        for line in (
            trace_result[
                "all_nfl_lines"
            ]
        ):

            print(
                line
            )

    if (
        trace_result[
            "writable_nfl_lines"
        ]
    ):

        print()
        print(
            "--- FORBIDDEN NFL.DB WRITE EVENTS ---"
        )

        for line in (
            trace_result[
                "writable_nfl_lines"
            ]
        ):

            print(
                line
            )

    if nfl_write_open_count != 0:

        raise SystemExit(
            "FAIL | LIVE process tree attempted "
            "writable access to nfl.db"
        )

    print(
        "PASS | LIVE process tree made zero "
        "writable nfl.db opens"
    )

    # ==================================================================
    # 7. OUTPUT CLASSIFICATION
    # ==================================================================

    print()
    print(
        "=== 7. ORCHESTRATOR FAILURE CLASSIFICATION ==="
    )

    classification = (
        classify_orchestrator_output(
            orchestrator_stdout,
            orchestrator_stderr,
            orchestrator_process.returncode,
        )
    )

    print(
        "RAW_CLASSIFICATION="
        f"{classification['classification']}"
    )

    print(
        "NFL_CHANGED_FAILURE_LINES="
        f"{len(classification['nfl_changed_lines'])}"
    )

    print(
        "EVENT_OUTCOME_LINES="
        f"{len(classification['event_outcome_lines'])}"
    )

    print(
        "EVENT_SUCCESS_LINES="
        f"{len(classification['event_success_lines'])}"
    )

    print(
        "EVENT_FAILURE_LINES="
        f"{len(classification['event_failure_lines'])}"
    )

    print(
        "OTHER_HARD_FAILURE_LINES="
        f"{len(classification['hard_fail_lines'])}"
    )

    print(
        "OUTPUT_INTEGRITY_OK="
        f"{int(classification['integrity_ok'])}"
    )

    print(
        "OUTPUT_FK_ZERO="
        f"{int(classification['fk_zero'])}"
    )

    print(
        "OUTPUT_LEDGER_UNCHANGED="
        f"{int(classification['forecast_ledger_ok'])}"
    )

    print(
        "OUTPUT_INGEST_UNCHANGED="
        f"{int(classification['ingest_unchanged'])}"
    )

    if (
        classification[
            "hard_fail_lines"
        ]
    ):

        print()
        print(
            "--- HARD FAILURE LINES ---"
        )

        for line in (
            classification[
                "hard_fail_lines"
            ]
        ):

            print(
                line
            )

    # ==================================================================
    # 8. POST-RUN PHYSICAL VERIFICATION
    # ==================================================================

    print()
    print(
        "=== 8. POST-RUN PHYSICAL VERIFICATION ==="
    )

    orchestrator_sha_after = (
        sha256_file(
            ORCHESTRATOR
        )
    )

    ingest_sha_after = (
        sha256_file(
            INGEST
        )
    )

    ledger_sha_after = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    nfl_sha_after = (
        sha256_file(
            NFL_DB
        )
    )

    live_schema_version_after, (
        live_schema_sha_after
    ) = live_schema_contract()

    orchestrator_unchanged = (
        orchestrator_sha_after
        == orchestrator_sha_before
    )

    ingest_unchanged = (
        ingest_sha_after
        == ingest_sha_before
    )

    ledger_unchanged = (
        ledger_sha_after
        == ledger_sha_before
    )

    nfl_changed = (
        nfl_sha_after
        != nfl_sha_before
    )

    live_schema_unchanged = (
        live_schema_version_after
        == live_schema_version_before
        and live_schema_sha_after
        == live_schema_sha_before
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
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        f"NFL_DB_CHANGED={int(nfl_changed)}"
    )

    print(
        "LIVE_SCHEMA_UNCHANGED="
        f"{int(live_schema_unchanged)}"
    )

    print(
        f"NFL_DB_SHA_AFTER={nfl_sha_after}"
    )

    # ==================================================================
    # 9. POST-HEALTH
    # ==================================================================

    nfl_integrity_after, (
        nfl_fk_after
    ) = sqlite_health(
        NFL_DB
    )

    live_integrity_after, (
        live_fk_after
    ) = sqlite_health(
        LIVE_DB
    )

    ledger_integrity_after, (
        ledger_fk_after
    ) = sqlite_health(
        FORECAST_LEDGER
    )

    print(
        f"NFL_INTEGRITY_AFTER={nfl_integrity_after}"
    )

    print(
        f"NFL_FK_AFTER={nfl_fk_after}"
    )

    print(
        f"LIVE_INTEGRITY_AFTER={live_integrity_after}"
    )

    print(
        f"LIVE_FK_AFTER={live_fk_after}"
    )

    print(
        "LEDGER_INTEGRITY_AFTER="
        f"{ledger_integrity_after}"
    )

    print(
        f"LEDGER_FK_AFTER={ledger_fk_after}"
    )

    physical_health_ok = (
        nfl_integrity_after == "ok"
        and nfl_fk_after == 0
        and live_integrity_after == "ok"
        and live_fk_after == 0
        and ledger_integrity_after == "ok"
        and ledger_fk_after == 0
    )

    frozen_surfaces_ok = (
        orchestrator_unchanged
        and ingest_unchanged
        and ledger_unchanged
        and live_schema_unchanged
    )

    if not physical_health_ok:

        raise SystemExit(
            "FAIL | post-run database health failure"
        )

    if not frozen_surfaces_ok:

        raise SystemExit(
            "FAIL | frozen surface changed"
        )

    # ==================================================================
    # 10. CONCURRENCY DECISION
    # ==================================================================

    print()
    print(
        "=== 10. LIVE-16R CONCURRENCY DECISION ==="
    )

    raw_class = (
        classification[
            "classification"
        ]
    )

    updater_ok = (
        updater_process is None
        or updater_rc == 0
    )

    if (
        raw_class
        == "NATIVE_PASS"
    ):

        final_status = (
            "PASS"
        )

        concurrency_result = (
            "NO_CONFLICT"
        )

        print(
            "PASS | orchestrator completed natively"
        )

        print(
            "PASS | LIVE made zero writable "
            "nfl.db opens"
        )

    elif (
        raw_class
        == "SAFE_FAIL_ONLY"
        and nfl_write_open_count == 0
        and frozen_surfaces_ok
        and physical_health_ok
        and updater_ok
    ):

        final_status = (
            "PASS"
        )

        concurrency_result = (
            "SAFE_FAIL_ONLY_CONTINUE"
        )

        print(
            "PASS | child SAFE_FAIL events remained fail-closed"
        )

        print(
            "PASS | no committed or contract event failures"
        )

        print(
            "PASS | LIVE made zero writable "
            "nfl.db opens"
        )

        print(
            "PASS | recurring polling may continue"
        )

    elif (
        raw_class
        == "NFL_EXTERNAL_DRIFT_CANDIDATE"
        and nfl_write_open_count == 0
        and nfl_changed
        and frozen_surfaces_ok
        and physical_health_ok
        and updater_ok
    ):

        final_status = (
            "PASS"
        )

        concurrency_result = (
            "AUTHORIZED_EXTERNAL_NFL_DB_DRIFT"
        )

        print(
            "PASS | nfl.db changed concurrently"
        )

        print(
            "PASS | LIVE process tree made zero "
            "writable nfl.db opens"
        )

        print(
            "PASS | all event outcomes were successful"
        )

        print(
            "PASS | only explicit failure was "
            "the nfl.db hash-drift gate"
        )

        print(
            "PASS | forecast ledger unchanged"
        )

        print(
            "PASS | LIVE schema unchanged"
        )

        print(
            "PASS | frozen LIVE code unchanged"
        )

        print(
            "PASS | all databases remain healthy"
        )

        if updater_process is not None:

            print(
                "PASS | authorized updater completed "
                "successfully"
            )

        print(
            "PASS | external nfl.db mutation "
            "classified as authorized coexistence"
        )

    else:

        final_status = (
            "FAIL"
        )

        concurrency_result = (
            "UNCLASSIFIED_OR_HARD_FAILURE"
        )

        print(
            "FAIL | concurrency result cannot be "
            "safely normalized"
        )

    # ==================================================================
    # 11. PRESERVE TRACE
    # ==================================================================

    final_trace_path = (
        LOG_DIR
        / (
            "live16r_trace_"
            + datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            + ".strace"
        )
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.move(
        str(
            trace_path
        ),
        str(
            final_trace_path
        ),
    )

    log(
        "LIVE16R_RESULT"
        f" | status={final_status}"
        f" | result={concurrency_result}"
        f" | nfl_changed={int(nfl_changed)}"
        f" | live_nfl_write_opens={nfl_write_open_count}"
        f" | trace={final_trace_path}"
    )

    # ==================================================================
    # 12. FINAL OUTPUT
    # ==================================================================

    print()
    print(
        "=" * 72
    )

    if final_status == "PASS":

        print(
            "PASS | LIVE-16R PROTECTED DB "
            "CONCURRENCY CONTRACT COMPLETE"
        )

    else:

        print(
            "FAIL | LIVE-16R PROTECTED DB "
            "CONCURRENCY CONTRACT FAILED"
        )

    print(
        "=" * 72
    )

    print(
        f"LIVE16R_STATUS={final_status}"
    )

    print(
        f"CONCURRENCY_RESULT={concurrency_result}"
    )

    print(
        "LIVE_NFL_DB_WRITE_OPEN_EVENTS="
        f"{nfl_write_open_count}"
    )

    print(
        f"NFL_DB_CHANGED={int(nfl_changed)}"
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
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        "LIVE_SCHEMA_UNCHANGED="
        f"{int(live_schema_unchanged)}"
    )

    print(
        f"ALL_DB_HEALTH_OK={int(physical_health_ok)}"
    )

    print(
        "EVENT_FAILURE_LINES="
        f"{len(classification['event_failure_lines'])}"
    )

    print(
        "OTHER_HARD_FAILURE_LINES="
        f"{len(classification['hard_fail_lines'])}"
    )

    print(
        f"TRACE_FILE={final_trace_path}"
    )

    if updater_process is not None:

        print(
            f"AUTHORIZED_UPDATER_RC={updater_rc}"
        )

    if final_status != "PASS":

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
