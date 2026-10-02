#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT = Path("/home/mwynn/nfl_data_engine")

WINDOWS_IP = "192.168.88.252"
WINDOWS_PORT = 22
WINDOWS_USER = "mwynn"
WINDOWS_HOST_EXPECTED = "WYNN_OFFICE"

WINDOWS_MAC = "28:00:AF:FD:D4:AF"
WOL_BROADCAST = "192.168.88.255"
WOL_PORT = 9

SSH_KEY = Path("/home/mwynn/.ssh/wfs_windows_refresh_ed25519")

TASK_NAME = "WFS FanDuel Production Refresh"
EXPECTED_TASK_SCRIPT = r"C:\WFS\FanDuel\download_validate_upload_all_slates_v4_3.py"

SLATE_DIR = PROJECT / "data" / "fanduel" / "slates"

SOLVER_READY_CSV = PROJECT / "data" / "csv" / "fanduel_solver_ready_pool.csv"
SOLVER_READY_PARQUET = PROJECT / "data" / "parquet" / "fanduel_solver_ready_pool.parquet"
SOLVER_READY_MANIFEST = PROJECT / "data" / "csv" / "fanduel_solver_ready_manifest.csv"
SOLVER_READY_GAPS = PROJECT / "data" / "csv" / "audit_fanduel_solver_ready_gaps.csv"

AUDIT_DIR = PROJECT / "data" / "audits" / "emergency_refresh_v1"

EXPECTED_SLATES = {
    "Main.csv",
    "thu-mon.csv",
    "4 pm only.csv",
    "1pm only.csv",
    "sun-mon.csv",
    "late sun-mon.csv",
    "snf-mnf.csv",
}

SSH_WAIT_SECONDS = 180
SSH_READY_WAIT_SECONDS = 180
TASK_START_WAIT_SECONDS = 60
TASK_COMPLETE_WAIT_SECONDS = 1200
POLL_SECONDS = 5


@dataclass(frozen=True)
class FileState:
    path: str
    size: int
    mtime_ns: int
    sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def file_state(path: Path) -> FileState:
    st = path.stat()

    return FileState(
        path=str(path),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        sha256=sha256_file(path),
    )


def run_command(
    args: Sequence[str],
    *,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
    )


def tcp_open(
    host: str,
    port: int,
    timeout: float = 3.0,
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ConnectTimeout=10",
        f"{WINDOWS_USER}@{WINDOWS_IP}",
    ]


def ssh(
    command: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        [*ssh_base(), command],
        timeout=timeout,
        check=False,
    )


def send_wol() -> None:
    mac_bytes = bytes.fromhex(
        WINDOWS_MAC.replace(":", "").replace("-", "")
    )

    packet = b"\xff" * 6 + mac_bytes * 16

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        for i in range(5):
            s.sendto(packet, (WOL_BROADCAST, WOL_PORT))
            print(
                f"PASS | WOL packet {i + 1}/5 sent "
                f"to {WOL_BROADCAST}:{WOL_PORT}"
            )
            time.sleep(1)


def wait_for_tcp() -> None:
    deadline = time.monotonic() + SSH_WAIT_SECONDS

    while time.monotonic() < deadline:
        if tcp_open(WINDOWS_IP, WINDOWS_PORT):
            print("PASS | Windows TCP/22 reachable.")
            return

        print("WAIT | Windows TCP/22 not reachable yet.")
        time.sleep(POLL_SECONDS)

    raise RuntimeError(
        f"Windows TCP/22 did not become reachable within "
        f"{SSH_WAIT_SECONDS} seconds."
    )


def validate_remote_identity() -> dict:
    result = ssh(
        "echo WFS_EMERGENCY_REFRESH_AUTH_PASS && whoami && hostname",
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Remote SSH identity validation failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    out = result.stdout.strip()

    if "WFS_EMERGENCY_REFRESH_AUTH_PASS" not in out:
        raise RuntimeError("Remote authentication marker missing.")

    if WINDOWS_USER.lower() not in out.lower():
        raise RuntimeError("Expected Windows username not returned.")

    if WINDOWS_HOST_EXPECTED.lower() not in out.lower():
        raise RuntimeError("Expected Windows hostname not returned.")

    return {
        "stdout": out,
        "returncode": result.returncode,
    }


def wait_for_remote_identity() -> dict:
    deadline = time.monotonic() + SSH_READY_WAIT_SECONDS
    attempt = 0
    last_error = "No SSH attempt completed."

    while time.monotonic() < deadline:
        attempt += 1

        try:
            identity = validate_remote_identity()

            print(
                f"PASS | Dedicated SSH identity ready "
                f"on attempt {attempt}."
            )

            return identity

        except Exception as exc:
            last_error = str(exc)

            summary = last_error.strip().splitlines()[-1]

            print(
                f"WAIT | SSH identity not ready | "
                f"attempt={attempt} | {summary}"
            )

            time.sleep(POLL_SECONDS)

    raise RuntimeError(
        f"Remote SSH identity did not become ready within "
        f"{SSH_READY_WAIT_SECONDS} seconds. "
        f"Last error: {last_error}"
    )


def query_task_raw() -> str:
    result = ssh(
        f'schtasks.exe /Query /TN "{TASK_NAME}" /V /FO LIST',
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Scheduled-task query failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def first_field(task_text: str, field_name: str) -> str:
    prefix = field_name.lower() + ":"

    for line in task_text.splitlines():
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()

    raise RuntimeError(f"Task field not found: {field_name}")


def validate_task_definition(task_text: str) -> None:
    if TASK_NAME.lower() not in task_text.lower():
        raise RuntimeError("Expected production task name not found.")

    if EXPECTED_TASK_SCRIPT.lower() not in task_text.lower():
        raise RuntimeError(
            "Expected frozen V4.3 downloader path not found."
        )

    if "Scheduled Task State:" not in task_text:
        raise RuntimeError("Scheduled task state not returned.")

    if "Enabled" not in task_text:
        raise RuntimeError("Production task is not enabled.")


def task_status(task_text: str) -> str:
    return first_field(task_text, "Status")


def task_last_run(task_text: str) -> str:
    return first_field(task_text, "Last Run Time")


def task_last_result(task_text: str) -> str:
    return first_field(task_text, "Last Result")


def validate_local_state() -> dict:
    if not SSH_KEY.is_file():
        raise RuntimeError(f"Missing SSH key: {SSH_KEY}")

    if not SLATE_DIR.is_dir():
        raise RuntimeError(f"Missing slate directory: {SLATE_DIR}")

    present = {
        p.name
        for p in SLATE_DIR.iterdir()
        if p.is_file()
    }

    missing = sorted(EXPECTED_SLATES - present)

    if missing:
        raise RuntimeError(
            "Missing expected FanDuel slate files: "
            + ", ".join(missing)
        )

    required_files = [
        SOLVER_READY_CSV,
        SOLVER_READY_PARQUET,
        SOLVER_READY_MANIFEST,
        SOLVER_READY_GAPS,
    ]

    for path in required_files:
        if not path.is_file():
            raise RuntimeError(
                f"Missing required production artifact: {path}"
            )

    slates = {
        name: asdict(file_state(SLATE_DIR / name))
        for name in sorted(EXPECTED_SLATES)
    }

    artifacts = {
        path.name: asdict(file_state(path))
        for path in required_files
    }

    return {
        "slates": slates,
        "artifacts": artifacts,
    }


def demand_start_task() -> dict:
    try:
        result = ssh(
            f'schtasks.exe /Run /TN "{TASK_NAME}"',
            timeout=30,
        )

    except subprocess.TimeoutExpired as exc:
        return {
            "state": "AMBIGUOUS_TIMEOUT",
            "output": (
                "SSH demand-start command timed out after "
                f"{exc.timeout} seconds. "
                "No second /Run will be issued."
            ),
        }

    combined = (
        result.stdout.strip()
        + "\n"
        + result.stderr.strip()
    ).strip()

    if result.returncode != 0:
        raise RuntimeError(
            "Task Scheduler demand-start failed:\n"
            + combined
        )

    if "SUCCESS" not in combined.upper():
        raise RuntimeError(
            "Task Scheduler did not report SUCCESS:\n"
            + combined
        )

    return {
        "state": "CONFIRMED",
        "output": combined,
    }


def wait_for_new_run(
    previous_last_run: str,
) -> dict:
    deadline = time.monotonic() + TASK_START_WAIT_SECONDS

    while time.monotonic() < deadline:
        task_text = query_task_raw()

        status = task_status(task_text)
        last_run = task_last_run(task_text)
        last_result = task_last_result(task_text)

        print(
            f"CHECK | status={status} | "
            f"last_run={last_run} | "
            f"last_result={last_result}"
        )

        if (
            status.lower() == "running"
            or last_run != previous_last_run
        ):
            return {
                "status": status,
                "last_run": last_run,
                "last_result": last_result,
            }

        time.sleep(2)

    raise RuntimeError(
        "No evidence that a new task execution began."
    )


def wait_for_task_completion(
    previous_last_run: str,
) -> dict:
    deadline = time.monotonic() + TASK_COMPLETE_WAIT_SECONDS

    while time.monotonic() < deadline:
        task_text = query_task_raw()

        status = task_status(task_text)
        last_run = task_last_run(task_text)
        last_result = task_last_result(task_text)

        print(
            f"CHECK | status={status} | "
            f"last_run={last_run} | "
            f"last_result={last_result}"
        )

        if (
            status.lower() == "ready"
            and last_run != previous_last_run
        ):
            if last_result != "0":
                raise RuntimeError(
                    f"Task completed with Last Result "
                    f"{last_result}, expected 0."
                )

            return {
                "status": status,
                "last_run": last_run,
                "last_result": last_result,
            }

        time.sleep(POLL_SECONDS)

    raise RuntimeError(
        f"Task did not complete within "
        f"{TASK_COMPLETE_WAIT_SECONDS} seconds."
    )


def verify_refresh(
    before: dict,
    after: dict,
    run_start_ns: int,
) -> dict:
    slate_results = {}

    for name in sorted(EXPECTED_SLATES):
        old = before["slates"][name]
        new = after["slates"][name]

        refreshed = (
            new["mtime_ns"] > old["mtime_ns"]
            and new["mtime_ns"] >= run_start_ns
        )

        slate_results[name] = {
            "refreshed": refreshed,
            "hash_changed": (
                new["sha256"] != old["sha256"]
            ),
            "before": old,
            "after": new,
        }

        if not refreshed:
            raise RuntimeError(
                f"Slate did not refresh: {name}"
            )

    old_solver = before["artifacts"][SOLVER_READY_CSV.name]
    new_solver = after["artifacts"][SOLVER_READY_CSV.name]

    solver_refreshed = (
        new_solver["mtime_ns"] > old_solver["mtime_ns"]
        and new_solver["mtime_ns"] >= run_start_ns
    )

    if not solver_refreshed:
        raise RuntimeError(
            "Solver-ready CSV was not rebuilt."
        )

    return {
        "slates": slate_results,
        "solver_ready_csv": {
            "refreshed": solver_refreshed,
            "hash_changed": (
                new_solver["sha256"]
                != old_solver["sha256"]
            ),
            "before": old_solver,
            "after": new_solver,
        },
    }


def write_audit(
    payload: dict,
    prefix: str,
) -> Path:
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = AUDIT_DIR / f"{prefix}_{stamp}.json"
    temp = path.with_suffix(".json.tmp")

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temp, path)

    return path


def validation_only() -> int:
    audit: dict = {
        "mode": "validate-only",
        "started_utc": utc_now(),
        "version": "EMERGENCY_REFRESH_V1",
        "windows_ip": WINDOWS_IP,
        "windows_mac": WINDOWS_MAC,
        "task_name": TASK_NAME,
        "expected_task_script": EXPECTED_TASK_SCRIPT,
    }

    try:
        print("=== LOCAL PRODUCTION STATE ===")

        local_state = validate_local_state()
        audit["local_state"] = local_state

        for name, state in local_state["slates"].items():
            print(
                f"PASS | SLATE | {name} | "
                f"{state['size']} bytes | "
                f"{state['sha256']}"
            )

        print(
            "PASS | solver-ready CSV | "
            + local_state["artifacts"][
                SOLVER_READY_CSV.name
            ]["sha256"]
        )

        print()
        print("=== WINDOWS TCP/22 ===")

        if not tcp_open(WINDOWS_IP, WINDOWS_PORT):
            raise RuntimeError(
                "Windows TCP/22 is not reachable in "
                "validate-only mode. No WOL packet sent."
            )

        print("PASS | Windows TCP/22 reachable.")

        print()
        print("=== DEDICATED SSH IDENTITY ===")

        identity = validate_remote_identity()
        audit["remote_identity"] = identity

        print(identity["stdout"])

        print()
        print("=== PRODUCTION TASK DEFINITION ===")

        task_text = query_task_raw()
        validate_task_definition(task_text)

        for field in (
            "TaskName",
            "Status",
            "Last Run Time",
            "Last Result",
            "Task To Run",
            "Start In",
            "Scheduled Task State",
            "Run As User",
        ):
            try:
                print(
                    f"{field}: "
                    f"{first_field(task_text, field)}"
                )
            except RuntimeError:
                pass

        audit["task_validation"] = {
            "validated": True,
            "expected_script_present": True,
        }

        audit["finished_utc"] = utc_now()
        audit["result"] = "PASS"

        audit_path = write_audit(
            audit,
            "validate_only",
        )

        print()
        print("======================================================================")
        print("PASS | EMERGENCY REFRESH V1 VALIDATION-ONLY MODE")
        print("NO WOL PACKETS SENT")
        print("FANDUEL TASK NOT STARTED")
        print("V4.3 UNCHANGED")
        print(f"AUDIT | {audit_path}")
        print("======================================================================")

        return 0

    except Exception as exc:
        audit["finished_utc"] = utc_now()
        audit["result"] = "FAIL"
        audit["error"] = str(exc)

        audit_path = write_audit(
            audit,
            "validate_only_fail",
        )

        print()
        print("======================================================================")
        print(f"FAIL | {exc}")
        print(f"AUDIT | {audit_path}")
        print("======================================================================")

        return 1


def execute_refresh() -> int:
    audit: dict = {
        "mode": "execute",
        "started_utc": utc_now(),
        "version": "EMERGENCY_REFRESH_V1",
        "windows_ip": WINDOWS_IP,
        "windows_mac": WINDOWS_MAC,
        "task_name": TASK_NAME,
        "expected_task_script": EXPECTED_TASK_SCRIPT,
    }

    try:
        print("=== CAPTURE BEFORE STATE ===")

        before = validate_local_state()
        audit["before"] = before

        print(
            "BEFORE | solver-ready CSV | "
            + before["artifacts"][
                SOLVER_READY_CSV.name
            ]["sha256"]
        )

        print()
        print("=== WINDOWS AVAILABILITY ===")

        if tcp_open(WINDOWS_IP, WINDOWS_PORT):
            print(
                "PASS | Windows already reachable. "
                "WOL not required."
            )
            audit["wol_sent"] = False

        else:
            print(
                "INFO | Windows TCP/22 unavailable. "
                "Sending WOL."
            )

            send_wol()
            audit["wol_sent"] = True

            wait_for_tcp()

        print()
        print("=== DEDICATED SSH READINESS ===")

        identity = wait_for_remote_identity()
        audit["remote_identity"] = identity

        print(identity["stdout"])

        print()
        print("=== PRODUCTION TASK PRE-CHECK ===")

        task_text = query_task_raw()
        validate_task_definition(task_text)

        status_before = task_status(task_text)
        last_run_before = task_last_run(task_text)
        last_result_before = task_last_result(task_text)

        print(f"Status: {status_before}")
        print(f"Last Run Time: {last_run_before}")
        print(f"Last Result: {last_result_before}")

        if status_before.lower() == "running":
            raise RuntimeError(
                "Production FanDuel task is already running. "
                "Refusing duplicate demand-start."
            )

        if status_before.lower() != "ready":
            raise RuntimeError(
                f"Production task is not Ready: "
                f"{status_before}"
            )

        audit["task_before"] = {
            "status": status_before,
            "last_run": last_run_before,
            "last_result": last_result_before,
        }

        print()
        print("=== DEMAND START ===")

        run_start_ns = time.time_ns()
        run_start_utc = utc_now()

        audit["run_start_ns"] = run_start_ns
        audit["run_start_utc"] = run_start_utc

        start_result = demand_start_task()

        audit["demand_start"] = start_result

        if start_result["state"] == "CONFIRMED":
            print(start_result["output"])
            print(
                "PASS | Task Scheduler accepted "
                "demand-start."
            )

        elif start_result["state"] == "AMBIGUOUS_TIMEOUT":
            print(
                "WARN | Demand-start SSH command timed out."
            )
            print(
                "INFO | Treating result as ambiguous; "
                "will reconcile task state."
            )
            print(
                "INFO | No second /Run will be issued."
            )
            print(start_result["output"])

        else:
            raise RuntimeError(
                "Unknown demand-start result state: "
                + str(start_result["state"])
            )

        print()
        print("=== PROVE NEW RUN STARTED ===")

        started = wait_for_new_run(
            last_run_before
        )

        audit["task_started"] = started

        if start_result["state"] == "AMBIGUOUS_TIMEOUT":
            print(
                "PASS | Ambiguous demand-start reconciled: "
                "new scheduled-task run observed."
            )
        else:
            print(
                "PASS | New scheduled-task run observed."
            )

        print()
        print("=== WAIT FOR TASK COMPLETION ===")

        completed = wait_for_task_completion(
            last_run_before
        )

        audit["task_completed"] = completed

        print(
            "PASS | Scheduled task completed "
            "with Last Result 0."
        )

        print()
        print("=== CAPTURE AFTER STATE ===")

        after = validate_local_state()
        audit["after"] = after

        print(
            "AFTER | solver-ready CSV | "
            + after["artifacts"][
                SOLVER_READY_CSV.name
            ]["sha256"]
        )

        print()
        print("=== VERIFY PRODUCTION REFRESH ===")

        verification = verify_refresh(
            before,
            after,
            run_start_ns,
        )

        audit["verification"] = verification

        for name in sorted(EXPECTED_SLATES):
            result = verification["slates"][name]

            print(
                f"PASS | {name} refreshed | "
                f"hash_changed="
                f"{result['hash_changed']}"
            )

        solver_result = verification[
            "solver_ready_csv"
        ]

        print(
            "PASS | solver-ready CSV rebuilt | "
            f"hash_changed="
            f"{solver_result['hash_changed']}"
        )

        audit["finished_utc"] = utc_now()
        audit["result"] = "PASS"

        audit_path = write_audit(
            audit,
            "execute_pass",
        )

        print()
        print("======================================================================")
        print("PASS | EMERGENCY REFRESH V1 EXECUTION")
        print("PASS | WINDOWS REACHABILITY")
        print("PASS | DEDICATED SSH AUTH")
        print("PASS | FROZEN V4.3 TASK DEFINITION")
        print("PASS | DEMAND-START")
        print("PASS | TASK LAST RESULT 0")
        print("PASS | ALL SIX SLATES REFRESHED")
        print("PASS | SOLVER-READY CSV REBUILT")
        print("V4.3 UNCHANGED")
        print(f"AUDIT | {audit_path}")
        print("======================================================================")

        return 0

    except Exception as exc:
        audit["finished_utc"] = utc_now()
        audit["result"] = "FAIL"
        audit["error"] = str(exc)

        audit_path = write_audit(
            audit,
            "execute_fail",
        )

        print()
        print("======================================================================")
        print(f"FAIL | {exc}")
        print("FAIL CLOSED")
        print(f"AUDIT | {audit_path}")
        print("======================================================================")

        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WFS Emergency FanDuel Refresh V1 controller."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate local state, Windows SSH identity, "
            "and production task definition only."
        ),
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the emergency refresh. Wakes Windows if "
            "needed, triggers the existing frozen V4.3 task, "
            "and verifies downstream refresh."
        ),
    )

    args = parser.parse_args()

    if args.validate_only:
        return validation_only()

    if args.execute:
        return execute_refresh()

    return 2


if __name__ == "__main__":
    sys.exit(main())
