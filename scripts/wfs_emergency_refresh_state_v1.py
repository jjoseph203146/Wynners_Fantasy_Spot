#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/home/mwynn/nfl_data_engine")
STATE_PATH = PROJECT / "data/fanduel/emergency_refresh_state.json"
SOLVER_READY = PROJECT / "data/csv/fanduel_solver_ready_pool.csv"

VALID_STATES = {
    "FRESH",
    "REFRESH_REQUIRED",
    "REFRESH_RUNNING",
    "REFRESH_FAILED",
}

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def current_solver_sha() -> str:
    if not SOLVER_READY.is_file():
        raise RuntimeError(
            f"Solver-ready pool is missing: {SOLVER_READY}"
        )

    return sha256_file(SOLVER_READY)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def validate_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("State payload is not a JSON object.")

    required = {
        "schema_version",
        "state",
        "updated_utc",
        "solver_ready_sha256",
        "reason",
    }

    missing = sorted(required - set(payload))

    if missing:
        raise RuntimeError(
            "State payload missing required fields: "
            + ", ".join(missing)
        )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported state schema_version: "
            + repr(payload["schema_version"])
        )

    state = payload["state"]

    if state not in VALID_STATES:
        raise RuntimeError(
            f"Invalid refresh state: {state!r}"
        )

    solver_sha = payload["solver_ready_sha256"]

    if not isinstance(solver_sha, str) or len(solver_sha) != 64:
        raise RuntimeError(
            "solver_ready_sha256 must be a 64-character SHA256."
        )

    if not isinstance(payload["updated_utc"], str):
        raise RuntimeError("updated_utc must be a string.")

    if not isinstance(payload["reason"], str):
        raise RuntimeError("reason must be a string.")

    return payload


def read_state() -> dict:
    if not STATE_PATH.is_file():
        raise RuntimeError(
            f"Refresh state file is missing: {STATE_PATH}"
        )

    try:
        payload = json.loads(
            STATE_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Refresh state file is unreadable: {exc}"
        ) from exc

    return validate_payload(payload)


def build_payload(state: str, reason: str) -> dict:
    if state not in VALID_STATES:
        raise RuntimeError(
            f"Invalid requested state: {state!r}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "updated_utc": utc_now(),
        "solver_ready_sha256": current_solver_sha(),
        "reason": reason,
    }


def write_state(state: str, reason: str) -> dict:
    payload = build_payload(state, reason)
    atomic_write_json(STATE_PATH, payload)

    verified = read_state()

    if verified != payload:
        raise RuntimeError(
            "Post-write verification did not match written payload."
        )

    return verified


def status() -> int:
    payload = read_state()
    current_sha = current_solver_sha()

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"CURRENT_SOLVER_SHA256={current_sha}")

    if payload["state"] != "FRESH":
        print(
            "DELIVERY_ALLOWED=NO | "
            f"state={payload['state']}"
        )
        return 2

    if payload["solver_ready_sha256"] != current_sha:
        print(
            "DELIVERY_ALLOWED=NO | "
            "FRESH state is anchored to a different solver-ready pool."
        )
        return 3

    print(
        "DELIVERY_ALLOWED=YES | "
        "state=FRESH and solver-ready SHA matches."
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WFS emergency FanDuel refresh-state authority."
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser("status")

    set_parser = sub.add_parser("set")
    set_parser.add_argument(
        "state",
        choices=sorted(VALID_STATES),
    )
    set_parser.add_argument(
        "--reason",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "status":
        return status()

    payload = write_state(
        args.state,
        args.reason,
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("PASS | State write verified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
