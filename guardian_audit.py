#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from guardian_classification import classify_failures


AUDIT_CONTRACT = "WFS_GUARDIAN_AUDIT_V1"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def append_guardian_audit(
    root: Path,
    schedule: dict | None,
    results: list[dict],
    verdict: str,
    lkg_written: bool,
    lkg_detail: str,
) -> tuple[bool, str]:
    """
    Append one immutable observation record per Guardian run.

    Guardian-owned state only.

    Does NOT:
      - modify production artifacts
      - modify production manifests
      - modify operational databases
      - repair anything
      - rollback anything
      - restart anything
    """

    guardian_dir = root / "data" / "guardian"
    guardian_dir.mkdir(parents=True, exist_ok=True)

    target = guardian_dir / "guardian_audit.jsonl"

    classification = classify_failures(results)

    failed = classification["failures"]

    checks = [
        {
            "check": str(row.get("check")),
            "status": str(row.get("status")),
            "detail": str(row.get("detail", "")),
        }
        for row in results
    ]

    p = root / "data" / "parquet"

    artifact_paths = {
        "stat_publication":
            p / "current_unified_stat_forecasts.parquet",

        "fanduel_publication":
            p / "current_unified_fanduel_expectation.parquet",

        "starter_verification":
            p / "current_starter_verification.parquet",

        "stage24_gav2":
            p / "nfl_current_fanduel_expectation_gav2.parquet",
    }

    artifacts = {}

    for name, path in artifact_paths.items():
        artifacts[name] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size": (
                path.stat().st_size
                if path.is_file()
                else None
            ),
            "sha256": _sha256(path),
        }

    if schedule is None:
        schedule_state = None
    else:
        schedule_state = {
            "season": int(schedule["season"]),
            "week": int(schedule["week"]),
            "unfinished_games": int(
                schedule["unfinished_games"]
            ),
            "unfinished_teams": int(
                schedule["unfinished_teams"]
            ),
            "completed_same_week": int(
                schedule["completed_same_week"]
            ),
            "next_staged_week": (
                int(schedule["next_staged_week"])
                if schedule.get("next_staged_week") is not None
                else None
            ),
            "unfinished_game_ids": sorted(
                str(v)
                for v in schedule["unfinished_game_ids"]
            ),
        }

    record = {
        "contract": AUDIT_CONTRACT,
        "guardian_version": "V2.5",
        "run_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "mode": "SHADOW",
        "production_read_only": True,
        "production_influence": False,
        "auto_repair": False,
        "auto_rollback": False,
        "auto_restart": False,

        "verdict": verdict,
        "failed_check_count": len(failed),
        "failed_checks": failed,

        "failure_classification_contract":
            classification["contract"],
        "failure_categories":
            classification["categories"],

        "schedule": schedule_state,

        "lkg_recorded": bool(lkg_written),
        "lkg_detail": str(lkg_detail),

        "production_action": "NONE",

        "artifacts": artifacts,
        "checks": checks,
    }

    encoded = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    # O_APPEND gives us append semantics. fsync ensures the record
    # reaches durable storage before Guardian reports success.
    fd = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )

    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)

    return (
        True,
        (
            f"path={target} "
            f"verdict={verdict} "
            f"failed_checks={len(failed)}"
        ),
    )
