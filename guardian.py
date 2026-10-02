#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from guardian_cross_stage import run_cross_stage
from guardian_lifecycle import run_lifecycle_checks, write_lkg_if_guardian_pass
from guardian_audit import append_guardian_audit
from guardian_classification import classify_failures
from guardian_schedule import derive_schedule_authority as _derive_schedule_authority

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARQUET = DATA / "parquet"

NFL_DB = DATA / "nfl.db"
FORECAST_LEDGER = DATA / "forecast_ledger.db"
PROJECTION_LEDGER = DATA / "player_projection_ledger.db"
ROLE_DB = DATA / "fanduel_player_role_performance.db"
LIVE_DB = DATA / "wfs_live.db"

STAT = PARQUET / "current_unified_stat_forecasts.parquet"
STAT_MANIFEST = PARQUET / "current_unified_stat_forecasts_manifest.json"

FD = PARQUET / "current_unified_fanduel_expectation.parquet"
FD_MANIFEST = PARQUET / "current_unified_fanduel_expectation_manifest.json"

STARTER = PARQUET / "current_starter_verification.parquet"
STARTER_MANIFEST = PARQUET / "current_starter_verification_manifest.json"

EXPECTED_NFL_TABLES = {
    "games",
    "player_identity",
    "weekly_rosters",
    "injuries",
    "injury_consensus_current",
    "fanduel_slate_pool",
    "fanduel_solver_ready_pool",
}

results = []


def record(name: str, status: str, detail: str) -> None:
    results.append(
        {
            "check": name,
            "status": status,
            "detail": detail,
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def db_tables(path: Path) -> set[str]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    try:
        return {
            row[0]
            for row in con.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                """
            )
        }
    finally:
        con.close()


def integrity(path: Path) -> str:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    try:
        row = con.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "NO_RESULT"
    finally:
        con.close()


def check_db(
    label: str,
    path: Path,
    required_tables: set[str] | None = None,
) -> None:
    if not path.is_file():
        record(label, "FAIL", f"missing: {path}")
        return

    if path.stat().st_size == 0:
        record(label, "FAIL", f"zero-byte database: {path}")
        return

    try:
        value = integrity(path)

        if value.lower() != "ok":
            record(label, "FAIL", f"integrity_check={value}")
            return

        tables = db_tables(path)

        if required_tables:
            missing = sorted(required_tables - tables)

            if missing:
                record(
                    label,
                    "FAIL",
                    "missing tables: " + ", ".join(missing),
                )
                return

        record(
            label,
            "PASS",
            f"integrity=ok size={path.stat().st_size:,}",
        )

    except Exception as exc:
        record(
            label,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def check_artifact(
    label: str,
    artifact: Path,
    manifest: Path,
    expected_contract: str,
) -> None:
    if not artifact.is_file():
        record(label, "FAIL", f"artifact missing: {artifact}")
        return

    if artifact.stat().st_size == 0:
        record(label, "FAIL", f"artifact zero-byte: {artifact}")
        return

    if not manifest.is_file():
        record(label, "FAIL", f"manifest missing: {manifest}")
        return

    try:
        meta = load_manifest(manifest)

    except Exception as exc:
        record(
            label,
            "FAIL",
            f"manifest unreadable: {type(exc).__name__}: {exc}",
        )
        return

    contract = meta.get("contract")

    if contract != expected_contract:
        record(
            label,
            "FAIL",
            f"contract={contract!r}, expected={expected_contract!r}",
        )
        return

    artifact_hash = sha256(artifact)

    declared_hash = (
        meta.get("current_sha256")
        or meta.get("sha256")
        or meta.get("artifact_sha256")
    )

    if declared_hash and declared_hash != artifact_hash:
        record(
            label,
            "FAIL",
            "manifest hash mismatch",
        )
        return

    extra = []

    for key in ("season", "week", "rows", "games", "teams"):
        if key in meta:
            extra.append(f"{key}={meta[key]}")

    record(
        label,
        "PASS",
        " ".join(extra)
        if extra
        else f"sha256={artifact_hash[:12]}...",
    )


def check_non_authority_db(path: Path) -> None:
    if not path.exists():
        record(
            f"NON_AUTHORITY:{path.name}",
            "PASS",
            "absent",
        )
        return

    size = path.stat().st_size

    record(
        f"NON_AUTHORITY:{path.name}",
        "PASS",
        f"ignored by Guardian size={size:,}",
    )


def derive_schedule_authority() -> dict | None:
    """Production wrapper around root-aware schedule authority."""
    return _derive_schedule_authority(
        ROOT,
        record,
    )

def check_manifest_schedule_alignment(
    label: str,
    manifest: Path,
    schedule: dict | None,
) -> None:
    """
    Shadow comparison only.

    If a publication manifest declares season/week/game/team inventory,
    compare those values with Guardian's independently derived schedule.
    """

    if schedule is None:
        record(
            label,
            "FAIL",
            "schedule authority unavailable",
        )
        return

    if not manifest.is_file():
        record(
            label,
            "FAIL",
            f"manifest missing: {manifest}",
        )
        return

    try:
        meta = load_manifest(manifest)
    except Exception as exc:
        record(
            label,
            "FAIL",
            f"manifest unreadable: {type(exc).__name__}: {exc}",
        )
        return

    comparisons = {
        "season": schedule["season"],
        "week": schedule["week"],
        "games": schedule["unfinished_games"],
        "teams": schedule["unfinished_teams"],
    }

    mismatches = []
    compared = []

    for key, expected in comparisons.items():
        if key not in meta:
            continue

        actual = meta.get(key)
        compared.append(f"{key}={actual}")

        try:
            equal = int(actual) == int(expected)
        except (TypeError, ValueError):
            equal = actual == expected

        if not equal:
            mismatches.append(
                f"{key}:{actual}!={expected}"
            )

    if mismatches:
        record(
            label,
            "FAIL",
            " ".join(mismatches),
        )
        return

    if not compared:
        record(
            label,
            "WARN",
            "manifest declares no schedule inventory",
        )
        return

    record(
        label,
        "PASS",
        " ".join(compared),
    )


def main() -> int:
    print("=" * 72)
    print("WFS GUARDIAN V2.5 — SHADOW MODE")
    print("=" * 72)
    print("MODE=READ_ONLY_PRODUCTION")
    print("GUARDIAN_STATE_WRITE=LKG_METADATA_ONLY")
    print("PRODUCTION_INFLUENCE=FALSE")
    print("AUTO_REPAIR=FALSE")
    print("AUTO_ROLLBACK=FALSE")
    print("AUTO_RESTART=FALSE")
    print("RUN_UTC=" + datetime.now(timezone.utc).isoformat())

    check_db(
        "NFL_DATA_AUTHORITY",
        NFL_DB,
        EXPECTED_NFL_TABLES,
    )

    check_db(
        "FORECAST_LEDGER",
        FORECAST_LEDGER,
        {
            "forecast_predictions",
            "forecast_snapshots",
        },
    )

    check_db(
        "PLAYER_PROJECTION_LEDGER",
        PROJECTION_LEDGER,
        {
            "player_projection_predictions",
            "player_projection_snapshots",
        },
    )

    check_db(
        "ROLE_PERFORMANCE_DB",
        ROLE_DB,
        {
            "dim_game",
            "dim_player",
            "fact_player_game_fanduel",
        },
    )

    check_db(
        "NFL_LIVE_DB",
        LIVE_DB,
        {
            "live_events",
            "live_plays",
            "live_ingest_audit",
        },
    )

    schedule = derive_schedule_authority()

    check_artifact(
        "STAT_FORECAST_PUBLICATION",
        STAT,
        STAT_MANIFEST,
        "WFS_STAT_FORECAST_PUBLISH_V1",
    )

    check_artifact(
        "FANDUEL_EXPECTATION_PUBLICATION",
        FD,
        FD_MANIFEST,
        "WFS_FANDUEL_EXPECTATION_PUBLISH_V1",
    )

    check_artifact(
        "STARTER_VERIFICATION_SHADOW",
        STARTER,
        STARTER_MANIFEST,
        "WFS_STARTER_VERIFICATION_CURRENT_V1",
    )

    check_manifest_schedule_alignment(
        "STAT_SCHEDULE_ALIGNMENT",
        STAT_MANIFEST,
        schedule,
    )

    check_manifest_schedule_alignment(
        "FANDUEL_SCHEDULE_ALIGNMENT",
        FD_MANIFEST,
        schedule,
    )

    check_manifest_schedule_alignment(
        "STARTER_SCHEDULE_ALIGNMENT",
        STARTER_MANIFEST,
        schedule,
    )

    results.extend(run_cross_stage(ROOT, schedule))
    results.extend(run_lifecycle_checks(ROOT, schedule))
    check_non_authority_db(ROOT / "nfl.db")
    check_non_authority_db(DATA / "nfl_data.db")

    print()
    print("-" * 72)

    rank = {
        "PASS": 0,
        "WARN": 1,
        "FAIL": 2,
    }

    for result in results:
        print(
            f"{result['check']:<36} "
            f"{result['status']:<5} "
            f"{result['detail']}"
        )

    verdict = max(
        (r["status"] for r in results),
        key=lambda x: rank[x],
        default="FAIL",
    )

    classification = classify_failures(results)

    print("-" * 72)

    categories = classification["categories"]

    print(
        "GUARDIAN_FAILURE_CLASSES="
        + (
            ",".join(categories)
            if categories
            else "NONE"
        )
    )
    print(
        "GUARDIAN_FAILED_CHECKS="
        + str(classification["failure_count"])
    )

    if verdict == "PASS":
        lkg_written, lkg_detail = write_lkg_if_guardian_pass(
            ROOT,
            schedule,
            results,
        )
        print(
            "GUARDIAN_LKG="
            + ("RECORDED" if lkg_written else "NOT_RECORDED")
        )
        print("GUARDIAN_LKG_DETAIL=" + lkg_detail)
    else:
        lkg_written = False
        lkg_detail = "guardian_verdict_not_pass"
        print("GUARDIAN_LKG=NOT_RECORDED")
        print("GUARDIAN_LKG_DETAIL=" + lkg_detail)

    try:
        audit_written, audit_detail = append_guardian_audit(
            ROOT,
            schedule,
            results,
            verdict,
            lkg_written,
            lkg_detail,
        )
    except Exception as exc:
        audit_written = False
        audit_detail = (
            f"{type(exc).__name__}:{exc}"
        )

    print(
        "GUARDIAN_AUDIT="
        + ("APPENDED" if audit_written else "WRITE_FAILED")
    )
    print("GUARDIAN_AUDIT_DETAIL=" + audit_detail)

    print(f"GUARDIAN_VERDICT={verdict}")
    print("PRODUCTION_ACTION=NONE")
    print("SHADOW_MODE=TRUE")
    print("=" * 72)

    # Shadow mode deliberately returns success to the OS.
    # Guardian V2.5 cannot block production.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
