#!/home/mwynn/nfl_data_engine/venv/bin/python

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# WFS FORECAST CENTER
# MANUAL LIVE FORECAST REFRESH V2
#
# Thin fail-closed orchestrator only.
#
# Sequence:
#   1. Verify frozen component hashes
#   2. Audit current DB / ledger state
#   3. Back up mutable state
#   4. Run authoritative updater
#   5. Run frozen Live CORE builder
#   6. Validate Live CORE audit
#   7. Run frozen V3 inference wrapper
#   8. Validate prediction artifact
#   9. Detect duplicate prediction SHA in ledger
#  10. Append prospective snapshot only if source SHA is new
#  11. Final integrity audit
#
# This file contains NO forecast/model/optimizer/UI logic.
# ============================================================================


ROOT = Path("/home/mwynn/nfl_data_engine")

PYTHON = ROOT / "venv" / "bin" / "python"

UPDATER = ROOT / "updater.py"

BUILDER = (
    ROOT
    / "scripts"
    / "build_forecast_live_core_v1.py"
)

V3 = (
    ROOT
    / "backups"
    / "forecast_live_core_v3_frozen"
    / "run_forecast_live_core_v3.py"
)

WRITER = (
    ROOT
    / "backups"
    / "forecast_snapshot_writer_v2_frozen"
    / "capture_forecast_snapshot_v2.py"
)

NFL_DB = ROOT / "data" / "nfl.db"

LEDGER_DB = (
    ROOT
    / "data"
    / "forecast_ledger.db"
)

CORE = (
    ROOT
    / "processed"
    / "forecast_live_core_v1.csv"
)

CORE_AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_audit.json"
)

PRED = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

PRED_AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions_audit.json"
)

BACKUP_ROOT = (
    ROOT
    / "backups"
    / "manual_forecast_refresh_v2"
)


# ============================================================================
# FROZEN HASH CONTRACT
# ============================================================================

EXPECTED_BUILDER_SHA = (
    "6ab248295dcab64108397c7a074d5210"
    "e3f252827f1e23adf5a069607455df02"
)

EXPECTED_V3_SHA = (
    "bf9a7e8b4645e3def7b9c67d2af7badc"
    "eb67fffbddf0cd0da755e6fb07c94853"
)

EXPECTED_WRITER_SHA = (
    "fdb18e85e204fa29d2bff9d6ab9776eb"
    "45cdf860f5018e2f783a20ca9488fb77"
)


# ============================================================================
# HELPERS
# ============================================================================


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def fail(message: str) -> None:
    raise SystemExit(
        f"FAIL | {message}"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def require_file(
    label: str,
    path: Path,
) -> None:

    if not path.is_file():
        fail(
            f"missing {label}: {path}"
        )

    print(
        f"PASS | {label:<24} | {path}"
    )


def verify_hash(
    label: str,
    path: Path,
    expected: str,
) -> None:

    actual = sha256(path)

    print(
        f"{label:<24} | {actual}"
    )

    if actual != expected:
        fail(
            f"{label} hash mismatch"
        )


def load_json(path: Path) -> dict:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        fail(
            f"cannot read JSON {path}: {exc}"
        )


def connect_ro(path: Path):
    return sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )


def database_gate(
    label: str,
    path: Path,
) -> tuple[int | None, int | None]:

    conn = connect_ro(path)

    try:
        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        snapshots = None
        predictions = None

        if label == "forecast_ledger.db":
            snapshots = conn.execute(
                """
                SELECT COUNT(*)
                FROM forecast_snapshots
                """
            ).fetchone()[0]

            predictions = conn.execute(
                """
                SELECT COUNT(*)
                FROM forecast_predictions
                """
            ).fetchone()[0]

    finally:
        conn.close()

    print(
        f"{label:<24} | "
        f"integrity={integrity} "
        f"| fk_rows={len(fk)}"
    )

    if integrity != "ok":
        fail(
            f"{label} integrity"
        )

    if fk:
        fail(
            f"{label} foreign-key integrity"
        )

    return (
        snapshots,
        predictions,
    )


def compile_script(
    label: str,
    path: Path,
) -> None:

    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(path),
        ],
        cwd=ROOT,
    )

    if result.returncode != 0:
        fail(
            f"{label} compile"
        )

    print(
        f"PASS | {label} compile"
    )


def run_stage(
    label: str,
    path: Path,
) -> None:

    section(label)

    result = subprocess.run(
        [
            str(PYTHON),
            str(path),
        ],
        cwd=ROOT,
    )

    print()
    print(
        f"{label} exit status | "
        f"{result.returncode}"
    )

    if result.returncode != 0:
        fail(
            f"{label} execution"
        )

    print(
        f"PASS | {label}"
    )


def copy_if_exists(
    path: Path,
    destination: Path,
) -> None:

    if not path.is_file():
        print(
            f"SKIP | not present | {path}"
        )
        return

    target = (
        destination
        / path.name
    )

    shutil.copy2(
        path,
        target,
    )

    source_sha = sha256(path)
    target_sha = sha256(target)

    if source_sha != target_sha:
        fail(
            f"backup verification failed: {path}"
        )

    print(
        f"PASS | backup | {path.name}"
    )


def ledger_source_count(
    prediction_sha: str,
) -> int:

    conn = connect_ro(
        LEDGER_DB
    )

    try:
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_snapshots
            WHERE source_prediction_sha256 = ?
            """,
            (prediction_sha,),
        ).fetchone()[0]

    finally:
        conn.close()

    return int(count)


def ledger_counts() -> tuple[int, int]:

    conn = connect_ro(
        LEDGER_DB
    )

    try:
        snapshots = conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_snapshots
            """
        ).fetchone()[0]

        predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            """
        ).fetchone()[0]

    finally:
        conn.close()

    return (
        int(snapshots),
        int(predictions),
    )


# ============================================================================
# AUDIT CONTRACTS
# ============================================================================


def validate_core_audit() -> str:

    require_file(
        "Live CORE CSV",
        CORE,
    )

    require_file(
        "Live CORE audit",
        CORE_AUDIT,
    )

    payload = load_json(
        CORE_AUDIT
    )

    core_sha = sha256(
        CORE
    )

    if payload.get(
        "status"
    ) != "PASS":
        fail(
            "Live CORE audit status"
        )

    if payload.get(
        "architecture"
    ) != "live_core_v1_frozen_exp004_semantics":
        fail(
            "Live CORE architecture"
        )

    output = payload.get(
        "output",
        {},
    )

    audit_core_sha = output.get(
        "csv_sha256"
    )

    if audit_core_sha != core_sha:
        fail(
            "Live CORE audit SHA binding"
        )

    game_count = output.get(
        "game_count"
    )

    team_row_count = output.get(
        "team_row_count"
    )

    expected_team_row_count = output.get(
        "expected_team_row_count"
    )

    if (
        not isinstance(game_count, int)
        or isinstance(game_count, bool)
        or game_count <= 0
    ):
        fail(
            "Live CORE audit dynamic game_count"
        )

    expected_rows = 2 * game_count

    if team_row_count != expected_rows:
        fail(
            "Live CORE audit dynamic team_row_count"
        )

    if expected_team_row_count != expected_rows:
        fail(
            "Live CORE audit dynamic expected_team_row_count"
        )

    if team_row_count != expected_team_row_count:
        fail(
            "Live CORE audit population agreement"
        )

    expected_values = {
        "duplicate_game_team_keys": 0,
        "infinite_core_values": 0,
    }

    for key, expected in expected_values.items():

        if output.get(
            key
        ) != expected:
            fail(
                f"Live CORE audit {key}"
            )

    contract = payload.get(
        "frozen_contract",
        {}
    )

    if contract.get(
        "core_feature_count"
    ) != 76:
        fail(
            "Live CORE 76-feature contract"
        )

    guards = payload.get(
        "guards",
        {}
    )

    required_false = [
        "database_write",
        "impact_enabled",
        "replacement_enabled",
        "model_executed",
        "optimizer_modified",
        "temp_wind_zero_filled",
        "ui_modified",
    ]

    for key in required_false:

        if guards.get(
            key
        ) is not False:
            fail(
                f"Live CORE guard {key}"
            )

    print(
        f"PASS | Live CORE audit | {core_sha}"
    )

    return core_sha


def validate_prediction_audit(
    expected_core_sha: str,
) -> tuple[str, str]:

    require_file(
        "prediction CSV",
        PRED,
    )

    require_file(
        "prediction audit",
        PRED_AUDIT,
    )

    payload = load_json(
        PRED_AUDIT
    )

    pred_sha = sha256(
        PRED
    )

    pred_audit_sha = sha256(
        PRED_AUDIT
    )

    if payload.get(
        "status"
    ) != "PASS":
        fail(
            "prediction audit status"
        )

    if payload.get(
        "model"
    ) != "WFS_CORE_RESIDUAL":
        fail(
            "prediction model contract"
        )

    if payload.get(
        "forecast_variant"
    ) != "LIVE_CORE_ONLY_V1":
        fail(
            "prediction variant contract"
        )

    lineage = payload.get(
        "frozen_lineage",
        {}
    )

    if lineage.get(
        "live_core_sha256"
    ) != expected_core_sha:
        fail(
            "prediction audit Live CORE provenance"
        )

    proof = payload.get(
        "reconstruction_proof",
        {}
    )

    if proof.get(
        "status"
    ) != "PASS":
        fail(
            "historical reconstruction proof"
        )

    if proof.get(
        "games"
    ) != 285:
        fail(
            "historical reconstruction game count"
        )

    if proof.get(
        "winner_mismatches"
    ) != 0:
        fail(
            "historical winner reconstruction"
        )

    max_diff = proof.get(
        "global_max_abs_difference"
    )

    tolerance = proof.get(
        "numeric_tolerance"
    )

    if (
        max_diff is None
        or tolerance is None
        or max_diff > tolerance
    ):
        fail(
            "historical numeric reconstruction"
        )

    print(
        f"PASS | prediction audit | {pred_sha}"
    )

    return (
        pred_sha,
        pred_audit_sha,
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    section(
        "WFS FORECAST CENTER — "
        "MANUAL LIVE FORECAST REFRESH V2"
    )

    started = datetime.now(
        timezone.utc
    )

    print(
        "Started UTC | "
        f"{started.isoformat()}"
    )

    print(
        "MODE        | "
        "FAIL-CLOSED THIN ORCHESTRATOR"
    )

    # ------------------------------------------------------------------
    # Required components
    # ------------------------------------------------------------------

    section(
        "STAGE 0 — COMPONENT INTEGRITY"
    )

    required = [
        (
            "Python",
            PYTHON,
        ),
        (
            "authoritative updater",
            UPDATER,
        ),
        (
            "frozen CORE builder",
            BUILDER,
        ),
        (
            "frozen V3 inference",
            V3,
        ),
        (
            "frozen snapshot writer",
            WRITER,
        ),
        (
            "nfl.db",
            NFL_DB,
        ),
        (
            "forecast ledger",
            LEDGER_DB,
        ),
    ]

    for label, path in required:
        require_file(
            label,
            path,
        )

    print()
    print(
        "=== FROZEN HASH GATES ==="
    )

    verify_hash(
        "CORE builder",
        BUILDER,
        EXPECTED_BUILDER_SHA,
    )

    verify_hash(
        "V3 inference",
        V3,
        EXPECTED_V3_SHA,
    )

    verify_hash(
        "snapshot writer",
        WRITER,
        EXPECTED_WRITER_SHA,
    )

    print()
    print(
        "=== COMPILE GATES ==="
    )

    compile_script(
        "authoritative updater",
        UPDATER,
    )

    compile_script(
        "CORE builder",
        BUILDER,
    )

    compile_script(
        "V3 inference",
        V3,
    )

    compile_script(
        "snapshot writer",
        WRITER,
    )

    print()
    print(
        "=== PRE-REFRESH DATABASE GATE ==="
    )

    database_gate(
        "nfl.db",
        NFL_DB,
    )

    before_snapshots, before_rows = (
        database_gate(
            "forecast_ledger.db",
            LEDGER_DB,
        )
    )

    # ------------------------------------------------------------------
    # Backup mutable state
    # ------------------------------------------------------------------

    section(
        "STAGE 1 — PRE-REFRESH BACKUP"
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    backup_dir = (
        BACKUP_ROOT
        / stamp
    )

    if backup_dir.exists():
        fail(
            f"backup directory exists: {backup_dir}"
        )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in [
        NFL_DB,
        LEDGER_DB,
        CORE,
        CORE_AUDIT,
        PRED,
        PRED_AUDIT,
    ]:
        copy_if_exists(
            path,
            backup_dir,
        )

    print(
        f"BACKUP | {backup_dir}"
    )

    # ------------------------------------------------------------------
    # Authoritative data refresh
    # ------------------------------------------------------------------

    run_stage(
        "STAGE 2 — AUTHORITATIVE NFL DATA UPDATE",
        UPDATER,
    )

    database_gate(
        "nfl.db",
        NFL_DB,
    )

    # ------------------------------------------------------------------
    # Live CORE rebuild
    # ------------------------------------------------------------------

    run_stage(
        "STAGE 3 — LIVE CORE FEATURE BUILD",
        BUILDER,
    )

    core_sha = (
        validate_core_audit()
    )

    # ------------------------------------------------------------------
    # V3 inference
    # ------------------------------------------------------------------

    run_stage(
        "STAGE 4 — LIVE CORE V3 INFERENCE",
        V3,
    )

    (
        pred_sha,
        pred_audit_sha,
    ) = validate_prediction_audit(
        core_sha
    )

    # ------------------------------------------------------------------
    # Ledger duplicate gate
    # ------------------------------------------------------------------

    section(
        "STAGE 5 — PROSPECTIVE LEDGER DECISION"
    )

    existing = ledger_source_count(
        pred_sha
    )

    print(
        f"Prediction SHA | {pred_sha}"
    )

    print(
        f"Existing ledger snapshots "
        f"for SHA | {existing}"
    )

    if existing > 1:
        fail(
            "duplicate source SHA already exists "
            "more than once in ledger"
        )

    if existing == 1:

        print(
            "PASS | prediction source already "
            "captured"
        )

        print(
            "SKIP | snapshot writer not invoked"
        )

        appended = False

    else:

        print(
            "NEW | prediction source has not "
            "been captured"
        )

        before_write_snapshots, (
            before_write_rows
        ) = ledger_counts()

        run_stage(
            "STAGE 6 — PROSPECTIVE SNAPSHOT CAPTURE",
            WRITER,
        )

        after_write_snapshots, (
            after_write_rows
        ) = ledger_counts()

        source_count = (
            ledger_source_count(
                pred_sha
            )
        )

        if source_count != 1:
            fail(
                "new prediction SHA was not "
                "captured exactly once"
            )

        if (
            after_write_snapshots
            != before_write_snapshots + 1
        ):
            fail(
                "ledger snapshot count did not "
                "increase by exactly one"
            )

        if (
            after_write_rows
            <= before_write_rows
        ):
            fail(
                "ledger prediction row count "
                "did not increase"
            )

        print(
            "PASS | new prospective snapshot "
            "captured exactly once"
        )

        appended = True

    # ------------------------------------------------------------------
    # Final audit
    # ------------------------------------------------------------------

    section(
        "STAGE 7 — FINAL INTEGRITY GATE"
    )

    database_gate(
        "nfl.db",
        NFL_DB,
    )

    final_snapshots, final_rows = (
        database_gate(
            "forecast_ledger.db",
            LEDGER_DB,
        )
    )

    final_source_count = (
        ledger_source_count(
            pred_sha
        )
    )

    if final_source_count != 1:
        fail(
            "final ledger source SHA count"
        )

    # Revalidate active artifacts after all writes.
    final_core_sha = (
        validate_core_audit()
    )

    if final_core_sha != core_sha:
        fail(
            "Live CORE changed after inference"
        )

    (
        final_pred_sha,
        final_pred_audit_sha,
    ) = validate_prediction_audit(
        final_core_sha
    )

    if final_pred_sha != pred_sha:
        fail(
            "prediction artifact changed "
            "after ledger capture"
        )

    if (
        final_pred_audit_sha
        != pred_audit_sha
    ):
        fail(
            "prediction audit changed "
            "after ledger capture"
        )

    ended = datetime.now(
        timezone.utc
    )

    duration = (
        ended
        - started
    ).total_seconds()

    section(
        "MANUAL LIVE FORECAST REFRESH V2 — FINAL GATE"
    )

    print(
        "PASS | AUTHORITATIVE NFL DATA UPDATE"
    )

    print(
        "PASS | LIVE CORE FEATURE BUILD"
    )

    print(
        "PASS | FROZEN V3 INFERENCE"
    )

    if appended:
        print(
            "PASS | NEW PROSPECTIVE SNAPSHOT APPENDED"
        )
    else:
        print(
            "PASS | EXISTING PROSPECTIVE SNAPSHOT PRESERVED"
        )

    print(
        "PASS | SQLITE INTEGRITY"
    )

    print(
        "PASS | FOREIGN KEY INTEGRITY"
    )

    print(
        "PASS | FROZEN COMPONENT HASHES"
    )

    print(
        "PASS | OPTIMIZER / UI UNTOUCHED"
    )

    print(
        "PASS | SITE RESTART NOT REQUIRED"
    )

    print()
    print(
        f"Live CORE SHA      | {core_sha}"
    )

    print(
        f"Prediction SHA     | {pred_sha}"
    )

    print(
        f"Prediction audit   | {pred_audit_sha}"
    )

    print(
        f"Ledger snapshots   | "
        f"{before_snapshots} -> "
        f"{final_snapshots}"
    )

    print(
        f"Ledger rows        | "
        f"{before_rows} -> "
        f"{final_rows}"
    )

    print(
        f"Backup             | {backup_dir}"
    )

    print(
        f"Finished UTC       | "
        f"{ended.isoformat()}"
    )

    print(
        f"Runtime seconds    | "
        f"{duration:.1f}"
    )

    print()
    print(
        "PASS | MANUAL LIVE FORECAST "
        "REFRESH V2 COMPLETE"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
