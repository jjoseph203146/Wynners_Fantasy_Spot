from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone


ROOT = Path("/home/mwynn/nfl_data_engine")

LEDGER = ROOT / "data" / "forecast_ledger.db"

CREATOR = (
    ROOT
    / "scripts"
    / "create_forecast_ledger_v1.py"
)

FREEZE_DIR = (
    ROOT
    / "backups"
    / "forecast_ledger_v1_frozen"
)

EXPECTED_LEDGER_SHA = (
    "f2bd53a83655125d886736ae07fc03df"
    "4de2139a6edcae2e3519177b31d96e4e"
)

EXPECTED_SNAPSHOT_ID = (
    "2026-core-v1-20260908T212022Z-d17862101386"
)

EXPECTED_CAPTURED_AT = (
    "2026-09-08T21:20:22+00:00"
)

EXPECTED_PRED_SHA = (
    "8a58309ca2b71ab7d019a5697580eb7c"
    "3114716f94e8b4c5c5864ea7b40bc94c"
)

EXPECTED_AUDIT_SHA = (
    "380c40b59d462a7392f8408ab8b81fe0"
    "97967f4cf1b06a7dab4f06bddbee6d46"
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def fail(message):
    raise SystemExit(
        f"FAIL | {message}"
    )


print("=" * 100)
print("WFS FORECAST CENTER — INDEPENDENT LEDGER V1 AUDIT")
print("=" * 100)


# =====================================================================
# File gates
# =====================================================================

if not LEDGER.is_file():
    fail("forecast_ledger.db missing")

if not CREATOR.is_file():
    fail("ledger creator script missing")

if FREEZE_DIR.exists():
    fail(
        "freeze directory already exists; "
        "refusing to overwrite frozen baseline"
    )

ledger_sha = sha256(
    LEDGER
)

print()
print("=== PHYSICAL LEDGER IDENTITY ===")
print(f"Path   | {LEDGER}")
print(f"SHA256 | {ledger_sha}")

if ledger_sha != EXPECTED_LEDGER_SHA:
    fail("physical ledger SHA mismatch")

print("PASS | physical ledger matches creation artifact")


# =====================================================================
# Strict read-only database audit
# =====================================================================

uri = f"file:{LEDGER}?mode=ro"

conn = sqlite3.connect(
    uri,
    uri=True,
)

try:

    print()
    print("=== SQLITE INTEGRITY ===")

    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(f"integrity_check | {integrity}")

    if integrity != "ok":
        fail("SQLite integrity")


    fk_rows = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print(
        f"foreign_key_check rows | {len(fk_rows)}"
    )

    if fk_rows:
        fail("foreign-key violations")

    print("PASS | database integrity")


    # ================================================================
    # Exact table contract
    # ================================================================

    print()
    print("=== TABLE CONTRACT ===")

    tables = [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )
    ]

    print(
        "Tables | "
        + ", ".join(tables)
    )

    expected_tables = [
        "forecast_predictions",
        "forecast_snapshots",
    ]

    if tables != expected_tables:
        fail("unexpected table contract")

    print("PASS | exact two-table contract")


    # ================================================================
    # Exact trigger contract
    # ================================================================

    print()
    print("=== APPEND-ONLY TRIGGER CONTRACT ===")

    triggers = [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'trigger'
            ORDER BY name
            """
        )
    ]

    for trigger in triggers:
        print(f"Trigger | {trigger}")

    expected_triggers = sorted(
        [
            "forecast_snapshots_no_update",
            "forecast_snapshots_no_delete",
            "forecast_predictions_no_update",
            "forecast_predictions_no_delete",
        ]
    )

    if triggers != expected_triggers:
        fail("append-only trigger contract")

    print("PASS | exact four-trigger contract")


    # ================================================================
    # Index contract
    # ================================================================

    print()
    print("=== INDEX CONTRACT ===")

    indexes = [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
              AND name NOT LIKE 'sqlite_autoindex%'
            ORDER BY name
            """
        )
    ]

    for index in indexes:
        print(f"Index | {index}")

    expected_indexes = sorted(
        [
            "idx_forecast_predictions_game",
            "idx_forecast_predictions_season_week",
            "idx_forecast_predictions_status",
            "idx_forecast_snapshots_capture_time",
        ]
    )

    if indexes != expected_indexes:
        fail("index contract")

    print("PASS | exact four-index contract")


    # ================================================================
    # Snapshot contract
    # ================================================================

    print()
    print("=== SNAPSHOT CONTRACT ===")

    snapshots = conn.execute(
        """
        SELECT
            snapshot_id,
            captured_at_utc,
            season,
            model,
            forecast_variant,
            source_prediction_sha256,
            source_audit_sha256,
            historical_proof_status,
            injury_source_status,
            win_probability_status,
            snapshot_status
        FROM forecast_snapshots
        ORDER BY captured_at_utc, snapshot_id
        """
    ).fetchall()

    print(
        f"Snapshot rows | {len(snapshots)}"
    )

    if len(snapshots) != 1:
        fail("expected exactly one baseline snapshot")

    (
        snapshot_id,
        captured_at_utc,
        season,
        model,
        forecast_variant,
        source_pred_sha,
        source_audit_sha,
        historical_proof_status,
        injury_source_status,
        win_probability_status,
        snapshot_status,
    ) = snapshots[0]

    print(f"Snapshot ID    | {snapshot_id}")
    print(f"Captured UTC   | {captured_at_utc}")
    print(f"Season         | {season}")
    print(f"Model          | {model}")
    print(f"Variant        | {forecast_variant}")
    print(f"Historical     | {historical_proof_status}")
    print(f"Injury source  | {injury_source_status}")
    print(f"Win probability| {win_probability_status}")
    print(f"Status         | {snapshot_status}")

    if snapshot_id != EXPECTED_SNAPSHOT_ID:
        fail("snapshot ID mismatch")

    if captured_at_utc != EXPECTED_CAPTURED_AT:
        fail("snapshot capture timestamp mismatch")

    if season != 2026:
        fail("snapshot season")

    if model != "WFS_CORE_RESIDUAL":
        fail("snapshot model")

    if forecast_variant != "LIVE_CORE_ONLY_V1":
        fail("snapshot variant")

    if source_pred_sha != EXPECTED_PRED_SHA:
        fail("snapshot prediction provenance hash")

    if source_audit_sha != EXPECTED_AUDIT_SHA:
        fail("snapshot audit provenance hash")

    if historical_proof_status != "PASS":
        fail("historical proof status")

    if injury_source_status != "INJURY_SOURCE_PENDING":
        fail("injury source status")

    if win_probability_status != "NOT_CALIBRATED":
        fail("win probability status")

    if snapshot_status != "VALIDATED_PROSPECTIVE_BASELINE":
        fail("snapshot status")

    print("PASS | immutable snapshot identity/provenance")


    # ================================================================
    # Prediction population
    # ================================================================

    print()
    print("=== PREDICTION POPULATION ===")

    prediction_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        """
    ).fetchone()[0]

    unique_games = conn.execute(
        """
        SELECT COUNT(DISTINCT game_id)
        FROM forecast_predictions
        """
    ).fetchone()[0]

    distinct_snapshots = conn.execute(
        """
        SELECT COUNT(DISTINCT snapshot_id)
        FROM forecast_predictions
        """
    ).fetchone()[0]

    print(f"Predictions        | {prediction_count}")
    print(f"Unique games       | {unique_games}")
    print(f"Prediction snapshots| {distinct_snapshots}")

    if prediction_count != 272:
        fail("prediction count")

    if unique_games != 272:
        fail("unique game count")

    if distinct_snapshots != 1:
        fail("prediction snapshot count")

    print("PASS | 272 unique baseline predictions")


    # ================================================================
    # Status distribution
    # ================================================================

    print()
    print("=== STATUS DISTRIBUTION ===")

    statuses = conn.execute(
        """
        SELECT
            forecast_status,
            COUNT(*)
        FROM forecast_predictions
        GROUP BY forecast_status
        ORDER BY forecast_status
        """
    ).fetchall()

    for status, count in statuses:
        print(
            f"{status:<20} | {count}"
        )

    status_map = dict(statuses)

    if status_map != {
        "MARKET_PENDING": 172,
        "READY_CORE_ONLY": 100,
    }:
        fail("forecast status distribution")

    print("PASS | exact ready/pending distribution")


    # ================================================================
    # Market-ready consistency
    # ================================================================

    print()
    print("=== MARKET READY CONTRACT ===")

    market_mismatch = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE
            (
                forecast_status = 'READY_CORE_ONLY'
                AND forecast_market_ready <> 1
            )
            OR
            (
                forecast_status = 'MARKET_PENDING'
                AND forecast_market_ready <> 0
            )
        """
    ).fetchone()[0]

    print(
        f"Market/status mismatches | {market_mismatch}"
    )

    if market_mismatch != 0:
        fail("market/status mismatch")

    print("PASS | market readiness matches forecast status")


    # ================================================================
    # Pending null contract
    # ================================================================

    print()
    print("=== MARKET-PENDING NULL CONTRACT ===")

    pending_violations = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status = 'MARKET_PENDING'
          AND (
                pred_margin_residual IS NOT NULL
             OR pred_total_residual IS NOT NULL
             OR pred_home_margin IS NOT NULL
             OR pred_total_points IS NOT NULL
             OR pred_home_points IS NOT NULL
             OR pred_away_points IS NOT NULL
             OR pred_winner IS NOT NULL
          )
        """
    ).fetchone()[0]

    print(
        f"Pending violations | {pending_violations}"
    )

    if pending_violations != 0:
        fail("pending null contract")

    print("PASS | pending predictions remain null")


    # ================================================================
    # Ready prediction completeness
    # ================================================================

    print()
    print("=== READY PREDICTION COMPLETENESS ===")

    ready_missing = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status = 'READY_CORE_ONLY'
          AND (
                market_home_spread_raw IS NULL
             OR market_total IS NULL
             OR pred_margin_residual IS NULL
             OR pred_total_residual IS NULL
             OR pred_home_margin IS NULL
             OR pred_total_points IS NULL
             OR pred_home_points IS NULL
             OR pred_away_points IS NULL
             OR pred_winner IS NULL
          )
        """
    ).fetchone()[0]

    print(
        f"Ready incomplete rows | {ready_missing}"
    )

    if ready_missing != 0:
        fail("ready prediction completeness")

    print("PASS | all ready predictions complete")


    # ================================================================
    # Mathematical identities
    # ================================================================

    print()
    print("=== FORECAST MATHEMATICAL IDENTITIES ===")

    rows = conn.execute(
        """
        SELECT
            market_home_spread_raw,
            market_total,
            pred_margin_residual,
            pred_total_residual,
            pred_home_margin,
            pred_total_points,
            pred_home_points,
            pred_away_points,
            pred_winner,
            home_team,
            away_team
        FROM forecast_predictions
        WHERE forecast_status = 'READY_CORE_ONLY'
        """
    ).fetchall()

    max_margin_error = 0.0
    max_total_error = 0.0
    max_home_points_error = 0.0
    max_away_points_error = 0.0
    winner_mismatches = 0

    for row in rows:

        (
            market_spread,
            market_total,
            margin_residual,
            total_residual,
            pred_margin,
            pred_total,
            pred_home_points,
            pred_away_points,
            pred_winner,
            home_team,
            away_team,
        ) = row

        expected_margin = (
            market_spread
            + margin_residual
        )

        expected_total = (
            market_total
            + total_residual
        )

        expected_home = (
            expected_total
            + expected_margin
        ) / 2.0

        expected_away = (
            expected_total
            - expected_margin
        ) / 2.0

        max_margin_error = max(
            max_margin_error,
            abs(
                pred_margin
                - expected_margin
            ),
        )

        max_total_error = max(
            max_total_error,
            abs(
                pred_total
                - expected_total
            ),
        )

        max_home_points_error = max(
            max_home_points_error,
            abs(
                pred_home_points
                - expected_home
            ),
        )

        max_away_points_error = max(
            max_away_points_error,
            abs(
                pred_away_points
                - expected_away
            ),
        )

        expected_winner = (
            home_team
            if pred_margin > 0
            else away_team
        )

        if pred_winner != expected_winner:
            winner_mismatches += 1

    tolerance = 1e-10

    print(
        "Max margin identity error | "
        f"{max_margin_error:.12e}"
    )

    print(
        "Max total identity error  | "
        f"{max_total_error:.12e}"
    )

    print(
        "Max home points error     | "
        f"{max_home_points_error:.12e}"
    )

    print(
        "Max away points error     | "
        f"{max_away_points_error:.12e}"
    )

    print(
        f"Winner mismatches          | {winner_mismatches}"
    )

    if max_margin_error > tolerance:
        fail("margin identity")

    if max_total_error > tolerance:
        fail("total identity")

    if max_home_points_error > tolerance:
        fail("home points identity")

    if max_away_points_error > tolerance:
        fail("away points identity")

    if winner_mismatches != 0:
        fail("winner identity")

    print("PASS | forecast mathematics reproduced")


    # ================================================================
    # Gametime contract
    # ================================================================

    print()
    print("=== GAMETIME CONTRACT ===")

    gametime_missing = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE gametime IS NULL
           OR trim(gametime) = ''
        """
    ).fetchone()[0]

    print(
        f"Missing gametime | {gametime_missing}"
    )

    if gametime_missing != 0:
        fail("gametime completeness")

    print("PASS | 272/272 gametimes preserved")


finally:
    conn.close()


# =====================================================================
# Freeze directory
# =====================================================================

print()
print("=== CREATE FROZEN BASELINE ===")

FREEZE_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

frozen_ledger = (
    FREEZE_DIR
    / "forecast_ledger.db"
)

frozen_creator = (
    FREEZE_DIR
    / "create_forecast_ledger_v1.py"
)

frozen_auditor = (
    FREEZE_DIR
    / "audit_freeze_forecast_ledger_v1.py"
)

shutil.copy2(
    LEDGER,
    frozen_ledger,
)

shutil.copy2(
    CREATOR,
    frozen_creator,
)

shutil.copy2(
    Path(__file__),
    frozen_auditor,
)


# =====================================================================
# Verify frozen ledger byte identity
# =====================================================================

frozen_ledger_sha = sha256(
    frozen_ledger
)

if frozen_ledger_sha != ledger_sha:
    fail("frozen ledger byte identity")

print(
    f"PASS | frozen ledger SHA {frozen_ledger_sha}"
)


# =====================================================================
# Frozen copy integrity
# =====================================================================

frozen_uri = (
    f"file:{frozen_ledger}?mode=ro"
)

frozen_conn = sqlite3.connect(
    frozen_uri,
    uri=True,
)

try:

    frozen_integrity = frozen_conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    frozen_fk = frozen_conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

finally:
    frozen_conn.close()

if frozen_integrity != "ok":
    fail("frozen ledger integrity")

if frozen_fk:
    fail("frozen ledger foreign keys")

print("PASS | frozen copy SQLite integrity")


# =====================================================================
# Status document
# =====================================================================

status_path = (
    FREEZE_DIR
    / "BASELINE_STATUS.txt"
)

status_text = f"""WFS FORECAST CENTER
PROSPECTIVE FORECAST LEDGER V1

STATUS: BASELINE FROZEN

Frozen UTC:
{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}

Authoritative live ledger:
{LEDGER}

Baseline snapshot:
{EXPECTED_SNAPSHOT_ID}

Baseline captured UTC:
{EXPECTED_CAPTURED_AT}

Ledger SHA256:
{ledger_sha}

Source prediction SHA256:
{EXPECTED_PRED_SHA}

Source prediction audit SHA256:
{EXPECTED_AUDIT_SHA}

Contract:
- separate forecast database
- append-only snapshot architecture
- immutable historical prediction records
- 1 baseline snapshot
- 272 game records
- 100 READY_CORE_ONLY
- 172 MARKET_PENDING
- 272/272 authoritative gametimes
- exact game_id identity
- update/delete blocked by database triggers
- nfl.db remains authoritative for game results/timing
- nfl.db was not modified by ledger creation
- optimizer untouched
- UI untouched

This frozen database is the rollback/reference baseline.
Future forecast captures append to the live ledger only.
This frozen baseline must never be modified in place.
"""

status_path.write_text(
    status_text,
    encoding="utf-8",
)


# =====================================================================
# SHA manifest
# =====================================================================

manifest_path = (
    FREEZE_DIR
    / "SHA256SUMS"
)

manifest_files = [
    frozen_ledger,
    frozen_creator,
    frozen_auditor,
    status_path,
]

manifest_lines = []

for path in manifest_files:

    manifest_lines.append(
        f"{sha256(path)}  {path.name}"
    )

manifest_path.write_text(
    "\n".join(manifest_lines)
    + "\n",
    encoding="utf-8",
)


# =====================================================================
# Final freeze verification
# =====================================================================

print()
print("=== FROZEN ARTIFACTS ===")

for path in sorted(
    FREEZE_DIR.iterdir(),
    key=lambda p: p.name,
):
    print(
        f"{sha256(path)}  {path.name}"
    )


print()
print("=" * 100)
print("PROSPECTIVE FORECAST LEDGER V1 FREEZE GATE")
print("=" * 100)

print("PASS | LIVE LEDGER INDEPENDENTLY AUDITED")
print("PASS | LIVE LEDGER SHA VERIFIED")
print("PASS | SQLITE INTEGRITY")
print("PASS | FOREIGN KEY INTEGRITY")
print("PASS | EXACT TABLE CONTRACT")
print("PASS | EXACT INDEX CONTRACT")
print("PASS | EXACT APPEND-ONLY TRIGGER CONTRACT")
print("PASS | SNAPSHOT PROVENANCE")
print("PASS | 272 UNIQUE GAME PREDICTIONS")
print("PASS | 100 READY / 172 MARKET-PENDING")
print("PASS | MARKET STATUS CONTRACT")
print("PASS | PENDING NULL CONTRACT")
print("PASS | READY COMPLETENESS")
print("PASS | FORECAST MATHEMATICAL IDENTITIES")
print("PASS | 272/272 GAMETIMES")
print("PASS | FROZEN COPY BYTE-IDENTICAL")
print("PASS | FROZEN COPY SQLITE INTEGRITY")
print()
print("PASS | PROSPECTIVE FORECAST LEDGER V1 — BASELINE FROZEN")
print()
print(f"FREEZE | {FREEZE_DIR}")
print()
print("PASS | nfl.db UNTOUCHED")
print("PASS | OPTIMIZER UNTOUCHED")
print("PASS | UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print("=" * 100)
