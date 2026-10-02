from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


# =====================================================================
# Paths / constants
# =====================================================================

ROOT = Path("/home/mwynn/nfl_data_engine")

NFL_DB = ROOT / "data" / "nfl.db"

LEDGER_DB = (
    ROOT
    / "data"
    / "forecast_ledger.db"
)

PREDICTIONS = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

PREDICTIONS_AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions_audit.json"
)

BACKUP_DIR = (
    ROOT
    / "backups"
    / "forecast_ledger_v1_preappend"
)

EXPECTED_MODEL = "WFS_CORE_RESIDUAL"

EXPECTED_VARIANT = "LIVE_CORE_ONLY_V1"

EXPECTED_SEASON = 2026

EXPECTED_HISTORICAL_PROOF_STATUS = "PASS"

EXPECTED_INJURY_SOURCE_STATUS = (
    "INJURY_SOURCE_PENDING"
)

EXPECTED_WIN_PROBABILITY_STATUS = (
    "NOT_CALIBRATED"
)

ET = ZoneInfo(
    "America/New_York"
)

TOLERANCE = 1e-10


# =====================================================================
# Helpers
# =====================================================================

def fail(message):
    raise SystemExit(
        f"FAIL | {message}"
    )


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def ro_connection(path):
    return sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )


def sql_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def normalize_int(value, label):

    if pd.isna(value):
        fail(
            f"{label} is NULL"
        )

    try:
        return int(value)
    except Exception:
        fail(
            f"{label} is not integer-compatible: "
            f"{value!r}"
        )


def parse_kickoff(
    game_date,
    gametime,
    game_id,
):

    raw = (
        f"{str(game_date).strip()} "
        f"{str(gametime).strip()}"
    )

    try:

        naive = datetime.strptime(
            raw,
            "%Y-%m-%d %H:%M",
        )

    except ValueError:

        fail(
            "invalid authoritative kickoff for "
            f"{game_id}: {raw}"
        )

    kickoff_et = naive.replace(
        tzinfo=ET
    )

    kickoff_utc = kickoff_et.astimezone(
        timezone.utc
    )

    return (
        kickoff_et,
        kickoff_utc,
    )


def logical_snapshot_sha(
    conn,
    snapshot_id,
):

    rows = conn.execute(
        """
        SELECT *
        FROM forecast_predictions
        WHERE snapshot_id = ?
        ORDER BY game_id
        """,
        (
            snapshot_id,
        ),
    ).fetchall()

    payload = "\n".join(
        "|".join(
            ""
            if value is None
            else str(value)
            for value in row
        )
        for row in rows
    ).encode(
        "utf-8"
    )

    return (
        len(rows),
        hashlib.sha256(
            payload
        ).hexdigest(),
    )


# =====================================================================
# Start
# =====================================================================

print("=" * 100)
print(
    "WFS FORECAST CENTER — "
    "PROSPECTIVE SNAPSHOT CAPTURE WRITER V1"
)
print("=" * 100)


# =====================================================================
# File gate
# =====================================================================

print()
print("=== REQUIRED FILES ===")

for path in [
    NFL_DB,
    LEDGER_DB,
    PREDICTIONS,
    PREDICTIONS_AUDIT,
]:

    if not path.is_file():

        fail(
            f"missing required file: {path}"
        )

    print(
        f"PASS | {path}"
    )


prediction_sha = sha256(
    PREDICTIONS
)

audit_sha = sha256(
    PREDICTIONS_AUDIT
)

print()
print("=== SOURCE ARTIFACT PROVENANCE ===")

print(
    f"Prediction SHA | {prediction_sha}"
)

print(
    f"Audit SHA      | {audit_sha}"
)


# =====================================================================
# Load source forecast
# =====================================================================

pred = pd.read_csv(
    PREDICTIONS
)

print()
print("=== SOURCE FORECAST CONTRACT ===")

print(
    f"Rows | {len(pred)}"
)

if len(pred) != 272:
    fail(
        "prediction artifact must contain 272 rows"
    )

if pred["game_id"].nunique() != 272:
    fail(
        "prediction game_id uniqueness"
    )


required_columns = [
    "game_id",
    "season",
    "week",
    "game_date",
    "away_team",
    "home_team",
    "market_home_spread_raw",
    "market_total",
    "forecast_market_ready",
    "pred_margin_residual",
    "pred_total_residual",
    "pred_home_margin",
    "pred_total_points",
    "pred_home_points",
    "pred_away_points",
    "pred_winner",
    "model",
    "forecast_variant",
    "impact_status",
    "replacement_status",
    "win_probability_status",
    "forecast_status",
]

missing_columns = [
    col
    for col in required_columns
    if col not in pred.columns
]

if missing_columns:

    fail(
        "source forecast missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# =====================================================================
# Model contract
# =====================================================================

models = set(
    pred["model"]
    .dropna()
    .astype(str)
    .unique()
)

variants = set(
    pred["forecast_variant"]
    .dropna()
    .astype(str)
    .unique()
)

impact_statuses = set(
    pred["impact_status"]
    .dropna()
    .astype(str)
    .unique()
)

replacement_statuses = set(
    pred["replacement_status"]
    .dropna()
    .astype(str)
    .unique()
)

winprob_statuses = set(
    pred["win_probability_status"]
    .dropna()
    .astype(str)
    .unique()
)


print(
    f"Model        | {models}"
)

print(
    f"Variant      | {variants}"
)

print(
    f"Impact       | {impact_statuses}"
)

print(
    f"Replacement  | {replacement_statuses}"
)

print(
    f"WinProb      | {winprob_statuses}"
)


if models != {
    EXPECTED_MODEL
}:
    fail(
        "unexpected model contract"
    )

if variants != {
    EXPECTED_VARIANT
}:
    fail(
        "unexpected forecast variant"
    )

if impact_statuses != {
    EXPECTED_INJURY_SOURCE_STATUS
}:
    fail(
        "unexpected impact status"
    )

if replacement_statuses != {
    EXPECTED_INJURY_SOURCE_STATUS
}:
    fail(
        "unexpected replacement status"
    )

if winprob_statuses != {
    EXPECTED_WIN_PROBABILITY_STATUS
}:
    fail(
        "unexpected win-probability status"
    )

if set(
    pred["season"]
    .dropna()
    .astype(int)
    .unique()
) != {
    EXPECTED_SEASON
}:
    fail(
        "unexpected season contract"
    )

print(
    "PASS | model/status contract"
)


# =====================================================================
# Status / null contract
# =====================================================================

print()
print("=== FORECAST STATUS CONTRACT ===")

status_counts = (
    pred["forecast_status"]
    .value_counts(
        dropna=False
    )
)

for status, count in status_counts.items():

    print(
        f"{str(status):<20} | {int(count)}"
    )


allowed_statuses = {
    "READY_CORE_ONLY",
    "MARKET_PENDING",
}

actual_statuses = set(
    pred["forecast_status"]
    .dropna()
    .astype(str)
    .unique()
)

if not actual_statuses.issubset(
    allowed_statuses
):
    fail(
        "unexpected forecast_status value"
    )


ready_mask = (
    pred["forecast_status"]
    == "READY_CORE_ONLY"
)

pending_mask = (
    pred["forecast_status"]
    == "MARKET_PENDING"
)


if not (
    pred.loc[
        ready_mask,
        "forecast_market_ready",
    ]
    .astype(int)
    .eq(1)
    .all()
):
    fail(
        "READY_CORE_ONLY market-ready mismatch"
    )


if not (
    pred.loc[
        pending_mask,
        "forecast_market_ready",
    ]
    .astype(int)
    .eq(0)
    .all()
):
    fail(
        "MARKET_PENDING market-ready mismatch"
    )


prediction_fields = [
    "pred_margin_residual",
    "pred_total_residual",
    "pred_home_margin",
    "pred_total_points",
    "pred_home_points",
    "pred_away_points",
    "pred_winner",
]


pending_violations = int(
    pred.loc[
        pending_mask,
        prediction_fields,
    ]
    .notna()
    .any(
        axis=1
    )
    .sum()
)

print(
    f"Pending null violations | "
    f"{pending_violations}"
)

if pending_violations != 0:
    fail(
        "market-pending null contract"
    )


ready_missing = int(
    pred.loc[
        ready_mask,
        [
            "market_home_spread_raw",
            "market_total",
            *prediction_fields,
        ],
    ]
    .isna()
    .any(
        axis=1
    )
    .sum()
)

print(
    f"Ready incomplete rows   | "
    f"{ready_missing}"
)

if ready_missing != 0:
    fail(
        "ready forecast completeness"
    )

print(
    "PASS | forecast status/null contract"
)


# =====================================================================
# Forecast math
# =====================================================================

print()
print("=== FORECAST MATH CONTRACT ===")

max_margin_error = 0.0
max_total_error = 0.0
max_home_error = 0.0
max_away_error = 0.0
winner_mismatch = 0


for row in pred.loc[
    ready_mask
].itertuples(
    index=False
):

    expected_margin = (
        float(
            row.market_home_spread_raw
        )
        + float(
            row.pred_margin_residual
        )
    )

    expected_total = (
        float(
            row.market_total
        )
        + float(
            row.pred_total_residual
        )
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
            float(
                row.pred_home_margin
            )
            - expected_margin
        ),
    )

    max_total_error = max(
        max_total_error,
        abs(
            float(
                row.pred_total_points
            )
            - expected_total
        ),
    )

    max_home_error = max(
        max_home_error,
        abs(
            float(
                row.pred_home_points
            )
            - expected_home
        ),
    )

    max_away_error = max(
        max_away_error,
        abs(
            float(
                row.pred_away_points
            )
            - expected_away
        ),
    )

    expected_winner = (
        row.home_team
        if float(
            row.pred_home_margin
        ) > 0
        else row.away_team
    )

    if (
        str(
            row.pred_winner
        )
        != str(
            expected_winner
        )
    ):
        winner_mismatch += 1


print(
    "Max margin error | "
    f"{max_margin_error:.12e}"
)

print(
    "Max total error  | "
    f"{max_total_error:.12e}"
)

print(
    "Max home error   | "
    f"{max_home_error:.12e}"
)

print(
    "Max away error   | "
    f"{max_away_error:.12e}"
)

print(
    f"Winner mismatch  | "
    f"{winner_mismatch}"
)


if max_margin_error > TOLERANCE:
    fail(
        "margin identity"
    )

if max_total_error > TOLERANCE:
    fail(
        "total identity"
    )

if max_home_error > TOLERANCE:
    fail(
        "home points identity"
    )

if max_away_error > TOLERANCE:
    fail(
        "away points identity"
    )

if winner_mismatch != 0:
    fail(
        "winner identity"
    )

print(
    "PASS | mathematical contract"
)


# =====================================================================
# Open nfl.db READ ONLY
# =====================================================================

nfl = ro_connection(
    NFL_DB
)

try:

    if (
        nfl.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        != "ok"
    ):
        fail(
            "nfl.db integrity"
        )

    games = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            game_date,
            gametime,
            away_team,
            home_team,
            completed
        FROM games
        WHERE season = 2026
        ORDER BY
            week,
            game_date,
            gametime,
            game_id
        """,
        nfl,
    )

finally:

    nfl.close()


print()
print("=== AUTHORITATIVE SCHEDULE ===")

print(
    f"Rows | {len(games)}"
)

if len(games) != 272:
    fail(
        "2026 authoritative schedule count"
    )

if games["game_id"].nunique() != 272:
    fail(
        "schedule game_id uniqueness"
    )

if games["game_date"].isna().any():
    fail(
        "missing authoritative game_date"
    )

if games["gametime"].isna().any():
    fail(
        "missing authoritative gametime"
    )

print(
    "PASS | authoritative schedule"
)


# =====================================================================
# Exact schedule / prediction identity
# =====================================================================

joined = pred.merge(
    games,
    on="game_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=(
        "_pred",
        "_db",
    ),
)


if int(
    (
        joined["_merge"]
        != "both"
    ).sum()
) != 0:
    fail(
        "forecast/schedule coverage mismatch"
    )


identity_pairs = [
    (
        "season_pred",
        "season_db",
        "season",
    ),
    (
        "week_pred",
        "week_db",
        "week",
    ),
    (
        "game_date_pred",
        "game_date_db",
        "game_date",
    ),
    (
        "away_team_pred",
        "away_team_db",
        "away_team",
    ),
    (
        "home_team_pred",
        "home_team_db",
        "home_team",
    ),
]


print()
print("=== EXACT GAME IDENTITY ===")

for left, right, label in identity_pairs:

    mismatch = int(
        (
            joined[left]
            .astype(str)
            !=
            joined[right]
            .astype(str)
        ).sum()
    )

    print(
        f"{label:<10} mismatches | "
        f"{mismatch}"
    )

    if mismatch != 0:

        fail(
            f"{label} identity mismatch"
        )

print(
    "PASS | exact forecast/schedule identity"
)


# =====================================================================
# Capture time / kickoff
# =====================================================================

captured_utc = datetime.now(
    timezone.utc
)

captured_et = captured_utc.astimezone(
    ET
)


print()
print("=== CAPTURE INSTANT ===")

print(
    f"Captured UTC | "
    f"{captured_utc.isoformat()}"
)

print(
    f"Captured ET  | "
    f"{captured_et.isoformat()}"
)


kickoff_et = []
kickoff_utc = []
pre_kickoff = []


for row in games.itertuples(
    index=False
):

    ko_et, ko_utc = parse_kickoff(
        row.game_date,
        row.gametime,
        row.game_id,
    )

    kickoff_et.append(
        ko_et
    )

    kickoff_utc.append(
        ko_utc
    )

    pre_kickoff.append(
        captured_utc
        < ko_utc
    )


games = games.copy()

games["kickoff_et"] = kickoff_et
games["kickoff_utc"] = kickoff_utc
games["pre_kickoff"] = pre_kickoff


eligible_count = int(
    games["pre_kickoff"].sum()
)

ineligible_count = int(
    (
        ~games["pre_kickoff"]
    ).sum()
)


print()
print("=== KICKOFF GATE ===")

print(
    f"Eligible pre-kickoff | "
    f"{eligible_count}"
)

print(
    f"At/after kickoff     | "
    f"{ineligible_count}"
)

if (
    eligible_count
    + ineligible_count
    != 272
):
    fail(
        "kickoff classification population"
    )


# =====================================================================
# Merge authoritative gametime / eligibility
# =====================================================================

capture_df = pred.merge(
    games[
        [
            "game_id",
            "gametime",
            "kickoff_et",
            "kickoff_utc",
            "pre_kickoff",
            "completed",
        ]
    ],
    on="game_id",
    how="left",
    validate="one_to_one",
)


eligible = capture_df.loc[
    capture_df["pre_kickoff"]
].copy()


if eligible.empty:

    fail(
        "no pre-kickoff games remain; "
        "refusing empty snapshot"
    )


print()
print("=== ELIGIBLE SNAPSHOT POPULATION ===")

print(
    f"Eligible rows | "
    f"{len(eligible)}"
)

eligible_status_counts = (
    eligible["forecast_status"]
    .value_counts(
        dropna=False
    )
)

for status, count in (
    eligible_status_counts.items()
):

    print(
        f"{str(status):<20} | "
        f"{int(count)}"
    )


# =====================================================================
# Ledger read-only preflight
# =====================================================================

ledger_ro = ro_connection(
    LEDGER_DB
)

try:

    print()
    print("=== LEDGER PREFLIGHT ===")

    integrity = ledger_ro.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    fk_rows = ledger_ro.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print(
        f"integrity_check       | "
        f"{integrity}"
    )

    print(
        f"foreign_key_check rows| "
        f"{len(fk_rows)}"
    )

    if integrity != "ok":
        fail(
            "ledger integrity before capture"
        )

    if fk_rows:
        fail(
            "ledger foreign keys before capture"
        )


    # -------------------------------------------------------------
    # Exact tables
    # -------------------------------------------------------------

    tables = [
        row[0]
        for row in ledger_ro.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )
    ]

    if tables != [
        "forecast_predictions",
        "forecast_snapshots",
    ]:
        fail(
            "unexpected ledger table contract"
        )


    # -------------------------------------------------------------
    # Existing source hash = HARD BLOCK
    # -------------------------------------------------------------

    duplicate = ledger_ro.execute(
        """
        SELECT
            snapshot_id,
            captured_at_utc
        FROM forecast_snapshots
        WHERE source_prediction_sha256 = ?
        ORDER BY captured_at_utc
        """,
        (
            prediction_sha,
        ),
    ).fetchall()


    if duplicate:

        print()
        print(
            "BLOCK | DUPLICATE SOURCE ARTIFACT"
        )

        for (
            existing_snapshot,
            existing_time,
        ) in duplicate:

            print(
                "Existing | "
                f"{existing_snapshot} | "
                f"{existing_time}"
            )

        print()
        print(
            "PASS | production writer refused "
            "duplicate source hash"
        )

        print(
            "PASS | no backup required because "
            "no ledger write was attempted"
        )

        print(
            "PASS | forecast_ledger.db UNTOUCHED"
        )

        print(
            "PASS | nfl.db UNTOUCHED"
        )

        print(
            "PASS | OPTIMIZER / UI UNTOUCHED"
        )

        print(
            "PASS | SITE RESTART NOT REQUIRED"
        )

        print("=" * 100)

        raise SystemExit(0)


    # -------------------------------------------------------------
    # Record old state/fingerprints
    # -------------------------------------------------------------

    old_snapshot_rows = (
        ledger_ro.execute(
            """
            SELECT
                snapshot_id
            FROM forecast_snapshots
            ORDER BY captured_at_utc,
                     snapshot_id
            """
        ).fetchall()
    )

    old_snapshot_ids = [
        row[0]
        for row in old_snapshot_rows
    ]

    old_fingerprints = {}

    for snapshot_id in old_snapshot_ids:

        old_fingerprints[
            snapshot_id
        ] = logical_snapshot_sha(
            ledger_ro,
            snapshot_id,
        )

    old_snapshot_count = len(
        old_snapshot_ids
    )

    old_prediction_count = (
        ledger_ro.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            """
        ).fetchone()[0]
    )

finally:

    ledger_ro.close()


print(
    "PASS | source artifact is new"
)

print(
    f"Existing snapshots   | "
    f"{old_snapshot_count}"
)

print(
    f"Existing predictions | "
    f"{old_prediction_count}"
)


# =====================================================================
# Generate immutable snapshot ID
# =====================================================================

capture_stamp = (
    captured_utc
    .strftime(
        "%Y%m%dT%H%M%SZ"
    )
)

snapshot_id = (
    "2026-core-v1-"
    f"{capture_stamp}-"
    f"{prediction_sha[:12]}"
)

captured_at_utc = (
    captured_utc
    .replace(
        microsecond=0
    )
    .isoformat()
)


print()
print("=== NEW SNAPSHOT IDENTITY ===")

print(
    f"Snapshot ID  | {snapshot_id}"
)

print(
    f"Captured UTC | {captured_at_utc}"
)

print(
    f"Rows         | {len(eligible)}"
)


# =====================================================================
# Ensure snapshot ID itself is new
# =====================================================================

ledger_ro = ro_connection(
    LEDGER_DB
)

try:

    collision = ledger_ro.execute(
        """
        SELECT COUNT(*)
        FROM forecast_snapshots
        WHERE snapshot_id = ?
        """,
        (
            snapshot_id,
        ),
    ).fetchone()[0]

finally:

    ledger_ro.close()


if collision != 0:

    fail(
        "generated snapshot_id already exists"
    )


# =====================================================================
# BACKUP BEFORE WRITE
# =====================================================================

print()
print("=== PRE-APPEND BACKUP ===")

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

prewrite_db_sha = sha256(
    LEDGER_DB
)

backup_name = (
    "forecast_ledger_before_"
    f"{capture_stamp}_"
    f"{prewrite_db_sha[:12]}"
    ".db"
)

backup_path = (
    BACKUP_DIR
    / backup_name
)


if backup_path.exists():

    fail(
        "pre-append backup path already exists"
    )


shutil.copy2(
    LEDGER_DB,
    backup_path,
)


backup_sha = sha256(
    backup_path
)

print(
    f"Live SHA   | {prewrite_db_sha}"
)

print(
    f"Backup     | {backup_path}"
)

print(
    f"Backup SHA | {backup_sha}"
)


if backup_sha != prewrite_db_sha:

    fail(
        "pre-append backup byte mismatch"
    )


backup_ro = ro_connection(
    backup_path
)

try:

    backup_integrity = (
        backup_ro.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
    )

    backup_fk = (
        backup_ro.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    )

finally:

    backup_ro.close()


if backup_integrity != "ok":
    fail(
        "pre-append backup integrity"
    )

if backup_fk:
    fail(
        "pre-append backup foreign keys"
    )

print(
    "PASS | pre-append rollback backup verified"
)


# =====================================================================
# INSERT TRANSACTION
# =====================================================================

print()
print("=== APPEND TRANSACTION ===")

conn = sqlite3.connect(
    LEDGER_DB
)

try:

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.execute(
        "BEGIN IMMEDIATE"
    )


    # -------------------------------------------------------------
    # Re-check duplicate INSIDE transaction
    # -------------------------------------------------------------

    duplicate_inside = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_snapshots
        WHERE source_prediction_sha256 = ?
        """,
        (
            prediction_sha,
        ),
    ).fetchone()[0]

    if duplicate_inside != 0:

        raise RuntimeError(
            "source prediction hash became duplicate "
            "before transaction insert"
        )


    # -------------------------------------------------------------
    # Snapshot metadata
    # -------------------------------------------------------------

    conn.execute(
        """
        INSERT INTO forecast_snapshots (
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            captured_at_utc,
            EXPECTED_SEASON,
            EXPECTED_MODEL,
            EXPECTED_VARIANT,
            prediction_sha,
            audit_sha,
            EXPECTED_HISTORICAL_PROOF_STATUS,
            EXPECTED_INJURY_SOURCE_STATUS,
            EXPECTED_WIN_PROBABILITY_STATUS,
            "PROSPECTIVE_CAPTURE",
        ),
    )


    # -------------------------------------------------------------
    # Prediction rows
    # -------------------------------------------------------------

    insert_sql = """
        INSERT INTO forecast_predictions (
            snapshot_id,
            game_id,
            season,
            week,
            game_date,
            gametime,
            away_team,
            home_team,
            market_home_spread_raw,
            market_total,
            forecast_market_ready,
            pred_margin_residual,
            pred_total_residual,
            pred_home_margin,
            pred_total_points,
            pred_home_points,
            pred_away_points,
            pred_winner,
            model,
            forecast_variant,
            impact_status,
            replacement_status,
            win_probability_status,
            forecast_status
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
    """


    insert_rows = []


    for row in eligible.itertuples(
        index=False
    ):

        insert_rows.append(
            (
                snapshot_id,
                str(
                    row.game_id
                ),
                normalize_int(
                    row.season,
                    "season",
                ),
                normalize_int(
                    row.week,
                    "week",
                ),
                str(
                    row.game_date
                ),
                str(
                    row.gametime
                ),
                str(
                    row.away_team
                ),
                str(
                    row.home_team
                ),
                sql_value(
                    row.market_home_spread_raw
                ),
                sql_value(
                    row.market_total
                ),
                normalize_int(
                    row.forecast_market_ready,
                    "forecast_market_ready",
                ),
                sql_value(
                    row.pred_margin_residual
                ),
                sql_value(
                    row.pred_total_residual
                ),
                sql_value(
                    row.pred_home_margin
                ),
                sql_value(
                    row.pred_total_points
                ),
                sql_value(
                    row.pred_home_points
                ),
                sql_value(
                    row.pred_away_points
                ),
                sql_value(
                    row.pred_winner
                ),
                str(
                    row.model
                ),
                str(
                    row.forecast_variant
                ),
                str(
                    row.impact_status
                ),
                str(
                    row.replacement_status
                ),
                str(
                    row.win_probability_status
                ),
                str(
                    row.forecast_status
                ),
            )
        )


    conn.executemany(
        insert_sql,
        insert_rows,
    )


    # -------------------------------------------------------------
    # Transaction-local validation
    # -------------------------------------------------------------

    new_snapshot_count = (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_snapshots
            WHERE snapshot_id = ?
            """,
            (
                snapshot_id,
            ),
        ).fetchone()[0]
    )

    new_prediction_count = (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE snapshot_id = ?
            """,
            (
                snapshot_id,
            ),
        ).fetchone()[0]
    )


    if new_snapshot_count != 1:

        raise RuntimeError(
            "transaction-local snapshot count != 1"
        )

    if new_prediction_count != len(
        eligible
    ):

        raise RuntimeError(
            "transaction-local prediction count mismatch"
        )


    duplicate_games = (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    game_id,
                    COUNT(*) AS n
                FROM forecast_predictions
                WHERE snapshot_id = ?
                GROUP BY game_id
                HAVING COUNT(*) <> 1
            )
            """,
            (
                snapshot_id,
            ),
        ).fetchone()[0]
    )

    if duplicate_games != 0:

        raise RuntimeError(
            "duplicate game rows inside new snapshot"
        )


    pending_violation = (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE snapshot_id = ?
              AND forecast_status = 'MARKET_PENDING'
              AND (
                    pred_margin_residual IS NOT NULL
                 OR pred_total_residual IS NOT NULL
                 OR pred_home_margin IS NOT NULL
                 OR pred_total_points IS NOT NULL
                 OR pred_home_points IS NOT NULL
                 OR pred_away_points IS NOT NULL
                 OR pred_winner IS NOT NULL
              )
            """,
            (
                snapshot_id,
            ),
        ).fetchone()[0]
    )

    if pending_violation != 0:

        raise RuntimeError(
            "pending null violation inside transaction"
        )


    fk_inside = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    if fk_inside:

        raise RuntimeError(
            "foreign key violation inside transaction"
        )


    conn.commit()

    print(
        "PASS | transaction committed"
    )


except Exception as exc:

    conn.rollback()

    print(
        f"ROLLBACK | {exc}"
    )

    raise


finally:

    conn.close()


# =====================================================================
# Post-write independent audit
# =====================================================================

print()
print("=== POST-WRITE READ-ONLY AUDIT ===")

post = ro_connection(
    LEDGER_DB
)

try:

    integrity_after = post.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    fk_after = post.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    snapshot_count_after = (
        post.execute(
            """
            SELECT COUNT(*)
            FROM forecast_snapshots
            """
        ).fetchone()[0]
    )

    prediction_count_after = (
        post.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            """
        ).fetchone()[0]
    )


    print(
        f"integrity_check | "
        f"{integrity_after}"
    )

    print(
        f"FK violations  | "
        f"{len(fk_after)}"
    )

    print(
        f"Snapshots      | "
        f"{snapshot_count_after}"
    )

    print(
        f"Predictions    | "
        f"{prediction_count_after}"
    )


    if integrity_after != "ok":
        fail(
            "post-write ledger integrity"
        )

    if fk_after:
        fail(
            "post-write foreign keys"
        )


    if (
        snapshot_count_after
        != old_snapshot_count + 1
    ):
        fail(
            "post-write snapshot population"
        )


    if (
        prediction_count_after
        != old_prediction_count
        + len(
            eligible
        )
    ):
        fail(
            "post-write prediction population"
        )


    # -------------------------------------------------------------
    # Exact new snapshot provenance
    # -------------------------------------------------------------

    new_meta = post.execute(
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
        WHERE snapshot_id = ?
        """,
        (
            snapshot_id,
        ),
    ).fetchone()


    expected_meta = (
        snapshot_id,
        captured_at_utc,
        EXPECTED_SEASON,
        EXPECTED_MODEL,
        EXPECTED_VARIANT,
        prediction_sha,
        audit_sha,
        EXPECTED_HISTORICAL_PROOF_STATUS,
        EXPECTED_INJURY_SOURCE_STATUS,
        EXPECTED_WIN_PROBABILITY_STATUS,
        "PROSPECTIVE_CAPTURE",
    )


    if new_meta != expected_meta:

        fail(
            "new snapshot metadata mismatch"
        )


    # -------------------------------------------------------------
    # Prove every old snapshot unchanged
    # -------------------------------------------------------------

    print()
    print(
        "=== PRIOR SNAPSHOT IMMUTABILITY ==="
    )

    for old_snapshot_id in old_snapshot_ids:

        before_count, before_sha = (
            old_fingerprints[
                old_snapshot_id
            ]
        )

        after_count, after_sha = (
            logical_snapshot_sha(
                post,
                old_snapshot_id,
            )
        )

        print(
            f"{old_snapshot_id}"
        )

        print(
            f"  rows before/after | "
            f"{before_count}/{after_count}"
        )

        print(
            f"  SHA before        | "
            f"{before_sha}"
        )

        print(
            f"  SHA after         | "
            f"{after_sha}"
        )


        if before_count != after_count:

            fail(
                "prior snapshot row count changed"
            )

        if before_sha != after_sha:

            fail(
                "prior snapshot logical fingerprint changed"
            )


    print(
        "PASS | all prior snapshots unchanged"
    )


    # -------------------------------------------------------------
    # New snapshot row count
    # -------------------------------------------------------------

    actual_new_rows = (
        post.execute(
            """
            SELECT COUNT(*)
            FROM forecast_predictions
            WHERE snapshot_id = ?
            """,
            (
                snapshot_id,
            ),
        ).fetchone()[0]
    )

    if actual_new_rows != len(
        eligible
    ):

        fail(
            "new snapshot final row count"
        )


finally:

    post.close()


postwrite_sha = sha256(
    LEDGER_DB
)


# =====================================================================
# Final report
# =====================================================================

print()
print("=" * 100)
print(
    "PROSPECTIVE SNAPSHOT CAPTURE V1 — FINAL GATE"
)
print("=" * 100)

print(
    f"PASS | SNAPSHOT APPENDED | "
    f"{snapshot_id}"
)

print(
    f"PASS | CAPTURED UTC      | "
    f"{captured_at_utc}"
)

print(
    f"PASS | ELIGIBLE ROWS     | "
    f"{len(eligible)}"
)

print(
    f"PASS | EXCLUDED KICKED   | "
    f"{ineligible_count}"
)

print(
    f"PASS | SOURCE SHA        | "
    f"{prediction_sha}"
)

print(
    f"PASS | SOURCE AUDIT SHA  | "
    f"{audit_sha}"
)

print(
    f"PASS | PREWRITE DB SHA   | "
    f"{prewrite_db_sha}"
)

print(
    f"PASS | POSTWRITE DB SHA  | "
    f"{postwrite_sha}"
)

print(
    f"PASS | ROLLBACK BACKUP   | "
    f"{backup_path}"
)

print()
print(
    "PASS | STRICT PRE-KICKOFF RULE"
)

print(
    "PASS | America/New_York DST-AWARE"
)

print(
    "PASS | DUPLICATE SOURCE PROTECTION"
)

print(
    "PASS | APPEND-ONLY TRANSACTION"
)

print(
    "PASS | PRIOR SNAPSHOTS IMMUTABLE"
)

print(
    "PASS | SQLITE INTEGRITY"
)

print(
    "PASS | FOREIGN KEY INTEGRITY"
)

print(
    "PASS | nfl.db READ-ONLY / UNTOUCHED"
)

print(
    "PASS | OPTIMIZER / UI UNTOUCHED"
)

print(
    "PASS | SITE RESTART NOT REQUIRED"
)

print("=" * 100)
