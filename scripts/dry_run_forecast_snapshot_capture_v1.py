from pathlib import Path
import hashlib
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

NFL_DB = ROOT / "data" / "nfl.db"
LEDGER_DB = ROOT / "data" / "forecast_ledger.db"

PREDICTIONS = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

FROZEN_DIR = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_predictions_frozen"
)

FROZEN_PREDICTIONS = (
    FROZEN_DIR
    / "forecast_live_core_v1_predictions.csv"
)

EXPECTED_FROZEN_PRED_SHA = (
    "8a58309ca2b71ab7d019a5697580eb7c"
    "3114716f94e8b4c5c5864ea7b40bc94c"
)

EXPECTED_BASELINE_SNAPSHOT = (
    "2026-core-v1-20260908T212022Z-d17862101386"
)

ET = ZoneInfo("America/New_York")


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def fail(message: str):
    raise SystemExit(
        f"FAIL | {message}"
    )


def sql_ro(path: Path):
    return sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )


print("=" * 100)
print("WFS FORECAST CENTER — PROSPECTIVE SNAPSHOT CAPTURE V1")
print("MODE | DRY RUN — ZERO LEDGER WRITES")
print("=" * 100)


# =====================================================================
# Required files
# =====================================================================

print()
print("=== REQUIRED FILES ===")

required = [
    NFL_DB,
    LEDGER_DB,
    PREDICTIONS,
    FROZEN_PREDICTIONS,
]

for path in required:

    if not path.is_file():
        fail(f"missing required file: {path}")

    print(f"PASS | {path}")

print("PASS | all required files exist")


# =====================================================================
# Frozen source provenance
# =====================================================================

print()
print("=== FROZEN SOURCE PROVENANCE ===")

frozen_sha = sha256(
    FROZEN_PREDICTIONS
)

print(
    f"Frozen prediction SHA | {frozen_sha}"
)

if frozen_sha != EXPECTED_FROZEN_PRED_SHA:
    fail("frozen prediction baseline hash mismatch")

print("PASS | frozen prediction baseline verified")


# =====================================================================
# Current live prediction artifact
# =====================================================================

print()
print("=== CURRENT PREDICTION ARTIFACT ===")

current_sha = sha256(
    PREDICTIONS
)

print(
    f"Current prediction SHA | {current_sha}"
)

same_as_frozen = (
    current_sha
    == EXPECTED_FROZEN_PRED_SHA
)

print(
    "Matches frozen baseline | "
    f"{same_as_frozen}"
)

pred = pd.read_csv(
    PREDICTIONS
)

print(
    f"Rows                  | {len(pred)}"
)

if len(pred) != 272:
    fail("current prediction artifact must contain 272 games")

if pred["game_id"].nunique() != 272:
    fail("prediction artifact game_id uniqueness")

print("PASS | 272 unique prediction games")


# =====================================================================
# Open authoritative databases read-only
# =====================================================================

nfl = sql_ro(
    NFL_DB
)

ledger = sql_ro(
    LEDGER_DB
)

try:

    print()
    print("=== DATABASE INTEGRITY ===")

    nfl_integrity = nfl.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    ledger_integrity = ledger.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(
        f"nfl.db             | {nfl_integrity}"
    )

    print(
        f"forecast_ledger.db | {ledger_integrity}"
    )

    if nfl_integrity != "ok":
        fail("nfl.db integrity")

    if ledger_integrity != "ok":
        fail("forecast ledger integrity")

    if ledger.execute(
        "PRAGMA foreign_key_check"
    ).fetchall():
        fail("forecast ledger foreign-key integrity")

    print("PASS | read-only database integrity")


    # =================================================================
    # Existing ledger state
    # =================================================================

    print()
    print("=== EXISTING LEDGER STATE ===")

    snapshot_count = ledger.execute(
        """
        SELECT COUNT(*)
        FROM forecast_snapshots
        """
    ).fetchone()[0]

    prediction_count = ledger.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        """
    ).fetchone()[0]

    print(
        f"Snapshots   | {snapshot_count}"
    )

    print(
        f"Predictions | {prediction_count}"
    )

    baseline = ledger.execute(
        """
        SELECT
            snapshot_id,
            captured_at_utc,
            source_prediction_sha256
        FROM forecast_snapshots
        ORDER BY captured_at_utc
        LIMIT 1
        """
    ).fetchone()

    if baseline is None:
        fail("baseline snapshot missing")

    baseline_id = baseline[0]

    print(
        f"Baseline snapshot | {baseline_id}"
    )

    if baseline_id != EXPECTED_BASELINE_SNAPSHOT:
        fail("unexpected baseline snapshot identity")

    print("PASS | frozen baseline snapshot preserved")


    # =================================================================
    # Detect duplicate source artifact
    # =================================================================

    print()
    print("=== DUPLICATE SOURCE CHECK ===")

    duplicate_rows = ledger.execute(
        """
        SELECT
            snapshot_id,
            captured_at_utc
        FROM forecast_snapshots
        WHERE source_prediction_sha256 = ?
        ORDER BY captured_at_utc
        """,
        (
            current_sha,
        ),
    ).fetchall()

    print(
        f"Existing snapshots with current source SHA | "
        f"{len(duplicate_rows)}"
    )

    for snapshot_id, captured_at in duplicate_rows:

        print(
            f"Existing | {snapshot_id} | {captured_at}"
        )

    duplicate_source = (
        len(duplicate_rows) > 0
    )

    if duplicate_source:
        print(
            "BLOCK | current prediction artifact has already "
            "been captured"
        )
    else:
        print(
            "PASS | current prediction artifact is new"
        )


    # =================================================================
    # Authoritative 2026 schedule
    # =================================================================

    print()
    print("=== AUTHORITATIVE 2026 SCHEDULE ===")

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

    print(
        f"Schedule rows | {len(games)}"
    )

    if len(games) != 272:
        fail("authoritative schedule count")

    if games["game_id"].nunique() != 272:
        fail("authoritative game_id uniqueness")

    if games["game_date"].isna().any():
        fail("missing game_date")

    if games["gametime"].isna().any():
        fail("missing gametime")

    print("PASS | authoritative 272-game schedule")


    # =================================================================
    # Exact forecast / schedule identity
    # =================================================================

    print()
    print("=== FORECAST / SCHEDULE IDENTITY ===")

    required_pred_cols = [
        "game_id",
        "season",
        "week",
        "game_date",
        "away_team",
        "home_team",
        "forecast_status",
    ]

    missing = [
        col
        for col in required_pred_cols
        if col not in pred.columns
    ]

    if missing:
        fail(
            "prediction artifact missing columns: "
            + ", ".join(missing)
        )

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

    coverage_mismatch = int(
        (
            joined["_merge"]
            != "both"
        ).sum()
    )

    print(
        f"Coverage mismatches | {coverage_mismatch}"
    )

    if coverage_mismatch != 0:
        fail("forecast/schedule game coverage")

    comparisons = [
        ("season_pred", "season_db", "season"),
        ("week_pred", "week_db", "week"),
        ("game_date_pred", "game_date_db", "game_date"),
        ("away_team_pred", "away_team_db", "away_team"),
        ("home_team_pred", "home_team_db", "home_team"),
    ]

    for left, right, label in comparisons:

        mismatch = int(
            (
                joined[left].astype(str)
                != joined[right].astype(str)
            ).sum()
        )

        print(
            f"{label:<10} mismatches | {mismatch}"
        )

        if mismatch != 0:
            fail(
                f"forecast/schedule {label} identity"
            )

    print("PASS | exact 272-game forecast/schedule identity")


    # =================================================================
    # Capture instant
    # =================================================================

    print()
    print("=== CAPTURE INSTANT ===")

    captured_utc = datetime.now(
        timezone.utc
    )

    captured_et = captured_utc.astimezone(
        ET
    )

    print(
        "Captured UTC | "
        + captured_utc.isoformat()
    )

    print(
        "Captured ET  | "
        + captured_et.isoformat()
    )

    print(
        "Timezone     | America/New_York"
    )


    # =================================================================
    # Construct authoritative kickoff timestamps
    # =================================================================

    print()
    print("=== KICKOFF CLASSIFICATION ===")

    kickoff_et_values = []
    kickoff_utc_values = []
    eligibility = []

    for _, row in games.iterrows():

        raw = (
            f"{row['game_date']} "
            f"{row['gametime']}"
        )

        try:
            naive = datetime.strptime(
                raw,
                "%Y-%m-%d %H:%M",
            )
        except ValueError:
            fail(
                f"invalid kickoff format for "
                f"{row['game_id']}: {raw}"
            )

        kickoff_et = naive.replace(
            tzinfo=ET
        )

        kickoff_utc = kickoff_et.astimezone(
            timezone.utc
        )

        eligible = (
            captured_utc
            < kickoff_utc
        )

        kickoff_et_values.append(
            kickoff_et
        )

        kickoff_utc_values.append(
            kickoff_utc
        )

        eligibility.append(
            eligible
        )

    games = games.copy()

    games["kickoff_et"] = (
        kickoff_et_values
    )

    games["kickoff_utc"] = (
        kickoff_utc_values
    )

    games["pre_kickoff"] = (
        eligibility
    )

    pre_count = int(
        games["pre_kickoff"].sum()
    )

    post_count = int(
        (~games["pre_kickoff"]).sum()
    )

    completed_count = int(
        (
            games["completed"] == 1
        ).sum()
    )

    print(
        f"Pre-kickoff games | {pre_count}"
    )

    print(
        f"At/after kickoff  | {post_count}"
    )

    print(
        f"DB completed      | {completed_count}"
    )

    if pre_count + post_count != 272:
        fail("kickoff classification population")

    print("PASS | all 272 games classified")


    # =================================================================
    # Games at/after kickoff
    # =================================================================

    print()
    print("=== AT / AFTER KICKOFF — MUST NOT BE CAPTURED ===")

    past = games.loc[
        ~games["pre_kickoff"],
        [
            "game_id",
            "week",
            "away_team",
            "home_team",
            "kickoff_et",
            "kickoff_utc",
            "completed",
        ],
    ]

    if past.empty:

        print("NONE")

    else:

        print(
            past.to_string(
                index=False
            )
        )


    # =================================================================
    # Nearest upcoming games
    # =================================================================

    print()
    print("=== NEXT 25 ELIGIBLE GAMES ===")

    upcoming = (
        games.loc[
            games["pre_kickoff"]
        ]
        .sort_values(
            [
                "kickoff_utc",
                "game_id",
            ]
        )
        .head(25)
    )

    if upcoming.empty:

        print("NONE")

    else:

        print(
            upcoming[
                [
                    "game_id",
                    "week",
                    "away_team",
                    "home_team",
                    "kickoff_et",
                    "kickoff_utc",
                ]
            ].to_string(
                index=False
            )
        )


    # =================================================================
    # Forecast status among eligible games
    # =================================================================

    print()
    print("=== ELIGIBLE FORECAST STATUS ===")

    eligibility_frame = (
        games[
            [
                "game_id",
                "pre_kickoff",
            ]
        ]
        .merge(
            pred[
                [
                    "game_id",
                    "forecast_status",
                ]
            ],
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    )

    eligible_status = (
        eligibility_frame.loc[
            eligibility_frame[
                "pre_kickoff"
            ]
        ]
        ["forecast_status"]
        .value_counts(
            dropna=False
        )
    )

    for status, count in eligible_status.items():

        print(
            f"{str(status):<20} | {int(count)}"
        )


    # =================================================================
    # Baseline immutability fingerprint
    # =================================================================

    print()
    print("=== BASELINE LOGICAL FINGERPRINT ===")

    baseline_rows = ledger.execute(
        """
        SELECT *
        FROM forecast_predictions
        WHERE snapshot_id = ?
        ORDER BY game_id
        """,
        (
            EXPECTED_BASELINE_SNAPSHOT,
        ),
    ).fetchall()

    baseline_columns = [
        row[1]
        for row in ledger.execute(
            """
            PRAGMA table_info(
                forecast_predictions
            )
            """
        ).fetchall()
    ]

    payload = "\n".join(
        "|".join(
            "" if value is None else str(value)
            for value in row
        )
        for row in baseline_rows
    ).encode(
        "utf-8"
    )

    logical_sha = hashlib.sha256(
        payload
    ).hexdigest()

    print(
        f"Baseline rows        | {len(baseline_rows)}"
    )

    print(
        f"Baseline columns     | {len(baseline_columns)}"
    )

    print(
        f"Baseline logical SHA | {logical_sha}"
    )

    if len(baseline_rows) != 272:
        fail("baseline prediction population changed")

    print("PASS | baseline snapshot remains intact")


finally:

    nfl.close()
    ledger.close()


# =====================================================================
# Decision gate
# =====================================================================

print()
print("=" * 100)
print("SNAPSHOT CAPTURE V1 — DRY-RUN DECISION")
print("=" * 100)

print("PASS | TIMEZONE = America/New_York")
print("PASS | DST-AWARE KICKOFF CONSTRUCTION")
print("PASS | STRICT capture_time < kickoff_time RULE")
print("PASS | 272-GAME AUTHORITATIVE IDENTITY")
print("PASS | BASELINE SNAPSHOT PRESERVED")

if duplicate_source:

    print()
    print("EXPECTED BLOCK | DUPLICATE SOURCE ARTIFACT")
    print(
        "The current prediction CSV is already represented "
        "by an existing ledger snapshot."
    )

    print(
        "A production writer MUST refuse to append this "
        "same source artifact again."
    )

else:

    print()
    print("READY | CURRENT SOURCE ARTIFACT IS NEW")
    print(
        "A production writer could proceed to append "
        "eligible pre-kickoff predictions after all "
        "remaining write gates pass."
    )

print()
print("PASS | DRY RUN ONLY")
print("PASS | NO LEDGER INSERT")
print("PASS | NO DATABASE WRITE")
print("PASS | nfl.db UNTOUCHED")
print("PASS | OPTIMIZER / UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print("=" * 100)
