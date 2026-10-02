from pathlib import Path
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

NFL_DB = (
    ROOT
    / "data"
    / "nfl.db"
)

FINAL_DB = (
    ROOT
    / "data"
    / "forecast_ledger.db"
)

PRED = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_predictions_frozen"
    / "forecast_live_core_v1_predictions.csv"
)

AUDIT = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_predictions_frozen"
    / "forecast_live_core_v1_predictions_audit.json"
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


def sql_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE forecast_snapshots (
    snapshot_id TEXT PRIMARY KEY,

    captured_at_utc TEXT NOT NULL,

    season INTEGER NOT NULL,

    model TEXT NOT NULL,
    forecast_variant TEXT NOT NULL,

    source_prediction_sha256 TEXT NOT NULL,
    source_audit_sha256 TEXT NOT NULL,

    historical_proof_status TEXT NOT NULL,
    injury_source_status TEXT NOT NULL,
    win_probability_status TEXT NOT NULL,

    snapshot_status TEXT NOT NULL,

    CHECK (
        length(snapshot_id) > 0
    ),

    CHECK (
        length(captured_at_utc) > 0
    ),

    CHECK (
        season >= 2000
    ),

    CHECK (
        length(source_prediction_sha256) = 64
    ),

    CHECK (
        length(source_audit_sha256) = 64
    )
);


CREATE TABLE forecast_predictions (
    snapshot_id TEXT NOT NULL,
    game_id TEXT NOT NULL,

    season INTEGER NOT NULL,
    week INTEGER,

    game_date TEXT,
    gametime TEXT,

    away_team TEXT NOT NULL,
    home_team TEXT NOT NULL,

    market_home_spread_raw REAL,
    market_total REAL,

    forecast_market_ready INTEGER NOT NULL,

    pred_margin_residual REAL,
    pred_total_residual REAL,

    pred_home_margin REAL,
    pred_total_points REAL,

    pred_home_points REAL,
    pred_away_points REAL,

    pred_winner TEXT,

    model TEXT NOT NULL,
    forecast_variant TEXT NOT NULL,

    impact_status TEXT NOT NULL,
    replacement_status TEXT NOT NULL,
    win_probability_status TEXT NOT NULL,
    forecast_status TEXT NOT NULL,

    PRIMARY KEY (
        snapshot_id,
        game_id
    ),

    FOREIGN KEY (
        snapshot_id
    )
    REFERENCES forecast_snapshots (
        snapshot_id
    )
    ON UPDATE RESTRICT
    ON DELETE RESTRICT,

    CHECK (
        forecast_market_ready IN (0, 1)
    ),

    CHECK (
        season >= 2000
    ),

    CHECK (
        week IS NULL
        OR (
            week >= 1
            AND week <= 25
        )
    ),

    CHECK (
        (
            forecast_status = 'MARKET_PENDING'
            AND pred_margin_residual IS NULL
            AND pred_total_residual IS NULL
            AND pred_home_margin IS NULL
            AND pred_total_points IS NULL
            AND pred_home_points IS NULL
            AND pred_away_points IS NULL
            AND pred_winner IS NULL
        )
        OR
        (
            forecast_status <> 'MARKET_PENDING'
        )
    )
);


CREATE INDEX idx_forecast_predictions_game
ON forecast_predictions (
    game_id
);


CREATE INDEX idx_forecast_predictions_season_week
ON forecast_predictions (
    season,
    week
);


CREATE INDEX idx_forecast_predictions_status
ON forecast_predictions (
    forecast_status
);


CREATE INDEX idx_forecast_snapshots_capture_time
ON forecast_snapshots (
    captured_at_utc
);


CREATE TRIGGER forecast_snapshots_no_update
BEFORE UPDATE ON forecast_snapshots
BEGIN
    SELECT RAISE(
        ABORT,
        'forecast_snapshots is append-only: UPDATE forbidden'
    );
END;


CREATE TRIGGER forecast_snapshots_no_delete
BEFORE DELETE ON forecast_snapshots
BEGIN
    SELECT RAISE(
        ABORT,
        'forecast_snapshots is append-only: DELETE forbidden'
    );
END;


CREATE TRIGGER forecast_predictions_no_update
BEFORE UPDATE ON forecast_predictions
BEGIN
    SELECT RAISE(
        ABORT,
        'forecast_predictions is append-only: UPDATE forbidden'
    );
END;


CREATE TRIGGER forecast_predictions_no_delete
BEFORE DELETE ON forecast_predictions
BEGIN
    SELECT RAISE(
        ABORT,
        'forecast_predictions is append-only: DELETE forbidden'
    );
END;
"""


INSERT_PREDICTION = """
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
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""


print("=" * 100)
print("WFS FORECAST CENTER — CREATE PHYSICAL PROSPECTIVE LEDGER V1")
print("=" * 100)


# =====================================================================
# Hard safety gate
# =====================================================================

if FINAL_DB.exists():
    raise SystemExit(
        f"FAIL | ledger already exists: {FINAL_DB}"
    )

if not NFL_DB.is_file():
    raise SystemExit(
        "FAIL | nfl.db missing"
    )

if not PRED.is_file():
    raise SystemExit(
        "FAIL | frozen prediction artifact missing"
    )

if not AUDIT.is_file():
    raise SystemExit(
        "FAIL | frozen prediction audit missing"
    )

print("PASS | physical ledger does not already exist")


# =====================================================================
# Frozen source integrity
# =====================================================================

print()
print("=== FROZEN SOURCE INTEGRITY ===")

pred_sha = sha256(PRED)
audit_sha = sha256(AUDIT)

print(f"Prediction SHA | {pred_sha}")
print(f"Audit SHA      | {audit_sha}")

if pred_sha != EXPECTED_PRED_SHA:
    raise SystemExit(
        "FAIL | frozen prediction SHA mismatch"
    )

if audit_sha != EXPECTED_AUDIT_SHA:
    raise SystemExit(
        "FAIL | frozen audit SHA mismatch"
    )

print("PASS | exact validated frozen forecast inputs")


# =====================================================================
# Load forecast
# =====================================================================

forecast = pd.read_csv(
    PRED
)

if len(forecast) != 272:
    raise SystemExit(
        "FAIL | frozen forecast row count != 272"
    )

if forecast["game_id"].nunique() != 272:
    raise SystemExit(
        "FAIL | frozen forecast game IDs not unique"
    )

ready = (
    forecast["forecast_status"]
    == "READY_CORE_ONLY"
)

pending = (
    forecast["forecast_status"]
    == "MARKET_PENDING"
)

if int(ready.sum()) != 100:
    raise SystemExit(
        "FAIL | ready count != 100"
    )

if int(pending.sum()) != 172:
    raise SystemExit(
        "FAIL | pending count != 172"
    )

print()
print("=== FROZEN FORECAST CONTRACT ===")
print(f"Rows    | {len(forecast)}")
print(f"Games   | {forecast['game_id'].nunique()}")
print(f"Ready   | {int(ready.sum())}")
print(f"Pending | {int(pending.sum())}")
print("PASS | frozen forecast contract")


# =====================================================================
# Read authoritative 2026 schedule from nfl.db READ-ONLY
# =====================================================================

print()
print("=== AUTHORITATIVE GAME ENRICHMENT ===")

nfl_uri = (
    f"file:{NFL_DB}?mode=ro"
)

src = sqlite3.connect(
    nfl_uri,
    uri=True,
)

integrity = src.execute(
    "PRAGMA integrity_check"
).fetchone()[0]

if integrity != "ok":
    src.close()

    raise SystemExit(
        "FAIL | nfl.db integrity_check"
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
    ORDER BY week, game_date, game_id
    """,
    src,
)

src.close()

if len(games) != 272:
    raise SystemExit(
        "FAIL | authoritative 2026 games != 272"
    )

if games["game_id"].nunique() != 272:
    raise SystemExit(
        "FAIL | authoritative 2026 game IDs not unique"
    )

joined = forecast.merge(
    games,
    on="game_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=(
        "_forecast",
        "_db",
    ),
)

if not (
    joined["_merge"] == "both"
).all():
    raise SystemExit(
        "FAIL | forecast/game coverage mismatch"
    )

identity_mismatches = 0

for field in [
    "season",
    "week",
    "game_date",
    "away_team",
    "home_team",
]:

    left = (
        joined[
            f"{field}_forecast"
        ]
        .fillna("<NA>")
        .astype(str)
    )

    right = (
        joined[
            f"{field}_db"
        ]
        .fillna("<NA>")
        .astype(str)
    )

    mismatches = int(
        (left != right).sum()
    )

    print(
        f"{field:<12} mismatches | {mismatches}"
    )

    identity_mismatches += mismatches

if identity_mismatches != 0:
    raise SystemExit(
        "FAIL | forecast/game identity mismatch"
    )

if joined["gametime"].notna().sum() != 272:
    raise SystemExit(
        "FAIL | authoritative gametime incomplete"
    )

print(
    "Gametime populated | "
    f"{int(joined['gametime'].notna().sum())}/272"
)

print("PASS | exact game_id enrichment from nfl.db")
print("PASS | nfl.db read-only")


# =====================================================================
# Build deterministic source frame for ledger
# =====================================================================

ledger_rows = pd.DataFrame(
    {
        "game_id":
            joined["game_id"],

        "season":
            joined["season_forecast"],

        "week":
            joined["week_forecast"],

        "game_date":
            joined["game_date_forecast"],

        "gametime":
            joined["gametime"],

        "away_team":
            joined["away_team_forecast"],

        "home_team":
            joined["home_team_forecast"],

        "market_home_spread_raw":
            joined["market_home_spread_raw"],

        "market_total":
            joined["market_total"],

        "forecast_market_ready":
            joined["forecast_market_ready"],

        "pred_margin_residual":
            joined["pred_margin_residual"],

        "pred_total_residual":
            joined["pred_total_residual"],

        "pred_home_margin":
            joined["pred_home_margin"],

        "pred_total_points":
            joined["pred_total_points"],

        "pred_home_points":
            joined["pred_home_points"],

        "pred_away_points":
            joined["pred_away_points"],

        "pred_winner":
            joined["pred_winner"],

        "model":
            joined["model"],

        "forecast_variant":
            joined["forecast_variant"],

        "impact_status":
            joined["impact_status"],

        "replacement_status":
            joined["replacement_status"],

        "win_probability_status":
            joined["win_probability_status"],

        "forecast_status":
            joined["forecast_status"],
    }
)

ledger_rows = ledger_rows.sort_values(
    [
        "week",
        "game_date",
        "game_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)


# =====================================================================
# Snapshot identity
# =====================================================================

captured_at_utc = (
    datetime.now(
        timezone.utc
    )
    .replace(
        microsecond=0
    )
    .isoformat()
)

snapshot_id = (
    "2026-core-v1-"
    + datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    + "-"
    + uuid.uuid4().hex[:12]
)

print()
print("=== SNAPSHOT IDENTITY ===")
print(f"Snapshot ID     | {snapshot_id}")
print(f"Captured UTC    | {captured_at_utc}")
print("Season          | 2026")
print("Model           | WFS_CORE_RESIDUAL")
print("Variant         | LIVE_CORE_ONLY_V1")


# =====================================================================
# Create TEMP database first
# =====================================================================

temp_db = FINAL_DB.with_name(
    FINAL_DB.name
    + f".tmp.{os.getpid()}"
)

if temp_db.exists():
    temp_db.unlink()

print()
print("=== CREATE TEMP LEDGER ===")
print(f"Temp | {temp_db}")

conn = sqlite3.connect(
    temp_db
)

try:

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.executescript(
        SCHEMA
    )

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
            2026,
            "WFS_CORE_RESIDUAL",
            "LIVE_CORE_ONLY_V1",
            pred_sha,
            audit_sha,
            "PASS",
            "INJURY_SOURCE_PENDING",
            "NOT_CALIBRATED",
            "VALIDATED_PROSPECTIVE_BASELINE",
        ),
    )

    for _, row in ledger_rows.iterrows():

        conn.execute(
            INSERT_PREDICTION,
            (
                snapshot_id,

                str(
                    row["game_id"]
                ),

                int(
                    sql_value(
                        row["season"]
                    )
                ),

                sql_value(
                    row["week"]
                ),

                sql_value(
                    row["game_date"]
                ),

                sql_value(
                    row["gametime"]
                ),

                str(
                    row["away_team"]
                ),

                str(
                    row["home_team"]
                ),

                sql_value(
                    row[
                        "market_home_spread_raw"
                    ]
                ),

                sql_value(
                    row[
                        "market_total"
                    ]
                ),

                int(
                    bool(
                        sql_value(
                            row[
                                "forecast_market_ready"
                            ]
                        )
                    )
                ),

                sql_value(
                    row[
                        "pred_margin_residual"
                    ]
                ),

                sql_value(
                    row[
                        "pred_total_residual"
                    ]
                ),

                sql_value(
                    row[
                        "pred_home_margin"
                    ]
                ),

                sql_value(
                    row[
                        "pred_total_points"
                    ]
                ),

                sql_value(
                    row[
                        "pred_home_points"
                    ]
                ),

                sql_value(
                    row[
                        "pred_away_points"
                    ]
                ),

                sql_value(
                    row[
                        "pred_winner"
                    ]
                ),

                str(
                    row["model"]
                ),

                str(
                    row[
                        "forecast_variant"
                    ]
                ),

                str(
                    row["impact_status"]
                ),

                str(
                    row[
                        "replacement_status"
                    ]
                ),

                str(
                    row[
                        "win_probability_status"
                    ]
                ),

                str(
                    row["forecast_status"]
                ),
            ),
        )

    conn.commit()


    # ================================================================
    # Physical DB audit before promotion
    # ================================================================

    print()
    print("=== TEMP LEDGER AUDIT ===")

    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(
        f"integrity_check | {integrity}"
    )

    if integrity != "ok":
        raise RuntimeError(
            "temp ledger integrity failure"
        )


    foreign_keys = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print(
        "foreign_key_check rows | "
        f"{len(foreign_keys)}"
    )

    if foreign_keys:
        raise RuntimeError(
            "temp ledger foreign-key violations"
        )


    snapshot_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_snapshots
        """
    ).fetchone()[0]


    prediction_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        """
    ).fetchone()[0]


    unique_games = conn.execute(
        """
        SELECT COUNT(
            DISTINCT game_id
        )
        FROM forecast_predictions
        """
    ).fetchone()[0]


    ready_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status =
              'READY_CORE_ONLY'
        """
    ).fetchone()[0]


    pending_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status =
              'MARKET_PENDING'
        """
    ).fetchone()[0]


    gametime_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE gametime IS NOT NULL
        """
    ).fetchone()[0]


    print(
        f"Snapshots        | {snapshot_count}"
    )

    print(
        f"Predictions      | {prediction_count}"
    )

    print(
        f"Unique games     | {unique_games}"
    )

    print(
        f"Ready            | {ready_count}"
    )

    print(
        f"Market pending   | {pending_count}"
    )

    print(
        f"Gametime populated | {gametime_count}"
    )


    if snapshot_count != 1:
        raise RuntimeError(
            "snapshot count != 1"
        )

    if prediction_count != 272:
        raise RuntimeError(
            "prediction count != 272"
        )

    if unique_games != 272:
        raise RuntimeError(
            "unique games != 272"
        )

    if ready_count != 100:
        raise RuntimeError(
            "ready count != 100"
        )

    if pending_count != 172:
        raise RuntimeError(
            "pending count != 172"
        )

    if gametime_count != 272:
        raise RuntimeError(
            "gametime count != 272"
        )


    # ================================================================
    # Verify append-only enforcement in physical temp DB
    # ================================================================

    print()
    print("=== APPEND-ONLY PHYSICAL TEST ===")


    try:

        conn.execute(
            """
            UPDATE forecast_snapshots
            SET snapshot_status = 'MUTATED'
            WHERE snapshot_id = ?
            """,
            (
                snapshot_id,
            ),
        )

        conn.commit()

        raise RuntimeError(
            "snapshot UPDATE was permitted"
        )

    except sqlite3.IntegrityError:

        conn.rollback()

        print(
            "PASS | snapshot UPDATE blocked"
        )


    first_game = conn.execute(
        """
        SELECT game_id
        FROM forecast_predictions
        ORDER BY game_id
        LIMIT 1
        """
    ).fetchone()[0]


    try:

        conn.execute(
            """
            UPDATE forecast_predictions
            SET pred_total_points = 999
            WHERE snapshot_id = ?
              AND game_id = ?
            """,
            (
                snapshot_id,
                first_game,
            ),
        )

        conn.commit()

        raise RuntimeError(
            "prediction UPDATE was permitted"
        )

    except sqlite3.IntegrityError:

        conn.rollback()

        print(
            "PASS | prediction UPDATE blocked"
        )


    # ================================================================
    # Final temp DB state
    # ================================================================

    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if integrity != "ok":
        raise RuntimeError(
            "post-test temp ledger integrity failure"
        )

    print("PASS | temp ledger fully validated")

finally:

    conn.close()


# =====================================================================
# Atomic promotion
# =====================================================================

if FINAL_DB.exists():

    if temp_db.exists():
        temp_db.unlink()

    raise SystemExit(
        "FAIL | final ledger appeared before promotion"
    )

os.replace(
    temp_db,
    FINAL_DB
)

print()
print("=== ATOMIC PROMOTION ===")
print(f"PASS | created {FINAL_DB}")


# =====================================================================
# Reopen FINAL database read-only
# =====================================================================

final_uri = (
    f"file:{FINAL_DB}?mode=ro"
)

check = sqlite3.connect(
    final_uri,
    uri=True,
)

try:

    integrity = check.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    snapshot_count = check.execute(
        """
        SELECT COUNT(*)
        FROM forecast_snapshots
        """
    ).fetchone()[0]

    prediction_count = check.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        """
    ).fetchone()[0]

    ready_count = check.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status =
              'READY_CORE_ONLY'
        """
    ).fetchone()[0]

    pending_count = check.execute(
        """
        SELECT COUNT(*)
        FROM forecast_predictions
        WHERE forecast_status =
              'MARKET_PENDING'
        """
    ).fetchone()[0]

finally:

    check.close()


if integrity != "ok":
    raise SystemExit(
        "FAIL | final ledger integrity"
    )

if snapshot_count != 1:
    raise SystemExit(
        "FAIL | final snapshot count"
    )

if prediction_count != 272:
    raise SystemExit(
        "FAIL | final prediction count"
    )

if ready_count != 100:
    raise SystemExit(
        "FAIL | final ready count"
    )

if pending_count != 172:
    raise SystemExit(
        "FAIL | final pending count"
    )


# =====================================================================
# SHA
# =====================================================================

ledger_sha = sha256(
    FINAL_DB
)

print()
print("=== FINAL LEDGER ===")
print(f"Path       | {FINAL_DB}")
print(f"SHA256     | {ledger_sha}")
print(f"Snapshot   | {snapshot_id}")
print(f"Captured   | {captured_at_utc}")
print(f"Snapshots  | {snapshot_count}")
print(f"Predictions| {prediction_count}")
print(f"Ready      | {ready_count}")
print(f"Pending    | {pending_count}")


# =====================================================================
# Final gate
# =====================================================================

print()
print("=" * 100)
print("PROSPECTIVE FORECAST LEDGER V1 CREATION GATE")
print("=" * 100)

print("PASS | PHYSICAL FORECAST LEDGER CREATED")
print("PASS | SEPARATE DATABASE")
print("PASS | SQLITE INTEGRITY")
print("PASS | FOREIGN KEY INTEGRITY")
print("PASS | 1 IMMUTABLE SNAPSHOT")
print("PASS | 272 IMMUTABLE GAME RECORDS")
print("PASS | 100 READY / 172 MARKET-PENDING")
print("PASS | 272/272 GAMETIMES ENRICHED")
print("PASS | EXACT GAME_ID IDENTITY")
print("PASS | SNAPSHOT UPDATE BLOCKED")
print("PASS | PREDICTION UPDATE BLOCKED")
print("PASS | FROZEN SOURCE HASHES VERIFIED")
print()
print("PASS | PROSPECTIVE FORECAST LEDGER V1 CREATED")
print()
print("PASS | nfl.db READ-ONLY / UNTOUCHED")
print("PASS | OPTIMIZER UNTOUCHED")
print("PASS | UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print("=" * 100)
