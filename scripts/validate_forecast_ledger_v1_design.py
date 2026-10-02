from pathlib import Path
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

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
    """
    Convert pandas / NumPy values to native Python scalar types
    before passing them to sqlite3.

    NaN / NA -> None
    np.generic -> native Python scalar
    everything else -> unchanged
    """

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

    CHECK (length(snapshot_id) > 0),
    CHECK (length(captured_at_utc) > 0),

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


def prediction_values(snapshot_id, row):
    return (
        snapshot_id,

        str(row["game_id"]),
        int(sql_value(row["season"])),

        sql_value(row["week"]),
        sql_value(row["game_date"]),

        # Frozen prediction artifact did not contain gametime.
        # Physical capture writer will enrich this later from
        # authoritative games table by exact game_id.
        None,

        str(row["away_team"]),
        str(row["home_team"]),

        sql_value(
            row["market_home_spread_raw"]
        ),

        sql_value(
            row["market_total"]
        ),

        int(
            bool(
                sql_value(
                    row["forecast_market_ready"]
                )
            )
        ),

        sql_value(
            row["pred_margin_residual"]
        ),

        sql_value(
            row["pred_total_residual"]
        ),

        sql_value(
            row["pred_home_margin"]
        ),

        sql_value(
            row["pred_total_points"]
        ),

        sql_value(
            row["pred_home_points"]
        ),

        sql_value(
            row["pred_away_points"]
        ),

        sql_value(
            row["pred_winner"]
        ),

        str(row["model"]),
        str(row["forecast_variant"]),
        str(row["impact_status"]),
        str(row["replacement_status"]),
        str(row["win_probability_status"]),
        str(row["forecast_status"]),
    )


def insert_snapshot(
    conn,
    snapshot_id,
    captured_at_utc,
    pred_sha,
    audit_sha,
    status,
):
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
            status,
        ),
    )


print("=" * 100)
print("WFS FORECAST CENTER — PROSPECTIVE LEDGER V1 DESIGN VALIDATION V2")
print("=" * 100)


# =====================================================================
# Frozen artifact integrity
# =====================================================================

print()
print("=== FROZEN INPUT INTEGRITY ===")

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

print("PASS | frozen prediction inputs intact")


# =====================================================================
# Input contract
# =====================================================================

df = pd.read_csv(PRED)

if len(df) != 272:
    raise SystemExit(
        "FAIL | expected 272 frozen prediction rows"
    )

if df["game_id"].nunique() != 272:
    raise SystemExit(
        "FAIL | expected 272 unique game IDs"
    )

ready = (
    df["forecast_status"]
    == "READY_CORE_ONLY"
)

pending = (
    df["forecast_status"]
    == "MARKET_PENDING"
)

print()
print("=== INPUT CONTRACT ===")
print(f"Rows    | {len(df)}")
print(f"Games   | {df['game_id'].nunique()}")
print(f"Ready   | {int(ready.sum())}")
print(f"Pending | {int(pending.sum())}")

if int(ready.sum()) != 100:
    raise SystemExit(
        "FAIL | expected 100 ready games"
    )

if int(pending.sum()) != 172:
    raise SystemExit(
        "FAIL | expected 172 pending games"
    )

required = [
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

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:
    raise SystemExit(
        "FAIL | missing prediction columns: "
        + ", ".join(missing)
    )

print("PASS | frozen input contract")


# =====================================================================
# Explicit scalar conversion proof
# =====================================================================

print()
print("=== SQLITE SCALAR NORMALIZATION ===")

sample = df.iloc[0]

week_raw = sample["week"]
week_sql = sql_value(week_raw)

print(
    "Raw week type  | "
    f"{type(week_raw).__name__}"
)

print(
    "SQL week type  | "
    f"{type(week_sql).__name__}"
)

if week_sql is not None:

    if not isinstance(
        week_sql,
        int,
    ):
        raise SystemExit(
            "FAIL | week not converted to native Python int"
        )

    if not (
        1 <= week_sql <= 25
    ):
        raise SystemExit(
            "FAIL | normalized week outside valid range"
        )

print("PASS | NumPy/pandas scalars normalized before SQLite binding")


# =====================================================================
# Build database in memory only
# =====================================================================

print()
print("=== BUILD IN-MEMORY LEDGER ===")

conn = sqlite3.connect(
    ":memory:"
)

conn.execute(
    "PRAGMA foreign_keys = ON"
)

conn.executescript(
    SCHEMA
)

print("PASS | proposed schema created in memory")


# =====================================================================
# Schema objects
# =====================================================================

tables = {
    row[0]
    for row in conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    )
}

expected_tables = {
    "forecast_snapshots",
    "forecast_predictions",
}

if not expected_tables.issubset(
    tables
):
    raise SystemExit(
        "FAIL | proposed tables missing"
    )

triggers = {
    row[0]
    for row in conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'trigger'
        """
    )
}

expected_triggers = {
    "forecast_snapshots_no_update",
    "forecast_snapshots_no_delete",
    "forecast_predictions_no_update",
    "forecast_predictions_no_delete",
}

if triggers != expected_triggers:
    raise SystemExit(
        "FAIL | append-only trigger contract"
    )

print("PASS | required tables exist")
print("PASS | four append-only triggers exist")


# =====================================================================
# Insert first full snapshot
# =====================================================================

snapshot_id_1 = (
    "test-"
    + str(
        uuid.uuid4()
    )
)

captured_at_utc = (
    datetime.now(
        timezone.utc
    )
    .replace(
        microsecond=0
    )
    .isoformat()
)

insert_snapshot(
    conn,
    snapshot_id_1,
    captured_at_utc,
    pred_sha,
    audit_sha,
    "VALIDATED_PROSPECTIVE_BASELINE",
)

for _, row in df.iterrows():

    conn.execute(
        INSERT_PREDICTION,
        prediction_values(
            snapshot_id_1,
            row,
        ),
    )

conn.commit()

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

print()
print("=== FIRST MOCK SNAPSHOT ===")
print(f"Snapshots   | {snapshot_count}")
print(f"Predictions | {prediction_count}")

if snapshot_count != 1:
    raise SystemExit(
        "FAIL | expected one snapshot"
    )

if prediction_count != 272:
    raise SystemExit(
        "FAIL | expected 272 predictions"
    )

print("PASS | complete 272-game snapshot inserted")


# =====================================================================
# Status preservation
# =====================================================================

status_rows = conn.execute(
    """
    SELECT
        forecast_status,
        COUNT(*)
    FROM forecast_predictions
    GROUP BY forecast_status
    ORDER BY forecast_status
    """
).fetchall()

status_map = dict(
    status_rows
)

print()
print("=== STATUS DISTRIBUTION ===")

for status, count in status_rows:
    print(
        f"{status:<20} | {count}"
    )

if status_map.get(
    "READY_CORE_ONLY"
) != 100:
    raise SystemExit(
        "FAIL | ready status count"
    )

if status_map.get(
    "MARKET_PENDING"
) != 172:
    raise SystemExit(
        "FAIL | pending status count"
    )

print("PASS | 100 ready / 172 pending preserved")


# =====================================================================
# Duplicate protection
# =====================================================================

print()
print("=== DUPLICATE PROTECTION ===")

try:

    conn.execute(
        INSERT_PREDICTION,
        prediction_values(
            snapshot_id_1,
            sample,
        ),
    )

    conn.commit()

    raise SystemExit(
        "FAIL | duplicate game accepted within snapshot"
    )

except sqlite3.IntegrityError:

    conn.rollback()

print("PASS | duplicate game blocked within same snapshot")


# =====================================================================
# Same game in second snapshot
# =====================================================================

print()
print("=== MULTIPLE SNAPSHOT CONTRACT ===")

snapshot_id_2 = (
    "test-"
    + str(
        uuid.uuid4()
    )
)

insert_snapshot(
    conn,
    snapshot_id_2,
    captured_at_utc,
    pred_sha,
    audit_sha,
    "TEST_SECOND_SNAPSHOT",
)

conn.execute(
    INSERT_PREDICTION,
    prediction_values(
        snapshot_id_2,
        sample,
    ),
)

conn.commit()

same_game_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM forecast_predictions
    WHERE game_id = ?
    """,
    (
        str(
            sample["game_id"]
        ),
    ),
).fetchone()[0]

print(
    f"Same game across snapshots | {same_game_count}"
)

if same_game_count != 2:
    raise SystemExit(
        "FAIL | same game not preserved across snapshots"
    )

print("PASS | same game allowed in multiple immutable snapshots")


# =====================================================================
# Append-only enforcement
# =====================================================================

print()
print("=== APPEND-ONLY ENFORCEMENT ===")


def must_block(
    sql,
    params,
    label,
):

    try:

        conn.execute(
            sql,
            params,
        )

        conn.commit()

    except sqlite3.IntegrityError as exc:

        conn.rollback()

        print(
            f"PASS | {label} blocked"
        )

        print(
            f"       SQLite: {exc}"
        )

        return

    raise SystemExit(
        f"FAIL | {label} was permitted"
    )


must_block(
    """
    UPDATE forecast_snapshots
    SET snapshot_status = ?
    WHERE snapshot_id = ?
    """,
    (
        "MUTATED",
        snapshot_id_1,
    ),
    "snapshot UPDATE",
)


must_block(
    """
    DELETE FROM forecast_snapshots
    WHERE snapshot_id = ?
    """,
    (
        snapshot_id_1,
    ),
    "snapshot DELETE",
)


must_block(
    """
    UPDATE forecast_predictions
    SET pred_total_points = ?
    WHERE snapshot_id = ?
      AND game_id = ?
    """,
    (
        999.0,
        snapshot_id_1,
        str(
            sample["game_id"]
        ),
    ),
    "prediction UPDATE",
)


must_block(
    """
    DELETE FROM forecast_predictions
    WHERE snapshot_id = ?
      AND game_id = ?
    """,
    (
        snapshot_id_1,
        str(
            sample["game_id"]
        ),
    ),
    "prediction DELETE",
)


# =====================================================================
# Foreign-key enforcement
# =====================================================================

print()
print("=== FOREIGN KEY CONTRACT ===")

fake = sample.copy()

try:

    conn.execute(
        INSERT_PREDICTION,
        prediction_values(
            "snapshot-does-not-exist",
            fake,
        ),
    )

    conn.commit()

    raise SystemExit(
        "FAIL | orphan prediction accepted"
    )

except sqlite3.IntegrityError:

    conn.rollback()

print("PASS | orphan prediction blocked")


# =====================================================================
# Null contract
# =====================================================================

print()
print("=== MARKET-PENDING NULL CONTRACT ===")

violations = conn.execute(
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
    f"Pending prediction violations | {violations}"
)

if violations != 0:
    raise SystemExit(
        "FAIL | market-pending null contract"
    )

print("PASS | market-pending games remain unpredicted")


# =====================================================================
# Database integrity
# =====================================================================

print()
print("=== SQLITE INTEGRITY ===")

integrity = conn.execute(
    "PRAGMA integrity_check"
).fetchone()[0]

print(
    f"integrity_check | {integrity}"
)

if integrity != "ok":
    raise SystemExit(
        "FAIL | SQLite integrity"
    )

conn.close()

print("PASS | in-memory database integrity")


# =====================================================================
# Final gate
# =====================================================================

print()
print("=" * 100)
print("PROSPECTIVE FORECAST LEDGER V1 DESIGN GATE")
print("=" * 100)

print("PASS | NUMPY/PANDAS SQLITE BINDING NORMALIZED")
print("PASS | SEPARATE FORECAST LEDGER DATABASE DESIGN")
print("PASS | SNAPSHOT / PREDICTION TABLE SEPARATION")
print("PASS | 272-GAME FROZEN SNAPSHOT LOAD")
print("PASS | 100 READY / 172 MARKET-PENDING PRESERVED")
print("PASS | DUPLICATE GAME BLOCKED WITHIN SNAPSHOT")
print("PASS | MULTIPLE SNAPSHOTS PER GAME ALLOWED")
print("PASS | SNAPSHOT UPDATE BLOCKED")
print("PASS | SNAPSHOT DELETE BLOCKED")
print("PASS | PREDICTION UPDATE BLOCKED")
print("PASS | PREDICTION DELETE BLOCKED")
print("PASS | FOREIGN KEY CONTRACT")
print("PASS | MARKET-PENDING NULL CONTRACT")
print("PASS | SQLITE INTEGRITY")
print()
print("PASS | LEDGER V1 DESIGN VALIDATED IN MEMORY")
print("READY | CREATE PHYSICAL FORECAST LEDGER V1")
print()
print("PASS | nfl.db UNTOUCHED")
print("PASS | NO forecast_ledger.db CREATED")
print("PASS | OPTIMIZER / UI UNTOUCHED")
print("=" * 100)
