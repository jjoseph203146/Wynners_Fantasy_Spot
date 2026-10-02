#!/usr/bin/env python3

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/nfl.db"

V1_PARSER_VERSION = "WFS_WEATHER_INTELLIGENCE_V1"
V2_FEATURE_VERSION = "WFS_WEATHER_INTELLIGENCE_V2"
V3_SHADOW_VERSION = "WFS_WEATHER_INTELLIGENCE_V3"

EXPECTED_SOURCE = "NFLWeather"

EXPECTED_GAME_COUNT = 16

PRODUCTION_INFLUENCE_ALLOWED = False


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS weather_game_shadow_v3 (
    shadow_id INTEGER PRIMARY KEY AUTOINCREMENT,

    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    source_v2_feature_id INTEGER NOT NULL,
    source_v2_feature_version TEXT NOT NULL,

    source_snapshot_id INTEGER NOT NULL,
    source_captured_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_parser_version TEXT NOT NULL,

    shadow_version TEXT NOT NULL,
    shadow_created_at TEXT NOT NULL,

    stadium_name TEXT,
    stadium_type TEXT,
    surface TEXT,
    location TEXT,

    kickoff_condition TEXT,
    kickoff_temperature_f REAL,
    kickoff_feels_like_f REAL,

    kickoff_wind_mph REAL,
    wind_max_mph REAL,

    kickoff_wind_direction TEXT,

    kickoff_gust_mph REAL,
    gust_max_mph REAL,

    kickoff_precipitation_probability_pct REAL,
    precipitation_probability_max_pct REAL,

    weather_exposure_state TEXT NOT NULL,
    wind_research_band TEXT NOT NULL,

    shadow_eligible INTEGER NOT NULL,
    shadow_reason TEXT NOT NULL,

    production_influence_allowed INTEGER NOT NULL,

    FOREIGN KEY(source_v2_feature_id)
        REFERENCES weather_game_features_v2(feature_id),

    FOREIGN KEY(source_snapshot_id)
        REFERENCES weather_game_snapshots(snapshot_id),

    UNIQUE(
        source_v2_feature_id,
        shadow_version
    )
)
"""


CREATE_INDEX_SQL = (
    """
    CREATE INDEX IF NOT EXISTS
        idx_weather_game_shadow_v3_game_capture
    ON weather_game_shadow_v3(
        game_id,
        source_captured_at
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_weather_game_shadow_v3_season_week
    ON weather_game_shadow_v3(
        season,
        week,
        source_captured_at
    )
    """,
)


DETERMINISTIC_COLUMNS = [
    "game_id",
    "season",
    "week",

    "source_v2_feature_id",
    "source_v2_feature_version",

    "source_snapshot_id",
    "source_captured_at",
    "source",
    "source_url",
    "source_parser_version",

    "shadow_version",

    "stadium_name",
    "stadium_type",
    "surface",
    "location",

    "kickoff_condition",
    "kickoff_temperature_f",
    "kickoff_feels_like_f",

    "kickoff_wind_mph",
    "wind_max_mph",
    "kickoff_wind_direction",

    "kickoff_gust_mph",
    "gust_max_mph",

    "kickoff_precipitation_probability_pct",
    "precipitation_probability_max_pct",

    "weather_exposure_state",
    "wind_research_band",

    "shadow_eligible",
    "shadow_reason",

    "production_influence_allowed",
]


INSERT_COLUMNS = [
    *DETERMINISTIC_COLUMNS,
    "shadow_created_at",
]


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def classify_exposure(stadium_type):
    value = (
        stadium_type or ""
    ).strip().lower()

    if value == "open":
        return "OPEN_AIR"

    if value == "dome":
        return "DOME_RAW_SOURCE"

    return "UNKNOWN_RAW_SOURCE"


def classify_wind_band(wind_mph):

    if wind_mph is None:
        return "UNKNOWN"

    wind = float(wind_mph)

    if wind < 0:
        raise RuntimeError(
            f"Negative wind value: {wind}"
        )

    if wind < 5:
        return "0-4"

    if wind < 10:
        return "5-9"

    if wind < 15:
        return "10-14"

    if wind < 20:
        return "15-19"

    return "20+"


def derive_shadow_state(row):

    exposure = classify_exposure(
        row["stadium_type"]
    )

    # Research band uses kickoff wind only.
    #
    # These are descriptive historical research bands.
    # They are NOT production thresholds.
    band = classify_wind_band(
        row["kickoff_wind_mph"]
    )

    if exposure == "OPEN_AIR":

        if row["kickoff_wind_mph"] is None:
            eligible = 0
            reason = "OPEN_AIR_MISSING_KICKOFF_WIND"

        else:
            eligible = 1
            reason = "OPEN_AIR_RESEARCH_ELIGIBLE"

    elif exposure == "DOME_RAW_SOURCE":

        # Keep raw NFLWeather values intact.
        #
        # Do not reinterpret dome weather semantics
        # and do not treat it as outdoor exposure.
        eligible = 0
        reason = "DOME_RAW_SOURCE_NOT_OUTDOOR_RESEARCH"

    else:
        eligible = 0
        reason = "UNKNOWN_EXPOSURE_FAIL_CLOSED"

    return (
        exposure,
        band,
        eligible,
        reason,
    )


def ensure_schema(con):

    con.execute(CREATE_TABLE_SQL)

    for sql in CREATE_INDEX_SQL:
        con.execute(sql)


def validate_parent_schema(con):

    required = {
        row["name"]
        for row in con.execute(
            """
            PRAGMA table_info(
                weather_game_features_v2
            )
            """
        ).fetchall()
    }

    wanted = {
        "feature_id",
        "game_id",
        "season",
        "week",

        "source_snapshot_id",
        "source_captured_at",
        "source",
        "source_url",
        "source_parser_version",

        "feature_version",

        "stadium_name",
        "stadium_type",
        "surface",
        "location",

        "kickoff_condition",
        "kickoff_temperature_f",
        "kickoff_feels_like_f",

        "kickoff_wind_mph",
        "wind_max_mph",
        "kickoff_wind_direction",

        "kickoff_gust_mph",
        "gust_max_mph",

        "kickoff_precipitation_probability_pct",
        "precipitation_probability_max_pct",
    }

    missing = sorted(
        wanted - required
    )

    if missing:
        raise RuntimeError(
            "V2 schema missing required columns: "
            + ", ".join(missing)
        )


def select_source_rows(con):

    captures = con.execute(
        """
        SELECT
            source_captured_at,
            COUNT(*) AS n,
            COUNT(DISTINCT game_id) AS games
        FROM weather_game_features_v2
        WHERE feature_version = ?
          AND source_parser_version = ?
          AND source = ?
        GROUP BY source_captured_at
        HAVING COUNT(*) = ?
           AND COUNT(DISTINCT game_id) = ?
        ORDER BY source_captured_at DESC
        """,
        (
            V2_FEATURE_VERSION,
            V1_PARSER_VERSION,
            EXPECTED_SOURCE,
            EXPECTED_GAME_COUNT,
            EXPECTED_GAME_COUNT,
        ),
    ).fetchall()

    if not captures:
        raise RuntimeError(
            "No complete validated V2 capture found."
        )

    capture = captures[0][
        "source_captured_at"
    ]

    rows = con.execute(
        """
        SELECT
            feature_id,
            game_id,
            season,
            week,

            source_snapshot_id,
            source_captured_at,
            source,
            source_url,
            source_parser_version,

            feature_version,

            stadium_name,
            stadium_type,
            surface,
            location,

            kickoff_condition,
            kickoff_temperature_f,
            kickoff_feels_like_f,

            kickoff_wind_mph,
            wind_max_mph,
            kickoff_wind_direction,

            kickoff_gust_mph,
            gust_max_mph,

            kickoff_precipitation_probability_pct,
            precipitation_probability_max_pct

        FROM weather_game_features_v2

        WHERE source_captured_at = ?
          AND feature_version = ?
          AND source_parser_version = ?
          AND source = ?

        ORDER BY game_id
        """,
        (
            capture,
            V2_FEATURE_VERSION,
            V1_PARSER_VERSION,
            EXPECTED_SOURCE,
        ),
    ).fetchall()

    if len(rows) != EXPECTED_GAME_COUNT:
        raise RuntimeError(
            "Validated V2 source row count changed "
            f"unexpectedly: {len(rows)}"
        )

    if (
        len({
            row["game_id"]
            for row in rows
        })
        != EXPECTED_GAME_COUNT
    ):
        raise RuntimeError(
            "Validated V2 source contains "
            "duplicate game IDs."
        )

    return capture, rows


def build_candidate(row):

    (
        exposure,
        band,
        eligible,
        reason,
    ) = derive_shadow_state(row)

    return {
        "game_id":
            row["game_id"],

        "season":
            row["season"],

        "week":
            row["week"],

        "source_v2_feature_id":
            row["feature_id"],

        "source_v2_feature_version":
            row["feature_version"],

        "source_snapshot_id":
            row["source_snapshot_id"],

        "source_captured_at":
            row["source_captured_at"],

        "source":
            row["source"],

        "source_url":
            row["source_url"],

        "source_parser_version":
            row["source_parser_version"],

        "shadow_version":
            V3_SHADOW_VERSION,

        "stadium_name":
            row["stadium_name"],

        "stadium_type":
            row["stadium_type"],

        "surface":
            row["surface"],

        "location":
            row["location"],

        "kickoff_condition":
            row["kickoff_condition"],

        "kickoff_temperature_f":
            row["kickoff_temperature_f"],

        "kickoff_feels_like_f":
            row["kickoff_feels_like_f"],

        "kickoff_wind_mph":
            row["kickoff_wind_mph"],

        "wind_max_mph":
            row["wind_max_mph"],

        "kickoff_wind_direction":
            row["kickoff_wind_direction"],

        "kickoff_gust_mph":
            row["kickoff_gust_mph"],

        "gust_max_mph":
            row["gust_max_mph"],

        "kickoff_precipitation_probability_pct":
            row[
                "kickoff_precipitation_probability_pct"
            ],

        "precipitation_probability_max_pct":
            row[
                "precipitation_probability_max_pct"
            ],

        "weather_exposure_state":
            exposure,

        "wind_research_band":
            band,

        "shadow_eligible":
            eligible,

        "shadow_reason":
            reason,

        "production_influence_allowed":
            0,
    }


def compare_existing(existing, candidate):

    differences = []

    for column in DETERMINISTIC_COLUMNS:

        old = existing[column]
        new = candidate[column]

        if old != new:
            differences.append(
                (
                    column,
                    old,
                    new,
                )
            )

    return differences


def persist(con, candidates):

    inserted = 0
    unchanged = 0

    placeholders = ", ".join(
        "?"
        for _ in INSERT_COLUMNS
    )

    columns_sql = ", ".join(
        INSERT_COLUMNS
    )

    insert_sql = (
        "INSERT INTO weather_game_shadow_v3 "
        f"({columns_sql}) "
        f"VALUES ({placeholders})"
    )

    for candidate in candidates:

        existing = con.execute(
            """
            SELECT *
            FROM weather_game_shadow_v3
            WHERE source_v2_feature_id = ?
              AND shadow_version = ?
            """,
            (
                candidate[
                    "source_v2_feature_id"
                ],
                candidate[
                    "shadow_version"
                ],
            ),
        ).fetchone()

        if existing is not None:

            differences = compare_existing(
                existing,
                candidate,
            )

            if differences:

                detail = "; ".join(
                    f"{column}: "
                    f"{old!r} -> {new!r}"
                    for column, old, new
                    in differences
                )

                raise RuntimeError(
                    "Immutable V3 lineage conflict for "
                    f"feature_id="
                    f"{candidate['source_v2_feature_id']}: "
                    f"{detail}. "
                    "Create a new shadow version."
                )

            unchanged += 1
            continue

        created_at = utc_now()

        values = []

        for column in INSERT_COLUMNS:

            if column == "shadow_created_at":
                values.append(created_at)
            else:
                values.append(
                    candidate[column]
                )

        con.execute(
            insert_sql,
            values,
        )

        inserted += 1

    return inserted, unchanged


def qa(con, source_capture):

    print("\n=== V3 STRUCTURAL QA ===")

    row = con.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT game_id) AS games,
            COUNT(DISTINCT source_v2_feature_id)
                AS v2_ids,
            COUNT(DISTINCT source_snapshot_id)
                AS snapshot_ids
        FROM weather_game_shadow_v3
        WHERE shadow_version = ?
          AND source_captured_at = ?
        """,
        (
            V3_SHADOW_VERSION,
            source_capture,
        ),
    ).fetchone()

    print(dict(row))

    if row["rows"] != EXPECTED_GAME_COUNT:
        raise RuntimeError(
            "V3 QA row count failure."
        )

    if row["games"] != EXPECTED_GAME_COUNT:
        raise RuntimeError(
            "V3 QA game cardinality failure."
        )

    if row["v2_ids"] != EXPECTED_GAME_COUNT:
        raise RuntimeError(
            "V3 QA V2 lineage cardinality failure."
        )

    if row["snapshot_ids"] != EXPECTED_GAME_COUNT:
        raise RuntimeError(
            "V3 QA V1 lineage cardinality failure."
        )

    duplicates = con.execute(
        """
        SELECT
            source_v2_feature_id,
            shadow_version,
            COUNT(*) AS n
        FROM weather_game_shadow_v3
        GROUP BY
            source_v2_feature_id,
            shadow_version
        HAVING COUNT(*) > 1
        """
    ).fetchall()

    print(
        "DUPLICATE_LINEAGE_KEYS =",
        len(duplicates),
    )

    if duplicates:
        raise RuntimeError(
            "Duplicate V3 lineage keys."
        )

    broken_v2 = con.execute(
        """
        SELECT COUNT(*) AS n
        FROM weather_game_shadow_v3 v3
        LEFT JOIN weather_game_features_v2 v2
          ON v2.feature_id =
             v3.source_v2_feature_id
        WHERE v2.feature_id IS NULL
        """
    ).fetchone()["n"]

    broken_v1 = con.execute(
        """
        SELECT COUNT(*) AS n
        FROM weather_game_shadow_v3 v3
        LEFT JOIN weather_game_snapshots v1
          ON v1.snapshot_id =
             v3.source_snapshot_id
        WHERE v1.snapshot_id IS NULL
        """
    ).fetchone()["n"]

    print(
        "BROKEN_V3_TO_V2_LINEAGE =",
        broken_v2,
    )

    print(
        "BROKEN_V3_TO_V1_LINEAGE =",
        broken_v1,
    )

    if broken_v2 or broken_v1:
        raise RuntimeError(
            "V3 lineage integrity failure."
        )

    production_true = con.execute(
        """
        SELECT COUNT(*)
        FROM weather_game_shadow_v3
        WHERE shadow_version = ?
          AND source_captured_at = ?
          AND production_influence_allowed != 0
        """,
        (
            V3_SHADOW_VERSION,
            source_capture,
        ),
    ).fetchone()[0]

    print(
        "PRODUCTION_INFLUENCE_VIOLATIONS =",
        production_true,
    )

    if production_true:
        raise RuntimeError(
            "V3 production boundary violation."
        )

    print("\n=== EXPOSURE DISTRIBUTION ===")

    for row in con.execute(
        """
        SELECT
            weather_exposure_state,
            shadow_eligible,
            shadow_reason,
            COUNT(*) AS n
        FROM weather_game_shadow_v3
        WHERE shadow_version = ?
          AND source_captured_at = ?
        GROUP BY
            weather_exposure_state,
            shadow_eligible,
            shadow_reason
        ORDER BY
            weather_exposure_state,
            shadow_eligible,
            shadow_reason
        """,
        (
            V3_SHADOW_VERSION,
            source_capture,
        ),
    ).fetchall():

        print(dict(row))

    print("\n=== WIND RESEARCH BANDS ===")

    for row in con.execute(
        """
        SELECT
            wind_research_band,
            COUNT(*) AS n
        FROM weather_game_shadow_v3
        WHERE shadow_version = ?
          AND source_captured_at = ?
        GROUP BY wind_research_band
        ORDER BY
            CASE wind_research_band
                WHEN '0-4' THEN 1
                WHEN '5-9' THEN 2
                WHEN '10-14' THEN 3
                WHEN '15-19' THEN 4
                WHEN '20+' THEN 5
                ELSE 6
            END
        """,
        (
            V3_SHADOW_VERSION,
            source_capture,
        ),
    ).fetchall():

        print(dict(row))

    print("\n=== ROW AUDIT ===")

    for row in con.execute(
        """
        SELECT
            game_id,
            stadium_type,
            kickoff_wind_mph,
            wind_max_mph,
            wind_research_band,
            weather_exposure_state,
            shadow_eligible,
            shadow_reason,
            production_influence_allowed
        FROM weather_game_shadow_v3
        WHERE shadow_version = ?
          AND source_captured_at = ?
        ORDER BY game_id
        """,
        (
            V3_SHADOW_VERSION,
            source_capture,
        ),
    ).fetchall():

        print(dict(row))

    fk_errors = con.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print(
        "\nFOREIGN_KEY_ERRORS =",
        len(fk_errors),
    )

    if fk_errors:
        raise RuntimeError(
            "Database foreign-key check failed."
        )


def main():

    print("=" * 92)
    print(" WEATHER V3 — SHADOW INTELLIGENCE BUILDER")
    print(" ISOLATED / RESEARCH ONLY / NO PRODUCTION INFLUENCE")
    print("=" * 92)

    if PRODUCTION_INFLUENCE_ALLOWED:
        raise RuntimeError(
            "Production influence must remain disabled."
        )

    con = sqlite3.connect(str(DB))

    try:
        con.row_factory = sqlite3.Row

        # Explicitly enable FK enforcement for this connection.
        con.execute(
            "PRAGMA foreign_keys = ON"
        )

        validate_parent_schema(con)

        source_capture, source_rows = (
            select_source_rows(con)
        )

        print(
            "\nSOURCE_CAPTURE =",
            source_capture,
        )

        print(
            "SOURCE_ROWS =",
            len(source_rows),
        )

        print(
            "SOURCE =",
            EXPECTED_SOURCE,
        )

        print(
            "SOURCE_PARSER_VERSION =",
            V1_PARSER_VERSION,
        )

        print(
            "SOURCE_V2_FEATURE_VERSION =",
            V2_FEATURE_VERSION,
        )

        print(
            "V3_SHADOW_VERSION =",
            V3_SHADOW_VERSION,
        )

        candidates = [
            build_candidate(row)
            for row in source_rows
        ]

        # DDL and inserts occur in one transaction.
        #
        # Any failure before commit rolls back the V3 mutation.
        with con:
            ensure_schema(con)

            inserted, unchanged = persist(
                con,
                candidates,
            )

            qa(
                con,
                source_capture,
            )

        print("\n=== PERSISTENCE ===")
        print(
            "INSERTED =",
            inserted,
        )
        print(
            "UNCHANGED =",
            unchanged,
        )

        print("\n=== V3 BOUNDARY ===")
        print("WEATHER_V1=FROZEN")
        print("WEATHER_V2=FROZEN")
        print(
            "WEATHER_V3="
            "WFS_WEATHER_INTELLIGENCE_V3"
        )

        print(
            "V3_PARENT="
            "weather_game_features_v2"
        )

        print(
            "V3_MODE=SHADOW_ONLY"
        )

        print(
            "WIND_RESEARCH_BANDS="
            "DESCRIPTIVE_ONLY"
        )

        print(
            "15_MPH_PRODUCTION_THRESHOLD="
            "NOT_CREATED"
        )

        print(
            "PRODUCTION_COEFFICIENT="
            "NOT_CREATED"
        )

        print(
            "PROJECTION_INFLUENCE="
            "DISABLED"
        )

        print(
            "OPTIMIZER_INFLUENCE="
            "DISABLED"
        )

        print(
            "STAGE25_INFLUENCE="
            "DISABLED"
        )

        print(
            "DOME_WEATHER_SEMANTICS="
            "RAW_SOURCE_ONLY"
        )

        print(
            "\n"
            "WEATHER_V3_SHADOW_BUILD=PASS"
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
