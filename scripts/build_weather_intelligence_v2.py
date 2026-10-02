#!/usr/bin/env python3

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone


DB_DEFAULT = "data/nfl.db"

SOURCE = "NFLWeather"
V1_PARSER_VERSION = "WFS_WEATHER_INTELLIGENCE_V1"
V2_FEATURE_VERSION = "WFS_WEATHER_INTELLIGENCE_V2"

EXPECTED_PERIODS = [
    "Kickoff",
    "Q2",
    "Q3",
    "Q4",
]

PERIOD_ORDER = {
    period: index
    for index, period in enumerate(EXPECTED_PERIODS)
}


# =====================================================================
# Schema
# =====================================================================

def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS weather_game_features_v2 (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,

            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,

            source_snapshot_id INTEGER NOT NULL,
            source_captured_at TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_parser_version TEXT NOT NULL,

            feature_version TEXT NOT NULL,
            feature_created_at TEXT NOT NULL,

            stadium_name TEXT,
            stadium_type TEXT,
            surface TEXT,
            location TEXT,

            period_count INTEGER NOT NULL,
            visibility_period_count INTEGER NOT NULL,

            kickoff_condition TEXT,
            q4_condition TEXT,
            condition_change INTEGER NOT NULL,
            unique_condition_count INTEGER NOT NULL,

            kickoff_temperature_f REAL,
            kickoff_feels_like_f REAL,
            kickoff_wind_mph REAL,
            kickoff_wind_direction TEXT,
            kickoff_gust_mph REAL,
            kickoff_precipitation_probability_pct REAL,
            kickoff_cloud_cover_pct REAL,
            kickoff_humidity_pct REAL,
            kickoff_dew_point_f REAL,
            kickoff_visibility_miles REAL,

            temperature_min_f REAL,
            temperature_max_f REAL,
            temperature_span_f REAL,

            feels_like_min_f REAL,
            feels_like_max_f REAL,

            wind_max_mph REAL,
            gust_max_mph REAL,

            precipitation_probability_max_pct REAL,
            cloud_cover_max_pct REAL,
            humidity_max_pct REAL,

            dew_point_min_f REAL,
            dew_point_max_f REAL,
            visibility_min_miles REAL,

            temperature_delta_f REAL,
            feels_like_delta_f REAL,
            wind_delta_mph REAL,
            gust_delta_mph REAL,
            precipitation_probability_delta_pct REAL,
            cloud_cover_delta_pct REAL,
            humidity_delta_pct REAL,
            dew_point_delta_f REAL,
            visibility_delta_miles REAL,

            FOREIGN KEY(source_snapshot_id)
                REFERENCES weather_game_snapshots(snapshot_id),

            UNIQUE(
                source_snapshot_id,
                feature_version
            )
        )
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS
            idx_weather_game_features_v2_game_capture
        ON weather_game_features_v2(
            game_id,
            source_captured_at
        )
    """)

    con.execute("""
        CREATE INDEX IF NOT EXISTS
            idx_weather_game_features_v2_season_week
        ON weather_game_features_v2(
            season,
            week,
            source_captured_at
        )
    """)


# =====================================================================
# Source contract
# =====================================================================

def resolve_latest_complete_capture(
    con: sqlite3.Connection,
    season: int,
    week: int,
) -> str:
    schedule_row = con.execute("""
        SELECT COUNT(*) AS n
        FROM games
        WHERE season = ?
          AND week = ?
    """, (season, week)).fetchone()

    if schedule_row is None:
        raise RuntimeError(
            "Could not resolve schedule game count."
        )

    schedule_games = int(schedule_row["n"])

    if schedule_games <= 0:
        raise RuntimeError(
            f"No schedule games found for "
            f"season={season} week={week}."
        )

    captures = con.execute("""
        SELECT
            captured_at,
            COUNT(*) AS game_rows,
            COUNT(DISTINCT game_id) AS unique_games
        FROM weather_game_snapshots
        WHERE season = ?
          AND week = ?
          AND source = ?
          AND parser_version = ?
        GROUP BY captured_at
        ORDER BY captured_at DESC
    """, (
        season,
        week,
        SOURCE,
        V1_PARSER_VERSION,
    )).fetchall()

    for row in captures:
        captured_at = row["captured_at"]
        game_rows = int(row["game_rows"])
        unique_games = int(row["unique_games"])

        if (
            game_rows == schedule_games
            and unique_games == schedule_games
        ):
            period_row = con.execute("""
                SELECT COUNT(*) AS n
                FROM weather_period_snapshots p
                JOIN weather_game_snapshots g
                  ON g.snapshot_id = p.snapshot_id
                WHERE g.season = ?
                  AND g.week = ?
                  AND g.source = ?
                  AND g.parser_version = ?
                  AND g.captured_at = ?
            """, (
                season,
                week,
                SOURCE,
                V1_PARSER_VERSION,
                captured_at,
            )).fetchone()

            period_rows = int(period_row["n"])

            if period_rows == schedule_games * 4:
                return captured_at

    raise RuntimeError(
        "No complete Weather V1 capture satisfies "
        f"the schedule contract for season={season} "
        f"week={week}."
    )


def load_source_rows(
    con: sqlite3.Connection,
    season: int,
    week: int,
    captured_at: str,
) -> list[sqlite3.Row]:
    rows = con.execute("""
        SELECT
            g.snapshot_id,
            g.game_id,
            g.season,
            g.week,
            g.source,
            g.source_url,
            g.captured_at,
            g.stadium_name,
            g.source_stadium_type,
            g.source_surface,
            g.source_location,
            g.parser_version,

            p.period,
            p.condition,
            p.temperature_f,
            p.feels_like_f,
            p.wind_mph,
            p.wind_direction,
            p.gust_mph,
            p.precipitation_probability_pct,
            p.cloud_cover_pct,
            p.humidity_pct,
            p.dew_point_f,
            p.visibility_miles

        FROM weather_game_snapshots g
        JOIN weather_period_snapshots p
          ON p.snapshot_id = g.snapshot_id

        WHERE g.season = ?
          AND g.week = ?
          AND g.source = ?
          AND g.parser_version = ?
          AND g.captured_at = ?

        ORDER BY
            g.game_id,
            CASE p.period
                WHEN 'Kickoff' THEN 1
                WHEN 'Q2' THEN 2
                WHEN 'Q3' THEN 3
                WHEN 'Q4' THEN 4
                ELSE 99
            END
    """, (
        season,
        week,
        SOURCE,
        V1_PARSER_VERSION,
        captured_at,
    )).fetchall()

    return rows


# =====================================================================
# Feature helpers
# =====================================================================

def numeric_values(
    rows: list[sqlite3.Row],
    field: str,
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row[field] is not None
    ]


def minimum(
    rows: list[sqlite3.Row],
    field: str,
):
    values = numeric_values(rows, field)
    return min(values) if values else None


def maximum(
    rows: list[sqlite3.Row],
    field: str,
):
    values = numeric_values(rows, field)
    return max(values) if values else None


def span(
    rows: list[sqlite3.Row],
    field: str,
):
    values = numeric_values(rows, field)

    if not values:
        return None

    return max(values) - min(values)


def delta(a, b):
    if a is None or b is None:
        return None

    return float(b) - float(a)


# =====================================================================
# Feature construction
# =====================================================================

def build_features(
    rows: list[sqlite3.Row],
    feature_created_at: str,
) -> list[dict]:
    by_game = defaultdict(list)

    for row in rows:
        by_game[row["game_id"]].append(row)

    features = []

    for game_id, game_rows in sorted(by_game.items()):
        game_rows = sorted(
            game_rows,
            key=lambda row: PERIOD_ORDER.get(
                row["period"],
                99,
            ),
        )

        periods = [
            row["period"]
            for row in game_rows
        ]

        if periods != EXPECTED_PERIODS:
            raise RuntimeError(
                f"{game_id}: invalid Weather V1 "
                f"period contract: {periods}"
            )

        snapshot_ids = {
            row["snapshot_id"]
            for row in game_rows
        }

        if len(snapshot_ids) != 1:
            raise RuntimeError(
                f"{game_id}: multiple source snapshot IDs "
                f"found: {snapshot_ids}"
            )

        kickoff = game_rows[0]
        q4 = game_rows[-1]

        conditions = [
            row["condition"]
            for row in game_rows
            if row["condition"] is not None
        ]

        feature = {
            "game_id": game_id,
            "season": kickoff["season"],
            "week": kickoff["week"],

            "source_snapshot_id":
                kickoff["snapshot_id"],
            "source_captured_at":
                kickoff["captured_at"],
            "source":
                kickoff["source"],
            "source_url":
                kickoff["source_url"],
            "source_parser_version":
                kickoff["parser_version"],

            "feature_version":
                V2_FEATURE_VERSION,
            "feature_created_at":
                feature_created_at,

            "stadium_name":
                kickoff["stadium_name"],
            "stadium_type":
                kickoff["source_stadium_type"],
            "surface":
                kickoff["source_surface"],
            "location":
                kickoff["source_location"],

            "period_count":
                len(game_rows),

            "visibility_period_count":
                sum(
                    row["visibility_miles"] is not None
                    for row in game_rows
                ),

            "kickoff_condition":
                kickoff["condition"],
            "q4_condition":
                q4["condition"],

            "condition_change":
                int(
                    kickoff["condition"]
                    != q4["condition"]
                ),

            "unique_condition_count":
                len(set(conditions)),

            "kickoff_temperature_f":
                kickoff["temperature_f"],
            "kickoff_feels_like_f":
                kickoff["feels_like_f"],
            "kickoff_wind_mph":
                kickoff["wind_mph"],
            "kickoff_wind_direction":
                kickoff["wind_direction"],
            "kickoff_gust_mph":
                kickoff["gust_mph"],
            "kickoff_precipitation_probability_pct":
                kickoff[
                    "precipitation_probability_pct"
                ],
            "kickoff_cloud_cover_pct":
                kickoff["cloud_cover_pct"],
            "kickoff_humidity_pct":
                kickoff["humidity_pct"],
            "kickoff_dew_point_f":
                kickoff["dew_point_f"],
            "kickoff_visibility_miles":
                kickoff["visibility_miles"],

            "temperature_min_f":
                minimum(
                    game_rows,
                    "temperature_f",
                ),
            "temperature_max_f":
                maximum(
                    game_rows,
                    "temperature_f",
                ),
            "temperature_span_f":
                span(
                    game_rows,
                    "temperature_f",
                ),

            "feels_like_min_f":
                minimum(
                    game_rows,
                    "feels_like_f",
                ),
            "feels_like_max_f":
                maximum(
                    game_rows,
                    "feels_like_f",
                ),

            "wind_max_mph":
                maximum(
                    game_rows,
                    "wind_mph",
                ),
            "gust_max_mph":
                maximum(
                    game_rows,
                    "gust_mph",
                ),

            "precipitation_probability_max_pct":
                maximum(
                    game_rows,
                    "precipitation_probability_pct",
                ),

            "cloud_cover_max_pct":
                maximum(
                    game_rows,
                    "cloud_cover_pct",
                ),

            "humidity_max_pct":
                maximum(
                    game_rows,
                    "humidity_pct",
                ),

            "dew_point_min_f":
                minimum(
                    game_rows,
                    "dew_point_f",
                ),
            "dew_point_max_f":
                maximum(
                    game_rows,
                    "dew_point_f",
                ),

            "visibility_min_miles":
                minimum(
                    game_rows,
                    "visibility_miles",
                ),

            "temperature_delta_f":
                delta(
                    kickoff["temperature_f"],
                    q4["temperature_f"],
                ),

            "feels_like_delta_f":
                delta(
                    kickoff["feels_like_f"],
                    q4["feels_like_f"],
                ),

            "wind_delta_mph":
                delta(
                    kickoff["wind_mph"],
                    q4["wind_mph"],
                ),

            "gust_delta_mph":
                delta(
                    kickoff["gust_mph"],
                    q4["gust_mph"],
                ),

            "precipitation_probability_delta_pct":
                delta(
                    kickoff[
                        "precipitation_probability_pct"
                    ],
                    q4[
                        "precipitation_probability_pct"
                    ],
                ),

            "cloud_cover_delta_pct":
                delta(
                    kickoff["cloud_cover_pct"],
                    q4["cloud_cover_pct"],
                ),

            "humidity_delta_pct":
                delta(
                    kickoff["humidity_pct"],
                    q4["humidity_pct"],
                ),

            "dew_point_delta_f":
                delta(
                    kickoff["dew_point_f"],
                    q4["dew_point_f"],
                ),

            "visibility_delta_miles":
                delta(
                    kickoff["visibility_miles"],
                    q4["visibility_miles"],
                ),
        }

        features.append(feature)

    return features


# =====================================================================
# Validation
# =====================================================================

def validate_features(
    con: sqlite3.Connection,
    season: int,
    week: int,
    features: list[dict],
) -> None:
    schedule_games = int(
        con.execute("""
            SELECT COUNT(*)
            FROM games
            WHERE season = ?
              AND week = ?
        """, (season, week)).fetchone()[0]
    )

    if len(features) != schedule_games:
        raise RuntimeError(
            f"Expected {schedule_games} V2 feature rows; "
            f"found {len(features)}."
        )

    game_ids = [
        row["game_id"]
        for row in features
    ]

    if len(set(game_ids)) != schedule_games:
        raise RuntimeError(
            "Weather V2 game IDs are not unique."
        )

    for row in features:
        if row["period_count"] != 4:
            raise RuntimeError(
                f"{row['game_id']}: period_count "
                f"is not 4."
            )

        if not row["source_snapshot_id"]:
            raise RuntimeError(
                f"{row['game_id']}: missing "
                f"source_snapshot_id."
            )

        if not row["source_captured_at"]:
            raise RuntimeError(
                f"{row['game_id']}: missing "
                f"source_captured_at."
            )

        if row["source"] != SOURCE:
            raise RuntimeError(
                f"{row['game_id']}: unexpected source "
                f"{row['source']!r}."
            )

        if (
            row["source_parser_version"]
            != V1_PARSER_VERSION
        ):
            raise RuntimeError(
                f"{row['game_id']}: unexpected V1 "
                f"parser version."
            )

        if (
            row["feature_version"]
            != V2_FEATURE_VERSION
        ):
            raise RuntimeError(
                f"{row['game_id']}: unexpected V2 "
                f"feature version."
            )

        if row["condition_change"] not in (0, 1):
            raise RuntimeError(
                f"{row['game_id']}: invalid "
                f"condition_change."
            )

        if not (
            0
            <= row["visibility_period_count"]
            <= 4
        ):
            raise RuntimeError(
                f"{row['game_id']}: invalid "
                f"visibility_period_count."
            )


# =====================================================================
# Output
# =====================================================================

INSERT_COLUMNS = [
    "game_id",
    "season",
    "week",

    "source_snapshot_id",
    "source_captured_at",
    "source",
    "source_url",
    "source_parser_version",

    "feature_version",
    "feature_created_at",

    "stadium_name",
    "stadium_type",
    "surface",
    "location",

    "period_count",
    "visibility_period_count",

    "kickoff_condition",
    "q4_condition",
    "condition_change",
    "unique_condition_count",

    "kickoff_temperature_f",
    "kickoff_feels_like_f",
    "kickoff_wind_mph",
    "kickoff_wind_direction",
    "kickoff_gust_mph",
    "kickoff_precipitation_probability_pct",
    "kickoff_cloud_cover_pct",
    "kickoff_humidity_pct",
    "kickoff_dew_point_f",
    "kickoff_visibility_miles",

    "temperature_min_f",
    "temperature_max_f",
    "temperature_span_f",

    "feels_like_min_f",
    "feels_like_max_f",

    "wind_max_mph",
    "gust_max_mph",

    "precipitation_probability_max_pct",
    "cloud_cover_max_pct",
    "humidity_max_pct",

    "dew_point_min_f",
    "dew_point_max_f",
    "visibility_min_miles",

    "temperature_delta_f",
    "feels_like_delta_f",
    "wind_delta_mph",
    "gust_delta_mph",
    "precipitation_probability_delta_pct",
    "cloud_cover_delta_pct",
    "humidity_delta_pct",
    "dew_point_delta_f",
    "visibility_delta_miles",
]


def print_feature_summary(
    features: list[dict],
) -> None:
    print()
    print("=== WEATHER V2 FEATURE SUMMARY ===")

    for row in features:
        print(
            f"{row['game_id']} "
            f"{row['stadium_type']} | "
            f"KO={row['kickoff_condition']} | "
            f"temp="
            f"{row['temperature_min_f']}"
            f"-{row['temperature_max_f']} | "
            f"wind_max={row['wind_max_mph']} | "
            f"gust_max={row['gust_max_mph']} | "
            f"precip_max="
            f"{row['precipitation_probability_max_pct']} | "
            f"vis_min="
            f"{row['visibility_min_miles']} | "
            f"condition_change="
            f"{bool(row['condition_change'])}"
        )


def write_features(
    con: sqlite3.Connection,
    features: list[dict],
) -> tuple[int, int]:
    placeholders = ", ".join(
        "?"
        for _ in INSERT_COLUMNS
    )

    columns_sql = ", ".join(
        INSERT_COLUMNS
    )

    insert_sql = f"""
        INSERT INTO weather_game_features_v2 (
            {columns_sql}
        )
        VALUES (
            {placeholders}
        )
    """

    compare_columns = [
        column
        for column in INSERT_COLUMNS
        if column != "feature_created_at"
    ]

    inserted = 0
    unchanged = 0

    for row in features:
        existing = con.execute("""
            SELECT *
            FROM weather_game_features_v2
            WHERE source_snapshot_id = ?
              AND feature_version = ?
        """, (
            row["source_snapshot_id"],
            row["feature_version"],
        )).fetchone()

        if existing is None:
            values = [
                row[column]
                for column in INSERT_COLUMNS
            ]

            con.execute(insert_sql, values)
            inserted += 1
            continue

        differences = []

        for column in compare_columns:
            stored_value = existing[column]
            derived_value = row[column]

            if stored_value != derived_value:
                differences.append(
                    (
                        column,
                        stored_value,
                        derived_value,
                    )
                )

        if differences:
            detail = "; ".join(
                f"{column}: "
                f"stored={stored!r} "
                f"derived={derived!r}"
                for column, stored, derived
                in differences
            )

            raise RuntimeError(
                f"{row['game_id']}: immutable V2 "
                f"feature conflict for "
                f"source_snapshot_id="
                f"{row['source_snapshot_id']} "
                f"feature_version="
                f"{row['feature_version']!r}; "
                f"{detail}. "
                f"Create a new feature version "
                f"for changed derivation logic."
            )

        unchanged += 1

    return inserted, unchanged


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Weather Intelligence V2 "
            "game-level analytical features from validated "
            "Weather V1 snapshots."
        )
    )

    parser.add_argument(
        "--db",
        default=DB_DEFAULT,
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    # Enforce FK behavior for this connection.
    con.execute("PRAGMA foreign_keys = ON")

    try:
        captured_at = resolve_latest_complete_capture(
            con,
            args.season,
            args.week,
        )

        source_rows = load_source_rows(
            con,
            args.season,
            args.week,
            captured_at,
        )

        feature_created_at = (
            datetime.now(timezone.utc).isoformat()
        )

        features = build_features(
            source_rows,
            feature_created_at,
        )

        validate_features(
            con,
            args.season,
            args.week,
            features,
        )

        print("=" * 72)
        print(V2_FEATURE_VERSION)
        print("=" * 72)

        print(
            f"season={args.season} "
            f"week={args.week}"
        )

        print(
            f"SOURCE_CAPTURED_AT={captured_at}"
        )

        print(
            f"SOURCE_PERIOD_ROWS={len(source_rows)}"
        )

        print(
            f"FEATURE_ROWS={len(features)}"
        )

        print_feature_summary(features)

        dome_games = [
            row["game_id"]
            for row in features
            if str(
                row["stadium_type"]
            ).lower() == "dome"
        ]

        print()
        print(
            f"DOME_GAMES={len(dome_games)}"
        )

        print(
            "DOME_WEATHER_SEMANTICS="
            "RAW_SOURCE_ONLY"
        )

        print(
            "PRODUCTION_INFLUENCE=DISABLED"
        )

        if args.dry_run:
            print()
            print(
                "DRY_RUN=PASS "
                "(no schema or database writes)"
            )
            return

        # Schema creation and feature writes occur only
        # after all source/feature validation succeeds.
        con.execute("BEGIN")

        ensure_schema(con)

        inserted, unchanged = write_features(
            con,
            features,
        )

        if inserted + unchanged != len(features):
            raise RuntimeError(
                "Feature persistence count mismatch."
            )

        stored = int(
            con.execute("""
                SELECT COUNT(*)
                FROM weather_game_features_v2
                WHERE season = ?
                  AND week = ?
                  AND source_captured_at = ?
                  AND feature_version = ?
            """, (
                args.season,
                args.week,
                captured_at,
                V2_FEATURE_VERSION,
            )).fetchone()[0]
        )

        if stored != len(features):
            raise RuntimeError(
                f"Stored feature count mismatch: "
                f"{stored} != {len(features)}"
            )

        con.commit()

        print()
        print(
            f"FEATURE_CREATED_AT="
            f"{feature_created_at}"
        )
        print(
            f"FEATURE_ROWS_INSERTED={inserted}"
        )
        print(
            f"FEATURE_ROWS_UNCHANGED={unchanged}"
        )
        print(
            "WEATHER_V2_FEATURE_WRITE=PASS"
        )

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


if __name__ == "__main__":
    main()
