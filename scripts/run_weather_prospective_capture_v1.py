#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import sqlite3
import subprocess
import sys

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data" / "nfl.db"

V1_PATH = (
    ROOT
    / "scripts"
    / "build_weather_intelligence_v1.py"
)

V2_PATH = (
    ROOT
    / "scripts"
    / "build_weather_intelligence_v2.py"
)

V3_PATH = (
    ROOT
    / "scripts"
    / "build_weather_intelligence_v3.py"
)


CONTROLLER_VERSION = (
    "WFS_WEATHER_PROSPECTIVE_CONTROLLER_V1"
)

EXPECTED_V1_SHA = (
    "e9e347e9d8f9aeee88794d3b7d003cf3a04b4a88701b748b42701ec5a880aa91"
)

EXPECTED_V2_SHA = (
    "e74428d8c4d09233235b1b9ffed734a3c241413da8c6974bc975f40f3e1483e9"
)

EXPECTED_V3_SHA = (
    "579849b9492ff8f7f7fb4cec788a01a855c5874d3d799151248eaefe97d2b3f7"
)


V1_PARSER_VERSION = (
    "WFS_WEATHER_INTELLIGENCE_V1"
)

V2_FEATURE_VERSION = (
    "WFS_WEATHER_INTELLIGENCE_V2"
)

V3_SHADOW_VERSION = (
    "WFS_WEATHER_INTELLIGENCE_V3"
)


ET = ZoneInfo(
    "America/New_York"
)

UTC = timezone.utc


CHECKPOINTS = (
    (
        "T_MINUS_72H",
        72,
    ),
    (
        "T_MINUS_24H",
        24,
    ),
    (
        "T_MINUS_3H",
        3,
    ),
)


EXECUTION_WINDOW_MINUTES = 5


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def utc_now():

    return datetime.now(
        UTC
    )


def sha256_file(path):

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def verify_frozen_builders():

    expected = {
        V1_PATH:
            EXPECTED_V1_SHA,

        V2_PATH:
            EXPECTED_V2_SHA,

        V3_PATH:
            EXPECTED_V3_SHA,
    }

    print(
        "\n=== FROZEN BUILDER VERIFICATION ==="
    )

    for path, wanted in expected.items():

        actual = sha256_file(
            path
        )

        print(
            path.name,
            actual,
            "MATCH=",
            actual == wanted,
        )

        if actual != wanted:

            raise RuntimeError(
                "Frozen builder hash mismatch: "
                + str(path)
            )

    print(
        "FROZEN_BUILDER_HASHES=PASS"
    )


def import_frozen_module(
    module_name,
    path,
):

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Unable to load frozen module: "
            + str(path)
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# SCHEDULE
# =============================================================================

def parse_kickoff(
    game_date,
    gametime,
):

    if not game_date:
        raise RuntimeError(
            "Missing game_date."
        )

    if not gametime:
        raise RuntimeError(
            "Missing gametime."
        )

    naive = datetime.fromisoformat(
        f"{game_date}T{gametime}"
    )

    kickoff_et = naive.replace(
        tzinfo=ET
    )

    kickoff_utc = (
        kickoff_et
        .astimezone(
            UTC
        )
    )

    return (
        kickoff_et,
        kickoff_utc,
    )


def load_schedule(
    season,
    week,
):

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    con.row_factory = (
        sqlite3.Row
    )

    try:

        rows = con.execute(
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
            WHERE season = ?
              AND week = ?
            ORDER BY
                game_date,
                gametime,
                game_id
            """,
            (
                season,
                week,
            ),
        ).fetchall()

    finally:
        con.close()


    if not rows:

        raise RuntimeError(
            "No schedule rows found."
        )


    games = []

    seen = set()

    for row in rows:

        game_id = row["game_id"]

        if game_id in seen:

            raise RuntimeError(
                "Duplicate schedule game_id: "
                + str(game_id)
            )

        seen.add(
            game_id
        )

        (
            kickoff_et,
            kickoff_utc,
        ) = parse_kickoff(
            row["game_date"],
            row["gametime"],
        )

        games.append({
            "game_id":
                game_id,

            "season":
                row["season"],

            "week":
                row["week"],

            "away_team":
                row["away_team"],

            "home_team":
                row["home_team"],

            "completed":
                row["completed"],

            "kickoff_et":
                kickoff_et,

            "kickoff_utc":
                kickoff_utc,
        })


    return games


def build_events(
    games,
    now,
):

    groups = defaultdict(
        list
    )

    for game in games:

        groups[
            game["kickoff_utc"]
        ].append(
            game
        )


    events = []

    for kickoff_utc in sorted(
        groups
    ):

        group_games = sorted(
            groups[kickoff_utc],
            key=lambda game:
                game["game_id"],
        )

        for (
            checkpoint,
            checkpoint_hours,
        ) in CHECKPOINTS:

            scheduled = (
                kickoff_utc
                - timedelta(
                    hours=
                        checkpoint_hours
                )
            )

            if now < scheduled:

                state = "FUTURE"

            elif now < kickoff_utc:

                state = "MISSED"

            else:

                state = "PAST"


            events.append({
                "checkpoint":
                    checkpoint,

                "checkpoint_hours":
                    checkpoint_hours,

                "scheduled_trigger_at_utc":
                    scheduled,

                "target_kickoff_at_utc":
                    kickoff_utc,

                "target_games":
                    group_games,

                "state":
                    state,
            })


    events.sort(
        key=lambda event: (
            event[
                "scheduled_trigger_at_utc"
            ],
            event[
                "target_kickoff_at_utc"
            ],
            event[
                "checkpoint_hours"
            ],
        )
    )

    return (
        groups,
        events,
    )


def execution_delta_seconds(
    event,
    now,
):

    return (
        now
        - event[
            "scheduled_trigger_at_utc"
        ]
    ).total_seconds()


def events_in_execution_window(
    events,
    now,
):

    window_seconds = (
        EXECUTION_WINDOW_MINUTES
        * 60
    )

    eligible = []

    for event in events:

        # Never execute at or after kickoff.
        if now >= event[
            "target_kickoff_at_utc"
        ]:

            continue

        delta = abs(
            execution_delta_seconds(
                event,
                now,
            )
        )

        if delta <= window_seconds:

            eligible.append(
                event
            )

    return eligible


# =============================================================================
# DATABASE STATE
# =============================================================================

def weather_counts():

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    try:

        result = {}

        for table in (
            "weather_game_snapshots",
            "weather_period_snapshots",
            "weather_game_features_v2",
            "weather_game_shadow_v3",
            "weather_prospective_capture_runs",
            "weather_prospective_capture_targets",
        ):

            result[table] = (
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM "{table}"
                    """
                ).fetchone()[0]
            )

        return result

    finally:
        con.close()


def capture_inventory(
    season,
    week,
):

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    con.row_factory = (
        sqlite3.Row
    )

    try:

        rows = con.execute(
            """
            SELECT
                captured_at,
                COUNT(*) AS rows,
                COUNT(
                    DISTINCT game_id
                ) AS games
            FROM weather_game_snapshots
            WHERE season = ?
              AND week = ?
              AND source = 'NFLWeather'
              AND parser_version = ?
            GROUP BY captured_at
            ORDER BY captured_at
            """,
            (
                season,
                week,
                V1_PARSER_VERSION,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        con.close()


def foreign_key_errors():

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    try:

        return con.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    finally:
        con.close()


def ledger_identity_exists(
    season,
    week,
    event,
):

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    try:

        row = con.execute(
            """
            SELECT
                run_id,
                status
            FROM weather_prospective_capture_runs
            WHERE season = ?
              AND week = ?
              AND checkpoint = ?
              AND target_kickoff_at_utc = ?
            """,
            (
                season,
                week,
                event[
                    "checkpoint"
                ],
                event[
                    "target_kickoff_at_utc"
                ].isoformat(),
            ),
        ).fetchone()

        return row

    finally:
        con.close()


# =============================================================================
# LEDGER WRITES
# =============================================================================

def create_ledger_attempt(
    season,
    week,
    event,
    weekly_game_count,
    attempt_started_at,
):

    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    con.execute(
        "PRAGMA foreign_keys = ON"
    )

    con.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:

        con.execute(
            "BEGIN IMMEDIATE"
        )

        existing = con.execute(
            """
            SELECT
                run_id,
                status
            FROM weather_prospective_capture_runs
            WHERE season = ?
              AND week = ?
              AND checkpoint = ?
              AND target_kickoff_at_utc = ?
            """,
            (
                season,
                week,
                event[
                    "checkpoint"
                ],
                event[
                    "target_kickoff_at_utc"
                ].isoformat(),
            ),
        ).fetchone()

        if existing is not None:

            raise RuntimeError(
                "Checkpoint ledger identity "
                "already exists."
            )


        cur = con.execute(
            """
            INSERT INTO
                weather_prospective_capture_runs
            (
                controller_version,
                season,
                week,
                checkpoint,
                checkpoint_hours,
                scheduled_trigger_at_utc,
                target_kickoff_at_utc,
                target_game_count,
                weekly_schedule_game_count,
                capture_scope,
                evaluation_scope,
                attempt_started_at_utc,
                status,
                failure_reason,
                production_influence_allowed
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'FULL_WEEK',
                'TARGET_KICKOFF_GROUP_ONLY',
                ?,
                'STARTED',
                NULL,
                0
            )
            """,
            (
                CONTROLLER_VERSION,
                season,
                week,
                event[
                    "checkpoint"
                ],
                event[
                    "checkpoint_hours"
                ],
                event[
                    "scheduled_trigger_at_utc"
                ].isoformat(),
                event[
                    "target_kickoff_at_utc"
                ].isoformat(),
                len(
                    event[
                        "target_games"
                    ]
                ),
                weekly_game_count,
                attempt_started_at.isoformat(),
            ),
        )

        run_id = cur.lastrowid


        for game in event[
            "target_games"
        ]:

            con.execute(
                """
                INSERT INTO
                    weather_prospective_capture_targets
                (
                    run_id,
                    game_id,
                    eligible_for_checkpoint_evaluation
                )
                VALUES (
                    ?,
                    ?,
                    1
                )
                """,
                (
                    run_id,
                    game[
                        "game_id"
                    ],
                ),
            )


        con.commit()

        return run_id

    except Exception:

        con.rollback()
        raise

    finally:
        con.close()


def mark_ledger_failed(
    run_id,
    reason,
):

    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    try:

        con.execute(
            """
            UPDATE
                weather_prospective_capture_runs
            SET
                status = 'FAILED',
                failure_reason = ?
            WHERE run_id = ?
              AND status = 'STARTED'
            """,
            (
                str(reason)[:4000],
                run_id,
            ),
        )

        if con.total_changes != 1:

            con.rollback()

            raise RuntimeError(
                "Unable to mark ledger FAILED."
            )

        con.commit()

    finally:
        con.close()


def mark_ledger_success(
    run_id,
    captured_at,
):

    completed_at = (
        utc_now()
        .isoformat()
    )

    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    try:

        con.execute(
            """
            UPDATE
                weather_prospective_capture_runs
            SET
                capture_completed_at_utc = ?,
                v1_captured_at = ?,
                v2_feature_version = ?,
                v3_shadow_version = ?,
                status = 'SUCCESS',
                failure_reason = NULL
            WHERE run_id = ?
              AND status = 'STARTED'
            """,
            (
                completed_at,
                captured_at,
                V2_FEATURE_VERSION,
                V3_SHADOW_VERSION,
                run_id,
            ),
        )

        if con.total_changes != 1:

            con.rollback()

            raise RuntimeError(
                "Unable to mark ledger SUCCESS."
            )

        con.commit()

    finally:
        con.close()


# =============================================================================
# EXACT LINEAGE
# =============================================================================

def run_v1(
    season,
    week,
):

    command = [
        sys.executable,
        str(V1_PATH),
        "--season",
        str(season),
        "--week",
        str(week),
    ]

    print(
        "\n=== V1 EXECUTION ==="
    )

    print(
        "COMMAND =",
        command,
    )

    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
    )

    print(
        "V1_RETURN_CODE =",
        completed.returncode,
    )

    if completed.returncode != 0:

        raise RuntimeError(
            "V1 execution failed."
        )


def resolve_exact_new_capture(
    before,
    after,
    expected_game_count,
):

    before_set = {
        row["captured_at"]
        for row in before
    }

    new_rows = [
        row
        for row in after
        if row["captured_at"]
        not in before_set
    ]

    print(
        "NEW_CAPTURE_COUNT =",
        len(new_rows),
    )

    for row in new_rows:
        print(
            "NEW_CAPTURE =",
            row,
        )


    if len(new_rows) != 1:

        raise RuntimeError(
            "Expected exactly one new "
            "V1 captured_at."
        )


    capture = new_rows[0]

    if (
        capture["rows"]
        != expected_game_count
    ):

        raise RuntimeError(
            "New V1 capture row count "
            "does not match schedule."
        )


    if (
        capture["games"]
        != expected_game_count
    ):

        raise RuntimeError(
            "New V1 capture game count "
            "does not match schedule."
        )


    return capture[
        "captured_at"
    ]


def build_exact_v2(
    v2,
    season,
    week,
    captured_at,
    expected_game_count,
):

    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    con.row_factory = (
        sqlite3.Row
    )

    con.execute(
        "PRAGMA foreign_keys = ON"
    )

    con.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:

        rows = v2.load_source_rows(
            con,
            season,
            week,
            captured_at,
        )

        expected_period_rows = (
            expected_game_count
            * 4
        )

        if len(rows) != expected_period_rows:

            raise RuntimeError(
                "Exact V1 period cardinality "
                "mismatch."
            )


        feature_created_at = (
            utc_now()
            .isoformat()
        )

        features = v2.build_features(
            rows,
            feature_created_at,
        )

        v2.validate_features(
            con,
            season,
            week,
            features,
        )

        if len(features) != expected_game_count:

            raise RuntimeError(
                "V2 feature cardinality mismatch."
            )


        con.execute(
            "BEGIN IMMEDIATE"
        )

        try:

            (
                inserted,
                unchanged,
            ) = v2.write_features(
                con,
                features,
            )

            con.commit()

        except Exception:

            con.rollback()
            raise


        print(
            "V2_INSERTED =",
            inserted,
        )

        print(
            "V2_UNCHANGED =",
            unchanged,
        )


        if inserted != expected_game_count:

            raise RuntimeError(
                "Prospective V2 expected all "
                "rows to be newly inserted."
            )

        if unchanged != 0:

            raise RuntimeError(
                "Unexpected existing V2 lineage."
            )

    finally:
        con.close()


def load_exact_v2_parents(
    season,
    week,
    captured_at,
    expected_game_count,
):

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    con.row_factory = (
        sqlite3.Row
    )

    try:

        rows = con.execute(
            """
            SELECT *
            FROM weather_game_features_v2
            WHERE season = ?
              AND week = ?
              AND source_captured_at = ?
              AND feature_version = ?
            ORDER BY game_id
            """,
            (
                season,
                week,
                captured_at,
                V2_FEATURE_VERSION,
            ),
        ).fetchall()

    finally:
        con.close()


    if len(rows) != expected_game_count:

        raise RuntimeError(
            "Exact V2 parent cardinality "
            "mismatch."
        )


    ids = {
        row[
            "feature_id"
        ]
        for row in rows
    }

    games = {
        row[
            "game_id"
        ]
        for row in rows
    }

    if len(ids) != expected_game_count:

        raise RuntimeError(
            "Duplicate V2 parent IDs."
        )

    if len(games) != expected_game_count:

        raise RuntimeError(
            "Duplicate V2 parent games."
        )


    return rows


def build_exact_v3(
    v3,
    parent_rows,
    expected_game_count,
):

    candidates = [
        v3.build_candidate(
            row
        )
        for row in parent_rows
    ]


    if len(candidates) != expected_game_count:

        raise RuntimeError(
            "V3 candidate cardinality mismatch."
        )


    con = sqlite3.connect(
        DB,
        timeout=30,
    )

    con.row_factory = (
        sqlite3.Row
    )

    con.execute(
        "PRAGMA foreign_keys = ON"
    )

    con.execute(
        "PRAGMA busy_timeout = 30000"
    )

    try:

        v3.validate_parent_schema(
            con
        )

        con.execute(
            "BEGIN IMMEDIATE"
        )

        try:

            (
                inserted,
                unchanged,
            ) = v3.persist(
                con,
                candidates,
            )

            con.commit()

        except Exception:

            con.rollback()
            raise


        print(
            "V3_INSERTED =",
            inserted,
        )

        print(
            "V3_UNCHANGED =",
            unchanged,
        )


        if inserted != expected_game_count:

            raise RuntimeError(
                "Prospective V3 expected all "
                "rows to be newly inserted."
            )

        if unchanged != 0:

            raise RuntimeError(
                "Unexpected existing V3 lineage."
            )

    finally:
        con.close()


def verify_exact_lineage(
    season,
    week,
    captured_at,
    expected_game_count,
):

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    con.row_factory = (
        sqlite3.Row
    )

    try:

        v1 = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(
                    DISTINCT game_id
                ) AS games,
                COUNT(
                    DISTINCT snapshot_id
                ) AS snapshot_ids
            FROM weather_game_snapshots
            WHERE season = ?
              AND week = ?
              AND captured_at = ?
              AND source = 'NFLWeather'
              AND parser_version = ?
            """,
            (
                season,
                week,
                captured_at,
                V1_PARSER_VERSION,
            ),
        ).fetchone()


        periods = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(
                    DISTINCT p.game_id
                ) AS games
            FROM weather_period_snapshots p
            JOIN weather_game_snapshots g
              ON g.snapshot_id =
                 p.snapshot_id
            WHERE g.season = ?
              AND g.week = ?
              AND g.captured_at = ?
              AND g.source = 'NFLWeather'
              AND g.parser_version = ?
            """,
            (
                season,
                week,
                captured_at,
                V1_PARSER_VERSION,
            ),
        ).fetchone()


        v2 = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(
                    DISTINCT game_id
                ) AS games,
                COUNT(
                    DISTINCT feature_id
                ) AS feature_ids,
                COUNT(
                    DISTINCT source_snapshot_id
                ) AS snapshot_ids
            FROM weather_game_features_v2
            WHERE season = ?
              AND week = ?
              AND source_captured_at = ?
              AND feature_version = ?
            """,
            (
                season,
                week,
                captured_at,
                V2_FEATURE_VERSION,
            ),
        ).fetchone()


        v3 = con.execute(
            """
            SELECT
                COUNT(*) AS rows,
                COUNT(
                    DISTINCT game_id
                ) AS games,
                COUNT(
                    DISTINCT source_v2_feature_id
                ) AS feature_ids,
                COUNT(
                    DISTINCT source_snapshot_id
                ) AS snapshot_ids,
                SUM(
                    CASE
                        WHEN production_influence_allowed
                             != 0
                        THEN 1
                        ELSE 0
                    END
                ) AS production_violations
            FROM weather_game_shadow_v3
            WHERE season = ?
              AND week = ?
              AND source_captured_at = ?
              AND shadow_version = ?
            """,
            (
                season,
                week,
                captured_at,
                V3_SHADOW_VERSION,
            ),
        ).fetchone()


        broken_v2 = con.execute(
            """
            SELECT COUNT(*)
            FROM weather_game_features_v2 f
            LEFT JOIN weather_game_snapshots g
              ON g.snapshot_id =
                 f.source_snapshot_id
            WHERE f.season = ?
              AND f.week = ?
              AND f.source_captured_at = ?
              AND f.feature_version = ?
              AND g.snapshot_id IS NULL
            """,
            (
                season,
                week,
                captured_at,
                V2_FEATURE_VERSION,
            ),
        ).fetchone()[0]


        broken_v3 = con.execute(
            """
            SELECT COUNT(*)
            FROM weather_game_shadow_v3 s
            LEFT JOIN weather_game_features_v2 f
              ON f.feature_id =
                 s.source_v2_feature_id
            WHERE s.season = ?
              AND s.week = ?
              AND s.source_captured_at = ?
              AND s.shadow_version = ?
              AND f.feature_id IS NULL
            """,
            (
                season,
                week,
                captured_at,
                V3_SHADOW_VERSION,
            ),
        ).fetchone()[0]


        fk = con.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    finally:
        con.close()


    print(
        "\n=== EXACT LINEAGE VERIFICATION ==="
    )

    print(
        "V1 =",
        dict(v1),
    )

    print(
        "PERIODS =",
        dict(periods),
    )

    print(
        "V2 =",
        dict(v2),
    )

    print(
        "V3 =",
        dict(v3),
    )

    print(
        "BROKEN_V2_TO_V1 =",
        broken_v2,
    )

    print(
        "BROKEN_V3_TO_V2 =",
        broken_v3,
    )

    print(
        "FOREIGN_KEY_ERRORS =",
        len(fk),
    )


    if (
        v1["rows"]
        != expected_game_count
        or v1["games"]
        != expected_game_count
        or v1["snapshot_ids"]
        != expected_game_count
    ):

        raise RuntimeError(
            "V1 exact-lineage verification failed."
        )


    if (
        periods["rows"]
        != expected_game_count * 4
        or periods["games"]
        != expected_game_count
    ):

        raise RuntimeError(
            "V1 period verification failed."
        )


    if (
        v2["rows"]
        != expected_game_count
        or v2["games"]
        != expected_game_count
        or v2["feature_ids"]
        != expected_game_count
        or v2["snapshot_ids"]
        != expected_game_count
    ):

        raise RuntimeError(
            "V2 exact-lineage verification failed."
        )


    production_violations = (
        v3[
            "production_violations"
        ]
        or 0
    )

    if (
        v3["rows"]
        != expected_game_count
        or v3["games"]
        != expected_game_count
        or v3["feature_ids"]
        != expected_game_count
        or v3["snapshot_ids"]
        != expected_game_count
        or production_violations != 0
    ):

        raise RuntimeError(
            "V3 exact-lineage verification failed."
        )


    if (
        broken_v2 != 0
        or broken_v3 != 0
        or fk
    ):

        raise RuntimeError(
            "Lineage/FK verification failed."
        )


    print(
        "EXACT_LINEAGE_VERIFICATION=PASS"
    )


# =============================================================================
# PRESENTATION
# =============================================================================

def print_schedule_summary(
    games,
    groups,
    events,
    now,
):

    print(
        "\n=== CONTROLLER CLOCK ==="
    )

    print(
        "NOW_UTC =",
        now.isoformat(),
    )

    print(
        "NOW_ET =",
        now.astimezone(
            ET
        ).isoformat(),
    )


    print(
        "\n=== SCHEDULE ==="
    )

    print(
        "EXPECTED_GAME_COUNT =",
        len(games),
    )

    print(
        "KICKOFF_GROUP_COUNT =",
        len(groups),
    )


    print(
        "\n=== CHECKPOINT EVENTS ==="
    )

    for event in events:

        print(
            f"{event['checkpoint']:<12} "
            f"scheduled="
            f"{event['scheduled_trigger_at_utc'].isoformat()} "
            f"kickoff="
            f"{event['target_kickoff_at_utc'].isoformat()} "
            f"target_games="
            f"{len(event['target_games']):<2} "
            f"state="
            f"{event['state']}"
        )


def print_boundary():

    print(
        "\n=== SAFETY BOUNDARY ==="
    )

    print(
        "V1_V2_V3=FROZEN"
    )

    print(
        "CAPTURE_SCOPE=FULL_WEEK"
    )

    print(
        "EVALUATION_SCOPE=TARGET_KICKOFF_GROUP_ONLY"
    )

    print(
        "CHECKPOINT_LABEL_PROPAGATION="
        "TARGET_GAMES_ONLY"
    )

    print(
        "MISSED_CHECKPOINT_BACKFILL=FORBIDDEN"
    )

    print(
        "POST_KICKOFF_PREGAME_CAPTURE=FORBIDDEN"
    )

    print(
        "PRODUCTION_INFLUENCE=DISABLED"
    )

    print(
        "PROJECTION_INFLUENCE=DISABLED"
    )

    print(
        "OPTIMIZER_INFLUENCE=DISABLED"
    )

    print(
        "STAGE25_INFLUENCE=DISABLED"
    )


# =============================================================================
# DRY RUN
# =============================================================================

def dry_run(
    season,
    week,
):

    verify_frozen_builders()

    now = utc_now()

    games = load_schedule(
        season,
        week,
    )

    (
        groups,
        events,
    ) = build_events(
        games,
        now,
    )

    print_schedule_summary(
        games,
        groups,
        events,
        now,
    )


    future = [
        event
        for event in events
        if event["state"]
        == "FUTURE"
    ]

    missed = [
        event
        for event in events
        if event["state"]
        == "MISSED"
    ]

    past = [
        event
        for event in events
        if event["state"]
        == "PAST"
    ]


    print(
        "\n=== EVENT COUNTS ==="
    )

    print(
        "FUTURE =",
        len(future),
    )

    print(
        "MISSED =",
        len(missed),
    )

    print(
        "PAST =",
        len(past),
    )


    eligible = (
        events_in_execution_window(
            events,
            now,
        )
    )

    print(
        "\n=== EXECUTION WINDOW ==="
    )

    print(
        "EXECUTION_WINDOW_MINUTES =",
        EXECUTION_WINDOW_MINUTES,
    )

    print(
        "ELIGIBLE_EVENT_COUNT =",
        len(eligible),
    )

    for event in eligible:

        print(
            "ELIGIBLE =",
            event[
                "checkpoint"
            ],
            event[
                "scheduled_trigger_at_utc"
            ].isoformat(),
            [
                game[
                    "game_id"
                ]
                for game
                in event[
                    "target_games"
                ]
            ],
        )


    if future:

        next_event = min(
            future,
            key=lambda event:
                event[
                    "scheduled_trigger_at_utc"
                ],
        )

        print(
            "\n=== NEXT FUTURE EVENT ==="
        )

        print(
            "NEXT_CHECKPOINT =",
            next_event[
                "checkpoint"
            ],
        )

        print(
            "NEXT_SCHEDULED_UTC =",
            next_event[
                "scheduled_trigger_at_utc"
            ].isoformat(),
        )

        print(
            "NEXT_SCHEDULED_ET =",
            next_event[
                "scheduled_trigger_at_utc"
            ].astimezone(
                ET
            ).isoformat(),
        )

        print(
            "TARGET_GAME_IDS =",
            [
                game[
                    "game_id"
                ]
                for game
                in next_event[
                    "target_games"
                ]
            ],
        )


    print_boundary()

    print(
        "\nDRY_RUN_ONLY=TRUE"
    )

    print(
        "WEATHER_FETCH=NONE"
    )

    print(
        "LEDGER_WRITES=NONE"
    )

    print(
        "SCHEDULER_CHANGES=NONE"
    )

    print()
    print(
        "WEATHER_PROSPECTIVE_CONTROLLER_V1_DRY_RUN=PASS"
    )


# =============================================================================
# EXECUTE
# =============================================================================

def execute(
    season,
    week,
):

    verify_frozen_builders()

    now = utc_now()

    games = load_schedule(
        season,
        week,
    )

    (
        groups,
        events,
    ) = build_events(
        games,
        now,
    )

    print_schedule_summary(
        games,
        groups,
        events,
        now,
    )


    eligible = (
        events_in_execution_window(
            events,
            now,
        )
    )


    print(
        "\n=== EXECUTION AUTHORIZATION ==="
    )

    print(
        "EXECUTION_WINDOW_MINUTES =",
        EXECUTION_WINDOW_MINUTES,
    )

    print(
        "ELIGIBLE_EVENT_COUNT =",
        len(eligible),
    )


    if len(eligible) == 0:

        print(
            "EXECUTION_AUTHORIZED=FALSE"
        )

        print(
            "FAIL_CLOSED_REASON="
            "NO_CHECKPOINT_IN_EXECUTION_WINDOW"
        )

        print(
            "WEATHER_FETCH=NONE"
        )

        print(
            "LEDGER_WRITES=NONE"
        )

        print(
            "WEATHER_PROSPECTIVE_CONTROLLER_"
            "EXECUTE_FAIL_CLOSED=PASS"
        )

        return 3


    if len(eligible) != 1:

        print(
            "EXECUTION_AUTHORIZED=FALSE"
        )

        print(
            "FAIL_CLOSED_REASON="
            "AMBIGUOUS_MULTIPLE_CHECKPOINTS"
        )

        print(
            "WEATHER_FETCH=NONE"
        )

        print(
            "LEDGER_WRITES=NONE"
        )

        return 4


    event = eligible[0]


    existing = ledger_identity_exists(
        season,
        week,
        event,
    )

    if existing is not None:

        print(
            "EXECUTION_AUTHORIZED=FALSE"
        )

        print(
            "FAIL_CLOSED_REASON="
            "CHECKPOINT_ALREADY_LEDGERED"
        )

        print(
            "EXISTING_LEDGER_RUN =",
            existing,
        )

        print(
            "WEATHER_FETCH=NONE"
        )

        print(
            "LEDGER_WRITES=NONE"
        )

        return 5


    # Recheck clock immediately before first write.
    authorization_time = utc_now()

    refreshed_eligible = (
        events_in_execution_window(
            events,
            authorization_time,
        )
    )

    if (
        len(refreshed_eligible) != 1
        or refreshed_eligible[0][
            "checkpoint"
        ] != event[
            "checkpoint"
        ]
        or refreshed_eligible[0][
            "target_kickoff_at_utc"
        ] != event[
            "target_kickoff_at_utc"
        ]
    ):

        print(
            "EXECUTION_AUTHORIZED=FALSE"
        )

        print(
            "FAIL_CLOSED_REASON="
            "AUTHORIZATION_WINDOW_CHANGED"
        )

        print(
            "WEATHER_FETCH=NONE"
        )

        print(
            "LEDGER_WRITES=NONE"
        )

        return 6


    print(
        "EXECUTION_AUTHORIZED=TRUE"
    )

    print(
        "CHECKPOINT =",
        event[
            "checkpoint"
        ],
    )

    print(
        "SCHEDULED_TRIGGER_AT_UTC =",
        event[
            "scheduled_trigger_at_utc"
        ].isoformat(),
    )

    print(
        "TARGET_KICKOFF_AT_UTC =",
        event[
            "target_kickoff_at_utc"
        ].isoformat(),
    )

    print(
        "TARGET_GAME_IDS =",
        [
            game[
                "game_id"
            ]
            for game
            in event[
                "target_games"
            ]
        ],
    )


    before_counts = (
        weather_counts()
    )

    before_captures = (
        capture_inventory(
            season,
            week,
        )
    )


    run_id = None

    try:

        run_id = create_ledger_attempt(
            season,
            week,
            event,
            len(games),
            authorization_time,
        )

        print(
            "LEDGER_RUN_ID =",
            run_id,
        )

        print(
            "LEDGER_STATUS=STARTED"
        )


        run_v1(
            season,
            week,
        )


        after_v1_captures = (
            capture_inventory(
                season,
                week,
            )
        )


        captured_at = (
            resolve_exact_new_capture(
                before_captures,
                after_v1_captures,
                len(games),
            )
        )


        print(
            "EXACT_V1_CAPTURE =",
            captured_at,
        )


        v2 = import_frozen_module(
            "weather_v2_frozen",
            V2_PATH,
        )

        build_exact_v2(
            v2,
            season,
            week,
            captured_at,
            len(games),
        )


        parent_rows = (
            load_exact_v2_parents(
                season,
                week,
                captured_at,
                len(games),
            )
        )


        v3 = import_frozen_module(
            "weather_v3_frozen",
            V3_PATH,
        )


        build_exact_v3(
            v3,
            parent_rows,
            len(games),
        )


        verify_exact_lineage(
            season,
            week,
            captured_at,
            len(games),
        )


        mark_ledger_success(
            run_id,
            captured_at,
        )


        print(
            "\nLEDGER_STATUS=SUCCESS"
        )

        print(
            "RUN_ID =",
            run_id,
        )

        print(
            "V1_CAPTURE =",
            captured_at,
        )


        after_counts = (
            weather_counts()
        )


        print(
            "\n=== TABLE COUNT DELTAS ==="
        )

        for table in sorted(
            before_counts
        ):

            print(
                table,
                "BEFORE=",
                before_counts[
                    table
                ],
                "AFTER=",
                after_counts[
                    table
                ],
                "DELTA=",
                after_counts[
                    table
                ]
                - before_counts[
                    table
                ],
            )


        fk = foreign_key_errors()

        print(
            "FOREIGN_KEY_ERRORS =",
            len(fk),
        )

        if fk:

            raise RuntimeError(
                "Foreign-key errors "
                "after capture."
            )


        print_boundary()

        print()
        print(
            "WEATHER_PROSPECTIVE_CAPTURE=PASS"
        )

        return 0


    except Exception as exc:

        print(
            "\nEXECUTION_FAILURE =",
            repr(exc),
        )

        if run_id is not None:

            try:

                mark_ledger_failed(
                    run_id,
                    exc,
                )

                print(
                    "LEDGER_STATUS=FAILED"
                )

            except Exception as ledger_exc:

                print(
                    "LEDGER_FAILURE_UPDATE_ERROR =",
                    repr(
                        ledger_exc
                    ),
                )


        print(
            "WEATHER_PROSPECTIVE_CAPTURE=FAIL"
        )

        return 10


# =============================================================================
# MAIN
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Prospective NFL weather "
            "capture controller."
        )
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


    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )


    return parser.parse_args()


def main():

    args = parse_args()

    print(
        "=" * 100
    )

    print(
        " WEATHER PROSPECTIVE CONTROLLER V1"
    )

    print(
        "=" * 100
    )

    print(
        "CONTROLLER_VERSION =",
        CONTROLLER_VERSION,
    )

    print(
        "SEASON =",
        args.season,
    )

    print(
        "WEEK =",
        args.week,
    )

    print(
        "MODE =",
        (
            "EXECUTE"
            if args.execute
            else "DRY_RUN"
        ),
    )


    try:

        if args.dry_run:

            dry_run(
                args.season,
                args.week,
            )

            return 0


        return execute(
            args.season,
            args.week,
        )

    except Exception as exc:

        print(
            "\nCONTROLLER_FATAL_ERROR =",
            repr(exc),
        )

        print(
            "FAIL_CLOSED=TRUE"
        )

        return 20


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
