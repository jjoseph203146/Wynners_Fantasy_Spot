#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


VERSION = "INJURY_PROSPECTIVE_CHECKPOINT_V1"

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"
SNAPSHOT_SCRIPT = ROOT / "scripts" / "capture_injury_prospective_snapshot_v1.py"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

CHECKPOINTS = (
    ("T_MINUS_72H", 72),
    ("T_MINUS_24H", 24),
    ("T_MINUS_3H", 3),
)

# Controller is intended to be run periodically.
# A checkpoint may execute only inside this forward window.
# Once outside it, it is explicitly MISSED and is never backfilled.
DUE_WINDOW_MINUTES = 60


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def parse_now_utc(value: str | None) -> datetime:
    if value is None:
        return utc_now()

    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        raise RuntimeError(
            "--now-utc must include UTC offset or Z"
        )

    return dt.astimezone(UTC)


def parse_kickoff(
    game_date,
    gametime,
) -> tuple[datetime, datetime]:
    date_text = str(game_date).strip()
    time_text = str(gametime).strip()

    if (
        not date_text
        or date_text.lower() in {"nan", "none"}
        or not time_text
        or time_text.lower() in {"nan", "none"}
    ):
        raise RuntimeError(
            "Kickoff unavailable: "
            f"game_date={game_date!r} "
            f"gametime={gametime!r}"
        )

    naive = datetime.fromisoformat(
        f"{date_text}T{time_text}"
    )

    kickoff_et = naive.replace(
        tzinfo=ET
    )

    kickoff_utc = kickoff_et.astimezone(
        UTC
    )

    return kickoff_et, kickoff_utc


def connect_rw() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def connect_ro() -> sqlite3.Connection:
    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )
    con.row_factory = sqlite3.Row
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS injury_prospective_capture_runs_v1 (
            run_id TEXT PRIMARY KEY,
            controller_version TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            checkpoint TEXT NOT NULL,
            checkpoint_hours INTEGER NOT NULL,
            scheduled_trigger_at_utc TEXT NOT NULL,
            target_kickoff_at_utc TEXT NOT NULL,
            target_game_count INTEGER NOT NULL,
            weekly_schedule_game_count INTEGER NOT NULL,
            capture_scope TEXT NOT NULL,
            evaluation_scope TEXT NOT NULL,
            attempt_started_at_utc TEXT,
            capture_completed_at_utc TEXT,
            injury_capture_id TEXT,
            status TEXT NOT NULL,
            failure_reason TEXT,
            kickoff_timezone_contract TEXT NOT NULL,
            historical_reconstruction INTEGER NOT NULL
                CHECK (historical_reconstruction = 0),
            forecast_mutation INTEGER NOT NULL
                CHECK (forecast_mutation = 0),
            player_projection_mutation INTEGER NOT NULL
                CHECK (player_projection_mutation = 0),
            solver_influence INTEGER NOT NULL
                CHECK (solver_influence = 0),
            wfs_influence INTEGER NOT NULL
                CHECK (wfs_influence = 0),
            production_influence_allowed INTEGER NOT NULL
                CHECK (production_influence_allowed = 0),
            UNIQUE (
                season,
                week,
                checkpoint,
                target_kickoff_at_utc
            )
        );

        CREATE TABLE IF NOT EXISTS injury_prospective_capture_targets_v1 (
            run_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            eligible_for_checkpoint_evaluation INTEGER NOT NULL
                CHECK (eligible_for_checkpoint_evaluation = 1),
            PRIMARY KEY (
                run_id,
                game_id
            ),
            FOREIGN KEY (
                run_id
            )
            REFERENCES injury_prospective_capture_runs_v1 (
                run_id
            )
            ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS
            idx_injury_prospective_runs_v1_lookup
        ON injury_prospective_capture_runs_v1 (
            season,
            week,
            checkpoint,
            target_kickoff_at_utc
        );

        CREATE INDEX IF NOT EXISTS
            idx_injury_prospective_targets_v1_game
        ON injury_prospective_capture_targets_v1 (
            game_id
        );
        """
    )


def load_schedule(
    season: int,
    week: int,
) -> list[dict]:
    con = connect_ro()

    try:
        rows = con.execute(
            """
            SELECT
                game_id,
                season,
                week,
                game_type,
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
            (season, week),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        raise RuntimeError(
            f"No schedule rows for season={season} week={week}"
        )

    games = []
    seen = set()

    for row in rows:
        game_id = str(row["game_id"])

        if game_id in seen:
            raise RuntimeError(
                f"Duplicate schedule game_id: {game_id}"
            )

        seen.add(game_id)

        kickoff_et, kickoff_utc = parse_kickoff(
            row["game_date"],
            row["gametime"],
        )

        games.append(
            {
                "game_id": game_id,
                "season": int(row["season"]),
                "week": int(row["week"]),
                "game_type": row["game_type"],
                "game_date": row["game_date"],
                "gametime": row["gametime"],
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "completed": int(row["completed"] or 0),
                "kickoff_et": kickoff_et,
                "kickoff_utc": kickoff_utc,
            }
        )

    return games


def build_events(
    games: list[dict],
    now: datetime,
) -> list[dict]:
    groups = defaultdict(list)

    for game in games:
        groups[game["kickoff_utc"]].append(game)

    events = []

    due_window = timedelta(
        minutes=DUE_WINDOW_MINUTES
    )

    for kickoff_utc in sorted(groups):
        group_games = sorted(
            groups[kickoff_utc],
            key=lambda x: x["game_id"],
        )

        for checkpoint, checkpoint_hours in CHECKPOINTS:
            scheduled = kickoff_utc - timedelta(
                hours=checkpoint_hours
            )

            due_until = scheduled + due_window

            if now < scheduled:
                state = "FUTURE"
            elif scheduled <= now < due_until and now < kickoff_utc:
                state = "DUE"
            elif now < kickoff_utc:
                state = "MISSED"
            else:
                state = "PAST"

            events.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_hours": checkpoint_hours,
                    "scheduled_trigger_at_utc": scheduled,
                    "target_kickoff_at_utc": kickoff_utc,
                    "target_games": group_games,
                    "state": state,
                }
            )

    events.sort(
        key=lambda e: (
            e["scheduled_trigger_at_utc"],
            e["target_kickoff_at_utc"],
            e["checkpoint_hours"],
        )
    )

    return events


def existing_run(
    con: sqlite3.Connection,
    season: int,
    week: int,
    event: dict,
):
    return con.execute(
        """
        SELECT *
        FROM injury_prospective_capture_runs_v1
        WHERE season = ?
          AND week = ?
          AND checkpoint = ?
          AND target_kickoff_at_utc = ?
        """,
        (
            season,
            week,
            event["checkpoint"],
            iso_utc(event["target_kickoff_at_utc"]),
        ),
    ).fetchone()


def insert_noncapture_event(
    con: sqlite3.Connection,
    season: int,
    week: int,
    weekly_game_count: int,
    event: dict,
    status: str,
) -> str:
    run_id = (
        "injury_checkpoint_v1_"
        + uuid.uuid4().hex
    )

    con.execute(
        """
        INSERT INTO injury_prospective_capture_runs_v1 (
            run_id,
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
            capture_completed_at_utc,
            injury_capture_id,
            status,
            failure_reason,
            kickoff_timezone_contract,
            historical_reconstruction,
            forecast_mutation,
            player_projection_mutation,
            solver_influence,
            wfs_influence,
            production_influence_allowed
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, NULL, NULL, NULL, ?, NULL,
            ?, 0, 0, 0, 0, 0, 0
        )
        """,
        (
            run_id,
            VERSION,
            season,
            week,
            event["checkpoint"],
            event["checkpoint_hours"],
            iso_utc(event["scheduled_trigger_at_utc"]),
            iso_utc(event["target_kickoff_at_utc"]),
            len(event["target_games"]),
            weekly_game_count,
            "WEEKLY_INJURY_CONSENSUS_SNAPSHOT",
            "TARGET_KICKOFF_GROUP_ONLY",
            status,
            "games.game_date+gametime interpreted "
            "America/New_York DST-aware then converted UTC",
        ),
    )

    return run_id


def snapshot_capture_ids(
    con: sqlite3.Connection,
    season: int,
    week: int,
) -> set[str]:
    rows = con.execute(
        """
        SELECT capture_id
        FROM injury_prospective_capture_v1
        WHERE season = ?
          AND week = ?
        """,
        (season, week),
    ).fetchall()

    return {
        str(row["capture_id"])
        for row in rows
    }


def run_snapshot(
    season: int,
    week: int,
) -> tuple[str, str]:
    before_con = connect_ro()

    try:
        before = snapshot_capture_ids(
            before_con,
            season,
            week,
        )
    finally:
        before_con.close()

    cmd = [
        sys.executable,
        str(SNAPSHOT_SCRIPT),
        "--season",
        str(season),
        "--week",
        str(week),
    ]

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    output = (
        (proc.stdout or "")
        + (proc.stderr or "")
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Snapshot V1 failed.\n"
            + output[-5000:]
        )

    after_con = connect_ro()

    try:
        after = snapshot_capture_ids(
            after_con,
            season,
            week,
        )
    finally:
        after_con.close()

    created = sorted(after - before)

    if len(created) != 1:
        raise RuntimeError(
            "Expected exactly one new immutable injury capture; "
            f"found {len(created)}."
        )

    return created[0], output


def execute_due_event(
    con: sqlite3.Connection,
    season: int,
    week: int,
    weekly_game_count: int,
    event: dict,
) -> str:
    run_id = (
        "injury_checkpoint_v1_"
        + uuid.uuid4().hex
    )

    started = utc_now()

    con.execute(
        """
        INSERT INTO injury_prospective_capture_runs_v1 (
            run_id,
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
            capture_completed_at_utc,
            injury_capture_id,
            status,
            failure_reason,
            kickoff_timezone_contract,
            historical_reconstruction,
            forecast_mutation,
            player_projection_mutation,
            solver_influence,
            wfs_influence,
            production_influence_allowed
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, NULL, NULL, ?, NULL,
            ?, 0, 0, 0, 0, 0, 0
        )
        """,
        (
            run_id,
            VERSION,
            season,
            week,
            event["checkpoint"],
            event["checkpoint_hours"],
            iso_utc(event["scheduled_trigger_at_utc"]),
            iso_utc(event["target_kickoff_at_utc"]),
            len(event["target_games"]),
            weekly_game_count,
            "WEEKLY_INJURY_CONSENSUS_SNAPSHOT",
            "TARGET_KICKOFF_GROUP_ONLY",
            iso_utc(started),
            "ATTEMPTING",
            "games.game_date+gametime interpreted "
            "America/New_York DST-aware then converted UTC",
        ),
    )

    for game in event["target_games"]:
        con.execute(
            """
            INSERT INTO injury_prospective_capture_targets_v1 (
                run_id,
                game_id,
                eligible_for_checkpoint_evaluation
            )
            VALUES (?, ?, 1)
            """,
            (
                run_id,
                game["game_id"],
            ),
        )

    con.commit()

    try:
        capture_id, _ = run_snapshot(
            season,
            week,
        )

        completed = utc_now()

        con.execute(
            """
            UPDATE injury_prospective_capture_runs_v1
            SET
                capture_completed_at_utc = ?,
                injury_capture_id = ?,
                status = 'CAPTURED_SHADOW_ONLY',
                failure_reason = NULL
            WHERE run_id = ?
            """,
            (
                iso_utc(completed),
                capture_id,
                run_id,
            ),
        )

        con.commit()

        return capture_id

    except Exception as exc:
        completed = utc_now()

        con.execute(
            """
            UPDATE injury_prospective_capture_runs_v1
            SET
                capture_completed_at_utc = ?,
                status = 'FAILED',
                failure_reason = ?
            WHERE run_id = ?
            """,
            (
                iso_utc(completed),
                str(exc)[:4000],
                run_id,
            ),
        )

        con.commit()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()

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
        "--now-utc",
        default=None,
        help=(
            "Optional aware UTC timestamp for deterministic "
            "dry-run/state validation."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute DUE checkpoint captures. "
            "Without this flag the controller is read-only "
            "apart from schema creation."
        ),
    )

    args = parser.parse_args()

    now = parse_now_utc(
        args.now_utc
    )

    games = load_schedule(
        args.season,
        args.week,
    )

    events = build_events(
        games,
        now,
    )

    print("=" * 100)
    print("INJURY PROSPECTIVE CHECKPOINT CONTROLLER V1")
    print("=" * 100)
    print("VERSION =", VERSION)
    print("SEASON =", args.season)
    print("WEEK =", args.week)
    print("NOW_UTC =", iso_utc(now))
    print("WEEKLY_GAMES =", len(games))
    print("TIMEZONE = America/New_York")
    print("DUE_WINDOW_MINUTES =", DUE_WINDOW_MINUTES)
    print("EXECUTE =", bool(args.execute))

    print("\nCHECKPOINT EVENTS:")

    for event in events:
        ids = ",".join(
            game["game_id"]
            for game in event["target_games"]
        )

        print(
            event["checkpoint"],
            "| scheduled=",
            iso_utc(event["scheduled_trigger_at_utc"]),
            "| kickoff=",
            iso_utc(event["target_kickoff_at_utc"]),
            "| games=",
            len(event["target_games"]),
            "| state=",
            event["state"],
            "|",
            ids,
        )

    if not args.execute:
        print("\nSTATUS: DRY_RUN_COMPLETE")
        print("NO_CAPTURE_EXECUTED=TRUE")
        print("PRODUCTION_INFLUENCE_ALLOWED=FALSE")
        return 0

    con = connect_rw()

    try:
        ensure_schema(con)
        con.commit()

        captured = 0
        missed_recorded = 0
        already_recorded = 0

        for event in events:
            current = existing_run(
                con,
                args.season,
                args.week,
                event,
            )

            if current is not None:
                already_recorded += 1
                continue

            if event["state"] == "DUE":
                capture_id = execute_due_event(
                    con,
                    args.season,
                    args.week,
                    len(games),
                    event,
                )

                captured += 1

                print(
                    "\nCAPTURED:",
                    event["checkpoint"],
                    iso_utc(event["target_kickoff_at_utc"]),
                    "capture_id=",
                    capture_id,
                )

            elif event["state"] == "MISSED":
                run_id = insert_noncapture_event(
                    con,
                    args.season,
                    args.week,
                    len(games),
                    event,
                    "MISSED_NO_BACKFILL",
                )

                for game in event["target_games"]:
                    con.execute(
                        """
                        INSERT INTO injury_prospective_capture_targets_v1 (
                            run_id,
                            game_id,
                            eligible_for_checkpoint_evaluation
                        )
                        VALUES (?, ?, 1)
                        """,
                        (
                            run_id,
                            game["game_id"],
                        ),
                    )

                con.commit()
                missed_recorded += 1

        print("\nEXECUTION SUMMARY:")
        print("captured =", captured)
        print("missed_recorded =", missed_recorded)
        print("already_recorded =", already_recorded)

        print("\nSAFETY:")
        print("historical_reconstruction=FALSE")
        print("forecast_mutation=FALSE")
        print("player_projection_mutation=FALSE")
        print("solver_influence=FALSE")
        print("wfs_influence=FALSE")
        print("production_influence_allowed=FALSE")

        print("\nSTATUS: PASS_SHADOW_ONLY")

        return 0

    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
