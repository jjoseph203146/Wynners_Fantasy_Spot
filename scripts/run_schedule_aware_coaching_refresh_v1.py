#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"
REFRESH = ROOT / "scripts" / "run_coaching_intelligence_refresh_v1.py"

VERSION = "WFS_SCHEDULE_AWARE_COACHING_REFRESH_V1"

ET = ZoneInfo("America/New_York")

PRE_KICKOFF = timedelta(minutes=90)
POST_KICKOFF_WINDOW = timedelta(hours=5, minutes=30)


def parse_kickoff(game_date: str, gametime: str) -> datetime:
    value = f"{game_date} {gametime}"
    naive = datetime.strptime(
        value,
        "%Y-%m-%d %H:%M",
    )
    return naive.replace(tzinfo=ET)


def main() -> int:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if not NFL_DB.is_file():
        print("SKIP: nfl.db unavailable")
        return 0

    if not REFRESH.is_file():
        print("FAIL: coaching refresh wrapper missing")
        return 2

    now = datetime.now(ET)

    # Include yesterday/today/tomorrow so late-night cleanup and
    # unusual kickoff windows are handled without hardcoded weekdays.
    date_min = (now.date() - timedelta(days=1)).isoformat()
    date_max = (now.date() + timedelta(days=1)).isoformat()

    with sqlite3.connect(
        f"file:{NFL_DB}?mode=ro",
        uri=True,
    ) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
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
                espn,
                completed
            FROM games
            WHERE game_date BETWEEN ? AND ?
              AND game_date IS NOT NULL
              AND gametime IS NOT NULL
            ORDER BY game_date, gametime, game_id
            """,
            (date_min, date_max),
        ).fetchall()

    if not rows:
        print(
            f"SKIP: no NFL games near {now.strftime('%Y-%m-%d %H:%M %Z')}"
        )
        return 0

    active_windows = []

    for row in rows:
        try:
            kickoff = parse_kickoff(
                str(row["game_date"]),
                str(row["gametime"]),
            )
        except Exception:
            # Malformed schedule rows do not activate the gate.
            continue

        window_start = kickoff - PRE_KICKOFF
        window_end = kickoff + POST_KICKOFF_WINDOW

        if window_start <= now <= window_end:
            active_windows.append(
                {
                    "game_id": str(row["game_id"]),
                    "away": str(row["away_team"]),
                    "home": str(row["home_team"]),
                    "kickoff": kickoff,
                    "completed": int(row["completed"] or 0),
                }
            )

    if not active_windows:
        print(
            f"SKIP: outside NFL game windows at "
            f"{now.strftime('%Y-%m-%d %H:%M %Z')}"
        )
        return 0

    print(
        f"RUN: {len(active_windows)} game window(s) active at "
        f"{now.strftime('%Y-%m-%d %H:%M %Z')}"
    )

    for game in active_windows:
        print(
            f"  {game['away']}@{game['home']} "
            f"{game['kickoff'].strftime('%Y-%m-%d %H:%M %Z')} "
            f"| completed={game['completed']}"
        )

    result = subprocess.run(
        [
            sys.executable,
            str(REFRESH),
        ],
        cwd=str(ROOT),
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": ".",
        },
    )

    if result.returncode != 0:
        print(
            f"FAIL: coaching refresh returned {result.returncode}"
        )
        return result.returncode

    print("PASS: SCHEDULE-AWARE COACHING REFRESH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
