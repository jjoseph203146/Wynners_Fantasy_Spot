#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def derive_schedule_authority(
    root: Path,
    record: Callable[[str, str, str], None],
    *,
    today_utc: str | None = None,
) -> dict | None:
    """
    Derive the current NFL production schedule target from authoritative
    schedule state.

    Read-only.

    Contract:
      - REG games only.
      - A week is active when it contains unfinished games and at least
        one game in that week is scheduled on or before the current UTC
        calendar date.
      - Use the latest season containing an active week.
      - Within that season, choose the earliest active week.
      - Future staged weeks must not advance production while the current
        active week remains unfinished.
      - Completed games inside the active week remain valid and are counted
        separately from the unfinished production target.
    """

    root = Path(root)
    nfl_db = root / "data" / "nfl.db"

    if today_utc is None:
        today_utc = datetime.now(timezone.utc).date().isoformat()

    if not nfl_db.is_file():
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            f"missing schedule authority: {nfl_db}",
        )
        return None

    con = sqlite3.connect(
        f"file:{nfl_db}?mode=ro",
        uri=True,
    )
    con.row_factory = sqlite3.Row

    try:
        rows = con.execute(
            """
            SELECT
                game_id,
                season,
                game_type,
                week,
                game_date,
                gametime,
                away_team,
                home_team,
                completed
            FROM games
            WHERE UPPER(COALESCE(game_type, '')) = 'REG'
              AND season IS NOT NULL
              AND week IS NOT NULL
            ORDER BY season, week, game_date, gametime, game_id
            """
        ).fetchall()
    except Exception as exc:
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            (
                "schedule_read_error="
                f"{type(exc).__name__}:{exc}"
            ),
        )
        return None
    finally:
        con.close()

    if not rows:
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            "no REG schedule rows",
        )
        return None

    grouped: dict[
        tuple[int, int],
        list[sqlite3.Row],
    ] = {}

    for row in rows:
        key = (
            int(row["season"]),
            int(row["week"]),
        )
        grouped.setdefault(key, []).append(row)

    candidates = []

    for (season, week), week_rows in grouped.items():
        unfinished = [
            row
            for row in week_rows
            if int(row["completed"] or 0) == 0
        ]

        if not unfinished:
            continue

        dated_rows = [
            row
            for row in week_rows
            if row["game_date"]
        ]

        if not dated_rows:
            continue

        first_game_date = min(
            str(row["game_date"])
            for row in dated_rows
        )

        if first_game_date <= today_utc:
            candidates.append(
                (
                    season,
                    week,
                    week_rows,
                )
            )

    if not candidates:
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            f"no active REG week as of {today_utc}",
        )
        return None

    latest_season = max(
        item[0]
        for item in candidates
    )

    same_season = [
        item
        for item in candidates
        if item[0] == latest_season
    ]

    same_season.sort(
        key=lambda item: item[1]
    )

    season, week, active_week_rows = same_season[0]

    unfinished_rows = [
        row
        for row in active_week_rows
        if int(row["completed"] or 0) == 0
    ]

    completed_rows = [
        row
        for row in active_week_rows
        if int(row["completed"] or 0) == 1
    ]

    unfinished_game_ids = {
        str(row["game_id"])
        for row in unfinished_rows
    }

    unfinished_teams = set()

    for row in unfinished_rows:
        if row["away_team"]:
            unfinished_teams.add(
                str(row["away_team"])
            )

        if row["home_team"]:
            unfinished_teams.add(
                str(row["home_team"])
            )

    all_game_ids = [
        str(row["game_id"])
        for row in active_week_rows
    ]

    if len(all_game_ids) != len(set(all_game_ids)):
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            (
                f"duplicate game_id "
                f"season={season} week={week}"
            ),
        )
        return None

    if len(unfinished_game_ids) != len(unfinished_rows):
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            (
                f"duplicate unfinished game_id "
                f"season={season} week={week}"
            ),
        )
        return None

    expected_team_count = len(unfinished_rows) * 2

    if len(unfinished_teams) != expected_team_count:
        record(
            "SCHEDULE_ACTIVE_WEEK",
            "FAIL",
            (
                f"season={season} week={week} "
                f"unfinished_games={len(unfinished_rows)} "
                f"unique_teams={len(unfinished_teams)} "
                f"expected_teams={expected_team_count}"
            ),
        )
        return None

    future_weeks = sorted(
        {
            w
            for (s, w) in grouped
            if s == season and w > week
        }
    )

    next_staged_week = (
        future_weeks[0]
        if future_weeks
        else None
    )

    record(
        "SCHEDULE_ACTIVE_WEEK",
        "PASS",
        (
            f"season={season} week={week} "
            f"unfinished_games={len(unfinished_rows)} "
            f"unfinished_teams={len(unfinished_teams)} "
            f"completed_same_week={len(completed_rows)} "
            f"next_staged_week={next_staged_week}"
        ),
    )

    return {
        "season": season,
        "week": week,
        "unfinished_games": len(unfinished_rows),
        "unfinished_teams": len(unfinished_teams),
        "completed_same_week": len(completed_rows),
        "next_staged_week": next_staged_week,
        "unfinished_game_ids": unfinished_game_ids,
    }
