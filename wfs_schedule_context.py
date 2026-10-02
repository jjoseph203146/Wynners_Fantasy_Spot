from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sqlite3
from typing import Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "data" / "nfl.db"


class ScheduleContextError(RuntimeError):
    """Raised when schedule state cannot be resolved safely."""


@dataclass(frozen=True)
class ScheduleWeekContext:
    season: int
    active_game_week: Optional[int]
    planning_week: int
    upcoming_week: Optional[int]


def _coerce_date(value: Optional[date | datetime]) -> date:
    if value is None:
        return datetime.now().astimezone().date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    raise ScheduleContextError(
        f"Unsupported as_of value: {type(value).__name__}"
    )


def resolve_schedule_week_context(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    season: Optional[int] = None,
    as_of: Optional[date | datetime] = None,
) -> ScheduleWeekContext:
    """
    Resolve NFL regular-season lifecycle context from the schedule database.

    Contract:
      - schedule is authoritative
      - no file-mtime inference
      - no hardcoded NFL week
      - planning week is the schedule week containing as_of, otherwise
        the nearest future scheduled week
      - an unfinished schedule week whose date window contains as_of
        is the active game week
      - upcoming_week is the nearest scheduled week at or after as_of
      - ambiguous or missing schedule state fails closed
    """

    db_path = Path(db_path)

    if not db_path.is_file():
        raise ScheduleContextError(
            f"Schedule database not found: {db_path}"
        )

    today = _coerce_date(as_of)

    db_uri = f"file:{db_path}?mode=ro"

    with sqlite3.connect(
        db_uri,
        uri=True,
    ) as conn:
        conn.row_factory = sqlite3.Row

        cols = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(games)"
            ).fetchall()
        }

        required = {
            "season",
            "week",
            "game_type",
            "game_date",
            "completed",
        }

        missing = required - cols
        if missing:
            raise ScheduleContextError(
                "Schedule database missing required columns: "
                + ", ".join(sorted(missing))
            )

        if season is None:
            season_rows = conn.execute(
                """
                SELECT DISTINCT season
                FROM games
                WHERE game_type = 'REG'
                  AND game_date IS NOT NULL
                  AND game_date != ''
                ORDER BY season
                """
            ).fetchall()

            candidates = []

            for row in season_rows:
                candidate = int(row["season"])

                bounds = conn.execute(
                    """
                    SELECT
                        MIN(game_date) AS first_date,
                        MAX(game_date) AS last_date
                    FROM games
                    WHERE season = ?
                      AND game_type = 'REG'
                    """,
                    (candidate,),
                ).fetchone()

                if not bounds:
                    continue

                first_date = date.fromisoformat(bounds["first_date"])
                last_date = date.fromisoformat(bounds["last_date"])

                if first_date <= today <= last_date:
                    candidates.append(candidate)

            if len(candidates) != 1:
                raise ScheduleContextError(
                    "Unable to resolve one authoritative NFL season "
                    f"for {today.isoformat()}: {candidates}"
                )

            season = candidates[0]

        season = int(season)

        rows = conn.execute(
            """
            SELECT
                week,
                MIN(game_date) AS first_date,
                MAX(game_date) AS last_date,
                COUNT(*) AS game_count,
                SUM(
                    CASE
                        WHEN completed = 1 THEN 1
                        ELSE 0
                    END
                ) AS completed_count
            FROM games
            WHERE season = ?
              AND game_type = 'REG'
              AND week IS NOT NULL
              AND game_date IS NOT NULL
              AND game_date != ''
            GROUP BY week
            ORDER BY week
            """,
            (season,),
        ).fetchall()

    if not rows:
        raise ScheduleContextError(
            f"No regular-season schedule rows for season {season}"
        )

    weeks = []

    for row in rows:
        week = int(row["week"])
        first_date = date.fromisoformat(row["first_date"])
        last_date = date.fromisoformat(row["last_date"])
        game_count = int(row["game_count"])
        completed_count = int(row["completed_count"] or 0)

        if game_count <= 0:
            raise ScheduleContextError(
                f"Week {week} has invalid game count"
            )

        if completed_count < 0 or completed_count > game_count:
            raise ScheduleContextError(
                f"Week {week} has invalid completion state"
            )

        weeks.append(
            {
                "week": week,
                "first_date": first_date,
                "last_date": last_date,
                "games": game_count,
                "completed": completed_count,
            }
        )

    containing = [
        row
        for row in weeks
        if row["first_date"] <= today <= row["last_date"]
    ]

    if len(containing) > 1:
        raise ScheduleContextError(
            f"Overlapping NFL week windows for {today.isoformat()}: "
            f"{[row['week'] for row in containing]}"
        )

    future = [
        row
        for row in weeks
        if row["first_date"] > today
    ]

    active_game_week: Optional[int] = None

    if containing:
        current = containing[0]

        planning_week = int(current["week"])

        if current["completed"] < current["games"]:
            active_game_week = int(current["week"])

        later = [
            row
            for row in weeks
            if row["week"] > current["week"]
        ]

        upcoming_week = (
            int(later[0]["week"])
            if later
            else None
        )

    else:
        if not future:
            raise ScheduleContextError(
                f"No current or future REG week for "
                f"{season} on {today.isoformat()}"
            )

        next_week = min(
            future,
            key=lambda row: (
                row["first_date"],
                row["week"],
            ),
        )

        planning_week = int(next_week["week"])
        upcoming_week = int(next_week["week"])

    return ScheduleWeekContext(
        season=season,
        active_game_week=active_game_week,
        planning_week=planning_week,
        upcoming_week=upcoming_week,
    )
