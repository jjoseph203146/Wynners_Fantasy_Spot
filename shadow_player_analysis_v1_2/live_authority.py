"""Explicit, read-only WFS -> V1.2 snapshot bridge. No network or publication.

Incomplete identity rows are excluded; duplicate GSIS IDs (even identical rows)
abort the snapshot. Shared names with distinct IDs are retained for the existing
Authority exact resolver to disambiguate or reject. No game association is added.

Use build_live_authorities(season=2026, week=3) for a current live read. The return
value contains ``identity`` and ``schedule`` arguments for the existing Authority
or build contracts. ``as_of_utc`` is an explicit snapshot-read timestamp override
for deterministic fixtures; it must never be used to assert historical knowledge
of current live rows. By default UTC is captured once after both reads complete.
"""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

from shadow_player_analysis_v1.core import (
    ANALYSIS_ONLY, PRODUCTION_INFLUENCE, SOLVER_INFLUENCE, PROJECTION_MUTATION,
    FORECAST_MUTATION, AVAILABILITY_AUTHORITY, INJURY_AUTHORITY,
    DEPTH_CHART_AUTHORITY, AI_ANALYST_INFLUENCE, DATABASE_MUTATION,
)

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "nfl.db"


def _as_of(value):
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str):
        raise ValueError("SNAPSHOT_UTC_REQUIRED")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if "T" not in value or parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("SNAPSHOT_UTC_REQUIRED")
    return value


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _kickoff(date, time):
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("INVALID_GAME_DATE")
    if not isinstance(time, str) or not re.fullmatch(r"\d{2}:\d{2}", time):
        raise ValueError("INVALID_GAMETIME")
    wall = datetime.fromisoformat(date + "T" + time)
    zone = ZoneInfo("America/New_York")
    candidates = set()
    for fold in (0, 1):
        utc = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if utc.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.add(utc)
    if len(candidates) != 1:
        raise ValueError("AMBIGUOUS_OR_NONEXISTENT_KICKOFF")
    return candidates.pop().isoformat().replace("+00:00", "Z")


def build_live_authorities(*, season, week, db_path=DEFAULT_DB, as_of_utc=None):
    """Return in-memory identity/schedule snapshots, or raise without partial output.

    Requires explicit positive integer season/week. All selected schedule rows,
    including completed games, must have unique IDs and valid Eastern kickoffs.
    Empty authorities fail closed. SQLite mode=ro prevents writes and creation;
    a single read transaction keeps the two table reads consistent.
    """
    if type(season) is not int or season <= 0 or type(week) is not int or week <= 0:
        raise ValueError("EXPLICIT_SEASON_WEEK_REQUIRED")
    if as_of_utc is not None:
        _as_of(as_of_utc)
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("BEGIN")  # Deferred read transaction; no write lock.
        players = connection.execute(
            "SELECT gsis_id, full_name, latest_team, position FROM player_identity"
        ).fetchall()
        games = connection.execute(
            "SELECT game_id, game_date, gametime FROM games WHERE season=? AND week=?",
            (season, week),
        ).fetchall()
        available = _as_of(as_of_utc)

    identities, seen_ids = [], set()
    for gsis, name, team, position in players:
        # Detect duplicate nonblank IDs even if one of their rows is incomplete.
        if _nonblank(gsis):
            if gsis in seen_ids:
                raise ValueError("DUPLICATE_GSIS_IDENTITY")
            seen_ids.add(gsis)
        if all(_nonblank(v) for v in (gsis, name, team, position)):
            identities.append(dict(player_name=name, team=team, position=position, gsis_id=gsis))
    schedule, seen_games = [], set()
    for game_id, date, time in games:
        if not _nonblank(game_id):
            raise ValueError("MISSING_GAME_ID")
        if game_id in seen_games:
            raise ValueError("DUPLICATE_GAME_ID")
        seen_games.add(game_id)
        schedule.append(dict(game_id=game_id, kickoff_at_utc=_kickoff(date, time)))
    if not identities or not schedule:
        raise ValueError("EMPTY_LIVE_AUTHORITY")
    return dict(
        identity=dict(authority="WFS_IDENTITY_SNAPSHOT", available_at_utc=available,
                      season=season, week=week, players=sorted(identities, key=lambda r: r["gsis_id"])),
        schedule=dict(authority="WFS_SCHEDULE_SNAPSHOT", available_at_utc=available,
                      games=sorted(schedule, key=lambda r: r["game_id"])),
    )
