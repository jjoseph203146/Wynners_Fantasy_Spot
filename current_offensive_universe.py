"""Read-only offensive membership, independent of availability and forecasts.

Selection boundary: current_slate_features.load_current_depth / attach_identity /
attach_current_roster_position, before build_current_rows computes any features.
Use the same offensive depth roles and roster-first canonical position authority,
but require exact GSIS and reject duplicates rather than selecting a name/ESPN row.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pandas as pd

from wfs_schedule_context import resolve_schedule_week_context


POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
DEPTH_ROLES = POSITIONS | {"FB"}
MAX_DEPTH_AGE = timedelta(hours=48)
COLUMNS = [
    "season", "week", "game_id", "player_id", "player_name",
    "team", "opponent_team", "position",
]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.fillna("").astype(str).apply(lambda col: col.str.strip())


def resolve_current_offensive_universe(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return identity only; no membership decision establishes ACTIVE or OUT.

    Latest per-team depth must be <=48h old and not future-dated. A completely
    empty depth slot (both name and GSIS absent) is not a player. Named players
    without GSIS, duplicate GSIS, missing team depth or identity contradictions
    fail closed. Roster-only or old-depth players are not introduced.

    When ANY target-week roster exists, all scheduled teams require coverage.
    Membership is the intersection of exact current offensive roster identities
    and current depth on the SAME team. Unmatched depth evidence is excluded,
    without assigning availability; ambiguous roster evidence still fails.
    Canonical roster
    position wins explicitly over depth role / identity position (e.g. FB -> RB).
    Only a wholly absent weekly snapshot permits exact depth + player_identity
    team AND canonical position agreement, with HB normalized to RB. No prior
    roster rows or availability columns are read.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise RuntimeError("OFFENSIVE_UNIVERSE_NOW_REQUIRES_TIMEZONE")
    db_path = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    context = resolve_schedule_week_context(db_path=db_path, as_of=now)
    if (int(season), int(week)) != (context.season, context.planning_week):
        raise RuntimeError("OFFENSIVE_UNIVERSE_WRONG_PLANNING_SEASON_WEEK")

    games = _clean(pd.read_sql_query(
        "SELECT game_id, home_team, away_team FROM games "
        "WHERE season=? AND week=? AND game_type='REG'",
        conn, params=(season, week),
    ))
    prefix = f"{int(season)}_{int(week):02d}_"
    if games.empty or games.game_id.duplicated().any() or not games.game_id.str.startswith(prefix).all():
        raise RuntimeError("OFFENSIVE_UNIVERSE_WRONG_WEEK_OR_DUPLICATE_SCHEDULE")
    targets = []
    for game in games.itertuples(index=False):
        if not game.home_team or not game.away_team or game.home_team == game.away_team:
            raise RuntimeError("OFFENSIVE_UNIVERSE_INVALID_SCHEDULE_TEAM")
        targets.extend([
            (game.home_team, game.game_id, game.away_team),
            (game.away_team, game.game_id, game.home_team),
        ])
    schedule = pd.DataFrame(targets, columns=["team", "game_id", "opponent_team"])
    if schedule.team.duplicated().any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_AMBIGUOUS_TEAM_GAME")
    teams = set(schedule.team)

    # Select the latest team snapshot before filtering roles: otherwise an old
    # offensive row could survive after disappearing from current depth.
    depth = _clean(pd.read_sql_query(
        "SELECT d.gsis_id, d.player_name, d.team, d.pos_abb, d.snapshot_dt "
        "FROM depth_charts d JOIN "
        "(SELECT team, MAX(snapshot_dt) snapshot_dt FROM depth_charts GROUP BY team) s "
        "ON d.team=s.team AND d.snapshot_dt=s.snapshot_dt",
        conn,
    ))
    depth = depth[depth.team.isin(teams)].copy()
    if set(depth.team) != teams:
        raise RuntimeError("OFFENSIVE_UNIVERSE_MISSING_TEAM_DEPTH")
    timestamps = pd.to_datetime(depth.snapshot_dt, errors="coerce", utc=True)
    age = pd.Timestamp(now) - timestamps
    if timestamps.isna().any() or (age > MAX_DEPTH_AGE).any() or (age < timedelta(0)).any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_STALE_OR_INVALID_DEPTH_SNAPSHOT")
    depth = depth[depth.pos_abb.isin(DEPTH_ROLES)].copy()
    vacant = depth.gsis_id.eq("") & depth.player_name.eq("")
    vacant_count = int(vacant.sum())
    depth = depth[~vacant].copy()
    if depth.gsis_id.eq("").any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_NAMED_DEPTH_PLAYER_WITHOUT_GSIS")
    if depth.gsis_id.duplicated().any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_DUPLICATE_DEPTH_GSIS")
    if set(depth.team) != teams:
        raise RuntimeError("OFFENSIVE_UNIVERSE_MISSING_OFFENSIVE_TEAM_DEPTH")

    identities = _clean(pd.read_sql_query(
        "SELECT gsis_id, full_name, latest_team, position FROM player_identity", conn,
    ))
    identities = identities[identities.gsis_id.isin(depth.gsis_id)]
    if identities.gsis_id.duplicated().any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_DUPLICATE_PLAYER_IDENTITY_GSIS")
    identity_map = identities.set_index("gsis_id")

    roster = _clean(pd.read_sql_query(
        "SELECT gsis_id, team, position FROM weekly_rosters WHERE season=? AND week=?",
        conn, params=(season, week),
    ))
    roster_absent = roster.empty
    if not roster_absent and not teams.issubset(set(roster.team)):
        raise RuntimeError("OFFENSIVE_UNIVERSE_PARTIAL_TARGET_WEEK_ROSTER_TEAMS")
    if roster.gsis_id.duplicated().any():
        raise RuntimeError("OFFENSIVE_UNIVERSE_DUPLICATE_CURRENT_ROSTER_GSIS")
    selected_roster = roster[
        roster.position.isin(POSITIONS) | roster.gsis_id.isin(depth.gsis_id)
    ]
    if not roster_absent and not set(selected_roster.team).issubset(teams):
        raise RuntimeError("OFFENSIVE_UNIVERSE_INVALID_CURRENT_ROSTER_SCHEDULE_TEAM")
    if not selected_roster.gsis_id.str.fullmatch(r"00-\d{7}").all():
        raise RuntimeError("OFFENSIVE_UNIVERSE_MALFORMED_CURRENT_ROSTER_GSIS")
    if not selected_roster.position.isin(POSITIONS).all():
        raise RuntimeError("OFFENSIVE_UNIVERSE_INVALID_CURRENT_ROSTER_POSITION")
    roster_map = selected_roster.set_index("gsis_id")

    excluded_depth = []
    if not roster_absent:
        # Current roster membership gates depth evidence, never the reverse.
        # No status column participates, and no exclusion implies a hard state.
        keep = []
        for player in depth.itertuples(index=False):
            roster_team = (
                roster_map.loc[player.gsis_id, "team"]
                if player.gsis_id in roster_map.index else ""
            )
            admitted = roster_team == player.team
            keep.append(admitted)
            if not admitted:
                excluded_depth.append(dict(
                    player_id=player.gsis_id, depth_team=player.team,
                    roster_team=roster_team,
                    reason=("CURRENT_ROSTER_TEAM_MISMATCH" if roster_team
                            else "ABSENT_FROM_CURRENT_ROSTER"),
                ))
        depth = depth.loc[keep].copy()
    if not set(depth.gsis_id).issubset(identity_map.index):
        raise RuntimeError("OFFENSIVE_UNIVERSE_MISSING_EXACT_GSIS_IDENTITY")
    if depth.empty:
        raise RuntimeError("OFFENSIVE_UNIVERSE_EMPTY_ROSTER_DEPTH_INTERSECTION")

    rows, errors = [], []
    position_overrides = []
    for player in depth.itertuples(index=False):
        identity = identity_map.loc[player.gsis_id]
        if not identity.full_name or identity.latest_team != player.team:
            errors.append(f"IDENTITY_TEAM_OR_NAME_CONTRADICTION:{player.gsis_id}")
            continue
        if not roster_absent:
            current = roster_map.loc[player.gsis_id]
            position = current.position
            if position != player.pos_abb or position != identity.position:
                position_overrides.append(player.gsis_id)
        else:
            position = {"HB": "RB"}.get(identity.position, identity.position)
            if position not in POSITIONS or position != player.pos_abb:
                errors.append(f"ABSENT_ROSTER_POSITION_CONTRADICTION:{player.gsis_id}")
                continue
        rows.append(dict(
            season=int(season), week=int(week), player_id=player.gsis_id,
            player_name=identity.full_name, team=player.team, position=position,
        ))
    if errors:
        raise RuntimeError("OFFENSIVE_UNIVERSE_FAIL_CLOSED: " + "; ".join(sorted(errors)))
    result = pd.DataFrame(rows).merge(schedule, on="team", validate="many_to_one")
    result = result[COLUMNS].sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)
    result.attrs["membership_evidence"] = dict(
        roster_snapshot="WHOLE_WEEK_ABSENT" if roster_absent else "CURRENT_WEEK_PRESENT",
        depth_snapshots=sorted(depth.snapshot_dt.unique().tolist()),
        vacant_depth_slots=vacant_count,
        roster_position_override_ids=sorted(position_overrides),
        excluded_depth_identities=sorted(excluded_depth, key=lambda row: row["player_id"]),
    )
    return result
