"""
WFS AI Analyst — Deterministic Evidence Engine V1

Purpose
-------
Build read-only, structured evidence packets from existing WFS NFL
data products.

This module:
- does not call an LLM;
- does not create forecasts;
- does not modify projections;
- does not modify NFL Live;
- does not modify nfl.db;
- does not modify forecast_ledger.db;
- does not modify FanDuel pools or lineups.

The evidence packet is the boundary between deterministic WFS data
and any future natural-language analyst layer.
"""

from __future__ import annotations

import math

from pathlib import Path
import json
import re
import sqlite3
from typing import Any

import pandas as pd
from forecast_publication_selector import select_forecast_path


ROOT = Path(__file__).resolve().parent

NFL_DB = ROOT / "data" / "nfl.db"
LIVE_DB = ROOT / "data" / "wfs_live.db"

FORECAST_CSV = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

STAT_FORECAST_CURRENT = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_current_unified_stat_forecasts.parquet"
)

STAT_FORECAST_SNAPSHOT_ROOT = (
    ROOT
    / "data"
    / "forecast_snapshots"
)

OFFENSIVE_POSITIONS = {
    "QB",
    "RB",
    "FB",
    "WR",
    "TE",
}


def _clean(value: Any) -> Any:
    """Convert pandas/numpy null-like values into JSON-safe None."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def _record(row: pd.Series) -> dict[str, Any]:
    return {
        str(key): _clean(value)
        for key, value in row.items()
    }


def _open_ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)

    uri = f"file:{path.resolve()}?mode=ro"

    return sqlite3.connect(
        uri,
        uri=True,
    )


def load_forecast(game_id: str) -> dict[str, Any] | None:
    forecast_csv = select_forecast_path()

    if not forecast_csv.is_file():
        return None

    df = pd.read_csv(
        forecast_csv,
        dtype={"game_id": str},
    )

    rows = df[
        df["game_id"].astype(str).eq(str(game_id))
    ]

    if rows.empty:
        return None

    if len(rows) != 1:
        raise ValueError(
            f"Forecast game_id is not unique: {game_id}"
        )

    return _record(rows.iloc[0])


def _stat_forecast_snapshot_path(
    game_id: str,
) -> Path:
    """
    Return the immutable kickoff snapshot path for one game.

    Snapshot identity is exact game_id. No name/team fuzzy matching
    is permitted.
    """
    return (
        STAT_FORECAST_SNAPSHOT_ROOT
        / str(game_id)
        / "stat_forecast_kickoff.parquet"
    )


def _load_stat_forecast_file_for_game(
    path: Path,
    game_id: str,
) -> list[dict[str, Any]]:
    """
    Read one statistical forecast artifact and return only rows
    belonging to the exact requested game_id.

    This function is read-only and does not execute any model.
    """
    if not path.is_file():
        return []

    df = pd.read_parquet(path)

    if "game_id" not in df.columns:
        raise ValueError(
            f"Stat forecast artifact has no game_id column: {path}"
        )

    rows = df[
        df["game_id"]
        .astype(str)
        .eq(str(game_id))
    ].copy()

    if rows.empty:
        return []

    duplicate_checks = []

    if "entity_type" in rows.columns:
        offense = rows[
            rows["entity_type"].astype(str).eq("OFFENSE_PLAYER")
        ]

        kicker = rows[
            rows["entity_type"].astype(str).eq("KICKER")
        ]

        dst = rows[
            rows["entity_type"].astype(str).eq("DST")
        ]

        if not offense.empty:
            player_col = (
                "player_id"
                if "player_id" in offense.columns
                else "entity_id"
            )

            if player_col in offense.columns:
                duplicate_checks.append(
                    (
                        "OFFENSE_PLAYER",
                        offense.duplicated(
                            subset=["game_id", player_col],
                            keep=False,
                        ).any(),
                    )
                )

        if not kicker.empty:
            player_col = (
                "player_id"
                if "player_id" in kicker.columns
                else "entity_id"
            )

            if player_col in kicker.columns:
                duplicate_checks.append(
                    (
                        "KICKER",
                        kicker.duplicated(
                            subset=["game_id", player_col],
                            keep=False,
                        ).any(),
                    )
                )

        if not dst.empty and "team" in dst.columns:
            duplicate_checks.append(
                (
                    "DST",
                    dst.duplicated(
                        subset=["game_id", "team"],
                        keep=False,
                    ).any(),
                )
            )

    bad = [
        label
        for label, has_dup in duplicate_checks
        if has_dup
    ]

    if bad:
        raise ValueError(
            "Duplicate statistical forecast identity for "
            f"{game_id}: {bad}"
        )

    return [
        _record(row)
        for _, row in rows.iterrows()
    ]


def load_stat_forecast_context(
    game_id: str,
    *,
    mode: str,
    current_path: Path | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    """
    Load read-only statistical forecast context.

    Source policy:
      PREGAME:
        canonical current statistical forecast artifact.

      LIVE / POSTGAME:
        immutable kickoff snapshot only.

    A LIVE/POSTGAME request without a frozen snapshot fails closed:
    no current/post-kickoff candidate is substituted.

    This loader never trains models, changes forecasts, writes a
    database, or performs fuzzy identity matching.
    """
    game_id = str(game_id).strip()
    mode = str(mode or "").strip().upper()

    if not game_id:
        raise ValueError("game_id is required")

    if mode not in {
        "PREGAME",
        "LIVE",
        "POSTGAME",
    }:
        raise ValueError(
            f"Unsupported stat forecast mode: {mode}"
        )

    current_path = (
        Path(current_path)
        if current_path is not None
        else STAT_FORECAST_CURRENT
    )

    snapshot_root = (
        Path(snapshot_root)
        if snapshot_root is not None
        else STAT_FORECAST_SNAPSHOT_ROOT
    )

    if mode == "PREGAME":
        source_path = current_path
        source_kind = "CANONICAL_CURRENT"

    else:
        source_path = (
            snapshot_root
            / game_id
            / "stat_forecast_kickoff.parquet"
        )
        source_kind = "KICKOFF_SNAPSHOT"

    result = {
        "available": False,
        "game_id": game_id,
        "mode": mode,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "frozen_at_kickoff": (
            source_kind == "KICKOFF_SNAPSHOT"
        ),
        "status": "NOT_AVAILABLE",
        "rows": [],
        "row_count": 0,
        "offense_player_count": 0,
        "kicker_count": 0,
        "dst_count": 0,
        "forecast_modified": False,
        "model_execution": False,
        "database_write": False,
        "fuzzy_identity_matching": False,
    }

    if not source_path.is_file():
        if mode in {"LIVE", "POSTGAME"}:
            result["status"] = (
                "KICKOFF_SNAPSHOT_NOT_AVAILABLE"
            )
        else:
            result["status"] = (
                "CANONICAL_CURRENT_NOT_AVAILABLE"
            )

        return result

    rows = _load_stat_forecast_file_for_game(
        source_path,
        game_id,
    )

    if not rows:
        result["status"] = "GAME_NOT_FOUND_IN_ARTIFACT"
        return result

    entity_counts = {
        "OFFENSE_PLAYER": 0,
        "KICKER": 0,
        "DST": 0,
    }

    for row in rows:
        entity_type = str(
            row.get("entity_type") or ""
        )

        if entity_type in entity_counts:
            entity_counts[entity_type] += 1

    result.update({
        "available": True,
        "status": "READY",
        "rows": rows,
        "row_count": len(rows),
        "offense_player_count": (
            entity_counts["OFFENSE_PLAYER"]
        ),
        "kicker_count": (
            entity_counts["KICKER"]
        ),
        "dst_count": (
            entity_counts["DST"]
        ),
    })

    return result


def load_game(game_id: str) -> dict[str, Any] | None:
    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
                game_id,
                season,
                game_type,
                week,
                game_date,
                weekday,
                gametime,
                away_team,
                home_team,
                away_score,
                home_score,
                completed,
                roof,
                surface,
                temp,
                wind,
                away_rest,
                home_rest,
                away_qb_name,
                home_qb_name,
                away_coach,
                home_coach,
                stadium
            FROM games
            WHERE game_id = ?
            """,
            (str(game_id),),
        ).fetchone()

    return dict(row) if row else None


def load_team_context(
    game_id: str,
) -> list[dict[str, Any]]:
    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                game_id,
                team,
                opponent_team,
                history_games,
                points_for_last,
                points_for_avg_3,
                points_for_avg_5,
                points_against_last,
                points_against_avg_3,
                points_against_avg_5,
                offensive_plays_avg_3,
                pass_attempts_avg_3,
                rush_attempts_avg_3,
                pass_rate_avg_3,
                rush_rate_avg_3,
                passing_yards_avg_3,
                rushing_yards_avg_3,
                passing_tds_avg_3,
                rushing_tds_avg_3,
                opponent_points_allowed_avg_3,
                opponent_pass_yards_allowed_avg_3,
                opponent_rush_yards_allowed_avg_3,
                opponent_pass_tds_allowed_avg_3,
                opponent_rush_tds_allowed_avg_3,
                team_scoring_trend,
                opponent_scoring_trend,
                pace_trend,
                pass_rate_trend
            FROM team_pregame_environment
            WHERE game_id = ?
            ORDER BY team
            """,
            (str(game_id),),
        ).fetchall()

    return [dict(row) for row in rows]


def load_player_context(
    game_id: str,
) -> list[dict[str, Any]]:
    placeholders = ",".join(
        "?" for _ in OFFENSIVE_POSITIONS
    )

    params = [
        str(game_id),
        *sorted(OFFENSIVE_POSITIONS),
    ]

    sql = f"""
        SELECT
            game_id,
            player_id,
            player_display_name,
            position,
            team,
            opponent_team,
            report_status,
            practice_status,
            primary_injury,
            active_flag,
            injury_flag,
            history_games,
            fd_last,
            fd_avg_3,
            fd_avg_5,
            fd_max_5,
            fd_std_5,
            opportunities_last,
            opportunities_avg_3,
            opportunities_avg_5,
            touches_last,
            touches_avg_3,
            snaps_last,
            snap_pct_last,
            snap_pct_avg_3,
            targets_last,
            targets_avg_3,
            carries_last,
            carries_avg_3,
            target_share_last,
            target_share_avg_3,
            air_yards_share_last,
            air_yards_share_avg_3,
            wopr_last,
            wopr_avg_3,
            opportunity_trend,
            snap_trend,
            target_trend,
            carry_trend,
            fanduel_trend,
            established_role_flag,
            rising_role_flag,
            declining_role_flag
        FROM player_pregame_features
        WHERE game_id = ?
          AND position IN ({placeholders})
        ORDER BY
            team,
            CASE position
                WHEN 'QB' THEN 1
                WHEN 'RB' THEN 2
                WHEN 'FB' THEN 3
                WHEN 'WR' THEN 4
                WHEN 'TE' THEN 5
                ELSE 9
            END,
            fd_avg_3 DESC,
            player_display_name
    """

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            sql,
            params,
        ).fetchall()

    return [dict(row) for row in rows]


def load_live_context(
    game_id: str,
) -> dict[str, Any]:
    """
    Bind nfl.db.games to wfs_live.db using the exact ESPN event ID.

    Identity contract:
        nfl.db.games.espn == wfs_live.db.live_events.event_id

    Team names are not used to establish identity.
    """
    result = {
        "database_available": LIVE_DB.is_file(),
        "game_id": str(game_id),
        "espn_event_id": None,
        "event_matched": False,
        "live_context_status": "NOT_AVAILABLE",
        "state": None,
        "detail": None,
        "period": None,
        "clock": None,
        "home_team": None,
        "away_team": None,
        "home_score": None,
        "away_score": None,
        "updated_at_utc": None,
        "play_count": 0,
        "recent_plays": [],
    }

    if not LIVE_DB.is_file():
        return result

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row

        game = conn.execute(
            """
            SELECT game_id, espn
            FROM games
            WHERE game_id = ?
            """,
            (str(game_id),),
        ).fetchone()

    if game is None:
        result["live_context_status"] = "GAME_NOT_FOUND"
        return result

    event_id = str(game["espn"] or "").strip()

    if not event_id:
        result["live_context_status"] = "ESPN_EVENT_ID_MISSING"
        return result

    result["espn_event_id"] = event_id

    with _open_ro(LIVE_DB) as conn:
        conn.row_factory = sqlite3.Row

        event_rows = conn.execute(
            """
            SELECT
                event_id,
                season,
                season_type,
                week,
                state,
                detail,
                period,
                clock,
                home_team,
                home_score,
                away_team,
                away_score,
                updated_at_utc
            FROM live_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchall()

        if len(event_rows) > 1:
            raise ValueError(
                f"Live event identity is not unique: {event_id}"
            )

        if not event_rows:
            result["live_context_status"] = "EVENT_NOT_TRACKED"
            return result

        event = event_rows[0]

        play_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM live_plays
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()[0]

        recent = conn.execute(
            """
            SELECT
                play_id,
                sequence_number,
                period,
                clock,
                home_score,
                away_score,
                play_text,
                is_scoring_play,
                is_turnover,
                is_penalty
            FROM live_plays
            WHERE event_id = ?
            ORDER BY sequence_number DESC, play_id DESC
            LIMIT 12
            """,
            (event_id,),
        ).fetchall()

    result.update({
        "event_matched": True,
        "live_context_status": "MATCHED_EXACT_ESPN_ID",
        "state": event["state"],
        "detail": event["detail"],
        "period": event["period"],
        "clock": event["clock"],
        "home_team": event["home_team"],
        "away_team": event["away_team"],
        "home_score": event["home_score"],
        "away_score": event["away_score"],
        "updated_at_utc": event["updated_at_utc"],
        "play_count": int(play_count),
        "recent_plays": [
            dict(row)
            for row in reversed(recent)
        ],
    })

    return result



def _extract_injury_token(play_text: str) -> tuple[str | None, str | None]:
    """Extract only an explicit ESPN injury/return token from play text."""
    text = str(play_text or "")
    patterns = (
        ("RETURNED", r"Injury Update:\s+[A-Z]+-([A-Za-z0-9.'’\-]+)\s+has returned to the game"),
        ("INJURED", r"\b[A-Z]+-([A-Za-z0-9.'’\-]+)\s+was injured during the play"),
    )
    for event_type, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return event_type, match.group(1)
    return None, None


def load_impact_injuries(
    game_id: str,
    player_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build full-game verified injury state.

    The injury token must map to exactly one live_play_players athlete_id.
    That ESPN athlete ID must then map to exactly one nfl.db player identity.
    Anything ambiguous fails closed.
    """
    if not LIVE_DB.is_file():
        return []

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row
        game = conn.execute(
            "SELECT espn FROM games WHERE game_id = ?",
            (str(game_id),),
        ).fetchone()

    if not game:
        return []

    event_id = str(game["espn"] or "").strip()
    if not event_id:
        return []

    with _open_ro(LIVE_DB) as conn:
        conn.row_factory = sqlite3.Row
        plays = conn.execute(
            """
            SELECT event_id, play_id, sequence_number, period, clock, play_text
            FROM live_plays
            WHERE event_id = ?
              AND (
                    lower(coalesce(play_text,'')) LIKE '%was injured during the play%'
                 OR lower(coalesce(play_text,'')) LIKE '%injury update:%returned to the game%'
              )
            ORDER BY sequence_number, play_id
            """,
            (event_id,),
        ).fetchall()

        events = []
        for play in plays:
            event_type, raw_token = _extract_injury_token(play["play_text"])
            if not event_type or not raw_token:
                continue

            links = conn.execute(
                """
                SELECT DISTINCT pp.athlete_id
                FROM live_play_players pp
                WHERE pp.event_id = ?
                  AND pp.play_id = ?
                  AND pp.raw_token = ?
                  AND pp.athlete_id IS NOT NULL
                """,
                (event_id, play["play_id"], raw_token),
            ).fetchall()

            athlete_ids = sorted({str(r["athlete_id"]) for r in links})
            if len(athlete_ids) != 1:
                continue

            events.append({
                "event_type": event_type,
                "athlete_id": athlete_ids[0],
                "period": play["period"],
                "clock": play["clock"],
                "sequence_number": play["sequence_number"],
                "play_text": play["play_text"],
            })

    if not events:
        return []

    athlete_ids = sorted({e["athlete_id"] for e in events})
    placeholders = ",".join("?" for _ in athlete_ids)

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row
        identities = conn.execute(
            f"""
            SELECT espn_id, gsis_id, full_name, position, latest_team
            FROM player_identity
            WHERE espn_id IN ({placeholders})
            ORDER BY espn_id
            """,
            athlete_ids,
        ).fetchall()

    by_espn = {}
    counts = {}
    for row in identities:
        key = str(row["espn_id"])
        counts[key] = counts.get(key, 0) + 1
        by_espn[key] = dict(row)

    context_by_player = {
        str(row.get("player_id")): row
        for row in (player_context or [])
        if row.get("player_id")
    }

    states = {}
    for event in events:
        espn_id = event["athlete_id"]
        if counts.get(espn_id) != 1:
            continue

        identity = by_espn[espn_id]
        player_id = str(identity.get("gsis_id") or "")
        position = str(identity.get("position") or "")
        context = context_by_player.get(player_id, {})

        state = states.setdefault(espn_id, {
            "espn_id": espn_id,
            "player_id": player_id or None,
            "player": identity.get("full_name"),
            "team": identity.get("latest_team"),
            "position": position or None,
            "status": None,
            "injury_period": None,
            "injury_clock": None,
            "return_period": None,
            "return_clock": None,
            "pregame_report_status": context.get("report_status"),
            "pregame_primary_injury": context.get("primary_injury"),
            "fd_avg_3": context.get("fd_avg_3"),
            "opportunities_avg_3": context.get("opportunities_avg_3"),
            "snap_pct_avg_3": context.get("snap_pct_avg_3"),
            "established_role_flag": context.get("established_role_flag"),
        })

        if event["event_type"] == "INJURED":
            state["status"] = "INJURY_REPORTED_NO_VERIFIED_RETURN"
            state["injury_period"] = event["period"]
            state["injury_clock"] = event["clock"]
            state["return_period"] = None
            state["return_clock"] = None
        elif event["event_type"] == "RETURNED" and state.get("status"):
            state["status"] = "RETURNED"
            state["return_period"] = event["period"]
            state["return_clock"] = event["clock"]

    offensive_line = {"C", "G", "LG", "RG", "T", "LT", "RT", "OL"}
    result = []

    for state in states.values():
        pos = str(state.get("position") or "")
        if pos in OFFENSIVE_POSITIONS:
            state["scope"] = "OFFENSIVE_FANTASY"
        elif pos in offensive_line:
            state["scope"] = "OFFENSIVE_LINE_CONTEXT"
        else:
            # Individual defenders remain outside the fantasy tracking surface.
            continue

        if state["status"] == "RETURNED":
            impact = "RESOLVED_RETURNED"
        elif pos == "QB":
            impact = "MAJOR_POTENTIAL_IMPACT"
        elif state["scope"] == "OFFENSIVE_LINE_CONTEXT":
            impact = "TEAM_CONTEXT"
        elif bool(state.get("established_role_flag")):
            impact = "POTENTIAL_IMPACT"
        else:
            impact = "INJURY_WATCH"

        state["impact_classification"] = impact
        state["verified_out"] = False
        state["causation_claimed"] = False
        result.append(state)

    order = {
        "MAJOR_POTENTIAL_IMPACT": 0,
        "POTENTIAL_IMPACT": 1,
        "TEAM_CONTEXT": 2,
        "INJURY_WATCH": 3,
        "RESOLVED_RETURNED": 4,
    }
    return sorted(
        result,
        key=lambda row: (
            order.get(str(row.get("impact_classification")), 9),
            str(row.get("team") or ""),
            str(row.get("player") or ""),
        ),
    )


def _injury_observations(injuries: list[dict[str, Any]]) -> list[str]:
    observations = []
    for row in injuries:
        player = str(row.get("player") or "Player")
        pos = str(row.get("position") or "").strip()
        team = str(row.get("team") or "").strip()
        who = f"{player} ({pos})" if pos else player
        if team:
            who += f", {team}"

        if row.get("status") == "RETURNED":
            observations.append(
                f"Injury update: {who} was reported injured but later returned."
            )
        elif row.get("impact_classification") == "MAJOR_POTENTIAL_IMPACT":
            observations.append(
                f"Impact injury: {who} was reported injured and no verified "
                f"return has been observed. A quarterback injury could materially "
                f"alter the team's offensive outlook."
            )
        elif row.get("impact_classification") == "POTENTIAL_IMPACT":
            observations.append(
                f"Impact injury: {who} was reported injured and no verified "
                f"return has been observed. His established pregame role makes "
                f"this potentially meaningful to the offense."
            )
        elif row.get("impact_classification") == "TEAM_CONTEXT":
            observations.append(
                f"Offensive-line injury: {who} was reported injured and no "
                f"verified return has been observed; this is team-level context."
            )
        else:
            observations.append(
                f"Injury watch: {who} was reported injured and no verified "
                f"return has been observed."
            )
    return observations


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _meaningful_pregame_role(
    role: dict[str, Any] | None,
) -> bool:
    """
    Determine whether prior/current pregame usage supports
    material offensive involvement.

    This never uses player-name exceptions.
    """
    if not role:
        return False

    if bool(role.get("established_role_flag")):
        return True

    return any((
        _safe_float(role.get("opportunities_avg_3")) >= 6.0,
        _safe_float(role.get("snap_pct_avg_3")) >= 0.50,
        _safe_float(role.get("targets_avg_3")) >= 5.0,
        _safe_float(role.get("carries_avg_3")) >= 6.0,
    ))


def _load_safe_pregame_role_context(
    conn: sqlite3.Connection,
    *,
    identity_key: str,
    game_id: str,
    season: int,
    week: int,
) -> tuple[dict[str, Any] | None, str]:
    """
    Load role context without postgame/future leakage.

    Priority:
      1. exact current-game pregame feature row;
      2. most recent strictly prior season/week row.

    Anything after the target week is forbidden.
    """
    columns = """
        game_id,
        season,
        week,
        identity_key,
        player_display_name,
        position,
        team,
        established_role_flag,
        rising_role_flag,
        declining_role_flag,
        opportunities_avg_3,
        snap_pct_avg_3,
        targets_avg_3,
        carries_avg_3,
        fd_avg_3,
        fd_max_5
    """

    current_rows = conn.execute(
        f"""
        SELECT {columns}
        FROM player_pregame_features
        WHERE game_id = ?
          AND identity_key = ?
        """,
        (str(game_id), str(identity_key)),
    ).fetchall()

    if len(current_rows) == 1:
        return dict(current_rows[0]), "CURRENT_GAME"

    # Fail closed if a supposedly unique current-game identity is duplicated.
    if len(current_rows) > 1:
        return None, "AMBIGUOUS_CURRENT_GAME"

    prior = conn.execute(
        f"""
        SELECT {columns}
        FROM player_pregame_features
        WHERE identity_key = ?
          AND (
                season < ?
                OR (
                    season = ?
                    AND week < ?
                )
          )
        ORDER BY
            season DESC,
            week DESC,
            game_id DESC
        LIMIT 1
        """,
        (
            str(identity_key),
            int(season),
            int(season),
            int(week),
        ),
    ).fetchone()

    if prior is None:
        return None, "NO_ROLE_CONTEXT"

    return dict(prior), "PRIOR_GAME"


def load_pregame_impact_injuries(
    game_id: str,
) -> list[dict[str, Any]]:
    """
    Build deterministic pregame injury impact context.

    Authority:
      injury_consensus_current

    Identity:
      exact gsis_id -> player_identity.identity_key

    Role context:
      exact current-game feature when available, otherwise the
      latest strictly prior feature row.

    Guardrails:
      - QUESTIONABLE is never treated as OUT;
      - DOUBTFUL retains its actual status;
      - individual defensive players are excluded;
      - forecasts are never modified;
      - no fuzzy identity matching.
    """
    offensive_line = {
        "C", "G", "LG", "RG",
        "T", "LT", "RT", "OL",
    }

    allowed_positions = (
        set(OFFENSIVE_POSITIONS)
        | offensive_line
    )

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row

        game = conn.execute(
            """
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team
            FROM games
            WHERE game_id = ?
            """,
            (str(game_id),),
        ).fetchone()

        if game is None:
            return []

        season = int(game["season"])
        week = int(game["week"])
        home_team = str(game["home_team"] or "")
        away_team = str(game["away_team"] or "")

        positions = sorted(allowed_positions)
        placeholders = ",".join(
            "?" for _ in positions
        )

        injury_rows = conn.execute(
            f"""
            SELECT
                ic.gsis_id,
                ic.team,
                ic.position,
                ic.player_name,
                ic.report_status,
                ic.practice_status,
                ic.primary_injury,
                ic.consensus_status,
                ic.injury_gate,
                ic.injury_gate_reason,
                ic.availability_risk,
                ic.authoritative_source,
                pi.identity_key,
                pi.full_name AS identity_full_name
            FROM injury_consensus_current ic
            JOIN player_identity pi
              ON CAST(pi.gsis_id AS TEXT)
               = CAST(ic.gsis_id AS TEXT)
            WHERE ic.season = ?
              AND ic.week = ?
              AND ic.team IN (?, ?)
              AND UPPER(TRIM(COALESCE(
                    ic.report_status, ''
                  ))) IN (
                    'OUT',
                    'INACTIVE',
                    'DOUBTFUL',
                    'QUESTIONABLE'
                  )
              AND UPPER(TRIM(COALESCE(
                    ic.position, ''
                  ))) IN ({placeholders})
            ORDER BY
                ic.team,
                ic.position,
                ic.player_name
            """,
            (
                season,
                week,
                home_team,
                away_team,
                *positions,
            ),
        ).fetchall()

        # Exact GSIS identity is required to be unique.
        gsis_counts: dict[str, int] = {}
        for row in injury_rows:
            gsis = str(row["gsis_id"] or "")
            gsis_counts[gsis] = (
                gsis_counts.get(gsis, 0) + 1
            )

        result = []

        for row in injury_rows:
            gsis_id = str(row["gsis_id"] or "")
            if not gsis_id:
                continue

            if gsis_counts.get(gsis_id) != 1:
                continue

            identity_key = str(
                row["identity_key"] or ""
            ).strip()

            if not identity_key:
                continue

            position = str(
                row["position"] or ""
            ).strip().upper()

            status = str(
                row["report_status"] or ""
            ).strip().upper()

            role, role_source = (
                _load_safe_pregame_role_context(
                    conn,
                    identity_key=identity_key,
                    game_id=str(game_id),
                    season=season,
                    week=week,
                )
            )

            meaningful_role = (
                _meaningful_pregame_role(role)
            )

            if position in offensive_line:
                scope = "OFFENSIVE_LINE_CONTEXT"

                if status in {"OUT", "DOUBTFUL"}:
                    impact = "TEAM_CONTEXT"
                elif status == "QUESTIONABLE":
                    impact = (
                        "LINE_AVAILABILITY_UNCERTAINTY"
                    )
                else:
                    impact = "INJURY_WATCH"

            else:
                scope = "OFFENSIVE_FANTASY"

                if (
                    position == "QB"
                    and status in {"OUT", "DOUBTFUL"}
                ):
                    impact = "MAJOR_PREGAME_IMPACT"

                elif (
                    status in {"OUT", "DOUBTFUL"}
                    and meaningful_role
                ):
                    impact = (
                        "MATERIAL_PREGAME_IMPACT"
                    )

                elif (
                    status == "QUESTIONABLE"
                    and meaningful_role
                ):
                    impact = (
                        "AVAILABILITY_UNCERTAINTY"
                    )

                else:
                    impact = "INJURY_WATCH"

            if status in {"OUT", "INACTIVE"}:
                availability = (
                    "UNAVAILABLE_CONFIRMED"
                )
            elif status == "DOUBTFUL":
                availability = (
                    "UNLIKELY_AVAILABLE"
                )
            else:
                availability = "UNCERTAIN"

            player_name = (
                row["identity_full_name"]
                or row["player_name"]
            )

            item = {
                "player_id": gsis_id,
                "player": player_name,
                "team": row["team"],
                "position": position,
                "status": status,
                "report_status": status,
                "practice_status": (
                    row["practice_status"]
                ),
                "primary_injury": (
                    row["primary_injury"]
                ),
                "consensus_status": (
                    row["consensus_status"]
                ),
                "injury_gate": row["injury_gate"],
                "injury_gate_reason": (
                    row["injury_gate_reason"]
                ),
                "availability_risk": (
                    row["availability_risk"]
                ),
                "authoritative_source": (
                    row["authoritative_source"]
                ),
                "availability_interpretation": (
                    availability
                ),
                "scope": scope,
                "impact_classification": impact,
                "meaningful_role": meaningful_role,
                "role_context_source": role_source,
                "role_context_game_id": (
                    role.get("game_id")
                    if role else None
                ),
                "role_context_season": (
                    role.get("season")
                    if role else None
                ),
                "role_context_week": (
                    role.get("week")
                    if role else None
                ),
                "established_role_flag": (
                    role.get("established_role_flag")
                    if role else None
                ),
                "opportunities_avg_3": (
                    role.get("opportunities_avg_3")
                    if role else None
                ),
                "snap_pct_avg_3": (
                    role.get("snap_pct_avg_3")
                    if role else None
                ),
                "targets_avg_3": (
                    role.get("targets_avg_3")
                    if role else None
                ),
                "carries_avg_3": (
                    role.get("carries_avg_3")
                    if role else None
                ),
                "fd_avg_3": (
                    role.get("fd_avg_3")
                    if role else None
                ),
                "fd_max_5": (
                    role.get("fd_max_5")
                    if role else None
                ),
                "verified_out": status == "OUT",
                "verified_unavailable": (
                    status in {"OUT", "INACTIVE"}
                ),
                "questionable_treated_as_out": False,
                "forecast_modified": False,
                "causation_claimed": False,
            }

            result.append(item)

    order = {
        "MAJOR_PREGAME_IMPACT": 0,
        "MATERIAL_PREGAME_IMPACT": 1,
        "AVAILABILITY_UNCERTAINTY": 2,
        "TEAM_CONTEXT": 3,
        "LINE_AVAILABILITY_UNCERTAINTY": 4,
        "INJURY_WATCH": 5,
    }

    return sorted(
        result,
        key=lambda item: (
            order.get(
                str(
                    item.get(
                        "impact_classification"
                    )
                ),
                9,
            ),
            str(item.get("team") or ""),
            str(item.get("player") or ""),
        ),
    )


def _pregame_injury_observations(
    injuries: list[dict[str, Any]],
) -> list[str]:
    observations = []

    for row in injuries:
        impact = str(
            row.get("impact_classification") or ""
        )

        # Keep lower-priority watch items in the structured
        # injury packet without cluttering What stands out.
        if impact == "INJURY_WATCH":
            continue

        player = str(
            row.get("player") or "Player"
        )

        pos = str(
            row.get("position") or ""
        ).strip()

        team = str(
            row.get("team") or ""
        ).strip()

        status = str(
            row.get("report_status") or ""
        ).strip().lower()

        injury = str(
            row.get("primary_injury") or ""
        ).strip()

        who = (
            f"{player} ({pos})"
            if pos else player
        )

        if team:
            who += f", {team}"

        detail = (
            f" with a {injury} injury"
            if injury else ""
        )

        if impact == "MAJOR_PREGAME_IMPACT":
            observations.append(
                f"Pregame injury impact: {who} is "
                f"listed {status}{detail}. "
                f"Quarterback availability could "
                f"materially affect the offensive "
                f"outlook."
            )

        elif impact == "MATERIAL_PREGAME_IMPACT":
            observations.append(
                f"Pregame injury impact: {who} is "
                f"listed {status}{detail}. "
                f"Recent offensive involvement makes "
                f"the availability loss potentially "
                f"meaningful."
            )

        elif impact == "AVAILABILITY_UNCERTAINTY":
            observations.append(
                f"Pregame injury watch: {who} is "
                f"listed questionable{detail}. "
                f"Availability remains uncertain, and "
                f"the player's recent role makes the "
                f"status important to monitor."
            )

        elif impact == "TEAM_CONTEXT":
            observations.append(
                f"Pregame offensive-line context: "
                f"{who} is listed {status}{detail}. "
                f"The absence could affect the offense "
                f"at the team level."
            )

        elif (
            impact
            == "LINE_AVAILABILITY_UNCERTAINTY"
        ):
            observations.append(
                f"Pregame offensive-line watch: "
                f"{who} is listed questionable"
                f"{detail}. Availability remains "
                f"uncertain."
            )

    return observations

def _forecast_evidence(
    forecast: dict[str, Any] | None,
) -> dict[str, Any]:
    if forecast is None:
        return {
            "available": False,
            "status": "NOT_AVAILABLE",
            "ready": False,
        }

    status = str(
        forecast.get("forecast_status") or ""
    ).strip()

    market_ready = (
        forecast.get("forecast_market_ready") == 1
    )

    ready = (
        status in {
            "READY_CORE_ONLY",
            "READY_INJURY_ADJUSTED",
        }
        and market_ready
    )

    evidence = {
        "available": True,
        "status": status or None,
        "ready": ready,
        "market_ready": market_ready,
        "model": forecast.get("model"),
        "variant": forecast.get("forecast_variant"),
        "impact_status": forecast.get("impact_status"),
        "replacement_status": forecast.get(
            "replacement_status"
        ),
        "win_probability_status": forecast.get(
            "win_probability_status"
        ),
    }

    if ready:
        evidence.update({
            "pred_winner": forecast.get("pred_winner"),
            "pred_home_margin": forecast.get(
                "pred_home_margin"
            ),
            "pred_total_points": forecast.get(
                "pred_total_points"
            ),
            "pred_home_points": forecast.get(
                "pred_home_points"
            ),
            "pred_away_points": forecast.get(
                "pred_away_points"
            ),
        })

    # Never expose an invented probability.
    if (
        forecast.get("win_probability_status")
        != "CALIBRATED"
    ):
        evidence["win_probability"] = None

    return evidence


def build_game_evidence(
    game_id: str,
) -> dict[str, Any]:
    game_id = str(game_id).strip()

    if not game_id:
        raise ValueError("game_id is required")

    game = load_game(game_id)

    if game is None:
        raise KeyError(
            f"Unknown game_id: {game_id}"
        )

    forecast = load_forecast(game_id)
    player_context = load_player_context(game_id)
    live_context = load_live_context(game_id)

    live_state = str(
        live_context.get("state") or ""
    ).lower()

    completed = bool(
        game.get("completed")
    )

    # WFS_ANALYST_COMPLETED_STATE_AUTHORITY_V2
    # Completed schedule/game authority outranks a stale live-event row.
    if completed or live_state == "post":
        stat_forecast_mode = "POSTGAME"
    elif (
        live_context.get("event_matched")
        and live_state in {"in", "live"}
    ):
        stat_forecast_mode = "LIVE"
    else:
        stat_forecast_mode = "PREGAME"

    return {
        "contract": "WFS_AI_ANALYST_EVIDENCE_V1",
        "game_id": game_id,
        "game": game,
        "forecast": _forecast_evidence(forecast),
        "stat_forecast_context": load_stat_forecast_context(
            game_id,
            mode=stat_forecast_mode,
        ),
        "team_context": load_team_context(game_id),
        "player_context": player_context,
        "live_context": live_context,
        "impact_injuries": load_impact_injuries(game_id, player_context),
        "pregame_impact_injuries": load_pregame_impact_injuries(game_id),
        "guardrails": {
            "read_only": True,
            "llm_used": False,
            "forecast_writes_allowed": False,
            "database_writes_allowed": False,
            "solver_writes_allowed": False,
            "projection_writes_allowed": False,
            "invent_probability_allowed": False,
            "individual_defensive_players_in_scope": False,
        },
    }

def load_final_player_results(
    game_id: str,
) -> list[dict[str, Any]]:
    """
    Load verified offensive fantasy results for a completed game.
    """
    positions = sorted(OFFENSIVE_POSITIONS)
    placeholders = ",".join("?" for _ in positions)

    sql = f"""
        SELECT
            game_id,
            player_id,
            player_display_name,
            position,
            team,
            opponent_team,
            completions,
            attempts,
            passing_yards,
            passing_tds,
            passing_interceptions,
            carries,
            rushing_yards,
            rushing_tds,
            receptions,
            targets,
            receiving_yards,
            receiving_tds,
            total_fumbles_lost,
            fanduel_points,
            fanduel_points_verified
        FROM player_game_stats
        WHERE game_id = ?
          AND position IN ({placeholders})
        ORDER BY
            fanduel_points DESC,
            player_display_name
    """

    with _open_ro(NFL_DB) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            sql,
            [str(game_id), *positions],
        ).fetchall()

    return [dict(row) for row in rows]


def _fmt_number(
    value: Any,
    digits: int = 1,
) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _build_postgame_response(
    packet: dict[str, Any],
) -> dict[str, Any]:
    game = packet["game"]
    live = packet["live_context"]

    results = load_final_player_results(
        packet["game_id"]
    )

    verified = [
        row
        for row in results
        if row.get("fanduel_points_verified") == 1
    ]

    away = str(
        live.get("away_team")
        or game.get("away_team")
        or ""
    )

    home = str(
        live.get("home_team")
        or game.get("home_team")
        or ""
    )

    # WFS_ANALYST_POSTGAME_SCORE_AUTHORITY_V2
    # A completed game uses persisted final scores first. This prevents
    # stale SAFE_FAIL live-event rows from leaking partial-game scores
    # into the postgame Analyst.
    if bool(game.get("completed")):
        away_score = game.get("away_score")
        home_score = game.get("home_score")
    else:
        away_score = live.get("away_score")
        home_score = live.get("home_score")

        if away_score is None:
            away_score = game.get("away_score")
        if home_score is None:
            home_score = game.get("home_score")

    if (
        away_score is not None
        and home_score is not None
    ):
        if away_score > home_score:
            winner = away
            loser = home
            margin = away_score - home_score
        elif home_score > away_score:
            winner = home
            loser = away
            margin = home_score - away_score
        else:
            winner = None
            loser = None
            margin = 0
    else:
        winner = None
        loser = None
        margin = None

    if winner:
        headline = (
            f"{winner} defeats {loser} "
            f"{max(away_score, home_score)}-"
            f"{min(away_score, home_score)}"
        )
    else:
        headline = f"{away} at {home} — Final"

    if margin is not None and winner:
        game_summary = (
            f"{winner} won by {margin} points. "
            f"The final score was "
            f"{away} {away_score}, "
            f"{home} {home_score}."
        )
    else:
        game_summary = (
            f"Final score: {away} {away_score}, "
            f"{home} {home_score}."
        )

    fantasy_leaders = []

    for row in verified[:5]:
        fantasy_leaders.append({
            "player": row.get("player_display_name"),
            "team": row.get("team"),
            "position": row.get("position"),
            "fanduel_points": row.get(
                "fanduel_points"
            ),
            "summary": (
                f"{row.get('player_display_name')} "
                f"scored "
                f"{_fmt_number(row.get('fanduel_points'))} "
                f"FanDuel points."
            ),
        })

    observations = []

    for row in verified[:10]:
        name = row.get("player_display_name")
        pos = row.get("position")
        fd = row.get("fanduel_points")

        if pos == "QB":
            pass_yards = row.get("passing_yards") or 0
            pass_tds = row.get("passing_tds") or 0
            rush_yards = row.get("rushing_yards") or 0

            if pass_tds >= 2:
                observations.append(
                    f"{name} threw for "
                    f"{int(pass_yards)} yards and "
                    f"{int(pass_tds)} touchdowns, "
                    f"finishing with "
                    f"{_fmt_number(fd)} FanDuel points."
                )

        elif pos in {"RB", "FB"}:
            rush_yards = row.get("rushing_yards") or 0
            rush_tds = row.get("rushing_tds") or 0
            receptions = row.get("receptions") or 0
            rec_yards = row.get("receiving_yards") or 0

            if rush_tds or rush_yards >= 60:
                observations.append(
                    f"{name} produced "
                    f"{int(rush_yards)} rushing yards"
                    + (
                        f" and {int(rush_tds)} rushing touchdown"
                        f"{'s' if rush_tds != 1 else ''}"
                        if rush_tds
                        else ""
                    )
                    + (
                        f", plus {int(receptions)} catches "
                        f"for {int(rec_yards)} yards"
                        if receptions
                        else ""
                    )
                    + "."
                )

        elif pos in {"WR", "TE"}:
            receptions = row.get("receptions") or 0
            targets = row.get("targets") or 0
            yards = row.get("receiving_yards") or 0
            tds = row.get("receiving_tds") or 0

            if tds or yards >= 70:
                observations.append(
                    f"{name} caught "
                    f"{int(receptions)} of "
                    f"{int(targets)} targets for "
                    f"{int(yards)} yards"
                    + (
                        f" and {int(tds)} touchdown"
                        f"{'s' if tds != 1 else ''}"
                        if tds
                        else ""
                    )
                    + "."
                )

        if len(observations) >= 5:
            break

    return {
        "mode": "POSTGAME",
        "headline": headline,
        "game_summary": game_summary,
        "fantasy_leaders": fantasy_leaders,
        "impact_injuries": packet.get("impact_injuries", []),
        "observations": (
            _injury_observations(packet.get("impact_injuries", []))
            + observations
        )[:8],
        "evidence_status": {
            "live_event_verified": (
                live.get("event_matched") is True
            ),
            "live_state": live.get("state"),
            "final_player_rows": len(results),
            "verified_fanduel_rows": len(verified),
            "forecast_available": (
                packet["forecast"].get("available")
                is True
            ),
        },
    }



def _stat_number(
    row: dict[str, Any],
    key: str,
    digits: int = 1,
) -> str | None:
    value = row.get(key)

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(value):
        return None

    return f"{value:.{digits}f}"


def _stat_player_name(
    row: dict[str, Any],
) -> str:
    for key in (
        "entity_name",
        "player_name",
        "name",
        "display_name",
        "full_name",
    ):
        value = str(
            row.get(key) or ""
        ).strip()

        if value:
            return value

    return "Player"


def _stat_outlook_sort_value(
    row: dict[str, Any],
) -> float:
    """
    Football-opportunity ordering used only to keep Stat Outlook
    concise. This is not fantasy scoring.
    """
    position = str(
        row.get("position") or ""
    ).upper()

    def num(key):
        try:
            value = float(
                row.get(key)
            )

            if pd.isna(value):
                return 0.0

            return max(
                value,
                0.0,
            )

        except (TypeError, ValueError):
            return 0.0

    if position == "QB":
        return (
            num("expected_attempts")
            + num("expected_carries")
        )

    if position in {"RB", "FB"}:
        return (
            num("expected_carries")
            + num("expected_targets")
        )

    if position in {"WR", "TE"}:
        return num(
            "expected_targets"
        )

    return 0.0



# WFS_STAT_OUTLOOK_COLD_START_FALLBACK_V1
#
# PREGAME presentation fallback only.
#
# Used only when:
#   * exact starter GSIS exists in authoritative depth lane
#   * player is not OUT / INACTIVE
#   * canonical statistical forecast row is missing
#   * an exact cold-start rookie shadow row exists
#
# Canonical forecast is never mutated here.
#
def _stat_outlook_cold_start_rows(
    team: str,
    game_ids: set[str],
    existing_player_ids: set[str],
    availability_by_player: dict[str, str],
) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parent

    shadow_path = (
        root
        / "data"
        / "parquet"
        / "current_cold_start_full_team_shadow_v2.parquet"
    )

    feasibility_path = (
        root
        / "processed"
        / "stage26gr8z_current_feasibility_cap.csv"
    )

    if not shadow_path.is_file():
        return []

    try:
        shadow = pd.read_parquet(
            shadow_path
        )
    except Exception:
        return []

    if shadow.empty:
        return []

    required = {
        "game_id",
        "team",
        "player_id",
        "position",
        "shadow_row_type",
    }

    if not required.issubset(
        shadow.columns
    ):
        return []

    team = str(
        team or ""
    ).strip().upper()

    shadow = shadow[
        shadow["team"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(team)
        &
        shadow["game_id"]
        .fillna("")
        .astype(str)
        .isin(game_ids)
        &
        shadow["shadow_row_type"]
        .fillna("")
        .astype(str)
        .eq("COLD_START_ROOKIE")
    ].copy()

    if shadow.empty:
        return []

    # Optional R8Z hard-feasibility carry override.
    #
    # Only the displayed fallback row is adjusted here.
    # Canonical production artifact remains untouched.
    feasibility_by_key = {}

    if feasibility_path.is_file():
        try:
            feas = pd.read_csv(
                feasibility_path
            )

            needed = {
                "game_id",
                "team",
                "player_id",
                "hybrid_feasible_carries",
            }

            if needed.issubset(
                feas.columns
            ):
                for row in feas.to_dict(
                    orient="records"
                ):
                    key = (
                        str(
                            row.get(
                                "game_id"
                            )
                            or ""
                        ),
                        str(
                            row.get(
                                "team"
                            )
                            or ""
                        )
                        .strip()
                        .upper(),
                        str(
                            row.get(
                                "player_id"
                            )
                            or ""
                        )
                        .strip(),
                    )

                    try:
                        value = float(
                            row.get(
                                "hybrid_feasible_carries"
                            )
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if math.isfinite(
                        value
                    ):
                        feasibility_by_key[
                            key
                        ] = max(
                            0.0,
                            value,
                        )

        except Exception:
            feasibility_by_key = {}

    out = []

    for raw in shadow.to_dict(
        orient="records"
    ):
        player_id = str(
            raw.get(
                "player_id"
            )
            or ""
        ).strip()

        if not player_id:
            continue

        # Canonical always wins.
        if player_id in existing_player_ids:
            continue

        status = str(
            availability_by_player.get(
                player_id,
                "",
            )
            or ""
        ).strip().upper()

        if status in {
            "OUT",
            "INACTIVE",
        }:
            continue

        row = dict(
            raw
        )

        row["entity_type"] = (
            "OFFENSE_PLAYER"
        )

        if not row.get(
            "player_name"
        ):
            row["player_name"] = (
                row.get(
                    "entity_name"
                )
                or ""
            )

        key = (
            str(
                row.get(
                    "game_id"
                )
                or ""
            ),
            team,
            player_id,
        )

        capped = (
            feasibility_by_key.get(
                key
            )
        )

        if capped is not None:
            original = row.get(
                "expected_carries"
            )

            try:
                original = float(
                    original
                )
            except (
                TypeError,
                ValueError,
            ):
                original = None

            if (
                original is not None
                and math.isfinite(
                    original
                )
                and original > 0
            ):
                ratio = (
                    capped
                    / original
                )

                row[
                    "expected_carries"
                ] = capped

                # Preserve original efficiency when the hard
                # feasibility cap trims carries.
                for field in [
                    "expected_rushing_yards",
                    "expected_rushing_tds",
                ]:
                    try:
                        value = float(
                            row.get(
                                field
                            )
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if math.isfinite(
                        value
                    ):
                        row[field] = (
                            value
                            * ratio
                        )

        row[
            "_stat_outlook_cold_start_fallback"
        ] = True

        out.append(
            row
        )

    return out


# WFS_STAT_OUTLOOK_TD_DISPLAY_RECONCILIATION_V1
#
# Presentation-only integer TD allocation.
#
# The underlying statistical expectations remain fractional and
# unchanged. We reconcile the displayed receiver TD counts with
# the displayed starting-QB passing TD count.
#
def _stat_outlook_reconcile_display_tds(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected:
        return selected

    rows = [
        dict(row)
        for row in selected
    ]

    quarterbacks = [
        row
        for row in rows
        if str(
            row.get(
                "position"
            )
            or ""
        ).strip().upper()
        == "QB"
    ]

    if not quarterbacks:
        return rows

    quarterback = max(
        quarterbacks,
        key=_stat_outlook_usage_value,
    )

    passing_tds = _stat_number(
        quarterback,
        "expected_passing_tds",
    )

    if passing_tds is None:
        return rows

    try:
        passing_tds = float(
            passing_tds
        )
    except (
        TypeError,
        ValueError,
    ):
        return rows

    if not math.isfinite(
        passing_tds
    ):
        return rows

    displayed_total = int(
        max(
            0.0,
            passing_tds,
        )
        + 0.5
    )

    quarterback[
        "_display_expected_passing_tds"
    ] = displayed_total

    receivers = []

    for row in rows:
        position = str(
            row.get(
                "position"
            )
            or ""
        ).strip().upper()

        if position not in {
            "RB",
            "FB",
            "WR",
            "TE",
        }:
            continue

        value = _stat_number(
            row,
            "expected_receiving_tds",
        )

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        if not math.isfinite(
            value
        ):
            value = 0.0

        receivers.append(
            (
                row,
                max(
                    0.0,
                    value,
                ),
            )
        )

    if not receivers:
        return rows

    for row, _ in receivers:
        row[
            "_display_expected_receiving_tds"
        ] = 0

    if displayed_total <= 0:
        return rows

    total_weight = sum(
        value
        for _, value in receivers
    )

    # If every receiving TD expectation is zero, do not invent a
    # scorer. Keep the QB display unchanged and leave receivers zero.
    if total_weight <= 0:
        return rows

    quotas = []

    assigned = 0

    for row, weight in receivers:
        quota = (
            displayed_total
            * weight
            / total_weight
        )

        base = int(
            math.floor(
                quota
            )
        )

        row[
            "_display_expected_receiving_tds"
        ] = base

        assigned += base

        player_id = str(
            row.get(
                "player_id"
            )
            or ""
        ).strip()

        quotas.append(
            (
                quota - base,
                weight,
                player_id,
                row,
            )
        )

    remaining = (
        displayed_total
        - assigned
    )

    quotas.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
        reverse=True,
    )

    for i in range(
        remaining
    ):
        row = quotas[
            i % len(
                quotas
            )
        ][3]

        row[
            "_display_expected_receiving_tds"
        ] += 1

    return rows



# WFS_STAT_OUTLOOK_EVENT_DISPLAY_RECONCILIATION_V1
#
# Presentation-only whole-number event coherence.
#
# Canonical statistical expectations remain fractional and unchanged.
# This layer prevents independently rounded public counts from
# producing impossible or internally contradictory football lines.
#
def _stat_outlook_reconcile_display_events(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected:
        return selected

    rows = [
        dict(row)
        for row in selected
    ]

    def displayed(value: Any) -> int | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return int(
            max(
                0.0,
                number,
            )
            + 0.5
        )

    for row in rows:
        position = str(
            row.get("position")
            or ""
        ).strip().upper()

        if position == "QB":
            attempts = displayed(
                row.get(
                    "expected_attempts"
                )
            )
            completions = displayed(
                row.get(
                    "expected_completions"
                )
            )
            passing_yards = displayed(
                row.get(
                    "expected_passing_yards"
                )
            )

            if attempts is not None:
                row["_display_expected_attempts"] = attempts

            if completions is not None:
                if attempts is not None:
                    completions = min(
                        completions,
                        attempts,
                    )

                row[
                    "_display_expected_completions"
                ] = completions

            if passing_yards is not None:
                if (
                    completions is not None
                    and completions == 0
                ):
                    passing_yards = 0

                row[
                    "_display_expected_passing_yards"
                ] = passing_yards

        if position in {
            "RB",
            "FB",
        }:
            carries = displayed(
                row.get(
                    "expected_carries"
                )
            )
            rushing_yards = displayed(
                row.get(
                    "expected_rushing_yards"
                )
            )

            if carries is not None:
                row["_display_expected_carries"] = carries

            if rushing_yards is not None:
                if (
                    carries is not None
                    and carries == 0
                ):
                    rushing_yards = 0

                row[
                    "_display_expected_rushing_yards"
                ] = rushing_yards

        if position in {
            "RB",
            "FB",
            "WR",
            "TE",
        }:
            targets = displayed(
                row.get(
                    "expected_targets"
                )
            )
            receptions = displayed(
                row.get(
                    "expected_receptions"
                )
            )
            receiving_yards = displayed(
                row.get(
                    "expected_receiving_yards"
                )
            )

            if targets is not None:
                row["_display_expected_targets"] = targets

            if receptions is not None:
                if targets is not None:
                    receptions = min(
                        receptions,
                        targets,
                    )

                row[
                    "_display_expected_receptions"
                ] = receptions

            if receiving_yards is not None:
                if (
                    receptions is not None
                    and receptions == 0
                ):
                    receiving_yards = 0

                row[
                    "_display_expected_receiving_yards"
                ] = receiving_yards

    return rows

def _format_projected_event(value: Any) -> str:
    """
    Present a discrete single-game football event as a whole count.

    Fractional statistical expectations remain unchanged internally.
    This function changes public presentation only.
    """
    if value is None:
        return "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(number):
        return "—"

    number = max(0.0, number)

    # Conventional half-up rounding for nonnegative event counts.
    return str(int(number + 0.5))


def _format_stat_outlook_row(
    row: dict[str, Any],
) -> str | None:
    position = str(
        row.get("position") or ""
    ).upper()

    name = _stat_player_name(
        row
    )

    def n(key):
        return _stat_number(
            row,
            key,
        )

    if position == "QB":
        attempts = row.get(
            "_display_expected_attempts"
        )
        if attempts is None:
            attempts = n(
                "expected_attempts"
            )

        completions = row.get(
            "_display_expected_completions"
        )
        if completions is None:
            completions = n(
                "expected_completions"
            )

        passing_yards = row.get(
            "_display_expected_passing_yards"
        )
        if passing_yards is None:
            passing_yards = n(
                "expected_passing_yards"
            )

        passing_tds = n(
            "expected_passing_tds"
        )

        display_passing_tds = row.get(
            "_display_expected_passing_tds"
        )

        if display_passing_tds is not None:
            passing_tds = display_passing_tds

        interceptions = n(
            "expected_interceptions"
        )

        carries = n(
            "expected_carries"
        )

        rushing_yards = n(
            "expected_rushing_yards"
        )

        rushing_tds = n(
            "expected_rushing_tds"
        )

        parts = []

        if any([
            attempts,
            completions,
            passing_yards,
            passing_tds,
            interceptions,
        ]):
            parts.append(
                "Pass "
                f"{_format_projected_event(completions)}/"
                f"{_format_projected_event(attempts)}, "
                f"{_format_projected_event(passing_yards)} yds, "
                f"{_format_projected_event(passing_tds)} TD, "
                f"{_format_projected_event(interceptions)} INT"
            )

        if any([
            carries,
            rushing_yards,
            rushing_tds,
        ]):
            parts.append(
                "Rush "
                f"{_format_projected_event(carries)} att, "
                f"{_format_projected_event(rushing_yards)} yds, "
                f"{_format_projected_event(rushing_tds)} TD"
            )

        if not parts:
            return None

        return (
            f"{name} (QB): "
            + " | ".join(parts)
        )

    if position in {
        "RB",
        "FB",
        "WR",
        "TE",
    }:
        parts = []

        if position in {"RB", "FB"}:
            carries = row.get(
                "_display_expected_carries"
            )
            if carries is None:
                carries = n(
                    "expected_carries"
                )

            rushing_yards = row.get(
                "_display_expected_rushing_yards"
            )
            if rushing_yards is None:
                rushing_yards = n(
                    "expected_rushing_yards"
                )

            rushing_tds = n(
                "expected_rushing_tds"
            )

            if any([
                carries,
                rushing_yards,
                rushing_tds,
            ]):
                parts.append(
                    "Rush "
                    f"{_format_projected_event(carries)} att, "
                    f"{_format_projected_event(rushing_yards)} yds, "
                    f"{_format_projected_event(rushing_tds)} TD"
                )

        targets = row.get(
            "_display_expected_targets"
        )
        if targets is None:
            targets = n(
                "expected_targets"
            )

        receptions = row.get(
            "_display_expected_receptions"
        )
        if receptions is None:
            receptions = n(
                "expected_receptions"
            )

        receiving_yards = row.get(
            "_display_expected_receiving_yards"
        )
        if receiving_yards is None:
            receiving_yards = n(
                "expected_receiving_yards"
            )

        receiving_tds = n(
            "expected_receiving_tds"
        )

        display_receiving_tds = row.get(
            "_display_expected_receiving_tds"
        )

        if display_receiving_tds is not None:
            receiving_tds = display_receiving_tds

        if any([
            targets,
            receptions,
            receiving_yards,
            receiving_tds,
        ]):
            parts.append(
                f"{_format_projected_event(targets)} tgt, "
                f"{_format_projected_event(receptions)} rec, "
                f"{_format_projected_event(receiving_yards)} yds, "
                f"{_format_projected_event(receiving_tds)} TD"
            )

        if not parts:
            return None

        return (
            f"{name} ({position}): "
            + " | ".join(parts)
        )

    return None


# WFS_STAT_OUTLOOK_KICKER_XP_TD_RECONCILIATION_V1
#
# Presentation-only. A kicker's extra points are earned by the
# team's own touchdowns, so displayed XP counts cannot exceed the
# touchdown total already shown in that team's offense lines.
def _team_offense_td_total(
    selected: list[dict[str, Any]],
) -> int:
    total = 0

    for row in selected:
        position = str(
            row.get("position") or ""
        ).strip().upper()

        if position in {"QB", "RB", "FB"}:
            total += _round_event_count(
                _stat_number(row, "expected_rushing_tds")
            )

        if position in {"RB", "FB", "WR", "TE"}:
            receiving_tds = row.get(
                "_display_expected_receiving_tds"
            )

            if receiving_tds is None:
                receiving_tds = _round_event_count(
                    _stat_number(row, "expected_receiving_tds")
                )

            total += receiving_tds

    return total


def _round_event_count(value: Any) -> int:
    if value is None:
        return 0

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0

    if not math.isfinite(number):
        return 0

    return int(max(0.0, number) + 0.5)


def _format_kicker_outlook(
    row: dict[str, Any],
    max_tds: int | None = None,
) -> str | None:
    name = _stat_player_name(
        row
    )

    fga = _stat_number(
        row,
        "expected_fga",
    )

    fgm = _stat_number(
        row,
        "expected_fgm",
    )

    xpa = _stat_number(
        row,
        "expected_xpa",
    )

    xpm = _stat_number(
        row,
        "expected_xpm",
    )

    if not any([
        fga,
        fgm,
        xpa,
        xpm,
    ]):
        return None

    xpa_display = _format_projected_event(xpa)
    xpm_display = _format_projected_event(xpm)

    if max_tds is not None:
        if xpa_display.isdigit():
            xpa_display = str(min(int(xpa_display), max_tds))

        if xpm_display.isdigit():
            xpm_display = str(min(int(xpm_display), max_tds))

    return (
        f"{name} (K): "
        f"{_format_projected_event(fga)} FG att, "
        f"{_format_projected_event(fgm)} FG made, "
        f"{xpa_display} XP att, "
        f"{xpm_display} XP made"
    )


def _format_dst_outlook(
    row: dict[str, Any],
) -> str | None:
    team = str(
        row.get("team") or ""
    ).strip().upper()

    sacks = _stat_number(
        row,
        "expected_sacks",
    )

    interceptions = _stat_number(
        row,
        "expected_interceptions",
    )

    fumble_recoveries = _stat_number(
        row,
        "expected_fumble_recoveries",
    )

    points_allowed = _stat_number(
        row,
        "expected_points_allowed",
    )

    defensive_tds = _stat_number(
        row,
        "expected_defensive_tds",
    )

    if not any([
        sacks,
        interceptions,
        fumble_recoveries,
        points_allowed,
        defensive_tds,
    ]):
        return None

    # QA D18: fumble-recovery count controls its own singular/plural noun
    # rather than always reading "fumble recoveries".
    fumble_recoveries_display = _format_projected_event(fumble_recoveries)
    fumble_recoveries_noun = (
        "fumble recovery"
        if fumble_recoveries_display == "1"
        else "fumble recoveries"
    )

    return (
        f"{team} D/ST: "
        f"{_format_projected_event(sacks)} sacks, "
        f"{_format_projected_event(interceptions)} INT, "
        f"{fumble_recoveries_display} {fumble_recoveries_noun}, "
        f"{_format_projected_event(points_allowed)} points allowed, "
        f"{_format_projected_event(defensive_tds)} defensive TD"
    )



# WFS_STAT_OUTLOOK_STARTER_SELECTION_V1
#
# Public Stat Outlook starter-resolution policy:
#
#   1. Exact GSIS identity only.
#   2. Current depth-chart football lane is primary role authority.
#   3. OUT / INACTIVE players are skipped.
#   4. Replacement is the first available player with the next
#      greater pos_rank in the SAME:
#          pos_grp + pos_abb + pos_slot
#   5. A player must already have an existing validated WFS
#      statistical forecast row. No replacement forecast is created.
#   6. Projected opportunity may resolve obviously stale/ambiguous
#      depth information conservatively, but cannot override a hard
#      availability block.
#   7. This is presentation selection only. Forecast values are never
#      changed here.
#
# LIVE / POSTGAME continue to use their frozen kickoff forecast rows.


def _stat_outlook_usage_value(
    row: dict[str, Any],
) -> float:
    """
    Conservative projected football opportunity signal.

    This is used only as a secondary starter-role cross-check.
    It does not change any projection.
    """
    position = str(
        row.get("position") or ""
    ).strip().upper()

    def num(key: str) -> float:
        try:
            value = float(
                row.get(key)
            )
        except (TypeError, ValueError):
            return 0.0

        if pd.isna(value):
            return 0.0

        return max(
            0.0,
            value,
        )

    if position == "QB":
        return (
            num("expected_attempts")
            + num("expected_carries")
        )

    if position in {"RB", "FB"}:
        return (
            num("expected_carries")
            + num("expected_targets")
        )

    if position in {"WR", "TE"}:
        return num(
            "expected_targets"
        )

    return 0.0


def _stat_outlook_depth_rows(
    team: str,
) -> pd.DataFrame:
    """
    Load the latest authoritative depth-chart snapshot for one team.

    Read-only. No fuzzy identity matching.
    """
    team = str(
        team or ""
    ).strip().upper()

    if not team:
        return pd.DataFrame()

    db_path = (
        Path(__file__).resolve().parent
        / "data"
        / "nfl.db"
    )

    if not db_path.is_file():
        return pd.DataFrame()

    with sqlite3.connect(
        db_path
    ) as conn:
        depth = pd.read_sql_query(
            """
            SELECT
                snapshot_dt,
                team,
                player_name,
                gsis_id,
                pos_grp,
                pos_name,
                pos_abb,
                pos_slot,
                pos_rank
            FROM depth_charts
            WHERE team = ?
              AND snapshot_dt = (
                  SELECT MAX(snapshot_dt)
                  FROM depth_charts
                  WHERE team = ?
              )
            """,
            conn,
            params=[
                team,
                team,
            ],
        )

    if depth.empty:
        return depth

    depth["gsis_id"] = (
        depth["gsis_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    depth["pos_abb"] = (
        depth["pos_abb"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    depth["pos_grp"] = (
        depth["pos_grp"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    depth["rank_num"] = pd.to_numeric(
        depth["pos_rank"],
        errors="coerce",
    )

    depth["slot_num"] = pd.to_numeric(
        depth["pos_slot"],
        errors="coerce",
    )

    depth = depth[
        depth["pos_abb"].isin(
            {
                "QB",
                "RB",
                "FB",
                "WR",
                "TE",
            }
        )
        & depth["gsis_id"].ne("")
        & depth["rank_num"].notna()
        & depth["slot_num"].notna()
        & depth["pos_grp"].ne(
            "Special Teams"
        )
    ].copy()

    return depth


# WFS_STAT_OUTLOOK_STARTER_VERIFICATION_BRIDGE_V1
def _stat_outlook_starter_verification_rows(
    season: int,
    week: int,
    game_type: str = "REG",
) -> pd.DataFrame:
    """
    Load validated current Starter Verification evidence.

    Fail closed: missing, malformed, wrong-week, wrong-contract,
    or failed identity evidence returns no bridge rows.
    """
    root = Path(__file__).resolve().parent
    parquet_path = (
        root
        / "data"
        / "parquet"
        / "current_starter_verification.parquet"
    )
    manifest_path = (
        root
        / "data"
        / "parquet"
        / "current_starter_verification_manifest.json"
    )

    if (
        not parquet_path.is_file()
        or not manifest_path.is_file()
    ):
        return pd.DataFrame()

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return pd.DataFrame()

    expected = (
        manifest.get("contract")
        == "WFS_STARTER_VERIFICATION_CURRENT_V1"
        and manifest.get("source_contract")
        == "WFS_STARTER_VERIFICATION_V1_1"
        and manifest.get("identity_gate") == "PASS"
        and int(manifest.get("season")) == int(season)
        and int(manifest.get("week")) == int(week)
        and str(
            manifest.get("game_type") or ""
        ).strip().upper()
        == str(game_type or "").strip().upper()
        and manifest.get("analysis_only") is True
        and manifest.get("production_influence") is False
    )

    if not expected:
        return pd.DataFrame()

    try:
        evidence = pd.read_parquet(
            parquet_path
        )
    except Exception:
        return pd.DataFrame()

    required = {
        "team",
        "position",
        "depth_gsis_id",
        "rotowire_gsis_id",
        "rw_injury_gate",
        "verification_status",
        "comparable_to_rotowire",
    }

    if not required.issubset(
        evidence.columns
    ):
        return pd.DataFrame()

    if len(evidence) != int(
        manifest.get("role_rows") or -1
    ):
        return pd.DataFrame()

    evidence = evidence.copy()

    for column in {
        "team",
        "position",
        "depth_gsis_id",
        "rotowire_gsis_id",
        "rw_injury_gate",
        "verification_status",
    }:
        evidence[column] = (
            evidence[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    evidence["team"] = (
        evidence["team"].str.upper()
    )
    evidence["position"] = (
        evidence["position"].str.upper()
    )
    evidence["rw_injury_gate"] = (
        evidence["rw_injury_gate"].str.upper()
    )
    evidence["verification_status"] = (
        evidence["verification_status"].str.upper()
    )

    return evidence


def _stat_outlook_resolve_starters(
    team: str,
    playable_offense: list[dict[str, Any]],
    availability_by_player: dict[str, str],
    injury_gate_by_player: dict[str, str],
    season: int,
    week: int,
    game_type: str = "REG",
) -> list[dict[str, Any]]:
    """
    Resolve available offensive starters from authoritative depth lanes.

    A depth-chart player is selectable only when an exact matching
    validated Stat Forecast row already exists.
    """
    if not playable_offense:
        return []

    forecast_by_id = {}

    for row in playable_offense:
        player_id = str(
            row.get("player_id") or ""
        ).strip()

        if not player_id:
            continue

        if player_id in forecast_by_id:
            raise RuntimeError(
                "Duplicate Stat Outlook forecast identity "
                f"for exact player_id {player_id}"
            )

        forecast_by_id[
            player_id
        ] = row

    blocked = {
        str(player_id).strip()
        for player_id, status
        in availability_by_player.items()
        if str(status or "").strip().upper()
        in {
            "OUT",
            "INACTIVE",
        }
        and str(player_id).strip()
    }

    depth = _stat_outlook_depth_rows(
        team
    )

    if depth.empty:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # Each unique football depth lane owns one current starter.
    #
    # For WR this correctly preserves separate receiver slots instead
    # of treating global WR rank 1/2/3 as one interchangeable lane.
    lane_columns = [
        "pos_grp",
        "pos_abb",
        "pos_slot",
    ]

    for _, lane in depth.groupby(
        lane_columns,
        dropna=False,
        sort=True,
    ):
        lane = lane.sort_values(
            [
                "rank_num",
                "player_name",
                "gsis_id",
            ],
            kind="stable",
        )

        chosen_id = None

        for candidate in lane.itertuples(
            index=False
        ):
            player_id = str(
                candidate.gsis_id or ""
            ).strip()

            if not player_id:
                continue

            # Hard current availability always wins over depth chart.
            if player_id in blocked:
                continue

            # First available player in the exact football depth lane.
            chosen_id = player_id
            break

        if not chosen_id:
            continue

        # No fabricated replacement forecast.
        if chosen_id not in forecast_by_id:
            continue

        if chosen_id in selected_ids:
            continue

        selected.append(
            forecast_by_id[
                chosen_id
            ]
        )
        selected_ids.add(
            chosen_id
        )

    # ------------------------------------------------------------
    # WFS_STAT_OUTLOOK_STARTER_VERIFICATION_BRIDGE_V1
    #
    # Presentation-only corroborated role repair.
    #
    # Current depth remains primary. A bridge row may act only when
    # the currently selected player is still the exact depth starter
    # recorded by the validated evidence. This prevents stale evidence
    # from replacing a current same-lane next-man-up.
    #
    # Hard availability always wins. No forecast is created or changed.
    # ------------------------------------------------------------
    verification = (
        _stat_outlook_starter_verification_rows(
            season,
            week,
            game_type,
        )
    )

    if not verification.empty:
        team_key = str(
            team or ""
        ).strip().upper()

        candidates = verification[
            (
                verification["team"]
                == team_key
            )
            & (
                verification[
                    "verification_status"
                ]
                == "DISAGREE_RW_PRIOR_USAGE_SUPPORT"
            )
            & (
                verification[
                    "comparable_to_rotowire"
                ]
                == True
            )
        ].copy()

        for evidence in candidates.itertuples(
            index=False
        ):
            depth_id = str(
                evidence.depth_gsis_id or ""
            ).strip()
            rw_id = str(
                evidence.rotowire_gsis_id or ""
            ).strip()

            if (
                not depth_id
                or not rw_id
                or depth_id == rw_id
            ):
                continue

            # Artifact-time hard availability cannot be overridden.
            if str(
                evidence.rw_injury_gate or ""
            ).strip().upper() == "BLOCK":
                continue

            # Request-time GAV2 authority wins over stale evidence.
            if str(
                injury_gate_by_player.get(
                    rw_id,
                    ""
                )
                or ""
            ).strip().upper() == "BLOCK":
                continue

            if rw_id in blocked:
                continue

            depth_row = forecast_by_id.get(
                depth_id
            )
            rw_row = forecast_by_id.get(
                rw_id
            )

            if (
                depth_row is None
                or rw_row is None
            ):
                continue

            # Bridge may act only if depth_id is still the player
            # actually selected by current depth/availability logic.
            if depth_id not in selected_ids:
                continue

            depth_position = str(
                depth_row.get("position") or ""
            ).strip().upper()
            rw_position = str(
                rw_row.get("position") or ""
            ).strip().upper()
            evidence_position = str(
                evidence.position or ""
            ).strip().upper()

            if (
                depth_position != rw_position
                or rw_position != evidence_position
            ):
                continue

            depth_team = str(
                depth_row.get("team") or ""
            ).strip().upper()
            rw_team = str(
                rw_row.get("team") or ""
            ).strip().upper()

            if (
                depth_team != team_key
                or rw_team != team_key
            ):
                continue

            depth_usage = (
                _stat_outlook_usage_value(
                    depth_row
                )
            )
            rw_usage = (
                _stat_outlook_usage_value(
                    rw_row
                )
            )

            if rw_usage <= depth_usage:
                continue

            selected = [
                row
                for row in selected
                if str(
                    row.get("player_id") or ""
                ).strip()
                != depth_id
            ]
            selected_ids.discard(
                depth_id
            )

            if rw_id not in selected_ids:
                selected.append(
                    rw_row
                )
                selected_ids.add(
                    rw_id
                )

    # ------------------------------------------------------------
    # Conservative projected-usage cross-check.
    #
    # This is intentionally NOT a general "top projection" selector.
    # It only repairs an obviously stale/ambiguous depth result.
    # Hard availability cannot be overridden.
    # ------------------------------------------------------------
    by_position = {}

    for row in playable_offense:
        player_id = str(
            row.get("player_id") or ""
        ).strip()

        if (
            not player_id
            or player_id in blocked
        ):
            continue

        position = str(
            row.get("position") or ""
        ).strip().upper()

        by_position.setdefault(
            position,
            [],
        ).append(
            row
        )

    for position, candidates in by_position.items():
        candidates = sorted(
            candidates,
            key=lambda row: (
                _stat_outlook_usage_value(
                    row
                ),
                str(
                    row.get("player_id")
                    or ""
                ),
            ),
            reverse=True,
        )

        if not candidates:
            continue

        leader = candidates[0]
        leader_id = str(
            leader.get("player_id") or ""
        ).strip()

        leader_usage = (
            _stat_outlook_usage_value(
                leader
            )
        )

        selected_same_position = [
            row
            for row in selected
            if str(
                row.get("position") or ""
            ).strip().upper()
            == position
        ]

        # QB:
        # If current Starter Verification independently resolves depth
        # and RotoWire to the same exact, allowed QB, projected usage
        # must not overthrow that authoritative starter identity.
        protected_qb_ids = set()

        if position == "QB":
            try:
                qb_evidence = (
                    _stat_outlook_starter_verification_rows(
                        season,
                        week,
                        game_type,
                    )
                )

                if not qb_evidence.empty:
                    qb_evidence = qb_evidence[
                        qb_evidence["team"]
                        .astype(str)
                        .str.upper()
                        .eq(team_key)
                        & qb_evidence["position"]
                        .astype(str)
                        .str.upper()
                        .eq("QB")
                    ].copy()

                    for evidence in qb_evidence.itertuples(
                        index=False
                    ):
                        depth_id = str(
                            evidence.depth_gsis_id or ""
                        ).strip()
                        rw_id = str(
                            evidence.rotowire_gsis_id or ""
                        ).strip()

                        if (
                            depth_id
                            and depth_id == rw_id
                            and bool(
                                evidence.comparable_to_rotowire
                            )
                            and str(
                                evidence.verification_status or ""
                            ).strip().upper() == "AGREE"
                            and str(
                                evidence.rw_injury_gate or ""
                            ).strip().upper() == "ALLOW"
                        ):
                            protected_qb_ids.add(rw_id)

            except Exception:
                protected_qb_ids = set()

        # QB:
        # If depth did not resolve a QB, a substantial passing workload
        # is strong starter evidence.
        if position == "QB":
            if (
                not selected_same_position
                and leader_usage >= 20.0
                and leader_id not in selected_ids
            ):
                selected.append(
                    leader
                )
                selected_ids.add(
                    leader_id
                )

            elif (
                len(selected_same_position) == 1
                and leader_id not in selected_ids
                and not any(
                    str(
                        row.get("player_id") or ""
                    ).strip() in protected_qb_ids
                    for row in selected_same_position
                )
            ):
                current = selected_same_position[
                    0
                ]
                current_usage = (
                    _stat_outlook_usage_value(
                        current
                    )
                )

                # Very conservative stale-depth override:
                # alternate must carry clear starter-level workload
                # AND materially exceed the depth-selected QB.
                if (
                    leader_usage >= 24.0
                    and leader_usage
                    >= current_usage + 10.0
                ):
                    current_id = str(
                        current.get(
                            "player_id"
                        )
                        or ""
                    ).strip()

                    selected = [
                        row
                        for row in selected
                        if str(
                            row.get(
                                "player_id"
                            )
                            or ""
                        ).strip()
                        != current_id
                    ]

                    selected_ids.discard(
                        current_id
                    )

                    selected.append(
                        leader
                    )
                    selected_ids.add(
                        leader_id
                    )

        # RB/FB:
        # Only rescue a missing depth starter with clearly meaningful
        # projected opportunity.
        elif position in {
            "RB",
            "FB",
        }:
            if (
                not selected_same_position
                and leader_usage >= 10.0
                and leader_id not in selected_ids
            ):
                selected.append(
                    leader
                )
                selected_ids.add(
                    leader_id
                )

        # TE:
        # A missing TE depth lane may be rescued by clear target volume.
        elif position == "TE":
            if (
                not selected_same_position
                and leader_usage >= 3.0
                and leader_id not in selected_ids
            ):
                selected.append(
                    leader
                )
                selected_ids.add(
                    leader_id
                )

        # WR:
        # Modern 3-WR structures normally produce up to three starting
        # receiver lanes. If the depth chart is incomplete/stale, allow
        # high-target forecast evidence to fill missing WR starter slots.
        elif position == "WR":
            wr_count = len(
                selected_same_position
            )

            if wr_count < 3:
                for candidate in candidates:
                    if wr_count >= 3:
                        break

                    candidate_id = str(
                        candidate.get(
                            "player_id"
                        )
                        or ""
                    ).strip()

                    if (
                        not candidate_id
                        or candidate_id in selected_ids
                        or candidate_id in blocked
                    ):
                        continue

                    usage = (
                        _stat_outlook_usage_value(
                            candidate
                        )
                    )

                    if usage < 3.0:
                        continue

                    selected.append(
                        candidate
                    )
                    selected_ids.add(
                        candidate_id
                    )
                    wr_count += 1

    position_order = {
        "QB": 0,
        "RB": 1,
        "FB": 2,
        "WR": 3,
        "TE": 4,
    }

    selected.sort(
        key=lambda row: (
            position_order.get(
                str(
                    row.get(
                        "position"
                    )
                    or ""
                ).strip().upper(),
                99,
            ),
            -_stat_outlook_usage_value(
                row
            ),
            _stat_player_name(
                row
            ),
        )
    )

    return selected

def _build_stat_outlook(
    packet: dict[str, Any],
) -> dict[str, Any]:
    """
    Build concise public statistical expectations from the
    already-validated statistical forecast evidence.

    No source selection, model execution, forecast mutation,
    fantasy scoring, or live-result adjustment occurs here.
    """
    context = (
        packet.get(
            "stat_forecast_context"
        )
        or {}
    )

    mode = str(
        context.get("mode") or ""
    ).upper()

    result = {
        "available": False,
        "heading": "Stat Outlook",
        "mode": mode,
        "source": (
            "PREGAME"
            if mode == "PREGAME"
            else "KICKOFF"
        ),
        "frozen_at_kickoff": bool(
            context.get(
                "frozen_at_kickoff"
            )
        ),
        "teams": [],
        "note": None,
    }

    if not context.get(
        "available"
    ):
        if mode in {
            "LIVE",
            "POSTGAME",
        }:
            result["note"] = (
                "Kickoff stat outlook is not available "
                "for this game."
            )
        else:
            result["note"] = (
                "Stat outlook is not available "
                "for this game."
            )

        return result

    rows = (
        context.get(
            "rows"
        )
        or []
    )

    # WFS_STAT_OUTLOOK_AVAILABILITY_PRESENTATION_V1
    #
    # Presentation authority only:
    #   forecast row identity -> player_id
    #   injury row identity   -> player_id (exact GSIS identity)
    #
    # Any exact player with injury_gate=BLOCK is displayed as
    # unavailable rather than as a numerical prediction.
    #
    # This includes OUT, INACTIVE, and WFS-policy DOUBTFUL.
    # QUESTIONABLE retains the canonical numerical forecast.
    #
    # No fuzzy matching, display-name matching, team/position fallback,
    # forecast recalculation, or forecast mutation occurs here.
    #
    # Apply only to PREGAME mode. LIVE / POSTGAME Stat Outlook is the
    # frozen kickoff outlook and must not be rewritten by later injury
    # information.
    availability_by_player: dict[str, str] = {}
    injury_gate_by_player: dict[str, str] = {}

    if mode == "PREGAME":
        for injury in (
            packet.get(
                "pregame_impact_injuries"
            )
            or []
        ):
            player_id = str(
                injury.get("player_id")
                or ""
            ).strip()

            if not player_id:
                continue

            status = str(
                injury.get("status")
                or injury.get("report_status")
                or injury.get("consensus_status")
                or ""
            ).strip().upper()

            if player_id in availability_by_player:
                existing = availability_by_player[
                    player_id
                ]

                if existing != status:
                    raise RuntimeError(
                        "Conflicting Stat Outlook injury "
                        "presentation authority for exact "
                        f"player_id {player_id}: "
                        f"{existing} != {status}"
                    )

            availability_by_player[
                player_id
            ] = status

            injury_gate = str(
                injury.get("injury_gate")
                or ""
            ).strip().upper()

            if player_id in injury_gate_by_player:
                existing_gate = (
                    injury_gate_by_player[
                        player_id
                    ]
                )
                if (
                    existing_gate
                    and injury_gate
                    and existing_gate
                    != injury_gate
                ):
                    raise RuntimeError(
                        "Conflicting Stat Outlook GAV2 "
                        "authority for exact player_id "
                        f"{player_id}: "
                        f"{existing_gate} != {injury_gate}"
                    )

            if injury_gate:
                injury_gate_by_player[
                    player_id
                ] = injury_gate

    teams = sorted(
        {
            str(
                row.get("team") or ""
            ).strip().upper()
            for row in rows
            if str(
                row.get("team") or ""
            ).strip()
        }
    )

    team_blocks = []
    selected_players = []

    for team in teams:
        team_rows = [
            row
            for row in rows
            if str(
                row.get("team") or ""
            ).strip().upper()
            == team
        ]

        offense = [
            row
            for row in team_rows
            if str(
                row.get("entity_type") or ""
            ) == "OFFENSE_PLAYER"
            and str(
                row.get("position") or ""
            ).upper()
            in {
                "QB",
                "RB",
                "FB",
                "WR",
                "TE",
            }
        ]

        unavailable_offense = []
        playable_offense = []

        for row in offense:
            player_id = str(
                row.get("player_id")
                or ""
            ).strip()

            status = availability_by_player.get(
                player_id,
                "",
            )

            injury_gate = injury_gate_by_player.get(
                player_id,
                "",
            )
            if (
                injury_gate == "BLOCK"
                or status in {
                    "OUT",
                    "INACTIVE",
                    "DOUBTFUL",
                }
            ):
                unavailable_offense.append(
                    row
                )
            else:
                playable_offense.append(
                    row
                )

        # WFS_STAT_OUTLOOK_COLD_START_FALLBACK_V1
        #
        # PREGAME only: if the authoritative starter has no
        # canonical row, expose an exact cold-start rookie row to
        # the existing starter resolver.
        #
        # Canonical rows remain primary and immutable.
        if mode == "PREGAME":
            game_ids = {
                str(
                    row.get(
                        "game_id"
                    )
                    or ""
                )
                for row in team_rows
                if str(
                    row.get(
                        "game_id"
                    )
                    or ""
                )
            }

            existing_player_ids = {
                str(
                    row.get(
                        "player_id"
                    )
                    or ""
                ).strip()
                for row in playable_offense
                if str(
                    row.get(
                        "player_id"
                    )
                    or ""
                ).strip()
            }

            cold_start_rows = (
                _stat_outlook_cold_start_rows(
                    team,
                    game_ids,
                    existing_player_ids,
                    availability_by_player,
                )
            )

            if cold_start_rows:
                playable_offense.extend(
                    cold_start_rows
                )

        # WFS_STAT_OUTLOOK_STARTER_SELECTION_V1
        #
        # Starter-aware public selection:
        #   availability/news -> depth lane -> next man up ->
        #   projected-usage confirmation.
        #
        # Forecast rows remain immutable.
        if mode == "PREGAME":
            selected = (
                _stat_outlook_resolve_starters(
                    team,
                    playable_offense,
                    availability_by_player,
                    injury_gate_by_player,
                    int(
                        packet["game"]["season"]
                    ),
                    int(
                        packet["game"]["week"]
                    ),
                    str(
                        packet["game"].get(
                            "game_type"
                        )
                        or "REG"
                    ),
                )
            )

            # Fail-soft presentation fallback only when the authoritative
            # depth source cannot resolve any eligible forecast rows.
            # This preserves the previous public behavior without
            # inventing starter identities.
            if not selected:
                quarterbacks = sorted(
                    [
                        row
                        for row in playable_offense
                        if str(
                            row.get(
                                "position"
                            )
                            or ""
                        ).upper()
                        == "QB"
                    ],
                    key=_stat_outlook_sort_value,
                    reverse=True,
                )

                skill = sorted(
                    [
                        row
                        for row in playable_offense
                        if str(
                            row.get(
                                "position"
                            )
                            or ""
                        ).upper()
                        in {
                            "RB",
                            "FB",
                            "WR",
                            "TE",
                        }
                    ],
                    key=_stat_outlook_sort_value,
                    reverse=True,
                )

                selected = []

                if quarterbacks:
                    selected.append(
                        quarterbacks[0]
                    )

                selected.extend(
                    skill[:3]
                )

        else:
            # LIVE / POSTGAME use the frozen kickoff Stat Outlook
            # selection behavior and are not rewritten by current depth
            # or injury information.
            quarterbacks = sorted(
                [
                    row
                    for row in playable_offense
                    if str(
                        row.get(
                            "position"
                        )
                        or ""
                    ).upper()
                    == "QB"
                ],
                key=_stat_outlook_sort_value,
                reverse=True,
            )

            skill = sorted(
                [
                    row
                    for row in playable_offense
                    if str(
                        row.get(
                            "position"
                        )
                        or ""
                    ).upper()
                    in {
                        "RB",
                        "FB",
                        "WR",
                        "TE",
                    }
                ],
                key=_stat_outlook_sort_value,
                reverse=True,
            )

            selected = []

            if quarterbacks:
                selected.append(
                    quarterbacks[0]
                )

            selected.extend(
                skill[:3]
            )

        # WFS_GLOBAL_PLAYER_AVAILABILITY_V2
        #
        # Confirmed hard-unavailable players are omitted from the
        # public Stat Outlook. Injury/news context remains available
        # through the injury analysis surface. No replacement
        # numerical forecast is created here.
        # WFS_STAT_OUTLOOK_TD_DISPLAY_RECONCILIATION_V1
        #
        # Presentation-only integer TD consistency.
        selected = (
            _stat_outlook_reconcile_display_tds(
                selected
            )
        )

        # WFS_STAT_OUTLOOK_EVENT_DISPLAY_RECONCILIATION_V1
        #
        # Presentation-only whole-number event coherence.
        # Canonical fractional forecasts remain unchanged.
        selected = (
            _stat_outlook_reconcile_display_events(
                selected
            )
        )

        offense_lines = []

        for row in selected:
            line = (
                _format_stat_outlook_row(
                    row
                )
            )

            if line:
                offense_lines.append(
                    line
                )
                selected_players.append({
                    "player_id": str(row.get("player_id") or "").strip(),
                    "player_name": str(row.get("player_name") or row.get("entity_name") or "").strip(),
                    "position": str(row.get("position") or "").strip().upper(),
                    "team": team,
                    "forecast_line": line,
                })

        kicker_rows = [
            row
            for row in team_rows
            if str(
                row.get("entity_type") or ""
            ) == "KICKER"
        ]

        kicker_line = None

        if len(kicker_rows) == 1:
            kicker_line = (
                _format_kicker_outlook(
                    kicker_rows[0],
                    max_tds=_team_offense_td_total(selected),
                )
            )

        dst_rows = [
            row
            for row in team_rows
            if str(
                row.get("entity_type") or ""
            ) == "DST"
        ]

        dst_line = None

        if len(dst_rows) == 1:
            dst_line = (
                _format_dst_outlook(
                    dst_rows[0]
                )
            )

        team_blocks.append({
            "team":
                team,

            "offense":
                offense_lines,

            "kicker":
                kicker_line,

            "dst":
                dst_line,
        })

    result.update({
        "available":
            True,

        "teams":
            team_blocks,

        "selected_players":
            selected_players,

        "note":
            (
                "Original kickoff outlook."
                if mode in {
                    "LIVE",
                    "POSTGAME",
                }
                else None
            ),
    })

    return result


def _build_pregame_response(
    packet: dict[str, Any],
) -> dict[str, Any]:
    game = packet["game"]
    forecast = packet["forecast"]
    pregame_injuries = packet.get(
        "pregame_impact_injuries", []
    )

    away = str(game.get("away_team") or "")
    home = str(game.get("home_team") or "")

    if not forecast.get("ready"):
        return {
            "mode": "PREGAME",
            "headline": f"{away} at {home}",
            "game_summary": (
                "A verified WFS forecast is not "
                "available for this matchup yet."
            ),
            "fantasy_leaders": [],
            "impact_injuries": pregame_injuries,
            "observations": _pregame_injury_observations(
                pregame_injuries
            )[:8],
            "evidence_status": {
                "forecast_available": (
                    forecast.get("available") is True
                ),
                "forecast_ready": False,
                "win_probability_available": False,
            },
        }

    winner = forecast.get("pred_winner")
    home_points = forecast.get("pred_home_points")
    away_points = forecast.get("pred_away_points")
    total = forecast.get("pred_total_points")

    observations = []

    if total is not None:
        observations.append(
            f"The WFS projected game total is "
            f"{_fmt_number(total)} points."
        )

    if (
        home_points is not None
        and away_points is not None
    ):
        observations.append(
            f"The projected score is "
            f"{away} {_fmt_number(away_points)} to "
            f"{home} {_fmt_number(home_points)}."
        )

    return {
        "mode": "PREGAME",
        "headline": (
            f"WFS projects {winner} over "
            f"{home if winner == away else away}"
            if winner
            else f"{away} at {home}"
        ),
        "game_summary": (
            f"WFS currently favors {winner}."
            if winner
            else
            "The matchup has a ready forecast."
        ),
        "fantasy_leaders": [],
        "impact_injuries": pregame_injuries,
        "observations": (
            _pregame_injury_observations(
                pregame_injuries
            )
            + observations
        )[:8],
        "evidence_status": {
            "forecast_available": True,
            "forecast_ready": True,
            "forecast_status": forecast.get("status"),
            "win_probability_status": forecast.get(
                "win_probability_status"
            ),
            "win_probability_available": (
                forecast.get("win_probability")
                is not None
            ),
        },
    }


def _build_live_response(
    packet: dict[str, Any],
) -> dict[str, Any]:
    game = packet["game"]
    live = packet["live_context"]

    away = str(
        live.get("away_team")
        or game.get("away_team")
        or ""
    )

    home = str(
        live.get("home_team")
        or game.get("home_team")
        or ""
    )

    if not live.get("event_matched"):
        return {
            "mode": "LIVE",
            "headline": f"{away} at {home}",
            "game_summary": (
                "Verified live game data is not "
                "available yet."
            ),
            "fantasy_leaders": [],
            "impact_injuries": [],
            "observations": [],
            "evidence_status": {
                "live_event_verified": False,
            },
        }

    period = live.get("period")
    clock = live.get("clock")

    status_parts = []

    if period is not None:
        status_parts.append(f"Q{period}")

    if clock:
        status_parts.append(str(clock))

    status = " ".join(status_parts)

    summary = (
        f"{away} {live.get('away_score')}, "
        f"{home} {live.get('home_score')}"
    )

    if status:
        summary += f" — {status}"

    recent = []

    for play in live.get("recent_plays", [])[-5:]:
        text = str(play.get("play_text") or "").strip()

        if text:
            recent.append(text)

    return {
        "mode": "LIVE",
        "headline": f"Live: {away} at {home}",
        "game_summary": summary,
        "fantasy_leaders": [],
        "impact_injuries": packet.get("impact_injuries", []),
        "observations": (
            _injury_observations(packet.get("impact_injuries", []))
            + recent
        )[:8],
        "evidence_status": {
            "live_event_verified": True,
            "live_state": live.get("state"),
            "play_count": live.get("play_count"),
        },
    }


def build_analyst_response(
    game_id: str,
) -> dict[str, Any]:
    """
    Produce deterministic football-language analysis from the
    verified WFS evidence packet.

    No LLM is used by this function.
    """
    packet = build_game_evidence(game_id)

    live = packet["live_context"]
    game = packet["game"]

    live_state = str(
        live.get("state") or ""
    ).lower()

    completed = bool(game.get("completed"))

    # WFS_ANALYST_RESPONSE_STATE_AUTHORITY_V2
    # Completed game authority outranks a stale SAFE_FAIL live row.
    if completed or live_state == "post":
        response = _build_postgame_response(packet)
    elif (
        live.get("event_matched")
        and live_state in {"in", "live"}
    ):
        response = _build_live_response(packet)
    else:
        response = _build_pregame_response(packet)

    response["stat_outlook"] = _build_stat_outlook(packet)

    response["contract"] = (
        "WFS_AI_ANALYST_RESPONSE_V1"
    )

    response["game_id"] = str(game_id)

    response["guardrails"] = {
        "deterministic": True,
        "llm_used": False,
        "read_only": True,
        "invent_probability_allowed": False,
        "individual_defensive_players_in_scope": False,
    }

    return response
