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

from pathlib import Path
import re
import sqlite3
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent

NFL_DB = ROOT / "data" / "nfl.db"
LIVE_DB = ROOT / "data" / "wfs_live.db"

FORECAST_CSV = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
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
    if not FORECAST_CSV.is_file():
        return None

    df = pd.read_csv(
        FORECAST_CSV,
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
        status == "READY_CORE_ONLY"
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

    return {
        "contract": "WFS_AI_ANALYST_EVIDENCE_V1",
        "game_id": game_id,
        "game": game,
        "forecast": _forecast_evidence(forecast),
        "team_context": load_team_context(game_id),
        "player_context": player_context,
        "live_context": load_live_context(game_id),
        "impact_injuries": load_impact_injuries(game_id, player_context),
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

    away_score = live.get("away_score")
    home_score = live.get("home_score")

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


def _build_pregame_response(
    packet: dict[str, Any],
) -> dict[str, Any]:
    game = packet["game"]
    forecast = packet["forecast"]

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
            "impact_injuries": [],
            "observations": [],
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
        "impact_injuries": [],
        "observations": observations,
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

    if live_state == "post" or completed:
        response = _build_postgame_response(packet)

    elif (
        live.get("event_matched")
        and live_state in {"in", "live"}
    ):
        response = _build_live_response(packet)

    else:
        response = _build_pregame_response(packet)

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
