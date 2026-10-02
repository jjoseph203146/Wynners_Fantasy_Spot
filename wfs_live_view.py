#!/usr/bin/env python3
"""
WFS NFL LIVE — APP READ-ONLY VIEW

Presentation-only Streamlit adapter for WFS_LIVE_DB_V2.

STRICT BOUNDARY:
    - Opens wfs_live.db with SQLite mode=ro.
    - Enables PRAGMA query_only.
    - Performs SELECT/PRAGMA reads only.
    - Never modifies LIVE persistence.
    - Never modifies nfl.db.
    - Never modifies forecast_ledger.db.
    - Never invokes live ingest/orchestrator/poller processes.
"""

from __future__ import annotations

import json
import urllib.request
import html
import textwrap
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
LIVE_DB_PATH = APP_DIR / "data" / "wfs_live.db"
NFL_DB_PATH = APP_DIR / "data" / "nfl.db"

EXPECTED_SCHEMA_VERSION = "WFS_LIVE_DB_V2"

REQUIRED_TABLES = {
    "live_meta",
    "live_events",
    "live_athletes",
    "live_plays",
    "live_play_players",
    "live_ingest_audit",
}


def _live_connect() -> sqlite3.Connection:
    """
    Open the LIVE database strictly read-only.

    mode=ro is the physical SQLite write barrier.
    PRAGMA query_only adds a second connection-level guard.
    """
    if not LIVE_DB_PATH.is_file():
        raise FileNotFoundError(
            f"LIVE database does not exist: {LIVE_DB_PATH}"
        )

    conn = sqlite3.connect(
        f"file:{LIVE_DB_PATH}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn



def _nfl_connect() -> sqlite3.Connection:
    """Open nfl.db strictly read-only."""
    if not NFL_DB_PATH.is_file():
        raise FileNotFoundError(
            f"NFL database does not exist: {NFL_DB_PATH}"
        )
    conn = sqlite3.connect(
        f"file:{NFL_DB_PATH}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn

def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _verify_live_contract(conn: sqlite3.Connection) -> None:
    tables = _table_names(conn)

    missing = REQUIRED_TABLES - tables
    if missing:
        raise RuntimeError(
            "LIVE database schema is incomplete. Missing: "
            + ", ".join(sorted(missing))
        )

    row = conn.execute(
        """
        SELECT schema_version
        FROM live_meta
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        raise RuntimeError("LIVE schema metadata is missing.")

    actual = str(row["schema_version"] or "").strip()

    if actual != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unexpected LIVE schema version: {actual!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION!r}"
        )


def _read_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    WFS_LIVE_SCHEDULE_BACKED_PRE_V1

    Build presentation inventory from the authoritative NFL schedule,
    then overlay persisted LIVE rows by exact ESPN event_id.

    PRE games remain presentation-only. This does not seed live_events.
    """
    live = pd.read_sql_query(
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
            home_team_id,
            home_team,
            home_score,
            away_team_id,
            away_team,
            away_score,
            current_drive_id,
            last_play_id,
            updated_at_utc
        FROM live_events
        """,
        conn,
    )

    with _nfl_connect() as nfl_conn:
        schedule = pd.read_sql_query(
            """
            SELECT
                CAST(espn AS TEXT) AS event_id,
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
                updated_at
            FROM games
            WHERE espn IS NOT NULL
              AND TRIM(CAST(espn AS TEXT)) <> ''
              AND season = (
                  SELECT MAX(season)
                  FROM games
                  WHERE espn IS NOT NULL
              )
            """,
            nfl_conn,
        )

    columns = list(live.columns)

    if schedule.empty:
        out = live.copy()
    else:
        # Schedule authority determines the planning week:
        # the earliest week in the latest season containing an
        # incomplete game. This remains schedule-driven.
        incomplete = schedule[
            pd.to_numeric(
                schedule["completed"],
                errors="coerce",
            ).fillna(0).astype(int).eq(0)
        ]

        if incomplete.empty:
            planning_week = int(
                pd.to_numeric(
                    schedule["week"],
                    errors="coerce",
                ).dropna().max()
            )
        else:
            planning_week = int(
                pd.to_numeric(
                    incomplete["week"],
                    errors="coerce",
                ).dropna().min()
            )

        scheduled = schedule[
            pd.to_numeric(
                schedule["week"],
                errors="coerce",
            ).eq(planning_week)
        ].copy()

        pre = pd.DataFrame(index=scheduled.index, columns=columns)
        pre["event_id"] = scheduled["event_id"].astype(str)
        pre["season"] = scheduled["season"]
        pre["season_type"] = scheduled["game_type"].map(
            {"REG": 2, "POST": 3}
        )
        pre["week"] = scheduled["week"]
        pre["state"] = "pre"
        pre["detail"] = scheduled.apply(
            lambda r: _format_kickoff(
                r.get("game_date"),
                r.get("weekday"),
                r.get("gametime"),
            ),
            axis=1,
        )
        pre["period"] = None
        pre["clock"] = None
        pre["home_team_id"] = None
        pre["home_team"] = scheduled["home_team"]
        pre["home_score"] = scheduled["home_score"]
        pre["away_team_id"] = None
        pre["away_team"] = scheduled["away_team"]
        pre["away_score"] = scheduled["away_score"]
        pre["current_drive_id"] = None
        pre["last_play_id"] = None
        pre["updated_at_utc"] = scheduled["updated_at"]

        # Fail closed on an unknown season type rather than inventing
        # a LIVE season_type value.
        pre = pre[pre["season_type"].notna()].copy()

        # LIVE wins for the same exact ESPN event_id.
        live_ids = set(live["event_id"].astype(str))
        pre = pre[
            ~pre["event_id"].astype(str).isin(live_ids)
        ]

        out = pd.concat(
            [live, pre],
            ignore_index=True,
        )

    if out.empty:
        return out

    state_rank = out["state"].map(
        {"in": 0, "pre": 1, "post": 2}
    ).fillna(3)

    out = (
        out.assign(_state_rank=state_rank)
        .sort_values(
            ["_state_rank", "week", "updated_at_utc", "event_id"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        .drop(columns=["_state_rank"])
        .reset_index(drop=True)
    )

    return out



def _apply_completed_game_authority(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    WFS_LIVE_COMPLETED_GAME_AUTHORITY_V1

    Completed nfl.db games override stale LIVE state and score
    in the presentation layer only.
    Exact ESPN event_id matching only.
    """
    if events.empty:
        return events

    event_ids = [
        str(x)
        for x in events["event_id"].astype(str).tolist()
    ]
    if not event_ids:
        return events

    placeholders = ",".join("?" for _ in event_ids)

    with _nfl_connect() as conn:
        rows = pd.read_sql_query(
            f"""
            SELECT
                CAST(espn AS TEXT) AS event_id,
                away_score,
                home_score,
                completed
            FROM games
            WHERE CAST(espn AS TEXT) IN ({placeholders})
            """,
            conn,
            params=event_ids,
        )

    if rows.empty:
        return events

    authority = {}
    for _, row in rows.iterrows():
        try:
            completed = int(row.get("completed") or 0)
        except Exception:
            completed = 0
        if completed == 1:
            authority[str(row["event_id"])] = row

    if not authority:
        return events

    out = events.copy()

    for idx, event in out.iterrows():
        event_id = str(event["event_id"])
        final_row = authority.get(event_id)
        if final_row is None:
            continue

        out.at[idx, "state"] = "post"
        out.at[idx, "detail"] = "Final"
        out.at[idx, "clock"] = "0:00"
        out.at[idx, "period"] = 0

        if pd.notna(final_row.get("away_score")):
            out.at[idx, "away_score"] = final_row["away_score"]

        if pd.notna(final_row.get("home_score")):
            out.at[idx, "home_score"] = final_row["home_score"]

    return out


@st.cache_data(ttl=60, show_spinner=False)
def _completed_espn_plays(event_id: str) -> pd.DataFrame:
    """
    WFS_LIVE_COMPLETED_ESPN_PBP_V1

    Read-only ESPN fallback for completed-game play-by-play.
    Returns the existing WFS play schema and performs no persistence.
    """
    event_id = str(event_id or '').strip()
    if not event_id:
        return pd.DataFrame()

    url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/football/nfl/summary"
        f"?event={event_id}"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WFS-LIVE-RUNNER",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read())
    except Exception:
        return pd.DataFrame()

    drives = data.get('drives') or {}
    all_drives = []

    if isinstance(drives, dict):
        for key in ('previous', 'current'):
            value = drives.get(key)
            if isinstance(value, list):
                all_drives.extend(value)
            elif isinstance(value, dict):
                all_drives.append(value)

    rows = []
    seen = set()

    for drive in all_drives:
        if not isinstance(drive, dict):
            continue

        for play in drive.get('plays') or []:
            if not isinstance(play, dict):
                continue

            play_id = str(play.get('id') or '').strip()
            if not play_id or play_id in seen:
                continue
            seen.add(play_id)

            period_raw = play.get('period')
            if isinstance(period_raw, dict):
                period = period_raw.get('number')
            else:
                period = period_raw

            clock_raw = play.get('clock')
            if isinstance(clock_raw, dict):
                clock = clock_raw.get('displayValue')
            else:
                clock = clock_raw

            rows.append({
                'event_id': event_id,
                'play_id': play_id,
                'sequence_number': play.get('sequenceNumber'),
                'period': period,
                'clock': clock,
                'home_score': play.get('homeScore'),
                'away_score': play.get('awayScore'),
                'play_text': play.get('text'),
                'is_scoring_play': int(bool(play.get('scoringPlay'))),
                'is_turnover': int(bool(play.get('isTurnover'))),
                'is_penalty': int(bool(play.get('isPenalty'))),
                'first_seen_at_utc': None,
                'updated_at_utc': None,
            })

    return pd.DataFrame(rows)


def _augment_completed_plays_from_espn(
    stored: pd.DataFrame,
    event_id: str,
) -> pd.DataFrame:
    """Fill missing completed-game plays without mutating LIVE storage."""
    espn = _completed_espn_plays(event_id)

    if espn.empty:
        return stored

    if stored.empty:
        combined = espn.copy()
    else:
        combined = pd.concat(
            [stored.copy(), espn],
            ignore_index=True,
            sort=False,
        )

    combined['_play_key'] = combined['play_id'].astype(str)
    combined = combined.drop_duplicates(
        subset=['_play_key'],
        keep='first',
    )

    combined['_period_sort'] = pd.to_numeric(
        combined['period'],
        errors='coerce',
    ).fillna(0)
    combined['_sequence_sort'] = pd.to_numeric(
        combined['sequence_number'],
        errors='coerce',
    ).fillna(0)

    combined = combined.sort_values(
        ['_period_sort', '_sequence_sort', '_play_key'],
        kind='stable',
    )

    return combined.drop(
        columns=['_play_key', '_period_sort', '_sequence_sort'],
        errors='ignore',
    ).reset_index(drop=True)

def _read_audit(
    conn: sqlite3.Connection,
    event_id: str,
) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            ingest_id,
            event_id,
            captured_at_utc,
            event_state,
            raw_play_rows,
            unique_play_rows,
            duplicates_removed,
            boxscore_athletes,
            roster_athletes,
            merged_athletes,
            resolved_occurrences,
            ambiguous_occurrences,
            unresolved_occurrences,
            ingest_status
        FROM live_ingest_audit
        WHERE event_id = ?
        ORDER BY ingest_id DESC
        LIMIT 20
        """,
        conn,
        params=(event_id,),
    )


def _read_plays(
    conn: sqlite3.Connection,
    event_id: str,
) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            event_id,
            play_id,
            sequence_number,
            period,
            clock,
            home_score,
            away_score,
            play_text,
            is_scoring_play,
            is_turnover,
            is_penalty,
            first_seen_at_utc,
            updated_at_utc
        FROM live_plays
        WHERE event_id = ?
        ORDER BY
            COALESCE(period, 0) ASC,
            COALESCE(sequence_number, 0) ASC,
            play_id ASC
        """,
        conn,
        params=(event_id,),
    )


def _read_play_players(
    conn: sqlite3.Connection,
    event_id: str,
) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            ppp.event_id,
            ppp.play_id,
            ppp.token_ordinal,
            ppp.raw_token,
            ppp.identity_key,
            ppp.athlete_id,
            ppp.resolution_method,
            ppp.identity_source,
            a.display_name,
            a.team,
            a.position
        FROM live_play_players AS ppp
        LEFT JOIN live_athletes AS a
            ON a.athlete_id = ppp.athlete_id
        WHERE ppp.event_id = ?
        ORDER BY
            ppp.play_id,
            ppp.token_ordinal
        """,
        conn,
        params=(event_id,),
    )


def _read_athletes(
    conn: sqlite3.Connection,
    team_ids: List[str],
) -> pd.DataFrame:
    if not team_ids:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in team_ids)

    return pd.read_sql_query(
        f"""
        SELECT
            athlete_id,
            display_name,
            team_id,
            team,
            identity_key,
            identity_source,
            position,
            jersey,
            updated_at_utc
        FROM live_athletes
        WHERE team_id IN ({placeholders})
        ORDER BY team, position, display_name
        """,
        conn,
        params=tuple(team_ids),
    )


def _players_by_play(
    refs: pd.DataFrame,
) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}

    if refs.empty:
        return output

    for play_id, group in refs.groupby("play_id", sort=False):
        names: List[str] = []

        for _, row in group.sort_values("token_ordinal").iterrows():
            name = str(row.get("display_name") or "").strip()

            if not name:
                name = str(row.get("raw_token") or "").strip()

            if name and name not in names:
                names.append(name)

        output[str(play_id)] = names

    return output


def _format_kickoff(date_str: str, weekday: str = "", time_str: str = "") -> str:
    """Format a schedule date/weekday/24h time into e.g. 'Sun, Sep 13 - 1:00 PM ET'."""
    from datetime import datetime

    date_str = str(date_str or "").strip()
    weekday = str(weekday or "").strip()[:3]
    time_str = str(time_str or "").strip()

    date_part = date_str
    try:
        date_part = datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %-d")
    except Exception:
        pass

    time_part = ""
    if time_str:
        try:
            hh, mm = time_str.split(":")[:2]
            hh = int(hh)
            mm = int(mm)
            period = "AM" if hh < 12 else "PM"
            hh12 = hh % 12 or 12
            time_part = f"{hh12}:{mm:02d} {period} ET"
        except Exception:
            time_part = time_str

    header = ", ".join(p for p in (weekday, date_part) if p)
    if header and time_part:
        return f"{header} • {time_part}"
    return header or time_part or date_str


def _event_state_label(state: str) -> str:
    state = str(state or "").lower()

    if state == "in":
        return "🔴 LIVE"

    if state == "post":
        return "✅ FINAL"

    if state == "pre":
        return "🕒 UPCOMING"

    return state.upper() or "UNKNOWN"


def _safe(value) -> str:
    if value is None:
        return "—"

    text = str(value).strip()
    return html.escape(text if text else "—")


def _score(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return "—"


def _render_event_header(event: pd.Series) -> None:
    away = _safe(event.get("away_team"))
    home = _safe(event.get("home_team"))

    away_score = _score(event.get("away_score"))
    home_score = _score(event.get("home_score"))

    state = _event_state_label(event.get("state"))
    detail = _safe(event.get("detail"))
    clock = _safe(event.get("clock"))

    if str(event.get('state') or '').lower() == 'post':
        period_text = "FINAL"
        clock = ""
    else:
        try:
            period = int(event.get("period"))
            period_text = f"Q{period}"
        except Exception:
            period_text = "—"

    header_html = (
        '<div style="background:#0f172a;'
        'border:1px solid #334155;'
        'border-radius:18px;'
        'padding:1rem 1.1rem;'
        'margin:.35rem 0 .8rem;'
        'color:#f8fafc;">'

        '<div style="font-size:.72rem;'
        'font-weight:900;'
        'letter-spacing:.09em;'
        'color:#f87171;'
        'margin-bottom:.5rem;">'
        f'{state}'
        '</div>'

        '<div style="display:grid;'
        'grid-template-columns:1fr auto 1fr;'
        'align-items:center;'
        'gap:.8rem;">'

        '<div>'
        '<div style="font-size:1.15rem;'
        'font-weight:900;">'
        f'{away}'
        '</div>'
        '<div style="font-size:2rem;'
        'font-weight:950;">'
        f'{away_score}'
        '</div>'
        '</div>'

        '<div style="text-align:center;'
        'color:#94a3b8;'
        'font-size:.78rem;'
        'font-weight:800;">'
        f'{period_text}<br>{clock}'
        '</div>'

        '<div style="text-align:right;">'
        '<div style="font-size:1.15rem;'
        'font-weight:900;">'
        f'{home}'
        '</div>'
        '<div style="font-size:2rem;'
        'font-weight:950;">'
        f'{home_score}'
        '</div>'
        '</div>'

        '</div>'

        '<div style="margin-top:.65rem;'
        'padding-top:.55rem;'
        'border-top:1px solid #334155;'
        'color:#cbd5e1;'
        'font-size:.82rem;">'
        f'{detail}'
        '</div>'

        '</div>'
    )

    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )


def _render_ingest_health(audit: pd.DataFrame) -> None:
    st.markdown("#### LIVE Ingest Health")

    if audit.empty:
        st.warning("No LIVE ingest audit row exists for this event.")
        return

    latest = audit.iloc[0]

    status = str(latest.get("ingest_status") or "UNKNOWN")
    ambiguous = int(latest.get("ambiguous_occurrences") or 0)
    unresolved = int(latest.get("unresolved_occurrences") or 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ingest", status)
    c2.metric(
        "Unique Plays",
        int(latest.get("unique_play_rows") or 0),
    )
    c3.metric("Ambiguous", ambiguous)
    c4.metric("Unresolved", unresolved)

    if (
        status == "PASS"
        and ambiguous == 0
        and unresolved == 0
    ):
        st.success(
            "LIVE identity and ingest contract is clean."
        )
    else:
        st.error(
            "LIVE ingest requires attention. "
            "The UI will not reinterpret unresolved identity data."
        )

    st.caption(
        "Last LIVE capture: "
        + str(latest.get("captured_at_utc") or "unknown")
        + " UTC"
    )


def _render_play_feed(
    plays: pd.DataFrame,
    refs: pd.DataFrame,
) -> None:
    st.markdown("#### Full Game Play-by-Play")

    if plays.empty:
        st.info("No plays have been stored for this event yet.")
        return

    player_map = _players_by_play(refs)

    period_labels = {
        1: "1st Quarter",
        2: "2nd Quarter",
        3: "3rd Quarter",
        4: "4th Quarter",
        5: "OT",
    }

    working = plays.copy()

    working["_period_num"] = pd.to_numeric(
        working["period"],
        errors="coerce",
    )

    available_periods = sorted(
        int(period)
        for period in working["_period_num"].dropna().unique()
    )

    if not available_periods:
        st.info("No quarter information is available for these plays.")
        return

    tab_labels = []

    for period in available_periods:
        if period in period_labels:
            tab_labels.append(period_labels[period])
        elif period > 5:
            tab_labels.append(f"OT {period - 4}")
        else:
            tab_labels.append(f"Period {period}")

    quarter_tabs = st.tabs(tab_labels)

    for period, quarter_tab in zip(
        available_periods,
        quarter_tabs,
    ):
        with quarter_tab:
            quarter_plays = working[
                working["_period_num"].eq(period)
            ]

            for _, play in quarter_plays.iterrows():
                flags: List[str] = []

                if int(play.get("is_scoring_play") or 0):
                    flags.append("🏈 SCORE")

                if int(play.get("is_turnover") or 0):
                    flags.append("🔄 TURNOVER")

                if int(play.get("is_penalty") or 0):
                    flags.append("🚩 PENALTY")

                flag_text = " · ".join(flags)

                play_period = play.get("period")
                clock = _safe(play.get("clock"))

                try:
                    period_text = f"Q{int(play_period)}"
                except Exception:
                    period_text = "—"

                score_text = (
                    f'{_score(play.get("away_score"))}'
                    f'–{_score(play.get("home_score"))}'
                )

                play_id = str(play.get("play_id"))
                players = player_map.get(play_id, [])

                player_text = " · ".join(
                    str(player)
                    for player in players
                    if str(player).strip()
                )

                extra = ""

                if flag_text:
                    extra += (
                        '<div style="color:#f59e0b;font-size:.72rem;'
                        'font-weight:850;margin-top:.35rem;">'
                        f'{html.escape(flag_text)}</div>'
                    )

                if player_text:
                    extra += (
                        '<div style="color:#93c5fd;font-size:.72rem;'
                        'margin-top:.25rem;">'
                        f'{html.escape(player_text)}</div>'
                    )

                card_html = (
                    '<div style="background:#ffffff;border:1px solid #dbe4ee;'
                    'border-radius:12px;padding:.7rem .8rem;margin:.42rem 0;">'
                    '<div style="display:flex;justify-content:space-between;'
                    'gap:.8rem;color:#64748b;font-size:.68rem;font-weight:850;">'
                    f'<span>{period_text} · {clock}</span>'
                    f'<span>{score_text}</span>'
                    '</div>'
                    '<div style="margin-top:.27rem;color:#172033;font-size:.86rem;'
                    'line-height:1.45;font-weight:600;">'
                    f'{_safe(play.get("play_text"))}'
                    '</div>'
                    f'{extra}'
                    '</div>'
                )

                st.markdown(card_html, unsafe_allow_html=True)


# =====================================================================
# WFS_NFL_LIVE_WEATHER_PRESENTATION_V1
#
# Presentation-only weather integration.
#
# Contracts:
# - nfl.db opened through the existing read-only _nfl_connect().
# - authoritative game identity comes from games.
# - weather comes from existing weather_period_snapshots /
#   weather_game_features_v2.
# - latest source snapshot wins deterministically.
# - no database writes.
# - no projection/model/optimizer influence.
# =====================================================================

def _weather_period_for_event(event: pd.Series) -> str:
    """Resolve the most useful weather period for the current game state."""
    state = str(event.get("state") or "").strip().lower()

    if state == "pre":
        return "Kickoff"

    try:
        period = int(float(event.get("period") or 0))
    except (TypeError, ValueError):
        period = 0

    if period <= 1:
        return "Kickoff"
    if period == 2:
        return "Q2"
    if period == 3:
        return "Q3"

    # Q4 is the closest available weather period for Q4/OT/postgame.
    return "Q4"


def _read_live_weather(event: pd.Series) -> dict | None:
    """
    Read the latest authoritative weather capture for this schedule game.

    This function is deliberately presentation-only. It does not write,
    mutate, score, adjust or otherwise influence any forecast.
    """
    season = event.get("season")
    week = event.get("week")
    away_team = str(event.get("away_team") or "").strip().upper()
    home_team = str(event.get("home_team") or "").strip().upper()

    if not away_team or not home_team:
        return None

    try:
        season_i = int(float(season))
        week_i = int(float(week))
    except (TypeError, ValueError):
        return None

    desired_period = _weather_period_for_event(event)

    with _nfl_connect() as conn:
        tables = _table_names(conn)

        if "games" not in tables:
            return None

        game = conn.execute(
            """
            SELECT
                game_id,
                roof,
                stadium,
                surface
            FROM games
            WHERE season = ?
              AND week = ?
              AND UPPER(TRIM(away_team)) = ?
              AND UPPER(TRIM(home_team)) = ?
            ORDER BY game_id
            LIMIT 1
            """,
            (
                season_i,
                week_i,
                away_team,
                home_team,
            ),
        ).fetchone()

        if game is None:
            return None

        game_id = str(game["game_id"] or "").strip()

        if not game_id:
            return None

        # Prefer the latest captured feature because source_captured_at
        # establishes deterministic weather-capture chronology.
        latest_feature = None

        if "weather_game_features_v2" in tables:
            latest_feature = conn.execute(
                """
                SELECT
                    feature_id,
                    source_snapshot_id,
                    source_captured_at,
                    source,
                    stadium_type,
                    kickoff_condition,
                    q4_condition,
                    kickoff_temperature_f,
                    kickoff_feels_like_f,
                    kickoff_wind_mph,
                    kickoff_wind_direction,
                    kickoff_gust_mph,
                    kickoff_precipitation_probability_pct,
                    kickoff_humidity_pct
                FROM weather_game_features_v2
                WHERE game_id = ?
                ORDER BY
                    datetime(source_captured_at) DESC,
                    feature_id DESC
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

        weather = {
            "game_id": game_id,
            "period": desired_period,
            "roof": game["roof"],
            "stadium": game["stadium"],
            "surface": game["surface"],
        }

        if latest_feature is not None:
            weather.update(dict(latest_feature))

        source_snapshot_id = (
            latest_feature["source_snapshot_id"]
            if latest_feature is not None
            else None
        )

        period_row = None

        if (
            source_snapshot_id is not None
            and "weather_period_snapshots" in tables
        ):
            period_row = conn.execute(
                """
                SELECT
                    snapshot_id,
                    period,
                    condition,
                    temperature_f,
                    feels_like_f,
                    wind_mph,
                    wind_direction,
                    gust_mph,
                    precipitation_probability_pct,
                    cloud_cover_pct,
                    humidity_pct,
                    visibility_miles
                FROM weather_period_snapshots
                WHERE game_id = ?
                  AND snapshot_id = ?
                  AND period = ?
                LIMIT 1
                """,
                (
                    game_id,
                    source_snapshot_id,
                    desired_period,
                ),
            ).fetchone()

        # Graceful fallback if an older capture does not contain the
        # requested period. Remain within the latest source snapshot.
        if (
            period_row is None
            and source_snapshot_id is not None
            and "weather_period_snapshots" in tables
        ):
            period_row = conn.execute(
                """
                SELECT
                    snapshot_id,
                    period,
                    condition,
                    temperature_f,
                    feels_like_f,
                    wind_mph,
                    wind_direction,
                    gust_mph,
                    precipitation_probability_pct,
                    cloud_cover_pct,
                    humidity_pct,
                    visibility_miles
                FROM weather_period_snapshots
                WHERE game_id = ?
                  AND snapshot_id = ?
                ORDER BY
                    CASE period
                        WHEN 'Kickoff' THEN 1
                        WHEN 'Q2' THEN 2
                        WHEN 'Q3' THEN 3
                        WHEN 'Q4' THEN 4
                        ELSE 99
                    END DESC
                LIMIT 1
                """,
                (
                    game_id,
                    source_snapshot_id,
                ),
            ).fetchone()

        if period_row is not None:
            weather.update(dict(period_row))

        # If period rows are unavailable, preserve useful kickoff fields.
        if weather.get("condition") is None:
            if desired_period == "Q4":
                weather["condition"] = weather.get("q4_condition")
            else:
                weather["condition"] = weather.get("kickoff_condition")

        fallback_map = {
            "temperature_f": "kickoff_temperature_f",
            "feels_like_f": "kickoff_feels_like_f",
            "wind_mph": "kickoff_wind_mph",
            "wind_direction": "kickoff_wind_direction",
            "gust_mph": "kickoff_gust_mph",
            "precipitation_probability_pct":
                "kickoff_precipitation_probability_pct",
            "humidity_pct": "kickoff_humidity_pct",
        }

        for destination, source in fallback_map.items():
            if weather.get(destination) is None:
                weather[destination] = weather.get(source)

        if (
            latest_feature is None
            and period_row is None
        ):
            return None

        return weather


def _weather_number(value, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if digits == 0:
        return str(int(round(number)))

    return f"{number:.{digits}f}"


def _weather_icon(condition: str) -> str:
    condition = str(condition or "").lower()

    if "thunder" in condition:
        return "⛈️"
    if "snow" in condition or "sleet" in condition:
        return "🌨️"
    if "rain" in condition or "shower" in condition:
        return "🌧️"
    if "fog" in condition or "mist" in condition:
        return "🌫️"
    if "cloud" in condition or "overcast" in condition:
        return "☁️"
    if "clear" in condition or "sun" in condition:
        return "☀️"

    return "🌤️"


def _render_live_weather(event: pd.Series) -> None:
    """Render compact public weather context for the selected game."""
    try:
        weather = _read_live_weather(event)
    except Exception:
        # Weather must never take down NFL Live.
        return

    if not weather:
        return

    roof = str(weather.get("roof") or "").strip()
    roof_lower = roof.lower()

    protected = any(
        token in roof_lower
        for token in (
            "dome",
            "closed",
            "indoor",
        )
    )

    if protected:
        st.markdown("#### 🏟️ Game Weather")
        st.caption(
            "Indoor / weather-protected game environment."
        )
        return

    condition = str(
        weather.get("condition") or "Weather available"
    ).strip()

    icon = _weather_icon(condition)

    temp = _weather_number(
        weather.get("temperature_f")
    )
    feels = _weather_number(
        weather.get("feels_like_f")
    )
    wind = _weather_number(
        weather.get("wind_mph")
    )
    gust = _weather_number(
        weather.get("gust_mph")
    )
    precip = _weather_number(
        weather.get("precipitation_probability_pct")
    )
    humidity = _weather_number(
        weather.get("humidity_pct")
    )

    direction = str(
        weather.get("wind_direction") or ""
    ).strip()

    period = str(
        weather.get("period") or ""
    ).strip()

    source = str(
        weather.get("source") or ""
    ).strip()

    captured = str(
        weather.get("source_captured_at") or ""
    ).strip()

    st.markdown(f"#### {icon} Game Weather — {condition}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Temperature",
        f"{temp}°F" if temp != "—" else "—",
        help=(
            f"Feels like {feels}°F"
            if feels != "—"
            else None
        ),
    )

    wind_value = (
        f"{direction} {wind} mph".strip()
        if wind != "—"
        else "—"
    )
    c2.metric("Wind", wind_value)

    c3.metric(
        "Gusts",
        f"{gust} mph" if gust != "—" else "—",
    )

    c4.metric(
        "Humidity",
        f"{humidity}%" if humidity != "—" else "—",
    )

    details = []

    if precip != "—":
        details.append(
            f"{precip}% precipitation probability"
        )

    if roof:
        if "out" in roof_lower or "open" in roof_lower:
            details.append("Open-air")
        else:
            details.append(roof.title())

    if period:
        details.append(f"{period} conditions")

    if details:
        st.caption(" • ".join(details))

    # Rain/showers are represented by the actual condition text from
    # the latest captured weather snapshot. Do not infer rain merely
    # from precipitation probability.
    condition_lower = condition.lower()

    if (
        period == "Q4"
        and (
            "rain" in condition_lower
            or "shower" in condition_lower
        )
    ):
        st.info(
            f"{condition} is indicated for the current Q4 "
            "weather period."
        )

    if source or captured:
        source_text = source or "Weather source"

        if captured:
            source_text += f" • captured {captured}"

        st.caption(source_text)


def _render_rosters(athletes: pd.DataFrame) -> None:
    st.markdown("#### Live Game Athletes")

    if athletes.empty:
        st.info("No athlete identity rows are available.")
        return

    display = athletes[
        [
            "display_name",
            "team",
            "position",
            "jersey",
            "identity_source",
        ]
    ].copy()

    display.columns = [
        "Player",
        "Team",
        "Position",
        "Jersey",
        "Identity Source",
    ]

    st.dataframe(
        display.reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


@st.fragment(run_every="45s")
def render_wfs_live_view(workspace_mode: str = "Public") -> None:
    """
    Render the WFS NFL LIVE dashboard.

    WFS_GAME_DAY_AUTO_REFRESH_V1

    Presentation-only 45-second refresh cadence aligned with the
    existing LIVE acquisition cycle. The fragment reruns only this
    LIVE presentation surface and does not modify LIVE persistence.
    """
    st.markdown("### 🔴 WFS NFL Live")

    st.caption(
        "Live scores, game updates, play-by-play and player information."
    )

    # WFS_LIVE_REFRESH_PRESERVE_SELECTION_V1
    # Streamlit button interaction already reruns the page.
    # Do not call st.rerun() here because that aborts execution before
    # the Game selectbox is rendered and can reset the selected event.
    st.button(
        "↻ Refresh Live",
        key="wfs_live_manual_refresh",
        width="stretch",
    )

    if workspace_mode == "Admin":
        st.caption(
            "Read-only app connection • LIVE ingestion remains "
            "owned by the systemd service."
        )

    try:
        with _live_connect() as conn:
            _verify_live_contract(conn)
            events = _read_events(conn)
            events = _apply_completed_game_authority(events)

            # WFS_LIVE_POST_AUTHORITY_ORDER_V1
            # Completed-game authority can change a stale LIVE event
            # from "in" to "post". Reapply the existing presentation
            # ordering after that authoritative state overlay so the
            # selector order reflects the final authoritative state.
            if not events.empty:
                state_rank = events["state"].map(
                    {"in": 0, "pre": 1, "post": 2}
                ).fillna(3)

                events = (
                    events.assign(_state_rank=state_rank)
                    .sort_values(
                        [
                            "_state_rank",
                            "week",
                            "updated_at_utc",
                            "event_id",
                        ],
                        ascending=[
                            True,
                            False,
                            False,
                            True,
                        ],
                        na_position="last",
                    )
                    .drop(columns=["_state_rank"])
                    .reset_index(drop=True)
                )

            if events.empty:
                st.info(
                    "No LIVE event has been persisted yet."
                )
                return

            # WFS_LIVE_EXACT_EVENT_SELECTOR_V2
            # Exact event_id is the widget value and sole render authority.
            event_ids = [
                str(x)
                for x in events["event_id"].astype(str).tolist()
            ]

            event_labels = {}
            for _, event_row in events.iterrows():
                event_id = str(event_row["event_id"])
                state = str(event_row.get("state") or "").lower()

                if state == "pre":
                    # Upcoming games have no score yet, so show kickoff
                    # time instead of "— @ —".
                    detail = str(event_row.get("detail") or "").strip()
                    weekday_part, _, time_part = detail.partition(" • ")
                    weekday_part = weekday_part.split(",")[0].strip()
                    kickoff = (
                        f"{weekday_part} {time_part}".strip()
                        if time_part
                        else detail
                    )
                    event_labels[event_id] = (
                        f"\U0001F552 {kickoff} • "
                        f"{event_row['away_team']} @ "
                        f"{event_row['home_team']}"
                        if kickoff
                        else (
                            f"{_event_state_label(event_row['state'])} • "
                            f"{event_row['away_team']} @ "
                            f"{event_row['home_team']}"
                        )
                    )
                else:
                    event_labels[event_id] = (
                        f"{_event_state_label(event_row['state'])} • "
                        f"{event_row['away_team']} "
                        f"{_score(event_row['away_score'])} @ "
                        f"{event_row['home_team']} "
                        f"{_score(event_row['home_score'])}"
                    )

            # WFS_LIVE_PLANNING_WEEK_SELECTION_V1
            #
            # Preserve a user's game selection during ordinary refreshes,
            # including PRE -> IN -> POST transitions for the same event.
            # When the authoritative planning week advances, however, a
            # historical event must not pin the selector to the prior week.
            #
            # Historical events remain available in event_ids for manual
            # selection; only stale automatic session-state retention is
            # cleared at the week boundary.
            week_values = pd.to_numeric(
                events["week"],
                errors="coerce",
            ).dropna()

            current_planning_week = (
                int(week_values.max())
                if not week_values.empty
                else None
            )

            old_value = st.session_state.get("wfs_live_event")
            old_week = None

            if old_value is not None:
                old_rows = events[
                    events["event_id"].astype(str).eq(str(old_value))
                ]

                if not old_rows.empty:
                    old_week_value = pd.to_numeric(
                        old_rows.iloc[0]["week"],
                        errors="coerce",
                    )

                    if pd.notna(old_week_value):
                        old_week = int(old_week_value)

            planning_week_key = "wfs_live_planning_week"
            previous_planning_week = st.session_state.get(
                planning_week_key
            )

            planning_advanced = False
            if (
                previous_planning_week is not None
                and current_planning_week is not None
            ):
                try:
                    planning_advanced = (
                        int(previous_planning_week)
                        < current_planning_week
                    )
                except (TypeError, ValueError):
                    planning_advanced = True

            bootstrap_stale = (
                previous_planning_week is None
                and current_planning_week is not None
                and old_week is not None
                and old_week < current_planning_week
            )

            old_event_missing = (
                old_value is not None
                and str(old_value) not in event_ids
            )

            stale_week_boundary_selection = (
                old_value is not None
                and current_planning_week is not None
                and old_week is not None
                and old_week < current_planning_week
                and (planning_advanced or bootstrap_stale)
            )

            if old_event_missing or stale_week_boundary_selection:
                st.session_state.pop("wfs_live_event", None)

            if current_planning_week is not None:
                st.session_state[planning_week_key] = (
                    current_planning_week
                )

            selected_event_id = st.selectbox(
                "Game",
                event_ids,
                format_func=lambda event_id: event_labels.get(
                    str(event_id),
                    str(event_id),
                ),
                key="wfs_live_event",
            )
            selected_event_id = str(selected_event_id)

            event_rows = events[
                events["event_id"].astype(str).eq(
                    selected_event_id
                )
            ]

            if event_rows.empty:
                st.error("Selected LIVE event disappeared.")
                return

            event = event_rows.iloc[0]

            # Schedule-backed PRE rows intentionally have no LIVE
            # audit/play/player/athlete records yet.
            if str(event.get("state") or "").lower() == "pre":
                audit = pd.DataFrame()
                plays = pd.DataFrame()
                refs = pd.DataFrame()
                athletes = pd.DataFrame()
            else:
                audit = _read_audit(
                    conn,
                    selected_event_id,
                )

                plays = _read_plays(
                    conn,
                    selected_event_id,
                )

                if str(event.get("state") or "").lower() == "post":
                    plays = _augment_completed_plays_from_espn(
                        plays,
                        selected_event_id,
                    )

                refs = _read_play_players(
                    conn,
                    selected_event_id,
                )

                athletes = _read_athletes(
                    conn,
                    [
                        str(event["away_team_id"]),
                        str(event["home_team_id"]),
                    ],
                )

        _render_event_header(event)

        # WFS_NFL_LIVE_WEATHER_PRESENTATION_V1
        # Existing weather intelligence, presentation-only.
        _render_live_weather(event)

        if workspace_mode == "Admin":
            health_tab, plays_tab, athletes_tab, audit_tab = st.tabs(
                [
                    "🩺 Health",
                    "🏈 Play-by-Play",
                    "👥 Athletes",
                    "📋 Audit",
                ]
            )

            with health_tab:
                _render_ingest_health(audit)

                h1, h2, h3 = st.columns(3)
                h1.metric("Stored Plays", len(plays))
                h2.metric(
                    "Resolved References",
                    len(refs),
                )
                h3.metric(
                    "Known Athletes",
                    len(athletes),
                )

                st.caption(
                    "Event DB updated: "
                    + str(event.get("updated_at_utc") or "unknown")
                )

            with plays_tab:
                _render_play_feed(
                    plays,
                    refs,
                )

            with athletes_tab:
                _render_rosters(athletes)

            with audit_tab:
                if audit.empty:
                    st.info("No audit rows are available.")
                else:
                    st.dataframe(
                        audit.reset_index(drop=True),
                        width="stretch",
                        hide_index=True,
                    )

        else:
            plays_tab, athletes_tab = st.tabs(
                [
                    "🏈 Play-by-Play",
                    "👥 Athletes",
                ]
            )

            with plays_tab:
                _render_play_feed(
                    plays,
                    refs,
                )

            with athletes_tab:
                _render_rosters(athletes)

        if workspace_mode == "Admin":
            st.caption(
                "WFS NFL LIVE app integration is presentation-only. "
                "The Streamlit process has no LIVE database write path."
            )

    except sqlite3.OperationalError as exc:
        st.error(
            "WFS NFL LIVE database could not be opened read-only."
        )
        st.caption(str(exc))

    except Exception as exc:
        st.error("WFS NFL LIVE contract check failed.")
        st.caption(str(exc))
