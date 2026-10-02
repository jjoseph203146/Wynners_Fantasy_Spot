#!/usr/bin/env python3
"""
app.py

Wynners Fantasy Spot — NFL FanDuel GPP UI + Fantasy Intelligence.

This is a presentation/control layer only.
It does NOT modify the frozen production projection or optimizer model.

Generation is delegated to:
    fanduel_nfl_ui_solver.py

The UI adapter preserves the production objective and adds only explicit
user-requested lock/exclude constraints.
"""

from __future__ import annotations

import base64
import hashlib
import logging

import csv
import io
import re
import sqlite3
import fcntl
import json
import urllib.parse
import urllib.request
import urllib.error
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken

from config import DATABASE_PATH, CSV_DIR
from forecast_publication_selector import select_forecast_path
from fanduel_nfl_late_swap_solver_stage2_v1 import LateSwapSettings, optimize_late_swap


APP_DIR = Path(__file__).resolve().parent
UI_SOLVER = APP_DIR / "fanduel_nfl_ui_solver.py"
SHOWDOWN_SOLVER = APP_DIR / "fanduel_nfl_showdown_solver_v1.py"
SHOWDOWN_POOL = APP_DIR / "data" / "fanduel" / "single_game" / "derived" / "single_game_projection_pool.parquet"
SOLVER_TABLE = "fanduel_solver_ready_pool"
CLASSIC_LINEUP_FRESHNESS_PATH = APP_DIR / "data" / "parquet" / "fanduel_solver_ready_pool.parquet"
PRODUCTION_PROJECTION_PATH = APP_DIR / "data" / "parquet" / "nfl_production_projection.parquet"
COLD_START_PROJECTION_PATH = APP_DIR / "data" / "parquet" / "nfl_cold_start_projection.parquet"
WFS_ACCOUNTS_DB_PATH = APP_DIR / "data" / "wfs_accounts.db"
PORTFOLIO_CACHE_DIR = APP_DIR / "data" / "cache" / "fanduel_ui_portfolios"
PORTFOLIO_CACHE_SIZE = 5
# Preserve the existing append-only +5 cache process while allowing
# a deep inventory of unique lineups. Exhaustion always fails closed.
PORTFOLIO_CACHE_MAX_SIZE = 1000
PORTFOLIO_CACHE_INCREMENT = 5
PORTFOLIO_CACHE_LOCK = PORTFOLIO_CACHE_DIR / ".solver.lock"
ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

MATCHUP_OUTLOOK_PATH = APP_DIR / "data" / "research" / "offense_defense_matchup_outlook_week3_shadow_v1.parquet"
RELATIONSHIP_OUTLOOK_PATH = APP_DIR / "data" / "research" / "ceiling_intelligence_week3_relationship_shadow_v1.parquet"


def _wfs_matchup_public_status(value: object) -> str:
    return {
        "SUPPORTED_CONTEXT": "Favorable evidence",
        "SUPPORTED": "Favorable evidence",
        "MIXED_CONTEXT": "Mixed evidence",
        "MIXED": "Mixed evidence",
        "LIMITED_EVIDENCE": "Limited sample",
        "NOT_SUPPORTED_CONTEXT": "No clear matchup support",
        "NOT_SUPPORTED": "No clear matchup support",
    }.get(str(value or "").strip().upper(), "Limited sample")


def _wfs_relationship_public_authorized(row: pd.Series, selected_ids: set[str]) -> bool:
    """Fail-closed relationship filter for the public selected-player population."""
    relationship_type = str(row.get("relationship_type") or "").strip()
    source_id = str(row.get("source_player_id") or "").strip()
    related_value = row.get("related_player_id")
    related_id = "" if pd.isna(related_value) else str(related_value).strip()
    if not source_id or source_id not in selected_ids:
        return False
    if relationship_type in {"QB_PASS_CATCHER", "OPPOSING_BRING_BACK"}:
        return bool(related_id and related_id in selected_ids)
    if relationship_type == "RB_TEAM_SCORING":
        return not related_id or related_id in selected_ids
    if relationship_type == "QB_MULTI_PASS_CATCHER_CONTEXT":
        return True
    if relationship_type == "GAME_ENVIRONMENT":
        return not related_id or related_id in selected_ids
    return False


def _wfs_matchup_outlook_for_selected(game_id: str, selected_players: list[dict]) -> dict | None:
    """Load descriptive context only for exact Stat Outlook player IDs."""
    try:
        matchup = pd.read_parquet(MATCHUP_OUTLOOK_PATH)
        relationships = pd.read_parquet(RELATIONSHIP_OUTLOOK_PATH)
        # Stat Outlook is the player-publication authority for Matchup Outlook.
        # Relationship evidence governs Game connections only; it must not
        # remove an otherwise valid Stat Outlook-selected player.
        selected = {
            str(row.get("player_id") or "").strip(): row
            for row in selected_players
            if str(row.get("player_id") or "").strip()
        }
        if not selected:
            return None
        player_rows = matchup[
            matchup["row_type"].eq("PLAYER")
            & matchup["game_id"].astype(str).eq(str(game_id))
            & matchup["player_id"].astype(str).isin(selected)
            & matchup["active_flag"].eq(1)
            & matchup["injury_gate"].eq("ALLOW")
            & matchup["current_reconciliation_role"].ne("CONTINGENCY_QB")
        ].copy()
        if set(player_rows["player_id"].astype(str)) != set(selected) or player_rows["player_id"].duplicated().any():
            return None
        team_rows = matchup[
            matchup["row_type"].eq("POSITION")
            & matchup["game_id"].astype(str).eq(str(game_id))
        ].drop_duplicates("team")
        if team_rows.empty:
            return None
        rel = relationships[
            relationships["game_id"].astype(str).eq(str(game_id))
        ].copy()
        rel = rel[
            rel.apply(
                _wfs_relationship_public_authorized,
                axis=1,
                selected_ids=set(selected),
            )
        ].copy()
        return {"players": player_rows, "teams": team_rows, "relationships": rel, "selected": selected}
    except Exception:
        return None


def _wfs_render_matchup_outlook(game_id: str, selected_players: list[dict]) -> None:
    context = _wfs_matchup_outlook_for_selected(game_id, selected_players)
    if not context:
        return

    def compact_status(channels: list[tuple[str, object]]) -> str:
        grouped = {}
        for label, value in channels:
            status = _wfs_matchup_public_status(value)
            grouped.setdefault(status, []).append(label)
        if len(grouped) == 1:
            return next(iter(grouped))
        return " · ".join(f"{'/'.join(labels)}: {status}" for status, labels in grouped.items())

    st.markdown("### Matchup Outlook")
    st.caption("Early-season matchup data is limited.")
    for _, team in context["teams"].sort_values("team").iterrows():
        team_name = str(team["team"])
        st.markdown(f"#### {team_name} vs {team['opponent_team']}")
        st.markdown(
            "Game: " + compact_status([
                ("Passing", team.get("pass_environment_status")),
                ("Rushing", team.get("rush_environment_status")),
                ("Scoring", team.get("scoring_environment_status")),
                ("Play volume", team.get("play_volume_status")),
            ])
        )
        team_players = context["players"][context["players"]["team"].eq(team_name)]
        for _, player in team_players.iterrows():
            selected = context["selected"].get(str(player["player_id"]), {})
            label = f"{selected.get('player_name') or player.get('player_name')} ({selected.get('position') or player.get('position')})"
            st.markdown(f"**{label}**")
            if selected.get("forecast_line"):
                # Reformat the authoritative display text without recalculating stats or TDs.
                forecast = str(selected["forecast_line"]).removeprefix(f"{label}: ")
                for part in forecast.split(" | "):
                    st.caption(part.replace(", ", " · "))
            st.markdown(
                "Matchup: " + compact_status([
                    ("Role", player.get("role_context_status")),
                    ("Position", player.get("position_matchup_status")),
                ])
            )
    # Summarize only the relationships admitted by the validated loader above.
    rels = context["relationships"]
    relationship_lines = []
    passing = rels[rels["relationship_type"].eq("QB_PASS_CATCHER")]
    for source_id, group in passing.groupby("source_player_id", sort=True):
        source = context["selected"][str(source_id)]["player_name"]
        receivers = [
            context["selected"][str(player_id)]["player_name"]
            for player_id in group["related_player_id"].drop_duplicates()
        ]
        relationship_lines.append(f"{source} → {', '.join(receivers)}")
    rushing = rels[rels["relationship_type"].eq("RB_TEAM_SCORING")]
    for _, relationship in rushing.drop_duplicates("source_player_id").iterrows():
        player = context["selected"][str(relationship["source_player_id"])]
        relationship_lines.append(
            f"{player['player_name']}: {player['team']} rushing/scoring context."
        )
    # Generic opposing, multi-receiver and game-environment rows add no public detail.
    if relationship_lines:
        st.markdown("### Game connections")
        for text in dict.fromkeys(relationship_lines):
            st.markdown(f"- {text}")


def _wfs_render_public_data_freshness(
    source_path: Path,
    *,
    product_label: str = "Player data",
    authority_mode: str = "classic",
) -> None:
    """
    WFS_PUBLIC_SCHEDULE_FRESHNESS_V2

    Public-facing lineup-data currentness indicator.

    Schedule authority determines whether the artifact belongs to the
    current planning week. File modification time is presentation
    metadata only and never establishes currentness.
    """
    try:
        if not source_path.exists():
            st.warning(
                "Current slate data is not available yet."
                if authority_mode == "classic" else
                f"⚠️ {product_label} freshness could not be verified. "
                "Please check again before generating lineups."
            )
            return

        from wfs_schedule_context import (
            resolve_schedule_week_context,
        )

        schedule_context = resolve_schedule_week_context()

        season = schedule_context.season
        planning_week = schedule_context.planning_week

        if season is None or planning_week is None:
            raise RuntimeError(
                "SCHEDULE_CONTEXT_INCOMPLETE"
            )

        authority_mode = str(
            authority_mode or ""
        ).strip().lower()

        artifact = pd.read_parquet(source_path)

        if artifact.empty:
            raise RuntimeError(
                "FRESHNESS_ARTIFACT_EMPTY"
            )

        if authority_mode == "classic":
            required = {"season", "week"}

            if not required.issubset(artifact.columns):
                raise RuntimeError(
                    "CLASSIC_FRESHNESS_IDENTITY_MISSING"
                )

            artifact_season = pd.to_numeric(
                artifact["season"],
                errors="coerce",
            )

            artifact_week = pd.to_numeric(
                artifact["week"],
                errors="coerce",
            )

            current_mask = (
                artifact_season.eq(int(season))
                & artifact_week.eq(int(planning_week))
            )

            if not bool(current_mask.any()):
                st.warning(
                    "Current slate data is not available yet."
                )
                return

            represented = (
                pd.DataFrame(
                    {
                        "season": artifact_season,
                        "week": artifact_week,
                    }
                )
                .dropna()
                .drop_duplicates()
            )

            if len(represented) != 1:
                raise RuntimeError(
                    "CLASSIC_FRESHNESS_MIXED_WEEK_AUTHORITY"
                )

            only = represented.iloc[0]

            if (
                int(only["season"]) != int(season)
                or int(only["week"]) != int(planning_week)
            ):
                raise RuntimeError(
                    "CLASSIC_FRESHNESS_WRONG_WEEK"
                )

        elif authority_mode == "single_game":
            if "public_slate_name" not in artifact.columns:
                raise RuntimeError(
                    "SINGLE_GAME_FRESHNESS_IDENTITY_MISSING"
                )

            slate_names = sorted(
                {
                    str(value).strip()
                    for value in artifact[
                        "public_slate_name"
                    ].dropna()
                    if str(value).strip()
                }
            )

            if not slate_names:
                raise RuntimeError(
                    "SINGLE_GAME_FRESHNESS_SLATE_EMPTY"
                )

            import sqlite3

            with sqlite3.connect(
                APP_DIR / "data" / "nfl.db"
            ) as conn:
                schedule = pd.read_sql_query(
                    """
                    SELECT
                        game_id,
                        away_team,
                        home_team
                    FROM games
                    WHERE season = ?
                      AND game_type = 'REG'
                      AND week = ?
                    """,
                    conn,
                    params=(
                        int(season),
                        int(planning_week),
                    ),
                )

            if schedule.empty:
                raise RuntimeError(
                    "SINGLE_GAME_CURRENT_SCHEDULE_EMPTY"
                )

            schedule = schedule.copy()
            schedule["_away"] = schedule[
                "away_team"
            ].map(normalize_team)
            schedule["_home"] = schedule[
                "home_team"
            ].map(normalize_team)

            for slate_name in slate_names:
                if " @ " not in slate_name:
                    raise RuntimeError(
                        "SINGLE_GAME_SLATE_FORMAT_INVALID"
                    )

                away_raw, home_raw = slate_name.split(
                    " @ ",
                    1,
                )

                away = normalize_team(away_raw)
                home = normalize_team(home_raw)

                if not away or not home or away == home:
                    raise RuntimeError(
                        "SINGLE_GAME_SLATE_IDENTITY_INVALID"
                    )

                matches = schedule[
                    schedule["_away"].eq(away)
                    & schedule["_home"].eq(home)
                ]

                if len(matches) != 1:
                    st.warning(
                        f"⚠️ {product_label} is not available for the "
                        "current slate yet. Please check again before "
                        "generating lineups."
                    )
                    return

        else:
            raise RuntimeError(
                "UNKNOWN_FRESHNESS_AUTHORITY_MODE"
            )

        et = ZoneInfo("America/New_York")

        updated_et = datetime.fromtimestamp(
            source_path.stat().st_mtime,
            tz=et,
        )

        updated_text = updated_et.strftime(
            "%b %d, %Y • %I:%M %p ET"
        )

        st.info(
            (
                "🔄 **Current slate**\n\n"
                "Player data is up to date.\n\n"
                if authority_mode == "classic" else
                "🔄 **Current slate player data**\n\n"
                "Schedule identity has been verified for the current slate.\n\n"
            )
            + f"**Last updated: {updated_text}**"
        )

    except Exception:
        # Public UI fails safely without exposing paths, backend details,
        # exception text, or implementation diagnostics.
        st.warning(
            "Current slate data is not available yet."
            if authority_mode == "classic" else
            f"⚠️ {product_label} freshness could not be verified. "
            "Please check again before generating lineups."
        )


def normalize_slate(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def safe_slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return out or "slate"


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_team(value) -> str:
    if pd.isna(value):
        return ""

    team = re.sub(
        r"[^A-Z]",
        "",
        str(value).upper(),
    )

    # Explicit external-provider -> WFS team aliases.
    #
    # No fuzzy team matching is permitted.
    team_aliases = {
        "JAC": "JAX",
    }

    return team_aliases.get(
        team,
        team,
    )


FANDUEL_NAME_ALIASES = {
    normalize_name("Kyle Pitts Sr."): normalize_name("Kyle Pitts"),
}


def normalize_fanduel_name(value) -> str:
    """
    Deterministic FanDuel -> WFS player-name normalization.

    No fuzzy matching is permitted. Any known naming exception must be
    explicitly represented in FANDUEL_NAME_ALIASES.
    """
    key = normalize_name(value)
    return FANDUEL_NAME_ALIASES.get(key, key)


def ui_player_key(row) -> str:
    return "|||".join([
        normalize_name(row["player_solver"]),
        normalize_team(row["team_solver"]),
        str(row["solver_position"]).strip().upper(),
    ])


def solver_table_freshness_token() -> tuple[int, int]:
    """
    Return deterministic publication freshness for the Classic solver pool.

    The solver-ready parquet is the publication freshness artifact while
    SQLite remains the authoritative table consumed by the UI. Including
    this token in the cached loader arguments invalidates Streamlit's cached
    DataFrame whenever a new Classic solver pool is published.
    """
    if not CLASSIC_LINEUP_FRESHNESS_PATH.is_file():
        raise RuntimeError(
            "Missing Classic lineup freshness artifact: "
            f"{CLASSIC_LINEUP_FRESHNESS_PATH}"
        )

    stat = CLASSIC_LINEUP_FRESHNESS_PATH.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


@st.cache_data(show_spinner=False)
def load_solver_table(
    freshness_token: tuple[int, int],
) -> pd.DataFrame:
    del freshness_token

    with sqlite3.connect(DATABASE_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (SOLVER_TABLE,),
        ).fetchone()
        if not exists:
            raise RuntimeError(f"Missing SQLite table: {SOLVER_TABLE}")
        return pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)


def current_classic_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Current Classic authority; deliberately evaluated outside Streamlit cache."""
    from wfs_schedule_context import resolve_schedule_week_context

    context = resolve_schedule_week_context()
    required = {"season", "week", "slate_name_solver"}
    if not required.issubset(df.columns):
        raise RuntimeError("Classic publication identity is unavailable.")
    season = pd.to_numeric(df["season"], errors="coerce")
    week = pd.to_numeric(df["week"], errors="coerce")
    return df.loc[
        season.eq(int(context.season)) & week.eq(int(context.planning_week))
    ].copy()


def _wfs_classic_failure_category(attempts, has_controls: bool) -> str:
    """Classify explicit diagnostics only; ambiguous failures stay UNKNOWN."""
    categories = []
    for attempt in attempts:
        if attempt.get("accepted"):
            continue
        detail = (attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "")
        if any(marker in detail for marker in (
            "Stage24AttachmentError:", "CLASSIC_PUBLICATION_",
            "RuntimeError: FAIL_CLOSED:", "RuntimeError: Final injury gate ",
            "RuntimeError: Missing SQLite table:", "RuntimeError: Missing offensive parquet:",
            "RuntimeError: Offensive parquet missing fields:",
            "RuntimeError: Offensive GPP slate not found:",
            "RuntimeError: Duplicate eligible player/team/salary rows detected:",
            "RuntimeError: Duplicate exact offensive name+team+position keys detected.",
            "RuntimeError: Duplicate UI player keys detected:",
            "RuntimeError: Eligible pool contains invalid ",
            "RuntimeError: Incomplete exact offensive GPP attachment:",
            "RuntimeError: Validated GPP field contains missing/nonfinite values.",
            "RuntimeError: RB gpp_score_20 coverage is incomplete.",
            "RuntimeError: WR gpp_score_15/25 coverage is incomplete.",
            "RuntimeError: TE gpp_score_15/20 coverage is incomplete.",
            "RuntimeError: GPP objective signal creation is incomplete.",
            "RuntimeError: Could not resolve opponent for eligible QB rows:",
            "RuntimeError: No eligible candidates for roster slot ",
            "RuntimeError: Insufficient ", "RuntimeError: Slate not found:",
        )):
            categories.append("AUTHORITY")
        elif attempt.get("validation_failure") == "AUDIT_FAILED" or "OUTPUT_VALIDATION_ERROR:" in detail:
            categories.append("UNKNOWN")
        elif has_controls and (
            any(marker in detail for marker in (
                "ValueError: A player cannot be both locked and excluded:",
                "RuntimeError: Locked player(s) not found in eligible slate pool:",
                "RuntimeError: Excluded player(s) not found in eligible slate pool:",
            ))
            or re.search(r"RuntimeError: Resume lineup \d+ (?:is missing one or more current locks|contains a currently excluded player)\.", detail)
        ):
            categories.append("PLAYER_CONTROL")
        elif re.search(
            r"Stopped at lineup \d+: constraints/exposure caps made the remaining portfolio infeasible\.", detail
        ) or "RuntimeError: No feasible lineups generated." in detail:
            categories.append("INFEASIBLE")
        else:
            categories.append("UNKNOWN")
    # Never let a strategy attempt hide an authority or unrecognized failure.
    for category in ("AUTHORITY", "UNKNOWN", "PLAYER_CONTROL", "INFEASIBLE"):
        if category in categories:
            return category
    return "UNKNOWN"


def _wfs_classic_failure_message(category: str) -> str:
    return {
        "AUTHORITY": (
            "We couldn't generate lineups because the latest lineup data is still "
            "being prepared. Please try again shortly."
        ),
        "PLAYER_CONTROL": (
            "We couldn't use your locked or excluded player selections. "
            "Try changing a lock or exclusion and generate again."
        ),
        "INFEASIBLE": (
            "We couldn't complete the requested FanDuel lineups for this slate. "
            "Try another slate or try again shortly."
        ),
    }.get(category, "We couldn't generate lineups right now. Please try again shortly.")


def _wfs_report_classic_exception(exc, workspace_mode, category=None, has_controls=False):
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    category = category or _wfs_classic_failure_category([{"stderr": detail}], has_controls)
    st.session_state["last_classic_generation_exception"] = detail
    st.session_state["last_classic_failure_category"] = category
    logging.getLogger(__name__).error(
        "Classic generation failure (%s)", category,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    if workspace_mode == "Admin":
        st.error("Classic lineup generation failed.")
        st.code(detail)
    else:
        st.error(_wfs_classic_failure_message(category))


@contextmanager
def _wfs_classic_generation_errors(workspace_mode, has_controls):
    """Present unhandled Public failures without exposing backend details."""
    try:
        yield
    except Exception as exc:
        if workspace_mode == "Admin":
            raise
        _wfs_report_classic_exception(exc, workspace_mode, has_controls=has_controls)
        st.stop()


def require_current_classic_action(selected_slate: str) -> None:
    """Re-read publication and schedule immediately before a Classic action."""
    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro", uri=True
        ) as conn:
            publication = pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)
        current = current_classic_publication(publication)
        selected = publication.loc[
            publication["slate_name_solver"].map(normalize_slate).eq(
                normalize_slate(selected_slate)
            )
        ]
        if selected.empty or len(current.loc[selected.index.intersection(current.index)]) != len(selected):
            raise RuntimeError("Selected Classic slate is no longer current.")
    except Exception as exc:
        _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
        st.stop()


def available_slates(df: pd.DataFrame) -> list[str]:
    if "slate_name_solver" not in df.columns:
        return []
    names = (
        df["slate_name_solver"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted([x for x in names.unique() if x], key=str.lower)


def slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    target = normalize_slate(slate_name)
    mask = (
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    )
    out = df.loc[mask].copy()
    eligible = (
        pd.to_numeric(out["solver_eligible"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    out = out.loc[eligible].copy()
    out["salary_solver"] = pd.to_numeric(out["salary_solver"], errors="coerce")
    out["projection_solver"] = pd.to_numeric(
        out["projection_solver"], errors="coerce"
    )
    out = out.dropna(subset=["salary_solver", "projection_solver"])
    out["salary_solver"] = out["salary_solver"].astype(int)
    out["_ui_key"] = out.apply(ui_player_key, axis=1)
    out["_label"] = out.apply(
        lambda r: (
            f'{r["player_solver"]} — {r["team_solver"]} '
            f'{r["solver_position"]} — ${int(r["salary_solver"]):,} '
            f'— {float(r["projection_solver"]):.2f}'
        ),
        axis=1,
    )
    return out.sort_values(
        ["solver_position", "projection_solver", "salary_solver", "player_solver"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )


def late_swap_slate_pool(df: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    # Full selected-slate view for Late Swap only.
    # Retains solver-blocked rows so already-started roster occupants can be preserved.
    target = normalize_slate(slate_name)
    mask = (
        df["slate_name_solver"].map(normalize_slate).eq(target)
        | df["slate_slug_solver"].map(normalize_slate).eq(target)
    )

    out = df.loc[mask].copy()

    out["salary_solver"] = pd.to_numeric(
        out["salary_solver"],
        errors="coerce",
    )
    out["projection_solver"] = pd.to_numeric(
        out["projection_solver"],
        errors="coerce",
    )

    out = out.dropna(subset=["salary_solver"]).copy()
    out = out.loc[out["salary_solver"].gt(0)].copy()
    out["salary_solver"] = out["salary_solver"].astype(int)

    out["_ui_key"] = out.apply(
        ui_player_key,
        axis=1,
    )

    def _late_swap_label(row):
        projection = row.get("projection_solver")
        projection_text = (
            f"{float(projection):.2f}"
            if pd.notna(projection)
            else "locked/no current projection"
        )

        position_value = row.get("solver_position")
        position = (
            str(position_value).strip()
            if pd.notna(position_value)
            else ""
        ) or "—"

        return (
            f'{row["player_solver"]} — {row["team_solver"]} '
            f'{position} — ${int(row["salary_solver"]):,} '
            f'— {projection_text}'
        )

    out["_label"] = out.apply(
        _late_swap_label,
        axis=1,
    )

    return out.sort_values(
        [
            "solver_position",
            "projection_solver",
            "salary_solver",
            "player_solver",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
        kind="mergesort",
        na_position="last",
    )


LATE_SWAP_SLOT_POSITIONS = {
    "QB": {"QB"},
    "RB1": {"RB"},
    "RB2": {"RB"},
    "WR1": {"WR"},
    "WR2": {"WR"},
    "WR3": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "DST": {"DST"},
}


def _late_swap_schedule_columns(schedule_df: pd.DataFrame) -> dict:
    """Discover schedule fields without inventing kickoff data."""
    if schedule_df is None or schedule_df.empty:
        return {}
    return {
        "home": _dc_col(schedule_df, ["home_team", "home"]),
        "away": _dc_col(schedule_df, ["away_team", "away"]),
        "season": _dc_col(schedule_df, ["season", "year"]),
        "date": _dc_col(
            schedule_df,
            ["start_time", "kickoff", "game_datetime", "gameday", "game_date", "date"],
        ),
        "time": _dc_col(schedule_df, ["gametime", "game_time", "time"]),
        "home_score": _dc_col(schedule_df, ["home_score", "homescore", "home_points"]),
        "away_score": _dc_col(schedule_df, ["away_score", "awayscore", "away_points"]),
    }


def _late_swap_parse_game_teams(value) -> set[str]:
    """Extract the two team abbreviations from the solver game label when possible."""
    raw = str(value or "").upper()
    tokens = re.findall(r"\b[A-Z]{2,3}\b", raw)
    return {normalize_team(x) for x in tokens if normalize_team(x)}


def _late_swap_kickoff_et(row: pd.Series, cols: dict):
    """
    Return an aware Eastern kickoff timestamp only when schedule data supports it.
    A date with no clock time is intentionally UNKNOWN (fail closed).
    """
    date_col = cols.get("date")
    time_col = cols.get("time")
    if not date_col or pd.isna(row.get(date_col)):
        return None

    date_raw = str(row.get(date_col)).strip()
    time_raw = ""
    if time_col and not pd.isna(row.get(time_col)):
        time_raw = str(row.get(time_col)).strip()

    has_clock_in_date = bool(re.search(r"\d{1,2}:\d{2}", date_raw)) or "T" in date_raw
    if time_raw:
        candidate = f"{date_raw} {time_raw}"
    elif has_clock_in_date:
        candidate = date_raw
    else:
        return None

    ts = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(ts):
        return None

    try:
        py_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if py_dt.tzinfo is None:
            return py_dt.replace(tzinfo=ZoneInfo("America/New_York"))
        return py_dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return None


def _late_swap_player_status(player_row: pd.Series, schedule_df: pd.DataFrame, now_et: datetime) -> dict:
    """Read-only lock audit. UNKNOWN is returned whenever evidence is insufficient."""
    if schedule_df is None or schedule_df.empty:
        return {"status": "UNKNOWN", "kickoff": "—", "reason": "Schedule unavailable"}

    cols = _late_swap_schedule_columns(schedule_df)
    if not cols.get("home") or not cols.get("away"):
        return {"status": "UNKNOWN", "kickoff": "—", "reason": "Schedule teams unavailable"}

    team = normalize_team(player_row.get("team_solver"))
    game_teams = _late_swap_parse_game_teams(player_row.get("game_solver"))

    candidates = schedule_df.copy()
    home_norm = candidates[cols["home"]].map(normalize_team)
    away_norm = candidates[cols["away"]].map(normalize_team)
    mask = home_norm.eq(team) | away_norm.eq(team)

    # When the solver game label exposes both teams, require that exact matchup.
    if len(game_teams) >= 2:
        mask &= home_norm.isin(game_teams) & away_norm.isin(game_teams)
    candidates = candidates.loc[mask].copy()

    season_col = cols.get("season")
    if season_col and len(candidates):
        season_num = pd.to_numeric(candidates[season_col], errors="coerce")
        current = candidates.loc[season_num.eq(now_et.year)]
        if len(current):
            candidates = current

    if candidates.empty:
        return {"status": "UNKNOWN", "kickoff": "—", "reason": "Exact game not found"}

    evaluated = []
    for _, sched_row in candidates.iterrows():
        kickoff = _late_swap_kickoff_et(sched_row, cols)
        home_score = sched_row.get(cols.get("home_score")) if cols.get("home_score") else None
        away_score = sched_row.get(cols.get("away_score")) if cols.get("away_score") else None
        final_by_score = (
            home_score is not None and away_score is not None
            and pd.notna(home_score) and pd.notna(away_score)
        )
        evaluated.append((sched_row, kickoff, final_by_score))

    # Prefer a dated game closest to now; this avoids selecting an old same-team matchup.
    dated = [x for x in evaluated if x[1] is not None]
    if dated:
        sched_row, kickoff, final_by_score = min(
            dated,
            key=lambda x: abs((x[1] - now_et).total_seconds()),
        )
    elif len(evaluated) == 1:
        sched_row, kickoff, final_by_score = evaluated[0]
    else:
        return {"status": "UNKNOWN", "kickoff": "—", "reason": "Kickoff time unavailable"}

    if final_by_score:
        status = "LOCKED"
        reason = "Game has scoring data"
    elif kickoff is None:
        status = "UNKNOWN"
        reason = "Kickoff time unavailable"
    elif kickoff <= now_et:
        status = "LOCKED"
        reason = "Game has started"
    else:
        status = "SWAPPABLE"
        reason = "Game has not started"

    kickoff_text = kickoff.strftime("%a %m/%d %I:%M %p ET") if kickoff else "—"
    return {"status": status, "kickoff": kickoff_text, "reason": reason}


def _late_swap_options_for_slot(pool: pd.DataFrame, slot: str) -> list[str]:
    allowed = LATE_SWAP_SLOT_POSITIONS[slot]
    rows = pool.loc[pool["solver_position"].astype(str).str.upper().isin(allowed)]
    return ["— Select player —"] + rows["_label"].astype(str).tolist()


def render_late_swap_stage1(pool: pd.DataFrame, selected_slate: str) -> None:
    """
    Reconstruct, audit, and optionally optimize one existing FanDuel lineup.

    Stage 1:
        - exact current-lineup reconstruction
        - FanDuel legality audit
        - LOCKED / SWAPPABLE / UNKNOWN game-state audit

    Stage 2:
        - deterministic late-swap optimization
        - exact-slot preservation for LOCKED and UNKNOWN slots
        - only SWAPPABLE players may be newly introduced
        - production projection/GPP frontier semantics

    The frozen production solver and frozen Stage 2 solver are not modified.
    """
    st.markdown("#### Late Swap")

    st.caption(
        "Recreate the lineup exactly as it is currently entered on FanDuel. "
        "WFS audits every roster slot before any late-swap optimization occurs."
    )

    # -----------------------------------------------------------------
    # Optional exact FanDuel contest boundary
    # -----------------------------------------------------------------

    st.markdown("##### Contest Boundary")

    slate_slug = normalize_slate(selected_slate)

    upload_key = f"late_swap_template_{slate_slug}"
    bytes_key = f"late_swap_template_bytes_{slate_slug}"
    name_key = f"late_swap_template_name_{slate_slug}"

    late_swap_template = st.file_uploader(
        "FanDuel entries template for Late Swap (recommended)",
        type=["csv"],
        key=upload_key,
        help=(
            "When supplied, Stage 2 may select only players that match the "
            "exact FanDuel contest player pool. Exact matching only; no fuzzy "
            "player resolution."
        ),
    )

    # Persist the exact uploaded bytes across Streamlit reruns.
    # Late Swap must not silently lose its contest boundary when the user
    # changes lineup selections or clicks Optimize.
    if late_swap_template is not None:
        uploaded_bytes = late_swap_template.getvalue()

        if uploaded_bytes:
            st.session_state[bytes_key] = uploaded_bytes
            st.session_state[name_key] = str(late_swap_template.name)

    persisted_template_bytes = st.session_state.get(bytes_key)
    persisted_template_name = st.session_state.get(name_key)

    late_swap_pool = pool.copy()
    late_swap_contest_verified = False

    if persisted_template_bytes:
        try:
            contest_pool, blocked_pool, contest_meta = (
                intersect_with_fanduel_contest(
                    pool,
                    persisted_template_bytes,
                )
            )

            if contest_pool.empty:
                st.error(
                    "Late Swap blocked: the uploaded FanDuel template has "
                    "zero exact overlap with this solver slate."
                )
                return

            late_swap_pool = contest_pool.copy()
            late_swap_contest_verified = True

            st.success(
                f'Contest verified — '
                f'{contest_meta["solver_players_in_contest"]} solver-ready '
                f'players matched exactly; '
                f'{contest_meta["solver_players_blocked"]} non-contest '
                f'players blocked.'
            )

            if persisted_template_name:
                st.caption(
                    f"Contest template locked for this session: "
                    f"{persisted_template_name}"
                )

            if st.button(
                "Clear Late Swap contest template",
                key=f"late_swap_clear_template_{slate_slug}",
            ):
                st.session_state.pop(bytes_key, None)
                st.session_state.pop(name_key, None)
                st.rerun()

        except Exception as exc:
            # Fail closed. Never fall back to the unrestricted slate after a
            # template has been supplied but cannot be validated.
            st.error(
                "Late Swap contest validation could not be completed. "
                "Please verify the FanDuel contest template and try again."
            )
            return

    else:
        st.warning(
            "No Late Swap FanDuel template is currently locked. "
            "Stage 2 can run against the selected solver slate, but exact "
            "FanDuel contest membership will not be verified."
        )

    if late_swap_pool.empty:
        st.error(
            "Late Swap blocked: no eligible players are available."
        )
        return

    # -----------------------------------------------------------------
    # Stage 1 — reconstruct current lineup
    # -----------------------------------------------------------------

    st.markdown("##### 1. Current FanDuel Lineup")

    st.caption(
        "Enter all nine players exactly as they currently appear in your "
        "FanDuel entry."
    )

    label_to_row = {
        str(row["_label"]): row
        for _, row in late_swap_pool.iterrows()
    }

    # -------------------------------------------------------------
    # Auto-load current FanDuel lineup from the verified template.
    #
    # FanDuel IDs are first resolved through FanDuel's own embedded
    # player table, then exact-matched to the already contest-filtered
    # WFS pool. No fuzzy matching is permitted.
    # -------------------------------------------------------------

    auto_slot_labels = {}
    auto_load_error = None

    if late_swap_contest_verified and persisted_template_bytes:
        try:
            parsed_template = parse_fanduel_template(
                persisted_template_bytes
            )

            entry_lineups = parsed_template.get(
                "entry_lineups",
                [],
            )

            fd_players_by_id = parsed_template.get(
                "player_rows_by_id",
                {},
            )

            if len(entry_lineups) == 1:
                fd_entry = entry_lineups[0]
                fd_slots = fd_entry.get("slots", {})

                for slot in ROSTER_SLOTS:
                    fd_id = str(
                        fd_slots.get(slot, "")
                    ).strip()

                    if not fd_id:
                        raise ValueError(
                            f"FanDuel entry is missing slot {slot}."
                        )

                    fd_player = fd_players_by_id.get(fd_id)

                    if fd_player is None:
                        raise ValueError(
                            f"FanDuel player ID {fd_id!r} from slot "
                            f"{slot} is missing from the embedded "
                            "FanDuel player table."
                        )

                    fd_name_key = normalize_fanduel_name(
                        fd_player.get("nickname", "")
                    )

                    fd_team = normalize_team(
                        fd_player.get("team", "")
                    )

                    fd_position = str(
                        fd_player.get("position", "")
                    ).strip().upper()

                    fd_salary = fd_player.get(
                        "salary"
                    )

                    # FanDuel identity has already been canonicalized
                    # by normalize_fanduel_name(). No fuzzy matching.
                    fd_match_name_key = fd_name_key

                    matches = []

                    for _, candidate in late_swap_pool.iterrows():
                        candidate_name_key = normalize_name(
                            candidate["player_solver"]
                        )

                        candidate_team = normalize_team(
                            candidate["team_solver"]
                        )

                        candidate_position = str(
                            candidate["solver_position"]
                        ).strip().upper()

                        try:
                            candidate_salary = int(
                                candidate["salary_solver"]
                            )
                        except Exception:
                            candidate_salary = None

                        if fd_position in {"D", "DEF", "DST"}:
                            identity_match = (
                                candidate_position
                                in {"DST", "DEF", "D"}
                                and candidate_team == fd_team
                            )
                        else:
                            identity_match = (
                                candidate_name_key
                                == fd_match_name_key
                                and candidate_team == fd_team
                                and candidate_position == fd_position
                            )

                        # When FanDuel supplied salary, require it too.
                        # This adds another exact identity guard.
                        if (
                            identity_match
                            and fd_salary is not None
                            and candidate_salary is not None
                        ):
                            identity_match = (
                                int(fd_salary)
                                == candidate_salary
                            )

                        if identity_match:
                            matches.append(candidate)

                    if len(matches) != 1:
                        raise ValueError(
                            f"FanDuel slot {slot} / player "
                            f"{fd_player.get('nickname', fd_id)!r} "
                            f"resolved to {len(matches)} WFS players; "
                            "exactly one is required."
                        )

                    matched_row = matches[0]
                    auto_slot_labels[slot] = str(
                        matched_row["_label"]
                    )

                if len(auto_slot_labels) != len(ROSTER_SLOTS):
                    raise ValueError(
                        "FanDuel current lineup did not resolve "
                        "all nine roster slots."
                    )

            elif len(entry_lineups) > 1:
                st.info(
                    f"FanDuel template contains {len(entry_lineups)} "
                    "existing lineups. Automatic current-lineup loading "
                    "is disabled because the specific entry to late swap "
                    "is ambiguous."
                )

        except Exception as exc:
            auto_load_error = (
                "Automatic FanDuel lineup loading could not be completed. "
                "Please verify the uploaded contest template."
            )

    if auto_load_error:
        st.error(
            "Automatic FanDuel lineup loading failed closed: "
            f"{auto_load_error}"
        )
        return

    # -------------------------------------------------------------
    # Initialize Streamlit widget state exactly once per template.
    #
    # The template fingerprint prevents reruns from continually
    # overwriting a user's intentional manual dropdown change.
    # -------------------------------------------------------------

    auto_state_key = (
        f"late_swap_autoload_signature_{slate_slug}"
    )

    template_signature = (
        hashlib.sha256(persisted_template_bytes).hexdigest()
        if persisted_template_bytes
        else ""
    )

    if auto_slot_labels:
        previous_signature = st.session_state.get(
            auto_state_key
        )

        if previous_signature != template_signature:
            for slot in ROSTER_SLOTS:
                widget_key = (
                    f"late_swap_stage1_"
                    f"{normalize_slate(selected_slate)}_{slot}"
                )

                options = _late_swap_options_for_slot(
                    late_swap_pool,
                    slot,
                )

                default_label = auto_slot_labels[slot]

                if default_label not in options:
                    st.error(
                        f"Automatic FanDuel lineup loading failed: "
                        f"{slot} player is not a legal option for "
                        "that roster slot."
                    )
                    return

                st.session_state[widget_key] = default_label

            st.session_state[
                auto_state_key
            ] = template_signature

            st.success(
                "Current FanDuel lineup loaded automatically "
                "from the uploaded template."
            )

    selections = {}

    for slot in ROSTER_SLOTS:
        selections[slot] = st.selectbox(
            slot,
            _late_swap_options_for_slot(
                late_swap_pool,
                slot,
            ),
            key=(
                f"late_swap_stage1_"
                f"{normalize_slate(selected_slate)}_{slot}"
            ),
        )

    chosen = {
        slot: label
        for slot, label in selections.items()
        if label != "— Select player —"
    }

    if not chosen:
        st.caption(
            "Select the players from the existing FanDuel lineup "
            "to begin the audit."
        )
        return

    selected_rows = [
        label_to_row[label]
        for label in chosen.values()
    ]

    selected_keys = [
        str(row["_ui_key"])
        for row in selected_rows
    ]

    duplicate_players = (
        len(selected_keys)
        != len(set(selected_keys))
    )

    total_salary = sum(
        int(row["salary_solver"])
        for row in selected_rows
    )

    team_counts = pd.Series(
        [
            normalize_team(row["team_solver"])
            for row in selected_rows
        ],
        dtype="object",
    ).value_counts().to_dict()

    max_team_count = (
        max(team_counts.values())
        if team_counts
        else 0
    )

    if duplicate_players:
        st.error(
            "The same player cannot occupy more than one "
            "FanDuel roster slot."
        )

    if max_team_count > 4:
        st.error(
            "FanDuel legality failed: more than 4 selected "
            "players are from one NFL team."
        )

    if total_salary > 60000:
        st.error(
            "FanDuel legality failed: selected salary exceeds "
            "the $60,000 cap."
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Slots entered",
        f"{len(chosen)}/9",
    )

    c2.metric(
        "Salary entered",
        f"${total_salary:,.0f}",
    )

    c3.metric(
        "Remaining",
        f"${max(0, 60000 - total_salary):,.0f}",
    )

    # -----------------------------------------------------------------
    # Game-state audit
    # -----------------------------------------------------------------

    try:
        schedule_df, _, _, _ = load_data_center()
    except Exception:
        schedule_df = pd.DataFrame()

    now_et = datetime.now(
        ZoneInfo("America/New_York")
    )

    audit_rows = []

    current_slot_keys = {}
    slot_status = {}

    for slot in ROSTER_SLOTS:
        label = selections[slot]

        if label == "— Select player —":
            audit_rows.append(
                {
                    "Slot": slot,
                    "Player": "—",
                    "Team": "—",
                    "Salary": "—",
                    "Kickoff": "—",
                    "Status": "UNKNOWN",
                    "Reason": "Player not entered",
                }
            )
            continue

        row = label_to_row[label]

        lock = _late_swap_player_status(
            row,
            schedule_df,
            now_et,
        )

        key = str(row["_ui_key"])

        current_slot_keys[slot] = key
        slot_status[slot] = lock["status"]

        audit_rows.append(
            {
                "Slot": slot,
                "Player": str(row["player_solver"]),
                "Team": str(row["team_solver"]),
                "Salary": (
                    f'${int(row["salary_solver"]):,}'
                ),
                "Kickoff": lock["kickoff"],
                "Status": lock["status"],
                "Reason": lock["reason"],
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    st.markdown("##### 2. Player Lock Status")

    st.dataframe(
        audit_df,
        width="stretch",
        hide_index=True,
    )

    valid_current_lineup = (
        len(chosen) == 9
        and not duplicate_players
        and max_team_count <= 4
        and total_salary <= 60000
    )

    if not valid_current_lineup:
        st.error(
            "Complete a valid nine-player FanDuel lineup before "
            "continuing with Late Swap."
        )
        return

    unknown_count = int(
        audit_df["Status"].eq("UNKNOWN").sum()
    )

    locked_count = int(
        audit_df["Status"].eq("LOCKED").sum()
    )

    swappable_count = int(
        audit_df["Status"].eq("SWAPPABLE").sum()
    )

    if unknown_count:
        st.warning(
            f"{unknown_count} roster slot(s) could not be confirmed as swappable. "
            "Those players will remain protected and will not be changed."
        )
    else:
        st.success(
            f"Late Swap status confirmed — "
            f"{locked_count} locked slot(s), "
            f"{swappable_count} available to swap."
        )

    # -----------------------------------------------------------------
    # Build player-level state for the entire legal candidate pool.
    #
    # Missing/uncertain evidence intentionally becomes UNKNOWN.
    # -----------------------------------------------------------------

    player_status = {}

    for _, player_row in late_swap_pool.iterrows():
        player_key = str(player_row["_ui_key"])

        state = _late_swap_player_status(
            player_row,
            schedule_df,
            now_et,
        )

        player_status[player_key] = state["status"]

    # -----------------------------------------------------------------
    # Current-lineup display
    # -----------------------------------------------------------------

    original_rows = []

    for slot in ROSTER_SLOTS:
        row = label_to_row[selections[slot]]

        original_rows.append(
            {
                "Slot": slot,
                "Player": str(row["player_solver"]),
                "Team": str(row["team_solver"]),
                "Pos": str(row["solver_position"]),
                "Salary": int(row["salary_solver"]),
                "Projection": float(
                    row["projection_solver"]
                ),
                "Game": str(row.get("game_solver", "")),
                "Status": slot_status[slot],
                "Player Key": str(row["_ui_key"]),
            }
        )

    original_df = pd.DataFrame(original_rows)

    # -----------------------------------------------------------------
    # Stage 2 launch
    # -----------------------------------------------------------------

    st.markdown("##### 3. Optimize Remaining Lineup")

    if swappable_count == 0:
        st.info(
            "No roster slots are currently available to swap."
        )
        return

    st.caption(
        "Players whose games have started or cannot be safely confirmed remain protected. "
        "Only eligible players from games that have not started can be added."
    )

    optimize = st.button(
        "⚡ Optimize Late Swap",
        type="primary",
        width="stretch",
        key=(
            f"late_swap_optimize_"
            f"{normalize_slate(selected_slate)}"
        ),
    )

    if not optimize:
        return

    # Production-aligned settings used by the validated Stage 2 tests.
    settings = LateSwapSettings(
        qb_stack=1,
        bring_back=0,
        projection_loss_pct=0.015,
        salary_floor=0,
        max_team=0,
    )

    try:
        with st.spinner(
            "Optimizing your Late Swap lineup…"
        ):
            result = optimize_late_swap(
                pool=late_swap_pool,
                current_slot_keys=current_slot_keys,
                slot_status=slot_status,
                player_status=player_status,
                settings=settings,
            )

    except Exception as exc:
        st.error(
            "Late Swap could not be completed. Please review the lineup and try again."
        )
        return

    if not result.get("audit_ok", False):
        st.error(
            "The generated Late Swap lineup did not meet FanDuel lineup requirements."
        )
        return

    if not result.get("exact_lock_ok", False):
        st.error(
            "Late Swap could not safely preserve all locked players."
        )
        return

    result_df = pd.DataFrame(
        result["lineup"]
    ).copy()

    slot_order = {
        slot: n
        for n, slot in enumerate(ROSTER_SLOTS)
    }

    result_df["_slot_order"] = (
        result_df["slot"]
        .map(slot_order)
        .fillna(999)
    )

    result_df = (
        result_df
        .sort_values(
            "_slot_order",
            kind="mergesort",
        )
        .drop(columns=["_slot_order"])
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # True roster-composition changes
    # -----------------------------------------------------------------

    original_key_to_player = dict(
        zip(
            original_df["Player Key"].astype(str),
            original_df["Player"].astype(str),
        )
    )

    result_key_to_player = dict(
        zip(
            result_df["player_key"].astype(str),
            result_df["player"].astype(str),
        )
    )

    original_keys = set(
        original_key_to_player
    )

    result_keys = set(
        result_key_to_player
    )

    removed_keys = sorted(
        original_keys - result_keys
    )

    added_keys = sorted(
        result_keys - original_keys
    )

    players_removed = [
        original_key_to_player[key]
        for key in removed_keys
    ]

    players_added = [
        result_key_to_player[key]
        for key in added_keys
    ]

    roster_composition_changed = bool(
        players_removed or players_added
    )

    # -----------------------------------------------------------------
    # Result headline
    # -----------------------------------------------------------------

    st.markdown("##### 4. Optimized Late Swap")

    if roster_composition_changed:
        st.success(
            "Stage 2 found a legal roster-composition change."
        )
    elif result.get("changed_slot_count", 0):
        st.info(
            "Stage 2 changed slot assignments but retained "
            "the same nine players."
        )
    else:
        st.info(
            "Current lineup remains the strongest legal "
            "late-swap result inside the projection frontier."
        )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Salary",
        f'${int(result["salary_total"]):,}',
    )

    m2.metric(
        "Remaining",
        f'${int(result["salary_remaining"]):,}',
    )

    m3.metric(
        "Projection",
        f'{float(result["projection_total"]):.3f}',
    )

    m4.metric(
        "Changed Slots",
        int(result["changed_slot_count"]),
    )

    m5, m6, m7, m8 = st.columns(4)

    m5.metric(
        "Projection Optimum",
        f'{float(result["projection_optimum"]):.3f}',
    )

    m6.metric(
        "Projection Floor",
        f'{float(result["projection_floor"]):.3f}',
    )

    m7.metric(
        "Max One Team",
        int(result["max_players_one_team"]),
    )

    m8.metric(
        "Team Limit",
        int(result["team_limit"]),
    )

    # -----------------------------------------------------------------
    # Before / after
    # -----------------------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.markdown("###### Before")

        before_display = original_df[
            [
                "Slot",
                "Player",
                "Team",
                "Pos",
                "Salary",
                "Projection",
                "Game",
                "Status",
            ]
        ].copy()

        before_display["Salary"] = (
            before_display["Salary"]
            .map(lambda x: f"${int(x):,}")
        )

        before_display["Projection"] = (
            before_display["Projection"]
            .map(lambda x: f"{float(x):.3f}")
        )

        st.dataframe(
            before_display,
            width="stretch",
            hide_index=True,
        )

    with right:
        st.markdown("###### After")

        after_display = result_df.rename(
            columns={
                "slot": "Slot",
                "player": "Player",
                "team": "Team",
                "position": "Pos",
                "salary": "Salary",
                "projection": "Projection",
                "game": "Game",
                "late_swap_status": "Status",
            }
        )[
            [
                "Slot",
                "Player",
                "Team",
                "Pos",
                "Salary",
                "Projection",
                "Game",
                "Status",
            ]
        ].copy()

        after_display["Salary"] = (
            after_display["Salary"]
            .map(lambda x: f"${int(x):,}")
        )

        after_display["Projection"] = (
            after_display["Projection"]
            .map(lambda x: f"{float(x):.3f}")
        )

        st.dataframe(
            after_display,
            width="stretch",
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # True OUT / IN changes
    # -----------------------------------------------------------------

    st.markdown("##### 5. Roster Changes")

    change_left, change_right = st.columns(2)

    with change_left:
        st.markdown("###### Players OUT")

        if players_removed:
            for player in players_removed:
                st.write(f"• {player}")
        else:
            st.caption("None")

    with change_right:
        st.markdown("###### Players IN")

        if players_added:
            for player in players_added:
                st.write(f"• {player}")
        else:
            st.caption("None")

    changed_slots = result.get(
        "changed_slots",
        [],
    )

    if changed_slots:
        st.caption(
            "Slot assignments changed: "
            + ", ".join(changed_slots)
        )
    else:
        st.caption(
            "No exact slot assignments changed."
        )

    # -----------------------------------------------------------------
    # FanDuel Late Swap export
    # -----------------------------------------------------------------

    st.markdown("##### 6. FanDuel Late Swap Export")

    if late_swap_contest_verified and persisted_template_bytes:
        try:
            # Convert Stage 2's slot-oriented result into the same
            # one-row lineup shape used by the shared FanDuel exporter.
            result_by_slot = {
                str(row["slot"]): str(row["player"])
                for _, row in result_df.iterrows()
            }

            missing_export_slots = [
                slot
                for slot in ROSTER_SLOTS
                if slot not in result_by_slot
                or not result_by_slot[slot].strip()
            ]

            if missing_export_slots:
                raise ValueError(
                    "Late Swap export missing roster slot(s): "
                    + ", ".join(missing_export_slots)
                )

            late_swap_export_df = pd.DataFrame(
                [
                    {
                        "lineup": 1,
                        "QB": result_by_slot["QB"],
                        "RB1": result_by_slot["RB1"],
                        "RB2": result_by_slot["RB2"],
                        "WR1": result_by_slot["WR1"],
                        "WR2": result_by_slot["WR2"],
                        "WR3": result_by_slot["WR3"],
                        "TE": result_by_slot["TE"],
                        "FLEX": result_by_slot["FLEX"],
                        "DST": result_by_slot["DST"],
                    }
                ]
            )

            late_swap_fd_bytes, late_swap_fd_meta = (
                build_fanduel_upload(
                    persisted_template_bytes,
                    late_swap_export_df,
                    late_swap_pool,
                )
            )

            validate_fanduel_upload(
                late_swap_fd_bytes,
                late_swap_fd_meta["entries_written"],
            )

            if late_swap_fd_meta["entries_written"] != 1:
                raise ValueError(
                    "Late Swap export expected exactly one existing "
                    "FanDuel entry row to be updated."
                )

            if late_swap_fd_meta["mapping_count"] != 9:
                raise ValueError(
                    "Late Swap export did not map exactly nine "
                    "FanDuel roster cells."
                )

            st.success(
                "FanDuel Late Swap export PASS — "
                "9/9 roster cells mapped to exact contest-specific "
                "FanDuel player IDs."
            )

            st.download_button(
                "Download FanDuel Late Swap CSV",
                data=late_swap_fd_bytes,
                file_name=(
                    f"FanDuel-NFL-{safe_slug(selected_slate)}-"
                    "late-swap-upload.csv"
                ),
                mime="text/csv",
                type="primary",
                key=(
                    f"late_swap_fd_download_"
                    f"{normalize_slate(selected_slate)}"
                ),
                on_click="ignore",
            )

            st.caption(
                "The original FanDuel template structure is preserved. "
                "Only the nine roster cells of the existing entry are changed."
            )

        except Exception as exc:
            st.error(
                "FanDuel Late Swap export was blocked because the lineup "
                "could not be validated against the selected contest."
            )
            st.caption(
                "Export requires all nine players to match the selected FanDuel contest."
            )

    else:
        st.warning(
            "FanDuel Late Swap export requires a verified "
            "FanDuel contest template."
        )

    # -----------------------------------------------------------------
    # Hard audit summary
    # -----------------------------------------------------------------

    st.markdown("##### 7. Late Swap Summary")

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "FanDuel Lineup",
        "PASS" if result.get("audit_ok") else "FAIL",
    )

    a2.metric(
        "Locked Players",
        "PASS" if result.get("exact_lock_ok") else "FAIL",
    )

    a3.metric(
        "Locked Slots",
        len(result.get("locked_slots", [])),
    )

    a4.metric(
        "Available Slots",
        len(result.get("swappable_slots", [])),
    )

    locked_slots = result.get(
        "locked_slots",
        [],
    )

    swappable_slots = result.get(
        "swappable_slots",
        [],
    )

    st.caption(
        "Protected exact-slot cells: "
        + (
            ", ".join(locked_slots)
            if locked_slots
            else "None"
        )
    )

    st.caption(
        "Replaceable cells: "
        + (
            ", ".join(swappable_slots)
            if swappable_slots
            else "None"
        )
    )

    if late_swap_contest_verified:
        st.success(
            "Late Swap is ready for the selected FanDuel contest."
        )
    else:
        st.warning(
            "Late Swap lineup created, but a FanDuel contest template was not provided. "
            "Confirm player eligibility before uploading."
        )



def intersect_with_fanduel_contest(
    slate_df: pd.DataFrame,
    template_bytes: bytes,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Intersect the solver-ready slate with the exact FanDuel contest player pool.

    This changes only player eligibility at the UI boundary. It does not alter
    projections, GPP ranks, objectives, exposure logic, or the frozen solver.

    Offense requires exact normalized player name + team + position.
    D/ST requires exact team and FanDuel position D.
    """
    parsed = parse_fanduel_template(template_bytes)
    fd_players = parsed["player_rows"]

    allowed = []
    blocked = []

    for idx, row in slate_df.iterrows():
        name_key = normalize_name(row["player_solver"])
        team = normalize_team(row["team_solver"])
        pos = str(row["solver_position"]).strip().upper()

        if pos in {"DST", "DEF", "D"}:
            matches = [
                p for p in fd_players
                if p["position"] == "D" and p["team"] == team
            ]
        else:
            matches = [
                p for p in fd_players
                if p["name_key"] == name_key
                and p["team"] == team
                and p["position"] == pos
            ]

        if len(matches) == 1:
            allowed.append(idx)
        else:
            blocked.append(idx)

    contest_pool = slate_df.loc[allowed].copy()
    outside_pool = slate_df.loc[blocked].copy()

    fd_teams = sorted({
        p["team"] for p in fd_players if p["team"]
    })
    solver_teams = sorted({
        normalize_team(x) for x in contest_pool["team_solver"].dropna()
    })

    return contest_pool, outside_pool, {
        "fanduel_players": len(fd_players),
        "solver_players_before": len(slate_df),
        "solver_players_in_contest": len(contest_pool),
        "solver_players_blocked": len(outside_pool),
        "fanduel_teams": fd_teams,
        "solver_teams_in_contest": solver_teams,
        "entry_count": len(parsed["entry_row_indexes"]),
    }


# ---------------------------------------------------------------------------
# WFS FAN DUEL SINGLE-GAME / SHOWDOWN UI — isolated from frozen Classic path
# ---------------------------------------------------------------------------

def showdown_pool_freshness_token() -> tuple[int, int]:
    """
    Return deterministic publication freshness for the isolated
    Single-Game projection pool.

    Including this token in the cached loader arguments invalidates
    Streamlit's cached DataFrame whenever the published pool changes.
    """
    if not SHOWDOWN_POOL.is_file():
        raise RuntimeError(
            "Single-Game player pool is unavailable."
        )

    stat = SHOWDOWN_POOL.stat()

    return (
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


@st.cache_data(show_spinner=False)
def load_showdown_projection_pool(
    freshness_token: tuple[int, int],
) -> pd.DataFrame:
    """Load only the isolated Single-Game projection pool; never the Classic table."""
    del freshness_token

    if not SHOWDOWN_POOL.exists():
        raise RuntimeError("Single-Game player pool is unavailable.")
    df = pd.read_parquet(SHOWDOWN_POOL).copy()
    required = {
        "public_slate_name", "game", "player", "player_id", "team", "position",
        "salary", "projection", "projection_source", "projection_status",
        "mvp_salary", "mvp_projection",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Single-Game player pool is missing required fields: " + ", ".join(missing))
    for col in ("salary", "projection", "mvp_salary", "mvp_projection"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_showdown_injury_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply authoritative injury/availability eligibility to the
    isolated Single-Game UI universe.

    Single-Game inherits Classic semantics:
      * BLOCK may only remove eligibility.
      * Missing consensus is healthy-by-absence.
      * Malformed authority fails closed.

    The underlying projection pool is not modified.
    """
    injury_path = (
        APP_DIR
        / "data"
        / "parquet"
        / "injury_consensus_current.parquet"
    )

    if not injury_path.exists():
        raise RuntimeError(
            "Single-Game availability authority is unavailable."
        )

    injury = pd.read_parquet(injury_path).copy()

    required = {
        "gsis_id",
        "injury_gate",
        "consensus_status",
        "injury_gate_reason",
        "injury_risk",
    }

    missing = sorted(required - set(injury.columns))

    if missing:
        raise RuntimeError(
            "Single-Game availability authority is missing "
            "required fields: "
            + ", ".join(missing)
        )

    if injury.empty:
        raise RuntimeError(
            "Single-Game availability authority returned no rows."
        )

    for col in required:
        injury[col] = (
            injury[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    injury["injury_gate"] = (
        injury["injury_gate"]
        .str.upper()
    )

    if injury["gsis_id"].eq("").any():
        raise RuntimeError(
            "Single-Game availability authority contains "
            "blank player IDs."
        )

    duplicate_ids = injury["gsis_id"].duplicated(
        keep=False
    )

    if duplicate_ids.any():
        raise RuntimeError(
            "Single-Game availability authority contains "
            "duplicate player IDs."
        )

    invalid = sorted(
        set(injury["injury_gate"])
        - {"ALLOW", "BLOCK"}
    )

    if invalid:
        raise RuntimeError(
            "Single-Game availability authority contains "
            "invalid gate values."
        )

    block_ids = set(
        injury.loc[
            injury["injury_gate"].eq("BLOCK"),
            "gsis_id",
        ]
    )

    out = df.copy()

    if "player_id" not in out.columns:
        raise RuntimeError(
            "Single-Game player pool is missing player_id."
        )

    player_ids = (
        out["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if player_ids.eq("").any():
        raise RuntimeError(
            "Single-Game player pool contains blank player IDs."
        )

    blocked_mask = player_ids.isin(block_ids)

    out = out.loc[
        ~blocked_mask
    ].copy()

    remaining_ids = set(
        out["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    escapes = remaining_ids & block_ids

    if escapes:
        raise RuntimeError(
            "Single-Game availability gate escape detected."
        )

    return out


def available_showdown_slates(df: pd.DataFrame) -> list[str]:
    """
    Return complete READY Single-Game slates whose authoritative NFL game
    is still active.

    Lifecycle authority is nfl.db -> games.completed.

    The Single-Game artifact currently carries matchup identity rather than
    game_id. Resolution therefore fails closed unless the matchup maps
    unambiguously to exactly one relevant schedule row.
    """
    if (
        df is None
        or df.empty
        or "public_slate_name" not in df.columns
        or "game" not in df.columns
    ):
        return []

    df = apply_showdown_injury_gate(df)

    if df.empty:
        return []

    def _clean_team(value) -> str:
        team = str(value or "").strip().upper()
        aliases = {
            "LA": "LAR",
            "WAS": "WSH",
            "JAC": "JAX",
        }
        return aliases.get(team, team)

    def _parse_matchup(value):
        raw = str(value or "").strip().upper()
        if "@" not in raw:
            return None

        parts = raw.split("@")
        if len(parts) != 2:
            return None

        away = _clean_team(parts[0])
        home = _clean_team(parts[1])

        if not away or not home:
            return None

        return away, home

    # Read schedule authority once for the selector build.
    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as conn:
            conn.row_factory = sqlite3.Row
            schedule_rows = conn.execute(
                """
                SELECT
                    game_id,
                    season,
                    week,
                    game_date,
                    away_team,
                    home_team,
                    completed
                FROM games
                """
            ).fetchall()
    except Exception:
        # Schedule authority is mandatory for lineup-generation eligibility.
        return []

    ready = []

    for slate_name, group in df.groupby(
        "public_slate_name",
        sort=True,
    ):
        slate_name = str(slate_name).strip()
        if not slate_name:
            continue

        # ------------------------------------------------------
        # Existing projection / identity readiness gates.
        # ------------------------------------------------------
        status_ok = (
            group["projection_status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.startswith("READY")
            .all()
        )

        numeric_ok = (
            group[
                [
                    "salary",
                    "projection",
                    "mvp_salary",
                    "mvp_projection",
                ]
            ]
            .notna()
            .all()
            .all()
        )

        identity_ok = (
            group["player_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .all()
            and group["team"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        )

        unique_ok = not group["player_id"].astype(str).duplicated().any()

        if not (
            status_ok
            and numeric_ok
            and identity_ok
            and unique_ok
        ):
            continue

        # ------------------------------------------------------
        # Exact matchup identity must be internally consistent.
        # ------------------------------------------------------
        game_values = sorted(
            {
                str(v).strip()
                for v in group["game"].tolist()
                if str(v).strip()
                and str(v).strip().lower()
                not in {"nan", "none", "null"}
            }
        )

        if len(game_values) != 1:
            continue

        matchup = _parse_matchup(game_values[0])
        if matchup is None:
            continue

        away_team, home_team = matchup

        # ------------------------------------------------------
        # Schedule lifecycle authority.
        #
        # Prefer the latest scheduled occurrence of the exact
        # away/home orientation. If more than one row exists on
        # that latest date, identity is ambiguous -> fail closed.
        # ------------------------------------------------------
        matches = [
            row
            for row in schedule_rows
            if _clean_team(row["away_team"]) == away_team
            and _clean_team(row["home_team"]) == home_team
        ]

        if not matches:
            continue

        dated_matches = [
            row
            for row in matches
            if str(row["game_date"] or "").strip()
        ]

        if not dated_matches:
            continue

        latest_date = max(
            str(row["game_date"]).strip()
            for row in dated_matches
        )

        relevant = [
            row
            for row in dated_matches
            if str(row["game_date"]).strip() == latest_date
        ]

        if len(relevant) != 1:
            continue

        game_row = relevant[0]

        try:
            completed = bool(int(game_row["completed"] or 0))
        except Exception:
            # Malformed lifecycle authority fails closed.
            continue

        if completed:
            continue

        ready.append(slate_name)

    return sorted(ready, key=str.lower)

def render_public_showdown_lineup_cards(
    display_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> None:
    """Render Single-Game lineups using the Classic public card presentation."""

    import html

    if display_df is None or display_df.empty:
        st.info("No lineups to display.")
        return

    if long_df is None or long_df.empty:
        raise RuntimeError(
            "Single-Game lineup card detail is unavailable."
        )

    required = {
        "lineup",
        "role",
        "player",
        "team",
        "salary",
        "projection",
        "lineup_salary",
        "lineup_projection",
    }
    missing = sorted(required - set(long_df.columns))
    if missing:
        raise RuntimeError(
            "Single-Game lineup card detail is missing required fields: "
            + ", ".join(missing)
        )

    cards = []

    for _, lineup_row in display_df.iterrows():
        lineup_id = int(lineup_row["Lineup"])

        detail = long_df[
            pd.to_numeric(long_df["lineup"], errors="coerce")
            == lineup_id
        ].copy()

        if len(detail) != 6:
            raise RuntimeError(
                f"Single-Game lineup card expected 6 players for lineup "
                f"{lineup_id}; found {len(detail)}."
            )

        detail["_role_order"] = (
            detail["role"]
            .astype(str)
            .str.upper()
            .map({"MVP": 0, "FLEX": 1})
            .fillna(99)
        )
        detail["_original_order"] = range(len(detail))
        detail = detail.sort_values(
            ["_role_order", "_original_order"],
            kind="mergesort",
        )

        salary = pd.to_numeric(
            pd.Series([lineup_row.get("Salary")]),
            errors="coerce",
        ).iloc[0]

        projection = pd.to_numeric(
            pd.Series([lineup_row.get("Projection")]),
            errors="coerce",
        ).iloc[0]

        salary_text = (
            f"${salary:,.0f}"
            if pd.notna(salary)
            else "—"
        )
        projection_text = (
            f"{projection:.2f} PTS"
            if pd.notna(projection)
            else "—"
        )

        rows = []

        for _, player_row in detail.iterrows():
            role = str(
                player_row.get("role", "")
            ).strip().upper()

            display_slot = (
                "MVP"
                if role == "MVP"
                else "FLEX"
            )

            player = html.escape(
                str(player_row.get("player", "—"))
            )

            raw_team = str(
                player_row.get("team", "") or ""
            ).strip()

            canonical_team = _dc_team_code(raw_team)
            team = html.escape(
                canonical_team or raw_team or "—"
            )

            opponent_value = (
                player_row.get("opponent", "—")
                if "opponent" in detail.columns
                else "—"
            )
            opponent = html.escape(
                str(opponent_value or "—")
            )

            team_logo = _dc_team_logo_html(
                canonical_team,
                "wfs-lc-team-logo",
            )

            if role == "MVP":
                player_salary = pd.to_numeric(
                    pd.Series([
                        player_row.get(
                            "mvp_salary",
                            player_row.get("salary"),
                        )
                    ]),
                    errors="coerce",
                ).iloc[0]

                player_projection = pd.to_numeric(
                    pd.Series([
                        player_row.get(
                            "mvp_projection",
                            player_row.get("projection"),
                        )
                    ]),
                    errors="coerce",
                ).iloc[0]
            else:
                player_salary = pd.to_numeric(
                    pd.Series([
                        player_row.get("salary")
                    ]),
                    errors="coerce",
                ).iloc[0]

                player_projection = pd.to_numeric(
                    pd.Series([
                        player_row.get("projection")
                    ]),
                    errors="coerce",
                ).iloc[0]

            player_salary_text = (
                f"${player_salary / 1000:.1f}K"
                if pd.notna(player_salary)
                else "—"
            )

            player_projection_text = (
                f"{player_projection:.2f}"
                if pd.notna(player_projection)
                else "—"
            )

            rows.append(
                f"""
                <div class="wfs-lc-row">
                    <div class="wfs-lc-pos">{html.escape(display_slot)}</div>
                    <div class="wfs-lc-player">
                        {team_logo}
                        <span class="wfs-lc-player-name">{player}</span>
                    </div>
                    <div class="wfs-lc-team">{team}</div>
                    <div class="wfs-lc-opp">{opponent}</div>
                    <div class="wfs-lc-sal">{player_salary_text}</div>
                    <div class="wfs-lc-fpts">{player_projection_text}</div>
                </div>
                """
            )

        cards.append(
            f"""
            <div class="wfs-lineup-card">
                <div class="wfs-lc-head">
                    <div class="wfs-lc-number">#{lineup_id}</div>
                    <div class="wfs-lc-proj">{projection_text}</div>
                    <div class="wfs-lc-salary">{salary_text}</div>
                </div>

                <div class="wfs-lc-columns">
                    <div>POS</div>
                    <div>PLAYER</div>
                    <div>TEAM</div>
                    <div>OPP</div>
                    <div>SAL</div>
                    <div>FPTS</div>
                </div>

                {''.join(rows)}
            </div>
            """
        )

    st.markdown(
        """
        <style>
        .wfs-lineup-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
            gap:14px;
            width:100%;
            margin:.35rem 0 1rem 0;
        }

        .wfs-lineup-card {
            min-width:0;
            overflow:hidden;
            border:1px solid rgba(148,163,184,.22);
            border-radius:12px;
            background:rgba(15,23,42,.72);
            box-shadow:0 8px 24px rgba(0,0,0,.14);
        }

        .wfs-lc-head {
            display:grid;
            grid-template-columns:1fr auto auto;
            gap:14px;
            align-items:center;
            padding:11px 13px;
            border-bottom:1px solid rgba(148,163,184,.20);
        }

        .wfs-lc-number {
            font-size:1rem;
            font-weight:900;
            color:#f8fafc;
        }

        .wfs-lc-proj {
            font-size:.88rem;
            font-weight:900;
            color:#e2e8f0;
            white-space:nowrap;
        }

        .wfs-lc-salary {
            font-size:.88rem;
            font-weight:900;
            color:#f8fafc;
            white-space:nowrap;
        }

        .wfs-lc-columns,
        .wfs-lc-row {
            display:grid;
            grid-template-columns:42px minmax(130px,1fr) 45px 45px 55px 52px;
            gap:5px;
            align-items:center;
        }

        .wfs-lc-columns {
            padding:7px 10px;
            font-size:.62rem;
            font-weight:900;
            letter-spacing:.04em;
            color:#94a3b8;
            border-bottom:1px solid rgba(148,163,184,.16);
        }

        .wfs-lc-row {
            padding:8px 10px;
            min-height:35px;
            border-bottom:1px solid rgba(148,163,184,.10);
            font-size:.76rem;
            color:#e2e8f0;
        }

        .wfs-lc-row:last-child {
            border-bottom:0;
        }

        .wfs-lc-pos {
            font-weight:900;
            color:#cbd5e1;
        }

        .wfs-lc-player {
            min-width:0;
            overflow:hidden;
            display:flex;
            align-items:center;
            gap:6px;
            white-space:nowrap;
            font-weight:800;
            color:#f8fafc;
        }

        .wfs-lc-player-name {
            min-width:0;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }

        .wfs-lc-team-logo {
            width:22px;
            height:22px;
            object-fit:contain;
            flex:0 0 22px;
        }

        .wfs-lc-team,
        .wfs-lc-opp {
            font-weight:700;
            color:#cbd5e1;
        }

        .wfs-lc-sal,
        .wfs-lc-fpts {
            text-align:right;
            font-variant-numeric:tabular-nums;
            white-space:nowrap;
        }

        .wfs-lc-fpts {
            font-weight:900;
            color:#f8fafc;
        }

        @media (max-width:640px) {
            .wfs-lineup-grid {
                grid-template-columns:minmax(0,1fr);
                gap:10px;
            }

            .wfs-lc-head {
                gap:8px;
                padding:10px 9px;
            }

            .wfs-lc-columns,
            .wfs-lc-row {
                grid-template-columns:
                    34px minmax(105px,1fr) 34px 34px 48px 43px;
                gap:3px;
                padding-left:7px;
                padding-right:7px;
            }

            .wfs-lc-columns {
                font-size:.54rem;
            }

            .wfs-lc-row {
                font-size:.68rem;
            }

            .wfs-lc-player {
                gap:4px;
            }

            .wfs-lc-team-logo {
                width:20px;
                height:20px;
                flex-basis:20px;
            }

            .wfs-lc-proj,
            .wfs-lc-salary {
                font-size:.78rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    card_html = (
        '<div class="wfs-lineup-grid">'
        + "".join(cards)
        + "</div>"
    )
    card_html = " ".join(card_html.split())

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


def _wfs_showdown_display(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert solver long-form output to one human-facing row per lineup."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    required = {"lineup", "role", "player", "lineup_salary", "lineup_projection"}
    missing = sorted(required - set(long_df.columns))
    if missing:
        raise RuntimeError("Single-Game solver output is missing required fields: " + ", ".join(missing))
    rows = []
    for lineup_id, group in long_df.groupby("lineup", sort=True):
        mvp = group.loc[group["role"].astype(str).eq("MVP")]
        flex = group.loc[group["role"].astype(str).eq("FLEX")]
        if len(mvp) != 1 or len(flex) != 5 or len(group) != 6:
            raise RuntimeError(f"Single-Game lineup {lineup_id} failed roster-shape validation.")
        players = group["player_id"].astype(str).tolist() if "player_id" in group.columns else group["player"].astype(str).tolist()
        if len(set(players)) != 6:
            raise RuntimeError(f"Single-Game lineup {lineup_id} contains a duplicate player.")
        salary = int(round(float(group["lineup_salary"].iloc[0])))
        projection = float(group["lineup_projection"].iloc[0])
        if salary > 60000:
            raise RuntimeError(f"Single-Game lineup {lineup_id} exceeds the $60,000 salary cap.")
        row = {"Lineup": int(lineup_id), "MVP": str(mvp["player"].iloc[0]), "Salary": salary, "Projection": projection}
        for n, player in enumerate(flex["player"].astype(str).tolist(), start=1):
            row[f"AnyFLEX {n}"] = player
        rows.append(row)
    cols = ["Lineup", "MVP", "AnyFLEX 1", "AnyFLEX 2", "AnyFLEX 3", "AnyFLEX 4", "AnyFLEX 5", "Salary", "Projection"]
    return pd.DataFrame(rows).reindex(columns=cols).sort_values("Lineup", kind="mergesort").reset_index(drop=True)


def render_showdown_builder(selected_slate: str, lineups: int, workspace_mode: str = "Public") -> None:
    """Render the isolated Single-Game builder without entering the Classic generator."""
    st.markdown('<div class="qt-section">1. Build New Lineups</div>', unsafe_allow_html=True)
    st.caption("Build FanDuel NFL Single-Game lineups from the selected matchup.")

    if workspace_mode == "Public":
        _wfs_render_public_data_freshness(
            SHOWDOWN_POOL,
            product_label="Single-Game player data",
            authority_mode="single_game",
        )

    try:
        showdown_pool = load_showdown_projection_pool(showdown_pool_freshness_token())
    except Exception as exc:
        if workspace_mode == "Admin":
            st.error(f"Single-Game is unavailable: {exc}")
        else:
            st.error("Single-Game contests are temporarily unavailable.")
        return
    slate_df = showdown_pool.loc[showdown_pool["public_slate_name"].astype(str).eq(str(selected_slate))].copy()
    if slate_df.empty:
        st.error("Single-Game is unavailable for the selected matchup.")
        return

    try:
        slate_df = apply_showdown_injury_gate(
            slate_df
        )
    except Exception as exc:
        if workspace_mode == "Admin":
            st.error(
                "Single-Game availability validation failed: "
                f"{exc}"
            )
        else:
            st.error(
                "Single-Game contests are temporarily unavailable."
            )
        return

    if slate_df.empty:
        st.error(
            "No eligible players are available for the "
            "selected matchup."
        )
        return

    ready_mask = slate_df["projection_status"].fillna("").astype(str).str.upper().str.startswith("READY")
    if not ready_mask.all():
        st.error("Single-Game player data is not fully ready for this matchup.")
        return

    st.markdown('<div class="qt-section">2. Contest Intelligence</div>', unsafe_allow_html=True)
    team_count = int(slate_df["team"].astype(str).nunique())
    top_projection = float(pd.to_numeric(slate_df["projection"], errors="coerce").max())
    salary_min = int(pd.to_numeric(slate_df["salary"], errors="coerce").min())
    salary_max = int(pd.to_numeric(slate_df["salary"], errors="coerce").max())
    st.caption(f"{selected_slate} Single-Game")
    st.markdown(
        f"""
        <div class="qt-card" style="padding:.9rem 1rem;margin:.55rem 0 1rem 0;">
          <div style="display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;font-size:.88rem;font-weight:800;color:#0f172a;">
            <span>{len(slate_df)} Players</span><span>•</span><span>{team_count} Teams</span><span>•</span>
            <span>Top projection: {top_projection:.2f} pts</span><span>•</span><span>${salary_min:,}–${salary_max:,}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if workspace_mode == "Public":
        with st.expander("🏈 FanDuel Single-Game Rules", expanded=False):
            st.markdown(
                """
                **FanDuel NFL Single-Game lineup**

                - **6 players:** 1 MVP + 5 AnyFLEX
                - **Salary cap:** $60,000 maximum
                - **MVP:** 1.5× fantasy points and 1.5× salary
                - **AnyFLEX:** Any eligible player from the selected Single-Game player pool
                - **One underlying player:** Cannot occupy both MVP and AnyFLEX
                """
            )
    else:
        with st.expander("Admin Single-Game diagnostics", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.metric("Pool rows", len(slate_df))
            d2.metric("Solver SHA", _wfs_hash_file(SHOWDOWN_SOLVER)[:12])
            d3.metric("Pool SHA", _wfs_hash_file(SHOWDOWN_POOL)[:12])
            source_counts = slate_df["projection_source"].fillna("UNKNOWN").astype(str).value_counts().rename_axis("Projection Source").reset_index(name="Rows")
            st.dataframe(source_counts, width="stretch", hide_index=True)

    st.markdown(
        '<div class="qt-section">3. Player Controls</div>',
        unsafe_allow_html=True,
    )

    control_df = slate_df.copy()

    control_df["_control_label"] = control_df.apply(
        lambda r: (
            f'{str(r["player"])} — '
            f'{str(r["team"])} '
            f'{str(r["position"])} — '
            f'${int(round(float(r["salary"]))):,} — '
            f'{float(r["projection"]):.2f}'
        ),
        axis=1,
    )

    label_to_player_id = {
        str(r["_control_label"]): str(r["player_id"])
        for _, r in control_df.iterrows()
    }

    player_labels = sorted(
        label_to_player_id.keys(),
        key=str.lower,
    )

    st.caption(
        "Customize this Single-Game build. "
        "Locks must appear in every lineup, excluded players "
        "cannot appear, and MVP Override forces your chosen "
        "player into the MVP spot."
    )

    locked_labels = st.multiselect(
        "🔒 Lock Players",
        player_labels,
        default=[],
        key=(
            "wfs_showdown_locks_"
            + safe_slug(selected_slate)
        ),
        help=(
            "Every locked player must appear in every generated lineup."
        ),
    )

    excluded_labels = st.multiselect(
        "🚫 Exclude Players",
        player_labels,
        default=[],
        key=(
            "wfs_showdown_excludes_"
            + safe_slug(selected_slate)
        ),
        help=(
            "Excluded players cannot appear in any generated lineup."
        ),
    )

    mvp_options = [
        "Automatic — let WFS choose"
    ] + player_labels

    mvp_choice = st.selectbox(
        "⭐ MVP",
        mvp_options,
        index=0,
        key=(
            "wfs_showdown_mvp_"
            + safe_slug(selected_slate)
        ),
        help=(
            "Leave this on Automatic to use the normal optimizer. "
            "Choose a player to force that player into the MVP role."
        ),
    )

    lock_player_ids = [
        label_to_player_id[label]
        for label in locked_labels
    ]

    exclude_player_ids = [
        label_to_player_id[label]
        for label in excluded_labels
    ]

    mvp_player_id = ""

    if mvp_choice != "Automatic — let WFS choose":
        mvp_player_id = label_to_player_id[mvp_choice]

    control_errors = []

    lock_set = set(lock_player_ids)
    exclude_set = set(exclude_player_ids)

    overlap = lock_set & exclude_set

    if overlap:
        overlap_names = sorted(
            control_df.loc[
                control_df["player_id"]
                .astype(str)
                .isin(overlap),
                "player",
            ]
            .astype(str)
            .tolist(),
            key=str.lower,
        )

        control_errors.append(
            "A player cannot be both locked and excluded: "
            + ", ".join(overlap_names)
        )

    if len(lock_set) > 6:
        control_errors.append(
            "Single-Game lineups contain only 6 players, "
            "so no more than 6 players can be locked."
        )

    if len(control_df) - len(exclude_set) < 6:
        control_errors.append(
            "Too many players are excluded to build a legal "
            "6-player Single-Game lineup."
        )

    if mvp_player_id and mvp_player_id in exclude_set:
        control_errors.append(
            "The selected MVP cannot also be excluded."
        )

    if mvp_player_id:
        mvp_name_rows = control_df.loc[
            control_df["player_id"]
            .astype(str)
            .eq(str(mvp_player_id)),
            "player",
        ]

        if not mvp_name_rows.empty:
            st.caption(
                "MVP Override: "
                + str(mvp_name_rows.iloc[0])
            )

    if locked_labels:
        st.caption(
            f"Locked players: {len(locked_labels)}"
        )

    if excluded_labels:
        st.caption(
            f"Excluded players: {len(excluded_labels)}"
        )

    for message in control_errors:
        st.error(message)

    injury_authority_path = (
        APP_DIR
        / "data"
        / "parquet"
        / "injury_consensus_current.parquet"
    )

    showdown_state_payload = {
        "cache_schema": 1,
        "slate": str(selected_slate),
        "lineups": int(lineups),
        "locks": sorted(str(x) for x in lock_set),
        "excludes": sorted(str(x) for x in exclude_set),
        "mvp": str(mvp_player_id or "AUTO"),
        "pool_sha256": _wfs_hash_file(
            SHOWDOWN_POOL
        ),
        "injury_sha256": _wfs_hash_file(
            injury_authority_path
        ),
        "solver_sha256": _wfs_hash_file(
            SHOWDOWN_SOLVER
        ),
    }

    showdown_state_signature = hashlib.sha256(
        json.dumps(
            showdown_state_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    previous_signature = st.session_state.get(
        "showdown_state_signature"
    )

    if (
        previous_signature
        and previous_signature
        != showdown_state_signature
    ):
        st.session_state.pop(
            "showdown_lineups_long_df",
            None,
        )
        st.session_state.pop(
            "showdown_lineups_display_df",
            None,
        )
        st.session_state.pop(
            "showdown_last_slate",
            None,
        )
        st.session_state.pop(
            "showdown_last_lineup_count",
            None,
        )
        st.session_state.pop(
            "showdown_state_signature",
            None,
        )

    st.markdown(
        '<div class="qt-section">4. Launch Portfolio</div>',
        unsafe_allow_html=True,
    )
    generate = st.button(
        "⚡ Generate Single-Game Lineups",
        type="primary",
        width="stretch",
        key=f"wfs_showdown_generate_{safe_slug(selected_slate)}_{int(lineups)}",
    )
    if generate:
        if control_errors:
            st.error(
                "Fix the Player Controls above before generating lineups."
            )
            return

        if not SHOWDOWN_SOLVER.exists():
            st.error("Single-Game solver is unavailable.")
            return
        cache_dir = APP_DIR / "data" / "cache" / "fanduel_showdown_ui"
        cache_dir.mkdir(parents=True, exist_ok=True)
        user_key = str((WFS_USER or {}).get("sub") or "anonymous")
        token_raw = f"{user_key}|{selected_slate}|{int(lineups)}|{datetime.utcnow().isoformat()}"
        token = hashlib.sha256(token_raw.encode("utf-8")).hexdigest()[:20]
        output_path = cache_dir / f"showdown_{safe_slug(selected_slate)}_{token}.csv"
        cmd = [
            sys.executable,
            str(SHOWDOWN_SOLVER),
            "--slate",
            str(selected_slate),
            "--lineups",
            str(int(lineups)),
            "--output",
            str(output_path),
        ]

        for player_id in sorted(lock_set):
            cmd.extend(
                ["--lock-player-id", player_id]
            )

        for player_id in sorted(exclude_set):
            cmd.extend(
                ["--exclude-player-id", player_id]
            )

        if mvp_player_id:
            cmd.extend(
                ["--mvp-player-id", mvp_player_id]
            )

        try:
            with st.spinner("Generating Single-Game lineups…"):
                proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True, timeout=120)
            if workspace_mode == "Admin":
                st.session_state["last_showdown_solver_stdout"] = proc.stdout
                st.session_state["last_showdown_solver_stderr"] = proc.stderr
                st.session_state["last_showdown_solver_returncode"] = proc.returncode
            if proc.returncode != 0:
                st.error("Single-Game lineup generation could not be completed.")
                if workspace_mode == "Admin":
                    with st.expander("Single-Game solver diagnostics", expanded=True):
                        st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))
                return
            if not output_path.exists():
                st.error("Single-Game lineup generation did not produce a validated output file.")
                return
            long_df = pd.read_csv(output_path)
            display_df = _wfs_showdown_display(long_df)
            if len(display_df) != int(lineups):
                st.error("Single-Game lineup count failed validation.")
                return
            st.session_state["showdown_lineups_long_df"] = long_df
            st.session_state["showdown_lineups_display_df"] = display_df
            st.session_state["showdown_last_slate"] = str(selected_slate)
            st.session_state["showdown_last_lineup_count"] = int(lineups)
            st.session_state[
                "showdown_state_signature"
            ] = showdown_state_signature
            st.success(f"Generated {len(display_df)} Single-Game lineup" + ("s." if len(display_df) != 1 else "."))
        except subprocess.TimeoutExpired:
            st.error("Single-Game lineup generation timed out and was stopped.")
        except Exception as exc:
            if workspace_mode == "Admin":
                st.error(
                    f"Single-Game lineup generation failed closed: {exc}"
                )
            else:
                st.error(
                    "Single-Game lineup generation could not be completed. "
                    "Please try again."
                )
        finally:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

    display_df = st.session_state.get("showdown_lineups_display_df")
    long_df = st.session_state.get("showdown_lineups_long_df")
    last_slate = st.session_state.get("showdown_last_slate")
    last_lineup_count = st.session_state.get(
        "showdown_last_lineup_count"
    )
    stored_state_signature = st.session_state.get(
        "showdown_state_signature"
    )

    showdown_result_current = (
        isinstance(display_df, pd.DataFrame)
        and not display_df.empty
        and last_slate == str(selected_slate)
        and last_lineup_count == int(lineups)
        and stored_state_signature
        == showdown_state_signature
    )

    if showdown_result_current:
        st.markdown('<div class="qt-section" style="margin-top:1.2rem;">5. Lineup Command Center</div>', unsafe_allow_html=True)
        st.caption("Review your generated Single-Game lineups and download when ready.")
        high_projection = pd.to_numeric(display_df["Projection"], errors="coerce").max()
        max_salary = pd.to_numeric(display_df["Salary"], errors="coerce").max()
        st.markdown(
            f"""
            <div class="qt-card" style="padding:.85rem 1rem;margin:.5rem 0 1rem 0;">
              <div style="display:flex;gap:.65rem;flex-wrap:wrap;align-items:center;font-size:.86rem;font-weight:800;color:#e2e8f0;">
                <span>{len(display_df)} Lineup{"s" if len(display_df) != 1 else ""}</span><span>•</span><span>Roster PASS</span>
                <span>•</span><span>High {high_projection:.2f}</span><span>•</span><span>Max ${max_salary:,.0f}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lineup_tab, exposure_tab = st.tabs(
            ["🏟 Lineups", "📊 Exposures"]
        )

        with lineup_tab:
            render_public_showdown_lineup_cards(
                display_df,
                long_df,
            )

        with exposure_tab:
            if isinstance(long_df, pd.DataFrame) and not long_df.empty:
                exposure_source = long_df.copy()

                exposure_source["player"] = (
                    exposure_source["player"]
                    .fillna("")
                    .astype(str)
                )

                exposure_source["role"] = (
                    exposure_source["role"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                )

                total_lineups = max(
                    1,
                    int(display_df["Lineup"].nunique()),
                )

                overall = (
                    exposure_source
                    .groupby("player", sort=False)["lineup"]
                    .nunique()
                    .rename("Lineups")
                )

                mvp = (
                    exposure_source[
                        exposure_source["role"].eq("MVP")
                    ]
                    .groupby("player", sort=False)["lineup"]
                    .nunique()
                    .rename("MVP")
                )

                exposure = (
                    pd.concat([overall, mvp], axis=1)
                    .fillna(0)
                    .reset_index()
                    .rename(columns={"player": "Player"})
                )

                exposure["Lineups"] = (
                    pd.to_numeric(
                        exposure["Lineups"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                )

                exposure["MVP"] = (
                    pd.to_numeric(
                        exposure["MVP"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                )

                exposure["Exposure"] = (
                    exposure["Lineups"] / total_lineups * 100
                ).map(lambda x: f"{x:.0f}%")

                exposure["MVP Exposure"] = (
                    exposure["MVP"] / total_lineups * 100
                ).map(lambda x: f"{x:.0f}%")

                exposure = exposure[
                    [
                        "Player",
                        "Lineups",
                        "Exposure",
                        "MVP",
                        "MVP Exposure",
                    ]
                ].sort_values(
                    ["Lineups", "MVP", "Player"],
                    ascending=[False, False, True],
                    kind="mergesort",
                )

                st.dataframe(
                    exposure,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No exposure data available.")
        st.download_button(
            "Download Single-Game lineup CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"nfl_single_game_lineups_{safe_slug(selected_slate)}.csv",
            mime="text/csv",
            key=f"wfs_showdown_download_{safe_slug(selected_slate)}",
        )
        st.caption("Name-based Single-Game CSV for review and auditing.")
        if workspace_mode == "Admin" and isinstance(long_df, pd.DataFrame):
            with st.expander("Admin Single-Game player detail"):
                st.dataframe(long_df, width="stretch", hide_index=True)

def output_paths(slate_name: str) -> dict[str, Path]:
    slug = safe_slug(slate_name)
    base = Path(CSV_DIR)
    return {
        "lineups": base / f"fanduel_gpp_ui_portfolio_{slug}.csv",
        "players": base / f"fanduel_gpp_ui_portfolio_players_{slug}.csv",
        "exposure": base / f"fanduel_gpp_ui_exposure_{slug}.csv",
        "audit": base / f"audit_fanduel_gpp_ui_portfolio_{slug}.csv",
    }


def read_outputs(slate_name: str):
    paths = output_paths(slate_name)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            "Solver completed but expected output files are missing:\n"
            + "\n".join(missing)
        )
    return (
        pd.read_csv(paths["lineups"]),
        pd.read_csv(paths["players"]),
        pd.read_csv(paths["exposure"]),
        pd.read_csv(paths["audit"]),
    )



def _wfs_hash_file(path: Path) -> str:
    """Small deterministic fingerprint for code/data-version invalidation."""
    path = Path(path)
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wfs_pool_fingerprint(df: pd.DataFrame) -> str:
    """
    Fingerprint the exact solver-ready slate visible to the UI.

    This intentionally includes every solver-table column so any salary,
    projection, objective signal, eligibility, team, game, or identity change
    invalidates the cached production portfolio.
    """
    if df is None or df.empty:
        return hashlib.sha256(b"EMPTY_POOL").hexdigest()

    work = df.copy()
    cols = sorted(str(c) for c in work.columns)
    work = work.reindex(columns=cols)

    if "_ui_key" in work.columns:
        work = work.sort_values("_ui_key", kind="mergesort")
    elif "player_solver" in work.columns:
        sort_cols = [
            c for c in ["player_solver", "team_solver", "solver_position", "salary_solver"]
            if c in work.columns
        ]
        if sort_cols:
            work = work.sort_values(sort_cols, kind="mergesort")

    work = work.reset_index(drop=True)
    payload = work.to_csv(index=False, na_rep="<NA>", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _classic_authority_transaction(selected_slate: str, displayed_pool: pd.DataFrame):
    """Keep one updater generation from fresh inputs through result acceptance.

    Display-time data is not an action snapshot. Re-read without Streamlit's
    cache and reject stale controls before fingerprinting or launching a solver.
    The parent owns this lock while subprocess.run waits; inner bridge reader
    locks use separate descriptors and cannot release this outer reader lock.
    """
    try:
        lock = (APP_DIR / "nfl_updater.lock").open("rb")
    except OSError as exc:
        _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
        st.stop()

    with lock:
        try:
            deadline = time.monotonic() + 1200.0
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.25)
        except OSError as exc:
            _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
            st.stop()

        try:
            try:
                with sqlite3.connect(
                    f"{Path(DATABASE_PATH).resolve().as_uri()}?mode=ro", uri=True,
                ) as conn:
                    publication = pd.read_sql_query(f'SELECT * FROM "{SOLVER_TABLE}"', conn)
                fresh_pool = slate_pool(current_classic_publication(publication), selected_slate)
                if fresh_pool.empty or _wfs_pool_fingerprint(fresh_pool) != _wfs_pool_fingerprint(displayed_pool):
                    raise RuntimeError("Classic selection changed since display")
            except Exception as exc:
                _wfs_report_classic_exception(exc, mode, category="AUTHORITY")
                st.stop()
            yield fresh_pool
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _wfs_portfolio_cache_key(
    slate_name: str,
    selected_pool: pd.DataFrame,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: bool,
    projection_loss_pct: float,
    max_player_exposure: float,
    max_qb_exposure: float,
    max_dst_exposure: float,
    lock_keys: list[str],
    exclude_keys: list[str],
) -> str:
    """
    Deterministic strategy fingerprint for a solver portfolio.

    Portfolio depth is intentionally NOT part of this fingerprint.  A single
    exact strategy family grows append-only from 5 -> 10 -> 15 -> 20.
    """
    from stage24_solver_attachment import classic_publication_fingerprint

    payload = {
        "cache_schema": 2,
        "classic_projection_authority": classic_publication_fingerprint(),
        "slate": str(slate_name),
        "solver_sha256": _wfs_hash_file(UI_SOLVER),
        "pool_sha256": _wfs_pool_fingerprint(selected_pool),
        "salary_floor": int(salary_floor),
        "max_overlap": int(max_overlap),
        "max_team": int(max_team),
        "qb_stack": int(qb_stack),
        "bring_back": bool(bring_back),
        "projection_loss_pct": float(projection_loss_pct),
        "max_player_exposure": float(max_player_exposure),
        "max_qb_exposure": float(max_qb_exposure),
        "max_dst_exposure": float(max_dst_exposure),
        "lock_keys": sorted(str(x) for x in lock_keys),
        "exclude_keys": sorted(str(x) for x in exclude_keys),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _wfs_portfolio_cache_paths(cache_key: str) -> dict[str, Path]:
    base = PORTFOLIO_CACHE_DIR / cache_key
    return {
        "base": base,
        "lineups": base / "lineups.csv",
        "players": base / "players.csv",
        "exposure": base / "exposure.csv",
        "audit": base / "audit.csv",
        "meta": base / "meta.json",
    }


def _wfs_read_cache_meta(cache_key: str) -> dict:
    paths = _wfs_portfolio_cache_paths(cache_key)
    if not paths["meta"].exists():
        return {}
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _wfs_read_cached_portfolio(
    cache_key: str,
    expected_size: int | None = None,
):
    paths = _wfs_portfolio_cache_paths(cache_key)
    required = ["lineups", "players", "exposure", "audit", "meta"]
    if any(not paths[name].exists() for name in required):
        return None

    try:
        meta = _wfs_read_cache_meta(cache_key)
        actual_size = int(meta.get("portfolio_size", 0))
        if actual_size < 1:
            return None
        if expected_size is not None and actual_size != int(expected_size):
            return None

        lineups_df = pd.read_csv(paths["lineups"])
        players_df = pd.read_csv(paths["players"])
        exposure_df = pd.read_csv(paths["exposure"])
        audit_df = pd.read_csv(paths["audit"])

        if len(lineups_df) != actual_size:
            return None
        if len(players_df) != actual_size * len(ROSTER_SLOTS):
            return None
        # Fail closed: a cache is valid only when solver audit authority is
        # present, non-empty, and every recorded audit is PASS.
        if "audit" not in audit_df.columns or audit_df.empty:
            return None
        if not audit_df["audit"].astype(str).eq("PASS").all():
            return None

        # Canonical nine-player sets must be unique across the shared portfolio.
        signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
        if signatures.duplicated().any():
            return None

        return lineups_df, players_df, exposure_df, audit_df
    except Exception:
        return None


def _wfs_write_cached_portfolio(
    cache_key: str,
    slate_name: str,
    lineups_df: pd.DataFrame,
    players_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    expected_size: int = PORTFOLIO_CACHE_SIZE,
    solver_profile: dict | None = None,
) -> None:
    expected_size = int(expected_size)
    if expected_size < 1:
        raise RuntimeError("Cache write blocked: invalid expected portfolio size.")

    if len(lineups_df) != expected_size:
        raise RuntimeError(
            f"Cache write blocked: expected {expected_size} "
            f"lineups, received {len(lineups_df)}."
        )
    if len(players_df) != expected_size * len(ROSTER_SLOTS):
        raise RuntimeError(
            "Cache write blocked: players.csv does not contain exactly nine "
            "rows per lineup."
        )

    signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
    if signatures.duplicated().any():
        raise RuntimeError(
            "Cache write blocked: duplicate canonical nine-player lineup detected."
        )

    # Fail closed on missing audit authority as well as explicit failures.
    if "audit" not in audit_df.columns or audit_df.empty:
        raise RuntimeError("Cache write blocked: solver audit authority is missing.")
    if not audit_df["audit"].astype(str).eq("PASS").all():
        raise RuntimeError("Cache write blocked: solver audit did not fully PASS.")

    paths = _wfs_portfolio_cache_paths(cache_key)
    paths["base"].mkdir(parents=True, exist_ok=True)

    # Meta is the commit marker. Readers reject mixed generations because the
    # lineups/player counts must agree with the committed portfolio_size.
    frames = {
        "lineups": lineups_df,
        "players": players_df,
        "exposure": exposure_df,
        "audit": audit_df,
    }
    tmp_paths = {}
    for name, frame in frames.items():
        tmp = paths["base"] / f".{name}.csv.tmp"
        frame.to_csv(tmp, index=False)
        tmp_paths[name] = tmp

    meta = {
        "cache_schema": 2,
        "cache_key": cache_key,
        "slate": str(slate_name),
        "portfolio_size": expected_size,
        "solver_sha256": _wfs_hash_file(UI_SOLVER),
        "solver_profile": solver_profile or {},
    }
    meta_tmp = paths["base"] / ".meta.json.tmp"
    meta_tmp.write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for name, tmp in tmp_paths.items():
        tmp.replace(paths[name])
    meta_tmp.replace(paths["meta"])


def _wfs_merge_portfolio_extension(
    existing,
    extension,
    target_size: int,
):
    old_lineups, old_players, _old_exposure, old_audit = existing
    new_lineups, new_players, new_exposure, new_audit = extension

    old_size = len(old_lineups)
    expected_new = int(target_size) - old_size
    if expected_new < 1 or len(new_lineups) != expected_new:
        raise RuntimeError(
            f"Portfolio extension expected {expected_new} new lineups, "
            f"received {len(new_lineups)}."
        )

    expected_numbers = list(range(old_size + 1, int(target_size) + 1))
    actual_numbers = (
        pd.to_numeric(new_lineups["lineup"], errors="raise").astype(int).tolist()
    )
    if actual_numbers != expected_numbers:
        raise RuntimeError(
            "Portfolio extension returned unexpected lineup numbering."
        )

    lineups_df = pd.concat([old_lineups, new_lineups], ignore_index=True)
    players_df = pd.concat([old_players, new_players], ignore_index=True)
    audit_df = pd.concat([old_audit, new_audit], ignore_index=True)

    if len(lineups_df) != int(target_size):
        raise RuntimeError("Portfolio extension merge produced wrong total depth.")

    signatures = lineups_df.apply(_wfs_lineup_signature, axis=1)
    if signatures.duplicated().any():
        raise RuntimeError(
            "Portfolio extension produced a duplicate canonical nine-player set."
        )

    return lineups_df, players_df, new_exposure, audit_df

def _wfs_lineup_signature(row) -> str:
    """
    Exact player-set signature. Slot ordering does not make the same nine-player
    lineup "new".
    """
    players = []
    for slot in ROSTER_SLOTS:
        if slot in row.index:
            players.append(normalize_name(row.get(slot)))
    raw = "|".join(sorted(x for x in players if x))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _wfs_take_next_unseen(
    portfolio_df: pd.DataFrame,
    requested: int,
    seen_signatures,
):
    """
    Serve the strongest unseen lineups in production order.

    Seen history is permanent for the authenticated user and exact cache key.
    Exhaustion fails closed. A previously served canonical nine-player set
    is never returned as a new lineup.
    """
    if portfolio_df is None or portfolio_df.empty:
        raise RuntimeError(
            "No unique lineup inventory is available."
        )

    requested = max(
        1,
        min(int(requested), len(portfolio_df)),
    )
    seen = {
        str(value).strip()
        for value in (seen_signatures or [])
        if str(value).strip()
    }

    rows = []
    selected_sigs = []

    for idx, row in portfolio_df.iterrows():
        sig = _wfs_lineup_signature(row)

        if sig in seen:
            continue

        rows.append(idx)
        selected_sigs.append(sig)

        if len(rows) >= requested:
            break

    if len(rows) != requested:
        raise RuntimeError(
            "No additional unique lineup is available for "
            "these slate settings and player controls."
        )

    seen.update(selected_sigs)

    out = (
        portfolio_df
        .loc[rows]
        .copy()
        .reset_index(drop=True)
    )

    if "lineup" in out.columns:
        out["portfolio_lineup"] = out["lineup"]
        out["lineup"] = range(
            1,
            len(out) + 1,
        )

    return out, sorted(seen), False


def _wfs_subset_detail_frame(df: pd.DataFrame, portfolio_lineup_ids) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if "lineup" not in df.columns:
        return df.copy()
    ids = {str(x) for x in portfolio_lineup_ids}
    mask = df["lineup"].astype(str).isin(ids)
    return df.loc[mask].copy().reset_index(drop=True)


def build_command(
    slate_name: str,
    lineups: int,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: bool,
    projection_loss_pct: float,
    max_player_exposure: float,
    max_qb_exposure: float,
    max_dst_exposure: float,
    lock_keys: list[str],
    exclude_keys: list[str],
    resume_players: str = "",
) -> list[str]:
    cmd = [
        sys.executable,
        str(UI_SOLVER),
        "--slate", slate_name,
        "--lineups", str(lineups),
        "--salary-floor", str(salary_floor),
        "--max-overlap", str(max_overlap),
        "--max-team", str(max_team),
        "--qb-stack", str(qb_stack),
        "--bring-back", "1" if bring_back else "0",
        "--projection-loss-pct", str(projection_loss_pct),
        "--max-player-exposure", str(max_player_exposure),
        "--max-qb-exposure", str(max_qb_exposure),
        "--max-dst-exposure", str(max_dst_exposure),
    ]

    if str(resume_players).strip():
        cmd.extend(["--resume-players", str(resume_players)])

    for key in lock_keys:
        cmd.extend(["--lock-key", key])
    for key in exclude_keys:
        cmd.extend(["--exclude-key", key])

    return cmd


# QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
#
# Deterministic, order-independent fingerprint of the Classic generation
# inputs actually passed to build_command. This is UI/session metadata
# only: it never touches solver inputs, portfolio cache keys, or output
# files. It exists solely so a previously displayed portfolio can be
# compared against the controls currently on screen and flagged as
# outdated when they no longer match.
def _wfs_classic_freshness_signature(
    slate_name: str,
    lineups: int,
    salary_floor: int,
    max_overlap: int,
    max_team: int,
    qb_stack: int,
    bring_back: bool,
    projection_loss_pct: float,
    max_player_exposure: float,
    max_qb_exposure: float,
    max_dst_exposure: float,
    lock_keys: list[str],
    exclude_keys: list[str],
) -> str:
    payload = {
        "slate": str(slate_name),
        "lineups": int(lineups),
        "salary_floor": int(salary_floor),
        "max_overlap": int(max_overlap),
        "max_team": int(max_team),
        "qb_stack": int(qb_stack),
        "bring_back": bool(bring_back),
        "projection_loss_pct": float(projection_loss_pct),
        "max_player_exposure": float(max_player_exposure),
        "max_qb_exposure": float(max_qb_exposure),
        "max_dst_exposure": float(max_dst_exposure),
        "lock_keys": sorted({str(x) for x in lock_keys}),
        "exclude_keys": sorted({str(x) for x in exclude_keys}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def _wfs_public_profiles(
    max_overlap: int,
    qb_stack: int,
    bring_back: bool,
    max_player_exposure: float,
    max_qb_exposure: float,
    max_dst_exposure: float,
) -> list[dict]:
    preferred = {
        "name": "PREFERRED",
        "max_overlap": int(max_overlap),
        "qb_stack": int(qb_stack),
        "bring_back": bool(bring_back),
        "max_player_exposure": float(max_player_exposure),
        "max_qb_exposure": float(max_qb_exposure),
        "max_dst_exposure": float(max_dst_exposure),
    }
    candidates = [
        preferred,
        {
            **preferred,
            "name": "EXPOSURE_RELIEF",
            "max_player_exposure": 1.0,
            "max_qb_exposure": 1.0,
            "max_dst_exposure": 1.0,
        },
        {
            **preferred,
            "name": "OVERLAP_RELIEF",
            "max_overlap": 8,
            "max_player_exposure": 1.0,
            "max_qb_exposure": 1.0,
            "max_dst_exposure": 1.0,
        },
        {
            **preferred,
            "name": "QB_STACK_RELIEF",
            "max_overlap": 8,
            "qb_stack": 1,
            "max_player_exposure": 1.0,
            "max_qb_exposure": 1.0,
            "max_dst_exposure": 1.0,
        },
        {
            **preferred,
            "name": "FULL_SOFT_RELIEF",
            "max_overlap": 8,
            "qb_stack": 1,
            "bring_back": False,
            "max_player_exposure": 1.0,
            "max_qb_exposure": 1.0,
            "max_dst_exposure": 1.0,
        },
    ]
    out = []
    seen = set()
    for profile in candidates:
        signature = (
            int(profile["max_overlap"]),
            int(profile["qb_stack"]),
            bool(profile["bring_back"]),
            float(profile["max_player_exposure"]),
            float(profile["max_qb_exposure"]),
            float(profile["max_dst_exposure"]),
        )
        if signature not in seen:
            seen.add(signature)
            out.append(profile)
    return out


def _wfs_run_public_solver_target(
    *,
    selected_slate: str,
    target_size: int,
    salary_floor: int,
    max_team: int,
    projection_loss_pct: float,
    lock_keys: list[str],
    exclude_keys: list[str],
    profiles: list[dict],
    existing=None,
    resume_players: str = "",
    display_count: int = 1,
):
    existing_size = 0 if existing is None else len(existing[0])
    expected_new = int(target_size) - existing_size
    if expected_new < 1:
        raise RuntimeError("Public solver target must exceed existing cache depth.")

    successful_profile = None
    attempt_log = []
    accepted = None

    spinner_text = (
        "Generating your lineup…"
        if int(display_count) == 1
        else "Generating your lineups…"
    )

    with st.spinner(spinner_text):
        for profile in profiles:
            cmd = build_command(
                slate_name=selected_slate,
                lineups=int(target_size),
                salary_floor=int(salary_floor),
                max_overlap=int(profile["max_overlap"]),
                max_team=int(max_team),
                qb_stack=int(profile["qb_stack"]),
                bring_back=bool(profile["bring_back"]),
                projection_loss_pct=float(projection_loss_pct),
                max_player_exposure=float(profile["max_player_exposure"]),
                max_qb_exposure=float(profile["max_qb_exposure"]),
                max_dst_exposure=float(profile["max_dst_exposure"]),
                lock_keys=lock_keys,
                exclude_keys=exclude_keys,
                resume_players=resume_players,
            )

            require_current_classic_action(selected_slate)
            proc = subprocess.run(
                cmd,
                cwd=APP_DIR,
                capture_output=True,
                text=True,
            )
            record = {
                "profile": str(profile["name"]),
                "returncode": int(proc.returncode),
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "generated": 0,
                "accepted": False,
            }

            if proc.returncode == 0:
                try:
                    candidate = read_outputs(selected_slate)
                    candidate_lineups = candidate[0]
                    candidate_audit = candidate[3]
                    record["generated"] = len(candidate_lineups)

                    full_count_ok = len(candidate_lineups) == expected_new
                    audit_ok = (
                        not candidate_audit.empty
                        and "audit" in candidate_audit.columns
                        and candidate_audit["audit"].astype(str).eq("PASS").all()
                    )
                    if not audit_ok:
                        record["validation_failure"] = "AUDIT_FAILED"
                    elif not full_count_ok:
                        record["validation_failure"] = "COUNT_MISMATCH"
                    if full_count_ok and audit_ok:
                        accepted = (
                            candidate
                            if existing is None
                            else _wfs_merge_portfolio_extension(
                                existing, candidate, int(target_size)
                            )
                        )
                        successful_profile = dict(profile)
                        record["accepted"] = True
                except Exception as exc:
                    record["stderr"] += (
                        "\nOUTPUT_VALIDATION_ERROR: "
                        f"{type(exc).__name__}: {exc}"
                    )

            attempt_log.append(record)
            if not record["accepted"]:
                logging.getLogger(__name__).error(
                    "Classic generation attempt failed: %s", record,
                )
            if accepted is not None:
                break

    st.session_state["last_solver_attempts"] = attempt_log
    st.session_state["last_solver_stdout"] = attempt_log[-1]["stdout"] if attempt_log else ""
    st.session_state["last_solver_stderr"] = attempt_log[-1]["stderr"] if attempt_log else ""
    st.session_state["last_returncode"] = attempt_log[-1]["returncode"] if attempt_log else -1
    st.session_state["last_slate"] = selected_slate
    st.session_state["last_public_solver_profile"] = (
        successful_profile["name"] if successful_profile else ""
    )
    st.session_state["last_classic_failure_category"] = (
        "" if accepted is not None else _wfs_classic_failure_category(
            attempt_log, bool(lock_keys or exclude_keys),
        )
    )
    return accepted, successful_profile


def render_public_lineup_cards(
    lineups_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> None:
    """Render compact responsive public lineup cards without changing solver data."""

    import html

    if lineups_df is None or lineups_df.empty:
        st.info("No lineups to display.")
        return

    slot_order = {
        "QB": 0,
        "RB1": 1,
        "RB2": 2,
        "WR1": 3,
        "WR2": 4,
        "WR3": 5,
        "TE": 6,
        "FLEX": 7,
        "DST": 8,
    }

    cards = []

    for _, lineup_row in lineups_df.iterrows():
        display_id = int(lineup_row.get("lineup", len(cards) + 1))

        portfolio_id = lineup_row.get("portfolio_lineup", display_id)
        detail = players_df[
            players_df["lineup"].astype(str) == str(portfolio_id)
        ].copy()

        if detail.empty:
            raise RuntimeError(
                f"Public lineup card detail missing for portfolio lineup {portfolio_id}."
            )

        if len(detail) != 9:
            raise RuntimeError(
                f"Public lineup card expected 9 players for portfolio lineup "
                f"{portfolio_id}; found {len(detail)}."
            )

        detail["_slot_order"] = (
            detail["slot"].astype(str).map(slot_order).fillna(99)
        )
        detail = detail.sort_values("_slot_order")

        salary = pd.to_numeric(
            pd.Series([lineup_row.get("salary")]), errors="coerce"
        ).iloc[0]
        projection = pd.to_numeric(
            pd.Series([lineup_row.get("projection")]), errors="coerce"
        ).iloc[0]

        salary_text = (
            f"${salary:,.0f}" if pd.notna(salary) else "—"
        )
        projection_text = (
            f"{projection:.2f} PTS" if pd.notna(projection) else "—"
        )

        rows = []

        for _, player_row in detail.iterrows():
            raw_slot = str(player_row.get("slot", "")).upper()
            display_slot = raw_slot

            if raw_slot in {"RB1", "RB2"}:
                display_slot = "RB"
            elif raw_slot in {"WR1", "WR2", "WR3"}:
                display_slot = "WR"

            player = html.escape(str(player_row.get("player", "—")))
            raw_team = str(player_row.get("team", "") or "").strip()
            canonical_team = _dc_team_code(raw_team)
            team = html.escape(canonical_team or raw_team or "—")
            opponent = html.escape(str(player_row.get("opponent", "—")))

            # WFS_PUBLIC_LINEUP_TEAM_LOGO_V1
            # Presentation only: reuse the NFL Data Center's local team-logo
            # resolver. No solver, lineup identity, cache, or scoring changes.
            team_logo = _dc_team_logo_html(
                canonical_team,
                "wfs-lc-team-logo",
            )

            player_salary = pd.to_numeric(
                pd.Series([player_row.get("salary")]), errors="coerce"
            ).iloc[0]
            player_projection = pd.to_numeric(
                pd.Series([player_row.get("projection")]), errors="coerce"
            ).iloc[0]

            player_salary_text = (
                f"${player_salary / 1000:.1f}K"
                if pd.notna(player_salary)
                else "—"
            )
            player_projection_text = (
                f"{player_projection:.2f}"
                if pd.notna(player_projection)
                else "—"
            )

            rows.append(
                f"""
                <div class="wfs-lc-row">
                    <div class="wfs-lc-pos">{html.escape(display_slot)}</div>
                    <div class="wfs-lc-player">
                        {team_logo}
                        <span class="wfs-lc-player-name">{player}</span>
                    </div>
                    <div class="wfs-lc-team">{team}</div>
                    <div class="wfs-lc-opp">{opponent}</div>
                    <div class="wfs-lc-sal">{player_salary_text}</div>
                    <div class="wfs-lc-fpts">{player_projection_text}</div>
                </div>
                """
            )

        cards.append(
            f"""
            <div class="wfs-lineup-card">
                <div class="wfs-lc-head">
                    <div class="wfs-lc-number">#{display_id}</div>
                    <div class="wfs-lc-proj">{projection_text}</div>
                    <div class="wfs-lc-salary">{salary_text}</div>
                </div>

                <div class="wfs-lc-columns">
                    <div>POS</div>
                    <div>PLAYER</div>
                    <div>TEAM</div>
                    <div>OPP</div>
                    <div>SAL</div>
                    <div>FPTS</div>
                </div>

                {''.join(rows)}
            </div>
            """
        )

    st.markdown(
        """
        <style>
        .wfs-lineup-grid {
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
            gap:14px;
            width:100%;
            margin:.35rem 0 1rem 0;
        }

        .wfs-lineup-card {
            min-width:0;
            overflow:hidden;
            border:1px solid rgba(148,163,184,.22);
            border-radius:12px;
            background:rgba(15,23,42,.72);
            box-shadow:0 8px 24px rgba(0,0,0,.14);
        }

        .wfs-lc-head {
            display:grid;
            grid-template-columns:1fr auto auto;
            gap:14px;
            align-items:center;
            padding:11px 13px;
            border-bottom:1px solid rgba(148,163,184,.20);
        }

        .wfs-lc-number {
            font-size:1rem;
            font-weight:900;
            color:#f8fafc;
        }

        .wfs-lc-proj {
            font-size:.88rem;
            font-weight:900;
            color:#e2e8f0;
            white-space:nowrap;
        }

        .wfs-lc-salary {
            font-size:.88rem;
            font-weight:900;
            color:#f8fafc;
            white-space:nowrap;
        }

        .wfs-lc-columns,
        .wfs-lc-row {
            display:grid;
            grid-template-columns:42px minmax(130px,1fr) 45px 45px 55px 52px;
            gap:5px;
            align-items:center;
        }

        .wfs-lc-columns {
            padding:7px 10px;
            font-size:.62rem;
            font-weight:900;
            letter-spacing:.04em;
            color:#94a3b8;
            border-bottom:1px solid rgba(148,163,184,.16);
        }

        .wfs-lc-row {
            padding:8px 10px;
            min-height:35px;
            border-bottom:1px solid rgba(148,163,184,.10);
            font-size:.76rem;
            color:#e2e8f0;
        }

        .wfs-lc-row:last-child {
            border-bottom:0;
        }

        .wfs-lc-pos {
            font-weight:900;
            color:#cbd5e1;
        }

        .wfs-lc-player {
            min-width:0;
            overflow:hidden;
            display:flex;
            align-items:center;
            gap:6px;
            white-space:nowrap;
            font-weight:800;
            color:#f8fafc;
        }

        .wfs-lc-player-name {
            min-width:0;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }

        .wfs-lc-team-logo {
            width:22px;
            height:22px;
            object-fit:contain;
            flex:0 0 22px;
        }

        .wfs-lc-team,
        .wfs-lc-opp {
            font-weight:700;
            color:#cbd5e1;
        }

        .wfs-lc-sal,
        .wfs-lc-fpts {
            text-align:right;
            font-variant-numeric:tabular-nums;
            white-space:nowrap;
        }

        .wfs-lc-fpts {
            font-weight:900;
            color:#f8fafc;
        }

        @media (max-width: 640px) {
            .wfs-lineup-grid {
                grid-template-columns:minmax(0,1fr);
                gap:10px;
            }

            .wfs-lc-head {
                gap:8px;
                padding:10px 9px;
            }

            .wfs-lc-columns,
            .wfs-lc-row {
                grid-template-columns:34px minmax(105px,1fr) 34px 34px 48px 43px;
                gap:3px;
                padding-left:7px;
                padding-right:7px;
            }

            .wfs-lc-columns {
                font-size:.54rem;
            }

            .wfs-lc-row {
                font-size:.68rem;
            }

            .wfs-lc-player {
                gap:4px;
            }

            .wfs-lc-team-logo {
                width:20px;
                height:20px;
                flex-basis:20px;
            }

            .wfs-lc-proj,
            .wfs-lc-salary {
                font-size:.78rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    card_html = '<div class="wfs-lineup-grid">' + "".join(cards) + "</div>"
    card_html = " ".join(card_html.split())

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


def lineup_display(lineups_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "lineup",
        *ROSTER_SLOTS,
        "salary",
        "projection",
        "best_feasible_projection",
        "projection_loss_actual_pct",
        "gpp_rank_total",
        "qb",
        "stack_type",
        "bring_back_players",
    ]
    cols = [c for c in cols if c in lineups_df.columns]
    out = lineups_df[cols].copy()
    if "projection_loss_actual_pct" in out.columns:
        out["projection_loss_actual_pct"] = (
            pd.to_numeric(out["projection_loss_actual_pct"], errors="coerce") * 100
        ).round(2)
        out = out.rename(
            columns={"projection_loss_actual_pct": "projection_loss_pct"}
        )
    for c in ["projection", "best_feasible_projection"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    return out


def generic_lineup_csv(lineups_df: pd.DataFrame) -> bytes:
    """
    Clean lineup export using player names by slot.

    This is intentionally NOT labeled FanDuel-upload-ready because the
    solver-ready schema may or may not expose FanDuel contest player IDs.
    """
    cols = [c for c in ["lineup", *ROSTER_SLOTS] if c in lineups_df.columns]
    return lineups_df[cols].to_csv(index=False).encode("utf-8")



def parse_fanduel_template(template_bytes: bytes) -> dict:
    """
    Parse an official FanDuel NFL lineup-upload template.

    Supports both known FanDuel layouts:

    1. Legacy entries layout:
       entry_id, contest_id, contest_name, entry_fee,
       QB, RB, RB, WR, WR, WR, TE, FLEX, DEF

    2. Current lineup-upload layout:
       QB, RB, RB, WR, WR, WR, TE, FLEX, DEF

       The current lineup rows appear directly beneath those nine roster
       columns and the embedded FanDuel player table appears farther right /
       farther down in the same CSV.

    Player identity remains exact. No fuzzy matching is introduced.
    """
    decoded = template_bytes.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(decoded)))

    if not rows:
        raise ValueError("FanDuel template is empty.")

    # ------------------------------------------------------------
    # Locate FanDuel embedded player table first.
    # ------------------------------------------------------------

    pool_header_row = None
    pool_start_col = None

    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            if str(value).strip() == "Player ID + Player Name":
                pool_header_row = r_idx
                pool_start_col = c_idx
                break

        if pool_header_row is not None:
            break

    if pool_header_row is None or pool_start_col is None:
        raise ValueError(
            "Could not locate FanDuel's embedded player table."
        )

    pool_headers = rows[pool_header_row][pool_start_col:]

    header_index = {
        str(name).strip(): pool_start_col + i
        for i, name in enumerate(pool_headers)
        if str(name).strip()
    }

    required = [
        "Id",
        "Position",
        "Nickname",
        "Team",
    ]

    missing = [
        name
        for name in required
        if name not in header_index
    ]

    if missing:
        raise ValueError(
            "FanDuel player table is missing: "
            + ", ".join(missing)
        )

    def read_cell(row, name):
        idx = header_index.get(name)

        if idx is None or idx >= len(row):
            return ""

        return str(row[idx]).strip()

    # ------------------------------------------------------------
    # Parse exact FanDuel player pool.
    # ------------------------------------------------------------

    player_rows = []

    for row in rows[pool_header_row + 1:]:

        fd_id = read_cell(row, "Id")
        raw_position = read_cell(
            row,
            "Position",
        ).upper()

        nickname = read_cell(
            row,
            "Nickname",
        )

        team = normalize_team(
            read_cell(
                row,
                "Team",
            )
        )

        if not fd_id or not raw_position:
            continue

        # FanDuel may express defense as D or DEF.
        # WFS internally uses DST, but intersect_with_fanduel_contest()
        # intentionally expects FanDuel defense as position D.
        if raw_position in {"D", "DEF", "DST"}:
            position = "D"
        else:
            position = raw_position

        salary_raw = read_cell(
            row,
            "Salary",
        )

        try:
            salary = (
                int(float(salary_raw))
                if salary_raw
                else None
            )
        except Exception:
            salary = None

        roster_position = read_cell(
            row,
            "Roster Position",
        )

        player_rows.append(
            {
                "fd_id": fd_id,
                "nickname": nickname,
                "name_key": normalize_fanduel_name(nickname),
                "position": position,
                "team": team,
                "salary": salary,
                "game": read_cell(row, "Game"),
                "opponent": normalize_team(
                    read_cell(
                        row,
                        "Opponent",
                    )
                ),
                "injury_indicator": read_cell(
                    row,
                    "Injury Indicator",
                ),
                "injury_details": read_cell(
                    row,
                    "Injury Details",
                ),
                "roster_position": roster_position,
            }
        )

    if not player_rows:
        raise ValueError(
            "FanDuel player mapping table contains no players."
        )

    fd_players_by_id = {
        str(player["fd_id"]): player
        for player in player_rows
    }

    # ------------------------------------------------------------
    # Detect lineup-header format.
    # ------------------------------------------------------------

    roster_header = [
        "QB",
        "RB",
        "RB",
        "WR",
        "WR",
        "WR",
        "TE",
        "FLEX",
        "DEF",
    ]

    roster_slots = [
        "QB",
        "RB1",
        "RB2",
        "WR1",
        "WR2",
        "WR3",
        "TE",
        "FLEX",
        "DST",
    ]

    legacy_header = [
        "entry_id",
        "contest_id",
        "contest_name",
        "entry_fee",
        "QB",
        "RB",
        "RB",
        "WR",
        "WR",
        "WR",
        "TE",
        "FLEX",
        "DEF",
    ]

    entry_header_row = None
    entry_start_col = None
    layout = None

    # Legacy format.
    if (
        len(rows[0]) >= len(legacy_header)
        and rows[0][:len(legacy_header)] == legacy_header
    ):
        entry_header_row = 0
        entry_start_col = 4
        layout = "legacy_entries"

    # Current FanDuel format.
    else:
        for r_idx, row in enumerate(
            rows[:min(len(rows), pool_header_row + 1)]
        ):
            normalized = [
                str(value).strip().upper()
                for value in row
            ]

            for c_idx in range(
                0,
                max(0, len(normalized) - 8),
            ):
                candidate = normalized[c_idx:c_idx + 9]

                if candidate == roster_header:
                    entry_header_row = r_idx
                    entry_start_col = c_idx
                    layout = "lineup_upload"
                    break

            if entry_header_row is not None:
                break

    if entry_header_row is None or entry_start_col is None:
        raise ValueError(
            "Unexpected FanDuel template header. "
            "Could not locate the NFL "
            "QB/RB/RB/WR/WR/WR/TE/FLEX/DEF lineup columns."
        )

    # ------------------------------------------------------------
    # Parse existing FanDuel lineup rows.
    # ------------------------------------------------------------

    def parse_player_token(value):
        """
        FanDuel accepts either:

            133104-88431:Tyler Shough

        or:

            133104-88431

        Return only the exact FanDuel ID.
        """
        raw = str(value or "").strip()

        if not raw:
            return ""

        if ":" in raw:
            return raw.split(":", 1)[0].strip()

        return raw

    entry_row_indexes = []
    entry_lineups = []

    for r_idx in range(
        entry_header_row + 1,
        pool_header_row,
    ):

        row = rows[r_idx]

        cells = []

        for offset in range(9):
            col = entry_start_col + offset

            value = (
                str(row[col]).strip()
                if col < len(row)
                else ""
            )

            cells.append(value)

        # Ignore FanDuel instruction / spacer rows.
        if not any(cells):
            continue

        # A late-swap current lineup must be complete.
        # Partial rows are not silently interpreted as entries.
        if not all(cells):
            continue

        slot_ids = {}

        for slot, raw_value in zip(
            roster_slots,
            cells,
        ):
            fd_id = parse_player_token(
                raw_value
            )

            if not fd_id:
                raise ValueError(
                    f"FanDuel lineup row {r_idx + 1} contains "
                    f"an empty player in slot {slot}."
                )

            if fd_id not in fd_players_by_id:
                raise ValueError(
                    f"FanDuel lineup row {r_idx + 1} contains "
                    f"unknown player ID {fd_id!r} in slot {slot}."
                )

            player = fd_players_by_id[fd_id]

            player_position = str(
                player["position"]
            ).upper()

            if slot == "QB":
                allowed = {"QB"}

            elif slot in {"RB1", "RB2"}:
                allowed = {"RB"}

            elif slot in {"WR1", "WR2", "WR3"}:
                allowed = {"WR"}

            elif slot == "TE":
                allowed = {"TE"}

            elif slot == "FLEX":
                allowed = {"RB", "WR", "TE"}

            elif slot == "DST":
                allowed = {"D"}

            else:
                raise ValueError(
                    f"Unknown FanDuel roster slot: {slot}"
                )

            if player_position not in allowed:
                raise ValueError(
                    f"FanDuel lineup row {r_idx + 1} has "
                    f"{player['nickname']!r} ({player_position}) "
                    f"in incompatible slot {slot}."
                )

            slot_ids[slot] = fd_id

        if len(set(slot_ids.values())) != 9:
            raise ValueError(
                f"FanDuel lineup row {r_idx + 1} contains "
                "duplicate players."
            )

        entry_row_indexes.append(
            r_idx
        )

        entry_lineups.append(
            {
                "row_index": r_idx,
                "slots": slot_ids,
            }
        )

    return {
        "rows": rows,
        "layout": layout,
        "entry_header_row": entry_header_row,
        "entry_start_col": entry_start_col,
        "entry_row_indexes": entry_row_indexes,
        "entry_lineups": entry_lineups,
        "player_rows": player_rows,
        "player_rows_by_id": fd_players_by_id,
        "pool_header_row": pool_header_row,
        "pool_start_col": pool_start_col,
    }



def resolve_solver_player(player_name: str, selected_pool: pd.DataFrame) -> dict:
    """Resolve a generated name to exactly one solver-ready row."""
    key = normalize_name(player_name)
    matches = selected_pool.loc[
        selected_pool["player_solver"].map(normalize_name).eq(key)
    ].copy()

    if len(matches) != 1:
        raise ValueError(
            f'"{player_name}" resolved to {len(matches)} solver-ready rows.'
        )

    row = matches.iloc[0]
    return {
        "player": str(row["player_solver"]),
        "team": normalize_team(row["team_solver"]),
        "position": str(row["solver_position"]).strip().upper(),
    }


def resolve_fanduel_id(
    solver_player: dict,
    roster_slot: str,
    fd_players: list[dict],
) -> str:
    """
    Exact deterministic WFS -> FanDuel player-ID mapping.

    Offense:
        canonical name + normalized team + compatible position

    Defense:
        normalized team + FanDuel defense position

    No fuzzy matching.
    """
    team = normalize_team(
        solver_player["team"]
    )

    name_key = normalize_fanduel_name(
        solver_player["player"]
    )

    if roster_slot == "DEF":
        candidates = [
            p
            for p in fd_players
            if p["position"] == "D"
            and p["team"] == team
        ]

    else:
        allowed = {
            "QB": {"QB"},
            "RB": {"RB"},
            "WR": {"WR"},
            "TE": {"TE"},
            "FLEX": {"RB", "WR", "TE"},
        }[roster_slot]

        candidates = [
            p
            for p in fd_players
            if p["team"] == team
            and p["position"] in allowed
            and p["name_key"] == name_key
        ]

    if len(candidates) != 1:
        raise ValueError(
            f'{solver_player["player"]} ({team}) '
            f'in {roster_slot}: '
            f'{len(candidates)} exact FanDuel matches.'
        )

    return candidates[0]["fd_id"]




def build_fanduel_upload(
    template_bytes: bytes,
    lineups_df: pd.DataFrame,
    selected_pool: pd.DataFrame,
) -> tuple[bytes, dict]:
    """
    Build a FanDuel-ready NFL upload using the uploaded template.

    Supports both:
        - legacy FanDuel entries templates
        - current FanDuel lineup-upload templates

    Only the nine roster cells of existing FanDuel lineup rows are
    changed. All other template content is preserved.

    Exact FanDuel player-ID mapping is mandatory.
    No fuzzy matching.
    """
    parsed = parse_fanduel_template(
        template_bytes
    )

    rows = [
        list(row)
        for row in parsed["rows"]
    ]

    entry_indexes = list(
        parsed["entry_row_indexes"]
    )

    fd_players = parsed["player_rows"]

    entry_start_col = int(
        parsed["entry_start_col"]
    )

    slot_map = [
        ("QB", "QB"),
        ("RB1", "RB"),
        ("RB2", "RB"),
        ("WR1", "WR"),
        ("WR2", "WR"),
        ("WR3", "WR"),
        ("TE", "TE"),
        ("FLEX", "FLEX"),
        ("DST", "DEF"),
    ]

    usable = min(
        len(entry_indexes),
        len(lineups_df),
    )

    if usable <= 0:
        raise ValueError(
            "No existing FanDuel lineup rows can be updated."
        )

    mapped_lineups = []
    errors = []

    for offset in range(usable):
        lineup = lineups_df.iloc[offset]
        mapped = []

        for lineup_col, fd_slot in slot_map:
            player_name = str(
                lineup[lineup_col]
            ).strip()

            try:
                solver_player = resolve_solver_player(
                    player_name,
                    selected_pool,
                )

                fd_id = resolve_fanduel_id(
                    solver_player,
                    fd_slot,
                    fd_players,
                )

                mapped.append(fd_id)

            except Exception as exc:
                errors.append(
                    f'Lineup '
                    f'{int(lineup.get("lineup", offset + 1))} '
                    f'{fd_slot}: {player_name} — {exc}'
                )

        mapped_lineups.append(mapped)

    if errors:
        raise ValueError(
            "FanDuel export blocked: exact player mapping "
            "did not reach 100% coverage.\n"
            + "\n".join(errors)
        )

    for offset, row_idx in enumerate(
        entry_indexes[:usable]
    ):
        required_len = (
            entry_start_col + 9
        )

        while len(rows[row_idx]) < required_len:
            rows[row_idx].append("")

        rows[row_idx][
            entry_start_col:
            entry_start_col + 9
        ] = mapped_lineups[offset]

    output = io.StringIO(
        newline=""
    )

    csv.writer(
        output,
        lineterminator="\n",
    ).writerows(rows)

    result = output.getvalue().encode(
        "utf-8"
    )

    return result, {
        "layout": parsed["layout"],
        "template_entries": len(entry_indexes),
        "generated_lineups": len(lineups_df),
        "entries_written": usable,
        "mapping_count": usable * 9,
        "entry_start_col": entry_start_col,
    }




def validate_fanduel_upload(
    upload_bytes: bytes,
    entries_written: int,
) -> None:
    """
    Final structural hard gate before exposing a FanDuel upload.

    Validation is performed through the same parser used for import,
    so both legacy and current FanDuel NFL template layouts are
    supported.

    Every written lineup must contain:
        QB/RB/RB/WR/WR/WR/TE/FLEX/DEF

    with nine valid, unique FanDuel player IDs.
    """
    parsed = parse_fanduel_template(
        upload_bytes
    )

    entry_lineups = parsed[
        "entry_lineups"
    ]

    if len(entry_lineups) < entries_written:
        raise ValueError(
            "Generated FanDuel file contains fewer "
            "validated lineup rows than expected."
        )

    checked = 0

    for entry in entry_lineups[
        :entries_written
    ]:
        slots = entry["slots"]

        expected_slots = {
            "QB",
            "RB1",
            "RB2",
            "WR1",
            "WR2",
            "WR3",
            "TE",
            "FLEX",
            "DST",
        }

        if set(slots) != expected_slots:
            raise ValueError(
                "Generated FanDuel lineup failed "
                "slot-structure validation."
            )

        ids = [
            str(slots[slot]).strip()
            for slot in (
                "QB",
                "RB1",
                "RB2",
                "WR1",
                "WR2",
                "WR3",
                "TE",
                "FLEX",
                "DST",
            )
        ]

        if any(not fd_id for fd_id in ids):
            raise ValueError(
                "Generated FanDuel lineup contains "
                "an empty player ID."
            )

        if len(set(ids)) != 9:
            raise ValueError(
                "Generated FanDuel lineup contains "
                "duplicate players."
            )

        checked += 1

    if checked != entries_written:
        raise ValueError(
            "Generated FanDuel file did not validate "
            "the expected number of lineup rows."
        )




def _dc_col(df, names):
    m = {str(c).lower(): str(c) for c in df.columns}
    return next((m[n.lower()] for n in names if n.lower() in m), None)


def _dc_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]


def _dc_discover(conn, kind):
    tables = _dc_tables(conn)
    preferred = (
        ["schedule", "schedules", "nfl_schedule", "games", "game_schedule"]
        if kind == "schedule"
        else ["player_game", "player_games", "boxscores", "player_boxscores",
              "player_game_stats", "weekly_player_stats"]
    )
    for t in preferred:
        if t in tables:
            return t
    for t in tables:
        cols = set(pd.read_sql_query(
            f'PRAGMA table_info("{t}")', conn
        )["name"].str.lower())
        if kind == "schedule":
            if ({"game_id", "gameid"} & cols and
                {"home_team", "home"} & cols and
                {"away_team", "away"} & cols):
                return t
        else:
            if ({"game_id", "gameid"} & cols and
                {"player_name", "player", "display_name", "fantasy_player_name"} & cols and
                {"passing_yards", "rushing_yards", "receiving_yards",
                 "receptions", "targets"} & cols):
                return t
    return None


@st.cache_data(show_spinner=False)
def load_data_center():
    with sqlite3.connect(DATABASE_PATH) as conn:
        stbl = _dc_discover(conn, "schedule")
        btbl = _dc_discover(conn, "box")
        sched = pd.read_sql_query(f'SELECT * FROM "{stbl}"', conn) if stbl else pd.DataFrame()
        box = pd.read_sql_query(f'SELECT * FROM "{btbl}"', conn) if btbl else pd.DataFrame()
    return sched, box, stbl, btbl


DC_STAT_FORECAST_PATH = (
    APP_DIR
    / "data"
    / "parquet"
    / "current_unified_stat_forecasts.parquet"
)


DC_EXPECTED_PASS_RATE_PATH = (
    APP_DIR
    / "processed"
    / "pregame_context_pass_expectation_v1.csv"
)


@st.cache_data(show_spinner=False, ttl=30)
def _dc_load_expected_pass_rates():
    """
    Read-only production pregame Expected Pass Rate.

    Exact identity is game_id + team.
    Missing, ambiguous, malformed, or out-of-range data fails closed.
    """
    try:
        if not DC_EXPECTED_PASS_RATE_PATH.is_file():
            return pd.DataFrame()

        frame = pd.read_csv(
            DC_EXPECTED_PASS_RATE_PATH,
            low_memory=False,
        )

        required = {
            "game_id",
            "team",
            "pregame_context_pass_expectation",
        }

        if not required.issubset(set(frame.columns)):
            return pd.DataFrame()

        frame = frame.copy()

        frame["game_id"] = (
            frame["game_id"]
            .astype(str)
            .str.strip()
        )

        frame["team"] = (
            frame["team"]
            .astype(str)
            .map(_dc_team_code)
        )

        frame["pregame_context_pass_expectation"] = pd.to_numeric(
            frame["pregame_context_pass_expectation"],
            errors="coerce",
        )

        valid = (
            frame["game_id"].ne("")
            & frame["team"].ne("")
            & frame["pregame_context_pass_expectation"].notna()
            & frame["pregame_context_pass_expectation"].between(
                0.0,
                1.0,
                inclusive="both",
            )
        )

        frame = frame.loc[
            valid,
            [
                "game_id",
                "team",
                "pregame_context_pass_expectation",
            ],
        ].copy()

        duplicate_identity = frame.duplicated(
            ["game_id", "team"],
            keep=False,
        )

        if duplicate_identity.any():
            return pd.DataFrame()

        return frame

    except Exception:
        return pd.DataFrame()


def _dc_expected_pass_rate(game_id, team):
    """
    Exact game + NFL team Expected Pass Rate lookup.

    Returns None unless exactly one production row exists.
    """
    game_id = str(game_id or "").strip()
    team = _dc_team_code(team)

    if not game_id or not team:
        return None

    frame = _dc_load_expected_pass_rates()

    if frame.empty:
        return None

    try:
        match = frame[
            frame["game_id"].astype(str).eq(game_id)
            &
            frame["team"].astype(str).eq(team)
        ]

        if len(match) != 1:
            return None

        value = float(
            match.iloc[0][
                "pregame_context_pass_expectation"
            ]
        )

        if pd.isna(value) or value < 0.0 or value > 1.0:
            return None

        return value

    except Exception:
        return None


def _dc_expected_pass_rate_text(game_id, team):
    value = _dc_expected_pass_rate(
        game_id,
        team,
    )

    if value is None:
        return "—"

    return f"{value * 100.0:.1f}%"


@st.cache_data(show_spinner=False, ttl=30)
def _dc_load_stat_forecasts():
    """
    Read-only canonical statistical forecasts.

    This loader never runs a model and never writes forecast data.
    Missing/unreadable forecast data fails closed to an empty frame.
    """
    try:
        if not DC_STAT_FORECAST_PATH.is_file():
            return pd.DataFrame()

        forecasts = pd.read_parquet(
            DC_STAT_FORECAST_PATH
        )

        required = {
            "entity_type",
            "game_id",
            "team",
            "player_id",
            "position",
        }

        if not required.issubset(
            set(forecasts.columns)
        ):
            return pd.DataFrame()

        return forecasts

    except Exception:
        return pd.DataFrame()


def _dc_exact_gsis_id_for_espn(espn_id):
    """
    Exact ESPN -> GSIS identity resolution.

    Fail closed unless exactly one GSIS identity exists.
    """
    espn_id = str(espn_id or "").strip()

    if not espn_id:
        return ""

    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as conn:
            rows = conn.execute(
                """
                SELECT gsis_id
                FROM player_identity
                WHERE CAST(espn_id AS TEXT) = ?
                  AND gsis_id IS NOT NULL
                  AND TRIM(gsis_id) <> ''
                """,
                (espn_id,),
            ).fetchall()

        gsis_ids = sorted(
            {
                str(row[0]).strip()
                for row in rows
                if str(row[0] or "").strip()
            }
        )

        if len(gsis_ids) != 1:
            return ""

        return gsis_ids[0]

    except Exception:
        return ""


def _dc_public_event_count(value):
    """
    Public single-game presentation for discrete football events.

    Forecast models may produce fractional expectations internally,
    but touchdowns, interceptions, sacks, fumble recoveries, etc.
    occur as whole events in an individual game.

    Presentation only. Does not mutate forecast data.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not pd.notna(number):
        return "—"

    number = max(0.0, number)

    # Conventional half-up rounding for nonnegative event counts.
    return str(int(number + 0.5))


def _dc_stat_forecast_for_player(
    game_id,
    player_id,
    team,
):
    """
    Exact game + GSIS player forecast lookup.

    No name matching.
    """
    game_id = str(game_id or "").strip()
    player_id = str(player_id or "").strip()
    team = _dc_team_code(team)

    if not game_id or not player_id:
        return {}

    forecasts = _dc_load_stat_forecasts()

    if forecasts.empty:
        return {}

    try:
        match = forecasts[
            forecasts["entity_type"]
            .astype(str)
            .eq("OFFENSE_PLAYER")
            &
            forecasts["game_id"]
            .astype(str)
            .eq(game_id)
            &
            forecasts["player_id"]
            .astype(str)
            .eq(player_id)
        ].copy()

        if team:
            match = match[
                match["team"]
                .astype(str)
                .str.upper()
                .eq(team)
            ]

        if len(match) != 1:
            return {}

        return match.iloc[0].to_dict()

    except Exception:
        return {}


def _dc_stat_forecast_for_kicker(
    game_id,
    player_id,
    team,
):
    game_id = str(game_id or "").strip()
    player_id = str(player_id or "").strip()
    team = _dc_team_code(team)

    if not game_id or not player_id:
        return {}

    forecasts = _dc_load_stat_forecasts()

    if forecasts.empty:
        return {}

    try:
        match = forecasts[
            forecasts["entity_type"]
            .astype(str)
            .eq("KICKER")
            &
            forecasts["game_id"]
            .astype(str)
            .eq(game_id)
            &
            forecasts["player_id"]
            .astype(str)
            .eq(player_id)
        ].copy()

        if team:
            match = match[
                match["team"]
                .astype(str)
                .str.upper()
                .eq(team)
            ]

        if len(match) != 1:
            return {}

        return match.iloc[0].to_dict()

    except Exception:
        return {}


def _dc_stat_forecast_for_dst(
    game_id,
    team,
):
    """
    Exact game + NFL team DST forecast lookup.
    """
    game_id = str(game_id or "").strip()
    team = _dc_team_code(team)

    if not game_id or not team:
        return {}

    forecasts = _dc_load_stat_forecasts()

    if forecasts.empty:
        return {}

    try:
        match = forecasts[
            forecasts["entity_type"]
            .astype(str)
            .eq("DST")
            &
            forecasts["game_id"]
            .astype(str)
            .eq(game_id)
            &
            forecasts["team"]
            .astype(str)
            .str.upper()
            .eq(team)
        ]

        if len(match) != 1:
            return {}

        return match.iloc[0].to_dict()

    except Exception:
        return {}


def _dc_prediction_number(value, digits=0):
    """
    Public presentation formatter for projected football counting stats.

    Forecast precision remains unchanged internally. Public game-level
    counting projections are displayed as conventional whole numbers.
    """
    try:
        number = float(value)

        if pd.isna(number):
            return None

        if number >= 0:
            return str(int(number + 0.5))

        return str(int(number - 0.5))

    except Exception:
        return None


def _dc_render_stat_prediction(position, forecast):
    """
    Compact public football-stat forecast presentation.
    """
    if not forecast:
        return False

    position = str(position or "").strip().upper()

    def n(key):
        return _dc_prediction_number(
            forecast.get(key)
        )

    lines = []

    if position == "QB":
        comp = n("expected_completions")
        att = n("expected_attempts")
        pass_yds = n("expected_passing_yards")
        pass_td = n("expected_passing_tds")
        interceptions = n("expected_interceptions")

        if any([
            comp,
            att,
            pass_yds,
            pass_td,
            interceptions,
        ]):
            lines.append(
                "Pass: "
                f"{comp or '—'}/{att or '—'}"
                f" • {pass_yds or '—'} Yds"
                f" • {_dc_public_event_count(pass_td)} TD"
                f" • {_dc_public_event_count(interceptions)} INT"
            )

        rush_att = n("expected_carries")
        rush_yds = n("expected_rushing_yards")
        rush_td = n("expected_rushing_tds")

        if any([
            rush_att,
            rush_yds,
            rush_td,
        ]):
            lines.append(
                "Rush: "
                f"{rush_att or '—'} Att"
                f" • {rush_yds or '—'} Yds"
                f" • {_dc_public_event_count(rush_td)} TD"
            )

    elif position in {"RB", "FB"}:
        rush_att = n("expected_carries")
        rush_yds = n("expected_rushing_yards")
        rush_td = n("expected_rushing_tds")

        if any([
            rush_att,
            rush_yds,
            rush_td,
        ]):
            lines.append(
                "Rush: "
                f"{rush_att or '—'} Att"
                f" • {rush_yds or '—'} Yds"
                f" • {_dc_public_event_count(rush_td)} TD"
            )

        targets = n("expected_targets")
        rec = n("expected_receptions")
        rec_yds = n("expected_receiving_yards")
        rec_td = n("expected_receiving_tds")

        if any([
            targets,
            rec,
            rec_yds,
            rec_td,
        ]):
            lines.append(
                "Receiving: "
                f"{targets or '—'} Tgt"
                f" • {rec or '—'} Rec"
                f" • {rec_yds or '—'} Yds"
                f" • {_dc_public_event_count(rec_td)} TD"
            )

    elif position in {"WR", "TE"}:
        targets = n("expected_targets")
        rec = n("expected_receptions")
        rec_yds = n("expected_receiving_yards")
        rec_td = n("expected_receiving_tds")

        if any([
            targets,
            rec,
            rec_yds,
            rec_td,
        ]):
            lines.append(
                f"{targets or '—'} Tgt"
                f" • {rec or '—'} Rec"
                f" • {rec_yds or '—'} Yds"
                f" • {_dc_public_event_count(rec_td)} TD"
            )

    elif position == "K":
        fga = n("expected_fga")
        fgm = n("expected_fgm")
        xpa = n("expected_xpa")
        xpm = n("expected_xpm")

        if any([fga, fgm]):
            lines.append(
                "FG: "
                f"{fga or '—'} Att"
                f" • {_dc_public_event_count(fgm)} Made"
            )

        if any([xpa, xpm]):
            lines.append(
                "XP: "
                f"{xpa or '—'} Att"
                f" • {_dc_public_event_count(xpm)} Made"
            )

    elif position in {"D/ST", "DST"}:
        sacks = n("expected_sacks")
        interceptions = n("expected_interceptions")
        fum_rec = n("expected_fumble_recoveries")
        points_allowed = n("expected_points_allowed")
        def_td = n("expected_defensive_tds")

        if any([
            sacks,
            interceptions,
            fum_rec,
        ]):
            lines.append(
                f"{_dc_public_event_count(sacks)} Sacks"
                f" • {_dc_public_event_count(interceptions)} INT"
                f" • {_dc_public_event_count(fum_rec)} Fum Rec"
            )

        if any([
            points_allowed,
            def_td,
        ]):
            lines.append(
                f"{points_allowed or '—'} Pts Allowed"
                f" • {_dc_public_event_count(def_td)} Def TD"
            )

    if not lines:
        return False

    st.markdown("**Stat Predictions**")

    for line in lines:
        st.caption(line)

    return True


def _dc_score_text(value):
    if pd.isna(value):
        return "—"
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except Exception:
        return str(value)



@st.cache_data(ttl=20, show_spinner=False)
def _dc_live_player_stats(event_id, espn_id):
    event_id = str(event_id or "").strip()
    espn_id = str(espn_id or "").strip()

    if not event_id or not espn_id:
        return {}

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
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
    except Exception:
        return {}

    out = {}

    for team in (data.get("boxscore") or {}).get("players") or []:
        for group in team.get("statistics") or []:
            name = str(group.get("name") or "").lower()
            labels = group.get("labels") or []

            for row in group.get("athletes") or []:
                athlete = row.get("athlete") or {}

                if str(athlete.get("id") or "") != espn_id:
                    continue

                stats = row.get("stats") or []
                vals = dict(zip(labels, stats))

                def num(key):
                    try:
                        return float(vals[key])
                    except Exception:
                        return None

                if name == "passing":
                    ca = str(vals.get("C/ATT") or "")
                    if "/" in ca:
                        try:
                            comp, att = ca.split("/", 1)
                            out["completions"] = float(comp)
                            out["attempts"] = float(att)
                        except Exception:
                            pass
                    out["passing_yards"] = num("YDS")
                    out["passing_tds"] = num("TD")
                    out["passing_interceptions"] = num("INT")

                elif name == "rushing":
                    out["carries"] = num("CAR")
                    out["rushing_yards"] = num("YDS")
                    out["rushing_tds"] = num("TD")

                elif name == "receiving":
                    out["receptions"] = num("REC")
                    out["receiving_yards"] = num("YDS")
                    out["receiving_tds"] = num("TD")
                    out["targets"] = num("TGTS")

    return {
        k: v
        for k, v in out.items()
        if v is not None
    }


# WFS_LIVE_MATCHUP_LOOKUP_V2
@st.cache_data(ttl=20, show_spinner=False)
def _dc_live_event_for_game(game_id):
    game_id = str(game_id or "").strip()
    if not game_id:
        return None

    try:
        nfl_db = "/home/mwynn/nfl_data_engine/data/nfl.db"

        with sqlite3.connect(
            f"file:{nfl_db}?mode=ro",
            uri=True,
        ) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT espn
                FROM games
                WHERE game_id = ?
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

        if row is None:
            return None

        event_id = str(row["espn"] or "").strip()
        if not event_id:
            return None

        url = (
            "https://site.api.espn.com/apis/site/v2/"
            "sports/football/nfl/summary"
            f"?event={event_id}"
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "Mozilla/5.0 WFS-LIVE-RUNNER",
                "Accept":
                    "application/json",
            },
        )

        with urllib.request.urlopen(
            req,
            timeout=10,
        ) as r:
            data = json.loads(r.read())

        header = data.get("header") or {}
        comps = header.get("competitions") or []

        if not comps:
            return None

        comp = comps[0]
        status = comp.get("status") or {}
        status_type = status.get("type") or {}

        state = str(
            status_type.get("state") or ""
        ).strip().lower()

        detail = str(
            status_type.get("detail") or ""
        ).strip()

        period = status.get("period")
        clock = status.get("displayClock")

        away_score = None
        home_score = None

        for team in comp.get("competitors") or []:
            side = str(
                team.get("homeAway") or ""
            ).strip().lower()

            try:
                score = int(float(team.get("score")))
            except Exception:
                score = None

            if side == "away":
                away_score = score
            elif side == "home":
                home_score = score

        return {
            "event_id": event_id,
            "state": state,
            "detail": detail,
            "period": period,
            "clock": clock,
            "away_score": away_score,
            "home_score": home_score,
        }

    except Exception:
        return None



# WFS_LIVE_GAME_AI_UI_V1
def _dc_live_coaching_interpretation_for_game(game_id):
    """
    Read-only presentation bridge for live coaching intelligence.

    Requires:
      - exact internal game_id
      - exact ESPN event_id
      - exact interpretation artifact event_id
      - exact interpretation artifact game_id
      - analysis-only policy contract
      - game currently LIVE

    Any mismatch fails closed.
    """
    game_id = str(game_id or "").strip()

    if not game_id:
        return None

    try:
        live_event = _dc_live_event_for_game(game_id)

        if not live_event:
            return None

        state = str(
            live_event.get("state") or ""
        ).strip().lower()

        if state not in {"in", "live"}:
            return None

        event_id = str(
            live_event.get("event_id") or ""
        ).strip()

        if not event_id:
            return None

        artifact = (
            APP_DIR
            / "processed"
            / "coaching_intelligence"
            / "live_interpretation"
            / f"{event_id}.json"
        )

        if not artifact.is_file():
            return None

        payload = json.loads(artifact.read_text())

        if not isinstance(payload, dict):
            return None

        if (
            str(payload.get("version") or "").strip()
            != "WFS_LIVE_COACHING_INTERPRETATION_V1"
        ):
            return None

        if (
            str(payload.get("event_id") or "").strip()
            != event_id
        ):
            return None

        if (
            str(payload.get("game_id") or "").strip()
            != game_id
        ):
            return None

        policy = payload.get("policy")

        if not isinstance(policy, dict):
            return None

        required_policy = {
            "analysis_only": True,
            "production_influence": False,
            "solver_influence": False,
            "forecast_mutation": False,
            "persistent_coach_prior_mutation": False,
            "current_event_excluded_from_regime_history": True,
            "fuzzy_matching": False,
        }

        for key, expected in required_policy.items():
            if policy.get(key) is not expected:
                return None

        teams = payload.get("teams")

        if not isinstance(teams, list):
            return None

        if len(teams) != 2:
            return None

        cleaned = []

        for row in teams:
            if not isinstance(row, dict):
                return None

            if (
                str(row.get("event_id") or "").strip()
                != event_id
            ):
                return None

            if (
                str(row.get("game_id") or "").strip()
                != game_id
            ):
                return None

            team = str(
                row.get("team") or ""
            ).strip().upper()

            coach = str(
                row.get("coach_identity") or ""
            ).strip()

            if not team or not coach:
                return None

            try:
                plays = int(
                    row.get("live_scrimmage_plays", 0) or 0
                )
            except Exception:
                return None

            if plays < 0:
                return None

            confidence = str(
                row.get("live_sample_confidence") or ""
            ).strip().upper()

            if confidence not in {
                "LOW",
                "MEDIUM",
                "HIGH",
            }:
                return None

            cleaned.append(dict(row))

        return {
            "event_id": event_id,
            "game_id": game_id,
            "teams": cleaned,
        }

    except Exception:
        return None


def _dc_live_ai_pct(value):
    if value is None:
        return "—"

    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "—"


def _dc_live_ai_pp(value):
    if value is None:
        return "—"

    try:
        return f"{float(value) * 100.0:+.1f} pp"
    except Exception:
        return "—"


def _dc_public_live_ai_read(row):
    """
    Concise public football-language presentation.

    Uses only the already-generated deterministic live interpretation.
    Does not recalculate coaching evidence or modify projections.
    """
    team = str(
        row.get("team") or ""
    ).strip().upper()

    live_rate = row.get("live_pass_rate")
    expected_rate = row.get("expected_pass_rate")
    deviation = str(
        row.get("paper_deviation") or ""
    ).strip().upper()

    if live_rate is None:
        return ""

    try:
        live_pct = float(live_rate) * 100.0
    except Exception:
        return ""

    if expected_rate is None:
        return (
            f"**{team}:** Current pass rate is "
            f"**{live_pct:.1f}%**. No matching pregame "
            "baseline is available."
        )

    try:
        expected_pct = float(expected_rate) * 100.0
    except Exception:
        return ""

    if deviation == "MATERIAL_PASS_HEAVY":
        read = "Leaning strongly toward the pass"
    elif deviation == "MODERATE_PASS_HEAVY":
        read = "Leaning toward the pass"
    elif deviation == "MATERIAL_RUN_HEAVY":
        read = "Leaning strongly toward the run"
    elif deviation == "MODERATE_RUN_HEAVY":
        read = "Leaning toward the run"
    else:
        read = "Playing close to the expected pass/run mix"

    return (
        f"**{team}:** {read} — "
        f"**{live_pct:.1f}%** pass rate vs. "
        f"**{expected_pct:.1f}%** expected."
    )


def _dc_render_live_game_ai(
    game_id,
    workspace_mode="Public",
):
    """
    Public:
      football-language interpretation only.

    Admin:
      adds exact coaching/sample diagnostics.

    Presentation only.
    """
    payload = _dc_live_coaching_interpretation_for_game(
        game_id
    )

    if not payload:
        return

    visible_rows = []

    for row in payload["teams"]:
        try:
            plays = int(
                row.get("live_scrimmage_plays", 0) or 0
            )
        except Exception:
            continue

        interpretation = str(
            row.get("paper_interpretation") or ""
        ).strip()

        if plays > 0 and interpretation:
            visible_rows.append(row)

    if not visible_rows:
        return

    st.markdown("#### 🧠 In-Game AI")

    for row in visible_rows:
        team = str(
            row.get("team") or ""
        ).strip().upper()

        interpretation = str(
            row.get("paper_interpretation") or ""
        ).strip()

        plays = int(
            row.get("live_scrimmage_plays", 0) or 0
        )

        confidence = str(
            row.get("live_sample_confidence") or ""
        ).strip().upper()

        public_read = _dc_public_live_ai_read(row)

        if public_read:
            st.markdown(public_read)
        else:
            st.markdown(
                f"**{team}:** {interpretation}"
            )

        if confidence == "HIGH":
            label = "High-confidence live sample"
        elif confidence == "MEDIUM":
            label = "Medium-confidence live sample"
        else:
            label = "Early low-confidence sample"

        st.caption(
            f"{label} • {plays} scrimmage plays"
        )

    if workspace_mode != "Admin":
        return

    with st.expander(
        "🔒 Admin In-Game AI diagnostics",
        expanded=False,
    ):
        diagnostics = []

        for row in payload["teams"]:
            diagnostics.append(
                {
                    "Team":
                        row.get("team"),
                    "Coach":
                        row.get("coach_identity"),
                    "Live Pass Rate":
                        _dc_live_ai_pct(
                            row.get("live_pass_rate")
                        ),
                    "Expected Pass Rate":
                        _dc_live_ai_pct(
                            row.get("expected_pass_rate")
                        ),
                    "Delta vs Paper":
                        _dc_live_ai_pp(
                            row.get("paper_delta")
                        ),
                    "Live Plays":
                        row.get(
                            "live_scrimmage_plays"
                        ),
                    "Live Confidence":
                        row.get(
                            "live_sample_confidence"
                        ),
                    "Prior Regime Games":
                        row.get(
                            "prior_regime_games"
                        ),
                    "Prior Regime Plays":
                        row.get(
                            "prior_regime_scrimmage_plays"
                        ),
                    "Prior Regime Pass Rate":
                        _dc_live_ai_pct(
                            row.get(
                                "prior_regime_pass_rate"
                            )
                        ),
                    "Regime Confidence":
                        row.get(
                            "regime_evidence_confidence"
                        ),
                }
            )

        st.dataframe(
            pd.DataFrame(diagnostics),
            width="stretch",
            hide_index=True,
        )

        for row in payload["teams"]:
            team = str(
                row.get("team") or ""
            ).strip().upper()

            history = str(
                row.get("regime_interpretation") or ""
            ).strip()

            if history:
                st.caption(
                    f"{team} — {history}"
                )


def _dc_game_status(away_score, home_score):
    if away_score is not None and home_score is not None:
        if pd.notna(away_score) and pd.notna(home_score):
            return "FINAL"
    return "UPCOMING"


NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def _dc_team_code(value):
    return str(value or "").strip().upper()


# QA F15 / WFS_DATA_CENTER_MULTI_TEAM_FILTER_V1 (extension)
#
# Deterministic full-team-name/nickname -> code aliases derived from the
# existing NFL_TEAM_NAMES canonical mapping. No fuzzy matching, no second
# team-name authority. Built once at import time.
_DC_TEAM_FULL_NAME_TO_CODE = {
    name.strip().upper(): code
    for code, name in NFL_TEAM_NAMES.items()
}
_DC_TEAM_NICKNAME_TO_CODE = {
    name.strip().split()[-1].upper(): code
    for code, name in NFL_TEAM_NAMES.items()
}


def _dc_resolve_team_token(value) -> str:
    """
    Resolve one user-typed team filter token to a canonical team code.

    Exact codes keep their already-validated behavior unchanged. Full team
    names ("Atlanta Falcons") and their single-word nickname ("Falcons")
    resolve via the existing NFL_TEAM_NAMES mapping. Unknown text returns
    "" rather than guessing.
    """
    text = str(value or "").strip()
    if not text:
        return ""

    code = _dc_team_code(text)
    if code in NFL_TEAM_NAMES:
        return code

    upper = text.upper()
    if upper in _DC_TEAM_FULL_NAME_TO_CODE:
        return _DC_TEAM_FULL_NAME_TO_CODE[upper]
    if upper in _DC_TEAM_NICKNAME_TO_CODE:
        return _DC_TEAM_NICKNAME_TO_CODE[upper]

    return ""


_NFL_LOGO_DATA_CACHE = {}


def _dc_team_logo_html(team_code, css_class="wfs-team-logo"):
    """Return a locally sourced NFL team logo as an embedded PNG."""
    team_code = _dc_team_code(team_code)

    if team_code not in NFL_TEAM_NAMES:
        return ""

    if team_code not in _NFL_LOGO_DATA_CACHE:
        logo_path = Path(__file__).resolve().parent / "assets" / "nfl_teams" / f"{team_code}.png"

        if not logo_path.is_file():
            _NFL_LOGO_DATA_CACHE[team_code] = ""
        else:
            try:
                encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
                _NFL_LOGO_DATA_CACHE[team_code] = encoded
            except Exception:
                _NFL_LOGO_DATA_CACHE[team_code] = ""

    encoded = _NFL_LOGO_DATA_CACHE.get(team_code, "")

    if not encoded:
        return ""

    return (
        f'<img class="{css_class}" '
        f'src="data:image/png;base64,{encoded}" '
        f'alt="{team_code} logo">'
    )


def _dc_matchup_has_team(row, cols, team_code):
    team_code = _dc_team_code(team_code)
    if not team_code:
        return False

    away = _dc_team_code(row[cols["away"]]) if cols.get("away") else ""
    home = _dc_team_code(row[cols["home"]]) if cols.get("home") else ""
    return team_code in {away, home}


def _dc_matchup_card(row, cols, favorite_team="", my_player_count=0):
    away = str(row[cols["away"]]) if cols["away"] else "Away"
    home = str(row[cols["home"]]) if cols["home"] else "Home"
    date_value = str(row[cols["date"]]) if cols["date"] and pd.notna(row[cols["date"]]) else ""
    if date_value:
        date_value = _wfs_format_kickoff(
            date_value,
            row.get("weekday", ""),
            row.get("gametime", ""),
        )
    week_value = str(row[cols["week"]]) if cols["week"] and pd.notna(row[cols["week"]]) else ""
    # WFS_LIVE_MATCHUP_CARD_V1
    away_score = row[cols["away_score"]] if cols["away_score"] else None
    home_score = row[cols["home_score"]] if cols["home_score"] else None
    gid = str(row[cols["gid"]]) if cols["gid"] else ""

    live_event = _dc_live_event_for_game(gid)

    if live_event:
        if live_event.get("away_score") is not None:
            away_score = live_event.get("away_score")

        if live_event.get("home_score") is not None:
            home_score = live_event.get("home_score")

        state = str(
            live_event.get("state") or ""
        ).strip().lower()

        if state in {"in", "live"}:
            status = "LIVE"
        elif state in {"post", "final", "completed"}:
            status = "FINAL"
        else:
            status = _dc_game_status(
                away_score,
                home_score,
            )
    else:
        status = _dc_game_status(
            away_score,
            home_score,
        )

    if status == "LIVE":
        status_class = "live"
    elif status == "FINAL":
        status_class = "final"
    else:
        status_class = "upcoming"


    favorite_team = _dc_team_code(favorite_team)
    away_is_favorite = favorite_team and _dc_team_code(away) == favorite_team
    home_is_favorite = favorite_team and _dc_team_code(home) == favorite_team
    favorite_game = away_is_favorite or home_is_favorite

    card_class = "wfs-game-card favorite" if favorite_game else "wfs-game-card"
    away_class = "wfs-team-row favorite-team" if away_is_favorite else "wfs-team-row"
    home_class = "wfs-team-row favorite-team" if home_is_favorite else "wfs-team-row"

    favorite_badge = (
        '<span class="wfs-my-team-pill">⭐ MY TEAM</span>'
        if favorite_game else ""
    )

    try:
        my_player_count = max(0, int(my_player_count or 0))
    except Exception:
        my_player_count = 0

    if my_player_count == 1:
        my_players_badge = (
            '<span class="wfs-my-players-pill">⭐ 1 MY PLAYER</span>'
        )
    elif my_player_count > 1:
        my_players_badge = (
            f'<span class="wfs-my-players-pill">'
            f'⭐ {my_player_count} MY PLAYERS</span>'
        )
    else:
        my_players_badge = ""

    away_star = '<span class="wfs-team-star">⭐</span>' if away_is_favorite else ""
    home_star = '<span class="wfs-team-star">⭐</span>' if home_is_favorite else ""

    away_logo = _dc_team_logo_html(away)
    home_logo = _dc_team_logo_html(home)

    return f"""
    <div class="{card_class}">
      <div class="wfs-game-top">
        <span>WEEK {week_value}</span>
        <div class="wfs-game-top-right">{favorite_badge}{my_players_badge}<span class="wfs-game-status {status_class}">{status}</span></div>
      </div>
      <div class="wfs-game-date">{date_value}</div>
      <div class="{away_class}">
        <div class="wfs-team-identity">{away_logo}<span class="wfs-team-badge">{away}</span><span class="wfs-team-label">{away}</span>{away_star}</div>
        <div class="wfs-team-right">
          <div class="wfs-score">{_dc_score_text(away_score)}</div>
        </div>
      </div>
      <div class="{home_class}">
        <div class="wfs-team-identity">{home_logo}<span class="wfs-team-badge">{home}</span><span class="wfs-team-label">{home}</span>{home_star}</div>
        <div class="wfs-team-right">
          <div class="wfs-score">{_dc_score_text(home_score)}</div>
        </div>
      </div>
    </div>
    """



DC_STAT_LABELS = {
    "player_name": "Player",
    "player": "Player",
    "display_name": "Player",
    "fantasy_player_name": "Player",
    "team": "Team",
    "recent_team": "Team",
    "posteam": "Team",
    "position": "Pos",
    "pos": "Pos",
    "passing_attempts": "Pass Att",
    "attempts": "Pass Att",
    "completions": "Comp",
    "passing_yards": "Pass Yds",
    "passing_tds": "Pass TD",
    "interceptions": "INT",
    "sacks": "Sacks",
    "carries": "Carries",
    "rushing_yards": "Rush Yds",
    "rushing_tds": "Rush TD",
    "targets": "Targets",
    "receptions": "Rec",
    "receiving_yards": "Rec Yds",
    "receiving_tds": "Rec TD",
    "fumbles_lost": "Fum Lost",
    "fantasy_points": "Fantasy Pts",
    "fantasy_points_ppr": "PPR Pts",
    "fd_points": "FanDuel Pts",
    "fanduel_points": "FanDuel Pts",
}


def _dc_public_label(col):
    return DC_STAT_LABELS.get(str(col).lower(), str(col).replace("_", " ").title())


def _dc_render_stat_table(stats, identity_cols, wanted, sort_candidates=None):
    cols = [x for x in identity_cols if x]
    stat_cols = []
    for name in wanted:
        col = _dc_col(stats, [name])
        if col and col not in cols:
            cols.append(col)
            stat_cols.append(col)

    if not stat_cols:
        st.info("No statistics for this category are stored in the current box-score table.")
        return

    view = stats[cols].copy()

    numeric = view[stat_cols].apply(pd.to_numeric, errors="coerce")
    active = numeric.fillna(0).abs().sum(axis=1) > 0
    if active.any():
        view = view.loc[active].copy()

    sort_col = None
    for candidate in (sort_candidates or []):
        candidate_col = _dc_col(view, [candidate])
        if candidate_col:
            sort_col = candidate_col
            break

    if sort_col:
        view["__dc_sort"] = pd.to_numeric(view[sort_col], errors="coerce").fillna(-999999)
        view = view.sort_values("__dc_sort", ascending=False, kind="stable").drop(columns="__dc_sort")

    # QA D12: display FanDuel point columns to a consistent one decimal place.
    # Formatting only -- the underlying stat_cols values are not modified.
    fd_point_labels = {
        _dc_public_label(c)
        for c in stat_cols
        if str(c).lower() in {"fd_points", "fanduel_points"}
    }

    view = view.rename(columns={c: _dc_public_label(c) for c in view.columns})

    column_config = {
        label: st.column_config.NumberColumn(format="%.1f")
        for label in fd_point_labels
        if label in view.columns
    }

    st.dataframe(
        view.reset_index(drop=True),
        width="stretch",
        hide_index=True,
        height=min(520, 38 + max(1, len(view)) * 35),
        column_config=column_config or None,
    )

ESPN_PRO_TEAM_ID_TO_NFL = {
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LA",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WAS",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}


def _dc_my_fantasy_players_by_team(user_sub):
    """
    Resolve the signed-in user's followed ESPN fantasy roster to canonical
    NFL team counts for Data Center presentation.

    Identity path:
        followed ESPN team
            -> exact ESPN roster
            -> player.proTeamId
            -> explicit ESPN pro-team ID map
            -> canonical WFS NFL team code

    No fuzzy player matching and no dependency on box scores or
    weekly_rosters. Unknown ESPN team IDs fail closed per player.
    """
    try:
        user_sub = str(user_sub or "").strip()
        if not user_sub:
            return {}

        swid, espn_s2 = _wfs_espn_credentials_for_user(user_sub)
        if not swid or not espn_s2:
            return {}

        with _wfs_accounts_connect() as conn:
            connections = conn.execute(
                """
                SELECT season, league_id, team_id
                FROM wfs_fantasy_connections
                WHERE user_sub = ?
                  AND provider = 'ESPN'
                  AND team_id IS NOT NULL
                ORDER BY season DESC, updated_at DESC, id DESC
                """,
                (user_sub,),
            ).fetchall()

        if not connections:
            return {}

        connection = connections[0]

        season = int(connection["season"])
        league_id = str(connection["league_id"]).strip()
        team_id = int(connection["team_id"])

        result = espn_league_get(
            league_id,
            season,
            ["mTeam", "mRoster"],
            swid,
            espn_s2,
        )

        if not result.get("ok"):
            return {}

        teams = (result.get("data") or {}).get("teams") or []

        exact_teams = []
        for team in teams:
            try:
                if int(team.get("id")) == team_id:
                    exact_teams.append(team)
            except Exception:
                continue

        if len(exact_teams) != 1:
            return {}

        roster_rows = _wfs_int_roster_rows(exact_teams[0])
        if not roster_rows:
            return {}

        counts = {}

        for row in roster_rows:
            try:
                pro_team_id = int(row.get("pro_team_id"))
            except (TypeError, ValueError):
                continue

            team_code = ESPN_PRO_TEAM_ID_TO_NFL.get(pro_team_id)

            if not team_code:
                continue

            team_code = _dc_team_code(team_code)
            if not team_code:
                continue

            counts[team_code] = counts.get(team_code, 0) + 1

        return counts

    except Exception:
        return {}



def _dc_my_players_game_context(
    user_sub,
    game_id,
    away_team,
    home_team,
    week,
):
    """
    Read-only personalized fantasy-game context.

    Identity:
        ESPN roster player.id
            -> nfl.db player_identity.espn_id
            -> player_identity.gsis_id
            -> player_game_stats.player_id

    Live fantasy scoring:
        ESPN appliedTotal using the connected league's scoring.

    Final NFL statistics:
        nfl.db player_game_stats only.

    No fuzzy player identity matching.
    No database writes.
    """
    result = {
        "mode": "UPCOMING",
        "detail": "",
        "score": "",
        "players": [],
        "espn_points_available": False,
    }

    try:
        user_sub = str(user_sub or "").strip()
        game_id = str(game_id or "").strip()

        if not user_sub or not game_id:
            return result

        try:
            target_week = int(float(str(week)))
        except Exception:
            target_week = 1

        away_team = _dc_team_code(away_team)
        home_team = _dc_team_code(home_team)
        game_teams = {away_team, home_team}

        # ----------------------------------------------------------
        # Exact connected ESPN fantasy team
        # ----------------------------------------------------------
        swid, espn_s2 = _wfs_espn_credentials_for_user(user_sub)

        if not swid or not espn_s2:
            return result

        with _wfs_accounts_connect() as account_conn:
            connections = account_conn.execute(
                """
                SELECT season, league_id, team_id
                FROM wfs_fantasy_connections
                WHERE user_sub = ?
                  AND provider = 'ESPN'
                  AND team_id IS NOT NULL
                ORDER BY season DESC, updated_at DESC, id DESC
                """,
                (user_sub,),
            ).fetchall()

        if not connections:
            return result

        connection = connections[0]
        season = int(connection["season"])
        league_id = str(connection["league_id"]).strip()
        fantasy_team_id = int(connection["team_id"])

        league_result = espn_league_get(
            league_id,
            season,
            ["mTeam", "mRoster"],
            swid,
            espn_s2,
        )

        if not league_result.get("ok"):
            return result

        teams = (league_result.get("data") or {}).get("teams") or []

        exact_teams = []
        for team in teams:
            try:
                if int(team.get("id")) == fantasy_team_id:
                    exact_teams.append(team)
            except Exception:
                continue

        if len(exact_teams) != 1:
            return result

        my_team = exact_teams[0]
        roster_rows = _wfs_int_roster_rows(my_team)

        # ESPN league-scoring actual points.
        espn_actuals = _wfs_espn_actual_points_for_week(
            my_team,
            target_week,
        )

        # ----------------------------------------------------------
        # Only roster players whose NFL team is in this matchup.
        # IDP is intentionally excluded. Team D/ST remains supported.
        # ----------------------------------------------------------
        game_roster = []

        for roster_row in roster_rows:
            position = str(roster_row.get("position") or "").strip()

            if position not in {
                "QB",
                "RB",
                "WR",
                "TE",
                "K",
                "D/ST",
            }:
                continue

            try:
                pro_team_id = int(roster_row.get("pro_team_id"))
            except (TypeError, ValueError):
                continue

            nfl_team = ESPN_PRO_TEAM_ID_TO_NFL.get(pro_team_id)
            nfl_team = _dc_team_code(nfl_team)

            if nfl_team not in game_teams:
                continue

            name = str(roster_row.get("player") or "").strip()
            name_key = normalize_name(name)

            fantasy_points = espn_actuals.get(name_key)

            if fantasy_points is not None:
                result["espn_points_available"] = True

            game_roster.append(
                {
                    "espn_id": str(
                        roster_row.get("player_id") or ""
                    ).strip(),
                    "player": name,
                    "position": position,
                    "team": nfl_team,
                    "espn_points": fantasy_points,
                }
            )

        if not game_roster:
            return result

        # ----------------------------------------------------------
        # Read-only pregame statistical forecasts.
        #
        # Player identity remains exact:
        # ESPN player ID -> player_identity.gsis_id -> forecast player_id.
        #
        # Team D/ST uses exact game + NFL team identity.
        # No name matching and no model execution occurs here.
        # ----------------------------------------------------------
        for player in game_roster:
            position = str(
                player.get("position") or ""
            ).strip().upper()

            team = _dc_team_code(
                player.get("team")
            )

            if position == "D/ST":
                player["stat_forecast"] = (
                    _dc_stat_forecast_for_dst(
                        game_id,
                        team,
                    )
                )
                continue

            gsis_id = _dc_exact_gsis_id_for_espn(
                player.get("espn_id")
            )

            if not gsis_id:
                player["stat_forecast"] = {}
                continue

            player["gsis_id"] = gsis_id

            if position == "K":
                player["stat_forecast"] = (
                    _dc_stat_forecast_for_kicker(
                        game_id,
                        gsis_id,
                        team,
                    )
                )
            else:
                player["stat_forecast"] = (
                    _dc_stat_forecast_for_player(
                        game_id,
                        gsis_id,
                        team,
                    )
                )

        # ----------------------------------------------------------
        # Exact NFL game / ESPN event identity from nfl.db.
        # ----------------------------------------------------------
        game_row = None

        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as conn:
            conn.row_factory = sqlite3.Row

            game_row = conn.execute(
                """
                SELECT
                    game_id,
                    espn,
                    completed,
                    away_team,
                    home_team,
                    away_score,
                    home_score
                FROM games
                WHERE game_id = ?
                LIMIT 1
                """,
                (game_id,),
            ).fetchone()

        espn_event_id = ""
        completed = False

        if game_row is not None:
            espn_event_id = str(
                game_row["espn"] or ""
            ).strip()

            try:
                completed = bool(int(game_row["completed"] or 0))
            except Exception:
                completed = bool(game_row["completed"])

            if completed:
                result["mode"] = "FINAL"

            try:
                away_score = game_row["away_score"]
                home_score = game_row["home_score"]

                if (
                    away_score is not None
                    and home_score is not None
                ):
                    result["score"] = (
                        f"{away_team} "
                        f"{_dc_score_text(away_score)}"
                        f" — "
                        f"{home_team} "
                        f"{_dc_score_text(home_score)}"
                    )
            except Exception:
                pass

        # ----------------------------------------------------------
        # Exact ESPN-event binding to WFS Live.
        # No matchup/date fuzzy matching.
        # ----------------------------------------------------------
        live_db = (
            Path(__file__).resolve().parent
            / "data"
            / "wfs_live.db"
        )

        if espn_event_id and live_db.exists():
            try:
                with sqlite3.connect(
                    f"file:{live_db}?mode=ro",
                    uri=True,
                ) as live_conn:
                    live_conn.row_factory = sqlite3.Row

                    live_event = live_conn.execute(
                        """
                        SELECT
                            state,
                            detail,
                            period,
                            clock,
                            away_team,
                            home_team,
                            away_score,
                            home_score
                        FROM live_events
                        WHERE event_id = ?
                        LIMIT 1
                        """,
                        (espn_event_id,),
                    ).fetchone()

                if live_event is not None:
                    live_state = str(
                        live_event["state"] or ""
                    ).strip().lower()

                    if live_state in {"in", "live"}:
                        result["mode"] = "LIVE"

                    elif live_state in {
                        "post",
                        "final",
                        "completed",
                    }:
                        result["mode"] = "FINAL"

                    detail = str(
                        live_event["detail"] or ""
                    ).strip()

                    clock = str(
                        live_event["clock"] or ""
                    ).strip()

                    try:
                        period = int(
                            live_event["period"] or 0
                        )
                    except Exception:
                        period = 0

                    live_parts = []

                    if detail:
                        live_parts.append(detail)
                    elif period:
                        live_parts.append(f"Q{period}")

                    if (
                        clock
                        and clock.lower()
                        not in {"0", "0:00"}
                        and clock not in detail
                    ):
                        live_parts.append(clock)

                    result["detail"] = " • ".join(
                        live_parts
                    )

                    try:
                        live_away_score = (
                            live_event["away_score"]
                        )
                        live_home_score = (
                            live_event["home_score"]
                        )

                        if (
                            live_away_score is not None
                            and live_home_score is not None
                        ):
                            result["score"] = (
                                f"{away_team} "
                                f"{_dc_score_text(live_away_score)}"
                                f" — "
                                f"{home_team} "
                                f"{_dc_score_text(live_home_score)}"
                            )
                    except Exception:
                        pass

            except Exception:
                pass

        # WFS_LIVE_PLAYER_STATS_ATTACHED_V1
        if result["mode"] == "LIVE" and espn_event_id:
            for player in game_roster:
                position = str(
                    player.get("position") or ""
                ).strip().upper()

                if position in {"K", "D/ST"}:
                    continue

                live_stats = _dc_live_player_stats(
                    espn_event_id,
                    player.get("espn_id"),
                )

                for key, value in live_stats.items():
                    player[key] = value

        # ----------------------------------------------------------
        # Final authoritative stat lines through exact ESPN -> GSIS
        # identity. No name matching.
        # ----------------------------------------------------------
        if result["mode"] == "FINAL":
            with sqlite3.connect(
                f"file:{DATABASE_PATH}?mode=ro",
                uri=True,
            ) as conn:
                conn.row_factory = sqlite3.Row

                for player in game_roster:
                    # Team D/ST does not use player_game_stats.
                    if str(
                        player.get("position") or ""
                    ).strip().upper() == "D/ST":
                        continue

                    gsis_id = str(
                        player.get("gsis_id") or ""
                    ).strip()

                    if not gsis_id:
                        gsis_id = _dc_exact_gsis_id_for_espn(
                            player.get("espn_id")
                        )

                    # Fail closed if exact identity is unavailable.
                    if not gsis_id:
                        continue

                    stats = conn.execute(
                        """
                        SELECT
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
                          AND player_id = ?
                        LIMIT 1
                        """,
                        (
                            game_id,
                            gsis_id,
                        ),
                    ).fetchone()

                    if stats is None:
                        continue

                    for key in stats.keys():
                        player[key] = stats[key]

        result["players"] = game_roster

        return result

    except Exception:
        return result


@st.fragment(run_every="30s")
def _dc_render_my_players_panel(
    user_sub,
    game_id,
    away_team,
    home_team,
    week,
):
    """
    Auto-refreshing personalized game panel.

    ESPN appliedTotal is the fantasy scoring authority.
    Verified NFL stat lines are shown after completion.
    """
    context = _dc_my_players_game_context(
        user_sub,
        game_id,
        away_team,
        home_team,
        week,
    )

    players = context.get("players") or []

    if not players:
        st.info(
            "Your ESPN fantasy players for this matchup "
            "are not available right now."
        )
        return

    mode = str(context.get("mode") or "UPCOMING").upper()
    score = str(context.get("score") or "").strip()
    detail = str(context.get("detail") or "").strip()

    if mode == "LIVE":
        status_text = "🔴 LIVE"
    elif mode == "FINAL":
        status_text = "✅ FINAL"
    else:
        status_text = "⏳ UPCOMING"

    header_parts = [status_text]

    if score:
        header_parts.append(score)

    if detail and mode == "LIVE":
        header_parts.append(detail)

    st.markdown(
        "**" + " • ".join(header_parts) + "**"
    )

    st.caption(
        "Fantasy points use your connected ESPN league's "
        "applied scoring."
    )

    def _clean_stat(value):
        if value is None:
            return None
        try:
            number = max(0.0, float(value))
            if number == 0:
                return None
            return str(int(number + 0.5))
        except Exception:
            value = str(value).strip()
            return value or None

    def _stat_piece(label, value):
        value = _clean_stat(value)
        if value is None:
            return None
        return f"**{label}:** {value}"

    for player in players:
        name = str(
            player.get("player") or "Player"
        ).strip()
        position = str(
            player.get("position") or ""
        ).strip().upper()
        nfl_team = str(
            player.get("team") or ""
        ).strip()

        points = player.get("espn_points")

        try:
            points_text = (
                f"{float(points):.2f}"
                if points is not None
                else "—"
            )
        except Exception:
            points_text = "—"

        st.markdown(
            f"### {name}"
        )

        identity_bits = [
            value
            for value in [position, nfl_team]
            if value
        ]

        if identity_bits:
            st.caption(
                " • ".join(identity_bits)
            )

        st.markdown(
            f"**ESPN Pts: {points_text}**"
        )

        # WFS_LIVE_STAT_RENDER_V1
        forecast = player.get(
            "stat_forecast"
        ) or {}

        if mode == "UPCOMING":
            forecast_rendered = (
                _dc_render_stat_prediction(
                    position,
                    forecast,
                )
            )

            if not forecast_rendered:
                st.caption(
                    "Stat predictions are not available "
                    "for this player right now."
                )

            st.divider()
            continue

        stat_parts = []

        if position == "QB":
            completions = _clean_stat(
                player.get("completions")
            )
            attempts = _clean_stat(
                player.get("attempts")
            )

            if completions or attempts:
                stat_parts.append(
                    f"**Comp/Att:** "
                    f"{completions or '0'}/"
                    f"{attempts or '0'}"
                )

            for label, key in [
                ("Pass Yds", "passing_yards"),
                ("Pass TD", "passing_tds"),
                ("INT", "passing_interceptions"),
                ("Rush Yds", "rushing_yards"),
            ]:
                piece = _stat_piece(
                    label,
                    player.get(key),
                )
                if piece:
                    stat_parts.append(piece)

        elif position == "RB":
            for label, key in [
                ("Carries", "carries"),
                ("Rush Yds", "rushing_yards"),
                ("Rush TD", "rushing_tds"),
                ("Rec", "receptions"),
                ("Rec Yds", "receiving_yards"),
                ("Rec TD", "receiving_tds"),
            ]:
                piece = _stat_piece(
                    label,
                    player.get(key),
                )
                if piece:
                    stat_parts.append(piece)

        elif position in {"WR", "TE"}:
            for label, key in [
                ("Rec", "receptions"),
                ("Targets", "targets"),
                ("Rec Yds", "receiving_yards"),
                ("Rec TD", "receiving_tds"),
            ]:
                piece = _stat_piece(
                    label,
                    player.get(key),
                )
                if piece:
                    stat_parts.append(piece)

        # K and D/ST intentionally show only ESPN fantasy points
        # until a verified public-facing stat contract is added.

        if stat_parts:
            st.markdown(
                " · ".join(stat_parts)
            )

        st.divider()

    if mode == "LIVE":
        st.caption(
            "Updates automatically during the game. "
            "Final player stats appear when the game is complete."
        )

    elif mode == "FINAL":
        st.caption(
            "Final stats and ESPN fantasy points are shown "
            "for your rostered players in this game."
        )

    else:
        st.caption(
            "Fantasy scoring will populate when this matchup begins."
        )



def render_data_center(workspace_mode="Public"):
    sched, box, stbl, btbl = load_data_center()

    dc_user = _wfs_current_user()
    favorite_team = ""
    dc_my_players_by_team = {}

    if dc_user:
        try:
            _wfs_upsert_user(dc_user)
            prefs = _wfs_user_preferences(dc_user["sub"])
            favorite_team = _dc_team_code(prefs.get("favorite_nfl_team", ""))

            # Presentation-only ESPN roster → canonical NFL team counts.
            # Exact ESPN player-ID matching through weekly_rosters.
            # Independent of box-score availability; fail closed on ambiguity.
            try:
                dc_my_players_by_team = _dc_my_fantasy_players_by_team(
                    dc_user["sub"]
                )
            except Exception:
                dc_my_players_by_team = {}

        except Exception:
            favorite_team = ""
            dc_my_players_by_team = {}

    st.markdown(
        """
        <style>
        .wfs-dc-hero {
            background:linear-gradient(110deg,#0b2f68 0%,#0b5ec7 72%,#0874df 100%);
            border-radius:16px;padding:1.05rem 1.25rem;margin-bottom:.85rem;
            color:white;box-shadow:0 8px 24px rgba(15,55,105,.12);
        }
        .wfs-dc-hero h2 {margin:0;font-size:1.35rem;font-weight:900;}
        .wfs-dc-hero p {margin:.28rem 0 0;color:#d9eaff;font-size:.86rem;}
        .wfs-game-card {
            background:#fff;border:1px solid #dbe5f1;border-radius:14px;
            padding:.85rem .95rem;margin-bottom:.72rem;
            box-shadow:0 3px 10px rgba(25,55,90,.05);
            min-height:176px;
        }
        .wfs-game-card.favorite {
            border:2px solid #f2b632;
            background:linear-gradient(180deg,#fffdf5 0%,#ffffff 100%);
            box-shadow:0 5px 16px rgba(242,182,50,.14);
        }
        .wfs-game-top-right {
            display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;justify-content:flex-end;
        }
        .wfs-my-team-pill {
            display:inline-flex;align-items:center;border-radius:999px;
            padding:.18rem .48rem;background:#fff4cc;color:#7a5200;
            font-size:.6rem;font-weight:900;letter-spacing:.05em;
        }
        .wfs-my-players-pill {
            display:inline-flex;
            align-items:center;
            border-radius:999px;
            padding:.18rem .48rem;
            background:#eaf3ff;
            color:#075db7;
            border:1px solid #cfe2fb;
            font-size:.6rem;
            font-weight:900;
            letter-spacing:.04em;
            white-space:nowrap;
        }
        .wfs-team-row.favorite-team {
            background:#fff9e8;border-radius:8px;padding-left:.35rem;padding-right:.35rem;
        }
        .wfs-team-star {
            margin-left:.35rem;font-size:.72rem;
        }
        .wfs-game-top {display:flex;justify-content:space-between;align-items:center;
            color:#64748b;font-size:.68rem;font-weight:850;letter-spacing:.08em;}
        .wfs-game-status {border-radius:999px;padding:.18rem .48rem;font-size:.62rem;}
        .wfs-game-status.upcoming {background:#eaf3ff;color:#075db7;}
        .wfs-game-status.final {background:#e8f8ef;color:#08783e;}
        .wfs-game-date {color:#7b8ca3;font-size:.72rem;margin:.28rem 0 .55rem;}
        .wfs-team-row {display:flex;justify-content:space-between;align-items:center;
            padding:.25rem 0;border-top:1px solid #f1f5f9;}
        .wfs-team-badge {display:inline-flex;align-items:center;justify-content:center;
            width:38px;height:28px;border-radius:7px;background:#eef5ff;color:#0b57a5;
            font-size:.7rem;font-weight:900;margin-right:.55rem;}
        .wfs-team-label {font-size:.9rem;font-weight:850;color:#172b45;}
        .wfs-score {font-size:1.05rem;font-weight:900;color:#0b4f9c;}
        .wfs-team-right {
            display:flex;
            flex-direction:column;
            align-items:flex-end;
            justify-content:center;
            gap:.05rem;
        }
        .wfs-pass-expectation {
            color:#6f8299;
            font-size:.62rem;
            font-weight:800;
            white-space:nowrap;
        }
        .wfs-game-id {margin-top:.4rem;color:#9aa9bb;font-size:.61rem;}
        .wfs-box-hero {background:#fff;border:1px solid #dbe5f1;border-radius:15px;
            padding:1rem 1.1rem;margin:.5rem 0 .9rem;box-shadow:0 3px 10px rgba(25,55,90,.05);}
        .wfs-team-identity {
            display:flex;
            align-items:center;
            gap:.48rem;
            min-width:0;
        }
        .wfs-team-logo {
            width:34px;
            height:34px;
            object-fit:contain;
            flex:0 0 34px;
        }
        .wfs-box-matchup {
            display:flex;
            align-items:center;
            gap:.7rem;
            flex-wrap:wrap;
            font-size:1.25rem;
            font-weight:900;
            color:#17365d;
        }
        .wfs-box-team {
            display:inline-flex;
            align-items:center;
            gap:.42rem;
            white-space:nowrap;
        }
        .wfs-box-team-logo {
            width:38px;
            height:38px;
            object-fit:contain;
            flex:0 0 38px;
        }
        .wfs-box-vs {
            color:#8293a7;
            font-weight:800;
        }
        .wfs-box-hero-row {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
        }
        .wfs-box-status {
            font-weight:800;
            letter-spacing:.08em;
        }
        .wfs-box-winner {
            background:#e8f8ef;
            color:#08783e;
            border-radius:999px;
            padding:.35rem .65rem;
            font-size:.68rem;
            font-weight:900;
            white-space:nowrap;
        }
        .wfs-box-sub {color:#6f8299;font-size:.76rem;margin-top:.2rem;}
        </style>
        <div class="wfs-dc-hero">
          <h2>🏈 NFL Data Center</h2>
          <p>Schedules, results, box scores and fantasy performance from around the NFL.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if workspace_mode == "Admin":
        a, b, c = st.columns(3)
        a.metric("Schedule Games", f"{len(sched):,}")
        b.metric("Player-Game Rows", f"{len(box):,}")
        c.metric("Data Status", "READY" if len(sched) else "CHECK DB")

    team_options = [""] + list(NFL_TEAM_NAMES.keys())

    def _favorite_team_label(code):
        if not code:
            return "No favorite team"
        return f"{code} — {NFL_TEAM_NAMES.get(code, code)}"

    if dc_user:
        current_index = (
            team_options.index(favorite_team)
            if favorite_team in team_options else 0
        )

        selected_favorite_team = st.selectbox(
            "⭐ My NFL Team",
            team_options,
            index=current_index,
            format_func=_favorite_team_label,
            key="dc_favorite_nfl_team",
            help="Your team will be highlighted throughout the NFL Data Center.",
        )

        selected_favorite_team = _dc_team_code(selected_favorite_team)

        if selected_favorite_team != favorite_team:
            _wfs_set_favorite_nfl_team(
                dc_user["sub"],
                selected_favorite_team,
            )
            favorite_team = selected_favorite_team
    else:
        st.caption("Sign in to save and highlight your favorite NFL team.")

    if workspace_mode == "Admin":
        st.markdown("### 🔒 Admin Postgame Feedback")
        from wfs_postgame_feedback import render_postgame_feedback_admin
        render_postgame_feedback_admin()
        st.divider()

    schedule_tab, box_tab, leaders_tab = st.tabs(
        ["📅 Schedule", "📊 Box Scores", "🏆 Weekly Leaders"]
    )

    with schedule_tab:
        if sched.empty:
            st.info("No NFL schedule is available yet.")
        else:
            season = _dc_col(sched, ["season", "season_year", "year"])
            week = _dc_col(sched, ["week", "week_num", "week_number"])
            away = _dc_col(sched, ["away_team", "away"])
            home = _dc_col(sched, ["home_team", "home"])
            date = _dc_col(sched, ["gameday", "game_date", "date", "start_time", "kickoff"])
            typ = _dc_col(sched, ["game_type", "season_type", "type", "season_stage"])
            ascore = _dc_col(sched, ["away_score", "away_points", "away_pts"])
            hscore = _dc_col(sched, ["home_score", "home_points", "home_pts"])
            gid = _dc_col(sched, ["game_id", "gameid"])
            work = sched.copy()

            f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1])
            if season:
                seasons = sorted(
                    pd.to_numeric(work[season], errors="coerce").dropna().astype(int).unique(),
                    reverse=True,
                )
                chosen = f1.selectbox("Season", seasons, key="dc_sched_season")
                work = work[pd.to_numeric(work[season], errors="coerce").eq(chosen)]

            if week:
                # Numeric week ordering: 1, 2, 3 ... 18.
                week_values = work[week].dropna().astype(str).unique().tolist()

                def _week_sort_key(value):
                    try:
                        return (0, int(float(str(value))))
                    except Exception:
                        return (1, str(value))

                weeks = sorted(week_values, key=_week_sort_key)
                week_options = list(weeks) + ["All Weeks"]

                # WFS_DATA_CENTER_SCHEDULE_WEEK_V1
                # Initialize from the authoritative NFL planning week.
                # Manual Data Center week selection remains sticky afterward.
                from wfs_schedule_context import (
                    ScheduleContextError,
                    resolve_schedule_week_context,
                )

                try:
                    dc_schedule_context = resolve_schedule_week_context(
                        season=int(chosen) if season else None,
                    )
                    dc_default_week = str(dc_schedule_context.planning_week)
                except ScheduleContextError as exc:
                    if "No current or future REG week for " not in str(exc):
                        raise
                    dc_default_week = weeks[-1] if weeks else "All Weeks"

                if dc_default_week not in weeks:
                    dc_default_week = weeks[-1] if weeks else "All Weeks"

                if "dc_selected_week" not in st.session_state:
                    st.session_state["dc_selected_week"] = dc_default_week

                if st.session_state["dc_selected_week"] not in week_options:
                    st.session_state["dc_selected_week"] = dc_default_week

                chosen_week = f2.selectbox(
                    "Week",
                    week_options,
                    index=week_options.index(st.session_state["dc_selected_week"]),
                    key="dc_sched_week",
                )
                st.session_state["dc_selected_week"] = chosen_week

                if chosen_week != "All Weeks":
                    work = work[work[week].astype(str).eq(chosen_week)]

            # QA F4 / WFS_DATA_CENTER_MULTI_TEAM_FILTER_V1
            # QA F15: team filter tokens also accept full team names and
            # nicknames (e.g. "Falcons", "Atlanta Falcons"), resolved via
            # the existing NFL_TEAM_NAMES canonical mapping. Exact-code
            # behavior is unchanged.
            team_filter = f3.text_input(
                "Team filter", placeholder="BUF, PHI, KC...", key="dc_team"
            )
            team_codes = {
                _dc_resolve_team_token(token)
                for token in str(team_filter).split(",")
                if _dc_resolve_team_token(token)
            }
            if team_codes and away and home:
                away_codes = work[away].map(_dc_team_code)
                home_codes = work[home].map(_dc_team_code)
                work = work[
                    away_codes.isin(team_codes)
                    | home_codes.isin(team_codes)
                ]

            view = f4.selectbox("View", ["Matchup Cards", "Data Table"], key="dc_sched_view")

            if date:
                try:
                    work = work.assign(
                        __sort_date=pd.to_datetime(work[date], errors="coerce")
                    ).sort_values(["__sort_date", gid] if gid else ["__sort_date"])
                except Exception:
                    pass

            if view == "Matchup Cards":
                card_cols = {
                    "away": away, "home": home, "date": date, "week": week,
                    "away_score": ascore, "home_score": hscore, "gid": gid,
                }

                if favorite_team and away and home and len(work):
                    work = work.copy()
                    work["__favorite_game"] = work.apply(
                        lambda r: 1 if _dc_matchup_has_team(
                            r, card_cols, favorite_team
                        ) else 0,
                        axis=1,
                    )

                    sort_cols = ["__favorite_game"]
                    ascending = [False]

                    if "__sort_date" in work.columns:
                        sort_cols.append("__sort_date")
                        ascending.append(True)

                    work = work.sort_values(
                        sort_cols,
                        ascending=ascending,
                        kind="stable",
                    ).drop(columns="__favorite_game")

                rows = list(work.iterrows())
                for i in range(0, len(rows), 3):
                    grid = st.columns(3)
                    for j, (_, row) in enumerate(rows[i:i + 3]):
                        with grid[j]:
                            card_away = _dc_team_code(
                                row[card_cols["away"]]
                            )
                            card_home = _dc_team_code(
                                row[card_cols["home"]]
                            )
                            card_game_id = str(
                                row[card_cols["gid"]]
                            )
                            card_week = (
                                row[card_cols["week"]]
                                if card_cols.get("week")
                                else chosen_week
                            )
                            card_my_player_count = (
                                int(
                                    dc_my_players_by_team.get(
                                        card_away,
                                        0,
                                    )
                                )
                                +
                                int(
                                    dc_my_players_by_team.get(
                                        card_home,
                                        0,
                                    )
                                )
                            )

                            st.markdown(
                                _dc_matchup_card(
                                    row,
                                    card_cols,
                                    favorite_team=favorite_team,
                                    my_player_count=0,
                                ),
                                unsafe_allow_html=True,
                            )

                            # WFS_LIVE_GAME_AI_UI_V1
                            _dc_render_live_game_ai(
                                card_game_id,
                                workspace_mode=workspace_mode,
                            )

                            if (
                                card_my_player_count > 0
                                and dc_user
                            ):
                                player_word = (
                                    "PLAYER"
                                    if card_my_player_count == 1
                                    else "PLAYERS"
                                )

                                with st.expander(
                                    (
                                        f"⭐ {card_my_player_count} "
                                        f"MY {player_word}"
                                    ),
                                    expanded=False,
                                ):
                                    _dc_render_my_players_panel(
                                        dc_user["sub"],
                                        card_game_id,
                                        card_away,
                                        card_home,
                                        card_week,
                                    )
            else:
                display = pd.DataFrame(index=work.index)
                for label, col in [
                    ("Date", date), ("Type", typ), ("Week", week), ("Away", away),
                    ("Away Score", ascore), ("Home", home), ("Home Score", hscore),
                    ("Game ID", gid),
                ]:
                    if col:
                        display[label] = work[col]
                st.dataframe(
                    (display if not display.empty else work).reset_index(drop=True),
                    width="stretch", hide_index=True,
                )

            st.caption(f"{len(work):,} game(s) shown.")

    with box_tab:
        if sched.empty or box.empty:
            st.info("Box scores will appear when game results and player statistics are available.")
        else:
            sgid = _dc_col(sched, ["game_id", "gameid"])
            bgid = _dc_col(box, ["game_id", "gameid"])
            season = _dc_col(sched, ["season", "season_year", "year"])
            week = _dc_col(sched, ["week", "week_num", "week_number"])
            away = _dc_col(sched, ["away_team", "away"])
            home = _dc_col(sched, ["home_team", "home"])
            ascore = _dc_col(sched, ["away_score", "away_points", "away_pts"])
            hscore = _dc_col(sched, ["home_score", "home_points", "home_pts"])

            if not sgid or not bgid:
                st.info("Box scores are temporarily unavailable for the selected games.")
            else:
                games = sched.copy()
                f1, f2 = st.columns(2)

                if season:
                    seasons = sorted(
                        pd.to_numeric(games[season], errors="coerce").dropna().astype(int).unique(),
                        reverse=True,
                    )
                    chosen = f1.selectbox("Season", seasons, key="dc_box_season")
                    games = games[pd.to_numeric(games[season], errors="coerce").eq(chosen)]

                if week:
                    week_values = games[week].dropna().astype(str).unique().tolist()

                    def _box_week_sort_key(value):
                        try:
                            return (0, int(float(str(value))))
                        except Exception:
                            return (1, str(value))

                    weeks = sorted(week_values, key=_box_week_sort_key)
                    week_options = list(weeks) + ["All Weeks"]

                    # WFS_DATA_CENTER_BOX_SCORES_LAST_PLAYED_WEEK_V1
                    # Box Scores show completed games, so default to the
                    # most recent week that has final scores rather than
                    # the Schedule tab's forward-looking planning week.
                    try:
                        if ascore and hscore:
                            played_mask = (
                                pd.to_numeric(games[ascore], errors="coerce").notna()
                                & pd.to_numeric(games[hscore], errors="coerce").notna()
                            )
                        else:
                            played_mask = pd.Series(False, index=games.index)

                        played_weeks = (
                            games.loc[played_mask, week]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )
                        dc_default_week = (
                            sorted(played_weeks, key=_box_week_sort_key)[-1]
                            if played_weeks
                            else (weeks[-1] if weeks else "All Weeks")
                        )
                    except Exception:
                        dc_default_week = weeks[-1] if weeks else "All Weeks"

                    if dc_default_week not in weeks:
                        dc_default_week = weeks[-1] if weeks else "All Weeks"

                    if "dc_box_selected_week" not in st.session_state:
                        st.session_state["dc_box_selected_week"] = dc_default_week

                    if st.session_state["dc_box_selected_week"] not in week_options:
                        st.session_state["dc_box_selected_week"] = dc_default_week

                    chosen_week = f2.selectbox(
                        "Week",
                        week_options,
                        index=week_options.index(st.session_state["dc_box_selected_week"]),
                        key="dc_box_week",
                    )
                    st.session_state["dc_box_selected_week"] = chosen_week

                    if chosen_week != "All Weeks":
                        games = games[games[week].astype(str).eq(chosen_week)]

                labels = {}
                schedule_rows = {}
                for _, r in games.iterrows():
                    game_id = str(r[sgid])
                    av = str(r[away]) if away else "Away"
                    hv = str(r[home]) if home else "Home"
                    prefix = f"W{r[week]} • " if week else ""
                    score = ""
                    if ascore and hscore and pd.notna(r[ascore]) and pd.notna(r[hscore]):
                        score = f" • {av} {_dc_score_text(r[ascore])} — {hv} {_dc_score_text(r[hscore])}"
                    label = f"{prefix}{av} @ {hv}{score}"
                    labels[label] = game_id
                    schedule_rows[game_id] = r

                if labels:
                    label = st.selectbox("Select Game", list(labels), key="dc_game")
                    game_id = labels[label]
                    stats = box[box[bgid].astype(str).eq(game_id)].copy()
                    game_row = schedule_rows[game_id]

                    av = str(game_row[away]) if away else "Away"
                    hv = str(game_row[home]) if home else "Home"
                    avs = _dc_score_text(game_row[ascore]) if ascore else "—"
                    hvs = _dc_score_text(game_row[hscore]) if hscore else "—"
                    status = _dc_game_status(
                        game_row[ascore] if ascore else None,
                        game_row[hscore] if hscore else None,
                    )

                    winner_note = ""
                    try:
                        a_num = float(game_row[ascore]) if ascore and pd.notna(game_row[ascore]) else None
                        h_num = float(game_row[hscore]) if hscore and pd.notna(game_row[hscore]) else None
                        if a_num is not None and h_num is not None:
                            if a_num > h_num:
                                winner_note = f"{av} WIN"
                            elif h_num > a_num:
                                winner_note = f"{hv} WIN"
                            else:
                                winner_note = "TIE"
                    except Exception:
                        winner_note = ""

                    winner_html = (
                        f'<div class="wfs-box-winner">{winner_note}</div>'
                        if winner_note
                        else ""
                    )

                    away_box_logo = _dc_team_logo_html(
                        av, "wfs-box-team-logo"
                    )
                    home_box_logo = _dc_team_logo_html(
                        hv, "wfs-box-team-logo"
                    )

                    box_hero_html = (
                        f'<div class="wfs-box-hero">'
                        f'<div class="wfs-box-hero-row">'
                        f'<div>'
                        f'<div class="wfs-box-sub wfs-box-status">{status}</div>'
                        f'<div class="wfs-box-matchup">'
                        f'<span class="wfs-box-team">{away_box_logo}<span>{av} {avs}</span></span>'
                        f'<span class="wfs-box-vs">—</span>'
                        f'<span class="wfs-box-team">{home_box_logo}<span>{hv} {hvs}</span></span>'
                        f'</div>'
                        f'</div>'
                        f'{winner_html}'
                        f'</div>'
                        f'</div>'
                    )

                    st.markdown(
                        box_hero_html,
                        unsafe_allow_html=True,
                    )

                    if stats.empty:
                        st.info("Player statistics are not available for this game yet.")
                    else:
                        # QA D12: prefer the canonical full display name over the
                        # abbreviated source "player_name" (e.g. "Bi.Robinson"),
                        # matching the priority already used by the Weekly Leaders
                        # table below. Presentation only -- no identity change.
                        p = _dc_col(stats, ["player_display_name", "player_name", "player", "display_name", "fantasy_player_name"])
                        tm = _dc_col(stats, ["team", "recent_team", "posteam"])
                        pos = _dc_col(stats, ["position", "pos"])

                        passing = ["passing_attempts", "completions", "passing_yards",
                                   "passing_tds", "interceptions", "sacks"]
                        rushing = ["carries", "rushing_yards", "rushing_tds"]
                        receiving = ["targets", "receptions", "receiving_yards", "receiving_tds"]
                        fantasy = ["fumbles_lost", "fantasy_points", "fantasy_points_ppr",
                                   "fd_points", "fanduel_points"]

                        identity = [p, tm, pos]

                        passing = [
                            "passing_attempts", "attempts", "completions",
                            "passing_yards", "passing_tds", "interceptions", "sacks",
                        ]
                        rushing = [
                            "carries", "rushing_yards", "rushing_tds",
                        ]
                        receiving = [
                            "targets", "receptions", "receiving_yards", "receiving_tds",
                        ]
                        fantasy = [
                            "fumbles_lost", "fd_points", "fanduel_points",
                            "fantasy_points", "fantasy_points_ppr",
                        ]

                        t1, t2, t3, t4, t5 = st.tabs(
                            ["⭐ Leaders", "Passing", "Rushing", "Receiving", "Fantasy"]
                        )

                        with t1:
                            lead_a, lead_b, lead_c = st.columns(3)

                            def leader_card(container, title, candidates):
                                sort_col = None
                                for candidate in candidates:
                                    found = _dc_col(stats, [candidate])
                                    if found:
                                        sort_col = found
                                        break
                                if not sort_col or not p:
                                    container.metric(title, "—")
                                    return
                                values = pd.to_numeric(stats[sort_col], errors="coerce")
                                if values.notna().any():
                                    idx = values.idxmax()
                                    player_name = str(stats.loc[idx, p])
                                    value = values.loc[idx]
                                    shown = int(value) if float(value).is_integer() else round(float(value), 2)
                                    # QA D11: this is a static leader value, not a
                                    # change/delta, so suppress the arrow and color
                                    # Streamlit otherwise infers from a numeric delta.
                                    container.metric(
                                        title,
                                        player_name,
                                        str(shown),
                                        delta_color="off",
                                        delta_arrow="off",
                                    )
                                else:
                                    container.metric(title, "—")

                            leader_card(lead_a, "Passing Leader", ["passing_yards"])
                            leader_card(lead_b, "Rushing Leader", ["rushing_yards"])
                            leader_card(lead_c, "Receiving Leader", ["receiving_yards"])

                            st.markdown("#### Fantasy Leaders")
                            _dc_render_stat_table(
                                stats, identity, fantasy,
                                ["fd_points", "fanduel_points", "fantasy_points", "fantasy_points_ppr"],
                            )

                        with t2:
                            _dc_render_stat_table(
                                stats, identity, passing,
                                ["passing_yards", "passing_tds"],
                            )
                        with t3:
                            _dc_render_stat_table(
                                stats, identity, rushing,
                                ["rushing_yards", "carries"],
                            )
                        with t4:
                            _dc_render_stat_table(
                                stats, identity, receiving,
                                ["receiving_yards", "targets"],
                            )
                        with t5:
                            _dc_render_stat_table(
                                stats, identity, fantasy,
                                ["fd_points", "fanduel_points", "fantasy_points", "fantasy_points_ppr"],
                            )

                        st.caption(
                            f"{len(stats):,} player performances shown."
                        )
                else:
                    st.info("No games match the selected filters.")

    # WFS_WEEKLY_NFL_LEADERS_V1
    # League-wide completed weekly player leaders from the same authoritative
    # player_game_stats dataframe already loaded by the NFL Data Center.
    with leaders_tab:
        if box.empty:
            st.info("Weekly leaders will appear when completed player statistics are available.")
        else:
            leaders = box.copy()

            lseason = _dc_col(leaders, ["season", "season_year", "year"])
            lweek = _dc_col(leaders, ["week", "week_num", "week_number"])
            ltype = _dc_col(leaders, ["season_type", "game_type", "type"])
            lname = _dc_col(
                leaders,
                ["player_display_name", "player_name", "player", "display_name",
                 "fantasy_player_name"],
            )
            lteam = _dc_col(leaders, ["team", "recent_team", "posteam"])

            if not lseason or not lweek or not lname or not lteam:
                st.info("Weekly leader data is temporarily unavailable.")
            else:
                if ltype:
                    season_type = leaders[ltype].fillna("").astype(str).str.upper()
                    regular_mask = season_type.isin(["REG", "REGULAR", "REGULAR SEASON"])
                    if regular_mask.any():
                        leaders = leaders[regular_mask].copy()

                season_numeric = pd.to_numeric(leaders[lseason], errors="coerce")
                available_seasons = sorted(
                    season_numeric.dropna().astype(int).unique().tolist(),
                    reverse=True,
                )

                if not available_seasons:
                    st.info("No completed NFL weeks are available yet.")
                else:
                    control_a, control_b = st.columns(2)

                    with control_a:
                        selected_leader_season = st.selectbox(
                            "Season",
                            available_seasons,
                            index=0,
                            key="dc_weekly_leaders_season",
                        )

                    season_rows = leaders[
                        season_numeric.eq(int(selected_leader_season))
                    ].copy()

                    week_numeric = pd.to_numeric(
                        season_rows[lweek], errors="coerce"
                    )
                    available_weeks = sorted(
                        week_numeric.dropna().astype(int).unique().tolist(),
                        reverse=True,
                    )

                    with control_b:
                        selected_leader_week = st.selectbox(
                            "Week",
                            available_weeks,
                            index=0,
                            key="dc_weekly_leaders_week",
                        )

                    week_rows = season_rows[
                        week_numeric.eq(int(selected_leader_week))
                    ].copy()

                    st.caption(
                        f"Top individual performances • "
                        f"{int(selected_leader_season)} Week "
                        f"{int(selected_leader_week)}"
                    )

                    def _leader_num(frame, names):
                        col = _dc_col(frame, names)
                        if not col:
                            return pd.Series(0.0, index=frame.index, dtype=float)
                        return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

                    def _leader_logo(team):
                        return _dc_team_logo_html(
                            _dc_team_code(team),
                            "wfs-leader-team-logo",
                        )

                    def _leader_player_html(row):
                        team = _dc_team_code(row.get(lteam, ""))
                        name = str(row.get(lname, "") or "").strip()
                        logo = _leader_logo(team)
                        return (
                            '<div class="wfs-leader-player">'
                            f'{logo}'
                            '<div>'
                            f'<div class="wfs-leader-name">{name}</div>'
                            f'<div class="wfs-leader-team">{team}</div>'
                            '</div>'
                            '</div>'
                        )

                    def _fmt_int(value):
                        try:
                            return f"{int(round(float(value))):,}"
                        except Exception:
                            return "—"

                    def _fmt_one(value):
                        try:
                            return f"{float(value):.1f}"
                        except Exception:
                            return "—"

                    def _fmt_pct(value):
                        try:
                            return f"{float(value):.1f}%"
                        except Exception:
                            return "—"

                    def _render_weekly_leader_table(
                        frame,
                        columns,
                        sort_key,
                        limit=10,
                    ):
                        if frame.empty:
                            st.info("No qualifying performances are available.")
                            return

                        shown = (
                            frame.sort_values(
                                sort_key,
                                ascending=False,
                                kind="stable",
                            )
                            .head(limit)
                            .copy()
                        )

                        header = "".join(
                            f'<div class="wfs-leader-th">{label}</div>'
                            for label, _, _ in columns
                        )

                        rows_html = []
                        for rank, (_, row) in enumerate(shown.iterrows(), start=1):
                            cells = []
                            for _, field, formatter in columns:
                                if field == "__player__":
                                    value = _leader_player_html(row)
                                else:
                                    value = formatter(row.get(field, 0))
                                cells.append(
                                    f'<div class="wfs-leader-td">{value}</div>'
                                )

                            rows_html.append(
                                '<div class="wfs-leader-row">'
                                f'<div class="wfs-leader-rank">{rank}</div>'
                                + "".join(cells)
                                + '</div>'
                            )

                        st.markdown(
                            '<div class="wfs-leader-table">'
                            '<div class="wfs-leader-header">'
                            '<div class="wfs-leader-rank">#</div>'
                            f'{header}'
                            '</div>'
                            + "".join(rows_html)
                            + '</div>',
                            unsafe_allow_html=True,
                        )

                    passing = week_rows.copy()
                    passing["_attempts"] = _leader_num(
                        passing, ["attempts", "passing_attempts"]
                    )
                    passing["_completions"] = _leader_num(
                        passing, ["completions", "passing_completions"]
                    )
                    passing["_pass_yards"] = _leader_num(
                        passing, ["passing_yards"]
                    )
                    passing["_pass_tds"] = _leader_num(
                        passing, ["passing_tds"]
                    )
                    passing["_ints"] = _leader_num(
                        passing, ["interceptions", "passing_interceptions"]
                    )
                    passing["_fd"] = _leader_num(
                        passing, ["fanduel_points", "fd_points"]
                    )
                    passing = passing[passing["_attempts"] > 0].copy()
                    passing["_comp_pct"] = (
                        passing["_completions"]
                        .div(passing["_attempts"])
                        .mul(100.0)
                    )

                    rushing = week_rows.copy()
                    rushing["_carries"] = _leader_num(
                        rushing, ["carries", "rushing_attempts"]
                    )
                    rushing["_rush_yards"] = _leader_num(
                        rushing, ["rushing_yards"]
                    )
                    rushing["_rush_tds"] = _leader_num(
                        rushing, ["rushing_tds"]
                    )
                    rushing["_fd"] = _leader_num(
                        rushing, ["fanduel_points", "fd_points"]
                    )
                    rushing = rushing[rushing["_carries"] > 0].copy()
                    rushing["_ypc"] = rushing["_rush_yards"].div(
                        rushing["_carries"]
                    )

                    receiving = week_rows.copy()
                    receiving["_targets"] = _leader_num(
                        receiving, ["targets"]
                    )
                    receiving["_receptions"] = _leader_num(
                        receiving, ["receptions"]
                    )
                    receiving["_rec_yards"] = _leader_num(
                        receiving, ["receiving_yards"]
                    )
                    receiving["_rec_tds"] = _leader_num(
                        receiving, ["receiving_tds"]
                    )
                    receiving["_fd"] = _leader_num(
                        receiving, ["fanduel_points", "fd_points"]
                    )
                    receiving = receiving[
                        (receiving["_targets"] > 0)
                        | (receiving["_receptions"] > 0)
                    ].copy()

                    st.markdown(
                        """
                        <style>
                        .wfs-leader-table {
                            width:100%;
                            overflow-x:auto;
                            border:1px solid #dbe5f1;
                            border-radius:14px;
                            background:#fff;
                            margin:.45rem 0 1rem;
                        }
                        .wfs-leader-header,
                        .wfs-leader-row {
                            display:grid;
                            grid-template-columns:38px minmax(190px,2fr)
                                repeat(6,minmax(76px,1fr));
                            align-items:center;
                            min-width:780px;
                        }
                        .wfs-leader-header {
                            background:#f5f8fc;
                            color:#64748b;
                            font-size:.67rem;
                            font-weight:900;
                            letter-spacing:.035em;
                            border-bottom:1px solid #dbe5f1;
                        }
                        .wfs-leader-row {
                            border-bottom:1px solid #edf2f7;
                            color:#17365d;
                        }
                        .wfs-leader-row:last-child {border-bottom:none;}
                        .wfs-leader-rank,
                        .wfs-leader-th,
                        .wfs-leader-td {
                            padding:.58rem .48rem;
                        }
                        .wfs-leader-rank {
                            text-align:center;
                            font-weight:900;
                            color:#7b8ca3;
                        }
                        .wfs-leader-player {
                            display:flex;
                            align-items:center;
                            gap:.55rem;
                            min-width:0;
                        }
                        .wfs-leader-team-logo {
                            width:26px;
                            height:26px;
                            object-fit:contain;
                            flex:0 0 26px;
                        }
                        .wfs-leader-name {
                            font-size:.82rem;
                            font-weight:900;
                            color:#17365d;
                            white-space:nowrap;
                        }
                        .wfs-leader-team {
                            color:#7b8ca3;
                            font-size:.64rem;
                            font-weight:800;
                        }
                        .wfs-leader-td {
                            font-size:.76rem;
                            font-weight:800;
                        }
                        @media (max-width: 700px) {
                            .wfs-leader-header,
                            .wfs-leader-row {
                                min-width:740px;
                            }
                            .wfs-leader-team-logo {
                                width:24px;
                                height:24px;
                                flex-basis:24px;
                            }
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    pass_tab, rush_tab, receive_tab = st.tabs(
                        ["🏈 Passing", "🏃 Rushing", "🎯 Receiving"]
                    )

                    with pass_tab:
                        _render_weekly_leader_table(
                            passing,
                            [
                                ("Player", "__player__", str),
                                ("Pass Yds", "_pass_yards", _fmt_int),
                                ("Pass TD", "_pass_tds", _fmt_int),
                                ("INT", "_ints", _fmt_int),
                                ("Comp %", "_comp_pct", _fmt_pct),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_pass_yards",
                        )

                    with rush_tab:
                        _render_weekly_leader_table(
                            rushing,
                            [
                                ("Player", "__player__", str),
                                ("Att", "_carries", _fmt_int),
                                ("Rush Yds", "_rush_yards", _fmt_int),
                                ("Rush TD", "_rush_tds", _fmt_int),
                                ("YPC", "_ypc", _fmt_one),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_rush_yards",
                        )

                    with receive_tab:
                        _render_weekly_leader_table(
                            receiving,
                            [
                                ("Player", "__player__", str),
                                ("Targets", "_targets", _fmt_int),
                                ("Rec", "_receptions", _fmt_int),
                                ("Rec Yds", "_rec_yards", _fmt_int),
                                ("Rec TD", "_rec_tds", _fmt_int),
                                ("FD Pts", "_fd", _fmt_one),
                            ],
                            "_rec_yards",
                        )

SLEEPER_API_BASE = "https://api.sleeper.app/v1"


@st.cache_data(ttl=300, show_spinner=False)
def sleeper_get(path: str):
    url = f"{SLEEPER_API_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WynnersFantasySpot/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"Sleeper returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Sleeper: {exc.reason}") from exc


@st.cache_data(ttl=86400, show_spinner=False)
def sleeper_players():
    data = sleeper_get("/players/nfl")
    return data if isinstance(data, dict) else {}


# WFS_SLEEPER_FANTASY_PROVIDER_V1
# Read-only Sleeper Fantasy provider layer.
# Exact Sleeper IDs only. No fuzzy identity resolution.
# This layer does not modify ESPN Fantasy behavior.

def _sleeper_user(username_or_id):
    value = str(username_or_id or "").strip()
    if not value:
        return None

    # Sleeper supports user lookup by username and by user ID.
    user = sleeper_get(f"/user/{urllib.parse.quote(value, safe='')}")
    if isinstance(user, dict) and user.get("user_id"):
        return user

    return None


def _sleeper_user_leagues(user_id, season):
    user_id = str(user_id or "").strip()
    if not user_id:
        return []

    data = sleeper_get(
        f"/user/{urllib.parse.quote(user_id, safe='')}/leagues/nfl/{int(season)}"
    )
    return data if isinstance(data, list) else []


def _sleeper_league(league_id):
    data = sleeper_get(
        f"/league/{urllib.parse.quote(str(league_id or '').strip(), safe='')}"
    )
    return data if isinstance(data, dict) else None


def _sleeper_league_users(league_id):
    data = sleeper_get(
        f"/league/{urllib.parse.quote(str(league_id or '').strip(), safe='')}/users"
    )
    return data if isinstance(data, list) else []


def _sleeper_league_rosters(league_id):
    data = sleeper_get(
        f"/league/{urllib.parse.quote(str(league_id or '').strip(), safe='')}/rosters"
    )
    return data if isinstance(data, list) else []


def _sleeper_league_matchups(league_id, week):
    data = sleeper_get(
        f"/league/{urllib.parse.quote(str(league_id or '').strip(), safe='')}"
        f"/matchups/{int(week)}"
    )
    return data if isinstance(data, list) else []


def _sleeper_roster_for_user(rosters, user_id):
    user_id = str(user_id or "").strip()
    matches = [
        roster
        for roster in (rosters or [])
        if isinstance(roster, dict)
        and str(roster.get("owner_id") or "").strip() == user_id
    ]
    return matches[0] if len(matches) == 1 else None


def _wfs_sleeper_connections(user_sub, season):
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, provider, season, league_id, league_name, team_id,
                   team_name, connection_type, created_at, updated_at
            FROM wfs_fantasy_connections
            WHERE user_sub = ? AND provider = 'SLEEPER' AND season = ?
            ORDER BY lower(league_name), league_id
            """,
            (str(user_sub), int(season)),
        ).fetchall()
    return [dict(row) for row in rows]


def _wfs_sleeper_upsert_connection(
    user_sub,
    season,
    league_id,
    league_name,
    team_id=None,
    team_name="",
):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_fantasy_connections(
                user_sub, provider, season, league_id, league_name,
                team_id, team_name, connection_type
            ) VALUES (?, 'SLEEPER', ?, ?, ?, ?, ?, 'PUBLIC')
            ON CONFLICT(user_sub, provider, season, league_id) DO UPDATE SET
                league_name=excluded.league_name,
                team_id=COALESCE(
                    excluded.team_id,
                    wfs_fantasy_connections.team_id
                ),
                team_name=CASE
                    WHEN excluded.team_id IS NOT NULL
                    THEN excluded.team_name
                    ELSE wfs_fantasy_connections.team_name
                END,
                connection_type='PUBLIC',
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(user_sub),
                int(season),
                str(league_id),
                str(league_name or ""),
                int(team_id) if team_id is not None else None,
                str(team_name or ""),
            ),
        )


def _wfs_sleeper_delete_connection(user_sub, season, league_id):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            DELETE FROM wfs_fantasy_connections
            WHERE user_sub = ?
              AND provider = 'SLEEPER'
              AND season = ?
              AND league_id = ?
            """,
            (str(user_sub), int(season), str(league_id)),
        )



ESPN_FANTASY_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
ESPN_FAN_PROFILE_BASE = "https://fan.api.espn.com/apis/v2/fans"


def _espn_secret_credentials():
    """
    Read private ESPN browser-session credentials from Streamlit secrets.
    Nothing is collected in the public WFS UI and values are never displayed.
    Expected:
      [espn]
      swid = "{...}"
      espn_s2 = "..."
    """
    try:
        section = st.secrets.get("espn", {})
        swid = str(section.get("swid", "") or "").strip()
        espn_s2 = str(section.get("espn_s2", "") or "").strip()
    except Exception:
        return "", ""
    return swid, espn_s2


def _espn_cookie_header(swid="", espn_s2=""):
    if not swid or not espn_s2:
        return ""
    return f"espn_s2={espn_s2}; SWID={swid}"


# QA S1 / WFS_ESPN_CREDENTIAL_RUNTIME_HARDENING_V1
# Authenticated ESPN requests intentionally bypass Streamlit data caching so
# decrypted browser-session credentials do not participate in cache keys/state.
def espn_league_get(league_id: str, season: int, views, swid="", espn_s2=""):
    league_id = str(league_id).strip()
    if not league_id.isdigit():
        return {"ok": False, "status": "INVALID", "message": "League ID must contain numbers only."}

    query = urllib.parse.urlencode([("view", v) for v in views])
    url = (
        f"{ESPN_FANTASY_BASE}/seasons/{int(season)}/segments/0/"
        f"leagues/{league_id}?{query}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 WFS-Fantasy-League-Hub/1.1",
        "Accept": "application/json,text/plain,*/*",
    }
    cookie = _espn_cookie_header(swid, espn_s2)
    if cookie:
        headers["Cookie"] = cookie

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "ok": True,
                "status": "PRIVATE_AUTH" if cookie else "PUBLIC",
                "data": payload,
            }
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {
                "ok": False,
                "status": "PRIVATE",
                "message": (
                    "This league is private. Use the private-league "
                    "setup above to connect it."
                ),
            }
        if exc.code in (400, 404):
            return {
                "ok": False,
                "status": "NOT_FOUND",
                "message": (
                    "We couldn't find that ESPN league. Double-check the "
                    "League ID. If it's a private league, use the "
                    "private-league setup above."
                ),
            }
        return {
            "ok": False,
            "status": "ERROR",
            "message": "ESPN didn't respond. Try again in a minute.",
        }
    except urllib.error.URLError:
        return {
            "ok": False,
            "status": "ERROR",
            "message": "ESPN didn't respond. Try again in a minute.",
        }
    except Exception:
        return {
            "ok": False,
            "status": "ERROR",
            "message": "ESPN didn't respond. Try again in a minute.",
        }


def espn_fan_profile(swid: str, espn_s2: str):
    """
    Read the signed-in ESPN fan profile. ESPN does not publish a stable developer
    contract for this response, so discovery below intentionally fails closed if
    no recognizable fantasy-football league IDs are found.
    """
    if not swid or not espn_s2:
        return {"ok": False, "status": "NO_AUTH", "message": "ESPN session credentials are not configured."}

    encoded_swid = urllib.parse.quote(swid, safe="")
    url = f"{ESPN_FAN_PROFILE_BASE}/{encoded_swid}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WFS-Fantasy-League-Hub/1.1",
            "Accept": "application/json,text/plain,*/*",
            "Cookie": _espn_cookie_header(swid, espn_s2),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return {"ok": True, "status": "AUTHENTICATED", "data": json.loads(response.read().decode("utf-8"))}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {
                "ok": False,
                "status": "AUTH_EXPIRED",
                "message": "The saved ESPN browser session is missing, expired, or no longer authorized.",
            }
        return {"ok": False, "status": "ERROR", "message": f"ESPN profile request returned HTTP {exc.code}."}
    except urllib.error.URLError as exc:
        return {"ok": False, "status": "ERROR", "message": f"Could not reach ESPN profile service: {exc.reason}"}
    except Exception as exc:
        return {
            "ok": False,
            "status": "ERROR",
            "message": "ESPN Fantasy could not load the account profile.",
        }


def _espn_recursive_league_candidates(obj, season):
    """
    ESPN's fan-profile shape is unofficial and can drift. Walk it conservatively
    and collect only objects containing a numeric-looking league identifier.
    """
    found = []

    def walk(node, path=""):
        if isinstance(node, dict):
            lower = {str(k).lower(): k for k in node}
            league_key = next(
                (lower[k] for k in ("leagueid", "league_id", "leagueidvalue") if k in lower),
                None,
            )
            if league_key is not None:
                raw_id = node.get(league_key)
                lid = str(raw_id or "").strip()
                blob = " ".join(str(v) for v in node.values() if isinstance(v, (str, int, float))).lower()
                year_values = []
                for key in ("seasonid", "season", "year"):
                    if key in lower:
                        try:
                            year_values.append(int(node.get(lower[key])))
                        except Exception:
                            pass
                footballish = any(x in blob for x in ("ffl", "football", "fantasy football"))
                seasonish = (not year_values) or int(season) in year_values
                if lid.isdigit() and seasonish and (footballish or "league" in path.lower()):
                    name = ""
                    team_name = ""
                    for key in ("leaguename", "league_name", "name"):
                        if key in lower and node.get(lower[key]):
                            name = str(node.get(lower[key])).strip()
                            break
                    for key in ("teamname", "team_name", "entryname", "entry_name"):
                        if key in lower and node.get(lower[key]):
                            team_name = str(node.get(lower[key])).strip()
                            break
                    found.append({"league_id": lid, "name": name, "team_name": team_name})
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}/{i}")

    walk(obj)

    unique = {}
    for item in found:
        unique.setdefault(item["league_id"], item)
        if not unique[item["league_id"]].get("name") and item.get("name"):
            unique[item["league_id"]]["name"] = item["name"]
        if not unique[item["league_id"]].get("team_name") and item.get("team_name"):
            unique[item["league_id"]]["team_name"] = item["team_name"]
    return list(unique.values())


def _espn_team_name(team):
    if not isinstance(team, dict):
        return "Team"
    location = str(team.get("location") or "").strip()
    nickname = str(team.get("nickname") or "").strip()
    full = f"{location} {nickname}".strip()
    return full or team.get("name") or team.get("abbrev") or f"Team {team.get('id', '?')}"


def _espn_record(team):
    overall = ((team.get("record") or {}).get("overall") or {})
    return (
        int(overall.get("wins", 0) or 0),
        int(overall.get("losses", 0) or 0),
        int(overall.get("ties", 0) or 0),
        float(overall.get("pointsFor", 0) or 0),
        float(overall.get("pointsAgainst", 0) or 0),
    )


def _espn_validate_discovered_leagues(candidates, season, swid, espn_s2):
    """
    Validate every discovered ID against ESPN's league endpoint so the UI never
    trusts an ambiguous fan-profile object by itself.
    """
    valid = []
    seen = set()
    for item in candidates:
        lid = str(item.get("league_id") or "")
        if not lid or lid in seen:
            continue
        seen.add(lid)
        result = espn_league_get(
            lid, season, ["mSettings", "mTeam", "mStandings"], swid, espn_s2
        )
        if not result.get("ok"):
            continue
        data = result.get("data") or {}
        settings = data.get("settings") or {}
        name = settings.get("name") or item.get("name") or f"ESPN League {lid}"
        valid.append({
            "league_id": lid,
            "name": name,
            "team_name": item.get("team_name") or "",
            "team_count": len(data.get("teams") or []),
        })
    return valid



def _espn_owner_display(team, member_by_id):
    owners = team.get("owners") or team.get("primaryOwner")
    if isinstance(owners, str):
        owners = [owners]
    if not owners:
        return ""
    for owner_id in owners:
        member = member_by_id.get(str(owner_id), {})
        name = (
            member.get("displayName")
            or member.get("firstName")
            or member.get("lastName")
        )
        if name:
            return str(name)
    return ""


def _espn_my_team_id(data, swid):
    """
    Resolve the signed-in user's team without fuzzy matching.
    Prefer member identity via SWID; then exact owner-id membership.
    """
    members = data.get("members") or []
    teams = data.get("teams") or []
    clean_swid = str(swid or "").strip().strip("{}").lower()

    member_ids = set()
    for member in members:
        mid = str(member.get("id") or "").strip()
        muid = str(member.get("uuid") or "").strip()
        mswid = str(member.get("swid") or "").strip().strip("{}").lower()
        if clean_swid and mswid == clean_swid:
            if mid:
                member_ids.add(mid)
            if muid:
                member_ids.add(muid)

    for team in teams:
        owners = team.get("owners") or []
        if isinstance(owners, str):
            owners = [owners]
        primary = team.get("primaryOwner")
        if primary:
            owners = list(owners) + [primary]
        if member_ids.intersection({str(x) for x in owners}):
            return team.get("id")
    return None


def _espn_find_matchup(schedule, team_id, week):
    if team_id is None:
        return None
    for game in schedule:
        if int(game.get("matchupPeriodId") or -1) != int(week):
            continue
        away = game.get("away") or {}
        home = game.get("home") or {}
        if away.get("teamId") == team_id or home.get("teamId") == team_id:
            return game
    return None


def _espn_side_score(side, week=None):
    side = side or {}

    try:
        total = float(side.get("totalPoints") or 0)
    except Exception:
        total = 0.0

    if total != 0:
        return total

    roster = (
        side.get("rosterForCurrentScoringPeriod")
        or side.get("roster")
        or {}
    )
    entries = roster.get("entries") or []

    fallback = 0.0
    found = False

    for entry in entries:
        try:
            slot = int(entry.get("lineupSlotId"))
        except Exception:
            slot = None

        # ESPN bench / IR do not count toward matchup score.
        if slot in {20, 21}:
            continue

        player = (
            (entry.get("playerPoolEntry") or {})
            .get("player")
            or {}
        )

        for stat in player.get("stats") or []:
            try:
                source_id = int(
                    stat.get("statSourceId", 0)
                )
            except Exception:
                continue

            if source_id != 0:
                continue

            if week is not None:
                try:
                    if int(
                        stat.get("scoringPeriodId")
                    ) != int(week):
                        continue
                except Exception:
                    continue

            value = stat.get("appliedTotal")
            if value is None:
                continue

            try:
                fallback += float(value)
                found = True
                break
            except Exception:
                continue

    return fallback if found else total


def _espn_side_projection(side):
    """
    ESPN response shapes vary. Use only explicit aggregate projected-point fields
    when present; never manufacture a projection from current points.
    """
    side = side or {}
    for key in ("totalProjectedPoints", "projectedPoints", "projectedTotalPoints"):
        if side.get(key) is not None:
            try:
                return float(side.get(key))
            except Exception:
                pass
    return None


def _espn_lineup_slot_name(slot_id):
    labels = {
        0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K",
        20: "Bench", 21: "IR", 23: "FLEX",
    }
    try:
        return labels.get(int(slot_id), str(slot_id))
    except Exception:
        return str(slot_id)



WFS_ESPN_TEAM_MAP_PATH = Path(__file__).resolve().parent / "data" / "wfs_espn_my_teams.json"


def _espn_team_map_load():
    try:
        if WFS_ESPN_TEAM_MAP_PATH.exists():
            raw = json.loads(WFS_ESPN_TEAM_MAP_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except Exception:
        pass
    return {}


def _espn_team_map_save(mapping):
    WFS_ESPN_TEAM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = WFS_ESPN_TEAM_MAP_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(WFS_ESPN_TEAM_MAP_PATH)


def _espn_team_map_key(season, league_id):
    return f"{int(season)}:{str(league_id).strip()}"



# ---------------------------------------------------------------------------
# WFS MULTI-USER ACCOUNT + ESPN CONNECTION LAYER v1
# Google OIDC `sub` is the stable WFS user key. ESPN browser-session values are
# encrypted at rest with a server-only Fernet key from Streamlit secrets.
# ---------------------------------------------------------------------------

def _wfs_current_user():
    """Return the authenticated WFS identity. Never use email as the primary key."""
    if not getattr(st.user, "is_logged_in", False):
        return None
    sub = str(st.user.get("sub") or "").strip()
    if not sub:
        return None
    return {
        "sub": sub,
        "name": str(st.user.get("name") or "WFS User").strip(),
        "email": str(st.user.get("email") or "").strip(),
    }


def _wfs_accounts_connect():
    WFS_ACCOUNTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        WFS_ACCOUNTS_DB_PATH,
        timeout=10.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _wfs_accounts_init():
    with _wfs_accounts_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS wfs_users (
                user_sub TEXT PRIMARY KEY,
                email TEXT NOT NULL DEFAULT '',
                display_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS wfs_user_preferences (
                user_sub TEXT PRIMARY KEY,
                favorite_nfl_team TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS wfs_user_navigation_state (
                user_sub TEXT PRIMARY KEY,
                last_page TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS wfs_user_portfolio_seen (
                user_sub TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                seen_signatures_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_sub, cache_key),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_user_portfolio_seen_user
            ON wfs_user_portfolio_seen(user_sub, updated_at);

            CREATE TABLE IF NOT EXISTS wfs_espn_credentials (
                user_sub TEXT PRIMARY KEY,
                swid_encrypted TEXT NOT NULL,
                espn_s2_encrypted TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS wfs_fantasy_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_sub TEXT NOT NULL,
                provider TEXT NOT NULL,
                season INTEGER NOT NULL,
                league_id TEXT NOT NULL,
                league_name TEXT NOT NULL DEFAULT '',
                team_id INTEGER,
                team_name TEXT NOT NULL DEFAULT '',
                connection_type TEXT NOT NULL DEFAULT 'PUBLIC',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_sub, provider, season, league_id),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_fantasy_connections_user
            ON wfs_fantasy_connections(user_sub, provider, season);

            CREATE TABLE IF NOT EXISTS wfs_player_watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_sub TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL DEFAULT '',
                position TEXT NOT NULL DEFAULT '',
                watch_type TEXT NOT NULL DEFAULT 'Monitor',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_sub, player_key),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_player_watchlist_user
            ON wfs_player_watchlist(user_sub, created_at);

            CREATE TABLE IF NOT EXISTS wfs_watchlist_state (
                user_sub TEXT NOT NULL,
                player_key TEXT NOT NULL,
                baseline_projection REAL,
                baseline_status TEXT NOT NULL DEFAULT '',
                baseline_team TEXT NOT NULL DEFAULT '',
                observed_projection REAL,
                observed_status TEXT NOT NULL DEFAULT '',
                observed_team TEXT NOT NULL DEFAULT '',
                changed_at TEXT,
                reviewed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_sub, player_key),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_watchlist_state_user
            ON wfs_watchlist_state(user_sub, updated_at);

            CREATE TABLE IF NOT EXISTS wfs_watchlist_projection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_sub TEXT NOT NULL,
                player_key TEXT NOT NULL,
                projection REAL,
                production_status TEXT NOT NULL DEFAULT '',
                team TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_watchlist_projection_history
            ON wfs_watchlist_projection_history(user_sub, player_key, observed_at);

            CREATE TABLE IF NOT EXISTS wfs_projection_mover_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_key TEXT NOT NULL,
                player_name TEXT NOT NULL DEFAULT '',
                position TEXT NOT NULL DEFAULT '',
                team TEXT NOT NULL DEFAULT '',
                projection REAL,
                production_status TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_projection_mover_history
            ON wfs_projection_mover_history(player_key, observed_at);

            CREATE TABLE IF NOT EXISTS wfs_weekly_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_sub TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT 'ESPN',
                season INTEGER NOT NULL,
                league_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                decision_key TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                sit_player TEXT NOT NULL DEFAULT '',
                start_player TEXT NOT NULL DEFAULT '',
                sit_projection REAL,
                start_projection REAL,
                projected_gain REAL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_sub, provider, season, league_id, week, decision_key),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_wfs_weekly_decisions_user_week
            ON wfs_weekly_decisions(user_sub, season, league_id, week);
            """
        )

        existing_decision_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(wfs_weekly_decisions)").fetchall()
        }
        for col_name, col_sql in (
            ("sit_actual", "REAL"),
            ("start_actual", "REAL"),
            ("actual_gain", "REAL"),
            ("graded_at", "TEXT"),
        ):
            if col_name not in existing_decision_cols:
                conn.execute(f"ALTER TABLE wfs_weekly_decisions ADD COLUMN {col_name} {col_sql}")


def _wfs_upsert_user(user):
    if not user:
        return
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_users(user_sub, email, display_name)
            VALUES (?, ?, ?)
            ON CONFLICT(user_sub) DO UPDATE SET
                email=excluded.email,
                display_name=excluded.display_name,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user["sub"], user.get("email", ""), user.get("name", "")),
        )


def _wfs_get_last_navigation(user_sub):
    if not user_sub:
        return ""
    with _wfs_accounts_connect() as conn:
        row = conn.execute(
            """
            SELECT last_page
            FROM wfs_user_navigation_state
            WHERE user_sub = ?
            """,
            (user_sub,),
        ).fetchone()
    return str(row["last_page"] or "").strip() if row else ""


def _wfs_set_last_navigation(user_sub, page):
    if not user_sub or not page:
        return
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_user_navigation_state(user_sub, last_page)
            VALUES (?, ?)
            ON CONFLICT(user_sub) DO UPDATE SET
                last_page=excluded.last_page,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_sub, str(page)),
        )


def _wfs_get_seen_portfolio_signatures(user_sub, cache_key):
    if not user_sub or not cache_key:
        return []

    with _wfs_accounts_connect() as conn:
        row = conn.execute(
            """
            SELECT seen_signatures_json
            FROM wfs_user_portfolio_seen
            WHERE user_sub = ? AND cache_key = ?
            """,
            (user_sub, cache_key),
        ).fetchone()

    if not row:
        return []

    try:
        values = json.loads(row["seen_signatures_json"] or "[]")
    except Exception:
        return []

    if not isinstance(values, list):
        return []

    return sorted({
        str(value).strip()
        for value in values
        if str(value).strip()
    })


def _wfs_set_seen_portfolio_signatures(user_sub, cache_key, signatures):
    if not user_sub or not cache_key:
        return

    clean = sorted({
        str(value).strip()
        for value in (signatures or [])
        if str(value).strip()
    })

    payload = json.dumps(clean, separators=(",", ":"))

    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_user_portfolio_seen(
                user_sub,
                cache_key,
                seen_signatures_json
            )
            VALUES (?, ?, ?)
            ON CONFLICT(user_sub, cache_key) DO UPDATE SET
                seen_signatures_json=excluded.seen_signatures_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_sub, cache_key, payload),
        )


def _wfs_user_preferences(user_sub):
    if not user_sub:
        return {"favorite_nfl_team": ""}

    with _wfs_accounts_connect() as conn:
        row = conn.execute(
            """
            SELECT favorite_nfl_team
            FROM wfs_user_preferences
            WHERE user_sub = ?
            """,
            (user_sub,),
        ).fetchone()

    if not row:
        return {"favorite_nfl_team": ""}

    return {
        "favorite_nfl_team": _dc_team_code(row["favorite_nfl_team"]),
    }


def _wfs_set_favorite_nfl_team(user_sub, team_code):
    if not user_sub:
        return

    team_code = _dc_team_code(team_code)
    if team_code and team_code not in NFL_TEAM_NAMES:
        raise ValueError(f"Unsupported NFL team code: {team_code}")

    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_user_preferences(
                user_sub, favorite_nfl_team
            )
            VALUES (?, ?)
            ON CONFLICT(user_sub) DO UPDATE SET
                favorite_nfl_team=excluded.favorite_nfl_team,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_sub, team_code),
        )


def _wfs_watchlist_rows(user_sub):
    if not user_sub:
        return []
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, player_key, player_name, team, position, watch_type, created_at
            FROM wfs_player_watchlist
            WHERE user_sub = ?
            ORDER BY created_at DESC, player_name COLLATE NOCASE ASC
            """,
            (user_sub,),
        ).fetchall()
    return [dict(r) for r in rows]


def _wfs_watchlist_upsert(user_sub, player_key, player_name, team="", position="", watch_type="Monitor"):
    if not user_sub or not player_key or not player_name:
        return
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_player_watchlist(
                user_sub, player_key, player_name, team, position, watch_type
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_sub, player_key) DO UPDATE SET
                player_name=excluded.player_name,
                team=excluded.team,
                position=excluded.position,
                watch_type=excluded.watch_type,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                user_sub,
                player_key,
                player_name,
                team or "",
                position or "",
                watch_type or "Monitor",
            ),
        )


def _wfs_watchlist_delete(user_sub, row_id):
    if not user_sub or not row_id:
        return
    with _wfs_accounts_connect() as conn:
        row = conn.execute(
            "SELECT player_key FROM wfs_player_watchlist WHERE user_sub = ? AND id = ?",
            (user_sub, int(row_id)),
        ).fetchone()
        conn.execute(
            "DELETE FROM wfs_player_watchlist WHERE user_sub = ? AND id = ?",
            (user_sub, int(row_id)),
        )
        if row:
            conn.execute(
                "DELETE FROM wfs_watchlist_state WHERE user_sub = ? AND player_key = ?",
                (user_sub, row["player_key"]),
            )
            conn.execute(
                "DELETE FROM wfs_watchlist_projection_history WHERE user_sub = ? AND player_key = ?",
                (user_sub, row["player_key"]),
            )


def _wfs_watchlist_record_projection_observations(user_sub, saved_rows, lookup):
    """Record only changed observations from the frozen WFS source."""
    if not user_sub:
        return

    with _wfs_accounts_connect() as conn:
        for item in saved_rows:
            key = item.get("player_key")
            current = lookup.get(key) or {}
            proj = current.get("projection")
            try:
                proj = float(proj) if proj is not None else None
            except Exception:
                proj = None
            status = str(current.get("production_status") or "NOT_IN_CURRENT_SOURCE")
            team = str(current.get("team") or item.get("team") or "").upper()

            last = conn.execute(
                """
                SELECT projection, production_status, team
                FROM wfs_watchlist_projection_history
                WHERE user_sub = ? AND player_key = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_sub, key),
            ).fetchone()

            changed = last is None
            if last is not None:
                last_proj = last["projection"]
                if last_proj is None and proj is not None:
                    changed = True
                elif last_proj is not None and proj is None:
                    changed = True
                elif last_proj is not None and proj is not None and abs(float(last_proj) - proj) >= 0.01:
                    changed = True
                if str(last["production_status"] or "") != status:
                    changed = True
                if str(last["team"] or "") != team:
                    changed = True

            if changed:
                conn.execute(
                    """
                    INSERT INTO wfs_watchlist_projection_history(
                        user_sub, player_key, projection, production_status, team
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_sub, key, proj, status, team),
                )


def _wfs_watchlist_trend_rows(user_sub, saved_rows, lookup):
    """Classify direction from saved observations; no projection math is altered."""
    _wfs_watchlist_record_projection_observations(user_sub, saved_rows, lookup)
    rows = []

    with _wfs_accounts_connect() as conn:
        for item in saved_rows:
            hist = conn.execute(
                """
                SELECT projection, production_status, team, observed_at
                FROM wfs_watchlist_projection_history
                WHERE user_sub = ? AND player_key = ?
                ORDER BY id ASC
                """,
                (user_sub, item.get("player_key")),
            ).fetchall()

            numeric = [float(r["projection"]) for r in hist if r["projection"] is not None]
            current = lookup.get(item.get("player_key")) or {}
            current_proj = current.get("projection")
            try:
                current_proj = float(current_proj) if current_proj is not None else None
            except Exception:
                current_proj = None

            if len(numeric) < 2:
                trend = "BASELINE"
                delta = None
                first = numeric[0] if numeric else current_proj
            else:
                first = numeric[0]
                last = numeric[-1]
                delta = last - first
                if delta >= 1.0:
                    trend = "RISING"
                elif delta <= -1.0:
                    trend = "FALLING"
                else:
                    trend = "STABLE"

            rows.append({
                "player_name": item.get("player_name") or current.get("player") or item.get("player_key"),
                "watch_type": item.get("watch_type") or "Monitor",
                "position": str(current.get("position") or item.get("position") or ""),
                "team": str(current.get("team") or item.get("team") or ""),
                "current_projection": current_proj,
                "first_projection": first,
                "delta": delta,
                "trend": trend,
                "observations": len(hist),
                "status": str(current.get("production_status") or "NOT_IN_CURRENT_SOURCE"),
            })
    return rows


def _wfs_render_player_trend_intelligence(user_sub, saved_rows, lookup):
    """Phase 19: direction of saved-player projection observations."""
    rows = _wfs_watchlist_trend_rows(user_sub, saved_rows, lookup)

    st.markdown("#### 📈 Player Trend Intelligence")
    st.markdown(
        """
        <div style="color:#475569;font-size:.84rem;font-weight:600;margin:.05rem 0 .55rem;">
          Trend direction uses only projection observations recorded while a player is on your WFS Watchlist.
          Rising/falling requires at least a 1.00-point net move from the first saved observation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    import html as _html

    # WFS_PLAYER_TREND_NONE_SAFE_V1
    def _fmt_trend_number(value, signed=False):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if signed:
            return f"{number:+.0f}"
        return f"{number:.0f}"

    icon = {"RISING": "📈", "FALLING": "📉", "STABLE": "➡️", "BASELINE": "🆕"}

    for row in rows:
        current_text = _fmt_trend_number(
            row.get("current_projection")
        )
        first_text = _fmt_trend_number(
            row.get("first_projection")
        )
        delta_text = _fmt_trend_number(
            row.get("delta"),
            signed=True,
        )

        proj_text = (
            f"{current_text} WFS"
            if current_text != "—"
            else "No current WFS projection"
        )

        if row["trend"] == "BASELINE":
            detail = (
                "Baseline established — another changed projection "
                "observation is needed for a trend."
            )
        else:
            detail = (
                f"{first_text} → {current_text} WFS "
                f"({delta_text}) across "
                f"{row['observations']} changed observations."
            )

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                        padding:.75rem .85rem;margin:.42rem 0;color:#f8fafc;">
              <div style="display:flex;justify-content:space-between;gap:.6rem;align-items:flex-start;">
                <div>
                  <div style="font-size:.68rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">
                    {_html.escape(str(row["watch_type"]).upper())} · {_html.escape(row["trend"])}
                  </div>
                  <div style="font-size:1.02rem;font-weight:900;margin:.12rem 0;">
                    {_html.escape(str(row["player_name"]))}
                  </div>
                </div>
                <div style="font-size:1.35rem;">{icon.get(row["trend"], "➡️")}</div>
              </div>
              <div style="font-size:.80rem;color:#cbd5e1;">
                {_html.escape(row["position"] or "—")} · {_html.escape(row["team"] or "—")} ·
                {_html.escape(proj_text)}
              </div>
              <div style="font-size:.78rem;color:#e2e8f0;line-height:1.45;margin-top:.28rem;">
                {_html.escape(detail)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Player trends reflect projection movement recorded while the player is on your watchlist."
    )


def _wfs_watchlist_state_sync(user_sub, saved_rows, lookup):
    """Persist a review baseline and detect meaningful current-source changes."""
    alerts = []
    if not user_sub:
        return alerts

    with _wfs_accounts_connect() as conn:
        for item in saved_rows:
            key = item.get("player_key")
            current = lookup.get(key) or {}
            current_proj = current.get("projection")
            try:
                current_proj = float(current_proj) if current_proj is not None else None
            except Exception:
                current_proj = None
            current_status = str(current.get("production_status") or "NOT_IN_CURRENT_SOURCE")
            current_team = str(current.get("team") or item.get("team") or "").upper()

            state = conn.execute(
                """
                SELECT baseline_projection, baseline_status, baseline_team,
                       observed_projection, observed_status, observed_team, changed_at
                FROM wfs_watchlist_state
                WHERE user_sub = ? AND player_key = ?
                """,
                (user_sub, key),
            ).fetchone()

            # First observation establishes a baseline; it is not an alert.
            if state is None:
                conn.execute(
                    """
                    INSERT INTO wfs_watchlist_state(
                        user_sub, player_key,
                        baseline_projection, baseline_status, baseline_team,
                        observed_projection, observed_status, observed_team
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_sub, key,
                        current_proj, current_status, current_team,
                        current_proj, current_status, current_team,
                    ),
                )
                continue

            baseline_proj = state["baseline_projection"]
            baseline_status = str(state["baseline_status"] or "")
            baseline_team = str(state["baseline_team"] or "")

            projection_changed = False
            delta = None
            if baseline_proj is None and current_proj is not None:
                projection_changed = True
            elif baseline_proj is not None and current_proj is None:
                projection_changed = True
            elif baseline_proj is not None and current_proj is not None:
                delta = current_proj - float(baseline_proj)
                projection_changed = abs(delta) >= 1.0

            status_changed = current_status != baseline_status
            team_changed = current_team != baseline_team
            meaningful = projection_changed or status_changed or team_changed

            # Keep the latest observation current while preserving the review baseline.
            conn.execute(
                """
                UPDATE wfs_watchlist_state
                SET observed_projection = ?,
                    observed_status = ?,
                    observed_team = ?,
                    changed_at = CASE
                        WHEN ? THEN COALESCE(changed_at, CURRENT_TIMESTAMP)
                        ELSE NULL
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_sub = ? AND player_key = ?
                """,
                (
                    current_proj, current_status, current_team,
                    1 if meaningful else 0,
                    user_sub, key,
                ),
            )

            if meaningful:
                alerts.append({
                    "player_key": key,
                    "player_name": item.get("player_name") or current.get("player") or key,
                    "watch_type": item.get("watch_type") or "Monitor",
                    "baseline_projection": baseline_proj,
                    "current_projection": current_proj,
                    "delta": delta,
                    "baseline_status": baseline_status,
                    "current_status": current_status,
                    "baseline_team": baseline_team,
                    "current_team": current_team,
                    "projection_changed": projection_changed,
                    "status_changed": status_changed,
                    "team_changed": team_changed,
                })

    return alerts


def _wfs_watchlist_mark_reviewed(user_sub, player_key):
    if not user_sub or not player_key:
        return
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            UPDATE wfs_watchlist_state
            SET baseline_projection = observed_projection,
                baseline_status = observed_status,
                baseline_team = observed_team,
                changed_at = NULL,
                reviewed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_sub = ? AND player_key = ?
            """,
            (user_sub, player_key),
        )


def _wfs_render_watchlist_intelligence(user_sub, saved_rows, lookup):
    """Phase 18: alerts only on saved-player source changes; no new projections."""
    alerts = _wfs_watchlist_state_sync(user_sub, saved_rows, lookup)

    st.markdown("#### ⭐ Watchlist Intelligence")
    st.markdown(
        """
        <div style="color:#475569;font-size:.84rem;font-weight:600;margin:.05rem 0 .55rem;">
          WFS compares each saved player's current projection, team and availability information
          with the last time you reviewed that player.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not alerts:
        st.success("No meaningful changes detected across your saved players.")
        return

    import html as _html
    for alert in alerts:
        details = []
        if alert["projection_changed"]:
            old = alert["baseline_projection"]
            new = alert["current_projection"]
            if old is None and new is not None:
                details.append(f"Projection now available: {new:.0f} WFS")
            elif old is not None and new is None:
                details.append(f"Projection removed (previously {float(old):.0f} WFS)")
            elif old is not None and new is not None:
                delta = float(new) - float(old)
                details.append(
                    f"Projection {float(old):.0f} → {float(new):.0f} WFS ({delta:+.0f})"
                )
        if alert["status_changed"]:
            details.append(
                f"Status {alert['baseline_status'] or '—'} → {alert['current_status'] or '—'}"
            )
        if alert["team_changed"]:
            details.append(
                f"Team {alert['baseline_team'] or '—'} → {alert['current_team'] or '—'}"
            )

        detail_text = " · ".join(details)
        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #f59e0b;border-radius:14px;
                        padding:.75rem .85rem;margin:.45rem 0 .15rem;color:#f8fafc;">
              <div style="font-size:.68rem;font-weight:900;letter-spacing:.08em;color:#fbbf24;">
                CHANGE DETECTED · {_html.escape(str(alert["watch_type"]).upper())}
              </div>
              <div style="font-size:1.02rem;font-weight:900;margin:.12rem 0;">
                {_html.escape(str(alert["player_name"]))}
              </div>
              <div style="font-size:.80rem;color:#e2e8f0;line-height:1.45;">
                {_html.escape(detail_text)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"✓ Mark {alert['player_name']} reviewed",
            key=f"wfs_watch_review_{alert['player_key']}",
            use_container_width=True,
        ):
            _wfs_watchlist_mark_reviewed(user_sub, alert["player_key"])
            st.rerun()

    st.caption(
        "Watchlist alerts help you monitor meaningful player changes over time."
    )


def _wfs_render_my_watchlist():
    """Phase 17: per-user saved player watchlist backed by the existing WFS account DB."""
    user = _wfs_current_user()
    if not user:
        return

    lookup, source_error = _wfs_int_projection_lookup()
    if not lookup:
        st.warning("Watchlist player source is unavailable right now.")
        return

    st.markdown("### ⭐ My Watchlist")
    st.markdown(
        """
        <style>
        /* Phase 17 watchlist-only label contrast fix */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stSelectbox"] label span {
            color:#334155 !important;
            opacity:1 !important;
            font-weight:750 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="color:#475569;font-size:.92rem;font-weight:600;margin:.05rem 0 .65rem;">
          Save players you want to monitor. This list belongs only to your signed-in WFS account.
          Saving a player does not add, drop, trade, or claim anyone in ESPN.
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = []
    for key, row in lookup.items():
        name = str(row.get("player") or "").strip()
        if not name:
            continue
        pos = str(row.get("position") or "").strip().upper()
        team = str(row.get("team") or "").strip().upper()
        proj = row.get("projection")
        status = str(row.get("production_status") or "UNKNOWN")
        label = f"{name} • {pos or '—'} • {team or '—'}"
        if proj is not None:
            label += f" • {float(proj):.0f} WFS"
        if status == "COLD_START ESTIMATE":
            label += " • Early Projection"
        options.append((label, key, row))
    options.sort(key=lambda x: x[0].lower())

    if options:
        with st.container(border=True):
            selected_label = st.selectbox(
                "Player",
                [x[0] for x in options],
                key="wfs_watchlist_player_select",
            )
            watch_type = st.selectbox(
                "Watch reason",
                ["Monitor", "Waiver", "Trade", "Breakout"],
                key="wfs_watchlist_type_select",
            )
            selected = next(x for x in options if x[0] == selected_label)
            if st.button("⭐ Save to My Watchlist", type="primary", use_container_width=True):
                row = selected[2]
                _wfs_watchlist_upsert(
                    user["sub"],
                    selected[1],
                    str(row.get("player") or ""),
                    str(row.get("team") or ""),
                    str(row.get("position") or ""),
                    watch_type,
                )
                st.success(f"{row.get('player')} saved to your WFS watchlist.")
                st.rerun()

    saved = _wfs_watchlist_rows(user["sub"])
    if not saved:
        st.info("Your watchlist is empty. Save a player above to start monitoring them.")
        return

    st.markdown(
        f"<div style='color:#475569;font-size:.82rem;font-weight:750;margin:.4rem 0 .5rem;'>"
        f"{len(saved)} saved player{'s' if len(saved) != 1 else ''}</div>",
        unsafe_allow_html=True,
    )

    _wfs_render_watchlist_intelligence(user["sub"], saved, lookup)
    _wfs_render_player_trend_intelligence(user["sub"], saved, lookup)

    for item in saved:
        current = lookup.get(item["player_key"]) or {}
        proj = current.get("projection")
        status = str(current.get("production_status") or "NOT_IN_CURRENT_SOURCE")
        team = str(current.get("team") or item.get("team") or "")
        pos = str(current.get("position") or item.get("position") or "")
        projection_text = (
            f"{float(proj):.0f} WFS"
            if proj is not None
            else "No current WFS projection"
        )
        status_text = status
        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                        padding:.75rem .85rem;margin:.45rem 0 .15rem;color:#f8fafc;">
              <div style="font-size:.68rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">
                {item.get("watch_type","Monitor").upper()}
              </div>
              <div style="font-size:1.02rem;font-weight:900;margin:.12rem 0;">
                {item.get("player_name","")}
              </div>
              <div style="font-size:.80rem;color:#cbd5e1;">
                {pos or "—"} · {team or "—"} · {projection_text} · {status_text}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Remove {item.get('player_name','player')}",
            key=f"wfs_watch_remove_{item['id']}",
            use_container_width=True,
        ):
            _wfs_watchlist_delete(user["sub"], item["id"])
            st.rerun()

    if source_error:
        st.caption(source_error)

def _wfs_fernet():
    try:
        section = st.secrets.get("wfs", {})
        key = str(section.get("credential_encryption_key", "") or "").strip()
    except Exception as exc:
        raise RuntimeError("WFS credential encryption key is unavailable.") from exc
    if not key:
        raise RuntimeError(
            "Missing [wfs] credential_encryption_key in Streamlit secrets."
        )
    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError("WFS credential encryption key is invalid.") from exc


def _wfs_encrypt_secret(value):
    value = str(value or "")
    if not value:
        return ""
    return _wfs_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def _wfs_decrypt_secret(value):
    value = str(value or "")
    if not value:
        return ""
    try:
        return _wfs_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(
            "Saved ESPN credentials could not be decrypted with the current WFS key."
        ) from exc


def _wfs_espn_credentials_for_user(user_sub):
    with _wfs_accounts_connect() as conn:
        row = conn.execute(
            """
            SELECT swid_encrypted, espn_s2_encrypted
            FROM wfs_espn_credentials
            WHERE user_sub = ?
            """,
            (str(user_sub),),
        ).fetchone()
    if not row:
        return "", ""
    return (
        _wfs_decrypt_secret(row["swid_encrypted"]),
        _wfs_decrypt_secret(row["espn_s2_encrypted"]),
    )


def _wfs_save_espn_credentials(user_sub, swid, espn_s2):
    swid = str(swid or "").strip()
    espn_s2 = str(espn_s2 or "").strip()
    if not swid or not espn_s2:
        raise ValueError("Both SWID and espn_s2 are required.")
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_espn_credentials(
                user_sub, swid_encrypted, espn_s2_encrypted
            ) VALUES (?, ?, ?)
            ON CONFLICT(user_sub) DO UPDATE SET
                swid_encrypted=excluded.swid_encrypted,
                espn_s2_encrypted=excluded.espn_s2_encrypted,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_sub, _wfs_encrypt_secret(swid), _wfs_encrypt_secret(espn_s2)),
        )


def _wfs_delete_espn_credentials(user_sub):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            "DELETE FROM wfs_espn_credentials WHERE user_sub = ?",
            (str(user_sub),),
        )


def _wfs_espn_connections(user_sub, season):
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, provider, season, league_id, league_name, team_id,
                   team_name, connection_type, created_at, updated_at
            FROM wfs_fantasy_connections
            WHERE user_sub = ? AND provider = 'ESPN' AND season = ?
            ORDER BY lower(league_name), league_id
            """,
            (str(user_sub), int(season)),
        ).fetchall()
    return [dict(row) for row in rows]


def _wfs_espn_upsert_connection(
    user_sub,
    season,
    league_id,
    league_name,
    connection_type,
    team_id=None,
    team_name="",
):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_fantasy_connections(
                user_sub, provider, season, league_id, league_name,
                team_id, team_name, connection_type
            ) VALUES (?, 'ESPN', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_sub, provider, season, league_id) DO UPDATE SET
                league_name=excluded.league_name,
                connection_type=excluded.connection_type,
                team_id=COALESCE(excluded.team_id, wfs_fantasy_connections.team_id),
                team_name=CASE
                    WHEN excluded.team_id IS NOT NULL THEN excluded.team_name
                    ELSE wfs_fantasy_connections.team_name
                END,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(user_sub), int(season), str(league_id), str(league_name or ""),
                int(team_id) if team_id is not None else None,
                str(team_name or ""), str(connection_type or "PUBLIC"),
            ),
        )


def _wfs_espn_set_team(user_sub, season, league_id, team_id, team_name):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            UPDATE wfs_fantasy_connections
            SET team_id = ?, team_name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_sub = ? AND provider = 'ESPN' AND season = ? AND league_id = ?
            """,
            (
                int(team_id), str(team_name or ""), str(user_sub),
                int(season), str(league_id),
            ),
        )


def _wfs_espn_clear_team(user_sub, season, league_id):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            UPDATE wfs_fantasy_connections
            SET team_id = NULL, team_name = '', updated_at = CURRENT_TIMESTAMP
            WHERE user_sub = ? AND provider = 'ESPN' AND season = ? AND league_id = ?
            """,
            (str(user_sub), int(season), str(league_id)),
        )


def _wfs_espn_delete_connection(user_sub, season, league_id):
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            DELETE FROM wfs_fantasy_connections
            WHERE user_sub = ? AND provider = 'ESPN' AND season = ? AND league_id = ?
            """,
            (str(user_sub), int(season), str(league_id)),
        )


# ---------------------------------------------------------------------------
# WFS FANTASY INTELLIGENCE v1
# Read-only layer on top of the frozen ESPN + NFL projection systems.
# No optimizer/projection formulas are modified here.
# ---------------------------------------------------------------------------

WFS_SUPPORTED_START_SIT_SLOTS = {0, 2, 4, 6, 23}  # QB, RB, WR, TE, FLEX
WFS_BENCH_SLOT_ID = 20
WFS_IR_SLOT_ID = 21


def _wfs_int_position_name(player):
    """Friendly ESPN default-position label; used only for display/fallback eligibility."""
    mapping = {
        1: "QB",
        2: "RB",
        3: "WR",
        4: "TE",
        5: "K",
        16: "D/ST",
    }
    try:
        return mapping.get(int(player.get("defaultPositionId")), "")
    except Exception:
        return ""


def _wfs_int_current_nfl_roster_name_keys(season):
    """
    Exact current NFL roster authority for action-oriented pickup surfaces.

    Uses the schedule-driven planning week and weekly_rosters. Missing or
    unverifiable authority fails closed. Projection data is never modified.
    """
    from wfs_schedule_context import resolve_schedule_week_context

    schedule_context = resolve_schedule_week_context(
        season=int(season) if season else None,
    )

    roster_season = int(schedule_context.season)
    roster_week = int(schedule_context.planning_week)

    with sqlite3.connect(DATABASE_PATH) as conn:
        authority = pd.read_sql_query(
            """
            SELECT
                gsis_id,
                full_name,
                team,
                position,
                status,
                status_description_abbr,
                updated_at
            FROM weekly_rosters
            WHERE season = ?
              AND week = ?
              AND game_type = 'REG'
              AND gsis_id IS NOT NULL
              AND TRIM(CAST(gsis_id AS TEXT)) <> ''
            """,
            conn,
            params=[roster_season, roster_week],
        )

    if authority.empty:
        raise ValueError(
            "Current weekly NFL roster authority is unavailable"
        )

    authority["_updated"] = pd.to_datetime(
        authority["updated_at"],
        errors="coerce",
        utc=True,
    )

    if authority["_updated"].isna().any():
        raise ValueError(
            "Current weekly NFL roster authority contains invalid updated_at"
        )

    for column in (
        "gsis_id",
        "full_name",
        "team",
        "position",
        "status",
        "status_description_abbr",
    ):
        authority[column] = (
            authority[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    authority = (
        authority
        .sort_values(
            [
                "gsis_id",
                "_updated",
                "team",
                "status",
                "status_description_abbr",
            ],
            ascending=[True, False, True, True, True],
            kind="stable",
        )
        .drop_duplicates(
            subset=["gsis_id"],
            keep="first",
        )
        .copy()
    )

    if authority["gsis_id"].duplicated().any():
        raise ValueError(
            "Latest weekly NFL roster authority is not singleton per GSIS"
        )

    hard_roster_statuses = {
        "RES",
        "CUT",
        "DEV",
        "EXE",
        "INA",
        "INACTIVE",
        "RET",
        "RETIRED",
        "WAI",
        "WAIVED",
    }

    allowed_positions = {"QB", "RB", "WR", "TE"}
    current_keys = set()

    for row in authority.itertuples(index=False):
        roster_status = str(row.status or "").strip().upper()
        roster_position = str(row.position or "").strip().upper()

        if roster_status in hard_roster_statuses:
            continue

        if roster_position not in allowed_positions:
            continue

        # Use full_name only. football_name can be non-unique.
        key = normalize_name(row.full_name)

        if key:
            current_keys.add(key)

    return current_keys



def _wfs_ui_current_nfl_roster_name_keys(season):
    """Keep unavailable authority distinct from a valid roster with no eligible names."""
    try:
        return _wfs_int_current_nfl_roster_name_keys(season)
    except ValueError as exc:
        if str(exc) != "Current weekly NFL roster authority is unavailable":
            raise
        st.warning(
            "Current-week NFL roster authority is temporarily unavailable. "
            "Roster-dependent pickup recommendations are suppressed until "
            "current-week authority is available."
        )
        return None


def _wfs_int_roster_rows(team):
    """Extract exact ESPN roster entries without fuzzy player matching."""
    entries = ((team.get("roster") or {}).get("entries") or [])
    rows = []
    for entry in entries:
        pool = entry.get("playerPoolEntry") or {}
        player = pool.get("player") or {}
        try:
            slot_id = int(entry.get("lineupSlotId"))
        except Exception:
            slot_id = -1

        eligible = []
        for value in (player.get("eligibleSlots") or []):
            try:
                eligible.append(int(value))
            except Exception:
                pass

        rows.append({
            "slot_id": slot_id,
            "slot": _espn_lineup_slot_name(slot_id),
            "player_id": player.get("id"),
            "player": str(player.get("fullName") or f"Player {player.get('id', '')}").strip(),
            "name_key": normalize_name(player.get("fullName") or ""),
            "position": _wfs_int_position_name(player),
            "eligible_slots": tuple(sorted(set(eligible))),
            "injury": str(player.get("injuryStatus") or "").replace("_", " ").title(),
            "pro_team_id": player.get("proTeamId"),
        })
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def _wfs_int_projection_lookup():
    """
    Build the Fantasy Intelligence projection lookup from two isolated sources:

      MODEL_READY -> frozen production ridge_projection
      COLD_START  -> validated Cold-Start v1 estimate when available

    Production remains the identity/status authority. Cold-start estimates are
    supplemental only, exact-name/team/position matched, and never alter the
    frozen production projection or FanDuel optimizer. No fuzzy matching.
    """
    try:
        if not PRODUCTION_PROJECTION_PATH.exists():
            return {}, (
                "WFS projection data is temporarily unavailable."
            )
        df = pd.read_parquet(PRODUCTION_PROJECTION_PATH).copy()
    except Exception as exc:
        return {}, "WFS projection data is temporarily unavailable."

    required = {
        "season",
        "week",
        "player_display_name",
        "ridge_projection",
        "production_status",
    }
    if not required.issubset(df.columns):
        return {}, (
            "Projection source is missing required "
            "season/week/player/status/projection fields."
        )

    try:
        from wfs_schedule_context import resolve_schedule_week_context

        schedule_context = resolve_schedule_week_context()
        projection_season = int(schedule_context.season)
        projection_week = int(schedule_context.planning_week)

        source_season = pd.to_numeric(
            df["season"],
            errors="coerce",
        )
        source_week = pd.to_numeric(
            df["week"],
            errors="coerce",
        )

        df = df.loc[
            source_season.eq(projection_season)
            & source_week.eq(projection_week)
        ].copy()

        if df.empty:
            return {}, (
                "Current-week WFS projection data is temporarily unavailable."
            )
    except Exception:
        return {}, (
            "Current-week WFS projection authority could not be verified."
        )

    df["ridge_projection"] = pd.to_numeric(
        df["ridge_projection"], errors="coerce"
    )
    df["_name_key"] = df["player_display_name"].map(normalize_name)

    cold_lookup = {}
    cold_source_error = ""
    try:
        if COLD_START_PROJECTION_PATH.exists():
            cold = pd.read_parquet(COLD_START_PROJECTION_PATH).copy()
            cold_required = {
                "player_display_name", "team", "position",
                "cold_start_status", "cold_start_estimate",
            }
            if not cold_required.issubset(cold.columns):
                cold_source_error = "Cold-start source is missing required fields."
            else:
                cold["cold_start_estimate"] = pd.to_numeric(
                    cold["cold_start_estimate"], errors="coerce"
                )
                cold["_name_key"] = cold["player_display_name"].map(normalize_name)
                valid = cold.loc[
                    cold["cold_start_status"].fillna("").astype(str).str.strip().str.upper().eq("COLD_START ESTIMATE")
                    & cold["cold_start_estimate"].notna()
                ].copy()
                for key, group in valid.groupby("_name_key", sort=True):
                    if not key:
                        continue
                    teams = {normalize_team(x) for x in group["team"].dropna() if normalize_team(x)}
                    positions = {str(x).strip().upper() for x in group["position"].dropna() if str(x).strip()}
                    if len(group) != 1 or len(teams) > 1 or len(positions) > 1:
                        continue
                    row = group.iloc[0]
                    cold_lookup[key] = {
                        "projection": float(row["cold_start_estimate"]),
                        "team": next(iter(teams), ""),
                        "position": next(iter(positions), ""),
                    }
        else:
            cold_source_error = (
                "Supplemental WFS projection data is temporarily unavailable."
            )
    except Exception as exc:
        cold_source_error = (
            "Supplemental WFS projection data is temporarily unavailable."
        )

    lookup = {}
    for key, group in df.groupby("_name_key", sort=True):
        if not key:
            continue

        teams = set()
        if "team" in group.columns:
            teams = {normalize_team(x) for x in group["team"].dropna() if normalize_team(x)}

        positions = set()
        if "position" in group.columns:
            positions = {str(x).strip().upper() for x in group["position"].dropna() if str(x).strip()}

        if len(teams) > 1 or len(positions) > 1:
            continue

        statuses = {
            str(x).strip().upper()
            for x in group["production_status"].dropna()
            if str(x).strip()
        }
        status = (
            next(iter(statuses))
            if len(statuses) == 1
            else ("AMBIGUOUS_STATUS" if len(statuses) > 1 else "UNKNOWN")
        )

        ready_rows = group.loc[
            group["production_status"].fillna("").astype(str).str.strip().str.upper().eq("MODEL_READY")
            & group["ridge_projection"].notna()
        ].copy()

        projection = None
        source_status = status
        if not ready_rows.empty:
            projection = float(ready_rows["ridge_projection"].max())
            source_status = "MODEL_READY"
        elif status == "COLD_START":
            cold_match = cold_lookup.get(key)
            if cold_match is not None:
                prod_team = next(iter(teams), "")
                prod_pos = next(iter(positions), "")
                if (
                    cold_match.get("team", "") == prod_team
                    and cold_match.get("position", "") == prod_pos
                ):
                    projection = float(cold_match["projection"])
                    source_status = "COLD_START ESTIMATE"

        display_group = ready_rows if not ready_rows.empty else group
        display = str(
            display_group.sort_values(
                ["player_display_name"], ascending=[True], kind="stable"
            ).iloc[0]["player_display_name"]
        )

        lookup[key] = {
            "player": display,
            "projection": projection,
            "production_status": source_status,
            "team": next(iter(teams), ""),
            "position": next(iter(positions), ""),
            "rows": int(len(group)),
        }

    return lookup, cold_source_error


def _wfs_int_attach_projection(roster_rows, lookup):
    out = []
    for row in roster_rows:
        item = dict(row)
        match = lookup.get(item["name_key"])
        item["wfs_projection"] = (
            float(match["projection"])
            if match is not None and match.get("projection") is not None
            else None
        )
        item["wfs_position"] = (
            str(match.get("position") or "") if match is not None else ""
        )
        item["wfs_team"] = (
            str(match.get("team") or "") if match is not None else ""
        )
        item["wfs_source_status"] = (
            str(match.get("production_status") or "UNKNOWN")
            if match is not None else "NOT_IN_PRODUCTION_SOURCE"
        )
        out.append(item)
    return out


def _wfs_int_can_fill_slot(player_row, starter_slot_id):
    """
    Exact eligibility first: ESPN eligibleSlots.
    Deterministic position fallback only when ESPN omitted eligibleSlots.
    """
    try:
        starter_slot_id = int(starter_slot_id)
    except Exception:
        return False

    eligible = set(player_row.get("eligible_slots") or ())
    if eligible:
        return starter_slot_id in eligible

    pos = str(player_row.get("position") or player_row.get("wfs_position") or "").upper()
    if starter_slot_id == 0:
        return pos == "QB"
    if starter_slot_id == 2:
        return pos == "RB"
    if starter_slot_id == 4:
        return pos == "WR"
    if starter_slot_id == 6:
        return pos == "TE"
    if starter_slot_id == 23:
        return pos in {"RB", "WR", "TE"}
    return False


def _wfs_int_best_swaps(roster_rows):
    """
    Deterministic maximum-gain one-for-one bench/start matching.

    Only recommends a move when BOTH players have WFS projections and the
    bench player is exactly eligible for the starter's ESPN slot. Each starter
    and bench player can appear in at most one recommendation.
    """
    starters = [
        r for r in roster_rows
        if r["slot_id"] in WFS_SUPPORTED_START_SIT_SLOTS
    ]
    bench = [
        r for r in roster_rows
        if r["slot_id"] == WFS_BENCH_SLOT_ID
        and r.get("wfs_projection") is not None
    ]

    # Keep only starters we can compare numerically.
    comparable = [
        r for r in starters if r.get("wfs_projection") is not None
    ]
    if not comparable or not bench:
        return []

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(starter_index, used_mask):
        if starter_index >= len(comparable):
            return 0.0, ()

        starter = comparable[starter_index]

        # Option 1: keep current starter.
        best_gain, best_pairs = solve(starter_index + 1, used_mask)

        # Option 2: replace with one compatible, unused bench player.
        for bench_index, candidate in enumerate(bench):
            bit = 1 << bench_index
            if used_mask & bit:
                continue
            if not _wfs_int_can_fill_slot(candidate, starter["slot_id"]):
                continue

            gain = float(candidate["wfs_projection"]) - float(starter["wfs_projection"])
            if gain <= 1e-9:
                continue

            tail_gain, tail_pairs = solve(starter_index + 1, used_mask | bit)
            total = gain + tail_gain
            pair = (starter_index, bench_index, gain)
            proposed_pairs = (pair,) + tail_pairs

            # Deterministic tie-break: total gain, then lexical recommendation tuple.
            def signature(pairs):
                labels = []
                for si, bi, g in pairs:
                    labels.append((
                        comparable[si]["slot"],
                        comparable[si]["player"],
                        bench[bi]["player"],
                        round(float(g), 8),
                    ))
                return tuple(labels)

            if (
                total > best_gain + 1e-9
                or (
                    abs(total - best_gain) <= 1e-9
                    and signature(proposed_pairs) < signature(best_pairs)
                )
            ):
                best_gain = total
                best_pairs = proposed_pairs

        return best_gain, best_pairs

    _, pairs = solve(0, 0)
    recommendations = []
    for starter_index, bench_index, gain in pairs:
        starter = comparable[starter_index]
        candidate = bench[bench_index]
        recommendations.append({
            "Slot": starter["slot"],
            "Sit": starter["player"],
            "Sit Proj": round(float(starter["wfs_projection"]), 2),
            "Start": candidate["player"],
            "Start Proj": round(float(candidate["wfs_projection"]), 2),
            "Gain": round(float(gain), 2),
        })

    recommendations.sort(
        key=lambda r: (-r["Gain"], r["Slot"], r["Start"], r["Sit"])
    )
    return recommendations


def _wfs_render_matchup_intelligence(my_team, opponent_team, season, league_id):
    """Read-only WFS projection comparison for the current ESPN matchup."""
    if not opponent_team:
        return

    lookup, projection_error = _wfs_int_projection_lookup()
    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    opp_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(opponent_team), lookup)

    def covered_starters(rows):
        return [
            r for r in rows
            if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS
            and r.get("wfs_projection") is not None
        ]

    mine = covered_starters(my_rows)
    theirs = covered_starters(opp_rows)
    my_all = [r for r in my_rows if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS]
    opp_all = [r for r in opp_rows if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS]
    if not mine and not theirs:
        return

    my_total = sum(float(r["wfs_projection"]) for r in mine)
    opp_total = sum(float(r["wfs_projection"]) for r in theirs)
    margin = my_total - opp_total
    swaps = _wfs_int_best_swaps(my_rows)
    gain = sum(float(r["Gain"]) for r in swaps)
    optimized_total = my_total + gain
    optimized_margin = optimized_total - opp_total

    def pos_totals(rows):
        out = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "FLEX": 0.0}
        for r in rows:
            slot = str(r.get("slot") or "").upper()
            bucket = "FLEX" if r.get("slot_id") == 23 else str(r.get("position") or r.get("wfs_position") or slot).upper()
            if bucket in out:
                out[bucket] += float(r["wfs_projection"])
        return out

    mt, ot = pos_totals(mine), pos_totals(theirs)
    edges=[]
    for pos in ("QB","RB","WR","TE","FLEX"):
        if mt[pos] or ot[pos]:
            edges.append({"Position":pos,"My WFS Proj":round(mt[pos],2),"Opponent WFS Proj":round(ot[pos],2),"Edge":round(mt[pos]-ot[pos],2)})

    st.markdown("### ⚔️ WFS Matchup Intelligence")
    st.caption("WFS projection comparison of covered ESPN starters. K/DST are excluded from this model comparison.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("My WFS Projection", f"{my_total:.0f}")
    c2.metric("Opponent WFS Projection", f"{opp_total:.0f}")
    c3.metric("WFS Margin", f"{margin:+.0f}")
    c4.metric("After Start/Sit", f"{optimized_margin:+.0f}", f"+{gain:.0f}" if gain > 0 else None)

    if projection_error:
        st.warning(projection_error)
    if margin > 0:
        st.success(f"WFS currently projects a {margin:.0f}-point edge across covered starters.")
    elif margin < 0:
        st.warning(f"WFS currently projects a {abs(margin):.0f}-point deficit across covered starters.")
    else:
        st.info("WFS currently has this matchup even across covered starters.")

    if edges:
        edf=pd.DataFrame(edges).sort_values(["Edge","Position"], ascending=[False,True], kind="stable")
        best=edf.iloc[0]
        worst=edf.iloc[-1]
        st.markdown(
            f"<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:0.35rem 0 0.55rem 0;'>"
            f"Strongest edge: {best['Position']} {best['Edge']:+.0f} • Weakest edge: {worst['Position']} {worst['Edge']:+.0f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("View position-by-position matchup", expanded=False):
            st.dataframe(edf.reset_index(drop=True), width="stretch", hide_index=True,
                column_config={"My WFS Proj":st.column_config.NumberColumn(format="%.0f"),"Opponent WFS Proj":st.column_config.NumberColumn(format="%.0f"),"Edge":st.column_config.NumberColumn(format="%+.0f")})

    st.markdown(
        f"<div style='color:#475569;font-size:0.95rem;font-weight:600;margin-top:0.45rem;'>"
        f"Projection coverage: My team {len(mine)}/{len(my_all)} • Opponent {len(theirs)}/{len(opp_all)} supported starters."
        f"</div>",
        unsafe_allow_html=True,
    )


def _wfs_render_matchup_game_plan(my_team, opponent_team, season, league_id):
    """Phase 6 read-only action layer connecting matchup deficits to exact bench swaps."""
    if not opponent_team:
        return

    lookup, projection_error = _wfs_int_projection_lookup()
    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    opp_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(opponent_team), lookup)

    def covered(rows):
        return [
            r for r in rows
            if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS
            and r.get("wfs_projection") is not None
        ]

    mine, theirs = covered(my_rows), covered(opp_rows)
    if not mine or not theirs:
        return

    def bucket(row):
        if row.get("slot_id") == 23:
            return "FLEX"
        return str(row.get("position") or row.get("wfs_position") or row.get("slot") or "").upper()

    def totals(rows):
        out = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "FLEX": 0.0}
        for row in rows:
            b = bucket(row)
            if b in out:
                out[b] += float(row["wfs_projection"])
        return out

    mt, ot = totals(mine), totals(theirs)
    current_edges = {pos: mt[pos] - ot[pos] for pos in mt if mt[pos] or ot[pos]}
    swaps = _wfs_int_best_swaps(my_rows)

    st.markdown("### 🎯 WFS Matchup Game Plan")
    st.markdown("<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:0.15rem 0 0.75rem 0;'>Projected lineup moves that may improve your weekly matchup.</div>", unsafe_allow_html=True)

    if projection_error:
        st.warning(projection_error)

    if not swaps:
        if current_edges:
            weakest = min(current_edges, key=lambda pos: (current_edges[pos], pos))
            edge = current_edges[weakest]
            if edge < 0:
                st.info(
                    f"Priority: {weakest}. WFS shows a {abs(edge):.0f}-point positional deficit, "
                    "but no eligible bench swap currently improves your projected score."
                )
            else:
                st.success("WFS does not find a positive projected bench swap for the covered starting lineup.")
        return

    plan_rows = []
    for rec in swaps:
        slot = str(rec.get("Slot") or "").upper()
        pos = "FLEX" if slot == "FLEX" else slot
        before = current_edges.get(pos)
        gain = float(rec["Gain"])
        after = (before + gain) if before is not None else None
        plan_rows.append({
            "Priority": pos,
            "Current Edge": round(before, 2) if before is not None else None,
            "Move": f"{rec['Start']} over {rec['Sit']}",
            "Gain": round(gain, 2),
            "After Move": round(after, 2) if after is not None else None,
        })

    plan_rows.sort(key=lambda r: (
        r["Current Edge"] if r["Current Edge"] is not None else 999.0,
        -r["Gain"], r["Priority"], r["Move"]
    ))

    top = plan_rows[0]
    before = top["Current Edge"]
    after = top["After Move"]
    if before is not None and before < 0:
        if after is not None and after >= 0:
            outcome = f"turns that into a {after:+.0f} edge"
        elif after is not None:
            outcome = f"cuts the deficit to {abs(after):.0f}"
        else:
            outcome = "improves the matchup"
        st.success(
            f"Priority: {top['Priority']} — you currently trail by {abs(before):.0f} WFS points. "
            f"{top['Move']} adds {top['Gain']:+.0f} and {outcome}."
        )
    else:
        st.success(
            f"Best actionable move: {top['Move']} at {top['Priority']} adds {top['Gain']:+.0f} WFS points."
        )

    with st.expander("View matchup game plan", expanded=False):
        # Mobile-first Phase 6 view: keep the table narrow and let move text wrap.
        rows_html = []
        for row in plan_rows:
            before_text = "—" if row["Current Edge"] is None else f"{row['Current Edge']:+.0f}"
            after_text = "—" if row["After Move"] is None else f"{row['After Move']:+.0f}"
            rows_html.append(
                "<tr>"
                f"<td>{row['Priority']}</td>"
                f"<td class='wfs-gameplan-move'>{row['Move']}</td>"
                f"<td>{row['Gain']:+.0f}</td>"
                f"<td>{after_text}</td>"
                "</tr>"
            )
        st.markdown(
            """
            <style>
            .wfs-gameplan-table-wrap {overflow-x:hidden;width:100%;}
            .wfs-gameplan-table {width:100%;border-collapse:collapse;table-layout:fixed;background:#0e1117;color:#f8fafc;border-radius:12px;overflow:hidden;}
            .wfs-gameplan-table th,.wfs-gameplan-table td {border:1px solid #29313d;padding:.58rem .48rem;vertical-align:top;font-size:.88rem;}
            .wfs-gameplan-table th {background:#171c24;color:#cbd5e1;text-align:left;font-weight:700;}
            .wfs-gameplan-table th:nth-child(1),.wfs-gameplan-table td:nth-child(1){width:13%;}
            .wfs-gameplan-table th:nth-child(2),.wfs-gameplan-table td:nth-child(2){width:49%;}
            .wfs-gameplan-table th:nth-child(3),.wfs-gameplan-table td:nth-child(3){width:17%;text-align:right;}
            .wfs-gameplan-table th:nth-child(4),.wfs-gameplan-table td:nth-child(4){width:21%;text-align:right;}
            .wfs-gameplan-move {white-space:normal;overflow-wrap:anywhere;word-break:normal;line-height:1.3;}
            @media (max-width: 600px) {
              .wfs-gameplan-table th,.wfs-gameplan-table td {font-size:.80rem;padding:.52rem .36rem;}
            }
            </style>
            <div class="wfs-gameplan-table-wrap">
              <table class="wfs-gameplan-table">
                <thead><tr><th>Pos</th><th>Move</th><th>Gain</th><th>New Edge</th></tr></thead>
                <tbody>
            """ + "".join(rows_html) + """
                </tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#475569;font-size:0.92rem;font-weight:600;margin-top:0.7rem;'>"
            "Read-only recommendations only. WFS does not change ESPN lineups or submit roster moves."
            "</div>",
            unsafe_allow_html=True,
        )



def _wfs_render_roster_strength_intelligence(teams, my_team, season, league_id):
    """Phase 7 roster-room strength snapshot from frozen WFS projections and returned ESPN rosters."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        return

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}

    def projected_room(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        vals = []
        names = []
        for row in rows:
            rpos = str(row.get("position") or row.get("wfs_position") or "").upper()
            proj = row.get("wfs_projection")
            if rpos != pos or proj is None or row.get("slot_id") == WFS_IR_SLOT_ID:
                continue
            vals.append(float(proj))
            names.append((str(row.get("player") or ""), float(proj)))
        vals.sort(reverse=True)
        names.sort(key=lambda x: (-x[1], x[0]))
        take = vals[:room_size[pos]]
        room_avg = (sum(take) / len(take)) if take else None
        return room_avg, len(vals), names

    team_scores = {pos: [] for pos in positions}
    for team in teams or []:
        for pos in positions:
            score, depth, _ = projected_room(team, pos)
            if score is not None:
                team_scores[pos].append(score)

    my_details = {}
    for pos in positions:
        score, depth, names = projected_room(my_team, pos)
        peers = team_scores[pos]
        if score is None or not peers:
            pct = None
        else:
            below_or_equal = sum(1 for x in peers if x <= score)
            pct = 100.0 * below_or_equal / len(peers)
        if pct is None:
            tier = "Unrated"
        elif pct >= 75:
            tier = "Strength"
        elif pct >= 40:
            tier = "Stable"
        else:
            tier = "Need"
        my_details[pos] = {
            "score": score,
            "depth": depth,
            "names": names,
            "percentile": pct,
            "tier": tier,
        }

    # Infer unrostered candidates exactly the same way as Waiver Wire Intelligence:
    # subtract exact normalized names on all returned ESPN rosters from the frozen WFS source.
    rostered_keys = set()
    for team in teams or []:
        for row in _wfs_int_roster_rows(team):
            if row.get("name_key"):
                rostered_keys.add(row["name_key"])

    best_available = {}
    current_nfl_roster_keys = _wfs_ui_current_nfl_roster_name_keys(season)
    if current_nfl_roster_keys is not None:
        for key, item in lookup.items():
            if key not in current_nfl_roster_keys:
                continue
            if key in rostered_keys:
                continue
            pos = str(item.get("position") or "").upper()
            proj = item.get("projection")
            status = str(item.get("production_status") or "")
            if pos not in positions or proj is None or status not in {"MODEL_READY", "COLD_START ESTIMATE"}:
                continue
            row = {
                "player": item.get("player") or key,
                "projection": float(proj),
                "status": status,
            }
            cur = best_available.get(pos)
            if cur is None or (row["projection"], row["player"]) > (cur["projection"], cur["player"]):
                best_available[pos] = row

    for pos in positions:
        detail = my_details[pos]
        add = best_available.get(pos)
        weakest = min(detail["names"], key=lambda x: (x[1], x[0])) if detail["names"] else None
        gain = None
        if add and weakest:
            gain = add["projection"] - weakest[1]
        detail["best_available"] = add
        detail["weakest"] = weakest
        detail["upgrade_gain"] = gain

    rated = [(pos, d) for pos, d in my_details.items() if d["percentile"] is not None]
    if not rated:
        return

    strongest_pos, strongest = max(rated, key=lambda x: (x[1]["percentile"], x[0]))
    weakest_pos, weakest_detail = min(rated, key=lambda x: (x[1]["percentile"], x[0]))

    st.markdown("### 🧭 WFS Roster Strength & Weakness")
    st.markdown(
        "<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:0.15rem 0 0.75rem 0;'>"
        "Compare the strength of your QB, RB, WR and TE groups with the rest of your ESPN league. "
        "Each position is evaluated using current WFS player projections and roster depth."
        "</div>",
        unsafe_allow_html=True,
    )

    if projection_error:
        st.warning(projection_error)

    if weakest_detail["percentile"] < 40:
        st.warning(
            f"Roster priority: {weakest_pos}. Your {weakest_pos} room ranks around the "
            f"{weakest_detail['percentile']:.0f}th percentile of projection-covered league rooms."
        )
    else:
        st.success(
            f"Roster foundation looks stable across rated rooms. Strongest area: {strongest_pos} "
            f"at about the {strongest['percentile']:.0f}th percentile."
        )

    cards = []
    for pos in positions:
        d = my_details[pos]
        pct = "—" if d["percentile"] is None else f"{d['percentile']:.0f}th"
        avg = "—" if d["score"] is None else f"{d['score']:.2f}"
        tier = d["tier"]
        tier_color = {"Strength":"#166534", "Stable":"#1d4ed8", "Need":"#b45309", "Unrated":"#64748b"}[tier]
        bg = {"Strength":"#ecfdf3", "Stable":"#eff6ff", "Need":"#fff7ed", "Unrated":"#f8fafc"}[tier]
        cards.append(
            f"<div style='background:{bg};border:1px solid #cbd5e1;border-radius:14px;padding:.75rem .8rem;'>"
            f"<div style='display:flex;justify-content:space-between;gap:.5rem;align-items:center;'>"
            f"<strong style='color:#0f172a;font-size:1.05rem;'>{pos}</strong>"
            f"<span style='color:{tier_color};font-weight:800;font-size:.78rem;'>{tier}</span></div>"
            f"<div style='color:#334155;font-size:.86rem;margin-top:.35rem;'>League percentile: <strong>{pct}</strong></div>"
            f"<div style='color:#334155;font-size:.86rem;'>Room avg: <strong>{avg}</strong> • Covered depth: <strong>{d['depth']}</strong></div>"
            f"</div>"
        )
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;margin:.35rem 0 .75rem;'>"
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )

    # Surface the most actionable inferred upgrade, but do not overclaim waiver eligibility.
    upgrade_rows = []
    for pos in positions:
        d = my_details[pos]
        add = d.get("best_available")
        drop = d.get("weakest")
        gain = d.get("upgrade_gain")
        if add and drop and gain is not None and gain > 0:
            upgrade_rows.append((gain, pos, add, drop))
    upgrade_rows.sort(key=lambda x: (-x[0], x[1], x[2]["player"]))
    if upgrade_rows:
        gain, pos, add, drop = upgrade_rows[0]
        st.info(
            f"Best inferred roster-room upgrade: {pos} — {add['player']} over {drop[0]} "
            f"adds {gain:+.2f} WFS projected points. Confirm ESPN availability before acting."
        )

    with st.expander("View roster room details", expanded=False):
        detail_rows = []
        for pos in positions:
            d = my_details[pos]
            add = d.get("best_available")
            gain = d.get("upgrade_gain")
            detail_rows.append({
                "Pos": pos,
                "Tier": d["tier"],
                "League %ile": None if d["percentile"] is None else round(d["percentile"]),
                "Room Avg": None if d["score"] is None else round(d["score"], 2),
                "Depth": d["depth"],
                "Best Inferred Add": "Unavailable" if current_nfl_roster_keys is None else (add["player"] if add else "—"),
                "Potential Gain": None if current_nfl_roster_keys is None else (round(gain, 2) if gain is not None and gain > 0 else 0.0),
            })
        st.dataframe(
            pd.DataFrame(detail_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Room Avg": st.column_config.NumberColumn(format="%.2f"),
                "Potential Gain": st.column_config.NumberColumn(format="+%.2f"),
            },
        )
        st.markdown(
            "<div style='color:#475569;font-size:0.88rem;font-weight:600;margin-top:0.55rem;'>"
            "This is a current roster-strength snapshot, not a rest-of-season forecast. "
            "Confirm player availability in ESPN before making roster moves."
            "</div>",
            unsafe_allow_html=True,
        )


def _wfs_render_trade_intelligence(teams, my_team, season, league_id):
    """Phase 8 read-only trade-target intelligence from frozen WFS projections and ESPN rosters."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup or not teams or not my_team:
        return

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}

    def room_detail(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        players = []
        for row in rows:
            rpos = str(row.get("position") or row.get("wfs_position") or "").upper()
            proj = row.get("wfs_projection")
            if rpos != pos or proj is None or row.get("slot_id") == WFS_IR_SLOT_ID:
                continue
            players.append({
                "player": str(row.get("player") or ""),
                "projection": float(proj),
                "status": str(row.get("wfs_status") or row.get("production_status") or "MODEL_READY"),
            })
        players.sort(key=lambda r: (-r["projection"], r["player"]))
        core = players[:room_size[pos]]
        score = (sum(r["projection"] for r in core) / len(core)) if core else None
        return {"score": score, "depth": len(players), "players": players}

    # Build league room distributions once, using only projection-covered players.
    league_rooms = {pos: [] for pos in positions}
    team_rooms = {}
    for team in teams:
        tid = team.get("id")
        rooms = {pos: room_detail(team, pos) for pos in positions}
        team_rooms[tid] = rooms
        for pos in positions:
            if rooms[pos]["score"] is not None:
                league_rooms[pos].append(rooms[pos]["score"])

    my_id = my_team.get("id")
    my_rooms = team_rooms.get(my_id) or {pos: room_detail(my_team, pos) for pos in positions}

    def percentile(pos, score):
        peers = league_rooms.get(pos) or []
        if score is None or not peers:
            return None
        return 100.0 * sum(1 for x in peers if x <= score) / len(peers)

    my_pct = {pos: percentile(pos, my_rooms[pos]["score"]) for pos in positions}
    rated = [(pos, pct) for pos, pct in my_pct.items() if pct is not None]
    if not rated:
        return

    need_pos, need_pct = min(rated, key=lambda x: (x[1], x[0]))

    # Prefer trading from a strong room with depth beyond the scoring core.
    trade_from_candidates = []
    for pos, pct in rated:
        if pos == need_pos:
            continue
        depth = my_rooms[pos]["depth"]
        surplus = max(0, depth - room_size[pos])
        trade_from_candidates.append((surplus > 0, pct, surplus, pos))
    trade_from_candidates.sort(reverse=True)
    trade_from = trade_from_candidates[0][3] if trade_from_candidates else None

    # Identify a concrete, lower-cost piece from the selected surplus room for context only.
    trade_piece = None
    if trade_from:
        players = my_rooms[trade_from]["players"]
        if len(players) > room_size[trade_from]:
            trade_piece = players[room_size[trade_from]]
        elif players:
            trade_piece = players[-1]

    target_rows = []
    for team in teams:
        if team.get("id") == my_id:
            continue
        rooms = team_rooms.get(team.get("id")) or {}
        target_room = rooms.get(need_pos) or {"players": []}
        if not target_room["players"]:
            continue

        partner_pct = None
        if trade_from and rooms.get(trade_from):
            partner_pct = percentile(trade_from, rooms[trade_from]["score"])

        # A lower percentile at our trade-from position means a more complementary roster fit.
        if partner_pct is None:
            fit = "Roster fit unknown"
        elif partner_pct < 40:
            fit = f"Needs {trade_from}"
        elif partner_pct < 75:
            fit = f"Could use {trade_from}"
        else:
            fit = f"Strong at {trade_from}"

        # Surface up to two covered players from the position we need; this is a target list,
        # not a claim of fair value or manager willingness.
        for player in target_room["players"][:2]:
            target_rows.append({
                "Team": _espn_team_name(team),
                "Target": player["player"],
                "Pos": need_pos,
                "WFS Proj": round(player["projection"], 2),
                "Partner Fit": fit,
                "partner_pct": partner_pct if partner_pct is not None else 999.0,
            })

    target_rows.sort(key=lambda r: (r["partner_pct"], -r["WFS Proj"], r["Team"], r["Target"]))

    st.markdown("### 🤝 WFS Trade Intelligence")
    st.markdown(
        "<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:0.15rem 0 0.75rem 0;'>"
        "Find possible trade targets based on your roster needs and the strengths of other teams in your ESPN league. "
        "WFS identifies roster fit but cannot predict whether another manager will accept a trade."
        "</div>",
        unsafe_allow_html=True,
    )

    if projection_error:
        st.warning(projection_error)

    need_text = f"{need_pct:.0f}th percentile"
    if trade_from:
        from_pct = my_pct.get(trade_from)
        from_text = "—" if from_pct is None else f"{from_pct:.0f}th percentile"
        st.info(
            f"Trade plan: target {need_pos} ({need_text}). Your best trade-from room is {trade_from} "
            f"({from_text}) based on league-relative strength and covered depth."
        )
    else:
        st.info(f"Trade plan: target {need_pos} ({need_text}). WFS does not find a clear surplus room to trade from.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #d8dee8;border-radius:18px;padding:1rem 1.05rem;min-height:118px;box-shadow:0 1px 2px rgba(15,23,42,.04);">
                <div style="color:#334155 !important;opacity:1;font-size:.92rem;font-weight:750;margin-bottom:.35rem;">Primary Need</div>
                <div style="color:#0b4f9c !important;font-size:1.72rem;font-weight:800;line-height:1.05;">{need_pos}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        trade_from_display = trade_from or "No clear surplus"
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #d8dee8;border-radius:18px;padding:1rem 1.05rem;min-height:118px;box-shadow:0 1px 2px rgba(15,23,42,.04);">
                <div style="color:#334155 !important;opacity:1;font-size:.92rem;font-weight:750;margin-bottom:.35rem;">Trade From</div>
                <div style="color:#0b4f9c !important;font-size:1.72rem;font-weight:800;line-height:1.05;overflow-wrap:anywhere;">{trade_from_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if trade_piece and trade_from:
        st.markdown(
            f"<div style='color:#334155;font-size:0.92rem;font-weight:650;margin:.25rem 0 .7rem 0;'>"
            f"Possible trade-capital context: <strong>{trade_piece['player']}</strong> ({trade_from}, "
            f"{trade_piece['projection']:.2f} WFS). This is not a fair-value recommendation by itself."
            f"</div>",
            unsafe_allow_html=True,
        )

    if not target_rows:
        st.warning(f"No projection-covered {need_pos} trade targets were found on the returned ESPN rosters.")
        return

    best = target_rows[0]
    st.success(
        f"Best roster-fit target to investigate: {best['Target']} ({need_pos}) on {best['Team']} — "
        f"{best['WFS Proj']:.2f} WFS projected points. {best['Partner Fit']}."
    )

    with st.expander("View trade targets", expanded=False):
        rows_html = []
        import html as _html
        for row in target_rows[:12]:
            rows_html.append(
                "<tr>"
                f"<td>{_html.escape(row['Team'])}</td>"
                f"<td>{_html.escape(row['Target'])}</td>"
                f"<td>{row['Pos']}</td>"
                f"<td>{row['WFS Proj']:.2f}</td>"
                f"<td>{_html.escape(row['Partner Fit'])}</td>"
                "</tr>"
            )
        st.markdown(
            """
            <style>
            .wfs-trade-wrap{width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;}
            .wfs-trade-table{width:100%;border-collapse:collapse;background:#0e1117;color:#f8fafc;min-width:620px;}
            .wfs-trade-table th,.wfs-trade-table td{border:1px solid #29313d;padding:.55rem .48rem;vertical-align:top;font-size:.84rem;}
            .wfs-trade-table th{background:#171c24;color:#cbd5e1;text-align:left;font-weight:800;}
            .wfs-trade-table td:nth-child(4){text-align:right;}
            </style>
            <div class="wfs-trade-wrap"><table class="wfs-trade-table">
            <thead><tr><th>Team</th><th>Target</th><th>Pos</th><th>WFS</th><th>Roster Fit</th></tr></thead><tbody>
            """ + "".join(rows_html) + """
            </tbody></table></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#475569;font-size:0.9rem;font-weight:600;margin-top:.7rem;'>"
            "Targets are roster-fit leads only. WFS does not model another manager's preferences, trade value charts, keeper rules, or acceptance probability."
            "</div>",
            unsafe_allow_html=True,
        )


def _wfs_render_league_power_rankings(teams, my_team, season, league_id):
    """Phase 9: deterministic league power rankings from frozen WFS roster-room projections."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup or not teams:
        return

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}

    def team_power(team):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        rooms = {}
        covered = 0
        for pos in positions:
            vals = []
            for row in rows:
                rpos = str(row.get("position") or row.get("wfs_position") or "").upper()
                proj = row.get("wfs_projection")
                if rpos != pos or proj is None or row.get("slot_id") == WFS_IR_SLOT_ID:
                    continue
                vals.append(float(proj))
            vals.sort(reverse=True)
            take = vals[:room_size[pos]]
            covered += len(take)
            rooms[pos] = (sum(take) / len(take)) if take else None

        rated = [rooms[p] for p in positions if rooms[p] is not None]
        # Equal-room composite intentionally matches the Phase 7 room-strength framework.
        score = sum(rated) if len(rated) == len(positions) else None
        return score, covered, rooms

    rankings = []
    for team in teams:
        score, covered, rooms = team_power(team)
        if score is None:
            continue
        rankings.append({
            "team_id": team.get("id"),
            "team": _espn_team_name(team),
            "score": score,
            "covered": covered,
            "rooms": rooms,
        })

    if not rankings:
        return

    rankings.sort(key=lambda r: (-r["score"], -r["covered"], r["team"].lower()))
    for i, row in enumerate(rankings, 1):
        row["rank"] = i

    my_id = my_team.get("id") if my_team else None
    mine = next((r for r in rankings if r["team_id"] == my_id), None)

    st.markdown("### 🏆 WFS League Power Rankings")
    st.markdown(
        "<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:0.15rem 0 0.75rem 0;'>"
        "Compare projected roster strength across your ESPN league. "
        "The WFS Power Score combines QB, RB, WR and TE position strength and is separate from ESPN standings."
        "</div>",
        unsafe_allow_html=True,
    )
    if projection_error:
        st.warning(projection_error)

    if mine:
        total = len(rankings)
        leader = rankings[0]
        gap = leader["score"] - mine["score"]
        if mine["rank"] == 1:
            st.success(
                f"Your team ranks #1 of {total} projection-covered teams with a {mine['score']:.2f} WFS Power Score."
            )
        else:
            st.info(
                f"Your team ranks #{mine['rank']} of {total} with a {mine['score']:.2f} WFS Power Score — "
                f"{gap:.2f} behind #1 {leader['team']}."
            )

    # Compact mobile leaderboard: full room detail is available below.
    import html as _html
    rows_html = []
    for row in rankings:
        is_mine = row["team_id"] == my_id
        style = " style='background:#172554;'" if is_mine else ""
        team_label = _html.escape(row["team"]) + (" • YOU" if is_mine else "")
        rows_html.append(
            f"<tr{style}><td>#{row['rank']}</td><td>{team_label}</td>"
            f"<td>{row['score']:.2f}</td><td>{row['covered']}/8</td></tr>"
        )
    st.markdown(
        """
        <style>
        .wfs-power-wrap{width:100%;overflow-x:hidden;}
        .wfs-power-table{width:100%;border-collapse:collapse;table-layout:fixed;background:#0e1117;color:#f8fafc;border-radius:12px;overflow:hidden;}
        .wfs-power-table th,.wfs-power-table td{border:1px solid #29313d;padding:.58rem .48rem;vertical-align:top;font-size:.86rem;}
        .wfs-power-table th{background:#171c24;color:#cbd5e1;text-align:left;font-weight:800;}
        .wfs-power-table th:nth-child(1),.wfs-power-table td:nth-child(1){width:13%;}
        .wfs-power-table th:nth-child(2),.wfs-power-table td:nth-child(2){width:49%;overflow-wrap:anywhere;}
        .wfs-power-table th:nth-child(3),.wfs-power-table td:nth-child(3){width:22%;text-align:right;}
        .wfs-power-table th:nth-child(4),.wfs-power-table td:nth-child(4){width:16%;text-align:right;}
        @media (max-width:600px){.wfs-power-table th,.wfs-power-table td{font-size:.80rem;padding:.52rem .34rem;}}
        </style>
        <div class="wfs-power-wrap"><table class="wfs-power-table">
        <thead><tr><th>Rank</th><th>Team</th><th>Power</th><th>Cov</th></tr></thead><tbody>
        """ + "".join(rows_html) + """
        </tbody></table></div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View position-room power details", expanded=False):
        detail_rows = []
        for row in rankings:
            rm = row["rooms"]
            detail_rows.append({
                "Rank": row["rank"], "Team": row["team"], "Power": round(row["score"], 2),
                "QB": round(rm["QB"], 2) if rm["QB"] is not None else None,
                "RB": round(rm["RB"], 2) if rm["RB"] is not None else None,
                "WR": round(rm["WR"], 2) if rm["WR"] is not None else None,
                "TE": round(rm["TE"], 2) if rm["TE"] is not None else None,
                "Coverage": f"{row['covered']}/8",
            })
        st.dataframe(detail_rows, hide_index=True, width="stretch")
        st.markdown(
            "<div style='color:#475569;font-size:0.9rem;font-weight:600;margin-top:.6rem;'>"
            "Coverage counts the scoring-core players used by this index: QB1 + RB3 + WR3 + TE1. "
            "K/DST are excluded, matching the WFS projection model scope."
            "</div>",
            unsafe_allow_html=True,
        )


def _wfs_render_contender_tiers(teams, my_team, season, league_id):
    """Phase 10: league-relative contender tiers from the locked Phase 9 power framework."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup or not teams:
        return

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}

    def team_power(team):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        rooms = {}
        covered = 0
        for pos in positions:
            vals = []
            for row in rows:
                rpos = str(row.get("position") or row.get("wfs_position") or "").upper()
                proj = row.get("wfs_projection")
                if rpos != pos or proj is None or row.get("slot_id") == WFS_IR_SLOT_ID:
                    continue
                vals.append(float(proj))
            vals.sort(reverse=True)
            take = vals[:room_size[pos]]
            covered += len(take)
            rooms[pos] = (sum(take) / len(take)) if take else None
        rated = [rooms[p] for p in positions if rooms[p] is not None]
        score = sum(rated) if len(rated) == len(positions) else None
        return score, covered, rooms

    rows = []
    for team in teams:
        score, covered, rooms = team_power(team)
        if score is None:
            continue
        rows.append({"team_id": team.get("id"), "team": _espn_team_name(team), "score": score,
                     "covered": covered, "rooms": rooms})
    if not rows:
        return

    rows.sort(key=lambda r: (-r["score"], -r["covered"], r["team"].lower()))
    n = len(rows)
    # Deterministic league-relative bands. These are descriptive tiers, not title probabilities.
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        pct = 1.0 if n == 1 else 1.0 - ((i - 1) / (n - 1))
        room_vals = list(row["rooms"].values())
        balance = min(room_vals) / max(room_vals) if room_vals and max(room_vals) > 0 else 0.0
        row["balance"] = balance
        if pct >= 0.75:
            tier = "Elite"
        elif pct >= 0.50:
            tier = "Contender"
        elif pct >= 0.25:
            tier = "Competitive"
        else:
            tier = "Needs Help"
        row["tier"] = tier

    my_id = my_team.get("id") if my_team else None
    mine = next((r for r in rows if r["team_id"] == my_id), None)

    st.markdown("### 🎯 WFS Contender Tiers")
    st.markdown(
        "<div style='color:#475569;font-size:0.95rem;font-weight:600;margin:.15rem 0 .75rem 0;'>"
        "League-relative contender classification built from the locked WFS Power Rankings. "
        "Tiers describe current projection-covered roster strength; they are not championship odds or a rest-of-season forecast."
        "</div>", unsafe_allow_html=True)
    if projection_error:
        st.warning(projection_error)
    if mine:
        st.success(f"Your team is currently **{mine['tier']}** — #{mine['rank']} of {n} with a {mine['score']:.2f} WFS Power Score.")

    import html as _html
    tier_rows = []
    for row in rows:
        is_mine = row["team_id"] == my_id
        style = " style='background:#172554;'" if is_mine else ""
        team_label = _html.escape(row["team"]) + (" • YOU" if is_mine else "")
        tier_rows.append(
            f"<tr{style}><td>#{row['rank']}</td><td>{team_label}</td><td>{row['tier']}</td><td>{row['score']:.2f}</td></tr>"
        )
    st.markdown("""
    <style>
    .wfs-tier-wrap{width:100%;overflow-x:hidden}.wfs-tier-table{width:100%;border-collapse:collapse;table-layout:fixed;background:#0e1117;color:#f8fafc;border-radius:12px;overflow:hidden}
    .wfs-tier-table th,.wfs-tier-table td{border:1px solid #29313d;padding:.58rem .45rem;font-size:.84rem;vertical-align:top}.wfs-tier-table th{background:#171c24;color:#cbd5e1;text-align:left;font-weight:800}
    .wfs-tier-table th:nth-child(1),.wfs-tier-table td:nth-child(1){width:13%}.wfs-tier-table th:nth-child(2),.wfs-tier-table td:nth-child(2){width:43%;overflow-wrap:anywhere}.wfs-tier-table th:nth-child(3),.wfs-tier-table td:nth-child(3){width:27%}.wfs-tier-table th:nth-child(4),.wfs-tier-table td:nth-child(4){width:17%;text-align:right}
    @media(max-width:600px){.wfs-tier-table th,.wfs-tier-table td{font-size:.78rem;padding:.50rem .30rem}}
    </style><div class='wfs-tier-wrap'><table class='wfs-tier-table'><thead><tr><th>Rank</th><th>Team</th><th>Tier</th><th>Power</th></tr></thead><tbody>""" + "".join(tier_rows) + "</tbody></table></div>", unsafe_allow_html=True)

    with st.expander("How WFS contender tiers work", expanded=False):
        st.markdown(
            "**Elite:** top quarter of projection-covered teams  \n"
            "**Contender:** 50th–74th percentile  \n"
            "**Competitive:** 25th–49th percentile  \n"
            "**Needs Help:** bottom quarter  \n\n"
            "The ordering follows the WFS Power Score using projected QB, RB, WR and TE roster strength. "
            "No championship probability, schedule simulation, ESPN standings prediction, or acceptance model is added."
        )


def _wfs_render_command_center_navigation():
    """Phase 16: interactive mobile-first in-page navigation. UX only; no intelligence changes."""
    st.markdown("### 📱 WFS Command Center")
    st.markdown(
        """
        <div style="color:#475569;font-size:.92rem;font-weight:600;margin:.05rem 0 .65rem;">
          Tap a workspace to jump directly to that part of your WFS command center.
          All intelligence continues to use the locked engines.
        </div>
        <div style="
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:.55rem;
            margin:.15rem 0 .9rem;
        ">
          <a href="#wfs-this-week" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">⚡ This Week</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Summary · readiness · alerts</div>
            </div>
          </a>
          <a href="#wfs-lineup" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">🏈 Lineup</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Matchup · Start/Sit · game plan</div>
            </div>
          </a>
          <a href="#wfs-waivers" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">🎯 Waivers</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Inferred unrostered candidates</div>
            </div>
          </a>
          <a href="#wfs-trades" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">🤝 Trades</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Roster-fit target intelligence</div>
            </div>
          </a>
          <a href="#wfs-league" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">🏆 League</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Power rankings · contender tiers</div>
            </div>
          </a>
          <a href="#wfs-roster" style="text-decoration:none;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">💪 My Roster</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Strengths · needs · improvement plan</div>
            </div>
          </a>
          <a href="#wfs-watchlist" style="text-decoration:none;grid-column:1 / -1;">
            <div style="height:100%;background:#0f172a;border:1px solid #334155;border-radius:13px;padding:.7rem .75rem;color:#f8fafc;">
              <div style="font-weight:900;">⭐ My Watchlist</div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">Saved players · waiver targets · trade targets · breakout watches</div>
            </div>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
def _wfs_projection_mover_record(lookup):
    """Persist changed observations from the locked WFS projection lookup."""
    if not lookup:
        return

    with _wfs_accounts_connect() as conn:
        for key, current in lookup.items():
            proj = current.get("projection")
            try:
                proj = float(proj) if proj is not None else None
            except Exception:
                proj = None

            name = str(current.get("player") or key)
            pos = str(current.get("position") or "").upper()
            team = str(current.get("team") or "").upper()
            status = str(current.get("production_status") or "NOT_IN_CURRENT_SOURCE")

            last = conn.execute(
                """
                SELECT projection, production_status, team
                FROM wfs_projection_mover_history
                WHERE player_key = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (key,),
            ).fetchone()

            changed = last is None
            if last is not None:
                old_proj = last["projection"]
                if old_proj is None and proj is not None:
                    changed = True
                elif old_proj is not None and proj is None:
                    changed = True
                elif old_proj is not None and proj is not None and abs(float(old_proj) - proj) >= 0.01:
                    changed = True
                if str(last["production_status"] or "") != status:
                    changed = True
                if str(last["team"] or "") != team:
                    changed = True

            if changed:
                conn.execute(
                    """
                    INSERT INTO wfs_projection_mover_history(
                        player_key, player_name, position, team,
                        projection, production_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, name, pos, team, proj, status),
                )


def _wfs_projection_mover_rows(lookup):
    """Return players with at least two real stored projection observations."""
    _wfs_projection_mover_record(lookup)
    rows = []

    with _wfs_accounts_connect() as conn:
        for key, current in lookup.items():
            hist = conn.execute(
                """
                SELECT projection, production_status, team, observed_at
                FROM wfs_projection_mover_history
                WHERE player_key = ? AND projection IS NOT NULL
                ORDER BY id ASC
                """,
                (key,),
            ).fetchall()
            if len(hist) < 2:
                continue

            first = float(hist[0]["projection"])
            latest = float(hist[-1]["projection"])
            delta = latest - first
            rows.append({
                "player": str(current.get("player") or key),
                "position": str(current.get("position") or "").upper(),
                "team": str(current.get("team") or "").upper(),
                "projection": latest,
                "delta": delta,
                "observations": len(hist),
                "status": str(current.get("production_status") or ""),
            })

    return rows


def _wfs_render_projection_movers():
    """Phase 20: league-independent movement radar from locked WFS outputs."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        return

    rows = _wfs_projection_mover_rows(lookup)

    st.markdown("### 🚀 WFS Projection Movers")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          Movement radar across the locked WFS projection source. WFS records only actual
          changed observations; refreshing the page does not manufacture movement.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info(
            "Projection-mover baseline established. WFS needs a future changed projection "
            "observation before risers or fallers can be identified."
        )
        st.caption(
            "Projection movement appears only after WFS records a real change in a player's projection."
        )
        return

    meaningful = [r for r in rows if abs(float(r["delta"])) >= 1.0]
    risers = sorted(
        [r for r in meaningful if r["delta"] > 0],
        key=lambda r: (-r["delta"], r["player"]),
    )[:5]
    fallers = sorted(
        [r for r in meaningful if r["delta"] < 0],
        key=lambda r: (r["delta"], r["player"]),
    )[:5]

    import html as _html

    def _cards(title, icon, items):
        st.markdown(f"#### {icon} {title}")
        if not items:
            st.caption("No ≥1.00-point movers in this direction yet.")
            return
        for row in items:
            delta = float(row["delta"])
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                            padding:.72rem .82rem;margin:.38rem 0;color:#f8fafc;">
                  <div style="display:flex;justify-content:space-between;gap:.65rem;">
                    <div>
                      <div style="font-size:1rem;font-weight:900;">
                        {_html.escape(row["player"])}
                      </div>
                      <div style="font-size:.78rem;color:#cbd5e1;margin-top:.12rem;">
                        {_html.escape(row["position"] or "—")} · {_html.escape(row["team"] or "—")} ·
                        {row["projection"]:.2f} WFS · {_html.escape(row["status"])}
                      </div>
                    </div>
                    <div style="font-size:1rem;font-weight:900;white-space:nowrap;">
                      {delta:+.2f}
                    </div>
                  </div>
                  <div style="font-size:.73rem;color:#93c5fd;margin-top:.3rem;font-weight:750;">
                    {row["observations"]} changed observations stored
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _cards("Top Risers", "📈", risers)
    _cards("Top Fallers", "📉", fallers)

    with st.expander("How WFS Projection Movers works"):
        st.write(
            "A player needs at least two genuinely changed stored projection observations. "
            "The displayed change compares the latest WFS projection with the first recorded projection. "
            "The riser/faller board requires an absolute movement of at least 1.00 WFS point."
        )
        st.write(
            "Projection movement is informational and does not make ESPN roster changes."
        )



def _wfs_projection_movement_summary_rows(lookup):
    """Summarize Phase 20's real stored movement observations by position."""
    rows = _wfs_projection_mover_rows(lookup)
    positions = ("QB", "RB", "WR", "TE")
    summary = []

    for pos in positions:
        pos_rows = [r for r in rows if str(r.get("position") or "").upper() == pos]
        meaningful = [r for r in pos_rows if abs(float(r.get("delta") or 0.0)) >= 1.0]
        risers = [r for r in meaningful if float(r.get("delta") or 0.0) > 0]
        fallers = [r for r in meaningful if float(r.get("delta") or 0.0) < 0]

        biggest_riser = max(risers, key=lambda r: float(r["delta"])) if risers else None
        biggest_faller = min(fallers, key=lambda r: float(r["delta"])) if fallers else None

        summary.append({
            "position": pos,
            "tracked": len(pos_rows),
            "movers": len(meaningful),
            "risers": len(risers),
            "fallers": len(fallers),
            "biggest_riser": biggest_riser,
            "biggest_faller": biggest_faller,
        })

    return rows, summary


def _wfs_render_projection_movement_summary():
    """Phase 21: compact league-wide movement pulse from locked Phase 20 history."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        return

    rows, summary = _wfs_projection_movement_summary_rows(lookup)
    meaningful = [r for r in rows if abs(float(r.get("delta") or 0.0)) >= 1.0]
    risers = [r for r in meaningful if float(r.get("delta") or 0.0) > 0]
    fallers = [r for r in meaningful if float(r.get("delta") or 0.0) < 0]

    st.markdown("### 🌡️ WFS Projection Movement Pulse")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          A quick look at meaningful player projection movement across the league.
          This section summarizes the signal; it does not create new projections or synthetic history.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info(
            "Movement Pulse is waiting for player projection changes. "
            "This section will update automatically when meaningful movement is recorded."
        )
        return

    if not meaningful:
        st.info(
            f"WFS has repeat projection observations for {len(rows)} players, but no player has "
            "moved by at least 1.00 WFS point from its first recorded projection yet."
        )
    else:
        biggest_up = max(risers, key=lambda r: float(r["delta"])) if risers else None
        biggest_down = min(fallers, key=lambda r: float(r["delta"])) if fallers else None

        import html as _html

        up_text = (
            f"{biggest_up['player']} {biggest_up['delta']:+.2f}"
            if biggest_up else "None"
        )
        down_text = (
            f"{biggest_down['player']} {biggest_down['delta']:+.2f}"
            if biggest_down else "None"
        )

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                        padding:.85rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.65rem;">
                <div>
                  <div style="font-size:.66rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">RISERS</div>
                  <div style="font-size:1.35rem;font-weight:900;">{len(risers)}</div>
                </div>
                <div>
                  <div style="font-size:.66rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">FALLERS</div>
                  <div style="font-size:1.35rem;font-weight:900;">{len(fallers)}</div>
                </div>
              </div>
              <div style="border-top:1px solid #334155;margin:.7rem 0 .55rem;"></div>
              <div style="font-size:.80rem;color:#e2e8f0;line-height:1.55;">
                📈 Biggest rise: <b>{_html.escape(up_text)}</b><br>
                📉 Biggest fall: <b>{_html.escape(down_text)}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    table_rows = []
    for item in summary:
        up = item["biggest_riser"]
        down = item["biggest_faller"]
        table_rows.append({
            "Pos": item["position"],
            "Tracked": item["tracked"],
            "Movers": item["movers"],
            "Up": item["risers"],
            "Down": item["fallers"],
            "Top Rise": f"{up['player']} {up['delta']:+.2f}" if up else "—",
            "Top Fall": f"{down['player']} {down['delta']:+.2f}" if down else "—",
        })

    with st.expander("View movement by position"):
        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "Meaningful movement requires an absolute change of at least 1.00 WFS point from the "
        "first recorded projection. Movement tracking is informational and does not make ESPN roster changes."
    )



def _wfs_roster_projection_movement_rows(my_team, lookup):
    """Join the user's exact ESPN roster to Phase 20's stored projection movement."""
    if not my_team or not lookup:
        return []

    mover_rows = _wfs_projection_mover_rows(lookup)
    mover_by_key = {
        normalize_name(r.get("player")): r
        for r in mover_rows
        if normalize_name(r.get("player"))
    }

    rows = []
    for roster_row in _wfs_int_roster_rows(my_team):
        name = str(roster_row.get("player") or "").strip()
        key = normalize_name(name)
        if not key or key not in mover_by_key:
            continue

        move = mover_by_key[key]
        delta = float(move.get("delta") or 0.0)
        rows.append({
            "player": name,
            "position": str(move.get("position") or roster_row.get("position") or "").upper(),
            "team": str(move.get("team") or "").upper(),
            "projection": float(move["projection"]) if move.get("projection") is not None else None,
            "delta": delta,
            "observations": int(move.get("observations") or 0),
            "status": str(move.get("status") or ""),
        })

    return rows


def _wfs_render_roster_projection_movement(my_team):
    """Phase 22: show which real Phase 20 movers directly affect the user's roster."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup or not my_team:
        return

    rows = _wfs_roster_projection_movement_rows(my_team, lookup)
    meaningful = [r for r in rows if abs(float(r.get("delta") or 0.0)) >= 1.0]
    risers = sorted(
        [r for r in meaningful if r["delta"] > 0],
        key=lambda r: (-r["delta"], r["player"]),
    )
    fallers = sorted(
        [r for r in meaningful if r["delta"] < 0],
        key=lambda r: (r["delta"], r["player"]),
    )

    st.markdown("### 🧭 My Roster Movement Impact")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          WFS highlights projection movement for players currently on your ESPN roster.
          This answers a simple question: which real projection movers directly affect my team?
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info(
            "Roster movement baseline established. None of your current ESPN roster players has "
            "another meaningful projection change yet."
        )
        st.caption(
            "Roster movement updates only when a meaningful player projection change is recorded."
        )
        return

    if not meaningful:
        st.info(
            f"WFS has repeat observations for {len(rows)} player(s) on your roster, but none has moved "
            "by at least 1.00 WFS point from the first recorded projection."
        )
    else:
        import html as _html
        net = sum(float(r["delta"]) for r in meaningful)
        direction = "positive" if net > 0 else "negative" if net < 0 else "neutral"

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                        padding:.85rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;">
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.55rem;">
                <div>
                  <div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">ROSTER MOVERS</div>
                  <div style="font-size:1.28rem;font-weight:900;">{len(meaningful)}</div>
                </div>
                <div>
                  <div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">RISERS</div>
                  <div style="font-size:1.28rem;font-weight:900;">{len(risers)}</div>
                </div>
                <div>
                  <div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">FALLERS</div>
                  <div style="font-size:1.28rem;font-weight:900;">{len(fallers)}</div>
                </div>
              </div>
              <div style="border-top:1px solid #334155;margin:.7rem 0 .5rem;"></div>
              <div style="font-size:.80rem;color:#e2e8f0;">
                Net tracked roster movement: <b>{net:+.2f} WFS</b> ({_html.escape(direction)}).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for row in risers + fallers:
            icon = "📈" if row["delta"] > 0 else "📉"
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                            padding:.72rem .82rem;margin:.38rem 0;color:#f8fafc;">
                  <div style="display:flex;justify-content:space-between;gap:.65rem;">
                    <div>
                      <div style="font-size:1rem;font-weight:900;">
                        {_html.escape(row["player"])}
                      </div>
                      <div style="font-size:.78rem;color:#cbd5e1;margin-top:.12rem;">
                        {_html.escape(row["position"] or "—")} · {_html.escape(row["team"] or "—")} ·
                        {row["projection"]:.2f} WFS · {_html.escape(row["status"])}
                      </div>
                    </div>
                    <div style="font-size:1rem;font-weight:900;white-space:nowrap;">
                      {icon} {row["delta"]:+.2f}
                    </div>
                  </div>
                  <div style="font-size:.73rem;color:#93c5fd;margin-top:.3rem;font-weight:750;">
                    {row["observations"]} changed observations stored
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("How roster movement impact works"):
        st.write(
            "WFS compares your current ESPN roster with previously recorded player projections. "
            "A player appears as a meaningful mover after a change of at least 1.00 WFS point."
        )
        st.write(
            "The net tracked movement is descriptive only. It is not a matchup projection, rest-of-season "
            "forecast, trade value, waiver recommendation, or lineup transaction."
        )

    st.caption(
        "Projection movement is informational and does not make roster changes."
    )



def _wfs_render_weekly_projection_change_digest(my_team):
    """Phase 23: compact weekly digest of real stored projection changes affecting this roster."""
    if not my_team:
        return

    lookup, _ = _wfs_int_projection_lookup()
    if not lookup:
        return

    league_rows = _wfs_projection_mover_rows(lookup)
    roster_rows = _wfs_roster_projection_movement_rows(my_team, lookup)

    league_meaningful = [
        r for r in league_rows
        if abs(float(r.get("delta") or 0.0)) >= 1.0
    ]
    roster_meaningful = [
        r for r in roster_rows
        if abs(float(r.get("delta") or 0.0)) >= 1.0
    ]

    st.markdown("### 🗞️ WFS Projection Change Digest")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          A weekly look at meaningful player projection changes, with your ESPN roster
          separated from league-wide movement.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not league_meaningful:
        st.info(
            "No player has moved by at least 1.00 WFS point from the first projection recorded this week yet."
        )
        st.caption(
            "This section will update automatically when meaningful projection movement is recorded."
        )
        return

    import html as _html

    league_up = sorted(
        [r for r in league_meaningful if float(r.get("delta") or 0.0) > 0],
        key=lambda r: (-float(r["delta"]), str(r.get("player") or "")),
    )
    league_down = sorted(
        [r for r in league_meaningful if float(r.get("delta") or 0.0) < 0],
        key=lambda r: (float(r["delta"]), str(r.get("player") or "")),
    )
    roster_up = sorted(
        [r for r in roster_meaningful if float(r.get("delta") or 0.0) > 0],
        key=lambda r: (-float(r["delta"]), str(r.get("player") or "")),
    )
    roster_down = sorted(
        [r for r in roster_meaningful if float(r.get("delta") or 0.0) < 0],
        key=lambda r: (float(r["delta"]), str(r.get("player") or "")),
    )

    biggest_up = league_up[0] if league_up else None
    biggest_down = league_down[0] if league_down else None

    up_text = (
        f"{biggest_up['player']} {float(biggest_up['delta']):+.2f}"
        if biggest_up else "None"
    )
    down_text = (
        f"{biggest_down['player']} {float(biggest_down['delta']):+.2f}"
        if biggest_down else "None"
    )

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                    padding:.9rem;margin:.35rem 0 .75rem;color:#f8fafc;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;">
            <div>
              <div style="font-size:.65rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">LEAGUE MOVERS</div>
              <div style="font-size:1.35rem;font-weight:900;">{len(league_meaningful)}</div>
            </div>
            <div>
              <div style="font-size:.65rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">MY ROSTER MOVERS</div>
              <div style="font-size:1.35rem;font-weight:900;">{len(roster_meaningful)}</div>
            </div>
          </div>
          <div style="border-top:1px solid #334155;margin:.7rem 0 .55rem;"></div>
          <div style="font-size:.80rem;color:#e2e8f0;line-height:1.6;">
            📈 League's biggest rise: <b>{_html.escape(up_text)}</b><br>
            📉 League's biggest fall: <b>{_html.escape(down_text)}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if roster_meaningful:
        st.markdown(
            '<div style="font-size:.76rem;font-weight:900;letter-spacing:.07em;color:#475569;margin:.2rem 0 .3rem;">MY ROSTER CHANGES</div>',
            unsafe_allow_html=True,
        )
        for row in (roster_up + roster_down):
            delta = float(row.get("delta") or 0.0)
            icon = "📈" if delta > 0 else "📉"
            proj = row.get("projection")
            proj_text = f"{float(proj):.2f} WFS" if proj is not None else "No current projection"
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                            padding:.72rem .82rem;margin:.35rem 0;color:#f8fafc;">
                  <div style="display:flex;justify-content:space-between;gap:.65rem;">
                    <div>
                      <div style="font-weight:900;">{_html.escape(str(row.get("player") or ""))}</div>
                      <div style="font-size:.76rem;color:#cbd5e1;margin-top:.12rem;">
                        {_html.escape(str(row.get("position") or "—"))} · {_html.escape(str(row.get("team") or "—"))} · {proj_text}
                      </div>
                    </div>
                    <div style="font-weight:900;white-space:nowrap;">{icon} {delta:+.2f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success(
            "League movement exists, but none of the meaningful movers is currently on your ESPN roster."
        )

    with st.expander("How the weekly change digest works"):
        st.write(
            "WFS compares recorded projection movement with your current ESPN roster "
            "to separate changes affecting your team from league-wide movers."
        )
        st.write(
            "A meaningful change requires at least 1.00 WFS point from the first recorded projection. "
            "The digest is informational and does not make roster moves."
        )

    st.caption(
        "Projection changes are informational and do not alter your ESPN roster."
    )



def _wfs_decision_current_week(my_team, authoritative_week=None):
    """Resolve the tracker week without silently inventing a Week 1 fallback."""
    if authoritative_week is not None:
        try:
            value = int(authoritative_week)
            return value if value > 0 else None
        except Exception:
            pass

    if isinstance(my_team, dict):
        for key in ("current_week", "week", "currentMatchupPeriod", "currentScoringPeriod"):
            try:
                value = my_team.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
            except Exception:
                pass
        return None

    for attr in ("current_week", "week"):
        try:
            value = getattr(my_team, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            pass
    return None


def _wfs_capture_weekly_start_sit_decisions(
    user_sub, my_team, season, league_id, authoritative_week=None
):
    """Snapshot locked Start/Sit recommendations for later audit."""
    if not user_sub or not my_team:
        return []

    week = _wfs_decision_current_week(my_team, authoritative_week)
    if week is None:
        return []

    lookup, _ = _wfs_int_projection_lookup()
    if not lookup:
        return []

    roster = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    swaps = _wfs_int_best_swaps(roster)

    with _wfs_accounts_connect() as conn:
        for rec in swaps:
            pos = str(rec.get("Pos") or "")
            sit = str(rec.get("Sit") or "")
            start = str(rec.get("Start") or "")
            sit_proj = rec.get("Sit Proj")
            start_proj = rec.get("Start Proj")
            gain = rec.get("Gain")
            try:
                sit_proj = float(sit_proj) if sit_proj is not None else None
            except Exception:
                sit_proj = None
            try:
                start_proj = float(start_proj) if start_proj is not None else None
            except Exception:
                start_proj = None
            try:
                gain = float(gain) if gain is not None else None
            except Exception:
                gain = None

            decision_key = f"START_SIT|{normalize_name(sit)}|{normalize_name(start)}|{pos}"
            conn.execute(
                """
                INSERT INTO wfs_weekly_decisions(
                    user_sub, provider, season, league_id, week,
                    decision_key, decision_type, position,
                    sit_player, start_player,
                    sit_projection, start_projection, projected_gain
                )
                VALUES (?, 'ESPN', ?, ?, ?, ?, 'START_SIT', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_sub, provider, season, league_id, week, decision_key)
                DO NOTHING
                """,
                (
                    user_sub, int(season), str(league_id), int(week),
                    decision_key, pos, sit, start,
                    sit_proj, start_proj, gain,
                ),
            )

        rows = conn.execute(
            """
            SELECT id, week, decision_type, position, sit_player, start_player,
                   sit_projection, start_projection, projected_gain, status, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN' AND season = ? AND league_id = ? AND week = ?
            ORDER BY projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id), int(week)),
        ).fetchall()

    return [dict(r) for r in rows]


def _wfs_espn_actual_points_for_week(my_team, week):
    """Exact-name map of ESPN league-scoring actual points for the connected roster."""
    actuals = {}
    try:
        target_week = int(week)
    except Exception:
        return actuals

    entries = ((my_team.get("roster") or {}).get("entries") or [])
    for entry in entries:
        player = ((entry.get("playerPoolEntry") or {}).get("player") or {})
        name = str(player.get("fullName") or "").strip()
        key = normalize_name(name)
        if not key:
            continue

        points = None
        for stat in player.get("stats") or []:
            try:
                scoring_period = int(stat.get("scoringPeriodId"))
                source_id = int(stat.get("statSourceId", 0))
            except Exception:
                continue
            if scoring_period != target_week or source_id != 0:
                continue
            value = stat.get("appliedTotal")
            if value is None:
                continue
            try:
                points = float(value)
                break
            except Exception:
                continue

        if points is not None:
            actuals[key] = points
    return actuals


def _wfs_grade_weekly_decisions(user_sub, my_team, season, league_id, week):
    """Grade only when ESPN exposes actual appliedTotal for BOTH players."""
    if not user_sub or not my_team:
        return []

    actuals = _wfs_espn_actual_points_for_week(my_team, week)

    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, week, decision_type, position, sit_player, start_player,
                   sit_projection, start_projection, projected_gain, status,
                   sit_actual, start_actual, actual_gain, graded_at, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ? AND week = ?
            ORDER BY projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id), int(week)),
        ).fetchall()

        for row in rows:
            if str(row["status"] or "").upper() != "PENDING":
                continue
            sit_key = normalize_name(row["sit_player"] or "")
            start_key = normalize_name(row["start_player"] or "")
            if sit_key not in actuals or start_key not in actuals:
                continue

            sit_actual = float(actuals[sit_key])
            start_actual = float(actuals[start_key])
            actual_gain = start_actual - sit_actual
            status = "HIT" if actual_gain > 0 else ("MISS" if actual_gain < 0 else "PUSH")

            conn.execute(
                """
                UPDATE wfs_weekly_decisions
                SET sit_actual = ?, start_actual = ?, actual_gain = ?,
                    status = ?, graded_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'PENDING'
                """,
                (sit_actual, start_actual, actual_gain, status, int(row["id"])),
            )

        rows = conn.execute(
            """
            SELECT id, week, decision_type, position, sit_player, start_player,
                   sit_projection, start_projection, projected_gain, status,
                   sit_actual, start_actual, actual_gain, graded_at, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ? AND week = ?
            ORDER BY projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id), int(week)),
        ).fetchall()

    return [dict(r) for r in rows]


def _wfs_render_weekly_decision_tracker(my_team, season, league_id, authoritative_week=None):
    """Phase 25: Phase 24 snapshots plus supported ESPN outcome grading."""
    user = _wfs_current_user()
    if not user or not my_team:
        return

    week = _wfs_decision_current_week(my_team, authoritative_week)
    snapshot_rows = _wfs_capture_weekly_start_sit_decisions(
        user["sub"], my_team, season, league_id, authoritative_week=week
    )
    snapshot_week = snapshot_rows[0].get("week") if snapshot_rows else week
    rows = (
        _wfs_grade_weekly_decisions(
            user["sub"], my_team, season, league_id, snapshot_week
        )
        if snapshot_rows else []
    )

    st.markdown("### 📋 WFS Weekly Decision Tracker")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          WFS preserves the locked Start/Sit recommendation snapshot for each week and now grades it
          only when ESPN exposes actual league-scoring points for both players in the decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        week_label = f"Week {week}" if week is not None else "Current week unavailable"
        st.info(f"{week_label}: no positive Start/Sit recommendation is currently available to snapshot.")
        return

    import html as _html
    projected_total = sum(float(r.get("projected_gain") or 0.0) for r in rows)
    resolved = [r for r in rows if str(r.get("status") or "").upper() in {"HIT", "MISS", "PUSH"}]
    hits = [r for r in resolved if str(r.get("status") or "").upper() == "HIT"]
    result_text = f"{len(hits)}/{len(resolved)} HIT" if resolved else "PENDING"

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                    padding:.85rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.55rem;">
            <div><div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">WEEK</div>
              <div style="font-size:1.28rem;font-weight:900;">{week}</div></div>
            <div><div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">RESULTS</div>
              <div style="font-size:1.12rem;font-weight:900;">{_html.escape(result_text)}</div></div>
            <div><div style="font-size:.64rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">PROJECTED GAIN</div>
              <div style="font-size:1.28rem;font-weight:900;">+{projected_total:.2f}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row in rows:
        status = str(row.get("status") or "PENDING").upper()
        status_icon = {"HIT": "✅", "MISS": "❌", "PUSH": "➖", "PENDING": "⏳"}.get(status, "⏳")
        sit_proj = row.get("sit_projection")
        start_proj = row.get("start_projection")
        gain = float(row.get("projected_gain") or 0.0)
        sit_text = f"{float(sit_proj):.2f}" if sit_proj is not None else "—"
        start_text = f"{float(start_proj):.2f}" if start_proj is not None else "—"

        if status in {"HIT", "MISS", "PUSH"}:
            sit_actual = float(row.get("sit_actual") or 0.0)
            start_actual = float(row.get("start_actual") or 0.0)
            actual_gain = float(row.get("actual_gain") or 0.0)
            outcome = f"ESPN actual: {sit_actual:.2f} → {start_actual:.2f} ({actual_gain:+.2f})"
        else:
            outcome = "Waiting for ESPN actual scoring for both players."

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;
                        padding:.75rem .82rem;margin:.38rem 0;color:#f8fafc;">
              <div style="font-size:.66rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">
                {_html.escape(str(row.get("position") or "—"))} · {status_icon} {_html.escape(status)}
              </div>
              <div style="font-size:.88rem;line-height:1.55;margin-top:.18rem;">
                Sit <b>{_html.escape(str(row.get("sit_player") or ""))}</b> ({sit_text})<br>
                Start <b>{_html.escape(str(row.get("start_player") or ""))}</b> ({start_text})
              </div>
              <div style="font-size:.80rem;color:#93c5fd;font-weight:850;margin-top:.25rem;">
                Snapshot projected gain: +{gain:.2f} WFS
              </div>
              <div style="font-size:.78rem;color:#e2e8f0;margin-top:.22rem;">{_html.escape(outcome)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not resolved:
        st.info(
            "Outcome grading is waiting for ESPN actual scoring. WFS will not convert missing results "
            "into zeroes or guess whether a recommendation won."
        )

    with st.expander("How Decision Outcome Grading works"):
        st.write(
            "WFS keeps the original Start/Sit recommendation and compares it with the final "
            "ESPN fantasy scoring for both players."
        )
        st.write(
            "HIT means the recommended Start player actually outscored the Sit player; MISS means the Sit player "
            "outscored the recommended Start; PUSH means they scored exactly the same. Missing actual scoring stays PENDING."
        )

    st.caption(
        "Outcome grading compares the original recommendation with completed ESPN fantasy scoring."
    )



def _wfs_recommendation_accuracy_rows(user_sub, season, league_id):
    """Read only persisted Phase 25 grades for this user/league/season."""
    if not user_sub:
        return []

    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT week, decision_type, position, sit_player, start_player,
                   projected_gain, status, sit_actual, start_actual,
                   actual_gain, graded_at, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ?
            ORDER BY week ASC, projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id)),
        ).fetchall()
    return [dict(r) for r in rows]


def _wfs_render_recommendation_accuracy(season, league_id):
    """Phase 26: aggregate only resolved Phase 25 recommendation outcomes."""
    user = _wfs_current_user()
    if not user:
        return

    rows = _wfs_recommendation_accuracy_rows(user["sub"], season, league_id)
    resolved = [
        r for r in rows
        if str(r.get("status") or "").upper() in {"HIT", "MISS", "PUSH"}
    ]
    pending = [
        r for r in rows
        if str(r.get("status") or "").upper() == "PENDING"
    ]

    st.markdown("### 🎯 WFS Recommendation Accuracy")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          A season-to-date scorecard of completed WFS Start/Sit recommendations.
          Pending decisions stay separate until final ESPN scoring is available.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not resolved:
        st.info(
            "Recommendation accuracy is waiting for completed ESPN outcomes. "
            "No Start/Sit recommendation has a final result yet."
        )
        if pending:
            st.caption(
                f"{len(pending)} recommendation{'s are' if len(pending) != 1 else ' is'} currently PENDING."
            )
        else:
            st.caption("No recommendation snapshots are available for this league and season yet.")
        return

    import html as _html

    hits = [r for r in resolved if str(r.get("status") or "").upper() == "HIT"]
    misses = [r for r in resolved if str(r.get("status") or "").upper() == "MISS"]
    pushes = [r for r in resolved if str(r.get("status") or "").upper() == "PUSH"]

    decisions_with_side = len(hits) + len(misses)
    hit_rate = (len(hits) / decisions_with_side * 100.0) if decisions_with_side else None
    total_actual_gain = sum(float(r.get("actual_gain") or 0.0) for r in resolved)
    avg_actual_gain = total_actual_gain / len(resolved) if resolved else 0.0

    hit_rate_text = f"{hit_rate:.1f}%" if hit_rate is not None else "—"

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                    padding:.85rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.55rem;">
            <div>
              <div style="font-size:.62rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">HIT RATE</div>
              <div style="font-size:1.22rem;font-weight:900;">{_html.escape(hit_rate_text)}</div>
            </div>
            <div>
              <div style="font-size:.62rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">RECORD</div>
              <div style="font-size:1.12rem;font-weight:900;">{len(hits)}-{len(misses)}-{len(pushes)}</div>
            </div>
            <div>
              <div style="font-size:.62rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">ACTUAL GAIN</div>
              <div style="font-size:1.22rem;font-weight:900;">{total_actual_gain:+.2f}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"{len(resolved)} graded decision{'s' if len(resolved) != 1 else ''} • "
        f"{len(pending)} pending • average realized Start-vs-Sit difference {avg_actual_gain:+.2f} points."
    )

    by_position = {}
    for row in resolved:
        pos = str(row.get("position") or "—").upper()
        bucket = by_position.setdefault(pos, {"hit": 0, "miss": 0, "push": 0, "gain": 0.0})
        status = str(row.get("status") or "").upper()
        if status == "HIT":
            bucket["hit"] += 1
        elif status == "MISS":
            bucket["miss"] += 1
        else:
            bucket["push"] += 1
        bucket["gain"] += float(row.get("actual_gain") or 0.0)

    st.markdown(
        '<div style="font-size:.72rem;font-weight:900;letter-spacing:.07em;color:#475569;margin:.55rem 0 .25rem;">BY POSITION</div>',
        unsafe_allow_html=True,
    )
    for pos in sorted(by_position):
        bucket = by_position[pos]
        sided = bucket["hit"] + bucket["miss"]
        pos_rate = (bucket["hit"] / sided * 100.0) if sided else None
        pos_rate_text = f"{pos_rate:.1f}%" if pos_rate is not None else "—"
        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
                        padding:.62rem .75rem;margin:.3rem 0;color:#f8fafc;">
              <div style="display:flex;justify-content:space-between;gap:.6rem;">
                <div style="font-weight:900;">{_html.escape(pos)}</div>
                <div style="font-weight:900;color:#93c5fd;">{_html.escape(pos_rate_text)}</div>
              </div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">
                {bucket["hit"]} HIT · {bucket["miss"]} MISS · {bucket["push"]} PUSH ·
                realized difference {bucket["gain"]:+.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How Recommendation Accuracy works"):
        st.write(
            "A HIT means the recommended Start player outscored the Sit player in ESPN league scoring. "
            "MISS means the Sit player scored more, and PUSH means they finished even."
        )
        st.write(
            "Hit rate uses HIT / (HIT + MISS), so PUSH results do not distort the percentage. "
            "PENDING recommendations are displayed separately and excluded until ESPN actual scoring resolves them."
        )

    st.caption(
        "Recommendation accuracy is based on completed ESPN fantasy results."
    )



def _wfs_weekly_results_rows(user_sub, season, league_id):
    """Read persisted Phase 25 grades for Phase 27 weekly rollups."""
    if not user_sub:
        return []
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT week, decision_type, position, sit_player, start_player,
                   projected_gain, status, sit_actual, start_actual,
                   actual_gain, graded_at, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ?
            ORDER BY week DESC, projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id)),
        ).fetchall()
    return [dict(r) for r in rows]


def _wfs_render_weekly_results_summary(season, league_id):
    """Phase 27: compact week-by-week summary of completed Phase 25 grades."""
    user = _wfs_current_user()
    if not user:
        return

    rows = _wfs_weekly_results_rows(user["sub"], season, league_id)
    st.markdown("### 🧾 WFS Weekly Results Summary")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          A week-by-week history of WFS Start/Sit recommendations and how those players
          actually performed in ESPN fantasy scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("No weekly recommendation snapshots are available for this league and season yet.")
        return

    by_week = {}
    for row in rows:
        by_week.setdefault(int(row.get("week") or 0), []).append(row)

    completed_weeks, waiting_weeks = [], []
    for week, week_rows in sorted(by_week.items(), reverse=True):
        resolved = [r for r in week_rows if str(r.get("status") or "").upper() in {"HIT", "MISS", "PUSH"}]
        pending = [r for r in week_rows if str(r.get("status") or "").upper() == "PENDING"]
        (completed_weeks if resolved else waiting_weeks).append(
            (week, week_rows, resolved, pending) if resolved else (week, week_rows)
        )

    if not completed_weeks:
        latest_week = max(by_week) if by_week else None
        pending_count = sum(1 for r in rows if str(r.get("status") or "").upper() == "PENDING")
        st.info(
            "Weekly Results Summary is waiting for completed ESPN outcomes. "
            "No week has a completed Start/Sit result yet."
        )
        if latest_week is not None:
            st.caption(
                f"Week {latest_week} is currently waiting with {pending_count} pending "
                f"recommendation{'s' if pending_count != 1 else ''} across the season."
            )
        return

    import html as _html
    for week, week_rows, resolved, pending in completed_weeks:
        hits = [r for r in resolved if str(r.get("status") or "").upper() == "HIT"]
        misses = [r for r in resolved if str(r.get("status") or "").upper() == "MISS"]
        pushes = [r for r in resolved if str(r.get("status") or "").upper() == "PUSH"]
        actual_gain = sum(float(r.get("actual_gain") or 0.0) for r in resolved)
        projected_gain = sum(float(r.get("projected_gain") or 0.0) for r in week_rows)
        sided = len(hits) + len(misses)
        hit_rate = (len(hits) / sided * 100.0) if sided else None
        hit_rate_text = f"{hit_rate:.0f}%" if hit_rate is not None else "—"
        record_text = f"{len(hits)}-{len(misses)}-{len(pushes)}"
        if actual_gain > 0:
            week_label, week_icon = "POSITIVE", "✅"
        elif actual_gain < 0:
            week_label, week_icon = "NEGATIVE", "❌"
        else:
            week_label, week_icon = "EVEN", "➖"

        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                        padding:.82rem .88rem;margin:.42rem 0;color:#f8fafc;">
              <div style="display:flex;justify-content:space-between;gap:.7rem;align-items:flex-start;">
                <div>
                  <div style="font-size:.64rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;">
                    WEEK {week} · {week_icon} {_html.escape(week_label)}
                  </div>
                  <div style="font-size:1.12rem;font-weight:900;margin-top:.12rem;">
                    {len(resolved)} graded · {len(pending)} pending
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:.64rem;font-weight:900;color:#93c5fd;">ACTUAL GAIN</div>
                  <div style="font-size:1.18rem;font-weight:900;">{actual_gain:+.2f}</div>
                </div>
              </div>
              <div style="font-size:.78rem;color:#cbd5e1;line-height:1.55;margin-top:.35rem;">
                Record {record_text} · Hit rate {hit_rate_text}<br>
                Snapshot projected gain {projected_gain:+.2f} WFS
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Week {week} decision details"):
            for row in resolved:
                status = str(row.get("status") or "").upper()
                icon = {"HIT": "✅", "MISS": "❌", "PUSH": "➖"}.get(status, "•")
                actual = float(row.get("actual_gain") or 0.0)
                st.write(
                    f"{icon} {row.get('position') or '—'}: "
                    f"{row.get('start_player') or '—'} over {row.get('sit_player') or '—'} "
                    f"• {status} • actual difference {actual:+.2f}"
                )
            if pending:
                st.caption(f"{len(pending)} decision{'s remain' if len(pending) != 1 else ' remains'} pending for this week.")

    if waiting_weeks:
        waiting_text = ", ".join(f"Week {week}" for week, _ in waiting_weeks[:4])
        if len(waiting_weeks) > 4:
            waiting_text += f" +{len(waiting_weeks) - 4} more"
        st.caption(f"Still waiting on ESPN outcomes: {waiting_text}.")

    with st.expander("How the Weekly Results Summary works"):
        st.write(
            "Phase 27 groups the persistent Phase 25 grades by week. It reports the recommendation record, "
            "hit rate, original projected gain, and realized Start-vs-Sit scoring difference for each completed week."
        )
        st.write(
            "A week can contain both completed and pending decisions. Missing ESPN results stay pending and are "
            "never treated as zeroes, misses, or completed outcomes."
        )

    st.caption(
        "Phase 27 is reporting only. It does not alter projections, grading rules, recommendation snapshots, "
        "the optimizer, Start/Sit logic, or ESPN rosters."
    )



def _wfs_long_term_performance_rows(user_sub, season, league_id):
    """Read persisted Phase 25 grades for the Phase 28 season performance view."""
    if not user_sub:
        return []
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT week, decision_type, position, sit_player, start_player,
                   projected_gain, status, sit_actual, start_actual,
                   actual_gain, graded_at, created_at
            FROM wfs_weekly_decisions
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ?
            ORDER BY week ASC, projected_gain DESC, id ASC
            """,
            (user_sub, int(season), str(league_id)),
        ).fetchall()
    return [dict(r) for r in rows]


def _wfs_render_long_term_decision_performance(season, league_id):
    """Phase 28: season-level performance history from completed Phase 25 grades only."""
    user = _wfs_current_user()
    if not user:
        return

    rows = _wfs_long_term_performance_rows(user["sub"], season, league_id)
    resolved = [
        r for r in rows
        if str(r.get("status") or "").upper() in {"HIT", "MISS", "PUSH"}
    ]
    pending = [
        r for r in rows
        if str(r.get("status") or "").upper() == "PENDING"
    ]

    st.markdown("### 📈 WFS Long-Term Decision Performance")
    st.markdown(
        """
        <div style="color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;">
          A season-level history of completed WFS Start/Sit decisions, preserving the original
          recommendation snapshots and the supported ESPN outcomes that graded them.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not resolved:
        st.info(
            "Long-term performance is waiting for completed ESPN outcomes. "
            "WFS will begin building season history after the first completed Start/Sit result."
        )
        if pending:
            weeks = sorted({int(r.get("week") or 0) for r in pending if int(r.get("week") or 0) > 0})
            week_text = ", ".join(str(w) for w in weeks[:6]) if weeks else "—"
            st.caption(
                f"{len(pending)} recommendation{'s are' if len(pending) != 1 else ' is'} pending "
                f"across week{'s' if len(weeks) != 1 else ''} {week_text}."
            )
        return

    import html as _html

    hits = [r for r in resolved if str(r.get("status") or "").upper() == "HIT"]
    misses = [r for r in resolved if str(r.get("status") or "").upper() == "MISS"]
    pushes = [r for r in resolved if str(r.get("status") or "").upper() == "PUSH"]
    sided = len(hits) + len(misses)
    hit_rate = (len(hits) / sided * 100.0) if sided else None
    hit_rate_text = f"{hit_rate:.1f}%" if hit_rate is not None else "—"

    actual_total = sum(float(r.get("actual_gain") or 0.0) for r in resolved)
    projected_total = sum(float(r.get("projected_gain") or 0.0) for r in resolved)
    avg_actual = actual_total / len(resolved) if resolved else 0.0
    completed_weeks = sorted({int(r.get("week") or 0) for r in resolved if int(r.get("week") or 0) > 0})

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:16px;
                    padding:.86rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;">
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.55rem;">
            <div>
              <div style="font-size:.61rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">SEASON HIT RATE</div>
              <div style="font-size:1.20rem;font-weight:900;">{_html.escape(hit_rate_text)}</div>
            </div>
            <div>
              <div style="font-size:.61rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">RECORD</div>
              <div style="font-size:1.12rem;font-weight:900;">{len(hits)}-{len(misses)}-{len(pushes)}</div>
            </div>
            <div>
              <div style="font-size:.61rem;font-weight:900;letter-spacing:.07em;color:#93c5fd;">REALIZED GAIN</div>
              <div style="font-size:1.20rem;font-weight:900;">{actual_total:+.2f}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"{len(resolved)} completed decision{'s' if len(resolved) != 1 else ''} across "
        f"{len(completed_weeks)} graded week{'s' if len(completed_weeks) != 1 else ''} • "
        f"{len(pending)} pending • average realized difference {avg_actual:+.2f}."
    )

    # Chronological week trend, using only completed grades.
    weekly = {}
    for row in resolved:
        week = int(row.get("week") or 0)
        bucket = weekly.setdefault(week, {"hit": 0, "miss": 0, "push": 0, "actual": 0.0, "projected": 0.0})
        status = str(row.get("status") or "").upper()
        if status == "HIT":
            bucket["hit"] += 1
        elif status == "MISS":
            bucket["miss"] += 1
        else:
            bucket["push"] += 1
        bucket["actual"] += float(row.get("actual_gain") or 0.0)
        bucket["projected"] += float(row.get("projected_gain") or 0.0)

    st.markdown(
        '<div style="font-size:.72rem;font-weight:900;letter-spacing:.07em;color:#475569;margin:.58rem 0 .25rem;">SEASON TIMELINE</div>',
        unsafe_allow_html=True,
    )
    cumulative = 0.0
    for week in sorted(weekly):
        bucket = weekly[week]
        cumulative += bucket["actual"]
        sided_week = bucket["hit"] + bucket["miss"]
        rate = (bucket["hit"] / sided_week * 100.0) if sided_week else None
        rate_text = f"{rate:.0f}%" if rate is not None else "—"
        st.markdown(
            f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;
                        padding:.62rem .75rem;margin:.3rem 0;color:#f8fafc;">
              <div style="display:flex;justify-content:space-between;gap:.65rem;">
                <div style="font-weight:900;">Week {week}</div>
                <div style="font-weight:900;color:#93c5fd;">{bucket["actual"]:+.2f}</div>
              </div>
              <div style="font-size:.76rem;color:#cbd5e1;margin-top:.18rem;">
                {bucket["hit"]} HIT · {bucket["miss"]} MISS · {bucket["push"]} PUSH ·
                hit rate {rate_text}<br>
                projected {bucket["projected"]:+.2f} · cumulative realized {cumulative:+.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Best/worst resolved decisions are historical descriptions, not future recommendations.
    best = max(resolved, key=lambda r: float(r.get("actual_gain") or 0.0))
    worst = min(resolved, key=lambda r: float(r.get("actual_gain") or 0.0))

    st.markdown(
        f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:14px;
                    padding:.72rem .78rem;margin:.55rem 0;color:#0f172a;">
          <div style="font-size:.68rem;font-weight:900;letter-spacing:.06em;color:#1d4ed8;">HISTORICAL RANGE</div>
          <div style="font-size:.82rem;line-height:1.55;margin-top:.2rem;">
            Best realized decision: <b>{_html.escape(str(best.get("start_player") or "—"))}</b>
            over {_html.escape(str(best.get("sit_player") or "—"))} ·
            {float(best.get("actual_gain") or 0.0):+.2f}<br>
            Lowest realized decision: <b>{_html.escape(str(worst.get("start_player") or "—"))}</b>
            over {_html.escape(str(worst.get("sit_player") or "—"))} ·
            {float(worst.get("actual_gain") or 0.0):+.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How Long-Term Decision Performance works"):
        st.write(
            "WFS compares saved Start/Sit recommendations with completed ESPN fantasy results "
            "throughout the connected league season."
        )
        st.write(
            "The timeline is descriptive history only. PENDING decisions remain excluded from completed hit-rate "
            "and realized-gain calculations until ESPN scoring resolves them."
        )

    st.caption(
        "This view summarizes completed fantasy decisions and results from the current season."
    )



def _wfs_weekly_checklist_init():
    """Phase 31 additive persistence for explicit user checklist state."""
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wfs_weekly_action_checklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_sub TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT 'ESPN',
                season INTEGER NOT NULL,
                league_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                action_key TEXT NOT NULL,
                action_type TEXT NOT NULL DEFAULT '',
                action_title TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'OPEN',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_sub, provider, season, league_id, week, action_key),
                FOREIGN KEY (user_sub) REFERENCES wfs_users(user_sub) ON DELETE CASCADE
            )
            """
        )


def _wfs_weekly_checklist_action_key(item):
    raw = "|".join([
        str(item.get("type") or "").strip().upper(),
        str(item.get("title") or "").strip(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _wfs_weekly_checklist_rows(user_sub, season, league_id, week):
    _wfs_weekly_checklist_init()
    with _wfs_accounts_connect() as conn:
        rows = conn.execute(
            """
            SELECT action_key, action_type, action_title, state, updated_at
            FROM wfs_weekly_action_checklist
            WHERE user_sub = ? AND provider = 'ESPN'
              AND season = ? AND league_id = ? AND week = ?
            """,
            (str(user_sub), int(season), str(league_id), int(week)),
        ).fetchall()
    return [dict(r) for r in rows]


def _wfs_weekly_checklist_set_state(user_sub, season, league_id, week, item, state):
    """Persist workflow state only; never alter WFS fantasy intelligence."""
    state = str(state or "OPEN").upper()
    if state not in {"OPEN", "REVIEWED", "COMPLETED"}:
        state = "OPEN"
    key = _wfs_weekly_checklist_action_key(item)
    _wfs_weekly_checklist_init()
    with _wfs_accounts_connect() as conn:
        conn.execute(
            """
            INSERT INTO wfs_weekly_action_checklist
                (user_sub, provider, season, league_id, week, action_key,
                 action_type, action_title, state, updated_at)
            VALUES (?, 'ESPN', ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_sub, provider, season, league_id, week, action_key)
            DO UPDATE SET action_type=excluded.action_type,
                          action_title=excluded.action_title,
                          state=excluded.state,
                          updated_at=CURRENT_TIMESTAMP
            """,
            (str(user_sub), int(season), str(league_id), int(week), key,
             str(item.get("type") or ""), str(item.get("title") or ""), state),
        )


def _wfs_opportunity_ranking_items(my_team, teams, lookup):
    """Rank only opportunities already produced by the locked Action Center."""
    items=[dict(x) for x in _wfs_action_center_items(my_team, teams, lookup)]
    urgency_rank={"NOW":0,"SOON":1,"MONITOR":2}
    for item in items:
        try:
            item["known_gain"]=float(item["gain"]) if item.get("gain") is not None else None
        except (TypeError,ValueError):
            item["known_gain"]=None
    items.sort(key=lambda x:(
        0 if x.get("known_gain") is not None else 1,
        -(x.get("known_gain") or 0.0),
        urgency_rank.get(str(x.get("urgency") or "MONITOR").upper(),9),
        int(x.get("priority") or 999),
        str(x.get("title") or ""),
    ))
    return items


def _wfs_render_opportunity_ranking(teams, my_team):
    st.markdown("### 🎯 WFS Opportunity Ranking")
    st.markdown(
        "<div style='color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;'>"
        "Ranks opportunities already supported by WFS. Known projected benefit comes first; "
        "WFS does not invent a value for opportunities without a comparable projected edge.</div>",
        unsafe_allow_html=True,
    )
    lookup,_=_wfs_int_projection_lookup()
    if not lookup or not my_team:
        st.info("Opportunity Ranking is waiting for the connected ESPN team and WFS projection source.")
        return
    items=_wfs_opportunity_ranking_items(my_team,teams,lookup)
    if not items:
        st.success("No supported WFS opportunities are currently waiting to be ranked.")
        return

    known=[x for x in items if x.get("known_gain") is not None]
    best=known[0]["known_gain"] if known else None
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #334155;border-radius:16px;padding:.82rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;'>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>OPPORTUNITIES</div><div style='font-size:1.16rem;font-weight:900;'>{len(items)}</div></div>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>KNOWN EDGE</div><div style='font-size:1.16rem;font-weight:900;'>{len(known)}</div></div>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>BEST EDGE</div><div style='font-size:1.16rem;font-weight:900;'>{('+'+format(best,'.2f')) if best is not None else '—'}</div></div>"
        f"</div></div>",unsafe_allow_html=True)

    import html as _html
    ui={"NOW":"🔴","SOON":"🟡","MONITOR":"🔵"}
    ti={"LINEUP":"🏈","ROSTER":"🧱","WATCH":"⭐"}
    for idx,item in enumerate(items[:5],1):
        urgency=str(item.get("urgency") or "MONITOR").upper()
        kind=str(item.get("type") or "REVIEW").upper()
        gain=item.get("known_gain")
        evidence=f"Known projected edge +{gain:.2f}" if gain is not None else "No comparable projected edge"
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:14px;padding:.72rem .8rem;margin:.38rem 0;color:#f8fafc;'>"
            f"<div style='font-size:.63rem;font-weight:900;color:#93c5fd;'>#{idx} · {ui.get(urgency,'🔵')} {_html.escape(urgency)} · {ti.get(kind,'📌')} {_html.escape(kind)}</div>"
            f"<div style='font-size:.90rem;font-weight:900;margin-top:.16rem;'>{_html.escape(str(item.get('title') or '—'))}</div>"
            f"<div style='font-size:.72rem;color:#cbd5e1;margin-top:.20rem;'>{_html.escape(evidence)}</div></div>",
            unsafe_allow_html=True)
    st.caption("Ranking is based on the current WFS fantasy information available for your team.")



def _wfs_render_weekly_command_brief(teams, my_team):
    """Phase 33 final compact briefing using only locked WFS action evidence."""
    st.markdown("### 🏁 WFS Weekly Command Brief")
    st.markdown(
        "<div style='color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;'>"
        "Your compact weekly command view, assembled from WFS actions, urgency, and opportunity evidence already supported above.</div>",
        unsafe_allow_html=True)
    lookup,_=_wfs_int_projection_lookup()
    if not lookup or not my_team:
        st.info("Weekly Command Brief is waiting for the connected ESPN team and WFS projection source.")
        return
    items=_wfs_opportunity_ranking_items(my_team,teams,lookup)
    if not items:
        st.success("No supported WFS action is currently waiting for your weekly command brief.")
        return

    now=sum(str(x.get("urgency") or "").upper()=="NOW" for x in items)
    soon=sum(str(x.get("urgency") or "").upper()=="SOON" for x in items)
    monitor=sum(str(x.get("urgency") or "").upper()=="MONITOR" for x in items)
    known=[x for x in items if x.get("known_gain") is not None]
    best=known[0]["known_gain"] if known else None
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #334155;border-radius:16px;padding:.82rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:.42rem;'>"
        f"<div><div style='font-size:.56rem;font-weight:900;color:#93c5fd;'>NOW</div><div style='font-size:1.12rem;font-weight:900;'>{now}</div></div>"
        f"<div><div style='font-size:.56rem;font-weight:900;color:#93c5fd;'>SOON</div><div style='font-size:1.12rem;font-weight:900;'>{soon}</div></div>"
        f"<div><div style='font-size:.56rem;font-weight:900;color:#93c5fd;'>MONITOR</div><div style='font-size:1.12rem;font-weight:900;'>{monitor}</div></div>"
        f"<div><div style='font-size:.56rem;font-weight:900;color:#93c5fd;'>BEST EDGE</div><div style='font-size:1.12rem;font-weight:900;'>{('+'+format(best,'.2f')) if best is not None else '—'}</div></div>"
        f"</div></div>",unsafe_allow_html=True)

    import html as _html
    ui={"NOW":"🔴","SOON":"🟡","MONITOR":"🔵"}
    ti={"LINEUP":"🏈","ROSTER":"🧱","WATCH":"⭐"}
    labels=["TOP PRIORITY","NEXT","KEEP WATCH"]
    for idx,item in enumerate(items[:3]):
        urgency=str(item.get("urgency") or "MONITOR").upper()
        kind=str(item.get("type") or "REVIEW").upper()
        gain=item.get("known_gain")
        evidence=f"+{gain:.2f} projected edge" if gain is not None else "Existing WFS evidence; no comparable projected edge"
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:14px;padding:.70rem .80rem;margin:.36rem 0;color:#f8fafc;'>"
            f"<div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>{labels[idx]}</div>"
            f"<div style='font-size:.68rem;font-weight:900;color:#bfdbfe;margin-top:.13rem;'>{ui.get(urgency,'🔵')} {_html.escape(urgency)} · {ti.get(kind,'📌')} {_html.escape(kind)}</div>"
            f"<div style='font-size:.90rem;font-weight:900;margin-top:.15rem;'>{_html.escape(str(item.get('title') or '—'))}</div>"
            f"<div style='font-size:.71rem;color:#cbd5e1;margin-top:.18rem;'>{_html.escape(evidence)}</div></div>",
            unsafe_allow_html=True)
    st.caption("This brief highlights the most important fantasy decisions currently identified for your team.")



def _wfs_render_weekly_checklist(teams, my_team, season, league_id):
    """Phase 31: personal weekly review/completion state for locked Phase30 actions."""
    lookup, _ = _wfs_int_projection_lookup()
    week = _wfs_decision_current_week(my_team)

    st.markdown("### ✅ WFS Weekly Checklist")
    st.markdown(
        "<div style='color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;'>"
        "Track which current WFS actions you have reviewed or completed this week. "
        "Checklist state is personal workflow only and never changes WFS projections or recommendations.</div>",
        unsafe_allow_html=True,
    )
    user=_wfs_current_user()
    if not user or not lookup or not my_team:
        st.info("Weekly Checklist is waiting for the signed-in user, connected ESPN team, and WFS projection source.")
        return
    if week is None:
        st.info("Weekly Checklist is waiting for ESPN to provide the current scoring week.")
        return

    items=_wfs_action_center_items(my_team, teams, lookup)[:5]
    if not items:
        st.success("No supported WFS action is currently waiting for your checklist.")
        return

    saved={r["action_key"]:str(r.get("state") or "OPEN").upper()
           for r in _wfs_weekly_checklist_rows(user["sub"],season,league_id,week)}
    states=[saved.get(_wfs_weekly_checklist_action_key(x),"OPEN") for x in items]
    open_count=sum(s=="OPEN" for s in states)
    reviewed=sum(s=="REVIEWED" for s in states)
    completed=sum(s=="COMPLETED" for s in states)

    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #334155;border-radius:16px;padding:.82rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;'>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>OPEN</div><div style='font-size:1.16rem;font-weight:900;'>{open_count}</div></div>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>REVIEWED</div><div style='font-size:1.16rem;font-weight:900;'>{reviewed}</div></div>"
        f"<div><div style='font-size:.58rem;font-weight:900;color:#93c5fd;'>COMPLETED</div><div style='font-size:1.16rem;font-weight:900;'>{completed}</div></div>"
        f"</div></div>",unsafe_allow_html=True)

    import html as _html
    icons={"OPEN":"⬜","REVIEWED":"👀","COMPLETED":"✅"}
    for idx,item in enumerate(items,1):
        key=_wfs_weekly_checklist_action_key(item)
        current=saved.get(key,"OPEN")
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:14px;padding:.68rem .78rem .48rem;margin:.38rem 0 .08rem;color:#f8fafc;'>"
            f"<div style='font-size:.64rem;font-weight:900;color:#93c5fd;'>#{idx} · {_html.escape(str(item.get('urgency') or 'MONITOR'))} · {_html.escape(str(item.get('type') or 'REVIEW'))} · {icons.get(current,'⬜')} {_html.escape(current)}</div>"
            f"<div style='font-size:.88rem;font-weight:900;margin-top:.14rem;'>{_html.escape(str(item.get('title') or '—'))}</div>"
            f"</div>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        if c1.button("Open",key=f"wfs_ck_o_{season}_{league_id}_{week}_{key}",use_container_width=True):
            _wfs_weekly_checklist_set_state(user["sub"],season,league_id,week,item,"OPEN"); st.rerun()
        if c2.button("Reviewed",key=f"wfs_ck_r_{season}_{league_id}_{week}_{key}",use_container_width=True):
            _wfs_weekly_checklist_set_state(user["sub"],season,league_id,week,item,"REVIEWED"); st.rerun()
        if c3.button("Completed",key=f"wfs_ck_c_{season}_{league_id}_{week}_{key}",use_container_width=True):
            _wfs_weekly_checklist_set_state(user["sub"],season,league_id,week,item,"COMPLETED"); st.rerun()

    st.caption("Checklist selections help you track your weekly review. They do not make changes in ESPN.")



def _wfs_action_center_items(my_team, teams, lookup):
    """Phase 29: prioritize existing locked WFS signals without creating a new engine."""
    if not my_team or not lookup:
        return []

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    items = []

    # Existing locked Start/Sit recommendations.
    for rec in _wfs_int_best_swaps(my_rows):
        gain = float(rec.get("Gain") or 0.0)
        if gain > 0:
            items.append({
                "priority": 10, "type": "LINEUP", "gain": gain,
                "title": f"Start {rec.get('Start') or '—'} over {rec.get('Sit') or '—'}",
                "detail": f"{rec.get('Pos') or '—'} · locked WFS Start/Sit edge {gain:+.2f}",
            })

    # Reuse the locked Phase 7 room definition exactly: QB1/RB3/WR3/TE1.
    def room_score(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        vals = [float(r["wfs_projection"]) for r in rows
                if str(r.get("position") or r.get("wfs_position") or "").upper() == pos
                and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        vals.sort(reverse=True)
        take = vals[:room_size[pos]]
        return (sum(take) / len(take)) if take else None

    room_pcts = {}
    for pos in positions:
        mine = room_score(my_team, pos)
        peers = [room_score(t, pos) for t in (teams or [])]
        peers = [x for x in peers if x is not None]
        room_pcts[pos] = (
            100.0 * sum(1 for x in peers if x <= mine) / len(peers)
            if mine is not None and peers else None
        )
    rated = [(p, v) for p, v in room_pcts.items() if v is not None]
    if rated:
        need_pos, need_pct = min(rated, key=lambda x: (x[1], x[0]))
        items.append({
            "priority": 30, "type": "ROSTER", "gain": None,
            "title": f"Prioritize {need_pos} depth",
            "detail": f"Locked roster-strength view · weakest room {need_pct:.0f}th percentile",
        })

    # Existing saved watchlist is review context only.
    user = _wfs_current_user()
    watch_rows = _wfs_watchlist_rows(user["sub"]) if user else []
    for row in watch_rows[:3]:
        name = str(row.get("player_name") or "").strip()
        if name:
            items.append({
                "priority": 40, "type": "WATCH", "gain": None,
                "title": f"Review {name}",
                "detail": f"Saved watchlist · {str(row.get('watch_type') or 'Monitor').upper()}",
            })

    # Phase 30 urgency is deterministic orchestration of Phase 29 categories only.
    # It intentionally does not invent kickoff, waiver, injury, or transaction deadlines.
    urgency_map = {
        "LINEUP": ("NOW", 10, "Act on this current-week lineup decision before your lineup locks."),
        "ROSTER": ("SOON", 20, "Address this roster-building need as league opportunities develop."),
        "WATCH": ("MONITOR", 30, "Keep this saved player under review; no transaction is implied."),
    }
    for item in items:
        label, rank, reason = urgency_map.get(
            str(item.get("type") or "").upper(),
            ("MONITOR", 30, "Keep this item under review.")
        )
        item["urgency"] = label
        item["urgency_rank"] = rank
        item["urgency_reason"] = reason

    items.sort(key=lambda r: (
        int(r.get("urgency_rank") or 999),
        int(r.get("priority") or 999),
        -float(r.get("gain") or 0.0),
        str(r.get("title") or ""),
    ))
    return items


def _wfs_render_action_center(teams, my_team):
    """One mobile-first prioritized queue from existing locked WFS intelligence."""
    lookup, _ = _wfs_int_projection_lookup()
    st.markdown('<div id="wfs-action-center"></div>', unsafe_allow_html=True)
    st.markdown("### ⏰ WFS Action Urgency")
    st.markdown(
        "<div style='color:#475569;font-size:.90rem;font-weight:600;margin:.05rem 0 .65rem;'>"
        "Your most important lineup, roster and player-watch items for the current fantasy week.</div>",
        unsafe_allow_html=True,
    )
    if not lookup or not my_team:
        st.info("Action Center is waiting for the connected ESPN team and WFS projection source.")
        return

    items = _wfs_action_center_items(my_team, teams, lookup)
    if not items:
        st.success("No supported WFS action is currently waiting for your attention.")
        return

    import html as _html
    top = items[:5]
    known_edge = sum(float(x.get("gain") or 0.0) for x in top if x.get("gain") is not None)
    now_count = sum(1 for x in top if x.get("urgency") == "NOW")
    soon_count = sum(1 for x in top if x.get("urgency") == "SOON")
    monitor_count = sum(1 for x in top if x.get("urgency") == "MONITOR")

    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #334155;border-radius:16px;padding:.86rem .9rem;margin:.35rem 0 .7rem;color:#f8fafc;'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:.42rem;'>"
        f"<div><div style='font-size:.57rem;font-weight:900;color:#93c5fd;'>NOW</div><div style='font-size:1.12rem;font-weight:900;'>{now_count}</div></div>"
        f"<div><div style='font-size:.57rem;font-weight:900;color:#93c5fd;'>SOON</div><div style='font-size:1.12rem;font-weight:900;'>{soon_count}</div></div>"
        f"<div><div style='font-size:.57rem;font-weight:900;color:#93c5fd;'>MONITOR</div><div style='font-size:1.12rem;font-weight:900;'>{monitor_count}</div></div>"
        f"<div><div style='font-size:.57rem;font-weight:900;color:#93c5fd;'>KNOWN EDGE</div><div style='font-size:1.12rem;font-weight:900;'>{known_edge:+.2f}</div></div>"
        f"</div></div>", unsafe_allow_html=True)

    for idx,item in enumerate(top,1):
        kind=str(item.get("type") or "REVIEW")
        urgency=str(item.get("urgency") or "MONITOR")
        icon={"LINEUP":"🏈","ROSTER":"🧱","WATCH":"⭐"}.get(kind,"•")
        urgency_icon={"NOW":"🔴","SOON":"🟡","MONITOR":"🔵"}.get(urgency,"🔵")
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:14px;padding:.72rem .8rem;margin:.36rem 0;color:#f8fafc;'>"
            f"<div style='font-size:.64rem;font-weight:900;letter-spacing:.08em;color:#93c5fd;'>#{idx} · {urgency_icon} {_html.escape(urgency)} · {icon} {_html.escape(kind)}</div>"
            f"<div style='font-size:.91rem;font-weight:900;margin-top:.16rem;'>{_html.escape(str(item.get('title') or '—'))}</div>"
            f"<div style='font-size:.77rem;color:#cbd5e1;line-height:1.45;margin-top:.2rem;'>{_html.escape(str(item.get('detail') or ''))}</div>"
            f"<div style='font-size:.72rem;color:#94a3b8;line-height:1.4;margin-top:.28rem;'>{_html.escape(str(item.get('urgency_reason') or ''))}</div>"
            f"</div>", unsafe_allow_html=True)

    st.caption(
        "Urgency helps organize which fantasy decisions deserve attention first."
    )



def _wfs_render_weekly_executive_summary(teams, my_team, opponent_team, season, league_id):
    """Phase 12: compact mobile-first summary of already-validated WFS intelligence."""
    if not my_team:
        return
    lookup, _ = _wfs_int_projection_lookup()
    if not lookup:
        return
    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    swaps = _wfs_int_best_swaps(my_rows)

    def room_score(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        vals = [float(r["wfs_projection"]) for r in rows
                if str(r.get("position") or r.get("wfs_position") or "").upper() == pos
                and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        vals.sort(reverse=True)
        take = vals[:room_size[pos]]
        return (sum(take) / len(take)) if take else None

    # Locked Phase 9/10 league context.
    power_rows = []
    for team in teams or []:
        vals = [room_score(team, p) for p in positions]
        if all(v is not None for v in vals):
            power_rows.append((sum(vals), team.get("id"), _espn_team_name(team)))
    power_rows.sort(key=lambda x: (-x[0], x[2].lower()))
    my_rank = next((i for i, r in enumerate(power_rows, 1) if r[1] == my_team.get("id")), None)
    n = len(power_rows)
    tier = "Unrated"
    if my_rank is not None:
        pct = 1.0 if n == 1 else 1.0 - ((my_rank - 1) / (n - 1))
        tier = "Elite" if pct >= .75 else "Contender" if pct >= .50 else "Competitive" if pct >= .25 else "Needs Help"

    # Locked Phase 7 room-percentile context.
    room_pcts = {}
    for pos in positions:
        mine = room_score(my_team, pos)
        peer = [room_score(t, pos) for t in (teams or [])]
        peer = [x for x in peer if x is not None]
        room_pcts[pos] = (100.0 * sum(1 for x in peer if x <= mine) / len(peer)) if mine is not None and peer else None
    rated = [(p, v) for p, v in room_pcts.items() if v is not None]
    need_pos, need_pct = min(rated, key=lambda x: (x[1], x[0])) if rated else (None, None)

    # Locked Start/Sit context.
    best_swap = max(swaps, key=lambda x: (float(x["Gain"]), x["Start"])) if swaps else None
    total_gain = sum(float(x["Gain"]) for x in swaps)

    # Locked inferred-unrostered waiver context, restricted to the weakest room.
    next_action = None
    current_nfl_roster_keys = None
    if need_pos:
        rostered = {r["name_key"] for t in (teams or []) for r in _wfs_int_roster_rows(t) if r.get("name_key")}
        own = [r for r in my_rows if str(r.get("position") or "").upper() == need_pos
               and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        weakest = min(own, key=lambda r: (float(r["wfs_projection"]), r["player"])) if own else None
        candidates = []
        current_nfl_roster_keys = _wfs_ui_current_nfl_roster_name_keys(season)
        if current_nfl_roster_keys is not None:
            for key, item in lookup.items():
                if key not in current_nfl_roster_keys:
                    continue
                if key in rostered or str(item.get("position") or "").upper() != need_pos or item.get("projection") is None:
                    continue
                if str(item.get("production_status") or "") not in {"MODEL_READY", "COLD_START ESTIMATE"}:
                    continue
                gain = float(item["projection"]) - (float(weakest["wfs_projection"]) if weakest else 0.0)
                if gain > 0:
                    candidates.append((gain, str(item.get("player") or key)))
        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            next_action = f"Check {candidates[0][1]} availability"

    # Locked Phase 5 matchup context.
    matchup_text = "No covered matchup available"
    if opponent_team:
        opp_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(opponent_team), lookup)
        mine = [r for r in my_rows if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS and r.get("wfs_projection") is not None]
        theirs = [r for r in opp_rows if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS and r.get("wfs_projection") is not None]
        if mine or theirs:
            margin = sum(float(r["wfs_projection"]) for r in mine) - sum(float(r["wfs_projection"]) for r in theirs)
            matchup_text = f"{margin:+.2f} edge → {margin + total_gain:+.2f} after WFS moves"

    league_text = f"#{my_rank} of {n} • {tier}" if my_rank else tier
    best_text = (f'{best_swap["Start"]} over {best_swap["Sit"]} • +{float(best_swap["Gain"]):.2f}'
                 if best_swap else "No positive covered Start/Sit swap")
    need_text = f"{need_pos} • {need_pct:.0f}th percentile" if need_pos and need_pct is not None else "No rated room"
    next_text = next_action or (f"Review {need_pos} waiver/trade options" if need_pos else "Maintain current roster core")
    if need_pos and current_nfl_roster_keys is None:
        next_text = "Pickup recommendations unavailable pending current-week NFL roster authority"

    import html as _html
    items = [
        ("🏆", "LEAGUE", league_text),
        ("⚔️", "MATCHUP", matchup_text),
        ("🔥", "BEST MOVE", best_text),
        ("🚨", "BIGGEST NEED", need_text),
        ("🎯", "NEXT ACTION", next_text),
    ]
    rows = "".join(
        f"<div class='wfs-exec-row'><span class='wfs-exec-icon'>{icon}</span><span class='wfs-exec-label'>{label}</span><span class='wfs-exec-value'>{_html.escape(value)}</span></div>"
        for icon, label, value in items
    )
    st.markdown("### ⚡ WFS Weekly Executive Summary")
    st.markdown(
        "<div style='color:#475569;font-size:.92rem;font-weight:600;margin:.1rem 0 .65rem;'>"
        "A quick summary of the most important fantasy information for your team this week.</div>"
        "<style>.wfs-exec{background:linear-gradient(135deg,#0f172a,#172554);border:1px solid #334155;border-radius:16px;padding:.45rem .75rem;margin-bottom:.85rem;color:#f8fafc}.wfs-exec-row{display:grid;grid-template-columns:1.7rem 7.1rem 1fr;gap:.35rem;align-items:start;padding:.55rem 0;border-bottom:1px solid rgba(148,163,184,.22)}.wfs-exec-row:last-child{border-bottom:0}.wfs-exec-label{font-size:.76rem;font-weight:900;letter-spacing:.04em;color:#93c5fd}.wfs-exec-value{font-size:.92rem;font-weight:750;color:#f8fafc;overflow-wrap:anywhere}@media(max-width:600px){.wfs-exec-row{grid-template-columns:1.5rem 6.3rem 1fr}.wfs-exec-label{font-size:.7rem}.wfs-exec-value{font-size:.84rem}}</style>"
        f"<div class='wfs-exec'>{rows}</div>", unsafe_allow_html=True)
    st.caption("Waiver availability is inferred from returned ESPN rosters and must be confirmed in ESPN. WFS does not submit lineup, waiver, or trade transactions.")




def _wfs_render_weekly_readiness(my_team, season, league_id):
    """Phase 14: compact readiness state assembled only from locked WFS signals."""
    if not my_team:
        return

    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        return

    roster = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    supported = [
        r for r in roster
        if r.get("slot_id") in WFS_SUPPORTED_START_SIT_SLOTS
        or r.get("slot_id") == WFS_BENCH_SLOT_ID
    ]
    matched = [r for r in supported if r.get("wfs_projection") is not None]
    swaps = _wfs_int_best_swaps(roster)
    cold = [r for r in roster if r.get("wfs_source_status") == "COLD_START ESTIMATE"]
    missing = [r for r in supported if r.get("wfs_projection") is None]

    if missing or projection_error:
        state = "DATA WARNING"
        icon = "🔴"
        state_color = "#fecaca"
        border_color = "#7f1d1d"
        detail = (
            f"{len(missing)} supported roster projection gap"
            f"{'s' if len(missing) != 1 else ''} require attention before WFS can call this roster ready."
        )
    elif swaps:
        state = "NEEDS ATTENTION"
        icon = "🟠"
        state_color = "#fed7aa"
        border_color = "#9a3412"
        detail = (
            f"{len(swaps)} lineup action{'s' if len(swaps) != 1 else ''} remain pending. "
            "Review your WFS recommendations before kickoff."
        )
    else:
        state = "READY"
        icon = "🟢"
        state_color = "#bbf7d0"
        border_color = "#166534"
        detail = "No covered Start/Sit improvements are currently flagged and projection coverage is healthy."

    coverage = f"{len(matched)}/{len(supported)}" if supported else "0/0"
    watch_count = len(cold)
    action_count = len(swaps)

    st.markdown("### 📅 WFS Weekly Readiness Check")
    st.markdown(
        "<div style='color:#475569;font-size:.92rem;font-weight:600;margin:.1rem 0 .65rem;'>"
        "A quick check of your lineup recommendations, player coverage and watch items for the week.</div>",
        unsafe_allow_html=True,
    )

    import html as _html
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid {border_color};border-radius:16px;padding:.9rem 1rem;color:#f8fafc;margin-bottom:.7rem;'>"
        f"<div style='display:flex;align-items:center;gap:.55rem;margin-bottom:.45rem;'>"
        f"<span style='font-size:1.25rem'>{icon}</span>"
        f"<span style='font-size:.72rem;font-weight:900;letter-spacing:.09em;color:#93c5fd'>WEEKLY STATUS</span>"
        f"</div>"
        f"<div style='font-size:1.35rem;font-weight:950;color:{state_color};margin-bottom:.35rem'>{_html.escape(state)}</div>"
        f"<div style='font-size:.88rem;line-height:1.45;color:#e2e8f0'>{_html.escape(detail)}</div>"
        f"<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.45rem;margin-top:.8rem'>"
        f"<div style='background:#111827;border:1px solid #334155;border-radius:10px;padding:.55rem'><div style='font-size:.66rem;font-weight:850;color:#93c5fd'>ACTIONS</div><div style='font-size:1.05rem;font-weight:900'>{action_count}</div></div>"
        f"<div style='background:#111827;border:1px solid #334155;border-radius:10px;padding:.55rem'><div style='font-size:.66rem;font-weight:850;color:#93c5fd'>COVERAGE</div><div style='font-size:1.05rem;font-weight:900'>{coverage}</div></div>"
        f"<div style='background:#111827;border:1px solid #334155;border-radius:10px;padding:.55rem'><div style='font-size:.66rem;font-weight:850;color:#93c5fd'>WATCH</div><div style='font-size:1.05rem;font-weight:900'>{watch_count}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if missing:
        names = ", ".join(str(r.get("player") or "") for r in missing[:4])
        st.caption(f"Projection gaps: {names}. WFS does not invent unsupported projections.")
    elif cold:
        st.caption(
            f"{len(cold)} rostered player{'s' if len(cold) != 1 else ''} currently use an early projection estimate and remain on watch."
        )
    else:
        st.caption("Recheck your weekly status when ESPN rosters, injuries or player projections change.")

def _wfs_render_alerts_watchlist(teams, my_team, season, league_id):
    """Phase 13: read-only alerts/watchlist built only from locked WFS intelligence."""
    if not my_team:
        return
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        return

    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    swaps = _wfs_int_best_swaps(my_rows)
    alerts = []

    # Immediate lineup actions: exact output of the locked Start/Sit engine.
    for swap in swaps:
        alerts.append({
            "level": "ACTION",
            "icon": "🔥",
            "title": f'{swap["Start"]} over {swap["Sit"]}',
            "detail": f'+{float(swap["Gain"]):.2f} WFS projected points at {swap["Slot"]}.',
        })

    # Watch any rostered player whose projection is explicitly a validated cold-start estimate.
    cold = [r for r in my_rows if r.get("wfs_source_status") == "COLD_START ESTIMATE"]
    cold.sort(key=lambda r: (str(r.get("position") or r.get("wfs_position") or ""), r.get("player", "")))
    if cold:
        names = ", ".join(str(r.get("player") or "") for r in cold[:4])
        if len(cold) > 4:
            names += f" +{len(cold)-4} more"
        alerts.append({
            "level": "WATCH",
            "icon": "👀",
            "title": f'{len(cold)} early projection estimate{"s" if len(cold) != 1 else ""} on roster',
            "detail": f'{names}. These players have limited recent projection history and remain on watch.',
        })

    # Data-quality alert only for supported QB/RB/WR/TE roster players without a usable WFS projection.
    supported = [r for r in my_rows if str(r.get("position") or r.get("wfs_position") or "").upper() in {"QB","RB","WR","TE"}]
    missing = [r for r in supported if r.get("wfs_projection") is None]
    if missing:
        names = ", ".join(str(r.get("player") or "") for r in missing[:4])
        if len(missing) > 4:
            names += f" +{len(missing)-4} more"
        alerts.append({
            "level": "DATA",
            "icon": "⚠️",
            "title": f'{len(missing)} roster projection gap{"s" if len(missing) != 1 else ""}',
            "detail": f'{names}. WFS will not invent a projection for unsupported coverage.',
        })

    # Locked Phase 7 weakest-room context + locked inferred-unrostered method.
    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    def room_score(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        vals = [float(r["wfs_projection"]) for r in rows
                if str(r.get("position") or r.get("wfs_position") or "").upper() == pos
                and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        vals.sort(reverse=True)
        take = vals[:room_size[pos]]
        return (sum(take) / len(take)) if take else None

    room_pcts = {}
    for pos in positions:
        mine = room_score(my_team, pos)
        peers = [room_score(t, pos) for t in (teams or [])]
        peers = [v for v in peers if v is not None]
        room_pcts[pos] = (100.0 * sum(1 for v in peers if v <= mine) / len(peers)) if mine is not None and peers else None
    rated = [(p, pct) for p, pct in room_pcts.items() if pct is not None]
    need_pos, need_pct = min(rated, key=lambda x: (x[1], x[0])) if rated else (None, None)

    current_nfl_roster_keys = None
    if need_pos:
        rostered = {r["name_key"] for t in (teams or []) for r in _wfs_int_roster_rows(t) if r.get("name_key")}
        own = [r for r in my_rows if str(r.get("position") or r.get("wfs_position") or "").upper() == need_pos
               and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        weakest = min(own, key=lambda r: (float(r["wfs_projection"]), r["player"])) if own else None
        candidates = []
        current_nfl_roster_keys = _wfs_ui_current_nfl_roster_name_keys(season)
        if current_nfl_roster_keys is not None:
            for key, item in lookup.items():
                if key not in current_nfl_roster_keys:
                    continue
                if key in rostered or str(item.get("position") or "").upper() != need_pos or item.get("projection") is None:
                    continue
                if str(item.get("production_status") or "") not in {"MODEL_READY", "COLD_START ESTIMATE"}:
                    continue
                gain = float(item["projection"]) - (float(weakest["wfs_projection"]) if weakest else 0.0)
                if gain > 0:
                    candidates.append((gain, str(item.get("player") or key), str(item.get("production_status") or "")))
        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            gain, player, status = candidates[0]
            alerts.append({
                "level": "WATCH",
                "icon": "🎯",
                "title": f'Check {player} availability',
                "detail": f'{need_pos} is the weakest room ({need_pct:.0f}th percentile); this player projects +{gain:.2f} vs the lowest-covered {need_pos} on your roster. Confirm availability in ESPN.',
            })

    st.markdown("### 🔔 WFS Alerts & Watchlist")
    st.markdown(
        "<div style='color:#475569;font-size:.92rem;font-weight:600;margin:.1rem 0 .65rem;'>"
        "Players and lineup situations that currently deserve your attention.</div>",
        unsafe_allow_html=True,
    )
    if projection_error:
        st.warning(projection_error)

    if not alerts:
        if need_pos and current_nfl_roster_keys is None:
            st.info("No independent lineup alerts. Pickup watchlist checks remain unavailable.")
        else:
            st.success("No current WFS alerts. Your covered lineup and roster watchlist have no flagged actions.")
        return

    import html as _html
    order = {"ACTION": 0, "WATCH": 1, "DATA": 2}
    alerts.sort(key=lambda a: (order.get(a["level"], 9), a["title"]))
    cards = []
    for a in alerts:
        cards.append(
            "<div class='wfs-alert-card'>"
            f"<div class='wfs-alert-icon'>{a['icon']}</div>"
            "<div>"
            f"<div class='wfs-alert-level'>{_html.escape(a['level'])}</div>"
            f"<div class='wfs-alert-title'>{_html.escape(a['title'])}</div>"
            f"<div class='wfs-alert-detail'>{_html.escape(a['detail'])}</div>"
            "</div></div>"
        )
    st.markdown(
        "<style>"
        ".wfs-alert-stack{display:grid;gap:.55rem;margin:.15rem 0 .75rem}.wfs-alert-card{display:grid;grid-template-columns:2rem 1fr;gap:.55rem;background:#0e1117;border:1px solid #334155;border-radius:13px;padding:.72rem .78rem;color:#f8fafc}.wfs-alert-icon{font-size:1.25rem;line-height:1.35}.wfs-alert-level{font-size:.68rem;font-weight:900;letter-spacing:.08em;color:#93c5fd}.wfs-alert-title{font-size:.94rem;font-weight:850;color:#f8fafc;margin:.08rem 0}.wfs-alert-detail{font-size:.82rem;line-height:1.45;color:#cbd5e1;overflow-wrap:anywhere}"
        "@media(max-width:600px){.wfs-alert-card{grid-template-columns:1.7rem 1fr;padding:.65rem}.wfs-alert-title{font-size:.88rem}.wfs-alert-detail{font-size:.78rem}}"
        "</style><div class='wfs-alert-stack'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Confirm player availability in ESPN before making waiver or roster decisions.")


def _wfs_render_team_improvement_plan(teams, my_team, season, league_id):
    """Phase 11: read-only orchestration of already-validated WFS intelligence."""
    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup or not my_team:
        return

    positions = ("QB", "RB", "WR", "TE")
    room_size = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    swaps = _wfs_int_best_swaps(my_rows)

    def room_score(team, pos):
        rows = _wfs_int_attach_projection(_wfs_int_roster_rows(team), lookup)
        vals = [float(r["wfs_projection"]) for r in rows
                if str(r.get("position") or r.get("wfs_position") or "").upper() == pos
                and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        vals.sort(reverse=True)
        take = vals[:room_size[pos]]
        return (sum(take) / len(take)) if take else None

    peers = {p: [] for p in positions}
    for team in teams or []:
        for pos in positions:
            val = room_score(team, pos)
            if val is not None:
                peers[pos].append(val)
    room_detail = {}
    for pos in positions:
        val = room_score(my_team, pos)
        pool = peers[pos]
        pct = (100.0 * sum(1 for x in pool if x <= val) / len(pool)) if val is not None and pool else None
        room_detail[pos] = (val, pct)
    rated = [(p, d[1]) for p, d in room_detail.items() if d[1] is not None]
    need_pos = min(rated, key=lambda x: (x[1], x[0]))[0] if rated else None

    # Same inferred-unrostered method as the locked Waiver Wire module.
    rostered = {r["name_key"] for t in (teams or []) for r in _wfs_int_roster_rows(t) if r.get("name_key")}
    waiver = []
    if need_pos:
        own = [r for r in my_rows if str(r.get("position") or "").upper() == need_pos
               and r.get("wfs_projection") is not None and r.get("slot_id") != WFS_IR_SLOT_ID]
        weakest = min(own, key=lambda r: (float(r["wfs_projection"]), r["player"])) if own else None
        current_nfl_roster_keys = _wfs_ui_current_nfl_roster_name_keys(season)
        if current_nfl_roster_keys is not None:
            for key, item in lookup.items():
                if key not in current_nfl_roster_keys:
                    continue
                if key in rostered or str(item.get("position") or "").upper() != need_pos or item.get("projection") is None:
                    continue
                status = str(item.get("production_status") or "")
                if status not in {"MODEL_READY", "COLD_START ESTIMATE"}:
                    continue
                gain = float(item["projection"]) - (float(weakest["wfs_projection"]) if weakest else 0.0)
                waiver.append((gain, str(item.get("player") or key), float(item["projection"]), status, weakest))
        waiver.sort(key=lambda x: (-x[0], x[1]))

    # Recreate locked Phase 9 power ordering for context only.
    power_rows = []
    for team in teams or []:
        vals = [room_score(team, p) for p in positions]
        if all(v is not None for v in vals):
            power_rows.append((sum(vals), team.get("id"), _espn_team_name(team)))
    power_rows.sort(key=lambda x: (-x[0], x[2].lower()))
    my_id = my_team.get("id")
    my_rank = next((i for i, r in enumerate(power_rows, 1) if r[1] == my_id), None)
    n = len(power_rows)
    if my_rank is None:
        tier = "Unrated"
    else:
        pct = 1.0 if n == 1 else 1.0 - ((my_rank - 1) / (n - 1))
        tier = "Elite" if pct >= .75 else "Contender" if pct >= .50 else "Competitive" if pct >= .25 else "Needs Help"

    actions = []
    for sw in swaps:
        actions.append((float(sw["Gain"]), "START/SIT",
                        f'Start {sw["Start"]} over {sw["Sit"]}',
                        f'+{float(sw["Gain"]):.2f} WFS projected points at {sw["Slot"]}.'))
    if waiver and waiver[0][0] > 0:
        gain, add, proj, status, drop = waiver[0]
        drop_name = drop["player"] if drop else "your lowest-covered player"
        actions.append((float(gain), "WAIVER CHECK",
                        f'Investigate {add} at {need_pos}',
                        f'Inferred unrostered candidate; +{gain:.2f} vs {drop_name}. Confirm ESPN availability before acting.'))
    if need_pos:
        pct = room_detail[need_pos][1]
        actions.append((0.0, "ROSTER BUILD",
                        f'Prioritize the {need_pos} room',
                        f'Weakest league-relative room ({pct:.0f}th percentile). Use waiver/trade intelligence to improve depth without weakening stronger rooms.'))

    priority = {"START/SIT": 0, "WAIVER CHECK": 1, "ROSTER BUILD": 2}
    actions.sort(key=lambda x: (priority.get(x[1], 9), -x[0], x[2]))

    st.markdown("### 📈 WFS Team Improvement Plan")
    st.markdown(
        "<div style='color:#475569;font-size:.95rem;font-weight:600;margin:.15rem 0 .75rem 0;'>"
        "A prioritized fantasy plan using your Start/Sit opportunities, roster strengths, waiver options "
        "and league context.</div>", unsafe_allow_html=True)
    if projection_error:
        st.warning(projection_error)
    if my_rank:
        st.success(f"Current foundation: **#{my_rank} of {n} • {tier}**. Improve weak points without unnecessarily breaking your strongest rooms.")

    if actions:
        import html as _html
        rows=[]
        for i, (_, kind, move, why) in enumerate(actions, 1):
            rows.append(f"<tr><td>{i}</td><td>{_html.escape(kind)}</td><td><b>{_html.escape(move)}</b><br><span style='color:#cbd5e1'>{_html.escape(why)}</span></td></tr>")
        st.markdown("""
        <style>.wfs-plan{width:100%;border-collapse:collapse;table-layout:fixed;background:#0e1117;color:#f8fafc;border-radius:12px;overflow:hidden}.wfs-plan th,.wfs-plan td{border:1px solid #29313d;padding:.62rem .48rem;vertical-align:top}.wfs-plan th{background:#171c24;color:#cbd5e1;text-align:left}.wfs-plan th:nth-child(1),.wfs-plan td:nth-child(1){width:10%;text-align:center}.wfs-plan th:nth-child(2),.wfs-plan td:nth-child(2){width:24%}.wfs-plan th:nth-child(3),.wfs-plan td:nth-child(3){width:66%;overflow-wrap:anywhere}@media(max-width:600px){.wfs-plan th,.wfs-plan td{font-size:.78rem;padding:.5rem .3rem}}</style>
        <table class='wfs-plan'><thead><tr><th>#</th><th>Action</th><th>Recommendation</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)
    else:
        st.info("No positive Start/Sit or inferred waiver upgrade was found from the current projection-covered data.")

    with st.expander("How WFS builds this plan", expanded=False):
        st.markdown(
            "**Start/Sit** highlights projected lineup improvements using eligible ESPN roster slots.  \n"
            "**Waiver checks** identify players who appear available and should be confirmed in ESPN.  \n"
            "**Roster build** identifies your weakest league-relative QB/RB/WR/TE position group.  \n"
            "**Trade Intelligence** helps identify possible roster fits and opportunities."
        )

def _wfs_render_my_team_command_center(my_team, season, league_id):
    """
    Read-only WFS intelligence layer.
    Uses broad frozen WFS production projections only; does not change ESPN or optimizer data.
    """
    roster = _wfs_int_roster_rows(my_team)
    if not roster:
        st.info("My Team roster is not available from ESPN yet.")
        return

    lookup, projection_error = _wfs_int_projection_lookup()
    roster = _wfs_int_attach_projection(roster, lookup)

    supported = [
        r for r in roster
        if r["slot_id"] in WFS_SUPPORTED_START_SIT_SLOTS
        or r["slot_id"] == WFS_BENCH_SLOT_ID
    ]
    matched = [r for r in supported if r.get("wfs_projection") is not None]
    starters = [r for r in roster if r["slot_id"] in WFS_SUPPORTED_START_SIT_SLOTS]
    covered_starters = [r for r in starters if r.get("wfs_projection") is not None]
    recommendations = _wfs_int_best_swaps(roster)

    current_projection = sum(float(r["wfs_projection"]) for r in covered_starters)
    total_gain = sum(float(r["Gain"]) for r in recommendations)
    optimized_projection = current_projection + total_gain
    coverage_pct = (100.0 * len(matched) / len(supported)) if supported else 0.0

    st.markdown(
        """
        <div style="
            background:linear-gradient(110deg,#f7fbff 0%,#eef6ff 100%);
            border:1px solid #cfe0f4;border-radius:15px;
            padding:.9rem 1rem;margin:.9rem 0 .65rem;">
          <div style="font-size:.72rem;color:#0b5eb8;font-weight:900;letter-spacing:.08em;">
            WFS FANTASY INTELLIGENCE
          </div>
          <div style="font-size:1.15rem;color:#17365d;font-weight:900;margin-top:.15rem;">
            My Team Command Center
          </div>
          <div style="font-size:.78rem;color:#6b7f99;margin-top:.2rem;">
            Start/Sit analysis using your ESPN roster and current WFS player projections.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projection Coverage", f"{len(matched)}/{len(supported)}", f"{coverage_pct:.0f}%")
    c2.metric("Covered Starter Proj", f"{current_projection:.2f}")
    c3.metric("Recommended Swaps", len(recommendations))
    c4.metric("Projected Gain", f"+{total_gain:.2f}" if total_gain > 0 else "0.00")

    if projection_error:
        st.warning(projection_error)

    if recommendations:
        st.success(
            f"WFS found {len(recommendations)} projected lineup improvement"
            f"{'s' if len(recommendations) != 1 else ''} worth "
            f"{total_gain:.2f} projected points across covered positions."
        )
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(
            rec_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Sit Proj": st.column_config.NumberColumn(format="%.2f"),
                "Start Proj": st.column_config.NumberColumn(format="%.2f"),
                "Gain": st.column_config.NumberColumn(format="+%.2f"),
            },
        )
    else:
        if covered_starters:
            st.info(
                "No positive bench-over-starter swap was found among players with "
                "current WFS projection coverage."
            )
        else:
            st.info(
                "WFS does not yet have enough player projection information on this ESPN roster "
                "to make a Start/Sit recommendation."
            )

    with st.expander("View WFS starter & bench board", expanded=False):
        board_rows = []
        recommended_sits = {r["Sit"] for r in recommendations}
        recommended_starts = {r["Start"] for r in recommendations}

        for r in roster:
            if r["slot_id"] == WFS_IR_SLOT_ID:
                group = "IR"
            elif r["slot_id"] == WFS_BENCH_SLOT_ID:
                group = "Bench"
            else:
                group = "Starter"

            if r["player"] in recommended_sits:
                signal = "SIT →"
            elif r["player"] in recommended_starts:
                signal = "START ↑"
            elif group == "Starter":
                signal = "START"
            elif group == "Bench":
                signal = "BENCH"
            else:
                signal = "IR"

            board_rows.append({
                "Status": signal,
                "Slot": r["slot"],
                "Player": r["player"],
                "Pos": r["position"] or r["wfs_position"],
                "Injury": r["injury"] or "—",
                "WFS Proj": (
                    round(float(r["wfs_projection"]), 2)
                    if r.get("wfs_projection") is not None else None
                ),
            })

        board = pd.DataFrame(board_rows)
        if not board.empty:
            order = {"Starter": 0, "Bench": 1, "IR": 2}
            board["__group"] = board["Slot"].map(
                lambda s: 1 if s == "Bench" else (2 if s == "IR" else 0)
            )
            board = board.sort_values(
                ["__group", "Slot", "WFS Proj", "Player"],
                ascending=[True, True, False, True],
                kind="stable",
                na_position="last",
            ).drop(columns="__group")
            st.dataframe(
                board.reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "WFS Proj": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        st.caption(
            "Start/Sit v1 analyzes QB/RB/WR/TE/FLEX only. Kicker is intentionally "
            "outside the frozen FanDuel projection model. Unmatched players remain "
            "unscored rather than being fuzzy-matched or assigned invented projections."
        )

def _wfs_render_waiver_wire_intelligence(teams, my_team, season, league_id):
    """Rank exact-name WFS projected players not rostered by any ESPN team in this league."""
    st.markdown("#### 🔎 Waiver Wire Intelligence")
    st.caption(
        "Find projected roster upgrades among players who appear to be available in your ESPN league."
    )

    lookup, projection_error = _wfs_int_projection_lookup()
    if not lookup:
        st.warning(projection_error or "WFS projection source is unavailable.")
        return

    rostered_keys = set()
    for team in teams or []:
        for row in _wfs_int_roster_rows(team):
            if row.get("name_key"):
                rostered_keys.add(row["name_key"])

    positions = {"QB", "RB", "WR", "TE"}
    candidates = []
    current_nfl_roster_keys = _wfs_ui_current_nfl_roster_name_keys(season)
    if current_nfl_roster_keys is None:
        return
    for key, item in lookup.items():
        if key not in current_nfl_roster_keys:
            continue
        if key in rostered_keys:
            continue
        pos = str(item.get("position") or "").upper()
        proj = item.get("projection")
        status = str(item.get("production_status") or "")
        if pos not in positions or proj is None:
            continue
        if status not in {"MODEL_READY", "COLD_START ESTIMATE"}:
            continue
        candidates.append({
            "Player": item.get("player") or key,
            "Pos": pos,
            "NFL Team": item.get("team") or "",
            "WFS Proj": float(proj),
            "_WFS Status": status,
        })

    if not candidates:
        st.info("No projected unrostered QB/RB/WR/TE candidates were found from the current WFS source.")
        return

    candidates.sort(key=lambda r: (-r["WFS Proj"], r["Pos"], r["Player"]))

    my_rows = _wfs_int_attach_projection(_wfs_int_roster_rows(my_team), lookup)
    my_by_pos = {}
    for row in my_rows:
        pos = str(row.get("position") or "").upper()
        proj = row.get("wfs_projection")
        if pos in positions and proj is not None and row.get("slot_id") != WFS_IR_SLOT_ID:
            my_by_pos.setdefault(pos, []).append(row)

    best_by_pos = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        pool = [r for r in candidates if r["Pos"] == pos]
        if pool:
            best_by_pos[pos] = pool[0]

    card_cols = st.columns(4)
    for idx, pos in enumerate(["QB", "RB", "WR", "TE"]):
        best = best_by_pos.get(pos)
        with card_cols[idx]:
            if best:
                st.metric(pos, best["Player"], f'{best["WFS Proj"]:.0f} WFS')
            else:
                st.metric(pos, "No candidate")

    upgrades = []
    for pos, add in best_by_pos.items():
        own = my_by_pos.get(pos, [])
        if not own:
            continue
        drop = min(own, key=lambda r: (float(r["wfs_projection"]), r["player"]))
        gain = float(add["WFS Proj"]) - float(drop["wfs_projection"])
        if gain > 0:
            upgrades.append({
                "Pos": pos,
                "Possible Drop": drop["player"],
                "Drop Proj": float(drop["wfs_projection"]),
                "Add": add["Player"],
                "Add Proj": float(add["WFS Proj"]),
                "Gain": gain,
                "_Status": add["_WFS Status"],
            })

    upgrades.sort(key=lambda r: (-r["Gain"], r["Pos"], r["Add"]))
    if upgrades:
        best = upgrades[0]
        st.success(
            f'Best projected roster upgrade: add {best["Add"]} over {best["Possible Drop"]} '
            f'for +{best["Gain"]:.0f} WFS projected points at {best["Pos"]}.'
        )
        st.dataframe(
            pd.DataFrame(upgrades).drop(columns=["_Status"], errors="ignore"),
            width="stretch",
            hide_index=True,
            column_config={
                "Drop Proj": st.column_config.NumberColumn(format="%.0f"),
                "Add Proj": st.column_config.NumberColumn(format="%.0f"),
                "Gain": st.column_config.NumberColumn(format="+%.0f"),
            },
        )
    else:
        st.info(
            "WFS did not find a positive same-position projection upgrade over your currently "
            "rostered, projection-covered QB/RB/WR/TE players."
        )

    with st.expander("View top projected unrostered candidates", expanded=False):
        board = []
        for pos in ["QB", "RB", "WR", "TE"]:
            board.extend([r for r in candidates if r["Pos"] == pos][:10])
        board.sort(key=lambda r: (r["Pos"], -r["WFS Proj"], r["Player"]))
        st.dataframe(
            pd.DataFrame(board).drop(columns=["_WFS Status"], errors="ignore"),
            width="stretch",
            hide_index=True,
            column_config={"WFS Proj": st.column_config.NumberColumn(format="%.0f")},
        )
        st.caption(
            "Player availability is based on the connected ESPN league. Confirm availability in ESPN before making a roster move."
        )

    if projection_error:
        st.caption(projection_error)


def _wfs_espn_weekly_leader_rows(teams, week):
    """Return actual ESPN fantasy points for rostered players for one scoring period."""
    rows = []
    seen = set()
    try:
        target_week = int(week)
    except Exception:
        target_week = 1

    for team in teams or []:
        fantasy_team = _espn_team_name(team)
        entries = ((team.get("roster") or {}).get("entries") or [])
        for entry in entries:
            pool = entry.get("playerPoolEntry") or {}
            player = pool.get("player") or {}
            player_id = player.get("id")
            if player_id is None or player_id in seen:
                continue

            position = _wfs_int_position_name(player)
            if position not in {"QB", "RB", "WR", "TE", "K", "D/ST"}:
                continue

            points = None
            for stat in player.get("stats") or []:
                try:
                    scoring_period = int(stat.get("scoringPeriodId"))
                except Exception:
                    continue
                if scoring_period != target_week:
                    continue
                try:
                    source_id = int(stat.get("statSourceId", 0))
                except Exception:
                    source_id = 0
                if source_id != 0:
                    continue
                value = stat.get("appliedTotal")
                if value is None:
                    continue
                try:
                    points = float(value)
                    break
                except Exception:
                    continue

            if points is None:
                continue

            seen.add(player_id)
            rows.append({
                "Player": str(player.get("fullName") or f"Player {player_id}").strip(),
                "Pos": position,
                "Fantasy Pts": points,
                "Fantasy Team": fantasy_team,
            })
    return rows


def _wfs_render_weekly_fantasy_leaders(teams, current_week, season, league_id):
    """League-scoring-aware weekly leaders from ESPN's applied fantasy totals."""
    st.markdown("#### 🏆 Weekly Fantasy Leaders")
    st.caption(
        "Actual ESPN fantasy points using this league's applied scoring. "
        "Only players currently returned on league rosters are included."
    )

    max_week = max(1, int(current_week or 1))
    selected_week = st.selectbox(
        "Week",
        list(range(1, max_week + 1)),
        index=max_week - 1,
        key=f"wfs_weekly_leaders_week_{season}_{league_id}",
    )
    rows = _wfs_espn_weekly_leader_rows(teams, selected_week)
    if not rows:
        st.info(
            f"Week {selected_week} fantasy leaders are not available yet. "
            "WFS will populate this automatically when ESPN returns actual scoring totals."
        )
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Fantasy Pts", "Player"], ascending=[False, True], kind="stable")

    categories = [("Top Overall", None), ("QB", "QB"), ("RB", "RB"), ("WR", "WR"),
                  ("TE", "TE"), ("K", "K"), ("D/ST", "D/ST")]
    leader_rows = []
    for label, pos in categories:
        pool = df if pos is None else df[df["Pos"].eq(pos)]
        if pool.empty:
            continue
        top = pool.iloc[0]
        leader_rows.append({
            "Leader": label,
            "Player": top["Player"],
            "Pos": top["Pos"],
            "Fantasy Pts": float(top["Fantasy Pts"]),
            "Fantasy Team": top["Fantasy Team"],
        })

    if leader_rows:
        st.dataframe(
            pd.DataFrame(leader_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Fantasy Pts": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with st.expander(f"View Week {selected_week} leaderboard", expanded=False):
        st.dataframe(
            df.reset_index(drop=True),
            width="stretch",
            hide_index=True,
            column_config={
                "Fantasy Pts": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def _wfs_extract_espn_league_id(raw: str) -> str:
    """Accept either a bare ESPN League ID or a pasted ESPN league URL."""
    raw = (raw or "").strip()
    if raw.isdigit():
        return raw
    match = re.search(r"[?&]leagueId=(\d+)", raw)
    if match:
        return match.group(1)
    match = re.search(r"(\d{5,})", raw)
    return match.group(1) if match else ""


def render_espn_league_hub():
    user = _wfs_current_user()
    if not user:
        st.error("A signed-in WFS account is required to use ESPN Fantasy.")
        return

    _wfs_accounts_init()
    _wfs_upsert_user(user)

    if "wfs_espn_season" not in st.session_state:
        st.session_state["wfs_espn_season"] = 2026

    try:
        swid, espn_s2 = _wfs_espn_credentials_for_user(user["sub"])
    except Exception as exc:
        st.error(
            "Your saved ESPN connection could not be loaded. "
            "Please reconnect ESPN and try again."
        )
        swid, espn_s2 = "", ""

    auth_ready = bool(swid and espn_s2)

    with st.container(border=True):
        st.markdown("### ESPN Fantasy Football")
        c1, c2 = st.columns([1.25, .75])
        seasons = [2026, 2025, 2024, 2023]
        season = c1.selectbox(
            "Season",
            seasons,
            index=seasons.index(st.session_state["wfs_espn_season"])
            if st.session_state["wfs_espn_season"] in seasons else 0,
            key="wfs_espn_season_selector_multiuser",
        )
        st.session_state["wfs_espn_season"] = season

        if auth_ready:
            c2.success("🔐 ESPN account connected")
        else:
            c2.info("🌐 Public leagues ready")

        st.caption(
            "Start with your ESPN League ID below. Public leagues connect immediately. "
            "If ESPN says the league is private, WFS will guide you through the secure connection option."
        )

        if auth_ready:
            with st.expander("🔐 Manage private ESPN connection", expanded=False):
                st.success("Your private ESPN session is stored encrypted for this WFS account.")
                if st.button(
                    "Disconnect Private ESPN",
                    width="stretch",
                    key="wfs_disconnect_private_espn",
                ):
                    _wfs_delete_espn_credentials(user["sub"])
                    st.success(
                        "Private ESPN session removed. Your saved league/team follows remain in WFS."
                    )
                    st.rerun()
        else:
            with st.expander("Advanced / Desktop Setup for Private ESPN", expanded=False):
                st.info(
                    "Only use this if WFS tells you your league is private. "
                    "A desktop browser is recommended."
                )

                st.markdown(
                    """
                    #### How to find your ESPN connection information

                    **1. Find your ESPN League ID**

                    Open your fantasy league on ESPN in a web browser.

                    Look at the address bar. The URL normally contains a value similar to:

                    `leagueId=123456789`

                    Your **League ID** is the number only:

                    `123456789`

                    You do not need SWID or espn_s2 for a public league.

                    **2. For a private league, sign in to ESPN first**

                    Make sure you are signed in to the ESPN account that belongs to the fantasy league.

                    **3. Open your browser's developer tools**

                    **Chrome / Edge**

                    `F12 → Application → Storage → Cookies → https://www.espn.com`

                    **Firefox**

                    `F12 → Storage → Cookies → https://www.espn.com`

                    **4. Find these two cookies**

                    - `SWID`
                    - `espn_s2`

                    **5. Copy the complete cookie values**

                    The SWID usually looks similar to:

                    `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`

                    Keep the `{ }` braces when copying the SWID.

                    Copy the complete `espn_s2` value exactly as shown.

                    **6. Paste them below and save**

                    After saving, retry your League ID.
                    """
                )

                st.warning(
                    "SWID and espn_s2 are sensitive ESPN browser-session credentials. "
                    "Only enter values from your own ESPN account. "
                    "Do not share them with another person. "
                    "WFS stores them encrypted and never asks for your ESPN password."
                )
                swid_input = st.text_input(
                    "ESPN SWID",
                    value="",
                    type="password",
                    placeholder="{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}",
                    key="wfs_user_espn_swid",
                )
                s2_input = st.text_input(
                    "ESPN espn_s2",
                    value="",
                    type="password",
                    placeholder="Paste your ESPN espn_s2 session value",
                    key="wfs_user_espn_s2",
                )
                if st.button(
                    "Save Secure ESPN Session",
                    type="primary",
                    width="stretch",
                    key="wfs_save_private_espn",
                ):
                    try:
                        _wfs_save_espn_credentials(user["sub"], swid_input, s2_input)
                        st.success(
                            "ESPN session saved encrypted. Retry your private League ID below."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(
                            "Your ESPN connection could not be saved. "
                            "Please verify the connection information and try again."
                        )

    st.markdown("### My Fantasy Teams")
    st.caption(
        "Add an ESPN League ID, validate it, then choose the team you want WFS to follow. "
        "Saved leagues are private to your signed-in WFS account."
    )

    with st.container(border=True):
        a1, a2 = st.columns([1.5, .55])
        add_id_raw = a1.text_input(
            "ESPN League ID",
            value="",
            placeholder="Enter ESPN League ID or paste your league link",
            key=f"wfs_espn_add_id_multiuser_{season}",
        ).strip()
        add_clicked = a2.button(
            "Validate + Add",
            type="primary",
            width="stretch",
            key=f"wfs_espn_validate_add_{season}",
        )

        if add_clicked:
            add_id = _wfs_extract_espn_league_id(add_id_raw)
            if not add_id:
                st.error(
                    "Enter your ESPN League ID, or paste your full ESPN "
                    "league link."
                )
            else:
                existing = {
                    str(x["league_id"])
                    for x in _wfs_espn_connections(user["sub"], season)
                }
                if add_id in existing:
                    st.info("That ESPN league is already saved to your WFS account.")
                else:
                    check = espn_league_get(
                        add_id,
                        season,
                        ["mSettings", "mTeam", "mStandings"],
                        swid,
                        espn_s2,
                    )
                    if not check.get("ok"):
                        message = check.get("message") or "ESPN did not validate that league."
                        st.error(message)
                    else:
                        data = check.get("data") or {}
                        settings = data.get("settings") or {}
                        league_name = settings.get("name") or f"ESPN League {add_id}"
                        _wfs_espn_upsert_connection(
                            user["sub"],
                            season,
                            add_id,
                            league_name,
                            check.get("status") or "PUBLIC",
                        )
                        st.success(f"{league_name} added to your WFS account.")
                        st.rerun()

    connections = _wfs_espn_connections(user["sub"], season)
    if not connections:
        st.info(
            f"You have no ESPN leagues saved for {season}. Add a League ID above to start following a team."
        )
        return

    valid = []
    unavailable = []
    for saved in connections:
        result = espn_league_get(
            saved["league_id"],
            season,
            ["mSettings", "mTeam", "mStandings"],
            swid,
            espn_s2,
        )
        if result.get("ok"):
            data = result.get("data") or {}
            settings = data.get("settings") or {}
            league_name = settings.get("name") or saved["league_name"] or f"ESPN League {saved['league_id']}"
            if league_name != saved.get("league_name") or (result.get("status") or "") != saved.get("connection_type"):
                _wfs_espn_upsert_connection(
                    user["sub"], season, saved["league_id"], league_name,
                    result.get("status") or saved.get("connection_type") or "PUBLIC",
                    saved.get("team_id"), saved.get("team_name") or "",
                )
                saved["league_name"] = league_name
                saved["connection_type"] = result.get("status") or saved.get("connection_type")
            valid.append(saved)
        else:
            unavailable.append((saved, result))

    if unavailable:
        with st.expander("Saved leagues currently unavailable"):
            for saved, result in unavailable:
                r1, r2 = st.columns([1.6, .4])
                r1.write(
                    f"**{saved.get('league_name') or saved['league_id']}** — "
                    f"{result.get('message') or 'Not available'}"
                )
                if r2.button(
                    "Remove",
                    key=f"wfs_remove_bad_{season}_{saved['league_id']}",
                    width="stretch",
                ):
                    _wfs_espn_delete_connection(
                        user["sub"], season, saved["league_id"]
                    )
                    st.rerun()

    if not valid:
        st.warning(
            "None of your saved ESPN leagues can currently be loaded. "
            "If they are private, reconnect your ESPN session above."
        )
        return

    valid = sorted(
        valid,
        key=lambda x: ((x.get("league_name") or "").lower(), str(x["league_id"])),
    )
    label_to_id = {
        (
            f"{x.get('league_name') or 'ESPN League'}"
            + (f" • {x.get('team_name')}" if x.get("team_name") else "")
            + f" • ID {x['league_id']}"
        ): str(x["league_id"])
        for x in valid
    }
    chosen = st.selectbox(
        "Choose ESPN League",
        list(label_to_id),
        key=f"espn_choose_multiuser_{season}",
    )
    league_id = label_to_id[chosen]
    connection = next(x for x in valid if str(x["league_id"]) == league_id)

    manage1, manage2 = st.columns([1.6, .4])
    followed = connection.get("team_name") or "Team not selected"
    manage1.caption(f"Following: {followed} • League ID {league_id}")
    if manage2.button(
        "Remove League",
        key=f"remove_active_multiuser_{season}_{league_id}",
        width="stretch",
    ):
        _wfs_espn_delete_connection(user["sub"], season, league_id)
        st.rerun()

    result = espn_league_get(
        league_id,
        season,
        ["mSettings", "mTeam", "mStandings", "mRoster", "mMatchupScore"],
        swid,
        espn_s2,
    )
    if not result.get("ok"):
        st.error(result.get("message") or "Could not load the selected ESPN league.")
        return

    data = result["data"]
    settings = data.get("settings") or {}
    teams = data.get("teams") or []
    schedule = data.get("schedule") or []
    status = data.get("status") or {}
    members = data.get("members") or []

    league_name = settings.get("name") or connection.get("league_name") or f"ESPN League {league_id}"
    current_week = (
        status.get("currentMatchupPeriod")
        or status.get("currentScoringPeriod")
        or 1
    )
    try:
        current_week = int(current_week)
    except Exception:
        current_week = 1

    team_by_id = {
        int(t.get("id")): t
        for t in teams
        if t.get("id") is not None
    }
    member_by_id = {}
    for member in members:
        for key in ("id", "uuid"):
            if member.get(key):
                member_by_id[str(member.get(key))] = member

    standings_rows = []
    for team in teams:
        wins, losses, ties, pf, pa = _espn_record(team)
        standings_rows.append({
            "Team": _espn_team_name(team),
            "Owner": _espn_owner_display(team, member_by_id),
            "W": wins,
            "L": losses,
            "T": ties,
            "PF": round(pf, 2),
            "PA": round(pa, 2),
            "Team ID": team.get("id"),
        })

    standings = pd.DataFrame(standings_rows)
    if not standings.empty:
        standings = standings.sort_values(
            ["W", "PF", "L"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        standings.insert(0, "Rank", range(1, len(standings) + 1))

    # Resolve followed team deterministically. Prefer the saved per-user follow.
    my_team_id = None
    saved_team_id = connection.get("team_id")
    if saved_team_id is not None:
        try:
            saved_int = int(saved_team_id)
            if saved_int in team_by_id:
                my_team_id = saved_int
            else:
                _wfs_espn_clear_team(user["sub"], season, league_id)
                connection["team_id"] = None
                connection["team_name"] = ""
        except Exception:
            _wfs_espn_clear_team(user["sub"], season, league_id)
            connection["team_id"] = None
            connection["team_name"] = ""

    # For a private authenticated ESPN session, use exact ESPN ownership only when
    # the user has not already chosen a followed team. Persist that exact result.
    if my_team_id is None and auth_ready:
        auto_team_id = _espn_my_team_id(data, swid)
        if auto_team_id is not None:
            try:
                auto_team_id = int(auto_team_id)
                if auto_team_id in team_by_id:
                    my_team_id = auto_team_id
                    auto_team_name = _espn_team_name(team_by_id[auto_team_id])
                    _wfs_espn_set_team(
                        user["sub"], season, league_id, auto_team_id, auto_team_name
                    )
                    connection["team_id"] = auto_team_id
                    connection["team_name"] = auto_team_name
            except Exception:
                pass

    my_team = team_by_id.get(int(my_team_id), {}) if my_team_id is not None else {}
    my_rank = None
    if my_team_id is not None and not standings.empty:
        hit = standings[standings["Team ID"].eq(my_team_id)]
        if not hit.empty:
            my_rank = int(hit.iloc[0]["Rank"])

    # Phase 2: make the followed team the center of the ESPN experience.
    # This is presentation-only; ESPN access, persistence, WFS projections, and
    # optimizer logic remain unchanged.
    if my_team:
        wins, losses, ties, pf, _ = _espn_record(my_team)
        record = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
        team_name = _espn_team_name(my_team)
        rank_text = f"#{my_rank}" if my_rank else "—"
        st.markdown(
            f"""
            <div style="background:linear-gradient(110deg,#10233d 0%,#173b68 58%,#0b73f6 100%);
                        border-radius:18px;padding:1.05rem 1.15rem;margin:.65rem 0 .85rem;
                        color:white;box-shadow:0 8px 24px rgba(15,55,105,.14);">
              <div style="font-size:.68rem;font-weight:900;letter-spacing:.10em;color:#9fd0ff;">
                MY FANTASY TEAM • ESPN • WEEK {current_week}
              </div>
              <div style="font-size:1.45rem;font-weight:950;margin-top:.18rem;">{team_name}</div>
              <div style="font-size:.82rem;color:#d8eaff;margin-top:.2rem;">
                {league_name} • {season} • Record {record} • Rank {rank_text}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # WFS_LIVE_POINTS_FOR_V1
        #
        # ESPN standings pointsFor can remain 0 during an active
        # scoring period. When that happens, use the exact current
        # matchup-side score. _espn_side_score() uses ESPN's own
        # appliedTotal scoring fallback for active lineup players.
        #
        display_pf = pf

        if display_pf == 0:
            current_pf_game = _espn_find_matchup(
                schedule,
                my_team_id,
                current_week,
            )

            if current_pf_game:
                pf_away = (
                    current_pf_game.get("away")
                    or {}
                )
                pf_home = (
                    current_pf_game.get("home")
                    or {}
                )

                if pf_away.get("teamId") == my_team_id:
                    display_pf = _espn_side_score(
                        pf_away,
                        current_week,
                    )
                elif pf_home.get("teamId") == my_team_id:
                    display_pf = _espn_side_score(
                        pf_home,
                        current_week,
                    )

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Record", record)
        t2.metric("League Rank", rank_text)
        t3.metric("Points For", f"{display_pf:.2f}")
        t4.metric("Week", current_week)

        rc1, rc2 = st.columns([1.6, .4])
        rc1.caption(
            "This exact ESPN Team ID is saved to your WFS account for this league and season."
        )
        if rc2.button(
            "Change Followed Team",
            width="stretch",
            key=f"espn_change_followed_team_{season}_{league_id}",
        ):
            _wfs_espn_clear_team(user["sub"], season, league_id)
            st.rerun()

        my_game = _espn_find_matchup(schedule, my_team_id, current_week)
        if my_game:
            away = my_game.get("away") or {}
            home = my_game.get("home") or {}
            is_away = away.get("teamId") == my_team_id
            mine = away if is_away else home
            opp = home if is_away else away
            opp_id = opp.get("teamId")
            opp_team = team_by_id.get(int(opp_id), {}) if opp_id is not None else {}
            mine_score = _espn_side_score(mine, current_week)
            opp_score = _espn_side_score(opp, current_week)
            mine_proj = _espn_side_projection(mine)
            opp_proj = _espn_side_projection(opp)

            st.markdown("#### This Week")
            projection_line = ""
            if mine_proj is not None and opp_proj is not None:
                projection_line = (
                    f'<div style="margin-top:.45rem;color:#6b7f99;font-size:.78rem;">'
                    f'Projected: {mine_proj:.2f} — {opp_proj:.2f}</div>'
                )
            st.markdown(
                f"""
                <div class="wfs-match-card" style="padding:1rem 1.1rem;">
                  <div style="font-size:.7rem;color:#7b8ca3;font-weight:900;letter-spacing:.06em;">
                    WEEK {current_week} • YOUR MATCHUP
                  </div>
                  <div class="wfs-match-team">
                    <span>{_espn_team_name(my_team)}</span>
                    <span class="wfs-match-score">{mine_score:.2f}</span>
                  </div>
                  <div class="wfs-match-team">
                    <span>{_espn_team_name(opp_team) if opp_team else "Opponent"}</span>
                    <span class="wfs-match-score">{opp_score:.2f}</span>
                  </div>
                  {projection_line}
                </div>
                """,
                unsafe_allow_html=True,
            )

        _wfs_render_command_center_navigation()

        st.markdown('<div id="wfs-watchlist"></div>', unsafe_allow_html=True)
        _wfs_render_my_watchlist()

        st.markdown('<div id="wfs-this-week"></div>', unsafe_allow_html=True)
        _wfs_render_projection_movers()
        _wfs_render_projection_movement_summary()
        _wfs_render_roster_projection_movement(my_team)
        _wfs_render_weekly_projection_change_digest(my_team)
        _wfs_render_weekly_decision_tracker(
            my_team, season, league_id, authoritative_week=current_week
        )
        _wfs_render_recommendation_accuracy(season, league_id)
        _wfs_render_weekly_results_summary(season, league_id)
        _wfs_render_long_term_decision_performance(season, league_id)
        _wfs_render_action_center(teams, my_team)
        _wfs_render_weekly_checklist(teams, my_team, season, league_id)
        _wfs_render_opportunity_ranking(teams, my_team)
        _wfs_render_weekly_command_brief(teams, my_team)
        _wfs_render_weekly_executive_summary(teams, my_team, opp_team, season, league_id)
        _wfs_render_weekly_readiness(my_team, season, league_id)
        _wfs_render_alerts_watchlist(teams, my_team, season, league_id)

        if my_game and opp_team:
            st.markdown('<div id="wfs-lineup"></div>', unsafe_allow_html=True)
            _wfs_render_matchup_intelligence(my_team, opp_team, season, league_id)
            _wfs_render_matchup_game_plan(my_team, opp_team, season, league_id)

        st.markdown('<div id="wfs-roster"></div>', unsafe_allow_html=True)
        _wfs_render_roster_strength_intelligence(teams, my_team, season, league_id)

        st.markdown('<div id="wfs-trades"></div>', unsafe_allow_html=True)
        _wfs_render_trade_intelligence(teams, my_team, season, league_id)

        st.markdown('<div id="wfs-league"></div>', unsafe_allow_html=True)
        _wfs_render_league_power_rankings(teams, my_team, season, league_id)
        _wfs_render_contender_tiers(teams, my_team, season, league_id)

        _wfs_render_team_improvement_plan(teams, my_team, season, league_id)
        _wfs_render_my_team_command_center(my_team, season, league_id)
        _wfs_render_weekly_fantasy_leaders(teams, current_week, season, league_id)

        st.markdown('<div id="wfs-waivers"></div>', unsafe_allow_html=True)
        _wfs_render_waiver_wire_intelligence(teams, my_team, season, league_id)
    else:
        st.info(
            "Choose the exact ESPN team you want this WFS account to follow. "
            "The Season + League ID + Team ID association is saved per user; no fuzzy matching."
        )
        with st.container(border=True):
            st.markdown("#### Follow a Team")
            assignment_options = {
                _espn_team_name(team): int(team.get("id"))
                for team in teams
                if team.get("id") is not None
            }
            if assignment_options:
                ac1, ac2 = st.columns([1.5, .55])
                selected_assignment = ac1.selectbox(
                    "Which team do you want to follow?",
                    list(assignment_options),
                    key=f"espn_follow_team_{season}_{league_id}",
                )
                if ac2.button(
                    "Follow Team",
                    type="primary",
                    width="stretch",
                    key=f"espn_save_follow_team_{season}_{league_id}",
                ):
                    selected_team_id = assignment_options[selected_assignment]
                    _wfs_espn_set_team(
                        user["sub"],
                        season,
                        league_id,
                        selected_team_id,
                        selected_assignment,
                    )
                    st.success(
                        f"WFS is now following {selected_assignment} in {league_name} ({season})."
                    )
                    st.rerun()
            else:
                st.warning("ESPN returned no teams that can be followed.")

    # League-wide detail remains available below the team-first command center.
    home_tab, matchup_tab, roster_tab, standings_tab = st.tabs(
        ["🏠 League Snapshot", "⚔️ Matchups", "👥 My Roster", "📈 Standings"]
    )

    with home_tab:
        left, right = st.columns([1.05, .95])
        with left:
            st.markdown("#### League Standings")
            if standings.empty:
                st.info("No standings are available yet.")
            else:
                compact_cols = ["Rank", "Team", "W", "L", "T", "PF"]
                st.dataframe(
                    standings[compact_cols].head(8),
                    width="stretch",
                    hide_index=True,
                )
        with right:
            st.markdown("#### Week Snapshot")
            week_games = [
                g for g in schedule
                if int(g.get("matchupPeriodId") or -1) == current_week
            ]
            if not week_games:
                st.info("No matchup schedule is available for the current week.")
            else:
                for game in week_games[:4]:
                    away = game.get("away") or {}
                    home = game.get("home") or {}
                    aid = away.get("teamId")
                    hid = home.get("teamId")
                    aname = _espn_team_name(team_by_id.get(int(aid), {})) if aid is not None else "Away"
                    hname = _espn_team_name(team_by_id.get(int(hid), {})) if hid is not None else "Home"
                    st.markdown(
                        f"""
                        <div class="wfs-match-card" style="padding:.65rem .8rem;">
                          <div class="wfs-match-team">
                            <span>{aname}</span><span class="wfs-match-score">{_espn_side_score(away):.2f}</span>
                          </div>
                          <div class="wfs-match-team">
                            <span>{hname}</span><span class="wfs-match-score">{_espn_side_score(home):.2f}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with matchup_tab:
        weeks = sorted({
            int(game.get("matchupPeriodId"))
            for game in schedule
            if game.get("matchupPeriodId") is not None
        })
        if not weeks:
            st.info("No ESPN matchup schedule is available yet.")
        else:
            default_week = current_week if current_week in weeks else weeks[0]
            week = st.selectbox(
                "Matchup Week",
                weeks,
                index=weeks.index(default_week),
                key=f"espn_matchup_week_multiuser_{season}_{league_id}",
            )
            games = [
                game for game in schedule
                if int(game.get("matchupPeriodId") or -1) == week
            ]
            for i in range(0, len(games), 2):
                cols = st.columns(2)
                for j, game in enumerate(games[i:i + 2]):
                    with cols[j]:
                        away = game.get("away") or {}
                        home = game.get("home") or {}
                        aid = away.get("teamId")
                        hid = home.get("teamId")
                        aname = _espn_team_name(team_by_id.get(int(aid), {})) if aid is not None else "Away"
                        hname = _espn_team_name(team_by_id.get(int(hid), {})) if hid is not None else "Home"
                        ap = _espn_side_score(away, week)
                        hp = _espn_side_score(home, week)
                        aproj = _espn_side_projection(away)
                        hproj = _espn_side_projection(home)
                        projection_line = ""
                        if aproj is not None and hproj is not None:
                            projection_line = (
                                f'<div style="margin-top:.4rem;color:#6b7f99;font-size:.76rem;">'
                                f'Projected: {aproj:.2f} — {hproj:.2f}</div>'
                            )
                        st.markdown(
                            f"""
                            <div class="wfs-match-card">
                              <div style="font-size:.68rem;color:#7b8ca3;font-weight:850;">WEEK {week}</div>
                              <div class="wfs-match-team"><span>{aname}</span><span class="wfs-match-score">{ap:.2f}</span></div>
                              <div class="wfs-match-team"><span>{hname}</span><span class="wfs-match-score">{hp:.2f}</span></div>
                              {projection_line}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with standings_tab:
        if standings.empty:
            st.info("No standings are available.")
        else:
            public = standings.drop(columns=["Team ID"]).copy()
            st.dataframe(public, width="stretch", hide_index=True)

    with roster_tab:
        roster_options = {_espn_team_name(team): team for team in teams}
        if not roster_options:
            st.info("No ESPN rosters are available.")
        else:
            default_roster = 0
            if my_team:
                my_name = _espn_team_name(my_team)
                names = list(roster_options)
                if my_name in names:
                    default_roster = names.index(my_name)
            selected_team = st.selectbox(
                "Team Roster",
                list(roster_options),
                index=default_roster,
                key=f"espn_roster_team_multiuser_{season}_{league_id}",
            )
            team = roster_options[selected_team]
            entries = ((team.get("roster") or {}).get("entries") or [])
            rows = []
            for entry in entries:
                pool = entry.get("playerPoolEntry") or {}
                player = pool.get("player") or {}
                rows.append({
                    "Slot": _espn_lineup_slot_name(entry.get("lineupSlotId")),
                    "Player": player.get("fullName") or f"Player {player.get('id', '')}",
                    "Pro Team ID": player.get("proTeamId", ""),
                    "Acquisition": str(entry.get("acquisitionType") or "").replace("_", " ").title(),
                    "Injury": str(player.get("injuryStatus") or "").replace("_", " ").title(),
                })
            if rows:
                rdf = pd.DataFrame(rows)
                rdf["__bench"] = rdf["Slot"].isin(["Bench", "IR"]).astype(int)
                rdf = rdf.sort_values(["__bench", "Slot", "Player"], kind="stable").drop(columns="__bench")
                st.dataframe(rdf.reset_index(drop=True), width="stretch", hide_index=True)
            else:
                st.info("No roster entries were returned for this team.")

    st.caption(
        "ESPN integration is read-only. League/team follows are stored per signed-in WFS account. "
        "Private ESPN session values are encrypted at rest with the server-only WFS key and are never displayed."
    )


from wfs_live_view import render_wfs_live_view


def _wfs_home_navigate(destination, open_late_swap=False):
    """Queue deterministic navigation through the existing main sidebar radio."""
    st.session_state["wfs_nav_request"] = destination

    if open_late_swap:
        st.session_state["wfs_open_late_swap"] = True

    st.rerun()



# =========================================================
# WFS_PUBLIC_DISCLAIMER_V1
# Public presentation only.
# No projection, solver, data, injury, or model mutation.
# =========================================================
def render_wfs_public_disclaimer():
    if globals().get("mode") == "Admin":
        return

    st.markdown(
        """
        <div style="
            margin-top:1rem;
            padding:.8rem .9rem;
            border-top:1px solid #334155;
            color:#94a3b8;
            font-size:.72rem;
            line-height:1.5;
            text-align:center;
        ">
          <b>Disclaimer:</b> Wynners Fantasy Spot is an independent fantasy
          sports analytics platform and is not affiliated with or endorsed by
          the NFL or FanDuel. Projections, forecasts and recommendations are
          provided for informational purposes only and do not guarantee
          results. Users are responsible for their own fantasy sports decisions
          and for complying with applicable laws and contest rules in their
          jurisdiction.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_wfs_home():
    """WFS public command-center landing page. Presentation only."""

    st.markdown(
        """
        <style>
        .wfs-home-hero {
            background: linear-gradient(115deg,#10233d 0%,#173b68 58%,#0b73f6 100%);
            border-radius: 22px;
            padding: 1.35rem 1.35rem 1.25rem;
            margin: .15rem 0 1rem;
            color: white;
            box-shadow: 0 10px 28px rgba(15,55,105,.16);
        }

        .wfs-home-eyebrow {
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .11em;
            color: #9fd0ff;
        }

        .wfs-home-title {
            font-size: 1.75rem;
            line-height: 1.08;
            font-weight: 950;
            margin-top: .25rem;
        }

        .wfs-home-subtitle {
            font-size: .9rem;
            line-height: 1.5;
            color: #d8eaff;
            margin-top: .45rem;
            max-width: 760px;
        }

        .wfs-myweek {
            background: #f7faff;
            border: 1px solid #d8e6f5;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin: .35rem 0 1rem;
            box-shadow: 0 4px 14px rgba(15,35,60,.045);
        }

        .wfs-myweek-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .6rem;
            margin-bottom: .75rem;
        }

        .wfs-myweek-title {
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .1em;
            color: #173b68;
        }

        .wfs-myweek-season {
            background: #e8f2ff;
            color: #1769c2;
            border-radius: 999px;
            padding: .25rem .55rem;
            font-size: .65rem;
            font-weight: 900;
        }

        .wfs-myweek-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: .65rem;
        }

        .wfs-myweek-item {
            background: #ffffff;
            border: 1px solid #e0e9f3;
            border-radius: 14px;
            padding: .78rem .82rem;
            min-width: 0;
        }

        .wfs-myweek-label {
            color: #71849b;
            font-size: .64rem;
            font-weight: 900;
            letter-spacing: .075em;
            text-transform: uppercase;
        }

        .wfs-myweek-value {
            color: #172033;
            font-size: .98rem;
            font-weight: 950;
            margin-top: .2rem;
            overflow-wrap: anywhere;
        }

        .wfs-myweek-detail {
            color: #71849b;
            font-size: .7rem;
            line-height: 1.35;
            margin-top: .16rem;
        }

        .wfs-myweek-empty {
            color: #71849b;
            font-size: .78rem;
            line-height: 1.45;
            margin-top: .7rem;
        }

        .wfs-home-card {
            background: #ffffff;
            border: 1px solid #dbe5f0;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 190px;
            box-shadow: 0 5px 16px rgba(15,35,60,.06);
            margin-bottom: .7rem;
        }

        .wfs-home-card-kicker {
            font-size: .67rem;
            font-weight: 900;
            letter-spacing: .09em;
            color: #64748b;
            text-transform: uppercase;
        }

        .wfs-home-card-title {
            font-size: 1.08rem;
            font-weight: 950;
            color: #172033;
            margin-top: .3rem;
        }

        .wfs-home-card-copy {
            font-size: .82rem;
            line-height: 1.48;
            color: #52657d;
            margin-top: .42rem;
        }

        .wfs-home-card-status {
            display: inline-block;
            margin-top: .7rem;
            padding: .28rem .55rem;
            border-radius: 999px;
            background: #eef6ff;
            color: #1769c2;
            font-size: .67rem;
            font-weight: 900;
            letter-spacing: .04em;
        }

        .wfs-home-flow {
            background: #f7faff;
            border: 1px solid #dbe7f4;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin: .45rem 0 .9rem;
        }

        .wfs-home-flowline {
            color: #173b68;
            font-size: .92rem;
            font-weight: 950;
            letter-spacing: .035em;
            line-height: 1.6;
        }

        .wfs-home-flowcopy {
            color: #60738b;
            font-size: .79rem;
            line-height: 1.5;
            margin-top: .3rem;
        }

        .wfs-home-capability {
            background: #ffffff;
            border-left: 4px solid #0b73f6;
            border-radius: 10px;
            padding: .72rem .8rem;
            margin: .45rem 0;
            color: #42556d;
            font-size: .8rem;
            line-height: 1.45;
        }

        .wfs-home-capability b {
            color: #172033;
        }

        /* ========================================================
           WFS_HOME_THEME_FIX_V1
           Theme-aware Home surfaces only.
           Hero styling remains intentionally branded.
           ======================================================== */

        .wfs-myweek,
        .wfs-home-flow {
            background: var(--st-secondary-background-color) !important;
            border-color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 18%,
                    transparent
                ) !important;
        }

        .wfs-myweek-item,
        .wfs-home-card,
        .wfs-home-capability {
            background: var(--st-background-color) !important;
            border-color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 18%,
                    transparent
                ) !important;
        }

        .wfs-myweek-title,
        .wfs-home-card-title,
        .wfs-home-flowline,
        .wfs-home-capability b,
        .wfs-myweek-value {
            color: var(--st-text-color) !important;
        }

        .wfs-myweek-label,
        .wfs-myweek-detail,
        .wfs-myweek-empty,
        .wfs-home-card-kicker,
        .wfs-home-card-copy,
        .wfs-home-flowcopy,
        .wfs-home-capability {
            color:
                color-mix(
                    in srgb,
                    var(--st-text-color) 72%,
                    transparent
                ) !important;
        }

        .wfs-myweek-season,
        .wfs-home-card-status {
            background:
                color-mix(
                    in srgb,
                    var(--st-primary-color) 14%,
                    var(--st-secondary-background-color)
                ) !important;
            color: var(--st-primary-color) !important;
        }

        @media (max-width: 700px) {
            .wfs-home-hero {
                padding: 1.1rem 1rem;
                border-radius: 18px;
            }

            .wfs-home-title {
                font-size: 1.48rem;
            }

            .wfs-home-subtitle {
                font-size: .84rem;
            }

            .wfs-home-card {
                min-height: 0;
            }

            .wfs-myweek-grid {
                grid-template-columns: 1fr;
            }

            .wfs-myweek {
                padding: .9rem;
            }
        }
        </style>

        <div class="wfs-home-hero">
          <div class="wfs-home-eyebrow">WYNNERS FANTASY SPOT</div>
          <div class="wfs-home-title">Your NFL Command Center</div>
          <div class="wfs-home-subtitle">
            Prepare, build, monitor and react throughout the NFL week.
            WFS combines NFL game forecasts, player projections, matchup and
            availability intelligence, FanDuel tournament lineup construction,
            late swap, live game tracking and ESPN & Sleeper fantasy tools in
            one focused workspace.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # MY WEEK
    #
    # Read-only summary from existing WFS account persistence.
    # No ESPN network request is made from Home.
    # =====================================================

    home_user = _wfs_current_user()

    favorite_code = ""
    favorite_name = ""
    espn_team_name = ""
    espn_league_name = ""
    espn_season = 2026
    espn_connection_count = 0

    if home_user:
        try:
            _wfs_accounts_init()
            _wfs_upsert_user(home_user)

            prefs = _wfs_user_preferences(home_user["sub"])
            favorite_code = _dc_team_code(
                prefs.get("favorite_nfl_team", "")
            )

            if favorite_code:
                favorite_name = NFL_TEAM_NAMES.get(
                    favorite_code,
                    favorite_code,
                )
        except Exception:
            favorite_code = ""
            favorite_name = ""

        try:
            espn_season = int(
                st.session_state.get("wfs_espn_season", 2026)
            )

            # WFS_HOME_MULTI_PROVIDER_MY_WEEK_V1
            # My Week should reflect whichever provider (ESPN or
            # Sleeper) the account actually has connected, not ESPN
            # only. _wfs_*_connections() each have a deterministic
            # database ordering.
            home_connections = (
                _wfs_espn_connections(home_user["sub"], espn_season)
                + _wfs_sleeper_connections(home_user["sub"], espn_season)
            )

            espn_connection_count = len(home_connections)

            followed_connections = [
                c for c in home_connections
                if c.get("team_id") is not None
                and str(c.get("team_name") or "").strip()
            ]

            if followed_connections:
                home_espn = followed_connections[0]
                espn_team_name = str(
                    home_espn.get("team_name") or ""
                ).strip()
                espn_league_name = str(
                    home_espn.get("league_name") or ""
                ).strip()

        except Exception:
            espn_team_name = ""
            espn_league_name = ""
            espn_connection_count = 0

    favorite_value = (
        f"{favorite_code} — {favorite_name}"
        if favorite_code and favorite_name
        else "Not selected"
    )

    favorite_detail = (
        "Highlighted throughout NFL Data Center"
        if favorite_code
        else "Choose your team in NFL Data Center"
    )

    fantasy_value = (
        espn_team_name
        if espn_team_name
        else "Not selected"
    )

    if espn_team_name and espn_league_name:
        fantasy_detail = espn_league_name
    elif espn_connection_count:
        fantasy_detail = (
            "Choose a followed team in Fantasy Hub"
        )
    else:
        fantasy_detail = (
            "Add an ESPN or Sleeper league in Fantasy Hub"
        )

    my_week_html = (
        f'<div class="wfs-myweek">'
        f'<div class="wfs-myweek-head">'
        f'<div class="wfs-myweek-title">MY WEEK</div>'
        f'<div class="wfs-myweek-season">NFL {espn_season}</div>'
        f'</div>'
        f'<div class="wfs-myweek-grid">'
        f'<div class="wfs-myweek-item">'
        f'<div class="wfs-myweek-label">⭐ My NFL Team</div>'
        f'<div class="wfs-myweek-value">{favorite_value}</div>'
        f'<div class="wfs-myweek-detail">{favorite_detail}</div>'
        f'</div>'
        f'<div class="wfs-myweek-item">'
        f'<div class="wfs-myweek-label">🏆 Fantasy League</div>'
        f'<div class="wfs-myweek-value">{fantasy_value}</div>'
        f'<div class="wfs-myweek-detail">{fantasy_detail}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        my_week_html,
        unsafe_allow_html=True,
    )

    st.markdown("### What do you want to do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏗 BUILD</div>
              <div class="wfs-home-card-title">Lineup Generator</div>
              <div class="wfs-home-card-copy">
                Build FanDuel NFL tournament lineups using WFS projections,
                matchup information and tournament strategy.
              </div>
              <div class="wfs-home-card-status">DFS • TOURNAMENT FOCUSED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Lineup Generator →",
            key="wfs_home_open_build",
            width="stretch",
        ):
            _wfs_home_navigate("🏗 Build Lineups")

    with c2:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">⚡ REACT</div>
              <div class="wfs-home-card-title">Late Swap</div>
              <div class="wfs-home-card-copy">
                Protect players whose games have started and optimize the
                remaining FanDuel roster decisions as the slate develops.
              </div>
              <div class="wfs-home-card-status">LOCKED PLAYERS • LIVE SLATE DECISIONS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Late Swap →",
            key="wfs_home_open_late_swap",
            width="stretch",
        ):
            _wfs_home_navigate(
                "🏗 Build Lineups",
                open_late_swap=True,
            )

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏈 FOLLOW</div>
              <div class="wfs-home-card-title">NFL Data Center</div>
              <div class="wfs-home-card-copy">
                Explore schedules, game forecasts, player projections,
                results and box scores with favorite-team personalization
                and fantasy-player context.
              </div>
              <div class="wfs-home-card-status">SCHEDULE • SCORES • BOX SCORES</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open NFL Data Center →",
            key="wfs_home_open_data_center",
            width="stretch",
        ):
            _wfs_home_navigate("🏈 NFL Data Center")

    with c4:
        st.markdown(
            """
            <div class="wfs-home-card">
              <div class="wfs-home-card-kicker">🏆 COMPETE</div>
              <div class="wfs-home-card-title">Fantasy League Hub</div>
              <div class="wfs-home-card-copy">
                Bring your ESPN or Sleeper leagues into WFS to follow
                your team, weekly matchup, roster and league standings.
              </div>
              <div class="wfs-home-card-status">ESPN • SLEEPER • MATCHUPS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Fantasy League Hub →",
            key="wfs_home_open_fantasy",
            width="stretch",
        ):
            _wfs_home_navigate("🏆 Fantasy League Hub")

    st.markdown("### Your WFS Workflow")

    st.markdown(
        """
        <div class="wfs-home-flow">
          <div class="wfs-home-flowline">
            PREPARE &nbsp;→&nbsp; BUILD &nbsp;→&nbsp; MONITOR &nbsp;→&nbsp; REACT
          </div>
          <div class="wfs-home-flowcopy">
            Study the slate and NFL week. Build tournament lineups.
            Monitor games and your fantasy players. Then use Late Swap
            when the remaining slate gives you another decision to make.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # WFS_PUBLIC_FEEDBACK_V1
    # Public slate availability + observation-only feedback.
    # No projection, solver, identity, injury, or slate mutation.
    # =====================================================

    st.markdown("### FanDuel Slate Availability")

    st.info(
        "Slates appear when FanDuel releases them. Main, 1 PM Only, "
        "4 PM Only, primetime, and other supported slates will "
        "automatically appear when available. "
        "For SNF–MNF, late-swap options are shown as the Sunday slate "
        "develops; players are protected once their games have started."
    )

    try:
        from wfs_user_feedback import render_public_feedback

        feedback_user_id = None

        if home_user:
            feedback_user_id = str(
                home_user.get("sub") or ""
            ).strip() or None

        # WFS_FEEDBACK_SCHEDULE_CONTEXT_V1
        # Planning context comes from authoritative NFL schedule state,
        # never file timestamps, ESPN state, or a hardcoded week.
        from wfs_schedule_context import resolve_schedule_week_context

        feedback_schedule_context = resolve_schedule_week_context()

        render_public_feedback(
            user_id=feedback_user_id,
            user_name=(
                str(home_user.get("name") or "").strip()
                if home_user
                else None
            ),
            user_email=(
                str(home_user.get("email") or "").strip()
                if home_user
                else None
            ),
            season=feedback_schedule_context.season,
            week=feedback_schedule_context.planning_week,
        )

    except Exception:
        st.info(
            "Feedback is temporarily unavailable. "
            "Please check back shortly."
        )

# WFS_FANTASY_MULTI_PROVIDER_HUB_V1
# WFS_SLEEPER_LEAGUE_VIEWS_V1
def render_sleeper_league_views(saved, season, week):
    """
    Provider-native Sleeper league views.

    Matchup, standings and roster data come directly from Sleeper.
    No ESPN-shaped translation and no fuzzy WFS identity resolution.
    """
    league_id = str(saved.get("league_id") or "").strip()

    try:
        roster_id = int(saved.get("team_id"))
    except (TypeError, ValueError):
        st.info("This Sleeper team does not have a valid roster assignment.")
        return

    if not league_id:
        return

    try:
        league = _sleeper_league(league_id)
        users = _sleeper_league_users(league_id)
        rosters = _sleeper_league_rosters(league_id)
        matchups = _sleeper_league_matchups(league_id, week)
        players = sleeper_players()
    except Exception:
        st.info(
            "Sleeper league details are temporarily unavailable. "
            "Please try again shortly."
        )
        return

    if not league or not rosters:
        st.info("Sleeper league details are currently unavailable.")
        return

    roster_by_id = {}
    for roster in rosters:
        if not isinstance(roster, dict):
            continue
        try:
            rid = int(roster.get("roster_id"))
        except (TypeError, ValueError):
            continue
        roster_by_id[rid] = roster

    my_roster = roster_by_id.get(roster_id)
    if my_roster is None:
        st.info(
            "Your saved Sleeper roster could not be verified in this league."
        )
        return

    user_by_id = {
        str(user.get("user_id")): user
        for user in users
        if isinstance(user, dict) and user.get("user_id") is not None
    }

    def team_name_for(roster):
        owner_id = str(roster.get("owner_id") or "")
        user = user_by_id.get(owner_id, {})
        metadata = user.get("metadata") or {}

        return (
            str(metadata.get("team_name") or "").strip()
            or str(user.get("display_name") or "").strip()
            or f"Roster {roster.get('roster_id')}"
        )

    def player_info(player_id):
        pid = str(player_id or "")
        player = players.get(pid) or {}

        name = (
            str(player.get("full_name") or "").strip()
            or str(player.get("first_name") or "").strip()
            or f"Sleeper Player {pid}"
        )

        return {
            "id": pid,
            "name": name,
            "position": str(player.get("position") or "—"),
            "team": str(player.get("team") or "FA"),
            "status": str(player.get("status") or "Unknown"),
        }

    # WFS_SLEEPER_OUTLOOK_V1
    # Analysis-only WFS enrichment for Sleeper views.
    # Exact Sleeper ID -> GSIS only. No fuzzy/name fallback.
    wfs_outlook_by_sleeper_id = {}

    try:
        import sqlite3

        outlook_path = (
            Path(__file__).resolve().parent
            / "data"
            / "parquet"
            / "current_unified_fanduel_expectation.parquet"
        )
        identity_path = (
            Path(__file__).resolve().parent
            / "data"
            / "nfl.db"
        )

        outlook_df = pd.read_parquet(outlook_path)

        outlook_df = outlook_df[
            outlook_df["source_component"].astype(str).eq("OFFENSE")
            & pd.to_numeric(
                outlook_df["season"], errors="coerce"
            ).eq(int(season))
            & pd.to_numeric(
                outlook_df["week"], errors="coerce"
            ).eq(int(week))
            & outlook_df["source_contract"].astype(str).eq(
                "WFS_FANDUEL_OFFENSE_EXPECTATION_V1"
            )
        ].copy()

        if not outlook_df.empty:
            outlook_df["entity_id"] = (
                outlook_df["entity_id"].astype(str).str.strip()
            )

            valid_contract = (
                not outlook_df["entity_id"].duplicated().any()
                and outlook_df["entity_id"]
                .str.match(r"^00-\d{7}$")
                .all()
            )

            if not valid_contract:
                raise ValueError(
                    "WFS outlook identity contract failed"
                )

            outlook_by_gsis = {
                str(row.entity_id): {
                    "points": float(
                        row.expected_fanduel_points
                    ),
                    "team": str(
                        row.team or ""
                    ).strip(),
                    "opponent": str(
                        row.opponent_team or ""
                    ).strip(),
                }
                for row in outlook_df.itertuples(index=False)
            }

            with sqlite3.connect(identity_path) as conn:
                identity_rows = conn.execute(
                    """
                    SELECT sleeper_id, gsis_id
                    FROM player_identity
                    WHERE sleeper_id IS NOT NULL
                      AND gsis_id IS NOT NULL
                      AND TRIM(CAST(sleeper_id AS TEXT)) <> ''
                      AND TRIM(CAST(gsis_id AS TEXT)) <> ''
                    """
                ).fetchall()

            sleeper_to_gsis = {}
            duplicate_ids = set()

            for sleeper_id, gsis_id in identity_rows:
                sid = str(sleeper_id).strip()
                gid = str(gsis_id).strip()

                if sid in sleeper_to_gsis:
                    duplicate_ids.add(sid)
                else:
                    sleeper_to_gsis[sid] = gid

            if duplicate_ids:
                raise ValueError(
                    "Duplicate Sleeper identity mappings detected"
                )

            for sid, gid in sleeper_to_gsis.items():
                outlook = outlook_by_gsis.get(gid)

                if outlook is not None:
                    wfs_outlook_by_sleeper_id[sid] = outlook

    except Exception:
        # WFS enrichment fails closed without breaking Sleeper V1.
        wfs_outlook_by_sleeper_id = {}

    def wfs_outlook(player_id):
        return wfs_outlook_by_sleeper_id.get(
            str(player_id or "").strip()
        )

    def wfs_outlook_text(player_id):
        outlook = wfs_outlook(player_id)

        if outlook is None:
            return "WFS Outlook —"

        context = ""

        if outlook["team"] and outlook["opponent"]:
            context = (
                f" • {outlook['team']} "
                f"vs {outlook['opponent']}"
            )

        return (
            f"WFS Outlook {outlook['points']:.2f}"
            f"{context}"
        )

    # WFS_SLEEPER_CORE_V1
    # League-specific analysis-only projection using the validated
    # WFS stat forecast contract and Sleeper scoring settings.
    # Exact Sleeper ID -> GSIS only. No fuzzy/name fallback.
    #
    # "Core" is intentional: passing INTs, offensive 2PT conversions,
    # and fumbles lost are not forecast by the current stat contract.
    wfs_sleeper_core_by_sleeper_id = {}

    try:
        scoring_settings = league.get("scoring_settings")

        if (
            not isinstance(scoring_settings, dict)
            or not scoring_settings
        ):
            raise ValueError(
                "Sleeper league scoring settings unavailable"
            )

        stat_path = (
            Path(__file__).resolve().parent
            / "data"
            / "parquet"
            / "nfl_current_unified_stat_forecasts.parquet"
        )

        stat_df = pd.read_parquet(stat_path)

        required_columns = {
            "entity_type",
            "player_id",
            "position",
            "expected_passing_yards",
            "expected_passing_tds",
            "expected_rushing_yards",
            "expected_rushing_tds",
            "expected_receptions",
            "expected_receiving_yards",
            "expected_receiving_tds",
        }

        if not required_columns.issubset(stat_df.columns):
            raise ValueError(
                "WFS Sleeper Core stat contract columns missing"
            )

        stat_df = stat_df[
            stat_df["entity_type"].astype(str).eq("OFFENSE_PLAYER")
        ].copy()

        if stat_df.empty:
            raise ValueError(
                "WFS Sleeper Core offense forecast is empty"
            )

        stat_df["player_id"] = (
            stat_df["player_id"].astype(str).str.strip()
        )

        valid_stat_identity = (
            not stat_df["player_id"].duplicated().any()
            and stat_df["player_id"]
            .str.match(r"^00-\d{7}$")
            .all()
        )

        if not valid_stat_identity:
            raise ValueError(
                "WFS Sleeper Core identity contract failed"
            )

        def _core_number(value):
            if pd.isna(value):
                return 0.0
            return float(value)

        def _core_points(row):
            position = str(row.position or "").strip().upper()

            if position == "QB":
                return (
                    _core_number(row.expected_passing_yards)
                    * float(scoring_settings.get("pass_yd", 0.0) or 0.0)
                    + _core_number(row.expected_passing_tds)
                    * float(scoring_settings.get("pass_td", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_yards)
                    * float(scoring_settings.get("rush_yd", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_tds)
                    * float(scoring_settings.get("rush_td", 0.0) or 0.0)
                )

            if position == "RB":
                return (
                    _core_number(row.expected_rushing_yards)
                    * float(scoring_settings.get("rush_yd", 0.0) or 0.0)
                    + _core_number(row.expected_rushing_tds)
                    * float(scoring_settings.get("rush_td", 0.0) or 0.0)
                    + _core_number(row.expected_receptions)
                    * float(scoring_settings.get("rec", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_yards)
                    * float(scoring_settings.get("rec_yd", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_tds)
                    * float(scoring_settings.get("rec_td", 0.0) or 0.0)
                )

            if position in {"WR", "TE"}:
                reception_value = float(
                    scoring_settings.get("rec", 0.0) or 0.0
                )

                if position == "TE":
                    reception_value += float(
                        scoring_settings.get(
                            "bonus_rec_te", 0.0
                        ) or 0.0
                    )

                return (
                    _core_number(row.expected_receptions)
                    * reception_value
                    + _core_number(row.expected_receiving_yards)
                    * float(scoring_settings.get("rec_yd", 0.0) or 0.0)
                    + _core_number(row.expected_receiving_tds)
                    * float(scoring_settings.get("rec_td", 0.0) or 0.0)
                )

            return None

        stat_by_gsis = {
            str(row.player_id): row
            for row in stat_df.itertuples(index=False)
        }

        # Reuse the exact identity map already validated by
        # WFS_SLEEPER_OUTLOOK_V1. If that boundary failed closed,
        # Core also fails closed.
        if "sleeper_to_gsis" not in locals():
            raise ValueError(
                "WFS Sleeper Core exact identity map unavailable"
            )

        for sid, gid in sleeper_to_gsis.items():
            row = stat_by_gsis.get(gid)

            if row is None:
                continue

            points = _core_points(row)

            if points is None:
                continue

            wfs_sleeper_core_by_sleeper_id[sid] = {
                "points": float(points),
            }

    except Exception:
        # Core enrichment fails closed without breaking Sleeper V1
        # or the already validated WFS Outlook surface.
        wfs_sleeper_core_by_sleeper_id = {}

    def wfs_sleeper_core_text(player_id):
        core = wfs_sleeper_core_by_sleeper_id.get(
            str(player_id or "").strip()
        )

        if core is None:
            return "Sleeper Core —"

        return f"Sleeper Core {core['points']:.2f}"

    # WFS_SLEEPER_START_SIT_V1
    # Analysis-only lineup decision intelligence built on the validated
    # Sleeper Core V1 projection layer.
    #
    # This does not change the user's Sleeper lineup. It compares the
    # provider-native current starters with the best legal Core lineup.
    # Exact Sleeper identity/Core availability is required; unavailable
    # players fail closed and are never assigned fabricated zero points.
    wfs_start_sit = {
        "available": False,
        "reason": "",
        "optimal_total": None,
        "current_total": None,
        "difference": None,
        "recommendations": [],
        "optimal_ids": set(),
        "replacement_costs": {},
        "median_cost": None,
        "p75_cost": None,
    }

    try:
        from ortools.sat.python import cp_model

        roster_positions = league.get("roster_positions") or []

        starter_slots = [
            str(slot).strip().upper()
            for slot in roster_positions
            if str(slot).strip().upper() != "BN"
        ]

        supported_slots = {
            "QB",
            "RB",
            "WR",
            "TE",
            "FLEX",
            "SUPER_FLEX",
        }

        if (
            not starter_slots
            or any(slot not in supported_slots for slot in starter_slots)
        ):
            raise ValueError(
                "Sleeper lineup contains unsupported starter slots"
            )

        roster_player_ids = [
            str(pid).strip()
            for pid in (my_roster.get("players") or [])
            if str(pid).strip()
        ]

        current_starter_ids = [
            str(pid).strip()
            for pid in (my_roster.get("starters") or [])
            if str(pid).strip()
        ]

        # WFS_SLEEPER_PLACEMENT_GATE_V1
        #
        # Reserve and taxi players remain provider-rostered,
        # but are not legal Start/Sit candidates.
        roster_player_set = set(roster_player_ids)

        reserve_player_ids = {
            str(pid).strip()
            for pid in (my_roster.get("reserve") or [])
            if str(pid).strip()
        }

        taxi_player_ids = {
            str(pid).strip()
            for pid in (my_roster.get("taxi") or [])
            if str(pid).strip()
        }

        if not reserve_player_ids.issubset(
            roster_player_set
        ):
            raise ValueError(
                "Sleeper reserve contains player outside roster"
            )

        if not taxi_player_ids.issubset(
            roster_player_set
        ):
            raise ValueError(
                "Sleeper taxi contains player outside roster"
            )

        if reserve_player_ids & taxi_player_ids:
            raise ValueError(
                "Sleeper reserve/taxi overlap"
            )

        startable_roster_player_ids = [
            pid
            for pid in roster_player_ids
            if pid not in reserve_player_ids
            and pid not in taxi_player_ids
        ]

        current_starter_set = set(
            current_starter_ids
        )

        if not current_starter_set.issubset(
            set(startable_roster_player_ids)
        ):
            raise ValueError(
                "Current Sleeper starters include "
                "reserve/taxi player"
            )

        if len(current_starter_ids) != len(starter_slots):
            raise ValueError(
                "Sleeper starter count does not match lineup slots"
            )

        def _decision_positions(player_id):
            player = players.get(str(player_id)) or {}

            raw_positions = player.get("fantasy_positions")

            if isinstance(raw_positions, (list, tuple, set)):
                positions = {
                    str(pos).strip().upper()
                    for pos in raw_positions
                    if str(pos).strip()
                }
            else:
                position = str(
                    player.get("position") or ""
                ).strip().upper()

                positions = {position} if position else set()

            return positions & {"QB", "RB", "WR", "TE"}

        def _decision_eligible(positions, slot):
            if slot in {"QB", "RB", "WR", "TE"}:
                return slot in positions

            if slot == "FLEX":
                return bool(
                    positions & {"RB", "WR", "TE"}
                )

            if slot == "SUPER_FLEX":
                return bool(
                    positions & {"QB", "RB", "WR", "TE"}
                )

            return False

        # WFS_SLEEPER_AVAILABILITY_GATE_V1
        #
        # Start/Sit candidates must pass the same current-player
        # availability principles used by GAV2. Core projections are
        # never rewritten or zeroed because of availability.
        #
        # Exact current season/week weekly_rosters is required for each
        # candidate. Hard roster unavailability blocks even when there
        # is no structured injury-consensus row. An exact current-week
        # injury-consensus BLOCK also removes the candidate.
        hard_roster_statuses = {
            "RES",
            "CUT",
            "DEV",
            "EXE",
            "INA",
            "INACTIVE",
            "RET",
            "RETIRED",
            "WAI",
            "WAIVED",
        }

        with sqlite3.connect(DATABASE_PATH) as availability_conn:
            roster_authority = pd.read_sql_query(
                """
                SELECT
                    gsis_id,
                    team,
                    status,
                    status_description_abbr,
                    updated_at
                FROM weekly_rosters
                WHERE season = ?
                  AND week = ?
                  AND gsis_id IS NOT NULL
                  AND TRIM(CAST(gsis_id AS TEXT)) <> ''
                """,
                availability_conn,
                params=[int(season), int(week)],
            )

            injury_authority = pd.read_sql_query(
                """
                SELECT
                    gsis_id,
                    injury_gate
                FROM injury_consensus_current
                WHERE season = ?
                  AND week = ?
                  AND game_type = 'REG'
                """,
                availability_conn,
                params=[int(season), int(week)],
            )

        roster_authority["gsis_id"] = (
            roster_authority["gsis_id"]
            .astype(str)
            .str.strip()
        )

        injury_authority["gsis_id"] = (
            injury_authority["gsis_id"]
            .astype(str)
            .str.strip()
        )

        # Match frozen GAV2 current-roster authority exactly:
        # latest valid updated_at record per exact GSIS controls state.
        roster_authority["_updated"] = pd.to_datetime(
            roster_authority["updated_at"],
            errors="coerce",
            utc=True,
        )

        if roster_authority["_updated"].isna().any():
            raise ValueError(
                "Current weekly roster contains invalid updated_at"
            )

        for column in (
            "team",
            "status",
            "status_description_abbr",
        ):
            roster_authority[column] = (
                roster_authority[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        roster_authority = roster_authority.sort_values(
            [
                "gsis_id",
                "_updated",
                "team",
                "status",
                "status_description_abbr",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )

        roster_authority = (
            roster_authority
            .drop_duplicates(
                subset=["gsis_id"],
                keep="first",
            )
            .drop(columns=["_updated"])
            .copy()
        )

        if roster_authority["gsis_id"].duplicated().any():
            raise ValueError(
                "Latest roster state is not singleton per GSIS"
            )

        if injury_authority["gsis_id"].duplicated().any():
            raise ValueError(
                "Current injury authority is not unique by GSIS"
            )

        roster_by_gsis = {
            str(row.gsis_id): row
            for row in roster_authority.itertuples(index=False)
        }

        injury_by_gsis = {
            str(row.gsis_id): row
            for row in injury_authority.itertuples(index=False)
        }

        blocked_decision_ids = set()
        decision_pool = []

        for pid in startable_roster_player_ids:
            core = wfs_sleeper_core_by_sleeper_id.get(pid)

            if core is None:
                continue

            gid = str(
                sleeper_to_gsis.get(pid) or ""
            ).strip()

            if not gid:
                continue

            roster_row = roster_by_gsis.get(gid)

            # Start/Sit is an action-oriented recommendation surface.
            # Missing current-week roster authority cannot establish
            # that a player is currently startable, so fail closed.
            if roster_row is None:
                blocked_decision_ids.add(pid)
                continue

            roster_status = str(
                roster_row.status or ""
            ).strip().upper()

            if roster_status in hard_roster_statuses:
                blocked_decision_ids.add(pid)
                continue

            injury_row = injury_by_gsis.get(gid)

            if injury_row is not None:
                injury_gate = str(
                    injury_row.injury_gate or ""
                ).strip().upper()

                if injury_gate == "BLOCK":
                    blocked_decision_ids.add(pid)
                    continue

            positions = _decision_positions(pid)

            if not positions:
                continue

            points = float(core["points"])

            decision_pool.append(
                {
                    "id": pid,
                    "positions": positions,
                    "points": points,
                }
            )

        if len(decision_pool) < len(starter_slots):
            raise ValueError(
                "Insufficient exact Core players for legal lineup"
            )

        def _solve_core_lineup(exclude_id=None):
            model = cp_model.CpModel()
            variables = {}

            for player_index, player in enumerate(decision_pool):
                if (
                    exclude_id is not None
                    and player["id"] == str(exclude_id)
                ):
                    continue

                for slot_index, slot in enumerate(starter_slots):
                    if _decision_eligible(
                        player["positions"],
                        slot,
                    ):
                        variables[
                            (player_index, slot_index)
                        ] = model.NewBoolVar(
                            f"wfs_ss_{player_index}_{slot_index}"
                        )

            for slot_index in range(len(starter_slots)):
                slot_vars = [
                    var
                    for (player_index, candidate_slot), var
                    in variables.items()
                    if candidate_slot == slot_index
                ]

                if not slot_vars:
                    return None

                model.Add(sum(slot_vars) == 1)

            for player_index in range(len(decision_pool)):
                player_vars = [
                    var
                    for (candidate_player, slot_index), var
                    in variables.items()
                    if candidate_player == player_index
                ]

                if player_vars:
                    model.Add(sum(player_vars) <= 1)

            model.Maximize(
                sum(
                    int(
                        round(
                            decision_pool[player_index]["points"]
                            * 100
                        )
                    )
                    * var
                    for (player_index, slot_index), var
                    in variables.items()
                )
            )

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 1.0
            solver.parameters.num_search_workers = 2

            status = solver.Solve(model)

            if status not in (
                cp_model.OPTIMAL,
                cp_model.FEASIBLE,
            ):
                return None

            lineup = []

            for slot_index, slot in enumerate(starter_slots):
                selected = None

                for (
                    player_index,
                    candidate_slot,
                ), var in variables.items():
                    if (
                        candidate_slot == slot_index
                        and solver.Value(var)
                    ):
                        selected = decision_pool[player_index]
                        break

                if selected is None:
                    return None

                lineup.append(
                    {
                        "slot": slot,
                        "id": selected["id"],
                        "points": selected["points"],
                    }
                )

            total = sum(
                player["points"]
                for player in lineup
            )

            return {
                "total": float(total),
                "lineup": lineup,
                "status": solver.StatusName(status),
            }

        optimal = _solve_core_lineup()

        if optimal is None:
            raise ValueError(
                "No legal WFS Sleeper Core lineup"
            )

        optimal_ids = {
            row["id"]
            for row in optimal["lineup"]
        }

        current_core_rows = [
            wfs_sleeper_core_by_sleeper_id.get(pid)
            for pid in current_starter_ids
        ]

        if any(row is None for row in current_core_rows):
            raise ValueError(
                "Current Sleeper starters are not fully Core-resolved"
            )

        current_total = sum(
            float(row["points"])
            for row in current_core_rows
        )

        replacement_costs = {}

        for row in optimal["lineup"]:
            pid = row["id"]

            alternative = _solve_core_lineup(
                exclude_id=pid
            )

            if alternative is None:
                continue

            replacement_costs[pid] = max(
                0.0,
                float(optimal["total"])
                - float(alternative["total"]),
            )

        cost_values = sorted(
            replacement_costs.values()
        )

        if not cost_values:
            raise ValueError(
                "No legal replacement-cost distribution"
            )

        cost_series = pd.Series(
            cost_values,
            dtype="float64",
        )

        median_cost = float(
            cost_series.quantile(0.50)
        )

        p75_cost = float(
            cost_series.quantile(0.75)
        )

        current_set = set(current_starter_ids)

        start_ids = [
            pid
            for pid in optimal_ids
            if pid not in current_set
        ]

        sit_ids = [
            pid
            for pid in current_starter_ids
            if pid not in optimal_ids
        ]

        recommendations = []

        for pid in start_ids:
            cost = replacement_costs.get(pid)

            if cost is None:
                strength = "Decision Edge"
            elif cost >= p75_cost:
                strength = "Clear Edge"
            elif cost >= median_cost:
                strength = "Meaningful Edge"
            else:
                strength = "Close Decision"

            recommendations.append(
                {
                    "action": "START",
                    "player_id": pid,
                    "replacement_cost": cost,
                    "strength": strength,
                }
            )

        for pid in sit_ids:
            recommendations.append(
                {
                    "action": "SIT",
                    "player_id": pid,
                    "replacement_cost": None,
                    "strength": None,
                }
            )

        recommendations.sort(
            key=lambda row: (
                0 if row["action"] == "START" else 1,
                -(
                    row["replacement_cost"]
                    if row["replacement_cost"] is not None
                    else -1.0
                ),
                player_info(row["player_id"])["name"].lower(),
            )
        )

        wfs_start_sit = {
            "available": True,
            "reason": "",
            "optimal_total": float(optimal["total"]),
            "current_total": float(current_total),
            "difference": float(
                optimal["total"] - current_total
            ),
            "recommendations": recommendations,
            "optimal_ids": optimal_ids,
            "replacement_costs": replacement_costs,
            "median_cost": median_cost,
            "p75_cost": p75_cost,
        }

    except Exception:
        # Start/Sit fails closed without changing provider-native Sleeper,
        # Outlook V1, Core V1, ESPN, or the FanDuel optimizer.
        wfs_start_sit = {
            "available": False,
            "reason": "",
            "optimal_total": None,
            "current_total": None,
            "difference": None,
            "recommendations": [],
            "optimal_ids": set(),
            "replacement_costs": {},
            "median_cost": None,
            "p75_cost": None,
        }

    # WFS_SLEEPER_START_SIT_UI_V1
    def render_wfs_start_sit():
        if not wfs_start_sit.get("available"):
            return

        recommendations = (
            wfs_start_sit.get("recommendations") or []
        )

        starts = [
            row
            for row in recommendations
            if row.get("action") == "START"
        ]

        sits = [
            row
            for row in recommendations
            if row.get("action") == "SIT"
        ]

        if not starts and not sits:
            return

        st.markdown("#### WFS Start/Sit")

        difference = wfs_start_sit.get("difference")

        if difference is not None:
            st.caption(
                "Projected lineup gain: "
                f"{float(difference):+.2f} Sleeper Core"
            )

        for row in starts:
            info = player_info(row.get("player_id"))
            name = info["name"]

            strength = str(
                row.get("strength") or ""
            ).strip()

            if strength:
                st.markdown(
                    f"**Start {name}** — {strength}"
                )
            else:
                st.markdown(
                    f"**Start {name}**"
                )

        for row in sits:
            info = player_info(row.get("player_id"))

            st.markdown(
                f"**Sit {info['name']}**"
            )

        st.caption(
            "WFS lineup recommendation based on "
            "current player outlook and your league settings."
        )

    def fpts_from_settings(settings):
        settings = settings or {}
        whole = settings.get("fpts", 0) or 0
        decimal = settings.get("fpts_decimal", 0) or 0

        try:
            return float(whole) + (float(decimal) / 100.0)
        except (TypeError, ValueError):
            return 0.0

    render_wfs_start_sit()
    st.divider()

    tabs = st.tabs(["🏈 Matchup", "📊 Standings", "👥 Roster"])

    # -------------------------
    # MATCHUP
    # -------------------------
    with tabs[0]:
        st.markdown(f"#### Week {week} Matchup")

        my_matchup = next(
            (
                row
                for row in matchups
                if isinstance(row, dict)
                and int(row.get("roster_id", -1)) == roster_id
            ),
            None,
        )

        if my_matchup is None:
            st.info(f"No Sleeper matchup is available for Week {week}.")
        else:
            matchup_id = my_matchup.get("matchup_id")

            opponent_matchup = next(
                (
                    row
                    for row in matchups
                    if isinstance(row, dict)
                    and row.get("matchup_id") == matchup_id
                    and int(row.get("roster_id", -1)) != roster_id
                ),
                None,
            )

            opponent_roster = None
            if opponent_matchup is not None:
                try:
                    opponent_roster = roster_by_id.get(
                        int(opponent_matchup.get("roster_id"))
                    )
                except (TypeError, ValueError):
                    opponent_roster = None

            my_name = team_name_for(my_roster)
            opponent_name = (
                team_name_for(opponent_roster)
                if opponent_roster
                else "Opponent"
            )

            my_points = float(my_matchup.get("points") or 0.0)
            opponent_points = (
                float(opponent_matchup.get("points") or 0.0)
                if opponent_matchup
                else 0.0
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(my_name, f"{my_points:.2f}")

            with c2:
                st.metric(opponent_name, f"{opponent_points:.2f}")

            st.caption(
                "Scores shown here are Sleeper's current fantasy scores."
            )

            left, right = st.columns(2)

            def render_starters(column, label, matchup_row):
                with column:
                    st.markdown(f"**{label}**")

                    starter_ids = (
                        matchup_row.get("starters") or []
                        if matchup_row
                        else []
                    )
                    starter_points = (
                        matchup_row.get("starters_points") or []
                        if matchup_row
                        else []
                    )

                    for idx, player_id in enumerate(starter_ids):
                        info = player_info(player_id)

                        points = 0.0
                        if idx < len(starter_points):
                            try:
                                points = float(
                                    starter_points[idx] or 0.0
                                )
                            except (TypeError, ValueError):
                                points = 0.0

                        st.markdown(
                            f"**{info['name']}**  \n"
                            f"{info['position']} • {info['team']} "
                            f"— {points:.2f}  \n"
                            f"{wfs_outlook_text(player_id)}  \n"
                            f"{wfs_sleeper_core_text(player_id)}"
                        )

            render_starters(
                left,
                my_name,
                my_matchup,
            )

            render_starters(
                right,
                opponent_name,
                opponent_matchup,
            )

    # -------------------------
    # STANDINGS
    # -------------------------
    with tabs[1]:
        st.markdown("#### League Standings")

        standings = []

        for roster in rosters:
            if not isinstance(roster, dict):
                continue

            settings = roster.get("settings") or {}

            try:
                rid = int(roster.get("roster_id"))
            except (TypeError, ValueError):
                continue

            standings.append(
                {
                    "roster_id": rid,
                    "team": team_name_for(roster),
                    "wins": int(settings.get("wins") or 0),
                    "losses": int(settings.get("losses") or 0),
                    "ties": int(settings.get("ties") or 0),
                    "points": fpts_from_settings(settings),
                }
            )

        standings.sort(
            key=lambda row: (
                -row["wins"],
                row["losses"],
                -row["ties"],
                -row["points"],
                row["team"].lower(),
            )
        )

        for rank, row in enumerate(standings, start=1):
            marker = " • Your Team" if row["roster_id"] == roster_id else ""

            st.markdown(
                f"**{rank}. {row['team']}**{marker}  \n"
                f"{row['wins']}-{row['losses']}"
                f"{('-' + str(row['ties'])) if row['ties'] else ''}"
                f" • {row['points']:.2f} PF"
            )

    # -------------------------
    # ROSTER
    # -------------------------
    with tabs[2]:
        st.markdown(f"#### {team_name_for(my_roster)}")

        all_players = [
            str(pid)
            for pid in (my_roster.get("players") or [])
        ]

        starters = [
            str(pid)
            for pid in (my_roster.get("starters") or [])
        ]

        taxi = {
            str(pid)
            for pid in (my_roster.get("taxi") or [])
        }

        reserve = {
            str(pid)
            for pid in (my_roster.get("reserve") or [])
        }

        starter_set = set(starters)

        bench = [
            pid
            for pid in all_players
            if pid not in starter_set
            and pid not in taxi
            and pid not in reserve
        ]

        def render_player_group(title, ids):
            st.markdown(f"**{title}**")

            if not ids:
                st.caption("None")
                return

            for pid in ids:
                info = player_info(pid)

                st.markdown(
                    f"**{info['name']}**  \n"
                    f"{info['position']} • {info['team']} "
                    f"• {info['status']}  \n"
                    f"{wfs_outlook_text(pid)}  \n"
                    f"{wfs_sleeper_core_text(pid)}"
                )

        render_player_group("Starters", starters)
        st.divider()

        render_player_group("Bench", bench)

        if reserve:
            st.divider()
            render_player_group("Reserve", sorted(reserve))

        if taxi:
            st.divider()
            render_player_group("Taxi", sorted(taxi))


def render_sleeper_league_hub():
    """
    Sleeper Fantasy Football V1 connection surface.

    Read-only Sleeper API integration. Users connect by Sleeper username;
    no Sleeper password or authentication secret is collected by WFS.
    """
    try:
        user = dict(st.user)
    except Exception:
        user = {}

    user_sub = str(user.get("sub") or "").strip()
    if not user_sub:
        st.info("Sign in to WFS to connect a Sleeper fantasy league.")
        return

    try:
        from wfs_schedule_context import resolve_schedule_week_context

        schedule_context = resolve_schedule_week_context()
        season = int(schedule_context.season)
    except Exception:
        st.info(
            "Sleeper league connection is temporarily unavailable because "
            "the current NFL season could not be verified."
        )
        return

    st.markdown("### Sleeper Fantasy Football")
    st.caption(
        "Connect with your Sleeper username. WFS never asks for or stores "
        "your Sleeper password."
    )

    username = st.text_input(
        "Sleeper username",
        key="wfs_sleeper_username",
        placeholder="Enter your Sleeper username",
    )

    if st.button(
        "Find My Sleeper Leagues",
        key="wfs_sleeper_find_leagues",
        width="stretch",
    ):
        value = str(username or "").strip()

        if not value:
            st.warning("Enter your Sleeper username first.")
        else:
            try:
                sleeper_user = _sleeper_user(value)

                if not sleeper_user:
                    st.warning(
                        "WFS could not find that Sleeper user. "
                        "Check the username and try again."
                    )
                else:
                    sleeper_user_id = str(
                        sleeper_user.get("user_id") or ""
                    ).strip()

                    leagues = _sleeper_user_leagues(
                        sleeper_user_id,
                        season,
                    )

                    valid_leagues = []

                    for league in leagues:
                        if not isinstance(league, dict):
                            continue

                        league_id = str(
                            league.get("league_id") or ""
                        ).strip()

                        if not league_id:
                            continue

                        if str(
                            league.get("sport") or ""
                        ).lower() != "nfl":
                            continue

                        if str(
                            league.get("season") or ""
                        ) != str(season):
                            continue

                        rosters = _sleeper_league_rosters(
                            league_id
                        )

                        my_roster = _sleeper_roster_for_user(
                            rosters,
                            sleeper_user_id,
                        )

                        if my_roster is None:
                            continue

                        roster_id = my_roster.get("roster_id")

                        if roster_id is None:
                            continue

                        metadata = (
                            sleeper_user.get("metadata") or {}
                        )

                        team_name = (
                            str(
                                metadata.get("team_name") or ""
                            ).strip()
                            or str(
                                sleeper_user.get(
                                    "display_name"
                                ) or ""
                            ).strip()
                            or str(
                                sleeper_user.get(
                                    "username"
                                ) or ""
                            ).strip()
                            or "My Sleeper Team"
                        )

                        valid_leagues.append(
                            {
                                "league_id": league_id,
                                "league_name": str(
                                    league.get("name")
                                    or (
                                        "Sleeper League "
                                        f"{league_id}"
                                    )
                                ),
                                "roster_id": int(roster_id),
                                "team_name": team_name,
                            }
                        )

                    st.session_state[
                        "wfs_sleeper_discovered_user_id"
                    ] = sleeper_user_id

                    st.session_state[
                        "wfs_sleeper_discovered_leagues"
                    ] = valid_leagues

                    if valid_leagues:
                        count = len(valid_leagues)
                        suffix = "" if count == 1 else "s"

                        st.success(
                            f"Found {count} Sleeper league"
                            f"{suffix} for {season}."
                        )
                    else:
                        st.info(
                            f"No {season} NFL leagues with an "
                            "owned roster were found for that "
                            "Sleeper user."
                        )

            except Exception:
                st.error(
                    "Sleeper is temporarily unavailable. "
                    "Please try again shortly."
                )

    discovered = st.session_state.get(
        "wfs_sleeper_discovered_leagues",
        [],
    )

    if discovered:
        options = {
            (
                f"{item['league_name']} • "
                f"Roster {item['roster_id']}"
            ): item
            for item in discovered
        }

        selected_label = st.selectbox(
            "Choose a Sleeper league",
            list(options),
            key="wfs_sleeper_discovered_choice",
        )

        selected = options[selected_label]

        if st.button(
            "Add Sleeper League",
            key="wfs_sleeper_add_league",
            width="stretch",
        ):
            existing = {
                str(row["league_id"])
                for row in _wfs_sleeper_connections(
                    user_sub,
                    season,
                )
            }

            if selected["league_id"] in existing:
                st.info(
                    "That Sleeper league is already saved "
                    "to your WFS account."
                )
            else:
                _wfs_sleeper_upsert_connection(
                    user_sub=user_sub,
                    season=season,
                    league_id=selected["league_id"],
                    league_name=selected["league_name"],
                    team_id=selected["roster_id"],
                    team_name=selected["team_name"],
                )

                st.success(
                    f"{selected['league_name']} added to "
                    "your WFS account."
                )

                st.rerun()

    st.markdown("### My Sleeper Teams")

    connections = _wfs_sleeper_connections(
        user_sub,
        season,
    )

    if not connections:
        st.info(
            f"You have no Sleeper leagues saved for {season}. "
            "Enter your Sleeper username above to connect one."
        )
        return

    for saved in connections:
        league_name = (
            saved.get("league_name")
            or f"Sleeper League {saved['league_id']}"
        )

        team_name = (
            saved.get("team_name")
            or "My Sleeper Team"
        )

        c1, c2 = st.columns([4, 1])

        with c1:
            st.markdown(f"**{league_name}**")
            st.caption(
                f"{team_name} • "
                f"League ID {saved['league_id']} • "
                f"Roster {saved.get('team_id')}"
            )

        with c2:
            if st.button(
                "Remove",
                key=(
                    "wfs_remove_sleeper_"
                    f"{season}_{saved['league_id']}"
                ),
                width="stretch",
            ):
                _wfs_sleeper_delete_connection(
                    user_sub,
                    season,
                    saved["league_id"],
                )
                st.rerun()

        render_sleeper_league_views(
            saved=saved,
            season=season,
            week=int(schedule_context.planning_week),
        )


def render_fantasy_league_hub():
    """
    WFS multi-provider Fantasy League Hub.

    Provider implementations remain isolated so adding Sleeper does not
    modify the validated ESPN Fantasy implementation.
    """
    st.markdown(
        """
        <div style="
            background:linear-gradient(
                110deg,
                #10233d 0%,
                #173b68 58%,
                #0b73f6 100%
            );
            border-radius:18px;
            padding:1.05rem 1.15rem;
            margin:0 0 .9rem;
            color:white;
            box-shadow:0 8px 24px rgba(15,55,105,.14);
        ">
          <div style="
              font-size:.68rem;
              font-weight:900;
              letter-spacing:.10em;
              color:#9fd0ff;
          ">
            WFS FANTASY FOOTBALL
          </div>

          <div style="
              font-size:1.55rem;
              font-weight:950;
              margin-top:.18rem;
          ">
            Fantasy League Hub
          </div>

          <div style="
              font-size:.84rem;
              color:#d8eaff;
              margin-top:.3rem;
              line-height:1.45;
          ">
            Connect your fantasy leagues, follow your teams and bring
            WFS football intelligence into your season.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    provider = st.radio(
        "Fantasy provider",
        ["ESPN", "Sleeper"],
        horizontal=True,
        key="wfs_fantasy_provider",
    )

    if provider == "Sleeper":
        render_sleeper_league_hub()
    else:
        render_espn_league_hub()


def render_admin_diagnostics(pool: pd.DataFrame, selected_pool: pd.DataFrame) -> None:
    st.subheader("Admin diagnostics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SQLite rows", f"{len(pool):,}")
    c2.metric("Slate eligible", f"{len(selected_pool):,}")
    c3.metric(
        "Missing salary",
        int(selected_pool["salary_solver"].isna().sum())
        if "salary_solver" in selected_pool else 0,
    )
    c4.metric(
        "Missing projection",
        int(selected_pool["projection_solver"].isna().sum())
        if "projection_solver" in selected_pool else 0,
    )

    pos = (
        selected_pool.groupby("solver_position")
        .size()
        .rename("eligible_players")
        .reset_index()
        .sort_values("solver_position")
    )
    st.dataframe(pos, width="stretch", hide_index=True)

    diagnostic_cols = [
        c for c in [
            "player_solver",
            "team_solver",
            "solver_position",
            "salary_solver",
            "projection_solver",
            "game_solver",
        ]
        if c in selected_pool.columns
    ]
    with st.expander("Eligible solver pool"):
        st.dataframe(
            selected_pool[diagnostic_cols],
            width="stretch",
            hide_index=True,
        )


st.markdown("""

<style>
/* WFS download buttons — force readable high-contrast text */
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] button {
    color: #ffffff !important;
}

div[data-testid="stDownloadButton"] > button *,
div[data-testid="stDownloadButton"] button * {
    color: #ffffff !important;
    opacity: 1 !important;
}

div[data-testid="stDownloadButton"] > button:disabled,
div[data-testid="stDownloadButton"] button:disabled {
    color: #ffffff !important;
    opacity: 0.75 !important;
}

div[data-testid="stDownloadButton"] > button:disabled *,
div[data-testid="stDownloadButton"] button:disabled * {
    color: #ffffff !important;
    opacity: 1 !important;
}
</style>

""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Wynners Fantasy Spot — NFL GPP",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# WFS account gate: Google OIDC is the identity authority for all user-owned data.
if not getattr(st.user, "is_logged_in", False):
    st.markdown("## 🔐 Wynners Fantasy Spot")
    st.write(
        "Sign in with Google to use WFS and securely save your fantasy-team connections."
    )
    if st.button("Continue with Google", type="primary", width="stretch"):
        st.login()
    st.stop()

WFS_USER = _wfs_current_user()
if not WFS_USER:
    st.error("Google sign-in succeeded, but WFS did not receive a stable account identifier.")
    if st.button("Log out", width="stretch"):
        st.logout()
    st.stop()

_wfs_accounts_init()
_wfs_upsert_user(WFS_USER)

st.markdown(
    """
    <style>
    :root {
        --wfs-navy: #0f172a;
        --wfs-blue: #1677ff;
        --wfs-blue-2: #30a7ff;
        --wfs-green: #18b66a;
        --wfs-bg: #f4f7fb;
        --wfs-panel: #ffffff;
        --wfs-border: #dbe4ef;
        --wfs-text: #172033;
        --wfs-muted: #66758a;
    }

    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(48,167,255,.09), transparent 28rem),
            linear-gradient(180deg, #f8fbff 0%, #f3f6fa 100%);
        color: var(--wfs-text);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.95);
        border-bottom: 1px solid #e6edf5;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12223b 0%, #0f1c31 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] * {
        color: #eef6ff;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #c5d3e3 !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: #172033 !important;
        background: #ffffff !important;
    }

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 3rem;
        max-width: 1540px;
    }

    .qt-hero {
        padding: 1rem 1.25rem;
        border: 1px solid var(--wfs-border);
        border-radius: 20px;
        background:
            linear-gradient(135deg, rgba(255,255,255,.98), rgba(242,249,255,.98));
        box-shadow: 0 16px 44px rgba(24, 46, 78, .08);
        margin-bottom: 1rem;
    }

    .qt-brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .qt-logo-wrap {
        display:flex;
        align-items:center;
        gap:1rem;
        min-width:0;
    }

    .qt-logo {
        width: 290px;
        max-width: 44vw;
        height: auto;
    }

    .qt-subtitle {
        color: var(--wfs-muted);
        font-size: .92rem;
        margin-top: .15rem;
    }

    .qt-badge {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .5rem .8rem;
        border-radius: 999px;
        border: 1px solid rgba(24,182,106,.28);
        background: rgba(24,182,106,.08);
        color: #12824c;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .qt-section {
        font-size: 1rem;
        font-weight: 850;
        color: #172033;
        letter-spacing: -.01em;
        margin: .85rem 0 .45rem 0;
    }

    .qt-card {
        border: 1px solid var(--wfs-border);
        border-radius: 16px;
        background: #ffffff;
        padding: .95rem 1rem;
        box-shadow: 0 8px 24px rgba(24,46,78,.045);
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--wfs-border);
        background: #ffffff;
        padding: .85rem 1rem;
        border-radius: 14px;
        box-shadow: 0 8px 22px rgba(24,46,78,.045);
    }

    div[data-testid="stMetricLabel"] {
        color: #708096;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid var(--wfs-border);
        border-radius: 14px;
        padding: .15rem;
    }

    .stButton > button[kind="primary"] {
        min-height: 3rem;
        border-radius: 12px;
        font-weight: 850;
        letter-spacing: .01em;
        background: linear-gradient(90deg, var(--wfs-blue), var(--wfs-blue-2));
        border: none;
        box-shadow: 0 10px 26px rgba(22,119,255,.18);
    }

    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 750;
    }

    div[data-baseweb="tab-list"] {
        gap: .25rem;
        border-bottom: 1px solid var(--wfs-border);
    }

    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--wfs-border);
        border-radius: 14px;
        overflow: hidden;
        background:#ffffff;
    }

    .qt-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--wfs-border);
        color: #7f8da1;
        font-size: .78rem;
        text-align: center;
    }
    
    /* Mobile readability: Streamlit light alerts + dark form controls */
    /* Streamlit expanders use a dark summary bar in the WFS theme. Force the
       expander title/chevron to remain visible on mobile and desktop. */
    div[data-testid="stExpander"] details summary {
        background: #1f2937 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
    }

    div[data-testid="stExpander"] details summary p {
        font-weight: 700 !important;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] li,
    div[data-testid="stAlert"] span {
        color: #172033 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextInput"] label span {
        color: #172033 !important;
        opacity: 1 !important;
        font-weight: 750 !important;
    }

    /* QA D1 / WFS_WIDGET_LABEL_CONTRAST_V1 */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSelectbox"] label span,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stMultiSelect"] label p,
    div[data-testid="stMultiSelect"] label span,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stNumberInput"] label span,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label p,
    div[data-testid="stSlider"] label span,
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] label p,
    div[data-testid="stToggle"] label span {
        color:#172033 !important;
        opacity:1 !important;
    }
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {
        font-weight:700 !important;
    }

    /* Sidebar nav is also an stRadio widget; the light-background contrast
       fix above (WFS_WIDGET_LABEL_CONTRAST_V1) otherwise wins on specificity
       and forces dark-navy text onto the dark-navy sidebar. */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label span {
        color:#eef6ff !important;
    }

    /* Keep typed/private values readable inside the existing dark inputs. */
    div[data-testid="stTextInput"] input {
        color: #f8fafc !important;
        caret-color: #f8fafc !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #aeb8c7 !important;
        opacity: 1 !important;
    }

    /* WFS mockup-target layout pass */
    .stApp {
        background: #f5f8fc !important;
    }

    .block-container {
        max-width: 1600px !important;
        padding-top: .65rem !important;
    }

    .wfs-stadium {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        min-height: 155px;
        margin-bottom: 1rem;
        border: 1px solid #d7e2ee;
        background:
          radial-gradient(circle at 72% -15%, rgba(255,255,255,.78) 0 2%, transparent 3%),
          radial-gradient(circle at 82% -12%, rgba(255,255,255,.72) 0 2%, transparent 3%),
          linear-gradient(180deg, #0b2746 0%, #123c65 52%, #e9f3fb 53%, #ffffff 100%);
        box-shadow: 0 14px 36px rgba(28,56,90,.10);
    }

    .wfs-stadium:after {
        content:"";
        position:absolute;
        left:0; right:0; bottom:0;
        height:48px;
        background:
          repeating-linear-gradient(90deg, rgba(22,119,255,.05) 0 80px, transparent 80px 160px),
          linear-gradient(180deg, rgba(255,255,255,.2), rgba(255,255,255,.96));
    }

    .wfs-header-content {
        position:relative;
        z-index:2;
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:1rem;
        padding:1.05rem 1.2rem;
    }

    .wfs-logo {
        width: 335px;
        max-width: 50vw;
        filter: drop-shadow(0 5px 12px rgba(0,0,0,.12));
    }

    .wfs-tagline {
        margin-left: .4rem;
        margin-top: -.25rem;
        color:#d7e9fa;
        font-size:.78rem;
        font-weight:700;
        letter-spacing:.13em;
        text-transform:uppercase;
    }

    .wfs-engine {
        margin-top:.15rem;
        padding:.7rem .9rem;
        border-radius:14px;
        background:rgba(8,44,34,.88);
        border:1px solid rgba(49,216,126,.42);
        color:#78f2aa;
        font-size:.73rem;
        font-weight:850;
        letter-spacing:.05em;
        text-transform:uppercase;
        box-shadow:0 10px 24px rgba(0,0,0,.15);
    }

    .wfs-steps {
        display:grid;
        grid-template-columns:repeat(5,1fr);
        gap:.7rem;
        margin:.25rem 0 1.05rem;
    }

    .wfs-step {
        min-height:96px;
        background:#fff;
        border:1px solid #dbe4ef;
        border-radius:14px;
        padding:.85rem .9rem;
        box-shadow:0 7px 20px rgba(28,56,90,.045);
    }

    .wfs-step.active {
        border:1.5px solid #1677ff;
        background:linear-gradient(180deg,#f7fbff,#eef6ff);
    }

    .wfs-step-num {
        width:29px;height:29px;border-radius:50%;
        display:inline-flex;align-items:center;justify-content:center;
        background:#52647a;color:white;font-weight:850;font-size:.78rem;
        margin-right:.45rem;
    }

    .wfs-step.active .wfs-step-num { background:#1677ff; }

    .wfs-step-title {
        color:#172033;font-size:.83rem;font-weight:850;
    }

    .wfs-step-copy {
        color:#748399;font-size:.70rem;line-height:1.35;
        margin:.45rem 0 0 2.25rem;
    }

    .qt-section {
        padding:.75rem .9rem .1rem !important;
        margin-top:.65rem !important;
        background:#fff;
        border:1px solid #dbe4ef;
        border-bottom:0;
        border-radius:14px 14px 0 0;
        color:#172033 !important;
        font-size:1.02rem !important;
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"] {
        box-shadow:0 7px 20px rgba(28,56,90,.045) !important;
    }

    div[data-testid="stMetric"] {
        min-height:105px;
    }

    div[data-testid="stMetricValue"] {
        color:#0c4f9f !important;
        font-weight:800 !important;
    }

    .stButton > button[kind="primary"] {
        background:linear-gradient(90deg,#0876f9,#1497ff) !important;
        min-height:3.25rem !important;
        font-size:.95rem !important;
    }

    /* WFS mobile contrast: secondary action buttons use the same dark surface
       as inputs/expanders, with explicitly bright text. */
    .stButton > button:not([kind="primary"]) {
        background:#202838 !important;
        border:1px solid #2f3a4d !important;
        color:#f8fafc !important;
        font-weight:750 !important;
    }

    .stButton > button:not([kind="primary"]) p,
    .stButton > button:not([kind="primary"]) span,
    .stButton > button:not([kind="primary"]) svg {
        color:#f8fafc !important;
        fill:#f8fafc !important;
        opacity:1 !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background:#2a3447 !important;
        border-color:#3a475d !important;
    }

    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#10243d 0%,#0b1a2d 100%) !important;
    }

    .wfs-sidebrand {
        text-align:center;
        padding:.25rem 0 1rem;
        border-bottom:1px solid rgba(255,255,255,.10);
        margin-bottom:.8rem;
    }

    .wfs-sidebrand img { width:205px; max-width:100%; }
    .wfs-sidebrand .small {
        color:#8299b3;font-size:.65rem;font-weight:750;
        letter-spacing:.11em;text-transform:uppercase;margin-top:-.3rem;
    }

    @media (max-width: 900px) {
        .wfs-steps { grid-template-columns:1fr; }
        .wfs-header-content { flex-direction:column; }
        .wfs-logo { max-width:90vw; }
    }

    
    /* WFS screenshot refinement pass */
    .wfs-stadium {
        min-height: 112px !important;
        background: linear-gradient(100deg, #10395f 0%, #174d79 62%, #0d3154 100%) !important;
        border-radius: 16px !important;
    }

    .wfs-stadium:after {
        height: 0 !important;
        display: none !important;
    }

    .wfs-header-content {
        min-height: 112px;
        align-items: center !important;
        padding: .55rem 1.15rem !important;
    }

    .wfs-logo {
        width: 300px !important;
        max-height: 96px !important;
        object-fit: contain !important;
        object-position: left center !important;
    }

    .wfs-tagline {
        color: #d9eaff !important;
        margin-top: -.5rem !important;
        margin-left: 8.7rem !important;
        font-size: .66rem !important;
        letter-spacing: .10em !important;
    }

    .wfs-engine {
        flex-shrink: 0;
        background: rgba(5, 47, 39, .94) !important;
        color: #72f3a7 !important;
    }

    .wfs-steps {
        gap: .6rem !important;
        margin-bottom: .85rem !important;
    }

    .wfs-step {
        min-height: 78px !important;
        padding: .68rem .75rem !important;
    }

    .wfs-step-copy {
        color: #66758a !important;
        margin-top: .3rem !important;
        font-size: .67rem !important;
    }

    .qt-section {
        padding: .58rem .78rem .18rem !important;
        margin-top: .5rem !important;
        font-size: .94rem !important;
        background: #ffffff !important;
    }

    .qt-card {
        color: #334155 !important;
    }

    .qt-card div {
        color: inherit;
    }

    div[data-testid="stMetric"] {
        min-height: 92px !important;
        padding: .72rem .85rem !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetricLabel"] div {
        color: #334155 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0b4f9c !important;
        font-family: inherit !important;
        font-size: 1.72rem !important;
        line-height: 1.05 !important;
        letter-spacing: -.025em !important;
    }

    div[data-testid="stMetricValue"] > div {
        font-family: inherit !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #66758a !important;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {
        color: #66758a;
    }

    [data-testid="stSidebar"] .wfs-sidebrand {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] .wfs-sidebrand img {
        width: 180px !important;
        height: 92px !important;
        object-fit: contain !important;
    }

    [data-testid="stSidebar"] .wfs-sidebrand .small {
        margin-top: -.55rem !important;
        color: #8ca6c2 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.08) !important;
    }

    /* WFS download buttons: readable high-contrast treatment. */
    [data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid rgba(15, 23, 42, .20) !important;
        font-weight: 800 !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border-color: rgba(15, 23, 42, .35) !important;
    }

    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: #0f172a !important;
    }

    /* Prevent Streamlit's white top chrome from visually overpowering the app. */
    [data-testid="stHeader"] {
        height: 2.2rem !important;
        background: #ffffff !important;
    }

    .block-container {
        padding-top: .35rem !important;
    }

    @media (max-width: 1100px) {
        .wfs-tagline { margin-left: 0 !important; }
        .wfs-logo { width: 270px !important; }
    }

    
    /* ============================================================
       WFS GLOBAL NAVIGATION CONTRAST
       Forced inactive-tab + sidebar control visibility
       ============================================================ */

    /* ---------- ALL TABS ---------- */

    /* Base tab button */
    [role="tab"],
    button[role="tab"],
    button[data-baseweb="tab"] {
        opacity: 1 !important;
        font-weight: 800 !important;
    }

    /* INACTIVE tabs — force dark readable text */
    [role="tab"][aria-selected="false"],
    button[role="tab"][aria-selected="false"],
    button[data-baseweb="tab"][aria-selected="false"] {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        opacity: 1 !important;
    }

    [role="tab"][aria-selected="false"] *,
    [role="tab"][aria-selected="false"] p,
    [role="tab"][aria-selected="false"] span,
    [role="tab"][aria-selected="false"] div,
    button[role="tab"][aria-selected="false"] *,
    button[data-baseweb="tab"][aria-selected="false"] * {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* SELECTED tabs — WFS coral */
    [role="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e84f5f !important;
        -webkit-text-fill-color: #e84f5f !important;
        opacity: 1 !important;
        font-weight: 850 !important;
    }

    [role="tab"][aria-selected="true"] *,
    [role="tab"][aria-selected="true"] p,
    [role="tab"][aria-selected="true"] span,
    [role="tab"][aria-selected="true"] div {
        color: #e84f5f !important;
        -webkit-text-fill-color: #e84f5f !important;
        opacity: 1 !important;
    }

    /* ---------- SIDEBAR OPEN/CLOSE ---------- */

    /* Open sidebar: collapse control */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background: #10243d !important;
        border: 1px solid #31506f !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        opacity: 1 !important;
        box-shadow: 0 3px 10px rgba(15,23,42,.18) !important;
    }

    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    /* Closed sidebar: known Streamlit variants */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button {
        background: #10243d !important;
        border: 1px solid #31506f !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        opacity: 1 !important;
        box-shadow: 0 3px 10px rgba(15,23,42,.18) !important;
    }

    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    /*
       Current Streamlit mobile header can place the sidebar opener
       inside the header action area rather than collapsedControl.
    */
    [data-testid="stHeader"] button {
        opacity: 1 !important;
    }

    [data-testid="stHeader"] button svg {
        color: #10243d !important;
        fill: #10243d !important;
        stroke: #10243d !important;
        opacity: 1 !important;
    }

    /* First header action is typically the sidebar/navigation opener. */
    [data-testid="stHeader"] button:first-of-type {
        background: #eaf2fb !important;
        border: 1px solid #b8cbe0 !important;
        border-radius: 10px !important;
        color: #10243d !important;
    }

        /* ============================================================
       WFS_MOBILE_SPACING_FIX_V2
       Mobile layout only.
       Theme colors are controlled by Streamlit.
       ============================================================ */

    @media (max-width: 900px) {
        [data-testid="stHeader"] {
            min-height: 3.4rem !important;
            height: 3.4rem !important;
        }

        .block-container {
            padding-top: 4.15rem !important;
        }

        .wfs-stadium {
            margin-top: 0 !important;
        }
    }

</style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="wfs-stadium">
      <div class="wfs-header-content">
        <div>
          <img class="wfs-logo" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODUgMTUwIj4KPGRlZnM+CiA8bGluZWFyR3JhZGllbnQgaWQ9Im4iIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjMTAyMzNkIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMDcxNDI2Ii8+PC9saW5lYXJHcmFkaWVudD4KIDxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjAiPjxzdG9wIHN0b3AtY29sb3I9IiMwYjczZjYiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMzMmE2ZmYiLz48L2xpbmVhckdyYWRpZW50Pgo8L2RlZnM+CjxwYXRoIGQ9Ik03MCAzNCA4NiAxMCAxMDIgMzEgMTIzIDE2IDEzMSA0Nkg0MmwxMC0zMHoiIGZpbGw9IiNmNWJkNDUiIHN0cm9rZT0iIzBiMTYyOCIgc3Ryb2tlLXdpZHRoPSI2Ii8+CjxlbGxpcHNlIGN4PSI4NyIgY3k9IjYyIiByeD0iNTMiIHJ5PSIyNSIgdHJhbnNmb3JtPSJyb3RhdGUoLTEyIDg3IDYyKSIgZmlsbD0iIzhiNGEyNiIgc3Ryb2tlPSIjMGIxNjI4IiBzdHJva2Utd2lkdGg9IjYiLz4KPHBhdGggZD0iTTYwIDYxYzE2LTEwIDM2LTEzIDU0LThNODEgNTBsMyAxOE05MSA0OGwzIDE4TTEwMSA0OWwzIDE2IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8cGF0aCBkPSJNMjMgODFoMTI2bDggNDItMjMgMTlINDhsLTM0LTI1eiIgZmlsbD0idXJsKCNuKSIgc3Ryb2tlPSIjMDcxNDI2IiBzdHJva2Utd2lkdGg9IjYiLz4KPHRleHQgeD0iODUiIHk9IjExNiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI5MDAiIGZvbnQtc2l6ZT0iNDIiIGZpbGw9IndoaXRlIj5XRlM8L3RleHQ+Cjx0ZXh0IHg9IjE2OSIgeT0iNjIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iOTAwIiBmb250LXNpemU9IjI4IiBmaWxsPSIjZWVmMmY4Ij5XWU5ORVJTPC90ZXh0Pgo8dGV4dCB4PSIxNjkiIHk9Ijk0IiBmb250LWZhbWlseT0iQXJpYWwsc2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9IjkwMCIgZm9udC1zaXplPSIyNiIgZmlsbD0idXJsKCNiKSI+RkFOVEFTWSBTUE9UPC90ZXh0Pgo8dGV4dCB4PSIxNzAiIHk9IjExOCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIGZvbnQtc2l6ZT0iMTAiIGxldHRlci1zcGFjaW5nPSIxLjgiIGZpbGw9IiM2YjdiOTAiPkRBVEEgRFJJVkVOIOKAoiBHUFAgRk9DVVNFRDwvdGV4dD4KPC9zdmc+" />
          <div class="wfs-tagline">NFL FanDuel GPP • Play Smarter • Build Better</div>
        </div>
        <div class="wfs-engine">🏈 NFL Fantasy Intelligence<br>
          <span style="color:#b7c9d8;font-size:.62rem;">FanDuel &nbsp;•&nbsp; Forecasts &nbsp;•&nbsp; Fantasy Football</span>
        </div>
      </div>
    </div>

    """,
    unsafe_allow_html=True,
)

try:
    pool = load_solver_table(solver_table_freshness_token())
    slates = available_slates(pool)
except Exception:
    pool = None
    slates = []


with st.sidebar:
    with st.container(border=True):
        st.markdown(f"**{WFS_USER.get('name') or 'WFS User'}**")
        if WFS_USER.get("email"):
            st.caption(WFS_USER["email"])
        if st.button("Log out", width="stretch", key="wfs_sidebar_logout"):
            st.logout()

    admin_emails = {
        str(email).strip().lower()
        for email in st.secrets.get("admin", {}).get("emails", [])
    }
    current_email = str(st.user.get("email") or "").strip().lower()
    is_admin = current_email in admin_emails

    nav_options = [
        "🏠 Home",
        "🏗 Build Lineups",
        "🔮 Forecast Center",
        "🤖 WFS Analyst",
        "🔴 NFL Live",
        "🏈 NFL Data Center",
        "🏆 Fantasy League Hub",
    ]

    if is_admin:
        nav_options.append("🔒 Admin")

    nav_request = st.session_state.pop("wfs_nav_request", None)

    if nav_request in nav_options:
        st.session_state["wfs_main_navigation"] = nav_request
    elif "wfs_main_navigation" not in st.session_state:
        persisted_page = _wfs_get_last_navigation(WFS_USER.get("sub"))
        if persisted_page in nav_options:
            st.session_state["wfs_main_navigation"] = persisted_page

    # WFS_SIDEBAR_HEADER_BY_PAGE_V1
    # The sidebar header used to always read "BUILD LINEUPS" no matter
    # which page was selected. Reflect the page about to be shown instead.
    _wfs_sidebar_headers = {
        "🏠 Home": ("HOME", "Your fantasy dashboard"),
        "🏗 Build Lineups": ("BUILD LINEUPS", "Configure portfolio rules"),
        "🔮 Forecast Center": ("FORECAST CENTER", "Pregame score projections"),
        "🤖 WFS Analyst": ("WFS ANALYST", "Game outlooks & takeaways"),
        "🔴 NFL Live": ("NFL LIVE", "Live scores & win probability"),
        "🏈 NFL Data Center": ("NFL DATA CENTER", "Schedule, scores & box scores"),
        "🏆 Fantasy League Hub": ("FANTASY LEAGUE HUB", "ESPN & Sleeper leagues"),
        "🔒 Admin": ("ADMIN", "Accounts, feedback & health"),
    }
    _wfs_sidebar_current_page = st.session_state.get(
        "wfs_main_navigation", nav_options[0]
    )
    _wfs_sidebar_title, _wfs_sidebar_subtitle = _wfs_sidebar_headers.get(
        _wfs_sidebar_current_page,
        ("BUILD LINEUPS", "Configure portfolio rules"),
    )

    st.markdown(
        f"""
        <div class="wfs-sidebrand">
          <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODUgMTUwIj4KPGRlZnM+CiA8bGluZWFyR3JhZGllbnQgaWQ9Im4iIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjMTAyMzNkIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMDcxNDI2Ii8+PC9saW5lYXJHcmFkaWVudD4KIDxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjAiPjxzdG9wIHN0b3AtY29sb3I9IiMwYjczZjYiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMzMmE2ZmYiLz48L2xpbmVhckdyYWRpZW50Pgo8L2RlZnM+CjxwYXRoIGQ9Ik03MCAzNCA4NiAxMCAxMDIgMzEgMTIzIDE2IDEzMSA0Nkg0MmwxMC0zMHoiIGZpbGw9IiNmNWJkNDUiIHN0cm9rZT0iIzBiMTYyOCIgc3Ryb2tlLXdpZHRoPSI2Ii8+CjxlbGxpcHNlIGN4PSI4NyIgY3k9IjYyIiByeD0iNTMiIHJ5PSIyNSIgdHJhbnNmb3JtPSJyb3RhdGUoLTEyIDg3IDYyKSIgZmlsbD0iIzhiNGEyNiIgc3Ryb2tlPSIjMGIxNjI4IiBzdHJva2Utd2lkdGg9IjYiLz4KPHBhdGggZD0iTTYwIDYxYzE2LTEwIDM2LTEzIDU0LThNODEgNTBsMyAxOE05MSA0OGwzIDE4TTEwMSA0OWwzIDE2IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8cGF0aCBkPSJNMjMgODFoMTI2bDggNDItMjMgMTlINDhsLTM0LTI1eiIgZmlsbD0idXJsKCNuKSIgc3Ryb2tlPSIjMDcxNDI2IiBzdHJva2Utd2lkdGg9IjYiLz4KPHRleHQgeD0iODUiIHk9IjExNiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI5MDAiIGZvbnQtc2l6ZT0iNDIiIGZpbGw9IndoaXRlIj5XRlM8L3RleHQ+Cjx0ZXh0IHg9IjE2OSIgeT0iNjIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iOTAwIiBmb250LXNpemU9IjI4IiBmaWxsPSIjZWVmMmY4Ij5XWU5ORVJTPC90ZXh0Pgo8dGV4dCB4PSIxNjkiIHk9Ijk0IiBmb250LWZhbWlseT0iQXJpYWwsc2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9IjkwMCIgZm9udC1zaXplPSIyNiIgZmlsbD0idXJsKCNiKSI+RkFOVEFTWSBTUE9UPC90ZXh0Pgo8dGV4dCB4PSIxNzAiIHk9IjExOCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIGZvbnQtc2l6ZT0iMTAiIGxldHRlci1zcGFjaW5nPSIxLjgiIGZpbGw9IiM2YjdiOTAiPkRBVEEgRFJJVkVOIOKAoiBHUFAgRk9DVVNFRDwvdGV4dD4KPC9zdmc+" />
          <div class="small">Data Driven • GPP Focused</div>
        </div>
        <div style="padding:.1rem 0 .65rem 0;">
          <div style="font-weight:850;font-size:.82rem;color:#ffffff;letter-spacing:.06em;">{_wfs_sidebar_title}</div>
          <div style="font-size:.72rem;color:#8299b3;margin-top:.18rem;">{_wfs_sidebar_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        nav_options,
        index=0,
        key="wfs_main_navigation",
    )

    if page in nav_options:
        _wfs_set_last_navigation(WFS_USER.get("sub"), page)

    mode = "Public"

    if page == "🔒 Admin":
        if not is_admin:
            st.error("Admin access is not authorized for this account.")
            st.stop()

        st.title("🔒 WFS Admin")
        st.caption("Private administration, account activity, and user feedback.")

        users_tab, feedback_tab, health_tab = st.tabs(
            ["👥 Users", "💬 User Feedback", "🩺 Health Monitor"]
        )

        with health_tab:
            import json as _hm_json
            from pathlib import Path as _HMPath

            st.markdown("### 🩺 NFL APP Health Monitor")
            st.caption(
                "Read-only operational status. "
                "This panel cannot execute repairs or modify monitor state."
            )

            _hm_path = _HMPath("var/health_monitor/incident_state.json")

            if not _hm_path.exists():
                st.warning("Health Monitor state has not been published yet.")
            else:
                try:
                    _hm_state = _hm_json.loads(_hm_path.read_text())
                except Exception:
                    _hm_state = None
                    st.error("Health Monitor state could not be read.")

                if isinstance(_hm_state, dict):
                    _hm_raw = _hm_state.get("incidents", {})

                    if isinstance(_hm_raw, dict):
                        _hm_incidents = [
                            dict(v, incident_id=v.get("incident_id", k))
                            if isinstance(v, dict)
                            else {"incident_id": k}
                            for k, v in _hm_raw.items()
                        ]
                    elif isinstance(_hm_raw, list):
                        _hm_incidents = [
                            x for x in _hm_raw if isinstance(x, dict)
                        ]
                    else:
                        _hm_incidents = []

                    _hm_active = []
                    for _row in _hm_incidents:
                        _state = str(
                            _row.get("state")
                            or _row.get("incident_state")
                            or ""
                        ).upper()

                        _closed = (
                            _state in {"CLOSED", "RESOLVED", "RECOVERED"}
                            or bool(_row.get("closed_at_utc"))
                            or bool(_row.get("resolved_at_utc"))
                            or bool(_row.get("recovered_at_utc"))
                        )

                        if not _closed:
                            _hm_active.append(_row)

                    def _hm_status(row):
                        return str(
                            row.get("severity")
                            or row.get("status")
                            or "UNKNOWN"
                        ).upper()

                    _hm_order = {
                        "OWNER_ACTION_REQUIRED": 4,
                        "CRITICAL": 3,
                        "WARNING": 2,
                        "HEALTHY": 1,
                    }

                    if _hm_active:
                        _hm_overall = max(
                            (_hm_status(x) for x in _hm_active),
                            key=lambda x: _hm_order.get(x, 0),
                        )
                    else:
                        _hm_overall = "HEALTHY"

                    _hm_owner = sum(
                        _hm_status(x) == "OWNER_ACTION_REQUIRED"
                        for x in _hm_active
                    )
                    _hm_critical = sum(
                        _hm_status(x) == "CRITICAL"
                        for x in _hm_active
                    )
                    _hm_warning = sum(
                        _hm_status(x) == "WARNING"
                        for x in _hm_active
                    )

                    _h1, _h2, _h3, _h4 = st.columns(4)
                    _hm_display_status = {
                        "OWNER_ACTION_REQUIRED": "ACTION REQUIRED",
                        "CRITICAL": "CRITICAL",
                        "WARNING": "WARNING",
                        "HEALTHY": "HEALTHY",
                    }.get(_hm_overall, _hm_overall)

                    _h1.metric("System Status", _hm_display_status)
                    _h2.metric("Active Incidents", len(_hm_active))
                    _h3.metric("Owner Action", _hm_owner)
                    _h4.metric(
                        "Critical / Warning",
                        _hm_critical + _hm_warning,
                    )

                    _hm_reason_labels = {
                        "CURRENT_CONSUMER_FORECAST_VALID": "Forecast Evidence Aging",
                        "STARTER_EVIDENCE_VALID": "Starter Evidence Aging",
                    }

                    st.markdown("#### Active Incidents")

                    if not _hm_active:
                        st.success("No active Health Monitor incidents.")
                    else:
                        _hm_rows = []
                        for _row in sorted(
                            _hm_active,
                            key=lambda x: -_hm_order.get(
                                _hm_status(x), 0
                            ),
                        ):
                            _hm_rows.append(
                                {
                                    "Status": _hm_status(_row),
                                    "Check": (
                                        _row.get("check_id")
                                        or _row.get("check")
                                        or "—"
                                    ),
                                    "Scope": _row.get("scope") or "—",
                                    "Reason": (
                                        _row.get("reason_code")
                                        or _row.get("reason")
                                        or "—"
                                    ),
                                    "First Seen": (
                                        _row.get("first_seen_utc") or "—"
                                    ),
                                    "Last Seen": (
                                        _row.get("last_seen_utc") or "—"
                                    ),
                                }
                            )

                        for _item in _hm_rows:
                            _display_status = {
                                "OWNER_ACTION_REQUIRED": "ACTION REQUIRED",
                                "CRITICAL": "CRITICAL",
                                "WARNING": "WARNING",
                                "HEALTHY": "HEALTHY",
                            }.get(
                                _item["Status"],
                                _item["Status"],
                            )

                            _reason_raw = str(_item["Reason"])
                            _reason_display = _hm_reason_labels.get(
                                _reason_raw,
                                _reason_raw.replace("_", " ").title(),
                            )

                            _check_display = str(
                                _item["Check"]
                            ).replace("_", " ").title()

                            with st.container(border=True):
                                st.markdown(
                                    f"**{_display_status} — "
                                    f"{_item['Scope']}**"
                                )
                                st.markdown(f"**{_check_display}**")
                                st.write(_reason_display)

                                def _hm_time(value):
                                    try:
                                        _ts = pd.to_datetime(
                                            value,
                                            utc=True,
                                        )
                                        return _ts.strftime(
                                            "%b %d • %I:%M %p UTC"
                                        ).replace(" 0", " ")
                                    except Exception:
                                        return str(value)

                                st.caption(
                                    "First seen: "
                                    f"{_hm_time(_item['First Seen'])}"
                                )
                                st.caption(
                                    "Last seen: "
                                    f"{_hm_time(_item['Last Seen'])}"
                                )

                    if isinstance(_hm_raw, dict):
                        _hm_recovery_source = [
                            v for v in _hm_raw.values()
                            if isinstance(v, dict)
                        ]
                    elif isinstance(_hm_raw, list):
                        _hm_recovery_source = [
                            v for v in _hm_raw
                            if isinstance(v, dict)
                        ]
                    else:
                        _hm_recovery_source = []

                    _hm_recovered = [
                        row for row in _hm_recovery_source
                        if (
                            row.get("resolved_at_utc")
                            or row.get("recovered_at_utc")
                            or row.get("closed_at_utc")
                        )
                    ]

                    def _hm_resolution_time(row):
                        return (
                            row.get("resolved_at_utc")
                            or row.get("recovered_at_utc")
                            or row.get("closed_at_utc")
                            or ""
                        )

                    _hm_recovered.sort(
                        key=lambda row: str(
                            _hm_resolution_time(row)
                        ),
                        reverse=True,
                    )

                    if _hm_recovered:
                        with st.expander(
                            f"✅ Recent Recoveries ({len(_hm_recovered)})",
                            expanded=False,
                        ):
                            for _rec in _hm_recovered[:10]:
                                _rec_scope = str(
                                    _rec.get("scope") or "System"
                                )
                                _rec_check = str(
                                    _rec.get("check_id") or "monitor"
                                ).replace("_", " ").title()
                                _rec_reason_raw = str(
                                    _rec.get("reason_code") or "RECOVERED"
                                )
                                _rec_reason = _hm_reason_labels.get(
                                    _rec_reason_raw,
                                    _rec_reason_raw.replace("_", " ").title(),
                                )

                                try:
                                    _rec_time = pd.to_datetime(
                                        _hm_resolution_time(_rec),
                                        utc=True,
                                    ).strftime(
                                        "%b %d • %I:%M %p UTC"
                                    ).replace(" 0", " ")
                                except Exception:
                                    _rec_time = str(
                                        _hm_resolution_time(_rec)
                                    )

                                with st.container(border=True):
                                    st.markdown(
                                        f"**RECOVERED — {_rec_scope}**"
                                    )
                                    st.write(
                                        f"{_rec_check} — {_rec_reason}"
                                    )
                                    st.caption(
                                        f"Recovered: {_rec_time}"
                                    )

                    st.caption(
                        "Display only • repair execution disabled"
                    )

        with users_tab:
            try:
                with _wfs_accounts_connect() as conn:
                    users_df = pd.read_sql_query(
                        """
                        SELECT
                            display_name,
                            email,
                            created_at,
                            updated_at
                        FROM wfs_users
                        ORDER BY datetime(created_at) DESC
                        """,
                        conn,
                    )

                now_utc = pd.Timestamp.now(tz="UTC")

                if not users_df.empty:
                    created = pd.to_datetime(
                        users_df["created_at"], errors="coerce", utc=True
                    )
                    updated = pd.to_datetime(
                        users_df["updated_at"], errors="coerce", utc=True
                    )

                    registered = len(users_df)
                    active_today = int(
                        (updated >= now_utc - pd.Timedelta(days=1)).sum()
                    )
                    active_week = int(
                        (updated >= now_utc - pd.Timedelta(days=7)).sum()
                    )
                    new_week = int(
                        (created >= now_utc - pd.Timedelta(days=7)).sum()
                    )
                else:
                    registered = active_today = active_week = new_week = 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Registered Users", registered)
                c2.metric("Active 24 Hours", active_today)
                c3.metric("Active 7 Days", active_week)
                c4.metric("New Users 7 Days", new_week)

                st.markdown("### Users")

                if users_df.empty:
                    st.info("No registered users yet.")
                else:
                    users_df["Signed Up"] = pd.to_datetime(
                        users_df["created_at"], errors="coerce", utc=True
                    )
                    users_df["Last Seen"] = pd.to_datetime(
                        users_df["updated_at"], errors="coerce", utc=True
                    )

                    users_df["Account Age"] = (
                        now_utc - users_df["Signed Up"]
                    ).dt.days.map(
                        lambda days: (
                            f"{int(days)} days"
                            if pd.notna(days)
                            else "—"
                        )
                    )

                    users_df = users_df.rename(
                        columns={
                            "display_name": "User",
                            "email": "Email",
                        }
                    )[
                        [
                            "User",
                            "Email",
                            "Signed Up",
                            "Last Seen",
                            "Account Age",
                        ]
                    ]

                    st.dataframe(
                        users_df,
                        width="stretch",
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(
                    f"Could not load Admin user activity: {exc}"
                )

        with feedback_tab:
            feedback_flash = st.session_state.pop(
                "wfs_admin_feedback_flash",
                None,
            )
            if feedback_flash:
                flash_kind, flash_message = feedback_flash
                if flash_kind == "success":
                    st.success(flash_message)
                else:
                    st.warning(flash_message)

            try:
                from wfs_user_feedback import (
                    list_feedback,
                    save_admin_response,
                    close_feedback,
                    email_feedback_response,
                )

                feedback_rows = list_feedback()

                with _wfs_accounts_connect() as conn:
                    identity_rows = conn.execute(
                        """
                        SELECT user_sub, display_name, email
                        FROM wfs_users
                        """
                    ).fetchall()

                identity_map = {
                    str(row[0]): {
                        "name": str(row[1] or "").strip(),
                        "email": str(row[2] or "").strip(),
                    }
                    for row in identity_rows
                }

                new_count = sum(
                    1 for row in feedback_rows
                    if row.get("status") == "NEW"
                )
                replied_count = sum(
                    1 for row in feedback_rows
                    if row.get("status") == "REPLIED"
                )
                closed_count = sum(
                    1 for row in feedback_rows
                    if row.get("status") == "CLOSED"
                )

                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Total Reports", len(feedback_rows))
                f2.metric("New", new_count)
                f3.metric("Replied", replied_count)
                f4.metric("Closed", closed_count)

                status_filter = st.selectbox(
                    "Status",
                    ["All", "NEW", "REPLIED", "CLOSED"],
                    key="wfs_admin_feedback_status",
                )

                visible_rows = [
                    row for row in feedback_rows
                    if status_filter == "All"
                    or row.get("status") == status_filter
                ]

                if not visible_rows:
                    st.info("No feedback reports match this status.")
                else:
                    for row in visible_rows:
                        feedback_id = str(row["feedback_id"])
                        short_id = feedback_id[:8].upper()
                        identity = identity_map.get(
                            str(row.get("user_id") or ""),
                            {},
                        )

                        user_name = (
                            identity.get("name")
                            or "WFS User"
                        )
                        user_email = (
                            identity.get("email")
                            or "Email unavailable"
                        )

                        status = str(
                            row.get("status") or "NEW"
                        ).upper()

                        category = str(
                            row.get("category") or "Other"
                        )

                        created = pd.to_datetime(
                            row.get("created_at_utc"),
                            errors="coerce",
                            utc=True,
                        )

                        created_label = (
                            created.strftime(
                                "%b %d, %Y %I:%M %p UTC"
                            )
                            if pd.notna(created)
                            else "Unknown time"
                        )

                        with st.expander(
                            f"{status} • {category} • "
                            f"{user_name} • {short_id}",
                            expanded=(status == "NEW"),
                        ):
                            st.caption(
                                f"{created_label} • {user_email}"
                            )

                            subject = str(
                                row.get("player_or_team") or ""
                            ).strip()

                            if subject:
                                st.write("Player / Team:", subject)

                            context_parts = []

                            if row.get("page"):
                                context_parts.append(
                                    f"Page: {row['page']}"
                                )

                            if row.get("slate"):
                                context_parts.append(
                                    f"Slate: {row['slate']}"
                                )

                            if row.get("season"):
                                context_parts.append(
                                    f"Season: {row['season']}"
                                )

                            if row.get("week"):
                                context_parts.append(
                                    f"Week: {row['week']}"
                                )

                            if context_parts:
                                st.caption(
                                    " • ".join(context_parts)
                                )

                            st.markdown("**User message**")
                            st.write(row.get("message") or "")

                            existing_response = str(
                                row.get("admin_response") or ""
                            )

                            response = st.text_area(
                                "Your response",
                                value=existing_response,
                                max_chars=4000,
                                key=(
                                    "wfs_admin_feedback_response_"
                                    f"{feedback_id}"
                                ),
                            )

                            b1, b2 = st.columns(2)

                            if b1.button(
                                "Save Reply",
                                key=(
                                    "wfs_admin_feedback_reply_"
                                    f"{feedback_id}"
                                ),
                                width="stretch",
                            ):
                                try:
                                    save_admin_response(
                                        feedback_id,
                                        response,
                                    )
                                except Exception as exc:
                                    st.error(
                                        f"Could not save reply: {exc}"
                                    )
                                else:
                                    # The response is already committed to
                                    # the feedback DB before email delivery.
                                    delivery_ok = False
                                    delivery_detail = (
                                        "User email is unavailable."
                                    )

                                    if (
                                        user_email
                                        and user_email
                                        != "Email unavailable"
                                    ):
                                        try:
                                            (
                                                delivery_ok,
                                                delivery_detail,
                                            ) = email_feedback_response(
                                                feedback_id=feedback_id,
                                                user_email=user_email,
                                                response=response,
                                            )
                                        except Exception as exc:
                                            delivery_detail = (
                                                "Email delivery failed: "
                                                f"{type(exc).__name__}"
                                            )

                                    if delivery_ok:
                                        st.session_state[
                                            "wfs_admin_feedback_flash"
                                        ] = (
                                            "success",
                                            "Reply saved and emailed.",
                                        )
                                    else:
                                        st.session_state[
                                            "wfs_admin_feedback_flash"
                                        ] = (
                                            "warning",
                                            "Reply saved, but email was "
                                            "not delivered. "
                                            f"{delivery_detail}",
                                        )

                                    st.rerun()

                            if status != "CLOSED":
                                if b2.button(
                                    "Close",
                                    key=(
                                        "wfs_admin_feedback_close_"
                                        f"{feedback_id}"
                                    ),
                                    width="stretch",
                                ):
                                    try:
                                        close_feedback(feedback_id)
                                    except Exception as exc:
                                        st.error(
                                            f"Could not close report: {exc}"
                                        )
                                    else:
                                        st.success("Report closed.")
                                        st.rerun()

                            if row.get("responded_at_utc"):
                                st.caption(
                                    "Response saved: "
                                    f"{row['responded_at_utc']}"
                                )

                            if row.get("closed_at_utc"):
                                st.caption(
                                    "Closed: "
                                    f"{row['closed_at_utc']}"
                                )

            except Exception as exc:
                st.error(
                    f"Could not load user feedback: {exc}"
                )

        st.stop()

    if page in {"🏗 Build Lineups", "🔴 NFL Live", "🏈 NFL Data Center"}:
        workspace_options = ["Public", "Admin"] if is_admin else ["Public"]

        mode = st.radio(
            "Workspace",
            workspace_options,
            horizontal=True,
        )

        if mode == "Admin" and not is_admin:
            st.error("Admin access is not authorized for this account.")
            st.stop()

    if page == "🏗 Build Lineups":
        contest_format = st.radio(
            "Contest Format",
            ["Classic", "Single-Game"],
            horizontal=True,
            key="wfs_contest_format",
        )

        if contest_format == "Classic":
            try:
                pool = current_classic_publication(pool)
                slates = available_slates(pool)
            except Exception:
                slates = []
            if not slates:
                st.info("Current slate data is not available yet." if mode == "Public" else "Current Classic contests are unavailable or staged pending publication. Please check again later.")
                st.stop()
            default_main = next(
                (i for i, s in enumerate(slates) if normalize_slate(s) == "main"),
                0,
            )
            selected_slate = st.selectbox(
                "Slate",
                slates,
                index=default_main,
            )

            if mode == "Public":
                lineups = st.selectbox(
                    "Lineups",
                    [1, 5],
                    index=0,
                    help="Choose 1 lineup or a 5-lineup portfolio.",
                )
            else:
                lineups = st.number_input(
                    "Lineups",
                    min_value=1,
                    max_value=150,
                    value=20,
                    step=1,
                    help="Admin mode retains direct production-solver portfolio sizing.",
                )

            if mode == "Public":
                salary_floor = 0
                max_overlap = 7
                qb_stack = 2
                bring_back = True
                max_player_exposure = 0.60
                max_qb_exposure = 0.40
                max_dst_exposure = 0.40
                projection_loss_pct = 0.01
                max_team = 0
            else:
                salary_floor = st.number_input(
                    "Salary floor",
                    min_value=0,
                    max_value=60000,
                    value=0,
                    step=100,
                    help=(
                        "0 disables the floor. WFS may leave salary unused when the "
                        "production objective prefers that lineup."
                    ),
                )

                max_overlap = st.slider(
                    "Maximum lineup overlap",
                    min_value=0,
                    max_value=8,
                    value=7,
                    step=1,
                )

                qb_stack = st.selectbox(
                    "QB pass-catcher stack",
                    [1, 2],
                    index=1,
                )

                bring_back = st.toggle(
                    "Require opponent bring-back",
                    value=True,
                )

                with st.expander("Exposure limits"):
                    max_player_exposure = st.slider(
                        "RB / WR / TE max exposure", 5, 100, 60, 5,
                    ) / 100.0
                    max_qb_exposure = st.slider(
                        "QB max exposure", 5, 100, 40, 5,
                    ) / 100.0
                    max_dst_exposure = st.slider(
                        "D/ST max exposure", 5, 100, 40, 5,
                    ) / 100.0

                with st.expander("Advanced"):
                    projection_loss_pct = st.slider(
                        "Projection loss allowance",
                        0.0, 5.0, 1.0, 0.25,
                        help="Production default is 1%.",
                    ) / 100.0
                    max_team = st.number_input(
                        "Max players from one team",
                        min_value=0,
                        max_value=9,
                        value=0,
                        step=1,
                        help="0 disables this optional strategy constraint.",
                    )

                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="qt-card">
                      <div style="font-size:.73rem;color:#64748b;text-transform:uppercase;
                                  letter-spacing:.08em;font-weight:800;">Portfolio profile</div>
                      <div style="margin-top:.45rem;color:#e2e8f0;font-size:.86rem;">
                        <b>{int(lineups)}</b> {"lineups" if mode == "Public" else "admin direct-solve lineups"} •
                        QB+{int(qb_stack)} stack • {"Bring-back ON" if bring_back else "Bring-back OFF"}
                      </div>
                      <div style="margin-top:.25rem;color:#94a3b8;font-size:.78rem;">
                        Projection frontier: {projection_loss_pct * 100:.2f}% •
                        Overlap ≤ {int(max_overlap)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            try:
                showdown_sidebar_pool = load_showdown_projection_pool(showdown_pool_freshness_token())
                showdown_slates = available_showdown_slates(showdown_sidebar_pool)
            except Exception as exc:
                if mode == "Admin":
                    st.error(f"Single-Game is unavailable: {exc}")
                else:
                    st.error("Single-Game contests are temporarily unavailable.")
                st.stop()

            if not showdown_slates:
                st.error("No Single-Game slates are available right now.")
                st.stop()

            selected_slate = st.selectbox(
                "Single-Game",
                showdown_slates,
                key="wfs_showdown_slate",
            )

            if mode == "Public":
                lineups = st.selectbox(
                    "Lineups",
                    [1, 5, 10, 20],
                    index=1,
                    help="Choose how many Single-Game lineups to generate.",
                    key="wfs_showdown_lineups_public",
                )
            else:
                lineups = st.number_input(
                    "Lineups",
                    min_value=1,
                    max_value=20,
                    value=20,
                    step=1,
                    help="Showdown Solver V1 is validated through 20 lineups.",
                    key="wfs_showdown_lineups_admin",
                )
    else:
        contest_format = "Classic"
        selected_slate = next(
            (s for s in slates if normalize_slate(s) == "main"),
            slates[0] if slates else None,
        )
        if page == "🏠 Home":
            side_title = "WFS COMMAND CENTER"
            side_sub = "Prepare • Build • Monitor • React"
        elif page == "🔮 Forecast Center":
            side_title = "FORECAST CENTER"
            side_sub = "WFS • Pregame NFL Forecasts"
        elif page == "🏈 NFL Data Center":
            side_title = "NFL DATA CENTER"
            side_sub = "Schedule • Results • Box Scores"
        elif page == "🤖 WFS Analyst":
            side_title = "WFS ANALYST"
            side_sub = "Game Outlooks & Takeaways"
        elif page == "🔴 NFL Live":
            side_title = "NFL LIVE"
            side_sub = "Live Scores & Win Probability"
        elif page == "🔒 Admin":
            side_title = "ADMIN"
            side_sub = "Accounts, Feedback & Health"
        else:
            side_title = "FANTASY LEAGUE HUB"
            side_sub = "Leagues • Teams • Matchups"
        st.markdown(
            f"""
            <div class="qt-card" style="margin-top:.8rem;">
              <div style="font-size:.74rem;color:#94a3b8;text-transform:uppercase;
                          letter-spacing:.08em;font-weight:800;">{side_title}</div>
              <div style="margin-top:.42rem;color:#334155;font-size:.84rem;">
                {side_sub}
              </div>
              <div style="margin-top:.25rem;color:#8299b3;font-size:.75rem;">
                NFL information and fantasy tools.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

selected_pool = (
    slate_pool(pool, selected_slate)
    if (
        contest_format == "Classic"
        and pool is not None
        and selected_slate is not None
    )
    else pd.DataFrame()
)
late_swap_selected_pool = (
    late_swap_slate_pool(pool, selected_slate)
    if (
        contest_format == "Classic"
        and pool is not None
        and selected_slate is not None
    )
    else pd.DataFrame()
)

if page == "🏠 Home":
    render_wfs_home()
    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">Wynners Fantasy Spot • NFL Command Center</div>',
        unsafe_allow_html=True,
    )
    st.stop()

def _wfs_format_kickoff(date_str, weekday="", time_str=""):
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


def _wfs_kickoff_lookup(season, week=None):
    """Return {game_id: kickoff text} for a season (optionally one week)."""
    try:
        query = (
            "SELECT game_id, game_date, weekday, gametime FROM games "
            "WHERE season = ?"
        )
        params = [int(season)]
        if week is not None:
            query += " AND week = ?"
            params.append(int(week))

        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as conn:
            rows = pd.read_sql_query(query, conn, params=params)
    except Exception:
        return {}

    return {
        str(r["game_id"]): _wfs_format_kickoff(
            r["game_date"], r["weekday"], r["gametime"]
        )
        for _, r in rows.iterrows()
    }


def _wfs_forecast_logo_uri(team_code):
    """Return a browser-safe embedded PNG URI for a local NFL team logo."""
    import base64

    logo_path = APP_DIR / "assets" / "nfl_teams" / f"{team_code}.png"

    if not logo_path.is_file():
        return ""

    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


if page == "🔮 Forecast Center":
    if st.button(
        "← Back to Home",
        key="wfs_back_home_forecast",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    st.title("🔮 WFS Forecast Center")
    st.caption(
        "Pregame NFL score projections, matchup outlooks and WFS game picks."
    )

    forecast_path = select_forecast_path()

    if not forecast_path.is_file():
        st.warning("Pregame forecasts are not available yet for the current NFL slate.")
    else:
        forecast_df = pd.read_csv(forecast_path)

        required = {
            "game_id",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "pred_away_points",
            "pred_home_points",
            "pred_winner",
            "pred_home_margin",
            "pred_total_points",
            "forecast_status",
        }

        missing = sorted(required - set(forecast_df.columns))

        if missing:
            st.error(
                "NFL forecasts are temporarily unavailable."
            )
        else:
            forecast_df = forecast_df[
                forecast_df["season"].eq(2026)
            ].copy()

            weeks = sorted(
                int(x)
                for x in forecast_df["week"].dropna().unique()
            )

            if not weeks:
                st.info("No 2026 forecasts are currently available.")
            else:
                try:
                    from wfs_schedule_context import resolve_schedule_week_context

                    forecast_context = resolve_schedule_week_context(
                        season=2026,
                    )
                    forecast_default_week = int(
                        forecast_context.planning_week
                    )
                except Exception:
                    forecast_default_week = None

                if forecast_default_week not in weeks:
                    st.warning(
                        "Current-week NFL forecasts are temporarily unavailable."
                    )
                    selected_week = None
                else:
                    if "wfs_forecast_week" not in st.session_state:
                        st.session_state["wfs_forecast_week"] = (
                            forecast_default_week
                        )

                    if st.session_state["wfs_forecast_week"] not in weeks:
                        st.session_state["wfs_forecast_week"] = (
                            forecast_default_week
                        )

                    selected_week = st.selectbox(
                        "NFL Week",
                        weeks,
                        index=weeks.index(
                            st.session_state["wfs_forecast_week"]
                        ),
                        key="wfs_forecast_week",
                    )

                if selected_week is None:
                    week_df = forecast_df.iloc[0:0].copy()
                else:
                    week_df = forecast_df[
                        forecast_df["week"].eq(selected_week)
                    ].copy()

                week_df = week_df.sort_values(
                    ["game_date", "game_id"]
                )

                kickoff_lookup = (
                    _wfs_kickoff_lookup(2026, selected_week)
                    if selected_week is not None
                    else {}
                )

                ready_count = int(
                    week_df["forecast_status"]
                    .isin({
                        "READY_CORE_ONLY",
                        "READY_INJURY_ADJUSTED",
                    })
                    .sum()
                )

                status_html = (
                    '<div class="qt-card" '
                    'style="padding:.65rem .8rem;'
                    'margin:.3rem 0 .8rem 0;'
                    'text-align:center;">'
                    '<div style="font-size:.76rem;'
                    'color:#94a3b8;font-weight:800;">'
                    f'2026 • WEEK {selected_week} • '
                    f'{len(week_df)} GAMES • {ready_count} READY'
                    '</div></div>'
                )

                st.markdown(
                    status_html,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"### Week {selected_week} Forecasts"
                )

                # WFS_PREDICTION_RECORD_UI_V1
                # Read-only public accuracy summary.
                try:
                    import json as _wfs_accuracy_json

                    _wfs_accuracy_path = (
                        Path("/home/mwynn/nfl_data_engine")
                        / "processed"
                        / "wfs_prediction_accuracy_v1.json"
                    )

                    if _wfs_accuracy_path.is_file():
                        _wfs_accuracy = _wfs_accuracy_json.loads(
                            _wfs_accuracy_path.read_text(
                                encoding="utf-8"
                            )
                        )

                        if (
                            _wfs_accuracy.get("version")
                            == "WFS_PREDICTION_ACCURACY_V1"
                        ):
                            _season = _wfs_accuracy.get("season") or {}

                            _week = next(
                                (
                                    row
                                    for row in (
                                        _wfs_accuracy.get("weeks") or []
                                    )
                                    if int(row.get("week", -1))
                                    == int(selected_week)
                                ),
                                None,
                            )

                            _season_wins = int(
                                _season.get("wins", 0)
                            )
                            _season_losses = int(
                                _season.get("losses", 0)
                            )
                            _season_pct = float(
                                _season.get("accuracy_pct", 0.0)
                            )

                            if _week is not None:
                                _week_wins = int(
                                    _week.get("wins", 0)
                                )
                                _week_losses = int(
                                    _week.get("losses", 0)
                                )
                                _week_pct = float(
                                    _week.get("accuracy_pct", 0.0)
                                )

                                _week_text = (
                                    f"Week {selected_week}: "
                                    f"<b>{_week_wins}-{_week_losses}</b> "
                                    f"• {_week_pct:.1f}%"
                                )
                            else:
                                _week_text = (
                                    f"Week {selected_week}: "
                                    "<b>Not graded yet</b>"
                                )

                            _accuracy_html = (
                                '<div class="qt-card" '
                                'style="padding:.72rem .85rem;'
                                'margin:.25rem 0 .85rem 0;'
                                'text-align:center;">'
                                '<div style="font-size:.66rem;'
                                'font-weight:850;'
                                'letter-spacing:.08em;'
                                'color:#94a3b8;">'
                                'WFS PREDICTION RECORD'
                                '</div>'
                                '<div style="margin-top:.32rem;'
                                'font-size:.92rem;'
                                'font-weight:750;'
                                'color:#e2e8f0;">'
                                f'{_week_text}'
                                '</div>'
                                '<div style="margin-top:.16rem;'
                                'font-size:.78rem;'
                                'color:#94a3b8;">'
                                f'Season: '
                                f'<b>{_season_wins}-{_season_losses}</b> '
                                f'• {_season_pct:.1f}%'
                                '</div>'
                                '</div>'
                            )

                            st.markdown(
                                _accuracy_html,
                                unsafe_allow_html=True,
                            )

                except Exception:
                    pass

                for _, game in week_df.iterrows():
                    away = str(game["away_team"])
                    home = str(game["home_team"])
                    status = str(game["forecast_status"])
                    game_date = kickoff_lookup.get(
                        str(game["game_id"]),
                        str(game["game_date"]),
                    )

                    if status in {
                          "READY_CORE_ONLY",
                          "READY_INJURY_ADJUSTED",
                      }:
                        away_score_raw = float(game["pred_away_points"])
                        home_score_raw = float(game["pred_home_points"])

                        # Read-only Expected Pass Rate lookup.
                        # Preserve exact game + team matching.
                        forecast_game_id = ""
                        for game_id_field in (
                            "game_id",
                            "gid",
                            "event_id",
                        ):
                            if game_id_field in game.index:
                                candidate_game_id = game.get(
                                    game_id_field
                                )
                                if (
                                    candidate_game_id is not None
                                    and pd.notna(candidate_game_id)
                                ):
                                    forecast_game_id = str(
                                        candidate_game_id
                                    ).strip()
                                    if forecast_game_id:
                                        break

                        away_expected_pass_rate = (
                            _dc_expected_pass_rate_text(
                                forecast_game_id,
                                away,
                            )
                        )
                        home_expected_pass_rate = (
                            _dc_expected_pass_rate_text(
                                forecast_game_id,
                                home,
                            )
                        )

                        # Public single-game prediction presentation:
                        # deterministic nonnegative half-up whole numbers.
                        away_score = int(max(0.0, away_score_raw) + 0.5)
                        home_score = int(max(0.0, home_score_raw) + 0.5)

                        # Public total and margin are derived from the
                        # displayed whole-number scores so the card is
                        # internally consistent.
                        total = away_score + home_score
                        margin_value = abs(home_score - away_score)

                        # Presentation only: a rounded tie has no winning team.
                        displayed_winner = (
                            home if home_score > away_score else away
                        )
                        # A checkmark reads as "confirmed" and this is a
                        # pregame pick, not a result, so use a pick marker
                        # instead of a checkmark.
                        lean_text = (
                            "Even" if margin_value == 0
                            else f"\U0001F3C8 {displayed_winner}"
                        )
                        margin_text = (
                            "Even" if margin_value == 0 else
                            f"{displayed_winner} by {margin_value}"
                        )

                        card_html = (
                            '<div class="qt-card" '
                            'style="padding:.8rem;'
                            'margin:.55rem 0;">'

                            '<div style="text-align:center;'
                            'color:#8299b3;'
                            'font-size:.68rem;'
                            'font-weight:750;'
                            'margin-bottom:.4rem;">'
                            f'{game_date}'
                            '</div>'

                            '<div style="display:flex;'
                            'align-items:center;'
                            'justify-content:space-between;'
                            'gap:.25rem;">'

                            '<div style="width:29%;'
                            'text-align:center;">'
                            f'<img src="{_wfs_forecast_logo_uri(away)}" '
                            'style="width:48px;height:48px;'
                            'object-fit:contain;'
                            'display:block;margin:0 auto .25rem auto;" '
                            f'alt="{away}">'
                            '<div style="font-size:1.15rem;'
                            'font-weight:900;'
                            'color:#e2e8f0;">'
                            f'{away}'
                            '</div>'
                            '<div style="font-size:1.8rem;'
                            'font-weight:900;'
                            'color:#8bdcf5;'
                            'line-height:1.1;">'
                            f'{away_score}'
                            '</div>'
                            '</div>'

                            '<div style="width:42%;'
                            'text-align:center;">'
                            '<div style="font-size:.62rem;'
                            'color:#8299b3;'
                            'font-weight:800;'
                            'letter-spacing:.08em;">'
                            'MODEL LEAN'
                            '</div>'
                            '<div style="font-size:1.2rem;'
                            'font-weight:900;'
                            'color:#ffffff;">'
                            f'{lean_text}'
                            '</div>'
                            '<div style="font-size:.67rem;'
                            'color:#94a3b8;">'
                            f'{away} @ {home}'
                            '</div>'
                            '</div>'

                            '<div style="width:29%;'
                            'text-align:center;">'
                            f'<img src="{_wfs_forecast_logo_uri(home)}" '
                            'style="width:48px;height:48px;'
                            'object-fit:contain;'
                            'display:block;margin:0 auto .25rem auto;" '
                            f'alt="{home}">'
                            '<div style="font-size:1.15rem;'
                            'font-weight:900;'
                            'color:#e2e8f0;">'
                            f'{home}'
                            '</div>'
                            '<div style="font-size:1.8rem;'
                            'font-weight:900;'
                            'color:#8bdcf5;'
                            'line-height:1.1;">'
                            f'{home_score}'
                            '</div>'
                            '</div>'

                            '</div>'

                            '<div style="border-top:1px solid #263548;'
                            'margin-top:.6rem;'
                            'padding-top:.5rem;'
                            'text-align:center;">'
                            '<div style="font-size:.64rem;'
                            'color:#8299b3;'
                            'font-weight:850;'
                            'letter-spacing:.06em;">'
                            'EXPECTED PASS RATE'
                            '</div>'
                            '<div style="font-size:.78rem;'
                            'color:#dbeafe;'
                            'font-weight:850;'
                            'margin-top:.16rem;">'
                            f'{away} {away_expected_pass_rate}'
                            '&nbsp;&nbsp;•&nbsp;&nbsp;'
                            f'{home} {home_expected_pass_rate}'
                            '</div>'
                            '<div style="font-size:.74rem;'
                            'color:#cbd5e1;'
                            'margin-top:.42rem;">'
                            f'<b>Total</b> {total}'
                            '&nbsp;&nbsp;•&nbsp;&nbsp;'
                            f'{margin_text}'
                            '</div>'
                            '</div>'

                            '</div>'
                        )

                        st.markdown(
                            card_html,
                            unsafe_allow_html=True,
                        )

                    else:
                        pending_html = (
                            '<div class="qt-card" '
                            'style="padding:.8rem;'
                            'margin:.55rem 0;">'
                            '<div style="font-weight:850;'
                            'color:#e2e8f0;">'
                            f'{away} @ {home}'
                            '</div>'
                            '<div style="color:#8299b3;'
                            'font-size:.74rem;">'
                            f'{game_date}'
                            '</div>'
                            '<div style="color:#f5bd45;'
                            'margin-top:.35rem;'
                            'font-size:.78rem;">'
                            'Forecast pending market inputs'
                            '</div>'
                            '</div>'
                        )

                        st.markdown(
                            pending_html,
                            unsafe_allow_html=True,
                        )

                st.caption(
                    "WFS pregame projections based on currently available "
                    "team, matchup and player information."
                )

    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">'
        'Wynners Fantasy Spot • Forecast Center'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


if page == "🏆 Fantasy League Hub":
    if st.button(
        "← Back to Home",
        key="wfs_back_home_fantasy",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    render_fantasy_league_hub()
    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">Wynners Fantasy Spot • Fantasy League Hub</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if page == "🤖 WFS Analyst":
    if st.button(
        "← Back to Home",
        key="wfs_back_home_analyst",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    from wfs_ai_analyst import (
        FORECAST_CSV as WFS_ANALYST_FORECAST_CSV,
        build_analyst_response,
    )

    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            .wfs-analyst-mobile-spacer {
                height: 2.75rem;
            }
        }
        @media (min-width: 769px) {
            .wfs-analyst-mobile-spacer {
                height: 0;
            }
        }
        </style>
        <div class="wfs-analyst-mobile-spacer"></div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 WFS Analyst")
    st.caption(
        "Game outlooks, fantasy takeaways and postgame "
        "performance analysis powered by WFS football data."
    )

    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH}?mode=ro",
            uri=True,
        ) as analyst_conn:
            analyst_games = pd.read_sql_query(
                """
                SELECT
                    game_id,
                    season,
                    week,
                    game_date,
                    away_team,
                    home_team,
                    completed
                FROM games
                WHERE season = (
                    SELECT MAX(season)
                    FROM games
                )
                ORDER BY game_date, game_id
                """,
                analyst_conn,
            )
    except Exception:
        analyst_games = pd.DataFrame()

    if analyst_games.empty:
        st.info(
            "Game analysis will appear when NFL matchup "
            "information is available."
        )
    else:
        analyst_games = analyst_games.copy()

        analyst_games["game_date_sort"] = pd.to_datetime(
            analyst_games["game_date"],
            errors="coerce",
        )

        analyst_games["completed"] = pd.to_numeric(
            analyst_games["completed"],
            errors="coerce",
        ).fillna(0).astype(int)

        # WFS_ANALYST_SCHEDULE_PRIORITY_V1
        # Preserve the full Analyst game inventory, but prioritize
        # the schedule-authoritative planning week. Completed games
        # remain available for postgame review and later future weeks
        # remain available for advance analysis.
        from wfs_schedule_context import (
            resolve_schedule_week_context,
        )

        analyst_schedule_context = (
            resolve_schedule_week_context()
        )
        analyst_planning_week = int(
            analyst_schedule_context.planning_week
        )

        analyst_games["analyst_priority"] = 2

        analyst_games.loc[
            analyst_games["completed"].eq(1),
            "analyst_priority",
        ] = 1

        analyst_games.loc[
            pd.to_numeric(
                analyst_games["week"],
                errors="coerce",
            ).eq(analyst_planning_week),
            "analyst_priority",
        ] = 0

        # Planning/future games sort chronologically.
        # Historical finals sort most-recent first by using a
        # separate numeric date key.
        analyst_games["analyst_date_key"] = (
            analyst_games["game_date_sort"]
            .map(
                lambda value: (
                    value.value
                    if pd.notna(value)
                    else 0
                )
            )
        )

        analyst_games.loc[
            analyst_games["analyst_priority"].eq(1),
            "analyst_date_key",
        ] *= -1

        analyst_games = analyst_games.sort_values(
            [
                "analyst_priority",
                "analyst_date_key",
                "game_id",
            ],
            ascending=[True, True, True],
        )

        analyst_games["status_label"] = analyst_games[
            "completed"
        ].map(
            lambda value: " • Final"
            if int(value) == 1
            else ""
        )

        analyst_games["analyst_label"] = (
            analyst_games["away_team"].astype(str)
            + " @ "
            + analyst_games["home_team"].astype(str)
            + " • Week "
            + analyst_games["week"].astype(str)
            + analyst_games["status_label"]
        )

        analyst_options = dict(
            zip(
                analyst_games["analyst_label"],
                analyst_games["game_id"],
            )
        )

        selected_analyst_label = st.selectbox(
            "Select matchup",
            list(analyst_options.keys()),
            key="wfs_analyst_game",
        )

        selected_analyst_game = analyst_options[
            selected_analyst_label
        ]

        try:
            analyst_response = build_analyst_response(
                selected_analyst_game
            )
        except Exception:
            analyst_response = None

        if not analyst_response:
            st.info(
                "Analysis is not available for this matchup yet."
            )
        else:
            analyst_mode = analyst_response.get(
                "mode",
                "",
            )

            if analyst_mode == "PREGAME":
                mode_label = "Pregame Outlook"
            elif analyst_mode == "LIVE":
                mode_label = "Live Game Analysis"
            elif analyst_mode == "POSTGAME":
                mode_label = "Postgame Review"
            else:
                mode_label = "Game Analysis"

            st.markdown(f"### {mode_label}")

            headline = analyst_response.get("headline")

            if headline:
                st.markdown(f"## {headline}")

            game_summary = analyst_response.get(
                "game_summary"
            )

            if game_summary:
                st.write(game_summary)

            stat_outlook = analyst_response.get("stat_outlook")

            if isinstance(stat_outlook, dict):
                outlook_available = bool(
                    stat_outlook.get("available")
                    or stat_outlook.get("teams")
                )

                st.markdown("### Stat Outlook")

                # QA F2 / WFS_STAT_OUTLOOK_PUBLIC_SCOPE_NOTE_V1
                #
                # Stat Outlook publishes selected, rounded player-level
                # projections. It is not an exhaustive decomposition of
                # the separate game-level projected team score.
                st.caption(
                    "Selected, rounded player projections. "
                    "These stat lines are not a complete scoring "
                    "breakdown of the projected team score."
                )

                if analyst_mode in {"LIVE", "POSTGAME"}:
                    st.caption("Original kickoff outlook.")

                if outlook_available:
                    outlook_teams = stat_outlook.get("teams") or []

                    for team_outlook in outlook_teams:
                        if not isinstance(team_outlook, dict):
                            continue

                        team_label = str(
                            team_outlook.get("team")
                            or team_outlook.get("team_name")
                            or ""
                        ).strip()

                        if team_label:
                            st.markdown(f"#### {team_label}")

                        offense_lines = (
                            team_outlook.get("offense")
                            or team_outlook.get("offense_lines")
                            or []
                        )

                        if isinstance(offense_lines, str):
                            offense_lines = [offense_lines]

                        for line in offense_lines:
                            line = str(line or "").strip()
                            if line:
                                st.markdown(f"- {line}")

                        kicker_line = (
                            team_outlook.get("kicker")
                            or team_outlook.get("kicker_line")
                        )
                        kicker_line = str(kicker_line or "").strip()

                        if kicker_line:
                            st.markdown(f"- {kicker_line}")

                        dst_line = (
                            team_outlook.get("dst")
                            or team_outlook.get("dst_line")
                            or team_outlook.get("defense")
                        )
                        dst_line = str(dst_line or "").strip()

                        if dst_line:
                            st.markdown(f"- {dst_line}")

                else:
                    unavailable_note = str(
                        stat_outlook.get("unavailable_note")
                        or stat_outlook.get("note")
                        or "Stat outlook is not available for this game."
                    ).strip()

                    if unavailable_note:
                        st.caption(unavailable_note)

                _wfs_render_matchup_outlook(
                    selected_analyst_game,
                    stat_outlook.get("selected_players") or [],
                )

            impact_injuries = (
                analyst_response.get("impact_injuries")
                or []
            )

            if (
                analyst_mode == "PREGAME"
                and impact_injuries
            ):
                public_pregame_injuries = [
                    injury
                    for injury in impact_injuries
                    if str(
                        injury.get(
                            "impact_classification"
                        )
                        or ""
                    )
                    != "INJURY_WATCH"
                ]

                if public_pregame_injuries:
                    st.markdown(
                        "### Pregame Injury Impact"
                    )

                    for injury in (
                        public_pregame_injuries
                    ):
                        player = str(
                            injury.get("player")
                            or "Player"
                        )

                        team = str(
                            injury.get("team")
                            or ""
                        ).strip()

                        position = str(
                            injury.get("position")
                            or ""
                        ).strip()

                        status = str(
                            injury.get("report_status")
                            or injury.get("status")
                            or ""
                        ).strip().upper()

                        primary_injury = str(
                            injury.get(
                                "primary_injury"
                            )
                            or ""
                        ).strip()

                        impact = str(
                            injury.get(
                                "impact_classification"
                            )
                            or ""
                        )

                        player_label = player

                        if position:
                            player_label += (
                                f" ({position})"
                            )

                        if team:
                            player_label += (
                                f" — {team}"
                            )

                        status_label = {
                            "OUT": "Out",
                            "DOUBTFUL": "Doubtful",
                            "QUESTIONABLE": (
                                "Questionable"
                            ),
                        }.get(
                            status,
                            status.title()
                            if status
                            else "Injury Report",
                        )

                        if primary_injury:
                            injury_text = (
                                f"{primary_injury} injury. "
                            )
                        else:
                            injury_text = ""

                        if (
                            impact
                            == "MAJOR_PREGAME_IMPACT"
                        ):
                            public_label = (
                                "Major Pregame Impact"
                            )
                            public_text = (
                                f"{injury_text}"
                                f"Listed {status_label.lower()}. "
                                "Because this involves the "
                                "quarterback, availability "
                                "could materially affect the "
                                "offensive outlook."
                            )

                        elif (
                            impact
                            == "MATERIAL_PREGAME_IMPACT"
                        ):
                            public_label = (
                                "Pregame Injury Impact"
                            )

                            if status == "OUT":
                                availability_text = (
                                    "The player is listed out."
                                )
                            elif status == "DOUBTFUL":
                                availability_text = (
                                    "The player is listed "
                                    "doubtful."
                                )
                            else:
                                availability_text = (
                                    f"The player is listed "
                                    f"{status_label.lower()}."
                                )

                            public_text = (
                                f"{injury_text}"
                                f"{availability_text} "
                                "Recent offensive involvement "
                                "makes the availability change "
                                "potentially meaningful."
                            )

                        elif (
                            impact
                            == "AVAILABILITY_UNCERTAINTY"
                        ):
                            public_label = (
                                "Pregame Injury Watch"
                            )
                            public_text = (
                                f"{injury_text}"
                                "Listed questionable. "
                                "Availability remains "
                                "uncertain, and the player's "
                                "recent role makes the status "
                                "important to monitor."
                            )

                        elif impact == "TEAM_CONTEXT":
                            public_label = (
                                "Offensive Line Impact"
                            )

                            if status == "OUT":
                                availability_text = (
                                    "The player is listed out."
                                )
                            elif status == "DOUBTFUL":
                                availability_text = (
                                    "The player is listed "
                                    "doubtful."
                                )
                            else:
                                availability_text = (
                                    f"The player is listed "
                                    f"{status_label.lower()}."
                                )

                            public_text = (
                                f"{injury_text}"
                                f"{availability_text} "
                                "The availability change "
                                "could affect the offense at "
                                "the team level."
                            )

                        elif (
                            impact
                            == "LINE_AVAILABILITY_UNCERTAINTY"
                        ):
                            public_label = (
                                "Offensive Line Watch"
                            )
                            public_text = (
                                f"{injury_text}"
                                "Listed questionable. "
                                "Availability remains "
                                "uncertain and could affect "
                                "the offense at the team "
                                "level."
                            )

                        else:
                            continue

                        st.markdown(
                            f"**{player_label} — "
                            f"{status_label}**"
                        )

            if (
                analyst_mode in {"LIVE", "POSTGAME"}
                and impact_injuries
            ):
                st.markdown("### Injury Impact")

                for injury in impact_injuries:
                    player = str(
                        injury.get("player")
                        or "Player"
                    )

                    team = str(
                        injury.get("team")
                        or ""
                    ).strip()

                    position = str(
                        injury.get("position")
                        or ""
                    ).strip()

                    status = str(
                        injury.get("status")
                        or ""
                    )

                    impact = str(
                        injury.get("impact_classification")
                        or ""
                    )

                    player_label = player

                    if position:
                        player_label += f" ({position})"

                    if team:
                        player_label += f" — {team}"

                    if status == "RETURNED":
                        public_label = "Injury Update"
                        public_text = (
                            "Was reported injured earlier "
                            "and later returned to the game."
                        )

                    elif impact == "MAJOR_POTENTIAL_IMPACT":
                        public_label = "Impact Injury"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. Because this involves "
                            "the quarterback, it could "
                            "materially affect the team's "
                            "offensive outlook."
                        )

                    elif impact == "POTENTIAL_IMPACT":
                        public_label = "Impact Injury"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. His established role "
                            "makes this potentially meaningful "
                            "to the offense."
                        )

                    elif impact == "TEAM_CONTEXT":
                        public_label = (
                            "Offensive Line Context"
                        )
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed. This could affect the "
                            "offense at a team level."
                        )

                    else:
                        public_label = "Injury Watch"
                        public_text = (
                            "Was reported injured and no "
                            "verified return update has been "
                            "observed."
                        )

                    st.markdown(
                        f"**{public_label} — "
                        f"{player_label}**"
                    )
                    st.write(public_text)

            observations = (
                analyst_response.get("observations")
                or []
            )

            injury_prefixes = (
                "Impact injury:",
                "Injury update:",
                "Offensive-line injury:",
                "Injury watch:",
                "Pregame injury impact:",
                "Pregame injury watch:",
                "Pregame offensive-line context:",
                "Pregame offensive-line watch:",
            )

            observations = [
                observation
                for observation in observations
                if not str(observation).startswith(
                    injury_prefixes
                )
            ]

            if observations:
                st.markdown("### What stands out")

                for observation in observations:
                    st.markdown(f"- {observation}")

            fantasy_leaders = (
                analyst_response.get("fantasy_leaders")
                or []
            )

            if fantasy_leaders:
                st.markdown("### Fantasy Leaders")

                for leader in fantasy_leaders:
                    player = leader.get(
                        "player",
                        "Player",
                    )
                    team = leader.get("team", "")
                    position = leader.get(
                        "position",
                        "",
                    )
                    points = leader.get(
                        "fanduel_points"
                    )

                    if points is None:
                        points_text = "—"
                    else:
                        try:
                            points_text = (
                                f"{float(points):.1f}"
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            points_text = str(points)

                    st.markdown(
                        f"**{player}** "
                        f"({team} {position}) — "
                        f"{points_text} FanDuel pts"
                    )

            if analyst_mode == "PREGAME":
                evidence_status = (
                    analyst_response.get(
                        "evidence_status"
                    )
                    or {}
                )

                if not evidence_status.get(
                    "win_probability_available",
                    False,
                ):
                    st.caption(
                        "Win probability is not shown until "
                        "WFS has a calibrated probability model."
                    )

    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">'
        'Wynners Fantasy Spot • WFS Analyst'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()


if page == "🔴 NFL Live":
    if st.button(
        "← Back to Home",
        key="wfs_back_home_live",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    render_wfs_live_view(mode)

    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">Wynners Fantasy Spot • WFS NFL Live</div>',
        unsafe_allow_html=True,
    )
    st.stop()


if page == "🏈 NFL Data Center":
    if st.button(
        "← Back to Home",
        key="wfs_back_home_data_center",
        width="stretch",
    ):
        _wfs_home_navigate("🏠 Home")

    try:
        render_data_center(mode)
    except Exception as exc:
        if mode == "Admin":
            st.error(f"NFL Data Center is unavailable: {exc}")
        else:
            st.error(
                "NFL Data Center is temporarily unavailable. "
                "Please try again shortly."
            )
        st.stop()

    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">Wynners Fantasy Spot • NFL Data Center • schedule & box scores</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if st.button(
    "← Back to Home",
    key="wfs_back_home_build",
    width="stretch",
):
    _wfs_home_navigate("🏠 Home")

if contest_format == "Single-Game":
    render_showdown_builder(
        selected_slate=selected_slate,
        lineups=int(lineups),
        workspace_mode=mode,
    )
    render_wfs_public_disclaimer()

    st.markdown(
        '<div class="qt-footer">Wynners Fantasy Spot • NFL Single-Game • tournament focused</div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown('<div class="qt-section">1. Build New Lineups</div>', unsafe_allow_html=True)
st.caption("Build directly from the selected FanDuel slate. No upload is required.")

if mode == "Public":
    _wfs_render_public_data_freshness(
        CLASSIC_LINEUP_FRESHNESS_PATH,
        product_label="Player data",
    )

if mode == "Public":
    open_late_swap = bool(
        st.session_state.pop("wfs_open_late_swap", False)
    )

    with st.expander("Late Swap", expanded=open_late_swap):
        render_late_swap_stage1(
            late_swap_selected_pool,
            selected_slate,
        )

# The normal Lineup Generator uses the backend solver-ready slate directly.
# FanDuel template import/export is intentionally isolated to Late Swap.
active_pool = selected_pool.copy()

label_to_key = dict(zip(active_pool["_label"], active_pool["_ui_key"]))
player_labels = list(label_to_key.keys())

st.markdown(
    '<div class="qt-section">2. Contest Intelligence</div>',
    unsafe_allow_html=True,
)
games_count = (
    active_pool["game_solver"].nunique()
    if "game_solver" in active_pool.columns and len(active_pool) else "—"
)
top_projection = (
    f'{active_pool["projection_solver"].max():.2f}'
    if len(active_pool) else "—"
)
salary_range = (
    (
        f'${active_pool["salary_solver"].min():,.0f}–'
        f'${active_pool["salary_solver"].max():,.0f}'
    )
    if len(active_pool) else "—"
)

if mode == "Public":
    st.caption(f"{selected_slate} slate")
    st.markdown(
        f"""
        <div class="qt-card" style="padding:.9rem 1rem;margin:.55rem 0 1rem 0;">
          <div style="display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;
                      font-size:.88rem;font-weight:800;color:#0f172a;">
            <span>{len(active_pool)} Players</span><span>•</span>
            <span>{games_count} Games</span><span>•</span>
            <span>Top projection: {top_projection} pts</span><span>•</span>
            <span>{salary_range}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption(
        f"{selected_slate} • backend solver-ready slate"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eligible players", len(active_pool))
    m2.metric("Games", games_count)
    m3.metric("Top projection", top_projection)
    m4.metric("Salary range", salary_range)

    st.caption("Backend solver-ready slate is active.")

if mode == "Public":
    with st.expander(
        "🏈 FanDuel Lineup Rules",
        expanded=False,
    ):
        st.markdown(
            """
            **FanDuel NFL Classic lineup**

            - **9 players:** QB, RB, RB, WR, WR, WR, TE, FLEX, D/ST
            - **Salary cap:** $60,000 maximum
            - **FLEX:** RB, WR, or TE
            - **Team limit:** Maximum 4 players from one NFL team
            - **Scoring:** FanDuel NFL Classic, including 0.5 points per reception
            - **No yardage bonuses:** No 300-yard passing or 100-yard rushing/receiving bonus

            **Player controls:** Locked players must appear in every generated
            lineup. Excluded players will never be selected. WFS may adapt
            portfolio strategy settings when necessary, but it will not ignore
            your locks/excludes or violate FanDuel roster rules.
            """
        )

st.markdown(
    '<div class="qt-section">3. Player Controls</div>',
    unsafe_allow_html=True,
)
if mode == "Public":
    with st.expander("Optional Player Controls", expanded=True):
        st.caption("Quickly find players by name.")
        locked_labels = st.multiselect(
            "Lock players",
            player_labels,
            placeholder="Search & lock players...",
            help="Tap and type a player name. Search uses fuzzy matching.",
            filter_mode="fuzzy",
            select_all=False,
            key="wfs_classic_locks_" + safe_slug(selected_slate),
        )
        excluded_options = [x for x in player_labels if x not in locked_labels]
        excluded_labels = st.multiselect(
            "Exclude players",
            excluded_options,
            placeholder="Search & exclude players...",
            help="Tap and type a player name. Search uses fuzzy matching.",
            filter_mode="fuzzy",
            select_all=False,
            key="wfs_classic_excludes_" + safe_slug(selected_slate),
        )
else:
    st.caption("Optional overrides. Leave blank to let the production engine build freely.")
    left, right = st.columns(2)
    with left:
        locked_labels = st.multiselect(
            "Lock players", player_labels,
            help="Locked players are forced into every generated lineup.",
            select_all=False,
            key="wfs_classic_admin_locks_" + safe_slug(selected_slate),
        )
    with right:
        excluded_options = [x for x in player_labels if x not in locked_labels]
        excluded_labels = st.multiselect(
            "Exclude players", excluded_options,
            help="Excluded players cannot appear in generated lineups.",
            select_all=False,
            key="wfs_classic_admin_excludes_" + safe_slug(selected_slate),
        )

lock_keys = [label_to_key[x] for x in locked_labels]
user_exclude_keys = [label_to_key[x] for x in excluded_labels]

# Normal generator exclusions come only from explicit user controls.
exclude_keys = sorted(set(user_exclude_keys))

# QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
#
# Signature of the generation inputs as they stand on THIS render. Compared
# against the signature stored at the time of the last successful generation
# to detect whether a displayed portfolio is stale.
current_generation_signature = _wfs_classic_freshness_signature(
    slate_name=selected_slate,
    lineups=int(lineups),
    salary_floor=int(salary_floor),
    max_overlap=int(max_overlap),
    max_team=int(max_team),
    qb_stack=int(qb_stack),
    bring_back=bool(bring_back),
    projection_loss_pct=float(projection_loss_pct),
    max_player_exposure=float(max_player_exposure),
    max_qb_exposure=float(max_qb_exposure),
    max_dst_exposure=float(max_dst_exposure),
    lock_keys=lock_keys,
    exclude_keys=exclude_keys,
)

# QA F1 / WFS_CLASSIC_LOCK_PREFLIGHT_V1
#
# Reject structurally impossible Classic lock combinations before launching
# the production solver. The production solver remains final authority.
def _wfs_classic_locks_fit_roster(locked_positions: list[str]) -> bool:
    slot_eligibility = {
        "QB": ("QB",),
        "RB": ("RB1", "RB2", "FLEX"),
        "WR": ("WR1", "WR2", "WR3", "FLEX"),
        "TE": ("TE", "FLEX"),
        "DST": ("DST",),
    }

    positions = [str(pos).strip().upper() for pos in locked_positions]

    if len(positions) > len(ROSTER_SLOTS):
        return False

    if any(pos not in slot_eligibility for pos in positions):
        return False

    positions.sort(
        key=lambda pos: (len(slot_eligibility[pos]), pos)
    )

    def assign(index: int, used_slots: set[str]) -> bool:
        if index >= len(positions):
            return True

        for slot in slot_eligibility[positions[index]]:
            if slot in used_slots:
                continue

            used_slots.add(slot)

            if assign(index + 1, used_slots):
                return True

            used_slots.remove(slot)

        return False

    return assign(0, set())


# QA F1 / WFS_CLASSIC_LOCK_PREFLIGHT_V1 (diagnostic)
#
# Given a set of locked positions that fails _wfs_classic_locks_fit_roster,
# identify and describe the specific slot conflict so the user knows what
# to unlock, instead of a generic "cannot all fit" message.
def _wfs_classic_lock_conflict_reason(locked_positions: list[str]) -> str:
    dedicated_capacity = {"QB": 1, "DST": 1, "RB": 2, "WR": 3, "TE": 1}
    plural = {"QB": "QBs", "DST": "D/STs", "RB": "RBs", "WR": "WRs", "TE": "TEs"}
    positions = [str(pos).strip().upper() for pos in locked_positions]

    counts = {pos: positions.count(pos) for pos in dedicated_capacity}

    for pos in ("QB", "DST"):
        cap = dedicated_capacity[pos]
        if counts[pos] > cap:
            label = "QB" if pos == "QB" else "D/ST"
            return (
                f"You locked {counts[pos]} {plural[pos]}. "
                f"A lineup allows {cap} {label}."
            )

    for pos in ("RB", "WR", "TE"):
        cap_with_flex = dedicated_capacity[pos] + 1
        if counts[pos] > cap_with_flex:
            return (
                f"You locked {counts[pos]} {plural[pos]}. "
                f"A lineup allows {dedicated_capacity[pos]} plus 1 FLEX."
            )

    flex_eligible_total = counts["RB"] + counts["WR"] + counts["TE"]
    flex_eligible_slots = (
        dedicated_capacity["RB"] + dedicated_capacity["WR"] + dedicated_capacity["TE"] + 1
    )
    if flex_eligible_total > flex_eligible_slots:
        return (
            f"You locked {flex_eligible_total} RB/WR/TE players, but a "
            f"lineup only has {flex_eligible_slots} RB/WR/TE slots (FLEX included)."
        )

    return (
        "Those locked players cannot all fit into one legal FanDuel "
        "Classic lineup. Adjust your player locks."
    )


if len(lock_keys) > 9:
    st.error("A FanDuel lineup contains only nine roster slots.")
    st.stop()

if lock_keys:
    locked_key_set = set(lock_keys)

    locked_rows = active_pool.loc[
        active_pool["_ui_key"].astype(str).isin(locked_key_set)
    ].copy()

    found_lock_keys = set(
        locked_rows["_ui_key"].astype(str).tolist()
    )

    missing_lock_keys = locked_key_set - found_lock_keys

    if missing_lock_keys:
        st.error(
            "One or more locked players are no longer available in "
            "the selected slate."
        )
        st.stop()

    locked_positions = (
        locked_rows["solver_position"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if not _wfs_classic_locks_fit_roster(locked_positions):
        st.error(_wfs_classic_lock_conflict_reason(locked_positions))
        st.stop()

    locked_salary = int(
        locked_rows["salary_solver"].sum()
    )

    if locked_salary > 60000:
        st.error(
            f"Your locked players cost ${locked_salary:,}. "
            "The cap is $60,000."
        )
        st.stop()

    effective_team_cap = (
        min(int(max_team), 4)
        if int(max_team) > 0
        else 4
    )

    locked_team_counts = (
        locked_rows["team_solver"]
        .astype(str)
        .value_counts()
    )

    if (
        not locked_team_counts.empty
        and int(locked_team_counts.max()) > effective_team_cap
    ):
        st.error(
            "Too many locked players are from the same NFL team "
            f"(maximum {effective_team_cap})."
        )
        st.stop()

st.markdown(
    '<div class="qt-section">4. Launch Portfolio</div>',
    unsafe_allow_html=True,
)
if mode == "Admin":
    st.caption("Backend solver-ready slate loaded. Ready to build.")

# QA F6 / WFS_CLASSIC_GENERATION_PROGRESS_V1
#
# Best-effort in-progress flag. Streamlit runs one script pass per session at
# a time and generation is already serialized by PORTFOLIO_CACHE_LOCK, so
# this cannot create overlapping solver runs; it exists to (a) give an
# accurate "already running" message on any rerun that lands while the flag
# is set, and (b) let the button render disabled on that rerun. The button
# rendered on the same click that starts generation cannot itself be
# disabled retroactively -- that is why the info message below is the
# primary, reliable form of immediate feedback.
generate = st.button(
    "⚡ Generate GPP Portfolio",
    type="primary",
    width="stretch",
    disabled=bool(st.session_state.get("wfs_classic_generation_in_progress")),
)

if generate and st.session_state.get("wfs_classic_generation_in_progress"):
    st.warning(
        "A lineup generation is already running for this session. "
        "Please wait for it to finish."
    )
elif generate:
    st.session_state["wfs_classic_generation_in_progress"] = True
    try:
        st.info(
            "⚡ Generating your GPP portfolio… this typically takes "
            "25–35 seconds. Please keep this tab open."
        )
        with _wfs_classic_generation_errors(mode, bool(lock_keys or exclude_keys)), _classic_authority_transaction(selected_slate, selected_pool) as selected_pool:
            require_current_classic_action(selected_slate)
            if not UI_SOLVER.exists():
                if mode == "Admin":
                    st.error(f"Missing UI solver: {UI_SOLVER}")
                else:
                    st.error("Lineup generation is temporarily unavailable.")
                st.stop()

            if mode == "Admin":
                cmd = build_command(
                    slate_name=selected_slate,
                    lineups=int(lineups),
                    salary_floor=int(salary_floor),
                    max_overlap=int(max_overlap),
                    max_team=int(max_team),
                    qb_stack=int(qb_stack),
                    bring_back=bool(bring_back),
                    projection_loss_pct=float(projection_loss_pct),
                    max_player_exposure=float(max_player_exposure),
                    max_qb_exposure=float(max_qb_exposure),
                    max_dst_exposure=float(max_dst_exposure),
                    lock_keys=lock_keys,
                    exclude_keys=exclude_keys,
                )

                PORTFOLIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with PORTFOLIO_CACHE_LOCK.open("a+") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        with st.spinner("Generating deterministic GPP portfolio…"):
                            require_current_classic_action(selected_slate)
                            proc = subprocess.run(
                                cmd,
                                cwd=APP_DIR,
                                capture_output=True,
                                text=True,
                            )

                        st.session_state["last_solver_stdout"] = proc.stdout
                        st.session_state["last_solver_stderr"] = proc.stderr
                        st.session_state["last_returncode"] = proc.returncode
                        st.session_state["last_slate"] = selected_slate

                        if proc.returncode != 0:
                            st.error("Portfolio generation failed.")
                            with st.expander("Solver error", expanded=True):
                                st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))
                            st.stop()

                        (
                            lineups_df,
                            players_df,
                            exposure_df,
                            audit_df,
                        ) = read_outputs(selected_slate)

                        st.session_state["lineups_df"] = lineups_df
                        st.session_state["players_df"] = players_df
                        st.session_state["exposure_df"] = exposure_df
                        st.session_state["audit_df"] = audit_df
                        st.session_state["portfolio_audit_df"] = audit_df
                        st.session_state["portfolio_cache_status"] = "ADMIN LIVE"
                        st.session_state["portfolio_wrapped"] = False
                        # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
                        st.session_state["classic_result_signature"] = (
                            current_generation_signature
                        )

                        st.success(
                            f"Generated {len(lineups_df)} deterministic admin lineups."
                        )
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            else:
                display_count = int(lineups)
                initial_size = int(PORTFOLIO_CACHE_SIZE)

                cache_key = _wfs_portfolio_cache_key(
                    slate_name=selected_slate,
                    selected_pool=selected_pool,
                    salary_floor=int(salary_floor),
                    max_overlap=int(max_overlap),
                    max_team=int(max_team),
                    qb_stack=int(qb_stack),
                    bring_back=bool(bring_back),
                    projection_loss_pct=float(projection_loss_pct),
                    max_player_exposure=float(max_player_exposure),
                    max_qb_exposure=float(max_qb_exposure),
                    max_dst_exposure=float(max_dst_exposure),
                    lock_keys=lock_keys,
                    exclude_keys=exclude_keys,
                )

                seen_state_key = (
                    f"wfs_seen_portfolio::{WFS_USER.get('sub', 'anonymous')}::{cache_key}"
                )
                if seen_state_key in st.session_state:
                    seen_before = set(st.session_state[seen_state_key])
                else:
                    seen_before = set(
                        _wfs_get_seen_portfolio_signatures(
                            WFS_USER.get("sub"),
                            cache_key,
                        )
                    )
                    st.session_state[seen_state_key] = seen_before

                cached = _wfs_read_cached_portfolio(cache_key)
                cache_status = "HIT"

                # Determine whether this request needs initial generation or an append-only
                # +5 extension.  We extend only when the current user has exhausted the
                # committed shared inventory needed for this request.
                need_generation = cached is None
                current_size = 0 if cached is None else len(cached[0])

                if cached is not None:
                    current_signatures = {
                        _wfs_lineup_signature(row)
                        for _, row in cached[0].iterrows()
                    }
                    unseen_count = len(current_signatures - seen_before)
                    if (
                        unseen_count < display_count
                        and current_size < int(PORTFOLIO_CACHE_MAX_SIZE)
                    ):
                        need_generation = True

                if need_generation:
                    PORTFOLIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with PORTFOLIO_CACHE_LOCK.open("a+") as lock_handle:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                        try:
                            # Re-read after acquiring the global solver-output lock.
                            cached = _wfs_read_cached_portfolio(cache_key)
                            current_size = 0 if cached is None else len(cached[0])

                            if cached is not None:
                                current_signatures = {
                                    _wfs_lineup_signature(row)
                                    for _, row in cached[0].iterrows()
                                }
                                unseen_count = len(current_signatures - seen_before)
                            else:
                                unseen_count = 0

                            should_build = cached is None or (
                                unseen_count < display_count
                                and current_size < int(PORTFOLIO_CACHE_MAX_SIZE)
                            )

                            if should_build:
                                if cached is None:
                                    target_size = initial_size
                                    profiles = _wfs_public_profiles(
                                        max_overlap=int(max_overlap),
                                        qb_stack=int(qb_stack),
                                        bring_back=bool(bring_back),
                                        max_player_exposure=float(max_player_exposure),
                                        max_qb_exposure=float(max_qb_exposure),
                                        max_dst_exposure=float(max_dst_exposure),
                                    )
                                    resume_players = ""
                                    existing = None
                                    cache_status = "BUILT"
                                else:
                                    target_size = min(
                                        current_size + int(PORTFOLIO_CACHE_INCREMENT),
                                        int(PORTFOLIO_CACHE_MAX_SIZE),
                                    )
                                    meta = _wfs_read_cache_meta(cache_key)
                                    saved_profile = meta.get("solver_profile") or {}
                                    required_profile = {
                                        "name",
                                        "max_overlap",
                                        "qb_stack",
                                        "bring_back",
                                        "max_player_exposure",
                                        "max_qb_exposure",
                                        "max_dst_exposure",
                                    }
                                    if not required_profile.issubset(saved_profile):
                                        raise RuntimeError(
                                            "This cached portfolio predates resumable profile "
                                            "metadata and cannot be safely extended."
                                        )
                                    # Extension must use the exact profile that created the
                                    # existing family; adaptive fallback cannot change midway.
                                    profiles = [saved_profile]
                                    resume_players = str(
                                        _wfs_portfolio_cache_paths(cache_key)["players"]
                                    )
                                    existing = cached
                                    cache_status = f"EXTENDED_{current_size}_TO_{target_size}"

                                generated, successful_profile = _wfs_run_public_solver_target(
                                    selected_slate=selected_slate,
                                    target_size=target_size,
                                    salary_floor=int(salary_floor),
                                    max_team=int(max_team),
                                    projection_loss_pct=float(projection_loss_pct),
                                    lock_keys=lock_keys,
                                    exclude_keys=exclude_keys,
                                    profiles=profiles,
                                    existing=existing,
                                    resume_players=resume_players,
                                    display_count=display_count,
                                )

                                if generated is None:
                                    st.error(
                                        _wfs_classic_failure_message(
                                            st.session_state.get("last_classic_failure_category", "UNKNOWN")
                                        )
                                    )
                                    st.stop()

                                _wfs_write_cached_portfolio(
                                    cache_key,
                                    selected_slate,
                                    *generated,
                                    expected_size=target_size,
                                    solver_profile=successful_profile,
                                )
                                cached = generated
                            else:
                                cache_status = "HIT"
                        finally:
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

                try:
                    (
                        portfolio_lineups_df,
                        portfolio_players_df,
                        portfolio_exposure_df,
                        portfolio_audit_df,
                    ) = cached

                    shown_lineups_df, seen_after, wrapped = _wfs_take_next_unseen(
                        portfolio_lineups_df,
                        display_count,
                        seen_before,
                    )

                    st.session_state[seen_state_key] = seen_after
                    _wfs_set_seen_portfolio_signatures(
                        WFS_USER.get("sub"),
                        cache_key,
                        seen_after,
                    )

                    portfolio_ids = (
                        shown_lineups_df["portfolio_lineup"].tolist()
                        if "portfolio_lineup" in shown_lineups_df.columns
                        else []
                    )
                    shown_players_df = _wfs_subset_detail_frame(
                        portfolio_players_df,
                        portfolio_ids,
                    )
                    shown_audit_df = _wfs_subset_detail_frame(
                        portfolio_audit_df,
                        portfolio_ids,
                    )

                    st.session_state["lineups_df"] = shown_lineups_df
                    st.session_state["players_df"] = shown_players_df
                    st.session_state["exposure_df"] = portfolio_exposure_df
                    st.session_state["audit_df"] = shown_audit_df
                    st.session_state["portfolio_audit_df"] = portfolio_audit_df
                    st.session_state["portfolio_cache_key"] = cache_key
                    st.session_state["portfolio_cache_status"] = cache_status
                    st.session_state["portfolio_wrapped"] = wrapped
                    st.session_state["last_slate"] = selected_slate
                    # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
                    st.session_state["classic_result_signature"] = (
                        current_generation_signature
                    )

                    st.success(
                        f"Generated {len(shown_lineups_df)} WFS lineup"
                        f'{"s" if len(shown_lineups_df) != 1 else ""}.'
                    )
                except Exception as exc:
                    if mode == "Admin":
                        st.error(str(exc))
                    else:
                        _wfs_report_classic_exception(exc, mode, category="UNKNOWN")
    finally:
        st.session_state["wfs_classic_generation_in_progress"] = False

if "lineups_df" in st.session_state:
    lineups_df = st.session_state["lineups_df"]
    players_df = st.session_state["players_df"]
    exposure_df = st.session_state["exposure_df"]
    audit_df = st.session_state["audit_df"]

    st.markdown(
        '<div class="qt-section" style="margin-top:1.2rem;">5. Lineup Command Center</div>',
        unsafe_allow_html=True,
    )

    # QA F7 / WFS_CLASSIC_RESULT_FRESHNESS_V1
    #
    # Compare the signature stored at the time of the last successful
    # generation against the signature of the controls on screen right now.
    # A missing stored signature (e.g. results carried over from before this
    # feature existed) is treated as not-stale rather than guessed at.
    _stored_generation_signature = st.session_state.get("classic_result_signature")
    _classic_result_is_stale = (
        _stored_generation_signature is not None
        and _stored_generation_signature != current_generation_signature
    )
    if _classic_result_is_stale:
        st.warning(
            "⚠️ Outdated — your Build Lineups settings have changed since this "
            "portfolio was generated. Click **Generate GPP Portfolio** above to "
            "refresh these results before reviewing or downloading them."
        )

    cache_status = st.session_state.get("portfolio_cache_status", "LIVE")
    if mode == "Public":
        st.caption("Review your generated lineups and download when ready.")
    else:
        st.caption(
            "Generation complete. Review quality, exposures, structure, and export. "
            f"Portfolio source: {cache_status}."
        )

    audit_source = st.session_state.get("portfolio_audit_df", audit_df)
    audit_pass = (
        "audit" in audit_source.columns
        and len(audit_source) > 0
        and audit_source["audit"].astype(str).eq("PASS").all()
    )
    projection_high = pd.to_numeric(lineups_df["projection"], errors="coerce").max()
    max_salary = pd.to_numeric(lineups_df["salary"], errors="coerce").max()

    if mode == "Public":
        st.markdown(
            f"""
            <div class="qt-card" style="padding:.85rem 1rem;margin:.5rem 0 1rem 0;">
              <div style="display:flex;gap:.65rem;flex-wrap:wrap;align-items:center;
                          font-size:.86rem;font-weight:800;color:#e2e8f0;">
                <span>{len(lineups_df)} Lineup{"s" if len(lineups_df) != 1 else ""}</span>
                <span>•</span><span>Top Projection {projection_high:.2f}</span>
                <span>•</span><span>Highest Salary ${max_salary:,.0f}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Generated", len(lineups_df))
        a2.metric("Audit", "PASS" if audit_pass else "FAIL")
        a3.metric("Projection high", f"{projection_high:.2f}")
        a4.metric(
            "Projection low",
            f'{pd.to_numeric(lineups_df["projection"], errors="coerce").min():.2f}',
        )
        a5.metric("Max salary", f"${max_salary:,.0f}")

    if mode == "Public":
        tab1, tab2 = st.tabs(
            ["🏟 Lineups", "📊 Exposures"]
        )
        tab3 = None
        tab4 = None
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🏟 Lineups", "📊 Exposures", "🧬 Portfolio Detail", "✅ Audit"]
        )

    with tab1:
        if mode == "Public":
            render_public_lineup_cards(lineups_df, players_df)
        else:
            display = lineup_display(lineups_df)
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
            )
        st.download_button(
            "Download lineup CSV",
            data=generic_lineup_csv(lineups_df),
            file_name=f"nfl_gpp_lineups_{safe_slug(selected_slate)}.csv",
            mime="text/csv",
        )
        st.caption(
            "Download your generated lineups for review or FanDuel entry preparation."
        )

    with tab2:
        if mode == "Public":
            st.caption("Portfolio exposure summary.")
        else:
            st.caption(
                "Exposure is audited across the full validated 20-lineup production "
                "portfolio, even when this click displays a smaller unseen subset."
            )
        exposure_view = exposure_df.copy()
        if "exposure_pct" in exposure_view.columns:
            exposure_view["exposure_pct"] = pd.to_numeric(
                exposure_view["exposure_pct"], errors="coerce"
            ).round(1)
        st.dataframe(
            exposure_view,
            width="stretch",
            hide_index=True,
        )

    if mode == "Admin":
        with tab3:
            st.dataframe(
                players_df,
                width="stretch",
                hide_index=True,
            )

        with tab4:
            st.dataframe(
                audit_df,
                width="stretch",
                hide_index=True,
            )

render_wfs_public_disclaimer()

st.markdown(
    """
    <div class="qt-footer">
      Wynners Fantasy Spot • NFL GPP • data driven • tournament focused
    </div>
    """,
    unsafe_allow_html=True,
)

if mode == "Admin":
    st.divider()
    render_admin_diagnostics(pool, active_pool)

    if "last_solver_stdout" in st.session_state:
        with st.expander("Raw solver diagnostics"):
            st.code(st.session_state.get("last_solver_stdout", ""))
        stderr = st.session_state.get("last_solver_stderr", "")
        if stderr:
            with st.expander("Solver stderr"):
                st.code(stderr)
    if st.session_state.get("last_classic_generation_exception"):
        with st.expander("Classic generation exception"):
            st.code(st.session_state["last_classic_generation_exception"])
