#!/usr/bin/env python3

"""
dst_history.py

Historical NFL D/ST raw-event ingestion layer.

Seasons
-------
2023, 2024, 2025

Source
------
nflverse play-by-play through nflreadpy.load_pbp().

Design
------
This is an AUDIT-FIRST raw defensive-event layer. It does NOT calculate
FanDuel fantasy points. It creates one row per historical game/team and stores
the defensive/special-teams events that can later feed an exact FanDuel D/ST
scoring engine.

It does not modify the frozen offensive projection pipeline.

Outputs
-------
SQLite:
    team_defense_game_stats

CSV:
    nfl_team_defense_game_stats.csv
    audit_dst_history_summary.csv
    audit_dst_history_schema.csv
    audit_dst_history_mismatches.csv

Parquet:
    nfl_team_defense_game_stats.parquet

Important
---------
"opponent_score" is the FINAL SCOREBOARD score from games. It is NOT labeled
FanDuel "points allowed", because exact FanDuel D/ST points-allowed semantics
will be handled separately in the scoring layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import nflreadpy as nfl
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


HISTORICAL_SEASONS = [2023, 2024, 2025]

TABLE_NAME = "team_defense_game_stats"

CSV_OUTPUT = Path(CSV_DIR) / "nfl_team_defense_game_stats.csv"
PARQUET_OUTPUT = Path(PARQUET_DIR) / "nfl_team_defense_game_stats.parquet"

AUDIT_SUMMARY = Path(CSV_DIR) / "audit_dst_history_summary.csv"
AUDIT_SCHEMA = Path(CSV_DIR) / "audit_dst_history_schema.csv"
AUDIT_MISMATCHES = Path(CSV_DIR) / "audit_dst_history_mismatches.csv"


# nflfastR/nflverse field names used if present.
PBP_FIELDS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "play_id",
    "posteam",
    "defteam",
    "play_type",
    "desc",
    "sack",
    "interception",
    "fumble",
    "fumble_lost",
    "fumble_recovery_1_team",
    "fumble_recovery_2_team",
    "safety",
    "punt_attempt",
    "punt_blocked",
    "field_goal_attempt",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "kickoff_attempt",
    "touchdown",
    "td_team",
    "defensive_two_point_conv",
]


REQUIRED_PBP_FIELDS = [
    "game_id",
    "play_id",
    "posteam",
    "defteam",
]


CORE_EVENT_FIELDS = [
    "sack",
    "interception",
    "fumble_recovery_1_team",
    "safety",
    "touchdown",
    "td_team",
]


def section(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def bool_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return numeric(df[column]).eq(1)


def lower_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return text(df[column]).fillna("").str.lower()


def load_games(conn: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    query = f"""
        SELECT
            game_id,
            season,
            week,
            game_type,
            away_team,
            home_team,
            away_score,
            home_score,
            completed
        FROM games
        WHERE season IN ({placeholders})
        ORDER BY season, week, game_id
    """

    games = pd.read_sql_query(
        query,
        conn,
        params=HISTORICAL_SEASONS,
    )

    if games.empty:
        raise RuntimeError(
            "No historical games found for seasons "
            f"{HISTORICAL_SEASONS}."
        )

    if games["game_id"].isna().any():
        raise RuntimeError("games contains NULL game_id values.")

    if games["game_id"].duplicated().any():
        raise RuntimeError("games contains duplicate game_id values.")

    return games


def build_expected_team_games(games: pd.DataFrame) -> pd.DataFrame:
    home = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "completed",
        ]
    ].copy()

    home["team"] = home["home_team"]
    home["opponent_team"] = home["away_team"]
    home["team_score"] = home["home_score"]
    home["opponent_score"] = home["away_score"]
    home["home_away"] = "HOME"

    away = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "completed",
        ]
    ].copy()

    away["team"] = away["away_team"]
    away["opponent_team"] = away["home_team"]
    away["team_score"] = away["away_score"]
    away["opponent_score"] = away["home_score"]
    away["home_away"] = "AWAY"

    keep = [
        "game_id",
        "season",
        "week",
        "game_type",
        "team",
        "opponent_team",
        "home_away",
        "team_score",
        "opponent_score",
        "completed",
    ]

    team_games = pd.concat(
        [home[keep], away[keep]],
        ignore_index=True,
    )

    dup = team_games.duplicated(["game_id", "team"]).sum()
    if dup:
        raise RuntimeError(
            f"Expected team-game map has {dup} duplicate game_id/team rows."
        )

    return team_games


def load_pbp_season(season: int) -> tuple[pd.DataFrame, list[dict]]:
    section(f"LOADING NFLVERSE PBP — {season}")

    pl_df = nfl.load_pbp(season)

    source_columns = list(pl_df.columns)
    source_set = set(source_columns)

    print(f"Source rows: {pl_df.height}")
    print(f"Source columns: {len(source_columns)}")

    missing_required = [
        c for c in REQUIRED_PBP_FIELDS if c not in source_set
    ]

    if missing_required:
        raise RuntimeError(
            f"{season} PBP missing required structural fields: "
            + ", ".join(missing_required)
        )

    schema_audit = []

    for field in PBP_FIELDS:
        schema_audit.append(
            {
                "season": season,
                "field": field,
                "present": int(field in source_set),
                "classification": (
                    "REQUIRED"
                    if field in REQUIRED_PBP_FIELDS
                    else (
                        "CORE_EVENT"
                        if field in CORE_EVENT_FIELDS
                        else "OPTIONAL_EVENT"
                    )
                ),
            }
        )

    selected = [c for c in PBP_FIELDS if c in source_set]

    pbp = pl_df.select(selected).to_pandas()

    if "season" not in pbp.columns:
        pbp["season"] = season

    # Add absent optional fields as NA so downstream logic remains deterministic.
    for field in PBP_FIELDS:
        if field not in pbp.columns:
            pbp[field] = pd.NA

    core_missing = [
        c for c in CORE_EVENT_FIELDS if c not in source_set
    ]

    if core_missing:
        print(
            "WARNING: core event fields missing: "
            + ", ".join(core_missing)
        )

    optional_missing = [
        c for c in PBP_FIELDS
        if c not in source_set
        and c not in REQUIRED_PBP_FIELDS
        and c not in CORE_EVENT_FIELDS
    ]

    if optional_missing:
        print(
            "Optional event fields absent: "
            + ", ".join(optional_missing)
        )

    return pbp, schema_audit


def classify_pbp_events(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Produce event rows with an explicitly credited team.

    The output is still raw-event data. FanDuel points are NOT assigned here.
    """
    events = []

    game_id = text(pbp["game_id"])
    defteam = text(pbp["defteam"])
    posteam = text(pbp["posteam"])
    td_team = text(pbp["td_team"])

    # ------------------------------------------------------------------
    # SACKS
    # ------------------------------------------------------------------
    mask = bool_flag(pbp, "sack") & defteam.notna()
    if mask.any():
        tmp = pd.DataFrame(
            {
                "game_id": game_id[mask],
                "team": defteam[mask],
                "event": "sack",
                "count": 1,
            }
        )
        events.append(tmp)

    # ------------------------------------------------------------------
    # INTERCEPTIONS
    # ------------------------------------------------------------------
    mask = bool_flag(pbp, "interception") & defteam.notna()
    if mask.any():
        tmp = pd.DataFrame(
            {
                "game_id": game_id[mask],
                "team": defteam[mask],
                "event": "interception",
                "count": 1,
            }
        )
        events.append(tmp)

    # ------------------------------------------------------------------
    # DEFENSIVE FUMBLE RECOVERIES
    #
    # Count at most one recovery for the defense per play. A recovery by the
    # possessing team is not credited to the opposing D/ST.
    # ------------------------------------------------------------------
    recovery1 = text(pbp["fumble_recovery_1_team"])
    recovery2 = text(pbp["fumble_recovery_2_team"])

    defensive_recovery_team = pd.Series(
        pd.NA,
        index=pbp.index,
        dtype="string",
    )

    mask1 = (
        recovery1.notna()
        & defteam.notna()
        & recovery1.eq(defteam)
    )
    defensive_recovery_team.loc[mask1] = recovery1.loc[mask1]

    mask2 = (
        defensive_recovery_team.isna()
        & recovery2.notna()
        & defteam.notna()
        & recovery2.eq(defteam)
    )
    defensive_recovery_team.loc[mask2] = recovery2.loc[mask2]

    mask = defensive_recovery_team.notna()

    if mask.any():
        tmp = pd.DataFrame(
            {
                "game_id": game_id[mask],
                "team": defensive_recovery_team[mask],
                "event": "fumble_recovery",
                "count": 1,
            }
        )
        events.append(tmp)

    # ------------------------------------------------------------------
    # SAFETIES
    # ------------------------------------------------------------------
    mask = bool_flag(pbp, "safety") & defteam.notna()
    if mask.any():
        tmp = pd.DataFrame(
            {
                "game_id": game_id[mask],
                "team": defteam[mask],
                "event": "safety",
                "count": 1,
            }
        )
        events.append(tmp)

    # ------------------------------------------------------------------
    # BLOCKED PUNTS / FIELD GOALS / EXTRA POINTS
    # Store the subtypes independently, then derive total blocked_kicks.
    # ------------------------------------------------------------------
    punt_blocked = bool_flag(pbp, "punt_blocked")
    if punt_blocked.any():
        mask = punt_blocked & defteam.notna()
        if mask.any():
            events.append(
                pd.DataFrame(
                    {
                        "game_id": game_id[mask],
                        "team": defteam[mask],
                        "event": "blocked_punt",
                        "count": 1,
                    }
                )
            )

    fg_result = lower_text(pbp, "field_goal_result")
    mask = fg_result.eq("blocked") & defteam.notna()
    if mask.any():
        events.append(
            pd.DataFrame(
                {
                    "game_id": game_id[mask],
                    "team": defteam[mask],
                    "event": "blocked_field_goal",
                    "count": 1,
                }
            )
        )

    xp_result = lower_text(pbp, "extra_point_result")
    mask = xp_result.eq("blocked") & defteam.notna()
    if mask.any():
        events.append(
            pd.DataFrame(
                {
                    "game_id": game_id[mask],
                    "team": defteam[mask],
                    "event": "blocked_extra_point",
                    "count": 1,
                }
            )
        )

    # ------------------------------------------------------------------
    # TOUCHDOWNS
    #
    # Separate ordinary defensive scores from special-teams scores.
    #
    # Kick/punt plays are special teams. A touchdown credited to defteam on
    # a non-kick play is classified as a defensive touchdown.
    # ------------------------------------------------------------------
    touchdown = bool_flag(pbp, "touchdown")

    special_teams_play = (
        bool_flag(pbp, "punt_attempt")
        | bool_flag(pbp, "kickoff_attempt")
        | bool_flag(pbp, "field_goal_attempt")
        | bool_flag(pbp, "extra_point_attempt")
    )

    special_td = (
        touchdown
        & td_team.notna()
        & special_teams_play
    )

    if special_td.any():
        events.append(
            pd.DataFrame(
                {
                    "game_id": game_id[special_td],
                    "team": td_team[special_td],
                    "event": "special_teams_td",
                    "count": 1,
                }
            )
        )

    defensive_td = (
        touchdown
        & td_team.notna()
        & defteam.notna()
        & td_team.eq(defteam)
        & ~special_teams_play
    )

    if defensive_td.any():
        events.append(
            pd.DataFrame(
                {
                    "game_id": game_id[defensive_td],
                    "team": td_team[defensive_td],
                    "event": "defensive_td",
                    "count": 1,
                }
            )
        )

    # ------------------------------------------------------------------
    # DEFENSIVE TWO-POINT / CONVERSION RETURNS
    #
    # nflverse's defensive_two_point_conv is retained as its own raw event.
    # We credit defteam because it is the defending unit on the conversion.
    # ------------------------------------------------------------------
    mask = (
        bool_flag(pbp, "defensive_two_point_conv")
        & defteam.notna()
    )

    if mask.any():
        events.append(
            pd.DataFrame(
                {
                    "game_id": game_id[mask],
                    "team": defteam[mask],
                    "event": "defensive_two_point_return",
                    "count": 1,
                }
            )
        )

    if not events:
        return pd.DataFrame(
            columns=["game_id", "team", "event", "count"]
        )

    out = pd.concat(events, ignore_index=True)

    out = (
        out.groupby(
            ["game_id", "team", "event"],
            as_index=False,
        )["count"]
        .sum()
    )

    return out


def aggregate_events(
    team_games: pd.DataFrame,
    all_events: pd.DataFrame,
) -> pd.DataFrame:
    event_columns = [
        "sack",
        "interception",
        "fumble_recovery",
        "safety",
        "blocked_punt",
        "blocked_field_goal",
        "blocked_extra_point",
        "defensive_td",
        "special_teams_td",
        "defensive_two_point_return",
    ]

    base = team_games.copy()

    if all_events.empty:
        for column in event_columns:
            base[column] = 0
    else:
        pivot = (
            all_events.pivot_table(
                index=["game_id", "team"],
                columns="event",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        pivot.columns.name = None

        base = base.merge(
            pivot,
            on=["game_id", "team"],
            how="left",
            validate="one_to_one",
        )

        for column in event_columns:
            if column not in base.columns:
                base[column] = 0
            base[column] = (
                pd.to_numeric(base[column], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    base["blocked_kicks"] = (
        base["blocked_punt"]
        + base["blocked_field_goal"]
        + base["blocked_extra_point"]
    )

    base["return_tds"] = (
        base["defensive_td"]
        + base["special_teams_td"]
    )

    # More descriptive plural names in the persistent table.
    base = base.rename(
        columns={
            "sack": "sacks",
            "interception": "interceptions",
            "fumble_recovery": "fumble_recoveries",
            "safety": "safeties",
            "blocked_punt": "blocked_punts",
            "blocked_field_goal": "blocked_field_goals",
            "blocked_extra_point": "blocked_extra_points",
            "defensive_td": "defensive_tds",
            "special_teams_td": "special_teams_tds",
            "defensive_two_point_return": "defensive_two_point_returns",
        }
    )

    base["source"] = "nflverse_pbp"
    base["updated_at"] = utc_now()

    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "team",
        "opponent_team",
        "home_away",
        "team_score",
        "opponent_score",
        "completed",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_punts",
        "blocked_field_goals",
        "blocked_extra_points",
        "blocked_kicks",
        "defensive_tds",
        "special_teams_tds",
        "return_tds",
        "defensive_two_point_returns",
        "source",
        "updated_at",
    ]

    return base[columns].copy()


def load_opponent_offense_reference(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build a validation reference from team_game_stats.

    For defense TEAM vs opponent OFFENSE:
      defense sacks       == opponent offense sacks_suffered
      defense interceptions == opponent offense interceptions

    This is validation only; the historical DST table remains PBP-derived.
    """
    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    query = f"""
        SELECT
            game_id,
            team AS offense_team,
            opponent_team AS defense_team,
            sacks_suffered,
            interceptions
        FROM team_game_stats
        WHERE season IN ({placeholders})
    """

    return pd.read_sql_query(
        query,
        conn,
        params=HISTORICAL_SEASONS,
    )


def audit_history(
    conn: sqlite3.Connection,
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    pbp_frames: list[pd.DataFrame],
    events: pd.DataFrame,
    defense: pd.DataFrame,
    schema_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("DST HISTORY AUDIT")

    summary = []
    mismatches = []

    def add(item, value, expected="", status="INFO"):
        summary.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    pbp = pd.concat(pbp_frames, ignore_index=True)

    expected_game_ids = set(games["game_id"].astype(str))
    pbp_game_ids = set(
        pbp["game_id"].dropna().astype(str)
    )

    unknown_pbp_games = sorted(
        pbp_game_ids - expected_game_ids
    )

    missing_pbp_games = sorted(
        expected_game_ids - pbp_game_ids
    )

    expected_rows = len(team_games)
    actual_rows = len(defense)

    duplicate_rows = int(
        defense.duplicated(["game_id", "team"]).sum()
    )

    null_game_id = int(defense["game_id"].isna().sum())
    null_team = int(defense["team"].isna().sum())

    negative_event_rows = 0

    event_cols = [
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_punts",
        "blocked_field_goals",
        "blocked_extra_points",
        "blocked_kicks",
        "defensive_tds",
        "special_teams_tds",
        "return_tds",
        "defensive_two_point_returns",
    ]

    for column in event_cols:
        negative_event_rows += int(
            (pd.to_numeric(defense[column], errors="coerce") < 0).sum()
        )

    add(
        "historical_games",
        len(games),
        855,
        "PASS" if len(games) == 855 else "CHECK",
    )
    add(
        "expected_team_game_rows",
        expected_rows,
        len(games) * 2,
        "PASS" if expected_rows == len(games) * 2 else "FAIL",
    )
    add(
        "stored_team_game_rows",
        actual_rows,
        expected_rows,
        "PASS" if actual_rows == expected_rows else "FAIL",
    )
    add(
        "duplicate_game_team_rows",
        duplicate_rows,
        0,
        "PASS" if duplicate_rows == 0 else "FAIL",
    )
    add(
        "null_game_id_rows",
        null_game_id,
        0,
        "PASS" if null_game_id == 0 else "FAIL",
    )
    add(
        "null_team_rows",
        null_team,
        0,
        "PASS" if null_team == 0 else "FAIL",
    )
    add(
        "pbp_unknown_game_ids",
        len(unknown_pbp_games),
        0,
        "PASS" if not unknown_pbp_games else "FAIL",
    )
    add(
        "scheduled_games_missing_pbp",
        len(missing_pbp_games),
        0,
        "PASS" if not missing_pbp_games else "CHECK",
    )
    add(
        "negative_event_rows",
        negative_event_rows,
        0,
        "PASS" if negative_event_rows == 0 else "FAIL",
    )

    for season in HISTORICAL_SEASONS:
        season_rows = int((defense["season"] == season).sum())
        add(
            f"team_game_rows_{season}",
            season_rows,
            570,
            "PASS" if season_rows == 570 else "CHECK",
        )

    # Event totals are descriptive, not hard-coded correctness targets.
    for column in event_cols:
        add(
            f"total_{column}",
            int(pd.to_numeric(defense[column], errors="coerce").sum()),
            "",
            "INFO",
        )

    # Schema presence audit.
    schema_df = pd.DataFrame(schema_rows)

    for field in CORE_EVENT_FIELDS:
        present_all = bool(
            schema_df.loc[
                schema_df["field"].eq(field),
                "present",
            ].eq(1).all()
        )

        add(
            f"core_field_present_all_seasons::{field}",
            int(present_all),
            1,
            "PASS" if present_all else "FAIL",
        )

    # PBP events must resolve to an expected game/team.
    if not events.empty:
        expected_keys = team_games[
            ["game_id", "team"]
        ].drop_duplicates()

        event_check = events.merge(
            expected_keys.assign(expected_key=1),
            on=["game_id", "team"],
            how="left",
        )

        unresolved_events = event_check[
            event_check["expected_key"].isna()
        ].copy()

        add(
            "unresolved_event_game_team_rows",
            len(unresolved_events),
            0,
            "PASS" if unresolved_events.empty else "FAIL",
        )

        if not unresolved_events.empty:
            unresolved_events["mismatch_type"] = (
                "PBP_EVENT_UNKNOWN_GAME_TEAM"
            )
            mismatches.append(unresolved_events)

    # Independent cross-check against team_game_stats opponent offense.
    reference = load_opponent_offense_reference(conn)

    compare = defense[
        [
            "game_id",
            "team",
            "opponent_team",
            "sacks",
            "interceptions",
        ]
    ].merge(
        reference,
        left_on=["game_id", "team", "opponent_team"],
        right_on=["game_id", "defense_team", "offense_team"],
        how="left",
        validate="one_to_one",
    )

    compare["ref_sacks"] = pd.to_numeric(
        compare["sacks_suffered"],
        errors="coerce",
    )

    compare["ref_interceptions"] = pd.to_numeric(
        compare["interceptions_y"]
        if "interceptions_y" in compare.columns
        else compare["interceptions"],
        errors="coerce",
    )

    # After merge, PBP interceptions is normally interceptions_x.
    pbp_interceptions_col = (
        "interceptions_x"
        if "interceptions_x" in compare.columns
        else "interceptions"
    )

    compare["pbp_interceptions"] = pd.to_numeric(
        compare[pbp_interceptions_col],
        errors="coerce",
    )

    compare["pbp_sacks"] = pd.to_numeric(
        compare["sacks"],
        errors="coerce",
    )

    reference_missing = compare["offense_team"].isna()

    sack_mismatch = (
        ~reference_missing
        & compare["ref_sacks"].notna()
        & compare["pbp_sacks"].ne(compare["ref_sacks"])
    )

    int_mismatch = (
        ~reference_missing
        & compare["ref_interceptions"].notna()
        & compare["pbp_interceptions"].ne(
            compare["ref_interceptions"]
        )
    )

    add(
        "opponent_offense_reference_missing",
        int(reference_missing.sum()),
        0,
        "PASS" if not reference_missing.any() else "FAIL",
    )

    add(
        "sack_crosscheck_mismatches",
        int(sack_mismatch.sum()),
        0,
        "PASS" if not sack_mismatch.any() else "CHECK",
    )

    add(
        "interception_crosscheck_mismatches",
        int(int_mismatch.sum()),
        0,
        "PASS" if not int_mismatch.any() else "CHECK",
    )

    if sack_mismatch.any():
        tmp = compare.loc[
            sack_mismatch,
            [
                "game_id",
                "team",
                "opponent_team",
                "pbp_sacks",
                "ref_sacks",
            ],
        ].copy()
        tmp["mismatch_type"] = "SACK_CROSSCHECK"
        mismatches.append(tmp)

    if int_mismatch.any():
        tmp = compare.loc[
            int_mismatch,
            [
                "game_id",
                "team",
                "opponent_team",
                "pbp_interceptions",
                "ref_interceptions",
            ],
        ].copy()
        tmp["mismatch_type"] = "INTERCEPTION_CROSSCHECK"
        mismatches.append(tmp)

    # Blocked kick derivation identity.
    block_identity_bad = defense[
        "blocked_kicks"
    ].ne(
        defense["blocked_punts"]
        + defense["blocked_field_goals"]
        + defense["blocked_extra_points"]
    )

    add(
        "blocked_kick_identity_errors",
        int(block_identity_bad.sum()),
        0,
        "PASS" if not block_identity_bad.any() else "FAIL",
    )

    return_identity_bad = defense[
        "return_tds"
    ].ne(
        defense["defensive_tds"]
        + defense["special_teams_tds"]
    )

    add(
        "return_td_identity_errors",
        int(return_identity_bad.sum()),
        0,
        "PASS" if not return_identity_bad.any() else "FAIL",
    )

    summary_df = pd.DataFrame(summary)

    if mismatches:
        mismatch_df = pd.concat(
            mismatches,
            ignore_index=True,
            sort=False,
        )
    else:
        mismatch_df = pd.DataFrame(
            columns=["mismatch_type"]
        )

    print(summary_df.to_string(index=False))

    structural_failures = summary_df[
        summary_df["status"].eq("FAIL")
    ]

    if not structural_failures.empty:
        print()
        print("STRUCTURAL AUDIT: FAIL")
        print(
            structural_failures[
                ["item", "value", "expected"]
            ].to_string(index=False)
        )
    else:
        print()
        print("STRUCTURAL AUDIT: PASS")

    return summary_df, mismatch_df


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {qident(TABLE_NAME)} (
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT,
            team TEXT NOT NULL,
            opponent_team TEXT,
            home_away TEXT,
            team_score INTEGER,
            opponent_score INTEGER,
            completed INTEGER,
            sacks INTEGER NOT NULL DEFAULT 0,
            interceptions INTEGER NOT NULL DEFAULT 0,
            fumble_recoveries INTEGER NOT NULL DEFAULT 0,
            safeties INTEGER NOT NULL DEFAULT 0,
            blocked_punts INTEGER NOT NULL DEFAULT 0,
            blocked_field_goals INTEGER NOT NULL DEFAULT 0,
            blocked_extra_points INTEGER NOT NULL DEFAULT 0,
            blocked_kicks INTEGER NOT NULL DEFAULT 0,
            defensive_tds INTEGER NOT NULL DEFAULT 0,
            special_teams_tds INTEGER NOT NULL DEFAULT 0,
            return_tds INTEGER NOT NULL DEFAULT 0,
            defensive_two_point_returns INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (game_id, team)
        )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_team_defense_game_stats_season_week
        ON {qident(TABLE_NAME)} (season, week)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_team_defense_game_stats_team
        ON {qident(TABLE_NAME)} (team)
        """
    )


def replace_historical_rows(
    conn: sqlite3.Connection,
    defense: pd.DataFrame,
) -> None:
    """
    Replace only the managed historical seasons in this derived DST table.
    No offensive table is touched.
    """
    create_table(conn)

    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    conn.execute(
        f"""
        DELETE FROM {qident(TABLE_NAME)}
        WHERE season IN ({placeholders})
        """,
        HISTORICAL_SEASONS,
    )

    columns = list(defense.columns)

    insert_sql = f"""
        INSERT INTO {qident(TABLE_NAME)}
        ({", ".join(qident(c) for c in columns)})
        VALUES ({", ".join(["?"] * len(columns))})
    """

    rows = []

    for row in defense.itertuples(index=False, name=None):
        cleaned = []
        for value in row:
            if pd.isna(value):
                cleaned.append(None)
            elif hasattr(value, "item"):
                cleaned.append(value.item())
            else:
                cleaned.append(value)
        rows.append(tuple(cleaned))

    conn.executemany(insert_sql, rows)


def export_outputs(
    conn: sqlite3.Connection,
    summary_df: pd.DataFrame,
    schema_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
) -> None:
    section("EXPORTING DST HISTORY")

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    out = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(TABLE_NAME)}
        WHERE season IN ({placeholders})
        ORDER BY season, week, game_id, team
        """,
        conn,
        params=HISTORICAL_SEASONS,
    )

    out.to_csv(CSV_OUTPUT, index=False)
    out.to_parquet(PARQUET_OUTPUT, index=False)

    summary_df.to_csv(AUDIT_SUMMARY, index=False)
    schema_df.to_csv(AUDIT_SCHEMA, index=False)
    mismatch_df.to_csv(AUDIT_MISMATCHES, index=False)

    print(f"Rows exported: {len(out)}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Parquet: {PARQUET_OUTPUT}")
    print(f"Audit summary: {AUDIT_SUMMARY}")
    print(f"Audit schema: {AUDIT_SCHEMA}")
    print(f"Audit mismatches: {AUDIT_MISMATCHES}")


def main() -> None:
    section("NFL HISTORICAL D/ST RAW EVENT BUILD")

    print(f"Database: {DATABASE_PATH}")
    print(f"Seasons: {HISTORICAL_SEASONS}")
    print("FanDuel scoring: NOT calculated in this script.")
    print("Frozen offensive pipeline: NOT modified.")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as conn:
        games = load_games(conn)

        section("EXPECTED HISTORICAL TEAM-GAME MAP")
        print(f"Games: {len(games)}")

        team_games = build_expected_team_games(games)
        print(f"Expected team-game rows: {len(team_games)}")

        pbp_frames = []
        schema_rows = []

        for season in HISTORICAL_SEASONS:
            pbp, season_schema = load_pbp_season(season)
            pbp_frames.append(pbp)
            schema_rows.extend(season_schema)

        section("CLASSIFYING RAW DEFENSIVE EVENTS")

        event_frames = []

        for season, pbp in zip(HISTORICAL_SEASONS, pbp_frames):
            classified = classify_pbp_events(pbp)
            classified["season_source"] = season
            event_frames.append(classified)

            print(
                f"{season}: {len(classified)} aggregated "
                "game/team/event records"
            )

        if event_frames:
            all_events = pd.concat(
                event_frames,
                ignore_index=True,
            )

            # A game/team/event should belong to exactly one season source.
            all_events = (
                all_events.groupby(
                    ["game_id", "team", "event"],
                    as_index=False,
                )["count"]
                .sum()
            )
        else:
            all_events = pd.DataFrame(
                columns=[
                    "game_id",
                    "team",
                    "event",
                    "count",
                ]
            )

        defense = aggregate_events(
            team_games,
            all_events,
        )

        schema_df = pd.DataFrame(schema_rows)

        summary_df, mismatch_df = audit_history(
            conn=conn,
            games=games,
            team_games=team_games,
            pbp_frames=pbp_frames,
            events=all_events,
            defense=defense,
            schema_rows=schema_rows,
        )

        structural_failures = summary_df[
            summary_df["status"].eq("FAIL")
        ]

        if not structural_failures.empty:
            schema_df.to_csv(AUDIT_SCHEMA, index=False)
            summary_df.to_csv(AUDIT_SUMMARY, index=False)
            mismatch_df.to_csv(AUDIT_MISMATCHES, index=False)

            raise RuntimeError(
                "DST history structural audit failed. "
                "Database write aborted. Review audit files."
            )

        section("WRITING DERIVED DST HISTORY TABLE")

        replace_historical_rows(
            conn,
            defense,
        )

        conn.commit()

        stored = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {qident(TABLE_NAME)}
            WHERE season IN (2023, 2024, 2025)
            """
        ).fetchone()[0]

        print(f"Stored historical rows: {stored}")

        if stored != len(defense):
            raise RuntimeError(
                f"Stored row mismatch: {stored} != {len(defense)}"
            )

        export_outputs(
            conn,
            summary_df,
            schema_df,
            mismatch_df,
        )

    section("DST HISTORY BUILD COMPLETE")

    print("Raw historical D/ST event layer created successfully.")
    print("No FanDuel fantasy points were calculated.")
    print(
        "Next layer after audit review: "
        "dst_fanduel_scoring.py"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 82)
        print("DST HISTORY BUILD FAILED")
        print("=" * 82)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
