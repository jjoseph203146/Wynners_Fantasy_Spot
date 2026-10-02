#!/usr/bin/env python3

"""
WFS NFL — Injury Consensus V2

Authoritative source:
    nfl.db -> injuries

Secondary source:
    nfl.db -> injury_news_signals

Design:
    Structured NFL injury status retains hard authority.

    FanDuel Research signals supplement practice/news
    information and may add players absent from the
    structured injury table.

Hard gate:
    OUT
    DOUBTFUL
    INACTIVE

Secondary signals alone never create a hard BLOCK.

Identity:
    GSIS ID is authoritative.
    No fuzzy matching occurs here.

Current signal:
    Latest secondary event per GSIS ID for target
    season/week.

Outputs:
    nfl.db -> injury_consensus_current
    nfl.db -> injury_consensus_audit

    data/csv/injury_consensus_current.csv
    data/parquet/injury_consensus_current.parquet
    data/csv/injury_consensus_audit.csv
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "nfl.db"

CSV_PATH = (
    ROOT
    / "data"
    / "csv"
    / "injury_consensus_current.csv"
)

PARQUET_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "injury_consensus_current.parquet"
)

AUDIT_CSV_PATH = (
    ROOT
    / "data"
    / "csv"
    / "injury_consensus_audit.csv"
)

HARD_BLOCK_STATUSES = {
    "OUT",
    "DOUBTFUL",
    "INACTIVE",
}

STATUS_ALIASES = {
    "O": "OUT",
    "OUT": "OUT",
    "D": "DOUBTFUL",
    "DOUBTFUL": "DOUBTFUL",
    "Q": "QUESTIONABLE",
    "QUESTIONABLE": "QUESTIONABLE",
    "INACTIVE": "INACTIVE",
}

PRACTICE_ALIASES = {
    "FULL PARTICIPATION IN PRACTICE": "FULL",
    "FULL PARTICIPANT": "FULL",
    "FULL": "FULL",

    "LIMITED PARTICIPATION IN PRACTICE": "LIMITED",
    "LIMITED PARTICIPANT": "LIMITED",
    "LIMITED": "LIMITED",

    "DID NOT PARTICIPATE IN PRACTICE": "DNP",
    "DID NOT PRACTICE": "DNP",
    "DNP": "DNP",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def canonical_status(value: Any) -> str:
    text = clean_text(value).upper()

    if not text:
        return ""

    return STATUS_ALIASES.get(
        text,
        text,
    )


def canonical_practice(value: Any) -> str:
    text = clean_text(value).upper()

    if not text:
        return ""

    return PRACTICE_ALIASES.get(
        text,
        text,
    )


def table_exists(
    conn: sqlite3.Connection,
    table: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        LIMIT 1
        """,
        (table,),
    ).fetchone()

    return row is not None


def table_columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    }


def latest_season_week(
    conn: sqlite3.Connection,
) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT
            season,
            week
        FROM injuries
        WHERE season IS NOT NULL
          AND week IS NOT NULL
        ORDER BY
            season DESC,
            week DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "Unable to determine latest "
            "injury season/week."
        )

    return int(row[0]), int(row[1])


def ensure_tables(
    conn: sqlite3.Connection,
) -> None:

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS
        injury_news_signals (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            gsis_id TEXT NOT NULL,
            player_name TEXT,
            team TEXT,
            source TEXT NOT NULL,
            source_timestamp TEXT,
            headline TEXT,
            raw_text TEXT,
            practice_signal TEXT,
            availability_signal TEXT,
            signal_strength TEXT,
            ai_interpretation TEXT,
            ai_confidence REAL,
            created_at TEXT NOT NULL,
            UNIQUE (
                season,
                week,
                gsis_id,
                source,
                source_timestamp,
                headline
            )
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS
        injury_consensus_current (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT NOT NULL,
            team TEXT NOT NULL,
            gsis_id TEXT NOT NULL,
            player_name TEXT,
            position TEXT,
            report_status_raw TEXT,
            report_status TEXT,
            practice_status_raw TEXT,
            practice_status TEXT,
            primary_injury TEXT,
            secondary_injury TEXT,
            structured_source TEXT NOT NULL DEFAULT 'NFLVERSE_INJURIES',
            secondary_practice_status TEXT,
            secondary_availability_status TEXT,
            secondary_signal_strength TEXT,
            secondary_source TEXT,
            secondary_source_timestamp TEXT,
            secondary_headline TEXT,
            structured_present INTEGER NOT NULL DEFAULT 1,
            secondary_present INTEGER NOT NULL DEFAULT 0,
            consensus_status TEXT,
            consensus_practice_status TEXT,
            injury_gate TEXT NOT NULL,
            injury_gate_reason TEXT NOT NULL,
            availability_risk TEXT NOT NULL DEFAULT 'LOW',
            injury_risk TEXT NOT NULL DEFAULT 'LOW',
            authoritative_source TEXT NOT NULL,
            news_signal_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT '',
            last_updated TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (
                season,
                week,
                game_type,
                team,
                gsis_id
            )
        )
        """
    )

    # V1 may already exist with fewer columns.
    # Add V2 columns without destroying the table.

    existing = table_columns(
        conn,
        "injury_consensus_current",
    )

    additions = {
        "secondary_injury":
            "TEXT",
        "secondary_practice_status":
            "TEXT",
        "secondary_availability_status":
            "TEXT",
        "secondary_signal_strength":
            "TEXT",
        "secondary_source":
            "TEXT",
        "secondary_source_timestamp":
            "TEXT",
        "secondary_headline":
            "TEXT",
        "structured_present":
            "INTEGER NOT NULL DEFAULT 1",
        "secondary_present":
            "INTEGER NOT NULL DEFAULT 0",
        "consensus_practice_status":
            "TEXT",
        "injury_risk":
            "TEXT NOT NULL DEFAULT 'LOW'",
        "last_updated":
            "TEXT NOT NULL DEFAULT ''",
    }

    for column, sql_type in additions.items():
        if column not in existing:
            conn.execute(
                f"""
                ALTER TABLE
                    injury_consensus_current
                ADD COLUMN
                    "{column}" {sql_type}
                """
            )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS
        injury_consensus_audit (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            metric TEXT NOT NULL,
            value INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (
                season,
                week,
                metric
            )
        )
        """
    )


def load_structured(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> pd.DataFrame:

    df = pd.read_sql_query(
        """
        SELECT
            season,
            week,
            game_type,
            team,
            gsis_id,
            full_name AS player_name,
            position,
            report_primary_injury
                AS primary_injury,
            report_secondary_injury
                AS secondary_injury,
            report_status,
            practice_status
        FROM injuries
        WHERE season = ?
          AND week = ?
        """,
        conn,
        params=(
            season,
            week,
        ),
    )

    if df.empty:
        raise RuntimeError(
            "Structured injury source "
            "returned zero rows."
        )

    for col in df.columns:
        if col not in {
            "season",
            "week",
        }:
            df[col] = df[col].map(
                clean_text
            )

    df["report_status"] = (
        df["report_status"]
        .map(canonical_status)
    )

    df["practice_status"] = (
        df["practice_status"]
        .map(canonical_practice)
    )

    blank_gsis = (
        df["gsis_id"]
        .eq("")
        .sum()
    )

    if blank_gsis:
        raise RuntimeError(
            "Structured injury rows contain "
            f"{blank_gsis} blank GSIS IDs."
        )

    dupes = df.duplicated(
        subset=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
        ],
        keep=False,
    )

    if dupes.any():
        raise RuntimeError(
            "Duplicate structured injury "
            "identity rows detected."
        )

    return df


def load_latest_secondary(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not table_exists(
        conn,
        "injury_news_signals",
    ):
        empty = pd.DataFrame()
        return empty, empty

    signal_cols = table_columns(
        conn,
        "injury_news_signals",
    )

    optional_select = []

    if "source_event_id" in signal_cols:
        optional_select.append(
            "source_event_id"
        )

    if "signal_key" in signal_cols:
        optional_select.append(
            "signal_key"
        )

    optional_sql = (
        ",\n            "
        + ",\n            ".join(optional_select)
        if optional_select
        else ""
    )

    raw = pd.read_sql_query(
        f"""
        SELECT
            season,
            week,
            gsis_id,
            player_name,
            team,
            source,
            source_timestamp,
            headline,
            raw_text,
            practice_signal,
            availability_signal,
            signal_strength,
            created_at
            {optional_sql}
        FROM injury_news_signals
        WHERE season = ?
          AND week = ?
        """,
        conn,
        params=(
            season,
            week,
        ),
    )

    if raw.empty:
        return raw, raw

    for col in raw.columns:
        if col not in {
            "season",
            "week",
        }:
            raw[col] = raw[col].map(
                clean_text
            )

    raw["practice_signal"] = (
        raw["practice_signal"]
        .map(canonical_practice)
    )

    raw["availability_signal"] = (
        raw["availability_signal"]
        .map(
            lambda x: clean_text(x).upper()
        )
    )

    raw["signal_strength"] = (
        raw["signal_strength"]
        .map(
            lambda x: clean_text(x).upper()
        )
    )

    if raw["gsis_id"].eq("").any():
        raise RuntimeError(
            "Secondary signal table contains "
            "blank GSIS IDs."
        )

    # ISO timestamps sort correctly in the current
    # source format. created_at is a deterministic
    # fallback when source_timestamp is blank.

    raw["_sort_timestamp"] = (
        raw["source_timestamp"]
        .where(
            raw["source_timestamp"].ne(""),
            raw["created_at"],
        )
    )

    # FanDuel V2 provides immutable source_event_id/signal_key.
    # Use them only as deterministic tie-breakers after event time.
    # Older/other sources remain supported with blank fallbacks.
    for tie_col in [
        "source_event_id",
        "signal_key",
    ]:
        if tie_col not in raw.columns:
            raw[tie_col] = ""

        raw[tie_col] = (
            raw[tie_col]
            .fillna("")
            .map(clean_text)
        )

    raw = raw.sort_values(
        by=[
            "gsis_id",
            "_sort_timestamp",
            "created_at",
            "source_event_id",
            "signal_key",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    )

    latest = raw.drop_duplicates(
        subset=["gsis_id"],
        keep="first",
    ).copy()

    counts = (
        raw.groupby(
            "gsis_id",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "news_signal_count"
            }
        )
    )

    latest = latest.merge(
        counts,
        on="gsis_id",
        how="left",
        validate="one_to_one",
    )

    latest = latest.drop(
        columns=["_sort_timestamp"]
    )

    return raw, latest


def current_roster_identity(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    required_gsis_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Resolve current roster identity ONLY for GSIS IDs
    required by secondary-only injury signals.

    Unrelated league-wide roster transaction conflicts
    do not stop injury consensus.

    For each required GSIS:
      - one current player name
      - one current team
      - one current position

    Fail closed only when a required identity itself
    is missing or ambiguous.
    """

    empty = pd.DataFrame(
        columns=[
            "gsis_id",
            "roster_player_name",
            "roster_team",
            "roster_position",
        ]
    )

    required = {
        clean_text(value)
        for value in (
            required_gsis_ids
            or set()
        )
        if clean_text(value)
    }

    if not required:
        return empty

    if not table_exists(
        conn,
        "weekly_rosters",
    ):
        raise RuntimeError(
            "weekly_rosters unavailable for "
            "secondary-only identity resolution."
        )

    cols = table_columns(
        conn,
        "weekly_rosters",
    )

    required_columns = {
        "season",
        "week",
        "gsis_id",
        "full_name",
        "team",
        "position",
    }

    if not required_columns.issubset(
        cols
    ):
        missing = sorted(
            required_columns - cols
        )

        raise RuntimeError(
            "weekly_rosters missing identity "
            "columns: "
            + ",".join(missing)
        )

    ordered_ids = sorted(required)

    placeholders = ",".join(
        ["?"] * len(ordered_ids)
    )

    sql = f"""
        SELECT
            gsis_id,
            full_name
                AS roster_player_name,
            team
                AS roster_team,
            position
                AS roster_position
        FROM weekly_rosters
        WHERE season = ?
          AND week = ?
          AND gsis_id IN ({placeholders})
    """

    params = [
        season,
        week,
        *ordered_ids,
    ]

    roster = pd.read_sql_query(
        sql,
        conn,
        params=params,
    )

    for col in roster.columns:
        roster[col] = (
            roster[col]
            .map(clean_text)
        )

    roster = roster[
        roster["gsis_id"].ne("")
    ].copy()

    found_ids = set(
        roster["gsis_id"].tolist()
    )

    missing_ids = sorted(
        required - found_ids
    )

    if missing_ids:
        raise RuntimeError(
            "Required secondary-only GSIS "
            "missing from current roster: "
            + ",".join(missing_ids)
        )

    resolved_rows = []

    for gsis_id in ordered_ids:

        rows = roster[
            roster["gsis_id"].eq(
                gsis_id
            )
        ].copy()

        names = sorted({
            clean_text(value)
            for value in rows[
                "roster_player_name"
            ].tolist()
            if clean_text(value)
        })

        teams = sorted({
            clean_text(value)
            for value in rows[
                "roster_team"
            ].tolist()
            if clean_text(value)
        })

        positions = sorted({
            clean_text(value)
            for value in rows[
                "roster_position"
            ].tolist()
            if clean_text(value)
        })

        if len(names) != 1:
            raise RuntimeError(
                "Required GSIS has ambiguous "
                f"current name: {gsis_id} {names}"
            )

        if len(teams) != 1:
            raise RuntimeError(
                "Required GSIS has ambiguous "
                f"current team: {gsis_id} {teams}"
            )

        if len(positions) != 1:
            raise RuntimeError(
                "Required GSIS has ambiguous "
                f"current position: "
                f"{gsis_id} {positions}"
            )

        resolved_rows.append(
            {
                "gsis_id":
                    gsis_id,

                "roster_player_name":
                    names[0],

                "roster_team":
                    teams[0],

                "roster_position":
                    positions[0],
            }
        )

    resolved = pd.DataFrame(
        resolved_rows
    )

    if len(resolved) != len(required):
        raise RuntimeError(
            "Resolved current roster identity "
            "count does not equal required GSIS "
            "count."
        )

    return resolved


def classify_row(row: pd.Series) -> tuple[
    str,
    str,
    str,
    str,
]:
    report = canonical_status(
        row.get("report_status")
    )

    structured_practice = (
        canonical_practice(
            row.get("practice_status")
        )
    )

    secondary_practice = (
        canonical_practice(
            row.get(
                "secondary_practice_status"
            )
        )
    )

    availability = clean_text(
        row.get(
            "secondary_availability_status"
        )
    ).upper()

    structured_present = int(
        row.get(
            "structured_present",
            0,
        )
        or 0
    )

    secondary_present = int(
        row.get(
            "secondary_present",
            0,
        )
        or 0
    )

    # Structured practice is authoritative when
    # present. Secondary practice fills a blank.

    consensus_practice = (
        structured_practice
        or secondary_practice
    )

    if report in HARD_BLOCK_STATUSES:
        return (
            report,
            consensus_practice,
            "BLOCK",
            "HIGH",
        )

    if report == "QUESTIONABLE":
        return (
            report,
            consensus_practice,
            "ALLOW",
            "ELEVATED",
        )

    if report:
        return (
            report,
            consensus_practice,
            "ALLOW",
            "MONITOR",
        )

    # No formal structured status.
    # Secondary availability never hard-blocks.

    if availability in {
        "OUT_SIGNAL",
        "DOUBTFUL_SIGNAL",
    }:
        return (
            "",
            consensus_practice,
            "ALLOW",
            "ELEVATED",
        )

    if availability in {
        "QUESTIONABLE_SIGNAL",
    }:
        return (
            "",
            consensus_practice,
            "ALLOW",
            "ELEVATED",
        )

    if consensus_practice == "DNP":
        return (
            "",
            consensus_practice,
            "ALLOW",
            "ELEVATED",
        )

    if consensus_practice == "LIMITED":
        return (
            "",
            consensus_practice,
            "ALLOW",
            "MONITOR",
        )

    if consensus_practice == "FULL":
        return (
            "",
            consensus_practice,
            "ALLOW",
            "LOW",
        )

    if secondary_present and not structured_present:
        return (
            "",
            consensus_practice,
            "ALLOW",
            "MONITOR",
        )

    return (
        "",
        consensus_practice,
        "ALLOW",
        "LOW",
    )


def reason_for_row(
    row: pd.Series,
) -> str:

    report = canonical_status(
        row.get("report_status")
    )

    structured_practice = (
        canonical_practice(
            row.get("practice_status")
        )
    )

    secondary_practice = (
        canonical_practice(
            row.get(
                "secondary_practice_status"
            )
        )
    )

    availability = clean_text(
        row.get(
            "secondary_availability_status"
        )
    ).upper()

    structured_present = int(
        row.get(
            "structured_present",
            0,
        )
        or 0
    )

    secondary_present = int(
        row.get(
            "secondary_present",
            0,
        )
        or 0
    )

    if report in HARD_BLOCK_STATUSES:
        return (
            "STRUCTURED_"
            + report
        )

    if report == "QUESTIONABLE":
        return "STRUCTURED_QUESTIONABLE"

    if report:
        return (
            "STRUCTURED_STATUS_"
            + report
        )

    if structured_practice == "DNP":
        return "STRUCTURED_PRACTICE_DNP"

    if structured_practice == "LIMITED":
        return "STRUCTURED_PRACTICE_LIMITED"

    if structured_practice == "FULL":
        return "STRUCTURED_PRACTICE_FULL"

    if availability in {
        "OUT_SIGNAL",
        "DOUBTFUL_SIGNAL",
        "QUESTIONABLE_SIGNAL",
        "EXPECTED_TO_PLAY",
    }:
        return (
            "SECONDARY_"
            + availability
        )

    if secondary_practice == "DNP":
        return "SECONDARY_PRACTICE_DNP"

    if secondary_practice == "LIMITED":
        return (
            "SECONDARY_PRACTICE_LIMITED"
        )

    if secondary_practice == "FULL":
        return "SECONDARY_PRACTICE_FULL"

    if secondary_present and not structured_present:
        return "SECONDARY_SIGNAL_ONLY"

    if structured_present:
        return "STRUCTURED_NO_STATUS"

    return "NO_INJURY_SIGNAL"


def build_consensus(
    structured: pd.DataFrame,
    secondary_latest: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:

    base = structured.copy()

    base["structured_present"] = 1

    base["secondary_present"] = 0

    base["secondary_practice_status"] = ""
    base["secondary_availability_status"] = ""
    base["secondary_signal_strength"] = ""
    base["secondary_source"] = ""
    base["secondary_source_timestamp"] = ""
    base["secondary_headline"] = ""
    base["news_signal_count"] = 0

    if not secondary_latest.empty:

        secondary = secondary_latest[
            [
                "gsis_id",
                "player_name",
                "team",
                "source",
                "source_timestamp",
                "headline",
                "practice_signal",
                "availability_signal",
                "signal_strength",
                "news_signal_count",
            ]
        ].copy()

        secondary = secondary.rename(
            columns={
                "player_name":
                    "secondary_player_name",
                "team":
                    "secondary_team",
                "source":
                    "secondary_source",
                "source_timestamp":
                    "secondary_source_timestamp",
                "headline":
                    "secondary_headline",
                "practice_signal":
                    "secondary_practice_status",
                "availability_signal":
                    "secondary_availability_status",
                "signal_strength":
                    "secondary_signal_strength",
            }
        )

        base = base.merge(
            secondary,
            on="gsis_id",
            how="left",
            suffixes=(
                "",
                "_sec",
            ),
            validate="many_to_one",
        )

        for col in [
            "secondary_source",
            "secondary_source_timestamp",
            "secondary_headline",
            "secondary_practice_status",
            "secondary_availability_status",
            "secondary_signal_strength",
        ]:
            sec_col = col + "_sec"

            if sec_col in base.columns:
                base[col] = (
                    base[sec_col]
                    .fillna("")
                    .map(clean_text)
                )
                base = base.drop(
                    columns=[sec_col]
                )

        if "news_signal_count_sec" in base.columns:
            base["news_signal_count"] = (
                pd.to_numeric(
                    base[
                        "news_signal_count_sec"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            base = base.drop(
                columns=[
                    "news_signal_count_sec"
                ]
            )

        base["secondary_present"] = (
            base["secondary_source"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(int)
        )

        # Add secondary-only GSIS identities.

        structured_ids = set(
            structured["gsis_id"]
            .astype(str)
            .tolist()
        )

        secondary_only = (
            secondary_latest[
                ~secondary_latest[
                    "gsis_id"
                ].isin(
                    structured_ids
                )
            ]
            .copy()
        )

        if not secondary_only.empty:

            if roster.empty:
                raise RuntimeError(
                    "Secondary-only injury "
                    "signals exist but current "
                    "roster identity is unavailable."
                )

            secondary_only = (
                secondary_only.merge(
                    roster,
                    on="gsis_id",
                    how="left",
                    validate="many_to_one",
                )
            )

            missing_roster = (
                secondary_only[
                    "roster_team"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
            )

            if missing_roster.any():
                bad = secondary_only.loc[
                    missing_roster,
                    [
                        "gsis_id",
                        "player_name",
                        "team",
                    ],
                ]

                raise RuntimeError(
                    "Secondary-only signal has "
                    "no current roster identity:\n"
                    + bad.to_string(
                        index=False
                    )
                )

            # FanDuel team must agree with current
            # roster team before the secondary-only
            # signal is accepted into consensus.

            fd_team = (
                secondary_only["team"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            roster_team = (
                secondary_only[
                    "roster_team"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            mismatch = (
                fd_team.ne("")
                & roster_team.ne("")
                & fd_team.ne(roster_team)
            )

            if mismatch.any():
                bad = secondary_only.loc[
                    mismatch,
                    [
                        "gsis_id",
                        "player_name",
                        "team",
                        "roster_team",
                    ],
                ]

                raise RuntimeError(
                    "Secondary/current-roster "
                    "team mismatch:\n"
                    + bad.to_string(
                        index=False
                    )
                )

            rows = []

            for item in secondary_only.itertuples(
                index=False
            ):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "team":
                            clean_text(
                                item.roster_team
                            ),
                        "gsis_id":
                            clean_text(
                                item.gsis_id
                            ),
                        "player_name":
                            clean_text(
                                item.roster_player_name
                            )
                            or clean_text(
                                item.player_name
                            ),
                        "position":
                            clean_text(
                                item.roster_position
                            ),
                        "primary_injury": "",
                        "secondary_injury": "",
                        "report_status": "",
                        "practice_status": "",
                        "secondary_practice_status":
                            canonical_practice(
                                item.practice_signal
                            ),
                        "secondary_availability_status":
                            clean_text(
                                item.availability_signal
                            ).upper(),
                        "secondary_signal_strength":
                            clean_text(
                                item.signal_strength
                            ).upper(),
                        "secondary_source":
                            clean_text(
                                item.source
                            ),
                        "secondary_source_timestamp":
                            clean_text(
                                item.source_timestamp
                            ),
                        "secondary_headline":
                            clean_text(
                                item.headline
                            ),
                        "structured_present": 0,
                        "secondary_present": 1,
                        "news_signal_count":
                            int(
                                item.news_signal_count
                            ),
                    }
                )

            base = pd.concat(
                [
                    base,
                    pd.DataFrame(rows),
                ],
                ignore_index=True,
                sort=False,
            )

    text_cols = [
        "game_type",
        "team",
        "gsis_id",
        "player_name",
        "position",
        "primary_injury",
        "secondary_injury",
        "report_status",
        "practice_status",
        "secondary_practice_status",
        "secondary_availability_status",
        "secondary_signal_strength",
        "secondary_source",
        "secondary_source_timestamp",
        "secondary_headline",
    ]

    for col in text_cols:
        if col not in base.columns:
            base[col] = ""

        base[col] = (
            base[col]
            .fillna("")
            .map(clean_text)
        )

    base["report_status"] = (
        base["report_status"]
        .map(canonical_status)
    )

    base["practice_status"] = (
        base["practice_status"]
        .map(canonical_practice)
    )

    base["secondary_practice_status"] = (
        base[
            "secondary_practice_status"
        ]
        .map(canonical_practice)
    )

    base["structured_present"] = (
        pd.to_numeric(
            base["structured_present"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    base["secondary_present"] = (
        pd.to_numeric(
            base["secondary_present"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    base["news_signal_count"] = (
        pd.to_numeric(
            base["news_signal_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    classified = base.apply(
        classify_row,
        axis=1,
    )

    base["consensus_status"] = [
        item[0]
        for item in classified
    ]

    base["consensus_practice_status"] = [
        item[1]
        for item in classified
    ]

    base["injury_gate"] = [
        item[2]
        for item in classified
    ]

    base["injury_risk"] = [
        item[3]
        for item in classified
    ]

    base["injury_gate_reason"] = (
        base.apply(
            reason_for_row,
            axis=1,
        )
    )

    base["authoritative_source"] = (
        "NFLVERSE_INJURIES"
    )

    base.loc[
        base["structured_present"].eq(0),
        "authoritative_source",
    ] = "CURRENT_ROSTER+SECONDARY_NEWS"

    now = utc_now()

    base["last_updated"] = now

    # Legacy compatibility contract.
    # These columns remain populated because the existing production
    # injury_consensus_current table requires them NOT NULL.
    base["structured_source"] = (
        base["structured_present"]
        .map(
            lambda value:
                "NFLVERSE_INJURIES"
                if int(value or 0) == 1
                else ""
        )
    )

    base["availability_risk"] = (
        base["injury_risk"]
        .fillna("LOW")
        .map(clean_text)
    )

    base["updated_at"] = now

    if base["gsis_id"].eq("").any():
        raise RuntimeError(
            "Consensus contains blank "
            "GSIS IDs."
        )

    duplicates = base.duplicated(
        subset=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
        ],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Consensus contains duplicate "
            "identity rows."
        )

    return base


def validate_consensus(
    structured: pd.DataFrame,
    secondary_latest: pd.DataFrame,
    consensus: pd.DataFrame,
) -> dict[str, int]:

    hard_escape = consensus[
        consensus[
            "consensus_status"
        ].isin(
            HARD_BLOCK_STATUSES
        )
        & ~consensus[
            "injury_gate"
        ].eq("BLOCK")
    ]

    if not hard_escape.empty:
        raise RuntimeError(
            "Hard injury status escaped "
            "BLOCK gate."
        )

    secondary_hard_block = consensus[
        consensus[
            "structured_present"
        ].eq(0)
        & consensus[
            "injury_gate"
        ].eq("BLOCK")
    ]

    if not secondary_hard_block.empty:
        raise RuntimeError(
            "Secondary-only signal created "
            "a hard BLOCK."
        )

    questionable_block = consensus[
        consensus[
            "consensus_status"
        ].eq("QUESTIONABLE")
        & consensus[
            "injury_gate"
        ].ne("ALLOW")
    ]

    if not questionable_block.empty:
        raise RuntimeError(
            "QUESTIONABLE failed ALLOW rule."
        )

    expected_ids = (
        set(
            structured["gsis_id"]
            .astype(str)
        )
        | set(
            secondary_latest.get(
                "gsis_id",
                pd.Series(
                    dtype=str
                ),
            )
            .astype(str)
        )
    )

    actual_ids = set(
        consensus["gsis_id"]
        .astype(str)
    )

    if expected_ids != actual_ids:
        raise RuntimeError(
            "Consensus GSIS universe does "
            "not equal structured union "
            "secondary GSIS universe."
        )

    metrics = {
        "structured_rows":
            len(structured),

        "secondary_latest_rows":
            len(secondary_latest),

        "consensus_rows":
            len(consensus),

        "structured_only_rows":
            int(
                (
                    consensus[
                        "structured_present"
                    ].eq(1)
                    & consensus[
                        "secondary_present"
                    ].eq(0)
                ).sum()
            ),

        "overlap_rows":
            int(
                (
                    consensus[
                        "structured_present"
                    ].eq(1)
                    & consensus[
                        "secondary_present"
                    ].eq(1)
                ).sum()
            ),

        "secondary_only_rows":
            int(
                (
                    consensus[
                        "structured_present"
                    ].eq(0)
                    & consensus[
                        "secondary_present"
                    ].eq(1)
                ).sum()
            ),

        "block_rows":
            int(
                consensus[
                    "injury_gate"
                ].eq("BLOCK").sum()
            ),

        "allow_rows":
            int(
                consensus[
                    "injury_gate"
                ].eq("ALLOW").sum()
            ),

        "hard_status_escape_rows":
            len(hard_escape),

        "secondary_hard_block_rows":
            len(secondary_hard_block),

        "questionable_block_rows":
            len(questionable_block),

        "practice_full_rows":
            int(
                consensus[
                    "consensus_practice_status"
                ].eq("FULL").sum()
            ),

        "practice_limited_rows":
            int(
                consensus[
                    "consensus_practice_status"
                ].eq("LIMITED").sum()
            ),

        "practice_dnp_rows":
            int(
                consensus[
                    "consensus_practice_status"
                ].eq("DNP").sum()
            ),
    }

    return metrics


def write_consensus(
    conn: sqlite3.Connection,
    consensus: pd.DataFrame,
    metrics: dict[str, int],
    season: int,
    week: int,
) -> None:

    # "current" is intentionally current-only in V2.
    # This avoids silently accumulating old weeks.

    conn.execute(
        """
        DELETE FROM
            injury_consensus_current
        """
    )

    desired_columns = [
        "season",
        "week",
        "game_type",
        "team",
        "gsis_id",
        "player_name",
        "position",
        "primary_injury",
        "secondary_injury",
        "report_status",
        "practice_status",
        "structured_source",
        "secondary_practice_status",
        "secondary_availability_status",
        "secondary_signal_strength",
        "secondary_source",
        "secondary_source_timestamp",
        "secondary_headline",
        "structured_present",
        "secondary_present",
        "consensus_status",
        "consensus_practice_status",
        "injury_gate",
        "injury_gate_reason",
        "availability_risk",
        "injury_risk",
        "authoritative_source",
        "news_signal_count",
        "updated_at",
        "last_updated",
    ]

    actual_columns = table_columns(
        conn,
        "injury_consensus_current",
    )

    columns = [
        column
        for column in desired_columns
        if column in actual_columns
    ]

    required_write_columns = {
        "season",
        "week",
        "game_type",
        "team",
        "gsis_id",
        "injury_gate",
        "injury_gate_reason",
        "authoritative_source",
        "news_signal_count",
        "structured_source",
        "availability_risk",
        "updated_at",
        "secondary_injury",
        "injury_risk",
        "last_updated",
    }

    missing_required = sorted(
        required_write_columns
        - set(columns)
    )

    if missing_required:
        raise RuntimeError(
            "Consensus table compatibility "
            "migration incomplete; missing write "
            "columns: "
            + ",".join(missing_required)
        )

    placeholders = ",".join(
        ["?"] * len(columns)
    )

    sql = (
        "INSERT INTO "
        "injury_consensus_current ("
        + ",".join(columns)
        + ") VALUES ("
        + placeholders
        + ")"
    )

    records = []

    for row in consensus[
        columns
    ].itertuples(
        index=False,
        name=None,
    ):
        records.append(tuple(row))

    conn.executemany(
        sql,
        records,
    )

    conn.execute(
        """
        DELETE FROM
            injury_consensus_audit
        WHERE season = ?
          AND week = ?
        """,
        (
            season,
            week,
        ),
    )

    now = utc_now()

    conn.executemany(
        """
        INSERT INTO
            injury_consensus_audit (
                season,
                week,
                metric,
                value,
                created_at
            )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                season,
                week,
                metric,
                int(value),
                now,
            )
            for metric, value
            in metrics.items()
        ],
    )


def export_outputs(
    consensus: pd.DataFrame,
    metrics: dict[str, int],
    season: int,
    week: int,
) -> None:

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARQUET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    consensus.to_csv(
        CSV_PATH,
        index=False,
    )

    consensus.to_parquet(
        PARQUET_PATH,
        index=False,
    )

    audit = pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "metric": metric,
                "value": value,
            }
            for metric, value
            in metrics.items()
        ]
    )

    audit.to_csv(
        AUDIT_CSV_PATH,
        index=False,
    )


def main() -> None:

    print("=" * 72)
    print(
        "WFS NFL INJURY CONSENSUS V2"
    )
    print("=" * 72)

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Missing database: {DB_PATH}"
        )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        ensure_tables(conn)

        season, week = (
            latest_season_week(conn)
        )

        print(
            f"Target: {season} Week {week}"
        )

        structured = load_structured(
            conn,
            season,
            week,
        )

        secondary_raw, secondary_latest = (
            load_latest_secondary(
                conn,
                season,
                week,
            )
        )

        structured_ids = set(
            structured["gsis_id"]
            .astype(str)
            .tolist()
        )

        secondary_ids = set(
            secondary_latest.get(
                "gsis_id",
                pd.Series(dtype=str),
            )
            .astype(str)
            .tolist()
        )

        secondary_only_ids = (
            secondary_ids
            - structured_ids
        )

        print(
            "Secondary-only GSIS IDs:",
            len(secondary_only_ids),
        )

        roster = current_roster_identity(
            conn,
            season,
            week,
            secondary_only_ids,
        )

        print(
            "Structured rows:",
            len(structured),
        )

        print(
            "Secondary raw rows:",
            len(secondary_raw),
        )

        print(
            "Secondary latest GSIS rows:",
            len(secondary_latest),
        )

        consensus = build_consensus(
            structured,
            secondary_latest,
            roster,
            season,
            week,
        )

        metrics = validate_consensus(
            structured,
            secondary_latest,
            consensus,
        )

        write_consensus(
            conn,
            consensus,
            metrics,
            season,
            week,
        )

        conn.commit()

    export_outputs(
        consensus,
        metrics,
        season,
        week,
    )

    print()
    print(
        "=== CONSENSUS GATES ==="
    )

    print(
        consensus[
            [
                "consensus_status",
                "injury_gate",
                "injury_risk",
            ]
        ]
        .value_counts()
        .reset_index(
            name="rows"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=== SOURCE COVERAGE ==="
    )

    print(
        consensus[
            [
                "structured_present",
                "secondary_present",
            ]
        ]
        .value_counts()
        .reset_index(
            name="rows"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=== SECONDARY-ONLY ==="
    )

    secondary_only = consensus[
        consensus[
            "structured_present"
        ].eq(0)
        & consensus[
            "secondary_present"
        ].eq(1)
    ]

    if secondary_only.empty:
        print(
            "No secondary-only players."
        )
    else:
        print(
            secondary_only[
                [
                    "player_name",
                    "gsis_id",
                    "team",
                    "position",
                    "secondary_practice_status",
                    "secondary_availability_status",
                    "consensus_practice_status",
                    "injury_gate",
                    "injury_risk",
                    "injury_gate_reason",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=== OVERLAPPING SECONDARY SIGNALS ==="
    )

    overlap = consensus[
        consensus[
            "structured_present"
        ].eq(1)
        & consensus[
            "secondary_present"
        ].eq(1)
    ]

    if overlap.empty:
        print(
            "No overlapping players."
        )
    else:
        print(
            overlap[
                [
                    "player_name",
                    "team",
                    "practice_status",
                    "secondary_practice_status",
                    "consensus_practice_status",
                    "report_status",
                    "injury_gate",
                    "news_signal_count",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=== AUDIT ==="
    )

    for metric, value in metrics.items():
        print(
            f"{metric} = {value}"
        )

    print()
    print("CSV     :", CSV_PATH)
    print("PARQUET :", PARQUET_PATH)
    print("AUDIT   :", AUDIT_CSV_PATH)

    print()
    print(
        "NFL_INJURY_CONSENSUS_V2_STATUS=PASS"
    )


if __name__ == "__main__":
    main()
