#!/usr/bin/env python3

"""
WFS NFL — Injury Consensus V2

Primary current report authority:
    ESPN structured NFL injury API, exact ESPN-to-GSIS crosswalk

Practice/fallback structured source:
    nfl.db -> injuries

Secondary source:
    nfl.db -> injury_news_signals

Design:
    Applicable ESPN report designations lead; NFLVERSE practice is retained.
    Non-hard ESPN designations never erase a valid hard block.

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

import espn_injury_ingest as espn
from wfs_schedule_context import resolve_schedule_week_context


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

    additions.update({column: "TEXT" for column in espn.LINEAGE})

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
        print(
            "STRUCTURED_INJURY_ROWS=0"
        )
        print(
            "STRUCTURED_INJURY_STATE="
            "TARGET_WEEK_EVIDENCE_ABSENT"
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

    if base.empty:
        # The schedule planning week can legitimately precede
        # publication of target-week injury evidence. Preserve the
        # empty evidence population; GAV2 later expands the canonical
        # offensive universe using exact-GSIS identity/depth
        # corroboration without fabricating injury evidence.
        base["consensus_status"] = ""
        base["consensus_practice_status"] = ""
        base["injury_gate"] = ""
        base["injury_risk"] = ""
        base["injury_gate_reason"] = ""
    else:
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

    desired_columns.extend(espn.LINEAGE)

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



# WFS_GLOBAL_PLAYER_AVAILABILITY_V2_CORE
#
# Global availability authority.
#
# Identity:
#   exact GSIS only.
#
# Roster:
#   latest updated_at record per GSIS controls current state.
#   primary roster status controls state family.
#   description/status code refines the primary family only.
#
# Hard unavailable roster families:
#   RES = reserve
#   CUT = cut / waived
#   DEV = practice/development squad
#   EXE = commissioner exempt / exempt
#
# ACT is an active roster state but does not override a current
# structured OUT/INACTIVE injury designation.
#
# QUESTIONABLE and DOUBTFUL are never zeroed by roster state alone.


def _gav2_clean(value: Any) -> str:
    return clean_text(value)


def _gav2_latest_roster(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> pd.DataFrame:

    if not table_exists(conn, "weekly_rosters"):
        raise RuntimeError(
            "WFS_GLOBAL_PLAYER_AVAILABILITY_V2 requires "
            "weekly_rosters"
        )

    cols = table_columns(conn, "weekly_rosters")

    required = {
        "season",
        "week",
        "gsis_id",
        "full_name",
        "team",
        "position",
        "status",
        "status_description_abbr",
        "updated_at",
    }

    missing = sorted(required - cols)

    if missing:
        raise RuntimeError(
            "weekly_rosters missing GAV2 columns: "
            + ",".join(missing)
        )

    roster = pd.read_sql_query(
        """
        SELECT
            season,
            week,
            gsis_id,
            full_name AS roster_player_name,
            team AS roster_team,
            position AS roster_position,
            status AS roster_status,
            status_description_abbr AS roster_status_code,
            updated_at AS roster_updated_at
        FROM weekly_rosters
        WHERE season = ?
          AND week = ?
          AND gsis_id IS NOT NULL
          AND TRIM(CAST(gsis_id AS TEXT)) <> ''
        """,
        conn,
        params=(season, week),
    )

    if roster.empty:
        print(
            "CURRENT_WEEKLY_ROSTER_ROWS=0"
        )
        print(
            "CURRENT_WEEKLY_ROSTER_STATE="
            "TARGET_WEEK_EVIDENCE_ABSENT"
        )
        return roster

    for col in roster.columns:
        if col not in {"season", "week"}:
            roster[col] = (
                roster[col]
                .fillna("")
                .map(_gav2_clean)
            )

    roster["_updated"] = pd.to_datetime(
        roster["roster_updated_at"],
        errors="coerce",
        utc=True,
    )

    if roster["_updated"].isna().any():
        bad = roster[
            roster["_updated"].isna()
        ][
            [
                "gsis_id",
                "roster_team",
                "roster_status",
                "roster_updated_at",
            ]
        ]

        raise RuntimeError(
            "Fail closed: current roster has invalid updated_at:\n"
            + bad.to_string(index=False)
        )

    roster = roster.sort_values(
        [
            "gsis_id",
            "_updated",
            "roster_team",
            "roster_status",
            "roster_status_code",
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

    latest = roster.drop_duplicates(
        subset=["gsis_id"],
        keep="first",
    ).copy()

    latest = latest.drop(
        columns=["_updated"]
    )

    if latest["gsis_id"].duplicated().any():
        raise RuntimeError(
            "Latest roster state is not singleton per GSIS"
        )

    return latest


def _gav2_roster_decision(
    status: Any,
    status_code: Any,
) -> tuple[bool, str, str]:

    primary = _gav2_clean(status).upper()
    code = _gav2_clean(status_code).upper()

    if primary == "ACT":
        return (
            False,
            "ACTIVE_ROSTER_STATE",
            "CURRENT_WEEKLY_ROSTER",
        )

    if primary == "RES":
        return (
            True,
            "ROSTER_RESERVE_STATUS:"
            + (code or "UNKNOWN"),
            "CURRENT_WEEKLY_ROSTER",
        )

    if primary == "CUT":
        return (
            True,
            "ROSTER_CUT_STATUS:"
            + (code or "UNKNOWN"),
            "CURRENT_WEEKLY_ROSTER",
        )

    if primary == "DEV":
        return (
            True,
            "ROSTER_PRACTICE_SQUAD_STATUS:"
            + (code or "UNKNOWN"),
            "CURRENT_WEEKLY_ROSTER",
        )

    if primary == "EXE":
        return (
            True,
            "ROSTER_EXEMPT_STATUS:"
            + (code or "UNKNOWN"),
            "CURRENT_WEEKLY_ROSTER",
        )

    if primary in {
        "INA",
        "INACTIVE",
        "RET",
        "RETIRED",
        "WAI",
        "WAIVED",
    }:
        return (
            True,
            "ROSTER_HARD_UNAVAILABLE:"
            + primary
            + ":"
            + (code or "UNKNOWN"),
            "CURRENT_WEEKLY_ROSTER",
        )

    return (
        False,
        "ROSTER_STATUS_UNRESOLVED:"
        + (primary or "BLANK")
        + ":"
        + (code or "BLANK"),
        "CURRENT_WEEKLY_ROSTER",
    )


def _gav2_expand_global_consensus(
    conn: sqlite3.Connection,
    consensus: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Expand injury consensus from injury-report population to the
    current exact-GSIS roster population.

    Structured injury authority wins over roster ACTIVE state.
    Hard roster unavailability wins when structured injury does
    not already establish a hard state.

    This function does not use fuzzy or display-name identity.
    """

    roster = _gav2_latest_roster(
        conn,
        season,
        week,
    )

    # WFS_GLOBAL_PLAYER_AVAILABILITY_V2_UNIVERSE
    #
    # Global roster expansion is restricted to the canonical
    # current WFS forecast universe. weekly_rosters is an
    # authority source for player state, not the definition of
    # the fantasy-player universe.
    forecast_path = (
        ROOT
        / "data"
        / "parquet"
        / "nfl_current_unified_stat_forecasts.parquet"
    )

    if not forecast_path.exists():
        raise RuntimeError(
            "Missing canonical stat forecast universe: "
            f"{forecast_path}"
        )

    forecast_universe = pd.read_parquet(
        forecast_path,
        columns=[
            "player_id",
            "entity_type",
            "game_id",
            "position",
        ],
    )

    for column in [
        "player_id",
        "entity_type",
        "game_id",
        "position",
    ]:
        forecast_universe[column] = (
            forecast_universe[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    forecast_universe["entity_type"] = (
        forecast_universe["entity_type"]
        .str.upper()
    )
    forecast_universe["position"] = (
        forecast_universe["position"]
        .str.upper()
    )

    # Global Availability V2 universe:
    # current-week offensive fantasy players only.
    #
    # KICKER rows are present in the unified stat artifact but
    # are not part of this availability contract.
    # DST has team identity rather than player GSIS identity and
    # is likewise outside this player-availability authority.
    offense_universe = forecast_universe[
        forecast_universe[
            "entity_type"
        ].eq("OFFENSE_PLAYER")
    ].copy()

    if offense_universe.empty:
        raise RuntimeError(
            "Canonical WFS offensive fantasy universe is empty"
        )

    blank_ids = int(
        offense_universe["player_id"]
        .eq("")
        .sum()
    )
    if blank_ids:
        raise RuntimeError(
            "Canonical WFS offensive fantasy universe has "
            f"{blank_ids} blank player IDs"
        )

    duplicate_ids = int(
        offense_universe["player_id"]
        .duplicated(keep=False)
        .sum()
    )
    if duplicate_ids:
        raise RuntimeError(
            "Canonical WFS offensive fantasy universe has "
            f"{duplicate_ids} duplicate player-ID rows"
        )

    valid_positions = {
        "QB",
        "RB",
        "WR",
        "TE",
    }
    invalid_positions = sorted(
        set(
            offense_universe[
                "position"
            ].tolist()
        )
        - valid_positions
    )
    if invalid_positions:
        raise RuntimeError(
            "Canonical WFS offensive fantasy universe has "
            "invalid positions: "
            + ",".join(invalid_positions)
        )

    expected_game_prefix = (
        f"{int(season)}_{int(week):02d}_"
    )
    bad_game_ids = offense_universe[
        ~offense_universe[
            "game_id"
        ].str.startswith(
            expected_game_prefix
        )
    ]
    if not bad_game_ids.empty:
        sample = sorted(
            set(
                bad_game_ids[
                    "game_id"
                ].tolist()
            )
        )[:10]
        raise RuntimeError(
            "Canonical WFS offensive fantasy universe contains "
            "rows outside the requested season/week: "
            + repr(sample)
        )

    fantasy_ids = set(
        offense_universe[
            "player_id"
        ].tolist()
    )

    print(
        "CANONICAL_OFFENSIVE_UNIVERSE_ROWS =",
        len(offense_universe),
    )
    print(
        "CANONICAL_OFFENSIVE_UNIQUE_IDS =",
        len(fantasy_ids),
    )
    print(
        "CANONICAL_OFFENSIVE_UNIVERSE_CONTRACT=PASS"
    )

    roster = roster[
        roster["gsis_id"].isin(
            fantasy_ids
        )
    ].copy()

    roster_ids = {
        clean_text(value)
        for value in roster[
            "gsis_id"
        ].tolist()
        if clean_text(value)
    }

    missing_roster_ids = sorted(
        fantasy_ids - roster_ids
    )

    # WFS_GLOBAL_PLAYER_AVAILABILITY_V2_MISSING_ROSTER
    #
    # Missing from the current weekly_rosters window does NOT
    # establish ACTIVE and does NOT establish OUT.
    #
    # Generic fallback authority requires:
    #   1. exact GSIS identity;
    #   2. player_identity.latest_team agrees with forecast team;
    #   3. latest depth-chart team agrees with forecast team;
    #   4. no fuzzy/display-name identity;
    #
    # When those conditions hold, the player remains in the
    # 348-player universe with availability UNKNOWN / gate ALLOW.
    # Historical roster status is never carried forward across
    # a missing current roster window.

    forecast_scope = pd.read_parquet(
        forecast_path,
        columns=[
            "player_id",
            "entity_name",
            "team",
            "position",
            "entity_type",
        ],
    )

    forecast_scope["player_id"] = (
        forecast_scope["player_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    forecast_scope["entity_type"] = (
        forecast_scope["entity_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    forecast_scope = forecast_scope[
        forecast_scope[
            "entity_type"
        ].eq("OFFENSE_PLAYER")
    ].copy()

    fallback_rows = []

    if missing_roster_ids:

        placeholders = ",".join(
            ["?"] * len(missing_roster_ids)
        )

        identity = pd.read_sql_query(
            f"""
            SELECT
                CAST(gsis_id AS TEXT) AS gsis_id,
                full_name AS identity_player_name,
                latest_team AS identity_team,
                position AS identity_position
            FROM player_identity
            WHERE CAST(gsis_id AS TEXT)
                  IN ({placeholders})
            """,
            conn,
            params=missing_roster_ids,
        )

        for col in identity.columns:
            identity[col] = (
                identity[col]
                .fillna("")
                .map(clean_text)
            )

        if (
            identity["gsis_id"]
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                "Missing-roster fallback has duplicate "
                "player_identity GSIS rows"
            )

        # Exact-GSIS team corroboration only when current depth
        # coverage is absent. Do NOT select roster status here:
        # historical availability must never carry into the
        # planning week.
        roster_team_history = pd.read_sql_query(
            f"""
            SELECT
                CAST(gsis_id AS TEXT) AS gsis_id,
                team,
                season,
                week
            FROM weekly_rosters
            WHERE CAST(gsis_id AS TEXT)
                  IN ({placeholders})
            ORDER BY season DESC, week DESC
            """,
            conn,
            params=missing_roster_ids,
        )

        for col in roster_team_history.columns:
            roster_team_history[col] = (
                roster_team_history[col]
                .fillna("")
                .map(clean_text)
            )

        depth = pd.read_sql_query(
            f"""
            SELECT
                gsis_id,
                team,
                player_name,
                pos_abb,
                pos_name,
                snapshot_dt,
                updated_at
            FROM depth_charts
            WHERE gsis_id IN ({placeholders})
            """,
            conn,
            params=missing_roster_ids,
        )

        for col in depth.columns:
            depth[col] = (
                depth[col]
                .fillna("")
                .map(clean_text)
            )

        if not depth.empty:

            depth["_snapshot"] = pd.to_datetime(
                depth["snapshot_dt"],
                errors="coerce",
                utc=True,
            )

            depth["_updated"] = pd.to_datetime(
                depth["updated_at"],
                errors="coerce",
                utc=True,
            )

            depth = depth.sort_values(
                [
                    "gsis_id",
                    "_snapshot",
                    "_updated",
                    "team",
                    "pos_abb",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    True,
                    True,
                ],
                kind="stable",
            )

            latest_depth = (
                depth.drop_duplicates(
                    subset=["gsis_id"],
                    keep="first",
                )
                .drop(
                    columns=[
                        "_snapshot",
                        "_updated",
                    ]
                )
            )

        else:
            latest_depth = depth.copy()

        now = utc_now()

        for gsis_id in missing_roster_ids:

            forecast_row = forecast_scope[
                forecast_scope[
                    "player_id"
                ].eq(gsis_id)
            ]

            if len(forecast_row) != 1:
                raise RuntimeError(
                    "Missing-roster fallback requires "
                    "one canonical forecast row for "
                    f"{gsis_id}; found "
                    f"{len(forecast_row)}"
                )

            forecast_row = (
                forecast_row.iloc[0]
            )

            identity_row = identity[
                identity["gsis_id"]
                .eq(gsis_id)
            ]

            if len(identity_row) != 1:
                raise RuntimeError(
                    "Missing-roster fallback requires "
                    "one exact player_identity row for "
                    f"{gsis_id}; found "
                    f"{len(identity_row)}"
                )

            identity_row = (
                identity_row.iloc[0]
            )

            depth_row = latest_depth[
                latest_depth["gsis_id"]
                .eq(gsis_id)
            ]

            roster_team_row = roster_team_history[
                roster_team_history["gsis_id"]
                .eq(gsis_id)
            ]

            if len(depth_row) > 1:
                raise RuntimeError(
                    "Missing-roster fallback has multiple "
                    "latest exact depth-chart rows for "
                    f"{gsis_id}; found "
                    f"{len(depth_row)}"
                )

            if len(depth_row) == 1:
                corroboration_team = clean_text(
                    depth_row.iloc[0]["team"]
                ).upper()
                corroboration_source = (
                    "DEPTH_CHART_CONTEXT"
                )
            else:
                if roster_team_row.empty:
                    raise RuntimeError(
                        "Missing-roster fallback has no exact "
                        "depth or historical roster team "
                        f"corroboration for {gsis_id}"
                    )

                roster_teams = {
                    clean_text(value).upper()
                    for value in roster_team_row["team"]
                    if clean_text(value)
                }

                # Historical team evidence may contain legitimate
                # old teams. Use only the latest season/week rows.
                roster_period = roster_team_row.copy()
                roster_period["_season"] = pd.to_numeric(
                    roster_period["season"],
                    errors="coerce",
                )
                roster_period["_week"] = pd.to_numeric(
                    roster_period["week"],
                    errors="coerce",
                )
                max_season = roster_period["_season"].max()
                roster_period = roster_period[
                    roster_period["_season"].eq(max_season)
                ]
                max_week = roster_period["_week"].max()
                roster_period = roster_period[
                    roster_period["_week"].eq(max_week)
                ]

                latest_teams = {
                    clean_text(value).upper()
                    for value in roster_period["team"]
                    if clean_text(value)
                }

                if len(latest_teams) != 1:
                    raise RuntimeError(
                        "Missing-roster historical team "
                        "corroboration is ambiguous for "
                        f"{gsis_id}: "
                        f"{sorted(latest_teams)}"
                    )

                corroboration_team = next(
                    iter(latest_teams)
                )
                corroboration_source = (
                    "HISTORICAL_ROSTER_TEAM_CONTEXT"
                )

            forecast_team = clean_text(
                forecast_row["team"]
            ).upper()

            identity_team = clean_text(
                identity_row[
                    "identity_team"
                ]
            ).upper()

            if not forecast_team:
                raise RuntimeError(
                    "Blank forecast team in missing-roster "
                    f"fallback for {gsis_id}"
                )

            if identity_team != forecast_team:
                raise RuntimeError(
                    "Missing-roster exact identity team "
                    "contradiction for "
                    f"{gsis_id}: "
                    f"forecast={forecast_team} "
                    f"identity={identity_team}"
                )

            if corroboration_team != forecast_team:
                raise RuntimeError(
                    "Missing-roster corroborating team "
                    "contradiction for "
                    f"{gsis_id}: "
                    f"forecast={forecast_team} "
                    f"corroboration={corroboration_team} "
                    f"source={corroboration_source}"
                )

            fallback_rows.append({
                "season":
                    season,

                "week":
                    week,

                "game_type":
                    "REG",

                "team":
                    forecast_team,

                "gsis_id":
                    gsis_id,

                "player_name":
                    clean_text(
                        identity_row[
                            "identity_player_name"
                        ]
                    )
                    or clean_text(
                        forecast_row[
                            "entity_name"
                        ]
                    ),

                "position":
                    clean_text(
                        forecast_row[
                            "position"
                        ]
                    ),

                "primary_injury":
                    "",

                "secondary_injury":
                    "",

                "report_status":
                    "",

                "practice_status":
                    "",

                "structured_source":
                    "",

                "secondary_practice_status":
                    "",

                "secondary_availability_status":
                    "",

                "secondary_signal_strength":
                    "",

                "secondary_source":
                    "",

                "secondary_source_timestamp":
                    "",

                "secondary_headline":
                    "",

                "structured_present":
                    0,

                "secondary_present":
                    0,

                "consensus_status":
                    "",

                "consensus_practice_status":
                    "",

                "injury_gate":
                    "ALLOW",

                "injury_gate_reason":
                    (
                        "CURRENT_ROSTER_MISSING_"
                        "EXACT_IDENTITY_"
                        + (
                            "DEPTH_CORROBORATED"
                            if corroboration_source
                            == "DEPTH_CHART_CONTEXT"
                            else
                            "HISTORICAL_ROSTER_TEAM_CORROBORATED"
                        )
                    ),

                "availability_risk":
                    "UNKNOWN",

                "injury_risk":
                    "UNKNOWN",

                "authoritative_source":
                    (
                        "PLAYER_IDENTITY+"
                        + corroboration_source
                    ),

                "news_signal_count":
                    0,

                "updated_at":
                    now,

                "last_updated":
                    now,
            })

    base = consensus.copy()

    # Structured injury/news sources are evidence authorities,
    # not universe authorities. Restrict all pre-existing
    # consensus rows to the same canonical offensive pool before
    # global roster expansion.
    base["gsis_id"] = (
        base["gsis_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    base = base[
        base["gsis_id"].isin(
            fantasy_ids
        )
    ].copy()

    # WFS availability policy:
    # DOUBTFUL retains its official source status but is
    # operationally unavailable for projections and lineups.
    #
    # QUESTIONABLE remains eligible. No source status is
    # rewritten and no fuzzy identity matching is used.
    doubtful = (
        base["consensus_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("DOUBTFUL")
    )
    base.loc[
        doubtful,
        "injury_gate",
    ] = "BLOCK"
    base.loc[
        doubtful,
        "injury_gate_reason",
    ] = "WFS_POLICY_DOUBTFUL_UNAVAILABLE"
    base.loc[
        doubtful,
        "availability_risk",
    ] = "HIGH"
    base.loc[
        doubtful,
        "injury_risk",
    ] = "HIGH"
    if "gsis_id" not in base.columns:
        raise RuntimeError(
            "Consensus missing gsis_id"
        )

    base["gsis_id"] = (
        base["gsis_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if base["gsis_id"].eq("").any():
        raise RuntimeError(
            "Consensus contains blank GSIS identity"
        )

    # Collapse the existing structured/secondary authority to
    # exactly one row per GSIS before joining the global roster.
    #
    # Current production may contain game_type/team distinctions,
    # but global availability must be singleton by player.
    base["_hard_rank"] = (
        base["injury_gate"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("BLOCK")
        .astype(int)
    )

    base["_structured_rank"] = (
        pd.to_numeric(
            base.get(
                "structured_present",
                pd.Series(
                    0,
                    index=base.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    base = base.sort_values(
        [
            "gsis_id",
            "_hard_rank",
            "_structured_rank",
            "team",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    )

    base = base.drop_duplicates(
        subset=["gsis_id"],
        keep="first",
    ).copy()

    base = base.drop(
        columns=[
            "_hard_rank",
            "_structured_rank",
        ]
    )

    existing = {
        str(x).strip()
        for x in base["gsis_id"].tolist()
        if str(x).strip()
    }

    new_rows = []

    for row in roster.itertuples(index=False):

        gsis_id = _gav2_clean(row.gsis_id)

        if not gsis_id or gsis_id in existing:
            continue

        hard, reason, source = (
            _gav2_roster_decision(
                row.roster_status,
                row.roster_status_code,
            )
        )

        now = utc_now()

        new_rows.append({
            "season": season,
            "week": week,
            "game_type": "REG",
            "team":
                _gav2_clean(row.roster_team),
            "gsis_id":
                gsis_id,
            "player_name":
                _gav2_clean(row.roster_player_name),
            "position":
                _gav2_clean(row.roster_position),
            "primary_injury": "",
            "secondary_injury": "",
            "report_status":
                "OUT" if hard else "",
            "practice_status": "",
            "structured_source": "",
            "secondary_practice_status": "",
            "secondary_availability_status": "",
            "secondary_signal_strength": "",
            "secondary_source": "",
            "secondary_source_timestamp": "",
            "secondary_headline": "",
            "structured_present": 0,
            "secondary_present": 0,
            "consensus_status":
                "OUT" if hard else "",
            "consensus_practice_status": "",
            "injury_gate":
                "BLOCK" if hard else "ALLOW",
            "injury_gate_reason":
                reason,
            "availability_risk":
                "HIGH" if hard else "LOW",
            "injury_risk":
                "HIGH" if hard else "LOW",
            "authoritative_source":
                source,
            "news_signal_count": 0,
            "updated_at": now,
            "last_updated": now,
        })

    if new_rows:
        base = pd.concat(
            [
                base,
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
            sort=False,
        )

    if fallback_rows:
        fallback_ids = {
            clean_text(
                row["gsis_id"]
            )
            for row in fallback_rows
        }

        existing_fallback_overlap = (
            set(
                base["gsis_id"]
                .fillna("")
                .astype(str)
                .str.strip()
                .tolist()
            )
            & fallback_ids
        )

        # Existing current structured/news evidence already owns
        # the row. Do not create a duplicate fallback identity.
        rows_to_add = [
            row
            for row in fallback_rows
            if clean_text(
                row["gsis_id"]
            )
            not in existing_fallback_overlap
        ]

        if rows_to_add:
            base = pd.concat(
                [
                    base,
                    pd.DataFrame(
                        rows_to_add
                    ),
                ],
                ignore_index=True,
                sort=False,
            )

    roster_map = roster.set_index(
        "gsis_id",
        drop=False,
    )

    # Apply roster hard state to existing non-hard consensus rows.
    # Structured OUT/INACTIVE remains hard regardless of ACT roster.
    for idx in base.index:

        gsis_id = _gav2_clean(
            base.at[idx, "gsis_id"]
        )

        if gsis_id not in roster_map.index:
            continue

        rr = roster_map.loc[gsis_id]

        if isinstance(rr, pd.DataFrame):
            raise RuntimeError(
                "Latest roster map is not singleton "
                f"for {gsis_id}"
            )

        roster_hard, roster_reason, roster_source = (
            _gav2_roster_decision(
                rr["roster_status"],
                rr["roster_status_code"],
            )
        )

        current_gate = _gav2_clean(
            base.at[idx, "injury_gate"]
        ).upper()

        current_status = canonical_status(
            base.at[idx, "consensus_status"]
        )

        structured_present = int(
            base.at[idx, "structured_present"]
            or 0
        )

        # Current structured hard injury authority wins.
        if (
            structured_present == 1
            and current_gate == "BLOCK"
        ):
            continue

        if roster_hard:
            base.at[
                idx,
                "report_status",
            ] = "OUT"

            base.at[
                idx,
                "consensus_status",
            ] = "OUT"

            base.at[
                idx,
                "injury_gate",
            ] = "BLOCK"

            base.at[
                idx,
                "injury_gate_reason",
            ] = roster_reason

            base.at[
                idx,
                "availability_risk",
            ] = "HIGH"

            base.at[
                idx,
                "injury_risk",
            ] = "HIGH"

            base.at[
                idx,
                "authoritative_source",
            ] = roster_source

        elif current_status in {
            "QUESTIONABLE",
            "DOUBTFUL",
        }:
            # Preserve the actual injury designation and its
            # numerical-forecast policy.
            pass

    if base["gsis_id"].duplicated().any():
        raise RuntimeError(
            "GAV2 output is not singleton per exact GSIS"
        )

    # Contract assertions.
    q_bad = base[
        base["consensus_status"]
        .eq("QUESTIONABLE")
        & base["injury_gate"].eq("BLOCK")
    ]

    if not q_bad.empty:
        raise RuntimeError(
            "QUESTIONABLE was hard-blocked by GAV2"
        )

    return base.reset_index(drop=True)


# Preserve the original builder so the global layer remains a
# deterministic extension rather than a second consensus engine.
_wfs_v1_build_consensus = build_consensus


def build_consensus(
    structured: pd.DataFrame,
    secondary_latest: pd.DataFrame,
    roster: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Compatibility wrapper.

    The original structured + secondary consensus is built first.
    Global roster expansion is applied by main() after this call,
    because the expansion requires the read-only DB connection.
    """
    return _wfs_v1_build_consensus(
        structured,
        secondary_latest,
        roster,
        season,
        week,
    )


_wfs_v1_validate_consensus = validate_consensus


def validate_consensus(
    structured: pd.DataFrame,
    secondary_latest: pd.DataFrame,
    consensus: pd.DataFrame,
) -> dict[str, int]:

    # V2 validation intentionally replaces the old invariant that
    # rejected every secondary/global hard block.
    hard_escape = consensus[
        consensus["consensus_status"]
        .isin({"OUT", "INACTIVE", "DOUBTFUL"})
        & ~consensus["injury_gate"].eq("BLOCK")
    ]

    if not hard_escape.empty:
        raise RuntimeError(
            "Hard unavailable status escaped BLOCK"
        )

    questionable_block = consensus[
        consensus["consensus_status"]
        .eq("QUESTIONABLE")
        & consensus["injury_gate"].ne("ALLOW")
    ]

    if not questionable_block.empty:
        raise RuntimeError(
            "QUESTIONABLE failed ALLOW rule"
        )

    doubtful_escape = consensus[
        consensus["consensus_status"]
        .eq("DOUBTFUL")
        & consensus["injury_gate"].ne("BLOCK")
    ]
    if not doubtful_escape.empty:
        raise RuntimeError(
            "DOUBTFUL escaped WFS unavailable policy"
        )
    if consensus["gsis_id"].duplicated().any():
        raise RuntimeError(
            "Global consensus contains duplicate GSIS IDs"
        )

    return {
        "structured_rows":
            int(len(structured)),
        "secondary_latest_rows":
            int(len(secondary_latest)),
        "consensus_rows":
            int(len(consensus)),
        "global_unique_gsis_rows":
            int(consensus["gsis_id"].nunique()),
        "block_rows":
            int(
                consensus["injury_gate"]
                .eq("BLOCK")
                .sum()
            ),
        "allow_rows":
            int(
                consensus["injury_gate"]
                .eq("ALLOW")
                .sum()
            ),
        "questionable_rows":
            int(
                consensus["consensus_status"]
                .eq("QUESTIONABLE")
                .sum()
            ),
        "doubtful_rows":
            int(
                consensus["consensus_status"]
                .eq("DOUBTFUL")
                .sum()
            ),
        "roster_hard_block_rows":
            int(
                (
                    consensus["authoritative_source"]
                    .eq("CURRENT_WEEKLY_ROSTER")
                    & consensus["injury_gate"]
                    .eq("BLOCK")
                ).sum()
            ),
    }


_wfs_v1_main = main


def main() -> None:
    """
    WFS_GLOBAL_PLAYER_AVAILABILITY_V2 production entry point.

    No fuzzy identity.
    No display-name identity.
    Global roster state is exact-GSIS only.
    """

    print("=" * 72)
    print(
        "WFS GLOBAL PLAYER AVAILABILITY V2"
    )
    print("=" * 72)

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Missing database: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as conn:

        ensure_tables(conn)

        schedule_ctx = resolve_schedule_week_context(
            db_path=DB_PATH,
        )
        season = int(schedule_ctx.season)
        week = int(schedule_ctx.planning_week)

        print(
            f"Target: {season} Week {week}"
        )
        print(
            "TARGET_AUTHORITY=SCHEDULE_CONTEXT"
        )

        structured = load_structured(
            conn,
            season,
            week,
        )

        espn_evidence = espn.ingest(conn, season, week)
        structured = espn.merge_structured(structured, espn_evidence, season, week)

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

        secondary_roster = (
            current_roster_identity(
                conn,
                season,
                week,
                secondary_only_ids,
            )
        )

        consensus = _wfs_v1_build_consensus(
            structured,
            secondary_latest,
            secondary_roster,
            season,
            week,
        )

        consensus = (
            _gav2_expand_global_consensus(
                conn,
                consensus,
                season,
                week,
            )
        )

        consensus = espn.attach_lineage(consensus, espn_evidence)

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

    print(
        "GLOBAL_AVAILABILITY_ROWS =",
        len(consensus),
    )

    print(
        "GLOBAL_AVAILABILITY_UNIQUE_GSIS =",
        consensus["gsis_id"].nunique(),
    )

    print(
        "GLOBAL_HARD_BLOCK_ROWS =",
        int(
            consensus["injury_gate"]
            .eq("BLOCK")
            .sum()
        ),
    )

    print(
        "GLOBAL_ROSTER_BLOCK_ROWS =",
        int(
            (
                consensus["authoritative_source"]
                .eq("CURRENT_WEEKLY_ROSTER")
                & consensus["injury_gate"]
                .eq("BLOCK")
            ).sum()
        ),
    )

    print(
        "WFS_GLOBAL_PLAYER_AVAILABILITY_V2_STATUS=PASS"
    )


if __name__ == "__main__":
    main()
