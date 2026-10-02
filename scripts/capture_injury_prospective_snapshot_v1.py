#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


VERSION = "INJURY_PROSPECTIVE_SNAPSHOT_V1"

DEFAULT_DB = Path("data/nfl.db")

CAPTURE_TABLE = "injury_prospective_capture_v1"
PLAYER_TABLE = "injury_prospective_player_snapshot_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def table_exists(
    conn: sqlite3.Connection,
    table: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table,),
    ).fetchone()

    return row is not None


def table_columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:
    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return {str(row[1]) for row in rows}


def require_columns(
    conn: sqlite3.Connection,
    table: str,
    required: set[str],
) -> None:
    existing = table_columns(
        conn,
        table,
    )

    missing = sorted(
        required - existing
    )

    if missing:
        fail(
            f"{table} missing required columns: "
            + ", ".join(missing)
        )


def create_tables(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {CAPTURE_TABLE} (
            capture_id TEXT PRIMARY KEY,

            version TEXT NOT NULL,

            season INTEGER NOT NULL,
            week INTEGER NOT NULL,

            captured_at_utc TEXT NOT NULL,

            game_count INTEGER NOT NULL,
            player_row_count INTEGER NOT NULL,

            source_table TEXT NOT NULL,
            source_content_sha256 TEXT NOT NULL,

            capture_status TEXT NOT NULL,

            kickoff_utc_verified INTEGER NOT NULL
                DEFAULT 0,

            historical_reconstruction INTEGER NOT NULL
                DEFAULT 0,

            forecast_mutation INTEGER NOT NULL
                DEFAULT 0,

            player_projection_mutation INTEGER NOT NULL
                DEFAULT 0,

            solver_influence INTEGER NOT NULL
                DEFAULT 0,

            wfs_influence INTEGER NOT NULL
                DEFAULT 0,

            production_influence_allowed INTEGER NOT NULL
                DEFAULT 0
        )
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {PLAYER_TABLE} (
            capture_id TEXT NOT NULL,

            version TEXT NOT NULL,

            captured_at_utc TEXT NOT NULL,

            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT,

            game_date TEXT,
            gametime TEXT,

            team TEXT NOT NULL,
            source_team TEXT NOT NULL,
            opponent_team TEXT,

            home_team TEXT,
            away_team TEXT,

            gsis_id TEXT NOT NULL,
            position TEXT,
            player_name TEXT,

            report_status_raw TEXT,
            report_status TEXT,

            practice_status_raw TEXT,
            practice_status TEXT,

            primary_injury TEXT,

            structured_source TEXT,

            news_signal_count INTEGER,

            consensus_status TEXT,

            injury_gate TEXT,
            injury_gate_reason TEXT,

            availability_risk TEXT,

            authoritative_source TEXT,

            consensus_updated_at TEXT,

            secondary_practice_status TEXT,
            secondary_availability_status TEXT,
            secondary_signal_strength TEXT,
            secondary_source TEXT,
            secondary_source_timestamp TEXT,
            secondary_headline TEXT,

            structured_present INTEGER,
            secondary_present INTEGER,

            consensus_practice_status TEXT,
            secondary_injury TEXT,

            injury_risk TEXT,

            consensus_last_updated TEXT,

            espn_athlete_id TEXT,
            espn_injury_record_id TEXT,
            espn_status TEXT,
            espn_record_date TEXT,
            espn_feed_timestamp TEXT,
            espn_team TEXT,
            espn_position TEXT,
            espn_mapped_gsis_id TEXT,
            espn_identity_result TEXT,
            espn_source TEXT,
            espn_authority_decision TEXT,

            source_timestamp_available INTEGER
                NOT NULL DEFAULT 0,

            source_timestamp TEXT,

            source_timestamp_kind TEXT NOT NULL,

            snapshot_before_kickoff_verified INTEGER
                NOT NULL DEFAULT 0,

            historical_reconstruction INTEGER
                NOT NULL DEFAULT 0,

            production_influence_allowed INTEGER
                NOT NULL DEFAULT 0,

            PRIMARY KEY (
                capture_id,
                game_id,
                team,
                gsis_id
            ),

            FOREIGN KEY (capture_id)
                REFERENCES {CAPTURE_TABLE}(capture_id)
        )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_injury_snapshot_game
        ON {PLAYER_TABLE} (
            game_id,
            captured_at_utc
        )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_injury_snapshot_player
        ON {PLAYER_TABLE} (
            gsis_id,
            captured_at_utc
        )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_injury_snapshot_week
        ON {PLAYER_TABLE} (
            season,
            week,
            captured_at_utc
        )
        """
    )


def load_games(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> pd.DataFrame:
    games = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            game_type,
            game_date,
            gametime,
            away_team,
            home_team,
            completed
        FROM games
        WHERE season = ?
          AND week = ?
          AND game_type = 'REG'
        ORDER BY
            game_date,
            gametime,
            game_id
        """,
        conn,
        params=(
            season,
            week,
        ),
    )

    if games.empty:
        fail(
            f"No REG games found for "
            f"season={season} week={week}"
        )

    if games["game_id"].duplicated().any():
        fail(
            "Duplicate game_id found in games source"
        )

    return games


def load_consensus(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> pd.DataFrame:
    injuries = pd.read_sql_query(
        """
        SELECT *
        FROM injury_consensus_current
        WHERE season = ?
          AND week = ?
        ORDER BY
            team,
            gsis_id
        """,
        conn,
        params=(
            season,
            week,
        ),
    )

    if injuries.empty:
        fail(
            f"No injury consensus rows found for "
            f"season={season} week={week}"
        )

    key_cols = [
        "season",
        "week",
        "game_type",
        "team",
        "gsis_id",
    ]

    if injuries.duplicated(
        subset=key_cols,
        keep=False,
    ).any():
        dupes = injuries[
            injuries.duplicated(
                subset=key_cols,
                keep=False,
            )
        ][key_cols]

        fail(
            "Duplicate injury consensus identity rows:\n"
            + dupes.to_string(index=False)
        )

    return injuries


def attach_games(
    injuries: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    # Deterministic source-system team aliases used only
    # for schedule attachment. Authoritative source values
    # themselves are never rewritten.
    TEAM_JOIN_ALIASES = {
        "LAR": "LA",
    }

    injuries = injuries.copy()

    # Preserve the exact team identity supplied by
    # injury_consensus_current for provenance.
    injuries["source_team"] = (
        injuries["team"]
        .astype(str)
        .str.strip()
    )

    # Canonical schedule identity used for game attachment.
    injuries["_schedule_team"] = (
        injuries["source_team"]
        .replace(TEAM_JOIN_ALIASES)
    )

    team_rows = []

    for row in games.itertuples(index=False):
        team_rows.append(
            {
                "game_id": row.game_id,
                "season": row.season,
                "week": row.week,
                "game_type": row.game_type,
                "game_date": row.game_date,
                "gametime": row.gametime,
                "_schedule_team": row.away_team,
                "opponent_team": row.home_team,
                "home_team": row.home_team,
                "away_team": row.away_team,
            }
        )

        team_rows.append(
            {
                "game_id": row.game_id,
                "season": row.season,
                "week": row.week,
                "game_type": row.game_type,
                "game_date": row.game_date,
                "gametime": row.gametime,
                "_schedule_team": row.home_team,
                "opponent_team": row.away_team,
                "home_team": row.home_team,
                "away_team": row.away_team,
            }
        )

    team_map = pd.DataFrame(team_rows)

    if team_map.duplicated(
        subset=[
            "season",
            "week",
            "game_type",
            "_schedule_team",
        ]
    ).any():
        fail(
            "Team appears in multiple games for "
            "the requested week"
        )

    out = injuries.merge(
        team_map,
        on=[
            "season",
            "week",
            "game_type",
            "_schedule_team",
        ],
        how="left",
        validate="many_to_one",
    )

    missing = out[
        out["game_id"].isna()
    ]

    if not missing.empty:
        cols = [
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
            "player_name",
        ]

        fail(
            "Injury consensus rows could not be "
            "mapped to a scheduled game:\n"
            + missing[cols]
            .head(50)
            .to_string(index=False)
        )

    # Downstream snapshot identity uses the canonical
    # schedule namespace while source_team retains the
    # original injury-consensus namespace.
    out["team"] = out["_schedule_team"]

    out = out.drop(
        columns=["_schedule_team"]
    )

    return out


def choose_source_timestamp(
    row: pd.Series,
) -> tuple[str, str]:
    candidates = [
        (
            "ESPN_FEED_TIMESTAMP",
            clean_text(
                row.get(
                    "espn_feed_timestamp"
                )
            ),
        ),
        (
            "SECONDARY_SOURCE_TIMESTAMP",
            clean_text(
                row.get(
                    "secondary_source_timestamp"
                )
            ),
        ),
    ]

    for kind, value in candidates:
        if value:
            return value, kind

    return "", "CAPTURE_ONLY"


def canonical_content_hash(
    frame: pd.DataFrame,
) -> str:
    hash_columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "game_date",
        "gametime",
        "team",
        "source_team",
        "opponent_team",
        "home_team",
        "away_team",
        "gsis_id",
        "position",
        "player_name",
        "report_status_raw",
        "report_status",
        "practice_status_raw",
        "practice_status",
        "primary_injury",
        "structured_source",
        "news_signal_count",
        "consensus_status",
        "injury_gate",
        "injury_gate_reason",
        "availability_risk",
        "authoritative_source",
        "updated_at",
        "secondary_practice_status",
        "secondary_availability_status",
        "secondary_signal_strength",
        "secondary_source",
        "secondary_source_timestamp",
        "secondary_headline",
        "structured_present",
        "secondary_present",
        "consensus_practice_status",
        "secondary_injury",
        "injury_risk",
        "last_updated",
        "espn_athlete_id",
        "espn_injury_record_id",
        "espn_status",
        "espn_record_date",
        "espn_feed_timestamp",
        "espn_team",
        "espn_position",
        "espn_mapped_gsis_id",
        "espn_identity_result",
        "espn_source",
        "espn_authority_decision",
    ]

    existing = [
        col
        for col in hash_columns
        if col in frame.columns
    ]

    canonical = frame[
        existing
    ].copy()

    canonical = canonical.fillna("")

    canonical = canonical.astype(str)

    canonical = canonical.sort_values(
        by=[
            "game_id",
            "team",
            "gsis_id",
        ],
        kind="mergesort",
    )

    payload = canonical.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def prepare_player_rows(
    frame: pd.DataFrame,
    capture_id: str,
    captured_at_utc: str,
) -> pd.DataFrame:
    out = frame.copy()

    source_values = out.apply(
        choose_source_timestamp,
        axis=1,
        result_type="expand",
    )

    source_values.columns = [
        "source_timestamp",
        "source_timestamp_kind",
    ]

    out[
        "source_timestamp"
    ] = source_values[
        "source_timestamp"
    ]

    out[
        "source_timestamp_kind"
    ] = source_values[
        "source_timestamp_kind"
    ]

    out[
        "source_timestamp_available"
    ] = (
        out["source_timestamp"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    out["capture_id"] = capture_id
    out["version"] = VERSION

    out[
        "captured_at_utc"
    ] = captured_at_utc

    # IMPORTANT:
    #
    # games.game_date + games.gametime do not
    # establish an authoritative UTC kickoff.
    #
    # Therefore V1 does NOT claim that capture
    # timing has been verified against kickoff.
    #
    # A downstream checkpoint controller may
    # later attach authoritative kickoff UTC
    # evidence without rewriting this snapshot.

    out[
        "snapshot_before_kickoff_verified"
    ] = 0

    out[
        "historical_reconstruction"
    ] = 0

    out[
        "production_influence_allowed"
    ] = 0

    rename = {
        "updated_at":
            "consensus_updated_at",
        "last_updated":
            "consensus_last_updated",
    }

    out = out.rename(
        columns=rename
    )

    player_columns = [
        "capture_id",
        "version",
        "captured_at_utc",
        "game_id",
        "season",
        "week",
        "game_type",
        "game_date",
        "gametime",
        "team",
        "source_team",
        "opponent_team",
        "home_team",
        "away_team",
        "gsis_id",
        "position",
        "player_name",
        "report_status_raw",
        "report_status",
        "practice_status_raw",
        "practice_status",
        "primary_injury",
        "structured_source",
        "news_signal_count",
        "consensus_status",
        "injury_gate",
        "injury_gate_reason",
        "availability_risk",
        "authoritative_source",
        "consensus_updated_at",
        "secondary_practice_status",
        "secondary_availability_status",
        "secondary_signal_strength",
        "secondary_source",
        "secondary_source_timestamp",
        "secondary_headline",
        "structured_present",
        "secondary_present",
        "consensus_practice_status",
        "secondary_injury",
        "injury_risk",
        "consensus_last_updated",
        "espn_athlete_id",
        "espn_injury_record_id",
        "espn_status",
        "espn_record_date",
        "espn_feed_timestamp",
        "espn_team",
        "espn_position",
        "espn_mapped_gsis_id",
        "espn_identity_result",
        "espn_source",
        "espn_authority_decision",
        "source_timestamp_available",
        "source_timestamp",
        "source_timestamp_kind",
        "snapshot_before_kickoff_verified",
        "historical_reconstruction",
        "production_influence_allowed",
    ]

    missing = [
        col
        for col in player_columns
        if col not in out.columns
    ]

    if missing:
        fail(
            "Prepared snapshot missing columns: "
            + ", ".join(missing)
        )

    out = out[
        player_columns
    ].copy()

    return out


def insert_capture(
    conn: sqlite3.Connection,
    capture_id: str,
    season: int,
    week: int,
    captured_at_utc: str,
    game_count: int,
    player_row_count: int,
    source_hash: str,
) -> None:
    conn.execute(
        f"""
        INSERT INTO {CAPTURE_TABLE} (
            capture_id,
            version,
            season,
            week,
            captured_at_utc,
            game_count,
            player_row_count,
            source_table,
            source_content_sha256,
            capture_status,
            kickoff_utc_verified,
            historical_reconstruction,
            forecast_mutation,
            player_projection_mutation,
            solver_influence,
            wfs_influence,
            production_influence_allowed
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            0, 0, 0, 0, 0, 0, 0
        )
        """,
        (
            capture_id,
            VERSION,
            season,
            week,
            captured_at_utc,
            game_count,
            player_row_count,
            "injury_consensus_current",
            source_hash,
            "CAPTURED_SHADOW_ONLY",
        ),
    )


def insert_players(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
) -> None:
    columns = list(
        frame.columns
    )

    quoted = ", ".join(
        f'"{col}"'
        for col in columns
    )

    placeholders = ", ".join(
        "?"
        for _ in columns
    )

    sql = (
        f"INSERT INTO {PLAYER_TABLE} "
        f"({quoted}) "
        f"VALUES ({placeholders})"
    )

    records = []

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        cleaned = []

        for value in row:
            try:
                if pd.isna(value):
                    value = None
            except Exception:
                pass

            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass

            cleaned.append(value)

        records.append(
            tuple(cleaned)
        )

    conn.executemany(
        sql,
        records,
    )


def validate_written_capture(
    conn: sqlite3.Connection,
    capture_id: str,
    expected_rows: int,
    expected_games: int,
) -> None:
    capture = pd.read_sql_query(
        f"""
        SELECT *
        FROM {CAPTURE_TABLE}
        WHERE capture_id = ?
        """,
        conn,
        params=(capture_id,),
    )

    if len(capture) != 1:
        fail(
            "Capture metadata row validation failed"
        )

    players = pd.read_sql_query(
        f"""
        SELECT *
        FROM {PLAYER_TABLE}
        WHERE capture_id = ?
        """,
        conn,
        params=(capture_id,),
    )

    if len(players) != expected_rows:
        fail(
            "Snapshot player row count mismatch: "
            f"expected={expected_rows} "
            f"actual={len(players)}"
        )

    game_count = int(
        players["game_id"].nunique()
    )

    if game_count != expected_games:
        fail(
            "Snapshot game count mismatch: "
            f"expected={expected_games} "
            f"actual={game_count}"
        )

    if players.duplicated(
        subset=[
            "capture_id",
            "game_id",
            "team",
            "gsis_id",
        ]
    ).any():
        fail(
            "Duplicate snapshot player identity detected"
        )

    safety_columns = [
        "historical_reconstruction",
        "production_influence_allowed",
    ]

    for col in safety_columns:
        if (
            pd.to_numeric(
                players[col],
                errors="coerce",
            )
            .fillna(1)
            .ne(0)
            .any()
        ):
            fail(
                f"Safety validation failed: {col}"
            )

    metadata_safety = [
        "historical_reconstruction",
        "forecast_mutation",
        "player_projection_mutation",
        "solver_influence",
        "wfs_influence",
        "production_influence_allowed",
    ]

    for col in metadata_safety:
        if int(
            capture.iloc[0][col]
        ) != 0:
            fail(
                f"Capture safety validation failed: {col}"
            )


def print_summary(
    conn: sqlite3.Connection,
    capture_id: str,
) -> None:
    capture = pd.read_sql_query(
        f"""
        SELECT *
        FROM {CAPTURE_TABLE}
        WHERE capture_id = ?
        """,
        conn,
        params=(capture_id,),
    )

    players = pd.read_sql_query(
        f"""
        SELECT
            game_id,
            team,
            gsis_id,
            player_name,
            position,
            report_status,
            consensus_status,
            injury_gate,
            availability_risk,
            injury_risk,
            authoritative_source,
            source_timestamp_kind,
            source_timestamp_available,
            snapshot_before_kickoff_verified,
            production_influence_allowed
        FROM {PLAYER_TABLE}
        WHERE capture_id = ?
        ORDER BY
            game_id,
            team,
            position,
            player_name
        """,
        conn,
        params=(capture_id,),
    )

    print("=" * 100)
    print("INJURY PROSPECTIVE SNAPSHOT V1")
    print("=" * 100)

    print("\nCAPTURE:")
    print(
        capture.to_string(
            index=False
        )
    )

    print("\nAUTHORITY DISTRIBUTION:")

    authority = (
        players.groupby(
            "authoritative_source",
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    print(
        authority.to_string(
            index=False
        )
    )

    print("\nINJURY GATE DISTRIBUTION:")

    gates = (
        players.groupby(
            "injury_gate",
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    print(
        gates.to_string(
            index=False
        )
    )

    print("\nSOURCE TIMESTAMP CONTRACT:")

    timestamp_summary = (
        players.groupby(
            [
                "source_timestamp_kind",
                "source_timestamp_available",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            [
                "source_timestamp_kind",
                "source_timestamp_available",
            ]
        )
    )

    print(
        timestamp_summary.to_string(
            index=False
        )
    )

    print("\nSAFETY:")
    print(
        "historical_reconstruction=FALSE"
    )
    print(
        "forecast_mutation=FALSE"
    )
    print(
        "player_projection_mutation=FALSE"
    )
    print(
        "solver_influence=FALSE"
    )
    print(
        "wfs_influence=FALSE"
    )
    print(
        "production_influence_allowed=FALSE"
    )
    print(
        "kickoff_utc_verified=FALSE"
    )

    print("\nSTATUS: PASS_SHADOW_ONLY")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture immutable prospective injury "
            "consensus evidence for research/shadow use."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db_path = args.db.resolve()

    if not db_path.exists():
        fail(
            f"Database not found: {db_path}"
        )

    captured_at_utc = utc_now_iso()

    capture_id = (
        f"injury_v1_"
        f"{args.season}_"
        f"{args.week:02d}_"
        f"{uuid.uuid4().hex}"
    )

    conn = sqlite3.connect(
        str(db_path)
    )

    try:
        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        if not table_exists(
            conn,
            "games",
        ):
            fail(
                "Required table missing: games"
            )

        if not table_exists(
            conn,
            "injury_consensus_current",
        ):
            fail(
                "Required table missing: "
                "injury_consensus_current"
            )

        require_columns(
            conn,
            "games",
            {
                "game_id",
                "season",
                "week",
                "game_type",
                "game_date",
                "gametime",
                "away_team",
                "home_team",
                "completed",
            },
        )

        require_columns(
            conn,
            "injury_consensus_current",
            {
                "season",
                "week",
                "game_type",
                "team",
                "gsis_id",
                "position",
                "player_name",
                "report_status_raw",
                "report_status",
                "practice_status_raw",
                "practice_status",
                "primary_injury",
                "structured_source",
                "news_signal_count",
                "consensus_status",
                "injury_gate",
                "injury_gate_reason",
                "availability_risk",
                "authoritative_source",
                "updated_at",
                "secondary_practice_status",
                "secondary_availability_status",
                "secondary_signal_strength",
                "secondary_source",
                "secondary_source_timestamp",
                "secondary_headline",
                "structured_present",
                "secondary_present",
                "consensus_practice_status",
                "secondary_injury",
                "injury_risk",
                "last_updated",
                "espn_athlete_id",
                "espn_injury_record_id",
                "espn_status",
                "espn_record_date",
                "espn_feed_timestamp",
                "espn_team",
                "espn_position",
                "espn_mapped_gsis_id",
                "espn_identity_result",
                "espn_source",
                "espn_authority_decision",
            },
        )

        games = load_games(
            conn,
            args.season,
            args.week,
        )

        injuries = load_consensus(
            conn,
            args.season,
            args.week,
        )

        source = attach_games(
            injuries,
            games,
        )

        if source.empty:
            fail(
                "No snapshot rows after game mapping"
            )

        source_hash = (
            canonical_content_hash(
                source
            )
        )

        players = prepare_player_rows(
            source,
            capture_id,
            captured_at_utc,
        )

        expected_rows = len(players)

        expected_games = int(
            players[
                "game_id"
            ].nunique()
        )

        # Transaction:
        # Existing production authority is read only.
        # Only the two new V1 snapshot tables are written.

        conn.execute("BEGIN")

        create_tables(conn)

        insert_capture(
            conn=conn,
            capture_id=capture_id,
            season=args.season,
            week=args.week,
            captured_at_utc=captured_at_utc,
            game_count=expected_games,
            player_row_count=expected_rows,
            source_hash=source_hash,
        )

        insert_players(
            conn,
            players,
        )

        validate_written_capture(
            conn,
            capture_id,
            expected_rows,
            expected_games,
        )

        conn.commit()

        print_summary(
            conn,
            capture_id,
        )

        return 0

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
