#!/usr/bin/env python3
"""
fanduel_slate_ingest_v2.py

Dynamic FanDuel NFL slate ingestion with schedule-authoritative game identity.

Core guarantees
---------------
- Positive FanDuel salary defines contest eligibility.
- Blank / N/A salary rows remain source-reference rows and are excluded from
  the optimizer contest pool.
- FanDuel salary is authoritative for contest membership and salary.
- Every salary-bearing contest row must resolve to exactly one NFL schedule
  game before the combined contest pool is written.
- Each individual slate file must resolve to exactly one season/week.
- Exact schedule identity is persisted as season, week, game_id, game_date,
  game_time, schedule_status, and schedule_match_count.
- D/ST coverage is required for every team in each actual slate.
- Duplicate/misnamed copies are detected deterministically and excluded.
- Source "fantasy" is retained only as an external comparison field.
- Existing SQLite pool/manifest outputs are replaced only after every PRIMARY
  slate passes the complete structural + schedule-identity audit.

Default source directory
------------------------
    data/fanduel/slates/

Optional explicit directory
---------------------------
    python fanduel_slate_ingest_v2.py /path/to/slate/files

Outputs
-------
SQLite:
    fanduel_slate_pool
    fanduel_slate_manifest

CSV:
    data/csv/fanduel_slate_manifest.csv
    data/csv/audit_fanduel_slate_ingest.csv
    data/csv/fanduel_slates/<slug>.csv

Parquet:
    data/parquet/fanduel_slates/<slug>.parquet
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, DATA_DIR, CSV_DIR, PARQUET_DIR
from scripts.fanduel_schedule_identity import (
    attach_schedule_identity,
    load_schedule,
)


DEFAULT_SLATE_DIR = Path(DATA_DIR) / "fanduel" / "slates"

CSV_OUT_DIR = Path(CSV_DIR) / "fanduel_slates"
PARQUET_OUT_DIR = Path(PARQUET_DIR) / "fanduel_slates"

MANIFEST_CSV = Path(CSV_DIR) / "fanduel_slate_manifest.csv"
AUDIT_CSV = Path(CSV_DIR) / "audit_fanduel_slate_ingest.csv"

POOL_TABLE = "fanduel_slate_pool"
MANIFEST_TABLE = "fanduel_slate_manifest"


TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "la chargers": "LAC",
    "los angeles rams": "LA",
    "la rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "ny giants": "NYG",
    "new york jets": "NYJ",
    "ny jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

ABBR_ALIASES = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LV": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "LAC": "LAC",
    "SD": "LAC",
    "SDG": "LAC",
    "LA": "LA",
    "LAR": "LA",
    "STL": "LA",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SF": "SF",
    "SFO": "SF",
    "SEA": "SEA",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}


def section(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    ).encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii",
    )

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_abbr(value) -> str | None:
    if pd.isna(value):
        return None

    raw = str(value).strip().upper()
    raw = re.sub(r"[^A-Z]", "", raw)

    return ABBR_ALIASES.get(raw)


def resolve_team(value) -> str | None:
    abbr = normalize_abbr(value)
    if abbr is not None:
        return abbr

    text = normalize_text(value)

    if text in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[text]

    for suffix in (
        " d st",
        " dst",
        " defense",
        " def",
    ):
        if text.endswith(suffix):
            stripped = text[: -len(suffix)].strip()

            if stripped in TEAM_NAME_TO_ABBR:
                return TEAM_NAME_TO_ABBR[stripped]

    return None


def slugify(value: str) -> str:
    text = normalize_text(value)

    if not text:
        return "slate"

    return re.sub(
        r"\s+",
        "_",
        text,
    )


def parse_salary(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def parse_game_info(value) -> tuple[str | None, str | None]:
    text = str(value).strip().upper()

    match = re.match(
        r"^\s*([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\s*$",
        text,
    )

    if match is None:
        return None, None

    away = normalize_abbr(
        match.group(1)
    )

    home = normalize_abbr(
        match.group(2)
    )

    return away, home


def is_dst_name(value) -> bool:
    text = normalize_text(value)

    return bool(
        re.search(
            r"(?:^| )d st(?:$| )",
            text,
        )
        or text.endswith(" dst")
        or text.endswith(" defense")
    )


def discover_files(source_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in source_dir.glob("*.csv")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No slate CSV files found in {source_dir}"
        )

    return files


def normalize_source_file(
    path: Path,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "player",
        "team",
        "gameInfo",
        "salary",
    }

    missing = sorted(
        required - set(raw.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path.name} missing required columns: "
            + ", ".join(missing)
        )

    out = raw.copy()

    out["slate_name"] = path.stem
    out["slate_slug"] = slugify(
        path.stem
    )
    out["source_file"] = path.name
    out["source_row"] = np.arange(
        1,
        len(out) + 1,
    )

    out["salary"] = parse_salary(
        out["salary"]
    )

    out["salary_valid"] = (
        out["salary"].notna()
        & out["salary"].gt(0)
    ).astype(int)

    out["team_internal"] = (
        out["team"]
        .map(resolve_team)
    )

    parsed_games = (
        out["gameInfo"]
        .map(parse_game_info)
    )

    out["away_team"] = [
        pair[0]
        for pair in parsed_games
    ]

    out["home_team"] = [
        pair[1]
        for pair in parsed_games
    ]

    out["game_key"] = out.apply(
        lambda row: (
            f"{row['away_team']}@{row['home_team']}"
            if (
                pd.notna(row["away_team"])
                and pd.notna(row["home_team"])
            )
            else ""
        ),
        axis=1,
    )

    out["is_dst"] = (
        out["player"]
        .map(is_dst_name)
        .astype(int)
    )

    if "fantasy" in out.columns:
        out["source_fantasy_projection"] = (
            pd.to_numeric(
                out["fantasy"],
                errors="coerce",
            )
        )
    else:
        out["source_fantasy_projection"] = np.nan

    out["player_normalized"] = (
        out["player"]
        .map(normalize_text)
    )

    # Positive FanDuel salary is the contest-universe gate.
    out["contest_eligible"] = (
        out["salary_valid"].eq(1)
    ).astype(int)

    # ---------------------------------------------------------
    # AUTHORITATIVE NFL SCHEDULE IDENTITY
    # ---------------------------------------------------------
    out = attach_schedule_identity(
        out,
        schedule,
        away_column="away_team",
        home_column="home_team",
    )

    # Human-readable deterministic key for downstream versioning/audits.
    out["slate_key"] = out.apply(
        lambda row: (
            f"{int(row['season'])}:"
            f"{int(row['week']):02d}:"
            f"{row['slate_slug']}"
            if (
                row["schedule_status"] == "RESOLVED"
                and pd.notna(row["season"])
                and pd.notna(row["week"])
            )
            else ""
        ),
        axis=1,
    )

    return out


def build_contest_pool(
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    pool = source_df[
        source_df["contest_eligible"].eq(1)
    ].copy()

    pool["optimizer_source_eligible"] = 1

    return pool.reset_index(
        drop=True
    )


def slate_fingerprint(
    pool: pd.DataFrame,
) -> str:
    pieces = []

    for row in pool.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team_internal",
            "player_normalized",
            "salary",
        ]
    ).itertuples(index=False):
        pieces.append(
            "|".join(
                [
                    str(row.season),
                    str(row.week),
                    str(row.game_id),
                    str(row.team_internal),
                    str(row.player_normalized),
                    (
                        f"{float(row.salary):.2f}"
                        if pd.notna(row.salary)
                        else "NO_SALARY"
                    ),
                ]
            )
        )

    payload = "\n".join(
        pieces
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def audit_slate(
    source_df: pd.DataFrame,
    pool: pd.DataFrame,
) -> dict:
    source_games = sorted(
        game
        for game in source_df[
            "game_key"
        ].unique()
        if game
    )

    pool_games = sorted(
        game
        for game in pool[
            "game_key"
        ].unique()
        if game
    )

    # Schedule identity is a property of the slate source itself, not only
    # of salary-bearing contest rows. This allows a future FanDuel slate
    # with all salary=N/A to be retained as staged inference inventory
    # without making those players optimizer-eligible.
    source_resolved = source_df[
        source_df[
            "schedule_status"
        ].eq("RESOLVED")
    ].copy()

    source_unresolved_schedule_rows = int(
        (
            ~source_df[
                "schedule_status"
            ].eq("RESOLVED")
        ).sum()
    )

    source_ambiguous_schedule_rows = int(
        source_df[
            "schedule_status"
        ].eq("AMBIGUOUS").sum()
    )

    source_invalid_schedule_rows = int(
        source_df[
            "schedule_status"
        ].eq("INVALID_MATCHUP").sum()
    )

    identity_pairs = (
        source_resolved[
            [
                "season",
                "week",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "week",
            ]
        )
    )

    schedule_identity_count = int(
        len(identity_pairs)
    )

    if schedule_identity_count == 1:
        slate_season = int(
            identity_pairs.iloc[0][
                "season"
            ]
        )
        slate_week = int(
            identity_pairs.iloc[0][
                "week"
            ]
        )
        slate_key = (
            f"{slate_season}:"
            f"{slate_week:02d}:"
            f"{source_df['slate_slug'].iloc[0]}"
        )
    else:
        slate_season = np.nan
        slate_week = np.nan
        slate_key = ""

    source_schedule_game_count = int(
        source_resolved[
            "game_id"
        ].dropna().nunique()
    )

    schedule_game_list = ";".join(
        sorted(
            str(value)
            for value in source_resolved[
                "game_id"
            ].dropna().unique()
            if str(value).strip()
        )
    )

    game_teams = (
        set(
            pool["away_team"]
            .dropna()
        )
        | set(
            pool["home_team"]
            .dropna()
        )
    )

    dst = pool[
        pool["is_dst"].eq(1)
    ].copy()

    dst_teams = set(
        dst["team_internal"]
        .dropna()
    )

    eligible_unresolved_team = int(
        pool[
            "team_internal"
        ].isna().sum()
    )

    eligible_invalid_gameinfo = int(
        (
            pool["away_team"].isna()
            | pool["home_team"].isna()
        ).sum()
    )

    invalid_salary_in_pool = int(
        (
            pool["salary"].isna()
            | pool["salary"].le(0)
        ).sum()
    )

    duplicate_player_team_salary = int(
        pool.duplicated(
            [
                "player_normalized",
                "team_internal",
                "salary",
                "game_id",
            ]
        ).sum()
    )

    contest_unresolved_schedule_rows = int(
        (
            ~pool[
                "schedule_status"
            ].eq("RESOLVED")
        ).sum()
    )

    contest_schedule_resolved_rows = int(
        pool[
            "schedule_status"
        ].eq("RESOLVED").sum()
    )

    # A zero-salary source must remain inference-only. It is explicitly
    # staged, never solver-eligible.
    staged_no_salary = int(
        len(source_df) > 0
        and len(pool) == 0
        and source_df[
            "salary_valid"
        ].eq(0).all()
    )

    fingerprint_df = (
        pool
        if not pool.empty
        else source_df
    )

    return {
        "slate_name": source_df[
            "slate_name"
        ].iloc[0],
        "slate_slug": source_df[
            "slate_slug"
        ].iloc[0],
        "source_file": source_df[
            "source_file"
        ].iloc[0],

        "season": slate_season,
        "week": slate_week,
        "slate_key": slate_key,

        "source_rows": len(
            source_df
        ),
        "contest_rows": len(
            pool
        ),
        "excluded_no_salary_rows": (
            len(source_df)
            - len(pool)
        ),
        "staged_no_salary": staged_no_salary,

        "source_games": len(
            source_games
        ),
        "games": len(
            pool_games
        ),
        "teams": len(
            game_teams
        ),

        "schedule_identity_count": (
            schedule_identity_count
        ),
        "schedule_game_count": (
            source_schedule_game_count
        ),
        "source_schedule_resolved_rows": int(
            source_df[
                "schedule_status"
            ].eq("RESOLVED").sum()
        ),
        "contest_schedule_resolved_rows": (
            contest_schedule_resolved_rows
        ),
        "unresolved_schedule_rows": (
            source_unresolved_schedule_rows
        ),
        "ambiguous_schedule_rows": (
            source_ambiguous_schedule_rows
        ),
        "invalid_schedule_rows": (
            source_invalid_schedule_rows
        ),
        "contest_unresolved_schedule_rows": (
            contest_unresolved_schedule_rows
        ),

        "dst_rows": len(
            dst
        ),
        "dst_team_coverage": len(
            dst_teams
        ),
        "dst_valid_salary_rows": int(
            dst["salary_valid"].sum()
        ),

        "eligible_unresolved_team_rows": (
            eligible_unresolved_team
        ),
        "eligible_invalid_gameinfo_rows": (
            eligible_invalid_gameinfo
        ),
        "invalid_salary_in_contest_pool": (
            invalid_salary_in_pool
        ),
        "duplicate_contest_rows": (
            duplicate_player_team_salary
        ),

        "all_game_teams_have_dst": int(
            dst_teams == game_teams
        ),

        "game_list": ";".join(
            source_games
        ),
        "schedule_game_list": (
            schedule_game_list
        ),

        "fingerprint": slate_fingerprint(
            fingerprint_df
        ),
    }

def classify_duplicates(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    manifest = manifest.copy()

    manifest[
        "duplicate_of"
    ] = ""

    manifest[
        "source_status"
    ] = "PRIMARY"

    seen = {}

    # Files are discovered alphabetically. First exact fingerprint becomes
    # primary; later identical contest universes are excluded deterministically.
    for idx, row in manifest.iterrows():
        fingerprint = row[
            "fingerprint"
        ]

        if fingerprint not in seen:
            seen[fingerprint] = (
                row["slate_name"]
            )
            continue

        manifest.at[
            idx,
            "duplicate_of",
        ] = seen[
            fingerprint
        ]

        manifest.at[
            idx,
            "source_status",
        ] = "DUPLICATE_SOURCE"

    return manifest


def assign_structural_status(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    manifest = manifest.copy()

    primary = manifest[
        "source_status"
    ].eq("PRIMARY")

    schedule_source_ok = (
        manifest[
            "source_rows"
        ].gt(0)
        & manifest[
            "source_games"
        ].gt(0)
        & manifest[
            "unresolved_schedule_rows"
        ].eq(0)
        & manifest[
            "ambiguous_schedule_rows"
        ].eq(0)
        & manifest[
            "invalid_schedule_rows"
        ].eq(0)
        & manifest[
            "schedule_identity_count"
        ].eq(1)
        & manifest[
            "schedule_game_count"
        ].eq(
            manifest[
                "source_games"
            ]
        )
        & manifest[
            "source_schedule_resolved_rows"
        ].eq(
            manifest[
                "source_rows"
            ]
        )
    )

    contest_ok = (
        manifest[
            "contest_rows"
        ].gt(0)
        & manifest[
            "games"
        ].gt(0)
        & manifest[
            "eligible_unresolved_team_rows"
        ].eq(0)
        & manifest[
            "eligible_invalid_gameinfo_rows"
        ].eq(0)
        & manifest[
            "invalid_salary_in_contest_pool"
        ].eq(0)
        & manifest[
            "duplicate_contest_rows"
        ].eq(0)
        & manifest[
            "contest_unresolved_schedule_rows"
        ].eq(0)
        & manifest[
            "contest_schedule_resolved_rows"
        ].eq(
            manifest[
                "contest_rows"
            ]
        )
        & manifest[
            "dst_rows"
        ].eq(
            manifest[
                "teams"
            ]
        )
        & manifest[
            "dst_team_coverage"
        ].eq(
            manifest[
                "teams"
            ]
        )
        & manifest[
            "dst_valid_salary_rows"
        ].eq(
            manifest[
                "dst_rows"
            ]
        )
        & manifest[
            "all_game_teams_have_dst"
        ].eq(1)
    )

    staged_ok = (
        manifest[
            "staged_no_salary"
        ].eq(1)
        & manifest[
            "contest_rows"
        ].eq(0)
    )

    manifest[
        "structural_status"
    ] = np.where(
        ~primary,
        "DUPLICATE_SOURCE",
        np.where(
            schedule_source_ok & contest_ok,
            "PASS",
            np.where(
                schedule_source_ok & staged_ok,
                "STAGED_NO_SALARY",
                "FAIL",
            ),
        ),
    )

    return manifest

def write_sqlite(
    pool: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:
        pool.to_sql(
            POOL_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        manifest.to_sql(
            MANIFEST_TABLE,
            conn,
            if_exists="replace",
            index=False,
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            idx_{POOL_TABLE}_slate
            ON {POOL_TABLE}(slate_slug)
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            idx_{POOL_TABLE}_slate_team
            ON {POOL_TABLE}(slate_slug, team_internal)
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            idx_{POOL_TABLE}_slate_game
            ON {POOL_TABLE}(slate_slug, game_id)
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            idx_{POOL_TABLE}_season_week
            ON {POOL_TABLE}(season, week)
            """
        )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS
            idx_{POOL_TABLE}_slate_key
            ON {POOL_TABLE}(slate_key)
            """
        )

        conn.commit()


def main() -> None:
    source_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_SLATE_DIR
    )

    section(
        "FANDUEL NFL DYNAMIC SLATE INGEST"
    )

    print(
        f"Source directory: {source_dir}"
    )
    print(
        f"Database: {DATABASE_PATH}"
    )
    print(
        "Positive FanDuel salary = contest eligibility"
    )
    print(
        "N/A / blank salary rows = excluded source-reference rows"
    )
    print(
        "Source fantasy projection = audit only"
    )

    schedule, schedule_path = load_schedule()

    print(
        f"Schedule authority: {schedule_path}"
    )

    files = discover_files(
        source_dir
    )

    print(
        f"Discovered slate files: {len(files)}"
    )

    CSV_OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARQUET_OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_frames = []
    contest_pools = []
    audits = []

    for path in files:
        source_df = normalize_source_file(
            path,
            schedule,
        )

        pool = build_contest_pool(
            source_df
        )

        audit = audit_slate(
            source_df,
            pool,
        )

        source_frames.append(
            source_df
        )

        contest_pools.append(
            pool
        )

        audits.append(
            audit
        )

    manifest = pd.DataFrame(
        audits
    )

    manifest = classify_duplicates(
        manifest
    )

    manifest = assign_structural_status(
        manifest
    )

    section(
        "SLATE MANIFEST / AUDIT"
    )

    display_columns = [
        "slate_name",
        "season",
        "week",
        "source_rows",
        "contest_rows",
        "excluded_no_salary_rows",
        "staged_no_salary",
        "source_games",
        "games",
        "schedule_game_count",
        "teams",
        "dst_rows",
        "dst_valid_salary_rows",
        "eligible_unresolved_team_rows",
        "eligible_invalid_gameinfo_rows",
        "unresolved_schedule_rows",
        "ambiguous_schedule_rows",
        "schedule_identity_count",
        "source_status",
        "structural_status",
        "duplicate_of",
    ]

    print(
        manifest[
            display_columns
        ].to_string(
            index=False
        )
    )

    hard_failures = manifest[
        manifest[
            "structural_status"
        ].eq("FAIL")
    ]

    if not hard_failures.empty:
        manifest.to_csv(
            AUDIT_CSV,
            index=False,
        )

        raise RuntimeError(
            "One or more PRIMARY slate files failed structural/schedule audit. "
            "PASS and STAGED_NO_SALARY inventory was not written."
        )

    primary_names = set(
        manifest.loc[
            manifest[
                "structural_status"
            ].eq("PASS"),
            "slate_name",
        ]
    )

    primary_pools = [
        pool
        for pool in contest_pools
        if pool[
            "slate_name"
        ].iloc[0] in primary_names
    ]

    if not primary_pools:
        raise RuntimeError(
            "No primary slate pools remained after duplicate filtering."
        )

    combined_pool = pd.concat(
        primary_pools,
        ignore_index=True,
    )

    # Re-export only primary contest pools.
    for pool in primary_pools:
        slug = pool[
            "slate_slug"
        ].iloc[0]

        pool.to_csv(
            CSV_OUT_DIR
            / f"{slug}.csv",
            index=False,
        )

        pool.to_parquet(
            PARQUET_OUT_DIR
            / f"{slug}.parquet",
            index=False,
        )

    # SQLite replacement occurs only after every PRIMARY slate passed.
    write_sqlite(
        combined_pool,
        manifest,
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    manifest.to_csv(
        AUDIT_CSV,
        index=False,
    )

    section(
        "PRIMARY SLATE INVENTORY"
    )

    primary_manifest = manifest[
        manifest[
            "structural_status"
        ].eq("PASS")
    ].copy()

    for row in primary_manifest.itertuples(
        index=False
    ):
        print(
            f"{row.slate_name:<20} "
            f"season={int(row.season)} "
            f"week={int(row.week):>2} "
            f"games={row.games:>2} "
            f"teams={row.teams:>2} "
            f"contest_players={row.contest_rows:>3} "
            f"D/ST={row.dst_rows:>2}/{row.dst_valid_salary_rows:>2} "
            f"PASS"
        )

        print(
            f"  slate_key: {row.slate_key}"
        )

        print(
            f"  excluded N/A salary rows: "
            f"{row.excluded_no_salary_rows}"
        )

        print(
            f"  matchups: {row.game_list}"
        )

        print(
            f"  game_ids: {row.schedule_game_list}"
        )

    staged_manifest = manifest[
        manifest[
            "structural_status"
        ].eq(
            "STAGED_NO_SALARY"
        )
    ].copy()

    if not staged_manifest.empty:
        section(
            "STAGED FUTURE SLATES - NO FANDUEL SALARIES YET"
        )

        for row in staged_manifest.itertuples(
            index=False
        ):
            print(
                f"{row.slate_name:<20} "
                f"season={int(row.season)} "
                f"week={int(row.week):>2} "
                f"source_games={row.source_games:>2} "
                f"source_rows={row.source_rows:>3} "
                f"contest_rows={row.contest_rows:>3} "
                f"STAGED_NO_SALARY"
            )

            print(
                f"  slate_key: {row.slate_key}"
            )

            print(
                f"  game_ids: {row.schedule_game_list}"
            )

            print(
                "  Solver eligibility: NONE until positive FanDuel salaries arrive."
            )

    duplicate_manifest = manifest[
        manifest[
            "structural_status"
        ].eq(
            "DUPLICATE_SOURCE"
        )
    ]

    if not duplicate_manifest.empty:
        section(
            "DUPLICATE SOURCE FILES EXCLUDED"
        )

        for row in duplicate_manifest.itertuples(
            index=False
        ):
            print(
                f"{row.slate_name} -> exact contest-pool duplicate of "
                f"{row.duplicate_of}"
            )

    section(
        "EXPORTS"
    )

    print(
        f"Manifest: {MANIFEST_CSV}"
    )
    print(
        f"Audit: {AUDIT_CSV}"
    )
    print(
        f"Per-slate CSVs: {CSV_OUT_DIR}"
    )
    print(
        f"Per-slate Parquet: {PARQUET_OUT_DIR}"
    )
    print(
        f"SQLite pool table: {POOL_TABLE}"
    )
    print(
        f"SQLite manifest table: {MANIFEST_TABLE}"
    )

    section(
        "FANDUEL SLATE INGEST COMPLETE"
    )

    print(
        f"Source files discovered: {len(manifest)}"
    )

    print(
        f"Salary-bearing primary slates loaded: {len(primary_manifest)}"
    )

    print(
        f"Staged no-salary slates retained in manifest: {len(staged_manifest)}"
    )

    print(
        f"Duplicate sources excluded: {len(duplicate_manifest)}"
    )

    print(
        f"Combined salary-bearing contest rows: {len(combined_pool)}"
    )

    print(
        "All source slates passed schedule identity; salary-bearing slates also passed "
        "salary/team/game/D-ST contest gates."
    )

    print(
        "Next layer: use manifest season/week/game IDs for staged inference targets, "
        "while attaching optimizer projections only to salary-bearing contest rows."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 104)
        print(
            "FANDUEL SLATE INGEST FAILED"
        )
        print("=" * 104)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)
