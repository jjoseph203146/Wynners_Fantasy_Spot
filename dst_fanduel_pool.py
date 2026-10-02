#!/usr/bin/env python3

"""
dst_fanduel_pool.py

Build the FanDuel D/ST player pool by deterministically merging the current
D/ST production projections with the authoritative FanDuel D-ST.csv file.

IMPORTANT
---------
- FanDuel salary is authoritative only from D-ST.csv.
- Salary is NEVER imputed, guessed, averaged, or filled.
- Rows without a valid positive FanDuel salary remain visible for audit but
  are NOT optimizer eligible.
- Production projections come from dst_production_projection.
- This script does not modify frozen historical D/ST tables.
- This script does not modify the existing offensive FanDuel player pool.

Inputs
------
SQLite:
    dst_production_projection

CSV:
    data/fanduel/D-ST.csv

Outputs
-------
SQLite:
    fanduel_dst_pool

CSV:
    data/csv/fanduel_dst_pool.csv
    data/csv/audit_fanduel_dst_pool_summary.csv
    data/csv/audit_fanduel_dst_unmatched.csv

Parquet:
    data/parquet/fanduel_dst_pool.parquet
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR, DATA_DIR


PROJECTION_TABLE = "dst_production_projection"
OUTPUT_TABLE = "fanduel_dst_pool"

FANDUEL_DIR = Path(DATA_DIR) / "fanduel"
FANDUEL_DST_PATH = FANDUEL_DIR / "D-ST.csv"

CSV_OUTPUT = Path(CSV_DIR) / "fanduel_dst_pool.csv"
PARQUET_OUTPUT = Path(PARQUET_DIR) / "fanduel_dst_pool.parquet"
AUDIT_OUTPUT = Path(CSV_DIR) / "audit_fanduel_dst_pool_summary.csv"
UNMATCHED_OUTPUT = Path(CSV_DIR) / "audit_fanduel_dst_unmatched.csv"

CURRENT_SEASON = 2026


# Internal team abbreviations used by nflverse / this project.
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
    "san francisco forty niners": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

ABBR_ALIASES = {
    "ARZ": "ARI",
    "ARI": "ARI",
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
    print("=" * 100)
    print(title)
    print("=" * 100)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


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
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_abbr(value) -> str | None:
    if pd.isna(value):
        return None

    raw = str(value).strip().upper()
    raw = re.sub(r"[^A-Z]", "", raw)

    if raw in ABBR_ALIASES:
        return ABBR_ALIASES[raw]

    return None


def resolve_team_from_value(value) -> str | None:
    if pd.isna(value):
        return None

    abbr = normalize_abbr(value)
    if abbr is not None:
        return abbr

    text = normalize_text(value)
    if text in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[text]

    # A few deterministic defense-label forms.
    for suffix in [
        " defense",
        " dst",
        " d st",
        " def",
    ]:
        if text.endswith(suffix):
            stripped = text[: -len(suffix)].strip()
            if stripped in TEAM_NAME_TO_ABBR:
                return TEAM_NAME_TO_ABBR[stripped]

    return None


def choose_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    lower_map = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def load_projections(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(PROJECTION_TABLE)}
        WHERE season = ?
        ORDER BY week, game_datetime, game_id, team
        """,
        conn,
        params=[CURRENT_SEASON],
    )

    if df.empty:
        raise RuntimeError(
            f"{PROJECTION_TABLE} returned zero rows for {CURRENT_SEASON}."
        )

    required = {
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
        "dst_projection",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Projection table missing required columns: "
            + ", ".join(missing)
        )

    duplicate_rows = int(
        df.duplicated(["game_id", "team"]).sum()
    )
    if duplicate_rows:
        raise RuntimeError(
            f"Projection source has {duplicate_rows} duplicate game/team rows."
        )

    df["team"] = df["team"].astype(str).str.strip().str.upper()

    return df


def load_fanduel_dst() -> tuple[pd.DataFrame, dict]:
    if not FANDUEL_DST_PATH.exists():
        raise FileNotFoundError(
            f"FanDuel D/ST file not found: {FANDUEL_DST_PATH}"
        )

    df = pd.read_csv(
        FANDUEL_DST_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if df.empty:
        raise RuntimeError(
            f"FanDuel D/ST file is empty: {FANDUEL_DST_PATH}"
        )

    columns = list(df.columns)

    salary_col = choose_column(
        columns,
        [
            "salary",
            "fd salary",
            "fanduel salary",
        ],
    )

    if salary_col is None:
        raise RuntimeError(
            "Could not find a salary column in D-ST.csv. "
            f"Columns: {columns}"
        )

    team_col = choose_column(
        columns,
        [
            "team",
            "teamabbr",
            "team abbr",
            "team abbreviation",
            "abbr",
            "nickname",
        ],
    )

    name_col = choose_column(
        columns,
        [
            "name",
            "player",
            "player name",
            "full name",
            "first name",
        ],
    )

    id_col = choose_column(
        columns,
        [
            "id",
            "player id",
            "fanduel id",
            "fd id",
        ],
    )

    position_col = choose_column(
        columns,
        [
            "position",
            "pos",
        ],
    )

    resolved = []

    for row in df.itertuples(index=False, name=None):
        row_dict = dict(zip(columns, row))

        team = None

        if team_col is not None:
            team = resolve_team_from_value(
                row_dict.get(team_col)
            )

        if team is None and name_col is not None:
            team = resolve_team_from_value(
                row_dict.get(name_col)
            )

        resolved.append(team)

    df["team_internal"] = resolved

    salary_text = (
        df[salary_col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df["salary"] = pd.to_numeric(
        salary_text,
        errors="coerce",
    )

    df["salary_valid"] = (
        df["salary"].notna()
        & df["salary"].gt(0)
    ).astype(int)

    if id_col is not None:
        df["fanduel_id"] = df[id_col].astype(str).str.strip()
    else:
        df["fanduel_id"] = ""

    if name_col is not None:
        df["fanduel_name"] = df[name_col].astype(str).str.strip()
    else:
        df["fanduel_name"] = ""

    if position_col is not None:
        df["fanduel_position"] = (
            df[position_col]
            .astype(str)
            .str.strip()
        )
    else:
        df["fanduel_position"] = "D"

    metadata = {
        "salary_column": salary_col,
        "team_column": team_col or "",
        "name_column": name_col or "",
        "id_column": id_col or "",
        "position_column": position_col or "",
    }

    return df, metadata


def build_pool(
    projections: pd.DataFrame,
    fd: pd.DataFrame,
) -> pd.DataFrame:
    # Exactly one FanDuel row should represent each team.
    fd_valid_team = fd[
        fd["team_internal"].notna()
    ].copy()

    team_counts = (
        fd_valid_team["team_internal"]
        .value_counts()
    )

    duplicate_teams = team_counts[
        team_counts.gt(1)
    ]

    if not duplicate_teams.empty:
        raise RuntimeError(
            "D-ST.csv contains duplicate resolved team rows: "
            + ", ".join(
                f"{team}={count}"
                for team, count in duplicate_teams.items()
            )
        )

    fd_keep = fd_valid_team[
        [
            "team_internal",
            "salary",
            "salary_valid",
            "fanduel_id",
            "fanduel_name",
            "fanduel_position",
        ]
    ].copy()

    merged = projections.merge(
        fd_keep,
        how="left",
        left_on="team",
        right_on="team_internal",
        validate="many_to_one",
    )

    merged["fanduel_row_matched"] = (
        merged["team_internal"].notna()
    ).astype(int)

    merged["optimizer_eligible"] = (
        merged["fanduel_row_matched"].eq(1)
        & merged["salary_valid"].fillna(0).eq(1)
        & pd.to_numeric(
            merged["dst_projection"],
            errors="coerce",
        ).notna()
    ).astype(int)

    # No salary-efficiency or optimizer score is created here.
    # Keep projection and source salary separate and auditable.
    merged = merged.drop(
        columns=["team_internal"],
        errors="ignore",
    )

    return merged


def build_audit(
    projections: pd.DataFrame,
    fd: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(item, value, expected="", status="INFO"):
        rows.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    add(
        "projection_rows_all_2026",
        len(projections),
        544,
        "PASS" if len(projections) == 544 else "CHECK",
    )

    add(
        "fanduel_dst_source_rows",
        len(fd),
        32,
        "PASS" if len(fd) == 32 else "CHECK",
    )

    resolved_source = int(
        fd["team_internal"].notna().sum()
    )

    add(
        "fanduel_rows_team_resolved",
        resolved_source,
        len(fd),
        "PASS" if resolved_source == len(fd) else "FAIL",
    )

    source_valid_salary = int(
        fd["salary_valid"].eq(1).sum()
    )

    add(
        "fanduel_rows_valid_salary",
        source_valid_salary,
        len(fd),
        "PASS" if source_valid_salary == len(fd) else "BLOCKED",
    )

    week1 = pool[
        pool["week"].eq(1)
    ].copy()

    add(
        "week1_projection_rows",
        len(week1),
        32,
        "PASS" if len(week1) == 32 else "FAIL",
    )

    matched = int(
        week1["fanduel_row_matched"].eq(1).sum()
    )

    add(
        "week1_fanduel_rows_matched",
        matched,
        32,
        "PASS" if matched == 32 else "FAIL",
    )

    valid_salary = int(
        week1["salary_valid"].fillna(0).eq(1).sum()
    )

    add(
        "week1_valid_salary_rows",
        valid_salary,
        32,
        "PASS" if valid_salary == 32 else "BLOCKED",
    )

    eligible = int(
        week1["optimizer_eligible"].eq(1).sum()
    )

    add(
        "week1_optimizer_eligible_rows",
        eligible,
        32,
        "PASS" if eligible == 32 else "BLOCKED",
    )

    dupes = int(
        pool.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    add(
        "duplicate_pool_game_team_rows",
        dupes,
        0,
        "PASS" if dupes == 0 else "FAIL",
    )

    null_projection = int(
        pd.to_numeric(
            pool["dst_projection"],
            errors="coerce",
        ).isna().sum()
    )

    add(
        "null_projection_rows",
        null_projection,
        0,
        "PASS" if null_projection == 0 else "FAIL",
    )

    return pd.DataFrame(rows)


def write_sqlite(
    conn: sqlite3.Connection,
    pool: pd.DataFrame,
) -> None:
    db_out = pool.copy()

    db_out.to_sql(
        OUTPUT_TABLE,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_game_team
        ON {OUTPUT_TABLE}(game_id, team)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_season_week
        ON {OUTPUT_TABLE}(season, week)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_eligible
        ON {OUTPUT_TABLE}(optimizer_eligible)
        """
    )

    conn.commit()


def main() -> None:
    section("NFL FANDUEL D/ST PLAYER POOL")

    print(f"Database: {DATABASE_PATH}")
    print(f"Projection source: {PROJECTION_TABLE}")
    print(f"FanDuel source: {FANDUEL_DST_PATH}")
    print("Salary authority: D-ST.csv ONLY")
    print("Salary imputation: DISABLED")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    Path(CSV_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(PARQUET_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as conn:
        projections = load_projections(conn)
        fd, metadata = load_fanduel_dst()

        section("FANDUEL SOURCE COLUMN RESOLUTION")

        for key, value in metadata.items():
            print(f"{key}: {value or '(not found / not required)'}")

        unresolved_fd = fd[
            fd["team_internal"].isna()
        ].copy()

        pool = build_pool(
            projections,
            fd,
        )

        summary = build_audit(
            projections,
            fd,
            pool,
        )

        section("FANDUEL D/ST POOL AUDIT")
        print(summary.to_string(index=False))

        hard_failures = summary[
            summary["status"].eq("FAIL")
        ]

        if not hard_failures.empty:
            print()
            print("STRUCTURAL AUDIT: FAIL")
            raise RuntimeError(
                "FanDuel D/ST pool structural audit failed. "
                "No SQLite output table written."
            )

        print()
        print("STRUCTURAL AUDIT: PASS")

        blocked = summary[
            summary["status"].eq("BLOCKED")
        ]

        if not blocked.empty:
            print()
            print(
                "OPTIMIZER ELIGIBILITY: BLOCKED BY FANDUEL SOURCE DATA"
            )
            print(
                "No salary was imputed. Replace/update D-ST.csv with "
                "valid FanDuel salaries and rerun this file."
            )
        else:
            print()
            print("OPTIMIZER ELIGIBILITY: READY")

        write_sqlite(
            conn,
            pool,
        )

    pool.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    pool.to_parquet(
        PARQUET_OUTPUT,
        index=False,
    )

    summary.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    unresolved_fd.to_csv(
        UNMATCHED_OUTPUT,
        index=False,
    )

    section("WEEK 1 FANDUEL D/ST POOL")

    week1 = (
        pool[
            pool["week"].eq(1)
        ]
        .sort_values(
            ["dst_projection", "team"],
            ascending=[False, True],
        )
    )

    display_cols = [
        "team",
        "opponent_team",
        "is_home",
        "dst_projection",
        "salary",
        "salary_valid",
        "fanduel_name",
        "fanduel_id",
        "fanduel_row_matched",
        "optimizer_eligible",
    ]

    print(
        week1[
            [c for c in display_cols if c in week1.columns]
        ].to_string(index=False)
    )

    section("EXPORTS")

    print(f"CSV: {CSV_OUTPUT}")
    print(f"Parquet: {PARQUET_OUTPUT}")
    print(f"Audit: {AUDIT_OUTPUT}")
    print(f"Unmatched source rows: {UNMATCHED_OUTPUT}")
    print(f"SQLite table: {OUTPUT_TABLE}")

    section("FANDUEL D/ST PLAYER POOL COMPLETE")

    week1_eligible = int(
        week1["optimizer_eligible"].eq(1).sum()
    )

    print(f"Week 1 D/ST rows: {len(week1)}")
    print(f"Week 1 optimizer eligible: {week1_eligible}/32")

    if week1_eligible == 32:
        print(
            "D/ST projection + salary layer is ready for final "
            "FanDuel lineup-pool integration."
        )
    else:
        print(
            "Projection layer is ready, but optimizer integration remains "
            "blocked until D-ST.csv contains valid source salaries."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 100)
        print("FANDUEL D/ST PLAYER POOL FAILED")
        print("=" * 100)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
