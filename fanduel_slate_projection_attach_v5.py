#!/usr/bin/env python3
"""
fanduel_slate_projection_attach_v5.py

Week-aware deterministic projection attachment for salary-bearing FanDuel
Classic slate rows.

Authority
---------
- fanduel_slate_pool:
    exact salary-bearing FanDuel slate universe and salary authority
- fanduel_player_pool:
    slate-driven offensive identity/projection bridge
- dst_production_projection:
    D/ST projection authority

Safety
------
- No fuzzy matching.
- No salary imputation.
- No hardcoded season/week.
- Offensive joins require exact slate + normalized player + team + game_id.
- D/ST joins require exact team + game_id.
- Ambiguous matches are quarantined, never guessed.
- Staged no-salary slates remain outside fanduel_slate_pool and therefore
  cannot become optimizer eligible here.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


SLATE_TABLE = "fanduel_slate_pool"
OFFENSE_TABLE = "fanduel_player_pool"
DST_TABLE = "dst_production_projection"

OFFENSE_FILE_CANDIDATES = [
    Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet",
    Path(CSV_DIR) / "nfl_fanduel_player_pool.csv",
]

OUTPUT_TABLE = "fanduel_slate_projection_pool"
MANIFEST_TABLE = "fanduel_slate_projection_manifest"

OUTPUT_CSV = Path(CSV_DIR) / "fanduel_slate_projection_pool.csv"
OUTPUT_PARQUET = Path(PARQUET_DIR) / "fanduel_slate_projection_pool.parquet"
MANIFEST_CSV = Path(CSV_DIR) / "fanduel_slate_projection_manifest.csv"
GAPS_CSV = Path(CSV_DIR) / "audit_fanduel_slate_projection_gaps.csv"


TEAM_ALIASES = {
    "ARZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL",
    "DEN": "DEN", "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND",
    "JAC": "JAX", "JAX": "JAX", "KC": "KC", "LAC": "LAC", "SD": "LAC",
    "LA": "LA", "LAR": "LA", "STL": "LA", "LV": "LV", "OAK": "LV",
    "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SF": "SF",
    "TB": "TB", "TEN": "TEN", "WAS": "WAS", "WSH": "WAS",
}


def section(title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def table_exists(conn, table_name):
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()
    return bool(row[0])


def read_table(conn, table_name):
    return pd.read_sql_query(
        f'SELECT * FROM "{table_name}"',
        conn,
    )


def first_col(columns, candidates, required=False, label=None):
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate

    if required:
        raise RuntimeError(
            f"Could not resolve {label or candidates[0]}. "
            f"Available columns: {', '.join(sorted(columns))}"
        )
    return None


def clean_text(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    return re.sub(r"\s+", " ", text)


def normalize_text(value):
    value = clean_text(value)
    if value is None:
        return None

    text = unicodedata.normalize("NFKD", value)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_team(value):
    value = clean_text(value)
    if value is None:
        return None

    key = re.sub(r"[^A-Z]", "", value.upper())
    return TEAM_ALIASES.get(key, key)


def normalize_position(value):
    value = clean_text(value)
    if value is None:
        return None

    value = value.upper()
    if value in {"QB", "RB", "WR", "TE", "DST"}:
        return value

    return None


def finite_number(value):
    try:
        return pd.notna(value) and np.isfinite(float(value))
    except Exception:
        return False


def load_offense_source(conn):
    if table_exists(conn, OFFENSE_TABLE):
        df = read_table(conn, OFFENSE_TABLE)
        if not df.empty:
            return df, f"SQLite:{OFFENSE_TABLE}"

    for path in OFFENSE_FILE_CANDIDATES:
        if not path.exists():
            continue

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if not df.empty:
            return df, str(path)

    raise RuntimeError(
        "Could not resolve offensive player-pool source. "
        "Run fanduel_player_pool.py first."
    )


def resolve_slate_schema(df):
    cols = list(df.columns)

    return {
        "player": first_col(
            cols,
            ["player", "fd_name", "player_name"],
            True,
            "slate player",
        ),
        "team": first_col(
            cols,
            ["team_internal", "team"],
            True,
            "slate team",
        ),
        "salary": first_col(
            cols,
            ["salary", "fd_salary"],
            True,
            "slate salary",
        ),
        "game": first_col(
            cols,
            ["gameInfo", "game_info", "game"],
            True,
            "slate gameInfo",
        ),
        "game_id": first_col(
            cols,
            ["game_id"],
            True,
            "slate game_id",
        ),
        "season": first_col(
            cols,
            ["season"],
            True,
            "slate season",
        ),
        "week": first_col(
            cols,
            ["week"],
            True,
            "slate week",
        ),
        "slate": first_col(
            cols,
            ["slate_name"],
            True,
            "slate name",
        ),
        "slug": first_col(
            cols,
            ["slate_slug"],
            True,
            "slate slug",
        ),
        "dst": first_col(
            cols,
            ["is_dst"],
            True,
            "D/ST flag",
        ),
        "position": first_col(
            cols,
            ["position", "fd_position", "solver_position"],
        ),
    }


def resolve_offense_schema(df):
    cols = list(df.columns)

    return {
        "player": first_col(
            cols,
            ["fd_name", "player", "player_display_name", "player_name"],
            True,
            "offensive player",
        ),
        "team": first_col(
            cols,
            ["team", "team_internal"],
            True,
            "offensive team",
        ),
        "game_id": first_col(
            cols,
            ["game_id", "model_game_id"],
            True,
            "offensive game_id",
        ),
        "season": first_col(
            cols,
            ["season", "model_season"],
        ),
        "week": first_col(
            cols,
            ["week", "model_week"],
        ),
        "slate_slug": first_col(
            cols,
            ["slate_slug"],
            True,
            "offensive slate_slug",
        ),
        "position": first_col(
            cols,
            ["fd_position", "position", "model_position"],
        ),
        "projection": first_col(
            cols,
            [
                "live_projection",
                "ridge_projection",
                "model_projection",
                "internal_projection",
                "production_projection",
            ],
            True,
            "offensive production projection",
        ),
        "eligible": first_col(
            cols,
            ["optimizer_eligible", "solver_eligible"],
        ),
        "status": first_col(
            cols,
            ["player_pool_status", "production_status"],
        ),
        "identity": first_col(
            cols,
            ["identity_key", "projection_identity"],
        ),
    }


def resolve_dst_schema(df):
    cols = list(df.columns)

    return {
        "team": first_col(
            cols,
            ["team", "team_internal"],
            True,
            "D/ST team",
        ),
        "opponent": first_col(
            cols,
            ["opponent_team", "opponent"],
        ),
        "projection": first_col(
            cols,
            ["dst_projection", "model_projection", "projection"],
            True,
            "D/ST projection",
        ),
        "game_id": first_col(
            cols,
            ["game_id"],
            True,
            "D/ST game_id",
        ),
        "season": first_col(
            cols,
            ["season"],
        ),
        "week": first_col(
            cols,
            ["week"],
        ),
    }


def prepare_slate(df, schema):
    out = df.copy()

    out["_player"] = out[schema["player"]].astype(str)
    out["_player_norm"] = out["_player"].map(normalize_text)
    out["_team"] = out[schema["team"]].map(normalize_team)
    out["_salary"] = pd.to_numeric(
        out[schema["salary"]],
        errors="coerce",
    )
    out["_game"] = out[schema["game"]].astype(str)
    out["_game_id"] = out[schema["game_id"]].map(clean_text)
    out["_season"] = pd.to_numeric(
        out[schema["season"]],
        errors="coerce",
    )
    out["_week"] = pd.to_numeric(
        out[schema["week"]],
        errors="coerce",
    )
    out["_slate"] = out[schema["slate"]].astype(str)
    out["_slug"] = out[schema["slug"]].astype(str)
    out["_is_dst"] = (
        pd.to_numeric(
            out[schema["dst"]],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    if schema["position"]:
        out["_position"] = out[schema["position"]].map(
            normalize_position
        )
    else:
        out["_position"] = None

    out.loc[
        out["_is_dst"].eq(1),
        "_position",
    ] = "DST"

    return out


def prepare_offense(df, schema):
    out = pd.DataFrame(index=df.index)

    out["_player"] = df[schema["player"]].astype(str)
    out["_player_norm"] = out["_player"].map(normalize_text)
    out["_team"] = df[schema["team"]].map(normalize_team)
    out["_game_id"] = df[schema["game_id"]].map(clean_text)
    out["_slug"] = df[schema["slate_slug"]].astype(str)

    if schema["season"]:
        out["_season"] = pd.to_numeric(
            df[schema["season"]],
            errors="coerce",
        )
    else:
        out["_season"] = np.nan

    if schema["week"]:
        out["_week"] = pd.to_numeric(
            df[schema["week"]],
            errors="coerce",
        )
    else:
        out["_week"] = np.nan

    if schema["position"]:
        out["_position"] = df[schema["position"]].map(
            normalize_position
        )
    else:
        out["_position"] = None

    out["_projection"] = pd.to_numeric(
        df[schema["projection"]],
        errors="coerce",
    )

    if schema["eligible"]:
        raw = df[schema["eligible"]]
        numeric = pd.to_numeric(
            raw,
            errors="coerce",
        )

        if numeric.notna().any():
            out["_source_eligible"] = (
                numeric.fillna(0).astype(int)
            )
        else:
            out["_source_eligible"] = (
                raw.astype(str)
                .str.lower()
                .isin({
                    "true",
                    "yes",
                    "y",
                    "ready",
                    "eligible",
                })
                .astype(int)
            )
    else:
        out["_source_eligible"] = (
            out["_projection"].map(finite_number).astype(int)
        )

    if schema["status"]:
        out["_source_status"] = (
            df[schema["status"]].astype(str)
        )
    else:
        out["_source_status"] = ""

    if schema["identity"]:
        out["_identity"] = (
            df[schema["identity"]]
            .fillna("")
            .astype(str)
        )
    else:
        out["_identity"] = ""

    return out


def prepare_dst(df, schema):
    out = pd.DataFrame(index=df.index)

    out["_team"] = df[schema["team"]].map(normalize_team)
    out["_game_id"] = df[schema["game_id"]].map(clean_text)
    out["_projection"] = pd.to_numeric(
        df[schema["projection"]],
        errors="coerce",
    )

    if schema["opponent"]:
        out["_opponent"] = df[schema["opponent"]].map(
            normalize_team
        )
    else:
        out["_opponent"] = None

    if schema["season"]:
        out["_season"] = pd.to_numeric(
            df[schema["season"]],
            errors="coerce",
        )
    else:
        out["_season"] = np.nan

    if schema["week"]:
        out["_week"] = pd.to_numeric(
            df[schema["week"]],
            errors="coerce",
        )
    else:
        out["_week"] = np.nan

    return out


def unique_exact_map(df, keys):
    valid = df.copy()

    for key in keys:
        valid = valid[
            valid[key].notna()
        ]

        if valid[key].dtype == object:
            valid = valid[
                valid[key].astype(str).ne("")
            ]

    counts = (
        valid.groupby(
            keys,
            dropna=False,
        )
        .size()
        .rename("_count")
        .reset_index()
    )

    unique_keys = counts[
        counts["_count"].eq(1)
    ][keys]

    return valid.merge(
        unique_keys,
        on=keys,
        how="inner",
    )


def attach_offense(slates, offense, offense_source_label):
    result = slates.copy()

    result["internal_projection"] = np.nan
    result["projection_source"] = ""
    result["projection_match_method"] = ""
    result["projection_identity"] = ""
    result["projection_status"] = "UNMATCHED"
    result["legacy_source_eligible"] = np.nan

    off_rows = result["_is_dst"].eq(0)

    keys = [
        "_slug",
        "_player_norm",
        "_team",
        "_game_id",
    ]

    unique = unique_exact_map(
        offense,
        keys,
    )

    source_ambiguous = len(offense) - len(unique)

    if source_ambiguous:
        print(
            "Offensive source rows excluded from exact unique map: "
            f"{source_ambiguous}"
        )

    match_map = {
        tuple(rec[key] for key in keys): rec
        for rec in unique.to_dict("records")
    }

    for idx in result.index[off_rows]:
        row = result.loc[idx]

        key = tuple(
            row[value]
            for value in keys
        )

        match = match_map.get(key)

        if match is None:
            result.at[
                idx,
                "projection_status",
            ] = "OFFENSE_UNMATCHED_EXACT_GAME"
            continue

        result.at[
            idx,
            "projection_source",
        ] = offense_source_label

        result.at[
            idx,
            "projection_match_method",
        ] = "exact_slate_name_team_game"

        result.at[
            idx,
            "projection_identity",
        ] = match["_identity"]

        result.at[
            idx,
            "legacy_source_eligible",
        ] = int(
            match["_source_eligible"]
        )

        if (
            int(match["_source_eligible"]) != 1
            or not finite_number(
                match["_projection"]
            )
        ):
            result.at[
                idx,
                "projection_status",
            ] = "MATCHED_NOT_MODEL_READY"
            continue

        result.at[
            idx,
            "internal_projection",
        ] = float(match["_projection"])

        result.at[
            idx,
            "projection_status",
        ] = "READY"

        if (
            result.at[idx, "_position"] is None
            and match["_position"] is not None
        ):
            result.at[
                idx,
                "_position",
            ] = match["_position"]

    return result


def attach_dst(result, dst):
    keys = [
        "_team",
        "_game_id",
    ]

    unique = unique_exact_map(
        dst,
        keys,
    )

    dst_map = {
        tuple(rec[key] for key in keys): rec
        for rec in unique.to_dict("records")
    }

    for idx in result.index[
        result["_is_dst"].eq(1)
    ]:
        key = (
            result.at[idx, "_team"],
            result.at[idx, "_game_id"],
        )

        match = dst_map.get(key)

        if match is None:
            result.at[
                idx,
                "projection_status",
            ] = "DST_UNMATCHED_EXACT_GAME"
            continue

        result.at[
            idx,
            "projection_source",
        ] = DST_TABLE

        result.at[
            idx,
            "projection_match_method",
        ] = "exact_team_game_id"

        result.at[
            idx,
            "projection_identity",
        ] = (
            f"DST:{result.at[idx, '_team']}:"
            f"{result.at[idx, '_game_id']}"
        )

        if not finite_number(
            match["_projection"]
        ):
            result.at[
                idx,
                "projection_status",
            ] = "DST_MATCHED_NO_PROJECTION"
            continue

        result.at[
            idx,
            "internal_projection",
        ] = float(match["_projection"])

        result.at[
            idx,
            "projection_status",
        ] = "READY"

    return result


def build_manifest(result):
    rows = []

    group_keys = [
        "_slate",
        "_slug",
        "_season",
        "_week",
    ]

    for keys, g in result.groupby(
        group_keys,
        sort=True,
        dropna=False,
    ):
        slate_name, slate_slug, season, week = keys

        offense = g[
            g["_is_dst"].eq(0)
        ]
        dst = g[
            g["_is_dst"].eq(1)
        ]

        ready = g[
            "projection_status"
        ].eq("READY")

        offense_ready = offense[
            "projection_status"
        ].eq("READY")

        dst_ready = dst[
            "projection_status"
        ].eq("READY")

        rows.append({
            "slate_name": slate_name,
            "slate_slug": slate_slug,
            "season": season,
            "week": week,
            "contest_rows": len(g),
            "offense_rows": len(offense),
            "dst_rows": len(dst),
            "projection_ready_rows": int(
                ready.sum()
            ),
            "projection_gap_rows": int(
                (~ready).sum()
            ),
            "offense_projection_ready_rows": int(
                offense_ready.sum()
            ),
            "offense_projection_gap_rows": int(
                (~offense_ready).sum()
            ),
            "dst_projection_ready_rows": int(
                dst_ready.sum()
            ),
            "dst_projection_gap_rows": int(
                (~dst_ready).sum()
            ),
            "coverage_pct": (
                round(
                    100.0 * ready.mean(),
                    3,
                )
                if len(g)
                else 0.0
            ),
            "dst_coverage_pct": (
                round(
                    100.0 * dst_ready.mean(),
                    3,
                )
                if len(dst)
                else 0.0
            ),
            "status": (
                "READY"
                if ready.all()
                else (
                    "DST_READY_OFFENSE_GAPS"
                    if len(dst)
                    and dst_ready.all()
                    else "BLOCKED"
                )
            ),
        })

    return pd.DataFrame(rows)


def clean_output(result):
    out = result.copy()

    out["model_projection"] = (
        out["internal_projection"]
    )

    out["optimizer_eligible"] = (
        out["projection_status"]
        .eq("READY")
        .astype(int)
    )

    # Carry deterministic normalized position for downstream position matching
    # when it is known. Original slate columns are otherwise preserved intact.
    if "position" not in out.columns:
        out["position"] = out["_position"]

    out["position"] = out[
        "position"
    ].where(
        out["position"].notna(),
        out["_position"],
    )

    drop = [
        "_slate",
        "_slug",
        "_player",
        "_player_norm",
        "_team",
        "_salary",
        "_game",
        "_game_id",
        "_season",
        "_week",
        "_is_dst",
        "_position",
    ]

    existing_drop = [
        c
        for c in drop
        if c in out.columns
    ]

    return out.drop(
        columns=existing_drop
    )


def main():
    section(
        "FANDUEL SLATE PROJECTION ATTACHMENT"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )
    print(
        f"Slate authority: {SLATE_TABLE}"
    )
    print(
        f"Offensive projection bridge: "
        f"{OFFENSE_TABLE}"
    )
    print(
        f"D/ST projection authority: "
        f"{DST_TABLE}"
    )
    print(
        "Matching: deterministic exact-game only; "
        "fuzzy matching disabled"
    )
    print(
        "Salary imputation: disabled"
    )
    print(
        "Hardcoded current week: disabled"
    )
    print(
        "Attachment engine: schedule-aware v5 replacement"
    )

    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:
        for table in (
            SLATE_TABLE,
            DST_TABLE,
        ):
            if not table_exists(
                conn,
                table,
            ):
                raise RuntimeError(
                    f"Required SQLite table not found: {table}"
                )

        slate_raw = read_table(
            conn,
            SLATE_TABLE,
        )

        (
            offense_raw,
            offense_source_label,
        ) = load_offense_source(
            conn
        )

        dst_raw = read_table(
            conn,
            DST_TABLE,
        )

        print(
            f"Resolved offensive source: "
            f"{offense_source_label}"
        )

        slate_schema = resolve_slate_schema(
            slate_raw
        )
        offense_schema = resolve_offense_schema(
            offense_raw
        )
        dst_schema = resolve_dst_schema(
            dst_raw
        )

        section(
            "RESOLVED SOURCE SCHEMAS"
        )

        print("Slate:")
        for key, value in slate_schema.items():
            print(
                f"  {key:<14} -> {value}"
            )

        print()
        print("Offense:")
        for key, value in offense_schema.items():
            print(
                f"  {key:<14} -> {value}"
            )

        print()
        print("D/ST:")
        for key, value in dst_schema.items():
            print(
                f"  {key:<14} -> {value}"
            )

        slates = prepare_slate(
            slate_raw,
            slate_schema,
        )
        offense = prepare_offense(
            offense_raw,
            offense_schema,
        )
        dst = prepare_dst(
            dst_raw,
            dst_schema,
        )

        if (
            slates["_salary"].isna().any()
            or slates["_salary"].le(0).any()
        ):
            raise RuntimeError(
                "fanduel_slate_pool contains "
                "missing/non-positive salary. "
                "Do not impute salary."
            )

        if slates["_game_id"].isna().any():
            raise RuntimeError(
                "fanduel_slate_pool contains "
                "missing game_id."
            )

        result = attach_offense(
            slates,
            offense,
            offense_source_label,
        )

        result = attach_dst(
            result,
            dst,
        )

        manifest = build_manifest(
            result
        )

        gaps = result[
            result[
                "projection_status"
            ].ne("READY")
        ].copy()

        section(
            "PROJECTION COVERAGE BY SLATE"
        )

        print(
            manifest[
                [
                    "slate_name",
                    "season",
                    "week",
                    "contest_rows",
                    "offense_rows",
                    "dst_rows",
                    "projection_ready_rows",
                    "projection_gap_rows",
                    "offense_projection_gap_rows",
                    "dst_projection_gap_rows",
                    "coverage_pct",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )

        section(
            "GLOBAL AUDIT"
        )

        print(
            f"Contest rows:                 "
            f"{len(result)}"
        )
        print(
            f"Projection ready:             "
            f"{result['projection_status'].eq('READY').sum()}"
        )
        print(
            f"Projection gaps:              "
            f"{len(gaps)}"
        )
        print(
            f"D/ST rows:                    "
            f"{result['_is_dst'].eq(1).sum()}"
        )
        print(
            "D/ST ready:                   "
            f"{(
                result['_is_dst'].eq(1)
                & result['projection_status'].eq('READY')
            ).sum()}"
        )
        print(
            "Duplicate slate/player rows:  "
            f"{result.duplicated(
                [
                    '_slug',
                    '_player_norm',
                    '_team',
                    '_salary',
                    '_game_id',
                ]
            ).sum()}"
        )
        print(
            f"Missing salaries:             "
            f"{result['_salary'].isna().sum()}"
        )
        print(
            f"Non-positive salaries:        "
            f"{result['_salary'].le(0).sum()}"
        )

        if not gaps.empty:
            section(
                "PROJECTION GAP BREAKDOWN"
            )

            breakdown = (
                gaps.groupby(
                    [
                        "_slate",
                        "projection_status",
                    ],
                    dropna=False,
                )
                .size()
                .rename("rows")
                .reset_index()
            )

            print(
                breakdown.to_string(
                    index=False
                )
            )

            section(
                "UNIQUE GAP PLAYERS"
            )

            unique_gaps = (
                gaps[
                    [
                        "_player",
                        "_team",
                        "_position",
                        "_game_id",
                        "projection_status",
                        "_salary",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "projection_status",
                        "_team",
                        "_player",
                    ],
                    na_position="last",
                )
            )

            print(
                unique_gaps.to_string(
                    index=False
                )
            )

        output = clean_output(
            result
        )

        output.to_sql(
            OUTPUT_TABLE,
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
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{OUTPUT_TABLE}_slate "
            f"ON {OUTPUT_TABLE}(slate_slug)"
        )

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS "
            f"idx_{OUTPUT_TABLE}_eligible "
            f"ON {OUTPUT_TABLE}"
            f"(slate_slug, optimizer_eligible)"
        )

        if "game_id" in output.columns:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS "
                f"idx_{OUTPUT_TABLE}_game "
                f"ON {OUTPUT_TABLE}(game_id)"
            )

        conn.commit()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    output.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    if gaps.empty:
        pd.DataFrame(
            columns=[
                "slate_name",
                "player",
                "team",
                "salary",
                "game_id",
                "projection_status",
            ]
        ).to_csv(
            GAPS_CSV,
            index=False,
        )
    else:
        gaps.to_csv(
            GAPS_CSV,
            index=False,
        )

    section(
        "EXPORTS"
    )

    print(
        f"Pool CSV:     {OUTPUT_CSV}"
    )
    print(
        f"Pool Parquet: {OUTPUT_PARQUET}"
    )
    print(
        f"Manifest:     {MANIFEST_CSV}"
    )
    print(
        f"Gap audit:    {GAPS_CSV}"
    )
    print(
        f"SQLite pool:  {OUTPUT_TABLE}"
    )
    print(
        f"SQLite audit: {MANIFEST_TABLE}"
    )

    section(
        "PROJECTION ATTACHMENT COMPLETE"
    )

    print(
        "Offense attaches by exact slate + player + team + game_id."
    )
    print(
        "D/ST attaches by exact team + game_id."
    )
    print(
        "No global week filter or hardcoded Week 1 logic remains."
    )
    print(
        "FanDuel salary remains authoritative from fanduel_slate_pool."
    )
    print(
        "Any row without an exact finite internal projection remains ineligible."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 108)
        print(
            "FANDUEL SLATE PROJECTION ATTACHMENT FAILED"
        )
        print("=" * 108)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)
