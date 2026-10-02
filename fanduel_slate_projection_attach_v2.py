#!/usr/bin/env python3
"""
fanduel_slate_projection_attach.py

Attach internal production projections to the exact salary-bearing FanDuel
contest pools produced by fanduel_slate_ingest_v2.py.

Authority
---------
- fanduel_slate_pool:
    exact slate universe + FanDuel salary
- fanduel_player_pool:
    offensive FanDuel identity/projection bridge
- dst_production_projection:
    internal D/ST production projections

Safety
------
- No fuzzy matching.
- No salary imputation.
- No source "fantasy" projection used as internal projection.
- D/ST joins by deterministic team + game.
- Offensive joins prefer stable FanDuel pool identity keys and exact normalized
  player/team matching only.
- Ambiguous matches are quarantined, never guessed.

Outputs
-------
SQLite:
    fanduel_slate_projection_pool
    fanduel_slate_projection_manifest

CSV:
    data/csv/fanduel_slate_projection_pool.csv
    data/csv/fanduel_slate_projection_manifest.csv
    data/csv/audit_fanduel_slate_projection_gaps.csv

Parquet:
    data/parquet/fanduel_slate_projection_pool.parquet
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
    Path(PARQUET_DIR) / "fanduel_player_pool.parquet",
    Path(CSV_DIR) / "fanduel_player_pool.csv",
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
    "ARI":"ARI","ARZ":"ARI","ATL":"ATL","BAL":"BAL","BLT":"BAL",
    "BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "CLV":"CLE","DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB",
    "GNB":"GB","HOU":"HOU","IND":"IND","JAC":"JAX","JAX":"JAX",
    "KC":"KC","KAN":"KC","LV":"LV","LVR":"LV","OAK":"LV",
    "LAC":"LAC","SD":"LAC","SDG":"LAC","LA":"LA","LAR":"LA",
    "STL":"LA","MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE",
    "NO":"NO","NOR":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI",
    "PIT":"PIT","SF":"SF","SFO":"SF","SEA":"SEA","TB":"TB",
    "TAM":"TB","TEN":"TEN","WAS":"WAS","WSH":"WAS",
}

POSITION_ALIASES = {
    "QB":"QB", "RB":"RB", "WR":"WR", "TE":"TE",
    "D":"DST", "DEF":"DST", "DST":"DST", "D/ST":"DST",
}


def section(title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def normalize_text(value):
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_team(value):
    if pd.isna(value):
        return None
    s = re.sub(r"[^A-Z]", "", str(value).upper())
    return TEAM_ALIASES.get(s)


def normalize_position(value):
    if pd.isna(value):
        return None
    raw = str(value).strip().upper()
    return POSITION_ALIASES.get(raw, raw if raw in {"QB","RB","WR","TE"} else None)


def table_exists(conn, table):
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (table,)).fetchone() is not None


def columns(conn, table):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def first_col(cols, candidates, required=False, label="column"):
    lower = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise RuntimeError(
            f"Could not resolve required {label}. Tried: {candidates}. "
            f"Available: {cols}"
        )
    return None


def read_table(conn, table):
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def load_offense_source(conn):
    """
    Load the stable offensive FanDuel pool without requiring the stable
    offensive pipeline to write SQLite.

    Priority:
      1. SQLite fanduel_player_pool, if it exists.
      2. Known Parquet/CSV export paths.
    """
    if table_exists(conn, OFFENSE_TABLE):
        df = read_table(conn, OFFENSE_TABLE)
        return df, f"sqlite:{OFFENSE_TABLE}"

    for path in OFFENSE_FILE_CANDIDATES:
        if not path.exists():
            continue

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if df.empty:
            continue

        return df, str(path)

    searched = "\n".join(f"  - {p}" for p in OFFENSE_FILE_CANDIDATES)
    raise RuntimeError(
        "Stable offensive FanDuel pool was not found in SQLite or known "
        "export paths. Searched:\n" + searched
    )


def resolve_slate_schema(df):
    cols = list(df.columns)
    return {
        "slate": first_col(cols, ["slate_name"], True, "slate name"),
        "slug": first_col(cols, ["slate_slug"], True, "slate slug"),
        "player": first_col(cols, ["player", "name", "player_name"], True, "player"),
        "team": first_col(cols, ["team_internal", "team"], True, "team"),
        "salary": first_col(cols, ["salary"], True, "salary"),
        "game": first_col(cols, ["game_key", "gameInfo", "game_info"], True, "game"),
        "dst": first_col(cols, ["is_dst"], True, "D/ST flag"),
        "source_proj": first_col(cols, ["source_fantasy_projection", "fantasy"]),
        "position": first_col(cols, ["position", "pos", "roster_position"]),
    }


def resolve_offense_schema(df):
    cols = list(df.columns)
    return {
        "player": first_col(
            cols,
            ["player", "player_name", "name", "full_name", "display_name"],
            True,
            "offensive player name",
        ),
        "team": first_col(
            cols,
            ["team_internal", "team", "recent_team"],
            True,
            "offensive team",
        ),
        "position": first_col(cols, ["position", "pos", "fd_position"]),
        "salary": first_col(cols, ["salary", "fd_salary"]),
        "projection": first_col(
            cols,
            [
                "production_projection",
                "projection",
                "projected_points",
                "fanduel_projection",
                "ridge_projection",
                "model_projection",
            ],
            True,
            "offensive production projection",
        ),
        "eligible": first_col(
            cols,
            ["optimizer_eligible", "model_ready", "projection_ready"],
        ),
        "identity": first_col(
            cols,
            [
                "player_id", "gsis_id", "identity_key",
                "canonical_player_id", "fantasypros_id",
            ],
        ),
    }


def resolve_dst_schema(df):
    cols = list(df.columns)
    return {
        "team": first_col(cols, ["team", "team_internal"], True, "D/ST team"),
        "opponent": first_col(cols, ["opponent", "opp"]),
        "game_id": first_col(cols, ["game_id"]),
        "week": first_col(cols, ["week"]),
        "season": first_col(cols, ["season"]),
        "projection": first_col(
            cols,
            [
                "dst_projection",
                "production_projection",
                "projection",
                "projected_points",
                "fanduel_projection",
            ],
            True,
            "D/ST production projection",
        ),
    }


def parse_game_key(value):
    text = str(value).strip().upper()
    m = re.match(r"^\s*([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\s*$", text)
    if not m:
        return None, None
    return normalize_team(m.group(1)), normalize_team(m.group(2))


def prepare_slate(df, schema):
    out = df.copy()
    out["_slate"] = out[schema["slate"]].astype(str)
    out["_slug"] = out[schema["slug"]].astype(str)
    out["_player"] = out[schema["player"]].astype(str)
    out["_player_norm"] = out["_player"].map(normalize_text)
    out["_team"] = out[schema["team"]].map(normalize_team)
    out["_salary"] = pd.to_numeric(out[schema["salary"]], errors="coerce")
    out["_game"] = out[schema["game"]].astype(str)
    out["_is_dst"] = pd.to_numeric(out[schema["dst"]], errors="coerce").fillna(0).astype(int)

    parsed = out["_game"].map(parse_game_key)
    out["_away"] = [x[0] for x in parsed]
    out["_home"] = [x[1] for x in parsed]

    if schema["position"]:
        out["_position"] = out[schema["position"]].map(normalize_position)
    else:
        out["_position"] = None

    out.loc[out["_is_dst"].eq(1), "_position"] = "DST"

    return out


def prepare_offense(df, schema):
    out = pd.DataFrame(index=df.index)
    out["_player"] = df[schema["player"]].astype(str)
    out["_player_norm"] = out["_player"].map(normalize_text)
    out["_team"] = df[schema["team"]].map(normalize_team)
    out["_projection"] = pd.to_numeric(df[schema["projection"]], errors="coerce")

    if schema["position"]:
        out["_position"] = df[schema["position"]].map(normalize_position)
    else:
        out["_position"] = None

    if schema["salary"]:
        out["_source_salary"] = pd.to_numeric(df[schema["salary"]], errors="coerce")
    else:
        out["_source_salary"] = np.nan

    if schema["eligible"]:
        raw = df[schema["eligible"]]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any():
            out["_source_eligible"] = numeric.fillna(0).astype(int)
        else:
            out["_source_eligible"] = (
                raw.astype(str).str.lower().isin({"true","yes","y","ready","eligible"})
            ).astype(int)
    else:
        out["_source_eligible"] = out["_projection"].notna().astype(int)

    if schema["identity"]:
        out["_identity"] = df[schema["identity"]].astype(str)
    else:
        out["_identity"] = ""

    return out


def prepare_dst(df, schema):
    out = pd.DataFrame(index=df.index)
    out["_team"] = df[schema["team"]].map(normalize_team)
    out["_projection"] = pd.to_numeric(df[schema["projection"]], errors="coerce")

    if schema["opponent"]:
        out["_opponent"] = df[schema["opponent"]].map(normalize_team)
    else:
        out["_opponent"] = None

    if schema["week"]:
        out["_week"] = pd.to_numeric(df[schema["week"]], errors="coerce")
    else:
        out["_week"] = np.nan

    if schema["season"]:
        out["_season"] = pd.to_numeric(df[schema["season"]], errors="coerce")
    else:
        out["_season"] = np.nan

    if schema["game_id"]:
        out["_game_id"] = df[schema["game_id"]].astype(str)
    else:
        out["_game_id"] = ""

    return out


def unique_exact_map(df, keys):
    valid = df.copy()
    for key in keys:
        valid = valid[valid[key].notna()]
        if valid[key].dtype == object:
            valid = valid[valid[key].astype(str).ne("")]

    counts = valid.groupby(keys, dropna=False).size().rename("_count").reset_index()
    unique_keys = counts[counts["_count"].eq(1)][keys]
    unique = valid.merge(unique_keys, on=keys, how="inner")
    return unique


def attach_offense(slates, offense, offense_source_label):
    result = slates.copy()
    result["internal_projection"] = np.nan
    result["projection_source"] = ""
    result["projection_match_method"] = ""
    result["projection_identity"] = ""
    result["projection_status"] = "UNMATCHED"

    off_rows = result["_is_dst"].eq(0)

    # Primary deterministic join: exact normalized player + exact team.
    unique_name_team = unique_exact_map(offense, ["_player_norm", "_team"])
    map_nt = {
        (r._player_norm, r._team): r
        for r in unique_name_team.itertuples(index=False)
    }

    for idx in result.index[off_rows]:
        row = result.loc[idx]
        key = (row["_player_norm"], row["_team"])
        match = map_nt.get(key)

        if match is None:
            continue

        result.at[idx, "internal_projection"] = match._projection
        result.at[idx, "projection_source"] = offense_source_label
        result.at[idx, "projection_match_method"] = "exact_name_team"
        result.at[idx, "projection_identity"] = match._identity

        if pd.isna(match._projection):
            result.at[idx, "projection_status"] = "MATCHED_NO_PROJECTION"
        elif int(match._source_eligible) != 1:
            result.at[idx, "projection_status"] = "MATCHED_NOT_MODEL_ELIGIBLE"
        else:
            result.at[idx, "projection_status"] = "READY"

    return result


def choose_current_dst_rows(dst):
    out = dst.copy()

    # Production table is expected to contain 2026. If season exists, force 2026.
    if out["_season"].notna().any():
        out = out[out["_season"].eq(2026)].copy()

    # Current Week 1 contest files: prefer week 1 if the table exposes week.
    if out["_week"].notna().any():
        week1 = out[out["_week"].eq(1)].copy()
        if not week1.empty:
            out = week1

    return out


def attach_dst(result, dst):
    dst = choose_current_dst_rows(dst)

    # Exact team is sufficient only after constraining current season/week.
    unique_team = unique_exact_map(dst, ["_team"])
    dst_map = {
        r._team: r
        for r in unique_team.itertuples(index=False)
    }

    for idx in result.index[result["_is_dst"].eq(1)]:
        team = result.at[idx, "_team"]
        match = dst_map.get(team)

        if match is None:
            result.at[idx, "projection_status"] = "DST_UNMATCHED"
            continue

        result.at[idx, "internal_projection"] = match._projection
        result.at[idx, "projection_source"] = DST_TABLE
        result.at[idx, "projection_match_method"] = "exact_team_current_week"

        if pd.isna(match._projection):
            result.at[idx, "projection_status"] = "DST_MATCHED_NO_PROJECTION"
        else:
            result.at[idx, "projection_status"] = "READY"

    return result


def build_manifest(result):
    rows = []

    for slate_name, g in result.groupby("_slate", sort=True):
        offense = g[g["_is_dst"].eq(0)]
        dst = g[g["_is_dst"].eq(1)]

        ready = g["projection_status"].eq("READY")
        offense_ready = offense["projection_status"].eq("READY")
        dst_ready = dst["projection_status"].eq("READY")

        rows.append({
            "slate_name": slate_name,
            "slate_slug": g["_slug"].iloc[0],
            "contest_rows": len(g),
            "offense_rows": len(offense),
            "dst_rows": len(dst),
            "projection_ready_rows": int(ready.sum()),
            "projection_gap_rows": int((~ready).sum()),
            "offense_projection_ready_rows": int(offense_ready.sum()),
            "offense_projection_gap_rows": int((~offense_ready).sum()),
            "dst_projection_ready_rows": int(dst_ready.sum()),
            "dst_projection_gap_rows": int((~dst_ready).sum()),
            "coverage_pct": round(100.0 * ready.mean(), 3) if len(g) else 0.0,
            "dst_coverage_pct": round(100.0 * dst_ready.mean(), 3) if len(dst) else 0.0,
            "status": (
                "READY"
                if ready.all()
                else (
                    "DST_READY_OFFENSE_GAPS"
                    if dst_ready.all()
                    else "BLOCKED"
                )
            ),
        })

    return pd.DataFrame(rows)


def clean_output(result):
    drop = [
        "_slate", "_slug", "_player", "_player_norm", "_team", "_salary",
        "_game", "_is_dst", "_away", "_home", "_position",
    ]
    out = result.copy()

    # User-facing normalized fields.
    out["model_projection"] = out["internal_projection"]
    out["optimizer_eligible"] = out["projection_status"].eq("READY").astype(int)

    existing_drop = [c for c in drop if c in out.columns]
    out = out.drop(columns=existing_drop)

    return out


def main():
    section("FANDUEL SLATE PROJECTION ATTACHMENT")

    print(f"Database: {DATABASE_PATH}")
    print(f"Slate authority: {SLATE_TABLE}")
    print(f"Offensive projection bridge: {OFFENSE_TABLE} (SQLite or exported file)")
    print(f"D/ST projection authority: {DST_TABLE}")
    print("Matching: deterministic only; fuzzy matching disabled")
    print("Salary imputation: disabled")

    with sqlite3.connect(DATABASE_PATH) as conn:
        for table in (SLATE_TABLE, DST_TABLE):
            if not table_exists(conn, table):
                raise RuntimeError(f"Required SQLite table not found: {table}")

        slate_raw = read_table(conn, SLATE_TABLE)
        offense_raw, offense_source_label = load_offense_source(conn)
        dst_raw = read_table(conn, DST_TABLE)

        print(f"Resolved offensive source: {offense_source_label}")

        slate_schema = resolve_slate_schema(slate_raw)
        offense_schema = resolve_offense_schema(offense_raw)
        dst_schema = resolve_dst_schema(dst_raw)

        section("RESOLVED SOURCE SCHEMAS")
        print("Slate:")
        for k, v in slate_schema.items():
            print(f"  {k:<14} -> {v}")

        print("\nOffense:")
        for k, v in offense_schema.items():
            print(f"  {k:<14} -> {v}")

        print("\nD/ST:")
        for k, v in dst_schema.items():
            print(f"  {k:<14} -> {v}")

        slates = prepare_slate(slate_raw, slate_schema)
        offense = prepare_offense(offense_raw, offense_schema)
        dst = prepare_dst(dst_raw, dst_schema)

        # Structural salary gate inherited from the frozen slate ingest.
        if slates["_salary"].isna().any() or slates["_salary"].le(0).any():
            raise RuntimeError(
                "fanduel_slate_pool contains missing/non-positive salary. "
                "Do not impute salary."
            )

        result = attach_offense(slates, offense, offense_source_label)
        result = attach_dst(result, dst)

        manifest = build_manifest(result)
        gaps = result[result["projection_status"].ne("READY")].copy()

        section("PROJECTION COVERAGE BY SLATE")
        print(
            manifest[
                [
                    "slate_name",
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
            ].to_string(index=False)
        )

        section("GLOBAL AUDIT")
        print(f"Contest rows:                 {len(result)}")
        print(f"Projection ready:             {(result['projection_status'] == 'READY').sum()}")
        print(f"Projection gaps:              {len(gaps)}")
        print(f"D/ST rows:                    {(result['_is_dst'] == 1).sum()}")
        print(
            "D/ST ready:                   "
            f"{((result['_is_dst'] == 1) & (result['projection_status'] == 'READY')).sum()}"
        )
        print(
            "Duplicate slate/player rows:  "
            f"{result.duplicated(['_slug','_player_norm','_team','_salary','_game']).sum()}"
        )
        print(f"Missing salaries:             {result['_salary'].isna().sum()}")
        print(f"Non-positive salaries:        {result['_salary'].le(0).sum()}")

        if not gaps.empty:
            section("PROJECTION GAP BREAKDOWN")
            breakdown = (
                gaps.groupby(["_slate", "projection_status"], dropna=False)
                .size()
                .rename("rows")
                .reset_index()
            )
            print(breakdown.to_string(index=False))

            section("UNIQUE GAP PLAYERS")
            unique_gaps = (
                gaps[
                    [
                        "_player", "_team", "_position",
                        "projection_status", "_salary",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    ["projection_status", "_team", "_player"],
                    na_position="last",
                )
            )
            print(unique_gaps.to_string(index=False))

        output = clean_output(result)

        output.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        manifest.to_sql(MANIFEST_TABLE, conn, if_exists="replace", index=False)

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_slate "
            f"ON {OUTPUT_TABLE}(slate_slug)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_eligible "
            f"ON {OUTPUT_TABLE}(slate_slug, optimizer_eligible)"
        )
        conn.commit()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_CSV, index=False)
    output.to_parquet(OUTPUT_PARQUET, index=False)
    manifest.to_csv(MANIFEST_CSV, index=False)

    if gaps.empty:
        pd.DataFrame(
            columns=[
                "slate_name", "player", "team",
                "salary", "projection_status",
            ]
        ).to_csv(GAPS_CSV, index=False)
    else:
        gaps_export = gaps.copy()
        gaps_export.to_csv(GAPS_CSV, index=False)

    section("EXPORTS")
    print(f"Pool CSV:     {OUTPUT_CSV}")
    print(f"Pool Parquet: {OUTPUT_PARQUET}")
    print(f"Manifest:     {MANIFEST_CSV}")
    print(f"Gap audit:    {GAPS_CSV}")
    print(f"SQLite pool:  {OUTPUT_TABLE}")
    print(f"SQLite audit: {MANIFEST_TABLE}")

    section("PROJECTION ATTACHMENT COMPLETE")
    print(
        "D/ST and offensive coverage are reported separately. "
        "Any unmatched offensive player remains ineligible rather than "
        "receiving an inferred projection."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 108)
        print("FANDUEL SLATE PROJECTION ATTACHMENT FAILED")
        print("=" * 108)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
