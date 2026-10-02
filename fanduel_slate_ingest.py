#!/usr/bin/env python3
"""
fanduel_slate_ingest.py

Dynamic FanDuel NFL slate ingestion.

Purpose
-------
Treat each FanDuel slate CSV as the authoritative contest universe and salary
source. Discover all slate CSVs dynamically, normalize teams/gameInfo/salary,
classify D/ST rows, and publish one normalized pool per slate plus a manifest.

This layer does NOT replace the project's model projections. The source
"fantasy" field is retained only as an external/source projection for audit.

Default source directory:
    data/fanduel/slates/

Optional:
    python fanduel_slate_ingest.py /path/to/slate/files

Outputs:
    data/csv/fanduel_slates/<slug>.csv
    data/parquet/fanduel_slates/<slug>.parquet
    data/csv/fanduel_slate_manifest.csv
    data/csv/audit_fanduel_slate_ingest.csv
    SQLite tables:
        fanduel_slate_manifest
        fanduel_slate_pool
"""

from __future__ import annotations

import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATABASE_PATH, DATA_DIR, CSV_DIR, PARQUET_DIR


DEFAULT_SLATE_DIR = Path(DATA_DIR) / "fanduel" / "slates"
CSV_OUT_DIR = Path(CSV_DIR) / "fanduel_slates"
PARQUET_OUT_DIR = Path(PARQUET_DIR) / "fanduel_slates"
MANIFEST_CSV = Path(CSV_DIR) / "fanduel_slate_manifest.csv"
AUDIT_CSV = Path(CSV_DIR) / "audit_fanduel_slate_ingest.csv"

POOL_TABLE = "fanduel_slate_pool"
MANIFEST_TABLE = "fanduel_slate_manifest"

TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL",
    "baltimore ravens": "BAL", "buffalo bills": "BUF",
    "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE",
    "dallas cowboys": "DAL", "denver broncos": "DEN",
    "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX", "kansas city chiefs": "KC",
    "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "la chargers": "LAC", "los angeles rams": "LA", "la rams": "LA",
    "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO",
    "new york giants": "NYG", "ny giants": "NYG",
    "new york jets": "NYJ", "ny jets": "NYJ",
    "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB", "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

ABBR = {
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


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def norm_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)


def norm_abbr(v):
    if pd.isna(v):
        return None
    s = re.sub(r"[^A-Z]", "", str(v).upper())
    return ABBR.get(s)


def resolve_team(v):
    a = norm_abbr(v)
    if a:
        return a
    t = norm_text(v)
    if t in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[t]
    for suffix in (" d st", " dst", " defense", " def"):
        if t.endswith(suffix):
            t2 = t[:-len(suffix)].strip()
            if t2 in TEAM_NAME_TO_ABBR:
                return TEAM_NAME_TO_ABBR[t2]
    return None


def slugify(name):
    s = norm_text(name)
    return re.sub(r"\s+", "_", s) or "slate"


def parse_salary(series):
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_game_info(value):
    text = str(value).strip().upper()
    m = re.match(r"^\s*([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\s*$", text)
    if not m:
        return None, None
    away = norm_abbr(m.group(1))
    home = norm_abbr(m.group(2))
    return away, home


def is_dst_name(value):
    t = norm_text(value)
    return bool(
        re.search(r"(?:^| )d st(?:$| )", t)
        or t.endswith(" dst")
        or t.endswith(" defense")
    )


def discover_files(source_dir):
    files = sorted(
        p for p in Path(source_dir).glob("*.csv")
        if p.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"No slate CSV files found in {source_dir}"
        )
    return files


def normalize_slate(path):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = {"player", "team", "gameInfo", "salary"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(
            f"{path.name} missing required columns: {', '.join(missing)}"
        )

    out = raw.copy()
    out["slate_name"] = path.stem
    out["slate_slug"] = slugify(path.stem)
    out["source_file"] = path.name
    out["source_row"] = np.arange(1, len(out) + 1)

    out["salary"] = parse_salary(out["salary"])
    out["salary_valid"] = (out["salary"].notna() & out["salary"].gt(0)).astype(int)

    out["team_internal"] = out["team"].map(resolve_team)

    games = out["gameInfo"].map(parse_game_info)
    out["away_team"] = [x[0] for x in games]
    out["home_team"] = [x[1] for x in games]
    out["game_key"] = out.apply(
        lambda r: (
            f"{r['away_team']}@{r['home_team']}"
            if pd.notna(r["away_team"]) and pd.notna(r["home_team"])
            else ""
        ),
        axis=1,
    )

    out["is_dst"] = out["player"].map(is_dst_name).astype(int)

    # Preserve source projection only as an audit/comparison field.
    if "fantasy" in out.columns:
        out["source_fantasy_projection"] = pd.to_numeric(
            out["fantasy"], errors="coerce"
        )
    else:
        out["source_fantasy_projection"] = np.nan

    out["row_valid"] = (
        out["team_internal"].notna()
        & out["away_team"].notna()
        & out["home_team"].notna()
        & out["salary_valid"].eq(1)
    ).astype(int)

    return out


def audit_one(df):
    games = sorted(g for g in df["game_key"].unique() if g)
    teams = set(df["team_internal"].dropna())
    game_teams = set(df["away_team"].dropna()) | set(df["home_team"].dropna())
    dst = df[df["is_dst"].eq(1)]

    return {
        "slate_name": df["slate_name"].iloc[0],
        "slate_slug": df["slate_slug"].iloc[0],
        "source_file": df["source_file"].iloc[0],
        "rows": len(df),
        "games": len(games),
        "teams": len(game_teams),
        "dst_rows": len(dst),
        "valid_salary_rows": int(df["salary_valid"].sum()),
        "invalid_salary_rows": int(df["salary_valid"].eq(0).sum()),
        "unresolved_team_rows": int(df["team_internal"].isna().sum()),
        "invalid_gameinfo_rows": int(
            (df["away_team"].isna() | df["home_team"].isna()).sum()
        ),
        "dst_valid_salary_rows": int(dst["salary_valid"].sum()),
        "dst_team_coverage": len(set(dst["team_internal"].dropna())),
        "all_game_teams_have_dst": int(
            set(dst["team_internal"].dropna()) == game_teams
        ),
        "row_valid_rows": int(df["row_valid"].sum()),
        "game_list": ";".join(games),
    }


def write_sqlite(pool, manifest):
    with sqlite3.connect(DATABASE_PATH) as conn:
        pool.to_sql(POOL_TABLE, conn, if_exists="replace", index=False)
        manifest.to_sql(MANIFEST_TABLE, conn, if_exists="replace", index=False)

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{POOL_TABLE}_slate "
            f"ON {POOL_TABLE}(slate_slug)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{POOL_TABLE}_slate_team "
            f"ON {POOL_TABLE}(slate_slug, team_internal)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{POOL_TABLE}_slate_game "
            f"ON {POOL_TABLE}(slate_slug, game_key)"
        )
        conn.commit()


def main():
    source_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SLATE_DIR

    section("FANDUEL NFL DYNAMIC SLATE INGEST")
    print(f"Source directory: {source_dir}")
    print(f"Database: {DATABASE_PATH}")
    print("Slate CSV = contest universe + salary authority")
    print("Source fantasy projection = audit only")

    files = discover_files(source_dir)
    print(f"Discovered slate files: {len(files)}")

    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pools = []
    audits = []

    for path in files:
        df = normalize_slate(path)
        audit = audit_one(df)

        slug = audit["slate_slug"]
        df.to_csv(CSV_OUT_DIR / f"{slug}.csv", index=False)
        df.to_parquet(PARQUET_OUT_DIR / f"{slug}.parquet", index=False)

        pools.append(df)
        audits.append(audit)

    pool = pd.concat(pools, ignore_index=True)
    manifest = pd.DataFrame(audits)

    # Hard structural failures only. We do not require identical player counts
    # across slates because each contest universe is intentionally different.
    manifest["structural_status"] = np.where(
        (manifest["unresolved_team_rows"] == 0)
        & (manifest["invalid_gameinfo_rows"] == 0)
        & (manifest["invalid_salary_rows"] == 0)
        & (manifest["dst_rows"] == manifest["teams"])
        & (manifest["dst_valid_salary_rows"] == manifest["dst_rows"])
        & (manifest["all_game_teams_have_dst"] == 1),
        "PASS",
        "FAIL",
    )

    section("SLATE MANIFEST / AUDIT")
    show = [
        "slate_name", "rows", "games", "teams", "dst_rows",
        "valid_salary_rows", "invalid_salary_rows",
        "unresolved_team_rows", "invalid_gameinfo_rows",
        "dst_valid_salary_rows", "structural_status",
    ]
    print(manifest[show].to_string(index=False))

    failures = manifest[manifest["structural_status"].eq("FAIL")]
    if not failures.empty:
        manifest.to_csv(AUDIT_CSV, index=False)
        raise RuntimeError(
            "One or more slate files failed structural audit. "
            "SQLite combined pool was not written."
        )

    write_sqlite(pool, manifest)
    manifest.to_csv(MANIFEST_CSV, index=False)
    manifest.to_csv(AUDIT_CSV, index=False)

    section("SLATE INVENTORY")
    for r in manifest.itertuples(index=False):
        print(
            f"{r.slate_name:<20} games={r.games:>2} "
            f"teams={r.teams:>2} players={r.rows:>3} "
            f"D/ST={r.dst_rows:>2}/{r.dst_valid_salary_rows:>2} PASS"
        )
        print(f"  {r.game_list}")

    section("EXPORTS")
    print(f"Manifest: {MANIFEST_CSV}")
    print(f"Audit: {AUDIT_CSV}")
    print(f"Per-slate CSVs: {CSV_OUT_DIR}")
    print(f"Per-slate Parquet: {PARQUET_OUT_DIR}")
    print(f"SQLite pool table: {POOL_TABLE}")
    print(f"SQLite manifest table: {MANIFEST_TABLE}")

    section("FANDUEL SLATE INGEST COMPLETE")
    print(f"Slates loaded: {len(manifest)}")
    print(f"Combined slate rows: {len(pool)}")
    print("All discovered slates passed salary/team/game/D-ST structural gates.")
    print(
        "Next layer: attach internal QB/RB/WR/TE/D-ST projections to each "
        "slate-specific contest pool."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 100)
        print("FANDUEL SLATE INGEST FAILED")
        print("=" * 100)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
