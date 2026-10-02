#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path, PureWindowsPath

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
SG_ROOT = ROOT / "data/fanduel/single_game"
RAW_DIR = SG_ROOT / "raw"
MANIFEST = SG_ROOT / "manifest/slate_readiness_r1.json"
FULL_POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
NFL_DB = ROOT / "data/nfl.db"

DERIVED_DIR = SG_ROOT / "derived"
OUT_PARQUET = DERIVED_DIR / "single_game_identity_pool.parquet"
OUT_REPORT = DERIVED_DIR / "single_game_identity_pool_report.json"

TEAM_ALIAS = {
    "LA": "LAR", "WAS": "WSH",

    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAC",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WSH",
}

RAW_REQUIRED_HEADER = [
    "player", "team", "gameInfo", "salary", "value",
    "completionsAttempts", "passingYards", "passingTouchdowns",
    "interceptionsThrown", "rushingAttempts", "rushingYards",
    "rushingTouchdowns", "receptions", "targets", "receivingYards",
    "receivingTouchdowns", "fantasy", "positionRank", "overallRank",
    "opponentDefensiveRank",
]

ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST", "D/ST", "DEF"}

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", s)

def canon_team(v):
    v = clean(v).upper()
    return TEAM_ALIAS.get(v, v)

def canon_game(v):
    m = re.fullmatch(r"\s*([A-Z]{2,3})\s+@\s+([A-Z]{2,3})\s*", clean(v))
    if not m:
        return None
    return f"{canon_team(m.group(1))} @ {canon_team(m.group(2))}"

def num(v):
    s = clean(v).replace("$", "").replace(",", "")
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def atomic_write_parquet(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

print("=" * 116)
print("WFS NFL — NFL-POSTGAME-1F-I-L")
print("SINGLE-GAME IDENTITY POOL BUILDER — ISOLATED DERIVED OUTPUT ONLY")
print("=" * 116)

for required in [MANIFEST, FULL_POOL, NFL_DB]:
    if not required.exists():
        print(f"FAIL_CLOSED_MISSING={required}")
        raise SystemExit(2)

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
ready_single = [
    s for s in manifest.get("slates", [])
    if s.get("production_eligible") is True
    and s.get("ready") is True
    and s.get("classification") == "SINGLE_GAME"
]

if not ready_single:
    print("FAIL_CLOSED_NO_READY_SINGLE_GAME_SLATES")
    raise SystemExit(2)

pool = pd.read_parquet(FULL_POOL).copy()

required_pool_cols = {
    "fd_name", "team", "fd_game_info", "player_id", "fd_position",
    "ridge_projection", "source_fantasy_projection"
}
missing_pool_cols = sorted(required_pool_cols - set(pool.columns))
if missing_pool_cols:
    print("FAIL_CLOSED_FULL_POOL_COLUMNS_MISSING=" + ",".join(missing_pool_cols))
    raise SystemExit(2)

pool["_name"] = pool["fd_name"].map(clean)
pool["_team"] = pool["team"].map(canon_team)
pool["_game"] = pool["fd_game_info"].map(canon_game)
pool["_player_id"] = pool["player_id"].map(clean)
pool["_position"] = pool["fd_position"].map(clean)

# nfl.db is read-only.
con = sqlite3.connect(f"file:{NFL_DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

# Prove required fallback tables/columns exist.
def table_cols(table):
    return {r["name"] for r in con.execute(f'PRAGMA table_info("{table}")')}

for table, needed in {
    "player_identity": {"gsis_id", "full_name", "position"},
    "weekly_rosters": {"gsis_id", "full_name", "position", "team"},
    "depth_charts": {"gsis_id", "player_name", "team"},
}.items():
    cols = table_cols(table)
    missing = sorted(needed - cols)
    if missing:
        print(f"FAIL_CLOSED_DB_SCHEMA|table={table}|missing={','.join(missing)}")
        con.close()
        raise SystemExit(2)

rows_out = []
failures = []
source_counts = {}

for slate in ready_single:
    label = clean(slate.get("public_slate_name"))
    games = slate.get("canonical_games") or []
    if len(games) != 1:
        failures.append({"slate": label, "reason": "READY_SINGLE_NOT_ONE_GAME"})
        continue

    game = canon_game(games[0])
    raw_name = PureWindowsPath(slate["staging_file"]).name
    raw_path = RAW_DIR / raw_name

    if not raw_path.exists():
        failures.append({"slate": label, "reason": "RAW_FILE_MISSING", "path": str(raw_path)})
        continue

    expected_sha = clean(slate.get("csv_sha256"))
    actual_sha = sha256_file(raw_path)
    if not expected_sha or actual_sha != expected_sha:
        failures.append({
            "slate": label,
            "reason": "RAW_SHA_MISMATCH",
            "expected": expected_sha,
            "actual": actual_sha,
        })
        continue

    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RAW_REQUIRED_HEADER:
            failures.append({"slate": label, "reason": "HEADER_MISMATCH"})
            continue
        raw_rows = list(reader)

    seen_slate_keys = set()

    for raw in raw_rows:
        player = clean(raw.get("player"))
        team = canon_team(raw.get("team"))
        raw_game = canon_game(raw.get("gameInfo"))
        salary = num(raw.get("salary"))

        # Only FanDuel rows with real salary are eligible for the derived identity pool.
        if not player or not team or raw_game != game or salary is None or salary <= 0:
            continue

        unique_key = (label, game, player, team)
        if unique_key in seen_slate_keys:
            failures.append({
                "slate": label,
                "player": player,
                "team": team,
                "game": game,
                "reason": "DUPLICATE_RAW_IDENTITY_ROW",
            })
            continue
        seen_slate_keys.add(unique_key)

        identity_source = None
        player_id = None
        position = None
        ridge_projection = None
        full_pool_source_projection = None

        exact = pool[
            (pool["_name"] == player)
            & (pool["_team"] == team)
            & (pool["_game"] == game)
        ]

        if len(exact):
            ids = sorted({clean(v) for v in exact["_player_id"] if clean(v)})
            poss = sorted({clean(v).upper() for v in exact["_position"] if clean(v)})

            if len(ids) == 1 and len(poss) == 1:
                identity_source = "FULL_POOL_EXACT"
                player_id = ids[0]
                position = poss[0]

                ridge_vals = sorted({
                    float(v) for v in exact["ridge_projection"]
                    if pd.notna(v) and math.isfinite(float(v))
                })
                src_vals = sorted({
                    float(v) for v in exact["source_fantasy_projection"]
                    if pd.notna(v) and math.isfinite(float(v))
                })

                if len(ridge_vals) == 1:
                    ridge_projection = ridge_vals[0]
                elif len(ridge_vals) > 1:
                    failures.append({
                        "slate": label, "player": player, "team": team, "game": game,
                        "reason": "AMBIGUOUS_FULL_POOL_RIDGE_PROJECTION",
                        "values": ridge_vals,
                    })
                    continue

                if len(src_vals) == 1:
                    full_pool_source_projection = src_vals[0]
                elif len(src_vals) > 1:
                    failures.append({
                        "slate": label, "player": player, "team": team, "game": game,
                        "reason": "AMBIGUOUS_FULL_POOL_SOURCE_PROJECTION",
                        "values": src_vals,
                    })
                    continue
            else:
                failures.append({
                    "slate": label, "player": player, "team": team, "game": game,
                    "reason": "AMBIGUOUS_FULL_POOL_IDENTITY",
                    "candidate_rows": len(exact),
                    "player_ids": ids,
                    "positions": poss,
                })
                continue

        if identity_source is None:
            # Exact-name identity fallback. No fuzzy matching.
            identity_rows = con.execute(
                """
                SELECT gsis_id, full_name, position
                FROM player_identity
                WHERE full_name = ?
                """,
                (player,),
            ).fetchall()

            ids = sorted({clean(r["gsis_id"]) for r in identity_rows if clean(r["gsis_id"])})
            poss = sorted({clean(r["position"]).upper() for r in identity_rows if clean(r["position"])})

            if len(ids) != 1 or len(poss) != 1:
                failures.append({
                    "slate": label, "player": player, "team": team, "game": game,
                    "reason": "DB_IDENTITY_NOT_UNIQUE",
                    "player_ids": ids,
                    "positions": poss,
                })
                continue

            candidate_id = ids[0]
            candidate_pos = poss[0]

            # Team compatibility must be proven by an exact-name record in an existing roster/depth source.
            roster_hits = con.execute(
                """
                SELECT gsis_id, full_name, position, team
                FROM weekly_rosters
                WHERE gsis_id = ? AND full_name = ?
                """,
                (candidate_id, player),
            ).fetchall()

            depth_hits = con.execute(
                """
                SELECT gsis_id, player_name, team
                FROM depth_charts
                WHERE gsis_id = ? AND player_name = ?
                """,
                (candidate_id, player),
            ).fetchall()

            compatible = False
            for r in roster_hits:
                if canon_team(r["team"]) == team:
                    compatible = True
                    break
            if not compatible:
                for r in depth_hits:
                    if canon_team(r["team"]) == team:
                        compatible = True
                        break

            if not compatible:
                failures.append({
                    "slate": label, "player": player, "team": team, "game": game,
                    "reason": "DB_TEAM_COMPATIBILITY_NOT_PROVEN",
                    "player_id": candidate_id,
                    "position": candidate_pos,
                })
                continue

            identity_source = "NFL_DB_EXACT_FALLBACK"
            player_id = candidate_id
            position = candidate_pos

        if position == "DEF":
            position = "DST"
        elif position == "D/ST":
            position = "DST"

        if position not in ALLOWED_POSITIONS:
            failures.append({
                "slate": label, "player": player, "team": team, "game": game,
                "reason": "POSITION_NOT_ALLOWED_FOR_SHOWDOWN_AUDIT",
                "position": position,
            })
            continue

        rows_out.append({
            "public_slate_name": label,
            "slate_type": "SINGLE_GAME",
            "game": game,
            "player": player,
            "team": team,
            "salary": int(round(salary)),
            "player_id": player_id,
            "position": position,
            "identity_source": identity_source,
            "raw_fantasy": num(raw.get("fantasy")),
            "raw_value": num(raw.get("value")),
            "ridge_projection": ridge_projection,
            "full_pool_source_fantasy_projection": full_pool_source_projection,
            "source_csv": raw_name,
            "source_csv_sha256": actual_sha,
        })
        source_counts[identity_source] = source_counts.get(identity_source, 0) + 1

con.close()

if failures:
    print(f"FAILURE_COUNT={len(failures)}")
    for f in failures[:100]:
        print("FAIL|" + json.dumps(f, sort_keys=True))
    print("WRITE_OPERATIONS=0")
    print("NFL_POSTGAME_1F_I_L_STATUS=FAIL_CLOSED")
    raise SystemExit(2)

df = pd.DataFrame(rows_out)

if df.empty:
    print("FAIL_CLOSED_EMPTY_DERIVED_POOL")
    raise SystemExit(2)

# Deterministic sort and integrity gates.
df = df.sort_values(
    ["public_slate_name", "game", "team", "position", "player", "player_id"],
    kind="mergesort",
).reset_index(drop=True)

dup_count = int(df.duplicated(
    subset=["public_slate_name", "game", "player", "team"],
    keep=False
).sum())

blank_id_count = int((df["player_id"].map(clean) == "").sum())
blank_pos_count = int((df["position"].map(clean) == "").sum())
bad_salary_count = int((df["salary"] <= 0).sum())

if dup_count or blank_id_count or blank_pos_count or bad_salary_count:
    print(f"FAIL_CLOSED_DUPLICATE_ROWS={dup_count}")
    print(f"FAIL_CLOSED_BLANK_PLAYER_ID_ROWS={blank_id_count}")
    print(f"FAIL_CLOSED_BLANK_POSITION_ROWS={blank_pos_count}")
    print(f"FAIL_CLOSED_BAD_SALARY_ROWS={bad_salary_count}")
    raise SystemExit(2)

# Projection coverage is audited, not silently filled.
ridge_ready = int(df["ridge_projection"].notna().sum())
raw_fantasy_ready = int(df["raw_fantasy"].notna().sum())
projection_missing = int(df["ridge_projection"].isna().sum())

report = {
    "stage": "NFL-POSTGAME-1F-I-L",
    "status": "PASS",
    "ready_single_game_count": len(ready_single),
    "derived_rows": int(len(df)),
    "identity_source_counts": source_counts,
    "position_counts": {
        str(k): int(v) for k, v in df["position"].value_counts(dropna=False).sort_index().items()
    },
    "ridge_projection_ready_rows": ridge_ready,
    "ridge_projection_missing_rows": projection_missing,
    "raw_fantasy_ready_rows": raw_fantasy_ready,
    "manifest_sha256": sha256_file(MANIFEST),
    "full_pool_sha256": sha256_file(FULL_POOL),
    "input_csvs": {
        str(p.name): sha256_file(p)
        for p in sorted(RAW_DIR.glob("*.csv"))
        if p.name in {PureWindowsPath(s["staging_file"]).name for s in ready_single}
    },
    "contracts": {
        "fuzzy_matching": False,
        "classic_production_modified": False,
        "public_solver_modified": False,
        "nfl_db_mode": "read_only",
        "projection_imputation": False,
    },
}

atomic_write_parquet(OUT_PARQUET, df)
report["output_parquet_sha256"] = sha256_file(OUT_PARQUET)
atomic_write_json(OUT_REPORT, report)
report_sha = sha256_file(OUT_REPORT)

print("\n=== SUMMARY ===")
print(f"READY_SINGLE_GAME_COUNT={len(ready_single)}")
print(f"DERIVED_ROWS={len(df)}")
print(f"FULL_POOL_EXACT_ROWS={source_counts.get('FULL_POOL_EXACT', 0)}")
print(f"NFL_DB_EXACT_FALLBACK_ROWS={source_counts.get('NFL_DB_EXACT_FALLBACK', 0)}")
print(f"RIDGE_PROJECTION_READY_ROWS={ridge_ready}")
print(f"RIDGE_PROJECTION_MISSING_ROWS={projection_missing}")
print(f"RAW_FANTASY_READY_ROWS={raw_fantasy_ready}")
print(f"DUPLICATE_ROWS={dup_count}")
print(f"BLANK_PLAYER_ID_ROWS={blank_id_count}")
print(f"BLANK_POSITION_ROWS={blank_pos_count}")
print(f"BAD_SALARY_ROWS={bad_salary_count}")
print(f"OUT_PARQUET={OUT_PARQUET}")
print(f"OUT_PARQUET_SHA256={sha256_file(OUT_PARQUET)}")
print(f"OUT_REPORT={OUT_REPORT}")
print(f"OUT_REPORT_SHA256={report_sha}")
print("FUZZY_MATCHING_USED=FALSE")
print("NFL_DB_WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_L_STATUS=PASS")
