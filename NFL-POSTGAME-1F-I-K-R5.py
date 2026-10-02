#!/usr/bin/env python3
import csv
import json
import re
import sqlite3
from pathlib import Path, PureWindowsPath

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
SG_ROOT = ROOT / "data/fanduel/single_game"
RAW_DIR = SG_ROOT / "raw"
MANIFEST = SG_ROOT / "manifest/slate_readiness_r1.json"
FULL_POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
NFL_DB = ROOT / "data/nfl.db"

TEAM_ALIAS = {
    "LA": "LAR", "WAS": "WSH",
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL", "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL", "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAC", "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV", "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR", "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN", "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT", "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WSH",
}

def clean(v):
    return re.sub(r"\s+", " ", str(v or "")).strip()

def canon_team(v):
    v = clean(v).upper()
    return TEAM_ALIAS.get(v, v)

def canon_game(v):
    m = re.fullmatch(r"\s*([A-Z]{2,3})\s+@\s+([A-Z]{2,3})\s*", clean(v))
    if not m:
        return None
    return f"{canon_team(m.group(1))} @ {canon_team(m.group(2))}"

print("=" * 112)
print("WFS NFL — NFL-POSTGAME-1F-I-K-R5")
print("SINGLE-GAME UNMATCHED IDENTITY DIAGNOSTIC — READ ONLY")
print("=" * 112)

for p in [MANIFEST, FULL_POOL, NFL_DB]:
    if not p.exists():
        print(f"FAIL_CLOSED_MISSING={p}")
        raise SystemExit(2)

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
pool = pd.read_parquet(FULL_POOL).copy()

pool["_name"] = pool["fd_name"].map(clean)
pool["_display"] = pool["player_display_name"].map(clean)
pool["_team"] = pool["team"].map(canon_team)
pool["_game"] = pool["fd_game_info"].map(canon_game)

unmatched = []

for s in manifest.get("slates", []):
    if not (
        s.get("production_eligible") is True
        and s.get("ready") is True
        and s.get("classification") == "SINGLE_GAME"
    ):
        continue

    game = canon_game((s.get("canonical_games") or [""])[0])
    raw_path = RAW_DIR / PureWindowsPath(s["staging_file"]).name

    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        player = clean(r.get("player"))
        team = canon_team(r.get("team"))
        salary_raw = clean(r.get("salary")).replace("$", "").replace(",", "")
        try:
            salary = float(salary_raw)
        except Exception:
            continue
        if not player or salary <= 0 or canon_game(r.get("gameInfo")) != game:
            continue

        exact = pool[
            (pool["_name"] == player)
            & (pool["_team"] == team)
            & (pool["_game"] == game)
        ]
        if exact.empty:
            unmatched.append({
                "label": s["public_slate_name"],
                "player": player,
                "team": team,
                "game": game,
                "salary": salary,
            })

print(f"UNMATCHED_COUNT={len(unmatched)}")

for u in unmatched:
    print("\n" + "-" * 112)
    print(
        f"UNMATCHED|label={u['label']!r}|player={u['player']!r}|"
        f"team={u['team']}|game={u['game']}|salary={u['salary']}"
    )

    by_fd_name = pool[pool["_name"] == u["player"]]
    by_display = pool[pool["_display"] == u["player"]]

    print(f"FULL_POOL_EXACT_FD_NAME_ROWS={len(by_fd_name)}")
    if len(by_fd_name):
        for _, r in by_fd_name.iterrows():
            print(
                "POOL_BY_FD_NAME|"
                f"fd_name={clean(r['fd_name'])!r}|"
                f"display={clean(r['player_display_name'])!r}|"
                f"team={clean(r['team'])}|"
                f"fd_game_info={clean(r['fd_game_info'])!r}|"
                f"player_id={clean(r['player_id'])!r}|"
                f"fd_position={clean(r['fd_position'])!r}|"
                f"game_id={clean(r['game_id'])!r}|"
                f"source_file={clean(r['source_file'])!r}|"
                f"optimizer_eligible={clean(r['optimizer_eligible'])!r}"
            )

    print(f"FULL_POOL_EXACT_DISPLAY_NAME_ROWS={len(by_display)}")
    if len(by_display):
        for _, r in by_display.iterrows():
            print(
                "POOL_BY_DISPLAY_NAME|"
                f"fd_name={clean(r['fd_name'])!r}|"
                f"display={clean(r['player_display_name'])!r}|"
                f"team={clean(r['team'])}|"
                f"fd_game_info={clean(r['fd_game_info'])!r}|"
                f"player_id={clean(r['player_id'])!r}|"
                f"fd_position={clean(r['fd_position'])!r}|"
                f"game_id={clean(r['game_id'])!r}"
            )

# Read-only DB schema/data discovery for unmatched exact names.
uri = f"file:{NFL_DB}?mode=ro"
con = sqlite3.connect(uri, uri=True)
con.row_factory = sqlite3.Row

tables = [
    r["name"]
    for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
]

name_candidates = {
    "player", "player_name", "name", "display_name", "player_display_name",
    "full_name", "gsis_name", "football_name"
}
position_candidates = {"position", "pos", "position_group"}
team_candidates = {"team", "team_abbr", "team_abbreviation", "club", "recent_team"}
id_candidates = {"player_id", "gsis_id", "nfl_id"}

searchable = []
for t in tables:
    cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{t}")')]
    name_cols = [c for c in cols if c.lower() in name_candidates]
    if not name_cols:
        continue
    searchable.append((t, cols, name_cols))

print("\n" + "=" * 112)
print("NFL_DB_EXACT_NAME_SEARCH")
print(f"SEARCHABLE_TABLE_COUNT={len(searchable)}")

for u in unmatched:
    print("\n" + "-" * 112)
    print(f"DB_SEARCH|player={u['player']!r}")
    hits = 0

    for t, cols, name_cols in searchable:
        lower_map = {c.lower(): c for c in cols}
        select_cols = []
        for wanted in list(id_candidates) + list(name_candidates) + list(position_candidates) + list(team_candidates):
            actual = lower_map.get(wanted)
            if actual and actual not in select_cols:
                select_cols.append(actual)

        for nc in name_cols:
            sql = f'SELECT {", ".join([f"""\"{c}\"""" for c in select_cols])} FROM "{t}" WHERE "{nc}" = ? LIMIT 20'
            try:
                rows = con.execute(sql, (u["player"],)).fetchall()
            except sqlite3.Error:
                continue

            for row in rows:
                hits += 1
                payload = "|".join(f"{k}={clean(row[k])!r}" for k in row.keys())
                print(f"DB_HIT|table={t}|match_column={nc}|{payload}")

    print(f"DB_EXACT_NAME_HITS={hits}")

con.close()

print("\n=== RESULT ===")
print("WRITE_OPERATIONS=0")
print("FUZZY_MATCHING_USED=FALSE")
print("NFL_POSTGAME_1F_I_K_R5_STATUS=PASS")
