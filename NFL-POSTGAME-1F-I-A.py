#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-I-A
DYNAMIC FANDUEL SLATE-DISCOVERY ARCHITECTURE AUDIT — READ ONLY

Purpose:
  Establish what can be proven locally before implementing any FanDuel Research
  slate downloader. This stage does NOT browse, download, write databases,
  modify production files, restart services, or change cron.

Proves:
  - current NFL season/week schedule universe from nfl.db
  - exact game_id uniqueness and matchup uniqueness within season/week
  - existing FanDuel source-file/slate provenance available in the full pool
  - deterministic schedule mapping of existing slate rows
  - safe classification contract:
        1 exact scheduled game -> SINGLE_GAME
        2+ exact scheduled games -> CLASSIC
  - identifies whether current local artifacts are sufficient to discover
    FanDuel Research selector entries (expected to fail closed if not)

No fuzzy matching.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sqlite3
import pandas as pd
import re
import sys

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data" / "nfl.db"
POOL = ROOT / "data" / "parquet" / "nfl_fanduel_player_pool.parquet"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ro_conn(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

def norm_team(x):
    if x is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper().strip())

def section(s):
    print(f"\n=== {s} ===")

print("=" * 110)
print("WFS NFL — NFL-POSTGAME-1F-I-A")
print("DYNAMIC FANDUEL SLATE-DISCOVERY ARCHITECTURE AUDIT — READ ONLY")
print("=" * 110)

fail = []

for p, label in [(DB, "NFL_DB"), (POOL, "FULL_PLAYER_POOL")]:
    print(f"{label}_EXISTS={p.exists()}")
    if p.exists():
        print(f"{label}_SHA256={sha256(p)}")
    else:
        fail.append(f"missing {label}")

if fail:
    print("NFL_POSTGAME_1F_I_A_STATUS=FAIL_CLOSED")
    sys.exit(2)

conn = ro_conn(DB)
integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
print(f"NFL_DB_INTEGRITY={integrity}")
if integrity != "ok":
    fail.append("nfl.db integrity")

# Resolve current season/week from full pool first; this is the same upstream
# artifact whose schedule-wide lineage was proven in prior stages.
pool = pd.read_parquet(POOL)
print(f"FULL_POOL_ROWS={len(pool)}")
print("FULL_POOL_COLUMNS=" + ",".join(map(str, pool.columns)))

required = {"season", "week", "game_id", "source_file"}
missing = sorted(required - set(pool.columns))
print("MISSING_REQUIRED_POOL_COLUMNS=" + (",".join(missing) if missing else "NONE"))
if missing:
    fail.append("missing required pool columns")

if not missing:
    sw = pool[["season", "week"]].dropna().drop_duplicates()
    sw_pairs = sorted((int(r.season), int(r.week)) for r in sw.itertuples(index=False))
    print("POOL_SEASON_WEEK_PAIRS=" + repr(sw_pairs))
    if len(sw_pairs) != 1:
        fail.append("pool does not resolve to exactly one season/week")
        season = week = None
    else:
        season, week = sw_pairs[0]
        print(f"RESOLVED_SEASON={season}")
        print(f"RESOLVED_WEEK={week}")
else:
    season = week = None

section("SCHEDULE AUTHORITY")

# Discover games columns without assuming more than necessary.
game_cols = [r[1] for r in conn.execute("PRAGMA table_info(games)").fetchall()]
print("GAMES_COLUMNS=" + ",".join(game_cols))
needed_games = {"game_id", "season", "week", "away_team", "home_team"}
if not needed_games.issubset(game_cols):
    fail.append("games table lacks required schedule columns")
    sched = pd.DataFrame()
elif season is not None:
    q = """
        SELECT game_id, season, week, away_team, home_team,
               game_date,
               gametime,
               completed,
               away_score,
               home_score
        FROM games
        WHERE season=? AND week=?
        ORDER BY game_id
    """
    # Trim optional fields if schema differs.
    optional = ["game_date","gametime","completed","away_score","home_score"]
    selected_optional = [c for c in optional if c in game_cols]
    q = "SELECT game_id, season, week, away_team, home_team" + (
        ", " + ", ".join(selected_optional) if selected_optional else ""
    ) + " FROM games WHERE season=? AND week=? ORDER BY game_id"
    sched = pd.read_sql_query(q, conn, params=(season, week))
    print(f"SCHEDULE_GAMES={len(sched)}")
    print(f"DISTINCT_SCHEDULE_GAME_IDS={sched['game_id'].nunique()}")
    if len(sched) != sched["game_id"].nunique():
        fail.append("duplicate schedule game_id")
    sched["_away"] = sched["away_team"].map(norm_team)
    sched["_home"] = sched["home_team"].map(norm_team)
    matchup_dups = int(sched.duplicated(["_away","_home"], keep=False).sum())
    print(f"SEASON_WEEK_MATCHUP_DUPLICATE_ROWS={matchup_dups}")
    if matchup_dups:
        fail.append("season/week matchup ambiguity")
    for r in sched.itertuples(index=False):
        print(f"SCHEDULE|{r.game_id}|{r.away_team}@{r.home_team}")
else:
    sched = pd.DataFrame()

section("EXISTING LOCAL SLATE PROVENANCE")

if not missing and not sched.empty:
    p = pool.copy()
    p["game_id"] = p["game_id"].fillna("").astype(str).str.strip()
    schedule_ids = set(sched["game_id"].astype(str))
    nonblank = p[p["game_id"] != ""].copy()
    exact = nonblank[nonblank["game_id"].isin(schedule_ids)].copy()
    unmatched = nonblank[~nonblank["game_id"].isin(schedule_ids)].copy()

    print(f"POOL_NONBLANK_GAME_ID_ROWS={len(nonblank)}")
    print(f"POOL_EXACT_CURRENT_WEEK_GAME_ROWS={len(exact)}")
    print(f"POOL_NONBLANK_UNMATCHED_GAME_ROWS={len(unmatched)}")
    print(f"POOL_EXACT_CURRENT_WEEK_DISTINCT_GAMES={exact['game_id'].nunique()}")

    # source_file is provenance, not assumed to be a slate selector ID.
    sources = sorted(str(x) for x in p["source_file"].dropna().unique())
    print(f"DISTINCT_SOURCE_FILES={len(sources)}")
    for src in sources:
        sub = exact[exact["source_file"].astype(str) == src]
        games = sorted(sub["game_id"].unique())
        print(f"SOURCE_FILE|{src}|rows={len(sub)}|games={len(games)}|game_ids={','.join(games)}")

    # Audit likely slate metadata columns if they already exist.
    candidate_cols = [
        c for c in p.columns
        if any(k in c.lower() for k in ("slate", "contest", "game_info", "gameinfo"))
    ]
    print("LOCAL_SLATE_METADATA_COLUMNS=" + (",".join(candidate_cols) if candidate_cols else "NONE"))
    for c in candidate_cols:
        vals = p[c].dropna().astype(str)
        print(f"METADATA|{c}|nonblank={(vals.str.strip()!='').sum()}|distinct={vals.nunique()}")

section("CLASSIFICATION CONTRACT")

print("SLATE_CLASSIFICATION_RULE|exact_scheduled_game_count=1|SINGLE_GAME")
print("SLATE_CLASSIFICATION_RULE|exact_scheduled_game_count>=2|CLASSIC")
print("SLATE_CLASSIFICATION_RULE|exact_scheduled_game_count=0|REJECT")
print("SLATE_CLASSIFICATION_RULE|ambiguous_or_unmapped_game|REJECT")
print("SLATE_NAME_AUTHORITY=NOT_REQUIRED_FOR_CLASSIFICATION")
print("SCHEDULE_IDENTITY_AUTHORITY=season_plus_week_plus_exact_game_id")
print("FUZZY_MATCHING_ALLOWED=FALSE")
print("PUBLIC_SOLVER_POOL_EXPANSION_ALLOWED=FALSE")
print("SHOWDOWN_AND_CLASSIC_SALARIES_MUST_REMAIN_SLATE_SCOPED=TRUE")

section("DISCOVERY READINESS")

# Local files can prove architecture, but cannot prove the live Research selector
# unless a selector manifest/download artifact already exists. We deliberately
# do not perform web/browser actions in this read-only local audit.
selector_candidates = []
patterns = ("slate", "contest", "research")
for base in [ROOT / "data" / "fanduel", ROOT / "data" / "csv", ROOT / "data" / "parquet"]:
    if base.exists():
        for f in base.iterdir():
            if f.is_file() and any(k in f.name.lower() for k in patterns):
                selector_candidates.append(f)

print(f"LOCAL_SELECTOR_ARTIFACT_CANDIDATES={len(selector_candidates)}")
for f in sorted(selector_candidates):
    print(f"SELECTOR_CANDIDATE|{f}|MTIME_UTC={datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}")

if selector_candidates:
    print("LIVE_RESEARCH_SELECTOR_DISCOVERY_PROVEN=FALSE")
    print("NOTE=local candidate names exist but are not accepted as proof of live selector contents")
else:
    print("LIVE_RESEARCH_SELECTOR_DISCOVERY_PROVEN=FALSE")

print("DOWNLOADER_IMPLEMENTATION_READINESS=FAIL_CLOSED_LIVE_SELECTOR_MECHANISM_NOT_YET_PROVEN")
print("NEXT_REQUIRED_ACTION=audit the FanDuel Research slate selector/download mechanism without modifying production inputs")
print("AUTOMATION_EDIT_ALLOWED=FALSE")

section("RESULT")
status = "PASS" if not fail else "FAIL_CLOSED"
if fail:
    for x in fail:
        print(f"FAIL_REASON={x}")
print("READ_ONLY_AUDIT=TRUE")
print("DATABASE_WRITES=0")
print("FILE_WRITES=0")
print("SERVICE_RESTARTS=0")
print("CRON_CHANGES=0")
print("UPDATER_CHANGES=0")
print("LIVE_CHANGES=0")
print("INJURY_PIPELINE_CHANGES=0")
print("SOLVER_CHANGES=0")
print(f"NFL_POSTGAME_1F_I_A_STATUS={status}")

conn.close()
sys.exit(0 if status == "PASS" else 3)
