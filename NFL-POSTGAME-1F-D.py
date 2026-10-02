#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-D
Schedule-wide immutable capture simulation — READ ONLY.

Simulates a historical player-projection capture directly from the full upstream
player pool using:
  games schedule authority
  exact game_id
  exact GSIS player_id
  ridge_projection as WFS model projection authority

Compares the simulated capture with the existing immutable player ledger.
NO WRITES.
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
import sqlite3, hashlib, json
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data/nfl.db"
LEDGER_DB = ROOT / "data/player_projection_ledger.db"
POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ATTACH = ROOT / "fanduel_slate_projection_attach_v5.py"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s or None

def number(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def kickoff_utc(date, time):
    raw = f"{date} {time}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ET).astimezone(UTC)
        except ValueError:
            pass
    raise RuntimeError(f"cannot parse kickoff: {raw}")

def ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c

def main():
    print("="*96)
    print("WFS NFL — NFL-POSTGAME-1F-D")
    print("SCHEDULE-WIDE IMMUTABLE CAPTURE SIMULATION — READ ONLY")
    print("="*96)

    for p in (NFL_DB, LEDGER_DB, POOL, ATTACH):
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    print(f"UPSTREAM_POOL_SHA256={sha(POOL)}")
    print(f"PROJECTION_ATTACH_SHA256={sha(ATTACH)}")

    df = pd.read_parquet(POOL)
    print(f"UPSTREAM_ROWS={len(df)}")

    required = {
        "season","week","game_id","player_id","player_display_name","team",
        "opponent_team","ridge_projection","production_status",
        "player_pool_status","optimizer_eligible"
    }
    missing = sorted(required - set(df.columns))
    print(f"UPSTREAM_REQUIRED_COLUMNS_MISSING={','.join(missing) if missing else 'NONE'}")
    if missing:
        raise RuntimeError(f"missing upstream columns: {missing}")

    sw = df[["season","week"]].dropna().drop_duplicates()
    print(f"UPSTREAM_DISTINCT_SEASON_WEEK={len(sw)}")
    if len(sw) != 1:
        raise RuntimeError("expected exactly one season/week in upstream pool")
    season, week = map(int, sw.iloc[0].tolist())
    print(f"RESOLVED_SEASON={season}")
    print(f"RESOLVED_WEEK={week}")

    with ro(NFL_DB) as n:
        integ = n.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integ}")
        if integ != "ok":
            raise RuntimeError("nfl.db integrity failure")

        games = n.execute("""
            SELECT game_id, season, week, game_date, gametime,
                   away_team, home_team, completed
            FROM games
            WHERE season=? AND week=?
            ORDER BY game_date,gametime,game_id
        """,(season,week)).fetchall()

    game_map = {str(g["game_id"]): g for g in games}
    print(f"SCHEDULE_GAMES={len(game_map)}")

    # Use current time only to simulate whether a capture performed NOW would be pregame.
    now = datetime.now(UTC)
    print(f"SIMULATION_AT_UTC={now.isoformat()}")

    # Exact upstream uniqueness.
    exact = defaultdict(list)
    for _, r in df.iterrows():
        gid, pid = norm(r["game_id"]), norm(r["player_id"])
        if gid and pid:
            exact[(gid,pid)].append(r)
    dup = {k:v for k,v in exact.items() if len(v)!=1}
    print(f"UPSTREAM_EXACT_GAME_PLAYER_KEYS={len(exact)}")
    print(f"UPSTREAM_DUPLICATE_GAME_PLAYER_KEYS={len(dup)}")
    if dup:
        print("SIMULATION_FAIL_REASON=DUPLICATE_EXACT_GAME_PLAYER_KEYS")
        raise SystemExit(2)

    counts = defaultdict(lambda: {"rows":0,"ridge":0,"eligible_now":0,"started":0})
    simulated = []
    bad_game = 0
    blank_id = 0
    no_ridge = 0

    for _, r in df.iterrows():
        gid = norm(r["game_id"])
        pid = norm(r["player_id"])
        if gid not in game_map:
            bad_game += 1
            continue
        if not pid:
            blank_id += 1
            continue

        g = game_map[gid]
        ko = kickoff_utc(g["game_date"], g["gametime"])
        ridge = number(r["ridge_projection"])
        c = counts[gid]
        c["rows"] += 1
        if ridge is not None:
            c["ridge"] += 1

        if now >= ko or int(g["completed"] or 0) == 1:
            c["started"] += 1
            continue

        if ridge is None:
            no_ridge += 1
            continue

        c["eligible_now"] += 1
        simulated.append({
            "season": season,
            "week": week,
            "game_id": gid,
            "player_id": pid,
            "player_name": norm(r["player_display_name"]),
            "team": norm(r["team"]),
            "opponent_team": norm(r["opponent_team"]),
            "projection": ridge,
        })

    print("\n=== GAME-BY-GAME SIMULATION ===")
    for g in games:
        gid = str(g["game_id"])
        ko = kickoff_utc(g["game_date"],g["gametime"])
        c = counts[gid]
        state = "STARTED_OR_COMPLETED" if now >= ko or int(g["completed"] or 0)==1 else "PREGAME"
        print(
            f"{gid}|{g['away_team']}@{g['home_team']}|kickoff={ko.isoformat()}|state={state}|"
            f"upstream_rows={c['rows']}|ridge_rows={c['ridge']}|"
            f"eligible_now={c['eligible_now']}|started_excluded={c['started']}"
        )

    sim_games = sorted({r["game_id"] for r in simulated})
    print("\n=== SIMULATED CAPTURE SUMMARY ===")
    print(f"SIMULATED_CAPTURE_ROWS={len(simulated)}")
    print(f"SIMULATED_CAPTURE_GAMES={len(sim_games)}")
    print(f"ROWS_WITH_GAME_NOT_IN_SCHEDULE={bad_game}")
    print(f"ROWS_WITH_BLANK_PLAYER_ID={blank_id}")
    print(f"PREGAME_ROWS_WITHOUT_RIDGE_PROJECTION={no_ridge}")
    print("SIMULATED_GAME_IDS=" + (",".join(sim_games) if sim_games else "NONE"))

    canonical = sorted(simulated, key=lambda x:(x["game_id"],x["player_id"]))
    payload = json.dumps(canonical, sort_keys=True, separators=(",",":"), ensure_ascii=False)
    print("SIMULATED_PAYLOAD_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())

    with ro(LEDGER_DB) as l:
        integ = l.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"PLAYER_LEDGER_INTEGRITY={integ}")
        if integ != "ok":
            raise RuntimeError("ledger integrity failure")

        snaps = l.execute("""
            SELECT snapshot_id,captured_at_utc,season,week,row_count,
                   offense_row_count,dst_row_count,snapshot_status
            FROM player_projection_snapshots
            ORDER BY captured_at_utc,snapshot_id
        """).fetchall()
        print(f"EXISTING_LEDGER_SNAPSHOTS={len(snaps)}")
        for s in snaps:
            print("LEDGER_SNAPSHOT=" + json.dumps(dict(s),sort_keys=True,default=str))

        existing_games = {
            str(r[0]) for r in l.execute(
                "SELECT DISTINCT game_id FROM player_projection_predictions"
            ).fetchall()
        }

    schedule_ids = set(game_map)
    print(f"EXISTING_LEDGER_DISTINCT_GAME_IDS={len(existing_games)}")
    print(f"SCHEDULE_GAMES_ALREADY_IN_LEDGER={len(schedule_ids & existing_games)}")
    print(f"SCHEDULE_GAMES_NOT_IN_LEDGER={len(schedule_ids - existing_games)}")
    for gid in sorted(schedule_ids - existing_games):
        print(f"LEDGER_MISSING_GAME={gid}")

    print("\n=== CONTRACT RESULT ===")
    print("SCHEDULE_AUTHORITY=games")
    print("OFFENSE_IDENTITY_AUTHORITY=exact_game_id_plus_GSIS_player_id")
    print("OFFENSE_PROJECTION_AUTHORITY=ridge_projection")
    print("FANDUEL_SLATE_MEMBERSHIP_REQUIRED_FOR_HISTORY=FALSE")
    print("STARTED_OR_COMPLETED_GAMES_EXCLUDED_FROM_NEW_CAPTURE=TRUE")
    print("NO_RETROACTIVE_CAPTURE=TRUE")
    print("NO_FUZZY_MATCHING=TRUE")
    print("AUTOMATION_WRITE_CHANGES_ALLOWED=FALSE")
    print("NFL_POSTGAME_1F_D_STATUS=PASS")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
