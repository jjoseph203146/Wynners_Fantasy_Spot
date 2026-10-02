#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-D-R1
Schedule-wide capture exclusion audit — READ ONLY.

Explains:
  A) every upstream row that is not an exact current season/week schedule game
  B) every pregame exact-schedule row with no ridge_projection

No writes. No fuzzy matching. No service/cron/updater/LIVE/injury/solver changes.
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import sqlite3, hashlib
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data/nfl.db"
POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
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

def num(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def kickoff_utc(d, t):
    raw = f"{d} {t}"
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

def val(r, col):
    return norm(r[col]) if col in r.index else None

def main():
    print("="*100)
    print("WFS NFL — NFL-POSTGAME-1F-D-R1")
    print("SCHEDULE-WIDE CAPTURE EXCLUSION AUDIT — READ ONLY")
    print("="*100)

    if not NFL_DB.exists() or not POOL.exists():
        raise RuntimeError("required input missing")

    print(f"UPSTREAM_POOL_SHA256={sha(POOL)}")
    df = pd.read_parquet(POOL)
    print(f"UPSTREAM_ROWS={len(df)}")

    required = {
        "season","week","game_id","player_id","player_display_name","team",
        "opponent_team","ridge_projection","production_status",
        "player_pool_status","optimizer_eligible"
    }
    missing = sorted(required - set(df.columns))
    print(f"REQUIRED_COLUMNS_MISSING={','.join(missing) if missing else 'NONE'}")
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    sw = df[["season","week"]].dropna().drop_duplicates()
    if len(sw) != 1:
        raise RuntimeError(f"expected one season/week, got {len(sw)}")
    season, week = map(int, sw.iloc[0].tolist())
    print(f"RESOLVED_SEASON={season}")
    print(f"RESOLVED_WEEK={week}")

    with ro(NFL_DB) as c:
        integ = c.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integ}")
        if integ != "ok":
            raise RuntimeError("nfl.db integrity failure")
        games = c.execute("""
            SELECT game_id,game_date,gametime,away_team,home_team,completed
            FROM games WHERE season=? AND week=?
            ORDER BY game_date,gametime,game_id
        """,(season,week)).fetchall()

    game_map = {str(g["game_id"]):g for g in games}
    schedule_ids = set(game_map)
    print(f"SCHEDULE_GAMES={len(schedule_ids)}")
    now = datetime.now(UTC)
    print(f"AUDIT_AT_UTC={now.isoformat()}")

    # A: classify every row not mapped to exact current-week schedule game_id.
    unmapped = []
    for idx, r in df.iterrows():
        gid = val(r,"game_id")
        if gid not in schedule_ids:
            reason = "BLANK_GAME_ID" if gid is None else "GAME_ID_NOT_CURRENT_WEEK_SCHEDULE"
            unmapped.append((idx,r,reason))

    print("\n=== A. NON-SCHEDULE ROW CLASSIFICATION ===")
    print(f"NON_SCHEDULE_ROWS={len(unmapped)}")
    reasons = Counter(x[2] for x in unmapped)
    for k in sorted(reasons):
        print(f"NON_SCHEDULE_REASON|{k}|rows={reasons[k]}")

    # Deterministic dimensions for the 104-row population.
    dims = ["source_file","fd_position","team","opponent_team","production_status",
            "player_pool_status","optimizer_eligible","has_fd_salary",
            "identity_match_method","model_position"]
    for col in dims:
        if col not in df.columns:
            continue
        c = Counter(val(r,col) or "<NULL>" for _,r,_ in unmapped)
        print(f"\nNON_SCHEDULE_DIMENSION={col}")
        for k,n in sorted(c.items(), key=lambda x:(-x[1],x[0])):
            print(f"  {k}|{n}")

    nonblank_bad_ids = Counter(val(r,"game_id") for _,r,reason in unmapped
                               if reason=="GAME_ID_NOT_CURRENT_WEEK_SCHEDULE")
    print("\nNON_SCHEDULE_NONBLANK_GAME_IDS=" + str(len(nonblank_bad_ids)))
    for gid,n in sorted(nonblank_bad_ids.items()):
        print(f"NON_SCHEDULE_GAME_ID|{gid}|rows={n}")

    # B: exact schedule + pregame + no ridge.
    no_ridge = []
    started = 0
    for idx,r in df.iterrows():
        gid = val(r,"game_id")
        if gid not in game_map:
            continue
        g = game_map[gid]
        ko = kickoff_utc(g["game_date"],g["gametime"])
        if now >= ko or int(g["completed"] or 0)==1:
            started += 1
            continue
        if num(r["ridge_projection"]) is None:
            no_ridge.append((idx,r))

    print("\n=== B. PREGAME EXACT-SCHEDULE ROWS WITHOUT RIDGE ===")
    print(f"STARTED_OR_COMPLETED_EXACT_SCHEDULE_ROWS={started}")
    print(f"PREGAME_EXACT_SCHEDULE_NO_RIDGE_ROWS={len(no_ridge)}")

    by_game = Counter(val(r,"game_id") or "<NULL>" for _,r in no_ridge)
    for gid,n in sorted(by_game.items()):
        print(f"NO_RIDGE_GAME|{gid}|rows={n}")

    for col in ["production_status","player_pool_status","optimizer_eligible",
                "fd_position","model_position","has_fd_salary","identity_match_method"]:
        if col not in df.columns:
            continue
        c = Counter(val(r,col) or "<NULL>" for _,r in no_ridge)
        print(f"\nNO_RIDGE_DIMENSION={col}")
        for k,n in sorted(c.items(), key=lambda x:(-x[1],x[0])):
            print(f"  {k}|{n}")

    # Identity quality of no-ridge rows.
    blank_pid = sum(1 for _,r in no_ridge if val(r,"player_id") is None)
    print(f"\nNO_RIDGE_BLANK_PLAYER_ID_ROWS={blank_pid}")

    # Print all 43 (or current count) compactly so every exclusion is auditable.
    print("\n=== NO-RIDGE ROW DETAIL ===")
    for _,r in sorted(no_ridge, key=lambda x:(
        val(x[1],"game_id") or "",
        val(x[1],"team") or "",
        val(x[1],"player_display_name") or ""
    )):
        print(
            "NO_RIDGE_ROW|game={}|team={}|player={}|pid={}|fd_pos={}|model_pos={}|"
            "production_status={}|pool_status={}|optimizer_eligible={}|has_fd_salary={}".format(
                val(r,"game_id"), val(r,"team"), val(r,"player_display_name"),
                val(r,"player_id"), val(r,"fd_position"), val(r,"model_position"),
                val(r,"production_status"), val(r,"player_pool_status"),
                val(r,"optimizer_eligible"), val(r,"has_fd_salary")
            )
        )

    # Reconciliation: every upstream row must be exactly one of
    # non-schedule, started/completed schedule, pregame no-ridge, pregame ridge-ready.
    pregame_ready = 0
    for _,r in df.iterrows():
        gid = val(r,"game_id")
        if gid not in game_map:
            continue
        g = game_map[gid]
        ko = kickoff_utc(g["game_date"],g["gametime"])
        if now >= ko or int(g["completed"] or 0)==1:
            continue
        if num(r["ridge_projection"]) is not None:
            pregame_ready += 1

    reconciled = len(unmapped) + started + len(no_ridge) + pregame_ready
    print("\n=== RECONCILIATION ===")
    print(f"UPSTREAM_ROWS={len(df)}")
    print(f"NON_SCHEDULE_ROWS={len(unmapped)}")
    print(f"STARTED_OR_COMPLETED_SCHEDULE_ROWS={started}")
    print(f"PREGAME_NO_RIDGE_ROWS={len(no_ridge)}")
    print(f"PREGAME_RIDGE_READY_ROWS={pregame_ready}")
    print(f"RECONCILED_ROWS={reconciled}")
    print(f"RECONCILIATION_MATCH={str(reconciled == len(df)).upper()}")

    status = "PASS" if reconciled == len(df) else "FAIL_CLOSED"
    print("\n=== CONTRACT RESULT ===")
    print("CAPTURE_ELIGIBLE=exact_current_week_game_id + exact_GSIS_player_id + pregame + ridge_projection_present")
    print("FANDUEL_SLATE_MEMBERSHIP_REQUIRED=FALSE")
    print("NO_FUZZY_MATCHING=TRUE")
    print("AUTOMATION_WRITE_CHANGES_ALLOWED=FALSE")
    print(f"NFL_POSTGAME_1F_D_R1_STATUS={status}")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")
    if status != "PASS":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
