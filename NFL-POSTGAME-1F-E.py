#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-E
Schedule-wide eligible identity compatibility audit — READ ONLY.

Purpose:
  1) Rebuild the exact current capture-eligible offensive population:
       exact current-week game_id
       exact upstream player_id
       pregame
       ridge_projection present
  2) Classify player_id formats.
  3) Test exact compatibility with nfl.db player_game_stats.player_id where
     historical/current actual rows exist.
  4) Prove whether the future writer may safely preserve raw upstream player_id
     without fuzzy/name recovery.

No writes. No service/cron/updater/LIVE/injury/solver changes.
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import sqlite3, hashlib, re
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data/nfl.db"
POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

GSIS_RE = re.compile(r"^\d{2}-\d{7}$")
ALT_RE = re.compile(r"^[A-Z]{3}\d+$")

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

def classify(pid):
    if pid is None:
        return "BLANK"
    if GSIS_RE.fullmatch(pid):
        return "GSIS_NUMERIC"
    if ALT_RE.fullmatch(pid):
        return "ALT_ALPHA_NUMERIC"
    return "OTHER"

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

def main():
    print("=" * 100)
    print("WFS NFL — NFL-POSTGAME-1F-E")
    print("SCHEDULE-WIDE ELIGIBLE IDENTITY COMPATIBILITY AUDIT — READ ONLY")
    print("=" * 100)

    for p in (NFL_DB, POOL):
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    print(f"UPSTREAM_POOL_SHA256={sha(POOL)}")
    df = pd.read_parquet(POOL)
    print(f"UPSTREAM_ROWS={len(df)}")

    required = {
        "season","week","game_id","player_id","player_display_name",
        "team","ridge_projection"
    }
    missing = sorted(required - set(df.columns))
    print(f"REQUIRED_COLUMNS_MISSING={','.join(missing) if missing else 'NONE'}")
    if missing:
        raise RuntimeError(f"missing upstream columns: {missing}")

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
            FROM games
            WHERE season=? AND week=?
            ORDER BY game_date,gametime,game_id
        """,(season,week)).fetchall()

        pgs_cols = {r["name"] for r in c.execute("PRAGMA table_info(player_game_stats)")}
        needed_pgs = {"game_id","player_id","player_display_name","team"}
        missing_pgs = sorted(needed_pgs - pgs_cols)
        print(f"PLAYER_GAME_STATS_REQUIRED_COLUMNS_MISSING={','.join(missing_pgs) if missing_pgs else 'NONE'}")
        if missing_pgs:
            raise RuntimeError(f"player_game_stats missing: {missing_pgs}")

        actual_rows = c.execute("""
            SELECT game_id,player_id,player_display_name,team
            FROM player_game_stats
            WHERE player_id IS NOT NULL AND TRIM(player_id) <> ''
        """).fetchall()

    game_map = {str(g["game_id"]):g for g in games}
    now = datetime.now(UTC)
    print(f"AUDIT_AT_UTC={now.isoformat()}")
    print(f"SCHEDULE_GAMES={len(game_map)}")

    # Exact current eligible population.
    eligible = []
    for _, r in df.iterrows():
        gid = norm(r["game_id"])
        pid = norm(r["player_id"])
        if gid not in game_map or pid is None:
            continue
        g = game_map[gid]
        ko = kickoff_utc(g["game_date"],g["gametime"])
        if now >= ko or int(g["completed"] or 0) == 1:
            continue
        if number(r["ridge_projection"]) is None:
            continue
        eligible.append({
            "game_id": gid,
            "player_id": pid,
            "player": norm(r["player_display_name"]),
            "team": norm(r["team"]),
            "projection": number(r["ridge_projection"]),
        })

    print("\n=== ELIGIBLE POPULATION ===")
    print(f"CAPTURE_ELIGIBLE_ROWS={len(eligible)}")
    print(f"CAPTURE_ELIGIBLE_GAMES={len(set(x['game_id'] for x in eligible))}")

    key_counts = Counter((x["game_id"],x["player_id"]) for x in eligible)
    dup = [k for k,n in key_counts.items() if n != 1]
    print(f"ELIGIBLE_DUPLICATE_GAME_PLAYER_KEYS={len(dup)}")
    if dup:
        for gid,pid in dup[:50]:
            print(f"DUPLICATE_ELIGIBLE_KEY={gid}|{pid}|rows={key_counts[(gid,pid)]}")

    formats = Counter(classify(x["player_id"]) for x in eligible)
    print("\n=== PLAYER_ID FORMAT CLASSIFICATION ===")
    for k,n in sorted(formats.items()):
        print(f"ELIGIBLE_PLAYER_ID_FORMAT|{k}|rows={n}")

    non_gsis = [x for x in eligible if classify(x["player_id"]) != "GSIS_NUMERIC"]
    print(f"ELIGIBLE_NON_GSIS_NUMERIC_ROWS={len(non_gsis)}")
    for x in sorted(non_gsis, key=lambda z:(z["game_id"],z["team"] or "",z["player"] or "")):
        print(
            f"NON_GSIS_ELIGIBLE|game={x['game_id']}|team={x['team']}|"
            f"player={x['player']}|player_id={x['player_id']}|"
            f"format={classify(x['player_id'])}"
        )

    # Actual-stat identity universe.
    actual_by_pid = defaultdict(list)
    actual_exact = set()
    for r in actual_rows:
        pid = norm(r["player_id"])
        gid = norm(r["game_id"])
        if pid:
            actual_by_pid[pid].append(r)
            if gid:
                actual_exact.add((gid,pid))

    print("\n=== PLAYER_GAME_STATS IDENTITY COMPATIBILITY ===")
    print(f"PLAYER_GAME_STATS_ROWS_WITH_PLAYER_ID={len(actual_rows)}")
    print(f"PLAYER_GAME_STATS_DISTINCT_PLAYER_IDS={len(actual_by_pid)}")

    eligible_distinct_pids = sorted(set(x["player_id"] for x in eligible))
    pid_seen_historically = [pid for pid in eligible_distinct_pids if pid in actual_by_pid]
    pid_never_seen = [pid for pid in eligible_distinct_pids if pid not in actual_by_pid]

    print(f"ELIGIBLE_DISTINCT_PLAYER_IDS={len(eligible_distinct_pids)}")
    print(f"ELIGIBLE_PLAYER_IDS_SEEN_IN_PLAYER_GAME_STATS={len(pid_seen_historically)}")
    print(f"ELIGIBLE_PLAYER_IDS_NOT_YET_SEEN_IN_PLAYER_GAME_STATS={len(pid_never_seen)}")

    unseen_formats = Counter(classify(pid) for pid in pid_never_seen)
    for k,n in sorted(unseen_formats.items()):
        print(f"UNSEEN_PLAYER_ID_FORMAT|{k}|rows={n}")

    # Exact same-game join is expected to be zero for pregame games, but report it explicitly.
    exact_same_game = sum(
        1 for x in eligible if (x["game_id"],x["player_id"]) in actual_exact
    )
    print(f"PREGAME_ELIGIBLE_EXACT_SAME_GAME_ACTUAL_ROWS={exact_same_game}")

    # For IDs already seen historically, check whether team/name differences exist.
    historical_name_team_disagree = 0
    samples = []
    for x in eligible:
        hits = actual_by_pid.get(x["player_id"], [])
        if not hits:
            continue
        # Compatibility is ID-based. Name/team are diagnostic only.
        if not any(
            norm(r["player_display_name"]) == x["player"] and norm(r["team"]) == x["team"]
            for r in hits
        ):
            historical_name_team_disagree += 1
            if len(samples) < 30:
                samples.append(x)

    print(f"HISTORICAL_ID_MATCH_WITHOUT_NAME_TEAM_MATCH_ROWS={historical_name_team_disagree}")
    for x in samples:
        print(
            f"HISTORICAL_NAME_TEAM_DIAGNOSTIC|pid={x['player_id']}|"
            f"current={x['player']}|{x['team']}"
        )

    # The writer must preserve exact upstream player_id verbatim.
    # We do not fail merely because a rookie/new ID has no historical actual yet.
    # We do fail on duplicate eligible exact keys or blank IDs (already excluded).
    status = "PASS" if not dup else "FAIL_CLOSED"

    print("\n=== CONTRACT RESULT ===")
    print("WRITER_PLAYER_ID_AUTHORITY=upstream_player_id_verbatim")
    print("WRITER_ID_FORMAT_RESTRICTION=NONE_BEYOND_NONBLANK_EXACT_ID")
    print("POSTGAME_JOIN_KEY=exact_game_id_plus_exact_player_id")
    print("NAME_MATCH_REQUIRED=FALSE")
    print("TEAM_MATCH_REQUIRED_FOR_IDENTITY=FALSE")
    print("FUZZY_RECOVERY_ALLOWED=FALSE")
    print("UNSEEN_NEW_PLAYER_IDS_ALLOWED_AT_CAPTURE=TRUE")
    print("ACTUAL_GRADING_REQUIRES_FUTURE_EXACT_PLAYER_GAME_STATS_JOIN=TRUE")
    print("AUTOMATION_WRITE_CHANGES_ALLOWED=FALSE")
    print(f"NFL_POSTGAME_1F_E_STATUS={status}")
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
