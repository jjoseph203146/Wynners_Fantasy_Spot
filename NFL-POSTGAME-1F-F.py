#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-F
Schedule-wide immutable OFFENSIVE projection writer.

Writes ONLY to data/player_projection_ledger.db.
Reads nfl.db read-only and the full upstream nfl_fanduel_player_pool.parquet.

Frozen eligibility contract:
  - exact current season/week schedule game_id
  - exact nonblank upstream player_id
  - kickoff strictly in the future at capture time
  - ridge_projection present
  - offense only (QB/RB/WR/TE)
  - no FanDuel slate-membership requirement
  - no fuzzy/name fallback
  - duplicate (game_id, player_id) fails closed
  - immutable semantic-hash snapshot
  - identical payload => idempotent no-op

This writer uses dedicated schedule-wide ledger tables and does NOT alter the
existing player_projection_snapshots / player_projection_predictions rows.
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import sqlite3, hashlib, json, shutil, sys
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data/nfl.db"
LEDGER_DB = ROOT / "data/player_projection_ledger.db"
POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
OFFENSE = {"QB","RB","WR","TE"}

SNAP_TABLE = "schedule_projection_snapshots"
PRED_TABLE = "schedule_projection_predictions"

def sha_file(p):
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

def ensure_schema(c):
    c.execute(f"""
      CREATE TABLE IF NOT EXISTS {SNAP_TABLE} (
        snapshot_id TEXT PRIMARY KEY,
        captured_at_utc TEXT NOT NULL,
        season INTEGER NOT NULL,
        week INTEGER NOT NULL,
        source_pool_sha256 TEXT NOT NULL,
        source_module TEXT NOT NULL,
        source_module_sha256 TEXT,
        payload_sha256 TEXT NOT NULL UNIQUE,
        row_count INTEGER NOT NULL,
        game_count INTEGER NOT NULL,
        snapshot_status TEXT NOT NULL,
        projection_authority TEXT NOT NULL,
        identity_authority TEXT NOT NULL,
        created_at_utc TEXT NOT NULL
      )
    """)
    c.execute(f"""
      CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
        snapshot_id TEXT NOT NULL,
        season INTEGER NOT NULL,
        week INTEGER NOT NULL,
        game_id TEXT NOT NULL,
        kickoff_utc TEXT NOT NULL,
        player_id TEXT NOT NULL,
        player_display_name TEXT,
        team TEXT,
        opponent_team TEXT,
        position TEXT,
        model_projection REAL NOT NULL,
        projection_authority TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, game_id, player_id),
        FOREIGN KEY (snapshot_id) REFERENCES {SNAP_TABLE}(snapshot_id)
      )
    """)
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{PRED_TABLE}_game_player ON {PRED_TABLE}(game_id,player_id)")
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{SNAP_TABLE}_season_week_capture ON {SNAP_TABLE}(season,week,captured_at_utc)")

def main():
    print("="*100)
    print("WFS NFL — NFL-POSTGAME-1F-F")
    print("SCHEDULE-WIDE IMMUTABLE OFFENSIVE PROJECTION WRITER")
    print("="*100)

    for p in (NFL_DB, LEDGER_DB, POOL):
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    module_path = Path(__file__).resolve()
    pool_sha = sha_file(POOL)
    module_sha = sha_file(module_path)
    print(f"WRITER_SHA256={module_sha}")
    print(f"UPSTREAM_POOL_SHA256={pool_sha}")

    df = pd.read_parquet(POOL)
    required = {"season","week","game_id","player_id","player_display_name",
                "team","opponent_team","ridge_projection"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"missing upstream columns: {missing}")

    sw = df[["season","week"]].dropna().drop_duplicates()
    if len(sw) != 1:
        raise RuntimeError(f"expected exactly one season/week, got {len(sw)}")
    season, week = map(int, sw.iloc[0].tolist())
    print(f"RESOLVED_SEASON={season}")
    print(f"RESOLVED_WEEK={week}")

    with ro(NFL_DB) as n:
        integ = n.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integ}")
        if integ != "ok":
            raise RuntimeError("nfl.db integrity failure")
        games = n.execute("""
          SELECT game_id,game_date,gametime,away_team,home_team,completed
          FROM games WHERE season=? AND week=?
        """,(season,week)).fetchall()

    game_map = {str(g["game_id"]):g for g in games}
    if not game_map:
        raise RuntimeError("no schedule games for resolved season/week")

    capture_dt = datetime.now(UTC)
    capture_iso = capture_dt.isoformat(timespec="seconds")
    rows = []
    exclusion = Counter()

    for _, r in df.iterrows():
        gid = norm(r["game_id"])
        pid = norm(r["player_id"])
        pos = norm(r["fd_position"]) if "fd_position" in df.columns else norm(r["model_position"]) if "model_position" in df.columns else None

        if gid not in game_map:
            exclusion["NO_EXACT_CURRENT_WEEK_GAME"] += 1
            continue
        if not pid:
            exclusion["BLANK_PLAYER_ID"] += 1
            continue
        if pos not in OFFENSE:
            exclusion["NON_OFFENSE"] += 1
            continue

        g = game_map[gid]
        ko = kickoff_utc(g["game_date"],g["gametime"])
        if capture_dt >= ko or int(g["completed"] or 0) == 1:
            exclusion["STARTED_OR_COMPLETED"] += 1
            continue

        proj = number(r["ridge_projection"])
        if proj is None:
            exclusion["NO_RIDGE_PROJECTION"] += 1
            continue

        rows.append({
            "season": season,
            "week": week,
            "game_id": gid,
            "kickoff_utc": ko.isoformat(),
            "player_id": pid,
            "player_display_name": norm(r["player_display_name"]),
            "team": norm(r["team"]),
            "opponent_team": norm(r["opponent_team"]),
            "position": pos,
            "model_projection": proj,
            "projection_authority": "ridge_projection",
        })

    keys = Counter((r["game_id"],r["player_id"]) for r in rows)
    dup = [k for k,n in keys.items() if n != 1]
    print(f"CAPTURE_ELIGIBLE_ROWS={len(rows)}")
    print(f"CAPTURE_ELIGIBLE_GAMES={len(set(r['game_id'] for r in rows))}")
    print(f"DUPLICATE_GAME_PLAYER_KEYS={len(dup)}")
    for k in sorted(exclusion):
        print(f"EXCLUDED_{k}={exclusion[k]}")
    if dup:
        raise RuntimeError(f"duplicate exact game/player keys: {dup[:10]}")
    if not rows:
        raise RuntimeError("zero eligible rows; refusing empty snapshot")

    canonical = sorted(rows, key=lambda r:(r["game_id"],r["player_id"]))
    payload_json = json.dumps(canonical, sort_keys=True, separators=(",",":"), ensure_ascii=False)
    payload_sha = hashlib.sha256(payload_json.encode()).hexdigest()
    snapshot_id = f"{season}-w{week:02d}-schedule-offense-v1-{payload_sha[:16]}"
    print(f"CAPTURE_PAYLOAD_SHA256={payload_sha}")
    print(f"SNAPSHOT_ID={snapshot_id}")

    # Backup before the first possible mutation of the ledger.
    backup = LEDGER_DB.with_name(
        f"{LEDGER_DB.name}.pre-1F-F-{capture_dt.strftime('%Y%m%dT%H%M%SZ')}.bak"
    )
    shutil.copy2(LEDGER_DB, backup)
    print(f"LEDGER_BACKUP={backup}")
    print(f"LEDGER_BACKUP_SHA256={sha_file(backup)}")

    c = sqlite3.connect(LEDGER_DB)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("BEGIN IMMEDIATE")
        ensure_schema(c)

        same_payload = c.execute(
            f"SELECT snapshot_id,row_count FROM {SNAP_TABLE} WHERE payload_sha256=?",
            (payload_sha,)
        ).fetchone()

        if same_payload:
            cnt = c.execute(
                f"SELECT COUNT(*) FROM {PRED_TABLE} WHERE snapshot_id=?",
                (same_payload["snapshot_id"],)
            ).fetchone()[0]
            if cnt != int(same_payload["row_count"]):
                raise RuntimeError("existing immutable snapshot row-count mismatch")
            c.rollback()
            action = "IDEMPOTENT_NOOP"
            snapshot_id = same_payload["snapshot_id"]
            print(f"EXISTING_PREDICTION_ROWS={cnt}")
        else:
            existing_id = c.execute(
                f"SELECT payload_sha256 FROM {SNAP_TABLE} WHERE snapshot_id=?",
                (snapshot_id,)
            ).fetchone()
            if existing_id:
                raise RuntimeError("snapshot_id collision with different payload")

            c.execute(f"""
              INSERT INTO {SNAP_TABLE} (
                snapshot_id,captured_at_utc,season,week,source_pool_sha256,
                source_module,source_module_sha256,payload_sha256,row_count,
                game_count,snapshot_status,projection_authority,
                identity_authority,created_at_utc
              ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(
                snapshot_id,capture_iso,season,week,pool_sha,
                str(module_path),module_sha,payload_sha,len(canonical),
                len(set(r["game_id"] for r in canonical)),
                "PROSPECTIVE_CAPTURE","ridge_projection",
                "exact_game_id_plus_exact_player_id",capture_iso
            ))

            c.executemany(f"""
              INSERT INTO {PRED_TABLE} (
                snapshot_id,season,week,game_id,kickoff_utc,player_id,
                player_display_name,team,opponent_team,position,
                model_projection,projection_authority
              ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,[(
                snapshot_id,r["season"],r["week"],r["game_id"],r["kickoff_utc"],
                r["player_id"],r["player_display_name"],r["team"],
                r["opponent_team"],r["position"],r["model_projection"],
                r["projection_authority"]
            ) for r in canonical])

            inserted = c.execute(
                f"SELECT COUNT(*) FROM {PRED_TABLE} WHERE snapshot_id=?",
                (snapshot_id,)
            ).fetchone()[0]
            if inserted != len(canonical):
                raise RuntimeError(f"insert verification mismatch: {inserted} != {len(canonical)}")
            c.commit()
            action = "INSERTED"
            print(f"INSERTED_PREDICTION_ROWS={inserted}")

        integ = c.execute("PRAGMA integrity_check").fetchone()[0]
        fk = c.execute("PRAGMA foreign_key_check").fetchall()
        snap_count = c.execute(f"SELECT COUNT(*) FROM {SNAP_TABLE}").fetchone()[0]
        pred_count = c.execute(f"SELECT COUNT(*) FROM {PRED_TABLE}").fetchone()[0]
        this_count = c.execute(
            f"SELECT COUNT(*) FROM {PRED_TABLE} WHERE snapshot_id=?",(snapshot_id,)
        ).fetchone()[0]

        print(f"CAPTURE_ACTION={action}")
        print(f"LEDGER_INTEGRITY={integ}")
        print(f"LEDGER_FOREIGN_KEY_ERRORS={len(fk)}")
        print(f"SCHEDULE_SNAPSHOT_ROWS={snap_count}")
        print(f"SCHEDULE_PREDICTION_ROWS={pred_count}")
        print(f"THIS_SNAPSHOT_PREDICTION_ROWS={this_count}")

        ok = (
            integ == "ok" and len(fk) == 0 and
            this_count == len(canonical)
        )
        print(f"NFL_POSTGAME_1F_F_STATUS={'PASS' if ok else 'FAIL_CLOSED'}")
        print("PUBLIC_SOLVER_CHANGES=0")
        print("UPDATER_CHANGES=0")
        print("CRON_CHANGES=0")
        print("LIVE_CHANGES=0")
        print("INJURY_PIPELINE_CHANGES=0")
        print("SERVICE_RESTARTS=0")
        if not ok:
            raise SystemExit(2)
    except Exception:
        try:
            c.rollback()
        except Exception:
            pass
        raise
    finally:
        c.close()

if __name__ == "__main__":
    main()
