#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-G
Post-write schedule-wide ledger audit — READ ONLY.

Audits the frozen 1F-F schedule-wide offensive snapshot without writes.
"""

from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import sqlite3, hashlib, json
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data/nfl.db"
LEDGER_DB = ROOT / "data/player_projection_ledger.db"
POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

EXPECTED_WRITER_SHA = "063e1bdfaf34208bd6d56705d808d13b3a21859744731ba49e3ff1bf02857315"
EXPECTED_SNAPSHOT_ID = "2026-w01-schedule-offense-v1-6cd23541aa9d4ad3"
EXPECTED_PAYLOAD_SHA = "6cd23541aa9d4ad3883ae910f94fe8af585c1365c0dd5b1c091749802805e224"
EXPECTED_ROWS = 347
EXPECTED_GAMES = 14

SNAP = "schedule_projection_snapshots"
PRED = "schedule_projection_predictions"
OLD_SNAP = "player_projection_snapshots"
OLD_PRED = "player_projection_predictions"

def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c

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

def table_exists(c, name):
    return c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None

def main():
    print("="*100)
    print("WFS NFL — NFL-POSTGAME-1F-G")
    print("POST-WRITE SCHEDULE-WIDE LEDGER AUDIT — READ ONLY")
    print("="*100)

    for p in (NFL_DB, LEDGER_DB, POOL):
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    writer = ROOT / "NFL-POSTGAME-1F-F.py"
    if not writer.exists():
        raise RuntimeError(f"missing frozen writer: {writer}")
    writer_sha = sha_file(writer)
    print(f"WRITER_SHA256={writer_sha}")
    print(f"WRITER_SHA_MATCH={str(writer_sha == EXPECTED_WRITER_SHA).upper()}")
    if writer_sha != EXPECTED_WRITER_SHA:
        raise RuntimeError("frozen writer hash mismatch")

    pool_sha = sha_file(POOL)
    print(f"CURRENT_UPSTREAM_POOL_SHA256={pool_sha}")

    with ro(NFL_DB) as n:
        integ = n.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integ}")
        if integ != "ok":
            raise RuntimeError("nfl.db integrity failure")
        games = n.execute("""
          SELECT game_id,season,week,game_date,gametime,away_team,home_team,completed
          FROM games WHERE season=2026 AND week=1
        """).fetchall()
    game_map = {str(g["game_id"]):g for g in games}
    print(f"WEEK1_SCHEDULE_GAMES={len(game_map)}")

    with ro(LEDGER_DB) as c:
        integ = c.execute("PRAGMA integrity_check").fetchone()[0]
        fk = c.execute("PRAGMA foreign_key_check").fetchall()
        print(f"LEDGER_INTEGRITY={integ}")
        print(f"LEDGER_FOREIGN_KEY_ERRORS={len(fk)}")
        if integ != "ok" or fk:
            raise RuntimeError("ledger integrity failure")

        for t in (SNAP,PRED,OLD_SNAP,OLD_PRED):
            print(f"TABLE_EXISTS|{t}|{str(table_exists(c,t)).upper()}")
            if not table_exists(c,t):
                raise RuntimeError(f"required table missing: {t}")

        old_snap_count = c.execute(f"SELECT COUNT(*) FROM {OLD_SNAP}").fetchone()[0]
        old_pred_count = c.execute(f"SELECT COUNT(*) FROM {OLD_PRED}").fetchone()[0]
        print(f"LEGACY_SNAPSHOT_ROWS={old_snap_count}")
        print(f"LEGACY_PREDICTION_ROWS={old_pred_count}")

        snap = c.execute(f"SELECT * FROM {SNAP} WHERE snapshot_id=?",
                         (EXPECTED_SNAPSHOT_ID,)).fetchone()
        if not snap:
            raise RuntimeError("expected schedule snapshot missing")

        print(f"SNAPSHOT_ID={snap['snapshot_id']}")
        print(f"SNAPSHOT_CAPTURED_AT_UTC={snap['captured_at_utc']}")
        print(f"SNAPSHOT_SEASON={snap['season']}")
        print(f"SNAPSHOT_WEEK={snap['week']}")
        print(f"SNAPSHOT_SOURCE_POOL_SHA256={snap['source_pool_sha256']}")
        print(f"SNAPSHOT_SOURCE_MODULE_SHA256={snap['source_module_sha256']}")
        print(f"SNAPSHOT_PAYLOAD_SHA256={snap['payload_sha256']}")
        print(f"SNAPSHOT_ROW_COUNT={snap['row_count']}")
        print(f"SNAPSHOT_GAME_COUNT={snap['game_count']}")
        print(f"SNAPSHOT_STATUS={snap['snapshot_status']}")
        print(f"SNAPSHOT_PROJECTION_AUTHORITY={snap['projection_authority']}")
        print(f"SNAPSHOT_IDENTITY_AUTHORITY={snap['identity_authority']}")

        rows = c.execute(f"""
          SELECT * FROM {PRED}
          WHERE snapshot_id=?
          ORDER BY game_id,player_id
        """,(EXPECTED_SNAPSHOT_ID,)).fetchall()

        dup = c.execute(f"""
          SELECT game_id,player_id,COUNT(*) n
          FROM {PRED} WHERE snapshot_id=?
          GROUP BY game_id,player_id HAVING COUNT(*)<>1
        """,(EXPECTED_SNAPSHOT_ID,)).fetchall()

    print("\n=== IMMUTABILITY / STRUCTURE ===")
    print(f"ACTUAL_PREDICTION_ROWS={len(rows)}")
    print(f"ACTUAL_DISTINCT_GAMES={len(set(r['game_id'] for r in rows))}")
    print(f"DUPLICATE_GAME_PLAYER_KEYS={len(dup)}")

    captured_at = datetime.fromisoformat(str(snap["captured_at_utc"]))
    if captured_at.tzinfo is None:
        raise RuntimeError("captured_at_utc is timezone-naive")

    bad_schedule = 0
    not_pregame = 0
    kickoff_mismatch = 0
    blank_id = 0
    bad_authority = 0

    for r in rows:
        gid = str(r["game_id"])
        pid = norm(r["player_id"])
        if gid not in game_map:
            bad_schedule += 1
            continue
        g = game_map[gid]
        ko = kickoff_utc(g["game_date"],g["gametime"])
        if captured_at >= ko:
            not_pregame += 1
        stored_ko = datetime.fromisoformat(str(r["kickoff_utc"]))
        if stored_ko != ko:
            kickoff_mismatch += 1
        if not pid:
            blank_id += 1
        if r["projection_authority"] != "ridge_projection":
            bad_authority += 1

    print(f"ROWS_GAME_NOT_IN_WEEK1_SCHEDULE={bad_schedule}")
    print(f"ROWS_NOT_STRICTLY_PREGAME_AT_CAPTURE={not_pregame}")
    print(f"ROWS_KICKOFF_UTC_MISMATCH={kickoff_mismatch}")
    print(f"ROWS_BLANK_PLAYER_ID={blank_id}")
    print(f"ROWS_BAD_PROJECTION_AUTHORITY={bad_authority}")

    # Recompute semantic payload from immutable rows exactly as 1F-F did.
    canonical = []
    for r in rows:
        canonical.append({
            "season": int(r["season"]),
            "week": int(r["week"]),
            "game_id": str(r["game_id"]),
            "kickoff_utc": str(r["kickoff_utc"]),
            "player_id": str(r["player_id"]),
            "player_display_name": norm(r["player_display_name"]),
            "team": norm(r["team"]),
            "opponent_team": norm(r["opponent_team"]),
            "position": norm(r["position"]),
            "model_projection": float(r["model_projection"]),
            "projection_authority": str(r["projection_authority"]),
        })
    canonical.sort(key=lambda x:(x["game_id"],x["player_id"]))
    payload = json.dumps(canonical,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    recomputed_sha = hashlib.sha256(payload.encode()).hexdigest()
    print(f"RECOMPUTED_PAYLOAD_SHA256={recomputed_sha}")
    print(f"PAYLOAD_HASH_MATCH={str(recomputed_sha == EXPECTED_PAYLOAD_SHA == snap['payload_sha256']).upper()}")

    # Exact preservation against current upstream is diagnostic. If the source has
    # legitimately changed since capture, immutable snapshot remains valid.
    df = pd.read_parquet(POOL)
    upstream = {}
    upstream_dup = 0
    for _,r in df.iterrows():
        gid,pid = norm(r.get("game_id")),norm(r.get("player_id"))
        if not gid or not pid:
            continue
        k=(gid,pid)
        if k in upstream:
            upstream_dup += 1
        upstream[k]=number(r.get("ridge_projection"))

    comparable = 0
    exact_proj = 0
    missing_current = 0
    changed_current = 0
    for r in rows:
        k=(str(r["game_id"]),str(r["player_id"]))
        if k not in upstream:
            missing_current += 1
            continue
        up=upstream[k]
        if up is None:
            changed_current += 1
            continue
        comparable += 1
        if float(r["model_projection"]) == float(up):
            exact_proj += 1
        else:
            changed_current += 1

    print("\n=== CURRENT-UPSTREAM DIAGNOSTIC ===")
    print(f"CURRENT_UPSTREAM_DUPLICATE_GAME_PLAYER_KEYS={upstream_dup}")
    print(f"SNAPSHOT_ROWS_COMPARABLE_TO_CURRENT_UPSTREAM={comparable}")
    print(f"SNAPSHOT_PROJECTION_EXACT_MATCH_CURRENT_UPSTREAM={exact_proj}")
    print(f"SNAPSHOT_ROWS_MISSING_FROM_CURRENT_UPSTREAM={missing_current}")
    print(f"SNAPSHOT_ROWS_CHANGED_VS_CURRENT_UPSTREAM={changed_current}")
    print(f"SOURCE_POOL_UNCHANGED_SINCE_CAPTURE={str(pool_sha == snap['source_pool_sha256']).upper()}")

    # Backup evidence: require at least the known pre-write backup.
    backups = sorted(LEDGER_DB.parent.glob("player_projection_ledger.db.pre-1F-F-*.bak"))
    print("\n=== BACKUP EVIDENCE ===")
    print(f"PRE_1F_F_BACKUP_FILES={len(backups)}")
    for b in backups[-5:]:
        print(f"BACKUP|{b.name}|sha256={sha_file(b)}")

    ok = all([
        writer_sha == EXPECTED_WRITER_SHA,
        int(snap["season"]) == 2026,
        int(snap["week"]) == 1,
        snap["payload_sha256"] == EXPECTED_PAYLOAD_SHA,
        int(snap["row_count"]) == EXPECTED_ROWS,
        int(snap["game_count"]) == EXPECTED_GAMES,
        snap["snapshot_status"] == "PROSPECTIVE_CAPTURE",
        snap["projection_authority"] == "ridge_projection",
        snap["identity_authority"] == "exact_game_id_plus_exact_player_id",
        len(rows) == EXPECTED_ROWS,
        len(set(r["game_id"] for r in rows)) == EXPECTED_GAMES,
        len(dup) == 0,
        bad_schedule == 0,
        not_pregame == 0,
        kickoff_mismatch == 0,
        blank_id == 0,
        bad_authority == 0,
        recomputed_sha == EXPECTED_PAYLOAD_SHA,
        len(backups) >= 1,
    ])

    print("\n=== CONTRACT RESULT ===")
    print("LEGACY_TABLES_WRITE_TARGET=FALSE")
    print("SCHEDULE_WIDE_TABLES_WRITE_TARGET=TRUE")
    print("GRADER_SELECTION_CONTRACT=latest_eligible_PROSPECTIVE_CAPTURE_strictly_before_kickoff")
    print("GRADER_JOIN_KEY=exact_game_id_plus_exact_player_id")
    print("FUZZY_RECOVERY_ALLOWED=FALSE")
    print("AUTOMATION_CHANGES_ALLOWED=FALSE")
    print(f"NFL_POSTGAME_1F_G_STATUS={'PASS' if ok else 'FAIL_CLOSED'}")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
