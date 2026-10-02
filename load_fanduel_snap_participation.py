#!/usr/bin/env python3
"""RP-2C: load exact-ID offensive snap participation into the FanDuel mart."""

from __future__ import annotations
import hashlib, json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/nfl.db"
TARGET = ROOT / "data/fanduel_player_role_performance.db"
CONTRACT = "WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
POSITIONS = {"QB", "RB", "WR", "TE"}

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def clean(v): return "" if v is None else str(v).strip()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def num(v):
    if v is None or clean(v)=="": return None
    x=float(v)
    if not math.isfinite(x): raise RuntimeError(f"NONFINITE_SNAP_VALUE:{v}")
    return x

def main():
    print("RP-2C FANDUEL SNAP PARTICIPATION")
    print(f"SOURCE={SOURCE}"); print(f"TARGET={TARGET}")
    print("SOURCE_DATABASE_MODE=READ_ONLY")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("SOLVER_INTEGRATION_ALLOWED=FALSE")
    if not SOURCE.is_file() or not TARGET.is_file():
        raise RuntimeError("REQUIRED_DATABASE_MISSING")
    s=sqlite3.connect(f"file:{SOURCE.as_posix()}?mode=ro",uri=True)
    s.row_factory=sqlite3.Row; s.execute("PRAGMA query_only=ON")
    duplicate_pfr=s.execute("""SELECT COUNT(*) FROM (
      SELECT pfr_id FROM player_identity WHERE pfr_id IS NOT NULL AND TRIM(pfr_id)<>''
      GROUP BY pfr_id HAVING COUNT(DISTINCT gsis_id)>1)""").fetchone()[0]
    if duplicate_pfr: raise RuntimeError(f"AMBIGUOUS_PLAYER_IDENTITY_PFR:{duplicate_pfr}")
    ambiguous_roster=s.execute("""SELECT COUNT(*) FROM (
      SELECT season,week,pfr_id FROM weekly_rosters
      WHERE pfr_id IS NOT NULL AND TRIM(pfr_id)<>''
      GROUP BY season,week,pfr_id HAVING COUNT(DISTINCT gsis_id)>1)""").fetchone()[0]
    rows=s.execute("""SELECT sc.season,sc.week,sc.game_id,sc.pfr_player_id,
      sc.position,sc.team,sc.offense_snaps,sc.offense_pct,
      pi.gsis_id,pi.full_name,pi.football_name
      FROM player_snap_counts sc LEFT JOIN player_identity pi ON pi.pfr_id=sc.pfr_player_id
      WHERE sc.position IN ('QB','RB','WR','TE')
      ORDER BY sc.season,sc.week,sc.game_id,sc.pfr_player_id""").fetchall()
    matched=[]; excluded={}; seen=set(); missing_2026=0; zero_offense_2026=0
    for r in rows:
        if not clean(r["gsis_id"]):
            key=(int(r["season"]),clean(r["position"]),clean(r["pfr_player_id"]) or "BLANK")
            excluded[key]=excluded.get(key,0)+1
            if int(r["season"])==2026:
                offense_snaps=num(r["offense_snaps"]) or 0.0
                offense_share=num(r["offense_pct"]) or 0.0
                if offense_snaps > 0 or offense_share > 0:
                    missing_2026 += 1
                else:
                    zero_offense_2026 += 1
            continue
        key=(int(r["season"]),int(r["week"]),clean(r["game_id"]),clean(r["gsis_id"]))
        if key in seen: raise RuntimeError(f"DUPLICATE_EXACT_SNAP_KEY:{key}")
        seen.add(key)
        snaps=num(r["offense_snaps"]); share=num(r["offense_pct"])
        if snaps is not None and snaps<0: raise RuntimeError(f"NEGATIVE_SNAPS:{key}")
        if share is not None and not 0<=share<=1: raise RuntimeError(f"INVALID_SNAP_SHARE:{key}:{share}")
        matched.append((key,r,clean(r["gsis_id"]),snaps,share))
    s.close()
    if missing_2026: raise RuntimeError(f"UNRESOLVED_2026_SNAP_ROWS:{missing_2026}")

    t=sqlite3.connect(TARGET); t.row_factory=sqlite3.Row; t.execute("PRAGMA foreign_keys=ON")
    c=t.execute("SELECT contract FROM schema_contract WHERE singleton_id=1").fetchone()
    if c is None or c["contract"]!=CONTRACT: raise RuntimeError("TARGET_CONTRACT_MISMATCH")
    games={r[0] for r in t.execute("SELECT game_id FROM dim_game")}
    players={r[0] for r in t.execute("SELECT gsis_id FROM dim_player")}
    facts=[]; additions={}
    for key,r,gid,snaps,share in matched:
        if key[2] not in games: raise RuntimeError(f"UNMATCHED_GAME_ID:{key[2]}")
        pos=clean(r["position"]).upper(); team=clean(r["team"]).upper()
        if gid not in players:
            name=clean(r["full_name"]) or clean(r["football_name"])
            if not name or pos not in POSITIONS: raise RuntimeError(f"INVALID_DIMENSION_ADDITION:{gid}")
            additions[gid]=(gid,name,pos,key[0],key[0],"EXACT_GSIS")
        facts.append((key[0],key[1],key[2],gid,team,snaps,share,"NFL.DB.PLAYER_SNAP_COUNTS_EXACT_PFR"))
    build="RP2C_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); stamp=now()
    try:
        t.execute("BEGIN IMMEDIATE")
        t.executemany("""INSERT INTO dim_player VALUES (?,?,?,?,?,?)
          ON CONFLICT(gsis_id) DO NOTHING""",list(additions.values()))
        t.execute("DELETE FROM fact_player_participation")
        t.executemany("INSERT INTO fact_player_participation VALUES (?,?,?,?,?,?,?,?)",facts)
        t.execute("DELETE FROM audit_result WHERE build_id IN (SELECT build_id FROM build_manifest WHERE stage='RP-2C')")
        audits=[]
        for (season,pos,pfr),count in sorted(excluded.items()):
            audits.append((build,"SNAP_IDENTITY_EXCLUSION",f"{season}:{pos}:{pfr}","WARN",str(count),"0","PFR_ID_NOT_IN_IDENTITY",stamp))
        audits.append((build,"WEEKLY_ROSTER_PFR_AMBIGUITY",None,"WARN",str(ambiguous_roster),"0","FALLBACK_NOT_USED",stamp))
        t.execute("""INSERT INTO build_manifest(build_id,stage,contract,source_nfl_db_path,
          source_nfl_db_sha256,source_cutoff_at,build_started_at,build_completed_at,status,
          production_tables_modified,solver_integration_allowed,notes)
          VALUES (?,'RP-2C',?,?,?,?,?,?,'OK',0,0,?)""",
          (build,CONTRACT,str(SOURCE),sha(SOURCE),stamp,stamp,now(),f"matched={len(facts)};excluded={sum(excluded.values())}"))
        t.executemany("""INSERT INTO audit_result(build_id,audit_name,entity_key,status,
          observed_value,expected_value,reason_code,audited_at) VALUES (?,?,?,?,?,?,?,?)""",audits)
        fk=t.execute("PRAGMA foreign_key_check").fetchall()
        if fk: raise RuntimeError(f"FOREIGN_KEY_ERRORS:{len(fk)}")
        t.commit()
    except Exception: t.rollback(); raise
    quick=t.execute("PRAGMA quick_check").fetchone()[0]; t.close()
    print(f"EXACT_PARTICIPATION_ROWS={len(facts)}")
    print(f"NEW_EXACT_PLAYERS_ADDED={len(additions)}")
    print(f"HISTORICAL_ROWS_QUARANTINED={sum(excluded.values())}")
    print(f"UNRESOLVED_2026_ROWS={missing_2026}")
    print(f"ZERO_OFFENSE_2026_ROWS_QUARANTINED={zero_offense_2026}")
    print(f"AMBIGUOUS_WEEKLY_ROSTER_MAPPINGS_NOT_USED={ambiguous_roster}")
    print(f"QUICK_CHECK={quick}"); print("STATUS=OK")
    return 0

if __name__=="__main__":
    try: sys.exit(main())
    except Exception as e:
        print(f"ERROR={type(e).__name__}:{e}"); print("STATUS=FAILED"); sys.exit(1)
