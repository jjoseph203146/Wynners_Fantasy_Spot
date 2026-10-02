#!/usr/bin/env python3
"""RP-3B: deterministic postgame FanDuel role classification."""

from __future__ import annotations
import json, sqlite3, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/"data/fanduel_player_role_performance.db"
DB_CONTRACT="WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
ROLE_CONTRACT="WFS_FANDUEL_OBSERVED_ROLE_V1"

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def val(row,key): return None if row[key] is None else float(row[key])
def ge(value,threshold): return value is not None and value>=threshold

def classify(row):
    pos=row["position"]; snap=val(row,"offense_snap_share")
    carries=val(row,"carries"); targets=val(row,"targets")
    tshare=val(row,"target_share"); air=val(row,"air_yards")
    airshare=val(row,"air_yards_share"); wopr=val(row,"wopr")
    reasons=[]
    if pos=="QB":
        primary="FULL_STARTER" if ge(snap,.85) else ("PARTIAL_STARTER" if snap is not None else "SNAP_DATA_MISSING")
        rushing="HIGH_RUSHING" if ge(carries,8) else ("MODERATE_RUSHING" if ge(carries,5) else "LOW_RUSHING")
        receiving="NOT_APPLICABLE"
        reasons += ["QB_SNAP_P85" if primary=="FULL_STARTER" else primary,
                    "QB_CARRIES_P90" if rushing=="HIGH_RUSHING" else rushing]
        required=(snap,carries); available=sum(x is not None for x in required)
    elif pos=="RB":
        if ge(snap,.74) and ge(carries,18): primary="WORKHORSE"
        elif ge(snap,.5675) and ge(carries,13): primary="LEAD_BACK"
        elif snap is None: primary="SNAP_DATA_MISSING"
        elif snap>=.34: primary="COMMITTEE"
        else: primary="LOW_PARTICIPATION"
        rushing="HIGH_VOLUME" if ge(carries,18) else ("LEAD_VOLUME" if ge(carries,13) else ("SECONDARY_VOLUME" if ge(carries,6) else "LOW_VOLUME"))
        receiving="RECEIVING_BACK" if ge(targets,5) or ge(tshare,.1562) else ("SECONDARY_RECEIVING" if ge(targets,3) or ge(tshare,.10) else "LIMITED_RECEIVING")
        reasons += [f"RB_PRIMARY_{primary}",f"RB_RUSH_{rushing}",f"RB_RECEIVE_{receiving}"]
        required=(snap,carries,targets,tshare); available=sum(x is not None for x in required)
    elif pos=="WR":
        full=ge(snap,.81); alpha=ge(tshare,.2857); primary_target=ge(tshare,.20)
        if full and alpha: primary="ALPHA_TARGET_FULL_TIME"
        elif full and primary_target: primary="PRIMARY_TARGET_FULL_TIME"
        elif full: primary="FULL_TIME_RECEIVER"
        elif snap is None: primary="SNAP_DATA_MISSING"
        elif snap>=.59: primary="REGULAR_ROTATION"
        elif snap>=.25: primary="LIMITED_ROTATION"
        else: primary="LOW_PARTICIPATION"
        rushing="DESIGNED_RUSH_USAGE" if ge(carries,1) else "NO_RUSH_USAGE"
        if alpha: receiving="ALPHA_TARGET"
        elif primary_target: receiving="PRIMARY_TARGET"
        elif ge(tshare,.1081): receiving="SECONDARY_TARGET"
        else: receiving="LIMITED_TARGET"
        if ge(air,108) or ge(airshare,.4143): receiving += "_DOWNFIELD"
        reasons += [f"WR_PRIMARY_{primary}",f"WR_RECEIVE_{receiving}"]
        required=(snap,targets,tshare,air,airshare); available=sum(x is not None for x in required)
    elif pos=="TE":
        full=ge(snap,.73); elite=ge(targets,7) or ge(tshare,.2188)
        strong=ge(targets,5) or ge(tshare,.1515)
        if full and elite: primary="ELITE_RECEIVING_FULL_TIME"
        elif full and strong: primary="STRONG_RECEIVING_FULL_TIME"
        elif full: primary="FULL_TIME_TE"
        elif snap is None: primary="SNAP_DATA_MISSING"
        elif snap>=.51: primary="REGULAR_ROTATION"
        elif snap>=.33: primary="LIMITED_ROTATION"
        else: primary="LOW_PARTICIPATION"
        rushing="NO_RUSH_USAGE" if not ge(carries,1) else "DESIGNED_RUSH_USAGE"
        receiving="ELITE_RECEIVING" if elite else ("STRONG_RECEIVING" if strong else ("SECONDARY_RECEIVING" if ge(targets,2) else "LIMITED_RECEIVING"))
        reasons += [f"TE_PRIMARY_{primary}",f"TE_RECEIVE_{receiving}"]
        required=(snap,targets,tshare,air,airshare); available=sum(x is not None for x in required)
    else: raise RuntimeError(f"INVALID_POSITION:{pos}")
    confidence=round(available/len(required),4)
    evidence={"offense_snap_share":snap,"carries":carries,"targets":targets,
              "target_share":tshare,"air_yards":air,"air_yards_share":airshare,"wopr":wopr}
    return primary,rushing,receiving,"UNKNOWN_SOURCE_UNAVAILABLE",confidence,evidence,reasons

def main():
    print("RP-3B FANDUEL OBSERVED ROLES"); print(f"DATABASE={DB}")
    print("FANDUEL_POINTS_USED_FOR_ROLE=FALSE"); print("PRODUCTION_DATABASE_WRITE=FALSE")
    if not DB.is_file(): raise RuntimeError("FANDUEL_DATABASE_MISSING")
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; c.execute("PRAGMA foreign_keys=ON")
    contract=c.execute("SELECT contract FROM schema_contract WHERE singleton_id=1").fetchone()
    if contract is None or contract[0]!=DB_CONTRACT: raise RuntimeError("CONTRACT_MISMATCH")
    rows=c.execute("""SELECT f.season,f.week,f.game_id,f.gsis_id,f.position,
      p.offense_snap_share,o.carries,o.targets,o.target_share,o.air_yards,
      o.air_yards_share,o.weighted_opportunity AS wopr
      FROM fact_player_game_fanduel f
      JOIN fact_player_opportunity o ON o.season=f.season AND o.week=f.week AND o.game_id=f.game_id AND o.gsis_id=f.gsis_id
      LEFT JOIN fact_player_participation p ON p.season=f.season AND p.week=f.week AND p.game_id=f.game_id AND p.gsis_id=f.gsis_id
      ORDER BY f.season,f.week,f.game_id,f.gsis_id""").fetchall()
    stamp=now(); output=[]; counts=Counter(); seen=set()
    for r in rows:
        key=(r["season"],r["week"],r["game_id"],r["gsis_id"])
        if key in seen: raise RuntimeError(f"DUPLICATE_ROLE_KEY:{key}")
        seen.add(key); primary,rush,receive,red,conf,evidence,reasons=classify(r)
        counts[(r["position"],primary)]+=1
        output.append((*key,ROLE_CONTRACT,primary,rush,receive,red,conf,
                       json.dumps(evidence,sort_keys=True,separators=(",",":")),
                       json.dumps(reasons,sort_keys=True,separators=(",",":")),stamp))
    build="RP3B_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        c.execute("BEGIN IMMEDIATE")
        c.execute("DELETE FROM fact_player_role_observed WHERE role_contract=?",(ROLE_CONTRACT,))
        c.executemany("INSERT INTO fact_player_role_observed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",output)
        c.execute("""INSERT INTO build_manifest(build_id,stage,contract,source_cutoff_at,
          build_started_at,build_completed_at,status,production_tables_modified,
          solver_integration_allowed,notes) VALUES (?,'RP-3B',?,?,?,?,'OK',0,0,?)""",
          (build,DB_CONTRACT,stamp,stamp,now(),f"role_contract={ROLE_CONTRACT};rows={len(output)}"))
        fk=c.execute("PRAGMA foreign_key_check").fetchall()
        if fk: raise RuntimeError(f"FOREIGN_KEY_ERRORS:{len(fk)}")
        c.commit()
    except Exception: c.rollback(); raise
    quick=c.execute("PRAGMA quick_check").fetchone()[0]; c.close()
    print(f"ROLE_ROWS={len(output)}"); print(f"ROLE_CONTRACT={ROLE_CONTRACT}")
    for (pos,role),n in sorted(counts.items()): print(f"{pos}|{role}|ROWS={n}")
    print("RED_ZONE_ROLE=UNKNOWN_SOURCE_UNAVAILABLE")
    print(f"QUICK_CHECK={quick}"); print("STATUS=OK"); return 0

if __name__=="__main__":
    try: sys.exit(main())
    except Exception as e:
        print(f"ERROR={type(e).__name__}:{e}"); print("STATUS=FAILED"); sys.exit(1)
