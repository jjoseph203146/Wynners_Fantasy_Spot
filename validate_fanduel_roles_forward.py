#!/usr/bin/env python3
"""RP-3C: leakage-safe next-game validation of observed FanDuel roles."""

from __future__ import annotations
import json, math, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/"data/fanduel_player_role_performance.db"
OUT=ROOT/"data/audits/fanduel_role_forward_validation"
ROLE_CONTRACT="WFS_FANDUEL_OBSERVED_ROLE_V1"
MIN_SAMPLE=25

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def pct(values,p):
    v=sorted(float(x) for x in values)
    if not v:return None
    k=(len(v)-1)*p; lo=math.floor(k); hi=math.ceil(k)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(k-lo)
def summary(values,ceiling):
    v=[float(x) for x in values]
    return {"n":len(v),"mean":round(statistics.fmean(v),3),
            "median":round(statistics.median(v),3),"p75":round(pct(v,.75),3),
            "p90":round(pct(v,.90),3),
            "ceiling_rate":round(sum(x>=ceiling for x in v)/len(v),4)}

def main():
    print("RP-3C FANDUEL ROLE FORWARD VALIDATION")
    print(f"DATABASE={DB}"); print("DATABASE_MODE=READ_ONLY")
    print("VALIDATION_TARGET=NEXT_GAME_FANDUEL_POINTS")
    print("SAME_GAME_OUTCOME_USED=FALSE")
    if not DB.is_file():raise RuntimeError("FANDUEL_DATABASE_MISSING")
    c=sqlite3.connect(f"file:{DB.as_posix()}?mode=ro",uri=True);c.row_factory=sqlite3.Row
    c.execute("PRAGMA query_only=ON")
    rows=c.execute("""SELECT f.season,f.week,f.game_id,f.gsis_id,f.position,
      f.calculated_fanduel_points AS fd_points,r.primary_role
      FROM fact_player_game_fanduel f JOIN fact_player_role_observed r
      ON r.season=f.season AND r.week=f.week AND r.game_id=f.game_id AND r.gsis_id=f.gsis_id
      WHERE r.role_contract=? ORDER BY f.season,f.gsis_id,f.week,f.game_id""",(ROLE_CONTRACT,)).fetchall()
    histories=defaultdict(list)
    for r in rows:histories[(r["season"],r["gsis_id"])].append(r)
    position_points=defaultdict(list)
    for r in rows:position_points[r["position"]].append(float(r["fd_points"]))
    ceilings={p:pct(v,.90) for p,v in position_points.items()}
    forward=defaultdict(list); stable=defaultdict(lambda:[0,0]); pairs=0
    for history in histories.values():
        for prior,nxt in zip(history,history[1:]):
            key=(prior["position"],prior["primary_role"])
            forward[key].append(float(nxt["fd_points"]));pairs+=1
            stable[key][1]+=1;stable[key][0]+=int(prior["primary_role"]==nxt["primary_role"])
    results=[]
    for (position,role),values in sorted(forward.items()):
        item={"position":position,"prior_role":role,
              **summary(values,ceilings[position]),
              "position_ceiling_threshold":round(ceilings[position],3),
              "role_stability_rate":round(stable[(position,role)][0]/stable[(position,role)][1],4),
              "sample_gate_passed":len(values)>=MIN_SAMPLE}
        results.append(item)
    report={"contract":"WFS_FANDUEL_ROLE_FORWARD_VALIDATION_V1","generated_at":now(),
            "role_contract":ROLE_CONTRACT,"same_game_outcome_used":False,
            "within_season_only":True,"minimum_sample":MIN_SAMPLE,
            "player_game_rows":len(rows),"forward_pairs":pairs,
            "position_ceiling_thresholds":ceilings,"roles":results}
    OUT.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp=OUT/f"role_forward_validation_{stamp}.json";tp=OUT/f"role_forward_validation_{stamp}.txt"
    jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lines=["RP-3C FANDUEL NEXT-GAME ROLE VALIDATION","="*88,
           f"PLAYER_GAME_ROWS={len(rows)}",f"FORWARD_PAIRS={pairs}","SAME_GAME_OUTCOME_USED=FALSE"]
    for x in results:
        if x["sample_gate_passed"]:
            lines.append(f"{x['position']}|{x['prior_role']}|N={x['n']}|NEXT_MEAN={x['mean']}|NEXT_MEDIAN={x['median']}|NEXT_P75={x['p75']}|CEILING_RATE={x['ceiling_rate']}|STABILITY={x['role_stability_rate']}")
    lines += ["",f"MIN_SAMPLE={MIN_SAMPLE}","STATUS=OK"]
    tp.write_text("\n".join(lines)+"\n",encoding="utf-8");c.close()
    print(f"PLAYER_GAME_ROWS={len(rows)}");print(f"FORWARD_PAIRS={pairs}")
    print(f"ROLES_PASSING_SAMPLE_GATE={sum(x['sample_gate_passed'] for x in results)}")
    print(f"JSON_REPORT={jp}");print(f"TEXT_REPORT={tp}")
    print("SAME_GAME_OUTCOME_USED=FALSE");print("STATUS=OK");return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"ERROR={type(e).__name__}:{e}");print("STATUS=FAILED");sys.exit(1)
