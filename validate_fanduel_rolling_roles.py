#!/usr/bin/env python3
"""RP-3D: compare 1/2/3-game rolling roles against the next FanDuel result."""
from __future__ import annotations
import json, math, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/"data/fanduel_player_role_performance.db"
OUT=ROOT/"data/audits/fanduel_rolling_role_validation"
WINDOWS=(1,2,3); MIN_SAMPLE=25
FIELDS=("snap","carries","targets","target_share","air_yards","air_share")
def now():return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def avg(rows,key):
    v=[float(r[key]) for r in rows if r[key] is not None]
    return sum(v)/len(v) if v else None
def ge(v,t):return v is not None and v>=t
def role(pos,m):
    s,c,t,ts,a,ash=(m[x] for x in FIELDS)
    if pos=="QB":return "FULL_STARTER" if ge(s,.85) else "PARTIAL_OR_UNKNOWN"
    if pos=="RB":
        if ge(s,.74) and ge(c,18):return "WORKHORSE"
        if ge(s,.5675) and ge(c,13):return "LEAD_BACK"
        if s is None:return "SNAP_DATA_MISSING"
        if s>=.34:return "COMMITTEE"
        return "LOW_PARTICIPATION"
    if pos=="WR":
        if ge(s,.81) and ge(ts,.2857):return "ALPHA_TARGET_FULL_TIME"
        if ge(s,.81) and ge(ts,.20):return "PRIMARY_TARGET_FULL_TIME"
        if ge(s,.81):return "FULL_TIME_RECEIVER"
        if s is None:return "SNAP_DATA_MISSING"
        if s>=.59:return "REGULAR_ROTATION"
        if s>=.25:return "LIMITED_ROTATION"
        return "LOW_PARTICIPATION"
    if pos=="TE":
        elite=ge(t,7) or ge(ts,.2188); strong=ge(t,5) or ge(ts,.1515)
        if ge(s,.73) and elite:return "ELITE_RECEIVING_FULL_TIME"
        if ge(s,.73) and strong:return "STRONG_RECEIVING_FULL_TIME"
        if ge(s,.73):return "FULL_TIME_TE"
        if s is None:return "SNAP_DATA_MISSING"
        if s>=.51:return "REGULAR_ROTATION"
        if s>=.33:return "LIMITED_ROTATION"
        return "LOW_PARTICIPATION"
    raise RuntimeError(f"INVALID_POSITION:{pos}")
def pct(v,p):
    x=sorted(v);k=(len(x)-1)*p;lo=math.floor(k);hi=math.ceil(k)
    return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(k-lo)

def main():
    print("RP-3D FANDUEL ROLLING ROLE VALIDATION");print(f"DATABASE={DB}")
    print("DATABASE_MODE=READ_ONLY");print("TARGET_GAME_INCLUDED_IN_WINDOW=FALSE")
    c=sqlite3.connect(f"file:{DB.as_posix()}?mode=ro",uri=True);c.row_factory=sqlite3.Row;c.execute("PRAGMA query_only=ON")
    rows=c.execute("""SELECT f.season,f.week,f.game_id,f.gsis_id,f.position,
      f.calculated_fanduel_points fd,p.offense_snap_share snap,o.carries,o.targets,
      o.target_share,o.air_yards,o.air_yards_share air_share
      FROM fact_player_game_fanduel f JOIN fact_player_opportunity o
      ON o.season=f.season AND o.week=f.week AND o.game_id=f.game_id AND o.gsis_id=f.gsis_id
      LEFT JOIN fact_player_participation p ON p.season=f.season AND p.week=f.week AND p.game_id=f.game_id AND p.gsis_id=f.gsis_id
      ORDER BY f.season,f.gsis_id,f.week,f.game_id""").fetchall()
    histories=defaultdict(list);position_fd=defaultdict(list)
    for r in rows:histories[(r["season"],r["gsis_id"])].append(r);position_fd[r["position"]].append(float(r["fd"]))
    ceilings={p:pct(v,.90) for p,v in position_fd.items()};groups=defaultdict(list);pairs={w:0 for w in WINDOWS}
    for hist in histories.values():
        for i,target in enumerate(hist):
            for w in WINDOWS:
                if i<w:continue
                prior=hist[i-w:i];metrics={k:avg(prior,k) for k in FIELDS}
                groups[(w,target["position"],role(target["position"],metrics))].append(float(target["fd"]));pairs[w]+=1
    results=[]
    for (w,pos,r),v in sorted(groups.items()):
        results.append({"window":w,"position":pos,"role":r,"n":len(v),
          "mean":round(statistics.fmean(v),3),"median":round(statistics.median(v),3),
          "p75":round(pct(v,.75),3),"ceiling_rate":round(sum(x>=ceilings[pos] for x in v)/len(v),4),
          "sample_gate_passed":len(v)>=MIN_SAMPLE})
    report={"contract":"WFS_FANDUEL_ROLLING_ROLE_VALIDATION_V1","generated_at":now(),
      "target_game_included_in_window":False,"within_season_only":True,"pairs":pairs,
      "minimum_sample":MIN_SAMPLE,"results":results}
    OUT.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp=OUT/f"rolling_role_validation_{stamp}.json";tp=OUT/f"rolling_role_validation_{stamp}.txt"
    jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lines=["RP-3D FANDUEL ROLLING ROLE VALIDATION","="*92,"TARGET_GAME_INCLUDED_IN_WINDOW=FALSE"]
    for w in WINDOWS:
        lines+= ["",f"WINDOW={w}|FORWARD_PAIRS={pairs[w]}"]
        for x in results:
            if x["window"]==w and x["sample_gate_passed"]:
                lines.append(f"{x['position']}|{x['role']}|N={x['n']}|MEAN={x['mean']}|MEDIAN={x['median']}|P75={x['p75']}|CEILING_RATE={x['ceiling_rate']}")
    lines += ["",f"MIN_SAMPLE={MIN_SAMPLE}","STATUS=OK"]
    tp.write_text("\n".join(lines)+"\n",encoding="utf-8");c.close()
    for w in WINDOWS:print(f"WINDOW={w}|FORWARD_PAIRS={pairs[w]}")
    print(f"JSON_REPORT={jp}");print(f"TEXT_REPORT={tp}")
    print("TARGET_GAME_INCLUDED_IN_WINDOW=FALSE");print("STATUS=OK");return 0
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:print(f"ERROR={type(e).__name__}:{e}");print("STATUS=FAILED");sys.exit(1)
