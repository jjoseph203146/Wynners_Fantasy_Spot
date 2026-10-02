#!/usr/bin/env python3
"""RP-3E: validate prior-season final-three roles against next-season debut."""
from __future__ import annotations
import json, math, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/"data/fanduel_player_role_performance.db"
OUT=ROOT/"data/audits/fanduel_cross_season_roles"
MIN_SAMPLE=15
FIELDS=("snap","carries","targets","target_share")
def now():return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def avg(rows,key):
    v=[float(r[key]) for r in rows if r[key] is not None];return sum(v)/len(v) if v else None
def ge(v,t):return v is not None and v>=t
def classify(pos,m):
    s,c,t,ts=(m[x] for x in FIELDS)
    if pos=="QB":return "FULL_STARTER" if ge(s,.85) else "PARTIAL_OR_UNKNOWN"
    if pos=="RB":
        if ge(s,.74) and ge(c,18):return "WORKHORSE"
        if ge(s,.5675) and ge(c,13):return "LEAD_BACK"
        if s is None:return "SNAP_DATA_MISSING"
        return "COMMITTEE" if s>=.34 else "LOW_PARTICIPATION"
    if pos=="WR":
        if ge(s,.81) and ge(ts,.2857):return "ALPHA_TARGET_FULL_TIME"
        if ge(s,.81) and ge(ts,.20):return "PRIMARY_TARGET_FULL_TIME"
        if ge(s,.81):return "FULL_TIME_RECEIVER"
        if s is None:return "SNAP_DATA_MISSING"
        return "REGULAR_ROTATION" if s>=.59 else ("LIMITED_ROTATION" if s>=.25 else "LOW_PARTICIPATION")
    if pos=="TE":
        elite=ge(t,7) or ge(ts,.2188);strong=ge(t,5) or ge(ts,.1515)
        if ge(s,.73) and elite:return "ELITE_RECEIVING_FULL_TIME"
        if ge(s,.73) and strong:return "STRONG_RECEIVING_FULL_TIME"
        if ge(s,.73):return "FULL_TIME_TE"
        if s is None:return "SNAP_DATA_MISSING"
        return "REGULAR_ROTATION" if s>=.51 else ("LIMITED_ROTATION" if s>=.33 else "LOW_PARTICIPATION")
def pct(v,p):
    x=sorted(v);k=(len(x)-1)*p;a=math.floor(k);b=math.ceil(k);return x[a] if a==b else x[a]+(x[b]-x[a])*(k-a)

def main():
    print("RP-3E FANDUEL CROSS-SEASON ROLE VALIDATION");print(f"DATABASE={DB}")
    print("DATABASE_MODE=READ_ONLY");print("TARGET_SEASON_DATA_USED_IN_ROLE=FALSE")
    c=sqlite3.connect(f"file:{DB.as_posix()}?mode=ro",uri=True);c.row_factory=sqlite3.Row;c.execute("PRAGMA query_only=ON")
    rows=c.execute("""SELECT f.season,f.week,f.game_id,f.gsis_id,f.position,f.team,
      f.calculated_fanduel_points fd,p.offense_snap_share snap,o.carries,o.targets,o.target_share
      FROM fact_player_game_fanduel f JOIN fact_player_opportunity o
      ON o.season=f.season AND o.week=f.week AND o.game_id=f.game_id AND o.gsis_id=f.gsis_id
      LEFT JOIN fact_player_participation p ON p.season=f.season AND p.week=f.week AND p.game_id=f.game_id AND p.gsis_id=f.gsis_id
      ORDER BY f.gsis_id,f.season,f.week,f.game_id""").fetchall()
    by_player_season=defaultdict(list);position_fd=defaultdict(list)
    for r in rows:by_player_season[(r["gsis_id"],r["season"])].append(r);position_fd[r["position"]].append(float(r["fd"]))
    ceilings={p:pct(v,.90) for p,v in position_fd.items()};groups=defaultdict(list)
    pairs=[]
    for (pid,season),current in by_player_season.items():
        prior=by_player_season.get((pid,season-1))
        if not prior:continue
        final3=prior[-3:];debut=current[0];pos=debut["position"]
        if len(final3)<3 or any(r["position"]!=pos for r in final3):continue
        m={k:avg(final3,k) for k in FIELDS};r=classify(pos,m)
        continuity="SAME_TEAM" if final3[-1]["team"]==debut["team"] else "CHANGED_TEAM"
        groups[(pos,r,continuity)].append(float(debut["fd"]));pairs.append((season,pid,pos,r,continuity))
    results=[]
    for (pos,r,continuity),v in sorted(groups.items()):
        results.append({"position":pos,"role":r,"continuity":continuity,"n":len(v),
          "mean":round(statistics.fmean(v),3),"median":round(statistics.median(v),3),
          "p75":round(pct(v,.75),3),"ceiling_rate":round(sum(x>=ceilings[pos] for x in v)/len(v),4),
          "sample_gate_passed":len(v)>=MIN_SAMPLE})
    report={"contract":"WFS_FANDUEL_CROSS_SEASON_ROLE_VALIDATION_V1","generated_at":now(),
      "prior_games":3,"target_season_data_used_in_role":False,"minimum_sample":MIN_SAMPLE,
      "pairs":len(pairs),"results":results}
    OUT.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp=OUT/f"cross_season_roles_{stamp}.json";tp=OUT/f"cross_season_roles_{stamp}.txt"
    jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lines=["RP-3E FANDUEL CROSS-SEASON ROLE VALIDATION","="*94,f"PAIRS={len(pairs)}","TARGET_SEASON_DATA_USED_IN_ROLE=FALSE"]
    for x in results:
        if x["sample_gate_passed"]:lines.append(f"{x['position']}|{x['role']}|{x['continuity']}|N={x['n']}|MEAN={x['mean']}|MEDIAN={x['median']}|P75={x['p75']}|CEILING_RATE={x['ceiling_rate']}")
    lines += ["",f"MIN_SAMPLE={MIN_SAMPLE}","STATUS=OK"]
    tp.write_text("\n".join(lines)+"\n",encoding="utf-8");c.close()
    print(f"CROSS_SEASON_PAIRS={len(pairs)}");print(f"GROUPS_PASSING_SAMPLE_GATE={sum(x['sample_gate_passed'] for x in results)}")
    print(f"JSON_REPORT={jp}");print(f"TEXT_REPORT={tp}");print("STATUS=OK");return 0
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:print(f"ERROR={type(e).__name__}:{e}");print("STATUS=FAILED");sys.exit(1)
