#!/usr/bin/env python3
"""RP-3A: read-only FanDuel role-evidence distribution audit."""

from __future__ import annotations
import json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DB=ROOT/"data/fanduel_player_role_performance.db"
OUT=ROOT/"data/audits/fanduel_role_distributions"
CONTRACT="WFS_FANDUEL_PLAYER_ROLE_PERFORMANCE_V1"
POSITIONS=("QB","RB","WR","TE")
PCTS=(0.10,0.25,0.50,0.75,0.85,0.90,0.95)
METRICS=("fanduel_points","offense_snaps","offense_snap_share","carries",
         "targets","target_share","air_yards","air_yards_share","wopr")

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def percentile(values,p):
    v=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not v: return None
    k=(len(v)-1)*p; lo=math.floor(k); hi=math.ceil(k)
    return round(v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(k-lo),4)
def summarize(values):
    clean=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {"n":len(clean),"mean":round(sum(clean)/len(clean),4) if clean else None,
            **{f"p{int(p*100):02d}":percentile(clean,p) for p in PCTS}}

def main():
    print("RP-3A FANDUEL ROLE DISTRIBUTION AUDIT")
    print(f"DATABASE={DB}"); print("DATABASE_MODE=READ_ONLY")
    print("ROLE_ASSIGNMENT_PERFORMED=FALSE")
    if not DB.is_file(): raise RuntimeError("FANDUEL_DATABASE_MISSING")
    c=sqlite3.connect(f"file:{DB.as_posix()}?mode=ro",uri=True); c.row_factory=sqlite3.Row
    c.execute("PRAGMA query_only=ON")
    contract=c.execute("SELECT contract FROM schema_contract WHERE singleton_id=1").fetchone()
    if contract is None or contract[0]!=CONTRACT: raise RuntimeError("CONTRACT_MISMATCH")
    rows=c.execute("""SELECT f.season,f.week,f.game_id,f.gsis_id,f.position,
      f.calculated_fanduel_points AS fanduel_points,p.offense_snaps,p.offense_snap_share,
      o.carries,o.targets,o.target_share,o.air_yards,o.air_yards_share,o.weighted_opportunity AS wopr
      FROM fact_player_game_fanduel f
      LEFT JOIN fact_player_participation p
        ON p.season=f.season AND p.week=f.week AND p.game_id=f.game_id AND p.gsis_id=f.gsis_id
      LEFT JOIN fact_player_opportunity o
        ON o.season=f.season AND o.week=f.week AND o.game_id=f.game_id AND o.gsis_id=f.gsis_id
      ORDER BY f.position,f.season,f.week,f.game_id,f.gsis_id""").fetchall()
    duplicate=c.execute("""SELECT COUNT(*) FROM (SELECT season,week,game_id,gsis_id,COUNT(*) n
      FROM fact_player_game_fanduel GROUP BY season,week,game_id,gsis_id HAVING n>1)""").fetchone()[0]
    if duplicate: raise RuntimeError(f"DUPLICATE_PLAYER_GAMES:{duplicate}")
    report={"contract":"WFS_FANDUEL_ROLE_DISTRIBUTION_AUDIT_V1","generated_at":now(),
            "rows":len(rows),"positions":{},"coverage":{}}
    for pos in POSITIONS:
        group=[r for r in rows if r["position"]==pos]
        report["positions"][pos]={m:summarize([r[m] for r in group]) for m in METRICS}
        report["coverage"][pos]={"player_games":len(group),
          "with_snaps":sum(r["offense_snap_share"] is not None for r in group),
          "positive_offense_snaps":sum((r["offense_snaps"] or 0)>0 for r in group),
          "with_opportunity":sum(r["carries"] is not None and r["targets"] is not None for r in group)}
    report["seasons"]={str(s):sum(r["season"]==s for r in rows) for s in sorted({r["season"] for r in rows})}
    c.close(); OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp=OUT/f"fanduel_role_distributions_{stamp}.json"; tp=OUT/f"fanduel_role_distributions_{stamp}.txt"
    jp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    lines=["RP-3A FANDUEL ROLE DISTRIBUTIONS","="*72,f"ROWS={len(rows)}"]
    for pos in POSITIONS:
        cov=report["coverage"][pos]
        lines.append(f"\n{pos}|ROWS={cov['player_games']}|WITH_SNAPS={cov['with_snaps']}|POSITIVE_SNAPS={cov['positive_offense_snaps']}")
        for metric in METRICS:
            s=report["positions"][pos][metric]
            lines.append(f"{metric}|N={s['n']}|P25={s['p25']}|P50={s['p50']}|P75={s['p75']}|P90={s['p90']}|P95={s['p95']}")
    lines.extend(["","ROLE_ASSIGNMENT_PERFORMED=FALSE","STATUS=OK"])
    tp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"PLAYER_GAME_ROWS={len(rows)}")
    for pos in POSITIONS:
        x=report["coverage"][pos]; print(f"{pos}|ROWS={x['player_games']}|WITH_SNAPS={x['with_snaps']}|POSITIVE_SNAPS={x['positive_offense_snaps']}")
    print(f"JSON_REPORT={jp}"); print(f"TEXT_REPORT={tp}")
    print("ROLE_ASSIGNMENT_PERFORMED=FALSE"); print("STATUS=OK"); return 0

if __name__=="__main__":
    try: sys.exit(main())
    except Exception as e:
        print(f"ERROR={type(e).__name__}:{e}"); print("STATUS=FAILED"); sys.exit(1)
