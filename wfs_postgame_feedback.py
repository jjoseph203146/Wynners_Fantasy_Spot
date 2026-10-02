from __future__ import annotations
import math, sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
NFL_DB=ROOT/"data"/"nfl.db"
FORECAST_DB=ROOT/"data"/"forecast_ledger.db"
PLAYER_DB=ROOT/"data"/"player_projection_ledger.db"
ET=ZoneInfo("America/New_York"); UTC=timezone.utc

def _ro(p):
    c=sqlite3.connect(f"file:{p}?mode=ro",uri=True); c.row_factory=sqlite3.Row; return c

def _utc(v):
    d=datetime.fromisoformat(str(v).replace("Z","+00:00"))
    if d.tzinfo is None: raise RuntimeError("naive capture timestamp")
    return d.astimezone(UTC)

def _ko(g):
    raw=f"{str(g['game_date']).strip()} {str(g['gametime']).strip()}"
    for fmt in ("%Y-%m-%d %H:%M","%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(raw,fmt).replace(tzinfo=ET).astimezone(UTC)
        except ValueError: pass
    raise RuntimeError(f"invalid kickoff: {g['game_id']}")

def _game_grades():
    out=[]
    with _ro(NFL_DB) as n, _ro(FORECAST_DB) as f:
        games=n.execute("SELECT game_id,season,week,game_date,gametime,away_team,home_team,away_score,home_score FROM games WHERE completed=1 AND away_score IS NOT NULL AND home_score IS NOT NULL ORDER BY season,week,game_id").fetchall()
        for g in games:
            cand=f.execute(
                "SELECT p.*, s.captured_at_utc, s.historical_proof_status, s.snapshot_status "
                "FROM forecast_predictions p "
                "JOIN forecast_snapshots s ON s.snapshot_id = p.snapshot_id "
                "WHERE p.game_id=? "
                "AND s.historical_proof_status='PASS' "
                "AND p.forecast_status LIKE 'READY%' "
                "ORDER BY s.captured_at_utc, p.snapshot_id",
                (g["game_id"],),
            ).fetchall()
            elig=[x for x in cand if _utc(x["captured_at_utc"])<_ko(g)]
            if not elig: continue
            t=max(_utc(x["captured_at_utc"]) for x in elig); latest=[x for x in elig if _utc(x["captured_at_utc"])==t]
            if len(latest)!=1: continue
            p=latest[0]; am=float(g["home_score"])-float(g["away_score"]); at=float(g["home_score"])+float(g["away_score"])
            aw=str(g["home_team"]) if am>0 else str(g["away_team"]) if am<0 else "TIE"
            out.append({"Week":int(g["week"]),"Game":f"{g['away_team']} @ {g['home_team']}","Game ID":g["game_id"],"Pred Winner":p["pred_winner"],"Actual Winner":aw,"Winner Correct":int(str(p["pred_winner"])==aw),"Margin AE":abs(float(p["pred_home_margin"])-am),"Total AE":abs(float(p["pred_total_points"])-at)})
    return pd.DataFrame(out)

def _player_grades():
    out=[]
    with _ro(NFL_DB) as n, _ro(PLAYER_DB) as l:
        gids={str(r[0]) for r in l.execute("SELECT DISTINCT game_id FROM player_projection_predictions WHERE is_dst=0 AND projection_status='READY'")}
        games=n.execute("SELECT game_id,season,week,game_date,gametime,away_team,home_team FROM games WHERE completed=1 ORDER BY season,week,game_id").fetchall()
        for g in games:
            gid=str(g["game_id"])
            if gid not in gids: continue
            snaps=l.execute("SELECT DISTINCT s.snapshot_id,s.captured_at_utc FROM player_projection_snapshots s JOIN player_projection_predictions p ON p.snapshot_id=s.snapshot_id WHERE p.game_id=? AND p.is_dst=0 AND p.projection_status='READY' AND s.snapshot_status='PROSPECTIVE_CAPTURE' ORDER BY s.captured_at_utc",(gid,)).fetchall()
            elig=[s for s in snaps if _utc(s["captured_at_utc"])<_ko(g)]
            if not elig: continue
            t=max(_utc(s["captured_at_utc"]) for s in elig); latest=[s for s in elig if _utc(s["captured_at_utc"])==t]
            if len(latest)!=1: continue
            sid=latest[0]["snapshot_id"]
            preds=l.execute("SELECT player_id,player_name,team,projection,salary FROM player_projection_predictions WHERE snapshot_id=? AND game_id=? AND is_dst=0 AND projection_status='READY' ORDER BY player_id,slate_slug",(sid,gid)).fetchall()
            pg=defaultdict(list)
            for p in preds: pg[str(p["player_id"])].append(p)
            ag=defaultdict(list)
            for a in n.execute("SELECT player_id,fanduel_points,fanduel_points_verified FROM player_game_stats WHERE game_id=?",(gid,)):
                pid=str(a["player_id"] or "").strip()
                if pid: ag[pid].append(a)
            for pid,pp in pg.items():
                sig={(float(x["projection"]),float(x["salary"]),str(x["player_name"]),str(x["team"])) for x in pp}
                aa=ag.get(pid,[])
                if len(sig)!=1 or len(aa)!=1 or int(aa[0]["fanduel_points_verified"] or 0)!=1: continue
                pred=float(pp[0]["projection"]); actual=float(aa[0]["fanduel_points"])
                out.append({"Week":int(g["week"]),"Game":f"{g['away_team']} @ {g['home_team']}","Player":pp[0]["player_name"],"Team":pp[0]["team"],"Salary":float(pp[0]["salary"]),"Projection":pred,"Actual FD":actual,"Error":pred-actual,"Abs Error":abs(pred-actual)})
    return pd.DataFrame(out)

def render_postgame_feedback_admin():
    st.markdown("### 🧪 Postgame Feedback")
    st.caption("Admin-only read-only grading of immutable pregame WFS forecasts and offensive FanDuel projections against completed results.")
    try:
        games=_game_grades(); players=_player_grades()
    except Exception as exc:
        st.error(f"Postgame feedback unavailable: {exc}"); return
    gt,pt=st.tabs(["Game Forecasts","Player Projections"])
    with gt:
        if games.empty: st.info("No completed game currently has an eligible immutable pregame forecast.")
        else:
            a,b,c=st.columns(3); a.metric("Graded Games",len(games)); b.metric("Winner Accuracy",f"{100*games['Winner Correct'].mean():.1f}%"); c.metric("Margin MAE",f"{games['Margin AE'].mean():.2f}")
            st.dataframe(games,width="stretch",hide_index=True)
    with pt:
        if players.empty: st.info("No completed game currently has an eligible immutable offensive player-projection snapshot.")
        else:
            a,b,c=st.columns(3); a.metric("Graded Players",len(players)); b.metric("MAE",f"{players['Abs Error'].mean():.2f}"); c.metric("RMSE",f"{math.sqrt((players['Error']**2).mean()):.2f}")
            st.dataframe(players.sort_values("Abs Error",ascending=False),width="stretch",hide_index=True)
    st.caption("Exact game/player identity only. D/ST grading is excluded. This view does not modify forecasts, projections, solver inputs, or historical results.")
