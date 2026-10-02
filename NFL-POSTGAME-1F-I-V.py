#!/usr/bin/env python3
from pathlib import Path
import json
import math

import nflreadpy as nfl
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
SEASONS = [2023, 2024, 2025]

TARGET_IDS = {
    "00-0032569", "00-0033303", "00-0040200",
    "00-0031492", "00-0039498", "00-0034173",
}

REQ = [
    "game_id", "season", "week", "posteam",
    "kicker_player_id", "kicker_player_name",
    "field_goal_attempt", "field_goal_result",
    "extra_point_attempt", "extra_point_result",
    "kick_distance",
]

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan","none","null"} else s

def flag(s):
    return s.fillna(0).astype(str).isin(["1","1.0","True","true"])

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-V")
print("KICKER HISTORICAL FEATURE/MODEL CONTRACT SIMULATION — READ ONLY")
print("=" * 118)

if not IDENTITY.exists():
    print(f"FAIL_CLOSED_MISSING={IDENTITY}")
    raise SystemExit(2)

identity = pd.read_parquet(IDENTITY)
kickers = identity[identity["position"].astype(str).str.upper().eq("K")].copy()

frames = []
for season in SEASONS:
    pl = nfl.load_pbp(season)
    missing = [c for c in REQ if c not in pl.columns]
    if missing:
        print(f"FAIL_CLOSED_SCHEMA|season={season}|missing={json.dumps(missing)}")
        raise SystemExit(2)
    df = pl.select(REQ).to_pandas()
    df = df[flag(df["field_goal_attempt"]) | flag(df["extra_point_attempt"])].copy()
    frames.append(df)

pbp = pd.concat(frames, ignore_index=True)
pbp["kicker_player_id"] = pbp["kicker_player_id"].map(clean)
pbp["kicker_player_name"] = pbp["kicker_player_name"].map(clean)
pbp["posteam"] = pbp["posteam"].map(clean)
pbp["field_goal_result"] = pbp["field_goal_result"].map(lambda x: clean(x).lower())
pbp["extra_point_result"] = pbp["extra_point_result"].map(lambda x: clean(x).lower())
pbp["kick_distance"] = pd.to_numeric(pbp["kick_distance"], errors="coerce")
pbp["fg"] = flag(pbp["field_goal_attempt"]).astype(int)
pbp["xp"] = flag(pbp["extra_point_attempt"]).astype(int)
pbp["fg_made"] = (pbp["fg"].eq(1) & pbp["field_goal_result"].eq("made")).astype(int)
pbp["xp_made"] = (pbp["xp"].eq(1) & pbp["extra_point_result"].isin(["good","made"])).astype(int)

# FanDuel Showdown K scoring buckets established from source:
# <50 made = 3; 50+ made = 5; XP made = 1.
pbp["fd_points"] = 0.0
pbp.loc[pbp["xp_made"].eq(1), "fd_points"] += 1.0
pbp.loc[pbp["fg_made"].eq(1) & pbp["kick_distance"].lt(50), "fd_points"] += 3.0
pbp.loc[pbp["fg_made"].eq(1) & pbp["kick_distance"].ge(50), "fd_points"] += 5.0

print(f"HISTORICAL_KICK_ROWS={len(pbp)}")
print(f"HISTORICAL_UNIQUE_KICKERS={pbp['kicker_player_id'].replace('', pd.NA).nunique(dropna=True)}")

print("\n=== CURRENT KICKER HISTORY ===")
usable = 0
cold = 0

for _, r in kickers.sort_values(["team","player"], kind="mergesort").iterrows():
    pid = clean(r["player_id"])
    name = clean(r["player"])
    team = clean(r["team"])
    h = pbp[pbp["kicker_player_id"].eq(pid)].copy()

    if h.empty:
        cold += 1
        print(f"K_HISTORY|player={name!r}|team={team}|player_id={pid}|events=0|games=0|status=NO_PERSONAL_HISTORY")
        continue

    usable += 1
    game = h.groupby(["season","week","game_id"], as_index=False).agg(
        fg_att=("fg","sum"),
        fg_made=("fg_made","sum"),
        xp_att=("xp","sum"),
        xp_made=("xp_made","sum"),
        fd_points=("fd_points","sum"),
    ).sort_values(["season","week","game_id"], kind="mergesort")

    last8 = game.tail(8)
    last16 = game.tail(16)
    print(
        f"K_HISTORY|player={name!r}|team={team}|player_id={pid}|events={len(h)}|games={len(game)}|"
        f"career_fg_att={int(game['fg_att'].sum())}|career_fg_made={int(game['fg_made'].sum())}|"
        f"career_xp_att={int(game['xp_att'].sum())}|career_xp_made={int(game['xp_made'].sum())}|"
        f"last8_fd_mean={last8['fd_points'].mean():.6f}|last16_fd_mean={last16['fd_points'].mean():.6f}|"
        f"last8_fg_att_mean={last8['fg_att'].mean():.6f}|last8_xp_att_mean={last8['xp_att'].mean():.6f}|"
        f"status=PERSONAL_HISTORY_READY"
    )

print("\n=== LEAGUE/TEAM COLD-START INPUTS ===")
# These are diagnostics only, not a projection.
game_all = pbp[pbp["kicker_player_id"].ne("")].groupby(
    ["season","week","game_id","posteam","kicker_player_id"], as_index=False
).agg(
    fg_att=("fg","sum"),
    fg_made=("fg_made","sum"),
    xp_att=("xp","sum"),
    xp_made=("xp_made","sum"),
    fd_points=("fd_points","sum"),
)

print(f"LEAGUE_KICKER_GAME_ROWS={len(game_all)}")
print(f"LEAGUE_FD_POINTS_MEAN={game_all['fd_points'].mean():.6f}")
print(f"LEAGUE_FG_ATT_MEAN={game_all['fg_att'].mean():.6f}")
print(f"LEAGUE_XP_ATT_MEAN={game_all['xp_att'].mean():.6f}")

for team in sorted(set(kickers["team"].map(clean))):
    tg = game_all[game_all["posteam"].eq(team)].sort_values(["season","week","game_id"], kind="mergesort").tail(16)
    if len(tg):
        print(
            f"TEAM_HISTORY|team={team}|games={len(tg)}|fd_points_mean={tg['fd_points'].mean():.6f}|"
            f"fg_att_mean={tg['fg_att'].mean():.6f}|xp_att_mean={tg['xp_att'].mean():.6f}"
        )
    else:
        print(f"TEAM_HISTORY|team={team}|games=0")

print("\n=== CONTRACT RESULT ===")
print(f"CURRENT_KICKERS={len(kickers)}")
print(f"PERSONAL_HISTORY_READY={usable}")
print(f"NO_PERSONAL_HISTORY={cold}")
print("MODEL_CANDIDATE=RECENCY_WEIGHTED_PERSONAL_HISTORY_WITH_TEAM_LEAGUE_SHRINKAGE")
print("COLD_START_REQUIRES_SEPARATE_FAIL_CLOSED_RULE=TRUE")
print("KICKER_PROJECTION_CREATED=FALSE")
print("LOCAL_FILE_WRITES=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("HISTORICAL_ACTUAL_USED_AS_CURRENT_PROJECTION=FALSE")
print("FUZZY_MATCHING_USED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_V_STATUS=PASS")
