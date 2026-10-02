#!/usr/bin/env python3
from pathlib import Path
import json
import math

import nflreadpy as nfl
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
SEASONS = [2023, 2024, 2025]

TEAM_ALIASES = {"LA": "LAR", "WAS": "WSH"}

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

def canon_team(v):
    s = clean(v).upper()
    return TEAM_ALIASES.get(s, s)

def flag(s):
    return s.fillna(0).astype(str).isin(["1","1.0","True","true"])

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-W")
print("KICKER PROJECTION MODEL SIMULATION — READ ONLY")
print("=" * 118)

if not IDENTITY.exists():
    print(f"FAIL_CLOSED_MISSING={IDENTITY}")
    raise SystemExit(2)

identity = pd.read_parquet(IDENTITY)
kickers = identity[identity["position"].astype(str).str.upper().eq("K")].copy()
if len(kickers) != 6:
    print(f"FAIL_CLOSED_KICKER_COUNT={len(kickers)}")
    raise SystemExit(2)

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
pbp["posteam"] = pbp["posteam"].map(canon_team)
pbp["field_goal_result"] = pbp["field_goal_result"].map(lambda x: clean(x).lower())
pbp["extra_point_result"] = pbp["extra_point_result"].map(lambda x: clean(x).lower())
pbp["kick_distance"] = pd.to_numeric(pbp["kick_distance"], errors="coerce")
pbp["fg"] = flag(pbp["field_goal_attempt"]).astype(int)
pbp["xp"] = flag(pbp["extra_point_attempt"]).astype(int)
pbp["fg_made"] = (pbp["fg"].eq(1) & pbp["field_goal_result"].eq("made")).astype(int)
pbp["xp_made"] = (pbp["xp"].eq(1) & pbp["extra_point_result"].isin(["good","made"])).astype(int)

# FanDuel Showdown scoring already established:
# made FG <50 = 3, made FG >=50 = 5, made XP = 1.
pbp["fd_points"] = 0.0
pbp.loc[pbp["xp_made"].eq(1), "fd_points"] += 1.0
pbp.loc[pbp["fg_made"].eq(1) & pbp["kick_distance"].lt(50), "fd_points"] += 3.0
pbp.loc[pbp["fg_made"].eq(1) & pbp["kick_distance"].ge(50), "fd_points"] += 5.0

game = pbp[pbp["kicker_player_id"].ne("")].groupby(
    ["season","week","game_id","posteam","kicker_player_id"], as_index=False
).agg(
    fd_points=("fd_points","sum"),
    fg_att=("fg","sum"),
    fg_made=("fg_made","sum"),
    xp_att=("xp","sum"),
    xp_made=("xp_made","sum"),
).sort_values(["season","week","game_id"], kind="mergesort")

league_mean = float(game["fd_points"].mean())

print("MODEL_FORMULA:")
print("  personal_recent = 0.60 * last8_mean + 0.40 * last16_mean")
print("  prior = team_last16_mean when available, otherwise league_mean")
print("  projection = (personal_games * personal_recent + 4 * prior) / (personal_games + 4)")
print("  personal_games = count of games available in last16")
print("  prior_strength_games = 4")
print("  team aliases: LA->LAR, WAS->WSH")
print(f"LEAGUE_MEAN={league_mean:.6f}")

print("\n=== KICKER MODEL SIMULATION ===")
rows = []

for _, r in kickers.sort_values(["team","player"], kind="mergesort").iterrows():
    pid = clean(r["player_id"])
    name = clean(r["player"])
    team = canon_team(r["team"])
    hist = game[game["kicker_player_id"].eq(pid)].copy()

    if hist.empty:
        print(f"K_MODEL_BLOCKED|player={name!r}|team={team}|reason=NO_PERSONAL_HISTORY")
        continue

    last16 = hist.tail(16)
    last8 = hist.tail(8)

    n16 = len(last16)
    if n16 == 0 or len(last8) == 0:
        print(f"K_MODEL_BLOCKED|player={name!r}|team={team}|reason=INSUFFICIENT_PERSONAL_HISTORY")
        continue

    p8 = float(last8["fd_points"].mean())
    p16 = float(last16["fd_points"].mean())
    personal_recent = 0.60 * p8 + 0.40 * p16

    team_hist = game[game["posteam"].eq(team)].tail(16)
    if len(team_hist):
        prior = float(team_hist["fd_points"].mean())
        prior_source = "TEAM_LAST16"
        prior_games = len(team_hist)
    else:
        prior = league_mean
        prior_source = "LEAGUE"
        prior_games = 0

    prior_strength = 4.0
    projection = (n16 * personal_recent + prior_strength * prior) / (n16 + prior_strength)

    rows.append({
        "player": name,
        "team": team,
        "player_id": pid,
        "personal_games": n16,
        "last8_mean": p8,
        "last16_mean": p16,
        "personal_recent": personal_recent,
        "prior_source": prior_source,
        "prior_mean": prior,
        "prior_games": prior_games,
        "projection_candidate": projection,
    })

    print(
        f"K_MODEL|player={name!r}|team={team}|player_id={pid}|"
        f"games={n16}|last8={p8:.6f}|last16={p16:.6f}|"
        f"personal_recent={personal_recent:.6f}|prior_source={prior_source}|"
        f"prior={prior:.6f}|projection_candidate={projection:.6f}"
    )

print("\n=== SUMMARY ===")
print(f"CURRENT_KICKERS={len(kickers)}")
print(f"MODEL_READY_KICKERS={len(rows)}")
print(f"MODEL_BLOCKED_KICKERS={len(kickers)-len(rows)}")

if len(rows) != len(kickers):
    print("NFL_POSTGAME_1F_I_W_STATUS=FAIL_CLOSED")
    raise SystemExit(2)

print("MODEL_STATUS=CANDIDATE_ONLY_NOT_FROZEN")
print("KICKER_PROJECTION_WRITTEN=FALSE")
print("LOCAL_FILE_WRITES=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("FUZZY_MATCHING_USED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_W_STATUS=PASS")
