#!/usr/bin/env python3
from pathlib import Path
import json
import math

import nflreadpy as nfl
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
BRIDGE = ROOT / "data/fanduel/single_game/derived/single_game_projection_bridge_r1.parquet"
OUT_DIR = ROOT / "data/fanduel/single_game/derived"
OUT_PARQUET = OUT_DIR / "single_game_projection_pool.parquet"
OUT_REPORT = OUT_DIR / "single_game_projection_pool_report.json"
SEASONS = [2023, 2024, 2025]

TEAM_ALIASES = {"LA": "LAR", "WAS": "WSH"}
REQ = [
    "game_id", "season", "week", "posteam",
    "kicker_player_id", "kicker_player_name",
    "field_goal_attempt", "field_goal_result",
    "extra_point_attempt", "extra_point_result", "kick_distance",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

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

def finite(v):
    try:
        return pd.notna(v) and math.isfinite(float(v))
    except Exception:
        return False

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-X")
print("SHOWDOWN 96/96 PROJECTION POOL — ISOLATED KICKER MODEL PROMOTION")
print("=" * 118)

if not BRIDGE.exists():
    print(f"FAIL_CLOSED_MISSING={BRIDGE}")
    raise SystemExit(2)

df = pd.read_parquet(BRIDGE).copy()
print(f"INPUT_BRIDGE_SHA256={sha256(BRIDGE)}")
print(f"INPUT_ROWS={len(df)}")

k_mask = df["position"].astype(str).str.upper().eq("K")
if int(k_mask.sum()) != 6:
    print(f"FAIL_CLOSED_KICKER_COUNT={int(k_mask.sum())}")
    raise SystemExit(2)

# Require the exact R1 fail-closed kicker state before promotion.
bad_k_state = df.loc[k_mask, "projection_status"].astype(str).ne(
    "BLOCKED_NO_KICKER_PROJECTION_AUTHORITY"
)
if bad_k_state.any():
    print("FAIL_CLOSED_UNEXPECTED_KICKER_INPUT_STATE=TRUE")
    raise SystemExit(2)

frames = []
for season in SEASONS:
    pl = nfl.load_pbp(season)
    missing = [c for c in REQ if c not in pl.columns]
    if missing:
        print(f"FAIL_CLOSED_PBP_SCHEMA|season={season}|missing={json.dumps(missing)}")
        raise SystemExit(2)
    p = pl.select(REQ).to_pandas()
    p = p[flag(p["field_goal_attempt"]) | flag(p["extra_point_attempt"])].copy()
    frames.append(p)

pbp = pd.concat(frames, ignore_index=True)
pbp["kicker_player_id"] = pbp["kicker_player_id"].map(clean)
pbp["posteam"] = pbp["posteam"].map(canon_team)
pbp["field_goal_result"] = pbp["field_goal_result"].map(lambda x: clean(x).lower())
pbp["extra_point_result"] = pbp["extra_point_result"].map(lambda x: clean(x).lower())
pbp["kick_distance"] = pd.to_numeric(pbp["kick_distance"], errors="coerce")
pbp["fg"] = flag(pbp["field_goal_attempt"]).astype(int)
pbp["xp"] = flag(pbp["extra_point_attempt"]).astype(int)
pbp["fg_made"] = (pbp["fg"].eq(1) & pbp["field_goal_result"].eq("made")).astype(int)
pbp["xp_made"] = (
    pbp["xp"].eq(1) & pbp["extra_point_result"].isin(["good","made"])
).astype(int)

# Frozen candidate scoring contract from I-V/I-W.
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
promoted = 0

print("\n=== KICKER PROMOTION ===")
for idx, r in df[k_mask].sort_values(["team","player"], kind="mergesort").iterrows():
    pid = clean(r["player_id"])
    name = clean(r["player"])
    team = canon_team(r["team"])
    hist = game[game["kicker_player_id"].eq(pid)].copy()

    if hist.empty:
        print(f"FAIL_CLOSED_KICKER|player={name!r}|reason=NO_PERSONAL_HISTORY")
        raise SystemExit(2)

    last16 = hist.tail(16)
    last8 = hist.tail(8)
    if len(last16) == 0 or len(last8) == 0:
        print(f"FAIL_CLOSED_KICKER|player={name!r}|reason=INSUFFICIENT_HISTORY")
        raise SystemExit(2)

    p8 = float(last8["fd_points"].mean())
    p16 = float(last16["fd_points"].mean())
    personal_recent = 0.60 * p8 + 0.40 * p16

    team_hist = game[game["posteam"].eq(team)].tail(16)
    if len(team_hist):
        prior = float(team_hist["fd_points"].mean())
        prior_source = "TEAM_LAST16"
    else:
        prior = league_mean
        prior_source = "LEAGUE"

    n16 = len(last16)
    projection = (n16 * personal_recent + 4.0 * prior) / (n16 + 4.0)

    df.at[idx, "projection"] = projection
    df.at[idx, "projection_source"] = "WFS_SHOWDOWN_KICKER_V1"
    df.at[idx, "projection_authority"] = "WFS_SHOWDOWN_MODEL"
    df.at[idx, "projection_status"] = "READY"
    df.at[idx, "mvp_salary"] = float(r["salary"]) * 1.5
    df.at[idx, "mvp_projection"] = projection * 1.5
    promoted += 1

    print(
        f"K_PROMOTED|player={name!r}|team={team}|player_id={pid}|"
        f"last8={p8:.6f}|last16={p16:.6f}|prior_source={prior_source}|"
        f"prior={prior:.6f}|projection={projection:.6f}|mvp_projection={projection*1.5:.6f}"
    )

ready = df["projection_status"].isin(["READY","READY_EXTERNAL"])
blocked = ~ready
bad_projection = ready & ~df["projection"].map(finite)
bad_salary = ready & ~pd.to_numeric(df["salary"], errors="coerce").gt(0)
dup = df.duplicated(["public_slate_name","game","player_id"], keep=False)

if promoted != 6 or blocked.any() or bad_projection.any() or bad_salary.any() or dup.any():
    print(f"PROMOTED_KICKERS={promoted}")
    print(f"BLOCKED_ROWS={int(blocked.sum())}")
    print(f"BAD_READY_PROJECTIONS={int(bad_projection.sum())}")
    print(f"BAD_READY_SALARIES={int(bad_salary.sum())}")
    print(f"DUPLICATE_ROWS={int(dup.sum())}")
    print("NFL_POSTGAME_1F_I_X_STATUS=FAIL_CLOSED")
    raise SystemExit(2)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)

report = {
    "input_bridge_sha256": sha256(BRIDGE),
    "rows": int(len(df)),
    "ready_rows": int(ready.sum()),
    "blocked_rows": int(blocked.sum()),
    "promoted_kickers": int(promoted),
    "sources": {
        str(k): int(v)
        for k, v in df["projection_source"].value_counts(dropna=False).items()
    },
    "kicker_model": {
        "name": "WFS_SHOWDOWN_KICKER_V1",
        "historical_seasons": SEASONS,
        "personal_recent": "0.60*last8_mean + 0.40*last16_mean",
        "prior": "team_last16_mean_else_league_mean",
        "prior_strength_games": 4,
        "formula": "(personal_games*personal_recent + 4*prior)/(personal_games+4)",
    },
    "fuzzy_matching_used": False,
    "classic_production_changed": False,
    "public_solver_changed": False,
}
OUT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

print("\n=== FINAL 96/96 AUDIT ===")
print(f"TOTAL_ROWS={len(df)}")
print(f"READY_ROWS={int(ready.sum())}")
print(f"BLOCKED_ROWS={int(blocked.sum())}")
print(f"KICKER_MODEL_ROWS={(df['projection_source'] == 'WFS_SHOWDOWN_KICKER_V1').sum()}")
print(f"WFS_RIDGE_ROWS={(df['projection_source'] == 'WFS_RIDGE').sum()}")
print(f"WFS_DST_ROWS={(df['projection_source'] == 'WFS_DST_PRODUCTION').sum()}")
print(f"EXTERNAL_FANDUEL_ROWS={(df['projection_source'] == 'FANDUEL_RAW_FANTASY').sum()}")
print(f"DUPLICATE_ROWS={int(dup.sum())}")
print(f"BAD_READY_PROJECTIONS={int(bad_projection.sum())}")
print(f"BAD_READY_SALARIES={int(bad_salary.sum())}")

print(f"\nOUT_PARQUET={OUT_PARQUET}")
print(f"OUT_PARQUET_SHA256={sha256(OUT_PARQUET)}")
print(f"OUT_REPORT={OUT_REPORT}")
print(f"OUT_REPORT_SHA256={sha256(OUT_REPORT)}")
print("FUZZY_MATCHING_USED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_X_STATUS=PASS")
