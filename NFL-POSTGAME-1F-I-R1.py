#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import math
import re
import sqlite3

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
DB = ROOT / "data/nfl.db"
OUT_DIR = ROOT / "data/fanduel/single_game/derived"
OUT_PARQUET = OUT_DIR / "single_game_projection_bridge_r1.parquet"
OUT_REPORT = OUT_DIR / "single_game_projection_bridge_r1_report.json"

TEAM_ALIASES = {"LA": "LAR", "WAS": "WSH", "JAC": "JAX"}  # STAGE18N_R6B

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

def finite(v):
    try:
        return pd.notna(v) and math.isfinite(float(v))
    except Exception:
        return False

def parse_matchup(label):
    m = re.fullmatch(r"\s*([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\s*", clean(label).upper())
    if not m:
        return None
    return canon_team(m.group(1)), canon_team(m.group(2))

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-R1")
print("ISOLATED SHOWDOWN PROJECTION BRIDGE — EXACT D/ST GAME/TEAM AUTHORITY")
print("=" * 118)

for p in (IDENTITY, DB):
    if not p.exists():
        print(f"FAIL_CLOSED_MISSING={p}")
        raise SystemExit(2)

df = pd.read_parquet(IDENTITY).copy()

required = {
    "public_slate_name","game","player","team","position","player_id","salary",
    "ridge_projection","raw_fantasy","full_pool_source_fantasy_projection"
}
missing = sorted(required - set(df.columns))
if missing:
    print("FAIL_CLOSED_SCHEMA_MISSING=" + ",".join(missing))
    raise SystemExit(2)

df["projection"] = pd.NA
df["projection_source"] = ""
df["projection_authority"] = ""
df["projection_status"] = "BLOCKED"

# 1) WFS ridge
ridge = df["ridge_projection"].map(finite)
df.loc[ridge, "projection"] = pd.to_numeric(df.loc[ridge, "ridge_projection"], errors="coerce")
df.loc[ridge, "projection_source"] = "WFS_RIDGE"
df.loc[ridge, "projection_authority"] = "WFS_MODEL"
df.loc[ridge, "projection_status"] = "READY"

# 2) Read exact schedule + D/ST model authority read-only.
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
games = pd.read_sql_query(
    """
    SELECT game_id, season, week, away_team, home_team, completed
    FROM games
    """,
    con,
)
dst = pd.read_sql_query(
    """
    SELECT game_id, season, week, team, opponent_team, dst_projection, model_status
    FROM dst_production_projection
    """,
    con,
)
# Retained WFS pregame D/ST model authority for games that have
# aged off the current production surface.
dst_retained = pd.read_sql_query(
    """
    SELECT game_id, season, week, team, opponent_team,
           dst_projection, model_status
    FROM fanduel_dst_pool
    """,
    con,
)

con.close()

for c in ["away_team","home_team"]:
    games[c] = games[c].map(canon_team)
for c in ["team","opponent_team"]:
    dst[c] = dst[c].map(canon_team)
    dst_retained[c] = dst_retained[c].map(canon_team)
games["game_id"] = games["game_id"].map(clean)
dst["game_id"] = dst["game_id"].map(clean)
dst_retained["game_id"] = dst_retained["game_id"].map(clean)

print("\n=== D/ST EXACT RESOLUTION ===")
dst_ready = 0
dst_blocked = 0

for idx, r in df[df["position"].astype(str).str.upper().isin(["DST","D/ST"])].iterrows():
    slate = clean(r["public_slate_name"])
    parsed = parse_matchup(slate)
    team = canon_team(r["team"])

    if not parsed:
        df.at[idx, "projection_status"] = "BLOCKED_DST_BAD_MATCHUP_LABEL"
        dst_blocked += 1
        print(f"DST_BLOCKED|player={clean(r['player'])!r}|reason=BAD_MATCHUP_LABEL|slate={slate!r}")
        continue

    away, home = parsed
    if team not in {away, home}:
        df.at[idx, "projection_status"] = "BLOCKED_DST_TEAM_NOT_IN_MATCHUP"
        dst_blocked += 1
        print(f"DST_BLOCKED|player={clean(r['player'])!r}|reason=TEAM_NOT_IN_MATCHUP|team={team}|slate={slate!r}")
        continue

    opp = home if team == away else away

    # Exact scheduled game identity by away/home pair. Choose current/latest season-week only
    # when exactly one maximum season/week pair exists; ties fail closed.
    gh = games[(games["away_team"] == away) & (games["home_team"] == home)].copy()
    if gh.empty:
        df.at[idx, "projection_status"] = "BLOCKED_DST_NO_SCHEDULE_GAME"
        dst_blocked += 1
        print(f"DST_BLOCKED|player={clean(r['player'])!r}|reason=NO_SCHEDULE_GAME|pair={away}@{home}")
        continue

    gh["season"] = pd.to_numeric(gh["season"], errors="coerce")
    gh["week"] = pd.to_numeric(gh["week"], errors="coerce")
    gh = gh.dropna(subset=["season","week"])
    if gh.empty:
        df.at[idx, "projection_status"] = "BLOCKED_DST_SCHEDULE_BAD_SEASON_WEEK"
        dst_blocked += 1
        continue

    max_season = gh["season"].max()
    gh2 = gh[gh["season"] == max_season]
    max_week = gh2["week"].max()
    gh3 = gh2[gh2["week"] == max_week]

    if len(gh3) != 1:
        df.at[idx, "projection_status"] = "BLOCKED_DST_SCHEDULE_AMBIGUOUS"
        dst_blocked += 1
        print(f"DST_BLOCKED|player={clean(r['player'])!r}|reason=SCHEDULE_AMBIGUOUS|rows={len(gh3)}|pair={away}@{home}")
        continue

    g = gh3.iloc[0]
    game_id = clean(g["game_id"])
    season = int(g["season"])
    week = int(g["week"])

    dh = dst[
        (dst["game_id"] == game_id)
        & (pd.to_numeric(dst["season"], errors="coerce") == season)
        & (pd.to_numeric(dst["week"], errors="coerce") == week)
        & (dst["team"] == team)
        & (dst["opponent_team"] == opp)
    ].copy()

    vals = sorted(set(float(v) for v in dh["dst_projection"].tolist() if finite(v)))

    if len(dh) == 1 and len(vals) == 1:
        val = vals[0]
        df.at[idx, "projection"] = val
        df.at[idx, "projection_source"] = "WFS_DST_PRODUCTION"
        df.at[idx, "projection_authority"] = "WFS_MODEL"
        df.at[idx, "projection_status"] = "READY"
        dst_ready += 1
        print(
            f"DST_READY|player={clean(r['player'])!r}|team={team}|opponent={opp}|"
            f"game_id={game_id}|season={season}|week={week}|projection={val}|"
            f"model_status={clean(dh.iloc[0]['model_status'])}"
        )
    else:
        # Current production authority is intentionally limited to
        # the current/pregame surface. For completed games only,
        # consult retained WFS FanDuel D/ST model authority using
        # exact game/season/week/team/opponent identity.
        completed = str(g["completed"]).strip().lower() in {"1", "1.0", "true"}

        rh = dst_retained[
            (dst_retained["game_id"] == game_id)
            & (pd.to_numeric(dst_retained["season"], errors="coerce") == season)
            & (pd.to_numeric(dst_retained["week"], errors="coerce") == week)
            & (dst_retained["team"] == team)
            & (dst_retained["opponent_team"] == opp)
        ].copy()

        rvals = sorted(
            set(
                float(v)
                for v in rh["dst_projection"].tolist()
                if finite(v)
            )
        )

        retained_ok = (
            completed
            and len(dh) == 0
            and len(rh) == 1
            and len(rvals) == 1
            and clean(rh.iloc[0]["model_status"]).upper() == "PROMOTED"
        )

        if retained_ok:
            val = rvals[0]
            df.at[idx, "projection"] = val
            df.at[idx, "projection_source"] = "WFS_DST_RETAINED_FANDUEL_POOL"
            df.at[idx, "projection_authority"] = "WFS_MODEL"
            df.at[idx, "projection_status"] = "READY"
            dst_ready += 1
            print(
                f"DST_READY_RETAINED|player={clean(r['player'])!r}|team={team}|"
                f"opponent={opp}|game_id={game_id}|season={season}|week={week}|"
                f"projection={val}|model_status={clean(rh.iloc[0]['model_status'])}"
            )
            continue

        # Preserve the existing external FanDuel pregame fallback as
        # last resort only when neither WFS model authority has a row.
        preserved = r["full_pool_source_fantasy_projection"]

        if completed and len(dh) == 0 and len(rh) == 0 and finite(preserved):
            val = float(preserved)
            df.at[idx, "projection"] = val
            df.at[idx, "projection_source"] = "FANDUEL_DST_PREGAME_SOURCE"
            df.at[idx, "projection_authority"] = "EXTERNAL_FANDUEL_SOURCE"
            df.at[idx, "projection_status"] = "READY_EXTERNAL"
            dst_ready += 1
            print(
                f"DST_READY_EXTERNAL|player={clean(r['player'])!r}|team={team}|"
                f"opponent={opp}|game_id={game_id}|season={season}|week={week}|"
                f"projection={val}|source=FANDUEL_DST_PREGAME_SOURCE"
            )
        else:
            df.at[idx, "projection_status"] = "BLOCKED_DST_NO_UNIQUE_MODEL_ROW"
            dst_blocked += 1
            print(
                f"DST_BLOCKED|player={clean(r['player'])!r}|reason=NO_UNIQUE_MODEL_ROW|"
                f"game_id={game_id}|team={team}|opponent={opp}|"
                f"production_rows={len(dh)}|production_values={vals}|"
                f"retained_rows={len(rh)}|retained_values={rvals}|"
                f"completed={completed}|preserved_pregame={clean(preserved)!r}"
            )

# 3) Explicit external fallback for non-special offensive gaps only.
for idx, r in df.iterrows():
    if df.at[idx, "projection_status"] == "READY":
        continue
    pos = clean(r["position"]).upper()
    if pos in {"K","DST","D/ST"}:
        continue
    if finite(r["raw_fantasy"]):
        df.at[idx, "projection"] = float(r["raw_fantasy"])
        df.at[idx, "projection_source"] = "FANDUEL_RAW_FANTASY"
        df.at[idx, "projection_authority"] = "EXTERNAL_FANDUEL_SOURCE"
        df.at[idx, "projection_status"] = "READY_EXTERNAL"

# 4) K stays blocked until real authority exists.
kmask = df["position"].astype(str).str.upper().eq("K")
df.loc[kmask, "projection"] = pd.NA
df.loc[kmask, "projection_source"] = ""
df.loc[kmask, "projection_authority"] = ""
df.loc[kmask, "projection_status"] = "BLOCKED_NO_KICKER_PROJECTION_AUTHORITY"

ready = df["projection_status"].isin(["READY","READY_EXTERNAL"])
df["mvp_salary"] = pd.NA
df["mvp_projection"] = pd.NA
df.loc[ready, "mvp_salary"] = pd.to_numeric(df.loc[ready, "salary"], errors="coerce") * 1.5
df.loc[ready, "mvp_projection"] = pd.to_numeric(df.loc[ready, "projection"], errors="coerce") * 1.5

dup = df.duplicated(["public_slate_name","game","player_id"], keep=False)
bad_ready = ready & ~df["projection"].map(finite)
bad_salary = ready & ~pd.to_numeric(df["salary"], errors="coerce").gt(0)

if dup.any() or bad_ready.any() or bad_salary.any():
    print(f"DUPLICATE_ROWS={int(dup.sum())}")
    print(f"BAD_READY_PROJECTIONS={int(bad_ready.sum())}")
    print(f"BAD_READY_SALARIES={int(bad_salary.sum())}")
    print("NFL_POSTGAME_1F_I_R1_STATUS=FAIL_CLOSED")
    raise SystemExit(2)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)

report = {
    "identity_pool_sha256": sha256(IDENTITY),
    "rows": int(len(df)),
    "ready_wfs": int((df["projection_status"] == "READY").sum()),
    "ready_external": int((df["projection_status"] == "READY_EXTERNAL").sum()),
    "blocked": int((~df["projection_status"].isin(["READY","READY_EXTERNAL"])).sum()),
    "blocked_kickers": int((df["projection_status"] == "BLOCKED_NO_KICKER_PROJECTION_AUTHORITY").sum()),
    "dst_ready": int(dst_ready),
    "dst_blocked": int(dst_blocked),
    "by_source": {str(k): int(v) for k, v in df["projection_source"].value_counts(dropna=False).items()},
    "fuzzy_matching_used": False,
    "classic_production_changed": False,
    "public_solver_changed": False,
}
OUT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

print("\n=== SUMMARY ===")
print(f"TOTAL_ROWS={len(df)}")
print(f"READY_WFS_ROWS={(df['projection_status'] == 'READY').sum()}")
print(f"READY_EXTERNAL_ROWS={(df['projection_status'] == 'READY_EXTERNAL').sum()}")
print(f"BLOCKED_ROWS={(~df['projection_status'].isin(['READY','READY_EXTERNAL'])).sum()}")
print(f"BLOCKED_KICKER_ROWS={(df['projection_status'] == 'BLOCKED_NO_KICKER_PROJECTION_AUTHORITY').sum()}")
print(f"DST_READY_ROWS={dst_ready}")
print(f"DST_BLOCKED_ROWS={dst_blocked}")

print("\n=== SOURCE COUNTS ===")
for k, v in df["projection_source"].value_counts(dropna=False).items():
    print(f"SOURCE|{k!r}|COUNT={int(v)}")

print("\n=== BLOCKED ===")
for _, r in df[~df["projection_status"].isin(["READY","READY_EXTERNAL"])].sort_values(
    ["public_slate_name","team","player"], kind="mergesort"
).iterrows():
    print(
        f"BLOCKED|slate={clean(r['public_slate_name'])!r}|player={clean(r['player'])!r}|"
        f"team={clean(r['team'])}|position={clean(r['position'])}|status={clean(r['projection_status'])}"
    )

print(f"\nOUT_PARQUET={OUT_PARQUET}")
print(f"OUT_PARQUET_SHA256={sha256(OUT_PARQUET)}")
print(f"OUT_REPORT={OUT_REPORT}")
print(f"OUT_REPORT_SHA256={sha256(OUT_REPORT)}")
print("FUZZY_MATCHING_USED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_R1_STATUS=PASS")
