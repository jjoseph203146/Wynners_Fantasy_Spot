#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import math
import sqlite3

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
DB = ROOT / "data/nfl.db"
OUT_DIR = ROOT / "data/fanduel/single_game/derived"
OUT_PARQUET = OUT_DIR / "single_game_projection_bridge.parquet"
OUT_REPORT = OUT_DIR / "single_game_projection_bridge_report.json"

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

def finite(v):
    try:
        return pd.notna(v) and math.isfinite(float(v))
    except Exception:
        return False

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-R")
print("ISOLATED SHOWDOWN PROJECTION BRIDGE — 90/96 READY, KICKERS FAIL CLOSED")
print("=" * 118)

if not IDENTITY.exists():
    print(f"FAIL_CLOSED_MISSING={IDENTITY}")
    raise SystemExit(2)
if not DB.exists():
    print(f"FAIL_CLOSED_MISSING={DB}")
    raise SystemExit(2)

df = pd.read_parquet(IDENTITY).copy()

required = {
    "public_slate_name","game","player","team","position","player_id","salary",
    "ridge_projection","raw_fantasy"
}
missing = sorted(required - set(df.columns))
if missing:
    print("FAIL_CLOSED_SCHEMA_MISSING=" + ",".join(missing))
    raise SystemExit(2)

df["projection"] = pd.NA
df["projection_source"] = ""
df["projection_authority"] = ""
df["projection_status"] = "BLOCKED"

# 1) WFS ridge where present
mask = df["ridge_projection"].map(finite)
df.loc[mask, "projection"] = pd.to_numeric(df.loc[mask, "ridge_projection"], errors="coerce")
df.loc[mask, "projection_source"] = "WFS_RIDGE"
df.loc[mask, "projection_authority"] = "WFS_MODEL"
df.loc[mask, "projection_status"] = "READY"

# 2) D/ST from exact game/team match in dst_production_projection
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
dst = pd.read_sql_query(
    """
    SELECT game_id, team, dst_projection, model_status
    FROM dst_production_projection
    """,
    con,
)
con.close()

dst["team"] = dst["team"].map(clean)
dst["game_id"] = dst["game_id"].map(clean)

# identity pool game is display matchup, so exact mapping via current rows in fanduel_slate_projection_pool
# use player/team exact against already-proven WFS D/ST attachment instead of guessing game_id format.
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
fd_dst = pd.read_sql_query(
    """
    SELECT player, team, gameInfo, internal_projection, model_projection, projection_status
    FROM fanduel_slate_projection_pool
    WHERE is_dst = 1
    """,
    con,
)
con.close()

def norm_team_name(s):
    return clean(s)

for idx, r in df[df["position"].astype(str).str.upper().isin(["DST","D/ST"])].iterrows():
    hits = fd_dst[
        fd_dst["player"].map(clean).eq(clean(r["player"]))
        & fd_dst["gameInfo"].map(clean).eq(clean(r["game"]))
    ]
    vals = sorted(set(
        float(v) for v in hits["internal_projection"].tolist() if finite(v)
    ))
    if len(vals) == 1:
        df.at[idx, "projection"] = vals[0]
        df.at[idx, "projection_source"] = "WFS_DST_PRODUCTION"
        df.at[idx, "projection_authority"] = "WFS_MODEL"
        df.at[idx, "projection_status"] = "READY"
    elif len(vals) > 1:
        df.at[idx, "projection_status"] = "BLOCKED_DST_AMBIGUOUS"
    else:
        df.at[idx, "projection_status"] = "BLOCKED_DST_NO_EXACT_MATCH"

# 3) Offensive ridge gaps: exact FanDuel raw fantasy fallback, explicitly external/non-WFS.
# Only applies to non-K, non-DST rows that are still blocked and have finite raw fantasy.
for idx, r in df.iterrows():
    pos = clean(r["position"]).upper()
    if df.at[idx, "projection_status"] == "READY":
        continue
    if pos in {"K","DST","D/ST"}:
        continue
    if finite(r["raw_fantasy"]):
        df.at[idx, "projection"] = float(r["raw_fantasy"])
        df.at[idx, "projection_source"] = "FANDUEL_RAW_FANTASY"
        df.at[idx, "projection_authority"] = "EXTERNAL_FANDUEL_SOURCE"
        df.at[idx, "projection_status"] = "READY_EXTERNAL"

# 4) Kickers remain blocked: no proven projection authority.
k_mask = df["position"].astype(str).str.upper().eq("K")
df.loc[k_mask, "projection"] = pd.NA
df.loc[k_mask, "projection_source"] = ""
df.loc[k_mask, "projection_authority"] = ""
df.loc[k_mask, "projection_status"] = "BLOCKED_NO_KICKER_PROJECTION_AUTHORITY"

# Derived MVP fields only for ready rows.
ready = df["projection_status"].isin(["READY","READY_EXTERNAL"])
df["mvp_salary"] = pd.NA
df["mvp_projection"] = pd.NA
df.loc[ready, "mvp_salary"] = pd.to_numeric(df.loc[ready, "salary"], errors="coerce") * 1.5
df.loc[ready, "mvp_projection"] = pd.to_numeric(df.loc[ready, "projection"], errors="coerce") * 1.5

# Fail closed on duplicates or invalid ready values.
dup = df.duplicated(["public_slate_name","game","player_id"], keep=False)
bad_ready = ready & ~df["projection"].map(finite)
bad_salary = ready & ~pd.to_numeric(df["salary"], errors="coerce").gt(0)

if dup.any() or bad_ready.any() or bad_salary.any():
    print(f"DUPLICATE_ROWS={int(dup.sum())}")
    print(f"BAD_READY_PROJECTIONS={int(bad_ready.sum())}")
    print(f"BAD_READY_SALARIES={int(bad_salary.sum())}")
    print("NFL_POSTGAME_1F_I_R_STATUS=FAIL_CLOSED")
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
    "by_source": df["projection_source"].value_counts(dropna=False).to_dict(),
    "fuzzy_matching_used": False,
    "classic_production_changed": False,
    "public_solver_changed": False,
}
OUT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

print(f"TOTAL_ROWS={len(df)}")
print(f"READY_WFS_ROWS={(df['projection_status'] == 'READY').sum()}")
print(f"READY_EXTERNAL_ROWS={(df['projection_status'] == 'READY_EXTERNAL').sum()}")
print(f"BLOCKED_ROWS={(~df['projection_status'].isin(['READY','READY_EXTERNAL'])).sum()}")
print(f"BLOCKED_KICKER_ROWS={(df['projection_status'] == 'BLOCKED_NO_KICKER_PROJECTION_AUTHORITY').sum()}")

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
print("NFL_POSTGAME_1F_I_R_STATUS=PASS")
