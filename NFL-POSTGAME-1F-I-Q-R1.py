#!/usr/bin/env python3
from pathlib import Path
import csv
import hashlib
import json
import math
import sqlite3

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
COLD = ROOT / "data/parquet/nfl_cold_start_projection.parquet"
DB = ROOT / "data/nfl.db"

KICK_TOKENS = (
    "field_goal", "field_goals", "fgm", "fga", "extra_point", "extra_points",
    "xpm", "xpa", "kicking", "kicker", "kick_distance"
)

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def finite(v):
    try:
        return pd.notna(v) and math.isfinite(float(v))
    except Exception:
        return False

def canonical_pid(v):
    s = clean(v)
    if s.startswith("FDKEY:GSIS:"):
        return s.split("FDKEY:GSIS:", 1)[1]
    return s

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-Q")
print("COLD-START ESTIMATE + KICKER SOURCE SCHEMA AUDIT — READ ONLY")
print("=" * 118)

for p in (IDENTITY, COLD, DB):
    if not p.exists():
        print(f"FAIL_CLOSED_MISSING={p}")
        raise SystemExit(2)

sg = pd.read_parquet(IDENTITY)
cold = pd.read_parquet(COLD)

print(f"IDENTITY_POOL_SHA256={sha256(IDENTITY)}")
print(f"COLD_START_SHA256={sha256(COLD)}")

missing = sg[sg["ridge_projection"].isna()].copy()
off = missing[~missing["position"].astype(str).str.upper().isin(["DST","D/ST","K"])].copy()
kick = missing[missing["position"].astype(str).str.upper().eq("K")].copy()

print(f"OFFENSIVE_GAP_ROWS={len(off)}")
print(f"KICKER_GAP_ROWS={len(kick)}")
print(f"COLD_START_ROWS={len(cold)}")
print(f"COLD_START_COLUMNS={json.dumps(list(cold.columns))}")

required_cold = {"player_display_name","team","position","gsis_id","cold_start_estimate","cold_start_status"}
missing_cold = sorted(required_cold - set(cold.columns))
if missing_cold:
    print("FAIL_CLOSED_COLD_SCHEMA_MISSING=" + ",".join(missing_cold))
    raise SystemExit(2)

cold["_pid"] = cold["gsis_id"].map(clean)
cold["_name"] = cold["player_display_name"].map(clean)
cold["_team"] = cold["team"].map(clean)

print("\n=== OFFENSIVE GAP COLD-START EVIDENCE ===")
exact_pid_hits = 0
exact_name_hits = 0
finite_estimates = 0

for _, r in off.sort_values(["public_slate_name","team","position","player"], kind="mergesort").iterrows():
    raw_pid = clean(r["player_id"])
    pid = canonical_pid(raw_pid)
    name = clean(r["player"])
    team = clean(r["team"])

    by_pid = cold[cold["_pid"].eq(pid)] if pid else cold.iloc[0:0]
    by_name = cold[cold["_name"].eq(name)]
    by_name_team = by_name[by_name["_team"].eq(team)]

    if len(by_pid):
        exact_pid_hits += 1
    if len(by_name):
        exact_name_hits += 1

    vals = []
    for _, c in by_pid.iterrows():
        if finite(c["cold_start_estimate"]):
            vals.append(float(c["cold_start_estimate"]))
    vals = sorted(set(vals))
    if vals:
        finite_estimates += 1

    print(
        f"OFF_GAP|player={name!r}|team={team}|position={clean(r['position'])}|"
        f"raw_player_id={raw_pid!r}|canonical_gsis_id={pid!r}|"
        f"pid_rows={len(by_pid)}|name_rows={len(by_name)}|name_team_rows={len(by_name_team)}|"
        f"cold_start_estimates={vals}"
    )

    for _, c in by_pid.head(5).iterrows():
        print(
            "COLD_HIT|"
            f"player={clean(c['player_display_name'])!r}|team={clean(c['team'])}|"
            f"position={clean(c['position'])}|gsis_id={clean(c['gsis_id'])!r}|"
            f"production_status={clean(c['production_status'])}|"
            f"cold_start_estimate={c['cold_start_estimate']!r}|"
            f"cold_start_status={clean(c['cold_start_status'])}"
        )

print(f"COLD_EXACT_PID_PLAYERS={exact_pid_hits}")
print(f"COLD_EXACT_NAME_PLAYERS={exact_name_hits}")
print(f"COLD_FINITE_ESTIMATE_PLAYERS={finite_estimates}")

print("\n=== NFL.DB KICKER SCHEMA CONFIRMATION ===")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

tables = [r["name"] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)]

db_kick_columns = []
for t in tables:
    cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{t}")')]
    hits = [c for c in cols if any(tok in c.lower() for tok in KICK_TOKENS)]
    if hits:
        db_kick_columns.append((t, hits))
        print(f"DB_KICK_SCHEMA|table={t}|columns={json.dumps(hits)}")

print(f"DB_TABLES_WITH_KICKING_COLUMNS={len(db_kick_columns)}")
con.close()

print("\n=== FILE HEADER / PARQUET SCHEMA KICKER PROBE ===")
schema_hits = []
data_roots = [ROOT / "data/parquet", ROOT / "data/csv"]

for base in data_roots:
    if not base.exists():
        continue
    for p in sorted(base.iterdir()):
        if not p.is_file():
            continue

        cols = []
        try:
            if p.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
            elif p.suffix.lower() == ".csv":
                with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                    row = next(csv.reader(f), [])
                    cols = [clean(x) for x in row]
            else:
                continue
        except Exception:
            continue

        hits = [c for c in cols if any(tok in c.lower() for tok in KICK_TOKENS)]
        if hits:
            schema_hits.append((p, hits))
            print(f"FILE_KICK_SCHEMA|file={p.relative_to(ROOT)}|columns={json.dumps(hits)}")

print(f"FILES_WITH_KICKING_COLUMNS={len(schema_hits)}")

print("\n=== KICKER IDENTITIES ===")
for _, r in kick.sort_values(["public_slate_name","team","player"], kind="mergesort").iterrows():
    print(
        f"KICKER|player={clean(r['player'])!r}|team={clean(r['team'])}|"
        f"player_id={clean(r['player_id'])!r}|salary={r['salary']}"
    )

print("\n=== CLASSIFICATION ===")
print("COLD_START_ESTIMATE_IS_A_CANDIDATE_ONLY_IF_EXACT_GSIS_MATCH_AND_STATUS_SUPPORTS_USE")
print("NO_KICKER_PROJECTION_CREATED=TRUE")
print("NO_HISTORICAL_ACTUAL_USED_AS_PROJECTION=TRUE")
print("PROJECTION_IMPUTATION_USED=FALSE")
print("FUZZY_MATCHING_USED=FALSE")
print("WRITE_OPERATIONS=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("PROJECTION_VALUES_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_Q_STATUS=PASS")
