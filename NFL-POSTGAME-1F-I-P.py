#!/usr/bin/env python3
from pathlib import Path
import sqlite3
import hashlib
import json
import math
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
COLD = ROOT / "data/parquet/nfl_cold_start_projection.parquet"
DB = ROOT / "data/nfl.db"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
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

def qident(s):
    return '"' + str(s).replace('"','""') + '"'

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-P")
print("SHOWDOWN KICKER INPUT + OFFENSIVE COLD-START AUTHORITY AUDIT — READ ONLY")
print("=" * 118)

for p in (IDENTITY, COLD, DB):
    if not p.exists():
        print(f"FAIL_CLOSED_MISSING={p}")
        raise SystemExit(2)

sg = pd.read_parquet(IDENTITY)
cold = pd.read_parquet(COLD)

print(f"IDENTITY_POOL_SHA256={sha256(IDENTITY)}")
print(f"COLD_START_SHA256={sha256(COLD)}")
print(f"IDENTITY_ROWS={len(sg)}")
print(f"COLD_START_ROWS={len(cold)}")

missing = sg[sg["ridge_projection"].isna()].copy()
off = missing[~missing["position"].astype(str).str.upper().isin(["DST","D/ST","K"])].copy()
kick = missing[missing["position"].astype(str).str.upper().eq("K")].copy()

print(f"RIDGE_MISSING_ROWS={len(missing)}")
print(f"OFFENSIVE_GAP_ROWS={len(off)}")
print(f"KICKER_GAP_ROWS={len(kick)}")

print("\n=== COLD START SCHEMA ===")
print(json.dumps(list(cold.columns)))

# Discover likely exact identity/projection columns.
cold_id_cols = [c for c in cold.columns if c.lower() in {"player_id","gsis_id"}]
cold_name_cols = [c for c in cold.columns if c.lower() in {"player","player_name","player_display_name","full_name","name"}]
cold_team_cols = [c for c in cold.columns if c.lower() in {"team","team_abbr","club"}]
cold_pos_cols = [c for c in cold.columns if c.lower() in {"position","pos"}]
cold_proj_cols = [c for c in cold.columns if "projection" in c.lower() or "projected" in c.lower()]

print(f"COLD_ID_COLS={cold_id_cols}")
print(f"COLD_NAME_COLS={cold_name_cols}")
print(f"COLD_TEAM_COLS={cold_team_cols}")
print(f"COLD_POSITION_COLS={cold_pos_cols}")
print(f"COLD_PROJECTION_COLS={cold_proj_cols}")

print("\n=== OFFENSIVE GAP EXACT COLD-START MATCH ===")
cold_exact_ready = 0
cold_exact_any = 0

for _, r in off.sort_values(["public_slate_name","team","position","player"], kind="mergesort").iterrows():
    pid = clean(r.get("player_id"))
    player = clean(r.get("player"))
    team = clean(r.get("team"))

    hits = cold.iloc[0:0]
    method = "NONE"

    if pid and cold_id_cols:
        masks = []
        for c in cold_id_cols:
            masks.append(cold[c].map(clean).eq(pid))
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        hits = cold[mask]
        method = "EXACT_PLAYER_ID"

    if len(hits) == 0 and cold_name_cols:
        masks = []
        for c in cold_name_cols:
            masks.append(cold[c].map(clean).eq(player))
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        name_hits = cold[mask]
        if cold_team_cols and len(name_hits):
            tm = pd.Series(False, index=name_hits.index)
            for c in cold_team_cols:
                tm = tm | name_hits[c].map(clean).eq(team)
            name_hits = name_hits[tm]
        hits = name_hits
        method = "EXACT_NAME_TEAM"

    ready_vals = []
    for c in cold_proj_cols:
        for v in hits[c].tolist() if c in hits else []:
            if finite(v):
                ready_vals.append((c, float(v)))

    if len(hits):
        cold_exact_any += 1
    if ready_vals:
        cold_exact_ready += 1

    print(
        f"OFF_GAP|player={player!r}|team={team}|position={clean(r.get('position'))}|"
        f"player_id={pid!r}|match_method={method}|match_rows={len(hits)}|"
        f"projection_values={ready_vals}"
    )

print(f"COLD_EXACT_MATCHED_PLAYERS={cold_exact_any}")
print(f"COLD_EXACT_PROJECTION_READY_PLAYERS={cold_exact_ready}")

print("\n=== NFL.DB KICKING INPUT DISCOVERY ===")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
tables = [x["name"] for x in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)]

kick_tokens = (
    "field_goal","field_goals","fg_","kicking","extra_point","extra_points",
    "xp_","kicker","kick_distance"
)
kick_tables = []

for t in tables:
    cols = [x["name"] for x in con.execute(f"PRAGMA table_info({qident(t)})")]
    blob = " ".join([t] + cols).lower()
    if any(tok in blob for tok in kick_tokens):
        kick_tables.append((t, cols))
        print(f"KICK_INPUT_TABLE|{t}|COLUMNS={json.dumps(cols)}")

print(f"KICK_INPUT_TABLE_COUNT={len(kick_tables)}")

print("\n=== EXACT KICKER HISTORY COVERAGE ===")
name_candidates = {"player_name","player_display_name","player","name","full_name"}
id_candidates = {"player_id","gsis_id"}
season_candidates = {"season"}
week_candidates = {"week"}
team_candidates = {"team","team_abbr"}

for _, r in kick.sort_values(["public_slate_name","team","player"], kind="mergesort").iterrows():
    pid = clean(r.get("player_id"))
    player = clean(r.get("player"))
    print(
        f"\nKICKER|player={player!r}|team={clean(r.get('team'))}|"
        f"player_id={pid!r}|salary={r.get('salary')}"
    )
    total_hits = 0

    for t, cols in kick_tables:
        lower = {c.lower(): c for c in cols}
        predicates, params = [], []

        for key in id_candidates:
            if key in lower and pid and not pid.startswith(("FDKEY:","DST:")):
                predicates.append(f"{qident(lower[key])} = ?")
                params.append(pid)

        for key in name_candidates:
            if key in lower:
                predicates.append(f"{qident(lower[key])} = ?")
                params.append(player)

        if not predicates:
            continue

        wanted = []
        for c in cols:
            lc = c.lower()
            if (
                lc in id_candidates or lc in name_candidates or lc in season_candidates
                or lc in week_candidates or lc in team_candidates
                or any(tok in lc for tok in kick_tokens)
            ):
                wanted.append(c)

        if not wanted:
            continue

        sql = (
            f"SELECT {', '.join(qident(c) for c in wanted[:30])} "
            f"FROM {qident(t)} WHERE " + " OR ".join(predicates) + " LIMIT 25"
        )
        try:
            rows = con.execute(sql, params).fetchall()
        except sqlite3.Error:
            continue

        if rows:
            total_hits += len(rows)
            for hit in rows[:5]:
                payload = {k: hit[k] for k in hit.keys()}
                print(f"KICK_HISTORY_HIT|table={t}|{json.dumps(payload, default=str, sort_keys=True)}")

    print(f"KICKER_HISTORY_HITS={total_hits}")

con.close()

print("\n=== CLASSIFICATION ===")
print("COLD_START_MAY_BE_USED_ONLY_IF_EXACT_MATCH_AND_PROJECTION_COLUMN_IS_PROVEN")
print("KICKER_HISTORY_IS_INPUT_EVIDENCE_ONLY_NOT_A_PREGAME_PROJECTION")
print("NO_KICKER_PROJECTION_CREATED=TRUE")
print("PROJECTION_IMPUTATION_USED=FALSE")
print("FUZZY_MATCHING_USED=FALSE")
print("WRITE_OPERATIONS=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("PROJECTION_VALUES_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_P_STATUS=PASS")
