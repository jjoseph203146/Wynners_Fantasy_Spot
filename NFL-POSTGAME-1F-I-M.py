#!/usr/bin/env python3
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DERIVED = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"
FULL_POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
NFL_DB = ROOT / "data/nfl.db"

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
print("WFS NFL — NFL-POSTGAME-1F-I-M")
print("SINGLE-GAME PROJECTION AUTHORITY AUDIT — READ ONLY")
print("=" * 118)

for p in (DERIVED, FULL_POOL, NFL_DB):
    if not p.exists():
        print(f"FAIL_CLOSED_MISSING={p}")
        raise SystemExit(2)

sg = pd.read_parquet(DERIVED)
pool = pd.read_parquet(FULL_POOL)

missing = sg[sg["ridge_projection"].isna()].copy()
print(f"SINGLE_GAME_ROWS={len(sg)}")
print(f"RIDGE_READY_ROWS={int(sg['ridge_projection'].notna().sum())}")
print(f"RIDGE_MISSING_ROWS={len(missing)}")

print("\n=== MISSING BY POSITION ===")
for k, v in missing["position"].value_counts(dropna=False).sort_index().items():
    print(f"POSITION|{k}|COUNT={int(v)}")

print("\n=== MISSING BY IDENTITY SOURCE ===")
for k, v in missing["identity_source"].value_counts(dropna=False).sort_index().items():
    print(f"IDENTITY_SOURCE|{k}|COUNT={int(v)}")

raw_ready = int(missing["raw_fantasy"].notna().sum())
raw_missing = int(missing["raw_fantasy"].isna().sum())
print(f"\nMISSING_RIDGE_WITH_RAW_FANTASY={raw_ready}")
print(f"MISSING_RIDGE_WITHOUT_RAW_FANTASY={raw_missing}")

# Read-only database discovery: inspect exact player_id/name rows in likely projection-bearing tables.
con = sqlite3.connect(f"file:{NFL_DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

tables = [r["name"] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)]

projection_terms = ("projection", "projected", "fantasy", "ridge", "predicted")
id_terms = {"player_id", "gsis_id"}
name_terms = {"player_display_name", "full_name", "player_name", "name"}

candidate_tables = []
for t in tables:
    cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{t}")')]
    lower = {c.lower(): c for c in cols}
    proj_cols = [c for c in cols if any(term in c.lower() for term in projection_terms)]
    id_cols = [lower[x] for x in id_terms if x in lower]
    nm_cols = [lower[x] for x in name_terms if x in lower]
    if proj_cols and (id_cols or nm_cols):
        candidate_tables.append((t, cols, proj_cols, id_cols, nm_cols))

print(f"\nDB_PROJECTION_CANDIDATE_TABLES={len(candidate_tables)}")
for t, _, proj_cols, id_cols, nm_cols in candidate_tables:
    print(
        f"DB_PROJECTION_TABLE|{t}|"
        f"projection_cols={','.join(proj_cols)}|"
        f"id_cols={','.join(id_cols)}|"
        f"name_cols={','.join(nm_cols)}"
    )

print("\n=== 22 MISSING RIDGE ROWS ===")
db_exact_projection_hits = 0

for _, r in missing.sort_values(
    ["public_slate_name","team","position","player"], kind="mergesort"
).iterrows():
    player = clean(r["player"])
    pid = clean(r["player_id"])
    print(
        f"\nROW|slate={clean(r['public_slate_name'])!r}|game={clean(r['game'])}|"
        f"player={player!r}|team={clean(r['team'])}|position={clean(r['position'])}|"
        f"player_id={pid!r}|identity_source={clean(r['identity_source'])}|"
        f"salary={r['salary']}|raw_fantasy={r['raw_fantasy']!r}"
    )

    # Exact matches in full pool by player_id where possible; otherwise exact player name.
    if pid and not pid.startswith(("DST:", "FDKEY:")) and "player_id" in pool.columns:
        fp = pool[pool["player_id"].map(clean) == pid]
    else:
        fp = pool[pool["fd_name"].map(clean) == player] if "fd_name" in pool.columns else pool.iloc[0:0]

    print(f"FULL_POOL_RELATED_ROWS={len(fp)}")
    if len(fp):
        ridge_vals = sorted({float(v) for v in fp["ridge_projection"] if finite(v)})
        src_vals = sorted({float(v) for v in fp["source_fantasy_projection"] if finite(v)})
        print(f"FULL_POOL_RIDGE_VALUES={ridge_vals}")
        print(f"FULL_POOL_SOURCE_FANTASY_VALUES={src_vals}")

    hit_count = 0
    for t, cols, proj_cols, id_cols, nm_cols in candidate_tables:
        predicates = []
        params = []
        if pid and not pid.startswith(("DST:", "FDKEY:")):
            for c in id_cols:
                predicates.append(f'"{c}" = ?')
                params.append(pid)
        for c in nm_cols:
            predicates.append(f'"{c}" = ?')
            params.append(player)

        if not predicates:
            continue

        select_cols = []
        for c in id_cols + nm_cols + proj_cols:
            if c not in select_cols:
                select_cols.append(c)

        sql = (
            f'SELECT {", ".join([f"""\"{c}\"""" for c in select_cols])} '
            f'FROM "{t}" WHERE ' + " OR ".join(predicates) + " LIMIT 20"
        )
        try:
            hits = con.execute(sql, params).fetchall()
        except sqlite3.Error:
            continue

        useful = []
        for hit in hits:
            vals = {c: hit[c] for c in proj_cols}
            if any(finite(v) for v in vals.values()):
                useful.append(hit)

        if useful:
            hit_count += len(useful)
            db_exact_projection_hits += len(useful)
            for hit in useful[:5]:
                payload = "|".join(f"{k}={clean(hit[k])!r}" for k in hit.keys())
                print(f"DB_PROJECTION_HIT|table={t}|{payload}")

    print(f"DB_EXACT_PROJECTION_HITS={hit_count}")

con.close()

print("\n=== AUTHORITY CLASSIFICATION ===")
print("RIDGE_PROJECTION=WFS_MODEL_AUTHORITY_WHERE_PRESENT")
print("RAW_FANTASY=FANDUEL_SOURCE_FIELD_UNPROVEN_AS_WFS_MODEL_AUTHORITY")
print("SOURCE_FANTASY_PROJECTION=NOT_WFS_MODEL_AUTHORITY")
print("PROJECTION_IMPUTATION_USED=FALSE")
print("FUZZY_MATCHING_USED=FALSE")
print("WRITE_OPERATIONS=0")
print(f"DB_EXACT_PROJECTION_HITS_TOTAL={db_exact_projection_hits}")
print("NFL_POSTGAME_1F_I_M_STATUS=PASS")
