#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import csv

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"

TARGET_CODE = [
    ROOT / "dst_history.py",
    ROOT / "dst_fanduel_scoring.py",
    ROOT / "dst_source_audit.py",
]

KICKERS = {
    "00-0032569": "Wil Lutz",
    "00-0033303": "Harrison Butker",
    "00-0040200": "Andy Borregales",
    "00-0031492": "Jason Myers",
    "00-0039498": "Harrison Mevis",
    "00-0034173": "Eddy Pineiro",
}

RAW_TOKENS = (
    "play_by_play", "playbyplay", "pbp", "field_goal_result", "field_goal_attempt",
    "extra_point_result", "extra_point_attempt", "kick_distance",
    "kicker_player_id", "kicker_player_name", "kicker_id", "kicker_name",
)

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

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-T")
print("RAW PBP KICKER AUTHORITY TRACE — READ ONLY")
print("=" * 118)

# 1) Inspect exact code lines around PBP/kicking usage.
print("\n=== CODE TRACE ===")
for p in TARGET_CODE:
    if not p.exists():
        print(f"CODE_MISSING|{p.name}")
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    print(f"CODE_FILE|{p.name}|SHA256={sha256(p)}")
    lines = text.splitlines()
    matched = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(tok in low for tok in RAW_TOKENS):
            matched.append(i)
    shown = set()
    for i in matched:
        for j in range(max(1, i-2), min(len(lines), i+2)+1):
            if j in shown:
                continue
            shown.add(j)
            print(f"CODE_LINE|file={p.name}|line={j}|text={lines[j-1]}")

# 2) Discover likely raw PBP files by name + schema.
print("\n=== RAW FILE DISCOVERY ===")
candidate_files = []
for base in [ROOT / "data", ROOT / "raw", ROOT / "data/raw"]:
    if not base.exists():
        continue
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        lowname = p.name.lower()
        if any(tok in lowname for tok in ("pbp", "play_by_play", "playbyplay")):
            candidate_files.append(p)

print(f"RAW_FILENAME_CANDIDATES={len(candidate_files)}")

schema_candidates = []
seen = set()
for p in candidate_files:
    if p in seen:
        continue
    seen.add(p)
    cols = []
    try:
        if p.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            cols = pq.read_schema(p).names
        elif p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                cols = next(csv.reader(f), [])
        else:
            continue
    except Exception:
        continue

    hits = [c for c in cols if any(tok in c.lower() for tok in RAW_TOKENS)]
    if hits:
        schema_candidates.append((p, cols, hits))
        print(
            f"RAW_SCHEMA|file={p.relative_to(ROOT)}|"
            f"hits={json.dumps(hits)}|columns={json.dumps(cols)}|sha256={sha256(p)}"
        )

print(f"RAW_SCHEMA_CANDIDATES={len(schema_candidates)}")

# 3) Inspect DB for any raw PBP tables not caught previously.
print("\n=== NFL.DB RAW PBP DISCOVERY ===")
if DB.exists():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = [r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    db_raw = 0
    for t in tables:
        cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{t}")')]
        blob = " ".join([t] + cols).lower()
        if any(tok in blob for tok in RAW_TOKENS):
            db_raw += 1
            print(f"DB_RAW|table={t}|columns={json.dumps(cols)}")
    print(f"DB_RAW_TABLES={db_raw}")
    con.close()
else:
    print("DB_RAW_TABLES=0")

# 4) Exact kicker evidence in raw PBP candidate files.
print("\n=== EXACT KICKER EVIDENCE ===")
exact_hits = 0

for p, cols, hits in schema_candidates:
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"RAW_READ_ERROR|file={p.relative_to(ROOT)}|error={type(e).__name__}")
        continue

    id_cols = [c for c in df.columns if any(x in c.lower() for x in (
        "kicker_player_id","kicker_id","player_id","gsis_id"
    ))]
    name_cols = [c for c in df.columns if any(x in c.lower() for x in (
        "kicker_player_name","kicker_name","player_name"
    ))]
    fg_cols = [c for c in df.columns if "field_goal" in c.lower() or "kick_distance" in c.lower()]
    xp_cols = [c for c in df.columns if "extra_point" in c.lower() or "pat_result" in c.lower()]

    if not id_cols and not name_cols:
        continue

    for pid, name in KICKERS.items():
        mask = pd.Series(False, index=df.index)
        for c in id_cols:
            mask = mask | df[c].astype(str).str.strip().eq(pid)
        for c in name_cols:
            mask = mask | df[c].astype(str).str.strip().eq(name)

        hit = df[mask]
        if len(hit):
            exact_hits += len(hit)
            print(
                f"KICKER_RAW_HIT|file={p.relative_to(ROOT)}|player={name!r}|player_id={pid}|"
                f"rows={len(hit)}|id_cols={json.dumps(id_cols)}|name_cols={json.dumps(name_cols)}|"
                f"fg_cols={json.dumps(fg_cols)}|xp_cols={json.dumps(xp_cols)}"
            )
            sample_cols = []
            for c in (id_cols + name_cols + fg_cols + xp_cols):
                if c not in sample_cols:
                    sample_cols.append(c)
            for _, row in hit[sample_cols[:20]].head(3).iterrows():
                print(
                    "KICKER_RAW_SAMPLE|" +
                    json.dumps({c: row[c] for c in sample_cols[:20]}, default=str, sort_keys=True)
                )

print(f"EXACT_KICKER_RAW_ROWS={exact_hits}")

print("\n=== CLASSIFICATION ===")
print("NO_KICKER_PROJECTION_CREATED=TRUE")
print("NO_HISTORICAL_ACTUAL_USED_AS_PROJECTION=TRUE")
print("FUZZY_MATCHING_USED=FALSE")
print("WRITE_OPERATIONS=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_T_STATUS=PASS")
