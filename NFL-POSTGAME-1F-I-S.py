#!/usr/bin/env python3
from pathlib import Path
import csv
import hashlib
import json
import sqlite3

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"

TOKENS = (
    "field_goal_result",
    "field_goal_attempt",
    "field_goal",
    "kick_distance",
    "kicker_player",
    "kicker_name",
    "placekicker",
    "extra_point_result",
    "extra_point_attempt",
    "extra_point",
    "pat_result",
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
    return "" if s.lower() in {"nan", "none", "null"} else s

def token_hits(cols):
    out = []
    for c in cols:
        lc = str(c).lower()
        if any(tok in lc for tok in TOKENS):
            out.append(c)
    return out

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-S")
print("KICKER RAW-SOURCE AUTHORITY DISCOVERY — READ ONLY")
print("=" * 118)

if not DB.exists():
    print(f"FAIL_CLOSED_MISSING={DB}")
    raise SystemExit(2)

print(f"NFL_DB_SHA256={sha256(DB)}")

# 1) SQLite schema discovery.
print("\n=== NFL.DB SCHEMA ===")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
tables = [r["name"] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)]
db_hits = 0
for t in tables:
    cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{t}")')]
    hits = token_hits(cols)
    if hits:
        db_hits += 1
        print(f"DB_KICK_SOURCE|table={t}|columns={json.dumps(hits)}")
print(f"DB_KICK_SOURCE_TABLES={db_hits}")
con.close()

# 2) Recursive parquet/csv schema discovery.
print("\n=== FILE SCHEMA DISCOVERY ===")
file_hits = 0
roots = [ROOT / "data", ROOT / "raw", ROOT / "data/raw"]

for base in roots:
    if not base.exists():
        continue
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in {".parquet", ".csv"}:
            continue

        try:
            if suf == ".parquet":
                import pyarrow.parquet as pq
                cols = pq.read_schema(p).names
            else:
                with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                    cols = next(csv.reader(f), [])
        except Exception:
            continue

        hits = token_hits(cols)
        if hits:
            file_hits += 1
            try:
                rel = p.relative_to(ROOT)
            except Exception:
                rel = p
            print(
                f"FILE_KICK_SOURCE|file={rel}|columns={json.dumps(hits)}|"
                f"bytes={p.stat().st_size}|sha256={sha256(p)}"
            )

print(f"FILE_KICK_SOURCE_COUNT={file_hits}")

# 3) Code-source discovery to identify existing ingestion/parsing support.
print("\n=== CODE TOKEN DISCOVERY ===")
code_hits = 0
for p in sorted(ROOT.glob("*.py")):
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    matched = sorted({tok for tok in TOKENS if tok in text.lower()})
    if matched:
        code_hits += 1
        print(f"CODE_KICK_SOURCE|file={p.name}|tokens={json.dumps(matched)}|sha256={sha256(p)}")
print(f"CODE_KICK_SOURCE_COUNT={code_hits}")

print("\n=== CLASSIFICATION ===")
print("NO_KICKER_PROJECTION_CREATED=TRUE")
print("NO_HISTORICAL_ACTUAL_USED_AS_PROJECTION=TRUE")
print("FUZZY_MATCHING_USED=FALSE")
print("WRITE_OPERATIONS=0")
print("NFL_DB_WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_S_STATUS=PASS")
