#!/usr/bin/env python3
from pathlib import Path
import ast
import hashlib
import json
import re

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
IDENTITY_POOL = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"

CANDIDATE_FILES = [
    ROOT / "production_projection.py",
    ROOT / "fanduel_player_pool.py",
    ROOT / "fanduel_slate_projection_attach_v5.py",
]

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

print("=" * 118)
print("WFS NFL — NFL-POSTGAME-1F-I-N")
print("SHOWDOWN PROJECTION EXTENSION ARCHITECTURE AUDIT — READ ONLY")
print("=" * 118)

if not IDENTITY_POOL.exists():
    print(f"FAIL_CLOSED_MISSING={IDENTITY_POOL}")
    raise SystemExit(2)

sg = pd.read_parquet(IDENTITY_POOL)
missing = sg[sg["ridge_projection"].isna()].copy()

print(f"IDENTITY_POOL_SHA256={sha256_file(IDENTITY_POOL)}")
print(f"SINGLE_GAME_ROWS={len(sg)}")
print(f"RIDGE_READY_ROWS={int(sg['ridge_projection'].notna().sum())}")
print(f"RIDGE_MISSING_ROWS={len(missing)}")

print("\n=== MISSING PROJECTION POPULATION ===")
for pos, n in missing["position"].value_counts(dropna=False).sort_index().items():
    print(f"MISSING_POSITION|{clean(pos)}|COUNT={int(n)}")

print("\n=== PIPELINE FILES ===")
existing = []
for p in CANDIDATE_FILES:
    if p.exists():
        existing.append(p)
        print(f"FILE|{p.name}|SHA256={sha256_file(p)}|BYTES={p.stat().st_size}")
    else:
        print(f"FILE_MISSING|{p}")

if not existing:
    print("FAIL_CLOSED_NO_PIPELINE_FILES")
    raise SystemExit(2)

KEYWORDS = [
    "ridge_projection",
    "source_fantasy_projection",
    "nfl_production_projection",
    "position",
    "QB", "RB", "WR", "TE", "FB", "DST", "D/ST", "K",
    "player_id",
    "identity_key",
    "optimizer_eligible",
]

print("\n=== RELEVANT SOURCE LINES ===")
for p in existing:
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print(f"\n--- {p.name} ---")
    shown = set()
    for i, line in enumerate(lines, 1):
        if any(k.lower() in line.lower() for k in KEYWORDS):
            lo = max(1, i - 2)
            hi = min(len(lines), i + 2)
            for j in range(lo, hi + 1):
                if j not in shown:
                    print(f"{p.name}:{j}:{lines[j-1]}")
                    shown.add(j)

print("\n=== STATIC POSITION FILTERS / LITERALS ===")
for p in existing:
    text = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"AST_PARSE_FAIL|{p.name}|{e}")
        continue

    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            vals = []
            ok = True
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    vals.append(elt.value)
                else:
                    ok = False
                    break
            if ok and vals and any(v.upper() in {"QB","RB","WR","TE","FB","K","DST","D/ST","DEF"} for v in vals):
                print(f"POSITION_LITERAL|{p.name}|line={getattr(node,'lineno','?')}|values={vals}")

print("\n=== OUTPUT / INPUT REFERENCES ===")
patterns = [
    r'["\']([^"\']*nfl_production_projection[^"\']*)["\']',
    r'["\']([^"\']*nfl_fanduel_player_pool[^"\']*)["\']',
    r'["\']([^"\']*fanduel[^"\']*\.csv)["\']',
]
for p in existing:
    text = p.read_text(encoding="utf-8", errors="replace")
    refs = []
    for pat in patterns:
        refs.extend(re.findall(pat, text, flags=re.I))
    for ref in sorted(set(refs)):
        print(f"REFERENCE|{p.name}|{ref}")

print("\n=== MISSING ROW DETAIL ===")
for _, r in missing.sort_values(
    ["public_slate_name","team","position","player"], kind="mergesort"
).iterrows():
    print(
        "MISSING|"
        f"slate={clean(r['public_slate_name'])!r}|"
        f"game={clean(r['game'])}|"
        f"player={clean(r['player'])!r}|"
        f"team={clean(r['team'])}|"
        f"position={clean(r['position'])}|"
        f"player_id={clean(r['player_id'])!r}|"
        f"identity_source={clean(r['identity_source'])}|"
        f"salary={r['salary']}|"
        f"raw_fantasy={r['raw_fantasy']!r}|"
        f"source_fantasy={r['full_pool_source_fantasy_projection']!r}"
    )

print("\n=== SAFETY ===")
print("WRITE_OPERATIONS=0")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")
print("PROJECTION_VALUES_CHANGED=FALSE")
print("NFL_POSTGAME_1F_I_N_STATUS=PASS")
