#!/usr/bin/env python3
"""
WFS_STAGE5_SG_RESULT_COMPLETION_V1

Research-only FanDuel Single-Game result completion.

Purpose
-------
Create an explicit evidence-backed zero-result sidecar for contest players
missing from player_game_stats.

A player receives 0.0 FanDuel points ONLY when:

1. exact season/week/player identity is available,
2. exact Week 4 snap authority proves participation,
3. verified player_game_stats has no exact result row,
4. authoritative Week 4 NFLverse PBP contains no event for that player.

No production tables are modified.
No generic missing -> zero rule exists.
Players without snap evidence remain unresolved.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import pandas as pd
import nflreadpy as nfl


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/nfl.db"

OUT_DIR = ROOT / "data/research/single_game_result_completion_v1"
OUT = OUT_DIR / "2026_week_04.parquet"
MANIFEST = OUT_DIR / "2026_week_04_manifest.json"

SEASON = 2026
WEEK = 4

TARGETS = {
    # Exact snap-confirmed participants established in Stage 5.3E-4.
    "00-0033307": "Kendrick Bourne",
    "00-0037157": "Zonovan Knight",
    "00-0038728": "Jared Wayne",
    "00-0038547": "Luke Schoonmaker",
    "00-0039868": "Troy Franklin",
    "00-0038783": "Nate Adkins",
    "00-0039050": "Payne Durham",
    "00-0038589": "Josh Whyle",
    "00-0036628": "John Bates",
    "00-0039912": "Ben Sinnott",
    "00-0033217": "Mo Alie-Cox",
    "00-0035631": "Drew Sample",
    "00-0040208": "Tahj Brooks",
    "00-0041116": "Jack Endries",
    "00-0039067": "Rashee Rice",
    "00-0037291": "Jalen Nailor",
    "00-0040078": "Brashard Smith",
    "00-0040739": "Elijah Arroyo",
    "00-0035125": "Alec Ingold",
    "00-0036980": "Elijah Moore",
    "00-0037252": "Greg Dulcich",
    "00-0036187": "Reggie Gilliam",
    "00-0028986": "Case Keenum",
    "00-0037805": "Jeremy Ruckert",
    "00-0033757": "Robert Tonyan",
    "00-0031595": "Michael Burton",
    "00-0039796": "Rasheen Ali",
    "00-0039648": "David Martin-Robinson",
}

EXPECTED_TARGET_COUNT = 28


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("=" * 72)
print("WFS_STAGE5_SG_RESULT_COMPLETION_V1")
print("RESEARCH ONLY — NO PRODUCTION WRITES")
print("=" * 72)

if len(TARGETS) != EXPECTED_TARGET_COUNT:
    raise RuntimeError(
        f"FAIL_CLOSED:TARGET_COUNT:{len(TARGETS)}"
    )

# ---------------------------------------------------------------------
# 1. Verify aggregate result absence
# ---------------------------------------------------------------------

uri = f"file:{DB.resolve()}?mode=ro&immutable=1"

with sqlite3.connect(uri, uri=True) as con:
    stats = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            player_id,
            team,
            position,
            fanduel_points,
            fanduel_points_verified
        FROM player_game_stats
        WHERE season = ?
          AND week = ?
        """,
        con,
        params=(SEASON, WEEK),
    )

target_ids = set(TARGETS)

existing = stats[
    stats["player_id"].astype(str).isin(target_ids)
].copy()

print("TARGET_COUNT=", len(TARGETS))
print("EXISTING_RESULT_ROWS=", len(existing))

if not existing.empty:
    print(existing.to_string(index=False))
    raise RuntimeError(
        "FAIL_CLOSED:TARGET_ALREADY_HAS_PLAYER_GAME_RESULT"
    )

# ---------------------------------------------------------------------
# 2. Reload authoritative current NFLverse PBP
# ---------------------------------------------------------------------

raw = nfl.load_pbp([SEASON])

if hasattr(raw, "to_pandas"):
    pbp = raw.to_pandas()
else:
    pbp = pd.DataFrame(raw)

if "week" not in pbp.columns:
    raise RuntimeError("FAIL_CLOSED:PBP_MISSING_WEEK")

pbp["week"] = pd.to_numeric(
    pbp["week"],
    errors="coerce",
)

w4 = pbp[pbp["week"].eq(WEEK)].copy()

if w4.empty:
    raise RuntimeError("FAIL_CLOSED:NO_WEEK4_PBP")

if "game_id" not in w4.columns:
    raise RuntimeError("FAIL_CLOSED:PBP_MISSING_GAME_ID")

print("WEEK4_PBP_ROWS=", len(w4))
print(
    "WEEK4_PBP_GAMES=",
    w4["game_id"].astype(str).nunique(),
)

# E-18 established 15 completed Week 4 games.
if w4["game_id"].astype(str).nunique() != 15:
    raise RuntimeError(
        "FAIL_CLOSED:UNEXPECTED_WEEK4_PBP_GAME_COUNT"
    )

id_cols = [
    c
    for c in w4.columns
    if c.endswith("_player_id")
    or c in {
        "player_id",
        "fantasy_player_id",
    }
]

if not id_cols:
    raise RuntimeError(
        "FAIL_CLOSED:NO_PBP_PLAYER_ID_COLUMNS"
    )

pbp_hits = {}

for pid in TARGETS:
    count = 0

    for col in id_cols:
        count += int(
            w4[col]
            .astype("string")
            .eq(pid)
            .sum()
        )

    pbp_hits[pid] = count

nonzero_hits = {
    pid: n
    for pid, n in pbp_hits.items()
    if n != 0
}

print("TARGETS_WITH_PBP_EVENT=", len(nonzero_hits))

if nonzero_hits:
    print(nonzero_hits)
    raise RuntimeError(
        "FAIL_CLOSED:TARGET_HAS_PBP_EVENT"
    )

# ---------------------------------------------------------------------
# 3. Build evidence-backed zero sidecar
#
# Snap participation itself was established in Stage 5.3E-4.
# This V1 is intentionally restricted to that exact frozen 28-player set.
# ---------------------------------------------------------------------

captured_at = datetime.now(timezone.utc).isoformat()

rows = []

for pid, name in TARGETS.items():
    rows.append(
        {
            "season": SEASON,
            "week": WEEK,
            "player_id": pid,
            "player_name": name,
            "fanduel_points": 0.0,
            "result_status": (
                "EVIDENCE_BACKED_PARTICIPATED_ZERO"
            ),
            "snap_participation_authority": (
                "STAGE5_3E4_EXACT_WEEK4_SNAP_CONFIRMATION"
            ),
            "aggregate_result_authority": (
                "PLAYER_GAME_STATS_EXACT_RESULT_ABSENT"
            ),
            "pbp_authority": (
                "NFLVERSE_CURRENT_2026_WEEK4"
            ),
            "pbp_event_count": 0,
            "zero_assignment_policy": (
                "PARTICIPATED_AND_NO_SCORING_EVENT"
            ),
            "production_influence": False,
            "solver_influence": False,
            "captured_at_utc": captured_at,
        }
    )

out = pd.DataFrame(rows).sort_values(
    ["player_id"],
    kind="stable",
).reset_index(drop=True)

if len(out) != EXPECTED_TARGET_COUNT:
    raise RuntimeError(
        "FAIL_CLOSED:OUTPUT_ROW_COUNT"
    )

if out["player_id"].duplicated().any():
    raise RuntimeError(
        "FAIL_CLOSED:DUPLICATE_PLAYER_ID"
    )

if not out["fanduel_points"].eq(0.0).all():
    raise RuntimeError(
        "FAIL_CLOSED:NONZERO_COMPLETION_RESULT"
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)

out.to_parquet(
    OUT,
    index=False,
)

manifest = {
    "contract": "WFS_STAGE5_SG_RESULT_COMPLETION_V1",
    "season": SEASON,
    "week": WEEK,
    "rows": len(out),
    "result_status": "EVIDENCE_BACKED_PARTICIPATED_ZERO",
    "zero_policy": (
        "exact frozen Stage5.3E-4 snap participant "
        "+ absent exact player_game_stats result "
        "+ zero authoritative Week4 NFLverse PBP events"
    ),
    "excluded_from_zero_completion": [
        "00-0035704",
        "00-0039814",
        "00-0037745",
    ],
    "production_influence": False,
    "solver_influence": False,
    "created_at_utc": captured_at,
    "artifact": str(OUT.relative_to(ROOT)),
}

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("===== COMPLETION ARTIFACT =====")
print(out.to_string(index=False))

print()
print("OUTPUT=", OUT)
print("ROWS=", len(out))
print("OUTPUT_SHA256=", sha256(OUT))
print("MANIFEST=", MANIFEST)
print("MANIFEST_SHA256=", sha256(MANIFEST))

print()
print("PRODUCTION_WRITES=False")
print("PRODUCTION_INFLUENCE=False")
print("SOLVER_INFLUENCE=False")
print("GENERIC_MISSING_TO_ZERO=False")
print("UNRESOLVED_ACTIVE_NO_SNAP=3")
print("STAGE5_RESULT_COMPLETION_V1=PASS")
