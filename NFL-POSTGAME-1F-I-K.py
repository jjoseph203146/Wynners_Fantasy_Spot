#!/usr/bin/env python3
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
SG_ROOT = ROOT / "data/fanduel/single_game"
RAW_DIR = SG_ROOT / "raw"
MANIFEST = SG_ROOT / "manifest/slate_readiness_r1.json"
FULL_POOL = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"

TEAM_ALIAS = {
    "LA": "LAR",
    "WAS": "WSH",
}

RAW_REQUIRED_HEADER = [
    "player", "team", "gameInfo", "salary", "value",
    "completionsAttempts", "passingYards", "passingTouchdowns",
    "interceptionsThrown", "rushingAttempts", "rushingYards",
    "rushingTouchdowns", "receptions", "targets", "receivingYards",
    "receivingTouchdowns", "fantasy", "positionRank", "overallRank",
    "opponentDefensiveRank",
]

# Exact candidate names only. No fuzzy column guessing.
POOL_PLAYER_CANDIDATES = ["player", "player_name", "name", "fd_player_name"]
POOL_TEAM_CANDIDATES = ["team", "team_abbr", "team_abbreviation"]
POOL_GAMEINFO_CANDIDATES = ["fd_game_info", "gameInfo", "game_info"]
POOL_PLAYER_ID_CANDIDATES = ["player_id"]
POOL_POSITION_CANDIDATES = ["position", "pos", "fd_position"]

def clean(v):
    return re.sub(r"\s+", " ", str(v or "")).strip()

def canon_team(v):
    v = clean(v).upper()
    return TEAM_ALIAS.get(v, v)

def canon_game(v):
    m = re.fullmatch(r"\s*([A-Z]{2,3})\s+@\s+([A-Z]{2,3})\s*", clean(v))
    if not m:
        return None
    return f"{canon_team(m.group(1))} @ {canon_team(m.group(2))}"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def first_exact_col(columns, candidates):
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None

print("=" * 112)
print("WFS NFL — NFL-POSTGAME-1F-I-K")
print("R610 SINGLE-GAME IDENTITY / POSITION AUTHORITY AUDIT — READ ONLY")
print("=" * 112)

for required in [MANIFEST, FULL_POOL]:
    if not required.exists():
        print(f"FAIL_CLOSED_MISSING={required}")
        raise SystemExit(2)

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
ready_single = [
    s for s in manifest.get("slates", [])
    if s.get("production_eligible") is True
    and s.get("ready") is True
    and s.get("classification") == "SINGLE_GAME"
]

print(f"MANIFEST_SHA256={sha256_file(MANIFEST)}")
print(f"FULL_POOL_SHA256={sha256_file(FULL_POOL)}")
print(f"READY_SINGLE_GAME_COUNT={len(ready_single)}")

pool = pd.read_parquet(FULL_POOL)

print(f"FULL_POOL_ROWS={len(pool)}")
print(f"FULL_POOL_COLUMNS={list(pool.columns)}")

player_col = first_exact_col(pool.columns, POOL_PLAYER_CANDIDATES)
team_col = first_exact_col(pool.columns, POOL_TEAM_CANDIDATES)
game_col = first_exact_col(pool.columns, POOL_GAMEINFO_CANDIDATES)
player_id_col = first_exact_col(pool.columns, POOL_PLAYER_ID_CANDIDATES)
position_col = first_exact_col(pool.columns, POOL_POSITION_CANDIDATES)

print(f"POOL_PLAYER_COLUMN={player_col}")
print(f"POOL_TEAM_COLUMN={team_col}")
print(f"POOL_GAMEINFO_COLUMN={game_col}")
print(f"POOL_PLAYER_ID_COLUMN={player_id_col}")
print(f"POOL_POSITION_COLUMN={position_col}")

missing_authority = []
if player_col is None:
    missing_authority.append("PLAYER_COLUMN")
if team_col is None:
    missing_authority.append("TEAM_COLUMN")
if player_id_col is None:
    missing_authority.append("PLAYER_ID_COLUMN")
if position_col is None:
    missing_authority.append("POSITION_COLUMN")

if missing_authority:
    print("FAIL_CLOSED_AUTHORITY_COLUMNS_MISSING=" + ",".join(missing_authority))
    print("NFL_POSTGAME_1F_I_K_STATUS=FAIL_CLOSED")
    raise SystemExit(2)

# Build exact deterministic pool keys. Whitespace trim only for names; explicit team aliases only.
work = pool.copy()
work["_audit_player"] = work[player_col].map(clean)
work["_audit_team"] = work[team_col].map(canon_team)
work["_audit_player_id"] = work[player_id_col].map(clean)
work["_audit_position"] = work[position_col].map(clean)

if game_col is not None:
    work["_audit_game"] = work[game_col].map(canon_game)
else:
    work["_audit_game"] = None

overall_unmatched = []
overall_ambiguous = []
overall_matched = 0
overall_raw_salary_rows = 0

for s in ready_single:
    label = s["public_slate_name"]
    games = s.get("canonical_games") or []
    if len(games) != 1:
        print(f"FAIL_CLOSED|label={label!r}|reason=READY_SINGLE_NOT_ONE_GAME")
        raise SystemExit(2)

    expected_game = canon_game(games[0])
    raw_path = Path(s["staging_file"])

    # On R610, readiness file still contains Windows paths. Resolve by basename into isolated raw dir.
    raw_path = RAW_DIR / raw_path.name

    if not raw_path.exists():
        print(f"FAIL_CLOSED|label={label!r}|reason=RAW_FILE_MISSING|path={raw_path}")
        raise SystemExit(2)

    actual_sha = sha256_file(raw_path)
    expected_sha = clean(s.get("csv_sha256"))
    if actual_sha != expected_sha:
        print(
            f"FAIL_CLOSED|label={label!r}|reason=RAW_SHA_MISMATCH|"
            f"expected={expected_sha}|actual={actual_sha}"
        )
        raise SystemExit(2)

    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"FAIL_CLOSED|label={label!r}|reason=NO_RAW_ROWS")
        raise SystemExit(2)

    if list(rows[0].keys()) != RAW_REQUIRED_HEADER:
        print(f"FAIL_CLOSED|label={label!r}|reason=HEADER_MISMATCH")
        raise SystemExit(2)

    valid_raw = []
    for r in rows:
        player = clean(r.get("player"))
        team = canon_team(r.get("team"))
        game = canon_game(r.get("gameInfo"))
        salary = clean(r.get("salary")).replace("$", "").replace(",", "")

        try:
            salary_num = float(salary)
        except Exception:
            continue

        if player and team and game == expected_game and salary_num > 0:
            valid_raw.append((player, team, game, salary_num))

    print(
        f"\nSLATE|label={label!r}|game={expected_game}|"
        f"raw_rows={len(rows)}|valid_salary_identity_rows={len(valid_raw)}"
    )

    overall_raw_salary_rows += len(valid_raw)

    slate_matched = 0
    slate_unmatched = []
    slate_ambiguous = []

    for player, team, game, salary in valid_raw:
        if game_col is not None:
            cand = work[
                (work["_audit_player"] == player) &
                (work["_audit_team"] == team) &
                (work["_audit_game"] == game)
            ]
        else:
            cand = work[
                (work["_audit_player"] == player) &
                (work["_audit_team"] == team)
            ]

        if len(cand) == 0:
            slate_unmatched.append((player, team, game, salary))
            continue

        # Exact identity must collapse to exactly one nonblank player_id and one nonblank position.
        ids = sorted({clean(x) for x in cand["_audit_player_id"] if clean(x)})
        positions = sorted({clean(x) for x in cand["_audit_position"] if clean(x)})

        if len(ids) != 1 or len(positions) != 1:
            slate_ambiguous.append({
                "player": player,
                "team": team,
                "game": game,
                "candidate_rows": len(cand),
                "player_ids": ids,
                "positions": positions,
            })
            continue

        slate_matched += 1

    overall_matched += slate_matched
    overall_unmatched.extend([(label, *x) for x in slate_unmatched])
    overall_ambiguous.extend([(label, x) for x in slate_ambiguous])

    print(f"EXACT_MATCHED_ROWS={slate_matched}")
    print(f"EXACT_UNMATCHED_ROWS={len(slate_unmatched)}")
    print(f"AMBIGUOUS_IDENTITY_ROWS={len(slate_ambiguous)}")

    if slate_unmatched:
        print("UNMATCHED_SAMPLE_BEGIN")
        for x in slate_unmatched[:20]:
            print(f"UNMATCHED|player={x[0]!r}|team={x[1]}|game={x[2]}|salary={x[3]}")
        print("UNMATCHED_SAMPLE_END")

    if slate_ambiguous:
        print("AMBIGUOUS_SAMPLE_BEGIN")
        for x in slate_ambiguous[:20]:
            print(
                f"AMBIGUOUS|player={x['player']!r}|team={x['team']}|game={x['game']}|"
                f"candidate_rows={x['candidate_rows']}|player_ids={x['player_ids']}|"
                f"positions={x['positions']}"
            )
        print("AMBIGUOUS_SAMPLE_END")

print("\n=== SUMMARY ===")
print(f"RAW_VALID_SALARY_IDENTITY_ROWS={overall_raw_salary_rows}")
print(f"EXACT_MATCHED_ROWS={overall_matched}")
print(f"EXACT_UNMATCHED_ROWS={len(overall_unmatched)}")
print(f"AMBIGUOUS_IDENTITY_ROWS={len(overall_ambiguous)}")
print("FUZZY_MATCHING_USED=FALSE")
print("RAW_SINGLE_GAME_FILES_CHANGED=FALSE")
print("FULL_POOL_CHANGED=FALSE")
print("NFl_DB_CHANGED=FALSE")
print("CLASSIC_PRODUCTION_CHANGED=FALSE")
print("PUBLIC_SOLVER_CHANGED=FALSE")

if len(overall_unmatched) == 0 and len(overall_ambiguous) == 0:
    print("IDENTITY_POSITION_JOIN_PROOF=PASS")
    print("NFL_POSTGAME_1F_I_K_STATUS=PASS")
else:
    print("IDENTITY_POSITION_JOIN_PROOF=FAIL_CLOSED")
    print("NFL_POSTGAME_1F_I_K_STATUS=FAIL_CLOSED")
