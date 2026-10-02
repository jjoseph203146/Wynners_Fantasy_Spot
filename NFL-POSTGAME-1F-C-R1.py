#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-C-R1
Projection semantic conflict diagnosis — READ ONLY.

Purpose:
1) Correct the 1F-C comparison bug by mapping downstream matchup keys
   (e.g. CLE@JAX) to exact season/week-scoped games.game_id values.
2) Explain the 30 cross-slate semantic conflicts without changing production.
3) Prove which upstream field feeds downstream internal/model projection.

No writes. No restarts. No cron/updater/LIVE/injury/solver changes.
"""

from pathlib import Path
import sqlite3, hashlib, math
from collections import defaultdict
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"
PARQUET = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ATTACH = ROOT / "fanduel_slate_projection_attach_v5.py"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v).strip()

def num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def eqnum(a, b, tol=1e-12):
    a, b = num(a), num(b)
    if a is None or b is None:
        return a is None and b is None
    return abs(a-b) <= tol

def main():
    print("="*96)
    print("WFS NFL — NFL-POSTGAME-1F-C-R1")
    print("PROJECTION SEMANTIC CONFLICT DIAGNOSIS — READ ONLY")
    print("="*96)

    print(f"UPSTREAM_PARQUET_SHA256={sha(PARQUET)}")
    print(f"PROJECTION_ATTACH_SHA256={sha(ATTACH)}")

    u = pd.read_parquet(PARQUET)
    print(f"UPSTREAM_ROWS={len(u)}")

    # Resolve season/week from upstream itself.
    sw = u[["season","week"]].dropna().drop_duplicates()
    print(f"UPSTREAM_DISTINCT_SEASON_WEEK={len(sw)}")
    for _, r in sw.iterrows():
        print(f"UPSTREAM_SEASON_WEEK={int(r['season'])}|{int(r['week'])}")
    if len(sw) != 1:
        raise RuntimeError("expected one upstream season/week")
    season, week = map(int, sw.iloc[0].tolist())
    print(f"RESOLVED_SEASON={season}")
    print(f"RESOLVED_WEEK={week}")

    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity failure")

        games = c.execute("""
            SELECT game_id, away_team, home_team
            FROM games
            WHERE season=? AND week=?
            ORDER BY game_id
        """, (season, week)).fetchall()

        matchup_to_game = defaultdict(list)
        for g in games:
            matchup_to_game[f"{g['away_team']}@{g['home_team']}"].append(str(g["game_id"]))

        ambiguous = {k:v for k,v in matchup_to_game.items() if len(v)!=1}
        print(f"SCHEDULE_MATCHUPS={len(matchup_to_game)}")
        print(f"AMBIGUOUS_SEASON_WEEK_MATCHUPS={len(ambiguous)}")
        if ambiguous:
            for k,v in sorted(ambiguous.items()):
                print(f"AMBIGUOUS_MATCHUP={k}|{','.join(v)}")
            raise RuntimeError("season/week matchup mapping ambiguous")

        drows = c.execute("""
            SELECT player, team_internal, game_key, is_dst, slate_slug,
                   source_fantasy_projection, internal_projection,
                   projection_source, projection_match_method,
                   projection_identity, projection_status, model_projection
            FROM fanduel_slate_projection_pool
            WHERE is_dst=0
            ORDER BY game_key, team_internal, player, slate_slug
        """).fetchall()

    # Build exact upstream lookup keyed by exact game_id + raw GSIS id.
    required = {"game_id","player_id","source_fantasy_projection","ridge_projection",
                "player_display_name","team"}
    miss = sorted(required - set(u.columns))
    if miss:
        raise RuntimeError(f"upstream missing columns: {miss}")

    upstream = defaultdict(list)
    for _, r in u.iterrows():
        gid = norm(r["game_id"])
        pid = norm(r["player_id"])
        if gid and pid:
            upstream[(gid,pid)].append(r)

    dup_keys = [k for k,v in upstream.items() if len(v)!=1]
    print(f"UPSTREAM_EXACT_GAME_PLAYER_KEYS={len(upstream)}")
    print(f"UPSTREAM_DUPLICATE_GAME_PLAYER_KEYS={len(dup_keys)}")

    # Group downstream duplicates across slates.
    groups = defaultdict(list)
    unmapped_game_keys = set()
    for r in drows:
        game_key = norm(r["game_key"])
        gids = matchup_to_game.get(game_key, [])
        if len(gids) != 1:
            unmapped_game_keys.add(game_key)
            continue
        gid = gids[0]
        key = (norm(r["player"]), norm(r["team_internal"]), gid)
        groups[key].append(r)

    print(f"DOWNSTREAM_ROWS={len(drows)}")
    print(f"DOWNSTREAM_UNMAPPED_GAME_KEYS={len(unmapped_game_keys)}")
    for x in sorted(unmapped_game_keys):
        print(f"UNMAPPED_GAME_KEY={x}")

    conflict_groups = []
    exact_agree_groups = 0
    for key, rows in groups.items():
        sigs = set()
        for r in rows:
            sigs.add((
                num(r["source_fantasy_projection"]),
                num(r["internal_projection"]),
                norm(r["projection_source"]),
                norm(r["projection_match_method"]),
                norm(r["projection_identity"]),
                norm(r["projection_status"]),
                num(r["model_projection"]),
            ))
        if len(sigs) > 1:
            conflict_groups.append((key, rows, sigs))
        else:
            exact_agree_groups += 1

    print(f"DOWNSTREAM_UNIQUE_PLAYER_GAME_KEYS={len(groups)}")
    print(f"CROSS_SLATE_EXACT_AGREEMENT_GROUPS={exact_agree_groups}")
    print(f"CROSS_SLATE_SEMANTIC_CONFLICT_GROUPS={len(conflict_groups)}")

    print("\n=== CONFLICT DIAGNOSIS ===")
    conflict_by_field = defaultdict(int)
    for key, rows, sigs in conflict_groups:
        fields = [
            "source_fantasy_projection","internal_projection","projection_source",
            "projection_match_method","projection_identity","projection_status","model_projection"
        ]
        vals = {f:set() for f in fields}
        for r in rows:
            for f in fields:
                v = num(r[f]) if f in {"source_fantasy_projection","internal_projection","model_projection"} else norm(r[f])
                vals[f].add(v)
        changed = [f for f in fields if len(vals[f]) > 1]
        for f in changed:
            conflict_by_field[f] += 1
        print(f"CONFLICT_KEY={key[2]}|{key[1]}|{key[0]}|changed={','.join(changed)}")
        for r in rows:
            print(
                "  ROW|slate={}|src={}|internal={}|model={}|source={}|method={}|identity={}|status={}".format(
                    norm(r["slate_slug"]),
                    num(r["source_fantasy_projection"]),
                    num(r["internal_projection"]),
                    num(r["model_projection"]),
                    norm(r["projection_source"]),
                    norm(r["projection_match_method"]),
                    norm(r["projection_identity"]),
                    norm(r["projection_status"]),
                )
            )

    print("\n=== CONFLICT FIELD COUNTS ===")
    for f in [
        "source_fantasy_projection","internal_projection","projection_source",
        "projection_match_method","projection_identity","projection_status","model_projection"
    ]:
        print(f"CONFLICT_GROUPS_CHANGED_{f}={conflict_by_field[f]}")

    # Compare every READY downstream row to upstream using corrected game_id mapping.
    comparable = 0
    missing_identity = 0
    missing_upstream = 0
    duplicate_upstream = 0
    internal_eq_source = 0
    internal_eq_ridge = 0
    model_eq_internal = 0
    identity_name_team_disagree = 0

    sample_mismatches = []

    for r in drows:
        if norm(r["projection_status"]) != "READY":
            continue

        if eqnum(r["model_projection"], r["internal_projection"]):
            model_eq_internal += 1

        gk = norm(r["game_key"])
        gids = matchup_to_game.get(gk, [])
        if len(gids) != 1:
            continue
        gid = gids[0]

        ident = norm(r["projection_identity"])
        if not ident:
            missing_identity += 1
            continue
        pid = ident[5:] if ident.startswith("GSIS:") else ident

        hits = upstream.get((gid,pid), [])
        if not hits:
            missing_upstream += 1
            continue
        if len(hits) != 1:
            duplicate_upstream += 1
            continue

        ur = hits[0]
        comparable += 1
        if eqnum(r["internal_projection"], ur["source_fantasy_projection"]):
            internal_eq_source += 1
        if eqnum(r["internal_projection"], ur["ridge_projection"]):
            internal_eq_ridge += 1

        if norm(ur["team"]) != norm(r["team_internal"]) or norm(ur["player_display_name"]) != norm(r["player"]):
            identity_name_team_disagree += 1
            if len(sample_mismatches) < 30:
                sample_mismatches.append((
                    gid, pid, norm(r["player"]), norm(r["team_internal"]),
                    norm(ur["player_display_name"]), norm(ur["team"])
                ))

    ready_count = sum(1 for r in drows if norm(r["projection_status"])=="READY")
    print("\n=== CORRECTED EXACT UPSTREAM/DOWNSTREAM COMPARISON ===")
    print(f"READY_DOWNSTREAM_ROWS={ready_count}")
    print(f"READY_MODEL_EQUALS_INTERNAL={model_eq_internal}/{ready_count}")
    print(f"READY_MISSING_PROJECTION_IDENTITY={missing_identity}")
    print(f"READY_MISSING_EXACT_UPSTREAM_GAME_PLAYER={missing_upstream}")
    print(f"READY_DUPLICATE_EXACT_UPSTREAM_GAME_PLAYER={duplicate_upstream}")
    print(f"EXACT_UPSTREAM_DOWNSTREAM_COMPARABLE={comparable}")
    print(f"INTERNAL_EQUALS_UPSTREAM_SOURCE_FANTASY={internal_eq_source}/{comparable}")
    print(f"INTERNAL_EQUALS_UPSTREAM_RIDGE={internal_eq_ridge}/{comparable}")
    print(f"EXACT_IDENTITY_NAME_TEAM_DISAGREEMENTS={identity_name_team_disagree}")

    for x in sample_mismatches:
        print(
            f"IDENTITY_NAME_TEAM_DISAGREE|game={x[0]}|pid={x[1]}|"
            f"downstream={x[2]}|{x[3]}|upstream={x[4]}|{x[5]}"
        )

    # PASS is diagnostic: no writes and corrected mapping executed.
    # We intentionally do not require semantic conflicts to be zero; the point is to explain them.
    print("\n=== R1 RESULT ===")
    print("CORRECTED_SEASON_WEEK_GAME_MAPPING=TRUE")
    print("NO_FUZZY_MATCHING=TRUE")
    print("AUTOMATION_WRITE_CHANGES_ALLOWED=FALSE")
    print("NFL_POSTGAME_1F_C_R1_STATUS=PASS")
    print("READ_ONLY_AUDIT=TRUE")
    print("DATABASE_WRITES=0")
    print("FILE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    print("UPDATER_CHANGES=0")
    print("LIVE_CHANGES=0")
    print("INJURY_PIPELINE_CHANGES=0")
    print("SOLVER_CHANGES=0")

if __name__ == "__main__":
    main()
