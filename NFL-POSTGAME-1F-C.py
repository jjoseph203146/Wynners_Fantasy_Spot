#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-C
Projection semantic authority audit — READ ONLY.

Traces:
nfl_fanduel_player_pool.parquet
 -> fanduel_slate_projection_attach_v5.py
 -> fanduel_slate_projection_pool

Goal:
Prove which upstream field becomes internal_projection/model_projection and
whether the same semantics can support a schedule-wide immutable history source.

No writes. No restarts. No cron/updater/LIVE/injury/solver changes.
"""

from pathlib import Path
import sqlite3, hashlib
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"
PARQUET = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
ATTACH = ROOT / "fanduel_slate_projection_attach_v5.py"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(v):
    if pd.isna(v):
        return None
    return str(v).strip()

def main():
    print("=" * 92)
    print("WFS NFL — NFL-POSTGAME-1F-C")
    print("PROJECTION SEMANTIC AUTHORITY AUDIT — READ ONLY")
    print("=" * 92)

    for p in (PARQUET, ATTACH, DB):
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    print(f"UPSTREAM_PARQUET_SHA256={sha(PARQUET)}")
    print(f"PROJECTION_ATTACH_SHA256={sha(ATTACH)}")

    source_text = ATTACH.read_text(errors="replace")
    print("\n=== ATTACH CODE SEMANTIC REFERENCES ===")
    tokens = [
        "source_fantasy_projection",
        "ridge_projection",
        "internal_projection",
        "model_projection",
        "projection_source",
        "projection_status",
        "exact_name_team",
    ]
    for token in tokens:
        count = source_text.count(token)
        print(f"CODE_TOKEN_COUNT|{token}|{count}")

    print("\n=== RELEVANT ATTACH CODE LINES ===")
    for i, line in enumerate(source_text.splitlines(), 1):
        if any(t in line for t in tokens):
            print(f"CODE_LINE|{i}|{line.rstrip()}")

    df = pd.read_parquet(PARQUET)
    print("\n=== UPSTREAM SOURCE ===")
    print(f"UPSTREAM_ROWS={len(df)}")
    print("UPSTREAM_COLUMNS=" + ",".join(map(str, df.columns)))

    required_upstream = [
        "team", "player_id", "player_display_name", "game_id",
        "source_fantasy_projection", "ridge_projection"
    ]
    for col in required_upstream:
        print(f"UPSTREAM_COLUMN_{col}_PRESENT={str(col in df.columns).upper()}")

    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity failure")

        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(fanduel_slate_projection_pool)"
        )}
        needed = {
            "player", "team_internal", "game_key", "is_dst",
            "source_fantasy_projection", "internal_projection",
            "projection_source", "projection_match_method",
            "projection_identity", "projection_status",
            "model_projection", "slate_slug"
        }
        missing = sorted(needed - cols)
        print(f"DOWNSTREAM_REQUIRED_COLUMNS_MISSING={','.join(missing) if missing else 'NONE'}")
        if missing:
            raise RuntimeError(f"downstream missing required columns: {missing}")

        rows = conn.execute("""
            SELECT player, team_internal, game_key, is_dst, slate_slug,
                   source_fantasy_projection, internal_projection,
                   projection_source, projection_match_method,
                   projection_identity, projection_status, model_projection
            FROM fanduel_slate_projection_pool
            WHERE is_dst=0
            ORDER BY game_key, team_internal, player, slate_slug
        """).fetchall()

    print(f"DOWNSTREAM_OFFENSE_ROWS={len(rows)}")

    # Collapse repeated slate rows by exact player/team/game and require semantic agreement.
    groups = {}
    for r in rows:
        key = (norm(r["player"]), norm(r["team_internal"]), norm(r["game_key"]))
        sig = (
            r["source_fantasy_projection"],
            r["internal_projection"],
            norm(r["projection_source"]),
            norm(r["projection_match_method"]),
            norm(r["projection_identity"]),
            norm(r["projection_status"]),
            r["model_projection"],
        )
        groups.setdefault(key, set()).add(sig)

    conflicts = [k for k, sigs in groups.items() if len(sigs) != 1]
    print(f"DOWNSTREAM_UNIQUE_OFFENSE_KEYS={len(groups)}")
    print(f"CROSS_SLATE_SEMANTIC_CONFLICTS={len(conflicts)}")
    for k in conflicts[:20]:
        print("SEMANTIC_CONFLICT=" + "|".join(x or "" for x in k))

    # Compare downstream fields against each other.
    ready = []
    model_eq_internal = 0
    internal_eq_source = 0
    for key, sigs in groups.items():
        if len(sigs) != 1:
            continue
        sig = next(iter(sigs))
        src_proj, internal, psource, method, ident, status, model = sig
        if status == "READY":
            ready.append((key, sig))
            if internal == model:
                model_eq_internal += 1
            if internal == src_proj:
                internal_eq_source += 1

    print(f"UNIQUE_READY_OFFENSE_KEYS={len(ready)}")
    print(f"READY_MODEL_EQUALS_INTERNAL={model_eq_internal}/{len(ready)}")
    print(f"READY_INTERNAL_EQUALS_SOURCE_FANTASY={internal_eq_source}/{len(ready)}")

    source_methods = sorted({sig[2] for _, sig in ready})
    match_methods = sorted({sig[3] for _, sig in ready})
    print("READY_PROJECTION_SOURCES=" + ",".join(x or "NULL" for x in source_methods))
    print("READY_MATCH_METHODS=" + ",".join(x or "NULL" for x in match_methods))

    # Upstream identity/game coverage and direct comparison where exact game_id + player identity exists.
    if {"game_id", "player_id", "source_fantasy_projection", "ridge_projection"}.issubset(df.columns):
        u = df.copy()
        u["_game"] = u["game_id"].map(norm)
        u["_pid"] = u["player_id"].map(norm)
        u = u[(u["_game"].notna()) & (u["_pid"].notna())]
        dup = int(u.duplicated(["_game", "_pid"], keep=False).sum())
        print(f"UPSTREAM_EXACT_GAME_PLAYER_DUP_ROWS={dup}")

        # Downstream projection_identity may be GSIS:<id>; normalize exact prefix only.
        comparable = 0
        source_match = 0
        ridge_match = 0
        lookup = {}
        for _, r in u.iterrows():
            key = (r["_game"], r["_pid"])
            lookup.setdefault(key, []).append(r)

        for key, sig in ready:
            _, _, game = key
            ident = sig[4]
            if not ident:
                continue
            pid = ident[5:] if ident.startswith("GSIS:") else ident
            hits = lookup.get((game, pid), [])
            if len(hits) != 1:
                continue
            comparable += 1
            ur = hits[0]
            internal = sig[1]
            if internal == ur.get("source_fantasy_projection"):
                source_match += 1
            if internal == ur.get("ridge_projection"):
                ridge_match += 1

        print(f"EXACT_UPSTREAM_DOWNSTREAM_COMPARABLE={comparable}")
        print(f"INTERNAL_EQUALS_UPSTREAM_SOURCE_FANTASY={source_match}/{comparable}")
        print(f"INTERNAL_EQUALS_UPSTREAM_RIDGE={ridge_match}/{comparable}")

    print("\n=== ARCHITECTURE RESULT ===")
    print("SCHEDULE_AUTHORITY=games")
    print("SLATE_FILTER_MUST_NOT_DEFINE_HISTORICAL_CAPTURE=TRUE")
    print("HISTORICAL_PROJECTION_SEMANTICS_MUST_MATCH_CURRENT_DFS_MODEL=TRUE")
    print("AUTOMATION_WRITE_CHANGES_ALLOWED=FALSE")

    if conflicts:
        print("NFL_POSTGAME_1F_C_STATUS=FAIL_CLOSED")
        raise SystemExit(2)

    print("NFL_POSTGAME_1F_C_STATUS=PASS")
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
