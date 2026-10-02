#!/usr/bin/env python3
from pathlib import Path
import sqlite3
import pandas as pd
import hashlib
import json
import re

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"
IDENTITY = ROOT / "data/fanduel/single_game/derived/single_game_identity_pool.parquet"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def qident(name):
    return '"' + str(name).replace('"', '""') + '"'

def main():
    print("=" * 116)
    print("WFS NFL — NFL-POSTGAME-1F-I-O")
    print("SHOWDOWN D/ST + K PROJECTION AUTHORITY AUDIT — READ ONLY")
    print("=" * 116)

    if not IDENTITY.exists():
        raise RuntimeError(f"Missing identity pool: {IDENTITY}")
    if not DB.exists():
        raise RuntimeError(f"Missing nfl.db: {DB}")

    pool = pd.read_parquet(IDENTITY)
    print(f"IDENTITY_POOL_SHA256={sha256(IDENTITY)}")
    print(f"IDENTITY_ROWS={len(pool)}")

    pos_col = "position" if "position" in pool.columns else "fd_position"
    name_col = "player" if "player" in pool.columns else ("fd_name" if "fd_name" in pool.columns else None)
    if not name_col or pos_col not in pool.columns:
        raise RuntimeError("Could not resolve identity-pool player/position columns.")

    special = pool[pool[pos_col].astype(str).str.upper().isin(["DST", "D/ST", "K"])].copy()
    print(f"SPECIAL_ROWS={len(special)}")
    print(f"DST_ROWS={(special[pos_col].astype(str).str.upper().isin(['DST','D/ST'])).sum()}")
    print(f"K_ROWS={(special[pos_col].astype(str).str.upper().eq('K')).sum()}")

    uri = f"file:{DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    print("\n=== CANDIDATE NFL.DB TABLES ===")
    candidates = []
    tokens = ("dst", "defen", "kick", "field", "projection", "special", "player_game", "fantasy")
    for t in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({qident(t)})").fetchall()]
        blob = (t + " " + " ".join(cols)).lower()
        if any(tok in blob for tok in tokens):
            candidates.append((t, cols))
            print(f"TABLE|{t}|COLUMNS={json.dumps(cols)}")

    print("\n=== EXACT SPECIAL-PLAYER SEARCH ===")
    wanted_names = sorted(set(special[name_col].dropna().astype(str)))
    id_col = "player_id" if "player_id" in special.columns else None
    team_col = "team" if "team" in special.columns else None

    plausible_name_cols = {"player","player_name","player_display_name","name","fd_name","display_name"}
    plausible_id_cols = {"player_id","gsis_id","identity_key"}
    plausible_team_cols = {"team","team_abbr","club","posteam"}
    plausible_pos_cols = {"position","pos","fd_position"}
    plausible_proj_cols = {
        "projection","projected_points","fantasy_projection","fanduel_projection",
        "fd_projection","model_projection","internal_projection","ridge_projection",
        "fantasy_points","fanduel_points"
    }

    for t, cols in candidates:
        colset = set(cols)
        name_cols = [c for c in cols if c.lower() in plausible_name_cols]
        if not name_cols:
            continue
        select_cols = []
        for c in cols:
            lc = c.lower()
            if (lc in plausible_name_cols or lc in plausible_id_cols or lc in plausible_team_cols
                or lc in plausible_pos_cols or lc in plausible_proj_cols
                or "kick" in lc or "field_goal" in lc or "extra_point" in lc):
                select_cols.append(c)
        select_cols = select_cols[:20]
        for nc in name_cols:
            placeholders = ",".join(["?"] * len(wanted_names))
            sql = (
                f"SELECT {', '.join(qident(c) for c in select_cols)} "
                f"FROM {qident(t)} WHERE {qident(nc)} IN ({placeholders})"
            )
            try:
                rows = conn.execute(sql, wanted_names).fetchall()
            except Exception as e:
                print(f"SEARCH_ERROR|table={t}|name_col={nc}|error={e}")
                continue
            for row in rows:
                payload = dict(zip(select_cols, row))
                print(f"DB_HIT|table={t}|{json.dumps(payload, default=str, sort_keys=True)}")

    print("\n=== D/ST AUTHORITY TABLE PROBE ===")
    for t, cols in candidates:
        blob = (t + " " + " ".join(cols)).lower()
        if "dst" in blob or "defen" in blob:
            count = conn.execute(f"SELECT COUNT(*) FROM {qident(t)}").fetchone()[0]
            print(f"DST_TABLE_CANDIDATE|{t}|ROWS={count}|COLUMNS={json.dumps(cols)}")

    print("\n=== KICKER / SPECIAL-TEAMS INPUT PROBE ===")
    for t, cols in candidates:
        blob = (t + " " + " ".join(cols)).lower()
        if any(x in blob for x in ("kick", "field_goal", "extra_point", "special")):
            count = conn.execute(f"SELECT COUNT(*) FROM {qident(t)}").fetchone()[0]
            print(f"K_TABLE_CANDIDATE|{t}|ROWS={count}|COLUMNS={json.dumps(cols)}")

    conn.close()

    print("\n=== FILE-SYSTEM PROJECTION SOURCE PROBE ===")
    patterns = [
        "*dst*", "*defense*", "*kicker*", "*kick*", "*projection*"
    ]
    seen = set()
    for base in [ROOT, ROOT / "data", ROOT / "data/parquet", ROOT / "data/csv"]:
        if not base.exists():
            continue
        for pat in patterns:
            for p in base.glob(pat):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    try:
                        print(f"FILE_CANDIDATE|{p.relative_to(ROOT)}|SHA256={sha256(p)}|BYTES={p.stat().st_size}")
                    except Exception:
                        pass

    print("\n=== SAFETY ===")
    print("WRITE_OPERATIONS=0")
    print("NFL_DB_WRITE_OPERATIONS=0")
    print("CLASSIC_PRODUCTION_CHANGED=FALSE")
    print("PUBLIC_SOLVER_CHANGED=FALSE")
    print("PROJECTION_VALUES_CHANGED=FALSE")
    print("FUZZY_MATCHING_USED=FALSE")
    print("NFL_POSTGAME_1F_I_O_STATUS=PASS")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED|{type(exc).__name__}|{exc}")
        print("WRITE_OPERATIONS=0")
        print("NFL_DB_WRITE_OPERATIONS=0")
        print("CLASSIC_PRODUCTION_CHANGED=FALSE")
        print("PUBLIC_SOLVER_CHANGED=FALSE")
        print("PROJECTION_VALUES_CHANGED=FALSE")
        print("NFL_POSTGAME_1F_I_O_STATUS=FAIL_CLOSED")
        raise
