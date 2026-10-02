#!/usr/bin/env python3
# WFS NFL — NFL-POSTGAME-1F-B
# Upstream projection-source coverage audit — READ ONLY.

from pathlib import Path
import sqlite3, hashlib, re
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data/nfl.db"
ATTACH = ROOT / "fanduel_slate_projection_attach_v5.py"
POOL = ROOT / "fanduel_player_pool.py"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ro_db():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c

def main():
    print("="*88)
    print("WFS NFL — NFL-POSTGAME-1F-B")
    print("UPSTREAM PROJECTION-SOURCE COVERAGE AUDIT — READ ONLY")
    print("="*88)
    print(f"fanduel_slate_projection_attach_v5.py_SHA256={sha(ATTACH)}")
    print(f"fanduel_player_pool.py_SHA256={sha(POOL)}")

    src = ATTACH.read_text(errors="replace")
    paths = sorted(set(re.findall(r"[\"']([^\"']+\\.(?:parquet|csv))[\"']", src)))
    print("\n=== STATIC SOURCE REFERENCES IN PROJECTION ATTACH ===")
    for x in paths:
        print(f"SOURCE_REFERENCE={x}")

    candidates = []
    for rel in paths:
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / rel
        if p.exists():
            candidates.append(p)

    known = ROOT / "data/parquet/nfl_fanduel_player_pool.parquet"
    if known.exists() and known not in candidates:
        candidates.append(known)
    print(f"EXISTING_SOURCE_FILES={len(candidates)}")

    with ro_db() as c:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity failure")
        games = c.execute(
            "SELECT game_id,season,week,away_team,home_team,game_date,gametime,completed "
            "FROM games WHERE season=2026 AND week=1 "
            "ORDER BY game_date,gametime,game_id"
        ).fetchall()
        print(f"WEEK1_SCHEDULE_GAMES={len(games)}")
        schedule_teams = {str(g["away_team"]) for g in games} | {str(g["home_team"]) for g in games}

    print("\n=== UPSTREAM FILE AUDIT ===")
    audited = 0
    full_team_coverage = False

    for p in candidates:
        audited += 1
        print(f"\nFILE={p}")
        print(f"FILE_SHA256={sha(p)}")
        try:
            df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        except Exception as e:
            print(f"READ_ERROR={type(e).__name__}:{e}")
            continue

        print(f"ROWS={len(df)}")
        print("COLUMNS=" + ",".join(map(str, df.columns)))
        lower = {str(col).lower(): col for col in df.columns}
        team_col = next((lower[k] for k in ("team_internal","team","team_abbr","team_abbreviation") if k in lower), None)
        opp_col = next((lower[k] for k in ("opponent_team","opponent","opp","opp_team") if k in lower), None)
        game_col = next((lower[k] for k in ("game_id","game_key","gameinfo","game_info") if k in lower), None)
        proj_col = next((lower[k] for k in ("model_projection","projection","fantasy_points","fantasy_projection","projected_points") if k in lower), None)
        id_col = next((lower[k] for k in ("player_id","gsis_id","projection_identity") if k in lower), None)

        print(f"DETECTED_TEAM_COLUMN={team_col}")
        print(f"DETECTED_OPPONENT_COLUMN={opp_col}")
        print(f"DETECTED_GAME_COLUMN={game_col}")
        print(f"DETECTED_PROJECTION_COLUMN={proj_col}")
        print(f"DETECTED_IDENTITY_COLUMN={id_col}")

        if team_col:
            series = df[team_col].dropna().astype(str).str.strip()
            teams = set(series)
            missing = sorted(schedule_teams - teams)
            print(f"DISTINCT_TEAMS={len(teams)}")
            print(f"WEEK1_SCHEDULE_TEAMS_COVERED={len(schedule_teams)-len(missing)}/{len(schedule_teams)}")
            print("WEEK1_MISSING_TEAMS=" + (",".join(missing) if missing else "NONE"))
            if not missing:
                full_team_coverage = True
            for g in games:
                a, h = str(g["away_team"]), str(g["home_team"])
                ac = int((series == a).sum())
                hc = int((series == h).sum())
                print(f"TEAM_ROW_COUNTS|{g['game_id']}|{a}={ac}|{h}={hc}")

        if team_col and opp_col:
            pairs = set()
            for _, r in df[[team_col, opp_col]].dropna().iterrows():
                pairs.add((str(r[team_col]).strip(), str(r[opp_col]).strip()))
            missing_games = []
            covered = 0
            for g in games:
                a, h = str(g["away_team"]), str(g["home_team"])
                if (a,h) in pairs or (h,a) in pairs:
                    covered += 1
                else:
                    missing_games.append(str(g["game_id"]))
            print(f"EXACT_MATCHUP_COVERAGE={covered}/{len(games)}")
            print("MISSING_MATCHUPS=" + (",".join(missing_games) if missing_games else "NONE"))

    print("\n=== ARCHITECTURE RESULT ===")
    print(f"UPSTREAM_FILES_AUDITED={audited}")
    print(f"ANY_UPSTREAM_FULL_WEEK_TEAM_COVERAGE={str(full_team_coverage).upper()}")
    print("LIKELY_NEXT_ACTION=" + (
        "DESIGN_SCHEDULE_WIDE_PROJECTION_CAPTURE_FROM_UPSTREAM_SOURCE"
        if full_team_coverage else
        "AUDIT_OR_EXPAND_PROJECTION_ACQUISITION_BEFORE_CAPTURE_AUTOMATION"
    ))
    print("\nNFL_POSTGAME_1F_B_STATUS=PASS")
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
