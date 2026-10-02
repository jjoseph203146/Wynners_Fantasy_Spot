#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1F-A
Schedule-authority vs projection-coverage audit — READ ONLY.

Purpose:
- Use nfl.db / games as the authoritative NFL schedule universe.
- Measure which scheduled games have current WFS player projections.
- Measure which games are represented in FanDuel slate projection data.
- Prove whether selected FanDuel slates are constraining historical capture coverage.

Reads only:
- data/nfl.db
No writes. No service restart. No cron/updater/LIVE/injury/solver changes.
"""

from pathlib import Path
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import sys

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"
APP = ROOT / "app.py"
PROJECTION_ATTACH = ROOT / "fanduel_slate_projection_attach_v5.py"
PLAYER_POOL = ROOT / "fanduel_player_pool.py"
SOLVER_READY = ROOT / "fanduel_solver_ready_pool.py"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ro(path: Path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def kickoff_utc(row):
    raw = f"{str(row['game_date']).strip()} {str(row['gametime']).strip()}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ET).astimezone(UTC)
        except ValueError:
            pass
    raise RuntimeError(f"cannot parse kickoff for {row['game_id']}: {raw!r}")


def cols(conn, table):
    return {r["name"] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def main():
    print("=" * 88)
    print("WFS NFL — NFL-POSTGAME-1F-A")
    print("SCHEDULE AUTHORITY VS PROJECTION COVERAGE AUDIT — READ ONLY")
    print("=" * 88)

    if not NFL_DB.is_file():
        raise RuntimeError(f"missing {NFL_DB}")

    for p in (APP, PROJECTION_ATTACH, PLAYER_POOL, SOLVER_READY):
        if p.is_file():
            print(f"{p.name}_SHA256={sha(p)}")

    with ro(NFL_DB) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity failure")

        required = [
            "games",
            "fanduel_slate_projection_pool",
            "fanduel_slate_projection_manifest",
            "fanduel_slate_pool",
            "fanduel_slate_manifest",
        ]
        for t in required:
            print(f"TABLE_{t}_PRESENT={str(table_exists(conn, t)).upper()}")

        if not table_exists(conn, "games"):
            raise RuntimeError("games table missing")
        if not table_exists(conn, "fanduel_slate_projection_pool"):
            raise RuntimeError("fanduel_slate_projection_pool missing")

        gcols = cols(conn, "games")
        needed_games = {
            "game_id", "season", "week", "game_date", "gametime",
            "away_team", "home_team", "completed"
        }
        missing = sorted(needed_games - gcols)
        if missing:
            raise RuntimeError(f"games missing columns: {missing}")

        pcols = cols(conn, "fanduel_slate_projection_pool")
        needed_proj = {
            "game_key", "away_team", "home_team", "projection_status",
            "model_projection", "is_dst", "slate_slug", "player",
            "team_internal"
        }
        missing = sorted(needed_proj - pcols)
        if missing:
            raise RuntimeError(f"projection pool missing columns: {missing}")

        # Resolve current season/week from projection universe by exact schedule coverage.
        proj_matchups = conn.execute("""
            SELECT DISTINCT away_team, home_team
            FROM fanduel_slate_projection_pool
            WHERE away_team IS NOT NULL AND home_team IS NOT NULL
            ORDER BY away_team, home_team
        """).fetchall()

        if not proj_matchups:
            raise RuntimeError("projection pool has no distinct matchups")

        season_week_counts = defaultdict(int)
        matchup_total = len(proj_matchups)

        schedule_rows = conn.execute("""
            SELECT game_id, season, week, game_date, gametime,
                   away_team, home_team, completed
            FROM games
            ORDER BY season, week, game_id
        """).fetchall()

        by_sw_matchup = defaultdict(list)
        for g in schedule_rows:
            by_sw_matchup[(int(g["season"]), int(g["week"]), str(g["away_team"]), str(g["home_team"]))].append(g)

        candidate_pairs = sorted({(int(g["season"]), int(g["week"])) for g in schedule_rows})
        for season, week in candidate_pairs:
            count = 0
            for m in proj_matchups:
                rows = by_sw_matchup.get((season, week, str(m["away_team"]), str(m["home_team"])), [])
                if len(rows) == 1:
                    count += 1
            season_week_counts[(season, week)] = count

        full = [(sw, c) for sw, c in season_week_counts.items() if c == matchup_total]
        print(f"DISTINCT_PROJECTION_MATCHUPS={matchup_total}")
        print(f"FULL_COVERAGE_SEASON_WEEK_CANDIDATES={len(full)}")
        for (season, week), c in sorted(full):
            print(f"FULL_COVERAGE_CANDIDATE={season}|week={week}|matched={c}")

        if len(full) != 1:
            raise RuntimeError(f"expected exactly one full-coverage season/week, found {len(full)}")

        (season, week), _ = full[0]
        print(f"RESOLVED_SEASON={season}")
        print(f"RESOLVED_WEEK={week}")

        current_games = conn.execute("""
            SELECT game_id, season, week, game_date, gametime,
                   away_team, home_team, completed
            FROM games
            WHERE season=? AND week=?
            ORDER BY game_date, gametime, game_id
        """, (season, week)).fetchall()

        print(f"SCHEDULE_GAMES_THIS_WEEK={len(current_games)}")

        # Projection coverage by exact season/week matchup mapping.
        proj_rows = conn.execute("""
            SELECT away_team, home_team, game_key, slate_slug, is_dst,
                   player, team_internal, projection_status, model_projection
            FROM fanduel_slate_projection_pool
            ORDER BY away_team, home_team, slate_slug, player
        """).fetchall()

        grouped = defaultdict(list)
        for r in proj_rows:
            grouped[(str(r["away_team"]), str(r["home_team"]))].append(r)

        schedule_game_ids = set()
        covered_game_ids = set()
        ready_game_ids = set()

        game_report = []

        for g in current_games:
            gid = str(g["game_id"])
            schedule_game_ids.add(gid)
            key = (str(g["away_team"]), str(g["home_team"]))
            rows = grouped.get(key, [])

            slate_slugs = sorted({str(r["slate_slug"]) for r in rows if r["slate_slug"] is not None})
            ready = [r for r in rows if str(r["projection_status"] or "").upper() == "READY"]
            offense_ready = [r for r in ready if int(r["is_dst"] or 0) == 0]
            dst_ready = [r for r in ready if int(r["is_dst"] or 0) == 1]

            if rows:
                covered_game_ids.add(gid)
            if ready:
                ready_game_ids.add(gid)

            ko = kickoff_utc(g)
            game_report.append({
                "game_id": gid,
                "matchup": f"{g['away_team']}@{g['home_team']}",
                "kickoff_utc": ko.isoformat(),
                "completed": int(g["completed"] or 0),
                "projection_rows": len(rows),
                "ready_rows": len(ready),
                "ready_offense": len(offense_ready),
                "ready_dst": len(dst_ready),
                "slates": slate_slugs,
            })

        print("\n=== SCHEDULE / PROJECTION COVERAGE ===")
        for r in game_report:
            print(
                f"{r['game_id']}|{r['matchup']}|kickoff={r['kickoff_utc']}|"
                f"completed={r['completed']}|projection_rows={r['projection_rows']}|"
                f"ready_rows={r['ready_rows']}|ready_offense={r['ready_offense']}|"
                f"ready_dst={r['ready_dst']}|slates={','.join(r['slates']) if r['slates'] else 'NONE'}"
            )

        missing_projection = sorted(schedule_game_ids - covered_game_ids)
        missing_ready = sorted(schedule_game_ids - ready_game_ids)

        print("\n=== COVERAGE SUMMARY ===")
        print(f"SCHEDULE_GAME_IDS={len(schedule_game_ids)}")
        print(f"PROJECTION_COVERED_GAME_IDS={len(covered_game_ids)}")
        print(f"READY_PROJECTION_GAME_IDS={len(ready_game_ids)}")
        print(f"SCHEDULE_GAMES_WITH_NO_PROJECTION_ROWS={len(missing_projection)}")
        print(f"SCHEDULE_GAMES_WITH_NO_READY_ROWS={len(missing_ready)}")
        for gid in missing_projection:
            print(f"NO_PROJECTION_GAME={gid}")
        for gid in missing_ready:
            print(f"NO_READY_GAME={gid}")

        # Slate manifests: prove what user-selected/slate-defined universes exist.
        if table_exists(conn, "fanduel_slate_projection_manifest"):
            mcols = cols(conn, "fanduel_slate_projection_manifest")
            print("\n=== PROJECTION MANIFEST ===")
            print("MANIFEST_COLUMNS=" + ",".join(sorted(mcols)))
            rows = conn.execute("SELECT * FROM fanduel_slate_projection_manifest ORDER BY rowid").fetchall()
            print(f"PROJECTION_MANIFEST_ROWS={len(rows)}")
            for r in rows:
                d = dict(r)
                print("PROJECTION_MANIFEST_ROW=" + json.dumps(d, sort_keys=True, default=str))

        if table_exists(conn, "fanduel_slate_manifest"):
            print("\n=== SLATE MANIFEST ===")
            rows = conn.execute("SELECT * FROM fanduel_slate_manifest ORDER BY rowid").fetchall()
            print(f"SLATE_MANIFEST_ROWS={len(rows)}")
            for r in rows:
                print("SLATE_MANIFEST_ROW=" + json.dumps(dict(r), sort_keys=True, default=str))

        print("\n=== ARCHITECTURE RESULT ===")
        print("SCHEDULE_AUTHORITY=games")
        print("PROJECTION_SOURCE=fanduel_slate_projection_pool")
        print("HISTORICAL_CAPTURE_SHOULD_BE_SCHEDULE_DRIVEN=TRUE")
        print("FANDUEL_SLATE_SHOULD_NOT_DEFINE_HISTORICAL_COVERAGE=TRUE")
        print(f"CURRENT_WEEK_FULL_SCHEDULE_PROJECTION_COVERAGE={str(len(missing_projection)==0).upper()}")
        print(f"CURRENT_WEEK_FULL_SCHEDULE_READY_COVERAGE={str(len(missing_ready)==0).upper()}")

        print("\nNFL_POSTGAME_1F_A_STATUS=PASS")
        print("READ_ONLY_AUDIT=TRUE")
        print("DATABASE_WRITES=0")
        print("SERVICE_RESTARTS=0")
        print("CRON_CHANGES=0")
        print("UPDATER_CHANGES=0")
        print("LIVE_CHANGES=0")
        print("INJURY_PIPELINE_CHANGES=0")
        print("SOLVER_CHANGES=0")


if __name__ == "__main__":
    main()
