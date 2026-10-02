#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1D-C-R1
READ-ONLY schedule disambiguation audit for player projection ledger capture.

Purpose:
- Explain/fix the 1D-C discovery that away_team/home_team alone is not globally unique.
- Determine whether the current projection slate maps uniquely to one season/week.
- No database writes.
- No service, cron, updater, solver, LIVE, injury, or projection changes.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"


def ro_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    print("=" * 70)
    print("WFS NFL — NFL-POSTGAME-1D-C-R1")
    print("SCHEDULE DISAMBIGUATION AUDIT — READ ONLY")
    print("=" * 70)

    with ro_connect(NFL_DB) as c:
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity check failed")

        proj_games = c.execute("""
            SELECT DISTINCT
                away_team,
                home_team,
                game_key
            FROM fanduel_slate_projection_pool
            ORDER BY away_team, home_team
        """).fetchall()

        print(f"DISTINCT_PROJECTION_GAMES={len(proj_games)}")

        proj_keys = {
            (str(r["away_team"]).strip(), str(r["home_team"]).strip())
            for r in proj_games
        }

        if len(proj_keys) != len(proj_games):
            raise RuntimeError("duplicate distinct projection matchup keys")

        schedule = c.execute("""
            SELECT
                game_id,
                season,
                week,
                away_team,
                home_team,
                game_date,
                gametime,
                completed
            FROM games
            WHERE season IS NOT NULL
              AND week IS NOT NULL
            ORDER BY season, week, game_id
        """).fetchall()

        grouped = defaultdict(list)
        for g in schedule:
            grouped[(int(g["season"]), int(g["week"]))].append(g)

        candidates = []

        for season_week, games in sorted(grouped.items()):
            by_key = defaultdict(list)
            for g in games:
                key = (
                    str(g["away_team"]).strip(),
                    str(g["home_team"]).strip(),
                )
                by_key[key].append(g)

            matched = 0
            ambiguous = 0
            missing = 0

            for key in proj_keys:
                hits = by_key.get(key, [])
                if len(hits) == 1:
                    matched += 1
                elif len(hits) == 0:
                    missing += 1
                else:
                    ambiguous += 1

            if matched:
                candidates.append(
                    (season_week, matched, missing, ambiguous)
                )

        print()
        print("=== SEASON/WEEK COVERAGE CANDIDATES ===")
        for (season, week), matched, missing, ambiguous in sorted(
            candidates,
            key=lambda x: (-x[1], x[0][0], x[0][1])
        )[:50]:
            print(
                f"season={season}|week={week}|"
                f"matched={matched}|missing={missing}|ambiguous={ambiguous}"
            )

        full = [
            (sw, m, miss, amb)
            for sw, m, miss, amb in candidates
            if m == len(proj_keys) and miss == 0 and amb == 0
        ]

        print()
        print(f"FULL_COVERAGE_CANDIDATES={len(full)}")

        if len(full) != 1:
            for sw, m, miss, amb in full:
                print(
                    f"FULL_CANDIDATE=season={sw[0]}|week={sw[1]}|"
                    f"matched={m}|missing={miss}|ambiguous={amb}"
                )
            raise RuntimeError(
                "current projection slate does not resolve to exactly one season/week"
            )

        (season, week), _, _, _ = full[0]
        print(f"RESOLVED_SEASON={season}")
        print(f"RESOLVED_WEEK={week}")

        by_key = defaultdict(list)
        rows = c.execute("""
            SELECT
                game_id,
                season,
                week,
                away_team,
                home_team,
                game_date,
                gametime,
                completed
            FROM games
            WHERE season=? AND week=?
            ORDER BY game_id
        """, (season, week)).fetchall()

        for g in rows:
            by_key[
                (
                    str(g["away_team"]).strip(),
                    str(g["home_team"]).strip(),
                )
            ].append(g)

        print()
        print("=== EXACT CURRENT SLATE GAME MAP ===")
        for key in sorted(proj_keys):
            hits = by_key[key]
            if len(hits) != 1:
                raise RuntimeError(f"{key}: expected exactly one game, got {len(hits)}")
            g = hits[0]
            print(
                f"{key[0]}@{key[1]}|"
                f"game_id={g['game_id']}|"
                f"date={g['game_date']}|time={g['gametime']}|"
                f"completed={g['completed']}"
            )

        player_rows = c.execute("""
            SELECT away_team, home_team, COUNT(*) n
            FROM fanduel_slate_projection_pool
            GROUP BY away_team, home_team
            ORDER BY away_team, home_team
        """).fetchall()

        total_mapped = 0
        for r in player_rows:
            key = (
                str(r["away_team"]).strip(),
                str(r["home_team"]).strip(),
            )
            if len(by_key.get(key, [])) != 1:
                raise RuntimeError(f"row-level map failed for {key}")
            total_mapped += int(r["n"])

        print()
        print(f"PROJECTION_ROWS_EXACTLY_MAPPED={total_mapped}")
        print("PROJECTION_ROWS_UNMAPPED=0")
        print("GLOBAL_MATCHUP_ONLY_MAPPING_SAFE=FALSE")
        print("SEASON_WEEK_SCOPED_MAPPING_SAFE=TRUE")

    print()
    print("NFL_POSTGAME_1D_C_R1_STATUS=PASS")
    print("READ_ONLY_AUDIT=TRUE")
    print("PRODUCTION_DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
