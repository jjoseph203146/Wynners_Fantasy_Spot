#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1D-D-B
Production offensive player projection grader — READ ONLY.

Reads:
  data/player_projection_ledger.db
  data/nfl.db

Writes:
  none

Authority:
- completed games only
- exact game_id
- exact GSIS player_id
- latest eligible PROSPECTIVE_CAPTURE strictly before kickoff
- tied latest capture timestamp => fail closed
- D/ST excluded
- actual FanDuel points must be verified
- cross-slate duplicate projections collapse only when projection/salary identity agrees
- no fuzzy matching, no name-only recovery
- stdout-only deterministic grading report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"
LEDGER_DB = ROOT / "data" / "player_projection_ledger.db"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

ELIGIBLE_SNAPSHOT_STATUS = "PROSPECTIVE_CAPTURE"


def ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_capture_utc(text: Any) -> datetime:
    if text is None:
        raise RuntimeError("NULL captured_at_utc")
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise RuntimeError(f"naive captured_at_utc: {text}")
    return dt.astimezone(UTC)


def kickoff_utc(game_date: Any, gametime: Any, game_id: str) -> datetime:
    if game_date is None or gametime is None:
        raise RuntimeError(f"{game_id}: missing kickoff")

    raw = f"{str(game_date).strip()} {str(gametime).strip()}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ET).astimezone(UTC)
        except ValueError:
            continue

    raise RuntimeError(f"{game_id}: invalid kickoff {raw!r}")


def finite(value: Any, label: str) -> float:
    if value is None:
        raise RuntimeError(f"{label}: NULL")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: invalid numeric {value!r}") from exc
    if not math.isfinite(x):
        raise RuntimeError(f"{label}: non-finite numeric {value!r}")
    return x


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_week_games(conn, season, week):
    """Exact REG week authority; never hide unfinished games by filtering them out."""
    rows = conn.execute(
        """
        SELECT game_id, season, week, game_date, gametime,
               away_team, home_team, away_score, home_score, completed
        FROM games
        WHERE season=? AND game_type='REG' AND week=?
        ORDER BY game_id
        """,
        (season, week),
    ).fetchall()
    ids = [row["game_id"] for row in rows]
    if (
        not rows
        or any(not isinstance(gid, str) or not gid.strip() for gid in ids)
        or len(set(ids)) != len(ids)
        or any(
            row["completed"] != 1
            or row["away_score"] is None
            or row["home_score"] is None
            for row in rows
        )
    ):
        raise RuntimeError(f"Season {season} week {week}: incomplete or invalid REG game set")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    args = parser.parse_args()
    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be supplied together")
    if args.week is not None and (args.season < 1 or args.week < 1):
        parser.error("--season and --week must be positive")

    print("=" * 76)
    print("WFS NFL — NFL-POSTGAME-1D-D-B")
    print("PRODUCTION OFFENSIVE PLAYER PROJECTION GRADER — READ ONLY")
    print("=" * 76)

    with ro(NFL_DB) as nfl, ro(LEDGER_DB) as led:
        nfl_integrity = nfl.execute("PRAGMA integrity_check").fetchone()[0]
        ledger_integrity = led.execute("PRAGMA integrity_check").fetchone()[0]
        ledger_fk = led.execute("PRAGMA foreign_key_check").fetchall()

        print(f"NFL_DB_INTEGRITY={nfl_integrity}")
        print(f"PLAYER_LEDGER_INTEGRITY={ledger_integrity}")
        print(f"PLAYER_LEDGER_FOREIGN_KEY_ERRORS={len(ledger_fk)}")

        if nfl_integrity != "ok":
            raise RuntimeError("nfl.db integrity failure")
        if ledger_integrity != "ok":
            raise RuntimeError("player ledger integrity failure")
        if ledger_fk:
            raise RuntimeError("player ledger foreign key failure")

        if args.week is None:
            completed = nfl.execute(
                """
                SELECT
                    game_id,
                    season,
                    week,
                    game_date,
                    gametime,
                    away_team,
                    home_team,
                    away_score,
                    home_score,
                    completed
                FROM games
                WHERE completed=1
                  AND away_score IS NOT NULL
                  AND home_score IS NOT NULL
                ORDER BY season, week, game_id
                """
            ).fetchall()
        else:
            completed = select_week_games(nfl, args.season, args.week)
            print("WEEK_SCOPE=" + json.dumps({
                "season": args.season,
                "week": args.week,
                "game_ids": [row["game_id"] for row in completed],
            }, sort_keys=True, separators=(",", ":")))

        print(f"COMPLETED_GAMES_WITH_FINAL_SCORE={len(completed)}")

        ledger_game_ids = {
            str(r[0])
            for r in led.execute(
                """
                SELECT DISTINCT game_id
                FROM player_projection_predictions
                WHERE is_dst=0
                  AND projection_status='READY'
                """
            )
        }
        print(f"LEDGER_OFFENSIVE_GAME_IDS={len(ledger_game_ids)}")

        grades: list[dict[str, Any]] = []
        graded_games = 0
        games_with_ledger = 0
        no_eligible_snapshot = 0
        tie_failures = 0
        cross_slate_conflicts = 0
        missing_actual_total = 0
        duplicate_actual_total = 0
        unverified_actual_total = 0

        for g in completed:
            game_id = str(g["game_id"])
            if game_id not in ledger_game_ids:
                continue

            games_with_ledger += 1
            ko = kickoff_utc(g["game_date"], g["gametime"], game_id)

            candidates = led.execute(
                """
                SELECT DISTINCT
                    s.snapshot_id,
                    s.captured_at_utc,
                    s.snapshot_status
                FROM player_projection_snapshots s
                JOIN player_projection_predictions p
                  ON p.snapshot_id=s.snapshot_id
                WHERE p.game_id=?
                  AND p.is_dst=0
                  AND p.projection_status='READY'
                ORDER BY s.captured_at_utc, s.snapshot_id
                """,
                (game_id,),
            ).fetchall()

            eligible = [
                s
                for s in candidates
                if str(s["snapshot_status"]) == ELIGIBLE_SNAPSHOT_STATUS
                and parse_capture_utc(s["captured_at_utc"]) < ko
            ]

            if not eligible:
                no_eligible_snapshot += 1
                print(f"GAME={game_id}|STATUS=NO_ELIGIBLE_PREGAME_SNAPSHOT")
                continue

            latest_capture = max(parse_capture_utc(s["captured_at_utc"]) for s in eligible)
            latest = [
                s
                for s in eligible
                if parse_capture_utc(s["captured_at_utc"]) == latest_capture
            ]

            if len(latest) != 1:
                tie_failures += 1
                print(
                    f"GAME={game_id}|STATUS=TIED_LATEST_CAPTURE_FAIL_CLOSED|"
                    f"count={len(latest)}|captured_at={latest_capture.isoformat()}"
                )
                continue

            snapshot = latest[0]
            snapshot_id = str(snapshot["snapshot_id"])

            preds = led.execute(
                """
                SELECT
                    game_id,
                    slate_slug,
                    player_id,
                    player_name,
                    team,
                    opponent_team,
                    salary,
                    projection,
                    projection_source,
                    projection_match_method,
                    projection_status,
                    optimizer_eligible
                FROM player_projection_predictions
                WHERE snapshot_id=?
                  AND game_id=?
                  AND is_dst=0
                  AND projection_status='READY'
                ORDER BY player_id, slate_slug, player_name, team
                """,
                (snapshot_id, game_id),
            ).fetchall()

            grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for p in preds:
                pid = str(p["player_id"] or "").strip()
                if not pid:
                    raise RuntimeError(f"{game_id}: blank ledger player_id")
                grouped[pid].append(p)

            canonical_preds: list[sqlite3.Row] = []

            for pid, rows in sorted(grouped.items()):
                signature = {
                    (
                        finite(r["projection"], f"{game_id}:{pid}:projection"),
                        finite(r["salary"], f"{game_id}:{pid}:salary"),
                        str(r["player_name"]),
                        str(r["team"]),
                        str(r["opponent_team"]),
                        str(r["projection_source"]),
                        str(r["projection_match_method"]),
                        int(r["optimizer_eligible"]),
                    )
                    for r in rows
                }

                if len(signature) != 1:
                    cross_slate_conflicts += 1
                    print(
                        f"GAME={game_id}|PLAYER_ID={pid}|"
                        "STATUS=CROSS_SLATE_CONFLICT_FAIL_CLOSED"
                    )
                    continue

                canonical_preds.append(rows[0])

            if cross_slate_conflicts:
                raise RuntimeError(
                    "cross-slate projection conflict detected; grading failed closed"
                )

            actual_rows = nfl.execute(
                """
                SELECT
                    game_id,
                    player_id,
                    player_display_name,
                    team,
                    position,
                    fanduel_points,
                    fanduel_points_verified
                FROM player_game_stats
                WHERE game_id=?
                ORDER BY player_id, team
                """,
                (game_id,),
            ).fetchall()

            actual_by_id: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for a in actual_rows:
                pid = str(a["player_id"] or "").strip()
                if pid:
                    actual_by_id[pid].append(a)

            game_grade_rows = 0
            game_missing = 0
            game_dupe = 0
            game_unverified = 0

            for p in canonical_preds:
                pid = str(p["player_id"])
                hits = actual_by_id.get(pid, [])

                if len(hits) == 0:
                    game_missing += 1
                    continue
                if len(hits) != 1:
                    game_dupe += 1
                    continue

                a = hits[0]

                if int(a["fanduel_points_verified"] or 0) != 1:
                    game_unverified += 1
                    continue

                pred = finite(p["projection"], f"{game_id}:{pid}:projection")
                actual = finite(a["fanduel_points"], f"{game_id}:{pid}:actual")
                error = pred - actual

                grades.append(
                    {
                        "game_id": game_id,
                        "season": int(g["season"]),
                        "week": int(g["week"]),
                        "snapshot_id": snapshot_id,
                        "captured_at_utc": str(snapshot["captured_at_utc"]),
                        "player_id": pid,
                        "player_name": str(p["player_name"]),
                        "team": str(p["team"]),
                        "opponent_team": str(p["opponent_team"]),
                        "salary": finite(p["salary"], f"{game_id}:{pid}:salary"),
                        "projection": pred,
                        "actual": actual,
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error * error,
                    }
                )
                game_grade_rows += 1

            missing_actual_total += game_missing
            duplicate_actual_total += game_dupe
            unverified_actual_total += game_unverified

            if game_grade_rows > 0:
                graded_games += 1

            print(
                f"GAME={game_id}|"
                f"SNAPSHOT={snapshot_id}|"
                f"captured={snapshot['captured_at_utc']}|"
                f"canonical_predictions={len(canonical_preds)}|"
                f"graded={game_grade_rows}|"
                f"missing_actual={game_missing}|"
                f"duplicate_actual={game_dupe}|"
                f"unverified_actual={game_unverified}"
            )

        print()
        print("=== PRODUCTION GRADING RESULT ===")
        print(f"COMPLETED_GAMES_WITH_LEDGER_ROWS={games_with_ledger}")
        print(f"GRADED_GAMES={graded_games}")
        print(f"NO_ELIGIBLE_PREGAME_SNAPSHOT_GAMES={no_eligible_snapshot}")
        print(f"LATEST_CAPTURE_TIE_FAILURES={tie_failures}")
        print(f"CROSS_SLATE_CONFLICTS={cross_slate_conflicts}")
        print(f"MISSING_ACTUAL_ROWS={missing_actual_total}")
        print(f"DUPLICATE_ACTUAL_ROWS={duplicate_actual_total}")
        print(f"UNVERIFIED_ACTUAL_ROWS={unverified_actual_total}")
        print(f"GRADED_OFFENSIVE_PLAYER_ROWS={len(grades)}")

        if tie_failures:
            raise RuntimeError("latest snapshot tie detected")
        if cross_slate_conflicts:
            raise RuntimeError("cross-slate conflicts detected")

        if grades:
            n = len(grades)
            mae = sum(r["absolute_error"] for r in grades) / n
            rmse = math.sqrt(sum(r["squared_error"] for r in grades) / n)
            bias = sum(r["error"] for r in grades) / n

            print(f"MAE={mae:.6f}")
            print(f"RMSE={rmse:.6f}")
            print(f"MEAN_ERROR_BIAS={bias:.6f}")

            by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in grades:
                by_game[r["game_id"]].append(r)

            print()
            print("=== GAME METRICS ===")
            for game_id in sorted(by_game):
                rows = by_game[game_id]
                gn = len(rows)
                gmae = sum(r["absolute_error"] for r in rows) / gn
                grmse = math.sqrt(sum(r["squared_error"] for r in rows) / gn)
                gbias = sum(r["error"] for r in rows) / gn
                print(
                    f"{game_id}|rows={gn}|"
                    f"mae={gmae:.6f}|rmse={grmse:.6f}|bias={gbias:.6f}"
                )

            print()
            print("=== LARGEST ABSOLUTE ERRORS ===")
            for r in sorted(
                grades,
                key=lambda x: (-x["absolute_error"], x["game_id"], x["player_id"]),
            )[:25]:
                print(
                    f"{r['game_id']}|{r['player_id']}|{r['player_name']}|{r['team']}|"
                    f"projection={r['projection']:.6f}|actual={r['actual']:.6f}|"
                    f"error={r['error']:.6f}|abs_error={r['absolute_error']:.6f}"
                )

        semantic_rows = [
            {
                "game_id": r["game_id"],
                "snapshot_id": r["snapshot_id"],
                "player_id": r["player_id"],
                "projection": r["projection"],
                "actual": r["actual"],
                "error": r["error"],
                "absolute_error": r["absolute_error"],
            }
            for r in sorted(
                grades,
                key=lambda x: (x["game_id"], x["snapshot_id"], x["player_id"]),
            )
        ]
        semantic_sha = canonical_sha(semantic_rows)
        print(f"SEMANTIC_SHA256={semantic_sha}")

    print()
    print("NFL_POSTGAME_1D_D_B_STATUS=PASS")
    print("READ_ONLY=TRUE")
    print("PRODUCTION_DATABASE_WRITES=0")
    print("PLAYER_LEDGER_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
