#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1D-D-A
Offensive player projection grader authority audit — READ ONLY.

Reads only:
  data/player_projection_ledger.db
  data/nfl.db

Purpose:
- Prove the immutable ledger can be joined to completed offensive player results.
- Select latest eligible projection snapshot strictly before kickoff.
- Exact game_id + exact GSIS player_id only.
- No fuzzy/name-only recovery.
- D/ST explicitly excluded.
- Produce grading metrics to stdout only.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"
LEDGER_DB = ROOT / "data" / "player_projection_ledger.db"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def parse_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise RuntimeError(f"naive UTC timestamp: {text}")
    return dt.astimezone(UTC)


def kickoff_utc(game_date, gametime, game_id: str) -> datetime:
    if game_date is None or gametime is None:
        raise RuntimeError(f"{game_id}: missing kickoff")
    raw = f"{str(game_date).strip()} {str(gametime).strip()}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=ET).astimezone(UTC)
        except ValueError:
            pass
    raise RuntimeError(f"{game_id}: invalid kickoff {raw!r}")


def finite(v, label: str) -> float:
    if v is None:
        raise RuntimeError(f"{label}: NULL")
    x = float(v)
    if not math.isfinite(x):
        raise RuntimeError(f"{label}: non-finite")
    return x


def main() -> int:
    print("=" * 72)
    print("WFS NFL — NFL-POSTGAME-1D-D-A")
    print("OFFENSIVE PLAYER PROJECTION GRADER AUTHORITY AUDIT — READ ONLY")
    print("=" * 72)

    with ro(NFL_DB) as nfl, ro(LEDGER_DB) as led:
        ni = nfl.execute("PRAGMA integrity_check").fetchone()[0]
        li = led.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={ni}")
        print(f"PLAYER_LEDGER_INTEGRITY={li}")
        if ni != "ok" or li != "ok":
            raise RuntimeError("database integrity failure")

        fk = led.execute("PRAGMA foreign_key_check").fetchall()
        print(f"PLAYER_LEDGER_FOREIGN_KEY_ERRORS={len(fk)}")
        if fk:
            raise RuntimeError("ledger foreign key failure")

        required_nfl = {"games", "player_game_stats"}
        nfl_tables = {r[0] for r in nfl.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        required_led = {"player_projection_snapshots", "player_projection_predictions"}
        led_tables = {r[0] for r in led.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not required_nfl <= nfl_tables:
            raise RuntimeError(f"missing nfl tables: {sorted(required_nfl-nfl_tables)}")
        if not required_led <= led_tables:
            raise RuntimeError(f"missing ledger tables: {sorted(required_led-led_tables)}")

        snapshots = led.execute("""
            SELECT snapshot_id, captured_at_utc, season, week,
                   row_count, offense_row_count, dst_row_count, snapshot_status
            FROM player_projection_snapshots
            ORDER BY captured_at_utc, snapshot_id
        """).fetchall()
        print(f"LEDGER_SNAPSHOTS={len(snapshots)}")
        for s in snapshots:
            print(
                "SNAPSHOT="
                f"{s['snapshot_id']}|captured={s['captured_at_utc']}|"
                f"season={s['season']}|week={s['week']}|"
                f"rows={s['row_count']}|offense={s['offense_row_count']}|"
                f"dst={s['dst_row_count']}|status={s['snapshot_status']}"
            )

        completed = nfl.execute("""
            SELECT game_id, season, week, game_date, gametime,
                   away_team, home_team, completed
            FROM games
            WHERE completed=1
            ORDER BY season, week, game_id
        """).fetchall()
        print(f"COMPLETED_GAMES={len(completed)}")

        grades = []
        completed_with_ledger = 0
        completed_without_eligible_snapshot = 0
        tie_failures = 0

        for g in completed:
            game_id = str(g["game_id"])
            ko = kickoff_utc(g["game_date"], g["gametime"], game_id)

            candidate_snaps = led.execute("""
                SELECT DISTINCT s.snapshot_id, s.captured_at_utc, s.snapshot_status
                FROM player_projection_snapshots s
                JOIN player_projection_predictions p
                  ON p.snapshot_id=s.snapshot_id
                WHERE p.game_id=?
                  AND p.is_dst=0
                  AND p.projection_status='READY'
                ORDER BY s.captured_at_utc
            """, (game_id,)).fetchall()

            eligible = [
                s for s in candidate_snaps
                if parse_utc(s["captured_at_utc"]) < ko
                and str(s["snapshot_status"]) == "PROSPECTIVE_CAPTURE"
            ]

            if not candidate_snaps:
                continue

            completed_with_ledger += 1

            if not eligible:
                completed_without_eligible_snapshot += 1
                print(f"GAME={game_id}|GRADE_STATUS=NO_ELIGIBLE_PREGAME_SNAPSHOT")
                continue

            latest_time = max(parse_utc(s["captured_at_utc"]) for s in eligible)
            latest = [
                s for s in eligible
                if parse_utc(s["captured_at_utc"]) == latest_time
            ]
            if len(latest) != 1:
                tie_failures += 1
                print(f"GAME={game_id}|GRADE_STATUS=TIED_LATEST_CAPTURE_FAIL_CLOSED")
                continue

            snap = latest[0]
            sid = str(snap["snapshot_id"])

            preds = led.execute("""
                SELECT game_id, slate_slug, player_id, player_name, team,
                       projection, salary, projection_source,
                       projection_match_method, optimizer_eligible
                FROM player_projection_predictions
                WHERE snapshot_id=?
                  AND game_id=?
                  AND is_dst=0
                  AND projection_status='READY'
                ORDER BY player_id, slate_slug
            """, (sid, game_id)).fetchall()

            # A player may exist on multiple FanDuel slates. For grading accuracy,
            # collapse only if every projection value for exact game/player agrees.
            grouped = defaultdict(list)
            for p in preds:
                grouped[str(p["player_id"])].append(p)

            canonical = []
            inconsistent = 0
            for pid, rows in grouped.items():
                values = {finite(r["projection"], f"{game_id}:{pid}:projection") for r in rows}
                if len(values) != 1:
                    inconsistent += 1
                    print(
                        f"GAME={game_id}|PLAYER_ID={pid}|"
                        "GRADE_STATUS=CROSS_SLATE_PROJECTION_CONFLICT"
                    )
                    continue
                canonical.append(rows[0])

            if inconsistent:
                raise RuntimeError(
                    f"{game_id}: {inconsistent} cross-slate projection conflicts"
                )

            actuals = nfl.execute("""
                SELECT game_id, player_id, player_display_name, team, position,
                       fanduel_points, fanduel_points_verified
                FROM player_game_stats
                WHERE game_id=?
            """, (game_id,)).fetchall()

            actual_by_id = defaultdict(list)
            for a in actuals:
                pid = str(a["player_id"] or "").strip()
                if pid:
                    actual_by_id[pid].append(a)

            joined = 0
            missing_actual = 0
            duplicate_actual = 0

            for p in canonical:
                pid = str(p["player_id"])
                hits = actual_by_id.get(pid, [])
                if len(hits) == 0:
                    missing_actual += 1
                    continue
                if len(hits) != 1:
                    duplicate_actual += 1
                    continue

                a = hits[0]
                if int(a["fanduel_points_verified"] or 0) != 1:
                    continue

                pred = finite(p["projection"], f"{game_id}:{pid}:projection")
                actual = finite(a["fanduel_points"], f"{game_id}:{pid}:actual")
                err = pred - actual

                grades.append({
                    "game_id": game_id,
                    "snapshot_id": sid,
                    "player_id": pid,
                    "player_name": str(p["player_name"]),
                    "team": str(p["team"]),
                    "projection": pred,
                    "actual": actual,
                    "error": err,
                    "absolute_error": abs(err),
                    "squared_error": err * err,
                })
                joined += 1

            print(
                f"GAME={game_id}|SNAPSHOT={sid}|"
                f"canonical_predictions={len(canonical)}|joined_verified={joined}|"
                f"missing_actual={missing_actual}|duplicate_actual={duplicate_actual}"
            )

        print()
        print("=== GRADING AUTHORITY RESULT ===")
        print(f"COMPLETED_GAMES_WITH_LEDGER_ROWS={completed_with_ledger}")
        print(f"COMPLETED_GAMES_WITHOUT_ELIGIBLE_PREGAME_SNAPSHOT={completed_without_eligible_snapshot}")
        print(f"LATEST_CAPTURE_TIE_FAILURES={tie_failures}")
        print(f"GRADED_OFFENSIVE_PLAYER_ROWS={len(grades)}")

        if grades:
            n = len(grades)
            mae = sum(x["absolute_error"] for x in grades) / n
            rmse = math.sqrt(sum(x["squared_error"] for x in grades) / n)
            bias = sum(x["error"] for x in grades) / n
            print(f"MAE={mae:.6f}")
            print(f"RMSE={rmse:.6f}")
            print(f"MEAN_ERROR_BIAS={bias:.6f}")
            print()
            print("=== SAMPLE GRADES ===")
            for x in sorted(grades, key=lambda z: (-z["absolute_error"], z["player_id"]))[:20]:
                print(
                    f"{x['game_id']}|{x['player_id']}|{x['player_name']}|{x['team']}|"
                    f"projection={x['projection']:.6f}|actual={x['actual']:.6f}|"
                    f"error={x['error']:.6f}|abs_error={x['absolute_error']:.6f}"
                )

        # A zero-grade result is valid at this stage if no completed game has a
        # prospective ledger snapshot. That is expected until a captured game finishes.
        if tie_failures:
            raise RuntimeError("latest eligible snapshot tie detected")

    print()
    print("NFL_POSTGAME_1D_D_A_STATUS=PASS")
    print("READ_ONLY_AUDIT=TRUE")
    print("PRODUCTION_DATABASE_WRITES=0")
    print("PLAYER_LEDGER_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
