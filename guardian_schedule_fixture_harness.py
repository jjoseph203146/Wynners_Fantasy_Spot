#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tempfile
from pathlib import Path

from guardian_schedule import derive_schedule_authority
from guardian_classification import classify_failures


ROOT = Path(__file__).resolve().parent
PROD_DB = ROOT / "data" / "nfl.db"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def run_schedule(root: Path):
    results = []

    def record(name, status, detail):
        results.append(
            {
                "check": name,
                "status": status,
                "detail": detail,
            }
        )

    schedule = derive_schedule_authority(
        root,
        record,
        today_utc="2026-09-20",
    )

    return schedule, results


def show(label, schedule, results):
    print("-" * 72)
    print(label)

    for row in results:
        print(
            f"  {row['check']:<30} "
            f"{row['status']:<5} "
            f"{row['detail']}"
        )

    print("  SCHEDULE =", schedule)


def main() -> int:
    print("=" * 72)
    print(
        "WFS GUARDIAN V2.6 PHASE 4B — "
        "SCHEDULE AUTHORITY FIXTURE TEST"
    )
    print("=" * 72)

    if not PROD_DB.is_file():
        raise SystemExit(
            f"FAIL: production DB missing: {PROD_DB}"
        )

    before = sha256(PROD_DB)

    print(
        "PRODUCTION_DB_SHA_BEFORE=",
        before,
    )

    with tempfile.TemporaryDirectory(
        prefix="wfs_guardian_schedule_"
    ) as tmp:
        fixture_root = Path(tmp)
        fixture_data = fixture_root / "data"
        fixture_data.mkdir(
            parents=True,
            exist_ok=True,
        )

        fixture_db = fixture_data / "nfl.db"

        # SQLite-safe copy of production into fixture.
        src = sqlite3.connect(
            f"file:{PROD_DB.resolve()}?mode=ro",
            uri=True,
        )

        dst = sqlite3.connect(fixture_db)

        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        # --------------------------------------------------------
        # CONTROL
        # --------------------------------------------------------

        schedule, results = run_schedule(
            fixture_root
        )

        show(
            "CONTROL — CLEAN DATABASE COPY",
            schedule,
            results,
        )

        if schedule is None:
            raise SystemExit(
                "FAIL: clean fixture schedule unresolved"
            )

        if (
            schedule["season"] != 2026
            or schedule["week"] != 2
            or schedule["unfinished_games"] != 15
            or schedule["unfinished_teams"] != 30
        ):
            raise SystemExit(
                "FAIL: clean fixture schedule differs "
                "from production authority"
            )

        if any(
            row["status"] == "FAIL"
            for row in results
        ):
            raise SystemExit(
                "FAIL: clean fixture produced failure"
            )

        print("CONTROL_SCHEDULE=PASS")

        # --------------------------------------------------------
        # TEST 1:
        # Make Week 3 eligible TODAY in fixture only.
        # Week 2 is still unfinished, so Week 3 MUST NOT advance.
        # --------------------------------------------------------

        con = sqlite3.connect(fixture_db)

        try:
            changed = con.execute(
                """
                UPDATE games
                SET game_date = '2026-09-20'
                WHERE season = 2026
                  AND week = 3
                  AND UPPER(COALESCE(game_type, '')) = 'REG'
                """
            ).rowcount

            con.commit()
        finally:
            con.close()

        print(
            "FIXTURE_WEEK3_ROWS_REDATED=",
            changed,
        )

        schedule, results = run_schedule(
            fixture_root
        )

        show(
            "FAULT TEST 1 — FUTURE/STAGED WEEK ELIGIBLE",
            schedule,
            results,
        )

        if schedule is None:
            raise SystemExit(
                "FAIL: staged-week test unexpectedly "
                "destroyed schedule authority"
            )

        if (
            schedule["season"] != 2026
            or schedule["week"] != 2
        ):
            raise SystemExit(
                "FAIL: staged Week 3 advanced production "
                f"to {schedule['season']} Week "
                f"{schedule['week']}"
            )

        print(
            "STAGED_WEEK_ADVANCE_BLOCKED=PASS"
        )

        # --------------------------------------------------------
        # Restore fixture from production for independent test 2.
        # --------------------------------------------------------

        fixture_db.unlink()

        src = sqlite3.connect(
            f"file:{PROD_DB.resolve()}?mode=ro",
            uri=True,
        )

        dst = sqlite3.connect(fixture_db)

        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        # --------------------------------------------------------
        # TEST 2:
        # Force duplicate team topology among unfinished Week 2
        # games. This must violate 2 unique teams per game.
        # --------------------------------------------------------

        con = sqlite3.connect(fixture_db)
        con.row_factory = sqlite3.Row

        try:
            rows = con.execute(
                """
                SELECT
                    game_id,
                    away_team,
                    home_team
                FROM games
                WHERE season = 2026
                  AND week = 2
                  AND UPPER(COALESCE(game_type, '')) = 'REG'
                  AND COALESCE(completed, 0) = 0
                ORDER BY game_id
                LIMIT 2
                """
            ).fetchall()

            if len(rows) != 2:
                raise SystemExit(
                    "FAIL: need two unfinished Week 2 games"
                )

            shared_team = str(
                rows[0]["away_team"]
            )

            target_game = str(
                rows[1]["game_id"]
            )

            con.execute(
                """
                UPDATE games
                SET away_team = ?
                WHERE game_id = ?
                """,
                (
                    shared_team,
                    target_game,
                ),
            )

            con.commit()

        finally:
            con.close()

        schedule, results = run_schedule(
            fixture_root
        )

        show(
            "FAULT TEST 2 — DUPLICATE ACTIVE TEAM",
            schedule,
            results,
        )

        if schedule is not None:
            raise SystemExit(
                "FAIL: invalid team topology was accepted"
            )

        failed = [
            row
            for row in results
            if row["status"] == "FAIL"
        ]

        if not failed:
            raise SystemExit(
                "FAIL: schedule fault produced no failure"
            )

        classification = classify_failures(
            results
        )

        print(
            "FAILURE_CATEGORIES=",
            classification["categories"],
        )

        if "SCHEDULE" not in classification["categories"]:
            raise SystemExit(
                "FAIL: schedule fault not classified "
                "as SCHEDULE"
            )

        print(
            "INVALID_TEAM_TOPOLOGY_DETECTED=PASS"
        )

    after = sha256(PROD_DB)

    print("-" * 72)
    print(
        "PRODUCTION_DB_SHA_AFTER=",
        after,
    )

    unchanged = before == after

    print(
        "PRODUCTION_DB_UNCHANGED=",
        unchanged,
    )

    if not unchanged:
        raise SystemExit(
            "FAIL: production nfl.db changed"
        )

    print("PRODUCTION_LKG_MUTATED=FALSE")
    print("PRODUCTION_AUDIT_MUTATED=FALSE")
    print("PRODUCTION_ACTION=NONE")
    print("ACTUAL_SCHEDULE_RESOLVER_USED=TRUE")
    print("V26_SCHEDULE_FIXTURE_TEST=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
