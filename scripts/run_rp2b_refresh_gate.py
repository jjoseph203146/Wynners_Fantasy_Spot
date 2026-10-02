#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "data" / "nfl.db"
TARGET_DB = ROOT / "data" / "fanduel_player_role_performance.db"
LOADER = ROOT / "load_fanduel_offensive_performance.py"

STATE_DIR = ROOT / "data" / "pipeline_state"
STATE_FILE = STATE_DIR / "rp2b_source.sha256"

SOURCE_SQL = """
SELECT
    season,
    week,
    game_id,
    player_id,
    position,
    team,
    opponent_team,
    passing_yards,
    passing_tds,
    passing_interceptions,
    rushing_yards,
    rushing_tds,
    receptions,
    receiving_yards,
    receiving_tds,
    passing_2pt_conversions,
    rushing_2pt_conversions,
    receiving_2pt_conversions,
    special_teams_tds,
    total_fumbles_lost,
    carries,
    targets,
    target_share,
    receiving_air_yards,
    air_yards_share,
    wopr,
    fanduel_points,
    fanduel_points_verified,
    fanduel_scored_at
FROM player_game_stats
WHERE position IN ('QB','RB','WR','TE')
  AND game_id IS NOT NULL
  AND TRIM(game_id) <> ''
  AND player_id IS NOT NULL
  AND TRIM(player_id) <> ''
ORDER BY season, week, game_id, player_id
"""


def fail(message: str) -> int:
    print(f"RP2B_GATE_STATUS=FAIL")
    print(f"RP2B_GATE_ERROR={message}")
    return 1


def database_ok(path: Path) -> bool:
    try:
        conn = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            result = conn.execute(
                "PRAGMA quick_check"
            ).fetchone()
            return bool(
                result
                and str(result[0]).strip().lower() == "ok"
            )
        finally:
            conn.close()
    except Exception:
        return False


def source_fingerprint() -> tuple[str, int]:
    conn = sqlite3.connect(
        f"file:{SOURCE_DB.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        conn.execute("PRAGMA query_only = ON")
        rows = conn.execute(SOURCE_SQL).fetchall()
    finally:
        conn.close()

    digest = hashlib.sha256()

    for row in rows:
        payload = "\x1f".join(
            "<NULL>" if value is None else str(value)
            for value in row
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\x1e")

    return digest.hexdigest(), len(rows)


def target_source_row_count() -> int:
    conn = sqlite3.connect(
        f"file:{TARGET_DB.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        conn.execute("PRAGMA query_only = ON")
        return int(
            conn.execute(
                "SELECT COUNT(*) "
                "FROM fact_player_game_fanduel"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def main() -> int:
    print("RP2B GUARDED REFRESH")

    for path in (SOURCE_DB, TARGET_DB, LOADER):
        if not path.is_file():
            return fail(
                f"REQUIRED_PATH_MISSING:{path}"
            )

    if not database_ok(SOURCE_DB):
        return fail("SOURCE_DATABASE_QUICK_CHECK_FAILED")

    if not database_ok(TARGET_DB):
        return fail("TARGET_DATABASE_QUICK_CHECK_FAILED")

    try:
        fingerprint, source_rows = source_fingerprint()
        target_rows = target_source_row_count()
    except Exception as exc:
        return fail(f"PREFLIGHT_ERROR:{exc}")

    previous = ""
    if STATE_FILE.is_file():
        previous = STATE_FILE.read_text(
            encoding="utf-8"
        ).strip()

    print(f"SOURCE_ROWS={source_rows}")
    print(f"TARGET_FACT_ROWS={target_rows}")
    print(f"SOURCE_FINGERPRINT={fingerprint}")

    if (
        previous == fingerprint
        and target_rows == source_rows
    ):
        print("SOURCE_CHANGED=FALSE")
        print("RP2B_GATE_STATUS=SKIPPED")
        return 0

    print("SOURCE_CHANGED=TRUE")
    print("RP2B_ACTION=REBUILD")

    result = subprocess.run(
        [sys.executable, str(LOADER)],
        cwd=str(ROOT),
        check=False,
    )

    if result.returncode != 0:
        return fail(
            f"RP2B_LOADER_EXIT_{result.returncode}"
        )

    if not database_ok(TARGET_DB):
        return fail(
            "TARGET_DATABASE_QUICK_CHECK_FAILED_AFTER_REBUILD"
        )

    try:
        after_rows = target_source_row_count()
    except Exception as exc:
        return fail(
            f"POSTBUILD_COUNT_ERROR:{exc}"
        )

    if after_rows != source_rows:
        return fail(
            "SOURCE_TARGET_ROW_COUNT_MISMATCH:"
            f"{source_rows}:{after_rows}"
        )

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        fingerprint + "\n",
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)

    print(f"TARGET_FACT_ROWS_AFTER={after_rows}")
    print("SOURCE_TARGET_ROW_COUNT=PASS")
    print("DATABASE_QUICK_CHECK=PASS")
    print("RP2B_GATE_STATUS=UPDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
