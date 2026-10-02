#!/usr/bin/env python3

from pathlib import Path
import hashlib
import sqlite3
import subprocess
import sys

ROOT = Path("/home/mwynn/nfl_data_engine")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wfs_schedule_context import resolve_schedule_week_context


PYTHON = ROOT / "venv/bin/python"
DB = ROOT / "data/nfl.db"

EXPECTED_PRODUCTION_SHA = (
    "99490a94c83075ad96d32e19c4ebfa1168bd318f0848e8c67b068c4b79e91915"
)

STAGES = [
    ("PLAYER_USAGE", ROOT / "player_usage.py"),
    ("PREGAME_FEATURES", ROOT / "pregame_features.py"),
    ("TEAM_ENVIRONMENT", ROOT / "team_environment.py"),
    ("FEATURE_MATRIX", ROOT / "feature_matrix.py"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def db_integrity() -> str:
    with sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        return str(
            conn.execute("PRAGMA integrity_check").fetchone()[0]
        )


def completed_preplanning_games(
    season: int,
    planning_week: int,
) -> set[str]:
    with sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        rows = conn.execute(
            """
            SELECT game_id
            FROM games
            WHERE season = ?
              AND game_type = 'REG'
              AND week < ?
              AND completed = 1
              AND game_id IS NOT NULL
              AND TRIM(game_id) != ''
            """,
            (season, planning_week),
        ).fetchall()

    return {str(row[0]).strip() for row in rows}


def usage_preplanning_games(
    season: int,
    planning_week: int,
) -> set[str]:
    with sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT game_id
            FROM player_weekly_usage
            WHERE season = ?
              AND week < ?
              AND UPPER(COALESCE(season_type, '')) = 'REG'
              AND game_id IS NOT NULL
              AND TRIM(game_id) != ''
            """,
            (season, planning_week),
        ).fetchall()

    return {str(row[0]).strip() for row in rows}


def verify_coverage(
    season: int,
    planning_week: int,
) -> None:
    expected = completed_preplanning_games(
        season,
        planning_week,
    )
    actual = usage_preplanning_games(
        season,
        planning_week,
    )

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    print(f"EXPECTED_COMPLETED_GAME_COUNT={len(expected)}")
    print(f"USAGE_GAME_COUNT={len(actual)}")
    print(f"MISSING_COMPLETED_GAME_COUNT={len(missing)}")
    print(f"EXTRA_USAGE_GAME_COUNT={len(extra)}")

    if missing:
        print("MISSING_GAME_IDS=" + ",".join(missing))

    if extra:
        print("EXTRA_GAME_IDS=" + ",".join(extra))

    if missing or extra:
        raise RuntimeError(
            "FAIL_CLOSED: historical usage game-ID coverage "
            "does not exactly match completed pre-planning "
            "regular-season schedule authority"
        )

    print("HISTORICAL_GAME_ID_COVERAGE=PASS")


def run_stage(name: str, script: Path) -> None:
    if not script.is_file():
        raise RuntimeError(
            f"FAIL_CLOSED: missing historical stage {script}"
        )

    print()
    print("=" * 78)
    print(f"HISTORICAL STAGE {name}")
    print("=" * 78)

    result = subprocess.run(
        [str(PYTHON), str(script)],
        cwd=str(ROOT),
    )

    print(f"{name}_RETURN_CODE={result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(
            f"FAIL_CLOSED: historical stage {name} failed"
        )

    integrity = db_integrity()
    print(f"{name}_POST_DB_INTEGRITY={integrity}")

    if integrity != "ok":
        raise RuntimeError(
            f"FAIL_CLOSED: DB integrity failed after {name}"
        )


def main() -> None:
    print("=== HISTORICAL ROLLOVER GATE V1 ===")

    if not DB.is_file():
        raise RuntimeError("FAIL_CLOSED: nfl.db missing")

    production_projection = ROOT / "production_projection.py"

    actual_sha = sha256(production_projection)
    print(f"PRODUCTION_PROJECTION_SHA256={actual_sha}")

    if actual_sha != EXPECTED_PRODUCTION_SHA:
        raise RuntimeError(
            "FAIL_CLOSED: production_projection.py does not "
            "match frozen rolling-feedback baseline"
        )

    context = resolve_schedule_week_context(
        db_path=DB,
    )

    season = int(context.season)
    planning_week = int(context.planning_week)

    print(f"SEASON={season}")
    print(f"PLANNING_WEEK={planning_week}")
    print(
        "ACTIVE_GAME_WEEK="
        + (
            str(context.active_game_week)
            if context.active_game_week is not None
            else "NONE"
        )
    )
    print(
        "UPCOMING_WEEK="
        + (
            str(context.upcoming_week)
            if context.upcoming_week is not None
            else "NONE"
        )
    )

    if planning_week < 1:
        raise RuntimeError(
            "FAIL_CLOSED: invalid planning week"
        )

    integrity = db_integrity()
    print(f"PRE_DB_INTEGRITY={integrity}")

    if integrity != "ok":
        raise RuntimeError(
            "FAIL_CLOSED: pre-run DB integrity failed"
        )

    for name, script in STAGES:
        run_stage(name, script)

    verify_coverage(
        season,
        planning_week,
    )

    integrity = db_integrity()
    print(f"FINAL_DB_INTEGRITY={integrity}")

    if integrity != "ok":
        raise RuntimeError(
            "FAIL_CLOSED: final DB integrity failed"
        )

    print("HISTORICAL_ROLLOVER_GATE_STATUS=PASS")


if __name__ == "__main__":
    main()
