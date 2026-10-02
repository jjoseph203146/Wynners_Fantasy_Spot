#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import sqlite3
import subprocess
import sys
import time

ROOT = Path("/home/mwynn/nfl_data_engine")
PYTHON = ROOT / "venv/bin/python"
DB = ROOT / "data/nfl.db"
FANDUEL_INVENTORY_STATE = (
    ROOT / "data/csv/fanduel_solver_inventory_state.json"
)

EXPECTED_PRODUCTION_SHA = (
    "99490a94c83075ad96d32e19c4ebfa1168bd318f0848e8c67b068c4b79e91915"
)

STAGES = [
    ("PLAYER_USAGE", ROOT / "player_usage.py"),
    ("PREGAME_FEATURES", ROOT / "pregame_features.py"),
    ("TEAM_ENVIRONMENT", ROOT / "team_environment.py"),
    ("FEATURE_MATRIX", ROOT / "feature_matrix.py"),
    ("CURRENT_SLATE_FEATURES", ROOT / "current_slate_features.py"),
    ("PRODUCTION_PROJECTION", ROOT / "production_projection.py"),
    ("FANDUEL_PLAYER_POOL", ROOT / "fanduel_player_pool.py"),
    ("DST_CURRENT_FEATURES", ROOT / "dst_current_features.py"),
    ("DST_PRODUCTION_PROJECTION", ROOT / "dst_production_projection.py"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def db_integrity():
    conn = sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True,
    )

    try:
        return conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
    finally:
        conn.close()


def table_count(table, where="1=1"):
    conn = sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True,
    )

    try:
        return conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE {where}
            """
        ).fetchone()[0]
    finally:
        conn.close()


def fanduel_lifecycle_state():
    if not FANDUEL_INVENTORY_STATE.exists():
        raise RuntimeError(
            "FAIL_CLOSED: FanDuel lifecycle state missing"
        )

    try:
        state = json.loads(
            FANDUEL_INVENTORY_STATE.read_text()
        )
    except Exception as exc:
        raise RuntimeError(
            "FAIL_CLOSED: FanDuel lifecycle state unreadable"
        ) from exc

    status = state.get("status")
    rows = state.get("solver_open_rows")
    authority = state.get("authority")

    if authority != "fanduel_slate_pool":
        raise RuntimeError(
            "FAIL_CLOSED: unexpected FanDuel lifecycle authority"
        )

    if status == "NO_OPEN_SLATES":
        if rows != 0:
            raise RuntimeError(
                "FAIL_CLOSED: contradictory NO_OPEN_SLATES state"
            )
        return status

    if status == "OPEN_SLATES_PRESENT":
        if not isinstance(rows, int) or rows <= 0:
            raise RuntimeError(
                "FAIL_CLOSED: contradictory OPEN_SLATES_PRESENT state"
            )
        return status

    raise RuntimeError(
        f"FAIL_CLOSED: unexpected FanDuel lifecycle status {status!r}"
    )


def run_stage(name, script):
    if not script.exists():
        raise RuntimeError(
            f"FAIL_CLOSED: missing stage {script}"
        )

    print()
    print("=" * 78)
    print(f"STAGE {name}")
    print("=" * 78)

    started = time.time()

    result = subprocess.run(
        [
            str(PYTHON),
            str(script),
        ],
        cwd=str(ROOT),
    )

    elapsed = time.time() - started

    print()
    print(
        f"{name}_RETURN_CODE={result.returncode}"
    )
    print(
        f"{name}_SECONDS={elapsed:.1f}"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FAIL_CLOSED: {name} failed"
        )


def main():
    print(
        "=== NFL-2026-FEEDBACK-1Q "
        "MANUAL FEEDBACK ORCHESTRATOR ==="
    )

    if not DB.exists():
        raise RuntimeError(
            "FAIL_CLOSED: nfl.db missing"
        )

    prod = ROOT / "production_projection.py"

    actual_sha = sha256(prod)

    print(
        "PRODUCTION_PROJECTION_SHA256="
        + actual_sha
    )

    if actual_sha != EXPECTED_PRODUCTION_SHA:
        raise RuntimeError(
            "FAIL_CLOSED: production_projection.py "
            "does not match frozen rolling-feedback baseline"
        )

    before_integrity = db_integrity()

    print(
        "PRE_DB_INTEGRITY="
        + before_integrity
    )

    if before_integrity != "ok":
        raise RuntimeError(
            "FAIL_CLOSED: pre-run DB integrity failed"
        )

    fd_state = fanduel_lifecycle_state()
    print(
        "FANDUEL_LIFECYCLE_STATE="
        + fd_state
    )

    for name, script in STAGES:
        if (
            name == "FANDUEL_PLAYER_POOL"
            and fd_state == "NO_OPEN_SLATES"
        ):
            print()
            print("=" * 78)
            print("STAGE FANDUEL_PLAYER_POOL")
            print("=" * 78)
            print(
                "FANDUEL_PLAYER_POOL_SKIPPED="
                "NO_OPEN_SLATES"
            )
            continue

        run_stage(
            name,
            script,
        )

        integrity = db_integrity()

        print(
            f"{name}_POST_DB_INTEGRITY="
            f"{integrity}"
        )

        if integrity != "ok":
            raise RuntimeError(
                f"FAIL_CLOSED: DB integrity "
                f"failed after {name}"
            )

    print()
    print("=" * 78)
    print("FINAL FEEDBACK AUDIT")
    print("=" * 78)

    usage_2026 = table_count(
        "player_weekly_usage",
        "season=2026",
    )

    pregame_2026 = table_count(
        "player_pregame_features",
        "season=2026",
    )

    matrix_2026 = table_count(
        "dfs_feature_matrix",
        "season=2026",
    )

    team_env_2026 = table_count(
        "team_pregame_environment",
        "season=2026",
    )

    print(
        "PLAYER_WEEKLY_USAGE_2026="
        f"{usage_2026}"
    )

    print(
        "PLAYER_PREGAME_FEATURES_2026="
        f"{pregame_2026}"
    )

    print(
        "DFS_FEATURE_MATRIX_2026="
        f"{matrix_2026}"
    )

    print(
        "TEAM_ENVIRONMENT_2026="
        f"{team_env_2026}"
    )

    final_integrity = db_integrity()

    print(
        "FINAL_DB_INTEGRITY="
        + final_integrity
    )

    required_files = [
        ROOT / "data/parquet/nfl_player_weekly_usage.parquet",
        ROOT / "data/parquet/nfl_player_pregame_features.parquet",
        ROOT / "data/parquet/nfl_team_pregame_environment.parquet",
        ROOT / "data/parquet/nfl_dfs_feature_matrix.parquet",
        ROOT / "data/parquet/nfl_current_slate_features.parquet",
        ROOT / "data/parquet/nfl_production_projection.parquet",
        ROOT / "data/parquet/nfl_fanduel_player_pool.parquet",
        ROOT / "data/parquet/nfl_dst_production_projection.parquet",
    ]

    print()
    print("ARTIFACT HASHES")

    for path in required_files:
        if not path.exists():
            raise RuntimeError(
                f"FAIL_CLOSED: missing artifact {path}"
            )

        print(
            f"{path.name} "
            f"{sha256(path)}"
        )

    passed = (
        final_integrity == "ok"
        and usage_2026 > 0
        and pregame_2026 > 0
        and matrix_2026 > 0
        and team_env_2026 > 0
    )

    print()

    print(
        "NFL_2026_FEEDBACK_1Q_STATUS="
        + (
            "PASS"
            if passed
            else "FAIL_CLOSED"
        )
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
