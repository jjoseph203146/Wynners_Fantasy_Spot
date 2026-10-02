#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


SLATE_TABLE = "fanduel_slate_pool"

TABLES_TO_CLEAR = (
    "fanduel_player_pool",
    "fanduel_slate_projection_pool",
    "fanduel_slate_projection_manifest",
    "fanduel_solver_ready_pool",
    "fanduel_solver_ready_manifest",
)

FILE_TARGETS = (
    (Path(CSV_DIR) / "nfl_fanduel_player_pool.csv", "csv"),
    (Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet", "parquet"),
    (Path(CSV_DIR) / "audit_fanduel_player_pool.csv", "csv"),
    (Path(PARQUET_DIR) / "audit_fanduel_player_pool.parquet", "parquet"),

    (Path(CSV_DIR) / "fanduel_slate_projection_pool.csv", "csv"),
    (Path(PARQUET_DIR) / "fanduel_slate_projection_pool.parquet", "parquet"),
    (Path(CSV_DIR) / "fanduel_slate_projection_manifest.csv", "csv"),
    (Path(CSV_DIR) / "audit_fanduel_slate_projection_gaps.csv", "csv"),

    (Path(CSV_DIR) / "fanduel_solver_ready_pool.csv", "csv"),
    (Path(PARQUET_DIR) / "fanduel_solver_ready_pool.parquet", "parquet"),
    (Path(CSV_DIR) / "fanduel_solver_ready_manifest.csv", "csv"),
    (Path(CSV_DIR) / "audit_fanduel_solver_ready_gaps.csv", "csv"),
)

STATE_FILE = Path(CSV_DIR) / "fanduel_solver_inventory_state.json"


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone() is not None


def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(
        conn.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
    )


def empty_existing_file(path: Path, kind: str) -> None:
    if not path.exists():
        return

    if kind == "csv":
        frame = pd.read_csv(path, nrows=0)
        frame.to_csv(path, index=False)
        return

    if kind == "parquet":
        frame = pd.read_parquet(path)
        frame.iloc[0:0].to_parquet(path, index=False)
        return

    raise RuntimeError(f"Unsupported file kind: {kind}")


def main() -> int:
    print("=" * 88)
    print("FANDUEL SOLVER-OPEN LIFECYCLE GATE")
    print("=" * 88)
    print(f"Database: {DATABASE_PATH}")

    with sqlite3.connect(DATABASE_PATH) as conn:
        if not table_exists(conn, SLATE_TABLE):
            raise RuntimeError(
                f"{SLATE_TABLE} does not exist. "
                "Run fanduel_slate_ingest_v2.py first."
            )

        open_rows = table_count(conn, SLATE_TABLE)

        print(f"Solver-open FanDuel rows: {open_rows}")

        if open_rows > 0:
            print("OPEN_SLATES_PRESENT")
            print("Downstream FanDuel build should continue.")
            return 0

        print()
        print("NO_OPEN_SLATES")
        print("Clearing stale derived FanDuel solver inventory.")

        conn.execute("BEGIN IMMEDIATE")

        try:
            for table in TABLES_TO_CLEAR:
                if not table_exists(conn, table):
                    print(f"SQLite {table}: not present")
                    continue

                before = table_count(conn, table)

                conn.execute(
                    f'DELETE FROM "{table}"'
                )

                after = table_count(conn, table)

                if after != 0:
                    raise RuntimeError(
                        f"Failed to clear {table}: "
                        f"{after} rows remain."
                    )

                print(
                    f"SQLite {table}: "
                    f"{before} -> {after}"
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

    for path, kind in FILE_TARGETS:
        if not path.exists():
            print(f"File absent: {path}")
            continue

        empty_existing_file(path, kind)
        print(f"Cleared file: {path}")

    state = {
        "status": "NO_OPEN_SLATES",
        "solver_open_rows": 0,
        "updated_at": datetime.now().astimezone().isoformat(),
        "authority": SLATE_TABLE,
    }

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_FILE.write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"State file: {STATE_FILE}")
    print("FANDUEL_NO_OPEN_SLATES_PASS")

    # Exit 10 means successful lifecycle no-work:
    # no FanDuel contests are currently solver-open and stale
    # downstream solver inventory has been cleared.
    return 10


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print("=" * 88)
        print("FANDUEL SOLVER-OPEN LIFECYCLE GATE FAILED")
        print("=" * 88)
        print(f"{type(exc).__name__}: {exc}")
        raise
