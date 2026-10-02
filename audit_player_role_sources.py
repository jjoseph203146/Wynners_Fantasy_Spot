#!/usr/bin/env python3
"""RP-0 read-only audit for the NFL player/role performance database.

Reads SQLite databases and selected parquet/csv metadata. It never creates,
updates, or deletes production database objects. Reports are written beneath
data/audits/player_role_sources/.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = DATA_DIR / "audits" / "player_role_sources"

EXPECTED_TABLES = (
    "games",
    "player_identity",
    "weekly_rosters",
    "depth_charts",
    "injuries",
    "player_game_stats",
    "player_snap_counts",
    "player_weekly_usage",
    "team_game_stats",
    "team_defense_game_stats",
    "team_defense_fanduel_scoring",
    "player_pregame_features",
    "team_pregame_environment",
    "team_position_dvp",
    "dfs_feature_matrix",
    "fanduel_slate_pool",
    "fanduel_slate_projection_pool",
    "fanduel_solver_ready_pool",
)

ROLE_FIELD_GROUPS = {
    "identity": ("player_id", "gsis_id", "player_name", "position", "team"),
    "game": ("season", "week", "game_id", "opponent", "kickoff", "status"),
    "participation": ("snaps", "snap_share", "routes", "route_share"),
    "opportunity": (
        "carries", "targets", "receptions", "air_yards", "target_share",
        "red_zone", "goal_line", "two_minute",
    ),
    "actual_scoring": ("fanduel_points", "fantasy_points", "target_fanduel_points"),
    "projection": ("projection", "ridge_projection", "expected_fanduel_points"),
    "time_boundary": ("captured_at", "created_at", "updated_at", "kickoff_at"),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(connection: sqlite3.Connection, sql: str) -> Any:
    return connection.execute(sql).fetchone()[0]


def database_candidates() -> list[Path]:
    roots = (PROJECT_ROOT, DATA_DIR)
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
            for path in root.rglob(pattern):
                if REPORT_DIR in path.parents or path.name.endswith(("-wal", "-shm")):
                    continue
                found.add(path.resolve())
    return sorted(found)


def open_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def table_columns(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    return [dict(row) for row in rows]


def table_indexes(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"PRAGMA index_list({quote_identifier(table)})"
    ).fetchall()
    return [dict(row) for row in rows]


def matching_columns(columns: list[str]) -> dict[str, list[str]]:
    lowered = {column.lower(): column for column in columns}
    result: dict[str, list[str]] = {}
    for group, needles in ROLE_FIELD_GROUPS.items():
        hits = []
        for lower, original in lowered.items():
            if any(needle in lower for needle in needles):
                hits.append(original)
        result[group] = sorted(set(hits))
    return result


def season_week_summary(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> list[dict[str, Any]]:
    lower = {column.lower(): column for column in columns}
    if "season" not in lower or "week" not in lower:
        return []
    season = quote_identifier(lower["season"])
    week = quote_identifier(lower["week"])
    sql = (
        f"SELECT {season} AS season, {week} AS week, COUNT(*) AS rows "
        f"FROM {quote_identifier(table)} GROUP BY {season}, {week} "
        f"ORDER BY {season}, {week}"
    )
    return [dict(row) for row in connection.execute(sql).fetchall()]


def duplicate_summary(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> dict[str, Any]:
    lower = {column.lower(): column for column in columns}
    player_id = lower.get("gsis_id") or lower.get("player_id")
    preferred = [lower.get("season"), lower.get("week"), lower.get("game_id"), player_id]
    keys = [value for value in preferred if value]
    if len(keys) < 2:
        return {"checked": False, "reason": "insufficient natural-key columns"}
    expressions = ", ".join(quote_identifier(key) for key in keys)
    sql = (
        "SELECT COUNT(*) FROM ("
        f"SELECT {expressions}, COUNT(*) AS n FROM {quote_identifier(table)} "
        f"GROUP BY {expressions} HAVING COUNT(*) > 1)"
    )
    return {
        "checked": True,
        "key_columns": keys,
        "duplicate_groups": int(scalar(connection, sql)),
    }


def null_summary(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
) -> dict[str, int]:
    important = {
        "season", "week", "game_id", "player_id", "gsis_id", "position",
        "team", "fanduel_points", "target_fanduel_points",
    }
    selected = [column for column in columns if column.lower() in important]
    result: dict[str, int] = {}
    for column in selected:
        sql = (
            f"SELECT COUNT(*) FROM {quote_identifier(table)} WHERE "
            f"{quote_identifier(column)} IS NULL OR "
            f"TRIM(CAST({quote_identifier(column)} AS TEXT)) = ''"
        )
        result[column] = int(scalar(connection, sql))
    return result


def audit_database(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "OK",
        "tables": {},
    }
    try:
        with open_read_only(path) as connection:
            integrity = scalar(connection, "PRAGMA quick_check")
            result["quick_check"] = integrity
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            table_names = [row[0] for row in table_rows]
            result["expected_tables_present"] = sorted(set(table_names) & set(EXPECTED_TABLES))
            result["expected_tables_missing"] = sorted(set(EXPECTED_TABLES) - set(table_names))

            for table in table_names:
                info = table_columns(connection, table)
                columns = [row["name"] for row in info]
                table_result = {
                    "rows": int(scalar(connection, f"SELECT COUNT(*) FROM {quote_identifier(table)}")),
                    "columns": info,
                    "indexes": table_indexes(connection, table),
                    "role_field_matches": matching_columns(columns),
                    "season_week": season_week_summary(connection, table, columns),
                    "natural_key_audit": duplicate_summary(connection, table, columns),
                    "important_nulls": null_summary(connection, table, columns),
                }
                result["tables"][table] = table_result
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_tabular_files() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not DATA_DIR.exists():
        return results
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or REPORT_DIR in path.parents:
            continue
        if path.suffix.lower() not in {".parquet", ".csv"}:
            continue
        entry: dict[str, Any] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": path.suffix.lower().lstrip("."),
            "status": "OK",
        }
        try:
            if path.suffix.lower() == ".parquet":
                import pandas as pd

                frame = pd.read_parquet(path)
            else:
                import pandas as pd

                frame = pd.read_csv(path, nrows=1000, low_memory=False)
            columns = [str(column) for column in frame.columns]
            entry["rows"] = int(len(frame)) if path.suffix.lower() == ".parquet" else None
            entry["columns"] = columns
            entry["role_field_matches"] = matching_columns(columns)
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return results


def build_text_report(report: dict[str, Any]) -> str:
    lines = [
        "NFL PLAYER/ROLE SOURCE AUDIT — RP-0",
        "=" * 72,
        f"Generated UTC: {report['generated_at']}",
        f"Project root: {report['project_root']}",
        "Mode: READ-ONLY SOURCE AUDIT",
        "",
        "DATABASE SUMMARY",
        "-" * 72,
    ]
    databases = report["databases"]
    if not databases:
        lines.append("NO SQLITE DATABASES FOUND")
    for database in databases:
        lines.append(
            f"{database['status']} | {database['path']} | "
            f"tables={len(database.get('tables', {}))} | "
            f"quick_check={database.get('quick_check', 'NOT_RUN')}"
        )
        for name, table in database.get("tables", {}).items():
            if name in EXPECTED_TABLES:
                duplicate = table["natural_key_audit"]
                duplicate_value = duplicate.get("duplicate_groups", "NOT_CHECKED")
                lines.append(
                    f"  {name}: rows={table['rows']} cols={len(table['columns'])} "
                    f"duplicate_groups={duplicate_value}"
                )

    lines.extend(["", "ROLE-FIELD AVAILABILITY", "-" * 72])
    field_counts: Counter[str] = Counter()
    for database in databases:
        for table in database.get("tables", {}).values():
            for group, matches in table["role_field_matches"].items():
                if matches:
                    field_counts[group] += 1
    for group in ROLE_FIELD_GROUPS:
        lines.append(f"{group}: present in {field_counts[group]} SQLite tables")

    lines.extend(["", "TABULAR FILE SUMMARY", "-" * 72])
    files = report["tabular_files"]
    if not files:
        lines.append("NO PARQUET/CSV FILES FOUND UNDER data/")
    for entry in files:
        lines.append(
            f"{entry['status']} | {entry['path']} | columns={len(entry.get('columns', []))}"
        )

    lines.extend(["", "RP-0 RESULT", "-" * 72])
    errors = report["summary"]["errors"]
    if errors:
        lines.append(f"REVIEW_REQUIRED: {errors} source(s) could not be audited")
    else:
        lines.append("SOURCE_INVENTORY_COMPLETE")
    lines.append("No production database was modified.")
    return "\n".join(lines) + "\n"


def main() -> int:
    print("RP-0 NFL player/role source audit")
    print(f"Project: {PROJECT_ROOT}")
    print("Mode: read-only databases; report files only")

    databases = [audit_database(path) for path in database_candidates()]
    tabular_files = audit_tabular_files()
    errors = sum(item["status"] != "OK" for item in databases + tabular_files)
    report = {
        "contract": "WFS_PLAYER_ROLE_SOURCE_AUDIT_V1",
        "stage": "RP-0",
        "generated_at": now_utc(),
        "project_root": str(PROJECT_ROOT),
        "database_write_policy": "READ_ONLY",
        "databases": databases,
        "tabular_files": tabular_files,
        "summary": {
            "database_count": len(databases),
            "tabular_file_count": len(tabular_files),
            "errors": errors,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"player_role_source_audit_{stamp}.json"
    text_path = REPORT_DIR / f"player_role_source_audit_{stamp}.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(build_text_report(report), encoding="utf-8")

    print(f"JSON_REPORT={json_path}")
    print(f"TEXT_REPORT={text_path}")
    print(f"DATABASES={len(databases)}")
    print(f"TABULAR_FILES={len(tabular_files)}")
    print(f"ERRORS={errors}")
    print("STATUS=OK" if errors == 0 else "STATUS=REVIEW_REQUIRED")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
