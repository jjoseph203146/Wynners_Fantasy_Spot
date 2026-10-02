#!/usr/bin/env python3
"""Load an official FanDuel NFL players-list CSV into the analysis database.

This loader never writes to data/nfl.db. It is deliberately strict and
idempotent: malformed pools are rejected, an automatic database backup is
created, and rerunning the same source hash replaces only that source's rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
TARGET_DB = PROJECT / "data" / "fanduel_player_role_performance.db"
REQUIRED_COLUMNS = {
    "Id",
    "Position",
    "Nickname",
    "Salary",
    "Game",
    "Team",
    "Opponent",
    "Injury Indicator",
    "Injury Details",
    "Roster Position",
}
ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE", "D"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def discover_source(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            fail(f"SOURCE_FILE_NOT_FOUND:{path}")
        return path

    roots = [PROJECT, PROJECT / "data" / "uploads", Path.home() / "Downloads"]
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(root.glob("*FanDuel*NFL*players-list*.csv"))
            matches.extend(root.glob("*fanduel*nfl*players-list*.csv"))
    unique = {path.resolve() for path in matches if path.is_file()}
    if not unique:
        fail("NO_FANDUEL_NFL_PLAYERS_LIST_FOUND")
    return max(unique, key=lambda path: path.stat().st_mtime)


def parse_filename(path: Path) -> tuple[str, str | None]:
    name = path.name
    date_match = re.search(
        r"(20\d{2})(?:\s*EDT)?-(\d{2})(?:\s*EDT)?-(\d{2})",
        name,
        flags=re.IGNORECASE,
    )
    if not date_match:
        fail("SLATE_DATE_NOT_FOUND_IN_FILENAME")
    year, month, day = date_match.groups()
    slate_date = f"{year}-{month}-{day}"

    suffix = name[date_match.end() :]
    time_match = re.search(r"-(\d{6})(?:-|\.)", suffix)
    captured_at = None
    if time_match:
        hhmmss = time_match.group(1)
        # FanDuel labeled this supplied filename EDT. Store an aware timestamp.
        captured_at = (
            f"{slate_date}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}-04:00"
            if "EDT" in name.upper()
            else f"{slate_date}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
        )
    return slate_date, captured_at


def read_pool(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [clean(column) for column in (reader.fieldnames or [])]
        missing = sorted(REQUIRED_COLUMNS - set(headers))
        if missing:
            fail("MISSING_COLUMNS:" + ",".join(missing))
        rows = [{clean(k): clean(v) for k, v in row.items() if k is not None} for row in reader]
    if not rows:
        fail("EMPTY_PLAYER_POOL")
    return rows, headers


def validate_and_prepare(
    rows: list[dict[str, str]], slate_date: str
) -> tuple[list[tuple], str, int, int]:
    ids: set[str] = set()
    games: set[str] = set()
    prepared: list[tuple] = []
    injured = 0
    defenses = 0

    for number, row in enumerate(rows, start=2):
        fd_id = clean(row.get("Id"))
        name = clean(row.get("Nickname"))
        position_raw = clean(row.get("Position")).upper()
        team = clean(row.get("Team")).upper()
        opponent = clean(row.get("Opponent")).upper()
        game = clean(row.get("Game")).upper()
        roster_position = clean(row.get("Roster Position")).upper()
        indicator = clean(row.get("Injury Indicator")).upper()
        details = clean(row.get("Injury Details"))

        if not fd_id or not name or not team or not game:
            fail(f"REQUIRED_VALUE_MISSING_AT_CSV_ROW:{number}")
        if fd_id in ids:
            fail(f"DUPLICATE_FANDUEL_ID:{fd_id}")
        ids.add(fd_id)
        if position_raw not in ALLOWED_POSITIONS:
            fail(f"UNSUPPORTED_POSITION:{position_raw}:CSV_ROW:{number}")
        try:
            salary = int(clean(row.get("Salary")))
        except ValueError:
            fail(f"INVALID_SALARY:CSV_ROW:{number}")
        if salary <= 0:
            fail(f"NONPOSITIVE_SALARY:CSV_ROW:{number}")
        if position_raw != "D" and not opponent:
            fail(f"MISSING_OPPONENT:CSV_ROW:{number}")

        games.add(game)
        position = "DST" if position_raw == "D" else position_raw
        if indicator == "IR":
            roster_status = "IR"
            injured += 1
        elif indicator or details:
            roster_status = "INJURY_REVIEW"
            injured += 1
        else:
            roster_status = "ACTIVE"
        identity_status = "TEAM_DEFENSE" if position == "DST" else "PENDING_EXACT_CROSSWALK"
        defenses += int(position == "DST")

        prepared.append(
            (
                slate_date,
                fd_id,
                None,
                name,
                team,
                opponent or None,
                position,
                salary,
                roster_status,
                identity_status,
            )
        )

    game_count = len(games)
    if game_count < 1:
        fail("NO_GAMES_IN_POOL")
    slate_type = "CLASSIC" if game_count > 1 else "SINGLE_GAME"
    if game_count == 2:
        slate_type = "CLASSIC_2_GAME"
    return prepared, slate_type, injured, defenses


def verify_schema(conn: sqlite3.Connection) -> None:
    expected = {
        "source_file_manifest": {
            "source_file_id", "source_type", "source_path", "source_filename",
            "source_sha256", "source_size_bytes", "slate_date", "slate_type",
            "captured_at", "imported_at", "import_status",
        },
        "fact_fanduel_slate_player": {
            "source_file_id", "slate_date", "slate_type", "fanduel_player_id",
            "gsis_id", "player_name", "team", "opponent_team", "position",
            "salary", "roster_status", "identity_status",
        },
    }
    for table, required in expected.items():
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if not columns:
            fail(f"TARGET_TABLE_MISSING:{table}")
        missing = required - columns
        if missing:
            fail(f"TARGET_COLUMNS_MISSING:{table}:" + ",".join(sorted(missing)))


def backup_database(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = PROJECT / "backups" / "fanduel_player_role_performance"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"pre_slate_pool_{stamp}.db"
    source_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    backup_conn = sqlite3.connect(target)
    try:
        source_conn.backup(backup_conn)
    finally:
        backup_conn.close()
        source_conn.close()
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="Official FanDuel players-list CSV")
    parser.add_argument("--slate-type", help="Optional explicit slate label")
    args = parser.parse_args()

    print("RP-4A OFFICIAL FANDUEL SLATE POOL LOAD")
    print(f"TARGET={TARGET_DB}")
    print("PRODUCTION_DATABASE_WRITE=FALSE")
    print("SALARY_AUTHORITY=OFFICIAL_FANDUEL_PLAYER_LIST")
    print("FUZZY_IDENTITY_MATCHING=FALSE")

    try:
        source = discover_source(args.source)
        if source.resolve() == TARGET_DB.resolve():
            fail("SOURCE_CANNOT_BE_TARGET_DATABASE")
        if not TARGET_DB.is_file():
            fail(f"TARGET_DATABASE_NOT_FOUND:{TARGET_DB}")

        slate_date, captured_from_name = parse_filename(source)
        rows, _ = read_pool(source)
        prepared, inferred_type, injured, defenses = validate_and_prepare(rows, slate_date)
        slate_type = clean(args.slate_type).upper() if args.slate_type else inferred_type
        if not slate_type:
            fail("EMPTY_SLATE_TYPE")

        digest = sha256_file(source)
        size = source.stat().st_size
        captured_at = captured_from_name or datetime.fromtimestamp(
            source.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        imported_at = datetime.now(timezone.utc).isoformat()
        backup = backup_database(TARGET_DB)

        conn = sqlite3.connect(TARGET_DB)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            verify_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT source_file_id
                FROM source_file_manifest
                WHERE source_type = ? AND source_sha256 = ?
                ORDER BY source_file_id
                LIMIT 1
                """,
                ("FANDUEL_NFL_PLAYER_LIST", digest),
            ).fetchone()
            if existing:
                source_file_id = int(existing[0])
                conn.execute(
                    "DELETE FROM fact_fanduel_slate_player WHERE source_file_id = ?",
                    (source_file_id,),
                )
                conn.execute(
                    """
                    UPDATE source_file_manifest
                    SET source_path=?, source_filename=?, source_size_bytes=?,
                        slate_date=?, slate_type=?, captured_at=?, imported_at=?,
                        import_status='IMPORTED'
                    WHERE source_file_id=?
                    """,
                    (
                        str(source), source.name, size, slate_date, slate_type,
                        captured_at, imported_at, source_file_id,
                    ),
                )
                load_mode = "RELOAD_SAME_HASH"
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO source_file_manifest (
                        source_type, source_path, source_filename, source_sha256,
                        source_size_bytes, slate_date, slate_type, captured_at,
                        imported_at, import_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'IMPORTED')
                    """,
                    (
                        "FANDUEL_NFL_PLAYER_LIST", str(source), source.name, digest,
                        size, slate_date, slate_type, captured_at, imported_at,
                    ),
                )
                source_file_id = int(cursor.lastrowid)
                load_mode = "NEW_SOURCE"

            conn.executemany(
                """
                INSERT INTO fact_fanduel_slate_player (
                    source_file_id, slate_date, slate_type, fanduel_player_id,
                    gsis_id, player_name, team, opponent_team, position, salary,
                    roster_status, identity_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(source_file_id, row[0], slate_type, *row[1:]) for row in prepared],
            )

            loaded = conn.execute(
                "SELECT COUNT(*) FROM fact_fanduel_slate_player WHERE source_file_id=?",
                (source_file_id,),
            ).fetchone()[0]
            if loaded != len(prepared):
                fail(f"ROW_COUNT_MISMATCH:EXPECTED={len(prepared)}:LOADED={loaded}")
            fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_errors:
                fail(f"FOREIGN_KEY_ERRORS:{len(fk_errors)}")
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if quick_check != "ok":
                fail(f"QUICK_CHECK:{quick_check}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        print(f"SOURCE={source}")
        print(f"SOURCE_SHA256={digest}")
        print(f"BACKUP={backup}")
        print(f"SOURCE_FILE_ID={source_file_id}")
        print(f"SLATE_DATE={slate_date}")
        print(f"SLATE_TYPE={slate_type}")
        print(f"PLAYERS_LOADED={loaded}")
        print(f"DEFENSES_LOADED={defenses}")
        print(f"INJURY_REVIEW_OR_IR={injured}")
        print(f"LOAD_MODE={load_mode}")
        print("IDENTITY_STATUS=PENDING_EXACT_CROSSWALK")
        print("QUICK_CHECK=ok")
        print("STATUS=OK")
        return 0
    except Exception as exc:
        print(f"ERROR={type(exc).__name__}:{exc}")
        print("STATUS=FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
