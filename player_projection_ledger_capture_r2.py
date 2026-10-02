#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1D-C-R2
Immutable player projection ledger creator / capture writer.

Writes ONLY:
    /home/mwynn/nfl_data_engine/data/player_projection_ledger.db

Reads:
    /home/mwynn/nfl_data_engine/data/nfl.db  (SQLite mode=ro)

Safety contract:
- Resolve the current projection universe to exactly one season/week by requiring
  complete, unique matchup coverage inside that season/week.
- Then map every projection row only within that resolved season/week.
- Capture only games strictly before kickoff and not completed.
- Offense requires exact projection_identity prefix "GSIS:".
- D/ST uses exact team + game_id identity.
- No fuzzy matching. No name-only recovery.
- Only projection_status == READY rows are captured.
- Existing snapshots/predictions are immutable.
- Identical canonical capture payloads are idempotent no-ops.
- No writes to nfl.db, forecast_ledger.db, updater, LIVE, injury, solver, cron,
  projection producers, or services.
"""

from __future__ import annotations

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
SOURCE_MODULE = ROOT / "fanduel_slate_projection_attach_v5.py"

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

SCHEMA_VERSION = 1


def ro_connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def sha256_file(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"missing source module: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_float(value: Any, label: str) -> float:
    if value is None:
        raise RuntimeError(f"{label}: NULL")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: invalid numeric {value!r}") from exc
    if not math.isfinite(x):
        raise RuntimeError(f"{label}: non-finite numeric {value!r}")
    return x


def parse_kickoff_utc(game_date: Any, gametime: Any, game_id: str) -> datetime:
    if game_date is None or gametime is None:
        raise RuntimeError(f"{game_id}: missing kickoff date/time")

    text = f"{str(game_date).strip()} {str(gametime).strip()}"
    parsed = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        raise RuntimeError(f"{game_id}: invalid kickoff {text!r}")

    return parsed.replace(tzinfo=ET).astimezone(UTC)


def resolve_season_week(
    projection_games: set[tuple[str, str]],
    schedule_rows: list[sqlite3.Row],
) -> tuple[int, int]:
    grouped: dict[tuple[int, int], list[sqlite3.Row]] = defaultdict(list)

    for g in schedule_rows:
        if g["season"] is None or g["week"] is None:
            continue
        grouped[(int(g["season"]), int(g["week"]))].append(g)

    full: list[tuple[int, int]] = []

    for season_week, games in grouped.items():
        by_key: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for g in games:
            key = (
                str(g["away_team"]).strip(),
                str(g["home_team"]).strip(),
            )
            by_key[key].append(g)

        matched = 0
        missing = 0
        ambiguous = 0

        for key in projection_games:
            hits = by_key.get(key, [])
            if len(hits) == 1:
                matched += 1
            elif len(hits) == 0:
                missing += 1
            else:
                ambiguous += 1

        if (
            matched == len(projection_games)
            and missing == 0
            and ambiguous == 0
        ):
            full.append(season_week)

    print(f"FULL_COVERAGE_SEASON_WEEK_CANDIDATES={len(full)}")
    if len(full) != 1:
        for season, week in sorted(full):
            print(f"FULL_CANDIDATE=season={season}|week={week}")
        raise RuntimeError(
            "projection universe does not resolve to exactly one season/week"
        )

    return full[0]


def initialize_ledger(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS player_projection_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            captured_at_utc TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            source_pool_sha256 TEXT NOT NULL,
            capture_payload_sha256 TEXT NOT NULL,
            source_module_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            offense_row_count INTEGER NOT NULL,
            dst_row_count INTEGER NOT NULL,
            excluded_started_or_completed INTEGER NOT NULL,
            excluded_offense_identity INTEGER NOT NULL,
            excluded_projection_not_ready INTEGER NOT NULL,
            snapshot_status TEXT NOT NULL,
            UNIQUE(season, week, capture_payload_sha256)
        );

        CREATE TABLE IF NOT EXISTS player_projection_predictions (
            snapshot_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            slate_slug TEXT NOT NULL,
            is_dst INTEGER NOT NULL CHECK(is_dst IN (0,1)),
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            opponent_team TEXT NOT NULL,
            salary REAL NOT NULL,
            projection REAL NOT NULL,
            projection_source TEXT NOT NULL,
            projection_match_method TEXT NOT NULL,
            projection_status TEXT NOT NULL,
            optimizer_eligible INTEGER NOT NULL CHECK(optimizer_eligible IN (0,1)),
            PRIMARY KEY (
                snapshot_id,
                slate_slug,
                game_id,
                is_dst,
                player_id,
                player_name,
                team
            ),
            FOREIGN KEY(snapshot_id)
                REFERENCES player_projection_snapshots(snapshot_id)
                ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_pp_snapshots_capture_time
            ON player_projection_snapshots(captured_at_utc);

        CREATE INDEX IF NOT EXISTS idx_pp_snapshots_season_week
            ON player_projection_snapshots(season, week);

        CREATE INDEX IF NOT EXISTS idx_pp_predictions_game
            ON player_projection_predictions(game_id);

        CREATE INDEX IF NOT EXISTS idx_pp_predictions_player
            ON player_projection_predictions(player_id);

        CREATE INDEX IF NOT EXISTS idx_pp_predictions_snapshot_game
            ON player_projection_predictions(snapshot_id, game_id);
        """
    )

    existing = conn.execute(
        "SELECT value FROM ledger_meta WHERE key='schema_version'"
    ).fetchone()

    if existing is None:
        conn.execute(
            "INSERT INTO ledger_meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
    elif str(existing[0]) != str(SCHEMA_VERSION):
        raise RuntimeError(
            f"ledger schema version mismatch: existing={existing[0]} "
            f"expected={SCHEMA_VERSION}"
        )


def main() -> int:
    capture_dt = datetime.now(UTC)
    capture_iso = capture_dt.isoformat(timespec="seconds")

    print("=" * 70)
    print("WFS NFL — NFL-POSTGAME-1D-C-R2")
    print("IMMUTABLE PLAYER PROJECTION LEDGER CAPTURE")
    print("=" * 70)

    with ro_connect(NFL_DB) as nfl:
        integrity = nfl.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity check failed")

        required_tables = {"games", "fanduel_slate_projection_pool"}
        actual_tables = {
            r[0]
            for r in nfl.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - actual_tables
        if missing:
            raise RuntimeError(f"missing required tables: {sorted(missing)}")

        raw = nfl.execute(
            """
            SELECT
                slate_slug,
                player,
                team_internal,
                away_team,
                home_team,
                game_key,
                is_dst,
                salary,
                internal_projection,
                model_projection,
                projection_source,
                projection_match_method,
                projection_identity,
                projection_status,
                optimizer_eligible
            FROM fanduel_slate_projection_pool
            ORDER BY
                slate_slug,
                game_key,
                is_dst,
                projection_identity,
                player,
                salary
            """
        ).fetchall()

        if not raw:
            raise RuntimeError("projection pool is empty")

        source_rows = [dict(r) for r in raw]
        source_pool_sha = canonical_sha(source_rows)

        print(f"SOURCE_PROJECTION_ROWS={len(source_rows)}")
        print(f"SOURCE_POOL_SEMANTIC_SHA256={source_pool_sha}")

        projection_games = {
            (
                str(r["away_team"] or "").strip(),
                str(r["home_team"] or "").strip(),
            )
            for r in raw
        }

        if any(not a or not h for a, h in projection_games):
            raise RuntimeError("blank away/home team in projection universe")

        print(f"DISTINCT_PROJECTION_GAMES={len(projection_games)}")

        schedule = nfl.execute(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                gametime,
                away_team,
                home_team,
                completed
            FROM games
            WHERE season IS NOT NULL
              AND week IS NOT NULL
            ORDER BY season, week, game_id
            """
        ).fetchall()

        season, week = resolve_season_week(projection_games, schedule)
        print(f"RESOLVED_SEASON={season}")
        print(f"RESOLVED_WEEK={week}")

        current_schedule = [
            g
            for g in schedule
            if int(g["season"]) == season and int(g["week"]) == week
        ]

        by_matchup: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for g in current_schedule:
            key = (
                str(g["away_team"]).strip(),
                str(g["home_team"]).strip(),
            )
            by_matchup[key].append(g)

        mapped: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        unmapped: list[str] = []

        for r in raw:
            key = (
                str(r["away_team"] or "").strip(),
                str(r["home_team"] or "").strip(),
            )
            hits = by_matchup.get(key, [])
            if len(hits) != 1:
                unmapped.append(
                    f"{r['slate_slug']}:{r['player']}:{key}:matches={len(hits)}"
                )
            else:
                mapped.append((r, hits[0]))

        print(f"EXACT_SEASON_WEEK_MAPPED_ROWS={len(mapped)}")
        print(f"EXACT_SEASON_WEEK_UNMAPPED_ROWS={len(unmapped)}")

        if unmapped:
            for x in unmapped[:25]:
                print(f"UNMAPPED={x}")
            raise RuntimeError("season/week scoped schedule mapping failed")

        captured: list[dict[str, Any]] = []
        excluded_started = 0
        excluded_identity = 0
        excluded_not_ready = 0

        for r, g in mapped:
            game_id = str(g["game_id"])
            kickoff_utc = parse_kickoff_utc(
                g["game_date"],
                g["gametime"],
                game_id,
            )

            if int(g["completed"] or 0) == 1 or capture_dt >= kickoff_utc:
                excluded_started += 1
                continue

            projection_status = str(r["projection_status"] or "").strip()
            if projection_status != "READY":
                excluded_not_ready += 1
                continue

            is_dst = int(r["is_dst"] or 0)
            if is_dst not in (0, 1):
                raise RuntimeError(
                    f"{game_id}:{r['player']}: invalid is_dst={r['is_dst']!r}"
                )

            team = str(r["team_internal"] or "").strip()
            away = str(r["away_team"] or "").strip()
            home = str(r["home_team"] or "").strip()

            if not team or team not in {away, home}:
                raise RuntimeError(
                    f"{game_id}:{r['player']}: invalid team_internal={team!r}"
                )

            opponent = home if team == away else away

            if is_dst == 0:
                ident = str(r["projection_identity"] or "").strip()
                if not ident.startswith("GSIS:") or len(ident) <= 5:
                    excluded_identity += 1
                    continue
                player_id = ident[5:]
            else:
                player_id = f"DST:{team}"

            player_name = str(r["player"] or "").strip()
            slate_slug = str(r["slate_slug"] or "").strip()

            if not player_name:
                raise RuntimeError(f"{game_id}: blank player name")
            if not slate_slug:
                raise RuntimeError(f"{game_id}:{player_name}: blank slate_slug")

            salary = finite_float(
                r["salary"],
                f"{game_id}:{player_name}:salary",
            )
            projection = finite_float(
                r["model_projection"],
                f"{game_id}:{player_name}:model_projection",
            )

            captured.append(
                {
                    "game_id": game_id,
                    "slate_slug": slate_slug,
                    "is_dst": is_dst,
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": team,
                    "opponent_team": opponent,
                    "salary": salary,
                    "projection": projection,
                    "projection_source": str(r["projection_source"] or ""),
                    "projection_match_method": str(
                        r["projection_match_method"] or ""
                    ),
                    "projection_status": projection_status,
                    "optimizer_eligible": int(r["optimizer_eligible"] or 0),
                }
            )

        if not captured:
            raise RuntimeError("no eligible pregame projection rows to capture")

        captured.sort(
            key=lambda x: (
                x["game_id"],
                x["slate_slug"],
                x["is_dst"],
                x["player_id"],
                x["player_name"],
                x["team"],
            )
        )

        immutable_keys = [
            (
                x["slate_slug"],
                x["game_id"],
                x["is_dst"],
                x["player_id"],
                x["player_name"],
                x["team"],
            )
            for x in captured
        ]

        if len(immutable_keys) != len(set(immutable_keys)):
            raise RuntimeError("duplicate immutable prediction keys in payload")

        capture_payload_sha = canonical_sha(captured)
        source_module_sha = sha256_file(SOURCE_MODULE)

        offense_rows = sum(1 for x in captured if x["is_dst"] == 0)
        dst_rows = sum(1 for x in captured if x["is_dst"] == 1)

        print(f"CAPTURED_AT_UTC={capture_iso}")
        print(f"CAPTURE_ROWS={len(captured)}")
        print(f"CAPTURE_OFFENSE_ROWS={offense_rows}")
        print(f"CAPTURE_DST_ROWS={dst_rows}")
        print(f"EXCLUDED_STARTED_OR_COMPLETED_ROWS={excluded_started}")
        print(f"EXCLUDED_OFFENSE_IDENTITY_ROWS={excluded_identity}")
        print(f"EXCLUDED_PROJECTION_NOT_READY_ROWS={excluded_not_ready}")
        print(f"CAPTURE_PAYLOAD_SHA256={capture_payload_sha}")
        print(f"SOURCE_MODULE_SHA256={source_module_sha}")

    snapshot_id = (
        f"{season}-w{week:02d}-player-v1-"
        f"{capture_payload_sha[:16]}"
    )
    print(f"SNAPSHOT_ID={snapshot_id}")

    LEDGER_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(LEDGER_DB)
    conn.row_factory = sqlite3.Row

    try:
        initialize_ledger(conn)

        existing = conn.execute(
            """
            SELECT *
            FROM player_projection_snapshots
            WHERE season=?
              AND week=?
              AND capture_payload_sha256=?
            """,
            (season, week, capture_payload_sha),
        ).fetchall()

        if len(existing) > 1:
            raise RuntimeError("duplicate idempotency rows already exist")

        if len(existing) == 1:
            old = existing[0]

            if str(old["snapshot_id"]) != snapshot_id:
                raise RuntimeError("existing snapshot_id mismatch")

            if str(old["source_pool_sha256"]) != source_pool_sha:
                raise RuntimeError(
                    "identical capture payload has different source pool hash"
                )

            if str(old["source_module_sha256"]) != source_module_sha:
                raise RuntimeError(
                    "identical capture payload has different source module hash"
                )

            existing_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM player_projection_predictions
                WHERE snapshot_id=?
                """,
                (snapshot_id,),
            ).fetchone()[0]

            if existing_count != len(captured):
                raise RuntimeError(
                    "existing snapshot prediction count mismatch: "
                    f"{existing_count} != {len(captured)}"
                )

            print("CAPTURE_ACTION=IDEMPOTENT_NOOP")
            print(f"EXISTING_PREDICTION_ROWS={existing_count}")
            conn.rollback()

        else:
            conn.execute("BEGIN IMMEDIATE")

            conn.execute(
                """
                INSERT INTO player_projection_snapshots (
                    snapshot_id,
                    captured_at_utc,
                    season,
                    week,
                    source_pool_sha256,
                    capture_payload_sha256,
                    source_module_sha256,
                    row_count,
                    offense_row_count,
                    dst_row_count,
                    excluded_started_or_completed,
                    excluded_offense_identity,
                    excluded_projection_not_ready,
                    snapshot_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    capture_iso,
                    season,
                    week,
                    source_pool_sha,
                    capture_payload_sha,
                    source_module_sha,
                    len(captured),
                    offense_rows,
                    dst_rows,
                    excluded_started,
                    excluded_identity,
                    excluded_not_ready,
                    "PROSPECTIVE_CAPTURE",
                ),
            )

            conn.executemany(
                """
                INSERT INTO player_projection_predictions (
                    snapshot_id,
                    game_id,
                    slate_slug,
                    is_dst,
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
                ) VALUES (
                    :snapshot_id,
                    :game_id,
                    :slate_slug,
                    :is_dst,
                    :player_id,
                    :player_name,
                    :team,
                    :opponent_team,
                    :salary,
                    :projection,
                    :projection_source,
                    :projection_match_method,
                    :projection_status,
                    :optimizer_eligible
                )
                """,
                [dict(x, snapshot_id=snapshot_id) for x in captured],
            )

            inserted = conn.execute(
                """
                SELECT COUNT(*)
                FROM player_projection_predictions
                WHERE snapshot_id=?
                """,
                (snapshot_id,),
            ).fetchone()[0]

            if inserted != len(captured):
                raise RuntimeError(
                    f"insert count mismatch: {inserted} != {len(captured)}"
                )

            conn.commit()
            print("CAPTURE_ACTION=INSERTED")
            print(f"INSERTED_PREDICTION_ROWS={inserted}")

        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()

        snapshot_rows = conn.execute(
            "SELECT COUNT(*) FROM player_projection_snapshots"
        ).fetchone()[0]

        prediction_rows = conn.execute(
            "SELECT COUNT(*) FROM player_projection_predictions"
        ).fetchone()[0]

        this_snapshot_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM player_projection_predictions
            WHERE snapshot_id=?
            """,
            (snapshot_id,),
        ).fetchone()[0]

        print(f"LEDGER_INTEGRITY={integrity}")
        print(f"LEDGER_FOREIGN_KEY_ERRORS={len(fk_errors)}")
        print(f"LEDGER_SNAPSHOT_ROWS={snapshot_rows}")
        print(f"LEDGER_PREDICTION_ROWS={prediction_rows}")
        print(f"THIS_SNAPSHOT_PREDICTION_ROWS={this_snapshot_rows}")

        if integrity != "ok":
            raise RuntimeError("ledger integrity check failed")
        if fk_errors:
            raise RuntimeError("ledger foreign key check failed")
        if this_snapshot_rows != len(captured):
            raise RuntimeError("post-write snapshot count verification failed")

    finally:
        conn.close()

    print("NFL_POSTGAME_1D_C_R2_STATUS=PASS")
    print("LEDGER_DB_CREATED_OR_VERIFIED=TRUE")
    print("NFL_DB_WRITES=0")
    print("FORECAST_LEDGER_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
