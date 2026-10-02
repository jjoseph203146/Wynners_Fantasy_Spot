#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1D-C
Immutable player projection ledger creator / capture writer.

Writes ONLY:
    data/player_projection_ledger.db

Reads:
    data/nfl.db (SQLite mode=ro)

It does NOT modify nfl.db, forecast_ledger.db, updater, cron, LIVE,
injury pipeline, solver, projection producer, or services.

Contract:
- current season/week is derived from projection rows -> exact schedule mapping
- capture only games strictly before kickoff at capture time
- completed/started games are excluded
- offense requires exact projection_identity prefix "GSIS:"
- DST uses exact team + game_id identity
- no fuzzy matching or name-only recovery
- source projection rows are canonically hashed
- identical canonical payloads for the same season/week are idempotent
- existing snapshots/predictions are immutable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
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


def ro_connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
            pass
    if parsed is None:
        raise RuntimeError(f"{game_id}: invalid kickoff {text!r}")
    return parsed.replace(tzinfo=ET).astimezone(UTC)


def require_finite(value: Any, label: str) -> float:
    if value is None:
        raise RuntimeError(f"{label}: NULL")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label}: invalid numeric value {value!r}") from exc
    if not math.isfinite(x):
        raise RuntimeError(f"{label}: non-finite numeric value {value!r}")
    return x


def init_ledger(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS player_projection_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            captured_at_utc TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            source_projection_sha256 TEXT NOT NULL,
            source_module_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            offense_row_count INTEGER NOT NULL,
            dst_row_count INTEGER NOT NULL,
            excluded_started_or_completed INTEGER NOT NULL,
            excluded_offense_identity INTEGER NOT NULL,
            snapshot_status TEXT NOT NULL,
            UNIQUE(season, week, source_projection_sha256)
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
                snapshot_id, slate_slug, game_id, is_dst, player_id, player_name, team
            ),
            FOREIGN KEY(snapshot_id)
                REFERENCES player_projection_snapshots(snapshot_id)
                ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_player_projection_predictions_game
            ON player_projection_predictions(game_id);

        CREATE INDEX IF NOT EXISTS idx_player_projection_predictions_player
            ON player_projection_predictions(player_id);

        CREATE INDEX IF NOT EXISTS idx_player_projection_snapshots_capture
            ON player_projection_snapshots(captured_at_utc);
        """
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfl-db", type=Path, default=NFL_DB)
    ap.add_argument("--ledger-db", type=Path, default=LEDGER_DB)
    ap.add_argument("--source-module", type=Path, default=SOURCE_MODULE)
    args = ap.parse_args()

    capture_dt = datetime.now(UTC)
    capture_iso = capture_dt.isoformat(timespec="seconds")

    print("=" * 70)
    print("WFS NFL — NFL-POSTGAME-1D-C")
    print("IMMUTABLE PLAYER PROJECTION LEDGER CAPTURE")
    print("=" * 70)

    with ro_connect(args.nfl_db) as nfl:
        integrity = nfl.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={integrity}")
        if integrity != "ok":
            raise RuntimeError("nfl.db integrity check failed")

        tables = {
            r[0]
            for r in nfl.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for required in ("games", "fanduel_slate_projection_pool"):
            if required not in tables:
                raise RuntimeError(f"missing required table: {required}")

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
                slate_slug, game_key, is_dst,
                projection_identity, player, salary
            """
        ).fetchall()

        if not raw:
            raise RuntimeError("projection pool is empty")

        source_rows = [dict(r) for r in raw]
        source_sha = canonical_sha(source_rows)
        print(f"SOURCE_PROJECTION_ROWS={len(source_rows)}")
        print(f"SOURCE_PROJECTION_SEMANTIC_SHA256={source_sha}")

        schedule = nfl.execute(
            """
            SELECT
                game_id, season, week, game_date, gametime,
                away_team, home_team, completed
            FROM games
            """
        ).fetchall()

        by_matchup: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for g in schedule:
            key = (str(g["away_team"]), str(g["home_team"]))
            by_matchup.setdefault(key, []).append(g)

        mapped: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        unmapped: list[str] = []

        for r in raw:
            key = (str(r["away_team"]), str(r["home_team"]))
            candidates = by_matchup.get(key, [])
            if len(candidates) != 1:
                unmapped.append(
                    f"{r['slate_slug']}:{r['player']}:{key}:matches={len(candidates)}"
                )
                continue
            mapped.append((r, candidates[0]))

        print(f"EXACT_SCHEDULE_MAPPED_ROWS={len(mapped)}")
        print(f"EXACT_SCHEDULE_UNMAPPED_ROWS={len(unmapped)}")
        if unmapped:
            for x in unmapped[:25]:
                print(f"UNMAPPED={x}")
            raise RuntimeError("exact schedule mapping failed")

        season_weeks = {
            (int(g["season"]), int(g["week"]))
            for _, g in mapped
            if g["season"] is not None and g["week"] is not None
        }
        if len(season_weeks) != 1:
            raise RuntimeError(
                f"projection pool spans multiple season/week values: {sorted(season_weeks)}"
            )
        season, week = next(iter(season_weeks))
        print(f"CAPTURE_SEASON={season}")
        print(f"CAPTURE_WEEK={week}")
        print(f"CAPTURED_AT_UTC={capture_iso}")

        captured: list[dict[str, Any]] = []
        excluded_started = 0
        excluded_identity = 0

        for r, g in mapped:
            game_id = str(g["game_id"])
            kickoff = parse_kickoff_utc(g["game_date"], g["gametime"], game_id)

            if int(g["completed"] or 0) == 1 or capture_dt >= kickoff:
                excluded_started += 1
                continue

            is_dst = int(r["is_dst"] or 0)
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
            if not player_name or not slate_slug:
                raise RuntimeError(f"{game_id}: blank player/slate identity")

            projection_status = str(r["projection_status"] or "").strip()
            if projection_status != "READY":
                continue

            projection = require_finite(
                r["model_projection"],
                f"{game_id}:{player_name}:model_projection",
            )
            salary = require_finite(
                r["salary"],
                f"{game_id}:{player_name}:salary",
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

        if not captured:
            raise RuntimeError("no eligible pregame projection rows to capture")

        # Detect duplicate immutable prediction keys before any ledger write.
        keys = [
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
        if len(keys) != len(set(keys)):
            raise RuntimeError("duplicate prediction keys in capture payload")

        payload_sha = canonical_sha(captured)
        module_sha = sha256_file(args.source_module)

        offense_n = sum(1 for x in captured if x["is_dst"] == 0)
        dst_n = sum(1 for x in captured if x["is_dst"] == 1)

        print(f"CAPTURE_ROWS={len(captured)}")
        print(f"CAPTURE_OFFENSE_ROWS={offense_n}")
        print(f"CAPTURE_DST_ROWS={dst_n}")
        print(f"EXCLUDED_STARTED_OR_COMPLETED_ROWS={excluded_started}")
        print(f"EXCLUDED_OFFENSE_IDENTITY_ROWS={excluded_identity}")
        print(f"CAPTURE_PAYLOAD_SHA256={payload_sha}")
        print(f"SOURCE_MODULE_SHA256={module_sha}")

    # Snapshot ID is content-addressed for deterministic idempotency.
    snapshot_id = (
        f"{season}-w{week:02d}-player-v1-"
        f"{payload_sha[:16]}"
    )
    print(f"SNAPSHOT_ID={snapshot_id}")

    args.ledger_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.ledger_db)
    conn.row_factory = sqlite3.Row
    try:
        init_ledger(conn)

        existing = conn.execute(
            """
            SELECT *
            FROM player_projection_snapshots
            WHERE season=? AND week=? AND source_projection_sha256=?
            """,
            (season, week, payload_sha),
        ).fetchall()

        if len(existing) > 1:
            raise RuntimeError("duplicate idempotency rows already exist")

        if len(existing) == 1:
            old = existing[0]
            if str(old["snapshot_id"]) != snapshot_id:
                raise RuntimeError("existing idempotent snapshot_id mismatch")

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
                    "existing snapshot row count does not match payload"
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
                    source_projection_sha256,
                    source_module_sha256,
                    row_count,
                    offense_row_count,
                    dst_row_count,
                    excluded_started_or_completed,
                    excluded_offense_identity,
                    snapshot_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    capture_iso,
                    season,
                    week,
                    payload_sha,
                    module_sha,
                    len(captured),
                    offense_n,
                    dst_n,
                    excluded_started,
                    excluded_identity,
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
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        snapshots = conn.execute(
            "SELECT COUNT(*) FROM player_projection_snapshots"
        ).fetchone()[0]
        predictions = conn.execute(
            "SELECT COUNT(*) FROM player_projection_predictions"
        ).fetchone()[0]

        print(f"LEDGER_INTEGRITY={integrity}")
        print(f"LEDGER_FOREIGN_KEY_ERRORS={len(fk)}")
        print(f"LEDGER_SNAPSHOT_ROWS={snapshots}")
        print(f"LEDGER_PREDICTION_ROWS={predictions}")

        if integrity != "ok" or fk:
            raise RuntimeError("ledger integrity/fk audit failed")

    finally:
        conn.close()

    print("NFL_POSTGAME_1D_C_STATUS=PASS")
    print("NFL_DB_WRITES=0")
    print("FORECAST_LEDGER_WRITES=0")
    print("SERVICE_RESTARTS=0")
    print("CRON_CHANGES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
