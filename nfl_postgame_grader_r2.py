#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1C-R2
Deterministic game-level postgame forecast grader.

Authority contract:
- grade only completed games with final scores
- join on exact game_id
- eligible forecast snapshots must be strictly before kickoff
- require snapshot historical_proof_status == PASS
- require forecast_status to begin with READY
- when multiple eligible pregame snapshots exist, select the latest captured_at_utc
- tie on captured_at_utc fails closed
- no fuzzy matching
- read-only against production databases

Output is stdout only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path("/home/mwynn/nfl_data_engine")
DEFAULT_NFL_DB = ROOT / "data" / "nfl.db"
DEFAULT_LEDGER_DB = ROOT / "data" / "forecast_ledger.db"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REQUIRED_FORECAST_COLUMNS = {
    "snapshot_id",
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "pred_home_margin",
    "pred_total_points",
    "pred_home_points",
    "pred_away_points",
    "pred_winner",
    "model",
    "forecast_variant",
    "forecast_status",
}

REQUIRED_SNAPSHOT_COLUMNS = {
    "snapshot_id",
    "captured_at_utc",
    "historical_proof_status",
    "snapshot_status",
}

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_date",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "completed",
}


def ro_connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def require_table_columns(
    conn: sqlite3.Connection,
    table: str,
    required: set[str],
) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not exists:
        raise RuntimeError(f"missing required table: {table}")
    cols = table_columns(conn, table)
    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(f"{table} missing required columns: {missing}")


def parse_kickoff_utc(game_date: Any, gametime: Any, game_id: str) -> datetime:
    if game_date is None or gametime is None:
        raise RuntimeError(f"{game_id}: missing game_date/gametime")

    text = f"{str(game_date).strip()} {str(gametime).strip()}"
    parsed = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            pass

    if parsed is None:
        raise RuntimeError(f"{game_id}: cannot parse kickoff {text!r}")

    return parsed.replace(tzinfo=ET).astimezone(UTC)


def parse_capture_utc(value: Any, snapshot_id: str) -> datetime:
    if value is None:
        raise RuntimeError(f"{snapshot_id}: captured_at_utc is NULL")
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(
            f"{snapshot_id}: invalid captured_at_utc={value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def as_float(value: Any, field: str, game_id: str) -> float:
    if value is None:
        raise RuntimeError(f"{game_id}: required numeric field {field} is NULL")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{game_id}: invalid numeric field {field}={value!r}"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{game_id}: non-finite {field}={value!r}")
    return result


def winner_from_score(
    away_team: str,
    home_team: str,
    away_score: float,
    home_score: float,
) -> str:
    if home_score > away_score:
        return home_team
    if away_score > home_score:
        return away_team
    return "TIE"


def normalize_pred_winner(
    value: Any,
    away_team: str,
    home_team: str,
) -> str:
    if value is None:
        return "UNAVAILABLE"
    text = str(value).strip().upper()
    if not text:
        return "UNAVAILABLE"
    if text in {away_team.upper(), home_team.upper(), "TIE"}:
        return text
    return text


def round6(value: float) -> float:
    return round(float(value), 6)


def select_authoritative_snapshot(
    ledger: sqlite3.Connection,
    game: sqlite3.Row,
) -> tuple[sqlite3.Row, dict[str, Any]]:
    game_id = str(game["game_id"])
    kickoff_utc = parse_kickoff_utc(
        game["game_date"], game["gametime"], game_id
    )

    rows = ledger.execute(
        """
        SELECT
            p.snapshot_id,
            s.captured_at_utc,
            s.snapshot_status,
            s.historical_proof_status,
            p.game_id,
            p.season,
            p.week,
            p.away_team,
            p.home_team,
            p.pred_home_margin,
            p.pred_total_points,
            p.pred_home_points,
            p.pred_away_points,
            p.pred_winner,
            p.model,
            p.forecast_variant,
            p.forecast_status
        FROM forecast_predictions p
        JOIN forecast_snapshots s
          ON s.snapshot_id = p.snapshot_id
        WHERE p.game_id=?
        ORDER BY s.captured_at_utc, p.snapshot_id
        """,
        (game_id,),
    ).fetchall()

    if not rows:
        raise RuntimeError(f"{game_id}: no forecast snapshots found")

    eligible: list[tuple[datetime, sqlite3.Row]] = []
    rejected_after_kickoff = 0
    rejected_proof = 0
    rejected_status = 0

    for row in rows:
        captured = parse_capture_utc(
            row["captured_at_utc"], str(row["snapshot_id"])
        )

        if captured >= kickoff_utc:
            rejected_after_kickoff += 1
            continue

        if str(row["historical_proof_status"]).strip().upper() != "PASS":
            rejected_proof += 1
            continue

        forecast_status = str(row["forecast_status"] or "").strip().upper()
        if not forecast_status.startswith("READY"):
            rejected_status += 1
            continue

        eligible.append((captured, row))

    if not eligible:
        raise RuntimeError(
            f"{game_id}: no eligible pregame forecast snapshot "
            f"(total={len(rows)}, postkick={rejected_after_kickoff}, "
            f"proof={rejected_proof}, status={rejected_status})"
        )

    eligible.sort(key=lambda x: (x[0], str(x[1]["snapshot_id"])))
    latest_time = eligible[-1][0]
    same_latest = [r for t, r in eligible if t == latest_time]
    if len(same_latest) != 1:
        ids = ",".join(str(r["snapshot_id"]) for r in same_latest)
        raise RuntimeError(
            f"{game_id}: authoritative snapshot timestamp tie: {ids}"
        )

    chosen = same_latest[0]
    meta = {
        "total_snapshots": len(rows),
        "eligible_pregame_snapshots": len(eligible),
        "rejected_at_or_after_kickoff": rejected_after_kickoff,
        "rejected_historical_proof": rejected_proof,
        "rejected_forecast_status": rejected_status,
        "kickoff_utc": kickoff_utc.isoformat(),
        "selected_snapshot_id": chosen["snapshot_id"],
        "selected_captured_at_utc": chosen["captured_at_utc"],
        "selected_minutes_before_kickoff": round6(
            (kickoff_utc - latest_time).total_seconds() / 60.0
        ),
    }
    return chosen, meta


def grade_row(f: sqlite3.Row, g: sqlite3.Row) -> dict[str, Any]:
    game_id = str(g["game_id"])

    for field in ("season", "week", "away_team", "home_team"):
        if f[field] != g[field]:
            raise RuntimeError(
                f"{game_id}: forecast/game mismatch {field}: "
                f"{f[field]!r} != {g[field]!r}"
            )

    if int(g["completed"] or 0) != 1:
        raise RuntimeError(f"{game_id}: game is not completed")

    away_score = as_float(g["away_score"], "away_score", game_id)
    home_score = as_float(g["home_score"], "home_score", game_id)

    pred_home_margin = as_float(
        f["pred_home_margin"], "pred_home_margin", game_id
    )
    pred_total = as_float(
        f["pred_total_points"], "pred_total_points", game_id
    )
    pred_home = as_float(
        f["pred_home_points"], "pred_home_points", game_id
    )
    pred_away = as_float(
        f["pred_away_points"], "pred_away_points", game_id
    )

    actual_home_margin = home_score - away_score
    actual_total = home_score + away_score

    actual_winner = winner_from_score(
        str(g["away_team"]),
        str(g["home_team"]),
        away_score,
        home_score,
    )
    predicted_winner = normalize_pred_winner(
        f["pred_winner"],
        str(g["away_team"]),
        str(g["home_team"]),
    )

    winner_correct = (
        None
        if predicted_winner == "UNAVAILABLE"
        else int(predicted_winner == actual_winner)
    )

    return {
        "snapshot_id": f["snapshot_id"],
        "game_id": game_id,
        "season": int(g["season"]),
        "week": None if g["week"] is None else int(g["week"]),
        "away_team": g["away_team"],
        "home_team": g["home_team"],
        "away_score": round6(away_score),
        "home_score": round6(home_score),
        "actual_winner": actual_winner,
        "pred_winner": predicted_winner,
        "winner_correct": winner_correct,
        "pred_home_margin": round6(pred_home_margin),
        "actual_home_margin": round6(actual_home_margin),
        "margin_error_signed": round6(
            pred_home_margin - actual_home_margin
        ),
        "margin_absolute_error": round6(
            abs(pred_home_margin - actual_home_margin)
        ),
        "pred_total_points": round6(pred_total),
        "actual_total_points": round6(actual_total),
        "total_error_signed": round6(pred_total - actual_total),
        "total_absolute_error": round6(abs(pred_total - actual_total)),
        "pred_home_points": round6(pred_home),
        "home_points_error_signed": round6(pred_home - home_score),
        "home_points_absolute_error": round6(abs(pred_home - home_score)),
        "pred_away_points": round6(pred_away),
        "away_points_error_signed": round6(pred_away - away_score),
        "away_points_absolute_error": round6(abs(pred_away - away_score)),
        "model": f["model"],
        "forecast_variant": f["forecast_variant"],
        "forecast_status": f["forecast_status"],
    }


def semantic_sha(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_week_games(conn, season, week):
    """Exact REG week authority; never hide unfinished games by filtering them out."""
    rows = conn.execute(
        """
        SELECT game_id, season, week, game_date, gametime,
               away_team, home_team, away_score, home_score, completed
        FROM games
        WHERE season=? AND game_type='REG' AND week=?
        ORDER BY game_id
        """,
        (season, week),
    ).fetchall()
    ids = [row["game_id"] for row in rows]
    if (
        not rows
        or any(not isinstance(gid, str) or not gid.strip() for gid in ids)
        or len(set(ids)) != len(ids)
        or any(
            row["completed"] != 1
            or row["away_score"] is None
            or row["home_score"] is None
            for row in rows
        )
    ):
        raise RuntimeError(f"Season {season} week {week}: incomplete or invalid REG game set")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--game-id")
    parser.add_argument("--nfl-db", type=Path, default=DEFAULT_NFL_DB)
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER_DB)
    args = parser.parse_args()
    if args.week is not None:
        if args.season is None or args.season < 1 or args.week < 1:
            parser.error("--week requires an explicit positive --season and positive --week")
        if args.game_id is not None:
            parser.error("--week and --game-id are mutually exclusive")
    if args.season is None:
        args.season = 2026  # Preserve the existing unscoped default.


    print("=" * 70)
    print("WFS NFL — NFL-POSTGAME-1C-R2")
    print("DETERMINISTIC GAME FORECAST GRADER — READ ONLY")
    print("=" * 70)

    with ro_connect(args.nfl_db) as nfl, ro_connect(args.ledger_db) as ledger:
        nfl_integrity = nfl.execute("PRAGMA integrity_check").fetchone()[0]
        ledger_integrity = ledger.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"NFL_DB_INTEGRITY={nfl_integrity}")
        print(f"LEDGER_DB_INTEGRITY={ledger_integrity}")

        if nfl_integrity != "ok" or ledger_integrity != "ok":
            raise RuntimeError("database integrity check failed")

        require_table_columns(nfl, "games", REQUIRED_GAME_COLUMNS)
        require_table_columns(
            ledger, "forecast_predictions", REQUIRED_FORECAST_COLUMNS
        )
        require_table_columns(
            ledger, "forecast_snapshots", REQUIRED_SNAPSHOT_COLUMNS
        )

        sql = """
            SELECT
                game_id, season, week, game_date, gametime,
                away_team, home_team, away_score, home_score, completed
            FROM games
            WHERE season=? AND completed=1
              AND away_score IS NOT NULL
              AND home_score IS NOT NULL
        """
        params: list[Any] = [args.season]
        if args.game_id:
            sql += " AND game_id=?"
            params.append(args.game_id)
        sql += " ORDER BY week, game_id"

        if args.week is None:
            games = nfl.execute(sql, params).fetchall()
        else:
            games = select_week_games(nfl, args.season, args.week)
            print("WEEK_SCOPE=" + json.dumps({
                "season": args.season,
                "week": args.week,
                "game_ids": [row["game_id"] for row in games],
            }, sort_keys=True, separators=(",", ":")))
        print(f"COMPLETED_FINAL_GAMES={len(games)}")

        if not games:
            raise RuntimeError("no completed final games found")

        graded: list[dict[str, Any]] = []

        for g in games:
            selected, authority = select_authoritative_snapshot(ledger, g)
            print(
                "AUTHORITY="
                + json.dumps(
                    authority,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            graded.append(grade_row(selected, g))

        print(f"GRADED_ROWS={len(graded)}")

        if len(graded) != len(games):
            raise RuntimeError("not every completed game was graded")

        for row in graded:
            print(
                "GRADE="
                + json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

        winner_rows = [
            r for r in graded if r["winner_correct"] is not None
        ]
        winner_correct = sum(r["winner_correct"] for r in winner_rows)

        mean_margin_abs = sum(
            r["margin_absolute_error"] for r in graded
        ) / len(graded)
        mean_total_abs = sum(
            r["total_absolute_error"] for r in graded
        ) / len(graded)
        mean_home_abs = sum(
            r["home_points_absolute_error"] for r in graded
        ) / len(graded)
        mean_away_abs = sum(
            r["away_points_absolute_error"] for r in graded
        ) / len(graded)

        print(f"WINNER_GRADED={len(winner_rows)}")
        print(f"WINNER_CORRECT={winner_correct}")
        if winner_rows:
            print(
                "WINNER_ACCURACY="
                f"{winner_correct / len(winner_rows):.6f}"
            )
        else:
            print("WINNER_ACCURACY=NA")

        print(f"MEAN_MARGIN_ABS_ERROR={mean_margin_abs:.6f}")
        print(f"MEAN_TOTAL_ABS_ERROR={mean_total_abs:.6f}")
        print(f"MEAN_HOME_POINTS_ABS_ERROR={mean_home_abs:.6f}")
        print(f"MEAN_AWAY_POINTS_ABS_ERROR={mean_away_abs:.6f}")

        digest = semantic_sha(graded)
        print(f"SEMANTIC_SHA256={digest}")

    print("NFL_POSTGAME_1C_R2_STATUS=PASS")
    print("READ_ONLY=TRUE")
    print("PRODUCTION_DATABASE_WRITES=0")
    print("SERVICE_RESTARTS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
