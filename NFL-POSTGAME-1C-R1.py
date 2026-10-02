#!/usr/bin/env python3
"""
WFS NFL — NFL-POSTGAME-1C-R1
Forecast snapshot authority audit for one completed game.

READ ONLY:
- forecast_ledger.db mode=ro
- nfl.db mode=ro
- no production writes
- no service interaction

Purpose:
Determine why multiple immutable forecast snapshots exist for a game and identify
the deterministic selection contract needed by the postgame grader.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/home/mwynn/nfl_data_engine")
NFL_DB = ROOT / "data" / "nfl.db"
LEDGER_DB = ROOT / "data" / "forecast_ledger.db"


def ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing database: {path}")
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def parse_game_kickoff_et(game_date, gametime):
    if not game_date or not gametime:
        return None
    text = f"{game_date} {gametime}"
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
            return naive.replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            pass
    return None


def parse_utc(text):
    if not text:
        return None
    value = str(text).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("UTC"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game-id", required=True)
    args = p.parse_args()

    print("=" * 70)
    print("WFS NFL — NFL-POSTGAME-1C-R1")
    print("FORECAST SNAPSHOT AUTHORITY AUDIT — READ ONLY")
    print("=" * 70)

    with ro(NFL_DB) as nfl, ro(LEDGER_DB) as led:
        print("NFL_DB_INTEGRITY=" + nfl.execute("PRAGMA integrity_check").fetchone()[0])
        print("LEDGER_DB_INTEGRITY=" + led.execute("PRAGMA integrity_check").fetchone()[0])

        g = nfl.execute("""
            SELECT game_id, season, week, game_date, gametime,
                   away_team, home_team, away_score, home_score, completed
            FROM games
            WHERE game_id=?
        """, (args.game_id,)).fetchone()

        if not g:
            raise RuntimeError("game_id not found in nfl.db")

        print(
            "GAME="
            f"{g['game_id']}|season={g['season']}|week={g['week']}|"
            f"{g['away_team']}={g['away_score']}|{g['home_team']}={g['home_score']}|"
            f"completed={g['completed']}|date={g['game_date']}|time={g['gametime']}"
        )

        kickoff_et = parse_game_kickoff_et(g["game_date"], g["gametime"])
        kickoff_utc = kickoff_et.astimezone(ZoneInfo("UTC")) if kickoff_et else None
        print("KICKOFF_ET=" + (kickoff_et.isoformat() if kickoff_et else "UNAVAILABLE"))
        print("KICKOFF_UTC=" + (kickoff_utc.isoformat() if kickoff_utc else "UNAVAILABLE"))

        rows = led.execute("""
            SELECT
                p.snapshot_id,
                s.captured_at_utc,
                s.snapshot_status,
                s.model AS snapshot_model,
                s.forecast_variant AS snapshot_variant,
                s.source_prediction_sha256,
                s.source_audit_sha256,
                s.historical_proof_status,
                s.injury_source_status,
                s.win_probability_status,
                p.game_id,
                p.season,
                p.week,
                p.away_team,
                p.home_team,
                p.market_home_spread_raw,
                p.market_total,
                p.forecast_market_ready,
                p.pred_home_margin,
                p.pred_total_points,
                p.pred_home_points,
                p.pred_away_points,
                p.pred_winner,
                p.model,
                p.forecast_variant,
                p.impact_status,
                p.replacement_status,
                p.win_probability_status AS prediction_win_probability_status,
                p.forecast_status
            FROM forecast_predictions p
            JOIN forecast_snapshots s
              ON s.snapshot_id = p.snapshot_id
            WHERE p.game_id=?
            ORDER BY s.captured_at_utc, p.snapshot_id
        """, (args.game_id,)).fetchall()

        print(f"SNAPSHOT_ROWS={len(rows)}")
        if not rows:
            raise RuntimeError("no forecast snapshots found")

        pre = []
        post = []
        unparsable = []

        for i, r in enumerate(rows, 1):
            captured = parse_utc(r["captured_at_utc"])
            if captured is None:
                relation = "UNPARSABLE"
                unparsable.append(r["snapshot_id"])
            elif kickoff_utc is None:
                relation = "KICKOFF_UNAVAILABLE"
            elif captured < kickoff_utc:
                relation = "PREGAME"
                pre.append((captured, r))
            else:
                relation = "AT_OR_AFTER_KICKOFF"
                post.append((captured, r))

            delta_min = None
            if captured is not None and kickoff_utc is not None:
                delta_min = (kickoff_utc - captured).total_seconds() / 60.0

            print()
            print(f"SNAPSHOT_{i}")
            print(f"snapshot_id={r['snapshot_id']}")
            print(f"captured_at_utc={r['captured_at_utc']}")
            print(f"kickoff_relation={relation}")
            print(
                "minutes_before_kickoff="
                + ("NA" if delta_min is None else f"{delta_min:.3f}")
            )
            print(f"snapshot_status={r['snapshot_status']}")
            print(f"historical_proof_status={r['historical_proof_status']}")
            print(f"injury_source_status={r['injury_source_status']}")
            print(f"snapshot_win_probability_status={r['win_probability_status']}")
            print(f"prediction_forecast_status={r['forecast_status']}")
            print(f"impact_status={r['impact_status']}")
            print(f"replacement_status={r['replacement_status']}")
            print(
                "prediction_win_probability_status="
                f"{r['prediction_win_probability_status']}"
            )
            print(f"model={r['model']}")
            print(f"forecast_variant={r['forecast_variant']}")
            print(f"pred_home_margin={r['pred_home_margin']}")
            print(f"pred_total_points={r['pred_total_points']}")
            print(f"pred_home_points={r['pred_home_points']}")
            print(f"pred_away_points={r['pred_away_points']}")
            print(f"pred_winner={r['pred_winner']}")
            print(f"market_home_spread_raw={r['market_home_spread_raw']}")
            print(f"market_total={r['market_total']}")
            print(f"forecast_market_ready={r['forecast_market_ready']}")
            print(f"source_prediction_sha256={r['source_prediction_sha256']}")
            print(f"source_audit_sha256={r['source_audit_sha256']}")

        print()
        print("=== AUTHORITY CANDIDATE SUMMARY ===")
        print(f"PREGAME_SNAPSHOTS={len(pre)}")
        print(f"AT_OR_AFTER_KICKOFF_SNAPSHOTS={len(post)}")
        print(f"UNPARSABLE_CAPTURE_TIMES={len(unparsable)}")

        if pre:
            pre.sort(key=lambda x: (x[0], str(x[1]["snapshot_id"])))
            latest_time, latest = pre[-1]
            print(f"LATEST_PREGAME_SNAPSHOT_ID={latest['snapshot_id']}")
            print(f"LATEST_PREGAME_CAPTURED_AT_UTC={latest['captured_at_utc']}")
            print(
                "LATEST_PREGAME_MINUTES_BEFORE_KICKOFF="
                f"{(kickoff_utc - latest_time).total_seconds()/60.0:.3f}"
            )
            print(f"LATEST_PREGAME_SNAPSHOT_STATUS={latest['snapshot_status']}")
            print(f"LATEST_PREGAME_FORECAST_STATUS={latest['forecast_status']}")
            print(f"LATEST_PREGAME_MODEL={latest['model']}")
            print(f"LATEST_PREGAME_VARIANT={latest['forecast_variant']}")
        else:
            print("LATEST_PREGAME_SNAPSHOT_ID=NONE")

        print()
        print("NFL_POSTGAME_1C_R1_STATUS=PASS")
        print("READ_ONLY=TRUE")
        print("PRODUCTION_DATABASE_WRITES=0")
        print("SERVICE_RESTARTS=0")


if __name__ == "__main__":
    main()
