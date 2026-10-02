#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from live_ingest import fetch_json


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_live_game_intelligence_v1.py"

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "football/nfl/scoreboard"
)


def fetch_scoreboard() -> dict:
    result = fetch_json(SCOREBOARD_URL)

    if isinstance(result, dict):
        return result

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict):
                return item

    raise RuntimeError("ESPN scoreboard payload not found")


def active_events(data: dict) -> list[str]:
    out = []

    for event in data.get("events") or []:
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue

        status = event.get("status") or {}
        status_type = status.get("type") or {}

        state = str(
            status_type.get("state") or ""
        ).strip().lower()

        if state == "in":
            out.append(event_id)

    return sorted(set(out))


def main():
    data = fetch_scoreboard()
    events = active_events(data)

    print("=" * 72)
    print("WFS LIVE GAME INTELLIGENCE — ALL ACTIVE GAMES")
    print("=" * 72)

    if not events:
        print("NO ACTIVE GAMES")
        return 0

    print("ACTIVE EVENTS:", ", ".join(events))
    print()

    failures = []

    for event_id in events:
        print(f">>> {event_id}")

        proc = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--event-id",
                event_id,
            ],
            cwd=str(ROOT),
        )

        if proc.returncode != 0:
            failures.append(event_id)

        print()

    if failures:
        print(
            "FAILURES:",
            ", ".join(failures),
        )
        return 1

    print("ALL ACTIVE GAME PAPERS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
