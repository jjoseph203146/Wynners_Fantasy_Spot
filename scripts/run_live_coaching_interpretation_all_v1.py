#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BUILDER = (
    ROOT
    / "scripts"
    / "build_live_coaching_interpretation_v1.py"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/scoreboard"
)

VERSION = "WFS_LIVE_COACHING_INTERPRETATION_ALL_V1"


def fetch_scoreboard() -> dict:
    request = urllib.request.Request(
        SCOREBOARD_URL,
        headers={
            "User-Agent":
                "Mozilla/5.0 WFS-LIVE-RUNNER",

            "Accept":
                "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=20,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def active_events(data: dict) -> list[str]:
    result = []

    for event in data.get(
        "events",
        [],
    ):
        event_id = str(
            event.get(
                "id",
                "",
            )
        ).strip()

        if not event_id:
            continue

        status = (
            event.get(
                "status",
                {},
            )
            .get(
                "type",
                {},
            )
        )

        state = str(
            status.get(
                "state",
                "",
            )
        ).strip().lower()

        if state == "in":
            result.append(
                event_id
            )

    return sorted(
        set(result)
    )


def run_event(
    event_id: str,
) -> None:
    print(
        f">>> {event_id}"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            event_id,
        ],
        cwd=str(ROOT),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"live coaching interpretation "
            f"failed for {event_id}; "
            f"rc={proc.returncode}"
        )


def main() -> int:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    if not BUILDER.exists():
        print(
            f"FAIL: missing builder: {BUILDER}"
        )
        return 1

    try:
        scoreboard = fetch_scoreboard()

        events = active_events(
            scoreboard
        )

        print(
            "ACTIVE EVENTS:",
            (
                ", ".join(events)
                if events
                else "NONE"
            ),
        )

        for event_id in events:
            run_event(
                event_id
            )

    except Exception as exc:
        print()
        print("FAIL:", exc)
        return 1

    print()
    print(
        "ALL ACTIVE COACHING INTERPRETATIONS PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
