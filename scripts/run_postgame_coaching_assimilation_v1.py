#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from live_ingest import fetch_json


ROOT = Path(__file__).resolve().parents[1]

GAME_INTELLIGENCE_BUILDER = (
    ROOT
    / "scripts"
    / "build_live_game_intelligence_v1.py"
)

POSTGAME_LEDGER_BUILDER = (
    ROOT
    / "scripts"
    / "build_postgame_coaching_evidence_v1.py"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "football/nfl/scoreboard"
)

VERSION = "WFS_POSTGAME_COACHING_ASSIMILATION_V1"


def fetch_scoreboard() -> dict:
    result = fetch_json(
        SCOREBOARD_URL
    )

    if isinstance(result, dict):
        return result

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict):
                return item

    raise RuntimeError(
        "ESPN scoreboard payload not found"
    )


def final_events(
    data: dict,
) -> list[dict]:
    out = []

    for event in (
        data.get("events") or []
    ):
        event_id = str(
            event.get("id") or ""
        ).strip()

        if not event_id:
            continue

        name = str(
            event.get("name") or ""
        ).strip()

        status = (
            event.get("status") or {}
        )

        status_type = (
            status.get("type") or {}
        )

        state = str(
            status_type.get("state")
            or ""
        ).strip().lower()

        completed = bool(
            status_type.get(
                "completed"
            )
        )

        if (
            state == "post"
            and completed
        ):
            out.append(
                {
                    "event_id": event_id,
                    "name": name,
                }
            )

    return sorted(
        out,
        key=lambda x: x[
            "event_id"
        ],
    )


def run_command(
    command: list[str],
) -> int:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
    )

    return int(
        proc.returncode
    )


def main() -> int:
    print("=" * 72)
    print(VERSION)
    print("=" * 72)

    data = fetch_scoreboard()

    events = final_events(
        data
    )

    if not events:
        print(
            "NO FINAL GAMES AVAILABLE"
        )
        return 0

    print(
        "FINAL EVENTS:",
        ", ".join(
            x["event_id"]
            for x in events
        ),
    )

    print()

    failures = []

    for event in events:
        event_id = event[
            "event_id"
        ]

        name = event[
            "name"
        ]

        print("=" * 72)
        print(
            f">>> {event_id} | {name}"
        )
        print("=" * 72)

        print(
            "[1/2] Regenerating "
            "final game intelligence"
        )

        rc = run_command(
            [
                sys.executable,
                str(
                    GAME_INTELLIGENCE_BUILDER
                ),
                "--event-id",
                event_id,
            ]
        )

        if rc != 0:
            print(
                "FAIL: final game "
                "intelligence"
            )

            failures.append(
                {
                    "event_id":
                        event_id,
                    "stage":
                        "game_intelligence",
                }
            )

            print()
            continue

        print(
            "[2/2] Assimilating "
            "postgame coaching evidence"
        )

        rc = run_command(
            [
                sys.executable,
                str(
                    POSTGAME_LEDGER_BUILDER
                ),
                "--event-id",
                event_id,
            ]
        )

        if rc != 0:
            print(
                "FAIL: postgame "
                "coaching evidence"
            )

            failures.append(
                {
                    "event_id":
                        event_id,
                    "stage":
                        "postgame_evidence",
                }
            )

            print()
            continue

        print(
            f"PASS: {event_id}"
        )
        print()

    print("=" * 72)

    if failures:
        print(
            "POSTGAME ASSIMILATION "
            "COMPLETED WITH FAILURES"
        )

        for item in failures:
            print(
                item["event_id"],
                "|",
                item["stage"],
            )

        return 1

    print(
        "ALL FINAL GAMES "
        "ASSIMILATED PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
