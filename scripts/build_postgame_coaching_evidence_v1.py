#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from live_ingest import fetch_json


ROOT = Path(__file__).resolve().parents[1]

LIVE_DIR = (
    ROOT
    / "processed"
    / "live_game_intelligence"
)

OUT_DIR = (
    ROOT
    / "processed"
    / "coaching_intelligence"
)

LEDGER_CSV = (
    OUT_DIR
    / "postgame_coaching_evidence_v1.csv"
)

LEDGER_JSON = (
    OUT_DIR
    / "postgame_coaching_evidence_v1.json"
)

VERSION = "WFS_POSTGAME_COACHING_EVIDENCE_V1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_summary(event_id: str) -> dict:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        f"football/nfl/summary?event={event_id}"
    )

    result = fetch_json(url)

    if isinstance(result, dict):
        return result

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict):
                return item

    raise RuntimeError(
        "ESPN summary payload not found"
    )


def event_state(data: dict) -> tuple[str, bool]:
    header = data.get("header") or {}
    comps = header.get("competitions") or []

    if not comps:
        return "", False

    status = comps[0].get("status") or {}
    status_type = status.get("type") or {}

    state = str(
        status_type.get("state") or ""
    ).strip().lower()

    completed = bool(
        status_type.get("completed")
    )

    return state, completed


def confidence(sample_size: int) -> str:
    if sample_size >= 50:
        return "HIGH"

    if sample_size >= 30:
        return "MEDIUM"

    return "LOW"


def top_usage_text(items) -> str:
    if not items:
        return ""

    out = []

    for item in items:
        if not isinstance(item, (list, tuple)):
            continue

        if len(item) != 2:
            continue

        name, count = item

        out.append(
            f"{name}:{count}"
        )

    return "|".join(out)


def row_from_team(
    payload: dict,
    team: dict,
) -> dict:

    scrimmage_plays = int(
        team.get("scrimmage_plays") or 0
    )

    expected = team.get(
        "expected_pass_rate"
    )

    live = team.get(
        "live_pass_rate"
    )

    delta = team.get(
        "pass_rate_delta"
    )

    return {
        "version": VERSION,

        "event_id": str(
            payload.get("event_id") or ""
        ),

        "game_id": str(
            payload.get("game_id") or ""
        ),

        "team": str(
            team.get("team") or ""
        ).upper(),

        "expected_pass_rate": expected,

        "final_pass_rate": live,

        "pass_rate_delta": delta,

        "early_down_pass_rate":
            team.get(
                "early_down_pass_rate"
            ),

        "red_zone_pass_rate":
            team.get(
                "red_zone_pass_rate"
            ),

        "third_down_pass_rate":
            team.get(
                "third_down_pass_rate"
            ),

        "short_yardage_pass_rate":
            team.get(
                "short_yardage_pass_rate"
            ),

        "shotgun_rate":
            team.get(
                "shotgun_rate"
            ),

        "no_huddle_rate":
            team.get(
                "no_huddle_rate"
            ),

        "fourth_down_scrimmage":
            int(
                team.get(
                    "fourth_down_scrimmage"
                )
                or 0
            ),

        "scrimmage_plays":
            scrimmage_plays,

        "pass_calls":
            int(
                team.get(
                    "pass_calls"
                )
                or 0
            ),

        "rush_calls":
            int(
                team.get(
                    "rush_calls"
                )
                or 0
            ),

        "target_concentration":
            top_usage_text(
                team.get(
                    "top_targets"
                )
                or []
            ),

        "carry_concentration":
            top_usage_text(
                team.get(
                    "top_carries"
                )
                or []
            ),

        "confidence":
            confidence(
                scrimmage_plays
            ),

        "interpretation":
            str(
                team.get(
                    "interpretation"
                )
                or ""
            ),

        "generated_at_utc":
            utc_now(),
    }


def load_existing() -> pd.DataFrame:
    if not LEDGER_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            LEDGER_CSV
        )

    except Exception:
        return pd.DataFrame()


def write_ledger(
    rows: list[dict],
) -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_df = pd.DataFrame(rows)

    old_df = load_existing()

    if old_df.empty:
        combined = new_df.copy()

    else:
        combined = pd.concat(
            [
                old_df,
                new_df,
            ],
            ignore_index=True,
        )

    if not combined.empty:
        # Normalize exact ledger identity keys BEFORE deduplication.
        #
        # Existing CSV rows may load event_id as integers while freshly
        # generated rows may arrive as strings. Pandas considers those
        # different values during drop_duplicates even though CSV reload
        # later normalizes them to the same integer.
        combined["event_id"] = (
            combined["event_id"]
            .astype(str)
            .str.strip()
        )

        combined["team"] = (
            combined["team"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        combined = combined.drop_duplicates(
            subset=[
                "event_id",
                "team",
            ],
            keep="last",
        )

        # Fail closed: one and only one row per event/team.
        duplicate_keys = combined[
            combined.duplicated(
                subset=[
                    "event_id",
                    "team",
                ],
                keep=False,
            )
        ]

        if not duplicate_keys.empty:
            print(
                "FAIL: duplicate postgame evidence keys remain"
            )
            print(
                duplicate_keys[
                    [
                        "event_id",
                        "game_id",
                        "team",
                    ]
                ].to_string(
                    index=False
                )
            )
            raise SystemExit(1)

        combined = combined.sort_values(
            [
                "game_id",
                "team",
            ],
            kind="stable",
        ).reset_index(drop=True)

    combined.to_csv(
        LEDGER_CSV,
        index=False,
    )

    records = combined.to_dict(
        orient="records"
    )

    LEDGER_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "generated_at_utc":
                    utc_now(),
                "rows": records,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


def process_event(
    event_id: str,
) -> list[dict]:

    paper_path = (
        LIVE_DIR
        / f"{event_id}.json"
    )

    if not paper_path.exists():
        raise RuntimeError(
            f"missing game intelligence JSON: "
            f"{paper_path}"
        )

    summary = fetch_summary(
        event_id
    )

    state, completed = event_state(
        summary
    )

    if (
        state != "post"
        and not completed
    ):
        raise RuntimeError(
            f"event {event_id} not final "
            f"(state={state}, "
            f"completed={completed})"
        )

    payload = json.loads(
        paper_path.read_text()
    )

    rows = []

    for team in (
        payload.get("teams") or []
    ):
        rows.append(
            row_from_team(
                payload,
                team,
            )
        )

    if len(rows) != 2:
        raise RuntimeError(
            f"expected 2 team rows, "
            f"got {len(rows)}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--event-id",
        action="append",
        dest="event_ids",
        required=True,
    )

    args = parser.parse_args()

    all_rows = []

    print("=" * 72)
    print(
        "WFS POSTGAME COACHING "
        "EVIDENCE V1"
    )
    print("=" * 72)

    for event_id in args.event_ids:
        event_id = str(
            event_id
        ).strip()

        print(
            f">>> {event_id}"
        )

        rows = process_event(
            event_id
        )

        all_rows.extend(
            rows
        )

        for row in rows:
            print(
                row["team"],
                "| final pass",
                (
                    f"{row['final_pass_rate'] * 100:.1f}%"
                    if row[
                        "final_pass_rate"
                    ]
                    is not None
                    else "—"
                ),
                "| delta",
                (
                    f"{row['pass_rate_delta'] * 100:+.1f}%"
                    if row[
                        "pass_rate_delta"
                    ]
                    is not None
                    else "—"
                ),
                "|",
                row["confidence"],
            )

    write_ledger(
        all_rows
    )

    print()
    print(
        "CSV :",
        LEDGER_CSV,
    )
    print(
        "JSON:",
        LEDGER_JSON,
    )
    print("PASS")


if __name__ == "__main__":
    main()
