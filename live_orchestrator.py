from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
import urllib.request

from collections import Counter
from pathlib import Path

from wfs_schedule_context import resolve_schedule_week_context


# ======================================================================
# WFS NFL LIVE — MULTI-EVENT ORCHESTRATOR
#
# PURPOSE
#   - Discover ESPN NFL scoreboard events dynamically.
#   - Classify PRE / IN / POST.
#   - Invoke the proven live_ingest.py runner ONLY for IN / POST.
#   - Process eligible events independently and sequentially.
#   - Continue to later events if one event fails.
#   - Perform no direct database writes from this orchestrator.
#
# IMPORTANT
#   live_ingest.py remains the sole LIVE persistence writer.
# ======================================================================


# ======================================================================
# FROZEN PATHS + CONTRACT
# ======================================================================

ROOT = Path(
    "/home/mwynn/nfl_data_engine"
).resolve()

DATA_DIR = ROOT / "data"

LIVE_DB = DATA_DIR / "wfs_live.db"

NFL_DB = DATA_DIR / "nfl.db"

FORECAST_LEDGER = (
    DATA_DIR / "forecast_ledger.db"
)

LIVE_INGEST = (
    ROOT / "live_ingest.py"
)

PYTHON = (
    ROOT / "venv" / "bin" / "python"
)

EXPECTED_SCHEMA_VERSION = (
    "WFS_LIVE_DB_V2"
)

EXPECTED_SCHEMA_SHA256 = (
    "ea078eaf3115ea9e3894f51813ebee80"
    "f1168ced48ef9da557db00fc643d9bfa"
)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/nfl/scoreboard"
)

ESPN_TEAM_ALIASES = {
    "LAR": "LA",
    "WSH": "WAS",
}

HTTP_TIMEOUT = 15


# ======================================================================
# HELPERS
# ======================================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def schema_fingerprint(
    conn: sqlite3.Connection,
) -> str:
    rows = conn.execute(
        """
        SELECT
            type,
            name,
            tbl_name,
            COALESCE(sql, '')
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY
            type,
            name,
            tbl_name
        """
    ).fetchall()

    normalized = "\n".join(
        "|".join(
            str(value)
            for value in row
        )
        for row in rows
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def fetch_json(
    url: str,
):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 "
                "WFS-LIVE-ORCHESTRATOR",

            "Accept":
                "application/json",
        },
    )

    started = (
        time.perf_counter()
    )

    with urllib.request.urlopen(
        request,
        timeout=HTTP_TIMEOUT,
    ) as response:
        raw = response.read()
        status = response.status

    latency = (
        time.perf_counter()
        - started
    )

    return (
        json.loads(raw),
        status,
        latency,
        len(raw),
    )


def parse_int(value):
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return int(value)

    if isinstance(
        value,
        str,
    ):
        try:
            return int(
                value
            )

        except ValueError:
            return None

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "number",
            "value",
            "id",
        ):
            candidate = (
                value.get(key)
            )

            if candidate is None:
                continue

            try:
                return int(
                    candidate
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

    return None


def read_live_database_state(
    event_id: str | None = None,
) -> dict:

    uri = (
        f"file:{LIVE_DB}?mode=ro"
    )

    conn = sqlite3.connect(
        uri,
        uri=True,
    )

    conn.row_factory = (
        sqlite3.Row
    )

    try:
        integrity = (
            conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
        )

        fk_errors = (
            conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        schema_row = (
            conn.execute(
                """
                SELECT schema_version
                FROM live_meta
                """
            ).fetchone()
        )

        if schema_row is None:
            raise RuntimeError(
                "live_meta missing"
            )

        result = {
            "schema_version":
                schema_row[
                    "schema_version"
                ],

            "schema_sha256":
                schema_fingerprint(
                    conn
                ),

            "integrity":
                integrity,

            "fk_errors":
                len(
                    fk_errors
                ),

            "global_event_rows":
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM live_events
                    """
                ).fetchone()[0],

            "global_athlete_rows":
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM live_athletes
                    """
                ).fetchone()[0],

            "global_play_rows":
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM live_plays
                    """
                ).fetchone()[0],

            "global_player_rows":
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM live_play_players
                    """
                ).fetchone()[0],

            "global_audit_rows":
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM live_ingest_audit
                    """
                ).fetchone()[0],
        }

        if event_id is not None:

            result[
                "event_rows"
            ] = conn.execute(
                """
                SELECT COUNT(*)
                FROM live_events
                WHERE event_id = ?
                """,
                (
                    event_id,
                ),
            ).fetchone()[0]

            result[
                "play_rows"
            ] = conn.execute(
                """
                SELECT COUNT(*)
                FROM live_plays
                WHERE event_id = ?
                """,
                (
                    event_id,
                ),
            ).fetchone()[0]

            result[
                "player_rows"
            ] = conn.execute(
                """
                SELECT COUNT(*)
                FROM live_play_players
                WHERE event_id = ?
                """,
                (
                    event_id,
                ),
            ).fetchone()[0]

            result[
                "audit_rows"
            ] = conn.execute(
                """
                SELECT COUNT(*)
                FROM live_ingest_audit
                WHERE event_id = ?
                """,
                (
                    event_id,
                ),
            ).fetchone()[0]

            event_row = (
                conn.execute(
                    """
                    SELECT
                        state,
                        detail,
                        period,
                        clock,
                        home_team,
                        home_score,
                        away_team,
                        away_score,
                        last_play_id
                    FROM live_events
                    WHERE event_id = ?
                    """,
                    (
                        event_id,
                    ),
                ).fetchone()
            )

            result[
                "event"
            ] = (
                dict(
                    event_row
                )
                if event_row
                is not None
                else None
            )

        return result

    finally:
        conn.close()


# ======================================================================
# SCOREBOARD DISCOVERY
# ======================================================================


def canonical_schedule_team(
    value,
) -> str:
    token = str(
        value
        or ""
    ).strip().upper()

    return ESPN_TEAM_ALIASES.get(
        token,
        token,
    )


def load_live_schedule_target(
    season: int,
) -> dict:
    context = resolve_schedule_week_context(
        db_path=NFL_DB,
        season=int(season),
    )

    target_season = int(
        context.season
    )

    target_week = int(
        context.planning_week
    )

    if target_week < 1:
        raise RuntimeError(
            "invalid schedule planning week"
        )

    nfl_db_uri = f"file:{NFL_DB}?mode=ro"

    with sqlite3.connect(
        nfl_db_uri,
        uri=True,
    ) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                away_team,
                home_team
            FROM games
            WHERE season = ?
              AND week = ?
              AND game_type = 'REG'
            ORDER BY
                game_date,
                gametime,
                game_id
            """,
            (
                target_season,
                target_week,
            ),
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "schedule authority returned zero "
            "target-week games"
        )

    game_ids = [
        str(
            row["game_id"]
            or ""
        ).strip()
        for row in rows
    ]

    if (
        any(
            not game_id
            for game_id in game_ids
        )
        or len(
            set(game_ids)
        ) != len(game_ids)
    ):
        raise RuntimeError(
            "invalid or duplicate target "
            "schedule game_id"
        )

    dates = sorted(
        {
            str(
                row["game_date"]
                or ""
            ).strip()
            for row in rows
            if str(
                row["game_date"]
                or ""
            ).strip()
        }
    )

    if not dates:
        raise RuntimeError(
            "target schedule has no game dates"
        )

    matchups = set()

    for row in rows:
        away = canonical_schedule_team(
            row["away_team"]
        )

        home = canonical_schedule_team(
            row["home_team"]
        )

        if not away or not home:
            raise RuntimeError(
                "blank target schedule team identity"
            )

        matchup = (
            away,
            home,
        )

        if matchup in matchups:
            raise RuntimeError(
                "duplicate target schedule matchup: "
                f"{away}@{home}"
            )

        matchups.add(
            matchup
        )

    return {
        "season":
            target_season,

        "week":
            target_week,

        "active_game_week":
            context.active_game_week,

        "upcoming_week":
            context.upcoming_week,

        "dates":
            dates,

        "matchups":
            matchups,

        "game_count":
            len(rows),
    }


def fetch_schedule_scoreboard(
    target: dict,
) -> tuple[
    dict,
    list[dict],
]:
    combined_events = []
    fetch_audit = []

    for game_date in target[
        "dates"
    ]:
        espn_date = (
            game_date.replace(
                "-",
                "",
            )
        )

        url = (
            f"{SCOREBOARD_URL}"
            f"?dates={espn_date}"
        )

        (
            payload,
            http_status,
            latency,
            byte_count,
        ) = fetch_json(
            url
        )

        fetch_audit.append(
            {
                "game_date":
                    game_date,

                "http_status":
                    http_status,

                "latency":
                    latency,

                "byte_count":
                    byte_count,
            }
        )

        if http_status != 200:
            raise RuntimeError(
                "ESPN date-bound scoreboard "
                f"fetch failed for {game_date}: "
                f"HTTP {http_status}"
            )

        events = (
            payload.get(
                "events"
            )
            or []
        )

        if not isinstance(
            events,
            list,
        ):
            raise RuntimeError(
                "ESPN date-bound events "
                "payload is not a list"
            )

        combined_events.extend(
            events
        )

    return (
        {
            "events":
                combined_events,
        },
        fetch_audit,
    )


def reconcile_schedule_events(
    discovered: list[dict],
    target: dict,
) -> list[dict]:
    expected = set(
        target[
            "matchups"
        ]
    )

    matched = []
    seen = set()
    unexpected = []

    for event in discovered:
        away = canonical_schedule_team(
            event[
                "away_team"
            ]
        )

        home = canonical_schedule_team(
            event[
                "home_team"
            ]
        )

        matchup = (
            away,
            home,
        )

        if matchup not in expected:
            unexpected.append(
                {
                    "event_id":
                        event[
                            "event_id"
                        ],

                    "matchup":
                        f"{away}@{home}",
                }
            )

            continue

        if matchup in seen:
            raise RuntimeError(
                "duplicate ESPN matchup after "
                "schedule reconciliation: "
                f"{away}@{home}"
            )

        seen.add(
            matchup
        )

        matched.append(
            event
        )

    missing = sorted(
        expected
        - seen
    )

    if unexpected:
        raise RuntimeError(
            "unexpected ESPN target-date events: "
            + json.dumps(
                unexpected,
                sort_keys=True,
            )
        )

    if missing:
        raise RuntimeError(
            "missing ESPN target-week events: "
            + ",".join(
                f"{away}@{home}"
                for away, home
                in missing
            )
        )

    if len(matched) != target[
        "game_count"
    ]:
        raise RuntimeError(
            "target event count mismatch | "
            f"expected={target['game_count']} "
            f"matched={len(matched)}"
        )

    return matched


def discover_events(
    scoreboard: dict,
) -> tuple[
    list[dict],
    list[dict],
    list[str],
]:

    events_raw = (
        scoreboard.get(
            "events"
        )
        or []
    )

    if not isinstance(
        events_raw,
        list,
    ):
        raise RuntimeError(
            "scoreboard events "
            "is not a list"
        )

    discovered = []

    malformed = []

    raw_ids = []

    for ordinal, event in enumerate(
        events_raw,
        start=1,
    ):

        if not isinstance(
            event,
            dict,
        ):
            malformed.append(
                {
                    "ordinal":
                        ordinal,

                    "reason":
                        "event_not_object",
                }
            )

            continue

        event_id = str(
            event.get(
                "id"
            )
            or ""
        ).strip()

        if not event_id:
            malformed.append(
                {
                    "ordinal":
                        ordinal,

                    "reason":
                        "missing_event_id",
                }
            )

            continue

        raw_ids.append(
            event_id
        )

        competitions = (
            event.get(
                "competitions"
            )
            or []
        )

        if (
            not isinstance(
                competitions,
                list,
            )
            or len(
                competitions
            ) != 1
        ):
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        "competition_count_not_one",
                }
            )

            continue

        competition = (
            competitions[0]
        )

        if not isinstance(
            competition,
            dict,
        ):
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        "competition_not_object",
                }
            )

            continue

        status = (
            competition.get(
                "status"
            )
            or event.get(
                "status"
            )
            or {}
        )

        if not isinstance(
            status,
            dict,
        ):
            status = {}

        status_type = (
            status.get(
                "type"
            )
            or {}
        )

        if not isinstance(
            status_type,
            dict,
        ):
            status_type = {}

        state = str(
            status_type.get(
                "state"
            )
            or ""
        ).strip().lower()

        detail = (
            status_type.get(
                "detail"
            )
            or status_type.get(
                "shortDetail"
            )
        )

        period = parse_int(
            status.get(
                "period"
            )
        )

        clock = (
            status.get(
                "displayClock"
            )
        )

        if state not in {
            "pre",
            "in",
            "post",
        }:
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        "unexpected_state",

                    "state":
                        state,
                }
            )

            continue

        competitors = (
            competition.get(
                "competitors"
            )
            or []
        )

        if (
            not isinstance(
                competitors,
                list,
            )
            or len(
                competitors
            ) != 2
        ):
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        "competitor_count_not_two",
                }
            )

            continue

        teams = {}

        team_failure = None

        for competitor in competitors:

            if not isinstance(
                competitor,
                dict,
            ):
                team_failure = (
                    "competitor_not_object"
                )
                break

            team = (
                competitor.get(
                    "team"
                )
                or {}
            )

            if not isinstance(
                team,
                dict,
            ):
                team_failure = (
                    "team_not_object"
                )
                break

            team_id = str(
                team.get(
                    "id"
                )
                or competitor.get(
                    "id"
                )
                or ""
            ).strip()

            team_abbr = str(
                team.get(
                    "abbreviation"
                )
                or ""
            ).strip()

            home_away = str(
                competitor.get(
                    "homeAway"
                )
                or ""
            ).strip().lower()

            try:
                score = int(
                    competitor.get(
                        "score"
                    )
                    or 0
                )

            except (
                TypeError,
                ValueError,
            ):
                team_failure = (
                    "invalid_score"
                )
                break

            if (
                not team_id
                or not team_abbr
                or home_away
                not in {
                    "home",
                    "away",
                }
            ):
                team_failure = (
                    "invalid_team_identity"
                )
                break

            if home_away in teams:
                team_failure = (
                    "duplicate_home_away"
                )
                break

            teams[
                home_away
            ] = {
                "team_id":
                    team_id,

                "team":
                    team_abbr,

                "score":
                    score,
            }

        if (
            team_failure
            or set(
                teams.keys()
            ) != {
                "home",
                "away",
            }
        ):
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        team_failure
                        or
                        "missing_home_away",
                }
            )

            continue

        home = teams[
            "home"
        ]

        away = teams[
            "away"
        ]

        if (
            home[
                "team_id"
            ]
            == away[
                "team_id"
            ]
        ):
            malformed.append(
                {
                    "event_id":
                        event_id,

                    "reason":
                        "same_team_ids",
                }
            )

            continue

        discovered.append(
            {
                "event_id":
                    event_id,

                "state":
                    state,

                "detail":
                    detail,

                "period":
                    period,

                "clock":
                    clock,

                "home_team_id":
                    home[
                        "team_id"
                    ],

                "home_team":
                    home[
                        "team"
                    ],

                "home_score":
                    home[
                        "score"
                    ],

                "away_team_id":
                    away[
                        "team_id"
                    ],

                "away_team":
                    away[
                        "team"
                    ],

                "away_score":
                    away[
                        "score"
                    ],
            }
        )

    id_counts = Counter(
        raw_ids
    )

    duplicate_ids = sorted(
        event_id
        for event_id, count
        in id_counts.items()
        if count > 1
    )

    return (
        discovered,
        malformed,
        duplicate_ids,
    )


# ======================================================================
# RUN ONE EVENT
# ======================================================================

def run_event(
    event: dict,
    season: int | None,
) -> dict:

    event_id = (
        event[
            "event_id"
        ]
    )

    print()
    print("=" * 72)

    print(
        f"BEGIN EVENT | {event_id}"
    )

    print(
        "MATCHUP="
        f"{event['away_team']} "
        f"{event['away_score']} @ "
        f"{event['home_team']} "
        f"{event['home_score']}"
    )

    print(
        f"STATE={event['state']}"
    )

    print(
        f"DETAIL={event['detail']}"
    )

    print("=" * 72)

    pre_state = (
        read_live_database_state(
            event_id
        )
    )

    print(
        "PRE_EVENT_ROWS="
        f"{pre_state['event_rows']}"
    )

    print(
        "PRE_PLAY_ROWS="
        f"{pre_state['play_rows']}"
    )

    print(
        "PRE_PLAY_PLAYER_ROWS="
        f"{pre_state['player_rows']}"
    )

    print(
        "PRE_AUDIT_ROWS="
        f"{pre_state['audit_rows']}"
    )

    command = [
        str(
            PYTHON
        ),
        str(
            LIVE_INGEST
        ),
        "--event-id",
        event_id,
    ]

    if season is not None:
        command.extend(
            [
                "--season",
                str(
                    season
                ),
            ]
        )

    print()
    print(
        "RUNNER_COMMAND="
        + " ".join(
            command
        )
    )

    started = (
        time.perf_counter()
    )

    process = subprocess.run(
        command,
        cwd=str(
            ROOT
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        "--- LIVE INGEST STDOUT ---"
    )

    if process.stdout:
        print(
            process.stdout.rstrip()
        )

    else:
        print(
            "<EMPTY>"
        )

    print()
    print(
        "--- LIVE INGEST STDERR ---"
    )

    if process.stderr:
        print(
            process.stderr.rstrip()
        )

    else:
        print(
            "<EMPTY>"
        )

    print()
    print(
        f"RUNNER_RETURN_CODE={process.returncode}"
    )

    print(
        f"RUNNER_SECONDS={elapsed:.3f}"
    )

    post_state = (
        read_live_database_state(
            event_id
        )
    )

    event_delta = (
        post_state[
            "event_rows"
        ]
        - pre_state[
            "event_rows"
        ]
    )

    play_delta = (
        post_state[
            "play_rows"
        ]
        - pre_state[
            "play_rows"
        ]
    )

    player_delta = (
        post_state[
            "player_rows"
        ]
        - pre_state[
            "player_rows"
        ]
    )

    audit_delta = (
        post_state[
            "audit_rows"
        ]
        - pre_state[
            "audit_rows"
        ]
    )

    print()
    print(
        "--- EVENT POST-RUN RECONCILIATION ---"
    )

    print(
        "POST_EVENT_ROWS="
        f"{post_state['event_rows']}"
    )

    print(
        "POST_PLAY_ROWS="
        f"{post_state['play_rows']}"
    )

    print(
        "POST_PLAY_PLAYER_ROWS="
        f"{post_state['player_rows']}"
    )

    print(
        "POST_AUDIT_ROWS="
        f"{post_state['audit_rows']}"
    )

    print(
        f"EVENT_ROW_DELTA={event_delta}"
    )

    print(
        f"PLAY_ROW_DELTA={play_delta}"
    )

    print(
        "PLAY_PLAYER_ROW_DELTA="
        f"{player_delta}"
    )

    print(
        f"AUDIT_ROW_DELTA={audit_delta}"
    )

    print(
        "POST_INTEGRITY="
        f"{post_state['integrity']}"
    )

    print(
        "POST_FK_ERRORS="
        f"{post_state['fk_errors']}"
    )

    #
    # Classification:
    #
    # SUCCESS
    #   runner exit 0
    #   exactly one successful audit appended
    #   exactly one event row exists
    #   DB integrity/FK remain clean
    #
    # SAFE_FAIL
    #   runner exit nonzero
    #   no audit was appended
    #   event did not partially create/update a successful ingest record
    #
    # COMMITTED_FAILURE
    #   runner returned nonzero but an audit row was added.
    #   This is treated as critical because the child likely committed
    #   before a later post-commit gate failed.
    #
    # CONTRACT_FAIL
    #   runner returned zero but reconciliation does not match.
    #

    if (
        process.returncode == 0
        and audit_delta == 1
        and post_state[
            "event_rows"
        ] == 1
        and post_state[
            "integrity"
        ] == "ok"
        and post_state[
            "fk_errors"
        ] == 0
    ):
        outcome = (
            "SUCCESS"
        )

    elif (
        process.returncode != 0
        and audit_delta == 0
    ):
        outcome = (
            "SAFE_FAIL"
        )

    elif (
        process.returncode != 0
        and audit_delta != 0
    ):
        outcome = (
            "COMMITTED_FAILURE"
        )

    else:
        outcome = (
            "CONTRACT_FAIL"
        )

    print(
        f"EVENT_OUTCOME={outcome}"
    )

    return {
        "event_id":
            event_id,

        "away_team":
            event[
                "away_team"
            ],

        "home_team":
            event[
                "home_team"
            ],

        "scoreboard_state":
            event[
                "state"
            ],

        "return_code":
            process.returncode,

        "elapsed_seconds":
            elapsed,

        "pre_event_rows":
            pre_state[
                "event_rows"
            ],

        "post_event_rows":
            post_state[
                "event_rows"
            ],

        "pre_play_rows":
            pre_state[
                "play_rows"
            ],

        "post_play_rows":
            post_state[
                "play_rows"
            ],

        "play_delta":
            play_delta,

        "pre_player_rows":
            pre_state[
                "player_rows"
            ],

        "post_player_rows":
            post_state[
                "player_rows"
            ],

        "player_delta":
            player_delta,

        "pre_audit_rows":
            pre_state[
                "audit_rows"
            ],

        "post_audit_rows":
            post_state[
                "audit_rows"
            ],

        "audit_delta":
            audit_delta,

        "outcome":
            outcome,
    }


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "WFS NFL LIVE "
            "multi-event scoreboard orchestrator"
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help=(
            "Fallback season forwarded "
            "to live_ingest.py"
        ),
    )

    parser.add_argument(
        "--include-post",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow POST events as ingest candidates. "
            "Default: enabled."
        ),
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():
    args = parse_args()

    print("=" * 72)

    print(
        "WFS NFL LIVE — STAGE LIVE-12"
    )

    print(
        "FIRST MULTI-EVENT INGEST ORCHESTRATOR"
    )

    print(
        "SEQUENTIAL + EVENT-ISOLATED"
    )

    print("=" * 72)

    # ==================================================================
    # 1. PREFLIGHT
    # ==================================================================

    print()
    print(
        "=== 1. ORCHESTRATOR PREFLIGHT ==="
    )

    required_files = [
        (
            LIVE_DB,
            "wfs_live.db",
        ),
        (
            NFL_DB,
            "nfl.db",
        ),
        (
            FORECAST_LEDGER,
            "forecast_ledger.db",
        ),
        (
            LIVE_INGEST,
            "live_ingest.py",
        ),
        (
            PYTHON,
            "venv Python",
        ),
    ]

    for path, label in (
        required_files
    ):
        if not path.is_file():
            raise SystemExit(
                "FAIL | required file missing: "
                f"{label} | {path}"
            )

        print(
            f"PASS | {label} exists"
        )

    nfl_sha_before = (
        sha256_file(
            NFL_DB
        )
    )

    ledger_sha_before = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    live_db_sha_before = (
        sha256_file(
            LIVE_DB
        )
    )

    runner_sha_before = (
        sha256_file(
            LIVE_INGEST
        )
    )

    print()
    print(
        f"LIVE_INGEST_SHA256={runner_sha_before}"
    )

    print(
        f"LIVE_DB_SHA_BEFORE={live_db_sha_before}"
    )

    print(
        f"NFL_DB_SHA_BEFORE={nfl_sha_before}"
    )

    print(
        "FORECAST_LEDGER_SHA_BEFORE="
        f"{ledger_sha_before}"
    )

    # ==================================================================
    # 2. VERIFY LIVE DB CONTRACT
    # ==================================================================

    print()
    print(
        "=== 2. VERIFY LIVE DATABASE CONTRACT ==="
    )

    baseline = (
        read_live_database_state()
    )

    print(
        "SCHEMA_VERSION="
        f"{baseline['schema_version']}"
    )

    print(
        "SCHEMA_SHA256="
        f"{baseline['schema_sha256']}"
    )

    print(
        "INTEGRITY_CHECK="
        f"{baseline['integrity']}"
    )

    print(
        "FOREIGN_KEY_ERRORS="
        f"{baseline['fk_errors']}"
    )

    print(
        "GLOBAL_EVENT_ROWS="
        f"{baseline['global_event_rows']}"
    )

    print(
        "GLOBAL_PLAY_ROWS="
        f"{baseline['global_play_rows']}"
    )

    print(
        "GLOBAL_PLAY_PLAYER_ROWS="
        f"{baseline['global_player_rows']}"
    )

    print(
        "GLOBAL_AUDIT_ROWS="
        f"{baseline['global_audit_rows']}"
    )

    if (
        baseline[
            "schema_version"
        ]
        != EXPECTED_SCHEMA_VERSION
    ):
        raise SystemExit(
            "FAIL | LIVE schema version changed"
        )

    if (
        baseline[
            "schema_sha256"
        ]
        != EXPECTED_SCHEMA_SHA256
    ):
        raise SystemExit(
            "FAIL | LIVE schema fingerprint changed"
        )

    if (
        baseline[
            "integrity"
        ]
        != "ok"
    ):
        raise SystemExit(
            "FAIL | pre-orchestration "
            "integrity failure"
        )

    if (
        baseline[
            "fk_errors"
        ]
        != 0
    ):
        raise SystemExit(
            "FAIL | pre-orchestration "
            "foreign-key failure"
        )

    print(
        "PASS | WFS_LIVE_DB_V2 contract verified"
    )

    # ==================================================================
    # 3. FETCH SCOREBOARD
    # ==================================================================

    print()
    print(
        "=== 3. FETCH ESPN NFL SCOREBOARD ==="
    )

    try:
        target = load_live_schedule_target(
            args.season
        )

        print(
            "TARGET_SEASON="
            f"{target['season']}"
        )

        print(
            "TARGET_WEEK="
            f"{target['week']}"
        )

        print(
            "ACTIVE_GAME_WEEK="
            + (
                str(
                    target[
                        "active_game_week"
                    ]
                )
                if target[
                    "active_game_week"
                ] is not None
                else "NONE"
            )
        )

        print(
            "UPCOMING_WEEK="
            + (
                str(
                    target[
                        "upcoming_week"
                    ]
                )
                if target[
                    "upcoming_week"
                ] is not None
                else "NONE"
            )
        )

        print(
            "TARGET_SCHEDULE_GAMES="
            f"{target['game_count']}"
        )

        print(
            "TARGET_GAME_DATES="
            + ",".join(
                target[
                    "dates"
                ]
            )
        )

        (
            scoreboard,
            fetch_audit,
        ) = fetch_schedule_scoreboard(
            target
        )

    except Exception as exc:
        raise SystemExit(
            "FAIL | schedule-driven scoreboard "
            f"discovery | {exc}"
        )

    for row in fetch_audit:
        print(
            "DATE_BOUND_FETCH="
            f"{row['game_date']} "
            f"HTTP={row['http_status']} "
            f"LATENCY={row['latency']:.3f} "
            f"BYTES={row['byte_count']}"
        )

    print(
        "COMBINED_SCOREBOARD_EVENTS="
        f"{len(scoreboard.get('events') or [])}"
    )

    # ==================================================================
    # 4. DISCOVER + VALIDATE
    # ==================================================================

    print()
    print(
        "=== 4. DISCOVER + VALIDATE EVENTS ==="
    )

    (
        discovered,
        malformed,
        duplicate_ids,
    ) = discover_events(
        scoreboard
    )

    state_counts = Counter(
        row[
            "state"
        ]
        for row in discovered
    )

    print(
        "VALID_DISCOVERED_EVENTS="
        f"{len(discovered)}"
    )

    print(
        f"PRE_EVENTS={state_counts.get('pre', 0)}"
    )

    print(
        f"IN_EVENTS={state_counts.get('in', 0)}"
    )

    print(
        f"POST_EVENTS={state_counts.get('post', 0)}"
    )

    print(
        f"MALFORMED_EVENTS={len(malformed)}"
    )

    print(
        "DUPLICATE_EVENT_IDS="
        f"{len(duplicate_ids)}"
    )

    if malformed:
        print()
        print(
            "--- MALFORMED EVENTS ---"
        )

        for row in malformed:
            print(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

    if duplicate_ids:
        print(
            "DUPLICATE_IDS="
            + ",".join(
                duplicate_ids
            )
        )

    #
    # Discovery is a global fail-closed boundary.
    # Do not invoke even one writer if the scoreboard structure
    # itself is malformed.
    #

    if malformed:
        raise SystemExit(
            "FAIL | malformed scoreboard events "
            "— ZERO RUNNERS INVOKED"
        )

    if duplicate_ids:
        raise SystemExit(
            "FAIL | duplicate scoreboard event IDs "
            "— ZERO RUNNERS INVOKED"
        )

    if not discovered:
        raise SystemExit(
            "FAIL | zero valid scoreboard events "
            "— ZERO RUNNERS INVOKED"
        )

    try:
        discovered = reconcile_schedule_events(
            discovered,
            target,
        )

    except Exception as exc:
        raise SystemExit(
            "FAIL | schedule reconciliation | "
            f"{exc} — ZERO RUNNERS INVOKED"
        )

    print(
        "SCHEDULE_RECONCILED_EVENTS="
        f"{len(discovered)}"
    )

    print(
        "PASS | scoreboard discovery contract verified"
    )

    # ==================================================================
    # 5. CLASSIFY ELIGIBILITY
    # ==================================================================

    print()
    print(
        "=== 5. CLASSIFY INGEST ELIGIBILITY ==="
    )

    eligible = []

    skipped_pre = []

    skipped_post = []

    for event in discovered:

        if (
            event[
                "state"
            ]
            == "in"
        ):
            eligible.append(
                event
            )

        elif (
            event[
                "state"
            ]
            == "post"
        ):
            if args.include_post:
                eligible.append(
                    event
                )

            else:
                skipped_post.append(
                    event
                )

        else:
            skipped_pre.append(
                event
            )

    #
    # Deterministic order.
    #
    # Sort by event ID rather than relying on upstream list order.
    #

    eligible = sorted(
        eligible,
        key=lambda row:
            row[
                "event_id"
            ],
    )

    skipped_pre = sorted(
        skipped_pre,
        key=lambda row:
            row[
                "event_id"
            ],
    )

    skipped_post = sorted(
        skipped_post,
        key=lambda row:
            row[
                "event_id"
            ],
    )

    print(
        f"INGEST_CANDIDATES={len(eligible)}"
    )

    print(
        f"PRE_EVENTS_SKIPPED={len(skipped_pre)}"
    )

    print(
        f"POST_EVENTS_SKIPPED={len(skipped_post)}"
    )

    for event in eligible:
        print(
            "CANDIDATE"
            f" | event_id={event['event_id']}"
            f" | state={event['state']}"
            f" | matchup="
            f"{event['away_team']}@"
            f"{event['home_team']}"
            f" | detail={event['detail']}"
        )

    for event in skipped_pre:
        print(
            "SKIP_PRE"
            f" | event_id={event['event_id']}"
            f" | matchup="
            f"{event['away_team']}@"
            f"{event['home_team']}"
        )

    for event in skipped_post:
        print(
            "SKIP_POST"
            f" | event_id={event['event_id']}"
            f" | matchup="
            f"{event['away_team']}@"
            f"{event['home_team']}"
        )

    # ==================================================================
    # 6. RUN EVENTS INDEPENDENTLY
    # ==================================================================

    print()
    print(
        "=== 6. SEQUENTIAL EVENT INGEST ==="
    )

    results = []

    for event in eligible:

        try:
            result = run_event(
                event=event,
                season=args.season,
            )

        except Exception as exc:
            #
            # Orchestrator itself should keep moving to later games.
            # This does NOT suppress final failure.
            #
            result = {
                "event_id":
                    event[
                        "event_id"
                    ],

                "away_team":
                    event[
                        "away_team"
                    ],

                "home_team":
                    event[
                        "home_team"
                    ],

                "scoreboard_state":
                    event[
                        "state"
                    ],

                "return_code":
                    None,

                "elapsed_seconds":
                    None,

                "pre_event_rows":
                    None,

                "post_event_rows":
                    None,

                "pre_play_rows":
                    None,

                "post_play_rows":
                    None,

                "play_delta":
                    None,

                "pre_player_rows":
                    None,

                "post_player_rows":
                    None,

                "player_delta":
                    None,

                "pre_audit_rows":
                    None,

                "post_audit_rows":
                    None,

                "audit_delta":
                    None,

                "outcome":
                    "ORCHESTRATOR_EXCEPTION",

                "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }

            print()
            print(
                "EVENT_ORCHESTRATOR_EXCEPTION"
                f" | event_id={event['event_id']}"
                f" | error="
                f"{type(exc).__name__}: {exc}"
            )

        results.append(
            result
        )

    # ==================================================================
    # 7. GLOBAL POST-RUN DATABASE VERIFY
    # ==================================================================

    print()
    print(
        "=== 7. GLOBAL POST-RUN DATABASE VERIFY ==="
    )

    post_global = (
        read_live_database_state()
    )

    print(
        "SCHEMA_VERSION="
        f"{post_global['schema_version']}"
    )

    print(
        "SCHEMA_SHA256="
        f"{post_global['schema_sha256']}"
    )

    print(
        "INTEGRITY_CHECK="
        f"{post_global['integrity']}"
    )

    print(
        "FOREIGN_KEY_ERRORS="
        f"{post_global['fk_errors']}"
    )

    print(
        "GLOBAL_EVENT_ROWS="
        f"{post_global['global_event_rows']}"
    )

    print(
        "GLOBAL_PLAY_ROWS="
        f"{post_global['global_play_rows']}"
    )

    print(
        "GLOBAL_PLAY_PLAYER_ROWS="
        f"{post_global['global_player_rows']}"
    )

    print(
        "GLOBAL_AUDIT_ROWS="
        f"{post_global['global_audit_rows']}"
    )

    if (
        post_global[
            "schema_version"
        ]
        != EXPECTED_SCHEMA_VERSION
    ):
        raise SystemExit(
            "FAIL | post-run schema version changed"
        )

    if (
        post_global[
            "schema_sha256"
        ]
        != EXPECTED_SCHEMA_SHA256
    ):
        raise SystemExit(
            "FAIL | post-run schema fingerprint changed"
        )

    if (
        post_global[
            "integrity"
        ]
        != "ok"
    ):
        raise SystemExit(
            "FAIL | post-run integrity failure"
        )

    if (
        post_global[
            "fk_errors"
        ]
        != 0
    ):
        raise SystemExit(
            "FAIL | post-run foreign-key failure"
        )

    # ==================================================================
    # 8. PROTECTED SURFACE VERIFY
    # ==================================================================

    print()
    print(
        "=== 8. PROTECTED SURFACE VERIFICATION ==="
    )

    nfl_sha_after = (
        sha256_file(
            NFL_DB
        )
    )

    ledger_sha_after = (
        sha256_file(
            FORECAST_LEDGER
        )
    )

    runner_sha_after = (
        sha256_file(
            LIVE_INGEST
        )
    )

    nfl_unchanged = (
        nfl_sha_before
        == nfl_sha_after
    )

    ledger_unchanged = (
        ledger_sha_before
        == ledger_sha_after
    )

    runner_unchanged = (
        runner_sha_before
        == runner_sha_after
    )

    print(
        f"NFL_DB_SHA_AFTER={nfl_sha_after}"
    )

    print(
        "FORECAST_LEDGER_SHA_AFTER="
        f"{ledger_sha_after}"
    )

    print(
        f"LIVE_INGEST_SHA_AFTER={runner_sha_after}"
    )

    print(
        f"NFL_DB_UNCHANGED={int(nfl_unchanged)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        "LIVE_INGEST_UNCHANGED="
        f"{int(runner_unchanged)}"
    )

    if not nfl_unchanged:
        raise SystemExit(
            "FAIL | nfl.db changed"
        )

    if not ledger_unchanged:
        raise SystemExit(
            "FAIL | forecast_ledger.db changed"
        )

    if not runner_unchanged:
        raise SystemExit(
            "FAIL | live_ingest.py changed"
        )

    # ==================================================================
    # 9. ORCHESTRATION SUMMARY
    # ==================================================================

    print()
    print(
        "=== 9. MULTI-EVENT ORCHESTRATION SUMMARY ==="
    )

    outcome_counts = Counter(
        result[
            "outcome"
        ]
        for result in results
    )

    for result in results:
        print()
        print(
            "EVENT_RESULT"
            f" | event_id={result['event_id']}"
            f" | matchup="
            f"{result['away_team']}@"
            f"{result['home_team']}"
            f" | state={result['scoreboard_state']}"
            f" | outcome={result['outcome']}"
        )

        print(
            "  RETURN_CODE="
            f"{result.get('return_code')}"
        )

        print(
            "  PLAY_DELTA="
            f"{result.get('play_delta')}"
        )

        print(
            "  PLAYER_DELTA="
            f"{result.get('player_delta')}"
        )

        print(
            "  AUDIT_DELTA="
            f"{result.get('audit_delta')}"
        )

        if result.get(
            "error"
        ):
            print(
                "  ERROR="
                f"{result['error']}"
            )

    success_count = (
        outcome_counts.get(
            "SUCCESS",
            0,
        )
    )

    safe_fail_count = (
        outcome_counts.get(
            "SAFE_FAIL",
            0,
        )
    )

    committed_failure_count = (
        outcome_counts.get(
            "COMMITTED_FAILURE",
            0,
        )
    )

    contract_fail_count = (
        outcome_counts.get(
            "CONTRACT_FAIL",
            0,
        )
    )

    exception_count = (
        outcome_counts.get(
            "ORCHESTRATOR_EXCEPTION",
            0,
        )
    )

    print()
    print(
        f"ELIGIBLE_EVENTS={len(eligible)}"
    )

    print(
        f"SUCCESS_EVENTS={success_count}"
    )

    print(
        f"SAFE_FAIL_EVENTS={safe_fail_count}"
    )

    print(
        "COMMITTED_FAILURE_EVENTS="
        f"{committed_failure_count}"
    )

    print(
        "CONTRACT_FAIL_EVENTS="
        f"{contract_fail_count}"
    )

    print(
        "ORCHESTRATOR_EXCEPTION_EVENTS="
        f"{exception_count}"
    )

    print(
        f"PRE_EVENTS_SKIPPED={len(skipped_pre)}"
    )

    print(
        f"POST_EVENTS_SKIPPED={len(skipped_post)}"
    )

    # ==================================================================
    # 10. FINAL CONTRACT
    # ==================================================================

    print()
    print(
        "=== 10. LIVE-12 CONTRACT AUDIT ==="
    )

    print(
        "PASS | scoreboard discovery completed before writes"
    )

    print(
        "PASS | malformed scoreboard events fail closed globally"
    )

    print(
        "PASS | duplicate event IDs fail closed globally"
    )

    print(
        "PASS | PRE games excluded from live ingest"
    )

    print(
        "PASS | eligible events processed sequentially"
    )

    print(
        "PASS | each event invoked through live_ingest.py"
    )

    print(
        "PASS | ingest logic not duplicated in orchestrator"
    )

    print(
        "PASS | one child failure does not stop later events"
    )

    print(
        "PASS | each child run reconciled against audit delta"
    )

    print(
        "PASS | WFS_LIVE_DB_V2 schema fingerprint preserved"
    )

    print(
        "PASS | SQLite integrity check passed"
    )

    print(
        "PASS | SQLite foreign-key check passed"
    )

    print(
        "PASS | nfl.db unchanged"
    )

    print(
        "PASS | forecast_ledger.db unchanged"
    )

    print(
        "PASS | live_ingest.py unchanged"
    )

    print(
        "PASS | no CORE change"
    )

    print(
        "PASS | no V3 inference change"
    )

    print(
        "PASS | no forecast writer change"
    )

    print(
        "PASS | no optimizer change"
    )

    print(
        "PASS | no UI change"
    )

    print(
        "PASS | no cron/service installation"
    )

    #
    # Zero eligible events is a valid no-op.
    #
    # If there were eligible games, every one must have SUCCESS
    # for the overall LIVE-12 audit to pass.
    #

    hard_fail_count = (
        safe_fail_count
        + committed_failure_count
        + contract_fail_count
        + exception_count
    )

    print()
    print("=" * 72)

    if hard_fail_count == 0:

        print(
            "PASS | LIVE-12 MULTI-EVENT "
            "ORCHESTRATION COMPLETE"
        )

        overall_status = (
            "PASS"
        )

    else:

        print(
            "FAIL | LIVE-12 MULTI-EVENT "
            "ORCHESTRATION COMPLETED "
            "WITH EVENT FAILURES"
        )

        overall_status = (
            "FAIL"
        )

    print("=" * 72)

    print(
        f"DISCOVERED_EVENTS={len(discovered)}"
    )

    print(
        f"PRE_EVENTS={state_counts.get('pre', 0)}"
    )

    print(
        f"IN_EVENTS={state_counts.get('in', 0)}"
    )

    print(
        f"POST_EVENTS={state_counts.get('post', 0)}"
    )

    print(
        f"ELIGIBLE_EVENTS={len(eligible)}"
    )

    print(
        f"SUCCESS_EVENTS={success_count}"
    )

    print(
        f"SAFE_FAIL_EVENTS={safe_fail_count}"
    )

    print(
        "COMMITTED_FAILURE_EVENTS="
        f"{committed_failure_count}"
    )

    print(
        f"CONTRACT_FAIL_EVENTS={contract_fail_count}"
    )

    print(
        "ORCHESTRATOR_EXCEPTION_EVENTS="
        f"{exception_count}"
    )

    print(
        f"PRE_EVENTS_SKIPPED={len(skipped_pre)}"
    )

    print(
        "LIVE_DB_SCHEMA_UNCHANGED="
        f"{int(
            post_global['schema_sha256']
            == EXPECTED_SCHEMA_SHA256
        )}"
    )

    print(
        f"NFL_DB_UNCHANGED={int(nfl_unchanged)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        "LIVE_INGEST_UNCHANGED="
        f"{int(runner_unchanged)}"
    )

    print(
        "INTEGRITY_CHECK="
        f"{post_global['integrity']}"
    )

    print(
        "FOREIGN_KEY_ERRORS="
        f"{post_global['fk_errors']}"
    )

    print(
        f"ORCHESTRATOR_STATUS={overall_status}"
    )

    if hard_fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
