from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
import urllib.request

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ======================================================================
# FROZEN LIVE DATABASE CONTRACT
# ======================================================================

ROOT = Path("/home/mwynn/nfl_data_engine").resolve()
DATA_DIR = ROOT / "data"

LIVE_DB = DATA_DIR / "wfs_live.db"
NFL_DB = DATA_DIR / "nfl.db"
FORECAST_LEDGER = DATA_DIR / "forecast_ledger.db"

EXPECTED_SCHEMA_VERSION = "WFS_LIVE_DB_V2"

EXPECTED_SCHEMA_SHA256 = (
    "ea078eaf3115ea9e3894f51813ebee80"
    "f1168ced48ef9da557db00fc643d9bfa"
)

DEFAULT_SEASON = 2026
TIMEOUT = 15


# ======================================================================
# HELPERS
# ======================================================================

def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

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
        "|".join(str(value) for value in row)
        for row in rows
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def ascii_text(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        value or "",
    )

    return "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )


def normalize(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        ascii_text(value).lower(),
    )


def strip_suffix(
    parts: list[str],
) -> list[str]:
    suffixes = {
        "jr",
        "jr.",
        "sr",
        "sr.",
        "ii",
        "iii",
        "iv",
        "v",
    }

    parts = list(parts)

    while (
        len(parts) > 1
        and parts[-1].lower() in suffixes
    ):
        parts.pop()

    return parts


def player_identity_key(
    display_name: str,
) -> str | None:
    parts = strip_suffix(
        display_name.strip().split()
    )

    if len(parts) < 2:
        return None

    first = normalize(parts[0])
    surname = normalize("".join(parts[1:]))

    if not first or not surname:
        return None

    return f"{first[0]}.{surname}"


def play_token_key(
    token: str,
) -> str | None:
    match = re.match(
        r"^([A-Z])\.(.+)$",
        token.strip(),
    )

    if not match:
        return None

    surname = normalize(
        match.group(2)
    )

    if not surname:
        return None

    return (
        f"{match.group(1).lower()}."
        f"{surname}"
    )


def extract_tokens(
    text: str,
) -> list[str]:
    if not text:
        return []

    pattern = re.compile(
        r"\b"
        r"(?:[A-Z]\.){1,3}"
        r"[A-Z][A-Za-z'’\-]+"
        r"(?:\.[ ]+[A-Z][A-Za-z'’\-]+)?"
        r"(?:"
        r"\s+"
        r"(?![A-Z'’\-]+\b)"
        r"[A-Z][A-Za-z'’\-]+"
        r")?"
        r"\b"
    )

    return [m.split(".  ", 1)[0] for m in pattern.findall(text)]


# WFS_SUPPLEMENTAL_BOUNDARY_TOKENS_V1
#
# ESPN occasionally places two abbreviated player identities on
# opposite sides of a sentence boundary in forms that the frozen
# production token parser intentionally does not split.
#
# This helper is supplemental only. extract_tokens() remains
# unchanged.
def extract_supplemental_boundary_tokens(
    text: str,
) -> list[str]:
    if not text:
        return []

    teams = (
        "ARI|ATL|BAL|BUF|CAR|CHI|CIN|CLE|DAL|DEN|DET|GB|"
        "HOU|IND|JAX|KC|LAC|LA|LV|MIA|MIN|NE|NO|NYG|NYJ|"
        "PHI|PIT|SEA|SF|TB|TEN|WAS"
    )

    found = []

    identity_boundary = re.compile(
        r"\b"
        r"(?P<left>"
        r"(?:[A-Z]\.){1,3}"
        r"[A-Z][A-Za-z'’\-]+"
        r")"
        r"\.\s+"
        r"(?P<right>"
        r"(?:[A-Z][a-z]+)\."
        r"[A-Z][A-Za-z'’\-]+"
        r")"
        r"\b"
    )

    for match in identity_boundary.finditer(text):
        found.append(match.group("left"))
        found.append(match.group("right"))

    team_boundary = re.compile(
        rf"\b"
        rf"(?P<left>"
        rf"(?:[A-Z]\.){{1,3}}"
        rf"[A-Z][A-Za-z'’\-]+"
        rf")"
        rf"\.\s+"
        rf"(?P<team>{teams})-"
        rf"(?P<right>"
        rf"(?:[A-Z]\.){{1,3}}"
        rf"[A-Z][A-Za-z'’\-]+"
        rf")"
        rf"\b"
    )

    for match in team_boundary.finditer(text):
        found.append(match.group("left"))
        found.append(match.group("right"))

    result = []

    for token in found:
        if token not in result:
            result.append(token)

    return result


def extract_main_play_tokens(
    text: str,
) -> list[str]:
    normal = extract_tokens(text)

    supplemental = (
        extract_supplemental_boundary_tokens(
            text
        )
    )

    cleaned = []

    for token in normal:
        malformed = any(
            token.startswith(
                corrected + ". "
            )
            for corrected in supplemental
        )

        if not malformed:
            cleaned.append(token)

    for token in supplemental:
        if token not in cleaned:
            cleaned.append(token)

    return cleaned


def fetch_json(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 WFS-LIVE-RUNNER",
            "Accept":
                "application/json",
        },
    )

    started = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=TIMEOUT,
    ) as response:
        raw = response.read()
        status = response.status

    latency = time.perf_counter() - started

    return (
        json.loads(raw),
        status,
        latency,
        len(raw),
    )


def sequence_key(
    play: dict,
) -> int:
    try:
        return int(
            play.get("sequenceNumber")
            or 0
        )
    except Exception:
        return 0


def bool_int(value) -> int:
    return 1 if bool(value) else 0


def parse_int_or_dict(
    value,
    *,
    keys: tuple[str, ...],
    default=None,
):
    if value is None:
        return default

    if isinstance(value, bool):
        return default

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default

    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)

            if candidate is None:
                continue

            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue

    return default


def get_period_number(play: dict):
    period = play.get("period")

    if isinstance(period, dict):
        return period.get("number")

    return period


def get_clock_display(play: dict):
    clock = play.get("clock")

    if isinstance(clock, dict):
        return (
            clock.get("displayValue")
            or clock.get("value")
        )

    if clock is None:
        return None

    return str(clock)


def play_hash(
    play: dict,
) -> str:
    return sha256_json(
        {
            "sequence_number":
                sequence_key(play),

            "period":
                get_period_number(play),

            "clock":
                get_clock_display(play),

            "home_score":
                play.get("homeScore"),

            "away_score":
                play.get("awayScore"),

            "text":
                play.get("text") or "",

            "is_scoring_play":
                bool_int(
                    play.get("scoringPlay")
                ),

            "is_turnover":
                bool_int(
                    play.get("isTurnover")
                ),

            "is_penalty":
                bool_int(
                    play.get("isPenalty")
                ),
        }
    )


# ======================================================================
# LIVE INGEST
# ======================================================================

def ingest_event(
    event_id: str,
    default_season: int,
) -> None:

    summary_url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/football/nfl/summary"
        f"?event={event_id}"
    )

    print("=" * 72)
    print("WFS NFL LIVE — REUSABLE INGEST RUNNER")
    print("ATOMIC ESPN EVENT INGEST")
    print("=" * 72)

    print(f"EVENT_ID={event_id}")
    print(f"LIVE_DB={LIVE_DB}")

    # ==================================================================
    # 1. PREFLIGHT + PROTECTED HASHES
    # ==================================================================

    print()
    print(
        "=== 1. PREFLIGHT + PROTECTED FILE HASHES ==="
    )

    for path, label in [
        (LIVE_DB, "wfs_live.db"),
        (NFL_DB, "nfl.db"),
        (
            FORECAST_LEDGER,
            "forecast_ledger.db",
        ),
    ]:
        if not path.is_file():
            raise SystemExit(
                f"FAIL | required file missing: {label}"
            )

    nfl_sha_before = sha256_file(
        NFL_DB
    )

    ledger_sha_before = sha256_file(
        FORECAST_LEDGER
    )

    print(
        f"NFL_DB_SHA_BEFORE={nfl_sha_before}"
    )

    print(
        "FORECAST_LEDGER_SHA_BEFORE="
        f"{ledger_sha_before}"
    )

    # ==================================================================
    # 2. LIVE DB CONTRACT + BASELINE
    # ==================================================================

    print()
    print(
        "=== 2. LIVE DATABASE CONTRACT + BASELINE ==="
    )

    ro_uri = f"file:{LIVE_DB}?mode=ro"

    ro = sqlite3.connect(
        ro_uri,
        uri=True,
    )

    ro.row_factory = sqlite3.Row

    schema_version_row = ro.execute(
        """
        SELECT schema_version
        FROM live_meta
        """
    ).fetchone()

    if schema_version_row is None:
        ro.close()

        raise SystemExit(
            "FAIL | live_meta missing"
        )

    schema_version = (
        schema_version_row[
            "schema_version"
        ]
    )

    schema_sha = schema_fingerprint(ro)

    integrity_before = ro.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    fk_before = ro.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    pre_event_rows = ro.execute(
        """
        SELECT COUNT(*)
        FROM live_events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()[0]

    pre_play_rows = ro.execute(
        """
        SELECT COUNT(*)
        FROM live_plays
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()[0]

    pre_player_rows = ro.execute(
        """
        SELECT COUNT(*)
        FROM live_play_players
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()[0]

    pre_audit_rows = ro.execute(
        """
        SELECT COUNT(*)
        FROM live_ingest_audit
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()[0]

    pre_athlete_rows = ro.execute(
        """
        SELECT COUNT(*)
        FROM live_athletes
        """
    ).fetchone()[0]

    pre_event = ro.execute(
        """
        SELECT *
        FROM live_events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    pre_play_hashes = {
        row["play_id"]:
            row["play_sha256"]
        for row in ro.execute(
            """
            SELECT
                play_id,
                play_sha256
            FROM live_plays
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchall()
    }

    ro.close()

    print(
        f"SCHEMA_VERSION={schema_version}"
    )

    print(
        f"SCHEMA_SHA256={schema_sha}"
    )

    print(
        f"INTEGRITY_CHECK={integrity_before}"
    )

    print(
        f"FOREIGN_KEY_ERRORS={len(fk_before)}"
    )

    print(
        f"PRE_EVENT_ROWS={pre_event_rows}"
    )

    print(
        f"PRE_PLAY_ROWS={pre_play_rows}"
    )

    print(
        f"PRE_PLAY_PLAYER_ROWS={pre_player_rows}"
    )

    print(
        f"PRE_ATHLETE_ROWS={pre_athlete_rows}"
    )

    print(
        f"PRE_INGEST_AUDIT_ROWS={pre_audit_rows}"
    )

    if pre_event is not None:
        print(
            f"PRE_STATE={pre_event['state']}"
        )

        print(
            f"PRE_DETAIL={pre_event['detail']}"
        )

        print(
            f"PRE_PERIOD={pre_event['period']}"
        )

        print(
            f"PRE_CLOCK={pre_event['clock']}"
        )

        print(
            "PRE_LAST_PLAY_ID="
            f"{pre_event['last_play_id']}"
        )

    if (
        schema_version
        != EXPECTED_SCHEMA_VERSION
    ):
        raise SystemExit(
            "FAIL | unexpected LIVE schema version"
        )

    if (
        schema_sha
        != EXPECTED_SCHEMA_SHA256
    ):
        raise SystemExit(
            "FAIL | LIVE schema fingerprint changed"
        )

    if integrity_before != "ok":
        raise SystemExit(
            "FAIL | pre-ingest integrity failure"
        )

    if fk_before:
        raise SystemExit(
            "FAIL | pre-ingest foreign-key failure"
        )

    # ==================================================================
    # 3. FETCH ESPN SUMMARY
    # ==================================================================

    print()
    print(
        "=== 3. FETCH ESPN EVENT SUMMARY ==="
    )

    (
        summary,
        status,
        latency,
        byte_count,
    ) = fetch_json(
        summary_url
    )

    print(f"HTTP_STATUS={status}")

    print(
        f"LATENCY_SECONDS={latency:.3f}"
    )

    print(f"BYTES={byte_count}")

    if status != 200:
        raise SystemExit(
            "FAIL | ESPN summary fetch"
        )

    # ==================================================================
    # 4. EVENT STATE
    # ==================================================================

    print()
    print(
        "=== 4. BUILD EVENT STATE ==="
    )

    header = (
        summary.get("header")
        or {}
    )

    if not isinstance(header, dict):
        raise SystemExit(
            "FAIL | ESPN header is not an object"
        )

    competitions = (
        header.get("competitions")
        or []
    )

    if len(competitions) != 1:
        raise SystemExit(
            "FAIL | expected exactly one competition"
        )

    competition = competitions[0]

    if not isinstance(
        competition,
        dict,
    ):
        raise SystemExit(
            "FAIL | competition is not an object"
        )

    status_obj = (
        competition.get("status")
        or {}
    )

    if not isinstance(
        status_obj,
        dict,
    ):
        status_obj = {}

    status_type = (
        status_obj.get("type")
        or {}
    )

    if not isinstance(
        status_type,
        dict,
    ):
        status_type = {}

    event_state = {
        "event_id":
            event_id,

        "state":
            status_type.get("state")
            or "",

        "detail":
            status_type.get("detail"),

        "period":
            parse_int_or_dict(
                status_obj.get("period"),
                keys=(
                    "number",
                    "value",
                ),
                default=None,
            ),

        "clock":
            status_obj.get(
                "displayClock"
            ),
    }

    raw_season = header.get(
        "season"
    )

    raw_week = header.get(
        "week"
    )

    season_value = parse_int_or_dict(
        raw_season,
        keys=(
            "year",
            "season",
            "value",
        ),
        default=default_season,
    )

    season_type = None

    if isinstance(
        raw_season,
        dict,
    ):
        season_type = parse_int_or_dict(
            raw_season.get("type"),
            keys=(
                "id",
                "value",
                "type",
            ),
            default=None,
        )

    if season_type is None:
        season_type = parse_int_or_dict(
            header.get(
                "seasonType"
            ),
            keys=(
                "id",
                "value",
                "type",
            ),
            default=None,
        )

    week_value = parse_int_or_dict(
        raw_week,
        keys=(
            "number",
            "week",
            "value",
        ),
        default=None,
    )

    team_id_to_abbr = {}
    team_data = {}

    for competitor in (
        competition.get(
            "competitors"
        )
        or []
    ):
        if not isinstance(
            competitor,
            dict,
        ):
            continue

        team = (
            competitor.get("team")
            or {}
        )

        if not isinstance(
            team,
            dict,
        ):
            continue

        team_id = str(
            team.get("id")
            or competitor.get("id")
            or ""
        )

        abbr = (
            team.get("abbreviation")
            or ""
        )

        home_away = (
            competitor.get(
                "homeAway"
            )
        )

        try:
            score = int(
                competitor.get("score")
                or 0
            )
        except Exception:
            raise SystemExit(
                "FAIL | invalid team score"
            )

        if (
            not team_id
            or not abbr
            or home_away
            not in {"home", "away"}
        ):
            raise SystemExit(
                "FAIL | invalid team inventory"
            )

        team_id_to_abbr[
            team_id
        ] = abbr

        team_data[
            team_id
        ] = {
            "team_id":
                team_id,
            "abbr":
                abbr,
            "home_away":
                home_away,
            "score":
                score,
        }

    if len(team_data) != 2:
        raise SystemExit(
            "FAIL | expected exactly two teams"
        )

    home_rows = [
        row
        for row in team_data.values()
        if row["home_away"]
        == "home"
    ]

    away_rows = [
        row
        for row in team_data.values()
        if row["home_away"]
        == "away"
    ]

    if (
        len(home_rows) != 1
        or len(away_rows) != 1
    ):
        raise SystemExit(
            "FAIL | home/away classification"
        )

    home = home_rows[0]
    away = away_rows[0]

    print(
        f"EVENT_STATE={event_state['state']}"
    )

    print(
        f"EVENT_DETAIL={event_state['detail']}"
    )

    print(
        f"PERIOD={event_state['period']}"
    )

    print(
        f"CLOCK={event_state['clock']}"
    )

    print(
        f"SEASON={season_value}"
    )

    print(
        f"SEASON_TYPE={season_type}"
    )

    print(
        f"WEEK={week_value}"
    )

    print(
        f"HOME={home['abbr']} | "
        f"id={home['team_id']} | "
        f"score={home['score']}"
    )

    print(
        f"AWAY={away['abbr']} | "
        f"id={away['team_id']} | "
        f"score={away['score']}"
    )

    # ==================================================================
    # 5. PLAY COLLECTION
    # ==================================================================

    print()
    print(
        "=== 5. COLLECT + DEDUPLICATE PLAYS ==="
    )

    drives = (
        summary.get("drives")
        or {}
    )

    if not isinstance(
        drives,
        dict,
    ):
        raise SystemExit(
            "FAIL | drives is not an object"
        )

    previous_drives = (
        drives.get("previous")
        or []
    )

    current_drive = (
        drives.get("current")
        or {}
    )

    if not isinstance(
        current_drive,
        dict,
    ):
        current_drive = {}

    raw_plays = []

    for drive in previous_drives:
        if not isinstance(
            drive,
            dict,
        ):
            continue

        drive_plays = (
            drive.get("plays")
            or []
        )

        if isinstance(
            drive_plays,
            list,
        ):
            raw_plays.extend(
                drive_plays
            )

    current_plays = (
        current_drive.get("plays")
        or []
    )

    if isinstance(
        current_plays,
        list,
    ):
        raw_plays.extend(
            current_plays
        )

    unique_by_id = {}

    for play in raw_plays:
        if not isinstance(
            play,
            dict,
        ):
            continue

        play_id = str(
            play.get("id")
            or ""
        )

        if not play_id:
            continue

        unique_by_id[
            play_id
        ] = play

    plays = sorted(
        unique_by_id.values(),
        key=sequence_key,
    )

    raw_count = len(
        raw_plays
    )

    unique_count = len(
        plays
    )

    duplicate_count = (
        raw_count
        - unique_count
    )

    print(
        f"RAW_PLAY_ROWS={raw_count}"
    )

    print(
        f"UNIQUE_PLAYS={unique_count}"
    )

    print(
        f"DUPLICATES_REMOVED={duplicate_count}"
    )

    if unique_count == 0:
        raise SystemExit(
            "FAIL | zero unique plays"
        )

    required_play_keys = {
        "id",
        "text",
        "clock",
        "period",
        "sequenceNumber",
        "teamParticipants",
    }

    for play in plays:
        missing = (
            required_play_keys
            - set(play.keys())
        )

        if missing:
            raise SystemExit(
                "FAIL | play schema "
                f"{play.get('id')} missing="
                + ",".join(
                    sorted(missing)
                )
            )

    # ==================================================================
    # 6. BOXSCORE ATHLETES
    # ==================================================================

    print()
    print(
        "=== 6. BUILD BOXSCORE ATHLETE DIRECTORY ==="
    )

    boxscore_athletes = {}

    boxscore = (
        summary.get("boxscore")
        or {}
    )

    if not isinstance(
        boxscore,
        dict,
    ):
        boxscore = {}

    for team_group in (
        boxscore.get("players")
        or []
    ):
        if not isinstance(
            team_group,
            dict,
        ):
            continue

        team = (
            team_group.get("team")
            or {}
        )

        if not isinstance(
            team,
            dict,
        ):
            continue

        team_id = str(
            team.get("id")
            or ""
        )

        abbr = (
            team.get("abbreviation")
            or team_id_to_abbr.get(
                team_id
            )
            or ""
        )

        for stat_group in (
            team_group.get(
                "statistics"
            )
            or []
        ):
            if not isinstance(
                stat_group,
                dict,
            ):
                continue

            for athlete_row in (
                stat_group.get(
                    "athletes"
                )
                or []
            ):
                if not isinstance(
                    athlete_row,
                    dict,
                ):
                    continue

                athlete = (
                    athlete_row.get(
                        "athlete"
                    )
                    or {}
                )

                if not isinstance(
                    athlete,
                    dict,
                ):
                    continue

                athlete_id = str(
                    athlete.get("id")
                    or ""
                )

                display_name = (
                    athlete.get(
                        "displayName"
                    )
                    or athlete.get(
                        "fullName"
                    )
                    or ""
                ).strip()

                if (
                    not athlete_id
                    or not display_name
                ):
                    continue

                boxscore_athletes[
                    athlete_id
                ] = {
                    "athlete_id":
                        athlete_id,

                    "display_name":
                        display_name,

                    "team_id":
                        team_id,

                    "team":
                        abbr,

                    "identity_key":
                        player_identity_key(
                            display_name
                        ),

                    "identity_source":
                        "boxscore",

                    "position":
                        None,

                    "jersey":
                        athlete.get(
                            "jersey"
                        ),
                }

    print(
        "BOXSCORE_ATHLETES="
        f"{len(boxscore_athletes)}"
    )

    if not boxscore_athletes:
        raise SystemExit(
            "FAIL | zero boxscore athletes"
        )

    # ==================================================================
    # 7. ROSTER FALLBACK
    # ==================================================================

    print()
    print(
        "=== 7. BUILD ROSTER FALLBACK DIRECTORY ==="
    )

    roster_athletes = {}

    roster_season = (
        season_value
        if season_value is not None
        else default_season
    )

    for team_id, team_row in (
        team_data.items()
    ):
        abbr = team_row["abbr"]

        roster_url = (
            "https://site.api.espn.com/apis/site/v2/"
            "sports/football/nfl/teams/"
            f"{team_id}/roster"
            f"?season={roster_season}"
        )

        (
            roster_payload,
            roster_status,
            roster_latency,
            roster_bytes,
        ) = fetch_json(
            roster_url
        )

        print(
            f"TEAM={abbr} | "
            f"HTTP={roster_status} | "
            f"BYTES={roster_bytes} | "
            f"LATENCY={roster_latency:.3f}"
        )

        if roster_status != 200:
            raise SystemExit(
                f"FAIL | roster fetch {abbr}"
            )

        discovered = {}

        def walk(value):
            if isinstance(
                value,
                dict,
            ):
                athlete_id = (
                    value.get("id")
                )

                display_name = (
                    value.get(
                        "displayName"
                    )
                    or value.get(
                        "fullName"
                    )
                )

                signals = {
                    "position",
                    "jersey",
                    "age",
                    "experience",
                    "college",
                    "headshot",
                    "status",
                }

                if (
                    athlete_id
                    and display_name
                    and isinstance(
                        display_name,
                        str,
                    )
                    and signals.intersection(
                        value.keys()
                    )
                ):
                    discovered[
                        str(athlete_id)
                    ] = value

                for child in (
                    value.values()
                ):
                    walk(child)

            elif isinstance(
                value,
                list,
            ):
                for child in value:
                    walk(child)

        walk(roster_payload)

        for athlete_id, athlete in (
            discovered.items()
        ):
            display_name = (
                athlete.get(
                    "displayName"
                )
                or athlete.get(
                    "fullName"
                )
                or ""
            ).strip()

            if not display_name:
                continue

            position = (
                athlete.get("position")
                or {}
            )

            if isinstance(
                position,
                dict,
            ):
                position_value = (
                    position.get(
                        "abbreviation"
                    )
                    or position.get(
                        "name"
                    )
                    or position.get(
                        "displayName"
                    )
                )

            else:
                position_value = (
                    str(position)
                    if position
                    else None
                )

            roster_athletes[
                athlete_id
            ] = {
                "athlete_id":
                    athlete_id,

                "display_name":
                    display_name,

                "team_id":
                    team_id,

                "team":
                    abbr,

                "identity_key":
                    player_identity_key(
                        display_name
                    ),

                "identity_source":
                    "roster",

                "position":
                    position_value,

                "jersey":
                    athlete.get(
                        "jersey"
                    ),
            }

    print(
        f"ROSTER_ATHLETES={len(roster_athletes)}"
    )

    if not roster_athletes:
        raise SystemExit(
            "FAIL | zero roster athletes"
        )

    # ==================================================================
    # 8. MERGED IDENTITY DIRECTORY
    # ==================================================================

    print()
    print(
        "=== 8. MERGE IDENTITY SOURCES ==="
    )

    merged_athletes = dict(
        roster_athletes
    )

    for athlete_id, box_row in (
        boxscore_athletes.items()
    ):
        roster_row = (
            merged_athletes.get(
                athlete_id
            )
        )

        if roster_row:
            merged = dict(
                roster_row
            )

            merged.update(
                box_row
            )

            if not merged.get(
                "position"
            ):
                merged["position"] = (
                    roster_row.get(
                        "position"
                    )
                )

            if not merged.get(
                "jersey"
            ):
                merged["jersey"] = (
                    roster_row.get(
                        "jersey"
                    )
                )

            merged[
                "identity_source"
            ] = "boxscore"

            merged_athletes[
                athlete_id
            ] = merged

        else:
            merged_athletes[
                athlete_id
            ] = box_row

    identity_index = defaultdict(
        list
    )

    for athlete in (
        merged_athletes.values()
    ):
        key = athlete.get(
            "identity_key"
        )

        if not key:
            continue

        identity_index[
            key
        ].append(
            athlete
        )

    collision_keys = {
        key: rows
        for key, rows
        in identity_index.items()
        if len(rows) > 1
    }

    print(
        f"MERGED_ATHLETES={len(merged_athletes)}"
    )

    print(
        f"IDENTITY_KEYS={len(identity_index)}"
    )

    print(
        f"COLLISION_KEYS={len(collision_keys)}"
    )

    for key, rows in sorted(
        collision_keys.items()
    ):
        print(
            f"COLLISION={key}"
        )

        for row in rows:
            print(
                "  "
                f"{row['display_name']} | "
                f"{row['team']} | "
                f"{row['athlete_id']} | "
                f"{row['identity_source']}"
            )

    # ==================================================================
    # 9. PLAYER TOKEN RESOLUTION
    # ==================================================================

    print()
    print(
        "=== 9. RESOLVE PLAYER IDENTITIES ==="
    )

    resolved_occurrences = []
    ambiguous_occurrences = []
    unresolved_occurrences = []

    play_token_map = {}

    for play in plays:
        play_id = str(
            play.get("id")
        )

        text = (
            play.get("text")
            or ""
        )

        participant_team_ids = []
        offense_team_ids = []
        defense_team_ids = []

        for participant in (
            play.get(
                "teamParticipants"
            )
            or []
        ):
            if not isinstance(
                participant,
                dict,
            ):
                continue

            team_id = str(
                participant.get("id")
                or ""
            )

            participant_type = str(
                participant.get("type")
                or ""
            ).strip().lower()

            if (
                team_id
                and team_id
                not in participant_team_ids
            ):
                participant_team_ids.append(
                    team_id
                )

            if (
                team_id
                and participant_type
                == "offense"
                and team_id
                not in offense_team_ids
            ):
                offense_team_ids.append(
                    team_id
                )

            if (
                team_id
                and participant_type
                == "defense"
                and team_id
                not in defense_team_ids
            ):
                defense_team_ids.append(
                    team_id
                )

        token_rows = []

        for ordinal, token in enumerate(
            extract_main_play_tokens(text)
        ):
            key = play_token_key(
                token
            )

            if not key:
                continue

            candidates = list(
                identity_index.get(
                    key,
                    [],
                )
            )

            candidates = list(
                {
                    row["athlete_id"]:
                        row
                    for row in candidates
                }.values()
            )

            chosen = None
            method = None

            if len(candidates) == 1:
                chosen = candidates[0]

                method = (
                    "GLOBAL_EXACT_UNIQUE"
                )

            elif len(candidates) > 1:
                # Event scope is authoritative for this ingest.
                #
                # The merged identity directory may contain players
                # from outside the current game whose abbreviated
                # play token collides with an event participant.
                # Restrict collisions to the two teams actually
                # participating in this event when that produces
                # exactly one candidate.  True event-level
                # collisions remain unresolved and therefore retain
                # the existing fail-closed behavior.
                event_team_ids = {
                    str(home["team_id"]),
                    str(away["team_id"]),
                }

                event_candidates = [
                    row
                    for row in candidates
                    if str(row.get("team_id") or "")
                    in event_team_ids
                ]

                event_candidates = list(
                    {
                        row["athlete_id"]: row
                        for row in event_candidates
                    }.values()
                )

                if len(event_candidates) == 1:
                    chosen = event_candidates[0]

                    method = (
                        "EVENT_TEAM_EXACT_UNIQUE"
                    )

                # WFS_OFFENSE_TEAM_EXACT_UNIQUE_V1
                #
                # For a true event-level abbreviation collision,
                # structured ESPN offense-team context may identify
                # exactly one candidate. Zero or multiple matches
                # remain fail-closed.
                if (
                    chosen is None
                    and len(offense_team_ids) == 1
                ):
                    offense_candidates = [
                        row
                        for row in candidates
                        if str(
                            row.get("team_id")
                            or ""
                        )
                        == offense_team_ids[0]
                    ]

                    offense_candidates = list(
                        {
                            row["athlete_id"]: row
                            for row
                            in offense_candidates
                        }.values()
                    )

                    if (
                        len(offense_candidates)
                        == 1
                    ):
                        chosen = (
                            offense_candidates[0]
                        )

                        method = (
                            "OFFENSE_TEAM_EXACT_UNIQUE"
                        )

                team_candidates = [
                    row
                    for row in candidates
                    if row["team_id"]
                    in participant_team_ids
                ]

                team_candidates = list(
                    {
                        row[
                            "athlete_id"
                        ]: row
                        for row
                        in team_candidates
                    }.values()
                )

                if (
                    len(team_candidates)
                    == 1
                ):
                    chosen = (
                        team_candidates[0]
                    )

                    method = (
                        "PLAY_TEAM_EXACT_UNIQUE"
                    )

                elif (
                    len(defense_team_ids)
                    == 1
                    and len(offense_team_ids)
                    == 1
                    and not bool(
                        play.get("isTurnover")
                    )
                ):
                    start_team_id = str(
                        (
                            (
                                play.get("start")
                                or {}
                            ).get("team")
                            or {}
                        ).get("id")
                        or ""
                    )

                    end_team_id = str(
                        (
                            (
                                play.get("end")
                                or {}
                            ).get("team")
                            or {}
                        ).get("id")
                        or ""
                    )

                    token_in_parentheses = bool(
                        re.search(
                            r"\([^)]*"
                            + re.escape(token)
                            + r"[^)]*\)",
                            text,
                        )
                    )

                    possession_stable = (
                        start_team_id
                        and end_team_id
                        and start_team_id
                        == end_team_id
                        == offense_team_ids[0]
                    )

                    if (
                        token_in_parentheses
                        and possession_stable
                    ):
                        defense_candidates = [
                            row
                            for row in candidates
                            if row["team_id"]
                            == defense_team_ids[0]
                        ]

                        defense_candidates = list(
                            {
                                row["athlete_id"]:
                                    row
                                for row
                                in defense_candidates
                            }.values()
                        )

                        if (
                            len(defense_candidates)
                            == 1
                        ):
                            chosen = (
                                defense_candidates[0]
                            )

                            method = (
                                "PLAY_TEAM_EXACT_UNIQUE"
                            )

            # WFS_NON_FANTASY_DEFENSIVE_ATTRIBUTION_V2
            #
            # ESPN PBP commonly ends scrimmage/return descriptions
            # with a parenthetical individual-defender attribution:
            #
            #   "... for 3 yards (A.Gilman; R.Thomas)."
            #
            # Individual defensive players are outside the WFS
            # fantasy-player identity surface. Preserve the raw PBP
            # text, but do not require those terminal defender tokens
            # to resolve to fantasy-player identities.
            #
            # This applies only when ESPN supplies exactly one
            # structured offense team and one structured defense
            # team, and the token occurs inside the FINAL
            # parenthetical group at the end of the play text.
            #
            # Tokens in the main play description remain subject to
            # the normal exact resolver and fail-closed gates.
            if chosen is None:
                terminal_attribution = re.search(
                    r"\(([^()]*)\)\.?\s*$",
                    text,
                )

                token_in_terminal_attribution = False

                if terminal_attribution:
                    attribution_text = (
                        terminal_attribution.group(1)
                        or ""
                    )

                    attribution_tokens = (
                        extract_tokens(
                            attribution_text
                        )
                    )

                    token_in_terminal_attribution = (
                        token in attribution_tokens
                    )

                structured_two_sided_play = (
                    len(offense_team_ids) == 1
                    and len(defense_team_ids) == 1
                    and offense_team_ids[0]
                    != defense_team_ids[0]
                )

                if (
                    structured_two_sided_play
                    and token_in_terminal_attribution
                ):
                    print(
                        "NON_FANTASY_DEFENSIVE_TOKEN_SKIPPED"
                        f" | play_id={play_id}"
                        f" | token={token}"
                        f" | defense_team_id={defense_team_ids[0]}"
                    )
                    continue

            if chosen is not None:
                result = {
                    "event_id":
                        event_id,

                    "play_id":
                        play_id,

                    "token_ordinal":
                        ordinal,

                    "raw_token":
                        token,

                    "identity_key":
                        key,

                    "athlete_id":
                        chosen[
                            "athlete_id"
                        ],

                    "display_name":
                        chosen[
                            "display_name"
                        ],

                    "team_id":
                        chosen[
                            "team_id"
                        ],

                    "team":
                        chosen["team"],

                    "resolution_method":
                        method,

                    "identity_source":
                        chosen[
                            "identity_source"
                        ],
                }

                resolved_occurrences.append(
                    result
                )

                token_rows.append(
                    result
                )

            elif len(candidates) > 1:
                # Individual defensive players are outside the
                # FanDuel fantasy-player tracking surface.
                #
                # Preserve their names in raw play text, but do not
                # allow an ambiguous defender token to fail the
                # entire event ingest. Team defense/DST remains
                # tracked independently at the team level.
                #
                # Offensive / fantasy-relevant ambiguity remains
                # fail-closed.
                fantasy_positions = {
                    "QB",
                    "RB",
                    "FB",
                    "WR",
                    "TE",
                    "PK",
                    "K",
                }

                candidate_positions = {
                    str(
                        row.get("position")
                        or ""
                    ).strip().upper()
                    for row in candidates
                }

                fantasy_candidates = [
                    row
                    for row in candidates
                    if str(
                        row.get("position")
                        or ""
                    ).strip().upper()
                    in fantasy_positions
                ]

                if fantasy_candidates:
                    ambiguous_occurrences.append(
                        {
                            "play_id":
                                play_id,

                            "token":
                                token,

                            "identity_key":
                                key,

                            "candidate_ids":
                                [
                                    row[
                                        "athlete_id"
                                    ]
                                    for row
                                    in candidates
                                ],

                            "candidate_positions":
                                sorted(
                                    candidate_positions
                                ),

                            "text":
                                text,
                        }
                    )

                else:
                    print(
                        "NON_FANTASY_AMBIGUITY_SKIPPED"
                        f" | play_id={play_id}"
                        f" | token={token}"
                        " | positions="
                        + ",".join(
                            sorted(
                                candidate_positions
                            )
                        )
                    )

            else:
                # ======================================================
                # WFS_ZERO_CANDIDATE_CONTEXT_V1
                # ======================================================
                #
                # Normal exact identity resolution found no candidate.
                # Two tightly scoped recovery/classification paths are
                # permitted below.
                #
                # 1. Fantasy-relevant offensive identity:
                #    same structured offense team + exact normalized
                #    surname + exactly one event athlete.
                #
                # 2. Proven non-fantasy attribution grammar:
                #    defensive attribution / defensive penalty /
                #    bracketed defender / explicit team-prefixed
                #    special-teams attribution.
                #
                # Anything else remains unresolved and therefore
                # fail-closed.

                zero_candidate_handled = False

                # ------------------------------------------------------
                # A. OFFENSE_SURNAME_EXACT_UNIQUE
                # ------------------------------------------------------

                token_match = re.match(
                    r"^(?:[A-Z]\.){1,3}(.+)$",
                    token.strip(),
                )

                token_surname = None

                if token_match:
                    token_surname = normalize(
                        token_match.group(1)
                    )

                if (
                    token_surname
                    and len(offense_team_ids) == 1
                ):
                    surname_candidates = []

                    for athlete in (
                        merged_athletes.values()
                    ):
                        if (
                            str(
                                athlete.get("team_id")
                                or ""
                            )
                            != offense_team_ids[0]
                        ):
                            continue

                        display_name = str(
                            athlete.get(
                                "display_name"
                            )
                            or ""
                        ).strip()

                        name_parts = strip_suffix(
                            display_name.split()
                        )

                        if len(name_parts) < 2:
                            continue

                        athlete_surname = normalize(
                            "".join(
                                name_parts[1:]
                            )
                        )

                        if (
                            athlete_surname
                            == token_surname
                        ):
                            surname_candidates.append(
                                athlete
                            )

                    surname_candidates = list(
                        {
                            row["athlete_id"]: row
                            for row
                            in surname_candidates
                        }.values()
                    )

                    if (
                        len(surname_candidates)
                        == 1
                    ):
                        chosen = (
                            surname_candidates[0]
                        )

                        result = {
                            "event_id":
                                event_id,

                            "play_id":
                                play_id,

                            "token_ordinal":
                                ordinal,

                            "raw_token":
                                token,

                            "identity_key":
                                key,

                            "athlete_id":
                                chosen[
                                    "athlete_id"
                                ],

                            "display_name":
                                chosen[
                                    "display_name"
                                ],

                            "team_id":
                                chosen[
                                    "team_id"
                                ],

                            "team":
                                chosen[
                                    "team"
                                ],

                            "resolution_method":
                                "OFFENSE_SURNAME_EXACT_UNIQUE",

                            "identity_source":
                                chosen[
                                    "identity_source"
                                ],
                        }

                        resolved_occurrences.append(
                            result
                        )

                        token_rows.append(
                            result
                        )

                        print(
                            "OFFENSE_SURNAME_EXACT_UNIQUE"
                            f" | play_id={play_id}"
                            f" | token={token}"
                            f" | athlete_id="
                            f"{chosen['athlete_id']}"
                            f" | display_name="
                            f"{chosen['display_name']}"
                            f" | team_id="
                            f"{chosen['team_id']}"
                        )

                        zero_candidate_handled = True

                # ------------------------------------------------------
                # B. Proven non-fantasy attribution contexts.
                #
                # This does NOT resolve an athlete identity and does
                # not create a fantasy-player reference.
                # ------------------------------------------------------

                if not zero_candidate_handled:

                    structured_two_sided_play = (
                        len(offense_team_ids) == 1
                        and len(defense_team_ids) == 1
                        and offense_team_ids[0]
                        != defense_team_ids[0]
                    )

                    escaped = re.escape(token)

                    # Defender immediately before a penalty sentence:
                    #
                    #   (S.Joseph).PENALTY ...
                    #   (A.Phillips).PENALTY ...
                    defender_before_penalty = bool(
                        re.search(
                            r"\([^)]*"
                            + escaped
                            + r"[^)]*\)"
                            + r"\.?\s*PENALTY\b",
                            text,
                        )
                    )

                    # Explicit defensive penalty attribution:
                    #
                    #   PENALTY on NYG-A.Phillips,
                    #   Defensive Pass Interference
                    defensive_penalty = False

                    if (
                        len(defense_team_ids)
                        == 1
                    ):
                        defense_team_abbr = None

                        for side in (
                            home,
                            away,
                        ):
                            if (
                                str(
                                    side.get(
                                        "team_id"
                                    )
                                    or ""
                                )
                                == defense_team_ids[0]
                            ):
                                defense_team_abbr = str(
                                    side.get("abbr")
                                    or ""
                                ).strip().upper()

                        if defense_team_abbr:
                            defensive_penalty = bool(
                                re.search(
                                    r"\bPENALTY\s+on\s+"
                                    + re.escape(
                                        defense_team_abbr
                                    )
                                    + r"-"
                                    + escaped
                                    + r"\s*,\s*Defensive\b",
                                    text,
                                    flags=re.IGNORECASE,
                                )
                            )

                    # Bracketed individual defensive pressure:
                    #
                    #   [L.Rodriguez]
                    bracketed_defender = bool(
                        re.search(
                            r"\["
                            + escaped
                            + r"\]",
                            text,
                        )
                    )

                    # Explicit team-prefixed special-teams attribution:
                    #
                    #   downed by NYJ-F.Mauigoa.
                    #
                    # Require a punt plus the prefix matching the
                    # structured offense/special-teams side.
                    team_prefixed_special_teams = False

                    if (
                        len(offense_team_ids)
                        == 1
                        and re.search(
                            r"\bpunts?\b",
                            text,
                            flags=re.IGNORECASE,
                        )
                    ):
                        offense_team_abbr = None

                        for side in (
                            home,
                            away,
                        ):
                            if (
                                str(
                                    side.get(
                                        "team_id"
                                    )
                                    or ""
                                )
                                == offense_team_ids[0]
                            ):
                                offense_team_abbr = str(
                                    side.get("abbr")
                                    or ""
                                ).strip().upper()

                        if offense_team_abbr:
                            team_prefixed_special_teams = bool(
                                re.search(
                                    r"\b"
                                    + re.escape(
                                        offense_team_abbr
                                    )
                                    + r"-"
                                    + escaped
                                    + r"\b",
                                    text,
                                )
                            )

                    non_fantasy_context = (
                        structured_two_sided_play
                        and (
                            defender_before_penalty
                            or defensive_penalty
                            or bracketed_defender
                            or team_prefixed_special_teams
                        )
                    )

                    if non_fantasy_context:

                        reasons = []

                        if defender_before_penalty:
                            reasons.append(
                                "DEFENDER_BEFORE_PENALTY"
                            )

                        if defensive_penalty:
                            reasons.append(
                                "DEFENSIVE_PENALTY"
                            )

                        if bracketed_defender:
                            reasons.append(
                                "BRACKETED_DEFENDER"
                            )

                        if team_prefixed_special_teams:
                            reasons.append(
                                "TEAM_PREFIXED_SPECIAL_TEAMS"
                            )

                        print(
                            "NON_FANTASY_ZERO_CANDIDATE_SKIPPED"
                            f" | play_id={play_id}"
                            f" | token={token}"
                            " | reason="
                            + ",".join(reasons)
                        )

                        zero_candidate_handled = True

                # ------------------------------------------------------
                # C. Preserve fail-closed behavior.
                # ------------------------------------------------------

                if not zero_candidate_handled:
                    unresolved_occurrences.append(
                        {
                            "play_id":
                                play_id,

                            "token":
                                token,

                            "identity_key":
                                key,

                            "text":
                                text,
                        }
                    )

        play_token_map[
            play_id
        ] = token_rows

    print(
        "RESOLVED_OCCURRENCES="
        f"{len(resolved_occurrences)}"
    )

    print(
        "AMBIGUOUS_OCCURRENCES="
        f"{len(ambiguous_occurrences)}"
    )

    print(
        "UNRESOLVED_OCCURRENCES="
        f"{len(unresolved_occurrences)}"
    )

    if ambiguous_occurrences:
        print()
        print(
            "--- AMBIGUOUS IDENTITIES ---"
        )

        for row in (
            ambiguous_occurrences
        ):
            print(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

    if unresolved_occurrences:
        print()
        print(
            "--- UNRESOLVED IDENTITIES ---"
        )

        for row in (
            unresolved_occurrences
        ):
            print(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

    # ==================================================================
    # 10. FAIL-CLOSED PRE-WRITE GATE
    # ==================================================================

    print()
    print(
        "=== 10. PRE-WRITE FAIL-CLOSED GATE ==="
    )

    if not resolved_occurrences:
        raise SystemExit(
            "FAIL | zero resolved player occurrences"
        )

    if ambiguous_occurrences:
        raise SystemExit(
            "FAIL | ambiguous identities remain "
            "— DATABASE UNCHANGED"
        )

    if unresolved_occurrences:
        raise SystemExit(
            "FAIL | unresolved identities remain "
            "— DATABASE UNCHANGED"
        )

    # Existing event history may never shrink.
    if (
        pre_play_rows > 0
        and unique_count
        < pre_play_rows
    ):
        raise SystemExit(
            "FAIL | current ESPN complete play "
            "population regressed below persisted "
            "population — DATABASE UNCHANGED"
        )

    print(
        "PASS | zero ambiguous identities"
    )

    print(
        "PASS | zero unresolved identities"
    )

    print(
        "PASS | complete current ESPN state assembled before write"
    )

    # ==================================================================
    # 11. DELTA CLASSIFICATION
    # ==================================================================

    print()
    print(
        "=== 11. CLASSIFY INGEST DELTA ==="
    )

    current_play_hashes = {
        str(play["id"]):
            play_hash(play)
        for play in plays
    }

    current_ids = set(
        current_play_hashes
    )

    pre_ids = set(
        pre_play_hashes
    )

    new_play_ids = sorted(
        current_ids - pre_ids
    )

    missing_prior_ids = sorted(
        pre_ids - current_ids
    )

    corrected_play_ids = sorted(
        play_id
        for play_id
        in current_ids.intersection(
            pre_ids
        )
        if (
            current_play_hashes[
                play_id
            ]
            != pre_play_hashes[
                play_id
            ]
        )
    )

    unchanged_play_ids = sorted(
        play_id
        for play_id
        in current_ids.intersection(
            pre_ids
        )
        if (
            current_play_hashes[
                play_id
            ]
            == pre_play_hashes[
                play_id
            ]
        )
    )

    print(
        f"NEW_PLAY_IDS={len(new_play_ids)}"
    )

    print(
        "CORRECTED_EXISTING_PLAY_IDS="
        f"{len(corrected_play_ids)}"
    )

    print(
        "UNCHANGED_EXISTING_PLAY_IDS="
        f"{len(unchanged_play_ids)}"
    )

    print(
        "MISSING_PRIOR_PLAY_IDS="
        f"{len(missing_prior_ids)}"
    )

    if missing_prior_ids:
        print(
            "MISSING_PRIOR_IDS="
            + ",".join(
                missing_prior_ids
            )
        )

        raise SystemExit(
            "FAIL | previously persisted stable "
            "play IDs disappeared from current "
            "ESPN state — DATABASE UNCHANGED"
        )

    # ==================================================================
    # 12. CURRENT DRIVE + LAST PLAY
    current_drive_id = str(
        current_drive.get("id")
        or ""
    ) or None

    last_play_id = (
        str(
            plays[-1].get("id")
        )
        if plays
        else None
    )

    captured_at = now_utc()

    # ==================================================================
    # 13. ATOMIC WRITE
    # ==================================================================

    print()
    print(
        "=== 12. BEGIN ATOMIC INGEST ==="
    )

    conn = sqlite3.connect(
        LIVE_DB,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    if (
        conn.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]
        != 1
    ):
        conn.close()

        raise SystemExit(
            "FAIL | foreign keys not enabled"
        )

    transaction_committed = False

    try:
        conn.execute(
            "BEGIN IMMEDIATE"
        )

        # ==============================================================
        # EVENT UPSERT
        # ==============================================================

        conn.execute(
            """
            INSERT INTO live_events (
                event_id,
                season,
                season_type,
                week,
                state,
                detail,
                period,
                clock,
                home_team_id,
                home_team,
                home_score,
                away_team_id,
                away_team,
                away_score,
                current_drive_id,
                last_play_id,
                updated_at_utc
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )

            ON CONFLICT(event_id)
            DO UPDATE SET
                season =
                    excluded.season,
                season_type =
                    excluded.season_type,
                week =
                    excluded.week,
                state =
                    excluded.state,
                detail =
                    excluded.detail,
                period =
                    excluded.period,
                clock =
                    excluded.clock,
                home_team_id =
                    excluded.home_team_id,
                home_team =
                    excluded.home_team,
                home_score =
                    excluded.home_score,
                away_team_id =
                    excluded.away_team_id,
                away_team =
                    excluded.away_team,
                away_score =
                    excluded.away_score,
                current_drive_id =
                    excluded.current_drive_id,
                last_play_id =
                    excluded.last_play_id,
                updated_at_utc =
                    excluded.updated_at_utc
            """,
            (
                event_id,
                int(season_value),
                (
                    int(season_type)
                    if season_type
                    is not None
                    else None
                ),
                (
                    int(week_value)
                    if week_value
                    is not None
                    else None
                ),
                event_state[
                    "state"
                ],
                event_state[
                    "detail"
                ],
                event_state[
                    "period"
                ],
                event_state[
                    "clock"
                ],
                home[
                    "team_id"
                ],
                home[
                    "abbr"
                ],
                home[
                    "score"
                ],
                away[
                    "team_id"
                ],
                away[
                    "abbr"
                ],
                away[
                    "score"
                ],
                current_drive_id,
                last_play_id,
                captured_at,
            ),
        )

        # ==============================================================
        # ATHLETES
        # ==============================================================

        for athlete in (
            merged_athletes.values()
        ):
            if not athlete.get(
                "identity_key"
            ):
                continue

            conn.execute(
                """
                INSERT INTO live_athletes (
                    athlete_id,
                    display_name,
                    team_id,
                    team,
                    identity_key,
                    identity_source,
                    position,
                    jersey,
                    updated_at_utc
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT(athlete_id)
                DO UPDATE SET
                    display_name =
                        excluded.display_name,

                    team_id =
                        excluded.team_id,

                    team =
                        excluded.team,

                    identity_key =
                        excluded.identity_key,

                    identity_source =
                        CASE
                            WHEN
                                live_athletes.identity_source
                                = 'boxscore'
                            THEN
                                'boxscore'

                            WHEN
                                excluded.identity_source
                                = 'boxscore'
                            THEN
                                'boxscore'

                            ELSE
                                excluded.identity_source
                        END,

                    position =
                        COALESCE(
                            excluded.position,
                            live_athletes.position
                        ),

                    jersey =
                        COALESCE(
                            excluded.jersey,
                            live_athletes.jersey
                        ),

                    updated_at_utc =
                        excluded.updated_at_utc
                """,
                (
                    athlete[
                        "athlete_id"
                    ],
                    athlete[
                        "display_name"
                    ],
                    athlete[
                        "team_id"
                    ],
                    athlete[
                        "team"
                    ],
                    athlete[
                        "identity_key"
                    ],
                    athlete[
                        "identity_source"
                    ],
                    athlete.get(
                        "position"
                    ),
                    (
                        str(
                            athlete.get(
                                "jersey"
                            )
                        )
                        if athlete.get(
                            "jersey"
                        )
                        is not None
                        else None
                    ),
                    captured_at,
                ),
            )

        # ==============================================================
        # PLAYS + CURRENT PLAYER REFERENCES
        # ==============================================================

        for play in plays:
            play_id = str(
                play.get("id")
            )

            conn.execute(
                """
                INSERT INTO live_plays (
                    event_id,
                    play_id,
                    sequence_number,
                    period,
                    clock,
                    home_score,
                    away_score,
                    play_text,
                    is_scoring_play,
                    is_turnover,
                    is_penalty,
                    play_sha256,
                    first_seen_at_utc,
                    updated_at_utc
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT(
                    event_id,
                    play_id
                )
                DO UPDATE SET
                    sequence_number =
                        excluded.sequence_number,

                    period =
                        excluded.period,

                    clock =
                        excluded.clock,

                    home_score =
                        excluded.home_score,

                    away_score =
                        excluded.away_score,

                    play_text =
                        excluded.play_text,

                    is_scoring_play =
                        excluded.is_scoring_play,

                    is_turnover =
                        excluded.is_turnover,

                    is_penalty =
                        excluded.is_penalty,

                    play_sha256 =
                        excluded.play_sha256,

                    updated_at_utc =
                        excluded.updated_at_utc
                """,
                (
                    event_id,
                    play_id,
                    sequence_key(
                        play
                    ),
                    get_period_number(
                        play
                    ),
                    get_clock_display(
                        play
                    ),
                    play.get(
                        "homeScore"
                    ),
                    play.get(
                        "awayScore"
                    ),
                    play.get(
                        "text"
                    )
                    or "",
                    bool_int(
                        play.get(
                            "scoringPlay"
                        )
                    ),
                    bool_int(
                        play.get(
                            "isTurnover"
                        )
                    ),
                    bool_int(
                        play.get(
                            "isPenalty"
                        )
                    ),
                    play_hash(
                        play
                    ),
                    captured_at,
                    captured_at,
                ),
            )

            # Current ESPN representation wins.
            conn.execute(
                """
                DELETE FROM live_play_players
                WHERE event_id = ?
                  AND play_id = ?
                """,
                (
                    event_id,
                    play_id,
                ),
            )

            for token_row in (
                play_token_map.get(
                    play_id,
                    [],
                )
            ):
                conn.execute(
                    """
                    INSERT INTO live_play_players (
                        event_id,
                        play_id,
                        token_ordinal,
                        raw_token,
                        identity_key,
                        athlete_id,
                        resolution_method,
                        identity_source
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        event_id,
                        play_id,
                        token_row[
                            "token_ordinal"
                        ],
                        token_row[
                            "raw_token"
                        ],
                        token_row[
                            "identity_key"
                        ],
                        token_row[
                            "athlete_id"
                        ],
                        token_row[
                            "resolution_method"
                        ],
                        token_row[
                            "identity_source"
                        ],
                    ),
                )

        # ==============================================================
        # INGEST AUDIT
        # ==============================================================

        conn.execute(
            """
            INSERT INTO live_ingest_audit (
                event_id,
                captured_at_utc,
                event_state,
                raw_play_rows,
                unique_play_rows,
                duplicates_removed,
                boxscore_athletes,
                roster_athletes,
                merged_athletes,
                resolved_occurrences,
                ambiguous_occurrences,
                unresolved_occurrences,
                ingest_status
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                event_id,
                captured_at,
                event_state[
                    "state"
                ],
                raw_count,
                unique_count,
                duplicate_count,
                len(
                    boxscore_athletes
                ),
                len(
                    roster_athletes
                ),
                len(
                    merged_athletes
                ),
                len(
                    resolved_occurrences
                ),
                0,
                0,
                "PASS",
            ),
        )

        # ==============================================================
        # PRE-COMMIT RECONCILIATION
        # ==============================================================

        print()
        print(
            "=== 13. PRE-COMMIT RECONCILIATION ==="
        )

        tx_event_rows = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM live_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()[0]
        )

        tx_play_rows = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM live_plays
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()[0]
        )

        tx_player_rows = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM live_play_players
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()[0]
        )

        tx_audit_rows = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM live_ingest_audit
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()[0]
        )

        duplicate_play_groups = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        event_id,
                        play_id,
                        COUNT(*) AS n
                    FROM live_plays
                    WHERE event_id = ?
                    GROUP BY
                        event_id,
                        play_id
                    HAVING COUNT(*) > 1
                )
                """,
                (event_id,),
            ).fetchone()[0]
        )

        duplicate_player_groups = (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        event_id,
                        play_id,
                        token_ordinal,
                        COUNT(*) AS n
                    FROM live_play_players
                    WHERE event_id = ?
                    GROUP BY
                        event_id,
                        play_id,
                        token_ordinal
                    HAVING COUNT(*) > 1
                )
                """,
                (event_id,),
            ).fetchone()[0]
        )

        fk_precommit = (
            conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        print(
            f"TX_EVENT_ROWS={tx_event_rows}"
        )

        print(
            f"TX_PLAY_ROWS={tx_play_rows}"
        )

        print(
            "TX_PLAY_PLAYER_ROWS="
            f"{tx_player_rows}"
        )

        print(
            "TX_INGEST_AUDIT_ROWS="
            f"{tx_audit_rows}"
        )

        print(
            "DUPLICATE_PLAY_GROUPS="
            f"{duplicate_play_groups}"
        )

        print(
            "DUPLICATE_PLAY_PLAYER_GROUPS="
            f"{duplicate_player_groups}"
        )

        print(
            "PRECOMMIT_FK_ERRORS="
            f"{len(fk_precommit)}"
        )

        if tx_event_rows != 1:
            raise RuntimeError(
                "event idempotency failure"
            )

        if tx_play_rows != unique_count:
            raise RuntimeError(
                "complete play reconciliation failure"
            )

        if (
            tx_player_rows
            != len(
                resolved_occurrences
            )
        ):
            raise RuntimeError(
                "player-reference reconciliation failure"
            )

        if (
            tx_audit_rows
            != pre_audit_rows + 1
        ):
            raise RuntimeError(
                "audit increment failure"
            )

        if duplicate_play_groups:
            raise RuntimeError(
                "duplicate stable play IDs"
            )

        if duplicate_player_groups:
            raise RuntimeError(
                "duplicate player references"
            )

        if fk_precommit:
            raise RuntimeError(
                "precommit foreign-key failure"
            )

        conn.commit()

        transaction_committed = True

    except Exception as exc:
        conn.rollback()

        print()
        print(
            "INGEST_TRANSACTION=ROLLED_BACK"
        )

        print(
            "ERROR="
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        conn.close()
        raise

    conn.close()

    if not transaction_committed:
        raise SystemExit(
            "FAIL | transaction did not commit"
        )

    print()
    print(
        "INGEST_TRANSACTION=COMMITTED"
    )

    # ==================================================================
    # 14. READ-ONLY POST-COMMIT VERIFY
    # ==================================================================

    print()
    print(
        "=== 14. READ-ONLY POST-COMMIT VERIFICATION ==="
    )

    verify = sqlite3.connect(
        ro_uri,
        uri=True,
    )

    verify.row_factory = (
        sqlite3.Row
    )

    integrity_after = (
        verify.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
    )

    fk_after = (
        verify.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    )

    post_event_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()[0]
    )

    post_athlete_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_athletes
            """
        ).fetchone()[0]
    )

    post_play_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_plays
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()[0]
    )

    post_player_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_play_players
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()[0]
    )

    post_audit_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_ingest_audit
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()[0]
    )

    post_event = (
        verify.execute(
            """
            SELECT *
            FROM live_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
    )

    duplicate_play_groups_after = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    event_id,
                    play_id,
                    COUNT(*) AS n
                FROM live_plays
                WHERE event_id = ?
                GROUP BY
                    event_id,
                    play_id
                HAVING COUNT(*) > 1
            )
            """,
            (event_id,),
        ).fetchone()[0]
    )

    duplicate_player_groups_after = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    event_id,
                    play_id,
                    token_ordinal,
                    COUNT(*) AS n
                FROM live_play_players
                WHERE event_id = ?
                GROUP BY
                    event_id,
                    play_id,
                    token_ordinal
                HAVING COUNT(*) > 1
            )
            """,
            (event_id,),
        ).fetchone()[0]
    )

    orphan_player_rows = (
        verify.execute(
            """
            SELECT COUNT(*)
            FROM live_play_players AS pp
            LEFT JOIN live_plays AS p
              ON p.event_id = pp.event_id
             AND p.play_id = pp.play_id
            LEFT JOIN live_athletes AS a
              ON a.athlete_id = pp.athlete_id
            WHERE pp.event_id = ?
              AND (
                  p.play_id IS NULL
                  OR a.athlete_id IS NULL
              )
            """,
            (event_id,),
        ).fetchone()[0]
    )

    last_audit = (
        verify.execute(
            """
            SELECT *
            FROM live_ingest_audit
            WHERE event_id = ?
            ORDER BY ingest_id DESC
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
    )

    verify.close()

    print(
        f"INTEGRITY_CHECK={integrity_after}"
    )

    print(
        f"FOREIGN_KEY_ERRORS={len(fk_after)}"
    )

    print(
        f"POST_EVENT_ROWS={post_event_rows}"
    )

    print(
        f"POST_ATHLETE_ROWS={post_athlete_rows}"
    )

    print(
        f"POST_PLAY_ROWS={post_play_rows}"
    )

    print(
        "POST_PLAY_PLAYER_ROWS="
        f"{post_player_rows}"
    )

    print(
        "POST_INGEST_AUDIT_ROWS="
        f"{post_audit_rows}"
    )

    print(
        "DUPLICATE_PLAY_GROUPS="
        f"{duplicate_play_groups_after}"
    )

    print(
        "DUPLICATE_PLAY_PLAYER_GROUPS="
        f"{duplicate_player_groups_after}"
    )

    print(
        f"ORPHAN_PLAYER_ROWS={orphan_player_rows}"
    )

    if post_event is not None:
        print(
            f"POST_STATE={post_event['state']}"
        )

        print(
            f"POST_DETAIL={post_event['detail']}"
        )

        print(
            f"POST_PERIOD={post_event['period']}"
        )

        print(
            f"POST_CLOCK={post_event['clock']}"
        )

        print(
            "POST_LAST_PLAY_ID="
            f"{post_event['last_play_id']}"
        )

    print(
        "LAST_AUDIT_STATUS="
        f"{last_audit['ingest_status']}"
    )

    print(
        "LAST_AUDIT_RESOLVED="
        f"{last_audit['resolved_occurrences']}"
    )

    print(
        "LAST_AUDIT_AMBIGUOUS="
        f"{last_audit['ambiguous_occurrences']}"
    )

    print(
        "LAST_AUDIT_UNRESOLVED="
        f"{last_audit['unresolved_occurrences']}"
    )

    # ==================================================================
    # 15. POST-COMMIT HARD GATES
    # ==================================================================

    if integrity_after != "ok":
        raise SystemExit(
            "FAIL | post-commit integrity failure"
        )

    if fk_after:
        raise SystemExit(
            "FAIL | post-commit foreign-key failure"
        )

    if post_event_rows != 1:
        raise SystemExit(
            "FAIL | event row count mismatch"
        )

    if post_play_rows != unique_count:
        raise SystemExit(
            "FAIL | persisted/current play mismatch"
        )

    if (
        post_player_rows
        != len(
            resolved_occurrences
        )
    ):
        raise SystemExit(
            "FAIL | persisted/current "
            "player-reference mismatch"
        )

    if (
        post_audit_rows
        != pre_audit_rows + 1
    ):
        raise SystemExit(
            "FAIL | ingest audit did not "
            "increment exactly once"
        )

    if duplicate_play_groups_after:
        raise SystemExit(
            "FAIL | duplicate play IDs"
        )

    if duplicate_player_groups_after:
        raise SystemExit(
            "FAIL | duplicate play-player rows"
        )

    if orphan_player_rows:
        raise SystemExit(
            "FAIL | orphan player references"
        )

    if (
        last_audit[
            "ingest_status"
        ]
        != "PASS"
    ):
        raise SystemExit(
            "FAIL | last ingest audit not PASS"
        )

    if (
        last_audit[
            "ambiguous_occurrences"
        ]
        != 0
    ):
        raise SystemExit(
            "FAIL | ambiguous identities persisted"
        )

    if (
        last_audit[
            "unresolved_occurrences"
        ]
        != 0
    ):
        raise SystemExit(
            "FAIL | unresolved identities persisted"
        )

    # ==================================================================
    # 16. PROTECTED DB VERIFICATION
    # ==================================================================

    print()
    print(
        "=== 16. PROTECTED DATABASE VERIFICATION ==="
    )

    nfl_sha_after = sha256_file(
        NFL_DB
    )

    ledger_sha_after = sha256_file(
        FORECAST_LEDGER
    )

    nfl_unchanged = (
        nfl_sha_before
        == nfl_sha_after
    )

    ledger_unchanged = (
        ledger_sha_before
        == ledger_sha_after
    )

    print(
        f"NFL_DB_SHA_AFTER={nfl_sha_after}"
    )

    print(
        "FORECAST_LEDGER_SHA_AFTER="
        f"{ledger_sha_after}"
    )

    print(
        f"NFL_DB_UNCHANGED={int(nfl_unchanged)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    if not nfl_unchanged:
        raise SystemExit(
            "FAIL | nfl.db changed"
        )

    if not ledger_unchanged:
        raise SystemExit(
            "FAIL | forecast_ledger.db changed"
        )

    # ==================================================================
    # 17. DELTA SUMMARY
    # ==================================================================

    play_row_delta = (
        post_play_rows
        - pre_play_rows
    )

    player_row_delta = (
        post_player_rows
        - pre_player_rows
    )

    athlete_row_delta = (
        post_athlete_rows
        - pre_athlete_rows
    )

    audit_delta = (
        post_audit_rows
        - pre_audit_rows
    )

    print()
    print(
        "=== 17. INGEST DELTA SUMMARY ==="
    )

    print(
        f"PLAY_ROW_DELTA={play_row_delta}"
    )

    print(
        "PLAY_PLAYER_ROW_DELTA="
        f"{player_row_delta}"
    )

    print(
        "ATHLETE_ROW_DELTA="
        f"{athlete_row_delta}"
    )

    print(
        f"INGEST_AUDIT_DELTA={audit_delta}"
    )

    print(
        f"NEW_PLAY_IDS={len(new_play_ids)}"
    )

    print(
        "CORRECTED_EXISTING_PLAY_IDS="
        f"{len(corrected_play_ids)}"
    )

    print(
        "UNCHANGED_EXISTING_PLAY_IDS="
        f"{len(unchanged_play_ids)}"
    )

    print(
        "MISSING_PRIOR_PLAY_IDS="
        f"{len(missing_prior_ids)}"
    )

    # ==================================================================
    # 18. FINAL CONTRACT
    # ==================================================================

    print()
    print(
        "=== 18. WFS LIVE INGEST CONTRACT ==="
    )

    print(
        "PASS | WFS_LIVE_DB_V2 schema verified"
    )

    print(
        "PASS | LIVE schema fingerprint verified"
    )

    print(
        "PASS | ESPN summary fetched"
    )

    print(
        "PASS | drive-tree collector active"
    )

    print(
        "PASS | stable ESPN play IDs deduplicated"
    )

    print(
        "PASS | boxscore identity source active"
    )

    print(
        "PASS | roster identity fallback active"
    )

    print(
        "PASS | boxscore identity precedence active"
    )

    print(
        "PASS | collisions surfaced deterministically"
    )

    print(
        "PASS | zero ambiguous identities"
    )

    print(
        "PASS | zero unresolved identities"
    )

    print(
        "PASS | no prior stable play IDs disappeared"
    )

    print(
        "PASS | atomic SQLite ingest committed"
    )

    print(
        "PASS | event remains one row"
    )

    print(
        "PASS | current plays reconciled"
    )

    print(
        "PASS | current player references reconciled"
    )

    print(
        "PASS | corrected play representations replace prior state"
    )

    print(
        "PASS | ingest audit incremented exactly once"
    )

    print(
        "PASS | no duplicate play rows"
    )

    print(
        "PASS | no duplicate play-player rows"
    )

    print(
        "PASS | no orphan player references"
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

    print()
    print("=" * 72)

    print(
        "PASS | WFS LIVE EVENT INGEST COMPLETE"
    )

    print("=" * 72)

    print(
        f"EVENT_ID={event_id}"
    )

    print(
        f"EVENT_STATE={event_state['state']}"
    )

    print(
        f"EVENT_DETAIL={event_state['detail']}"
    )

    print(
        f"PRE_PLAY_ROWS={pre_play_rows}"
    )

    print(
        f"POST_PLAY_ROWS={post_play_rows}"
    )

    print(
        f"PLAY_ROW_DELTA={play_row_delta}"
    )

    print(
        "PRE_PLAY_PLAYER_ROWS="
        f"{pre_player_rows}"
    )

    print(
        "POST_PLAY_PLAYER_ROWS="
        f"{post_player_rows}"
    )

    print(
        "PLAY_PLAYER_ROW_DELTA="
        f"{player_row_delta}"
    )

    print(
        f"NEW_PLAY_IDS={len(new_play_ids)}"
    )

    print(
        "CORRECTED_EXISTING_PLAY_IDS="
        f"{len(corrected_play_ids)}"
    )

    print(
        "MISSING_PRIOR_PLAY_IDS="
        f"{len(missing_prior_ids)}"
    )

    print(
        "AMBIGUOUS_OCCURRENCES="
        f"{len(ambiguous_occurrences)}"
    )

    print(
        "UNRESOLVED_OCCURRENCES="
        f"{len(unresolved_occurrences)}"
    )

    print(
        f"POST_EVENT_ROWS={post_event_rows}"
    )

    print(
        f"POST_INGEST_AUDIT_ROWS={post_audit_rows}"
    )

    print(
        f"NFL_DB_UNCHANGED={int(nfl_unchanged)}"
    )

    print(
        "FORECAST_LEDGER_UNCHANGED="
        f"{int(ledger_unchanged)}"
    )

    print(
        "INGEST_TRANSACTION=COMMITTED"
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "WFS NFL deterministic live ESPN "
            "event ingest runner"
        )
    )

    parser.add_argument(
        "--event-id",
        required=True,
        help="ESPN NFL event ID",
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=(
            "Fallback season used only if "
            "ESPN header season is absent"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    event_id = (
        str(args.event_id)
        .strip()
    )

    if not event_id:
        raise SystemExit(
            "FAIL | empty event ID"
        )

    ingest_event(
        event_id=event_id,
        default_season=args.season,
    )


if __name__ == "__main__":
    main()
