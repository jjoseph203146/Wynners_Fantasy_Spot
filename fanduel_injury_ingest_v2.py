#!/usr/bin/env python3

"""
WFS NFL — FanDuel Player News Multicategory Ingest V3

WFS_GLOBAL_PLAYER_AVAILABILITY_V2

Required typed acquisition:
    INJURIES
    GAME_UPDATES
    TRANSACTIONS

Supplemental/degradable acquisition:
    GENERAL_PLAYER_NEWS

Identity:
    Exact deterministic WFS identity only.
    No fuzzy matching.
    No display-name-only identity authority.

Typed FanDuel newsType enums are discovered from each typed
endpoint's embedded first page. They are not guessed or hardcoded.

--dry-run:
    network reads + read-only DB identity validation only.
    no DB mutation.
    no CSV writes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sqlite3
import time
import unicodedata

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from wfs_schedule_context import resolve_schedule_week_context


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "nfl.db"
CSV_DIR = ROOT / "data" / "csv"

INGEST_CSV = (
    CSV_DIR
    / "fanduel_injury_ingest_current.csv"
)

QUARANTINE_CSV = (
    CSV_DIR
    / "fanduel_injury_quarantine.csv"
)

GRAPHQL_URL = (
    "https://www.fanduel.com/research/api/graphql"
)

SOURCE = "FANDUEL_RESEARCH"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/140 Safari/537.36"
)

PAGE_LIMIT = 10
REQUIRED_MAX_PAGES = 120
GENERAL_MAX_PAGES = 120

HTTP_CONNECT_TIMEOUT = 20
HTTP_READ_TIMEOUT = 60
HTTP_MAX_ATTEMPTS = 4
HTTP_BACKOFF_SECONDS = (2, 5, 10)

ET = ZoneInfo("America/New_York")

WINDOW_BEFORE = timedelta(days=7)
WINDOW_AFTER = timedelta(hours=24)

MULTICATEGORY_SOURCES = {
    "INJURIES": {
        "url":
            "https://www.fanduel.com/research/nfl/player-news/injuries",
        "required":
            True,
        "typed":
            True,
    },

    "GAME_UPDATES": {
        "url":
            "https://www.fanduel.com/research/nfl/player-news/game-updates",
        "required":
            True,
        "typed":
            True,
    },

    "TRANSACTIONS": {
        "url":
            "https://www.fanduel.com/research/nfl/player-news/transactions",
        "required":
            True,
        "typed":
            True,
    },

    "GENERAL_PLAYER_NEWS": {
        "url":
            "https://www.fanduel.com/research/nfl/player-news",
        "required":
            False,
        "typed":
            False,
    },
}


# Match the embedded HTML event coverage so pagination duplicates can
# pass the existing full-payload comparison without excluding any fields.
PLAYER_NEWS_QUERY = r"""
query getPlayerNewsQuery($filter: ShortFormSearchInput!) {
  getShortForms(filter: $filter) {
    pageInfo {
      cursor
      hasNextPage
    }
    shortForms {
      cursor
      entity {
        id
        slug
        title
        fact
        analysis
        quant
        newsType {
          name
          enum
        }
        primaryRef {
          ... on Player {
            image { url }
            identifier
            numberFireId
            numberFireSlug
            name
            number
            position
            playerPageUrl
            team {
              sports { id name slug }
              image { url }
              teamJerseyImage { url }
              sportsbookLink
              abbreviation
              name
            }
          }
        }
        sport { id name slug contentSport logo { url } }
        author { id name thumbnailUrl socialMedia { __typename } }
        attribution { __typename }
        firstPublishedAt
        lastPublishedAt
        description
      }
    }
  }
}
""".strip()


TEAM_ALIASES = {
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LA",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
}

DIRECT_TEAMS = {
    x: x
    for x in [
        "ARI", "ATL", "BAL", "BUF",
        "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB",
        "HOU", "IND", "JAX", "KC",
        "LV", "LAC", "LA", "MIA",
        "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SF",
        "SEA", "TB", "TEN", "WAS",
    ]
}

DIRECT_TEAMS.update({
    "GBP": "GB",
    "JAC": "JAX",
    "KAN": "KC",
    "LVR": "LV",
    "LAR": "LA",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "WSH": "WAS",
})


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value: Any) -> str:

    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def normalize_name(value: Any) -> str:

    text = unicodedata.normalize(
        "NFKD",
        clean_text(value),
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    ).upper()

    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^A-Z0-9]+",
            " ",
            text,
        ),
    ).strip()


def normalize_team(value: Any) -> str:

    text = clean_text(value).upper()

    if not text:
        return ""

    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    return DIRECT_TEAMS.get(
        compact,
        text,
    )


def normalize_position(value: Any) -> str:

    text = clean_text(value).upper()

    return {
        "HB": "RB",
        "FB": "RB",
    }.get(
        text,
        text,
    )


def flatten_blocks(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):

        raw = value.strip()

        if not raw:
            return ""

        try:
            return flatten_blocks(
                json.loads(raw)
            )
        except Exception:
            return raw

    if isinstance(value, list):
        return " ".join(
            filter(
                None,
                (
                    flatten_blocks(x)
                    for x in value
                ),
            )
        ).strip()

    if isinstance(value, dict):

        if "text" in value:
            return clean_text(
                value.get("text")
            )

        return " ".join(
            filter(
                None,
                (
                    flatten_blocks(x)
                    for x
                    in value.values()
                ),
            )
        ).strip()

    return clean_text(value)


def entity_news_type(entity: Any) -> str:

    if not isinstance(entity, dict):
        return ""

    value = entity.get("newsType") or {}

    if not isinstance(value, dict):
        return ""

    return clean_text(
        value.get("enum")
    ).upper()


HTTP_SESSION = requests.Session()

HTTP_SESSION.headers.update({
    "User-Agent":
        USER_AGENT,
})


def request_with_retry(
    method: str,
    url: str,
    **kwargs,
) -> requests.Response:

    last_error = None

    for attempt in range(
        1,
        HTTP_MAX_ATTEMPTS + 1,
    ):

        try:

            response = HTTP_SESSION.request(
                method=method,
                url=url,
                timeout=(
                    HTTP_CONNECT_TIMEOUT,
                    HTTP_READ_TIMEOUT,
                ),
                **kwargs,
            )

            if (
                response.status_code == 429
                or 500 <= response.status_code <= 599
            ):

                raise requests.HTTPError(
                    "Transient FanDuel HTTP "
                    f"{response.status_code}",
                    response=response,
                )

            response.raise_for_status()

            return response

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:

            last_error = exc

            retryable = True

            if isinstance(
                exc,
                requests.HTTPError,
            ):

                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )

                retryable = (
                    status == 429
                    or (
                        status is not None
                        and 500 <= status <= 599
                    )
                )

            if (
                not retryable
                or attempt >= HTTP_MAX_ATTEMPTS
            ):
                raise

            delay = HTTP_BACKOFF_SECONDS[
                attempt - 1
            ]

            print(
                "HTTP_RETRY|"
                f"METHOD={method.upper()}|"
                f"ATTEMPT={attempt}|"
                f"NEXT_ATTEMPT={attempt + 1}|"
                f"DELAY_SECONDS={delay}|"
                f"ERROR={type(exc).__name__}"
            )

            time.sleep(delay)

    raise RuntimeError(
        "FanDuel request retry loop "
        f"exhausted: {last_error}"
    )


def http_get(url: str) -> requests.Response:

    return request_with_retry(
        "GET",
        url,
    )


def fetch_next_data(url: str) -> dict:

    response = http_get(url)

    matches = re.findall(
        (
            r'<script[^>]*'
            r'id=["\']__NEXT_DATA__["\']'
            r'[^>]*>(.*?)</script>'
        ),
        response.text,
        flags=re.I | re.S,
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one __NEXT_DATA__ "
            f"block; found {len(matches)}"
        )

    data = json.loads(
        html.unescape(
            matches[0].strip()
        )
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "FanDuel __NEXT_DATA__ root "
            "is not an object"
        )

    return data


def embedded_candidates(data: Any):

    found = []

    def walk(obj):

        if isinstance(obj, dict):

            gs = obj.get(
                "getShortForms"
            )

            if (
                isinstance(gs, dict)
                and isinstance(
                    gs.get("shortForms"),
                    list,
                )
                and isinstance(
                    gs.get("pageInfo"),
                    dict,
                )
            ):

                entities = []

                for wrapper in gs[
                    "shortForms"
                ]:

                    if not isinstance(
                        wrapper,
                        dict,
                    ):
                        continue

                    entity = wrapper.get(
                        "entity"
                    )

                    if isinstance(
                        entity,
                        dict,
                    ):
                        entities.append(entity)

                if entities:
                    found.append(
                        (
                            entities,
                            gs["pageInfo"],
                        )
                    )

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):

            for value in obj:
                walk(value)

    walk(data)

    return found


def detect_typed_embedded_page(
    data: dict,
    family: str,
) -> tuple[
    list[dict],
    dict,
    str,
]:

    candidates = (
        embedded_candidates(data)
    )

    scored = []

    for entities, page_info in candidates:

        counts = {}

        for entity in entities:

            enum = entity_news_type(
                entity
            )

            if enum:
                counts[enum] = (
                    counts.get(enum, 0)
                    + 1
                )

        if not counts:
            continue

        ordered = sorted(
            counts.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
            reverse=True,
        )

        dominant_enum, count = (
            ordered[0]
        )

        # Typed page must be homogeneous enough
        # to establish its observed category.
        matching = [
            entity
            for entity in entities
            if entity_news_type(entity)
            == dominant_enum
        ]

        scored.append(
            (
                count,
                len(entities),
                dominant_enum,
                matching,
                page_info,
            )
        )

    if not scored:
        raise RuntimeError(
            f"No typed embedded page "
            f"detected for {family}"
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    best = scored[0]

    enum = best[2]
    entities = best[3]
    page_info = best[4]

    if not enum:
        raise RuntimeError(
            f"Blank observed newsType enum "
            f"for {family}"
        )

    if not entities:
        raise RuntimeError(
            f"Zero typed entities for {family}"
        )

    return (
        entities,
        page_info,
        enum,
    )


def detect_general_embedded_page(
    data: dict,
) -> tuple[
    list[dict],
    dict,
]:

    candidates = (
        embedded_candidates(data)
    )

    if not candidates:
        raise RuntimeError(
            "No General Player News "
            "embedded page found"
        )

    candidates.sort(
        key=lambda item: len(
            item[0]
        ),
        reverse=True,
    )

    return candidates[0]


def graphql_page(
    cursor: str,
    news_type: str | None,
) -> tuple[
    list[dict],
    dict,
]:

    variables = {
        "filter": {
            "afterCursor":
                cursor,

            "limit":
                PAGE_LIMIT,

            "player": {
                "positionAbbrev":
                    None,
            },

            "publishedWithin":
                None,

            "shortFormNewsType":
                news_type,

            "sport":
                "NFL",

            "team": {
                "numberFireId":
                    None,
            },
        }
    }

    response = request_with_retry(
        "POST",
        GRAPHQL_URL,
        headers={
            "Content-Type":
                "application/json",
        },
        json={
            "query":
                PLAYER_NEWS_QUERY,

            "variables":
                variables,
        },
    )

    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(
            "FanDuel GraphQL errors: "
            f"{payload['errors']}"
        )

    result = (
        (payload.get("data") or {})
        .get("getShortForms")
        or {}
    )

    rows = result.get("shortForms")
    page_info = result.get("pageInfo")

    if (
        not isinstance(rows, list)
        or not isinstance(
            page_info,
            dict,
        )
    ):
        raise RuntimeError(
            "FanDuel GraphQL response "
            "contract changed"
        )

    entities = []

    for wrapper in rows:

        if not isinstance(
            wrapper,
            dict,
        ):
            continue

        entity = wrapper.get("entity")

        if not isinstance(
            entity,
            dict,
        ):
            continue

        if (
            news_type is None
            or entity_news_type(
                entity
            ) == news_type
        ):
            entities.append(entity)

    return (
        entities,
        page_info,
    )


def parse_source_timestamp(
    value: Any,
) -> datetime | None:

    text = clean_text(value)

    if not text:
        return None

    try:
        dt = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    return dt.astimezone(
        timezone.utc
    )


def page_before_window(
    entities: list[dict],
    start_utc: datetime,
) -> bool:

    values = []

    for entity in entities:

        ts = parse_source_timestamp(
            entity.get(
                "firstPublishedAt"
            )
        )

        if ts is not None:
            values.append(ts)

    return bool(
        entities
        and len(values) == len(entities)
        and max(values) < start_utc
    )


def fetch_family(
    family: str,
    config: dict,
    evidence_start_utc: datetime,
):

    required = bool(
        config["required"]
    )

    typed = bool(
        config["typed"]
    )

    url = config["url"]

    audit = {
        "family":
            family,

        "required":
            required,

        "typed":
            typed,

        "url":
            url,

        "observed_news_type":
            "",

        "pages":
            0,

        "raw_rows":
            0,

        "unique_events":
            0,

        "boundary_duplicates":
            0,

        "terminal_has_next_page":
            None,

        "window_cutoff":
            False,

        "degraded":
            False,

        "degraded_reason":
            "",
    }

    try:

        data = fetch_next_data(url)

        if typed:

            (
                first,
                page_info,
                observed_enum,
            ) = detect_typed_embedded_page(
                data,
                family,
            )

            audit[
                "observed_news_type"
            ] = observed_enum

            news_type = observed_enum

        else:

            (
                first,
                page_info,
            ) = detect_general_embedded_page(
                data
            )

            news_type = None

        unique = {}

        audit["pages"] = 1
        audit["raw_rows"] = len(first)

        for entity in first:

            event_id = clean_text(
                entity.get("id")
            )

            if not event_id:
                raise RuntimeError(
                    f"Blank event ID in {family}"
                )

            unique[event_id] = entity

        current_page = first

        max_pages = (
            REQUIRED_MAX_PAGES
            if required
            else GENERAL_MAX_PAGES
        )

        while bool(
            page_info.get(
                "hasNextPage"
            )
        ):

            if page_before_window(
                current_page,
                evidence_start_utc,
            ):
                audit["window_cutoff"] = True
                audit[
                    "degraded_reason"
                ] = (
                    family
                    + "_BEFORE_EVIDENCE_WINDOW"
                )
                break
            if (
                audit["pages"]
                >= max_pages
            ):

                if required:
                    raise RuntimeError(
                        f"{family} exceeded "
                        f"MAX_PAGES={max_pages}"
                    )

                audit["degraded"] = True

                audit[
                    "degraded_reason"
                ] = (
                    "GENERAL_PLAYER_NEWS_"
                    "MAX_PAGE_SAFETY_CAP"
                )

                break

            cursor = clean_text(
                page_info.get(
                    "cursor"
                )
            )

            if not cursor:
                raise RuntimeError(
                    f"{family}: blank cursor "
                    "while hasNextPage=true"
                )

            try:

                (
                    page,
                    page_info,
                ) = graphql_page(
                    cursor,
                    news_type,
                )

            except Exception as exc:

                if required:
                    raise

                audit["degraded"] = True

                audit[
                    "degraded_reason"
                ] = (
                    "GENERAL_PLAYER_NEWS_"
                    "FETCH_ERROR:"
                    + type(exc).__name__
                    + ":"
                    + str(exc)
                )

                break

            audit["pages"] += 1
            audit["raw_rows"] += len(page)

            current_page = page

            for entity in page:

                event_id = clean_text(
                    entity.get("id")
                )

                if not event_id:
                    raise RuntimeError(
                        f"Blank event ID "
                        f"in {family}"
                    )

                if event_id in unique:

                    left = json.dumps(
                        unique[event_id],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )

                    right = json.dumps(
                        entity,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )

                    if left != right:
                        raise RuntimeError(
                            "Duplicate event ID "
                            "changed payload: "
                            f"{event_id}"
                        )

                    print(
                        "DUPLICATE_EQUIVALENT|"
                        "REASON=FULL_CANONICAL_PAYLOAD_EQUAL|"
                        "EXCLUDED_FIELDS=NONE|"
                        "PAYLOAD_SHA256="
                        + hashlib.sha256(
                            left.encode("utf-8")
                        ).hexdigest()
                    )

                    audit[
                        "boundary_duplicates"
                    ] += 1

                else:
                    unique[
                        event_id
                    ] = entity

        audit[
            "terminal_has_next_page"
        ] = bool(
            page_info.get(
                "hasNextPage"
            )
        )

        if (
            required
            and audit[
                "terminal_has_next_page"
            ]
            and not audit[
                "window_cutoff"
            ]
        ):
            raise RuntimeError(
                f"Required family {family} "
                "did not terminate"
            )

        audit[
            "unique_events"
        ] = len(unique)

        return (
            list(unique.values()),
            audit,
        )

    except Exception as exc:

        if required:
            raise

        audit["degraded"] = True
        audit["degraded_reason"] = (
            type(exc).__name__
            + ":"
            + str(exc)
        )

        return (
            [],
            audit,
        )


def fetch_all_families(
    evidence_start_utc: datetime,
):

    combined = {}
    audits = []

    for family, config in (
        MULTICATEGORY_SOURCES.items()
    ):

        events, audit = fetch_family(
            family,
            config,
            evidence_start_utc,
        )

        audits.append(audit)

        for entity in events:

            event_id = clean_text(
                entity.get("id")
            )

            if event_id not in combined:

                combined[event_id] = {
                    "entity":
                        entity,

                    "families":
                        set(),
                }

            combined[
                event_id
            ][
                "families"
            ].add(
                family
            )

    output = []

    for event_id in sorted(combined):

        item = combined[event_id]

        entity = dict(
            item["entity"]
        )

        entity[
            "_wfs_source_categories"
        ] = sorted(
            item["families"]
        )

        output.append(entity)

    return output, audits


def extract_player_name(
    entity: dict,
) -> str:

    primary = (
        entity.get("primaryRef")
        or {}
    )

    for key in (
        "name",
        "fullName",
        "displayName",
        "playerName",
    ):

        value = clean_text(
            primary.get(key)
        )

        if value:
            return value

    return ""


def extract_team(
    entity: dict,
) -> str:

    primary = (
        entity.get("primaryRef")
        or {}
    )

    team = primary.get("team") or {}

    if isinstance(team, dict):
        return clean_text(
            team.get("name")
        )

    return clean_text(team)


def extract_position(
    entity: dict,
) -> str:

    return clean_text(
        (
            entity.get("primaryRef")
            or {}
        ).get("position")
    )


def classify_practice_signal(
    title: str,
    description: str,
) -> str:

    text = (
        f"{title} {description}"
        .lower()
    )

    if (
        "did not participate"
        in text
        or "did not practice"
        in text
    ):
        return "DNP"

    if (
        "limited participant"
        in text
        or "limited in practice"
        in text
    ):
        return "LIMITED"

    if (
        "full participant"
        in text
        or "full participation"
        in text
    ):
        return "FULL"

    return ""


def classify_availability_signal(
    title: str,
    description: str,
) -> str:
    title_text = str(title or "").lower()
    text = (
        f"{title or ''} {description or ''}"
        .lower()
    )

    # Remove explicit negations before evaluating hard-OUT
    # phrases. "Not ruled out" must never become OUT_SIGNAL.
    text = re.sub(
        r"\b(?:has\s+)?not\s+(?:been\s+)?ruled\s+out\b",
        "",
        text,
    )
    title_text = re.sub(
        r"\b(?:has\s+)?not\s+(?:been\s+)?ruled\s+out\b",
        "",
        title_text,
    )

    # FanDuel definitive headline:
    # "<player> Out For Week N ..."
    speculative = any(
        value in title_text
        for value in (
            "could be out",
            "may be out",
            "might be out",
        )
    )
    explicit_week_out = bool(
        re.search(
            r"\b(?:is\s+)?out\s+for\s+week\s+\d+\b",
            title_text,
        )
    ) and "worked out for week" not in title_text

    if (
        explicit_week_out
        and not speculative
    ):
        return "OUT_SIGNAL"

    if any(
        value in text
        for value in (
            "ruled out",
            "will not play",
            "won't play",
            "inactive",
            "will miss",
            "won't suit up",
        )
    ):
        return "OUT_SIGNAL"
    if "doubtful" in text:
        return "DOUBTFUL_SIGNAL"
    if "questionable" in text:
        return "QUESTIONABLE_SIGNAL"
    if (
        "expected to play"
        in text
        or "expected to suit up"
        in text
    ):
        return "EXPECTED_TO_PLAY"
    return ""


def signal_strength(
    practice: str,
    availability: str,
) -> str:

    if availability in {
        "OUT_SIGNAL",
        "DOUBTFUL_SIGNAL",
    }:
        return "HIGH"

    if (
        availability
        in {
            "QUESTIONABLE_SIGNAL",
            "EXPECTED_TO_PLAY",
        }
        or practice == "DNP"
    ):
        return "MEDIUM"

    return "LOW"


def explicit_weeks(text: str):

    return {
        int(x)
        for x in re.findall(
            r"\bWeek\s+(\d{1,2})\b",
            text,
            flags=re.I,
        )
    }


def extract_records(
    entities: list[dict],
) -> pd.DataFrame:

    rows = []

    for entity in entities:

        title = clean_text(
            entity.get("title")
        )

        description = clean_text(
            entity.get("description")
        )

        fact = flatten_blocks(
            entity.get("fact")
        )

        if not description:
            description = fact

        practice = (
            classify_practice_signal(
                title,
                description,
            )
        )

        availability = (
            classify_availability_signal(
                title,
                description,
            )
        )

        event_id = clean_text(
            entity.get("id")
        )

        categories = (
            entity.get(
                "_wfs_source_categories"
            )
            or []
        )

        rows.append({
            "source_event_id":
                event_id,

            "signal_key":
                hashlib.sha256(
                    f"{SOURCE}|{event_id}"
                    .encode()
                ).hexdigest(),

            "source_category":
                ",".join(
                    sorted(categories)
                ),

            "news_type":
                entity_news_type(
                    entity
                ),

            "player_name":
                extract_player_name(
                    entity
                ),

            "team":
                normalize_team(
                    extract_team(entity)
                ),

            "position":
                normalize_position(
                    extract_position(entity)
                ),

            "title":
                title,

            "description":
                description,

            "fact":
                fact,

            "practice_signal":
                practice,

            "availability_signal":
                availability,

            "signal_strength":
                signal_strength(
                    practice,
                    availability,
                ),

            "source_timestamp":
                clean_text(
                    entity.get(
                        "firstPublishedAt"
                    )
                ),

            "explicit_weeks":
                sorted(
                    explicit_weeks(
                        f"{title} "
                        f"{description} "
                        f"{fact}"
                    )
                ),
        })

    df = pd.DataFrame(rows)

    if not df.empty:

        df[
            "normalized_name"
        ] = (
            df["player_name"]
            .map(normalize_name)
        )

    return df


def table_columns(
    conn: sqlite3.Connection,
    table: str,
):

    return {
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    }


def latest_season_week(
    conn: sqlite3.Connection,
):

    row = conn.execute(
        """
        SELECT season, week
        FROM injuries
        WHERE season IS NOT NULL
          AND week IS NOT NULL
        ORDER BY season DESC, week DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "Unable to determine "
            "current season/week"
        )

    return int(row[0]), int(row[1])


def schedule_window(
    conn: sqlite3.Connection,
    season: int,
    week: int,
):

    rows = conn.execute(
        """
        SELECT
            game_id,
            game_date,
            gametime,
            away_team,
            home_team
        FROM games
        WHERE season = ?
          AND week = ?
        """,
        (
            season,
            week,
        ),
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "No current-week games"
        )

    kickoffs = []
    teams = set()

    for (
        game_id,
        date,
        time,
        away,
        home,
    ) in rows:

        naive = datetime.strptime(
            f"{clean_text(date)} "
            f"{clean_text(time)}",
            "%Y-%m-%d %H:%M",
        )

        kickoffs.append(
            naive.replace(
                tzinfo=ET
            ).astimezone(
                timezone.utc
            )
        )

        teams.update({
            normalize_team(away),
            normalize_team(home),
        })

    return (
        min(kickoffs)
        - WINDOW_BEFORE,

        max(kickoffs)
        + WINDOW_AFTER,

        {x for x in teams if x},
    )


def filter_current_week(
    records: pd.DataFrame,
    week: int,
    start_utc: datetime,
    end_utc: datetime,
    teams: set[str],
):

    keep = []
    reasons = []

    for row in records.itertuples(
        index=False
    ):

        timestamp = (
            parse_source_timestamp(
                row.source_timestamp
            )
        )

        weeks = set(
            row.explicit_weeks
            if isinstance(
                row.explicit_weeks,
                list,
            )
            else []
        )

        team = normalize_team(
            row.team
        )

        ok = True
        reason = "CURRENT_WINDOW"

        if timestamp is None:
            ok = False
            reason = "UNDATED"

        elif not (
            start_utc
            <= timestamp
            <= end_utc
        ):
            ok = False
            reason = (
                "OUTSIDE_SCHEDULE_WINDOW"
            )

        elif (
            weeks
            and weeks != {week}
        ):
            ok = False
            reason = (
                "EXPLICIT_WEEK_CONFLICT"
            )

        elif (
            team
            and team not in teams
        ):
            ok = False
            reason = (
                "TEAM_NOT_IN_CURRENT_SCHEDULE"
            )

        keep.append(ok)
        reasons.append(reason)

    output = records.copy()

    output[
        "current_filter_reason"
    ] = reasons

    mask = pd.Series(
        keep,
        index=output.index,
    )

    return (
        output[mask].copy(),
        output[~mask].copy(),
    )


def build_identity_index(
    conn: sqlite3.Connection,
    season: int,
    week: int,
):

    frames = []

    if {
        "gsis_id",
        "full_name",
        "latest_team",
        "position",
    }.issubset(
        table_columns(
            conn,
            "player_identity",
        )
    ):

        frames.append(
            pd.read_sql_query(
                """
                SELECT
                    CAST(gsis_id AS TEXT)
                        AS gsis_id,
                    full_name
                        AS player_name,
                    latest_team
                        AS team,
                    position
                FROM player_identity
                WHERE gsis_id IS NOT NULL
                  AND TRIM(
                      CAST(gsis_id AS TEXT)
                  ) <> ''
                """,
                conn,
            )
        )

    roster = pd.read_sql_query(
        """
        SELECT
            CAST(gsis_id AS TEXT)
                AS gsis_id,
            full_name
                AS player_name,
            team,
            position,
            updated_at
        FROM weekly_rosters
        WHERE season = ?
          AND week = ?
          AND gsis_id IS NOT NULL
          AND TRIM(
              CAST(gsis_id AS TEXT)
          ) <> ''
        """,
        conn,
        params=(
            season,
            week,
        ),
    )

    if not roster.empty:

        roster["_updated"] = (
            pd.to_datetime(
                roster["updated_at"],
                errors="coerce",
                utc=True,
            )
        )

        roster = (
            roster.sort_values(
                [
                    "gsis_id",
                    "_updated",
                    "team",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
                kind="stable",
            )
            .drop_duplicates(
                "gsis_id",
                keep="first",
            )
        )

        frames.append(
            roster[
                [
                    "gsis_id",
                    "player_name",
                    "team",
                    "position",
                ]
            ]
        )

    if not frames:
        raise RuntimeError(
            "No exact identity authority"
        )

    identity = pd.concat(
        frames,
        ignore_index=True,
    )

    identity["gsis_id"] = (
        identity["gsis_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    identity["normalized_name"] = (
        identity["player_name"]
        .map(normalize_name)
    )

    identity["team"] = (
        identity["team"]
        .map(normalize_team)
    )

    identity["position"] = (
        identity["position"]
        .map(normalize_position)
    )

    return identity.drop_duplicates(
        subset=[
            "gsis_id",
            "normalized_name",
            "team",
            "position",
        ]
    )


def resolve_identity(
    row: pd.Series,
    identity: pd.DataFrame,
):

    name = clean_text(
        row.get("normalized_name")
    )

    team = normalize_team(
        row.get("team")
    )

    position = normalize_position(
        row.get("position")
    )

    if not name:
        return (
            "",
            "QUARANTINE",
            "BLANK_PLAYER_NAME",
        )

    if not team and not position:
        return (
            "",
            "QUARANTINE",
            "DISPLAY_NAME_ONLY_IDENTITY_PROHIBITED",
        )

    candidates = identity[
        identity[
            "normalized_name"
        ].eq(name)
    ].copy()

    if team:
        candidates = candidates[
            candidates[
                "team"
            ].eq(team)
        ]

    if position:
        candidates = candidates[
            candidates[
                "position"
            ].eq(position)
        ]

    ids = sorted(
        set(
            candidates[
                "gsis_id"
            ].tolist()
        )
    )

    if len(ids) == 1:
        return (
            ids[0],
            "MATCHED",
            "EXACT_NAME_CONTEXT_GSIS",
        )

    if not ids:
        return (
            "",
            "QUARANTINE",
            "IDENTITY_CONTEXT_MISMATCH",
        )

    return (
        "",
        "QUARANTINE",
        "AMBIGUOUS_EXACT_IDENTITY",
    )


def attach_identity(
    records: pd.DataFrame,
    identity: pd.DataFrame,
):

    output = records.copy()

    resolved = output.apply(
        lambda row:
            resolve_identity(
                row,
                identity,
            ),
        axis=1,
    )

    output["gsis_id"] = [
        x[0]
        for x in resolved
    ]

    output[
        "resolution_status"
    ] = [
        x[1]
        for x in resolved
    ]

    output[
        "resolution_detail"
    ] = [
        x[2]
        for x in resolved
    ]

    return output


def migrate_schema(
    conn: sqlite3.Connection,
):

    cols = table_columns(
        conn,
        "injury_news_signals",
    )

    if "source_event_id" not in cols:
        conn.execute(
            """
            ALTER TABLE injury_news_signals
            ADD COLUMN source_event_id TEXT
            """
        )

    if "signal_key" not in cols:
        conn.execute(
            """
            ALTER TABLE injury_news_signals
            ADD COLUMN signal_key TEXT
            """
        )

    if "source_category" not in cols:
        conn.execute(
            """
            ALTER TABLE injury_news_signals
            ADD COLUMN source_category TEXT
            """
        )

    if "news_type" not in cols:
        conn.execute(
            """
            ALTER TABLE injury_news_signals
            ADD COLUMN news_type TEXT
            """
        )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        ux_injury_news_signals_signal_key
        ON injury_news_signals(signal_key)
        WHERE signal_key IS NOT NULL
        """
    )


def write_signals(
    conn: sqlite3.Connection,
    matched: pd.DataFrame,
    season: int,
    week: int,
):

    inserted = 0
    existing = 0
    now = utc_now()

    for row in matched.itertuples(
        index=False
    ):

        before = conn.total_changes

        conn.execute(
            """
            INSERT OR IGNORE INTO
            injury_news_signals (
                season,
                week,
                gsis_id,
                player_name,
                team,
                source,
                source_timestamp,
                headline,
                raw_text,
                practice_signal,
                availability_signal,
                signal_strength,
                ai_interpretation,
                ai_confidence,
                created_at,
                source_event_id,
                signal_key,
                source_category,
                news_type
            )
            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                season,
                week,
                clean_text(
                    row.gsis_id
                ),
                clean_text(
                    row.player_name
                ),
                normalize_team(
                    row.team
                ),
                SOURCE,
                clean_text(
                    row.source_timestamp
                ),
                clean_text(
                    row.title
                ),
                clean_text(
                    row.description
                ),
                clean_text(
                    row.practice_signal
                ),
                clean_text(
                    row.availability_signal
                ),
                clean_text(
                    row.signal_strength
                ),
                "",
                None,
                now,
                clean_text(
                    row.source_event_id
                ),
                clean_text(
                    row.signal_key
                ),
                clean_text(
                    row.source_category
                ),
                clean_text(
                    row.news_type
                ),
            ),
        )

        if (
            conn.total_changes
            > before
        ):
            inserted += 1

        else:
            existing += 1

    return inserted, existing


def run(write_db: bool):

    print("=" * 72)
    print(
        "WFS FANDUEL PLAYER NEWS "
        "MULTICATEGORY INGEST V3"
    )
    print("=" * 72)

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Missing database: {DB_PATH}"
        )

    if write_db:
        CSV_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        conn = sqlite3.connect(
            DB_PATH
        )

    else:
        conn = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro",
            uri=True,
        )

    try:

        schedule_ctx = resolve_schedule_week_context(
            db_path=DB_PATH,
        )
        season = int(schedule_ctx.season)
        week = int(schedule_ctx.planning_week)

        (
            start_utc,
            end_utc,
            teams,
        ) = schedule_window(
            conn,
            season,
            week,
        )

        print(
            f"WFS_TARGET={season}_WEEK_{week}"
        )

        print(
            "WINDOW_START_UTC="
            + start_utc.isoformat()
        )

        print(
            "WINDOW_END_UTC="
            + end_utc.isoformat()
        )

        (
            entities,
            audits,
        ) = fetch_all_families(
            start_utc
        )

        for audit in audits:

            print(
                "FAMILY_AUDIT|"
                f"FAMILY={audit['family']}|"
                f"REQUIRED={audit['required']}|"
                f"TYPED={audit['typed']}|"
                f"OBSERVED_NEWS_TYPE="
                f"{audit['observed_news_type']}|"
                f"PAGES={audit['pages']}|"
                f"RAW={audit['raw_rows']}|"
                f"UNIQUE={audit['unique_events']}|"
                f"DUPES="
                f"{audit['boundary_duplicates']}|"
                f"TERMINAL_HAS_NEXT_PAGE="
                f"{audit['terminal_has_next_page']}|"
                f"DEGRADED={audit['degraded']}|"
                f"REASON={audit['degraded_reason']}"
            )

        records = extract_records(
            entities
        )

        if records.empty:
            raise RuntimeError(
                "Zero multicategory records"
            )

        current, rejected = (
            filter_current_week(
                records,
                week,
                start_utc,
                end_utc,
                teams,
            )
        )

        print(
            f"ALL_SOURCE_EVENTS={len(records)}"
        )

        print(
            f"CURRENT_WINDOW_EVENTS={len(current)}"
        )

        print(
            f"REJECTED_EVENTS={len(rejected)}"
        )

        if current.empty:
            print(
                "TARGET_WEEK_EVIDENCE_ABSENT"
            )
            print(
                "WRITE_ROWS=0"
            )
            print(
                "INGEST_RESULT=SUCCESS_NOOP"
            )
            return

        identity = build_identity_index(
            conn,
            season,
            week,
        )

        resolved = attach_identity(
            current,
            identity,
        )

        matched = resolved[
            resolved[
                "resolution_status"
            ].eq("MATCHED")
        ].copy()

        quarantine = resolved[
            ~resolved[
                "resolution_status"
            ].eq("MATCHED")
        ].copy()

        print(
            f"MATCHED={len(matched)}"
        )

        print(
            f"QUARANTINED={len(quarantine)}"
        )

        if matched.empty:
            raise RuntimeError(
                "Zero matched exact identities"
            )

        if write_db:

            migrate_schema(conn)

            (
                inserted,
                existing,
            ) = write_signals(
                conn,
                matched,
                season,
                week,
            )

            conn.commit()

            resolved.to_csv(
                INGEST_CSV,
                index=False,
            )

            quarantine.to_csv(
                QUARANTINE_CSV,
                index=False,
            )

            print(
                f"INSERTED={inserted}"
            )

            print(
                f"ALREADY_PRESENT={existing}"
            )

        else:

            print(
                "DATABASE_WRITE_EXECUTED=FALSE"
            )

            print(
                "CSV_WRITE_EXECUTED=FALSE"
            )

        print(
            "FUZZY_IDENTITY=FALSE"
        )

        print(
            "DISPLAY_NAME_ONLY_IDENTITY=FALSE"
        )

        print(
            "EXPLICIT_WEEK_REQUIRED=FALSE"
        )

        print(
            "EXPLICIT_WEEK_CONFLICT_REJECTED=TRUE"
        )

        print(
            "GENERAL_PLAYER_NEWS_SUPPLEMENTAL=TRUE"
        )

        print(
            "REQUIRED_TYPED_FEEDS="
            "INJURIES,GAME_UPDATES,TRANSACTIONS"
        )

        print(
            "FANDUEL_MULTICATEGORY_INGEST_V3_STATUS=PASS"
        )

    finally:
        conn.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        write_db=(
            not args.dry_run
        )
    )


if __name__ == "__main__":
    main()
