#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nfl.db"

BASE_URL = "https://www.nflweather.com"
SOURCE = "NFLWeather"
VERSION = "WFS_WEATHER_INTELLIGENCE_V1"

USER_AGENT = "Mozilla/5.0 WFS-Weather-Intelligence/1.0"

PERIODS = ("Kickoff", "Q2", "Q3", "Q4")

# NFLWeather URL slug -> canonical WFS schedule abbreviation.
# Explicit aliases only. No fuzzy team identity matching.
NFLWEATHER_TEAM_MAP = {
    "cardinals": "ARI",
    "falcons": "ATL",
    "ravens": "BAL",
    "bills": "BUF",
    "panthers": "CAR",
    "bears": "CHI",
    "bengals": "CIN",
    "browns": "CLE",
    "cowboys": "DAL",
    "broncos": "DEN",
    "lions": "DET",
    "packers": "GB",
    "texans": "HOU",
    "colts": "IND",
    "jaguars": "JAX",
    "chiefs": "KC",
    "raiders": "LV",
    "chargers": "LAC",
    "rams": "LA",
    "dolphins": "MIA",
    "vikings": "MIN",
    "patriots": "NE",
    "saints": "NO",
    "giants": "NYG",
    "jets": "NYJ",
    "eagles": "PHI",
    "steelers": "PIT",
    "49ers": "SF",
    "seahawks": "SEA",
    "buccaneers": "TB",
    "titans": "TEN",
    "commanders": "WAS",
    "washington": "WAS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def fetch(url: str) -> str:
    r = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.text


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_game_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            stadium_name TEXT,
            source_stadium_type TEXT,
            source_surface TEXT,
            source_location TEXT,
            source_forecast_text TEXT,
            parser_version TEXT NOT NULL,
            UNIQUE(game_id, source, captured_at)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_period_snapshots (
            snapshot_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            period TEXT NOT NULL,
            condition TEXT,
            temperature_f REAL,
            low_f REAL,
            high_f REAL,
            feels_like_f REAL,
            wind_mph REAL,
            wind_direction TEXT,
            gust_mph REAL,
            precipitation_probability_pct REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL,
            dew_point_f REAL,
            visibility_miles REAL,
            PRIMARY KEY(snapshot_id, period),
            FOREIGN KEY(snapshot_id)
                REFERENCES weather_game_snapshots(snapshot_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_weather_game_snapshots_game_capture
        ON weather_game_snapshots(game_id, captured_at)
        """
    )


def number(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return None

    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def extract_card_blocks(raw: str) -> list[str]:
    starts = list(
        re.finditer(
            r'<div[^>]*class=["\'][^"\']*'
            r'(?:^|\s)weather-report(?:\s|["\'])'
            r'[^>]*>',
            raw,
            flags=re.I,
        )
    )

    if len(starts) != 4:
        raise RuntimeError(
            f"expected 4 weather cards; found {len(starts)}"
        )

    blocks = []

    for i, match in enumerate(starts):
        start = match.start()

        if i + 1 < len(starts):
            end = starts[i + 1].start()
        else:
            stadium = re.search(
                r'class=["\'][^"\']*stadium-container',
                raw[start:],
                flags=re.I,
            )

            end = (
                start + stadium.start()
                if stadium
                else len(raw)
            )

        blocks.append(raw[start:end])

    return blocks


def parse_period(block: str) -> dict:
    text = clean_text(block)

    title_match = re.search(
        r'weather-title[^>]*>\s*([^<]+)',
        block,
        flags=re.I,
    )

    if not title_match:
        raise RuntimeError("weather card has no title")

    period = clean_text(title_match.group(1))

    if period not in PERIODS:
        raise RuntimeError(
            f"unexpected weather period: {period}"
        )

    temp_match = re.search(
        r'weather-temperature[^>]*>'
        r'.*?(-?\d+(?:\.\d+)?)\s*°F',
        block,
        flags=re.I | re.S,
    )

    temperature = (
        float(temp_match.group(1))
        if temp_match
        else None
    )

    condition_match = re.search(
        r'weather-image[^>]*>.*?<p[^>]*>'
        r'.*?<span[^>]*>([^<]+)</span>',
        block,
        flags=re.I | re.S,
    )

    condition = (
        clean_text(condition_match.group(1))
        if condition_match
        else None
    )

    range_match = re.search(
        r'(-?\d+(?:\.\d+)?)\s*°F\s*/\s*'
        r'(-?\d+(?:\.\d+)?)\s*°F',
        text,
        flags=re.I,
    )

    low_f = (
        float(range_match.group(1))
        if range_match
        else None
    )

    high_f = (
        float(range_match.group(2))
        if range_match
        else None
    )

    wind_match = re.search(
        r'(\d+(?:\.\d+)?)\s*mph\s+'
        r'(?:[a-z_]+\s+)?'
        r'(N|NE|E|SE|S|SW|W|NW)\b',
        text,
        flags=re.I,
    )

    wind_mph = (
        float(wind_match.group(1))
        if wind_match
        else None
    )

    wind_direction = (
        wind_match.group(2).upper()
        if wind_match
        else None
    )

    return {
        "period": period,
        "condition": condition,
        "temperature_f": temperature,
        "low_f": low_f,
        "high_f": high_f,
        "feels_like_f": number(
            r"Feels Like:\s*(-?\d+(?:\.\d+)?)\s*°F",
            text,
        ),
        "wind_mph": wind_mph,
        "wind_direction": wind_direction,
        "gust_mph": number(
            r"Gusts:\s*(\d+(?:\.\d+)?)\s*mph",
            text,
        ),
        "precipitation_probability_pct": number(
            r"Prec\.\s*Prob\.:\s*(\d+(?:\.\d+)?)\s*%",
            text,
        ),
        "cloud_cover_pct": number(
            r"Cloud Cover:\s*(\d+(?:\.\d+)?)\s*%",
            text,
        ),
        "humidity_pct": number(
            r"Humidity:\s*(\d+(?:\.\d+)?)\s*%",
            text,
        ),
        "dew_point_f": number(
            r"Dew Point:\s*(-?\d+(?:\.\d+)?)\s*°F",
            text,
        ),
        "visibility_miles": number(
            r"Visibility:\s*(\d+(?:\.\d+)?)\s*m\b",
            text,
        ),
    }


def parse_stadium(raw: str) -> dict:
    text = clean_text(raw)

    surface = None
    stadium_type = None
    location = None

    stadium_container = re.search(
        r'<a[^>]*class=["\'][^"\']*stadium-container[^"\']*["\'][^>]*>'
        r'(.*?)'
        r'</a>',
        raw,
        flags=re.I | re.S,
    )

    stadium_name = None

    if stadium_container:
        title = re.search(
            r'<div[^>]*class=["\'][^"\']*\btitle\b[^"\']*["\'][^>]*>'
            r'\s*(.*?)\s*</div>',
            stadium_container.group(1),
            flags=re.I | re.S,
        )

        if title:
            stadium_title = clean_text(title.group(1))
            stadium_name = stadium_title.split(",", 1)[0].strip()

    m = re.search(
        r"([^|]{1,120}?),\s*[^,]+,\s*[A-Z]{2}\s+"
        r"(?:grass\s+)?Surface:\s*(.*?)\s+house\s+Location:",
        text,
        flags=re.I,
    )

    if m:
        surface = clean_text(m.group(2))

    m = re.search(
        r"Location:\s*(.*?)\s+mail\s+Zip:",
        text,
        flags=re.I,
    )

    if m:
        location = clean_text(m.group(1))

    m = re.search(
        r"Type:\s*(.*?)\s+explore\s+Orientation:",
        text,
        flags=re.I,
    )

    if m:
        stadium_type = clean_text(m.group(1))

    return {
        "stadium_name": stadium_name,
        "source_stadium_type": stadium_type,
        "source_surface": surface,
        "source_location": location,
    }


def discover_week_urls(
    season: int,
    week: int,
) -> list[str]:
    raw = fetch(BASE_URL + "/")

    paths = re.findall(
        rf'href=["\']'
        rf'(/games/{season}/week-{week}/[^"\']+)'
        rf'["\']',
        raw,
        flags=re.I,
    )

    return sorted(
        set(urljoin(BASE_URL, p) for p in paths)
    )


def game_rows(
    conn: sqlite3.Connection,
    season: int,
    week: int,
) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row

    return conn.execute(
        """
        SELECT
            game_id,
            season,
            week,
            away_team,
            home_team
        FROM games
        WHERE season = ?
          AND game_type = 'REG'
          AND week = ?
        ORDER BY game_id
        """,
        (season, week),
    ).fetchall()


def resolve_weather_game(
    conn: sqlite3.Connection,
    url: str,
    season: int,
    week: int,
) -> sqlite3.Row:
    slug = url.rstrip("/").rsplit("/", 1)[-1].lower()

    if "-at-" not in slug:
        raise RuntimeError(
            f"invalid NFLWeather game slug: {slug}"
        )

    away_slug, home_slug = slug.split("-at-", 1)

    away_team = NFLWEATHER_TEAM_MAP.get(away_slug)
    home_team = NFLWEATHER_TEAM_MAP.get(home_slug)

    if not away_team or not home_team:
        raise RuntimeError(
            "unmapped NFLWeather team slug: "
            f"{away_slug}@{home_slug}"
        )

    rows = conn.execute(
        """
        SELECT
            game_id,
            season,
            week,
            away_team,
            home_team
        FROM games
        WHERE season = ?
          AND game_type = 'REG'
          AND week = ?
          AND away_team = ?
          AND home_team = ?
        """,
        (
            season,
            week,
            away_team,
            home_team,
        ),
    ).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            "NFLWeather exact schedule identity failure: "
            f"{away_team}@{home_team} "
            f"matches={len(rows)}"
        )

    return rows[0]


def page_teams(raw: str) -> tuple[str, str]:
    text = clean_text(raw)

    m = re.search(
        r"\d{4}\s*-\s*Week\s+\d+\s+"
        r"(.+?)\s+At\s+(.+?)\s+"
        r"\d{2}/\d{2}/\d{2}",
        text,
        flags=re.I,
    )

    if not m:
        raise RuntimeError(
            "unable to parse NFLWeather game teams"
        )

    return (
        clean_text(m.group(1)),
        clean_text(m.group(2)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    urls = discover_week_urls(
        args.season,
        args.week,
    )

    with sqlite3.connect(DB) as conn:
        rows = game_rows(
            conn,
            args.season,
            args.week,
        )

        if len(urls) != len(rows):
            raise RuntimeError(
                "NFLWeather/game schedule count mismatch: "
                f"weather={len(urls)} schedule={len(rows)}"
            )

        # Deterministic pairing is validated by page order only after
        # every page is parsed; no fuzzy identity matching is allowed.
        pages = []

        for url in urls:
            raw = fetch(url)
            away_name, home_name = page_teams(raw)

            game = resolve_weather_game(
                conn,
                url,
                args.season,
                args.week,
            )

            pages.append({
                "url": url,
                "raw": raw,
                "away_name": away_name,
                "home_name": home_name,
                "game_id": str(game["game_id"]),
                "away_team": str(game["away_team"]),
                "home_team": str(game["home_team"]),
            })

        # NFLWeather identity has already been resolved through the
        # explicit source-slug map and exact authoritative schedule match.
        print("=" * 72)
        print(VERSION)
        print("=" * 72)
        print(
            f"season={args.season} week={args.week} "
            f"schedule_games={len(rows)} "
            f"weather_pages={len(pages)}"
        )

        parsed_pages = []

        for page in pages:
            cards = [
                parse_period(block)
                for block in extract_card_blocks(
                    page["raw"]
                )
            ]

            if [x["period"] for x in cards] != list(PERIODS):
                raise RuntimeError(
                    "weather period contract failure: "
                    f'{page["game_id"]}'
                )

            stadium = parse_stadium(
                page["raw"]
            )

            parsed_pages.append({
                **page,
                "cards": cards,
                "stadium": stadium,
            })

            print(
                page["url"].rsplit("/", 1)[-1],
                f'{page["away_team"]}@{page["home_team"]}',
                "->",
                page["game_id"],
                stadium["source_stadium_type"],
                [x["period"] for x in cards],
            )

        game_ids = [
            page["game_id"]
            for page in parsed_pages
        ]

        if len(set(game_ids)) != len(game_ids):
            raise RuntimeError(
                "duplicate WFS game_id resolved from NFLWeather pages"
            )

        print(
            "EXACT_GAME_IDENTITIES="
            f"{len(game_ids)}/{len(rows)}"
        )

        if args.dry_run:
            print(
                "DRY_RUN=PASS "
                "(no database writes)"
            )
            return 0

        captured_at = utc_now()

        try:
            ensure_schema(conn)

            game_snapshot_count = 0
            period_snapshot_count = 0

            for page in parsed_pages:
                stadium = page["stadium"]

                cur = conn.execute(
                    """
                    INSERT INTO weather_game_snapshots (
                        game_id,
                        season,
                        week,
                        source,
                        source_url,
                        captured_at,
                        stadium_name,
                        source_stadium_type,
                        source_surface,
                        source_location,
                        source_forecast_text,
                        parser_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page["game_id"],
                        args.season,
                        args.week,
                        SOURCE,
                        page["url"],
                        captured_at,
                        stadium["stadium_name"],
                        stadium["source_stadium_type"],
                        stadium["source_surface"],
                        stadium["source_location"],
                        None,
                        VERSION,
                    ),
                )

                snapshot_id = int(cur.lastrowid)
                game_snapshot_count += 1

                for card in page["cards"]:
                    conn.execute(
                        """
                        INSERT INTO weather_period_snapshots (
                            snapshot_id,
                            game_id,
                            period,
                            condition,
                            temperature_f,
                            low_f,
                            high_f,
                            feels_like_f,
                            wind_mph,
                            wind_direction,
                            gust_mph,
                            precipitation_probability_pct,
                            cloud_cover_pct,
                            humidity_pct,
                            dew_point_f,
                            visibility_miles
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            snapshot_id,
                            page["game_id"],
                            card["period"],
                            card["condition"],
                            card["temperature_f"],
                            card["low_f"],
                            card["high_f"],
                            card["feels_like_f"],
                            card["wind_mph"],
                            card["wind_direction"],
                            card["gust_mph"],
                            card["precipitation_probability_pct"],
                            card["cloud_cover_pct"],
                            card["humidity_pct"],
                            card["dew_point_f"],
                            card["visibility_miles"],
                        ),
                    )

                    period_snapshot_count += 1

            if game_snapshot_count != len(rows):
                raise RuntimeError(
                    "game snapshot count failure: "
                    f"{game_snapshot_count}/{len(rows)}"
                )

            expected_periods = len(rows) * len(PERIODS)

            if period_snapshot_count != expected_periods:
                raise RuntimeError(
                    "period snapshot count failure: "
                    f"{period_snapshot_count}/{expected_periods}"
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        print(f"CAPTURED_AT={captured_at}")
        print(
            f"GAME_SNAPSHOTS_WRITTEN={game_snapshot_count}"
        )
        print(
            f"PERIOD_SNAPSHOTS_WRITTEN={period_snapshot_count}"
        )
        print("WEATHER_SNAPSHOT_WRITE=PASS")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
