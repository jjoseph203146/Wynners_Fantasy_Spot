#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path("/home/mwynn/nfl_data_engine")
LIVE_DIR = PROJECT / "data" / "fanduel" / "slates"
ARCHIVE_ROOT = LIVE_DIR / "archive"

SCHEDULE_CANDIDATES = [
    PROJECT / "data" / "csv" / "nfl_schedule.csv",
]

TEAM_ALIASES = {
    "JAC": "JAX",
    "JAX": "JAX",
    "WSH": "WAS",
    "WAS": "WAS",
    "LAR": "LA",
    "LA": "LA",
    "LV": "LV",
    "OAK": "LV",
    "SD": "LAC",
    "LAC": "LAC",
    "STL": "LA",
}


def norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def norm_team(value: object) -> str:
    s = str(value or "").strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    return TEAM_ALIASES.get(s, s)


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "slate"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm_col(c): c for c in df.columns}

    for candidate in candidates:
        key = norm_col(candidate)
        if key in lookup:
            return lookup[key]

    return None


def parse_matchup(raw: object) -> tuple[str, str] | None:
    text = str(raw or "").strip().upper()

    if not text:
        return None

    match = re.search(
        r"\b([A-Z]{2,3})\s*(?:@|VS\.?|V\.?)\s*([A-Z]{2,3})\b",
        text,
    )

    if not match:
        return None

    away = norm_team(match.group(1))
    home = norm_team(match.group(2))

    if not away or not home:
        return None

    return away, home


def load_schedule() -> tuple[pd.DataFrame, Path]:
    for path in SCHEDULE_CANDIDATES:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        season_col = find_column(df, ["season"])
        week_col = find_column(df, ["week"])
        away_col = find_column(
            df,
            ["away_team", "away", "awayteam", "visitor_team"],
        )
        home_col = find_column(
            df,
            ["home_team", "home", "hometeam"],
        )

        if not all([season_col, week_col, away_col, home_col]):
            print(f"Schedule candidate missing required columns: {path}")
            print("Columns:", list(df.columns))
            continue

        out = pd.DataFrame(
            {
                "season": pd.to_numeric(df[season_col], errors="coerce"),
                "week": pd.to_numeric(df[week_col], errors="coerce"),
                "away_team": df[away_col].map(norm_team),
                "home_team": df[home_col].map(norm_team),
            }
        )

        game_type_col = find_column(
            df,
            ["game_type", "gametype", "season_type"],
        )

        if game_type_col:
            out["game_type"] = (
                df[game_type_col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

            regular = out["game_type"].isin(
                ["REG", "REGULAR", "REGULAR SEASON", ""]
            )

            if regular.any():
                out = out.loc[regular].copy()

        out = out.dropna(subset=["season", "week"])
        out["season"] = out["season"].astype(int)
        out["week"] = out["week"].astype(int)

        return out, path

    raise RuntimeError(
        "Could not find a usable NFL schedule CSV. "
        "Expected data/csv/nfl_schedule.csv."
    )


def load_slate_games(path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)

    game_col = find_column(
        df,
        [
            "gameInfo",
            "game_info",
            "game",
            "gameinformation",
            "matchup",
        ],
    )

    if not game_col:
        raise RuntimeError(
            f"{path.name}: no gameInfo/game/matchup column found. "
            f"Columns={list(df.columns)}"
        )

    games: set[tuple[str, str]] = set()

    for value in df[game_col].dropna().astype(str):
        parsed = parse_matchup(value)

        if parsed:
            games.add(parsed)

    if not games:
        raise RuntimeError(
            f"{path.name}: no parseable NFL matchups found."
        )

    return sorted(games)


def classify_slate(
    slate_games: list[tuple[str, str]],
    schedule: pd.DataFrame,
) -> dict:
    matches = []

    for away, home in slate_games:
        exact = schedule.loc[
            schedule["away_team"].eq(away)
            & schedule["home_team"].eq(home)
        ].copy()

        if exact.empty:
            exact = schedule.loc[
                schedule["away_team"].eq(home)
                & schedule["home_team"].eq(away)
            ].copy()

        if exact.empty:
            matches.append(
                {
                    "game": f"{away}@{home}",
                    "season": None,
                    "week": None,
                }
            )
            continue

        seasons = sorted(exact["season"].dropna().unique())
        latest_season = seasons[-1]

        exact = exact.loc[
            exact["season"].eq(latest_season)
        ].copy()

        weeks = sorted(exact["week"].dropna().unique())

        if not weeks:
            matches.append(
                {
                    "game": f"{away}@{home}",
                    "season": int(latest_season),
                    "week": None,
                }
            )
            continue

        matches.append(
            {
                "game": f"{away}@{home}",
                "season": int(latest_season),
                "week": int(weeks[0]),
            }
        )

    resolved = [
        row
        for row in matches
        if row["season"] is not None and row["week"] is not None
    ]

    unresolved = [
        row["game"]
        for row in matches
        if row["season"] is None or row["week"] is None
    ]

    identities = sorted(
        set(
            (row["season"], row["week"])
            for row in resolved
        )
    )

    return {
        "matches": matches,
        "resolved": resolved,
        "unresolved": unresolved,
        "identities": identities,
    }


def archive_file(
    src: Path,
    season: int,
    week: int,
    apply: bool,
) -> Path:
    slate_slug = slugify(src.stem)

    target_dir = (
        ARCHIVE_ROOT
        / str(season)
        / f"week_{week:02d}"
    )

    target = target_dir / f"{slate_slug}.csv"

    source_hash = file_sha256(src)

    if target.exists():
        target_hash = file_sha256(target)

        if target_hash == source_hash:
            print(f"  EXISTS IDENTICAL: {target}")
            return target

        short_hash = source_hash[:12]
        target = target_dir / f"{slate_slug}__{short_hash}.csv"

        if target.exists():
            if file_sha256(target) == source_hash:
                print(f"  EXISTS IDENTICAL VERSION: {target}")
                return target

    if apply:
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, target)

        if file_sha256(target) != source_hash:
            raise RuntimeError(
                f"Hash verification failed after copy: {target}"
            )

        print(f"  ARCHIVED: {target}")
    else:
        print(f"  WOULD ARCHIVE: {target}")

    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve FanDuel Classic slate CSVs by NFL season/week "
            "without modifying live files."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create archive copies.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("FANDUEL WEEK-AWARE SLATE ARCHIVER")
    print("=" * 100)
    print("MODE:", "APPLY" if args.apply else "DRY RUN")

    schedule, schedule_path = load_schedule()

    print("Schedule authority:", schedule_path)
    print("Schedule seasons:", sorted(schedule["season"].unique()))

    files = sorted(
        p
        for p in LIVE_DIR.glob("*.csv")
        if p.is_file()
    )

    if not files:
        raise RuntimeError(
            f"No CSV files found in {LIVE_DIR}"
        )

    print("Live slate files:", len(files))

    failures = 0

    for path in files:
        print()
        print("-" * 100)
        print("FILE:", path.name)
        print("SHA256:", file_sha256(path))

        try:
            games = load_slate_games(path)
        except Exception as exc:
            failures += 1
            print("STATUS: BLOCKED")
            print("REASON:", exc)
            continue

        print(
            "GAMES:",
            ", ".join(
                f"{away}@{home}"
                for away, home in games
            ),
        )

        result = classify_slate(
            games,
            schedule,
        )

        print("IDENTITIES:", result["identities"])

        if result["unresolved"]:
            failures += 1
            print("STATUS: BLOCKED_UNRESOLVED_GAMES")
            print(
                "UNRESOLVED:",
                ", ".join(result["unresolved"]),
            )
            continue

        if len(result["identities"]) != 1:
            failures += 1
            print("STATUS: BLOCKED_MIXED_WEEK_FILE")

            for row in result["matches"]:
                print(
                    " ",
                    row["game"],
                    "->",
                    row["season"],
                    row["week"],
                )

            continue

        season, week = result["identities"][0]

        print(
            f"CLASSIFIED: season={season} week={week}"
        )

        archive_file(
            path,
            season,
            week,
            args.apply,
        )

        print("STATUS: PASS")

    print()
    print("=" * 100)

    if failures:
        print(
            f"COMPLETE WITH {failures} BLOCKED FILE(S)"
        )
        print("No blocked file was archived.")
        return 2

    print("ALL LIVE SLATE FILES CLASSIFIED")
    print("Live source files were NOT modified or deleted.")

    if not args.apply:
        print(
            "Dry run only. Run again with --apply "
            "after reviewing classifications."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
