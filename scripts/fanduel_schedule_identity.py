#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT = Path("/home/mwynn/nfl_data_engine")

SCHEDULE_CANDIDATES = [
    PROJECT / "data" / "csv" / "nfl_schedule.csv",
]


TEAM_ALIASES = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LV": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "LAC": "LAC",
    "SD": "LAC",
    "SDG": "LAC",
    "LA": "LA",
    "LAR": "LA",
    "STL": "LA",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SF": "SF",
    "SFO": "SF",
    "SEA": "SEA",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}


def norm_col(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def norm_team(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z]", "", text)

    return TEAM_ALIASES.get(text, text)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        norm_col(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = norm_col(candidate)

        if key in lookup:
            return lookup[key]

    return None


def parse_matchup(
    raw: object,
) -> tuple[str, str] | None:
    text = str(raw or "").strip().upper()

    if not text:
        return None

    match = re.search(
        r"\b([A-Z]{2,3})\s*"
        r"(?:@|VS\.?|V\.?)\s*"
        r"([A-Z]{2,3})\b",
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

        raw = pd.read_csv(path)

        season_col = find_column(
            raw,
            ["season"],
        )

        week_col = find_column(
            raw,
            ["week"],
        )

        game_id_col = find_column(
            raw,
            [
                "game_id",
                "gameid",
            ],
        )

        away_col = find_column(
            raw,
            [
                "away_team",
                "away",
                "awayteam",
                "visitor_team",
            ],
        )

        home_col = find_column(
            raw,
            [
                "home_team",
                "home",
                "hometeam",
            ],
        )

        date_col = find_column(
            raw,
            [
                "game_date",
                "gameday",
                "date",
            ],
        )

        time_col = find_column(
            raw,
            [
                "gametime",
                "game_time",
                "time",
            ],
        )

        game_type_col = find_column(
            raw,
            [
                "game_type",
                "gametype",
                "season_type",
            ],
        )

        required = {
            "season": season_col,
            "week": week_col,
            "game_id": game_id_col,
            "away_team": away_col,
            "home_team": home_col,
        }

        missing = [
            logical_name
            for logical_name, column
            in required.items()
            if column is None
        ]

        if missing:
            print(
                f"Schedule candidate missing "
                f"required columns: {path}"
            )
            print(
                "Missing:",
                ", ".join(missing),
            )
            print(
                "Columns:",
                list(raw.columns),
            )
            continue

        out = pd.DataFrame(
            {
                "season": pd.to_numeric(
                    raw[season_col],
                    errors="coerce",
                ),
                "week": pd.to_numeric(
                    raw[week_col],
                    errors="coerce",
                ),
                "game_id": (
                    raw[game_id_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                ),
                "away_team": (
                    raw[away_col]
                    .map(norm_team)
                ),
                "home_team": (
                    raw[home_col]
                    .map(norm_team)
                ),
            }
        )

        if date_col is not None:
            out["game_date"] = pd.to_datetime(
                raw[date_col],
                errors="coerce",
            )
        else:
            out["game_date"] = pd.NaT

        if time_col is not None:
            out["game_time"] = (
                raw[time_col]
                .fillna("")
                .astype(str)
                .str.strip()
            )
        else:
            out["game_time"] = ""

        if game_type_col is not None:
            out["game_type"] = (
                raw[game_type_col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )
        else:
            out["game_type"] = ""

        out = out.dropna(
            subset=[
                "season",
                "week",
            ]
        ).copy()

        out["season"] = (
            out["season"]
            .astype(int)
        )

        out["week"] = (
            out["week"]
            .astype(int)
        )

        out = out[
            out["game_id"].ne("")
            & out["away_team"].ne("")
            & out["home_team"].ne("")
        ].copy()

        regular_mask = (
            out["game_type"]
            .isin(
                [
                    "",
                    "REG",
                    "REGULAR",
                    "REGULAR SEASON",
                ]
            )
        )

        if regular_mask.any():
            out = out.loc[
                regular_mask
            ].copy()

        duplicate_identity = out.duplicated(
            [
                "season",
                "week",
                "game_id",
            ],
            keep=False,
        )

        if duplicate_identity.any():
            duplicates = out.loc[
                duplicate_identity,
                [
                    "season",
                    "week",
                    "game_id",
                    "away_team",
                    "home_team",
                ],
            ]

            raise RuntimeError(
                "Schedule contains duplicate "
                "season/week/game_id rows:\n"
                + duplicates.to_string(
                    index=False
                )
            )

        out = out.sort_values(
            [
                "season",
                "week",
                "game_date",
                "game_time",
                "game_id",
            ]
        ).reset_index(
            drop=True
        )

        return out, path

    raise RuntimeError(
        "Could not find a usable NFL schedule CSV. "
        "Expected data/csv/nfl_schedule.csv."
    )


def resolve_matchup(
    schedule: pd.DataFrame,
    away_team: object,
    home_team: object,
) -> dict:
    away = norm_team(away_team)
    home = norm_team(home_team)

    result = {
        "away_team": away or None,
        "home_team": home or None,
        "season": None,
        "week": None,
        "game_id": None,
        "game_date": None,
        "game_time": None,
        "schedule_status": "UNRESOLVED",
        "schedule_match_count": 0,
    }

    if not away or not home:
        result["schedule_status"] = (
            "INVALID_MATCHUP"
        )
        return result

    exact = schedule[
        schedule["away_team"].eq(away)
        & schedule["home_team"].eq(home)
    ].copy()

    if exact.empty:
        result["schedule_status"] = (
            "UNRESOLVED"
        )
        return result

    latest_season = int(
        exact["season"].max()
    )

    exact = exact[
        exact["season"].eq(
            latest_season
        )
    ].copy()

    result["schedule_match_count"] = int(
        len(exact)
    )

    if len(exact) != 1:
        result["schedule_status"] = (
            "AMBIGUOUS"
        )
        return result

    row = exact.iloc[0]

    result.update(
        {
            "season": int(
                row["season"]
            ),
            "week": int(
                row["week"]
            ),
            "game_id": str(
                row["game_id"]
            ),
            "game_date": (
                row["game_date"]
                if pd.notna(
                    row["game_date"]
                )
                else None
            ),
            "game_time": str(
                row["game_time"]
            ),
            "schedule_status": "RESOLVED",
            "schedule_match_count": 1,
        }
    )

    return result


def attach_schedule_identity(
    df: pd.DataFrame,
    schedule: pd.DataFrame,
    away_column: str = "away_team",
    home_column: str = "home_team",
) -> pd.DataFrame:
    if away_column not in df.columns:
        raise RuntimeError(
            f"Missing away-team column: "
            f"{away_column}"
        )

    if home_column not in df.columns:
        raise RuntimeError(
            f"Missing home-team column: "
            f"{home_column}"
        )

    unique_games = (
        df[
            [
                away_column,
                home_column,
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    resolutions = []

    for row in unique_games.itertuples(
        index=False,
        name=None,
    ):
        away, home = row

        resolutions.append(
            resolve_matchup(
                schedule,
                away,
                home,
            )
        )

    identity = pd.DataFrame(
        resolutions
    )

    identity = identity.rename(
        columns={
            "away_team": away_column,
            "home_team": home_column,
        }
    )

    out = df.merge(
        identity,
        how="left",
        on=[
            away_column,
            home_column,
        ],
        validate="many_to_one",
    )

    out["schedule_status"] = (
        out["schedule_status"]
        .fillna("UNRESOLVED")
    )

    out[
        "schedule_match_count"
    ] = (
        pd.to_numeric(
            out[
                "schedule_match_count"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    return out


def summarize_slate_identity(
    df: pd.DataFrame,
) -> dict:
    required = [
        "season",
        "week",
        "game_id",
        "schedule_status",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Cannot summarize schedule identity; "
            "missing columns: "
            + ", ".join(missing)
        )

    resolved = df[
        df[
            "schedule_status"
        ].eq("RESOLVED")
    ].copy()

    unresolved_rows = int(
        (
            ~df[
                "schedule_status"
            ].eq("RESOLVED")
        ).sum()
    )

    identities = (
        resolved[
            [
                "season",
                "week",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "week",
            ]
        )
    )

    season_week_pairs = [
        (
            int(row.season),
            int(row.week),
        )
        for row in identities.itertuples(
            index=False
        )
    ]

    game_ids = sorted(
        str(value)
        for value in resolved[
            "game_id"
        ].dropna().unique()
        if str(value).strip()
    )

    return {
        "unresolved_schedule_rows": (
            unresolved_rows
        ),
        "schedule_identity_count": (
            len(season_week_pairs)
        ),
        "schedule_identities": (
            season_week_pairs
        ),
        "schedule_game_ids": (
            game_ids
        ),
    }
