#!/usr/bin/env python3

"""
dst_fanduel_scoring.py

Verified historical FanDuel NFL D/ST scoring layer.

Input
-----
SQLite:
    team_defense_game_stats
    games

nflverse play-by-play:
    2023, 2024, 2025 via nflreadpy.load_pbp()

Output
------
SQLite:
    team_defense_fanduel_scoring

CSV:
    nfl_team_defense_fanduel_scoring.csv
    audit_dst_fanduel_scoring_summary.csv
    audit_dst_fanduel_points_allowed.csv

Parquet:
    nfl_team_defense_fanduel_scoring.parquet

FanDuel D/ST scoring used
-------------------------
Sack                         +1
Opponent fumble recovered    +2
Interception                 +2
Safety                       +2
Blocked punt/kick            +2
Return touchdown             +6
Extra-point / conversion
return                       +2

Points allowed:
0                            +10
1-6                           +7
7-13                          +4
14-20                         +1
21-27                          0
28-34                         -1
35+                            -4

Important FanDuel PA semantics
------------------------------
FanDuel points allowed are NOT simply the opponent's final scoreboard total.
This script reconstructs points allowed from offensive/kicking scoring:
    - touchdowns credited to the opponent's possession team
    - made field goals
    - made extra points
    - successful two-point conversions

Defensive and special-teams return scores are therefore excluded from the
points-allowed total.

This is a DERIVED scoring layer. It does not modify:
    - team_defense_game_stats
    - any offensive player table
    - any offensive projection/model artifact
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import nflreadpy as nfl
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


HISTORICAL_SEASONS = [2023, 2024, 2025]

SOURCE_TABLE = "team_defense_game_stats"
OUTPUT_TABLE = "team_defense_fanduel_scoring"

CSV_OUTPUT = Path(CSV_DIR) / "nfl_team_defense_fanduel_scoring.csv"
PARQUET_OUTPUT = (
    Path(PARQUET_DIR) / "nfl_team_defense_fanduel_scoring.parquet"
)

AUDIT_SUMMARY = (
    Path(CSV_DIR) / "audit_dst_fanduel_scoring_summary.csv"
)
AUDIT_POINTS_ALLOWED = (
    Path(CSV_DIR) / "audit_dst_fanduel_points_allowed.csv"
)


PA_PBP_FIELDS = [
    "game_id",
    "season",
    "week",
    "play_id",
    "posteam",
    "defteam",
    "touchdown",
    "td_team",
    "field_goal_attempt",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "two_point_attempt",
    "two_point_conv_result",
    "punt_attempt",
    "kickoff_attempt",
    "interception",
    "fumble",
    "fumble_lost",
    "fumble_recovery_1_team",
    "return_touchdown",
    "rush_touchdown",
    "pass_touchdown",
    "desc",
]

REQUIRED_PA_FIELDS = [
    "game_id",
    "play_id",
    "posteam",
    "touchdown",
    "td_team",
    "field_goal_result",
    "extra_point_result",
    "two_point_conv_result",
]


def section(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
    )


def lower_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            "",
            index=df.index,
            dtype="string",
        )

    return (
        normalize_text(df[column])
        .fillna("")
        .str.lower()
    )


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            0.0,
            index=df.index,
            dtype=float,
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(0.0)
    )


def bool_flag(df: pd.DataFrame, column: str) -> pd.Series:
    return numeric(df, column).eq(1)


def load_raw_dst(conn: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(
        ["?"] * len(HISTORICAL_SEASONS)
    )

    query = f"""
        SELECT *
        FROM {qident(SOURCE_TABLE)}
        WHERE season IN ({placeholders})
        ORDER BY season, week, game_id, team
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=HISTORICAL_SEASONS,
    )

    if df.empty:
        raise RuntimeError(
            f"{SOURCE_TABLE} has no historical rows."
        )

    required = {
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
        "opponent_score",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
        "defensive_two_point_returns",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"{SOURCE_TABLE} missing required fields: "
            + ", ".join(missing)
        )

    duplicates = int(
        df.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"{SOURCE_TABLE} contains "
            f"{duplicates} duplicate game/team rows."
        )

    return df


def load_pbp_season(season: int) -> pd.DataFrame:
    section(
        f"LOADING PBP FOR FANDUEL POINTS ALLOWED — {season}"
    )

    pl_df = nfl.load_pbp(season)

    source_columns = set(pl_df.columns)

    missing = [
        c
        for c in REQUIRED_PA_FIELDS
        if c not in source_columns
    ]

    if missing:
        raise RuntimeError(
            f"{season} PBP missing required PA fields: "
            + ", ".join(missing)
        )

    selected = [
        c
        for c in PA_PBP_FIELDS
        if c in source_columns
    ]

    df = pl_df.select(selected).to_pandas()

    for field in PA_PBP_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA

    print(f"Source rows: {len(df)}")
    print(
        "Required PA fields: PASS"
    )

    return df


def classify_scoring_plays(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct points charged to the opposing D/ST for FanDuel PA.

    We credit scoring to the possession team when:
      - touchdown=1 and td_team == posteam
      - field_goal_result == made
      - extra_point_result == good/made
      - two_point_conv_result == success

    This excludes defensive and special-teams return touchdowns because
    those scores are not credited to the possession offense.
    """

    game_id = normalize_text(pbp["game_id"])
    posteam = normalize_text(pbp["posteam"])
    td_team = normalize_text(pbp["td_team"])

    plays = []

    # --------------------------------------------------------------
    # OFFENSIVE TOUCHDOWNS
    #
    # FanDuel PA charges touchdowns scored by the opponent offense.
    # Using td_team == posteam naturally includes ordinary rushing /
    # receiving TDs and offensive own-fumble-recovery TDs, while
    # excluding defensive and return TDs by the non-possession team.
    # --------------------------------------------------------------
    offensive_td = (
        bool_flag(pbp, "touchdown")
        & posteam.notna()
        & td_team.notna()
        & td_team.eq(posteam)
    )

    if offensive_td.any():
        plays.append(
            pd.DataFrame(
                {
                    "game_id": game_id[offensive_td],
                    "scoring_team": posteam[offensive_td],
                    "scoring_type": "OFFENSIVE_TD",
                    "pa_points": 6,
                }
            )
        )

    # --------------------------------------------------------------
    # FIELD GOALS
    # --------------------------------------------------------------
    fg_result = lower_text(
        pbp,
        "field_goal_result",
    )

    made_fg = (
        posteam.notna()
        & fg_result.isin(
            ["made", "good"]
        )
    )

    if made_fg.any():
        plays.append(
            pd.DataFrame(
                {
                    "game_id": game_id[made_fg],
                    "scoring_team": posteam[made_fg],
                    "scoring_type": "FIELD_GOAL",
                    "pa_points": 3,
                }
            )
        )

    # --------------------------------------------------------------
    # EXTRA POINTS
    # --------------------------------------------------------------
    xp_result = lower_text(
        pbp,
        "extra_point_result",
    )

    made_xp = (
        posteam.notna()
        & xp_result.isin(
            ["good", "made"]
        )
    )

    if made_xp.any():
        plays.append(
            pd.DataFrame(
                {
                    "game_id": game_id[made_xp],
                    "scoring_team": posteam[made_xp],
                    "scoring_type": "EXTRA_POINT",
                    "pa_points": 1,
                }
            )
        )

    # --------------------------------------------------------------
    # TWO-POINT CONVERSIONS
    # --------------------------------------------------------------
    two_result = lower_text(
        pbp,
        "two_point_conv_result",
    )

    made_two = (
        posteam.notna()
        & two_result.isin(
            [
                "success",
                "successful",
                "good",
                "made",
            ]
        )
    )

    if made_two.any():
        plays.append(
            pd.DataFrame(
                {
                    "game_id": game_id[made_two],
                    "scoring_team": posteam[made_two],
                    "scoring_type": "TWO_POINT_CONVERSION",
                    "pa_points": 2,
                }
            )
        )

    if not plays:
        return pd.DataFrame(
            columns=[
                "game_id",
                "scoring_team",
                "scoring_type",
                "pa_points",
            ]
        )

    out = pd.concat(
        plays,
        ignore_index=True,
    )

    return out


def aggregate_points_allowed(
    raw_dst: pd.DataFrame,
    scoring_plays: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert opponent offensive/kicking scoring into one FanDuel PA total
    per defense/team-game.
    """

    if scoring_plays.empty:
        scoring_by_team = pd.DataFrame(
            columns=[
                "game_id",
                "scoring_team",
                "fanduel_points_allowed",
                "offensive_tds_allowed",
                "field_goals_allowed",
                "extra_points_allowed",
                "two_point_conversions_allowed",
            ]
        )
    else:
        totals = (
            scoring_plays.groupby(
                ["game_id", "scoring_team"],
                as_index=False,
            )["pa_points"]
            .sum()
            .rename(
                columns={
                    "pa_points":
                        "fanduel_points_allowed"
                }
            )
        )

        counts = (
            scoring_plays.assign(n=1)
            .pivot_table(
                index=[
                    "game_id",
                    "scoring_team",
                ],
                columns="scoring_type",
                values="n",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        counts.columns.name = None

        rename = {
            "OFFENSIVE_TD":
                "offensive_tds_allowed",
            "FIELD_GOAL":
                "field_goals_allowed",
            "EXTRA_POINT":
                "extra_points_allowed",
            "TWO_POINT_CONVERSION":
                "two_point_conversions_allowed",
        }

        counts = counts.rename(
            columns=rename
        )

        for col in rename.values():
            if col not in counts.columns:
                counts[col] = 0

        scoring_by_team = totals.merge(
            counts,
            on=[
                "game_id",
                "scoring_team",
            ],
            how="outer",
            validate="one_to_one",
        )

    out = raw_dst.copy()

    out = out.merge(
        scoring_by_team,
        left_on=[
            "game_id",
            "opponent_team",
        ],
        right_on=[
            "game_id",
            "scoring_team",
        ],
        how="left",
        validate="many_to_one",
    )

    count_cols = [
        "offensive_tds_allowed",
        "field_goals_allowed",
        "extra_points_allowed",
        "two_point_conversions_allowed",
    ]

    out[
        "fanduel_points_allowed"
    ] = (
        pd.to_numeric(
            out["fanduel_points_allowed"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    for col in count_cols:
        out[col] = (
            pd.to_numeric(
                out[col],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    out = out.drop(
        columns=["scoring_team"],
        errors="ignore",
    )

    return out, scoring_by_team


def points_allowed_bonus(
    points: pd.Series,
) -> pd.Series:
    """
    FanDuel D/ST points-allowed tier scoring.
    """
    p = pd.to_numeric(
        points,
        errors="raise",
    )

    result = pd.Series(
        0.0,
        index=p.index,
    )

    result.loc[p.eq(0)] = 10.0
    result.loc[p.between(1, 6)] = 7.0
    result.loc[p.between(7, 13)] = 4.0
    result.loc[p.between(14, 20)] = 1.0
    result.loc[p.between(21, 27)] = 0.0
    result.loc[p.between(28, 34)] = -1.0
    result.loc[p.ge(35)] = -4.0

    return result


def score_dst(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    integer_fields = [
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
        "defensive_two_point_returns",
    ]

    for col in integer_fields:
        out[col] = (
            pd.to_numeric(
                out[col],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    out["sack_points"] = (
        out["sacks"] * 1.0
    )

    out["interception_points"] = (
        out["interceptions"] * 2.0
    )

    out["fumble_recovery_points"] = (
        out["fumble_recoveries"] * 2.0
    )

    out["safety_points"] = (
        out["safeties"] * 2.0
    )

    out["blocked_kick_points"] = (
        out["blocked_kicks"] * 2.0
    )

    out["return_td_points"] = (
        out["return_tds"] * 6.0
    )

    out[
        "defensive_two_point_return_points"
    ] = (
        out[
            "defensive_two_point_returns"
        ] * 2.0
    )

    out[
        "points_allowed_tier_points"
    ] = points_allowed_bonus(
        out[
            "fanduel_points_allowed"
        ]
    )

    component_cols = [
        "sack_points",
        "interception_points",
        "fumble_recovery_points",
        "safety_points",
        "blocked_kick_points",
        "return_td_points",
        "defensive_two_point_return_points",
        "points_allowed_tier_points",
    ]

    out["fanduel_dst_points"] = (
        out[component_cols]
        .sum(axis=1)
        .round(2)
    )

    out["fanduel_scoring_verified"] = 1
    out["fanduel_scored_at"] = utc_now()

    return out


def build_pa_audit(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    audit = scored[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
            "opponent_score",
            "fanduel_points_allowed",
            "offensive_tds_allowed",
            "field_goals_allowed",
            "extra_points_allowed",
            "two_point_conversions_allowed",
        ]
    ].copy()

    audit[
        "scoreboard_minus_fanduel_pa"
    ] = (
        pd.to_numeric(
            audit["opponent_score"],
            errors="coerce",
        )
        -
        pd.to_numeric(
            audit[
                "fanduel_points_allowed"
            ],
            errors="coerce",
        )
    )

    audit["pa_exceeds_scoreboard"] = (
        audit[
            "fanduel_points_allowed"
        ]
        >
        audit["opponent_score"]
    ).astype(int)

    audit["negative_fanduel_pa"] = (
        audit[
            "fanduel_points_allowed"
        ] < 0
    ).astype(int)

    return audit


def audit_scoring(
    raw_dst: pd.DataFrame,
    scored: pd.DataFrame,
    pa_audit: pd.DataFrame,
) -> pd.DataFrame:
    section("FANDUEL D/ST SCORING AUDIT")

    rows = []

    def add(
        item,
        value,
        expected="",
        status="INFO",
    ):
        rows.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    expected_rows = len(raw_dst)

    add(
        "raw_dst_rows",
        len(raw_dst),
        1710,
        (
            "PASS"
            if len(raw_dst) == 1710
            else "CHECK"
        ),
    )

    add(
        "scored_rows",
        len(scored),
        expected_rows,
        (
            "PASS"
            if len(scored) == expected_rows
            else "FAIL"
        ),
    )

    duplicates = int(
        scored.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    add(
        "duplicate_game_team_rows",
        duplicates,
        0,
        (
            "PASS"
            if duplicates == 0
            else "FAIL"
        ),
    )

    null_fd = int(
        scored[
            "fanduel_dst_points"
        ].isna().sum()
    )

    add(
        "null_fanduel_dst_points",
        null_fd,
        0,
        (
            "PASS"
            if null_fd == 0
            else "FAIL"
        ),
    )

    null_pa = int(
        scored[
            "fanduel_points_allowed"
        ].isna().sum()
    )

    add(
        "null_fanduel_points_allowed",
        null_pa,
        0,
        (
            "PASS"
            if null_pa == 0
            else "FAIL"
        ),
    )

    negative_pa = int(
        pa_audit[
            "negative_fanduel_pa"
        ].sum()
    )

    add(
        "negative_fanduel_pa_rows",
        negative_pa,
        0,
        (
            "PASS"
            if negative_pa == 0
            else "FAIL"
        ),
    )

    pa_exceeds = int(
        pa_audit[
            "pa_exceeds_scoreboard"
        ].sum()
    )

    add(
        "fanduel_pa_exceeds_scoreboard_rows",
        pa_exceeds,
        0,
        (
            "PASS"
            if pa_exceeds == 0
            else "FAIL"
        ),
    )

    component_cols = [
        "sack_points",
        "interception_points",
        "fumble_recovery_points",
        "safety_points",
        "blocked_kick_points",
        "return_td_points",
        "defensive_two_point_return_points",
        "points_allowed_tier_points",
    ]

    reconstructed = (
        scored[component_cols]
        .sum(axis=1)
        .round(2)
    )

    component_errors = int(
        reconstructed.ne(
            scored[
                "fanduel_dst_points"
            ].round(2)
        ).sum()
    )

    add(
        "component_sum_errors",
        component_errors,
        0,
        (
            "PASS"
            if component_errors == 0
            else "FAIL"
        ),
    )

    tier_valid_values = {
        10.0,
        7.0,
        4.0,
        1.0,
        0.0,
        -1.0,
        -4.0,
    }

    invalid_tiers = int(
        ~scored[
            "points_allowed_tier_points"
        ].isin(tier_valid_values)
    .sum()
    ) if False else int(
        (
            ~scored[
                "points_allowed_tier_points"
            ].isin(tier_valid_values)
        ).sum()
    )

    add(
        "invalid_points_allowed_tier_values",
        invalid_tiers,
        0,
        (
            "PASS"
            if invalid_tiers == 0
            else "FAIL"
        ),
    )

    verified_bad = int(
        scored[
            "fanduel_scoring_verified"
        ].ne(1).sum()
    )

    add(
        "unverified_scoring_rows",
        verified_bad,
        0,
        (
            "PASS"
            if verified_bad == 0
            else "FAIL"
        ),
    )

    # Historical PA differences are expected where opponent points came
    # from defense/special teams. They are informational unless negative
    # or impossible.
    pa_diff_rows = int(
        pa_audit[
            "scoreboard_minus_fanduel_pa"
        ].fillna(0).ne(0).sum()
    )

    add(
        "scoreboard_differs_from_fanduel_pa_rows",
        pa_diff_rows,
        "",
        "INFO",
    )

    add(
        "max_scoreboard_minus_fanduel_pa",
        int(
            pa_audit[
                "scoreboard_minus_fanduel_pa"
            ]
            .dropna()
            .max()
        ),
        "",
        "INFO",
    )

    for season in HISTORICAL_SEASONS:
        season_df = scored[
            scored["season"].eq(season)
        ]

        add(
            f"rows_{season}",
            len(season_df),
            570,
            (
                "PASS"
                if len(season_df) == 570
                else "CHECK"
            ),
        )

        add(
            f"avg_fanduel_dst_points_{season}",
            round(
                float(
                    season_df[
                        "fanduel_dst_points"
                    ].mean()
                ),
                4,
            ),
            "",
            "INFO",
        )

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    failures = summary[
        summary["status"].eq("FAIL")
    ]

    if failures.empty:
        print()
        print("STRUCTURAL SCORING AUDIT: PASS")
    else:
        print()
        print("STRUCTURAL SCORING AUDIT: FAIL")

    return summary


def create_table(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {qident(OUTPUT_TABLE)} (
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT,
            team TEXT NOT NULL,
            opponent_team TEXT,
            home_away TEXT,
            team_score INTEGER,
            opponent_score INTEGER,
            completed INTEGER,

            sacks INTEGER NOT NULL,
            interceptions INTEGER NOT NULL,
            fumble_recoveries INTEGER NOT NULL,
            safeties INTEGER NOT NULL,
            blocked_punts INTEGER NOT NULL,
            blocked_field_goals INTEGER NOT NULL,
            blocked_extra_points INTEGER NOT NULL,
            blocked_kicks INTEGER NOT NULL,
            defensive_tds INTEGER NOT NULL,
            special_teams_tds INTEGER NOT NULL,
            return_tds INTEGER NOT NULL,
            defensive_two_point_returns INTEGER NOT NULL,

            offensive_tds_allowed INTEGER NOT NULL,
            field_goals_allowed INTEGER NOT NULL,
            extra_points_allowed INTEGER NOT NULL,
            two_point_conversions_allowed INTEGER NOT NULL,
            fanduel_points_allowed INTEGER NOT NULL,

            sack_points REAL NOT NULL,
            interception_points REAL NOT NULL,
            fumble_recovery_points REAL NOT NULL,
            safety_points REAL NOT NULL,
            blocked_kick_points REAL NOT NULL,
            return_td_points REAL NOT NULL,
            defensive_two_point_return_points REAL NOT NULL,
            points_allowed_tier_points REAL NOT NULL,

            fanduel_dst_points REAL NOT NULL,
            fanduel_scoring_verified INTEGER NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            fanduel_scored_at TEXT NOT NULL,

            PRIMARY KEY (game_id, team)
        )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_team_defense_fd_season_week
        ON {qident(OUTPUT_TABLE)}
        (season, week)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_team_defense_fd_team
        ON {qident(OUTPUT_TABLE)}
        (team)
        """
    )


def prepare_persistent_columns(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "team",
        "opponent_team",
        "home_away",
        "team_score",
        "opponent_score",
        "completed",

        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_punts",
        "blocked_field_goals",
        "blocked_extra_points",
        "blocked_kicks",
        "defensive_tds",
        "special_teams_tds",
        "return_tds",
        "defensive_two_point_returns",

        "offensive_tds_allowed",
        "field_goals_allowed",
        "extra_points_allowed",
        "two_point_conversions_allowed",
        "fanduel_points_allowed",

        "sack_points",
        "interception_points",
        "fumble_recovery_points",
        "safety_points",
        "blocked_kick_points",
        "return_td_points",
        "defensive_two_point_return_points",
        "points_allowed_tier_points",

        "fanduel_dst_points",
        "fanduel_scoring_verified",
        "source",
        "updated_at",
        "fanduel_scored_at",
    ]

    missing = [
        c
        for c in columns
        if c not in scored.columns
    ]

    if missing:
        raise RuntimeError(
            "Scored frame missing persistent fields: "
            + ", ".join(missing)
        )

    return scored[columns].copy()


def replace_scoring_rows(
    conn: sqlite3.Connection,
    scored: pd.DataFrame,
) -> None:
    create_table(conn)

    placeholders = ",".join(
        ["?"] * len(HISTORICAL_SEASONS)
    )

    conn.execute(
        f"""
        DELETE FROM {qident(OUTPUT_TABLE)}
        WHERE season IN ({placeholders})
        """,
        HISTORICAL_SEASONS,
    )

    columns = list(scored.columns)

    sql = f"""
        INSERT INTO {qident(OUTPUT_TABLE)}
        ({", ".join(qident(c) for c in columns)})
        VALUES ({", ".join(["?"] * len(columns))})
    """

    rows = []

    for row in scored.itertuples(
        index=False,
        name=None,
    ):
        cleaned = []

        for value in row:
            if pd.isna(value):
                cleaned.append(None)
            elif hasattr(value, "item"):
                cleaned.append(value.item())
            else:
                cleaned.append(value)

        rows.append(tuple(cleaned))

    conn.executemany(sql, rows)


def export_outputs(
    conn: sqlite3.Connection,
    summary: pd.DataFrame,
    pa_audit: pd.DataFrame,
) -> None:
    section("EXPORTING FANDUEL D/ST SCORING")

    Path(CSV_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(PARQUET_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    placeholders = ",".join(
        ["?"] * len(HISTORICAL_SEASONS)
    )

    out = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(OUTPUT_TABLE)}
        WHERE season IN ({placeholders})
        ORDER BY season, week, game_id, team
        """,
        conn,
        params=HISTORICAL_SEASONS,
    )

    out.to_csv(
        CSV_OUTPUT,
        index=False,
    )
    out.to_parquet(
        PARQUET_OUTPUT,
        index=False,
    )

    summary.to_csv(
        AUDIT_SUMMARY,
        index=False,
    )

    pa_audit.to_csv(
        AUDIT_POINTS_ALLOWED,
        index=False,
    )

    print(f"Rows exported: {len(out)}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Parquet: {PARQUET_OUTPUT}")
    print(f"Audit summary: {AUDIT_SUMMARY}")
    print(
        "Points-allowed audit: "
        f"{AUDIT_POINTS_ALLOWED}"
    )


def print_leaders(
    scored: pd.DataFrame,
) -> None:
    section("HISTORICAL D/ST FANDUEL LEADERS")

    cols = [
        "season",
        "week",
        "team",
        "opponent_team",
        "fanduel_points_allowed",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "return_tds",
        "fanduel_dst_points",
    ]

    leaders = (
        scored[
            cols
        ]
        .sort_values(
            [
                "fanduel_dst_points",
                "season",
                "week",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(20)
    )

    print(
        leaders.to_string(
            index=False
        )
    )


def main() -> None:
    section("HISTORICAL FANDUEL D/ST SCORING BUILD")

    print(f"Database: {DATABASE_PATH}")
    print(f"Seasons: {HISTORICAL_SEASONS}")
    print(
        "Raw source table: "
        f"{SOURCE_TABLE}"
    )
    print(
        "Output scoring table: "
        f"{OUTPUT_TABLE}"
    )
    print(
        "Offensive production pipeline: "
        "NOT modified."
    )

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: "
            f"{DATABASE_PATH}"
        )

    with sqlite3.connect(
        DATABASE_PATH
    ) as conn:

        raw_dst = load_raw_dst(conn)

        print()
        print(
            f"Raw D/ST team-game rows: "
            f"{len(raw_dst)}"
        )

        pbp_frames = []

        for season in HISTORICAL_SEASONS:
            pbp_frames.append(
                load_pbp_season(season)
            )

        section(
            "RECONSTRUCTING FANDUEL POINTS ALLOWED"
        )

        scoring_play_frames = []

        for season, pbp in zip(
            HISTORICAL_SEASONS,
            pbp_frames,
        ):
            plays = classify_scoring_plays(
                pbp
            )
            plays["season_source"] = season
            scoring_play_frames.append(
                plays
            )

            print(
                f"{season}: "
                f"{len(plays)} PA scoring plays"
            )

        scoring_plays = pd.concat(
            scoring_play_frames,
            ignore_index=True,
        )

        scored_base, _ = (
            aggregate_points_allowed(
                raw_dst,
                scoring_plays,
            )
        )

        section("CALCULATING FANDUEL D/ST POINTS")

        scored = score_dst(
            scored_base
        )

        pa_audit = build_pa_audit(
            scored
        )

        summary = audit_scoring(
            raw_dst,
            scored,
            pa_audit,
        )

        failures = summary[
            summary["status"].eq("FAIL")
        ]

        if not failures.empty:
            Path(CSV_DIR).mkdir(
                parents=True,
                exist_ok=True,
            )

            summary.to_csv(
                AUDIT_SUMMARY,
                index=False,
            )

            pa_audit.to_csv(
                AUDIT_POINTS_ALLOWED,
                index=False,
            )

            raise RuntimeError(
                "FanDuel D/ST scoring structural "
                "audit failed. Database write aborted."
            )

        persistent = (
            prepare_persistent_columns(
                scored
            )
        )

        section(
            "WRITING DERIVED FANDUEL D/ST TABLE"
        )

        replace_scoring_rows(
            conn,
            persistent,
        )

        conn.commit()

        stored = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {qident(OUTPUT_TABLE)}
            WHERE season IN (2023, 2024, 2025)
            """
        ).fetchone()[0]

        print(
            f"Stored historical scoring rows: "
            f"{stored}"
        )

        if stored != len(persistent):
            raise RuntimeError(
                "Stored row mismatch: "
                f"{stored} != {len(persistent)}"
            )

        export_outputs(
            conn,
            summary,
            pa_audit,
        )

        print_leaders(
            scored
        )

    section(
        "FANDUEL D/ST SCORING BUILD COMPLETE"
    )

    print(
        "Historical D/ST scoring layer created."
    )
    print(
        "Raw DST history remains unchanged."
    )
    print(
        "Offensive production baseline remains unchanged."
    )
    print(
        "Next layer after audit review: "
        "leakage-safe D/ST pregame features."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 84)
        print("FANDUEL D/ST SCORING BUILD FAILED")
        print("=" * 84)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)
