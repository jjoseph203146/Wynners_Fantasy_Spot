#!/usr/bin/env python3

"""
dst_pregame_features.py

Leakage-safe historical NFL D/ST pregame feature builder.

Frozen inputs
-------------
SQLite:
    games
    team_defense_game_stats
    team_defense_fanduel_scoring
    team_game_stats

Output
------
SQLite:
    dst_pregame_features

CSV:
    nfl_dst_pregame_features.csv
    audit_dst_pregame_features_summary.csv

Parquet:
    nfl_dst_pregame_features.parquet

Design rules
------------
1. Every rolling/statistical feature is shifted by one game before rolling.
2. Current-game D/ST outcome is retained only as the modeling target.
3. Opponent offensive features are also strictly lagged by one game.
4. No current-game score, sack, turnover, or fantasy-point information is
   allowed into current-game predictors.
5. Frozen historical D/ST source/scoring tables are never modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


HISTORICAL_SEASONS = [2023, 2024, 2025]

SOURCE_SCORE_TABLE = "team_defense_fanduel_scoring"
SOURCE_TEAM_TABLE = "team_game_stats"
OUTPUT_TABLE = "dst_pregame_features"

CSV_OUTPUT = Path(CSV_DIR) / "nfl_dst_pregame_features.csv"
PARQUET_OUTPUT = Path(PARQUET_DIR) / "nfl_dst_pregame_features.parquet"
AUDIT_OUTPUT = Path(CSV_DIR) / "audit_dst_pregame_features_summary.csv"


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_dst_scoring(conn: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(SOURCE_SCORE_TABLE)}
        WHERE season IN ({placeholders})
        ORDER BY season, week, game_id, team
        """,
        conn,
        params=HISTORICAL_SEASONS,
    )

    if df.empty:
        raise RuntimeError(
            f"{SOURCE_SCORE_TABLE} returned zero rows."
        )

    required = {
        "game_id",
        "season",
        "week",
        "game_type",
        "team",
        "opponent_team",
        "home_away",
        "team_score",
        "opponent_score",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
        "fanduel_points_allowed",
        "fanduel_dst_points",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Missing required D/ST scoring columns: "
            + ", ".join(missing)
        )

    dupes = int(df.duplicated(["game_id", "team"]).sum())
    if dupes:
        raise RuntimeError(
            f"{SOURCE_SCORE_TABLE} has {dupes} duplicate game/team rows."
        )

    return df


def load_games(conn: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    df = pd.read_sql_query(
        f"""
        SELECT
            game_id,
            season,
            game_type,
            week,
            game_date,
            gametime,
            away_team,
            home_team,
            away_score,
            home_score,
            completed
        FROM games
        WHERE season IN ({placeholders})
        """,
        conn,
        params=HISTORICAL_SEASONS,
    )

    if df.empty:
        raise RuntimeError("games returned zero historical rows.")

    return df


def load_team_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(HISTORICAL_SEASONS))

    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(SOURCE_TEAM_TABLE)}
        WHERE season IN ({placeholders})
        """,
        conn,
        params=HISTORICAL_SEASONS,
    )

    if df.empty:
        raise RuntimeError(
            f"{SOURCE_TEAM_TABLE} returned zero historical rows."
        )

    if "game_id" not in df.columns or "team" not in df.columns:
        raise RuntimeError(
            f"{SOURCE_TEAM_TABLE} must contain game_id and team."
        )

    dupes = int(df.duplicated(["game_id", "team"]).sum())
    if dupes:
        raise RuntimeError(
            f"{SOURCE_TEAM_TABLE} has {dupes} duplicate game/team rows."
        )

    return df


def attach_game_datetime(
    dst: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    game_cols = [
        "game_id",
        "game_date",
        "gametime",
        "away_team",
        "home_team",
        "completed",
    ]

    out = dst.merge(
        games[game_cols],
        on="game_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_game"),
    )

    if out["game_date"].isna().any():
        missing = int(out["game_date"].isna().sum())
        raise RuntimeError(
            f"{missing} D/ST rows failed game-date attachment."
        )

    date_part = pd.to_datetime(
        out["game_date"],
        errors="coerce",
    )

    time_text = (
        out["gametime"]
        .astype("string")
        .fillna("00:00")
        .str.strip()
    )

    time_text = time_text.where(
        time_text.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False),
        "00:00",
    )

    out["game_datetime"] = pd.to_datetime(
        date_part.dt.strftime("%Y-%m-%d")
        + " "
        + time_text,
        errors="coerce",
    )

    fallback = out["game_datetime"].isna()
    out.loc[fallback, "game_datetime"] = date_part[fallback]

    if out["game_datetime"].isna().any():
        raise RuntimeError(
            "Unable to construct game_datetime for all D/ST rows."
        )

    return out


def add_team_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        ["team", "game_datetime", "game_id"]
    ).copy()

    previous_date = (
        out.groupby("team")["game_datetime"]
        .shift(1)
    )

    rest_days = (
        out["game_datetime"] - previous_date
    ).dt.total_seconds() / 86400.0

    out["def_rest_days"] = rest_days
    out["def_short_week_flag"] = (
        out["def_rest_days"].lt(6)
    ).astype("Int64")
    out["def_long_rest_flag"] = (
        out["def_rest_days"].gt(8)
    ).astype("Int64")

    return out


def add_defense_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        ["team", "game_datetime", "game_id"]
    ).copy()

    metrics = {
        "fanduel_dst_points": "fd",
        "fanduel_points_allowed": "pa",
        "sacks": "sacks",
        "interceptions": "ints",
        "fumble_recoveries": "fum_rec",
        "safeties": "safeties",
        "blocked_kicks": "blocks",
        "return_tds": "return_tds",
    }

    for raw_col, short in metrics.items():
        shifted = out.groupby("team")[raw_col].shift(1)

        out[f"def_{short}_last"] = shifted

        for window in (3, 5):
            out[f"def_{short}_avg_{window}"] = (
                shifted.groupby(out["team"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        if raw_col == "fanduel_dst_points":
            out["def_fd_std_5"] = (
                shifted.groupby(out["team"])
                .rolling(5, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

    out["def_games_prior"] = (
        out.groupby("team").cumcount()
    )

    shifted_fd = (
        out.groupby("team")["fanduel_dst_points"]
        .shift(1)
    )

    # Preserve first-observed-game history as NULL.
    # A direct .gt/.ge on NaN converts the missing lag to False/0,
    # which is not leakage but incorrectly makes a history feature
    # appear populated on the team's first observed game.
    positive_flag = pd.Series(
        np.where(
            shifted_fd.isna(),
            np.nan,
            shifted_fd.gt(0).astype(float),
        ),
        index=out.index,
        dtype=float,
    )

    double_digit_flag = pd.Series(
        np.where(
            shifted_fd.isna(),
            np.nan,
            shifted_fd.ge(10).astype(float),
        ),
        index=out.index,
        dtype=float,
    )

    out["def_positive_games_5"] = (
        positive_flag.groupby(out["team"])
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    out["def_double_digit_games_5"] = (
        double_digit_flag.groupby(out["team"])
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    out["def_takeaways_last"] = (
        out["def_ints_last"]
        + out["def_fum_rec_last"]
    )

    out["def_takeaways_avg_3"] = (
        out["def_ints_avg_3"]
        + out["def_fum_rec_avg_3"]
    )

    out["def_takeaways_avg_5"] = (
        out["def_ints_avg_5"]
        + out["def_fum_rec_avg_5"]
    )

    return out


def infer_team_stat_columns(team_stats: pd.DataFrame) -> dict[str, str]:
    """
    Resolve the offensive team-game columns that are useful for opponent
    pregame matchup features. We only use exact column-name candidates.
    """

    candidates = {
        "pass_attempts": [
            "attempts",
            "passing_attempts",
        ],
        "passing_yards": [
            "passing_yards",
        ],
        "passing_tds": [
            "passing_tds",
        ],
        "interceptions_thrown": [
            "interceptions",
        ],
        "sacks_allowed": [
            "sacks_suffered",
        ],
        "rush_attempts": [
            "carries",
            "rushing_attempts",
        ],
        "rushing_yards": [
            "rushing_yards",
        ],
        "rushing_tds": [
            "rushing_tds",
        ],
        "total_yards": [
            "total_yards",
        ],
        "special_teams_tds": [
            "special_teams_tds",
        ],
    }

    resolved = {}

    for target, options in candidates.items():
        for option in options:
            if option in team_stats.columns:
                resolved[target] = option
                break

    return resolved


def build_opponent_offense_features(
    team_stats: pd.DataFrame,
    game_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build strictly lagged offensive rolling features by team.
    """

    mapping = infer_team_stat_columns(team_stats)

    if not mapping:
        raise RuntimeError(
            "No recognized offensive team-game columns found."
        )

    base = team_stats[["game_id", "team"] + sorted(set(mapping.values()))].copy()

    base = base.merge(
        game_lookup[["game_id", "game_datetime"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    if base["game_datetime"].isna().any():
        raise RuntimeError(
            "Opponent offense rows missing game_datetime."
        )

    base = base.sort_values(
        ["team", "game_datetime", "game_id"]
    ).copy()

    out = base[["game_id", "team", "game_datetime"]].copy()

    for target, source_col in mapping.items():
        values = safe_numeric(base[source_col])
        shifted = values.groupby(base["team"]).shift(1)

        out[f"opp_{target}_last"] = shifted

        for window in (3, 5):
            out[f"opp_{target}_avg_{window}"] = (
                shifted.groupby(base["team"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    out["opp_off_games_prior"] = (
        base.groupby("team").cumcount()
    )

    rename_map = {
        "team": "opponent_team"
    }

    return out.rename(columns=rename_map)


def attach_opponent_features(
    dst: pd.DataFrame,
    opp: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [
        c
        for c in opp.columns
        if c not in {"game_datetime"}
    ]

    out = dst.merge(
        opp[feature_cols],
        on=["game_id", "opponent_team"],
        how="left",
        validate="one_to_one",
    )

    return out


def add_opponent_rest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Opponent rest is structurally known pregame and does not use outcome.
    """
    all_team_games = pd.concat(
        [
            df[
                [
                    "game_id",
                    "game_datetime",
                    "team",
                ]
            ].rename(
                columns={
                    "team": "lookup_team"
                }
            ),
            df[
                [
                    "game_id",
                    "game_datetime",
                    "opponent_team",
                ]
            ].rename(
                columns={
                    "opponent_team": "lookup_team"
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(
        ["game_id", "lookup_team"]
    )

    all_team_games = all_team_games.sort_values(
        ["lookup_team", "game_datetime", "game_id"]
    )

    all_team_games["opp_rest_days"] = (
        (
            all_team_games["game_datetime"]
            -
            all_team_games.groupby("lookup_team")["game_datetime"].shift(1)
        )
        .dt.total_seconds()
        .div(86400.0)
    )

    opp_rest = all_team_games[
        ["game_id", "lookup_team", "opp_rest_days"]
    ].rename(
        columns={
            "lookup_team": "opponent_team"
        }
    )

    out = df.merge(
        opp_rest,
        on=["game_id", "opponent_team"],
        how="left",
        validate="one_to_one",
    )

    out["opp_short_week_flag"] = (
        out["opp_rest_days"].lt(6)
    ).astype("Int64")

    out["opp_long_rest_flag"] = (
        out["opp_rest_days"].gt(8)
    ).astype("Int64")

    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_home"] = (
        out["home_away"]
        .astype("string")
        .str.upper()
        .eq("HOME")
    ).astype(int)

    out["is_playoffs"] = (
        out["game_type"]
        .astype("string")
        .str.upper()
        .ne("REG")
    ).astype(int)

    return out


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "game_id",
        "season",
        "week",
        "game_type",
        "game_date",
        "gametime",
        "game_datetime",
        "team",
        "opponent_team",
        "home_away",
        "is_home",
        "is_playoffs",
    ]

    target = [
        "fanduel_dst_points",
        "fanduel_points_allowed",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
    ]

    feature_prefixes = (
        "def_",
        "opp_",
    )

    feature_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith(feature_prefixes)
            and c not in target
        ]
    )

    out = df[
        metadata + feature_cols + target
    ].copy()

    out["feature_built_at"] = utc_now()

    return out


def leakage_audit(
    original: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    section("D/ST PREGAME FEATURE AUDIT")

    rows = []

    def add(item, value, expected="", status="INFO"):
        rows.append(
            {
                "item": item,
                "value": value,
                "expected": expected,
                "status": status,
            }
        )

    add(
        "source_rows",
        len(original),
        1710,
        "PASS" if len(original) == 1710 else "CHECK",
    )

    add(
        "feature_rows",
        len(features),
        len(original),
        "PASS" if len(features) == len(original) else "FAIL",
    )

    dupes = int(
        features.duplicated(["game_id", "team"]).sum()
    )
    add(
        "duplicate_game_team_rows",
        dupes,
        0,
        "PASS" if dupes == 0 else "FAIL",
    )

    null_targets = int(
        features["fanduel_dst_points"].isna().sum()
    )
    add(
        "null_target_rows",
        null_targets,
        0,
        "PASS" if null_targets == 0 else "FAIL",
    )

    for season in HISTORICAL_SEASONS:
        n = int(features["season"].eq(season).sum())
        add(
            f"rows_{season}",
            n,
            570,
            "PASS" if n == 570 else "CHECK",
        )

    # --------------------------------------------------------------
    # Strong leakage checks:
    # On each team's first observed game, all defense outcome-based
    # rolling fields must be null / zero-history.
    # --------------------------------------------------------------
    first = features["def_games_prior"].eq(0)

    add(
        "first_observed_team_games",
        int(first.sum()),
        "",
        "INFO",
    )

    def_history_cols = [
        c
        for c in features.columns
        if (
            c.startswith("def_")
            and c
            not in {
                "def_games_prior",
                "def_rest_days",
                "def_short_week_flag",
                "def_long_rest_flag",
            }
        )
    ]

    first_history_nonnull = int(
        features.loc[
            first,
            def_history_cols,
        ]
        .notna()
        .any(axis=1)
        .sum()
    )

    add(
        "first_game_def_history_nonnull_rows",
        first_history_nonnull,
        0,
        "PASS" if first_history_nonnull == 0 else "FAIL",
    )

    # Current game target must not equal any explicitly named *_last
    # field by construction on first games. This is an additional
    # structural check that first-game lag features are null.
    last_cols = [
        c
        for c in def_history_cols
        if c.endswith("_last")
    ]

    first_last_nonnull = int(
        features.loc[first, last_cols]
        .notna()
        .any(axis=1)
        .sum()
    )

    add(
        "first_game_last_feature_nonnull_rows",
        first_last_nonnull,
        0,
        "PASS" if first_last_nonnull == 0 else "FAIL",
    )

    # Opponent offensive history should also be blank on its first game.
    opp_first = features["opp_off_games_prior"].eq(0)

    opp_hist_cols = [
        c
        for c in features.columns
        if (
            c.startswith("opp_")
            and c
            not in {
                "opp_off_games_prior",
                "opp_rest_days",
                "opp_short_week_flag",
                "opp_long_rest_flag",
            }
        )
    ]

    opp_first_nonnull = int(
        features.loc[
            opp_first,
            opp_hist_cols,
        ]
        .notna()
        .any(axis=1)
        .sum()
    )

    add(
        "first_game_opp_history_nonnull_rows",
        opp_first_nonnull,
        0,
        "PASS" if opp_first_nonnull == 0 else "FAIL",
    )

    # No negative history counters.
    negative_history = int(
        (
            safe_numeric(features["def_games_prior"]).lt(0)
            | safe_numeric(features["opp_off_games_prior"]).lt(0)
        ).sum()
    )

    add(
        "negative_history_counter_rows",
        negative_history,
        0,
        "PASS" if negative_history == 0 else "FAIL",
    )

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    failures = summary[summary["status"].eq("FAIL")]

    if failures.empty:
        print()
        print("STRUCTURAL / LEAKAGE AUDIT: PASS")
    else:
        print()
        print("STRUCTURAL / LEAKAGE AUDIT: FAIL")

    return summary


def create_table(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {qident(OUTPUT_TABLE)}")

    column_defs = []

    integer_like = {
        "season",
        "week",
        "is_home",
        "is_playoffs",
        "def_games_prior",
        "opp_off_games_prior",
        "def_short_week_flag",
        "def_long_rest_flag",
        "opp_short_week_flag",
        "opp_long_rest_flag",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "safeties",
        "blocked_kicks",
        "return_tds",
    }

    text_like = {
        "game_id",
        "game_type",
        "game_date",
        "gametime",
        "game_datetime",
        "team",
        "opponent_team",
        "home_away",
        "feature_built_at",
    }

    for col in df.columns:
        if col in text_like:
            dtype = "TEXT"
        elif col in integer_like:
            dtype = "INTEGER"
        else:
            dtype = "REAL"

        column_defs.append(
            f"{qident(col)} {dtype}"
        )

    sql = f"""
        CREATE TABLE {qident(OUTPUT_TABLE)} (
            {", ".join(column_defs)},
            PRIMARY KEY (game_id, team)
        )
    """

    conn.execute(sql)

    conn.execute(
        f"""
        CREATE INDEX idx_dst_pregame_season_week
        ON {qident(OUTPUT_TABLE)} (season, week)
        """
    )

    conn.execute(
        f"""
        CREATE INDEX idx_dst_pregame_team
        ON {qident(OUTPUT_TABLE)} (team)
        """
    )


def insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    columns = list(df.columns)

    sql = f"""
        INSERT INTO {qident(OUTPUT_TABLE)}
        ({", ".join(qident(c) for c in columns)})
        VALUES ({", ".join(["?"] * len(columns))})
    """

    rows = []

    for row in df.itertuples(index=False, name=None):
        cleaned = []

        for value in row:
            if pd.isna(value):
                cleaned.append(None)
            elif isinstance(value, pd.Timestamp):
                cleaned.append(value.isoformat())
            elif hasattr(value, "item"):
                cleaned.append(value.item())
            else:
                cleaned.append(value)

        rows.append(tuple(cleaned))

    conn.executemany(sql, rows)


def export_outputs(
    conn: sqlite3.Connection,
    summary: pd.DataFrame,
) -> None:
    section("EXPORTING D/ST PREGAME FEATURES")

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    out = pd.read_sql_query(
        f"""
        SELECT *
        FROM {qident(OUTPUT_TABLE)}
        ORDER BY season, week, game_id, team
        """,
        conn,
    )

    out.to_csv(CSV_OUTPUT, index=False)
    out.to_parquet(PARQUET_OUTPUT, index=False)
    summary.to_csv(AUDIT_OUTPUT, index=False)

    print(f"Rows exported: {len(out)}")
    print(f"Columns exported: {len(out.columns)}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Parquet: {PARQUET_OUTPUT}")
    print(f"Audit: {AUDIT_OUTPUT}")


def print_feature_inventory(df: pd.DataFrame) -> None:
    section("D/ST FEATURE INVENTORY")

    def_cols = [
        c for c in df.columns if c.startswith("def_")
    ]
    opp_cols = [
        c for c in df.columns if c.startswith("opp_")
    ]

    print(f"Defense pregame features: {len(def_cols)}")
    print(f"Opponent offense features: {len(opp_cols)}")
    print(f"Total columns: {len(df.columns)}")

    print()
    print("Defense features:")
    for col in def_cols:
        print(f"  {col}")

    print()
    print("Opponent features:")
    for col in opp_cols:
        print(f"  {col}")


def main() -> None:
    section("NFL D/ST PREGAME FEATURE BUILD")

    print(f"Database: {DATABASE_PATH}")
    print(f"Seasons: {HISTORICAL_SEASONS}")
    print(f"Frozen scoring source: {SOURCE_SCORE_TABLE}")
    print("Current-game D/ST outcome used only as target.")
    print("Rolling features: SHIFT(1) BEFORE ROLLING.")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    with sqlite3.connect(DATABASE_PATH) as conn:
        scoring = load_dst_scoring(conn)
        games = load_games(conn)
        team_stats = load_team_stats(conn)

        section("ATTACHING GAME ORDER")

        base = attach_game_datetime(
            scoring,
            games,
        )

        print(f"Rows with game datetime: {len(base)}")

        section("BUILDING DEFENSE PRIOR-GAME FEATURES")

        base = add_team_rest_features(base)
        base = add_defense_rolling_features(base)

        print("Defense rolling features built.")

        section("BUILDING OPPONENT OFFENSE PRIOR-GAME FEATURES")

        game_lookup = (
            base[["game_id", "game_datetime"]]
            .drop_duplicates("game_id")
        )

        opp_features = build_opponent_offense_features(
            team_stats,
            game_lookup,
        )

        base = attach_opponent_features(
            base,
            opp_features,
        )

        base = add_opponent_rest(base)
        base = add_context_features(base)

        print("Opponent rolling features built.")

        features = select_output_columns(base)

        summary = leakage_audit(
            scoring,
            features,
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
                AUDIT_OUTPUT,
                index=False,
            )

            raise RuntimeError(
                "D/ST pregame feature audit failed. "
                "Database write aborted."
            )

        section("WRITING D/ST PREGAME FEATURE TABLE")

        create_table(conn, features)
        insert_dataframe(conn, features)
        conn.commit()

        stored = conn.execute(
            f"SELECT COUNT(*) FROM {qident(OUTPUT_TABLE)}"
        ).fetchone()[0]

        print(f"Stored feature rows: {stored}")

        if stored != len(features):
            raise RuntimeError(
                f"Stored row mismatch: {stored} != {len(features)}"
            )

        export_outputs(
            conn,
            summary,
        )

        print_feature_inventory(features)

    section("D/ST PREGAME FEATURE BUILD COMPLETE")

    print("Leakage-safe D/ST historical feature layer created.")
    print("Frozen D/ST history/scoring tables were not modified.")
    print(
        "Next layer after audit review: "
        "D/ST feature research / pruning."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 88)
        print("D/ST PREGAME FEATURE BUILD FAILED")
        print("=" * 88)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
