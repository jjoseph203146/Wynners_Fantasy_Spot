#!/usr/bin/env python3

"""
dst_current_features.py

Build leakage-safe current/upcoming NFL D/ST features for scheduled 2026 games.

Historical inputs (read-only)
-----------------------------
team_defense_fanduel_scoring
team_game_stats
games

Output
------
current_dst_features

Design
------
- One row per upcoming team-game.
- Uses only COMPLETED historical games prior to the upcoming game.
- Reconstructs the exact 20 promoted D/ST benchmark features.
- No current-game outcome fields are used.
- Does not modify frozen historical D/ST tables.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from config import DATABASE_PATH, CSV_DIR, PARQUET_DIR


OUTPUT_TABLE = "current_dst_features"
CSV_OUTPUT = Path(CSV_DIR) / "nfl_current_dst_features.csv"
PARQUET_OUTPUT = Path(PARQUET_DIR) / "nfl_current_dst_features.parquet"
AUDIT_OUTPUT = Path(CSV_DIR) / "audit_current_dst_features_summary.csv"

CURRENT_SEASON = 2026

PROMOTED_FEATURES = [
    "opp_sacks_allowed_avg_5",
    "opp_rushing_yards_avg_5",
    "opp_rush_attempts_avg_5",
    "opp_rushing_tds_avg_5",
    "opp_rushing_yards_last",
    "opp_passing_tds_avg_5",
    "opp_passing_tds_avg_3",
    "def_fd_avg_5",
    "def_pa_avg_5",
    "def_fd_avg_3",
    "opp_sacks_allowed_last",
    "def_takeaways_avg_3",
    "opp_pass_attempts_avg_5",
    "def_takeaways_avg_5",
    "opp_rush_attempts_last",
    "def_double_digit_games_5",
    "is_home",
    "def_ints_avg_5",
    "def_fum_rec_avg_3",
    "opp_rushing_tds_last",
]


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(
        f"PRAGMA table_info({qident(table)})"
    ).fetchall()
    return [r[1] for r in rows]


def build_datetime(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["game_date"], errors="coerce")

    time_text = (
        df["gametime"]
        .fillna("00:00")
        .astype(str)
        .str.strip()
        .replace({"": "00:00", "nan": "00:00", "None": "00:00"})
    )

    dt = pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + time_text,
        errors="coerce",
    )

    return dt


def load_upcoming_games(conn: sqlite3.Connection) -> pd.DataFrame:
    games = pd.read_sql_query(
        """
        SELECT *
        FROM games
        WHERE season = ?
        ORDER BY week, game_date, gametime, game_id
        """,
        conn,
        params=[CURRENT_SEASON],
    )

    if games.empty:
        raise RuntimeError(f"No {CURRENT_SEASON} games found in games table.")

    games["game_datetime"] = build_datetime(games)

    # Upcoming = no completed final score. The schedule engine has a completed
    # flag, but score-null logic is retained as a defensive fallback.
    if "completed" in games.columns:
        completed = pd.to_numeric(
            games["completed"], errors="coerce"
        ).fillna(0).astype(int)
        upcoming = games[completed.eq(0)].copy()
    else:
        upcoming = games[
            games["home_score"].isna()
            | games["away_score"].isna()
        ].copy()

    if upcoming.empty:
        raise RuntimeError(
            f"No upcoming {CURRENT_SEASON} games found."
        )

    return upcoming


def expand_team_games(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "game_type": games["game_type"],
            "game_date": games["game_date"],
            "gametime": games["gametime"],
            "game_datetime": games["game_datetime"],
            "team": games["home_team"],
            "opponent_team": games["away_team"],
            "is_home": 1,
        }
    )

    away = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "week": games["week"],
            "game_type": games["game_type"],
            "game_date": games["game_date"],
            "gametime": games["gametime"],
            "game_datetime": games["game_datetime"],
            "team": games["away_team"],
            "opponent_team": games["home_team"],
            "is_home": 0,
        }
    )

    out = pd.concat([home, away], ignore_index=True)

    out = out.dropna(
        subset=["game_id", "team", "opponent_team", "game_datetime"]
    ).copy()

    return out.sort_values(
        ["game_datetime", "game_id", "team"]
    ).reset_index(drop=True)


CURRENT_DST_ACTUALS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "parquet"
    / "nfl_current_dst_postgame_actuals.parquet"
)


def load_defense_history(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = set(table_columns(conn, "team_defense_fanduel_scoring"))

    required = {
        "game_id",
        "team",
        "fanduel_dst_points",
        "fanduel_points_allowed",
        "interceptions",
        "fumble_recoveries",
    }

    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(
            "team_defense_fanduel_scoring missing columns: "
            + ", ".join(missing)
        )

    df = pd.read_sql_query(
        """
        SELECT
            s.game_id,
            s.team,
            s.fanduel_dst_points,
            s.fanduel_points_allowed,
            s.interceptions,
            s.fumble_recoveries,
            g.season,
            g.week,
            g.game_date,
            g.gametime
        FROM team_defense_fanduel_scoring s
        INNER JOIN games g
            ON g.game_id = s.game_id
        WHERE g.season IN (2023, 2024, 2025)
        """,
        conn,
    )

    df["game_datetime"] = build_datetime(df)

    if not CURRENT_DST_ACTUALS_PATH.is_file():
        raise RuntimeError(
            "Missing current-season D/ST actual authority: "
            f"{CURRENT_DST_ACTUALS_PATH}"
        )

    current = pd.read_parquet(CURRENT_DST_ACTUALS_PATH)

    required_current = {
        "game_id",
        "season",
        "week",
        "team",
        "fanduel_dst_points",
        "fanduel_points_allowed",
        "interceptions",
        "fumble_recoveries",
    }

    missing_current = sorted(
        required_current - set(current.columns)
    )

    if missing_current:
        raise RuntimeError(
            "Current-season D/ST actual authority missing columns: "
            + ", ".join(missing_current)
        )

    current = current[
        pd.to_numeric(
            current["season"], errors="coerce"
        ).eq(CURRENT_SEASON)
    ].copy()

    if current.empty:
        raise RuntimeError(
            "Current-season D/ST actual authority has no "
            f"{CURRENT_SEASON} rows."
        )

    current = current.merge(
        pd.read_sql_query(
            """
            SELECT
                game_id,
                game_date,
                gametime,
                completed AS schedule_completed
            FROM games
            WHERE season = ?
            """,
            conn,
            params=[CURRENT_SEASON],
        ),
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    if current["game_date"].isna().any():
        raise RuntimeError(
            "Current D/ST actual contains game_id absent "
            "from schedule authority."
        )

    completed = pd.to_numeric(
        current["schedule_completed"], errors="coerce"
    ).fillna(0).astype(int)

    if completed.ne(1).any():
        raise RuntimeError(
            "Current D/ST actual contains non-completed games."
        )

    current["game_datetime"] = build_datetime(current)

    current = current[
        [
            "game_id",
            "team",
            "fanduel_dst_points",
            "fanduel_points_allowed",
            "interceptions",
            "fumble_recoveries",
            "season",
            "week",
            "game_date",
            "gametime",
            "game_datetime",
        ]
    ].copy()

    combined = pd.concat(
        [df, current],
        ignore_index=True,
    )

    duplicates = int(
        combined.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"Combined D/ST history has {duplicates} "
            "duplicate game/team rows."
        )

    return combined.sort_values(
        ["team", "game_datetime", "game_id"]
    ).reset_index(drop=True)


def choose_col(columns: set[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def load_offense_history(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = set(table_columns(conn, "team_game_stats"))

    mappings = {
        "pass_attempts": [
            "attempts",
            "passing_attempts",
            "pass_attempts",
        ],
        "passing_tds": [
            "passing_tds",
            "passing_touchdowns",
        ],
        "rushing_yards": [
            "rushing_yards",
        ],
        "rush_attempts": [
            "carries",
            "rushing_attempts",
            "rush_attempts",
        ],
        "rushing_tds": [
            "rushing_tds",
            "rushing_touchdowns",
        ],
        "sacks_allowed": [
            "sacks_suffered",
            "sacks_allowed",
        ],
    }

    resolved = {}
    for logical, candidates in mappings.items():
        col = choose_col(cols, candidates)
        if col is None:
            raise RuntimeError(
                f"Could not resolve team_game_stats field for {logical}. "
                f"Tried: {candidates}"
            )
        resolved[logical] = col

    select_bits = [
        "t.game_id",
        "t.team",
    ]

    for logical, source in resolved.items():
        select_bits.append(
            f"t.{qident(source)} AS {qident(logical)}"
        )

    sql = f"""
        SELECT
            {", ".join(select_bits)},
            g.season,
            g.week,
            g.game_date,
            g.gametime
        FROM team_game_stats t
        INNER JOIN games g
            ON g.game_id = t.game_id
        WHERE g.season IN (2023, 2024, 2025, 2026)
          AND (
              g.season < 2026
              OR COALESCE(g.completed, 0) = 1
          )
    """

    df = pd.read_sql_query(sql, conn)
    df["game_datetime"] = build_datetime(df)

    for logical in resolved:
        df[logical] = pd.to_numeric(
            df[logical], errors="coerce"
        )

    return df.sort_values(
        ["team", "game_datetime", "game_id"]
    ).reset_index(drop=True)


def prior_rows(
    history: pd.DataFrame,
    team: str,
    before_dt: pd.Timestamp,
) -> pd.DataFrame:
    return history[
        history["team"].eq(team)
        & history["game_datetime"].notna()
        & history["game_datetime"].lt(before_dt)
    ].sort_values(
        ["game_datetime", "game_id"]
    )


def last_value(history: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(
        history[column], errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def avg_value(
    history: pd.DataFrame,
    column: str,
    window: int,
) -> float:
    values = pd.to_numeric(
        history[column], errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.tail(window).mean())


def build_current_features(
    upcoming: pd.DataFrame,
    defense_history: pd.DataFrame,
    offense_history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for game in upcoming.itertuples(index=False):
        game_dt = pd.Timestamp(game.game_datetime)

        d = prior_rows(
            defense_history,
            game.team,
            game_dt,
        )

        o = prior_rows(
            offense_history,
            game.opponent_team,
            game_dt,
        )

        fd_values = pd.to_numeric(
            d["fanduel_dst_points"],
            errors="coerce",
        ).dropna()

        row = {
            "game_id": game.game_id,
            "season": int(game.season),
            "week": int(game.week),
            "game_type": game.game_type,
            "game_date": game.game_date,
            "gametime": game.gametime,
            "game_datetime": game.game_datetime,
            "team": game.team,
            "opponent_team": game.opponent_team,
            "is_home": int(game.is_home),

            "opp_sacks_allowed_avg_5": avg_value(
                o, "sacks_allowed", 5
            ),
            "opp_rushing_yards_avg_5": avg_value(
                o, "rushing_yards", 5
            ),
            "opp_rush_attempts_avg_5": avg_value(
                o, "rush_attempts", 5
            ),
            "opp_rushing_tds_avg_5": avg_value(
                o, "rushing_tds", 5
            ),
            "opp_rushing_yards_last": last_value(
                o, "rushing_yards"
            ),
            "opp_passing_tds_avg_5": avg_value(
                o, "passing_tds", 5
            ),
            "opp_passing_tds_avg_3": avg_value(
                o, "passing_tds", 3
            ),
            "def_fd_avg_5": avg_value(
                d, "fanduel_dst_points", 5
            ),
            "def_pa_avg_5": avg_value(
                d, "fanduel_points_allowed", 5
            ),
            "def_fd_avg_3": avg_value(
                d, "fanduel_dst_points", 3
            ),
            "opp_sacks_allowed_last": last_value(
                o, "sacks_allowed"
            ),
            "def_takeaways_avg_3": (
                avg_value(d, "interceptions", 3)
                + avg_value(d, "fumble_recoveries", 3)
                if (
                    pd.notna(avg_value(d, "interceptions", 3))
                    and pd.notna(avg_value(d, "fumble_recoveries", 3))
                )
                else np.nan
            ),
            "opp_pass_attempts_avg_5": avg_value(
                o, "pass_attempts", 5
            ),
            "def_takeaways_avg_5": (
                avg_value(d, "interceptions", 5)
                + avg_value(d, "fumble_recoveries", 5)
                if (
                    pd.notna(avg_value(d, "interceptions", 5))
                    and pd.notna(avg_value(d, "fumble_recoveries", 5))
                )
                else np.nan
            ),
            "opp_rush_attempts_last": last_value(
                o, "rush_attempts"
            ),
            "def_double_digit_games_5": (
                float((fd_values.tail(5) >= 10).sum())
                if not fd_values.empty
                else np.nan
            ),
            "def_ints_avg_5": avg_value(
                d, "interceptions", 5
            ),
            "def_fum_rec_avg_3": avg_value(
                d, "fumble_recoveries", 3
            ),
            "opp_rushing_tds_last": last_value(
                o, "rushing_tds"
            ),

            "def_history_games": int(len(d)),
            "opp_off_history_games": int(len(o)),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def audit_features(
    features: pd.DataFrame,
    expected_rows: int,
) -> pd.DataFrame:
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
        "expected_upcoming_team_rows",
        expected_rows,
        "",
        "INFO",
    )

    add(
        "feature_rows",
        len(features),
        expected_rows,
        "PASS" if len(features) == expected_rows else "FAIL",
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

    missing_promoted = [
        c for c in PROMOTED_FEATURES
        if c not in features.columns
    ]
    add(
        "missing_promoted_feature_columns",
        len(missing_promoted),
        0,
        "PASS" if not missing_promoted else "FAIL",
    )

    current_outcome_columns = [
        c for c in [
            "fanduel_dst_points",
            "fanduel_points_allowed",
            "sacks",
            "interceptions",
            "fumble_recoveries",
            "return_tds",
        ]
        if c in features.columns
    ]

    add(
        "current_game_outcome_columns",
        len(current_outcome_columns),
        0,
        "PASS" if not current_outcome_columns else "FAIL",
    )

    add(
        "rows_with_def_history",
        int(features["def_history_games"].gt(0).sum()),
        "",
        "INFO",
    )

    add(
        "rows_with_opp_off_history",
        int(features["opp_off_history_games"].gt(0).sum()),
        "",
        "INFO",
    )

    complete = features[PROMOTED_FEATURES].notna().all(axis=1)

    add(
        "rows_complete_all_20_features",
        int(complete.sum()),
        "",
        "INFO",
    )

    add(
        "rows_missing_any_promoted_feature",
        int((~complete).sum()),
        "",
        "INFO",
    )

    return pd.DataFrame(rows)


def write_sqlite(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
) -> None:
    out = features.copy()

    if pd.api.types.is_datetime64_any_dtype(out["game_datetime"]):
        out["game_datetime"] = out["game_datetime"].astype(str)

    out.to_sql(
        OUTPUT_TABLE,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_game_team
        ON {OUTPUT_TABLE}(game_id, team)
        """
    )

    conn.commit()


def main() -> None:
    section("NFL CURRENT 2026 D/ST FEATURE BUILD")

    print(f"Database: {DATABASE_PATH}")
    print(f"Season: {CURRENT_SEASON}")
    print("Promoted benchmark features: 20")
    print("Historical D/ST tables: READ ONLY")
    print("Upcoming game outcomes: NOT used")

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as conn:
        games = load_upcoming_games(conn)
        upcoming = expand_team_games(games)

        section("UPCOMING 2026 TEAM-GAMES")
        print(f"Upcoming games: {len(games)}")
        print(f"Upcoming team-games: {len(upcoming)}")
        print(f"First week present: {upcoming['week'].min()}")
        print(f"Last week present: {upcoming['week'].max()}")

        defense_history = load_defense_history(conn)
        offense_history = load_offense_history(conn)

        section("HISTORICAL INPUTS")
        print(f"D/ST scoring history rows: {len(defense_history)}")
        print(f"Opponent offense history rows: {len(offense_history)}")

        features = build_current_features(
            upcoming,
            defense_history,
            offense_history,
        )

        section("CURRENT D/ST FEATURE AUDIT")
        summary = audit_features(
            features,
            expected_rows=len(upcoming),
        )
        print(summary.to_string(index=False))

        failures = summary[
            summary["status"].eq("FAIL")
        ]

        if not failures.empty:
            print()
            print("STRUCTURAL AUDIT: FAIL")
            raise RuntimeError(
                "Current D/ST feature structural audit failed. "
                "No output table written."
            )

        print()
        print("STRUCTURAL AUDIT: PASS")

        section("WRITING CURRENT D/ST FEATURES")
        write_sqlite(conn, features)

    features.to_csv(
        CSV_OUTPUT,
        index=False,
    )
    features.to_parquet(
        PARQUET_OUTPUT,
        index=False,
    )
    summary.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    section("CURRENT D/ST FEATURE INVENTORY")

    print(f"Stored rows: {len(features)}")
    print(f"Total columns: {len(features.columns)}")
    print()
    print("Promoted features:")
    for i, feature in enumerate(PROMOTED_FEATURES, start=1):
        non_null = int(features[feature].notna().sum())
        print(
            f"  {i:2d}. {feature:<32} "
            f"non-null={non_null}"
        )

    section("EXPORTS")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Parquet: {PARQUET_OUTPUT}")
    print(f"Audit: {AUDIT_OUTPUT}")

    section("CURRENT 2026 D/ST FEATURE BUILD COMPLETE")
    print(
        "Next layer after audit review: train the promoted ridge "
        "on 2023-2025 and score current 2026 D/ST rows."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 92)
        print("CURRENT D/ST FEATURE BUILD FAILED")
        print("=" * 92)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
