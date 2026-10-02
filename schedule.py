from datetime import datetime, timezone

import pandas as pd
import nflreadpy as nfl

from config import (
    CURRENT_SEASON,
    SCHEDULE_PARQUET,
    SCHEDULE_CSV,
)

from database import (
    get_connection,
    initialize_database,
)


EXPECTED_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
    "total",
    "overtime",
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
    "stadium_id",
    "stadium",
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df


def download_schedule(season: int = CURRENT_SEASON) -> pd.DataFrame:
    print("=" * 70)
    print(f"Downloading NFL schedule for {season}")
    print("=" * 70)

    schedule = nfl.load_schedules(season)

    df = schedule.to_pandas()

    if df.empty:
        raise RuntimeError(
            f"No NFL schedule data returned for season {season}"
        )

    df = ensure_columns(df)

    print(f"Downloaded {len(df)} games")

    return df


def safe_int(value):
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def safe_float(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_text(value):
    if pd.isna(value):
        return None

    return str(value)


def safe_bool_int(value):
    if pd.isna(value):
        return None

    return 1 if bool(value) else 0


def update_schedule_database(df: pd.DataFrame):
    initialize_database()

    updated_at = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO games (
            game_id,
            season,
            game_type,
            week,
            game_date,
            weekday,
            gametime,
            away_team,
            home_team,
            away_score,
            home_score,
            result,
            total,
            overtime,
            old_game_id,
            gsis,
            nfl_detail_id,
            pfr,
            pff,
            espn,
            away_rest,
            home_rest,
            away_moneyline,
            home_moneyline,
            spread_line,
            away_spread_odds,
            home_spread_odds,
            total_line,
            under_odds,
            over_odds,
            div_game,
            roof,
            surface,
            temp,
            wind,
            away_qb_id,
            home_qb_id,
            away_qb_name,
            home_qb_name,
            away_coach,
            home_coach,
            referee,
            stadium_id,
            stadium,
            completed,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(game_id)
        DO UPDATE SET
            season = excluded.season,
            game_type = excluded.game_type,
            week = excluded.week,
            game_date = excluded.game_date,
            weekday = excluded.weekday,
            gametime = excluded.gametime,
            away_team = excluded.away_team,
            home_team = excluded.home_team,
            away_score = excluded.away_score,
            home_score = excluded.home_score,
            result = excluded.result,
            total = excluded.total,
            overtime = excluded.overtime,
            old_game_id = excluded.old_game_id,
            gsis = excluded.gsis,
            nfl_detail_id = excluded.nfl_detail_id,
            pfr = excluded.pfr,
            pff = excluded.pff,
            espn = excluded.espn,
            away_rest = excluded.away_rest,
            home_rest = excluded.home_rest,
            away_moneyline = excluded.away_moneyline,
            home_moneyline = excluded.home_moneyline,
            spread_line = excluded.spread_line,
            away_spread_odds = excluded.away_spread_odds,
            home_spread_odds = excluded.home_spread_odds,
            total_line = excluded.total_line,
            under_odds = excluded.under_odds,
            over_odds = excluded.over_odds,
            div_game = excluded.div_game,
            roof = excluded.roof,
            surface = excluded.surface,
            temp = excluded.temp,
            wind = excluded.wind,
            away_qb_id = excluded.away_qb_id,
            home_qb_id = excluded.home_qb_id,
            away_qb_name = excluded.away_qb_name,
            home_qb_name = excluded.home_qb_name,
            away_coach = excluded.away_coach,
            home_coach = excluded.home_coach,
            referee = excluded.referee,
            stadium_id = excluded.stadium_id,
            stadium = excluded.stadium,
            completed = excluded.completed,
            updated_at = excluded.updated_at
    """

    rows = []

    for _, row in df.iterrows():
        away_score = safe_int(row["away_score"])
        home_score = safe_int(row["home_score"])

        completed = (
            1
            if away_score is not None
            and home_score is not None
            else 0
        )

        values = (
            safe_text(row["game_id"]),
            safe_int(row["season"]),
            safe_text(row["game_type"]),
            safe_int(row["week"]),
            safe_text(row["gameday"]),
            safe_text(row["weekday"]),
            safe_text(row["gametime"]),
            safe_text(row["away_team"]),
            safe_text(row["home_team"]),
            away_score,
            home_score,
            safe_int(row["result"]),
            safe_int(row["total"]),
            safe_int(row["overtime"]),
            safe_text(row["old_game_id"]),
            safe_int(row["gsis"]),
            safe_text(row["nfl_detail_id"]),
            safe_text(row["pfr"]),
            safe_text(row["pff"]),
            safe_text(row["espn"]),
            safe_int(row["away_rest"]),
            safe_int(row["home_rest"]),
            safe_float(row["away_moneyline"]),
            safe_float(row["home_moneyline"]),
            safe_float(row["spread_line"]),
            safe_float(row["away_spread_odds"]),
            safe_float(row["home_spread_odds"]),
            safe_float(row["total_line"]),
            safe_float(row["under_odds"]),
            safe_float(row["over_odds"]),
            safe_bool_int(row["div_game"]),
            safe_text(row["roof"]),
            safe_text(row["surface"]),
            safe_float(row["temp"]),
            safe_float(row["wind"]),
            safe_text(row["away_qb_id"]),
            safe_text(row["home_qb_id"]),
            safe_text(row["away_qb_name"]),
            safe_text(row["home_qb_name"]),
            safe_text(row["away_coach"]),
            safe_text(row["home_coach"]),
            safe_text(row["referee"]),
            safe_text(row["stadium_id"]),
            safe_text(row["stadium"]),
            completed,
            updated_at,
        )

        if len(values) != 46:
            raise RuntimeError(
                f"Expected 46 values, got {len(values)} "
                f"for game {row['game_id']}"
            )

        rows.append(values)

    with get_connection() as conn:
        conn.executemany(sql, rows)

    print(
        f"Inserted/updated {len(rows)} games "
        f"in the SQLite database."
    )


def export_schedule():
    query = """
        SELECT *
        FROM games
        ORDER BY
            season,
            week,
            game_date,
            gametime
    """

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)

    df.to_parquet(
        SCHEDULE_PARQUET,
        index=False
    )

    df.to_csv(
        SCHEDULE_CSV,
        index=False
    )

    print()
    print("Schedule exports created:")
    print(f"Parquet: {SCHEDULE_PARQUET}")
    print(f"CSV:     {SCHEDULE_CSV}")


def print_summary():
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                season,
                COUNT(*) AS games,
                SUM(completed) AS completed_games
            FROM games
            GROUP BY season
            ORDER BY season DESC
            """
        )

        rows = cursor.fetchall()

    print()
    print("=" * 70)
    print("NFL DATABASE SUMMARY")
    print("=" * 70)

    for row in rows:
        completed = row["completed_games"] or 0

        print(
            f"Season {row['season']}: "
            f"{row['games']} games | "
            f"{completed} completed"
        )


def run_schedule_update():
    df = download_schedule()

    update_schedule_database(df)

    export_schedule()

    print_summary()


if __name__ == "__main__":
    run_schedule_update()
