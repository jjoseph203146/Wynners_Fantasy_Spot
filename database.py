import sqlite3
from contextlib import contextmanager

from config import DATABASE_PATH


# =========================================================
# CONNECTION
# =========================================================

@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_database():

    with get_connection() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS games (

                game_id TEXT PRIMARY KEY,

                season INTEGER NOT NULL,
                game_type TEXT,
                week INTEGER,

                game_date TEXT,
                weekday TEXT,
                gametime TEXT,

                away_team TEXT,
                home_team TEXT,

                away_score INTEGER,
                home_score INTEGER,

                result INTEGER,
                total INTEGER,
                overtime INTEGER,

                old_game_id TEXT,

                gsis INTEGER,
                nfl_detail_id TEXT,
                pfr TEXT,
                pff TEXT,
                espn TEXT,

                away_rest INTEGER,
                home_rest INTEGER,

                away_moneyline REAL,
                home_moneyline REAL,

                spread_line REAL,
                away_spread_odds REAL,
                home_spread_odds REAL,

                total_line REAL,
                under_odds REAL,
                over_odds REAL,

                div_game INTEGER,

                roof TEXT,
                surface TEXT,
                temp REAL,
                wind REAL,

                away_qb_id TEXT,
                home_qb_id TEXT,

                away_qb_name TEXT,
                home_qb_name TEXT,

                away_coach TEXT,
                home_coach TEXT,

                referee TEXT,
                stadium_id TEXT,
                stadium TEXT,

                completed INTEGER DEFAULT 0,

                updated_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_season
            ON games(season)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_week
            ON games(season, week)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_date
            ON games(game_date)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_away
            ON games(away_team)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_home
            ON games(home_team)
            """
        )


if __name__ == "__main__":
    initialize_database()

    print(f"Database initialized:")
    print(DATABASE_PATH)
