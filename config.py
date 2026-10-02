from pathlib import Path
from datetime import date


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"
CSV_DIR = DATA_DIR / "csv"
LOG_DIR = BASE_DIR / "logs"

DATABASE_PATH = DATA_DIR / "nfl.db"


# ---------------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ---------------------------------------------------------

for directory in [
    DATA_DIR,
    PARQUET_DIR,
    CSV_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# NFL SEASON
# ---------------------------------------------------------

def get_target_nfl_season() -> int:
    """
    Determine the NFL season we should currently track.

    March through December:
        use current calendar year

    January and February:
        games normally belong to previous NFL season
    """

    today = date.today()

    if today.month >= 3:
        return today.year

    return today.year - 1


CURRENT_SEASON = get_target_nfl_season()


# ---------------------------------------------------------
# EXPORT PATHS
# ---------------------------------------------------------

SCHEDULE_PARQUET = PARQUET_DIR / "nfl_schedule.parquet"
SCHEDULE_CSV = CSV_DIR / "nfl_schedule.csv"
