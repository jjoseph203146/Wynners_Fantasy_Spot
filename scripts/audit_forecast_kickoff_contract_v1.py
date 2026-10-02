from pathlib import Path
import sqlite3
from collections import Counter
from datetime import datetime

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

NFL_DB = (
    ROOT
    / "data"
    / "nfl.db"
)

LEDGER = (
    ROOT
    / "data"
    / "forecast_ledger.db"
)


def fail(message):
    raise SystemExit(
        f"FAIL | {message}"
    )


print("=" * 100)
print("WFS FORECAST CENTER — KICKOFF TIME CONTRACT AUDIT V1")
print("=" * 100)


# =====================================================================
# File existence
# =====================================================================

if not NFL_DB.is_file():
    fail("nfl.db missing")

if not LEDGER.is_file():
    fail("forecast_ledger.db missing")

print("PASS | required databases exist")


# =====================================================================
# Open both databases STRICTLY read-only
# =====================================================================

nfl = sqlite3.connect(
    f"file:{NFL_DB}?mode=ro",
    uri=True,
)

ledger = sqlite3.connect(
    f"file:{LEDGER}?mode=ro",
    uri=True,
)

try:

    print()
    print("=== DATABASE INTEGRITY ===")

    nfl_integrity = nfl.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    ledger_integrity = ledger.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(
        f"nfl.db            | {nfl_integrity}"
    )

    print(
        f"forecast_ledger.db| {ledger_integrity}"
    )

    if nfl_integrity != "ok":
        fail("nfl.db integrity")

    if ledger_integrity != "ok":
        fail("forecast ledger integrity")

    print("PASS | both databases opened read-only")


    # =================================================================
    # Exact 2026 timing population
    # =================================================================

    games = pd.read_sql_query(
        """
        SELECT
            game_id,
            season,
            week,
            game_date,
            weekday,
            gametime,
            away_team,
            home_team,
            completed,
            updated_at
        FROM games
        WHERE season = 2026
        ORDER BY
            week,
            game_date,
            gametime,
            game_id
        """,
        nfl,
    )

    print()
    print("=== 2026 GAME TIMING POPULATION ===")

    print(
        f"Rows              | {len(games)}"
    )

    print(
        "game_date non-null| "
        f"{int(games['game_date'].notna().sum())}"
    )

    print(
        "gametime non-null | "
        f"{int(games['gametime'].notna().sum())}"
    )

    print(
        "weekday non-null  | "
        f"{int(games['weekday'].notna().sum())}"
    )

    if len(games) != 272:
        fail("expected 272 2026 games")

    if games["game_id"].nunique() != 272:
        fail("2026 game_id uniqueness")

    if games["game_date"].isna().any():
        fail("missing 2026 game_date")

    if games["gametime"].isna().any():
        fail("missing 2026 gametime")

    print("PASS | 272/272 timing fields populated")


    # =================================================================
    # Raw date format analysis
    # =================================================================

    print()
    print("=== RAW GAME_DATE CONTRACT ===")

    date_lengths = Counter(
        games["game_date"]
        .astype(str)
        .map(len)
    )

    for length, count in sorted(
        date_lengths.items()
    ):
        print(
            f"Length {length:<2} | {count}"
        )

    parsed_dates = pd.to_datetime(
        games["game_date"],
        errors="coerce",
        format="%Y-%m-%d",
    )

    bad_dates = int(
        parsed_dates.isna().sum()
    )

    print(
        f"YYYY-MM-DD parse failures | {bad_dates}"
    )

    if bad_dates != 0:
        fail("game_date is not uniformly YYYY-MM-DD")

    print("PASS | game_date exact YYYY-MM-DD contract")


    # =================================================================
    # Raw gametime format analysis
    # =================================================================

    print()
    print("=== RAW GAMETIME CONTRACT ===")

    time_values = (
        games["gametime"]
        .astype(str)
        .tolist()
    )

    time_lengths = Counter(
        len(value)
        for value in time_values
    )

    for length, count in sorted(
        time_lengths.items()
    ):
        print(
            f"Length {length:<2} | {count}"
        )

    print()
    print("Unique gametime values:")

    time_counts = Counter(
        time_values
    )

    for value, count in sorted(
        time_counts.items()
    ):
        print(
            f"{value:<12} | {count}"
        )


    # Test likely formats without changing any data.

    formats = [
        "%H:%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I:%M%p",
    ]

    format_results = {}

    for fmt in formats:

        failures = 0

        for value in time_values:

            try:
                datetime.strptime(
                    value,
                    fmt,
                )

            except ValueError:
                failures += 1

        format_results[fmt] = failures

    print()
    print("Gametime parse candidates:")

    for fmt, failures in format_results.items():
        print(
            f"{fmt:<10} failures | {failures}"
        )

    exact_formats = [
        fmt
        for fmt, failures
        in format_results.items()
        if failures == 0
    ]

    if len(exact_formats) != 1:

        print()
        print(
            "HOLD | gametime format not uniquely proven"
        )

    else:

        print()
        print(
            "PASS | exact raw gametime format | "
            f"{exact_formats[0]}"
        )


    # =================================================================
    # Show first games exactly as stored
    # =================================================================

    print()
    print("=== FIRST 25 GAMES — RAW DATABASE VALUES ===")

    display_cols = [
        "game_id",
        "week",
        "game_date",
        "weekday",
        "gametime",
        "away_team",
        "home_team",
        "completed",
    ]

    print(
        games[
            display_cols
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


    # =================================================================
    # Week 1 / Week 2 exact schedule
    # =================================================================

    print()
    print("=== WEEK 1 EXACT TIMING ===")

    print(
        games.loc[
            games["week"] == 1,
            display_cols,
        ].to_string(
            index=False
        )
    )

    print()
    print("=== WEEK 2 EXACT TIMING ===")

    print(
        games.loc[
            games["week"] == 2,
            display_cols,
        ].to_string(
            index=False
        )
    )


    # =================================================================
    # Ledger timing equivalence
    # =================================================================

    print()
    print("=== LEDGER / NFL TIMING EQUIVALENCE ===")

    ledger_times = pd.read_sql_query(
        """
        SELECT
            game_id,
            game_date,
            gametime
        FROM forecast_predictions
        WHERE snapshot_id = (
            SELECT snapshot_id
            FROM forecast_snapshots
            ORDER BY captured_at_utc
            LIMIT 1
        )
        ORDER BY game_id
        """,
        ledger,
    )

    timing_join = ledger_times.merge(
        games[
            [
                "game_id",
                "game_date",
                "gametime",
            ]
        ],
        on="game_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=(
            "_ledger",
            "_nfl",
        ),
    )

    coverage_mismatch = int(
        (
            timing_join["_merge"]
            != "both"
        ).sum()
    )

    date_mismatch = int(
        (
            timing_join[
                "game_date_ledger"
            ].astype(str)
            !=
            timing_join[
                "game_date_nfl"
            ].astype(str)
        ).sum()
    )

    time_mismatch = int(
        (
            timing_join[
                "gametime_ledger"
            ].astype(str)
            !=
            timing_join[
                "gametime_nfl"
            ].astype(str)
        ).sum()
    )

    print(
        f"Coverage mismatches | {coverage_mismatch}"
    )

    print(
        f"Date mismatches     | {date_mismatch}"
    )

    print(
        f"Gametime mismatches | {time_mismatch}"
    )

    if coverage_mismatch != 0:
        fail("ledger/NFL timing coverage")

    if date_mismatch != 0:
        fail("ledger/NFL game_date mismatch")

    if time_mismatch != 0:
        fail("ledger/NFL gametime mismatch")

    print("PASS | baseline ledger preserved authoritative timing exactly")


    # =================================================================
    # Timezone evidence audit
    # =================================================================

    print()
    print("=== TIMEZONE EVIDENCE ===")

    print(
        "NOTE | game_date and gametime contain no timezone offset column."
    )

    print(
        "NOTE | raw clock-time format alone does NOT prove whether "
        "gametime is ET, UTC, venue-local, or another convention."
    )

    print(
        "NOTE | writer must not invent timezone semantics."
    )

    print(
        "NOTE | pre-kickoff enforcement will remain HOLD until "
        "timezone convention is proven from authoritative project data "
        "or existing ingestion semantics."
    )


finally:

    nfl.close()
    ledger.close()


# =====================================================================
# Final gate
# =====================================================================

print()
print("=" * 100)
print("KICKOFF CONTRACT AUDIT GATE")
print("=" * 100)

print("PASS | nfl.db READ-ONLY")
print("PASS | forecast_ledger.db READ-ONLY")
print("PASS | 272/272 GAME DATES")
print("PASS | 272/272 RAW GAMETIMES")
print("PASS | EXACT LEDGER/NFL TIMING EQUIVALENCE")
print()
print("HOLD | TIMEZONE SEMANTICS MUST BE PROVEN BEFORE")
print("       PROSPECTIVE PRE-KICKOFF WRITER IS ENABLED")
print()
print("PASS | NO DATABASE WRITES")
print("PASS | NO FORECAST WRITES")
print("PASS | OPTIMIZER / UI UNTOUCHED")
print("PASS | SITE RESTART NOT REQUIRED")
print("=" * 100)
