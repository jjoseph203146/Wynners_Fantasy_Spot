#!/usr/bin/env python3

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data" / "nfl.db"

COACH_CSV = (
    ROOT
    / "processed"
    / "forecast_v1_coach_identity.csv"
)

OUT_CSV = (
    ROOT
    / "processed"
    / "forecast_v1_coaching_intelligence.csv"
)

OUT_AUDIT = (
    ROOT
    / "processed"
    / "forecast_v1_coaching_intelligence_audit.json"
)


# ============================================================
# APPROVED V1 FEATURE BOUNDARY
#
# These fields are already pregame / lagged in
# team_pregame_environment.
#
# Do NOT add player_pregame_features-derived concentration
# metrics here. Its target-game player population exactly
# overlaps realized player_weekly_usage.
#
# Do NOT fabricate play-level coaching metrics.
# ============================================================

TEAM_FEATURES = [
    "points_for_last",
    "points_for_avg_3",
    "points_for_avg_5",
    "points_against_last",
    "points_against_avg_3",
    "points_against_avg_5",
    "offensive_plays_avg_3",
    "offensive_plays_avg_5",
    "pass_attempts_avg_3",
    "pass_attempts_avg_5",
    "rush_attempts_avg_3",
    "rush_attempts_avg_5",
    "pass_rate_avg_3",
    "pass_rate_avg_5",
    "rush_rate_avg_3",
    "rush_rate_avg_5",
    "passing_yards_avg_3",
    "rushing_yards_avg_3",
    "passing_tds_avg_3",
    "rushing_tds_avg_3",
    "pace_trend",
    "pass_rate_trend",
]


COACH_FEATURES = [
    "coach_change_flag",
    "coach_team_era_start_flag",
    "coach_team_era_games_prior",
    "coach_career_games_prior",
    "coach_career_wins_prior",
    "coach_career_win_pct_prior",
    "coach_career_points_for_avg_prior",
    "coach_career_points_against_avg_prior",
    "coach_team_wins_prior",
    "coach_team_win_pct_prior",
    "coach_team_points_for_avg_prior",
    "coach_team_points_against_avg_prior",
    "coach_cold_start_flag",
    "coach_team_cold_start_flag",
    "coach_alias_applied_flag",
]


IDENTITY_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "game_date",
    "gametime",
    "team",
    "opponent_team",
    "home_flag",
    "coach_raw",
    "coach_identity",
    "previous_coach_identity",
    "coach_history_tier",
    "coach_team_history_tier",
]


# Explicit concepts forbidden from V1.
FORBIDDEN_COLUMN_TERMS = [
    "proe",
    "neutral_pass",
    "early_down",
    "seconds_per_play",
    "fourth_down",
    "play_action",
    "rpo",
    "motion",
    "personnel",
    "blitz",
    "coverage",
    "red_zone",
    "redzone",
    "target_concentration",
    "rb_committee",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def fail(message):
    raise SystemExit(f"FAIL | {message}")


def main():
    print("=" * 68)
    print(
        "WFS FORECAST CENTER — "
        "COACHING INTELLIGENCE V1 HISTORICAL FOUNDATION"
    )
    print("=" * 68)

    # --------------------------------------------------------
    # Fail-safe inputs / outputs
    # --------------------------------------------------------

    if not DB.exists():
        fail(f"missing database: {DB}")

    if not COACH_CSV.exists():
        fail(
            "missing frozen Coach Identity working artifact: "
            f"{COACH_CSV}"
        )

    if OUT_CSV.exists():
        fail(f"output already exists: {OUT_CSV}")

    if OUT_AUDIT.exists():
        fail(f"audit already exists: {OUT_AUDIT}")

    # --------------------------------------------------------
    # Coach Identity foundation
    # --------------------------------------------------------

    coach = pd.read_csv(COACH_CSV)

    required_coach = (
        IDENTITY_COLUMNS
        + COACH_FEATURES
    )

    missing_coach = [
        c
        for c in required_coach
        if c not in coach.columns
    ]

    if missing_coach:
        fail(
            "Coach Identity missing columns: "
            f"{missing_coach}"
        )

    if len(coach) != 2254:
        fail(
            "expected 2,254 Coach Identity rows, "
            f"got {len(coach):,}"
        )

    coach_dupes = int(
        coach.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    if coach_dupes:
        fail(
            f"duplicate Coach Identity keys: {coach_dupes}"
        )

    print("PASS | Coach Identity V1 loaded")
    print("PASS | 2,254 Coach Identity rows")
    print("PASS | Coach Identity keys unique")

    # --------------------------------------------------------
    # Read team pregame data from SQLite READ ONLY
    # --------------------------------------------------------

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    try:
        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            fail(
                f"SQLite integrity: {integrity}"
            )

        print("PASS | nfl.db opened read-only")
        print("PASS | SQLite integrity")

        tpe = pd.read_sql_query(
            """
            SELECT *
            FROM team_pregame_environment
            """,
            con,
        )

    finally:
        con.close()

    print(
        "Team pregame rows loaded: "
        f"{len(tpe):,}"
    )

    if len(tpe) != 1710:
        fail(
            "expected 1,710 team pregame rows, "
            f"got {len(tpe):,}"
        )

    # --------------------------------------------------------
    # Required source columns
    # --------------------------------------------------------

    required_tpe = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent_team",
    ] + TEAM_FEATURES

    missing_tpe = [
        c
        for c in required_tpe
        if c not in tpe.columns
    ]

    if missing_tpe:
        fail(
            "team_pregame_environment missing columns: "
            f"{missing_tpe}"
        )

    tpe_dupes = int(
        tpe.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    if tpe_dupes:
        fail(
            f"duplicate team pregame keys: {tpe_dupes}"
        )

    print("PASS | required lagged team features present")
    print("PASS | team pregame keys unique")

    # --------------------------------------------------------
    # Historical boundary
    # --------------------------------------------------------

    seasons = sorted(
        pd.to_numeric(
            tpe["season"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        "Team pregame seasons: "
        + ", ".join(map(str, seasons))
    )

    if seasons != [2023, 2024, 2025]:
        fail(
            "unexpected team pregame seasons: "
            f"{seasons}"
        )

    if (
        pd.to_numeric(
            tpe["season"],
            errors="raise",
        )
        .astype(int)
        .ge(2026)
        .any()
    ):
        fail(
            "2026/future rows present in historical foundation"
        )

    print("PASS | historical boundary = 2023-2025 only")

    # --------------------------------------------------------
    # Exact key coverage
    #
    # TPE is authoritative for the historical row population.
    # Every TPE row must find exactly one Coach Identity row.
    # --------------------------------------------------------

    coach_join = coach[
        required_coach
    ].copy()

    joined = tpe[
        required_tpe
    ].merge(
        coach_join,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_tpe", ""),
    )

    matched = int(
        joined["_merge"].eq("both").sum()
    )

    unmatched = int(
        joined["_merge"].ne("both").sum()
    )

    print()
    print("=== EXACT JOIN COVERAGE ===")
    print(f"Matched rows:   {matched:,}")
    print(f"Unmatched rows: {unmatched:,}")

    if matched != 1710 or unmatched != 0:
        fail(
            "historical Coach Identity join coverage"
        )

    joined = joined.drop(
        columns=["_merge"]
    )

    print("PASS | exact game_id + team join")

    # --------------------------------------------------------
    # Cross-source identity consistency.
    #
    # season/week/opponent exist in both sources. They must
    # agree exactly.
    # --------------------------------------------------------

    consistency_pairs = [
        ("season_tpe", "season", "season"),
        ("week_tpe", "week", "week"),
        (
            "opponent_team_tpe",
            "opponent_team",
            "opponent_team",
        ),
    ]

    print()
    print("=== CROSS-SOURCE CONSISTENCY ===")

    for left, right, label in consistency_pairs:
        if left not in joined.columns:
            fail(
                f"expected merged source column missing: {left}"
            )

        if right not in joined.columns:
            fail(
                f"expected coach column missing: {right}"
            )

        a = (
            joined[left]
            .astype(str)
            .str.strip()
        )

        b = (
            joined[right]
            .astype(str)
            .str.strip()
        )

        bad = int(
            a.ne(b).sum()
        )

        print(
            f"{label:<16} mismatches: {bad}"
        )

        if bad:
            fail(
                f"cross-source {label} mismatch"
            )

    print("PASS | season/week/opponent consistency")

    # --------------------------------------------------------
    # Construct clean output.
    #
    # Keep Coach Identity metadata and approved coaching
    # features. Use TPE team features as-is.
    # --------------------------------------------------------

    output_columns = (
        IDENTITY_COLUMNS
        + COACH_FEATURES
        + TEAM_FEATURES
    )

    out = joined[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    print()
    print("=== STRUCTURAL AUDIT ===")

    output_dupes = int(
        out.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    bad_game_counts = int(
        (
            out.groupby("game_id").size()
            != 2
        ).sum()
    )

    unique_games = int(
        out["game_id"].nunique()
    )

    unique_teams = int(
        out["team"].nunique()
    )

    print(f"Rows:                    {len(out):,}")
    print(f"Games:                   {unique_games:,}")
    print(f"Teams:                   {unique_teams:,}")
    print(f"Duplicate keys:          {output_dupes}")
    print(f"Bad two-team games:      {bad_game_counts}")

    if len(out) != 1710:
        fail(
            f"expected 1,710 output rows, got {len(out):,}"
        )

    if unique_games != 855:
        fail(
            f"expected 855 games, got {unique_games:,}"
        )

    if unique_teams != 32:
        fail(
            f"expected 32 teams, got {unique_teams}"
        )

    if output_dupes:
        fail(
            f"duplicate output keys: {output_dupes}"
        )

    if bad_game_counts:
        fail(
            f"games without exactly two teams: {bad_game_counts}"
        )

    print("PASS | historical structure")

    # --------------------------------------------------------
    # Approved-feature audit
    # --------------------------------------------------------

    print()
    print("=== FEATURE BOUNDARY AUDIT ===")

    player_population_terms = [
        "target_concentration",
        "rb_committee",
        "player_count",
        "player_population",
    ]

    player_population_columns = [
        c
        for c in out.columns
        if any(
            term in c.lower()
            for term in player_population_terms
        )
    ]

    if player_population_columns:
        fail(
            "player-population features present: "
            f"{player_population_columns}"
        )

    print(
        "PASS | no realized-population concentration features"
    )

    forbidden = [
        c
        for c in out.columns
        if any(
            term in c.lower()
            for term in FORBIDDEN_COLUMN_TERMS
        )
    ]

    if forbidden:
        fail(
            "unsupported play-level fields present: "
            f"{forbidden}"
        )

    print("PASS | no unsupported play-level features")

    # --------------------------------------------------------
    # Numeric feature health
    # --------------------------------------------------------

    numeric_features = (
        COACH_FEATURES
        + TEAM_FEATURES
    )

    non_numeric = []

    infinity_columns = []

    for col in numeric_features:
        converted = pd.to_numeric(
            out[col],
            errors="coerce",
        )

        # Values that were present but could not be converted.
        invalid = (
            out[col].notna()
            & converted.isna()
        )

        if invalid.any():
            non_numeric.append(col)

        finite = converted.dropna()

        if len(finite):
            arr = finite.to_numpy(
                dtype=float
            )

            if not pd.Series(arr).map(
                lambda x: x != float("inf")
                and x != float("-inf")
            ).all():
                infinity_columns.append(col)

    if non_numeric:
        fail(
            "non-numeric values in numeric features: "
            f"{non_numeric}"
        )

    if infinity_columns:
        fail(
            "infinite numeric values: "
            f"{infinity_columns}"
        )

    print("PASS | approved model features numeric")
    print("PASS | no infinities")

    # --------------------------------------------------------
    # Missingness report.
    #
    # Missing lagged history is allowed. It must remain
    # visible rather than being silently filled here.
    # --------------------------------------------------------

    print()
    print("=== FEATURE MISSINGNESS ===")

    missingness = {}

    for col in numeric_features:
        n = int(
            out[col].isna().sum()
        )

        pct = (
            100.0 * n / len(out)
            if len(out)
            else 0.0
        )

        missingness[col] = {
            "missing": n,
            "pct": pct,
        }

        print(
            f"{col:<38} | "
            f"{n:>4} | "
            f"{pct:6.2f}%"
        )

    # --------------------------------------------------------
    # Coach cold-start / transition diagnostics
    # --------------------------------------------------------

    print()
    print("=== COACH CONTEXT ===")

    coach_changes = int(
        out["coach_change_flag"].sum()
    )

    era_starts = int(
        out[
            "coach_team_era_start_flag"
        ].sum()
    )

    coach_cold = int(
        out[
            "coach_cold_start_flag"
        ].sum()
    )

    team_cold = int(
        out[
            "coach_team_cold_start_flag"
        ].sum()
    )

    print(
        f"Coach-change rows:       {coach_changes:,}"
    )
    print(
        f"Coach-team era starts:   {era_starts:,}"
    )
    print(
        f"Coach cold-start rows:   {coach_cold:,}"
    )
    print(
        f"Team-era cold rows:      {team_cold:,}"
    )

    # --------------------------------------------------------
    # No future rows
    # --------------------------------------------------------

    output_seasons = sorted(
        pd.to_numeric(
            out["season"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if output_seasons != [2023, 2024, 2025]:
        fail(
            "output season boundary changed"
        )

    print("PASS | no 2026 rows")

    # --------------------------------------------------------
    # Output ordering
    # --------------------------------------------------------

    out = out.sort_values(
        [
            "game_date",
            "gametime",
            "game_id",
            "home_flag",
            "team",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Write NEW artifacts only.
    # --------------------------------------------------------

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_CSV,
        index=False,
    )

    audit = {
        "artifact":
            "forecast_v1_coaching_intelligence",

        "status":
            "BUILD_COMPLETE_NOT_FROZEN",

        "scope":
            "HISTORICAL_TRAINING_FOUNDATION_ONLY",

        "database":
            str(DB),

        "database_mode":
            "READ_ONLY",

        "coach_identity_source":
            str(COACH_CSV),

        "team_source":
            "team_pregame_environment",

        "source_coach_rows":
            int(len(coach)),

        "source_team_pregame_rows":
            int(len(tpe)),

        "output_rows":
            int(len(out)),

        "unique_games":
            unique_games,

        "unique_teams":
            unique_teams,

        "seasons":
            output_seasons,

        "duplicate_game_team_keys":
            output_dupes,

        "games_not_exactly_two_rows":
            bad_game_counts,

        "join_matched":
            matched,

        "join_unmatched":
            unmatched,

        "coach_change_rows":
            coach_changes,

        "coach_team_era_start_rows":
            era_starts,

        "coach_cold_start_rows":
            coach_cold,

        "coach_team_cold_start_rows":
            team_cold,

        "team_features":
            TEAM_FEATURES,

        "coach_features":
            COACH_FEATURES,

        "player_population_features_included":
            False,

        "unsupported_play_level_features_included":
            False,

        "actual_play_caller_included":
            False,

        "true_proe_included":
            False,

        "fuzzy_matching":
            False,

        "sqlite_writes":
            False,

        "missingness":
            missingness,

        "notes": [
            (
                "player_pregame_features excluded from "
                "population-based concentration features "
                "because its game/team/player population "
                "exactly overlaps realized "
                "player_weekly_usage"
            ),
            (
                "2026 is excluded because "
                "team_pregame_environment currently has "
                "historical 2023-2025 coverage only"
            ),
            (
                "play-level coaching concepts require a "
                "future legitimate play-by-play source"
            ),
        ],
    }

    with OUT_AUDIT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    csv_hash = sha256_file(OUT_CSV)
    audit_hash = sha256_file(OUT_AUDIT)

    print()
    print("=== OUTPUT ARTIFACTS ===")
    print(
        OUT_CSV.relative_to(ROOT)
    )
    print(
        OUT_AUDIT.relative_to(ROOT)
    )

    print()
    print("=== SHA256 ===")
    print(
        f"{csv_hash}  "
        f"{OUT_CSV.relative_to(ROOT)}"
    )
    print(
        f"{audit_hash}  "
        f"{OUT_AUDIT.relative_to(ROOT)}"
    )

    print()
    print("=" * 68)
    print(
        "PASS | COACHING INTELLIGENCE V1 "
        "HISTORICAL FOUNDATION BUILT"
    )
    print("STATUS | NOT YET FROZEN")
    print("=" * 68)


if __name__ == "__main__":
    main()
