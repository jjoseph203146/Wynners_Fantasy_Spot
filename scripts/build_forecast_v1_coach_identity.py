#!/usr/bin/env python3

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")
DB = ROOT / "data" / "nfl.db"

OUT_CSV = (
    ROOT
    / "processed"
    / "forecast_v1_coach_identity.csv"
)

OUT_AUDIT = (
    ROOT
    / "processed"
    / "forecast_v1_coach_identity_audit.json"
)


# ------------------------------------------------------------
# EXPLICIT IDENTITY ALIASES ONLY.
#
# No fuzzy matching.
# Raw source value is preserved separately.
# ------------------------------------------------------------

COACH_ALIASES = {
    "Klint Kubliak": "Klint Kubiak",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def normalize_coach(raw):
    if raw is None:
        return None

    raw = str(raw).strip()

    if not raw:
        return None

    return COACH_ALIASES.get(raw, raw)


def make_kickoff_dt(game_date, gametime):
    """
    Deterministic chronology key.

    game_date is authoritative date.
    gametime is used when present.

    This value is used for ordering only.
    No future game result/stat is consumed.
    """

    date_text = str(game_date).strip()

    time_text = (
        str(gametime).strip()
        if gametime is not None
        else ""
    )

    if not time_text:
        time_text = "00:00"

    candidates = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]

    value = f"{date_text} {time_text}"

    for fmt in candidates:
        try:
            return datetime.strptime(
                value,
                fmt,
            )
        except ValueError:
            pass

    # Deterministic date-only fallback.
    return datetime.strptime(
        date_text,
        "%Y-%m-%d",
    )


def history_tier(prior_games):
    if prior_games == 0:
        return "COLD"

    if prior_games <= 2:
        return "LIMITED"

    if prior_games <= 4:
        return "PARTIAL"

    return "FULL"


def safe_mean(values):
    if not values:
        return None

    return sum(values) / len(values)


def main():
    print(
        "============================================================"
    )
    print(
        "WFS FORECAST CENTER — COACH IDENTITY V1"
    )
    print(
        "============================================================"
    )

    if not DB.exists():
        raise SystemExit(
            f"FAIL | database missing: {DB}"
        )

    if OUT_CSV.exists():
        raise SystemExit(
            f"FAIL | output already exists: {OUT_CSV}"
        )

    if OUT_AUDIT.exists():
        raise SystemExit(
            f"FAIL | audit already exists: {OUT_AUDIT}"
        )

    # --------------------------------------------------------
    # Read-only database connection.
    # --------------------------------------------------------

    con = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True,
    )

    con.row_factory = sqlite3.Row

    try:
        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise SystemExit(
                f"FAIL | SQLite integrity: {integrity}"
            )

        print("PASS | nfl.db opened read-only")
        print("PASS | SQLite integrity")

        games = con.execute(
            """
            SELECT
                game_id,
                season,
                week,
                game_type,
                game_date,
                gametime,
                home_team,
                away_team,
                home_coach,
                away_coach,
                home_score,
                away_score,
                completed
            FROM games
            ORDER BY
                game_date,
                gametime,
                game_id
            """
        ).fetchall()

    finally:
        con.close()

    print(f"Games loaded: {len(games):,}")

    if len(games) != 1127:
        raise SystemExit(
            "FAIL | expected 1,127 games"
        )

    # --------------------------------------------------------
    # Convert physical games to team-game rows.
    # --------------------------------------------------------

    rows = []

    for g in games:
        kickoff = make_kickoff_dt(
            g["game_date"],
            g["gametime"],
        )

        home_coach_raw = (
            str(g["home_coach"]).strip()
            if g["home_coach"] is not None
            else None
        )

        away_coach_raw = (
            str(g["away_coach"]).strip()
            if g["away_coach"] is not None
            else None
        )

        home_identity = normalize_coach(
            home_coach_raw
        )

        away_identity = normalize_coach(
            away_coach_raw
        )

        if not home_identity or not away_identity:
            raise SystemExit(
                f"FAIL | missing coach: {g['game_id']}"
            )

        rows.append(
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "week": g["week"],
                "game_type": g["game_type"],
                "game_date": g["game_date"],
                "gametime": g["gametime"],
                "kickoff_dt": kickoff,
                "team": g["home_team"],
                "opponent_team": g["away_team"],
                "home_flag": 1,
                "coach_raw": home_coach_raw,
                "coach_identity": home_identity,
                "completed": g["completed"],
                "points_for": g["home_score"],
                "points_against": g["away_score"],
            }
        )

        rows.append(
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "week": g["week"],
                "game_type": g["game_type"],
                "game_date": g["game_date"],
                "gametime": g["gametime"],
                "kickoff_dt": kickoff,
                "team": g["away_team"],
                "opponent_team": g["home_team"],
                "home_flag": 0,
                "coach_raw": away_coach_raw,
                "coach_identity": away_identity,
                "completed": g["completed"],
                "points_for": g["away_score"],
                "points_against": g["home_score"],
            }
        )

    if len(rows) != 2254:
        raise SystemExit(
            "FAIL | expected 2,254 team-game rows"
        )

    print(
        "PASS | 2,254 team-game identity rows"
    )

    # --------------------------------------------------------
    # Deterministic chronology.
    # --------------------------------------------------------

    rows.sort(
        key=lambda r: (
            r["kickoff_dt"],
            r["game_id"],
            r["team"],
        )
    )

    # --------------------------------------------------------
    # Historical state.
    #
    # coach_history:
    #     Exact normalized coach identity across all teams.
    #
    # coach_team_history:
    #     Exact normalized coach identity with current team.
    #
    # team_previous:
    #     Last chronological team game, used to detect coach
    #     changes / era starts.
    #
    # IMPORTANT:
    # Historical state is updated AFTER the target row is built.
    # Therefore target-game outcomes cannot enter its features.
    # --------------------------------------------------------

    coach_history = defaultdict(list)
    coach_team_history = defaultdict(list)
    team_previous = {}

    output = []

    alias_rows = 0
    leakage_violations = 0
    completed_history_updates = 0

    for r in rows:
        coach = r["coach_identity"]
        team = r["team"]

        coach_prior = coach_history[coach]
        coach_team_prior = coach_team_history[
            (coach, team)
        ]

        previous = team_previous.get(team)

        coach_change_flag = 0
        era_start_flag = 0

        previous_coach_identity = None

        if previous is None:
            era_start_flag = 1
        else:
            previous_coach_identity = (
                previous["coach_identity"]
            )

            if previous_coach_identity != coach:
                coach_change_flag = 1
                era_start_flag = 1

        # ----------------------------------------------------
        # Strict temporal validation.
        # ----------------------------------------------------

        for h in coach_prior:
            if h["kickoff_dt"] >= r["kickoff_dt"]:
                leakage_violations += 1

        for h in coach_team_prior:
            if h["kickoff_dt"] >= r["kickoff_dt"]:
                leakage_violations += 1

        coach_prior_games = len(coach_prior)
        coach_team_prior_games = len(
            coach_team_prior
        )

        coach_prior_wins = sum(
            h["win"]
            for h in coach_prior
        )

        coach_team_prior_wins = sum(
            h["win"]
            for h in coach_team_prior
        )

        coach_pf = [
            h["points_for"]
            for h in coach_prior
            if h["points_for"] is not None
        ]

        coach_pa = [
            h["points_against"]
            for h in coach_prior
            if h["points_against"] is not None
        ]

        coach_team_pf = [
            h["points_for"]
            for h in coach_team_prior
            if h["points_for"] is not None
        ]

        coach_team_pa = [
            h["points_against"]
            for h in coach_team_prior
            if h["points_against"] is not None
        ]

        if r["coach_raw"] != coach:
            alias_rows += 1

        out = {
            "game_id": r["game_id"],
            "season": r["season"],
            "week": r["week"],
            "game_type": r["game_type"],
            "game_date": r["game_date"],
            "gametime": r["gametime"],
            "team": team,
            "opponent_team": r["opponent_team"],
            "home_flag": r["home_flag"],

            "coach_raw": r["coach_raw"],
            "coach_identity": coach,

            "previous_coach_identity":
                previous_coach_identity,

            "coach_change_flag":
                coach_change_flag,

            "coach_team_era_start_flag":
                era_start_flag,

            # Number of completed prior games in THIS
            # exact coach-team era.
            "coach_team_era_games_prior":
                coach_team_prior_games,

            # Prior completed games for this coach,
            # including previous teams.
            "coach_career_games_prior":
                coach_prior_games,

            "coach_career_wins_prior":
                coach_prior_wins,

            "coach_career_win_pct_prior":
                (
                    coach_prior_wins
                    / coach_prior_games
                    if coach_prior_games
                    else None
                ),

            "coach_career_points_for_avg_prior":
                safe_mean(coach_pf),

            "coach_career_points_against_avg_prior":
                safe_mean(coach_pa),

            "coach_team_wins_prior":
                coach_team_prior_wins,

            "coach_team_win_pct_prior":
                (
                    coach_team_prior_wins
                    / coach_team_prior_games
                    if coach_team_prior_games
                    else None
                ),

            "coach_team_points_for_avg_prior":
                safe_mean(coach_team_pf),

            "coach_team_points_against_avg_prior":
                safe_mean(coach_team_pa),

            "coach_history_tier":
                history_tier(
                    coach_prior_games
                ),

            "coach_team_history_tier":
                history_tier(
                    coach_team_prior_games
                ),

            "coach_cold_start_flag":
                int(coach_prior_games == 0),

            "coach_team_cold_start_flag":
                int(coach_team_prior_games == 0),

            "coach_alias_applied_flag":
                int(r["coach_raw"] != coach),
        }

        output.append(out)

        # ----------------------------------------------------
        # UPDATE HISTORICAL STATE ONLY AFTER FEATURE CREATION.
        #
        # Only completed games with real scores are allowed
        # into future historical aggregates.
        # ----------------------------------------------------

        completed = int(r["completed"] or 0)

        if (
            completed == 1
            and r["points_for"] is not None
            and r["points_against"] is not None
        ):
            pf = float(r["points_for"])
            pa = float(r["points_against"])

            hist = {
                "kickoff_dt": r["kickoff_dt"],
                "game_id": r["game_id"],
                "team": team,
                "points_for": pf,
                "points_against": pa,
                "win": int(pf > pa),
            }

            coach_history[coach].append(hist)
            coach_team_history[
                (coach, team)
            ].append(hist)

            completed_history_updates += 1

        # Team previous identity tracks scheduled chronology,
        # not results. Current coach identity itself is
        # pregame-known information.
        team_previous[team] = {
            "kickoff_dt": r["kickoff_dt"],
            "game_id": r["game_id"],
            "coach_identity": coach,
        }

    if leakage_violations != 0:
        raise SystemExit(
            "FAIL | temporal leakage violations: "
            f"{leakage_violations}"
        )

    df = pd.DataFrame(output)

    # --------------------------------------------------------
    # Structural audits.
    # --------------------------------------------------------

    duplicate_keys = int(
        df.duplicated(
            ["game_id", "team"]
        ).sum()
    )

    game_counts = (
        df.groupby("game_id")
        .size()
    )

    bad_game_counts = int(
        (game_counts != 2).sum()
    )

    missing_identity = int(
        df["coach_identity"]
        .isna()
        .sum()
    )

    coach_change_count = int(
        df["coach_change_flag"].sum()
    )

    era_start_count = int(
        df[
            "coach_team_era_start_flag"
        ].sum()
    )

    cold_count = int(
        df[
            "coach_cold_start_flag"
        ].sum()
    )

    team_cold_count = int(
        df[
            "coach_team_cold_start_flag"
        ].sum()
    )

    if duplicate_keys:
        raise SystemExit(
            "FAIL | duplicate game_id/team keys"
        )

    if bad_game_counts:
        raise SystemExit(
            "FAIL | games without exactly 2 team rows"
        )

    if missing_identity:
        raise SystemExit(
            "FAIL | missing normalized coach identity"
        )

    # Explicit alias validation.
    bad_alias = df[
        (
            df["coach_raw"]
            == "Klint Kubliak"
        )
        &
        (
            df["coach_identity"]
            != "Klint Kubiak"
        )
    ]

    if len(bad_alias):
        raise SystemExit(
            "FAIL | Klint Kubiak alias validation"
        )

    unexpected_alias = df[
        df["coach_alias_applied_flag"].eq(1)
        &
        ~df["coach_raw"].isin(
            COACH_ALIASES.keys()
        )
    ]

    if len(unexpected_alias):
        raise SystemExit(
            "FAIL | unexpected coach alias"
        )

    # --------------------------------------------------------
    # Write new Forecast artifacts only.
    # --------------------------------------------------------

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    audit = {
        "artifact":
            "forecast_v1_coach_identity",

        "status":
            "BUILD_COMPLETE_NOT_FROZEN",

        "database":
            str(DB),

        "database_mode":
            "READ_ONLY",

        "source_games":
            len(games),

        "output_rows":
            len(df),

        "unique_games":
            int(df["game_id"].nunique()),

        "unique_teams":
            int(df["team"].nunique()),

        "unique_raw_coaches":
            int(df["coach_raw"].nunique()),

        "unique_normalized_coaches":
            int(
                df["coach_identity"].nunique()
            ),

        "duplicate_game_team_keys":
            duplicate_keys,

        "games_not_exactly_two_rows":
            bad_game_counts,

        "missing_coach_identity":
            missing_identity,

        "coach_change_rows":
            coach_change_count,

        "coach_team_era_starts":
            era_start_count,

        "coach_cold_start_rows":
            cold_count,

        "coach_team_cold_start_rows":
            team_cold_count,

        "alias_rows":
            alias_rows,

        "aliases": COACH_ALIASES,

        "completed_history_updates":
            completed_history_updates,

        "temporal_leakage_violations":
            leakage_violations,

        "questionable_logic":
            "N/A",

        "fuzzy_matching":
            False,

        "sqlite_writes":
            False,
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
    print("=== OUTPUT ===")
    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Games: {df['game_id'].nunique():,}"
    )
    print(
        f"Teams: {df['team'].nunique():,}"
    )
    print(
        "Raw coaches: "
        f"{df['coach_raw'].nunique():,}"
    )
    print(
        "Normalized coaches: "
        f"{df['coach_identity'].nunique():,}"
    )

    print()
    print("=== IDENTITY ===")
    print(
        f"Coach change rows: {coach_change_count:,}"
    )
    print(
        f"Coach-team era starts: {era_start_count:,}"
    )
    print(
        f"Alias rows: {alias_rows:,}"
    )

    print()
    print("=== HISTORY ===")
    print(
        "Coach cold-start rows: "
        f"{cold_count:,}"
    )
    print(
        "Coach-team cold-start rows: "
        f"{team_cold_count:,}"
    )
    print(
        "Completed historical updates: "
        f"{completed_history_updates:,}"
    )

    print()
    print("=== SAFETY ===")
    print(
        f"Duplicate keys: {duplicate_keys}"
    )
    print(
        "Bad game row counts: "
        f"{bad_game_counts}"
    )
    print(
        "Missing identities: "
        f"{missing_identity}"
    )
    print(
        "Temporal leakage violations: "
        f"{leakage_violations}"
    )
    print("Fuzzy matching: FALSE")
    print("SQLite writes: FALSE")

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
    print(
        "============================================================"
    )
    print(
        "PASS | COACH IDENTITY V1 BUILD COMPLETE"
    )
    print(
        "STATUS | NOT YET FROZEN"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
