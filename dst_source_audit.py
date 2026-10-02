#!/usr/bin/env python3

"""
dst_source_audit.py

Read-only audit for the NFL FanDuel D/ST modeling pipeline.

Purpose
-------
Inspect the existing SQLite database and current FanDuel D-ST.csv without
modifying any database table or any frozen offensive projection artifact.

The audit answers:
1. What tables/columns currently exist for historical team/game data?
2. Which likely D/ST scoring components are already available?
3. Can points allowed be derived from the games table?
4. What current D-ST.csv fields are available for later live integration?
5. Which additional source components, if any, must be ingested before an
   exact historical FanDuel D/ST scoring target can be built?

This script intentionally makes no assumptions about nflverse column names.
It discovers the actual local schema first.
"""

from pathlib import Path
import sqlite3
import pandas as pd

from config import DATABASE_PATH, CSV_DIR


FANDUEL_DIR = Path(__file__).resolve().parent / "data" / "fanduel"
DST_FILE = FANDUEL_DIR / "D-ST.csv"

AUDIT_SCHEMA_CSV = Path(CSV_DIR) / "audit_dst_source_schema.csv"
AUDIT_COVERAGE_CSV = Path(CSV_DIR) / "audit_dst_source_coverage.csv"
AUDIT_SAMPLE_CSV = Path(CSV_DIR) / "audit_dst_source_samples.csv"

TARGET_TABLES = [
    "games",
    "team_game_stats",
    "player_game_stats",
]

SCORING_CONCEPTS = {
    "sacks": [
        "sack",
        "sacks",
        "def_sack",
        "def_sacks",
        "sacks_defense",
    ],
    "interceptions": [
        "interception",
        "interceptions",
        "def_interception",
        "def_interceptions",
        "interceptions_defense",
    ],
    "fumble_recoveries": [
        "fumble_recovery",
        "fumble_recoveries",
        "fumbles_recovered",
        "def_fumble_recoveries",
    ],
    "safeties": [
        "safety",
        "safeties",
        "def_safety",
        "def_safeties",
    ],
    "blocked_kicks": [
        "blocked_kick",
        "blocked_kicks",
        "blocked_punt",
        "blocked_punts",
        "blocked_field_goal",
        "blocked_field_goals",
    ],
    "return_touchdowns": [
        "return_touchdown",
        "return_touchdowns",
        "return_tds",
        "special_teams_tds",
        "defensive_tds",
        "def_tds",
        "dst_tds",
        "punt_return_tds",
        "kickoff_return_tds",
        "interception_return_tds",
        "fumble_return_tds",
    ],
    "extra_point_returns": [
        "extra_point_return",
        "extra_point_returns",
        "two_point_return",
        "two_point_returns",
    ],
}


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def qident(name):
    return '"' + str(name).replace('"', '""') + '"'


def table_exists(conn, table):
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def get_schema(conn, table):
    rows = conn.execute(
        f"PRAGMA table_info({qident(table)})"
    ).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "cid",
            "name",
            "type",
            "notnull",
            "default_value",
            "pk",
        ],
    )


def get_row_count(conn, table):
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {qident(table)}"
        ).fetchone()[0]
    )


def normalize_column(name):
    return str(name).strip().lower()


def concept_matches(columns, candidates):
    normalized = {
        normalize_column(c): c
        for c in columns
    }

    exact = []
    for candidate in candidates:
        if candidate in normalized:
            exact.append(normalized[candidate])

    if exact:
        return exact

    partial = []
    for column in columns:
        lowered = normalize_column(column)
        for candidate in candidates:
            if candidate in lowered or lowered in candidate:
                partial.append(column)
                break

    return sorted(set(partial))


def numeric_coverage(conn, table, column):
    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(
                CASE
                    WHEN {qident(column)} IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS nonnull_rows
        FROM {qident(table)}
    """

    row = conn.execute(query).fetchone()

    total = int(row[0] or 0)
    nonnull = int(row[1] or 0)

    return total, nonnull


def audit_tables(conn):
    schema_rows = []
    coverage_rows = []

    for table in TARGET_TABLES:
        exists = table_exists(conn, table)

        print()
        print(f"{table}: {'FOUND' if exists else 'MISSING'}")

        if not exists:
            coverage_rows.append(
                {
                    "source": table,
                    "item": "table_exists",
                    "value": 0,
                    "detail": "Table not found",
                }
            )
            continue

        schema = get_schema(conn, table)
        row_count = get_row_count(conn, table)

        print(f"Rows: {row_count}")
        print("Columns:")
        print(
            schema[
                ["name", "type", "pk"]
            ].to_string(index=False)
        )

        coverage_rows.append(
            {
                "source": table,
                "item": "table_exists",
                "value": 1,
                "detail": f"{row_count} rows",
            }
        )

        for _, row in schema.iterrows():
            schema_rows.append(
                {
                    "source": table,
                    "column": row["name"],
                    "type": row["type"],
                    "pk": int(row["pk"]),
                }
            )

        columns = schema["name"].tolist()

        print()
        print("Likely D/ST scoring fields:")

        any_match = False

        for concept, candidates in SCORING_CONCEPTS.items():
            matches = concept_matches(
                columns,
                candidates,
            )

            if matches:
                any_match = True
                print(
                    f"  {concept}: "
                    + ", ".join(matches)
                )

                for column in matches:
                    total, nonnull = numeric_coverage(
                        conn,
                        table,
                        column,
                    )

                    coverage_rows.append(
                        {
                            "source": table,
                            "item": concept,
                            "value": nonnull,
                            "detail":
                                f"{column}: "
                                f"{nonnull}/{total} non-null",
                        }
                    )

        if not any_match:
            print("  None discovered by schema-name audit.")

    return (
        pd.DataFrame(schema_rows),
        pd.DataFrame(coverage_rows),
    )


def audit_games_points_allowed(conn, coverage_rows):
    section("POINTS-ALLOWED DERIVATION AUDIT")

    if not table_exists(conn, "games"):
        print("games table missing.")
        coverage_rows.append(
            {
                "source": "games",
                "item": "points_allowed_derivable",
                "value": 0,
                "detail": "games table missing",
            }
        )
        return pd.DataFrame()

    schema = get_schema(conn, "games")
    columns = set(schema["name"].tolist())

    required = {
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = sorted(required - columns)

    if missing:
        print(
            "Cannot derive points allowed. Missing: "
            + ", ".join(missing)
        )

        coverage_rows.append(
            {
                "source": "games",
                "item": "points_allowed_derivable",
                "value": 0,
                "detail":
                    "Missing columns: "
                    + ", ".join(missing),
            }
        )
        return pd.DataFrame()

    query = """
        SELECT
            game_id,
            season,
            week,
            game_type,
            home_team,
            away_team,
            home_score,
            away_score,
            completed
        FROM games
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, game_id
    """

    games = pd.read_sql_query(
        query,
        conn,
    )

    print(
        f"Completed/scored games available: "
        f"{len(games)}"
    )

    if games.empty:
        coverage_rows.append(
            {
                "source": "games",
                "item": "points_allowed_derivable",
                "value": 0,
                "detail": "No scored games",
            }
        )
        return games

    home = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    home["team"] = home["home_team"]
    home["opponent_team"] = home["away_team"]
    home["points_scored"] = home["home_score"]
    home["points_allowed"] = home["away_score"]

    away = games[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    away["team"] = away["away_team"]
    away["opponent_team"] = away["home_team"]
    away["points_scored"] = away["away_score"]
    away["points_allowed"] = away["home_score"]

    team_games = pd.concat(
        [home, away],
        ignore_index=True,
    )

    print(
        f"Team-game points-allowed rows derivable: "
        f"{len(team_games)}"
    )

    print()
    print("Sample:")
    print(
        team_games[
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent_team",
                "points_allowed",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    coverage_rows.append(
        {
            "source": "games",
            "item": "points_allowed_derivable",
            "value": 1,
            "detail":
                f"{len(team_games)} team-game rows",
        }
    )

    return team_games


def audit_team_game_relationship(conn, coverage_rows):
    section("TEAM-GAME RELATIONSHIP AUDIT")

    if not (
        table_exists(conn, "games")
        and
        table_exists(conn, "team_game_stats")
    ):
        print(
            "games and/or team_game_stats missing; "
            "relationship audit skipped."
        )
        return

    game_schema = get_schema(conn, "games")
    team_schema = get_schema(conn, "team_game_stats")

    game_columns = set(
        game_schema["name"].tolist()
    )

    team_columns = set(
        team_schema["name"].tolist()
    )

    if "game_id" not in game_columns:
        print("games.game_id missing.")
        return

    if "game_id" not in team_columns:
        print("team_game_stats.game_id missing.")
        return

    if "team" not in team_columns:
        print("team_game_stats.team missing.")
        return

    query = """
        SELECT
            COUNT(*) AS total_team_rows,
            SUM(
                CASE
                    WHEN g.game_id IS NULL
                    THEN 1
                    ELSE 0
                END
            ) AS unmatched_game_rows
        FROM team_game_stats t
        LEFT JOIN games g
          ON t.game_id = g.game_id
    """

    row = conn.execute(query).fetchone()

    total = int(row[0] or 0)
    unmatched = int(row[1] or 0)

    print(
        f"team_game_stats rows: {total}"
    )
    print(
        f"Unmatched game_id rows: {unmatched}"
    )

    coverage_rows.append(
        {
            "source": "team_game_stats",
            "item": "unmatched_game_id_rows",
            "value": unmatched,
            "detail": f"{unmatched}/{total}",
        }
    )

    dup_query = """
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                COUNT(*) AS n
            FROM team_game_stats
            GROUP BY
                game_id,
                team
            HAVING COUNT(*) > 1
        )
    """

    duplicates = int(
        conn.execute(
            dup_query
        ).fetchone()[0]
    )

    print(
        f"Duplicate game_id/team groups: "
        f"{duplicates}"
    )

    coverage_rows.append(
        {
            "source": "team_game_stats",
            "item": "duplicate_game_team_groups",
            "value": duplicates,
            "detail": "Expected 0",
        }
    )


def audit_dst_csv(coverage_rows):
    section("CURRENT FANDUEL D-ST SOURCE AUDIT")

    print(
        f"D-ST source: {DST_FILE}"
    )

    if not DST_FILE.exists():
        print("D-ST.csv not found.")

        coverage_rows.append(
            {
                "source": "D-ST.csv",
                "item": "file_exists",
                "value": 0,
                "detail": str(DST_FILE),
            }
        )

        return pd.DataFrame()

    dst = pd.read_csv(
        DST_FILE
    )

    print(
        f"Rows: {len(dst)}"
    )

    print(
        "Columns:"
    )

    for column in dst.columns:
        print(
            f"  {column}"
        )

    coverage_rows.append(
        {
            "source": "D-ST.csv",
            "item": "file_exists",
            "value": 1,
            "detail": f"{len(dst)} rows",
        }
    )

    if "salary" in dst.columns:
        salary = pd.to_numeric(
            dst["salary"],
            errors="coerce",
        )

        print()
        print(
            f"Salary-bearing rows: "
            f"{int(salary.notna().sum())}"
        )

        print(
            f"Blank/invalid salary rows: "
            f"{int(salary.isna().sum())}"
        )

        coverage_rows.append(
            {
                "source": "D-ST.csv",
                "item": "salary_bearing_rows",
                "value": int(
                    salary.notna().sum()
                ),
                "detail":
                    f"{int(salary.isna().sum())} "
                    "blank/invalid",
            }
        )

    expected_live = [
        "player",
        "team",
        "gameInfo",
        "salary",
        "value",
        "pointsAllowed",
        "yardsAllowed",
        "sacks",
        "interceptions",
        "fumblesRecovered",
        "touchdowns",
        "fantasy",
        "positionRank",
        "opponentOffensiveRank",
    ]

    print()
    print("Expected live fields:")

    for field in expected_live:
        present = field in dst.columns

        print(
            f"  {field}: "
            f"{'YES' if present else 'NO'}"
        )

        coverage_rows.append(
            {
                "source": "D-ST.csv",
                "item": field,
                "value": int(present),
                "detail":
                    "present"
                    if present
                    else "missing",
            }
        )

    return dst


def build_component_summary(schema_df):
    section("HISTORICAL D/ST COMPONENT SUMMARY")

    if schema_df.empty:
        print(
            "No historical schema rows available."
        )
        return pd.DataFrame()

    rows = []

    for concept, candidates in SCORING_CONCEPTS.items():
        matches = []

        for source in TARGET_TABLES:
            source_columns = (
                schema_df.loc[
                    schema_df[
                        "source"
                    ]
                    ==
                    source,
                    "column",
                ]
                .tolist()
            )

            found = concept_matches(
                source_columns,
                candidates,
            )

            for column in found:
                matches.append(
                    f"{source}.{column}"
                )

        rows.append(
            {
                "component": concept,
                "schema_match":
                    " | ".join(matches)
                    if matches
                    else "",
                "available_by_schema":
                    int(bool(matches)),
            }
        )

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "NOTE: A schema-name match proves only that a "
        "candidate field exists. It does not yet prove "
        "FanDuel scoring semantics. Exact scoring will "
        "be implemented only after this audit is reviewed."
    )

    return summary


def export_audits(
    schema_df,
    coverage_df,
    component_summary,
    points_allowed_sample,
    dst_sample,
):
    section("EXPORTING DST SOURCE AUDIT")

    Path(CSV_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_df.to_csv(
        AUDIT_SCHEMA_CSV,
        index=False,
    )

    combined_coverage = pd.concat(
        [
            coverage_df,
            component_summary.assign(
                source="component_summary",
                item=lambda x:
                    x["component"],
                value=lambda x:
                    x["available_by_schema"],
                detail=lambda x:
                    x["schema_match"],
            )[
                [
                    "source",
                    "item",
                    "value",
                    "detail",
                ]
            ],
        ],
        ignore_index=True,
    )

    combined_coverage.to_csv(
        AUDIT_COVERAGE_CSV,
        index=False,
    )

    sample_frames = []

    if (
        points_allowed_sample
        is not None
        and
        not points_allowed_sample.empty
    ):
        pa = (
            points_allowed_sample[
                [
                    "game_id",
                    "season",
                    "week",
                    "team",
                    "opponent_team",
                    "points_allowed",
                ]
            ]
            .head(100)
            .copy()
        )

        pa.insert(
            0,
            "sample_type",
            "POINTS_ALLOWED",
        )

        sample_frames.append(pa)

    if (
        dst_sample
        is not None
        and
        not dst_sample.empty
    ):
        ds = (
            dst_sample
            .head(100)
            .copy()
        )

        ds.insert(
            0,
            "sample_type",
            "CURRENT_DST_SOURCE",
        )

        sample_frames.append(ds)

    if sample_frames:
        samples = pd.concat(
            sample_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        samples = pd.DataFrame(
            columns=[
                "sample_type",
            ]
        )

    samples.to_csv(
        AUDIT_SAMPLE_CSV,
        index=False,
    )

    print(
        f"Schema audit: "
        f"{AUDIT_SCHEMA_CSV}"
    )

    print(
        f"Coverage audit: "
        f"{AUDIT_COVERAGE_CSV}"
    )

    print(
        f"Sample audit: "
        f"{AUDIT_SAMPLE_CSV}"
    )


def main():
    section("NFL D/ST SOURCE AUDIT — READ ONLY")

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        "No database writes will be performed."
    )

    if not Path(DATABASE_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: "
            f"{DATABASE_PATH}"
        )

    uri = (
        "file:"
        + str(
            Path(DATABASE_PATH).resolve()
        )
        + "?mode=ro"
    )

    coverage_rows = []

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:

        section("DATABASE TABLE / SCHEMA AUDIT")

        schema_df, table_coverage = (
            audit_tables(
                conn
            )
        )

        if not table_coverage.empty:
            coverage_rows.extend(
                table_coverage.to_dict(
                    orient="records"
                )
            )

        points_allowed = (
            audit_games_points_allowed(
                conn,
                coverage_rows,
            )
        )

        audit_team_game_relationship(
            conn,
            coverage_rows,
        )

    dst = audit_dst_csv(
        coverage_rows
    )

    component_summary = (
        build_component_summary(
            schema_df
        )
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    export_audits(
        schema_df,
        coverage_df,
        component_summary,
        points_allowed,
        dst,
    )

    section("DST SOURCE AUDIT COMPLETE")

    print(
        "This audit did not modify SQLite "
        "or the frozen offensive pipeline."
    )

    print()
    print(
        "Next decision:"
    )

    print(
        "Use the reported historical fields "
        "to determine whether exact FanDuel "
        "D/ST scoring can be built directly "
        "or whether an additional historical "
        "defensive source is required."
    )


if __name__ == "__main__":
    main()
