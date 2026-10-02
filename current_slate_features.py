import numpy as np
import pandas as pd

from config import (
    CURRENT_SEASON,
    CSV_DIR,
    PARQUET_DIR,
)

from database import get_connection


# =========================================================
# CONFIGURATION
# =========================================================

POSITIONS = [
    "QB",
    "RB",
    "WR",
    "TE",
]

CORE_FEATURES_CSV = (
    CSV_DIR / "nfl_core_projection_features.csv"
)

OUTPUT_CSV = (
    CSV_DIR / "nfl_current_slate_features.csv"
)

OUTPUT_PARQUET = (
    PARQUET_DIR / "nfl_current_slate_features.parquet"
)

AUDIT_CSV = (
    CSV_DIR / "audit_current_slate_features.csv"
)

AUDIT_PARQUET = (
    PARQUET_DIR / "audit_current_slate_features.parquet"
)


# =========================================================
# GENERAL HELPERS
# =========================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def table_exists(
    conn,
    table_name,
):

    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (
            table_name,
        ),
    ).fetchone()

    return bool(
        row[0]
    )


def table_columns(
    conn,
    table_name,
):

    rows = conn.execute(
        f'''
        PRAGMA table_info(
            "{table_name}"
        )
        '''
    ).fetchall()

    return [
        row[1]
        for row in rows
    ]


def list_tables(conn):

    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def first_existing(
    columns,
    candidates,
):

    columns = set(
        columns
    )

    for candidate in candidates:

        if candidate in columns:

            return candidate

    return None


def require_column(
    columns,
    candidates,
    logical_name,
    source_name,
):

    column = first_existing(
        columns,
        candidates,
    )

    if column is None:

        print()
        print(
            f"Available columns in {source_name}:"
        )
        print()

        for value in sorted(
            columns
        ):

            print(
                f"  {value}"
            )

        raise RuntimeError(
            f"Could not resolve "
            f"{logical_name} in "
            f"{source_name}."
        )

    return column


def normalize_position(value):

    if pd.isna(value):

        return None

    value = str(
        value
    ).strip().upper()

    if value in POSITIONS:

        return value

    return None


def normalize_depth_role(value):

    if pd.isna(value):

        return None

    value = str(
        value
    ).strip().upper()

    if value in set(POSITIONS) | {"FB"}:

        return value

    return None


def normalize_key(value):

    if pd.isna(value):

        return None

    value = str(
        value
    ).strip()

    if not value:

        return None

    return value


def numeric_series(
    df,
    column,
    default=0.0,
):

    if column is None:

        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(default)
        .astype(float)
    )


# =========================================================
# CORE FEATURES
# =========================================================

def load_core_features():

    section(
        "LOADING FROZEN CORE FEATURE SET"
    )

    if not CORE_FEATURES_CSV.exists():

        raise RuntimeError(
            f"Missing core feature file: "
            f"{CORE_FEATURES_CSV}"
        )

    df = pd.read_csv(
        CORE_FEATURES_CSV
    )

    required = [
        "position",
        "core_rank",
        "feature",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Core feature file missing: "
            +
            ", ".join(
                missing
            )
        )

    df = df[
        df[
            "position"
        ].isin(
            POSITIONS
        )
    ].copy()

    print(
        f"Core feature rows: "
        f"{len(df)}"
    )

    for position in POSITIONS:

        count = len(
            df[
                df[
                    "position"
                ]
                ==
                position
            ]
        )

        print(
            f"{position}: "
            f"{count}"
        )

    return df


# =========================================================
# LOAD NORMALIZED GAMES
# =========================================================

def load_games(conn):

    if not table_exists(
        conn,
        "games",
    ):

        raise RuntimeError(
            "games table does not exist."
        )

    columns = table_columns(
        conn,
        "games",
    )

    game_id_col = require_column(
        columns,
        [
            "game_id",
        ],
        "game ID",
        "games",
    )

    season_col = require_column(
        columns,
        [
            "season",
        ],
        "season",
        "games",
    )

    week_col = require_column(
        columns,
        [
            "week",
        ],
        "week",
        "games",
    )

    game_type_col = require_column(
        columns,
        [
            "game_type",
        ],
        "game type",
        "games",
    )

    date_col = require_column(
        columns,
        [
            "date",
            "gameday",
            "game_date",
        ],
        "game date",
        "games",
    )

    time_col = first_existing(
        columns,
        [
            "time",
            "gametime",
            "game_time",
        ],
    )

    away_col = require_column(
        columns,
        [
            "away_team",
        ],
        "away team",
        "games",
    )

    home_col = require_column(
        columns,
        [
            "home_team",
        ],
        "home team",
        "games",
    )

    completed_col = require_column(
        columns,
        [
            "completed",
        ],
        "completed flag",
        "games",
    )

    if time_col is None:

        time_expression = (
            "NULL AS game_time"
        )

    else:

        time_expression = (
            f'"{time_col}" AS game_time'
        )

    query = f"""
        SELECT
            "{game_id_col}" AS game_id,
            "{season_col}" AS season,
            "{week_col}" AS week,
            "{game_type_col}" AS game_type,
            "{date_col}" AS game_date,
            {time_expression},
            "{away_col}" AS away_team,
            "{home_col}" AS home_team,
            "{completed_col}" AS completed

        FROM games
    """

    games = pd.read_sql_query(
        query,
        conn,
    )

    games[
        "season"
    ] = pd.to_numeric(
        games[
            "season"
        ],
        errors="coerce",
    )

    games[
        "week"
    ] = pd.to_numeric(
        games[
            "week"
        ],
        errors="coerce",
    )

    games[
        "completed"
    ] = (
        pd.to_numeric(
            games[
                "completed"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    games[
        "game_date"
    ] = pd.to_datetime(
        games[
            "game_date"
        ],
        errors="coerce",
    )

    return games


# =========================================================
# CURRENT SLATE
# =========================================================

def load_inference_targets(
    conn,
    games,
):
    """
    Build the exact live inference universe.

    Authority:
      1. ACTIVE target = earliest unfinished current-season REG week,
         but only the unfinished games in that week.
      2. STAGED targets = exact game_ids persisted by the schedule-aware
         FanDuel slate manifest for STAGED_NO_SALARY slates.

    We intentionally do NOT select every future incomplete NFL game.
    """

    section(
        "DETECTING ACTIVE + STAGED INFERENCE TARGETS"
    )

    season_games = games[
        games[
            "season"
        ]
        ==
        CURRENT_SEASON
    ].copy()

    if season_games.empty:

        raise RuntimeError(
            f"No games found for "
            f"{CURRENT_SEASON}."
        )

    incomplete = season_games[
        season_games[
            "completed"
        ]
        ==
        0
    ].copy()

    regular = incomplete[
        incomplete[
            "game_type"
        ]
        .astype(str)
        .str.upper()
        ==
        "REG"
    ].copy()

    if not regular.empty:

        incomplete = regular

    target_frames = []

    # -----------------------------------------------------
    # ACTIVE NFL TARGET
    # -----------------------------------------------------

    if not incomplete.empty:

        active_week = int(
            incomplete[
                "week"
            ].min()
        )

        active = incomplete[
            incomplete[
                "week"
            ]
            ==
            active_week
        ].copy()

        active[
            "inference_source"
        ] = "ACTIVE_UNFINISHED"

        target_frames.append(
            active
        )

        print(
            f"Active target: "
            f"{CURRENT_SEASON} week {active_week}"
        )

        print(
            f"Active unfinished games: "
            f"{len(active)}"
        )

    else:

        print(
            "No unfinished current-season "
            "schedule games remain."
        )

    # -----------------------------------------------------
    # STAGED FANDUEL TARGETS
    # -----------------------------------------------------

    if table_exists(
        conn,
        "fanduel_slate_manifest",
    ):

        manifest_columns = set(
            table_columns(
                conn,
                "fanduel_slate_manifest",
            )
        )

        required_manifest = {
            "season",
            "week",
            "schedule_game_list",
            "structural_status",
        }

        if required_manifest.issubset(
            manifest_columns
        ):

            staged_manifest = pd.read_sql_query(
                """
                SELECT
                    slate_name,
                    slate_slug,
                    season,
                    week,
                    slate_key,
                    schedule_game_list,
                    structural_status
                FROM fanduel_slate_manifest
                WHERE structural_status = 'STAGED_NO_SALARY'
                """,
                conn,
            )

            for _, staged_row in staged_manifest.iterrows():

                staged_season = pd.to_numeric(
                    pd.Series(
                        [
                            staged_row[
                                "season"
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                staged_week = pd.to_numeric(
                    pd.Series(
                        [
                            staged_row[
                                "week"
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if (
                    pd.isna(
                        staged_season
                    )
                    or
                    pd.isna(
                        staged_week
                    )
                ):

                    raise RuntimeError(
                        "Staged FanDuel manifest row "
                        "has invalid season/week."
                    )

                staged_season = int(
                    staged_season
                )

                staged_week = int(
                    staged_week
                )

                raw_game_ids = str(
                    staged_row[
                        "schedule_game_list"
                    ]
                )

                game_ids = [
                    value.strip()
                    for value in raw_game_ids.split(
                        ";"
                    )
                    if value.strip()
                ]

                if not game_ids:

                    raise RuntimeError(
                        "STAGED_NO_SALARY manifest row "
                        f"{staged_row['slate_name']} "
                        "has no schedule game IDs."
                    )

                staged_games = games[
                    games[
                        "game_id"
                    ].astype(str).isin(
                        game_ids
                    )
                ].copy()

                resolved_ids = set(
                    staged_games[
                        "game_id"
                    ].astype(str)
                )

                missing_ids = sorted(
                    set(
                        game_ids
                    )
                    -
                    resolved_ids
                )

                if missing_ids:

                    raise RuntimeError(
                        "Staged FanDuel target contains "
                        "game IDs missing from games: "
                        +
                        ", ".join(
                            missing_ids
                        )
                    )

                pairs = (
                    staged_games[
                        [
                            "season",
                            "week",
                        ]
                    ]
                    .drop_duplicates()
                )

                if len(
                    pairs
                ) != 1:

                    raise RuntimeError(
                        "Staged FanDuel game IDs span "
                        "multiple season/week pairs."
                    )

                resolved_season = int(
                    pairs.iloc[
                        0
                    ][
                        "season"
                    ]
                )

                resolved_week = int(
                    pairs.iloc[
                        0
                    ][
                        "week"
                    ]
                )

                if (
                    resolved_season
                    !=
                    staged_season
                    or
                    resolved_week
                    !=
                    staged_week
                ):

                    raise RuntimeError(
                        "Staged FanDuel manifest "
                        "season/week disagrees with "
                        "games schedule authority."
                    )

                staged_games[
                    "inference_source"
                ] = (
                    "STAGED_FANDUEL:"
                    +
                    str(
                        staged_row[
                            "slate_slug"
                        ]
                    )
                )

                target_frames.append(
                    staged_games
                )

                print(
                    f"Staged target: "
                    f"{staged_season} "
                    f"week {staged_week} "
                    f"{staged_row['slate_name']} "
                    f"games={len(staged_games)}"
                )

    if not target_frames:

        raise RuntimeError(
            "No active or staged inference "
            "targets were found."
        )

    targets = pd.concat(
        target_frames,
        ignore_index=True,
    )

    # A game can appear in both authorities. Schedule game_id is the
    # canonical deduplication key; ACTIVE takes precedence only for the
    # source label when overlap occurs.
    targets[
        "_source_priority"
    ] = np.where(
        targets[
            "inference_source"
        ].eq(
            "ACTIVE_UNFINISHED"
        ),
        0,
        1,
    )

    targets = (
        targets.sort_values(
            [
                "_source_priority",
                "season",
                "week",
                "game_date",
                "game_time",
                "game_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "game_id",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_source_priority",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    targets = targets.sort_values(
        [
            "season",
            "week",
            "game_date",
            "game_time",
            "game_id",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Resolved inference targets:"
    )

    print(
        targets[
            [
                "season",
                "week",
                "game_id",
                "game_date",
                "game_time",
                "away_team",
                "home_team",
                "inference_source",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Target groups:"
    )

    print(
        targets.groupby(
            [
                "season",
                "week",
            ]
        )[
            "game_id"
        ]
        .nunique()
        .to_string()
    )

    return targets


def target_groups(
    targets,
):
    groups = []

    for (
        target_season,
        target_week,
    ), slate in targets.groupby(
        [
            "season",
            "week",
        ],
        sort=True,
    ):

        slate = slate.copy()

        teams = sorted(
            set(
                slate[
                    "away_team"
                ].dropna()
            )
            |
            set(
                slate[
                    "home_team"
                ].dropna()
            )
        )

        groups.append(
            (
                int(
                    target_season
                ),
                int(
                    target_week
                ),
                slate,
                teams,
            )
        )

    return groups


# =========================================================
# CURRENT DEPTH CHART
# =========================================================

def load_current_depth(
    conn,
    teams,
):

    section(
        "LOADING CURRENT DEPTH CHART"
    )

    if not table_exists(
        conn,
        "depth_charts",
    ):

        raise RuntimeError(
            "depth_charts table does not exist."
        )

    columns = table_columns(
        conn,
        "depth_charts",
    )

    snapshot_col = require_column(
        columns,
        [
            "snapshot_dt",
            "dt",
        ],
        "snapshot timestamp",
        "depth_charts",
    )

    team_col = require_column(
        columns,
        [
            "team",
        ],
        "team",
        "depth_charts",
    )

    name_col = require_column(
        columns,
        [
            "player_name",
            "full_name",
            "name",
        ],
        "player name",
        "depth_charts",
    )

    position_col = require_column(
        columns,
        [
            "position",
            "pos_abb",
            "pos_name",
        ],
        "position",
        "depth_charts",
    )

    rank_col = require_column(
        columns,
        [
            "pos_rank",
            "depth_rank",
        ],
        "depth rank",
        "depth_charts",
    )

    gsis_col = first_existing(
        columns,
        [
            "gsis_id",
        ],
    )

    espn_col = first_existing(
        columns,
        [
            "espn_id",
        ],
    )

    placeholders = ",".join(
        [
            "?"
            for _ in teams
        ]
    )

    if gsis_col:

        gsis_expr = (
            f'"{gsis_col}" AS gsis_id'
        )

    else:

        gsis_expr = (
            "NULL AS gsis_id"
        )

    if espn_col:

        espn_expr = (
            f'"{espn_col}" AS espn_id'
        )

    else:

        espn_expr = (
            "NULL AS espn_id"
        )

    query = f"""
        SELECT
            "{snapshot_col}" AS snapshot_dt,
            "{team_col}" AS team,
            "{name_col}" AS player_name,
            "{position_col}" AS position,
            "{rank_col}" AS pos_rank,
            {gsis_expr},
            {espn_expr}

        FROM depth_charts

        WHERE
            "{team_col}" IN (
                {placeholders}
            )
    """

    depth = pd.read_sql_query(
        query,
        conn,
        params=teams,
    )

    if depth.empty:

        raise RuntimeError(
            "No current depth-chart rows found."
        )

    depth[
        "snapshot_dt"
    ] = pd.to_datetime(
        depth[
            "snapshot_dt"
        ],
        errors="coerce",
    )

    latest_snapshot = (
        depth.groupby(
            "team"
        )[
            "snapshot_dt"
        ]
        .transform(
            "max"
        )
    )

    depth = depth[
        depth[
            "snapshot_dt"
        ]
        ==
        latest_snapshot
    ].copy()

    depth[
        "depth_role"
    ] = depth[
        "position"
    ].apply(
        normalize_depth_role
    )

    depth = depth[
        depth[
            "depth_role"
        ].notna()
    ].copy()

    depth[
        "position"
    ] = depth[
        "depth_role"
    ]

    depth[
        "pos_rank"
    ] = pd.to_numeric(
        depth[
            "pos_rank"
        ],
        errors="coerce",
    )

    depth = (
        depth.sort_values(
            [
                "team",
                "position",
                "pos_rank",
                "player_name",
            ]
        )
        .drop_duplicates(
            subset=[
                "team",
                "position",
                "player_name",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Latest offensive depth rows: "
        f"{len(depth)}"
    )

    print()

    print(
        depth.groupby(
            "position"
        )
        .size()
        .to_string()
    )

    return depth


def attach_current_roster_position(
    depth,
    target_season,
    target_week,
):

    section(
        "RESOLVING CURRENT ROSTER POSITION AUTHORITY"
    )

    with get_connection() as conn:

        if not table_exists(
            conn,
            "weekly_rosters",
        ):

            raise RuntimeError(
                "weekly_rosters table does not exist."
            )

        roster = pd.read_sql_query(
            """
            SELECT
                team,
                gsis_id,
                position AS roster_position,
                depth_chart_position AS roster_depth_position,
                status AS roster_status,
                status_description_abbr AS roster_status_abbr
            FROM weekly_rosters
            WHERE season = ?
              AND week = ?
            """,
            conn,
            params=(
                int(target_season),
                int(target_week),
            ),
        )

    roster[
        "roster_position"
    ] = roster[
        "roster_position"
    ].apply(
        normalize_position
    )

    roster = roster[
        roster[
            "gsis_id"
        ].notna()
    ].copy()

    duplicate = (
        roster.groupby(
            [
                "team",
                "gsis_id",
            ],
            dropna=False,
        )[
            "roster_position"
        ]
        .nunique(
            dropna=False
        )
        .reset_index(
            name="position_count"
        )
    )

    duplicate = duplicate[
        duplicate[
            "position_count"
        ]
        >
        1
    ]

    if not duplicate.empty:

        raise RuntimeError(
            "Conflicting weekly roster positions for "
            "exact team+GSIS rows:\n"
            + duplicate.to_string(
                index=False
            )
        )

    roster = roster.drop_duplicates(
        subset=[
            "team",
            "gsis_id",
        ],
        keep="last",
    )

    result = depth.merge(
        roster,
        how="left",
        left_on=[
            "team",
            "resolved_gsis_id",
        ],
        right_on=[
            "team",
            "gsis_id",
        ],
        validate="many_to_one",
        sort=False,
        suffixes=(
            "",
            "_roster",
        ),
    )

    result[
        "position"
    ] = result[
        "roster_position"
    ]

    fallback = (
        result[
            "position"
        ].isna()
        & result[
            "depth_role"
        ].isin(
            POSITIONS
        )
    )

    result.loc[
        fallback,
        "position",
    ] = result.loc[
        fallback,
        "depth_role",
    ]

    result[
        "position_source"
    ] = "WEEKLY_ROSTER_EXACT_GSIS"

    result.loc[
        fallback,
        "position_source",
    ] = "LATEST_DEPTH_STANDARD_POSITION_FALLBACK"

    unresolved = ~result[
        "position"
    ].isin(
        POSITIONS
    )

    print(
        "Exact weekly-roster positions: "
        f"{int(result['roster_position'].notna().sum())}"
    )

    print(
        "Standard depth-position fallbacks: "
        f"{int(fallback.sum())}"
    )

    print(
        "Unresolved canonical positions: "
        f"{int(unresolved.sum())}"
    )

    result = result[
        ~unresolved
    ].copy()

    return result


# =========================================================
# CANONICAL PLAYER IDENTITY
# =========================================================

def attach_identity(
    conn,
    depth,
):

    section(
        "RESOLVING PLAYER IDENTITIES"
    )

    if not table_exists(
        conn,
        "player_identity",
    ):

        raise RuntimeError(
            "player_identity table does not exist."
        )

    identities = pd.read_sql_query(
        """
        SELECT
            identity_key,
            gsis_id,
            espn_id,
            full_name,
            football_name,
            position,
            latest_team

        FROM player_identity
        """,
        conn,
    )

    if identities.empty:

        raise RuntimeError(
            "player_identity is empty."
        )

    result = depth.copy()

    result[
        "identity_key"
    ] = None

    result[
        "resolved_gsis_id"
    ] = None

    result[
        "identity_display_name"
    ] = None

    # -----------------------------------------------------
    # PRIMARY MATCH: GSIS
    # -----------------------------------------------------

    gsis_map = (
        identities[
            identities[
                "gsis_id"
            ].notna()
        ][
            [
                "identity_key",
                "gsis_id",
                "full_name",
                "football_name",
            ]
        ]
        .drop_duplicates(
            subset=[
                "gsis_id"
            ]
        )
        .rename(
            columns={
                "identity_key":
                    "_gsis_identity_key",

                "gsis_id":
                    "_matched_gsis_id",

                "full_name":
                    "_gsis_full_name",

                "football_name":
                    "_gsis_football_name",
            }
        )
    )

    result = result.merge(
        gsis_map,
        how="left",
        left_on="gsis_id",
        right_on="_matched_gsis_id",
    )

    result[
        "identity_key"
    ] = result[
        "_gsis_identity_key"
    ]

    result[
        "resolved_gsis_id"
    ] = result[
        "_matched_gsis_id"
    ]

    result[
        "identity_display_name"
    ] = result[
        "_gsis_full_name"
    ].combine_first(
        result[
            "_gsis_football_name"
        ]
    )

    result = result.drop(
        columns=[
            "_gsis_identity_key",
            "_matched_gsis_id",
            "_gsis_full_name",
            "_gsis_football_name",
        ]
    )

    # -----------------------------------------------------
    # FALLBACK MATCH: ESPN
    # -----------------------------------------------------

    unresolved = (
        result[
            "identity_key"
        ].isna()
    )

    if unresolved.any():

        espn_map = (
            identities[
                identities[
                    "espn_id"
                ].notna()
            ][
                [
                    "identity_key",
                    "gsis_id",
                    "espn_id",
                    "full_name",
                    "football_name",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "espn_id"
                ]
            )
        )

        fallback = (
            result.loc[
                unresolved
            ][
                [
                    "espn_id",
                ]
            ]
            .reset_index()
            .merge(
                espn_map,
                how="left",
                on="espn_id",
            )
            .set_index(
                "index"
            )
        )

        result.loc[
            fallback.index,
            "identity_key",
        ] = fallback[
            "identity_key"
        ]

        result.loc[
            fallback.index,
            "resolved_gsis_id",
        ] = fallback[
            "gsis_id"
        ]

        fallback_name = (
            fallback[
                "full_name"
            ]
            .combine_first(
                fallback[
                    "football_name"
                ]
            )
        )

        result.loc[
            fallback.index,
            "identity_display_name",
        ] = fallback_name

    result[
        "identity_key"
    ] = result[
        "identity_key"
    ].apply(
        normalize_key
    )

    resolved = int(
        result[
            "identity_key"
        ].notna().sum()
    )

    unresolved_count = int(
        result[
            "identity_key"
        ].isna().sum()
    )

    gsis_missing = int(
        (
            result[
                "identity_key"
            ].notna()
            &
            result[
                "resolved_gsis_id"
            ].isna()
        ).sum()
    )

    print(
        f"Resolved canonical identities: "
        f"{resolved}"
    )

    print(
        f"Unresolved canonical identities: "
        f"{unresolved_count}"
    )

    print(
        f"Canonical identities without GSIS: "
        f"{gsis_missing}"
    )

    if unresolved_count:

        print()
        print(
            "Unresolved players:"
        )

        print(
            result[
                result[
                    "identity_key"
                ].isna()
            ][
                [
                    "team",
                    "position",
                    "player_name",
                    "gsis_id",
                    "espn_id",
                ]
            ]
            .to_string(
                index=False
            )
        )

    return result


# =========================================================
# DISCOVER STABLE PLAYER USAGE SOURCE
# =========================================================

def usage_schema_score(
    columns,
):

    columns = set(
        columns
    )

    score = 0

    if "game_id" in columns:

        score += 10

    if "identity_key" in columns:

        score += 20

    if (
        "player_id" in columns
        or
        "gsis_id" in columns
    ):

        score += 5

    if "fanduel_points" in columns:

        score += 5

    if "carries" in columns:

        score += 2

    if "targets" in columns:

        score += 2

    if "offense_snaps" in columns:

        score += 2

    if "offense_pct" in columns:

        score += 2

    if "role_expansion_flag" in columns:

        score += 1

    if "starter_usage_flag" in columns:

        score += 1

    if "high_usage_flag" in columns:

        score += 1

    return score


def discover_usage_table(
    conn,
):

    section(
        "DISCOVERING PLAYER USAGE SOURCE"
    )

    preferred = [
        "player_weekly_usage",
        "nfl_player_weekly_usage",
        "player_usage",
    ]

    scored = []

    for table_name in list_tables(
        conn
    ):

        columns = table_columns(
            conn,
            table_name,
        )

        score = usage_schema_score(
            columns
        )

        if table_name in preferred:

            score += 20

        scored.append(
            {
                "table":
                    table_name,

                "score":
                    score,

                "columns":
                    columns,
            }
        )

    scored = sorted(
        scored,
        key=lambda row: (
            -row[
                "score"
            ],
            row[
                "table"
            ],
        ),
    )

    print(
        "Highest-scoring SQLite sources:"
    )

    print()

    for row in scored[:10]:

        print(
            f"{row['table']:35s} "
            f"score={row['score']}"
        )

    for row in scored:

        columns = set(
            row[
                "columns"
            ]
        )

        required = {
            "game_id",
            "identity_key",
            "fanduel_points",
            "carries",
            "targets",
            "offense_snaps",
            "offense_pct",
        }

        if required.issubset(
            columns
        ):

            print()
            print(
                f"Selected SQLite usage source: "
                f"{row['table']}"
            )

            return row[
                "table"
            ]

    raise RuntimeError(
        "No compatible player usage table "
        "with canonical identity_key found."
    )


# =========================================================
# LOAD COMPLETED HISTORY
# =========================================================

def load_player_history(
    conn,
    games,
):

    section(
        "LOADING COMPLETED PLAYER HISTORY"
    )

    usage_table = discover_usage_table(
        conn
    )

    usage = pd.read_sql_query(
        f'''
        SELECT *
        FROM "{usage_table}"
        ''',
        conn,
    )

    print()
    print(
        f"Usage rows loaded: "
        f"{len(usage)}"
    )

    print(
        f"Usage columns: "
        f"{len(usage.columns)}"
    )

    completed_games = games[
        games[
            "completed"
        ]
        ==
        1
    ][
        [
            "game_id",
            "season",
            "week",
            "game_date",
        ]
    ].copy()

    completed_games = (
        completed_games.rename(
            columns={
                "season":
                    "_game_season",

                "week":
                    "_game_week",

                "game_date":
                    "_game_date",
            }
        )
    )

    history = usage.merge(
        completed_games,
        how="inner",
        on="game_id",
        validate="many_to_one",
    )

    history = history[
        history[
            "_game_season"
        ]
        <=
        CURRENT_SEASON
    ].copy()

    print()
    print(
        f"Usage source: "
        f"SQLite:{usage_table}"
    )

    print(
        f"Completed history rows: "
        f"{len(history)}"
    )

    if history.empty:

        raise RuntimeError(
            "Completed player history is empty."
        )

    return (
        history,
        usage_table,
    )


# =========================================================
# HISTORY FIELD RESOLUTION
# =========================================================

def resolve_history_columns(
    history,
    usage_table,
):

    section(
        "RESOLVING HISTORY STAT COLUMNS"
    )

    columns = set(
        history.columns
    )

    candidates = {

        "identity_key": [
            "identity_key",
        ],

        "player_id": [
            "player_id",
            "gsis_id",
        ],

        "team": [
            "team",
            "recent_team",
        ],

        "fanduel_points": [
            "fanduel_points",
            "fd_points",
            "fantasy_points_fd",
        ],

        "carries": [
            "carries",
            "rushing_attempts",
        ],

        "targets": [
            "targets",
        ],

        "receptions": [
            "receptions",
        ],

        "rushing_yards": [
            "rushing_yards",
        ],

        "receiving_yards": [
            "receiving_yards",
        ],

        "snaps": [
            "offense_snaps",
            "offensive_snaps",
            "snaps",
        ],

        "snap_pct": [
            "offense_pct",
            "offensive_snap_pct",
            "snap_pct",
        ],

        "target_share": [
            "target_share",
        ],

        "air_yards_share": [
            "air_yards_share",
        ],

        "fanduel_per_snap": [
            "fanduel_per_snap",
        ],

        "fanduel_per_touch": [
            "fanduel_per_touch",
        ],

        "yards_per_opportunity": [
            "yards_per_opportunity",
        ],

        "role_expansion": [
            "role_expansion_flag",
            "role_expansion",
        ],

        "starter_usage": [
            "starter_usage_flag",
            "starter_usage",
        ],

        "high_usage": [
            "high_usage_flag",
            "high_usage",
        ],
    }

    mapping = {}

    for logical_name, options in (
        candidates.items()
    ):

        mapping[
            logical_name
        ] = first_existing(
            columns,
            options,
        )

        print(
            f"{logical_name:22s} -> "
            f"{mapping[logical_name]}"
        )

    required = [
        "identity_key",
        "fanduel_points",
        "carries",
        "targets",
        "snaps",
        "snap_pct",
        "fanduel_per_snap",
        "fanduel_per_touch",
        "yards_per_opportunity",
    ]

    missing = [
        field
        for field in required
        if mapping[
            field
        ] is None
    ]

    if missing:

        print()
        print(
            f"Available columns in "
            f"{usage_table}:"
        )

        print()

        for column in sorted(
            columns
        ):

            print(
                f"  {column}"
            )

        raise RuntimeError(
            "Required history fields "
            "could not be resolved: "
            +
            ", ".join(
                missing
            )
        )

    return mapping


# =========================================================
# NORMALIZE COMPLETED HISTORY
# =========================================================

def normalize_history(
    history,
    mapping,
):

    section(
        "NORMALIZING COMPLETED PLAYER HISTORY"
    )

    result = pd.DataFrame(
        index=history.index
    )

    result[
        "game_id"
    ] = history[
        "game_id"
    ]

    result[
        "season"
    ] = pd.to_numeric(
        history[
            "_game_season"
        ],
        errors="coerce",
    )

    result[
        "week"
    ] = pd.to_numeric(
        history[
            "_game_week"
        ],
        errors="coerce",
    )

    result[
        "game_date"
    ] = pd.to_datetime(
        history[
            "_game_date"
        ],
        errors="coerce",
    )

    result[
        "identity_key"
    ] = history[
        mapping[
            "identity_key"
        ]
    ].apply(
        normalize_key
    )

    if mapping[
        "player_id"
    ] is not None:

        result[
            "player_id"
        ] = history[
            mapping[
                "player_id"
            ]
        ].astype(
            "string"
        )

    else:

        result[
            "player_id"
        ] = None

    if mapping[
        "team"
    ] is not None:

        result[
            "team"
        ] = history[
            mapping[
                "team"
            ]
        ]

    else:

        result[
            "team"
        ] = None

    result[
        "fd"
    ] = numeric_series(
        history,
        mapping[
            "fanduel_points"
        ],
    )

    result[
        "carries"
    ] = numeric_series(
        history,
        mapping[
            "carries"
        ],
    )

    result[
        "targets"
    ] = numeric_series(
        history,
        mapping[
            "targets"
        ],
    )

    result[
        "receptions"
    ] = numeric_series(
        history,
        mapping[
            "receptions"
        ],
    )

    result[
        "rushing_yards"
    ] = numeric_series(
        history,
        mapping[
            "rushing_yards"
        ],
    )

    result[
        "receiving_yards"
    ] = numeric_series(
        history,
        mapping[
            "receiving_yards"
        ],
    )

    result[
        "snaps"
    ] = numeric_series(
        history,
        mapping[
            "snaps"
        ],
    )

    result[
        "snap_pct"
    ] = numeric_series(
        history,
        mapping[
            "snap_pct"
        ],
    )

    result[
        "target_share"
    ] = numeric_series(
        history,
        mapping[
            "target_share"
        ],
    )

    result[
        "air_yards_share"
    ] = numeric_series(
        history,
        mapping[
            "air_yards_share"
        ],
    )

    # -----------------------------------------------------
    # SOURCE-AUTHORITATIVE EFFICIENCY METRICS
    #
    # These values are consumed directly from the stable
    # player_weekly_usage layer. They are NOT recomputed
    # here. This preserves exact train/live feature parity
    # with pregame_features.py.
    # -----------------------------------------------------

    result[
        "fd_per_snap"
    ] = numeric_series(
        history,
        mapping[
            "fanduel_per_snap"
        ],
    )

    result[
        "fd_per_touch"
    ] = numeric_series(
        history,
        mapping[
            "fanduel_per_touch"
        ],
    )

    result[
        "yards_per_opportunity"
    ] = numeric_series(
        history,
        mapping[
            "yards_per_opportunity"
        ],
    )

    result[
        "role_expansion"
    ] = numeric_series(
        history,
        mapping[
            "role_expansion"
        ],
    )

    result[
        "starter_usage"
    ] = numeric_series(
        history,
        mapping[
            "starter_usage"
        ],
    )

    result[
        "high_usage"
    ] = numeric_series(
        history,
        mapping[
            "high_usage"
        ],
    )

    # -----------------------------------------------------
    # EXACT GAME-LEVEL DERIVED METRICS USED BY THE
    # PREGAME FEATURE ENGINE.
    # -----------------------------------------------------

    result[
        "opportunities"
    ] = (
        result[
            "carries"
        ]
        +
        result[
            "targets"
        ]
    )

    result[
        "touches"
    ] = (
        result[
            "carries"
        ]
        +
        result[
            "receptions"
        ]
    )

    result[
        "yards"
    ] = (
        result[
            "rushing_yards"
        ]
        +
        result[
            "receiving_yards"
        ]
    )

    result = result[
        result[
            "identity_key"
        ].notna()
    ].copy()

    result = (
        result.sort_values(
            [
                "identity_key",
                "game_date",
                "season",
                "week",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Normalized history rows: "
        f"{len(result)}"
    )

    print(
        f"Unique canonical identities: "
        f"{result['identity_key'].nunique()}"
    )

    return result


# =========================================================
# ROLLING HELPERS
# =========================================================

def trailing_mean(
    history,
    column,
    games,
):

    values = history[
        column
    ].tail(
        games
    )

    if values.empty:

        return 0.0

    value = values.mean()

    if pd.isna(
        value
    ):

        return 0.0

    return float(
        value
    )


def trailing_sum(
    history,
    column,
    games,
):

    values = history[
        column
    ].tail(
        games
    )

    if values.empty:

        return 0.0

    value = values.sum()

    if pd.isna(
        value
    ):

        return 0.0

    return float(
        value
    )


def trailing_std(
    history,
    column,
    games,
):

    values = history[
        column
    ].tail(
        games
    )

    if len(
        values
    ) <= 1:

        return 0.0

    value = values.std(
        ddof=0
    )

    if pd.isna(
        value
    ):

        return 0.0

    return float(
        value
    )


def last_value(
    history,
    column,
):

    if history.empty:

        return 0.0

    value = history.iloc[
        -1
    ][
        column
    ]

    if pd.isna(
        value
    ):

        return 0.0

    return float(
        value
    )


# =========================================================
# EXACT ESTABLISHED ROLE DEFINITION
# =========================================================

def calculate_established_role_flag(
    history,
):

    # -----------------------------------------------------
    # Exact parity with pregame_features.py:
    #
    # established_role_flag =
    #
    #     snap_pct_avg_3 >= 0.60
    #
    #     OR
    #
    #     opportunities_avg_3 >= 12.0
    #
    # Both inputs use COMPLETED PRIOR GAMES ONLY.
    # -----------------------------------------------------

    snap_pct_avg_3 = trailing_mean(
        history,
        "snap_pct",
        3,
    )

    opportunities_avg_3 = trailing_mean(
        history,
        "opportunities",
        3,
    )

    established = (
        (
            snap_pct_avg_3
            >=
            0.60
        )
        or
        (
            opportunities_avg_3
            >=
            12.0
        )
    )

    return int(
        established
    )


# =========================================================
# CORE FEATURE CALCULATION
# =========================================================

def calculate_feature(
    history,
    feature,
):

    calculations = {

        "fd_avg_3":
            lambda:
            trailing_mean(
                history,
                "fd",
                3,
            ),

        "fd_avg_5":
            lambda:
            trailing_mean(
                history,
                "fd",
                5,
            ),

        "fd_last":
            lambda:
            last_value(
                history,
                "fd",
            ),

        "fd_std_5":
            lambda:
            trailing_std(
                history,
                "fd",
                5,
            ),

        "carries_avg_5":
            lambda:
            trailing_mean(
                history,
                "carries",
                5,
            ),

        "carries_last":
            lambda:
            last_value(
                history,
                "carries",
            ),

        "opportunities_avg_5":
            lambda:
            trailing_mean(
                history,
                "opportunities",
                5,
            ),

        "opportunities_last":
            lambda:
            last_value(
                history,
                "opportunities",
            ),

        "targets_avg_5":
            lambda:
            trailing_mean(
                history,
                "targets",
                5,
            ),

        "snaps_avg_3":
            lambda:
            trailing_mean(
                history,
                "snaps",
                3,
            ),

        "snaps_avg_5":
            lambda:
            trailing_mean(
                history,
                "snaps",
                5,
            ),

        "snap_pct_avg_3":
            lambda:
            trailing_mean(
                history,
                "snap_pct",
                3,
            ),

        "target_share_last":
            lambda:
            last_value(
                history,
                "target_share",
            ),

        "air_yards_share_avg_3":
            lambda:
            trailing_mean(
                history,
                "air_yards_share",
                3,
            ),

        "air_yards_share_last":
            lambda:
            last_value(
                history,
                "air_yards_share",
            ),

        "yards_last":
            lambda:
            last_value(
                history,
                "yards",
            ),

        "fd_per_snap_avg_3":
            lambda:
            trailing_mean(
                history,
                "fd_per_snap",
                3,
            ),

        "fd_per_touch_avg_3":
            lambda:
            trailing_mean(
                history,
                "fd_per_touch",
                3,
            ),

        "yards_per_opportunity_avg_3":
            lambda:
            trailing_mean(
                history,
                "yards_per_opportunity",
                3,
            ),

        "role_expansions_3":
            lambda:
            trailing_sum(
                history,
                "role_expansion",
                3,
            ),

        "starter_usage_games_3":
            lambda:
            trailing_sum(
                history,
                "starter_usage",
                3,
            ),

        "high_usage_games_3":
            lambda:
            trailing_sum(
                history,
                "high_usage",
                3,
            ),

        "established_role_flag":
            lambda:
            calculate_established_role_flag(
                history
            ),
    }

    if feature not in calculations:

        raise RuntimeError(
            f"Unsupported current feature: "
            f"{feature}"
        )

    return calculations[
        feature
    ]()


# =========================================================
# GAME MAP
# =========================================================

def build_game_map(
    slate,
):

    rows = []

    for _, game in slate.iterrows():

        rows.append(
            {
                "game_id":
                    game[
                        "game_id"
                    ],

                "team":
                    game[
                        "away_team"
                    ],

                "opponent_team":
                    game[
                        "home_team"
                    ],

                "home_away":
                    "AWAY",

                "game_date":
                    game[
                        "game_date"
                    ],

                "game_time":
                    game[
                        "game_time"
                    ],
            }
        )

        rows.append(
            {
                "game_id":
                    game[
                        "game_id"
                    ],

                "team":
                    game[
                        "home_team"
                    ],

                "opponent_team":
                    game[
                        "away_team"
                    ],

                "home_away":
                    "HOME",

                "game_date":
                    game[
                        "game_date"
                    ],

                "game_time":
                    game[
                        "game_time"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# BUILD CURRENT PLAYER FEATURES
# =========================================================

def build_current_rows(
    depth,
    slate,
    target_season,
    target_week,
    history,
    core_df,
):

    section(
        "BUILDING CURRENT SLATE PLAYER FEATURES"
    )

    game_map = build_game_map(
        slate
    )

    players = depth.merge(
        game_map,
        how="inner",
        on="team",
        validate="many_to_one",
    )

    features_by_position = {}

    all_core_features = []

    for position in POSITIONS:

        features = (
            core_df[
                core_df[
                    "position"
                ]
                ==
                position
            ]
            .sort_values(
                "core_rank"
            )[
                "feature"
            ]
            .tolist()
        )

        features_by_position[
            position
        ] = features

        for feature in features:

            if feature not in all_core_features:

                all_core_features.append(
                    feature
                )

    # -----------------------------------------------------
    # CANONICAL HISTORY GROUPS
    #
    # This is the critical production correction:
    #
    # current identity_key
    #       ->
    # historical identity_key
    #
    # GSIS is NOT required for history lookup.
    # -----------------------------------------------------

    history_groups = {
        identity_key:
            group
        for identity_key, group
        in history.groupby(
            "identity_key",
            sort=False,
        )
    }

    output_rows = []
    audit_rows = []

    for _, player in players.iterrows():

        position = player[
            "position"
        ]

        identity_key = normalize_key(
            player[
                "identity_key"
            ]
        )

        if identity_key is None:

            audit_rows.append(
                {
                    "team":
                        player[
                            "team"
                        ],

                    "player_name":
                        player[
                            "player_name"
                        ],

                    "position":
                        position,

                    "identity_key":
                        None,

                    "status":
                        "UNRESOLVED_CANONICAL_IDENTITY",

                    "history_games":
                        0,

                    "detail":
                        "",
                }
            )

            continue

        player_history = (
            history_groups.get(
                identity_key,
                history.iloc[
                    0:0
                ],
            )
        )

        history_games = len(
            player_history
        )

        display_name = player[
            "identity_display_name"
        ]

        if pd.isna(
            display_name
        ):

            display_name = player[
                "player_name"
            ]

        resolved_gsis = player[
            "resolved_gsis_id"
        ]

        if pd.isna(
            resolved_gsis
        ):

            resolved_gsis = None

        else:

            resolved_gsis = str(
                resolved_gsis
            )

        row = {

            "season":
                target_season,

            "week":
                target_week,

            "game_id":
                player[
                    "game_id"
                ],

            "game_date":
                player[
                    "game_date"
                ],

            "game_time":
                player[
                    "game_time"
                ],

            "identity_key":
                identity_key,

            "player_id":
                resolved_gsis,

            "player_display_name":
                display_name,

            "position":
                position,

            "team":
                player[
                    "team"
                ],

            "opponent_team":
                player[
                    "opponent_team"
                ],

            "home_away":
                player[
                    "home_away"
                ],

            "depth_rank":
                player[
                    "pos_rank"
                ],

            "depth_snapshot_dt":
                player[
                    "snapshot_dt"
                ],

            "player_history_games":
                history_games,
        }

        # -------------------------------------------------
        # Deterministic output schema.
        # -------------------------------------------------

        for feature in all_core_features:

            row[
                feature
            ] = 0.0

        try:

            for feature in (
                features_by_position[
                    position
                ]
            ):

                row[
                    feature
                ] = calculate_feature(
                    player_history,
                    feature,
                )

        except Exception as exc:

            audit_rows.append(
                {
                    "team":
                        player[
                            "team"
                        ],

                    "player_name":
                        display_name,

                    "position":
                        position,

                    "identity_key":
                        identity_key,

                    "status":
                        "FEATURE_ERROR",

                    "history_games":
                        history_games,

                    "detail":
                        str(
                            exc
                        ),
                }
            )

            continue

        output_rows.append(
            row
        )

        audit_rows.append(
            {
                "team":
                    player[
                        "team"
                    ],

                "player_name":
                    display_name,

                "position":
                    position,

                "identity_key":
                    identity_key,

                "status":
                    (
                        "READY"
                        if history_games
                        >
                        0
                        else
                        "COLD_START"
                    ),

                "history_games":
                    history_games,

                "detail":
                    "",
            }
        )

    output = pd.DataFrame(
        output_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    print(
        f"Depth/player pool rows: "
        f"{len(players)}"
    )

    print(
        f"Current feature rows: "
        f"{len(output)}"
    )

    print()

    if not output.empty:

        print(
            output.groupby(
                "position"
            )
            .size()
            .to_string()
        )

    return (
        players,
        output,
        audit,
    )


# =========================================================
# FEATURE / COVERAGE AUDIT
# =========================================================

def audit_output(
    players,
    output,
    audit,
    core_df,
):

    section(
        "CURRENT SLATE FEATURE AUDIT"
    )

    if output.empty:

        raise RuntimeError(
            "No current feature rows produced."
        )

    depth_rows = len(
        players
    )

    canonical_resolved = int(
        players[
            "identity_key"
        ].notna().sum()
    )

    unresolved_canonical = int(
        players[
            "identity_key"
        ].isna().sum()
    )

    output_rows = len(
        output
    )

    dropped_rows = (
        depth_rows
        -
        output_rows
    )

    duplicates = (
        output.groupby(
            [
                "game_id",
                "identity_key",
                "team",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    duplicates = duplicates[
        duplicates[
            "n"
        ]
        >
        1
    ]

    duplicate_identity_rows = (
        output.groupby(
            [
                "game_id",
                "identity_key",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    duplicate_identity_rows = (
        duplicate_identity_rows[
            duplicate_identity_rows[
                "n"
            ]
            >
            1
        ]
    )

    invalid_positions = int(
        (
            ~output[
                "position"
            ].isin(
                POSITIONS
            )
        ).sum()
    )

    target_present = int(
        "target_fanduel_points"
        in output.columns
    )

    missing_core = []
    null_core = []
    nonfinite_core = []

    for position in POSITIONS:

        position_rows = output[
            output[
                "position"
            ]
            ==
            position
        ]

        features = (
            core_df[
                core_df[
                    "position"
                ]
                ==
                position
            ][
                "feature"
            ]
            .tolist()
        )

        for feature in features:

            if feature not in output.columns:

                missing_core.append(
                    (
                        position,
                        feature,
                    )
                )

                continue

            values = pd.to_numeric(
                position_rows[
                    feature
                ],
                errors="coerce",
            )

            null_count = int(
                values.isna().sum()
            )

            if null_count:

                null_core.append(
                    (
                        position,
                        feature,
                        null_count,
                    )
                )

            finite = np.isfinite(
                values.fillna(
                    0.0
                )
                .to_numpy(
                    dtype=float
                )
            )

            bad_count = int(
                (
                    ~finite
                ).sum()
            )

            if bad_count:

                nonfinite_core.append(
                    (
                        position,
                        feature,
                        bad_count,
                    )
                )

    # -----------------------------------------------------
    # ESTABLISHED ROLE PARITY AUDIT
    # -----------------------------------------------------

    established_rows = 0

    if (
        "established_role_flag"
        in output.columns
    ):

        rb = output[
            output[
                "position"
            ]
            ==
            "RB"
        ]

        established_rows = int(
            pd.to_numeric(
                rb[
                    "established_role_flag"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    print(
        f"Depth/player pool rows: "
        f"{depth_rows}"
    )

    print(
        f"Canonical identities resolved: "
        f"{canonical_resolved}"
    )

    print(
        f"Unresolved canonical identities: "
        f"{unresolved_canonical}"
    )

    print(
        f"Live feature rows: "
        f"{output_rows}"
    )

    print(
        f"Dropped player rows: "
        f"{dropped_rows}"
    )

    print(
        f"Duplicate player-game rows: "
        f"{len(duplicates)}"
    )

    print(
        f"Duplicate canonical identities "
        f"within game: "
        f"{len(duplicate_identity_rows)}"
    )

    print(
        f"Invalid positions: "
        f"{invalid_positions}"
    )

    print(
        f"Target column present: "
        f"{target_present}"
    )

    print(
        f"Missing core feature groups: "
        f"{len(missing_core)}"
    )

    print(
        f"NULL core feature groups: "
        f"{len(null_core)}"
    )

    print(
        f"Non-finite core feature groups: "
        f"{len(nonfinite_core)}"
    )

    print(
        f"RB established-role rows: "
        f"{established_rows}"
    )

    problems = (
        unresolved_canonical
        +
        dropped_rows
        +
        len(
            duplicates
        )
        +
        len(
            duplicate_identity_rows
        )
        +
        invalid_positions
        +
        target_present
        +
        len(
            missing_core
        )
        +
        len(
            null_core
        )
        +
        len(
            nonfinite_core
        )
    )

    if problems:

        print()

        problem_audit = audit[
            ~audit[
                "status"
            ].isin(
                [
                    "READY",
                    "COLD_START",
                ]
            )
        ]

        if not problem_audit.empty:

            print(
                "Problem audit rows:"
            )

            print()

            print(
                problem_audit.to_string(
                    index=False
                )
            )

        raise RuntimeError(
            "Current slate feature audit failed."
        )

    print()

    print(
        "PASS: live player coverage and "
        "core feature parity passed."
    )


# =========================================================
# COVERAGE REPORT
# =========================================================

def print_coverage(
    output,
    audit,
):

    section(
        "CURRENT SLATE COVERAGE"
    )

    print(
        audit.groupby(
            [
                "position",
                "status",
            ]
        )
        .size()
        .to_string()
    )

    print()

    summary = (
        output.groupby(
            "position"
        )[
            "player_history_games"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        summary.to_string()
    )

    print()

    cold = output[
        output[
            "player_history_games"
        ]
        ==
        0
    ]

    print(
        f"Cold-start rows: "
        f"{len(cold)}"
    )

    if not cold.empty:

        print()

        cold_summary = (
            cold.groupby(
                "position"
            )
            .size()
        )

        print(
            "Cold starts by position:"
        )

        print()

        print(
            cold_summary.to_string()
        )

    print()

    rb = output[
        output[
            "position"
        ]
        ==
        "RB"
    ].copy()

    if (
        not rb.empty
        and
        "established_role_flag"
        in rb.columns
    ):

        established = int(
            rb[
                "established_role_flag"
            ].sum()
        )

        print(
            f"RB established roles: "
            f"{established} / {len(rb)}"
        )


# =========================================================
# EXPORT
# =========================================================

def export_results(
    output,
    audit,
):

    section(
        "EXPORTING CURRENT SLATE FEATURES"
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    output.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    audit.to_csv(
        AUDIT_CSV,
        index=False,
    )

    audit.to_parquet(
        AUDIT_PARQUET,
        index=False,
    )

    print(
        f"Feature rows: "
        f"{len(output)}"
    )

    print(
        f"Audit rows: "
        f"{len(audit)}"
    )

    print()

    print(
        f"Features CSV: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Features Parquet: "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"Audit CSV: "
        f"{AUDIT_CSV}"
    )


# =========================================================
# MAIN
# =========================================================

def run_current_slate_features():

    section(
        "NFL CURRENT SLATE FEATURE ENGINE"
    )

    core_df = load_core_features()

    with get_connection() as conn:

        games = load_games(
            conn
        )

        targets = load_inference_targets(
            conn,
            games,
        )

        groups = target_groups(
            targets
        )

        all_teams = sorted(
            set(
                targets[
                    "away_team"
                ].dropna()
            )
            |
            set(
                targets[
                    "home_team"
                ].dropna()
            )
        )

        depth = load_current_depth(
            conn,
            all_teams,
        )

        depth = attach_identity(
            conn,
            depth,
        )

        (
            history,
            usage_table,
        ) = load_player_history(
            conn,
            games,
        )

    mapping = resolve_history_columns(
        history,
        usage_table,
    )

    history = normalize_history(
        history,
        mapping,
    )

    player_frames = []
    output_frames = []
    audit_frames = []

    for (
        target_season,
        target_week,
        slate,
        teams,
    ) in groups:

        section(
            f"BUILD TARGET "
            f"{target_season} WEEK {target_week}"
        )

        target_depth = depth[
            depth[
                "team"
            ].isin(
                teams
            )
        ].copy()

        target_depth = attach_current_roster_position(
            target_depth,
            target_season,
            target_week,
        )

        (
            players,
            output,
            audit,
        ) = build_current_rows(
            target_depth,
            slate,
            target_season,
            target_week,
            history,
            core_df,
        )

        if not players.empty:

            players[
                "target_season"
            ] = target_season

            players[
                "target_week"
            ] = target_week

        if not audit.empty:

            audit[
                "season"
            ] = target_season

            audit[
                "week"
            ] = target_week

        player_frames.append(
            players
        )

        output_frames.append(
            output
        )

        audit_frames.append(
            audit
        )

    players = pd.concat(
        player_frames,
        ignore_index=True,
    )

    output = pd.concat(
        output_frames,
        ignore_index=True,
    )

    audit = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    output = output.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
            "position",
            "depth_rank",
            "player_display_name",
        ]
    ).reset_index(
        drop=True
    )

    audit_output(
        players,
        output,
        audit,
        core_df,
    )

    print_coverage(
        output,
        audit,
    )

    export_results(
        output,
        audit,
    )

    section(
        "CURRENT SLATE FEATURE BUILD SUCCESSFUL"
    )

    print()
    print(
        "Production safety:"
    )

    print(
        "Inference universe = earliest unfinished "
        "REG week plus exact STAGED_NO_SALARY "
        "FanDuel game IDs."
    )

    print(
        "Future schedule games are not inferred "
        "unless they are explicitly staged by "
        "FanDuel manifest authority."
    )

    print(
        "Each season/week target is built "
        "independently before outputs are combined."
    )

    print(
        "Current players = latest "
        "depth-chart snapshot."
    )

    print(
        "Canonical identity_key is the "
        "player-history join authority."
    )

    print(
        "GSIS is metadata only and is not "
        "required for live history matching."
    )

    print(
        "Historical features use completed "
        "games only."
    )

    print(
        "Efficiency features are consumed "
        "directly from player_weekly_usage."
    )

    print(
        "established_role_flag exactly matches "
        "pregame_features.py semantics:"
    )

    print(
        "snap_pct_avg_3 >= 0.60 OR "
        "opportunities_avg_3 >= 12.0"
    )

    print(
        "No current-game realized target "
        "was created."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Build production_projection.py "
        "per season/week inference target."
    )


if __name__ == "__main__":

    run_current_slate_features()
