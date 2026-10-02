#!/usr/bin/env python3
"""
fanduel_player_pool.py

Week-aware FanDuel offensive player bridge built from the authoritative
salary-bearing slate ingest, not legacy QB/RB/WR/TE/D-ST positional CSVs.

Authority
---------
- fanduel_slate_pool:
    FanDuel contest membership, salary, team, exact season/week/game_id
- player_identity:
    canonical identity and stable football position fallback
- depth_charts:
    latest current-team football position authority
- nfl_production_projection.parquet:
    current multi-target production projection

Safety
------
- No fuzzy player matching.
- No salary imputation.
- No inferred contest membership.
- No global "current week" assumption.
- Exact game_id is required for production attachment.
- Overlapping FanDuel slates remain separate because FanDuel salary may differ
  by slate even for the same player/game.
- DST rows are retained but remain pending the separate DST model.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import CSV_DIR, PARQUET_DIR, DATABASE_PATH


SLATE_TABLE = "fanduel_slate_pool"
OUTPUT_TABLE = "fanduel_player_pool"

OUTPUT_CSV = Path(CSV_DIR) / "nfl_fanduel_player_pool.csv"
OUTPUT_PARQUET = Path(PARQUET_DIR) / "nfl_fanduel_player_pool.parquet"
AUDIT_CSV = Path(CSV_DIR) / "audit_fanduel_player_pool.csv"
AUDIT_PARQUET = Path(PARQUET_DIR) / "audit_fanduel_player_pool.parquet"

PROJECTION_PARQUET = Path(PARQUET_DIR) / "nfl_production_projection.parquet"

OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}

TEAM_ALIASES = {
    "ARZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL",
    "DEN": "DEN", "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND",
    "JAC": "JAX", "JAX": "JAX", "KC": "KC", "LAC": "LAC", "SD": "LAC",
    "LA": "LA", "LAR": "LA", "STL": "LA", "LV": "LV", "OAK": "LV",
    "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SF": "SF",
    "TB": "TB", "TEN": "TEN", "WAS": "WAS", "WSH": "WAS",
}

PLAYER_NAME_ALIASES = {
    "kenneth gainwell": "kenny gainwell",
    "zonovan knight": "bam knight",
    "marquise brown": "hollywood brown",
    "josh palmer": "joshua palmer",
    "chigoziem okonkwo": "chig okonkwo",
    "andrew ogletree": "drew ogletree",
}


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()
    return bool(row[0])


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    ]


def first_existing(columns, candidates):
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    return re.sub(r"\s+", " ", value)


def normalize_name(value):
    value = clean_text(value)
    if value is None:
        return None

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = value.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def identity_name_key(value):
    key = normalize_name(value)
    if key is None:
        return None
    return PLAYER_NAME_ALIASES.get(key, key)


def normalize_team(value):
    value = clean_text(value)
    if value is None:
        return None
    upper = re.sub(r"[^A-Z]", "", value.upper())
    return TEAM_ALIASES.get(upper, upper)


def normalize_position(value):
    value = clean_text(value)
    if value is None:
        return None

    value = value.upper()

    return value if value in OFFENSIVE_POSITIONS else None


def normalize_depth_role(value):
    value = clean_text(value)
    if value is None:
        return None
    value = value.upper()
    return value if value in (OFFENSIVE_POSITIONS | {"FB"}) else None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_slate_pool(conn: sqlite3.Connection) -> pd.DataFrame:
    section("LOADING AUTHORITATIVE FANDUEL SLATE POOL")

    if not table_exists(conn, SLATE_TABLE):
        raise RuntimeError(
            f"{SLATE_TABLE} does not exist. Run fanduel_slate_ingest_v2.py first."
        )

    pool = pd.read_sql_query(
        f'SELECT * FROM "{SLATE_TABLE}"',
        conn,
    )

    if pool.empty:
        raise RuntimeError(
            f"{SLATE_TABLE} is empty."
        )

    required = {
        "player",
        "salary",
        "season",
        "week",
        "game_id",
        "slate_name",
        "slate_slug",
        "is_dst",
    }

    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(
            "Schedule-aware FanDuel slate pool is missing required columns: "
            + ", ".join(missing)
        )

    team_col = first_existing(
        pool.columns,
        ["team_internal", "team"],
    )
    if team_col is None:
        raise RuntimeError(
            "Could not resolve FanDuel team column."
        )

    game_info_col = first_existing(
        pool.columns,
        ["gameInfo", "game_info", "game"],
    )

    source_proj_col = first_existing(
        pool.columns,
        ["source_fantasy_projection", "fantasy"],
    )

    result = pd.DataFrame(index=pool.index)
    result["fd_name"] = pool["player"].apply(clean_text)
    result["team"] = pool[team_col].apply(normalize_team)
    result["salary"] = to_numeric(pool["salary"])
    result["season"] = to_numeric(pool["season"])
    result["week"] = to_numeric(pool["week"])
    result["game_id"] = pool["game_id"].apply(clean_text)
    result["is_dst"] = to_numeric(pool["is_dst"]).fillna(0).astype(int)
    result["slate_name"] = pool["slate_name"].apply(clean_text)
    result["slate_slug"] = pool["slate_slug"].apply(clean_text)

    if game_info_col:
        result["fd_game_info"] = pool[game_info_col].apply(clean_text)
    else:
        result["fd_game_info"] = None

    if source_proj_col:
        result["source_fantasy_projection"] = to_numeric(pool[source_proj_col])
    else:
        result["source_fantasy_projection"] = np.nan

    result["_name_key"] = result["fd_name"].apply(identity_name_key)

    valid_salary = (
        result["salary"].notna()
        & (result["salary"] > 0)
    )

    if not valid_salary.all():
        bad = result.loc[
            ~valid_salary,
            ["fd_name", "team", "salary", "slate_name", "game_id"],
        ]
        raise RuntimeError(
            "fanduel_slate_pool contains non-positive/missing salary rows. "
            "The repaired ingest should expose salary-bearing rows only:\n"
            + bad.head(50).to_string(index=False)
        )

    if result["game_id"].isna().any():
        raise RuntimeError(
            "fanduel_slate_pool contains rows without schedule game_id."
        )

    print(f"Salary-bearing slate rows: {len(result)}")
    print(
        result.groupby(["season", "week", "slate_name"])
        .size()
        .to_string()
    )

    return result


def preserve_slate_specific_rows(pool: pd.DataFrame) -> pd.DataFrame:
    section("PRESERVING SLATE-SPECIFIC FANDUEL SALARIES")

    result = pool.copy()

    key = [
        "slate_slug",
        "fd_name",
        "team",
        "game_id",
        "is_dst",
    ]

    duplicate_counts = (
        result.groupby(
            key,
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
    )

    duplicate_conflicts = duplicate_counts[
        duplicate_counts["rows"] > 1
    ]

    if not duplicate_conflicts.empty:
        raise RuntimeError(
            "Duplicate player/game rows exist inside the same FanDuel slate:\n"
            + duplicate_conflicts.head(100).to_string(index=False)
        )

    # Cross-slate salary differences are valid FanDuel source states.
    cross_slate = (
        result.groupby(
            [
                "fd_name",
                "team",
                "game_id",
                "is_dst",
            ],
            dropna=False,
        )["salary"]
        .nunique(dropna=True)
        .reset_index(name="salary_variants")
    )

    varying = cross_slate[
        cross_slate["salary_variants"] > 1
    ]

    print(f"Slate-specific player/game rows: {len(result)}")
    print(
        "Player/game identities with valid cross-slate salary variation: "
        f"{len(varying)}"
    )

    if not varying.empty:
        print(
            "Cross-slate salary variation is preserved; "
            "no salary is collapsed or averaged."
        )

    return result.reset_index(drop=True)



def load_identity(conn: sqlite3.Connection) -> pd.DataFrame:
    section("LOADING CANONICAL PLAYER IDENTITY")

    if not table_exists(conn, "player_identity"):
        raise RuntimeError(
            "player_identity table does not exist."
        )

    identity = pd.read_sql_query(
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

    if identity.empty:
        raise RuntimeError(
            "player_identity is empty."
        )

    identity["latest_team"] = identity["latest_team"].apply(normalize_team)
    identity["_full_key"] = identity["full_name"].apply(identity_name_key)
    identity["_football_key"] = identity["football_name"].apply(identity_name_key)
    identity["identity_position"] = identity["position"].apply(normalize_position)

    print(f"Canonical identities: {len(identity)}")
    return identity


def unique_map(identity, name_col, use_team=True):
    required = ["identity_key", name_col]
    if use_team:
        required.append("latest_team")

    usable = identity.dropna(subset=required).copy()

    group_cols = [name_col]
    if use_team:
        group_cols = ["latest_team", name_col]

    counts = (
        usable.groupby(group_cols)["identity_key"]
        .nunique()
        .reset_index(name="_identity_count")
    )

    usable = usable.merge(
        counts,
        on=group_cols,
        how="left",
    )

    usable = usable[
        usable["_identity_count"] == 1
    ].copy()

    keep = [
        "identity_key",
        "gsis_id",
        "identity_position",
        "latest_team",
        name_col,
    ]

    return usable[keep].drop_duplicates(group_cols)


def identity_pass(pool, identity, name_col, method, use_team=True):
    result = pool.copy()
    mapping = unique_map(
        identity,
        name_col,
        use_team=use_team,
    ).rename(
        columns={
            "identity_key": "_candidate_identity",
            "gsis_id": "_candidate_gsis_id",
            "identity_position": "_candidate_identity_position",
            "latest_team": "_candidate_team",
            name_col: "_candidate_name",
        }
    )

    unresolved = result["identity_key"].isna()
    if not unresolved.any():
        return result

    left_cols = ["_name_key"]
    left_on = ["_name_key"]
    right_on = ["_candidate_name"]

    if use_team:
        left_cols = ["team", "_name_key"]
        left_on = ["team", "_name_key"]
        right_on = ["_candidate_team", "_candidate_name"]

    candidates = (
        result.loc[unresolved, left_cols]
        .reset_index()
        .merge(
            mapping,
            how="left",
            left_on=left_on,
            right_on=right_on,
        )
        .set_index("index")
    )

    matched = candidates["_candidate_identity"].notna()
    indices = candidates.loc[matched].index

    result.loc[indices, "identity_key"] = candidates.loc[
        indices, "_candidate_identity"
    ]
    result.loc[indices, "resolved_gsis_id"] = candidates.loc[
        indices, "_candidate_gsis_id"
    ]
    result.loc[indices, "identity_position"] = candidates.loc[
        indices, "_candidate_identity_position"
    ]
    result.loc[indices, "identity_match_method"] = method

    return result


def attach_identity(pool: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    section("RESOLVING FANDUEL PLAYERS TO CANONICAL IDENTITY")

    result = pool.copy()
    result["identity_key"] = None
    result["resolved_gsis_id"] = None
    result["identity_position"] = None
    result["identity_match_method"] = None

    offense = result["is_dst"] == 0
    working = result.loc[offense].copy()

    working = identity_pass(
        working,
        identity,
        "_full_key",
        "TEAM_FULL_NAME_EXACT",
        use_team=True,
    )
    working = identity_pass(
        working,
        identity,
        "_football_key",
        "TEAM_FOOTBALL_NAME_EXACT",
        use_team=True,
    )
    working = identity_pass(
        working,
        identity,
        "_full_key",
        "GLOBAL_FULL_NAME_UNIQUE_EXACT",
        use_team=False,
    )
    working = identity_pass(
        working,
        identity,
        "_football_key",
        "GLOBAL_FOOTBALL_NAME_UNIQUE_EXACT",
        use_team=False,
    )

    for column in [
        "identity_key",
        "resolved_gsis_id",
        "identity_position",
        "identity_match_method",
    ]:
        result.loc[working.index, column] = working[column]

    print(f"Offensive bridge rows: {len(working)}")
    print(f"Resolved identities: {int(working['identity_key'].notna().sum())}")
    print(f"Unresolved identities: {int(working['identity_key'].isna().sum())}")

    print()
    print("Identity match methods:")
    print(
        working["identity_match_method"]
        .fillna("UNRESOLVED")
        .value_counts()
        .to_string()
    )

    return result


def load_latest_depth(conn: sqlite3.Connection) -> pd.DataFrame:
    section("LOADING LATEST DEPTH-CHART POSITION AUTHORITY")

    if not table_exists(conn, "depth_charts"):
        raise RuntimeError(
            "depth_charts table does not exist."
        )

    cols = table_columns(conn, "depth_charts")

    snapshot_col = first_existing(cols, ["snapshot_dt", "dt"])
    team_col = first_existing(cols, ["team"])
    name_col = first_existing(cols, ["player_name", "full_name", "name"])
    position_col = first_existing(cols, ["position", "pos_abb", "pos_name"])

    if not all([snapshot_col, team_col, name_col, position_col]):
        raise RuntimeError(
            "depth_charts is missing snapshot/team/name/position authority."
        )

    depth = pd.read_sql_query(
        f'''SELECT
            "{snapshot_col}" AS snapshot_dt,
            "{team_col}" AS team,
            "{name_col}" AS player_name,
            "{position_col}" AS position
        FROM depth_charts''',
        conn,
    )

    if depth.empty:
        raise RuntimeError(
            "depth_charts is empty."
        )

    depth["snapshot_dt"] = pd.to_datetime(
        depth["snapshot_dt"],
        errors="coerce",
    )
    depth["team"] = depth["team"].apply(normalize_team)
    depth["_name_key"] = depth["player_name"].apply(identity_name_key)
    depth["depth_role"] = depth["position"].apply(normalize_depth_role)

    latest = (
        depth.groupby("team")["snapshot_dt"]
        .transform("max")
    )

    depth = depth[
        depth["snapshot_dt"] == latest
    ].copy()

    depth = depth[
        depth["depth_role"].notna()
    ].copy()

    counts = (
        depth.groupby(["team", "_name_key"])["depth_role"]
        .nunique()
        .reset_index(name="_position_count")
    )

    depth = depth.merge(
        counts,
        on=["team", "_name_key"],
        how="left",
    )

    depth = depth[
        depth["_position_count"] == 1
    ].copy()

    depth = (
        depth.sort_values(
            ["team", "_name_key", "snapshot_dt"]
        )
        .drop_duplicates(
            ["team", "_name_key"],
            keep="last",
        )
    )

    print(f"Latest usable depth rows: {len(depth)}")
    return depth[
        ["team", "_name_key", "depth_role"]
    ].copy()


def load_weekly_roster_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    section("LOADING WEEKLY ROSTER POSITION AUTHORITY")

    if not table_exists(conn, "weekly_rosters"):
        raise RuntimeError(
            "weekly_rosters table does not exist."
        )

    roster = pd.read_sql_query(
        """
        SELECT
            season,
            week,
            team,
            gsis_id,
            position AS roster_position,
            depth_chart_position AS roster_depth_position,
            status AS roster_status,
            status_description_abbr AS roster_status_abbr
        FROM weekly_rosters
        """,
        conn,
    )

    roster["team"] = roster["team"].apply(normalize_team)
    roster["roster_position"] = roster["roster_position"].apply(normalize_position)
    roster = roster[roster["gsis_id"].notna()].copy()

    conflicts = (
        roster.groupby(["season", "week", "team", "gsis_id"])["roster_position"]
        .nunique(dropna=False)
        .reset_index(name="position_count")
    )
    conflicts = conflicts[conflicts["position_count"] > 1]
    if not conflicts.empty:
        raise RuntimeError(
            "Conflicting weekly roster positions for exact season/week/team/GSIS rows:\n"
            + conflicts.head(100).to_string(index=False)
        )

    roster = roster.drop_duplicates(
        ["season", "week", "team", "gsis_id"],
        keep="last",
    )

    print(f"Weekly roster position rows: {len(roster)}")
    return roster


def attach_position(
    pool: pd.DataFrame,
    depth: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    section("RESOLVING FANDUEL ROSTER POSITION")

    result = pool.copy()

    result = result.merge(
        depth,
        how="left",
        on=["team", "_name_key"],
        validate="many_to_one",
        sort=False,
    )

    result = result.merge(
        roster,
        how="left",
        left_on=["season", "week", "team", "resolved_gsis_id"],
        right_on=["season", "week", "team", "gsis_id"],
        validate="many_to_one",
        sort=False,
    )

    offense = result["is_dst"] == 0

    result["fd_position"] = np.where(
        result["is_dst"] == 1,
        "DST",
        result["roster_position"],
    )

    depth_fallback = (
        offense
        & result["fd_position"].isna()
        & result["depth_role"].isin(OFFENSIVE_POSITIONS)
    )
    result.loc[depth_fallback, "fd_position"] = result.loc[
        depth_fallback, "depth_role"
    ]

    identity_fallback = (
        offense
        & result["fd_position"].isna()
        & result["identity_position"].isin(OFFENSIVE_POSITIONS)
    )
    result.loc[identity_fallback, "fd_position"] = result.loc[
        identity_fallback, "identity_position"
    ]

    result["position_source"] = np.where(
        result["is_dst"] == 1,
        "FANDUEL_DST",
        np.where(
            result["roster_position"].notna(),
            "WEEKLY_ROSTER_EXACT_GSIS",
            np.where(
                depth_fallback,
                "LATEST_DEPTH_STANDARD_POSITION_FALLBACK",
                np.where(
                    identity_fallback,
                    "PLAYER_IDENTITY_FALLBACK",
                    None,
                ),
            ),
        ),
    )

    invalid = (
        offense
        & ~result["fd_position"].isin(OFFENSIVE_POSITIONS)
    )

    print(
        "Exact weekly-roster positions: "
        f"{int((offense & result['roster_position'].notna()).sum())}"
    )
    print(
        "Standard depth-position fallbacks: "
        f"{int(depth_fallback.sum())}"
    )
    print(
        "Identity position fallbacks: "
        f"{int(identity_fallback.sum())}"
    )
    print(
        "Offensive positions resolved: "
        f"{int((offense & ~invalid).sum())}/{int(offense.sum())}"
    )
    print(
        f"Unresolved offensive positions: {int(invalid.sum())}"
    )

    if invalid.any():
        print()
        print("Unresolved position rows:")
        print(
            result.loc[
                invalid,
                [
                    "fd_name",
                    "team",
                    "salary",
                    "game_id",
                    "identity_key",
                    "identity_match_method",
                ],
            ].to_string(index=False)
        )

    return result


def load_projection() -> pd.DataFrame:
    section("LOADING MULTI-TARGET PRODUCTION PROJECTION")

    if not PROJECTION_PARQUET.exists():
        raise RuntimeError(
            f"Missing projection file: {PROJECTION_PARQUET}"
        )

    projection = pd.read_parquet(
        PROJECTION_PARQUET
    ).copy()

    required = {
        "season",
        "week",
        "game_id",
        "identity_key",
        "position",
        "team",
        "opponent_team",
        "production_status",
        "ridge_projection",
    }

    missing = sorted(required - set(projection.columns))
    if missing:
        raise RuntimeError(
            "Production projection missing required columns: "
            + ", ".join(missing)
        )

    projection["team"] = projection["team"].apply(normalize_team)
    projection["position"] = projection["position"].apply(normalize_position)

    duplicate = (
        projection[
            projection["identity_key"].notna()
        ]
        .groupby(["identity_key", "team", "game_id"])
        .size()
        .reset_index(name="rows")
    )

    duplicate = duplicate[
        duplicate["rows"] > 1
    ]

    if not duplicate.empty:
        raise RuntimeError(
            "Production projection has duplicate identity/team/game rows:\n"
            + duplicate.to_string(index=False)
        )

    print(f"Projection rows: {len(projection)}")
    print(
        projection.groupby(
            ["season", "week", "production_status"]
        ).size().to_string()
    )

    return projection


def attach_projection(pool: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    section("ATTACHING EXACT PLAYER/GAME PRODUCTION PROJECTION")

    result = pool.copy()

    model = projection.copy().rename(
        columns={
            "team": "model_team",
            "position": "model_position",
            "game_id": "model_game_id",
            "season": "model_season",
            "week": "model_week",
            "opponent_team": "model_opponent_team",
        }
    )

    result = result.merge(
        model,
        how="left",
        left_on=[
            "identity_key",
            "team",
            "game_id",
        ],
        right_on=[
            "identity_key",
            "model_team",
            "model_game_id",
        ],
        validate="many_to_one",
        sort=False,
    )

    attached = result["model_game_id"].notna()

    result["opponent_team"] = np.where(
        attached,
        result["model_opponent_team"],
        None,
    )

    print(
        "Exact offensive model attachments: "
        f"{int(attached.sum())}"
    )

    print()
    print("Attachments by FanDuel target season/week:")
    if attached.any():
        print(
            result.loc[attached]
            .groupby(["season", "week"])
            .size()
            .to_string()
        )
    else:
        print("NONE")

    return result


def assign_status(pool: pd.DataFrame) -> pd.DataFrame:
    section("ASSIGNING FANDUEL PLAYER-POOL STATUS")

    result = pool.copy()
    result["player_pool_status"] = None
    result["optimizer_eligible"] = 0
    result["live_projection"] = result["injury_adjusted_projection"].fillna(result["ridge_projection"])

    dst = result["is_dst"] == 1
    offense = ~dst

    result.loc[
        dst,
        "player_pool_status",
    ] = "DST_PENDING_MODEL"

    unresolved_identity = (
        offense
        & result["identity_key"].isna()
    )
    result.loc[
        unresolved_identity,
        "player_pool_status",
    ] = "UNRESOLVED_IDENTITY"

    unresolved_position = (
        offense
        & result["identity_key"].notna()
        & ~result["fd_position"].isin(OFFENSIVE_POSITIONS)
    )
    result.loc[
        unresolved_position,
        "player_pool_status",
    ] = "UNRESOLVED_POSITION"

    no_projection = (
        offense
        & result["identity_key"].notna()
        & result["fd_position"].isin(OFFENSIVE_POSITIONS)
        & result["model_game_id"].isna()
    )
    result.loc[
        no_projection,
        "player_pool_status",
    ] = "NO_CURRENT_PROJECTION"

    cold = (
        offense
        & result["model_game_id"].notna()
        & (result["production_status"] == "COLD_START")
    )
    result.loc[
        cold,
        "player_pool_status",
    ] = "COLD_START"

    ready = (
        offense
        & result["model_game_id"].notna()
        & (result["production_status"] == "MODEL_READY")
        & result["ridge_projection"].notna()
        & result["fd_position"].isin(OFFENSIVE_POSITIONS)
    )
    result.loc[
        ready,
        "player_pool_status",
    ] = "MODEL_READY"

    # QB optimizer eligibility requires the current primary
    # depth-chart lane. Backup QBs may remain MODEL_READY
    # for projection/forecast purposes but are not optimizer
    # eligible merely because a production projection exists.
    qb = result["fd_position"].eq("QB")

    eligible_ready = (
        ready
        & (
            ~qb
            | result["depth_rank"].eq(1)
        )
    )

    result.loc[
        eligible_ready,
        "optimizer_eligible",
    ] = 1

    print(
        result.groupby(
            ["fd_position", "player_pool_status"],
            dropna=False,
        ).size().to_string()
    )

    return result


def build_audit(pool: pd.DataFrame) -> pd.DataFrame:
    section("FANDUEL PLAYER POOL AUDIT")

    rows = []

    def add(metric, value):
        rows.append(
            {
                "metric": metric,
                "value": int(value),
            }
        )

    offense = pool["is_dst"] == 0

    add("bridge_rows", len(pool))
    add("offensive_rows", offense.sum())
    add("dst_rows", (~offense).sum())
    add("invalid_salary_rows", (
        pool["salary"].isna() | (pool["salary"] <= 0)
    ).sum())
    add("missing_game_id_rows", pool["game_id"].isna().sum())
    add("offensive_unresolved_identity", (
        offense & pool["identity_key"].isna()
    ).sum())
    add("offensive_unresolved_position", (
        offense & ~pool["fd_position"].isin(OFFENSIVE_POSITIONS)
    ).sum())
    add("model_ready_rows", (
        pool["player_pool_status"] == "MODEL_READY"
    ).sum())
    add("cold_start_rows", (
        pool["player_pool_status"] == "COLD_START"
    ).sum())
    add("no_current_projection_rows", (
        pool["player_pool_status"] == "NO_CURRENT_PROJECTION"
    ).sum())
    add("optimizer_eligible_rows", (
        pool["optimizer_eligible"] == 1
    ).sum())
    add("eligible_unresolved_position", (
        (pool["optimizer_eligible"] == 1)
        & ~pool["fd_position"].isin(OFFENSIVE_POSITIONS)
    ).sum())
    add("eligible_null_projection", (
        (pool["optimizer_eligible"] == 1)
        & pool["ridge_projection"].isna()
    ).sum())
    add("eligible_backup_qb_rows", (
        (pool["optimizer_eligible"] == 1)
        & pool["fd_position"].eq("QB")
        & pool["depth_rank"].gt(1)
    ).sum())
    add("eligible_game_mismatch", (
        (pool["optimizer_eligible"] == 1)
        & (
            pool["game_id"].astype("string")
            != pool["model_game_id"].astype("string")
        )
    ).sum())
    add("eligible_position_mismatch", (
        (pool["optimizer_eligible"] == 1)
        & pool["model_position"].notna()
        & (
            pool["fd_position"].astype("string")
            != pool["model_position"].astype("string")
        )
    ).sum())

    audit = pd.DataFrame(rows)
    print(audit.to_string(index=False))

    fatal_metrics = {
        "invalid_salary_rows",
        "missing_game_id_rows",
        "eligible_unresolved_position",
        "eligible_null_projection",
        "eligible_backup_qb_rows",
        "eligible_game_mismatch",
        "eligible_position_mismatch",
    }

    fatal = audit[
        audit["metric"].isin(fatal_metrics)
        & (audit["value"] > 0)
    ]

    if not fatal.empty:
        raise RuntimeError(
            "Fatal FanDuel player-pool integrity failure:\n"
            + fatal.to_string(index=False)
        )

    print()
    print(
        "PASS: salary, schedule identity, and optimizer-eligible "
        "position/model joins passed."
    )
    print(
        "Rows with unresolved roster position remain quarantined "
        "and optimizer-ineligible."
    )

    return audit


def clean_output(pool: pd.DataFrame) -> pd.DataFrame:
    result = pool.copy()

    result = result.drop(
        columns=[
            "_name_key",
            "depth_position",
        ],
        errors="ignore",
    )

    result = result.sort_values(
        [
            "season",
            "week",
            "slate_slug",
            "game_id",
            "fd_position",
            "team",
            "salary",
            "fd_name",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    return result


def export_outputs(conn, pool: pd.DataFrame, audit: pd.DataFrame) -> None:
    section("EXPORTING FANDUEL PLAYER POOL")

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)

    output = clean_output(pool)

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

    output.to_sql(
        OUTPUT_TABLE,
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute(
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_game '
        f'ON "{OUTPUT_TABLE}"(game_id)'
    )
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_identity_game '
        f'ON "{OUTPUT_TABLE}"(identity_key, game_id)'
    )
    conn.commit()

    print(f"Player bridge rows: {len(output)}")
    print(f"SQLite table: {OUTPUT_TABLE}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Parquet: {OUTPUT_PARQUET}")
    print(f"Audit CSV: {AUDIT_CSV}")


def main():
    section("NFL FANDUEL SLATE-DRIVEN PLAYER POOL ENGINE")

    print("FanDuel source: fanduel_slate_pool")
    print("Legacy data/fanduel/QB.csv, RB.csv, WR.csv, TE.csv, D-ST.csv are NOT used.")
    print("FanDuel salary and contest membership come only from schedule-aware slate ingest.")
    print("Position comes from exact weekly roster GSIS authority with standard depth/identity fallback.")
    print("Production joins require canonical identity + team + exact game_id.")

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        slate_pool = load_slate_pool(conn)
        bridge = preserve_slate_specific_rows(slate_pool)

        identity = load_identity(conn)
        bridge = attach_identity(
            bridge,
            identity,
        )

        depth = load_latest_depth(conn)
        roster = load_weekly_roster_positions(conn)
        bridge = attach_position(
            bridge,
            depth,
            roster,
        )

        projection = load_projection()
        bridge = attach_projection(
            bridge,
            projection,
        )

        bridge = assign_status(
            bridge
        )

        audit = build_audit(
            bridge
        )

        export_outputs(
            conn,
            bridge,
            audit,
        )

    finally:
        conn.close()

    section("FANDUEL PLAYER POOL BUILD SUCCESSFUL")

    print()
    print("Full FanDuel slate ingest is the only contest/salary authority.")
    print("Overlapping slates remain separate so each slate keeps its exact FanDuel salary.")
    print("No cross-slate salary is collapsed, averaged, or inferred.")
    print("No fuzzy player matching is performed.")
    print("No salary is imputed.")
    print(
        "Unresolved QB/RB/WR/TE roster positions are quarantined "
        "instead of guessed."
    )
    print("DST remains pending the separate DST projection layer.")
    print("Staged no-salary slates remain outside optimizer eligibility until salary arrives.")


if __name__ == "__main__":
    main()
