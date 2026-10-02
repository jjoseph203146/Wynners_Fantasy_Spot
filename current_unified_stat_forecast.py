#!/usr/bin/env python3

"""
Build the current schedule-authoritative unified NFL stat forecast.

Contract:
    WFS_UNIFIED_STAT_FORECAST_V1

Authorities:
- offense: nfl_current_offensive_stat_forecasts.parquet
- kicker:  nfl_current_kicker_stat_forecasts.parquet
- DST:     nfl_current_dst_stat_forecasts.parquet
- schedule identity/opponent: nfl.db -> games

Rules:
- earliest unfinished 2026 REG week
- exact game/team identity
- no fuzzy matching
- non-applicable statistics are NULL by design
- no model training
- no projection modification
"""

from __future__ import annotations
from forecast_publication_selector import select_forecast_path

from pathlib import Path
import hashlib
import io
import argparse
import json
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "nfl.db"

OFFENSE_PATH = (
    ROOT / "data/parquet/"
    "nfl_current_offensive_stat_forecasts.parquet"
)

KICKER_PATH = (
    ROOT / "data/parquet/"
    "nfl_current_kicker_stat_forecasts.parquet"
)

DST_PATH = (
    ROOT / "data/parquet/"
    "nfl_current_dst_stat_forecasts.parquet"
)

V3_RECONCILIATION_PATH = (
    ROOT / "processed/"
    "offensive_team_reconciliation_shadow_v3.csv"
)

OUTPUT_PATH = (
    ROOT / "data/parquet/"
    "nfl_current_unified_stat_forecasts.parquet"
)

SEASON = 2026

COLUMNS = [
    "entity_type",
    "game_id",
    "team",
    "opponent_team",
    "player_id",
    "entity_name",
    "position",
    "model_group",
    "expected_attempts",
    "expected_carries",
    "expected_completions",
    "expected_defensive_tds",
    "expected_fga",
    "expected_fgm",
    "expected_fumble_recoveries",
    "expected_interceptions",
    "expected_passing_tds",
    "expected_passing_yards",
    "expected_points_allowed",
    "expected_receiving_tds",
    "expected_receiving_yards",
    "expected_receptions",
    "expected_rushing_tds",
    "expected_rushing_yards",
    "expected_sacks",
    "expected_targets",
    "expected_xpa",
    "expected_xpm",
]

OFFENSE_STATS = [
    "expected_attempts",
    "expected_carries",
    "expected_completions",
    "expected_interceptions",
    "expected_passing_tds",
    "expected_passing_yards",
    "expected_receiving_tds",
    "expected_receiving_yards",
    "expected_receptions",
    "expected_rushing_tds",
    "expected_rushing_yards",
    "expected_targets",
]

KICKER_STATS = [
    "expected_fga",
    "expected_fgm",
    "expected_xpa",
    "expected_xpm",
]

DST_STATS = [
    "expected_defensive_tds",
    "expected_fumble_recoveries",
    "expected_interceptions",
    "expected_points_allowed",
    "expected_sacks",
]

REQUIRED_STATS = sorted(
    set(
        OFFENSE_STATS
        + KICKER_STATS
        + DST_STATS
    )
)


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)
    tmp = Path(tmp_name)

    try:
        df.to_parquet(
            tmp,
            index=False,
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def active_schedule() -> tuple[int, pd.DataFrame]:
    with sqlite3.connect(DB) as conn:
        games = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_type,
                home_team,
                away_team,
                completed
            FROM games
            WHERE season = ?
              AND game_type = 'REG'
            ORDER BY week, game_id
            """,
            conn,
            params=(SEASON,),
        )

    if games.empty:
        raise RuntimeError(
            "NO_CURRENT_REGULAR_SEASON_SCHEDULE"
        )

    games["week"] = pd.to_numeric(
        games["week"],
        errors="coerce",
    )

    completed = pd.to_numeric(
        games["completed"],
        errors="coerce",
    ).fillna(0)

    unfinished = games[
        completed.ne(1)
    ].copy()

    if unfinished.empty:
        raise RuntimeError(
            "NO_UNFINISHED_REGULAR_SEASON_GAMES"
        )

    week = int(
        unfinished["week"].min()
    )

    target = unfinished[
        unfinished["week"].eq(week)
    ].copy()

    if target["game_id"].duplicated().any():
        raise RuntimeError(
            "SCHEDULE_DUPLICATE_GAME_ID"
        )

    return week, target


def schedule_teams(
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    home = schedule[
        ["game_id", "home_team", "away_team"]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent_team",
        }
    )

    away = schedule[
        ["game_id", "away_team", "home_team"]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent_team",
        }
    )

    teams = pd.concat(
        [home, away],
        ignore_index=True,
    )

    for col in ["team", "opponent_team"]:
        teams[col] = (
            teams[col]
            .astype(str)
            .str.strip()
        )

    if teams[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "SCHEDULE_TEAM_IDENTITY_DUPLICATES"
        )

    return teams


def load_component(
    path: Path,
    week: int,
    label: str,
    verified_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if verified_frame is None and not path.is_file():
        raise FileNotFoundError(
            f"MISSING_{label}_ARTIFACT:{path}"
        )

    df = verified_frame.copy() if verified_frame is not None else pd.read_parquet(path)

    required = {
        "game_id",
        "team",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label}_IDENTITY_SCHEMA_MISSING:"
            + ",".join(missing)
        )

    if "season" in df.columns:
        season_num = pd.to_numeric(
            df["season"],
            errors="coerce",
        )

        df = df[
            season_num.eq(SEASON)
        ].copy()

    if "week" in df.columns:
        week_num = pd.to_numeric(
            df["week"],
            errors="coerce",
        )

        df = df[
            week_num.eq(week)
        ].copy()

    if df.empty:
        raise RuntimeError(
            f"{label}_ACTIVE_WEEK_EMPTY"
        )

    df["team"] = (
        df["team"]
        .astype(str)
        .str.strip()
    )

    return df


def attach_schedule(
    df: pd.DataFrame,
    teams: pd.DataFrame,
    label: str,
) -> pd.DataFrame:

    # Explicit, deterministic aliases between component
    # conventions and authoritative nfl.db schedule codes.
    #
    # No fuzzy matching.
    TEAM_ALIASES = {
        "LAR": "LA",
        "WSH": "WAS",
        "JAC": "JAX",
    }

    df = df.copy()

    df["team"] = (
        df["team"]
        .astype(str)
        .str.strip()
        .replace(TEAM_ALIASES)
    )

    # If the component already carries an opponent,
    # canonicalize it before comparing to schedule authority.
    if "opponent_team" in df.columns:
        df["opponent_team"] = (
            df["opponent_team"]
            .astype("string")
            .str.strip()
            .replace(TEAM_ALIASES)
        )

    valid_games = set(
        teams["game_id"].astype(str)
    )

    df = df[
        df["game_id"].astype(str).isin(
            valid_games
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"{label}_NO_ACTIVE_SCHEDULE_ROWS"
        )

    expected = set(
        map(
            tuple,
            teams[
                ["game_id", "team"]
            ].astype(str).to_numpy(),
        )
    )

    actual = set(
        map(
            tuple,
            df[
                ["game_id", "team"]
            ].astype(str).to_numpy(),
        )
    )

    extra = sorted(
        actual - expected
    )

    if extra:
        raise RuntimeError(
            f"{label}_SCHEDULE_IDENTITY_EXTRA:{extra}"
        )

    # Preserve an existing component opponent so it can
    # be validated against schedule authority.
    if "opponent_team" in df.columns:
        df = df.rename(
            columns={
                "opponent_team":
                    "component_opponent_team"
            }
        )

    df = df.merge(
        teams[
            [
                "game_id",
                "team",
                "opponent_team",
            ]
        ],
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    if df["opponent_team"].isna().any():
        raise RuntimeError(
            f"{label}_OPPONENT_RESOLUTION_FAILURE"
        )

    if "component_opponent_team" in df.columns:
        component = (
            df["component_opponent_team"]
            .astype("string")
            .str.strip()
        )

        schedule_opponent = (
            df["opponent_team"]
            .astype("string")
            .str.strip()
        )

        present = component.notna()

        mismatch = (
            present
            & component.ne(schedule_opponent)
        )

        if mismatch.any():
            bad = df.loc[
                mismatch,
                [
                    "game_id",
                    "team",
                    "component_opponent_team",
                    "opponent_team",
                ],
            ].to_dict("records")

            raise RuntimeError(
                f"{label}_OPPONENT_SCHEDULE_MISMATCH:"
                f"{bad}"
            )

        df = df.drop(
            columns=[
                "component_opponent_team"
            ]
        )

    return df

def blank_frame(
    rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: [np.nan] * rows
            for col in COLUMNS
        }
    )


def first_existing(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        f"{label}_MISSING_COLUMNS:"
        + "|".join(candidates)
    )


def apply_v3_offensive_reconciliation(
    offense: pd.DataFrame,
    validated_v3: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Overlay validated WFS offensive reconciliation V3 component
    statistics onto the current offensive stat forecast.

    Exact identity:
        game_id + player_id

    Only V3 reconciliation-pool roles are authoritative:
        PRIMARY_QB
        ACTIVE_ROTATION

    The upstream offensive forecast remains authoritative for rows
    outside that V3 pool. No fuzzy matching is permitted.
    """
    if validated_v3 is None and not V3_RECONCILIATION_PATH.is_file():
        raise RuntimeError(
            "V3_RECONCILIATION_SOURCE_MISSING:"
            + str(V3_RECONCILIATION_PATH)
        )

    v3 = validated_v3.copy() if validated_v3 is not None else pd.read_csv(
        V3_RECONCILIATION_PATH, low_memory=False,
    )

    required_v3 = {
        "game_id",
        "player_id",
        "reconciliation_role",
        "reconciled_attempts",
        "reconciled_completions",
        "reconciled_passing_yards",
        "reconciled_passing_tds",
        "reconciled_interceptions",
        "reconciled_carries",
        "reconciled_rushing_yards",
        "reconciled_rushing_tds",
        "reconciled_targets",
        "reconciled_receptions",
        "reconciled_receiving_yards",
        "reconciled_receiving_tds",
    }

    missing = sorted(
        required_v3 - set(v3.columns)
    )

    if missing:
        raise RuntimeError(
            "V3_RECONCILIATION_COLUMNS_MISSING:"
            + ",".join(missing)
        )

    player_col = first_existing(
        offense,
        ["player_id", "gsis_id"],
        "OFFENSE_PLAYER_ID",
    )

    work = offense.copy()

    work["game_id"] = (
        work["game_id"]
        .astype(str)
        .str.strip()
    )
    work[player_col] = (
        work[player_col]
        .astype(str)
        .str.strip()
    )

    v3 = v3.copy()

    v3["game_id"] = (
        v3["game_id"]
        .astype(str)
        .str.strip()
    )
    v3["player_id"] = (
        v3["player_id"]
        .astype(str)
        .str.strip()
    )
    v3["reconciliation_role"] = (
        v3["reconciliation_role"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    v3 = v3[
        v3["reconciliation_role"].isin(
            {"PRIMARY_QB", "ACTIVE_ROTATION"}
        )
    ].copy()

    if v3.empty:
        raise RuntimeError(
            "V3_RECONCILIATION_ZERO_AUTHORITY_ROWS"
        )

    if v3[
        ["game_id", "player_id"]
    ].duplicated().any():
        raise RuntimeError(
            "V3_RECONCILIATION_IDENTITY_DUPLICATES"
        )

    mapping = {
        "reconciled_attempts":
            "expected_attempts",
        "reconciled_completions":
            "expected_completions",
        "reconciled_passing_yards":
            "expected_passing_yards",
        "reconciled_passing_tds":
            "expected_passing_tds",
        "reconciled_interceptions":
            "expected_interceptions",
        "reconciled_carries":
            "expected_carries",
        "reconciled_rushing_yards":
            "expected_rushing_yards",
        "reconciled_rushing_tds":
            "expected_rushing_tds",
        "reconciled_targets":
            "expected_targets",
        "reconciled_receptions":
            "expected_receptions",
        "reconciled_receiving_yards":
            "expected_receiving_yards",
        "reconciled_receiving_tds":
            "expected_receiving_tds",
    }

    keep = [
        "game_id",
        "player_id",
        "reconciliation_role",
        *mapping.keys(),
    ]

    v3 = v3[keep].copy()

    for source in mapping:
        v3[source] = pd.to_numeric(
            v3[source],
            errors="coerce",
        )

    merged = work.merge(
        v3,
        how="left",
        left_on=["game_id", player_col],
        right_on=["game_id", "player_id"],
        validate="one_to_one",
        suffixes=("", "_v3_identity"),
    )

    authority = (
        merged["reconciliation_role"]
        .notna()
    )

    matched = int(authority.sum())

    if matched == 0:
        raise RuntimeError(
            "V3_RECONCILIATION_ZERO_OFFENSE_MATCHES"
        )

    applicability = {
        "QB": {
            "expected_attempts",
            "expected_carries",
            "expected_completions",
            "expected_interceptions",
            "expected_passing_tds",
            "expected_passing_yards",
            "expected_rushing_tds",
            "expected_rushing_yards",
        },
        "RB_FB": {
            "expected_carries",
            "expected_receiving_tds",
            "expected_receiving_yards",
            "expected_receptions",
            "expected_rushing_tds",
            "expected_rushing_yards",
            "expected_targets",
        },
        "WR": {
            "expected_receiving_tds",
            "expected_receiving_yards",
            "expected_receptions",
            "expected_targets",
        },
        "TE": {
            "expected_receiving_tds",
            "expected_receiving_yards",
            "expected_receptions",
            "expected_targets",
        },
    }

    if "model_group" not in merged.columns:
        raise RuntimeError(
            "V3_MODEL_GROUP_MISSING"
        )

    groups = (
        merged["model_group"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    authority_groups = set(
        groups.loc[authority].unique()
    )

    unknown_groups = sorted(
        authority_groups - set(applicability)
    )

    if unknown_groups:
        raise RuntimeError(
            "V3_UNKNOWN_MODEL_GROUP:"
            + ",".join(unknown_groups)
        )

    applied_cells = 0

    for source, target in mapping.items():
        if target not in merged.columns:
            raise RuntimeError(
                "V3_TARGET_COLUMN_MISSING:"
                + target
            )

        applicable = authority & groups.map(
            lambda group: (
                target in applicability.get(
                    group,
                    set(),
                )
            )
        )

        if not applicable.any():
            continue

        values = pd.to_numeric(
            merged.loc[applicable, source],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                "V3_RECONCILIATION_NULL:"
                + source
            )

        arr = values.to_numpy(dtype=float)

        if not np.isfinite(arr).all():
            raise RuntimeError(
                "V3_RECONCILIATION_NONFINITE:"
                + source
            )

        if (arr < 0).any():
            raise RuntimeError(
                "V3_RECONCILIATION_NEGATIVE:"
                + source
            )

        merged.loc[
            applicable,
            target,
        ] = arr

        applied_cells += int(
            applicable.sum()
        )

    # Verify every applicable V3 value survived exactly.
    max_gap = 0.0

    for source, target in mapping.items():
        applicable = authority & groups.map(
            lambda group: (
                target in applicability.get(
                    group,
                    set(),
                )
            )
        )

        if not applicable.any():
            continue

        left = pd.to_numeric(
            merged.loc[applicable, target],
            errors="coerce",
        ).to_numpy(dtype=float)

        right = pd.to_numeric(
            merged.loc[applicable, source],
            errors="coerce",
        ).to_numpy(dtype=float)

        gap = np.abs(left - right)

        if len(gap):
            max_gap = max(
                max_gap,
                float(np.max(gap)),
            )

    print(
        "V3 offensive reconciliation cells applied:",
        applied_cells,
    )

    if max_gap > 1e-9:
        raise RuntimeError(
            "V3_RECONCILIATION_OVERLAY_MISMATCH:"
            f"{max_gap:.12f}"
        )

    drop_cols = [
        "player_id_v3_identity",
        "reconciliation_role",
        *mapping.keys(),
    ]

    merged = merged.drop(
        columns=[
            c for c in drop_cols
            if c in merged.columns
        ]
    )

    if len(merged) != len(work):
        raise RuntimeError(
            "V3_RECONCILIATION_ROW_COUNT_CHANGED"
        )

    print(
        "V3 offensive reconciliation rows applied:",
        matched,
    )
    print(
        "V3 offensive reconciliation max overlay gap:",
        f"{max_gap:.12f}",
    )

    return merged


def build_offense(
    df: pd.DataFrame,
) -> pd.DataFrame:
    player_col = first_existing(
        df,
        ["player_id", "gsis_id"],
        "OFFENSE_PLAYER_ID",
    )

    name_col = first_existing(
        df,
        ["entity_name", "player_name"],
        "OFFENSE_PLAYER_NAME",
    )

    position_col = first_existing(
        df,
        ["position"],
        "OFFENSE_POSITION",
    )

    model_group_col = first_existing(
        df,
        ["model_group"],
        "OFFENSE_MODEL_GROUP",
    )

    missing_stats = sorted(
        set(OFFENSE_STATS)
        - set(df.columns)
    )

    if missing_stats:
        raise RuntimeError(
            "OFFENSE_STATS_MISSING:"
            + ",".join(missing_stats)
        )

    if df[player_col].isna().any():
        raise RuntimeError(
            "OFFENSE_PLAYER_ID_NULL"
        )

    if df[
        ["game_id", player_col]
    ].duplicated().any():
        raise RuntimeError(
            "OFFENSE_IDENTITY_DUPLICATES"
        )

    out = blank_frame(len(df))

    out["entity_type"] = "OFFENSE_PLAYER"
    out["game_id"] = df["game_id"].to_numpy()
    out["team"] = df["team"].to_numpy()
    out["opponent_team"] = (
        df["opponent_team"].to_numpy()
    )
    out["player_id"] = df[player_col].to_numpy()
    out["entity_name"] = df[name_col].to_numpy()
    out["position"] = df[position_col].to_numpy()
    out["model_group"] = (
        df[model_group_col].to_numpy()
    )

    for col in OFFENSE_STATS:
        out[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy()

    return out


def build_kicker(
    df: pd.DataFrame,
) -> pd.DataFrame:
    player_col = first_existing(
        df,
        [
            "player_id",
            "kicker_player_id",
            "gsis_id",
        ],
        "KICKER_PLAYER_ID",
    )

    name_col = first_existing(
        df,
        [
            "entity_name",
            "player_name",
            "kicker_name",
            "full_name",
        ],
        "KICKER_PLAYER_NAME",
    )

    missing_stats = sorted(
        set(KICKER_STATS)
        - set(df.columns)
    )

    if missing_stats:
        raise RuntimeError(
            "KICKER_STATS_MISSING:"
            + ",".join(missing_stats)
        )

    if df[player_col].isna().any():
        raise RuntimeError(
            "KICKER_PLAYER_ID_NULL"
        )

    if df[
        ["game_id", player_col]
    ].duplicated().any():
        raise RuntimeError(
            "KICKER_IDENTITY_DUPLICATES"
        )

    out = blank_frame(len(df))

    out["entity_type"] = "KICKER"
    out["game_id"] = df["game_id"].to_numpy()
    out["team"] = df["team"].to_numpy()
    out["opponent_team"] = (
        df["opponent_team"].to_numpy()
    )
    out["player_id"] = df[player_col].to_numpy()
    out["entity_name"] = df[name_col].to_numpy()
    out["position"] = "K"
    out["model_group"] = "K"

    for col in KICKER_STATS:
        out[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy()

    return out



# WFS_DST_PA_SCORE_RECONCILIATION_V1
def apply_dst_points_allowed_reconciliation(
    dst: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconcile published D/ST expected_points_allowed to the
    authoritative WFS game-score forecast.

    Authority:
        forecast_publication_selector.select_forecast_path()

    Exact contract:
        away D/ST expected_points_allowed =
            pred_home_points

        home D/ST expected_points_allowed =
            pred_away_points

    Identity:
        exact game_id + team

    Stage23G4 remains untouched. Only expected_points_allowed is
    replaced here in the unified publication layer. Sacks,
    interceptions, fumble recoveries, and defensive touchdowns
    remain exactly as produced by the DST model.
    """
    required_dst = {
        "game_id",
        "team",
        "opponent_team",
        "expected_points_allowed",
    }
    missing_dst = sorted(
        required_dst - set(dst.columns)
    )
    if missing_dst:
        raise RuntimeError(
            "DST_PA_DST_COLUMNS_MISSING:"
            + ",".join(missing_dst)
        )

    forecast_path = select_forecast_path()

    if not forecast_path.is_file():
        raise RuntimeError(
            "DST_PA_FORECAST_SOURCE_MISSING:"
            + str(forecast_path)
        )

    forecast = pd.read_csv(
        forecast_path,
        low_memory=False,
    )

    required_forecast = {
        "game_id",
        "away_team",
        "home_team",
        "pred_away_points",
        "pred_home_points",
        "forecast_status",
    }
    missing_forecast = sorted(
        required_forecast - set(forecast.columns)
    )
    if missing_forecast:
        raise RuntimeError(
            "DST_PA_FORECAST_COLUMNS_MISSING:"
            + ",".join(missing_forecast)
        )

    work = dst.copy()

    work["game_id"] = (
        work["game_id"]
        .astype(str)
        .str.strip()
    )
    work["team"] = (
        work["team"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    work["opponent_team"] = (
        work["opponent_team"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    forecast = forecast.copy()

    forecast["game_id"] = (
        forecast["game_id"]
        .astype(str)
        .str.strip()
    )
    forecast["away_team"] = (
        forecast["away_team"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    forecast["home_team"] = (
        forecast["home_team"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    forecast["forecast_status"] = (
        forecast["forecast_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Only the same ready states exposed by the Forecast Center
    # are authoritative.
    ready_statuses = {
        "READY_CORE_ONLY",
        "READY_INJURY_ADJUSTED",
    }

    forecast = forecast[
        forecast["forecast_status"].isin(
            ready_statuses
        )
    ].copy()

    # Limit authority to the exact games represented by the
    # current DST component. This prevents historical/future rows
    # in the publication artifact from entering the merge.
    current_games = set(
        work["game_id"].dropna().astype(str)
    )

    forecast = forecast[
        forecast["game_id"].isin(current_games)
    ].copy()

    if forecast.empty:
        raise RuntimeError(
            "DST_PA_ZERO_READY_FORECAST_GAMES"
        )

    if forecast["game_id"].duplicated().any():
        dupes = sorted(
            forecast.loc[
                forecast["game_id"].duplicated(
                    keep=False
                ),
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "DST_PA_FORECAST_GAME_DUPLICATES:"
            + ",".join(dupes)
        )

    for col in [
        "pred_away_points",
        "pred_home_points",
    ]:
        forecast[col] = pd.to_numeric(
            forecast[col],
            errors="coerce",
        )

        values = forecast[col]

        if values.isna().any():
            raise RuntimeError(
                "DST_PA_FORECAST_NULL:"
                + col
            )

        arr = values.to_numpy(dtype=float)

        if not np.isfinite(arr).all():
            raise RuntimeError(
                "DST_PA_FORECAST_NONFINITE:"
                + col
            )

        if (arr < 0).any():
            raise RuntimeError(
                "DST_PA_FORECAST_NEGATIVE:"
                + col
            )

    # Convert each game forecast into two exact team authority
    # rows. The opponent's projected points are the team's PA.
    away_authority = pd.DataFrame({
        "game_id":
            forecast["game_id"].to_numpy(),
        "team":
            forecast["away_team"].to_numpy(),
        "forecast_opponent":
            forecast["home_team"].to_numpy(),
        "wfs_expected_points_allowed":
            forecast["pred_home_points"].to_numpy(),
    })

    home_authority = pd.DataFrame({
        "game_id":
            forecast["game_id"].to_numpy(),
        "team":
            forecast["home_team"].to_numpy(),
        "forecast_opponent":
            forecast["away_team"].to_numpy(),
        "wfs_expected_points_allowed":
            forecast["pred_away_points"].to_numpy(),
    })

    authority = pd.concat(
        [
            away_authority,
            home_authority,
        ],
        ignore_index=True,
    )

    if authority[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "DST_PA_AUTHORITY_IDENTITY_DUPLICATES"
        )

    merged = work.merge(
        authority,
        how="left",
        on=["game_id", "team"],
        validate="one_to_one",
    )

    matched = (
        merged["wfs_expected_points_allowed"]
        .notna()
    )

    matched_rows = int(matched.sum())

    # DST has exactly two rows per game. Every current DST row
    # must bind to a ready authoritative score forecast.
    if matched_rows != len(work):
        missing_rows = merged.loc[
            ~matched,
            [
                "game_id",
                "team",
                "opponent_team",
            ],
        ]

        raise RuntimeError(
            "DST_PA_INCOMPLETE_AUTHORITY:"
            + missing_rows.to_dict(
                orient="records"
            ).__repr__()
        )

    opponent_mismatch = (
        merged["opponent_team"]
        != merged["forecast_opponent"]
    )

    if opponent_mismatch.any():
        bad = merged.loc[
            opponent_mismatch,
            [
                "game_id",
                "team",
                "opponent_team",
                "forecast_opponent",
            ],
        ]

        raise RuntimeError(
            "DST_PA_OPPONENT_MISMATCH:"
            + bad.to_dict(
                orient="records"
            ).__repr__()
        )

    # Preserve the original G4 value in memory for verification.
    original_pa = pd.to_numeric(
        merged["expected_points_allowed"],
        errors="coerce",
    )

    if original_pa.isna().any():
        raise RuntimeError(
            "DST_PA_ORIGINAL_G4_NULL"
        )

    merged["expected_points_allowed"] = (
        pd.to_numeric(
            merged[
                "wfs_expected_points_allowed"
            ],
            errors="coerce",
        )
    )

    final_pa = pd.to_numeric(
        merged["expected_points_allowed"],
        errors="coerce",
    ).to_numpy(dtype=float)

    authority_pa = pd.to_numeric(
        merged[
            "wfs_expected_points_allowed"
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    gap = np.abs(
        final_pa - authority_pa
    )

    max_gap = (
        float(np.max(gap))
        if len(gap)
        else 0.0
    )

    if max_gap > 1e-9:
        raise RuntimeError(
            "DST_PA_RECONCILIATION_MISMATCH:"
            f"{max_gap:.12f}"
        )

    changed = int(
        (
            np.abs(
                original_pa.to_numpy(dtype=float)
                - final_pa
            )
            > 1e-9
        ).sum()
    )

    merged = merged.drop(
        columns=[
            "forecast_opponent",
            "wfs_expected_points_allowed",
        ]
    )

    if len(merged) != len(work):
        raise RuntimeError(
            "DST_PA_ROW_COUNT_CHANGED"
        )

    print(
        "DST PA score reconciliation source:",
        forecast_path,
    )
    print(
        "DST PA score reconciliation games:",
        forecast["game_id"].nunique(),
    )
    print(
        "DST PA score reconciliation rows:",
        matched_rows,
    )
    print(
        "DST PA score reconciliation changed rows:",
        changed,
    )
    print(
        "DST PA score reconciliation max gap:",
        f"{max_gap:.12f}",
    )

    return merged

def build_dst(
    df: pd.DataFrame,
) -> pd.DataFrame:
    missing_stats = sorted(
        set(DST_STATS)
        - set(df.columns)
    )

    if missing_stats:
        raise RuntimeError(
            "DST_STATS_MISSING:"
            + ",".join(missing_stats)
        )

    if df[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "DST_IDENTITY_DUPLICATES"
        )

    out = blank_frame(len(df))

    out["entity_type"] = "DST"
    out["game_id"] = df["game_id"].to_numpy()
    out["team"] = df["team"].to_numpy()
    out["opponent_team"] = (
        df["opponent_team"].to_numpy()
    )
    out["player_id"] = np.nan
    out["entity_name"] = (
        df["team"].astype(str) + " DST"
    ).to_numpy()
    out["position"] = "DST"
    out["model_group"] = "DST"

    for col in DST_STATS:
        out[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy()

    return out


def validate_entity(
    df: pd.DataFrame,
    entity_type: str,
    applicable: list[str],
) -> None:
    part = df[
        df["entity_type"].eq(entity_type)
    ].copy()

    if part.empty:
        raise RuntimeError(
            f"{entity_type}_ZERO_ROWS"
        )

    # Frozen C3 proves that offensive applicability is
    # model-group specific, not entity-type wide.
    if entity_type == "OFFENSE_PLAYER":
        offense_applicability = {
            "QB": {
                "expected_attempts",
                "expected_carries",
                "expected_completions",
                "expected_interceptions",
                "expected_passing_tds",
                "expected_passing_yards",
                "expected_rushing_tds",
                "expected_rushing_yards",
            },
            "RB_FB": {
                "expected_carries",
                "expected_receiving_tds",
                "expected_receiving_yards",
                "expected_receptions",
                "expected_rushing_tds",
                "expected_rushing_yards",
                "expected_targets",
            },
            "WR": {
                "expected_receiving_tds",
                "expected_receiving_yards",
                "expected_receptions",
                "expected_targets",
            },
            "TE": {
                "expected_receiving_tds",
                "expected_receiving_yards",
                "expected_receptions",
                "expected_targets",
            },
        }

        observed_groups = set(
            part["model_group"]
            .dropna()
            .astype(str)
            .unique()
        )

        expected_groups = set(
            offense_applicability
        )

        unknown_groups = sorted(
            observed_groups - expected_groups
        )

        if unknown_groups:
            raise RuntimeError(
                "OFFENSE_UNKNOWN_MODEL_GROUP:"
                + ",".join(unknown_groups)
            )

        if part["model_group"].isna().any():
            raise RuntimeError(
                "OFFENSE_MODEL_GROUP_NULL"
            )

        for group, required_stats in (
            offense_applicability.items()
        ):
            group_part = part[
                part["model_group"]
                .astype(str)
                .eq(group)
            ]

            if group_part.empty:
                continue

            for col in REQUIRED_STATS:
                values = pd.to_numeric(
                    group_part[col],
                    errors="coerce",
                )

                if col in required_stats:
                    if values.isna().any():
                        raise RuntimeError(
                            f"OFFENSE_{group}_"
                            f"REQUIRED_NULL:{col}"
                        )

                    arr = values.to_numpy(
                        dtype=float
                    )

                    if not np.isfinite(arr).all():
                        raise RuntimeError(
                            f"OFFENSE_{group}_"
                            f"REQUIRED_NONFINITE:{col}"
                        )

                    if (arr < 0).any():
                        raise RuntimeError(
                            f"OFFENSE_{group}_"
                            f"REQUIRED_NEGATIVE:{col}"
                        )

                else:
                    if group_part[col].notna().any():
                        raise RuntimeError(
                            f"OFFENSE_{group}_"
                            f"NON_APPLICABLE_NOT_NULL:{col}"
                        )

        return

    # Kicker and DST applicability is entity-wide and was
    # proven by the frozen C3 contract.
    applicable_set = set(applicable)

    for col in REQUIRED_STATS:
        values = pd.to_numeric(
            part[col],
            errors="coerce",
        )

        if col in applicable_set:
            if values.isna().any():
                raise RuntimeError(
                    f"{entity_type}_REQUIRED_NULL:{col}"
                )

            arr = values.to_numpy(
                dtype=float
            )

            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{entity_type}_REQUIRED_NONFINITE:{col}"
                )

            if (arr < 0).any():
                raise RuntimeError(
                    f"{entity_type}_REQUIRED_NEGATIVE:{col}"
                )

        else:
            if part[col].notna().any():
                raise RuntimeError(
                    f"{entity_type}_"
                    f"NON_APPLICABLE_NOT_NULL:{col}"
                )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-validation", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    validated_v3 = binding = verified_offense = None
    if args.v3_validation:
        from scripts.preflight_offensive_reconciliation_v3 import load_validated_candidate
        validated_v3, _, binding = load_validated_candidate(args.v3_validation)
        offense_bytes = (args.v3_validation.parent / "offense.parquet").read_bytes()
        if hashlib.sha256(offense_bytes).hexdigest() != binding["inputs"]["offense.parquet"]:
            raise RuntimeError("Validated offensive input hash mismatch")
        verified_offense = pd.read_parquet(io.BytesIO(offense_bytes))
        if (args.output.resolve().parent != args.v3_validation.resolve().parent
                or args.output.name != "unified.parquet" or args.output.exists()
                or args.output.with_suffix(".binding.json").exists()):
            raise RuntimeError("Bound stat assembly requires a new isolated output path")
    section(
        "CURRENT UNIFIED STAT FORECAST C3"
    )

    week, schedule = active_schedule()
    if binding and (binding["season"] != SEASON or binding["week"] != week):
        raise RuntimeError("Validated V3/stat schedule mismatch")
    teams = schedule_teams(schedule)

    print("Season:", SEASON)
    print("Active week:", week)
    print(
        "Schedule games:",
        schedule["game_id"].nunique(),
    )
    print(
        "Schedule teams:",
        len(teams),
    )

    offense = load_component(
        OFFENSE_PATH,
        week,
        "OFFENSE",
        verified_frame=verified_offense,
    )

    kicker = load_component(
        KICKER_PATH,
        week,
        "KICKER",
    )

    dst = load_component(
        DST_PATH,
        week,
        "DST",
    )

    offense = attach_schedule(
        offense,
        teams,
        "OFFENSE",
    )

    offense = apply_v3_offensive_reconciliation(
        offense, validated_v3=validated_v3,
    )

    kicker = attach_schedule(
        kicker,
        teams,
        "KICKER",
    )

    dst = attach_schedule(
        dst,
        teams,
        "DST",
    )
    dst = apply_dst_points_allowed_reconciliation(
        dst
    )

    section("SOURCE INVENTORY")

    print(
        "Offense:",
        len(offense),
        "rows /",
        offense["game_id"].nunique(),
        "games",
    )

    print(
        "Kicker:",
        len(kicker),
        "rows /",
        kicker["game_id"].nunique(),
        "games",
    )

    print(
        "DST:",
        len(dst),
        "rows /",
        dst["game_id"].nunique(),
        "games",
    )

    # Kicker and DST require exact two-team game coverage.
    for label, part in [
        ("KICKER", kicker),
        ("DST", dst),
    ]:
        counts = (
            part.groupby("game_id")["team"]
            .nunique()
        )

        if not counts.eq(2).all():
            bad = counts[
                counts.ne(2)
            ].to_dict()

            raise RuntimeError(
                f"{label}_GAME_TEAM_COVERAGE:{bad}"
            )

    unified = pd.concat(
        [
            build_offense(offense),
            build_kicker(kicker),
            build_dst(dst),
        ],
        ignore_index=True,
    )

    unified = unified[COLUMNS]

    validate_entity(
        unified,
        "OFFENSE_PLAYER",
        OFFENSE_STATS,
    )

    validate_entity(
        unified,
        "KICKER",
        KICKER_STATS,
    )

    validate_entity(
        unified,
        "DST",
        DST_STATS,
    )

    # Identity rules mirror frozen C3.
    offense_part = unified[
        unified["entity_type"].eq(
            "OFFENSE_PLAYER"
        )
    ]

    kicker_part = unified[
        unified["entity_type"].eq(
            "KICKER"
        )
    ]

    dst_part = unified[
        unified["entity_type"].eq("DST")
    ]

    if offense_part[
        ["game_id", "player_id"]
    ].duplicated().any():
        raise RuntimeError(
            "UNIFIED_OFFENSE_IDENTITY_DUPLICATES"
        )

    if kicker_part[
        ["game_id", "player_id"]
    ].duplicated().any():
        raise RuntimeError(
            "UNIFIED_KICKER_IDENTITY_DUPLICATES"
        )

    if dst_part[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "UNIFIED_DST_IDENTITY_DUPLICATES"
        )

    expected_rows = (
        len(offense)
        + len(kicker)
        + len(dst)
    )

    if len(unified) != expected_rows:
        raise RuntimeError(
            "UNIFIED_ROW_COUNT_FAILURE"
        )

    if (
        unified["game_id"].nunique()
        != schedule["game_id"].nunique()
    ):
        raise RuntimeError(
            "UNIFIED_GAME_COVERAGE_FAILURE"
        )

    entity_counts = (
        unified["entity_type"]
        .value_counts()
        .to_dict()
    )

    unified = (
        unified
        .sort_values(
            [
                "game_id",
                "team",
                "entity_type",
                "player_id",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    atomic_write_parquet(
        unified,
        args.output,
    )
    if binding:
        provenance = {"contract": "WFS_V3_BOUND_UNIFIED_V1", "candidate_id": binding["candidate_id"],
                      "v3_sha256": binding["candidate_sha256"], "validation_path": str(args.v3_validation.resolve()),
                      "season": binding["season"], "week": binding["week"], "rows": len(unified),
                      "source_path": str(args.output.resolve()), "source_sha256": sha256_file(args.output)}
        with args.output.with_suffix(".binding.json").open("x") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
        args.output.chmod(0o444)
        args.output.with_suffix(".binding.json").chmod(0o444)

    section("CURRENT C3 OUTPUT")

    print("Rows:", len(unified))
    print(
        "Games:",
        unified["game_id"].nunique(),
    )
    print(
        "Teams:",
        unified["team"].nunique(),
    )
    print(
        "Columns:",
        len(unified.columns),
    )
    print(
        "Entity counts:",
        entity_counts,
    )

    print()
    print("Output:", args.output)
    print(
        "SHA256:",
        sha256_file(args.output),
    )

    section(
        "CURRENT UNIFIED STAT FORECAST C3 PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 92)
        print(
            "CURRENT UNIFIED STAT FORECAST C3 FAILED"
        )
        print("=" * 92)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise
