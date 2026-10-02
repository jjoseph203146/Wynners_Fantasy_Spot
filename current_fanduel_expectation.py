#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

CURRENT_STAT = (
    ROOT
    / "data/parquet"
    / "nfl_current_unified_stat_forecasts.parquet"
)

CURRENT_OUTPUT = (
    ROOT
    / "data/parquet"
    / "nfl_current_fanduel_expectation.parquet"
)

FROZEN_STAT = (
    ROOT
    / "data/model_candidates/stat_forecast"
    / "stage24j_c_r7_r14g_c3_20260912T194646Z"
    / "current_unified_stat_forecasts_535.parquet"
)

FROZEN_FD = (
    ROOT
    / "data/model_candidates/fanduel_expectation"
    / "stage24j_c_r7_r14h_c_20260912T195227Z"
    / "current_unified_fanduel_expectation_535.parquet"
)

OUTPUT_COLUMNS = [
    "game_id",
    "team",
    "position",
    "entity_id",
    "entity_name",
    "source_component",
    "source_contract",
    "expected_fanduel_points",
    "opponent_team",
    "season",
    "week",
    "fanduel_expectation_contract",
    "candidate_only",
    "solver_consumable",
]

FD_CONTRACT = "WFS_FANDUEL_EXPECTATION_V1"

OFFENSE_CONTRACT = (
    "WFS_FANDUEL_OFFENSE_EXPECTATION_V1"
)
KICKER_CONTRACT = (
    "WFS_FANDUEL_KICKER_EXPECTATION_V1"
)
DST_CONTRACT = (
    "WFS_FANDUEL_DST_EXPECTATION_V1"
)

# Frozen Stage24E contract.
P_FG_50_PLUS = 0.19208262
P_FG_UNDER_50 = 0.80791738

# Frozen Stage24G-R4 league prior.
DST_OMITTED_EVENT_PRIOR = (
    0.42923976608187137
)


def pa_tier_points(
    points_allowed: float,
) -> float:
    """
    Frozen Stage24F SCOREBOARD_PLUGIN / PLUGIN_MEAN_BUCKET.

    Bucket boundaries:
      < 0.5   -> 10
      < 6.5   -> 7
      < 13.5  -> 4
      < 20.5  -> 1
      < 27.5  -> 0
      < 34.5  -> -1
      >=34.5  -> -4
    """
    x = float(points_allowed)

    if not math.isfinite(x):
        raise RuntimeError(
            "DST_POINTS_ALLOWED_NONFINITE"
        )

    if x < 0.5:
        return 10.0
    if x < 6.5:
        return 7.0
    if x < 13.5:
        return 4.0
    if x < 20.5:
        return 1.0
    if x < 27.5:
        return 0.0
    if x < 34.5:
        return -1.0
    return -4.0


def numeric(
    row: pd.Series,
    col: str,
    default: float = 0.0,
) -> float:
    value = row.get(col)

    if pd.isna(value):
        return float(default)

    value = float(value)

    if not math.isfinite(value):
        raise RuntimeError(
            f"NONFINITE:{col}"
        )

    return value


def offense_points(
    row: pd.Series,
) -> float:
    return (
        numeric(
            row,
            "expected_passing_yards",
        ) * 0.04
        + numeric(
            row,
            "expected_passing_tds",
        ) * 4.0
        - numeric(
            row,
            "expected_interceptions",
        ) * 1.0
        + numeric(
            row,
            "expected_rushing_yards",
        ) * 0.1
        + numeric(
            row,
            "expected_rushing_tds",
        ) * 6.0
        + numeric(
            row,
            "expected_receptions",
        ) * 0.5
        + numeric(
            row,
            "expected_receiving_yards",
        ) * 0.1
        + numeric(
            row,
            "expected_receiving_tds",
        ) * 6.0
    )


def kicker_points(
    row: pd.Series,
) -> float:
    fgm = numeric(
        row,
        "expected_fgm",
    )

    xpm = numeric(
        row,
        "expected_xpm",
    )

    expected_under_50 = (
        fgm * P_FG_UNDER_50
    )

    expected_50_plus = (
        fgm * P_FG_50_PLUS
    )

    return (
        expected_under_50 * 3.0
        + expected_50_plus * 5.0
        + xpm
    )


def dst_points(
    row: pd.Series,
) -> float:
    supported = (
        numeric(
            row,
            "expected_sacks",
        ) * 1.0
        + numeric(
            row,
            "expected_interceptions",
        ) * 2.0
        + numeric(
            row,
            "expected_fumble_recoveries",
        ) * 2.0
        + numeric(
            row,
            "expected_defensive_tds",
        ) * 6.0
        + pa_tier_points(
            numeric(
                row,
                "expected_points_allowed",
            )
        )
    )

    return (
        supported
        + DST_OMITTED_EVENT_PRIOR
    )


def build(
    stat: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in stat.iterrows():
        entity_type = str(
            row["entity_type"]
        )

        if entity_type == "OFFENSE_PLAYER":
            points = offense_points(row)

            position = str(
                row["position"]
            )

            entity_id = str(
                row["player_id"]
            )

            source_component = "OFFENSE"
            source_contract = OFFENSE_CONTRACT

        elif entity_type == "KICKER":
            points = kicker_points(row)

            position = "K"

            entity_id = str(
                row["player_id"]
            )

            source_component = "KICKER"
            source_contract = KICKER_CONTRACT

        elif entity_type == "DST":
            points = dst_points(row)

            position = "DST"

            entity_id = (
                "DST:"
                + str(row["team"])
            )

            source_component = "DST"
            source_contract = DST_CONTRACT

        else:
            raise RuntimeError(
                "UNKNOWN_ENTITY_TYPE:"
                + entity_type
            )

        if (
            not math.isfinite(points)
            or points < 0
        ):
            raise RuntimeError(
                "INVALID_FANDUEL_EXPECTATION:"
                f"{row['game_id']}:"
                f"{row['team']}:"
                f"{entity_id}:"
                f"{points}"
            )

        rows.append(
            {
                "game_id": row["game_id"],
                "team": row["team"],
                "position": position,
                "entity_id": entity_id,
                "entity_name": row[
                    "entity_name"
                ],
                "source_component":
                    source_component,
                "source_contract":
                    source_contract,
                "expected_fanduel_points":
                    float(points),
                "opponent_team":
                    row["opponent_team"],
                "season": np.nan,
                "week": np.nan,
                "fanduel_expectation_contract":
                    FD_CONTRACT,
                "candidate_only": True,
                "solver_consumable": False,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if len(out) != len(stat):
        raise RuntimeError(
            "ROW_COUNT_MISMATCH"
        )

    if out[
        "expected_fanduel_points"
    ].isna().any():
        raise RuntimeError(
            "FD_NULLS"
        )

    values = out[
        "expected_fanduel_points"
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "FD_NONFINITE"
        )

    if (values < 0).any():
        raise RuntimeError(
            "FD_NEGATIVE"
        )

    return out


def regression() -> None:
    stat = pd.read_parquet(
        FROZEN_STAT
    )

    expected = pd.read_parquet(
        FROZEN_FD
    )

    actual = build(stat)

    keys = [
        "game_id",
        "team",
        "position",
        "entity_id",
    ]

    if actual.duplicated(keys).any():
        raise RuntimeError(
            "REGRESSION_ACTUAL_DUPLICATE_IDENTITY"
        )

    if expected.duplicated(keys).any():
        raise RuntimeError(
            "REGRESSION_EXPECTED_DUPLICATE_IDENTITY"
        )

    left = set(
        map(
            tuple,
            actual[keys].astype(str).to_numpy(),
        )
    )

    right = set(
        map(
            tuple,
            expected[keys].astype(str).to_numpy(),
        )
    )

    print(
        "ACTUAL_ROWS =",
        len(actual),
    )
    print(
        "EXPECTED_ROWS =",
        len(expected),
    )
    print(
        "IDENTITY_MISSING =",
        len(right - left),
    )
    print(
        "IDENTITY_EXTRA =",
        len(left - right),
    )

    if left != right:
        raise RuntimeError(
            "REGRESSION_IDENTITY_MISMATCH"
        )

    merged = actual.merge(
        expected[
            keys
            + [
                "expected_fanduel_points"
            ]
        ],
        on=keys,
        how="inner",
        suffixes=(
            "_actual",
            "_expected",
        ),
        validate="one_to_one",
    )

    merged["abs_delta"] = (
        merged[
            "expected_fanduel_points_actual"
        ]
        - merged[
            "expected_fanduel_points_expected"
        ]
    ).abs()

    print()
    print(
        "MAX_ABS_DELTA =",
        merged["abs_delta"].max(),
    )

    print(
        "MISMATCH_GT_1E_10 =",
        int(
            (
                merged["abs_delta"]
                > 1e-10
            ).sum()
        ),
    )

    print()
    print(
        "COMPONENT MAX DELTAS"
    )

    for component in [
        "OFFENSE",
        "KICKER",
        "DST",
    ]:
        ids = set(
            actual.loc[
                actual[
                    "source_component"
                ].eq(component),
                keys,
            ]
            .astype(str)
            .apply(tuple, axis=1)
        )

        mask = (
            merged[keys]
            .astype(str)
            .apply(tuple, axis=1)
            .isin(ids)
        )

        part = merged[mask]

        print(
            component,
            "ROWS=",
            len(part),
            "MAX_DELTA=",
            part["abs_delta"].max(),
            "MISMATCH_GT_1E_10=",
            int(
                (
                    part["abs_delta"]
                    > 1e-10
                ).sum()
            ),
        )

    if (
        merged["abs_delta"].max()
        > 1e-10
    ):
        worst = (
            merged.sort_values(
                "abs_delta",
                ascending=False,
            )
            .head(20)
        )

        print()
        print("WORST DELTAS")
        print(
            worst[
                keys
                + [
                    "expected_fanduel_points_actual",
                    "expected_fanduel_points_expected",
                    "abs_delta",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "STAGE24_REGRESSION_FAILED"
        )

    print()
    print(
        "STAGE24_FROZEN_REGRESSION=PASS"
    )


def current() -> None:
    stat = pd.read_parquet(
        CURRENT_STAT
    )

    out = build(stat)

    # Current-build only: attach authoritative season/week
    # from data/nfl.db using exact game_id identity.
    out = attach_current_schedule_metadata(out)

    print(
        "CURRENT_ROWS =",
        len(out),
    )

    print(
        "CURRENT_GAMES =",
        out["game_id"].nunique(),
    )

    print(
        "CURRENT_TEAMS =",
        out["team"].nunique(),
    )

    print(
        "COMPONENT_COUNTS =",
        out[
            "source_component"
        ].value_counts().to_dict(),
    )

    CURRENT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = CURRENT_OUTPUT.with_suffix(
        ".parquet.tmp"
    )

    out.to_parquet(
        tmp,
        index=False,
    )

    tmp.replace(
        CURRENT_OUTPUT
    )

    print(
        "OUTPUT =",
        CURRENT_OUTPUT,
    )

    print(
        "CURRENT_STAGE24_SCORING=PASS"
    )



def attach_current_schedule_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach authoritative season/week metadata for a current
    FanDuel expectation artifact using exact game_id identity.

    Current-build only. Frozen regression behavior is unchanged.
    Fail closed on missing, duplicate, or ambiguous schedule data.
    """
    db_path = (
        Path(__file__).resolve().parent
        / "data"
        / "nfl.db"
    )

    if not db_path.is_file():
        raise RuntimeError(
            f"SCHEDULE_DB_MISSING:{db_path}"
        )

    game_ids = (
        df["game_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if game_ids.eq("").any():
        raise RuntimeError(
            "CURRENT_STAGE24_BLANK_GAME_ID"
        )

    wanted = sorted(set(game_ids))

    placeholders = ",".join(
        "?" for _ in wanted
    )

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        schedule = pd.read_sql_query(
            f"""
            SELECT
                game_id,
                season,
                week
            FROM games
            WHERE game_id IN ({placeholders})
            """,
            conn,
            params=wanted,
        )

    schedule["game_id"] = (
        schedule["game_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if schedule["game_id"].duplicated().any():
        dupes = sorted(
            schedule.loc[
                schedule["game_id"].duplicated(
                    keep=False
                ),
                "game_id",
            ].unique().tolist()
        )

        raise RuntimeError(
            "CURRENT_STAGE24_DUPLICATE_SCHEDULE_GAME_ID:"
            + ",".join(dupes)
        )

    schedule_ids = set(
        schedule["game_id"]
    )

    missing = sorted(
        set(wanted) - schedule_ids
    )

    if missing:
        raise RuntimeError(
            "CURRENT_STAGE24_SCHEDULE_GAME_MISSING:"
            + ",".join(missing)
        )

    schedule["season"] = pd.to_numeric(
        schedule["season"],
        errors="raise",
    )

    schedule["week"] = pd.to_numeric(
        schedule["week"],
        errors="raise",
    )

    if schedule["season"].isna().any():
        raise RuntimeError(
            "CURRENT_STAGE24_SCHEDULE_SEASON_NULL"
        )

    if schedule["week"].isna().any():
        raise RuntimeError(
            "CURRENT_STAGE24_SCHEDULE_WEEK_NULL"
        )

    season_counts = (
        schedule["season"]
        .astype(int)
        .value_counts()
    )

    week_counts = (
        schedule["week"]
        .astype(int)
        .value_counts()
    )

    if len(season_counts) != 1:
        raise RuntimeError(
            "CURRENT_STAGE24_MULTIPLE_SEASONS:"
            + str(
                sorted(
                    season_counts.index.tolist()
                )
            )
        )

    if len(week_counts) != 1:
        raise RuntimeError(
            "CURRENT_STAGE24_MULTIPLE_WEEKS:"
            + str(
                sorted(
                    week_counts.index.tolist()
                )
            )
        )

    metadata = schedule.rename(
        columns={
            "season":
                "_schedule_season",
            "week":
                "_schedule_week",
        }
    )

    out = df.drop(
        columns=["season", "week"],
        errors="ignore",
    ).merge(
        metadata,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    if out["_schedule_season"].isna().any():
        raise RuntimeError(
            "CURRENT_STAGE24_SEASON_JOIN_FAILED"
        )

    if out["_schedule_week"].isna().any():
        raise RuntimeError(
            "CURRENT_STAGE24_WEEK_JOIN_FAILED"
        )

    out["season"] = (
        out["_schedule_season"]
        .astype(int)
    )

    out["week"] = (
        out["_schedule_week"]
        .astype(int)
    )

    out = out.drop(
        columns=[
            "_schedule_season",
            "_schedule_week",
        ]
    )

    # Restore exact Stage24 output column ordering.
    out = out[OUTPUT_COLUMNS].copy()

    if len(out) != len(df):
        raise RuntimeError(
            "CURRENT_STAGE24_SCHEDULE_JOIN_ROW_DRIFT"
        )

    return out

def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--regression",
        action="store_true",
    )

    group.add_argument(
        "--current",
        action="store_true",
    )

    args = parser.parse_args()

    if args.regression:
        regression()
    else:
        current()


if __name__ == "__main__":
    main()
