from __future__ import annotations

from pathlib import Path
from itertools import product
import hashlib
import json
import math

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

DETAIL_PATH = (
    ROOT
    / "processed/stage26gr8s_dvp_weight_detail.csv"
)

SUMMARY_PATH = (
    ROOT
    / "processed/stage26gr8s_dvp_weight_summary.csv"
)

CANONICAL_PATH = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

STATS_PATH = (
    ROOT
    / "data/parquet/nfl_player_game_stats.parquet"
)

ROSTER_PATH = (
    ROOT
    / "data/parquet/nfl_weekly_rosters.parquet"
)

SCHEDULE_PATH = (
    ROOT
    / "data/parquet/nfl_schedule.parquet"
)

TEAM_PATH = (
    ROOT
    / "data/parquet/nfl_team_game_stats.parquet"
)

ENV_PATH = (
    ROOT
    / "data/parquet/nfl_team_pregame_environment.parquet"
)

DVP_PATH = (
    ROOT
    / "data/parquet/nfl_team_position_dvp.parquet"
)

DB_PATH = (
    ROOT
    / "data/nfl.db"
)

OUT_GRID = (
    ROOT
    / "processed/stage26gr8t_rb_joint_grid.csv"
)

OUT_TOP = (
    ROOT
    / "processed/stage26gr8t_rb_joint_top50.csv"
)

OUT_AUDIT = (
    ROOT
    / "processed/stage26gr8t_rb_joint_grid_audit.json"
)

WEIGHTS = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

COMPONENT_NAMES = [
    "catch_rate",
    "receiving_ypr",
    "receiving_td_rate",
    "rushing_ypc",
    "rushing_td_rate",
]


# ============================================================
# HELPERS
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def num(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default

    if not np.isfinite(value):
        return default

    return value


def clamp(value, low, high):
    if pd.isna(value):
        return value

    return max(
        low,
        min(high, value),
    )


def blend(
    prior_value,
    dvp_value,
    dvp_weight,
):
    prior_value = num(
        prior_value
    )

    dvp_value = num(
        dvp_value
    )

    if pd.isna(prior_value):
        return np.nan

    if pd.isna(dvp_value):
        return prior_value

    return (
        (1.0 - dvp_weight)
        * prior_value
        +
        dvp_weight
        * dvp_value
    )


def fd_points(values):
    return (
        0.10
        * num(
            values.get(
                "rushing_yards"
            ),
            0.0,
        )
        +
        6.0
        * num(
            values.get(
                "rushing_tds"
            ),
            0.0,
        )
        +
        0.50
        * num(
            values.get(
                "receptions"
            ),
            0.0,
        )
        +
        0.10
        * num(
            values.get(
                "receiving_yards"
            ),
            0.0,
        )
        +
        6.0
        * num(
            values.get(
                "receiving_tds"
            ),
            0.0,
        )
    )


def metrics(
    actual,
    predicted,
):
    actual = pd.to_numeric(
        actual,
        errors="coerce",
    )

    predicted = pd.to_numeric(
        predicted,
        errors="coerce",
    )

    mask = (
        actual.notna()
        &
        predicted.notna()
    )

    if not mask.any():
        return None

    err = (
        predicted[mask]
        -
        actual[mask]
    )

    return {
        "n":
            int(mask.sum()),

        "mae":
            float(
                err.abs().mean()
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),

        "bias":
            float(
                err.mean()
            ),
    }


# ============================================================
# CANONICAL SAFETY HASH
# ============================================================

canonical_sha_before = (
    sha256(
        CANONICAL_PATH
    )
)


# ============================================================
# REBUILD EXACT R8S RB EVENT SET
#
# We intentionally reconstruct the date-blocked RB events
# rather than trying to infer them from aggregate R8S CSVs.
# ============================================================

from zoneinfo import ZoneInfo
import sqlite3

NY = ZoneInfo(
    "America/New_York"
)

UTC = ZoneInfo(
    "UTC"
)


def parse_kickoff(
    game_date,
    gametime,
):
    if (
        pd.isna(game_date)
        or pd.isna(gametime)
    ):
        return pd.NaT

    d = pd.to_datetime(
        game_date,
        errors="coerce",
    )

    if pd.isna(d):
        return pd.NaT

    text = str(
        gametime
    ).strip()

    if not text:
        return pd.NaT

    dt = pd.to_datetime(
        f"{d.date()} {text}",
        errors="coerce",
    )

    if pd.isna(dt):
        return pd.NaT

    if dt.tzinfo is None:
        dt = dt.tz_localize(
            NY
        )
    else:
        dt = dt.tz_convert(
            NY
        )

    return dt.tz_convert(
        UTC
    )


stats = pd.read_parquet(
    STATS_PATH
)

roster = pd.read_parquet(
    ROSTER_PATH
)

schedule = pd.read_parquet(
    SCHEDULE_PATH
)

team = pd.read_parquet(
    TEAM_PATH
)

env = pd.read_parquet(
    ENV_PATH
)

dvp = pd.read_parquet(
    DVP_PATH
)


with sqlite3.connect(
    f"file:{DB_PATH}?mode=ro",
    uri=True,
) as conn:

    depth = pd.read_sql_query(
        """
        SELECT
            snapshot_dt,
            team,
            player_name,
            gsis_id,
            pos_grp,
            pos_abb,
            pos_slot,
            pos_rank
        FROM depth_charts
        """,
        conn,
    )


for frame in [
    stats,
    roster,
    schedule,
    team,
    env,
    dvp,
]:
    for column in [
        "season",
        "week",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )


for frame in [
    stats,
    roster,
]:
    if "position" in frame.columns:
        frame["position"] = (
            frame["position"]
            .fillna("")
            .astype(str)
            .str.upper()
        )


for frame in [
    stats,
    roster,
    team,
    env,
]:
    if "team" in frame.columns:
        frame["team"] = (
            frame["team"]
            .fillna("")
            .astype(str)
            .str.upper()
        )


stats["opponent_team"] = (
    stats["opponent_team"]
    .fillna("")
    .astype(str)
    .str.upper()
)


dvp["defense_team"] = (
    dvp["defense_team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

dvp["position"] = (
    dvp["position"]
    .fillna("")
    .astype(str)
    .str.upper()
)


depth["snapshot_dt"] = pd.to_datetime(
    depth["snapshot_dt"],
    errors="coerce",
    utc=True,
)

depth["team"] = (
    depth["team"]
    .fillna("")
    .astype(str)
    .str.upper()
)

depth["pos_abb"] = (
    depth["pos_abb"]
    .fillna("")
    .astype(str)
    .str.upper()
)

depth["pos_rank"] = pd.to_numeric(
    depth["pos_rank"],
    errors="coerce",
)


schedule["kickoff_utc"] = [
    parse_kickoff(
        d,
        t,
    )
    for d, t in zip(
        schedule["game_date"],
        schedule["gametime"],
    )
]


kickoff = (
    schedule[
        [
            "game_id",
            "season",
            "week",
            "kickoff_utc",
        ]
    ]
    .drop_duplicates(
        "game_id"
    )
)


for c in [
    "years_exp",
    "rookie_year",
    "entry_year",
]:
    roster[c] = pd.to_numeric(
        roster[c],
        errors="coerce",
    )


rookies = roster[
    roster["gsis_id"].notna()
    &
    roster["position"].eq(
        "RB"
    )
    &
    (
        roster["years_exp"].eq(0)
        |
        roster["rookie_year"].eq(
            roster["season"]
        )
        |
        roster["entry_year"].eq(
            roster["season"]
        )
    )
].copy()


rookie_authority = (
    rookies[
        [
            "gsis_id",
            "season",
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "gsis_id":
                "player_id",
        }
    )
)


stats["player_id"] = (
    stats["player_id"]
    .fillna("")
    .astype(str)
)


games = stats.merge(
    rookie_authority,
    on=[
        "player_id",
        "season",
    ],
    how="inner",
)


games = games.merge(
    kickoff,
    on=[
        "game_id",
        "season",
        "week",
    ],
    how="left",
)


games = games[
    games[
        "kickoff_utc"
    ].notna()
].copy()


resolved = []

for _, row in games.iterrows():

    pid = str(
        row["player_id"]
    )

    team_name = str(
        row["team"]
    )

    ko = row[
        "kickoff_utc"
    ]

    d = depth[
        depth["gsis_id"]
        .fillna("")
        .astype(str)
        .eq(pid)
        &
        depth["team"].eq(
            team_name
        )
        &
        depth["pos_abb"].eq(
            "RB"
        )
        &
        depth["snapshot_dt"].lt(
            ko
        )
    ].copy()

    if d.empty:
        continue

    latest_dt = (
        d[
            "snapshot_dt"
        ].max()
    )

    latest = d[
        d[
            "snapshot_dt"
        ].eq(
            latest_dt
        )
    ].sort_values(
        [
            "pos_rank",
            "pos_slot",
            "gsis_id",
        ]
    )

    best = latest.iloc[0]

    result = row.to_dict()

    result[
        "pregame_depth_rank"
    ] = num(
        best[
            "pos_rank"
        ]
    )

    resolved.append(
        result
    )


resolved = pd.DataFrame(
    resolved
)


starter_games = resolved[
    resolved[
        "pregame_depth_rank"
    ].eq(1)
].copy()


starter_games = starter_games.sort_values(
    [
        "kickoff_utc",
        "player_id",
    ]
)


first_starts = (
    starter_games
    .groupby(
        [
            "player_id",
            "season",
        ],
        as_index=False,
        group_keys=False,
    )
    .head(1)
    .copy()
)


# ============================================================
# TEAM TARGET RATE
# ============================================================

team_hist = team.merge(
    kickoff,
    on=[
        "game_id",
        "season",
        "week",
    ],
    how="left",
)


for c in [
    "attempts",
    "targets",
    "carries",
]:
    team_hist[c] = pd.to_numeric(
        team_hist[c],
        errors="coerce",
    )


team_hist[
    "target_per_attempt"
] = (
    team_hist["targets"]
    /
    team_hist["attempts"]
)


def historical_target_rate(
    event_time,
):
    prior = team_hist[
        team_hist[
            "kickoff_utc"
        ].notna()
        &
        team_hist[
            "kickoff_utc"
        ].lt(
            event_time
        )
        &
        team_hist[
            "attempts"
        ].gt(0)
        &
        team_hist[
            "target_per_attempt"
        ].notna()
    ]

    if prior.empty:
        return None

    return float(
        prior[
            "target_per_attempt"
        ].mean()
    )


# ============================================================
# DATE-BLOCKED RB PRIOR
# ============================================================

RB_STATS = [
    "carries",
    "targets",
    "receptions",
    "rushing_yards",
    "receiving_yards",
    "rushing_tds",
    "receiving_tds",
]


def historical_prior(
    event_time,
    current_player,
):
    first_prior = first_starts[
        first_starts[
            "kickoff_utc"
        ].lt(
            event_time
        )
        &
        ~first_starts[
            "player_id"
        ].eq(
            current_player
        )
    ].copy()

    all_prior = starter_games[
        starter_games[
            "kickoff_utc"
        ].lt(
            event_time
        )
        &
        ~starter_games[
            "player_id"
        ].eq(
            current_player
        )
    ].copy()

    if (
        first_prior.empty
        or all_prior.empty
    ):
        return None

    first_n = len(
        first_prior
    )

    all_n = len(
        all_prior
    )

    w = clamp(
        first_n / all_n,
        0.0,
        1.0,
    )

    result = {}

    for stat in RB_STATS:

        first_mean = (
            pd.to_numeric(
                first_prior[
                    stat
                ],
                errors="coerce",
            )
            .mean()
        )

        all_mean = (
            pd.to_numeric(
                all_prior[
                    stat
                ],
                errors="coerce",
            )
            .mean()
        )

        if (
            pd.isna(first_mean)
            or pd.isna(all_mean)
        ):
            return None

        result[stat] = (
            w
            * float(
                first_mean
            )
            +
            (1.0 - w)
            * float(
                all_mean
            )
        )

    return result


def div(
    a,
    b,
    default=np.nan,
):
    a = num(
        a
    )

    b = num(
        b
    )

    if (
        pd.isna(a)
        or pd.isna(b)
        or abs(b) < 1e-12
    ):
        return default

    return a / b


# ============================================================
# ENVIRONMENT / DVP LOOKUPS
# ============================================================

def environment_for_event(
    game_id,
    team_name,
    season,
):
    exact = env[
        env["game_id"]
        .astype(str)
        .eq(
            str(game_id)
        )
        &
        env["team"].eq(
            team_name
        )
    ].copy()

    if not exact.empty:
        row = exact.iloc[0]

        p5 = num(
            row.get(
                "pass_attempts_avg_5"
            )
        )

        r5 = num(
            row.get(
                "rush_attempts_avg_5"
            )
        )

        if (
            pd.notna(p5)
            and pd.notna(r5)
        ):
            return row

    fallback = env[
        env["team"].eq(
            team_name
        )
        &
        env[
            "season"
        ].lt(
            season
        )
    ].sort_values(
        [
            "season",
            "week",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if fallback.empty:
        return None

    return fallback.iloc[0]


def dvp_for_event(
    game_id,
    opponent,
    season,
):
    exact = dvp[
        dvp["game_id"]
        .astype(str)
        .eq(
            str(game_id)
        )
        &
        dvp[
            "defense_team"
        ].eq(
            opponent
        )
        &
        dvp[
            "position"
        ].eq(
            "RB"
        )
    ].copy()

    if not exact.empty:
        return exact.iloc[0]

    fallback = dvp[
        dvp[
            "defense_team"
        ].eq(
            opponent
        )
        &
        dvp[
            "position"
        ].eq(
            "RB"
        )
        &
        dvp[
            "season"
        ].lt(
            season
        )
    ].sort_values(
        [
            "season",
            "week",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if fallback.empty:
        return None

    return fallback.iloc[0]


# ============================================================
# BUILD HISTORICAL RB EVENTS
# ============================================================

events = []

for _, event in first_starts.sort_values(
    "kickoff_utc"
).iterrows():

    ko = event[
        "kickoff_utc"
    ]

    pid = str(
        event[
            "player_id"
        ]
    )

    prior = historical_prior(
        ko,
        pid,
    )

    if prior is None:
        continue

    target_rate = (
        historical_target_rate(
            ko
        )
    )

    if target_rate is None:
        continue

    env_row = (
        environment_for_event(
            event[
                "game_id"
            ],
            event[
                "team"
            ],
            int(
                event[
                    "season"
                ]
            ),
        )
    )

    if env_row is None:
        continue

    dvp_row = (
        dvp_for_event(
            event[
                "game_id"
            ],
            event[
                "opponent_team"
            ],
            int(
                event[
                    "season"
                ]
            ),
        )
    )

    if dvp_row is None:
        continue

    pass_pool = num(
        env_row.get(
            "pass_attempts_avg_5"
        )
    )

    rush_pool = num(
        env_row.get(
            "rush_attempts_avg_5"
        )
    )

    if (
        pd.isna(pass_pool)
        or pd.isna(rush_pool)
    ):
        continue

    target_pool = (
        pass_pool
        * target_rate
    )

    pred_targets = min(
        max(
            0.0,
            num(
                prior[
                    "targets"
                ],
                0.0,
            ),
        ),
        target_pool,
    )

    pred_carries = min(
        max(
            0.0,
            num(
                prior[
                    "carries"
                ],
                0.0,
            ),
        ),
        rush_pool,
    )

    prior_catch_rate = div(
        prior[
            "receptions"
        ],
        prior[
            "targets"
        ],
    )

    prior_rec_ypr = div(
        prior[
            "receiving_yards"
        ],
        prior[
            "receptions"
        ],
    )

    prior_rec_td_rate = div(
        prior[
            "receiving_tds"
        ],
        prior[
            "targets"
        ],
    )

    prior_rush_ypc = div(
        prior[
            "rushing_yards"
        ],
        prior[
            "carries"
        ],
    )

    prior_rush_td_rate = div(
        prior[
            "rushing_tds"
        ],
        prior[
            "carries"
        ],
    )

    dvp_targets = num(
        dvp_row.get(
            "targets_allowed_avg_3"
        )
    )

    dvp_receptions = num(
        dvp_row.get(
            "receptions_allowed_avg_3"
        )
    )

    dvp_rec_yards = num(
        dvp_row.get(
            "receiving_yards_allowed_avg_3"
        )
    )

    dvp_rec_tds = num(
        dvp_row.get(
            "receiving_tds_allowed_avg_3"
        )
    )

    dvp_carries = num(
        dvp_row.get(
            "carries_allowed_avg_3"
        )
    )

    dvp_rush_yards = num(
        dvp_row.get(
            "rushing_yards_allowed_avg_3"
        )
    )

    dvp_rush_tds = num(
        dvp_row.get(
            "rushing_tds_allowed_avg_3"
        )
    )

    dvp_catch_rate = div(
        dvp_receptions,
        dvp_targets,
    )

    dvp_rec_ypr = div(
        dvp_rec_yards,
        dvp_receptions,
    )

    dvp_rec_td_rate = div(
        dvp_rec_tds,
        dvp_targets,
    )

    dvp_rush_ypc = div(
        dvp_rush_yards,
        dvp_carries,
    )

    dvp_rush_td_rate = div(
        dvp_rush_tds,
        dvp_carries,
    )

    actual = {
        "receptions":
            num(
                event[
                    "receptions"
                ],
                0.0,
            ),

        "receiving_yards":
            num(
                event[
                    "receiving_yards"
                ],
                0.0,
            ),

        "receiving_tds":
            num(
                event[
                    "receiving_tds"
                ],
                0.0,
            ),

        "rushing_yards":
            num(
                event[
                    "rushing_yards"
                ],
                0.0,
            ),

        "rushing_tds":
            num(
                event[
                    "rushing_tds"
                ],
                0.0,
            ),
    }

    events.append(
        {
            "game_id":
                event[
                    "game_id"
                ],

            "player_id":
                pid,

            "player_name":
                event.get(
                    "player_display_name",
                    pid,
                ),

            "pred_targets":
                pred_targets,

            "pred_carries":
                pred_carries,

            "prior_catch_rate":
                prior_catch_rate,

            "prior_rec_ypr":
                prior_rec_ypr,

            "prior_rec_td_rate":
                prior_rec_td_rate,

            "prior_rush_ypc":
                prior_rush_ypc,

            "prior_rush_td_rate":
                prior_rush_td_rate,

            "dvp_catch_rate":
                dvp_catch_rate,

            "dvp_rec_ypr":
                dvp_rec_ypr,

            "dvp_rec_td_rate":
                dvp_rec_td_rate,

            "dvp_rush_ypc":
                dvp_rush_ypc,

            "dvp_rush_td_rate":
                dvp_rush_td_rate,

            "actual_fd_points":
                fd_points(
                    actual
                ),
        }
    )


events = pd.DataFrame(
    events
)

if len(events) != 10:
    raise RuntimeError(
        "FAIL | expected exact R8S RB sample n=10, "
        f"got n={len(events)}"
    )


print()
print("===== RB EVENT COVERAGE =====")
print(f"RB_EVENTS={len(events)}")


# ============================================================
# GRID SEARCH
# ============================================================

grid_rows = []

for (
    catch_w,
    rec_ypr_w,
    rec_td_w,
    rush_ypc_w,
    rush_td_w,
) in product(
    WEIGHTS,
    WEIGHTS,
    WEIGHTS,
    WEIGHTS,
    WEIGHTS,
):

    predicted = []

    for _, row in events.iterrows():

        catch_rate = clamp(
            blend(
                row[
                    "prior_catch_rate"
                ],
                row[
                    "dvp_catch_rate"
                ],
                catch_w,
            ),
            0.0,
            1.0,
        )

        rec_ypr = max(
            0.0,
            blend(
                row[
                    "prior_rec_ypr"
                ],
                row[
                    "dvp_rec_ypr"
                ],
                rec_ypr_w,
            ),
        )

        rec_td_rate = max(
            0.0,
            blend(
                row[
                    "prior_rec_td_rate"
                ],
                row[
                    "dvp_rec_td_rate"
                ],
                rec_td_w,
            ),
        )

        rush_ypc = max(
            0.0,
            blend(
                row[
                    "prior_rush_ypc"
                ],
                row[
                    "dvp_rush_ypc"
                ],
                rush_ypc_w,
            ),
        )

        rush_td_rate = max(
            0.0,
            blend(
                row[
                    "prior_rush_td_rate"
                ],
                row[
                    "dvp_rush_td_rate"
                ],
                rush_td_w,
            ),
        )

        receptions = (
            row[
                "pred_targets"
            ]
            * catch_rate
        )

        receiving_yards = (
            receptions
            * rec_ypr
        )

        receiving_tds = (
            row[
                "pred_targets"
            ]
            * rec_td_rate
        )

        rushing_yards = (
            row[
                "pred_carries"
            ]
            * rush_ypc
        )

        rushing_tds = (
            row[
                "pred_carries"
            ]
            * rush_td_rate
        )

        predicted.append(
            fd_points(
                {
                    "receptions":
                        receptions,

                    "receiving_yards":
                        receiving_yards,

                    "receiving_tds":
                        receiving_tds,

                    "rushing_yards":
                        rushing_yards,

                    "rushing_tds":
                        rushing_tds,
                }
            )
        )

    m = metrics(
        events[
            "actual_fd_points"
        ],
        pd.Series(
            predicted,
            index=events.index,
        ),
    )

    grid_rows.append(
        {
            "catch_rate_weight":
                catch_w,

            "receiving_ypr_weight":
                rec_ypr_w,

            "receiving_td_weight":
                rec_td_w,

            "rushing_ypc_weight":
                rush_ypc_w,

            "rushing_td_weight":
                rush_td_w,

            "total_dvp_weight":
                (
                    catch_w
                    +
                    rec_ypr_w
                    +
                    rec_td_w
                    +
                    rush_ypc_w
                    +
                    rush_td_w
                ),

            "n":
                m[
                    "n"
                ],

            "fd_mae":
                m[
                    "mae"
                ],

            "fd_rmse":
                m[
                    "rmse"
                ],

            "fd_bias":
                m[
                    "bias"
                ],

            "abs_fd_bias":
                abs(
                    m[
                        "bias"
                    ]
                ),
        }
    )


grid = pd.DataFrame(
    grid_rows
)


if len(grid) != 7776:
    raise RuntimeError(
        "FAIL | expected 7776 combinations, "
        f"got {len(grid)}"
    )


# ============================================================
# DETERMINISTIC RANKING
# ============================================================

grid = grid.sort_values(
    [
        "fd_mae",
        "fd_rmse",
        "abs_fd_bias",
        "total_dvp_weight",
        "catch_rate_weight",
        "receiving_ypr_weight",
        "receiving_td_weight",
        "rushing_ypc_weight",
        "rushing_td_weight",
    ],
    ascending=[
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ],
).reset_index(
    drop=True
)


best = grid.iloc[0]


baseline = grid[
    grid[
        [
            "catch_rate_weight",
            "receiving_ypr_weight",
            "receiving_td_weight",
            "rushing_ypc_weight",
            "rushing_td_weight",
        ]
    ]
    .eq(0.0)
    .all(
        axis=1
    )
]


if len(baseline) != 1:
    raise RuntimeError(
        "FAIL | baseline combination not unique"
    )


baseline = baseline.iloc[0]


mae_improvement = (
    float(
        baseline[
            "fd_mae"
        ]
    )
    -
    float(
        best[
            "fd_mae"
        ]
    )
)


mae_improvement_pct = (
    mae_improvement
    /
    float(
        baseline[
            "fd_mae"
        ]
    )
    * 100.0
)


rmse_delta = (
    float(
        best[
            "fd_rmse"
        ]
    )
    -
    float(
        baseline[
            "fd_rmse"
        ]
    )
)


# ============================================================
# PROMOTION DIAGNOSTIC
#
# Research gate only.
#
# Require:
# 1. strictly lower FD MAE
# 2. no material RMSE regression
#
# Material RMSE regression threshold:
# > +2.5% relative to baseline.
# ============================================================

max_allowed_rmse = (
    float(
        baseline[
            "fd_rmse"
        ]
    )
    * 1.025
)


strict_mae_win = (
    float(
        best[
            "fd_mae"
        ]
    )
    <
    float(
        baseline[
            "fd_mae"
        ]
    )
    -
    1e-12
)


rmse_gate = (
    float(
        best[
            "fd_rmse"
        ]
    )
    <=
    max_allowed_rmse
)


if (
    strict_mae_win
    and rmse_gate
):
    diagnostic = (
        "JOINT_MATCHUP_CANDIDATE"
    )
else:
    diagnostic = (
        "REJECT_MATCHUP_KEEP_ZERO_DVP"
    )


# ============================================================
# CONSERVATIVE STABILITY VIEW
#
# Show the best solutions within 1%, 2.5%, and 5% of
# the winning MAE so we can see whether the exact weight
# vector is stable or just one narrow optimum.
# ============================================================

best_mae = float(
    best[
        "fd_mae"
    ]
)


for threshold in [
    0.01,
    0.025,
    0.05,
]:
    column = (
        "within_"
        + str(
            threshold
        )
        .replace(
            ".",
            "_"
        )
    )

    grid[
        column
    ] = (
        grid[
            "fd_mae"
        ]
        <=
        best_mae
        * (
            1.0
            +
            threshold
        )
    )


# ============================================================
# WRITE ARTIFACTS
# ============================================================

OUT_GRID.parent.mkdir(
    parents=True,
    exist_ok=True,
)


grid.to_csv(
    OUT_GRID,
    index=False,
)


grid.head(
    50
).to_csv(
    OUT_TOP,
    index=False,
)


canonical_sha_after = (
    sha256(
        CANONICAL_PATH
    )
)


if (
    canonical_sha_before
    != canonical_sha_after
):
    raise RuntimeError(
        "FAIL | canonical forecast mutated"
    )


audit = {
    "stage":
        "STAGE26G-R8T",

    "purpose":
        "joint RB DvP weight grid calibration using FanDuel MAE",

    "rb_event_count":
        int(
            len(events)
        ),

    "grid_combinations":
        int(
            len(grid)
        ),

    "weights_tested":
        WEIGHTS,

    "primary_objective":
        "minimum FanDuel MAE",

    "tie_breakers": [
        "FanDuel RMSE",
        "absolute FanDuel bias",
        "lowest total DvP weight",
        "lexicographically lowest component weights",
    ],

    "baseline": {
        "catch_rate_weight":
            0.0,

        "receiving_ypr_weight":
            0.0,

        "receiving_td_weight":
            0.0,

        "rushing_ypc_weight":
            0.0,

        "rushing_td_weight":
            0.0,

        "fd_mae":
            float(
                baseline[
                    "fd_mae"
                ]
            ),

        "fd_rmse":
            float(
                baseline[
                    "fd_rmse"
                ]
            ),

        "fd_bias":
            float(
                baseline[
                    "fd_bias"
                ]
            ),
    },

    "best": {
        "catch_rate_weight":
            float(
                best[
                    "catch_rate_weight"
                ]
            ),

        "receiving_ypr_weight":
            float(
                best[
                    "receiving_ypr_weight"
                ]
            ),

        "receiving_td_weight":
            float(
                best[
                    "receiving_td_weight"
                ]
            ),

        "rushing_ypc_weight":
            float(
                best[
                    "rushing_ypc_weight"
                ]
            ),

        "rushing_td_weight":
            float(
                best[
                    "rushing_td_weight"
                ]
            ),

        "fd_mae":
            float(
                best[
                    "fd_mae"
                ]
            ),

        "fd_rmse":
            float(
                best[
                    "fd_rmse"
                ]
            ),

        "fd_bias":
            float(
                best[
                    "fd_bias"
                ]
            ),

        "total_dvp_weight":
            float(
                best[
                    "total_dvp_weight"
                ]
            ),
    },

    "mae_improvement":
        mae_improvement,

    "mae_improvement_pct":
        mae_improvement_pct,

    "rmse_delta":
        rmse_delta,

    "max_allowed_rmse":
        max_allowed_rmse,

    "strict_mae_win":
        bool(
            strict_mae_win
        ),

    "rmse_gate":
        bool(
            rmse_gate
        ),

    "diagnostic":
        diagnostic,

    "production_promotion":
        False,

    "canonical_sha_before":
        canonical_sha_before,

    "canonical_sha_after":
        canonical_sha_after,

    "canonical_unchanged":
        (
            canonical_sha_before
            ==
            canonical_sha_after
        ),

    "publisher_mutation":
        False,

    "analyst_mutation":
        False,

    "app_mutation":
        False,

    "solver_mutation":
        False,

    "team_environment_mutation":
        False,
}


OUT_AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("R8T BASELINE")
print("============================================================")

print(
    baseline[
        [
            "catch_rate_weight",
            "receiving_ypr_weight",
            "receiving_td_weight",
            "rushing_ypc_weight",
            "rushing_td_weight",
            "fd_mae",
            "fd_rmse",
            "fd_bias",
        ]
    ].to_string()
)


print()
print("============================================================")
print("R8T BEST JOINT COMBINATION")
print("============================================================")

print(
    best[
        [
            "catch_rate_weight",
            "receiving_ypr_weight",
            "receiving_td_weight",
            "rushing_ypc_weight",
            "rushing_td_weight",
            "total_dvp_weight",
            "fd_mae",
            "fd_rmse",
            "fd_bias",
        ]
    ].to_string()
)


print()
print("============================================================")
print("R8T IMPROVEMENT")
print("============================================================")

print(
    f"FD_MAE_IMPROVEMENT="
    f"{mae_improvement:.6f}"
)

print(
    f"FD_MAE_IMPROVEMENT_PCT="
    f"{mae_improvement_pct:.6f}%"
)

print(
    f"FD_RMSE_DELTA="
    f"{rmse_delta:.6f}"
)

print(
    f"MAX_ALLOWED_RMSE="
    f"{max_allowed_rmse:.6f}"
)

print(
    f"STRICT_MAE_WIN="
    f"{strict_mae_win}"
)

print(
    f"RMSE_GATE="
    f"{rmse_gate}"
)

print(
    f"DIAGNOSTIC="
    f"{diagnostic}"
)


print()
print("============================================================")
print("R8T TOP 20 COMBINATIONS")
print("============================================================")

print(
    grid.head(
        20
    )[
        [
            "catch_rate_weight",
            "receiving_ypr_weight",
            "receiving_td_weight",
            "rushing_ypc_weight",
            "rushing_td_weight",
            "total_dvp_weight",
            "fd_mae",
            "fd_rmse",
            "fd_bias",
        ]
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8T STABILITY")
print("============================================================")

for threshold in [
    0.01,
    0.025,
    0.05,
]:
    column = (
        "within_"
        + str(
            threshold
        )
        .replace(
            ".",
            "_"
        )
    )

    subset = grid[
        grid[
            column
        ]
    ]

    print(
        f"WITHIN_{threshold * 100:.1f}%_OF_BEST="
        f"{len(subset)}"
    )

    if not subset.empty:
        print(
            "  MEAN_WEIGHTS="
            f"catch={subset['catch_rate_weight'].mean():.3f}, "
            f"rec_ypr={subset['receiving_ypr_weight'].mean():.3f}, "
            f"rec_td={subset['receiving_td_weight'].mean():.3f}, "
            f"rush_ypc={subset['rushing_ypc_weight'].mean():.3f}, "
            f"rush_td={subset['rushing_td_weight'].mean():.3f}"
        )


print()
print("============================================================")
print("SAFETY / MUTATION AUDIT")
print("============================================================")

print(
    f"CANONICAL_SHA_BEFORE="
    f"{canonical_sha_before}"
)

print(
    f"CANONICAL_SHA_AFTER="
    f"{canonical_sha_after}"
)

print(
    "CANONICAL_UNCHANGED="
    f"{canonical_sha_before == canonical_sha_after}"
)

print(
    "PRODUCTION_PROMOTION=NONE"
)

print(
    "PUBLISHER_MUTATION=NONE"
)

print(
    "ANALYST_MUTATION=NONE"
)

print(
    "APP_MUTATION=NONE"
)

print(
    "SOLVER_MUTATION=NONE"
)

print(
    "TEAM_ENVIRONMENT_MUTATION=NONE"
)


print()
print("============================================================")
print("ARTIFACTS")
print("============================================================")

print(
    f"GRID={OUT_GRID}"
)

print(
    f"TOP50={OUT_TOP}"
)

print(
    f"AUDIT={OUT_AUDIT}"
)


print()
print("============================================================")
print("STAGE26G_R8T_RB_JOINT_GRID=PASS")
print("============================================================")
