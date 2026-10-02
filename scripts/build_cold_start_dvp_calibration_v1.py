from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

STATS_PATH = ROOT / "data/parquet/nfl_player_game_stats.parquet"
ROSTER_PATH = ROOT / "data/parquet/nfl_weekly_rosters.parquet"
SCHEDULE_PATH = ROOT / "data/parquet/nfl_schedule.parquet"
TEAM_PATH = ROOT / "data/parquet/nfl_team_game_stats.parquet"
ENV_PATH = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
DVP_PATH = ROOT / "data/parquet/nfl_team_position_dvp.parquet"
CANONICAL_PATH = ROOT / "data/parquet/current_unified_stat_forecasts.parquet"
DB_PATH = ROOT / "data/nfl.db"

OUT_DETAIL = (
    ROOT
    / "processed/stage26gr8s_dvp_weight_detail.csv"
)

OUT_SUMMARY = (
    ROOT
    / "processed/stage26gr8s_dvp_weight_summary.csv"
)

OUT_COMBINED = (
    ROOT
    / "processed/stage26gr8s_combined_model_summary.csv"
)

OUT_AUDIT = (
    ROOT
    / "processed/stage26gr8s_dvp_calibration_audit.json"
)

SUPPORTED = {"QB", "RB", "WR", "TE"}

WEIGHTS = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ============================================================
# BASIC HELPERS
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


def div(a, b, default=np.nan):
    a = num(a)
    b = num(b)

    if (
        pd.isna(a)
        or pd.isna(b)
        or abs(b) < 1e-12
    ):
        return default

    return a / b


def clamp(value, low, high):
    if pd.isna(value):
        return value

    return max(
        low,
        min(high, value),
    )


def weighted_blend(
    prior_value,
    dvp_value,
    dvp_weight,
):
    prior_value = num(prior_value)
    dvp_value = num(dvp_value)

    if pd.isna(prior_value):
        return np.nan

    # Fail closed to the player/position prior when
    # matchup evidence does not exist.
    if pd.isna(dvp_value):
        return prior_value

    return (
        (1.0 - dvp_weight)
        * prior_value
        +
        dvp_weight
        * dvp_value
    )


def parse_kickoff(
    game_date,
    gametime,
):
    if pd.isna(game_date) or pd.isna(gametime):
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
        dt = dt.tz_localize(NY)
    else:
        dt = dt.tz_convert(NY)

    return dt.tz_convert(UTC)


def metric_values(
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

    a = actual[mask]
    p = predicted[mask]

    err = p - a

    return {
        "n": int(mask.sum()),
        "mae": float(
            err.abs().mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),
        "bias": float(
            err.mean()
        ),
    }


def fd_points(values):
    return (
        0.04
        * num(
            values.get(
                "passing_yards"
            ),
            0.0,
        )
        +
        4.0
        * num(
            values.get(
                "passing_tds"
            ),
            0.0,
        )
        -
        num(
            values.get(
                "interceptions"
            ),
            0.0,
        )
        +
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


# ============================================================
# LOAD
# ============================================================

canonical_sha_before = sha256(
    CANONICAL_PATH
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


# ============================================================
# NORMALIZE
# ============================================================

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


if "opponent_team" in stats.columns:
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


# ============================================================
# KICKOFF AUTHORITY
# ============================================================

required_schedule = {
    "game_id",
    "season",
    "week",
    "game_date",
    "gametime",
}

missing = (
    required_schedule
    - set(schedule.columns)
)

if missing:
    raise RuntimeError(
        "FAIL | missing schedule columns: "
        f"{sorted(missing)}"
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


# ============================================================
# TRUE ROOKIE AUTHORITY
# ============================================================

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
    roster["position"].isin(
        SUPPORTED
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
            "position",
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "gsis_id":
                "player_id",

            "position":
                "rookie_position",
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
    games["kickoff_utc"].notna()
].copy()


# ============================================================
# EXACT PREGAME STARTER RESOLUTION
# ============================================================

resolved = []

for _, row in games.iterrows():

    pid = str(
        row["player_id"]
    )

    team_name = str(
        row["team"]
    )

    position = str(
        row["rookie_position"]
    )

    ko = row["kickoff_utc"]

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
            position
        )
        &
        depth["snapshot_dt"].lt(
            ko
        )
    ].copy()

    if d.empty:
        continue

    latest_dt = (
        d["snapshot_dt"].max()
    )

    latest = d[
        d["snapshot_dt"].eq(
            latest_dt
        )
    ].copy()

    latest = latest.sort_values(
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
        best["pos_rank"]
    )

    result[
        "pregame_depth_snapshot"
    ] = latest_dt

    resolved.append(
        result
    )


resolved = pd.DataFrame(
    resolved
)

if resolved.empty:
    raise RuntimeError(
        "FAIL | zero resolved rookie games"
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
# TEAM TARGET RATE HISTORY
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


# ============================================================
# DATE-BLOCKED ROOKIE PRIORS
# ============================================================

STAT_COLS = {
    "QB": [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],

    "RB": [
        "carries",
        "targets",
        "receptions",
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
    ],

    "WR": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],

    "TE": [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
}


def historical_prior(
    position,
    event_time,
    current_player,
):

    first_prior = first_starts[
        first_starts[
            "rookie_position"
        ].eq(position)
        &
        first_starts[
            "kickoff_utc"
        ].lt(event_time)
        &
        ~first_starts[
            "player_id"
        ].eq(current_player)
    ].copy()

    all_prior = starter_games[
        starter_games[
            "rookie_position"
        ].eq(position)
        &
        starter_games[
            "kickoff_utc"
        ].lt(event_time)
        &
        ~starter_games[
            "player_id"
        ].eq(current_player)
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

    weight = clamp(
        first_n / all_n,
        0.0,
        1.0,
    )

    result = {}

    for stat in STAT_COLS[position]:

        if (
            stat
            not in first_prior.columns
            or stat
            not in all_prior.columns
        ):
            return None

        first_mean = (
            pd.to_numeric(
                first_prior[stat],
                errors="coerce",
            )
            .mean()
        )

        all_mean = (
            pd.to_numeric(
                all_prior[stat],
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
            weight
            * float(first_mean)
            +
            (1.0 - weight)
            * float(all_mean)
        )

    result["_first_n"] = first_n
    result["_all_n"] = all_n
    result["_weight"] = weight

    return result


def historical_target_rate(
    event_time,
):

    history = team_hist[
        team_hist[
            "kickoff_utc"
        ].notna()
        &
        team_hist[
            "kickoff_utc"
        ].lt(event_time)
        &
        team_hist[
            "attempts"
        ].gt(0)
        &
        team_hist[
            "target_per_attempt"
        ].notna()
    ]

    if history.empty:
        return None

    return float(
        history[
            "target_per_attempt"
        ].mean()
    )


# ============================================================
# PRE-GAME TEAM ENVIRONMENT
# ============================================================

def environment_for_event(
    game_id,
    team_name,
    season,
):

    exact = env[
        env["game_id"]
        .astype(str)
        .eq(str(game_id))
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
            return (
                row,
                "EXACT_PREGAME",
            )

    # Week 1 / unavailable current-season history:
    # latest previous-season avg5.
    fallback = env[
        env["team"].eq(
            team_name
        )
        &
        env["season"].lt(
            season
        )
    ].copy()

    fallback = fallback.sort_values(
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
        return None, "NONE"

    row = fallback.iloc[0]

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
        pd.isna(p5)
        or pd.isna(r5)
    ):
        return None, "NONE"

    return (
        row,
        "PRIOR_SEASON_FALLBACK",
    )


# ============================================================
# DVP LOOKUP
# ============================================================

def dvp_for_event(
    game_id,
    opponent,
    position,
    season,
):

    exact = dvp[
        dvp["game_id"]
        .astype(str)
        .eq(str(game_id))
        &
        dvp[
            "defense_team"
        ].eq(opponent)
        &
        dvp[
            "position"
        ].eq(position)
    ].copy()

    if not exact.empty:
        return (
            exact.iloc[0],
            "EXACT_PREGAME",
        )

    fallback = dvp[
        dvp[
            "defense_team"
        ].eq(opponent)
        &
        dvp[
            "position"
        ].eq(position)
        &
        dvp[
            "season"
        ].lt(season)
    ].copy()

    fallback = fallback.sort_values(
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
        return None, "NONE"

    return (
        fallback.iloc[0],
        "PRIOR_SEASON_FALLBACK",
    )


# ============================================================
# BUILD DATE-BLOCKED CALIBRATION EVENTS
# ============================================================

events = []

for _, event in first_starts.sort_values(
    "kickoff_utc"
).iterrows():

    pos = str(
        event[
            "rookie_position"
        ]
    )

    # R8S only calibrates receiving/rushing matchup
    # efficiency. QB remains an R8R control.
    if pos not in {
        "RB",
        "WR",
        "TE",
    }:
        continue

    pid = str(
        event["player_id"]
    )

    ko = event[
        "kickoff_utc"
    ]

    prior = historical_prior(
        pos,
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

    env_row, env_source = (
        environment_for_event(
            event["game_id"],
            event["team"],
            int(event["season"]),
        )
    )

    if env_row is None:
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

    dvp_row, dvp_source = (
        dvp_for_event(
            event["game_id"],
            event[
                "opponent_team"
            ],
            pos,
            int(event["season"]),
        )
    )

    if dvp_row is None:
        continue

    row = {
        "game_id":
            event["game_id"],

        "season":
            int(event["season"]),

        "week":
            int(event["week"]),

        "kickoff_utc":
            ko,

        "team":
            event["team"],

        "opponent_team":
            event[
                "opponent_team"
            ],

        "player_id":
            pid,

        "player_name":
            event.get(
                "player_display_name",
                event.get(
                    "player_name",
                    pid,
                ),
            ),

        "position":
            pos,

        "prior_first_n":
            prior["_first_n"],

        "prior_all_n":
            prior["_all_n"],

        "pass_pool":
            pass_pool,

        "target_pool":
            target_pool,

        "rush_pool":
            rush_pool,

        "environment_source":
            env_source,

        "dvp_source":
            dvp_source,
    }

    # Fixed opportunity layer.
    row[
        "pred_targets"
    ] = min(
        max(
            0.0,
            num(
                prior.get(
                    "targets"
                ),
                0.0,
            ),
        ),
        target_pool,
    )

    row[
        "pred_carries"
    ] = min(
        max(
            0.0,
            num(
                prior.get(
                    "carries"
                ),
                0.0,
            ),
        ),
        rush_pool,
    )

    # Actuals.
    for stat in STAT_COLS[pos]:
        row[
            f"actual_{stat}"
        ] = num(
            event.get(
                stat
            ),
            0.0,
        )

    # ------------------------------
    # Prior efficiency rates
    # ------------------------------

    row[
        "prior_catch_rate"
    ] = div(
        prior.get(
            "receptions"
        ),
        prior.get(
            "targets"
        ),
    )

    row[
        "prior_rec_ypr"
    ] = div(
        prior.get(
            "receiving_yards"
        ),
        prior.get(
            "receptions"
        ),
    )

    row[
        "prior_rec_td_per_target"
    ] = div(
        prior.get(
            "receiving_tds"
        ),
        prior.get(
            "targets"
        ),
    )

    row[
        "prior_rush_ypc"
    ] = div(
        prior.get(
            "rushing_yards"
        ),
        prior.get(
            "carries"
        ),
    )

    row[
        "prior_rush_td_per_carry"
    ] = div(
        prior.get(
            "rushing_tds"
        ),
        prior.get(
            "carries"
        ),
    )

    # ------------------------------
    # DvP efficiency rates
    # ------------------------------

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

    row[
        "dvp_catch_rate"
    ] = div(
        dvp_receptions,
        dvp_targets,
    )

    row[
        "dvp_rec_ypr"
    ] = div(
        dvp_rec_yards,
        dvp_receptions,
    )

    row[
        "dvp_rec_td_per_target"
    ] = div(
        dvp_rec_tds,
        dvp_targets,
    )

    row[
        "dvp_rush_ypc"
    ] = div(
        dvp_rush_yards,
        dvp_carries,
    )

    row[
        "dvp_rush_td_per_carry"
    ] = div(
        dvp_rush_tds,
        dvp_carries,
    )

    events.append(
        row
    )


event_df = pd.DataFrame(
    events
)

if event_df.empty:
    raise RuntimeError(
        "FAIL | zero R8S calibration events"
    )


print()
print("===== R8S EVENT COVERAGE =====")

print(
    event_df.groupby(
        "position"
    )
    .size()
    .to_string()
)


# ============================================================
# COMPONENT DEFINITIONS
#
# Calibrate each efficiency component independently.
# Opportunity (targets/carries) remains fixed.
# ============================================================

COMPONENTS = {
    "RB": {
        "catch_rate": {
            "prior":
                "prior_catch_rate",

            "dvp":
                "dvp_catch_rate",

            "actual":
                "actual_receptions",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * clamp(
                        rate,
                        0.0,
                        1.0,
                    ),
        },

        "receiving_ypr": {
            "prior":
                "prior_rec_ypr",

            "dvp":
                "dvp_rec_ypr",

            "actual":
                "actual_receiving_yards",

            "predict":
                lambda r, rate:
                    (
                        r["pred_targets"]
                        *
                        clamp(
                            weighted_blend(
                                r[
                                    "prior_catch_rate"
                                ],
                                r[
                                    "dvp_catch_rate"
                                ],
                                0.0,
                            ),
                            0.0,
                            1.0,
                        )
                        *
                        max(
                            0.0,
                            rate,
                        )
                    ),
        },

        "receiving_td_rate": {
            "prior":
                "prior_rec_td_per_target",

            "dvp":
                "dvp_rec_td_per_target",

            "actual":
                "actual_receiving_tds",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * max(
                        0.0,
                        rate,
                    ),
        },

        "rushing_ypc": {
            "prior":
                "prior_rush_ypc",

            "dvp":
                "dvp_rush_ypc",

            "actual":
                "actual_rushing_yards",

            "predict":
                lambda r, rate:
                    r["pred_carries"]
                    * max(
                        0.0,
                        rate,
                    ),
        },

        "rushing_td_rate": {
            "prior":
                "prior_rush_td_per_carry",

            "dvp":
                "dvp_rush_td_per_carry",

            "actual":
                "actual_rushing_tds",

            "predict":
                lambda r, rate:
                    r["pred_carries"]
                    * max(
                        0.0,
                        rate,
                    ),
        },
    },

    "WR": {
        "catch_rate": {
            "prior":
                "prior_catch_rate",

            "dvp":
                "dvp_catch_rate",

            "actual":
                "actual_receptions",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * clamp(
                        rate,
                        0.0,
                        1.0,
                    ),
        },

        "receiving_ypr": {
            "prior":
                "prior_rec_ypr",

            "dvp":
                "dvp_rec_ypr",

            "actual":
                "actual_receiving_yards",

            "predict":
                lambda r, rate:
                    (
                        r["pred_targets"]
                        *
                        clamp(
                            r[
                                "prior_catch_rate"
                            ],
                            0.0,
                            1.0,
                        )
                        *
                        max(
                            0.0,
                            rate,
                        )
                    ),
        },

        "receiving_td_rate": {
            "prior":
                "prior_rec_td_per_target",

            "dvp":
                "dvp_rec_td_per_target",

            "actual":
                "actual_receiving_tds",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * max(
                        0.0,
                        rate,
                    ),
        },
    },

    "TE": {
        "catch_rate": {
            "prior":
                "prior_catch_rate",

            "dvp":
                "dvp_catch_rate",

            "actual":
                "actual_receptions",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * clamp(
                        rate,
                        0.0,
                        1.0,
                    ),
        },

        "receiving_ypr": {
            "prior":
                "prior_rec_ypr",

            "dvp":
                "dvp_rec_ypr",

            "actual":
                "actual_receiving_yards",

            "predict":
                lambda r, rate:
                    (
                        r["pred_targets"]
                        *
                        clamp(
                            r[
                                "prior_catch_rate"
                            ],
                            0.0,
                            1.0,
                        )
                        *
                        max(
                            0.0,
                            rate,
                        )
                    ),
        },

        "receiving_td_rate": {
            "prior":
                "prior_rec_td_per_target",

            "dvp":
                "dvp_rec_td_per_target",

            "actual":
                "actual_receiving_tds",

            "predict":
                lambda r, rate:
                    r["pred_targets"]
                    * max(
                        0.0,
                        rate,
                    ),
        },
    },
}


# ============================================================
# GRID CALIBRATION
# ============================================================

detail_rows = []

for position, components in (
    COMPONENTS.items()
):

    p = event_df[
        event_df[
            "position"
        ].eq(position)
    ].copy()

    for component, spec in (
        components.items()
    ):

        for weight in WEIGHTS:

            actual_values = []
            predicted_values = []

            usable_events = 0

            for _, row in p.iterrows():

                prior_value = num(
                    row[
                        spec["prior"]
                    ]
                )

                dvp_value = num(
                    row[
                        spec["dvp"]
                    ]
                )

                actual_value = num(
                    row[
                        spec["actual"]
                    ]
                )

                if (
                    pd.isna(prior_value)
                    or pd.isna(actual_value)
                ):
                    continue

                blended_rate = (
                    weighted_blend(
                        prior_value,
                        dvp_value,
                        weight,
                    )
                )

                if pd.isna(
                    blended_rate
                ):
                    continue

                prediction = (
                    spec["predict"](
                        row,
                        blended_rate,
                    )
                )

                if pd.isna(
                    prediction
                ):
                    continue

                usable_events += 1

                actual_values.append(
                    actual_value
                )

                predicted_values.append(
                    prediction
                )

            if usable_events == 0:
                continue

            metrics = metric_values(
                pd.Series(
                    actual_values
                ),
                pd.Series(
                    predicted_values
                ),
            )

            detail_rows.append(
                {
                    "position":
                        position,

                    "component":
                        component,

                    "dvp_weight":
                        weight,

                    "prior_weight":
                        1.0 - weight,

                    "n":
                        metrics["n"],

                    "mae":
                        metrics["mae"],

                    "rmse":
                        metrics["rmse"],

                    "bias":
                        metrics["bias"],
                }
            )


detail = pd.DataFrame(
    detail_rows
)

if detail.empty:
    raise RuntimeError(
        "FAIL | no blend results"
    )


# ============================================================
# SELECT DIAGNOSTIC WINNER
#
# Primary: MAE
# Tiebreak 1: RMSE
# Tiebreak 2: lower DvP weight
#
# Lower matchup weight wins exact statistical ties.
# ============================================================

summary_rows = []

for (
    position,
    component,
), group in detail.groupby(
    [
        "position",
        "component",
    ]
):

    group = group.sort_values(
        [
            "mae",
            "rmse",
            "dvp_weight",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    best = group.iloc[0]

    baseline = group[
        group[
            "dvp_weight"
        ].eq(0.0)
    ]

    if baseline.empty:
        raise RuntimeError(
            "FAIL | missing 0.00 baseline "
            f"{position} {component}"
        )

    baseline = baseline.iloc[0]

    n = int(
        best["n"]
    )

    if n >= 8:
        evidence = (
            "CALIBRATION_CANDIDATE"
        )

    elif n >= 5:
        evidence = (
            "WEAK_EVIDENCE"
        )

    else:
        evidence = (
            "INSUFFICIENT"
        )

    improvement = (
        float(
            baseline["mae"]
        )
        -
        float(
            best["mae"]
        )
    )

    improvement_pct = (
        improvement
        /
        float(
            baseline["mae"]
        )
        * 100.0
        if float(
            baseline["mae"]
        ) > 0
        else 0.0
    )

    # Fail-closed usable weight:
    #
    # - insufficient sample -> 0 matchup weight
    # - best candidate must actually improve MAE
    # - otherwise stay at raw prior
    if (
        evidence
        == "INSUFFICIENT"
        or improvement <= 0.0
    ):
        shadow_recommended_weight = (
            0.0
        )
    else:
        shadow_recommended_weight = (
            float(
                best[
                    "dvp_weight"
                ]
            )
        )

    summary_rows.append(
        {
            "position":
                position,

            "component":
                component,

            "n":
                n,

            "baseline_dvp_weight":
                0.0,

            "baseline_mae":
                float(
                    baseline["mae"]
                ),

            "baseline_rmse":
                float(
                    baseline["rmse"]
                ),

            "baseline_bias":
                float(
                    baseline["bias"]
                ),

            "best_diagnostic_weight":
                float(
                    best[
                        "dvp_weight"
                    ]
                ),

            "best_mae":
                float(
                    best["mae"]
                ),

            "best_rmse":
                float(
                    best["rmse"]
                ),

            "best_bias":
                float(
                    best["bias"]
                ),

            "mae_improvement":
                improvement,

            "mae_improvement_pct":
                improvement_pct,

            "evidence":
                evidence,

            "shadow_recommended_weight":
                shadow_recommended_weight,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# BUILD COMBINED SHADOW MODEL
#
# Uses fail-closed shadow_recommended_weight.
#
# This is still calibration output only.
# ============================================================

weight_lookup = {
    (
        row["position"],
        row["component"],
    ):
        float(
            row[
                "shadow_recommended_weight"
            ]
        )

    for _, row
    in summary.iterrows()
}


combined_rows = []

for _, row in event_df.iterrows():

    position = row[
        "position"
    ]

    if position not in COMPONENTS:
        continue

    pred = {
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "carries": num(
            row[
                "pred_carries"
            ],
            0.0,
        ),
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
    }

    baseline = {
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "carries": num(
            row[
                "pred_carries"
            ],
            0.0,
        ),
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
    }

    # ------------------------------
    # Receiving
    # ------------------------------

    catch_w = weight_lookup[
        (
            position,
            "catch_rate",
        )
    ]

    ypr_w = weight_lookup[
        (
            position,
            "receiving_ypr",
        )
    ]

    td_w = weight_lookup[
        (
            position,
            "receiving_td_rate",
        )
    ]

    catch_rate = clamp(
        weighted_blend(
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

    baseline_catch = clamp(
        row[
            "prior_catch_rate"
        ],
        0.0,
        1.0,
    )

    ypr = max(
        0.0,
        weighted_blend(
            row[
                "prior_rec_ypr"
            ],
            row[
                "dvp_rec_ypr"
            ],
            ypr_w,
        ),
    )

    baseline_ypr = max(
        0.0,
        num(
            row[
                "prior_rec_ypr"
            ],
            0.0,
        ),
    )

    td_rate = max(
        0.0,
        weighted_blend(
            row[
                "prior_rec_td_per_target"
            ],
            row[
                "dvp_rec_td_per_target"
            ],
            td_w,
        ),
    )

    baseline_td_rate = max(
        0.0,
        num(
            row[
                "prior_rec_td_per_target"
            ],
            0.0,
        ),
    )

    targets = num(
        row["pred_targets"],
        0.0,
    )

    pred["receptions"] = (
        targets
        * catch_rate
    )

    pred["receiving_yards"] = (
        pred["receptions"]
        * ypr
    )

    pred["receiving_tds"] = (
        targets
        * td_rate
    )

    baseline["receptions"] = (
        targets
        * baseline_catch
    )

    baseline["receiving_yards"] = (
        baseline["receptions"]
        * baseline_ypr
    )

    baseline["receiving_tds"] = (
        targets
        * baseline_td_rate
    )

    # ------------------------------
    # Rushing — RB only calibrated
    # ------------------------------

    if position == "RB":

        ypc_w = weight_lookup[
            (
                position,
                "rushing_ypc",
            )
        ]

        rush_td_w = weight_lookup[
            (
                position,
                "rushing_td_rate",
            )
        ]

        ypc = max(
            0.0,
            weighted_blend(
                row[
                    "prior_rush_ypc"
                ],
                row[
                    "dvp_rush_ypc"
                ],
                ypc_w,
            ),
        )

        baseline_ypc = max(
            0.0,
            num(
                row[
                    "prior_rush_ypc"
                ],
                0.0,
            ),
        )

        rush_td_rate = max(
            0.0,
            weighted_blend(
                row[
                    "prior_rush_td_per_carry"
                ],
                row[
                    "dvp_rush_td_per_carry"
                ],
                rush_td_w,
            ),
        )

        baseline_rush_td_rate = max(
            0.0,
            num(
                row[
                    "prior_rush_td_per_carry"
                ],
                0.0,
            ),
        )

        carries = num(
            row[
                "pred_carries"
            ],
            0.0,
        )

        pred[
            "rushing_yards"
        ] = (
            carries
            * ypc
        )

        pred[
            "rushing_tds"
        ] = (
            carries
            * rush_td_rate
        )

        baseline[
            "rushing_yards"
        ] = (
            carries
            * baseline_ypc
        )

        baseline[
            "rushing_tds"
        ] = (
            carries
            * baseline_rush_td_rate
        )

    # WR rushing intentionally stays on baseline
    # and is excluded from R8S DvP calibration.
    elif position == "WR":

        carries = num(
            row[
                "pred_carries"
            ],
            0.0,
        )

        prior_ypc = num(
            row[
                "prior_rush_ypc"
            ],
            0.0,
        )

        prior_td = num(
            row[
                "prior_rush_td_per_carry"
            ],
            0.0,
        )

        pred[
            "rushing_yards"
        ] = (
            carries
            * prior_ypc
        )

        pred[
            "rushing_tds"
        ] = (
            carries
            * prior_td
        )

        baseline[
            "rushing_yards"
        ] = pred[
            "rushing_yards"
        ]

        baseline[
            "rushing_tds"
        ] = pred[
            "rushing_tds"
        ]

    actual = {
        "receptions":
            num(
                row.get(
                    "actual_receptions"
                ),
                0.0,
            ),

        "receiving_yards":
            num(
                row.get(
                    "actual_receiving_yards"
                ),
                0.0,
            ),

        "receiving_tds":
            num(
                row.get(
                    "actual_receiving_tds"
                ),
                0.0,
            ),

        "rushing_yards":
            num(
                row.get(
                    "actual_rushing_yards"
                ),
                0.0,
            ),

        "rushing_tds":
            num(
                row.get(
                    "actual_rushing_tds"
                ),
                0.0,
            ),
    }

    combined_rows.append(
        {
            "game_id":
                row["game_id"],

            "player_id":
                row["player_id"],

            "player_name":
                row["player_name"],

            "position":
                position,

            "actual_fd_points":
                fd_points(
                    actual
                ),

            "baseline_fd_points":
                fd_points(
                    baseline
                ),

            "combined_fd_points":
                fd_points(
                    pred
                ),
        }
    )


combined = pd.DataFrame(
    combined_rows
)


combined_summary_rows = []

for position, group in (
    combined.groupby(
        "position"
    )
):

    baseline_metrics = metric_values(
        group[
            "actual_fd_points"
        ],
        group[
            "baseline_fd_points"
        ],
    )

    combined_metrics = metric_values(
        group[
            "actual_fd_points"
        ],
        group[
            "combined_fd_points"
        ],
    )

    combined_summary_rows.append(
        {
            "position":
                position,

            "n":
                len(group),

            "baseline_fd_mae":
                baseline_metrics["mae"],

            "combined_fd_mae":
                combined_metrics["mae"],

            "fd_mae_delta":
                (
                    combined_metrics["mae"]
                    -
                    baseline_metrics["mae"]
                ),

            "baseline_fd_rmse":
                baseline_metrics["rmse"],

            "combined_fd_rmse":
                combined_metrics["rmse"],

            "fd_rmse_delta":
                (
                    combined_metrics["rmse"]
                    -
                    baseline_metrics["rmse"]
                ),

            "baseline_fd_bias":
                baseline_metrics["bias"],

            "combined_fd_bias":
                combined_metrics["bias"],
        }
    )


combined_summary = pd.DataFrame(
    combined_summary_rows
)


# ============================================================
# SAVE CALIBRATION ARTIFACTS
# ============================================================

OUT_DETAIL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail.to_csv(
    OUT_DETAIL,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

combined_summary.to_csv(
    OUT_COMBINED,
    index=False,
)


canonical_sha_after = sha256(
    CANONICAL_PATH
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
        "STAGE26G-R8S",

    "purpose":
        "strict date-blocked DvP blend calibration",

    "weights_tested":
        WEIGHTS,

    "events":
        int(
            len(event_df)
        ),

    "events_by_position":
        {
            str(k): int(v)
            for k, v
            in event_df.groupby(
                "position"
            )
            .size()
            .to_dict()
            .items()
        },

    "selection_rule":
        "min MAE, then RMSE, then lower DvP weight",

    "insufficient_sample_rule":
        "n < 5 => recommended DvP weight forced to 0.0",

    "candidate_sample_rule":
        "n >= 8 => calibration candidate only, not production promotion",

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

    "production_mutation":
        False,

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
print("R8S COMPONENT CALIBRATION")
print("============================================================")

show = summary[
    [
        "position",
        "component",
        "n",
        "baseline_mae",
        "best_diagnostic_weight",
        "best_mae",
        "mae_improvement",
        "mae_improvement_pct",
        "evidence",
        "shadow_recommended_weight",
    ]
].sort_values(
    [
        "position",
        "component",
    ]
)

print(
    show.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8S FULL WEIGHT GRID")
print("============================================================")

for (
    position,
    component,
), group in detail.groupby(
    [
        "position",
        "component",
    ]
):

    print()
    print(
        f"{position} | {component}"
    )

    print(
        group[
            [
                "dvp_weight",
                "n",
                "mae",
                "rmse",
                "bias",
            ]
        ]
        .sort_values(
            "dvp_weight"
        )
        .to_string(
            index=False
        )
    )


print()
print("============================================================")
print("R8S COMBINED SHADOW MODEL")
print("============================================================")

print(
    combined_summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8S INTERPRETATION")
print("============================================================")

for _, row in summary.sort_values(
    [
        "position",
        "component",
    ]
).iterrows():

    print(
        f"{row['position']} | "
        f"{row['component']} | "
        f"n={int(row['n'])} | "
        f"best_diagnostic="
        f"{row['best_diagnostic_weight']:.2f} | "
        f"recommended_shadow="
        f"{row['shadow_recommended_weight']:.2f} | "
        f"{row['evidence']} | "
        f"MAE improvement="
        f"{row['mae_improvement']:.6f}"
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
    "PRODUCTION_MUTATION=NONE"
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
    f"DETAIL={OUT_DETAIL}"
)

print(
    f"SUMMARY={OUT_SUMMARY}"
)

print(
    f"COMBINED={OUT_COMBINED}"
)

print(
    f"AUDIT={OUT_AUDIT}"
)


print()
print("============================================================")
print("STAGE26G_R8S_DVP_BLEND_CALIBRATION=PASS")
print("============================================================")
