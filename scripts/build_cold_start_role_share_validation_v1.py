from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

STATS_PATH = ROOT / "data/parquet/nfl_player_game_stats.parquet"
ROSTER_PATH = ROOT / "data/parquet/nfl_weekly_rosters.parquet"
SCHEDULE_PATH = ROOT / "data/parquet/nfl_schedule.parquet"
TEAM_PATH = ROOT / "data/parquet/nfl_team_game_stats.parquet"
ENV_PATH = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
DB_PATH = ROOT / "data/nfl.db"

CANONICAL_PATH = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

DETAIL_OUT = (
    ROOT
    / "processed/stage26gr8y_role_share_validation_detail.csv"
)

SUMMARY_OUT = (
    ROOT
    / "processed/stage26gr8y_role_share_validation_summary.csv"
)

AUDIT_OUT = (
    ROOT
    / "processed/stage26gr8y_role_share_validation_audit.json"
)

EXPECTED_CANONICAL_SHA = (
    "6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9"
)

TARGET_RATE_FALLBACK = 0.954436

SUPPORTED = {"RB", "FB", "WR", "TE"}

MIN_PRIOR_ROWS = 3


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


def safe_div(a, b):
    a = num(a)
    b = num(b)

    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return np.nan

    return a / b


def parse_kickoff(game_date, gametime):
    if pd.isna(game_date) or pd.isna(gametime):
        return pd.NaT

    try:
        date_part = pd.Timestamp(game_date).date()

        local = pd.Timestamp(
            f"{date_part} {str(gametime).strip()}",
            tz=ZoneInfo("America/New_York"),
        )

        return local.tz_convert("UTC")

    except Exception:
        return pd.NaT


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    mask = (
        np.isfinite(a)
        &
        np.isfinite(p)
    )

    a = a[mask]
    p = p[mask]

    if len(a) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    err = p - a

    return {
        "n": int(len(a)),
        "mae": float(
            np.mean(
                np.abs(err)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),
        "bias": float(
            np.mean(err)
        ),
    }


# ============================================================
# SAFETY
# ============================================================

for path in [
    STATS_PATH,
    ROSTER_PATH,
    SCHEDULE_PATH,
    TEAM_PATH,
    ENV_PATH,
    DB_PATH,
    CANONICAL_PATH,
]:
    if not path.exists():
        raise RuntimeError(
            f"FAIL | missing required input: {path}"
        )

canonical_sha_before = sha256(
    CANONICAL_PATH
)

if canonical_sha_before != EXPECTED_CANONICAL_SHA:
    raise RuntimeError(
        "FAIL | canonical SHA mismatch\n"
        f"EXPECTED={EXPECTED_CANONICAL_SHA}\n"
        f"ACTUAL={canonical_sha_before}"
    )


# ============================================================
# LOAD
# ============================================================

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

for df in [
    stats,
    roster,
    team,
    env,
]:
    if "team" in df.columns:
        df["team"] = (
            df["team"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

stats["position"] = (
    stats["position"]
    .fillna("")
    .astype(str)
    .str.upper()
)

roster["position"] = (
    roster["position"]
    .fillna("")
    .astype(str)
    .str.upper()
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

depth["pos_slot"] = pd.to_numeric(
    depth["pos_slot"],
    errors="coerce",
)


# ============================================================
# KICKOFF
# ============================================================

schedule["kickoff_utc"] = [
    parse_kickoff(d, t)
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
    .drop_duplicates("game_id")
)


# ============================================================
# TRUE ROOKIES
# ============================================================

rookies = roster[
    roster["gsis_id"].notna()
    &
    roster["position"].isin(SUPPORTED)
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
            "gsis_id": "player_id",
            "position": "rookie_position",
        }
    )
)


# ============================================================
# ROOKIE GAMES
# ============================================================

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
# EXACT PREGAME STARTERS
# ============================================================

resolved = []

for _, row in games.iterrows():

    pid = str(
        row["player_id"]
    )

    team_name = str(
        row["team"]
    ).upper()

    position = str(
        row["rookie_position"]
    ).upper()

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
        d["snapshot_dt"]
        .max()
    )

    latest = d[
        d["snapshot_dt"].eq(
            latest_dt
        )
    ].copy()

    if latest.empty:
        continue

    latest = latest.sort_values(
        [
            "pos_rank",
            "pos_slot",
        ],
        na_position="last",
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
    pd.to_numeric(
        resolved[
            "pregame_depth_rank"
        ],
        errors="coerce",
    ).eq(1)
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
        sort=False,
        as_index=False,
    )
    .head(1)
    .copy()
)


# ============================================================
# ACTUAL TEAM OPPORTUNITY
# ============================================================

team_actual = team[
    [
        "game_id",
        "season",
        "week",
        "team",
        "carries",
        "targets",
    ]
].copy()

team_actual = team_actual.rename(
    columns={
        "carries":
            "actual_team_carries",

        "targets":
            "actual_team_targets",
    }
)

starter_games = starter_games.merge(
    team_actual,
    on=[
        "game_id",
        "season",
        "week",
        "team",
    ],
    how="left",
)


# ============================================================
# ACTUAL QB CARRIES
# ============================================================

qb_actual = stats[
    stats["position"].eq(
        "QB"
    )
].copy()

qb_actual["carries"] = (
    pd.to_numeric(
        qb_actual["carries"],
        errors="coerce",
    )
    .fillna(0.0)
)

qb_carries = (
    qb_actual
    .groupby(
        [
            "game_id",
            "season",
            "week",
            "team",
        ],
        as_index=False,
    )["carries"]
    .sum()
    .rename(
        columns={
            "carries":
                "actual_qb_carries",
        }
    )
)

starter_games = starter_games.merge(
    qb_carries,
    on=[
        "game_id",
        "season",
        "week",
        "team",
    ],
    how="left",
)

starter_games[
    "actual_qb_carries"
] = (
    pd.to_numeric(
        starter_games[
            "actual_qb_carries"
        ],
        errors="coerce",
    )
    .fillna(0.0)
)

for c in [
    "actual_team_carries",
    "actual_team_targets",
    "carries",
    "targets",
]:
    starter_games[c] = pd.to_numeric(
        starter_games[c],
        errors="coerce",
    )

starter_games[
    "actual_non_qb_rush_pool"
] = (
    starter_games[
        "actual_team_carries"
    ]
    -
    starter_games[
        "actual_qb_carries"
    ]
)

starter_games[
    "actual_non_qb_rush_share"
] = np.where(
    starter_games[
        "actual_non_qb_rush_pool"
    ].gt(0),
    starter_games["carries"]
    /
    starter_games[
        "actual_non_qb_rush_pool"
    ],
    np.nan,
)

starter_games[
    "actual_team_target_share"
] = np.where(
    starter_games[
        "actual_team_targets"
    ].gt(0),
    starter_games["targets"]
    /
    starter_games[
        "actual_team_targets"
    ],
    np.nan,
)


# ============================================================
# FIRST START KEYS
# ============================================================

first_keys = set(
    zip(
        first_starts[
            "game_id"
        ].astype(str),

        first_starts[
            "player_id"
        ].astype(str),
    )
)

starter_games[
    "is_first_start"
] = [
    (
        str(gid),
        str(pid),
    ) in first_keys
    for gid, pid in zip(
        starter_games["game_id"],
        starter_games["player_id"],
    )
]


# ============================================================
# PREGAME ENVIRONMENT AUTHORITY
#
# Exact current game first.
# If unavailable, latest prior-season row for same team.
# Mirrors the previously validated R8O/R8S family logic.
# ============================================================

for col in [
    "pass_attempts_avg_5",
    "rush_attempts_avg_5",
]:
    env[col] = pd.to_numeric(
        env[col],
        errors="coerce",
    )


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
        env["team"]
        .astype(str)
        .str.upper()
        .eq(str(team_name).upper())
    ].copy()

    if not exact.empty:

        exact = exact.sort_values(
            [
                c
                for c in [
                    "season",
                    "week",
                ]
                if c in exact.columns
            ]
        )

        row = exact.iloc[-1]

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
            np.isfinite(p5)
            and p5 > 0
            and np.isfinite(r5)
            and r5 > 0
        ):
            return row, "EXACT_GAME"

    prior = env[
        env["team"]
        .astype(str)
        .str.upper()
        .eq(str(team_name).upper())
        &
        pd.to_numeric(
            env["season"],
            errors="coerce",
        ).lt(
            int(season)
        )
    ].copy()

    if prior.empty:
        return None, "NONE"

    sort_cols = [
        c
        for c in [
            "season",
            "week",
        ]
        if c in prior.columns
    ]

    prior = prior.sort_values(
        sort_cols
    )

    row = prior.iloc[-1]

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
        np.isfinite(p5)
        and p5 > 0
        and np.isfinite(r5)
        and r5 > 0
    ):
        return row, "PRIOR_SEASON"

    return None, "NONE"


# ============================================================
# DATE-BLOCKED PRIORS
# ============================================================

def historical_prior_rows(
    position,
    event_time,
    current_player,
):

    all_prior = starter_games[
        starter_games[
            "rookie_position"
        ]
        .astype(str)
        .str.upper()
        .eq(position)
        &
        starter_games[
            "kickoff_utc"
        ]
        .lt(event_time)
        &
        ~starter_games[
            "player_id"
        ]
        .astype(str)
        .eq(current_player)
    ].copy()

    first_prior = all_prior[
        all_prior[
            "is_first_start"
        ]
    ].copy()

    return (
        first_prior,
        all_prior,
    )


def hierarchical_value(
    first_values,
    all_values,
):

    first_values = pd.to_numeric(
        first_values,
        errors="coerce",
    ).dropna()

    all_values = pd.to_numeric(
        all_values,
        errors="coerce",
    ).dropna()

    first_n = len(
        first_values
    )

    all_n = len(
        all_values
    )

    if all_n < MIN_PRIOR_ROWS:
        return None

    all_mean = float(
        all_values.mean()
    )

    if first_n > 0:
        first_mean = float(
            first_values.mean()
        )
    else:
        first_mean = all_mean

    weight = (
        first_n / all_n
        if all_n > 0
        else 0.0
    )

    value = (
        weight * first_mean
        +
        (1.0 - weight) * all_mean
    )

    return {
        "value": value,
        "first_n": first_n,
        "all_n": all_n,
        "first_weight": weight,
    }


# ============================================================
# BUILD HELD-OUT FIRST-START EVENTS
# ============================================================

rows = []

for _, event in first_starts.sort_values(
    "kickoff_utc"
).iterrows():

    pos = str(
        event[
            "rookie_position"
        ]
    ).upper()

    if pos not in SUPPORTED:
        continue

    pid = str(
        event["player_id"]
    )

    ko = event[
        "kickoff_utc"
    ]

    first_prior, all_prior = (
        historical_prior_rows(
            pos,
            ko,
            pid,
        )
    )

    absolute_carry_prior = hierarchical_value(
        first_prior[
            "carries"
        ],
        all_prior[
            "carries"
        ],
    )

    absolute_target_prior = hierarchical_value(
        first_prior[
            "targets"
        ],
        all_prior[
            "targets"
        ],
    )

    rush_share_prior = hierarchical_value(
        first_prior[
            "actual_non_qb_rush_share"
        ],
        all_prior[
            "actual_non_qb_rush_share"
        ],
    )

    target_share_prior = hierarchical_value(
        first_prior[
            "actual_team_target_share"
        ],
        all_prior[
            "actual_team_target_share"
        ],
    )

    if any(
        x is None
        for x in [
            absolute_carry_prior,
            absolute_target_prior,
            rush_share_prior,
            target_share_prior,
        ]
    ):
        continue

    env_row, env_source = (
        environment_for_event(
            event["game_id"],
            event["team"],
            int(
                event["season"]
            ),
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
        not np.isfinite(pass_pool)
        or pass_pool <= 0
        or not np.isfinite(rush_pool)
        or rush_pool <= 0
    ):
        continue

    target_pool = (
        pass_pool
        * TARGET_RATE_FALLBACK
    )

    # Pregame QB carry estimate:
    # date-blocked historical average QB share of team rush attempts
    # for this team, using only games strictly before event kickoff.

    team_before = stats[
        stats["team"]
        .astype(str)
        .str.upper()
        .eq(
            str(
                event["team"]
            ).upper()
        )
    ].merge(
        kickoff,
        on=[
            "game_id",
            "season",
            "week",
        ],
        how="left",
    )

    team_before = team_before[
        team_before[
            "kickoff_utc"
        ].lt(ko)
    ].copy()

    qb_before = (
        team_before[
            team_before[
                "position"
            ]
            .astype(str)
            .str.upper()
            .eq("QB")
        ]
        .groupby(
            "game_id",
            as_index=False,
        )["carries"]
        .sum()
    )

    team_before_totals = (
        team[
            team["team"]
            .astype(str)
            .str.upper()
            .eq(
                str(
                    event["team"]
                ).upper()
            )
        ]
        .merge(
            kickoff,
            on=[
                "game_id",
                "season",
                "week",
            ],
            how="left",
        )
    )

    team_before_totals = (
        team_before_totals[
            team_before_totals[
                "kickoff_utc"
            ].lt(ko)
        ][
            [
                "game_id",
                "carries",
            ]
        ]
    )

    qb_share_hist = (
        team_before_totals
        .merge(
            qb_before,
            on="game_id",
            how="left",
            suffixes=(
                "_team",
                "_qb",
            ),
        )
    )

    qb_share_hist[
        "carries_qb"
    ] = pd.to_numeric(
        qb_share_hist[
            "carries_qb"
        ],
        errors="coerce",
    ).fillna(0.0)

    qb_share_hist[
        "carries_team"
    ] = pd.to_numeric(
        qb_share_hist[
            "carries_team"
        ],
        errors="coerce",
    )

    qb_share_hist[
        "qb_rush_share"
    ] = np.where(
        qb_share_hist[
            "carries_team"
        ].gt(0),
        qb_share_hist[
            "carries_qb"
        ]
        /
        qb_share_hist[
            "carries_team"
        ],
        np.nan,
    )

    qb_share_hist = qb_share_hist[
        np.isfinite(
            qb_share_hist[
                "qb_rush_share"
            ]
        )
    ]

    if len(qb_share_hist) > 0:
        qb_share = float(
            qb_share_hist[
                "qb_rush_share"
            ]
            .tail(5)
            .mean()
        )
    else:
        qb_share = 0.0

    qb_share = min(
        max(
            qb_share,
            0.0,
        ),
        1.0,
    )

    projected_qb_carries = (
        rush_pool
        * qb_share
    )

    projected_non_qb_pool = max(
        0.0,
        rush_pool
        - projected_qb_carries
    )

    role_share_carries = (
        projected_non_qb_pool
        * rush_share_prior[
            "value"
        ]
    )

    role_share_targets = (
        target_pool
        * target_share_prior[
            "value"
        ]
    )

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

        "actual_carries":
            num(
                event["carries"],
                0.0,
            ),

        "actual_targets":
            num(
                event["targets"],
                0.0,
            ),

        "absolute_pred_carries":
            absolute_carry_prior[
                "value"
            ],

        "absolute_pred_targets":
            absolute_target_prior[
                "value"
            ],

        "role_share_pred_carries":
            role_share_carries,

        "role_share_pred_targets":
            role_share_targets,

        "rush_share_prior":
            rush_share_prior[
                "value"
            ],

        "target_share_prior":
            target_share_prior[
                "value"
            ],

        "pass_pool":
            pass_pool,

        "target_pool":
            target_pool,

        "rush_pool":
            rush_pool,

        "projected_qb_carries":
            projected_qb_carries,

        "projected_non_qb_pool":
            projected_non_qb_pool,

        "env_source":
            env_source,

        "prior_all_n":
            all_prior.shape[0],

        "prior_first_n":
            first_prior.shape[0],
    }

    rows.append(
        row
    )


detail = pd.DataFrame(
    rows
)

if detail.empty:
    raise RuntimeError(
        "FAIL | zero held-out validation events"
    )


# ============================================================
# METRICS
# ============================================================

summary_rows = []

evaluation_map = {
    "RB": [
        ("carries", "actual_carries"),
        ("targets", "actual_targets"),
    ],

    "WR": [
        ("targets", "actual_targets"),
    ],

    "TE": [
        ("targets", "actual_targets"),
    ],

    "FB": [
        ("carries", "actual_carries"),
        ("targets", "actual_targets"),
    ],
}

for pos, metrics_to_check in (
    evaluation_map.items()
):

    p = detail[
        detail[
            "position"
        ].eq(pos)
    ].copy()

    if p.empty:
        continue

    for metric_name, actual_col in (
        metrics_to_check
    ):

        abs_col = (
            "absolute_pred_carries"
            if metric_name == "carries"
            else "absolute_pred_targets"
        )

        role_col = (
            "role_share_pred_carries"
            if metric_name == "carries"
            else "role_share_pred_targets"
        )

        m_abs = metrics(
            p[actual_col],
            p[abs_col],
        )

        m_role = metrics(
            p[actual_col],
            p[role_col],
        )

        if (
            m_abs["n"] == 0
            or m_role["n"] == 0
        ):
            continue

        summary_rows.append(
            {
                "position":
                    pos,

                "metric":
                    metric_name,

                "n":
                    m_abs["n"],

                "absolute_mae":
                    m_abs["mae"],

                "role_share_mae":
                    m_role["mae"],

                "mae_delta_role_minus_abs":
                    m_role["mae"]
                    - m_abs["mae"],

                "mae_improvement_pct":
                    (
                        (
                            m_abs["mae"]
                            - m_role["mae"]
                        )
                        /
                        m_abs["mae"]
                        * 100.0
                    )
                    if m_abs["mae"] > 0
                    else np.nan,

                "absolute_rmse":
                    m_abs["rmse"],

                "role_share_rmse":
                    m_role["rmse"],

                "rmse_delta_role_minus_abs":
                    m_role["rmse"]
                    - m_abs["rmse"],

                "absolute_bias":
                    m_abs["bias"],

                "role_share_bias":
                    m_role["bias"],

                "mae_gate":
                    bool(
                        m_role["mae"]
                        < m_abs["mae"]
                    ),

                "rmse_gate":
                    bool(
                        m_role["rmse"]
                        <= m_abs["rmse"]
                    ),
            }
        )


summary = pd.DataFrame(
    summary_rows
)

if summary.empty:
    raise RuntimeError(
        "FAIL | zero validation summary rows"
    )


# ============================================================
# EVENT-LEVEL WIN COUNTS
# ============================================================

event_win_rows = []

for pos, metrics_to_check in (
    evaluation_map.items()
):

    p = detail[
        detail[
            "position"
        ].eq(pos)
    ].copy()

    if p.empty:
        continue

    for metric_name, actual_col in (
        metrics_to_check
    ):

        abs_col = (
            "absolute_pred_carries"
            if metric_name == "carries"
            else "absolute_pred_targets"
        )

        role_col = (
            "role_share_pred_carries"
            if metric_name == "carries"
            else "role_share_pred_targets"
        )

        wins = 0
        ties = 0
        losses = 0

        for _, row in p.iterrows():

            actual = num(
                row[actual_col]
            )

            absolute = num(
                row[abs_col]
            )

            role = num(
                row[role_col]
            )

            if not all(
                np.isfinite(x)
                for x in [
                    actual,
                    absolute,
                    role,
                ]
            ):
                continue

            abs_err = abs(
                absolute
                - actual
            )

            role_err = abs(
                role
                - actual
            )

            if role_err < abs_err - 1e-12:
                wins += 1

            elif abs_err < role_err - 1e-12:
                losses += 1

            else:
                ties += 1

        event_win_rows.append(
            {
                "position": pos,
                "metric": metric_name,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_gate":
                    wins > losses,
            }
        )


event_wins = pd.DataFrame(
    event_win_rows
)

summary = summary.merge(
    event_wins,
    on=[
        "position",
        "metric",
    ],
    how="left",
)


# ============================================================
# PROMOTION DIAGNOSTIC
#
# Research recommendation only.
# No production mutation.
# ============================================================

summary[
    "promotion_candidate"
] = (
    summary[
        "mae_gate"
    ]
    &
    summary[
        "rmse_gate"
    ]
    &
    summary[
        "win_gate"
    ]
)

all_primary_pass = bool(
    summary[
        "promotion_candidate"
    ].all()
)

diagnostic = (
    "ROLE_SHARE_OPPORTUNITY_CANDIDATE"
    if all_primary_pass
    else
    "ROLE_SHARE_OPPORTUNITY_NOT_YET_VALIDATED"
)


# ============================================================
# WRITE
# ============================================================

DETAIL_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail.to_csv(
    DETAIL_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# SAFETY
# ============================================================

canonical_sha_after = sha256(
    CANONICAL_PATH
)

if (
    canonical_sha_after
    != canonical_sha_before
):
    raise RuntimeError(
        "FAIL | canonical forecast mutated"
    )

audit = {
    "stage":
        "STAGE26G-R8Y",

    "purpose":
        "strict date-blocked held-out first-start comparison of absolute opportunity priors versus role-share opportunity priors",

    "validation_population":
        "true rookie pregame rank1 first starts",

    "absolute_baseline":
        "hierarchical historical absolute carries/targets",

    "role_share_candidate":
        "hierarchical historical role share x pregame projected team opportunity",

    "qb_carry_reservation":
        "team-specific last-5 historical QB rush share prior to event",

    "target_pool":
        f"pass_attempts_avg_5 x {TARGET_RATE_FALLBACK}",

    "summary_rows":
        int(
            len(summary)
        ),

    "events":
        int(
            len(detail)
        ),

    "all_primary_pass":
        all_primary_pass,

    "diagnostic":
        diagnostic,

    "canonical_sha_before":
        canonical_sha_before,

    "canonical_sha_after":
        canonical_sha_after,

    "canonical_unchanged":
        canonical_sha_before
        == canonical_sha_after,

    "production_promotion":
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

AUDIT_OUT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
        default=str,
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("R8Y HELD-OUT EVENT COVERAGE")
print("============================================================")

print(
    detail.groupby(
        "position"
    )
    .size()
    .to_string()
)


print()
print("============================================================")
print("R8Y OPPORTUNITY MODEL COMPARISON")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8Y EVENT DETAIL")
print("============================================================")

show_cols = [
    "game_id",
    "season",
    "week",
    "team",
    "player_name",
    "position",
    "actual_carries",
    "absolute_pred_carries",
    "role_share_pred_carries",
    "actual_targets",
    "absolute_pred_targets",
    "role_share_pred_targets",
    "rush_pool",
    "projected_qb_carries",
    "projected_non_qb_pool",
    "target_pool",
    "prior_first_n",
    "prior_all_n",
    "env_source",
]

print(
    detail
    .sort_values(
        [
            "position",
            "kickoff_utc",
        ]
    )[
        show_cols
    ]
    .to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8Y PROMOTION DIAGNOSTIC")
print("============================================================")

print(
    f"ALL_PRIMARY_PASS={all_primary_pass}"
)

print(
    f"DIAGNOSTIC={diagnostic}"
)


print()
print("============================================================")
print("SAFETY / MUTATION AUDIT")
print("============================================================")

print(
    f"CANONICAL_SHA_BEFORE={canonical_sha_before}"
)

print(
    f"CANONICAL_SHA_AFTER={canonical_sha_after}"
)

print(
    "CANONICAL_UNCHANGED="
    f"{canonical_sha_before == canonical_sha_after}"
)

print("PRODUCTION_PROMOTION=NONE")
print("PUBLISHER_MUTATION=NONE")
print("ANALYST_MUTATION=NONE")
print("APP_MUTATION=NONE")
print("SOLVER_MUTATION=NONE")
print("TEAM_ENVIRONMENT_MUTATION=NONE")


print()
print("============================================================")
print("ARTIFACTS")
print("============================================================")

print(
    f"DETAIL={DETAIL_OUT}"
)

print(
    f"SUMMARY={SUMMARY_OUT}"
)

print(
    f"AUDIT={AUDIT_OUT}"
)


print()
print("============================================================")
print("STAGE26G_R8Y_ROLE_SHARE_VALIDATION=PASS")
print("============================================================")
