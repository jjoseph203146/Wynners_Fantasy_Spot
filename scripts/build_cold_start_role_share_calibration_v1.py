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
DB_PATH = ROOT / "data/nfl.db"

R8V_PATH = (
    ROOT
    / "data/parquet/current_cold_start_full_team_shadow_v2.parquet"
)

R8V_AUDIT_PATH = (
    ROOT
    / "data/parquet/current_cold_start_full_team_shadow_v2_audit.json"
)

CANONICAL_PATH = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

DETAIL_OUT = (
    ROOT
    / "processed/stage26gr8x_role_share_detail.csv"
)

SUMMARY_OUT = (
    ROOT
    / "processed/stage26gr8x_role_share_summary.csv"
)

CURRENT_OUT = (
    ROOT
    / "processed/stage26gr8x_current_role_share_shadow.csv"
)

AUDIT_OUT = (
    ROOT
    / "processed/stage26gr8x_role_share_calibration_audit.json"
)

EXPECTED_CANONICAL_SHA = (
    "6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9"
)

SUPPORTED = {"RB", "FB", "WR", "TE"}

CURRENT_PLAYERS = {
    "00-0041027": "Jeremiyah Love",
    "00-0041438": "Carnell Tate",
}


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


def parse_kickoff(game_date, gametime):
    if pd.isna(game_date) or pd.isna(gametime):
        return pd.NaT

    try:
        date_part = pd.Timestamp(game_date).date()
        time_part = str(gametime).strip()

        local = pd.Timestamp(
            f"{date_part} {time_part}",
            tz=ZoneInfo("America/New_York"),
        )

        return local.tz_convert("UTC")

    except Exception:
        return pd.NaT


# ============================================================
# INPUT / PRODUCTION SAFETY
# ============================================================

required_paths = [
    STATS_PATH,
    ROSTER_PATH,
    SCHEDULE_PATH,
    TEAM_PATH,
    DB_PATH,
    R8V_PATH,
    R8V_AUDIT_PATH,
    CANONICAL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise RuntimeError(
            f"FAIL | missing required input: {path}"
        )

canonical_sha_before = sha256(CANONICAL_PATH)

if canonical_sha_before != EXPECTED_CANONICAL_SHA:
    raise RuntimeError(
        "FAIL | canonical SHA mismatch\n"
        f"EXPECTED={EXPECTED_CANONICAL_SHA}\n"
        f"ACTUAL={canonical_sha_before}"
    )


# ============================================================
# LOAD
# ============================================================

stats = pd.read_parquet(STATS_PATH)
roster = pd.read_parquet(ROSTER_PATH)
schedule = pd.read_parquet(SCHEDULE_PATH)
team = pd.read_parquet(TEAM_PATH)
r8v = pd.read_parquet(R8V_PATH)

with R8V_AUDIT_PATH.open() as f:
    r8v_audit = json.load(f)

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
# NORMALIZATION
# ============================================================

for df in [stats, roster, team]:
    if "team" in df.columns:
        df["team"] = (
            df["team"]
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
# KICKOFF AUTHORITY
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
# TRUE ROOKIE AUTHORITY
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
# ROOKIE GAME ROWS
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
# EXACT PREGAME STARTER RESOLUTION
#
# Same contract as R8S:
# exact GSIS + team + position
# latest depth snapshot strictly before kickoff
# rank 1 only
# ============================================================

resolved = []

for _, row in games.iterrows():

    pid = str(row["player_id"])
    team_name = str(row["team"]).upper()
    position = str(row["rookie_position"]).upper()
    ko = row["kickoff_utc"]

    d = depth[
        depth["gsis_id"]
        .fillna("")
        .astype(str)
        .eq(pid)
        &
        depth["team"].eq(team_name)
        &
        depth["pos_abb"].eq(position)
        &
        depth["snapshot_dt"].lt(ko)
    ].copy()

    if d.empty:
        continue

    latest_dt = d["snapshot_dt"].max()

    latest = d[
        d["snapshot_dt"].eq(latest_dt)
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

    result["pregame_depth_rank"] = num(
        best["pos_rank"]
    )

    result["pregame_depth_slot"] = num(
        best["pos_slot"]
    )

    result["pregame_depth_snapshot"] = latest_dt

    resolved.append(result)


resolved = pd.DataFrame(resolved)

if resolved.empty:
    raise RuntimeError(
        "FAIL | zero resolved rookie games"
    )

starter_games = resolved[
    pd.to_numeric(
        resolved["pregame_depth_rank"],
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
        "carries": "actual_team_carries",
        "targets": "actual_team_targets",
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
#
# Sum all QB carries for the same team/game.
# This is the actual denominator subtraction used only for
# historical calibration.
# ============================================================

qb_actual = stats[
    stats["position"].eq("QB")
].copy()

qb_actual["carries"] = pd.to_numeric(
    qb_actual["carries"],
    errors="coerce",
).fillna(0.0)

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
            "carries": "actual_qb_carries",
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

starter_games["actual_qb_carries"] = (
    pd.to_numeric(
        starter_games["actual_qb_carries"],
        errors="coerce",
    )
    .fillna(0.0)
)

starter_games["actual_team_carries"] = pd.to_numeric(
    starter_games["actual_team_carries"],
    errors="coerce",
)

starter_games["actual_team_targets"] = pd.to_numeric(
    starter_games["actual_team_targets"],
    errors="coerce",
)

starter_games["carries"] = pd.to_numeric(
    starter_games["carries"],
    errors="coerce",
).fillna(0.0)

starter_games["targets"] = pd.to_numeric(
    starter_games["targets"],
    errors="coerce",
).fillna(0.0)


# ============================================================
# ROLE-SHARE DENOMINATORS
# ============================================================

starter_games[
    "actual_non_qb_rush_pool"
] = (
    starter_games["actual_team_carries"]
    - starter_games["actual_qb_carries"]
)

starter_games[
    "non_qb_rush_share"
] = np.where(
    starter_games["actual_non_qb_rush_pool"] > 0,
    starter_games["carries"]
    / starter_games["actual_non_qb_rush_pool"],
    np.nan,
)

starter_games[
    "team_target_share"
] = np.where(
    starter_games["actual_team_targets"] > 0,
    starter_games["targets"]
    / starter_games["actual_team_targets"],
    np.nan,
)

# Fail closed on structurally impossible historical rows.
valid = (
    starter_games["actual_non_qb_rush_pool"].gt(0)
    &
    starter_games["actual_team_targets"].gt(0)
    &
    starter_games["non_qb_rush_share"].ge(0)
    &
    starter_games["team_target_share"].ge(0)
)

starter_games = starter_games[
    valid
].copy()

if starter_games.empty:
    raise RuntimeError(
        "FAIL | zero usable starter role-share rows"
    )


# ============================================================
# FIRST-START FLAG
# ============================================================

first_keys = set(
    zip(
        first_starts["game_id"].astype(str),
        first_starts["player_id"].astype(str),
    )
)

starter_games["is_first_start"] = [
    (
        str(game_id),
        str(player_id),
    ) in first_keys
    for game_id, player_id in zip(
        starter_games["game_id"],
        starter_games["player_id"],
    )
]


# ============================================================
# SUMMARY
#
# Preserve R8Q1 hierarchical convention:
# first_weight = first_n / all_n
#
# But apply it to role-share means instead of absolute
# opportunity counts.
# ============================================================

summary_rows = []

for position in sorted(SUPPORTED):

    all_pos = starter_games[
        starter_games["rookie_position"]
        .astype(str)
        .str.upper()
        .eq(position)
    ].copy()

    first_pos = all_pos[
        all_pos["is_first_start"]
    ].copy()

    if all_pos.empty:
        continue

    first_n = len(first_pos)
    all_n = len(all_pos)

    first_weight = (
        first_n / all_n
        if all_n > 0
        else 0.0
    )

    for metric in [
        "non_qb_rush_share",
        "team_target_share",
    ]:

        all_values = pd.to_numeric(
            all_pos[metric],
            errors="coerce",
        ).dropna()

        first_values = pd.to_numeric(
            first_pos[metric],
            errors="coerce",
        ).dropna()

        if all_values.empty:
            continue

        all_mean = float(
            all_values.mean()
        )

        all_median = float(
            all_values.median()
        )

        first_mean = (
            float(first_values.mean())
            if not first_values.empty
            else all_mean
        )

        first_median = (
            float(first_values.median())
            if not first_values.empty
            else all_median
        )

        hierarchical_mean = (
            first_weight * first_mean
            +
            (1.0 - first_weight) * all_mean
        )

        hierarchical_median = (
            first_weight * first_median
            +
            (1.0 - first_weight) * all_median
        )

        summary_rows.append(
            {
                "position": position,
                "metric": metric,

                "first_n": int(
                    len(first_values)
                ),

                "all_n": int(
                    len(all_values)
                ),

                "first_weight": first_weight,

                "first_mean": first_mean,
                "first_median": first_median,

                "all_mean": all_mean,
                "all_median": all_median,

                "hierarchical_mean":
                    hierarchical_mean,

                "hierarchical_median":
                    hierarchical_median,

                "p10": float(
                    all_values.quantile(0.10)
                ),

                "p25": float(
                    all_values.quantile(0.25)
                ),

                "p50": float(
                    all_values.quantile(0.50)
                ),

                "p75": float(
                    all_values.quantile(0.75)
                ),

                "p90": float(
                    all_values.quantile(0.90)
                ),
            }
        )


summary = pd.DataFrame(
    summary_rows
)

if summary.empty:
    raise RuntimeError(
        "FAIL | zero role-share summary rows"
    )


# ============================================================
# CURRENT LOVE / TATE SHADOW APPLICATION
#
# IMPORTANT:
# This does not alter R8V or canonical.
#
# Current projected denominators:
# - target pool comes from R8V team audit
# - rush pool comes from R8V team audit
# - QB carries are preserved from canonical
# ============================================================

r8v_rookies = r8v[
    r8v["shadow_row_type"]
    .astype(str)
    .eq("COLD_START_ROOKIE")
].copy()

current_rows = []

for pid, expected_name in CURRENT_PLAYERS.items():

    rookie = r8v_rookies[
        r8v_rookies["player_id"]
        .astype(str)
        .eq(pid)
    ]

    if len(rookie) != 1:
        raise RuntimeError(
            f"FAIL | expected one current rookie row for {pid}"
        )

    rookie = rookie.iloc[0]

    game_id = str(
        rookie["game_id"]
    )

    team_name = str(
        rookie["team"]
    ).upper()

    position = str(
        rookie["position"]
    ).upper()

    audit_match = [
        x
        for x in r8v_audit["team_audits"]
        if str(x["game_id"]) == game_id
        and str(x["team"]).upper() == team_name
        and str(x["rookie_player_id"]) == pid
    ]

    if len(audit_match) != 1:
        raise RuntimeError(
            "FAIL | missing exact R8V team audit "
            f"{game_id} {team_name} {pid}"
        )

    team_audit = audit_match[0]

    rush_pool = float(
        team_audit["rush_pool"]
    )

    target_pool = float(
        team_audit["target_pool"]
    )

    canonical = pd.read_parquet(
        CANONICAL_PATH
    )

    canonical_team = canonical[
        canonical["game_id"]
        .astype(str)
        .eq(game_id)
        &
        canonical["team"]
        .astype(str)
        .str.upper()
        .eq(team_name)
    ].copy()

    qb_rows = canonical_team[
        canonical_team["position"]
        .astype(str)
        .str.upper()
        .eq("QB")
    ]

    qb_reserved = float(
        pd.to_numeric(
            qb_rows["expected_carries"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    available_non_qb = (
        rush_pool - qb_reserved
    )

    if available_non_qb < 0:
        raise RuntimeError(
            "FAIL | negative projected non-QB rush pool "
            f"{game_id} {team_name}"
        )

    rush_prior = summary[
        summary["position"].eq(position)
        &
        summary["metric"].eq(
            "non_qb_rush_share"
        )
    ]

    target_prior = summary[
        summary["position"].eq(position)
        &
        summary["metric"].eq(
            "team_target_share"
        )
    ]

    if len(rush_prior) != 1:
        raise RuntimeError(
            f"FAIL | missing rush-share prior for {position}"
        )

    if len(target_prior) != 1:
        raise RuntimeError(
            f"FAIL | missing target-share prior for {position}"
        )

    rush_share = float(
        rush_prior.iloc[0][
            "hierarchical_mean"
        ]
    )

    target_share = float(
        target_prior.iloc[0][
            "hierarchical_mean"
        ]
    )

    projected_carries = (
        available_non_qb
        * rush_share
    )

    projected_targets = (
        target_pool
        * target_share
    )

    current_rows.append(
        {
            "game_id": game_id,
            "team": team_name,
            "player_id": pid,
            "player_name": expected_name,
            "position": position,

            "team_rush_pool": rush_pool,
            "qb_carries_reserved": qb_reserved,
            "available_non_qb_rush_pool":
                available_non_qb,

            "hierarchical_non_qb_rush_share":
                rush_share,

            "role_share_projected_carries":
                projected_carries,

            "r8v_absolute_prior_carries":
                float(
                    rookie["expected_carries"]
                ),

            "team_target_pool": target_pool,

            "hierarchical_team_target_share":
                target_share,

            "role_share_projected_targets":
                projected_targets,

            "r8v_absolute_prior_targets":
                float(
                    rookie["expected_targets"]
                ),

            "rush_feasible":
                projected_carries
                <= available_non_qb + 1e-9,

            "target_feasible":
                projected_targets
                <= target_pool + 1e-9,
        }
    )


current = pd.DataFrame(
    current_rows
)


# ============================================================
# WRITE RESEARCH ARTIFACTS
# ============================================================

DETAIL_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail_cols = [
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "team",
    "opponent_team",
    "player_id",
    "player_display_name",
    "rookie_position",
    "pregame_depth_rank",
    "pregame_depth_slot",
    "pregame_depth_snapshot",
    "is_first_start",
    "carries",
    "targets",
    "actual_team_carries",
    "actual_qb_carries",
    "actual_non_qb_rush_pool",
    "actual_team_targets",
    "non_qb_rush_share",
    "team_target_share",
]

detail_cols = [
    c
    for c in detail_cols
    if c in starter_games.columns
]

starter_games[
    detail_cols
].to_csv(
    DETAIL_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

current.to_csv(
    CURRENT_OUT,
    index=False,
)


# ============================================================
# SAFETY
# ============================================================

canonical_sha_after = sha256(
    CANONICAL_PATH
)

if canonical_sha_after != canonical_sha_before:
    raise RuntimeError(
        "FAIL | canonical forecast mutated"
    )

audit = {
    "stage": "STAGE26G-R8X",

    "purpose":
        "cold-start rookie starter opportunity role-share calibration",

    "starter_contract":
        "exact GSIS/team/position + latest depth snapshot strictly before kickoff + pos_rank=1",

    "rookie_contract":
        "years_exp=0 OR rookie_year=season OR entry_year=season",

    "rush_share_definition":
        "player carries / (actual team carries - actual QB carries)",

    "target_share_definition":
        "player targets / actual team targets",

    "prediction_rush_denominator":
        "projected team rush pool - canonical projected QB carries",

    "prediction_target_denominator":
        "projected team target pool",

    "hierarchical_weight":
        "first_start_n / all_rookie_starter_n",

    "starter_rows":
        int(len(starter_games)),

    "first_start_rows":
        int(starter_games["is_first_start"].sum()),

    "positions":
        sorted(
            starter_games[
                "rookie_position"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),

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
print("R8X HISTORICAL STARTER COVERAGE")
print("============================================================")

coverage = (
    starter_games
    .groupby(
        "rookie_position"
    )
    .agg(
        starter_games=(
            "player_id",
            "size",
        ),
        first_starts=(
            "is_first_start",
            "sum",
        ),
        unique_players=(
            "player_id",
            "nunique",
        ),
    )
)

print(
    coverage.to_string()
)


print()
print("============================================================")
print("R8X ROLE-SHARE SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8X FIRST-START DETAIL")
print("============================================================")

first_detail = starter_games[
    starter_games["is_first_start"]
].copy()

show = [
    "game_id",
    "team",
    "player_display_name",
    "rookie_position",
    "carries",
    "actual_non_qb_rush_pool",
    "non_qb_rush_share",
    "targets",
    "actual_team_targets",
    "team_target_share",
]

show = [
    c
    for c in show
    if c in first_detail.columns
]

print(
    first_detail[
        show
    ]
    .sort_values(
        [
            "rookie_position",
            "game_id",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8X CURRENT LOVE / TATE ROLE-SHARE SHADOW")
print("============================================================")

print(
    current.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8X FEASIBILITY")
print("============================================================")

print(
    current[
        [
            "player_name",
            "position",
            "available_non_qb_rush_pool",
            "role_share_projected_carries",
            "rush_feasible",
            "team_target_pool",
            "role_share_projected_targets",
            "target_feasible",
        ]
    ].to_string(
        index=False
    )
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

print(f"DETAIL={DETAIL_OUT}")
print(f"SUMMARY={SUMMARY_OUT}")
print(f"CURRENT={CURRENT_OUT}")
print(f"AUDIT={AUDIT_OUT}")


print()
print("============================================================")
print("STAGE26G_R8X_ROLE_SHARE_CALIBRATION=PASS")
print("============================================================")
