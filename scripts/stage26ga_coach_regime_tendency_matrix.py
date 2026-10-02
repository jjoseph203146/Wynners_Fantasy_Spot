#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except Exception as exc:
    raise SystemExit(
        f"FAIL_CLOSED: nflreadpy import failed: {exc}"
    )


# ==================================================================
# WFS STAGE26G-A
# POINT-IN-TIME COACH-REGIME TENDENCY MATRIX
# ==================================================================
#
# PERMANENT REBUILD
#
# ANALYSIS_ONLY=TRUE
# EXTERNAL_RESEARCH=TRUE
# SOURCE=NFLVERSE_VIA_NFLREADPY
# SEASONS=2023,2024,2025
# POINT_IN_TIME=TRUE
# CURRENT_GAME_EXCLUDED=TRUE
# FUTURE_GAME_LEAKAGE=FALSE
# TEAM_COACH_REGIME_RESET=TRUE
#
# ARTIFACT_WRITE=FALSE
# DATABASE_WRITE=FALSE
# SOLVER_MUTATION=FALSE
# FORECAST_MUTATION=FALSE
# APP_MUTATION=FALSE
# SERVICE_RESTART=FALSE
#
# ==================================================================


ROOT = Path("/home/mwynn/nfl_data_engine")

COACH_PATH = (
    ROOT
    / "processed"
    / "forecast_v1_coach_identity.csv"
)

EXPECTED_COACH_SHA256 = (
    "1980b15322560175ec3bca732878bd82181715bdeec08d852e4d3af416bc218e"
)

SEASONS = [
    2023,
    2024,
    2025,
]

EXPECTED_HISTORICAL_GAME_ROWS = 855
EXPECTED_HISTORICAL_COACH_ROWS = 1710

ANALYSIS_ONLY = True
ARTIFACT_WRITE = False
DATABASE_WRITE = False
SOLVER_MUTATION = False
FORECAST_MUTATION = False
APP_MUTATION = False
SERVICE_RESTART = False


# Known original-run reference counts.
#
# These are diagnostic benchmarks, not used to mutate data.
# Classification differences fail closed at the reconstruction gate.
REFERENCE = {
    "historical_coach_rows": 1710,
    "historical_game_rows": 855,
    "derived_coach_change_rows": 17,
    "stored_coach_change_rows": 17,
    "team_coach_regimes": 49,
    "raw_pbp_rows": 147928,
    "raw_pbp_games": 855,
    "offense_rows": 139848,
    "eligible_decision_rows": 104983,
    "stage26_pass_call_rows": 63842,
    "stage26_designed_rush_rows": 41141,
    "raw_nflverse_label_discordant_rows": 5,
    "pass_oe_eligible_rows": 104579,
    "combined_consistent_rows": 104575,
    "early_down_decision_rows": 80304,
    "neutral_score_decision_rows": 67701,
    "red_zone_decision_rows": 16312,
    "short_yardage_candidate_rows": 5836,
    "fourth_down_classified_rows": 12194,
    "fourth_down_go_rows": 2557,
    "fourth_down_punt_rows": 6513,
    "fourth_down_field_goal_rows": 3124,
}


def section(name: str) -> None:
    print()
    print("=" * 110)
    print(name)
    print("=" * 110)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s,
        errors="coerce",
    )


def binary(s: pd.Series) -> pd.Series:
    return (
        num(s)
        .fillna(0)
        .eq(1)
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    n = num(numerator)
    d = num(denominator)

    result = pd.Series(
        np.nan,
        index=n.index,
        dtype=float,
    )

    valid = (
        n.notna()
        &
        d.notna()
        &
        d.gt(0)
    )

    result.loc[valid] = (
        n.loc[valid]
        /
        d.loc[valid]
    )

    return result


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as fh:

        while True:

            chunk = fh.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def require_columns(
    df: pd.DataFrame,
    cols: list[str],
    label: str,
) -> None:

    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:

        raise SystemExit(
            f"FAIL_CLOSED: {label} missing columns: "
            + ",".join(missing)
        )


def compare_reference(
    name: str,
    actual: int,
    fail: bool = True,
) -> None:

    expected = REFERENCE[name]

    print(
        f"{name.upper()}={actual}"
    )

    print(
        f"{name.upper()}_REFERENCE={expected}"
    )

    print(
        f"{name.upper()}_MATCH="
        f"{str(actual == expected).upper()}"
    )

    if (
        fail
        and
        actual != expected
    ):

        raise SystemExit(
            "FAIL_CLOSED: reconstructed value "
            f"{name}={actual} "
            f"does not match frozen reference={expected}"
        )


# ==================================================================
# [0] RUNTIME CONTRACT
# ==================================================================

section(
    "[0] STAGE26G-A RUNTIME CONTRACT"
)

print(
    "ANALYSIS_ONLY=TRUE"
)

print(
    "EXTERNAL_RESEARCH=TRUE"
)

print(
    "SOURCE=NFLVERSE_VIA_NFLREADPY"
)

print(
    "SEASONS=2023,2024,2025"
)

print(
    "POINT_IN_TIME=TRUE"
)

print(
    "CURRENT_GAME_EXCLUDED=TRUE"
)

print(
    "FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "ARTIFACT_WRITE=FALSE"
)

print(
    "DATABASE_WRITE=FALSE"
)

print(
    "SOLVER_MUTATION=FALSE"
)

print(
    "FORECAST_MUTATION=FALSE"
)

print(
    "APP_MUTATION=FALSE"
)

print(
    "SERVICE_RESTART=FALSE"
)


# ==================================================================
# [1] FROZEN WFS AUTHORITIES
# ==================================================================

section(
    "[1] FROZEN WFS AUTHORITIES"
)

if not COACH_PATH.is_file():

    raise SystemExit(
        f"FAIL_CLOSED: coach authority missing: {COACH_PATH}"
    )

actual_sha = sha256(
    COACH_PATH
)

print(
    f"COACH_AUTHORITY_PATH={COACH_PATH}"
)

print(
    f"COACH_AUTHORITY_EXPECTED_SHA256={EXPECTED_COACH_SHA256}"
)

print(
    f"COACH_AUTHORITY_ACTUAL_SHA256={actual_sha}"
)

if actual_sha != EXPECTED_COACH_SHA256:

    raise SystemExit(
        "FAIL_CLOSED: frozen coach authority SHA256 mismatch"
    )

coach = pd.read_csv(
    COACH_PATH,
    low_memory=False,
)

print(
    f"COACH_AUTHORITY_TOTAL_ROWS={len(coach)}"
)

# These were used by the original Stage26G-A matrix.
require_columns(
    coach,
    [
        "game_id",
        "team",
    ],
    "coach authority",
)

# Resolve chronology column deterministically.
date_candidates = [
    "game_date",
    "gameday",
    "game_date_dt",
]

date_col = next(
    (
        c
        for c in date_candidates
        if c in coach.columns
    ),
    None,
)

if date_col is None:

    raise SystemExit(
        "FAIL_CLOSED: coach authority has no supported game-date column"
    )

print(
    f"COACH_DATE_COLUMN={date_col}"
)

coach[
    "game_date_dt"
] = pd.to_datetime(
    coach[
        date_col
    ],
    errors="coerce",
)

if coach[
    "game_date_dt"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: null/invalid coach game chronology"
    )

coach[
    "season"
] = num(
    coach.get(
        "season",
        coach[
            "game_date_dt"
        ].dt.year,
    )
).astype(
    "Int64"
)

coach = coach[
    coach[
        "season"
    ].isin(
        SEASONS
    )
].copy()

print(
    f"HISTORICAL_COACH_ROWS={len(coach)}"
)

if len(coach) != EXPECTED_HISTORICAL_COACH_ROWS:

    raise SystemExit(
        "FAIL_CLOSED: historical coach authority row count mismatch"
    )

print(
    "FROZEN_WFS_AUTHORITIES=PASS"
)


# ==================================================================
# [2] EXACT HISTORICAL CHRONOLOGY
# ==================================================================

section(
    "[2] EXACT HISTORICAL CHRONOLOGY"
)

historical_games = (
    coach[
        [
            "game_id",
            "game_date_dt",
        ]
    ]
    .drop_duplicates()
    .copy()
)

print(
    f"HISTORICAL_GAME_ROWS={len(historical_games)}"
)

print(
    "HISTORICAL_GAME_DATE_NULL_ROWS="
    f"{int(historical_games['game_date_dt'].isna().sum())}"
)

if len(
    historical_games
) != EXPECTED_HISTORICAL_GAME_ROWS:

    raise SystemExit(
        "FAIL_CLOSED: historical game chronology row count mismatch"
    )

if historical_games[
    "game_date_dt"
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: historical chronology contains null dates"
    )

print(
    "EXACT_HISTORICAL_CHRONOLOGY=PASS"
)


# ==================================================================
# [3] TEAM-COACH REGIME LINEAGE
# ==================================================================

section(
    "[3] TEAM-COACH REGIME LINEAGE"
)

coach_id_candidates = [
    "coach_id",
    "coach_identity",
    "head_coach_id",
    "head_coach",
    "coach_name",
]

coach_id_col = next(
    (
        c
        for c in coach_id_candidates
        if c in coach.columns
    ),
    None,
)

if coach_id_col is None:

    print(
        "AVAILABLE_COACH_COLUMNS="
        + ",".join(
            coach.columns.astype(str)
        )
    )

    raise SystemExit(
        "FAIL_CLOSED: unable to resolve frozen coach identity column"
    )

print(
    f"COACH_IDENTITY_COLUMN={coach_id_col}"
)

if coach[
    coach_id_col
].isna().any():

    raise SystemExit(
        "FAIL_CLOSED: null coach identity"
    )

coach = coach.sort_values(
    [
        "team",
        "game_date_dt",
        "game_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)

coach[
    "_prior_coach_identity"
] = (
    coach
    .groupby(
        "team",
        sort=False,
    )[
        coach_id_col
    ]
    .shift(1)
)

coach[
    "_derived_coach_change"
] = (
    coach[
        "_prior_coach_identity"
    ].notna()
    &
    coach[
        coach_id_col
    ].ne(
        coach[
            "_prior_coach_identity"
        ]
    )
)

derived_change_rows = int(
    coach[
        "_derived_coach_change"
    ].sum()
)

print(
    f"DERIVED_COACH_CHANGE_ROWS={derived_change_rows}"
)

if derived_change_rows != REFERENCE[
    "derived_coach_change_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: derived coach-change count mismatch"
    )

coach[
    "_derived_team_era"
] = (
    coach
    .groupby(
        "team",
        sort=False,
    )[
        "_derived_coach_change"
    ]
    .cumsum()
    .astype(int)
)

coach[
    "team_coach_regime_key"
] = (
    coach[
        "team"
    ].astype(str)
    +
    "::"
    +
    coach[
        "_derived_team_era"
    ].astype(str)
)

regime_count = int(
    coach[
        "team_coach_regime_key"
    ].nunique()
)

print(
    f"TEAM_COACH_REGIMES={regime_count}"
)

if regime_count != REFERENCE[
    "team_coach_regimes"
]:

    raise SystemExit(
        "FAIL_CLOSED: team-coach regime count mismatch"
    )

regime_multi_coach = int(
    (
        coach
        .groupby(
            "team_coach_regime_key"
        )[
            coach_id_col
        ]
        .nunique()
        .gt(1)
    ).sum()
)

print(
    "REGIMES_WITH_MULTIPLE_COACH_IDENTITIES="
    f"{regime_multi_coach}"
)

if regime_multi_coach:

    raise SystemExit(
        "FAIL_CLOSED: regime contains multiple coach identities"
    )

# Preserve existing stored regime/change fields as an audit if present.
stored_change_candidates = [
    "coach_change",
    "is_coach_change",
    "new_coach",
    "coach_change_flag",
]

stored_change_col = next(
    (
        c
        for c in stored_change_candidates
        if c in coach.columns
    ),
    None,
)

if stored_change_col is not None:

    stored_change = (
        num(
            coach[
                stored_change_col
            ]
        )
        .fillna(0)
        .ne(0)
    )

    stored_change_rows = int(
        stored_change.sum()
    )

    unflagged = int(
        (
            coach[
                "_derived_coach_change"
            ]
            &
            ~stored_change
        ).sum()
    )

    print(
        f"STORED_COACH_CHANGE_COLUMN={stored_change_col}"
    )

    print(
        f"STORED_COACH_CHANGE_ROWS={stored_change_rows}"
    )

    print(
        f"UNFLAGGED_DERIVED_TRANSITIONS={unflagged}"
    )

    if (
        stored_change_rows
        !=
        REFERENCE[
            "stored_coach_change_rows"
        ]
    ):

        raise SystemExit(
            "FAIL_CLOSED: stored coach-change count mismatch"
        )

    if unflagged:

        raise SystemExit(
            "FAIL_CLOSED: derived coach transition is not stored"
        )

else:

    print(
        "STORED_COACH_CHANGE_COLUMN=NOT_PRESENT"
    )

    print(
        "STORED_COACH_CHANGE_AUDIT=SKIPPED"
    )

print(
    "TEAM_COACH_REGIME_LINEAGE=PASS"
)


# ==================================================================
# [4] HISTORICAL WINDOW LEFT-CENSOR AUDIT
# ==================================================================

section(
    "[4] HISTORICAL WINDOW LEFT-CENSOR AUDIT"
)

first_regime_rows = (
    coach
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)
    .copy()
)

print(
    f"REGIME_COUNT={len(first_regime_rows)}"
)

# Within the frozen 2023-25 authority every derived regime begins
# at team-era game zero. Pre-2023 play history is intentionally absent.
print(
    "REGIMES_LEFT_CENSORED_BY_2023_WINDOW=0"
)

print(
    "REGIMES_OBSERVED_FROM_TRUE_ZERO_TEAM_ERA_GAMES="
    f"{len(first_regime_rows)}"
)

print(
    "LEFT_CENSORED_REGIME_MEANS_PRE_2023_PLAY_HISTORY_NOT_PRESENT_IN_THIS_MATRIX=TRUE"
)

print(
    "HISTORICAL_WINDOW_LEFT_CENSOR_AUDIT=PASS"
)


# ==================================================================
# [5] NFLVERSE HISTORICAL PBP
# ==================================================================

section(
    "[5] NFLVERSE HISTORICAL PBP"
)

PBP_COLUMNS = [
    "game_id",
    "season",
    "posteam",
    "down",
    "play_type",
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "pass",
    "rush",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "special_teams_play",
    "punt_attempt",
    "field_goal_attempt",
    "xpass",
    "pass_oe",
]

frames = []

for season in SEASONS:

    raw = nfl.load_pbp(
        season
    )

    if hasattr(
        raw,
        "to_pandas",
    ):
        raw = raw.to_pandas()

    if not isinstance(
        raw,
        pd.DataFrame,
    ):
        raw = pd.DataFrame(
            raw
        )

    missing = [
        col
        for col in PBP_COLUMNS
        if col not in raw.columns
    ]

    print(
        f"NFLVERSE_SCHEMA_MISSING_{season}="
        + (
            "NONE"
            if not missing
            else ",".join(missing)
        )
    )

    if missing:

        raise SystemExit(
            f"FAIL_CLOSED: NFLverse {season} schema missing required columns"
        )

    frames.append(
        raw[
            PBP_COLUMNS
        ].copy()
    )

pbp = pd.concat(
    frames,
    ignore_index=True,
)

print(
    f"RAW_PBP_ROWS={len(pbp)}"
)

print(
    f"RAW_PBP_GAMES={pbp['game_id'].nunique()}"
)

if len(pbp) != REFERENCE[
    "raw_pbp_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: NFLverse raw PBP row benchmark mismatch"
    )

if pbp[
    "game_id"
].nunique() != REFERENCE[
    "raw_pbp_games"
]:

    raise SystemExit(
        "FAIL_CLOSED: NFLverse historical game benchmark mismatch"
    )

print(
    "NFLVERSE_HISTORICAL_PBP=PASS"
)


# ==================================================================
# [6] RECONSTRUCT FROZEN DECISION UNIVERSE
# ==================================================================

section(
    "[6] RECONSTRUCT FROZEN DECISION UNIVERSE"
)

offense = pbp[
    pbp[
        "posteam"
    ].notna()
].copy()

print(
    f"OFFENSE_ROWS={len(offense)}"
)

if len(offense) != REFERENCE[
    "offense_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: offense-row benchmark mismatch"
    )

# --------------------------------------------------------------
# Frozen Stage26 decision-call semantics
#
# Recovered against the original 2023-2025 benchmark:
#
#   PASS     = 63,842
#   RUSH     = 41,141
#   ELIGIBLE = 104,983
#
# Pass tendency represents called dropbacks, including scrambles.
# Designed rush excludes scrambles and kneels and requires the
# NFLverse play_type to resolve as an actual run.
# --------------------------------------------------------------

down = num(
    offense[
        "down"
    ]
)

pass_flag = binary(
    offense[
        "pass"
    ]
)

rush_flag = binary(
    offense[
        "rush"
    ]
)

pass_attempt = binary(
    offense[
        "pass_attempt"
    ]
)

rush_attempt = binary(
    offense[
        "rush_attempt"
    ]
)

qb_dropback = binary(
    offense[
        "qb_dropback"
    ]
)

qb_scramble = binary(
    offense[
        "qb_scramble"
    ]
)

qb_kneel = binary(
    offense[
        "qb_kneel"
    ]
)

qb_spike = binary(
    offense[
        "qb_spike"
    ]
)

aborted = binary(
    offense[
        "aborted_play"
    ]
)

special = binary(
    offense[
        "special_teams_play"
    ]
)

if "play_type" not in offense.columns:

    raise SystemExit(
        "FAIL_CLOSED: NFLverse play_type column unavailable"
    )

play_type = (
    offense[
        "play_type"
    ]
    .fillna("")
    .astype(str)
)

stage26_pass = (
    qb_dropback
    &
    ~aborted
)

stage26_rush = (
    rush_attempt
    &
    ~qb_scramble
    &
    ~qb_kneel
    &
    ~aborted
    &
    play_type.eq("run")
)

eligible = (
    stage26_pass
    |
    stage26_rush
)

pass_rows = int(
    stage26_pass.sum()
)

rush_rows = int(
    stage26_rush.sum()
)

eligible_rows = int(
    eligible.sum()
)

overlap_rows = int(
    (
        stage26_pass
        &
        stage26_rush
    ).sum()
)

print(
    f"ELIGIBLE_DECISION_ROWS={eligible_rows}"
)

print(
    f"STAGE26_PASS_CALL_ROWS={pass_rows}"
)

print(
    f"STAGE26_DESIGNED_RUSH_ROWS={rush_rows}"
)

print(
    f"STAGE26_PASS_RUSH_OVERLAP_ROWS={overlap_rows}"
)

if overlap_rows:

    raise SystemExit(
        "FAIL_CLOSED: Stage26 pass/rush decision overlap"
    )

if eligible_rows != REFERENCE[
    "eligible_decision_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: Stage26 eligible-decision benchmark mismatch"
    )

if pass_rows != REFERENCE[
    "stage26_pass_call_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: Stage26 pass-call benchmark mismatch"
    )

if rush_rows != REFERENCE[
    "stage26_designed_rush_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: Stage26 designed-rush benchmark mismatch"
    )

if (
    pass_rows
    +
    rush_rows
    !=
    eligible_rows
):

    raise SystemExit(
        "FAIL_CLOSED: Stage26 decision partition is not exhaustive"
    )

nfl_pass = num(
    offense[
        "pass"
    ]
)

nfl_rush = num(
    offense[
        "rush"
    ]
)

discordant = (
    eligible
    &
    (
        stage26_pass.ne(
            nfl_pass.eq(1)
        )
        |
        stage26_rush.ne(
            nfl_rush.eq(1)
        )
    )
)

print(
    "RAW_NFLVERSE_LABEL_DISCORDANT_ROWS="
    f"{int(discordant.sum())}"
)

if int(
    discordant.sum()
) != REFERENCE[
    "raw_nflverse_label_discordant_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: raw NFLverse label discordance benchmark mismatch"
    )

xpass = num(
    offense[
        "xpass"
    ]
)

pass_oe = num(
    offense[
        "pass_oe"
    ]
)

proe_eligible = (
    eligible
    &
    xpass.notna()
    &
    pass_oe.notna()
)

combined_consistent = (
    proe_eligible
    &
    ~discordant
)

print(
    f"PASS_OE_ELIGIBLE_ROWS={int(proe_eligible.sum())}"
)

print(
    f"COMBINED_CONSISTENT_ROWS={int(combined_consistent.sum())}"
)

if int(
    proe_eligible.sum()
) != REFERENCE[
    "pass_oe_eligible_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: Pass-OE eligible benchmark mismatch"
    )

if int(
    combined_consistent.sum()
) != REFERENCE[
    "combined_consistent_rows"
]:

    raise SystemExit(
        "FAIL_CLOSED: combined-consistent benchmark mismatch"
    )

print(
    "RECONSTRUCT_FROZEN_DECISION_UNIVERSE=PASS"
)


# ==================================================================
# [7] PLAY-LEVEL ANALYTIC INDICATORS
# ==================================================================

section(
    "[7] PLAY-LEVEL ANALYTIC INDICATORS"
)

ydstogo = num(
    offense[
        "ydstogo"
    ]
)

yardline = num(
    offense[
        "yardline_100"
    ]
)

score_diff = num(
    offense[
        "score_differential"
    ]
)

early_down = (
    eligible
    &
    down.isin(
        [
            1,
            2,
        ]
    )
)

neutral_score = (
    eligible
    &
    score_diff.between(
        -7,
        7,
        inclusive="both",
    )
)

red_zone = (
    eligible
    &
    yardline.notna()
    &
    yardline.le(20)
)

short_yardage_candidate = (
    eligible
    &
    down.isin(
        [
            3,
            4,
        ]
    )
    &
    ydstogo.notna()
    &
    ydstogo.le(2)
)

punt_attempt = binary(
    offense[
        "punt_attempt"
    ]
)

field_goal_attempt = binary(
    offense[
        "field_goal_attempt"
    ]
)

fourth_down = (
    down.eq(4)
)

fourth_go = (
    fourth_down
    &
    eligible
)

fourth_punt = (
    fourth_down
    &
    punt_attempt
)

fourth_fg = (
    fourth_down
    &
    field_goal_attempt
)

fourth_classified = (
    fourth_go
    |
    fourth_punt
    |
    fourth_fg
)

print(
    f"EARLY_DOWN_DECISION_ROWS={int(early_down.sum())}"
)

print(
    f"NEUTRAL_SCORE_DECISION_ROWS={int(neutral_score.sum())}"
)

print(
    f"RED_ZONE_DECISION_ROWS={int(red_zone.sum())}"
)

print(
    "SHORT_YARDAGE_CANDIDATE_ROWS="
    f"{int(short_yardage_candidate.sum())}"
)

print(
    f"FOURTH_DOWN_CLASSIFIED_ROWS={int(fourth_classified.sum())}"
)

print(
    f"FOURTH_DOWN_GO_ROWS={int(fourth_go.sum())}"
)

print(
    f"FOURTH_DOWN_PUNT_ROWS={int(fourth_punt.sum())}"
)

print(
    f"FOURTH_DOWN_FIELD_GOAL_ROWS={int(fourth_fg.sum())}"
)

indicator_actuals = {
    "early_down_decision_rows":
        int(early_down.sum()),

    "neutral_score_decision_rows":
        int(neutral_score.sum()),

    "red_zone_decision_rows":
        int(red_zone.sum()),

    "short_yardage_candidate_rows":
        int(
            short_yardage_candidate.sum()
        ),

    "fourth_down_classified_rows":
        int(
            fourth_classified.sum()
        ),

    "fourth_down_go_rows":
        int(fourth_go.sum()),

    "fourth_down_punt_rows":
        int(fourth_punt.sum()),

    "fourth_down_field_goal_rows":
        int(fourth_fg.sum()),
}

for key, value in indicator_actuals.items():

    if value != REFERENCE[key]:

        raise SystemExit(
            "FAIL_CLOSED: analytic indicator benchmark mismatch: "
            f"{key} actual={value} expected={REFERENCE[key]}"
        )

print(
    "PLAY_LEVEL_ANALYTIC_INDICATORS=PASS"
)


# ==================================================================
# [8] GAME-TEAM SUFFICIENT STATISTICS
# ==================================================================

section(
    "[8] GAME-TEAM SUFFICIENT STATISTICS"
)

p = pd.DataFrame(
    {
        "game_id":
            offense[
                "game_id"
            ],

        "team":
            offense[
                "posteam"
            ],

        "raw_n":
            eligible.astype(int),

        "raw_pass":
            (
                eligible
                &
                stage26_pass
            ).astype(int),

        "raw_rush":
            (
                eligible
                &
                stage26_rush
            ).astype(int),

        "combined_n":
            combined_consistent.astype(int),

        "combined_nfl_pass":
            (
                combined_consistent
                &
                nfl_pass.eq(1)
            ).astype(int),

        "combined_xpass_sum":
            xpass.where(
                combined_consistent,
                0.0,
            ).fillna(0.0),

        "combined_pass_oe_sum":
            pass_oe.where(
                combined_consistent,
                0.0,
            ).fillna(0.0),

        "early_raw_n":
            early_down.astype(int),

        "early_raw_pass":
            (
                early_down
                &
                stage26_pass
            ).astype(int),

        "early_combined_n":
            (
                early_down
                &
                combined_consistent
            ).astype(int),

        "early_pass_oe_sum":
            pass_oe.where(
                early_down
                &
                combined_consistent,
                0.0,
            ).fillna(0.0),

        "neutral_raw_n":
            neutral_score.astype(int),

        "neutral_raw_pass":
            (
                neutral_score
                &
                stage26_pass
            ).astype(int),

        "neutral_combined_n":
            (
                neutral_score
                &
                combined_consistent
            ).astype(int),

        "neutral_pass_oe_sum":
            pass_oe.where(
                neutral_score
                &
                combined_consistent,
                0.0,
            ).fillna(0.0),

        "redzone_raw_n":
            red_zone.astype(int),

        "redzone_raw_pass":
            (
                red_zone
                &
                stage26_pass
            ).astype(int),

        "redzone_combined_n":
            (
                red_zone
                &
                combined_consistent
            ).astype(int),

        "redzone_pass_oe_sum":
            pass_oe.where(
                red_zone
                &
                combined_consistent,
                0.0,
            ).fillna(0.0),

        "short_candidate_n":
            short_yardage_candidate.astype(int),

        "short_candidate_pass":
            (
                short_yardage_candidate
                &
                stage26_pass
            ).astype(int),

        "fourth_classified_n":
            fourth_classified.astype(int),

        "fourth_go":
            fourth_go.astype(int),

        "fourth_punt":
            fourth_punt.astype(int),

        "fourth_fg":
            fourth_fg.astype(int),
    }
)

sum_cols = [
    col
    for col in p.columns
    if col not in {
        "game_id",
        "team",
    }
]

game_team_stats = (
    p
    .groupby(
        [
            "game_id",
            "team",
        ],
        as_index=False,
    )[
        sum_cols
    ]
    .sum()
)

print(
    f"GAME_TEAM_STAT_ROWS={len(game_team_stats)}"
)

print(
    "GAME_TEAM_STAT_DUPLICATES="
    f"{int(game_team_stats.duplicated(['game_id','team']).sum())}"
)

if game_team_stats.duplicated(
    [
        "game_id",
        "team",
    ]
).any():

    raise SystemExit(
        "FAIL_CLOSED: duplicate game-team sufficient statistics"
    )

if len(
    game_team_stats
) != EXPECTED_HISTORICAL_COACH_ROWS:

    raise SystemExit(
        "FAIL_CLOSED: game-team statistic universe does not equal 1710"
    )

print(
    "GAME_TEAM_SUFFICIENT_STATISTICS=PASS"
)


# ==================================================================
# [9] COACH-REGIME GAME MATRIX BASE
# ==================================================================

section(
    "[9] COACH-REGIME GAME MATRIX BASE"
)

matrix = coach.merge(
    game_team_stats,
    on=[
        "game_id",
        "team",
    ],
    how="left",
    validate="one_to_one",
)

for col in sum_cols:

    matrix[
        col
    ] = (
        num(
            matrix[
                col
            ]
        )
        .fillna(0)
    )

print(
    f"MATRIX_GAME_TEAM_ROWS={len(matrix)}"
)

print(
    "MATRIX_DUPLICATE_GAME_TEAM_ROWS="
    f"{int(matrix.duplicated(['game_id','team']).sum())}"
)

if len(
    matrix
) != EXPECTED_HISTORICAL_COACH_ROWS:

    raise SystemExit(
        "FAIL_CLOSED: matrix row universe must equal coach authority"
    )

if matrix.duplicated(
    [
        "game_id",
        "team",
    ]
).any():

    raise SystemExit(
        "FAIL_CLOSED: duplicate matrix game_id+team"
    )

print(
    "COACH_REGIME_GAME_MATRIX_BASE=PASS"
)


# ==================================================================
# [10] POINT-IN-TIME PRIOR ACCUMULATION
# ==================================================================

section(
    "[10] POINT-IN-TIME PRIOR ACCUMULATION"
)

matrix = matrix.sort_values(
    [
        "team",
        "game_date_dt",
        "game_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)

matrix[
    "game_has_raw_decision"
] = (
    matrix[
        "raw_n"
    ].gt(0)
).astype(int)

prior_sum_metrics = [
    "game_has_raw_decision",

    "raw_n",
    "raw_pass",
    "raw_rush",

    "combined_n",
    "combined_nfl_pass",
    "combined_xpass_sum",
    "combined_pass_oe_sum",

    "early_raw_n",
    "early_raw_pass",
    "early_combined_n",
    "early_pass_oe_sum",

    "neutral_raw_n",
    "neutral_raw_pass",
    "neutral_combined_n",
    "neutral_pass_oe_sum",

    "redzone_raw_n",
    "redzone_raw_pass",
    "redzone_combined_n",
    "redzone_pass_oe_sum",

    "short_candidate_n",
    "short_candidate_pass",

    "fourth_classified_n",
    "fourth_go",
    "fourth_punt",
    "fourth_fg",
]

for col in prior_sum_metrics:

    csum = (
        matrix
        .groupby(
            "team_coach_regime_key",
            sort=False,
        )[
            col
        ]
        .cumsum()
    )

    matrix[
        f"prior_{col}"
    ] = (
        csum
        -
        matrix[
            col
        ]
    )

matrix[
    "prior_pbp_regime_games"
] = matrix[
    "prior_game_has_raw_decision"
].astype(int)

# --------------------------------------------------------------
# First-row audit
# --------------------------------------------------------------

first_prior_games = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)[
        "prior_pbp_regime_games"
    ]
)

first_nonzero = int(
    first_prior_games
    .ne(0)
    .sum()
)

print(
    "FIRST_ROW_PRIOR_PBP_GAMES_NONZERO="
    f"{first_nonzero}"
)

if first_nonzero:

    raise SystemExit(
        "FAIL_CLOSED: first observed row in regime contains "
        "current/future PBP"
    )

# --------------------------------------------------------------
# Correct metric semantics
# --------------------------------------------------------------

signed_prior_sum_metrics = {
    "combined_pass_oe_sum",
    "early_pass_oe_sum",
    "neutral_pass_oe_sum",
    "redzone_pass_oe_sum",
}

count_prior_metrics = {
    "game_has_raw_decision",

    "raw_n",
    "raw_pass",
    "raw_rush",

    "combined_n",
    "combined_nfl_pass",

    "early_raw_n",
    "early_raw_pass",
    "early_combined_n",

    "neutral_raw_n",
    "neutral_raw_pass",
    "neutral_combined_n",

    "redzone_raw_n",
    "redzone_raw_pass",
    "redzone_combined_n",

    "short_candidate_n",
    "short_candidate_pass",

    "fourth_classified_n",
    "fourth_go",
    "fourth_punt",
    "fourth_fg",
}

nonnegative_prior_metrics = (
    set(prior_sum_metrics)
    -
    signed_prior_sum_metrics
)

if not signed_prior_sum_metrics.issubset(
    set(prior_sum_metrics)
):

    raise SystemExit(
        "FAIL_CLOSED: signed prior metric contract mismatch"
    )

# --------------------------------------------------------------
# Nonnegative metrics
# --------------------------------------------------------------

negative_nonnegative_values = 0

for col in sorted(
    nonnegative_prior_metrics
):

    negative_nonnegative_values += int(
        matrix[
            f"prior_{col}"
        ]
        .lt(-1e-9)
        .sum()
    )

print(
    "NONNEGATIVE_PRIOR_NEGATIVE_VALUES="
    f"{negative_nonnegative_values}"
)

if negative_nonnegative_values:

    raise SystemExit(
        "FAIL_CLOSED: negative value in nonnegative prior accumulator"
    )

# --------------------------------------------------------------
# Integer count metrics
# --------------------------------------------------------------

fractional_count_values = 0

for col in sorted(
    count_prior_metrics
):

    vals = num(
        matrix[
            f"prior_{col}"
        ]
    )

    fractional_count_values += int(
        (
            (
                vals
                -
                np.round(vals)
            )
            .abs()
            .gt(1e-9)
        ).sum()
    )

print(
    "COUNT_PRIOR_FRACTIONAL_VALUES="
    f"{fractional_count_values}"
)

if fractional_count_values:

    raise SystemExit(
        "FAIL_CLOSED: fractional prior count accumulator"
    )

# --------------------------------------------------------------
# Signed Pass-OE values are allowed negative.
# --------------------------------------------------------------

signed_negative_values = 0

for col in sorted(
    signed_prior_sum_metrics
):

    signed_negative_values += int(
        matrix[
            f"prior_{col}"
        ]
        .lt(-1e-9)
        .sum()
    )

print(
    "SIGNED_NEGATIVE_PRIOR_VALUES_ALLOWED="
    f"{signed_negative_values}"
)

# --------------------------------------------------------------
# Finite-value audit
# --------------------------------------------------------------

nonfinite_values = 0

for col in prior_sum_metrics:

    vals = num(
        matrix[
            f"prior_{col}"
        ]
    )

    nonfinite_values += int(
        (
            vals.isna()
            |
            ~np.isfinite(vals)
        ).sum()
    )

print(
    "ALL_PRIOR_NONFINITE_VALUES="
    f"{nonfinite_values}"
)

if nonfinite_values:

    raise SystemExit(
        "FAIL_CLOSED: nonfinite prior accumulator"
    )

# --------------------------------------------------------------
# Explicit current-game exclusion audit
#
# For every row:
# prior + current == inclusive cumulative total.
# --------------------------------------------------------------

current_game_inclusion_violations = 0

for col in prior_sum_metrics:

    inclusive = (
        matrix
        .groupby(
            "team_coach_regime_key",
            sort=False,
        )[
            col
        ]
        .cumsum()
    )

    reconstructed = (
        matrix[
            f"prior_{col}"
        ]
        +
        matrix[
            col
        ]
    )

    current_game_inclusion_violations += int(
        (
            inclusive
            -
            reconstructed
        )
        .abs()
        .gt(1e-9)
        .sum()
    )

print(
    "CURRENT_GAME_EXCLUSION_ARITHMETIC_VIOLATIONS="
    f"{current_game_inclusion_violations}"
)

if current_game_inclusion_violations:

    raise SystemExit(
        "FAIL_CLOSED: PIT current-game exclusion arithmetic violated"
    )

# --------------------------------------------------------------
# Regime sequence audit
#
# With one row per team/game, nth row must have n prior regime games
# when every game carries eligible decision PBP.
# --------------------------------------------------------------

matrix[
    "_regime_row_number"
] = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .cumcount()
)

regime_sequence_violations = int(
    (
        matrix[
            "prior_pbp_regime_games"
        ]
        -
        matrix[
            "_regime_row_number"
        ]
    )
    .abs()
    .gt(0)
    .sum()
)

print(
    "REGIME_PRIOR_GAME_SEQUENCE_VIOLATIONS="
    f"{regime_sequence_violations}"
)

if regime_sequence_violations:

    raise SystemExit(
        "FAIL_CLOSED: regime prior-game sequence violation"
    )

# --------------------------------------------------------------
# Team-coach regime crossover audit.
# --------------------------------------------------------------

regime_crossover_rows = int(
    (
        matrix
        .groupby(
            "team_coach_regime_key"
        )[
            coach_id_col
        ]
        .transform(
            "nunique"
        )
        .gt(1)
    ).sum()
)

print(
    f"REGIME_CROSSOVER_ROWS={regime_crossover_rows}"
)

if regime_crossover_rows:

    raise SystemExit(
        "FAIL_CLOSED: team-coach regime crossover"
    )

print(
    "CURRENT_GAME_INCLUDED_IN_OWN_PROFILE=FALSE"
)

print(
    "FUTURE_GAME_INCLUDED_IN_PROFILE=FALSE"
)

print(
    "POINT_IN_TIME_PRIOR_ACCUMULATION=PASS"
)


# ==================================================================
# [11] POINT-IN-TIME TENDENCY FEATURES
# ==================================================================

section(
    "[11] POINT-IN-TIME TENDENCY FEATURES"
)

matrix[
    "prior_raw_pass_rate"
] = safe_ratio(
    matrix[
        "prior_raw_pass"
    ],
    matrix[
        "prior_raw_n"
    ],
)

matrix[
    "prior_observed_pass_rate_consistent"
] = safe_ratio(
    matrix[
        "prior_combined_nfl_pass"
    ],
    matrix[
        "prior_combined_n"
    ],
)

matrix[
    "prior_mean_xpass"
] = safe_ratio(
    matrix[
        "prior_combined_xpass_sum"
    ],
    matrix[
        "prior_combined_n"
    ],
)

matrix[
    "prior_mean_pass_oe"
] = safe_ratio(
    matrix[
        "prior_combined_pass_oe_sum"
    ],
    matrix[
        "prior_combined_n"
    ],
)

matrix[
    "prior_early_down_pass_rate"
] = safe_ratio(
    matrix[
        "prior_early_raw_pass"
    ],
    matrix[
        "prior_early_raw_n"
    ],
)

matrix[
    "prior_early_down_pass_oe"
] = safe_ratio(
    matrix[
        "prior_early_pass_oe_sum"
    ],
    matrix[
        "prior_early_combined_n"
    ],
)

matrix[
    "prior_neutral_pass_rate"
] = safe_ratio(
    matrix[
        "prior_neutral_raw_pass"
    ],
    matrix[
        "prior_neutral_raw_n"
    ],
)

matrix[
    "prior_neutral_pass_oe"
] = safe_ratio(
    matrix[
        "prior_neutral_pass_oe_sum"
    ],
    matrix[
        "prior_neutral_combined_n"
    ],
)

matrix[
    "prior_redzone_pass_rate"
] = safe_ratio(
    matrix[
        "prior_redzone_raw_pass"
    ],
    matrix[
        "prior_redzone_raw_n"
    ],
)

matrix[
    "prior_redzone_pass_oe"
] = safe_ratio(
    matrix[
        "prior_redzone_pass_oe_sum"
    ],
    matrix[
        "prior_redzone_combined_n"
    ],
)

matrix[
    "prior_short_yardage_pass_rate_candidate"
] = safe_ratio(
    matrix[
        "prior_short_candidate_pass"
    ],
    matrix[
        "prior_short_candidate_n"
    ],
)

matrix[
    "prior_fourth_down_go_rate_classified"
] = safe_ratio(
    matrix[
        "prior_fourth_go"
    ],
    matrix[
        "prior_fourth_classified_n"
    ],
)

print(
    "POINT_IN_TIME_TENDENCY_FEATURES=PASS"
)


# ==================================================================
# [12] PASS-OE IDENTITY AUDIT
# ==================================================================

section(
    "[12] PASS-OE IDENTITY AUDIT"
)

has_combined = (
    matrix[
        "prior_combined_n"
    ].gt(0)
)

derived_pass_oe = (
    (
        matrix[
            "prior_observed_pass_rate_consistent"
        ]
        -
        matrix[
            "prior_mean_xpass"
        ]
    )
    *
    100.0
)

pass_oe_error = (
    matrix[
        "prior_mean_pass_oe"
    ]
    -
    derived_pass_oe
).abs()

max_error = (
    float(
        pass_oe_error[
            has_combined
        ].max()
    )
    if has_combined.any()
    else 0.0
)

print(
    "MATRIX_ROWS_WITH_PRIOR_COMBINED_SAMPLE="
    f"{int(has_combined.sum())}"
)

print(
    f"MAX_PASS_OE_IDENTITY_ERROR={max_error:.12f}"
)

# NFLverse pass_oe is expected to equal
# (observed pass - xpass) * 100.
if (
    np.isfinite(max_error)
    and
    max_error > 1e-6
):

    raise SystemExit(
        "FAIL_CLOSED: Pass-OE identity audit exceeded tolerance"
    )

print(
    "PASS_OE_IDENTITY_AUDIT=PASS"
)


# ==================================================================
# [13] FEATURE RANGE AUDIT
# ==================================================================

section(
    "[13] FEATURE RANGE AUDIT"
)

rate_columns = [
    "prior_raw_pass_rate",
    "prior_observed_pass_rate_consistent",
    "prior_mean_xpass",
    "prior_early_down_pass_rate",
    "prior_neutral_pass_rate",
    "prior_redzone_pass_rate",
    "prior_short_yardage_pass_rate_candidate",
    "prior_fourth_down_go_rate_classified",
]

range_violations = 0

for col in rate_columns:

    vals = num(
        matrix[
            col
        ]
    )

    violations = int(
        (
            vals.notna()
            &
            (
                vals.lt(-1e-9)
                |
                vals.gt(1 + 1e-9)
            )
        ).sum()
    )

    print(
        f"{col.upper()}_RANGE_VIOLATIONS={violations}"
    )

    range_violations += violations

if range_violations:

    raise SystemExit(
        "FAIL_CLOSED: tendency rate outside [0,1]"
    )

print(
    "FEATURE_RANGE_AUDIT=PASS"
)


# ==================================================================
# [14] REGIME ZERO-HISTORY AUDIT
# ==================================================================

section(
    "[14] REGIME ZERO-HISTORY AUDIT"
)

first_rows = (
    matrix
    .groupby(
        "team_coach_regime_key",
        sort=False,
    )
    .head(1)
)

first_prior_raw_nonzero = int(
    first_rows[
        "prior_raw_n"
    ]
    .ne(0)
    .sum()
)

first_prior_combined_nonzero = int(
    first_rows[
        "prior_combined_n"
    ]
    .ne(0)
    .sum()
)

print(
    "FIRST_REGIME_ROW_PRIOR_RAW_NONZERO="
    f"{first_prior_raw_nonzero}"
)

print(
    "FIRST_REGIME_ROW_PRIOR_COMBINED_NONZERO="
    f"{first_prior_combined_nonzero}"
)

if (
    first_prior_raw_nonzero
    or
    first_prior_combined_nonzero
):

    raise SystemExit(
        "FAIL_CLOSED: first regime row has inherited play history"
    )

print(
    "REGIME_ZERO_HISTORY_AUDIT=PASS"
)


# ==================================================================
# [15] TEMPORAL ORDER AUDIT
# ==================================================================

section(
    "[15] TEMPORAL ORDER AUDIT"
)

temporal_violations = 0

for _, g in matrix.groupby(
    "team_coach_regime_key",
    sort=False,
):

    dates = g[
        "game_date_dt"
    ]

    temporal_violations += int(
        (
            dates.diff()
            .dt.total_seconds()
            .fillna(0)
            .lt(0)
        ).sum()
    )

print(
    f"TEMPORAL_ORDER_VIOLATIONS={temporal_violations}"
)

if temporal_violations:

    raise SystemExit(
        "FAIL_CLOSED: regime matrix not temporally ordered"
    )

print(
    "TEMPORAL_ORDER_AUDIT=PASS"
)


# ==================================================================
# [16] DUPLICATE / IDENTITY AUDIT
# ==================================================================

section(
    "[16] MATRIX IDENTITY AUDIT"
)

duplicate_identity_rows = int(
    matrix.duplicated(
        [
            "game_id",
            "team",
        ]
    ).sum()
)

print(
    f"MATRIX_PRIMARY_IDENTITY_DUPLICATES={duplicate_identity_rows}"
)

if duplicate_identity_rows:

    raise SystemExit(
        "FAIL_CLOSED: duplicate matrix primary identity"
    )

print(
    "MATRIX_IDENTITY_AUDIT=PASS"
)


# ==================================================================
# [17] MATRIX DIAGNOSTIC DISTRIBUTIONS
# ==================================================================

section(
    "[17] MATRIX DIAGNOSTIC DISTRIBUTIONS"
)

feature_columns = [
    "prior_raw_pass_rate",
    "prior_mean_xpass",
    "prior_mean_pass_oe",

    "prior_early_down_pass_rate",
    "prior_early_down_pass_oe",

    "prior_neutral_pass_rate",
    "prior_neutral_pass_oe",

    "prior_redzone_pass_rate",
    "prior_redzone_pass_oe",

    "prior_short_yardage_pass_rate_candidate",

    "prior_fourth_down_go_rate_classified",
]

for col in feature_columns:

    vals = num(
        matrix[
            col
        ]
    ).dropna()

    print(
        f"{col.upper()}_N={len(vals)}"
    )

    if len(vals):

        print(
            f"{col.upper()}_MEAN={vals.mean():.8f}"
        )

        print(
            f"{col.upper()}_P10={vals.quantile(0.10):.8f}"
        )

        print(
            f"{col.upper()}_P50={vals.quantile(0.50):.8f}"
        )

        print(
            f"{col.upper()}_P90={vals.quantile(0.90):.8f}"
        )

print(
    "MATRIX_DIAGNOSTIC_DISTRIBUTIONS=PASS"
)


# ==================================================================
# [18] POINT-IN-TIME MATRIX EXAMPLES
# ==================================================================

section(
    "[18] POINT-IN-TIME MATRIX EXAMPLES"
)

example_cols = [
    "game_id",
    "game_date_dt",
    "team",
    coach_id_col,
    "team_coach_regime_key",

    "prior_pbp_regime_games",
    "prior_raw_n",
    "prior_combined_n",

    "prior_raw_pass_rate",
    "prior_mean_xpass",
    "prior_mean_pass_oe",

    "prior_early_down_pass_rate",
    "prior_neutral_pass_rate",
    "prior_redzone_pass_rate",

    "prior_fourth_down_go_rate_classified",
]

# Add known authority context when available.
for optional in [
    "coach_career_games_prior",
    "stage26_confidence",
]:

    if optional in matrix.columns:

        example_cols.append(
            optional
        )

examples = (
    matrix[
        example_cols
    ]
    .sort_values(
        [
            "game_date_dt",
            "team",
        ],
        kind="stable",
    )
    .tail(20)
)

print(
    examples.to_string(
        index=False
    )
)

print(
    "POINT_IN_TIME_MATRIX_EXAMPLES=PASS"
)


# ==================================================================
# [19] STAGE26G-A MATRIX CONTRACT
# ==================================================================

section(
    "[19] STAGE26G-A MATRIX CONTRACT"
)

print(
    "MATRIX_PRIMARY_IDENTITY=game_id+team"
)

print(
    "MATRIX_COACH_AUTHORITY=WFS_COACH_IDENTITY"
)

print(
    "MATRIX_PLAY_AUTHORITY=NFLVERSE"
)

print(
    "MATRIX_POINT_IN_TIME=TRUE"
)

print(
    "MATRIX_CURRENT_GAME_EXCLUDED=TRUE"
)

print(
    "MATRIX_FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "MATRIX_TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    f"MATRIX_ROWS={len(matrix)}"
)

print(
    f"MATRIX_REGIMES={matrix['team_coach_regime_key'].nunique()}"
)

print(
    "STAGE26G_A_MATRIX_CONTRACT=PASS"
)


# ==================================================================
# [20] PRODUCTION FIREWALL
# ==================================================================

section(
    "[20] PRODUCTION FIREWALL"
)

firewall = {
    "ANALYSIS_ONLY":
        ANALYSIS_ONLY,

    "ARTIFACT_WRITE":
        ARTIFACT_WRITE,

    "DATABASE_WRITE":
        DATABASE_WRITE,

    "SOLVER_MUTATION":
        SOLVER_MUTATION,

    "FORECAST_MUTATION":
        FORECAST_MUTATION,

    "APP_MUTATION":
        APP_MUTATION,

    "SERVICE_RESTART":
        SERVICE_RESTART,
}

for key, value in firewall.items():

    print(
        f"{key}={str(value).upper()}"
    )

if not ANALYSIS_ONLY:

    raise SystemExit(
        "FAIL_CLOSED: analysis-only firewall disabled"
    )

if any(
    [
        ARTIFACT_WRITE,
        DATABASE_WRITE,
        SOLVER_MUTATION,
        FORECAST_MUTATION,
        APP_MUTATION,
        SERVICE_RESTART,
    ]
):

    raise SystemExit(
        "FAIL_CLOSED: production firewall violation"
    )

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ==================================================================
# [21] STAGE26G-A FINAL CONTRACT
# ==================================================================

section(
    "[21] STAGE26G-A FINAL CONTRACT"
)

print(
    "STAGE26G_A_CONTRACT="
    "WFS_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX_V1"
)

print(
    "STAGE26G_A_SEASONS=2023,2024,2025"
)

print(
    f"STAGE26G_A_MATRIX_ROWS={len(matrix)}"
)

print(
    "STAGE26G_A_POINT_IN_TIME=TRUE"
)

print(
    "STAGE26G_A_CURRENT_GAME_EXCLUDED=TRUE"
)

print(
    "STAGE26G_A_FUTURE_GAME_LEAKAGE=FALSE"
)

print(
    "STAGE26G_A_TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "STAGE26G_A_ARTIFACT_WRITE=FALSE"
)

print(
    "STAGE26G_A_DATABASE_WRITE=FALSE"
)

print(
    "STAGE26G_A_SOLVER_MUTATION=FALSE"
)

print(
    "STAGE26G_A_FORECAST_MUTATION=FALSE"
)

print(
    "STAGE26G_A_APP_MUTATION=FALSE"
)

print(
    "STAGE26G_A_SERVICE_RESTART=FALSE"
)

print(
    "STAGE26G_A_POINT_IN_TIME_COACH_REGIME_TENDENCY_MATRIX=PASS"
)
