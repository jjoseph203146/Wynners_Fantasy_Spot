#!/usr/bin/env python3

"""
Stage26G-I
WFS Coach Intelligence — Shadow Integration Contract

Purpose
-------
Validate the integration boundary between:

1. the existing WFS team pregame environment, and
2. the validated Stage26G coach-regime Pass-OE signal.

This stage is intentionally ANALYSIS / SHADOW ONLY.

It does NOT:
- modify team_environment.py
- write database rows
- write artifacts
- mutate forecasts
- mutate solver inputs
- mutate the app
- restart services

Important deployment constraint
-------------------------------
The historical Stage26G research baseline used realized current-game xPass
for retrospective validation.

That value is NOT a deployable pregame feature.

Therefore this stage does NOT manufacture a fake pregame xPass baseline.

Instead it validates that the coach signal can be attached safely to the
existing point-in-time team pregame environment.

A later activation step may consume the attached coach signal only after
a legitimate pregame contextual expectation is explicitly defined.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RUNTIME CONTRACT
# ============================================================

ANALYSIS_ONLY = True
SHADOW_INTEGRATION = True

COACH_SIGNAL = "prior_mean_pass_oe"
COACH_WEIGHT = 0.70

COLD_START_ZERO_ADJUSTMENT = True
TEAM_COACH_REGIME_RESET = True

PREGAME_CONTEXT_BASELINE_REQUIRED = True
REALIZED_CURRENT_GAME_XPASS_ALLOWED = False

ARTIFACT_WRITE = False
DATABASE_WRITE = False
CODE_MUTATION = False
SOLVER_MUTATION = False
FORECAST_MUTATION = False
APP_MUTATION = False
SERVICE_RESTART = False
PRODUCTION_INFLUENCE = False


ROOT = Path(
    os.environ.get(
        "NFL_DATA_ENGINE_ROOT",
        "/home/mwynn/nfl_data_engine",
    )
)

SCRIPTS = ROOT / "scripts"

STAGE26GA_SCRIPT = (
    SCRIPTS
    / "stage26ga_coach_regime_tendency_matrix.py"
)

TEAM_ENV_CSV = (
    ROOT
    / "data"
    / "csv"
    / "nfl_team_pregame_environment.csv"
)

TEAM_ENV_PARQUET = (
    ROOT
    / "data"
    / "parquet"
    / "nfl_team_pregame_environment.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message: str) -> None:
    raise RuntimeError(message)


def bool_text(value: bool) -> str:
    return "TRUE" if bool(value) else "FALSE"


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def load_team_environment() -> tuple[pd.DataFrame, Path]:
    """
    Read the existing persisted team pregame environment.

    Prefer parquet. Fall back to CSV.
    """

    if TEAM_ENV_PARQUET.exists():
        df = pd.read_parquet(
            TEAM_ENV_PARQUET
        )
        return df, TEAM_ENV_PARQUET

    if TEAM_ENV_CSV.exists():
        df = pd.read_csv(
            TEAM_ENV_CSV
        )
        return df, TEAM_ENV_CSV

    fail(
        "No existing team pregame environment "
        "artifact found.\n"
        f"Checked:\n"
        f"  {TEAM_ENV_PARQUET}\n"
        f"  {TEAM_ENV_CSV}"
    )


def recover_matrix(namespace: dict) -> pd.DataFrame:
    """
    Recover the Stage26G-A matrix without assuming one single
    variable name unless necessary.
    """

    preferred_names = [
        "matrix",
        "coach_matrix",
        "tendency_matrix",
        "stage26ga_matrix",
    ]

    for name in preferred_names:
        value = namespace.get(name)

        if isinstance(
            value,
            pd.DataFrame,
        ):
            required = {
                "game_id",
                "team",
                "team_coach_regime_key",
                "prior_pbp_regime_games",
                COACH_SIGNAL,
            }

            if required.issubset(
                value.columns
            ):
                return value.copy()

    candidates = []

    for name, value in namespace.items():

        if not isinstance(
            value,
            pd.DataFrame,
        ):
            continue

        required = {
            "game_id",
            "team",
            "team_coach_regime_key",
            "prior_pbp_regime_games",
            COACH_SIGNAL,
        }

        if required.issubset(
            value.columns
        ):
            candidates.append(
                (
                    name,
                    len(value),
                    value,
                )
            )

    if not candidates:
        fail(
            "Unable to recover Stage26G-A matrix "
            "from runpy namespace."
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    name, rows, value = candidates[0]

    print(
        f"Recovered G-A matrix variable: "
        f"{name} ({rows} rows)"
    )

    return value.copy()


# ============================================================
# RUNTIME
# ============================================================

section(
    "STAGE26G-I — COACH SHADOW INTEGRATION"
)

print(
    f"ANALYSIS_ONLY="
    f"{bool_text(ANALYSIS_ONLY)}"
)

print(
    f"SHADOW_INTEGRATION="
    f"{bool_text(SHADOW_INTEGRATION)}"
)

print(
    f"COACH_SIGNAL="
    f"{COACH_SIGNAL}"
)

print(
    f"COACH_WEIGHT="
    f"{COACH_WEIGHT:.2f}"
)

print(
    "COLD_START_ZERO_ADJUSTMENT="
    f"{bool_text(COLD_START_ZERO_ADJUSTMENT)}"
)

print(
    "PREGAME_CONTEXT_BASELINE_REQUIRED="
    f"{bool_text(PREGAME_CONTEXT_BASELINE_REQUIRED)}"
)

print(
    "REALIZED_CURRENT_GAME_XPASS_ALLOWED="
    f"{bool_text(REALIZED_CURRENT_GAME_XPASS_ALLOWED)}"
)


# ============================================================
# 1. LOAD EXISTING TEAM ENVIRONMENT
# ============================================================

section(
    "1. EXISTING TEAM PREGAME ENVIRONMENT"
)

team_env, team_env_source = (
    load_team_environment()
)

print(
    f"TEAM_ENV_SOURCE="
    f"{team_env_source}"
)

print(
    f"TEAM_ENV_ROWS="
    f"{len(team_env)}"
)

required_env = {
    "game_id",
    "season",
    "week",
    "team",
    "opponent_team",
    "history_games",
    "pass_rate_avg_3",
    "pass_rate_avg_5",
    "pass_rate_trend",
}

missing_env = (
    required_env
    -
    set(team_env.columns)
)

if missing_env:
    fail(
        "Team environment missing required "
        f"columns: {sorted(missing_env)}"
    )

team_env = team_env.copy()

team_env["game_id"] = (
    team_env["game_id"]
    .astype(str)
)

team_env["team"] = (
    team_env["team"]
    .astype(str)
    .str.strip()
    .str.upper()
)

env_duplicates = int(
    team_env.duplicated(
        subset=[
            "game_id",
            "team",
        ]
    ).sum()
)

print(
    "TEAM_ENV_DUPLICATE_GAME_TEAM_ROWS="
    f"{env_duplicates}"
)

if env_duplicates != 0:
    fail(
        "Existing team environment violates "
        "game_id+team uniqueness."
    )

print(
    "TEAM_ENVIRONMENT_AUTHORITY=PASS"
)


# ============================================================
# 2. LOAD FROZEN STAGE26G-A AUTHORITY
# ============================================================

section(
    "2. FROZEN STAGE26G-A COACH AUTHORITY"
)

if not STAGE26GA_SCRIPT.exists():
    fail(
        "Stage26G-A script not found: "
        f"{STAGE26GA_SCRIPT}"
    )

namespace = runpy.run_path(
    str(STAGE26GA_SCRIPT)
)

coach = recover_matrix(
    namespace
)

required_coach = {
    "game_id",
    "season",
    "week",
    "team",
    "team_coach_regime_key",
    "prior_pbp_regime_games",
    COACH_SIGNAL,
}

missing_coach = (
    required_coach
    -
    set(coach.columns)
)

if missing_coach:
    fail(
        "G-A matrix missing required "
        f"columns: {sorted(missing_coach)}"
    )

coach = coach[
    [
        "game_id",
        "season",
        "week",
        "team",
        "team_coach_regime_key",
        "prior_pbp_regime_games",
        COACH_SIGNAL,
    ]
].copy()

coach["game_id"] = (
    coach["game_id"]
    .astype(str)
)

coach["team"] = (
    coach["team"]
    .astype(str)
    .str.strip()
    .str.upper()
)

coach["prior_pbp_regime_games"] = (
    pd.to_numeric(
        coach[
            "prior_pbp_regime_games"
        ],
        errors="coerce",
    )
)

coach[COACH_SIGNAL] = (
    pd.to_numeric(
        coach[COACH_SIGNAL],
        errors="coerce",
    )
)

coach_duplicates = int(
    coach.duplicated(
        subset=[
            "game_id",
            "team",
        ]
    ).sum()
)

print(
    "COACH_DUPLICATE_GAME_TEAM_ROWS="
    f"{coach_duplicates}"
)

if coach_duplicates != 0:
    fail(
        "G-A matrix violates game_id+team "
        "uniqueness."
    )

print(
    f"COACH_ROWS="
    f"{len(coach)}"
)

print(
    "STAGE26G_A_AUTHORITY=PASS"
)


# ============================================================
# 3. JOIN CONTRACT
# ============================================================

section(
    "3. SHADOW JOIN CONTRACT"
)

shadow = team_env.merge(
    coach,
    how="left",
    on=[
        "game_id",
        "team",
    ],
    suffixes=(
        "",
        "_coach",
    ),
    validate="one_to_one",
    indicator=True,
)

matched = int(
    shadow["_merge"]
    .eq("both")
    .sum()
)

env_only = int(
    shadow["_merge"]
    .eq("left_only")
    .sum()
)

match_rate = (
    matched
    /
    len(shadow)
    if len(shadow)
    else 0.0
)

print(
    f"SHADOW_ROWS="
    f"{len(shadow)}"
)

print(
    f"COACH_MATCHED_ROWS="
    f"{matched}"
)

print(
    f"TEAM_ENV_ONLY_ROWS="
    f"{env_only}"
)

print(
    f"COACH_MATCH_RATE="
    f"{match_rate:.8f}"
)

if len(shadow) != len(team_env):
    fail(
        "Coach join changed team environment "
        "row cardinality."
    )

# Historical environments can legitimately contain
# seasons outside the Stage26G research window.
#
# Therefore unmatched rows are NOT silently assigned
# coach history. They become explicit no-adjustment rows.

shadow["coach_signal_available"] = (
    shadow["_merge"].eq("both")
    &
    shadow[
        "prior_pbp_regime_games"
    ].notna()
)


# ============================================================
# 4. IDENTITY CONSISTENCY
# ============================================================

section(
    "4. GAME / TEAM IDENTITY CONSISTENCY"
)

season_mismatch = 0
week_mismatch = 0

matched_mask = (
    shadow["_merge"].eq("both")
)

if "season_coach" in shadow.columns:

    season_mismatch = int(
        (
            finite_numeric(
                shadow.loc[
                    matched_mask,
                    "season",
                ]
            )
            !=
            finite_numeric(
                shadow.loc[
                    matched_mask,
                    "season_coach",
                ]
            )
        ).sum()
    )

if "week_coach" in shadow.columns:

    week_mismatch = int(
        (
            finite_numeric(
                shadow.loc[
                    matched_mask,
                    "week",
                ]
            )
            !=
            finite_numeric(
                shadow.loc[
                    matched_mask,
                    "week_coach",
                ]
            )
        ).sum()
    )

print(
    f"SEASON_IDENTITY_MISMATCH_ROWS="
    f"{season_mismatch}"
)

print(
    f"WEEK_IDENTITY_MISMATCH_ROWS="
    f"{week_mismatch}"
)

if (
    season_mismatch != 0
    or
    week_mismatch != 0
):
    fail(
        "Coach/team environment identity "
        "mismatch detected."
    )

print(
    "GAME_TEAM_IDENTITY_CONTRACT=PASS"
)


# ============================================================
# 5. COLD-START / REGIME POLICY
# ============================================================

section(
    "5. COACH REGIME COLD-START POLICY"
)

prior_games = finite_numeric(
    shadow[
        "prior_pbp_regime_games"
    ]
)

signal = finite_numeric(
    shadow[
        COACH_SIGNAL
    ]
)

cold_start = (
    shadow["coach_signal_available"]
    &
    prior_games.eq(0)
)

established = (
    shadow["coach_signal_available"]
    &
    prior_games.gt(0)
    &
    signal.notna()
)

cold_start_rows = int(
    cold_start.sum()
)

established_rows = int(
    established.sum()
)

print(
    f"COLD_START_ROWS="
    f"{cold_start_rows}"
)

print(
    f"ESTABLISHED_REGIME_ROWS="
    f"{established_rows}"
)

shadow[
    "coach_pass_oe_signal"
] = 0.0

shadow.loc[
    established,
    "coach_pass_oe_signal",
] = signal.loc[
    established
]

shadow[
    "coach_pass_oe_weight"
] = 0.0

shadow.loc[
    established,
    "coach_pass_oe_weight",
] = COACH_WEIGHT

# Pass-OE is expressed in percentage points.
# Convert to pass-probability scale for integration.
shadow[
    "coach_pass_rate_adjustment_shadow"
] = (
    shadow[
        "coach_pass_oe_signal"
    ]
    *
    shadow[
        "coach_pass_oe_weight"
    ]
    /
    100.0
)

cold_adjustment_nonzero = int(
    (
        shadow.loc[
            cold_start,
            "coach_pass_rate_adjustment_shadow",
        ]
        .abs()
        >
        1e-12
    ).sum()
)

print(
    "COLD_START_NONZERO_ADJUSTMENT_ROWS="
    f"{cold_adjustment_nonzero}"
)

if cold_adjustment_nonzero != 0:
    fail(
        "Cold-start coach rows received a "
        "nonzero adjustment."
    )

print(
    "COLD_START_POLICY=ZERO_COACH_ADJUSTMENT"
)

print(
    "TEAM_COACH_REGIME_RESET="
    f"{bool_text(TEAM_COACH_REGIME_RESET)}"
)

print(
    "COACH_REGIME_POLICY=PASS"
)


# ============================================================
# 6. SHADOW SIGNAL DISTRIBUTION
# ============================================================

section(
    "6. SHADOW COACH ADJUSTMENT DISTRIBUTION"
)

adjustments = (
    shadow.loc[
        established,
        "coach_pass_rate_adjustment_shadow",
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .dropna()
)

nonfinite_adjustments = int(
    (
        ~np.isfinite(
            shadow[
                "coach_pass_rate_adjustment_shadow"
            ]
        )
    ).sum()
)

print(
    "NONFINITE_COACH_ADJUSTMENT_ROWS="
    f"{nonfinite_adjustments}"
)

if nonfinite_adjustments != 0:
    fail(
        "Nonfinite coach adjustment detected."
    )

if len(adjustments):

    quantiles = adjustments.quantile(
        [
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )

    print(
        f"ADJUSTMENT_P01="
        f"{quantiles.loc[0.01]:.8f}"
    )

    print(
        f"ADJUSTMENT_P05="
        f"{quantiles.loc[0.05]:.8f}"
    )

    print(
        f"ADJUSTMENT_P50="
        f"{quantiles.loc[0.50]:.8f}"
    )

    print(
        f"ADJUSTMENT_P95="
        f"{quantiles.loc[0.95]:.8f}"
    )

    print(
        f"ADJUSTMENT_P99="
        f"{quantiles.loc[0.99]:.8f}"
    )

    print(
        f"ADJUSTMENT_MAX_ABS="
        f"{adjustments.abs().max():.8f}"
    )

else:

    print(
        "No established coach-regime rows "
        "available for distribution audit."
    )

print(
    "SHADOW_SIGNAL_DISTRIBUTION=PASS"
)


# ============================================================
# 7. EXISTING ENVIRONMENT IMMUTABILITY
# ============================================================

section(
    "7. EXISTING ENVIRONMENT IMMUTABILITY"
)

original_columns = list(
    team_env.columns
)

existing_mutated = False

for column in original_columns:

    left = team_env[column].reset_index(
        drop=True
    )

    right = shadow[column].reset_index(
        drop=True
    )

    if not left.equals(right):
        existing_mutated = True
        print(
            "MUTATED_EXISTING_COLUMN="
            f"{column}"
        )

print(
    "EXISTING_TEAM_ENVIRONMENT_MUTATED="
    f"{bool_text(existing_mutated)}"
)

if existing_mutated:
    fail(
        "Shadow integration mutated an "
        "existing team environment field."
    )

print(
    "EXISTING_ENVIRONMENT_IMMUTABILITY=PASS"
)


# ============================================================
# 8. PREGAME DEPLOYMENT FIREWALL
# ============================================================

section(
    "8. PREGAME DEPLOYMENT FIREWALL"
)

forbidden_realized_columns = {
    "current_mean_xpass",
    "current_pass_oe",
    "current_observed_pass_rate",
}

used_shadow_columns = {
    "game_id",
    "team",
    "team_coach_regime_key",
    "prior_pbp_regime_games",
    COACH_SIGNAL,
}

forbidden_used = (
    forbidden_realized_columns
    &
    used_shadow_columns
)

print(
    "REALIZED_CURRENT_GAME_FEATURES_USED="
    f"{bool_text(bool(forbidden_used))}"
)

print(
    "PREGAME_CONTEXT_BASELINE_REQUIRED="
    f"{bool_text(PREGAME_CONTEXT_BASELINE_REQUIRED)}"
)

print(
    "REALIZED_CURRENT_GAME_XPASS_ALLOWED="
    f"{bool_text(REALIZED_CURRENT_GAME_XPASS_ALLOWED)}"
)

if forbidden_used:
    fail(
        "Realized current-game feature crossed "
        "the pregame deployment firewall."
    )

print(
    "PREGAME_DEPLOYMENT_FIREWALL=PASS"
)


# ============================================================
# 9. PRODUCTION FIREWALL
# ============================================================

section(
    "9. PRODUCTION FIREWALL"
)

firewall = {
    "ANALYSIS_ONLY":
        ANALYSIS_ONLY,

    "SHADOW_INTEGRATION":
        SHADOW_INTEGRATION,

    "ARTIFACT_WRITE":
        ARTIFACT_WRITE,

    "DATABASE_WRITE":
        DATABASE_WRITE,

    "CODE_MUTATION":
        CODE_MUTATION,

    "SOLVER_MUTATION":
        SOLVER_MUTATION,

    "FORECAST_MUTATION":
        FORECAST_MUTATION,

    "APP_MUTATION":
        APP_MUTATION,

    "SERVICE_RESTART":
        SERVICE_RESTART,

    "PRODUCTION_INFLUENCE":
        PRODUCTION_INFLUENCE,
}

for key, value in firewall.items():

    print(
        f"{key}="
        f"{bool_text(value)}"
    )

for key in [
    "ARTIFACT_WRITE",
    "DATABASE_WRITE",
    "CODE_MUTATION",
    "SOLVER_MUTATION",
    "FORECAST_MUTATION",
    "APP_MUTATION",
    "SERVICE_RESTART",
    "PRODUCTION_INFLUENCE",
]:

    if firewall[key]:
        fail(
            f"Production firewall violated: "
            f"{key}=TRUE"
        )

print(
    "PRODUCTION_FIREWALL=PASS"
)


# ============================================================
# 10. FINAL CONTRACT
# ============================================================

section(
    "10. STAGE26G-I FINAL CONTRACT"
)

integration_ready = (
    not existing_mutated
    and
    env_duplicates == 0
    and
    coach_duplicates == 0
    and
    season_mismatch == 0
    and
    week_mismatch == 0
    and
    cold_adjustment_nonzero == 0
    and
    nonfinite_adjustments == 0
    and
    not bool(forbidden_used)
)

print(
    "STAGE26G_I_CONTRACT="
    "WFS_COACH_SHADOW_INTEGRATION_V1"
)

print(
    "SOURCE_TEAM_ENVIRONMENT="
    "WFS_TEAM_PREGAME_ENVIRONMENT"
)

print(
    "SOURCE_COACH_SIGNAL="
    "STAGE26G_A_PRIOR_MEAN_PASS_OE"
)

print(
    f"COACH_WEIGHT="
    f"{COACH_WEIGHT:.2f}"
)

print(
    "COACH_ADJUSTMENT_SCALE="
    "PASS_PROBABILITY"
)

print(
    "COLD_START_POLICY="
    "ZERO_COACH_ADJUSTMENT"
)

print(
    "TEAM_COACH_REGIME_RESET=TRUE"
)

print(
    "PREGAME_CONTEXT_BASELINE_REQUIRED=TRUE"
)

print(
    "REALIZED_CURRENT_GAME_XPASS_USED=FALSE"
)

print(
    "DATABASE_WRITE=FALSE"
)

print(
    "ARTIFACT_WRITE=FALSE"
)

print(
    "FORECAST_MUTATION=FALSE"
)

print(
    "SOLVER_MUTATION=FALSE"
)

print(
    "APP_MUTATION=FALSE"
)

print(
    "SERVICE_RESTART=FALSE"
)

print(
    "PRODUCTION_INFLUENCE=FALSE"
)

print(
    "COACH_SIGNAL_READY_FOR_PREGAME_INTEGRATION="
    f"{bool_text(integration_ready)}"
)

if not integration_ready:
    fail(
        "Stage26G-I integration contract failed."
    )

print()
print(
    "STAGE26G_I_COACH_SHADOW_INTEGRATION=PASS"
)
