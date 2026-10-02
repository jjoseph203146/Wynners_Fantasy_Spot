from __future__ import annotations

from pathlib import Path
import hashlib
import json
import runpy
import contextlib
import io

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

R8Q1_SCRIPT = ROOT / "scripts/build_cold_start_stat_forecasts_shadow_v1.py"
R8Q1_SHADOW = ROOT / "data/parquet/current_cold_start_stat_forecasts_shadow.parquet"
CANONICAL = ROOT / "data/parquet/current_unified_stat_forecasts.parquet"

OUT = ROOT / "data/parquet/current_cold_start_full_team_shadow_v2.parquet"
AUDIT = ROOT / "data/parquet/current_cold_start_full_team_shadow_v2_audit.json"

EXPECTED_R8Q1_SCRIPT_SHA = (
    "c3a9cecbb8b6efc2bbce57c60d77f791ce83265045b2c0600bb4736e453333aa"
)

EXPECTED_CANONICAL_SHA = (
    "6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9"
)

OFFENSE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def n(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def safe_div(a, b, default=0.0):
    a = n(a, np.nan)
    b = n(b, np.nan)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return a / b


def fd_points(row) -> float:
    return (
        0.04 * n(row.get("expected_passing_yards"))
        + 4.0 * n(row.get("expected_passing_tds"))
        - 1.0 * n(row.get("expected_interceptions"))
        + 0.10 * n(row.get("expected_rushing_yards"))
        + 6.0 * n(row.get("expected_rushing_tds"))
        + 0.50 * n(row.get("expected_receptions"))
        + 0.10 * n(row.get("expected_receiving_yards"))
        + 6.0 * n(row.get("expected_receiving_tds"))
    )


# ============================================================
# SAFETY GATES
# ============================================================

for path in [R8Q1_SCRIPT, R8Q1_SHADOW, CANONICAL]:
    if not path.exists():
        raise RuntimeError(f"FAIL | missing {path}")

r8q1_script_sha = sha256(R8Q1_SCRIPT)
canonical_sha_before = sha256(CANONICAL)

if r8q1_script_sha != EXPECTED_R8Q1_SCRIPT_SHA:
    raise RuntimeError(
        "FAIL | R8Q1 script SHA mismatch\n"
        f"EXPECTED={EXPECTED_R8Q1_SCRIPT_SHA}\n"
        f"ACTUAL={r8q1_script_sha}"
    )

if canonical_sha_before != EXPECTED_CANONICAL_SHA:
    raise RuntimeError(
        "FAIL | canonical SHA mismatch\n"
        f"EXPECTED={EXPECTED_CANONICAL_SHA}\n"
        f"ACTUAL={canonical_sha_before}"
    )


# ============================================================
# LOAD R8Q1 NAMESPACE
#
# Reuse its exact validated historical rookie-prior machinery.
# Its execution rewrites shadow research artifacts only.
# Production remains protected by canonical SHA gates.
# ============================================================

capture = io.StringIO()

with contextlib.redirect_stdout(capture):
    ns = runpy.run_path(
        str(R8Q1_SCRIPT),
        run_name="__stage26gr8v_r8q1_loader__",
    )

shadow = pd.read_parquet(R8Q1_SHADOW)
canon = pd.read_parquet(CANONICAL)

if len(shadow) != 2:
    raise RuntimeError(
        f"FAIL | expected current R8Q1 rookie rows=2, got {len(shadow)}"
    )


# ============================================================
# LOCATE EXACT R8Q1 HIERARCHICAL PRIORS
# ============================================================

prior_lookup = None

# R8Q1 may expose the constructed prior dictionary/dataframe
# under different internal variable names. Identify it by
# structure, never by fuzzy player matching.
for name, obj in ns.items():

    if isinstance(obj, dict) and obj:
        sample = next(iter(obj.values()))

        if isinstance(sample, dict):
            keys = set(sample.keys())

            if {
                "targets",
                "receptions",
                "receiving_yards",
            }.issubset(keys):
                prior_lookup = obj
                break


# If no direct lookup survived at module scope, use the exact
# prior-construction function already defined by R8Q1.
prior_function = None

for name, obj in ns.items():
    if callable(obj):
        lname = name.lower()

        if (
            "prior" in lname
            and "rookie" in lname
        ):
            prior_function = obj
            break


def get_prior_for_shadow_row(row):
    """
    Use the exact validated R8Q1 hierarchical prior builder.

    R8Q1 exposes:
        hierarchical_prior(position: str) -> dict[str, float]

    This is the authoritative prior used by the original
    cold-start shadow and avoids any reconstruction or
    reverse engineering from DvP-contaminated outputs.
    """

    pos = str(
        row["position"]
    ).upper()

    builder = ns.get(
        "hierarchical_prior"
    )

    if not callable(builder):
        raise RuntimeError(
            "FAIL | R8Q1 hierarchical_prior function unavailable"
        )

    priors = ns.get(
        "PRIORS"
    )

    if not isinstance(
        priors,
        dict,
    ):
        raise RuntimeError(
            "FAIL | R8Q1 PRIORS authority unavailable"
        )

    if pos not in priors:
        raise RuntimeError(
            f"FAIL | no hierarchical prior configured for {pos}"
        )

    prior = builder(
        pos
    )

    if not isinstance(
        prior,
        dict,
    ):
        raise RuntimeError(
            f"FAIL | invalid hierarchical prior for {pos}"
        )

    required = {
        "_first_n",
        "_all_n",
        "_first_weight",
    }

    missing = (
        required
        - set(
            prior.keys()
        )
    )

    if missing:
        raise RuntimeError(
            "FAIL | hierarchical prior missing metadata: "
            f"{sorted(missing)}"
        )

    return dict(
        prior
    )


# ============================================================
# BUILD FULL-TEAM SHADOW
# ============================================================

all_output = []
team_audits = []

for _, rookie in shadow.iterrows():

    game_id = str(rookie["game_id"])
    team_name = str(rookie["team"]).upper()
    rookie_pid = str(rookie["player_id"])
    rookie_pos = str(rookie["position"]).upper()

    prior = get_prior_for_shadow_row(rookie)

    target_pool = n(rookie["team_target_pool"])
    rush_pool = n(rookie["team_rush_attempt_pool"])

    rookie_targets = min(
        max(0.0, n(rookie["rookie_desired_targets"])),
        target_pool,
    )

    rookie_carries = min(
        max(0.0, n(rookie["rookie_desired_carries"])),
        rush_pool,
    )

    prior_catch = safe_div(
        prior.get("receptions"),
        prior.get("targets"),
    )

    prior_ypr = safe_div(
        prior.get("receiving_yards"),
        prior.get("receptions"),
    )

    prior_rec_td_rate = safe_div(
        prior.get("receiving_tds"),
        prior.get("targets"),
    )

    prior_ypc = safe_div(
        prior.get("rushing_yards"),
        prior.get("carries"),
    )

    prior_rush_td_rate = safe_div(
        prior.get("rushing_tds"),
        prior.get("carries"),
    )

    rookie_receptions = rookie_targets * prior_catch
    rookie_rec_yards = rookie_receptions * prior_ypr
    rookie_rec_tds = rookie_targets * prior_rec_td_rate

    rookie_rush_yards = rookie_carries * prior_ypc
    rookie_rush_tds = rookie_carries * prior_rush_td_rate

    team_rows = canon[
        canon["game_id"].astype(str).eq(game_id)
        &
        canon["team"].astype(str).str.upper().eq(team_name)
        &
        canon["position"].astype(str).str.upper().isin(OFFENSE_POSITIONS)
    ].copy()

    if team_rows.empty:
        raise RuntimeError(
            f"FAIL | no canonical offense for {game_id} {team_name}"
        )

    # --------------------------------------------------------
    # TARGET CONSERVATION
    # --------------------------------------------------------

    target_mask = (
        team_rows["position"]
        .astype(str)
        .str.upper()
        .isin({"RB", "FB", "WR", "TE"})
    )

    incumbent_targets_before = float(
        pd.to_numeric(
            team_rows.loc[target_mask, "expected_targets"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    remaining_targets = max(
        0.0,
        target_pool - rookie_targets,
    )

    target_scale = (
        remaining_targets / incumbent_targets_before
        if incumbent_targets_before > 0
        else 0.0
    )

    # --------------------------------------------------------
    # RUSH CONSERVATION
    # --------------------------------------------------------

    rush_mask = (
        team_rows["position"]
        .astype(str)
        .str.upper()
        .isin(OFFENSE_POSITIONS)
    )

    incumbent_carries_before = float(
        pd.to_numeric(
            team_rows.loc[rush_mask, "expected_carries"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    remaining_carries = max(
        0.0,
        rush_pool - rookie_carries,
    )

    rush_scale = (
        remaining_carries / incumbent_carries_before
        if incumbent_carries_before > 0
        else 0.0
    )

    # --------------------------------------------------------
    # ADJUST INCUMBENTS
    # --------------------------------------------------------

    adjusted_rows = []

    for _, base in team_rows.iterrows():

        out = base.to_dict()

        pos = str(
            out.get("position", "")
        ).upper()

        old_targets = n(
            out.get("expected_targets")
        )

        old_receptions = n(
            out.get("expected_receptions")
        )

        old_rec_yards = n(
            out.get("expected_receiving_yards")
        )

        old_rec_tds = n(
            out.get("expected_receiving_tds")
        )

        old_carries = n(
            out.get("expected_carries")
        )

        old_rush_yards = n(
            out.get("expected_rushing_yards")
        )

        old_rush_tds = n(
            out.get("expected_rushing_tds")
        )

        if pos in {"RB", "FB", "WR", "TE"}:
            new_targets = old_targets * target_scale

            out["expected_targets"] = new_targets

            if old_targets > 0:
                ratio = new_targets / old_targets
                out["expected_receptions"] = old_receptions * ratio
                out["expected_receiving_yards"] = old_rec_yards * ratio
                out["expected_receiving_tds"] = old_rec_tds * ratio

        if pos in OFFENSE_POSITIONS:
            new_carries = old_carries * rush_scale

            out["expected_carries"] = new_carries

            if old_carries > 0:
                ratio = new_carries / old_carries
                out["expected_rushing_yards"] = old_rush_yards * ratio
                out["expected_rushing_tds"] = old_rush_tds * ratio

        out["shadow_row_type"] = "ADJUSTED_INCUMBENT"
        out["cold_start_driver_player_id"] = rookie_pid
        out["cold_start_driver_name"] = rookie["entity_name"]
        out["cold_start_target_scale"] = target_scale
        out["cold_start_rush_scale"] = rush_scale
        out["cold_start_efficiency_policy"] = "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP"

        out["expected_fd_points"] = fd_points(out)

        adjusted_rows.append(out)

    # --------------------------------------------------------
    # ADD ROOKIE
    # --------------------------------------------------------

    rookie_out = {
        col: np.nan
        for col in canon.columns
    }

    rookie_out.update({
        "entity_type": "player",
        "game_id": game_id,
        "season": int(n(rookie["season"])),
        "team": team_name,
        "opponent_team": rookie["opponent_team"],
        "player_id": rookie_pid,
        "entity_name": rookie["entity_name"],
        "position": rookie_pos,
        "model_group": "COLD_START_SHADOW_V2",
        "expected_completions": 0.0,
        "expected_attempts": 0.0,
        "expected_passing_yards": 0.0,
        "expected_passing_tds": 0.0,
        "expected_interceptions": 0.0,
        "expected_carries": rookie_carries,
        "expected_targets": rookie_targets,
        "expected_receptions": rookie_receptions,
        "expected_rushing_yards": rookie_rush_yards,
        "expected_receiving_yards": rookie_rec_yards,
        "expected_rushing_tds": rookie_rush_tds,
        "expected_receiving_tds": rookie_rec_tds,
    })

    rookie_out["shadow_row_type"] = "COLD_START_ROOKIE"
    rookie_out["cold_start_driver_player_id"] = rookie_pid
    rookie_out["cold_start_driver_name"] = rookie["entity_name"]
    rookie_out["cold_start_target_scale"] = target_scale
    rookie_out["cold_start_rush_scale"] = rush_scale
    rookie_out["cold_start_efficiency_policy"] = "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP"
    rookie_out["expected_fd_points"] = fd_points(rookie_out)

    adjusted_rows.append(rookie_out)

    adjusted = pd.DataFrame(adjusted_rows)

    skill_mask = (
        adjusted["position"]
        .astype(str)
        .str.upper()
        .isin({"RB", "FB", "WR", "TE"})
    )

    offense_mask = (
        adjusted["position"]
        .astype(str)
        .str.upper()
        .isin(OFFENSE_POSITIONS)
    )

    targets_after = float(
        pd.to_numeric(
            adjusted.loc[skill_mask, "expected_targets"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    carries_after = float(
        pd.to_numeric(
            adjusted.loc[offense_mask, "expected_carries"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    target_error = targets_after - target_pool
    rush_error = carries_after - rush_pool

    if abs(target_error) > 1e-6:
        raise RuntimeError(
            f"FAIL | target conservation {game_id} {team_name}: "
            f"{target_error}"
        )

    if abs(rush_error) > 1e-6:
        raise RuntimeError(
            f"FAIL | rush conservation {game_id} {team_name}: "
            f"{rush_error}"
        )

    team_audits.append({
        "game_id": game_id,
        "team": team_name,
        "rookie_player_id": rookie_pid,
        "rookie_name": rookie["entity_name"],
        "rookie_position": rookie_pos,
        "target_pool": target_pool,
        "targets_after": targets_after,
        "target_conservation_error": target_error,
        "rush_pool": rush_pool,
        "carries_after": carries_after,
        "rush_conservation_error": rush_error,
        "incumbent_target_scale": target_scale,
        "incumbent_rush_scale": rush_scale,
        "rookie_targets": rookie_targets,
        "rookie_carries": rookie_carries,
        "rookie_receptions": rookie_receptions,
        "rookie_receiving_yards": rookie_rec_yards,
        "rookie_receiving_tds": rookie_rec_tds,
        "rookie_rushing_yards": rookie_rush_yards,
        "rookie_rushing_tds": rookie_rush_tds,
        "rookie_fd_points": fd_points(rookie_out),
    })

    all_output.append(adjusted)


full_shadow = pd.concat(
    all_output,
    ignore_index=True,
    sort=False,
)


# ============================================================
# DUPLICATE / IDENTITY GATES
# ============================================================

dup = full_shadow.duplicated(
    ["game_id", "team", "player_id"],
    keep=False,
)

if dup.any():
    bad = full_shadow.loc[
        dup,
        ["game_id", "team", "player_id", "entity_name"]
    ]
    raise RuntimeError(
        "FAIL | duplicate exact identities in full-team shadow\n"
        + bad.to_string(index=False)
    )

rookie_rows = full_shadow[
    full_shadow["shadow_row_type"].eq("COLD_START_ROOKIE")
]

if len(rookie_rows) != 2:
    raise RuntimeError(
        f"FAIL | expected 2 rookie rows, got {len(rookie_rows)}"
    )

if set(rookie_rows["player_id"].astype(str)) != {
    "00-0041027",
    "00-0041438",
}:
    raise RuntimeError(
        "FAIL | exact Love/Tate identity gate"
    )


# ============================================================
# WRITE SHADOW ONLY
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

full_shadow.to_parquet(
    OUT,
    index=False,
)


canonical_sha_after = sha256(CANONICAL)

if canonical_sha_after != canonical_sha_before:
    raise RuntimeError(
        "FAIL | canonical forecast mutated"
    )


audit = {
    "stage": "STAGE26G-R8V",
    "model": "COLD_START_FULL_TEAM_SHADOW_V2",
    "rookie_efficiency_policy": "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP",
    "r8u_decision": "RB_MATCHUP_SIGNAL_FAILS_LOO_KEEP_ZERO_DVP",
    "wr_dvp_policy": "ZERO_DVP_INSUFFICIENT_EVIDENCE",
    "te_dvp_policy": "ZERO_DVP_INSUFFICIENT_EVIDENCE",
    "full_team_rows": int(len(full_shadow)),
    "rookie_rows": int(len(rookie_rows)),
    "team_audits": team_audits,
    "canonical_sha_before": canonical_sha_before,
    "canonical_sha_after": canonical_sha_after,
    "canonical_unchanged": canonical_sha_before == canonical_sha_after,
    "production_promotion": False,
    "publisher_mutation": False,
    "analyst_mutation": False,
    "app_mutation": False,
    "solver_mutation": False,
    "team_environment_mutation": False,
}

AUDIT.write_text(
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

audit_df = pd.DataFrame(team_audits)

print()
print("============================================================")
print("R8V ROOKIE FORECASTS — ZERO DVP")
print("============================================================")

cols = [
    "game_id",
    "team",
    "entity_name",
    "position",
    "expected_carries",
    "expected_targets",
    "expected_receptions",
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_rushing_tds",
    "expected_receiving_tds",
    "expected_fd_points",
]

print(
    rookie_rows[cols].to_string(index=False)
)

print()
print("============================================================")
print("R8V TEAM CONSERVATION")
print("============================================================")

print(
    audit_df[
        [
            "game_id",
            "team",
            "target_pool",
            "targets_after",
            "target_conservation_error",
            "rush_pool",
            "carries_after",
            "rush_conservation_error",
            "incumbent_target_scale",
            "incumbent_rush_scale",
        ]
    ].to_string(index=False)
)

print()
print("============================================================")
print("R8V FULL TEAM SHADOW")
print("============================================================")

show = [
    "game_id",
    "team",
    "player_id",
    "entity_name",
    "position",
    "shadow_row_type",
    "expected_carries",
    "expected_targets",
    "expected_receptions",
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_rushing_tds",
    "expected_receiving_tds",
    "expected_fd_points",
]

print(
    full_shadow[
        [c for c in show if c in full_shadow.columns]
    ]
    .sort_values(
        ["game_id", "team", "position", "entity_name"]
    )
    .to_string(index=False)
)

print()
print("============================================================")
print("SAFETY / MUTATION AUDIT")
print("============================================================")

print(f"CANONICAL_SHA_BEFORE={canonical_sha_before}")
print(f"CANONICAL_SHA_AFTER={canonical_sha_after}")
print(f"CANONICAL_UNCHANGED={canonical_sha_before == canonical_sha_after}")
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
print(f"SHADOW={OUT}")
print(f"AUDIT={AUDIT}")

print()
print("============================================================")
print("STAGE26G_R8V_FULL_TEAM_SHADOW_V2=PASS")
print("============================================================")
