from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

R8V_PATH = (
    ROOT
    / "data/parquet/current_cold_start_full_team_shadow_v2.parquet"
)

R8V_AUDIT = (
    ROOT
    / "data/parquet/current_cold_start_full_team_shadow_v2_audit.json"
)

CANONICAL = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

OUT = (
    ROOT
    / "data/parquet/current_cold_start_position_aware_shadow_v1.parquet"
)

AUDIT = (
    ROOT
    / "data/parquet/current_cold_start_position_aware_shadow_v1_audit.json"
)

EXPECTED_CANONICAL_SHA = (
    "6067d8ea4a31ae60c5667c647c13c6bede4667c1369724c7861cf32929af13b9"
)

EXPECTED_R8V_SHA = (
    "91c980aba4c7d2352f4c6c6091d7f6b56ccb9d06c6a406d7f921a30856cc6ed1"
)

OFFENSE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}
TARGET_POSITIONS = {"RB", "FB", "WR", "TE"}
NON_QB_RUSH_POSITIONS = {"RB", "FB", "WR", "TE"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def num(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default

    if not np.isfinite(value):
        return default

    return value


def fd_points(row) -> float:
    return (
        0.04 * num(row.get("expected_passing_yards"))
        + 4.0 * num(row.get("expected_passing_tds"))
        - 1.0 * num(row.get("expected_interceptions"))
        + 0.10 * num(row.get("expected_rushing_yards"))
        + 6.0 * num(row.get("expected_rushing_tds"))
        + 0.50 * num(row.get("expected_receptions"))
        + 0.10 * num(row.get("expected_receiving_yards"))
        + 6.0 * num(row.get("expected_receiving_tds"))
    )


# ============================================================
# INPUT / SHA GATES
# ============================================================

for path in [
    R8V_PATH,
    R8V_AUDIT,
    CANONICAL,
]:
    if not path.exists():
        raise RuntimeError(
            f"FAIL | missing required artifact: {path}"
        )

canonical_sha_before = sha256(CANONICAL)
r8v_sha = sha256(R8V_PATH)

if canonical_sha_before != EXPECTED_CANONICAL_SHA:
    raise RuntimeError(
        "FAIL | canonical SHA mismatch\n"
        f"EXPECTED={EXPECTED_CANONICAL_SHA}\n"
        f"ACTUAL={canonical_sha_before}"
    )

if r8v_sha != EXPECTED_R8V_SHA:
    raise RuntimeError(
        "FAIL | R8V shadow SHA mismatch\n"
        f"EXPECTED={EXPECTED_R8V_SHA}\n"
        f"ACTUAL={r8v_sha}"
    )


# ============================================================
# LOAD AUTHORITIES
# ============================================================

r8v = pd.read_parquet(R8V_PATH)
canon = pd.read_parquet(CANONICAL)

rookies = r8v[
    r8v["shadow_row_type"]
    .astype(str)
    .eq("COLD_START_ROOKIE")
].copy()

if len(rookies) != 2:
    raise RuntimeError(
        f"FAIL | expected 2 R8V rookie rows, got {len(rookies)}"
    )

expected_rookies = {
    "00-0041027",
    "00-0041438",
}

if set(
    rookies["player_id"].astype(str)
) != expected_rookies:
    raise RuntimeError(
        "FAIL | exact Love/Tate identity gate"
    )


# ============================================================
# REBUILD EACH AFFECTED TEAM FROM CANONICAL
#
# R8V rookie row = cold-start rookie stat authority.
# Canonical = incumbent stat authority.
#
# We intentionally do NOT use R8V-adjusted incumbents because
# R8W is testing corrected redistribution semantics.
# ============================================================

outputs = []
team_audits = []

for _, rookie in rookies.iterrows():

    game_id = str(
        rookie["game_id"]
    )

    team = str(
        rookie["team"]
    ).upper()

    rookie_pid = str(
        rookie["player_id"]
    )

    rookie_name = str(
        rookie["entity_name"]
    )

    rookie_pos = str(
        rookie["position"]
    ).upper()

    rookie_targets = num(
        rookie["expected_targets"]
    )

    rookie_carries = num(
        rookie["expected_carries"]
    )

    # R8V contains the independent team pools as cold-start
    # metadata on every generated row.
    #
    # Recover them from the R8V audit because the full-team
    # parquet intentionally mirrors canonical stat columns.

    with R8V_AUDIT.open() as f:
        r8v_audit = json.load(f)

    matching_audits = [
        x
        for x in r8v_audit["team_audits"]
        if str(x["game_id"]) == game_id
        and str(x["team"]).upper() == team
        and str(x["rookie_player_id"]) == rookie_pid
    ]

    if len(matching_audits) != 1:
        raise RuntimeError(
            "FAIL | expected exactly one R8V team audit for "
            f"{game_id} {team} {rookie_pid}"
        )

    source_audit = matching_audits[0]

    target_pool = num(
        source_audit["target_pool"]
    )

    rush_pool = num(
        source_audit["rush_pool"]
    )

    team_rows = canon[
        canon["game_id"]
        .astype(str)
        .eq(game_id)
        &
        canon["team"]
        .astype(str)
        .str.upper()
        .eq(team)
        &
        canon["position"]
        .astype(str)
        .str.upper()
        .isin(OFFENSE_POSITIONS)
    ].copy()

    if team_rows.empty:
        raise RuntimeError(
            f"FAIL | no canonical offense for {game_id} {team}"
        )

    # ========================================================
    # TARGET POLICY
    #
    # Preserve canonical incumbent targets unless:
    #
    # canonical incumbent targets + rookie targets > pool.
    #
    # Only actual rookie displacement causes downward scaling.
    # Never inflate incumbents merely to fill a team-pool gap.
    # ========================================================

    target_mask = (
        team_rows["position"]
        .astype(str)
        .str.upper()
        .isin(TARGET_POSITIONS)
    )

    incumbent_targets_before = float(
        pd.to_numeric(
            team_rows.loc[
                target_mask,
                "expected_targets",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    target_total_before_reconcile = (
        incumbent_targets_before
        + rookie_targets
    )

    if (
        incumbent_targets_before > 0
        and target_total_before_reconcile > target_pool
    ):
        incumbent_target_scale = (
            max(
                0.0,
                target_pool - rookie_targets,
            )
            / incumbent_targets_before
        )
        target_policy = "ROOKIE_DISPLACEMENT_SCALE_DOWN"

    else:
        incumbent_target_scale = 1.0
        target_policy = "PRESERVE_CANONICAL_INCUMBENTS"

    incumbent_targets_after = (
        incumbent_targets_before
        * incumbent_target_scale
    )

    target_total_after = (
        incumbent_targets_after
        + rookie_targets
    )

    target_pool_gap_after = (
        target_pool
        - target_total_after
    )

    # This is intentionally NOT required to equal zero.
    #
    # Positive gap means canonical was below the independent
    # team pool and R8W refused to manufacture extra targets
    # for incumbents merely to force conservation.


    # ========================================================
    # RUSH POLICY
    #
    # 1. Preserve all canonical QB carries.
    # 2. Reserve rookie carries.
    # 3. Remaining rush pool belongs to incumbent non-QBs.
    # 4. Scale RB/FB/WR/TE incumbent carries only.
    # ========================================================

    qb_mask = (
        team_rows["position"]
        .astype(str)
        .str.upper()
        .eq("QB")
    )

    non_qb_rush_mask = (
        team_rows["position"]
        .astype(str)
        .str.upper()
        .isin(NON_QB_RUSH_POSITIONS)
    )

    qb_carries_reserved = float(
        pd.to_numeric(
            team_rows.loc[
                qb_mask,
                "expected_carries",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    incumbent_non_qb_carries_before = float(
        pd.to_numeric(
            team_rows.loc[
                non_qb_rush_mask,
                "expected_carries",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    remaining_non_qb_rush_pool = (
        rush_pool
        - qb_carries_reserved
        - rookie_carries
    )

    if remaining_non_qb_rush_pool < -1e-9:
        raise RuntimeError(
            "FAIL | QB reserved carries + rookie carries exceed "
            f"team rush pool for {game_id} {team}: "
            f"pool={rush_pool:.6f} "
            f"qb={qb_carries_reserved:.6f} "
            f"rookie={rookie_carries:.6f}"
        )

    remaining_non_qb_rush_pool = max(
        0.0,
        remaining_non_qb_rush_pool,
    )

    if incumbent_non_qb_carries_before > 0:
        incumbent_non_qb_rush_scale = (
            remaining_non_qb_rush_pool
            / incumbent_non_qb_carries_before
        )
    else:
        incumbent_non_qb_rush_scale = 0.0

    # ========================================================
    # APPLY INCUMBENT ADJUSTMENTS
    # ========================================================

    adjusted = []

    for _, base in team_rows.iterrows():

        out = base.to_dict()

        pos = str(
            out.get("position", "")
        ).upper()

        old_targets = num(
            out.get("expected_targets")
        )

        old_receptions = num(
            out.get("expected_receptions")
        )

        old_rec_yards = num(
            out.get("expected_receiving_yards")
        )

        old_rec_tds = num(
            out.get("expected_receiving_tds")
        )

        old_carries = num(
            out.get("expected_carries")
        )

        old_rush_yards = num(
            out.get("expected_rushing_yards")
        )

        old_rush_tds = num(
            out.get("expected_rushing_tds")
        )

        # Receiving opportunity adjustment.
        if pos in TARGET_POSITIONS:

            new_targets = (
                old_targets
                * incumbent_target_scale
            )

            out["expected_targets"] = new_targets

            if old_targets > 0:
                receiving_ratio = (
                    new_targets / old_targets
                )

                out["expected_receptions"] = (
                    old_receptions
                    * receiving_ratio
                )

                out["expected_receiving_yards"] = (
                    old_rec_yards
                    * receiving_ratio
                )

                out["expected_receiving_tds"] = (
                    old_rec_tds
                    * receiving_ratio
                )

        # QB rushing is explicitly preserved.
        if pos == "QB":
            out["expected_carries"] = old_carries
            out["expected_rushing_yards"] = old_rush_yards
            out["expected_rushing_tds"] = old_rush_tds

        # Only non-QB offensive rushers absorb redistribution.
        elif pos in NON_QB_RUSH_POSITIONS:

            new_carries = (
                old_carries
                * incumbent_non_qb_rush_scale
            )

            out["expected_carries"] = new_carries

            if old_carries > 0:
                rushing_ratio = (
                    new_carries / old_carries
                )

                out["expected_rushing_yards"] = (
                    old_rush_yards
                    * rushing_ratio
                )

                out["expected_rushing_tds"] = (
                    old_rush_tds
                    * rushing_ratio
                )

        out["shadow_row_type"] = (
            "ADJUSTED_INCUMBENT"
        )

        out["cold_start_driver_player_id"] = (
            rookie_pid
        )

        out["cold_start_driver_name"] = (
            rookie_name
        )

        out["cold_start_target_policy"] = (
            target_policy
        )

        out["cold_start_target_scale"] = (
            incumbent_target_scale
        )

        out["cold_start_qb_carries_reserved"] = (
            qb_carries_reserved
        )

        out["cold_start_non_qb_rush_scale"] = (
            incumbent_non_qb_rush_scale
        )

        out["cold_start_efficiency_policy"] = (
            "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP"
        )

        out["expected_fd_points"] = (
            fd_points(out)
        )

        adjusted.append(out)

    # ========================================================
    # ADD R8V ROOKIE UNCHANGED
    # ========================================================

    rookie_out = {
        col: np.nan
        for col in canon.columns
    }

    for col in canon.columns:
        if col in rookie.index:
            rookie_out[col] = rookie[col]

    rookie_out.update({
        "entity_type": "player",
        "game_id": game_id,
        "team": team,
        "opponent_team": rookie["opponent_team"],
        "player_id": rookie_pid,
        "entity_name": rookie_name,
        "position": rookie_pos,
        "model_group": "COLD_START_SHADOW_V3",
        "expected_carries": rookie_carries,
        "expected_targets": rookie_targets,
        "expected_receptions": num(
            rookie["expected_receptions"]
        ),
        "expected_rushing_yards": num(
            rookie["expected_rushing_yards"]
        ),
        "expected_receiving_yards": num(
            rookie["expected_receiving_yards"]
        ),
        "expected_rushing_tds": num(
            rookie["expected_rushing_tds"]
        ),
        "expected_receiving_tds": num(
            rookie["expected_receiving_tds"]
        ),
    })

    rookie_out["shadow_row_type"] = (
        "COLD_START_ROOKIE"
    )

    rookie_out["cold_start_driver_player_id"] = (
        rookie_pid
    )

    rookie_out["cold_start_driver_name"] = (
        rookie_name
    )

    rookie_out["cold_start_target_policy"] = (
        target_policy
    )

    rookie_out["cold_start_target_scale"] = (
        incumbent_target_scale
    )

    rookie_out["cold_start_qb_carries_reserved"] = (
        qb_carries_reserved
    )

    rookie_out["cold_start_non_qb_rush_scale"] = (
        incumbent_non_qb_rush_scale
    )

    rookie_out["cold_start_efficiency_policy"] = (
        "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP"
    )

    rookie_out["expected_fd_points"] = (
        fd_points(rookie_out)
    )

    adjusted.append(
        rookie_out
    )

    team_shadow = pd.DataFrame(
        adjusted
    )

    # ========================================================
    # POST-ADJUSTMENT AUDITS
    # ========================================================

    qb_after = float(
        pd.to_numeric(
            team_shadow.loc[
                team_shadow["position"]
                .astype(str)
                .str.upper()
                .eq("QB"),
                "expected_carries",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    non_qb_after = float(
        pd.to_numeric(
            team_shadow.loc[
                team_shadow["position"]
                .astype(str)
                .str.upper()
                .isin(NON_QB_RUSH_POSITIONS),
                "expected_carries",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    total_rush_after = (
        qb_after
        + non_qb_after
    )

    rush_error = (
        total_rush_after
        - rush_pool
    )

    if abs(
        qb_after - qb_carries_reserved
    ) > 1e-9:
        raise RuntimeError(
            f"FAIL | QB rushing changed for {game_id} {team}"
        )

    if abs(rush_error) > 1e-6:
        raise RuntimeError(
            "FAIL | position-aware rush conservation "
            f"{game_id} {team}: {rush_error}"
        )

    target_after_check = float(
        pd.to_numeric(
            team_shadow.loc[
                team_shadow["position"]
                .astype(str)
                .str.upper()
                .isin(TARGET_POSITIONS),
                "expected_targets",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    if abs(
        target_after_check
        - target_total_after
    ) > 1e-6:
        raise RuntimeError(
            f"FAIL | target accounting mismatch {game_id} {team}"
        )

    if target_after_check > target_pool + 1e-6:
        raise RuntimeError(
            f"FAIL | targets exceed pool {game_id} {team}"
        )

    team_audits.append({
        "game_id": game_id,
        "team": team,
        "rookie_player_id": rookie_pid,
        "rookie_name": rookie_name,
        "rookie_position": rookie_pos,

        "target_pool": target_pool,
        "incumbent_targets_before": incumbent_targets_before,
        "rookie_targets": rookie_targets,
        "target_total_before_reconcile": target_total_before_reconcile,
        "target_policy": target_policy,
        "incumbent_target_scale": incumbent_target_scale,
        "target_total_after": target_after_check,
        "target_pool_gap_after": (
            target_pool
            - target_after_check
        ),

        "rush_pool": rush_pool,
        "qb_carries_reserved": qb_carries_reserved,
        "rookie_carries": rookie_carries,
        "incumbent_non_qb_carries_before": incumbent_non_qb_carries_before,
        "remaining_non_qb_rush_pool": remaining_non_qb_rush_pool,
        "incumbent_non_qb_rush_scale": incumbent_non_qb_rush_scale,
        "qb_carries_after": qb_after,
        "total_rush_after": total_rush_after,
        "rush_conservation_error": rush_error,

        "rookie_fd_points": num(
            rookie_out["expected_fd_points"]
        ),
    })

    outputs.append(
        team_shadow
    )


# ============================================================
# COMBINE / EXACT IDENTITY GATES
# ============================================================

full_shadow = pd.concat(
    outputs,
    ignore_index=True,
    sort=False,
)

duplicate_mask = full_shadow.duplicated(
    [
        "game_id",
        "team",
        "player_id",
    ],
    keep=False,
)

if duplicate_mask.any():

    bad = full_shadow.loc[
        duplicate_mask,
        [
            "game_id",
            "team",
            "player_id",
            "entity_name",
        ],
    ]

    raise RuntimeError(
        "FAIL | duplicate exact identities\n"
        + bad.to_string(index=False)
    )


# ============================================================
# ROOKIE FORECAST IMMUTABILITY GATE
# ============================================================

for _, source in rookies.iterrows():

    match = full_shadow[
        full_shadow["game_id"]
        .astype(str)
        .eq(str(source["game_id"]))
        &
        full_shadow["team"]
        .astype(str)
        .str.upper()
        .eq(str(source["team"]).upper())
        &
        full_shadow["player_id"]
        .astype(str)
        .eq(str(source["player_id"]))
    ]

    if len(match) != 1:
        raise RuntimeError(
            "FAIL | rookie identity lost in R8W"
        )

    target = match.iloc[0]

    compare_cols = [
        "expected_carries",
        "expected_targets",
        "expected_receptions",
        "expected_rushing_yards",
        "expected_receiving_yards",
        "expected_rushing_tds",
        "expected_receiving_tds",
        "expected_fd_points",
    ]

    for col in compare_cols:

        a = num(
            source.get(col)
        )

        b = num(
            target.get(col)
        )

        if abs(a - b) > 1e-9:
            raise RuntimeError(
                "FAIL | rookie forecast changed "
                f"{source['entity_name']} {col}: "
                f"{a} -> {b}"
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

canonical_sha_after = sha256(
    CANONICAL
)

if canonical_sha_after != canonical_sha_before:
    raise RuntimeError(
        "FAIL | canonical forecast mutated"
    )


audit = {
    "stage": "STAGE26G-R8W",
    "model": "COLD_START_POSITION_AWARE_SHADOW_V1",

    "rookie_efficiency_policy":
        "HIERARCHICAL_ROOKIE_PRIOR_ONLY_ZERO_DVP",

    "target_policy":
        "DISPLACEMENT_ONLY_NO_UPWARD_INCUMBENT_SCALING",

    "rush_policy":
        "PRESERVE_QB_THEN_REDISTRIBUTE_NON_QB",

    "rows": int(
        len(full_shadow)
    ),

    "rookie_rows": 2,

    "team_audits":
        team_audits,

    "r8v_source_sha":
        r8v_sha,

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

audit_df = pd.DataFrame(
    team_audits
)

rookie_result = full_shadow[
    full_shadow["shadow_row_type"]
    .eq("COLD_START_ROOKIE")
].copy()


print()
print("============================================================")
print("R8W ROOKIE FORECAST IMMUTABILITY")
print("============================================================")

print(
    rookie_result[
        [
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
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8W TARGET RECONCILIATION")
print("============================================================")

print(
    audit_df[
        [
            "game_id",
            "team",
            "target_pool",
            "incumbent_targets_before",
            "rookie_targets",
            "target_total_before_reconcile",
            "target_policy",
            "incumbent_target_scale",
            "target_total_after",
            "target_pool_gap_after",
        ]
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8W POSITION-AWARE RUSH RECONCILIATION")
print("============================================================")

print(
    audit_df[
        [
            "game_id",
            "team",
            "rush_pool",
            "qb_carries_reserved",
            "rookie_carries",
            "incumbent_non_qb_carries_before",
            "remaining_non_qb_rush_pool",
            "incumbent_non_qb_rush_scale",
            "qb_carries_after",
            "total_rush_after",
            "rush_conservation_error",
        ]
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8W FULL TEAM SHADOW")
print("============================================================")

show_cols = [
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
        show_cols
    ]
    .sort_values(
        [
            "game_id",
            "team",
            "position",
            "entity_name",
        ]
    )
    .to_string(
        index=False
    )
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
    "ROOKIE_FORECASTS_UNCHANGED=True"
)

print(
    "QB_RUSHING_PRESERVED=True"
)

print(
    "TARGET_UPWARD_SCALING_ALLOWED=False"
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
    f"SHADOW={OUT}"
)

print(
    f"AUDIT={AUDIT}"
)


print()
print("============================================================")
print("STAGE26G_R8W_POSITION_AWARE_SHADOW=PASS")
print("============================================================")
