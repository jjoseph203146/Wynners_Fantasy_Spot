from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

R8Y_DETAIL = (
    ROOT
    / "processed/stage26gr8y_role_share_validation_detail.csv"
)

R8X_CURRENT = (
    ROOT
    / "processed/stage26gr8x_current_role_share_shadow.csv"
)

CANONICAL = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

DETAIL_OUT = (
    ROOT
    / "processed/stage26gr8z_feasibility_cap_detail.csv"
)

SUMMARY_OUT = (
    ROOT
    / "processed/stage26gr8z_feasibility_cap_summary.csv"
)

CURRENT_OUT = (
    ROOT
    / "processed/stage26gr8z_current_feasibility_cap.csv"
)

AUDIT_OUT = (
    ROOT
    / "processed/stage26gr8z_feasibility_cap_audit.json"
)


EXPECTED_R8Y_DETAIL_SHA = (
    "619754b12ba5fad01b7a2ad1e9df9164"
    "df2e3133004e7ed3f7b7fff6758626b4"
)

EXPECTED_R8X_CURRENT_SHA = (
    "927305e0d2b65ecfde39060a07e1a945"
    "6559589da24967e514f546f2999e4065"
)

EXPECTED_CANONICAL_SHA = (
    "6067d8ea4a31ae60c5667c647c13c6be"
    "de4667c1369724c7861cf32929af13b9"
)


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
# SAFETY GATES
# ============================================================

for path in [
    R8Y_DETAIL,
    R8X_CURRENT,
    CANONICAL,
]:
    if not path.exists():
        raise RuntimeError(
            f"FAIL | missing required input: {path}"
        )


actual_r8y_sha = sha256(
    R8Y_DETAIL
)

if actual_r8y_sha != EXPECTED_R8Y_DETAIL_SHA:
    raise RuntimeError(
        "FAIL | R8Y detail SHA mismatch\n"
        f"EXPECTED={EXPECTED_R8Y_DETAIL_SHA}\n"
        f"ACTUAL={actual_r8y_sha}"
    )


actual_r8x_sha = sha256(
    R8X_CURRENT
)

if actual_r8x_sha != EXPECTED_R8X_CURRENT_SHA:
    raise RuntimeError(
        "FAIL | R8X current SHA mismatch\n"
        f"EXPECTED={EXPECTED_R8X_CURRENT_SHA}\n"
        f"ACTUAL={actual_r8x_sha}"
    )


canonical_sha_before = sha256(
    CANONICAL
)

if canonical_sha_before != EXPECTED_CANONICAL_SHA:
    raise RuntimeError(
        "FAIL | canonical SHA mismatch\n"
        f"EXPECTED={EXPECTED_CANONICAL_SHA}\n"
        f"ACTUAL={canonical_sha_before}"
    )


# ============================================================
# LOAD VALIDATED R8Y EVENTS
# ============================================================

detail = pd.read_csv(
    R8Y_DETAIL
)

required = {
    "game_id",
    "player_id",
    "player_name",
    "position",
    "actual_carries",
    "absolute_pred_carries",
    "projected_non_qb_pool",
}

missing = (
    required
    - set(detail.columns)
)

if missing:
    raise RuntimeError(
        "FAIL | missing R8Y columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# RB CARRY FEASIBILITY CANDIDATE
#
# Preserve the validated absolute opportunity forecast unless
# it exceeds the projected non-QB team rushing opportunity.
#
# No role-share predictive replacement.
# No percentile tuning.
# No fitted hyperparameter.
# ============================================================

rb = detail[
    detail["position"]
    .astype(str)
    .str.upper()
    .eq("RB")
].copy()

if rb.empty:
    raise RuntimeError(
        "FAIL | zero RB validation events"
    )


for c in [
    "actual_carries",
    "absolute_pred_carries",
    "projected_non_qb_pool",
]:
    rb[c] = pd.to_numeric(
        rb[c],
        errors="coerce",
    )


rb = rb[
    rb[
        [
            "actual_carries",
            "absolute_pred_carries",
            "projected_non_qb_pool",
        ]
    ]
    .notna()
    .all(axis=1)
].copy()

if rb.empty:
    raise RuntimeError(
        "FAIL | zero usable RB validation events"
    )


if (
    rb[
        "projected_non_qb_pool"
    ] < 0
).any():
    raise RuntimeError(
        "FAIL | negative projected non-QB pool"
    )


rb[
    "hybrid_pred_carries"
] = np.minimum(
    rb[
        "absolute_pred_carries"
    ],
    rb[
        "projected_non_qb_pool"
    ],
)


rb[
    "absolute_structurally_feasible"
] = (
    rb[
        "absolute_pred_carries"
    ]
    <=
    rb[
        "projected_non_qb_pool"
    ]
    + 1e-12
)


rb[
    "hybrid_structurally_feasible"
] = (
    rb[
        "hybrid_pred_carries"
    ]
    <=
    rb[
        "projected_non_qb_pool"
    ]
    + 1e-12
)


rb[
    "cap_applied"
] = (
    rb[
        "hybrid_pred_carries"
    ]
    <
    rb[
        "absolute_pred_carries"
    ]
    - 1e-12
)


rb[
    "cap_amount"
] = (
    rb[
        "absolute_pred_carries"
    ]
    -
    rb[
        "hybrid_pred_carries"
    ]
)


if not rb[
    "hybrid_structurally_feasible"
].all():
    raise RuntimeError(
        "FAIL | hybrid feasibility constraint violated"
    )


# ============================================================
# VALIDATION METRICS
# ============================================================

absolute_metrics = metrics(
    rb[
        "actual_carries"
    ],
    rb[
        "absolute_pred_carries"
    ],
)

hybrid_metrics = metrics(
    rb[
        "actual_carries"
    ],
    rb[
        "hybrid_pred_carries"
    ],
)


wins = 0
ties = 0
losses = 0

for _, row in rb.iterrows():

    actual = num(
        row[
            "actual_carries"
        ]
    )

    absolute = num(
        row[
            "absolute_pred_carries"
        ]
    )

    hybrid = num(
        row[
            "hybrid_pred_carries"
        ]
    )

    absolute_error = abs(
        absolute - actual
    )

    hybrid_error = abs(
        hybrid - actual
    )

    if hybrid_error < absolute_error - 1e-12:
        wins += 1

    elif absolute_error < hybrid_error - 1e-12:
        losses += 1

    else:
        ties += 1


mae_non_degrade = bool(
    hybrid_metrics[
        "mae"
    ]
    <=
    absolute_metrics[
        "mae"
    ]
    + 1e-12
)

rmse_non_degrade = bool(
    hybrid_metrics[
        "rmse"
    ]
    <=
    absolute_metrics[
        "rmse"
    ]
    + 1e-12
)

all_feasible = bool(
    rb[
        "hybrid_structurally_feasible"
    ].all()
)


research_pass = bool(
    mae_non_degrade
    and rmse_non_degrade
    and all_feasible
)


summary = pd.DataFrame(
    [
        {
            "position":
                "RB",

            "metric":
                "carries",

            "n":
                absolute_metrics[
                    "n"
                ],

            "absolute_mae":
                absolute_metrics[
                    "mae"
                ],

            "hybrid_mae":
                hybrid_metrics[
                    "mae"
                ],

            "mae_delta_hybrid_minus_absolute":
                hybrid_metrics[
                    "mae"
                ]
                -
                absolute_metrics[
                    "mae"
                ],

            "absolute_rmse":
                absolute_metrics[
                    "rmse"
                ],

            "hybrid_rmse":
                hybrid_metrics[
                    "rmse"
                ],

            "rmse_delta_hybrid_minus_absolute":
                hybrid_metrics[
                    "rmse"
                ]
                -
                absolute_metrics[
                    "rmse"
                ],

            "absolute_bias":
                absolute_metrics[
                    "bias"
                ],

            "hybrid_bias":
                hybrid_metrics[
                    "bias"
                ],

            "historical_cap_events":
                int(
                    rb[
                        "cap_applied"
                    ].sum()
                ),

            "historical_infeasible_absolute_events":
                int(
                    (
                        ~rb[
                            "absolute_structurally_feasible"
                        ]
                    ).sum()
                ),

            "wins":
                wins,

            "ties":
                ties,

            "losses":
                losses,

            "mae_non_degrade":
                mae_non_degrade,

            "rmse_non_degrade":
                rmse_non_degrade,

            "all_hybrid_feasible":
                all_feasible,

            "research_pass":
                research_pass,
        }
    ]
)


# ============================================================
# CURRENT LOVE / TATE APPLICATION
# ============================================================

current = pd.read_csv(
    R8X_CURRENT
)

required_current = {
    "game_id",
    "team",
    "player_id",
    "player_name",
    "position",
    "available_non_qb_rush_pool",
    "r8v_absolute_prior_carries",
}

missing_current = (
    required_current
    - set(current.columns)
)

if missing_current:
    raise RuntimeError(
        "FAIL | missing current R8X columns: "
        f"{sorted(missing_current)}"
    )


current[
    "available_non_qb_rush_pool"
] = pd.to_numeric(
    current[
        "available_non_qb_rush_pool"
    ],
    errors="coerce",
)


current[
    "r8v_absolute_prior_carries"
] = pd.to_numeric(
    current[
        "r8v_absolute_prior_carries"
    ],
    errors="coerce",
)


current[
    "hybrid_feasible_carries"
] = np.minimum(
    current[
        "r8v_absolute_prior_carries"
    ],
    current[
        "available_non_qb_rush_pool"
    ],
)


current[
    "cap_applied"
] = (
    current[
        "hybrid_feasible_carries"
    ]
    <
    current[
        "r8v_absolute_prior_carries"
    ]
    - 1e-12
)


current[
    "cap_amount"
] = (
    current[
        "r8v_absolute_prior_carries"
    ]
    -
    current[
        "hybrid_feasible_carries"
    ]
)


current[
    "structurally_feasible"
] = (
    current[
        "hybrid_feasible_carries"
    ]
    <=
    current[
        "available_non_qb_rush_pool"
    ]
    + 1e-12
)


if not current[
    "structurally_feasible"
].all():
    raise RuntimeError(
        "FAIL | current hybrid carries infeasible"
    )


# ============================================================
# WRITE RESEARCH ARTIFACTS
# ============================================================

DETAIL_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

rb.to_csv(
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
# PRODUCTION SAFETY
# ============================================================

canonical_sha_after = sha256(
    CANONICAL
)

if (
    canonical_sha_after
    != canonical_sha_before
):
    raise RuntimeError(
        "FAIL | canonical mutated"
    )


audit = {
    "stage":
        "STAGE26G-R8Z",

    "purpose":
        "validate deterministic hard feasibility cap on cold-start RB absolute carry prior",

    "candidate_formula":
        "min(absolute_prior_carries, projected_non_qb_rush_pool)",

    "predictive_model_replacement":
        False,

    "role_share_used_as_prediction":
        False,

    "arbitrary_percentile_cap":
        False,

    "historical_rb_events":
        int(
            len(rb)
        ),

    "historical_cap_events":
        int(
            rb[
                "cap_applied"
            ].sum()
        ),

    "mae_non_degrade":
        mae_non_degrade,

    "rmse_non_degrade":
        rmse_non_degrade,

    "all_hybrid_feasible":
        all_feasible,

    "research_pass":
        research_pass,

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
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("R8Z HISTORICAL HARD-CAP VALIDATION")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8Z HISTORICAL EVENT DETAIL")
print("============================================================")

show = [
    "game_id",
    "team",
    "player_name",
    "actual_carries",
    "absolute_pred_carries",
    "projected_non_qb_pool",
    "hybrid_pred_carries",
    "cap_applied",
    "cap_amount",
]

print(
    rb[
        show
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8Z CURRENT COLD-START APPLICATION")
print("============================================================")

current_show = [
    "game_id",
    "team",
    "player_name",
    "position",
    "r8v_absolute_prior_carries",
    "available_non_qb_rush_pool",
    "hybrid_feasible_carries",
    "cap_applied",
    "cap_amount",
    "structurally_feasible",
]

print(
    current[
        current_show
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8Z DECISION")
print("============================================================")

print(
    f"MAE_NON_DEGRADE={mae_non_degrade}"
)

print(
    f"RMSE_NON_DEGRADE={rmse_non_degrade}"
)

print(
    f"ALL_HYBRID_FEASIBLE={all_feasible}"
)

print(
    f"HISTORICAL_CAP_EVENTS={int(rb['cap_applied'].sum())}"
)

print(
    f"RESEARCH_PASS={research_pass}"
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
print("STAGE26G_R8Z_FEASIBILITY_CAP_VALIDATION=PASS")
print("============================================================")
