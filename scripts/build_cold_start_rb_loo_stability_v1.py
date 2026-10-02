from __future__ import annotations

from pathlib import Path
from itertools import product
import contextlib
import hashlib
import io
import json
import runpy

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

R8T_SCRIPT = (
    ROOT
    / "scripts/build_cold_start_rb_joint_grid_v1.py"
)

CANONICAL_PATH = (
    ROOT
    / "data/parquet/current_unified_stat_forecasts.parquet"
)

OUT_FOLDS = (
    ROOT
    / "processed/stage26gr8u_rb_loo_folds.csv"
)

OUT_WEIGHT_SUMMARY = (
    ROOT
    / "processed/stage26gr8u_rb_loo_weight_summary.csv"
)

OUT_AUDIT = (
    ROOT
    / "processed/stage26gr8u_rb_loo_audit.json"
)

EXPECTED_R8T_SCRIPT_SHA = (
    "d61a9e326fda209f3797c0b474d271ea"
    "1aa8004ec4b83dc5a4c6d736e4db9f91"
)

WEIGHTS = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

COMPONENT_COLUMNS = [
    "catch_rate_weight",
    "receiving_ypr_weight",
    "receiving_td_weight",
    "rushing_ypc_weight",
    "rushing_td_weight",
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
        min(
            high,
            value,
        ),
    )


def blend(
    prior_value,
    dvp_value,
    weight,
):
    prior_value = num(
        prior_value
    )

    dvp_value = num(
        dvp_value
    )

    if pd.isna(
        prior_value
    ):
        return np.nan

    if pd.isna(
        dvp_value
    ):
        return prior_value

    return (
        (1.0 - weight)
        * prior_value
        +
        weight
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


def predict_event(
    row,
    weights,
):
    (
        catch_w,
        rec_ypr_w,
        rec_td_w,
        rush_ypc_w,
        rush_td_w,
    ) = weights

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
        num(
            row[
                "pred_targets"
            ],
            0.0,
        )
        * catch_rate
    )

    receiving_yards = (
        receptions
        * rec_ypr
    )

    receiving_tds = (
        num(
            row[
                "pred_targets"
            ],
            0.0,
        )
        * rec_td_rate
    )

    rushing_yards = (
        num(
            row[
                "pred_carries"
            ],
            0.0,
        )
        * rush_ypc
    )

    rushing_tds = (
        num(
            row[
                "pred_carries"
            ],
            0.0,
        )
        * rush_td_rate
    )

    return fd_points(
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


def fit_grid(train):
    rows = []

    actual = (
        train[
            "actual_fd_points"
        ]
        .astype(float)
        .to_numpy()
    )

    for weights in product(
        WEIGHTS,
        WEIGHTS,
        WEIGHTS,
        WEIGHTS,
        WEIGHTS,
    ):
        predicted = np.array(
            [
                predict_event(
                    row,
                    weights,
                )
                for _, row
                in train.iterrows()
            ],
            dtype=float,
        )

        errors = (
            predicted
            -
            actual
        )

        mae = float(
            np.mean(
                np.abs(
                    errors
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        )

        bias = float(
            np.mean(
                errors
            )
        )

        rows.append(
            {
                "catch_rate_weight":
                    weights[0],

                "receiving_ypr_weight":
                    weights[1],

                "receiving_td_weight":
                    weights[2],

                "rushing_ypc_weight":
                    weights[3],

                "rushing_td_weight":
                    weights[4],

                "total_dvp_weight":
                    sum(
                        weights
                    ),

                "train_fd_mae":
                    mae,

                "train_fd_rmse":
                    rmse,

                "train_fd_bias":
                    bias,

                "abs_train_fd_bias":
                    abs(
                        bias
                    ),
            }
        )

    grid = pd.DataFrame(
        rows
    )

    if len(grid) != 7776:
        raise RuntimeError(
            "FAIL | expected 7776 grid rows, "
            f"got {len(grid)}"
        )

    grid = grid.sort_values(
        [
            "train_fd_mae",
            "train_fd_rmse",
            "abs_train_fd_bias",
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

    return grid.iloc[0]


# ============================================================
# SAFETY GATES
# ============================================================

if not R8T_SCRIPT.exists():
    raise RuntimeError(
        "FAIL | R8T script missing"
    )

r8t_sha = sha256(
    R8T_SCRIPT
)

if (
    r8t_sha
    != EXPECTED_R8T_SCRIPT_SHA
):
    raise RuntimeError(
        "FAIL | R8T script SHA mismatch\n"
        f"EXPECTED={EXPECTED_R8T_SCRIPT_SHA}\n"
        f"ACTUAL={r8t_sha}"
    )

canonical_sha_before = (
    sha256(
        CANONICAL_PATH
    )
)


# ============================================================
# LOAD EXACT R8T HISTORICAL EVENT SET
#
# R8T already contains the validated, strict-date-blocked
# reconstruction logic. Execute it in an isolated namespace
# and suppress its reporting. We reuse only its in-memory
# historical `events` dataframe.
#
# This may deterministically rewrite R8T research CSVs, but
# cannot mutate production/canonical artifacts.
# ============================================================

capture = io.StringIO()

with contextlib.redirect_stdout(
    capture
):
    namespace = runpy.run_path(
        str(
            R8T_SCRIPT
        ),
        run_name=(
            "__stage26gr8u_r8t_loader__"
        ),
    )


events = namespace.get(
    "events"
)

if not isinstance(
    events,
    pd.DataFrame,
):
    raise RuntimeError(
        "FAIL | R8T did not expose events dataframe"
    )

events = (
    events.copy()
    .reset_index(
        drop=True
    )
)

if len(events) != 10:
    raise RuntimeError(
        "FAIL | expected exact R8T RB sample n=10, "
        f"got n={len(events)}"
    )


required = {
    "game_id",
    "player_id",
    "player_name",
    "pred_targets",
    "pred_carries",
    "prior_catch_rate",
    "prior_rec_ypr",
    "prior_rec_td_rate",
    "prior_rush_ypc",
    "prior_rush_td_rate",
    "dvp_catch_rate",
    "dvp_rec_ypr",
    "dvp_rec_td_rate",
    "dvp_rush_ypc",
    "dvp_rush_td_rate",
    "actual_fd_points",
}

missing = (
    required
    - set(
        events.columns
    )
)

if missing:
    raise RuntimeError(
        "FAIL | missing event columns: "
        f"{sorted(missing)}"
    )


print()
print("===== R8U EVENT COVERAGE =====")
print(
    f"RB_EVENTS={len(events)}"
)


# ============================================================
# FULL-SAMPLE R8T CONTROL
#
# Confirm this event reconstruction reproduces the known
# R8T winner before doing any leave-one-out analysis.
# ============================================================

full_best = fit_grid(
    events
)

expected_full_weights = (
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
)

actual_full_weights = tuple(
    float(
        full_best[c]
    )
    for c
    in COMPONENT_COLUMNS
)

if (
    actual_full_weights
    != expected_full_weights
):
    raise RuntimeError(
        "FAIL | full-sample control does not reproduce R8T "
        f"winner: {actual_full_weights}"
    )


print()
print("===== FULL-SAMPLE CONTROL =====")
print(
    "FULL_SAMPLE_WINNER="
    f"{actual_full_weights}"
)
print(
    "FULL_SAMPLE_MAE="
    f"{float(full_best['train_fd_mae']):.6f}"
)


# ============================================================
# LEAVE-ONE-EVENT-OUT
# ============================================================

baseline_weights = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

fold_rows = []

for holdout_idx in range(
    len(events)
):
    holdout = events.iloc[
        holdout_idx
    ]

    train = events.drop(
        index=holdout_idx
    ).reset_index(
        drop=True
    )

    best = fit_grid(
        train
    )

    best_weights = tuple(
        float(
            best[c]
        )
        for c
        in COMPONENT_COLUMNS
    )

    actual = float(
        holdout[
            "actual_fd_points"
        ]
    )

    baseline_pred = float(
        predict_event(
            holdout,
            baseline_weights,
        )
    )

    selected_pred = float(
        predict_event(
            holdout,
            best_weights,
        )
    )

    baseline_abs_error = abs(
        baseline_pred
        -
        actual
    )

    selected_abs_error = abs(
        selected_pred
        -
        actual
    )

    fold_rows.append(
        {
            "fold":
                holdout_idx + 1,

            "heldout_game_id":
                holdout[
                    "game_id"
                ],

            "heldout_player_id":
                holdout[
                    "player_id"
                ],

            "heldout_player_name":
                holdout[
                    "player_name"
                ],

            "catch_rate_weight":
                best_weights[0],

            "receiving_ypr_weight":
                best_weights[1],

            "receiving_td_weight":
                best_weights[2],

            "rushing_ypc_weight":
                best_weights[3],

            "rushing_td_weight":
                best_weights[4],

            "total_dvp_weight":
                sum(
                    best_weights
                ),

            "train_fd_mae":
                float(
                    best[
                        "train_fd_mae"
                    ]
                ),

            "train_fd_rmse":
                float(
                    best[
                        "train_fd_rmse"
                    ]
                ),

            "actual_fd_points":
                actual,

            "baseline_pred_fd":
                baseline_pred,

            "selected_pred_fd":
                selected_pred,

            "baseline_abs_error":
                baseline_abs_error,

            "selected_abs_error":
                selected_abs_error,

            "holdout_error_delta":
                (
                    selected_abs_error
                    -
                    baseline_abs_error
                ),

            "selected_beats_baseline":
                bool(
                    selected_abs_error
                    <
                    baseline_abs_error
                    -
                    1e-12
                ),

            "selected_ties_baseline":
                bool(
                    abs(
                        selected_abs_error
                        -
                        baseline_abs_error
                    )
                    <= 1e-12
                ),
        }
    )


folds = pd.DataFrame(
    fold_rows
)


# ============================================================
# OUT-OF-SAMPLE AGGREGATE
# ============================================================

actual = folds[
    "actual_fd_points"
].to_numpy(
    dtype=float
)

baseline_pred = folds[
    "baseline_pred_fd"
].to_numpy(
    dtype=float
)

selected_pred = folds[
    "selected_pred_fd"
].to_numpy(
    dtype=float
)


baseline_errors = (
    baseline_pred
    -
    actual
)

selected_errors = (
    selected_pred
    -
    actual
)


baseline_loo_mae = float(
    np.mean(
        np.abs(
            baseline_errors
        )
    )
)

selected_loo_mae = float(
    np.mean(
        np.abs(
            selected_errors
        )
    )
)

baseline_loo_rmse = float(
    np.sqrt(
        np.mean(
            baseline_errors ** 2
        )
    )
)

selected_loo_rmse = float(
    np.sqrt(
        np.mean(
            selected_errors ** 2
        )
    )
)

baseline_loo_bias = float(
    np.mean(
        baseline_errors
    )
)

selected_loo_bias = float(
    np.mean(
        selected_errors
    )
)

loo_mae_improvement = (
    baseline_loo_mae
    -
    selected_loo_mae
)

loo_mae_improvement_pct = (
    loo_mae_improvement
    /
    baseline_loo_mae
    * 100.0
    if baseline_loo_mae > 0
    else 0.0
)

loo_rmse_delta = (
    selected_loo_rmse
    -
    baseline_loo_rmse
)

fold_wins = int(
    folds[
        "selected_beats_baseline"
    ].sum()
)

fold_ties = int(
    folds[
        "selected_ties_baseline"
    ].sum()
)

fold_losses = int(
    len(folds)
    -
    fold_wins
    -
    fold_ties
)


# ============================================================
# WEIGHT STABILITY SUMMARY
# ============================================================

weight_summary_rows = []

for component in COMPONENT_COLUMNS:
    values = folds[
        component
    ].astype(
        float
    )

    counts = (
        values
        .value_counts()
        .sort_index()
    )

    mode_values = (
        values
        .mode()
        .tolist()
    )

    weight_summary_rows.append(
        {
            "component":
                component,

            "mean":
                float(
                    values.mean()
                ),

            "median":
                float(
                    values.median()
                ),

            "minimum":
                float(
                    values.min()
                ),

            "maximum":
                float(
                    values.max()
                ),

            "mode":
                ",".join(
                    f"{float(x):.2f}"
                    for x
                    in mode_values
                ),

            "zero_folds":
                int(
                    (
                        values
                        == 0.0
                    ).sum()
                ),

            "positive_folds":
                int(
                    (
                        values
                        > 0.0
                    ).sum()
                ),

            "high_folds_ge_0_50":
                int(
                    (
                        values
                        >= 0.50
                    ).sum()
                ),

            "weight_counts":
                "; ".join(
                    f"{float(k):.2f}:{int(v)}"
                    for k, v
                    in counts.items()
                ),
        }
    )


weight_summary = pd.DataFrame(
    weight_summary_rows
)


# ============================================================
# STRUCTURAL SIGNAL TEST
#
# R8T hypothesis:
# - receiving matchup signal survives
# - TD/rushing matchup remains near zero
#
# This is diagnostic, not a production rule.
# ============================================================

catch_positive_rate = float(
    (
        folds[
            "catch_rate_weight"
        ]
        > 0.0
    ).mean()
)

rec_ypr_positive_rate = float(
    (
        folds[
            "receiving_ypr_weight"
        ]
        > 0.0
    ).mean()
)

rec_td_zero_rate = float(
    (
        folds[
            "receiving_td_weight"
        ]
        == 0.0
    ).mean()
)

rush_ypc_zero_rate = float(
    (
        folds[
            "rushing_ypc_weight"
        ]
        == 0.0
    ).mean()
)

rush_td_zero_rate = float(
    (
        folds[
            "rushing_td_weight"
        ]
        == 0.0
    ).mean()
)


# Require a clear majority of folds to preserve structure.
structural_gate = bool(
    catch_positive_rate
    >= 0.70
    and rec_ypr_positive_rate
    >= 0.70
    and rec_td_zero_rate
    >= 0.70
    and rush_ypc_zero_rate
    >= 0.70
    and rush_td_zero_rate
    >= 0.70
)


# ============================================================
# GENERALIZATION GATE
#
# Candidate survives only if:
# - selected LOO MAE beats zero-DvP baseline
# - selected LOO RMSE does not regress >2.5%
# - at least 6/10 holdouts beat baseline
# - structural receiving-only signal persists
# ============================================================

mae_gate = bool(
    selected_loo_mae
    <
    baseline_loo_mae
    -
    1e-12
)

rmse_gate = bool(
    selected_loo_rmse
    <=
    baseline_loo_rmse
    * 1.025
)

fold_win_gate = bool(
    fold_wins
    >= 6
)


if (
    mae_gate
    and rmse_gate
    and fold_win_gate
    and structural_gate
):
    diagnostic = (
        "RB_MATCHUP_SIGNAL_SURVIVES_LOO"
    )
else:
    diagnostic = (
        "RB_MATCHUP_SIGNAL_FAILS_LOO_KEEP_ZERO_DVP"
    )


# ============================================================
# WRITE ARTIFACTS
# ============================================================

OUT_FOLDS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

folds.to_csv(
    OUT_FOLDS,
    index=False,
)

weight_summary.to_csv(
    OUT_WEIGHT_SUMMARY,
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
        "STAGE26G-R8U",

    "purpose":
        "leave-one-event-out RB matchup-weight stability validation",

    "r8t_script_sha":
        r8t_sha,

    "rb_events":
        int(
            len(events)
        ),

    "folds":
        int(
            len(folds)
        ),

    "grid_combinations_per_fold":
        7776,

    "baseline_weights": {
        "catch_rate":
            0.0,

        "receiving_ypr":
            0.0,

        "receiving_td":
            0.0,

        "rushing_ypc":
            0.0,

        "rushing_td":
            0.0,
    },

    "full_sample_r8t_weights": {
        "catch_rate":
            1.0,

        "receiving_ypr":
            1.0,

        "receiving_td":
            0.0,

        "rushing_ypc":
            0.0,

        "rushing_td":
            0.0,
    },

    "loo": {
        "baseline_mae":
            baseline_loo_mae,

        "selected_mae":
            selected_loo_mae,

        "mae_improvement":
            loo_mae_improvement,

        "mae_improvement_pct":
            loo_mae_improvement_pct,

        "baseline_rmse":
            baseline_loo_rmse,

        "selected_rmse":
            selected_loo_rmse,

        "rmse_delta":
            loo_rmse_delta,

        "baseline_bias":
            baseline_loo_bias,

        "selected_bias":
            selected_loo_bias,

        "fold_wins":
            fold_wins,

        "fold_ties":
            fold_ties,

        "fold_losses":
            fold_losses,
    },

    "structural_rates": {
        "catch_positive_rate":
            catch_positive_rate,

        "receiving_ypr_positive_rate":
            rec_ypr_positive_rate,

        "receiving_td_zero_rate":
            rec_td_zero_rate,

        "rushing_ypc_zero_rate":
            rush_ypc_zero_rate,

        "rushing_td_zero_rate":
            rush_td_zero_rate,
    },

    "gates": {
        "mae_gate":
            mae_gate,

        "rmse_gate":
            rmse_gate,

        "fold_win_gate":
            fold_win_gate,

        "structural_gate":
            structural_gate,
    },

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
print("R8U FOLD WINNERS")
print("============================================================")

print(
    folds[
        [
            "fold",
            "heldout_game_id",
            "heldout_player_name",
            "catch_rate_weight",
            "receiving_ypr_weight",
            "receiving_td_weight",
            "rushing_ypc_weight",
            "rushing_td_weight",
            "train_fd_mae",
            "baseline_abs_error",
            "selected_abs_error",
            "holdout_error_delta",
        ]
    ].to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8U WEIGHT STABILITY")
print("============================================================")

print(
    weight_summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("R8U OUT-OF-SAMPLE PERFORMANCE")
print("============================================================")

print(
    f"BASELINE_LOO_MAE="
    f"{baseline_loo_mae:.6f}"
)

print(
    f"SELECTED_LOO_MAE="
    f"{selected_loo_mae:.6f}"
)

print(
    f"LOO_MAE_IMPROVEMENT="
    f"{loo_mae_improvement:.6f}"
)

print(
    f"LOO_MAE_IMPROVEMENT_PCT="
    f"{loo_mae_improvement_pct:.6f}%"
)

print(
    f"BASELINE_LOO_RMSE="
    f"{baseline_loo_rmse:.6f}"
)

print(
    f"SELECTED_LOO_RMSE="
    f"{selected_loo_rmse:.6f}"
)

print(
    f"LOO_RMSE_DELTA="
    f"{loo_rmse_delta:.6f}"
)

print(
    f"BASELINE_LOO_BIAS="
    f"{baseline_loo_bias:.6f}"
)

print(
    f"SELECTED_LOO_BIAS="
    f"{selected_loo_bias:.6f}"
)

print(
    f"FOLD_WINS="
    f"{fold_wins}"
)

print(
    f"FOLD_TIES="
    f"{fold_ties}"
)

print(
    f"FOLD_LOSSES="
    f"{fold_losses}"
)


print()
print("============================================================")
print("R8U STRUCTURAL SIGNAL")
print("============================================================")

print(
    f"CATCH_POSITIVE_RATE="
    f"{catch_positive_rate:.3f}"
)

print(
    f"REC_YPR_POSITIVE_RATE="
    f"{rec_ypr_positive_rate:.3f}"
)

print(
    f"REC_TD_ZERO_RATE="
    f"{rec_td_zero_rate:.3f}"
)

print(
    f"RUSH_YPC_ZERO_RATE="
    f"{rush_ypc_zero_rate:.3f}"
)

print(
    f"RUSH_TD_ZERO_RATE="
    f"{rush_td_zero_rate:.3f}"
)


print()
print("============================================================")
print("R8U GATES")
print("============================================================")

print(
    f"MAE_GATE={mae_gate}"
)

print(
    f"RMSE_GATE={rmse_gate}"
)

print(
    f"FOLD_WIN_GATE={fold_win_gate}"
)

print(
    f"STRUCTURAL_GATE={structural_gate}"
)

print(
    f"DIAGNOSTIC={diagnostic}"
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
    f"FOLDS={OUT_FOLDS}"
)

print(
    f"WEIGHT_SUMMARY={OUT_WEIGHT_SUMMARY}"
)

print(
    f"AUDIT={OUT_AUDIT}"
)


print()
print("============================================================")
print("STAGE26G_R8U_RB_LOO_STABILITY=PASS")
print("============================================================")
