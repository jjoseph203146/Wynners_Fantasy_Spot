#!/usr/bin/env python3

"""
STAGE26G-J
PREGAME CONTEXT PASS EXPECTATION VALIDATION

Purpose
-------
Build and validate a strictly pregame / point-in-time contextual pass
expectation using the frozen Exp004 CORE feature universe, excluding all
legacy coach-history variables.

Then, in shadow only, test the already-validated Stage26G coach adjustment:

    context_pass_expectation
    + 0.70 * (prior_mean_pass_oe / 100)

This script is validation-only.

It does NOT:
- write artifacts
- write databases
- mutate forecast models
- mutate the solver
- mutate the app
- restart services
- enable production influence
- modify any frozen Stage26G artifact

Strict temporal rule
--------------------
For every target game date:

    training game_date < target game_date

Same-game, same-date, and future-date training are forbidden.
"""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# RUNTIME CONTRACT
# =============================================================================

ANALYSIS_ONLY = True
DATE_BLOCKED_WALK_FORWARD = True
TRAIN_DATE_STRICTLY_LT_TARGET_DATE = True
SAME_GAME_TRAINING_ALLOWED = False
SAME_DATE_TRAINING_ALLOWED = False
FUTURE_DATE_TRAINING_ALLOWED = False

ARTIFACT_WRITE = False
DATABASE_WRITE = False
CODE_MUTATION = False
SOLVER_MUTATION = False
FORECAST_MUTATION = False
APP_MUTATION = False
SERVICE_RESTART = False
PRODUCTION_INFLUENCE = False

ROOT = Path("/home/mwynn/nfl_data_engine")

EXP004_SCRIPT = (
    ROOT
    / "scripts"
    / "run_forecast_v1_experiment_004.py"
)

EXP004_MANIFEST = (
    ROOT
    / "experiments"
    / "forecast_v1_exp004"
    / "forecast_v1_exp004_features.json"
)

STAGE26GA_SCRIPT = (
    ROOT
    / "scripts"
    / "stage26ga_coach_regime_tendency_matrix.py"
)

COACH_WEIGHT = 0.70

MIN_TRAIN_ROWS = 200

ALPHA_GRID = [
    0.0,
    0.1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
]

LEGACY_COACH_FIELDS = {
    "coach_prior_games",
    "coach_prior_points_against_avg",
    "coach_prior_points_for_avg",
    "coach_prior_win_pct",
    "opponent_coach_prior_games",
    "opponent_coach_prior_points_against_avg",
    "opponent_coach_prior_points_for_avg",
    "opponent_coach_prior_win_pct",
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def fail(message: str) -> None:
    raise RuntimeError(message)


def safe_mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(a[mask] - b[mask])
        )
    )


def safe_rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if not mask.any():
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                (a[mask] - b[mask]) ** 2
            )
        )
    )


def safe_corr(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if int(mask.sum()) < 3:
        return float("nan")

    aa = a[mask]
    bb = b[mask]

    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")

    return float(
        np.corrcoef(
            aa,
            bb,
        )[0, 1]
    )


# =============================================================================
# DETERMINISTIC RIDGE
# =============================================================================

class DeterministicRidge:

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

        self.columns_ = None
        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.keep_mask_ = None
        self.coef_ = None

    def _raw_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[
            self.columns_
        ].copy()

        for col in self.columns_:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

        return x.to_numpy(
            dtype=float
        )

    def fit(
        self,
        X: pd.DataFrame,
        y,
    ) -> "DeterministicRidge":

        self.columns_ = list(X.columns)

        matrix = self._raw_matrix(X)

        y = np.asarray(
            y,
            dtype=float,
        )

        if len(matrix) != len(y):
            fail(
                "Ridge X/y length mismatch."
            )

        if not np.isfinite(y).all():
            fail(
                "Non-finite ridge target."
            )

        with np.errstate(all="ignore"):
            medians = np.nanmedian(
                matrix,
                axis=0,
            )

        medians = np.where(
            np.isfinite(medians),
            medians,
            0.0,
        )

        matrix = np.where(
            np.isfinite(matrix),
            matrix,
            np.nan,
        )

        nan_rows, nan_cols = np.where(
            np.isnan(matrix)
        )

        if len(nan_rows):
            matrix[
                nan_rows,
                nan_cols,
            ] = medians[nan_cols]

        means = np.mean(
            matrix,
            axis=0,
        )

        stds = np.std(
            matrix,
            axis=0,
        )

        keep = (
            np.isfinite(stds)
            &
            (stds > 0)
        )

        if not keep.any():
            fail(
                "No usable nonconstant features."
            )

        z = (
            matrix[:, keep]
            - means[keep]
        ) / stds[keep]

        design = np.column_stack(
            [
                np.ones(
                    len(z),
                    dtype=float,
                ),
                z,
            ]
        )

        penalty = np.eye(
            design.shape[1],
            dtype=float,
        )

        penalty[0, 0] = 0.0

        lhs = (
            design.T
            @ design
            +
            self.alpha
            * penalty
        )

        rhs = (
            design.T
            @ y
        )

        coef = np.linalg.pinv(
            lhs
        ) @ rhs

        if not np.isfinite(coef).all():
            fail(
                "Non-finite ridge coefficients."
            )

        self.medians_ = medians
        self.means_ = means
        self.stds_ = stds
        self.keep_mask_ = keep
        self.coef_ = coef

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        matrix = self._raw_matrix(X)

        matrix = np.where(
            np.isfinite(matrix),
            matrix,
            np.nan,
        )

        nan_rows, nan_cols = np.where(
            np.isnan(matrix)
        )

        if len(nan_rows):
            matrix[
                nan_rows,
                nan_cols,
            ] = self.medians_[nan_cols]

        z = (
            matrix[
                :,
                self.keep_mask_
            ]
            -
            self.means_[
                self.keep_mask_
            ]
        ) / self.stds_[
            self.keep_mask_
        ]

        design = np.column_stack(
            [
                np.ones(
                    len(z),
                    dtype=float,
                ),
                z,
            ]
        )

        pred = (
            design
            @ self.coef_
        )

        if not np.isfinite(pred).all():
            fail(
                "Non-finite ridge predictions."
            )

        return pred


# =============================================================================
# EXP004 MANIFEST
# =============================================================================

def load_context_feature_contract():

    if not EXP004_MANIFEST.exists():
        fail(
            f"Missing Exp004 manifest: "
            f"{EXP004_MANIFEST}"
        )

    obj = json.loads(
        EXP004_MANIFEST.read_text()
    )

    core = obj.get(
        "core_features"
    )

    if not isinstance(core, list):
        fail(
            "Exp004 manifest missing "
            "core_features list."
        )

    if len(core) != 76:
        fail(
            f"Expected 76 frozen CORE features; "
            f"found {len(core)}"
        )

    missing_legacy = sorted(
        LEGACY_COACH_FIELDS
        - set(core)
    )

    if missing_legacy:
        fail(
            "Frozen CORE missing expected "
            "legacy coach fields: "
            f"{missing_legacy}"
        )

    context_features = [
        c
        for c in core
        if c not in LEGACY_COACH_FIELDS
    ]

    if len(context_features) != 68:
        fail(
            f"Expected 68 context features; "
            f"found {len(context_features)}"
        )

    return core, context_features


# =============================================================================
# SOURCE DISCOVERY
# =============================================================================

def read_frame(path: Path) -> pd.DataFrame:

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(
            path
        )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    fail(
        f"Unsupported tabular source: {path}"
    )


def discover_context_source(
    required_features,
):

    namespace = runpy.run_path(
        str(EXP004_SCRIPT),
        run_name="stage26gj_exp004_probe",
    )

    candidates = []

    for name, value in namespace.items():

        if (
            isinstance(value, Path)
            and value.exists()
            and value.is_file()
            and value.suffix.lower()
            in {".csv", ".parquet"}
        ):
            candidates.append(
                (
                    f"EXP004_NAMESPACE::{name}",
                    value,
                )
            )

    extra_candidates = [
        ROOT
        / "processed"
        / "forecast_v1_dataset.parquet",

        ROOT
        / "processed"
        / "forecast_v1_dataset.csv",

        ROOT
        / "data"
        / "parquet"
        / "forecast_v1_dataset.parquet",

        ROOT
        / "data"
        / "csv"
        / "forecast_v1_dataset.csv",

        ROOT
        / "processed"
        / "forecast_v1_training_dataset.parquet",

        ROOT
        / "processed"
        / "forecast_v1_training_dataset.csv",
    ]

    for path in extra_candidates:
        if path.exists():
            candidates.append(
                (
                    "STANDARD_CANDIDATE",
                    path,
                )
            )

    # de-duplicate paths while preserving order
    seen = set()
    unique = []

    for origin, path in candidates:
        key = str(
            path.resolve()
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(
            (
                origin,
                path,
            )
        )

    required = (
        {
            "game_id",
            "team",
        }
        |
        set(required_features)
    )

    diagnostics = []

    for origin, path in unique:

        try:
            frame = read_frame(
                path
            )

        except Exception as exc:
            diagnostics.append(
                (
                    str(path),
                    f"READ_ERROR={exc}",
                )
            )
            continue

        missing = sorted(
            required
            - set(frame.columns)
        )

        diagnostics.append(
            (
                str(path),
                (
                    f"ROWS={len(frame)} "
                    f"MISSING_REQUIRED={len(missing)}"
                ),
            )
        )

        if not missing:

            if frame.duplicated(
                ["game_id", "team"]
            ).any():
                continue

            return (
                frame,
                path,
                origin,
                diagnostics,
            )

    lines = [
        "No Exp004-compatible context source found.",
        "",
        "Candidate diagnostics:",
    ]

    for path, detail in diagnostics:
        lines.append(
            f"  {path} | {detail}"
        )

    fail(
        "\n".join(lines)
    )


# =============================================================================
# STAGE26G-A AUTHORITY
# =============================================================================

def recover_stage26ga_matrix():

    if not STAGE26GA_SCRIPT.exists():
        fail(
            f"Missing Stage26G-A script: "
            f"{STAGE26GA_SCRIPT}"
        )

    ns = runpy.run_path(
        str(STAGE26GA_SCRIPT)
    )

    matrix = ns.get(
        "matrix"
    )

    if not isinstance(
        matrix,
        pd.DataFrame,
    ):
        fail(
            "Stage26G-A did not expose "
            "matrix DataFrame."
        )

    matrix = matrix.copy()

    required = {
        "game_id",
        "team",
        "season",
        "week",
        "game_date_dt",
        "raw_n",
        "raw_pass",
        "prior_pbp_regime_games",
        "prior_mean_pass_oe",
        "team_coach_regime_key",
    }

    missing = sorted(
        required
        - set(matrix.columns)
    )

    if missing:
        fail(
            "Stage26G-A matrix missing: "
            f"{missing}"
        )

    if matrix.duplicated(
        ["game_id", "team"]
    ).any():
        fail(
            "Stage26G-A duplicate "
            "game_id + team keys."
        )

    return matrix


# =============================================================================
# BUILD VALIDATION FRAME
# =============================================================================

def build_validation_frame(
    context_source,
    context_features,
    coach_matrix,
):

    source = context_source.copy()

    keep = (
        [
            "game_id",
            "team",
        ]
        +
        context_features
    )

    optional_identity = [
        "season",
        "week",
        "game_date",
    ]

    for col in optional_identity:
        if (
            col in source.columns
            and col not in keep
        ):
            keep.append(col)

    source = source[
        keep
    ].copy()

    coach_keep = [
        "game_id",
        "team",
        "game_date_dt",
        "raw_n",
        "raw_pass",
        "prior_pbp_regime_games",
        "prior_mean_pass_oe",
        "team_coach_regime_key",
    ]

    merged = source.merge(
        coach_matrix[
            coach_keep
        ],
        on=[
            "game_id",
            "team",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_context",
            "_coach",
        ),
    )

    if len(merged) < 1600:
        fail(
            "Context/G-A overlap unexpectedly low: "
            f"{len(merged)} rows."
        )

    merged[
        "game_date_dt"
    ] = pd.to_datetime(
        merged["game_date_dt"],
        errors="raise",
    )

    raw_n = pd.to_numeric(
        merged["raw_n"],
        errors="raise",
    )

    raw_pass = pd.to_numeric(
        merged["raw_pass"],
        errors="raise",
    )

    if (
        raw_n <= 0
    ).any():
        fail(
            "Non-positive frozen decision count."
        )

    merged[
        "current_observed_pass_rate"
    ] = (
        raw_pass
        /
        raw_n
    )

    rate = merged[
        "current_observed_pass_rate"
    ]

    bad_rate = (
        (~np.isfinite(rate))
        |
        rate.lt(0)
        |
        rate.gt(1)
    )

    if bad_rate.any():
        fail(
            "Observed pass-rate target "
            "outside [0,1]."
        )

    merged[
        "prior_pbp_regime_games"
    ] = pd.to_numeric(
        merged[
            "prior_pbp_regime_games"
        ],
        errors="raise",
    ).astype(int)

    merged[
        "prior_mean_pass_oe"
    ] = pd.to_numeric(
        merged[
            "prior_mean_pass_oe"
        ],
        errors="coerce",
    )

    cold = (
        merged[
            "prior_pbp_regime_games"
        ]
        .eq(0)
    )

    merged[
        "coach_pass_rate_adjustment"
    ] = np.where(
        cold,
        0.0,
        (
            merged[
                "prior_mean_pass_oe"
            ].fillna(0.0)
            *
            COACH_WEIGHT
            /
            100.0
        ),
    )

    return (
        merged
        .sort_values(
            [
                "game_date_dt",
                "game_id",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# STRICT DATE-BLOCKED MODEL SELECTION
# =============================================================================

def choose_alpha_from_prior_only(
    train: pd.DataFrame,
    features,
):

    dates = (
        train[
            "game_date_dt"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # If training history is still shallow, use deterministic
    # conservative default rather than peeking forward.
    if len(dates) < 8:
        return 30.0

    split_index = int(
        math.floor(
            len(dates)
            * 0.80
        )
    )

    split_index = max(
        1,
        min(
            split_index,
            len(dates) - 1,
        ),
    )

    validation_start = pd.Timestamp(
        dates[
            split_index
        ]
    )

    inner_train = train[
        train[
            "game_date_dt"
        ]
        .lt(
            validation_start
        )
    ].copy()

    inner_valid = train[
        train[
            "game_date_dt"
        ]
        .ge(
            validation_start
        )
    ].copy()

    if (
        len(inner_train) < 100
        or len(inner_valid) < 25
    ):
        return 30.0

    y_train = inner_train[
        "current_observed_pass_rate"
    ].to_numpy(
        dtype=float
    )

    y_valid = inner_valid[
        "current_observed_pass_rate"
    ].to_numpy(
        dtype=float
    )

    results = []

    for alpha in ALPHA_GRID:

        model = DeterministicRidge(
            alpha
        )

        model.fit(
            inner_train[
                features
            ],
            y_train,
        )

        pred = model.predict(
            inner_valid[
                features
            ]
        )

        results.append(
            (
                safe_mae(
                    y_valid,
                    pred,
                ),
                safe_rmse(
                    y_valid,
                    pred,
                ),
                float(alpha),
            )
        )

    results.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        )
    )

    return float(
        results[0][2]
    )


def date_blocked_walk_forward(
    frame,
    features,
):

    out = frame.copy()

    out[
        "context_pass_expectation"
    ] = np.nan

    out[
        "selected_alpha"
    ] = np.nan

    out[
        "training_rows"
    ] = 0

    out[
        "max_training_date"
    ] = pd.NaT

    target_dates = (
        out[
            "game_date_dt"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    evaluated_dates = 0

    for target_date_raw in target_dates:

        target_date = pd.Timestamp(
            target_date_raw
        )

        train = out[
            out[
                "game_date_dt"
            ]
            .lt(
                target_date
            )
        ].copy()

        target = out[
            out[
                "game_date_dt"
            ]
            .eq(
                target_date
            )
        ].copy()

        if len(train) < MIN_TRAIN_ROWS:
            continue

        alpha = choose_alpha_from_prior_only(
            train,
            features,
        )

        model = DeterministicRidge(
            alpha
        )

        model.fit(
            train[
                features
            ],
            train[
                "current_observed_pass_rate"
            ].to_numpy(
                dtype=float
            ),
        )

        pred = model.predict(
            target[
                features
            ]
        )

        idx = target.index

        out.loc[
            idx,
            "context_pass_expectation"
        ] = pred

        out.loc[
            idx,
            "selected_alpha"
        ] = alpha

        out.loc[
            idx,
            "training_rows"
        ] = len(train)

        out.loc[
            idx,
            "max_training_date"
        ] = train[
            "game_date_dt"
        ].max()

        evaluated_dates += 1

    return (
        out,
        evaluated_dates,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "STAGE26G-J — PREGAME CONTEXT PASS EXPECTATION VALIDATION"
    )

    print(
        f"ANALYSIS_ONLY={str(ANALYSIS_ONLY).upper()}"
    )
    print(
        "DATE_BLOCKED_WALK_FORWARD=TRUE"
    )
    print(
        "TRAIN_DATE_STRICTLY_LT_TARGET_DATE=TRUE"
    )
    print(
        "SAME_GAME_TRAINING_ALLOWED=FALSE"
    )
    print(
        "SAME_DATE_TRAINING_ALLOWED=FALSE"
    )
    print(
        "FUTURE_DATE_TRAINING_ALLOWED=FALSE"
    )
    print(
        f"COACH_SIGNAL_WEIGHT={COACH_WEIGHT:.2f}"
    )
    print(
        "PRODUCTION_INFLUENCE=FALSE"
    )

    # -------------------------------------------------------------------------
    section(
        "1. FROZEN EXP004 CONTEXT FEATURE CONTRACT"
    )

    (
        core_features,
        context_features,
    ) = load_context_feature_contract()

    print(
        f"FROZEN_CORE_FEATURES={len(core_features)}"
    )
    print(
        "LEGACY_COACH_FIELDS_EXCLUDED="
        f"{len(LEGACY_COACH_FIELDS)}"
    )
    print(
        "PREGAME_CONTEXT_FEATURES="
        f"{len(context_features)}"
    )
    print(
        "EXP004_CONTEXT_FEATURE_CONTRACT=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "2. EXP004 PREGAME CONTEXT SOURCE"
    )

    (
        context_source,
        context_path,
        context_origin,
        diagnostics,
    ) = discover_context_source(
        context_features
    )

    print(
        f"CONTEXT_SOURCE={context_path}"
    )
    print(
        f"CONTEXT_SOURCE_ORIGIN={context_origin}"
    )
    print(
        f"CONTEXT_SOURCE_ROWS={len(context_source)}"
    )
    print(
        "CONTEXT_SOURCE_DUPLICATE_GAME_TEAM_ROWS="
        f"{int(context_source.duplicated(['game_id','team']).sum())}"
    )
    print(
        "EXP004_PREGAME_CONTEXT_SOURCE=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "3. FROZEN STAGE26G-A PASS-CALL TARGET AUTHORITY"
    )

    coach_matrix = recover_stage26ga_matrix()

    print(
        f"STAGE26G_A_ROWS={len(coach_matrix)}"
    )
    print(
        "STAGE26G_A_DUPLICATE_GAME_TEAM_ROWS="
        f"{int(coach_matrix.duplicated(['game_id','team']).sum())}"
    )
    print(
        "TARGET_DECISION_UNIVERSE="
        "STAGE26G_A_FROZEN_PASS_CALL_VS_DESIGNED_RUSH"
    )
    print(
        "STAGE26G_A_TARGET_AUTHORITY=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "4. VALIDATION FRAME"
    )

    frame = build_validation_frame(
        context_source,
        context_features,
        coach_matrix,
    )

    print(
        f"VALIDATION_SOURCE_ROWS={len(frame)}"
    )
    print(
        "VALIDATION_REGIMES="
        f"{frame['team_coach_regime_key'].nunique()}"
    )
    print(
        "TARGET_PASS_RATE_MIN="
        f"{frame['current_observed_pass_rate'].min():.8f}"
    )
    print(
        "TARGET_PASS_RATE_MAX="
        f"{frame['current_observed_pass_rate'].max():.8f}"
    )
    print(
        "VALIDATION_FRAME=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "5. DATE-BLOCKED WALK-FORWARD CONTEXT MODEL"
    )

    (
        scored,
        evaluated_dates,
    ) = date_blocked_walk_forward(
        frame,
        context_features,
    )

    oos = scored[
        scored[
            "context_pass_expectation"
        ].notna()
    ].copy()

    if len(oos) == 0:
        fail(
            "No strict OOS context predictions generated."
        )

    temporal_violations = int(
        (
            pd.to_datetime(
                oos["max_training_date"]
            )
            >=
            pd.to_datetime(
                oos["game_date_dt"]
            )
        ).sum()
    )

    if temporal_violations:
        fail(
            "Strict date-blocked temporal leakage."
        )

    print(
        f"TARGET_DATES_EVALUATED={evaluated_dates}"
    )
    print(
        f"STRICT_OOS_ROWS={len(oos)}"
    )
    print(
        "MIN_TRAIN_ROWS_OBSERVED="
        f"{int(oos['training_rows'].min())}"
    )
    print(
        "MAX_TRAIN_DATE_GE_TARGET_DATE_ROWS="
        f"{temporal_violations}"
    )
    print(
        "CURRENT_GAME_INCLUDED_IN_TRAINING=FALSE"
    )
    print(
        "SAME_DATE_INCLUDED_IN_TRAINING=FALSE"
    )
    print(
        "FUTURE_GAME_INCLUDED_IN_TRAINING=FALSE"
    )
    print(
        "DATE_BLOCKED_CONTEXT_MODEL=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "6. CONTEXT MODEL OOS PERFORMANCE"
    )

    y = oos[
        "current_observed_pass_rate"
    ].to_numpy(
        dtype=float
    )

    context_pred = oos[
        "context_pass_expectation"
    ].to_numpy(
        dtype=float
    )

    context_mae = safe_mae(
        y,
        context_pred,
    )

    context_rmse = safe_rmse(
        y,
        context_pred,
    )

    context_corr = safe_corr(
        y,
        context_pred,
    )

    print(
        f"CONTEXT_MAE={context_mae:.8f}"
    )
    print(
        f"CONTEXT_RMSE={context_rmse:.8f}"
    )
    print(
        f"CONTEXT_CORR={context_corr:.8f}"
    )

    context_range_violations = int(
        (
            (oos["context_pass_expectation"] < 0)
            |
            (oos["context_pass_expectation"] > 1)
        ).sum()
    )

    print(
        "CONTEXT_RAW_RANGE_VIOLATIONS="
        f"{context_range_violations}"
    )
    print(
        "CONTEXT_PREDICTION_CLIPPING_APPLIED=FALSE"
    )

    # -------------------------------------------------------------------------
    section(
        "7. SIMPLE PREGAME PASS-RATE BASELINES"
    )

    baseline_results = {}

    for col in [
        "team_pass_rate_avg_3",
        "team_pass_rate_avg_5",
    ]:

        if col not in oos.columns:
            print(
                f"{col}=NOT_AVAILABLE"
            )
            continue

        pred = pd.to_numeric(
            oos[col],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        m = np.isfinite(pred)

        baseline_results[
            col
        ] = {
            "n": int(m.sum()),
            "mae": safe_mae(
                y[m],
                pred[m],
            ),
            "rmse": safe_rmse(
                y[m],
                pred[m],
            ),
            "corr": safe_corr(
                y[m],
                pred[m],
            ),
        }

        r = baseline_results[col]

        print(
            f"{col.upper()}_N={r['n']}"
        )
        print(
            f"{col.upper()}_MAE={r['mae']:.8f}"
        )
        print(
            f"{col.upper()}_RMSE={r['rmse']:.8f}"
        )
        print(
            f"{col.upper()}_CORR={r['corr']:.8f}"
        )

    # -------------------------------------------------------------------------
    section(
        "8. FROZEN STAGE26G COACH ADJUSTMENT SHADOW TEST"
    )

    oos[
        "coach_adjusted_pass_expectation"
    ] = (
        oos[
            "context_pass_expectation"
        ]
        +
        oos[
            "coach_pass_rate_adjustment"
        ]
    )

    adjusted = oos[
        "coach_adjusted_pass_expectation"
    ].to_numpy(
        dtype=float
    )

    adjusted_mae = safe_mae(
        y,
        adjusted,
    )

    adjusted_rmse = safe_rmse(
        y,
        adjusted,
    )

    adjusted_corr = safe_corr(
        y,
        adjusted,
    )

    delta_mae = (
        adjusted_mae
        -
        context_mae
    )

    delta_rmse = (
        adjusted_rmse
        -
        context_rmse
    )

    context_abs_error = np.abs(
        y
        -
        context_pred
    )

    adjusted_abs_error = np.abs(
        y
        -
        adjusted
    )

    pct_rows_improved = float(
        np.mean(
            adjusted_abs_error
            <
            context_abs_error
        )
    )

    adjusted_range_violations = int(
        (
            (
                oos[
                    "coach_adjusted_pass_expectation"
                ]
                < 0
            )
            |
            (
                oos[
                    "coach_adjusted_pass_expectation"
                ]
                > 1
            )
        ).sum()
    )

    cold_nonzero = int(
        (
            oos[
                "prior_pbp_regime_games"
            ].eq(0)
            &
            oos[
                "coach_pass_rate_adjustment"
            ].abs().gt(1e-15)
        ).sum()
    )

    print(
        f"COACH_WEIGHT={COACH_WEIGHT:.2f}"
    )
    print(
        f"CONTEXT_ONLY_MAE={context_mae:.8f}"
    )
    print(
        f"COACH_ADJUSTED_MAE={adjusted_mae:.8f}"
    )
    print(
        f"COACH_ADJUSTED_DELTA_MAE={delta_mae:.8f}"
    )
    print(
        f"CONTEXT_ONLY_RMSE={context_rmse:.8f}"
    )
    print(
        f"COACH_ADJUSTED_RMSE={adjusted_rmse:.8f}"
    )
    print(
        f"COACH_ADJUSTED_DELTA_RMSE={delta_rmse:.8f}"
    )
    print(
        f"CONTEXT_ONLY_CORR={context_corr:.8f}"
    )
    print(
        f"COACH_ADJUSTED_CORR={adjusted_corr:.8f}"
    )
    print(
        "COACH_ADJUSTED_PCT_ROWS_IMPROVED="
        f"{pct_rows_improved:.8f}"
    )
    print(
        "COACH_ADJUSTED_RAW_RANGE_VIOLATIONS="
        f"{adjusted_range_violations}"
    )
    print(
        "COACH_ADJUSTED_CLIPPING_APPLIED=FALSE"
    )
    print(
        "COLD_START_NONZERO_ADJUSTMENT_ROWS="
        f"{cold_nonzero}"
    )

    if cold_nonzero:
        fail(
            "Cold-start coach adjustment violated."
        )

    # -------------------------------------------------------------------------
    section(
        "9. ALPHA SELECTION DISTRIBUTION"
    )

    alpha_counts = (
        oos[
            "selected_alpha"
        ]
        .value_counts()
        .sort_index()
    )

    for alpha, count in alpha_counts.items():
        print(
            "SELECTED_ALPHA_"
            f"{float(alpha):g}="
            f"{int(count)}"
        )

    # -------------------------------------------------------------------------
    section(
        "10. SEASON ROBUSTNESS"
    )

    season_col = (
        "season_coach"
        if "season_coach" in oos.columns
        else "season"
    )

    seasons = sorted(
        pd.to_numeric(
            oos[
                season_col
            ],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    season_improved_count = 0

    for season in seasons:

        s = oos[
            pd.to_numeric(
                oos[
                    season_col
                ],
                errors="raise",
            )
            .astype(int)
            .eq(
                season
            )
        ].copy()

        sy = s[
            "current_observed_pass_rate"
        ].to_numpy(
            dtype=float
        )

        sb = s[
            "context_pass_expectation"
        ].to_numpy(
            dtype=float
        )

        sa = s[
            "coach_adjusted_pass_expectation"
        ].to_numpy(
            dtype=float
        )

        b_mae = safe_mae(
            sy,
            sb,
        )

        a_mae = safe_mae(
            sy,
            sa,
        )

        delta = (
            a_mae
            -
            b_mae
        )

        improved = (
            delta < 0
        )

        season_improved_count += int(
            improved
        )

        print(
            f"SEASON_{season}_N={len(s)}"
        )
        print(
            f"SEASON_{season}_CONTEXT_MAE="
            f"{b_mae:.8f}"
        )
        print(
            f"SEASON_{season}_COACH_ADJUSTED_MAE="
            f"{a_mae:.8f}"
        )
        print(
            f"SEASON_{season}_DELTA_MAE="
            f"{delta:.8f}"
        )
        print(
            f"SEASON_{season}_IMPROVED="
            f"{str(improved).upper()}"
        )

    # -------------------------------------------------------------------------
    section(
        "11. COACH REGIME ROBUSTNESS"
    )

    regime_results = []

    for regime, g in oos.groupby(
        "team_coach_regime_key",
        sort=True,
    ):

        if len(g) < 5:
            continue

        gy = g[
            "current_observed_pass_rate"
        ].to_numpy(
            dtype=float
        )

        gb = g[
            "context_pass_expectation"
        ].to_numpy(
            dtype=float
        )

        ga = g[
            "coach_adjusted_pass_expectation"
        ].to_numpy(
            dtype=float
        )

        delta = (
            safe_mae(
                gy,
                ga,
            )
            -
            safe_mae(
                gy,
                gb,
            )
        )

        regime_results.append(
            delta
        )

    regime_results = np.asarray(
        regime_results,
        dtype=float,
    )

    regime_improved = int(
        np.sum(
            regime_results < 0
        )
    )

    regime_count = int(
        len(regime_results)
    )

    regime_improvement_rate = (
        float(
            regime_improved
            /
            regime_count
        )
        if regime_count
        else float("nan")
    )

    regime_median_delta = (
        float(
            np.median(
                regime_results
            )
        )
        if regime_count
        else float("nan")
    )

    print(
        f"REGIMES_EVALUATED={regime_count}"
    )
    print(
        f"REGIMES_IMPROVED={regime_improved}"
    )
    print(
        "REGIME_IMPROVEMENT_RATE="
        f"{regime_improvement_rate:.8f}"
    )
    print(
        "REGIME_MEDIAN_DELTA_MAE="
        f"{regime_median_delta:.8f}"
    )

    # -------------------------------------------------------------------------
    section(
        "12. DECISION GATE"
    )

    baseline_maes = [
        r["mae"]
        for r in baseline_results.values()
        if np.isfinite(
            r["mae"]
        )
    ]

    if baseline_maes:
        best_simple_baseline_mae = min(
            baseline_maes
        )

        context_beats_best_simple = (
            context_mae
            <
            best_simple_baseline_mae
        )

    else:
        best_simple_baseline_mae = float(
            "nan"
        )

        context_beats_best_simple = False

    coach_improves_context_mae = (
        delta_mae < 0
    )

    coach_improves_context_rmse = (
        delta_rmse < 0
    )

    strict_temporal_pass = (
        temporal_violations == 0
    )

    context_model_research_pass = (
        strict_temporal_pass
        and
        context_beats_best_simple
    )

    coach_incremental_shadow_pass = (
        strict_temporal_pass
        and
        coach_improves_context_mae
        and
        coach_improves_context_rmse
    )

    print(
        "BEST_SIMPLE_PREGAME_BASELINE_MAE="
        f"{best_simple_baseline_mae:.8f}"
    )
    print(
        "CONTEXT_BEATS_BEST_SIMPLE_BASELINE="
        f"{str(context_beats_best_simple).upper()}"
    )
    print(
        "COACH_IMPROVES_CONTEXT_MAE="
        f"{str(coach_improves_context_mae).upper()}"
    )
    print(
        "COACH_IMPROVES_CONTEXT_RMSE="
        f"{str(coach_improves_context_rmse).upper()}"
    )
    print(
        "STRICT_TEMPORAL_VALIDATION_PASS="
        f"{str(strict_temporal_pass).upper()}"
    )
    print(
        "PREGAME_CONTEXT_MODEL_RESEARCH_PASS="
        f"{str(context_model_research_pass).upper()}"
    )
    print(
        "COACH_INCREMENTAL_SHADOW_PASS="
        f"{str(coach_incremental_shadow_pass).upper()}"
    )

    production_ready = (
        context_model_research_pass
        and
        coach_incremental_shadow_pass
        and
        context_range_violations == 0
        and
        adjusted_range_violations == 0
    )

    print(
        "PRODUCTION_READY="
        f"{str(production_ready).upper()}"
    )

    # -------------------------------------------------------------------------
    section(
        "13. PRODUCTION FIREWALL"
    )

    print(
        "ANALYSIS_ONLY=TRUE"
    )
    print(
        "ARTIFACT_WRITE=FALSE"
    )
    print(
        "DATABASE_WRITE=FALSE"
    )
    print(
        "CODE_MUTATION=FALSE"
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
    print(
        "PRODUCTION_INFLUENCE=FALSE"
    )
    print(
        "PRODUCTION_FIREWALL=PASS"
    )

    # -------------------------------------------------------------------------
    section(
        "14. STAGE26G-J FINAL CONTRACT"
    )

    print(
        "STAGE26G_J_CONTRACT="
        "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_VALIDATION_V1"
    )
    print(
        "SOURCE_FEATURE_CONTRACT="
        "FROZEN_EXP004_CORE_V1"
    )
    print(
        "FROZEN_CORE_FEATURE_COUNT=76"
    )
    print(
        "LEGACY_COACH_FEATURES_EXCLUDED=8"
    )
    print(
        "PREGAME_CONTEXT_FEATURE_COUNT=68"
    )
    print(
        "TARGET_AUTHORITY="
        "STAGE26G_A_FROZEN_DECISION_UNIVERSE"
    )
    print(
        "DATE_BLOCKED_WALK_FORWARD=TRUE"
    )
    print(
        "TRAIN_DATE_STRICTLY_LT_TARGET_DATE=TRUE"
    )
    print(
        "SAME_DATE_TRAINING_ALLOWED=FALSE"
    )
    print(
        "FUTURE_DATE_TRAINING_ALLOWED=FALSE"
    )
    print(
        "COACH_SIGNAL="
        "STAGE26G_A_PRIOR_MEAN_PASS_OE"
    )
    print(
        f"COACH_WEIGHT={COACH_WEIGHT:.2f}"
    )
    print(
        "COLD_START_POLICY="
        "ZERO_COACH_ADJUSTMENT"
    )
    print(
        "REALIZED_CURRENT_GAME_XPASS_USED=FALSE"
    )
    print(
        "PREDICTION_CLIPPING_APPLIED=FALSE"
    )
    print(
        "ARTIFACT_WRITE=FALSE"
    )
    print(
        "DATABASE_WRITE=FALSE"
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
        "STAGE26G_J_PREGAME_CONTEXT_PASS_EXPECTATION_VALIDATION=PASS"
    )


if __name__ == "__main__":
    main()
