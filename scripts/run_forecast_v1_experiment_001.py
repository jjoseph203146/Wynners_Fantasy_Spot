#!/usr/bin/env python3

"""
WFS NFL FORECAST CENTER
Experiment 001

PURPOSE
-------
Out-of-time comparison:

    MARKET
    vs
    WFS CORE
    vs
    WFS CORE + IMPACT V1

TRAINING DESIGN
---------------
2023:
    inner training

2024:
    inner validation for ridge alpha selection

2023 + 2024:
    final model fit

2025:
    untouched out-of-time test

DEPENDENCIES
------------
pandas
numpy

No sklearn.
No joblib.
No SQLite writes.
No production files modified.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/home/mwynn/nfl_data_engine")

DATA_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_v1.csv"
)

EXP_DIR = (
    ROOT
    / "experiments"
    / "forecast_v1_exp001"
)

PREDICTIONS_FILE = (
    EXP_DIR
    / "forecast_v1_exp001_predictions_2025.csv"
)

METRICS_FILE = (
    EXP_DIR
    / "forecast_v1_exp001_metrics.json"
)

FEATURES_FILE = (
    EXP_DIR
    / "forecast_v1_exp001_features.json"
)


# =============================================================================
# EXPERIMENT CONSTANTS
# =============================================================================

INNER_TRAIN_SEASON = 2023
INNER_VALID_SEASON = 2024

FINAL_TRAIN_SEASONS = [
    2023,
    2024,
]

FINAL_TEST_SEASON = 2025


# Fixed candidate grid.
#
# Selection occurs using 2023 -> 2024 only.
# 2025 is never used for hyperparameter selection.

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

TARGET_COLUMNS = {
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
}

IDENTITY_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "game_date",
    "weekday",
    "gametime",
    "team",
    "opponent_team",
    "coach",
    "opponent_coach",
    "team_qb_name",
    "opponent_qb_name",
    "roof",
    "surface",
}

# season is intentionally excluded as a predictor.
#
# week is retained because point in season is legitimate
# pregame context.
#
# is_home is retained.
#
# game_type is excluded from Experiment 001 because it is
# categorical and this initial baseline is numeric-only.


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def passed(name, detail=""):
    suffix = f" | {detail}" if detail else ""
    print(
        f"PASS | {name}{suffix}"
    )


def fail(message):
    raise RuntimeError(message)


# =============================================================================
# BASIC METRICS
# =============================================================================

def mae(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def rmse(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    - y_pred
                )
                ** 2
            )
        )
    )


# =============================================================================
# RIDGE MODEL
# =============================================================================

class DeterministicRidge:
    """
    Small deterministic ridge regression implementation.

    Behavior:
    - median imputation learned on training data only
    - mean/std scaling learned on training data only
    - zero-variance columns dropped
    - intercept is NOT penalized
    - deterministic NumPy linear algebra
    """

    def __init__(self, alpha):
        self.alpha = float(alpha)

        self.columns_ = None
        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.keep_mask_ = None
        self.coef_ = None

    def _to_numeric_matrix(self, frame):
        frame = frame[
            self.columns_
        ].copy()

        for col in self.columns_:
            frame[col] = pd.to_numeric(
                frame[col],
                errors="coerce",
            )

        return frame.to_numpy(
            dtype=float,
        )

    def fit(self, X, y):
        self.columns_ = list(
            X.columns
        )

        matrix = self._to_numeric_matrix(
            X
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        if len(matrix) != len(y):
            fail(
                "X/y row count mismatch."
            )

        # ---------------------------------------------------------
        # Training-only median imputation
        # ---------------------------------------------------------

        with np.errstate(
            all="ignore"
        ):
            medians = np.nanmedian(
                matrix,
                axis=0,
            )

        # A completely missing training feature is not usable.
        if np.isnan(medians).any():
            bad = [
                self.columns_[i]
                for i, value in enumerate(medians)
                if np.isnan(value)
            ]

            fail(
                "All-missing training feature(s): "
                f"{bad}"
            )

        self.medians_ = medians

        matrix = np.where(
            np.isnan(matrix),
            self.medians_,
            matrix,
        )

        # ---------------------------------------------------------
        # Training-only standardization
        # ---------------------------------------------------------

        means = np.mean(
            matrix,
            axis=0,
        )

        stds = np.std(
            matrix,
            axis=0,
            ddof=0,
        )

        keep = (
            np.isfinite(stds)
            &
            (stds > 1e-12)
        )

        if not keep.any():
            fail(
                "No usable non-constant features."
            )

        self.means_ = means
        self.stds_ = stds
        self.keep_mask_ = keep

        matrix = matrix[:, keep]

        means_kept = means[keep]
        stds_kept = stds[keep]

        z = (
            matrix
            - means_kept
        ) / stds_kept

        # ---------------------------------------------------------
        # Intercept column
        # ---------------------------------------------------------

        design = np.column_stack(
            [
                np.ones(
                    len(z),
                    dtype=float,
                ),
                z,
            ]
        )

        # ---------------------------------------------------------
        # Ridge
        # ---------------------------------------------------------

        penalty = np.eye(
            design.shape[1],
            dtype=float,
        )

        # Never penalize intercept.
        penalty[0, 0] = 0.0

        lhs = (
            design.T @ design
            +
            self.alpha * penalty
        )

        rhs = (
            design.T @ y
        )

        try:
            coef = np.linalg.solve(
                lhs,
                rhs,
            )

        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(
                lhs
            ) @ rhs

        if not np.isfinite(
            coef
        ).all():
            fail(
                "Non-finite ridge coefficients."
            )

        self.coef_ = coef

        return self

    def predict(self, X):
        if self.coef_ is None:
            fail(
                "Model must be fit before predict."
            )

        matrix = self._to_numeric_matrix(
            X
        )

        matrix = np.where(
            np.isnan(matrix),
            self.medians_,
            matrix,
        )

        matrix = matrix[
            :,
            self.keep_mask_
        ]

        means = self.means_[
            self.keep_mask_
        ]

        stds = self.stds_[
            self.keep_mask_
        ]

        z = (
            matrix
            - means
        ) / stds

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

        if not np.isfinite(
            pred
        ).all():
            fail(
                "Non-finite predictions generated."
            )

        return pred


# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================

def add_market_orientation(frame):
    """
    Add validated team-oriented market fields.

    WFS validated convention:

        market_home_spread_raw > 0
            home favored

        market_home_spread_raw < 0
            away favored

    Therefore:

        home row:
            team_market_spread = raw spread

        away row:
            team_market_spread = -raw spread
    """

    out = frame.copy()

    raw_spread = pd.to_numeric(
        out["market_home_spread_raw"],
        errors="coerce",
    )

    is_home = pd.to_numeric(
        out["is_home"],
        errors="coerce",
    )

    market_total = pd.to_numeric(
        out["market_total"],
        errors="coerce",
    )

    out[
        "team_market_spread"
    ] = np.where(
        is_home.eq(1),
        raw_spread,
        -raw_spread,
    )

    out[
        "team_market_implied_points"
    ] = (
        market_total
        +
        out["team_market_spread"]
    ) / 2.0

    out[
        "opponent_market_implied_points"
    ] = (
        market_total
        -
        out["team_market_spread"]
    ) / 2.0

    return out


def build_feature_lists(frame):
    """
    Initial Experiment 001 deliberately uses numeric
    pregame predictors only.

    CORE:
        all approved numeric pregame predictors
        except Impact V1

    CORE + IMPACT:
        CORE plus all 60 Impact V1 features

    Raw home spread is excluded and replaced by
    team_market_spread so orientation is explicit.
    """

    numeric_cols = (
        frame.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    impact_cols = sorted(
        c
        for c in frame.columns
        if (
            c.startswith(
                "team_impact_"
            )
            or
            c.startswith(
                "opp_impact_"
            )
        )
    )

    excluded = set(
        TARGET_COLUMNS
        |
        IDENTITY_COLUMNS
    )

    excluded.add(
        "market_home_spread_raw"
    )

    # Provenance is audit-only if it ever appears.
    excluded.add(
        "impact_source_timestamp_proven"
    )

    core = []

    for col in numeric_cols:
        if col in excluded:
            continue

        if col in impact_cols:
            continue

        core.append(col)

    # Derived market features were added after CSV load and
    # therefore may not be in the original numeric_cols list.

    for col in [
        "team_market_spread",
        "team_market_implied_points",
        "opponent_market_implied_points",
    ]:
        if col not in core:
            core.append(col)

    core = sorted(
        set(core)
    )

    core_impact = sorted(
        set(
            core
            + impact_cols
        )
    )

    return (
        core,
        impact_cols,
        core_impact,
    )


# =============================================================================
# ALPHA SELECTION
# =============================================================================

def select_alpha(
    frame,
    feature_columns,
    label,
):
    """
    Strict inner chronological selection:

        fit 2023
        validate 2024

    Criterion:
        team-points MAE
    """

    train = frame[
        frame["season"].eq(
            INNER_TRAIN_SEASON
        )
    ].copy()

    valid = frame[
        frame["season"].eq(
            INNER_VALID_SEASON
        )
    ].copy()

    X_train = train[
        feature_columns
    ]

    y_train = pd.to_numeric(
        train["target_team_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    X_valid = valid[
        feature_columns
    ]

    y_valid = pd.to_numeric(
        valid["target_team_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    results = []

    for alpha in RIDGE_ALPHAS:
        model = DeterministicRidge(
            alpha=alpha
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_valid
        )

        score_mae = mae(
            y_valid,
            pred,
        )

        score_rmse = rmse(
            y_valid,
            pred,
        )

        results.append(
            {
                "alpha":
                    float(alpha),

                "validation_team_score_mae":
                    float(score_mae),

                "validation_team_score_rmse":
                    float(score_rmse),
            }
        )

    results.sort(
        key=lambda row: (
            row[
                "validation_team_score_mae"
            ],
            row["alpha"],
        )
    )

    best = results[0]

    print()
    print(
        f"{label} alpha selection:"
    )

    for row in results:
        marker = (
            " <-- SELECTED"
            if row["alpha"] == best["alpha"]
            else ""
        )

        print(
            f"  alpha={row['alpha']:>8.2f} | "
            f"MAE={row['validation_team_score_mae']:.4f} | "
            f"RMSE={row['validation_team_score_rmse']:.4f}"
            f"{marker}"
        )

    return (
        best["alpha"],
        results,
    )


# =============================================================================
# FINAL MODEL FIT
# =============================================================================

def fit_final_model(
    frame,
    features,
    alpha,
):
    train = frame[
        frame["season"].isin(
            FINAL_TRAIN_SEASONS
        )
    ].copy()

    X_train = train[
        features
    ]

    y_train = pd.to_numeric(
        train["target_team_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    model = DeterministicRidge(
        alpha=alpha
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =============================================================================
# GAME-LEVEL RECONSTRUCTION
# =============================================================================

def build_game_predictions(
    test_rows,
    model_name,
    team_predictions,
):
    """
    Convert team-perspective predictions into one physical
    game row.

    Uses the actual home row as the game-level anchor.
    """

    temp = test_rows[
        [
            "game_id",
            "season",
            "week",
            "game_date",
            "team",
            "opponent_team",
            "is_home",
            "market_home_spread_raw",
            "market_total",
            "target_team_points",
            "target_opponent_points",
            "target_margin",
            "target_total_points",
            "target_result",
        ]
    ].copy()

    temp[
        "_prediction"
    ] = np.asarray(
        team_predictions,
        dtype=float,
    )

    home = temp[
        pd.to_numeric(
            temp["is_home"],
            errors="coerce",
        ).eq(1)
    ].copy()

    away = temp[
        pd.to_numeric(
            temp["is_home"],
            errors="coerce",
        ).eq(0)
    ].copy()

    home = home.rename(
        columns={
            "team":
                "home_team",

            "opponent_team":
                "away_team",

            "target_team_points":
                "actual_home_points",

            "target_opponent_points":
                "actual_away_points",

            "target_margin":
                "actual_home_margin",

            "_prediction":
                "pred_home_points",
        }
    )

    away = away.rename(
        columns={
            "team":
                "away_team_check",

            "opponent_team":
                "home_team_check",

            "_prediction":
                "pred_away_points",
        }
    )

    game = home.merge(
        away[
            [
                "game_id",
                "away_team_check",
                "home_team_check",
                "pred_away_points",
            ]
        ],
        left_on=[
            "game_id",
            "home_team",
            "away_team",
        ],
        right_on=[
            "game_id",
            "home_team_check",
            "away_team_check",
        ],
        how="left",
        validate="one_to_one",
    )

    if game[
        "pred_away_points"
    ].isna().any():
        fail(
            f"{model_name}: missing reverse away predictions."
        )

    game = game.drop(
        columns=[
            "away_team_check",
            "home_team_check",
        ]
    )

    game[
        "pred_home_margin"
    ] = (
        game["pred_home_points"]
        -
        game["pred_away_points"]
    )

    game[
        "pred_total_points"
    ] = (
        game["pred_home_points"]
        +
        game["pred_away_points"]
    )

    game[
        "pred_winner"
    ] = np.where(
        game["pred_home_margin"] > 0,
        game["home_team"],
        np.where(
            game["pred_home_margin"] < 0,
            game["away_team"],
            "TIE",
        ),
    )

    game[
        "actual_winner"
    ] = np.where(
        game["actual_home_margin"] > 0,
        game["home_team"],
        np.where(
            game["actual_home_margin"] < 0,
            game["away_team"],
            "TIE",
        ),
    )

    game[
        "model"
    ] = model_name

    return game


# =============================================================================
# MARKET BASELINE
# =============================================================================

def build_market_game_predictions(
    test_rows,
):
    """
    Physical-game market implied score baseline.

    Validated WFS spread convention:

        positive raw home spread = home favored

    Therefore:

        implied home =
            (total + spread) / 2

        implied away =
            (total - spread) / 2
    """

    home = test_rows[
        pd.to_numeric(
            test_rows["is_home"],
            errors="coerce",
        ).eq(1)
    ].copy()

    spread = pd.to_numeric(
        home[
            "market_home_spread_raw"
        ],
        errors="raise",
    )

    total = pd.to_numeric(
        home["market_total"],
        errors="raise",
    )

    home_pred = (
        total + spread
    ) / 2.0

    away_pred = (
        total - spread
    ) / 2.0

    temp = home[
        [
            "game_id",
            "season",
            "week",
            "game_date",
            "team",
            "opponent_team",
            "market_home_spread_raw",
            "market_total",
            "target_team_points",
            "target_opponent_points",
            "target_margin",
            "target_total_points",
        ]
    ].copy()

    temp = temp.rename(
        columns={
            "team":
                "home_team",

            "opponent_team":
                "away_team",

            "target_team_points":
                "actual_home_points",

            "target_opponent_points":
                "actual_away_points",

            "target_margin":
                "actual_home_margin",
        }
    )

    temp[
        "pred_home_points"
    ] = home_pred.to_numpy(
        dtype=float,
    )

    temp[
        "pred_away_points"
    ] = away_pred.to_numpy(
        dtype=float,
    )

    temp[
        "pred_home_margin"
    ] = (
        temp["pred_home_points"]
        -
        temp["pred_away_points"]
    )

    temp[
        "pred_total_points"
    ] = (
        temp["pred_home_points"]
        +
        temp["pred_away_points"]
    )

    temp[
        "pred_winner"
    ] = np.where(
        temp["pred_home_margin"] > 0,
        temp["home_team"],
        np.where(
            temp["pred_home_margin"] < 0,
            temp["away_team"],
            "TIE",
        ),
    )

    temp[
        "actual_winner"
    ] = np.where(
        temp["actual_home_margin"] > 0,
        temp["home_team"],
        np.where(
            temp["actual_home_margin"] < 0,
            temp["away_team"],
            "TIE",
        ),
    )

    temp[
        "model"
    ] = "MARKET"

    return temp


# =============================================================================
# PHYSICAL-GAME METRICS
# =============================================================================

def evaluate_games(game):
    if len(game) != 285:
        fail(
            f"Expected 285 test games; found {len(game)}"
        )

    actual_home = pd.to_numeric(
        game["actual_home_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    actual_away = pd.to_numeric(
        game["actual_away_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    pred_home = pd.to_numeric(
        game["pred_home_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    pred_away = pd.to_numeric(
        game["pred_away_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    actual_margin = pd.to_numeric(
        game["actual_home_margin"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    pred_margin = pd.to_numeric(
        game["pred_home_margin"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    actual_total = pd.to_numeric(
        game["target_total_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    pred_total = pd.to_numeric(
        game["pred_total_points"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    # Treat the physical game as the unit.
    #
    # Score MAE for a game =
    # mean(error home, error away).

    game_score_abs_error = (
        np.abs(
            actual_home - pred_home
        )
        +
        np.abs(
            actual_away - pred_away
        )
    ) / 2.0

    game_score_squared_error = (
        (
            actual_home - pred_home
        )
        ** 2
        +
        (
            actual_away - pred_away
        )
        ** 2
    ) / 2.0

    non_tie = (
        actual_margin != 0
    )

    winner_correct = (
        np.sign(
            pred_margin[non_tie]
        )
        ==
        np.sign(
            actual_margin[non_tie]
        )
    )

    return {
        "games":
            int(len(game)),

        "winner_games_excluding_ties":
            int(non_tie.sum()),

        "ties_excluded_from_winner_accuracy":
            int(
                (~non_tie).sum()
            ),

        "team_score_mae_game_averaged":
            float(
                np.mean(
                    game_score_abs_error
                )
            ),

        "team_score_rmse_game_averaged":
            float(
                np.sqrt(
                    np.mean(
                        game_score_squared_error
                    )
                )
            ),

        "margin_mae":
            mae(
                actual_margin,
                pred_margin,
            ),

        "margin_rmse":
            rmse(
                actual_margin,
                pred_margin,
            ),

        "total_mae":
            mae(
                actual_total,
                pred_total,
            ),

        "total_rmse":
            rmse(
                actual_total,
                pred_total,
            ),

        "winner_accuracy":
            float(
                np.mean(
                    winner_correct
                )
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "WFS NFL FORECAST CENTER — EXPERIMENT 001"
    )
    print("=" * 110)
    print(
        "MARKET vs WFS CORE vs WFS CORE + IMPACT V1"
    )
    print(
        "2023/2024 TRAINING — 2025 OUT-OF-TIME TEST"
    )
    print(
        "NUMPY DETERMINISTIC RIDGE"
    )
    print(
        "NO SQLITE WRITES / NO PRODUCTION FILE MODIFICATION"
    )

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------------------

    section(
        "1. LOAD VALIDATED FORECAST DATASET"
    )

    if not DATA_FILE.exists():
        fail(
            f"Missing dataset: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    if len(df) != 1710:
        fail(
            f"Expected 1710 rows; found {len(df)}"
        )

    if df[
        "game_id"
    ].nunique() != 855:
        fail(
            "Expected 855 physical games."
        )

    if df.duplicated(
        [
            "game_id",
            "team",
        ]
    ).any():
        fail(
            "Duplicate game/team keys detected."
        )

    df = add_market_orientation(
        df
    )

    passed(
        "validated dataset loaded",
        f"{len(df)} rows / {df['game_id'].nunique()} games",
    )

    # -------------------------------------------------------------------------
    # 2. SPLIT
    # -------------------------------------------------------------------------

    section(
        "2. CHRONOLOGICAL SPLIT"
    )

    inner_train = df[
        df["season"].eq(2023)
    ]

    inner_valid = df[
        df["season"].eq(2024)
    ]

    final_train = df[
        df["season"].isin(
            FINAL_TRAIN_SEASONS
        )
    ]

    final_test = df[
        df["season"].eq(
            FINAL_TEST_SEASON
        )
    ]

    print(
        f"2023 rows/games: "
        f"{len(inner_train)} / "
        f"{inner_train['game_id'].nunique()}"
    )

    print(
        f"2024 rows/games: "
        f"{len(inner_valid)} / "
        f"{inner_valid['game_id'].nunique()}"
    )

    print(
        f"Final train rows/games: "
        f"{len(final_train)} / "
        f"{final_train['game_id'].nunique()}"
    )

    print(
        f"2025 test rows/games: "
        f"{len(final_test)} / "
        f"{final_test['game_id'].nunique()}"
    )

    if (
        len(inner_train) != 570
        or
        len(inner_valid) != 570
        or
        len(final_train) != 1140
        or
        len(final_test) != 570
    ):
        fail(
            "Unexpected chronological split sizes."
        )

    passed(
        "chronological partitions"
    )

    # -------------------------------------------------------------------------
    # 3. FEATURES
    # -------------------------------------------------------------------------

    section(
        "3. FEATURE SETS"
    )

    (
        core_features,
        impact_features,
        core_impact_features,
    ) = build_feature_lists(
        df
    )

    print(
        f"CORE features        : {len(core_features)}"
    )

    print(
        f"Impact V1 features   : {len(impact_features)}"
    )

    print(
        f"CORE + Impact        : {len(core_impact_features)}"
    )

    if len(
        impact_features
    ) != 60:
        fail(
            "Expected exactly 60 Impact V1 features; "
            f"found {len(impact_features)}"
        )

    overlap = set(
        core_features
    ) & set(
        impact_features
    )

    if overlap:
        fail(
            f"CORE unexpectedly contains Impact features: "
            f"{sorted(overlap)}"
        )

    passed(
        "impact ablation isolated",
        "CORE contains zero Impact V1 fields",
    )

    # -------------------------------------------------------------------------
    # 4. INNER ALPHA SELECTION
    # -------------------------------------------------------------------------

    section(
        "4. 2023 -> 2024 RIDGE ALPHA SELECTION"
    )

    core_alpha, core_alpha_results = (
        select_alpha(
            df,
            core_features,
            "WFS CORE",
        )
    )

    impact_alpha, impact_alpha_results = (
        select_alpha(
            df,
            core_impact_features,
            "WFS CORE + IMPACT",
        )
    )

    print()
    print(
        f"Selected CORE alpha        : {core_alpha}"
    )

    print(
        f"Selected CORE+IMPACT alpha : {impact_alpha}"
    )

    # -------------------------------------------------------------------------
    # 5. FINAL FIT
    # -------------------------------------------------------------------------

    section(
        "5. FINAL 2023 + 2024 MODEL FIT"
    )

    core_model = fit_final_model(
        df,
        core_features,
        core_alpha,
    )

    impact_model = fit_final_model(
        df,
        core_impact_features,
        impact_alpha,
    )

    passed(
        "CORE final fit"
    )

    passed(
        "CORE + IMPACT final fit"
    )

    # -------------------------------------------------------------------------
    # 6. 2025 PREDICTION
    # -------------------------------------------------------------------------

    section(
        "6. 2025 OUT-OF-TIME PREDICTIONS"
    )

    core_team_pred = (
        core_model.predict(
            final_test[
                core_features
            ]
        )
    )

    impact_team_pred = (
        impact_model.predict(
            final_test[
                core_impact_features
            ]
        )
    )

    market_games = (
        build_market_game_predictions(
            final_test
        )
    )

    core_games = (
        build_game_predictions(
            final_test,
            "WFS_CORE",
            core_team_pred,
        )
    )

    impact_games = (
        build_game_predictions(
            final_test,
            "WFS_CORE_IMPACT",
            impact_team_pred,
        )
    )

    if not (
        len(market_games)
        ==
        len(core_games)
        ==
        len(impact_games)
        ==
        285
    ):
        fail(
            "Game reconstruction row count failure."
        )

    passed(
        "285 physical games reconstructed for all models"
    )

    # -------------------------------------------------------------------------
    # 7. METRICS
    # -------------------------------------------------------------------------

    section(
        "7. 2025 OUT-OF-TIME RESULTS"
    )

    metrics = {
        "MARKET":
            evaluate_games(
                market_games
            ),

        "WFS_CORE":
            evaluate_games(
                core_games
            ),

        "WFS_CORE_IMPACT":
            evaluate_games(
                impact_games
            ),
    }

    metric_order = [
        "team_score_mae_game_averaged",
        "team_score_rmse_game_averaged",
        "margin_mae",
        "margin_rmse",
        "total_mae",
        "total_rmse",
        "winner_accuracy",
    ]

    header = (
        f"{'MODEL':<20}"
        f"{'SCORE MAE':>12}"
        f"{'SCORE RMSE':>13}"
        f"{'MARGIN MAE':>13}"
        f"{'MARGIN RMSE':>14}"
        f"{'TOTAL MAE':>12}"
        f"{'TOTAL RMSE':>13}"
        f"{'WIN ACC':>11}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for model_name in [
        "MARKET",
        "WFS_CORE",
        "WFS_CORE_IMPACT",
    ]:

        row = metrics[
            model_name
        ]

        print(
            f"{model_name:<20}"
            f"{row['team_score_mae_game_averaged']:>12.4f}"
            f"{row['team_score_rmse_game_averaged']:>13.4f}"
            f"{row['margin_mae']:>13.4f}"
            f"{row['margin_rmse']:>14.4f}"
            f"{row['total_mae']:>12.4f}"
            f"{row['total_rmse']:>13.4f}"
            f"{row['winner_accuracy']:>11.4%}"
        )

    # -------------------------------------------------------------------------
    # 8. ABLATION
    # -------------------------------------------------------------------------

    section(
        "8. IMPACT V1 ABLATION"
    )

    core = metrics[
        "WFS_CORE"
    ]

    impact = metrics[
        "WFS_CORE_IMPACT"
    ]

    ablation = {}

    for metric in [
        "team_score_mae_game_averaged",
        "team_score_rmse_game_averaged",
        "margin_mae",
        "margin_rmse",
        "total_mae",
        "total_rmse",
    ]:

        before = core[
            metric
        ]

        after = impact[
            metric
        ]

        delta = (
            after - before
        )

        pct = (
            (
                before - after
            )
            /
            before
            *
            100.0
        )

        ablation[
            metric
        ] = {
            "core":
                float(before),

            "core_impact":
                float(after),

            "delta_core_impact_minus_core":
                float(delta),

            "improvement_pct":
                float(pct),
        }

        label = (
            "IMPROVED"
            if after < before
            else
            "WORSE"
            if after > before
            else
            "TIED"
        )

        print(
            f"{metric:<38} "
            f"CORE={before:>9.4f} | "
            f"IMPACT={after:>9.4f} | "
            f"{label:<8} | "
            f"{pct:>8.3f}%"
        )

    core_acc = core[
        "winner_accuracy"
    ]

    impact_acc = impact[
        "winner_accuracy"
    ]

    accuracy_delta = (
        impact_acc - core_acc
    )

    ablation[
        "winner_accuracy"
    ] = {
        "core":
            float(core_acc),

        "core_impact":
            float(impact_acc),

        "delta":
            float(accuracy_delta),
    }

    label = (
        "IMPROVED"
        if accuracy_delta > 0
        else
        "WORSE"
        if accuracy_delta < 0
        else
        "TIED"
    )

    print()
    print(
        f"{'winner_accuracy':<38} "
        f"CORE={core_acc:>9.4%} | "
        f"IMPACT={impact_acc:>9.4%} | "
        f"{label:<8} | "
        f"delta={accuracy_delta:>+.4%}"
    )

    # -------------------------------------------------------------------------
    # 9. WFS VS MARKET
    # -------------------------------------------------------------------------

    section(
        "9. WFS + IMPACT VS MARKET"
    )

    market = metrics[
        "MARKET"
    ]

    versus_market = {}

    for metric in [
        "team_score_mae_game_averaged",
        "margin_mae",
        "total_mae",
    ]:

        market_value = market[
            metric
        ]

        wfs_value = impact[
            metric
        ]

        improvement_pct = (
            (
                market_value
                -
                wfs_value
            )
            /
            market_value
            *
            100.0
        )

        versus_market[
            metric
        ] = {
            "market":
                float(market_value),

            "wfs_core_impact":
                float(wfs_value),

            "improvement_pct":
                float(improvement_pct),
        }

        label = (
            "WFS BETTER"
            if wfs_value < market_value
            else
            "MARKET BETTER"
            if wfs_value > market_value
            else
            "TIED"
        )

        print(
            f"{metric:<38} "
            f"MARKET={market_value:>9.4f} | "
            f"WFS={wfs_value:>9.4f} | "
            f"{label:<13} | "
            f"{improvement_pct:>8.3f}%"
        )

    print()

    market_acc = market[
        "winner_accuracy"
    ]

    wfs_acc = impact[
        "winner_accuracy"
    ]

    print(
        f"{'winner_accuracy':<38} "
        f"MARKET={market_acc:>9.4%} | "
        f"WFS={wfs_acc:>9.4%} | "
        f"delta={wfs_acc - market_acc:>+.4%}"
    )

    # -------------------------------------------------------------------------
    # 10. EXPORT PREDICTIONS
    # -------------------------------------------------------------------------

    section(
        "10. EXPORT"
    )

    prediction_frames = []

    for frame in [
        market_games,
        core_games,
        impact_games,
    ]:

        prediction_frames.append(
            frame[
                [
                    "game_id",
                    "season",
                    "week",
                    "game_date",
                    "home_team",
                    "away_team",
                    "model",
                    "market_home_spread_raw",
                    "market_total",
                    "actual_home_points",
                    "actual_away_points",
                    "pred_home_points",
                    "pred_away_points",
                    "actual_home_margin",
                    "pred_home_margin",
                    "target_total_points",
                    "pred_total_points",
                    "actual_winner",
                    "pred_winner",
                ]
            ].copy()
        )

    prediction_output = pd.concat(
        prediction_frames,
        axis=0,
        ignore_index=True,
    )

    prediction_output = (
        prediction_output.sort_values(
            [
                "game_date",
                "game_id",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    feature_payload = {
        "experiment":
            "forecast_v1_exp001",

        "dataset":
            str(DATA_FILE),

        "split": {
            "inner_train":
                2023,

            "inner_validation":
                2024,

            "final_train":
                [
                    2023,
                    2024,
                ],

            "final_test":
                2025,
        },

        "estimator":
            "deterministic_numpy_ridge",

        "ridge_alpha_candidates":
            RIDGE_ALPHAS,

        "selected_core_alpha":
            float(core_alpha),

        "selected_core_impact_alpha":
            float(impact_alpha),

        "core_feature_count":
            int(
                len(core_features)
            ),

        "impact_feature_count":
            int(
                len(impact_features)
            ),

        "core_impact_feature_count":
            int(
                len(core_impact_features)
            ),

        "core_features":
            core_features,

        "impact_features":
            impact_features,

        "core_impact_features":
            core_impact_features,
    }

    with FEATURES_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_payload,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    metrics_payload = {
        "experiment":
            "forecast_v1_exp001",

        "test_season":
            2025,

        "test_games":
            285,

        "ties_excluded_from_binary_winner_metrics":
            1,

        "metrics":
            metrics,

        "impact_ablation":
            ablation,

        "wfs_core_impact_vs_market":
            versus_market,

        "core_alpha_selection":
            core_alpha_results,

        "core_impact_alpha_selection":
            impact_alpha_results,
    }

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_payload,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print(
        "Predictions:"
    )
    print(
        PREDICTIONS_FILE
    )

    print()
    print(
        "Metrics:"
    )
    print(
        METRICS_FILE
    )

    print()
    print(
        "Features:"
    )
    print(
        FEATURES_FILE
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    section(
        "11. EXPERIMENT COMPLETE"
    )

    print(
        "FORECAST V1 EXPERIMENT 001: COMPLETE"
    )

    print()
    print(
        "No model was deployed."
    )

    print(
        "No SQLite tables were modified."
    )

    print(
        "No frozen Forecast artifacts were modified."
    )

    print(
        "No NFL optimizer or production UI files were modified."
    )


if __name__ == "__main__":
    main()
