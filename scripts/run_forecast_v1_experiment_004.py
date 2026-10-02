#!/usr/bin/env python3

"""
WFS NFL FORECAST CENTER
Experiment 004 — Target-Specific Replacement Quality Ablation

PURPOSE
-------
Predict where the market is wrong rather than independently
relearning the entire NFL score.

Physical-game modeling unit:
    ONE row per game, anchored to the home-team row.

Targets:
    margin_residual =
        actual_home_margin - market_home_spread_raw

    total_residual =
        actual_total - market_total

Final forecast:
    WFS margin =
        market margin + predicted margin residual

    WFS total =
        market total + predicted total residual

    WFS home points =
        (WFS total + WFS margin) / 2

    WFS away points =
        (WFS total - WFS margin) / 2

MODELS
------
MARKET
WFS CORE residual
WFS CORE + IMPACT residual
WFS TARGET-SPECIFIC residual
    Margin: CORE + IMPACT + REPLACEMENT
    Total:  CORE + IMPACT

TIME SPLIT
----------
2023:
    inner training

2024:
    inner validation / ridge alpha selection

2023 + 2024:
    final fit

2025:
    untouched out-of-time evaluation

DEPENDENCIES
------------
pandas
numpy

NO sklearn.
NO joblib.
NO SQLite writes.
NO production files modified.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/home/mwynn/nfl_data_engine")

DATA_FILE = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_replacement_v1.csv"
)

EXP_DIR = (
    ROOT
    / "experiments"
    / "forecast_v1_exp004"
)

PREDICTIONS_FILE = (
    EXP_DIR
    / "forecast_v1_exp004_predictions_2025.csv"
)

METRICS_FILE = (
    EXP_DIR
    / "forecast_v1_exp004_metrics.json"
)

FEATURES_FILE = (
    EXP_DIR
    / "forecast_v1_exp004_features.json"
)


# =============================================================================
# CONSTANTS
# =============================================================================

INNER_TRAIN_SEASON = 2023
INNER_VALID_SEASON = 2024

FINAL_TRAIN_SEASONS = [
    2023,
    2024,
]

FINAL_TEST_SEASON = 2025

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
    3000.0,
]


# =============================================================================
# EXCLUSIONS
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

# Market spread and total are deliberately NOT predictors
# of their own residual targets in Experiment 004.
#
# The market is the baseline being corrected.
#
# Moneylines are also excluded to prevent the residual model
# from simply learning another representation of the same
# market opinion.

MARKET_COLUMNS = {
    "market_home_spread_raw",
    "market_total",
    "team_moneyline",
    "opponent_moneyline",
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def passed(name, detail=""):
    suffix = f" | {detail}" if detail else ""
    print(f"PASS | {name}{suffix}")


def fail(message):
    raise RuntimeError(message)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


# =============================================================================
# DETERMINISTIC RIDGE
# =============================================================================

class DeterministicRidge:

    def __init__(self, alpha):
        self.alpha = float(alpha)

        self.columns_ = None
        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.keep_mask_ = None
        self.coef_ = None

    def _matrix(self, frame):
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

    def fit(self, X, y):
        self.columns_ = list(
            X.columns
        )

        matrix = self._matrix(X)

        y = np.asarray(
            y,
            dtype=float,
        )

        if len(matrix) != len(y):
            fail("X/y length mismatch.")

        with np.errstate(all="ignore"):
            medians = np.nanmedian(
                matrix,
                axis=0,
            )

        if np.isnan(medians).any():
            bad = [
                self.columns_[i]
                for i, value in enumerate(medians)
                if np.isnan(value)
            ]

            fail(
                "All-missing training features: "
                f"{bad}"
            )

        self.medians_ = medians

        matrix = np.where(
            np.isnan(matrix),
            medians,
            matrix,
        )

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
                "No usable nonconstant features."
            )

        self.means_ = means
        self.stds_ = stds
        self.keep_mask_ = keep

        matrix = matrix[:, keep]

        z = (
            matrix
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

        # Do not penalize intercept.
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
            coef = (
                np.linalg.pinv(lhs)
                @ rhs
            )

        if not np.isfinite(coef).all():
            fail(
                "Non-finite model coefficients."
            )

        self.coef_ = coef

        return self

    def predict(self, X):
        if self.coef_ is None:
            fail(
                "Model has not been fit."
            )

        matrix = self._matrix(X)

        matrix = np.where(
            np.isnan(matrix),
            self.medians_,
            matrix,
        )

        matrix = matrix[
            :,
            self.keep_mask_
        ]

        z = (
            matrix
            - self.means_[
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
                "Non-finite predictions."
            )

        return pred


# =============================================================================
# ONE PHYSICAL GAME PER ROW
# =============================================================================

def build_home_game_frame(df):

    is_home = pd.to_numeric(
        df["is_home"],
        errors="coerce",
    )

    home = df[
        is_home.eq(1)
    ].copy()

    if len(home) != 855:
        fail(
            f"Expected 855 home-game rows; found {len(home)}"
        )

    if home["game_id"].nunique() != 855:
        fail(
            "Home frame does not contain 855 unique games."
        )

    if home.duplicated(
        ["game_id"]
    ).any():
        fail(
            "Duplicate physical game in home frame."
        )

    home[
        "actual_home_margin"
    ] = pd.to_numeric(
        home["target_margin"],
        errors="raise",
    )

    home[
        "actual_total"
    ] = pd.to_numeric(
        home["target_total_points"],
        errors="raise",
    )

    home[
        "market_margin"
    ] = pd.to_numeric(
        home["market_home_spread_raw"],
        errors="raise",
    )

    home[
        "market_game_total"
    ] = pd.to_numeric(
        home["market_total"],
        errors="raise",
    )

    home[
        "margin_residual"
    ] = (
        home["actual_home_margin"]
        -
        home["market_margin"]
    )

    home[
        "total_residual"
    ] = (
        home["actual_total"]
        -
        home["market_game_total"]
    )

    return home


# =============================================================================
# FEATURE SETS
# =============================================================================

def build_feature_lists(frame):

    numeric_cols = (
        frame.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    impact_features = sorted(
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

    replacement_features = sorted(
        c
        for c in frame.columns
        if c.startswith(
            "replacement_"
        )
    )

    excluded = (
        TARGET_COLUMNS
        |
        IDENTITY_COLUMNS
        |
        MARKET_COLUMNS
        |
        {
            "actual_home_margin",
            "actual_total",
            "market_margin",
            "market_game_total",
            "margin_residual",
            "total_residual",
            "impact_source_timestamp_proven",
        }
    )

    core = []

    for col in numeric_cols:

        if col in excluded:
            continue

        if col in impact_features:
            continue

        if col in replacement_features:
            continue

        core.append(col)

    core = sorted(
        set(core)
    )

    core_impact = sorted(
        set(
            core
            + impact_features
        )
    )

    core_impact_replacement = sorted(
        set(
            core
            + impact_features
            + replacement_features
        )
    )

    return (
        core,
        impact_features,
        replacement_features,
        core_impact,
        core_impact_replacement,
    )


# =============================================================================
# ALPHA SELECTION
# =============================================================================

def select_alpha(
    frame,
    features,
    target,
    label,
):

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
        features
    ]

    y_train = pd.to_numeric(
        train[target],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    X_valid = valid[
        features
    ]

    y_valid = pd.to_numeric(
        valid[target],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    results = []

    for alpha in RIDGE_ALPHAS:

        model = DeterministicRidge(
            alpha
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_valid
        )

        results.append(
            {
                "alpha":
                    float(alpha),

                "mae":
                    mae(
                        y_valid,
                        pred,
                    ),

                "rmse":
                    rmse(
                        y_valid,
                        pred,
                    ),
            }
        )

    results.sort(
        key=lambda r: (
            r["mae"],
            r["alpha"],
        )
    )

    best = results[0]

    print()
    print(label)

    for row in results:

        selected = (
            " <-- SELECTED"
            if row["alpha"] == best["alpha"]
            else ""
        )

        print(
            f"  alpha={row['alpha']:>8.2f} | "
            f"MAE={row['mae']:.4f} | "
            f"RMSE={row['rmse']:.4f}"
            f"{selected}"
        )

    return (
        best["alpha"],
        results,
    )


# =============================================================================
# FINAL FIT
# =============================================================================

def fit_final(
    frame,
    features,
    target,
    alpha,
):

    train = frame[
        frame["season"].isin(
            FINAL_TRAIN_SEASONS
        )
    ]

    model = DeterministicRidge(
        alpha
    )

    model.fit(
        train[features],
        pd.to_numeric(
            train[target],
            errors="raise",
        ).to_numpy(
            dtype=float,
        ),
    )

    return model


# =============================================================================
# FORECAST RECONSTRUCTION
# =============================================================================

def build_prediction_frame(
    test,
    model_name,
    margin_residual_pred,
    total_residual_pred,
):

    out = test[
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

    out = out.rename(
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

            "target_total_points":
                "actual_total_points",
        }
    )

    market_margin = pd.to_numeric(
        out["market_home_spread_raw"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    market_total = pd.to_numeric(
        out["market_total"],
        errors="raise",
    ).to_numpy(
        dtype=float,
    )

    out[
        "pred_margin_residual"
    ] = np.asarray(
        margin_residual_pred,
        dtype=float,
    )

    out[
        "pred_total_residual"
    ] = np.asarray(
        total_residual_pred,
        dtype=float,
    )

    out[
        "pred_home_margin"
    ] = (
        market_margin
        +
        out[
            "pred_margin_residual"
        ].to_numpy(dtype=float)
    )

    out[
        "pred_total_points"
    ] = (
        market_total
        +
        out[
            "pred_total_residual"
        ].to_numpy(dtype=float)
    )

    out[
        "pred_home_points"
    ] = (
        out["pred_total_points"]
        +
        out["pred_home_margin"]
    ) / 2.0

    out[
        "pred_away_points"
    ] = (
        out["pred_total_points"]
        -
        out["pred_home_margin"]
    ) / 2.0

    out[
        "actual_winner"
    ] = np.where(
        out["actual_home_margin"] > 0,
        out["home_team"],
        np.where(
            out["actual_home_margin"] < 0,
            out["away_team"],
            "TIE",
        ),
    )

    out[
        "pred_winner"
    ] = np.where(
        out["pred_home_margin"] > 0,
        out["home_team"],
        np.where(
            out["pred_home_margin"] < 0,
            out["away_team"],
            "TIE",
        ),
    )

    out["model"] = model_name

    return out


def build_market_prediction_frame(test):

    zeros = np.zeros(
        len(test),
        dtype=float,
    )

    return build_prediction_frame(
        test,
        "MARKET",
        zeros,
        zeros,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(frame):

    if len(frame) != 285:
        fail(
            f"Expected 285 test games; found {len(frame)}"
        )

    actual_home = pd.to_numeric(
        frame["actual_home_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    actual_away = pd.to_numeric(
        frame["actual_away_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    pred_home = pd.to_numeric(
        frame["pred_home_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    pred_away = pd.to_numeric(
        frame["pred_away_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    actual_margin = pd.to_numeric(
        frame["actual_home_margin"],
        errors="raise",
    ).to_numpy(dtype=float)

    pred_margin = pd.to_numeric(
        frame["pred_home_margin"],
        errors="raise",
    ).to_numpy(dtype=float)

    actual_total = pd.to_numeric(
        frame["actual_total_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    pred_total = pd.to_numeric(
        frame["pred_total_points"],
        errors="raise",
    ).to_numpy(dtype=float)

    score_abs = (
        np.abs(
            actual_home - pred_home
        )
        +
        np.abs(
            actual_away - pred_away
        )
    ) / 2.0

    score_sq = (
        (
            actual_home - pred_home
        ) ** 2
        +
        (
            actual_away - pred_away
        ) ** 2
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
            int(len(frame)),

        "winner_games_excluding_ties":
            int(non_tie.sum()),

        "ties_excluded":
            int((~non_tie).sum()),

        "team_score_mae_game_averaged":
            float(
                np.mean(score_abs)
            ),

        "team_score_rmse_game_averaged":
            float(
                np.sqrt(
                    np.mean(score_sq)
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
        "WFS NFL FORECAST CENTER — EXPERIMENT 004"
    )
    print("=" * 110)
    print(
        "GAME-LEVEL MARKET RESIDUAL MODEL"
    )
    print(
        "MARKET vs CORE vs CORE + IMPACT vs CORE + IMPACT + REPLACEMENT"
    )
    print(
        "2023 -> 2024 INNER VALIDATION"
    )
    print(
        "2023+2024 FINAL TRAIN / 2025 TEST"
    )
    print(
        "NO SQLITE WRITES / NO PRODUCTION MODIFICATION"
    )

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. LOAD
    # -------------------------------------------------------------------------

    section("1. LOAD")

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

    if df["game_id"].nunique() != 855:
        fail(
            "Expected 855 physical games."
        )

    passed(
        "validated source dataset",
        "1710 rows / 855 games",
    )

    # -------------------------------------------------------------------------
    # 2. GAME-LEVEL FRAME
    # -------------------------------------------------------------------------

    section("2. ONE PHYSICAL GAME PER ROW")

    games = build_home_game_frame(
        df
    )

    print(
        f"Physical game rows: {len(games)}"
    )

    print(
        games.groupby("season")
        .size()
        .to_string()
    )

    if not (
        games.groupby("season")
        .size()
        .reindex(
            [2023, 2024, 2025]
        )
        .eq(285)
        .all()
    ):
        fail(
            "Expected 285 games in each season."
        )

    passed(
        "one-row-per-game modeling unit"
    )

    # -------------------------------------------------------------------------
    # 3. RESIDUAL TARGET INTEGRITY
    # -------------------------------------------------------------------------

    section("3. MARKET RESIDUAL TARGETS")

    expected_margin_residual = (
        games["actual_home_margin"]
        -
        games["market_margin"]
    )

    expected_total_residual = (
        games["actual_total"]
        -
        games["market_game_total"]
    )

    if not np.allclose(
        games["margin_residual"],
        expected_margin_residual,
    ):
        fail(
            "Margin residual identity failure."
        )

    if not np.allclose(
        games["total_residual"],
        expected_total_residual,
    ):
        fail(
            "Total residual identity failure."
        )

    print(
        "Margin residual:"
    )
    print(
        games["margin_residual"]
        .describe()
        .to_string()
    )

    print()
    print(
        "Total residual:"
    )
    print(
        games["total_residual"]
        .describe()
        .to_string()
    )

    passed(
        "market residual identities"
    )

    # -------------------------------------------------------------------------
    # 4. FEATURE SETS
    # -------------------------------------------------------------------------

    section("4. FEATURE SETS")

    (
        core_features,
        impact_features,
        replacement_features,
        core_impact_features,
        core_impact_replacement_features,
    ) = build_feature_lists(
        games
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

    print(
        f"Replacement V1       : {len(replacement_features)}"
    )

    print(
        "CORE + Impact + Replacement: "
        f"{len(core_impact_replacement_features)}"
    )

    if len(impact_features) != 60:
        fail(
            "Expected 60 Impact features; "
            f"found {len(impact_features)}"
        )

    if len(replacement_features) != 33:
        fail(
            "Expected 33 Replacement features; "
            f"found {len(replacement_features)}"
        )

    forbidden = (
        MARKET_COLUMNS
        &
        set(core_features)
    )

    if forbidden:
        fail(
            "Market predictors leaked into residual CORE: "
            f"{sorted(forbidden)}"
        )

    if (
        set(core_features)
        &
        set(impact_features)
    ):
        fail(
            "Impact fields leaked into CORE."
        )

    if (
        set(core_features)
        &
        set(replacement_features)
    ):
        fail(
            "Replacement fields leaked into CORE."
        )

    if (
        set(core_impact_features)
        &
        set(replacement_features)
    ):
        fail(
            "Replacement fields leaked into CORE + Impact."
        )

    passed(
        "market variables excluded from residual predictors"
    )

    passed(
        "CORE / Impact / Replacement ablations isolated"
    )

    # -------------------------------------------------------------------------
    # 5. SPLIT
    # -------------------------------------------------------------------------

    section("5. CHRONOLOGICAL SPLIT")

    for season in [
        2023,
        2024,
        2025,
    ]:

        count = int(
            games[
                games["season"].eq(
                    season
                )
            ]["game_id"].nunique()
        )

        print(
            f"{season}: {count} games"
        )

        if count != 285:
            fail(
                f"Unexpected game count for {season}"
            )

    passed(
        "2023 -> 2024 -> 2025 chronology"
    )

    # -------------------------------------------------------------------------
    # 6. ALPHA SELECTION
    # -------------------------------------------------------------------------

    section(
        "6. 2023 -> 2024 ALPHA SELECTION"
    )

    selections = {}

    model_specs = [
        (
            "CORE_MARGIN",
            core_features,
            "margin_residual",
        ),
        (
            "CORE_TOTAL",
            core_features,
            "total_residual",
        ),
        (
            "IMPACT_MARGIN",
            core_impact_features,
            "margin_residual",
        ),
        (
            "IMPACT_TOTAL",
            core_impact_features,
            "total_residual",
        ),
        (
            "REPLACEMENT_MARGIN",
            core_impact_replacement_features,
            "margin_residual",
        ),
    ]

    for (
        label,
        features,
        target,
    ) in model_specs:

        alpha, results = select_alpha(
            games,
            features,
            target,
            label,
        )

        selections[label] = {
            "alpha":
                float(alpha),

            "results":
                results,
        }

    # -------------------------------------------------------------------------
    # 7. FINAL FIT
    # -------------------------------------------------------------------------

    section(
        "7. FINAL 2023 + 2024 FIT"
    )

    core_margin_model = fit_final(
        games,
        core_features,
        "margin_residual",
        selections[
            "CORE_MARGIN"
        ]["alpha"],
    )

    core_total_model = fit_final(
        games,
        core_features,
        "total_residual",
        selections[
            "CORE_TOTAL"
        ]["alpha"],
    )

    impact_margin_model = fit_final(
        games,
        core_impact_features,
        "margin_residual",
        selections[
            "IMPACT_MARGIN"
        ]["alpha"],
    )

    impact_total_model = fit_final(
        games,
        core_impact_features,
        "total_residual",
        selections[
            "IMPACT_TOTAL"
        ]["alpha"],
    )

    replacement_margin_model = fit_final(
        games,
        core_impact_replacement_features,
        "margin_residual",
        selections[
            "REPLACEMENT_MARGIN"
        ]["alpha"],
    )

    passed(
        "five final residual models fit"
    )

    # -------------------------------------------------------------------------
    # 8. 2025 TEST
    # -------------------------------------------------------------------------

    section(
        "8. 2025 OUT-OF-TIME FORECAST"
    )

    test = games[
        games["season"].eq(2025)
    ].copy()

    if len(test) != 285:
        fail(
            "Expected 285 2025 games."
        )

    market_pred = (
        build_market_prediction_frame(
            test
        )
    )

    core_margin_resid = (
        core_margin_model.predict(
            test[core_features]
        )
    )

    core_total_resid = (
        core_total_model.predict(
            test[core_features]
        )
    )

    impact_margin_resid = (
        impact_margin_model.predict(
            test[
                core_impact_features
            ]
        )
    )

    impact_total_resid = (
        impact_total_model.predict(
            test[
                core_impact_features
            ]
        )
    )

    replacement_margin_resid = (
        replacement_margin_model.predict(
            test[
                core_impact_replacement_features
            ]
        )
    )

    core_pred = build_prediction_frame(
        test,
        "WFS_CORE_RESIDUAL",
        core_margin_resid,
        core_total_resid,
    )

    impact_pred = build_prediction_frame(
        test,
        "WFS_CORE_IMPACT_RESIDUAL",
        impact_margin_resid,
        impact_total_resid,
    )

    target_specific_pred = build_prediction_frame(
        test,
        "WFS_TARGET_SPECIFIC_RESIDUAL",
        replacement_margin_resid,
        impact_total_resid,
    )

    passed(
        "285 physical-game predictions generated for all model variants"
    )

    # -------------------------------------------------------------------------
    # 9. METRICS
    # -------------------------------------------------------------------------

    section(
        "9. 2025 OUT-OF-TIME RESULTS"
    )

    metrics = {
        "MARKET":
            evaluate(
                market_pred
            ),

        "WFS_CORE_RESIDUAL":
            evaluate(
                core_pred
            ),

        "WFS_CORE_IMPACT_RESIDUAL":
            evaluate(
                impact_pred
            ),

        "WFS_TARGET_SPECIFIC_RESIDUAL":
            evaluate(
                target_specific_pred
            ),
    }

    header = (
        f"{'MODEL':<28}"
        f"{'SCORE MAE':>12}"
        f"{'SCORE RMSE':>13}"
        f"{'MARGIN MAE':>13}"
        f"{'MARGIN RMSE':>14}"
        f"{'TOTAL MAE':>12}"
        f"{'TOTAL RMSE':>13}"
        f"{'WIN ACC':>11}"
    )

    print(header)
    print("-" * len(header))

    for name in [
        "MARKET",
        "WFS_CORE_RESIDUAL",
        "WFS_CORE_IMPACT_RESIDUAL",
        "WFS_TARGET_SPECIFIC_RESIDUAL",
    ]:

        row = metrics[name]

        print(
            f"{name:<28}"
            f"{row['team_score_mae_game_averaged']:>12.4f}"
            f"{row['team_score_rmse_game_averaged']:>13.4f}"
            f"{row['margin_mae']:>13.4f}"
            f"{row['margin_rmse']:>14.4f}"
            f"{row['total_mae']:>12.4f}"
            f"{row['total_rmse']:>13.4f}"
            f"{row['winner_accuracy']:>11.4%}"
        )

    # -------------------------------------------------------------------------
    # 10. IMPACT ABLATION
    # -------------------------------------------------------------------------

    section(
        "10. IMPACT V1 RESIDUAL ABLATION"
    )

    core = metrics[
        "WFS_CORE_RESIDUAL"
    ]

    impact = metrics[
        "WFS_CORE_IMPACT_RESIDUAL"
    ]

    impact_ablation = {}

    for metric in [
        "team_score_mae_game_averaged",
        "team_score_rmse_game_averaged",
        "margin_mae",
        "margin_rmse",
        "total_mae",
        "total_rmse",
    ]:

        a = core[metric]
        b = impact[metric]

        pct = (
            (a - b)
            / a
            * 100.0
        )

        impact_ablation[
            metric
        ] = {
            "core":
                float(a),

            "core_impact":
                float(b),

            "improvement_pct":
                float(pct),
        }

        label = (
            "IMPROVED"
            if b < a
            else
            "WORSE"
            if b > a
            else
            "TIED"
        )

        print(
            f"{metric:<38} "
            f"CORE={a:>9.4f} | "
            f"IMPACT={b:>9.4f} | "
            f"{label:<8} | "
            f"{pct:>8.3f}%"
        )

    core_acc = core[
        "winner_accuracy"
    ]

    impact_acc = impact[
        "winner_accuracy"
    ]

    print()

    print(
        f"{'winner_accuracy':<38} "
        f"CORE={core_acc:>9.4%} | "
        f"IMPACT={impact_acc:>9.4%} | "
        f"delta={impact_acc-core_acc:>+.4%}"
    )

    # -------------------------------------------------------------------------
    # 10B. REPLACEMENT QUALITY ABLATION
    # -------------------------------------------------------------------------

    section(
        "10B. TARGET-SPECIFIC REPLACEMENT V1 ABLATION"
    )

    target_specific = metrics[
        "WFS_TARGET_SPECIFIC_RESIDUAL"
    ]

    replacement_ablation = {}

    for metric in [
        "team_score_mae_game_averaged",
        "team_score_rmse_game_averaged",
        "margin_mae",
        "margin_rmse",
        "total_mae",
        "total_rmse",
    ]:

        a = impact[metric]
        b = target_specific[metric]

        pct = (
            (a - b)
            / a
            * 100.0
        )

        replacement_ablation[
            metric
        ] = {
            "core_impact":
                float(a),

            "target_specific":
                float(b),

            "improvement_pct":
                float(pct),
        }

        label = (
            "IMPROVED"
            if b < a
            else
            "WORSE"
            if b > a
            else
            "TIED"
        )

        print(
            f"{metric:<38} "
            f"IMPACT={a:>9.4f} | "
            f"TARGET={b:>9.4f} | "
            f"{label:<8} | "
            f"{pct:>8.3f}%"
        )

    target_specific_acc = target_specific[
        "winner_accuracy"
    ]

    print()

    print(
        f"{'winner_accuracy':<38} "
        f"IMPACT={impact_acc:>9.4%} | "
        f"TARGET={target_specific_acc:>9.4%} | "
        f"delta={target_specific_acc-impact_acc:>+.4%}"
    )

    # -------------------------------------------------------------------------
    # 11. WFS VS MARKET
    # -------------------------------------------------------------------------

    section(
        "11. WFS RESIDUAL VS MARKET"
    )

    market = metrics["MARKET"]

    vs_market = {}

    for model_name in [
        "WFS_CORE_RESIDUAL",
        "WFS_CORE_IMPACT_RESIDUAL",
        "WFS_TARGET_SPECIFIC_RESIDUAL",
    ]:

        vs_market[
            model_name
        ] = {}

        print()
        print(model_name)

        for metric in [
            "team_score_mae_game_averaged",
            "margin_mae",
            "total_mae",
        ]:

            m = market[metric]
            w = metrics[
                model_name
            ][metric]

            pct = (
                (m - w)
                / m
                * 100.0
            )

            vs_market[
                model_name
            ][metric] = {
                "market":
                    float(m),

                "wfs":
                    float(w),

                "improvement_pct":
                    float(pct),
            }

            label = (
                "WFS BETTER"
                if w < m
                else
                "MARKET BETTER"
                if w > m
                else
                "TIED"
            )

            print(
                f"  {metric:<36} "
                f"MARKET={m:>9.4f} | "
                f"WFS={w:>9.4f} | "
                f"{label:<13} | "
                f"{pct:>8.3f}%"
            )

        market_acc = market[
            "winner_accuracy"
        ]

        wfs_acc = metrics[
            model_name
        ]["winner_accuracy"]

        print(
            f"  {'winner_accuracy':<36} "
            f"MARKET={market_acc:>9.4%} | "
            f"WFS={wfs_acc:>9.4%} | "
            f"delta={wfs_acc-market_acc:>+.4%}"
        )

    # -------------------------------------------------------------------------
    # 12. RESIDUAL MAGNITUDE
    # -------------------------------------------------------------------------

    section(
        "12. PREDICTED MARKET CORRECTION MAGNITUDE"
    )

    for name, frame in [
        (
            "WFS_CORE_RESIDUAL",
            core_pred,
        ),
        (
            "WFS_CORE_IMPACT_RESIDUAL",
            impact_pred,
        ),
        (
            "WFS_TARGET_SPECIFIC_RESIDUAL",
            target_specific_pred,
        ),
    ]:

        print()
        print(name)

        margin_shift = (
            frame[
                "pred_margin_residual"
            ].abs()
        )

        total_shift = (
            frame[
                "pred_total_residual"
            ].abs()
        )

        print(
            f"  Mean |margin correction| : "
            f"{margin_shift.mean():.4f}"
        )

        print(
            f"  Median |margin correction|: "
            f"{margin_shift.median():.4f}"
        )

        print(
            f"  Max |margin correction|  : "
            f"{margin_shift.max():.4f}"
        )

        print(
            f"  Mean |total correction|  : "
            f"{total_shift.mean():.4f}"
        )

        print(
            f"  Median |total correction|: "
            f"{total_shift.median():.4f}"
        )

        print(
            f"  Max |total correction|   : "
            f"{total_shift.max():.4f}"
        )

    # -------------------------------------------------------------------------
    # 13. EXPORT
    # -------------------------------------------------------------------------

    section("13. EXPORT")

    predictions = pd.concat(
        [
            market_pred,
            core_pred,
            impact_pred,
            target_specific_pred,
        ],
        ignore_index=True,
    )

    predictions = (
        predictions.sort_values(
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

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    features_payload = {
        "experiment":
            "forecast_v1_exp004",

        "architecture":
            "game_level_market_residual_target_specific",

        "target_specific_margin_features":
            "core_impact_replacement",

        "target_specific_total_features":
            "core_impact",

        "modeling_unit":
            "one_physical_game_home_row",

        "inner_train":
            2023,

        "inner_validation":
            2024,

        "final_train":
            [2023, 2024],

        "final_test":
            2025,

        "core_feature_count":
            len(core_features),

        "impact_feature_count":
            len(impact_features),

        "core_impact_feature_count":
            len(core_impact_features),

        "replacement_feature_count":
            len(replacement_features),

        "core_impact_replacement_feature_count":
            len(core_impact_replacement_features),

        "core_features":
            core_features,

        "impact_features":
            impact_features,

        "replacement_features":
            replacement_features,

        "core_impact_features":
            core_impact_features,

        "core_impact_replacement_features":
            core_impact_replacement_features,

        "market_predictors_excluded":
            sorted(
                MARKET_COLUMNS
            ),

        "alpha_selections":
            selections,
    }

    with FEATURES_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            features_payload,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    metrics_payload = {
        "experiment":
            "forecast_v1_exp004",

        "test_games":
            285,

        "test_season":
            2025,

        "metrics":
            metrics,

        "impact_ablation":
            impact_ablation,

        "replacement_ablation":
            replacement_ablation,

        "vs_market":
            vs_market,

        "alpha_selections":
            selections,
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
        "14. EXPERIMENT COMPLETE"
    )

    print(
        "FORECAST V1 EXPERIMENT 004: COMPLETE"
    )

    print()
    print(
        "No model deployed."
    )

    print(
        "No SQLite tables modified."
    )

    print(
        "No frozen Forecast artifacts modified."
    )

    print(
        "No NFL optimizer files modified."
    )

    print(
        "No production UI files modified."
    )


if __name__ == "__main__":
    main()
