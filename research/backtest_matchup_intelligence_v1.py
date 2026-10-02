from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")

SOURCE = (
    ROOT
    / "data/research/matchup_intelligence_history_v1.parquet"
)

PLAYER_FEATURES = [
    "usage_3g",
    "snap_pct_3g",
    "target_3g",
    "carry_3g",
    "fanduel_3g",
    "prior_player_games",
]

TEAM_FEATURES = [
    "team_history_games",
    "points_for_avg_3",
    "points_for_avg_5",
    "offensive_plays_avg_3",
    "offensive_plays_avg_5",
    "pass_attempts_avg_3",
    "pass_attempts_avg_5",
    "rush_attempts_avg_3",
    "rush_attempts_avg_5",
    "pass_rate_avg_3",
    "pass_rate_avg_5",
    "rush_rate_avg_3",
    "rush_rate_avg_5",
    "passing_yards_avg_3",
    "rushing_yards_avg_3",
    "passing_tds_avg_3",
    "rushing_tds_avg_3",
    "opponent_points_allowed_avg_3",
    "opponent_points_allowed_avg_5",
    "opponent_pass_yards_allowed_avg_3",
    "opponent_rush_yards_allowed_avg_3",
    "opponent_pass_tds_allowed_avg_3",
    "opponent_rush_tds_allowed_avg_3",
]

DVP_FEATURES = [
    "dvp_history_games",
    "fd_allowed_avg_3",
    "fd_allowed_avg_5",
    "targets_allowed_avg_3",
    "carries_allowed_avg_3",
    "receptions_allowed_avg_3",
    "receiving_yards_allowed_avg_3",
    "rushing_yards_allowed_avg_3",
    "receiving_tds_allowed_avg_3",
    "rushing_tds_allowed_avg_3",
    "opportunities_allowed_avg_3",
]

MATCHUP_FEATURES = (
    PLAYER_FEATURES
    + TEAM_FEATURES
    + DVP_FEATURES
)

print("=== MATCHUP INTELLIGENCE STEP 6H — FORWARD BACKTEST ===")
print("MODE=ANALYSIS_ONLY")

df = pd.read_parquet(SOURCE)

# Only rows with actual prior player/team/DvP evidence.
df = df[
    df["core_matchup_history_available"].eq(1)
].copy()

# Defensive integrity.
if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

for c in MATCHUP_FEATURES + ["fanduel_points"]:
    s = pd.to_numeric(df[c], errors="coerce")

    if s.isna().any():
        raise RuntimeError(f"NULL_NUMERIC:{c}")

    if not np.isfinite(s).all():
        raise RuntimeError(f"NONFINITE_NUMERIC:{c}")

def model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

def score(y, p):
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "corr": (
            np.corrcoef(y, p)[0, 1]
            if len(y) > 1
            else np.nan
        ),
    }

results = []

# Strict season-forward folds.
# No future season can train an earlier season.
for test_season in [2024, 2025, 2026]:

    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()

    if train.empty or test.empty:
        continue

    for position in ["QB", "RB", "WR", "TE"]:

        tr = train[train["position"].eq(position)]
        te = test[test["position"].eq(position)]

        if tr.empty or te.empty:
            continue

        y_train = tr["fanduel_points"].astype(float)
        y_test = te["fanduel_points"].astype(float)

        baseline = model()
        baseline.fit(
            tr[PLAYER_FEATURES],
            y_train,
        )

        bp = baseline.predict(
            te[PLAYER_FEATURES]
        )

        matchup = model()
        matchup.fit(
            tr[MATCHUP_FEATURES],
            y_train,
        )

        mp = matchup.predict(
            te[MATCHUP_FEATURES]
        )

        bs = score(y_test, bp)
        ms = score(y_test, mp)

        results.append(
            {
                "test_season": test_season,
                "position": position,
                "train_rows": len(tr),
                "test_rows": len(te),
                "baseline_mae": bs["mae"],
                "matchup_mae": ms["mae"],
                "mae_delta": (
                    ms["mae"] - bs["mae"]
                ),
                "baseline_rmse": bs["rmse"],
                "matchup_rmse": ms["rmse"],
                "rmse_delta": (
                    ms["rmse"] - bs["rmse"]
                ),
                "baseline_corr": bs["corr"],
                "matchup_corr": ms["corr"],
                "corr_delta": (
                    ms["corr"] - bs["corr"]
                ),
            }
        )

out = pd.DataFrame(results)

if out.empty:
    raise RuntimeError("NO_BACKTEST_RESULTS")

print()
print(out.to_string(index=False))

print()
print("=== AGGREGATE DIRECTION ===")

print(
    "MAE_IMPROVED_FOLDS="
    f"{int((out['mae_delta'] < 0).sum())}/"
    f"{len(out)}"
)

print(
    "RMSE_IMPROVED_FOLDS="
    f"{int((out['rmse_delta'] < 0).sum())}/"
    f"{len(out)}"
)

print(
    "CORR_IMPROVED_FOLDS="
    f"{int((out['corr_delta'] > 0).sum())}/"
    f"{len(out)}"
)

print(
    "MEAN_MAE_DELTA="
    f"{out['mae_delta'].mean():.6f}"
)

print(
    "MEAN_RMSE_DELTA="
    f"{out['rmse_delta'].mean():.6f}"
)

print(
    "MEAN_CORR_DELTA="
    f"{out['corr_delta'].mean():.6f}"
)

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
