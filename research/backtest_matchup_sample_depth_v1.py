from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")
SOURCE = ROOT / "data/research/matchup_intelligence_history_v1.parquet"

PLAYER = [
    "usage_3g",
    "snap_pct_3g",
    "target_3g",
    "carry_3g",
    "fanduel_3g",
    "prior_player_games",
]

TEAM = [
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

DVP = [
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

MATCHUP = PLAYER + TEAM + DVP

def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

def metrics(y, pred):
    return (
        mean_absolute_error(y, pred),
        mean_squared_error(y, pred) ** 0.5,
        np.corrcoef(y, pred)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6J — SAMPLE DEPTH STABILITY ===")
print("MODE=ANALYSIS_ONLY")

df = pd.read_parquet(SOURCE)

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

rows = []

for minimum in [1, 2, 3, 5]:

    eligible = df[
        (df["prior_player_games"] >= minimum)
        & (df["team_history_games"] >= minimum)
        & (df["dvp_history_games"] >= minimum)
    ].copy()

    print()
    print(
        f"MIN_HISTORY={minimum} "
        f"ROWS={len(eligible)} "
        f"GAMES={eligible['game_id'].nunique()} "
        f"PLAYERS={eligible['player_id'].nunique()}"
    )

    for test_season in [2024, 2025, 2026]:

        train = eligible[
            eligible["season"] < test_season
        ]

        test = eligible[
            eligible["season"] == test_season
        ]

        for pos in ["QB", "RB", "WR", "TE"]:

            tr = train[train["position"].eq(pos)]
            te = test[test["position"].eq(pos)]

            if tr.empty or te.empty:
                continue

            ytr = tr["fanduel_points"].astype(float)
            yte = te["fanduel_points"].astype(float)

            base = make_model()
            base.fit(tr[PLAYER], ytr)
            bp = base.predict(te[PLAYER])

            match = make_model()
            match.fit(tr[MATCHUP], ytr)
            mp = match.predict(te[MATCHUP])

            b_mae, b_rmse, b_corr = metrics(yte, bp)
            m_mae, m_rmse, m_corr = metrics(yte, mp)

            rows.append(
                {
                    "min_history": minimum,
                    "test_season": test_season,
                    "position": pos,
                    "train_rows": len(tr),
                    "test_rows": len(te),
                    "mae_delta": m_mae - b_mae,
                    "rmse_delta": m_rmse - b_rmse,
                    "corr_delta": m_corr - b_corr,
                }
            )

out = pd.DataFrame(rows)

if out.empty:
    raise RuntimeError("NO_RESULTS")

print()
print("=== POSITION × SAMPLE DEPTH SUMMARY ===")

summary = (
    out.groupby(["min_history", "position"])
    .agg(
        folds=("mae_delta", "size"),
        test_rows=("test_rows", "sum"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_rmse_delta=("rmse_delta", "mean"),
        mean_corr_delta=("corr_delta", "mean"),
        mae_improved=("mae_delta", lambda x: int((x < 0).sum())),
        rmse_improved=("rmse_delta", lambda x: int((x < 0).sum())),
        corr_improved=("corr_delta", lambda x: int((x > 0).sum())),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("=== OVERALL SAMPLE DEPTH SUMMARY ===")

overall = (
    out.groupby("min_history")
    .agg(
        folds=("mae_delta", "size"),
        test_rows=("test_rows", "sum"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_rmse_delta=("rmse_delta", "mean"),
        mean_corr_delta=("corr_delta", "mean"),
        mae_improved=("mae_delta", lambda x: int((x < 0).sum())),
        rmse_improved=("rmse_delta", lambda x: int((x < 0).sum())),
        corr_improved=("corr_delta", lambda x: int((x > 0).sum())),
    )
    .reset_index()
)

print(overall.to_string(index=False))

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
