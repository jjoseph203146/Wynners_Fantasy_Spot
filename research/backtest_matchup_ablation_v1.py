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

FEATURE_SETS = {
    "PLAYER": PLAYER,
    "PLAYER_TEAM": PLAYER + TEAM,
    "PLAYER_DVP": PLAYER + DVP,
    "PLAYER_TEAM_DVP": PLAYER + TEAM + DVP,
}

def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

def metrics(y, p):
    return (
        mean_absolute_error(y, p),
        mean_squared_error(y, p) ** 0.5,
        np.corrcoef(y, p)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6I — COMPONENT ABLATION ===")
print("MODE=ANALYSIS_ONLY")

df = pd.read_parquet(SOURCE)
df = df[df["core_matchup_history_available"].eq(1)].copy()

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

all_features = sorted(set(PLAYER + TEAM + DVP))

for c in all_features + ["fanduel_points"]:
    s = pd.to_numeric(df[c], errors="coerce")
    if s.isna().any():
        raise RuntimeError(f"NULL_NUMERIC:{c}")
    if not np.isfinite(s).all():
        raise RuntimeError(f"NONFINITE_NUMERIC:{c}")

rows = []

for test_season in [2024, 2025, 2026]:
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]

    for pos in ["QB", "RB", "WR", "TE"]:
        tr = train[train["position"].eq(pos)]
        te = test[test["position"].eq(pos)]

        if tr.empty or te.empty:
            continue

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        fold = {
            "test_season": test_season,
            "position": pos,
            "train_rows": len(tr),
            "test_rows": len(te),
        }

        for name, features in FEATURE_SETS.items():
            m = make_model()
            m.fit(tr[features], ytr)
            pred = m.predict(te[features])

            mae, rmse, corr = metrics(yte, pred)

            fold[f"{name}_mae"] = mae
            fold[f"{name}_rmse"] = rmse
            fold[f"{name}_corr"] = corr

        rows.append(fold)

out = pd.DataFrame(rows)

baseline = "PLAYER"

for variant in [
    "PLAYER_TEAM",
    "PLAYER_DVP",
    "PLAYER_TEAM_DVP",
]:
    out[f"{variant}_mae_delta"] = (
        out[f"{variant}_mae"] - out[f"{baseline}_mae"]
    )
    out[f"{variant}_rmse_delta"] = (
        out[f"{variant}_rmse"] - out[f"{baseline}_rmse"]
    )
    out[f"{variant}_corr_delta"] = (
        out[f"{variant}_corr"] - out[f"{baseline}_corr"]
    )

show = [
    "test_season",
    "position",
    "test_rows",
]

for variant in [
    "PLAYER_TEAM",
    "PLAYER_DVP",
    "PLAYER_TEAM_DVP",
]:
    show += [
        f"{variant}_mae_delta",
        f"{variant}_rmse_delta",
        f"{variant}_corr_delta",
    ]

print()
print(out[show].to_string(index=False))

print()
print("=== COMPONENT SUMMARY ===")

for variant in [
    "PLAYER_TEAM",
    "PLAYER_DVP",
    "PLAYER_TEAM_DVP",
]:
    print()
    print(f"COMPONENT={variant}")
    print(
        "MAE_IMPROVED="
        f"{(out[f'{variant}_mae_delta'] < 0).sum()}/{len(out)}"
    )
    print(
        "RMSE_IMPROVED="
        f"{(out[f'{variant}_rmse_delta'] < 0).sum()}/{len(out)}"
    )
    print(
        "CORR_IMPROVED="
        f"{(out[f'{variant}_corr_delta'] > 0).sum()}/{len(out)}"
    )
    print(
        "MEAN_MAE_DELTA="
        f"{out[f'{variant}_mae_delta'].mean():.6f}"
    )
    print(
        "MEAN_RMSE_DELTA="
        f"{out[f'{variant}_rmse_delta'].mean():.6f}"
    )
    print(
        "MEAN_CORR_DELTA="
        f"{out[f'{variant}_corr_delta'].mean():.6f}"
    )

print()
print("=== POSITION SUMMARY ===")

for pos in ["QB", "RB", "WR", "TE"]:
    z = out[out["position"].eq(pos)]

    print()
    print(f"POSITION={pos}")

    for variant in [
        "PLAYER_TEAM",
        "PLAYER_DVP",
        "PLAYER_TEAM_DVP",
    ]:
        print(
            f"{variant} "
            f"MAE={z[f'{variant}_mae_delta'].mean():.6f} "
            f"RMSE={z[f'{variant}_rmse_delta'].mean():.6f} "
            f"CORR={z[f'{variant}_corr_delta'].mean():.6f}"
        )

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
