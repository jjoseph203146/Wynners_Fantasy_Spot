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

TEAM_VOLUME = [
    "team_history_games",
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
]

TEAM_PRODUCTION = [
    "team_history_games",
    "points_for_avg_3",
    "points_for_avg_5",
    "passing_yards_avg_3",
    "rushing_yards_avg_3",
    "passing_tds_avg_3",
    "rushing_tds_avg_3",
]

DEFENSE_TEAM = [
    "team_history_games",
    "opponent_points_allowed_avg_3",
    "opponent_points_allowed_avg_5",
    "opponent_pass_yards_allowed_avg_3",
    "opponent_rush_yards_allowed_avg_3",
    "opponent_pass_tds_allowed_avg_3",
    "opponent_rush_tds_allowed_avg_3",
]

DVP_VOLUME = [
    "dvp_history_games",
    "targets_allowed_avg_3",
    "carries_allowed_avg_3",
    "receptions_allowed_avg_3",
    "opportunities_allowed_avg_3",
]

DVP_PRODUCTION = [
    "dvp_history_games",
    "fd_allowed_avg_3",
    "fd_allowed_avg_5",
    "receiving_yards_allowed_avg_3",
    "rushing_yards_allowed_avg_3",
    "receiving_tds_allowed_avg_3",
    "rushing_tds_allowed_avg_3",
]

GROUPS = {
    "PLAYER": PLAYER,
    "TEAM_VOLUME": PLAYER + TEAM_VOLUME,
    "TEAM_PRODUCTION": PLAYER + TEAM_PRODUCTION,
    "DEFENSE_TEAM": PLAYER + DEFENSE_TEAM,
    "DVP_VOLUME": PLAYER + DVP_VOLUME,
    "DVP_PRODUCTION": PLAYER + DVP_PRODUCTION,
    "ALL_GROUPS": (
        PLAYER
        + TEAM_VOLUME
        + TEAM_PRODUCTION
        + DEFENSE_TEAM
        + DVP_VOLUME
        + DVP_PRODUCTION
    ),
}

# Remove duplicate column names while preserving order.
for name, cols in GROUPS.items():
    GROUPS[name] = list(dict.fromkeys(cols))

def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

def score(y, pred):
    return (
        mean_absolute_error(y, pred),
        mean_squared_error(y, pred) ** 0.5,
        np.corrcoef(y, pred)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6M — FEATURE GROUP ABLATION ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

df = pd.read_parquet(SOURCE)

df = df[
    (df["prior_player_games"] >= 3)
    & (df["team_history_games"] >= 3)
    & (df["dvp_history_games"] >= 3)
].copy()

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

all_features = sorted({
    c
    for cols in GROUPS.values()
    for c in cols
})

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

        result = {
            "test_season": test_season,
            "position": pos,
            "test_rows": len(te),
        }

        for name, features in GROUPS.items():

            model = make_model()
            model.fit(tr[features], ytr)

            pred = model.predict(te[features])

            mae, rmse, corr = score(yte, pred)

            result[f"{name}_mae"] = mae
            result[f"{name}_rmse"] = rmse
            result[f"{name}_corr"] = corr

        rows.append(result)

out = pd.DataFrame(rows)

if out.empty:
    raise RuntimeError("NO_RESULTS")

variants = [
    "TEAM_VOLUME",
    "TEAM_PRODUCTION",
    "DEFENSE_TEAM",
    "DVP_VOLUME",
    "DVP_PRODUCTION",
    "ALL_GROUPS",
]

for name in variants:
    out[f"{name}_mae_delta"] = (
        out[f"{name}_mae"] - out["PLAYER_mae"]
    )
    out[f"{name}_rmse_delta"] = (
        out[f"{name}_rmse"] - out["PLAYER_rmse"]
    )
    out[f"{name}_corr_delta"] = (
        out[f"{name}_corr"] - out["PLAYER_corr"]
    )

print()
print("=== COMPONENT SUMMARY ===")

for name in variants:

    print()
    print(f"GROUP={name}")

    print(
        "MAE_IMPROVED="
        f"{(out[f'{name}_mae_delta'] < 0).sum()}/{len(out)}"
    )

    print(
        "RMSE_IMPROVED="
        f"{(out[f'{name}_rmse_delta'] < 0).sum()}/{len(out)}"
    )

    print(
        "CORR_IMPROVED="
        f"{(out[f'{name}_corr_delta'] > 0).sum()}/{len(out)}"
    )

    print(
        "MEAN_MAE_DELTA="
        f"{out[f'{name}_mae_delta'].mean():.6f}"
    )

    print(
        "MEAN_RMSE_DELTA="
        f"{out[f'{name}_rmse_delta'].mean():.6f}"
    )

    print(
        "MEAN_CORR_DELTA="
        f"{out[f'{name}_corr_delta'].mean():.6f}"
    )

print()
print("=== POSITION SUMMARY ===")

for pos in ["QB", "RB", "WR", "TE"]:

    z = out[out["position"].eq(pos)]

    print()
    print(f"POSITION={pos}")

    for name in variants:
        print(
            f"{name} "
            f"MAE={z[f'{name}_mae_delta'].mean():.6f} "
            f"RMSE={z[f'{name}_rmse_delta'].mean():.6f} "
            f"CORR={z[f'{name}_corr_delta'].mean():.6f}"
        )

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
