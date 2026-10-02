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

MATCHUP = list(dict.fromkeys(TEAM + DVP))

ROLE_BY_POSITION = {
    "RB": [
        "carry_share_proxy",
        "usage_share_proxy",
        "carry_rank_team_pos",
        "usage_rank_team_pos",
    ],
    "WR": [
        "target_share_proxy",
        "usage_share_proxy",
        "target_rank_team_pos",
        "usage_rank_team_pos",
    ],
    "TE": [
        "target_share_proxy",
        "usage_share_proxy",
        "target_rank_team_pos",
        "usage_rank_team_pos",
    ],
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

def score(y, pred):
    return (
        mean_absolute_error(y, pred),
        mean_squared_error(y, pred) ** 0.5,
        np.corrcoef(y, pred)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6P — MATCHUP + ROLE ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

df = pd.read_parquet(SOURCE)

df = df[
    df["position"].isin(["RB", "WR", "TE"])
].copy()

df = df[
    (df["prior_player_games"] >= 3)
    & (df["team_history_games"] >= 3)
    & (df["dvp_history_games"] >= 3)
].copy()

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

# ---------------------------------------------------------
# PIT-safe relative role proxies.
# ---------------------------------------------------------

df["team_target3_total"] = (
    df.groupby(["game_id", "team"])["target_3g"]
    .transform("sum")
)

df["team_carry3_total"] = (
    df.groupby(["game_id", "team"])["carry_3g"]
    .transform("sum")
)

df["team_usage3_total"] = (
    df.groupby(["game_id", "team"])["usage_3g"]
    .transform("sum")
)

df["target_share_proxy"] = np.where(
    df["team_target3_total"] > 0,
    df["target_3g"] / df["team_target3_total"],
    0.0,
)

df["carry_share_proxy"] = np.where(
    df["team_carry3_total"] > 0,
    df["carry_3g"] / df["team_carry3_total"],
    0.0,
)

df["usage_share_proxy"] = np.where(
    df["team_usage3_total"] > 0,
    df["usage_3g"] / df["team_usage3_total"],
    0.0,
)

df["target_rank_team_pos"] = (
    df.groupby(["game_id", "team", "position"])["target_3g"]
    .rank(method="min", ascending=False)
)

df["carry_rank_team_pos"] = (
    df.groupby(["game_id", "team", "position"])["carry_3g"]
    .rank(method="min", ascending=False)
)

df["usage_rank_team_pos"] = (
    df.groupby(["game_id", "team", "position"])["usage_3g"]
    .rank(method="min", ascending=False)
)

rows = []

for test_season in [2024, 2025, 2026]:

    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]

    for pos in ["RB", "WR", "TE"]:

        tr = train[train["position"].eq(pos)]
        te = test[test["position"].eq(pos)]

        if tr.empty or te.empty:
            continue

        role = ROLE_BY_POSITION[pos]

        feature_sets = {
            "PLAYER": PLAYER,
            "MATCHUP": list(dict.fromkeys(PLAYER + MATCHUP)),
            "ROLE": list(dict.fromkeys(PLAYER + role)),
            "MATCHUP_ROLE": list(
                dict.fromkeys(PLAYER + MATCHUP + role)
            ),
        }

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        result = {
            "test_season": test_season,
            "position": pos,
            "train_rows": len(tr),
            "test_rows": len(te),
        }

        for name, features in feature_sets.items():

            for c in features:
                if tr[c].isna().any() or te[c].isna().any():
                    raise RuntimeError(
                        f"NULL_FEATURE:{pos}:{name}:{c}"
                    )

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

for variant in ["MATCHUP", "ROLE", "MATCHUP_ROLE"]:

    out[f"{variant}_vs_player_mae"] = (
        out[f"{variant}_mae"] - out["PLAYER_mae"]
    )

    out[f"{variant}_vs_player_rmse"] = (
        out[f"{variant}_rmse"] - out["PLAYER_rmse"]
    )

    out[f"{variant}_vs_player_corr"] = (
        out[f"{variant}_corr"] - out["PLAYER_corr"]
    )

# Critical incremental test:
# Does role still add signal AFTER matchup is already present?
out["ROLE_AFTER_MATCHUP_mae"] = (
    out["MATCHUP_ROLE_mae"] - out["MATCHUP_mae"]
)

out["ROLE_AFTER_MATCHUP_rmse"] = (
    out["MATCHUP_ROLE_rmse"] - out["MATCHUP_rmse"]
)

out["ROLE_AFTER_MATCHUP_corr"] = (
    out["MATCHUP_ROLE_corr"] - out["MATCHUP_corr"]
)

print()
print("=== SEASON × POSITION ===")

cols = [
    "test_season",
    "position",
    "test_rows",
    "MATCHUP_vs_player_mae",
    "ROLE_vs_player_mae",
    "MATCHUP_ROLE_vs_player_mae",
    "ROLE_AFTER_MATCHUP_mae",
    "ROLE_AFTER_MATCHUP_rmse",
    "ROLE_AFTER_MATCHUP_corr",
]

print(out[cols].to_string(index=False))

print()
print("=== POSITION SUMMARY ===")

for pos in ["RB", "WR", "TE"]:

    z = out[out["position"].eq(pos)]

    print()
    print(f"POSITION={pos}")

    for variant in [
        "MATCHUP",
        "ROLE",
        "MATCHUP_ROLE",
    ]:
        print(
            f"{variant}_VS_PLAYER "
            f"MAE={z[f'{variant}_vs_player_mae'].mean():.6f} "
            f"RMSE={z[f'{variant}_vs_player_rmse'].mean():.6f} "
            f"CORR={z[f'{variant}_vs_player_corr'].mean():.6f}"
        )

    print(
        "ROLE_AFTER_MATCHUP "
        f"MAE={z['ROLE_AFTER_MATCHUP_mae'].mean():.6f} "
        f"RMSE={z['ROLE_AFTER_MATCHUP_rmse'].mean():.6f} "
        f"CORR={z['ROLE_AFTER_MATCHUP_corr'].mean():.6f} "
        f"MAE_WINS={(z['ROLE_AFTER_MATCHUP_mae'] < 0).sum()}/{len(z)} "
        f"RMSE_WINS={(z['ROLE_AFTER_MATCHUP_rmse'] < 0).sum()}/{len(z)} "
        f"CORR_WINS={(z['ROLE_AFTER_MATCHUP_corr'] > 0).sum()}/{len(z)}"
    )

print()
print("=== OVERALL ROLE AFTER MATCHUP ===")

print(
    "MAE_IMPROVED="
    f"{(out['ROLE_AFTER_MATCHUP_mae'] < 0).sum()}/{len(out)}"
)

print(
    "RMSE_IMPROVED="
    f"{(out['ROLE_AFTER_MATCHUP_rmse'] < 0).sum()}/{len(out)}"
)

print(
    "CORR_IMPROVED="
    f"{(out['ROLE_AFTER_MATCHUP_corr'] > 0).sum()}/{len(out)}"
)

print(
    "MEAN_MAE_DELTA="
    f"{out['ROLE_AFTER_MATCHUP_mae'].mean():.6f}"
)

print(
    "MEAN_RMSE_DELTA="
    f"{out['ROLE_AFTER_MATCHUP_rmse'].mean():.6f}"
)

print(
    "MEAN_CORR_DELTA="
    f"{out['ROLE_AFTER_MATCHUP_corr'].mean():.6f}"
)

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
