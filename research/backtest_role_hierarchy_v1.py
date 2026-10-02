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

print("=== MATCHUP INTELLIGENCE STEP 6O — ROLE HIERARCHY SIGNAL ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

df = pd.read_parquet(SOURCE)

df = df[
    df["position"].isin(["RB", "WR", "TE"])
].copy()

df = df[
    df["prior_player_games"] >= 3
].copy()

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

# ---------------------------------------------------------
# PIT-safe team-relative workload context.
# These use only the already validated lagged 3-game fields.
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

# Ranks preserve ties.
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
        enhanced = PLAYER + role

        for c in enhanced + ["fanduel_points"]:
            if tr[c].isna().any() or te[c].isna().any():
                raise RuntimeError(
                    f"NULL_FEATURE:{pos}:{c}"
                )

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        base = make_model()
        base.fit(tr[PLAYER], ytr)
        bp = base.predict(te[PLAYER])

        role_model = make_model()
        role_model.fit(tr[enhanced], ytr)
        rp = role_model.predict(te[enhanced])

        b_mae, b_rmse, b_corr = score(yte, bp)
        r_mae, r_rmse, r_corr = score(yte, rp)

        rows.append({
            "test_season": test_season,
            "position": pos,
            "train_rows": len(tr),
            "test_rows": len(te),
            "mae_delta": r_mae - b_mae,
            "rmse_delta": r_rmse - b_rmse,
            "corr_delta": r_corr - b_corr,
        })

out = pd.DataFrame(rows)

if out.empty:
    raise RuntimeError("NO_RESULTS")

print()
print("=== SEASON × POSITION ===")
print(out.to_string(index=False))

print()
print("=== POSITION SUMMARY ===")

summary = (
    out.groupby("position")
    .agg(
        folds=("mae_delta", "size"),
        test_rows=("test_rows", "sum"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_rmse_delta=("rmse_delta", "mean"),
        mean_corr_delta=("corr_delta", "mean"),
        mae_improved=(
            "mae_delta",
            lambda x: int((x < 0).sum()),
        ),
        rmse_improved=(
            "rmse_delta",
            lambda x: int((x < 0).sum()),
        ),
        corr_improved=(
            "corr_delta",
            lambda x: int((x > 0).sum()),
        ),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("=== OVERALL ===")

print(
    "MAE_IMPROVED_FOLDS="
    f"{(out['mae_delta'] < 0).sum()}/{len(out)}"
)

print(
    "RMSE_IMPROVED_FOLDS="
    f"{(out['rmse_delta'] < 0).sum()}/{len(out)}"
)

print(
    "CORR_IMPROVED_FOLDS="
    f"{(out['corr_delta'] > 0).sum()}/{len(out)}"
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
