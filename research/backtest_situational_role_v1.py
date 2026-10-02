from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")
BASE_PATH = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
SIT_PATH = ROOT / "data/research/situational_role_history_v1.parquet"

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

ROLE_BY_POSITION = {
    "QB": [],
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

SITUATIONAL_BY_POSITION = {
    "QB": [
        "rz_pass_attempts_3g",
        "i10_pass_attempts_3g",
        "g2g_pass_attempts_3g",
        "qb_scrambles_3g",
    ],
    "RB": [
        "rz_carries_3g",
        "i10_carries_3g",
        "g2g_carries_3g",
    ],
    "WR": [
        "rz_targets_3g",
        "i10_targets_3g",
        "g2g_targets_3g",
    ],
    "TE": [
        "rz_targets_3g",
        "i10_targets_3g",
        "g2g_targets_3g",
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

print("=== MATCHUP INTELLIGENCE STEP 6T — SITUATIONAL ROLE SIGNAL ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

base = pd.read_parquet(BASE_PATH).copy()
sit = pd.read_parquet(SIT_PATH).copy()

KEY = ["game_id", "player_id"]

sit_cols = sorted({
    c
    for cols in SITUATIONAL_BY_POSITION.values()
    for c in cols
})

df = base.merge(
    sit[KEY + sit_cols],
    on=KEY,
    how="left",
    validate="one_to_one",
)

if df[sit_cols].isna().any().any():
    raise RuntimeError("SITUATIONAL_JOIN_MISSING")

# ---------------------------------------------------------
# Reconstruct PIT-safe relative role exactly as 6P.
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

df = df[
    (df["prior_player_games"] >= 3)
    & (df["team_history_games"] >= 3)
    & (df["dvp_history_games"] >= 3)
].copy()

MATCHUP = list(dict.fromkeys(TEAM + DVP))

rows = []

for test_season in [2024, 2025, 2026]:

    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]

    for pos in ["QB", "RB", "WR", "TE"]:

        tr = train[train["position"].eq(pos)]
        te = test[test["position"].eq(pos)]

        if tr.empty or te.empty:
            continue

        role = ROLE_BY_POSITION[pos]
        situational = SITUATIONAL_BY_POSITION[pos]

        core = list(
            dict.fromkeys(
                PLAYER + MATCHUP + role
            )
        )

        enhanced = list(
            dict.fromkeys(
                core + situational
            )
        )

        for c in enhanced + ["fanduel_points"]:
            if tr[c].isna().any() or te[c].isna().any():
                raise RuntimeError(
                    f"NULL_FEATURE:{pos}:{c}"
                )

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        core_model = make_model()
        core_model.fit(tr[core], ytr)
        core_pred = core_model.predict(te[core])

        sit_model = make_model()
        sit_model.fit(tr[enhanced], ytr)
        sit_pred = sit_model.predict(te[enhanced])

        c_mae, c_rmse, c_corr = score(yte, core_pred)
        s_mae, s_rmse, s_corr = score(yte, sit_pred)

        rows.append({
            "test_season": test_season,
            "position": pos,
            "train_rows": len(tr),
            "test_rows": len(te),
            "mae_delta": s_mae - c_mae,
            "rmse_delta": s_rmse - c_rmse,
            "corr_delta": s_corr - c_corr,
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
    "MAE_IMPROVED="
    f"{(out['mae_delta'] < 0).sum()}/{len(out)}"
)

print(
    "RMSE_IMPROVED="
    f"{(out['rmse_delta'] < 0).sum()}/{len(out)}"
)

print(
    "CORR_IMPROVED="
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
