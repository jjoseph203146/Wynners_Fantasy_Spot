from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")
BASE_PATH = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
SIT_PATH = ROOT / "data/research/situational_role_history_v1.parquet"

PLAYER = [
    "usage_3g", "snap_pct_3g", "target_3g",
    "carry_3g", "fanduel_3g", "prior_player_games",
]

TEAM = [
    "team_history_games",
    "points_for_avg_3", "points_for_avg_5",
    "offensive_plays_avg_3", "offensive_plays_avg_5",
    "pass_attempts_avg_3", "pass_attempts_avg_5",
    "rush_attempts_avg_3", "rush_attempts_avg_5",
    "pass_rate_avg_3", "pass_rate_avg_5",
    "rush_rate_avg_3", "rush_rate_avg_5",
    "passing_yards_avg_3", "rushing_yards_avg_3",
    "passing_tds_avg_3", "rushing_tds_avg_3",
    "opponent_points_allowed_avg_3",
    "opponent_points_allowed_avg_5",
    "opponent_pass_yards_allowed_avg_3",
    "opponent_rush_yards_allowed_avg_3",
    "opponent_pass_tds_allowed_avg_3",
    "opponent_rush_tds_allowed_avg_3",
]

DVP = [
    "dvp_history_games",
    "fd_allowed_avg_3", "fd_allowed_avg_5",
    "targets_allowed_avg_3", "carries_allowed_avg_3",
    "receptions_allowed_avg_3",
    "receiving_yards_allowed_avg_3",
    "rushing_yards_allowed_avg_3",
    "receiving_tds_allowed_avg_3",
    "rushing_tds_allowed_avg_3",
    "opportunities_allowed_avg_3",
]

ROLE = {
    "RB": [
        "carry_share_proxy", "usage_share_proxy",
        "carry_rank_team_pos", "usage_rank_team_pos",
    ],
    "TE": [
        "target_share_proxy", "usage_share_proxy",
        "target_rank_team_pos", "usage_rank_team_pos",
    ],
}

GROUPS = {
    "RB": {
        "I10": ["i10_carries_3g"],
        "G2G": ["g2g_carries_3g"],
        "I10_G2G": [
            "i10_carries_3g",
            "g2g_carries_3g",
        ],
    },
    "TE": {
        "I10": ["i10_targets_3g"],
        "G2G": ["g2g_targets_3g"],
        "I10_G2G": [
            "i10_targets_3g",
            "g2g_targets_3g",
        ],
    },
}

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
    return (
        mean_absolute_error(y, p),
        mean_squared_error(y, p) ** 0.5,
        np.corrcoef(y, p)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6V — I10/G2G REDUNDANCY ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

base = pd.read_parquet(BASE_PATH)
sit = pd.read_parquet(SIT_PATH)

sit_features = [
    "rz_carries_3g", "i10_carries_3g", "g2g_carries_3g",
    "rz_targets_3g", "i10_targets_3g", "g2g_targets_3g",
]

df = base.merge(
    sit[["game_id", "player_id"] + sit_features],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

if df[sit_features].isna().any().any():
    raise RuntimeError("SITUATIONAL_JOIN_MISSING")

df["team_target3_total"] = (
    df.groupby(["game_id", "team"])["target_3g"].transform("sum")
)
df["team_carry3_total"] = (
    df.groupby(["game_id", "team"])["carry_3g"].transform("sum")
)
df["team_usage3_total"] = (
    df.groupby(["game_id", "team"])["usage_3g"].transform("sum")
)

df["target_share_proxy"] = np.where(
    df["team_target3_total"] > 0,
    df["target_3g"] / df["team_target3_total"], 0.0,
)
df["carry_share_proxy"] = np.where(
    df["team_carry3_total"] > 0,
    df["carry_3g"] / df["team_carry3_total"], 0.0,
)
df["usage_share_proxy"] = np.where(
    df["team_usage3_total"] > 0,
    df["usage_3g"] / df["team_usage3_total"], 0.0,
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

for season in [2024, 2025, 2026]:
    train = df[df["season"] < season]
    test = df[df["season"] == season]

    for pos in ["RB", "TE"]:
        tr = train[train["position"].eq(pos)]
        te = test[test["position"].eq(pos)]

        core = list(dict.fromkeys(PLAYER + MATCHUP + ROLE[pos]))

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        m = model()
        m.fit(tr[core], ytr)
        pred = m.predict(te[core])
        bmae, brmse, bcorr = score(yte, pred)

        for group, extra in GROUPS[pos].items():
            features = core + extra

            m = model()
            m.fit(tr[features], ytr)
            pred = m.predict(te[features])
            mae, rmse, corr = score(yte, pred)

            rows.append({
                "season": season,
                "position": pos,
                "group": group,
                "test_rows": len(te),
                "mae_delta": mae - bmae,
                "rmse_delta": rmse - brmse,
                "corr_delta": corr - bcorr,
            })

out = pd.DataFrame(rows)

print()
print("=== SEASON × POSITION × GROUP ===")
print(out.to_string(index=False))

print()
print("=== GROUP SUMMARY ===")

summary = (
    out.groupby(["position", "group"])
    .agg(
        folds=("mae_delta", "size"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_rmse_delta=("rmse_delta", "mean"),
        mean_corr_delta=("corr_delta", "mean"),
        mae_wins=("mae_delta", lambda x: int((x < 0).sum())),
        rmse_wins=("rmse_delta", lambda x: int((x < 0).sum())),
        corr_wins=("corr_delta", lambda x: int((x > 0).sum())),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
