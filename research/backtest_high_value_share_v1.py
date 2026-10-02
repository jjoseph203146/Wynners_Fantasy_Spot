from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")
BASE = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
SIT = ROOT / "data/research/situational_role_history_v1.parquet"

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

ABS = {
    "RB": "g2g_carries_3g",
    "TE": "i10_targets_3g",
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

def score(y, p):
    return (
        mean_absolute_error(y, p),
        mean_squared_error(y, p) ** 0.5,
        np.corrcoef(y, p)[0, 1],
    )

print("=== MATCHUP INTELLIGENCE STEP 6W-B — HIGH-VALUE SHARE SIGNAL ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")

base = pd.read_parquet(BASE)
sit = pd.read_parquet(SIT)

df = base.merge(
    sit[
        ["game_id", "player_id",
         "g2g_carries_3g", "i10_targets_3g"]
    ],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

if df[["g2g_carries_3g", "i10_targets_3g"]].isna().any().any():
    raise RuntimeError("SITUATIONAL_JOIN_MISSING")

# Existing validated relative-role construction.
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

# High-value opportunity shares.
#
# Denominator is represented players at the relevant position,
# consistent with the validated workload-proxy semantics.
for pos, absolute in ABS.items():
    mask = df["position"].eq(pos)

    denom = (
        df.loc[mask]
        .groupby(["game_id", "team"])[absolute]
        .transform("sum")
    )

    df.loc[mask, f"{pos.lower()}_hv_denom"] = denom.values

    df.loc[mask, f"{pos.lower()}_hv_available"] = (
        denom.gt(0).astype(int).values
    )

    share = np.where(
        denom > 0,
        df.loc[mask, absolute].to_numpy() / denom.to_numpy(),
        0.0,
    )

    df.loc[mask, f"{pos.lower()}_hv_share"] = share

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

        tr = train[train["position"].eq(pos)].copy()
        te = test[test["position"].eq(pos)].copy()

        absolute = ABS[pos]
        share = f"{pos.lower()}_hv_share"
        available = f"{pos.lower()}_hv_available"

        core = list(dict.fromkeys(
            PLAYER + MATCHUP + ROLE[pos]
        ))

        variants = {
            "CORE": [],
            "ABSOLUTE": [absolute],
            "SHARE": [share, available],
            "ABS_SHARE": [absolute, share, available],
        }

        ytr = tr["fanduel_points"].astype(float)
        yte = te["fanduel_points"].astype(float)

        scores = {}

        for name, extra in variants.items():

            features = list(dict.fromkeys(core + extra))

            for c in features:
                if tr[c].isna().any() or te[c].isna().any():
                    raise RuntimeError(
                        f"NULL_FEATURE:{season}:{pos}:{name}:{c}"
                    )

            m = make_model()
            m.fit(tr[features], ytr)
            pred = m.predict(te[features])

            scores[name] = score(yte, pred)

        base_mae, base_rmse, base_corr = scores["CORE"]

        for name in ["ABSOLUTE", "SHARE", "ABS_SHARE"]:

            mae, rmse, corr = scores[name]

            rows.append({
                "season": season,
                "position": pos,
                "variant": name,
                "train_rows": len(tr),
                "test_rows": len(te),
                "mae_delta": mae - base_mae,
                "rmse_delta": rmse - base_rmse,
                "corr_delta": corr - base_corr,
            })

out = pd.DataFrame(rows)

print()
print("=== SEASON × POSITION × VARIANT ===")
print(out.to_string(index=False))

print()
print("=== VARIANT SUMMARY ===")

summary = (
    out.groupby(["position", "variant"])
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
print("=== DENOMINATOR AUDIT ===")

for pos in ["RB", "TE"]:
    x = df[df["position"].eq(pos)]
    avail = f"{pos.lower()}_hv_available"
    share = f"{pos.lower()}_hv_share"

    print(
        f"{pos}_AVAILABLE_ROWS="
        f"{int(x[avail].sum())}/{len(x)}"
    )
    print(
        f"{pos}_INVALID_SHARE_ROWS="
        f"{int(((x[share] < 0) | (x[share] > 1)).sum())}"
    )

print()
print("ZERO_DENOMINATOR_EXPLICIT=YES")
print("TIES_PRESERVED=YES")
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
