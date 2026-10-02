from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")
BASE = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
SIT = ROOT / "data/research/situational_role_history_v1.parquet"

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

ROLE = {
    "RB": [
        "carry_share_proxy",
        "usage_share_proxy",
        "carry_rank_team_pos",
        "usage_rank_team_pos",
    ],
    "TE": [
        "target_share_proxy",
        "usage_share_proxy",
        "target_rank_team_pos",
        "usage_rank_team_pos",
    ],
}

SITUATIONAL = {
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


def metrics(y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5

    if len(y) >= 3 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    else:
        corr = np.nan

    return mae, rmse, corr


print("=== MATCHUP INTELLIGENCE STEP 6X-B — WEEKLY STABILITY ===")
print("MODE=ANALYSIS_ONLY")
print("MIN_HISTORY=3")
print("WEEKLY_MODEL_REFIT=NO")
print("SEASON_FORWARD_PREDICTIONS=YES")

base = pd.read_parquet(BASE)
sit = pd.read_parquet(SIT)

required_sit = [
    "game_id",
    "player_id",
    "g2g_carries_3g",
    "i10_targets_3g",
]

df = base.merge(
    sit[required_sit],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

if df[["g2g_carries_3g", "i10_targets_3g"]].isna().any().any():
    raise RuntimeError("SITUATIONAL_JOIN_MISSING")

# ------------------------------------------------------------
# Validated 6P relative workload-role representation
# ------------------------------------------------------------

df["team_target3_total"] = (
    df.groupby(["game_id", "team"])["target_3g"].transform("sum")
)

df["team_carry3_total"] = (
    df.groupby(["game_id", "team"])["carry_3g"].transform("sum")
)

df["team_usage3_total"] = (
    df.groupby(["game_id", "team"])["usage_3g"].transform("sum")
)

df["target_share_proxy"] = np.divide(
    df["target_3g"].to_numpy(dtype=float),
    df["team_target3_total"].to_numpy(dtype=float),
    out=np.zeros(len(df), dtype=float),
    where=df["team_target3_total"].to_numpy(dtype=float) > 0,
)

df["carry_share_proxy"] = np.divide(
    df["carry_3g"].to_numpy(dtype=float),
    df["team_carry3_total"].to_numpy(dtype=float),
    out=np.zeros(len(df), dtype=float),
    where=df["team_carry3_total"].to_numpy(dtype=float) > 0,
)

df["usage_share_proxy"] = np.divide(
    df["usage_3g"].to_numpy(dtype=float),
    df["team_usage3_total"].to_numpy(dtype=float),
    out=np.zeros(len(df), dtype=float),
    where=df["team_usage3_total"].to_numpy(dtype=float) > 0,
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

# ------------------------------------------------------------
# High-value opportunity share
#
# Preserve zero-denominator semantics with explicit availability.
# np.divide avoids the harmless np.where divide warning seen in 6W-B.
# ------------------------------------------------------------

for pos, absolute in SITUATIONAL.items():

    mask = df["position"].eq(pos)

    denom = (
        df.loc[mask]
        .groupby(["game_id", "team"])[absolute]
        .transform("sum")
        .astype(float)
    )

    numerator = df.loc[mask, absolute].to_numpy(dtype=float)
    denominator = denom.to_numpy(dtype=float)

    share = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(numerator), dtype=float),
        where=denominator > 0,
    )

    df.loc[mask, f"{pos.lower()}_hv_share"] = share
    df.loc[mask, f"{pos.lower()}_hv_available"] = (
        denominator > 0
    ).astype(int)

# ------------------------------------------------------------
# Same minimum-history contract as 6W-B
# ------------------------------------------------------------

df = df[
    (df["prior_player_games"] >= 3)
    & (df["team_history_games"] >= 3)
    & (df["dvp_history_games"] >= 3)
].copy()

MATCHUP = list(dict.fromkeys(TEAM + DVP))

season_rows = []
weekly_rows = []

for season in [2024, 2025, 2026]:

    train = df[df["season"] < season].copy()
    test = df[df["season"] == season].copy()

    for pos in ["RB", "TE"]:

        tr = train[train["position"].eq(pos)].copy()
        te = test[test["position"].eq(pos)].copy()

        share = f"{pos.lower()}_hv_share"
        available = f"{pos.lower()}_hv_available"

        core = list(
            dict.fromkeys(
                PLAYER + MATCHUP + ROLE[pos]
            )
        )

        enhanced = list(
            dict.fromkeys(
                core + [share, available]
            )
        )

        required = list(dict.fromkeys(core + enhanced))

        for col in required:
            if tr[col].isna().any():
                raise RuntimeError(
                    f"TRAIN_NULL:{season}:{pos}:{col}"
                )
            if te[col].isna().any():
                raise RuntimeError(
                    f"TEST_NULL:{season}:{pos}:{col}"
                )

        y_train = tr["fanduel_points"].astype(float)
        y_test = te["fanduel_points"].astype(float)

        core_model = make_model()
        core_model.fit(tr[core], y_train)

        share_model = make_model()
        share_model.fit(tr[enhanced], y_train)

        te = te.copy()

        te["core_pred"] = core_model.predict(te[core])
        te["share_pred"] = share_model.predict(te[enhanced])

        c_mae, c_rmse, c_corr = metrics(
            y_test,
            te["core_pred"],
        )

        s_mae, s_rmse, s_corr = metrics(
            y_test,
            te["share_pred"],
        )

        season_rows.append(
            {
                "season": season,
                "position": pos,
                "test_rows": len(te),
                "mae_delta": s_mae - c_mae,
                "rmse_delta": s_rmse - c_rmse,
                "corr_delta": s_corr - c_corr,
            }
        )

        for week, w in te.groupby("week", sort=True):

            y = w["fanduel_points"].astype(float)

            c_mae, c_rmse, c_corr = metrics(
                y,
                w["core_pred"],
            )

            s_mae, s_rmse, s_corr = metrics(
                y,
                w["share_pred"],
            )

            weekly_rows.append(
                {
                    "season": season,
                    "week": week,
                    "position": pos,
                    "rows": len(w),
                    "mae_delta": s_mae - c_mae,
                    "rmse_delta": s_rmse - c_rmse,
                    "corr_delta": (
                        s_corr - c_corr
                        if np.isfinite(s_corr)
                        and np.isfinite(c_corr)
                        else np.nan
                    ),
                }
            )

season_out = pd.DataFrame(season_rows)
weekly = pd.DataFrame(weekly_rows)

print()
print("=== SEASON RECONCILIATION ===")
print(season_out.to_string(index=False))

print()
print("=== WEEKLY STABILITY SUMMARY ===")

summary = (
    weekly.groupby("position")
    .agg(
        weeks=("week", "size"),
        rows=("rows", "sum"),
        mean_mae_delta=("mae_delta", "mean"),
        median_mae_delta=("mae_delta", "median"),
        mean_rmse_delta=("rmse_delta", "mean"),
        median_rmse_delta=("rmse_delta", "median"),
        mean_corr_delta=("corr_delta", "mean"),
        median_corr_delta=("corr_delta", "median"),
        mae_wins=(
            "mae_delta",
            lambda x: int((x < 0).sum()),
        ),
        mae_losses=(
            "mae_delta",
            lambda x: int((x > 0).sum()),
        ),
        rmse_wins=(
            "rmse_delta",
            lambda x: int((x < 0).sum()),
        ),
        rmse_losses=(
            "rmse_delta",
            lambda x: int((x > 0).sum()),
        ),
        corr_wins=(
            "corr_delta",
            lambda x: int((x > 0).sum()),
        ),
        corr_losses=(
            "corr_delta",
            lambda x: int((x < 0).sum()),
        ),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("=== SEASON × POSITION WEEKLY SUMMARY ===")

by_season = (
    weekly.groupby(["season", "position"])
    .agg(
        weeks=("week", "size"),
        rows=("rows", "sum"),
        mean_mae_delta=("mae_delta", "mean"),
        median_mae_delta=("mae_delta", "median"),
        mae_wins=(
            "mae_delta",
            lambda x: int((x < 0).sum()),
        ),
        mae_losses=(
            "mae_delta",
            lambda x: int((x > 0).sum()),
        ),
        mean_rmse_delta=("rmse_delta", "mean"),
        median_rmse_delta=("rmse_delta", "median"),
        rmse_wins=(
            "rmse_delta",
            lambda x: int((x < 0).sum()),
        ),
        rmse_losses=(
            "rmse_delta",
            lambda x: int((x > 0).sum()),
        ),
        mean_corr_delta=("corr_delta", "mean"),
        median_corr_delta=("corr_delta", "median"),
        corr_wins=(
            "corr_delta",
            lambda x: int((x > 0).sum()),
        ),
        corr_losses=(
            "corr_delta",
            lambda x: int((x < 0).sum()),
        ),
    )
    .reset_index()
)

print(by_season.to_string(index=False))

print()
print("=== VALIDATION ===")
print(f"WEEKLY_ROWS={len(weekly)}")
print(
    "NONFINITE_MAE_DELTA="
    f"{int((~np.isfinite(weekly['mae_delta'])).sum())}"
)
print(
    "NONFINITE_RMSE_DELTA="
    f"{int((~np.isfinite(weekly['rmse_delta'])).sum())}"
)
print(
    "CORR_UNAVAILABLE="
    f"{int(weekly['corr_delta'].isna().sum())}"
)

print()
print("WEEKLY_MODEL_REFIT=NO")
print("SEASON_FORWARD_PREDICTIONS=YES")
print("ZERO_DENOMINATOR_EXPLICIT=YES")
print("TIES_PRESERVED=YES")
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
