from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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

def ceiling_metrics(y, pred):
    z = pd.DataFrame({
        "actual": np.asarray(y, dtype=float),
        "pred": np.asarray(pred, dtype=float),
    })

    actual_cut = z["actual"].quantile(0.80)
    pred20_cut = z["pred"].quantile(0.80)
    pred10_cut = z["pred"].quantile(0.90)

    actual_top = z["actual"] >= actual_cut
    pred_top20 = z["pred"] >= pred20_cut
    pred_top10 = z["pred"] >= pred10_cut

    actual_top_count = int(actual_top.sum())

    recall20 = (
        (actual_top & pred_top20).sum() / actual_top_count
        if actual_top_count else np.nan
    )

    hit10 = (
        (actual_top & pred_top10).sum() / pred_top10.sum()
        if pred_top10.sum() else np.nan
    )

    captured_actual = z.loc[pred_top20, "actual"].mean()

    return recall20, hit10, captured_actual

print("=== MATCHUP INTELLIGENCE STEP 6K — CEILING SIGNAL ===")
print("MODE=ANALYSIS_ONLY")

df = pd.read_parquet(SOURCE)

# Use full 3-game evidence windows at all three levels.
df = df[
    (df["prior_player_games"] >= 3)
    & (df["team_history_games"] >= 3)
    & (df["dvp_history_games"] >= 3)
].copy()

if df.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_GAME_PLAYER")

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

        base = make_model()
        base.fit(tr[PLAYER], ytr)
        bp = base.predict(te[PLAYER])

        match = make_model()
        match.fit(tr[MATCHUP], ytr)
        mp = match.predict(te[MATCHUP])

        b_recall, b_hit10, b_capture = ceiling_metrics(yte, bp)
        m_recall, m_hit10, m_capture = ceiling_metrics(yte, mp)

        rows.append({
            "test_season": test_season,
            "position": pos,
            "test_rows": len(te),

            "baseline_top20_recall": b_recall,
            "matchup_top20_recall": m_recall,
            "top20_recall_delta": m_recall - b_recall,

            "baseline_top10_hit": b_hit10,
            "matchup_top10_hit": m_hit10,
            "top10_hit_delta": m_hit10 - b_hit10,

            "baseline_top20_actual_mean": b_capture,
            "matchup_top20_actual_mean": m_capture,
            "top20_actual_mean_delta": m_capture - b_capture,
        })

out = pd.DataFrame(rows)

if out.empty:
    raise RuntimeError("NO_RESULTS")

print()
print(out.to_string(index=False))

print()
print("=== POSITION SUMMARY ===")

summary = (
    out.groupby("position")
    .agg(
        folds=("top20_recall_delta", "size"),
        test_rows=("test_rows", "sum"),
        mean_recall_delta=("top20_recall_delta", "mean"),
        mean_top10_hit_delta=("top10_hit_delta", "mean"),
        mean_actual_capture_delta=("top20_actual_mean_delta", "mean"),
        recall_improved=(
            "top20_recall_delta",
            lambda x: int((x > 0).sum()),
        ),
        top10_hit_improved=(
            "top10_hit_delta",
            lambda x: int((x > 0).sum()),
        ),
        capture_improved=(
            "top20_actual_mean_delta",
            lambda x: int((x > 0).sum()),
        ),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("=== OVERALL ===")

print(
    "TOP20_RECALL_IMPROVED_FOLDS="
    f"{(out['top20_recall_delta'] > 0).sum()}/{len(out)}"
)

print(
    "TOP10_HIT_IMPROVED_FOLDS="
    f"{(out['top10_hit_delta'] > 0).sum()}/{len(out)}"
)

print(
    "ACTUAL_CAPTURE_IMPROVED_FOLDS="
    f"{(out['top20_actual_mean_delta'] > 0).sum()}/{len(out)}"
)

print(
    "MEAN_TOP20_RECALL_DELTA="
    f"{out['top20_recall_delta'].mean():.6f}"
)

print(
    "MEAN_TOP10_HIT_DELTA="
    f"{out['top10_hit_delta'].mean():.6f}"
)

print(
    "MEAN_ACTUAL_CAPTURE_DELTA="
    f"{out['top20_actual_mean_delta'].mean():.6f}"
)

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
