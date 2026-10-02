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

def evaluate_weekly(test, base_pred, matchup_pred):
    z = test[
        ["season", "week", "position", "fanduel_points"]
    ].copy()

    z["base_pred"] = base_pred
    z["matchup_pred"] = matchup_pred

    rows = []

    for (season, week, pos), g in z.groupby(
        ["season", "week", "position"],
        sort=True,
    ):
        if len(g) < 5:
            continue

        n20 = max(1, int(np.ceil(len(g) * 0.20)))
        n10 = max(1, int(np.ceil(len(g) * 0.10)))

        actual_top20 = set(
            g.nlargest(n20, "fanduel_points").index
        )

        base_top20 = set(
            g.nlargest(n20, "base_pred").index
        )

        matchup_top20 = set(
            g.nlargest(n20, "matchup_pred").index
        )

        base_top10 = set(
            g.nlargest(n10, "base_pred").index
        )

        matchup_top10 = set(
            g.nlargest(n10, "matchup_pred").index
        )

        base_recall20 = (
            len(actual_top20 & base_top20)
            / len(actual_top20)
        )

        matchup_recall20 = (
            len(actual_top20 & matchup_top20)
            / len(actual_top20)
        )

        base_hit10 = (
            len(actual_top20 & base_top10)
            / len(base_top10)
        )

        matchup_hit10 = (
            len(actual_top20 & matchup_top10)
            / len(matchup_top10)
        )

        base_capture = g.loc[
            list(base_top20), "fanduel_points"
        ].mean()

        matchup_capture = g.loc[
            list(matchup_top20), "fanduel_points"
        ].mean()

        rows.append({
            "season": season,
            "week": week,
            "position": pos,
            "players": len(g),
            "top20_recall_delta":
                matchup_recall20 - base_recall20,
            "top10_hit_delta":
                matchup_hit10 - base_hit10,
            "actual_capture_delta":
                matchup_capture - base_capture,
        })

    return pd.DataFrame(rows)

print("=== MATCHUP INTELLIGENCE STEP 6L — WEEKLY CEILING RANKING ===")
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

results = []

for test_season in [2024, 2025, 2026]:

    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]

    for pos in ["QB", "RB", "WR", "TE"]:

        tr = train[train["position"].eq(pos)]
        te = test[test["position"].eq(pos)].copy()

        if tr.empty or te.empty:
            continue

        ytr = tr["fanduel_points"].astype(float)

        base = make_model()
        base.fit(tr[PLAYER], ytr)
        bp = base.predict(te[PLAYER])

        match = make_model()
        match.fit(tr[MATCHUP], ytr)
        mp = match.predict(te[MATCHUP])

        weekly = evaluate_weekly(te, bp, mp)

        if weekly.empty:
            continue

        weekly["test_season"] = test_season
        results.append(weekly)

if not results:
    raise RuntimeError("NO_WEEKLY_RESULTS")

out = pd.concat(results, ignore_index=True)

print()
print("=== SEASON × POSITION ===")

season_summary = (
    out.groupby(["test_season", "position"])
    .agg(
        weeks=("week", "size"),
        mean_recall_delta=("top20_recall_delta", "mean"),
        mean_top10_hit_delta=("top10_hit_delta", "mean"),
        mean_capture_delta=("actual_capture_delta", "mean"),
        recall_win_weeks=(
            "top20_recall_delta",
            lambda x: int((x > 0).sum()),
        ),
        recall_loss_weeks=(
            "top20_recall_delta",
            lambda x: int((x < 0).sum()),
        ),
        top10_win_weeks=(
            "top10_hit_delta",
            lambda x: int((x > 0).sum()),
        ),
        capture_win_weeks=(
            "actual_capture_delta",
            lambda x: int((x > 0).sum()),
        ),
    )
    .reset_index()
)

print(season_summary.to_string(index=False))

print()
print("=== POSITION SUMMARY ===")

position_summary = (
    out.groupby("position")
    .agg(
        weeks=("week", "size"),
        mean_recall_delta=("top20_recall_delta", "mean"),
        mean_top10_hit_delta=("top10_hit_delta", "mean"),
        mean_capture_delta=("actual_capture_delta", "mean"),
        recall_win_weeks=(
            "top20_recall_delta",
            lambda x: int((x > 0).sum()),
        ),
        recall_loss_weeks=(
            "top20_recall_delta",
            lambda x: int((x < 0).sum()),
        ),
        top10_win_weeks=(
            "top10_hit_delta",
            lambda x: int((x > 0).sum()),
        ),
        top10_loss_weeks=(
            "top10_hit_delta",
            lambda x: int((x < 0).sum()),
        ),
        capture_win_weeks=(
            "actual_capture_delta",
            lambda x: int((x > 0).sum()),
        ),
        capture_loss_weeks=(
            "actual_capture_delta",
            lambda x: int((x < 0).sum()),
        ),
    )
    .reset_index()
)

print(position_summary.to_string(index=False))

print()
print("=== OVERALL ===")

print(f"WEEK_POSITION_GROUPS={len(out)}")

print(
    "RECALL_WIN_GROUPS="
    f"{(out['top20_recall_delta'] > 0).sum()}"
)

print(
    "RECALL_LOSS_GROUPS="
    f"{(out['top20_recall_delta'] < 0).sum()}"
)

print(
    "TOP10_WIN_GROUPS="
    f"{(out['top10_hit_delta'] > 0).sum()}"
)

print(
    "TOP10_LOSS_GROUPS="
    f"{(out['top10_hit_delta'] < 0).sum()}"
)

print(
    "CAPTURE_WIN_GROUPS="
    f"{(out['actual_capture_delta'] > 0).sum()}"
)

print(
    "CAPTURE_LOSS_GROUPS="
    f"{(out['actual_capture_delta'] < 0).sum()}"
)

print(
    "MEAN_RECALL_DELTA="
    f"{out['top20_recall_delta'].mean():.6f}"
)

print(
    "MEAN_TOP10_HIT_DELTA="
    f"{out['top10_hit_delta'].mean():.6f}"
)

print(
    "MEAN_CAPTURE_DELTA="
    f"{out['actual_capture_delta'].mean():.6f}"
)

print()
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
