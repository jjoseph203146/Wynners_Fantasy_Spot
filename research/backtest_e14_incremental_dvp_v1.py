from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/home/mwynn/nfl_data_engine")

MI_PATH = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
PLAYER_PATH = ROOT / "data/parquet/nfl_player_pregame_features.parquet"
TEAM_PATH = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
STATS_PATH = ROOT / "data/parquet/nfl_player_game_stats.parquet"
BUILDER_PATH = ROOT / "current_offensive_stat_forecast.py"

print("=== STEP 7K-B5D — E14 VS E14 + POSITION DVP ===")
print("MODE=ANALYSIS_ONLY")
print("PRODUCTION_INFLUENCE=NO")

# ------------------------------------------------------------
# Load exact production E14 feature contract
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "e14_contract",
    BUILDER_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_, e14_features = mod.load_feature_contract()

if len(e14_features) != 79:
    raise RuntimeError(
        f"E14_FEATURE_COUNT_INVALID:{len(e14_features)}"
    )

# ------------------------------------------------------------
# Load historical authorities
# ------------------------------------------------------------

mi = pd.read_parquet(MI_PATH).copy()
player = pd.read_parquet(PLAYER_PATH).copy()
team = pd.read_parquet(TEAM_PATH).copy()
stats = pd.read_parquet(STATS_PATH).copy()

for df in [mi, player, team, stats]:
    df["game_id"] = df["game_id"].astype(str)

for df in [mi, player, stats]:
    df["player_id"] = df["player_id"].astype(str)

for df in [mi, player, team]:
    df["team"] = df["team"].astype(str)

# Restrict to MI historical skill-position spine.
mi = mi[
    mi["position"].isin(["QB", "RB", "WR", "TE"])
].copy()

if mi.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("MI_DUPLICATE_GAME_PLAYER")

# ------------------------------------------------------------
# Build exact historical E14 feature frame
# ------------------------------------------------------------

player_features = []

for feature in e14_features:
    if feature in player.columns:
        player_features.append(feature)

player_keep = list(dict.fromkeys(
    ["game_id", "player_id", "team"]
    + player_features
    + (
        ["history_games"]
        if "history_games_player" in e14_features
        else []
    )
))

p = player[player_keep].copy()

if "history_games_player" in e14_features:
    if "history_games" not in p.columns:
        raise RuntimeError(
            "PLAYER_HISTORY_AUTHORITY_MISSING"
        )

    p = p.rename(
        columns={
            "history_games": "history_games_player"
        }
    )

team_features = []

for feature in e14_features:
    if feature in team.columns:
        team_features.append(feature)

team_keep = list(dict.fromkeys(
    ["game_id", "team"]
    + team_features
    + (
        ["history_games"]
        if "history_games_team" in e14_features
        else []
    )
))

t = team[team_keep].copy()

if "history_games_team" in e14_features:
    if "history_games" not in t.columns:
        raise RuntimeError(
            "TEAM_HISTORY_AUTHORITY_MISSING"
        )

    t = t.rename(
        columns={
            "history_games": "history_games_team"
        }
    )

# Avoid duplicated E14 columns where season/week exist
# in both player and team authorities. Player is canonical.
team_payload = [
    c for c in t.columns
    if c in {"game_id", "team"}
    or c not in p.columns
]

t = t[team_payload].copy()

base = mi[
    [
        "game_id",
        "player_id",
        "position",
        "team",
        "season",
        "week",
        "fd_allowed_avg_3",
        "targets_allowed_avg_3",
        "carries_allowed_avg_3",
        "opportunities_allowed_avg_3",
    ]
].copy()

base = base.merge(
    p,
    on=["game_id", "player_id", "team"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_player"),
)

base = base.merge(
    t,
    on=["game_id", "team"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_team"),
)

# Resolve season/week to MI spine where duplicate historical
# authority names were generated during merge.
for feature in ["season", "week"]:
    if feature not in base.columns:
        raise RuntimeError(
            f"E14_METADATA_MISSING:{feature}"
        )

missing_e14 = [
    c for c in e14_features
    if c not in base.columns
]

if missing_e14:
    raise RuntimeError(
        f"E14_FEATURES_MISSING:{missing_e14}"
    )

# ------------------------------------------------------------
# Actual targets
# ------------------------------------------------------------

targets = [
    "attempts",
    "completions",
    "passing_yards",
    "carries",
    "rushing_yards",
    "passing_tds",
    "interceptions",
    "targets",
    "receptions",
    "receiving_yards",
]

actual = stats[
    ["game_id", "player_id"] + targets
].copy()

base = base.merge(
    actual,
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# Model-group/target contract
# ------------------------------------------------------------

contracts = {
    "QB": [
        "attempts",
        "completions",
        "passing_yards",
        "carries",
        "rushing_yards",
        "passing_tds",
        "interceptions",
    ],
    "RB": [
        "carries",
        "rushing_yards",
        "targets",
        "receptions",
        "receiving_yards",
    ],
    "WR": [
        "targets",
        "receptions",
        "receiving_yards",
    ],
    "TE": [
        "targets",
        "receptions",
        "receiving_yards",
    ],
}

dvp_features = [
    "fd_allowed_avg_3",
    "targets_allowed_avg_3",
    "carries_allowed_avg_3",
    "opportunities_allowed_avg_3",
]

# ------------------------------------------------------------
# Numeric validation
# ------------------------------------------------------------

for c in e14_features + dvp_features:
    base[c] = pd.to_numeric(
        base[c],
        errors="coerce",
    )

e14_bad = int(
    base[e14_features]
    .isna()
    .sum()
    .sum()
)

dvp_bad = int(
    base[dvp_features]
    .isna()
    .sum()
    .sum()
)

print(f"ROWS={len(base)}")
print(f"E14_FEATURE_COUNT={len(e14_features)}")
print(f"E14_NULL_CELLS={e14_bad}")
print(f"DVP_FEATURE_COUNT={len(dvp_features)}")
print(f"DVP_NULL_CELLS={dvp_bad}")

if e14_bad:
    raise RuntimeError(
        f"E14_NULL_FEATURE_CELLS:{e14_bad}"
    )

if dvp_bad:
    raise RuntimeError(
        f"DVP_NULL_FEATURE_CELLS:{dvp_bad}"
    )

# ------------------------------------------------------------
# Season-forward evaluation
#
# Train only seasons strictly before test season.
# 2023 therefore acts as history/training seed.
# Test seasons: 2024, 2025, 2026.
# ------------------------------------------------------------

results = []

for position, position_targets in contracts.items():

    pos = base[
        base["position"].eq(position)
    ].copy()

    for target in position_targets:

        pos[target] = pd.to_numeric(
            pos[target],
            errors="coerce",
        )

        work = pos[
            pos[target].notna()
        ].copy()

        for test_season in [2024, 2025, 2026]:

            train = work[
                work["season"] < test_season
            ].copy()

            test = work[
                work["season"] == test_season
            ].copy()

            if train.empty or test.empty:
                continue

            x_train_base = train[
                e14_features
            ].to_numpy(dtype=float)

            x_test_base = test[
                e14_features
            ].to_numpy(dtype=float)

            x_train_mi = train[
                e14_features + dvp_features
            ].to_numpy(dtype=float)

            x_test_mi = test[
                e14_features + dvp_features
            ].to_numpy(dtype=float)

            y_train = train[target].to_numpy(
                dtype=float
            )

            y_test = test[target].to_numpy(
                dtype=float
            )

            baseline = HistGradientBoostingRegressor(
                random_state=0,
            )

            incremental = HistGradientBoostingRegressor(
                random_state=0,
            )

            baseline.fit(
                x_train_base,
                y_train,
            )

            incremental.fit(
                x_train_mi,
                y_train,
            )

            pred_base = baseline.predict(
                x_test_base
            )

            pred_mi = incremental.predict(
                x_test_mi
            )

            mae_base = mean_absolute_error(
                y_test,
                pred_base,
            )

            mae_mi = mean_absolute_error(
                y_test,
                pred_mi,
            )

            rmse_base = mean_squared_error(
                y_test,
                pred_base,
            ) ** 0.5

            rmse_mi = mean_squared_error(
                y_test,
                pred_mi,
            ) ** 0.5

            if len(y_test) >= 2:
                corr_base = np.corrcoef(
                    y_test,
                    pred_base,
                )[0, 1]

                corr_mi = np.corrcoef(
                    y_test,
                    pred_mi,
                )[0, 1]
            else:
                corr_base = np.nan
                corr_mi = np.nan

            results.append(
                {
                    "position": position,
                    "target": target,
                    "test_season": test_season,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "mae_e14": mae_base,
                    "mae_e14_dvp": mae_mi,
                    "mae_delta": mae_mi - mae_base,
                    "rmse_e14": rmse_base,
                    "rmse_e14_dvp": rmse_mi,
                    "rmse_delta": rmse_mi - rmse_base,
                    "corr_e14": corr_base,
                    "corr_e14_dvp": corr_mi,
                    "corr_delta": corr_mi - corr_base,
                }
            )

r = pd.DataFrame(results)

if r.empty:
    raise RuntimeError("NO_BACKTEST_RESULTS")

print()
print("=== SEASON-FORWARD RESULTS ===")

for row in r.itertuples(index=False):
    print(
        f"{row.position}:{row.target}:"
        f"SEASON={row.test_season}:"
        f"TRAIN={row.train_rows}:"
        f"TEST={row.test_rows}:"
        f"MAE_DELTA={row.mae_delta:+.6f}:"
        f"RMSE_DELTA={row.rmse_delta:+.6f}:"
        f"CORR_DELTA={row.corr_delta:+.6f}"
    )

print()
print("=== POSITION SUMMARY ===")

for position in ["QB", "RB", "WR", "TE"]:

    q = r[
        r["position"].eq(position)
    ]

    if q.empty:
        continue

    print(
        f"POSITION={position}:"
        f"TESTS={len(q)}:"
        f"MAE_WINS={int((q['mae_delta'] < 0).sum())}:"
        f"RMSE_WINS={int((q['rmse_delta'] < 0).sum())}:"
        f"CORR_WINS={int((q['corr_delta'] > 0).sum())}:"
        f"MEDIAN_MAE_DELTA={q['mae_delta'].median():+.6f}:"
        f"MEDIAN_RMSE_DELTA={q['rmse_delta'].median():+.6f}:"
        f"MEDIAN_CORR_DELTA={q['corr_delta'].median():+.6f}"
    )

print()
print("=== OVERALL ===")
print(f"TOTAL_TESTS={len(r)}")
print(f"MAE_WINS={int((r['mae_delta'] < 0).sum())}")
print(f"RMSE_WINS={int((r['rmse_delta'] < 0).sum())}")
print(f"CORR_WINS={int((r['corr_delta'] > 0).sum())}")
print(f"MEDIAN_MAE_DELTA={r['mae_delta'].median():+.6f}")
print(f"MEDIAN_RMSE_DELTA={r['rmse_delta'].median():+.6f}")
print(f"MEDIAN_CORR_DELTA={r['corr_delta'].median():+.6f}")

print()
print("=== SAFETY ===")
print("RESEARCH_MODEL_TRAINING=YES")
print("PRODUCTION_MODEL_TRAINING=NO")
print("CANONICAL_MODELS_MODIFIED=NO")
print("PRODUCTION_ARTIFACT_WRITES=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("UPDATER_RUN=NO")
print("STEP_7K_B5D_STATUS=PASS")
