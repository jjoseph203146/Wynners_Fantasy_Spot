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
SIT_PATH = ROOT / "data/research/situational_role_history_v1.parquet"
STATS_PATH = ROOT / "data/parquet/nfl_player_game_stats.parquet"
BUILDER_PATH = ROOT / "current_offensive_stat_forecast.py"

print("=== STEP 7K-B6C — E14 HIGH-VALUE SHARE WEEKLY STABILITY ===")
print("MODE=ANALYSIS_ONLY")
print("PRODUCTION_INFLUENCE=NO")

# ------------------------------------------------------------
# Load exact canonical E14 feature contract
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "current_offensive_stat_forecast",
    BUILDER_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_, e14_features = mod.load_feature_contract()

e14_features = list(e14_features)

if len(e14_features) != 79:
    raise RuntimeError(
        f"UNEXPECTED_E14_FEATURE_COUNT:{len(e14_features)}"
    )

# ------------------------------------------------------------
# Authorities
# ------------------------------------------------------------

mi = pd.read_parquet(MI_PATH)
player = pd.read_parquet(PLAYER_PATH)
team = pd.read_parquet(TEAM_PATH)
sit = pd.read_parquet(SIT_PATH)
stats = pd.read_parquet(STATS_PATH)

for name, df in [
    ("MI", mi),
    ("PLAYER", player),
    ("TEAM", team),
    ("SIT", sit),
    ("STATS", stats),
]:
    if df.duplicated(["game_id", "player_id"]).any() if name in {
        "MI", "PLAYER", "SIT", "STATS"
    } else False:
        raise RuntimeError(f"{name}_GAME_PLAYER_DUP")

if team.duplicated(["game_id", "team"]).any():
    raise RuntimeError("TEAM_GAME_TEAM_DUP")

# ------------------------------------------------------------
# Historical E14 information set
# ------------------------------------------------------------

base = mi[
    [
        "game_id",
        "season",
        "week",
        "player_id",
        "position",
        "team",
        "opponent_team",
    ]
].copy()

player_features = []

for feature in e14_features:
    if feature in player.columns:
        player_features.append(feature)

player_payload = list(
    dict.fromkeys(
        ["game_id", "player_id", "team"]
        + player_features
        + ["history_games"]
    )
)

p = player[player_payload].copy()

if "history_games_player" not in p.columns:
    p = p.rename(
        columns={
            "history_games":
            "history_games_player"
        }
    )

base = base.merge(
    p,
    on=["game_id", "player_id", "team"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_player"),
)

team_features = [
    f
    for f in e14_features
    if f in team.columns
    and f not in base.columns
]

team_payload = list(
    dict.fromkeys(
        ["game_id", "team"]
        + team_features
        + ["history_games"]
    )
)

t = team[team_payload].copy()

if "history_games_team" not in t.columns:
    t = t.rename(
        columns={
            "history_games":
            "history_games_team"
        }
    )

base = base.merge(
    t,
    on=["game_id", "team"],
    how="left",
    validate="many_to_one",
)

missing_e14 = [
    c for c in e14_features
    if c not in base.columns
]

if missing_e14:
    raise RuntimeError(
        f"E14_MISSING:{missing_e14}"
    )

e14_nulls = int(
    base[e14_features].isna().sum().sum()
)

if e14_nulls:
    raise RuntimeError(
        f"E14_NULL_CELLS:{e14_nulls}"
    )

# ------------------------------------------------------------
# Exact frozen 6W/6X high-value-share representation
# ------------------------------------------------------------

sit_required = [
    "game_id",
    "player_id",
    "g2g_carries_3g",
    "i10_targets_3g",
]

missing_sit = [
    c for c in sit_required
    if c not in sit.columns
]

if missing_sit:
    raise RuntimeError(
        f"SITUATIONAL_MISSING:{missing_sit}"
    )

base = base.merge(
    sit[sit_required],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

if base[
    ["g2g_carries_3g", "i10_targets_3g"]
].isna().any().any():
    raise RuntimeError(
        "SITUATIONAL_JOIN_MISSING"
    )

contracts = {
    "RB": {
        "absolute": "g2g_carries_3g",
        "share": "rb_hv_share",
        "available": "rb_hv_available",
        "targets": [
            "carries",
            "rushing_yards",
            "targets",
            "receptions",
            "receiving_yards",
        ],
    },
    "TE": {
        "absolute": "i10_targets_3g",
        "share": "te_hv_share",
        "available": "te_hv_available",
        "targets": [
            "targets",
            "receptions",
            "receiving_yards",
        ],
    },
}

for pos, cfg in contracts.items():

    mask = base["position"].eq(pos)

    denom = (
        base.loc[mask]
        .groupby(["game_id", "team"])[
            cfg["absolute"]
        ]
        .transform("sum")
        .astype(float)
    )

    numerator = (
        base.loc[mask, cfg["absolute"]]
        .to_numpy(dtype=float)
    )

    denominator = denom.to_numpy(dtype=float)

    share = np.divide(
        numerator,
        denominator,
        out=np.zeros(
            len(numerator),
            dtype=float,
        ),
        where=denominator > 0,
    )

    base.loc[
        mask,
        cfg["share"],
    ] = share

    base.loc[
        mask,
        cfg["available"],
    ] = (
        denominator > 0
    ).astype(int)

# ------------------------------------------------------------
# Actual component targets
# ------------------------------------------------------------

target_columns = sorted(
    {
        target
        for cfg in contracts.values()
        for target in cfg["targets"]
    }
)

stats_payload = (
    ["game_id", "player_id"]
    + target_columns
)

base = base.merge(
    stats[stats_payload],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# Minimum-history contract
#
# E14 aliases:
# player.history_games -> history_games_player
# team.history_games   -> history_games_team
#
# MI supplies dvp_history_games.
# ------------------------------------------------------------

if "dvp_history_games" not in mi.columns:
    raise RuntimeError(
        "MI_DVP_HISTORY_GAMES_MISSING"
    )

base = base.merge(
    mi[
        [
            "game_id",
            "player_id",
            "dvp_history_games",
        ]
    ],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

base = base[
    (base["history_games_player"] >= 3)
    & (base["history_games_team"] >= 3)
    & (base["dvp_history_games"] >= 3)
].copy()

print(f"ROWS_AFTER_MIN_HISTORY={len(base)}")
print(f"E14_FEATURE_COUNT={len(e14_features)}")
print(f"E14_NULL_CELLS={int(base[e14_features].isna().sum().sum())}")

for pos, cfg in contracts.items():

    x = base[
        base["position"].eq(pos)
    ]

    share = cfg["share"]
    available = cfg["available"]

    invalid = int(
        (
            (x[share] < 0)
            | (x[share] > 1)
        ).sum()
    )

    print(
        f"{pos}_ROWS={len(x)}:"
        f"AVAILABLE={int(x[available].sum())}:"
        f"UNAVAILABLE={int((x[available] == 0).sum())}:"
        f"INVALID_SHARE={invalid}"
    )

    if invalid:
        raise RuntimeError(
            f"{pos}_INVALID_SHARE"
        )

# ------------------------------------------------------------
# Season-forward weekly stability for predeclared B6B candidates
# ------------------------------------------------------------
#
# RB:
#   g2g share + availability
#   targets = carries, rushing_yards
#
# TE:
#   i10 share + availability
#   targets = receptions, receiving_yards
#
# One model pair per target per season-forward split.
# Weekly evaluation uses those same predictions.
# No weekly refit.
# ------------------------------------------------------------

WEEKLY_CONTRACTS = {
    "RB": ["carries", "rushing_yards"],
    "TE": ["receptions", "receiving_yards"],
}

TEST_SEASONS = [2024, 2025, 2026]

season_results = []
weekly_results = []

for pos, target_list in WEEKLY_CONTRACTS.items():

    cfg = contracts[pos]

    pos_df = base[
        base["position"].eq(pos)
    ].copy()

    incremental_features = (
        e14_features
        + [
            cfg["share"],
            cfg["available"],
        ]
    )

    for target in target_list:

        pos_df[target] = pd.to_numeric(
            pos_df[target],
            errors="coerce",
        )

        work = pos_df[
            pos_df[target].notna()
        ].copy()

        for season in TEST_SEASONS:

            train = work[
                work["season"] < season
            ].copy()

            test = work[
                work["season"] == season
            ].copy()

            if train.empty or test.empty:
                continue

            xb_tr = train[
                e14_features
            ].to_numpy(dtype=float)

            xb_te = test[
                e14_features
            ].to_numpy(dtype=float)

            xi_tr = train[
                incremental_features
            ].to_numpy(dtype=float)

            xi_te = test[
                incremental_features
            ].to_numpy(dtype=float)

            y_tr = train[target].to_numpy(
                dtype=float
            )

            y_te = test[target].to_numpy(
                dtype=float
            )

            baseline = HistGradientBoostingRegressor(
                random_state=0,
            )

            incremental = HistGradientBoostingRegressor(
                random_state=0,
            )

            baseline.fit(
                xb_tr,
                y_tr,
            )

            incremental.fit(
                xi_tr,
                y_tr,
            )

            pb = baseline.predict(xb_te)
            pi = incremental.predict(xi_te)

            mae_b = mean_absolute_error(
                y_te,
                pb,
            )
            mae_i = mean_absolute_error(
                y_te,
                pi,
            )

            rmse_b = mean_squared_error(
                y_te,
                pb,
            ) ** 0.5
            rmse_i = mean_squared_error(
                y_te,
                pi,
            ) ** 0.5

            if (
                len(y_te) >= 2
                and np.std(y_te) > 0
                and np.std(pb) > 0
                and np.std(pi) > 0
            ):
                corr_b = np.corrcoef(
                    y_te,
                    pb,
                )[0, 1]

                corr_i = np.corrcoef(
                    y_te,
                    pi,
                )[0, 1]

                corr_delta = corr_i - corr_b
            else:
                corr_delta = np.nan

            season_results.append(
                {
                    "position": pos,
                    "target": target,
                    "season": season,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "mae_delta": mae_i - mae_b,
                    "rmse_delta": rmse_i - rmse_b,
                    "corr_delta": corr_delta,
                }
            )

            scored = test[
                [
                    "season",
                    "week",
                    "game_id",
                    "player_id",
                ]
            ].copy()

            scored["actual"] = y_te
            scored["pred_base"] = pb
            scored["pred_inc"] = pi

            for week, q in scored.groupby(
                "week",
                sort=True,
            ):

                actual = q[
                    "actual"
                ].to_numpy(dtype=float)

                pred_base = q[
                    "pred_base"
                ].to_numpy(dtype=float)

                pred_inc = q[
                    "pred_inc"
                ].to_numpy(dtype=float)

                w_mae_b = mean_absolute_error(
                    actual,
                    pred_base,
                )

                w_mae_i = mean_absolute_error(
                    actual,
                    pred_inc,
                )

                w_rmse_b = mean_squared_error(
                    actual,
                    pred_base,
                ) ** 0.5

                w_rmse_i = mean_squared_error(
                    actual,
                    pred_inc,
                ) ** 0.5

                if (
                    len(actual) >= 2
                    and np.std(actual) > 0
                    and np.std(pred_base) > 0
                    and np.std(pred_inc) > 0
                ):
                    w_corr_b = np.corrcoef(
                        actual,
                        pred_base,
                    )[0, 1]

                    w_corr_i = np.corrcoef(
                        actual,
                        pred_inc,
                    )[0, 1]

                    w_corr_delta = (
                        w_corr_i - w_corr_b
                    )
                else:
                    w_corr_delta = np.nan

                weekly_results.append(
                    {
                        "position": pos,
                        "target": target,
                        "season": int(season),
                        "week": int(week),
                        "rows": len(q),
                        "mae_delta": (
                            w_mae_i - w_mae_b
                        ),
                        "rmse_delta": (
                            w_rmse_i - w_rmse_b
                        ),
                        "corr_delta": w_corr_delta,
                    }
                )

season_r = pd.DataFrame(season_results)
week_r = pd.DataFrame(weekly_results)

if season_r.empty or week_r.empty:
    raise RuntimeError(
        "NO_HIGH_VALUE_WEEKLY_RESULTS"
    )

print()
print("=== SEASON RECONCILIATION ===")

for row in season_r.itertuples(index=False):
    print(
        f"POSITION={row.position}:"
        f"TARGET={row.target}:"
        f"SEASON={row.season}:"
        f"TRAIN={row.train_rows}:"
        f"TEST={row.test_rows}:"
        f"MAE_DELTA={row.mae_delta:+.6f}:"
        f"RMSE_DELTA={row.rmse_delta:+.6f}:"
        f"CORR_DELTA={row.corr_delta:+.6f}"
    )

print()
print("=== WEEKLY STABILITY BY TARGET ===")

for pos, target_list in WEEKLY_CONTRACTS.items():

    for target in target_list:

        q = week_r[
            week_r["position"].eq(pos)
            & week_r["target"].eq(target)
        ].copy()

        corr = q[
            q["corr_delta"].notna()
        ]

        print(
            f"POSITION={pos}:"
            f"TARGET={target}:"
            f"WEEKS={len(q)}:"
            f"MAE_WINS={int((q['mae_delta'] < 0).sum())}:"
            f"RMSE_WINS={int((q['rmse_delta'] < 0).sum())}:"
            f"CORR_VALID_WEEKS={len(corr)}:"
            f"CORR_WINS={int((corr['corr_delta'] > 0).sum())}:"
            f"MEDIAN_MAE_DELTA={q['mae_delta'].median():+.6f}:"
            f"MEDIAN_RMSE_DELTA={q['rmse_delta'].median():+.6f}:"
            f"MEDIAN_CORR_DELTA={corr['corr_delta'].median():+.6f}"
        )

print()
print("=== WEEKLY STABILITY BY SEASON + TARGET ===")

for pos, target_list in WEEKLY_CONTRACTS.items():

    for target in target_list:

        for season in TEST_SEASONS:

            q = week_r[
                week_r["position"].eq(pos)
                & week_r["target"].eq(target)
                & week_r["season"].eq(season)
            ].copy()

            if q.empty:
                continue

            corr = q[
                q["corr_delta"].notna()
            ]

            print(
                f"POSITION={pos}:"
                f"TARGET={target}:"
                f"SEASON={season}:"
                f"WEEKS={len(q)}:"
                f"MAE_WINS={int((q['mae_delta'] < 0).sum())}:"
                f"RMSE_WINS={int((q['rmse_delta'] < 0).sum())}:"
                f"CORR_VALID_WEEKS={len(corr)}:"
                f"CORR_WINS={int((corr['corr_delta'] > 0).sum())}:"
                f"MEDIAN_MAE_DELTA={q['mae_delta'].median():+.6f}:"
                f"MEDIAN_RMSE_DELTA={q['rmse_delta'].median():+.6f}:"
                f"MEDIAN_CORR_DELTA={corr['corr_delta'].median():+.6f}"
            )

print()
print("=== POSITION WEEKLY SUMMARY ===")

for pos in ["RB", "TE"]:

    q = week_r[
        week_r["position"].eq(pos)
    ]

    corr = q[
        q["corr_delta"].notna()
    ]

    print(
        f"POSITION={pos}:"
        f"TARGET_WEEK_TESTS={len(q)}:"
        f"MAE_WINS={int((q['mae_delta'] < 0).sum())}:"
        f"RMSE_WINS={int((q['rmse_delta'] < 0).sum())}:"
        f"CORR_VALID={len(corr)}:"
        f"CORR_WINS={int((corr['corr_delta'] > 0).sum())}:"
        f"MEDIAN_MAE_DELTA={q['mae_delta'].median():+.6f}:"
        f"MEDIAN_RMSE_DELTA={q['rmse_delta'].median():+.6f}:"
        f"MEDIAN_CORR_DELTA={corr['corr_delta'].median():+.6f}"
    )

print()
print("=== B6C CONTRACT ===")
print("BASELINE=EXACT_79_FEATURE_E14_INFORMATION_SET")
print("RB_INCREMENT=rb_hv_share+rb_hv_available")
print("RB_TARGETS=carries,rushing_yards")
print("TE_INCREMENT=te_hv_share+te_hv_available")
print("TE_TARGETS=receptions,receiving_yards")
print("SHARE_SEMANTICS=EXACT_6W_6X")
print("WEEKLY_MODEL_REFIT=NO")
print("SEASON_FORWARD_PREDICTIONS=YES")
print("FEATURE_SEARCH=NO")
print("COMBINATION_SEARCH=NO")
print("PRODUCTION_PROMOTION=NOT_AUTHORIZED")


print()
print("=== SAFETY ===")
print("RESEARCH_MODEL_TRAINING=YES")
print("PRODUCTION_MODEL_TRAINING=NO")
print("CANONICAL_MODELS_MODIFIED=NO")
print("PRODUCTION_ARTIFACT_WRITES=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("UPDATER_RUN=NO")
print("STEP_7K_B6C_STATUS=PASS")
