from pathlib import Path
import hashlib
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/home/mwynn/nfl_data_engine")

MI_PATH = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
PLAYER_PATH = ROOT / "data/parquet/nfl_player_pregame_features.parquet"
TEAM_PATH = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
SIT_PATH = ROOT / "data/research/situational_role_history_v1.parquet"
STATS_PATH = ROOT / "data/parquet/nfl_player_game_stats.parquet"
FORECAST_PATH = ROOT / "current_offensive_stat_forecast.py"

OUT_DIR = ROOT / "data/research/models/matchup_intelligence_v1"

TARGET_SEASON = 2026
TARGET_WEEK = 3

print("=== STEP 7L-I — TRAIN ISOLATED MI CANDIDATE MODELS ===")
print("MODE=ANALYSIS_ONLY")
print(f"TARGET={TARGET_SEASON}_WEEK_{TARGET_WEEK}")

# ------------------------------------------------------------
# Exact frozen E14 feature contract
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "current_offensive_stat_forecast",
    FORECAST_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_, e14_features = mod.load_feature_contract()
e14_features = list(e14_features)

if len(e14_features) != 79:
    raise RuntimeError(
        f"E14_FEATURE_COUNT_INVALID:{len(e14_features)}"
    )

# ------------------------------------------------------------
# Authorities
# ------------------------------------------------------------

mi = pd.read_parquet(MI_PATH).copy()
player = pd.read_parquet(PLAYER_PATH).copy()
team = pd.read_parquet(TEAM_PATH).copy()
sit = pd.read_parquet(SIT_PATH).copy()
stats = pd.read_parquet(STATS_PATH).copy()

for df in [mi, player, sit, stats]:
    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)

for df in [mi, player, team]:
    df["team"] = df["team"].astype(str)

mi = mi[
    mi["position"].isin(["QB", "RB", "WR", "TE"])
].copy()

if mi.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("MI_GAME_PLAYER_DUP")

if player.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("PLAYER_GAME_PLAYER_DUP")

if team.duplicated(["game_id", "team"]).any():
    raise RuntimeError("TEAM_GAME_TEAM_DUP")

if sit.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("SIT_GAME_PLAYER_DUP")

if stats.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("STATS_GAME_PLAYER_DUP")

# Strict PIT boundary.
mi = mi[
    (mi["season"] < TARGET_SEASON)
    | (
        (mi["season"] == TARGET_SEASON)
        & (mi["week"] < TARGET_WEEK)
    )
].copy()

if mi.empty:
    raise RuntimeError("EMPTY_TRAINING_SPINE")

if (
    (mi["season"] > TARGET_SEASON)
    | (
        (mi["season"] == TARGET_SEASON)
        & (mi["week"] >= TARGET_WEEK)
    )
).any():
    raise RuntimeError("TARGET_OR_FUTURE_LEAKAGE")

# ------------------------------------------------------------
# Reconstruct exact E14 historical information set
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
        "dvp_history_games",
        "opportunities_allowed_avg_3",
    ]
].copy()

player_features = [
    f for f in e14_features
    if f in player.columns
]

player_payload = list(dict.fromkeys(
    ["game_id", "player_id", "team"]
    + player_features
    + ["history_games"]
))

p = player[player_payload].copy()

if "history_games_player" not in p.columns:
    p = p.rename(
        columns={"history_games": "history_games_player"}
    )

base = base.merge(
    p,
    on=["game_id", "player_id", "team"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_player"),
)

team_features = [
    f for f in e14_features
    if f in team.columns
    and f not in base.columns
]

team_payload = list(dict.fromkeys(
    ["game_id", "team"]
    + team_features
    + ["history_games"]
))

t = team[team_payload].copy()

if "history_games_team" not in t.columns:
    t = t.rename(
        columns={"history_games": "history_games_team"}
    )

team_payload = [
    c for c in t.columns
    if c in {"game_id", "team"}
    or c not in base.columns
]

t = t[team_payload].copy()

base = base.merge(
    t,
    on=["game_id", "team"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_team"),
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
# Historical high-value channels
# ------------------------------------------------------------

sit_payload = [
    "game_id",
    "player_id",
    "g2g_carries_3g",
    "i10_targets_3g",
]

base = base.merge(
    sit[sit_payload],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

# Exact research share semantics:
# denominator is position/team/game sum of the absolute channel.
base["rb_g2g_share"] = 0.0
base["rb_g2g_share_available"] = 0.0
base["te_i10_share"] = 0.0
base["te_i10_share_available"] = 0.0

rb_mask = base["position"].eq("RB")
te_mask = base["position"].eq("TE")

rb_den = (
    base.loc[rb_mask]
    .groupby(["game_id", "team"])["g2g_carries_3g"]
    .transform("sum")
)

te_den = (
    base.loc[te_mask]
    .groupby(["game_id", "team"])["i10_targets_3g"]
    .transform("sum")
)

rb_positive = rb_den > 0
te_positive = te_den > 0

base.loc[
    base.index[rb_mask][rb_positive],
    "rb_g2g_share",
] = (
    base.loc[
        base.index[rb_mask][rb_positive],
        "g2g_carries_3g",
    ].to_numpy(dtype=float)
    / rb_den[rb_positive].to_numpy(dtype=float)
)

base.loc[
    base.index[rb_mask][rb_positive],
    "rb_g2g_share_available",
] = 1.0

base.loc[
    base.index[te_mask][te_positive],
    "te_i10_share",
] = (
    base.loc[
        base.index[te_mask][te_positive],
        "i10_targets_3g",
    ].to_numpy(dtype=float)
    / te_den[te_positive].to_numpy(dtype=float)
)

base.loc[
    base.index[te_mask][te_positive],
    "te_i10_share_available",
] = 1.0

# ------------------------------------------------------------
# Actual component targets
# ------------------------------------------------------------

target_columns = [
    "carries",
    "rushing_yards",
    "targets",
    "receptions",
    "receiving_yards",
]

base = base.merge(
    stats[
        ["game_id", "player_id"] + target_columns
    ],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# Numeric validation
# ------------------------------------------------------------

numeric_columns = list(dict.fromkeys(
    e14_features
    + [
        "dvp_history_games",
        "opportunities_allowed_avg_3",
        "rb_g2g_share",
        "rb_g2g_share_available",
        "te_i10_share",
        "te_i10_share_available",
    ]
    + target_columns
))

for c in numeric_columns:
    base[c] = pd.to_numeric(
        base[c],
        errors="coerce",
    )

e14_null = int(
    base[e14_features].isna().sum().sum()
)

if e14_null:
    raise RuntimeError(
        f"E14_NULL_CELLS:{e14_null}"
    )

# ------------------------------------------------------------
# Frozen candidate contracts
# ------------------------------------------------------------

contracts = {
    "RB": {
        "targets": [
            "carries",
            "rushing_yards",
        ],
        "incremental": [
            "rb_g2g_share",
            "rb_g2g_share_available",
        ],
        "history_gate": True,
    },
    "WR": {
        "targets": [
            "targets",
            "receptions",
            "receiving_yards",
        ],
        "incremental": [
            "opportunities_allowed_avg_3",
        ],
        "history_gate": False,
    },
    "TE": {
        "targets": [
            "receptions",
            "receiving_yards",
        ],
        "incremental": [
            "te_i10_share",
            "te_i10_share_available",
        ],
        "history_gate": True,
    },
}

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

manifest = {
    "version": "matchup_intelligence_v1",
    "status": "ANALYSIS_ONLY_CANDIDATE",
    "target_season": TARGET_SEASON,
    "target_week": TARGET_WEEK,
    "history_through": "2026_WEEK_2",
    "model_class": "HistGradientBoostingRegressor",
    "random_state": 0,
    "e14_feature_count": len(e14_features),
    "e14_features": e14_features,
    "models": [],
}

trained = 0

for position, cfg in contracts.items():

    pos_df = base[
        base["position"].eq(position)
    ].copy()

    if cfg["history_gate"]:
        pos_df = pos_df[
            (pos_df["history_games_player"] >= 3)
            & (pos_df["history_games_team"] >= 3)
            & (pos_df["dvp_history_games"] >= 3)
        ].copy()

    features = (
        e14_features
        + cfg["incremental"]
    )

    expected_count = (
        80 if position == "WR"
        else 81
    )

    if len(features) != expected_count:
        raise RuntimeError(
            f"{position}_FEATURE_COUNT_INVALID:"
            f"{len(features)}"
        )

    feature_null = int(
        pos_df[features].isna().sum().sum()
    )

    if feature_null:
        raise RuntimeError(
            f"{position}_FEATURE_NULL_CELLS:"
            f"{feature_null}"
        )

    for target in cfg["targets"]:

        work = pos_df[
            pos_df[target].notna()
        ].copy()

        if work.empty:
            raise RuntimeError(
                f"{position}_{target}_EMPTY_TRAINING"
            )

        x = work[
            features
        ].to_numpy(dtype=float)

        y = work[
            target
        ].to_numpy(dtype=float)

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"{position}_{target}_NONFINITE_X"
            )

        if not np.isfinite(y).all():
            raise RuntimeError(
                f"{position}_{target}_NONFINITE_Y"
            )

        model = HistGradientBoostingRegressor(
            random_state=0,
        )

        model.fit(x, y)

        filename = (
            f"{position.lower()}__"
            f"{target}.joblib"
        )

        model_path = OUT_DIR / filename

        joblib.dump(
            model,
            model_path,
        )

        sha = hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest()

        manifest["models"].append(
            {
                "position": position,
                "target": target,
                "feature_count": len(features),
                "features": features,
                "training_rows": len(work),
                "training_seasons": sorted(
                    int(v)
                    for v in work["season"]
                    .dropna()
                    .unique()
                ),
                "latest_training_season": int(
                    work["season"].max()
                ),
                "latest_training_week_2026": (
                    int(
                        work.loc[
                            work["season"].eq(2026),
                            "week",
                        ].max()
                    )
                    if work["season"].eq(2026).any()
                    else None
                ),
                "artifact": filename,
                "sha256": sha,
            }
        )

        trained += 1

        print(
            f"TRAINED={position}:{target}:"
            f"ROWS={len(work)}:"
            f"FEATURES={len(features)}:"
            f"SHA256={sha}"
        )

manifest_path = OUT_DIR / "manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

manifest_sha = hashlib.sha256(
    manifest_path.read_bytes()
).hexdigest()

print()
print(f"TRAINED_MODEL_COUNT={trained}")
print(f"MANIFEST={manifest_path}")
print(f"MANIFEST_SHA256={manifest_sha}")

if trained != 7:
    raise RuntimeError(
        f"EXPECTED_7_MODELS_GOT:{trained}"
    )

print("STEP_7L_I_STATUS=PASS")
print("OUTPUT_SCOPE=DATA_RESEARCH_ONLY")
print("PRODUCTION_MODEL_MUTATION=NO")
print("E14_MODEL_MUTATION=NO")
print("PROJECTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("UNIFIED_FORECAST_INFLUENCE=NO")
print("UPDATER_RUN=NO")
