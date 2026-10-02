from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/mwynn/nfl_data_engine")

USAGE = ROOT / "data/parquet/nfl_player_weekly_usage.parquet"
TEAM = ROOT / "data/parquet/nfl_team_pregame_environment.parquet"
DVP = ROOT / "data/parquet/nfl_team_position_dvp.parquet"

OUTPUT = ROOT / "data/research/matchup_intelligence_history_v1.parquet"

POSITIONS = ["QB", "RB", "WR", "TE"]

PLAYER_FEATURES = [
    "usage_3g",
    "snap_pct_3g",
    "target_3g",
    "carry_3g",
    "fanduel_3g",
]

TEAM_FEATURES = [
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

DVP_FEATURES = [
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

print("=== MATCHUP INTELLIGENCE HISTORY V1 ===")
print("MODE=ANALYSIS_ONLY")

u = pd.read_parquet(USAGE)
t = pd.read_parquet(TEAM)
d = pd.read_parquet(DVP)

u = u[u["position"].isin(POSITIONS)].copy()

# Deterministic player chronology.
u = u.sort_values(
    ["player_id", "season", "week", "game_id"]
).reset_index(drop=True)

# Strictly prior completed player observations.
u["prior_player_games"] = (
    u.groupby("player_id").cumcount()
)

# Preserve only approved identity, predictor, and evaluation fields.
u = u[
    [
        "game_id",
        "season",
        "week",
        "season_type",
        "identity_key",
        "player_id",
        "player_name",
        "player_display_name",
        "position",
        "position_group",
        "team",
        "opponent_team",
        "prior_player_games",
        *PLAYER_FEATURES,
        "fanduel_points",
    ]
].copy()

# Rename history counters before joining.
t = t.rename(
    columns={"history_games": "team_history_games"}
)

d = d.rename(
    columns={"history_games": "dvp_history_games"}
)

t_keep = [
    "game_id",
    "team",
    "opponent_team",
    "team_history_games",
    *TEAM_FEATURES,
]

d_keep = [
    "game_id",
    "defense_team",
    "position",
    "dvp_history_games",
    *DVP_FEATURES,
]

# Fail closed on duplicate authorities.
if u.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("DUPLICATE_USAGE_GAME_PLAYER")

if t.duplicated(["game_id", "team"]).any():
    raise RuntimeError("DUPLICATE_TEAM_ENV_GAME_TEAM")

if d.duplicated(
    ["game_id", "defense_team", "position"]
).any():
    raise RuntimeError("DUPLICATE_DVP_GAME_DEFENSE_POSITION")

x = u.merge(
    t[t_keep],
    on=["game_id", "team"],
    how="left",
    suffixes=("", "_teamenv"),
    validate="many_to_one",
    indicator="_team_join",
)

if not x["_team_join"].eq("both").all():
    raise RuntimeError("TEAM_ENV_JOIN_INCOMPLETE")

if not (
    x["opponent_team"] == x["opponent_team_teamenv"]
).all():
    raise RuntimeError("TEAM_ENV_OPPONENT_MISMATCH")

x = x.drop(
    columns=["_team_join", "opponent_team_teamenv"]
)

x = x.merge(
    d[d_keep],
    left_on=["game_id", "opponent_team", "position"],
    right_on=["game_id", "defense_team", "position"],
    how="left",
    validate="many_to_one",
    indicator="_dvp_join",
)

if not x["_dvp_join"].eq("both").all():
    raise RuntimeError("DVP_JOIN_INCOMPLETE")

if not (
    x["opponent_team"] == x["defense_team"]
).all():
    raise RuntimeError("DVP_DEFENSE_MISMATCH")

x = x.drop(columns=["_dvp_join"])

# Core numeric integrity.
predictors = (
    PLAYER_FEATURES
    + TEAM_FEATURES
    + DVP_FEATURES
)

for c in predictors:
    s = pd.to_numeric(x[c], errors="coerce")

    if s.isna().any():
        raise RuntimeError(f"NULL_PREDICTOR:{c}")

    if not np.isfinite(s).all():
        raise RuntimeError(f"NONFINITE_PREDICTOR:{c}")

# Sample-depth integrity.
for c in [
    "prior_player_games",
    "team_history_games",
    "dvp_history_games",
]:
    if x[c].isna().any():
        raise RuntimeError(f"NULL_SAMPLE_DEPTH:{c}")

    if (x[c] < 0).any():
        raise RuntimeError(f"NEGATIVE_SAMPLE_DEPTH:{c}")

# Explicit evidence availability flags.
x["player_history_available"] = (
    x["prior_player_games"] > 0
).astype(int)

x["team_history_available"] = (
    x["team_history_games"] > 0
).astype(int)

x["dvp_history_available"] = (
    x["dvp_history_games"] > 0
).astype(int)

x["core_matchup_history_available"] = (
    x["player_history_available"].eq(1)
    & x["team_history_available"].eq(1)
    & x["dvp_history_available"].eq(1)
).astype(int)

# Deterministic output order.
x = x.sort_values(
    [
        "season",
        "week",
        "game_id",
        "team",
        "position",
        "player_id",
    ]
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
x.to_parquet(OUTPUT, index=False)

print(f"OUTPUT={OUTPUT}")
print(f"ROWS={len(x)}")
print(f"GAMES={x['game_id'].nunique()}")
print(f"PLAYERS={x['player_id'].nunique()}")
print(f"COLUMNS={len(x.columns)}")

print(
    "CORE_HISTORY_AVAILABLE_ROWS="
    f"{x['core_matchup_history_available'].sum()}"
)

print(
    "PLAYER_COLD_START_ROWS="
    f"{x['player_history_available'].eq(0).sum()}"
)

print(
    "TEAM_COLD_START_ROWS="
    f"{x['team_history_available'].eq(0).sum()}"
)

print(
    "DVP_COLD_START_ROWS="
    f"{x['dvp_history_available'].eq(0).sum()}"
)

print(
    "DUP_GAME_PLAYER="
    f"{x.duplicated(['game_id','player_id']).sum()}"
)

print("PRODUCTION_DATABASE_WRITE=NO")
print("PRODUCTION_ARTIFACT_WRITE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
