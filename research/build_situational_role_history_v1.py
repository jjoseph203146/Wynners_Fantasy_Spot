from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

ROOT = Path("/home/mwynn/nfl_data_engine")
SPINE_PATH = ROOT / "data/research/matchup_intelligence_history_v1.parquet"
OUT_PATH = ROOT / "data/research/situational_role_history_v1.parquet"

print("=== MATCHUP INTELLIGENCE STEP 6R — SITUATIONAL ROLE HISTORY ===")
print("MODE=ANALYSIS_ONLY")

spine = pd.read_parquet(SPINE_PATH).copy()

key = ["game_id", "player_id"]

if spine.duplicated(key).any():
    raise RuntimeError("DUPLICATE_SPINE_GAME_PLAYER")

needed = [
    "game_id",
    "season",
    "week",
    "player_id",
    "position",
    "team",
]

missing = [c for c in needed if c not in spine.columns]
if missing:
    raise RuntimeError(f"MISSING_SPINE_COLUMNS:{missing}")

pbp = nfl.load_pbp([2023, 2024, 2025, 2026])

if hasattr(pbp, "to_pandas"):
    pbp = pbp.to_pandas()

required = [
    "game_id",
    "season",
    "week",
    "yardline_100",
    "goal_to_go",
    "pass_attempt",
    "rush_attempt",
    "qb_scramble",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
]

missing = [c for c in required if c not in pbp.columns]
if missing:
    raise RuntimeError(f"MISSING_PBP_COLUMNS:{missing}")

for c in [
    "yardline_100",
    "goal_to_go",
    "pass_attempt",
    "rush_attempt",
    "qb_scramble",
]:
    pbp[c] = pd.to_numeric(pbp[c], errors="coerce")

pbp["is_rz"] = pbp["yardline_100"].le(20)
pbp["is_i10"] = pbp["yardline_100"].le(10)
pbp["is_g2g"] = pbp["goal_to_go"].fillna(0).eq(1)

def event_table(player_col, masks):
    parts = []

    for feature, mask in masks.items():
        z = pbp.loc[
            mask & pbp[player_col].notna(),
            ["game_id", player_col],
        ].copy()

        z = (
            z.groupby(["game_id", player_col])
            .size()
            .rename(feature)
            .reset_index()
            .rename(columns={player_col: "player_id"})
        )

        parts.append(z)

    if not parts:
        return pd.DataFrame(columns=key)

    out = parts[0]

    for z in parts[1:]:
        out = out.merge(
            z,
            on=key,
            how="outer",
            validate="one_to_one",
        )

    return out.fillna(0)

pass_attempt = pbp["pass_attempt"].fillna(0).eq(1)
rush_attempt = pbp["rush_attempt"].fillna(0).eq(1)
scramble = pbp["qb_scramble"].fillna(0).eq(1)

receiver = event_table(
    "receiver_player_id",
    {
        "rz_targets": pbp["is_rz"] & pass_attempt,
        "i10_targets": pbp["is_i10"] & pass_attempt,
        "g2g_targets": pbp["is_g2g"] & pass_attempt,
    },
)

rusher = event_table(
    "rusher_player_id",
    {
        "rz_carries": pbp["is_rz"] & rush_attempt,
        "i10_carries": pbp["is_i10"] & rush_attempt,
        "g2g_carries": pbp["is_g2g"] & rush_attempt,
    },
)

passer = event_table(
    "passer_player_id",
    {
        "rz_pass_attempts": pbp["is_rz"] & pass_attempt,
        "i10_pass_attempts": pbp["is_i10"] & pass_attempt,
        "g2g_pass_attempts": pbp["is_g2g"] & pass_attempt,
    },
)

scrambler = event_table(
    "rusher_player_id",
    {
        "qb_scrambles": scramble,
    },
)

passer = passer.merge(
    scrambler,
    on=key,
    how="outer",
    validate="one_to_one",
)

events = receiver.merge(
    rusher,
    on=key,
    how="outer",
    validate="one_to_one",
)

events = events.merge(
    passer,
    on=key,
    how="outer",
    validate="one_to_one",
)

event_cols = [
    "rz_targets",
    "i10_targets",
    "g2g_targets",
    "rz_carries",
    "i10_carries",
    "g2g_carries",
    "rz_pass_attempts",
    "i10_pass_attempts",
    "g2g_pass_attempts",
    "qb_scrambles",
]

for c in event_cols:
    if c not in events.columns:
        events[c] = 0

    events[c] = pd.to_numeric(
        events[c],
        errors="coerce",
    ).fillna(0)

out = spine[needed].merge(
    events[key + event_cols],
    on=key,
    how="left",
    validate="one_to_one",
)

# No event row means zero observed event of that type,
# not an identity failure.
out[event_cols] = out[event_cols].fillna(0)

out = out.sort_values(
    ["player_id", "season", "week", "game_id"]
).reset_index(drop=True)

out["prior_player_games"] = (
    out.groupby("player_id").cumcount()
)

# ---------------------------------------------------------
# Critical PIT contract:
# shift BEFORE rolling.
# ---------------------------------------------------------

for c in event_cols:

    out[f"{c}_3g"] = (
        out.groupby("player_id")[c]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(3, min_periods=1)
                .mean()
        )
        .fillna(0.0)
    )

rolling_cols = [
    f"{c}_3g"
    for c in event_cols
]

# First career observation must contain no current-game
# situational information.
first = out["prior_player_games"].eq(0)

for c in rolling_cols:
    if not out.loc[first, c].eq(0).all():
        raise RuntimeError(
            f"FIRST_OBSERVATION_LEAKAGE:{c}"
        )

# Current raw event counts are retained only for audit.
# They are NOT approved predictors.
if out.duplicated(key).any():
    raise RuntimeError("DUPLICATE_OUTPUT_GAME_PLAYER")

if out[rolling_cols].isna().any().any():
    raise RuntimeError("NULL_ROLLING_FEATURE")

if not np.isfinite(
    out[rolling_cols].to_numpy(dtype=float)
).all():
    raise RuntimeError("NONFINITE_ROLLING_FEATURE")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)

print(f"ROWS={len(out)}")
print(f"GAMES={out['game_id'].nunique()}")
print(f"PLAYERS={out['player_id'].nunique()}")
print(f"COLUMNS={len(out.columns)}")

print()
print("=== CURRENT RAW EVENT TOTALS — AUDIT ONLY ===")

for c in event_cols:
    print(
        f"{c.upper()}={int(out[c].sum())}"
    )

print()
print("=== PIT FEATURE COVERAGE ===")

for c in rolling_cols:
    print(
        f"{c.upper()} "
        f"NONZERO={(out[c] > 0).sum()} "
        f"ZERO={(out[c] == 0).sum()}"
    )

print()
print(
    "FIRST_OBSERVATION_ROWS="
    f"{first.sum()}"
)

print(
    "FIRST_OBSERVATION_ROLLING_ZERO=PASS"
)

print(
    "DUP_GAME_PLAYER="
    f"{out.duplicated(key).sum()}"
)

print()
print("RAW_CURRENT_EVENTS_APPROVED_AS_PREDICTORS=NO")
print("ROLLING_SHIFT_BEFORE_WINDOW=YES")
print("PRODUCTION_INFLUENCE=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("STATUS=PASS")
