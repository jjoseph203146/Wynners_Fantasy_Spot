#!/usr/bin/env python3

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ALIGNMENT_PATH = ROOT / "processed/formula_matchup_alignment_v1.parquet"
DB_PATH = ROOT / "data/nfl.db"

OUT_DIR = ROOT / "processed/validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUT = OUT_DIR / "formula_component_weather_context_v1.parquet"
SUMMARY_OUT = OUT_DIR / "formula_component_weather_context_summary_v1.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_corr(x, y, method="spearman"):
    d = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(d) < 3:
        return np.nan

    if d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan

    return d["x"].corr(d["y"], method=method)


def weather_class(row):
    """
    Descriptive research classification only.
    These thresholds are NOT fitted or optimized.
    """

    if not bool(row["weather_shadow_eligible"]):
        return "NON_EXPOSED_OR_INELIGIBLE"

    wind = row["kickoff_wind_mph"]
    gust = row["kickoff_gust_mph"]
    precip = row["kickoff_precipitation_probability_pct"]
    condition = str(row["kickoff_condition"] or "").upper()

    adverse_condition_words = (
        "RAIN",
        "SNOW",
        "THUNDER",
        "STORM",
        "SLEET",
        "FREEZ",
    )

    condition_adverse = any(
        x in condition for x in adverse_condition_words
    )

    adverse = (
        (pd.notna(wind) and wind >= 10)
        or (pd.notna(gust) and gust >= 20)
        or (pd.notna(precip) and precip >= 50)
        or condition_adverse
    )

    if adverse:
        return "EXPOSED_ADVERSE"

    return "EXPOSED_NON_ADVERSE"


def checkpoint_rank(checkpoint):
    return {
        "T_MINUS_3H": 1,
        "T_MINUS_24H": 2,
        "T_MINUS_72H": 3,
    }.get(str(checkpoint), 99)


# ---------------------------------------------------------------------
# Load alignment
# ---------------------------------------------------------------------

alignment = pd.read_parquet(ALIGNMENT_PATH)

if alignment.duplicated(["game_id", "team"]).any():
    raise RuntimeError(
        "Alignment source contains duplicate game_id + team rows."
    )

alignment = alignment[
    (alignment["season"] == 2026)
    & (alignment["formula_history_games"] > 0)
].copy()


# ---------------------------------------------------------------------
# Load prospective weather provenance
# ---------------------------------------------------------------------

con = sqlite3.connect(
    f"file:{DB_PATH}?mode=ro",
    uri=True,
)

runs = pd.read_sql_query(
    """
    SELECT
        run_id,
        season,
        week,
        checkpoint,
        checkpoint_hours,
        scheduled_trigger_at_utc,
        target_kickoff_at_utc,
        attempt_started_at_utc,
        capture_completed_at_utc,
        status,
        production_influence_allowed
    FROM weather_prospective_capture_runs
    """,
    con,
)

targets = pd.read_sql_query(
    """
    SELECT
        run_id,
        game_id,
        eligible_for_checkpoint_evaluation
    FROM weather_prospective_capture_targets
    """,
    con,
)

snapshots = pd.read_sql_query(
    """
    SELECT
        snapshot_id,
        game_id,
        season,
        week,
        captured_at
    FROM weather_game_snapshots
    """,
    con,
)

features = pd.read_sql_query(
    """
    SELECT
        feature_id,
        game_id,
        season,
        week,
        source_snapshot_id,
        source_captured_at,
        stadium_name,
        stadium_type,
        kickoff_condition,
        kickoff_temperature_f,
        kickoff_feels_like_f,
        kickoff_wind_mph,
        kickoff_gust_mph,
        kickoff_precipitation_probability_pct,
        wind_max_mph,
        gust_max_mph,
        precipitation_probability_max_pct
    FROM weather_game_features_v2
    """,
    con,
)

shadow = pd.read_sql_query(
    """
    SELECT
        source_v2_feature_id,
        weather_exposure_state,
        wind_research_band,
        shadow_eligible,
        shadow_reason,
        production_influence_allowed
    FROM weather_game_shadow_v3
    """,
    con,
)

con.close()


# ---------------------------------------------------------------------
# Successful prospective runs only
# ---------------------------------------------------------------------

runs = runs[
    (runs["status"] == "SUCCESS")
    & runs["capture_completed_at_utc"].notna()
].copy()

if (runs["production_influence_allowed"].fillna(0) != 0).any():
    raise RuntimeError(
        "Weather prospective source unexpectedly permits production influence."
    )

prospective = targets.merge(
    runs,
    on="run_id",
    how="inner",
    validate="many_to_one",
)

prospective = prospective[
    prospective["eligible_for_checkpoint_evaluation"] == 1
].copy()


# ---------------------------------------------------------------------
# Exact snapshot -> feature -> shadow linkage
# ---------------------------------------------------------------------

weather = snapshots.merge(
    features,
    left_on=["snapshot_id", "game_id"],
    right_on=["source_snapshot_id", "game_id"],
    how="inner",
    suffixes=("_snapshot", "_feature"),
    validate="one_to_one",
)

weather = weather.merge(
    shadow,
    left_on="feature_id",
    right_on="source_v2_feature_id",
    how="left",
    validate="one_to_one",
)

if (
    weather["production_influence_allowed"]
    .fillna(0)
    .ne(0)
    .any()
):
    raise RuntimeError(
        "Weather shadow unexpectedly permits production influence."
    )


# ---------------------------------------------------------------------
# Reconstruct exact prospective run -> snapshot relationship
# ---------------------------------------------------------------------

prospective["attempt_started_at_utc"] = pd.to_datetime(
    prospective["attempt_started_at_utc"],
    utc=True,
)

prospective["capture_completed_at_utc"] = pd.to_datetime(
    prospective["capture_completed_at_utc"],
    utc=True,
)

weather["captured_at"] = pd.to_datetime(
    weather["captured_at"],
    utc=True,
)

candidates = prospective.merge(
    weather,
    on="game_id",
    how="inner",
    suffixes=("_run", "_weather"),
)

candidates = candidates[
    (candidates["captured_at"] >= candidates["attempt_started_at_utc"])
    & (
        candidates["captured_at"]
        <= candidates["capture_completed_at_utc"]
    )
].copy()


# ---------------------------------------------------------------------
# Validate provenance mapping
# ---------------------------------------------------------------------

run_game_counts = (
    candidates
    .groupby(["run_id", "game_id"])
    .size()
    .rename("n")
    .reset_index()
)

ambiguous = run_game_counts[run_game_counts["n"] != 1]

if len(ambiguous):
    print("=== AMBIGUOUS RUN/GAME WEATHER LINKS ===")
    print(ambiguous.to_string(index=False))
    raise RuntimeError(
        "Prospective weather provenance is not one-to-one."
    )


# ---------------------------------------------------------------------
# Choose best checkpoint per game
# T-3H > T-24H > T-72H
# ---------------------------------------------------------------------

candidates["checkpoint_rank"] = (
    candidates["checkpoint"].map(checkpoint_rank)
)

candidates = candidates.sort_values(
    [
        "game_id",
        "checkpoint_rank",
        "captured_at",
    ]
)

selected = (
    candidates
    .groupby("game_id", as_index=False)
    .head(1)
    .copy()
)

if selected["game_id"].duplicated().any():
    raise RuntimeError(
        "Selected weather contains duplicate game_id."
    )

selected = selected.rename(
    columns={
        "checkpoint": "weather_checkpoint",
        "checkpoint_hours": "weather_checkpoint_hours",
        "captured_at": "weather_captured_at",
        "shadow_eligible": "weather_shadow_eligible",
        "shadow_reason": "weather_shadow_reason",
    }
)

selected["weather_asof_verified"] = True
selected["injury_asof_verified"] = False

selected["weather_context"] = selected.apply(
    weather_class,
    axis=1,
)


# ---------------------------------------------------------------------
# Join weather onto team-game alignment rows
# ---------------------------------------------------------------------

keep_weather = [
    "game_id",
    "weather_checkpoint",
    "weather_checkpoint_hours",
    "weather_captured_at",
    "weather_asof_verified",
    "injury_asof_verified",
    "stadium_name",
    "stadium_type",
    "kickoff_condition",
    "kickoff_temperature_f",
    "kickoff_feels_like_f",
    "kickoff_wind_mph",
    "kickoff_gust_mph",
    "kickoff_precipitation_probability_pct",
    "wind_max_mph",
    "gust_max_mph",
    "precipitation_probability_max_pct",
    "weather_exposure_state",
    "wind_research_band",
    "weather_shadow_eligible",
    "weather_shadow_reason",
    "weather_context",
]

df = alignment.merge(
    selected[keep_weather],
    on="game_id",
    how="left",
    validate="many_to_one",
)

df["weather_asof_verified"] = (
    df["weather_asof_verified"]
    .fillna(False)
    .astype(bool)
)

df["injury_asof_verified"] = False

df["weather_context"] = df["weather_context"].fillna(
    "NO_VERIFIED_WEATHER"
)


# ---------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------

df["weather_controlled"] = df["weather_asof_verified"]
df["injury_controlled"] = False

df["causal_claim_allowed"] = False
df["weights_fitted"] = False
df["thresholds_optimized"] = False
df["production_authorized"] = False
df["solver_authorized"] = False
df["forecast_mutation"] = False
df["player_projection_mutation"] = False


# ---------------------------------------------------------------------
# Research tests
# ---------------------------------------------------------------------

TESTS = [
    (
        "RUN",
        "MATCHUP",
        "opponent_rush_yards_allowed_vs_league",
        "current_rushing_yards_delta_vs_baseline",
        "YARDS_DELTA",
    ),
    (
        "RUN",
        "FORMULA",
        "prior_3_positive_rush_epa_game_rate",
        "current_actual_rushing_epa",
        "EPA",
    ),
    (
        "PASS",
        "MATCHUP",
        "opponent_pass_yards_allowed_vs_league",
        "current_passing_yards_delta_vs_baseline",
        "YARDS_DELTA",
    ),
    (
        "PASS",
        "FORMULA",
        "prior_3_positive_pass_epa_game_rate",
        "current_actual_passing_epa",
        "EPA",
    ),
]

summary_rows = []

verified = df[df["weather_asof_verified"]].copy()

for phase, family, component, target, target_kind in TESTS:

    for context in [
        "ALL_VERIFIED_WEATHER",
        "EXPOSED_NON_ADVERSE",
        "EXPOSED_ADVERSE",
        "NON_EXPOSED_OR_INELIGIBLE",
    ]:

        if context == "ALL_VERIFIED_WEATHER":
            d = verified.copy()
        else:
            d = verified[
                verified["weather_context"] == context
            ].copy()

        d = d[[component, target]].dropna()

        summary_rows.append(
            {
                "phase": phase,
                "family": family,
                "component": component,
                "target": target,
                "target_kind": target_kind,
                "weather_context": context,
                "team_games": len(d),
                "games_approx": len(d) / 2.0,
                "pearson": safe_corr(
                    d[component],
                    d[target],
                    "pearson",
                ),
                "spearman": safe_corr(
                    d[component],
                    d[target],
                    "spearman",
                ),
                "component_mean": (
                    d[component].mean()
                    if len(d) else np.nan
                ),
                "target_mean": (
                    d[target].mean()
                    if len(d) else np.nan
                ),
                "weather_controlled": True,
                "injury_controlled": False,
                "causal_claim_allowed": False,
                "weights_fitted": False,
                "thresholds_optimized": False,
                "production_authorized": False,
            }
        )

summary = pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------

df.to_parquet(
    DETAIL_OUT,
    index=False,
)

summary.to_parquet(
    SUMMARY_OUT,
    index=False,
)


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

print("=== CONTEXTUAL COMPONENT VALIDATION V1 ===")

print("\n=== SOURCE ===")
print("2026 alignment team-games:", len(alignment))
print(
    "verified weather team-games:",
    int(df["weather_asof_verified"].sum()),
)
print(
    "verified weather games:",
    df.loc[
        df["weather_asof_verified"],
        "game_id",
    ].nunique(),
)

print("\n=== SELECTED CHECKPOINTS ===")
print(
    selected["weather_checkpoint"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n=== WEATHER CONTEXT ===")
print(
    df.loc[
        df["weather_asof_verified"],
        "weather_context",
    ]
    .value_counts(dropna=False)
    .to_string()
)

print("\n=== VERIFIED WEATHER GAME LIST ===")

game_cols = [
    "game_id",
    "weather_checkpoint",
    "weather_captured_at",
    "stadium_type",
    "kickoff_condition",
    "kickoff_wind_mph",
    "kickoff_gust_mph",
    "kickoff_precipitation_probability_pct",
    "weather_context",
]

print(
    selected[game_cols]
    .sort_values("game_id")
    .to_string(index=False)
)

print("\n=== COMPONENT RESULTS BY WEATHER CONTEXT ===")

print(
    summary[
        [
            "phase",
            "family",
            "component",
            "target_kind",
            "weather_context",
            "team_games",
            "pearson",
            "spearman",
            "target_mean",
        ]
    ].to_string(index=False)
)

print("\n=== SAFETY / CONTEXT CONTRACT ===")
print("weather_asof_verified=PER_GAME")
print("injury_asof_verified=False")
print("injury_controlled=False")
print("causal_claim_allowed=False")
print("weights_fitted=False")
print("thresholds_optimized=False")
print("production_authorized=False")
print("solver_authorized=False")
print("forecast_mutation=False")
print("player_projection_mutation=False")

print("\n=== WRITE COMPLETE ===")
print(DETAIL_OUT)
print(SUMMARY_OUT)
