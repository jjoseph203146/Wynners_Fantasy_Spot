import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Direct script execution starts sys.path in this shadow directory.
# Disable bytecode writes before importing either frozen dependency.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from simulator_primitive_generator_v1.intelligence import (
    PrimitiveIntelligenceError,
    VOLUME_VERSION,
    YARDAGE_VERSION,
    estimate_team_volume,
    estimate_team_yardage,
)
from simulator_primitive_generator_v1.scoring import (
    SCORING_VERSION,
    estimate_team_scoring,
)

OUT=ROOT/"future_primitive_distribution_shadow_v1"

TRAINING=ROOT/"processed/forecast_v1_team_game_training.csv"
HISTORY=ROOT/"processed/offensive_reconciliation_team_history_v1.csv"
CALIBRATION=OUT/"CALIBRATION_V1.json"

BANK=OUT/"RESIDUAL_BANK_V1.csv"
MANIFEST=OUT/"RESIDUAL_BANK_V1_MANIFEST.json"

FEATURES=[
    "team_offensive_plays_avg_3",
    "team_offensive_plays_avg_5",
    "opp_offensive_plays_avg_3",
    "opp_offensive_plays_avg_5",
    "team_pass_rate_avg_3",
    "team_pass_rate_avg_5",
    "opp_pass_rate_avg_3",
    "opp_pass_rate_avg_5",
    "team_passing_yards_avg_3",
    "team_pass_attempts_avg_3",
    "team_rushing_yards_avg_3",
    "team_rush_attempts_avg_3",
    "team_opponent_pass_yards_allowed_avg_3",
    "team_opponent_rush_yards_allowed_avg_3",
    "opp_pass_attempts_avg_3",
    "opp_rush_attempts_avg_3",
    "team_passing_tds_avg_3",
    "team_rushing_tds_avg_3",
    "team_opponent_pass_tds_allowed_avg_3",
    "team_opponent_rush_tds_allowed_avg_3",
    "team_points_for_avg_3",
    "team_points_for_avg_5",
    "team_opponent_points_allowed_avg_3",
    "team_opponent_points_allowed_avg_5",
]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def replay_state(r):
    return {
        "state_manifest":{
            "context":{
                "team":str(r["team"]),
                "opponent":str(r["opponent_team"]),
            }
        },
        "feature_reconstruction":[
            {
                "name":name,
                "value":float(r[name]),
            }
            for name in FEATURES
        ],
    }

with open(CALIBRATION) as f:
    calibration=json.load(f)

if calibration["authorization"]["monte_carlo"]:
    raise RuntimeError("CALIBRATION_UNEXPECTEDLY_AUTHORIZES_MONTE_CARLO")

if calibration["population"]["eligible_rows"] != 1678:
    raise RuntimeError("CALIBRATION_ELIGIBLE_ROW_CONTRACT_CHANGED")

EXPECTED_REJECTION = "PrimitiveIntelligenceError:INVALID_EFFICIENCY_DENOMINATOR:OFFENSE_PASS"
for key, expected in {"training_rows": 1710, "eligible_rows": 1678,
                      "rejected_rows": 32, "seasons": [2023, 2024, 2025],
                      "rejection_reason": "ZERO_HISTORY_COLD_START",
                      "historical_temporal_provenance": "UNVERIFIED"}.items():
    if calibration["population"][key] != expected:
        raise RuntimeError(f"CALIBRATION_CONTRACT_CHANGED:{key}")
if calibration["center"] != {"package": "simulator_primitive_generator_v1", "status": "VALIDATED_FROZEN"}:
    raise RuntimeError("CALIBRATION_CENTER_CHANGED")
if any(calibration["authorization"].values()):
    raise RuntimeError("CALIBRATION_AUTHORIZATION_CHANGED")
source_hashes = {str(p.relative_to(ROOT)): sha256(p)
                 for p in (TRAINING, HISTORY, CALIBRATION)}

t=pd.read_csv(TRAINING)
h=pd.read_csv(HISTORY)

if len(t) != 1710:
    raise RuntimeError(f"TRAINING_ROW_COUNT_FAILURE:{len(t)}")
identity = ["season", "week", "game_id", "team", "opponent_team", "team_history_games"]
if t[identity].isna().any().any() or t.duplicated(["game_id", "team"]).any():
    raise RuntimeError("INVALID_TRAINING_IDENTITY")
for col in ("season", "week", "team_history_games"):
    if not t[col].map(lambda v: math.isfinite(v) and v == int(v) and v >= 0).all():
        raise RuntimeError(f"INVALID_INTEGER_IDENTITY:{col}")
if sorted(t.season.unique().tolist()) != calibration["population"]["seasons"]:
    raise RuntimeError("TRAINING_SEASON_MISMATCH")

accepted=[]
rejections=[]

for _,r in t.iterrows():
    try:
        state=replay_state(r)

        volume=estimate_team_volume(state)
        yardage=estimate_team_yardage(state,volume)
        scoring=estimate_team_scoring(
            state,
            volume,
            yardage,
        )

        accepted.append({
            "season":int(r["season"]),
            "week":int(r["week"]),
            "game_id":str(r["game_id"]),
            "team":str(r["team"]),
            "opponent_team":str(r["opponent_team"]),
            "team_history_games":int(r["team_history_games"]),

            "expected_plays":
                int(volume["offensive_plays"]),
            "expected_pass_rate":
                float(volume["pass_rate"]),
            "expected_pass_ypa":
                float(yardage["passing_yards_per_attempt"]),
            "expected_rush_ypc":
                float(yardage["rushing_yards_per_carry"]),
            "expected_passing_tds":
                int(scoring["passing_tds"]),
            "expected_rushing_tds":
                int(scoring["rushing_tds"]),
            "expected_points":
                int(scoring["points"]),
        })

    except PrimitiveIntelligenceError as exc:
        reason = f"{type(exc).__name__}:{exc}"
        if reason != EXPECTED_REJECTION or r["team_history_games"] != 0:
            raise RuntimeError(f"UNEXPECTED_GENERATOR_REJECTION:{r['game_id']}:{r['team']}:{reason}") from exc
        rejections.append({
            "season":int(r["season"]),
            "week":int(r["week"]),
            "game_id":str(r["game_id"]),
            "team":str(r["team"]),
            "team_history_games":int(r["team_history_games"]),
            "reason":f"{type(exc).__name__}:{exc}",
        })

if len(accepted) != 1678 or len(rejections) != 32:
    raise RuntimeError(f"POPULATION_FAILURE:{len(accepted)}:{len(rejections)}")
a=pd.DataFrame(accepted)

actual=h[[
    "game_id",
    "team",
    "attempts",
    "carries",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
]].copy()

actual["game_id"]=actual["game_id"].astype(str)

bank=a.merge(
    actual,
    on=["game_id","team"],
    how="left",
    validate="one_to_one",
)

if len(bank) != 1678:
    raise RuntimeError(
        f"ROW_COUNT_FAILURE:{len(bank)}"
    )

actual_cols=[
    "attempts",
    "carries",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
]

if bank[actual_cols].isna().any().any():
    raise RuntimeError("MISSING_REALIZED_PRIMITIVES")

if bank.duplicated(["game_id","team"]).any():
    raise RuntimeError("DUPLICATE_GAME_TEAM")

if not bank[actual_cols].map(lambda x: math.isfinite(float(x))).all().all():
    raise RuntimeError("NONFINITE_REALIZED_PRIMITIVES")
if (bank[["attempts", "carries"]] <= 0).any().any():
    raise RuntimeError("INVALID_REALIZED_DENOMINATOR")

bank["actual_plays"] = (
    bank["attempts"] + bank["carries"]
)

bank["actual_pass_rate"] = (
    bank["attempts"] / bank["actual_plays"]
)

bank["actual_pass_ypa"] = (
    bank["passing_yards"] / bank["attempts"]
)

bank["actual_rush_ypc"] = (
    bank["rushing_yards"] / bank["carries"]
)

bank["resid_plays"] = (
    bank["actual_plays"] -
    bank["expected_plays"]
)

bank["resid_pass_rate"] = (
    bank["actual_pass_rate"] -
    bank["expected_pass_rate"]
)

bank["resid_pass_ypa"] = (
    bank["actual_pass_ypa"] -
    bank["expected_pass_ypa"]
)

bank["resid_rush_ypc"] = (
    bank["actual_rush_ypc"] -
    bank["expected_rush_ypc"]
)

bank["resid_passing_tds"] = (
    bank["passing_tds"] -
    bank["expected_passing_tds"]
)

bank["resid_rushing_tds"] = (
    bank["rushing_tds"] -
    bank["expected_rushing_tds"]
)

residual_cols=[
    "resid_plays",
    "resid_pass_rate",
    "resid_pass_ypa",
    "resid_rush_ypc",
    "resid_passing_tds",
    "resid_rushing_tds",
]

for c in residual_cols:
    if not bank[c].map(
        lambda x: math.isfinite(float(x))
    ).all():
        raise RuntimeError(
            f"NONFINITE_RESIDUAL:{c}"
        )

# Stable canonical row order.
bank=bank.sort_values(
    ["season","week","game_id","team"],
    kind="mergesort",
).reset_index(drop=True)

columns=[
    "season",
    "week",
    "game_id",
    "team",
    "opponent_team",
    "team_history_games",

    "expected_plays",
    "expected_pass_rate",
    "expected_pass_ypa",
    "expected_rush_ypc",
    "expected_passing_tds",
    "expected_rushing_tds",
    "expected_points",

    "actual_plays",
    "actual_pass_rate",
    "actual_pass_ypa",
    "actual_rush_ypc",
    "passing_tds",
    "rushing_tds",

    "resid_plays",
    "resid_pass_rate",
    "resid_pass_ypa",
    "resid_rush_ypc",
    "resid_passing_tds",
    "resid_rushing_tds",
]

# Validate against the previously measured frozen-generator residuals.
# Standard deviations use sample ddof=1; tolerances allow reported rounding only.
def require_close(label, measured, expected, tolerance):
    if not math.isfinite(measured) or abs(measured - expected) > tolerance:
        raise RuntimeError(f"VALIDATION_FAILURE:{label}:{measured}:{expected}")

bias_targets = dict(zip(
    ["plays", "pass_rate", "pass_ypa", "rush_ypc", "passing_tds", "rushing_tds"],
    [-0.1389, -0.0005, 0.0104, -0.1451, -0.0006, -0.0292],
))
residual_statistics = {}
for mechanism, bias in bias_targets.items():
    values = bank[f"resid_{mechanism}"]
    measured = {"bias": float(values.mean()), "std": float(values.std(ddof=1))}
    measured.update({f"p{int(q*100):02d}": float(values.quantile(q)) for q in (.05, .50, .95)})
    require_close(f"{mechanism}:bias", measured["bias"], bias, 0.0001)
    for key, expected in calibration["mechanism_residuals"][mechanism].items():
        require_close(f"{mechanism}:{key}", measured[key], expected, 0.0001)
    residual_statistics[mechanism] = measured

correlations = {}
for left, right, expected in [
    ("pass_rate", "pass_ypa", -0.305),
    ("pass_ypa", "passing_tds", 0.422),
    ("pass_rate", "rushing_tds", -0.286),
    ("rush_ypc", "rushing_tds", 0.245),
]:
    measured = float(bank[f"resid_{left}"].corr(bank[f"resid_{right}"]))
    require_close(f"correlation:{left}:{right}", measured, expected, 0.001)
    correlations[f"{left}/{right}"] = measured

points = bank.merge(t[["game_id", "team", "target_team_points"]],
                    on=["game_id", "team"], validate="one_to_one")
points["residual"] = points.target_team_points - points.expected_points
if not points.residual.map(math.isfinite).all():
    raise RuntimeError("NONFINITE_POINTS_DIAGNOSTIC")
paired = points.merge(points[["game_id", "team", "opponent_team", "residual"]],
                      left_on=["game_id", "opponent_team", "team"],
                      right_on=["game_id", "team", "opponent_team"],
                      validate="one_to_one", suffixes=("", "_opponent"))
if len(paired) != len(bank):
    raise RuntimeError("INCOMPLETE_OPPONENT_PAIRS")
points_diagnostic = {
    "bias": float(points.residual.mean()),
    "std": float(points.residual.std(ddof=1)),
    "p05": float(points.residual.quantile(.05)),
    "p50": float(points.residual.quantile(.50)),
    "p95": float(points.residual.quantile(.95)),
    "within_game_opponent_residual_correlation": float(paired.residual.corr(paired.residual_opponent)),
}
for key, expected in calibration["points_diagnostic"].items():
    require_close(f"points:{key}", points_diagnostic[key], expected, .001)
for path, expected in source_hashes.items():
    if sha256(ROOT / path) != expected:
        raise RuntimeError(f"SOURCE_CHANGED_DURING_BUILD:{path}")

bank[columns].to_csv(
    BANK,
    index=False,
    float_format="%.12g",
)

rejection_counts=Counter(
    x["reason"] for x in rejections
)

manifest={
    "version":
        "FUTURE_PRIMITIVE_RESIDUAL_BANK_V1",
    "status":"SHADOW_ONLY",

    "training_rows": len(t),
    "accepted_rows": len(bank),
    "row_count":len(bank),
    "rejections": rejections,
    "residual_statistics": residual_statistics,
    "residual_correlations": correlations,
    "points_diagnostic": points_diagnostic,
    "residual_stat_validation": "PASS",
    "residual_correlation_validation": "PASS",
    "validation_tolerances": {"mechanism_statistics": .0001, "correlations": .001, "points": .001},
    "standard_deviation_ddof": 1,
    "row_order": ["season", "week", "game_id", "team"],
    "historical_temporal_provenance_status": "HISTORICAL_TEMPORAL_PROVENANCE_UNVERIFIED",
    "unique_game_team_rows":
        int(
            bank[["game_id","team"]]
            .drop_duplicates()
            .shape[0]
        ),

    "rejected_rows":len(rejections),
    "rejection_reasons":
        dict(sorted(rejection_counts.items())),

    "generator_versions":{
        "volume":VOLUME_VERSION,
        "yardage":YARDAGE_VERSION,
        "scoring":SCORING_VERSION,
    },

    "residual_vector":residual_cols,

    "sources":{
        str(TRAINING.relative_to(ROOT)):
            sha256(TRAINING),
        str(HISTORY.relative_to(ROOT)):
            sha256(HISTORY),
        str(CALIBRATION.relative_to(ROOT)):
            sha256(CALIBRATION),
    },

    "artifact":{
        "path":str(BANK.relative_to(ROOT)),
        "sha256":sha256(BANK),
    },

    "historical_temporal_provenance":
        "UNVERIFIED",

    "sampling_performed":False,
    "rng_used":False,
    "monte_carlo_authorized":False,
    "production_influence":"NONE",
    "fanduel_solver_influence":"NONE",
}

with open(MANIFEST,"w") as f:
    json.dump(
        manifest,
        f,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    f.write("\n")

print("=== RESIDUAL BANK V1 ===")
print("rows =",len(bank))
print(
    "unique game/team =",
    manifest["unique_game_team_rows"]
)
print("rejected =",len(rejections))

print("\n=== REJECTIONS ===")
for reason,count in sorted(
    rejection_counts.items()
):
    print(count,reason)

print("\n=== RESIDUAL SUMMARY ===")
print(
    bank[residual_cols]
    .describe(
        percentiles=[.05,.25,.5,.75,.95]
    )
    .T
    .round(4)
    .to_string()
)

print("\n=== RESIDUAL CORRELATION ===")
print(
    bank[residual_cols]
    .corr()
    .round(3)
    .to_string()
)

print("\n=== ARTIFACT HASHES ===")
print("bank =",sha256(BANK))
print("manifest =",sha256(MANIFEST))

print("\nRESIDUAL_BANK_CREATED=YES")
print("RNG_USED=NO")
print("SAMPLING_PERFORMED=NO")
print("MONTE_CARLO_AUTHORIZED=NO")
print("PRODUCTION_INFLUENCE=NONE")
print("FANDUEL_SOLVER_INFLUENCE=NONE")
