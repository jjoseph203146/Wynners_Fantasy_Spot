from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path("/home/mwynn/nfl_data_engine")

MODEL_DIR = ROOT / "data/research/models/matchup_intelligence_v1"
MANIFEST_PATH = MODEL_DIR / "manifest.json"

MATRIX_PATH = ROOT / "data/parquet/nfl_current_offensive_model_matrix.parquet"
MI_PATH = ROOT / "data/research/matchup_intelligence_current_shadow_v1.parquet"
BASE_PATH = ROOT / "data/parquet/nfl_current_offensive_stat_forecasts.parquet"

OUT_PATH = ROOT / "data/research/matchup_intelligence_candidate_forecast_shadow_v1.parquet"

print("=== STEP 7L-K — WEEK 3 MI CANDIDATE SHADOW INFERENCE ===")
print("MODE=ANALYSIS_ONLY")

manifest = json.loads(MANIFEST_PATH.read_text())

matrix = pd.read_parquet(MATRIX_PATH).copy()
mi = pd.read_parquet(MI_PATH).copy()
base = pd.read_parquet(BASE_PATH).copy()

for df in [matrix, mi, base]:
    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)

for name, df in [
    ("MATRIX", matrix),
    ("MI", mi),
    ("BASE", base),
]:
    if df.duplicated(["game_id", "player_id"]).any():
        raise RuntimeError(f"{name}_DUPLICATE_GAME_PLAYER")

if len(matrix) != len(base):
    raise RuntimeError("MATRIX_BASE_ROWCOUNT_MISMATCH")

mi_cols = [
    "game_id",
    "player_id",
    "opportunities_allowed_avg_3",
    "rb_g2g_share",
    "rb_g2g_share_available",
    "te_i10_share",
    "te_i10_share_available",
]

work = matrix.merge(
    mi[mi_cols],
    on=["game_id", "player_id"],
    how="left",
    validate="one_to_one",
)

base_stats = [
    "expected_carries",
    "expected_rushing_yards",
    "expected_targets",
    "expected_receptions",
    "expected_receiving_yards",
]

missing_base = [
    c for c in base_stats
    if c not in base.columns
]

if missing_base:
    raise RuntimeError(
        f"BASE_COMPONENTS_MISSING:{missing_base}"
    )

out = base.copy()

# Preserve canonical values explicitly.
for c in base_stats:
    out[f"e14_{c}"] = pd.to_numeric(
        out[c],
        errors="coerce",
    )

out["mi_candidate_applied"] = False
out["mi_candidate_model"] = ""

model_count = 0
prediction_count = 0

for entry in manifest["models"]:

    position = entry["position"]
    target = entry["target"]
    features = entry["features"]

    model_path = MODEL_DIR / entry["artifact"]

    actual_sha = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    if actual_sha != entry["sha256"]:
        raise RuntimeError(
            f"MODEL_HASH_MISMATCH:{entry['artifact']}"
        )

    model = joblib.load(model_path)

    q = work[
        work["position"].eq(position)
    ].copy()

    if position == "RB":
        available = (
            q["rb_g2g_share_available"]
            .fillna(False)
            .astype(bool)
        )
        q.loc[
            ~available,
            "rb_g2g_share",
        ] = 0.0
        q["rb_g2g_share_available"] = (
            available.astype(float)
        )

    if position == "TE":
        available = (
            q["te_i10_share_available"]
            .fillna(False)
            .astype(bool)
        )
        q.loc[
            ~available,
            "te_i10_share",
        ] = 0.0
        q["te_i10_share_available"] = (
            available.astype(float)
        )

    x = q[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if x.isna().any().any():
        raise RuntimeError(
            f"{position}_{target}_NULL_INPUT"
        )

    arr = x.to_numpy(dtype=float)

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{position}_{target}_NONFINITE_INPUT"
        )

    pred = model.predict(arr)

    if not np.isfinite(pred).all():
        raise RuntimeError(
            f"{position}_{target}_NONFINITE_PREDICTION"
        )

    # Production learned-model contract floors component
    # predictions at zero.
    pred = np.maximum(pred, 0.0)

    expected_col = f"expected_{target}"

    if expected_col not in out.columns:
        raise RuntimeError(
            f"OUTPUT_COMPONENT_MISSING:{expected_col}"
        )

    pred_frame = q[
        ["game_id", "player_id"]
    ].copy()

    pred_frame["_mi_prediction"] = pred

    out = out.merge(
        pred_frame,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    mask = out["_mi_prediction"].notna()

    out.loc[
        mask,
        expected_col,
    ] = out.loc[
        mask,
        "_mi_prediction",
    ]

    label = f"{position}:{target}"

    existing = out.loc[
        mask,
        "mi_candidate_model",
    ].astype(str)

    out.loc[
        mask,
        "mi_candidate_model",
    ] = np.where(
        existing.eq(""),
        label,
        existing + "|" + label,
    )

    out.loc[
        mask,
        "mi_candidate_applied",
    ] = True

    prediction_count += int(mask.sum())
    model_count += 1

    out = out.drop(
        columns=["_mi_prediction"]
    )

    print(
        f"INFERRED={label}:ROWS={int(mask.sum())}"
    )

if model_count != 7:
    raise RuntimeError(
        f"EXPECTED_7_MODELS_GOT:{model_count}"
    )

if len(out) != len(base):
    raise RuntimeError("OUTPUT_ROWCOUNT_CHANGED")

if out.duplicated(["game_id", "player_id"]).any():
    raise RuntimeError("OUTPUT_DUPLICATE_GAME_PLAYER")

# Reuse the frozen production position/model-group applicability contract.
# Nonapplicable component NaNs are structural, not missing forecasts.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mi_e14_component_contract", ROOT / "current_offensive_stat_forecast.py"
)
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
if not out["model_group"].isin(["QB", "RB_FB", "WR", "TE"]).all():
    raise RuntimeError("UNKNOWN_MODEL_GROUP")
_contract.validate_required_stats(out)
# Inference must not fill, erase, or relocate structural nulls.
stat_columns = _contract.H2_COLUMNS[9:]
base_by_key = base.set_index(["game_id", "player_id"])
out_by_key = out.set_index(["game_id", "player_id"])
if not out_by_key[stat_columns].isna().equals(
    base_by_key.loc[out_by_key.index, stat_columns].isna()
):
    raise RuntimeError("STRUCTURAL_NULL_REPRESENTATION_CHANGED")
bad = 0
print("VALIDATION_CONTRACT=PRODUCTION_POSITION_MODEL_GROUP_COMPONENTS")
print("ILLEGAL_APPLICABLE_NULL_COUNT=0")

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_parquet(
    OUT_PATH,
    index=False,
)

sha = hashlib.sha256(
    OUT_PATH.read_bytes()
).hexdigest()

print(f"OUTPUT={OUT_PATH}")
print(f"OUTPUT_ROWS={len(out)}")
print(
    "PLAYERS_WITH_MI_CANDIDATE="
    f"{int(out['mi_candidate_applied'].sum())}"
)
print(f"COMPONENT_PREDICTIONS={prediction_count}")
print(f"VALIDATION_BAD={bad}")
print(f"OUTPUT_SHA256={sha}")

gate = (
    model_count == 7
    and len(out) == len(base)
    and bad == 0
)

print(
    "STEP_7L_K_STATUS="
    + ("PASS" if gate else "FAIL_CLOSED")
)

print("OUTPUT_SCOPE=DATA_RESEARCH_ONLY")
print("CANONICAL_FORECAST_MODIFIED=NO")
print("E14_MODEL_MUTATION=NO")
print("V3_RECONCILIATION_APPLIED=NO")
print("SOLVER_INFLUENCE=NO")
print("PROJECTION_INFLUENCE=NO")
print("UPDATER_RUN=NO")
