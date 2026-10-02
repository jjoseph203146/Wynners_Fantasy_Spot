#!/usr/bin/env python3

"""
Current offensive stat forecast adapter.

Purpose
-------
Run the frozen Stage23E14 offensive stat models against a current E14
79-feature model matrix and reproduce the Stage24J R14G-B inference contract.

This module:
- DOES NOT retrain models.
- DOES NOT modify E14 artifacts.
- DOES NOT modify current feature builders.
- DOES NOT calculate FanDuel fantasy points.
- DOES NOT publish Stage24.
- DOES NOT modify solver inputs.
- DOES NOT modify SQLite.
- DOES NOT modify updater orchestration.

Production input:
    data/parquet/nfl_current_offensive_model_matrix.parquet

Production output:
    data/parquet/nfl_current_offensive_stat_forecasts.parquet

Regression mode:
    --regression-week1

Regression mode uses the frozen Week-1 R12 E14 matrix and compares the
result against the frozen R14G-B offensive stat forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent

E14_DIR = (
    APP_DIR
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage23e14_20260912T154412Z"
)

MODEL_DIR = E14_DIR / "models"
FEATURE_CONTRACT_PATH = E14_DIR / "feature_contract.json"
MODEL_MANIFEST_PATH = E14_DIR / "manifest.json"
SPARSE_STATE_PATH = E14_DIR / "baseline_state" / "sparse_prior3_state.parquet"

CURRENT_MATRIX_PATH = (
    APP_DIR / "data" / "parquet" / "nfl_current_offensive_model_matrix.parquet"
)

HISTORY_PATH = (
    APP_DIR / "data" / "parquet" / "nfl_player_game_stats.parquet"
)

OUTPUT_PATH = (
    APP_DIR / "data" / "parquet" / "nfl_current_offensive_stat_forecasts.parquet"
)

FROZEN_R12_MATRIX_PATH = (
    APP_DIR
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage24j_c_r7_r12_20260912T185444Z"
    / "offensive_model_matrix_79_current_479.parquet"
)

FROZEN_R14G_DIR = (
    APP_DIR
    / "data"
    / "model_candidates"
    / "stat_forecast"
    / "stage24j_c_r7_r14g_b_20260912T194129Z"
)

FROZEN_R14G_FORECAST_PATH = (
    FROZEN_R14G_DIR / "current_offensive_stat_forecasts_479.parquet"
)

FROZEN_R14G_CONTRACT_PATH = FROZEN_R14G_DIR / "candidate_contract.json"


IDENTITY_COLUMNS = [
    "game_id",
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
]

H2_COLUMNS = [
    "game_id",
    "season",
    "week",
    "team",
    "opponent_team",
    "player_id",
    "player_name",
    "position",
    "model_group",
    "expected_attempts",
    "expected_carries",
    "expected_completions",
    "expected_passing_yards",
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_receptions",
    "expected_targets",
    "expected_interceptions",
    "expected_passing_tds",
    "expected_rushing_tds",
    "expected_receiving_tds",
]


EXPECTED_INFERENCE_CONTRACT = {
    "learned": "MODEL_PREDICT_ALL_MATCHING_ROWS_THEN_NONNEGATIVE_FLOOR",
    "position_prior_scope": "SEASON_LT_FORECAST_SEASON",
    "sparse_td_0": "POSITION_TARGET_HISTORICAL_MEAN",
    "sparse_td_1_to_2": "AVAILABLE_TARGET_HISTORY_MEAN",
    "sparse_td_ge3": "MOST_RECENT_3_TARGET_HISTORY_MEAN",
}


LEARNED_OUTPUT_MAP = {
    "attempts": "expected_attempts",
    "carries": "expected_carries",
    "completions": "expected_completions",
    "passing_yards": "expected_passing_yards",
    "rushing_yards": "expected_rushing_yards",
    "receiving_yards": "expected_receiving_yards",
    "receptions": "expected_receptions",
    "targets": "expected_targets",
    "interceptions": "expected_interceptions",
    "passing_tds": "expected_passing_tds",
}


SPARSE_TD_TARGETS = {
    "rushing_tds": "expected_rushing_tds",
    "receiving_tds": "expected_receiving_tds",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"MISSING_{label}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"INVALID_JSON_OBJECT: {path}")
    return obj


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonical_position(value: Any) -> str:
    return normalize_text(value).upper()


def model_group_for_position(position: Any) -> str | None:
    pos = canonical_position(position)

    if pos == "QB":
        return "QB"
    if pos in {"RB", "FB"}:
        return "RB_FB"
    if pos == "WR":
        return "WR"
    if pos == "TE":
        return "TE"

    return None


def load_feature_contract() -> tuple[dict[str, Any], list[str]]:
    contract = load_json(FEATURE_CONTRACT_PATH)

    features = contract.get("feature_columns")
    if features is None:
        features = contract.get("features")

    if not isinstance(features, list):
        raise RuntimeError(
            "E14_FEATURE_CONTRACT_MISSING_ORDERED_FEATURE_LIST"
        )

    features = [str(x) for x in features]

    if len(features) != 79:
        raise RuntimeError(
            f"E14_FEATURE_COUNT_INVALID:{len(features)}"
        )

    if len(set(features)) != len(features):
        raise RuntimeError("E14_FEATURE_COLUMNS_NOT_UNIQUE")

    football_only = contract.get("football_only")
    if football_only is not True:
        raise RuntimeError(
            f"E14_FOOTBALL_ONLY_CONTRACT_INVALID:{football_only}"
        )

    fantasy_derived = contract.get("fantasy_derived_features")
    if fantasy_derived is not False:
        raise RuntimeError(
            "E14_FANTASY_DERIVED_FEATURE_CONTRACT_INVALID:"
            f"{fantasy_derived}"
        )

    return contract, features


def validate_frozen_r14g_contract() -> dict[str, Any]:
    contract = load_json(FROZEN_R14G_CONTRACT_PATH)

    if contract.get("contract") != (
        "WFS_CURRENT_OFFENSIVE_STAT_FORECAST_CANDIDATE_V1"
    ):
        raise RuntimeError(
            "R14G_CONTRACT_ID_MISMATCH:"
            f"{contract.get('contract')}"
        )

    h2 = contract.get("h2_columns")
    if h2 != H2_COLUMNS:
        raise RuntimeError("R14G_H2_SCHEMA_CONTRACT_MISMATCH")

    inference = contract.get("inference_contract")
    if inference != EXPECTED_INFERENCE_CONTRACT:
        raise RuntimeError(
            "R14G_INFERENCE_CONTRACT_MISMATCH:"
            f"{inference}"
        )

    output = contract.get("output", {})
    if int(output.get("columns", -1)) != len(H2_COLUMNS):
        raise RuntimeError("R14G_OUTPUT_COLUMN_COUNT_MISMATCH")

    return contract


def load_model_manifest() -> list[dict[str, Any]]:
    manifest = load_json(MODEL_MANIFEST_PATH)

    models = manifest.get("models")
    if not isinstance(models, list):
        raise RuntimeError("E14_MANIFEST_MODELS_MISSING")

    if len(models) != 18:
        raise RuntimeError(
            f"E14_MODEL_COUNT_INVALID:{len(models)}"
        )

    seen: set[tuple[str, str]] = set()

    for entry in models:
        if not isinstance(entry, dict):
            raise RuntimeError("E14_MODEL_ENTRY_INVALID")

        group = normalize_text(entry.get("group"))
        target = normalize_text(entry.get("target"))
        artifact = normalize_text(entry.get("artifact"))
        expected_sha = normalize_text(entry.get("sha256"))

        if not group or not target or not artifact or not expected_sha:
            raise RuntimeError(
                f"E14_MODEL_MANIFEST_ENTRY_INCOMPLETE:{entry}"
            )

        key = (group, target)
        if key in seen:
            raise RuntimeError(
                f"E14_DUPLICATE_MODEL_GROUP_TARGET:{group}:{target}"
            )
        seen.add(key)

        path = E14_DIR / artifact
        require_file(path, "E14_MODEL")

        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "E14_MODEL_SHA_MISMATCH:"
                f"{path.name}:expected={expected_sha}:actual={actual_sha}"
            )

    return models


def load_matrix(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    require_file(path, "MODEL_MATRIX")

    df = pd.read_parquet(path).copy()

    expected_columns = IDENTITY_COLUMNS + feature_columns

    if list(df.columns) != expected_columns:
        raise RuntimeError(
            "MODEL_MATRIX_SCHEMA_ORDER_MISMATCH:"
            f"expected={len(expected_columns)}:"
            f"actual={len(df.columns)}"
        )

    if df.empty:
        raise RuntimeError("MODEL_MATRIX_EMPTY")

    for col in ["game_id", "player_id", "team", "opponent_team", "position"]:
        blank = df[col].isna() | df[col].astype(str).str.strip().eq("")
        if blank.any():
            raise RuntimeError(
                f"MODEL_MATRIX_BLANK_IDENTITY:{col}:{int(blank.sum())}"
            )

    duplicate_count = int(
        df.duplicated(subset=["game_id", "player_id"], keep=False).sum()
    )
    if duplicate_count:
        raise RuntimeError(
            f"MODEL_MATRIX_DUPLICATE_IDENTITY:{duplicate_count}"
        )

    numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    null_count = int(numeric.isna().sum().sum())
    if null_count:
        raise RuntimeError(
            f"MODEL_MATRIX_NULL_FEATURE_CELLS:{null_count}"
        )

    values = numeric.to_numpy(dtype=float)

    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise RuntimeError(
            f"MODEL_MATRIX_NONFINITE_FEATURE_CELLS:{nonfinite}"
        )

    df.loc[:, feature_columns] = numeric

    return df


def initialize_output(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix[IDENTITY_COLUMNS].copy()

    # season/week are E14 feature columns, but they are also required
    # metadata fields in the frozen H2 output contract.
    out["season"] = pd.to_numeric(matrix["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(matrix["week"], errors="raise").astype(int)

    out["model_group"] = out["position"].map(model_group_for_position)

    missing_group = out["model_group"].isna()
    if missing_group.any():
        examples = (
            out.loc[missing_group, ["player_id", "player_name", "position"]]
            .head(10)
            .to_dict("records")
        )
        raise RuntimeError(
            "UNSUPPORTED_CURRENT_POSITION:"
            f"{int(missing_group.sum())}:{examples}"
        )

    for col in H2_COLUMNS[9:]:
        out[col] = np.nan

    return out[H2_COLUMNS]


def apply_learned_models(
    matrix: pd.DataFrame,
    out: pd.DataFrame,
    feature_columns: list[str],
    manifest_models: list[dict[str, Any]],
) -> None:
    groups = out["model_group"]

    for entry in manifest_models:
        group = normalize_text(entry["group"])
        target = normalize_text(entry["target"])

        if target not in LEARNED_OUTPUT_MAP:
            raise RuntimeError(
                f"UNEXPECTED_LEARNED_TARGET:{group}:{target}"
            )

        output_col = LEARNED_OUTPUT_MAP[target]

        mask = groups.eq(group)
        row_count = int(mask.sum())

        if row_count == 0:
            continue

        model_path = E14_DIR / normalize_text(entry["artifact"])
        model = joblib.load(model_path)

        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(feature_columns):
            raise RuntimeError(
                "MODEL_FEATURE_COUNT_MISMATCH:"
                f"{model_path.name}:{n_features}"
            )

        x = (
            matrix.loc[mask, feature_columns]
            .to_numpy(dtype=float, copy=True)
        )

        pred = np.asarray(model.predict(x), dtype=float).reshape(-1)

        if len(pred) != row_count:
            raise RuntimeError(
                "MODEL_PREDICTION_LENGTH_MISMATCH:"
                f"{model_path.name}:{len(pred)}:{row_count}"
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"MODEL_NONFINITE_PREDICTION:{model_path.name}"
            )

        # Frozen R14G learned inference contract:
        # MODEL_PREDICT_ALL_MATCHING_ROWS_THEN_NONNEGATIVE_FLOOR
        pred = np.maximum(pred, 0.0)

        out.loc[mask, output_col] = pred


def prepare_history() -> pd.DataFrame:
    require_file(HISTORY_PATH, "HISTORICAL_PLAYER_GAME_STATS")

    history = pd.read_parquet(HISTORY_PATH).copy()

    required = [
        "season",
        "week",
        "game_id",
        "player_id",
        "position",
        "rushing_tds",
        "receiving_tds",
    ]

    missing = [c for c in required if c not in history.columns]
    if missing:
        raise RuntimeError(
            f"HISTORY_REQUIRED_COLUMNS_MISSING:{missing}"
        )

    history["player_id"] = history["player_id"].astype("string").str.strip()
    history["position"] = history["position"].astype("string").str.upper().str.strip()

    history["model_group"] = history["position"].map(model_group_for_position)

    history["season"] = pd.to_numeric(history["season"], errors="coerce")
    history["week"] = pd.to_numeric(history["week"], errors="coerce")

    for target in SPARSE_TD_TARGETS:
        history[target] = pd.to_numeric(history[target], errors="coerce")

    history = history[
        history["player_id"].notna()
        & history["player_id"].ne("")
        & history["model_group"].notna()
        & history["season"].notna()
        & history["week"].notna()
    ].copy()

    # Deterministic chronological ordering.
    history["_game_id_sort"] = history["game_id"].astype("string").fillna("")

    history = history.sort_values(
        ["player_id", "season", "week", "_game_id_sort"],
        kind="mergesort",
    ).reset_index(drop=True)

    return history


def sparse_td_prediction(
    *,
    player_id: str,
    model_group: str,
    target: str,
    forecast_season: int,
    history: pd.DataFrame,
) -> tuple[float, int, str]:
    # R14G position prior scope:
    # SEASON_LT_FORECAST_SEASON
    prior_scope = history[
        (history["season"] < forecast_season)
        & history["model_group"].eq(model_group)
    ]

    prior_values = pd.to_numeric(
        prior_scope[target],
        errors="coerce",
    ).dropna()

    if prior_values.empty:
        raise RuntimeError(
            "SPARSE_POSITION_PRIOR_EMPTY:"
            f"{model_group}:{target}:season<{forecast_season}"
        )

    position_prior = float(prior_values.mean())

    # Sparse history is model-group specific. A player's history from
    # another position/model group must not establish eligibility for the
    # current model group. This reproduces the frozen E14/R14G contract.
    player_history = history[
        history["player_id"].eq(player_id)
        & history["model_group"].eq(model_group)
        & (history["season"] < forecast_season)
    ]

    target_values = pd.to_numeric(
        player_history[target],
        errors="coerce",
    ).dropna()

    history_count = int(len(target_values))

    if history_count == 0:
        prediction = position_prior
        source = "POSITION_TARGET_PRIOR"

    elif history_count <= 2:
        prediction = float(target_values.mean())
        source = "AVAILABLE_TARGET_HISTORY"

    else:
        prediction = float(target_values.iloc[-3:].mean())
        source = "STRICT_PRIOR3_RECOMPUTED"

    if not math.isfinite(prediction):
        raise RuntimeError(
            "SPARSE_TD_NONFINITE:"
            f"{player_id}:{model_group}:{target}"
        )

    prediction = max(prediction, 0.0)

    return prediction, history_count, source


def apply_sparse_td_routes(
    out: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        player_id = normalize_text(row["player_id"])
        player_name = normalize_text(row["player_name"])
        group = normalize_text(row["model_group"])
        season = int(row["season"])

        if group == "QB":
            applicable = ["rushing_tds"]
        elif group == "RB_FB":
            applicable = ["rushing_tds", "receiving_tds"]
        elif group in {"WR", "TE"}:
            applicable = ["receiving_tds"]
        else:
            raise RuntimeError(
                f"SPARSE_UNSUPPORTED_MODEL_GROUP:{group}"
            )

        for target in applicable:
            prediction, history_count, source = sparse_td_prediction(
                player_id=player_id,
                model_group=group,
                target=target,
                forecast_season=season,
                history=history,
            )

            output_col = SPARSE_TD_TARGETS[target]
            out.at[idx, output_col] = prediction

            audit_rows.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "model_group": group,
                    "target": target,
                    "target_history_games": history_count,
                    "source": source,
                    "prediction": prediction,
                }
            )

    return pd.DataFrame(audit_rows)


def validate_required_stats(out: pd.DataFrame) -> None:
    required_by_group = {
        "QB": [
            "expected_attempts",
            "expected_carries",
            "expected_completions",
            "expected_passing_yards",
            "expected_rushing_yards",
            "expected_interceptions",
            "expected_passing_tds",
            "expected_rushing_tds",
        ],
        "RB_FB": [
            "expected_carries",
            "expected_rushing_yards",
            "expected_receiving_yards",
            "expected_receptions",
            "expected_targets",
            "expected_rushing_tds",
            "expected_receiving_tds",
        ],
        "WR": [
            "expected_receiving_yards",
            "expected_receptions",
            "expected_targets",
            "expected_receiving_tds",
        ],
        "TE": [
            "expected_receiving_yards",
            "expected_receptions",
            "expected_targets",
            "expected_receiving_tds",
        ],
    }

    issue_cells = 0

    for group, cols in required_by_group.items():
        mask = out["model_group"].eq(group)

        if not mask.any():
            continue

        numeric = out.loc[mask, cols].apply(pd.to_numeric, errors="coerce")

        issue_cells += int(numeric.isna().sum().sum())

        arr = numeric.to_numpy(dtype=float)
        issue_cells += int((~np.isfinite(arr)).sum())

        issue_cells += int((arr < 0).sum())

    if issue_cells:
        raise RuntimeError(
            f"REQUIRED_STAT_ISSUE_CELLS:{issue_cells}"
        )


def build_forecast(
    matrix: pd.DataFrame,
    feature_columns: list[str],
    manifest_models: list[dict[str, Any]],
    history: pd.DataFrame,
    apply_matchup_intelligence: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = initialize_output(matrix)

    apply_learned_models(
        matrix=matrix,
        out=out,
        feature_columns=feature_columns,
        manifest_models=manifest_models,
    )

    sparse_audit = apply_sparse_td_routes(out, history)

    # E14 is complete here; downstream V3 remains the reconciled authority.
    if apply_matchup_intelligence:
        from matchup_intelligence_v1 import apply_components
        mi_audit = apply_components(matrix, out, feature_columns)
        print("MATCHUP_INTELLIGENCE_V1=" + json.dumps(mi_audit, sort_keys=True))

    validate_required_stats(out)

    if list(out.columns) != H2_COLUMNS:
        raise RuntimeError("OUTPUT_H2_SCHEMA_ORDER_MISMATCH")

    duplicates = int(
        out.duplicated(subset=["game_id", "player_id"], keep=False).sum()
    )
    if duplicates:
        raise RuntimeError(
            f"OUTPUT_DUPLICATE_IDENTITY:{duplicates}"
        )

    if len(out) != len(matrix):
        raise RuntimeError(
            f"OUTPUT_ROW_COUNT_MISMATCH:{len(out)}:{len(matrix)}"
        )

    return out, sparse_audit


def compare_regression(
    actual: pd.DataFrame,
    frozen: pd.DataFrame,
    sparse_audit: pd.DataFrame,
) -> None:
    if list(frozen.columns) != H2_COLUMNS:
        raise RuntimeError("FROZEN_R14G_SCHEMA_MISMATCH")

    key = ["game_id", "player_id"]

    if actual.duplicated(key).any():
        raise RuntimeError("REGRESSION_ACTUAL_DUPLICATE_IDENTITY")

    if frozen.duplicated(key).any():
        raise RuntimeError("REGRESSION_FROZEN_DUPLICATE_IDENTITY")

    actual_keys = set(map(tuple, actual[key].astype(str).to_numpy()))
    frozen_keys = set(map(tuple, frozen[key].astype(str).to_numpy()))

    missing = frozen_keys - actual_keys
    extra = actual_keys - frozen_keys

    print("REGRESSION_ACTUAL_ROWS =", len(actual))
    print("REGRESSION_FROZEN_ROWS =", len(frozen))
    print("REGRESSION_MISSING_IDENTITIES =", len(missing))
    print("REGRESSION_EXTRA_IDENTITIES =", len(extra))

    if missing or extra:
        raise RuntimeError(
            "FROZEN_R14G_IDENTITY_MISMATCH:"
            f"missing={len(missing)}:extra={len(extra)}"
        )

    merged = actual.merge(
        frozen,
        on=key,
        how="inner",
        suffixes=("_actual", "_frozen"),
        validate="one_to_one",
    )

    metadata = [
        "season",
        "week",
        "team",
        "opponent_team",
        "player_name",
        "position",
        "model_group",
    ]

    metadata_mismatch = 0

    for col in metadata:
        a = merged[f"{col}_actual"].astype("string").fillna("")
        b = merged[f"{col}_frozen"].astype("string").fillna("")
        metadata_mismatch += int((a != b).sum())

    print("REGRESSION_METADATA_MISMATCH =", metadata_mismatch)

    if metadata_mismatch:
        raise RuntimeError(
            f"FROZEN_R14G_METADATA_MISMATCH:{metadata_mismatch}"
        )

    stat_columns = H2_COLUMNS[9:]

    max_abs_delta = 0.0
    value_mismatch = 0
    null_mismatch = 0

    for col in stat_columns:
        a = pd.to_numeric(
            merged[f"{col}_actual"],
            errors="coerce",
        )
        b = pd.to_numeric(
            merged[f"{col}_frozen"],
            errors="coerce",
        )

        null_diff = a.isna() ^ b.isna()
        null_mismatch += int(null_diff.sum())

        both = a.notna() & b.notna()

        if both.any():
            delta = (a[both] - b[both]).abs()

            if len(delta):
                max_abs_delta = max(
                    max_abs_delta,
                    float(delta.max()),
                )

            value_mismatch += int((delta > 1e-8).sum())

    print("REGRESSION_NULL_MISMATCH =", null_mismatch)
    print("REGRESSION_MAX_ABS_DELTA =", max_abs_delta)
    print("REGRESSION_VALUE_MISMATCH_GT_1E8 =", value_mismatch)

    source_counts = (
        sparse_audit["source"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("SPARSE_ROUTING_CELLS =", len(sparse_audit))
    print("SPARSE_ROUTING_SOURCE_COUNTS =", source_counts)

    expected_source_counts = {
        "AVAILABLE_TARGET_HISTORY": 15,
        "POSITION_TARGET_PRIOR": 107,
        "STRICT_PRIOR3_RECOMPUTED": 458,
    }

    if len(sparse_audit) != 580:
        raise RuntimeError(
            f"FROZEN_SPARSE_CELL_COUNT_MISMATCH:{len(sparse_audit)}"
        )

    if source_counts != expected_source_counts:
        raise RuntimeError(
            "FROZEN_SPARSE_ROUTING_COUNT_MISMATCH:"
            f"{source_counts}"
        )

    if null_mismatch:
        raise RuntimeError(
            f"FROZEN_R14G_NULL_MISMATCH:{null_mismatch}"
        )

    if value_mismatch:
        raise RuntimeError(
            "FROZEN_R14G_VALUE_MISMATCH:"
            f"{value_mismatch}:max_delta={max_abs_delta}"
        )

    print("FROZEN_R14G_REGRESSION=PASS")


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        df.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_production_output(
    out: pd.DataFrame,
    sparse_audit: pd.DataFrame,
) -> None:
    if out.empty:
        raise RuntimeError("PRODUCTION_FORECAST_EMPTY")

    if list(out.columns) != H2_COLUMNS:
        raise RuntimeError("PRODUCTION_H2_SCHEMA_MISMATCH")

    seasons = sorted(
        pd.to_numeric(out["season"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    weeks = sorted(
        pd.to_numeric(out["week"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )

    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(
            f"PRODUCTION_TARGET_NOT_SINGLE_WEEK:{seasons}:{weeks}"
        )

    duplicates = int(
        out.duplicated(["game_id", "player_id"], keep=False).sum()
    )
    if duplicates:
        raise RuntimeError(
            f"PRODUCTION_DUPLICATE_IDENTITIES:{duplicates}"
        )

    required_identity = [
        "game_id",
        "team",
        "opponent_team",
        "player_id",
        "position",
        "model_group",
    ]

    for col in required_identity:
        blank = out[col].isna() | out[col].astype(str).str.strip().eq("")
        if blank.any():
            raise RuntimeError(
                f"PRODUCTION_BLANK_IDENTITY:{col}:{int(blank.sum())}"
            )

    validate_required_stats(out)

    print("PRODUCTION_ROWS =", len(out))
    print("PRODUCTION_GAMES =", out["game_id"].nunique())
    print("PRODUCTION_TEAMS =", out["team"].nunique())
    print("PRODUCTION_SEASON =", seasons)
    print("PRODUCTION_WEEK =", weeks)
    print(
        "PRODUCTION_MODEL_GROUPS =",
        out["model_group"].value_counts().sort_index().to_dict(),
    )
    print("PRODUCTION_SPARSE_CELLS =", len(sparse_audit))
    print(
        "PRODUCTION_SPARSE_SOURCES =",
        sparse_audit["source"].value_counts().sort_index().to_dict(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regression-week1",
        action="store_true",
        help=(
            "Run frozen Week-1 R12 matrix through frozen E14 models "
            "and compare against frozen R14G-B output. "
            "Does not publish production output."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("===== CURRENT OFFENSIVE STAT FORECAST =====")

    validate_frozen_r14g_contract()

    _, feature_columns = load_feature_contract()

    print("E14_FEATURE_COUNT =", len(feature_columns))
    print("E14_FEATURE_CONTRACT=PASS")

    manifest_models = load_model_manifest()

    print("E14_MODEL_COUNT =", len(manifest_models))
    print("E14_MODEL_SHA_VALIDATION=PASS")

    history = prepare_history()

    print("HISTORICAL_ROWS =", len(history))
    print("HISTORICAL_PLAYER_IDS =", history["player_id"].nunique())

    if args.regression_week1:
        print()
        print("MODE = FROZEN_WEEK1_REGRESSION")

        matrix = load_matrix(
            FROZEN_R12_MATRIX_PATH,
            feature_columns,
        )

        frozen = pd.read_parquet(
            FROZEN_R14G_FORECAST_PATH
        ).copy()

        print("INPUT_MATRIX_ROWS =", len(matrix))
        print("FROZEN_FORECAST_ROWS =", len(frozen))

        actual, sparse_audit = build_forecast(
            matrix=matrix,
            feature_columns=feature_columns,
            manifest_models=manifest_models,
            history=history,
        )

        compare_regression(
            actual=actual,
            frozen=frozen,
            sparse_audit=sparse_audit,
        )

        print("CURRENT_OFFENSIVE_STAT_FORECAST_REGRESSION=PASS")
        return 0

    print()
    print("MODE = CURRENT_PRODUCTION")

    matrix = load_matrix(
        CURRENT_MATRIX_PATH,
        feature_columns,
    )

    print("INPUT_MATRIX_ROWS =", len(matrix))

    out, sparse_audit = build_forecast(
        matrix=matrix,
        feature_columns=feature_columns,
        manifest_models=manifest_models,
        history=history,
        apply_matchup_intelligence=True,
    )

    validate_production_output(out, sparse_audit)

    atomic_write_parquet(out, OUTPUT_PATH)

    print("PUBLISHED =", OUTPUT_PATH)
    print("OUTPUT_SHA256 =", sha256_file(OUTPUT_PATH))
    print("CURRENT_OFFENSIVE_STAT_FORECAST=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
