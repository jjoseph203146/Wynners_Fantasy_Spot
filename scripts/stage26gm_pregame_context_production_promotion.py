#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

SHADOW = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1_shadow.csv"
)

SHADOW_AUDIT = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1_shadow_audit.json"
)

LIVE_CORE = (
    ROOT
    / "processed"
    / "forecast_live_core_v1.csv"
)

FORECAST_PREDICTIONS = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1.csv"
)

AUDIT = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1_audit.json"
)

EXPECTED_SHADOW_CONTRACT = (
    "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_SHADOW_V1"
)

PRODUCTION_CONTRACT = (
    "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_V1"
)

PRODUCTION_COLUMNS = [
    "game_id",
    "season",
    "week",
    "team",
    "opponent_team",
    "pregame_context_pass_expectation",
]


def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    tmp.replace(path)


def main() -> None:

    section(
        "STAGE26G-M — PREGAME CONTEXT PRODUCTION PROMOTION"
    )

    print("SOURCE=VALIDATED_STAGE26G_L_SHADOW")
    print("MODEL_RETRAIN=FALSE")
    print("MODEL_MUTATION=FALSE")
    print("COACH_ADJUSTMENT_INCLUDED=FALSE")
    print("TEAM_ENVIRONMENT_MUTATION=FALSE")
    print("LIVE_CORE_MUTATION=FALSE")
    print("FORECAST_MUTATION=FALSE")
    print("SOLVER_MUTATION=FALSE")
    print("DATABASE_WRITE=FALSE")
    print("APP_MUTATION=FALSE")
    print("SERVICE_RESTART=FALSE")

    # ================================================================
    # 1. Authorities
    # ================================================================

    section("1. SOURCE AUTHORITIES")

    for path in [
        SHADOW,
        SHADOW_AUDIT,
        LIVE_CORE,
    ]:
        if not path.exists():
            fail(
                f"Required authority missing: {path}"
            )

    shadow_audit = json.loads(
        SHADOW_AUDIT.read_text()
    )

    if shadow_audit.get("status") != "PASS":
        fail(
            "Stage26G-L audit status is not PASS."
        )

    if (
        shadow_audit.get("contract")
        != EXPECTED_SHADOW_CONTRACT
    ):
        fail(
            "Unexpected Stage26G-L shadow contract: "
            f"{shadow_audit.get('contract')}"
        )

    model_info = shadow_audit.get(
        "model",
        {}
    )

    if (
        model_info.get("context_feature_count")
        != 68
    ):
        fail(
            "Stage26G-L feature count is not 68."
        )

    if (
        model_info.get("coach_adjustment_included")
        is not False
    ):
        fail(
            "Stage26G-L unexpectedly includes coach adjustment."
        )

    print(
        f"SHADOW_CONTRACT={EXPECTED_SHADOW_CONTRACT}"
    )
    print(
        "SHADOW_AUDIT_STATUS=PASS"
    )
    print(
        "SOURCE_CONTEXT_FEATURE_COUNT=68"
    )
    print(
        "SOURCE_COACH_ADJUSTMENT_INCLUDED=FALSE"
    )
    print(
        "SOURCE_AUTHORITIES=PASS"
    )

    # ================================================================
    # 2. Source identity
    # ================================================================

    section("2. SOURCE IDENTITY CONTRACT")

    shadow = pd.read_csv(
        SHADOW,
        low_memory=False,
    )

    live = pd.read_csv(
        LIVE_CORE,
        low_memory=False,
    )

    required_shadow = set(
        PRODUCTION_COLUMNS
        + [
            "coach_adjustment_applied",
            "production_influence",
        ]
    )

    missing = sorted(
        required_shadow
        - set(shadow.columns)
    )

    if missing:
        fail(
            f"Shadow missing columns: {missing}"
        )

    for frame in [
        shadow,
        live,
    ]:
        frame["game_id"] = (
            frame["game_id"].astype(str)
        )

        frame["team"] = (
            frame["team"].astype(str)
        )

    shadow_dup = int(
        shadow.duplicated(
            ["game_id", "team"],
            keep=False,
        ).sum()
    )

    live_dup = int(
        live.duplicated(
            ["game_id", "team"],
            keep=False,
        ).sum()
    )

    if shadow_dup:
        fail(
            f"Shadow duplicate identity rows={shadow_dup}"
        )

    if live_dup:
        fail(
            f"Live CORE duplicate identity rows={live_dup}"
        )

    shadow_keys = set(
        zip(
            shadow["game_id"],
            shadow["team"],
        )
    )

    live_keys = set(
        zip(
            live["game_id"],
            live["team"],
        )
    )

    missing_from_shadow = (
        live_keys
        - shadow_keys
    )

    extra_in_shadow = (
        shadow_keys
        - live_keys
    )

    if missing_from_shadow:
        fail(
            "Live CORE identities missing from shadow: "
            f"{len(missing_from_shadow)}"
        )

    if extra_in_shadow:
        fail(
            "Shadow contains identities absent from live CORE: "
            f"{len(extra_in_shadow)}"
        )

    if len(shadow) != len(live):
        fail(
            f"Row cardinality mismatch: "
            f"shadow={len(shadow)} live={len(live)}"
        )

    game_counts = shadow.groupby(
        "game_id"
    )["team"].size()

    bad_game_counts = int(
        (game_counts != 2).sum()
    )

    if bad_game_counts:
        fail(
            "Games without exactly two team rows="
            f"{bad_game_counts}"
        )

    print(
        f"SHADOW_ROWS={len(shadow)}"
    )
    print(
        f"LIVE_CORE_ROWS={len(live)}"
    )
    print(
        f"GAMES={shadow['game_id'].nunique()}"
    )
    print(
        f"SHADOW_DUPLICATE_GAME_TEAM_ROWS={shadow_dup}"
    )
    print(
        f"LIVE_DUPLICATE_GAME_TEAM_ROWS={live_dup}"
    )
    print(
        "MISSING_LIVE_IDENTITIES_IN_SHADOW=0"
    )
    print(
        "EXTRA_SHADOW_IDENTITIES=0"
    )
    print(
        "SOURCE_IDENTITY_CONTRACT=PASS"
    )

    # ================================================================
    # 3. Production metric
    # ================================================================

    section("3. PRODUCTION METRIC CONTRACT")

    expectation = pd.to_numeric(
        shadow[
            "pregame_context_pass_expectation"
        ],
        errors="coerce",
    )

    nonfinite = int(
        (
            ~np.isfinite(
                expectation.to_numpy(
                    dtype=float
                )
            )
        ).sum()
    )

    range_violations = int(
        (
            (expectation < 0)
            | (expectation > 1)
        ).sum()
    )

    coach_applied = int(
        shadow[
            "coach_adjustment_applied"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    prior_influence = int(
        shadow[
            "production_influence"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if nonfinite:
        fail(
            f"Nonfinite expectations={nonfinite}"
        )

    if range_violations:
        fail(
            f"Expectation range violations={range_violations}"
        )

    if coach_applied:
        fail(
            "Coach adjustment was applied in shadow."
        )

    if prior_influence:
        fail(
            "Shadow source already reports production influence."
        )

    print(
        f"EXPECTATION_NONFINITE_ROWS={nonfinite}"
    )
    print(
        f"EXPECTATION_RANGE_VIOLATIONS={range_violations}"
    )
    print(
        "PREDICTION_CLIPPING_APPLIED=FALSE"
    )
    print(
        "COACH_ADJUSTMENT_APPLIED=FALSE"
    )
    print(
        "PRODUCTION_METRIC_CONTRACT=PASS"
    )

    # ================================================================
    # 4. Build production artifact
    # ================================================================

    section("4. BUILD PRODUCTION ARTIFACT")

    production = shadow[
        PRODUCTION_COLUMNS
    ].copy()

    production[
        "season"
    ] = pd.to_numeric(
        production["season"],
        errors="raise",
    ).astype(int)

    production[
        "week"
    ] = pd.to_numeric(
        production["week"],
        errors="raise",
    ).astype(int)

    production[
        "pregame_context_pass_expectation"
    ] = expectation.astype(float)

    if list(production.columns) != PRODUCTION_COLUMNS:
        fail(
            "Production column order violation."
        )

    if production.duplicated(
        ["game_id", "team"]
    ).any():
        fail(
            "Production artifact duplicate identities."
        )

    print(
        f"PRODUCTION_ROWS={len(production)}"
    )
    print(
        f"PRODUCTION_GAMES={production['game_id'].nunique()}"
    )
    print(
        "PRODUCTION_IDENTITY=game_id+team"
    )
    print(
        "PRODUCTION_METRIC="
        "pregame_context_pass_expectation"
    )
    print(
        "PRODUCTION_COLUMN_CONTRACT=PASS"
    )

    # ================================================================
    # 5. Mutation firewall
    # ================================================================

    section("5. MUTATION FIREWALL")

    live_sha_before = sha256_file(
        LIVE_CORE
    )

    forecast_sha_before = (
        sha256_file(
            FORECAST_PREDICTIONS
        )
        if FORECAST_PREDICTIONS.exists()
        else None
    )

    atomic_write_csv(
        production,
        OUTPUT,
    )

    live_sha_after = sha256_file(
        LIVE_CORE
    )

    forecast_sha_after = (
        sha256_file(
            FORECAST_PREDICTIONS
        )
        if FORECAST_PREDICTIONS.exists()
        else None
    )

    if live_sha_before != live_sha_after:
        fail(
            "Live CORE mutated during promotion."
        )

    if forecast_sha_before != forecast_sha_after:
        fail(
            "Forecast predictions mutated during promotion."
        )

    print(
        "TEAM_ENVIRONMENT_MUTATED=FALSE"
    )
    print(
        "LIVE_CORE_MUTATED=FALSE"
    )
    print(
        "FORECAST_PREDICTIONS_MUTATED=FALSE"
    )
    print(
        "SOLVER_MUTATION=FALSE"
    )
    print(
        "DATABASE_WRITE=FALSE"
    )
    print(
        "APP_MUTATION=FALSE"
    )
    print(
        "SERVICE_RESTART=FALSE"
    )
    print(
        "MUTATION_FIREWALL=PASS"
    )

    # ================================================================
    # 6. Audit
    # ================================================================

    section("6. PRODUCTION AUDIT")

    output_sha = sha256_file(
        OUTPUT
    )

    audit = {
        "status":
            "PASS",

        "contract":
            PRODUCTION_CONTRACT,

        "source_contract":
            EXPECTED_SHADOW_CONTRACT,

        "metric":
            "pregame_context_pass_expectation",

        "display_name":
            "Expected Pass Rate",

        "identity":
            [
                "game_id",
                "team",
            ],

        "rows":
            len(production),

        "games":
            int(
                production[
                    "game_id"
                ].nunique()
            ),

        "model": {
            "source_model":
                "STAGE26G_J_VALIDATED_68_FEATURE_CONTEXT_MODEL",

            "model_class":
                model_info.get(
                    "model_class"
                ),

            "selected_alpha":
                model_info.get(
                    "selected_alpha"
                ),

            "context_feature_count":
                model_info.get(
                    "context_feature_count"
                ),

            "coach_adjustment_included":
                False,

            "prediction_clipping_applied":
                False,
        },

        "lineage": {
            "shadow_sha256":
                sha256_file(
                    SHADOW
                ),

            "shadow_audit_sha256":
                sha256_file(
                    SHADOW_AUDIT
                ),

            "live_core_sha256":
                live_sha_after,

            "production_output_sha256":
                output_sha,
        },

        "firewall": {
            "team_environment_mutated":
                False,

            "live_core_mutated":
                False,

            "forecast_predictions_mutated":
                False,

            "solver_mutation":
                False,

            "database_write":
                False,

            "app_mutation":
                False,

            "service_restart":
                False,
        },
    }

    atomic_write_json(
        audit,
        AUDIT,
    )

    print(
        f"PRODUCTION_OUTPUT={OUTPUT}"
    )
    print(
        f"PRODUCTION_AUDIT={AUDIT}"
    )
    print(
        f"PRODUCTION_OUTPUT_SHA256={output_sha}"
    )
    print(
        "PRODUCTION_AUDIT=PASS"
    )

    # ================================================================
    # 7. Final contract
    # ================================================================

    section("7. STAGE26G-M FINAL CONTRACT")

    print(
        "STAGE26G_M_CONTRACT="
        "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_V1"
    )
    print(
        "SOURCE_STAGE=STAGE26G_L"
    )
    print(
        "SOURCE_MODEL="
        "STAGE26G_J_VALIDATED_68_FEATURE_CONTEXT_MODEL"
    )
    print(
        "PRODUCTION_METRIC="
        "pregame_context_pass_expectation"
    )
    print(
        "PUBLIC_DISPLAY_NAME=Expected Pass Rate"
    )
    print(
        "IDENTITY=game_id+team"
    )
    print(
        "COACH_ADJUSTMENT_INCLUDED=FALSE"
    )
    print(
        "PREDICTION_CLIPPING_APPLIED=FALSE"
    )
    print(
        "TEAM_ENVIRONMENT_MUTATION=FALSE"
    )
    print(
        "FORECAST_MUTATION=FALSE"
    )
    print(
        "SOLVER_MUTATION=FALSE"
    )
    print(
        "STAGE26G_M_PREGAME_CONTEXT_PRODUCTION_PROMOTION=PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 96)
        print(f"FAIL | {exc}")
        print("=" * 96)
        raise
