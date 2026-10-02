#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import runpy

import numpy as np
import pandas as pd


ROOT = Path("/home/mwynn/nfl_data_engine")

GJ_SCRIPT = (
    ROOT
    / "scripts"
    / "stage26gj_pregame_context_pass_expectation_validation.py"
)

MANIFEST = (
    ROOT
    / "experiments"
    / "forecast_v1_exp004"
    / "forecast_v1_exp004_features.json"
)

HISTORICAL = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_replacement_v1.csv"
)

LIVE_CORE = (
    ROOT
    / "processed"
    / "forecast_live_core_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1_shadow.csv"
)

AUDIT = (
    ROOT
    / "processed"
    / "pregame_context_pass_expectation_v1_shadow_audit.json"
)

LEGACY_COACH_FIELDS = {
    "coach_prior_games",
    "coach_prior_points_against_avg",
    "coach_prior_points_for_avg",
    "coach_prior_win_pct",
    "opponent_coach_prior_games",
    "opponent_coach_prior_points_against_avg",
    "opponent_coach_prior_points_for_avg",
    "opponent_coach_prior_win_pct",
}

EXPECTED_CORE_COUNT = 76
EXPECTED_CONTEXT_COUNT = 68


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
        "STAGE26G-L — PREGAME CONTEXT DEPLOYMENT SHADOW"
    )

    print("ANALYSIS_ONLY=TRUE")
    print("SHADOW_ARTIFACT_WRITE=TRUE")
    print("EXISTING_LIVE_CORE_MUTATION=FALSE")
    print("EXISTING_FORECAST_MUTATION=FALSE")
    print("COACH_ADJUSTMENT_INCLUDED=FALSE")
    print("SOLVER_MUTATION=FALSE")
    print("DATABASE_WRITE=FALSE")
    print("APP_MUTATION=FALSE")
    print("SERVICE_RESTART=FALSE")

    # -----------------------------------------------------------------
    # 1. Frozen authorities
    # -----------------------------------------------------------------

    section(
        "1. FROZEN AUTHORITIES"
    )

    for path in [
        GJ_SCRIPT,
        MANIFEST,
        HISTORICAL,
        LIVE_CORE,
    ]:
        if not path.exists():
            fail(
                f"Required authority missing: {path}"
            )

    manifest = json.loads(
        MANIFEST.read_text()
    )

    core = manifest.get(
        "core_features"
    )

    if not isinstance(
        core,
        list,
    ):
        fail(
            "Manifest core_features missing."
        )

    if len(core) != EXPECTED_CORE_COUNT:
        fail(
            f"Frozen CORE count={len(core)}, "
            f"expected={EXPECTED_CORE_COUNT}"
        )

    missing_legacy = sorted(
        LEGACY_COACH_FIELDS
        - set(core)
    )

    if missing_legacy:
        fail(
            "Frozen CORE missing legacy coach fields: "
            f"{missing_legacy}"
        )

    context_features = [
        c
        for c in core
        if c not in LEGACY_COACH_FIELDS
    ]

    if len(context_features) != EXPECTED_CONTEXT_COUNT:
        fail(
            f"Context count={len(context_features)}, "
            f"expected={EXPECTED_CONTEXT_COUNT}"
        )

    print(
        f"FROZEN_CORE_FEATURES={len(core)}"
    )
    print(
        "LEGACY_COACH_FIELDS_EXCLUDED="
        f"{len(LEGACY_COACH_FIELDS)}"
    )
    print(
        "PREGAME_CONTEXT_FEATURES="
        f"{len(context_features)}"
    )
    print(
        "FROZEN_AUTHORITY_CONTRACT=PASS"
    )

    # -----------------------------------------------------------------
    # 2. Exact G-J modeling authority
    # -----------------------------------------------------------------

    section(
        "2. VALIDATED G-J MODELING AUTHORITY"
    )

    gj = runpy.run_path(
        str(GJ_SCRIPT),
        run_name="stage26gj_shadow_import",
    )

    required = [
        "DeterministicRidge",
        "choose_alpha_from_prior_only",
        "recover_stage26ga_matrix",
    ]

    missing = [
        name
        for name in required
        if name not in gj
    ]

    if missing:
        fail(
            "Validated G-J objects unavailable: "
            f"{missing}"
        )

    DeterministicRidge = gj[
        "DeterministicRidge"
    ]

    choose_alpha = gj[
        "choose_alpha_from_prior_only"
    ]

    recover_ga = gj[
        "recover_stage26ga_matrix"
    ]

    print(
        "MODEL_CLASS=DeterministicRidge"
    )
    print(
        "ALPHA_SELECTION=choose_alpha_from_prior_only"
    )
    print(
        "TARGET_AUTHORITY=STAGE26G_A"
    )
    print(
        "VALIDATED_MODELING_AUTHORITY=PASS"
    )

    # -----------------------------------------------------------------
    # 3. Historical training frame
    # -----------------------------------------------------------------

    section(
        "3. HISTORICAL DEPLOYMENT TRAINING FRAME"
    )

    hist = pd.read_csv(
        HISTORICAL,
        low_memory=False,
    )

    missing_hist = sorted(
        set(context_features)
        - set(hist.columns)
    )

    if missing_hist:
        fail(
            "Historical source missing context features: "
            f"{missing_hist}"
        )

    ga = recover_ga()

    required_ga = [
        "game_id",
        "team",
        "game_date_dt",
        "raw_n",
        "raw_pass",
    ]

    missing_ga = [
        c
        for c in required_ga
        if c not in ga.columns
    ]

    if missing_ga:
        fail(
            "G-A matrix missing required fields: "
            f"{missing_ga}"
        )

    hist[
        "game_id"
    ] = hist[
        "game_id"
    ].astype(str)

    hist[
        "team"
    ] = hist[
        "team"
    ].astype(str)

    ga = ga[
        required_ga
    ].copy()

    ga[
        "game_id"
    ] = ga[
        "game_id"
    ].astype(str)

    ga[
        "team"
    ] = ga[
        "team"
    ].astype(str)

    if hist.duplicated(
        [
            "game_id",
            "team",
        ]
    ).any():
        fail(
            "Historical source duplicate game/team keys."
        )

    if ga.duplicated(
        [
            "game_id",
            "team",
        ]
    ).any():
        fail(
            "G-A target duplicate game/team keys."
        )

    train = hist[
        [
            "game_id",
            "team",
        ]
        + context_features
    ].merge(
        ga,
        on=[
            "game_id",
            "team",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(train) != 1710:
        fail(
            f"Historical joined rows={len(train)}, expected=1710"
        )

    train[
        "game_date_dt"
    ] = pd.to_datetime(
        train[
            "game_date_dt"
        ],
        errors="coerce",
    )

    if train[
        "game_date_dt"
    ].isna().any():
        fail(
            "Historical training date contains null values."
        )

    raw_n = pd.to_numeric(
        train[
            "raw_n"
        ],
        errors="coerce",
    )

    raw_pass = pd.to_numeric(
        train[
            "raw_pass"
        ],
        errors="coerce",
    )

    if raw_n.isna().any():
        fail(
            "raw_n contains null/non-numeric values."
        )

    if raw_pass.isna().any():
        fail(
            "raw_pass contains null/non-numeric values."
        )

    if (
        raw_n <= 0
    ).any():
        fail(
            "raw_n contains nonpositive values."
        )

    train[
        "current_observed_pass_rate"
    ] = (
        raw_pass
        / raw_n
    )

    target = train[
        "current_observed_pass_rate"
    ].to_numpy(
        dtype=float
    )

    if (
        ~np.isfinite(
            target
        )
    ).any():
        fail(
            "Historical target contains nonfinite values."
        )

    if (
        (target < 0)
        | (target > 1)
    ).any():
        fail(
            "Historical target outside [0,1]."
        )

    train = (
        train
        .sort_values(
            [
                "game_date_dt",
                "game_id",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"HISTORICAL_TRAIN_ROWS={len(train)}"
    )
    print(
        "HISTORICAL_DUPLICATE_GAME_TEAM_ROWS=0"
    )
    print(
        "TARGET=current_observed_pass_rate"
    )
    print(
        "TARGET_CURRENT_GAME_XPASS_USED=FALSE"
    )
    print(
        "HISTORICAL_DEPLOYMENT_TRAINING_FRAME=PASS"
    )

    # -----------------------------------------------------------------
    # 4. Final prior-only alpha selection
    # -----------------------------------------------------------------

    section(
        "4. FINAL PRIOR-ONLY ALPHA SELECTION"
    )

    alpha = choose_alpha(
        train,
        context_features,
    )

    alpha = float(
        alpha
    )

    if not np.isfinite(
        alpha
    ):
        fail(
            "Selected alpha is nonfinite."
        )

    print(
        f"FINAL_SELECTED_ALPHA={alpha:.12g}"
    )
    print(
        "LIVE_ROWS_USED_FOR_ALPHA_SELECTION=FALSE"
    )
    print(
        "FINAL_PRIOR_ONLY_ALPHA_SELECTION=PASS"
    )

    # -----------------------------------------------------------------
    # 5. Final historical model
    # -----------------------------------------------------------------

    section(
        "5. FINAL HISTORICAL MODEL FIT"
    )

    model = DeterministicRidge(
        alpha
    )

    model.fit(
        train[
            context_features
        ],
        train[
            "current_observed_pass_rate"
        ].to_numpy(
            dtype=float
        ),
    )

    print(
        f"FINAL_TRAIN_ROWS={len(train)}"
    )
    print(
        f"FINAL_FEATURE_COUNT={len(context_features)}"
    )
    print(
        "LIVE_2026_ROWS_INCLUDED_IN_MODEL_FIT=FALSE"
    )
    print(
        "FINAL_HISTORICAL_MODEL_FIT=PASS"
    )

    # -----------------------------------------------------------------
    # 6. Live CORE contract
    # -----------------------------------------------------------------

    section(
        "6. LIVE CORE INPUT CONTRACT"
    )

    live = pd.read_csv(
        LIVE_CORE,
        low_memory=False,
    )

    required_live = (
        {
            "game_id",
            "team",
            "opponent_team",
            "season",
            "week",
        }
        | set(
            context_features
        )
    )

    missing_live = sorted(
        required_live
        - set(live.columns)
    )

    if missing_live:
        fail(
            "Live CORE missing required fields: "
            f"{missing_live}"
        )

    live[
        "game_id"
    ] = live[
        "game_id"
    ].astype(str)

    live[
        "team"
    ] = live[
        "team"
    ].astype(str)

    duplicate_live = int(
        live.duplicated(
            [
                "game_id",
                "team",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_live:
        fail(
            "Live CORE duplicate game/team keys."
        )

    game_counts = live.groupby(
        "game_id"
    )[
        "team"
    ].size()

    bad_game_counts = int(
        (
            game_counts
            != 2
        ).sum()
    )

    if bad_game_counts:
        fail(
            "Live games without exactly "
            f"two team rows={bad_game_counts}"
        )

    live_seasons = sorted(
        pd.to_numeric(
            live[
                "season"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if live_seasons != [2026]:
        fail(
            f"Unexpected live seasons={live_seasons}"
        )

    print(
        f"LIVE_TEAM_ROWS={len(live)}"
    )
    print(
        f"LIVE_GAMES={live['game_id'].nunique()}"
    )
    print(
        f"LIVE_DUPLICATE_GAME_TEAM_ROWS={duplicate_live}"
    )
    print(
        f"LIVE_GAMES_WITH_BAD_TEAM_ROW_COUNT={bad_game_counts}"
    )
    print(
        "LIVE_CORE_INPUT_CONTRACT=PASS"
    )

    # -----------------------------------------------------------------
    # 7. Shadow inference
    # -----------------------------------------------------------------

    section(
        "7. PREGAME CONTEXT SHADOW INFERENCE"
    )

    pred = np.asarray(
        model.predict(
            live[
                context_features
            ]
        ),
        dtype=float,
    )

    if len(pred) != len(live):
        fail(
            "Prediction row count differs from live CORE."
        )

    nonfinite = int(
        (
            ~np.isfinite(
                pred
            )
        ).sum()
    )

    if nonfinite:
        fail(
            f"Nonfinite live expectations={nonfinite}"
        )

    range_violations = int(
        (
            (pred < 0)
            | (pred > 1)
        ).sum()
    )

    shadow = live[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent_team",
        ]
    ].copy()

    shadow[
        "pregame_context_pass_expectation"
    ] = pred

    shadow[
        "coach_adjustment_applied"
    ] = False

    shadow[
        "production_influence"
    ] = False

    if shadow.duplicated(
        [
            "game_id",
            "team",
        ]
    ).any():
        fail(
            "Shadow output duplicate game/team keys."
        )

    print(
        f"SHADOW_ROWS={len(shadow)}"
    )
    print(
        f"SHADOW_GAMES={shadow['game_id'].nunique()}"
    )
    print(
        f"PREDICTION_NONFINITE_ROWS={nonfinite}"
    )
    print(
        f"PREDICTION_RANGE_VIOLATIONS={range_violations}"
    )
    print(
        "PREDICTION_CLIPPING_APPLIED=FALSE"
    )
    print(
        "COACH_ADJUSTMENT_APPLIED=FALSE"
    )

    if range_violations:
        fail(
            "Live context expectation outside [0,1]; "
            "clipping is not authorized."
        )

    print(
        "PREGAME_CONTEXT_SHADOW_INFERENCE=PASS"
    )

    # -----------------------------------------------------------------
    # 8. Firewall + artifact write
    # -----------------------------------------------------------------

    section(
        "8. PRODUCTION FIREWALL"
    )

    live_core_sha_before = sha256_file(
        LIVE_CORE
    )

    prediction_path = (
        ROOT
        / "processed"
        / "forecast_live_core_v1_predictions.csv"
    )

    prediction_sha_before = (
        sha256_file(
            prediction_path
        )
        if prediction_path.exists()
        else None
    )

    atomic_write_csv(
        shadow,
        OUTPUT,
    )

    live_core_sha_after = sha256_file(
        LIVE_CORE
    )

    prediction_sha_after = (
        sha256_file(
            prediction_path
        )
        if prediction_path.exists()
        else None
    )

    if (
        live_core_sha_before
        != live_core_sha_after
    ):
        fail(
            "Existing live CORE mutated."
        )

    if (
        prediction_sha_before
        != prediction_sha_after
    ):
        fail(
            "Existing live forecast predictions mutated."
        )

    output_sha = sha256_file(
        OUTPUT
    )

    audit = {
        "status":
            "PASS",

        "contract":
            "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_SHADOW_V1",

        "source_model":
            "STAGE26G_J_VALIDATED_68_FEATURE_CONTEXT_MODEL",

        "model": {
            "model_class":
                "DeterministicRidge",

            "frozen_core_feature_count":
                len(core),

            "legacy_coach_fields_excluded":
                len(
                    LEGACY_COACH_FIELDS
                ),

            "context_feature_count":
                len(
                    context_features
                ),

            "context_features":
                context_features,

            "selected_alpha":
                alpha,

            "historical_training_rows":
                len(train),

            "target":
                "current_observed_pass_rate",

            "coach_adjustment_included":
                False,
        },

        "live": {
            "team_rows":
                len(shadow),

            "games":
                int(
                    shadow[
                        "game_id"
                    ].nunique()
                ),

            "duplicate_game_team_rows":
                int(
                    shadow.duplicated(
                        [
                            "game_id",
                            "team",
                        ],
                        keep=False,
                    ).sum()
                ),

            "prediction_nonfinite_rows":
                nonfinite,

            "prediction_range_violations":
                range_violations,

            "prediction_clipping_applied":
                False,
        },

        "lineage": {
            "stage26gj_script_sha256":
                sha256_file(
                    GJ_SCRIPT
                ),

            "manifest_sha256":
                sha256_file(
                    MANIFEST
                ),

            "historical_source_sha256":
                sha256_file(
                    HISTORICAL
                ),

            "live_core_sha256":
                live_core_sha_after,

            "shadow_output_sha256":
                output_sha,
        },

        "firewall": {
            "existing_live_core_mutated":
                False,

            "existing_forecast_predictions_mutated":
                False,

            "database_write":
                False,

            "solver_mutation":
                False,

            "app_mutation":
                False,

            "service_restart":
                False,

            "production_influence":
                False,
        },
    }

    atomic_write_json(
        audit,
        AUDIT,
    )

    print(
        "EXISTING_LIVE_CORE_MUTATED=FALSE"
    )
    print(
        "EXISTING_FORECAST_PREDICTIONS_MUTATED=FALSE"
    )
    print(
        "DATABASE_WRITE=FALSE"
    )
    print(
        "SOLVER_MUTATION=FALSE"
    )
    print(
        "APP_MUTATION=FALSE"
    )
    print(
        "SERVICE_RESTART=FALSE"
    )
    print(
        "PRODUCTION_INFLUENCE=FALSE"
    )
    print(
        "PRODUCTION_FIREWALL=PASS"
    )

    # -----------------------------------------------------------------
    # 9. Final contract
    # -----------------------------------------------------------------

    section(
        "9. STAGE26G-L FINAL CONTRACT"
    )

    print(
        "STAGE26G_L_CONTRACT="
        "WFS_PREGAME_CONTEXT_PASS_EXPECTATION_SHADOW_V1"
    )
    print(
        "SOURCE_MODEL="
        "STAGE26G_J_VALIDATED_68_FEATURE_CONTEXT_MODEL"
    )
    print(
        "MODEL_CLASS=DeterministicRidge"
    )
    print(
        "FROZEN_CORE_FEATURE_COUNT=76"
    )
    print(
        "LEGACY_COACH_FEATURES_EXCLUDED=8"
    )
    print(
        "PREGAME_CONTEXT_FEATURE_COUNT=68"
    )
    print(
        "TARGET_AUTHORITY="
        "STAGE26G_A_FROZEN_DECISION_UNIVERSE"
    )
    print(
        f"FINAL_SELECTED_ALPHA={alpha:.12g}"
    )
    print(
        "CURRENT_GAME_XPASS_USED=FALSE"
    )
    print(
        "COACH_ADJUSTMENT_INCLUDED=FALSE"
    )
    print(
        "PREDICTION_CLIPPING_APPLIED=FALSE"
    )
    print(
        f"SHADOW_OUTPUT={OUTPUT}"
    )
    print(
        f"SHADOW_AUDIT={AUDIT}"
    )
    print(
        "EXISTING_FORECAST_MUTATION=FALSE"
    )
    print(
        "SOLVER_MUTATION=FALSE"
    )
    print(
        "PRODUCTION_INFLUENCE=FALSE"
    )
    print(
        "STAGE26G_L_PREGAME_CONTEXT_DEPLOYMENT_SHADOW=PASS"
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
