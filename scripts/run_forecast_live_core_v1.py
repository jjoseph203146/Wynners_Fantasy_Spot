from pathlib import Path
import ast
import hashlib
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path("/home/mwynn/nfl_data_engine")

EXP004_DIR = (
    ROOT
    / "backups"
    / "forecast_v1_exp004_frozen"
)

LIVE_CORE_DIR = (
    ROOT
    / "backups"
    / "forecast_live_core_v1_frozen"
)

EXP004_SCRIPT = (
    EXP004_DIR
    / "run_forecast_v1_experiment_004.py"
)

TRAINING = (
    ROOT
    / "processed"
    / "forecast_v1_team_game_training_impact_replacement_v1.csv"
)

EXP004_PRED = (
    EXP004_DIR
    / "forecast_v1_exp004_predictions_2025.csv"
)

EXP004_FEATURES = (
    EXP004_DIR
    / "forecast_v1_exp004_features.json"
)

LIVE_CORE = (
    LIVE_CORE_DIR
    / "forecast_live_core_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions.csv"
)

AUDIT = (
    ROOT
    / "processed"
    / "forecast_live_core_v1_predictions_audit.json"
)


# =====================================================================
# AUTHORITATIVE HASHES
# =====================================================================

EXPECTED_EXP004_SCRIPT = (
    "aa7cba483d4e5795303300f3ff861ddf"
    "246c4bd5033e73e0bdb06a6cd3d690d2"
)

EXPECTED_TRAINING = (
    "76aa9a35ea10d7a04518e709a410a5aa"
    "e7ca84dbadc5b7d0462f3e2d47ed75e1"
)

EXPECTED_EXP004_PRED = (
    "a26ccdfd59b13f47f842e4c62a55622e"
    "edcd390e5763864acfd9b54404c0e769"
)

EXPECTED_EXP004_FEATURES = (
    "417807fcf1fa5fa435a6336dea908ed45"
    "b3a87114d30a5f04ed7ceb950392aae"
)

EXPECTED_LIVE_CORE = (
    "6b769b46e29fe95e60e7a0a50f99b98d"
    "ba78d18e4bf9ff04dadd83509beec924"
)


# =====================================================================
# FROZEN FEATURE-CLASSIFICATION CONSTANTS
# =====================================================================

TARGET_COLUMNS = {
    "target_team_points",
    "target_opponent_points",
    "target_margin",
    "target_total_points",
    "target_win",
    "target_result",
}

IDENTITY_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "game_date",
    "weekday",
    "gametime",
    "team",
    "opponent_team",
    "coach",
    "opponent_coach",
    "team_qb_name",
    "opponent_qb_name",
    "roof",
    "surface",
}

MARKET_COLUMNS = {
    "market_home_spread_raw",
    "market_total",
    "team_moneyline",
    "opponent_moneyline",
}


# =====================================================================
# HELPERS
# =====================================================================

def section(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def atomic_write_csv(frame, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    tmp = Path(tmp_name)

    try:
        frame.to_csv(
            tmp,
            index=False,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(payload, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    tmp = Path(tmp_name)

    try:
        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


# =====================================================================
# EXACT FROZEN EXP004 IMPLEMENTATION EXTRACTION
# =====================================================================

def load_frozen_exp004_implementation():

    source = EXP004_SCRIPT.read_text(
        encoding="utf-8",
        errors="replace",
    )

    tree = ast.parse(
        source
    )

    wanted_classes = {
        "DeterministicRidge",
    }

    wanted_functions = {
        "build_home_game_frame",
        "build_feature_lists",
        "fit_final",
        "build_prediction_frame",
    }

    nodes = []

    for node in tree.body:

        if (
            isinstance(node, ast.ClassDef)
            and node.name in wanted_classes
        ):
            nodes.append(
                node
            )

        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted_functions
        ):
            nodes.append(
                node
            )

    found_classes = {
        n.name
        for n in nodes
        if isinstance(
            n,
            ast.ClassDef,
        )
    }

    found_functions = {
        n.name
        for n in nodes
        if isinstance(
            n,
            ast.FunctionDef,
        )
    }

    if found_classes != wanted_classes:
        raise RuntimeError(
            "Frozen DeterministicRidge extraction failed"
        )

    if found_functions != wanted_functions:
        missing = sorted(
            wanted_functions
            - found_functions
        )

        raise RuntimeError(
            f"Frozen function extraction failed: {missing}"
        )

    module = ast.Module(
        body=nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    namespace = {
        "np": np,
        "pd": pd,
        "math": math,

        "FINAL_TRAIN_SEASONS": [
            2023,
            2024,
        ],

        "FINAL_TEST_SEASON": 2025,

        "TARGET_COLUMNS":
            TARGET_COLUMNS,

        "IDENTITY_COLUMNS":
            IDENTITY_COLUMNS,

        "MARKET_COLUMNS":
            MARKET_COLUMNS,
    }

    exec(
        compile(
            module,
            filename=str(
                EXP004_SCRIPT
            ),
            mode="exec",
        ),
        namespace,
    )

    return namespace


# =====================================================================
# FROZEN RECONSTRUCTION PROOF
# =====================================================================

def prove_frozen_core_model(
    implementation,
    manifest,
):

    training = pd.read_csv(
        TRAINING
    )

    build_home_game_frame = implementation[
        "build_home_game_frame"
    ]

    build_feature_lists = implementation[
        "build_feature_lists"
    ]

    fit_final = implementation[
        "fit_final"
    ]

    build_prediction_frame = implementation[
        "build_prediction_frame"
    ]

    games = build_home_game_frame(
        training
    )

    (
        core,
        impact_features,
        replacement_features,
        core_impact,
        core_impact_replacement,
    ) = build_feature_lists(
        games
    )

    if core != manifest[
        "core_features"
    ]:
        raise RuntimeError(
            "Frozen CORE feature contract mismatch"
        )

    if len(core) != 76:
        raise RuntimeError(
            f"Frozen CORE count != 76: {len(core)}"
        )

    if len(
        impact_features
    ) != 60:
        raise RuntimeError(
            "Frozen Impact count != 60"
        )

    if len(
        replacement_features
    ) != 33:
        raise RuntimeError(
            "Frozen Replacement count != 33"
        )

    if len(
        core_impact
    ) != 136:
        raise RuntimeError(
            "Frozen CORE+Impact count != 136"
        )

    if len(
        core_impact_replacement
    ) != 169:
        raise RuntimeError(
            "Frozen CORE+Impact+Replacement count != 169"
        )

    alpha_margin = float(
        manifest[
            "alpha_selections"
        ][
            "CORE_MARGIN"
        ][
            "alpha"
        ]
    )

    alpha_total = float(
        manifest[
            "alpha_selections"
        ][
            "CORE_TOTAL"
        ][
            "alpha"
        ]
    )

    if (
        alpha_margin != 3000.0
        or alpha_total != 3000.0
    ):
        raise RuntimeError(
            "Frozen CORE alpha changed"
        )

    margin_model = fit_final(
        games,
        core,
        "margin_residual",
        alpha_margin,
    )

    total_model = fit_final(
        games,
        core,
        "total_residual",
        alpha_total,
    )

    test = games[
        games["season"]
        == 2025
    ].copy()

    if len(test) != 285:
        raise RuntimeError(
            f"2025 physical-game count != 285: {len(test)}"
        )

    X_test = test[
        core
    ].copy()

    margin_resid = (
        margin_model.predict(
            X_test
        )
    )

    total_resid = (
        total_model.predict(
            X_test
        )
    )

    reconstructed = (
        build_prediction_frame(
            test,
            "WFS_CORE_RESIDUAL",
            margin_resid,
            total_resid,
        )
    )

    frozen_all = pd.read_csv(
        EXP004_PRED
    )

    frozen = frozen_all[
        frozen_all["model"]
        == "WFS_CORE_RESIDUAL"
    ].copy()

    if len(frozen) != 285:
        raise RuntimeError(
            "Frozen 2025 CORE reference count != 285"
        )

    keys = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "model",
    ]

    numeric_cols = [
        "market_home_spread_raw",
        "market_total",
        "actual_home_points",
        "actual_away_points",
        "actual_home_margin",
        "actual_total_points",
        "pred_margin_residual",
        "pred_total_residual",
        "pred_home_margin",
        "pred_total_points",
        "pred_home_points",
        "pred_away_points",
    ]

    cmp = frozen[
        keys + numeric_cols
    ].merge(
        reconstructed[
            keys + numeric_cols
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=(
            "_frozen",
            "_recon",
        ),
    )

    if len(cmp) != 285:
        raise RuntimeError(
            "Frozen reconstruction merge count != 285"
        )

    if not (
        cmp["_merge"]
        == "both"
    ).all():
        raise RuntimeError(
            "Frozen reconstruction key mismatch"
        )

    tolerance = 1e-10
    global_max = 0.0

    for col in numeric_cols:

        a = pd.to_numeric(
            cmp[
                f"{col}_frozen"
            ],
            errors="coerce",
        )

        b = pd.to_numeric(
            cmp[
                f"{col}_recon"
            ],
            errors="coerce",
        )

        both_nan = (
            a.isna()
            & b.isna()
        )

        one_nan = (
            a.isna()
            ^ b.isna()
        )

        diff = (
            a - b
        ).abs()

        bad = (
            one_nan
            |
            (
                (~both_nan)
                &
                (~one_nan)
                &
                (diff > tolerance)
            )
        )

        if int(
            bad.sum()
        ) != 0:
            raise RuntimeError(
                f"Frozen reconstruction failed: {col}"
            )

        valid = diff[
            (~both_nan)
            &
            (~one_nan)
        ]

        if len(valid):
            global_max = max(
                global_max,
                float(
                    valid.max()
                ),
            )

    winner_cmp = frozen[
        [
            "game_id",
            "pred_winner",
        ]
    ].merge(
        reconstructed[
            [
                "game_id",
                "pred_winner",
            ]
        ],
        on="game_id",
        validate="one_to_one",
        suffixes=(
            "_frozen",
            "_recon",
        ),
    )

    winner_mismatch = int(
        (
            winner_cmp[
                "pred_winner_frozen"
            ].fillna("<NA>")
            !=
            winner_cmp[
                "pred_winner_recon"
            ].fillna("<NA>")
        ).sum()
    )

    if winner_mismatch != 0:
        raise RuntimeError(
            "Frozen winner reconstruction failed"
        )

    proof = {
        "games": 285,
        "numeric_tolerance": tolerance,
        "global_max_abs_difference":
            global_max,
        "winner_mismatches":
            winner_mismatch,
        "status": "PASS",
    }

    return (
        games,
        core,
        margin_model,
        total_model,
        proof,
    )


# =====================================================================
# LIVE 2026 FRAME
# =====================================================================

def build_live_home_frame(
    core,
):

    live = pd.read_csv(
        LIVE_CORE
    )

    if len(live) != 544:
        raise RuntimeError(
            f"Live team rows != 544: {len(live)}"
        )

    if live[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live game count != 272"
        )

    duplicate_keys = int(
        live.duplicated(
            [
                "game_id",
                "team",
            ]
        ).sum()
    )

    if duplicate_keys != 0:
        raise RuntimeError(
            "Duplicate live game/team keys"
        )

    missing_core = [
        c
        for c in core
        if c not in live.columns
    ]

    if missing_core:
        raise RuntimeError(
            f"Live CORE missing features: {missing_core}"
        )

    actual_core_order = [
        c
        for c in live.columns
        if c in set(core)
    ]

    if actual_core_order != core:
        raise RuntimeError(
            "Live CORE order differs from frozen contract"
        )

    is_home = pd.to_numeric(
        live[
            "is_home"
        ],
        errors="coerce",
    )

    home = live[
        is_home == 1
    ].copy()

    if len(home) != 272:
        raise RuntimeError(
            f"Live home rows != 272: {len(home)}"
        )

    if home[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live physical-game IDs not unique"
        )

    return (
        live,
        home,
    )


# =====================================================================
# LIVE INFERENCE
# =====================================================================

def predict_live_core(
    home,
    core,
    margin_model,
    total_model,
):

    required_metadata = [
        "game_id",
        "season",
        "week",
        "game_date",
        "team",
        "opponent_team",
        "market_home_spread_raw",
        "market_total",
        "forecast_market_ready",
    ]

    missing_metadata = [
        c
        for c in required_metadata
        if c not in home.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"Live metadata missing: {missing_metadata}"
        )

    out = pd.DataFrame()

    out[
        "game_id"
    ] = home[
        "game_id"
    ].astype(str)

    out[
        "season"
    ] = pd.to_numeric(
        home["season"],
        errors="coerce",
    )

    out[
        "week"
    ] = pd.to_numeric(
        home["week"],
        errors="coerce",
    )

    out[
        "game_date"
    ] = home[
        "game_date"
    ]

    out[
        "home_team"
    ] = home[
        "team"
    ].astype(str)

    out[
        "away_team"
    ] = home[
        "opponent_team"
    ].astype(str)

    out[
        "market_home_spread_raw"
    ] = pd.to_numeric(
        home[
            "market_home_spread_raw"
        ],
        errors="coerce",
    )

    out[
        "market_total"
    ] = pd.to_numeric(
        home[
            "market_total"
        ],
        errors="coerce",
    )

    ready = (
        out[
            "market_home_spread_raw"
        ].notna()
        &
        out[
            "market_total"
        ].notna()
    )

    source_ready = (
        pd.to_numeric(
            home[
                "forecast_market_ready"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )

    if not (
        ready.to_numpy()
        ==
        source_ready.to_numpy()
    ).all():
        raise RuntimeError(
            "forecast_market_ready disagrees with market fields"
        )

    out[
        "forecast_market_ready"
    ] = ready.astype(int)

    out[
        "pred_margin_residual"
    ] = np.nan

    out[
        "pred_total_residual"
    ] = np.nan

    out[
        "pred_home_margin"
    ] = np.nan

    out[
        "pred_total_points"
    ] = np.nan

    out[
        "pred_home_points"
    ] = np.nan

    out[
        "pred_away_points"
    ] = np.nan

    out[
        "pred_winner"
    ] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="object",
    )

    ready_index = out.index[
        ready
    ]

    if len(
        ready_index
    ):

        X = home.loc[
            ready_index,
            core,
        ].copy()

        margin_resid = (
            margin_model.predict(
                X
            )
        )

        total_resid = (
            total_model.predict(
                X
            )
        )

        market_margin = (
            out.loc[
                ready_index,
                "market_home_spread_raw",
            ]
            .to_numpy(
                dtype=float
            )
        )

        market_total = (
            out.loc[
                ready_index,
                "market_total",
            ]
            .to_numpy(
                dtype=float
            )
        )

        pred_margin = (
            market_margin
            + margin_resid
        )

        pred_total = (
            market_total
            + total_resid
        )

        pred_home_points = (
            pred_total
            + pred_margin
        ) / 2.0

        pred_away_points = (
            pred_total
            - pred_margin
        ) / 2.0

        out.loc[
            ready_index,
            "pred_margin_residual",
        ] = margin_resid

        out.loc[
            ready_index,
            "pred_total_residual",
        ] = total_resid

        out.loc[
            ready_index,
            "pred_home_margin",
        ] = pred_margin

        out.loc[
            ready_index,
            "pred_total_points",
        ] = pred_total

        out.loc[
            ready_index,
            "pred_home_points",
        ] = pred_home_points

        out.loc[
            ready_index,
            "pred_away_points",
        ] = pred_away_points

        winner = np.where(
            pred_margin > 0,
            out.loc[
                ready_index,
                "home_team",
            ],
            np.where(
                pred_margin < 0,
                out.loc[
                    ready_index,
                    "away_team",
                ],
                "TIE",
            ),
        )

        out.loc[
            ready_index,
            "pred_winner",
        ] = winner

    out[
        "model"
    ] = "WFS_CORE_RESIDUAL"

    out[
        "forecast_variant"
    ] = "LIVE_CORE_ONLY_V1"

    out[
        "impact_status"
    ] = "INJURY_SOURCE_PENDING"

    out[
        "replacement_status"
    ] = "INJURY_SOURCE_PENDING"

    out[
        "win_probability_status"
    ] = "NOT_CALIBRATED"

    out[
        "forecast_status"
    ] = np.where(
        ready,
        "READY_CORE_ONLY",
        "MARKET_PENDING",
    )

    out = out.sort_values(
        [
            "week",
            "game_date",
            "game_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return out


# =====================================================================
# MAIN
# =====================================================================

def main():

    section(
        "WFS FORECAST CENTER — LIVE CORE V1 INFERENCE"
    )

    # -----------------------------------------------------------------
    # Frozen lineage integrity
    # -----------------------------------------------------------------

    integrity = {
        "exp004_script":
            (
                sha256(
                    EXP004_SCRIPT
                )
                ==
                EXPECTED_EXP004_SCRIPT
            ),

        "training":
            (
                sha256(
                    TRAINING
                )
                ==
                EXPECTED_TRAINING
            ),

        "exp004_predictions":
            (
                sha256(
                    EXP004_PRED
                )
                ==
                EXPECTED_EXP004_PRED
            ),

        "exp004_features":
            (
                sha256(
                    EXP004_FEATURES
                )
                ==
                EXPECTED_EXP004_FEATURES
            ),

        "live_core":
            (
                sha256(
                    LIVE_CORE
                )
                ==
                EXPECTED_LIVE_CORE
            ),
    }

    for name, passed in integrity.items():

        if not passed:
            raise RuntimeError(
                f"Frozen integrity failure: {name}"
            )

        print(
            f"PASS | frozen integrity: {name}"
        )

    manifest = json.loads(
        EXP004_FEATURES.read_text(
            encoding="utf-8"
        )
    )

    # -----------------------------------------------------------------
    # Exact implementation
    # -----------------------------------------------------------------

    implementation = (
        load_frozen_exp004_implementation()
    )

    print(
        "PASS | exact frozen Exp004 implementation loaded"
    )

    # -----------------------------------------------------------------
    # Mandatory historical reproduction
    # -----------------------------------------------------------------

    (
        historical_games,
        core,
        margin_model,
        total_model,
        reproduction,
    ) = prove_frozen_core_model(
        implementation,
        manifest,
    )

    print(
        "PASS | frozen 2025 CORE predictions reproduced"
    )

    print(
        "2025 max abs difference | "
        f"{reproduction['global_max_abs_difference']:.15g}"
    )

    # -----------------------------------------------------------------
    # Live contract
    # -----------------------------------------------------------------

    (
        live_team_rows,
        live_home,
    ) = build_live_home_frame(
        core
    )

    print(
        f"Live team rows | {len(live_team_rows)}"
    )

    print(
        f"Live games     | {len(live_home)}"
    )

    print(
        "PASS | exact frozen 76-feature live contract"
    )

    # -----------------------------------------------------------------
    # Live predictions
    # -----------------------------------------------------------------

    predictions = predict_live_core(
        live_home,
        core,
        margin_model,
        total_model,
    )

    if len(
        predictions
    ) != 272:
        raise RuntimeError(
            "Live output row count != 272"
        )

    if predictions[
        "game_id"
    ].nunique() != 272:
        raise RuntimeError(
            "Live output game IDs not unique"
        )

    market_ready = int(
        (
            predictions[
                "forecast_status"
            ]
            == "READY_CORE_ONLY"
        ).sum()
    )

    market_pending = int(
        (
            predictions[
                "forecast_status"
            ]
            == "MARKET_PENDING"
        ).sum()
    )

    if (
        market_ready
        + market_pending
        != 272
    ):
        raise RuntimeError(
            "Live forecast status partition failed"
        )

    if market_ready != 100:
        raise RuntimeError(
            f"Expected 100 market-ready games; got {market_ready}"
        )

    ready = (
        predictions[
            "forecast_status"
        ]
        == "READY_CORE_ONLY"
    )

    prediction_cols = [
        "pred_margin_residual",
        "pred_total_residual",
        "pred_home_margin",
        "pred_total_points",
        "pred_home_points",
        "pred_away_points",
    ]

    if predictions.loc[
        ready,
        prediction_cols,
    ].isna().any().any():

        raise RuntimeError(
            "Ready games contain missing prediction values"
        )

    if predictions.loc[
        ~ready,
        prediction_cols,
    ].notna().any().any():

        raise RuntimeError(
            "Market-pending games unexpectedly contain predictions"
        )

    numeric_ready = (
        predictions.loc[
            ready,
            prediction_cols,
        ]
        .to_numpy(
            dtype=float
        )
    )

    if np.isinf(
        numeric_ready
    ).any():
        raise RuntimeError(
            "Infinite live prediction values"
        )

    # -----------------------------------------------------------------
    # Audit
    # -----------------------------------------------------------------

    audit = {
        "status":
            "PASS",

        "artifact":
            "forecast_live_core_v1_predictions",

        "architecture":
            "game_level_market_residual_core_only",

        "model":
            "WFS_CORE_RESIDUAL",

        "forecast_variant":
            "LIVE_CORE_ONLY_V1",

        "training_seasons": [
            2023,
            2024,
        ],

        "live_season":
            2026,

        "frozen_lineage": {
            "exp004_script_sha256":
                sha256(
                    EXP004_SCRIPT
                ),

            "training_sha256":
                sha256(
                    TRAINING
                ),

            "exp004_predictions_sha256":
                sha256(
                    EXP004_PRED
                ),

            "exp004_features_sha256":
                sha256(
                    EXP004_FEATURES
                ),

            "live_core_sha256":
                sha256(
                    LIVE_CORE
                ),
        },

        "model_contract": {
            "core_feature_count":
                len(
                    core
                ),

            "core_features":
                core,

            "margin_alpha":
                float(
                    manifest[
                        "alpha_selections"
                    ][
                        "CORE_MARGIN"
                    ][
                        "alpha"
                    ]
                ),

            "total_alpha":
                float(
                    manifest[
                        "alpha_selections"
                    ][
                        "CORE_TOTAL"
                    ][
                        "alpha"
                    ]
                ),
        },

        "reconstruction_proof":
            reproduction,

        "live_output": {
            "games":
                len(
                    predictions
                ),

            "market_ready_games":
                market_ready,

            "market_pending_games":
                market_pending,

            "unique_game_ids":
                int(
                    predictions[
                        "game_id"
                    ].nunique()
                ),

            "prediction_rows_with_inf":
                int(
                    np.isinf(
                        numeric_ready
                    ).any(
                        axis=1
                    ).sum()
                )
                if len(
                    numeric_ready
                )
                else 0,
        },

        "availability": {
            "impact":
                "INJURY_SOURCE_PENDING",

            "replacement":
                "INJURY_SOURCE_PENDING",

            "calibrated_win_probability":
                False,

            "temperature_missing_preserved":
                True,

            "wind_missing_preserved":
                True,
        },

        "safety": {
            "sqlite_write":
                False,

            "optimizer_modified":
                False,

            "ui_modified":
                False,

            "impact_enabled":
                False,

            "replacement_enabled":
                False,
        },
    }

    # -----------------------------------------------------------------
    # Atomic writes
    # -----------------------------------------------------------------

    atomic_write_csv(
        predictions,
        OUTPUT,
    )

    output_sha = sha256(
        OUTPUT
    )

    audit[
        "output"
    ] = {
        "path":
            str(
                OUTPUT
            ),

        "sha256":
            output_sha,
    }

    atomic_write_json(
        audit,
        AUDIT,
    )

    audit_sha = sha256(
        AUDIT
    )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    section(
        "LIVE CORE V1 INFERENCE RESULT"
    )

    print(
        f"Games              | {len(predictions)}"
    )

    print(
        f"Market-ready       | {market_ready}"
    )

    print(
        f"Market-pending     | {market_pending}"
    )

    print(
        f"CORE features      | {len(core)}"
    )

    print(
        "Historical proof   | PASS"
    )

    print(
        "2025 proof max diff| "
        f"{reproduction['global_max_abs_difference']:.15g}"
    )

    print()
    print(
        f"CSV       | {OUTPUT}"
    )

    print(
        f"CSV SHA   | {output_sha}"
    )

    print(
        f"Audit     | {AUDIT}"
    )

    print(
        f"Audit SHA | {audit_sha}"
    )

    print()
    print(
        "PASS | EXACT FROZEN EXP004 CORE MODEL RECONSTRUCTED"
    )

    print(
        "PASS | 2025 REFERENCE PREDICTIONS REPRODUCED"
    )

    print(
        "PASS | 2026 CORE-ONLY INFERENCE GENERATED"
    )

    print(
        "PASS | MARKET-PENDING GAMES LEFT UNPREDICTED"
    )

    print(
        "PASS | IMPACT/REPLACEMENT REMAIN DISABLED"
    )

    print(
        "PASS | NO CALIBRATED WIN PROBABILITY INVENTED"
    )

    print(
        "PASS | NO SQLITE WRITE"
    )

    print(
        "PASS | OPTIMIZER / UI UNTOUCHED"
    )


if __name__ == "__main__":
    main()
