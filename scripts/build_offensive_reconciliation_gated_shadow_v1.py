#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

V1_PATH = (
    ROOT
    / "processed"
    / "offensive_team_reconciliation_shadow_v1.csv"
)

V2_PATH = (
    ROOT
    / "processed"
    / "offensive_team_reconciliation_shadow_v2.csv"
)

MATRIX_PATH = (
    ROOT
    / "data/parquet"
    / "nfl_current_offensive_model_matrix.parquet"
)

PRODUCTION_PATH = (
    ROOT
    / "data/parquet"
    / "nfl_production_projection.parquet"
)

GATE_PATH = (
    ROOT
    / "processed"
    / "offensive_reconciliation_gated_ensemble_rules_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_gated_shadow_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_gated_shadow_v1_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.zeros(len(frame)),
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(0.0)


def fd_points(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.Series:
    return (
        values(
            frame,
            f"{prefix}passing_yards",
        ) * 0.04
        + values(
            frame,
            f"{prefix}passing_tds",
        ) * 4.0
        - values(
            frame,
            f"{prefix}interceptions",
        )
        + values(
            frame,
            f"{prefix}rushing_yards",
        ) * 0.10
        + values(
            frame,
            f"{prefix}rushing_tds",
        ) * 6.0
        + values(
            frame,
            f"{prefix}receiving_yards",
        ) * 0.10
        + values(
            frame,
            f"{prefix}receiving_tds",
        ) * 6.0
        + values(
            frame,
            f"{prefix}receptions",
        ) * 0.50
    )


def main() -> None:
    for required in (
        V1_PATH,
        V2_PATH,
        MATRIX_PATH,
        PRODUCTION_PATH,
        GATE_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    v1 = pd.read_csv(V1_PATH)
    v2 = pd.read_csv(V2_PATH)
    matrix = pd.read_parquet(MATRIX_PATH)
    production = pd.read_parquet(
        PRODUCTION_PATH
    )
    gates = pd.read_csv(GATE_PATH)

    keys = [
        "game_id",
        "team",
        "player_id",
    ]

    for frame in (
        v1,
        v2,
        matrix,
        production,
    ):
        for column in keys:
            if column in frame.columns:
                frame[column] = (
                    frame[column]
                    .fillna("")
                    .astype(str)
                )

    reconciled_columns = [
        column
        for column in v1.columns
        if column.startswith(
            "reconciled_"
        )
    ]

    v1_stats = v1[
        keys + reconciled_columns
    ].rename(columns={
        column:
            "v1_" + column.removeprefix(
                "reconciled_"
            )
        for column in reconciled_columns
    })

    v2_stats = v2[
        keys + reconciled_columns
    ].rename(columns={
        column:
            "hybrid_" + column.removeprefix(
                "reconciled_"
            )
        for column in reconciled_columns
    })

    result = v2.drop(
        columns=reconciled_columns,
        errors="ignore",
    ).merge(
        v1_stats,
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        v2_stats,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    history_column = (
        "history_games_player"
        if "history_games_player"
        in matrix.columns
        else "player_history_games"
    )

    authority = matrix[
        keys
        + [
            "position",
            history_column,
            "active_flag",
            "injury_flag",
        ]
    ].drop_duplicates(
        keys,
        keep="last",
    ).rename(columns={
        "position": "matrix_position",
        history_column: "history_games",
        "active_flag":
            "matrix_active_flag",
        "injury_flag":
            "matrix_injury_flag",
    })

    result = result.merge(
        authority,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    result["gate_position"] = (
        result["matrix_position"]
        .fillna(result["position"])
        .fillna("")
        .astype(str)
        .str.upper()
    )

    result["history_games"] = pd.to_numeric(
        result["history_games"],
        errors="coerce",
    ).fillna(0.0)

    result["history_class"] = (
        "COLD_START_0"
    )

    result.loc[
        result["history_games"].between(
            1,
            4,
        ),
        "history_class",
    ] = "SPARSE_1_TO_4"

    result.loc[
        result["history_games"].ge(5),
        "history_class",
    ] = "ESTABLISHED_5_PLUS"

    projection_column = (
        "injury_adjusted_projection"
        if "injury_adjusted_projection"
        in production.columns
        else "ridge_projection"
    )

    production_authority = production[
        keys
        + [
            projection_column,
            "production_status",
        ]
    ].drop_duplicates(
        keys,
        keep="last",
    ).rename(columns={
        projection_column:
            "ridge_model_projection",
    })

    result = result.merge(
        production_authority,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    gates = gates.rename(columns={
        "position": "gate_position",
    })

    result = result.merge(
        gates[
            [
                "gate_position",
                "history_class",
                "selected_model",
                "selection_level",
                "selection_rows",
                "exact_segment_rows",
            ]
        ],
        on=[
            "gate_position",
            "history_class",
        ],
        how="left",
        validate="many_to_one",
    )

    db_path = ROOT / "data/nfl.db"
    uri = f"file:{db_path.resolve()}?mode=ro"

    with sqlite3.connect(
        uri,
        uri=True,
    ) as conn:
        game_status = pd.read_sql_query(
            """
            SELECT
                game_id,
                season,
                week,
                game_date,
                completed
            FROM games
            """,
            conn,
        )

    game_status["game_id"] = (
        game_status["game_id"]
        .fillna("")
        .astype(str)
    )

    result = result.merge(
        game_status,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    # Exclude games that have already been completed.
    result = result[
        ~pd.to_numeric(
            result["completed"],
            errors="coerce",
        ).fillna(0).eq(1)
    ].copy()

    result["original_fd_points"] = fd_points(
        result,
        "expected_",
    )

    result["v1_fd_points"] = fd_points(
        result,
        "v1_",
    )

    result["hybrid_fd_points"] = fd_points(
        result,
        "hybrid_",
    )

    result["model_authority_projection"] = (
        pd.to_numeric(
            result[
                "ridge_model_projection"
            ],
            errors="coerce",
        )
    )

    result[
        "model_authority_fallback_flag"
    ] = (
        result[
            "model_authority_projection"
        ].isna()
        .astype(int)
    )

    result[
        "model_authority_projection"
    ] = result[
        "model_authority_projection"
    ].fillna(
        result["original_fd_points"]
    )

    result["gate_selected_original"] = (
        result["selected_model"]
    )

    # Cold-start evidence was insufficient for exact
    # position/history gates. Keep the existing model as
    # the fail-closed authority for every cold start.
    cold = result[
        "history_class"
    ].eq("COLD_START_0")

    result.loc[
        cold,
        "selected_model",
    ] = "HISTORICAL_MODEL"

    result["gate_reason"] = np.where(
        cold,
        "COLD_START_SAFETY_OVERRIDE",
        result["selection_level"]
        .fillna("MISSING_GATE"),
    )

    result["gated_fd_projection"] = 0.0

    historical_mask = result[
        "selected_model"
    ].eq("HISTORICAL_MODEL")

    v1_mask = result[
        "selected_model"
    ].eq("RECONCILED_V1")

    hybrid_mask = result[
        "selected_model"
    ].eq(
        "RECONCILED_HYBRID"
    )

    result.loc[
        historical_mask,
        "gated_fd_projection",
    ] = result.loc[
        historical_mask,
        "model_authority_projection",
    ]

    result.loc[
        v1_mask,
        "gated_fd_projection",
    ] = result.loc[
        v1_mask,
        "v1_fd_points",
    ]

    result.loc[
        hybrid_mask,
        "gated_fd_projection",
    ] = result.loc[
        hybrid_mask,
        "hybrid_fd_points",
    ]

    active_role = result[
        "reconciliation_role"
    ].isin(
        [
            "PRIMARY_QB",
            "ACTIVE_ROTATION",
        ]
    )

    # Inactive and contingency players remain exactly zero.
    result.loc[
        ~active_role,
        "gated_fd_projection",
    ] = 0.0

    result["gated_projection_status"] = np.where(
        active_role,
        "ACTIVE_SHADOW",
        "ZERO_NON_ACTIVE_ROLE",
    )

    duplicate_keys = int(
        result.duplicated(keys).sum()
    )

    missing_gates = int(
        result["selected_model"].isna().sum()
    )

    nonfinite = int(
        (~np.isfinite(
            result[
                [
                    "original_fd_points",
                    "v1_fd_points",
                    "hybrid_fd_points",
                    "model_authority_projection",
                    "gated_fd_projection",
                ]
            ].to_numpy(dtype=float)
        )).sum()
    )

    negative = int(
        (
            result[
                "gated_fd_projection"
            ] < 0
        ).sum()
    )

    nonactive_nonzero = int(
        (
            result.loc[
                ~active_role,
                "gated_fd_projection",
            ].abs() > 1e-9
        ).sum()
    )

    hard_failures = {
        "duplicate_player_keys":
            duplicate_keys,
        "missing_gates":
            missing_gates,
        "nonfinite_projections":
            nonfinite,
        "negative_projections":
            negative,
        "nonactive_nonzero":
            nonactive_nonzero,
        "wrong_upcoming_game_count":
            int(
                result[
                    "game_id"
                ].nunique() != 15
            ),
    }

    status = (
        "PASS_HARD_CONTRACTS"
        if sum(
            hard_failures.values()
        ) == 0
        else "FAIL_HARD_CONTRACTS"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = result.sort_values(
        [
            "game_id",
            "team",
            "gate_position",
            "entity_name",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT,
        index=False,
    )

    # Recompute the mask after sorting/resetting the index.
    active_role = result[
        "reconciliation_role"
    ].isin(
        [
            "PRIMARY_QB",
            "ACTIVE_ROTATION",
        ]
    )

    active = result[
        active_role
    ].copy()

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_"
            "GATED_SHADOW_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "completed_games_excluded": True,
        "cold_start_policy":
            "EXISTING_MODEL_FAIL_CLOSED",
        "games": int(
            result["game_id"].nunique()
        ),
        "rows": int(len(result)),
        "active_rows": int(len(active)),
        "model_authority_fallback_rows":
            int(
                result[
                    "model_authority_fallback_flag"
                ].sum()
            ),
        "hard_failures":
            hard_failures,
        "selected_sources": {
            str(key): int(value)
            for key, value
            in result[
                "selected_model"
            ].value_counts().items()
        },
        "outputs": {
            "shadow": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
        },
    }

    AUDIT_OUTPUT.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "WFS OFFENSIVE RECONCILIATION "
        "CURRENT GATED SHADOW V1"
    )
    print("=" * 80)
    print(f"ROWS={len(result)}")
    print(
        "GAMES="
        f"{result['game_id'].nunique()}"
    )
    print(f"ACTIVE_ROWS={len(active)}")
    print(
        "MODEL_AUTHORITY_FALLBACK_ROWS="
        f"{int(result['model_authority_fallback_flag'].sum())}"
    )

    print("\n=== SELECTED SOURCES ===")
    print(
        result[
            "selected_model"
        ].value_counts().to_string()
    )

    print("\n=== ACTIVE SOURCE BY POSITION/HISTORY ===")
    print(
        active.groupby(
            [
                "gate_position",
                "history_class",
                "selected_model",
                "gate_reason",
            ],
            dropna=False,
        ).size().rename(
            "rows"
        ).reset_index().to_string(
            index=False
        )
    )

    print("\n=== LARGEST ACTIVE GATED PROJECTIONS ===")
    print(
        active[
            [
                "game_id",
                "team",
                "entity_name",
                "gate_position",
                "history_class",
                "selected_model",
                "gate_reason",
                "original_fd_points",
                "v1_fd_points",
                "hybrid_fd_points",
                "model_authority_projection",
                "gated_fd_projection",
            ]
        ]
        .sort_values(
            "gated_fd_projection",
            ascending=False,
        )
        .head(40)
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(f"\nSHADOW_OUTPUT={OUTPUT}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(
        "GATED_SHADOW_STATUS="
        f"{status}"
    )

    if status != "PASS_HARD_CONTRACTS":
        raise RuntimeError(
            "Current gated shadow contract failed"
        )


if __name__ == "__main__":
    main()
