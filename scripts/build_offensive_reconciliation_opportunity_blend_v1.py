#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_walkforward_opportunity_v1.csv"
)

OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_opportunity_blend_v1.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_opportunity_blend_summary_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "processed"
    / "offensive_reconciliation_opportunity_blend_v1_audit.json"
)

CLASSES = [
    "COLD_START_0",
    "SPARSE_1_TO_4",
    "ESTABLISHED_5_PLUS",
]

LAMBDA_GRID = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_to_team_budget(
    frame: pd.DataFrame,
    raw_column: str,
    budget_column: str,
) -> pd.Series:
    group_columns = [
        "game_id",
        "team",
    ]

    denominator = frame.groupby(
        group_columns
    )[raw_column].transform("sum")

    budget = frame.groupby(
        group_columns
    )[budget_column].transform("first")

    result = pd.Series(
        0.0,
        index=frame.index,
        dtype=float,
    )

    valid = denominator.gt(0)

    result.loc[valid] = (
        frame.loc[valid, raw_column]
        * budget.loc[valid]
        / denominator.loc[valid]
    )

    return result


def apply_blend(
    frame: pd.DataFrame,
    stat: str,
    lambdas: dict[str, float],
) -> pd.Series:
    work = frame.copy()

    lambda_values = (
        work["history_class"]
        .map(lambdas)
        .fillna(0.0)
        .astype(float)
    )

    work["_raw_blend"] = (
        (1.0 - lambda_values)
        * pd.to_numeric(
            work[f"v1_{stat}"],
            errors="coerce",
        ).fillna(0.0)
        + lambda_values
        * pd.to_numeric(
            work[f"v2_{stat}"],
            errors="coerce",
        ).fillna(0.0)
    )

    budget_column = {
        "carries": "adaptive_carries",
        "targets": "target_budget",
    }[stat]

    if stat == "targets":
        work["target_budget"] = (
            pd.to_numeric(
                work["adaptive_attempts"],
                errors="coerce",
            ).fillna(0.0)
            * 0.94
        )

    return normalize_to_team_budget(
        work,
        "_raw_blend",
        budget_column,
    )


def mae(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    return float(
        (
            pd.to_numeric(
                predicted,
                errors="coerce",
            ).fillna(0.0)
            - pd.to_numeric(
                actual,
                errors="coerce",
            ).fillna(0.0)
        ).abs().mean()
    )


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(INPUT)

    development_mask = (
        data["season"].eq(2025)
        & data["week"].between(8, 13)
    )

    holdout_mask = (
        (
            data["season"].eq(2025)
            & data["week"].between(14, 18)
        )
        | (
            data["season"].eq(2026)
            & data["week"].eq(1)
        )
    )

    development = data[
        development_mask
    ].copy()

    holdout = data[
        holdout_mask
    ].copy()

    if development.empty or holdout.empty:
        raise RuntimeError(
            "Missing development or holdout rows"
        )

    selected = {}
    tuning_rows = []

    combinations = list(
        product(
            LAMBDA_GRID,
            repeat=len(CLASSES),
        )
    )

    for stat in ("carries", "targets"):
        candidates = []

        for combination in combinations:
            lambdas = dict(
                zip(
                    CLASSES,
                    combination,
                )
            )

            prediction = apply_blend(
                development,
                stat,
                lambdas,
            )

            score = mae(
                development[
                    f"actual_{stat}"
                ],
                prediction,
            )

            candidates.append(
                (
                    score,
                    sum(combination),
                    combination,
                )
            )

            tuning_rows.append({
                "stat": stat,
                "development_mae": score,
                **{
                    f"lambda_{history_class}":
                        lambdas[
                            history_class
                        ]
                    for history_class in CLASSES
                },
            })

        # Prefer the simpler/lower-V2 blend when MAE ties.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        best = candidates[0]

        selected[stat] = dict(
            zip(
                CLASSES,
                best[2],
            )
        )

    for stat in ("carries", "targets"):
        data[f"hybrid_{stat}"] = apply_blend(
            data,
            stat,
            selected[stat],
        )

    summary_rows = []

    populations = {
        "DEVELOPMENT":
            development_mask,
        "HOLDOUT":
            holdout_mask,
    }

    for population, population_mask in populations.items():
        population_frame = data[
            population_mask
        ]

        for history_class, class_frame in [
            ("ALL", population_frame),
            *[
                (str(name), block)
                for name, block
                in population_frame.groupby(
                    "history_class"
                )
            ],
        ]:
            for stat in ("carries", "targets"):
                for model in (
                    "v1",
                    "v2",
                    "hybrid",
                ):
                    actual = pd.to_numeric(
                        class_frame[
                            f"actual_{stat}"
                        ],
                        errors="coerce",
                    ).fillna(0.0)

                    predicted = pd.to_numeric(
                        class_frame[
                            f"{model}_{stat}"
                        ],
                        errors="coerce",
                    ).fillna(0.0)

                    error = predicted - actual

                    summary_rows.append({
                        "population": population,
                        "history_class":
                            history_class,
                        "stat": stat,
                        "model": model.upper(),
                        "rows": int(
                            len(class_frame)
                        ),
                        "mae": float(
                            error.abs().mean()
                        ),
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.square(error)
                                )
                            )
                        ),
                        "bias": float(
                            error.mean()
                        ),
                    })

    summary = pd.DataFrame(summary_rows)

    conservation_rows = []

    for (
        game_id,
        team,
    ), block in data.groupby(
        ["game_id", "team"]
    ):
        carry_budget = float(
            pd.to_numeric(
                block["adaptive_carries"],
                errors="coerce",
            ).iloc[0]
        )

        target_budget = float(
            pd.to_numeric(
                block["adaptive_attempts"],
                errors="coerce",
            ).iloc[0]
            * 0.94
        )

        conservation_rows.append({
            "game_id": game_id,
            "team": team,
            "carry_gap": (
                carry_budget
                - block[
                    "hybrid_carries"
                ].sum()
            ),
            "target_gap": (
                target_budget
                - block[
                    "hybrid_targets"
                ].sum()
            ),
        })

    conservation = pd.DataFrame(
        conservation_rows
    )

    max_gap = float(
        conservation[
            [
                "carry_gap",
                "target_gap",
            ]
        ].abs().max().max()
    )

    holdout_all = summary[
        summary["population"].eq(
            "HOLDOUT"
        )
        & summary["history_class"].eq(
            "ALL"
        )
    ]

    holdout_pivot = holdout_all.pivot(
        index="stat",
        columns="model",
        values="mae",
    )

    for required_model in (
        "V1",
        "V2",
        "HYBRID",
    ):
        if required_model not in holdout_pivot.columns:
            raise RuntimeError(
                f"Missing model: {required_model}"
            )

    holdout_pivot[
        "HYBRID_VS_V1"
    ] = (
        holdout_pivot["V1"]
        - holdout_pivot["HYBRID"]
    )

    holdout_pivot[
        "HYBRID_VS_V2"
    ] = (
        holdout_pivot["V2"]
        - holdout_pivot["HYBRID"]
    )

    duplicate_keys = int(
        data.duplicated(
            ["game_id", "team", "player_id"]
        ).sum()
    )

    hard_failures = {
        "duplicate_player_keys":
            duplicate_keys,
        "conservation_failure":
            int(max_gap > 1e-8),
        "development_rows_missing":
            int(development.empty),
        "holdout_rows_missing":
            int(holdout.empty),
        "nonfinite_hybrid":
            int(
                (~np.isfinite(
                    data[
                        [
                            "hybrid_carries",
                            "hybrid_targets",
                        ]
                    ].to_numpy(dtype=float)
                )).sum()
            ),
        "negative_hybrid":
            int(
                (
                    data[
                        [
                            "hybrid_carries",
                            "hybrid_targets",
                        ]
                    ] < 0
                ).sum().sum()
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

    data.to_csv(
        OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    tuning_output = OUTPUT.with_name(
        "offensive_reconciliation_"
        "opportunity_blend_tuning_v1.csv"
    )

    pd.DataFrame(
        tuning_rows
    ).to_csv(
        tuning_output,
        index=False,
    )

    audit = {
        "version":
            "WFS_OFFENSIVE_RECONCILIATION_"
            "OPPORTUNITY_BLEND_V1",
        "status": status,
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "database_mode": "READ_ONLY",
        "production_modified": False,
        "development":
            "2025_WEEKS_8_TO_13",
        "holdout":
            "2025_WEEKS_14_TO_18_AND_2026_WEEK_1",
        "lambda_grid": LAMBDA_GRID,
        "selected_lambdas": selected,
        "max_conservation_gap":
            max_gap,
        "hard_failures":
            hard_failures,
        "holdout_mae":
            holdout_pivot.reset_index()
            .to_dict(orient="records"),
        "outputs": {
            "player": {
                "path": str(
                    OUTPUT.resolve()
                ),
                "sha256": sha256(OUTPUT),
            },
            "summary": {
                "path": str(
                    SUMMARY_OUTPUT.resolve()
                ),
                "sha256": sha256(
                    SUMMARY_OUTPUT
                ),
            },
            "tuning": {
                "path": str(
                    tuning_output.resolve()
                ),
                "sha256": sha256(
                    tuning_output
                ),
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
        "OPPORTUNITY BLEND V1"
    )
    print("=" * 80)

    print("\n=== LOCKED HISTORY-CLASS BLENDS ===")
    for stat, values in selected.items():
        print(f"{stat.upper()}")
        for history_class, value in values.items():
            print(
                f"  {history_class} "
                f"V2_WEIGHT={value:.2f}"
            )

    print("\n=== UNTOUCHED HOLDOUT MAE ===")
    print(
        holdout_pivot.reset_index().to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\n=== HOLDOUT BY HISTORY CLASS ===")
    print(
        summary[
            summary["population"].eq(
                "HOLDOUT"
            )
            & summary[
                "history_class"
            ].ne("ALL")
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\nMAX_CONSERVATION_GAP="
        f"{max_gap:.18e}"
    )
    print(f"PLAYER_OUTPUT={OUTPUT}")
    print(f"SUMMARY_OUTPUT={SUMMARY_OUTPUT}")
    print(f"TUNING_OUTPUT={tuning_output}")
    print(f"AUDIT_OUTPUT={AUDIT_OUTPUT}")
    print("SQLITE_MODIFIED=FALSE")
    print("PRODUCTION_MODIFIED=FALSE")
    print(
        "OPPORTUNITY_BLEND_STATUS="
        f"{status}"
    )

    if status != "PASS_HARD_CONTRACTS":
        raise RuntimeError(
            "Opportunity blend contract failed"
        )


if __name__ == "__main__":
    main()
